#!/usr/bin/env python3
"""
Nightwatch · 评测脚本

================================================================================
这个脚本是整个项目里最该被认真对待的部分
================================================================================
理由（两组数字）：

    Gartner：到 2027 年底，超过 40% 的 Agentic AI 项目会被取消或降级
    MIT Media Lab：95% 的企业 AI 试点未产生任何可衡量的财务回报（300 个样本）

也就是说：**"做出来"的人很多，"能度量"的人极少。**
而调研里 5/7 份 JD 把评测列为要求、并把它定位成"真正的分水岭"，
同时它又是"培训班不教、也最难自己练"的东西 —— 这就是稀缺性所在。

================================================================================
本脚本跑两组实验（这两组是刻意的，不是随便跑跑）
================================================================================
    实验 A · 分层表现
        按问题类型分层报告指标。
        为什么必须分层？见 eval/metrics.py 的 StratifiedReport 说明 ——
        总分会被常见类型的高表现掩盖住稀有类型的问题。

    实验 B · 记忆效果对照（fresh vs warm）
        **这是本项目最有区分度的实验。**
        同一批 issue 跑两次：
          fresh 组：记忆库为空
          warm  组：记忆库已有历史
        对比步数 / token / F1。

      它直接回答"记忆到底有没有用"，而不是笼统说"我实现了记忆系统"。

================================================================================
怎么跑
================================================================================
    python scripts/run_eval.py                  # 用合成数据 + Mock 模型
    python scripts/run_eval.py --holdout        # 只用留出集（最终报数用）
    python scripts/run_eval.py --json out.json  # 输出机器可读结果

⚠️ 重要：默认跑的是**合成数据 + Mock 模型**，出来的数字只能验证"流程正确"，
   **不能写进简历。** 要写进简历必须换成真实 issue 队列 + 真实模型。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import (                                    # noqa: E402
    MODELS, MODEL_ROUTING, THINKING_POLICY, load_settings,
)
from nightwatch.eval.calibration import calibration_summary            # noqa: E402
from nightwatch.eval.dataset import (                              # noqa: E402
    DatasetBuilder, EvalDataset, build_synthetic_dataset,
)
from nightwatch.eval.metrics import (                              # noqa: E402
    Metrics, SampleResult, StratifiedReport,
)
from nightwatch.llm.client import build_llm                         # noqa: E402
from nightwatch.loop.react import TriageAgent                       # noqa: E402
from nightwatch.memory.store import (                               # noqa: E402
    HashEmbedding, MemoryStore, build_encoder,
)
from nightwatch.models import HumanVerdict                          # noqa: E402
from nightwatch.tools.registry import IssueRepository, ToolRegistry  # noqa: E402


# =============================================================================
# 全局编码器
# =============================================================================
# 为什么要显式固定它？
# 因为**向量编码器是评测结果的一部分** —— 换一个 embedding 模型，
# 相似度分布会完全不同，duplicate 阈值要重新标定，指标也不可比。
# 所以评测报告里必须写明用的是哪个编码器。
#
# 不固定的话会出现这种情况：本地跑（哈希）0.50，换台机器（有 API key → 语义）
# 变成 0.75，然后你不知道差异是来自代码改动还是来自编码器变了。
# **评测里每一个会影响结果的变量，都应该被显式声明。**
ENCODER = None

# 允许的标签集（从数据集自动提取）
#
# ★ 为什么要显式告诉模型这个？
# 真实项目（pandas）有 91 个标签，我们的评测口径只认「类型」层。
# 如果不告诉模型，它会自己发明标签（实测输出了 Timezones / IO SAS），
# 全被算成误报 —— 而那是"任务没说清"，不是"模型不行"。
#
# **标签空间是任务定义的一部分**：让模型猜用什么标签，等于让它猜你的评测口径。
ALLOWED_LABELS: list[str] = []

# System prompt 变体
#   "default" → 常规版
#   "strict"  → 强化版（显式流程 + 工具调用上限，用于"关掉思考链"的场景）
#
# 为什么要做成开关而不是直接改？
# 因为"强化提示词"是有代价的：它把执行路径写死了（第1步查历史、第2步查文档），
# 遇到需要灵活判断的 issue 可能反而不如自由发挥。
# **所以它必须能被 A/B 对比，而不是拍脑袋替换掉原来的。**
PROMPT_VARIANT: str = "default"

# 标签判定参考（提示词用，从 config.LABEL_GUIDE 生成）
#
# ★ 为什么需要它：实测发现模型分不清「主标签」和「细分标签」——
#   它用 `Error Reporting` 表达"这是个 bug"，而 gold 是 `Bug`。
#   根因是提示词没给标签的边界定义，它只能猜。
LABEL_GUIDE_TEXT: str = ""


def _new_store() -> MemoryStore:
    """新建记忆库，使用本次评测固定的编码器。"""
    return MemoryStore(":memory:", encoder=ENCODER)


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    RED = "\033[31m"; MAGENTA = "\033[35m"

    @classmethod
    def disable(cls):
        for n in dir(cls):
            if n.isupper():
                setattr(cls, n, "")


def hr(title: str = "", width: int = 78) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{C.BOLD}── {title} {'─' * pad}{C.RESET}")
    else:
        print(f"{C.DIM}{'─' * width}{C.RESET}")


def kv(k: str, v, color: str = "") -> None:
    print(f"  {C.DIM}{k:<24}{C.RESET}{color}{v}{C.RESET}")


# =============================================================================
# 单次处理 → SampleResult
# =============================================================================

def evaluate_one(issue, agent: TriageAgent, run_idx: int, stratum: str) -> SampleResult:
    """处理一个 issue 并转成可统计的结果。

    ★ 会把 ALLOWED_LABELS 传下去 —— 否则模型不知道项目的标签体系，
    会自己发明标签（实测：输出了 Timezones / IO SAS 这类领域标签，
    全被算成误报，Precision 被拉到 0.5）。

    **那不是模型的错，是任务没说清。** 详见 react._build_state_card。
    """
    result = agent.triage(issue, allowed_labels=ALLOWED_LABELS or None,
                          prompt_variant=PROMPT_VARIANT,
                          label_guide=LABEL_GUIDE_TEXT)
    trace = result.trace

    return SampleResult(
        issue_id=issue.issue_id,
        stratum=stratum,
        run=run_idx,
        predicted_labels=result.predicted_labels,
        predicted_duplicate_of=result.duplicate_of,
        confidence=result.confidence,
        outcome=result.outcome.value,
        gold_labels=issue.gold_labels,
        gold_duplicate_of=issue.gold_is_duplicate_of,
        n_steps=result.n_steps,
        elapsed_ms=result.elapsed_ms,
        total_tokens=trace.total_tokens if trace else 0,
        memory_recalled=len(trace.memory_recalled) if trace else 0,
        memory_used=len(trace.memory_used) if trace else 0,
    )


# =============================================================================
# 实验 A：分层表现
# =============================================================================

def _make_repo(issues, docs: dict | None = None) -> IssueRepository:
    """构造"项目历史库"。

    ============================================================
    评测设置的选择会决定指标能不能反映真实能力 —— 这里踩过一个坑
    ============================================================
    最初的实现是**每个 issue 配一个只含它自己的 repo**。结果：
        · search_similar_issues 永远只检索到它自己（然后被排除）→ 返回空
        · 于是 duplicate 检测的 TP/FP 永远是 0，Recall 永远 0

    但这个 0 **不是模型能力差，是评测设置错了** ——
    真实场景里，判断"这个 issue 是不是重复的"，靠的是拿到**全部历史 issue** 去比对。
    只给它自己，等于要求它在真空里判断重复。

    教训：**指标异常时，先检查评测设置，再怀疑模型能力。**
    这两者的区别是：前者改代码能修，后者要改模型/提示词才修。
    分不清这两者，就会在错误的方向上优化很久。
    """
    repo = IssueRepository(list(issues))
    repo.add_doc("CONTRIBUTING.md", (docs or {}).get("CONTRIBUTING.md", (
        "标签规范：bug/timeout 超时类；bug/permission 权限类；bug/config 配置类；"
        "bug/dependency 依赖类；bug/crash 崩溃类；bug/oom 内存类；"
        "enhancement 功能请求。若与已有 issue 高度相似，标记 duplicate 而非重新分类。"
    )))
    return repo


def experiment_stratified(issues, llm, settings, runs: int) -> list[SampleResult]:
    """按分层跑一遍。

    ============================================================
    实验设计：为什么是"跑 N 遍完整批次"而不是"每个样本跑 N 次"
    ============================================================
    两种做法看起来差不多，但有一个关键差别：**记忆的累积效应**。

        做法 A（每样本跑 N 次，每次新建 store）：
            → 每个样本都从空记忆开始，记忆永远不会累积
            → 「记忆质量」类指标恒为 0，因为压根没有记忆可召回
            → 这测的不是真实场景（真实 Agent 是持续运行的）

        做法 B（跑 N 遍完整批次，每遍一个新 store）：← 本项目采用
            → 每遍内部记忆正常累积，模拟真实持续运行
            → N 遍之间相互独立，可以用来消除随机性
            → 每遍的记忆累积顺序一致，所以各遍之间可比

    这个区分很典型：**实验的"单位"选错了，指标就失去意义。**
    """
    hr("实验 A · 分层表现")

    all_repo = _make_repo(issues)
    strata = [DatasetBuilder._stratum_of(i) for i in issues]
    n = len(issues)

    print(f"  {C.DIM}样本 {n} 条 × {runs} 遍完整批次（每遍独立记忆库）")
    print(f"  历史库规模 {n} 条（每个 issue 处理时排除自身）{C.RESET}\n")

    # runs 遍完整批次：(issue_id, run_idx) → SampleResult
    per_run: dict[tuple[str, int], SampleResult] = {}

    # ★ 实时进度输出
    #
    # 为什么必须加？这是踩过的坑：
    # 真实模型跑 20 条要 20+ 分钟，而原来的实现是"跑完整批次才打印结果"——
    # 于是整整 20 分钟**一行输出都没有**，你完全不知道它是：
    #   · 卡在第 1 条
    #   · 还是正常跑在第 15 条
    #   · 还是已经死了
    #
    # 在跑长任务时，**"看得见进度"比"输出好看"重要得多** ——
    # 没有进度就没有判断依据，只能靠猜（或者白等）。
    #
    # 这两行在真实模型下尤其重要：单次调用可能几十秒，
    # 而规则版 Mock 是毫秒级 —— **Mock 时代永远不会暴露这个问题。**
    n_total = len(issues) * runs
    n_done = 0

    for run_idx in range(runs):
        # 每遍一个新记忆库：保证各遍之间独立可比
        store = _new_store()
        for issue, stratum in zip(issues, strata):
            registry = ToolRegistry(all_repo, store=store,
                                    current_issue_id=issue.issue_id)
            agent = TriageAgent(registry, llm=llm, store=store, settings=settings)

            n_done += 1
            t0 = time.time()
            print(f"  [{n_done:>3}/{n_total}] {issue.issue_id:<8}"
                  f"{stratum[:14]:<16}", end="", flush=True)

            r = evaluate_one(issue, agent, run_idx, stratum)
            per_run[(issue.issue_id, run_idx)] = r

            _, _, f1 = Metrics.label_f1(r.predicted_labels, r.gold_labels)
            color = C.GREEN if f1 > 0.5 else (C.YELLOW if f1 > 0 else C.RED)
            print(f"{int((time.time() - t0) * 1000):>6}ms  "
                  f"F1={color}{f1:.2f}{C.RESET}  "
                  f"步数={r.n_steps}  预测={','.join(r.predicted_labels) or '—'}",
                  flush=True)

    # 按 issue 聚合：取中位数那一次作为代表（**不是最好的一次**）
    results: list[SampleResult] = []
    for idx, (issue, stratum) in enumerate(zip(issues, strata), 1):
        runs_of_issue = [per_run[(issue.issue_id, r)] for r in range(runs)]
        runs_of_issue.sort(key=lambda x: x.n_steps)
        median_run = runs_of_issue[len(runs_of_issue) // 2]
        results.append(median_run)

        _, _, f1 = Metrics.label_f1(median_run.predicted_labels, median_run.gold_labels)
        dup_mark = f" dup→{median_run.predicted_duplicate_of}" \
            if median_run.predicted_duplicate_of else ""
        mark = "✓" if f1 > 0.5 else ("·" if f1 > 0 else "✗")
        color = C.GREEN if f1 > 0.5 else (C.YELLOW if f1 > 0 else C.RED)
        print(f"  {color}{mark}{C.RESET} [{idx:>2}/{n}] "
              f"{issue.issue_id:<6} {stratum:<11} "
              f"预测={','.join(median_run.predicted_labels) or '—':<22} "
              f"gold={','.join(median_run.gold_labels) or '—':<18}"
              f"F1={f1:.2f}{dup_mark}")

    return results


# =============================================================================
# 实验 B：记忆效果对照
# =============================================================================

def experiment_memory_effect(issues, llm, settings) -> dict:
    """fresh（无记忆） vs warm（有记忆）对照实验。

    ============================================================
    实验设计的三个要点（这部分本身就是"实验设计能力"的体现）
    ============================================================
    ① **同一批 issue、同一模型、同一工具集** —— 唯一变量是记忆开关
    ② **warm 组的记忆来自"前一阶段"** —— 也就是说，先用前半批 issue
       让系统积累记忆，再用后半批测 warm 表现。这样测的是"真实的经验复用"，
       而不是"人为注入的记忆"。
    ③ **报告必须包含没有提升的情况** —— 如果 warm 组没更好，
       要如实说明并解释原因（召回精度不够？这批 issue 重复性不高？）

    最后一个要点最容易被忽略，也最能体现诚实：**不报失败结果的评测不是评测。**
    """
    hr("实验 B · 记忆效果对照（fresh vs warm）")

    mid = len(issues) // 2
    warmup, test = issues[:mid], issues[mid:]
    if not test:
        print(f"  {C.YELLOW}样本太少，跳过{C.RESET}")
        return {}

    print(f"  {C.DIM}预热集 {len(warmup)} 条 → 积累记忆；"
          f"测试集 {len(test)} 条 → 对比表现{C.RESET}\n")

    # 两组共用同一个"项目历史库" ——
    # 唯一变量是**记忆库是否已积累经验**，不能混入其他变量（比如历史库规模不同）
    shared_repo = _make_repo(issues)

    # ---------- 阶段 1：预热，积累记忆 ----------
    print(f"  {C.DIM}[阶段 1] 预热：让系统积累经验{C.RESET}")
    warm_store = _new_store()
    for issue in warmup:
        registry = ToolRegistry(shared_repo, store=warm_store,
                                current_issue_id=issue.issue_id)
        agent = TriageAgent(registry, llm=llm, store=warm_store, settings=settings)
        # 模拟维护者采纳 —— 人工反馈会提升记忆权重
        agent.triage(issue, human_verdict=HumanVerdict.ACCEPTED)

    n_mem = warm_store.count()
    print(f"         → 积累 {C.CYAN}{n_mem}{C.RESET} 条记忆\n")

    # ---------- 阶段 2：fresh 组（无记忆） ----------
    print(f"  {C.DIM}[阶段 2] fresh 组：空记忆库处理测试集{C.RESET}")
    fresh: list[SampleResult] = []
    for issue in test:
        store = _new_store()
        registry = ToolRegistry(shared_repo, store=store,
                                current_issue_id=issue.issue_id)
        agent = TriageAgent(registry, llm=llm, store=store, settings=settings)
        fresh.append(evaluate_one(issue, agent, 0, "test"))

    # ---------- 阶段 3：warm 组（带记忆） ----------
    print(f"  {C.DIM}[阶段 3] warm 组：带记忆处理同一批测试集{C.RESET}")
    warm: list[SampleResult] = []
    for issue in test:
        registry = ToolRegistry(shared_repo, store=warm_store,
                                current_issue_id=issue.issue_id)
        agent = TriageAgent(registry, llm=llm, store=warm_store, settings=settings)
        warm.append(evaluate_one(issue, agent, 1, "test"))

    result = Metrics.memorization_effect(fresh, warm)

    # ---------- 输出对照 ----------
    print()
    f, w = result["fresh_组"], result["warm_组"]
    print(f"  {'指标':<14}{'fresh(无记忆)':>16}{'warm(有记忆)':>16}{'变化':>14}")
    print(f"  {'─' * 62}")
    for key, label in (("中位步数", "步数"), ("中位token", "Token"), ("中位F1", "F1")):
        change = result[f"{'步数' if key == '中位步数' else ('token' if key == '中位token' else 'F1')}变化"]
        pct = change["百分比"]
        color = ""
        if pct != "N/A":
            # 步数和 token 降低是好事；F1 提高是好事
            val = float(pct.replace("%", "").replace("+", ""))
            good = (val < 0) if key != "中位F1" else (val > 0)
            color = C.GREEN if good else (C.RED if val != 0 else "")
        print(f"  {label:<14}{f[key]:>16}{w[key]:>16}{color}{pct:>14}{C.RESET}")

    print(f"\n  {C.DIM}{result['口径说明']}{C.RESET}")

    # ---------- 诚实报告：没有提升就说没有 ----------
    f1_change = result["F1变化"]["百分比"]
    if f1_change != "N/A" and float(f1_change.replace("%", "").replace("+", "")) <= 0:
        print(f"""
  {C.YELLOW}⚠ 说明：本次对照中 F1 没有提升。{C.RESET}
  {C.DIM}  可能的原因（必须如实说明，不能藏起来）：
      ① 合成数据的重复性不够高，记忆没发挥空间
      ② Mock 模型的"使用记忆"是规则分支，不代表真实模型的行为
      ③ 召回精度不足（相似度阈值/向量质量问题）
      → 这些都要在报告里写出来，而不是只报好看的那组。{C.RESET}""")

    return result


# =============================================================================
# 主流程
# =============================================================================

def main() -> int:
    # ★ global 声明必须在函数内**该变量被使用之前** ——
    #   否则 Python 会报 "name 'X' is used prior to global declaration"。
    #   这个坑很隐蔽：变量名本身是对的，只是声明的位置不对。
    global PROMPT_VARIANT, LABEL_GUIDE_TEXT

    ap = argparse.ArgumentParser(description="Nightwatch 评测")
    ap.add_argument("--dataset", help="评测集 JSON 路径（不传则用合成数据）")
    ap.add_argument("--holdout", action="store_true", help="只用留出集")
    ap.add_argument("--runs", type=int, default=3, help="每个样本跑几次（默认 3）")
    ap.add_argument("--json", help="把结果写成 JSON")
    ap.add_argument("--real-llm", action="store_true",
                    help="使用真实模型（需要 NW_LLM_API_KEY）")
    ap.add_argument("--limit", type=int, default=0,
                    help=("只评测前 N 条（0=全部）。真实模型单次约 17 秒，"
                          "全量 200 条要 1 小时 —— 建议先 --limit 10 验证流程"))
    ap.add_argument("--strict-prompt", action="store_true",
                    help=("用强化版 system prompt（显式流程 + 工具调用上限）。"
                          "配合关掉思考链使用 —— 用流程约束替代思考链的规划能力"))
    ap.add_argument("--tuned-prompt", action="store_true",
                    help=("用**调优版** system prompt（标签数量预期 + 主/细分标签"
                          "分级 + 给「信息不足」正名）。针对「多打标签」的失败模式"))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="覆盖最大步数（0=用 config 默认值）")
    ap.add_argument("--skip-memory-exp", action="store_true",
                    help=("跳过实验 B（记忆效果对照）。在完整 holdout 上它能省一半时间 ——"
                          "因为它要把测试集再跑两遍（fresh + warm）"))
    args = ap.parse_args()

    if not sys.stdout.isatty():
        C.disable()

    print(f"\n{C.BOLD}{C.CYAN}Nightwatch · 评测{C.RESET}")
    hr()

    # --- 数据集 ---
    if args.dataset:
        dataset: EvalDataset = EvalDataset.from_json(args.dataset)
        source_note = f"真实数据集: {args.dataset}"
    else:
        dataset = build_synthetic_dataset()
        source_note = "合成数据集（⚠️ 仅用于跑通流程）"

    kv("数据集来源", source_note,
       C.YELLOW if dataset.source == "synthetic" else C.GREEN)
    kv("版本", dataset.version)
    kv("train / holdout", f"{len(dataset.train)} / {len(dataset.holdout)}")

    issues = dataset.holdout if args.holdout else dataset.train
    if args.limit > 0:
        before = len(issues)
        issues = issues[:args.limit]
        print(f"  {C.DIM}--limit {args.limit}：从 {before} 条中取前 {len(issues)} 条{C.RESET}")

    # --- 从数据集自动提取"允许的标签集" ---
    # ★ 必须从**整个数据集**提取，不能用当前批次的子集。
    #
    # 踩过的坑：第一版从 `issues`（当前评测批次）提取，
    # 结果 `--limit 8` 时只有 3 个标签（Bug / Performance / Visualization）。
    # 模型遇到该打 Docs 的 issue 时无处可选，只能硬塞一个 Bug ——
    # **这不是模型判断错，是我们没给它选项。**
    #
    # 标签空间是"任务的可行域"，它属于任务定义，
    # 不应该随评测批次的大小而变。
    global ALLOWED_LABELS
    ALLOWED_LABELS = sorted({
        lbl for i in (dataset.train + dataset.holdout)
        for lbl in (i.gold_labels or [])
    })
    if ALLOWED_LABELS:
        kv("允许的标签集", f"{len(ALLOWED_LABELS)} 个（从整个数据集提取）")
        print(f"  {C.DIM}  {'、'.join(ALLOWED_LABELS[:14])}"
              f"{' …' if len(ALLOWED_LABELS) > 14 else ''}{C.RESET}")

    # --- 生成标签判定参考（供 tuned prompt 用）---
    if PROMPT_VARIANT == "tuned":
        from nightwatch.config import LABEL_GUIDE, build_label_guide
        LABEL_GUIDE_TEXT = build_label_guide(ALLOWED_LABELS)
        defined = sum(1 for l in ALLOWED_LABELS if l in LABEL_GUIDE)
        kv("标签判定参考", f"{defined}/{len(ALLOWED_LABELS)} 个标签有定义"
                           + ("（其余按字面理解）" if defined < len(ALLOWED_LABELS) else ""))
    if not issues:
        print(f"\n{C.RED}评测集为空，检查数据源{C.RESET}")
        return 1
    kv("本次使用", f"{'holdout' if args.holdout else 'train'} "
                   f"({len(issues)} 条)")

    # --- 模型 ---
    settings = load_settings()
    llm = build_llm(use_mock=not args.real_llm)

    # --- prompt 变体与步数上限 ---
    if args.tuned_prompt:
        PROMPT_VARIANT = "tuned"
    elif args.strict_prompt:
        PROMPT_VARIANT = "strict"
    else:
        PROMPT_VARIANT = "default"
    if args.max_steps > 0:
        settings.reliability["max_steps"] = args.max_steps
    variant_label = {"tuned": "调优版（标签数量约束 + 主/细分分级 + 信息不足正名）",
                     "strict": "强化版（显式流程 + 工具上限）",
                     "default": "常规版"}[PROMPT_VARIANT]
    kv("System prompt", variant_label,
       C.CYAN if PROMPT_VARIANT != "default" else "")
    kv("最大步数", settings.reliability["max_steps"])

    # ★ 日志里必须打印**具体模型名**，不能只打印类名。
    #
    # 这是用户提出的一个很实在的要求：「通过日志确认实际调用的模型已切换」。
    # 而原来的实现只打印 `OpenAICompatLLM` —— 换成任何模型都是这几个字，
    # **根本没法从日志确认用的是哪个模型。**
    #
    # 这在真实场景里是个大问题：跑完一小时评测，你无法从日志证明
    # 它用的是哪个模型 → 结果不可追溯 → 换个模型重跑时无法对比。
    #
    # 教训：**日志要记录"会影响结果的可变量"，而不是"代码的类名"。**
    if args.real_llm:
        routing = "，".join(
            f"{task.split('_')[-1]}={MODELS[tier]}"
            for task, tier in MODEL_ROUTING.items()
        )
        kv("模型", f"{type(llm).__name__}", C.GREEN)
        print(f"  {C.DIM}  档位映射：{routing}{C.RESET}")
        kv("模型来源", "环境变量 / .env 覆盖" if os.getenv("NW_MODEL_STRONG")
           else "config.py 默认值", C.GREEN)
    else:
        kv("模型", f"{type(llm).__name__}（规则版，非真实模型）", C.YELLOW)

    # --- 向量编码器 ---
    # 显式声明并打印：因为它直接影响 duplicate 判定和记忆召回的效果，
    # 是评测结果的一部分（换编码器 → 阈值要重标 → 指标不可比）
    global ENCODER
    ENCODER = build_encoder(use_api=args.real_llm)
    kv("向量编码器", type(ENCODER).__name__
       + ("" if args.real_llm else "（哈希伪向量，判别力有限）"),
       C.GREEN if args.real_llm else C.YELLOW)

    if dataset.source == "synthetic" or not args.real_llm:
        print(f"""
{C.YELLOW}⚠ 本次评测用的是合成数据 + Mock 模型。{C.RESET}
{C.DIM}  这些数字只能验证"流程正确"，**不能写进简历**。
  要得到可用于简历的数字，必须：
    1. 用 scripts/collect_issues.py 拉取真实项目的 issue
    2. 用 --real-llm 跑真实模型
    3. 用 --holdout 在留出集上报最终数字{C.RESET}""")

    # --- 实验 A ---
    stratified = experiment_stratified(issues, llm, settings, args.runs)

    # --- 实验 B ---
    if args.skip_memory_exp:
        hr("实验 B · 记忆效果对照")
        print(f"  {C.DIM}已跳过（--skip-memory-exp）{C.RESET}")
        mem_effect = {}
    else:
        mem_effect = experiment_memory_effect(issues, llm, settings)

    # --- 汇总报告 ---
    hr("汇总报告")

    f1s = [Metrics.label_f1(r.predicted_labels, r.gold_labels) for r in stratified]
    prec = statistics.median([p for p, _, _ in f1s]) if f1s else 0
    rec = statistics.median([r for _, r, _ in f1s]) if f1s else 0
    f1 = statistics.median([f for _, _, f in f1s]) if f1s else 0

    print(f"""
  {C.BOLD}整体指标（中位数）{C.RESET}""")
    kv("标签 Precision", f"{prec:.3f}")
    kv("标签 Recall", f"{rec:.3f}")
    kv("标签 F1", f"{f1:.3f}", C.GREEN if f1 > 0.6 else C.YELLOW)

    # --- 多口径对比（诊断"差异来自哪里"）---
    # ★ 单一口径会误导结论：严格 F1 低，可能是"多打标签"，也可能是"抓不住核心分类"。
    #   这两个原因对应的改进方向完全不同（调提示词 vs 换模型），所以必须并排看。
    mm = Metrics.multi_metric_report(stratified)
    if mm:
        hr("多口径对比（口径敏感性）")
        st, lo, an = mm["严格口径"], mm["宽松口径"], mm["标注倾向"]
        print(f"\n  {'口径':<18}{'值':>10}   含义")
        print(f"  {'-' * 62}")
        print(f"  {'严格 F1':<18}{st['F1']:>10.3f}   多标签严格匹配（漏报=误报）")
        print(f"  {'Top-1 准确率':<18}{lo['Top-1准确率']:>10.3f}   只看最核心分类对不对")
        print(f"  {'Hit@2':<18}{lo['Hit@2']:>10.3f}   正确标签在前 2 个里的比例")
        print(f"  {'Hit@3':<18}{lo['Hit@3']:>10.3f}   正确标签在前 3 个里的比例")
        print(f"\n  平均预测标签数 {an['平均预测标签数']}　"
              f"平均 gold 标签数 {an['平均gold标签数']}　"
              f"过度标注指数 {an['过度标注指数']:+.2f}")
        print(f"\n  {C.BOLD}诊断：{C.RESET}{mm['口径敏感性']}")

    hr("分层明细")
    strat_report = StratifiedReport.build(stratified)
    print(f"  {'分层':<14}{'样本':>6}{'中位F1':>10}{'最低F1':>10}{'中位步数':>10}")
    print(f"  {'─' * 52}")
    for name, s in strat_report.items():
        color = C.GREEN if s["中位F1"] > 0.6 else (C.YELLOW if s["中位F1"] > 0.3 else C.RED)
        print(f"  {name:<14}{s['样本量']:>6}{color}{s['中位F1']:>10.3f}{C.RESET}"
              f"{s['最低F1']:>10.3f}{s['中位步数']:>10}")

    hr("duplicate 检测（四象限）")
    dup_report = Metrics.duplicate_detection(stratified)
    print(f"""
  {C.DIM}为什么分开报而不是只给一个准确率？
  因为两类错误的代价不对称：
    漏报（真重复但没识别）→ 维护者重复处理，浪费时间
    误报（不重复却说是重复）→ **更糟**，新问题被误判为已解决而关闭{C.RESET}
""")
    kv("TP / FP", f"{dup_report['tp']} / {dup_report['fp']}")
    kv("FN / TN", f"{dup_report['fn']} / {dup_report['tn']}")
    kv("Precision", f"{dup_report['precision']:.3f}")
    kv("Recall", f"{dup_report['recall']:.3f}")
    kv("F1", f"{dup_report['f1']:.3f}")

    hr("可靠性指标")
    rel = Metrics.reliability_report(stratified)
    for k in ("自助解决率", "转人工率", "未收敛率", "误诊率"):
        v = rel[k]
        color = C.GREEN if (k == "自助解决率" and v > 0.6) else \
                (C.RED if (k == "误诊率" and v > 0.1) else "")
        kv(k, f"{v:.1%}", color)
    kv("中位步数 / P95步数", f"{rel['中位步数']} / {rel['P95步数']}")
    kv("P95 时延", f"{rel['P95时延ms']} ms")

    hr("记忆质量")
    mem_q = Metrics.memory_quality(stratified)
    kv("总召回次数", mem_q["总召回次数"])
    kv("实际使用次数", mem_q["实际使用次数"])
    kv("记忆命中率", f"{mem_q['记忆命中率']:.1%}",
       C.GREEN if mem_q["记忆命中率"] > 0.5 else C.YELLOW)
    kv("无效召回率", f"{mem_q['无效召回率']:.1%}",
       C.RED if mem_q["无效召回率"] > 0.5 else "")
    print(f"\n  {C.DIM}{mem_q['说明']}{C.RESET}")

    # ============================================================
    # 置信度校准（决定"能不能做转人工分流"）
    # ============================================================
    # ★ 这一节的存在理由：准确率只说"整体对不对"，
    #   不说"**哪条靠谱**"。而"转人工分流"靠的正是后者 ——
    #   置信度若和实际正确率对不上，任何阈值分流都会失效。
    cal = calibration_summary(stratified)
    hr("置信度校准与转人工阈值")

    if not cal.get("可用"):
        print(f"  {C.YELLOW}⚠ 不可用：{cal.get('原因', '')}{C.RESET}")
        if cal.get("修复方向"):
            print(f"  {C.DIM}修复方向：{cal['修复方向']}{C.RESET}")
        print(f"  {C.DIM}这不是 bug，是数据问题 —— 本节的指标没有可用输入。{C.RESET}")
    else:
        e = cal["ece"]
        rc = cal["risk_coverage"]
        esc = cal["escalation"]

        kv("ECE（期望校准误差）", f"{e['ece']:.4f}",
           C.GREEN if e["ece"] < 0.05 else (C.YELLOW if e["ece"] < 0.15 else C.RED))
        kv("MCE（最差桶偏离）", f"{e['mce']:.4f}")
        kv("分桶数", e["n_bins"])
        print(f"\n  {C.DIM}{e['verdict']}{C.RESET}")

        # 只打印**有样本**的桶 —— 空桶对判断没有帮助，只会拉长输出
        print(f"\n  {C.DIM}可靠性分桶（只看非空桶）{C.RESET}")
        print(f"  {'区间':<16}{'样本':>6}{'平均置信度':>12}{'实际正确率':>12}{'偏离':>9}")
        print(f"  {'-' * 58}")
        for b in e["bins"]:
            if not b["n"]:
                continue
            gap = b["gap"]
            col = C.RED if abs(gap) > 0.15 else (C.YELLOW if abs(gap) > 0.08 else "")
            print(f"  {'[' + format(b['range'][0], '.1f') + ',' + format(b['range'][1], '.1f') + ')':<16}"
                  f"{b['n']:>6}{b['avg_confidence']:>12.4f}{b['accuracy']:>12.4f}"
                  f"{col}{gap:>+9.4f}{C.RESET}")
        print(f"  {C.DIM}偏离为正 = 欠自信；为负 = 过度自信{C.RESET}")

        kv("AURC", f"{rc['aurc']:.4f}")
        kv("整体错误率", f"{rc['overall_error']:.4f}",
           C.DIM)
        print(f"\n  {C.DIM}{rc['verdict']}{C.RESET}")
        print(f"\n  {C.DIM}关键覆盖率点{C.RESET}")
        for cov, point in rc["key_points"].items():
            print(f"    保留 {cov:<5}（{point['n_kept']:>3} 条）"
                  f"→ 错误率 {point['risk']:.3f}　准确率 {point['accuracy']:.3f}")

        print()
        if esc["found"]:
            kv("建议转人工阈值", f"{esc['threshold']:.4f}", C.CYAN)
            kv("覆盖率（直出比例）", f"{esc['coverage']:.1%}")
            kv("观测准确率", f"{esc['expected_accuracy']:.1%}",
               C.GREEN if esc["expected_accuracy"] >= 0.9 else C.YELLOW)
            kv("转人工条数", esc["n_escalated"])
            print(f"\n  {C.DIM}{esc['verdict']}{C.RESET}")
            # ★ 乐观偏差必须写在同一屏内 —— 否则这个数字会被当成线上指标引用
            print(f"  {C.YELLOW}⚠ {esc['note']}{C.RESET}")
        else:
            print(f"  {C.YELLOW}{esc['verdict']}{C.RESET}")
            print(f"  {C.DIM}{esc['note']}{C.RESET}")

        if e.get("note"):
            print(f"\n  {C.DIM}{e['note']}{C.RESET}")

    # --- 输出 JSON ---
    payload = {
        "meta": {
            "dataset_source": dataset.source,
            "dataset_version": dataset.version,
            "n_issues": len(issues),
            "use_subset": "holdout" if args.holdout else "train",
            "model": type(llm).__name__,
            "model_names": dict(MODELS),
            "prompt_variant": PROMPT_VARIANT,
            "thinking_policy": dict(THINKING_POLICY),
            "runs_per_sample": args.runs,
            "warning": ("合成数据 + Mock 模型，数字不可用于简历"
                        if dataset.source == "synthetic" or not args.real_llm else ""),
        },
        "overall": {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4)},
        "per_stratum": strat_report,
        "duplicate_detection": dup_report,
        "reliability": rel,
        # ★ 新增字段（不改动任何已有字段）——
        #   置信度校准是"转人工分流"能否成立的前提，必须跟着报告一起出去，
        #   否则每次都要重新跑一遍才能看。
        "calibration": cal,
        "memory_quality": mem_q,
        "memory_effect": mem_effect,
        # ★ 逐条数据必须存 —— 这是个真实缺陷的修复。
        #
        # 第一次跑完整 holdout 时，JSON 里只有汇总指标，没有逐条结果。
        # 结果我发现"中位数 1.000 但均值 0.800"时，**无法从 JSON 重算**，
        # 只能回头用正则从日志文本里解析 —— 既脆弱又难看。
        #
        # 教训：**评测结果必须存逐条原始数据，不能只存汇总。**
        # 因为汇总口径是当时的判断，而原始数据支持未来任何角度的重算 ——
        # 比如后来加 Top-1 指标时，如果当时存了逐条数据，直接重算就行，
        # 不用重跑 55 分钟。
        "rows": [
            {
                "issue_id": r.issue_id,
                "stratum": r.stratum,
                "predicted_labels": r.predicted_labels,
                "gold_labels": r.gold_labels,
                "predicted_duplicate_of": r.predicted_duplicate_of,
                "gold_duplicate_of": r.gold_duplicate_of,
                "outcome": r.outcome,
                "n_steps": r.n_steps,
                "elapsed_ms": r.elapsed_ms,
                "total_tokens": r.total_tokens,
                # 逐条多口径（方便事后按任意切片重算）
                "strict_f1": round(Metrics.label_f1(
                    r.predicted_labels, r.gold_labels)[2], 4),
                "top1": Metrics.top1_accuracy(r.predicted_labels, r.gold_labels),
                "hit2": round(Metrics.hit_at_k(
                    r.predicted_labels, r.gold_labels, 2), 4),
            }
            for r in stratified
        ],
    }

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  {C.GREEN}结果已写入 {args.json}{C.RESET}")

    hr("下一步")
    print(f"""
  1. 拉真实数据：  python scripts/collect_issues.py --repo owner/name
  2. 真实模型评测：NW_LLM_API_KEY=xxx python scripts/run_eval.py --real-llm --holdout
  3. 理解指标口径：docs/05-评测设计.md

  {C.YELLOW}记住三条纪律：不报单次最好成绩 / 不只报赢的那组 / 都注明样本量。{C.RESET}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
