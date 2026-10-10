#!/usr/bin/env python3
"""
Nightwatch · 前瞻验证评测（轨道 A）

    # 快速验证流程（Mock）
    python scripts/eval_prospective.py --mock

    # 真实模型（正式数据）
    python scripts/eval_prospective.py --real-llm

    # 只看已有结果
    python scripts/eval_prospective.py --report-only

==============================================================================
这个脚本回答的问题，和 `run_eval.py` 完全不同
==============================================================================
`run_eval.py` 问的是："在已关闭的历史 issue 上，gold 标签能对上多少？"
**本脚本问的是**："用创建时的信息预测，能对上维护者后来真实打上的标签多少？"

两者都是算"一致率"，但**证据等级不同**：

    离线评测     gold 标签是"已经发生的事实"，用它评测是**解释过去**
    前瞻验证     预测发生在标签之前，这是**预测未来**

**面试里那句"你的指标是历史数据上算的，怎么知道线上有用"** ——
这个脚本就是答案。

==============================================================================
★ 三条保证严谨性的设计
==============================================================================
① **绝对不看标签**：Agent 拿到的只有标题 + 正文，
   连"当前有哪些标签"都不给它（否则就是作弊）。

② **对照的是"首个类型标签"**，不是"当前所有标签" ——
   见 `collect_prospective.py` 的说明（首次标签才是基于创建时信息打的）。

③ **预测和真实标签都逐条打印** —— 让人能人工核对，而不是只信一个汇总数字。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import load_settings                        # noqa: E402
from nightwatch.eval.error_taxonomy import (                       # noqa: E402
    classify_error_detailed, taxonomy_report,
)
from nightwatch.llm.client import build_llm                        # noqa: E402
from nightwatch.loop.react import TriageAgent                      # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore     # noqa: E402
from nightwatch.models import (                                    # noqa: E402
    Issue, IssueState, MemoryRecord, MemoryTier, Outcome,
)
from nightwatch.tools.registry import IssueRepository, ToolRegistry    # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


# =============================================================================
# ★ 关键设计：Agent 能看到什么、不能看到什么
# =============================================================================

def to_issue(rec: dict) -> Issue:
    """把采集记录转成 Issue 对象。

    ============================================================
    ★★ 这里做的"信息隔离"是整个实验有效性的关键 ★★
    ============================================================
    传进去的**只有标题和正文** —— 这两个字段在 issue 创建时就有了。

    刻意**不传**的东西：
      · `labels`        ← 那是要预测的目标（给了就是作弊）
      · `gold_labels`   ← 同上
      · `state`         ← "open/closed" 是后来的结果，会泄露信息
      · `body_edited`   ← 与本实验无关

    **如果不做这个隔离，评测会虚高** —— 而且虚高得毫无意义，
    因为在真实场景里，Agent 面对的确实是"只有标题和正文"的 issue。

    （这个原则和 `docs/04` 里"评测要构造能测出差异的用例"是同一类：
      评测的有效性首先取决于"有没有把不该给的给了"。）
    """
    return Issue(
        issue_id=rec["issue_id"],
        title=rec["title"],
        body=rec.get("body", ""),
        author=rec.get("author", ""),
        repo=rec.get("repo", ""),
        labels=[],                       # ← 刻意清空
        stack_trace="",
        state=IssueState.OPEN,           # ← 刻意置为 open（不泄露最终状态）
        gold_labels=[],                  # ← 刻意清空
    )


def build_seed_memory(records: list[dict], exclude_ids: set[str]) -> MemoryStore:
    """构造记忆库。

    ★ 这里有个**信息泄露**的风险，必须处理：

    如果我把"这 61 条 issue 的标签"写进记忆库，那 Agent 召回时
    可能直接看到答案 —— 那就等于变相作弊。

    所以：**记忆库只用"训练集"的 issue 构造**，
    而评测样本（`exclude_ids`）**绝不进记忆库**。

    实际实现上更简单：用**采集之前就有的历史数据**（`pandas_eval.json`
    的 holdout 之外的样本）构造记忆库，并且强制排除当前评测的 issue。

    （这个坑在 `docs/04` 里也有对应案例：
      "评测单位选错导致指标恒为 0" —— 评测里最容易错的就是这种"边界"。）
    """
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=256))
    # 用固定的种子记忆（不来自评测样本）
    seeds = [
        ("s1", "CI 构建超时失败", ["Build", "timeout"], "Build"),
        ("s2", "文档示例无法运行", ["Docs"], "Docs"),
        ("s3", "内存占用过高", ["Performance", "memory"], "Performance"),
        ("s4", "merge 结果行数异常", ["merge"], "Bug"),
        ("s5", "新功能请求：支持某参数", ["enhancement"], "Enhancement"),
        ("s6", "read_csv 读大文件 OOM", ["memory", "oom"], "Performance"),
        ("s7", "测试用例不稳定偶发失败", ["testing"], "Bug"),
        ("s8", "API 文档描述有误", ["Docs"], "Docs"),
    ]
    for mid, symptom, tags, res in seeds:
        store.insert(MemoryRecord(
            memory_id=mid, tier=MemoryTier.EPISODIC,
            issue_id=f"#{mid}", symptom=symptom, symptom_tags=tags,
            resolution=res, outcome=Outcome.RESOLVED))
    return store


def predict(rec: dict, store, llm, settings) -> dict:
    """对单条 issue 跑预测。"""
    issue = to_issue(rec)
    # 工具集里给一批"陪跑"的 issue（不含评测样本的标签）
    repo = IssueRepository([])
    registry = ToolRegistry(repo, store=store,
                            current_issue_id=issue.issue_id)
    agent = TriageAgent(registry, llm=llm, store=store, settings=settings)
    try:
        # ★★ 必须显式指定 prompt_variant="tuned"
        #
        # 这里踩了一个"配置不一致"的坑，值得完整记录：
        #
        #   `triage()` 的 prompt_variant 默认是 "default"
        #   `run_eval.py`（离线评测）显式传 "tuned"
        #   而本脚本最初**没传** → 用了 "default"
        #
        # **后果：两套评测用的根本不是同一个提示词。**
        #   而这在数字上看不出来（两边都是"合理的结果"）。
        #
        # 我是怎么发现的：改了 `SYSTEM_PROMPT_TUNED` 里的 Build 规则后
        # 重跑，发现命中率毫无变化 —— 顺着调用链查才发现用错了版本。
        #
        # **教训**（和 docs/05 的 D2 是同一条）：
        #   对照/评测的两条路径要**逐字对齐配置**。
        #   "我以为是一致的"是最危险的状态 —— 因为数字看起来正常。
        #
        # 现在的约定：**tuned 是正式版**（它是针对标签问题调优过的），
        # 所有正式评测都显式传它。
        r = agent.triage(issue, prompt_variant="tuned")
        tr = r.trace
        return {
            "predicted": list(tr.predicted_labels or []),
            "steps": tr.n_steps,
            "tokens": tr.total_tokens,
            "error": "",
        }
    except Exception as e:                                  # noqa: BLE001
        return {"predicted": [], "steps": 0, "tokens": 0,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


# =============================================================================
# ★★ 标签归一化：细分格式 → 粗标签（这是本脚本最关键的一段）★★
# =============================================================================
# 踩过的坑：第一版直接拿 `predicted_labels` 和 gold 比，**结果全错**。
#
# 原因：Agent 输出的是**细分格式** `bug/crash`，而 pandas 的 gold 是**粗标签** `Bug`。
#
#     Agent 输出：['bug/crash']
#     pandas 标签：['Bug']
#     → 字符串不相等 → 判为错误
#
# 而主项目自己的评测能对上，是因为**两边都是细分格式** ——
# `Metrics.normalize_label` 会把两边都切成"最后一段有区分度的词"：
#
#     'bug/crash'  → 'crash'
#     'bug/segfault' → 'segfault'
#
# **但前瞻验证的场景不同**：预测是细分，gold 是粗标签。
# 所以这里要做的是**反过来**：把细分**向上归到粗标签**。
#
# ============================================================
# 一个值得记的教训
# ============================================================
# **"标签归一化"没有通用解 —— 它取决于两边的粒度关系。**
#
#   两个细分对两个细分   → 取"有区分度的那一段"（主项目评测的做法）
#   一个细分对一个粗标签 → **归到粗的那一端**（本脚本的做法）
#
# 如果照搬主项目的 `normalize_label`，`bug/crash` 会变成 `crash`，
# 而 gold 是 `bug` —— **依然对不上，而且错得很隐蔽**
# （因为字母都对，只是不是同一个词）。
#
# **对齐标签之前，先确认两边的粒度。**

# 细分第一段 → 粗标签
_COARSE_MAP = {
    "bug": "Bug",
    "doc": "Docs",
    "docs": "Docs",
    "documentation": "Docs",
    "enh": "Enhancement",
    "enhancement": "Enhancement",
    "feature": "Enhancement",
    "perf": "Performance",
    "performance": "Performance",
    "build": "Build",
    # ★ `ci` / `bld` 是 pandas 里 Build 的**下级表达**：
    #   实测发现模型常输出 `CI: xxx` / `BLD: xxx` 作为标签名
    #   （因为它照抄了标题前缀），而 pandas 把它们都归入 `Build`。
    #   第一版漏了这两条 → `['CI', 'config']` 归一化成空 → 被算作"未给出预测"，
    #   而真相是"给出了但归一化失败"。**这是评测侧的 bug，不是模型的错。**
    "ci": "Build",
    "bld": "Build",
    "regression": "Regression",
    # ★ `depr` 是 pandas 标题里的标准前缀（`DEPR:` = deprecation 提案）
    #
    # 和 `ci` / `bld` 是**同一类遗漏** —— 第一版只写了完整的 `deprecate`，
    # 没写缩写 `depr`。
    #
    # 后果：如果模型输出 `['DEPR', 'Indexing']`，
    # `to_coarse('DEPR')` 返回 None → 整个列表归一化成空 →
    # **被算作"未给出预测"，而真相是"给了但归一化失败"**。
    #
    # 「评测侧的缩写映射要覆盖**模型可能输出的所有写法**」——
    # 模型会照抄标题里的前缀，所以标题里有的缩写（CI/BLD/DEPR/TST/REF）
    # 都要在映射表里。**这是评测的职责，不是模型的。**
    "depr": "Deprecate",
    "deprecate": "Deprecate",
    "deprecation": "Deprecate",
    "tst": "Testing",
    "ref": "Refactor",
    # ★ `incorrect` —— MockLLM 新增的"计算结果不正确"类别
    #
    # **这是同一个坑的第三次**（前两次：`ci`/`bld`、`depr`/`tst`/`ref`）。
    # 上游 `MockLLM._tags` 加了新类别，下游 `_COARSE_MAP` 忘了补映射，
    # 于是新标签归一化成 None → **被算作"未给出预测"**，
    # 而真相是"给了但归一化失败"。**评测侧的 bug，不是模型的错。**
    #
    # ★ 这一条值得记的是：**它是"上下游有隐式契约"的典型表现。**
    #   两边各维护一份标签表，靠"记得同步"来保证一致 ——
    #   而"记得同步"不是一种机制。真正该做的是**让归一化表从标签定义生成**，
    #   而不是手抄。本次因为不改动主项目结构，只补映射 + 留下这条注释。
    "incorrect": "Bug",
}

# 这些**不是类型标签**，是状态/流程标签 —— 出现在预测里说明"Agent 没给出类型判断"
_NON_TYPE = {
    "needs-triage", "needs_triage", "triage", "testing", "test",
    "duplicate", "needs-info", "invalid", "wontfix", "stale",
    "question", "usage", "unknown", "general",
}


def to_coarse(label: str) -> str | None:
    """把 Agent 输出的标签归到粗标签。归不到就返回 None。

    返回 None 的语义是"这条预测不是有效的类型判断" ——
    它会被算作"未给出预测"，而不是"预测错误"。

    **这个区分很重要**：
      · "预测成别的类型" = 判断错误
      · "没给出类型判断" = 没回答
    两者对系统的评价完全不同（后者是可以通过改提示词解决的）。
    """
    s = str(label).strip().lower()
    if not s:
        return None
    # 统一切分（和主项目的 normalize_label 用同一套分隔符）
    for sep in ("/", ":", "-", "_", " "):
        s = s.replace(sep, "|")
    parts = [p for p in s.split("|") if p]
    if not parts:
        return None

    # ★ 从**第一段**开始找（而不是像主项目那样从最后一段找）
    #   因为 `bug/crash` 的粗标签是 `bug`，不是 `crash`。
    for p in parts:
        if p in _COARSE_MAP:
            return _COARSE_MAP[p]
    # 全都不认识 → 看是不是纯状态标签
    joined = "".join(parts)
    if any(n.replace("-", "").replace("_", "") == joined for n in _NON_TYPE):
        return None
    return None


def normalize_pred(pred: list[str]) -> list[str]:
    """把预测标签列表归一化（去重、排序）。"""
    out = []
    for p in pred or []:
        c = to_coarse(p)
        if c and c not in out:
            out.append(c)
    return out


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="前瞻验证评测")
    ap.add_argument("--data", default=str(ROOT / "data" / "prospective_issues.json"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    ap.add_argument("--mock", action="store_true", help="用 Mock（快速验证流程）")
    ap.add_argument("--real-llm", action="store_true", help="用真实模型")
    ap.add_argument("--save", default=None,
                    help="结果输出路径。不指定时按模式自动选择 —— "
                         "真实模型 → data/prospective_eval.json；"
                         "Mock → data/_prospective_eval_mock.json")
    ap.add_argument("--report-only", action="store_true",
                    help="不跑预测，直接用已有结果出报告")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"{C['red']}✗ 找不到 {data_path}{C['reset']}")
        print(f"  先跑：python scripts/collect_prospective.py --limit 90")
        return 1

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    records = payload["issues"]
    meta = payload.get("meta", {})
    if args.limit:
        records = records[:args.limit]

    real = args.real_llm and not args.mock

    # ★★ 默认输出路径**按模式分开** —— 这是一个真实踩过的坑 ★★
    #
    # 原来 `--save` 的默认值固定指向 `data/prospective_eval.json`，
    # 于是 `--mock`（用于快速验证流程）会**直接覆盖正式评测数据**。
    #
    # 后果很隐蔽：跑完 `--mock` 一切正常、报告照出、退出码 0，
    # 但正式的那份 v1 结果已经被 Mock 结果替换掉了。
    # 直到下次跑 `compare_prospective.py` 才会发现
    # "v1 命中率怎么从 93.4% 变成了 8.2%" —— 而此时原始数据已经没了。
    #
    # 这类问题正是本项目一贯关注的那一类：
    # **不报错，只是把东西弄坏了。** 而 `--mock` 恰好是文档里推荐的
    # "快速验证"入口 —— 也就是说这个坑是**推荐用法必然会踩到的**。
    #
    # Mock 的输出改用 `_` 前缀（本项目惯例：`_` 前缀 = 临时产物，见 .gitignore）。
    if args.save:
        out_path = Path(args.save)
    else:
        out_path = ROOT / "data" / (
            "prospective_eval.json" if real else "_prospective_eval_mock.json")

    # ---------------------------------------------------------- 跑预测
    if not args.report_only:
        print(f"\n{C['bold']}{C['cyan']}前瞻验证评测{C['reset']}")
        hr()
        print(f"""
  数据        {data_path.name}（{len(records)} 条，来自 {meta.get('repo','?')}）
  模型        {'真实' if real else 'Mock'}
  对照目标    维护者打上的**首个类型标签**

  {C['bold']}信息隔离{C['reset']}（这是实验有效性的前提）
    Agent 只拿到：标题 + 正文（创建时就有的信息）
    **刻意不给**：当前标签 / 最终状态 / gold 标签
    {C['dim']}给了就是作弊 —— 评测会虚高且毫无意义。{C['reset']}
""")
        if real:
            est = len(records) * 66 / 60
            print(f"  {C['yellow']}⚠ 预计约 {est:.0f} 分钟（按 66 秒/条估算）{C['reset']}\n")

        settings = load_settings()
        store = build_seed_memory(records, {r["issue_id"] for r in records})
        llm = build_llm(use_mock=not real)

        results = []
        t0 = time.time()
        for i, rec in enumerate(records, 1):
            p = predict(rec, store, llm, settings)
            gold = [rec["first_type_label"]]
            # ★ 归一化：细分格式（bug/crash）→ 粗标签（Bug）
            pred_coarse = normalize_pred(p["predicted"])
            hit = bool(set(pred_coarse) & set(gold))
            exact = sorted(pred_coarse) == sorted(gold)
            results.append({
                "issue_id": rec["issue_id"],
                "title": rec["title"][:70],
                "predicted": pred_coarse,
                "predicted_raw": p["predicted"],
                "gold": gold,
                "hit": hit,
                "exact": exact,
                "lag_days": rec.get("first_type_lag_days", 0),
                "state_labels": rec.get("state_labels", []),
                "steps": p["steps"],
                "tokens": p["tokens"],
                "error": p["error"],
            })
            mark = (f"{C['green']}✓{C['reset']}" if hit
                    else f"{C['red']}✗{C['reset']}")
            print(f"  [{i:>3}/{len(records)}] {mark} {rec['issue_id']:<9} "
                  f"预测={str(pred_coarse):<20} 原始={str(p['predicted'])[:18]:<18} "
                  f"真实={gold}")

        elapsed = time.time() - t0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "meta": {**meta, "model": "real" if real else "mock",
                     "n": len(results), "elapsed_sec": round(elapsed, 1)},
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  {C['green']}✓{C['reset']} 结果 → {out_path}"
              f"（{elapsed / 60:.1f} 分钟）")
    else:
        if not out_path.exists():
            print(f"{C['red']}✗ 没有已有结果：{out_path}{C['reset']}")
            return 1
        d = json.loads(out_path.read_text(encoding="utf-8"))
        results = d["results"]

    # ---------------------------------------------------------- 报告
    hr("★ 前瞻验证结果")

    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    exacts = sum(1 for r in results if r["exact"])
    errors = [r for r in results if r["error"]]
    empty = [r for r in results if not r["predicted"]]

    print(f"""
  样本数        {n}
  {C['bold']}命中率（预测包含真实标签）{C['reset']}   {C['green']}{hits / n:.1%}{C['reset']}（{hits}/{n}）
  完全一致（标签集合完全相同）  {exacts / n:.1%}（{exacts}/{n}）
  未给出预测                   {len(empty)}（{len(empty) / n:.0%}）
  执行异常                     {len(errors)}""")

    # 按标签分层
    hr("按标签分层（哪类问题更难预测）")
    by_gold: dict[str, list] = defaultdict(list)
    for r in results:
        by_gold[r["gold"][0]].append(r)

    print(f"\n  {'标签':<14}{'样本':>6}{'命中':>6}{'命中率':>9}　说明")
    print(f"  {'-' * 62}")
    for lab, items in sorted(by_gold.items(),
                             key=lambda x: -len(x[1])):
        h = sum(1 for r in items if r["hit"])
        rate = h / len(items)
        col = C["green"] if rate >= 0.7 else (C["yellow"] if rate >= 0.4 else C["red"])
        note = ""
        if len(items) < 5:
            note = f"{C['dim']}样本少，参考价值有限{C['reset']}"
        print(f"  {lab:<14}{len(items):>6}{h:>6}{col}{rate:>8.0%}{C['reset']}　{note}")

    # 混淆矩阵（如果预测里有错的）
    hr("错误明细（人工可核对）")
    wrong = [r for r in results if not r["hit"]]
    verdicts = []
    if not wrong:
        print(f"  {C['green']}全部命中{C['reset']}")
    else:
        # ★ 逐条做归因分类 —— 但"归因"和"对错"是两件事：
        #   对错只有一个是非，归因有五种性质完全不同的原因，
        #   而其中两种**根本不是模型的问题**。
        #   只报"错了 N 条"会把行动方向指向模型，而病灶可能在评测。
        for r in wrong:
            verdicts.append(classify_error_detailed(
                r["predicted"], r["gold"],
                {
                    "title": r["title"],
                    "error": r.get("error", ""),
                    # ★ 粗化函数由**调用方**传入 —— 别名表只有一份（就是本脚本的
                    #   `to_coarse`），taxonomy 模块不复制它。
                    "coarsen": to_coarse,
                    # ★ 必须传粗化**之前**的原始预测：线索正是在粗化那一步丢的。
                    "predicted_raw": r.get("predicted_raw", []),
                    # 系统已知的前缀 = 粗化表的键。不在这个集合里的前缀
                    # 说明模型**没有渠道**知道它该归到哪一类。
                    "known_prefixes": set(_COARSE_MAP.keys()),
                }))

        print(f"\n  {'issue':<10}{'预测':<26}{'真实':<14}"
              f"{'归因':<18}标题")
        print(f"  {'-' * 90}")
        for r, v in zip(wrong, verdicts):
            cat = v.category.value
            if cat == "evaluation_bug":
                col = C["yellow"]      # 评测的锅 —— 要显眼，因为它最容易被误记到模型头上
            elif cat == "tool_failure":
                col = C["red"]
            elif cat == "reasoning_error":
                col = C["dim"]
            else:
                col = C["green"]
            print(f"  {r['issue_id']:<10}{str(r['predicted']):<26}"
                  f"{r['gold'][0]:<14}{col}{cat:<18}{C['reset']}"
                  f"{r['title'][:30]}")
        if len(wrong) > 15:
            print(f"  {C['dim']}... 还有 {len(wrong) - 15} 条{C['reset']}")

        # ------------------------------------------------------ 分类汇总
        hr("错误分类（error taxonomy）")
        rep = taxonomy_report(verdicts)

        print(f"\n  样本量  {rep['样本量']} 条**错误**"
              f"　{C['dim']}（不是评测集大小 —— 两者别混）{C['reset']}\n")
        print(f"  {'类别':<14}{'数量':>5}{'占比':>8}   责任归属")
        print(f"  {'-' * 58}")
        for name_cn, d in rep["分类"].items():
            blame = d["责任归属"]
            col = (C["yellow"] if "评测" in blame
                   else (C["red"] if "工程" in blame else C["dim"]))
            print(f"  {name_cn:<14}{d['数量']:>5}{d['占比']:>7.0%}   "
                  f"{col}{blame}{C['reset']}")

        print(f"\n  {C['bold']}★ 责任归属聚合（决定下一步该动哪里）{C['reset']}")
        for blame, cnt in rep["责任归属聚合"].items():
            print(f"    {blame:<20}{cnt:>3} 条")

        # ★ 这一段的用意：把"该找谁修"直接写在报告里。
        #   本项目真实的 4 条错误里 3 条是评测侧的 ——
        #   如果只报"错了 4 条"，下一步会去调模型，方向完全是错的。
        n_model = sum(c for b, c in rep["责任归属聚合"].items() if "模型" in b)
        n_other = rep["样本量"] - n_model
        if n_other > n_model:
            print(f"""
  {C['yellow']}⚠ {rep['样本量']} 条错误里有 {n_other} 条**不是模型的问题**{C['reset']}
    {C['dim']}只报"错了 {rep['样本量']} 条"会把方向指向模型，
      而真正的病灶在评测/工程侧。**修归因指向的那一环**，
      改模型是白费力气。{C['reset']}""")

        print(f"\n  该怎么做：")
        for name_cn, d in rep["分类"].items():
            print(f"    {name_cn:<14}{d['该怎么做']}")

        print(f"\n  {C['dim']}{rep['提示']}{C['reset']}")
        if rep["证据强度不足比例"]:
            print(f"  {C['dim']}证据强度不足比例 {rep['证据强度不足比例']:.0%} —— "
                  f"这部分落到了兜底分支（未定位到具体误导源），"
                  f"需要人工看原文才能进一步细分{C['reset']}")
        if rep["信息不足比例"]:
            print(f"  {C['dim']}信息不足比例 {rep['信息不足比例']:.0%} —— "
                  f"这部分分类不可靠，需人工复核{C['reset']}")

        # ★ Mock 数据的结论不能进正式报告 —— 这句必须出现在**同一屏内**，
        #   否则有人会截图 taxonomy 段落去引用（它看起来很像结论）。
        if not real:
            print(f"""
  {C['red']}⚠ 以上分类基于 Mock 模型（规则版）的预测{C['reset']}
    {C['dim']}Mock 只能验证**评测流程通不通**，不能验证**模型准不准** ——
      这里的高错误率反映的是 Mock 的能力上限，不是系统的真实水平。
      **本段结论不用于正式报告**（评论文档、对外材料）
      正式口径需要真实模型的结果。{C['reset']}""")

    # 与离线评测对比
    hr("★ 前瞻验证 vs 离线评测（本实验最有价值的部分）")
    offline = ROOT / "data" / "eval_pandas_fixed.json"
    offline_note = ""
    if offline.exists():
        try:
            od = json.loads(offline.read_text(encoding="utf-8"))
            # 尝试取离线指标（结构可能不同，取不到就跳过）
            om = od.get("metrics") or od.get("summary") or {}
            f1 = om.get("strict_f1_median") or om.get("f1_median")
            if f1:
                offline_note = f"离线（gold 标签）严格 F1 中位：{f1}"
        except Exception:                                   # noqa: BLE001
            pass

    print(f"""
  离线评测（历史 holdout）    {offline_note or '见 data/eval_pandas_fixed.json'}
  前瞻验证（本实验）          命中率 {hits / n:.1%}

  {C['bold']}为什么这两个数字不能直接比大小{C['reset']}

    · **样本不同**：离线是已关闭 issue（分布偏"已解决"），
      前瞻是刚创建的 issue（含"信息不足"的）
    · **对照目标不同**：离线对的是多层 gold 标签，
      前瞻对的是"维护者打的第一个类型标签"
    · **难度不同**：前瞻更接近真实场景（信息更少、更模糊）

  {C['dim']}★ 关键不是"数字谁高"，而是：
    **前瞻验证的数字才是能对面试官说的那个** ——
    因为它回答了"在不知道答案的时候，你的系统准不准"。{C['reset']}""")

    # 状态标签的对照（辅助分析）
    with_state = [r for r in results if r["state_labels"]]
    if with_state:
        hr("辅助观察：维护者的分诊动作")
        cnt = Counter()
        for r in with_state:
            for s in r["state_labels"]:
                cnt[s] += 1
        print(f"\n  有状态标签的样本 {len(with_state)}/{n}")
        for s, c in cnt.most_common():
            print(f"    {s:<24}{c}")
        print(f"""
  {C['dim']}  ★ 状态标签（Needs Info / Duplicate Report）代表维护者的**分诊动作**。
    我们的 Agent 目前**不预测**这些 —— 这是一个诚实的边界：
    "打什么类型标签"和"要不要退回给作者补充信息"是两个不同的判断。

    如果要做，可以参考 `docs/06` 的 P2-1（记忆污染实验）的思路：
    把"维护者退回"当成一个可对照的信号来验证。{C['reset']}""")

    # 数据质量
    if meta:
        hr("数据质量（诚实边界）")
        print(f"""
  采集窗口    {meta.get('window_days')} 天前创建
  无类型标签  {meta.get('n_no_label', '?')}/{meta.get('n_candidates', '?')}
              （{C['dim']}不是所有 issue 都会被分诊 —— 这个比例本身是信息{C['reset']}）
  正文被编辑  {meta.get('n_valid', 0)} 条中 0 条（本次）
  分诊延迟    中位 0 天 → 说明 pandas triage 很快，
              真前瞻实验**只需等 1~7 天就能对账**

  {C['bold']}本实验的诚实边界{C['reset']}
    · 只覆盖 pandas 一个仓库
    · 对照的是"首个类型标签"，不含组件标签
    · 样本量 {n}，按标签分层后每层样本偏少
    · 未排除"维护者参考了类似 issue"这类信息（Agent 也没有）
""")

    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
