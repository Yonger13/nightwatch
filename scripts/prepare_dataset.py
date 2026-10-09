#!/usr/bin/env python3
"""
Nightwatch · 数据集准备：把"原始 issue"变成"可评测的数据集"

================================================================================
为什么必须有这个脚本
================================================================================
`collect_issues.py` 采下来的是**原始数据**。原始数据不能直接拿去评测，
因为真实项目的标签体系不是为评测设计的。

以 pandas 为例（这是实测结果，不是假设）：

    · 300 条 issue 里有 **91 个不同的标签**
    · 每条 issue 平均 **2 个标签**（最多 5 个）
    · 标签混杂了三种性质完全不同的东西：
        问题类型：Bug / Docs / Enhancement / Performance / Regression  ← 可以评测
        领域组件：Arrow / IO CSV / Datetime / Groupby                 ← 需读源码，超出范围
        流程状态：Needs Triage / Closing Candidate / Needs Discussion ← 是状态，不是分类

**如果直接把 91 个标签丢给 Agent，评测结果毫无意义** ——
它不可能同时预测"是 Bug"和"属于 IO CSV 子模块"和"当前处于 Needs Triage 阶段"。

================================================================================
本脚本做的四件事
================================================================================
    ① 标签分层    把 91 个标签拆成 {类型 / 领域 / 流程} 三类
    ② 收敛口径    只保留"类型"作为评测目标（这是分流任务的真实目标）
    ③ 修正 ground truth   重新提取 duplicate 关系（原始提取有误匹配）
    ④ 分层采样    按类型分层 + 划分留出集

================================================================================
一个重要的设计观点
================================================================================
**收敛标签口径不是"降低难度"，而是"把任务定义清楚"。**

Agent 在这个场景里的职责是**分流**：判断这是什么类型的问题、该给谁处理。
判断"这属于 pandas 的哪个子模块"需要读源码，那是维护者做的事。

把不该它做的事算进指标里，得出的数字既低又没意义 ——
而且会误导优化方向（你会去优化一个不该优化的东西）。

**先想清楚"任务是什么"，再设计指标。** 这个顺序不能反。

================================================================================
用法
================================================================================
    python scripts/prepare_dataset.py --input data/real_pandas.json \\
        --output data/pandas_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.eval.dataset import DatasetBuilder                   # noqa: E402
from nightwatch.models import Issue, IssueState, to_dict             # noqa: E402


# =============================================================================
# 标签分层
# =============================================================================
# 这三张表是**看了 300 条真实数据之后**总结的，不是凭空想的。
# 如果你换了项目，必须重新整理这三张表 —— 不同项目的标签体系完全不同。
#
# 判断标准：
#   TYPE       回答"这是什么问题"      → 分流任务的目标
#   DOMAIN     回答"涉及哪个模块"      → 需要领域知识，超出范围
#   WORKFLOW   回答"当前处于什么阶段"  → 是状态，不是分类

TYPE_LABELS = {
    "Bug",              # 缺陷
    "Enhancement",      # 功能增强
    "Feature",          # 功能请求（不同项目用词不同）
    "Docs",             # 文档问题（pandas 用 Docs）
    "Documentation",    # 文档问题（其他项目常用词）
    "Performance",      # 性能问题
    "Regression",       # 回归（曾经能跑，现在不行）
    "Testing",          # 测试相关
    "Build",            # 构建/打包
    "Deprecate",        # 弃用
    "Usage Question",   # 用法提问（应该转成讨论/文档，不是代码问题）
    "API Design",       # API 设计讨论
    "CI",               # 持续集成
    "Refactor",         # 重构
    "Internals",        # 内部实现
    "Benchmark",        # 基准测试
    "Typing",           # 类型标注
    "Visualization",    # 可视化
    "Error Reporting",  # 报错信息质量
    "Warnings",         # 警告
}

# 流程/状态标签：**不参与评测**，但可以用来判断"这条样本算什么结果"
WORKFLOW_LABELS = {
    "Needs Triage",      # 待分类 —— 维护者还没判断
    "Needs Discussion",  # 需要讨论
    "Needs Info",        # 需要补充信息
    "Closing Candidate", # 候选关闭
    "Duplicate Report",  # 重复报告
    "Upstream issue",    # 上游问题（不是本项目的问题）
    "Non-Nano",          # pandas 特有的流程标签
    "Change Default Keyword/Option Request",
}

# 其余全部视为 DOMAIN（领域/组件）
# 不显式列举，因为数量多且随项目变化 —— 用"不在前两类里"来判定


# =============================================================================
# 标题前缀 → 类型标签（项目领域约定）
# =============================================================================
# ★ 这个映射表来自"人工通读真实数据"的发现，不是从文档里抄的。
#
# 现象：很多 issue 维护者**完全没打标签**，但标题里带着约定俗成的前缀：
#
#     BUG: read_sas segfaults on a SAS7BDAT file...
#     TST: pandas/tests/api/test_api.py::TestApi::test_api fails
#     DIST: Include Vendored Licenses in Wheel
#     CI regression: Doc Build and pre-commit jobs failing
#     [BUG] Use-after-free in int64ToIso
#     BLD|CI: Failing to Build Wheels on macosx_x86_64
#
# **这些前缀本身就是分类信号** —— 而且比维护者的标签更可靠（因为它是
# 提交者写标题时的自觉约定，覆盖率更高）。
#
# 第一版实现完全没利用它，导致把这些 issue 当成"无标签 → 信息不足 → 该转人工"，
# 而实际上它们的类型判断**非常明确**。
#
# 教训：**领域约定要靠通读数据发现，不能靠看文档。**
# 这正是第 0 周"人工通读 30~50 条"的价值 —— 它找的是这类没写在文档里的规则。
#
# ⚠️ 换项目时必须重写这张表（不同项目的标题约定完全不同）。

TITLE_PREFIX_TO_TYPE = {
    "BUG": "Bug",
    "ENH": "Enhancement",
    "ENHANCEMENT": "Enhancement",
    "FEAT": "Enhancement",
    "DOC": "Docs",
    "DOCS": "Docs",
    "TST": "Testing",
    "TEST": "Testing",
    "BLD": "Build",
    "BUILD": "Build",
    "DIST": "Build",
    "CI": "CI",
    "DEPR": "Deprecate",
    "PERF": "Performance",
    "REG": "Regression",
    "REF": "Refactor",
    "API": "API Design",
    "VIZ": "Visualization",
    "TYP": "Typing",
    "QST": "Usage Question",
    "QUESTION": "Usage Question",
    "SEC": "Bug",
}


# 正文小标题 → 类型（pandas 的 issue 模板用 markdown 标题分节）
#
# 第二轮补漏时发现的：`#65987 Add logging support...` 的标题没有前缀，
# 但正文第一行是 `## Feature Request` —— **项目自带的 issue 模板在帮忙做分类**。
# 这类信号同样不该浪费。
BODY_HEADING_TO_TYPE = {
    "feature request": "Enhancement",
    "bug report": "Bug",
    "bug description": "Bug",
    "describe the bug": "Bug",
    "enhancement": "Enhancement",
    "documentation": "Docs",
    "performance issue": "Performance",
}


def infer_type_from_body(body: str) -> Optional[str]:
    """从正文开头的 markdown 小标题推断类型。

    只扫前 15 行 —— 因为项目模板的标记总在最前面，
    扫全文反而可能命中正文里引用的别的标题。
    """
    if not body:
        return None
    for line in body.splitlines()[:15]:
        s = line.strip().lstrip("#").strip().lower()
        if s in BODY_HEADING_TO_TYPE:
            return BODY_HEADING_TO_TYPE[s]
    return None


def infer_type_from_title(title: str) -> Optional[str]:
    """从标题前缀推断类型标签。

    支持的格式（pandas 里都出现过）：
        `BUG: xxx`          `[BUG] xxx`        `BLD|CI: xxx`
        `TST: xxx`          `[ENH] xxx`        `DOC|DOCS: xxx`

    匹配规则：取标题开头的第一个 2~8 个字母的词（可带方括号），
    查表命中就返回类型。**命中不了就返回 None**（不猜）——
    因为"猜错类型"比"没有类型"更糟。
    """
    if not title:
        return None
    # 形如 `[BUG] ...` 或 `BUG: ...` 或 `BLD|CI: ...`
    m = re.match(r"^\s*(?:\[)?([A-Za-z]{2,8})(?:\])?\s*(?::|：|\||\])", title)
    if not m:
        # 形如 `[BUG] 描述`（方括号后直接空格）
        m = re.match(r"^\s*\[([A-Za-z]{2,8})\]\s", title)
        if not m:
            return None
    return TITLE_PREFIX_TO_TYPE.get(m.group(1).upper())


def classify_label(label: str) -> str:
    """判断一个标签属于哪一层。"""
    if label in TYPE_LABELS:
        return "type"
    if label in WORKFLOW_LABELS:
        return "workflow"
    return "domain"


# =============================================================================
# duplicate 关系的修正提取
# =============================================================================

def extract_duplicate(issue: Issue) -> Optional[str]:
    """从标签、关闭原因、正文引用里提取"重复于哪个 issue"。

    ============================================================
    原提取逻辑的问题（实测发现）
    ============================================================
    `collect_issues.py` 里的提取是：一旦标签含 "dup"，就从正文里抓第一个 `#数字`。
    结果是 `#66035 -> #1` —— `#1` 显然是从正文的"第1步"之类抓来的误匹配。

    **根因：正则太宽 + 没有校验。** `#(\\d+)` 会匹配任何 `#数字`，
    包括标题里的编号、Markdown 标题（`# 一级标题`）、正文里的步骤编号。

    这里收紧三点：
      ① 只匹配 `#12345` 这种**5 位以上**的 issue 编号（pandas 的编号都是 5 位数）
      ② 排除 `# 标题` 这种 Markdown 语法（# 后面跟空格）
      ③ 排除自己（不能"重复于自己"）
    """
    # 情况 1：标签里明确标了 duplicate
    has_dup_label = any("dup" in l.lower() for l in issue.labels)

    # 情况 2：关闭原因是 duplicate（这个信号最可靠）
    is_dup_closed = "duplicate" in (issue.gold_action or "").lower()

    if not (has_dup_label or is_dup_closed):
        return None

    # 从正文里找引用（收紧后的正则）
    text = f"{issue.title}\n{issue.body}"
    # (?<!#) 避免匹配 Markdown 的 "## "；\d{4,} 要求至少 4 位数字
    candidates = re.findall(r"(?<!#)#(\d{4,})", text)
    for c in candidates:
        ref = f"#{c}"
        if ref != issue.issue_id:          # 排除自己
            return ref
    return None


def is_likely_duplicate_by_title(issue: Issue) -> bool:
    """标题层面的重复线索（用于人工复核，不直接当 ground truth）。

    pandas 的重复 issue 标题里常有 "DUPLICATE:" 前缀或 "same as" 之类表述。
    这类信号不严谨，所以只作为**辅助标记**，不写进 ground truth。
    """
    t = (issue.title or "").lower()
    return any(k in t for k in ("duplicate:", "dup of", "same as", "重复"))


# =============================================================================
# 主流程
# =============================================================================

def load_raw(path: Path) -> list[Issue]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in raw.get("train", []):
        out.append(Issue(
            issue_id=d["issue_id"], title=d["title"], body=d["body"],
            author=d["author"], repo=d.get("repo", ""),
            labels=d.get("labels", []), stack_trace=d.get("stack_trace", ""),
            state=IssueState(d.get("state", "closed")),
            gold_labels=d.get("gold_labels", []),
            gold_is_duplicate_of=d.get("gold_is_duplicate_of"),
            gold_action=d.get("gold_action", ""),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="把原始 issue 转成可评测数据集")
    ap.add_argument("--input", required=True, help="collect_issues.py 产出的 JSON")
    ap.add_argument("--output", required=True, help="输出的评测集 JSON")
    ap.add_argument("--holdout-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", help="把标签分层报告写到这个文件")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"✗ 输入文件不存在：{src}")
        return 1

    issues = load_raw(src)
    print(f"\n读入 {len(issues)} 条原始 issue（来自 {src.name}）")

    # ---------------- ① 标签分层 ----------------
    layer_counter: dict[str, Counter] = {"type": Counter(), "domain": Counter(),
                                          "workflow": Counter()}
    for it in issues:
        for l in it.gold_labels:
            layer_counter[classify_label(l)][l] += 1

    print(f"""
{'=' * 74}
① 标签分层（原始数据里有 {len(set(l for i in issues for l in i.gold_labels))} 个标签）
{'=' * 74}""")
    for layer, cn in (("type", "问题类型"), ("workflow", "流程状态"),
                      ("domain", "领域组件")):
        total = sum(layer_counter[layer].values())
        kinds = len(layer_counter[layer])
        note = {"type": "← 评测目标", "workflow": "← 不参与评测",
                "domain": "← 不参与评测（需读源码）"}[layer]
        print(f"\n  【{cn}】{total} 个标签实例 / {kinds} 种 {note}")
        for k, n in layer_counter[layer].most_common(12):
            print(f"    {n:>4}  {k}")

    # ---------------- ② 收敛口径 ----------------
    # ★ 这里必须**真的改数据**，不能只统计。
    #
    # 第一版实现只数了"有多少条有类型标签"，却没过滤 gold_labels ——
    # 结果是：分层采样时 `_stratum_of` 取 `gold_labels[0]`，
    # 而很多 issue 的第一个标签是「领域标签」（Indexing / IO JSON / Datetime），
    # 于是分层表里混进了领域层，评测口径根本没收敛。
    #
    # ============================================================
    # ★★ 第二版修复：必须区分「无标签」和「只有非类型标签」
    # ============================================================
    # 跑完整 holdout 时发现："unlabeled 层"的 14 条里只有 2 条真的信息不足，
    # 其余 12 条标题明明写着 `BUG: ...`（信息非常充分）。
    #
    # 根因：这 12 条的原始标签是**领域标签**（IO SAS / Plotting / Missing-data），
    # 被上面的"只保留类型标签"过滤掉后变成空数组 ——
    # 而我错误地把「空 gold」等同于「正确答案是转人工」。
    #
    # 后果：**模型答对了（预测 Bug / Docs 完全正确）却被判 0 分。**
    # 还让"转人工率偏低"这个结论也变成假的 —— 模型根本不该转人工。
    #
    # 所以现在区分三种情况：
    #   ① 有类型标签        → 正常参与分类评测
    #   ② 只有领域/流程标签 → **剔除，不参与评测**
    #                         （不是"信息不足"，是"我们没测那一层"）
    #   ③ 完全没有任何标签  → 正确答案是「转人工」（gold = []）
    #                         （这才是真正的信息不足，例如 "Test Issue from API"）
    #
    # 教训：**"没有 gold"和"gold 被过滤掉了"是两件完全不同的事。**
    # 把它们混为一谈，会让模型背锅。
    n_with_type = 0
    n_only_domain = 0
    n_inferred = 0
    n_no_label = 0
    kept: list = []

    for it in issues:
        has_any = bool(it.gold_labels)                    # 先记录原始状态
        types = [l for l in it.gold_labels if classify_label(l) == "type"]
        it.gold_labels = types

        if types:
            n_with_type += 1
            kept.append(it)
        elif has_any:
            n_only_domain += 1                            # ② 剔除
        else:
            # ③ 完全无标签 → 依次尝试两种推断信号：
            #    先看标题前缀（BUG: / TST: / DIST:），再看正文小标题（## Feature Request）
            #
            #    这两条都是"项目约定"，而且都比维护者标签覆盖更全。
            #    第一版完全没利用它们，导致把信息充分的 issue 当成"该转人工"。
            inferred = infer_type_from_title(it.title) or infer_type_from_body(it.body)
            if inferred:
                it.gold_labels = [inferred]
                n_inferred += 1
                kept.append(it)
            else:
                n_no_label += 1                           # 真的信息不足
                kept.append(it)

    issues = kept
    n_total = n_with_type + n_only_domain + n_inferred + n_no_label

    print(f"""
{'=' * 74}
② 收敛评测口径：只保留「问题类型」这一层
{'=' * 74}

  ① 有类型标签的：        {n_with_type} 条（{n_with_type / n_total:.1%}）
  ② 只有领域/流程标签：    {n_only_domain} 条 → **剔除，不参与评测**
  ③ 无标签但标题有前缀：   {n_inferred} 条 → **从前缀推断类型**（如 BUG: / TST: / DIST:）
  ④ 完全无信息的：         {n_no_label} 条 → 正确答案是「转人工」

  ⚠ gold_labels 已被**改写**为只含类型标签 ——
    否则下游分层会误用领域标签（Indexing / IO JSON），口径就不算收敛。

  {C_Y}★ 为什么第 ② 类要剔除，而第 ③ 类要推断？{C_R}""")
    print("""    第 ② 类（只有领域标签）的标题写着 `BUG: read_sas segfaults...` ——
    信息**非常充分**，只是维护者打的标签恰好是领域标签（IO SAS），
    被我们的口径过滤掉了。把它们当成「信息不足」会让模型答对却被判 0 分。

    第 ③ 类（完全没标签）同样信息充分 —— **pandas 有标题前缀约定**：
        BUG: / TST: / DIST: / CI regression: / [BUG] ...
    这些前缀本身就是分类信号，而且比维护者的标签覆盖更全。

    **「没有 gold」和「gold 被过滤掉了」是两件完全不同的事。**
    第一版把它们混为一谈，让模型背了锅。""")

    # ---------------- ③ 修正 ground truth ----------------
    dup_fixed = 0
    dup_dropped = 0
    for it in issues:
        old = it.gold_is_duplicate_of
        new = extract_duplicate(it)
        if old and not new:
            dup_dropped += 1
        if not old and new:
            dup_fixed += 1
        it.gold_is_duplicate_of = new

    total_dup = sum(1 for it in issues if it.gold_is_duplicate_of)
    by_title = sum(1 for it in issues if is_likely_duplicate_by_title(it)
                   and not it.gold_is_duplicate_of)

    print(f"""
{'=' * 74}
③ 修正 duplicate 关系
{'=' * 74}

  原始提取：{sum(1 for i in load_raw(src) if i.gold_is_duplicate_of)} 条
  修正后：  {total_dup} 条
    · 新提取出：{dup_fixed} 条（原逻辑漏掉的）
    · 剔除误匹配：{dup_dropped} 条（比如 "-> #1" 这种）
    · 标题疑似重复但无引用：{by_title} 条（**不作为 ground truth**，
      因为标题里的"same as"之类表述不严谨，只能人工复核）

  {C_Y}样本量警告{C_R}{'' if total_dup >= 10 else ''}""")
    if total_dup < 10:
        print(f"""    duplicate 样本只有 {total_dup} 条 —— **不足以评测判重能力**。
    诚实结论：在这个项目上，duplicate 指标不可报告（样本量太小，
    95% 置信区间会宽到没有意义）。
    → 要么扩大采集量（--limit 1000），要么在简历里明确说"判重未评测"。""")

    # ---------------- ④ 分层采样 ----------------
    builder = DatasetBuilder(seed=args.seed, holdout_ratio=args.holdout_ratio,
                             min_per_stratum=3)
    ds = builder.build(
        issues,
        source=f"github:{src.stem}",
        version="pandas-real-v1",
        notes=("真实数据（GitHub issue），ground truth 来自维护者的标签与关闭动作。"
               "**已收敛口径**：只保留「问题类型」层标签作为评测目标，"
               "领域标签与流程标签不参与评测（理由见 prepare_dataset.py 的文档）。"
               "ground truth 是维护者的判断，不是绝对正确答案。"))

    print(f"""
{'=' * 74}
④ 分层采样与数据集划分
{'=' * 74}""")
    summary = ds.summary()
    print(f"  版本：{summary['版本']}")
    print(f"  来源：{summary['来源']}")
    print(f"  train {summary['train规模']} 条 / holdout {summary['holdout规模']} 条")
    print(f"\n  train 分层分布：")
    for k, v in list(summary["train分层分布"].items())[:14]:
        print(f"    {v:>4}  {k}")
    print(f"\n  holdout 分层分布：")
    for k, v in list(summary["holdout分层分布"].items())[:14]:
        print(f"    {v:>4}  {k}")

    # ---------------- 保存 ----------------
    ds.to_json(args.output)
    print(f"\n{'=' * 74}")
    print(f"✓ 已保存到 {args.output}")
    print(f"{'=' * 74}")
    print(f"""
  下一步：
    # 用真实数据 + 真实模型跑评测
    python scripts/run_eval.py --dataset {args.output} --real-llm --holdout

  {C_DIM}注意：
    · holdout 集在调试期间不要用，只在最后报数时用
    · 每格样本量偏小时（<5）指标波动大，报告里要注明
    · 分层里样本数少于 3 的层已全部并入 train（不做数据增强凑数，
      因为凑出来的样本会污染指标）{C_R}""")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "source": str(src),
            "n_raw": len(issues),
            "labels_by_layer": {
                k: dict(v.most_common()) for k, v in layer_counter.items()},
            "n_with_type_label": n_with_type,
            "n_without_type_label": n_no_type,
            "duplicate_samples": total_dup,
            "duplicate_fixed": dup_fixed,
            "duplicate_dropped": dup_dropped,
            "dataset_summary": summary,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  分层报告已写入 {args.report}")

    return 0


# 终端颜色（简版）
C_Y = "\033[33m"
C_DIM = "\033[2m"
C_R = "\033[0m"


if __name__ == "__main__":
    raise SystemExit(main())
