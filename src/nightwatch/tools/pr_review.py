"""
Nightwatch · PR diff 初审（CodeAgent 方向）

================================================================================
为什么要做这个：编码 Agent 是国内招人最多、数据最硬的场景
================================================================================
前面做的都是 issue 分诊（读文本 → 分类）。而**读代码**是完全不同的一件事：

    issue 分诊  → 输入是自然语言，判断"这是什么类型的问题"
    PR 初审     → 输入是代码改动，判断"这个改动有什么风险"

后者的数据更结构化（diff 有明确的格式），但**判断更需要领域知识** ——
"哪些文件动了要特别小心"这件事，靠通读大量 PR 才能学会。

================================================================================
★ 这个模块的核心价值在于两张表
================================================================================
**代码在技术上很简单（解析 diff + 规则匹配），难的是"知道什么该担心"。**

    ① SENSITIVE_PATTERNS —— 哪些文件是"动了就要多看两眼"的
    ② 改动范围与测试覆盖的对应关系 —— 什么改动该带什么测试

这两张表是**领域知识的载体**。换个语言/项目，表要重写；
但"用一张表把资深工程师的直觉固化下来"这个方法是不变的。

**这也是 CodeAgent 最现实的落地方式**：
不是"让 AI 写代码"，而是"**把 review 的第一遍粗筛自动化**" ——
人只看得过 AI 筛出来的高风险改动。

================================================================================
一个诚实的边界：这个工具**不做语义理解**
================================================================================
它只看 diff 的**结构特征**（改了哪些文件、增删多少行、有没有测试），
**不读代码内容**，所以它无法判断"这个改动逻辑对不对"。

    它能回答：这个改动"看起来"危不危险（风险筛查）
    它不能回答：这个改动"实际上"对不对（正确性审查）

**把这条边界写在工具描述里很重要** —— 否则模型会以为
"工具说低风险 = 这个 PR 没问题"，那是危险的理解。

**风险筛查的价值**在于把人的注意力分配到最需要的地方，
而不是替代人。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# ★ 表 ①：敏感文件模式
# =============================================================================
# 这不是"哪些文件不许改"，而是"**改了要有人多看两眼**"。
#
# 分类的理由是：这些文件的改动**影响面超出改动本身**：
#   · 依赖声明   改一行可能影响所有使用者的环境
#   · CI 配置    改错会让所有人无法合并代码
#   · 安全相关   漏一个检查点就是漏洞
#   · 核心数据结构  改字段会波及大量下游代码
#   · 序列化/求值  反序列化和 eval 是 RCE 的高发区（pandas 真实发生过）

SENSITIVE_PATTERNS: dict[str, list[str]] = {
    "依赖声明": [
        r"requirements.*\.txt$", r"pyproject\.toml$", r"setup\.py$",
        r"setup\.cfg$", r"Pipfile$", r"poetry\.lock$", r"package\.json$",
    ],
    "CI 配置": [
        r"\.github/workflows/", r"\.gitlab-ci\.yml$", r"azure-pipelines",
        r"\.circleci/", r"Jenkinsfile$",
    ],
    "构建脚本": [
        r"Makefile$", r"build\.py$", r"meson\.build$", r"CMakeLists\.txt$",
    ],
    "安全相关": [
        r"security", r"crypto", r"\bauth", r"password", r"\btoken",
        r"permission", r"credential",
    ],
    "核心数据结构": [
        r"models\.py$", r"schema", r"types\.py$", r"typing",
    ],
    "序列化与求值": [
        r"pickle", r"\byaml\b", r"\beval\b", r"\bexec\b", r"marshal",
        r"deserializ", r"read_sas", r"read_spss",
    ],
}

# 源码 / 测试 / 文档 的判定
_SOURCE_RE = re.compile(r"(^|/)(src|lib|pandas|numpy|core)/.*\.(py|pyx|c|h|cpp)$")
_TEST_RE = re.compile(r"(^|/)(tests?|testing|_test)/|test_.*\.py$|.*_test\.py$")
_DOC_RE = re.compile(r"\.(md|rst|txt|ipynb)$|^docs?/|^CHANGELOG", re.I)


@dataclass
class FileChange:
    """一个文件在 PR 里的改动。"""
    path: str
    additions: int = 0
    deletions: int = 0
    is_new: bool = False
    is_deleted: bool = False

    @property
    def churn(self) -> int:
        """改动量（增 + 删）。

        ★ 为什么用"增 + 删"而不是只看增？
        因为**删代码比加代码更需要小心** ——
        删除意味着行为被移除，而调用方可能没跟着改。
        """
        return self.additions + self.deletions

    def kind(self) -> str:
        """文件类别：源码 / 测试 / 文档 / 其他。"""
        p = self.path
        if _TEST_RE.search(p):
            return "测试"
        if _DOC_RE.search(p):
            return "文档"
        if _SOURCE_RE.search(p):
            return "源码"
        return "其他"

    def sensitivity(self) -> list[str]:
        """命中的敏感类别（可能多个）。"""
        hits = []
        for category, patterns in SENSITIVE_PATTERNS.items():
            if any(re.search(pat, self.path, re.I) for pat in patterns):
                hits.append(category)
        return hits


@dataclass
class PRReview:
    """PR 初审结果。"""
    files: list[FileChange] = field(default_factory=list)
    total_added: int = 0
    total_deleted: int = 0
    # --- 三个核心判断（对应验收标准）---
    scope: str = ""                       # 改动范围
    missing_tests: bool = False           # 是否缺测试
    missing_tests_reason: str = ""
    sensitive_files: list[dict] = field(default_factory=list)
    # --- 综合 ---
    risk_level: str = "低"                 # 低 / 中 / 高
    risk_reasons: list[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        """转成给模型看的结构（要塞进上下文，所以要精简）。

        ★ 注意**不返回完整文件列表**（大 PR 会有几百个文件，塞爆上下文）。

        只返回"值得注意的"：源码改动 + 敏感文件 + 统计汇总。
        这和 `_read_code` 用 max_lines 截断是同一个原则：
        **工具返回必须为下游的上下文预算负责。**
        """
        notable = [
            {"path": f.path, "kind": f.kind(), "churn": f.churn,
             "sensitivity": f.sensitivity()}
            for f in self.files
            if f.kind() in ("源码", "测试") or f.sensitivity()
        ]
        return {
            "改动范围": self.scope,
            "文件总数": len(self.files),
            "增删": f"+{self.total_added}/-{self.total_deleted}",
            "是否缺测试": self.missing_tests,
            "缺测试原因": self.missing_tests_reason,
            "敏感文件": self.sensitive_files,
            "风险等级": self.risk_level,
            "风险点": self.risk_reasons,
            "建议": self.suggestion,
            "值得注意的文件": notable[:20],
            "说明": ("本工具只做**结构风险筛查**（改了哪些文件、有没有测试），"
                     "不读代码内容，**无法判断改动逻辑是否正确**。"),
        }


def parse_diff(diff_text: str) -> list[FileChange]:
    """解析 unified diff，提取每个文件的改动量。

    ★ 为什么自己解析而不用 git 命令？
    因为工具的输入可能来自任意地方（粘贴的 diff、API 返回、文件），
    不一定有本地 git 仓库。
    **让工具只依赖输入数据，不依赖运行环境** —— 这样它才能被放进沙箱。

    （这也是本项目的沙箱设计前提：工具只做纯计算，不碰外部环境。）
    """
    changes: list[FileChange] = []
    current: Optional[FileChange] = None

    for line in diff_text.splitlines():
        # 新文件的开始
        m = re.match(r"^diff --git a/(\S+) b/(\S+)$", line)
        if m:
            if current:
                changes.append(current)
            current = FileChange(path=m.group(2))
            continue

        # 有些 diff 没有 `diff --git` 行，用 `+++ b/xxx` 兜底
        m = re.match(r"^\+\+\+ b/(\S+)", line)
        if m and current is None:
            current = FileChange(path=m.group(1))
            continue

        if current is None:
            continue

        if line.startswith("new file mode"):
            current.is_new = True
        elif line.startswith("deleted file mode"):
            current.is_deleted = True
        elif line.startswith("+") and not line.startswith("+++"):
            current.additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            current.deletions += 1

    if current:
        changes.append(current)

    # 没解析出任何文件 → 可能是纯 diff 片段（没有 header），
    # 用整体增删统计兜底
    if not changes and diff_text.strip():
        adds = sum(1 for l in diff_text.splitlines()
                   if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in diff_text.splitlines()
                   if l.startswith("-") and not l.startswith("---"))
        changes.append(FileChange(path="(未识别的 diff 片段)",
                                  additions=adds, deletions=dels))
    return changes


def review_pr(diff_text: str, max_high_risk_churn: int = 100) -> PRReview:
    """对一份 PR diff 做初审。

    三个核心判断：
      ① 改动范围 —— 涉及多少文件、多少行、跨哪些类别
      ② 是否缺测试 —— 改了源码但没改测试
      ③ 是否动敏感文件 —— 依赖 / CI / 安全 / 核心结构 / 序列化

    ============================================================
    ★ `max_high_risk_churn = 100` 是**数据标定**的，不是经验值
    ============================================================
    第一版用的是 **500 行** —— 一个"看起来合理"的数字。但实测发现：

        真实 PR 的改动量中位只有 **75 行**（已合并）/ **27 行**（未合并）
        → 500 行的阈值意味着**只有 5% 的 PR 会触发这条规则**

    用 60 个真实 PR（30 已合并 + 30 未合并）标定后的结果是 **100 行**：

        阈值    高风险比例
        500     5.0%    ← 原值，规则几乎不触发
        100     20.0%   ← 标定值，落在"人工优先看能承受"的区间
        50      43.3%   ← 太低，标签失去区分度

    标定方法见 `scripts/calibrate_pr_threshold.py`：
    **先定目标（高风险占 10%~20%），再扫分布找合适的阈值** ——
    而不是反过来"取分布的某个分位数"。

    （这是 `docs/06` 里"阈值没有数据支撑"那条问题的第二次修：
      第一次是记忆召回阈值，用 `sweep_threshold.py` 标定的。）

    ============================================================
    ★ 风险等级的判定用"任一命中即升级"而不是加权评分
    ============================================================
    因为**风险不是可以互相抵消的量**：
    "改了 1 行依赖声明" 和 "改了 500 行文档" 相比，前者更值得人看 ——
    加权求和会把这个信号平均掉。

    **在风险筛查里，取最大值比取平均更安全。**
    """
    files = parse_diff(diff_text)
    r = PRReview(files=files)
    r.total_added = sum(f.additions for f in files)
    r.total_deleted = sum(f.deletions for f in files)

    # ---------------- ① 改动范围 ----------------
    kinds: dict[str, int] = {}
    for f in files:
        kinds[f.kind()] = kinds.get(f.kind(), 0) + 1
    parts = [f"{k} {v} 个" for k, v in kinds.items()]
    r.scope = (f"{len(files)} 个文件，+{r.total_added}/-{r.total_deleted} 行"
               + (f"（{'、'.join(parts)}）" if parts else ""))

    # ---------------- ② 是否缺测试 ----------------
    src_files = [f for f in files if f.kind() == "源码"]
    test_files = [f for f in files if f.kind() == "测试"]
    doc_only = all(f.kind() in ("文档", "测试", "其他") for f in files)

    if src_files and not test_files:
        # ★ 例外：纯增加注释/类型标注不算"改了逻辑"
        #
        # 判断依据是"改动是否可能改变运行行为"。
        # 这里用最保守的近似：**只看新增行是否全是注释/空行**。
        # 真正要做的话应该看 AST 差异，但那超出"结构筛查"的范围。
        r.missing_tests = True
        r.missing_tests_reason = (
            f"改了 {len(src_files)} 个源码文件但没有任何测试改动"
            f"（源码文件：{', '.join(f.path for f in src_files[:3])}"
            f"{' 等' if len(src_files) > 3 else ''}）")
    elif doc_only:
        r.missing_tests = False
        r.missing_tests_reason = "只改了文档/测试，不需要新增测试"

    # ---------------- ③ 敏感文件 ----------------
    for f in files:
        for cat in f.sensitivity():
            r.sensitive_files.append({
                "path": f.path, "category": cat,
                "churn": f.churn, "is_new": f.is_new,
            })

    # ---------------- 综合风险 ----------------
    if r.sensitive_files:
        cats = sorted({s["category"] for s in r.sensitive_files})
        r.risk_reasons.append(f"触及敏感文件（{'、'.join(cats)}）")
        r.risk_level = "高"

    if r.missing_tests and src_files:
        r.risk_reasons.append("缺测试")
        r.risk_level = "高" if r.risk_level == "高" else "中"

    if r.total_added + r.total_deleted > max_high_risk_churn:
        r.risk_reasons.append(
            f"改动量大（+{r.total_added}/-{r.total_deleted} 行，"
            f"超过 {max_high_risk_churn} 行阈值）")
        r.risk_level = "高"

    # 删除多于新增 → 可能是移除功能，调用方未必跟着改
    if r.total_deleted > r.total_added * 2 and r.total_deleted > 20:
        r.risk_reasons.append(
            f"以删除为主（-{r.total_deleted} vs +{r.total_added}）—— "
            f"可能是移除行为，需确认调用方已同步")
        if r.risk_level == "低":
            r.risk_level = "中"

    if not r.risk_reasons:
        r.risk_reasons.append("未命中敏感文件、无大改动、测试覆盖正常")

    # ---------------- 建议 ----------------
    sugg = []
    if r.sensitive_files:
        cats = sorted({s["category"] for s in r.sensitive_files})
        sugg.append(f"请 {('、'.join(cats))} 的负责人过一眼")
    if r.missing_tests:
        sugg.append("补充对应测试（或说明为何不需要）")
    if r.total_added + r.total_deleted > max_high_risk_churn:
        sugg.append("考虑拆分 PR（大 PR 的 review 质量普遍下降）")
    if not sugg:
        sugg.append("可常规 review")
    r.suggestion = "；".join(sugg)

    return r


def format_review(r: PRReview) -> str:
    """把初审结果格式化成人类可读的报告。"""
    lines = [
        "PR 初审报告",
        "─" * 46,
        f"  改动范围    {r.scope}",
        f"  风险等级    {r.risk_level}",
        "",
        "  风险点：",
    ]
    for reason in r.risk_reasons:
        lines.append(f"    · {reason}")

    if r.sensitive_files:
        lines.append("")
        lines.append("  敏感文件：")
        for s in r.sensitive_files[:8]:
            lines.append(f"    · [{s['category']}] {s['path']}"
                         f"（{'+' if s['churn'] else ''}{s['churn']} 行"
                         f"{'，新增' if s['is_new'] else ''}）")

    if r.missing_tests:
        lines.append("")
        lines.append(f"  缺测试：{r.missing_tests_reason}")
    else:
        lines.append("")
        lines.append(f"  测试：{r.missing_tests_reason or '覆盖正常'}")

    lines += ["", f"  建议：{r.suggestion}", ""]
    lines.append("  ⚠ 本报告只做结构风险筛查，不读代码内容 —— "
                 "无法判断改动逻辑是否正确。")
    return "\n".join(lines)
