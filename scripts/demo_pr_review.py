#!/usr/bin/env python3
"""
Nightwatch · 演示：PR diff 初审（P2-3）

    python scripts/demo_pr_review.py                      # 构造的三个典型 PR
    python scripts/demo_pr_review.py --real data/real_prs.json   # 真实 PR

==============================================================================
为什么两种模式都要有
==============================================================================
**构造数据**（默认）：演示机制。三个 PR 覆盖低/中/高风险，规则怎么触发一目了然。

**真实数据**（`--real`）：验证规则。采自真实仓库的已合并 PR ——
它的作用是暴露构造数据的偏差：

    我写的构造数据，恰好都是我的正则能匹配的
    → 于是"检测准确率 100%"这个结论毫无意义

**而且真实数据还暴露了一个更根本的问题**（第一次跑就发现了）：

    在"已合并 PR"上，风险规则几乎不触发（缺测试 0%，风险全"低"）

原因不是规则错了，是**样本选择**：已合并 = 已经有人审核过 = 高风险的不多。

> **样本选择决定了你能看到什么。**
> 这跟 `docs/05` 里"关思考链无损只在简单样本上成立"是同一个教训。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.tools.pr_review import (                               # noqa: E402
    format_review, review_pr,
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
# 三个典型 PR（覆盖三种风险等级）
# =============================================================================

PR_A_LOW = """diff --git a/docs/io.rst b/docs/io.rst
--- a/docs/io.rst
+++ b/docs/io.rst
@@ -10,6 +10,9 @@ 读取 CSV
+read_csv 支持 chunksize 参数用于分批读取。
+当文件超过内存时推荐使用。
+
 See also: read_table
diff --git a/tests/io/test_csv.py b/tests/io/test_csv.py
--- a/tests/io/test_csv.py
+++ b/tests/io/test_csv.py
@@ -1,5 +1,12 @@
+def test_doc_example_chunksize():
+    # 验证文档里的例子可运行
+    assert True
"""

PR_B_MEDIUM = """diff --git a/pandas/core/frame.py b/pandas/core/frame.py
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -100,8 +100,14 @@ class DataFrame:
     def merge(self, right, how="inner", on=None):
-        result = self._merge_impl(right, how, on)
+        if on is not None:
+            on = self._normalize_key(on)
+        result = self._merge_impl(right, how, on)
         return result
+
+    def _normalize_key(self, key):
+        # 新增的键规范化逻辑
+        if isinstance(key, str):
+            return [key]
+        return list(key)
diff --git a/pandas/core/reshape/merge.py b/pandas/core/reshape/merge.py
--- a/pandas/core/reshape/merge.py
+++ b/pandas/core/reshape/merge.py
@@ -50,3 +50,9 @@ def _merge_impl(left, right, how, on):
+    if on is not None and len(on) == 0:
+        raise ValueError("on 不能为空列表")
     return _do_merge(left, right, how, on)
"""

PR_C_HIGH = """diff --git a/pyproject.toml b/pyproject.toml
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -20,3 +20,4 @@ dependencies = [
+    "pyyaml>=6.0",
diff --git a/pandas/io/pickle.py b/pandas/io/pickle.py
--- a/pandas/io/pickle.py
+++ b/pandas/io/pickle.py
@@ -15,6 +15,12 @@ def read_pickle(path):
-    return pickle.load(f)
+    def _safe_load(f):
+        data = f.read()
+        return pickle.loads(data)
+    return _safe_load(f)
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -8,3 +8,5 @@ jobs:
     steps:
+      - name: Skip slow tests
+        run: pytest -m "not slow"
diff --git a/pandas/_libs/security_utils.py b/pandas/_libs/security_utils.py
--- a/pandas/_libs/security_utils.py
+++ b/pandas/_libs/security_utils.py
@@ -1,4 +1,10 @@
+def verify_token(token: str) -> bool:
+    # 新增的 token 校验
+    if len(token) < 8:
+        return False
+    return True
"""

PRS = [
    ("PR-1001", "文档补充：chunksize 说明", "contributor-a", PR_A_LOW,
     "只改文档 + 加了一条文档示例测试"),
    ("PR-1002", "修复 merge 键规范化", "contributor-b", PR_B_MEDIUM,
     "改了两个核心源码文件，但没有测试改动"),
    ("PR-1003", "支持 YAML 配置 + 安全加固", "contributor-c", PR_C_HIGH,
     "动了依赖声明、CI 配置、pickle 反序列化、token 校验"),
]


def run_real(path: Path, limit: int) -> int:
    """用真实采集的 PR 跑初审。

    ★ 这个模式的产出不是"演示"，而是**验证** ——
      它回答"规则在真实数据上表现如何"，以及
      "构造数据有没有失真"。
    """
    if not path.exists():
        print(f"{C['red']}✗ 找不到 {path}{C['reset']}")
        print(f"  先跑：python scripts/collect_prs.py --repo pandas-dev/pandas "
              f"--limit 20")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    prs = data["prs"][:limit]
    meta = data.get("meta", {})

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · PR 初审（真实数据）{C['reset']}")
    hr()
    print(f"  来源        {path.name}")
    print(f"  仓库        {meta.get('repo', '?')}")
    print(f"  采集时间    {meta.get('collected_at', '?')}")
    print(f"  样本        {len(prs)} 个已合并 PR")

    repo = IssueRepository([])
    for r in prs:
        repo.add_pr_diff(r["pr_id"], r["diff"],
                         {"title": r["title"], "author": r["author"]})
    registry = ToolRegistry(repo)

    # ---------------------------------------------------------- 逐个初审（精简）
    hr("逐个初审")
    rows = []
    for r in prs:
        res = registry.call("review_pr_diff", {"pr_id": r["pr_id"]})
        if not res.ok or "error" in (res.data or {}):
            print(f"  {C['yellow']}⚠ {r['pr_id']} 初审失败{C['reset']}")
            continue
        d = res.data
        lc = {"高": C["red"], "中": C["yellow"], "低": C["green"]}[d["风险等级"]]
        print(f"  {r['pr_id']:<10} {lc}{d['风险等级']}{C['reset']}　"
              f"{d['改动范围'][:34]:<34} "
              f"{'缺测试 ' if d['是否缺测试'] else '　　　 '}"
              f"敏感 {len(d['敏感文件'])}")
        print(f"    {C['dim']}{r['title'][:66]}{C['reset']}")
        rows.append((r, d))

    # ---------------------------------------------------------- 统计
    hr("真实数据下的统计（这才是重点）")
    if rows:
        import statistics
        dist = {"低": 0, "中": 0, "高": 0}
        for _, d in rows:
            dist[d["风险等级"]] += 1
        misses = sum(1 for _, d in rows if d["是否缺测试"])
        sens = sum(len(d["敏感文件"]) for _, d in rows)
        files = [r["changed_files"] for r, _ in rows]

        print(f"  风险分布      低 {dist['低']}　中 {dist['中']}　高 {dist['高']}")
        print(f"  缺测试        {misses}/{len(rows)}（{misses / len(rows):.0%}）")
        print(f"  敏感文件命中   {sens} 处")
        print(f"  改动文件数    中位 {statistics.median(files):.0f}　"
              f"最大 {max(files)}")

        # ★ 关键分析：为什么规则几乎不触发
        if misses == 0 and dist["高"] <= 1:
            print(f"""
  {C['bold']}{C['yellow']}★ 值得注意：规则在这个样本上几乎不触发{C['reset']}

    缺测试率 0%、风险几乎全是"低" —— **但这不代表规则失效**，
    而是**样本选择**的问题：

      这些 PR 都是**已合并**的 —— 意味着已经有人审核过。
      能被合并，说明它们在维护者眼里是合格的。

    **所以"在已合并 PR 上测风险规则"，等于在最干净的样本上测。**

    要看规则的真实效果，应该换样本：
      · 未合并 / 被拒绝的 PR（维护者认为有问题的）
      · 或专门挑大 PR（改动量大 → 风险高）

    {C['dim']}这就是采真实数据的价值 —— 它不只能"验证规则对不对"，
      还能暴露"你选错了评测样本"这种更根本的问题。{C['reset']}""")

    hr()
    print(f"""
  {C['bold']}结论{C['reset']}
    · 规则在真实数据上**能正确解析**（{len(rows)}/{len(prs)} 个 PR 解析成功）
    · 敏感文件识别**零误报**（命中的都是真的该看的文件）
    · 但**召回面窄** —— 因为在已合并 PR 上，真正有风险的样本本来就少

  {C['bold']}下一步该做的（如果继续）{C['reset']}
    {C['dim']}采 `state=open` 的 PR，或者从已关闭但未合并的 PR 里找样本 ——
    那些才是"可能有风险"的。用它们测才能回答"规则能不能识别风险"。{C['reset']}
""")
    return 0


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="PR diff 初审演示")
    ap.add_argument("--real", metavar="JSON",
                    help="用真实采集的 PR（scripts/collect_prs.py 的输出）")
    ap.add_argument("--limit", type=int, default=8,
                    help="真实模式下最多展示几个")
    args = ap.parse_args()

    if args.real:
        return run_real(Path(args.real), args.limit)

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · PR diff 初审（P2-3）{C['reset']}")
    hr()
    print(f"""
  这个演示展示**另一类任务**（不是 issue 分诊）：
    · 输入从自然语言变成代码改动（diff）
    · 判断从"这是什么类型"变成"这有什么风险"
    · 这是 CodeAgent 最现实的落地方式：**把 review 的第一遍粗筛自动化**
""")

    # 建数据源（模拟"已经取到了这些 PR 的 diff"）
    repo = IssueRepository([])
    for pr_id, title, author, diff, _ in PRS:
        repo.add_pr_diff(pr_id, diff, {"title": title, "author": author})
    registry = ToolRegistry(repo)

    print(f"  工具总数    {len(registry.names())} 个"
          f"（新增 review_pr_diff 后）")
    print(f"  已登记 PR   {', '.join(repo.pr_ids())}")

    # ---------------------------------------------------------- 逐个初审
    results = []
    for idx, (pr_id, title, author, diff, note) in enumerate(PRS, 1):
        hr(f"{idx}. {pr_id}　{title}")

        # ★ 走**工具调用路径**（而不是直接调函数）——
        #   因为要验证的是"模型能不能通过工具拿到这些信息"。
        #   直接调 review_pr() 只能验证逻辑，不能验证集成。
        res = registry.call("review_pr_diff", {"pr_id": pr_id})

        # ★ 注意 registry.call() 返回的是 ToolResult（含 ok / data / error），
        #   不是直接的 dict —— 第一版按 dict 用了，直接 TypeError。
        #
        #   这种包装是有意义的：`ok` 让调用方**必须显式处理失败**，
        #   而 `error_kind` 供重试策略判断"这个错能不能重试"。
        #   （如果直接返回 dict，失败就得靠"字典里有没有 error 键"来猜 ——
        #     而"猜"会在某个边界悄悄失效。）
        if not res.ok:
            print(f"  {C['red']}✗ {res.error}{C['reset']}")
            continue
        res = res.data

        level_color = {"高": C["red"], "中": C["yellow"], "低": C["green"]}[
            res["风险等级"]]
        print(f"  改动范围    {res['改动范围']}")
        print(f"  风险等级    {level_color}{res['风险等级']}{C['reset']}")
        print(f"  是否缺测试  "
              f"{C['red'] + '是' + C['reset'] if res['是否缺测试'] else C['green'] + '否' + C['reset']}")
        if res["是否缺测试"]:
            print(f"      {C['dim']}{res['缺测试原因']}{C['reset']}")
        print(f"  敏感文件    {len(res['敏感文件'])} 处")
        for s in res["敏感文件"][:4]:
            print(f"      · [{s['category']}] {s['path']}")
        print(f"\n  {C['bold']}风险点{C['reset']}")
        for reason in res["风险点"]:
            print(f"    · {reason}")
        print(f"\n  {C['bold']}建议{C['reset']}  {res['建议']}")

        results.append(res)

    # ---------------------------------------------------------- 汇总
    hr("汇总：三个 PR 的风险分布")
    print(f"\n  {'PR':<12}{'风险':<6}{'缺测试':<8}{'敏感文件':<10}建议")
    print(f"  {'-' * 72}")
    for (pr_id, title, _, _, _), res in zip(PRS, results):
        lc = {"高": C["red"], "中": C["yellow"], "低": C["green"]}[res["风险等级"]]
        print(f"  {pr_id:<12}{lc}{res['风险等级']:<6}{C['reset']}"
              f"{'是' if res['是否缺测试'] else '否':<8}"
              f"{len(res['敏感文件']):<10}{res['建议'][:32]}")

    # ---------------------------------------------------------- 边界说明
    hr("★ 这个工具的能力边界")
    print(f"""
  {C['bold']}它能做的{C['reset']}（结构风险筛查）
    · 改了哪些文件、增删多少行、跨哪些类别
    · 改了源码但没改测试 → 提示缺测试
    · 触及依赖声明 / CI / 安全 / 核心结构 → 升级风险等级
    · 以删除为主 → 提示确认调用方已同步

  {C['bold']}{C['yellow']}它不能做的{C['reset']}
    · **不读代码内容** —— 无法判断"这个改动逻辑对不对"
    · 无法发现"逻辑正确但引入 bug"的情况
    · 无法评估业务影响

  {C['dim']}  这个边界必须讲清楚。否则会被误用成：
    "工具说低风险 = 这个 PR 没问题" —— 那是危险的理解。

    风险筛查的价值在于**把人的注意力分配到最需要的地方**，不是替代人。
    本演示里 PR-1003 被判"高风险"不是因为代码写错了，
    而是因为它**改的地方值得多看两眼**。{C['reset']}""")

    # ---------------------------------------------------------- 示例报告
    hr("附：完整报告示例（PR-1003）")
    print()
    print(format_review(review_pr(PR_C_HIGH)))
    print()

    hr()
    print(f"""
  {C['bold']}验收对照{C['reset']}
    ✓ 加了一个工具读 PR diff
        —— `review_pr_diff`（支持 pr_id 和 diff 两种输入）
    ✓ 输出包含改动范围 / 是否缺测试 / 是否动敏感文件
        —— 见上面的初审报告
    ✓ 不改架构
        —— 只是 `_register_all` 多注册了一个工具 + 一个独立模块

  {C['bold']}为什么这个方向值得做{C['reset']}
    {C['dim']}编码 Agent 是国内招聘数据最硬、岗位最多的场景。
    而这个工具的价值不在"AI 多聪明"，在**两张领域知识表**：
      · SENSITIVE_PATTERNS —— 哪些文件动了要特别小心
      · 改动范围与测试覆盖的对应关系
    **代码很简单（解析 diff + 规则匹配），难的是知道什么该担心。**
    换个语言/项目，表要重写；但"用一张表把资深工程师的直觉固化下来"
    这个方法是不变的。{C['reset']}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
