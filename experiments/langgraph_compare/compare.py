#!/usr/bin/env python3
"""
LangGraph 对照实验 · 对比运行器

    python experiments/langgraph_compare/compare.py
    python experiments/langgraph_compare/compare.py --json out.json

==============================================================================
这个脚本要产出什么
==============================================================================
**不是"框架好还是手写好"的结论** —— 那种结论没有信息量。

它要产出的是**可验证的差异清单**：

  ① 功能等价性    两版的预测结果是不是一样？（如果不一样，对照就不成立）
  ② 编排代码量    "只算编排部分"的 LOC 差多少？（不含共用逻辑）
  ③ 运行时开销    框架版的额外开销是多少？
  ④ 框架多给了什么  可视化、checkpoint、可单独测试的节点 —— 逐条验证
  ⑤ 框架藏起了什么  路由的间接性、reducer 的必要性、状态的序列化约束

**⑤ 比 ④ 更重要** —— 因为"多给的东西"看文档就知道，
而"藏起来的东西"只有真写过才会遇到。

==============================================================================
★ 方法说明：为什么用 Mock LLM
==============================================================================
对照实验测的是**编排方式**，不是模型能力。

用真实模型的话，两版的输出差异会和"模型随机性"混在一起 ——
你会分不清"这次结果不同是因为图搭得不一样，还是因为采样"。

**控制变量：一次只测一个东西。**（这条在 `docs/05` 里出现过很多次）

MockLLM 是确定性的规则版，同样的输入必然给同样的输出 ——
所以**如果两版结果不一致，那一定是编排差异导致的**。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from shared import (                                               # noqa: E402
    EXPECTED_LABELS, MAX_STEPS, equivalence_valid, label_accuracy,
    sample_issues,
)

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def count_loc(path: Path) -> dict:
    """统计代码行数（区分总行/代码行/注释行/生命周期行）。

    ★ 为什么要区分"代码行"和"注释行"？
    因为"LOC"很容易被操纵 —— 多写注释就能让代码看起来更多。
    **只比"代码行"（不含注释和空行）。**
    （这个坑在技术圈很常见：拿总行数吹工程量的项目，一多半是注释。）

    ★★ 为什么还要再分出"生命周期行" ★★
    ============================================================================
    "框架版多 89%" 这个结论有个隐藏前提：**多出来的行都是"编排声明开销"**。
    但第一版里有一大块是**资源收口补丁**（模块级收集器 + `_track` +
    统一关闭循环）—— 那是我"没用框架的依赖注入机制"的代价，
    **不是框架本身的声明成本**。把两者混在一起算，会**高估框架**。

    所以源码里用两种标记把这类行标出来：
        `# [lifecycle]`                    → 单行标记
        `# >>> lifecycle: begin` ... `# <<< lifecycle: end`  → 区间标记
    **标记写在源码里**（而不是在这里写死行号）→ 代码一改，计数自动跟着变，
    不会出现"数字和代码对不上"。本项目在别处吃过这个亏
    （`docs/08` 里引用的行数会随代码演进过期，得全文搜索才能改全）。
    ============================================================================
    """
    total = code = comment = blank = lifecycle = 0
    in_docstring = False
    in_lifecycle = False
    for line in path.read_text(encoding="utf-8").splitlines():
        total += 1
        s = line.strip()

        # --- 生命周期标记（标记行本身算注释，不算代码）---
        if s.startswith("# >>> lifecycle: begin"):
            in_lifecycle = True
            comment += 1
            continue
        if s.startswith("# <<< lifecycle: end"):
            in_lifecycle = False
            comment += 1
            continue
        mark_single = s.endswith("# [lifecycle]")

        if not s:
            blank += 1
            continue
        n_triple = s.count('"""') + s.count("'''")
        if in_docstring:
            comment += 1
            if n_triple % 2 == 1:
                in_docstring = False
            continue
        if n_triple % 2 == 1 and (s.startswith('"""') or s.startswith("'''")):
            in_docstring = True
            comment += 1
            continue
        if s.startswith("#"):
            comment += 1
            continue
        code += 1
        if in_lifecycle or mark_single:
            lifecycle += 1
    return {"total": total, "code": code, "comment": comment, "blank": blank,
            # ★ orchestration = "只算编排声明"的行数（去掉生命周期收口）
            #   —— 这才是能归因给"框架 vs 手写"的那部分
            "lifecycle": lifecycle, "orchestration": code - lifecycle}


def _time_setup() -> float:
    """测一次 `setup_tools()` 的耗时（毫秒）。

    ★ 为什么需要它：旧版的耗时数据里混着"组件重建"的水分，
      要**量化**那部分是多少，才能解释"为什么旧数字不可信"。
      现场测而不是写死 —— 写死的数字会随代码演进过期。
    """
    from shared import setup_tools
    t0 = time.perf_counter()
    store, _tools = setup_tools()
    dt = (time.perf_counter() - t0) * 1000
    store.close()
    return dt


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="手写 vs LangGraph 对照实验")
    ap.add_argument("--json", help="结果 JSON 路径")
    ap.add_argument("--repeat", type=int, default=3,
                    help="重复次数（取中位数，抵消计时抖动）")
    args = ap.parse_args()

    print(f"\n{C['bold']}{C['cyan']}手写 ReAct vs LangGraph —— 对照实验{C['reset']}")
    hr()
    print(f"""
  对照对象    同一个 issue 分诊链路（召回 → 思考 → 调工具 → 循环 → 出结论）
  共用组件    提示词 / 动作解析 / 工具 / 记忆 —— 全在 shared.py，两版一字不差
  唯一变量    **编排方式**（while 循环 vs 图 + 条件边）
  模型        MockLLM（确定性）—— 排除模型随机性对对照的干扰
  样本        {len(sample_issues())} 个 issue × {args.repeat} 次重复
  循环上限    MAX_STEPS = {MAX_STEPS}
""")

    # ---------------------------------------------------------- 导入两版
    try:
        from custom_agent import run_custom_agent
    except Exception as e:                                  # noqa: BLE001
        print(f"  {C['red']}✗ 手写版导入失败：{e}{C['reset']}")
        return 1

    try:
        from graph_agent import render_mermaid, run_graph_agent
        langgraph_ok = True
    except ImportError as e:
        print(f"  {C['red']}✗ LangGraph 未安装（{e}）{C['reset']}")
        print(f"    安装：pip install langgraph")
        return 1

    issues = sample_issues()

    # ---------------------------------------------------------- ① 功能等价性
    hr("① 功能等价性（对照成立的前提）")

    custom_results, graph_results = [], []
    for _ in range(args.repeat):
        for iss in issues:
            custom_results.append(run_custom_agent(iss, use_mock=True))
            graph_results.append(run_graph_agent(iss, use_mock=True))

    # 对齐比较（按 issue_id 取最后一次）
    c_map = {r.issue_id: r for r in custom_results}
    g_map = {r.issue_id: r for r in graph_results}

    same = diff = 0
    diffs = []
    empty_both: list[str] = []          # ★ 两版都为空 —— 一致，但不是"有效一致"
    for iid, cr in c_map.items():
        gr = g_map.get(iid)
        if not gr:
            continue
        if sorted(cr.predicted) == sorted(gr.predicted):
            same += 1
            if not cr.predicted:
                empty_both.append(iid)
        else:
            diff += 1
            diffs.append((iid, cr.predicted, gr.predicted))

    print(f"\n  {'issue':<10}{'手写版':<26}{'LangGraph 版':<26}一致")
    print(f"  {'-' * 70}")
    for iid in c_map:
        cr, gr = c_map[iid], g_map.get(iid)
        if not gr:
            continue
        ok = sorted(cr.predicted) == sorted(gr.predicted)
        # ★ 三种状态，不是两种：
        #   ✓ 一致且非空   → 真正证明了等价
        #   ⚠ 一致但都空   → 可能是"两版一起失败"（证明力为 0）
        #   ✗ 不一致       → 编排差异真的影响了行为
        cc = C["green"] if (ok and cr.predicted) else (
            C["yellow"] if ok else C["red"])
        mark = "✓" if (ok and cr.predicted) else ("⚠" if ok else "✗")
        print(f"  {iid:<10}{str(cr.predicted):<26}{str(gr.predicted):<26}"
              f"{cc}{mark}{C['reset']}")

    eq_rate = same / (same + diff) if (same + diff) else 0
    valid = same - len(empty_both)

    # ---------------------------------------------------- ★ 准确率（新增）
    #
    # ★★ 为什么"一致率"不够 ★★
    #   一致率只回答"两版行为一不一样"，**不回答"它们对不对"**。
    #   两版一起错时，一致率照样 100% —— 而那时"等价性"这个结论是**空的**：
    #   它证明的只是"两版一起废掉了"。
    #   本项目在 docs/08"坑 2"记过这个陷阱（对称的错误最危险），
    #   这一段就是它的**机器化检查** —— 不再靠人记得去看。
    #
    # ★ 参照是 `shared.EXPECTED_LABELS` —— 硬编码的设计意图，
    #   且**放在被测对象之外**（见那份注释）。不能用两版里任何一方当参照，
    #   否则就是循环论证（和之前测试里"按被测阈值分组"是同一个坑）。
    acc_c = label_accuracy(c_map, "custom")
    acc_g = label_accuracy(g_map, "graph")

    print(f"\n  {C['bold']}准确率（对照设计意图 —— 一致率不区分"
          f"「两版都对」和「两版一起错」）{C['reset']}")
    print(f"\n  {'':<12}{'Top-1 正确':>11}{'Top-1':>8}{'严格':>8}   错的样本")
    print(f"  {'-' * 66}")
    for a in (acc_c, acc_g):
        nm = "手写版" if a["side"] == "custom" else "LangGraph 版"
        cc = C["green"] if a["rate"] == 1.0 else C["red"]
        print(f"  {nm:<12}{len(a['right']):>8}/{a['n']:<3}"
              f"{cc}{a['rate']:>7.0%}{C['reset']}"
              f"{a['strict_rate']:>8.0%}"
              f"   {', '.join(a['wrong']) if a['wrong'] else '—'}")
    print(f"  {C['dim']}Top-1 = 预测的第一个标签对不对（复用主项目 "
          f"`Metrics.top1_accuracy`，不重写）；严格 = 集合恰好相等。{C['reset']}")
    print(f"  {C['dim']}Top-1 低 → 主要分类没抓住；Top-1 高而严格低 → "
          f"多打了候选（可能是误匹配捞出来的假阳性）。{C['reset']}")

    # ★★ 最关键的一条告警 ★★
    #
    # 一致率 100% + 准确率 < 100% 是**最危险的组合**：
    # 它看起来"两版完全等价"，实际上"两版一起错" —— 等价性结论无效。
    # 而**准确率低但一致率高**这件事本身不会触发任何其它告警，
    # 所以必须在这里显式喊出来。
    acc_min = min(acc_c["rate"], acc_g["rate"])
    if eq_rate >= 1.0 and acc_min < 1.0:
        bad = sorted(set(acc_c["wrong"]) | set(acc_g["wrong"]))
        detail = "；".join(
            f"{i} 期望 {EXPECTED_LABELS.get(i)} 实得 "
            f"{c_map[i].predicted if i in c_map else '—'}"
            for i in bad)
        print(f"""
  {C['red']}{C['bold']}✗ 等价性结论无效 —— 一致率 100%，但准确率只有 {acc_min:.0%}{C['reset']}
    {C['red']}两版**一起错**在：{detail}{C['reset']}
    {C['dim']}一致率高只说明"两版行为一致"；准确率低说明"行为一致地不对"。
    这种情况下**不能**说"两版功能等价" —— 只能说"两版错得一样"。

    ★ 已知根因（三层，2026-10-11 定位；**修复另开一轮，本轮先让度量落地**）：
      ① `parse_final` 不认 Mock 输出里的 `predicted_labels` 字段
         → 它的"路径①"（从 JSON 取）**永远失效**，只剩兜底
      ② 兜底是**子串匹配** → `bug/timeout` 里含 "bug" 就被判成 `['Bug']`
         → **预测实际是个常数，不携带分类信息**
      ③ 主项目 `MockLLM._compose_answer` 的注释写"记忆优先"，
         实现却是"记忆兜底"（`if mem_tags and (not own_tags or
         own_tags == ["general"])`）→ **已由记忆验证过的 `Build`/`Performance`
         被自己现场猜的 `timeout`/`oom` 覆盖**；
         且所有标签被无条件加 `bug/` 前缀（`Build` → `bug/Build`）

      ①② 在实验侧，③ 在主项目。**实测：只修 ①② 准确率仍是 3/6**
      （因为 #901/#903 的正确答案本来就只能从记忆里来）；
      三层一起修才能到 6/6。这就是为什么本轮没顺手改掉 ——
      ③ 会改动主项目的推断行为，得单独评估对主项目评测的影响。{C['reset']}""")
    elif acc_min < 1.0:
        print(f"\n  {C['yellow']}⚠ 准确率未满（最低 {acc_min:.0%}）—— "
              f"但对等价性结论不构成致命影响（一致率本身也不满）{C['reset']}")

    print(f"\n  预测一致率   {eq_rate:.0%}（{same}/{same + diff}）")
    print(f"  有效一致率   {valid}/{same + diff}"
          f"（排除两版都为空的 {len(empty_both)} 个）")
    print(f"  准确率 Top-1 {C['bold']}手写 {acc_c['rate']:.0%} / "
          f"LangGraph {acc_g['rate']:.0%}{C['reset']}"
          f"　（设计意图参照，{acc_c['n']} 个样本）"
          f"{'' if acc_c['rate'] >= 1.0 else '  ← ⚠ 未满'}")

    # ------------------------------------------------ 防空转告警（两道）
    #
    # ★ 为什么必须单独统计"两版都空"
    #
    #   "一致率"只能证明"两版行为一样"，**不能证明"两版都在正常工作"**。
    #   本项目真的踩过：两版都解析不出动作 → 都返回空 → 一致率 100%，
    #   而实际上什么都没测到（见 docs/08"坑 1"）。
    #   所以"空"必须被单独看见，而不是混在 ✓ 里。
    if empty_both:
        print(f"""
  {C['yellow']}⚠ {len(empty_both)} 个样本两版都返回空：{', '.join(empty_both)}{C['reset']}
    {C['dim']}这些样本对等价性的证明力是 **0**。
    常见根因：字段名对不上（如模型输出 predicted_labels、解析只认 labels）、
    或标签体系不一致（模型输出 bug/timeout、解析只认 Bug/Docs/...）。
    —— 二者都属于"上游少一个约定，下游整个静默失效"。{C['reset']}""")

    # ★ 循环覆盖率：本实验对照的**就是循环编排**，
    #   所以必须先确认循环真的被执行过。
    #
    #   如果每次都在第 1 步就结束（比如记忆命中触发了"直接 finalize"的捷径），
    #   那回边和 fallback 分支**一次都没走到** ——
    #   两版代码量的差异里有一部分是**从未执行的代码**，
    #   结论（"框架版多了 67 行"）就成了无的放矢。
    step_dist: dict[int, int] = {}
    for r in custom_results:
        step_dist[r.steps] = step_dist.get(r.steps, 0) + 1
    multi = sum(v for k, v in step_dist.items() if k >= 2)
    total_runs = sum(step_dist.values())
    print(f"\n  循环覆盖     {multi}/{total_runs} 次运行走了 ≥2 步"
          f"（步数分布 {dict(sorted(step_dist.items()))}）")
    if multi == 0:
        print(f"""
  {C['yellow']}⚠ 没有任何运行走到第 2 步 —— 循环从未被执行{C['reset']}
    {C['dim']}修法：样本里至少要有一条**记忆不命中**的 issue。
    记忆一旦命中，MockLLM 会走"无需重复检索 → 直接 finalize"的捷径
    （见 MockLLM._decide_next_action），于是循环在第 1 步就结束了。{C['reset']}""")

    # ★ 判定用 `shared.equivalence_valid` —— 和测试、JSON 共用同一份实现。
    #   顺序上先报"更靠上游"的失效：覆盖不足 → 一起错 → 不一致。
    _covered = bool(not empty_both and multi > 0)
    if not _covered:
        print(f"  {C['yellow']}△ 两版结果一致，但**对照覆盖不充分** —— "
              f"先修上方告警，再谈等价性{C['reset']}")
    elif acc_min < 1.0:
        # ★★ 一致但都不对 —— 不能说"等价" ★★
        print(f"  {C['red']}✗ 两版一致，但**一起错**（准确率 {acc_min:.0%}）—— "
              f"不能叫「功能等价」{C['reset']}")
        print(f"    {C['dim']}等价性的前提是「行为正确」，"
              f"否则只是「错得一样」。先修准确率再谈等价。{C['reset']}")
    elif eq_rate < 1.0:
        print(f"""
  {C['yellow']}⚠ 两版结果不完全一致 —— 这本身是个发现{C['reset']}
    {C['dim']}可能的原因：路由函数的边界处理、兜底路径的差异、
      或 reducer 导致的 history 内容不同。
      **结果不一致说明编排差异真的影响了行为** ——
      这比"两版完全一样"更有信息量，但也意味着要逐个排查。{C['reset']}""")
    else:
        print(f"  {C['green']}✓ 两版功能等价 —— 对照成立（差异只在编排，不在行为）"
              f"{C['reset']}")

    # ---------------------------------------------------------- ② 代码量
    hr("② 编排代码量（只算代码行；再分出「资源收口」）")

    c_loc = count_loc(HERE / "custom_agent.py")
    g_loc = count_loc(HERE / "graph_agent.py")
    s_loc = count_loc(HERE / "shared.py")

    print(f"""
  {'文件':<20}{'总行':>8}{'代码行':>9}{'注释':>8}{'空行':>8}
  {'-' * 53}
  {'custom_agent.py':<20}{c_loc['total']:>8}{c_loc['code']:>9}{c_loc['comment']:>8}{c_loc['blank']:>8}
  {'graph_agent.py':<20}{g_loc['total']:>8}{g_loc['code']:>9}{g_loc['comment']:>8}{g_loc['blank']:>8}
  {'shared.py（共用）':<20}{s_loc['total']:>8}{s_loc['code']:>9}{s_loc['comment']:>8}{s_loc['blank']:>8}

  {C['dim']}★ 只比"代码行"：注释和空行不该算进工程量 ——
    否则"多写注释"就能让任何方案看起来更重。
    这个坑在技术圈很常见：拿总行数吹工程量的项目，一多半是注释。{C['reset']}

  {C['bold']}★ 再分一层：把"资源生命周期"的行单独摘出去{C['reset']}

    {'':<20}{'代码行':>9}{'其中生命周期':>14}{'编排声明':>10}
    {'-' * 53}
    {'custom_agent.py':<20}{c_loc['code']:>9}{c_loc['lifecycle']:>14}{c_loc['orchestration']:>10}
    {'graph_agent.py':<20}{g_loc['code']:>9}{g_loc['lifecycle']:>14}{g_loc['orchestration']:>10}

  {C['dim']}为什么必须分这一层：「框架版多 89%」有个隐藏前提 ——
    **多出来的行都是"编排声明开销"**。但第一版里有一大块是
    *资源收口补丁*（模块级收集器 + `_track` + 逐个关闭），
    那是我"没用框架的依赖注入机制"的代价，**不是框架的声明成本**。
    混在一起算会**高估框架**。

    源码里用 `# [lifecycle]` / `# >>> lifecycle: begin` 标记这类行 ——
    标记写在源码里，代码改了计数自动跟着变（不写死行号）。{C['reset']}

  {C['bold']}编排部分的净对比（两种口径都给）{C['reset']}
    {'口径':<26}{'手写':>7}{'LangGraph':>12}{'差':>9}
    {'-' * 56}
    {'全部代码行':<26}{c_loc['code']:>7}{g_loc['code']:>12}{g_loc['code'] - c_loc['code']:>+9d}
    {'扣掉生命周期后的"编排"':<26}{c_loc['orchestration']:>7}{g_loc['orchestration']:>12}{g_loc['orchestration'] - c_loc['orchestration']:>+9d}
""")

    _loc_delta = g_loc['code'] - c_loc['code']
    _orch_delta = g_loc['orchestration'] - c_loc['orchestration']
    if _loc_delta != _orch_delta:
        print(f"  {C['yellow']}★ 两个口径差 {_loc_delta - _orch_delta:+d} 行 —— "
              f"这就是「资源收口」对 LOC 结论的影响量。{C['reset']}")

    # ---------------------------------------------------------- ③ 运行时开销
    hr("③ 运行时的额外开销")

    c_ms = [r.elapsed_ms for r in custom_results if r.terminated_by != "error"]
    g_ms = [r.elapsed_ms for r in graph_results if r.terminated_by != "error"]

    if c_ms and g_ms:
        c_med, g_med = statistics.median(c_ms), statistics.median(g_ms)
        # ★★ 组件重建开销的量化（用来解释"旧数字为什么不可信"）★★
        #
        # 旧版 graph_agent 在**每个节点内**调 `setup_tools()`，
        # 于是一次运行重建 `1(recall) + N(think 步数)` 次。
        # 现在改成注入后只建 1 次 —— 两次测量差的就是这部分。
        #
        # 这里现场测一次 `setup_tools()` 的单价，用来把"旧数字里的水分"量化。
        # **不写死**（写死就会随代码演进过期 —— 本项目吃过这个亏）。
        _one = statistics.median([
            _time_setup() for _ in range(9)
        ])
        print(f"""
  手写版        中位 {c_med:>7.0f} ms　（{len(c_ms)} 次运行）
  LangGraph 版  中位 {g_med:>7.0f} ms　（{len(g_ms)} 次运行）
  额外开销      {g_med - c_med:+.0f} ms（{(g_med / max(c_med, 1) - 1):+.0%}）

  {C['green']}★ 这次两版**组件生命周期已对齐**（都是一次运行建一次）{C['reset']}
    {C['dim']}所以差值里**不再含"每轮重建组件"的开销** —— 可以归因给编排本身了。
    （这是 2026-10-11 的改造：graph 版从"节点内各自建 + 全局收集器收口"
      改成 `config["configurable"]` 依赖注入。）{C['reset']}

  {C['bold']}再扣掉"两版都要付的那一次组件构造"（更纯的编排对比）{C['reset']}
    组件构造（两版各一次）        ≈ {_one:.2f} ms
    手写版      总 {c_med:>4.1f} − {_one:.2f}  →  纯编排 ≈ {max(c_med - _one, 0):.1f} ms
    LangGraph   总 {g_med:>4.1f} − {_one:.2f}  →  纯编排 ≈ {g_med - _one:.1f} ms
    {'-' * 46}
    {C['bold']}纯编排差价  {g_med - c_med:+.1f} ms{C['reset']}　← 这才是"选哪种编排"的代价

    {C['dim']}★ 两版都要建一次组件（那是"跑这个任务"的成本，不是"选哪种编排"的成本）。
      扣掉它之后的**差价**与是否扣无关（两边减同一个数），
      但它让"框架版到底慢在哪"变得可读：
      手写版的总耗时几乎就是组件构造（纯编排≈0），
      而框架版在同样的组件构造之外还多花了几毫秒 —— 那几毫秒才是图的代价
      （channel 合并、reducer 调用、节点调度）。
    ★ 手写版那一格出现极小负值是因为**测量精度就是毫秒级** ——
      不假装更准，超过精度的位数直接按 0 报。{C['reset']}

  {C['dim']}旧的"手写 2ms / 框架 8ms"为什么不能信：
     那时框架版每个节点都建一次组件，一次运行重建 `1 + 步数` 次。
     实测单次 `setup_tools()` ≈ {_one:.2f} ms ——
     1 步的样本多花 ≈{_one * 2:.1f} ms，3 步的多花 ≈{_one * 4:.1f} ms。
     **旧差值 6ms 里有一半以上是"组件重建"，不是编排开销。**
     把它当成"框架慢 4 倍"是错的 —— 那是我自己的收口补丁慢。{C['reset']}""")

    # ---------------------------------------------------------- ④ 框架多给了什么
    hr("④ 框架多给了什么（逐条验证）")

    mermaid = render_mermaid()
    has_checkpoint = True
    try:
        from graph_agent import run_graph_agent as _rg
        _rg(issues[0], with_checkpoint=True)
    except Exception:                                       # noqa: BLE001
        has_checkpoint = False

    print(f"""
  {C['green']}① 流程图自动生成{C['reset']}
     手写版：**没有** —— 要画图得自己维护一份"和代码同步"的图，
             而它一定会不同步（因为没人拦着你改代码不改图）
     LangGraph：一行 `app.get_graph().draw_mermaid()`
     {C['dim']}（实测导出 {len(mermaid.splitlines())} 行 Mermaid）{C['reset']}

  {C['green']}② 状态持久化 + 断点续跑{C['reset']}
     手写版：要自己写状态快照 + 恢复逻辑（估计几十行）
     LangGraph：挂一个 checkpointer 就行
     {'✓ 实测可用' if has_checkpoint else '✗ 本次不可用'}

  {C['green']}③ 节点可单独调试{C['reset']}
     手写版：只能整个函数跑，中间状态靠打日志
     LangGraph：每个节点可以单独 invoke，输入输出都是结构化 state

  {C['green']}④ 路由逻辑可纯函数测试{C['reset']}
     手写版：`if action is None: break` 和循环体耦合，不好单测
     LangGraph：`route_after_think(state) -> str` 是纯函数，直接断言

  {C['green']}⑤ 依赖注入通道（`config["configurable"]`）{C['reset']}
     手写版：闭包天然共享外层变量，不需要"注入"这个概念
     LangGraph：节点拿不到外层局部变量，但框架**自带**注入通道 ——
               节点声明第二个参数 `config` 就能拿到运行时传进来的组件

     {C['dim']}★ 这条是 2026-10-11 改造时发现的：我第一版以为"框架让资源管理
       变麻烦"（节点各自建 → 得用全局收集器收口），
       其实是我**没用它提供的机制**。用上之后组件生命周期和手写版一样干净。{C['reset']}

  {C['dim']}注：④ 这一条是**框架强制**带来的 ——
    因为它不允许路由函数有副作用，所以它天然可测。{C['reset']}""")

    # ---------------------------------------------------------- ⑤ 框架藏起了什么
    hr("⑤ 框架藏起了什么（只有真写过才知道）")

    print(f"""
  {C['yellow']}① 循环不可见{C['reset']}
     手写版：`for step in range(MAX_STEPS)` —— 一眼看到"最多 4 次"
     LangGraph：循环是一条**回边**（think → think）。
     在几十个节点的大图里，你要顺着边才能找到循环 —— 而且**容易漏**。

  {C['yellow']}② 状态更新语义必须显式声明{C['reset']}
     手写版：`state.history.append(x)` —— 就地改，语义隐含在代码顺序里
     LangGraph：必须声明 reducer（`Annotated[list, _append]`）。
     **不声明的话会默认覆盖 —— 而且不报错**，
     只会让你发现"历史怎么只有一条"。

     {C['dim']}这是本次对照里最值得记的一条：
     框架把"状态怎么变"从**隐含**变成**显式**。
     代价是啰嗦，收益是可推理（也才可能做 checkpoint）。{C['reset']}

  {C['yellow']}③ 状态的序列化约束{C['reset']}
     手写版：state 可以是任意 Python 对象（含不可序列化的东西）
     LangGraph：想要 checkpoint 就必须**可序列化** ——
     所以 dataclass 要变 TypedDict，复杂对象要拆成基本类型。
     **这个约束在你用不上 checkpoint 时是纯负担。**

  {C['yellow']}④ 多一层间接{C['reset']}
     手写版：读代码 = 读流程（从上到下）
     LangGraph：读 **图定义 + 每个节点 + 每个路由函数** 三处才能拼出流程。
     {C['dim']}小图里这不算问题；节点超过 10 个时，"这步之后去哪"
       会变成一个需要来回翻代码才能回答的问题。{C['reset']}

  {C['yellow']}⑤ 注入通道是**约定式**的，不是词法可见的{C['reset']}
     手写版：`store` 是外层变量，闭包里**语法上就看得见** ——
             写错名字静态就能发现（IDE 能查、能跳转）
     LangGraph：节点里写 `config["configurable"]["store"]` ——
             key 名写错**编译期查不出来**，要到运行时才 `KeyError`

     {C['dim']}★ 这是"依赖注入"换来的代价，必须和它的好处一起说：
       好处是组件生命周期能收干净（见上方 ⑤），
       代价是**依赖关系从"语法"降级成"约定"**。
       缓解办法是把取值收进一个 `_deps(config)` 函数 ——
       只有一个地方拼 key，写错也只错一处（且缺依赖会**立刻炸**，
       不要用 `.get()` 返回 None 把错误推迟到调用现场）。{C['reset']}""")

    # ---------------------------------------------------------- ⑥ 结论
    hr("⑥ 结论（怎么选）")

    # ★ 结论里的"多/少"**由数据算出来**，不写死形容词。
    #
    #   为什么？实测数字是动态的（改流程、加节点都会变），
    #   而形容词如果是写死的，一旦哪次跑出来方向反过来，
    #   **文本会和数据自相矛盾，而且不报错** ——
    #   和 docs/05 里反复出现的"静默失效"是同一类问题。
    #
    #   （本项目真的犯过这个错：这段原本写死"图的代码量比手写少"，
    #     而实测是 146 vs 79 —— 多 67 行。
    #     **数字是对的，形容词是错的**，而且没人发现，
    #     因为它被原样抄进了 docs/08 的面试话术里。）
    _delta = g_loc['code'] - c_loc['code']
    _pct = abs(_delta) / max(c_loc['code'], 1) * 100
    if _delta > 0:
        _verb, _why = "多", ("声明开销（TypedDict + reducer + 节点边界）"
                             "超过了省下的循环控制")
    elif _delta < 0:
        _verb, _why = "少", "条件边省掉的循环控制多于状态声明的开销"
    else:
        _verb, _why = "持平", "省下的循环控制刚好抵消状态声明开销"
    loc_verdict = (f"图的代码量比手写{_verb} {abs(_delta)} 行"
                   f"（{g_loc['code']} vs {c_loc['code']}，{_pct:.0f}%）—— {_why}")

    print(f"""
  {C['bold']}不是"谁更好"，而是"代价换什么"：{C['reset']}

    ┌──────────────────┬────────────────────────┬────────────────────────┐
    │                  │ 手写                    │ LangGraph               │
    ├──────────────────┼────────────────────────┼────────────────────────┤
    │ 可读性（小流程）  │ {C['green']}好{C['reset']}（一条直线读到底）  │ 一般（要拼三处）        │
    │ 可读性（大流程）  │ 差（函数越堆越多）      │ {C['green']}好{C['reset']}（图结构清晰）    │
    │ 调试              │ 打日志                  │ {C['green']}节点级断点{C['reset']}          │
    │ 中断续跑          │ 自己写                  │ {C['green']}内置{C['reset']}                │
    │ 依赖              │ {C['green']}零{C['reset']}                    │ 一串（langchain-core…） │
    │ 状态语义          │ 隐含                    │ {C['green']}显式{C['reset']}                │
    │ 上手成本          │ 会 Python 就行          │ 要先理解图/状态/reducer │
    └──────────────────┴────────────────────────┴────────────────────────┘

  {C['bold']}我的判断（基于这次实写的体验）：{C['reset']}

    · **流程 < 5 步、分支少** → 手写。图的间接成本大于收益。
    · **流程 > 10 步、或者要"人审中断后继续"** → 框架。
      这是手写最难补的能力（不是不能写，是写完要自己维护）。
    · **原型阶段** → 框架（改结构快）。
    · **要抠性能 / 要极致可控** → 手写。

  {C['dim']}★ 注意这个结论里的"步骤数"是**实写之后**才敢说的 ——
    写之前我以为"框架总是更啰嗦"，实测：{loc_verdict}

    这就是为什么要做这个对照，而不是"看文档觉得"。
""")

    # ---------------------------------------------------------- 输出
    out = Path(args.json) if args.json else HERE / "compare_result.json"
    payload = {
        "n_issues": len(issues), "repeat": args.repeat,
        "max_steps": MAX_STEPS,
        "equivalence": {"same": same, "diff": diff, "rate": round(eq_rate, 4),
                        # ★ 这两个是"防空转"指标 —— 一致率高的同时也要看它们
                        "empty_both": empty_both,
                        "valid": valid,
                        "loop_coverage": {"multi_step_runs": multi,
                                          "total_runs": total_runs,
                                          "step_dist": dict(sorted(step_dist.items()))},
                        "mismatches": [{"issue": i, "custom": c, "graph": g}
                                       for i, c, g in diffs]},
        # ★★ 准确率：与"一致率"并排 —— 一致率高只说明两版行为一样，
        #    准确率才说明"行为对不对"。两者都满才能说"功能等价"。
        "accuracy": {
            "reference": "shared.EXPECTED_LABELS（硬编码设计意图，在被测对象之外）",
            "judgement": "Top-1（预测的第一个标签 == 期望；复用主项目 "
                         "nightwatch.eval.metrics.Metrics.top1_accuracy）",
            "strict_judgement": "辅助口径：预测集合 == 期望单标签",
            "custom": {k: v for k, v in acc_c.items() if k != "side"},
            "graph": {k: v for k, v in acc_g.items() if k != "side"},
            "equivalence_valid": equivalence_valid(
                eq_rate, acc_c["rate"], acc_g["rate"],
                covered=bool(not empty_both and multi > 0)),
            "expected_labels": EXPECTED_LABELS,
            "known_root_cause": [
                "parse_final 不认 Mock 的 predicted_labels 字段 → 路径①永远失效（实验侧）",
                "parse_final 退化成子串匹配 → 含 'bug' 即判 Bug，预测成常数（实验侧）",
                "MockLLM._compose_answer 注释写'记忆优先'、实现是'记忆兜底'，"
                "且所有标签被无条件加 bug/ 前缀（主项目）",
                "实测：只修前两层准确率仍是 3/6；三层一起修才到 6/6（修复另开一轮）",
            ],
        },
        "loc": {"custom": c_loc, "graph": g_loc, "shared": s_loc},
        "timing_ms": {"custom_median": statistics.median(c_ms) if c_ms else 0,
                      "graph_median": statistics.median(g_ms) if g_ms else 0,
                      # ★ 一次组件构造的单价 —— 两版都要付，扣掉它才是"纯编排"差价
                      "setup_once": round(_one, 3),
                      # ⚠️ 取 max(…, 0)：手写版总耗时几乎就等于组件构造，
                      #    相减会出现 -0.03 这种负值，那是**测量噪声**（精度只到毫秒），
                      #    不是"负耗时"。这里按 0 报，和报告里显示的一致。
                      "custom_orchestration_only":
                          round(max(c_med - _one, 0.0), 3) if c_ms else 0,
                      "graph_orchestration_only":
                          round(max(g_med - _one, 0.0), 3) if g_ms else 0,
                      "note": "两版组件生命周期已对齐（一次运行建一次），"
                              "故中位数之差可归因给编排；再扣掉一次组件构造"
                              "即得纯编排差价。毫秒级测量，负值按 0 报"},
        "framework_gives": ["流程图自动生成", "状态持久化+断点续跑",
                            "节点可单独调试", "路由逻辑可纯函数测试",
                            "依赖注入通道（config.configurable）—— "
                            "节点共享资源而不必用全局变量"],
        "framework_hides": ["循环不可见（回边）",
                            "状态更新语义必须显式声明（reducer）",
                            "状态的序列化约束",
                            "多一层间接（读流程要拼三处）",
                            "注入通道是约定式的（key 写错只能运行时发现）"],
        "caveats": [
            "MockLLM（确定性）—— 测编排不测模型能力",
            "组件生命周期已对齐（两版都是一次运行建一次组件，经 "
            "config.configurable 注入）—— 耗时差可归因给编排",
            "只覆盖 6 个 issue、单次链路 —— 不是生产级压测",
            "准确率参照 MockLLM 能表达的标签空间（受其关键词表限制）—— "
            "它守的是『实验在测正确的事』，不是『分诊效果好不好』",
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  {C['green']}✓{C['reset']} JSON → {out}")

    # Mermaid 也存一份（可以在 GitHub / 支持 Mermaid 的地方渲染）
    if mermaid and not mermaid.startswith("（"):
        (HERE / "graph.mmd").write_text(mermaid, encoding="utf-8")
        print(f"  {C['green']}✓{C['reset']} Mermaid → {HERE / 'graph.mmd'}")

    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
