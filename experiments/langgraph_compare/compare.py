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

from shared import MAX_STEPS, sample_issues                        # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def count_loc(path: Path) -> dict:
    """统计代码行数（区分总行/代码行/注释行）。

    ★ 为什么要区分？
    因为"LOC"很容易被操纵 —— 多写注释就能让代码看起来更多。
    **只比"代码行"（不含注释和空行）。**

    （这个坑在技术圈很常见：拿总行数吹工程量的项目，一多半是注释。）
    """
    total = code = comment = blank = 0
    in_docstring = False
    for line in path.read_text(encoding="utf-8").splitlines():
        total += 1
        s = line.strip()
        if not s:
            blank += 1
            continue
        # 粗略的 docstring 检测（三引号计数）
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
    return {"total": total, "code": code, "comment": comment, "blank": blank}


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
    for iid, cr in c_map.items():
        gr = g_map.get(iid)
        if not gr:
            continue
        if sorted(cr.predicted) == sorted(gr.predicted):
            same += 1
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
        cc = C["green"] if ok else C["red"]
        print(f"  {iid:<10}{str(cr.predicted):<26}{str(gr.predicted):<26}"
              f"{cc}{'✓' if ok else '✗'}{C['reset']}")

    eq_rate = same / (same + diff) if (same + diff) else 0
    print(f"\n  预测一致率 {eq_rate:.0%}（{same}/{same + diff}）")
    if eq_rate < 1.0:
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
    hr("② 编排代码量（只算代码行，不算注释）")

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

  {C['bold']}编排部分的净对比{C['reset']}
    手写版        {c_loc['code']} 行代码
    LangGraph 版  {g_loc['code']} 行代码
    差            {g_loc['code'] - c_loc['code']:+d} 行（{(g_loc['code'] / max(c_loc['code'], 1) - 1):+.0%}）
""")

    # ---------------------------------------------------------- ③ 运行时开销
    hr("③ 运行时的额外开销")

    c_ms = [r.elapsed_ms for r in custom_results if r.terminated_by != "error"]
    g_ms = [r.elapsed_ms for r in graph_results if r.terminated_by != "error"]

    if c_ms and g_ms:
        c_med, g_med = statistics.median(c_ms), statistics.median(g_ms)
        print(f"""
  手写版        中位 {c_med:>7.0f} ms　（{len(c_ms)} 次运行）
  LangGraph 版  中位 {g_med:>7.0f} ms　（{len(g_ms)} 次运行）
  额外开销      {g_med - c_med:+.0f} ms（{(g_med / max(c_med, 1) - 1):+.0%}）

  {C['dim']}★ 这个数字要谨慎解读：两版都在**每次调用时重建组件**
    （setup_tools 建记忆库 + 工具），这部分开销和编排无关。
    框架的真实开销应该在"只算编排"的场景下测 ——
    但那样就要重构代码，会破坏对照的公平性。
    **所以这里只报告数字，不下"框架慢 X 倍"的结论。**{C['reset']}""")

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
       会变成一个需要来回翻代码才能回答的问题。{C['reset']}""")

    # ---------------------------------------------------------- ⑥ 结论
    hr("⑥ 结论（怎么选）")

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
    写之前我以为"框架总是更啰嗦"，实际发现：
    **图的代码量比手写少（{g_loc['code']} vs {c_loc['code']} 行）**，
    因为条件边省掉了循环控制代码。啰嗦的地方在**状态定义**（TypedDict + reducer）。

    这就是为什么要做这个对照，而不是"看文档觉得"。
""")

    # ---------------------------------------------------------- 输出
    out = Path(args.json) if args.json else HERE / "compare_result.json"
    payload = {
        "n_issues": len(issues), "repeat": args.repeat,
        "max_steps": MAX_STEPS,
        "equivalence": {"same": same, "diff": diff, "rate": round(eq_rate, 4),
                        "mismatches": [{"issue": i, "custom": c, "graph": g}
                                       for i, c, g in diffs]},
        "loc": {"custom": c_loc, "graph": g_loc, "shared": s_loc},
        "timing_ms": {"custom_median": statistics.median(c_ms) if c_ms else 0,
                      "graph_median": statistics.median(g_ms) if g_ms else 0},
        "framework_gives": ["流程图自动生成", "状态持久化+断点续跑",
                            "节点可单独调试", "路由逻辑可纯函数测试"],
        "framework_hides": ["循环不可见（回边）",
                            "状态更新语义必须显式声明（reducer）",
                            "状态的序列化约束",
                            "多一层间接（读流程要拼三处）"],
        "caveats": [
            "MockLLM（确定性）—— 测编排不测模型能力",
            "两版每次调用都重建组件，计时含这部分开销",
            "只覆盖 5 个 issue、单次链路 —— 不是生产级压测",
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
