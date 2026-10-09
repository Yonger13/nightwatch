"""
LangGraph 对照实验 · 框架版

================================================================================
这个文件要展示的：LangGraph 把"手写版的三个控制流概念"翻译成了什么
================================================================================
手写版 `custom_agent.py` 里有三处**看得见**的控制流：

    ┌─ while 循环        →  代码里的一个语句
    ├─ state 字典        →  一个普通变量，改的时候你知道
    └─ if 判断终止       →  一个布尔表达式

LangGraph 把它们分别翻译成：

    ┌─ 图的回边（add_conditional_edges 指回自己）
    ├─ TypedDict + reducer（状态合并规则）
    └─ 路由函数（纯函数：state → 下一个节点名）

**这三处翻译就是本对照实验的全部内容。**

================================================================================
★ 先讲一个容易踩的坑：state 必须变成 TypedDict，而不是 dataclass
================================================================================
`shared.AgentState` 是个 dataclass，手写版直接用它。

但交给 LangGraph 时必须转成 `TypedDict` —— 因为框架对状态有两个**隐含要求**：

    ① **可合并**：每个节点返回的是"状态更新"，框架要把它 merge 进总状态。
       所以对同一个字段，多个节点的更新必须能定义"怎么合"（reducer）。
    ② **可序列化**：为了 checkpoint（把状态存下来以便恢复/中断续跑）。

**手写时你完全不会想到这两点** —— 因为你的 state 就是个内存对象，
改了就改了，不存在"合并"和"存盘"的问题。

**这不是框架找麻烦**，而是它提供的能力（中断续跑、并行分支）
对状态提出了要求。**想要那个能力，就得接受这个约束。**
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import END, START, StateGraph                 # noqa: E402
from langgraph.graph.message import add_messages                    # noqa: E402

from nightwatch.models import Issue                                # noqa: E402

from shared import (                                               # noqa: E402
    MAX_STEPS, build_prompt, make_llm, parse_action, parse_final,
    recall_memory, setup_tools,
)


def _append(old: list, new: list) -> list:
    """★ 自定义 reducer：**追加**而不是覆盖。

    ============================================================
    这是 LangGraph 暴露出来的、手写时看不见的一个概念
    ============================================================
    在手写版里，`state.history.append(x)` 是**就地修改** ——
    语义上就是"这份历史属于这个对象"。

    但在图里，每个节点返回的是**更新片段**而不是完整状态。
    框架要回答："两个节点都动了 history，怎么合？"

        默认行为：**覆盖**（后写的赢）
        这里需要：**追加**（保留全部）

    如果不声明这个 reducer，`history` 会被最后一个节点的返回值
    覆盖掉 —— 而**不会报错**，只会让你发现"历史怎么只有一条"。

    **这是对照实验里最值得记的一课**：
    手写时"状态怎么变"是隐含在代码顺序里的，
    框架里必须**显式声明**。显式的代价是啰嗦，收益是可推理。
    """
    return (old or []) + (new or [])


class GraphState(TypedDict, total=False):
    """图的状态定义。

    ★ `total=False` 表示字段都是可选的 —— 因为初始输入不需要提供全部字段。
    （手写版的 dataclass 用 `field(default_factory=...)` 达到同样目的，
      这是两种表达同一意图的方式。）
    """
    issue: Any                                        # 只读 → 普通声明
    memory: list                                      # 只写一次 → 普通声明
    history: Annotated[list, _append]                 # ★ 只追加 → 需要 reducer
    step: int                                         # 可覆盖 → 普通声明
    final: list                                       # 可覆盖 → 普通声明
    # 下面两个不是业务状态，是**为了对齐对照指标**额外加的
    tokens: int
    tool_calls: int
    terminated_by: str


# =============================================================================
# 节点（= 手写版循环体里的一个"阶段"）
# =============================================================================

def recall_node(state: GraphState) -> dict:
    """记忆召回节点。

    ★ 在手写版里，这是循环**之前**的两行代码。
      抽成节点后，它变成一个"有名字、有输入输出、可单独测试"的单元。

      **这是框架的一个隐性收益**：强制你把流程拆成有名字的阶段 ——
      而"给阶段起名字"这个动作本身会逼你想清楚每步在干嘛。
      （手写时很容易把三件事塞进一个函数里，因为没人拦你。）
    """
    issue = state["issue"]
    # 组件在这里现造（对照实验里不引入复杂的依赖注入）
    store, tools = setup_tools()
    memory, _ = recall_memory(issue, store)
    return {"memory": memory, "step": 0, "tokens": 0, "tool_calls": 0}


def think_node(state: GraphState) -> dict:
    """思考节点：拼提示词 → 调模型 → 解析动作。

    ★ 这个名字来自 ReAct 的"Thought"。
      在手写版里它就是循环体那 5 行；
      在这里它是一个**可被单独调试的节点**（可以只跑它，看输入输出）。
    """
    from dataclasses import dataclass

    # LangGraph 的 state 是 TypedDict，但要塞进 build_prompt，
    # 需要一个带属性的对象 —— 这个"翻译"动作本身就是一个小摩擦点。
    # （本可以改写 build_prompt 让它接受 dict，但那样两版就不共用了，
    #   对照实验的公平性优先。）
    @dataclass
    class _S:
        issue: Any
        memory: list
        history: list
        step: int

    s = _S(issue=state["issue"], memory=state.get("memory", []),
           history=state.get("history", []), step=state.get("step", 0) + 1)

    store, tools = setup_tools()
    llm = make_llm(use_mock=True)

    messages = build_prompt(s)
    resp = llm.complete(messages, task="agent_step")
    text = resp.text
    tokens = getattr(resp, "total_tokens", 0) or 0

    action = parse_action(text)

    if action is None:
        # 模型给了最终答案 → 这个节点把结果写进 final，路由函数据此结束
        return {
            "step": s.step,
            "final": parse_final(text),
            "tokens": state.get("tokens", 0) + tokens,
            "terminated_by": "final_answer",
        }

    # 有工具调用 → 执行并追加历史
    tr = tools.call(action["name"], action.get("args", {}))
    return {
        "step": s.step,
        "history": [{                        # ← reducer 会把它追加进去
            "action": action,
            "observation": (tr.data if tr.ok else f"ERROR: {tr.error}"),
            "ok": tr.ok,
        }],
        "tokens": state.get("tokens", 0) + tokens,
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


# =============================================================================
# 路由（= 手写版的 `if action is None: break`）
# =============================================================================

def route_after_think(state: GraphState) -> str:
    """条件边的路由函数。

    ★ 这是"终止条件"的框架表达。

    **它是纯函数**：输入 state，输出节点名。不产生副作用，不改状态。

    对比手写版的 `if action is None: break`：
        手写版：`break` 直接改变了控制流（命令式）
        框架版：返回一个字符串，由框架决定去哪（声明式）

    **声明式的好处**：路由逻辑可以被单独测试、可视化（画图时能看到边的条件）
    **声明式的代价**：多一层间接 —— "到底跳去哪"要顺着图定义看
    """
    if state.get("final"):
        return "end"
    if state.get("step", 0) >= MAX_STEPS:
        # ★ 兜底路径也必须在路由函数里显式处理。
        #   手写版的 `for-else` 是隐式的（语法糖），
        #   框架版必须写成一条明确的分支 —— **更啰嗦，但更明确。**
        return "end"
    return "continue"


def fallback_node(state: GraphState) -> dict:
    """兜底节点：只有在**没有答案时**才给一个空答案。

    ============================================================
    ★★ 这里踩了一个坑，正好印证了本文档反复强调的那个点 ★★
    ============================================================
    我最初的设计是"让所有路径都收口到 fallback"（看起来更整洁）：

        add_conditional_edges("think", route, {
            "continue": "think",
            "end": "fallback",       # ← 两条路都过 fallback
        })

    而 `fallback_node` 无条件返回 `{"final": []}`。

    **结果**：think 节点明明已经写了分类，走到 fallback 时**被空列表覆盖了** ——
    所以 LangGraph 版的预测永远是 `[]`，而手写版正常。

    跑出来的现象：两版**一致率只有 40%**，而且失败的 3 个都是
    "手写版有答案、框架版没有"。

    ============================================================
    为什么会这样？因为 `final` 没有 reducer，语义是**覆盖**
    ============================================================
    回到 `GraphState` 的定义：

        history: Annotated[list, _append]     ← 追加（显式声明了 reducer）
        final: list                            ← 覆盖（默认行为）

    `history` 我显式声明了 reducer，所以不会被覆盖；
    `final` 我没声明，于是**后写的赢** —— fallback 的空列表赢了。

    **这正是本章第 3.1 节讲的那个机制**：
    框架把"状态怎么变"从隐含变成显式。
    而"没显式声明"不等于"没有语义" —— 它有一个**默认语义（覆盖）**，
    只是你没看见。

    **手写版不会有这个问题**，因为：
        - 手写版没有"所有路径收口"这个结构（`break` 直接跳出去）
        - 就算有，`state.final = []` 这种覆盖在代码里**是看得见的**

    ============================================================
    修法：兜底节点必须先检查，不能无条件写
    ============================================================
    """
    # 已经有答案 → 什么都不做（返回空 dict 表示"无更新"）
    #
    # ★ 返回 `{}` 而不是 `{"final": state["final"]}`：
    #   后者虽然结果一样，但会让"我是主动保持"和"我碰巧写了同样的值"
    #   无法区分。**返回空 dict 的语义更清晰：这个节点没改任何东西。**
    if state.get("final"):
        return {}
    return {"terminated_by": "max_steps", "final": []}


# =============================================================================
# 建图
# =============================================================================

def build_graph(*, with_checkpoint: bool = False):
    """构造图。

    ★ 这里是整个对照实验的**核心差异点**：
      手写版的"循环"在这里是一个**回边**（think → think）。

    回边让"循环"从"代码结构"变成了"图结构" ——
    好处是它**可以被画出来**（`graph.get_graph().draw_mermaid()`），
    代价是读代码时看不到 `while` 了。

    `with_checkpoint=True` 时挂一个内存 checkpointer ——
    这演示了"框架多给的东西"：状态持久化 + 断点续跑。
    **手写版要实现这个，得自己写状态快照 + 恢复逻辑**（几十行）。
    """
    g = StateGraph(GraphState)

    g.add_node("recall", recall_node)
    g.add_node("think", think_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "recall")
    g.add_edge("recall", "think")

    # ★ 条件边：think 之后要么继续 think（形成循环），要么结束
    g.add_conditional_edges(
        "think",
        route_after_think,
        {
            "continue": "think",     # ← 这就是"循环"（回边指回自己）
            "end": "fallback",       # 兜底节点统一收口
        },
    )
    g.add_edge("fallback", END)

    if with_checkpoint:
        from langgraph.checkpoint.memory import MemorySaver
        return g.compile(checkpointer=MemorySaver())
    return g.compile()


# =============================================================================
# 对外入口（签名和手写版对齐，方便对比运行器统一调用）
# =============================================================================

@dataclass
class RunResult:
    """和手写版**同名同结构**的产出 —— 这样对比才对齐。"""
    issue_id: str
    predicted: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    elapsed_ms: int = 0
    terminated_by: str = ""
    error: str = ""


def run_graph_agent(issue: Issue, *, use_mock: bool = True,
                    verbose: bool = False, with_checkpoint: bool = False
                    ) -> RunResult:
    """跑一次 LangGraph 版。"""
    t0 = time.time()
    out = RunResult(issue_id=issue.issue_id)

    try:
        app = build_graph(with_checkpoint=with_checkpoint)
        cfg = {"configurable": {"thread_id": f"thread-{issue.issue_id}"}} \
            if with_checkpoint else {}
        final_state = app.invoke({"issue": issue, "history": []}, cfg)

        # ★ 注意 `final` 有两条来路：
        #   · 正常结束：think 节点写的
        #   · 兜底：fallback 节点写的（空列表）
        #   因为两条路都经过 fallback，所以这里直接读 final 即可。
        out.predicted = final_state.get("final", []) or []
        out.steps = final_state.get("step", 0)
        out.tool_calls = final_state.get("tool_calls", 0)
        out.tokens = final_state.get("tokens", 0)
        out.terminated_by = final_state.get("terminated_by", "") or (
            "max_steps" if not out.predicted else "final_answer")

        if verbose:
            print(f"    [graph] steps={out.steps} tools={out.tool_calls} "
                  f"→ {out.predicted}")

    except Exception as e:                                  # noqa: BLE001
        out.terminated_by = "error"
        out.error = f"{type(e).__name__}: {str(e)[:200]}"

    out.elapsed_ms = int((time.time() - t0) * 1000)
    return out


def render_mermaid(with_checkpoint: bool = False) -> str:
    """把图导出成 Mermaid 图 —— **这是框架白送的能力**。

    手写版要画出这个流程图，得自己维护一份"和代码同步"的图 ——
    而它一定会和代码不同步（因为没人拦着你改代码不改图）。
    """
    app = build_graph(with_checkpoint=with_checkpoint)
    try:
        return app.get_graph().draw_mermaid()
    except Exception as e:                                  # noqa: BLE001
        return f"（Mermaid 导出失败：{e}）"


if __name__ == "__main__":
    from shared import sample_issues
    for iss in sample_issues()[:2]:
        r = run_graph_agent(iss, verbose=True)
        print(f"  {r.issue_id}: {r.predicted}  steps={r.steps} "
              f"tools={r.tool_calls}  {r.elapsed_ms}ms ({r.terminated_by})")
    print()
    print(render_mermaid())
