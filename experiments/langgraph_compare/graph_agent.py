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

================================================================================
★★ 第三个坑（2026-10-11 修）：节点拿不到外面的 store —— 生命周期被拆散了 ★★
================================================================================
第一版里 `recall_node` / `think_node` **各自调用 `setup_tools()`**，
于是每跑一轮回边就重建一次 MemoryStore + 工具集。

为什么会写成那样？因为**节点是独立可调用的单元**：
它不能假设"外面有个共享的 store" —— 那个假设会让节点无法单独测试，
也就丢掉了框架给的好处之一（④ 节点可单独调试）。

代价是**资源的生命周期不再集中**：手写版 `store` 建一次、`finally` 关一次；
框架版建 N 次，想关掉就得额外找地方收口。
第一版我用了一个**模块级全局收集器** `_OPEN_STORES` 来绕过 —— 那是绕过，不是解决：
  · 它**不线程安全**（两个线程同时跑会互相污染收集器）
  · 它引入了节点和"运行器"之间的**隐式耦合**（靠全局变量通信）
  · 生产里这么写会让 QPS 掉得很惨（每轮重建数据库连接）

★★ 正确做法：`config["configurable"]` —— 框架提供的依赖注入通道 ★★

    def recall_node(state, config):          # ← 节点声明第二个参数
        store = config["configurable"]["store"]   # ← 从 config 里取，不自己建

    app.invoke({...}, {"configurable": {"store": store, ...}})   # ← 运行时注入

**这才是框架的正确用法**，也是本次对照真正该展示的东西：
框架不是"要么全局变量、要么自己造轮子"，它**自带**依赖注入通道。
第一版之所以看着"生命周期更麻烦"，是因为我**没用它提供的机制**。

★ 这一条同时改变了 LOC 结论，见 `compare.py` 的 LOC 分解：
  第一版那 168 行里有相当一部分是**收口补丁**（收集器 + `_track` + 统一关闭），
  不是"编排声明开销"。**把补丁算进编排开销，会高估框架的声明成本。**
  本文件用 `# >>> lifecycle` / `# <<< lifecycle` 标出这部分，让 `compare.py`
  能把两类开销分开算 —— 否则这个数字靠人读代码判断，不可复现。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import END, START, StateGraph                 # noqa: E402
from langchain_core.runnables import RunnableConfig                 # noqa: E402

from nightwatch.models import Issue                                # noqa: E402

from shared import (                                               # noqa: E402
    MAX_STEPS, build_prompt, make_llm, parse_action, parse_final,
    recall_memory, setup_tools,
)


# =============================================================================
# ★ 依赖注入通道（本文件唯一的"资源生命周期"代码）
# =============================================================================
# >>> lifecycle: begin
#
# ★ 为什么这么短？因为**框架自带注入通道**，我们只需要读它。
#
#   第一版是"节点各自建 + 模块级收集器兜底"：
#       _OPEN_STORES = None                      ← 全局变量
#       def _track(store): _OPEN_STORES.append(store)
#       节点里：store, tools = setup_tools(); _track(store)
#       运行器里：created=[]; global _OPEN_STORES; _OPEN_STORES=created
#                 finally: for s in created: s.close()
#   —— 那一堆就是**收口补丁**，是"没用框架机制"的代价。
#
#   ⚠️ 注意：这里**没有**变成"零成本"。仍然有一层间接：
#     "这个节点的依赖从哪来"不再是语法上可见的，而是**约定**（key 名要对上）。
#     手写版靠闭包，语法上就看得见；这里靠 `config["configurable"]["store"]`
#     这个字符串约定 —— 写错 key 会在**运行时**才报 KeyError，不是静态错误。
#     这是依赖注入通道的真实代价，必须记下来，不能只说好处。

def _deps(config: RunnableConfig):
    """从框架的注入通道里取出本次运行共享的组件。

    ★ 用 `[]` 而不是 `.get()`：**缺依赖要立刻炸**。
      如果用 `.get()` 返回 None，错误会推迟到"调用 store 的某个方法时"，
      报错点离病因很远。**早失败 > 晚失败。**
    """
    ctx = config["configurable"]
    return ctx["store"], ctx["tools"], ctx["llm"]
# <<< lifecycle: end


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


def _add_int(old: int, new: int) -> int:
    """★ 第二个 reducer：**累加**而不是覆盖。

    这个 reducer 是给 `steps_executed` / `tokens` / `tool_calls` 这类
    **计数器**用的。

    ============================================================
    为什么这里必须显式写 reducer —— 一个真实踩到的坑
    ============================================================
    第一版我把 `tokens` / `tool_calls` 写成普通字段（默认语义 = 覆盖），
    然后在节点里手动做 `state.get("tokens", 0) + tokens`。

    **看起来对，其实脆弱**：
      · 一旦某个节点忘了写这个"读取旧值再加"的模板（比如新加的节点），
        它的返回值会**静默覆盖**掉之前累计的数 —— 不报错，只是数字变小。
      · 而"忘了写模板"这件事没有任何机制拦你（不像类型错了会报错）。

    **声明 reducer 的价值就在这里**：把"怎么合"从"每个节点各写一遍"
    变成"在状态定义里声明一次"。**写一次 > 抄十遍。**

    这也正好回应了对照实验的主题：
    手写版里 `result.tokens += x` 是**每处都要写对**；
    框架版里 reducer 是**一处声明，全场生效**。
    前者更直接，后者更不容易漏 —— 各有代价。
    """
    return (old or 0) + (new or 0)


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
    # 下面几个不是业务状态，是**为了对齐对照指标**额外加的
    #
    # ★ `steps_executed` 和后两个一样，是**计数器**而不是"当前步号" ——
    #   为什么不能直接用 `step`？见 `run_graph_agent` 的说明：
    #   `step` 的语义是"最后写入的步号"，在有回边的情况下
    #   它恰好等于实际步数（因为每次都 +1）... 但**只在图跑到底时成立**。
    #   一旦哪天真加了"跳过某步"的分支，两者就会分叉。
    #   **所以指标用的计数器必须独立声明**，不能搭业务字段的便车。
    steps_executed: Annotated[int, _add_int]
    tokens: Annotated[int, _add_int]
    tool_calls: Annotated[int, _add_int]
    terminated_by: str


# =============================================================================
# 节点（= 手写版循环体里的一个"阶段"）
# =============================================================================

def recall_node(state: GraphState, config: RunnableConfig) -> dict:
    """记忆召回节点。

    ★ 在手写版里，这是循环**之前**的两行代码。
      抽成节点后，它变成一个"有名字、有输入输出、可单独测试"的单元。

      **这是框架的一个隐性收益**：强制你把流程拆成有名字的阶段 ——
      而"给阶段起名字"这个动作本身会逼你想清楚每步在干嘛。
      （手写时很容易把三件事塞进一个函数里，因为没人拦你。）

    ★★ 组件从 `config` 注入，**不再自己建** ★★
      第一版这里是 `store, tools = setup_tools()` —— 每轮重建一次。
      改注入之后，`store` 的生命周期回到"一次运行一次"，
      和手写版对齐了 —— **这才是公平对照该有的样子**：
      两版在"资源怎么管"上不该有差异，否则测出来的耗时差
      里面混着"组件重建开销"，而那不是编排的开销。

    ★ 顺带一个观察：节点多了第二个参数（`config`）之后，
      **它的签名就不再是"纯 state → state 更新"了**。
      手写版的循环体没有这个"额外通道" —— 它直接闭包访问外层变量。
      **两种方式都能共享资源，区别是"显式约定"和"词法可见"。**
    """
    store, _tools, _llm = _deps(config)
    issue = state["issue"]
    memory, _ = recall_memory(issue, store)
    # ★ 返回的是"更新片段"而不是完整状态 —— 这是图的约定。
    #   计数器字段不写"读取旧值再加"（那样就是手写版的写法了），
    #   直接返回本节点产生的增量，由 reducer 去累加。
    return {"memory": memory, "step": 0}


def think_node(state: GraphState, config: RunnableConfig) -> dict:
    """思考节点：拼提示词 → 调模型 → 解析动作。

    ★ 这个名字来自 ReAct 的"Thought"。
      在手写版里它就是循环体那 5 行；
      在这里它是一个**可被单独调试的节点**（可以只跑它，看输入输出）。

    ★★ 注意这个节点**每轮回边都会执行** ★★
      所以第一版"节点内建组件"的写法在这里代价最大：
      走 3 步的样本会重建 3 次 MemoryStore（含 SQLite 连接 + 4 条记忆的
      向量化）。改注入后这些开销全部消失 —— 这正是本次重测耗时想分离的东西。
    """
    from dataclasses import dataclass

    store, tools, llm = _deps(config)

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

    messages = build_prompt(s)
    resp = llm.complete(messages, task="agent_step")
    text = resp.text
    tokens = getattr(resp, "total_tokens", 0) or 0

    action = parse_action(text)

    if action is None:
        # 模型给了最终答案 → 这个节点把结果写进 final，路由函数据此结束
        #
        # ★ 计数器只返回**本节点的增量**（`tokens` / `steps_executed`），
        #   由 reducer 负责累加。对比手写版里的 `result.tokens += tokens` ——
        #   那是"每处都要写对"，这里只写增量即可。
        return {
            "step": s.step,
            "steps_executed": 1,
            "final": parse_final(text),
            "tokens": tokens,
            "terminated_by": "final_answer",
        }

    # 有工具调用 → 执行并追加历史
    tr = tools.call(action["name"], action.get("args", {}))
    return {
        "step": s.step,
        "steps_executed": 1,
        "history": [{                        # ← reducer 会把它追加进去
            "action": action,
            "observation": (tr.data if tr.ok else f"ERROR: {tr.error}"),
            "ok": tr.ok,
        }],
        "tokens": tokens,
        "tool_calls": 1,
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

    # ★★ 组件在这里建**一次**，通过 config 注入给所有节点 ★★
    #
    # 第一版是"节点各自建 + 模块级收集器收口"，本函数要维护一个
    # `created` 列表、还要 `global _OPEN_STORES` 置入/清空 ——
    # 那段代码就是 LOC 里的"资源收口补丁"（见 `compare.py` 的 LOC 分解）。
    #
    # 现在改注入后：
    #   · 组件生命周期和手写版**对齐**（一次运行一次）—— 对照更公平
    #   · 耗时差里不再含"每轮重建组件"的开销 —— 数字才归因得了编排
    #   · 不需要全局变量 —— 线程安全
    #
    # ⚠️ 但要说清楚：**这不是"框架版凭空少了一堆麻烦"**，
    #    而是"我第一版没用框架提供的机制"。框架给了 `config.configurable`，
    #    用上它，麻烦就没了 —— 代价是要多学一个概念（依赖注入通道），
    #    而且这个通道是**约定式的**（key 名写错只能运行时发现）。
    store = None                                            # [lifecycle]
    try:
        # >>> lifecycle: begin
        # 建组件（一次）+ 注入（一次）
        store, tools = setup_tools()
        llm = make_llm(use_mock=use_mock)
        app = build_graph(with_checkpoint=with_checkpoint)
        cfg: dict = {"configurable": {"store": store, "tools": tools, "llm": llm}}
        if with_checkpoint:
            cfg["configurable"]["thread_id"] = f"thread-{issue.issue_id}"
        # <<< lifecycle: end
        final_state = app.invoke({"issue": issue, "history": []}, cfg)

        # ★ 注意 `final` 有两条来路：
        #   · 正常结束：think 节点写的
        #   · 兜底：fallback 节点写的（空列表）
        #   因为两条路都经过 fallback，所以这里直接读 final 即可。
        out.predicted = final_state.get("final", []) or []
        # ★ 用独立的计数器而不是 `step` —— 见 `AgentState` 里对
        #   `steps_executed` 的说明：`step` 是"当前步号"，
        #   它恰好等于步数只是巧合（每个节点都 +1），不该拿来当指标。
        out.steps = final_state.get("steps_executed", 0)
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
    finally:
        # >>> lifecycle: begin
        # ★ 只关一次 —— 因为只建了一次。
        #   （第一版这里要遍历 `created` 列表逐个关，因为建了 N 次。）
        if store is not None:
            try:
                store.close()
            except Exception:                               # noqa: BLE001
                pass
        # <<< lifecycle: end

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
