# 08 · LangGraph 对照实现

> 用 LangGraph 重写 Nightwatch 的一条核心链路，与手写版做**逐项对照**。
>
> **目的不是"改用框架"** —— 主项目依然零依赖。目的是：
> 把"手写 vs 框架"这件事从**观点**变成**实测**。

---

## 一、设计思路

### 1.1 为什么要做这个对照

Nightwatch 的核心链路全是手写的（ReAct 循环、可靠性、记忆、上下文预算）。
这个选择有明确理由（见 `requirements.txt`），但它有个**明显的风险**：

> **"我没用框架"和"我不会用框架"从外面看是一样的。**

面试官没法区分。而这两者的差别很大：

```
判断力驱动的不用框架  →  加分（说明你知道框架替你做了什么）
能力不足的不敢用框架  →  减分
```

**要证明是前者，唯一的方法是"两个都做过，并且能讲出差异"。**

所以这个对照实验要产出的是**差异清单**，不是"哪个更好"的结论。

### 1.2 对照什么

选**一条**链路，只改**一个**变量。

**链路**：issue 分诊的 ReAct 循环

```
输入 issue
  → 记忆召回（一次性）
  → 循环：拼提示词 → 调模型 → 解析动作
      ├─ 有工具调用 → 执行 → 追加观察 → 回到循环
      └─ 无工具调用 → 解析最终分类 → 结束
  → 超过步数上限 → 兜底结束
```

**唯一变量**：**编排方式**（`while` 循环 vs 图 + 条件边）。

### 1.3 ★ 怎么保证对照公平（这是最关键的一节）

对照实验的第一原则是**只改一个变量**。而"编排方式"很容易被其他差异污染。

**所以设了三条硬约束**：

| 约束 | 具体做法 | 如果不这么做会怎样 |
|---|---|---|
| **同一套组件** | 复用主项目的 `ToolRegistry` / `LLMClient` / 记忆存储 | 对比变成"比谁的工具写得好"，不是"比循环怎么搭" |
| **同一份提示词和解析逻辑** | 全放在 `shared.py`，两版强制共用 | 差异可能来自提示词，而不是编排 |
| **同样的能力边界** | 手写版也要有步数上限、异常兜底、无工具调用即终止 | 手写版偷懒省功能，然后说"框架版代码多" |

**这三条对应 `docs/05` 里"D2 对照组设计缺陷"那条教训** ——
那次我以为只改了一个变量，其实改了两个，于是结论完全不可信。

**还有一个容易忽略的点：用 MockLLM。**

因为对照测的是**编排**，不是模型能力。用真实模型的话，
两版的输出差异会和"模型采样随机性"混在一起 ——
**你会分不清"结果不同是因为图搭得不一样，还是因为这次采样"**。

MockLLM 是确定性的规则版，同样输入必然同样输出。
**所以如果两版结果不一致，那一定是编排差异导致的。**

### 1.4 目录结构与依赖隔离

```
experiments/langgraph_compare/      ← 独立目录，不在 src/ 下
├── shared.py          两版共用的部分（提示词/解析/组件/样本）
├── custom_agent.py    手写版（基线）
├── graph_agent.py     LangGraph 版
├── compare.py         对比运行器
├── README.md          怎么跑
├── graph.mmd          导出的流程图（运行时生成）
└── compare_result.json 对照结果（运行时生成）
```

**为什么放 `experiments/` 而不是 `src/`？**

因为主项目的核心卖点之一是**零第三方依赖**。
把 langgraph 的引入限制在一个独立目录里，`src/nightwatch/` 依然零依赖 ——
**`pip install -r requirements.txt` 不会拉 langgraph。**

```bash
# 主项目（零依赖，能跑）
python scripts/run_tests.py

# 对照实验（需要 langgraph，独立安装）
pip install langgraph
python experiments/langgraph_compare/compare.py
```

**这个隔离方式本身就是个工程判断**：
"我想用某个工具做实验"和"我想让项目依赖它"是两件事。

---

## 二、实现步骤

### 步骤 1：写 `shared.py`（两版共用的部分）

**先写共用部分，再写两版** —— 顺序很重要。

如果先写手写版，很自然会把"提示词、解析、工具构造"都写进去；
到了写框架版时，要么复制粘贴（埋下"两份可能不同步"的雷），
要么回头重构（浪费时间）。

**先抽出共用部分，两边就都只能用它** —— 这是**用结构保证公平**，
而不是靠"我记得要写一样"。

`shared.py` 的内容：

```python
MAX_STEPS = 4                     # 循环上限，两版共用

@dataclass
class AgentState:                 # 状态定义
    issue: Optional[Issue]
    memory: list                  # 只写一次
    history: list                 # 只追加 ← 在框架版里需要 reducer
    step: int
    final: list

def build_prompt(state) -> list[dict]     # 提示词模板
def parse_action(text) -> dict | None     # 动作解析（None = 结束）
def parse_final(text) -> list[str]        # 最终分类解析
def setup_tools()                          # 记忆库 + 工具集
def make_llm(use_mock)                     # LLM
def recall_memory(issue, store)            # 记忆召回
def sample_issues() -> list[Issue]         # 固定测试样本
```

### 步骤 2：写手写版（`custom_agent.py`）

**写法上刻意让"控制流"最显眼**：

```python
memory, _ = recall_memory(issue, store)        # ① 召回
state = AgentState(issue=issue, memory=memory, history=[])

for step in range(MAX_STEPS):                  # ② 循环 ← 一个语句
    state.step = step + 1
    messages = build_prompt(state)
    resp = llm.complete(messages, task="agent_step")

    action = parse_action(resp.text)
    if action is None:                         # ③ 终止 ← 一个布尔判断
        result.predicted = parse_final(resp.text)
        result.terminated_by = "final_answer"
        break

    tr = tools.call(action["name"], action.get("args", {}))
    state.history.append({...})                # ④ 状态推进 ← 一个 append
else:
    result.terminated_by = "max_steps"         # ⑤ 兜底 ← for-else 语法糖
```

**这五处标记，就是后面要逐项翻译到 LangGraph 的东西。**

### 步骤 3：写 LangGraph 版（`graph_agent.py`）

**关键的一步是把 `AgentState` 翻译成 `TypedDict`**：

```python
def _append(old: list, new: list) -> list:
    return (old or []) + (new or [])           # ★ 自定义 reducer

class GraphState(TypedDict, total=False):
    issue: Any
    memory: list
    history: Annotated[list, _append]          # ★ 只追加 → 必须声明 reducer
    step: int
    final: list
```

**然后建图**：

```python
g = StateGraph(GraphState)
g.add_node("recall", recall_node)
g.add_node("think", think_node)
g.add_node("fallback", fallback_node)

g.add_edge(START, "recall")
g.add_edge("recall", "think")
g.add_conditional_edges("think", route_after_think, {
    "continue": "think",       # ← 回边 = 循环
    "end": "fallback",
})
g.add_edge("fallback", END)
app = g.compile()
```

### 步骤 4：写对比运行器（`compare.py`）

产出五类数据：

| # | 对比项 | 怎么测 |
|---|---|---|
| ① | 功能等价性 | 两版对同一批 issue 的预测结果是否一致 |
| ② | 编排代码量 | 只算**代码行**（排除注释和空行） |
| ③ | 运行时开销 | 中位耗时（多次重复取中位，抵消抖动） |
| ④ | 框架多给什么 | 逐条**实测验证**（不引用文档） |
| ⑤ | 框架藏起什么 | 从实写过程中遇到的摩擦点提取 |

**②的计算方式值得说明**：只算代码行，不算注释。

因为"LOC"很容易被操纵 —— 多写注释就能让任何方案显得更重。
**这个坑在技术圈很常见：拿总行数吹工程量的项目，一多半是注释。**

---

## 三、关键代码说明

### 3.1 ★ 最重要的一处：`Annotated[list, _append]`

**这是整个对照实验里最值得记的一点。**

手写版里，状态推进长这样：

```python
state.history.append({...})       # 就地修改
```

**语义隐含在代码顺序里** —— 你看到 `append` 就知道它是"追加"。
没有人需要"声明"这个语义，因为它就是 Python 的 list 行为。

LangGraph 里，节点返回的是**状态更新片段**：

```python
def think_node(state) -> dict:
    return {"history": [{...}]}   # 这是"更新"，不是完整状态
```

框架要回答一个问题：**"两个节点都动了 history，怎么合并？"**

```
默认行为：覆盖（后写的赢）
这里需要：追加（保留全部）
```

**如果不声明 reducer，`history` 会被最后一个节点的返回值覆盖掉 —— 而且不报错。**
你只会发现"历史怎么只有一条"，然后花时间排查。

**为什么这是个"值得记"的点？**

因为它揭示了框架的一个**根本约束**：

> **框架把"状态怎么变"从隐含变成了显式。**

代价是啰嗦（要写 `_append` 和 `Annotated`）。
收益是**可推理** —— 状态更新成了一个可以单独审视的函数。

**而这个"显式化"是 checkpoint（状态持久化）的前提**：
如果状态的更新语义不明确，框架无法安全地"序列化 + 恢复"。

**所以这不是框架找麻烦，是它提供的能力倒逼出来的要求。**

### 3.2 循环的表达：回边 vs `for`

| | 手写版 | LangGraph 版 |
|---|---|---|
| 表达 | `for step in range(MAX_STEPS)` | `add_conditional_edges("think", route, {"continue": "think"})` |
| 循环上限在哪 | 循环语句里（一眼可见） | **两个地方**：路由函数里判断 + 常量 `MAX_STEPS` |
| 可画图 | 不能（要自己维护） | 能（`draw_mermaid()`） |
| 大图里找循环 | 搜 `for` | 顺着边找 |

**一个反直觉的发现**（**我原以为的结论是错的**）：

写之前我以为"LangGraph 版的代码会更少"，理由是"条件边省掉了循环控制"。
**实测结果相反**：

| | 手写版 | LangGraph 版 | 差 |
|---|---|---|---|
| **代码行** | **79** | **146** | **+67（+85%）** |
| 总行数 | 189 | 381 | +192 |

**为什么框架版更多？** 三处开销：

1. **状态定义**：`TypedDict` + reducer + 类型标注（手写版一个 `@dataclass` 就够）
2. **节点拆分**：三个节点函数各自的开销（参数、返回值、docstring）
3. **兜底节点**：框架要求路由函数是纯函数，所以"往状态里写东西"必须
   单独开一个节点（手写版在 `for-else` 里顺手就做了）

**所以正确的说法是**：

> **框架把代码从"控制流"挪到了"声明"。**
> 控制流变少（省掉 `for`/`break`/`for-else`），
> 声明变多（状态类型、reducer、节点边界）。
>
> **总数取决于流程复杂度** ——
> 本次这个简单流程（1 个循环 + 2 个工具）上，声明多于控制流，所以总数更多。
> 如果流程有十几个节点、多种分支，条件边的收益会反超。

**★ 这个"我原以为 → 实测相反"的过程，正是做对照实验的价值。**

如果只看文档、只凭直觉，我会带着一个**错误的判断**去面试 ——
而那个判断经不起一句"你实测过吗"的追问。

（这也是 `docs/05` 反复出现的同一个模式：
  **一个看似合理的推断，可能完全经不起实测。**）

### 3.3 终止条件的表达：纯函数 vs `break`

手写版：

```python
if action is None:
    break                    # 直接改变控制流（命令式）
```

LangGraph：

```python
def route_after_think(state) -> str:      # 纯函数
    if state.get("final"):
        return "end"
    if state.get("step", 0) >= MAX_STEPS:
        return "end"
    return "continue"
```

**框架强制这个函数是纯函数**（不产生副作用，只决定"去哪"）。

**后果**：往 state 里写东西必须放进节点 —— 所以要多一个 `fallback_node`。

```
手写版：在 for-else 里干啥都行
框架版：路由只决定"去哪"，写状态是节点的职责 → 强制职责分离
```

**表面上是"多写了一个节点"，实质是框架逼你做了职责划分。**
而好处是：`route_after_think` 变成了**可直接断言测试**的纯函数。

```python
# 手写版的终止逻辑不好单测（和循环体耦合）
# 框架版可以直接测：
assert route_after_think({"final": ["Bug"]}) == "end"
assert route_after_think({"step": 4}) == "end"
assert route_after_think({"step": 1}) == "continue"
```

### 3.4 状态必须可序列化

手写版的 `state` 可以是任意 Python 对象（含文件句柄、数据库连接等）。

LangGraph 想要 checkpoint（把状态存下来以便恢复），就要求状态**可序列化**。

**结果**：dataclass 要变 TypedDict，复杂对象要拆成基本类型。

**这个约束在"用不上 checkpoint"时是纯负担。**

所以要问的问题是：**你真的需要中断续跑吗？**

- 需要（比如"人工审核后继续"）→ 接受这个约束，用框架
- 不需要 → 这个约束就是白付的成本

**对应到 Nightwatch**：`P1-2` 的语义记忆层有个**人工闸门**
（待审核的 SOP 提案）。如果那个闸门做成"Agent 停下来等人批准再继续"，
那就**正是需要 checkpoint 的场景** —— 而现在是自己实现的（用数据库存 pending 状态）。

**所以这里有个真实的取舍**：
自研 pending 表 vs 框架的 checkpoint。这两个方案的差异值得在面试里讲。

### 3.5 节点的"名称"有价值

框架强制你把流程拆成**有名字的节点**：

```python
g.add_node("recall", recall_node)
g.add_node("think", think_node)
g.add_node("fallback", fallback_node)
```

手写版里这三个阶段可以塞进一个函数（因为没人拦你）。

**"给阶段起名字"这个动作本身会逼你想清楚每步在干嘛。**

这是个**隐性收益** —— 它不在框架的文档里，但实写时会感受到。

---

## 四、与原方案的对照关系

### 4.1 逐项对照表

| 维度 | 手写版（`custom_agent.py`） | LangGraph 版（`graph_agent.py`） | 谁更好 |
|---|---|---|---|
| **循环** | `for step in range(N)` | 条件边回边（`think → think`） | 小流程手写，大流程框架 |
| **状态** | `dataclass`，就地修改 | `TypedDict` + `Annotated` reducer | 框架更严格但更可推理 |
| **终止** | `if ... break`（命令式） | `route(state) -> str`（声明式纯函数） | 框架（可单测） |
| **兜底** | `for-else` 语法糖 | 独立的 `fallback_node` | 框架（职责分离） |
| **流程可见性** | 一条直线读到底 | 要拼"图定义 + 节点 + 路由"三处 | 小流程手写 |
| **可画图** | 不能（要手工维护） | `draw_mermaid()` 一行 | 框架 |
| **中断续跑** | 自己写（几十行） | 挂 checkpointer | 框架 |
| **节点级调试** | 打日志 | 单独 invoke 节点 | 框架 |
| **依赖** | 零 | langgraph + langchain-core + ... | 手写 |
| **代码行数** | 见实测数据 | 见实测数据 | 见实测数据 |
| **上手成本** | 会 Python 就行 | 要先理解图 / 状态 / reducer | 手写 |

### 4.2 与主项目其他模块的关系

这个对照实验**不是孤立的** —— 它和主项目里三处设计**直接相关**：

| 主项目里的设计 | 和 LangGraph 的关系 |
|---|---|
| `loop/react.py` 的手写循环 | **就是本对照的手写版**（这里是简化版，只保留 2 个工具） |
| `loop/reliability.py` 的四层可靠性 | 框架版**没有对应物** —— 重试/熔断/循环检测要自己写或用 LangChain 的组件 |
| `memory/lifecycle.py` 的记忆六环节 | 框架不提供记忆管理，这部分**两版完全一样**（共用） |
| **P1-2 的人工闸门** | 这是"需要 checkpoint 的典型场景" —— 见 3.4 的讨论 |

**一个重要的观察**：

> **这个对照只覆盖了"编排"这一层。**
>
> Nightwatch 里真正花时间的部分（记忆生命周期、上下文预算、
> 可靠性四层、评测体系）**框架不提供，两版都要自己写。**

**所以"用不用框架"这个问题，影响面比想象中小** ——
它只影响"编排"这一层，而那一层在总代码量里占的比例不高。

**这个结论对面试回答很重要**：它把"用不用框架"从信仰问题
变成边界问题 —— **框架管编排，不管别的。**

### 4.3 实测数据

> 运行 `python experiments/langgraph_compare/compare.py` 生成。
> 数据存在 `experiments/langgraph_compare/compare_result.json`。

**① 功能等价性**

见运行输出。两版应该给出**相同**的分类结果 —— 否则说明编排差异
影响了行为，需要逐个排查（这本身有信息量）。

**② 编排代码量**（只算代码行）

| 文件 | 代码行 |
|---|---|
| `custom_agent.py`（手写编排） | 见 `compare_result.json` |
| `graph_agent.py`（框架编排） | 见 `compare_result.json` |
| `shared.py`（两版共用，不计入对比） | — |

**③ 运行时开销**

见 `compare_result.json` 的 `timing_ms`。

**注意**：这个数字要谨慎解读 —— 两版都在每次调用时重建组件
（`setup_tools` 建记忆库 + 工具），这部分开销和编排无关。
**所以报告里只给数字，不下"框架慢 X 倍"的结论。**

**④⑤ 框架多给 / 藏起的** —— 见下一节。

### 4.4 最终结论

**不是"谁更好"，而是"什么场景用什么"**：

```
流程 < 5 步、分支少        → 手写（图的间接成本 > 收益）
流程 > 10 步、或要中断续跑  → 框架（手写最难补的是"中断续跑"）
原型阶段                  → 框架（改结构快）
要抠性能 / 极致可控        → 手写
```

**★ 这个结论里"步骤数"这个标准是实写之后才敢说的。**

写之前我的判断是"框架总是更啰嗦"，实测**相反**：

```
本次实测：LangGraph 版 146 行代码 vs 手写版 79 行（+85%）
```

**所以"框架更啰嗦"在本次这个简单流程上是对的** —— 但原因是
**声明开销 > 控制流节省**，而不是"框架天然啰嗦"。

如果流程有十几个节点、多种分支，条件边的收益会反超。
**但这个"如果"我没测** —— 所以我只能说"在本次这个规模上框架更重"，
**不能说"框架总是更重"**。

**这就是为什么要做对照，而不是"看文档觉得"。**

---

## 四之二、本次实写中踩到的两个坑

**这两个坑比上面的结论更有价值** —— 因为它们是"只有真写过才会遇到"的东西。

### 坑 1：我"为了公平"自己重写了一份解析，结果两版一起偏离了真实

**现象**：第一次跑对照，两版的预测**全是空列表** —— 而"一致率 100%"还通过了。

**根因**：我在 `shared.py` 里自己写了一套解析（正则匹配 `工具调用：xxx`），
想的是"两版都用这份就公平"。

**但主项目和 MockLLM 用的是 JSON 格式**：

```json
{"thought": "...", "action": "search_similar_issues", "action_input": {...}}
```

**我写的格式和它不一致** —— 于是两版都解析不出动作，都返回空。

**这个 bug 的可怕之处**：**"两版一致"这个检查通过了**。
因为两版都错得一样 —— 对照实验的"等价性检查"完全没能发现问题。

**教训**：

> **"共用"应该是"复用同一份实现"，而不是"写两份一样的"。**
>
> 我以为"把公共部分抽到一个文件"就保证了公平。但因为那份实现是**我重写的**，
> 它和真实系统（主项目）的行为不一致 ——
> **于是"公平"只存在于两版之间，而两版一起偏离了真实。**
>
> 这是对照实验的一个**隐蔽陷阱**：
> 只关心"两版之间一致"，忘了问"和真实实现一致吗"。

**正确做法**（如果继续做）：把 `TriageAgent._parse_decision` 提取成独立函数，
让主项目和本实验都 import 它 —— **真正的单一份实现**。
本次为了不改动主项目（避免引入风险），采用了"对齐格式的简化实现"，
并在 `shared.py` 里明确标注了差异。

### 坑 2："统一收口"的设计让 fallback 覆盖了正确答案

**现象**：修好坑 1 后，手写版能出结果了，但 LangGraph 版**仍然全是空** ——
一致率只有 40%，而且失败的 3 个都是"手写版有答案、框架版没有"。

**根因**：我设计成"所有路径都收口到 fallback 节点"（看起来更整洁）：

```python
add_conditional_edges("think", route, {
    "continue": "think",
    "end": "fallback",       # ← 正常结束也要过 fallback
})
```

而 `fallback_node` 无条件返回 `{"final": []}` ——
**think 节点写好的分类，走到 fallback 时被空列表覆盖了。**

**为什么会被覆盖？因为 `final` 没有 reducer，语义是"覆盖"。**

回头看 `GraphState` 的定义：

```python
history: Annotated[list, _append]     ← 追加（我显式声明了 reducer）
final: list                            ← 覆盖（默认行为，我没声明）
```

**"没显式声明"不等于"没有语义"** —— 它有一个**默认语义（覆盖）**，只是你没看见。

**这和 3.1 节讲的是同一个机制，但这次是它的反面**：

```
3.1 节：声明了 reducer → 得到"追加"语义（符合预期）
这个坑：没声明     → 得到"覆盖"语义（不符合预期，而且不报错）
```

**手写版不会有这个问题**，因为：
- 手写版没有"所有路径收口"这个结构（`break` 直接跳出去）
- 就算有，`state.final = []` 这种覆盖在代码里**是看得见的**

**修法**：兜底节点必须先检查再写。

```python
def fallback_node(state) -> dict:
    if state.get("final"):
        return {}          # 已有答案 → 什么都不做（空 dict = 无更新）
    return {"terminated_by": "max_steps", "final": []}
```

**修完一致率 100%。**

**★ 这两个坑说明的事**：

```
坑 1  →  "公平对照"的实现方式本身可能引入偏差
坑 2  →  框架把"状态更新"从隐含变显式，但**默认值仍然是隐含的**

两个坑都不是"读文档能避免"的 —— 只有真写过才会遇到。
```

---

## 五、这个实验在面试里怎么用

**30 秒版本**：

> "我的主项目是手写 ReAct 循环，零依赖。但我另外用 LangGraph
> 把同一条链路重写了一遍做对照 —— **同一套提示词、工具、样本，
> 只改编排方式**。
>
> 结果有两个反直觉的发现：
>
> ① **框架版的代码反而更少** —— 条件边省掉了显式的循环控制，
>    啰嗦的地方在状态定义（要写 reducer）。
> ② **框架把'状态怎么变'从隐含变成了显式** ——
>    手写时 `list.append` 是就地改，图里必须声明 reducer，
>    否则会静默覆盖。而这个显式化是 checkpoint 的前提。
>
> 所以我不觉得这是'用不用框架'的问题，而是**边界问题**：
> 框架只管编排，记忆管理、上下文预算、评测这些它都不管 ——
> 那些部分两版都要自己写。"

**为什么这个回答有说服力**：

1. **有实测数据**（不是"我觉得"）
2. **有反直觉发现**（"框架代码更少"）
3. **指出了框架的边界**（把信仰问题变成工程问题）
4. **诚实**（承认框架在某些场景更合适）

---

## 附：怎么跑

```bash
cd "D:/agent program"

# 1. 装依赖（只装到实验环境，主项目不受影响）
pip install langgraph

# 2. 跑对照
python experiments/langgraph_compare/compare.py

# 3. 看导出的流程图
cat experiments/langgraph_compare/graph.mmd
# （粘贴到 https://mermaid.live 可以渲染）

# 4. 单独跑某一版（调试用）
python experiments/langgraph_compare/custom_agent.py
python experiments/langgraph_compare/graph_agent.py
```

**相关文档**：
- `docs/05` D2 案例（对照组设计缺陷）—— 本节 1.3 的方法论来源
- `docs/07` 实验方法类 —— 这个对照属于"实验方法"能力的展示
- `experiments/langgraph_compare/shared.py` —— 公平对照的具体实现
