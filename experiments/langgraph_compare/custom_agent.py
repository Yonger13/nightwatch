"""
LangGraph 对照实验 · 手写版基线

================================================================================
这个文件的唯一目的：给 LangGraph 版提供一个**公平的对照基线**
================================================================================
"公平"体现在三条硬约束上：

  ① **同一套组件** —— 复用主项目的 `ToolRegistry` / `LLMClient` / 记忆存储，
     不重新实现。否则对比就变成"比谁的工具写得好"，而不是"比循环怎么搭"。

  ② **同一套提示词和解析逻辑** —— 提示词模板、动作解析都是共享的
     （放在 `shared.py`），两版一字不差地用同一份。
     否则差异来自提示词，而不是来自编排方式。

  ③ **同样的能力边界** —— 手写版也要有最大步数、异常兜底、无工具调用即终止。
     不能"手写版偷懒省掉某功能，然后说框架版代码多"。

**如果做不到这三条，对照实验就是自欺欺人** —— 你会得到自己想要的结论。

================================================================================
★ 这个文件的写法是刻意的：**把控制流看得见的都写出来**
================================================================================
手写版的价值在于"每一步都在你眼前"。所以这里：

    · `while` 循环 —— 循环在代码里是个**语句**
    · `state` 字典 —— 状态是个**普通变量**，改它的时候你知道
    · `if` 判断终止 —— 终止条件是**一个布尔表达式**

这些在 LangGraph 里分别变成 `conditional_edge` / `TypedDict` / 路由函数。
**差异就藏在这三处，所以本文件刻意让它们最显眼。**
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 复用主项目（不重写）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.llm.client import LLMClient                      # noqa: E402
from nightwatch.memory.store import MemoryStore                  # noqa: E402
from nightwatch.models import Issue                              # noqa: E402
from nightwatch.tools.registry import ToolRegistry               # noqa: E402

from shared import (                                              # noqa: E402
    MAX_STEPS, AgentState, build_prompt, make_llm, parse_action,
    parse_final, recall_memory, setup_tools,
)


@dataclass
class RunResult:
    """一次运行的产出（两版共用这个结构，才能对齐比较）。"""
    issue_id: str
    predicted: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    elapsed_ms: int = 0
    terminated_by: str = ""      # "final_answer" / "max_steps" / "error"
    error: str = ""


def run_custom_agent(issue: Issue, *, use_mock: bool = True,
                     verbose: bool = False) -> RunResult:
    """手写 ReAct 循环。

    流程（和 LangGraph 版完全一致，只是表达方式不同）：

        ① 记忆召回（一次性）
        ② 循环：拼提示词 → 调模型 → 解析动作
             ├─ 有工具调用 → 执行 → 把观察追加进历史 → 回到 ②
             └─ 无工具调用 → 解析出最终答案 → 结束
        ③ 超过 MAX_STEPS → 兜底结束

    ============================================================
    ★ 对照点 1：循环怎么写
    ============================================================
    这里是一个 `for step in range(MAX_STEPS)`。

    **它有两个职责混在一起**：
      · 迭代控制（最多跑几次）
      · 状态推进（每轮往 history 里追加东西）

    在 LangGraph 里这两件事被拆开了：
      · 迭代由**图的回边**表达
      · 状态推进由**每个节点返回的 state 更新**表达

    谁更好？取决于你要不要"看清循环"。见 docs/08 的分析。

    ============================================================
    ★ 对照点 2：终止条件怎么表达
    ============================================================
    这里是两个 `return`：
      · 模型不再调工具 → 认为它给出了结论
      · 步数耗尽 → 兜底

    LangGraph 里第二个 `return` 会变成一个**条件边的路由函数返回值**。

    **注意**：路由函数是纯函数（输入 state，输出下一个节点名），
    而这里的 `return` 直接改变了控制流。这是"声明式 vs 命令式"的典型差别。
    """
    t0 = time.time()
    result = RunResult(issue_id=issue.issue_id)

    try:
        store, tools, llm = _setup(use_mock=use_mock)

        # ---------- ① 记忆召回 ----------
        # ★ 这一步在 LangGraph 里是一个节点（因为它是"图的一个阶段"）；
        #   在手写版里它就是循环前的两行代码。
        memory, mem_tokens = recall_memory(issue, store)
        result.tokens += mem_tokens

        # ---------- ② 循环 ----------
        state = AgentState(issue=issue, memory=memory, history=[])
        for step in range(MAX_STEPS):
            state.step = step + 1

            # 拼提示词（两版共用同一个函数）
            messages = build_prompt(state)

            # 调模型
            resp = llm.complete(messages, task="agent_step")
            result.tokens += getattr(resp, "total_tokens", 0) or 0
            text = resp.text

            # 解析动作
            action = parse_action(text)

            # ★ 终止条件之一：模型不再调工具
            if action is None:
                result.predicted = parse_final(text)
                result.terminated_by = "final_answer"
                break

            # 执行工具
            tr = tools.call(action["name"], action.get("args", {}))
            result.tool_calls += 1

            # 把观察追加进历史
            # ★ 这里就是"状态推进" —— 一个显式的 list.append
            state.history.append({
                "action": action,
                "observation": (tr.data if tr.ok else f"ERROR: {tr.error}"),
                "ok": tr.ok,
            })

            if verbose:
                print(f"    [手写] step {step + 1}: {action['name']} "
                      f"→ {'ok' if tr.ok else 'fail'}")
        else:
            # ★ for-else：循环跑满都没 break → 说明没得到最终答案
            #   这个语法糖很容易被忽略，但它正是"终止条件"的第三处表达
            result.terminated_by = "max_steps"
            # 兜底：用最后一条观察凑一个答案
            result.predicted = parse_final("")

    except Exception as e:                                  # noqa: BLE001
        result.terminated_by = "error"
        result.error = f"{type(e).__name__}: {str(e)[:200]}"

    result.steps = state.step if "state" in dir() else 0
    result.elapsed_ms = int((time.time() - t0) * 1000)
    return result


def _setup(use_mock: bool):
    """构造组件（两版共用，保证公平）。"""
    store, tools = setup_tools()
    llm = make_llm(use_mock=use_mock)
    return store, tools, llm


if __name__ == "__main__":
    # 独立跑一次（用于调试）
    from shared import sample_issues
    issues = sample_issues()
    for iss in issues[:2]:
        r = run_custom_agent(iss, use_mock=True, verbose=True)
        print(f"  {r.issue_id}: {r.predicted}  "
              f"steps={r.steps} tools={r.tool_calls}  "
              f"{r.elapsed_ms}ms  ({r.terminated_by})")
