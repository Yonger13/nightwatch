"""
LangGraph 对照实验 · 两版共用的部分

================================================================================
★ 为什么要有这个文件（对照实验的方法论核心）
================================================================================
对照实验的第一原则是**只改一个变量**。

如果手写版和 LangGraph 版各自写一套提示词、一套动作解析、
一套工具构造，那最后测出的差异**无法归因** ——
你不知道差异来自"编排方式"，还是来自"提示词写得好不好"。

所以凡是**和编排无关**的东西，全部放这里，两版强制共用：

    提示词模板      build_prompt()
    动作解析        parse_action() / parse_final()
    组件构造        setup_tools() / make_llm()
    记忆召回        recall_memory()
    状态定义        AgentState

**只留"编排"这一件事给两版各自实现。** 这是本对照实验成立的前提。

（这个做法和 `docs/05` 里"D2 对照组设计缺陷"那条教训是同一个：
  那次我以为只改了一个变量，其实改了两个，于是结论完全不可信。）

================================================================================
`AgentState` 的设计说明
================================================================================
这里的 `AgentState` 是个普通 dataclass，两版都用它。

**但用法完全不同**：
  · 手写版：直接改这个对象（`state.history.append(...)`）
  · LangGraph 版：把它包成 `TypedDict` 交给图，靠**节点返回值**更新

所以这个类在 LangGraph 版里会被"翻译"成 TypedDict ——
**翻译本身就是对照要观察的点之一**：
"一个 dataclass 变成 TypedDict + reducer" 这个动作，
暴露了框架对状态的两个隐含要求（可合并、可序列化）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.llm.client import build_llm                      # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore   # noqa: E402
from nightwatch.models import Issue, IssueState, MemoryRecord, Outcome  # noqa: E402
from nightwatch.tools.registry import IssueRepository, ToolRegistry     # noqa: E402

# 循环上限 —— 两版共用同一个值
MAX_STEPS = 4


@dataclass
class AgentState:
    """Agent 的运行时状态。

    ★ 注意字段的分工，它决定了状态管理的复杂度：

        issue     只读（整个流程不变）          → 不需要 reducer
        memory    只写一次（召回后不变）        → 不需要 reducer
        history   只追加（每轮 append）         → **需要 reducer**
        step      可覆盖（每轮 +1）             → 需要 reducer
        final     后写覆盖（只有最后一步写）    → 需要 reducer

    **"只追加"和"可覆盖"是两种不同的更新语义** ——
    手写版里这个区别不重要（都是赋值），
    但在 LangGraph 里必须显式声明（`Annotated[list, add]` vs 直接覆盖）。

    **这就是框架暴露出来的、手写时你看不到的一个概念。**
    """
    issue: Optional[Issue] = None
    memory: list[tuple[MemoryRecord, float]] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    step: int = 0
    final: list[str] = field(default_factory=list)


# =============================================================================
# 提示词模板（两版共用）
# =============================================================================
# ★ 关于这个模板的格式约束
#
# MockLLM 是**规则版**：它靠"在上下文里搜字符串"来判断该做什么。
# 它检查这几个标记：
#
#     "search_similar_issues("   → 已经调过这个工具了
#     "search_docs("             → 同上
#     "[语义相似]"               → 上下文里已经有语义相似的记忆了
#     "[同一提交者的历史]"        → 同上
#
# **所以 prompt 里必须真的出现这些字符串** —— 否则 Mock 会认为
# "还没有记忆、还没调过工具"，于是一轮轮地重复调度，
# 直到步数耗尽。
#
# 这也是"Mock 能验链路"的**前提条件**：你得让它的规则能识别上下文。
# （主项目的 `react.py` 里就是这么拼提示词的 —— 这里是对齐它，
#   而不是另发明一套格式。）

PROMPT_HEADER = """你是开源项目的 issue 分诊助手。

【当前 issue】
标题：{title}
正文：{body}

【历史相似问题】
{memory}

【可选标签】
{labels}

【已经做过的操作】
{history}

请决定下一步动作，**只输出一个 JSON**，不要有其它文字：

如果信息已经足够，输出：
{{"thought": "信息足够，给出分类", "action": "finalize",
 "action_input": {{"labels": ["Bug"]}}}}

如果需要查信息，输出（只列可用的工具）：
{{"thought": "需要先检索历史", "action": "search_similar_issues",
 "action_input": {{"query": "症状关键词", "top_k": 5}}}}"""


def build_prompt(state: AgentState) -> list[dict]:
    """拼提示词。

    ★ 这个函数在 LangGraph 版里会被**包进一个节点**；
      在手写版里它就是循环体内的一行调用。

      **功能完全相同，差异只在"它挂在哪个位置"** ——
      这正是我们要观察的。
    """
    issue = state.issue
    labels = ["Bug", "Docs", "Enhancement", "Performance", "Build"]

    # ★ 有记忆时必须打 `[语义相似]` 标记 —— 见上面注释，
    #   MockLLM 靠它判断"记忆已覆盖问题，不必再检索"。
    if state.memory:
        mem_lines = ["[语义相似] 以下历史问题与本 issue 相关："]
        for m, score in state.memory[:3]:
            mem_lines.append(f"- {m.symptom[:50]}（相似度 {score:.2f}）"
                             f"→ 当时分类：{m.resolution[:30]}")
        memory_text = "\n".join(mem_lines)
    else:
        memory_text = "（无）"

    # `工具名(` 这个格式是 MockLLM 识别的"已调用"痕迹
    if state.history:
        hist_lines = []
        for i, h in enumerate(state.history, 1):
            name = h["action"]["name"]
            obs = str(h.get("observation", ""))[:200]
            hist_lines.append(f"{i}. 调用 {name}("
                              f"{h['action'].get('args', {})}) → {obs}")
        history_text = "\n".join(hist_lines)
    else:
        history_text = "（还没做过任何操作）"

    user = PROMPT_HEADER.format(
        title=issue.title if issue else "",
        body=(issue.body or "")[:600] if issue else "",
        memory=memory_text,
        labels=", ".join(labels),
        history=history_text,
    )
    return [
        {"role": "system", "content": "你是一个严谨的开源项目 issue 分诊助手。"},
        {"role": "user", "content": user},
    ]


# =============================================================================
# 动作解析（两版共用）
# =============================================================================
# ★★ 这里踩了一个坑，值得完整记录 ★★
#
# 我原本在 `shared.py` 里自己写了一套解析（正则匹配"工具调用：xxx"）。
# 写的时候想的是"两版都用这一份就公平了"。
#
# **但它和主项目的解析逻辑不一致** —— 主项目（以及 MockLLM）用的是
# **JSON 格式**：
#
#     {"thought": "...", "action": "search_similar_issues",
#      "action_input": {"query": "..."}}
#
# 我写的是文本格式（"工具调用：xxx\n参数：{...}"）。
#
# 跑起来的后果：**两版都解析不出任何动作，预测结果全是空列表** ——
# 而"两版一致"这个检查还通过了（因为都错得一样）。
#
# ============================================================
# 教训：**"共用"应该是"复用同一份实现"，而不是"写两份一样的"**
# ============================================================
# 我以为"把公共部分抽到一个文件"就保证了公平 ——
# 但因为那份实现是**我重写的**，它和真实系统（主项目）的行为不一致，
# 于是"公平"只存在于两版之间，而两版一起偏离了真实。
#
# **这是对照实验的一个隐蔽陷阱**：
#   只关心"两版之间一致"，忘了问"和真实实现一致吗"。
#
# 正确做法（如果继续做）：把 `TriageAgent._parse_decision` 提取成
# 独立函数，让主项目和本实验都 import 它 —— 真正的单一份实现。
# 本次为了不改动主项目（避免引入风险），采用了"对齐格式的简化实现"，
# 并在此明确标注差异。

_TOOL_NAMES = {
    "search_issues", "search_similar_issues", "read_code",
    "search_docs", "apply_labels", "post_comment", "finalize",
}


def _strip_fence(text: str) -> str:
    """去掉 markdown code fence —— 真实模型经常把 JSON 包在 ```json 里。

    （这个需求和主项目的 `_parse_decision` 完全一样，
      所以这里是"照抄它的一小步"，而不是"重新设计"。）
    """
    if "```" not in text:
        return text
    for part in text.split("```"):
        p = part.strip()
        if p.startswith("json"):
            p = p[4:].strip()
        if p.startswith("{"):
            return p
    return text


def parse_action(text: str) -> Optional[dict]:
    """从模型输出里解析动作。

    返回：
        {"name": ..., "args": {...}}   有工具调用
        None                            模型给了最终答案（= 结束）

    ★ 返回 None 的语义是"结束"，这个约定两版必须一致 ——
      否则终止条件的对比就不公平了。

    格式（和主项目 / MockLLM 对齐）：
        {"thought": "...", "action": "工具名", "action_input": {...}}
        {"thought": "...", "action": "finalize", "action_input": {...}}
    """
    import json
    if not text:
        return None

    raw = _strip_fence(text.strip())

    d = None
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        # 从文本里抠 JSON（真实模型会加前言后语）
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except (ValueError, TypeError):
                d = None

    if not isinstance(d, dict):
        return None

    action = d.get("action")

    # ★ 类型校验必须做 —— 真实模型有时把 action 输出成**对象**而不是字符串。
    #   主项目在真实模型压测时踩过这个坑（见 react.py 的 _parse_decision 注释）：
    #   `{"action": {"name": "xxx", "params": {...}}}`
    #   如果直接拿它当字典键，会在很远的地方报 `unhashable type: 'dict'`。
    if isinstance(action, dict):
        # 兼容这种不规范的输出
        action = action.get("name") or action.get("tool") or ""
    if not isinstance(action, str):
        return None
    action = action.strip()

    if not action:
        return None
    # finalize = 模型主动结束（这是主项目的约定）
    if action == "finalize":
        return None
    # 不认识的工具名 → 当作结束，避免下游报"未知工具"刷屏
    if action not in _TOOL_NAMES:
        return None

    args = d.get("action_input") or d.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return {"name": action, "args": args}


def parse_final(text: str) -> list[str]:
    """解析最终分类。

    支持两种来源：
      ① `finalize` 动作里的 action_input（主项目的约定）
      ② 输出里直接出现标签名（Mock 的兜底路径）

    ★ 解析不出来就返回空列表 —— **不猜**。
      猜错比不猜更糟：空列表在评测里是明确的"没给答案"，
      而猜错会被记成"答案错误"，两者对排查的意义完全不同。
    """
    import json
    if not text:
        return []

    raw = _strip_fence(text.strip())
    labels = ("Bug", "Docs", "Enhancement", "Performance", "Build")

    # ① 从 JSON 里取
    d = None
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        if m:
            try:
                d = json.loads(m.group(0))
            except (ValueError, TypeError):
                d = None

    if isinstance(d, dict):
        payload = d.get("action_input") or {}
        if isinstance(payload, dict):
            # 尝试多个可能的字段名（不同模型输出习惯不同）
            for key in ("labels", "label", "classification", "answer"):
                v = payload.get(key)
                if isinstance(v, list):
                    got = [str(x).strip() for x in v]
                    out = [l for l in got
                           if l.lower() in {x.lower() for x in labels}]
                    if out:
                        return out
                if isinstance(v, str) and v.strip():
                    got = [s.strip() for s in v.replace("、", ",").split(",")]
                    out = [l for l in got
                           if l.lower() in {x.lower() for x in labels}]
                    if out:
                        return out

    # ② 直接从文本里找标签名
    found = []
    low = raw.lower()
    for lab in labels:
        if lab.lower() in low:
            found.append(lab)
    return found


# =============================================================================
# 组件构造（两版共用）
# =============================================================================

def _seed_memories() -> list[MemoryRecord]:
    """预置的历史记忆 —— 两版用同一份，保证记忆召回的输入相同。"""
    return [
        MemoryRecord(memory_id="h1", issue_id="#100", symptom="CI 构建超时失败",
                     symptom_tags=["Build", "timeout"],
                     resolution="Bug", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h2", issue_id="#101", symptom="文档示例代码无法运行",
                     symptom_tags=["Docs"],
                     resolution="Docs", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h3", issue_id="#102", symptom="read_csv 内存占用过高",
                     symptom_tags=["Performance", "memory"],
                     resolution="Performance", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h4", issue_id="#103", symptom="merge 结果行数异常",
                     symptom_tags=["Bug"],
                     resolution="Bug", outcome=Outcome.RESOLVED),
    ]


def setup_tools():
    """构造记忆库 + 工具集（两版共用）。

    ★ 只注册**两个工具**（而不是主项目的七个）——
      因为对照实验要控制变量：工具越多，循环越长，
      编排差异之外的噪声就越大。**两个工具足以触发"多轮循环"。**
    """
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=64))
    for m in _seed_memories():
        store.insert(m)

    repo = IssueRepository([])
    tools = ToolRegistry(repo, store=store, current_issue_id="")
    return store, tools


def make_llm(use_mock: bool = True):
    """构造 LLM（两版共用）。

    默认 Mock —— 因为对照实验测的是**编排**，不是模型能力。
    用真实模型会引入随机性，让"两版差异"和"模型波动"混在一起。
    **控制变量：一次只测一个东西。**
    """
    return build_llm(use_mock=use_mock)


def recall_memory(issue: Issue, store) -> tuple[list, int]:
    """记忆召回（两版共用）。

    返回 (记忆列表, 消耗 token)。手写版里 token 记 0
    —— 因为召回不调模型，不该算进 LLM token。
    """
    try:
        hits = store.recall(issue.search_text(), top_k=3)
        # ★ recall 返回 [(记忆, 分数), ...] —— 必须解包
        return [(m, s) for m, s in hits], 0
    except Exception:                                       # noqa: BLE001
        return [], 0


# =============================================================================
# 测试样本（两版共用）
# =============================================================================

def sample_issues() -> list[Issue]:
    """对照实验用的 issue（固定样本，保证两版跑的是同一批）。"""
    rows = [
        ("#901", "CI 构建在测试阶段超时",
         "pytest 跑满 300 秒后超时退出，本地可以跑完", ["Build"]),
        ("#902", "文档中的示例代码报错",
         "docs/io.rst 里的 read_csv 示例执行报 TypeError", ["Docs"]),
        ("#903", "读取大 CSV 时内存暴涨",
         "read_csv 读 2GB 文件时内存占用超过 8GB", ["Performance"]),
        ("#904", "merge 后行数比预期多",
         "两表 merge 之后行数从 1000 变成 5000", ["Bug"]),
        ("#905", "安装依赖时报编译错误",
         "pip install 时 C 扩展编译失败", ["Build"]),
    ]
    return [
        Issue(issue_id=i, title=t, body=b, author="tester",
              state=IssueState.CLOSED, gold_labels=g)
        for i, t, b, g in rows
    ]
