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


def format_memory(mem, score: float, tag: str = "语义相似") -> str:
    """把一条记忆渲染成给模型看的文本。

    ============================================================================
    ★★ 这个函数是"照抄主项目"的，不是"重新设计"的 —— 原因值得完整记录
    ============================================================================
    主项目的渲染逻辑在 `nightwatch/loop/react.py::TriageAgent._format_memory`。
    这里**逐字对齐它的字段格式**，而不是自己发明一套。

    ★ 为什么非要对齐？因为 MarkLLM 是靠**在上下文里抓字段名**来读记忆的：

        MockLLM._extract_memory_tags(ctx):
            re.search(r"标签[:：]\\s*([a-zA-Z0-9_,\\s/]+)", ctx)

    它只认 `标签: xxx` 这个写法。而本文件**第一版**自己写的是：

        "- {症状}（相似度 {分数}）→ 当时分类：{resolution}"

    **没有 `标签:` 字段** → `_extract_memory_tags` 永远返回空 →
    记忆里的标签（`Build`/`Docs`/`Performance`）完全没被利用。

    ============================================================
    这个错误的症状有多隐蔽（这才是值得记的部分）
    ============================================================
    它**不报错**，只表现为"四条命中样本里两条输出 needs-triage"：

        #901 CI 构建在测试阶段超时   → ['bug/timeout']   ✓
        #902 文档中的示例代码报错     → ['needs-triage']  ✗
        #903 读取大 CSV 时内存暴涨    → ['bug/oom']       ✓
        #904 merge 后行数比预期多     → ['needs-triage']  ✗

    **看起来像"模型能力不足"** —— 因为正好是"超时""内存"这两个
    在 MockLLM 关键词表里的词过了，"文档报错""merge 行数"这两个没过。

    如果当时顺着"这是模型不行"去调提示词或调关键词表，
    **就会把评测基准往 Mock 的实现偏差上拧** ——
    而真正的问题是"我的渲染和真实系统不一致"。

    ★ 这正是本文件开头那条教训的第三次出现：
      > "共用"应该是"复用同一份实现"，而不是"写两份一样的"。
      我第一版把**解析**对齐了（parse_action 照抄 JSON 格式），
      却漏了**渲染** —— 于是两版一起偏离真实。
      **"只改了一处"的感觉是假的**：对齐是一个"面"，不是"点"。

    ★ 正确做法（如果继续做）：把 `_format_memory` 提成模块级函数放
      `nightwatch/` 里，主项目和本实验都 import 它 —— 真正的单一份实现。
      本次为不改动主项目（避免引入风险）采用"逐字对齐的简化实现"，
      并把差异**显式标注**在这里。

    ⚠️ 与主项目的已知差异（有意保留，不影响本实验）：
      · 主项目渲染 `根因` / `处置` / 失败动作；这里只渲染 `标签` + `症状`
        —— 因为对照实验只要触发"记忆命中"，不需要完整记忆内容。
      · 但 **`标签:` 这一行必须保留** —— 它是 MockLLM 读记忆的唯一入口。
    """
    lines = [f"[{tag}] 记忆ID={mem.memory_id} | 相似度 {score:.2f} "
             f"| 来源 issue {mem.issue_id}"]
    if mem.symptom_tags:
        lines.append(f"  标签: {', '.join(mem.symptom_tags)}")
    lines.append(f"  症状: {mem.symptom}")
    if mem.resolution:
        lines.append(f"  处置: {mem.resolution}")
    return "\n".join(lines)


def build_prompt(state: AgentState) -> list[dict]:
    """拼提示词。

    ★ 这个函数在 LangGraph 版里会被**包进一个节点**；
      在手写版里它就是循环体内的一行调用。

      **功能完全相同，差异只在"它挂在哪个位置"** ——
      这正是我们要观察的。
    """
    issue = state.issue
    labels = ["Bug", "Docs", "Enhancement", "Performance", "Build"]

    # ★ 有记忆时必须打 `[语义相似]` / `[同一提交者的历史]` 标记 ——
    #   见上面注释，MockLLM 靠它判断"记忆已覆盖问题，不必再检索"。
    #
    #   ⚠️ 注意 `[语义相似]` 这个字符串**必须原样出现** ——
    #     MockLLM 检查的是 `"[语义相似]" in ctx`（带方括号）。
    #     这里改成 `format_memory(..., tag="语义相似")` 后，
    #     方括号由 format_memory 里的 `[{tag}]` 补上 —— 所以格式仍然对。
    if state.memory:
        memory_text = "\n".join(
            format_memory(m, score, tag="语义相似")
            for m, score in state.memory[:3]
        )
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
    """预置的历史记忆 —— 两版用同一份，保证记忆召回的输入相同。

    ★ 措辞是**标定过**的（不是随手写的）—— 这直接决定实验能不能构造出
      "命中"和"不命中"两组样本。改动这里的任何一句话，
      都要重跑 `scripts/` 下的相似度校准，确认两组分布仍然分得开。

      标定原则：**记忆用词要含该症状域里最有判别力的那个词**
      （CI 构建 / 文档示例 / read_csv 内存 / merge 行数），
      这样"属于这个域"的样本能拿到高分，别的域拿不到。
    """
    return [
        MemoryRecord(memory_id="h1", issue_id="#100",
                     symptom="CI 构建在测试阶段超时失败",
                     symptom_tags=["Build", "timeout"],
                     resolution="Bug", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h2", issue_id="#101",
                     symptom="文档中的示例代码无法运行",
                     symptom_tags=["Docs"],
                     resolution="Docs", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h3", issue_id="#102",
                     symptom="read_csv 读取大文件时内存占用过高",
                     symptom_tags=["Performance", "memory"],
                     resolution="Performance", outcome=Outcome.RESOLVED),
        MemoryRecord(memory_id="h4", issue_id="#103",
                     symptom="merge 之后行数比预期多",
                     symptom_tags=["Bug"],
                     resolution="Bug", outcome=Outcome.RESOLVED),
    ]


def setup_tools():
    """构造记忆库 + 工具集（两版共用）。

    ★ 只注册**两个工具**（而不是主项目的七个）——
      因为对照实验要控制变量：工具越多，循环越长，
      编排差异之外的噪声就越大。**两个工具足以触发"多轮循环"。**

    ★ 嵌入维度显式传 `EMBED_DIM`（= 256，**与主项目 `HashEmbedding` 默认值一致**）
      —— 见 `EMBED_DIM` 的说明。

      **为什么"显式传"而不是"不传、用默认值"**：
      这两者当前**结果相同**（都是 256），但显式传是刻意的 ——
      它把"实验用哪个维度"变成一行**看得见、可搜索、可对照**的代码，
      而不是一个藏在 `store.py` 里的隐式默认值。
      下面 `dim_sweep.py` 就是靠改这一个常量来跑单变量对照的。

      ⚠️ 早期版本这里写的是 `dim=64`（**自己拍的，不是默认值**）——
        结果哈希碰撞严重到无关样本拿到 0.587，两组分布重叠、阈值切不开。
        见 `EMBED_DIM` 的完整诊断。
    """
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=EMBED_DIM))
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


# =============================================================================
# 记忆召回（两版共用）
# =============================================================================
#: 记忆召回的**最低相似度**（低于它的不算"命中"）。
#
# ★★ 为什么必须加这个阈值（踩过的坑，值得完整记录）
# ============================================================================
# 这个实验第一版的循环覆盖率是 **0/15** —— 所有样本第 1 步就 finalize，
# `while`/条件边回边**从未执行**。而"框架版多 67 行"里相当一部分正是这些
# 未执行的代码 —— **结论建立在从未跑过的分支上**。
#
# 排查发现根因有三层（前两层是逻辑，第三层是数值）：
#   ① `store.recall()` **没有相似度阈值** —— `_rank()` 只是排序后取 top_k。
#      所以**只要记忆库非空，任何 issue 都会"命中"**。
#   ② `build_prompt` 只要记忆列表非空就打 `[语义相似]` 标记，
#      而 MockLLM 看到这个标记就 finalize。
#      → ①②合起来：**"不命中"这个状态根本不存在**，
#        所以"加几条症状不同的样本"治不了本。
#   ③ **嵌入维度太低导致哈希碰撞**（这一层最隐蔽，见下）。
#
# ---------------------------------------------------------------------------
# ★★ 第三层：实验脚本自己写死的 dim=64（**不是**主项目默认值）★★
# ---------------------------------------------------------------------------
# 加了阈值之后，我以为问题解决了 —— 因为热门样本的 TOP1 分数看起来"分得开"。
# 但探针测试（"UI 按钮错位 / flex 布局在 Safari 下换行" —— 和 CI 构建毫无关系）
# 打出了 **0.587**，比"该命中"的 #901（0.473）还高。
#
# 诊断过程（值得记，因为这是"阈值调不出来"的典型症状）：
#
#     重合 bigram: 只有 ['i ']           ← 字面上只共有一个"i 空格"
#     q buckets: 26 个      m buckets: 18 个
#     intersect: [4, 9, 13, 27, 32, 33, 35, 37, 39, 50]  ← 撞了 10 个桶
#
# **只重合 1 个 bigram，却有 10 个桶相撞** —— 这说明碰撞不是来自真实重合，
# 而是来自**维度太窄**：64 个桶装几百个 bigram，平均每个桶要塞 5~6 个不同的
# token，于是"毫不相干的字符对"落进同一个桶就被当成相似。
#
# ⚠️ 而这个 64 **不是主项目默认值** —— 是实验脚本自己写死的参数
#    （`setup_tools()` 早期版本写 `HashEmbedding(dim=64)`）。
#    主项目 `HashEmbedding` 的真实默认值是 **256**（`store.py:199`）。
#    **把"自己拍的数"说成"默认值"是事实陈述错误** ——
#    它会让读者以为主项目就这么窄，从而得出错误结论
#    （"主项目的召回质量这么差"）。**这比参数选错更严重。**
#
# ---------------------------------------------------------------------------
# ★ dim 单变量对照的实测结果（`dim_sweep.py`，只改 dim 一个变量）
# ---------------------------------------------------------------------------
#     维度    相关组区间         无关组最高   空档       结论
#     64      [0.473, 0.668]    0.587      -0.115     重叠 —— 切不开
#     256     [0.355, 0.568]    0.293      +0.061     分得开（余量小）
#     1024    [0.366, 0.550]    0.222      +0.144     分得开（余量充足）
#
#   64 → 256：空档从 -0.115 翻到 +0.061（**这是质变**：从重叠到可分）。
#   256 → 1024：空档再扩 0.083（量变，收益递减）。
#
#   **所以"维度太低"这个诊断本身是对的，只是我把"太低"的参照点说错了** ——
#   出问题的是实验自己写的 64，不是主项目的 256。
#
#   做法因此变成"**实验参数未复用主项目默认值**" ——
#   用 256 就行了，不必额外造一个 1024。
#
# ---------------------------------------------------------------------------
# ★★ 但必须说清楚：分数尺度也变了，不只是碰撞没了 ★★
# ---------------------------------------------------------------------------
#   维度提高会让**所有**余弦分数普遍下降（同一批样本的两组均值）：
#
#     dim     相关组均值   无关组均值   全体均值
#     64      0.579        0.480        0.524
#     256     0.454        0.233        0.331
#     1024    0.442        0.147        0.278
#
#   原因：维度越高，单个桶分到的质量越稀，任意两个向量的**点积被整体压小**。
#   例：#901 从 0.473（64）掉到 0.366（1024）—— 只降了 0.107，
#   而无关组的探针从 0.587 掉到 0.169 —— 降了 0.418。
#
#   **两组都降，但无关组降得狠得多。**
#   这才是分得开的**真正来源**：不是"碰撞被消除"这一件事，
#   而是"碰撞造成的虚高被消除，同时真实重合的分数被等比稀释" ——
#   两者的**相对**关系反转了。
#
#   ⚠️ 所以"区分度出现"里**含有尺度变化的成分**，不是纯碰撞消除。
#   这在别处也有实际影响：**跨维度比较绝对分数是没有意义的**，
#   以及"阈值要随维度重新标定"——不能把一个维度上调好的阈值搬到另一个维度。
#
# **教训：当"调阈值"怎么调都不对时，先检查信号本身有没有区分度。**
#   阈值只能切分分布，**不能把重叠的分布切开**。
#   我当时的第一反应是"再调调阈值" —— 而用户明确说了"不要调松阈值"，
#   这句话逼我去找根因，才发现问题在嵌入维度，不在阈值。
#
# ---------------------------------------------------------------------------
# ★ 阈值取值的依据（dim=256 下实测的分布，不是拍的）
# ---------------------------------------------------------------------------
#     样本              最高相似度   应该算命中吗
#     #901 CI 构建超时   0.355       ✓
#     #902 文档示例报错   0.461       ✓
#     #903 读大 CSV 内存  0.434       ✓
#     #904 merge 行数多   0.568       ✓
#     #906 时区转换错误   0.286       ✗
#     #907 分组聚合 NaN   0.185       ✗
#     探针 UI 按钮错位     0.293       ✗  ← ★ 无关组最高其实是它，不是 #906
#     探针 SQL 注入        0.177       ✗
#     探针 主题切换闪烁     0.224       ✗
#
#   相关组最低 0.355，无关组最高 0.293（**含 3 个未标定探针**）
#   → 空档 = [0.293, 0.355]，宽度 0.061，**中点 = 0.324**。
#
# ⚠️⚠️⚠️ 这个阈值**依赖于 EMBED_DIM** —— 改 dim 必须重新取空档中点 ⚠️⚠️⚠️
#
#   **不能沿用旧 dim 下标定出来的值。** 两个理由：
#
#   ① **空档位置会整体平移**。dim 提高会让所有余弦分数普遍下降
#      （见 `EMBED_DIM` 里的尺度表：全体均值 0.524 → 0.331 → 0.278）。
#      同一个 0.30，在 dim=64 下落在重叠区、在 dim=256 下才勉强可用 ——
#      **它对应的"松紧程度"在每个维度下都不同**。
#
#   ② **空档宽度也会变**。64 时是 -0.115（重叠，根本切不开），
#      256 时 +0.061，1024 时 +0.144。可用区间本身在变。
#
#   本次就是活生生的例子：0.30 是**按 dim=1024 的空档 [0.222, 0.366]
#   取的中点**（中点 0.294，取整到 0.30）。EMBED_DIM 改成 256 之后，
#   空档变成 [0.293, 0.355]，中点 0.324 —— **0.30 已不再是任何中点**，
#   而且它的下限余量只剩 +0.007（0.300 - 0.293），几乎贴着探针。
#   所以才重标定到 0.32。
#
#   ★ 换 dim 时的**操作顺序**（别反了）：
#       1. 先跑 `dim_sweep.py`，看新 dim 下的空档 [lo, hi] 与是否 >0
#       2. 空档 ≤0 → 换 dim（这个维度切不开，阈值救不了）
#       3. 空档 >0 → 取中点 (lo+hi)/2，再按"偏保守"微调
#       4. 重跑 `compare.py`，确认循环覆盖率和一致率没变
#     **顺序反了（先调阈值再看能不能分开）就会掉进
#     "阈值怎么调都不对"的那个坑 —— 因为分布本身是重叠的。**
#
# ---------------------------------------------------------------------------
# ★ 取 0.32 的依据：中点 0.324 往"危险的那一侧"让一点
# ---------------------------------------------------------------------------
#   两个方向的余量（dim=256，含探针）：
#
#     到下限（阈值 - 无关组最高）  ← 排除**误召回**（不该命中的命中了）
#     到上限（相关组最低 - 阈值）  ← 保证**不漏召回**（该命中的没命中）
#
#     阈值      到下限        到上限        说明
#     0.300    +0.007        +0.055       ← 旧值：下限只剩千分之七
#     0.324    +0.031        +0.031       ← 严格中点（两侧相等）
#     0.320    +0.027        +0.035       ← **当前取值**
#
#   ★ **为什么取 0.32 而不是严格中点 0.324**：
#
#     先说清楚方向：**阈值降低 → 到下限变小、到上限变大**。
#     所以 0.32 < 0.324 意味着 0.32 的到下限（+0.027）**比中点还小**
#     —— 它是让"危险的那一侧"更紧，**不是更松**。
#
#     那为什么还取 0.32？因为它仍然**显著优于旧值 0.30**：
#       到下限  +0.007 → +0.027（**约 4 倍**）   ← 危险侧，主要收益
#       到上限  +0.055 → +0.035（仍宽裕）        ← 安全侧，付出一点
#     即：**把安全侧多余的余量搬一部分给危险侧**。
#     这正是"偏保守"的含义 —— 不是"两边都留够"，而是
#     **"宁可漏召回的余量小一点，也要让误召回那一侧不贴着悬崖"**。
#
#     ★ 真正的原则：**余量要按"两边的代价"分配，不是平均分。**
#       本项目实测过边界是**单侧**的（见下），误召回那一侧才是真会破的，
#       所以让它多留。**"取中点"是默认策略，不是最优策略** ——
#       只有在"两侧代价相等"时中点才是最优。
#
# ⚠️ 破坏性验证（`dim_sweep.py` 的实测，把 #906 措辞往 h1 靠一点点）：
#
#     #906 措辞                         TOP1     后果
#     原版「时区转换结果偏移了 8 小时」   0.286    不命中 ✓
#     加个「超时」                       0.300    越过旧阈值 →
#                                               #906 变"命中"，
#                                               循环覆盖率 6/18 → 3/18
#     更像 h1（加"测试阶段超时"）         0.329    同上
#
#   ★ 注意 0.329 这一档：它**已经越过新阈值 0.32** ——
#     所以提到 0.32 **不是"让这个坑消失"**，只是把安全垫从
#     千分之七加厚到百分之二点七。**措辞再靠近一点，照样会破。**
#     这个参数永远需要那两条断言盯着，不能因为"调过了"就放心。
#
#   ★ 反方向不会触发问题：把"该命中"的样本改远一点，余量反而变大。
#     因为此时"不命中组最高"换成了更低的样本。
#     **所以这个边界是单侧的**：只对"无关样本变热"敏感。
#     加样本时要小心的不是"已有样本够不够突出"，而是
#     **"新样本的措辞会不会靠近已有记忆"**。
#
#   ⚠️ 余量比 dim=1024 时**紧得多**（0.061 vs 0.144）。
#   这是用 256 换"与主项目一致"付出的代价，必须明说。
#
# ⚠️ 但要说清楚：**这是"实验参数"，不是"产品参数"。**
#    真实的召回阈值应该按业务需求标定（漏召回 vs 误召回的代价权衡），
#    而不是"让这个实验跑通"。这里取 0.32 是因为它**同时满足两件事**：
#      · 让该命中的样本命中（保留"记忆命中 → 0 轮"这条对照路径）
#      · 让该不命中的样本不命中（产生"走完整循环"的样本）
MIN_RECALL_SIMILARITY = 0.32

#: 「不该命中」的探针样本 —— 用来测"无关组的**真实底噪**"。
#
# ★★ 为什么这份定义放在 shared.py 而不是 dim_sweep.py ★★
#   原来它在 `dim_sweep.py` 里，**测试看不到** —— 于是出现了"两套口径"：
#       dim_sweep 含探针 → 真实余量 +0.007（当时阈值 0.30）
#       测试只用标定样本 → 乐观余量 +0.069
#   两者差了一个数量级，而**测试报的是那个乐观的数**。
#   守卫必须测**最危险的那个数** —— 也就是含探针的那个。
#   所以定义上提到 shared.py，让 `dim_sweep.py` 和 `test_smoke.py`
#   从**同一个常量**取，口径不可能再分叉。
#   （这正是本文件反复讲的原则：**"共用"要复用实现，不要各写一份**。
#     上次是解析/渲染，这次是样本集 —— 同一个坑的又一次。）
#
# ★ 为什么光靠 #906/#907 不够
#   它们是**同一批标定过的样本**，措辞本来就刻意避开了 h1~h4 的词域。
#   如果只用它们，测到的是"标定过的无关"，不是"一般意义上的无关"。
#   探针样本是**没参与标定**的额外输入，用来暴露"底噪"本身有多高 ——
#   底噪高说明哈希碰撞严重，那是维度问题的直接症状。
#
# ⚠️ 实测（dim=256，阈值 0.32）：这 3 个探针里"UI 按钮错位"拿到 **0.293**，
#    比标定样本 #906（0.286）**还高** —— 也就是说：
#    **真正决定阈值能不能用下限的，是探针，不是标定样本。**
#    只用标定样本量余量，会量出一个偏乐观的数。
PROBES: list[tuple[str, str]] = [
    ("UI 按钮错位", "flex 布局在 Safari 下换行"),
    ("SQL 注入漏洞", "参数未做转义"),
    ("UI 主题切换闪烁", "暗色模式下首屏闪白"),
]


def sample_probes() -> list[str]:
    """把探针转成可直接召回用的文本（和 `dim_sweep.py` 保持一致）。

    ★ 拼法必须和 `dim_sweep.eval_dim()` 里一致（`f"{title}\\n{body}"`）——
      否则两处算出来的探针分数不同，口径又分叉了。
      所以这里抽成函数，而不是让两边各拼一次字符串。
    """
    return [f"{t}\n{b}" for t, b in PROBES]


def separation_margin(store) -> dict:
    """量出**含探针的完整样本集**下的空档与两个方向余量。

    ★★ 这是本文件里唯一"权威"的余量口径 ★★
      测试和 `dim_sweep.py` 都调它，不再各自实现。
      返回:
        low_hit    相关组最低分（该命中的里最低的那个）
        high_miss  无关组最高分（含 3 个探针）
        gap        空档宽度 = low_hit - high_miss
        slack_lo   到下限 = MIN_RECALL_SIMILARITY - high_miss（>0 才排除误召回）
        slack_hi   到上限 = low_hit - MIN_RECALL_SIMILARITY（>0 才保证不漏召回）
        worst      两个余量里**更小**的那个 —— 这才是"最危险的那个数"

    ★ 为什么要单列 `worst`：只看 gap 会骗人。
      gap 是"两个方向余量之和"，可能一边 +0.040、另一边 +0.002，
      总 gap 看着有 0.042 挺宽，而实际已经贴着悬崖了。
      **守卫要盯的是 min(两侧余量)，不是它们的和。**
    """
    WANT_HIT = {"#901", "#902", "#903", "#904"}
    hits: list[float] = []
    for iss in sample_issues():
        if iss.issue_id in WANT_HIT:
            r = store.recall(iss.search_text(), top_k=1)
            hits.append(r[0][1] if r else 0.0)
    misses: list[float] = []
    for iss in sample_issues():
        if iss.issue_id not in WANT_HIT:
            r = store.recall(iss.search_text(), top_k=1)
            misses.append(r[0][1] if r else 0.0)
    for text in sample_probes():
        r = store.recall(text, top_k=1)
        misses.append(r[0][1] if r else 0.0)

    low_hit, high_miss = min(hits), max(misses)
    slack_lo = MIN_RECALL_SIMILARITY - high_miss
    slack_hi = low_hit - MIN_RECALL_SIMILARITY
    return {
        "low_hit": low_hit, "high_miss": high_miss,
        "gap": low_hit - high_miss,
        "slack_lo": slack_lo, "slack_hi": slack_hi,
        "worst": min(slack_lo, slack_hi),
        "n_related": len(hits), "n_unrelated": len(misses),
    }


#: 嵌入维度。
#
# ★ 为什么是 256 而不是别的数：**256 是主项目 `HashEmbedding` 的默认值**
#   （`src/nightwatch/memory/store.py:199`，`def __init__(self, dim: int = 256)`）。
#
#   ★★ 这一版比上一版"更省事" —— 上一版写的是 1024，理由是"64 太低"。★★
#      那个理由**方向对、参照点错**：出问题的是实验自己写的 64，
#      不是主项目的 256。实测 256 的空档 +0.061 > 0（能分开）——
#      所以**没必要额外造一个 1024**，直接用主项目默认值即可。
#
#      这也顺带修掉了本文件原来的一个**事实陈述错误**：
#      旧注释写"为什么从**默认的 64** 提到 1024" ——
#      64 从来不是默认值，是本文件 `setup_tools()` 早期自己写死的。
#      把"自己拍的数"说成"默认值"，会让读者以为主项目就这么窄。
#
# ★ 为什么"用主项目默认值"比"实验自己挑一个数"更好（不只是省事）：
#      · **减少一个变量**：实验和主项目共用同一套嵌入参数，
#        差异就只剩"编排方式"（这正是本实验要测的唯一变量）。
#      · **发现的问题更可能是真的**：如果实验用一套自己的参数，
#        那么"召回/不命中"的行为和主项目不一致 ——
#        在实验里观察到的问题**可能是实验自己造出来的**。
#      · 这是本文件开头反复讲的同一个原则：
#        **"共用"要复用实现，不要重写一份** ——
#        参数也是实现的一部分，"我自己挑个 256 也挺好"和"用默认值"是两件事。
#
# ★ 为什么不继续用 1024（它空档更宽，+0.144）：
#     **空档更宽不是"更对"，只是"更宽松"。** 1024 的代价是：
#       · 与主项目参数不一致 → 引入了本不必有的差异
#       · 掩盖了"标定只做了刚好够"这个事实
#     当前 256 的空档 +0.061 已经够用（阈值 0.32 落在里面）。
#     **用"够用的最小值"而不是"最宽松的值"** —— 后者会让实验比真实系统
#     更稳健，从而**测不出真实系统的脆弱点**。
#
#     （这一点在别的项目里也成立：把测试环境调得比生产宽松，
#       测出来的"通过"就是假的。**实验的稳健度应该匹配被测系统，不能超过它。**）
#
# ⚠️⚠️ **改这个值必须连带重标 `MIN_RECALL_SIMILARITY`** ⚠️⚠️
#     两者是**绑定的**，不是一个可以单独调的旋钮。
#     原因：改 dim 会同时挪动"空档的位置"和"空档的宽度"
#     （64: -0.115 → 256: +0.061 → 1024: +0.144），
#     旧阈值在新维度下对应的松紧程度完全不同。
#     所以改这里的正确顺序是：
#       1. 改 `EMBED_DIM`
#       2. 跑 `dim_sweep.py` 看新空档 [lo, hi]
#       3. lo >= hi（重叠）→ 这个 dim 不能用，回去改 dim
#       4. lo < hi → 取新中点 (lo+hi)/2，按单侧代价微调 → 更新阈值
#       5. 重跑 `compare.py` + 全量测试
#     **跳过第 2~4 步直接跑，会遇到"阈值怎么调都不对"** ——
#     那就是本项目第 2 层根因的原始症状。
#
# ⚠️ 残留风险（诚实清单）：
#     256 的空档只有 0.061，而且**到下限只有 +0.027**（见
#     `MIN_RECALL_SIMILARITY` 上方的拆解）。改样本或记忆的**任何措辞**
#     都可能让它翻掉。所以 `TestCompareExperiment` 里那条余量断言是
#     **用含探针的完整样本集测 min(两侧余量)** —— 它不是形式主义，
#     是这个参数真的悬。
#
# ★★ dim 提高还会改变分数尺度 —— 这一点必须和"区分度"分开记 ★★
#     维度提高 → 每个桶分到的质量变稀 → **所有**余弦分数普遍下降：
#
#       dim     相关组均值   无关组均值   全体均值
#       64      0.579        0.480        0.524
#       256     0.454        0.233        0.331
#       1024    0.442        0.147        0.278
#
#     例：#901 从 0.473（dim=64）掉到 0.366（dim=1024）。
#
#     **所以"区分度出现"里含有尺度变化的成分，不是纯碰撞消除。**
#     两组都降，但无关组降得狠得多（0.480→0.147，降 0.333）
#     而相关组只降 0.137 —— **两者的相对关系反转了**，
#     这才是"分得开"的真正来源。
#
#     两个实际后果（别踩）：
#       · **跨维度比较绝对分数没有意义** ——
#         "0.5 算高分吗"这个问题在不同 dim 下答案不同。
#       · **阈值必须随维度重新标定** ——
#         不能在 1024 上调好 0.30 然后搬到 256 上直接用
#         （本次是反过来的：先按 256 的分布重新确认了 0.30 才换过来）。
EMBED_DIM = 256


def recall_memory(issue: Issue, store) -> tuple[list, int]:
    """记忆召回（两版共用）。

    返回 (记忆列表, 消耗 token)。手写版里 token 记 0
    —— 因为召回不调模型，不该算进 LLM token。

    ★ 加了一道**最低相似度过滤** —— 见上面 `MIN_RECALL_SIMILARITY` 的说明。
      这一步让"记忆不命中"成为**可能**（否则 memory 非空就永远命中）。
    """
    try:
        hits = store.recall(issue.search_text(), top_k=3)
        # ★ recall 返回 [(记忆, 分数), ...] —— 必须解包
        out = [(m, s) for m, s in hits if s >= MIN_RECALL_SIMILARITY]
        return out, 0
    except Exception:                                       # noqa: BLE001
        return [], 0


# =============================================================================
# 测试样本（两版共用）
# =============================================================================

def sample_issues() -> list[Issue]:
    """对照实验用的 issue（固定样本，保证两版跑的是同一批）。

    ★★ 样本要覆盖**两条决策路径**（这是修过一次的地方）
    ============================================================================
    第一版 5 条样本**全部命中记忆** → MockLLM 全部走"第 1 步就 finalize"
    → **循环从未执行** → 对照实验的核心（`for` vs 条件边回边）没被测到。

    所以样本集必须**同时包含**：

        命中记忆的（#901~#904）  → 走"记忆预取 → 直接结论"路径（0 次循环）
        不命中记忆的（#906~#907）→ 走"search_similar_issues → search_docs
                                   → finalize"路径（**2 次循环**）

    **只测其中一条路径的对照实验是残缺的** —— 而它不会报错，
    只会让"覆盖率 0"这种数字安静地躺在输出里。

    ★★ 这套措辞是**实测标定**的，不是随手写的（dim=256，阈值 0.32）
    ============================================================================
    | 样本 | 最高相似度 | 命中哪条 | 期望路径            |
    |------|-----------|---------|---------------------|
    | #901 | 0.355     | h1      | 命中 → 0 轮         |
    | #902 | 0.461     | h2      | 命中 → 0 轮         |
    | #903 | 0.434     | h3      | 命中 → 0 轮         |
    | #904 | 0.568     | h4      | 命中 → 0 轮         |
    | #906 | 0.286     | —       | **不命中 → 2 轮**   |
    | #907 | 0.185     | —       | **不命中 → 2 轮**   |

    命中组最低 0.355，不命中组最高 0.286（**仅指标定样本**）。

    ⚠️⚠️ 但**别用这组数估余量** —— 加上 3 个探针后无关组最高是 **0.293**
    （"UI 按钮错位"），比 #906 还高。也就是说：
    **真正卡住阈值下限的是探针，不是标定样本。**
    正确的余量口径见 `separation_margin()`，它含探针。

    当前（阈值 0.32，含探针）：空档 0.061，到下限 +0.027，到上限 +0.035。

    ⚠️ **余量比 dim=1024 时紧得多**（0.061 vs 0.144）。
    所以循环覆盖率对"样本措辞"相当敏感，改 `sample_issues()` 里的文案后
    **必须重跑 `compare.py`** 确认覆盖率没塌成 0。
    这一点是实测踩出来的，不是保险话。

    ★ 新样本（#906/#907）的设计要点：**用词和 h1~h4 完全不重叠**
      h1~h4 的词域：CI / 构建 / 超时 / 文档 / 示例 / read_csv / 内存 /
                     merge / 行数
      新样本用的词域：时区 / tz_localize / 夏令时 / 分组聚合 / 自定义函数
      → 哈希嵌入捕捉的是**字面重合**，词域不同 → 相似度自然低

    ⚠️ 反过来说：**哈希嵌入的"不命中"是廉价的** ——
       它只证明"字面不同"，不证明"语义不同"。
       真 embedding 下这两个样本可能仍然会命中（因为它们都是 pandas 的
       数据处理问题）。所以这个实验证明的是**编排在两条路径上等价**，
       而不是"召回质量好/差"。见 docs/08 的边界说明。

    ★★ 每个样本还带一个 `expected_label`（设计意图，硬编码）★★
    ============================================================================
    为什么需要它：**"一致率"不能区分"两版都对"和"两版一起错"**。
    两版一起错时一致率照样 100%，于是"相等价"这个结论是空的。

    所以必须有**独立的正确性参照**，且它要满足两条：
      ① **硬编码** —— 不能从任何被测逻辑里推出来（推出来就是循环论证）
      ② **放在被测对象之外** —— 见下面 `EXPECTED_LABELS` 的说明

    值就是"这个 issue 该分到哪一类"，凭常识判断：
    CI 构建挂了 → Build；文档示例跑不通 → Docs；
    读大文件爆内存 → Performance；merge 行数不对 → Bug。

    ⚠️ **不要为了让准确率好看而改这个值** —— 那正是"迁就被测系统"的开始。
       准确率低于 100% 是要被**看见**的信号，不是要抹平的噪音。
    """
    rows = _SAMPLE_ROWS
    return [
        Issue(issue_id=i, title=t, body=b, author="tester",
              state=IssueState.CLOSED, gold_labels=g)
        for i, t, b, _exp, g in rows
    ]


# =============================================================================
# 样本定义（**唯一数据源**）
# =============================================================================
# 每行: (issue_id, 标题, 正文, expected_label 设计意图, gold_labels)
#
# ★★ 为什么拆成模块级常量 ★★
#   因为 `sample_issues()` 和 `EXPECTED_LABELS` 必须来自**同一份定义**。
#   各写一份 = "同一件事长出两份实现" —— 本项目已经踩过四次
#   （`parse_action` / `_format_memory` / 探针集 / 余量口径）。
#   **共用要复用实现，不要各写一份。**
_SAMPLE_ROWS: list[tuple[str, str, str, str, list[str]]] = [
    # --- 命中记忆的（保留"记忆 → 0 轮"这条路径）---
    #    id      标题                      正文                                              设计意图        gold
    ("#901", "CI 构建在测试阶段超时",
     "pytest 跑满 300 秒后超时退出，本地可以跑完",
     "Build", ["Build"]),
    ("#902", "文档中的示例代码报错",
     "docs/io.rst 里的示例执行报 TypeError，照着文档抄下来跑不通",
     "Docs", ["Docs"]),
    ("#903", "读取大 CSV 时内存暴涨",
     "read_csv 读 2GB 文件时内存占用超过 8GB",
     "Performance", ["Performance"]),
    ("#904", "merge 后行数比预期多",
     "两表 merge 之后行数从 1000 变成 5000",
     "Bug", ["Bug"]),
    # --- 不命中记忆的（让循环真正跑起来）★ 本次新增 ---
    #
    # ⚠️ 这两条的**标题措辞**也是标定过的 —— 不是随便写的。
    #
    # 原因：`MockLLM._extract_symptom` 只从**标题**取症状（正文不参与标签推断），
    # 所以标题里必须出现关键词表认识的那个词，否则会落到 general
    # → 输出 needs-triage → `parse_final` 解析出空列表 →
    # **这个样本对等价性的证明力变成 0**（两版都空，一致但没意义）。
    #
    # 实测踩过：#907 原标题「分组聚合时自定义函数的返回值被忽略」
    # 虽然正文里有 NaN，但标题里一个关键词都没有 → 落 general。
    # 改成「分组聚合结果返回 NaN」后命中 `incorrect` 类 → 输出 Bug。
    #
    # ★ 注意这里**不要**为了让 Mock 通过而把标题写得不像人话。
    #   改后的标题「分组聚合结果返回 NaN」是真实的用户写法
    #   （用户就是会说"返回 NaN"），所以这不是"迁就 Mock"，
    #   而是"本来标题就该这么写"。
    ("#906", "时区转换结果偏移了 8 小时",
     "tz_localize 之后再 tz_convert，得到的本地时间比预期早了 8 小时，"
     "夏令时切换当天更明显。UTC 存储没问题，只是展示层不对。",
     "Bug", ["Bug"]),
    ("#907", "分组聚合结果返回 NaN",
     "groupby 之后 agg 自定义函数明明返回了标量，结果列却是 NaN。"
     "换成内置的 sum 就正常，只有自定义函数会这样。",
     "Bug", ["Bug"]),
]


#: 设计意图标签表：`issue_id -> 应该分到哪一类`（硬编码，**不依赖任何被测逻辑**）
#
# ★★ 为什么需要它 ★★
#   "一致率"不区分"两版都对"和"两版一起错" —— 两版一起错时一致率照样 100%
#   （本项目在 docs/08"坑 2"记过这个陷阱）。所以必须有一个**独立的正确性参照**，
#   否则"两版功能等价"这个结论是空的：它可能只是"两版一起废掉了"。
#
# ★★ 为什么放在**模块级字典**里，而不是只挂在 Issue 对象上 ★★
#     因为 `Issue` 会**被传进被测代码**（Agent 拿到的就是这个对象）。
#     把标准答案挂在被测对象身上，就存在"被测逻辑顺手读到答案"的风险 ——
#     主项目里 `react.py:1081` **就真的读** `issue.gold_labels` 来判误诊。
#     本实验路径今天不读它，但**"今天不读"不是保证**。
#
#     **权威的那份留在外面**（这个字典）；被测对象身上的 `gold_labels`
#     只是"让它看起来像个真实 issue"的装饰。
#     这样将来即使给实验换上真模型/真逻辑，准确率也不会被污染。
#
#     ★ 这是"参照必须来自被测对象之外"的**落实**，不只是写在注释里：
#       和 `separation_margin` 把探针收回 shared、测试里用硬编码
#       `WANT_HIT`/`WANT_MISS` 分组，是同一个原则的三次应用。
EXPECTED_LABELS: dict[str, str] = {r[0]: r[3] for r in _SAMPLE_ROWS}


def expected_label(issue_id: str) -> str | None:
    """取某个样本的设计意图标签（拿不到返回 None —— **不猜**）。"""
    return EXPECTED_LABELS.get(issue_id)


def accuracy(predicted: list[str], issue_id: str) -> bool | None:
    """Top-1 准确率：**预测的第一个标签**是否等于设计意图。

    ★★ 为什么用 Top-1 口径，而且**直接复用主项目的实现** ★★
      ① **复用，不重写** —— 直接调
         `nightwatch.eval.metrics.Metrics.top1_accuracy`。
         本项目的头号教训就是"同一件事长出两份实现"
         （`parse_action` / `_format_memory` / 探针集 / 余量口径 —— 已经四次）。
      ② **为什么不是"集合严格相等"**：分诊是多标签任务，
         一次预测里冒出多个候选是正常的（主项目自己的 `top1_accuracy`
         注释写着"不惩罚多打"）。用严格相等会把"多给了一个候选"
         判成全错 —— 那不是本实验要守的东西。
         Top-1 守的才是"**最主流的那个分类对不对**"，
         也就是"这次分诊有没有抓住重点"。
      ③ 严格口径也一并算（见 `label_accuracy`）——
         因为"多打"正是"看起来很准"的假象来源，值得对照着看。

    ★ 返回 None = "这个 issue 没有设计意图"，与 False（"答错了"）分开 ——
      "没标准答案"和"答错"是两件事（主项目 `to_coarse` 也是这个约定）。
    """
    exp = EXPECTED_LABELS.get(issue_id)
    if exp is None:
        return None
    from nightwatch.eval.metrics import Metrics
    return bool(Metrics.top1_accuracy(list(predicted or []), [exp]) == 1.0)


def accuracy_strict(predicted: list[str], issue_id: str) -> bool | None:
    """严格口径：预测集合**恰好等于**设计意图（多打少打都算错）。

    它和 Top-1 的关系是互补的，不是替代：
        Top-1 高 + 严格低  → 主体判对了，但"多打了候选"（可能是误匹配捞出来的）
        Top-1 低          → 连主要分类都没抓住，问题更靠前
    实测 #902 就是前者：Top-1 说它错了（第一个是 Bug），
    而它的预测 `['Bug','Docs']` 里那个 `Bug` 正是 `parse_final`
    子串兜底捞出来的**假阳性** —— 两个口径一起看才发现这点。
    （本项目在别处也用过"多口径并排"：主项目的严格 F1 / Top-1 / Hit@k。）
    """
    exp = EXPECTED_LABELS.get(issue_id)
    if exp is None:
        return None
    return {str(x).strip().lower() for x in (predicted or [])} == {exp.lower()}


def label_accuracy(results: dict, side: str) -> dict:
    """按 issue 汇总一侧的准确率（Top-1 为主，严格口径并排）。

    `results`: `{issue_id: RunResult}`（每个 issue 取最后一次运行）。
    """
    right, wrong, unknown = [], [], []
    strict_right = 0
    for iid, r in results.items():
        pred = getattr(r, "predicted", [])
        ok = accuracy(pred, iid)
        if ok is None:
            unknown.append(iid)
            continue
        if ok:
            right.append(iid)
        else:
            wrong.append(iid)
        if accuracy_strict(pred, iid):
            strict_right += 1
    n = len(right) + len(wrong)
    return {
        "side": side, "n": n,
        "right": right, "wrong": wrong, "unknown": unknown,
        "rate": (len(right) / n) if n else 0.0,
        # ★ 严格口径：Top-1 高而它低 = "多打了候选"（可能是误匹配）
        "strict_right": strict_right,
        "strict_rate": (strict_right / n) if n else 0.0,
    }


def equivalence_valid(consistency: float, acc_custom: float,
                      acc_graph: float, *, covered: bool = True) -> bool:
    """判定"两版功能等价"这个结论**能不能成立**。

    ★★ 三个前提缺一不可 ★★
      `covered`     —— 对照**覆盖**了要测的东西吗？
                       （循环真的跑过？没有样本两版都返回空？）
                       覆盖不足时，一致率再漂亮也证明不了等价性：
                       可能就是"两版都提前退化成同一个空答案"。
      `consistency` —— 两版**行为一致**吗？（这是通常说的"一致率"）
      `accuracy`    —— 两版**都对**吗？

    ★★ 为什么"一致率 100%"远远不够 ★★
        一致率只说明"两版行为一样"，不说明"行为正确"。
        两版可以在同一个地方一起错，然后拿到 100% 的一致率 ——
        那时说"功能等价"是**自欺**：真正的含义是"两版错得一样"。
        本项目在 docs/08"坑 2"记过这个陷阱（对称的错误最危险），
        而**这里把它变成了机器判据** —— 不再靠人记得去看。

    ★ 返回 False **不等于**"两版有差异"。要分清是哪一类失效：
        · coverage 不足    → 先修覆盖（加样本 / 修路由）
        · consistency 不满 → 编排差异真的影响了行为（看路由/兜底/reducer）
        · consistency 满但 accuracy 不满 → 两版一起错，且错在**更上游**
          （标签体系 / 解析 / 词表 —— 和编排无关）
      **三者的排查方向完全不同，所以报告里必须并排显示，只报一个都会误导。**

    ★ 这个判据放在 shared 里（而不是写在 compare.py 的打印语句里），
      是为了让测试和报告**用同一份实现** —— 否则又是"同一件事两份实现"，
      而本项目已经在 `parse_action` / `_format_memory` / 探针集 /
      余量口径上踩过四次了。
    """
    return bool(covered and consistency >= 1.0
                and min(acc_custom, acc_graph) >= 1.0)
