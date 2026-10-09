"""
Nightwatch · LLM 客户端

================================================================================
为什么要有"Mock 模式"
================================================================================
这是本项目一个刻意的设计：**没有 API key 也能完整跑通**。

理由和直觉相反，但很重要：
  1. 学习时最大的阻碍是"环境没配好就动不了"。Mock 模式让你立刻看到完整链路，
     先理解系统怎么运作，再去关心模型质量。
  2. **调试时你必须能把"模型问题"和"系统问题"分开。**
     Mock 是确定性的 —— 如果同一个输入在 Mock 下稳定、在真模型下不稳定，
     那问题出在模型；如果 Mock 下也不稳定，那是你的循环/记忆/工具层有 bug。
     没有这个对照，你会在两个层面之间反复横跳。
  3. 跑评测时 Mock 是免费的基线。任何"真实模型带来多少提升"的说法，
     都需要一个基线对照 —— 否则你没法证明提升来自模型而不是随机性。

================================================================================
关键概念：接口隔离
================================================================================
主循环不应该知道"模型是哪个供应商"。它只知道一件事：
    给我 messages，还我一个结构化的决策。

这样做的收益：换模型不用改主循环；做组件组合实验时能简单地切换配置。
这就是"依赖倒置"在 Agent 项目里的具体应用。
================================================================================
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from ..config import (
    LLM_API_KEY, LLM_BASE_URL, MODEL_ROUTING, MODELS,
    THINKING_DEFAULT, THINKING_POLICY,
)


# =============================================================================
# 返回结构
# =============================================================================

@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(Exception):
    """LLM 调用错误。

    error_kind 直接决定后续行为 —— 这是可靠性机制的第一层（重试）的输入：
        timeout / rate_limit  → 值得重试
        invalid_args / auth   → 重试没意义，快速失败
    """
    def __init__(self, msg: str, kind: str = "unknown"):
        super().__init__(msg)
        self.kind = kind


# =============================================================================
# 接口
# =============================================================================

class LLMClient(Protocol):
    def complete(self, messages: list[dict], task: str = "agent_step",
                 temperature: float = 0.0) -> LLMResponse:
        ...


# =============================================================================
# 实现一：Mock（确定性）
# =============================================================================

class MockLLM:
    """确定性的假模型。

    它不是"随便返回点东西" —— 它实现了一个**规则版 ReAct 决策器**，
    能根据当前状态合理选择下一步动作，所以整条链路（含记忆、工具、评测）
    都能在没有任何外部依赖的情况下真实跑通。

    这个类还有一个隐藏好处：它是你理解"Agent 循环到底在做什么"的最好教材。
    把这里的规则读懂，你就明白了主循环对模型的真实要求是什么 ——
    只需要它做一件事：**看当前状态，输出下一步动作**。
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._call_count = 0

    def complete(self, messages: list[dict], task: str = "agent_step",
                 temperature: float = 0.0) -> LLMResponse:
        self._call_count += 1
        text = self._route(messages, task)
        return LLMResponse(
            text=text,
            model=f"mock:{task}",
            prompt_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
            completion_tokens=len(text) // 4,
            latency_ms=1,
        )

    # --- 按任务类型分派 ---

    def _route(self, messages: list[dict], task: str) -> str:
        joined = "\n".join(m.get("content", "") for m in messages)

        if task == "agent_step":
            return self._decide_next_action(joined)
        if task == "memory_extract":
            return self._extract_memory(joined)
        if task == "memory_resolve":
            return self._resolve_conflict(joined)
        if task == "eval_judge":
            return self._judge(joined)
        return json.dumps({"result": "ok"}, ensure_ascii=False)

    def _decide_next_action(self, ctx: str) -> str:
        """规则版的任务规划：按"信息缺口"依次补齐。

        ============================================================
        注意这里的逻辑顺序 —— 它体现了真实 Agent 的思考方式
        ============================================================
        不是"猜一个答案"，而是"先搞清楚我还缺什么信息"。

        而且它示范了一个关键判断：**上下文里已经有相关信息时，不要再调工具。**
        这也是"记忆让 Agent 少走弯路"的实现方式 ——
        预取到的历史经验如果已经覆盖了问题，就不需要再检索一遍。
        """
        # --- 判断已经调过哪些工具 ---
        # 检测方式是在上下文里搜工具调用痕迹。
        # 真实模型不需要这样"自己数自己"（它从对话历史里就能看到），
        # 但规则版必须显式判断。
        called_similar = "search_similar_issues(" in ctx
        called_docs = "search_docs(" in ctx

        # --- 记忆预取命中：上下文里已有历史经验 → 直接下结论 ---
        # 这是演示"记忆效果"的关键分支：
        #   无记忆时 → 需要自己去检索（多花轮次）
        #   有记忆时 → 历史经验已经在上下文里，直接可用
        has_prefetched_memory = ("[同一提交者的历史]" in ctx
                                 or "[语义相似]" in ctx)

        if has_prefetched_memory and not called_similar:
            return json.dumps({
                "thought": "上下文已包含该提交者的历史相关经验，无需重复检索",
                "action": "finalize",
                "action_input": self._compose_answer(ctx, used_memory=True),
            }, ensure_ascii=False)

        # --- 已经有足够的检索结果 → 下结论 ---
        if called_similar and called_docs:
            return json.dumps({
                "thought": "已获得相似历史与维护规范，可以给出分类建议",
                "action": "finalize",
                "action_input": self._compose_answer(ctx, used_memory=has_prefetched_memory),
            }, ensure_ascii=False)

        # --- 下一步：查相似历史（判重的核心步骤）---
        if not called_similar:
            return json.dumps({
                "thought": "先检索历史相似问题，判断是否已存在同类 issue",
                "action": "search_similar_issues",
                "action_input": {"query": self._extract_symptom(ctx), "top_k": 5},
            }, ensure_ascii=False)

        # --- 再下一步：查维护规范（确认标签口径）---
        if not called_docs:
            return json.dumps({
                "thought": "检索维护文档，确认项目的标签分类标准",
                "action": "search_docs",
                "action_input": {"query": "label 分类标准 contributing", "top_k": 3},
            }, ensure_ascii=False)

        return json.dumps({
            "thought": "已收集足够信息，输出分类建议",
            "action": "finalize",
            "action_input": self._compose_answer(ctx, used_memory=has_prefetched_memory),
        }, ensure_ascii=False)

    def _compose_answer(self, ctx: str, used_memory: bool) -> dict:
        """规则版地"生成最终答案"。

        真实场景下这是模型综合所有信息后的输出。规则版做的事情是：
        从上下文里提取症状 → 推断标签 → 给出理由和引用。

        ============================================================
        为什么这个实现值得看
        ============================================================
        它展示了最终答案应该包含什么：
          predicted_labels —— 结论
          summary          —— 理由（**只说结论不给理由的建议没有价值**）
          cites            —— 依据（维护者要能验证）
          confidence       —— 置信度（**允许低，低就转人工**）

        缺任何一项，这个输出在生产环境里都是不合格的。

        ============================================================
        这里踩过两个坑，都是"看起来对但在真实数据上会错"的典型
        ============================================================
        坑 1：**把自己判成自己的重复。** 一开始从上下文里正则找 `#\\d+`，
              结果第一个匹配到的就是当前 issue 自己的编号，于是输出
              "与 #202 高度相似" —— 而 #202 就是它自己。
              教训：**任何从上下文里提取实体 id 的逻辑，都必须显式排除"我自己"。**
              这个坑在真实项目里也会犯（拼接历史记录时把当前记录也拼进去了）。

        坑 2：**有记忆却不用。** 记忆里明明有 verified 的标签（bug/timeout），
              但规则版仍然从自然语言症状重新推断，推不出来就退回 general。
              这在真实模型上也会犯 —— 如果记忆渲染时没把标签带出来，
              模型就只能重新猜。**所以记忆渲染的字段选择是个设计决策，不是细节。**
        """
        symptom = self._extract_symptom(ctx)
        current_id = self._extract_current_id(ctx)

        # ============================================================
        # ★ 硬约束是否还在上下文中
        # ============================================================
        # 这是长程一致性压测的核心机制。规则很简单：
        #
        #   上下文里有约束声明 → 按约束行事（给引用、置信度与证据匹配、主动判重）
        #   上下文里没有约束   → **行为漂移**（不给引用、无依据也高置信、忽略判重线索）
        #
        # 为什么要让 Mock 表现出这种差异？
        # 因为如果不这样，压测就测不出任何东西 —— 两组会得到完全一样的结果，
        # 而那个"一样"是假的（因为 Mock 本来就不读约束）。
        #
        # 这个模拟是合理的：真实模型的行为也确实是这样 ——
        # 指令在上下文里就遵守，被挤走就漂移。
        # 差别只在于真实模型的漂移更微妙（部分遵守、概率性遗忘），
        # 而 Mock 是确定性的（要么全遵守，要么全漂移）。
        #
        # ⚠️ 所以这个实验证明的是"**约束可见性**决定了行为，
        #    而状态卡保证了约束的可见性"，而不是"真实模型的漂移率是多少"。
        #    后者必须用真实模型测。这个区别在文档里要写清楚。
        has_constraint = ("硬约束" in ctx) or ("必须给出引用依据" in ctx)

        # --- 标签推断：记忆优先 ---
        # 这个顺序很重要：**已经验证过的历史标签，比重新推断更可靠。**
        # 这正是"记忆让系统更准"的具体机制。
        own_tags = self._tags(symptom)
        mem_tags = self._extract_memory_tags(ctx)

        if mem_tags and (not own_tags or own_tags == ["general"]):
            tags = mem_tags
            used_memory = True
        else:
            tags = own_tags

        labels = [f"bug/{t}" if t != "general" else "needs-triage" for t in tags]

        # --- 判重：直接读工具给出的 likely_duplicate ---
        # 为什么不自己从上下文里找 `#\d+`？
        # 见下面"坑 1"的说明 —— 那样会把自己判成自己的重复。
        # 正确做法是让工具做判定（工具知道"当前是谁"），模型只负责读结果。
        #
        # ★ 漂移模式：约束"判重优先于分类"不在上下文里时，
        #   模型就不会主动去读判重线索（因为它不再被要求优先判重）。
        import re as _re
        dup = None
        if has_constraint:
            m = _re.search(r'"likely_duplicate":\s*"(#[^"]+)"', ctx)
            if m:
                candidate = m.group(1)
                if candidate != current_id:      # ← 双保险，工具层已经排除过一次
                    dup = candidate

        # --- 引用依据 ---
        # 分两类收集：issue 引用 和 记忆引用。
        #
        # 为什么要单独区分"记忆引用"？
        # 因为**它是"记忆有没有被真的用上"这个指标的唯一数据来源**。
        # 只看"召回了 5 条"是没用的 —— 有些记忆被召回纯粹是向量相似度碰巧高，
        # 并没有参与决策（这叫"噪音吸铁石"）。
        # 只有 Agent 显式声明"我引用了哪条记忆"，才能算出无效召回率。
        import re as _re

        issue_cites: list[str] = []
        for m in _re.finditer(r'"issue_id":\s*"(#\d+)"', ctx):
            cid = m.group(1)
            if cid != current_id and cid not in issue_cites:
                issue_cites.append(cid)
        for m in _re.finditer(r"来源 issue\s*#(\d+)", ctx):
            cid = f"#{m.group(1)}"
            if cid != current_id and cid not in issue_cites:
                issue_cites.append(cid)

        mem_cites: list[str] = []
        for m in _re.finditer(r"记忆ID=(ep_[0-9a-f]+)", ctx):
            mid = m.group(1)
            if mid not in mem_cites:
                mem_cites.append(mid)

        cites: list[str] = issue_cites[:2]
        # 只有真的用了记忆，才引用它 —— 没用上却引用，是在虚报命中率
        if used_memory and mem_cites and mem_cites[0] not in cites:
            cites.append(mem_cites[0])
        if "CONTRIBUTING" in ctx:
            cites.append("CONTRIBUTING.md")

        # ★ 漂移模式：约束"必须给出引用依据"不在上下文里时，就不附引用
        if not has_constraint:
            cites = []

        reason_bits = [f"症状特征匹配 {', '.join(tags)}"]
        if used_memory and mem_cites:
            reason_bits.append(f"已参考历史经验（记忆 {mem_cites[0]}）")
        elif used_memory:
            reason_bits.append("已参考该提交者的历史经验")
        if dup:
            reason_bits.append(f"与 {dup} 高度相似，建议按重复处理")

        # 置信度：有依据 → 高；没有 → 低（低置信度会触发转人工）
        confidence = 0.75 if cites else 0.4
        if not symptom or symptom in ("未知症状", "用不了", "有 bug"):
            confidence = 0.2      # 信息不足，明确低置信

        # ★ 漂移模式：约束"置信度与证据匹配"不在上下文里时，
        #   就会在无依据的情况下给出高置信度 —— 这是最危险的漂移形式，
        #   因为它不只是"漏了一件事"，而是"自信地做了错事"。
        if not has_constraint:
            confidence = 0.85

        # ============================================================
        # 为什么输出里必须带"根因"
        # ============================================================
        # 这是从诊断里发现的一个真问题：最初的输出只有"标签 + 建议"，
        # 结果记忆抽取时抽不到 root_cause 字段 → causal_depth 只有 0.3
        # → 记忆总分 0.535 < 阈值 0.55 → **所有记忆都被筛掉了，记忆系统等于不存在**。
        #
        # 这暴露了一条链式依赖：
        #     主循环的输出结构  →  记忆抽取能拿到什么  →  记忆的评分  →  能不能被写进库
        #
        # 也就是说：**上游少输出一个字段，下游一个模块就整个失效。**
        # 这种问题在真实项目里很常见，而且是"系统能跑但没效果"的典型原因 ——
        # 没有 trace 就永远发现不了。
        #
        # 教训：Agent 的最终输出结构不是"随便写写"，它同时是下游记忆系统的输入契约。
        root_cause = self._guess_root_cause(tags)

        return {
            "predicted_labels": labels,
            "duplicate_of": dup or "",
            "root_cause": root_cause,
            "summary": (f"根因：{root_cause}。"
                        + "；".join(reason_bits)
                        + "。建议：" + (
                            f"按 {labels[0]} 处理，参考历史处置方式"
                            if labels else "转人工确认")),
            "cites": cites,
            "confidence": confidence,
        }

    @staticmethod
    def _guess_root_cause(tags: list[str]) -> str:
        """规则版根因推断。

        Mock 当然无法真的定位根因 —— 但**它必须输出一个"看起来像根因"的字段**，
        因为下游的记忆抽取需要它，而且真实模型也确实会输出类似内容。

        这里的映射表来自对这类故障的常见归因。真实场景下这应该是模型
        结合日志、代码、历史 case 综合推断的结果。

        注意措辞用了"疑似" —— 因为**未经证实的根因不应该被表述为确定事实**。
        这个细节在记忆系统里很重要：记忆里存的如果是一个被当成事实的错误根因，
        它会污染后续所有相关判断。
        """
        table = {
            "timeout": "疑似下游响应慢或调用方超时设置不合理",
            "oom": "疑似内存未释放或数据量超出预估",
            "permission": "疑似凭证 scope 不足或角色未授权",
            "not_found": "疑似资源未创建或路径配置错误",
            "crash": "疑似未处理的边界输入或空指针",
            "config": "疑似配置加载顺序问题或环境变量未注入",
            "dependency": "疑似依赖版本冲突或运行环境不匹配",
            "duplicate": "疑似与已有问题重复",
            "feature": "非缺陷，属于功能请求",
            "general": "证据不足，未能定位根因",
        }
        for t in tags:
            if t in table:
                return table[t]
        return table["general"]

    @staticmethod
    def _extract_current_id(ctx: str) -> str:
        """提取"当前正在处理的 issue 编号"。

        这个信息在任务状态卡里（state_card 的"编号"字段），
        所以优先从那里取 —— 比从整段上下文里瞎找可靠得多。
        """
        import re as _re
        m = _re.search(r'"编号":\s*"(#?\d+)"', ctx)
        if m:
            v = m.group(1)
            return v if v.startswith("#") else f"#{v}"
        m = _re.search(r"issue[_\s]*id[\"'\s:：]+(#?\d+)", ctx, _re.IGNORECASE)
        if m:
            v = m.group(1)
            return v if v.startswith("#") else f"#{v}"
        return ""

    @staticmethod
    def _extract_memory_tags(ctx: str) -> list[str]:
        """从渲染后的记忆文本里提取标签。

        对应 react.py 的 _format_memory 里输出的 `标签: timeout, performance` 这一行。
        """
        import re as _re
        m = _re.search(r"标签[:：]\s*([a-zA-Z0-9_,\s/]+)", ctx)
        if not m:
            return []
        return [t.strip() for t in m.group(1).split(",") if t.strip()]

    @staticmethod
    def _extract_symptom(ctx: str) -> str:
        """从上下文里提取症状描述。

        ============================================================
        为什么要合并"标题 + 异常类型"，而不是只用标题
        ============================================================
        只用标题会丢信息。对比这个例子：

            标题: "升级 Python 版本后无法导入模块"
            堆栈: "ImportError: cannot import name 'cached_property'"

        只看标题，关键词表里匹配不到任何东西 → 归类成 general → 转人工。
        但异常类型 ImportError 是很强的信号，它明确指向依赖问题。

        而且这个问题在真实模型上同样存在 —— 如果症状文本里没有关键信息，
        模型也只能瞎猜。**所以"症状怎么组装"是一个会直接影响准确率的决策**，
        不是无关紧要的实现细节。

        正确的做法：标题（人类的高层描述）+ 异常类型（机器的精确信号）都要。
        这也解释了为什么 `Issue.search_text()` 要把 title 和 stack_trace 拼在一起。
        """
        import re as _re
        title = ""

        # 优先：任务状态卡里的标题（最可靠，因为是我们自己放进去的）
        m = _re.search(r'"标题":\s*"([^"]{2,120})"', ctx)
        if m:
            title = m.group(1)
        else:
            m = _re.search(r"^标题[:：]\s*(.+)$", ctx, _re.MULTILINE)
            if m:
                title = m.group(1).strip()[:120]
            else:
                m = _re.search(r"^症状[:：]\s*(.+)$", ctx, _re.MULTILINE)
                if m:
                    title = m.group(1).strip()[:120]

        # 补充：从**当前 issue 的堆栈**里抽异常类型
        #
        # ★ 注意这里的取值范围 —— 只从状态卡的"报错堆栈"字段里找，
        #   绝不能从整段上下文里正则搜索。
        #
        # 这是踩过的坑：早期实现用 `re.search(异常类型, ctx)` 全局搜，
        # 结果抓到了**历史 issue 的堆栈**。实测表现是一个内容只有"用不了"
        # 的 issue，因为上下文里有别的 issue 的 PermissionError，
        # 被误判成权限问题。
        #
        # 教训：**在拼接了多个信息源的上下文里做提取，必须限定取值范围。**
        #   全局搜索在单文档场景下没问题，在多文档场景下一定会串。
        stack = ""
        m_stack = _re.search(r'"报错堆栈":\s*"([^"]{0,400})"', ctx)
        if m_stack:
            stack = m_stack.group(1)

        exc = ""
        if stack:
            m2 = _re.search(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Fault))\b",
                            stack)
            if m2:
                exc = m2.group(1)

        combined = f"{title} {exc}".strip()
        return combined or "未知症状"

    def _extract_memory(self, ctx: str) -> str:
        """规则版记忆抽取。

        真实场景这里是小模型的结构化输出任务。规则版的目的是：
        让"记忆抽取"这个环节在没有模型时也能产生合理结果，从而能测整条链路。

        **注意这里踩过一个坑，值得记下来：**
        一开始用 `re.DOTALL` 匹配字段，结果 `.` 跨过了换行，
        把整篇提示词都吞进了 symptom 字段，导致记忆内容全是垃圾。
        教训：**解析结构化文本时，字段边界要显式限定在行内**，
        不要指望"贪婪/非贪婪"去猜边界。
        """
        # 用 [^\n] 显式限定在行内，避免跨字段污染
        symptom = self._grab(ctx, r"(?:症状|symptom)[:：][ \t]*([^\n]{2,150})") or "未知症状"
        root = self._grab(ctx, r"(?:根因|root.?cause)[:：][ \t]*([^\n]{2,150})")
        resolution = self._grab(ctx, r"(?:建议|resolution)[:：][ \t]*([^\n]{2,200})")

        return json.dumps({
            "symptom": symptom.strip(),
            "symptom_tags": self._tags(symptom),
            "root_cause": root.strip(),
            "resolution": resolution.strip(),
            "reusability": 0.7,
            "causal_depth": 0.8 if root else 0.3,
            "non_triviality": 0.6,
        }, ensure_ascii=False)

    def _resolve_conflict(self, ctx: str) -> str:
        return json.dumps({
            "winner": "B",
            "reason": "新记忆有工具调用日志佐证，证据强度更高",
            "confidence": 0.72,
        }, ensure_ascii=False)

    def _judge(self, ctx: str) -> str:
        """规则版裁判。

        注意：它拆成了**可判定的子项**，而不是给一个"总体好不好"的分数。
        这直接回应了那个面试问题："LLM-as-Judge 的偏差你怎么处理？"
        答案的第一条就是：把主观评分改成可判定的子项，子项比整体判断稳定得多。
        """
        has_label = bool(re.search(r"(?:建议标签|labels?)[:：]", ctx))
        has_reason = bool(re.search(r"(?:理由|reason|因为)", ctx))
        has_citation = bool(re.search(r"(?:#\d+|issue\s*\d+|引用)", ctx))
        score = 0.4 * has_label + 0.3 * has_reason + 0.3 * has_citation
        return json.dumps({
            "has_label": has_label,
            "has_reason": has_reason,
            "has_citation": has_citation,
            "score": round(score, 2),
        }, ensure_ascii=False)

    # --- 小工具 ---

    @staticmethod
    def _grab(text: str, pattern: str) -> str:
        """从文本里抓一个字段。

        **不要加 re.DOTALL** —— 那会让 `.` 跨行，把后面的字段一起吞掉。
        这里的 pattern 都用 `[^\\n]` 显式限定在行内。
        """
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _tags(symptom: str) -> list[str]:
        """从症状文本里提取粗粒度标签（规则版）。

        关键词表的覆盖度直接决定了能识别多少类型。
        真实场景里这个表应该从项目的**历史标签分布**里统计出来，
        而不是拍脑袋写 —— 这也是"评测驱动设计"的一个应用：
        先看你的数据里实际有哪些类型，再决定识别哪些。

        **注意 feature 类的冲突处理：**
        "建议"这个词在 issue 文本里很常见，但有两种完全不同的用法：
            标题「建议增加暗色主题」     → 这是功能请求（feature）
            正文「建议修复一下超时问题」  → 这是 bug 报告里的建议，不是 feature
        所以规则是：**如果同时匹配到 bug 类和 feature，优先 bug 类。**
        这个"优先级"是必须显式写的 —— 否则分类会被高频词带偏。
        """
        bug_table = {
            "timeout": ["超时", "timeout", "timed out", "无响应", "卡住",
                        "hang", "504", "未响应"],
            "oom": ["内存", "oom", "memory", "out of memory"],
            "permission": ["权限", "permission", "denied", "403", "forbidden",
                           "无权限", "越权"],
            "not_found": ["不存在", "not found", "404"],
            "crash": ["崩溃", "crash", "segfault", "异常退出", "aborted"],
            "config": ["配置", "config", ".env", "环境变量", "部署"],
            "dependency": ["依赖", "dependency", "install", "版本冲突",
                           "importerror"],
            "duplicate": ["重复", "duplicate"],
        }
        feature_words = ["希望", "建议", "支持", "增加", "enhancement",
                         "feature", "request", "能否", "可以加", "需求"]

        low = symptom.lower()
        hits = [k for k, words in bug_table.items() if any(w in low for w in words)]

        if hits:
            return hits                       # 优先 bug 类
        if any(w in low for w in feature_words):
            return ["feature"]
        return ["general"]


# =============================================================================
# 实现二：OpenAI 兼容接口（真实模型）
# =============================================================================

class OpenAICompatLLM:
    """任何兼容 OpenAI /chat/completions 协议的服务都能用。

    为什么选这个协议而不是各家 SDK？
    因为它是事实标准 —— 阿里百炼、DeepSeek、Moonshot、vLLM 本地部署都兼容。
    只用 HTTP，不引入供应商 SDK，换供应商时改一个环境变量就行。

    ============================================================
    ★ 思考链：按任务分档控制（不是一刀切）
    ============================================================
    起因：用户把模型换成 `qwen3.8-flash`（本应更快），结果反而更慢
    （单条 triage 从 17 秒涨到 57~241 秒）。

    诊断（scripts/diagnose_latency.py）找到了根因：**它默认开启思考链**，
    输出 token 多约 10 倍（987 vs 100），把生成速度的优势彻底吃掉了。

    但**"关掉思考"不是无脑优化** —— 这点花了两轮实验才搞清楚：

        配置                        步数        平均 F1
        qwen-max（无思考）           2~3        0.734   ← 基线
        flash + 思考                3~7        0.494
        flash 无思考 + 常规提示词    5/20/9/12  0.46
        flash 无思考 + 强化提示词    **20/20**  **≈0.00**  ← 最差

    关掉思考后困难 case 全失败，而且加"强化提示词"也救不回来。
    根因：**提示词能替代「流程」，但不能替代「判断」。**

    所以本类的策略是**按任务分档**（见 config.THINKING_POLICY）：

        需要判断力的任务（分诊、路由、矛盾裁决）  → 开思考
        判断路径可枚举的任务（结构化抽取、子项检查）→ 关思考

    这样既能省掉"不需要思考"的那部分开销，又不会牺牲"需要判断"的质量。

    `disable_thinking` 参数：
        None  → 按 config.THINKING_POLICY 逐任务决定（**推荐，默认**）
        True  → 全部关掉（仅供 A/B 实验用）
        False → 全部开启（仅供 A/B 实验用）
    """

    # 各家的"关闭思考"参数名不一样，全试一遍（多传未知参数一般会被忽略）
    THINKING_OFF_PARAMS = (
        {"enable_thinking": False},
        {"thinking": {"type": "disabled"}},
    )

    def __init__(self, base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY,
                 max_retries: int = 3, disable_thinking: bool | None = None,
                 force_tier: str | None = None):
        # force_tier：把整条链路固定在某一档（实验用，见 _model_for 的说明）
        self.force_tier = force_tier
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self._thinking_off_verified = False   # 只探测一次，避免每次都试

    def complete(self, messages: list[dict], task: str = "agent_step",
                 temperature: float = 0.0) -> LLMResponse:
        model = self._model_for(task)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        started = time.time()

        # ★ 按任务决定要不要关掉思考（见类文档 + config.THINKING_POLICY）
        disable_thinking = self._should_disable_thinking(task)

        # 延迟导入：没装 httpx 也能 import 这个模块（Mock 模式不需要它）
        try:
            import httpx
        except ImportError as e:
            raise LLMError("需要安装 httpx 才能调用真实模型：pip install httpx",
                           kind="invalid_args") from e

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._post(httpx, payload, disable_thinking)
                if resp.status_code == 429:
                    raise LLMError("rate limited", kind="rate_limit")
                if resp.status_code >= 500:
                    raise LLMError(f"server error {resp.status_code}", kind="connection")
                if resp.status_code >= 400:
                    # ★ 如果是因为"关思考"的参数不被支持，去掉它重试一次
                    if disable_thinking and not self._thinking_off_verified \
                            and resp.status_code == 400:
                        self._thinking_off_verified = True
                        disable_thinking = False
                        continue
                    # 4xx（除 429）是客户端问题，重试没有意义 —— 快速失败
                    raise LLMError(f"bad request {resp.status_code}: {resp.text[:200]}",
                                   kind="invalid_args")

                data = resp.json()
                usage = data.get("usage", {}) or {}
                msg = data["choices"][0].get("message", {}) or {}

                # ★ 如果服务端仍然返回了思考内容，说明这个模型的思考关不掉。
                #   记录下来（不报错）—— 因为那是模型特性，不是我们的 bug。
                if disable_thinking and (msg.get("reasoning_content") or ""):
                    self._thinking_off_verified = True

                return LLMResponse(
                    text=msg.get("content") or "",
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_ms=int((time.time() - started) * 1000),
                )
            except LLMError as e:
                # 不可重试的错误直接抛
                if e.kind in ("invalid_args", "auth"):
                    raise
                last_err = e
            except Exception as e:  # 网络层异常
                last_err = LLMError(str(e), kind="timeout")

            # 指数退避。真实系统里这里还会加 jitter（随机抖动），
            # 避免多个实例同时重试造成"惊群"
            if attempt < self.max_retries - 1:
                time.sleep(min(0.5 * (2 ** attempt), 8.0))

        raise last_err or LLMError("unknown failure")

    def _should_disable_thinking(self, task: str) -> bool:
        """按任务决定要不要关掉思考链。

        优先级：
          1. 实例级显式设置（`disable_thinking=True/False`）→ 覆盖一切
             这是为了做 A/B 实验（比如"全部关掉"对比"全部开启"）
          2. 否则查 `config.THINKING_POLICY`（按任务）
          3. 表里没有的任务 → `THINKING_DEFAULT`（默认开，保守）

        **注意默认是"开"而不是"关"** ——
        因为关掉的风险（困难 case 失败）比多花点 token 严重得多。
        省成本不能牺牲正确性。
        """
        if self.disable_thinking is not None:
            return self.disable_thinking          # 实验用的强制覆盖
        return not THINKING_POLICY.get(task, THINKING_DEFAULT)

    def _post(self, httpx, payload: dict, disable_thinking: bool):
        """发一次请求，必要时带上"关闭思考"的参数。

        ★ 为什么要"全试一遍"而不是只传一个？
        因为不同供应商/不同版本的参数名不一样：
          · 阿里百炼 Qwen3 系列：enable_thinking
          · 部分网关：thinking: {type: disabled}
          · OpenAI o 系列：reasoning_effort
        多传一个未知参数，服务端通常会忽略（而不是报错）——
        所以这里用"都带上"的策略，最省事也最兼容。

        如果服务端因为多了参数而报 400，调用方会去掉参数重试一次。
        """
        body = dict(payload)
        if disable_thinking:
            for params in self.THINKING_OFF_PARAMS:
                body.update(params)

        return httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120.0,
        )

    def _model_for(self, task: str) -> str:
        """任务 → 模型档次的映射。这是 config.MODEL_ROUTING 的消费点。

        ★ `force_tier` 的用途：**组件组合实验（P1-3）需要固定模型档次**。

        正常情况下"按任务匹配模型"是对的（agent_step 用 strong、
        memory_extract 用 small）。但做实验时，我需要把**整个链路**
        固定在某一档上，否则"换模型"这个因子就被路由策略混淆了 ——
        你无法区分"效果变化"是来自模型档次，还是来自某个任务换了档。
        """
        tier = self.force_tier or MODEL_ROUTING.get(task, "strong")
        return MODELS.get(tier, tier)


# =============================================================================
# 工厂：根据配置返回实现
# =============================================================================

def build_llm(use_mock: bool = False, force_tier: str | None = None,
              disable_thinking: bool | None = None) -> LLMClient:
    """构造 LLM 客户端。

    默认策略：**没有 API key 就自动用 Mock**。
    这个"优雅降级"是刻意的 —— 让学习者能在零配置下跑通全链路。

    `force_tier` / `disable_thinking` 供组件组合实验（P1-3）固定配置用 ——
    正常情况下这两个参数不该传（让路由策略和思考链策略自己决定）。
    """
    if use_mock or not LLM_API_KEY:
        return MockLLM()
    return OpenAICompatLLM(force_tier=force_tier,
                           disable_thinking=disable_thinking)
