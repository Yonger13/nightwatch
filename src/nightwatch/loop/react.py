"""
Nightwatch · ReAct 主循环（手写）

================================================================================
为什么手写，不用 LangGraph / CrewAI
================================================================================
这是本项目一个刻意的、且经过论证的决定：

  1. **框架会把循环藏起来。** 而循环是 Agent 的心脏 ——
     面试问"Agent 一次任务经历几次模型调用、每轮传什么进去"时，
     套框架的人答不上来，因为他没见过那个 while。
  2. 手写之后，**每一轮传什么上下文、什么时候压缩、失败怎么恢复**
     全部在你手里。这三个恰恰是真正的难点，而框架帮你做掉的不是难点，
     是体力活。
  3. 参考调研数据：6/7 份 JD 提到框架但都放在"及格线"，
     而 4 场公开面经里只有 1 场问框架对比，**3 场把记忆/循环摊开问**。

不是说框架不好，是说：**这个阶段你需要理解循环，而不是使用循环。**
懂了原理之后，用框架只是节省体力。

================================================================================
循环的结构（每轮做五件事）
================================================================================
    ┌─ 1. 组装上下文（预算调度：该卸载的卸载、该压缩的压缩）
    │
    ├─ 2. 调模型 → 得到"思考 + 下一步动作"
    │
    ├─ 3. 可靠性检查（熔断？循环？超轮数？）
    │       ↓ 通过
    ├─ 4. 执行工具 → 得到观察结果 → 写回上下文
    │       ↓ 不通过
    │       └─ 把"为什么不能调"作为观察结果告诉模型（让它换个策略）
    │
    └─ 5. 记录 trace（可观测 + 评测数据源）
            ↓
       动作是 finalize？→ 结束
       超过 max_steps？→ 兜底产出（带证据的降级）

================================================================================
一个关键设计：记忆预取发生在循环**之前**，不是循环里的工具调用
================================================================================
多数项目的做法是把"检索"当成一个普通工具，在循环内部调用。
本项目不用这个做法，原因是：

    **记忆决定的是"我该怎么理解这个问题"，属于上下文准备，不是动作。**

区别在实践中很明显：
  - 作为工具调用：要先花一轮让模型"决定去查记忆"，然后才能理解问题 —— 浪费一轮
  - 作为上下文预取：问题一进来，相关的历史经验已经在上下文里了，
    模型第一轮就是在"带着背景"思考问题

这个区分能体现你对 Harness 的理解层次，面试时可以主动讲。
================================================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Settings, load_settings
from ..context.budget import ContextBudgetManager, estimate_tokens
from ..llm.client import LLMClient, build_llm
from ..memory.lifecycle import Extractor, Resolver, Scorer, Writer
from ..memory.store import MemoryStore
from ..models import (
    HumanVerdict, Issue, IssueState, Outcome, Step, ToolCall, Trace, _hash,
)
from ..tools.registry import ToolRegistry, ToolResult
from .reliability import ReliabilityGuard


# =============================================================================
# 系统提示词
# =============================================================================
# 这段文字本身就是设计的一部分，不是随手写的。
#
# 三个设计点：
#   ① 明确输出格式（JSON）—— 否则解析会很难做
#   ② 明确"可以不确定"—— 否则模型会硬编一个答案，而"不确定"是重要信号
#   ③ 明确"不要重复调用"—— 这是对循环的软性预防（硬性预防在 reliability.py）

SYSTEM_PROMPT = """你是开源项目的问题处理助手。你的任务是阅读一个新提交的 issue，
给出分类建议和处理建议，供项目维护者决策。

## 你的能力边界
- 你能查询历史 issue、检索相似问题、读代码、查维护文档
- 你能给 issue 打标签、发评论，但这些都是**建议** —— 最终由维护者确认
- 你无法直接关闭 issue，也无法修改别人的评论

## 判断准则
1. **先查再判。** 不要凭标题猜测，先检索历史 issue 确认是否已存在同类问题。
2. **判重优先于分类。** 如果发现高度相似的历史 issue，首要结论应该是
   "重复于 #xxx"，而不是给它一个新标签。
3. **允许不确定。** 如果信息不足以判断，明确说"证据不足，建议人工确认"，
   不要编造一个看起来合理的根因。
4. **不要重复调用同一个工具。** 如果上一次调用没有给你新信息，换一个策略，
   或者基于现有信息给出结论。
5. **给出依据。** 每条结论都要说明来自哪里（引用 issue 编号或文档名）。

## 输出格式
每次回复必须是一个 JSON 对象，不要输出任何其他内容：

{{"thought": "你的推理（一句话）",
 "action": "工具名，或 finalize 表示结束",
 "action_input": {{...工具参数...}}}}

当 action 为 "finalize" 时，action_input 格式为：
{{"predicted_labels": ["标签1", "标签2"],
 "duplicate_of": "重复的 issue 编号，没有则留空",
 "summary": "给维护者的处理建议",
 "cites": ["引用的 issue 编号或文档名"],
 "confidence": 0.0}}

## 可用工具
{tools}
"""


# =============================================================================
# 调优版系统提示词（针对"多打标签"的失败模式）
# =============================================================================
# ★ 由来：实测发现 flash 的 F1 只有 0.508，而失败模式高度一致 ——
#   **每条都多打 1~3 个标签**：
#
#     #64546  预测=Bug,Visualization,Error Reporting     gold=Bug,Visualization  多 1 个
#     #66550  预测=Bug,Internals                         gold=Bug                多 1 个
#     #67970  预测=Performance,Internals                 gold=Performance        多 1 个
#     #65914  预测=Bug,Deprecate,Warnings,Visualization  gold=Bug                多 3 个
#     #66475  预测=Bug                                   gold=（空，应转人工）   该说不知道却硬猜
#
# 诊断出的三个根因：
#   ① 模型不知道"该打几个标签" —— 没有数量预期，就倾向多给（宁多勿少）
#   ② 模型分不清"主标签"和"细分标签" —— 用 Error Reporting 表达"这是个 bug"
#   ③ 模型不敢说"信息不足" —— 硬猜一个 Bug，而不是承认判断不了
#
# 所以这个版本的提示词**针对性地写死这三条**：
#   · 给出真实的标签数量分布（用数据说话，而不是干说"少打点"）
#   · 列出主标签 / 细分标签两级，明确"优先用主标签"
#   · **给"信息不足"正名** —— 明确说"输出空标签是正确的做法"
#
# 最后一条尤其重要：如果提示词不明确允许"不回答"，模型会倾向给一个
# 看起来合理的答案。而**在分流场景下，"该转人工时转人工"比"硬猜一个"正确得多**。

SYSTEM_PROMPT_TUNED = """你是开源项目的问题处理助手。阅读 issue，给出**类型标签**建议。

## 输出标签的规则（最重要，请严格遵守）

### 规则 1：必须给出结论，且至少打 1 个主标签
**这是硬性要求。** 除非满足规则 3 的"信息不足"条件，否则你必须至少给出 1 个主标签。

该项目的真实标签数量分布是：
    1 个标签 14%　|　2 个标签 44%　|　3 个标签 22%　|　4 个及以上 8%

所以：**默认打 1 个最核心的主标签就够了**，最多 2~3 个。
不要打 4 个以上 —— 那几乎一定是过度标注。

### 规则 2：优先用「主标签」，不要用「细分标签」
可用标签分两级：
  · **主标签（强烈优先）**：Bug / Docs / Enhancement / Performance / Regression /
    **Build / Deprecate** 等 —— 回答"这是什么类型的问题"
  · 细分标签（仅当非常明确时才用）：Warnings / Error Reporting / Internals /
    Testing / CI / Refactor 等 —— 它们是主标签的下级细分

如果只是想表达"这是个缺陷"，请用 `Bug`，**不要用 `Error Reporting`**。

★ 关于 `Deprecate`（来自真实运行的教训）：
  它和 `Bug` / `Enhancement` **平级** —— 是"类型标签"而不是"细分标签"。
  涉及"某行为应该被弃用 / 某 API 要移除 / 某写法不再支持"的 issue
  **都应该用 `Deprecate`**，哪怕正文读起来像讨论或改进建议。

### 规则 2.5：小心标题前缀的陷阱（★ 这是从真实数据里学到的）
issue 标题常带前缀（`BUG:` / `CI:` / `BLD:` / `DOC:` / `ENH:` / `PERF:`），
但**前缀不总是等于类型标签**：

  · **`BUG:` 只表示"这是一个缺陷报告"，不表示"类型是 Bug"**。
    如果这个缺陷涉及**构建 / CI / 打包 / 依赖**，应该打 `Build`。
    例：`BUG: validate_test_dependencies.py is blind to pixi ...`
    → 这是测试依赖问题，正确标签是 **Build**，不是 Bug。

  · **`CI:` / `BLD:` 前缀 → `Build`**
    例：`CI: pixi-version is pinned in three places`
    → 正确标签是 **Build**

  · **`DEPR:` 前缀 → `Deprecate`**
    例：`DEPR: setting a 2-D value into a single row`
    → 正确标签是 **Deprecate**（涉及弃用 API / 移除旧行为）

    注意：这类 issue 的正文常是**代码示例 + 行为论证**，
    读起来像"讨论"或"改进建议" —— **但前缀已经说明它是弃用提案**。
    **以标题前缀为准，不要被正文的语气带偏。**

  · `DOC:` → `Docs`　`ENH:` → `Enhancement`　`PERF:` → `Performance`
  · `TST:` → `Testing`　`REF:` → `Refactor`

**判断类型要看 issue 的内容（改的是什么），不要只看前缀。**
**但反过来：如果前缀是一个明确的类型标记（如 `DEPR:`），要优先采纳它。**

★ 一个真实案例（来自每日运行的对账）：
```
#70257  DEPR: setting a 2-D value into a single row
        Agent 预测 Enhancement，真实标签 Deprecate  ← 错在这里
        原因：正文是代码示例 + 论证，读起来像"改进讨论"，
              而 `DEPR:` 这个明确的信号被忽略了
```

### 规则 3：只有这几种情况才算「信息不足」
本章节的"不打标签"是**例外**，不是常态。**只有**当 issue 同时满足以下条件时，
才可以输出空数组 `[]` 并设置信度 ≤ 0.2：
  · 正文少于 20 个字，且
  · 没有报错信息 / 堆栈，且
  · 标题本身也没说清是什么问题（例如只有「用不了」「有 bug」「帮我看看」）

**除此之外的任何情况，你都必须给出至少 1 个主标签。**
只要标题或正文描述了具体现象（报错、行为异常、期望的功能），就不算"信息不足"。
**信息不够完整不等于信息不足** —— 大多数真实 issue 都是不完整的，
按最接近的类型归类即可，这是正常做法。

### 规则 4：最多检索 2 次，然后必须给结论
用 search_similar_issues 和 search_docs 各查 1 次就够了。
**查完就下结论，不要换关键词反复检索。**
反复检索试图找到"更确定的答案"，比直接按现有信息判断更糟 ——
浪费预算，而且通常也找不到。
如果检索没给出明确依据，就基于 issue 本身的内容判断，置信度设 0.5 左右。

### 规则 5：只从「允许使用的标签」里选
不要发明新标签；不要输出子模块名（如 Indexing / IO CSV / Datetime / Groupby）。

## 判断依据（按优先级）
1. **issue 的标题和报错信息** —— 包含最多判别信息，优先看这个
2. **search_similar_issues 的返回** —— 如果找到高度相似的历史 issue，参考它的标签
3. **search_docs 的规范** —— 确认标签口径

## 输出格式（每次回复必须严格遵守）
每次回复必须是一个 JSON 对象，不要输出任何其他内容（不要解释、不要 markdown 代码块）：

{{"thought": "你的推理（一句话）",
 "action": "工具名，或 finalize 表示结束",
 "action_input": {{...工具参数...}}}}

**判断完就直接用 `finalize` 结束**，action_input 格式为：
{{"predicted_labels": ["标签1", "标签2"],
 "duplicate_of": "重复的 issue 编号，没有则留空",
 "summary": "给维护者的处理建议",
 "cites": ["引用的 issue 编号或文档名"],
 "confidence": 0.0}}

## 可用工具
{tools}
"""


# =============================================================================
# 强化版系统提示词（用于"关掉思考链"的场景）
# =============================================================================
# ★ 由来：实测发现关掉思考链后，简单任务快 9 倍，但**困难任务会陷入试错循环**
#   （步数从 3 涨到 20，触顶后彻底失败）。
#
# 根因：思考链的作用是"自己想清楚该怎么做"。关掉它之后，
# 模型失去了这个规划能力，只能靠"试一次、不行再试"来摸索 —— 于是步数暴增。
#
# 所以解法不是"关了思考就完事"，而是**用显式的流程约束替代它本来要自己推理的东西**：
#   · 思考链会想到"我该先查历史再判断"   → 这里直接写成第 1 步 / 第 2 步
#   · 思考链会想到"别重复调工具了"       → 这里写死"每个工具只用一次"
#   · 思考链会想"想不出来就别硬凑"       → 这里明说"不确定就转人工，这比反复检索正确"
#
# 一句话：**关掉模型的思考，就要由提示词来承担"规划"这份工作。**
#
# 这是从"依赖模型能力"转向"依赖系统设计"的一个具体例子 ——
# 也是这个项目一直在做的事。

SYSTEM_PROMPT_STRICT = """你是开源项目的问题处理助手。阅读一个新提交的 issue，
给出分类建议和处理建议，供项目维护者决策。

## 执行流程（按顺序执行，不要跳步，不要多做）
第 1 步：调用 search_similar_issues 检索历史相似问题 —— **只调 1 次**
第 2 步：调用 search_docs 查维护规范确认标签口径 —— **只调 1 次**
第 3 步：**立即**输出 finalize 给出结论。不要调第三个工具。

## 硬性规则（违反会导致任务失败）
1. **工具调用总数最多 3 次。** 执行完上面的流程就停。
2. **每个工具只用一次。** 绝不用相同或相似的参数重复调用同一个工具。
3. **检索不到就停。** 如果 search_similar_issues 没返回相关结果，
   不要换成别的关键词再查 —— 直接基于 issue 本身的内容判断。
4. **不确定就给低置信度并转人工。** 这是正确的做法，不是失败。
   反复检索试图找到答案，比直接说「信息不足，建议人工确认」更糟。
5. 每条结论必须给出引用依据（issue 编号或文档名）；没有依据时降低置信度。

## 输出格式（每次回复必须严格遵守）
每次回复必须是一个 JSON 对象，不要输出任何其他内容：

{{"thought": "你的推理（一句话）",
 "action": "工具名，或 finalize 表示结束",
 "action_input": {{...工具参数...}}}}

当 action 为 "finalize" 时，action_input 格式为：
{{"predicted_labels": ["标签1", "标签2"],
 "duplicate_of": "重复的 issue 编号，没有则留空",
 "summary": "给维护者的处理建议",
 "cites": ["引用的 issue 编号或文档名"],
 "confidence": 0.0}}

## 可用工具
{tools}
"""


# =============================================================================
# 无约束版系统提示词（仅供对照实验使用）
# =============================================================================
# ★ 这个版本的由来，是本项目最值得记录的一次"实验设计缺陷"：
#
# 长程一致性压测的第一版，对照组只把"任务状态卡"关掉了，
# 结果在真实模型上几乎测不出差异（两组都 100%）。
#
# 根因：**上面那个 system prompt 里本来就写着 5 条判断准则**，
# 而 system 桶是**不可压缩**的 —— 所以约束始终在上下文里。
# 对照组不是"没有约束"，只是"少了一份每轮重申的强调"。
#
# 教训：
#   设计对照实验时，必须把**这个变量在所有位置的出现**都列出来。
#   我改了状态卡，却忘了 system prompt 里还有一份 ——
#   这和第 D2 号缺陷（对照组混淆了任务信息）是同一类错误的另一个变体：
#   **你以为只改了一个地方，其实同一个东西在别处还有一份副本。**
#
# 所以这里提供一个"真的没有约束"的版本，用来做第三组对照。
# 注意：**生产环境不应该用这个版本** —— 它只是实验工具。

SYSTEM_PROMPT_NO_CONSTRAINTS = """你是开源项目的问题处理助手。你的任务是阅读一个新提交的 issue，
给出分类建议和处理建议，供项目维护者决策。

## 可用工具
{tools}
"""


# =============================================================================
# 结果
# =============================================================================

@dataclass
class TriageResult:
    """一次处理的最终产出。"""
    issue_id: str
    predicted_labels: list[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None
    summary: str = ""
    cites: list[str] = field(default_factory=list)
    confidence: float = 0.0
    outcome: Outcome = Outcome.UNFINISHED
    n_steps: int = 0
    elapsed_ms: int = 0
    trace: Optional[Trace] = None
    fallback: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "predicted_labels": self.predicted_labels,
            "duplicate_of": self.duplicate_of,
            "summary": self.summary,
            "cites": self.cites,
            "confidence": self.confidence,
            "outcome": self.outcome.value,
            "n_steps": self.n_steps,
            "elapsed_ms": self.elapsed_ms,
            "fallback": self.fallback,
        }


# =============================================================================
# 主循环
# =============================================================================

class TriageAgent:
    """issue 分类处理 Agent。

    这是整个项目的"主程序"。它组装所有部件并驱动循环。
    """

    def __init__(self,
                 registry: ToolRegistry,
                 llm: LLMClient | None = None,
                 store: MemoryStore | None = None,
                 settings: Settings | None = None,
                 window_size: int = 32000):
        self.settings = settings or load_settings()
        self.registry = registry
        self.llm = llm or build_llm(self.settings.use_mock_llm)
        self.store = store
        self.window_size = window_size

        # 可靠性：横切关注点，从主循环里抽出来
        self.guard = ReliabilityGuard(self.settings.reliability)

        # 记忆生命周期
        if self.store is not None:
            self.extractor = Extractor(self.llm, self.settings.memory)
            self.scorer = Scorer(self.settings.memory)
            self.writer = Writer(self.store, self.settings.memory)
            self.resolver = Resolver(self.llm, self.store, self.settings.memory)
        else:
            self.extractor = self.scorer = self.writer = self.resolver = None

    # ------------------------------------------------------------------ 主入口

    def triage(self, issue: Issue, human_verdict: HumanVerdict = HumanVerdict.UNKNOWN,
               human_edit: str = "",
               context_padding: int = 0,
               enable_state_card: bool = True,
               include_system_constraints: bool = True,
               allowed_labels: list[str] | None = None,
               prompt_variant: str = "default",
               label_guide: str = "") -> TriageResult:
        """处理一个 issue。

        参数里的 human_verdict / human_edit 是可选的 —— 如果调用方从外部
        （比如真实的 GitHub 事件）已经知道维护者的反馈，就传进来，
        它会直接影响记忆的写入质量。

        ============================================================
        另外三个参数是给"长程一致性压测"用的（见 scripts/run_longhorizon.py）
        ============================================================
        `context_padding`：注入 N 轮"前置工作记录"，模拟 Agent 已经跑了一段长任务。
           为什么要注入而不是"让它真的跑很多轮"？因为那样不可控 ——
           任务该几步收敛是任务本身决定的，人为拉长会变成测别的东西。
           注入前置工作是**可控地制造上下文压力**。

        `enable_state_card`：是否启用"任务状态卡每轮重注入"。

        `include_system_constraints`：system prompt 里是否包含"判断准则"。
           ★ 这个开关是踩坑之后加的。第一版实验漏掉了它，导致对照组
             （关掉状态卡）其实还保留着 system prompt 里的约束 ——
             而 system 桶不可压缩，所以约束从未真正消失，实验测不出差异。

           所以真正严谨的对照需要三组：
             A. system有约束 + 状态卡  ← 完整设计
             B. system有约束 + 无状态卡 ← 第一版对照组（不彻底）
             C. system无约束 + 无状态卡 ← 真的没有保护

           **生产环境永远用 A**，B 和 C 只是实验工具。
        """
        started = time.time()
        trace = Trace(trace_id=f"tr_{_hash(issue.issue_id, time.time())}",
                      issue_id=issue.issue_id)

        # --- 上下文预算调度器 ---
        ctx = ContextBudgetManager(window_size=self.window_size)

        # System prompt 四选一：
        #   tuned             → 调优版（针对"多打标签"失败模式，用真实分布数据约束）
        #   strict            → 强化版（用显式流程替代思考链的规划能力）
        #   no_constraints    → 无约束版（仅供三组对照实验用）
        #   default           → 常规版
        if prompt_variant == "tuned":
            sys_text = SYSTEM_PROMPT_TUNED.format(
                tools=self._render_tools(),
                label_guide=label_guide or "  （无）",
            )
        elif prompt_variant == "strict":
            sys_text = SYSTEM_PROMPT_STRICT.format(tools=self._render_tools())
        elif not include_system_constraints:
            sys_text = SYSTEM_PROMPT_NO_CONSTRAINTS.format(
                tools=self._render_tools())
        else:
            sys_text = SYSTEM_PROMPT.format(tools=self._render_tools())
        ctx.set_system(sys_text)

        # --- 【关键】记忆预取：发生在循环之前 ---
        recalled = self._prefetch_memory(issue, ctx, trace)

        # --- 任务状态卡：长程一致性的核心机制 ---
        # 这份卡每轮重建，包含最初的目标和硬约束。
        # 它不参与压缩（在 task 桶里），所以即使 history 被压掉，约束仍在。
        state_card = self._build_state_card(issue, allowed_labels)

        if not enable_state_card:
            # ============================================================
            # 对照组：把任务信息一次性写在开头（不再每轮重申）
            # ============================================================
            # 这模拟的是最常见的做法：「把指令写进 prompt 开头就以为万事大吉」。
            #
            # ★ 两个必须注意的细节，都是踩坑换来的：
            #
            # ① **必须也写当前 issue 信息。**
            #    一开始我只写了约束，结果实验中出现了混淆变量：
            #    对照组连「当前在干什么」都不知道（因为 issue 信息也在状态卡里），
            #    于是测出来的差异可能来自「任务信息缺失」而不是「约束漂移」。
            #    → 这是典型的「你以为只改了一个变量，实际改了俩」。
            #
            # ② **是否附带约束，由 include_system_constraints 决定。**
            #    这是为了支持三组对照（见 triage 的文档说明）：
            #      B 组（system有约束 + 无状态卡）：这里**要**写约束
            #      C 组（system无约束 + 无状态卡）：这里**不**写约束
            #    如果不区分，C 组实际上还是有约束（只不过在 history 里），
            #    那就不是「无保护基线」，实验又白做了。
            #
            # 格式要和状态卡保持一致（用 `标题：` 这样的前缀），
            # 因为下游的症状提取逻辑依赖它。
            task_info = (
                f"[任务说明]\n"
                f"标题：{issue.title}\n"
                f"编号：{issue.issue_id}　提交者：{issue.author}\n"
                f"报错堆栈：{(issue.stack_trace or '无')[:200]}\n"
                f"目标：为该 issue 给出分类建议与处理建议"
            )
            if include_system_constraints:
                task_info += (
                    "\n硬约束：① 判重优先于分类 ② 不得编造根因；证据不足时明确说明 "
                    "③ 每条结论必须给出引用依据"
                )
            ctx.add_history(task_info)
        else:
            ctx.set_task(state_card)

        # --- 注入前置工作（模拟长任务）---
        if context_padding > 0:
            self._inject_padding(ctx, context_padding, issue)

        # ============================================================
        # 主循环
        # ============================================================
        for step_no in range(1, self.guard.max_steps + 1):
            self.guard.stats["steps_executed"] = step_no

            # --- 1. 组装上下文 ---
            if enable_state_card:
                # 每轮重新注入状态卡（对抗指令漂移）
                # ★ 注意这一步在 build() 之前 —— 顺序很重要：
                #   先更新 task 桶，再触发调度和渲染。
                #
                # ★ 顺便把「步数预算」写进去 —— 让模型有进度感。
                #
                # 为什么需要这个？
                # 关掉思考链后，模型会陷入"再试一次也许能找到"的循环
                # （实测步数从 3 涨到 20）。它不知道自己在浪费预算。
                #
                # 把剩余预算显式写出来，等于给它一个"该收敛了"的信号 ——
                # 这比在提示词里干说「最多 3 次」更有用，
                # 因为**它是动态的**：跑到第 8 步时模型才真正意识到
                # "我已经用掉很多了"。
                remaining = self.guard.max_steps - step_no
                budget = f"第 {step_no} 步 / 最多 {self.guard.max_steps} 步"
                if remaining <= 2:
                    budget += "　⚠ 预算即将耗尽，请立即 finalize 给出结论"
                elif remaining <= 5:
                    budget += "　（建议尽快收敛）"
                state_card["进度"] = budget
                ctx.set_task(state_card)
            messages, budget = ctx.build()
            trace.context_usage.append(budget.to_dict())

            # --- 2. 调模型 ---
            try:
                resp = self.llm.complete(messages, task="agent_step")
                trace.total_tokens += resp.total_tokens
            except Exception as e:                        # noqa: BLE001
                # 模型调用失败 → 兜底（**带着已有的证据**）
                return self._finalize_with_fallback(
                    issue, trace, started,
                    reason=f"模型调用失败: {e}")

            decision = self._parse_decision(resp.text)
            if decision is None:
                # 输出解析失败：不当作致命错误，把它作为观察结果反馈给模型
                # （这是"容忍模型输出不规范"的设计 —— 真实模型经常加解释文字）
                ctx.add_history(
                    f"[第 {step_no} 轮] 你的输出无法解析为 JSON。"
                    f"请严格只输出 JSON 对象，不要有解释文字或代码块标记。"
                )
                trace.record(Step(step=step_no, thought="(输出解析失败)"))
                continue

            thought = decision.get("thought", "")
            action = decision.get("action", "")
            action_input = decision.get("action_input") or {}

            step = Step(step=step_no, thought=thought)

            # --- 3. 终止条件：finalize ---
            if action == "finalize":
                return self._finalize(issue, trace, started, step_no,
                                      action_input, recalled, human_verdict, human_edit)

            # --- 4. 可靠性检查（熔断 / 循环）---
            blocked = self.guard.before_tool_call(action, action_input)
            if blocked:
                # **把"为什么不能调"作为观察结果告诉模型** ——
                # 而不是直接中断。因为模型可能只是策略选错了，换个工具就能过。
                # 只有当它反复撞墙（超轮数）才走兜底。
                ctx.add_history(f"[第 {step_no} 轮] 动作被可靠性机制阻止：{blocked}")
                step.observation = f"BLOCKED: {blocked}"
                trace.record(step)
                state_card["已完成步骤"].append(f"{step_no}. {action} → 被阻止")
                continue

            # --- 5. 执行工具 ---
            t0 = time.time()
            result = self._call_tool(action, action_input)
            latency = int((time.time() - t0) * 1000)
            self.guard.after_tool_call(action, result.ok)

            call = ToolCall(
                step=step_no,
                tool_name=action,
                tool_input=action_input,
                ok=result.ok,
                output=result.data,
                error=result.error,
                error_kind=result.error_kind,
                latency_ms=latency,
            )
            step.action = call

            # --- 6. 观察结果写回上下文 ---
            if result.ok:
                observation = json.dumps(result.data, ensure_ascii=False, default=str)
                offloaded = ctx.add_tool_result(action, result.data)
                call.offloaded = offloaded
                step.observation = observation[:500]
            else:
                observation = f"ERROR({result.error_kind}): {result.error}"
                ctx.add_tool_result(action, observation)
                step.observation = observation

            ctx.add_history(
                f"[第 {step_no} 轮] 思考: {thought}\n"
                f"动作: {action}({json.dumps(action_input, ensure_ascii=False)[:200]})\n"
                f"结果: {step.observation[:300]}"
            )
            state_card["已完成步骤"].append(f"{step_no}. {action} → {'成功' if result.ok else '失败'}")
            trace.record(step)

        # --- 超出轮数上限 → 有产出的兜底 ---
        return self._finalize_with_fallback(
            issue, trace, started,
            reason=f"达到轮数上限 {self.guard.max_steps} 轮仍未收敛")

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _build_state_card(issue: Issue,
                          allowed_labels: list[str] | None = None) -> dict:
        """构造任务状态卡。

        ============================================================
        为什么状态卡里必须包含"报错堆栈"
        ============================================================
        这是从诊断里发现的一个真问题：
        如果堆栈只存在于工具返回里，那么当检索到多条历史 issue 时，
        上下文里会同时出现**好几个不同 issue 的堆栈**。
        此时任何"从上下文里找异常类型"的逻辑都会抓错 ——
        会把历史 issue 的报错类型，当成当前问题的报错类型。

        实测表现：一个内容只有"用不了"的 issue，因为上下文里有别的
        issue 的堆栈（PermissionError），被误判成了权限问题。

        教训：**"当前任务的关键事实"必须放在一个不会被混淆的位置**，
        而不是散落在工具返回里和别人的信息混在一起。
        这就是状态卡存在的意义 —— 它是唯一的、权威的、无歧义的。

        ============================================================
        为什么必须告诉模型"允许的标签集"
        ============================================================
        这是跑真实数据评测时发现的（详见 docs/05 的 F1 号案例）：

        真实项目（pandas）有 91 个标签，分三层（类型 / 领域 / 流程）。
        我们的评测口径只认「类型」层 ——
        因为判断"属于哪个子模块"需要读源码，超出分流任务的范围。

        但**提示词里没有告诉模型这件事**，于是它自己发挥：
        输出了 Timezones / IO SAS / Segfault 这类领域标签，
        而它们不在 gold 里，全被算成误报，Precision 被拉到 0.5。

        **这个低分是「任务没说清」造成的，不是「模型不行」。**
        把它算进指标是错的，而且会误导优化方向 ——
        你可能会去优化模型能力，而问题其实在提示词。

        教训：**标签空间是任务定义的一部分，必须显式给出来。**
        让模型猜"该用什么标签"，等于让它猜你的评测口径。
        """
        card: dict = {
            "目标": "为该 issue 给出分类建议与处理建议",
            "硬约束": [
                "判重优先于分类",
                "不得编造根因；证据不足时明确说明",
                "每条结论必须给出引用依据",
            ],
            "当前 issue": {
                "编号": issue.issue_id,
                "标题": issue.title,
                "提交者": issue.author,
                "当前标签": issue.labels,
                "报错堆栈": (issue.stack_trace or "")[:400],
            },
            "已完成步骤": [],
        }

        if allowed_labels:
            card["允许使用的标签"] = list(allowed_labels)
            card["硬约束"].append(
                f"只能从上面「允许使用的标签」里选（共 {len(allowed_labels)} 个）；"
                f"不要发明新标签，也不要输出子模块 / 组件类标签"
            )
        return card

    @staticmethod
    def _inject_padding(ctx: ContextBudgetManager, n: int,
                        issue: Issue) -> None:
        """注入 N 轮"前置工作记录"，用来模拟 Agent 已经跑了一段长任务。

        ============================================================
        两个设计选择，都会直接影响实验结论的可信度
        ============================================================

        ① **注入的是"有意义的工作"，不是垃圾文本。**
           真实的长任务里，上下文被占满是由真实工作造成的。
           如果注入随机字符或无关废话，测的就变成了"噪音干扰"，
           而不是"长任务导致的指令漂移"—— 这是两个不同的问题，
           结论也不能互相套用。

        ② **记录内容必须是"已完成的历史"，不能和当前任务混淆。**
           所以每条都明确标注 `[前置工作 i/n]`，
           而且内容描述的是"别的 issue"，避免污染当前任务的判断。

        这是实验设计里很容易被忽略的一点：
        **你注入什么，决定了你测的是什么。**
        """
        for i in range(1, n + 1):
            # 每条都带足够细节（模拟真实的工具调用记录 + 结论）
            # 这样上下文占用才会像真的一样增长
            ctx.add_history(
                f"[前置工作 {i}/{n}] 已处理历史 issue #{100 + i}（提交者 user_{i % 7}）：\n"
                f"  症状：调用接口时出现 {'超时' if i % 3 == 0 else ('权限错误' if i % 3 == 1 else '配置不生效')}\n"
                f"  检索结果：命中 3 条相似历史，最高相似度 0.{60 + i % 30:.2f}\n"
                f"  标签判定：{'bug/timeout' if i % 3 == 0 else ('bug/permission' if i % 3 == 1 else 'bug/config')}"
                f"（置信度 0.{70 + i % 20:.2f}）\n"
                f"  调用工具：search_similar_issues → search_docs → apply_labels\n"
                f"  处置：已生成建议并写入工单系统，状态置为 triaged\n"
                f"  耗时 3{i % 10}0ms，消耗 4{i % 9}00 tokens"
            )

    def _prefetch_memory(self, issue: Issue, ctx: ContextBudgetManager,
                         trace: Trace) -> list[str]:
        """记忆预取。

        **注意这一步在循环之外。** 理由见文件头说明。

        召回两路：
          ① 按 author 精确召回 —— "这个提交者以前报过什么"
          ② 按 symptom 语义召回 —— "有没有人和他遇到一样的问题"
        这就是"混合检索"的业务含义：**两种意图需要两种信号**。

        ============================================================
        为什么必须有相似度门槛
        ============================================================
        最初这里是无条件取 top_k —— 只要库里有记忆就注入。

        结果很糟：一个内容只有"有 bug"的 issue，被注入了完全无关的记忆，
        然后模型**照着那条记忆的标签给它分类**，得到了一个错误答案。
        （实测：预测成 bug/config，而它其实应该转人工。）

        这是记忆系统特有的风险：**错误的召回比不召回更糟** ——
        因为模型会相信上下文里的东西。没有门槛的召回等于往提示词里灌噪音。

        所以门槛是必需的。而且这也解释了「无效召回率」这个指标为什么重要：
        在没有门槛的情况下它会非常高（实测 76.9%），
        而那些无效召回**正在主动误导模型**，不只是浪费 token。
        """
        if self.store is None:
            return []

        # 从配置读门槛（可在 config.MEMORY 里调）
        min_sim = self.settings.memory.get("recall_min_similarity", 0.30)

        recalled_ids: list[str] = []
        snippets: list[str] = []
        seen: set[str] = set()

        # 路径 ①：同一作者的历史（结构化精确匹配）
        for mem, score in self.store.recall(
                query=issue.search_text(), author=issue.author, top_k=3):
            if score < min_sim:
                continue
            recalled_ids.append(mem.memory_id)
            seen.add(mem.memory_id)
            snippets.append(self._format_memory(mem, score, tag="同一提交者的历史"))

        # 路径 ②：语义相似（向量检索）
        for mem, score in self.store.recall(query=issue.search_text(), top_k=5):
            if mem.memory_id in seen or score < min_sim:
                continue
            recalled_ids.append(mem.memory_id)
            seen.add(mem.memory_id)
            snippets.append(self._format_memory(mem, score, tag="语义相似"))

        if snippets:
            ctx.inject_memory(snippets)
        trace.memory_recalled = recalled_ids
        return recalled_ids

    @staticmethod
    def _format_memory(mem, score: float, tag: str) -> str:
        """把记忆渲染成给模型看的文本。

        **五个设计点：**
        ① 明确标注"这是历史参考，不是事实" —— 防止模型把旧结论当权威
        ② 带上来源和相似度 —— 让模型能自己判断该不该信
        ③ **带上标签** —— 标签是结构化信息，模型可以直接复用，
           不必从自然语言症状里重新推断（这一步能显著提升一致性）
        ④ **带上 memory_id 并要求引用** ——
           这是"记忆命中率"能测出来的前提：只有模型显式声明它用了哪条记忆，
           我们才能区分"召回了但没用"和"召回了并且用了"。
           如果没有这一项，无效召回率这个指标根本无从计算。
           真实模型同样应该被要求这么做 —— 这也是可追溯性的要求
           （"某次判断当时依据了哪条记忆"必须能回答）。
        ⑤ **把失败过的尝试也带出来** —— 这是情景记忆比知识库多出来的价值
        """
        lines = [f"[{tag}] 记忆ID={mem.memory_id} | 相似度 {score:.2f} "
                 f"| 来源 issue {mem.issue_id}"]
        if mem.symptom_tags:
            lines.append(f"  标签: {', '.join(mem.symptom_tags)}")
        lines.append(f"  症状: {mem.symptom}")
        if mem.root_cause:
            lines.append(f"  根因: {mem.root_cause}")
        if mem.resolution:
            lines.append(f"  处置: {mem.resolution}")
        failed = [a["tool"] for a in mem.actions if not a.get("ok")]
        if failed:
            lines.append(f"  （历史尝试中失败的动作: {', '.join(failed)}）")
        return "\n".join(lines)

    def _call_tool(self, name: str, args: dict) -> ToolResult:
        """调工具，带一层重试（第一层可靠性）。

        注意重试只在**可重试**的错误上生效（超时/限流/连接），
        参数错误会直接失败 —— 因为重试它没有意义，只会拖慢。
        """
        def once():
            r = self.registry.call(name, args)
            if not r.ok:
                raise _ToolError(r.error, r.error_kind)
            return r

        def classify(e: Exception) -> str:
            return getattr(e, "kind", "unknown")

        result, err = self.guard.retry.execute(once, classify)
        if err is not None:
            self.guard.stats["retries"] += 1
            return ToolResult(ok=False, error=str(err),
                              error_kind=getattr(err, "kind", "unknown"))
        return result

    def _parse_decision(self, text: str) -> Optional[dict]:
        """解析模型的输出为决策。

        真实模型经常不老实（加解释、包 code fence、前后有废话），
        所以这里做了多层容错。**这是必须做的事** ——
        不做的话，解析失败率会高到让整个系统不可用。

        ============================================================
        ★ 一个只有真实模型才会暴露的坑（Mock 测不出来）
        ============================================================
        跑真实模型压测时，最后一格崩了：

            TypeError: unhashable type: 'dict'
            at reliability.py: is_open(tool_name)

        根因：**模型有时把 `action` 输出成对象而不是字符串**：

            {"thought": "...",
             "action": {"name": "search_similar_issues",     ← 对象，不是字符串
                        "params": {"query": "..."}},
             "action_input": {}}

        而下游（熔断器、工具路由）都假设 action 是字符串。

        为什么 Mock 测不出来？因为规则版 Mock 永远输出规范的字符串 ——
        **它按设计输出，而真实模型按它自己的理解输出。**
        这就是"Mock 能验链路、不能验健壮性"的具体例子。

        教训两条：
          ① 解析模型输出时，**每个字段都要校验类型**，
             不能只检查 key 在不在（`"action" in d` 是不够的）。
          ② 校验失败时应**当作解析失败并反馈给模型重试**，
             而不是让异常往下传 —— 那样会在很远的地方崩，
             而且报错信息（unhashable type）对排查毫无线索。
        """
        if not text:
            return None
        t = text.strip()

        # 去掉 markdown code fence
        if "```" in t:
            parts = [p.strip() for p in t.split("```")]
            for p in parts:
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    t = p
                    break

        # 直接解析
        try:
            d = json.loads(t)
            if isinstance(d, dict):
                valid = self._validate_decision(d)
                if valid is not None:
                    return valid
        except json.JSONDecodeError:
            pass

        # 从文本里抠 JSON
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                d = json.loads(t[start:end + 1])
                if isinstance(d, dict):
                    return self._validate_decision(d)
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _validate_decision(d: dict) -> Optional[dict]:
        """校验并**尽力修复**模型返回的决策结构。

        做三件事：
          ① 把 `action` 规范化成字符串（能救就救，见上面那个坑）
          ② 确保 `action_input` 是 dict
          ③ 救不回来就返回 None → 走"解析失败"分支让模型重试

        为什么不直接抛异常？
        因为"模型输出不规范"是**预期内的常态**，不是异常情况。
        把它当异常处理，会导致：一次不规范输出整轮任务失败 ——
        而实际上重试一次大概率就好了。
        """
        if "action" not in d:
            return None

        act = d["action"]

        if isinstance(act, str):
            pass                                    # 正常

        elif isinstance(act, dict):
            # 模型把它写成了对象 —— 试着从常见键名里提取
            extracted = None
            for key in ("name", "tool", "tool_name", "action", "function"):
                if isinstance(act.get(key), str):
                    extracted = act[key]
                    break
            if extracted is None:
                return None                         # 提取不出来
            d["action"] = extracted

            # 顺便把参数也捞出来（如果模型没单独给 action_input）
            if not d.get("action_input"):
                for pkey in ("params", "arguments", "args", "input"):
                    if isinstance(act.get(pkey), dict):
                        d["action_input"] = act[pkey]
                        break
                else:
                    d["action_input"] = {
                        k: v for k, v in act.items()
                        if k not in ("name", "tool", "tool_name", "action",
                                     "function", "params", "arguments",
                                     "args", "input")
                    }
        else:
            return None                             # 数字/列表等，救不回来

        # --- 确保 action_input 是 dict ---
        ai = d.get("action_input")
        if ai is None:
            d["action_input"] = {}
        elif not isinstance(ai, dict):
            d["action_input"] = {"input": ai}       # 包一层，避免下游崩

        return d

    def _render_tools(self) -> str:
        """把工具描述渲染进系统提示词。"""
        return json.dumps(self.registry.specs(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ 收尾

    def _finalize(self, issue: Issue, trace: Trace, started: float, step_no: int,
                  action_input: dict, recalled: list[str],
                  human_verdict: HumanVerdict, human_edit: str) -> TriageResult:
        """正常收敛（Agent 自己认为完成了）。"""
        predicted = action_input.get("predicted_labels") or []
        dup = action_input.get("duplicate_of") or None
        summary = action_input.get("summary", "")
        cites = action_input.get("cites") or []
        confidence = float(action_input.get("confidence", 0.0) or 0.0)

        outcome = self._judge_outcome(predicted, confidence, issue)

        trace.final_answer = summary
        trace.predicted_labels = predicted
        trace.predicted_duplicate_of = dup
        trace.confidence = confidence
        trace.outcome = outcome
        trace.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        trace.memory_used = self._detect_used_memories(trace, recalled)

        if self.store is not None:
            self.store.save_trace(trace)
            # 记录记忆使用情况 —— 这是"无效召回率"指标的数据来源
            for mid in recalled:
                self.store.touch_access(mid, used=(mid in trace.memory_used))
            # 记忆沉淀：这是"回写闭环"的最后一环
            self._consolidate(issue, trace, human_verdict, human_edit)

        return TriageResult(
            issue_id=issue.issue_id,
            predicted_labels=predicted,
            duplicate_of=dup,
            summary=summary,
            cites=cites,
            confidence=confidence,
            outcome=outcome,
            n_steps=step_no,
            elapsed_ms=int((time.time() - started) * 1000),
            trace=trace,
        )

    @staticmethod
    def _judge_outcome(predicted: list[str], confidence: float,
                       issue: Issue) -> Outcome:
        """判定这次处理属于哪种结果。

        ============================================================
        这里有一个**业务上很重要、但很容易被做错**的区分
        ============================================================
        三种"看起来都是失败"的情况，实际含义完全不同：

            ① 转人工（ESCALATED）
               Agent 明确说"我判断不了，需要人来看"
               → **这不是失败**，这是正确的自我认知。
                 现实中的客服系统里，"该转人工时转人工"是核心能力之一 ——
                 比"硬答一个错的"好得多。

            ② 误诊（MISDIAGNOSED）
               Agent 给出了明确的分类，但和维护者的判断不一致
               → 这才是真正的失败，因为它会误导后续处理。

            ③ 未完成（UNFINISHED）
               达到轮数上限还没收敛（在 _finalize_with_fallback 里处理）

        **把 ① 和 ② 混在一起统计是错的** —— 会导致：
          · 一个"凡事先说不知道"的保守系统，看起来和"总答错"的系统一样差
          · 而实际上前者可用得多（它只是效率低，不会造成错误决策）

        所以这里的判定顺序是：**先看它有没有给出实质判断，再看判断对不对。**
        """
        # 归一化标签，避免 "bug/timeout" vs "timeout" 这类格式差异造成误判
        try:
            from ..eval.metrics import Metrics
            norm = Metrics.normalize_label
        except Exception:                                   # noqa: BLE001
            norm = lambda x: str(x).strip().lower()         # noqa: E731

        pred_set = {norm(x) for x in predicted if str(x).strip()}
        # "needs-triage" / "triage" / 空 → 等价于"没有给出实质分类"
        pred_set.discard("triage")
        pred_set.discard("")

        if not pred_set:
            # 没给出实质判断 → 转人工（**不是失败**）
            return Outcome.ESCALATED
        if confidence < 0.3:
            return Outcome.ESCALATED

        outcome = Outcome.RESOLVED
        # 有 ground truth 时才能判定误诊
        if issue.gold_labels:
            gold_set = {norm(x) for x in issue.gold_labels if str(x).strip()}
            gold_set.discard("triage")
            gold_set.discard("")
            if gold_set and not (gold_set & pred_set):
                outcome = Outcome.MISDIAGNOSED
        return outcome

    def _finalize_with_fallback(self, issue: Issue, trace: Trace, started: float,
                                reason: str) -> TriageResult:
        """降级收尾：有产出的失败。

        与"抛异常"的区别在于：这里返回一个**包含已收集证据的结构化结果**，
        人能拿着它接着往下做。
        """
        fb = self.guard.make_fallback(
            step=self.guard.stats["steps_executed"], reason=reason, trace=trace)

        trace.outcome = Outcome.UNFINISHED
        trace.final_answer = fb.suggestion
        trace.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        if self.store is not None:
            self.store.save_trace(trace)

        return TriageResult(
            issue_id=issue.issue_id,
            summary=fb.suggestion,
            outcome=Outcome.UNFINISHED,
            n_steps=self.guard.stats["steps_executed"],
            elapsed_ms=int((time.time() - started) * 1000),
            trace=trace,
            fallback=fb.to_dict(),
        )

    @staticmethod
    def _detect_used_memories(trace: Trace, recalled: list[str]) -> list[str]:
        """判断哪些被召回的记忆**真的被用上了**。

        启发式：如果最终答案里出现了记忆中特有的关键词（根因、处置的片段），
        就认为这条记忆被使用了。

        **这个判断不完美**，但它的价值在于：它让"无效召回率"可测量。
        没有它，你只能看到"召回了几条"，看不到"有几条在起作用"。
        （生产环境更好的做法是让模型显式引用 memory_id —— 这也是个可以讲的改进方向。）
        """
        used = []
        answer = trace.final_answer or ""
        for mid in recalled:
            # 简化判断：答案中是否包含该记忆 id 或显著的片段
            if mid in answer:
                used.append(mid)
        return used

    def _consolidate(self, issue: Issue, trace: Trace,
                     human_verdict: HumanVerdict, human_edit: str) -> None:
        """记忆沉淀 —— 回写闭环的最后一环。

        **这一步是整个项目跟"无状态问答机器人"的分水岭。**
        没有它，Agent 每次都是第一次；有了它，第二次会更快更准。
        """
        if self.extractor is None or self.writer is None:
            return
        cand = self.extractor.extract(trace, issue, human_verdict, human_edit)
        if cand is None:
            return

        score = self.scorer.score(cand)
        if not self.scorer.should_write(score):
            # 分数不够 → 不写。
            # 这不是"丢弃"，是"筛选"。记忆库的价值来自它的信噪比，不是它的规模。
            return

        self.writer.commit(cand, issue, trace, human_verdict, human_edit)


class _ToolError(Exception):
    """内部异常：把 ToolResult 的失败转成可重试框架认识的异常。"""
    def __init__(self, msg: str, kind: str):
        super().__init__(msg)
        self.kind = kind
