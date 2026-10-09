"""
Nightwatch · 记忆生命周期（六环节）

================================================================================
这是整个项目最核心的文件。建议学习顺序：先读这个，再读 store.py。
================================================================================

记忆系统有一句常被引用的话：
    **"真正困难的不是存储，而是写入与读取策略。"
     什么值得保存？何时合并重复事实？怎样处理互相冲突的记忆？
     检索多少条才不会挤压当前任务？错误记忆如何被纠正？**

下面六个环节就是对这五个问题的回答。

    ① 抽取 Extract      —— 从"一次完整的处理轨迹"里提炼出候选记忆
    ② 评分 Score        —— 判断这条值不值得长期保留
    ③ 写入 Write        —— 过阈值才落库，且保证幂等
    ④ 合并 Consolidate  —— 同一件事被记了两遍，合并而不是新增
    ⑤ 消解 Resolve      —— 两条记忆矛盾时怎么裁决
    ⑥ 衰减 Decay        —— 时间让记忆降权，但不是删掉

还有一个**升维通道**：
    情景记忆（具体的事）──高频共性模式──▶ 语义记忆（抽象的规则/SOP）
    注意这条通道上有一道**人工审核闸门**，不是自动的。理由见 resolve() 的说明。

================================================================================
一个贯穿全篇的设计观点：不信"全自动"
================================================================================
你在下面会看到三处"把决定权交回给人"的设计：
  - 冲突消解不下最终结论，进 pending 队列等人工裁决
  - 情景记忆升维成语义记忆，需要人工审核
  - 被驳回的记忆不删除，而是降权并保留审计痕迹

这不是实现偷懒，是一个判断：**记忆系统不可能永远自洽**。
承认这一点，并把它明确地暴露给人，比让 Agent 自己拍板可靠得多。
在真实生产环境里，这个态度是加分项。
================================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..config import MEMORY
from ..llm.client import LLMClient
from ..models import (
    HumanVerdict, MemoryRecord, MemoryTier, Outcome, PendingConflict, Trace, _hash,
)
from .store import MemoryStore, cosine


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return _now()


# =============================================================================
# ① 抽取 Extract
# =============================================================================

@dataclass
class Candidate:
    """从轨迹里抽出来的候选记忆（还没决定要不要存）。"""
    symptom: str
    symptom_tags: list[str] = field(default_factory=list)
    root_cause: str = ""
    resolution: str = ""
    actions: list[dict] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    total_score: float = 0.0


class Extractor:
    """环节 ①：从轨迹里提炼候选记忆。

    ============================================================
    关键设计一：抽什么、不抽什么
    ============================================================
    **要抽**：症状、根因、处置动作序列、最终结论
    **不抽**：寒暄、无关的中间对话、单次性的临时参数（比如某次查询用的时间戳）

    判断标准就一条：**"如果下次遇到类似问题，这条信息能不能让我少走弯路？"**

    ============================================================
    关键设计二：为什么这里是异步的
    ============================================================
    抽取的价值在未来，不在当下 —— 当前这次处理并不依赖"这次的经验有没有被存下来"。
    所以**没有任何理由让它阻塞主链路**。

    异步之后要处理三件事（这才是真实的工程问题）：
      1. 失败重试：抽取失败了不能影响工单本身
      2. 幂等：重试时不能写两条（由 store 的幂等键在数据库层保证）
      3. 死信：重试到上限还失败，要记录下来人工处理，不能静默丢弃
    """

    def __init__(self, llm: LLMClient, cfg: dict | None = None):
        self.llm = llm
        self.cfg = cfg or MEMORY

    def extract(self, trace: Trace, issue, human_verdict: HumanVerdict = HumanVerdict.UNKNOWN,
                human_edit: str = "") -> Optional[Candidate]:
        """从一次处理轨迹里抽候选记忆。

        注意入参里的 human_verdict / human_edit ——
        **人工反馈是抽取环节的输入，不只是评测的输入**。
        工程师改了什么，直接决定这条记忆怎么写。
        """
        # --- 收集处置动作，包括失败的 ---
        # ★ 这是"值钱字段"之一：大多数项目只记成功路径，
        #   但"试过 A 不行，应该试 B"恰恰是下次省时间的地方。
        #   知识库告诉你"正确答案"，记忆告诉你"在你这个环境下哪条路走不通"。
        actions = []
        for step in trace.steps:
            if step.action is not None:
                actions.append({
                    "tool": step.action.tool_name,
                    "ok": step.action.ok,
                    "note": (step.action.error or
                             str(step.action.output)[:120] if step.action.output else ""),
                })

        # --- 用（小）模型做结构化抽取 ---
        # 这里对应 config.MODEL_ROUTING 里的 "memory_extract" → small 档。
        # 为什么用小模型：这是调用量最大的任务（每条 issue 一次），
        # 而且是结构化输出（不需要强推理），成本敏感。
        prompt = self._build_prompt(trace, issue)
        try:
            resp = self.llm.complete(
                [{"role": "user", "content": prompt}], task="memory_extract")
            data = json.loads(self._strip_json(resp.text))
        except (json.JSONDecodeError, Exception):
            # 抽取失败 → 返回 None，由调用方决定重试或记死信。
            # **不能让抽取失败影响主链路** —— 这是异步设计的意义。
            return None

        return Candidate(
            symptom=(data.get("symptom") or "").strip(),
            symptom_tags=data.get("symptom_tags") or [],
            root_cause=(data.get("root_cause") or "").strip(),
            resolution=(data.get("resolution") or "").strip(),
            actions=actions,
            scores={
                "reusability": float(data.get("reusability", 0.5)),
                "causal_depth": float(data.get("causal_depth", 0.5)),
                "non_triviality": float(data.get("non_triviality", 0.5)),
                # human_signal 不从模型来 —— 它是客观信号，直接由人工反馈决定
                "human_signal": self._human_signal(human_verdict),
            },
        )

    @staticmethod
    def _human_signal(verdict: HumanVerdict) -> Optional[float]:
        """把人工反馈映射成 0~1 的分数，**没有反馈时返回 None**。

        ============================================================
        为什么 UNKNOWN 返回 None 而不是 0.5（中性值）
        ============================================================
        这是一个从诊断里发现的设计缺陷。最初的实现把 UNKNOWN 当"中性 0.5"，

        算出来的分数是：
            0.35×0.7 + 0.25×0.3 + 0.25×0.5 + 0.15×0.6 = 0.535
                                                          ↓
                                              低于阈值 0.55 → 记忆写不进去

        也就是说：**在没有人给反馈的场景下（比如离线评测、自动化批处理），
        记忆系统会几乎完全停止工作。** 而"没有反馈"恰恰是最常见的状态。

        正确的理解是：「没有信号」和「中性的信号」是两件事。
            · 中性 0.5 的意思是"我收到了反馈，反馈是中性的"
            · None     的意思是"我没收到任何反馈"
        后者应该**从加权平均里剔除**，而不是拉低总分。

        所以这里返回 None，由 Scorer 做权重重归一化。

        这也是一个通用的设计教训：**不要把"缺失"编码成"中间值"。**
        缺失就是缺失，用 None/NaN 表达，让下游显式处理它。
        """
        mapping = {
            HumanVerdict.ACCEPTED: 1.0,
            HumanVerdict.EDITED: 0.7,     # 修改后采纳：方向对但细节错
            HumanVerdict.REJECTED: 0.0,   # 明确错误，必须压制
            HumanVerdict.UNKNOWN: None,   # 无反馈 = 无信号，不参与加权
        }
        return mapping.get(verdict, None)

    @staticmethod
    def _build_prompt(trace: Trace, issue) -> str:
        """抽取用的提示词。

        设计要点：**给出明确的输出 schema，并说明"允许字段为空"**。
        如果不说明"根因可以为空"，模型会硬编一个根因出来 ——
        而"未定位到根因"本身是有价值的信息（它意味着这个问题需要更深的排查）。
        """
        final = trace.final_answer[:800]
        return f"""你是一个运维经验提炼助手。请从下面这次问题处理记录中，提取一条可复用的经验。

【问题】
标题：{issue.title}
症状：{issue.body[:600]}

【处理过程与结论】
{final}

请严格按以下 JSON 输出，不要输出任何其他内容：
{{
  "symptom": "一句话描述问题症状（客观描述，不加猜测）",
  "symptom_tags": ["从这些里选：timeout/oom/permission/not_found/crash/config/dependency/duplicate/general"],
  "root_cause": "根因。如果没能定位到根因，这项留空字符串，不要编造",
  "resolution": "最终处置方式",
  "reusability": 0.0,
  "causal_depth": 0.0,
  "non_triviality": 0.0
}}

其中三个分数的含义：
- reusability：这类问题未来还会不会再出现？（会重复出现的给高分）
- causal_depth：是否定位到了根因？（有根因给高分，只缓解了症状给低分）
- non_triviality：这个解决方案是不是常识？（"重启一下"这类给低分）"""

    @staticmethod
    def _strip_json(text: str) -> str:
        """从模型输出里抠出 JSON。

        真实项目必须做这一步 —— 模型经常在 JSON 外面包一层 ```json 或加解释文字。
        不处理的话，解析失败率会高得离谱。
        """
        t = text.strip()
        if "```" in t:
            parts = t.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    return p
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return t[start:end + 1]
        return t


# =============================================================================
# ② 评分 Score
# =============================================================================

class Scorer:
    """环节 ②：重要性评分。

    ============================================================
    为什么不用模型打分，而用可解释的加权和
    ============================================================
    两个理由：
      1. **"为什么这条被记下来了"必须能回答。** 如果评分是个黑盒，
         你无法调试"为什么记忆库里全是垃圾"这类问题。
      2. 模型打分本身有随机性，而写入决策需要稳定 ——
         同一个候选记忆今天打分 0.6、明天 0.5，库的行为就不可预测了。

    权重是通过**标注一小批样本标定**的，不是拍脑袋定的。标定过程见
    docs/05-评测设计.md —— "你怎么调的权重"是必然会被追问的问题，
    有标定过程才答得上来。
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or MEMORY

    def score(self, cand: Candidate) -> float:
        """加权和 → 总分。**对缺失因子做权重重归一化。**

        五个因子的含义：
          reusability    可复用性 —— 未来还会再出现吗
          causal_depth   因果完整性 —— 定位到根因了吗
          human_signal   人工反馈 —— 采纳加分、驳回大幅扣分
                         （None 表示没有反馈，从加权里剔除，而不是当 0）
          non_triviality 非平凡性 —— 是不是"重启一下"这种谁都知道的

        为什么要重归一化权重，而不是把缺失值当 0 或 0.5 填进去？
        见 _human_signal 里的详细说明 —— 简单说：
        **把"没有信号"当成"弱信号"，会让系统在没人反馈的场景下整体失效。**
        """
        w = self.cfg["score_weights"]
        s = cand.scores

        # 只统计"有信号"的因子
        available = {k: float(v) for k, v in s.items()
                     if k in w and v is not None}
        if not available:
            cand.total_score = 0.0
            return 0.0

        total_w = sum(w[k] for k in available)
        if total_w <= 0:
            cand.total_score = 0.0
            return 0.0

        total = sum(w[k] * v for k, v in available.items()) / total_w
        cand.total_score = round(total, 4)
        return cand.total_score

    def should_write(self, score: float) -> bool:
        """过阈值才写入。

        这个阈值的标定是个真实的取舍：
          太低 → 记忆库被垃圾淹没，召回质量崩塌
          太高 → 该记的没记住，系统不会进步

        标定方法（docs/05 有完整流程）：
          人工标注一批"是否值得长期保留"，然后扫阈值，
          看"被召回率"和"污染率"这两个指标的平衡点在哪。
        """
        return score >= self.cfg["write_threshold"]


# =============================================================================
# ③ 写入 Write + ④ 合并 Consolidate
# =============================================================================

def _extract_code_refs(trace: Trace) -> list[str]:
    """从 trace 里抽出这条记忆**依赖过的代码文件路径**。

    ==========================================================================
    为什么从 trace 抽，而不是让模型"自己说它依赖什么"
    ==========================================================================
    **trace 是事实记录，模型的自我报告是总结。**

    两者在需要"精确、可机检"的场景下差别很大：

        trace 里记的：`read_code(path="pandas/core/frame.py")` ← 客观发生过
        模型说的：   "我参考了 frame.py 的相关逻辑"    ← 可能含糊、遗漏

    而 `code_refs` 的用途是**集合运算**（`code_refs ∩ changed_files`），
    这要求路径是**准确且可直接比对**的。模型输出的自然语言路径
    （"core/frame"、"frame.py"、"pandas/core/frame.py"）格式不一，
    做交集会大量漏判。

    **能用结构化数据就不要用自然语言。** 这是 `docs/05` 里
    "用痕迹而不是信号"那条原则的又一次应用。

    ==========================================================================
    ★ 一个诚实的边界：这里抽的是"读过"，不是"依赖"
    ==========================================================================
    Agent 读了一个文件，不代表它的结论**一定**建立在这个文件上。
    比如它可能只是随手 read 了一下就否决了那条线索。

    所以 `code_refs` 是**过度包含**的（over-approximate）——
    它会引入一些"读了但没用到"的路径。

    **这个偏差方向是刻意选的**：
      · 过度包含 → 复核队列里多一些条目 → 人工多看一眼（成本可控）
      · 过度排除 → 漏掉真正失效的记忆 → 错误继续传播（代价高）

    **在"宁可多查、不可漏查"的场景里，过度包含是正确的偏差方向。**
    （这和 `docs/06` 里 PR 风险阈值"取区间最小值"是同一个取舍逻辑。）
    """
    paths: list[str] = []
    seen: set[str] = set()
    for step in getattr(trace, "steps", []) or []:
        call = getattr(step, "action", None)
        if call is None:
            continue
        if getattr(call, "tool_name", "") != "read_code":
            continue
        # tool_input 是 dict（见 models.ToolCall 的说明）
        args = getattr(call, "tool_input", None) or {}
        p = args.get("path") or args.get("file") or ""
        p = str(p).strip()
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


class Writer:
    """环节 ③④：写入与合并。

    为什么把这两个放在一起？因为它们是一个决策的两面：
        "这个信息进来时，我是新增一条，还是更新已有的一条？"

    分开写会导致重复代码，而且容易漏掉"先去重再写入"的顺序。
    """

    def __init__(self, store: MemoryStore, cfg: dict | None = None):
        self.store = store
        self.cfg = cfg or MEMORY

    def commit(self, cand: Candidate, issue, trace: Trace,
               human_verdict: HumanVerdict = HumanVerdict.UNKNOWN,
               human_edit: str = "",
               repo_head: str = "") -> tuple[Optional[MemoryRecord], str]:
        """写入或合并。返回 (记忆对象, 动作说明)。

        动作说明是 'inserted' / 'merged' / 'skipped_duplicate'，
        用于可观测 —— 你能在统计里看到"合并率"这个指标。
        合并率高说明抽取环节有重复产出，需要优化；合并率为 0 说明阈值太松。

        ★ `repo_head` 为什么是可选参数、默认空串？

          因为**当前调用链拿不到真实的 commit** ——
          Agent 是只读的，它不调 git，trace 里也没有代码版本信息。

          那为什么还要留这个参数？两个理由：

            ① **先把它做出来，比"以后再加"便宜得多。**
               加了参数但传空串 → 字段存在、值为空；
               以后从 PR 事件拿到 commit 时，只是把值填上，
               **不需要改 schema、不需要改调用链**。

            ② 它让"这条字段为什么是空的"这个事实**显式可见** ——
               而不是让后来的人以为"忘了填"。

          **诚实留白 > 假装完备。** 字段存在但值为空，
          和字段不存在，在排查时的含义完全不同。
        """
        # ★ 抽出这条记忆依赖过的代码路径（仓库变更时效性的地基）
        #
        # 为什么在写入的那一刻抽，而不是事后补？
        # 因为 trace 就在手上 —— 事后要从数据库里翻 trace 再解析，
        # 而且**已写入的记忆如果漏了这一步，就永远补不回来了**
        # （trace 可能已被清理）。
        # **一次写入时顺手做的事，别留成"以后再补"。**
        code_refs = _extract_code_refs(trace)

        mem = MemoryRecord(
            memory_id=f"ep_{_hash(issue.repo, issue.issue_id, cand.symptom[:40])}",
            tier=MemoryTier.EPISODIC,
            repo=issue.repo,
            issue_id=issue.issue_id,
            author=issue.author,
            symptom=cand.symptom,
            symptom_tags=cand.symptom_tags,
            root_cause=cand.root_cause,
            resolution=cand.resolution,
            actions=cand.actions,       # ★ 含失败的尝试
            outcome=trace.outcome,
            human_verdict=human_verdict,  # ★ 人工反馈
            human_edit=human_edit,        # ★ 修改内容
            confidence=trace.confidence,
            importance=cand.total_score,
            # ★★ 代码状态绑定（仓库变更时效性的地基，见 models.MemoryRecord）
            code_refs=code_refs,
            repo_head=repo_head,
        )
        mem.embedding = self.store.encoder.encode(self.store._embed_text(mem))

        # --- 路径一：幂等键已存在 → 同一条事，跳过 ---
        # 这是异步重试的正常路径，不是错误
        existing = self.store.get_by_idem(mem.idempotency_key())
        if existing is not None:
            return existing, "skipped_duplicate"

        # --- 路径二：找到高度相似的记忆 → 合并 ---
        # 合并解决的是"同一件事从不同角度被记了两遍"。
        # 比如同一个问题，一次描述成"接口超时"，一次描述成"请求无响应"。
        similar = self.store.find_similar(mem, self.cfg["merge_similarity"])
        if similar is not None:
            merged = self._merge(similar, mem)
            self.store.update(merged)
            return merged, "merged"

        # --- 路径三：新记忆 ---
        ok = self.store.insert(mem)
        return (mem, "inserted") if ok else (None, "skipped_duplicate")

    @staticmethod
    def _merge(old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        """合并两条相似记忆。

        合并策略（不是简单覆盖）：
          - symptom 保留更长的那个（通常信息更全）
          - root_cause 优先保留有的那个（"没定位到"不如"定位到了"）
          - tags 取并集（信息不丢）
          - actions 累加去重（保留两次的失败尝试，这是最有价值的部分）
          - access/use 计数累加（历史热度要继承）
          - 人工反馈取更权威的（EDITED 比 UNKNOWN 可信）
        """
        merged = MemoryRecord(**{**old.__dict__})
        merged.memory_id = old.memory_id

        if len(new.symptom) > len(old.symptom):
            merged.symptom = new.symptom
        if not merged.root_cause and new.root_cause:
            merged.root_cause = new.root_cause
        merged.symptom_tags = sorted(set(old.symptom_tags) | set(new.symptom_tags))

        seen = {json.dumps(a, sort_keys=True, ensure_ascii=False) for a in merged.actions}
        for a in new.actions:
            key = json.dumps(a, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                merged.actions.append(a)
                seen.add(key)

        merged.access_count = old.access_count + new.access_count
        merged.use_count = old.use_count + new.use_count

        # 权威性排序：EDITED > ACCEPTED > REJECTED > UNKNOWN
        authority = {HumanVerdict.EDITED: 3, HumanVerdict.ACCEPTED: 2,
                     HumanVerdict.REJECTED: 1, HumanVerdict.UNKNOWN: 0}
        if authority[new.human_verdict] > authority[old.human_verdict]:
            merged.human_verdict = new.human_verdict
            merged.human_edit = new.human_edit

        merged.importance = max(old.importance, new.importance)
        return merged


# =============================================================================
# ⑤ 消解 Resolve
# =============================================================================

class Resolver:
    """环节 ⑤：冲突消解。

    ============================================================
    核心设计：三级裁决，但**不下最终结论**
    ============================================================
    三级裁决的依据：
      ① 新近度（recency）   —— 时间近的更可能对
         但**单独用很危险**：新记录本身可能是错的
      ② 来源权威性（authority）—— 权重最高
         人工编辑过 > 人工确认过 > Agent 自动抽取
         （所以 human_edit 字段的价值在这里体现）
      ③ 证据强度（evidence）—— 有工具调用日志佐证的 > 仅靠模型推断的

    **但裁决结果不是"覆盖"，而是进 pending 队列。**

    为什么？因为记忆系统不可能永远自洽。如果让 Agent 自己拍板覆盖，
    一旦拍错，错误会通过召回传播到后续所有相关判断 —— 错上加错。
    把矛盾明确挂出来等人判，看起来慢，但可靠。

    面试时这个点是加分项：它说明你不相信"全自动"，也说明你理解
    "记忆系统特有的失败模式是错误被放大"。
    """

    def __init__(self, llm: LLMClient, store: MemoryStore, cfg: dict | None = None):
        self.llm = llm
        self.store = store
        self.cfg = cfg or MEMORY

    def detect_conflict(self, a: MemoryRecord, b: MemoryRecord) -> bool:
        """判断两条记忆是否真的矛盾。

        ============================================================
        这里有个容易搞错的点：**相似 ≠ 矛盾**
        ============================================================
        两条记忆描述同一个问题（相似度高）是正常的，不构成冲突。
        冲突的定义要更严格：**它们在同一个事实上给出了不相容的结论**。

        具体到本场景的判据：
          - symptom 相似（是同一个问题）
          - 但 root_cause 都存在、且字面差异很大（对根因的判断不一致）
          - 或 resolution 都存在、且明显不同（处置方式不一致）

        只看相似度就判冲突，会产生大量假阳性，pending 队列会被淹没，
        人工就不会看了 —— 这是设计"人工回环"时最容易犯的错。
        """
        sim = cosine(a.embedding, b.embedding)
        if sim < 0.75:          # 不是同一个问题，谈不上冲突
            return False

        if a.root_cause and b.root_cause:
            overlap = self._overlap(a.root_cause, b.root_cause)
            if overlap < 0.3:   # 对根因的判断差得很远
                return True

        if a.resolution and b.resolution:
            overlap = self._overlap(a.resolution, b.resolution)
            if overlap < 0.25:
                return True

        return False

    def resolve(self, a: MemoryRecord, b: MemoryRecord) -> tuple[Optional[MemoryRecord], PendingConflict | None]:
        """尝试消解。返回 (胜出者 或 None, 待裁决记录 或 None)。

        分差足够大 → 自动判定（但仍然记录，可审计）
        分差太小 → 进 pending，交人工
        """
        w = self.cfg["resolve_weights"]
        score_a = self._authority_score(a, w)
        score_b = self._authority_score(b, w)
        margin = abs(score_a - score_b)

        if margin < self.cfg["resolve_min_margin"]:
            # 分不出来 → 明确地承认"分不出来"
            c = PendingConflict(
                conflict_id=f"cf_{_hash(a.memory_id, b.memory_id)}",
                memory_a=a.memory_id,
                memory_b=b.memory_id,
                reason=(f"无法自动裁决（分差 {margin:.3f} < 阈值 "
                        f"{self.cfg['resolve_min_margin']}）："
                        f"根因分别判断为『{a.root_cause[:40]}』与『{b.root_cause[:40]}』"),
            )
            self.store.add_conflict(c)
            return None, c

        winner = a if score_a > score_b else b
        loser = b if winner is a else a

        # 败者降权而不是删除 —— 保留审计痕迹。
        # "某次判断当时依据了哪几条记忆"这个问题必须能回答，
        # 尤其在医疗、金融这类场景，留痕是合规要求。
        loser.decay_weight *= 0.4
        loser.archived = True
        self.store.update(loser)

        winner.confidence = min(1.0, winner.confidence + 0.05)
        self.store.update(winner)
        return winner, None

    def _authority_score(self, m: MemoryRecord, w: dict) -> float:
        """按三级裁决算权威分。"""
        # ① 新近度：越新越高，用 30 天做尺度
        age_days = (_now() - _parse_ts(m.created_at)).total_seconds() / 86400
        recency = math.exp(-age_days / 30.0)

        # ② 来源权威性
        authority = {
            HumanVerdict.EDITED: 1.0,     # 人工修改过 = 最高可信
            HumanVerdict.ACCEPTED: 0.85,
            HumanVerdict.UNKNOWN: 0.5,    # 自动抽取且无人复核
            HumanVerdict.REJECTED: 0.1,
        }.get(m.human_verdict, 0.5)

        # ③ 证据强度：有成功的工具调用佐证 → 分高
        ok_actions = sum(1 for a in m.actions if a.get("ok"))
        evidence = min(1.0, 0.4 + 0.2 * ok_actions)

        return (w["recency"] * recency
                + w["authority"] * authority
                + w["evidence"] * evidence)

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        """字符 bigram 的 Jaccard 相似度，用来判断两个文本是否在说同一件事。"""
        def grams(s: str) -> set[str]:
            s = s.lower().replace(" ", "")
            return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)


# =============================================================================
# ⑥ 衰减 Decay
# =============================================================================

class Decayer:
    """环节 ⑥：衰减。

    ============================================================
    三个容易做错的地方
    ============================================================

    一、**衰减不等于删除**
       本项目的分层处理：
         权重还在 → 正常参与召回
         权重低   → 移入冷存储（archived=1），仍可精确查询，不参与向量召回
         保留期满 → 归档（真实项目还要考虑审计要求，不能真删）

    二、**访问频次高不等于重要**（这是最反直觉的一点）
       有些记忆被频繁召回，仅仅因为它的向量"吸引"了很多不相关查询
       —— 我们叫它"噪音吸铁石"。
       所以有效权重必须引入"召回后真的被用上了"的比例，
       而不是简单地看 access_count。

    三、**人工确认过的记忆不该轻易衰减**
       工程师亲口确认过的经验，价值不随时间的流逝而降低那么多。
       所以衰减公式里要给人工作为证据的记忆一个折扣系数。
    """

    def __init__(self, store: MemoryStore, cfg: dict | None = None):
        self.store = store
        self.cfg = cfg or MEMORY

    def decay_all(self) -> dict:
        """对所有记忆跑一次衰减。通常按天定时执行。"""
        half_life = self.cfg["decay_half_life_days"]
        archive_th = self.cfg["archive_threshold"]
        stats = {"scanned": 0, "archived": 0, "restored": 0}

        for mem in self.store.all_memories():
            stats["scanned"] += 1
            weight = self.compute_weight(mem, half_life)

            if weight < archive_th and not mem.archived:
                mem.archived = True
                stats["archived"] += 1
            elif weight >= archive_th and mem.archived and mem.human_verdict in (
                    HumanVerdict.ACCEPTED, HumanVerdict.EDITED):
                # 被人工确认过的不该被埋 —— 这是个例外规则
                mem.archived = False
                stats["restored"] += 1

            mem.decay_weight = round(weight, 4)
            self.store.update(mem)

        return stats

    def compute_weight(self, mem: MemoryRecord, half_life: float) -> float:
        """衰减权重 = 时间衰减 × 有效性修正 × 权威性加成。

        三个因子的顺序和量级是有讲究的，不是随便乘在一起。
        """
        # --- 因子 1：时间衰减 ---
        # 半衰期公式：w = 0.5 ^ (age / half_life)
        # 为什么用半衰期而不是线性衰减？因为经验的"过时"是指数型的 ——
        # 前一个月几乎不过时，一年后基本没用。
        age_days = (_now() - _parse_ts(mem.created_at)).total_seconds() / 86400
        time_decay = 0.5 ** (age_days / max(half_life, 1.0))

        # --- 因子 2：有效性修正（本环节的核心设计）---
        # 关键：用 use_count / access_count 而不是 access_count 本身。
        #
        # 如果一条记忆被召回了 50 次但一次都没被用上，它不是在帮忙，是在占位置。
        # 给它惩罚，让它自然沉淀下去。
        if mem.access_count > 0:
            effectiveness = mem.use_count / mem.access_count
        else:
            effectiveness = 0.5   # 还没被召回过的保持中性，不给奖也不给罚

        # effectiveness ∈ [0,1]，映射到一个"不致命但明显"的调节区间
        # 为什么不做成乘法（× effectiveness）？因为那会太狠 ——
        # 一条从未被用过的记忆会被直接清零，而"从未被用"可能只是因为它还没遇到合适的问题。
        eff_factor = 0.5 + 0.5 * effectiveness

        # --- 因子 3：权威性加成 ---
        # 人工确认过的经验，衰减得慢一些（"折扣系数"）
        authority_bonus = {
            HumanVerdict.EDITED: 1.0,      # 不打折
            HumanVerdict.ACCEPTED: 1.0,
            HumanVerdict.UNKNOWN: 0.9,     # 无人复核的轻微打折
            HumanVerdict.REJECTED: 0.3,    # 被驳回的快速下沉
        }.get(mem.human_verdict, 0.9)

        weight = max(0.0, min(1.0, time_decay * eff_factor * authority_bonus))

        # --- 因子 4：人工确认的权重下限（保护机制）---
        #
        # ============================================================
        # 为什么需要这个下限
        # ============================================================
        # 单元测试发现的问题：一条被工程师明确 ACCEPTED 的经验，
        # 400 天后权重降到 0.03，被系统当成垃圾归档了。
        #
        # 这在业务上是错的，因为：
        #   · **验证过是对的**，这件事本身有很高的价值 —— 而且验证成本很高
        #     （需要人真的去处理一次问题、真的判断它对不对）
        #   · 这类经验通常描述的是**稳定事实**（架构缺陷、协同模式），
        #     不是易变的状态（客户配额、版本号）—— 所以它不该随时间快速贬值
        #
        # 时间衰减的隐含假设是"旧信息更可能过时"。但这个假设对
        # "被人工验证过的结构性经验"不成立。
        #
        # 所以给它一个下限：低于这个值就托住。
        # 这是把领域知识写进公式的一个例子 —— 纯数学的衰减公式不知道
        # "这条经验被人验证过"意味着什么，得靠设计者补上。
        if mem.human_verdict in (HumanVerdict.ACCEPTED, HumanVerdict.EDITED):
            weight = max(weight, 0.25)

        return weight


# =============================================================================
# 升维通道：情景记忆 → 语义记忆
# =============================================================================

class SemanticPromoter:
    """把高频出现的共性模式，沉淀成 SOP（语义记忆）。

    ============================================================
    这条通道上有一道**人工审核闸门**
    ============================================================
    为什么不让它自动完成？因为语义记忆的影响面太大 ——
    它是"我们约定怎么处理这类问题"，会被所有后续任务召回。
    一条错误的 SOP 会污染大量判断。

    而情景记忆错了只影响一次召回。两者风险量级不同，所以门槛不同。

    这个设计还有一个好处：它回答了一个可能的质疑 ——
    "你的记忆系统会不会自己跑偏？" 答：会，所以我给影响力大的那一层加了人工闸门。
    """

    # 触发提炼的条件：同一 tags 组合出现次数达到这个值
    PROMOTE_MIN_OCCURRENCES = 3

    def __init__(self, llm: LLMClient, store: MemoryStore):
        self.llm = llm
        self.store = store

    def find_patterns(self) -> list[dict]:
        """找出高频的标签组合 —— 它们是"共性模式"的候选。"""
        from collections import Counter
        counter: Counter = Counter()
        samples: dict[str, list[MemoryRecord]] = {}

        for mem in self.store.all_memories(MemoryTier.EPISODIC):
            if mem.archived:
                continue
            if not mem.symptom_tags:
                continue
            key = "|".join(sorted(mem.symptom_tags))
            counter[key] += 1
            samples.setdefault(key, []).append(mem)

        result = []
        for key, count in counter.most_common():
            if count < self.PROMOTE_MIN_OCCURRENCES:
                break
            result.append({
                "pattern": key,
                "occurrences": count,
                "samples": samples[key][:5],
                "needs_human_review": True,   # 永远为 True —— 闸门在这里
            })
        return result

    def promote(self, pattern: str, occurrences: int, samples: list[MemoryRecord],
                approved: bool) -> Optional[MemoryRecord]:
        """把模式提炼成 SOP。只有 approved=True 才会真的写入语义记忆。

        注意：调用方必须显式传入 approved。
        没有默认值 True —— 这是刻意的，**让"需要人工批准"这件事在 API 层面无法被绕过**。
        """
        if not approved:
            return None

        causes = [s.root_cause for s in samples if s.root_cause]
        fixes = [s.resolution for s in samples if s.resolution]
        if not causes and not fixes:
            return None

        sop = MemoryRecord(
            memory_id=f"sem_{_hash(pattern)}",
            tier=MemoryTier.SEMANTIC,
            symptom=f"当出现 {pattern} 类型的症状时",
            symptom_tags=pattern.split("|"),
            root_cause="；".join(sorted(set(causes))[:3]),
            resolution="；".join(sorted(set(fixes))[:3]),
            # 语义记忆的生信度高 —— 它是从多条被验证过的情景记忆里提炼的
            confidence=0.8,
            importance=0.9,
        )
        sop.embedding = self.store.encoder.encode(self.store._embed_text(sop))
        self.store.insert(sop)
        return sop

    # ============================================================
    # ★ P1-2 新接的入口：让这套机制真的被调用起来
    # ============================================================
    # 在 P1-2 之前，`find_patterns` / `promote` 是"写好了但没人调"——
    # 这等于功能没做。**"能跑"和"在跑"是两件事。**
    #
    # 现在补上两个入口：
    #   scan()   —— 观察（幂等，可以随便调）
    #   review() —— 决策（**唯一能写语义记忆的路径**）

    def scan(self) -> list["PendingPromotion"]:
        """扫描高频模式，登记为待审核提案。

        ★ **这个方法只观察，不写语义记忆。**
        它可以被反复调用（任务结束、定时任务、人工触发）而不产生副作用 ——
        因为 `add_promotion` 对同一 pattern 幂等。

        **把"观察"和"决策"分成两个方法**，是为了让自动化的部分
        可以放心地高频执行，而需要人的部分被明确隔离出来。

        如果 scan() 里顺手就 promote 了，那"人工闸门"就只是注释里的承诺。

        ============================================================
        ★ 触发时机：**不要挂在每次任务之后**
        ============================================================
        `docs/06` 的待办里写的是"需要接上调用入口"。这里说明为什么
        **入口不该是主循环**：

        ① **成本**：scan() 要遍历全部情景记忆（O(n)），
           而主循环是热路径 —— 每次任务都扫全表是明显的浪费。
        ② **语义**：SOP 提炼是**低频动作**。"这个模式值得固化成规则"
           这个判断需要模式积累到一定量，不是每个任务结束都该重新评估。

        所以正确的入口是：

            定时任务（推荐）：  cron 每天跑一次 --scan
            人工触发：        需要时手动跑
            阈值触发（进阶）： 每累积 N 次任务触发一次

        **"写好了但没接上"和"接错地方"是两种不同的错误。**
        前者的解法是补一个入口，后者的解法是换一个入口 ——
        而如果不想清楚就接，很容易做成后者。
        """
        from ..models import PendingPromotion, _hash

        created: list[PendingPromotion] = []
        for p in self.find_patterns():
            samples = p["samples"]
            # 给审核人一段摘要 —— 让他不用点进每条样本就能判断
            #
            # ★ 这一点很重要：**审核的可用性决定了闸门会不会被绕过。**
            #   如果审核要翻 5 条记忆才能看懂，实际运行中大家就会"随便点了通过"。
            #   把判断依据直接摆到面前，闸门才真的有效。
            causes = [s.root_cause for s in samples if s.root_cause]
            fixes = [s.resolution for s in samples if s.resolution]
            summary = (
                f"标签组合：{p['pattern']}　出现 {p['occurrences']} 次\n"
                f"  常见根因：{'；'.join(sorted(set(causes))[:2]) or '（未记录）'}\n"
                f"  常见处置：{'；'.join(sorted(set(fixes))[:2]) or '（未记录）'}"
            )
            promo = PendingPromotion(
                promotion_id=f"promo_{_hash(p['pattern'])}",
                pattern=p["pattern"],
                occurrences=p["occurrences"],
                sample_ids=[s.memory_id for s in samples],
                sample_summary=summary,
            )
            if self.store.add_promotion(promo):
                created.append(promo)
        return created

    def review(self, promotion_id: str, approved: bool, reviewer: str,
               note: str = "") -> Optional[MemoryRecord]:
        """审核一条提案。**这是唯一能让模式进入语义层的路径。**

        ★ 参数 `approved` 没有默认值 —— 和 `promote` 一样，
        让"必须明确表态"在 API 层面无法被绕过。

        ★ 返回：
            批准成功 → 新生成的语义记忆
            拒绝     → None（但审核记录会被保存）
            重复审核 → None

        **为什么"拒绝"也要记录？**
        因为同一个模式以后可能又被扫描出来。审核人看到
        "上次有人在 X 时间拒绝过，理由是 Y"，就不用重复判断。
        **审核历史是资产，不是日志。**
        """
        promo = self.store.get_promotion(promotion_id)
        if promo is None:
            return None
        if not promo.is_pending:
            return None          # 已审核过 —— 不允许覆盖历史决定

        sop: Optional[MemoryRecord] = None
        if approved:
            # 从 sample_ids 重建样本（promote 需要它们来提炼根因/处置）
            samples = []
            for mid in promo.sample_ids:
                m = self.store.get(mid)
                if m:
                    samples.append(m)
            sop = self.promote(promo.pattern, promo.occurrences, samples,
                               approved=True)

        # 记录审核结果（无论批准还是拒绝）
        self.store.review_promotion(
            promotion_id, approved, reviewer, note,
            promoted_memory_id=sop.memory_id if sop else "")

        # 记录审核动作到统计，方便回答"闸门有没有在用"
        self.review_stats = getattr(self, "review_stats", {
            "approved": 0, "rejected": 0})
        self.review_stats["approved" if approved else "rejected"] += 1
        return sop

    def pending(self) -> list["PendingPromotion"]:
        """当前待审核队列（入口封装，方便调用方不用直接碰 store）。"""
        return self.store.pending_promotions()
