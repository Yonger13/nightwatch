"""
Nightwatch · 结果反馈闭环：用"判断错了"反推"记忆可能过时了"

================================================================================
这个模块解决什么问题（以及它为什么不可替代）
================================================================================
`stale.py` 用**代码变更**检测过时记忆：

    code_refs ∩ changed_files  →  嫌疑

**但那只能抓"结构性过时"** —— 文件被改了、被重构了、路径变了。

**有一类过时它完全抓不到**：

    一个 API 还在、签名没变、文件没被碰 —— **但语义变了**。

    例：
      · `dropna()` 的默认行为改了（从"删所有含 NaN 的行"变成"按列删"）
      · 某个函数的边界条件从"含上界"改成"不含上界"
      · 依赖的下游库升级，相同调用现在有不同结果

    路径交集**完全不命中**（文件根本没被改），
    但用它得出的结论已经不对了。

**本模块补的就是这一类** —— 它不看代码，只看**结果**：

    这条记忆被用了 → 那次判断**错了** → 说明它可能已经过时

================================================================================
★★ 为什么这个信号比"代码变更"更难做，但也更值钱
================================================================================
**难在归因**：

    判断错了，可能是三个原因：
      ① 记忆过时了（它引用的知识不再成立）        ← 我们要抓的
      ② 模型这一次自己判断失误（记忆没问题）      ← 噪声
      ③ 信息本来就不足（谁都判断不了）            ← 噪声

    从"一次错误"里**分不出**是哪个原因。

**值钱在覆盖**：它是**唯一不依赖"知道代码改了什么"**的信号。

**所以设计的核心问题是"怎么把噪声压下去"** —— 答案是**统计**：

    · 单次错误不算数（可能只是模型失手）→ 要求 `misuse >= 2`
    · 用**误诊率**而不是误诊次数（用 100 次错 2 次是好事，用 2 次错 2 次才是问题）
    · 用**拉普拉斯平滑**处理小样本（用 1 次错 1 次 → 1/4，不足以定罪）

**这三条合起来的效果**：只有"反复出错"的记忆才会进复核队列。

================================================================================
★ 与 stale.py 的关系：互补，不是替代
================================================================================
    stale.py    ← 结构信号（代码变了）      快、准、但覆盖窄
    feedback.py ← 语义信号（结果错了）      覆盖广、但需要积累、有噪声

两者**共用同一个待复核队列**（`pending_stale_reviews`），只是 `trigger` 不同：
    trigger='repo_change'      来自 stale.py
    trigger='feedback_misuse'  来自本模块

**人看到的仍是同一个队列** —— 这是刻意的：
审核员不该关心"这条是怎么被发现的"，只该关心"它是不是真的过时了"。

================================================================================
★ 红线（和 stale.py 完全一致）
================================================================================
**累加嫌疑分是自动的；把记忆降权归档必须过人工闸门。**

误诊率高的记忆**仍然照常参与召回** —— 直到人确认。
理由还是那一条：「判断错了」也 ≠「记忆失效」（可能是模型自己的问题）。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ..models import MemoryRecord
from .stale import StaleDetector, StaleSuspect


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 阈值与公式（集中在这里，方便标定）
# =============================================================================

#: 拉普拉斯平滑项。加 3 的效果：用 1 次错 1 次 → 1/4 = 0.25（不足以定罪）
SMOOTHING = 3.0

#: 嫌疑分阈值（平滑后的误诊率）
DEFAULT_SCORE_THRESHOLD = 0.25

#: 最少误诊次数（**单次错误不足以定罪** —— 可能是模型自己失手）
DEFAULT_MIN_MISUSE = 2


@dataclass
class MisuseRecord:
    """一条误诊明细。"""
    event_id: str
    issue_id: str
    memory_id: str
    evidence: str
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "issue_id": self.issue_id,
            "memory_id": self.memory_id, "evidence": self.evidence[:120],
            "created_at": self.created_at,
        }


class MisuseTracker:
    """误诊追踪器：记录"记忆被用了但判断错了"，并按阈值推进复核队列。

    用法（在评测/对账脚本里）：

        tracker = MisuseTracker(store)
        flagged = tracker.record_triage_result(
            issue_id="#123",
            memory_used=["ep_abc", "ep_def"],   # trace.memory_used
            correct=False,                       # 这次判断对不对
            evidence="预测 Bug，真实 Build",
        )
        # flagged = 这次因为累积误诊而**新进复核队列**的记忆 ID
    """

    def __init__(self, store,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 min_misuse: int = DEFAULT_MIN_MISUSE):
        self.store = store
        self.score_threshold = score_threshold
        self.min_misuse = min_misuse
        # 复用 stale 的闸门（共用同一个待复核队列）
        self.detector = StaleDetector(store)

    # ---------------------------------------------------------------- 记录

    def record_triage_result(self, issue_id: str, memory_used: list[str],
                             correct: bool,
                             evidence: str = "") -> list[str]:
        """记录一次完整的分诊结果。

        ★ 注意这里**只处理"错误"**的情况。

        为什么"正确"不需要记录？
          因为"被使用次数"（`use_count`）已经在主链路里更新了
          （`react.py` → `store.touch_access`）。
          本模块只补那个主链路不产生的信号：**误诊**。

        **不做重复计数**是这类统计功能最容易搞错的地方 ——
        两处都加就会把 use_count 算成两倍，让误诊率凭空减半。

        ★ 返回值语义（这里踩过一次歧义，明确一下）：

            返回的是**实际新入队的复核项 ID**（`_enqueue` 的结果），
            **不是**"达阈值的记忆 ID 列表"。

            两者的区别在于：达阈值 ≠ 入队 ——
            如果这条记忆已经在队列里（或人已经判定过），
            它会被幂等逻辑拦下，**不会产生新的复核项**。

            调用方通常只关心"有没有新东西需要人看" ——
            所以返回入队结果更有用。要拿"达阈值的"可以用 `suspects()`。
        """
        if correct:
            # 判断正确 → 不产生新信息（use_count 已在主链路更新）
            return []

        #: 达阈值的记忆（下一步过滤幂等后才是真正入队的）
        flagged: list[str] = []
        for mid in (memory_used or []):
            if not mid:
                continue
            mem = self.store.get(mid)
            if mem is None or mem.archived:
                # 记忆不存在或已归档 → 不必再记
                continue

            # 1) 明细（可追溯："这条是被哪几次误诊牵连的"）
            self.store.conn.execute(
                """INSERT INTO misuse_events
                   (event_id, issue_id, memory_id, evidence, created_at)
                   VALUES (?,?,?,?,?)""",
                (f"me_{uuid.uuid4().hex[:8]}", issue_id, mid,
                 evidence[:300], _now()))

            # 2) 汇总计数（快速排序/过滤用）
            mem.misuse_count = int(getattr(mem, "misuse_count", 0)) + 1
            self.store.update(mem)

            # 3) 检查是否达到"该进复核队列"的阈值
            if self.is_suspect(mem):
                flagged.append(mid)

        self.store.conn.commit()
        if not flagged:
            return []
        # 返回**实际入队**的结果（幂等过滤后的）
        return self._enqueue(flagged, issue_id, evidence)

    # ---------------------------------------------------------------- 评分

    @staticmethod
    def suspect_score(mem: MemoryRecord) -> float:
        """嫌疑分 = 平滑后的误诊率。

        ============================================================
        公式：misuse / (use + SMOOTHING)
        ============================================================
        为什么用**比率**而不是**次数**：

            记忆 A：用 100 次，错 2 次  →  2/103 = 0.019   健康
            记忆 B：用 3 次，错 2 次    →  2/6   = 0.333   可疑

            如果只看次数，A 和 B 的 misuse 都是 2，分不出来。
            **但它们的性质完全不同。**

        为什么加平滑项：

            用 1 次错 1 次 → 1/(1+3) = 0.25  ← 不到 0.25 的高阈值
            用 2 次错 2 次 → 2/(2+3) = 0.40  ← 过了

            小样本的"100% 错误率"看着吓人，但证据其实不足 ——
            平滑项就是把这个直觉编码进去。

        ★ 兜底：`use_count` 可能小于 `misuse_count`
          （理论上不该发生，但历史数据/异常路径可能有）。
          取 `max` 避免算出 > 1 的分数 —— **不合理的数字要夹住，
          否则它会在阈值判断里产生"意外总是通过"的效果。**
        """
        misuse = int(getattr(mem, "misuse_count", 0))
        if misuse <= 0:
            return 0.0
        use = max(int(getattr(mem, "use_count", 0)), misuse)
        return round(misuse / (use + SMOOTHING), 4)

    def is_suspect(self, mem: MemoryRecord) -> bool:
        """是否达到"该进复核队列"的程度。

        **两个条件都要满足**：

            misuse_count >= min_misuse     ← 单次错误不足以定罪
            score >= score_threshold       ← 误诊率要够高
        """
        misuse = int(getattr(mem, "misuse_count", 0))
        if misuse < self.min_misuse:
            return False
        return self.suspect_score(mem) >= self.score_threshold

    # ---------------------------------------------------------------- 入队

    def _enqueue(self, memory_ids: list[str], issue_id: str,
                 evidence: str) -> list[str]:
        """把达阈值的记忆推进待复核队列（复用 stale 的闸门）。

        ============================================================
        ★★ 这里踩了一个幂等的坑，值得完整记录
        ============================================================
        第一版把 `trigger_detail` 写成了每次误诊的具体信息：

            trigger_detail = f"{issue_id} 判断错误：{evidence}"

        结果**幂等完全失效** —— 每次误诊的 issue_id 和 evidence 都不同，
        于是 `_already_handled` 每次都判定"这是新触发"，**重复入队**。
        实测：同一批记忆被反复推进队列，队列长度一路涨。

        **根因是两条线的"幂等语义"不同**：

            repo_change 线    trigger_detail = 一批文件（同样的变更 → 同样的 detail）✓
            feedback 线       trigger_detail = 每次误诊的具体信息（每次都不同）✗

        **修法：用稳定的类型标识做 trigger_detail，具体信息放进明细表。**

            队列里：trigger_detail = "feedback_misuse"（稳定 → 幂等生效）
            明细里：misuse_events 表存"哪次误诊、什么证据"（完整可追溯）

        **这个分工是对的**：
          · 队列需要的是"这条记忆该不该看"（去重维度 = 记忆身份）
          · 明细需要的是"为什么怀疑它"（去重维度 = 无，全都要留）

        把两者混在一个字段里，就会两头不讨好 ——
        要么幂等失效（第一版），要么丢失追溯信息。

        ============================================================
        ★ 关于"人判定过之后还继续误诊"怎么办
        ============================================================
        有人可能觉得：判定"记忆没问题"之后，如果它**继续被误诊**，
        是不是该重新问一遍？

        **不需要。** 因为：
          · 人已经确认过"这条记忆本身是正确的"
          · 那继续误诊的原因就**不在记忆上** ——
            要么是模型能力问题，要么是信息不足
          · **而这两件事都该由别的手段解决，不该再占用人的复核时间**

        这条推理和 `stale.py` 里"新变更要重新问"是**不同**的：
          · 新变更 → 记忆赖以成立的**前提**变了 → 值得重新评估
          · 继续误诊 → 前提没变，只是结果不好 → 前提不需要重新评估

        **判断标准是"前提变了吗"，不是"结果好不好"。**
        """
        suspects = []
        for mid in memory_ids:
            mem = self.store.get(mid)
            if mem is None:
                continue
            suspects.append(StaleSuspect(
                memory_id=mid,
                symptom=mem.symptom,
                matched_refs=[],        # 结果反馈线没有"命中路径"
                confidence=mem.confidence,
                human_verdict=(mem.human_verdict.value
                               if hasattr(mem.human_verdict, "value")
                               else str(mem.human_verdict)),
                importance=mem.importance,
            ))
        return self.detector.mark(
            suspects, trigger="feedback_misuse",
            # ★ 稳定的类型标识（不是具体信息）—— 幂等靠它，
            #   具体信息在 misuse_events 明细表里
            trigger_detail="feedback_misuse")

    # ---------------------------------------------------------------- 查询

    def suspects(self, limit: int = 30) -> list[dict]:
        """当前所有达阈值的记忆（按嫌疑分降序）。

        用于人工巡检："哪些记忆一直在误导我们？"
        """
        out = []
        for mem in self.store.all_memories():
            if mem.archived:
                continue
            if self.is_suspect(mem):
                out.append({
                    "memory_id": mem.memory_id,
                    "symptom": mem.symptom[:70],
                    "misuse_count": int(getattr(mem, "misuse_count", 0)),
                    "use_count": mem.use_count,
                    "suspect_score": self.suspect_score(mem),
                    "human_verdict": (mem.human_verdict.value
                                      if hasattr(mem.human_verdict, "value")
                                      else str(mem.human_verdict)),
                })
        out.sort(key=lambda x: -x["suspect_score"])
        return out[:limit]

    def history(self, memory_id: str = "", limit: int = 50) -> list[MisuseRecord]:
        """误诊明细（可按记忆过滤）。"""
        if memory_id:
            cur = self.store.conn.execute(
                "SELECT * FROM misuse_events WHERE memory_id=?"
                " ORDER BY created_at DESC LIMIT ?", (memory_id, limit))
        else:
            cur = self.store.conn.execute(
                "SELECT * FROM misuse_events ORDER BY created_at DESC LIMIT ?",
                (limit,))
        return [MisuseRecord(
            event_id=r["event_id"], issue_id=r["issue_id"],
            memory_id=r["memory_id"], evidence=r["evidence"] or "",
            created_at=r["created_at"] or "") for r in cur.fetchall()]

    def stats(self) -> dict:
        """统计（用于回答"这个机制在生产里是空转的吗"）。"""
        total_events = self.store.conn.execute(
            "SELECT COUNT(*) FROM misuse_events").fetchone()[0]
        n_mem = self.store.count()
        n_with_misuse = self.store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE misuse_count > 0"
        ).fetchone()[0]
        n_suspect = len(self.suspects(limit=10**6))
        n_flagged = self.store.conn.execute(
            "SELECT COUNT(*) FROM pending_stale_reviews"
            " WHERE trigger='feedback_misuse'").fetchone()[0]
        return {
            "misuse_events": total_events,
            "memories": n_mem,
            "memories_with_misuse": n_with_misuse,
            "suspects": n_suspect,
            "flagged_to_review": n_flagged,
            "score_threshold": self.score_threshold,
            "min_misuse": self.min_misuse,
            "smoothing": SMOOTHING,
        }
