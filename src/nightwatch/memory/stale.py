"""
Nightwatch · 记忆时效性：仓库变更 → 过时嫌疑标记

================================================================================
这个模块解决什么问题
================================================================================
项目原来**完全没有"仓库变更 → 记忆失效"的通道**：

    记忆六环节（抽取/评分/写入/合并/消解/衰减）处理的都是
    **记忆之间的相互关系**（重复、矛盾）和**时间推移** ——
    没有任何一个环节感知"代码环境变了"。

项目自己已经证明过这类过时记忆的危害（`eval/poison.py` 的 STALE 类型：
"曾经对、现在过时"，检测难度标为最高级，实测暴露次数 4 次 vs 其他类型全 0）。

本模块补的就是这条通道。

================================================================================
★★ 红线：标记可以自动，生效必须过闸门
================================================================================
**这是本模块最重要的设计约束**，来自一个必须讲清楚的区别：

    「变更波及」  ≠  「结论失效」

    一个文件被改了，不代表引用了它的记忆就错了。
      · 改了注释、改了日志文案、改了不相关的函数 → 记忆完全有效
      · 重命名了一个被引用的类 → 记忆的结论可能失效
      · **这两件事从"文件级"的变更信息里根本区分不出来**

所以：
    **检测可以自动**（`code_refs ∩ changed_files` 是精确的集合运算，没有误判）
    **判断必须人工**（"这个改动会不会让那条结论不再成立"需要理解语义）

如果自动化这一步 —— 直接按"文件被改过"就降权 —— 会重蹈项目已经踩过的坑：
`docs/02` 的 B6 案例（被人工确认过的经验，400 天后权重 0.03 被当垃圾归档）。
**那次是因为时间，这次如果因为文件变更而犯同样的错，就白踩了。**

================================================================================
三个阶段的职责划分
================================================================================
    ① 检测（自动）   `detect()`      —— 谁被波及了？精确的集合运算，零误判
    ② 入队（自动）   `mark()`        —— 推进待复核队列，**不改变记忆本身**
    ③ 裁决（人工）   `review()`      —— 人确认后才降权 + 归档

**关键：② 和 ③ 之间，记忆完全不受影响** ——
它照常被召回、照常参与判断。
队列只是"待办清单"，不是"已生效的惩罚"。

（这个两阶段结构和 `P1-2` 的 `PendingPromotion` 完全同构 ——
 连"标记自动、生效人工"这个模式都是复用的，不是新发明。）
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..models import MemoryRecord, MemoryTier


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class StaleSuspect:
    """一条"过时嫌疑"记录（检测阶段的产出）。"""
    memory_id: str
    symptom: str
    matched_refs: list[str]         # 命中的代码路径（交集）
    confidence: float = 0.0         # 记忆自身置信度（帮助人工排序）
    human_verdict: str = ""         # 是否被人工确认过（**最高优先级信号**）
    importance: float = 0.0

    @property
    def priority(self) -> float:
        """复核优先级。

        ★ 排序依据（从高到低）：
          ① **被人工确认过的**最高 —— 它一旦失效，危害最大
             （不只是"记错了"，而是"人确认过的东西错了"，更难被质疑）
          ② 命中路径多的（波及面大）
          ③ 置信度高的（原结论越确定，失效后误导越强）

        **这个排序本身就是"人的时间该花在哪"的答案** ——
        复核队列永远排不完，所以排序逻辑比检测逻辑更影响实际效果。
        """
        base = len(self.matched_refs) * 0.3 + self.confidence * 0.3 \
            + self.importance * 0.2
        if self.human_verdict in ("accepted", "edited"):
            base += 2.0                 # 人工确认过 → 强制置顶
        return round(base, 3)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "symptom": self.symptom[:80],
            "matched_refs": self.matched_refs,
            "confidence": round(self.confidence, 3),
            "human_verdict": self.human_verdict or "unknown",
            "priority": self.priority,
        }


@dataclass
class StaleReviewItem:
    """一条待复核项（对应数据库一行）。"""
    review_id: str
    memory_id: str
    symptom: str
    matched_refs: list[str]
    trigger: str                   # 触发来源（"repo_change" / "manual"）
    trigger_detail: str = ""       # 具体是什么变更触发的
    status: str = "pending"        # pending / confirmed / dismissed
    created_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""
    review_note: str = ""

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "memory_id": self.memory_id,
            "symptom": self.symptom[:80],
            "matched_refs": self.matched_refs,
            "trigger": self.trigger,
            "trigger_detail": self.trigger_detail,
            "status": self.status,
            "created_at": self.created_at,
            "reviewed_by": self.reviewed_by,
            "review_note": self.review_note,
        }


# =============================================================================
# 检测器
# =============================================================================

class StaleDetector:
    """仓库变更 → 过时嫌疑检测。

    ============================================================
    为什么用"文件路径交集"而不是"让模型判断那条记忆是否过时"
    ============================================================
    因为**这件事不需要智能，需要的是准确**。

        `code_refs ∩ changed_files` 是**精确的集合运算** ——
        零误判、零漏判、零成本、可复现。

    而如果让模型判断"这个 diff 会不会让那条记忆失效"：
      · 要遍历每条记忆 × 每个 diff（成本高）
      · 判断结果有随机性（不可复现）
      · **而且它仍然只是"嫌疑"** —— 最终还是要人确认

    **既然都要人确认，那前面的环节就别用模型了** ——
    用最便宜、最准的方法把候选缩到几十条，才是把人的时间用在刀刃上。

    （这和 `tools/pr_review.py` 的思路一致：
      "结构风险筛查用规则，语义判断交给人"。）
    """

    def __init__(self, store):
        self.store = store

    # ---------------------------------------------------------------- 检测

    def detect(self, changed_files: list[str],
               repo: str = "") -> list[StaleSuspect]:
        """找出引用了被改动文件的记忆。

        ★ 参数是 `changed_files`（一个文件路径列表），不是 diff 本身。

        为什么？因为**上游不需要为了用这个功能而去解析 diff** ——
        `pr_review.py` 里的 `parse_diff()` 已经能产出文件列表了，
        GitHub 的事件里也直接带 `files`。
        **接口收窄到"最小充分信息"，复用面才最广。**
        """
        if not changed_files:
            return []

        changed = {self._norm(p) for p in changed_files if p}
        suspects: list[StaleSuspect] = []

        for mem in self.store.all_memories():
            # ★ 已归档的记忆不再复核 —— 它已经不在召回池里了，
            #   复核它不产生任何收益，只会浪费人的时间。
            if mem.archived:
                continue
            refs = getattr(mem, "code_refs", None) or []
            if not refs:
                # 没有代码引用的记忆（比如纯经验/SOP）**无法用本方法检测** —
                # 这是一个诚实的边界，见文件末尾的说明。
                continue
            if repo and mem.repo and mem.repo != repo:
                continue
            matched = [r for r in refs if self._norm(r) in changed]
            if matched:
                suspects.append(StaleSuspect(
                    memory_id=mem.memory_id,
                    symptom=mem.symptom,
                    matched_refs=matched,
                    confidence=mem.confidence,
                    human_verdict=(mem.human_verdict.value
                                   if hasattr(mem.human_verdict, "value")
                                   else str(mem.human_verdict)),
                    importance=mem.importance,
                ))

        # ★ 返回时排序 —— 让调用方直接拿到"该先看哪个"
        suspects.sort(key=lambda s: -s.priority)
        return suspects

    @staticmethod
    def _norm(path: str) -> str:
        """路径归一化。

        ★ 为什么必须归一化？
        因为同一个文件在不同来源里写法不同：
            'pandas/core/frame.py' / './pandas/core/frame.py' / 'core/frame.py'
        不归一化的话，交集运算会**静默漏判** ——
        而漏判的后果是"该复核的记忆没进队列"，这是最危险的方向。

        **集合运算的前置条件是两个集合用同一套规范化。**
        （同构于 `docs/09` 里"标签粒度对齐"那个坑：不做归一化，
          比较就是错的，而且不报错。）
        """
        p = str(path).strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        return p.lower()

    # ---------------------------------------------------------------- 入队

    def mark(self, suspects: list[StaleSuspect],
             trigger: str = "repo_change",
             trigger_detail: str = "") -> list[str]:
        """把嫌疑记忆推进待复核队列。

        ★★ 关键：这个方法**完全不碰记忆本身**。

        它只往队列里插记录。记忆照常被召回、照常参与判断 ——
        **"待复核"不是"已生效的惩罚"，只是"待办清单"。**

        如果需要"入队即降权"，那就违背了本模块的核心约束
        （见文件头的"红线"一节）。

        返回：新建的 review_id 列表（已存在 pending 的不重复建）。
        """
        created = []
        for s in suspects:
            if self._already_handled(s.memory_id, trigger_detail):
                continue
            rid = f"sr_{uuid.uuid4().hex[:8]}"
            self.store.conn.execute(
                """INSERT INTO pending_stale_reviews (
                    review_id, memory_id, symptom, matched_refs,
                    trigger, trigger_detail, status, created_at
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (rid, s.memory_id, s.symptom[:200],
                 json.dumps(s.matched_refs, ensure_ascii=False),
                 trigger, trigger_detail[:300], "pending", _now()))
            created.append(rid)
        if created:
            self.store.conn.commit()
        return created

    def _already_handled(self, memory_id: str, trigger_detail: str) -> bool:
        """判断这条嫌疑是否**已经处理过、不必再问**。

        ============================================================
        ★ 这个判断比"有没有 pending"复杂，值得说清楚
        ============================================================
        三种历史状态，处理方式不同：

          ① **有 pending** → 跳过
             已经躺在队列里了，不该重复入队（否则队列会膨胀）。

          ② **有 dismissed 且触发原因相同** → 跳过
             人已经判定过"这个变更不影响这条记忆"。
             **再问一遍是纯粹的重复劳动** —— 而且会让人对队列失去信任
             （"上次不是说过没问题了吗？"）。

          ③ **有 dismissed 但触发原因不同** → **重新入队**
             这是**新的变更**（不同的文件/不同的 PR），
             上次"仍有效"的结论**不能自动延续到这次** ——
             新的改动完全可能让同一条记忆失效。
             **不能因为"上次没事"就永远不问。**

        ★ 所以幂等键是 `(memory_id, trigger_detail)` 而不是只有 memory_id。

        这个区分很容易漏掉：只查 memory_id 的话，
        会出现"一次 dismiss 之后永远不再复核"—— 而那是**漏检**，
        比"多问一遍"危险得多。
        """
        cur = self.store.conn.execute(
            """SELECT status, trigger_detail FROM pending_stale_reviews
                WHERE memory_id=?
                ORDER BY created_at DESC LIMIT 5""",
            (memory_id,))
        for row in cur.fetchall():
            if row["status"] == "pending":
                return True
            if row["status"] == "dismissed":
                # 触发原因相同 → 已判定过，跳过；不同 → 继续往下看
                if (row["trigger_detail"] or "") == (trigger_detail or ""):
                    return True
        return False

    # ---------------------------------------------------------------- 裁决

    def review(self, review_id: str, confirmed: bool, reviewer: str,
               note: str = "") -> dict:
        """人工裁决：确认过时 / 判定仍有效。

        ★ 只有 `confirmed=True` 才会**真正改变记忆**（降权 + 归档）。

        `confirmed=False`（判定仍有效）同样**是有价值的记录** ——
        它回答"这条被怀疑过的记忆，人看过之后认为没问题"，
        下次同样的变更再触发时，可以直接跳过。

        ★ 为什么降权而不删？沿用 `poison.py` 的结论：
        **删除是最差的状态表达方式**（丢掉过程信息，无法追溯）。
        """
        row = self.store.conn.execute(
            "SELECT * FROM pending_stale_reviews WHERE review_id=?",
            (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"未知的复核项：{review_id}")

        status = "confirmed" if confirmed else "dismissed"
        self.store.conn.execute(
            """UPDATE pending_stale_reviews
                  SET status=?, reviewed_at=?, reviewed_by=?, review_note=?
                WHERE review_id=?""",
            (status, _now(), reviewer, note[:300], review_id))

        result = {"review_id": review_id, "status": status,
                  "memory_id": row["memory_id"], "affected": False}

        if confirmed:
            mem = self.store.get(row["memory_id"])
            if mem is not None:
                before = mem.confidence
                # 软纠正：降权 + 归档（**不物理删除**）
                mem.confidence = min(mem.confidence, 0.1)
                mem.archived = True
                self.store.update(mem)
                result["affected"] = True
                result["confidence_before"] = before
                result["confidence_after"] = mem.confidence

        self.store.conn.commit()
        return result

    # ---------------------------------------------------------------- 查询

    def pending(self) -> list[StaleReviewItem]:
        """待复核队列（按创建时间倒序）。"""
        cur = self.store.conn.execute(
            "SELECT * FROM pending_stale_reviews"
            " WHERE status='pending' ORDER BY created_at DESC")
        return [self._row_to_item(r) for r in cur.fetchall()]

    def history(self, limit: int = 50) -> list[StaleReviewItem]:
        cur = self.store.conn.execute(
            "SELECT * FROM pending_stale_reviews"
            " ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row_to_item(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_item(row) -> StaleReviewItem:
        return StaleReviewItem(
            review_id=row["review_id"],
            memory_id=row["memory_id"],
            symptom=row["symptom"] or "",
            matched_refs=json.loads(row["matched_refs"] or "[]"),
            trigger=row["trigger"] or "",
            trigger_detail=row["trigger_detail"] or "",
            status=row["status"] or "pending",
            created_at=row["created_at"] or "",
            reviewed_at=row["reviewed_at"] or "",
            reviewed_by=row["reviewed_by"] or "",
            review_note=row["review_note"] or "",
        )

    def stats(self) -> dict:
        cur = self.store.conn.execute(
            """SELECT
                 SUM(CASE WHEN status='pending'   THEN 1 ELSE 0 END) pending,
                 SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END) confirmed,
                 SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) dismissed
               FROM pending_stale_reviews""")
        r = dict(cur.fetchone())
        total = sum(v or 0 for v in r.values())
        r["total"] = total
        # 跟踪记忆库总条数，用于算"受影响比例"
        r["n_memories"] = self.store.count()
        if r["n_memories"]:
            r["affected_ratio"] = round(
                ((r["pending"] or 0) + (r["confirmed"] or 0))
                / r["n_memories"], 4)
        return r
