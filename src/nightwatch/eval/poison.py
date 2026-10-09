"""
Nightwatch · 记忆污染注入与纠正

================================================================================
为什么做这个实验：这是"记忆系统特有的失败模式"
================================================================================
一个没有记忆的 Agent，**不会因为记忆而犯错**。

而一旦有了记忆，就出现了一类全新的失败模式：

    记忆库里存了一条错误信息
      → 它被召回
      → 它影响了判断
      → 判断被写回记忆（作为"经验"）
      → 又被召回 …
      → **错误在系统内部自我强化**

这叫做**记忆污染（memory poisoning）**。它和"模型答错了"有本质区别：

    模型答错  → 一次性的，下次可能对
    记忆污染  → **持续性的**，而且会随时间扩散

**所以"记忆系统"这个卖点本身就带来了一个新的风险面** ——
而大多数讲记忆的项目从不提这件事。

================================================================================
★ 五种污染类型（这是"注入方案"的核心）
================================================================================
不同类型的污染，**危害方式和扩散速度完全不同**。分开分类不是为了好看，
而是因为它们的检测难度和纠正成本差很多：

| 类型 | 说明 | 检测难度 | 为什么危险 |
|---|---|---|---|
| `WRONG_RESOLUTION` | 处置写错 | 中 | 后续照错方法做，失败后可能归因到别处 |
| `WRONG_ROOT_CAUSE` | 根因写错 | 中 | 误导诊断方向，让人查错地方 |
| `WRONG_TAGS` | 标签打错 | **高** | **被错误召回** —— 内容看起来对，只是匹配错了对象 |
| `STALE` | 曾经对、现在过时 | **最高** | 内容完全合理，**只有知道版本变更才能发现** |
| `OVER_GENERALIZED` | 个案当通则 | 高 | 单看每条都对，**组合起来是错的** |

**最值得强调的是 `STALE` 和 `OVER_GENERALIZED`** ——
它们的特点是**"内容本身看起来完全正常"**。
一条写着"加 timeout 参数可解决"的记忆，在 2024 年是对的，
2025 年 API 改了就失效了 —— 但**你从内容上根本看不出来**。

这两种污染无法靠"读一遍记忆库"发现，只能靠**使用后的反馈**。
所以"几次反馈能纠正"这个指标对它们最有意义。

================================================================================
★ 纠正机制：为什么需要"多次"反馈
================================================================================
直觉上，人工说一句"这条错了"就该修好。但实际上：

    第 1 次反馈：纠正了那条记忆
      ↓ 但 Agent 可能已经基于它写下了**派生记忆**（新的错误经验）
      ↓ 派生记忆还在被召回
    第 2 次反馈：纠正派生的那条
      ↓ 但可能还有别的派生
    第 N 次反馈：收敛

**所以"纠正所需反馈次数"衡量的是污染的"传染深度"。**
一次注入如果只污染 1 条记忆，1 次反馈就能修；
如果它已经被用来判断了 5 个问题、每个又写回一条新记忆，
那就需要多次。

**这个指标才是有意义的** —— 它直接反映"记忆系统的抗污染能力"。

（这也解释了为什么本模块要追踪"暴露"：不知道污染影响了谁，
 就不知道纠正要纠正多少条。）
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..models import MemoryRecord, MemoryTier, Outcome


class PoisonType(str, Enum):
    """污染类型 —— 危害方式和检测难度都不同。"""
    WRONG_RESOLUTION = "wrong_resolution"
    WRONG_ROOT_CAUSE = "wrong_root_cause"
    WRONG_TAGS = "wrong_tags"
    STALE = "stale"
    OVER_GENERALIZED = "over_generalized"

    @property
    def detect_difficulty(self) -> str:
        """检测难度 —— 用于报告里说明"这类污染多难发现"。"""
        return {
            PoisonType.WRONG_RESOLUTION: "中（照做失败后能发现）",
            PoisonType.WRONG_ROOT_CAUSE: "中（排查方向偏了才有感觉）",
            PoisonType.WRONG_TAGS: "高（内容对，只是匹配错对象）",
            PoisonType.STALE: "最高（内容合理，只有知道版本变更才能发现）",
            PoisonType.OVER_GENERALIZED: "高（单条都对，组合起来才错）",
        }[self]


@dataclass
class Poison:
    """一次污染注入的记录。"""
    poison_id: str
    poison_type: PoisonType
    memory_id: str                  # 被注入的那条记忆
    issue_id: str = ""              # 针对哪个 issue（用于报告展示）
    injected_at: str = ""
    description: str = ""
    # --- 纠正过程 ---
    feedback_count: int = 0         # 用了多少次反馈才纠正
    corrected_at: str = ""
    corrected_by: str = ""
    # --- 影响范围 ---
    exposed_traces: list[str] = field(default_factory=list)   # 被哪些判断召回使用
    derived_memory_ids: list[str] = field(default_factory=list)  # 由它派生的错误记忆

    @property
    def is_corrected(self) -> bool:
        return bool(self.corrected_at)

    @property
    def spread(self) -> int:
        """扩散度 = 影响的判断数 + 派生的记忆数。

        ★ 为什么把这两个加起来？
        因为它们代表两种不同的"传染"：
          · 影响的判断    → 横向扩散（影响面）
          · 派生的记忆    → 纵向扩散（污染被固化下来，会继续传染）
        后者更麻烦，因为它不依赖原污染存在。
        """
        return len(self.exposed_traces) + len(self.derived_memory_ids)


@dataclass
class FeedbackEvent:
    """一次人工反馈事件（用于统计"第几次反馈才纠正"）。"""
    poison_id: str
    seq: int                        # 这是第几次反馈
    action: str                     # mark_wrong / correct / confirm
    by: str
    note: str = ""
    at: str = field(default_factory=lambda: _now())


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# 注入器
# =============================================================================

class PoisonInjector:
    """往记忆库注入污染。

    ★ 设计原则：**注入必须和真实写入走同一条路径**。

    如果注入器用一条"特殊通道"写进记忆库，那实验就失真了 ——
    因为真实场景下污染是通过**正常的记忆写入流程**进去的
    （比如维护者写错、或者 Agent 从错误结论里学到的）。

    所以 `inject()` 直接调 `store.insert()`，和正常写入没有区别。
    **污染之所以危险，正是因为它和正确记忆长得一模一样。**
    """

    def __init__(self, store):
        self.store = store
        self.poisons: dict[str, Poison] = {}

    def inject(self, poison_type: PoisonType, target_issue_id: str,
               symptom: str, wrong_content: str,
               tags: list[str] | None = None,
               correct_content: str = "") -> Poison:
        """注入一条污染记忆。

        参数：
            poison_type    污染类型
            target_issue_id 针对哪个 issue（用于后续验证）
            symptom        症状描述（照抄真实记忆的写法，让它"看起来正常"）
            wrong_content  错误内容（处置或根因）
            tags           标签
            correct_content 正确内容（记下来，用于对比验证纠正效果）

        ★ 注意 `wrong_content` 的构造要求：**它必须看起来合理**。

        如果注入一条明显荒谬的记忆（"解决方法：删除整个项目"），
        那测出来的"纠正次数"毫无意义 —— 因为现实中没人会写这种记忆，
        而且一眼就能看出错。

        **真实污染的特征是"看起来对"。** 这是本实验有效性的前提。
        """
        mem = MemoryRecord(
            memory_id=f"poison_{uuid.uuid4().hex[:8]}",
            tier=MemoryTier.EPISODIC,
            issue_id=target_issue_id,
            symptom=symptom,
            symptom_tags=list(tags or []),
            root_cause=wrong_content if poison_type is PoisonType.WRONG_ROOT_CAUSE else "（见处置）",
            resolution=wrong_content if poison_type is not PoisonType.WRONG_ROOT_CAUSE else "（见根因）",
            outcome=Outcome.RESOLVED,
            confidence=0.8,          # ★ 给正常置信度 —— 污染不该有明显特征
        )
        self.store.insert(mem)

        p = Poison(
            poison_id=f"p_{uuid.uuid4().hex[:8]}",
            poison_type=poison_type,
            memory_id=mem.memory_id,
            injected_at=_now(),
            description=f"{poison_type.value}: {wrong_content[:60]}",
        )
        self.poisons[p.poison_id] = p
        return p

    # ------------------------------------------------------------ 影响追踪

    def record_exposure(self, poison_id: str, trace_id: str) -> None:
        """记录"某次判断用到了这条污染"。

        ★ 这是"影响范围定位"的数据来源。

        在真实系统里，这个信息来自 trace 的 `memory_recalled` / `memory_used`
        （本项目已经在记忆六环节里记了）——
        所以本实验**不需要改任何生产代码**，只是把已有数据用起来。

        **能被观测的系统才能被诊断。** 如果 trace 里不记"用了哪些记忆"，
        这个实验根本做不了。
        """
        p = self.poisons.get(poison_id)
        if p and trace_id not in p.exposed_traces:
            p.exposed_traces.append(trace_id)

    def record_derived(self, poison_id: str, derived_memory_id: str) -> None:
        """记录"基于污染派生出的新记忆"。

        派生记忆是污染扩散的关键 —— 它**独立于原污染存在**。
        即使删掉原污染，派生记忆还在继续误导。
        """
        p = self.poisons.get(poison_id)
        if p and derived_memory_id not in p.derived_memory_ids:
            p.derived_memory_ids.append(derived_memory_id)

    def detect_exposure(self, trace) -> list[str]:
        """从一条 trace 里检出它碰到了哪些污染。

        ★ 为什么用 trace 而不是猜？
        因为 trace 里记了 `memory_recalled`（召回了哪些）。
        这是**事实记录**，不是推测。
        """
        if not trace:
            return []
        recalled = set(getattr(trace, "memory_recalled", []) or [])
        hit = []
        for pid, p in self.poisons.items():
            if p.memory_id in recalled:
                self.record_exposure(pid, trace.trace_id)
                hit.append(pid)
        return hit

    # ------------------------------------------------------------ 纠正

    def mark_wrong(self, poison_id: str, by: str, note: str = "") -> FeedbackEvent:
        """人工反馈：标记这条记忆是错的。

        ★ 这个动作做两件事，**顺序很重要**：

          ① 降低记忆的可信度（让它不再被高优先召回）
          ② 记录反馈次数

        **为什么不能直接删掉？**
        因为"删除"抹掉了历史 —— 你无法回答"这条错误记忆当初影响了哪些判断"。
        所以本项目采取的是**软纠正**（降低权重 / 标记 archived），
        保留追溯能力。

        （这也和 `docs/06` 里 P1-2 的处理一致：
          "删除是最差的状态表达方式"。）
        """
        p = self.poisons.get(poison_id)
        if not p:
            raise KeyError(f"未知污染：{poison_id}")
        p.feedback_count += 1

        mem = self.store.get(p.memory_id)
        if mem:
            # 软纠正：标记归档（不再被召回）+ 调低置信度
            mem.archived = True
            mem.confidence = 0.1
            self.store.update(mem)

        event = FeedbackEvent(poison_id, p.feedback_count, "mark_wrong", by, note)
        if p.feedback_count == 1:
            # 第一次反馈就标记了原污染 —— 但派生记忆还没处理
            p.corrected_at = _now()
            p.corrected_by = by
        return event

    def correct_derived(self, poison_id: str, by: str, note: str = "") -> FeedbackEvent:
        """纠正派生记忆（污染的"二次处理"）。

        ★ 这个方法的存在本身就是实验要说明的问题：

        **删除原污染 ≠ 污染被清除。** 派生记忆还在。

        实测里会出现这种情况：第一次反馈标记了原污染，
        但 Agent 已经基于它写过新记忆，那条新记忆继续误导 ——
        所以要第二次反馈。
        """
        p = self.poisons.get(poison_id)
        if not p:
            raise KeyError(f"未知污染：{poison_id}")
        p.feedback_count += 1
        for mid in p.derived_memory_ids:
            m = self.store.get(mid)
            if m and not m.archived:
                m.archived = True
                m.confidence = 0.1
                self.store.update(m)
        return FeedbackEvent(poison_id, p.feedback_count, "correct", by, note)

    def is_still_effective(self, poison_id: str) -> bool:
        """检查这条污染是否还在"起作用"（还能被召回）。"""
        p = self.poisons.get(poison_id)
        if not p:
            return False
        mem = self.store.get(p.memory_id)
        if mem and not mem.archived:
            return True
        # 原污染归档了，但派生记忆还在 → 仍然有效
        for mid in p.derived_memory_ids:
            m = self.store.get(mid)
            if m and not m.archived:
                return True
        return False

    # ------------------------------------------------------------ 报告

    def summary(self) -> dict:
        """汇总所有污染的统计。

        ★ 输出的三个核心指标直接对应验收标准：
          · injections           → 注入方案
          · feedback_to_correct  → 纠正所需反馈次数
          · exposure             → 影响的历史判断范围
        """
        out = {
            "注入总数": len(self.poisons),
            "已纠正": sum(1 for p in self.poisons.values() if p.is_corrected),
            "仍有效": sum(1 for p in self.poisons.values()
                       if self.is_still_effective(p.poison_id)),
            "总暴露次数": sum(len(p.exposed_traces) for p in self.poisons.values()),
            "总派生记忆": sum(len(p.derived_memory_ids) for p in self.poisons.values()),
            "按类型": {},
        }
        for pt in PoisonType:
            group = [p for p in self.poisons.values() if p.poison_type is pt]
            if not group:
                continue
            fb = [p.feedback_count for p in group]
            out["按类型"][pt.value] = {
                "数量": len(group),
                "平均反馈次数": round(sum(fb) / len(fb), 2) if fb else 0,
                "总暴露": sum(len(p.exposed_traces) for p in group),
                "总派生": sum(len(p.derived_memory_ids) for p in group),
                "检测难度": pt.detect_difficulty,
            }
        return out
