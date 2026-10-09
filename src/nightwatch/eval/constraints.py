"""
Nightwatch · 硬约束判定器

================================================================================
这个模块解决什么问题
================================================================================
要验证"长程一致性"，必须先回答一个问题：

    **"约束还在不在"这件事，怎么测？**

如果靠模型自评（"你觉得你遵守约束了吗"），那测出来的不是一致性，
是模型的自信心 —— 而这两者经常不一致（模型越跑偏越自信）。

所以这里的核心设计是：**把每条硬约束翻译成可判定的规则。**
只用规则的判定结果，不用模型的自我报告。

================================================================================
为什么要做这个实验（它的价值在哪）
================================================================================
这个项目有一个核心设计主张：

    **把约束放进"任务状态卡"，每轮重新注入，就不怕长任务里指令漂移。**

但如果只是"我这么设计了"，那它只是一个说法。要证明它有效，需要：

    对照组（关掉状态卡重注入）：约束遵守率随轮数增加而**下降**
    实验组（开启状态卡重注入）：约束遵守率随轮数增加**保持稳定**

**这个对照实验就是把"设计主张"变成"可验证结论"的方式。**

这也是这个模块存在的理由 —— 没有它，长程一致性只是个形容词。

================================================================================
一个重要的诚实说明
================================================================================
这些约束是**可判定的近似**，不等于"回答质量好"。

比如"必须给出引用"只检查 cites 非空，不检查引用是否恰当。
理由：**可判定的弱指标，胜过不可判定的强指标。**
一个能自动跑、结论稳定的近似指标，比一个需要人工判断的完美指标更有工程价值 ——
因为前者可以进 CI，后者只能偶尔抽查。

这个取舍本身也值得在面试里讲。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ..models import Issue, Trace


# =============================================================================
# 违规记录
# =============================================================================

@dataclass
class Violation:
    """一次约束违反。"""
    constraint: str          # 哪条约束
    detail: str              # 具体怎么违反的
    severity: str = "medium" # low / medium / high

    def __str__(self) -> str:
        return f"[{self.severity}] {self.constraint}: {self.detail}"


@dataclass
class ConstraintReport:
    """一次处理的约束遵守情况。"""
    issue_id: str
    n_steps: int = 0
    violations: list[Violation] = field(default_factory=list)
    checks_run: int = 0

    @property
    def adherence(self) -> float:
        """约束遵守率 = 通过的检查数 / 总检查数。

        用比例而不是绝对数，因为不同场景下能触发的检查数不同
        （比如没有历史相似 issue 时，"判重优先"这条检查就不适用）。
        """
        if self.checks_run == 0:
            return 1.0
        return 1.0 - len(self.violations) / self.checks_run

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "n_steps": self.n_steps,
            "checks_run": self.checks_run,
            "violations": [str(v) for v in self.violations],
            "adherence": round(self.adherence, 4),
        }


# =============================================================================
# 判定器
# =============================================================================

class ConstraintChecker:
    """把硬约束翻译成可判定的规则。

    四条约束，全部只用规则判定，不调用模型：

        ① 判重优先于分类   —— 检索到高相似度历史时，必须给出重复判定
        ② 必须给出依据     —— 结论不能没有引用
        ③ 置信度与证据匹配 —— 没有证据时必须低置信（防止"自信地胡说"）
        ④ 不得重复调用同一工具 —— 参数相同的连续调用视为失控

    每条都对应一个真实的业务风险，不是凭空想的：
        ① 漏判重复 → 维护者重复处理；误判重复 → 新问题被当已解决而关闭
        ② 无依据的结论 → 维护者无法验证，只能整体不信
        ③ 高置信的错误 → 比低置信的错误危险得多（会误导决策）
        ④ 死循环 → 烧 token，而且永远不收敛
    """

    # 置信度阈值：没有引用的结论，置信度不得超过这个值
    UNSUPPORTED_CONFIDENCE_CAP = 0.3

    def check(self, result, trace: Trace, issue: Issue) -> ConstraintReport:
        """对一次处理做全部检查。

        参数：
            result —— TriageResult
            trace  —— 完整的执行轨迹（要从里面读工具调用记录）
            issue  —— 原始 issue
        """
        report = ConstraintReport(
            issue_id=issue.issue_id,
            n_steps=trace.n_steps if trace else 0,
        )

        # 收集工具调用的输出（用于判断"有没有高相似度历史"）
        tool_outputs = self._collect_tool_outputs(trace)

        self._check_duplicate_priority(result, tool_outputs, report)
        self._check_citation(result, report)
        self._check_confidence_evidence(result, report)
        self._check_tool_loop(trace, report)

        return report

    # ------------------------------------------------------------ ① 判重优先

    def _check_duplicate_priority(self, result, tool_outputs: list[dict],
                                  report: ConstraintReport) -> None:
        """约束①：如果检索到高度相似的历史 issue，结论里必须有重复判定。

        ============================================================
        为什么这条重要（两类错误的代价不对称）
        ============================================================
        漏判重复 → 维护者重复处理一遍，浪费一点时间（轻）
        误判重复 → 新问题被标记为"已解决"，**永远不会被处理**（重）

        所以这里只检查"该判重的时候判了没有"（防漏判），
        不检查"判重判得对不对"（那是 duplicate 指标的事）。

        **检查数只有在"确实检索到高相似度历史"时才 +1** —— 否则这道题不适用，
        计入分母会稀释其他约束的违规。

        这是设计可判定指标时的一个关键细节：
        **不适用的检查不应该计入分母**，否则遵守率会被"不适用"的检查拉高。
        """
        retrieved_dup = None
        for out in tool_outputs:
            if isinstance(out, dict) and out.get("likely_duplicate"):
                retrieved_dup = out["likely_duplicate"]
                break

        if not retrieved_dup:
            return          # 没有高相似度历史 → 这条约束不适用，不计入分母

        report.checks_run += 1

        # 判定：输出里有没有体现"重复"？
        has_dup_field = bool(getattr(result, "duplicate_of", None))
        labels = getattr(result, "predicted_labels", []) or []
        has_dup_label = any("dup" in str(l).lower() for l in labels)

        if not (has_dup_field or has_dup_label):
            report.violations.append(Violation(
                constraint="判重优先于分类",
                detail=(f"检索到高相似度历史 issue {retrieved_dup}，"
                        f"但结论中未给出重复判定"
                        f"（labels={labels}, duplicate_of=None）"),
                severity="high",
            ))

    # ------------------------------------------------------------ ② 必须给出依据

    def _check_citation(self, result, report: ConstraintReport) -> None:
        """约束②：结论必须带引用。

        **为什么"没引用"是硬约束而不是建议**：

        维护者要判断这条建议可不可信，唯一的方式是看它的依据。
        一条没有依据的建议，无论对错，都无法被验证 ——
        所以它在业务流程里的实际价值接近于零（维护者只能选择整体不信）。

        这个道理可以迁移：**任何要交给别人做决策的输出，都必须带可验证的依据。**
        """
        report.checks_run += 1

        cites = getattr(result, "cites", None) or []
        if not cites:
            report.violations.append(Violation(
                constraint="必须给出引用依据",
                detail=f"结论未附带任何引用（cites 为空）。摘要：{result.summary[:80]}",
                severity="medium",
            ))

    # ------------------------------------------------------------ ③ 置信度匹配

    def _check_confidence_evidence(self, result,
                                   report: ConstraintReport) -> None:
        """约束③：没有依据的结论，置信度不能高。

        ============================================================
        为什么用置信度而不是"对不对"
        ============================================================
        因为"对不对"需要 ground truth 才能判定，而这条约束检查的是
        **系统自己有没有自我校准** —— 这个不需要 ground truth。

        一个"高置信但错误"的系统比"低置信但错误"的系统危险得多：
        前者会让维护者放下戒备直接采纳，后者至少会让人多看一眼。

        所以这条约束守的是：**不允许在证据不足时表现出自信。**
        """
        report.checks_run += 1

        cites = getattr(result, "cites", None) or []
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)

        if not cites and confidence > self.UNSUPPORTED_CONFIDENCE_CAP:
            report.violations.append(Violation(
                constraint="置信度与证据匹配",
                detail=(f"无引用依据却给出高置信度 "
                        f"（confidence={confidence:.2f} > "
                        f"{self.UNSUPPORTED_CONFIDENCE_CAP}），"
                        f"属于「自信地胡说」的典型模式"),
                severity="high",
            ))

    # ------------------------------------------------------------ ④ 不得死循环

    def _check_tool_loop(self, trace: Trace,
                         report: ConstraintReport) -> None:
        """约束④：不得连续重复调用参数完全相同的工具。

        注意：这与 reliability.LoopDetector 是**两道独立的检查**。
        LoopDetector 在运行中实时阻止；这里在运行后做复盘。

        为什么要重复检查？因为：
          · LoopDetector 可能被配置得比较宽松（window=4, max_repeats=3）
          · 而且它一旦阻止过一次就会 reset，后面的重复不会被记录

        事后复盘能发现"虽然没有触发阻止，但确实在打转"的情况 ——
        这类"缓慢兜圈子"恰恰是实时检测最容易漏掉的。
        """
        if not trace or not trace.steps:
            return

        report.checks_run += 1

        signatures = []
        for step in trace.steps:
            if step.action is not None:
                signatures.append(step.action.norm_key())

        # 检查是否有连续 2 次以上完全相同的调用
        repeats = 0
        for i in range(1, len(signatures)):
            if signatures[i] == signatures[i - 1]:
                repeats += 1

        if repeats >= 2:
            report.violations.append(Violation(
                constraint="不得重复调用同一工具",
                detail=f"检测到 {repeats} 次连续相同参数的工具调用",
                severity="medium",
            ))

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _collect_tool_outputs(trace: Optional[Trace]) -> list:
        """把轨迹里所有工具调用的返回收集起来。

        注意这里只取 dict/list 类型的输出 —— 因为要从中读结构化的字段
        （比如 likely_duplicate）。字符串类型的输出（错误信息）跳过。
        """
        outputs = []
        if not trace:
            return outputs
        for step in trace.steps:
            if step.action is not None and step.action.ok:
                if isinstance(step.action.output, (dict, list)):
                    outputs.append(step.action.output)
        return outputs


# =============================================================================
# 批量汇总
# =============================================================================

class LongHorizonReport:
    """把多组、多轮数的结果聚合成对照表。

    ============================================================
    这个类存在的意义：把"我觉得"变成"数据显示"
    ============================================================
    一个单独的实验结果说明不了什么（可能是偶然）。
    只有把"轮数 × 是否启用状态卡"做成一张二维表，
    趋势才会显形：

        轮数        关闭状态卡     开启状态卡
        0 轮          100%          100%
        5 轮           95%          100%
        15 轮          80%          100%
        40 轮          60%          100%
                       ↑ 下降趋势      ↑ 稳定

    看到这张表，你才能说"状态卡重注入确实对抗了指令漂移"。

    **没有对照组的实验不是实验，是演示。**
    """

    @staticmethod
    def summarize(rows: list[dict], groups: list[str] | None = None) -> dict:
        """把原始行聚合成按 (轮数, 组别) 的汇总表。

        每一行形如：{"padding": 15, "group": "A · ...", "adherence": 1.0, ...}

        ★ 关于"组"的设计变更：
        第一版只支持两组（状态卡开/关），结果在真实模型上测不出差异 ——
        因为 system prompt 里还有一份约束，而 system 桶不可压缩。
        现在支持任意组数，用来做更严谨的对照：

            A. system有约束 + 状态卡     ← 完整设计（生产环境用这个）
            B. system有约束 + 无状态卡   ← 分离出状态卡的贡献
            C. system无约束 + 无状态卡   ← 真的没有保护（基线）

        有了 C 组才能回答："如果没有 system prompt 的兜底会怎样"。
        没有 C 组的实验，会把 system prompt 的功劳误记到状态卡头上。
        """
        from collections import defaultdict

        if groups is None:
            groups = []
            for r in rows:
                g = r.get("group", "未分组")
                if g not in groups:
                    groups.append(g)

        buckets: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            buckets[(r["padding"], r.get("group", "未分组"))].append(r["adherence"])

        # 转成二维表
        paddings = sorted({r["padding"] for r in rows})
        table = []
        for p in paddings:
            row = {"轮数": p}
            for g in groups:
                vals = buckets.get((p, g), [])
                row[g] = round(sum(vals) / len(vals), 4) if vals else None
                row[f"{g}_n"] = len(vals)
            table.append(row)

        # 趋势判断
        trend = {}
        for g in groups:
            pairs = [(p, buckets.get((p, g), [])) for p in paddings]
            pairs = [(p, sum(v) / len(v)) for p, v in pairs if v]
            if len(pairs) >= 2:
                first, last = pairs[0][1], pairs[-1][1]
                trend[g] = {
                    "首": round(first, 4),
                    "末": round(last, 4),
                    "变化": round(last - first, 4),
                    "结论": ("稳定" if abs(last - first) < 0.05 else
                             ("下降" if last < first else "上升")),
                }

        return {
            "表": table,
            "组": groups,
            "趋势": trend,
            "口径": {
                "指标": "约束遵守率（1 - 违规数/检查数）",
                "检查数说明": "不适用的检查不计入分母（如无历史相似 issue 时跳过判重检查）",
                "局限": ("约束判定是规则化的近似，只检查'有没有'，"
                         "不检查'好不好'；样本量偏小时趋势仅供方向参考"),
            },
        }
