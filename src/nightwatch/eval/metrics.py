"""
Nightwatch · 评测指标

================================================================================
为什么评测是这个项目最重要的模块（而不是收尾工作）
================================================================================
两组数字值得记住：

    Gartner：到 2027 年底，超过 40% 的 Agentic AI 项目会被取消或降级
    MIT Media Lab：95% 的企业 AI 试点未产生任何可衡量的财务回报（300 个样本）

结论很直接：**"做出来"的人很多，"能度量"的人极少。**

所以调研里 5/7 份 JD 要求评测能力、被定位为"真正的分水岭"，而它同时是
"培训班不教、也最难自己练"的东西 —— 这就是你的稀缺性所在。

================================================================================
三条纪律（写在代码前面，因为它们决定了指标可信不可信）
================================================================================
    ① 不报单次最好成绩。大模型有随机性，每个样本跑 N 次报中位数和分布。
    ② 不只报赢的那组对比。没提升的也要写出来并解释原因。
    ③ 所有指标都注明样本量和评测集版本。

**面试官真正看的是"这个人知不知道自己数字的边界在哪"**，
不是数字好不好看。一个说"这个指标我只测了 30 条，置信区间很宽"的候选人，
比一个报出漂亮数字却答不出实验设置的候选人可信度高得多。

================================================================================
关键设计：拆成可判定的子项
================================================================================
不评"回答得好不好"（主观、不稳定），而是拆成：
    是否给出标签 / 是否说明理由 / 是否给出引用依据

这个做法直接回应了"LLM-as-Judge 的偏差怎么处理"这道题 ——
第一层答案就是：**把主观评分改成可判定的子项，子项比整体判断稳定得多。**
================================================================================
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# 单样本结果
# =============================================================================

@dataclass
class SampleResult:
    """一个评测样本的结果。"""
    issue_id: str
    stratum: str = ""                         # 所属分层（按问题类型）
    run: int = 0

    # --- 预测 ---
    predicted_labels: list[str] = field(default_factory=list)
    predicted_duplicate_of: Optional[str] = None
    confidence: float = 0.0
    outcome: str = ""

    # --- ground truth（维护者的真实动作）---
    gold_labels: list[str] = field(default_factory=list)
    gold_duplicate_of: Optional[str] = None

    # --- 过程指标 ---
    n_steps: int = 0
    elapsed_ms: int = 0
    total_tokens: int = 0
    memory_recalled: int = 0
    memory_used: int = 0

    # --- 子项判定（由 judge 填充）---
    sub_scores: dict = field(default_factory=dict)


# =============================================================================
# 指标计算
# =============================================================================

class Metrics:
    """指标计算。

    **注意每个指标都明确写了"怎么测才公平"** —— 这比数字本身重要。
    """

    # ============================================================
    # 标签归一化 —— 评测里最容易踩的坑，没有之一
    # ============================================================
    # 真实项目里，维护者的标签格式和你的预测格式经常不一致：
    #     ground truth:  "timeout"  /  "bug:timeout"  /  "kind/bug-timeout"
    #     你的预测:       "bug/timeout"
    # 如果不做归一化，F1 会**永远是 0** —— 而这根本不是模型的问题，
    # 是评测代码的问题。
    #
    # 教训：**当所有指标都接近 0 或 1 时，先怀疑评测代码，再怀疑模型。**
    # 完美的 0 和完美的 1 都是可疑信号。
    #
    # 这条经验值得写进任何 Agent 项目的排查清单。

    # 常见的"命名空间前缀"，归一化时剥掉
    _PREFIXES = ("bug", "kind", "type", "category", "issue", "label",
                 "status", "priority", "area", "comp")

    @classmethod
    def normalize_label(cls, label: str) -> str:
        """把标签归一化成可比较的核心词。

        步骤：
          1. 转小写、去空白
          2. 按常见分隔符切分，取最后一段（去掉命名空间）
          3. 如果最后一段是通用前缀词（如 "bug"），取前一段
        """
        s = str(label).strip().lower()
        # 统一分隔符
        for sep in ("/", ":", "-", "_", " "):
            s = s.replace(sep, "|")
        parts = [p for p in s.split("|") if p]
        if not parts:
            return s
        # 从后往前找第一个不是通用前缀的词
        for p in reversed(parts):
            if p not in cls._PREFIXES:
                return p
        return parts[-1]

    @classmethod
    def label_f1(cls, pred: list[str], gold: list[str]) -> tuple[float, float, float]:
        """标签预测的 精确率 / 召回率 / F1。

        为什么用 F1 而不是准确率？
        因为标签分布通常是**长尾**的：常见问题占大多数，稀有类型很少。
        这种数据下"全猜最常见的标签"能拿到很高的准确率，但毫无价值。
        F1 会惩罚这种偷懒策略。

        **比较前先做归一化** —— 见上面的说明。
        """
        p = {cls.normalize_label(x) for x in pred if str(x).strip()}
        g = {cls.normalize_label(x) for x in gold if str(x).strip()}
        if not p and not g:
            return 1.0, 1.0, 1.0
        if not p or not g:
            return 0.0, 0.0, 0.0
        tp = len(p & g)
        precision = tp / len(p)
        recall = tp / len(g)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    # ============================================================
    # 多口径指标：严格 F1 会系统性惩罚"多打标签"的模型
    # ============================================================
    # ★ 添加这三个指标的原因（一次真实的口径争议）：
    #
    # 实测发现 qwen3.8-flash 平均打 2.4 个标签、qwen-max 打 1.4 个，
    # 于是 flash 的严格 F1 更低（0.494 vs 0.734）。
    # 但**这个差距里有多少是"模型差"、有多少是"口径偏严"？** 分不清。
    #
    # 因为严格多标签 F1 隐含一个假设：**漏报和误报的代价相同**。
    # 而 issue 分流的真实代价是**不对称**的：
    #     漏报（漏了正确标签）→ 问题可能被忽略
    #     误报（多打了标签）  → 维护者看到无关标签，影响小得多
    #
    # 所以单一口径会误导选型决策。**必须同时报告多种口径**，
    # 让"口径敏感性"本身暴露出来 ——
    # 如果换个口径结论就反转，那说明这个差异不是模型能力差异，是口径造成的。
    #
    # 这也是评测的一条通用原则：
    # **当你用单一指标下结论时，先问"换个合理口径结论还成立吗"。**

    @classmethod
    def top1_accuracy(cls, pred: list[str], gold: list[str]) -> float:
        """Top-1 准确率：**预测的第一个标签**是否命中 gold。

        这是最宽松的"抓核心分类"口径 ——
        适合"我只关心最主流的那个分类对不对"的场景。

        注意它不惩罚多打：打 5 个标签，只要第一个对了就算对。
        所以它和严格 F1 是**互补**的，不是替代关系。
        """
        p = [cls.normalize_label(x) for x in pred if str(x).strip()]
        g = {cls.normalize_label(x) for x in gold if str(x).strip()}
        if not p and not g:
            return 1.0
        if not p or not g:
            return 0.0
        return 1.0 if p[0] in g else 0.0

    @classmethod
    def hit_at_k(cls, pred: list[str], gold: list[str], k: int = 2) -> float:
        """Hit@k：**gold 的标签有多少比例**落在预测的前 k 个里。

        这是**召回导向**的口径（只关心"有没有漏"，不惩罚"多打了"）。
        适合"宁可多给几个候选，也不能漏"的场景 ——
        比如交给人工复核的分流建议，多给一个候选只是让人多看一眼。

        返回的是比例（0~1），不是布尔值 ——
        因为一条 issue 可能有 2 个 gold 标签，命中 1 个算 0.5。
        """
        p = {cls.normalize_label(x) for x in pred[:k] if str(x).strip()}
        g = {cls.normalize_label(x) for x in gold if str(x).strip()}
        if not g:
            return 1.0 if not p else 0.0     # gold 为空时，不打标签才算对
        if not p:
            return 0.0
        return len(p & g) / len(g)

    @classmethod
    def multi_metric_report(cls, results: list) -> dict:
        """一次给出**多口径**指标 + 过度标注量化。

        ============================================================
        为什么要把多个口径并排输出
        ============================================================
        因为单个数字会诱导错误结论。举个真实例子：

            严格 F1     flash 0.494  <  max 0.734   → 看起来 flash 差很多
            Top-1 准确率  flash ?     vs  max ?      → 可能差不多

        如果 Top-1 差距很小、只有严格 F1 差距大，
        那说明差异**主要来自"多打标签"这一种行为**，
        而不是"抓不住核心分类"。

        **这两个结论导向完全不同的改进方向**：
          · 如果是抓不住核心 → 换模型
          · 如果只是多打标签   → 调提示词就够（成本低得多）

        所以并排报告不是"多给几个数字"，而是**为了定位问题性质**。
        """
        if not results:
            return {}

        def median(vals):
            import statistics
            return round(statistics.median(vals), 4) if vals else 0.0

        strict_f1, prec, rec = [], [], []
        top1, hit2, hit3 = [], [], []
        n_pred, n_gold = [], []

        for r in results:
            p_, rc, f_ = cls.label_f1(r.predicted_labels, r.gold_labels)
            prec.append(p_); rec.append(rc); strict_f1.append(f_)
            top1.append(cls.top1_accuracy(r.predicted_labels, r.gold_labels))
            hit2.append(cls.hit_at_k(r.predicted_labels, r.gold_labels, k=2))
            hit3.append(cls.hit_at_k(r.predicted_labels, r.gold_labels, k=3))
            n_pred.append(len(r.predicted_labels or []))
            n_gold.append(len(r.gold_labels or []))

        return {
            "样本量": len(results),
            "严格口径": {
                "precision": median(prec),
                "recall": median(rec),
                "F1": median(strict_f1),
                "说明": "多标签严格匹配；**假设漏报与误报代价相同**",
            },
            "宽松口径": {
                "Top-1准确率": median(top1),
                "Hit@2": median(hit2),
                "Hit@3": median(hit3),
                "说明": "Top-1 只看最核心分类；Hit@k 只看有没有漏（召回导向）",
            },
            "标注倾向": {
                "平均预测标签数": round(sum(n_pred) / len(n_pred), 2),
                "平均gold标签数": round(sum(n_gold) / len(n_gold), 2),
                "过度标注指数": round(
                    (sum(n_pred) / len(n_pred)) - (sum(n_gold) / len(n_gold)), 2),
                "说明": "过度标注指数 > 0 表示模型倾向多打标签",
            },
            "口径敏感性": cls._sensitivity_note(median(strict_f1),
                                                median(top1)),
        }

    @staticmethod
    def _sensitivity_note(strict_f1: float, top1: float) -> str:
        """根据两个口径的差距，提示"问题性质"。

        这是把"诊断逻辑"写进指标报告里 —— 让读报告的人不用自己推断。
        """
        gap = top1 - strict_f1
        if gap > 0.25:
            return ("宽松口径明显高于严格口径 → 差异**主要来自「多打标签」**，"
                    "而不是抓不住核心分类。**优先调提示词**（成本低），"
                    "换模型未必是必要的。")
        if gap > 0.10:
            return ("两个口径有差距但不大 → 既有过度标注、也有核心分类错误。"
                    "建议同时调提示词并考虑模型档位。")
        return ("两个口径接近 → 模型既没有明显过度标注、核心分类也稳。"
                "此时严格 F1 的差距更能反映真实能力，**换模型是有效手段**。")

    @staticmethod
    def duplicate_detection(results: list[SampleResult]) -> dict:
        """duplicate 检出情况。

        分四象限统计，而不是只算一个准确率 —— 因为两类错误的代价不对称：
          漏报（真的重复但没识别）→ 维护者要重复处理，浪费时间
          误报（不重复却说是重复）→ **更糟**，会把新问题误判为已解决而关闭

        所以这个指标必须分开报，而且要说明你更在意哪一个。
        这就是"业务理解"在指标设计上的体现。
        """
        tp = fp = fn = tn = 0
        for r in results:
            gold_dup = r.gold_duplicate_of is not None
            pred_dup = r.predicted_duplicate_of is not None
            if gold_dup and pred_dup:
                tp += 1
            elif gold_dup and not pred_dup:
                fn += 1
            elif not gold_dup and pred_dup:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round((2 * precision * recall / (precision + recall))
                        if (precision + recall) else 0.0, 4),
            "样本量": len(results),
            # 主动说明局限，而不是等面试官问
            "confidence_interval_note": (
                f"样本量 {len(results)}，F1 的 95% 置信区间约为 ±"
                f"{Metrics._wilson_margin(len(results)):.2f}；"
                f"样本量偏小时该指标不宜下强结论"
            ),
        }

    @staticmethod
    def _wilson_margin(n: int) -> float:
        """粗略估计比例的置信区间半宽（正态近似）。"""
        if n <= 0:
            return 1.0
        return 1.96 * 0.5 / (n ** 0.5)

    @staticmethod
    def memorization_effect(fresh: list[SampleResult],
                            warm: list[SampleResult]) -> dict:
        """记忆效果对比 —— **这是本项目最有区分度的指标**。

        设计思路（对应"长程一致性"和"记忆有没有用"两个问题）：
          把同一批 issue 分两组处理：
            fresh 组：记忆库为空（或屏蔽召回）
            warm  组：记忆库有历史
          对比两组的 步数 / token / 准确率

        为什么这个设计好？因为它**直接回答了"记忆到底有没有用"**，
        而不是笼统地说"我实现了记忆系统"。

        注意：报告必须包含"没有提升"的情况。如果 warm 组没有明显更好，
        要如实说明并解释（比如：可能是记忆召回精度不够，
        或者是这批 issue 本身的重复性不高）。
        """
        def avg(rs, key):
            vals = [getattr(r, key) for r in rs if getattr(r, key)]
            return statistics.median(vals) if vals else 0

        f_steps, w_steps = avg(fresh, "n_steps"), avg(warm, "n_steps")
        f_tok, w_tok = avg(fresh, "total_tokens"), avg(warm, "total_tokens")
        f_f1 = statistics.median([Metrics.label_f1(r.predicted_labels, r.gold_labels)[2]
                                  for r in fresh]) if fresh else 0.0
        w_f1 = statistics.median([Metrics.label_f1(r.predicted_labels, r.gold_labels)[2]
                                  for r in warm]) if warm else 0.0

        return {
            "fresh_组": {"中位步数": f_steps, "中位token": f_tok, "中位F1": round(f_f1, 4)},
            "warm_组":  {"中位步数": w_steps, "中位token": w_tok, "中位F1": round(w_f1, 4)},
            "步数变化": Metrics._delta(f_steps, w_steps),
            "token变化": Metrics._delta(f_tok, w_tok),
            "F1变化": Metrics._delta(f_f1, w_f1),
            "口径说明": (
                "报告的是中位数而非平均值（受极值影响小）。"
                "每组样本量相同、同一批 issue、同一模型配置；"
                "唯一变量是记忆召回是否开启。"
            ),
        }

    @staticmethod
    def _delta(before: float, after: float) -> dict:
        if not before:
            return {"绝对值": 0, "百分比": "N/A"}
        abs_d = round(after - before, 4)
        return {"绝对值": abs_d, "百分比": f"{abs_d / before * 100:+.1f}%"}

    @staticmethod
    def reliability_report(results: list[SampleResult]) -> dict:
        """可靠性指标 —— 对应四层机制的效果。

        "把 Agent 失控导致的无效 token 消耗降低 X%"这句话要成立，
        就必须能测出这类消耗。
        """
        unfinished = [r for r in results if r.outcome == "unfinished"]
        escalated = [r for r in results if r.outcome == "escalated"]
        misdiagnosed = [r for r in results if r.outcome == "misdiagnosed"]
        resolved = [r for r in results if r.outcome == "resolved"]
        total = len(results) or 1

        return {
            "自助解决率": round(len(resolved) / total, 4),
            "转人工率": round((len(escalated) + len(unfinished)) / total, 4),
            "未收敛率": round(len(unfinished) / total, 4),
            "误诊率": round(len(misdiagnosed) / total, 4),
            "中位步数": statistics.median([r.n_steps for r in results]) if results else 0,
            "P95步数": Metrics._percentile([r.n_steps for r in results], 95),
            "P95时延ms": Metrics._percentile([r.elapsed_ms for r in results], 95),
            "样本量": len(results),
            # 这批指标的难点是"误诊"的判定 —— 它需要 ground truth。
            # 而开源场景恰好有：维护者的实际动作就是 ground truth。
            "误诊判定依据": "预测标签与维护者实际打的标签无交集",
        }

    @staticmethod
    def memory_quality(results: list[SampleResult]) -> dict:
        """记忆系统的质量指标。

        **注意这里引入了"无效召回率"** —— 这是最容易被忽略的指标。
        只看"召回了几条"会误导你：有些记忆被频繁召回只是因为它的向量
        吸引了很多不相关查询（噪音吸铁石）。真正要优化的是
        "召回来之后真的被用上"的比例。
        """
        recalled = sum(r.memory_recalled for r in results)
        used = sum(r.memory_used for r in results)
        return {
            "总召回次数": recalled,
            "实际使用次数": used,
            "记忆命中率": round(used / recalled, 4) if recalled else 0.0,
            "无效召回率": round(1 - used / recalled, 4) if recalled else 0.0,
            "平均召回条数": round(recalled / len(results), 2) if results else 0.0,
            "说明": ("记忆命中率 = 召回后被实际使用的比例。"
                     "这个指标低说明召回精度不够（阈值太松或向量质量差），"
                     "而不是说明记忆没用。"),
        }

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


# =============================================================================
# 分层报告
# =============================================================================

class StratifiedReport:
    """按问题类型分层的报告。

    ============================================================
    为什么必须分层
    ============================================================
    如果只报一个总分，你会被常见类型的高表现掩盖住稀有类型的问题。

    举例：假设 70% 的 issue 是 timeout 类且模型表现很好（F1=0.9），
    30% 是配置类且模型表现很差（F1=0.2）。
    总分会是 0.69，看起来"还不错" —— 但实际上有一整类问题它处理不了。

    **分层报告才是诚实的报告。** 而"要不要分层"这个决定，
    本身就体现了你有没有真的做过评测。
    """

    @staticmethod
    def build(results: list[SampleResult]) -> dict:
        by_stratum: dict[str, list[SampleResult]] = {}
        for r in results:
            by_stratum.setdefault(r.stratum or "unknown", []).append(r)

        out = {}
        for name, rs in sorted(by_stratum.items()):
            f1s = [Metrics.label_f1(r.predicted_labels, r.gold_labels)[2] for r in rs]
            out[name] = {
                "样本量": len(rs),
                "中位F1": round(statistics.median(f1s), 4) if f1s else 0.0,
                "最低F1": round(min(f1s), 4) if f1s else 0.0,
                "中位步数": statistics.median([r.n_steps for r in rs]) if rs else 0,
                # 标注样本不足的分层 —— 提醒读者不要过度解读
                "提示": ("样本量偏小，指标仅供方向参考" if len(rs) < 5 else ""),
            }
        return out


# =============================================================================
# LLM-as-Judge 的偏差处理
# =============================================================================

class JudgeCalibrator:
    """裁判校准。

    这道题必然被问："用 LLM 做裁判，裁判本身的偏差怎么处理？"

    四件事（按重要性排序）：
      ① **校准** —— 先人工标一批，测裁判与人工的一致率。一致率不够就别用。
      ② **拆细粒度** —— 不评"好不好"，拆成可判定的子项。
      ③ **位置偏差** —— 做 A/B 对比时交换顺序跑两次。
      ④ **规则兜底** —— 能用确定性规则判的绝不用模型。

    这个类实现的是 ① 和 ②。
    """

    def __init__(self, min_agreement: float = 0.7):
        self.min_agreement = min_agreement

    def agreement(self, judge_scores: list[float],
                  human_scores: list[float],
                  threshold: float = 0.5) -> dict:
        """计算裁判与人工的一致率。

        **一致率低于阈值时，这个裁判就不可用** ——
        不是"凑合用"，是"不能用"。因为一个不可靠的裁判比没有裁判更糟：
        它会给你一个错误的信号，让你优化错方向。
        """
        if len(judge_scores) != len(human_scores) or not judge_scores:
            return {"可用": False, "原因": "样本不足或长度不匹配"}

        agree = sum(1 for j, h in zip(judge_scores, human_scores)
                    if (j >= threshold) == (h >= threshold))
        rate = agree / len(judge_scores)
        return {
            "一致率": round(rate, 4),
            "样本量": len(judge_scores),
            "可用": rate >= self.min_agreement,
            "结论": ("裁判可用" if rate >= self.min_agreement else
                     f"一致率 {rate:.2f} 低于阈值 {self.min_agreement}，"
                     f"该裁判不可用，需要改评分标准或增加人工样本"),
        }

    @staticmethod
    def subitems_from_json(judge_output: str) -> dict:
        """从裁判输出里提取子项分数。

        子项设计的三个原则：
          - 可判定（不是"好不好"，是"有没有"）
          - 正交（不重复计量同一个东西）
          - 少（3-4 个，多了会互相干扰）
        """
        try:
            d = json.loads(judge_output)
        except json.JSONDecodeError:
            start, end = judge_output.find("{"), judge_output.rfind("}")
            if start < 0:
                return {}
            try:
                d = json.loads(judge_output[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {k: v for k, v in d.items()
                if k in ("has_label", "has_reason", "has_citation", "score")}
