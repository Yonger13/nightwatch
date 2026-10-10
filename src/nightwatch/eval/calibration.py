"""
Nightwatch · 置信度校准与转人工阈值

================================================================================
为什么"准确率 90%"还不够，还要看校准
================================================================================
假设两个系统都报"准确率 90%"：

    系统 A：每条都报 confidence = 0.9，其中 90% 是对的
    系统 B：一半样本报 0.99（全对），一半报 0.55（只对一半）

两者的**准确率相同**，但能做的事完全不同：

    · A 无法区分"哪条靠谱"，只能整批处理
    · B 可以**把 0.99 的那一半直接出、0.55 的那一半转人工** ——
      于是整体准确率能拉到接近 100%

**决定"能不能分流"的不是准确率，是校准质量。**
如果一个系统的 confidence 和实际正确率对不上（"过度自信"），
那么任何基于阈值的分流都会失效：你以为转的是低质量样本，实际不是。

================================================================================
三个指标各自回答什么问题
================================================================================
    ece(confidences, corrects)
        → "模型的自信**准不准**？" 一个数字描述整体校准质量
          附带分桶数据，能看出是"普遍过度自信"还是"只在某段区间失准"

    risk_coverage_curve(confidences, corrects)
        → "如果我只出最自信的 x%，错误率是多少？"
          完整刻画"质量 vs 覆盖率"的权衡，AURC 是它的单一数字概括

    suggest_escalation_threshold(confidences, corrects, min_accuracy)
        → "要保证直出部分准确率 ≥ 某值，阈值该定在哪？"
          给出可执行的切点

三者是递进关系：ECE 说"能不能用"，曲线说"用多少"，阈值说"切在哪"。

================================================================================
★★ 必须写在前面的边界（否则这些数字会被误用）★★
================================================================================
① **confidence 是模型自报的，不是概率。**
   大模型输出的"置信度"往往是它对自己答案的主观评估，不是校准过的概率。
   校准分析测的是"**自报值和实际正确率的一致性**"，
   而不是"它是不是真的概率" —— 后者需要模型内部 logits，本项目拿不到。

② **ECE 依赖分桶数，换桶必须重报。**
   桶越少，每桶的平均越"平"，ECE 会系统性偏小。
   所以 `n_bins` 是 ECE 的**隐含参数**，不是实现细节 ——
   它放在 `config.EVAL["calibration_bins"]`，报告里必须写明用了几个桶。

③ **切点是在同一批数据上挑出来的 → 有乐观偏差。**
   `suggest_escalation_threshold` 返回的 `expected_accuracy` 是
   **在这批样本上观测到的**准确率，不是"上线后的真实准确率"。
   在数据上挑出来的最优切点天然偏高（选择偏差），
   必须在**独立数据**上复验 —— 这和阈值扫描、超参搜索是同一类问题。

④ **样本量 < 50 时切点仅供参考。**
   蒙特卡洛误差和少数样本的偶然波动都会主导结论。
   本模块会在返回值里显式标记，而不是让你自己去记。

================================================================================
零第三方依赖
================================================================================
只用 `statistics`。真要做可靠性图、Brier 分解、温度缩放
（temperature scaling）生产上会用 `scikit-learn` / `netcal` ——
本项目保持零依赖，所以只实现"能指导决策"的那部分。
好处是不引入依赖，代价是没有做**事后校准**（把 confidence 映射成真概率）——
那需要一个可优化的映射函数，本模块不做，只在文档里点明它是下一步。
================================================================================
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    from nightwatch.config import EVAL
    from nightwatch.eval.metrics import Metrics
except ImportError:                     # pragma: no cover
    EVAL = {"calibration_bins": 10, "escalation_min_accuracy": 0.9,
            "calibration_min_samples": 50}

    class Metrics:                      # type: ignore[no-redef]
        @staticmethod
        def top1_accuracy(pred, gold):
            raise RuntimeError("需要 nightwatch.eval.metrics")


# =============================================================================
# 内部工具
# =============================================================================

def _validate(confidences: Sequence[float],
              corrects: Sequence) -> tuple[list[float], list[bool]]:
    """校验并规整输入。

    ★ 长度不等**直接抛异常** —— 两个序列必须一一对应，
      静默截断会让结果看起来正常但其实是错的。
    """
    conf = [float(c) for c in confidences]
    corr = [bool(c) for c in corrects]
    if len(conf) != len(corr):
        raise ValueError(
            f"置信度与正确性标注必须一一对应，实际 {len(conf)} vs {len(corr)}")
    if any(c < 0.0 or c > 1.0 for c in conf):
        bad = [c for c in conf if c < 0.0 or c > 1.0][:3]
        raise ValueError(f"置信度必须落在 [0, 1]，发现越界值：{bad}")
    return conf, corr


def _small_sample_note(n: int) -> str:
    """样本量不足时的提示（沿用项目一贯的"小样本"措辞风格）。"""
    threshold = int(EVAL.get("calibration_min_samples", 50))
    if n < threshold:
        return (f"样本量 {n} < {threshold}，校准曲线和切点仅作**方向参考**，"
                f"不宜据此定策略")
    return ""


# =============================================================================
# ① 期望校准误差（ECE）+ 可靠性分桶
# =============================================================================

def ece(confidences: Sequence[float], corrects: Sequence,
        n_bins: int | None = None) -> dict:
    """期望校准误差（Expected Calibration Error）+ 分桶可靠性数据。

    参数
    ----
    confidences : 每个样本的置信度（0~1）
    corrects    : 每个样本是否正确（布尔或 0/1）
    n_bins      : 分桶数，默认取 `config.EVAL["calibration_bins"]`。

    返回
    ----
    `{ece, mce, n, n_bins, bins: [...], ...}`

    ========================================================================
    怎么算的
    ========================================================================
        把 [0, 1] 等分成 n_bins 个桶，样本按置信度落桶
        每个桶里：平均置信度 conf_b，实际正确率 acc_b，样本数 n_b
        ECE = Σ (n_b / n) · |acc_b − conf_b|

    直觉：**它衡量的是"对角线偏离"**。
    如果模型说"我有 80% 把握"的那些样本里恰好 80% 是对的，
    那么这个桶对 ECE 的贡献是 0 —— 这就是"校准良好"。

    `mce`（Maximum Calibration Error）= max_b |acc_b − conf_b|，
    即"最差的那个桶偏了多少"。报告里两个都给是因为：
        ECE 小但 MCE 大 → 整体不错，但**某一段区间严重失准**，
                          而那一段可能正是阈值要切的地方。

    ========================================================================
    已知局限
    ========================================================================
      · **ECE 的数值依赖 n_bins**。桶少 → 每桶平均更平 → ECE 偏小。
        所以换桶数必须重报，且报告里要写明桶数。
      · 桶内样本很少时，acc_b 本身的波动就很大 —— 少数样本
        就能把某个桶的 ECE 贡献拉高。`bins` 里保留了每桶样本数
        供人工判断，也可以用 `min_bin_size` 过滤后再看。
      · 它只衡量"平均偏离"，**不区分过自信和欠自信**
        （两者都取绝对值）。要方向就看每个桶的 `gap` 符号。
    """
    n_bins = int(n_bins if n_bins is not None else
                 EVAL.get("calibration_bins", 10))
    if n_bins < 1:
        raise ValueError(f"分桶数必须 ≥ 1，实际 {n_bins}")

    conf, corr = _validate(confidences, corrects)
    n = len(conf)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "n": 0, "n_bins": n_bins, "bins": [],
                "note": "没有样本，无法计算校准误差"}

    # 桶边界：[0, 1/n_bins), [1/n_bins, 2/n_bins), ..., [(n-1)/n_bins, 1]
    # ★ 最后一个桶要**闭区间**，否则 confidence = 1.0 的样本无处可去。
    buckets: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(conf):
        idx = min(int(c * n_bins), n_bins - 1)
        buckets[idx].append(i)

    bins, ece_val, mce_val = [], 0.0, 0.0
    for b, idxs in enumerate(buckets):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if not idxs:
            bins.append({
                "bin": b, "range": [round(lo, 4), round(hi, 4)],
                "n": 0, "avg_confidence": None,
                "accuracy": None, "gap": None,
            })
            continue
        avg_conf = sum(conf[i] for i in idxs) / len(idxs)
        acc = sum(1 for i in idxs if corr[i]) / len(idxs)
        gap = acc - avg_conf                 # 正 = 欠自信；负 = 过度自信
        ece_val += (len(idxs) / n) * abs(gap)
        mce_val = max(mce_val, abs(gap))
        bins.append({
            "bin": b, "range": [round(lo, 4), round(hi, 4)],
            "n": len(idxs),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })

    # 给一句人话判断（把诊断逻辑写进指标报告里，读的人不必自己推断）
    if ece_val < 0.05:
        verdict = "校准良好（平均偏离 < 0.05）—— 置信度可以直接用于阈值分流"
    elif ece_val < 0.15:
        verdict = "校准一般 —— 阈值分流可用，但直出部分的实际准确率会低于名义值"
    else:
        verdict = ("校准较差 —— 置信度**不宜直接当概率用**；"
                   "若要做分流，需先做事后校准或改用更保守的阈值")

    return {
        "ece": round(ece_val, 4),
        "mce": round(mce_val, 4),
        "n": n,
        "n_bins": n_bins,
        "bins": bins,
        "verdict": verdict,
        "note": (f"ECE 依赖分桶数（本轮 {n_bins} 桶）—— 换桶数必须重报。"
                 + _small_sample_note(n)),
        "局限": [
            "confidence 是模型自报值，不是校准过的概率；本指标测的是"
            "「自报值与实际正确率的一致性」",
            "ECE 不区分过自信与欠自信（取绝对值），方向看每个桶的 gap 符号",
        ],
    }


# =============================================================================
# ② 风险-覆盖率曲线 + AURC
# =============================================================================

def risk_coverage_curve(confidences: Sequence[float],
                        corrects: Sequence) -> dict:
    """风险-覆盖率曲线：只出最自信的 x% 时，错误率是多少。

    参数
    ----
    confidences, corrects : 同 `ece`。

    返回
    ----
    `{aurc, curve: [{coverage, n_kept, risk, accuracy}], key_points, n, ...}`

    ★ `curve` 里的 `risk` 就是**错误率**（1 − accuracy）——
      之所以用"风险"这个词，是因为这条曲线的标准叫法就是
      risk-coverage curve，读文献时对得上。

    ========================================================================
    AURC（Area Under the Risk-Coverage curve）
    ========================================================================
        AURC = (1/n) · Σ_{k=1..n} risk(k)

    即"对所有可能的覆盖率取平均的错误率"。

    **怎么读它**：
      · AURC 越低越好
      · **置信度完全无信息时，AURC 的期望值就等于整体错误率**
        —— 所以 AURC 只有和"整体错误率"对比才有意义：
            AURC 明显低于整体错误率 → 置信度**确实有区分力**
            AURC ≈ 整体错误率      → 置信度**没有提供任何排序信息**
      本函数把 `overall_error` 一起返回，就是为了做这个对比 ——
      单看 AURC 的绝对值没有参照物。

      ⚠️ 是「期望值」而不是恰好相等：无信息时 AURC 是个随机变量
         （取决于碰巧的排序），其均值等于整体错误率，
         单次取值会在它附近浮动。

    ========================================================================
    已知局限
    ========================================================================
      · 它衡量的是**排序质量**（能不能把对的排在前面），
        不是"置信度准不准" —— 那是 ECE 的事。
        一个**单调但整体偏大**的置信度（全部说 0.99）排序完全正确，
        但 ECE 很差。**两者必须一起看。**
      · 曲线在每个 k 上都只有一个点，样本少时抖动很大；
        看趋势，别抠某一个点。
    """
    conf, corr = _validate(confidences, corrects)
    n = len(conf)
    if n == 0:
        return {"aurc": 0.0, "curve": [], "key_points": {}, "n": 0,
                "overall_error": 0.0, "note": "没有样本，无法计算"}

    order = sorted(range(n), key=lambda i: -conf[i])     # 置信度降序
    curve: list[dict] = []
    n_correct = 0
    for k, i in enumerate(order, start=1):
        n_correct += int(corr[i])
        acc = n_correct / k
        curve.append({
            "coverage": round(k / n, 4),
            "n_kept": k,
            "risk": round(1.0 - acc, 4),
            "accuracy": round(acc, 4),
        })

    aurc = sum(p["risk"] for p in curve) / n
    overall_error = 1.0 - (sum(1 for c in corr if c) / n)

    # 关键覆盖率点 —— 报告里好用，不必让人去数组里找
    key_points = {}
    for target in (0.5, 0.8, 0.9, 1.0):
        want = max(1, int(round(target * n)))
        key_points[f"{target:.0%}"] = curve[want - 1]

    # ★ 判据要和"整体错误率"比 —— 光看 AURC 绝对值没有参照物
    #
    # ⚠️ 但有一个**退化情况**必须先处理：整体错误率为 0 时，
    #    AURC 必然也是 0，此时"AURC ≈ 整体错误率"会得出
    #    "置信度没有排序能力"的结论 —— 而这是**误导**：
    #    根本没有错误可以分流，谈不上排序能力。
    #    （这个坑是在 Mock 数据上跑出来的：整体准确率 100%，脚本却报
    #      "基于它做任何阈值分流都不会有效"。）
    if overall_error == 0.0:
        verdict = ("全部样本都正确，**没有错误可供分流** —— "
                   "此时 AURC 和排序能力都无从谈起。"
                   "要评估置信度的区分力，需要在有错误的样本上做。")
    elif aurc < overall_error * 0.8:
        verdict = ("置信度有明确区分力：只出最自信的部分可显著降低错误率"
                   f"（AURC {aurc:.3f} 明显低于整体错误率 {overall_error:.3f}）")
    elif aurc < overall_error * 0.95:
        verdict = (f"置信度区分力有限（AURC {aurc:.3f} vs 整体 "
                   f"{overall_error:.3f}）—— 分流收益不大")
    else:
        verdict = (f"**置信度几乎没有排序能力**（AURC {aurc:.3f} ≈ 整体错误率 "
                   f"{overall_error:.3f}）—— 基于它做任何阈值分流都不会有效。"
                   f"此时该修的是置信度的来源，而不是调阈值。")

    return {
        "aurc": round(aurc, 4),
        "overall_error": round(overall_error, 4),
        "curve": curve,
        "key_points": key_points,
        "n": n,
        "verdict": verdict,
        "note": ("AURC 要和 overall_error 对比着读 —— "
                 "置信度无信息时 AURC 的期望值就等于整体错误率。"
                 + _small_sample_note(n)),
        "局限": [
            "衡量的是排序质量（能否把对的排前面），不是置信度准不准 —— "
            "后者看 ece()。一个单调但整体偏大的置信度排序正确、ECE 很差",
            "每个覆盖率上只有一个点，样本少时抖动大，看趋势不要抠单点",
        ],
    }


# =============================================================================
# ③ 转人工阈值
# =============================================================================

def suggest_escalation_threshold(confidences: Sequence[float],
                                 corrects: Sequence,
                                 min_accuracy: float | None = None) -> dict:
    """找一个置信度切点：切点以上直接出，以下转人工。

    参数
    ----
    confidences, corrects : 同 `ece`。
    min_accuracy          : 直出部分要求的准确率下限。
                            默认取 `config.EVAL["escalation_min_accuracy"]`。

    返回
    ----
    `{threshold, coverage, expected_accuracy, n_kept, found, ...}`

    ========================================================================
    怎么挑的：**在满足准确率约束的前提下，覆盖率最大化**
    ========================================================================
    按置信度降序累加，找**最大的 k** 使得"前 k 条的准确率 ≥ min_accuracy"。

    这个目标函数的含义是：
        "只要直出部分的质量达标，就尽量少转人工（转人工是有成本的）"

    ★ 为什么不用"准确率最大化"：那样会把 k 压到极小（甚至 1 条），
      准确率 100% 但覆盖率 1%，对业务毫无意义。
      **阈值问题永远是"质量 × 覆盖"的权衡**，不能只优化一头。

    ========================================================================
    ★★ 最需要警惕的一处：`expected_accuracy` 有乐观偏差 ★★
    ========================================================================
    切点是在**这批数据上**挑出来的，然后在**同一批数据上**评估 ——
    挑出来的切点天然偏向"表现好的那一段"。

    所以 `expected_accuracy` 是**观测值**，不是上线后的真实准确率。
    真实的准确率会**更低**，低多少取决于：
      · 样本量（越少越严重）
      · 挑了多少个候选切点（等价于多重比较）

    **必须做的**：把切点拿到**独立数据**上复验。
    这和"阈值扫描里挑 F1 最优"、"超参搜索里挑最优配置"是同一类问题 ——
    在数据上挑最优 = 选择偏差，见 `doe.bonferroni_note`。

    ========================================================================
    已知局限
    ========================================================================
      · 累加准确率**不保证单调**：可能 k=10 时准确率 0.91、k=11 时掉到 0.88、
        k=12 又回到 0.90。本函数取"最大的满足约束的 k"，
        所以给出的切点可能**跳过**一个刚好不达标的中间点 ——
        这是刻意的（要最大覆盖），但报告的读者应当知道这一点。
      · 完全没有考虑"转人工的代价"和"漏检的代价"哪个大。
        真实系统里这两者不对称，阈值应据此调整 —— 本函数只给
        "满足质量约束的最大覆盖"，不替你决定代价权重。
    """
    min_accuracy = float(min_accuracy if min_accuracy is not None
                         else EVAL.get("escalation_min_accuracy", 0.9))
    if not 0.0 < min_accuracy <= 1.0:
        raise ValueError(f"min_accuracy 必须在 (0, 1]，实际 {min_accuracy}")

    conf, corr = _validate(confidences, corrects)
    n = len(conf)
    if n == 0:
        return {"threshold": None, "coverage": 0.0, "expected_accuracy": 0.0,
                "n_kept": 0, "found": False, "n": 0,
                "note": "没有样本，无法给出切点"}

    order = sorted(range(n), key=lambda i: -conf[i])
    best_k = 0
    n_correct = 0
    for k, i in enumerate(order, start=1):
        n_correct += int(corr[i])
        if n_correct / k >= min_accuracy:
            best_k = k                          # 持续记录**最大**的可行 k

    if best_k == 0:
        return {
            "threshold": None, "coverage": 0.0, "expected_accuracy": 0.0,
            "n_kept": 0, "found": False, "n": n,
            "min_accuracy_target": min_accuracy,
            "verdict": (f"**找不到满足约束的切点** —— 即使只保留最自信的 1 条，"
                        f"也达不到 {min_accuracy:.0%}。"
                        f"说明需要先改善模型或置信度质量，调阈值救不了。"),
            "note": ("找不到切点本身就是一个有用的结论："
                     "它说明问题不在阈值，而在上游。"),
        }

    kept = order[:best_k]
    acc = sum(1 for i in kept if corr[i]) / best_k
    # 阈值 = 被保留样本里**最低**的那个置信度（这是能切出该覆盖率的边界）
    threshold = min(conf[i] for i in kept)

    return {
        "threshold": round(threshold, 4),
        "coverage": round(best_k / n, 4),
        "expected_accuracy": round(acc, 4),
        "n_kept": best_k,
        "n_escalated": n - best_k,
        "found": True,
        "n": n,
        "min_accuracy_target": min_accuracy,
        "verdict": (f"置信度 ≥ {threshold:.4f} 时直出（保留 {best_k}/{n} = "
                    f"{best_k / n:.0%}），这批样本上准确率 {acc:.1%}"
                    f"（目标 ≥ {min_accuracy:.0%}）"),
        "note": ("★ expected_accuracy 是在**挑切点的同一批数据**上观测到的，"
                 "有乐观偏差 —— 上线后的真实准确率会更低。"
                 "必须在独立数据上复验，不可直接当作线上指标。"
                 + _small_sample_note(n)),
        "局限": [
            "累加准确率不保证单调；本函数取「最大的可行 k」以获得最大覆盖，"
            "因此可能跳过中间某个刚好不达标的位置",
            "未考虑「转人工成本」与「漏检代价」的不对称性 —— "
            "真实系统应据此调整，本函数只给满足质量约束的最大覆盖",
        ],
    }


# =============================================================================
# 与 SampleResult 的桥接（数据来源）
# =============================================================================

def extract_confidence_pairs(results: Sequence[Any]) -> tuple[list[float], list[bool]]:
    """从 `SampleResult` 列表里取出 `(置信度, 是否正确)` 两组。

    ★ 正确性判定**复用 `Metrics.top1_accuracy`**，不重写 ——
      否则"什么算对"会长出两份定义，而两份定义迟早会不一致
      （本项目已经在 `parse_action` / `_format_memory` / 探针集 /
      余量口径上踩过四次）。

    ⚠️ 注意 `SampleResult.confidence` **默认是 0.0** ——
       历史数据（这个字段被采集但从未被消费之前跑出来的）
       会全是 0，此时校准分析**没有意义**。
       本函数会把这种情况显式报出来，而不是算出一堆看似正常的数字。
    """
    confidences: list[float] = []
    corrects: list[bool] = []
    for r in results:
        confidences.append(float(getattr(r, "confidence", 0.0) or 0.0))
        corrects.append(
            Metrics.top1_accuracy(list(getattr(r, "predicted_labels", []) or []),
                                  list(getattr(r, "gold_labels", []) or [])) >= 1.0)
    return confidences, corrects


def calibration_summary(results: Sequence[Any]) -> dict:
    """一次跑完 ECE + 风险-覆盖率曲线 + 切点（报告用的汇总入口）。

    输入是 `SampleResult` 列表。如果一个样本都没采集到置信度，
    会**明确返回"不可用"**，而不是给出三个 0 让你误以为"校准完美"。
    """
    if not results:
        return {"可用": False, "原因": "没有样本"}

    conf, corr = extract_confidence_pairs(results)
    n = len(conf)
    n_nonzero = sum(1 for c in conf if c > 0.0)

    if n_nonzero == 0:
        return {
            "可用": False,
            "原因": ("全部样本的 confidence 都是 0 —— "
                     "说明这批数据不是带着置信度采集的"
                     "（该字段曾长期被采集但未被消费），"
                     "校准分析没有意义。"),
            "n": n,
            "修复方向": "重新跑一次评测，确保 SampleResult.confidence 被真实填写",
        }

    return {
        "可用": True,
        "n": n,
        "有置信度的样本": n_nonzero,
        "ece": ece(conf, corr),
        "risk_coverage": risk_coverage_curve(conf, corr),
        "escalation": suggest_escalation_threshold(conf, corr),
        "说明": ("三个指标是递进关系：ECE 说「能不能用」，"
                 "曲线说「能用多少」，切点说「切在哪」。"),
    }
