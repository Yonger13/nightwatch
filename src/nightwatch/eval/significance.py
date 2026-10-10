"""
Nightwatch · 配对显著性检验

================================================================================
为什么"提升了多少"必须配一个 p 值
================================================================================
评测报告里最常见的句式是"修复后命中率从 78% 提升到 92%"。
但这个句式**缺了一半信息**：如果样本只有 61 条，
78% → 92% 也就是 48 条 → 56 条，**8 条样本的翻转**。

在 61 条样本上，8 条翻转到底算不算"真的变好了"？
不检验的话，这个问题只能靠感觉回答。而感觉在两个方向上都会出错：

    · 把噪声当成果 → 下次改动照着这个方向优化，其实什么都没改
    · 把成果当噪声 → 一个真正有效的改动被否掉

==============================================================================
★★ 这里有个反复出现的坑：**配对设计和独立设计要用不同的检验** ★★
==============================================================================
同一个实验，如果两组样本是**同一批样本**（改动前 / 改动后），
那就是**配对设计**，必须用配对检验。

如果误用成独立样本检验（比如两独立样本 t 检验），会：
    · 丢掉"每个样本自身的基线"这一层信息 → **统计功效显著降低**
    · 在这类数据上，往往把真实存在的差异判成"不显著"

反过来说，如果两组本来就是不同样本，却硬套配对检验，
那就是**凭空制造相关性**，p 值会虚假地小。

判断方法只有一句话：**两次测量是不是发生在同一个对象上。**
本项目的所有前后对比（v1/v2、开/关记忆、不同阈值）
都是同一批 issue → **一律配对**。

==============================================================================
三个函数的分工（选错检验比不做检验更糟）
==============================================================================
    ┌────────────────────┬──────────────────────┬────────────────────┐
    │ 指标类型           │ 用什么               │ 为什么             │
    ├────────────────────┼──────────────────────┼────────────────────┤
    │ 连续（F1、步数、   │ paired_bootstrap     │ 不假设分布形状；   │
    │ token、时延）      │ 或                   │ 差值分布常常很偏， │
    │                    │ wilcoxon_signed_rank │ 均值的正态假设不牢 │
    ├────────────────────┼──────────────────────┼────────────────────┤
    │ 二元（命中/未命中）│ mcnemar              │ 二元数据没有"大小" │
    │                    │                      │ 只有"翻没翻"，     │
    │                    │                      │ 只需看不一致的那部分│
    └────────────────────┴──────────────────────┴────────────────────┘

**为什么二元指标不能用 bootstrap 或 Wilcoxon？**
因为二元值的差值只有三种可能（-1 / 0 / +1），
分布极其离散，秩检验会大量并列、bootstrap 的均值也不满足正态性。
而 McNemar 恰好只关心"两边不一致的对数"（b 和 c），
把"两边一致的"（都对 / 都错）**排除在外** —— 这正是二元配对的最优做法。

**paired_bootstrap 和 wilcoxon 怎么二选一？**
    · 关心"平均提升了多少"、要给置信区间 → **bootstrap**（它直接给 CI）
    · 只关心"有没有系统性地变好"、样本有小有大、差值有极端值 → **wilcoxon**
      （它只看符号和秩，抗极端值）
两个都跑一遍也不冲突，但报告里要说明**主口径是哪个** ——
两个都报却不说明哪个是主判据，等于把选择的负担推给读者。

==============================================================================
与 `Metrics._wilson_margin` 的分工（重要，别混用）
==============================================================================
`metrics.py` 里的 `_wilson_margin(n)` 和本模块**解决的是不同问题**：

    `_wilson_margin`        → **单比例的区间标注**。
                              输入是样本量，输出"这个比例本身的误差范围有多大"。
                              它是**描述性**的：回答"这个数字有多虚"。
                              它不做假设检验，也**不能用来比较两组**。

    本模块（significance）  → **组间对比**。
                              输入是两组成对观测，输出"差异是否可归因于随机"。
                              它是**推断性**的：回答"这个差异是真的吗"。

为什么不能拿 `_wilson_margin` 来比两组？
因为它用的是 `1.96 × 0.5 / √n` —— 一个**与数据无关的保守上界**
（p=0.5 时正态近似方差最大）。两组的"各自的误差范围"即使不重叠，
也不等于"差异显著"：**区间不重叠是显著的必要不充分条件**，
而且它完全没有利用配对结构。

    ★ 一句话记法：`_wilson_margin` 说"这个数字有多虚"，
      本模块说"两个数字的差是不是真的"。

`_wilson_margin` 保持原样不动 —— 它现在服务的那几处报告仍然正确。

================================================================================
零第三方依赖
================================================================================
只用 `math` / `random` / `statistics`。

真要做更复杂的检验（置换检验、FDR 校正、多层模型）生产上会用 `scipy` /
`statsmodels` —— 但**这个项目的身份是"零依赖可跑"**，
所以自己实现。代价是覆盖面窄（只做这三个最常用的），
好处是每个数字是怎么算出来的都能逐行讲清楚。
================================================================================
"""

from __future__ import annotations

import math
import random
import statistics

try:                                    # 允许独立运行（脚本目录直跑）
    from nightwatch.config import EVAL
except ImportError:                     # pragma: no cover
    EVAL = {                            # 兜底默认值，和 config 保持一致
        "bootstrap_n": 10000, "bootstrap_seed": 42,
        "bootstrap_ci_level": 0.95, "exact_test_threshold": 25,
    }


# =============================================================================
# 内部工具
# =============================================================================

def _paired(a, b) -> tuple[list[float], list[float]]:
    """校验两组成对观测，返回浮点化后的元组。

    ★ 长度不等就**直接抛异常**，不做截断或填充。
      配对检验的前提就是"一一对应"；长度不等说明调用方拿错了数据，
      这时候静默取较短的那个长度会让结果看起来正常但其实是错的 ——
      又一类"失败不报错"。
    """
    xa = [float(x) for x in a]
    xb = [float(x) for x in b]
    if len(xa) != len(xb):
        raise ValueError(
            f"配对检验要求两组等长，实际 {len(xa)} vs {len(xb)}。"
            f"若两组本就不是同一批样本，应改用独立样本检验（本模块不提供）——"
            f"但请先确认真的是独立设计，而不是数据对错了。")
    return xa, xb


def _quantile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（与 `statistics.quantiles` 的默认口径一致）。

    ★ 自己写而不用 `statistics.quantiles`：后者在 n=1 时抛异常、
      且返回的是"切点"而不是任意分位数，边界处理不合用。
      自己写能把边界情况说清楚。
    """
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _average_ranks(values: list[float]) -> list[float]:
    """返回**与输入同序**的平均秩（1 开始）。并列值取平均秩。

    ★ 并列必须取平均，不能按出现顺序编号 ——
      否则"先出现的那个值"会凭空获得更高的秩，
      而输入顺序在配对的 a/b 里是有意义的，这就引入了人造偏差。
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _tie_group_sizes(values: list[float]) -> list[int]:
    """各组并列值的个数（用于正态近似的方差校正）。"""
    s = sorted(values)
    sizes, i = [], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            sizes.append(j - i + 1)
        i = j + 1
    return sizes


# =============================================================================
# ① 配对 bootstrap
# =============================================================================

def paired_bootstrap(a: list[float], b: list[float],
                     n_boot: int | None = None,
                     seed: int | None = None,
                     ci_level: float | None = None) -> dict:
    """配对 bootstrap：对**差值向量**做有放回重采样。

    参数
    ----
    a, b      : 等长的两组观测（配对）。约定 **delta = b - a**，
                即 `mean_delta > 0` 表示 b 组更高。
    n_boot    : 重采样次数，默认取 `config.EVAL["bootstrap_n"]`。
    seed      : 随机种子，默认取 `config.EVAL["bootstrap_seed"]`。
                **必须可复现** —— 同一份数据两次跑出不同的 p 值，
                报告就没法被验证。
    ci_level  : 置信水平，默认 0.95。

    返回
    ----
    `{p_value, ci_low, ci_high, mean_delta, n, ...}`（外加方法说明字段）。

    ========================================================================
    为什么两件事用了**两种不同的重采样**（这是最容易写错的地方）
    ========================================================================
        置信区间 → 重采样**原始差值**
        p 值     → 重采样**中心化后的差值**

    原因是它们要回答的问题不同：

      · CI 要刻画"这个差值估计值本身有多不稳定" →
        必须围绕**观测到的**差值分布重采样，所以用原始差值。
        这也是"百分位 bootstrap"的标准做法。

      · p 值要回答"如果真实差异是 0，看到这么大的差异有多罕见" →
        必须**先假设 H0 成立**（把差值的均值平移到 0），再重采样。
        如果直接拿原始差值去数"有多少次跨过 0"，
        得到的是"CI 是否包含 0"，**和 p 值不是一回事** ——
        在偏斜的差值分布上两者会明显分歧。

    ★ 一句话：**CI 在"观测世界"里重采样，p 值在"零假设世界"里重采样。**
      两者的重采样对象不同，不能共用一个分布。

    ========================================================================
    已知局限
    ========================================================================
      · 样本量很小时（< 10 对），bootstrap 分布本身不可靠，
        p 值只能当**极粗略**的参考 —— 报告里必须写出 n。
      · n_boot 决定了 p 值的**最小可分辨刻度**（约 1/(n_boot+1)）。
        想报 p < 0.001 就得把 n_boot 调到万级以上。
      · 它检验的是**均值差**。若差值分布被极端值主导，
        均值本身就不是好的概括量 —— 这时看 `wilcoxon_signed_rank`
        （它对极端值不敏感），两个结论不一致时以 Wilcoxon 为准更稳。
    """
    n_boot = int(n_boot if n_boot is not None else EVAL["bootstrap_n"])
    seed = int(seed if seed is not None else EVAL["bootstrap_seed"])
    ci_level = float(ci_level if ci_level is not None
                     else EVAL["bootstrap_ci_level"])

    xa, xb = _paired(a, b)
    n = len(xa)
    if n == 0:
        return {"p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0,
                "mean_delta": 0.0, "n": 0, "n_boot": n_boot,
                "ci_level": ci_level, "method": "paired bootstrap",
                "note": "没有配对样本，无法检验"}

    deltas = [y - x for x, y in zip(xa, xb)]
    obs = statistics.fmean(deltas)
    rng = random.Random(seed)

    # --- ① 置信区间：原始差值的百分位法 ---
    boot_ci = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(n_boot))
    alpha = (1.0 - ci_level) / 2.0
    ci_low = _quantile(boot_ci, alpha)
    ci_high = _quantile(boot_ci, 1.0 - alpha)

    # --- ② p 值：在 H0（均值差 = 0）下模拟 ---
    centered = [d - obs for d in deltas]
    tol = 1e-12 * max(1.0, abs(obs))
    hits = sum(1 for _ in range(n_boot)
               if abs(sum(rng.choices(centered, k=n)) / n) >= abs(obs) - tol)
    # +1 平滑：避免 p = 0（重采样永远给不出"不可能发生"的结论）
    p_value = (hits + 1) / (n_boot + 1)

    # --- 人话结论，直接可放进报告 ---
    if p_value < 0.01:
        verdict = "差异高度显著（p < 0.01）"
    elif p_value < 0.05:
        verdict = "差异显著（p < 0.05）"
    else:
        verdict = f"差异不显著（p = {p_value:.3f} ≥ 0.05）—— 不能排除随机波动"

    return {
        "p_value": round(p_value, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "mean_delta": round(obs, 6),
        "n": n,
        "n_boot": n_boot,
        "ci_level": ci_level,
        "seed": seed,
        "method": "paired bootstrap",
        "verdict": verdict,
        "note": (f"{n} 对样本；CI 用原始差值百分位法，"
                 f"p 值用 H0 中心化重采样（两者重采样对象不同，见 docstring）。"
                 + ("" if n >= 10 else "⚠ 样本量 < 10，bootstrap 分布不可靠，仅作方向参考。")),
    }


# =============================================================================
# ② Wilcoxon 符号秩检验
# =============================================================================

def wilcoxon_signed_rank(a: list[float], b: list[float],
                         exact_threshold: int | None = None) -> dict:
    """Wilcoxon 符号秩检验（配对，非参数）。

    参数
    ----
    a, b            : 等长的两组观测。delta = b - a。
    exact_threshold : 小于该样本量用**精确检验**，否则用正态近似。
                      默认取 `config.EVAL["exact_test_threshold"]`（25）。

    返回
    ----
    `{p_value, statistic, n, n_zero, method, note, ...}`

    ========================================================================
    为什么用非参数检验：它检验的是"中位数差"而不是"均值差"
    ========================================================================
    它只看两点：**差值的方向**（正还是负）和**差值的秩**（相对大小），
    完全不用差值本身的大小。

    好处：少数几个极端值**无法**左右结论。
    代价：它对"差异的大小"不敏感 —— 如果所有样本都提升了 0.001，
    它也会报显著（"方向一致"），而实际收益微乎其微。

    **所以它必须和 `paired_bootstrap` 一起看**：
        两者都显著 + CI 明显远离 0  → 既真且有用
        显著但 CI 贴着 0            → 真的在提升，但幅度很小（可能不值得）
        不显著但 CI 很宽            → 样本不够，别急着下结论

    ========================================================================
    零差值怎么处理（这是本函数最需要说清的一处）
    ========================================================================
    标准做法是**剔除零差值**（它们不含方向信息），只对非零部分排秩，
    同时在结果的 `n_zero` 里如实报出剔除了几个。

    ⚠️ 但如果**全部**差值都是 0，本函数**不会**返回一个虚假的 p 值，
       而是明确告知"无差异可检验"（p_value = 1.0，附说明）——
       这种情况常见于"两次跑完全一致的确定性流程"，
       此时任何 p 值都是没有意义的（数据里根本没有变异）。

    ========================================================================
    精确检验的实现（纯标准库，不用查表）
    ========================================================================
    n < 25 时不再查临界值表，而是**直接算出 W+ 的完整分布**：
    W+ 等于"从这些秩里任取一个子集，秩和是多少"，
    用子集和计数的动态规划一次算完（O(n · Σ秩)）。

    ★ 一个小技巧：并列值的平均秩会出现 0.5（如 1.5、2.5），
      直接按整数做 DP 会漏掉它们。把秩**全部 ×2** 之后都是整数，
      DP 用列表下标即可，最后再按比例比较。

    已知局限：n 很大时 DP 的规模是 O(n²)，所以设了 25 的上限；
    超过就走正态近似（含并列校正与连续性校正）。
    """
    exact_threshold = int(exact_threshold if exact_threshold is not None
                          else EVAL["exact_test_threshold"])

    xa, xb = _paired(a, b)
    deltas = [y - x for x, y in zip(xa, xb)]
    nonzero = [d for d in deltas if d != 0.0]
    n_zero = len(deltas) - len(nonzero)
    n = len(nonzero)

    base = {"statistic": 0.0, "n": n, "n_zero": n_zero,
            "n_pairs": len(deltas)}

    # --- 全零：如实说"检验不了"，而不是编一个 p 值出来 ---
    if n == 0:
        return {**base, "p_value": 1.0,
                "method": "无差异可检验",
                "verdict": "无差异可检验",
                "note": (f"全部 {len(deltas)} 对样本的差值都是 0 —— "
                         f"数据里没有变异，任何 p 值都没有意义。"
                         f"这常见于重复跑同一套确定性流程。")}

    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)

    if n < exact_threshold:
        scaled = sorted(int(round(r * 2)) for r in ranks)
        total = sum(scaled)
        counts = [0] * (total + 1)
        counts[0] = 1
        for r in scaled:                    # 0/1 背包式子集和计数
            for s in range(total - r, -1, -1):
                if counts[s]:
                    counts[s + r] += counts[s]
        denom = float(sum(counts))          # = 2 ** n
        target = int(round(w * 2))
        p_value = min(1.0, 2.0 * sum(counts[:target + 1]) / denom)
        method = f"精确检验（n={n} < {exact_threshold}，枚举 W+ 的完整分布）"
    else:
        mean_w = n * (n + 1) / 4.0
        tie_corr = sum(t ** 3 - t for t in _tie_group_sizes(
            [abs(d) for d in nonzero]))
        var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
        if var_w <= 0:
            p_value, method = 1.0, "正态近似（方差退化，无法计算）"
        else:
            # 连续性校正：把统计量朝远离期望的方向挪 0.5
            z = max(0.0, abs(w - mean_w) - 0.5) / math.sqrt(var_w)
            p_value = math.erfc(z / math.sqrt(2.0))     # = 2·(1−Φ(z))
            method = (f"正态近似（n={n} ≥ {exact_threshold}，"
                      f"含并列校正与连续性校正）")

    if p_value < 0.01:
        verdict = "差异高度显著（p < 0.01）"
    elif p_value < 0.05:
        verdict = "差异显著（p < 0.05）"
    else:
        verdict = f"差异不显著（p = {p_value:.3f} ≥ 0.05）"

    return {
        **base,
        "p_value": round(min(1.0, p_value), 6),
        "statistic": w,                       # W = min(W+, W-)
        "w_plus": w_plus,
        "w_minus": w_minus,
        "method": method,
        "verdict": verdict,
        "note": (f"检验的是中位数差（不敏感于极端值）。"
                 + (f"剔除了 {n_zero} 对零差值。" if n_zero else "")
                 + "需与 paired_bootstrap 的 CI 一起看："
                   "显著但 CI 贴 0 说明幅度很小。"),
    }


# =============================================================================
# ③ McNemar 检验（二元配对）
# =============================================================================

def mcnemar(b_count: int, c_count: int,
            exact_threshold: int | None = None) -> dict:
    """McNemar 检验：**二元**结果的配对检验。

    参数
    ----
    b_count : 方向一的不一致对数。约定 **b = "A 对、B 错"** 的数量。
    c_count : 方向二的不一致对数，即 **c = "A 错、B 对"** 的数量。
    exact_threshold : `b + c` 小于该值时用二项精确检验，否则卡方近似。

    返回
    ----
    `{p_value, b, c, n_discordant, method, verdict, note, ...}`

    ========================================================================
    为什么只数"不一致"的对（b 和 c），把"都对的"和"都错的"扔掉？
    ========================================================================
    这是 McNemar 的全部思想，也是它比"两个比例相减"强的地方。

    考虑 61 条样本：如果 50 条两次都对、9 条两次都错、1 条从错变对、1 条从对变错。
        "都对的 50 条"   → 对**差异**没有任何信息（两边都是 1，抵消）
        "都错的 9 条"    → 同样没有信息（两边都是 0，抵消）
        真正的证据全在 b=1 / c=1 这两条里

    用"命中率 51/61 vs 51/61"这种比例相减，会被那 50 条"都对的"稀释，
    **看起来差异微不足道**；而 McNemar 直接问：
    "在 2 次翻转里，1 比 1 分成两半，这像不像随机？" —— 答案是像，不显著。
    这个结论才是对的，而它的依据只有 2 个数据点，也如实反映了证据之弱。

    ★ 一句话：**配对二元数据的有效样本量是 b + c，不是 n。**
      报告里应写明 b、c 和 b+c —— 只看 b+c 很小的 p 值会误以为证据很强。

    ========================================================================
    精确 vs 近似
    ========================================================================
    b + c < 25：H0 下 b ~ Binomial(b+c, 0.5)，直接算双侧精确 p 值。
                这是"抛硬币"的检验，完全不依赖近似。
    b + c ≥ 25：卡方近似（**含连续性校正**）。
                不加校正时卡方值偏大、p 值偏小，在小样本上会虚报显著。

    b + c = 0 时**不做检验**，如实返回"无差异可检验" ——
    此时两版在所有样本上表现完全一致。

    ========================================================================
    已知局限
    ========================================================================
      · 它只回答"翻转是否对称"，**不回答"翻转的幅度"**。
        想知道"从错变对的那些，F1 提升了多少"，
        要把那一子集单独拎出来用 `paired_bootstrap`。
      · 它假设每对样本独立。若样本间有分组结构（如同一作者提交的多条 issue），
        需要换成考虑聚类的检验 —— 本模块不提供。
    """
    exact_threshold = int(exact_threshold if exact_threshold is not None
                          else EVAL["exact_test_threshold"])
    b, c = int(b_count), int(c_count)
    if b < 0 or c < 0:
        raise ValueError(f"不一致对数不能为负：b={b}, c={c}")
    n_disc = b + c

    base = {"b": b, "c": c, "n_discordant": n_disc,
            "direction": "b = A 对 B 错；c = A 错 B 对"}

    if n_disc == 0:
        return {**base, "p_value": 1.0,
                "method": "无差异可检验",
                "verdict": "无差异可检验",
                "note": ("两版在所有样本上表现完全一致（0 次翻转）—— "
                         "此时没有可检验的证据，不是'证明了等价'，"
                         "而是'这批数据里看不出差别'。")}

    if n_disc < exact_threshold:
        k = min(b, c)
        cum = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
        p_value = min(1.0, 2.0 * cum)
        method = (f"二项精确检验（b+c={n_disc} < {exact_threshold}，"
                  f"H0 下 b ~ Binomial({n_disc}, 0.5)）")
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc        # 连续性校正
        chi2 = max(0.0, chi2)
        p_value = math.erfc(math.sqrt(chi2 / 2.0))   # 1 自由度卡方的双侧 p
        method = f"卡方近似（b+c={n_disc} ≥ {exact_threshold}，含连续性校正）"

    if p_value < 0.01:
        verdict = "翻转方向高度不对称（p < 0.01）"
    elif p_value < 0.05:
        verdict = "翻转方向显著不对称（p < 0.05）"
    else:
        verdict = (f"翻转方向对称（p = {p_value:.3f} ≥ 0.05）——"
                   f"两版差异不显著")

    return {
        **base,
        "p_value": round(min(1.0, p_value), 6),
        "method": method,
        "verdict": verdict,
        "note": (f"有效样本量是 b+c={n_disc}（不是总样本数）——"
                 f"一致的样本对差异没有信息，已被排除。"
                 + ("" if n_disc >= 10 else
                    "⚠ b+c < 10，即使显著也只是「方向性证据」，"
                    "不宜单独作为结论依据。")),
    }


# =============================================================================
# ④ 多重比较提示
# =============================================================================

def bonferroni_note(k: int, alpha: float | None = None) -> str:
    """生成一句"多重比较"提示文本，供报告直接打印。

    参数
    ----
    k     : 一共比较了几个配置 / 阈值 / 候选。
    alpha : 原始显著性水平，默认取 `config.EVAL["alpha"]`（0.05）。

    返回
    ----
    可直接放进报告的中文提示（单段字符串）。

    ==========================================================================
    ★★ 为什么"在 k 个候选里挑最优"这件事本身需要提示 ★★
    ==========================================================================
    这是最容易被忽略、又最容易造成错误结论的一类问题：

        扫描 18 个阈值、挑 F1 最高的那个 → 那个"最优"必然偏高
        跑 9 组正交实验、挑效果最好的配置 → 同上

    原因很直接：**比较次数越多，"碰巧看起来最好"的那一个就越可能出现。**
    即使所有配置的真实效果完全相同，k 次比较里也几乎必然有一个
    看起来"显著更好" —— 这就是选择偏差（selection bias），
    也叫"在多条岔路上挑了最好看的那条"。

    Bonferroni 给出的是一个**保守**的修正：要把单次比较的显著性门槛从
    α 降到 **α/k**，才能保证整体错误率不超过 α。

    ==========================================================================
    ★ 为什么这里只给提示、不做完整校正计算
    ==========================================================================
    三个理由：
      ① 完整的多重比较校正需要**全部**候选的 p 值列表。
         而阈值扫描/L9 实验当前只保留了"最优值"，中间的 p 值没存下来 ——
         没有输入，算不出正确的校正结果。
      ② **更根本的问题是选择偏差，而不只是 p 值门槛。**
         即使做了 Bonferroni，那个"最优配置"仍然是在这批数据上挑出来的，
         它的效果估计依然偏高。真正要做的不是把门槛调严，
         而是**把挑出来的配置拿到独立数据上复验**。
      ③ 目标是**把意识写进报告**，而不是给出一个精确数字。
         报告里写明"存在多重比较问题、最优值需复验"，
         比给一个没人检查的校正 p 值有用得多。

    ★ 若将来真的要做完整校正：候选少（k 小）用 Bonferroni 就够；
      k 很大时它过于保守，应改用 Benjamini-Hochberg 的 FDR 控制
      （允许一定比例的假阳性，功效更高）—— 但那需要完整的 p 值列表，
      所以第一步是把候选的 p 值都存下来。
    """
    alpha = float(alpha if alpha is not None else EVAL.get("alpha", 0.05))
    k = int(k)
    if k < 1:
        raise ValueError(f"比较次数必须 ≥ 1，实际 {k}")

    if k == 1:
        return ("只比较了 1 个候选，**不存在多重比较问题**，"
                "无需校正。")

    corrected = alpha / k
    return (
        f"⚠ 多重比较提示：以上结果是在 {k} 个候选里挑出的**最优值**。\n"
        f"    · 未校正的多重比较：单次比较用了 α = {alpha}，"
        f"但比较了 {k} 次\n"
        f"    · Bonferroni 校正后，单次显著性阈值应为 "
        f"α/k = {alpha}/{k} = {corrected:.5f}\n"
        f"    · 换句话说：只有 p < {corrected:.5f} 才够资格称为"
        f"「在多候选比较下仍然显著」\n"
        f"    ⚠ 但**校正只解决门槛问题，不解决选择偏差** ——\n"
        f"      这个最优值仍是在**同一批数据**上挑出来的，效果估计天然偏高。\n"
        f"      **正确做法：把最优配置拿到独立数据上复验**，再对外引用。\n"
        f"    （候选数量大时 Bonferroni 过于保守，可考虑 BH 的 FDR 控制，"
        f"但那需要保存所有候选的 p 值）"
    )


# =============================================================================
# 汇总：一次给出"选哪个"的建议
# =============================================================================

def recommend_test(metric_kind: str) -> str:
    """按指标类型给出该用哪个检验（把 docstring 里的选择规则变成可调用的东西）。

    `metric_kind` 取 `"binary"`（命中/未命中）或 `"continuous"`
    （F1 / 步数 / token / 时延等）。

    ★ 为什么要做成函数而不是让人去读 docstring：
      选择规则如果只存在于文档里，就一定会有人凭直觉选错。
      写成函数后，调用点可以显式声明"我知道这是二元指标"，
      而且能被测试覆盖。
    """
    kind = str(metric_kind).strip().lower()
    if kind in ("binary", "bool", "boolean", "二元"):
        return "mcnemar"
    if kind in ("continuous", "float", "连续"):
        return "paired_bootstrap（要看幅度和 CI）或 wilcoxon_signed_rank（要抗极端值）"
    raise ValueError(f"未知的指标类型 {metric_kind!r}；"
                     f"只支持 'binary' 和 'continuous'")


def significance_summary(a: list[float], b: list[float],
                         metric_kind: str = "continuous") -> dict:
    """一次跑完该跑的检验，并把"主判据是哪个"说明白。

    连续指标 → 同时给 bootstrap（幅度 + CI）和 Wilcoxon（是否系统性）；
    二元指标 → 只给 McNemar（此时 a/b 应当是 0/1 序列，
               内部自行为统计 b、c 两个方向的不一致对数）。

    ★ 为什么要有一个"汇总"入口：报告里最怕的是**同一份数据跑了两个检验
      却不说以哪个为准**。这个函数把主判据写在返回值的 `primary` 里，
      让它成为报告的一部分，而不是留在作者脑子里。
    """
    kind = str(metric_kind).strip().lower()
    is_binary = kind in ("binary", "bool", "boolean", "二元")

    if is_binary:
        xa, xb = _paired(a, b)
        # b = "a 对、b 错"；c = "a 错、b 对"
        b_cnt = sum(1 for x, y in zip(xa, xb) if x == 1 and y == 0)
        c_cnt = sum(1 for x, y in zip(xa, xb) if x == 0 and y == 1)
        return {
            "metric_kind": "binary",
            "primary": "mcnemar",
            "mcnemar": mcnemar(b_cnt, c_cnt),
            "why": ("二元指标没有「幅度」，只有「翻没翻」，"
                    "有效样本量是翻转对数（b+c）。"),
        }

    return {
        "metric_kind": "continuous",
        "primary": "paired_bootstrap",
        "paired_bootstrap": paired_bootstrap(a, b),
        "wilcoxon_signed_rank": wilcoxon_signed_rank(a, b),
        "why": ("bootstrap 给「平均提升多少」和置信区间（主判据）；"
                "Wilcoxon 给「是否系统性变好」，抗极端值（辅助印证）。"
                "两者结论冲突时：差值有极端值就以 Wilcoxon 为准。"),
    }
