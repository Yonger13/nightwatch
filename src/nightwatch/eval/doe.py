"""
Nightwatch · 实验设计（DOE, Design of Experiments）

================================================================================
为什么这个模块存在：单变量实验会给出错误结论
================================================================================
MIT 那个受控实验（44 张发票 × 4 种模型配置）的结论是反直觉的：

    **能力最强、成本最高的模型配置，反而表现最差。**

但它的真正启示不是"用便宜模型"，而是：

    单独测试每个组件会得出错误结论，
    因为 Agent 的表现来自**整条链路的配合**。

举例说明为什么单变量测试会骗人：

    测试 A：固定记忆策略=开，比较模型    → 强模型更好
    测试 B：固定模型=强，比较记忆策略     → 开记忆更好
    结论：强模型 + 开记忆 = 最优？

    **错。** 因为可能存在交互效应：
    强模型已经能从上下文里推断出答案，记忆召回的噪声反而干扰它；
    而弱模型需要记忆补充信息，开记忆收益很大。

    两个单独看都"更好"的选项，组合起来可能更差。

**要发现交互效应，必须让多个因子同时变化。**

================================================================================
正交实验：用 9 次试验代替 27 次
================================================================================
3 个因子、每个 3 个水平，全因子需要 3³ = 27 次试验。
但如果只关心**主效应**（每个因子单独的影响），
正交表 L9(3³) 只需要 **9 次** —— 而且能分离出主效应。

    L9 的设计原则：**任意两个因子的水平组合都恰好出现一次**。
    这让"因子 A 的不同水平"在因子 B 的每个水平下都被比较过，
    所以 A 的主效应不会被 B 的分布失衡所干扰。

    代价：**主效应可以估，但两两交互效应估不了**（会混在一起）。
    全因子才能估交互。

所以本模块提供两种设计：
    orthogonal_l9()   9 次，估主效应（便宜，先用这个筛）
    full_factorial()  27 次，估主效应 + 交互（贵，需要时再用）

**这个"先正交筛、再全因子验证"的策略，
是工业界做实验的标准顺序** —— 不是所有组合都值得花全因子的成本。

================================================================================
一个诚实的说明
================================================================================
正交设计的"精度"是用**自由度**换来的。
9 次试验只有 8 个自由度，分给 3 个因子（各 2 个）后，
就只剩 2 个给误差项 —— **统计功效有限**。

所以本项目的结论会明确标注：
  · 这是一个**筛选性**实验，不是确证性实验
  · 差异小于 0.1 的结论不该采信（样本量和自由度都不够）
  · 要确证需要全因子 + 更多样本

**把统计局限写清楚，比给一个漂亮的数字更有价值。**
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Factor:
    """一个实验因子：有名字、有若干个水平（取值）。"""
    name: str
    levels: list[Any]
    description: str = ""

    def __post_init__(self):
        if len(self.levels) < 2:
            raise ValueError(f"因子 {self.name} 至少需要 2 个水平")


@dataclass
class Run:
    """一次试验的配置（一个因子水平的组合）。"""
    run_id: str
    settings: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """人类可读的短标签，用于图表。"""
        return " / ".join(str(v) for v in self.settings.values())


@dataclass
class Design:
    """一次实验设计。"""
    name: str
    factors: list[Factor]
    runs: list[Run]
    kind: str = "orthogonal"          # orthogonal | full
    notes: str = ""

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def summary(self) -> str:
        lines = [f"{self.name}（{self.kind}，{self.n_runs} 次试验）"]
        for f in self.factors:
            lines.append(f"  因子 {f.name}：{f.levels}")
        return "\n".join(lines)


# =============================================================================
# L9(3³) 正交表
# =============================================================================
# 这是标准的三因子三水平正交表。每一行是一次试验。
#
# 读法：第 i 行第 j 列的数字表示"第 j 个因子取第几水平的索引"。
# 例如第一行 (0,0,0) 表示三个因子都取第一个水平。

_L9_MATRIX = [
    (0, 0, 0),
    (0, 1, 1),
    (0, 2, 2),
    (1, 0, 1),
    (1, 1, 2),
    (1, 2, 0),
    (2, 0, 2),
    (2, 1, 0),
    (2, 2, 1),
]


def orthogonal_l9(factors: list[Factor], name: str = "L9 正交实验") -> Design:
    """生成 L9(3³) 正交设计 —— **恰好支持 3 个因子，每个 3 个水平**。

    ★ 为什么限制得这么死？
    因为 L9 就是为"3 因子 3 水平"设计的。如果因子数或水平数不是 3，
    硬套会破坏正交性（"任意两因子的组合恰好出现一次"这个性质没了），
    那样算出来的主效应就是错的 —— **而错误的实验设计比不做实验更糟**，
    因为它会给出一个看起来很正式的错误结论。

    所以这里**宁可报错也不硬套**。
    """
    if len(factors) != 3:
        raise ValueError(
            f"L9 正交表要求恰好 3 个因子，收到 {len(factors)} 个。"
            f"（因子数不匹配时硬套会破坏正交性，让主效应估计失真 —— "
            f"需要非 3 因子请用 full_factorial）")
    for f in factors:
        if len(f.levels) != 3:
            raise ValueError(
                f"L9 要求每个因子恰好 3 个水平，因子 {f.name} 有 "
                f"{len(f.levels)} 个")

    runs = []
    for i, combo in enumerate(_L9_MATRIX, 1):
        runs.append(Run(
            run_id=f"run{i:02d}",
            settings={f.name: f.levels[idx] for f, idx in zip(factors, combo)},
        ))
    return Design(
        name=name, factors=factors, runs=runs, kind="orthogonal",
        notes=("L9(3³) 正交表：9 次试验可估 3 个因子的主效应，"
               "但估不了两两交互（自由度不够）。属于筛选性实验。"),
    )


def full_factorial(factors: list[Factor], name: str = "全因子实验") -> Design:
    """生成全因子设计 —— 因子数不受限，但组合数是指数增长。

    ★ 什么时候值得用全因子？
    当你要**验证交互效应**的时候。

    正交表只能告诉你"A 比 B 好"，全因子能告诉你
    "A 在 C 的条件下比 B 好，但在 D 的条件下不如 B"。

    **后者往往才是真正的发现** —— 因为 Agent 系统的表现
    恰恰高度依赖组件之间的搭配。
    """
    if not factors:
        raise ValueError("至少需要一个因子")

    runs = []
    for i, combo in enumerate(itertools.product(*[f.levels for f in factors]), 1):
        runs.append(Run(
            run_id=f"run{i:02d}",
            settings={f.name: v for f, v in zip(factors, combo)},
        ))

    total = 1
    for f in factors:
        total *= len(f.levels)
    return Design(
        name=name, factors=factors, runs=runs, kind="full",
        notes=(f"{len(factors)} 因子全组合，共 {total} 次试验。"
               f"可估主效应 + 两两交互，但成本随因子数指数增长。"),
    )


# =============================================================================
# 效应分析
# =============================================================================

def main_effects(design: Design, results: dict[str, float]) -> dict[str, dict]:
    """计算各因子的主效应。

    ============================================================
    主效应 = "这个因子单独看，不同水平的表现差多少"
    ============================================================
    算法：对每个因子的每个水平，取"该水平出现的所有试验"的效果均值，
    然后比较各水平均值。

    由于（正交设计下）每个水平出现的次数相同，而且其他因子的水平
    分布也是均衡的，所以这个均值差异**只反映该因子的影响**。

    如果用的是非正交的（比如随机的）设计，这个结论就不成立 ——
    因为某个水平的试验可能恰好都碰到了"另一个因子的坏取值"。
    **这就是正交性的价值。**

    参数：
        results: run_id → 效果分数（如 F1）
    """
    out: dict[str, dict] = {}
    for f in design.factors:
        buckets: dict[Any, list[float]] = {lv: [] for lv in f.levels}
        for run in design.runs:
            if run.run_id not in results:
                continue
            buckets[run.settings[f.name]].append(results[run.run_id])
        means = {lv: (sum(v) / len(v) if v else 0.0)
                 for lv, v in buckets.items()}
        best = max(means, key=lambda k: means[k]) if means else None
        worst = min(means, key=lambda k: means[k]) if means else None
        out[f.name] = {
            "means": means,
            "best": best,
            "worst": worst,
            # 极差（best - worst）：这个因子对结果的影响幅度
            "range": (means[best] - means[worst]) if means else 0.0,
        }
    return out


def cost_effect_points(design: Design, results: dict[str, float],
                       costs: dict[str, float]) -> list[dict]:
    """把每次试验整理成"成本-效果"坐标点，用于画曲线。

    返回的每一项包含 run_id / 配置 / 成本 / 效果，
    以及一个**帕累托最优标记** ——
    只有帕累托前沿上的点才是"值得选"的配置。
    """
    pts = []
    for run in design.runs:
        if run.run_id not in results:
            continue
        pts.append({
            "run_id": run.run_id,
            "label": run.label(),
            "settings": dict(run.settings),
            "cost": costs.get(run.run_id, 0.0),
            "effect": results[run.run_id],
        })

    # 帕累托前沿：没有被任何其他点"成本更低且效果更好"的点
    #
    # ★ 为什么用帕累托而不是"加权求和排名"？
    #   因为"该多花 1 块钱换 0.01 的 F1"这个问题**没有客观答案** ——
    #   它取决于业务容忍度。
    #   帕累托前沿给出的是"所有理性选择"，把取舍留给决策者。
    #   **评测报告不该替业务做价值判断。**
    for p in pts:
        dominated = any(
            (q["cost"] <= p["cost"] and q["effect"] > p["effect"]) or
            (q["cost"] < p["cost"] and q["effect"] >= p["effect"])
            for q in pts if q is not p
        )
        p["on_pareto_frontier"] = not dominated
    return pts


def pareto_frontier(points: list[dict]) -> list[dict]:
    """从成本-效果点里挑出帕累托前沿（按成本升序）。"""
    front = [p for p in points if p.get("on_pareto_frontier")]
    return sorted(front, key=lambda p: p["cost"])


def recommend(design: Design, results: dict[str, float],
              costs: dict[str, float]) -> dict:
    """给出推荐配置。

    ★ **这里刻意给出三个不同的"最优"，而不是一个。**

    因为"最优"取决于优化目标，而目标是业务决定的：

        最省钱    → 效果达标的前提下成本最低
        效果最好  → 不管成本，效果最高的那个
        性价比    → 效果/成本比值最高（拐点附近）

    **只给一个"最优"是在替业务做决定**，而评测工具的职责是
    把权衡摊开，让决定有依据。
    """
    pts = cost_effect_points(design, results, costs)
    if not pts:
        return {"error": "没有可用的实验结果"}

    best_effect = max(pts, key=lambda p: p["effect"])
    cheapest = min(pts, key=lambda p: p["cost"])
    # 性价比：效果 / 成本（避免除零）
    cost_effect_ratio = [
        {**p, "ratio": (p["effect"] / p["cost"] if p["cost"] > 0 else 0.0)}
        for p in pts
    ]
    best_ratio = max(cost_effect_ratio, key=lambda p: p["ratio"])

    # 效果达标的候选里，成本最低的
    threshold = best_effect["effect"] * 0.98      # 允许 2% 的损耗
    qualified = [p for p in pts if p["effect"] >= threshold]
    cheapest_qualified = min(qualified, key=lambda p: p["cost"]) \
        if qualified else cheapest

    return {
        "最省效果": best_effect,
        "最省钱": cheapest,
        "性价比最高": best_ratio,
        "达标里最省": cheapest_qualified,
        "帕累托前沿": pareto_frontier(pts),
    }
