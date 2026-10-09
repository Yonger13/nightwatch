#!/usr/bin/env python3
"""
Nightwatch · PR 风险阈值标定（P2-6）

    # 用已有数据标定
    python scripts/calibrate_pr_threshold.py \\
        --merged data/real_prs.json --unmerged data/unmerged_prs.json

    # 只看分布不扫阈值
    python scripts/calibrate_pr_threshold.py --merged ... --dist-only

==============================================================================
为什么需要这个脚本
==============================================================================
`pr_review.py` 里的大改动阈值是 **500 行** —— 一个"看起来合理"的经验值。

但实测发现真实 PR 的改动量中位只有 **76 行**（已合并）/ **26 行**（未合并）。
**所以 500 行这个阈值几乎从不触发** —— 一个 44 文件、+252 行的 PR
仍然被判"低风险"。

这就是 `docs/06` 里"**阈值没有数据支撑**"那条问题的第二次出现
（第一次是记忆召回阈值，P0-3 修过）。

**这个脚本的作用：把阈值从"经验值"变成"数据标定值"。**

==============================================================================
★ 标定方法：先定目标，再看分布
==============================================================================
**关键**：阈值不该"取分布的某个分位数"，而应该**由目标决定**。

    阈值的目标是什么？
    → 让"高风险"这个标签有区分度：**值得人优先看的那一小部分**。

如果阈值太低，所有 PR 都是高风险 → 标签失去意义（人还是要全看）
如果阈值太高，没有 PR 是高风险的 → 规则从不触发（等于没有规则）

所以标定问题是：**"高风险应该占多少比例"是合理的？**

本脚本的做法：
  ① 扫描一批候选阈值
  ② 对每个阈值算出"高风险比例"
  ③ **推荐让高风险比例落在 10%~20% 的阈值**

**为什么是 10%~20%？**
  · 这是"人工优先看"能承受的量级（一天 20 个 PR → 优先看 2~4 个）
  · 而且它让"高风险"成为**少数派**，才有信号价值
  · 如果业务的 review 能力不同，这个区间应该相应调整 ——
    **所以脚本把它做成参数，而不是硬编码。**

==============================================================================
★ 用两类样本做对比（这是本脚本最有价值的部分）
==============================================================================
已合并 vs 未合并，两组分布应该**有差异** —— 否则"改动量"就不是个有效信号。

    如果两组分布几乎重合 → 改动量区分不出"被接受的"和"被拒绝的"
                          → 该换信号（比如看"是否动了敏感文件"）

**这个对比本身就是在验证"信号有效性"，而不只是标定阈值。**
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.tools.pr_review import parse_diff                    # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def load_prs(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("prs", [])


def churn_of(pr: dict) -> int:
    """取一个 PR 的改动量（增 + 删）。

    ★ 优先用已存的字段；如果缺了就从 diff 现算。

    **这个"回退"很重要**：数据是不同时间采集的，
    早期版本可能没有 `additions`/`deletions` 字段。
    如果直接 KeyError，那些样本就会被静默丢掉 ——
    而"静默丢样本"会让统计结果失真。
    """
    a = pr.get("additions")
    d = pr.get("deletions")
    if a is None or d is None:
        files = parse_diff(pr.get("diff", ""))
        a = sum(f.additions for f in files)
        d = sum(f.deletions for f in files)
    return int(a) + int(d)


def describe(values: list[int], label: str) -> dict:
    """输出一组数据的分布描述（分位数）。"""
    if not values:
        return {}
    s = sorted(values)

    def pct(p: float) -> int:
        # 用最近秩法取分位数 —— 不做插值，因为结果是"行数"（整数更直观）
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[idx]

    return {
        "label": label,
        "n": len(s),
        "min": s[0],
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": s[-1],
        "mean": round(statistics.mean(s), 1),
    }


def print_dist(d: dict) -> None:
    if not d:
        return
    print(f"  {d['label']}（n={d['n']}）")
    print(f"    最小 {d['min']:>6}　p25 {d['p25']:>6}　"
          f"p50 {d['p50']:>6}　p75 {d['p75']:>6}")
    print(f"    p90 {d['p90']:>6}　p95 {d['p95']:>6}　"
          f"最大 {d['max']:>6}　均值 {d['mean']}")


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="PR 风险阈值标定")
    ap.add_argument("--merged", default=str(ROOT / "data" / "real_prs.json"),
                    help="已合并 PR 的采集结果")
    ap.add_argument("--unmerged",
                    default=str(ROOT / "data" / "unmerged_prs.json"),
                    help="未合并 PR 的采集结果")
    ap.add_argument("--target-low", type=float, default=0.10,
                    help="目标高风险比例下限（默认 10%%）")
    ap.add_argument("--target-high", type=float, default=0.20,
                    help="目标高风险比例上限（默认 20%%）")
    ap.add_argument("--dist-only", action="store_true",
                    help="只看分布不扫阈值")
    ap.add_argument("--json", help="结果 JSON 路径")
    args = ap.parse_args()

    merged = load_prs(Path(args.merged))
    unmerged = load_prs(Path(args.unmerged))

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · PR 风险阈值标定{C['reset']}")
    hr()
    print(f"  已合并样本    {len(merged)} 个（{Path(args.merged).name}）")
    print(f"  未合并样本    {len(unmerged)} 个（{Path(args.unmerged).name}）")

    if not merged and not unmerged:
        print(f"\n  {C['red']}✗ 没有可用数据{C['reset']}")
        print(f"    先采集：python scripts/collect_prs.py --limit 20")
        return 1

    m_churn = [churn_of(p) for p in merged]
    u_churn = [churn_of(p) for p in unmerged]
    all_churn = m_churn + u_churn

    # ---------------------------------------------------------- ① 分布
    hr("① 改动量分布（这就是「数据支撑」）")

    dm = describe(m_churn, "已合并 PR")
    du = describe(u_churn, "未合并 PR")
    da = describe(all_churn, "全部样本")
    print_dist(dm)
    print()
    print_dist(du)
    print()
    print_dist(da)

    # ---------------------------------------------------------- ② 信号有效性
    hr("② ★ 信号有效性检验：两组分布有差异吗？")

    if dm and du:
        gap_p50 = dm["p50"] - du["p50"]
        gap_p90 = dm["p90"] - du["p90"]
        print(f"""
  已合并 p50 = {dm['p50']}　未合并 p50 = {du['p50']}　差 {gap_p50:+d}
  已合并 p90 = {dm['p90']}　未合并 p90 = {du['p90']}　差 {gap_p90:+d}
""")
        # ★ 判断标准分两层：**有没有差异**，以及**方向对不对**
        #
        #   第一版只判了"差异够不够大"，结果漏掉了一个更重要的事：
        #   **差异的方向是反的。**
        #
        #   假设是"改动大的 PR 更容易被拒"。
        #   实测却是：未合并的 p50（27）**比已合并的（139）还小**。
        #
        #   这说明"改动量"和"被接受"之间**没有正相关** ——
        #   小改动被拒的更多（方向不对、重复劳动、粒度不合适），
        #   而改动大不一定是被拒的原因。
        base = max(du["p50"], 1)
        ratio = abs(gap_p50) / base
        has_gap = ratio >= 0.5
        direction_expected = gap_p50 > 0      # 期望"已合并的改动量更大"

        if has_gap and direction_expected:
            print(f"  {C['green']}✓ 两组分布有差异，且方向符合预期"
                  f"（已合并的改动量更大）{C['reset']}")
            print(f"    → 改动量是个有效信号")
        elif has_gap and not direction_expected:
            print(f"""  {C['red']}✗ 两组有差异，但**方向和预期相反**{C['reset']}

    未合并 PR 的改动量（p50={du['p50']}）**比已合并的（p50={dm['p50']}）还小**。

    这说明"改动量大"**不是**被拒绝的原因 —— 恰恰相反，
    未合并的 PR 里有大量**小改动**（可能因为方向不对、
    重复劳动、粒度不合适而被拒）。

    {C['bold']}所以这个结果的含义是：{C['reset']}
      · **"改动量大 → 风险高"这个假设，在"是否被接受"这个标准上不成立**
      · 但注意区分两件事：
          「改动量大」→ 值得被人优先看（**工作量分配**）  ← 这个仍然成立
          「改动量大」→ 更容易被拒绝（**预测性**）        ← 这个不成立

    {C['dim']}  阈值的作用是前者（把人的注意力导到复杂改动上），
      不是后者（预测 PR 会不会被拒）。**这两个目标不该混为一谈。**
      本脚本标定的是前者 —— 所以下面的阈值扫描仍然有意义，
      只是不该宣称"它能预测 PR 是否被接受"。{C['reset']}""")
        else:
            print(f"""  {C['yellow']}⚠ 两组分布没有明显差异{C['reset']}

    {C['dim']}改动量区分不出"被接受的"和"被拒绝的" PR。
    阈值仍然可以作为"工作量分配"的依据（让复杂改动优先被看），
    但**不能宣称它能预测 PR 的接受结果**。{C['reset']}""")

    # ---------------------------------------------------------- ③ 阈值扫描
    if args.dist_only:
        return 0

    hr("③ 阈值扫描：不同阈值下有多少 PR 会被判「高风险」")

    lo, hi = args.target_low, args.target_high
    print(f"""
  {C['dim']}标定目标：让"高风险"比例落在 {lo:.0%} ~ {hi:.0%}
    · 太低 → 全被判高风险，标签失去区分度
    · 太高 → 规则从不触发，等于没有规则
    {lo:.0%}~{hi:.0%} 对应"人工优先看"能承受的量级。{C['reset']}
""")

    n_all = len(all_churn)
    candidates = sorted({10, 50, 100, 150, 200, 300, 400, 500, 800, 1000,
                         max(1200, max(all_churn) if all_churn else 1200)})

    print(f"  {'阈值(行)':>10}{'高风险数':>10}{'比例':>9}　{'已合并中':>10}"
          f"{'未合并中':>10}")
    print(f"  {'-' * 54}")

    rows = []
    for th in candidates:
        n_high = sum(1 for c in all_churn if c > th)
        rate = n_high / n_all if n_all else 0
        m_high = sum(1 for c in m_churn if c > th)
        u_high = sum(1 for c in u_churn if c > th)
        rows.append({"threshold": th, "n_high": n_high, "rate": rate,
                     "merged_high": m_high, "unmerged_high": u_high})
        mark = ""
        if lo <= rate <= hi:
            mark = f"  {C['green']}← 落在目标区间{C['reset']}"
        print(f"  {th:>10}{n_high:>10}{rate:>8.1%}　{m_high:>10}{u_high:>10}{mark}")

    # ---------------------------------------------------------- ④ 推荐
    hr("④ 推荐阈值")

    in_range = [r for r in rows if lo <= r["rate"] <= hi]
    if in_range:
        # 取区间内最保守的（阈值最小 = 抓得最多 = 最不容易漏）
        #
        # ★ 为什么取"抓得最多"而不是"最接近区间中点"？
        #   在风险筛查场景里，**漏报的代价高于误报** ——
        #   误报只是让人多看一眼，漏报可能让有问题的改动直接合并。
        #   这个原则和 docs/06 里记忆召回阈值的选择是同一个。
        pick = min(in_range, key=lambda r: r["threshold"])
        print(f"""
  {C['green']}✓ 推荐阈值：{pick['threshold']} 行{C['reset']}
     高风险 {pick['n_high']}/{n_all}（{pick['rate']:.1%}）
     已合并中有 {pick['merged_high']} 个，未合并中有 {pick['unmerged_high']} 个

  {C['dim']}  取值理由：落在目标区间 {lo:.0%}~{hi:.0%} 内，且取**最小的那个**
    （抓得最多 = 最不容易漏）。因为风险筛查里**漏报代价 > 误报代价**：
    误报只是让人多看一眼，漏报可能让有问题的改动直接合并。{C['reset']}""")
        recommend_th = pick["threshold"]
    else:
        # 没有候选落在区间内 → 用分位数反推
        if da:
            recommend_th = da["p90"]
            print(f"""
  {C['yellow']}⚠ 没有候选阈值正好落在目标区间，改用 p90 分位数{C['reset']}
     推荐阈值：{recommend_th} 行（p90）
     {C['dim']}即"只有改动量最大的 10% 被判高风险" ——
       这个定义下"高风险"天然是少数派，有区分度。{C['reset']}""")
        else:
            recommend_th = 500

    # 和当前值对比
    from nightwatch.tools.pr_review import review_pr
    print(f"""
  {C['bold']}和当前默认值（500 行）对比{C['reset']}
    · 当前 500 行 → 高风险 {sum(1 for c in all_churn if c > 500)}/{n_all}
      （{sum(1 for c in all_churn if c > 500) / n_all:.1%}）
    · 推荐 {recommend_th} 行 → 高风险 {pick['n_high'] if in_range else sum(1 for c in all_churn if c > recommend_th)}/{n_all}

  {C['dim']}  500 行这个值是拍脑袋定的（"看起来合理"）。
    实测真实 PR 的改动量中位只有 {da['p50'] if da else '?'} 行 ——
    500 行意味着**只有极少数 PR 会触发这条规则**。{C['reset']}""")

    # ---------------------------------------------------------- 输出
    out = Path(args.json) if args.json else \
        ROOT / "data" / "pr_threshold_calibration.json"
    payload = {
        "n_merged": len(m_churn), "n_unmerged": len(u_churn),
        "dist_merged": dm, "dist_unmerged": du, "dist_all": da,
        "signal_valid": (abs(gap_p50) / max(du["p50"], 1) >= 0.5
                         and gap_p50 > 0
                         if dm and du else None),
        "sweep": rows,
        "target_range": [lo, hi],
        "recommended_threshold": recommend_th,
        "caveats": [
            f"样本量较小（已合并 {len(m_churn)} / 未合并 {len(u_churn)}）",
            "只用了改动量这一个信号；敏感文件命中数未纳入标定",
            "不同仓库的改动量分布差异很大，跨仓库要重新标定",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {C['green']}✓{C['reset']} JSON → {out}")

    hr()
    print(f"""
  {C['bold']}怎么用这个结果{C['reset']}
    把 `tools/pr_review.py` 里 `review_pr(max_high_risk_churn=500)`
    的默认值改成 **{recommend_th}**，并同步更新工具描述里的说明。

    {C['dim']}★ 关键不是"改成 {recommend_th}"，而是"它现在有数据支撑了"。
      面试时能回答"这个值怎么来的"，而不是"经验值"。{C['reset']}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
