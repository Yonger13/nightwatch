#!/usr/bin/env python3
"""
Nightwatch · 前瞻验证 · 修复前后对比

    python scripts/compare_prospective.py
    python scripts/compare_prospective.py --before data/prospective_eval.json \\
                                          --after  data/prospective_eval_v2.json

==============================================================================
这个脚本要产出什么
==============================================================================
一次完整的**"发现问题 → 定位根因 → 修复 → 验证"**闭环的证据：

    修复前（v1）：Build 类命中率 20%
    修复后（v2）：Build 类命中率 100%

**不只是一个"变好了"的结论**，而是逐条对比：
哪几条从错变对、哪几条没变、有没有**变差的**。

==============================================================================
★ 为什么一定要检查"有没有变差的"
==============================================================================
改提示词是一种**全局干预** —— 它可能修好 A 类的同时弄坏 B 类。

只报"总体从 93% 提升到 100%"是不负责任的：
如果某个类从 100% 掉到 80%，总体可能还是提升的，**但问题被掩盖了**。

所以要**逐类对比**，而不是只看总数。

（这也是 `docs/05` 里"多口径"那条教训的同构：
  单一汇总数字会掩盖结构。）

==============================================================================
★ 但"逐条看过了"仍然不够 —— 还差一个显著性检验
==============================================================================
上面那些做法解决的是"提升是怎么来的"，没有解决"**提升是不是真的**"。

61 条样本里翻转 4 条，听上去很清楚；但 4 条翻转完全可能是随机波动。
不检验的话，"提升"这两个字就没有依据 ——
而**没有依据的结论比没有结论更危险**，因为它会被当成既成事实继续引用。

所以本脚本在逐条对比之后还会跑一次配对检验（见 `significance.py`），
并**同时给两个口径**：

    连续口径（每条样本的 F1）→ paired bootstrap：给"提升多少"和置信区间
    二元口径（命中 / 未命中）→ McNemar：给"翻转是否对称"

两个口径的结论**可能不一致**，而这种不一致本身就是信息 ——
脚本会按实际数据生成解读，而不是套一段固定文案
（固定文案会在某种组合下说反话，而读者不会去核对）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.eval.metrics import Metrics                            # noqa: E402
from nightwatch.eval.significance import (                             # noqa: E402
    mcnemar, paired_bootstrap,
)

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def load(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {r["issue_id"]: r for r in d.get("results", [])}


def rate(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, 0.0
    h = sum(1 for r in rows if r["hit"])
    return h, len(rows), h / len(rows)


def f1_change_buckets(before: dict, after: dict,
                      ids: list) -> tuple[list, list, int]:
    """按 **F1 口径**把样本分成 (提升, 下降, 无变化)。

    ========================================================================
    ★ 为什么需要一个单独的函数，而不是就地写在 main() 里
    ========================================================================
    两个理由：

      ① **可测**。这条逻辑要守住的东西很具体 —— "命中口径看不见的回归，
         F1 口径要看得见" —— 而这类判断一旦错了是**静默**的：
         报告照样生成、数字照样好看，只是漏报了一条降级。
         所以它必须能被单元测试直接调用。

      ② **单一实现**。第三项统计（结论段的"两个口径各自的变差数"）
         和明细列表用的是同一份判断。就地写两遍 = 又一个"同一件事
         两份实现"（这个项目已经踩过四次）。

    参数
    ----
    before, after : `{issue_id: 记录}`，记录里要有 `predicted` / `gold`
    ids           : 要比对的 issue id 列表（通常是两版的交集）

    返回
    ----
    `(gains, drops, n_same)`
      gains / drops 是 `[(issue_id, f1_before, f1_after), ...]`

    ========================================================================
    为什么阈值用 1e-9 而不是 != 0
    ========================================================================
    F1 是浮点算出来的（分子分母都是整数，但会做除法），
    理论上相等的两个值可能在末位有差异。用 `abs(delta) < 1e-9` 判"无变化"，
    几率的量级上足够安全，同时不会把真实的微小变化误判成相等。

    ========================================================================
    为什么不用集合严格相等来判断"有没有变差"
    ========================================================================
    因为**"多打了标签"型回归恰恰是 Top-1 仍然正确的那种** ——
    用集合相等（`exact`）会把"多打一个标签"和"少打一个标签"都算成变化，
    而我们要单独盯住的是**下降**（F1 变小），不是"变了"。
    """
    gains, drops, n_same = [], [], 0
    for i in ids:
        fb = Metrics.label_f1(before[i]["predicted"], before[i]["gold"])[2]
        fa = Metrics.label_f1(after[i]["predicted"], after[i]["gold"])[2]
        delta = fa - fb
        if abs(delta) < 1e-9:
            n_same += 1
        elif delta > 0:
            gains.append((i, fb, fa))
        else:
            drops.append((i, fb, fa))
    return gains, drops, n_same


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="前瞻验证前后对比")
    ap.add_argument("--before", default=str(ROOT / "data" / "prospective_eval.json"))
    ap.add_argument("--after", default=str(ROOT / "data" / "prospective_eval_v2.json"))
    args = ap.parse_args()

    b_path, a_path = Path(args.before), Path(args.after)
    for p in (b_path, a_path):
        if not p.exists():
            print(f"{C['red']}✗ 找不到 {p}{C['reset']}")
            return 1

    before, after = load(b_path), load(a_path)

    print(f"\n{C['bold']}{C['cyan']}前瞻验证 · 修复前后对比{C['reset']}")
    hr()

    # 只看两边都有的
    common = sorted(set(before) & set(after))
    if not common:
        print(f"{C['red']}✗ 两次结果没有共同的 issue{C['reset']}")
        return 1

    b_rows = [before[i] for i in common]
    a_rows = [after[i] for i in common]

    bh, bn, br = rate(b_rows)
    ah, an, ar = rate(a_rows)

    print(f"""
  样本（两边都有）  {bn}
  修复前命中率      {bh}/{bn} = {C['yellow']}{br:.1%}{C['reset']}
  修复后命中率      {ah}/{an} = {C['green']}{ar:.1%}{C['reset']}
  变化              {C['bold']}{(ah - bh):+d} 条{(ar - br):+.1%}{C['reset']}""")

    # ---------------------------------------------------------- 逐条变化
    hr("逐条变化")

    fixed, broke, same_ok, same_bad = [], [], [], []
    for i in common:
        b, a = before[i]["hit"], after[i]["hit"]
        if not b and a:
            fixed.append(i)
        elif b and not a:
            broke.append(i)
        elif b and a:
            same_ok.append(i)
        else:
            same_bad.append(i)

    print(f"""
  {C['green']}修好了（✗ → ✓）{C['reset']}   {len(fixed)} 条
  {C['red']}变差了（✓ → ✗）{C['reset']}   {len(broke)} 条
  一直正确             {len(same_ok)} 条
  一直错误             {len(same_bad)} 条""")

    if fixed:
        print(f"\n  {C['green']}修好的明细（这是修复生效的直接证据）{C['reset']}")
        print(f"\n    {'issue':<10}{'修复前预测':<24}{'修复后预测':<24}真实")
        print(f"    {'-' * 70}")
        for i in fixed:
            print(f"    {i:<10}{str(before[i]['predicted']):<24}"
                  f"{str(after[i]['predicted']):<24}{after[i]['gold'][0]}")

    if broke:
        print(f"\n  {C['red']}⚠ 变差的明细（必须检查是不是修复引入的副作用）{C['reset']}")
        for i in broke:
            print(f"    {i:<10}{str(before[i]['predicted']):<24}"
                  f"{str(after[i]['predicted']):<24}{after[i]['gold'][0]}")

    if same_bad:
        print(f"\n  {C['yellow']}仍然错误的（修复没覆盖到）{C['reset']}")
        for i in same_bad:
            print(f"    {i:<10}{str(after[i]['predicted']):<24}"
                  f"真实={after[i]['gold'][0]}　{after[i]['title'][:34]}")

    # ---------------------------------------------------------- F1 口径的回归
    #
    # ★★ 上面那段"逐条变化"按 **hit** 比对，有一个盲区 ★★
    #
    # hit 只问"Top-1 对不对"。所以一条样本如果**多打了标签**，
    # 只要 Top-1 仍然正确，hit 就是 True → 落在"一直正确"里 →
    # 归入"没有任何类别变差"。**而它的 F1 其实掉了。**
    #
    # 真实例证（data/prospective_eval.json vs _v2.json 的 #68433）：
    #     v1  ['Enhancement']           F1 = 1.000   hit = True
    #     v2  ['Enhancement', 'Build']  F1 = 0.667   hit = True   ← 多打一个标签
    # hit 口径看不见它；F1 口径看得见。
    #
    # 所以两个口径**都要报各自的"变差数"**，而不是只报一个。
    # 只报 hit 的 0 条会让人以为"修复零副作用"，而事实上有一条降级了 ——
    # 这正是本项目一贯警惕的那类问题：**单一口径会掩盖结构**。
    # 逻辑收在 `f1_change_buckets()` 里（模块级纯函数）——
    # 这样它能被单元测试直接调用，也能被下面结论段复用同一份判断。
    f1_b = [Metrics.label_f1(r["predicted"], r["gold"])[2] for r in b_rows]
    f1_a = [Metrics.label_f1(r["predicted"], r["gold"])[2] for r in a_rows]

    f1_gains, f1_drops, f1_same = f1_change_buckets(before, after, common)

    print(f"\n  {C['bold']}F1 口径的逐条变化（hit 口径看不见「多打标签」型回归）{C['reset']}")
    print(f"""
    F1 提升   （delta > 0）   {len(f1_gains)} 条   合计 {sum(f - b for _, b, f in f1_gains):+.4f}
    F1 下降   （delta < 0）   {len(f1_drops)} 条   合计 {sum(f - b for _, b, f in f1_drops):+.4f}
    无变化                    {f1_same} 条""")

    if f1_drops:
        print(f"\n  {C['red']}⚠ F1 下降的明细（hit 口径看不到，必须逐条检查）{C['reset']}")
        print(f"\n    {'issue':<10}{'F1 前':>8}{'F1 后':>8}{'delta':>9}  "
              f"{'hit':<12}{'预测变化'}")
        print(f"    {'-' * 84}")
        for i, fb, fa in f1_drops:
            hb = "✓" if before[i]["hit"] else "✗"
            ha = "✓" if after[i]["hit"] else "✗"
            note = ("  ← hit 没变，只有 F1 掉了" if hb == ha else "")
            print(f"    {i:<10}{fb:>8.3f}{fa:>8.3f}{fa - fb:>+9.3f}  "
                  f"{hb} → {ha:<8}"
                  f"{str(before[i]['predicted'])} → {str(after[i]['predicted'])}{note}")
    else:
        print(f"\n  {C['green']}✓ F1 口径下也没有任何条目下降{C['reset']}")

    # ---------------------------------------------------------- 显著性检验
    #
    # ★ 上面的"命中率 93.4% → 100%"只说了**提升了多少**，
    #   没说**这个提升是不是真的**。61 条样本里 4 条翻转，
    #   完全可能是随机波动。不检验的话，"提升"这两个字没有依据。
    hr("显著性检验（这个提升是真的吗）")

    # f1_b / f1_a 已在上面算过（F1 口径回归检测要用），这里直接复用 ——
    # 不重算一遍（"同一件事两份实现"是这个项目踩过四次的坑）。
    boot = paired_bootstrap(f1_b, f1_a)
    # b = 修复前对、修复后错（变差的）；c = 修复前错、修复后对（修好的）
    # ★ 这两个数就是上面"逐条变化"里已经数出来的 broke / fixed ——
    #   配对二元检验需要的正是这两个方向的不一致对数，不用另外再算一遍。
    mcn = mcnemar(len(broke), len(fixed))

    print(f"""
  {C['bold']}① 连续口径：每条样本的 F1{C['reset']}
    中位 F1        修复前 {statistics.median(f1_b):.4f}  →  修复后 {statistics.median(f1_a):.4f}
    平均变化       {boot['mean_delta']:+.4f}
    {boot['ci_level']:.0%} 置信区间   [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]
    p 值           {boot['p_value']:.4f}　（{boot['n']} 对样本，重采样 {boot['n_boot']:,} 次）
    结论           {boot['verdict']}

  {C['bold']}② 二元口径：是否命中{C['reset']}
    b（修复前对、修复后错）  {mcn['b']} 条
    c（修复前错、修复后对）  {mcn['c']} 条
    {C['bold']}有效样本量 b+c = {mcn['n_discordant']}{C['reset']}　← 注意不是 {bn}
    p 值           {mcn['p_value']:.4f}
    方法           {mcn['method']}
    结论           {mcn['verdict']}

  {C['bold']}★ 为什么两个口径都要跑，而不是挑一个{C['reset']}

    {C['dim']}因为它们**回答的不是同一个问题**：
      · F1 口径（bootstrap）问"平均**提升多少**"→ 给的是幅度和区间
      · 命中口径（McNemar）问"**翻转是否对称**"→ 给的是方向性证据
      两个结论**不一致也不矛盾**，而且不一致本身往往就是最有信息量的部分。{C['reset']}
""")

    # ------------------------------------------------------- 结果组合的解读
    #
    # ★ 这里刻意**按实际数据生成解读**，而不是写一段固定文案。
    #   因为固定文案会在某种组合下说反话 —— 而读者不会去核对。
    #   （本项目一贯的做法：让报告说的话和数字强绑定。）
    sig_boot = boot["p_value"] < 0.05
    sig_mcn = mcn["p_value"] < 0.05

    # 二元精确检验在 b+c 很小的时候，**理论上不可能显著** ——
    # 双侧最小 p 值 = 2 / 2^(b+c)（所有翻转都朝同一个方向时取到）。
    # 这是检验功效的硬上限，和"有没有效果"无关。
    min_possible_p = 2.0 / (2 ** mcn["n_discordant"]) if mcn["n_discordant"] else 1.0
    power_limited = (not sig_mcn) and min_possible_p >= 0.05

    print(f"  {C['bold']}★ 这两个结论怎么合起来读{C['reset']}")
    print()
    if sig_boot and sig_mcn:
        print(f"    {C['green']}两个口径都显著 → 证据较强{C['reset']}"
              f"（既有方向、也有幅度）")
    elif sig_boot and not sig_mcn:
        if power_limited:
            print(f"    {C['yellow']}F1 口径显著、命中口径不显著 —— "
                  f"但这次是**检验功效不足**，不是'没差异'{C['reset']}")
            print(f"      {C['dim']}b+c = {mcn['n_discordant']} 时，双侧精确检验的最小可能 "
                  f"p 值 = 2/2^{mcn['n_discordant']} = {min_possible_p:.4f} ≥ 0.05 ——")
            print(f"      **在数学上就不可能达到显著**。所以这里的「不显著」"
                  f"完全不含「没有效果」的信息。{C['reset']}")
            print(f"      {C['dim']}诚实写法：「{mcn['n_discordant']} 条翻转全部朝同一方向"
                  f"（{mcn['b']} 条变差、{mcn['c']} 条变好），方向一致；")
            print(f"      但翻转数太少，配对二元检验无法给出显著结论。"
                  f"这里的依据是**方向的一致性**和 F1 的幅度，不是 p 值。」{C['reset']}")
        else:
            print(f"    {C['yellow']}F1 口径显著、命中口径不显著{C['reset']}")
            print(f"      {C['dim']}b+c={mcn['n_discordant']}（达到显著所需的量级），"
                  f"但翻转不够对称 → 说明有涨有跌、净效果靠幅度而非方向。{C['reset']}")
    elif sig_mcn and not sig_boot:
        print(f"    {C['yellow']}命中口径显著、F1 口径不显著{C['reset']}")
        print(f"      {C['dim']}方向一致但区间贴着 0 → 确实在变好，**但幅度很小**，"
              f"改动的性价比需要另议。{C['reset']}")
    else:
        print(f"    {C['red']}两个口径都不显著 → 说「提升」缺乏依据{C['reset']}")
        print(f"      {C['dim']}这和「没有提升」不是一回事：也可能是样本量不够。"
              f"报告里应写成「在 {bn} 条样本上未能检出显著差异」。{C['reset']}")

    print(f"""
  {C['bold']}★ 二元口径的样本量陷阱（最容易被误读的一处）{C['reset']}
    有效样本量是 {C['bold']}b+c = {mcn['n_discordant']}{C['reset']}，不是总样本数 {bn}。
    两次都对的 {len(same_ok)} 条、两次都错的 {len(same_bad)} 条
    **对「差异」没有任何信息**（两边抵消），所以不参与检验。
    报告里必须写 b+c —— 否则读的人会以为这是「{bn} 条样本的证据」，
    而实际只有 {mcn['n_discordant']} 条。

  {C['dim']}★ 还有一条更普遍的纪律：
    **「不显著」≠「没有差异」。** 不显著只说明"这批数据没提供足够证据"，
    可能的原因是样本量不够、效应太小、或检验功效不足 ——
    三者要用不同的方式去回应，混在一起就会得出"改动没用"的错误结论。{C['reset']}
""")

    # 分类对比
    hr("按标签分层对比（检查有没有「修好 A 弄坏 B」）")

    by_gold_b: dict[str, list] = defaultdict(list)
    by_gold_a: dict[str, list] = defaultdict(list)
    for i in common:
        g = before[i]["gold"][0]
        by_gold_b[g].append(before[i])
        by_gold_a[g].append(after[i])

    print(f"\n  {'标签':<14}{'样本':>5}{'修复前':>10}{'修复后':>10}{'变化':>10}")
    print(f"  {'-' * 52}")
    any_regress = False
    for g in sorted(by_gold_b, key=lambda x: -len(by_gold_b[x])):
        _, n1, r1 = rate(by_gold_b[g])
        _, n2, r2 = rate(by_gold_a[g])
        delta = r2 - r1
        col = C["green"] if delta > 0 else (C["red"] if delta < 0 else C["dim"])
        print(f"  {g:<14}{n1:>5}{r1:>9.0%}{r2:>10.0%}{col}{delta:>+10.0%}{C['reset']}")
        if delta < 0:
            any_regress = True

    if any_regress:
        print(f"""
  {C['red']}⚠ 有类别变差了 —— 说明这次提示词改动有副作用{C['reset']}
    {C['dim']}提示词是**全局干预**：修 A 类可能弄坏 B 类。
      必须逐类看，不能只看总数。{C['reset']}""")
    else:
        print(f"""
  {C['green']}✓ 没有任何类别变差 —— 修复是"净改进"{C['reset']}""")

    # ---------------------------------------------------------- 结论
    hr("结论")
    print(f"""
  {C['bold']}一次完整的闭环{C['reset']}

    ① 发现问题    前瞻验证显示 Build 类命中率仅 20%（其他类 100%）
    ② 定位根因    提示词把 `Build` 错放在"细分标签"里，还标注"仅当非常明确时才用"
                  → 顺带发现更深的问题：**两套评测用了不同的 prompt_variant**
    ③ 修复        ① 把 Build 移到主标签　② 加"前缀陷阱"说明（CI:/BLD: → Build）
                  ③ 修评测侧漏掉的 ci/bld 映射　④ 对齐 prompt_variant
    ④ 验证        总体 {br:.1%} → {ar:.1%}（{bh} → {ah} 条）
                  显著性：F1 口径 p = {boot['p_value']:.4f}（{boot['mean_delta']:+.4f}，
                  95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]）；
                  命中口径 b+c = {mcn['n_discordant']}，p = {mcn['p_value']:.4f}

  {C['bold']}两个口径各自的「变差数」（必须分开报）{C['reset']}

    命中口径（Top-1 对不对）    变差 {len(broke)} 条
    F1 口径（标签集合的匹配度）  变差 {len(f1_drops)} 条{'' if len(f1_drops) == len(broke) else '   ← 两个数不一样'}

    {C['dim']}★ 为什么不能只报一个：hit 只问"Top-1 对不对"，
      所以"多打了标签"型回归（Top-1 仍对、F1 下降）它**看不见**。
      只报命中口径的 {len(broke)} 条会让人以为"修复零副作用"，
      而 F1 口径下实际有 {len(f1_drops)} 条降级了。
      **换一个合理口径就露出来的东西，说明它一直在那里。**{C['reset']}

  {C['dim']}★ 第 ② 步里"顺带发现的更深问题"才是这个项目一贯的模式：
    修一个问题时，往往会暴露出更深的一层。{C['reset']}

  {C['bold']}诚实边界{C['reset']}
    · 样本 {bn} 条，只覆盖 pandas 一个仓库
    · **100% 命中率本身就是可疑信号** —— 按 docs/05 的第 1 条原则，
      先怀疑"样本是不是太简单"，而不是先庆祝。
      这里的原因是：这 61 条是"已被维护者快速分诊"的 issue（延迟中位 0 天），
      天然偏简单；31% 一直没被打标签的 issue **根本不在样本里**。
    · 已知偏差：`prepare_dataset.py` 的 gold 从标题前缀推，
      和真实标签存在系统性差异（见 4.1.1）
    · **显著性检验的证据强度受限于翻转数**：本轮只有 {mcn['n_discordant']} 条翻转，
      配对二元检验的功效上限就是 p ≥ {min_possible_p:.4f}。
      也就是说这个口径**在本轮数据上无法给出显著结论** ——
      这是数据量的问题，不是"效果不存在"。要真正把这个问题关掉，
      需要的是**更多翻转型样本**（扩大样本或改动幅度更大），而不是换检验方法。
    · bootstrap 的 p 值依赖重采样次数与随机种子（本轮 n_boot={boot['n_boot']:,}，
      seed={boot['seed']}）—— 换种子会上下来回，
      所以它适合回答"是否显著"，不适合当作精确的 p 值读。
""")

    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
