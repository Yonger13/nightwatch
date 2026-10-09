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
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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

    # ---------------------------------------------------------- 分类对比
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
""")

    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
