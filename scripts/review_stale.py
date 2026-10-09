#!/usr/bin/env python3
"""
Nightwatch · 过时记忆复核 CLI

    # 看待复核队列（按优先级排序）
    python scripts/review_stale.py --list

    # 手动触发检测：给一批被改动的文件
    python scripts/review_stale.py --detect pandas/core/frame.py pandas/io/csv.py

    # 裁决：确认过时（会降权归档） / 判定仍有效（只记录，不改变记忆）
    python scripts/review_stale.py --confirm sr_abc123 --by 张三 --note "重构后该结论已失效"
    python scripts/review_stale.py --dismiss sr_def456 --by 张三 --note "核对过，仍然有效"

    # 统计与历史
    python scripts/review_stale.py --stats
    python scripts/review_stale.py --history

==============================================================================
这个 CLI 是"人工闸门"的落地形式
==============================================================================
`stale.py` 把检测和入队都自动化了，但**生效必须经过人**。
而"必须经过人"要有具体的形式 —— 否则它就只是一句设计口号。

这个 CLI 就是那个形式：**一个可以真正跑起来、有幂等、有审计记录的审核流程。**

==============================================================================
★ 为什么 `--dismiss` 和 `--confirm` 一样重要
==============================================================================
`--confirm`（确认过时）会改变记忆（降权 + 归档）。
`--dismiss`（判定仍有效）**不改变任何记忆**。

但它**同样会被记录**，因为它回答了一个有价值的问题：

    "这条被怀疑过的记忆，人看过之后认为没问题。"

记录它的收益：
  · 下次**同样的变更**再触发时，直接跳过（不重复问人）
  · 累计起来能回答"我们的检测准确率有多少"（dismiss 率高 → 检测太宽）

**只记"确认"不记"否决"，等于强迫人做重复劳动**，
而且丢掉了"检测质量"这个可优化的指标。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.memory.stale import StaleDetector                   # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore      # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def open_store(db: Path) -> MemoryStore:
    return MemoryStore(str(db), encoder=HashEmbedding(dim=256))


def show_pending(det: StaleDetector) -> int:
    hr("待复核队列")
    items = det.pending()
    if not items:
        print(f"  {C['dim']}（空）{C['reset']}")
        print(f"""
  {C['dim']}队列空有两种情况：
    ① 记忆库里没有带 code_refs 的记忆（比如都是老数据）
    ② 确实没有变更波及到任何记忆
    用 --stats 看一下 code_refs 覆盖率可以区分。{C['reset']}""")
        return 0

    print(f"\n  {len(items)} 条待复核\n")
    for it in items:
        print(f"  {C['bold']}{it.review_id}{C['reset']}　"
              f"{C['dim']}{it.created_at}{C['reset']}")
        print(f"    记忆      {it.memory_id}")
        print(f"    症状      {it.symptom[:64]}")
        print(f"    命中路径  {', '.join(it.matched_refs)}")
        print(f"    触发      {it.trigger}（{it.trigger_detail[:50]}）")
        print()
    print(f"  {C['dim']}裁决：--confirm <id> 确认过时 / --dismiss <id> 判定仍有效"
          f"（都需 --by）{C['reset']}")
    return len(items)


def show_history(det: StaleDetector) -> None:
    hr("复核历史")
    items = det.history(limit=30)
    if not items:
        print(f"  {C['dim']}（无）{C['reset']}")
        return
    print(f"\n  {'review_id':<14}{'状态':<12}{'记忆':<10}审核人　备注")
    print(f"  {'-' * 70}")
    for it in items:
        col = {"confirmed": C["red"], "dismissed": C["green"],
               "pending": C["yellow"]}.get(it.status, "")
        mark = {"confirmed": "确认过时", "dismissed": "仍有效",
                "pending": "待复核"}.get(it.status, it.status)
        print(f"  {it.review_id:<14}{col}{mark:<12}{C['reset']}"
              f"{it.memory_id:<10}{it.reviewed_by or '-':<8}"
              f"{it.review_note[:30]}")


def show_stats(det: StaleDetector, store: MemoryStore) -> None:
    hr("统计")
    st = det.stats()
    print(f"""
  记忆总数        {st.get('n_memories', 0)}
  待复核          {st.get('pending') or 0}
  确认过时        {st.get('confirmed') or 0}
  判定仍有效      {st.get('dismissed') or 0}
  受影响比例      {(st.get('affected_ratio') or 0):.1%}""")

    # ★ 覆盖率是判断"这个机制有没有用"的关键数字
    rows = store.all_memories()
    with_refs = sum(1 for m in rows if getattr(m, "code_refs", None))
    with_head = sum(1 for m in rows if getattr(m, "repo_head", ""))
    print(f"""
  {C['bold']}地基覆盖率（这个机制能不能起作用的前提）{C['reset']}
    有 code_refs 的记忆   {with_refs}/{len(rows)}
                          （{with_refs / max(len(rows), 1):.0%}）
    有 repo_head 的记忆   {with_head}/{len(rows)}""")
    if rows and with_refs == 0:
        print(f"""
  {C['yellow']}⚠ code_refs 覆盖率为 0 —— 检测逻辑目前不会命中任何东西{C['reset']}
    {C['dim']}原因：老数据的 code_refs 是空的（字段是后加的，且读路径容错）。
      只有**在本次改动之后新写入的记忆**才会带上 code_refs。
      这不是 bug，是迁移的必然状态 —— 覆盖率会随新数据积累而上升。{C['reset']}""")

    # dismiss 率 → 检测质量
    reviewed = (st.get("confirmed") or 0) + (st.get("dismissed") or 0)
    if reviewed:
        confirm_rate = (st.get("confirmed") or 0) / reviewed
        print(f"""
  {C['bold']}检测质量（人工裁决后的反馈）{C['reset']}
    已裁决 {reviewed} 条，其中确认过时 {st.get('confirmed') or 0} 条
    确认率 {confirm_rate:.0%}

    {C['dim']}★ 这个数字衡量的是"检测准不准"：
      · 确认率过低（大量 dismiss）→ 检测太宽，把无关的也抓进来了，浪费人的时间
      · 确认率过高（几乎全 confirm）→ 可能检测太窄，漏了很多
      它和记忆召回的"无效召回率"是同一类指标：**衡量过滤器的质量**。{C['reset']}""")


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="过时记忆复核")
    ap.add_argument("--db", default=str(ROOT / "data" / "memory.db"),
                    help="记忆库路径")
    ap.add_argument("--list", action="store_true", help="列出待复核队列")
    ap.add_argument("--history", action="store_true", help="看复核历史")
    ap.add_argument("--stats", action="store_true", help="统计")
    ap.add_argument("--detect", nargs="+", metavar="FILE",
                    help="手动触发检测：给一批被改动的文件路径")
    ap.add_argument("--trigger", default="manual",
                    help="触发来源标签（配 --detect 用，用于幂等去重）")
    ap.add_argument("--confirm", metavar="ID", help="确认过时（降权归档）")
    ap.add_argument("--dismiss", metavar="ID", help="判定仍有效（只记录）")
    ap.add_argument("--by", default="", help="审核人（必填）")
    ap.add_argument("--note", default="", help="备注")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"{C['red']}✗ 记忆库不存在：{db}{C['reset']}")
        print(f"  换个路径：--db <你的库>")
        return 1

    store = open_store(db)
    det = StaleDetector(store)

    # ---------------------------------------------------------- 检测
    if args.detect:
        hr("触发检测")
        changed = list(args.detect)
        print(f"  被改动的文件：{len(changed)} 个")
        for f in changed[:8]:
            print(f"    · {f}")
        suspects = det.detect(changed)
        print(f"\n  命中 {len(suspects)} 条嫌疑记忆")
        if not suspects:
            print(f"""
  {C['dim']}没命中可能有两个原因：
    ① 确实没有记忆引用了这些文件（正常）
    ② 记忆的 code_refs 是空的（老数据）—— 用 --stats 看覆盖率{ C['reset']}""")
            return 0
        for s in suspects[:14]:
            print(f"    {s.memory_id:<28} 优先级 {s.priority:>5.2f}　"
                  f"命中 {len(s.matched_refs)} 个路径"
                  + ("　← 人工确认过" if s.human_verdict in ("accepted", "edited")
                     else ""))
        created = det.mark(suspects, trigger=args.trigger,
                           trigger_detail=", ".join(changed[:5]))
        print(f"\n  {C['green']}✓ 已入队 {len(created)} 条"
              f"（{len(suspects) - len(created)} 条已处理过，跳过）{C['reset']}")
        print(f"  {C['dim']}入队**不改变记忆** —— 需要人用 --confirm 裁决才生效。"
              f"{C['reset']}")
        return 0

    # ---------------------------------------------------------- 裁决
    if args.confirm or args.dismiss:
        if not args.by:
            print(f"{C['red']}✗ --by 是必填的（审核要可追溯）{C['reset']}")
            return 1
        rid = args.confirm or args.dismiss
        confirmed = bool(args.confirm)
        hr("裁决")
        try:
            r = det.review(rid, confirmed=confirmed, reviewer=args.by,
                           note=args.note)
        except KeyError as e:
            print(f"  {C['red']}✗ {e}{C['reset']}")
            return 1
        if confirmed:
            before = r.get("confidence_before", 0)
            after = r.get("confidence_after", 0)
            print(f"""
  {C['red']}已确认过时{C['reset']}　{rid}
    记忆          {r['memory_id']}
    confidence    {before:.2f} → {after:.2f}
    已归档（软纠正，不物理删除）

  {C['dim']}  这条记忆不会再被召回。但它**还在库里** ——
    保留是为了能回答"当初是哪些判断引用了它"。{C['reset']}""")
        else:
            print(f"""
  {C['green']}已判定仍有效{C['reset']}　{rid}
    记忆 {r['memory_id']} **未做任何改动**
    {C['dim']}这条记录的价值：下次同样的变更再触发时直接跳过，
      而且它累积起来可以算"检测准确率"。{C['reset']}""")
        return 0

    # ---------------------------------------------------------- 查看
    if args.list:
        show_pending(det)
        return 0
    if args.history:
        show_history(det)
        return 0
    if args.stats:
        show_stats(det, store)
        return 0

    # 默认：状态概览
    st = det.stats()
    print(f"\n{C['bold']}{C['cyan']}过时记忆复核{C['reset']}")
    hr()
    print(f"""
  记忆库        {db}
  待复核        {st.get('pending') or 0}
  已确认过时    {st.get('confirmed') or 0}
  判定仍有效    {st.get('dismissed') or 0}""")
    print(f"""
  常用命令：
    python scripts/review_stale.py --list                # 看待复核队列
    python scripts/review_stale.py --stats               # 看覆盖率与检测质量
    python scripts/review_stale.py --detect <文件...>    # 手动触发检测
    python scripts/review_stale.py --confirm <id> --by <人>
    python scripts/review_stale.py --dismiss <id> --by <人>
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
