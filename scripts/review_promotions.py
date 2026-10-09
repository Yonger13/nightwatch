#!/usr/bin/env python3
"""
Nightwatch · SOP 提炼提案的审核工具

    # 看当前待审核队列
    python scripts/review_promotions.py --db data/memory.db --list

    # 先扫描出新的模式，再列出（相当于"跑一次提炼"）
    python scripts/review_promotions.py --db data/memory.db --scan

    # 批准
    python scripts/review_promotions.py --db data/memory.db \
        --approve promo_abc123 --by zhangsan --note "根因一致，值得固化"

    # 拒绝（★ 必须给理由）
    python scripts/review_promotions.py --db data/memory.db \
        --reject promo_def456 --by zhangsan --note "样本根因不一致，需要更多数据"

    # 看审核历史（含已批准的 SOP）
    python scripts/review_promotions.py --db data/memory.db --history

==============================================================================
★ 这个 CLI 的三个设计决定
==============================================================================
① **拒绝必须给理由**（`--note` 在 reject 时是必填的）

   因为同一个模式以后可能又被扫描出来（add_promotion 只对 pending 幂等）。
   审核人看到"上次谁在什么时候拒过、为什么拒"，就不用重复判断。

   **如果拒绝不需要理由，"审核"就退化成了"点一下按钮"** ——
   三个月后没人知道当初为什么拒。

② **`--by` 必填**

   审核是一个**人为决定**，必须能追溯到人。
   没有署名的审核记录，等于没有记录。

③ **审核后不能改**（要改只能重新走流程）

   这不是 CLI 的限制，是底层 `review_promotion` 的约束。
   **允许反复改的话，你就无法回答"这条 SOP 到底是谁批的"。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.llm.client import build_llm                            # noqa: E402
from nightwatch.memory.lifecycle import SemanticPromoter               # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore         # noqa: E402
from nightwatch.models import MemoryTier                               # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(w: int = 76) -> None:
    print(f"{C['dim']}{'─' * w}{C['reset']}")


def show_pending(promoter: SemanticPromoter) -> int:
    """展示待审核队列。返回条数（方便做退出码）。"""
    pending = promoter.pending()
    if not pending:
        print(f"  {C['green']}✓{C['reset']} 待审核队列为空")
        return 0

    print(f"  待审核 {C['bold']}{len(pending)}{C['reset']} 条：\n")
    for i, p in enumerate(pending, 1):
        print(f"  {C['bold']}[{i}]{C['reset']} {p.promotion_id}")
        print(f"      {p.sample_summary}")
        print(f"      {C['dim']}创建于 {p.created_at}，"
              f"样本 {len(p.sample_ids)} 条{C['reset']}")
        print()
    print(f"  {C['dim']}批准：--approve <id> --by <你的名字> [--note ...]")
    print(f"  拒绝：--reject  <id> --by <你的名字> --note <理由>（必填）{C['reset']}")
    return len(pending)


def show_history(store: MemoryStore) -> None:
    """展示审核历史（含被拒的）。"""
    all_p = store.all_promotions()
    if not all_p:
        print("  （还没有任何提案）")
        return

    stat = {"pending": 0, "approved": 0, "rejected": 0}
    for p in all_p:
        stat[p.status] = stat.get(p.status, 0) + 1

    print(f"  共 {C['bold']}{len(all_p)}{C['reset']} 条提案　"
          f"待审 {stat['pending']}　"
          f"{C['green']}已批准 {stat['approved']}{C['reset']}　"
          f"{C['red']}已拒绝 {stat['rejected']}{C['reset']}\n")

    for p in all_p:
        color = {"approved": C["green"], "rejected": C["red"],
                 "pending": C["yellow"]}[p.status]
        print(f"  {p.promotion_id:<26} {color}{p.status:<9}{C['reset']}"
              f" {p.pattern}")
        if p.reviewed_by:
            print(f"      by {p.reviewed_by} at {p.reviewed_at}")
        if p.review_note:
            print(f"      {C['dim']}备注：{p.review_note}{C['reset']}")
        if p.promoted_memory_id:
            print(f"      {C['dim']}→ 生成的语义记忆：{p.promoted_memory_id}{C['reset']}")
        print()

    # 语义层现状 —— 审核的最终产出
    n_sem = store.count(MemoryTier.SEMANTIC)
    print(f"  语义层现有 {C['bold']}{n_sem}{C['reset']} 条 SOP"
          f"　{C['dim']}（这是所有「批准」动作的累积结果）{C['reset']}")


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(
        description="SOP 提炼提案的审核工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("====")[0])
    ap.add_argument("--db", default=str(ROOT / "data" / "memory.db"),
                    help="记忆库路径（默认 data/memory.db）")
    ap.add_argument("--list", action="store_true", help="列出待审核队列")
    ap.add_argument("--scan", action="store_true",
                    help="先扫描高频模式，再列出待审核队列")
    ap.add_argument("--history", action="store_true",
                    help="展示全部提案与审核历史")
    ap.add_argument("--approve", metavar="ID", help="批准指定提案")
    ap.add_argument("--reject", metavar="ID", help="拒绝指定提案（必须给 --note）")
    ap.add_argument("--by", default="", help="审核人（必填，用于追溯）")
    ap.add_argument("--note", default="", help="审核备注（拒绝时必填）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists() and not args.scan:
        print(f"{C['red']}✗ 记忆库不存在：{db}{C['reset']}")
        print(f"  提示：先跑一次 --scan，或用 --db 指定别的路径")
        return 1

    db.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db, encoder=HashEmbedding(dim=256))
    promoter = SemanticPromoter(build_llm(use_mock=True), store)

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · SOP 审核{C['reset']}")
    print(f"  记忆库 {db}")
    hr()

    # --- 扫描 ---
    if args.scan:
        print(f"\n{C['bold']}扫描高频模式{C['reset']}")
        created = promoter.scan()
        if created:
            print(f"  {C['green']}✓{C['reset']} 新登记 {len(created)} 条提案：")
            for p in created:
                print(f"      · {p.pattern}　出现 {p.occurrences} 次")
        else:
            print(f"  · 没有新提案（要么模式不够高频，要么已在队列里）")
        print()

    # --- 审核动作 ---
    if args.approve or args.reject:
        target_id = args.approve or args.reject
        approved = bool(args.approve)

        # ★ 必填校验 —— 放在真正执行之前
        if not args.by:
            print(f"  {C['red']}✗ 缺少 --by（审核人）{C['reset']}")
            print(f"    审核是人为决定，必须能追溯到人。")
            return 2
        if not approved and not args.note:
            print(f"  {C['red']}✗ 拒绝时必须给 --note（理由）{C['reset']}")
            print(f"    理由会被下次审核参考 —— 不需要理由的审核，")
            print(f"    三个月后没人知道当初为什么拒。")
            return 2

        p = store.get_promotion(target_id)
        if p is None:
            print(f"  {C['red']}✗ 找不到提案：{target_id}{C['reset']}")
            return 1
        if not p.is_pending:
            print(f"  {C['red']}✗ 提案 {target_id} 已经审核过了"
                  f"（status={p.status}, by {p.reviewed_by}）{C['reset']}")
            print(f"    审核是不可逆的人为决定。要改结论请重新走流程。")
            return 1

        action = "批准" if approved else "拒绝"
        color = C["green"] if approved else C["red"]
        print(f"\n  {color}{action}{C['reset']} {target_id}")
        print(f"    模式    {p.pattern}（出现 {p.occurrences} 次）")
        print(f"    审核人  {args.by}")
        if args.note:
            print(f"    备注    {args.note}")

        sop = promoter.review(target_id, approved=approved,
                              reviewer=args.by, note=args.note)

        if approved:
            if sop:
                print(f"\n  {C['green']}✓{C['reset']} 已生成语义记忆 "
                      f"{C['bold']}{sop.memory_id}{C['reset']}")
                print(f"      {sop.symptom}")
                print(f"      根因：{sop.root_cause[:60]}")
                print(f"      处置：{sop.resolution[:60]}")
            else:
                print(f"\n  {C['yellow']}⚠ 批准了，但没能生成 SOP"
                      f"（样本里没有可提炼的根因/处置）{C['reset']}")
        else:
            print(f"\n  {C['green']}✓{C['reset']} 已拒绝并记录理由"
                  f"（该模式**不会**进入语义层）")

        hr()
        print(f"  语义层现有 {store.count(MemoryTier.SEMANTIC)} 条 SOP")
        return 0

    # --- 历史 ---
    if args.history:
        show_history(store)
        return 0

    # --- 默认：列出待审核 ---
    n = show_pending(promoter)
    print()
    hr()
    print(f"  语义层现有 {store.count(MemoryTier.SEMANTIC)} 条 SOP")
    if n:
        print(f"  {C['dim']}有 {n} 条待审核 —— 它们**还没**进入语义层。"
              f"这就是人工闸门。{C['reset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
