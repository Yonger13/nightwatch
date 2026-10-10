"""阈值扫描：为 duplicate 判定标定相似度阈值。

================================================================================
为什么"标定"这件事必须做，而且必须留下记录
================================================================================
面试时几乎一定会被问："你这个阈值怎么定的？"

    ✗ 差的回答："经验值 0.75"            → 追问一句就崩
    ✓ 好的回答："我扫了 0.1~0.9，看 duplicate 检测的
                precision/recall 曲线，取了两类错误代价的平衡点。"
                然后能拿出这张表。

这个脚本就是生成那张表的。跑出来的结果会写进 data/_thr_sweep.txt，
文档里引用它。

⚠️ 重要：本脚本用的是**合成数据 + 哈希伪向量**。
   换成真实数据/真实 embedding 后，相似度分布完全不同，**必须重新跑这个脚本**。
   这也是为什么要把标定过程脚本化 —— 让"重新标定"是一件 30 秒的事。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import re  # noqa: E402

from nightwatch.eval.dataset import build_synthetic_dataset      # noqa: E402
from nightwatch.eval.significance import bonferroni_note         # noqa: E402
from nightwatch.memory.store import HashEmbedding, cosine         # noqa: E402


def make_query(issue) -> str:
    """复现系统里"query 是怎么构造的"。

    必须和 ToolRegistry._search_similar 收到的 query 同源，
    否则标定出来的阈值在实际运行时会失效。

    实际运行时的 query 来自 MockLLM._extract_symptom：
        标题 + 从【当前 issue 的堆栈】里抽的异常类型
    """
    exc = ""
    if issue.stack_trace:
        m = re.search(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Fault))\b",
                      issue.stack_trace)
        if m:
            exc = m.group(1)
    return f"{issue.title} {exc}".strip()


def main() -> None:
    ds = build_synthetic_dataset()
    issues = ds.train + ds.holdout

    enc = HashEmbedding()
    lines: list[str] = []
    def p(*a):
        lines.append(" ".join(str(x) for x in a))

    p("=" * 78)
    p("阈值扫描 · duplicate 判定")
    p("=" * 78)
    p(f"样本：{len(issues)} 条  "
      f"(其中 {sum(1 for i in issues if i.gold_is_duplicate_of)} 条为已知重复)")
    p()
    p("说明：对每个 issue，用它的 query 检索所有其他 issue，取相似度最高的一条。")
    p("      如果 gold 标记它是重复，则正确的 top1 应该是它的原型。")
    p()

    # ---------- 逐条计算 top1 ----------
    rows = []
    p("-" * 78)
    p(f"{'issue':<8}{'top1':<8}{'相似度':>8}  {'是否重复':<8}{'top1 正确?':<12}  query")
    p("-" * 78)
    for issue in issues:
        q = make_query(issue)
        qv = enc.encode(q)
        best_id, best_sim = "", 0.0
        for other in issues:
            if other.issue_id == issue.issue_id:
                continue
            ov = enc.encode(f"{other.title} {other.body[:300]}")
            s = cosine(qv, ov)
            if s > best_sim:
                best_id, best_sim = other.issue_id, s

        is_dup = issue.gold_is_duplicate_of is not None
        correct = (best_id == issue.gold_is_duplicate_of) if is_dup else None
        rows.append((issue.issue_id, best_id, best_sim, is_dup, correct))
        flag = "—" if correct is None else ("✓" if correct else "✗")
        p(f"{issue.issue_id:<8}{best_id:<8}{best_sim:>8.4f}  "
          f"{'是' if is_dup else '否':<8}{flag:<12}  {q[:40]}")

    # ---------- 两组分布对比 ----------
    p()
    p("-" * 78)
    p("两组相似度分布（这是阈值可不可分的直接证据）")
    p("-" * 78)
    dup_sims = [r[2] for r in rows if r[3]]
    non_sims = [r[2] for r in rows if not r[3]]
    if dup_sims:
        p(f"  重复样本的 top1 相似度: "
          f"min={min(dup_sims):.4f}  max={max(dup_sims):.4f}  "
          f"({len(dup_sims)} 条)")
    if non_sims:
        p(f"  非重复样本 top1 相似度: "
          f"min={min(non_sims):.4f}  max={max(non_sims):.4f}  "
          f"({len(non_sims)} 条)")
    if dup_sims and non_sims:
        gap_lo, gap_hi = max(non_sims), min(dup_sims)
        p()
        if gap_lo < gap_hi:
            p(f"  → 两组之间有清晰间隔: ({gap_lo:.4f}, {gap_hi:.4f})")
            p(f"  → 建议阈值取间隔中点或偏保守位置: "
              f"{(gap_lo + gap_hi) / 2:.2f} ~ {gap_hi:.2f}")
        else:
            p(f"  → ⚠ 两组重叠（非重复最高 {gap_lo:.4f} ≥ "
              f"重复最低 {gap_hi:.4f}）")
            p("     说明当前向量模型不足以分开这两组，任何阈值都会有错误。")
            p("     真实场景应改用语义 embedding 后重新扫描。")

    # ---------- 扫阈值 ----------
    p()
    p("-" * 78)
    p("阈值扫描：precision / recall / F1")
    p("-" * 78)
    p(f"{'阈值':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}"
      f"{'precision':>12}{'recall':>10}{'F1':>10}")
    p("-" * 78)

    best = (0.0, None)
    n_tried = 0                      # 实际比较了几个阈值（用于多重比较提示）
    for t10 in range(10, 96, 5):
        thr = t10 / 100
        n_tried += 1
        tp = fp = fn = tn = 0
        for _, _, sim, is_dup, _ in rows:
            pred_dup = sim >= thr
            if is_dup and pred_dup:
                tp += 1
            elif is_dup and not pred_dup:
                fn += 1
            elif not is_dup and pred_dup:
                fp += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        mark = ""
        if f1 > best[0]:
            best = (f1, thr)
            mark = "  ←"
        p(f"{thr:>6.2f}{tp:>6}{fp:>6}{fn:>6}{tn:>6}"
          f"{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}{mark}")

    p()
    if best[1] is not None:
        p(f"F1 最优阈值 = {best[1]:.2f} (F1={best[0]:.3f})")
        p()
        p("但**不要直接用 F1 最优的那个值** —— 还要看两类错误的代价：")
        p("  漏报（FN）：维护者重复处理一遍，浪费一点时间")
        p("  误报（FP）：新问题被标记为已解决，**永远不会被处理**")
        p("误报严重得多，所以实际取值应该**比 F1 最优点更偏保守（更高）**。")

        # ================================================================
        # ★★ 多重比较提示（选择偏差）★★
        # ================================================================
        # 上面那个"F1 最优"是**在 n_tried 个阈值里挑出来的最大值** ——
        # 挑最大值这个动作本身就会让结果偏高：
        # 哪怕所有阈值的真实表现完全一样，18 次里也总有一个碰巧最高。
        #
        # 所以报告里必须写清楚两件事：
        #   ① 单次比较的显著性门槛要按比较次数校正（Bonferroni）
        #   ② **校正还不够** —— 它只让步门槛更严，不消除"挑出来的那个偏高"
        #      这个事实。真正要做的是把最优阈值拿到**独立数据**上复验。
        p()
        for line in bonferroni_note(n_tried).split("\n"):
            p(line)
        p()
        p("  本脚本的适用场景正是「**筛选候选**」—— 它的产出是一个待验证的")
        p("  候选阈值，而不是可以直接上线的配置。")
    p()
    p("=" * 78)
    p("⚠ 本结果基于合成数据 + HashEmbedding（伪向量）。")
    p("  真实数据/真实 embedding 的相似度分布完全不同，必须重新扫描。")
    p("=" * 78)

    out = ROOT / "data" / "_thr_sweep.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
