#!/usr/bin/env python3
"""
Nightwatch · 前瞻验证 · 数据采集（轨道 A：历史回溯）

    # 采集：创建于 14~28 天前、且已经有类型标签的 issue
    python scripts/collect_prospective.py --repo pandas-dev/pandas --limit 80

    # 只看统计
    python scripts/collect_prospective.py --limit 80 --stats

==============================================================================
这个脚本和 `collect_issues.py` 的本质区别
==============================================================================
`collect_issues.py` 采的是**已关闭**的 issue —— 那是"事后回顾"，
gold 标签是**已经发生的事实**，用它评测有个天然弱点：

    "你的指标是算在历史数据上的 —— 怎么知道线上真的有用？"

**本脚本要解决这个问题**：它模拟"前瞻预测"的严格性 ——

    输入：issue 创建时的信息（标题 + 正文）
    输出：维护者后来打上的**类型标签**

关键在**时间顺序**：
    先用创建时的信息做预测 → 再对照后来真实发生的标签

**这和"拿 gold 标签算 F1"的区别是根本性的**：
前者是"预测未来"，后者是"解释过去"。

==============================================================================
★ 三条严格的采集约束（这是本脚本的核心价值）
==============================================================================
① **只用创建时的信息**
   · 不做任何"补全"（不猜后续对话、不看别人的评论）
   · **诚实边界**：GitHub API 不提供 issue body 的编辑历史，
     所以如果作者事后编辑过正文，我这里用的会是编辑后的版本。
     缓解办法：timeline 里有 `body_edited` 事件，标记出来（见下）。

② **只对照"首次打上的类型标签"**
   · 维护者后来可能追加标签（尤其**几个月后**发现了新信息）
   · 用"首次"才能保证标签是**基于当时的信息**打的

③ **要求标签在合理窗口内出现**
   · 默认 14 天 —— 超过这个时间的标签，可能是"事后很久才回头处理"，
     那时的判断依据已经不是"issue 创建时的信息"了
   · 这样得到的对照才有意义
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from collect_issues import GITHUB_TOKEN, GitHubClient               # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}

# =============================================================================
# ★ 标签映射：pandas 的真实标签 → 我们的预测体系
# =============================================================================
# 这是整个"对账"能不能成立的前提。
#
# 实测确认（用 API 拉了 pandas 的全部 160 个标签）：
#   · 类型类标签确实存在：Bug / Build / Docs / Enhancement / Performance
#   · 但**一个 issue 通常同时打"类型标签 + 组件标签"**
#       例：["Bug", "Indexing", "Needs Triage"]
#   · 所以必须**只提取类型标签**，丢掉组件标签
#
# 另外还有几个"看起来像类型"但不在我们预测范围的标签：
#   Deprecate / Refactor / Regression / Testing / Unreliable Test
#   → 这些**不纳入对照**（我们的 Agent 不预测它们），
#     但如果一个 issue 只打了这些标签，那它就**无法用于对照**（见 use_for_eval）。

TYPE_LABELS = {
    "bug": "Bug",
    "build": "Build",
    "docs": "Docs",
    "enhancement": "Enhancement",
    "performance": "Performance",
}

# 这些标签存在但不在我们的预测范围 —— 用来判断"这条 issue 是否可用于对照"
OUT_OF_SCOPE_TYPE_LABELS = {
    "deprecate", "refactor", "regression", "testing",
    "unreliable test", "clean", "code style",
}

# 状态类标签（维护者的分诊动作）—— 用于辅助分析，不参与主对账
TRIAGE_STATE_LABELS = {
    "needs triage", "needs info", "needs tests", "needs discussion",
    "duplicate report", "awaiting review", "needs issue assignment",
}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_issues(client: GitHubClient, repo: str, since_days: int,
                 until_days: int, limit: int) -> list[dict]:
    """抓"创建于 [until_days, since_days] 天前"的 issue。

    ★ 为什么是"14~28 天前"这个窗口？
      · 至少 14 天前创建 → 给维护者足够时间打标签（多数在 7 天内）
      · 最多 28 天前创建 → 避免"半年前的 issue"（那时标签可能被反复改过）

      **这个窗口的选择本身就是方法的一部分** ——
      窗口太短会漏掉"还没被分诊"的，太长会引入"标签被后续修改"的噪声。
    """
    now = datetime.now(timezone.utc)
    # ★ 命名要防歧义 —— 这里踩过一个坑：
    #
    #   第一版用 `t_since = now - since_days` 和 `t_until = now - until_days`，
    #   然后写成 `if created > t_since: continue` —— **逻辑整个反了**，
    #   结果翻了几十页一条都没抓到。
    #
    #   原因是 "since/until" 这两个词**在时间轴上的相对关系不直观**：
    #     · since_days=28 → 时刻在"28 天前"（更早）
    #     · until_days=14 → 时刻在"14 天前"（更晚）
    #   所以 t_since < t_until，和我脑子里"since < until"的直觉刚好相反。
    #
    #   **改成语义明确的名字：earliest / latest** ——
    #   时间轴上的位置一眼可见，不会再搞反。
    earliest = now - timedelta(days=since_days)   # 28 天前（更早）
    latest = now - timedelta(days=until_days)     # 14 天前（更晚）
    # 想要的区间：earliest < created < latest

    out: list[dict] = []
    page = 1
    # ★ 翻页上限必须够大 —— 这里踩过一次：
    #
    #   第一版用 `per_page=50, page<=10`，结果翻满 10 页（500 条）
    #   仍然**一条候选都没抓到**。
    #
    #   原因：pandas 每天新增 issue + PR 约 30~50 条（issues API 会把 PR 一起返回），
    #   所以 500 条只覆盖了最近 **约 12 天** —— 还没走到"14 天前"这条线。
    #
    #   修法两条：
    #     ① per_page 提到 100（API 上限）
    #     ② 翻页上限提到 40（= 4000 条，覆盖约 80~130 天，够用）
    #
    #   **教训**：按时间倒序 + 过滤时间窗口时，
    #   "翻几页够用"取决于**数据产生速率**，不能拍一个数字。
    #   估算方法：窗口天数 × 每天新增量 ÷ 每页条数 = 需要的最小页数。
    per_page = 100
    max_pages = 40
    while len(out) < limit and page <= max_pages:
        batch = client.get(f"/repos/{repo}/issues", {
            "state": "all",
            "sort": "created",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
        })
        if not batch:
            break
        stop = False
        for it in batch:
            created = parse_ts(it["created_at"])
            # 排除 PR（issues API 会把 PR 也返回，靠 pull_request 字段区分）
            if "pull_request" in it:
                continue
            if created > latest:
                continue            # 太新（晚于 14 天前）→ 跳过
            if created < earliest:
                stop = True         # 太老（早于 28 天前）→ 后面的更老，可以停
                break
            out.append(it)
            if len(out) >= limit:
                break
        # 打印进度时带上"已经翻到多久以前"，方便判断还需要几页
        oldest_seen = batch[-1]["created_at"][:10] if batch else "?"
        print(f"  已获取 {len(out)}/{limit} 个候选"
              f"（第 {page} 页，已翻到 {oldest_seen}，"
              f"速率剩余 {client.remaining}）")
        if stop:
            break
        page += 1
    return out[:limit]


def fetch_first_type_label(client: GitHubClient, repo: str, number: int
                           ) -> dict:
    """查 issue 的 timeline，找**第一个类型标签**是什么时候打的。

    ============================================================
    ★ 为什么必须查 timeline
    ============================================================
    GitHub 的 issue API 只给"当前有哪些标签"，**不给"标签是什么时候打的"**。

    而时间正是本实验的核心：
      · 如果标签是创建后 1 天打的 → 维护者当时就是基于创建时的信息判断的 ✓
      · 如果标签是 3 个月后打的 → 那时可能已经有了大量讨论，
        标签反映的是"讨论后的结论"，**不能用来验证"仅凭创建信息能否分诊"** ✗

    所以必须走 timeline API（`/issues/{n}/timeline`），
    它能返回 `labeled` 事件及其时间戳。

    **这一步是整个前瞻验证能不能成立的技术前提。**
    没有它，就只能拿"当前标签"当 gold —— 而那会高估 Agent 的能力
    （因为标签可能是在更多信息之后才打的）。

    ============================================================
    两个真实遇到的边界
    ============================================================
    ① **timeline 里的事件不只有 labeled**：还有 commented / closed /
        cross-referenced / body_edited 等。要按 `event` 字段过滤。
    ② **有的 issue 的标签是创建时自带的**（比如从 issue 模板或 bot 自动打的），
       这种情况下第一个 labeled 事件的时间 ≈ 创建时间 —— 是合法的对照样本。
    """
    try:
        events = client.get(f"/repos/{repo}/issues/{number}/timeline",
                            {"per_page": 100})
    except RuntimeError as e:
        return {"error": str(e)[:80]}

    first_type = None
    body_edited = False
    n_label_events = 0

    for ev in events:
        etype = ev.get("event")
        if etype == "body_edited":
            body_edited = True
        if etype != "labeled":
            continue
        n_label_events += 1
        name = (ev.get("label") or {}).get("name", "")
        key = name.strip().lower()
        if first_type is None and key in TYPE_LABELS:
            first_type = {
                "label": TYPE_LABELS[key],
                "raw": name,
                "at": ev.get("created_at", ""),
            }

    return {
        "first_type_label": first_type,
        "body_edited": body_edited,
        "n_label_events": n_label_events,
    }


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="前瞻验证数据采集（历史回溯）")
    ap.add_argument("--repo", default="pandas-dev/pandas")
    ap.add_argument("--limit", type=int, default=60, help="目标 issue 数")
    ap.add_argument("--since-days", type=int, default=28,
                    help="最早创建时间（天前）")
    ap.add_argument("--until-days", type=int, default=14,
                    help="最晚创建时间（天前）")
    ap.add_argument("--max-label-days", type=int, default=14,
                    help="首个类型标签必须在创建后多少天内出现")
    ap.add_argument("--save", default=str(ROOT / "data" / "prospective_issues.json"))
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    print(f"\n{C['bold']}{C['cyan']}前瞻验证 · 数据采集（轨道 A）{C['reset']}")
    hr()
    print(f"""
  仓库        {args.repo}
  时间窗口    {args.until_days}~{args.since_days} 天前创建的 issue
  标签窗口    首个类型标签必须在创建后 {args.max_label_days} 天内出现
  认证        {'✓' if GITHUB_TOKEN else '✗ 未配 token（60 次/小时，会很慢）'}

  {C['dim']}方法说明：用**创建时的信息**（标题+正文）预测，
    对照**后来真实打上的**类型标签 —— 这是"预测未来"，
    而不是"用已关闭 issue 的 gold 标签算 F1"。{C['reset']}""")

    client = GitHubClient()

    # ---------------------------------------------------------- ① 抓 issue
    hr("① 抓候选 issue")
    raw_issues = fetch_issues(client, args.repo, args.since_days,
                              args.until_days, args.limit)
    if not raw_issues:
        print(f"  {C['red']}✗ 没抓到候选 issue{C['reset']}")
        return 1
    print(f"  候选数 {len(raw_issues)}")

    # ---------------------------------------------------------- ② 查 timeline
    hr("② 查 timeline，定位「首个类型标签」")
    print(f"  {C['dim']}这一步要逐个查 timeline（API 不提供标签的时间戳）——"
          f"约需 {len(raw_issues)} 次请求{C['reset']}\n")

    records = []
    n_no_label = n_too_late = n_ok = 0
    t0 = time.time()

    for i, it in enumerate(raw_issues, 1):
        num = it["number"]
        created = parse_ts(it["created_at"])
        tl = fetch_first_type_label(client, args.repo, num)

        if "error" in tl:
            print(f"  [{i}/{len(raw_issues)}] #{num} "
                  f"{C['red']}timeline 失败{C['reset']}")
            continue

        ft = tl.get("first_type_label")
        if not ft:
            n_no_label += 1
            continue

        label_at = parse_ts(ft["at"]) if ft["at"] else created
        lag_days = (label_at - created).total_seconds() / 86400
        if lag_days > args.max_label_days:
            n_too_late += 1
            continue

        # 当前所有标签（用于辅助分析）
        cur_labels = [l["name"] for l in it.get("labels", [])]
        cur_types = [TYPE_LABELS[l.strip().lower()]
                     for l in cur_labels if l.strip().lower() in TYPE_LABELS]
        cur_states = [l for l in cur_labels
                      if l.strip().lower() in TRIAGE_STATE_LABELS]

        records.append({
            "issue_id": f"#{num}",
            "number": num,
            "title": it.get("title", ""),
            "body": (it.get("body") or "")[:4000],
            "author": (it.get("user") or {}).get("login", ""),
            "created_at": it["created_at"],
            "url": it.get("html_url", ""),
            # --- 对照目标 ---
            "gold_types": cur_types,          # 全部类型标签（当前）
            "first_type_label": ft["label"],  # ★ 首个类型标签（主对照目标）
            "first_type_at": ft["at"],
            "first_type_lag_days": round(lag_days, 1),
            "state_labels": cur_states,       # 状态类标签（辅助）
            "all_labels": cur_labels,
            # --- 有效性标记 ---
            "body_edited": tl.get("body_edited", False),
            "n_label_events": tl.get("n_label_events", 0),
            # --- 用于对齐主项目的 Issue 结构 ---
            "repo": args.repo,
            "state": it.get("state", "open"),
            "labels": cur_labels,
            "stack_trace": "",
        })
        n_ok += 1

        if i % 10 == 0 or i == len(raw_issues):
            el = time.time() - t0
            print(f"  [{i}/{len(raw_issues)}] 有效 {n_ok}　"
                  f"无类型标签 {n_no_label}　标签太晚 {n_too_late}　"
                  f"（{el:.0f}s）")

    # ---------------------------------------------------------- ③ 统计
    hr("③ 采集结果")

    total = len(raw_issues)
    print(f"""
  候选 issue        {total}
  有效样本          {C['green']}{n_ok}{C['reset']}（有类型标签且打得不晚）
  无类型标签        {n_no_label}（{n_no_label / max(total, 1):.0%}）—— 维护者还没分诊
  标签打得太晚      {n_too_late}（超过 {args.max_label_days} 天）

  {C['dim']}★ "无类型标签"这一栏本身就是个诚实的信号：
    说明**不是所有 issue 都会被分诊** ——
    如果只报"有效率"，会掩盖这个事实。{C['reset']}""")

    if not records:
        print(f"  {C['red']}✗ 没有有效样本{C['reset']}")
        return 1

    # 标签分布
    from collections import Counter
    dist = Counter(r["first_type_label"] for r in records)
    print(f"\n  首个类型标签的分布：")
    for lab, cnt in dist.most_common():
        bar = "█" * max(1, int(cnt / max(dist.values()) * 22))
        print(f"    {lab:<14}{cnt:>4}  {C['cyan']}{bar}{C['reset']}")

    # 响应延迟
    lags = sorted(r["first_type_lag_days"] for r in records)
    import statistics
    print(f"""
  分诊延迟（创建 → 首个类型标签）
    中位 {statistics.median(lags):.1f} 天　
    最快 {lags[0]:.1f} 天　最慢 {lags[-1]:.1f} 天

  {C['dim']}  这个数字决定了"真前瞻"实验要等多久才能对账 ——
    中位数就是最小观测窗口。{C['reset']}""")

    # 数据质量
    edited = sum(1 for r in records if r["body_edited"])
    print(f"""
  数据质量
    正文被编辑过   {edited}/{len(records)}（{edited / len(records):.0%}）
    {C['dim']}  ⚠ 被编辑过的标题/正文，我这里用的是**编辑后**的版本 ——
      GitHub API 不提供 body 的编辑历史。
      这是本实验一个**必须声明的边界**：
      对编辑过的样本，Agent 看到的信息比当时维护者看到的更多。{C['reset']}""")

    # ---------------------------------------------------------- 保存
    if args.stats:
        print(f"\n  {C['dim']}（--stats 模式，不保存）{C['reset']}")
        return 0

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "repo": args.repo,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_candidates": total,
            "n_valid": n_ok,
            "n_no_label": n_no_label,
            "n_too_late": n_too_late,
            "window_days": [args.until_days, args.since_days],
            "max_label_days": args.max_label_days,
            "purpose": ("前瞻验证（历史回溯）：用 issue 创建时的信息预测，"
                        "对照维护者后来打上的首个类型标签。"),
            "caveats": [
                "body 被编辑过的样本，用的是编辑后版本（API 无编辑历史）",
                "对照的是「首个类型标签」，不含组件标签与状态标签",
                "只覆盖 pandas 一个仓库",
            ],
        },
        "issues": records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {C['green']}✓{C['reset']} 已保存 → {out}"
          f"（{out.stat().st_size / 1024:.0f} KB）")
    print(f"    下一步：python scripts/eval_prospective.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
