#!/usr/bin/env python3
"""
Nightwatch · 真实 PR 采集（P2-3 的延伸）

    # 采集 pandas 的 30 个已合并 PR（含 diff）
    python scripts/collect_prs.py --repo pandas-dev/pandas --limit 30

    # 只看统计，不保存
    python scripts/collect_prs.py --repo pandas-dev/pandas --limit 30 --stats

    # 用不同的仓库
    python scripts/collect_prs.py --repo numpy/numpy --limit 20

==============================================================================
为什么要采真实 PR
==============================================================================
`demo_pr_review.py` 里的三个 PR 是**手工构造**的 —— 它们能演示机制，
但有个致命问题：**构造数据会不自觉地"配合"检测规则。**

    我写的敏感文件例子，恰好都是我的正则能匹配的
    → 于是"检测准确率 100%"这个结论毫无意义

**真实数据的作用是暴露这种偏差**：真实的 PR 文件名千奇百怪，
真实 diff 里有 renames / binary files / 自动生成的代码 ——
而构造数据里不会有这些。

> 这正是 `docs/05` 里反复出现的同一个教训：
> **Mock/合成数据能验链路，不能验效果。**

==============================================================================
关于 API 的两个实际约束
==============================================================================
① **rate limit**：未认证 60 次/小时，认证 5000 次/小时。
   拉 30 个 PR = 1 次列表请求 + 30 次 diff 请求 = 31 次。
   所以未认证时只能拉一两个 PR 就要等一小时。

② **diff 的 Accept 头**：GitHub 返回 diff 需要
   `Accept: application/vnd.github.v3.diff`，返回的是**纯文本**而不是 JSON。
   所以不能复用 `collect_issues.py` 里的 `get()`（它假定响应是 JSON）——
   本脚本加了一个 `get_text()`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# 复用 issue 采集脚本里的客户端（认证 / 重试 / 速率限制都已处理好）
from collect_issues import GITHUB_TOKEN, API, GitHubClient        # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


class PRClient(GitHubClient):
    """加一个 `get_text()` —— 因为拉 diff 拿到的不是 JSON。

    ★ 为什么用继承而不是改 `GitHubClient`？

    因为 `collect_issues.py` 是**已验证可用**的脚本，
    改它等于给一个稳定组件引入变更风险。

    **扩展优先于修改** —— 尤其是当原组件"能跑"而你不确定改动的全部影响时。
    （这也是开闭原则的实际用法：加能力，不改已有行为。）
    """

    def get_text(self, path: str, accept: str,
                 _max_retries: int = 3) -> str:
        """GET 纯文本响应（用于 diff）。

        重试逻辑和父类的 `get()` 一致 ——
        大项目 diff 可能几 MB，中途断连的概率不低。
        """
        import http.client
        import urllib.error
        import urllib.request

        url = path if path.startswith("http") else f"{API}{path}"
        last_err: Exception | None = None

        for attempt in range(_max_retries):
            req = urllib.request.Request(url)
            req.add_header("Accept", accept)
            req.add_header("User-Agent", "nightwatch-collector")
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    self.remaining = int(
                        resp.headers.get("X-RateLimit-Remaining", -1))
                    return resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code == 406:
                    # ★ 406 = diff 太大，GitHub 拒绝返回。
                    #   这是个真实存在的边界（大型重构 PR 的 diff 可能几十 MB）。
                    #   处理方式是**明确报告**，而不是返回空字符串 ——
                    #   否则调用方会以为"这个 PR 没有改动"。
                    raise RuntimeError(
                        f"diff 过大，GitHub 拒绝返回（HTTP 406）：{path}\n"
                        f"  → 这类 PR 恰恰是最需要人工关注的（改动巨大）") from e
                if 500 <= e.code < 600 and attempt < _max_retries - 1:
                    last_err = e
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {path}") from e
            except (urllib.error.URLError, OSError,
                    http.client.HTTPException) as e:
                last_err = e
                if attempt < _max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    print(f"  ⚠ diff 拉取失败（{type(e).__name__}），"
                          f"{delay:.0f}s 后重试", file=sys.stderr)
                    time.sleep(delay)
                    continue

        raise RuntimeError(f"diff 拉取失败（重试 {_max_retries} 次）：{last_err}")


def fetch_pr_list(client: PRClient, repo: str, limit: int,
                  state: str = "closed",
                  outcome: str = "merged") -> list[dict]:
    """拉 PR 列表（分页）。

    ============================================================
    ★ 关于 `outcome`：为什么要能采"未合并"的 PR
    ============================================================
    第一版只采**已合并**的 PR，理由是"有'人类审核过'这个隐含标签"。

    **但那恰恰让样本变得没价值** —— 实测发现：

        已合并 PR 上，缺测试率 0%、风险几乎全是"低"

    因为"能合并"本身就意味着"维护者认为它合格"。
    **在最干净的样本上测风险规则，等于没测。**

    要看规则能不能识别风险，样本必须包含"**有问题的**"那些 ——
    也就是被关闭但未合并的 PR（维护者选择了不接受）。

    所以现在支持三类：
      · `merged`   已合并（"合格"样本，作为对照）
      · `unmerged` 已关闭但未合并（"被拒绝"样本，这才是考点）
      · `all`      两者都要（用于对比两组分布）
    """
    out: list[dict] = []
    merged_count = unmerged_count = 0
    page = 1
    page_size = 50          # 拿满一页再筛，否则筛掉太多会导致翻页不足

    while len(out) < limit:
        batch = client.get("/repos/%s/pulls" % repo, {
            "state": state, "per_page": page_size, "page": page,
            "sort": "updated", "direction": "desc",
        })
        if not batch:
            break
        for pr in batch:
            is_merged = bool(pr.get("merged_at"))
            if outcome == "merged" and not is_merged:
                continue
            if outcome == "unmerged" and is_merged:
                continue
            # outcome == "all" 时两者都要
            out.append(pr)
            merged_count += 1 if is_merged else 0
            unmerged_count += 0 if is_merged else 1
            if len(out) >= limit:
                break

        print(f"  已获取 {len(out)}/{limit}"
              f"（已合并 {merged_count}，未合并 {unmerged_count}）"
              f"　第 {page} 页，速率剩余 {client.remaining}")
        page += 1
        if page > 40:      # 安全上限（未合并的比例可能很低，需要多翻几页）
            break
    return out[:limit]


def to_pr_record(raw: dict, diff: str) -> dict:
    """把 GitHub 的 PR 对象转成我们需要的精简结构。

    ★ 只保留"初审需要"的字段。

    为什么要精简？因为 `docs/03` 的上下文预算里有一条原则：
    **工具返回必须为下游的上下文预算负责。**
    GitHub 的 PR 对象有 80+ 个字段（绝大部分是 URL 和嵌套对象），
    原样塞进上下文是纯浪费。

    ============================================================
    ★ 改动统计从 diff 里算，不用 API 的字段
    ============================================================
    第一版直接用 `raw["changed_files"]` / `additions` / `deletions`，
    结果**全是 0** —— 因为 GitHub 的 PR **列表** API
    （`/repos/{owner}/{repo}/pulls`）**不返回这些统计字段**，
    它们只在**单个 PR 详情** API（`/pulls/{number}`）里有。

    要拿它们就得多发一次请求 —— 但**我们已经有 diff 了**，
    而 diff 是改动内容的**权威来源**（API 的统计字段也是从它算的）。

    所以改成用 `parse_diff()` 从 diff 里统计：
      · 省一次 API 请求（对 rate limit 敏感的采集很值）
      · 数据更准（不依赖 API 的聚合字段）

    **"手上有原始数据时，不要再去要一份聚合结果"** ——
    原始数据包含更多信息，而且不用额外成本。
    """
    files = changes = 0
    additions = deletions = 0
    if diff:
        from nightwatch.tools.pr_review import parse_diff
        parsed = parse_diff(diff)
        files = len(parsed)
        additions = sum(f.additions for f in parsed)
        deletions = sum(f.deletions for f in parsed)

    return {
        "pr_id": f"PR-{raw['number']}",
        "number": raw["number"],
        "title": raw.get("title", ""),
        "author": (raw.get("user") or {}).get("login", ""),
        "state": raw.get("state", ""),
        "merged": bool(raw.get("merged_at")),
        "created_at": raw.get("created_at", ""),
        "merged_at": raw.get("merged_at", ""),
        "base": (raw.get("base") or {}).get("ref", ""),
        # 统计来自 diff（权威来源），不是 API 的聚合字段
        "changed_files": files,
        "additions": additions,
        "deletions": deletions,
        # 保留 API 的原值以便对比（能验证"从 diff 算得对不对"）
        "api_changed_files": raw.get("changed_files"),
        "diff": diff,
        "diff_chars": len(diff),
    }


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="采集真实 PR（含 diff）")
    ap.add_argument("--repo", default="pandas-dev/pandas",
                    help="仓库（owner/name）")
    ap.add_argument("--limit", type=int, default=20, help="采集几个 PR")
    ap.add_argument("--state", default="closed",
                    choices=["open", "closed", "all"])
    ap.add_argument("--outcome", default="merged",
                    choices=["merged", "unmerged", "all"],
                    help=("采哪些结果的 PR：merged=已合并（合格样本）/"
                          "unmerged=已关闭未合并（被拒绝样本，测风险规则要用它）/"
                          "all=两者都要"))
    ap.add_argument("--save", default=str(ROOT / "data" / "real_prs.json"))
    ap.add_argument("--stats", action="store_true", help="只统计不保存")
    ap.add_argument("--no-diff", action="store_true",
                    help="不拉 diff（省额度，只看元数据分布）")
    args = ap.parse_args()

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 真实 PR 采集{C['reset']}")
    hr()
    _label = {"merged": "已合并", "unmerged": "已关闭未合并", "all": "全部"}[args.outcome]
    print(f"  仓库        {args.repo}")
    print(f"  目标        {args.limit} 个 PR（{_label}）")
    print(f"  认证        {'✓ 已配置 token' if GITHUB_TOKEN else '✗ 未配置（60 次/小时）'}")

    client = PRClient()
    try:
        prs = fetch_pr_list(client, args.repo, args.limit, args.state,
                            args.outcome)
    except RuntimeError as e:
        print(f"\n  {C['red']}✗ {e}{C['reset']}")
        return 1

    if not prs:
        print(f"  {C['yellow']}没有取到已合并的 PR{C['reset']}")
        return 1

    # ---------------------------------------------------------- 拉 diff
    records = []
    if not args.no_diff:
        hr("拉取 diff")
        for i, raw in enumerate(prs, 1):
            num = raw["number"]
            try:
                diff = client.get_text(
                    f"/repos/{args.repo}/pulls/{num}",
                    accept="application/vnd.github.v3.diff")
            except RuntimeError as e:
                # ★ 单个 PR 失败不该中断整个采集 —— 但要记下来。
                #   因为"哪些失败了"本身是好信息（比如都是超大 diff）。
                print(f"  [{i}/{len(prs)}] PR-{num} "
                      f"{C['red']}✗ {str(e)[:60]}{C['reset']}")
                continue
            rec = to_pr_record(raw, diff)
            records.append(rec)
            print(f"  [{i}/{len(prs)}] PR-{num:<6} "
                  f"{rec['changed_files']:>3} 文件  "
                  f"+{rec['additions']}/-{rec['deletions']:<6} "
                  f"diff {rec['diff_chars']:>7} 字符  "
                  f"{C['dim']}{client.remaining} 剩余{C['reset']}")
    else:
        records = [to_pr_record(r, "") for r in prs]

    if not records:
        print(f"  {C['red']}✗ 没有成功采到任何 PR{C['reset']}")
        return 1

    # ---------------------------------------------------------- 统计
    hr("统计（这些数字决定了初审规则的有效性）")

    import statistics
    files = [r["changed_files"] for r in records]
    adds = [r["additions"] for r in records]
    dels = [r["deletions"] for r in records]

    print(f"  采集成功    {len(records)} 个")
    print(f"  改动文件数  中位 {statistics.median(files):.0f}　"
          f"最大 {max(files)}　最小 {min(files)}")
    print(f"  新增行数    中位 {statistics.median(adds):.0f}　"
          f"最大 {max(adds)}")
    print(f"  删除行数    中位 {statistics.median(dels):.0f}　"
          f"最大 {max(dels)}")
    if records and records[0]["diff_chars"]:
        chars = [r["diff_chars"] for r in records]
        print(f"  diff 体积   中位 {statistics.median(chars):.0f} 字符　"
              f"最大 {max(chars)}")

    # ★ 这个统计是判断"构造数据够不够真实"的依据
    print(f"""
  {C['dim']}  ★ 为什么要看这些数字？
    如果真实 PR 的中位改动是 3 个文件 / +50 行，而构造数据里是 4 个文件，那"接近"。
    但如果真实的中位是 20 个文件 / +800 行 —— 那构造数据明显"太干净"，
    用它测出来的"风险识别率"会虚高。

    **先看真实分布，再决定构造数据要不要改。** 这就是采真实数据的意义。{C['reset']}""")

    # 用初审逻辑跑一遍，看真实数据下的风险分布
    if records and records[0]["diff_chars"]:
        hr("真实数据下的风险分布（用现有初审逻辑跑）")
        from nightwatch.tools.pr_review import review_pr
        dist = {"低": 0, "中": 0, "高": 0}
        missing_tests = 0
        sensitive_hits = 0
        for r in records:
            try:
                rev = review_pr(r["diff"])
            except Exception as e:                          # noqa: BLE001
                print(f"  {C['yellow']}⚠ {r['pr_id']} 解析失败："
                      f"{type(e).__name__}{C['reset']}")
                continue
            dist[rev.risk_level] = dist.get(rev.risk_level, 0) + 1
            missing_tests += 1 if rev.missing_tests else 0
            sensitive_hits += len(rev.sensitive_files)
        print(f"  风险分布    低 {dist['低']}　中 {dist['中']}　高 {dist['高']}")
        print(f"  缺测试        {missing_tests}/{len(records)} "
              f"（{missing_tests / len(records):.0%}）")
        print(f"  敏感文件命中  {sensitive_hits} 处")
        print(f"""
  {C['dim']}  这些数字就是"真实数据下的表现" —— 它们比构造数据上的
    100% 检测率更有说服力（也更可能暴露规则的问题）。{C['reset']}""")

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
            "n_prs": len(records),
            "authenticated": bool(GITHUB_TOKEN),
            "outcome": args.outcome,
            "note": ("真实 GitHub PR，含完整 diff。"
                     "用于验证 PR 初审规则在真实数据上的表现。"),
        },
        "prs": records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n  {C['green']}✓{C['reset']} 已保存 → {out}（{size_mb:.1f} MB）")
    print(f"    用真实 PR 跑初审：")
    print(f"      python scripts/demo_pr_review.py --real {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
