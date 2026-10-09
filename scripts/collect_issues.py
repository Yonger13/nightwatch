#!/usr/bin/env python3
"""
Nightwatch · 第 0 周工具：从真实开源项目采集 issue

================================================================================
为什么这个脚本是整个项目的第一步
================================================================================
业务场景调研里有一条明确的低 ROI 画像：

    「**你自己都没搞清楚的流程** —— Agent 不会修复这个缺口，
      它会立刻、反复地把它暴露出来。」

所以在写任何 Agent 代码之前，必须先做一件事：
**把真实项目的 issue 处理流程看明白。**

这个脚本就是为这件事服务的。它做两件事：

  ① **采集**：把某个项目的真实 issue 拉下来（含维护者的标注 = ground truth）
  ② **观察**：输出标签分布、重复情况、时间跨度等统计

跑完之后你应该能回答这些问题（答不上来就别开始写代码）：
  · 这个项目实际用哪些标签？各自的含义和边界是什么？
  · 维护者在什么情况下会把一个 issue 标记为 duplicate？
  · 他们通常多久响应一次？一个 issue 从开到关平均经历什么？
  · 哪些类型的问题最常见？哪些最少见（测评测集要保证覆盖到）？

================================================================================
为什么用 GitHub API 而不是爬网页
================================================================================
三个理由：
  1. 官方 API 返回的是**结构化数据**（JSON），不需要写解析器
  2. 它有**明确的速率限制**（未认证 60 次/小时，认证 5000 次/小时）——
     这是个真实的工程约束，值得体验一次
  3. 网页结构会变，API 相对稳定

================================================================================
⚠️ 两个必须知道的限制
================================================================================
  1. **速率限制**：不带 token 每小时只能请求 60 次。要采集大项目必须设 token：
         export NW_GITHUB_TOKEN=ghp_xxxxx
     （GitHub → Settings → Developer settings → Personal access tokens）
     只需要 `public_repo` 或干脆只读权限即可。

  2. **ground truth 的局限**：维护者打的标签是他的判断，**不等于绝对正确**。
     · 有的 issue 维护者根本没打标签（数据缺失）
     · 有的标签是机器人打的（要识别并排除）
     · 不同维护者的标准可能不一致
     所以在报告里要说清"ground truth 来自维护者标注"，而不是"正确答案"。

================================================================================
用法
================================================================================
    # 查看某个项目最近 100 个 issue 的概况（第 0 周观察）
    python scripts/collect_issues.py --repo pallets/flask --limit 100 --stats

    # 采集并保存为评测集格式
    python scripts/collect_issues.py --repo pallets/flask --limit 300 \\
        --save data/flask_dataset.json

    # 只看已关闭的（关闭的 issue 才有完整的处理链条，是更好的样本）
    python scripts/collect_issues.py --repo owner/name --state closed --limit 200

配合学习指南：docs/01-第0周-观察真实项目.md
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ★ 导入 config 不只是为了用它 —— 更是为了**触发 .env 加载**。
#
# 这是个踩过的坑：`_load_dotenv()` 写在 config.py 里，
# 而本脚本原来只导入了 `nightwatch.models`（不经过 config）——
# 于是 `.env` 里的 NW_GITHUB_TOKEN 读不到，
# 程序会静默地按"未认证"运行（限速 60 次/小时而不是 5000）。
#
# 危险之处在于：**它不会报错**，只是慢/采不够量，
# 而你会以为是"GitHub 限流"而不是"token 没读到"。
#
# 教训：**配置的加载必须发生在所有入口路径上。**
# 只在一个模块里加载，等于在赌所有调用方都会经过它。
from nightwatch import config as _config          # noqa: F401  触发 .env 加载
from nightwatch.models import Issue, IssueState, to_dict          # noqa: E402


API = "https://api.github.com"

# GitHub 对未认证请求的限速很严（60 次/小时）。
# 看这个数字能理解一个真实的工程问题：**如果你要采 1000 个 issue，
# 每次请求拿 100 条（API 上限），也需要 10 次请求 + 分页，
# 未认证时几分钟就用完了额度。** 所以生产级采集必须认证 + 带重试和退避。
GITHUB_TOKEN = os.getenv("NW_GITHUB_TOKEN", "")


class GitHubClient:
    """极简 GitHub 客户端（只用标准库，避免引入依赖）。

    这个类刻意写得小 —— 它的目的不是做 SDK，是让你看清楚
    "调一个 REST API 到底需要处理哪些事"：
      · 认证头
      · 速率限制（看响应头，而不是撞了才知道）
      · 分页
      · 错误处理
    """

    def __init__(self, token: str = ""):
        self.token = token or GITHUB_TOKEN
        self.remaining: Optional[int] = None
        self._last_reset: float = 0.0

    def get(self, path: str, params: dict | None = None,
            _max_retries: int = 3) -> Any:
        """GET 请求（带重试）。

        ============================================================
        为什么必须加重试 —— 一个实测踩到的坑
        ============================================================
        采 pandas 的 issue 时报了：

            http.client.IncompleteRead: IncompleteRead(503862 bytes read,
                                                       22112 more expected)

        原因是这类大项目的响应体很大（500KB+），传输中途连接被掐断。
        而原实现没有重试，一次网络抖动就让整个采集失败。

        **网络请求必须假定会失败** —— 这是分布式系统的基本假设，
        也是最容易在"本地开发时从没遇到过"的地方被忽略的。

        另一条相关经验：**减小单次请求的体积也能降低失败率**。
        所以下面把 per_page 从 100 降到 50 —— 用更多次小请求
        换更低的失败率，这在网络不稳的环境下是划算的。
        """
        url = path if path.startswith("http") else f"{API}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        last_err: Exception | None = None
        for attempt in range(_max_retries):
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("User-Agent", "nightwatch-collector")
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    self.remaining = int(resp.headers.get("X-RateLimit-Remaining", -1))
                    self._last_reset = float(resp.headers.get("X-RateLimit-Reset", 0))
                    raw = resp.read()
                return json.loads(raw.decode("utf-8"))

            except urllib.error.HTTPError as e:
                # 4xx 是客户端问题，重试没意义 —— 直接抛（快速失败）
                if e.code == 403 and self.remaining == 0:
                    wait = max(0, self._last_reset - time.time())
                    raise RuntimeError(
                        f"GitHub API 速率限制已用尽，需等待 {wait / 60:.1f} 分钟。\n"
                        f"  → 解决：设置环境变量 NW_GITHUB_TOKEN=ghp_xxx 后重试"
                        f"（认证后额度从 60 次/小时提到 5000 次/小时）"
                    ) from e
                if e.code == 404:
                    raise RuntimeError(
                        f"仓库不存在或不可访问：{url}\n"
                        f"  → 检查 repo 名称格式是否正确（应该是 owner/name）"
                    ) from e
                if 500 <= e.code < 600 and attempt < _max_retries - 1:
                    last_err = e          # 5xx 是服务端问题，可重试
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
                ) from e

            except (urllib.error.URLError, OSError,
                    http.client.HTTPException) as e:
                # ★ IncompleteRead 必须单独捕获 —— 这里踩了一个坑：
                #
                # 我第一版把它归到 OSError 里，还在注释里"确认"了
                # "IncompleteRead 继承自 http.client.HTTPException，
                #  而它是 OSError 的子类"。
                #
                # **这个继承关系是错的。** 实际是：
                #     BaseException
                #      └── Exception
                #           ├── OSError
                #           │    └── URLError
                #           └── http.client.HTTPException
                #                └── IncompleteRead
                # HTTPException 和 OSError 是**兄弟**，不是父子。
                #
                # 后果：重试逻辑完全没生效，一次网络中断仍然让采集失败。
                #
                # 教训：**不要凭记忆写异常继承关系。**
                # 要么查文档，要么在代码里显式列出来（像下面这样），
                # 要么用 `except Exception` 兜底后再按类型分流。
                last_err = e
                if attempt < _max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    print(f"  ⚠ 请求失败（{type(e).__name__}），"
                          f"{delay:.0f}s 后重试 {attempt + 2}/{_max_retries}",
                          file=sys.stderr)
                    time.sleep(delay)
                    continue

        raise RuntimeError(
            f"网络请求失败（重试 {_max_retries} 次后仍失败）：{last_err}\n"
            f"  → 检查网络连通性；大项目响应体较大，可降低 --limit 或加 --page-size"
        )


def fetch_issues(client: GitHubClient, repo: str, limit: int = 100,
                 state: str = "all", with_comments: int = 5) -> list[dict]:
    """采集 issue（分页）。

    ============================================================
    为什么要排除 pull request
    ============================================================
    GitHub 的 /issues 接口**同时返回 issue 和 PR**（因为 PR 在数据模型上
    就是特殊的 issue）。如果不排除，你的数据集里会混进大量 PR，
    而 PR 的分流逻辑和 issue 完全不同 —— 这会污染指标。

    这是个很典型的"不知道就会踩"的坑：接口名字叫 issues，
    返回的却不只有 issue。
    """
    out: list[dict] = []
    page = 1
    # ★ per_page 从 GitHub 允许的最大值 100 降到 50
    #
    # 为什么主动降级？因为大项目的单条 issue 很长（正文 + 标签 + 用户信息），
    # 100 条一页的响应体经常超过 500KB，在弱网环境下容易传输中断
    # （实测报过 IncompleteRead）。
    #
    # 用"更多次小请求"换"更低的失败率"是个划算的交易 ——
    # 反正 API 额度是 5000/小时，不是瓶颈。
    per_page = min(50, limit)

    while len(out) < limit:
        batch = client.get(f"/repos/{repo}/issues", {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc",
        })
        if not batch:
            break

        for raw in batch:
            # ★ 排除 PR
            if "pull_request" in raw:
                continue
            out.append(raw)
            if len(out) >= limit:
                break

        if len(batch) < per_page:
            break                       # 没有下一页了
        page += 1

        if client.remaining is not None and client.remaining < 10:
            print(f"  ⚠ 剩余 API 额度 {client.remaining}，提前停止", file=sys.stderr)
            break

    # --- 补充评论（可选）---
    # 评论里有维护者的判断依据，是"决策过程"的最好材料。
    # 但每个 issue 一次请求，会让额度消耗快 N 倍，所以限制条数。
    if with_comments > 0:
        for i, raw in enumerate(out[:with_comments]):
            try:
                raw["_comments_detail"] = client.get(
                    f"/repos/{repo}/issues/{raw['number']}/comments", {"per_page": 5})
            except RuntimeError as e:
                print(f"  ⚠ 获取 #{raw['number']} 评论失败：{e}", file=sys.stderr)

    return out


def is_bot(user: dict) -> bool:
    """判断是否机器人账号。

    ★ 这一步非常重要，但很容易漏：
    很多仓库有自动化机器人（dependabot、stale bot 等）在打标签、关 issue。
    如果不过滤，你的 ground truth 里会混入大量"机器人按规则打的标签"，
    而那不是人类维护者的判断 —— 会影响评测的可信度。

    识别方式：
      · 用户名以 [bot] 结尾（GitHub App 的约定）
      · 用户类型是 Bot
      · 用户名包含常见机器人关键词
    """
    login = (user or {}).get("login", "")
    return (user or {}).get("type") == "Bot" \
        or login.endswith("[bot]") \
        or any(k in login.lower() for k in ("bot", "dependabot", "renovate"))


def to_issue(raw: dict, repo: str) -> Optional[Issue]:
    """转成本项目的 Issue 对象。

    ★ 关键：把维护者的真实动作提取成 ground truth。
    这些就是"免费的真实标注" —— 是这个场景最大的价值。
    """
    if is_bot(raw.get("user", {})):
        # 机器人提交的 issue（比如自动生成的报告）：跳过。
        # 它们的格式高度模板化，放在评测集里会造成"虚高"。
        return None

    labels = [l["name"] for l in raw.get("labels", []) if isinstance(l, dict)]

    # --- 提取 ground truth ---
    # ① "重复"关系：维护者通常用标签表达（如 duplicate / wontfix / 已重复）
    dup_marker = None
    for l in labels:
        low = l.lower()
        if "dup" in low or "重复" in low:
            body = (raw.get("body") or "") + " " + (raw.get("title") or "")
            import re
            m = re.search(r"#(\d+)", body)
            if m:
                dup_marker = f"#{m.group(1)}"
            break

    # ② 处置动作：关闭状态 + 关闭原因
    state = IssueState.CLOSED if raw.get("state") == "closed" else IssueState.OPEN
    action = ""
    if raw.get("state") == "closed":
        reason = raw.get("state_reason") or ""
        action = f"closed({reason})" if reason else "closed"

    # --- 提取堆栈（判重的强信号）---
    body = raw.get("body") or ""
    import re
    stack_lines = []
    for line in body.splitlines():
        if re.search(r"(Traceback|Error:|Exception|at \w+\.|File \")", line):
            stack_lines.append(line.strip())
        if len(stack_lines) >= 20:
            break

    return Issue(
        issue_id=f"#{raw['number']}",
        title=(raw.get("title") or "")[:300],
        body=body[:4000],
        author=raw.get("user", {}).get("login", "unknown"),
        repo=repo,
        labels=labels,
        attachments=[],
        stack_trace="\n".join(stack_lines),
        state=state,
        created_at=raw.get("created_at", ""),
        # --- ground truth ---
        gold_labels=labels,
        gold_is_duplicate_of=dup_marker,
        gold_action=action,
    )


def print_stats(issues: list[Issue], raw_count: int, skipped: int) -> None:
    """输出观察统计 —— 这是第 0 周真正的产出。

    ============================================================
    为什么要看这些数字，而不是直接开写代码
    ============================================================
    因为这些数字直接决定你后面所有的设计决策：

      标签分布 → 决定关键词表要覆盖哪些类型（也是评测集分层的依据）
      无标签比例 → 决定有多少样本能用（太低就换个项目）
      重复比例 → 决定 duplicate 检测值不值得投入
      作者集中度 → 决定"同一作者历史"这条路召回有没有意义
      时间跨度 → 决定衰减半衰期该设多少
    """
    n = len(issues)
    if n == 0:
        print("没有采集到有效样本")
        return

    print(f"\n{'=' * 74}")
    print(f"观察统计  （原始 {raw_count} 条 → 有效 {n} 条，过滤掉 {skipped} 条）")
    print(f"{'=' * 74}")

    # --- 标签分布 ---
    label_counter: Counter = Counter()
    for i in issues:
        label_counter.update(i.gold_labels)

    unlabeled = sum(1 for i in issues if not i.gold_labels)
    print(f"\n【标签分布】有标签 {n - unlabeled}/{n} 条"
          f"（{unlabeled / n * 100:.1f}% 无标签）")
    if unlabeled / n > 0.3:
        print("  ⚠ 无标签比例超过 30% —— 这个项目的维护者打标不积极，"
              "ground truth 会很少，建议换一个更规范的项目")
    for label, cnt in label_counter.most_common(20):
        bar = "█" * max(1, int(cnt / max(label_counter.values()) * 30))
        print(f"  {cnt:>5}  {label:<28} {bar}")

    # --- 状态与处置 ---
    closed = sum(1 for i in issues if i.state == IssueState.CLOSED)
    print(f"\n【状态】已关闭 {closed}/{n}（{closed / n * 100:.1f}%）")
    action_counter = Counter(i.gold_action for i in issues if i.gold_action)
    for act, cnt in action_counter.most_common(6):
        print(f"  {cnt:>5}  {act}")

    # --- 重复情况 ---
    dups = [i for i in issues if i.gold_is_duplicate_of]
    print(f"\n【重复】显式标记为重复的 {len(dups)} 条"
          f"（{len(dups) / n * 100:.1f}%）")
    if len(dups) / n < 0.03:
        print("  ⚠ 重复比例低于 3% —— duplicate 检测在这个项目上收益有限，"
              "可以把精力放在分类准确率上")
    else:
        print(f"  ✓ 有 {len(dups)} 条重复样本，足以评测 duplicate 检测能力")

    # --- 作者集中度 ---
    author_counter = Counter(i.author for i in issues)
    top_authors = author_counter.most_common(5)
    print(f"\n【提交者集中度】共 {len(author_counter)} 个提交者")
    for a, c in top_authors:
        print(f"  {c:>5}  {a}")
    if top_authors and top_authors[0][1] / n > 0.2:
        print("  ✓ 头部提交者占比高 —— 『同一作者历史召回』这条路有实际价值")
    else:
        print("  ⚠ 提交者很分散 —— 按 author 精确召回可能命中率低，"
              "应更依赖语义召回")

    # --- 堆栈可用性 ---
    with_stack = sum(1 for i in issues if i.stack_trace)
    print(f"\n【堆栈信息】含报错堆栈的 {with_stack}/{n}"
          f"（{with_stack / n * 100:.1f}%）")
    print("  → 堆栈是判重的强信号；占比高说明可以依赖它做相似度检索")

    # --- 时间跨度 ---
    dates = sorted(i.created_at for i in issues if i.created_at)
    if dates:
        print(f"\n【时间跨度】{dates[0][:10]} ~ {dates[-1][:10]}")
        print("  → 这个跨度决定记忆衰减半衰期的量级"
              "（跨度 3 年就别把半衰期设为 30 天）")

    print(f"\n{'=' * 74}")
    print("下一步（第 0 周任务，见 docs/01-第0周-观察真实项目.md）")
    print(f"{'=' * 74}")
    print("""
  1. 人工通读 30~50 条 issue，把「维护者做了什么判断」记下来
  2. 画出一张「一个 issue 从提交到关闭」的流程图
  3. 检查你是否能回答下面三个问题 —— 答不上来就继续读：
       · 这个项目的标签体系里，最容易混淆的是哪两个？
       · 维护者在什么情况下判 duplicate？依据是什么？
       · 哪一类问题你想让 Agent 处理，哪一类必须转人工？
  4. 用 --save 保存数据集，后面评测直接用
""")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 GitHub 采集真实 issue 用于第 0 周观察与后续评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--repo", required=True, help="仓库全名，如 pallets/flask")
    ap.add_argument("--limit", type=int, default=100, help="采集条数（默认 100）")
    ap.add_argument("--state", default="all",
                    choices=["all", "open", "closed"],
                    help="issue 状态（默认 all；只做评测建议用 closed）")
    ap.add_argument("--comments", type=int, default=0,
                    help="额外抓多少个 issue 的评论（消耗额度，默认 0）")
    ap.add_argument("--save", help="保存为评测集 JSON 的路径")
    ap.add_argument("--stats", action="store_true", help="输出详细观察统计")
    args = ap.parse_args()

    if "/" not in args.repo:
        print("错误：--repo 必须是 owner/name 格式，例如 pallets/flask",
              file=sys.stderr)
        return 2

    print(f"\n从 {args.repo} 采集 issue（最多 {args.limit} 条，state={args.state}）")
    if not GITHUB_TOKEN:
        print("⚠ 未设置 NW_GITHUB_TOKEN —— 未认证请求每小时上限 60 次。")
        print("  采集超过约 60 条时会被限流。设置方式：")
        print("    export NW_GITHUB_TOKEN=ghp_xxxxxxxx")
        print("  （GitHub → Settings → Developer settings → "
              "Personal access tokens → 勾选 public_repo 即可）\n")

    client = GitHubClient()
    try:
        raws = fetch_issues(client, args.repo, args.limit, args.state,
                            args.comments)
    except RuntimeError as e:
        print(f"\n✗ 采集失败：\n{e}", file=sys.stderr)
        return 1

    issues, skipped = [], 0
    for raw in raws:
        issue = to_issue(raw, args.repo)
        if issue is None:
            skipped += 1
            continue
        issues.append(issue)

    print(f"✓ 采集完成：原始 {len(raws)} 条 → 有效 {len(issues)} 条"
          f"（过滤机器人生成 {skipped} 条）")
    if client.remaining is not None:
        print(f"  API 剩余额度：{client.remaining}")

    if args.stats or not args.save:
        print_stats(issues, len(raws), skipped)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": f"real-{args.repo.replace('/', '_')}",
            "source": "github",
            "_source_repo": args.repo,
            "notes": (f"来自 {args.repo} 的真实 issue（state={args.state}）。"
                      f"ground truth 来自维护者的标签与关闭动作，"
                      f"是**维护者的判断**而非绝对正确答案。"
                      f"已过滤机器人账号提交的内容。"),
            # 全部放 train；留出集的划分由 DatasetBuilder 负责
            "train": [to_dict(i) for i in issues],
            "holdout": [],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n✓ 已保存到 {out}")
        print(f"  用它跑评测：python scripts/run_eval.py --dataset {out} "
              f"--holdout")
        print(f"  （注意：当前全部样本在 train 里，"
              f"正式报数前应该用 DatasetBuilder 重新划分留出集）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
