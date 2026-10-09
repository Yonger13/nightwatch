#!/usr/bin/env python3
"""
Nightwatch · 每日分诊循环（轨道 B：真前瞻）

    # 跑一轮（抓新 issue → 预测 → 回填旧的对账）
    python scripts/daily_triage.py --once

    # 只回填，不抓新的（对账用）
    python scripts/daily_triage.py --reconcile-only

    # 看状态
    python scripts/daily_triage.py --status

==============================================================================
这个脚本和前面所有脚本的根本区别：**它是有状态的**
==============================================================================
之前每个脚本都是"跑一次、出个报告、结束" —— 那是**脚本**。

本脚本要长期跑（每天一次），所以必须解决四个"脚本不需要管"的问题：

    ① **幂等**       同一条 issue 不能被预测两次（否则重复计数）
    ② **增量**       每天只处理"上次之后新出现的"
    ③ **对账**       预测发出后，要回头查它的真实结果
    ④ **可恢复**     某天失败（网络/额度/CI 挂了）不能损坏状态

**这四个问题就是"脚本"和"系统"的分界线。**

==============================================================================
状态设计
==============================================================================
用一张 SQLite 表存所有预测：

    predictions(
        issue_id      主键（幂等的依据）
        repo
        predicted_at  预测时间
        predicted     预测出的粗标签（JSON 数组）
        predicted_raw 模型原始输出（留着排查用）
        gold          ★ 对账结果（NULL = 还没对账）
        gold_at       对账时间
        lag_days      issue 创建 → 首个标签的间隔
        n_steps / tokens / error
    )

★ 为什么用 SQLite 而不是 JSON 文件？

  · **幂等**需要"按主键查存在性" —— 文件要全量加载再判断，O(n)
  · **并发**（Actions 可能重入）需要事务 —— 文件做不到
  · **查询**（"还有多少没对账"）—— 文件要遍历

  但更关键的理由是：**这个状态会跨几周累积**。
  用文件的话，每天重写一遍全量数据 —— 中间崩溃就会丢历史。
  **持久化状态该用数据库，这是不需要犹豫的。**
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nightwatch.config import load_settings                        # noqa: E402
from nightwatch.llm.client import build_llm                        # noqa: E402
from nightwatch.loop.react import TriageAgent                      # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore     # noqa: E402
from nightwatch.models import (                                    # noqa: E402
    Issue, IssueState, MemoryRecord, MemoryTier, Outcome,
)
from nightwatch.tools.registry import IssueRepository, ToolRegistry  # noqa: E402

from collect_issues import GITHUB_TOKEN, GitHubClient              # noqa: E402
from collect_prospective import TYPE_LABELS, fetch_first_type_label  # noqa: E402
from eval_prospective import normalize_pred, to_coarse             # noqa: E402
from nightwatch.memory.feedback import MisuseTracker               # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}

DEFAULT_DB = ROOT / "data" / "daily_triage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    issue_id      TEXT PRIMARY KEY,
    repo          TEXT NOT NULL,
    title         TEXT,
    created_at    TEXT,
    predicted_at  TEXT NOT NULL,
    predicted     TEXT,           -- JSON 数组（粗标签）
    predicted_raw TEXT,           -- 模型原始输出
    n_steps       INTEGER DEFAULT 0,
    tokens        INTEGER DEFAULT 0,
    error         TEXT DEFAULT '',
    -- 对账字段（NULL = 还没对账）
    gold          TEXT,
    gold_at       TEXT,
    first_type_at TEXT,
    lag_days      REAL,
    -- 结果反馈闭环：这次判断用到了哪些记忆（JSON 数组）
    --
    -- ★ 为什么要存下来？
    --   对账是**几天后**才做的（--min-age-days 3），那时 trace 早就不在了。
    --   所以必须在预测的当下把它记下来 —— **数据要在产生的时候存**，
    --   不能指望"需要的时候再去取"。
    memory_used   TEXT
);
CREATE INDEX IF NOT EXISTS idx_gold ON predictions(gold);
CREATE INDEX IF NOT EXISTS idx_pred_at ON predictions(predicted_at);

-- 每次运行留一行日志 —— 便于回答"某天为什么没跑"
CREATE TABLE IF NOT EXISTS runs (
    run_at     TEXT PRIMARY KEY,
    mode       TEXT,
    n_new      INTEGER DEFAULT 0,
    n_reconciled INTEGER DEFAULT 0,
    note       TEXT DEFAULT ''
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


class Store:
    """状态存储（SQLite）。

    ★ 所有写操作都包在事务里 —— 因为 GitHub Actions 可能重入
      （比如上一次超时被杀、这次又启动），
      没有事务的话会出现"同一 issue 有两条预测"。
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """增量迁移（同 store.py 的做法）。

        `CREATE TABLE IF NOT EXISTS` 只在表不存在时生效 ——
        已有的库不会获得新列。加 `memory_used` 时必须走 ALTER。
        """
        cur = self.conn.execute("PRAGMA table_info(predictions)")
        existing = {row[1] for row in cur.fetchall()}
        if "memory_used" not in existing:
            self.conn.execute(
                "ALTER TABLE predictions ADD COLUMN memory_used TEXT")
            print("  [state] 已迁移：predictions 表补上 memory_used 列")

    def has(self, issue_id: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM predictions WHERE issue_id=?", (issue_id,))
        return cur.fetchone() is not None

    def add(self, rec: dict) -> bool:
        """写入一条预测。已存在则跳过（**幂等**）。

        返回 True 表示新写入，False 表示已存在被跳过。
        """
        if self.has(rec["issue_id"]):
            return False
        self.conn.execute(
            "INSERT INTO predictions (issue_id, repo, title, created_at,"
            " predicted_at, predicted, predicted_raw, n_steps, tokens, error,"
            " memory_used)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rec["issue_id"], rec["repo"], rec["title"][:200],
             rec.get("created_at", ""), now_iso(),
             json.dumps(rec["predicted"], ensure_ascii=False),
             json.dumps(rec.get("predicted_raw", []), ensure_ascii=False),
             rec.get("n_steps", 0), rec.get("tokens", 0),
             rec.get("error", ""),
             json.dumps(rec.get("memory_used", []), ensure_ascii=False)))
        self.conn.commit()
        return True

    def pending(self, min_age_days: int = 3) -> list[dict]:
        """取"该对账但还没对账"的记录。

        ★ `min_age_days` 的意义：issue 刚创建时还没标签，
          立刻去查必然是空的。等几天再查命中率更高。

          实测 pandas 的分诊延迟中位是 **0 天**（很快），
          所以 3 天已经足够宽松 —— 这个值来自 `collect_prospective.py`
          的实测数据，不是拍脑袋。
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=min_age_days)).isoformat(timespec="seconds")
        cur = self.conn.execute(
            "SELECT * FROM predictions WHERE gold IS NULL AND predicted_at < ?",
            (cutoff,))
        return [dict(r) for r in cur.fetchall()]

    def reconcile(self, issue_id: str, gold: str, first_type_at: str,
                  lag_days: float) -> None:
        self.conn.execute(
            "UPDATE predictions SET gold=?, gold_at=?, first_type_at=?,"
            " lag_days=? WHERE issue_id=?",
            (gold, now_iso(), first_type_at, lag_days, issue_id))
        self.conn.commit()

    def log_run(self, mode: str, n_new: int, n_recon: int,
                note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO runs (run_at, mode, n_new, n_reconciled, note)"
            " VALUES (?,?,?,?,?)",
            (now_iso(), mode, n_new, n_recon, note))
        self.conn.commit()

    def stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT COUNT(*) n,"
            " SUM(CASE WHEN gold IS NOT NULL THEN 1 ELSE 0 END) reconciled,"
            " SUM(CASE WHEN error != '' THEN 1 ELSE 0 END) errored"
            " FROM predictions")
        r = dict(cur.fetchone())
        cur2 = self.conn.execute("SELECT * FROM runs ORDER BY run_at DESC LIMIT 5")
        r["recent_runs"] = [dict(x) for x in cur2.fetchall()]
        return r

    def all_with_gold(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM predictions WHERE gold IS NOT NULL")
        return [dict(r) for r in cur.fetchall()]


# =============================================================================
# 三个动作：抓新的 / 预测 / 对账
# =============================================================================

def fetch_recent_issues(client: GitHubClient, repo: str, days: int,
                        limit: int) -> list[dict]:
    """抓最近 N 天新开的 issue（增量）。"""
    earliest = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []
    page = 1
    while len(out) < limit and page <= 15:
        batch = client.get(f"/repos/{repo}/issues", {
            "state": "all", "sort": "created", "direction": "desc",
            "per_page": 100, "page": page,
        })
        if not batch:
            break
        stop = False
        for it in batch:
            if "pull_request" in it:
                continue
            created = datetime.fromisoformat(it["created_at"].replace("Z", "+00:00"))
            if created < earliest:
                stop = True
                break
            out.append(it)
            if len(out) >= limit:
                break
        if stop:
            break
        page += 1
    return out[:limit]


def build_memory(db_path: Path | None = None) -> MemoryStore:
    """构造**持久**记忆库。

    ============================================================
    ★★ 为什么必须持久化（这是结果反馈闭环的前提）
    ============================================================
    第一版用的是 `:memory:` —— 每次运行都新建一个空库。
    这对"当天跑一次"没问题，但**结果反馈闭环完全无法工作**：

        误诊计数（misuse_count）记在**记忆对象**上，
        库一关就没了 —— 明天重新建库，计数从 0 开始。

    **跨天积累**是这个机制的全部意义所在：
      "一条记忆被用了 3 次、错了 2 次" —— 这个判断需要历史。

    所以记忆库必须落到磁盘（`data/memory.db`）。

    ★ 种子记忆只在**首次建库**时插入（靠幂等键去重）。
      这样重复启动不会重复插入，也不需要额外的"是否初始化过"标记 ——
      **复用幂等机制，比加一个新状态更简单。**
    """
    if db_path is None:
        db_path = ROOT / "data" / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(db_path), encoder=HashEmbedding(dim=256))
    seeds = [
        ("s1", "CI 构建超时失败", ["Build"], "Build"),
        ("s2", "文档示例无法运行", ["Docs"], "Docs"),
        ("s3", "内存占用过高", ["Performance"], "Performance"),
        ("s4", "merge 结果行数异常", ["merge"], "Bug"),
        ("s5", "新功能请求支持某参数", ["enhancement"], "Enhancement"),
        ("s6", "read_csv 读大文件 OOM", ["memory"], "Performance"),
        ("s7", "测试偶发失败", ["testing"], "Bug"),
        ("s8", "API 文档描述有误", ["Docs"], "Docs"),
    ]
    for mid, symptom, tags, res in seeds:
        store.insert(MemoryRecord(
            memory_id=mid, tier=MemoryTier.EPISODIC, issue_id=f"#{mid}",
            symptom=symptom, symptom_tags=tags, resolution=res,
            outcome=Outcome.RESOLVED))
    return store


def predict_one(raw: dict, store: MemoryStore, llm, settings, repo: str) -> dict:
    """对一条 issue 跑预测（信息隔离同 `eval_prospective`）。"""
    issue = Issue(
        issue_id=f"#{raw['number']}",
        title=raw.get("title", ""),
        body=(raw.get("body") or "")[:4000],
        author=(raw.get("user") or {}).get("login", ""),
        repo=repo,
        labels=[],
        state=IssueState.OPEN,
        gold_labels=[],
    )
    reg = ToolRegistry(IssueRepository([]), store=store,
                       current_issue_id=issue.issue_id)
    agent = TriageAgent(reg, llm=llm, store=store, settings=settings)
    try:
        # ★ prompt_variant="tuned" 必须显式传 —— 详见 eval_prospective.py
        #   里同名的注释（"配置不一致"那个坑）。
        r = agent.triage(issue, prompt_variant="tuned")
        tr = r.trace
        raw_pred = list(tr.predicted_labels or [])
        return {
            "issue_id": issue.issue_id, "repo": repo,
            "title": issue.title, "created_at": raw.get("created_at", ""),
            "predicted": normalize_pred(raw_pred),
            "predicted_raw": raw_pred,
            "n_steps": tr.n_steps, "tokens": tr.total_tokens, "error": "",
            # ★ 结果反馈闭环的数据来源：这次判断真的用到了哪些记忆
            "memory_used": list(getattr(tr, "memory_used", []) or []),
        }
    except Exception as e:                                  # noqa: BLE001
        return {
            "issue_id": issue.issue_id, "repo": repo, "title": issue.title,
            "created_at": raw.get("created_at", ""),
            "predicted": [], "predicted_raw": [],
            "n_steps": 0, "tokens": 0,
            "error": f"{type(e).__name__}: {str(e)[:120]}",
        }


def do_reconcile(store: Store, client: GitHubClient, repo: str,
                 min_age_days: int, mem_store=None) -> int:
    """对账：查已预测 issue 的真实标签。

    ============================================================
    ★★ 这里是"结果反馈闭环"的接入点
    ============================================================
    对账有**两个产出**，第二个才是闭环的关键：

        ① 更新 gold —— 这是**评测**：知道"预测准不准"
        ② 记录误诊 —— 这是**反馈**：知道"哪些记忆在误导我们"

    第一版只做了 ①。但"预测错了"这个事实**本身就是一个信号**：
    如果这次判断用到了某条记忆，而最终结论是错的 ——
    那条记忆**有嫌疑**（它可能已经过时了）。

    **这个信号是 `stale.py` 的路径交集抓不到的** ——
    因为它不看代码，只看结果（详见 feedback.py 的模块文档）。

    ★ 关键：只有 `memory_used` 非空时才记录。
      Agent 可能完全靠自己判断（没用到任何记忆），
      那这次错误**和记忆无关**，不该牵连任何记忆 ——
      **冤枉记忆比漏掉更糟**（会把好记忆推进复核队列，浪费人的时间）。
    """
    pending = store.pending(min_age_days=min_age_days)
    if not pending:
        print(f"  （没有待对账的记录）")
        return 0

    print(f"  待对账 {len(pending)} 条\n")
    n_ok = 0
    for i, rec in enumerate(pending, 1):
        num = rec["issue_id"].lstrip("#")
        try:
            tl = fetch_first_type_label(client, repo, int(num))
        except Exception as e:                              # noqa: BLE001
            print(f"  [{i}/{len(pending)}] {rec['issue_id']} "
                  f"{C['red']}查询失败{C['reset']} {str(e)[:40]}")
            continue
        ft = tl.get("first_type_label")
        if not ft:
            # ★ 还没打标签 → **保持 NULL**，下次再查。
            #   不能写成"失败"或"未知" —— 那会让"尚未分诊"和"分诊结果未知"
            #   混在一起，而它们的含义完全不同。
            print(f"  [{i}/{len(pending)}] {rec['issue_id']} "
                  f"{C['dim']}维护者还没打类型标签（保持待对账）{C['reset']}")
            continue

        created = datetime.fromisoformat(
            (rec.get("created_at") or "").replace("Z", "+00:00")) \
            if rec.get("created_at") else None
        lag = 0.0
        if created and ft.get("at"):
            lag = (datetime.fromisoformat(ft["at"].replace("Z", "+00:00"))
                   - created).total_seconds() / 86400

        store.reconcile(rec["issue_id"], ft["label"], ft.get("at", ""), lag)
        pred = json.loads(rec.get("predicted") or "[]")
        hit = ft["label"] in pred
        mark = f"{C['green']}✓{C['reset']}" if hit else f"{C['red']}✗{C['reset']}"

        # ★★ 结果反馈：预测错 → 记录误诊（牵连这次用到的记忆）
        fb_note = ""
        if not hit and mem_store is not None:
            used = json.loads(rec["memory_used"] or "[]") if rec.get("memory_used") else []
            if used:
                tracker = MisuseTracker(mem_store)
                flagged = tracker.record_triage_result(
                    issue_id=rec["issue_id"], memory_used=used, correct=False,
                    evidence=f"预测 {pred}，真实 {ft['label']}")
                fb_note = f"　{C['yellow']}↑牵连 {len(used)} 条记忆" + (
                    f"，{len(flagged)} 条新进复核队列{C['reset']}" if flagged
                    else f"{C['reset']}")

        print(f"  [{i}/{len(pending)}] {mark} {rec['issue_id']:<9}"
              f"预测={str(pred):<22} 真实={ft['label']}{fb_note}")
        n_ok += 1
    return n_ok


def show_status(store: Store) -> None:
    s = store.stats()
    print(f"""
  累计预测      {s.get('n') or 0}
  已对账        {s.get('reconciled') or 0}
  执行异常      {s.get('errored') or 0}

  最近几次运行：""")
    for r in s.get("recent_runs", []):
        print(f"    {r['run_at']}  {r['mode']:<16}"
              f" 新 {r['n_new']}  对账 {r['n_reconciled']}")

    rows = store.all_with_gold()
    if rows:
        hits = sum(1 for r in rows
                   if r["gold"] in json.loads(r["predicted"] or "[]"))
        print(f"""
  {C['bold']}★ 对账结果（这才是能对外说的数字）{C['reset']}
    样本        {len(rows)}
    命中率      {C['green']}{hits / len(rows):.1%}{C['reset']}（{hits}/{len(rows)}）
    {C['dim']}  这是"用创建时信息预测 → 对照维护者真实标签"的结果，
      不是离线 gold 标签上算的 F1。{C['reset']}""")


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="每日分诊循环（轨道 B）")
    ap.add_argument("--repo", default="pandas-dev/pandas")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--once", action="store_true", help="跑一轮")
    ap.add_argument("--reconcile-only", action="store_true", help="只对账")
    ap.add_argument("--status", action="store_true", help="看状态")
    ap.add_argument("--mock", action="store_true", help="用 Mock（不调真实模型）")
    ap.add_argument("--new-days", type=int, default=7,
                    help="抓最近多少天新开的 issue")
    ap.add_argument("--limit", type=int, default=20, help="本轮最多预测几条")
    ap.add_argument("--min-age-days", type=int, default=3,
                    help="预测后至少等几天才去对账")
    ap.add_argument("--memory-db", default=str(ROOT / "data" / "memory.db"),
                    help="持久记忆库路径（结果反馈闭环需要它跨天积累）")
    args = ap.parse_args()

    store = Store(Path(args.db))
    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 每日分诊循环{C['reset']}")
    print(f"  仓库 {args.repo}　状态库 {Path(args.db).name}")

    if args.status:
        hr("状态")
        show_status(store)
        return 0

    real = not args.mock
    client = GitHubClient()
    n_new = n_recon = 0

    # ---------------------------------------------------------- 预测新 issue
    if not args.reconcile_only:
        hr("① 抓最近新开的 issue（增量）")
        raw_issues = fetch_recent_issues(client, args.repo, args.new_days,
                                         args.limit * 2)
        # ★ 幂等过滤：已经在库里的跳过
        fresh = [r for r in raw_issues if not store.has(f"#{r['number']}")]
        print(f"  抓到 {len(raw_issues)} 条，其中 {C['bold']}{len(fresh)}{C['reset']} "
              f"条是新的（{len(raw_issues) - len(fresh)} 条已预测过，跳过）")

        if fresh:
            fresh = fresh[:args.limit]
            hr("② 跑预测")
            settings = load_settings()
            mem = build_memory(Path(args.memory_db))
            llm = build_llm(use_mock=args.mock)
            for i, raw in enumerate(fresh, 1):
                rec = predict_one(raw, mem, llm, settings, args.repo)
                added = store.add(rec)
                n_new += 1 if added else 0
                print(f"  [{i}/{len(fresh)}] {rec['issue_id']:<9}"
                      f"预测={str(rec['predicted']):<24}"
                      f"{rec['title'][:34]}")
        else:
            print(f"  {C['dim']}（没有新的 issue 需要预测）{C['reset']}")

    # ---------------------------------------------------------- 对账
    hr("③ 对账（查已预测 issue 的真实标签）")
    # ★ 传持久记忆库进去 —— 反馈闭环需要它来记误诊
    mem_store_fb = build_memory(Path(args.memory_db))
    n_recon = do_reconcile(store, client, args.repo, args.min_age_days,
                           mem_store=mem_store_fb)

    store.log_run("once" if not args.reconcile_only else "reconcile",
                  n_new, n_recon)

    # ---------------------------------------------------------- 状态
    hr("④ 状态")
    show_status(store)

    hr()
    print(f"""
  {C['green']}✓{C['reset']} 本轮完成：新增预测 {n_new}，完成对账 {n_recon}

  {C['dim']}挂到 GitHub Actions 后，这个脚本每天自动跑一次。
    要"跑两周"的不是脚本，**是它的状态库在积累** ——
    这正是"脚本"和"系统"的区别。{C['reset']}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
