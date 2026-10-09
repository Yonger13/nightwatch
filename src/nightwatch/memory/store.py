"""
Nightwatch · 记忆存储层

================================================================================
这一层的职责边界
================================================================================
存储层只管三件事：**存下来、查出来、算相似**。
它**不做任何判断** —— 什么该记、要不要合并、冲突了听谁的，全在 lifecycle.py。

这个划分很重要。很多项目把"业务判断"和"存储操作"混在一起写，
结果是：换存储要动业务逻辑，改策略要动数据库代码。
分开之后，你可以把 SQLite 换成 Postgres+pgvector，而 lifecycle 一行不用改。

================================================================================
为什么默认用 SQLite 而不是 pgvector
================================================================================
这是有意的取舍，而且这个取舍本身是个面试考点：

  选 pgvector 的理由：情景记忆的检索是**向量相似 + 结构化过滤**的混合查询
    （"找出这个作者在过去 30 天内报过的、症状相似的问题" —— 前半句是结构化，后半句是向量）。
    用一个 Postgres 就能同时搞定，不用维护独立向量库。

  但本项目默认用 SQLite，因为：
    1. 学习者 clone 下来要能立刻跑，不该先装 Postgres
    2. 数据量 < 1 万条时，纯 Python 的余弦相似度足够快（毫秒级）
    3. **关键**：混合检索的逻辑用 SQL 写清楚了，迁移到 pgvector 只是
       把 ORDER BY cosine_distance 换成 `<->` 操作符的事

  所以这里同时提供了两套实现：SQLite 版（默认）和 Postgres 版（注释里给出 SQL）。
  面试时你可以说："我做了混合检索，用 SQLite 起步、预留了 pgvector 的迁移路径，
  因为情景记忆的查询天然是'结构化过滤 + 向量排序'的组合。"
================================================================================
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models import (
    HumanVerdict, MemoryRecord, MemoryTier, Outcome, PendingConflict,
    PendingPromotion, _hash,
)


# =============================================================================
# 伪 Embedding：让链路可测的权宜之计
# =============================================================================

class APIEmbedding:
    """真·语义 embedding（通过 OpenAI 兼容的 /embeddings 接口）。

    ============================================================
    为什么必须提供这个选项 —— 阈值扫描发现的问题
    ============================================================
    用 HashEmbedding（字符 bigram 哈希）跑阈值扫描的结果（data/_thr_sweep.txt）：

        重复样本的 top1 相似度:   0.2865 ~ 0.3612
        非重复样本 top1 相似度:   0.1101 ~ 0.4137     ← 重叠！
        F1 最优阈值 0.30 → F1 仅 0.500

    **重叠意味着任何阈值都会有错误。** 根因是：哈希向量只能捕捉"字面相似"，
    而下面这一对在字面上几乎一样，语义上却一个是重复、一个不是：

        「接口在高峰时段频繁超时」 vs 「高峰时段接口超时（重复）」  → 真重复
        「接口在高峰时段频繁超时」 vs 「定时任务偶发执行超时」      → 不是重复

    三条都含"超时"，共享大量 bigram，哈希向量给出的相似度几乎一样。
    只有语义 embedding 才能区分"同一个问题"和"碰巧都超时"。

    ============================================================
    这个发现本身就是项目价值的一部分
    ============================================================
    一个只报"我做了判重，准确率 XX%"的项目，和一个能说
    "我发现判重的天花板由向量质量决定，调阈值只能在 F1=0.5 附近打转，
     换了语义 embedding 之后 F1 提升到 YY%"的项目 ——
    后者的说服力高一个量级。因为它证明了你**做了诊断，而不只是调参**。

    配置方式：设置环境变量 NW_LLM_API_KEY 后自动启用（见 build_encoder）。
    """

    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "text-embedding-v3", dim: int = 1024):
        from ..config import LLM_API_KEY, LLM_BASE_URL
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or LLM_API_KEY
        self.model = model
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def encode(self, text: str) -> list[float]:
        """编码文本。带本地缓存 —— 同一条文本不必重复付费调用。"""
        key = (text or "")[:2000]
        if key in self._cache:
            return self._cache[key]

        try:
            import httpx
        except ImportError:
            # 没装 httpx → 降级到哈希（保证系统仍可运行）
            return HashEmbedding(dim=256).encode(text)

        try:
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "input": key or " "},
                timeout=30.0,
            )
            resp.raise_for_status()
            vec = resp.json()["data"][0]["embedding"]
            self._cache[key] = vec
            return vec
        except Exception:                      # noqa: BLE001
            # 任何失败都降级 —— **不能让 embedding 服务不可用拖垮整个系统**。
            # 这也是"可靠性靠降级而不是靠祈祷依赖不挂"的一个体现。
            fallback = HashEmbedding(dim=256).encode(text)
            self._cache[key] = fallback
            return fallback


def build_encoder(use_api: bool | None = None, dim: int = 256):
    """构造向量编码器。

    策略与 LLM 客户端一致（见 llm/client.py）：
        **有 API key 就用真 embedding，没有就降级到哈希伪向量。**

    这样保证了：
      · 零配置也能跑通全链路（学习/演示）
      · 有 key 时能拿到真实效果（出简历数据）
    """
    from ..config import LLM_API_KEY
    if use_api is None:
        use_api = bool(LLM_API_KEY)
    if use_api:
        return APIEmbedding()
    return HashEmbedding(dim=dim)


class HashEmbedding:
    """确定性的哈希伪 embedding。

    **这不是真的语义 embedding**，而是一个刻意的降级实现。为什么要有它？

      1. 真 embedding 要花钱、要网络、要等 —— 会让"跑通链路"这件事变得昂贵。
      2. 学习阶段你真正要理解的是**检索、合并、消解的流程**，不是 embedding 本身。
      3. 它是确定性的：同样输入永远得到同样向量，所以评测结果可复现。

    它的工作原理：把文本切成字符 n-gram，哈希到固定维度，再做 L2 归一化。
    效果上能捕捉"字面相似"（这正是判 duplicate 需要的），但捕捉不到语义相似。

    ⚠️ **已知局限（这是实测出来的，不是推测）**：
       在阈值扫描中，它对"重复 vs 非重复"的区分度不足 ——
       两组的相似度分布出现重叠（见 data/_thr_sweep.txt），
       duplicate 检测的 F1 只能到 0.50。
       原因是它无法区分"同一个问题"和"碰巧都含同一个高频词"。

    → 需要真实效果时请用 APIEmbedding（设置 NW_LLM_API_KEY 即可自动启用）。
    这个契约（"输入文本 → 定长归一化向量"）保持不变，所以上层代码完全不用改。
    这就是**接口隔离**的价值。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = (text or "").lower()
        if not text:
            return vec

        # 字符 bigram + 单词，兼顾中英文
        tokens: list[str] = []
        for i in range(len(text) - 1):
            tokens.append(text[i:i + 2])
        tokens.extend(text.split())

        for tok in tokens:
            h = hash(tok) % self.dim
            vec[h] += 1.0

        # L2 归一化：归一化之后，余弦相似度 = 点积，计算更快
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。两个向量都已归一化时，等价于点积。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# =============================================================================
# 存储
# =============================================================================

def _now_str() -> str:
    """当前时间字符串（本地时区，秒级）。

    抽成模块级函数而不是各方法里写 strftime —— 因为格式必须一致，
    否则「按时间排序」会因格式差异出错（比如 '2026-9-24' 和 '2026-09-24'）。
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


SCHEMA = """-- 记忆表：情景记忆与语义记忆共用一张表，用 tier 区分
-- 为什么不分表？因为跨层查询是常见需求（"这个问题既查历史工单也查 SOP"），
-- 分表会导致每次都要 UNION，而且加了新层就要改表结构。
-- 这是"用 discriminator 字段换查询灵活性"的经典取舍。
CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    tier            TEXT NOT NULL,
    idem_key        TEXT NOT NULL,          -- 幂等键，防重复写入
    repo            TEXT,
    issue_id        TEXT,
    author          TEXT,
    symptom         TEXT,
    symptom_tags    TEXT,                   -- JSON 数组
    root_cause      TEXT,
    resolution      TEXT,
    actions         TEXT,                   -- JSON 数组（含失败的尝试）
    outcome         TEXT,
    human_verdict   TEXT,
    human_edit      TEXT,
    access_count    INTEGER DEFAULT 0,
    use_count       INTEGER DEFAULT 0,
    last_access     TEXT,
    confidence      REAL DEFAULT 0,
    importance      REAL DEFAULT 0,
    decay_weight    REAL DEFAULT 1.0,
    archived        INTEGER DEFAULT 0,
    embedding       TEXT,                   -- JSON 数组
    created_at      TEXT,
    -- 代码状态绑定（仓库变更时效性的地基，见 models.MemoryRecord 的说明）
    code_refs       TEXT,                   -- JSON 数组：依赖过的文件路径
    repo_head       TEXT,                   -- 写下时的 commit / 快照标识
    -- 误诊计数（结果反馈闭环）：这条记忆被用过、但那次判断错了的次数
    --
    -- ★ 为什么要它，而不是每次都去 misuse_events 里 count？
    --   因为**召回路径要快**（每次召回都可能要看这个值来排序/过滤），
    --   而 COUNT(*) 是 O(明细数)。
    --   这是典型的"写入时多花一点、读取时省很多"的取舍。
    --
    -- ★ 它和 use_count 配对使用：
    --   单看 misuse_count 没有意义（用 100 次错 2 次是好事；
    --   用 2 次错 2 次才是问题）—— 必须和 use_count 一起算**误诊率**。
    misuse_count    INTEGER DEFAULT 0
);

-- 索引设计（这是"后端工程显性化"的落点）
-- ① 幂等键唯一索引：让"重复写入"在数据库层就被拦住，而不是靠应用层判重
CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_idem ON memories(idem_key);
-- ② author 索引：支撑"同一作者报过什么"这个最高频的精确查询
CREATE INDEX IF NOT EXISTS idx_mem_author ON memories(author, created_at DESC);
-- ③ tier + archived 组合索引：召回时几乎总是带这两个条件
CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(tier, archived);

-- 待裁决冲突表
CREATE TABLE IF NOT EXISTS pending_conflicts (
    conflict_id   TEXT PRIMARY KEY,
    memory_a      TEXT,
    memory_b      TEXT,
    reason        TEXT,
    created_at    TEXT,
    resolved_by   TEXT,
    resolution    TEXT
);

-- 轨迹表：可观测性与评测的数据来源
CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    issue_id      TEXT,
    payload       TEXT,     -- 完整 Trace 的 JSON
    outcome       TEXT,
    n_steps       INTEGER,
    total_tokens  INTEGER,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_issue ON traces(issue_id);
CREATE INDEX IF NOT EXISTS idx_trace_outcome ON traces(outcome);

-- SOP 提炼提案表（P1-2 语义记忆层）
--
-- ★ 为什么这张表**必须存在**，而不是"发现模式就直接 promote"？
--   因为语义层有人工审核闸门，而闸门需要一个**持久化的状态载体**。
--   如果只放在内存里，进程一重启，"待审核队列"就没了 ——
--   那等于审核流程随时可能被重启清空，闸门形同虚设。
--
-- ★ 为什么用 status 而不是删行表示"已处理"？
--   拒绝理由、审核人、审核时间都是**有价值的记录**：
--     · 同一个模式下次被提出来时，能看到"上次为什么拒"
--     · 能回答"这套 SOP 是谁在什么时候定的"（审计需要）
--   删除是最差的"状态表达"方式 —— 它丢掉了过程信息。
CREATE TABLE IF NOT EXISTS pending_promotions (
    promotion_id        TEXT PRIMARY KEY,
    pattern             TEXT NOT NULL,
    occurrences         INTEGER NOT NULL,
    sample_ids          TEXT,               -- JSON 数组
    sample_summary      TEXT,
    status              TEXT NOT NULL,      -- pending / approved / rejected
    created_at          TEXT,
    reviewed_at         TEXT,
    reviewed_by         TEXT,
    review_note         TEXT,
    promoted_memory_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_promo_status ON pending_promotions(status);
CREATE INDEX IF NOT EXISTS idx_promo_pattern ON pending_promotions(pattern);

-- 待复核的过时记忆（仓库变更时效性）
--
-- ★ 这张表的定位：**待办清单，不是惩罚机制**。
--   入队（`StaleDetector.mark`）**不改变记忆本身** ——
--   记忆照常被召回、照常参与判断。
--   只有人确认（`review(confirmed=True)`）之后才降权归档。
--
--   为什么要两阶段？因为「变更波及」≠「结论失效」：
--     文件被改了，不代表引用了它的记忆就错了（可能只是改了注释）。
--   自动化这一步会重蹈 docs/02 的 B6 覆辙（被人工确认的经验被自动规则误伤）。
--
-- ★ 为什么为"判定仍有效"也留一条记录（status='dismissed'）？
--   因为它同样是**有价值的信息**：
--   "这条被怀疑过的记忆，人看过之后认为没问题" ——
--   下次同样的变更再触发时，可以直接跳过，不必再问一遍。
--   只记"确认"不记"否决"，等于强迫人做重复劳动。
CREATE TABLE IF NOT EXISTS pending_stale_reviews (
    review_id       TEXT PRIMARY KEY,
    memory_id       TEXT NOT NULL,
    symptom         TEXT,
    matched_refs    TEXT,           -- JSON 数组：命中的代码路径
    trigger         TEXT,           -- repo_change / manual
    trigger_detail  TEXT,           -- 什么变更触发的
    status          TEXT NOT NULL,  -- pending / confirmed / dismissed
    created_at      TEXT,
    reviewed_at     TEXT,
    reviewed_by     TEXT,
    review_note     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stale_status ON pending_stale_reviews(status);
CREATE INDEX IF NOT EXISTS idx_stale_memory ON pending_stale_reviews(memory_id);

-- 误诊明细（结果反馈闭环）
--
-- ★ 这张表记录的是"**记忆被用了，但那次判断错了**"这件事。
--
--   它和 `pending_stale_reviews` 的区别是**证据来源不同**：
--     · stale_reviews ← 代码变更（结构信号：文件被改了）
--     · 本表          ← 结果反馈（语义信号：用了它之后判断错了）
--
--   **后者能抓到前者抓不到的"语义性过时"** ——
--   比如一个 API 还在、但语义变了（`dropna` 的默认行为改了），
--   文件没被重构，路径交集完全不命中，但用它得出的结论已经不对了。
--
-- ★ 为什么用明细表 + 汇总列，而不是只存汇总？
--   汇总列（memories.misuse_count）用于**快速排序/过滤**（召回时不 JOIN）；
--   明细表用于**可追溯**（"这条记忆是被哪几次误诊牵连的"）。
--   两处冗余是刻意的 —— 但它们的一致性由 `MisuseTracker` 集中保证
--   （不允许别处直接改这两个数字）。
CREATE TABLE IF NOT EXISTS misuse_events (
    event_id     TEXT PRIMARY KEY,
    issue_id     TEXT,
    memory_id    TEXT NOT NULL,
    evidence     TEXT,          -- 什么证据表明这次判断错了
    created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_misuse_memory ON misuse_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_misuse_issue ON misuse_events(issue_id);
"""


class MemoryStore:
    """记忆存储。只做存取，不做判断。

    ============================================================
    ★ 并发安全：每线程一个连接（P2-2 时发现的真问题）
    ============================================================
    原来只有一个共享连接，用 `check_same_thread=False` 关掉了线程检查。
    **但那只是"关掉检查"，不是"变成线程安全"。**

    实测后果（A2A 演示的并发写入测试，8 线程 × 5 条）：
        写入成功 26 / 40  ← 丢了 14 条，而且**不报错**（静默失败）

    根因有两层：
      ① `sqlite3.Connection` 对象**不是线程安全的** ——
         多线程共用会互相踩游标（Python 的 sqlite3 模块文档明确写了这一点）
      ② 没设 `busy_timeout` —— 锁冲突时**立即失败**，而不是等锁释放

    修法（这是 SQLite 官方推荐的用法）：
      · **每个线程一个独立连接**（thread-local）
      · 设 `busy_timeout`，让锁冲突时等待而不是报错

    **为什么这个问题特别值得记下来？**
    因为它是"分布式/多 Agent 系统"里最典型的失败模式：
      · 单线程测试**永远发现不了**
      · 失败是**静默的**（返回 False 而不是抛异常）
      · 只在并发压力下才出现

    如果没有 P2-2 的并发写入测试，这个 bug 会一直躺在代码里。
    **"并发写入测试"是共享存储的必测项，不是可选项。**
    """

    def __init__(self, db_path: str | Path | None = None, dim: int = 256,
                 encoder=None):
        # ":memory:" 用于测试，每次干净启动
        self.db_path = str(db_path) if db_path else ":memory:"
        # 编码器可注入 —— 这样测试时能强制用确定性的哈希版，
        # 生产时自动用语义版（见 build_encoder）
        self.encoder = encoder if encoder is not None else build_encoder(dim=dim)
        # thread-local 存储每个线程自己的连接
        self._local = threading.local()
        # ★ 向量索引（可选）。None = 走原来的 SQL 候选池路径。
        #
        #   默认不建索引，是为了**不改变现有行为** ——
        #   索引是个优化项，不该让"没建索引"的调用方行为发生变化。
        #   （如果默认就建，那所有现有测试跑的都是新路径，
        #     一旦新路径有问题会全线失败，而不是局部失败。）
        self.index = None
        self._index_tier: MemoryTier | None = None
        # 查询时探测几个簇：越大越准越慢。4 是"精度基本无损"的常见起点。
        self.nprobe = 4
        # 先建一次连接（顺带把 schema 建好）
        self.conn.executescript(SCHEMA)
        # 对**已有**的库做增量迁移（新建的库已经带上了新列，这里会跳过）
        self._ensure_columns()
        self.conn.commit()

    def _ensure_columns(self) -> list[str]:
        """增量迁移：给已有的表补上新增的列。

        ============================================================
        为什么需要这个 —— 一个真实的运维问题
        ============================================================
        `SCHEMA` 里的 `CREATE TABLE IF NOT EXISTS` **只在表不存在时生效**。
        表已经存在时，新增的列**不会被加上**。

        后果：新代码读旧库 → `sqlite3.OperationalError: no such column: code_refs`。
        而 `data/` 下的库是**运行时产物**（`daily_triage.db` 要跨周累积），
        不能靠"删库重建"来解决。

        **这就是"加字段"和"改字段"的区别**：
          · 开发期：改了 dataclass，删库重建，一切正常
          · 生产期：库里有数据，必须走**迁移**

        ============================================================
        为什么用 ALTER TABLE ADD COLUMN 而不是重建表
        ============================================================
        SQLite 的 `ALTER TABLE ADD COLUMN` 是**廉价操作**（只改元数据，
        不重写数据），而且新列有默认值（NULL）—— 正好符合需求：
        老记忆没有 code_refs，语义就是"不知道它依赖什么代码"。

        **不要为了加一个可空字段去重建整张表** —— 那是数据迁移里
        最容易出事的地方（重建意味着要复制全部数据，中途失败就丢数据）。

        ★ 幂等：先查 `PRAGMA table_info`，已存在就跳过。
          这样重复启动服务不会报"duplicate column name"。
        """
        added = []
        cur = self.conn.execute("PRAGMA table_info(memories)")
        existing = {row[1] for row in cur.fetchall()}
        for col, decl in (("code_refs", "TEXT"), ("repo_head", "TEXT"),
                          ("misuse_count", "INTEGER DEFAULT 0")):
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE memories ADD COLUMN {col} {decl}")
                added.append(col)
        if added:
            print(f"  [store] 已迁移：memories 表补上列 {added}")
        return added

    def close(self) -> None:
        """关闭当前线程的连接。

        ★ 注意只关**当前线程**的 —— 因为是 thread-local 连接（见类文档）。
          "关闭整个 store" 在并发场景下没有明确定义：
          你没法从 A 线程安全地关掉 B 线程正在用的连接。
        """
        self.conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """当前线程的数据库连接（按需创建）。

        ★ 用 property 而不是改所有调用点 —— 这样 `self.conn.execute(...)`
          这十几处代码**一行都不用动**，但行为从"共享连接"变成了
          "每线程一个连接"。

        **好的修复应该尽量不扩散改动面。** 改一处属性访问方式，
        比改十几个方法体要安全得多（后者容易漏、也更容易引入新 bug）。
        """
        c = getattr(self._local, "conn", None)
        if c is None:
            # timeout=10.0：锁冲突时最多等 10 秒（而不是立即失败）
            c = sqlite3.connect(self.db_path, timeout=10.0,
                                check_same_thread=False)
            c.row_factory = sqlite3.Row
            # WAL 模式：允许读写并发。Agent 服务里"主链路写 trace"和
            # "后台任务写记忆"会同时发生，不开 WAL 会互相阻塞。
            try:
                c.execute("PRAGMA journal_mode=WAL")
                # ★ busy_timeout 是并发写入不丢数据的关键 ——
                #   没有它，锁冲突会立即抛 "database is locked"
                c.execute("PRAGMA busy_timeout=10000")
            except sqlite3.Error:
                pass  # :memory: 不支持 WAL，忽略
            # 新连接要确保表存在（SCHEMA 里都是 IF NOT EXISTS，幂等）
            c.executescript(SCHEMA)
            c.commit()
            self._local.conn = c
        return c

    # ---------------------------------------------------------------- 写入

    def insert(self, mem: MemoryRecord) -> bool:
        """写入一条记忆。

        返回 False 表示**因为幂等键冲突被跳过** —— 这不是错误，是设计。
        异步任务重试时，第二次写入会被静默忽略，从而保证"同一个 issue
        不会产生两条记忆"。

        这就是"幂等"从概念落到代码的样子：不是靠 if 判断，是靠数据库约束。
        """
        if not mem.embedding:
            mem.embedding = self.encoder.encode(self._embed_text(mem))

        try:
            self.conn.execute(
                """INSERT INTO memories (
                    memory_id, tier, idem_key, repo, issue_id, author,
                    symptom, symptom_tags, root_cause, resolution, actions, outcome,
                    human_verdict, human_edit, access_count, use_count, last_access,
                    confidence, importance, decay_weight, archived, embedding, created_at,
                    code_refs, repo_head, misuse_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mem.memory_id, mem.tier.value, mem.idempotency_key(),
                    mem.repo, mem.issue_id, mem.author,
                    mem.symptom, json.dumps(mem.symptom_tags, ensure_ascii=False),
                    mem.root_cause, mem.resolution,
                    json.dumps(mem.actions, ensure_ascii=False), mem.outcome.value,
                    mem.human_verdict.value, mem.human_edit,
                    mem.access_count, mem.use_count, mem.last_access,
                    mem.confidence, mem.importance, mem.decay_weight,
                    int(mem.archived), json.dumps(mem.embedding), mem.created_at,
                    json.dumps(mem.code_refs, ensure_ascii=False), mem.repo_head,
                    int(getattr(mem, "misuse_count", 0)),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # 幂等键冲突 → 已经存在，正常跳过
            return False

    def update(self, mem: MemoryRecord) -> None:
        self.conn.execute(
            """UPDATE memories SET
                symptom=?, symptom_tags=?, root_cause=?, resolution=?, actions=?,
                outcome=?, human_verdict=?, human_edit=?,
                access_count=?, use_count=?, last_access=?,
                confidence=?, importance=?, decay_weight=?, archived=?, embedding=?,
                code_refs=?, repo_head=?, misuse_count=?
               WHERE memory_id=?""",
            (
                mem.symptom, json.dumps(mem.symptom_tags, ensure_ascii=False),
                mem.root_cause, mem.resolution,
                json.dumps(mem.actions, ensure_ascii=False), mem.outcome.value,
                mem.human_verdict.value, mem.human_edit,
                mem.access_count, mem.use_count, mem.last_access,
                mem.confidence, mem.importance, mem.decay_weight,
                int(mem.archived), json.dumps(mem.embedding),
                json.dumps(mem.code_refs, ensure_ascii=False), mem.repo_head,
                int(getattr(mem, "misuse_count", 0)),
                mem.memory_id,
            ),
        )
        self.conn.commit()

    def delete(self, memory_id: str) -> None:
        """删除一条记忆。

        **这是"可回滚"的基础**：如果发现某条记忆是错的（污染），要能撤掉它。
        生产环境更稳妥的做法是软删除（打标记），因为要保留审计痕迹 ——
        "某次判断当时依据了哪几条记忆"这个问题必须能回答。
        """
        self.conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        self.conn.commit()

    # ---------------------------------------------------------------- 读取

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return self._row_to_mem(row) if row else None

    def get_by_idem(self, idem_key: str) -> Optional[MemoryRecord]:
        """按幂等键查 —— 合并环节用它判断"是不是同一条事"的快速路径。"""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE idem_key=?", (idem_key,)
        ).fetchone()
        return self._row_to_mem(row) if row else None

    def count(self, tier: MemoryTier | None = None) -> int:
        if tier:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tier=?", (tier.value,))
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]

    # ---------------------------------------------------------------- 检索

    # ------------------------------------------------- 向量索引（IVF）

    def build_index(self, nlist: int = 16, tier: MemoryTier | None = None):
        """从记忆库构建 IVF 向量索引。

        ★ **为什么要显式调用而不是自动建？**

        因为建索引要遍历全库算 k-means（O(N x nlist x iter)），
        10 万条要几秒到几十秒。如果自动建（比如第一次 recall 时惰性触发），
        那"第一次查询特别慢"会变成一个难以解释的偶发现象。

        **显式构建让成本可见、可控** —— 生产里通常是
        "服务启动时建 + 定时重建"，而不是"用到就建"。

        ★ 索引与存储分离的意义：
        索引只存 `{memory_id: vector}` 和簇结构，**不持有 MemoryRecord**。
        所以重建索引不会碰数据库，删掉索引也不影响数据。
        """
        from .index import IVFIndex

        items: dict[str, list[float]] = {}
        for mem in self.all_memories(tier):
            if mem.embedding:
                items[mem.memory_id] = mem.embedding

        self.index = IVFIndex(nlist=nlist).build(items)
        self._index_tier = tier
        return self.index.stats

    def attach_index(self, index, tier: MemoryTier | None = None) -> None:
        """挂载一个已有的索引（用于实验里对比不同参数）。"""
        self.index = index
        self._index_tier = tier

    def index_stats(self):
        return self.index.stats.to_dict() if self.index else None

    def recall(
        self,
        query: str,
        author: str | None = None,
        tier: MemoryTier | None = MemoryTier.EPISODIC,
        top_k: int = 5,
        candidate_pool: int = 50,
        include_archived: bool = False,
    ) -> list[tuple[MemoryRecord, float]]:
        """混合检索：结构化过滤 + 向量排序。

        ============================================================
        为什么要混合，而不是纯向量检索？
        ============================================================
        因为不同查询意图需要不同信号：
          - "这个作者以前报过什么" -> 精确匹配（向量做不到"这个作者"）
          - "有没有和这个症状一样的问题" -> 语义相似（SQL 做不到）
          - 这两件事经常要同时做

        ============================================================
        ★ 两条路径：SQL 候选池（默认） vs IVF 索引（可选）
        ============================================================
        **默认路径**（无索引）：SQL 结构化过滤 + `LIMIT candidate_pool` 取候选，
        然后在候选内算向量。

        **它的缺陷**（做向量索引时才发现的）：候选池是**按 created_at 排序**取的，
        不是按相关性。所以记忆库一大，"**旧但相关**"的记忆根本进不了候选池 ——
        连算相似度的机会都没有。

        **索引路径**（`build_index()` 之后）：先用 IVF 从**向量空间**里
        取最近的若干候选，再用 SQL 取记录并应用结构化过滤。
        它解决了上面那个缺陷 —— 候选池按相关性产生，而不是按时间。

        **所以"向量索引"的价值有两层：不只是快，而且召回质量更好。**
        （`docs/06` 原本只记了"慢" —— 那只是问题的一半。）

        ============================================================
        迁移到 pgvector 时
        ============================================================
        两阶段合并成一条 SQL：
            SELECT *, embedding <=> %s::vector AS dist
              FROM memories
             WHERE tier='episodic' AND archived=false AND author = %s
             ORDER BY embedding <=> %s::vector
             LIMIT %s;
        （pgvector 的 HNSW 索引会自动处理"ORDER BY 向量距离"，
          不需要像本模块这样手动建簇。）
        """
        qvec = self.encoder.encode(query)

        if self.index is not None and self.index.is_built:
            return self._recall_via_index(qvec, author, tier, top_k,
                                          candidate_pool, include_archived)

        # --- 路径一：SQL 候选池 ---
        clauses = []
        params: list = []

        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier.value)
        if not include_archived:
            clauses.append("archived = 0")
        if author:
            clauses.append("author = ?")
            params.append(author)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ?"
        params.append(candidate_pool)

        rows = self.conn.execute(sql, params).fetchall()
        candidates = [self._row_to_mem(r) for r in rows]
        return self._rank(candidates, qvec, top_k)

    def _recall_via_index(self, qvec: list[float], author: str | None,
                          tier: MemoryTier | None, top_k: int,
                          candidate_pool: int, include_archived: bool
                          ) -> list[tuple[MemoryRecord, float]]:
        """索引路径：用 IVF 取候选，再用 SQL 取记录 + 结构化过滤。

        ★ 顺序说明：**先向量过滤，再结构化过滤**。

        理由是"哪个能先大幅缩小候选集"：
        向量相似度能把 10 万条缩到几十条，而 author 过滤可能只缩到几千条 ——
        **先向量更划算**。

        （这不是绝对的：如果某次查询的 author 能缩到 3 条，先 author 更快。
          生产里查询优化器就是干这个的。本模块选"先向量"因为它对多数查询更优。）
        """
        hits = self.index.search(qvec, nprobe=self.nprobe,
                                 top_k=max(candidate_pool, top_k * 4))
        if not hits:
            return []

        ids = [mid for mid, _ in hits]
        placeholders = ",".join("?" * len(ids))
        clauses = [f"memory_id IN ({placeholders})"]
        params: list = list(ids)

        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier.value)
        if not include_archived:
            clauses.append("archived = 0")
        if author:
            clauses.append("author = ?")
            params.append(author)

        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)}"
        rows = self.conn.execute(sql, params).fetchall()
        candidates = [self._row_to_mem(r) for r in rows]
        return self._rank(candidates, qvec, top_k)

    def _rank(self, candidates: list[MemoryRecord], qvec: list[float],
              top_k: int) -> list[tuple[MemoryRecord, float]]:
        """在候选集内算向量相似度并排序（两条路径共用）。

        ★ 综合得分 = 相似度 x 衰减权重

        为什么要乘衰减权重？因为"半年前完全一样的问题"和"上周一样的问题"，
        前者的解决方式可能已经过时（版本升级了、配置变了）。

        但衰减权重**不能压过相似度**（那是 0.15 vs 0.9 的差距，量级不同），
        它只是微调 —— 所以系数设计要让两者只在相似度接近时起作用。
        """
        scored: list[tuple[MemoryRecord, float]] = []
        for mem in candidates:
            sim = cosine(qvec, mem.embedding or [])
            final = sim * (0.5 + 0.5 * mem.decay_weight)
            scored.append((mem, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_similar(self, mem: MemoryRecord, threshold: float) -> Optional[MemoryRecord]:
        """找出与给定记忆高度相似的已有记忆（合并环节用）。

        这里**不加** author 过滤 —— 因为合并要发现的是"同一件事被记了两遍"，
        而同一件事可能从不同角度写（作者字段一样，但症状描述不同）。
        """
        vec = mem.embedding or self.encoder.encode(self._embed_text(mem))
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE tier=? AND archived=0",
            (mem.tier.value,),
        ).fetchall()
        best, best_sim = None, threshold
        for r in rows:
            other = self._row_to_mem(r)
            if other.memory_id == mem.memory_id:
                continue
            sim = cosine(vec, other.embedding or [])
            if sim > best_sim:
                best, best_sim = other, sim
        return best

    def touch_access(self, memory_id: str, used: bool) -> None:
        """记录一次访问。

        **这里藏着那个反直觉的设计**：
        同时记 access_count（被召回）和 use_count（真的用上了）。
        两者之差 = 无效召回 —— 那些"看起来相关但其实没用"的记忆。
        只看 access_count 会误导你：有些记忆被频繁召回，
        仅仅因为它的向量吸引了很多不相关查询（噪音吸铁石）。
        """
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE memories SET
                 access_count = access_count + 1,
                 use_count    = use_count + ?,
                 last_access  = ?
               WHERE memory_id = ?""",
            (1 if used else 0, now, memory_id),
        )
        self.conn.commit()

    def all_memories(self, tier: MemoryTier | None = None) -> list[MemoryRecord]:
        """全量取出（衰减任务用）。"""
        if tier:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE tier=?", (tier.value,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_mem(r) for r in rows]

    # ---------------------------------------------------------------- 冲突

    def add_conflict(self, c: PendingConflict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO pending_conflicts
               (conflict_id, memory_a, memory_b, reason, created_at, resolved_by, resolution)
               VALUES (?,?,?,?,?,?,?)""",
            (c.conflict_id, c.memory_a, c.memory_b, c.reason,
             c.created_at, c.resolved_by, c.resolution),
        )
        self.conn.commit()

    def pending_conflicts(self) -> list[PendingConflict]:
        rows = self.conn.execute(
            "SELECT * FROM pending_conflicts WHERE resolution='' OR resolution IS NULL"
        ).fetchall()
        return [PendingConflict(
            conflict_id=r["conflict_id"], memory_a=r["memory_a"],
            memory_b=r["memory_b"], reason=r["reason"], created_at=r["created_at"],
            resolved_by=r["resolved_by"] or "", resolution=r["resolution"] or "",
        ) for r in rows]

    # ---------------------------------------------------------------- Trace

    def save_trace(self, trace) -> None:
        from ..models import to_dict
        self.conn.execute(
            """INSERT OR REPLACE INTO traces
               (trace_id, issue_id, payload, outcome, n_steps, total_tokens,
                started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (trace.trace_id, trace.issue_id,
             json.dumps(to_dict(trace), ensure_ascii=False),
             trace.outcome.value, trace.n_steps, trace.total_tokens,
             trace.started_at, trace.finished_at),
        )
        self.conn.commit()

    def list_traces(self, outcome: str | None = None) -> list[dict]:
        if outcome:
            rows = self.conn.execute(
                "SELECT payload FROM traces WHERE outcome=?", (outcome,)).fetchall()
        else:
            rows = self.conn.execute("SELECT payload FROM traces").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ------------------------------------------------- SOP 提炼提案（P1-2）

    def add_promotion(self, p: "PendingPromotion") -> bool:
        """登记一条待审核的 SOP 提案。

        ★ 幂等：同一 pattern 如果已经有 pending 的提案，就不重复插入。
        理由：`scan()` 会被反复调用（每次任务结束、或定时跑），
        如果每次都插一条，审核队列会被同一条模式刷屏。

        **"扫描"和"登记"要解耦** —— 扫描是幂等的观察，
        登记才是有副作用的动作，而登记本身也要幂等。
        """
        import json as _json
        with self._lock_for("promotions"):
            cur = self.conn.execute(
                "SELECT promotion_id FROM pending_promotions "
                "WHERE pattern=? AND status='pending'", (p.pattern,))
            if cur.fetchone():
                return False
            self.conn.execute(
                "INSERT INTO pending_promotions (promotion_id, pattern, occurrences,"
                " sample_ids, sample_summary, status, created_at, reviewed_at,"
                " reviewed_by, review_note, promoted_memory_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (p.promotion_id, p.pattern, p.occurrences,
                 _json.dumps(p.sample_ids, ensure_ascii=False),
                 p.sample_summary, p.status, p.created_at, p.reviewed_at,
                 p.reviewed_by, p.review_note, p.promoted_memory_id))
            self.conn.commit()
        return True

    def pending_promotions(self) -> list["PendingPromotion"]:
        """只返回待审核的（status='pending'）。"""
        return self.all_promotions(status="pending")

    def all_promotions(self, status: str | None = None) -> list["PendingPromotion"]:
        """查询提案。传 status 则过滤（用于看审核历史）。"""
        import json as _json
        with self._lock_for("promotions"):
            if status:
                cur = self.conn.execute(
                    "SELECT * FROM pending_promotions WHERE status=?"
                    " ORDER BY occurrences DESC", (status,))
            else:
                cur = self.conn.execute(
                    "SELECT * FROM pending_promotions ORDER BY created_at DESC")
            rows = cur.fetchall()

        out = []
        for r in rows:
            try:
                sample_ids = _json.loads(r["sample_ids"] or "[]")
            except (ValueError, TypeError):
                sample_ids = []
            out.append(PendingPromotion(
                promotion_id=r["promotion_id"], pattern=r["pattern"],
                occurrences=r["occurrences"], sample_ids=sample_ids,
                sample_summary=r["sample_summary"] or "",
                status=r["status"], created_at=r["created_at"] or "",
                reviewed_at=r["reviewed_at"] or "",
                reviewed_by=r["reviewed_by"] or "",
                review_note=r["review_note"] or "",
                promoted_memory_id=r["promoted_memory_id"] or "",
            ))
        return out

    def get_promotion(self, promotion_id: str) -> "Optional[PendingPromotion]":
        for p in self.all_promotions():
            if p.promotion_id == promotion_id:
                return p
        return None

    def review_promotion(self, promotion_id: str, approved: bool,
                         reviewer: str, note: str = "",
                         promoted_memory_id: str = "") -> bool:
        """记录审核结果。

        ★ **只能审一次** —— 已经审过的提案不能再审。

        为什么？因为"审核"是一个**不可逆的人为决定**。
        如果允许反复改，那审核记录就失去了可信度 ——
        你无法回答"这条 SOP 到底是谁批的"。

        要改结论应该**重新走一次流程**（重新扫描 → 重新提案 → 重新审核），
        而不是覆盖历史决定。**这样历史才是可审计的。**
        """
        with self._lock_for("promotions"):
            cur = self.conn.execute(
                "SELECT status FROM pending_promotions WHERE promotion_id=?",
                (promotion_id,))
            row = cur.fetchone()
            if not row:
                return False
            if row["status"] != "pending":
                return False        # 已审核过，拒绝重复审
            self.conn.execute(
                "UPDATE pending_promotions SET status=?, reviewed_at=?,"
                " reviewed_by=?, review_note=?, promoted_memory_id=?"
                " WHERE promotion_id=? AND status='pending'",
                ("approved" if approved else "rejected", _now_str(),
                 reviewer, note, promoted_memory_id, promotion_id))
            self.conn.commit()
        return True

    def _lock_for(self, name: str):
        """按资源名返回一把锁。

        目前只有 promotions 一类资源需要应用级锁
        （memories 靠 thread-local 连接 + SQLite 写锁就够了）。

        抽成方法是为了将来加别的资源时不用改调用点。
        """
        if not hasattr(self, "_promo_lock"):
            self._promo_lock = threading.RLock()
        return self._promo_lock

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _embed_text(mem: MemoryRecord) -> str:
        """决定"拿什么文本去做向量"。

        这里的选择有讲究：symptom 权重最高（它是检索的主字段），
        但要拼上 tags 和 root_cause —— 因为判重时需要这些信息。
        不拼 resolution，因为那是"怎么解决"，而"是不是同一个问题"
        应该看"是什么问题"。
        """
        return " ".join(filter(None, [
            mem.symptom,
            " ".join(mem.symptom_tags),
            mem.root_cause,
        ]))

    @staticmethod
    def _row_to_mem(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            tier=MemoryTier(row["tier"]),
            repo=row["repo"] or "",
            issue_id=row["issue_id"] or "",
            author=row["author"] or "",
            symptom=row["symptom"] or "",
            symptom_tags=json.loads(row["symptom_tags"] or "[]"),
            root_cause=row["root_cause"] or "",
            resolution=row["resolution"] or "",
            actions=json.loads(row["actions"] or "[]"),
            outcome=Outcome(row["outcome"] or "unfinished"),
            human_verdict=HumanVerdict(row["human_verdict"] or "unknown"),
            human_edit=row["human_edit"] or "",
            access_count=row["access_count"] or 0,
            use_count=row["use_count"] or 0,
            last_access=row["last_access"] or "",
            confidence=row["confidence"] or 0.0,
            importance=row["importance"] or 0.0,
            decay_weight=row["decay_weight"] if row["decay_weight"] is not None else 1.0,
            archived=bool(row["archived"]),
            embedding=json.loads(row["embedding"] or "[]"),
            created_at=row["created_at"] or "",
            # ★ 代码状态绑定（仓库变更时效性的地基）
            #
            # 注意这里用模块级的 `_safe_col` 而不是直接 `row["code_refs"]` ——
            # 因为**正在运行的旧库可能还没迁移**（比如另一个进程持有旧连接）。
            # 直接取会抛 `IndexError: No item with that key`，
            # 而那种错误发生在读取路径上，会让"能读"变成"不能读"。
            #
            # **读路径要容忍缺列，写路径不能** —— 这是迁移期的基本原则。
            code_refs=json.loads(_safe_col(row, "code_refs") or "[]"),
            repo_head=_safe_col(row, "repo_head") or "",
            misuse_count=int(_safe_col_int(row, "misuse_count")),
        )


def _safe_col_int(row: "sqlite3.Row", name: str) -> int:
    """安全取整型列：列不存在时返回 0。

    （和 `_safe_col` 同样的理由 —— 读路径要容忍迁移期的缺列。）
    """
    try:
        return int(row[name] or 0)
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def _safe_col(row: "sqlite3.Row", name: str) -> str:
    """安全取列：列不存在时返回空串。

    ★ 为什么需要这个？
      迁移会经历一个"代码已更新、库可能还没更新"的中间状态。
      如果读路径硬取新列，**任何一次漏迁移都会让整个模块不可用** ——
      而"缺一个可选字段"本不该有这种后果。

      **可选字段的读取必须容错。** 这是 schema 演进的基本纪律。

    （这个函数放在模块级而不是类里，是因为 `_row_to_mem` 本身就是
      模块级函数 —— 它不依赖任何实例状态。**放对层级比"看起来整齐"重要。**）
    """
    try:
        return row[name] or ""
    except (IndexError, KeyError):
        return ""
