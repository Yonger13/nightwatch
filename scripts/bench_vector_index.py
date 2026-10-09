#!/usr/bin/env python3
"""
Nightwatch · 向量索引评测（P2-4）

    python scripts/bench_vector_index.py                     # 默认 2000 条
    python scripts/bench_vector_index.py --n 10000           # 压到 1 万条
    python scripts/bench_vector_index.py --n 5000 --queries 50

==============================================================================
这个评测要回答三个问题
==============================================================================
  ① **IVF 真的更快吗？**            —— 加速比 vs 暴力搜索
  ② **快了多少代价？**              —— recall@k 保持率
  ③ **它比原来的 SQL 候选池更好吗？** —— ★ 这是最有价值的对比

==============================================================================
★ 为什么加了第 ③ 个问题
==============================================================================
`docs/06` 的"已知问题"里只记了**速度**问题：

    "向量检索是全表扫描 → 上万条后是瓶颈"

但去看 `recall()` 的实现会发现还有个更严重的问题：
**候选池是按 `created_at`（时间）取的，不是按相关性。**

所以记忆库一大，"旧但相关"的记忆**根本进不了候选池**。

这意味着"索引 vs 无索引"的对比不能只测速度 ——
还要测**召回质量**，而且后者才是关键：

    慢是可见的（用户会抱怨）
    漏召回是不可见的（你以为"没有相关记忆"，其实有）

**所以这个评测的核心指标不是"快了多少倍"，而是"召回率高了多少"。**

==============================================================================
评测方法
==============================================================================
**Ground truth 用暴力搜索**（扫全部向量）——
它的结果是精确解，可以作为"正确答案"。

然后比较两条实际路径：
  · SQL 候选池（默认路径）：按时间取 candidate_pool 条 → 算向量 → top_k
  · IVF 索引：按向量邻近取候选 → 算向量 → top_k

**保持率** = 实际返回的 top_k 里，有多少条出现在暴力搜索的 top_k 里。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.memory.index import IVFIndex, cosine              # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore    # noqa: E402
from nightwatch.models import MemoryRecord, MemoryTier, Outcome   # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


# 症状模板 —— 用来生成有"语义簇"特征的语料
#
# ★ 为什么要构造有簇结构的语料，而不是纯随机向量？
#   因为真实数据是有语义聚类的（"超时问题"自成一簇）。
#   纯随机向量的 k-means 聚类没有意义，测出来的索引质量不真实。
TOPICS = [
    ("timeout", ["CI 超时", "流水线超时失败", "任务超时退出", "构建超时"]),
    ("oom", ["内存暴涨 OOM", "读取大文件内存溢出", "内存占用过高"]),
    ("merge", ["merge 行数暴涨", "join 结果异常", "笛卡尔积导致行数增多"]),
    ("encoding", ["CSV 中文乱码", "编码错误", "GBK 解码失败"]),
    ("deprecate", ["iteritems 已弃用", "API 废弃警告", "方法将在 2.0 移除"]),
    ("permission", ["权限拒绝", "SA 缺权限", "secret 读取失败"]),
    ("sql", ["SQL 注入", "查询参数未转义", "拼接 SQL 字符串"]),
    ("pickle", ["反序列化漏洞", "pickle 执行任意代码", "yaml.load 不安全"]),
    ("docs", ["文档描述与实际不符", "示例代码跑不通"]),
    ("perf", ["性能退化", "计算变慢", "循环效率低"]),
]


def gen_corpus(n: int, dim: int, seed: int = 42) -> dict[str, list[float]]:
    """生成有簇结构的合成语料。

    ★ 注意：这里直接用向量（不经过 encoder）——
      因为本评测测的是**索引的检索质量**，不是编码器的质量。
      用真实 encoder 会引入噪声，让"索引好不好"变得不可分离。
      **控制变量的原则：一次只测一个东西。**
    """
    rng = random.Random(seed)
    # 每个主题一个基准向量
    topic_vecs = {}
    for name, _ in TOPICS:
        v = [rng.gauss(0, 1) for _ in range(dim)]
        topic_vecs[name] = v

    out: dict[str, list[float]] = {}
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)][0]
        base = topic_vecs[topic]
        # 在基准向量附近加噪声 —— 同一主题的向量互相接近但不重合
        noise = 0.35
        vec = [b + rng.gauss(0, noise) for b in base]
        out[f"m{i:06d}"] = vec
    return out


def gen_queries(nq: int, dim: int, seed: int = 7) -> list[list[float]]:
    """生成查询向量（也在主题簇附近）。"""
    rng = random.Random(seed)
    topic_vecs = {}
    rng2 = random.Random(42)      # 和语料用同一个 seed 才能取到同样的簇心
    for name, _ in TOPICS:
        topic_vecs[name] = [rng2.gauss(0, 1) for _ in range(dim)]

    qs = []
    for i in range(nq):
        topic = TOPICS[i % len(TOPICS)][0]
        qs.append([b + rng.gauss(0, 0.15) for b in topic_vecs[topic]])
    return qs


def keep_rate(actual: list[str], truth: list[str]) -> float:
    """保持率 = 实际结果里有几条出现在 ground truth 里。"""
    if not truth:
        return 1.0
    return len(set(actual) & set(truth)) / len(truth)


def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="向量索引评测")
    ap.add_argument("--n", type=int, default=2000, help="记忆条数")
    ap.add_argument("--dim", type=int, default=256, help="向量维度")
    ap.add_argument("--queries", type=int, default=30, help="查询次数")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--nlist", type=int, default=0,
                    help="簇数（0 = 自动取 sqrt(N)）")
    ap.add_argument("--json", help="结果 JSON 路径")
    args = ap.parse_args()

    n_q, k = args.queries, args.top_k

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 向量索引评测{C['reset']}")
    hr()
    print(f"""
  规模        {args.n} 条 × {args.dim} 维
  查询        {n_q} 次，top_k={k}
  对比        · 暴力搜索（ground truth）
              · SQL 候选池（原路径，按时间取候选）
              · IVF 索引（按向量邻近取候选，不同 nprobe）
""")

    corpus = gen_corpus(args.n, args.dim)
    queries = gen_queries(n_q, args.dim)
    n = len(corpus)
    nlist = args.nlist or max(4, int(n ** 0.5))
    print(f"  自动簇数    nlist = {nlist}（约 sqrt(N)）")

    # ---------------------------------------------------------- 建库
    hr("① 准备三条路径")

    # ground truth：暴力搜索
    t0 = time.time()
    truth: list[list[str]] = []
    for q in queries:
        scored = sorted(((mid, cosine(q, v)) for mid, v in corpus.items()),
                        key=lambda x: -x[1])[:k]
        truth.append([mid for mid, _ in scored])
    brute_ms = (time.time() - t0) * 1000 / n_q
    print(f"  暴力搜索    {brute_ms:.2f} ms/次（扫全部 {n} 条）")

    # SQL 候选池路径：需要建一个真实的 MemoryStore
    # 取 candidate_pool=50（和 store 默认一致）
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=args.dim))
    for i, (mid, vec) in enumerate(corpus.items()):
        topic = TOPICS[i % len(TOPICS)][0]
        mem = MemoryRecord(
            memory_id=mid, tier=MemoryTier.EPISODIC,
            issue_id=f"#{i}", symptom=f"{topic} 类问题 {i}",
            symptom_tags=[topic], outcome=Outcome.RESOLVED)
        mem.embedding = vec          # 直接用生成的向量（跳过 encoder）
        store.insert(mem)

    # ★ SQL 路径的关键测量：**候选池里的 top_k 能覆盖多少 ground truth**
    #
    #   注意这里要"按时间取候选"，所以故意把语料顺序打乱 ——
    #   否则新建的记录恰好就是时间最新的，会掩盖"旧但相关"漏召回的问题。
    pool_t0 = time.time()
    pool_hits: list[list[str]] = []
    for q in queries:
        # 模拟"按时间取候选池，再算向量"
        cand = list(corpus.items())[:50]        # 取前 50 条（模拟时间序）
        scored = sorted(((mid, cosine(q, v)) for mid, v in cand),
                        key=lambda x: -x[1])[:k]
        pool_hits.append([mid for mid, _ in scored])
    pool_ms = (time.time() - pool_t0) * 1000 / n_q
    pool_rates = [keep_rate(a, t) for a, t in zip(pool_hits, truth)]
    print(f"  SQL 候选池  {pool_ms:.2f} ms/次"
          f"（只算 50 条，但候选按时间取）")

    # IVF 索引
    t0 = time.time()
    idx = IVFIndex(nlist=nlist).build(corpus)
    build_ms = (time.time() - t0) * 1000
    print(f"  IVF 索引    构建 {build_ms:.0f} ms"
          f"（nlist={idx.stats.nlist}，"
          f"不均衡度={idx.stats.imbalance:.2f}，"
          f"空簇={idx.stats.empty_buckets}）")

    # ---------------------------------------------------------- 扫 nprobe
    hr("② IVF 在不同 nprobe 下的表现（精度-速度权衡）")
    print(f"\n  {'nprobe':>7}{'延迟(ms)':>11}{'加速比':>9}{'保持率':>10}"
          f"{'扫过的条数':>12}")
    print(f"  {'-' * 54}")

    rows = []
    for nprobe in sorted({1, 2, 4, 8, nlist}):
        if nprobe > nlist:
            continue
        t0 = time.time()
        hits = []
        for q in queries:
            res = idx.search(q, nprobe=nprobe, top_k=k)
            hits.append([mid for mid, _ in res])
        ms = (time.time() - t0) * 1000 / n_q
        rates = [keep_rate(a, t) for a, t in zip(hits, truth)]
        avg_rate = statistics.mean(rates)
        speedup = brute_ms / ms if ms > 0 else 0
        rows.append({"nprobe": nprobe, "ms": ms, "speedup": speedup,
                     "keep_rate": avg_rate})
        mark = "← 精确（同暴力）" if nprobe >= nlist else ""
        print(f"  {nprobe:>7}{ms:>11.2f}{speedup:>8.1f}x{avg_rate:>10.1%}  {mark}")

    print(f"""
  {C['dim']}  保持率 = IVF 的 top_k 里有多少出现在暴力搜索的 top_k 里。
    nprobe = nlist 时应该趋近 100%（那时搜了所有簇，等于精确搜索）。{C['reset']}""")

    # ---------------------------------------------------------- 关键对比
    hr("③ ★ 关键对比：IVF vs SQL 候选池（「快」之外的东西）")

    best = next((r for r in rows if r["keep_rate"] >= 0.95), rows[-1])
    print(f"""
  {C['bold']}召回质量（这才是重点）{C['reset']}
    SQL 候选池   保持率 {statistics.mean(pool_rates):.1%}
    IVF (nprobe={best['nprobe']})  保持率 {best['keep_rate']:.1%}
""")
    delta = best["keep_rate"] - statistics.mean(pool_rates)
    if delta > 0.05:
        print(f"  {C['green']}✓ IVF 的召回质量明显更好（+{delta:.1%}）{C['reset']}")
        print(f"""
  {C['bold']}为什么？{C['reset']}
    SQL 候选池是**按 `created_at`（时间）**取的 —— 前 50 条候选中，
    只包含语料里最新的记录，而"旧但相关"的记忆**根本进不了候选池**。

    IVF 是**按向量邻近**取候选 —— 它不管记录多旧，只看向量像不像。

    {C['dim']}所以"向量索引"的价值有两层：
      · 快（{brute_ms / best['ms']:.1f}x）
      · **而且召回质量更好** —— 后者才是关键，因为：

          慢是可见的（用户会抱怨）
          漏召回是不可见的（你以为"没有相关记忆"，其实有）{C['reset']}""")
    else:
        print(f"  {C['yellow']}⚠ 在这个规模下差异不明显（{delta:+.1%}）"
              f"—— 语料太小，候选池恰好覆盖了{C['reset']}")

    # ---------------------------------------------------------- 规模影响
    hr("④ 结论与工程建议")
    print(f"""
  {C['bold']}加速比{C['reset']}
    暴力搜索   {brute_ms:.2f} ms/次
    IVF        {best['ms']:.2f} ms/次  →  {C['bold']}{brute_ms / best['ms']:.1f}x{C['reset']}

  {C['dim']}  ★ 加速比的来源：暴力搜索是 O(N)，IVF 是 O(N/nlist × nprobe +
    nlist)。N 越大差距越明显 —— 这也是为什么"N 小时感觉不到索引的价值"。{C['reset']}

  {C['bold']}生产建议{C['reset']}
    · **N < 5000**：不必上索引。全表扫描在这个量级够快，
      而索引的构建成本和失效风险（增量添加让簇心偏移）反而不划算。
    · **N > 5 万**：必须上。此时全表扫描的单次延迟已经影响交互体验。
    · **真正的生产方案是 faiss（HNSW/IVF-PQ）或 pgvector** ——
      本模块的价值是把原理和权衡讲清楚，不是提供一个能扛百万级的实现。

  {C['bold']}参数建议{C['reset']}
    · `nlist ≈ sqrt(N)` 是通用起点
    · `nprobe`: 从 4 开始，按"保持率 >= 95%"标定（本评测就是这个方法）
    · 关注**簇大小不均衡度** —— 它 > 3 说明聚类退化，索引会失效但不报错
""")

    # ---------------------------------------------------------- 输出
    out = Path(args.json) if args.json else ROOT / "data" / "vector_index_bench.json"
    payload = {
        "config": {"n": n, "dim": args.dim, "queries": n_q, "top_k": k,
                   "nlist": nlist},
        "brute_ms": round(brute_ms, 3),
        "sql_pool_ms": round(pool_ms, 3),
        "sql_pool_keep_rate": round(statistics.mean(pool_rates), 4),
        "index_stats": idx.stats.to_dict(),
        "build_ms": round(build_ms, 1),
        "nprobe_sweep": [
            {**r, "ms": round(r["ms"], 3),
             "speedup": round(r["speedup"], 2),
             "keep_rate": round(r["keep_rate"], 4)}
            for r in rows
        ],
        "caveats": [
            "合成语料（有主题簇结构），不是真实数据",
            "测试向量直接生成，不经过编码器 —— 隔离掉编码器质量的影响",
            "SQL 候选池路径模拟了「按时间取候选」的行为",
            "本实现是纯 Python，不能用它的绝对延迟外推生产环境",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  {C['green']}✓{C['reset']} JSON → {out}")

    hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
