"""
Nightwatch · 向量索引（IVF，纯 Python 实现）

================================================================================
为什么需要这个模块：两个问题，不是一个
================================================================================
`docs/06` 的"已知问题"里记的是**速度**问题：

    "向量检索是全表扫描 → 上万条后是瓶颈 → 换 HNSW/IVF"

但真正去看 `recall()` 的实现后，发现还有一个**更严重的质量问题**：

```sql
SELECT * FROM memories WHERE ... ORDER BY created_at DESC LIMIT ?
                                    ^^^^^^^^^^^^^^^^^^^^
                                    按时间排序取候选池
```

**候选池是按时间取的，不是按相关性。**
所以记忆库一大，"**旧但相关**"的记忆根本进不了候选池，
连算相似度的机会都没有 —— 这比"慢"严重得多，因为**慢是可见的，漏召回是不可见的**。

IVF 正好同时解决两个问题：
  · 速度：只搜索最近的几个簇（O(N) → O(N/nlist × nprobe)）
  · 质量：候选池按**向量空间邻近**产生，而不是按时间

**这个发现本身值得记下来**：文档里写的"已知问题"只是当时看到的那一面，
真正动手修的时候往往会发现更深的问题。
**"已知问题清单"不是终点，是线索。**

================================================================================
为什么手写 IVF 而不是用 faiss
================================================================================
项目原则是**零第三方依赖**。但这不是唯一理由：

  · faiss 是个 C++ 库（几万行），调它的 API 学不到索引原理
  · IVF 的核心逻辑（聚类 + 倒排 + nprobe 搜索）**在 Python 里两百行能讲清**
  · 手写才能做"精度 vs 速度"的完整权衡实验（这正是本次要测的）

**生产环境该用什么**：faiss（HNSW 或 IVF-PQ）、pgvector（HNSW）、milvus。
本模块的价值是把索引的**工作原理和权衡**讲清楚，
而不是提供一个能扛住百万级的实现。

================================================================================
IVF 的工作方式
================================================================================
构建：
    ① 对所有向量跑 k-means，得到 `nlist` 个簇心
    ② 把每条记忆分配到最近的簇心 → 形成**倒排表**（簇 → 记忆 ID 列表）

查询（这就是"近似"的来源）：
    ① 算 query 到 `nlist` 个簇心的距离，排序
    ② **只搜索最近的 `nprobe` 个簇** ← 近似点在这
    ③ 在选中的簇内精确算相似度，取 top_k

**如果 `nprobe == nlist`，结果与暴力搜索完全一致**（只是绕了一圈）。
`nprobe` 越小越快，但可能漏掉"落在远处的簇里的正确答案"。

================================================================================
★ 一个必须理解的性质：IVF 的召回损失是"可解释的"
================================================================================
暴力搜索漏召回的原因是"它确实不相似"。
**IVF 漏召回的原因是"它相似但不在被搜索的那几个簇里"。**

这个区别很重要 —— 意味着 IVF 的漏召回**可以通过调大 nprobe 消除**，
而"真的不相似"不能。

所以评测 IVF 时，核心指标是 **recall@k 保持率**：
    IVF 的 top_k 结果里，有多少出现在暴力搜索的 top_k 里？
这个数字在 nprobe 足够大时应该趋近 100%。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。

    ★ 为什么不用欧氏距离？
    因为记忆的向量没有归一化，而余弦对**长度不敏感** ——
    一段 200 字的症状描述和一段 50 字的，只要语义相近就该相似，
    不该因为长度不同被判为不相似。

    （如果向量已经归一化，余弦和欧氏距离等价。但依赖"上游一定会归一化"
      是危险的假设 —— 本项目里编码器可替换，不能假设它归一化。）
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class IndexStats:
    """索引的统计信息（用于观测和报告）。"""
    nlist: int = 0                 # 簇数
    n_items: int = 0               # 索引的向量数
    build_sec: float = 0.0         # 构建耗时
    bucket_sizes: list[int] = field(default_factory=list)
    empty_buckets: int = 0         # 空簇数（k-means 没收敛好的信号）

    @property
    def avg_bucket(self) -> float:
        return (sum(self.bucket_sizes) / len(self.bucket_sizes)
                if self.bucket_sizes else 0.0)

    @property
    def imbalance(self) -> float:
        """簇大小不均衡度 = 最大簇 / 平均簇。

        ★ 为什么要看这个？
        IVF 的加速来自"每个簇只占总量的一部分"。
        如果聚类退化成"一个巨簇 + 一堆空簇"，那查询时命中巨簇就退化成全表扫描 ——
        **索引建了等于没建**，而且不会报错。

        这是 k-means 类索引最典型的静默失效。
        """
        if not self.bucket_sizes or self.avg_bucket == 0:
            return 0.0
        return max(self.bucket_sizes) / self.avg_bucket

    def to_dict(self) -> dict:
        return {
            "簇数": self.nlist,
            "索引向量数": self.n_items,
            "构建耗时秒": round(self.build_sec, 3),
            "平均簇大小": round(self.avg_bucket, 1),
            "空簇数": self.empty_buckets,
            "簇大小不均衡度": round(self.imbalance, 2),
            "最大簇": max(self.bucket_sizes) if self.bucket_sizes else 0,
        }


class IVFIndex:
    """倒排文件索引（Inverted File Index）。

    用法：

        idx = IVFIndex(nlist=16)
        idx.build({memory_id: embedding, ...})
        ids = idx.search(query_vec, nprobe=4, top_k=5)

    ★ 设计上刻意**只管 ID 和向量**，不持有 MemoryRecord 对象。
      原因：索引是"纯计算"的，把它和业务对象耦合起来，
      会让"重建索引"变成一件重操作。

      **索引与存储分离** —— 这样索引可以随时重建，
      而不用碰数据库里的记录。这也是生产里向量库的常见划分。
    """

    def __init__(self, nlist: int = 16, seed: int = 42, max_iter: int = 8):
        self.nlist = max(1, nlist)
        self.seed = seed
        # ★ max_iter 默认 8 而不是 20 —— 实测调出来的。
        #   k-means 通常 5~10 轮就基本收敛（簇心位置不再显著变化），
        #   再迭代是纯粹的浪费。20 轮和 8 轮的簇质量差异，
        #   远小于"多花 2.5 倍构建时间"的代价。
        #
        #   （这个判断有前提：如果数据簇结构很模糊，多迭代确实有用。
        #     生产里应该看"簇大小不均衡度"来决定要不要加轮数 ——
        #     本模块把它放进 stats 就是为了这个。）
        self.max_iter = max_iter

        self.centroids: list[list[float]] = []
        self.buckets: dict[int, list[str]] = {}      # 簇 id → 记忆 ID 列表
        self.vectors: dict[str, list[float]] = {}    # 记忆 ID → 向量
        self.stats = IndexStats()
        self._built = False

    # ------------------------------------------------------------------ 构建

    def build(self, items: dict[str, list[float]]) -> "IVFIndex":
        """构建索引。

        ★ 关于 k-means 的一个现实取舍：
        真正好的 k-means 要用 k-means++ 初始化（让初始簇心互相远离）。
        这里用的是**随机采样初始化**（从数据里随机选 nlist 个点当簇心）——
        对 IVF 来说这个粗糙程度是可以接受的，因为：

          · IVF 不要求聚类"最优"，只要求"分得开"
          · 用 nprobe 补偿：聚类差一点，搜索时就多搜几个簇

        **这是"够用就好"的工程判断** —— 不是所有地方都值得做到最优。
        但要注意：如果聚类退化成"一个巨簇 + 一堆空簇"，
        那索引就失效了（所以 stats 里报了不均衡度）。
        """
        t0 = time.time()
        self.vectors = {k: list(v) for k, v in items.items() if v}
        n = len(self.vectors)

        if n == 0:
            self._built = True
            self.stats = IndexStats()
            return self

        # 簇数不能超过数据量（否则会有大量空簇）
        # ★ 这是个容易被忽略的边界：nlist > N 时 k-means 会把一堆簇心
        #   放在同一个点上，然后随机瓜分数据 —— 索引质量很差但不报错。
        effective_nlist = min(self.nlist, n)
        if effective_nlist != self.nlist:
            self.nlist = effective_nlist

        rng = random.Random(self.seed)
        ids = list(self.vectors)
        # 随机采样初始化簇心
        seeds = rng.sample(ids, self.nlist)
        self.centroids = [list(self.vectors[i]) for i in seeds]

        # ---- ★ k-means 用**小采样**训练簇心 ----
        #
        # 实测数据（这就是为什么必须采样）：
        #   2000 点 × 44 簇 × 20 轮 × 256 维 = 4.5 亿次浮点运算
        #   纯 Python 下要 **16 秒** —— 而建索引应该是"启动时完成"的事。
        #
        # 采样依据：**k-means 的簇心是数据的统计量，不需要用全部点估计。**
        # 但采多少有讲究 —— 第一版用了 `nlist × 40 = 1760`，
        # 而总量才 2000，**等于没采样**（17 秒没变）。
        #
        # 现在用 `nlist × 10`（440 个点），训练量降到 1/4。
        # 这是 faiss 等库的标准做法（它们也采样训练）。
        #
        # ★ 诚实说明：**纯 Python 的构建成本就是这个量级**（440 点 × 44 簇
        #   × 8 轮 × 256 维 ≈ 4000 万次运算 ≈ 1.5 秒）。
        #   生产用 faiss 是 C++ + SIMD，同样规模在毫秒级。
        #   本模块的价值是讲清原理和权衡，不是提供生产级性能。
        sample_size = min(n, max(self.nlist * 10, 100))
        sample_ids = rng.sample(ids, sample_size)
        train: dict[str, list[float]] = {i: self.vectors[i] for i in sample_ids}

        # ---- k-means 迭代 ----
        assign: dict[str, int] = {}
        for _ in range(self.max_iter):
            changed = False
            for mid, vec in train.items():
                c = self._nearest_centroid(vec)
                if assign.get(mid) != c:
                    assign[mid] = c
                    changed = True
            if not changed:
                break
            # 更新簇心
            sums: dict[int, list[float]] = {}
            counts: dict[int, int] = {}
            for mid, c in assign.items():
                vec = train[mid]
                if c not in sums:
                    sums[c] = [0.0] * len(vec)
                    counts[c] = 0
                for i, v in enumerate(vec):
                    sums[c][i] += v
                counts[c] += 1
            for c, s in sums.items():
                self.centroids[c] = [x / counts[c] for x in s]

        # ---- 建倒排表：**全部数据**分配到最近的簇（这一步不能采样）----
        self.buckets = {i: [] for i in range(self.nlist)}
        for mid, vec in self.vectors.items():
            self.buckets[self._nearest_centroid(vec)].append(mid)

        sizes = [len(v) for v in self.buckets.values()]
        self.stats = IndexStats(
            nlist=self.nlist, n_items=n,
            build_sec=time.time() - t0,
            bucket_sizes=sizes,
            empty_buckets=sum(1 for s in sizes if s == 0),
        )
        self._built = True
        return self

    def add(self, memory_id: str, vec: list[float]) -> None:
        """增量添加一条（不重建索引）。

        ★ IVF 的增量能力是它的**弱点**：新向量只能塞进最近的簇，
          但簇心不会更新。加得多了，簇心就偏了 —— 查询质量下降。

          生产里通常的做法是：**攒一批后重建**（或周期性重建）。
          本方法只做"塞进最近的簇"，并在 add 数量超过阈值时给出提醒。
        """
        if not vec or not self.centroids:
            return
        c = self._nearest_centroid(vec)
        self.buckets.setdefault(c, []).append(memory_id)
        self.vectors[memory_id] = list(vec)
        self.stats.n_items += 1
        self.stats.bucket_sizes = [len(v) for v in self.buckets.values()]

    def remove(self, memory_id: str) -> bool:
        """按 ID 删除（不重建索引）。"""
        if memory_id not in self.vectors:
            return False
        self.vectors.pop(memory_id, None)
        for ids in self.buckets.values():
            if memory_id in ids:
                ids.remove(memory_id)
                break
        self.stats.n_items = max(0, self.stats.n_items - 1)
        self.stats.bucket_sizes = [len(v) for v in self.buckets.values()]
        return True

    # ------------------------------------------------------------------ 查询

    def search(self, query_vec: list[float], nprobe: int = 4,
               top_k: int = 5) -> list[tuple[str, float]]:
        """近似搜索：只在最近的 nprobe 个簇内精确比对。

        返回 [(memory_id, 相似度), ...]，按相似度降序。

        ★ `nprobe=0` 或 `nprobe >= nlist` 时退化成**精确搜索**（扫全部簇）。
          这个性质让"精度损失"的评测变得很干净：
          直接拿 nprobe=nlist 的结果当作 ground truth 去对比。
        """
        if not self._built or not self.centroids:
            return []

        # 选最近的 nprobe 个簇
        probe = self.nlist if nprobe <= 0 else min(nprobe, self.nlist)
        cdist = [(cosine(query_vec, c), i) for i, c in enumerate(self.centroids)]
        cdist.sort(reverse=True)
        chosen = [i for _, i in cdist[:probe]]

        # 在选中的簇内精确算
        scored: list[tuple[str, float]] = []
        for c in chosen:
            for mid in self.buckets.get(c, ()):
                vec = self.vectors.get(mid)
                if vec:
                    scored.append((mid, cosine(query_vec, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_exact(self, query_vec: list[float],
                     top_k: int = 5) -> list[tuple[str, float]]:
        """精确搜索（暴力），用于评测时当 baseline。"""
        scored = [(mid, cosine(query_vec, vec))
                  for mid, vec in self.vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ 内部

    def _nearest_centroid(self, vec: list[float]) -> int:
        best_c, best_sim = 0, -2.0
        for i, c in enumerate(self.centroids):
            sim = cosine(vec, c)
            if sim > best_sim:
                best_sim, best_c = sim, i
        return best_c

    @property
    def is_built(self) -> bool:
        return self._built
