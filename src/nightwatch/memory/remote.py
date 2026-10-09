"""
Nightwatch · 远程记忆客户端

================================================================================
为什么需要这个模块
================================================================================
原本 `MemoryStore` 是**进程内**的（主循环直接持有它）。这有两个问题：

  ① **不能称之为"分布式"** —— 字节的 JD 把「分布式系统与微服务架构实操经验」
     放在基本要求里，而单机进程内的存储写"分布式"会在面试中被问穿。

  ② **架构上不健康** —— 记忆是**有状态**的，而 Agent 主循环是**无状态**的。
     两者绑在一个进程里，意味着：
       · 不能水平扩展 Agent（每个副本都要自己的一份记忆）
       · 记忆库升级/备份要停掉整个 Agent
       · 记忆的负载会直接影响决策延迟

拆开之后，记忆成了一个**独立的、可独立伸缩的服务** —— 这才是"分布式"的起点。

================================================================================
★ 核心设计：降级（degradation）
================================================================================
**这个类最重要的设计不是"怎么调用"，而是"调用失败时怎么办"。**

记忆对 Agent 来说是什么？—— **增强，不是必需**。
所以正确的行为是：

    记忆服务挂了 → Agent 退化成"没有记忆的版本"继续工作
                 ✗ 而不是整体崩溃

这是**有意的架构选择**，不是偷懒。理由：

  · 用户的 issue 不会因为"记忆库连不上"就不用处理了
  · 没有记忆的 Agent 仍然能做分类（只是少了历史参考）
  · 记忆服务可以后台恢复，不影响当前任务

**每一个方法在失败时都返回"安全默认值"**：

    recall()        → []      （没有记忆参考，继续判断）
    insert()        → False   （这次经验没沉淀，但不影响当前结论）
    find_similar()  → None    （不判重，只做分类）
    count()         → 0
    ...

**注意"安全默认值"不是随便定的** —— 它应该是**"最保守、最不容易导致错误决策"**的取值：
  · `recall` 返回 `[]` 而不是抛异常 → 调用方少一层判断
  · `find_similar` 返回 `None` → 表示"没找到重复"，不会误判成"是重复"

================================================================================
★ 与"空结果"的区分
================================================================================
`recall()` 失败和"查询无结果"**都返回空列表** —— 从调用方看是一样的。

但**统计上必须区分**，否则无法诊断：
  · `stats["degraded_calls"]` 记录降级次数
  · `stats["failed_calls"]` 记录失败次数
  · `healthy` 属性反映当前健康状态

**如果只有"返回空"这一个信号，你会误以为"记忆没命中"，
而实际上是"记忆服务挂了"** —— 这两者的处理方式完全不同。

================================================================================
生产环境应该怎么做（本项目用标准库的理由）
================================================================================
· 传输：gRPC 比 HTTP/JSON 更省（二进制 + 流式），但 HTTP/JSON 更好调试
· 服务发现：Consul / etcd / K8s Service，而不是配置文件
· 熔断：达到失败阈值就"打开断路器"，快速失败而不是每次都超时等待
· 可观测：OpenTelemetry 链路追踪 + 每路由的 P99 延迟指标
· 一致性：记忆写入如果用异步复制，要考虑"刚写的记忆查不到"的读己之写问题

本项目用 `urllib`（标准库）实现，是为了**零依赖 + 代码可读** ——
重点是把"服务边界"和"降级语义"讲清楚，而不是用了什么框架。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from ..models import (
    MemoryRecord, MemoryTier, PendingConflict,
    conflict_from_dict, memory_from_dict, to_dict,
)
from .store import MemoryStore


# 默认超时（秒）
#
# ★ 这个值必须**短**。
# 因为记忆召回在**主循环的热路径**上（每轮决策前都要调）。
# 如果记忆服务卡住 30 秒，整个 Agent 就卡 30 秒 ——
# 而"少一条记忆参考"的损失远小于"卡 30 秒"。
#
# 宁可降级，不要等待。这是热路径依赖的通用原则。
DEFAULT_TIMEOUT = 1.0

# 默认重试次数
#
# ★ 默认 **0 次**（即不重试）—— 这是实测调出来的，不是拍的。
#
# 过程：最初设成 1 次（共 2 次尝试），结果杀掉服务后的降级耗时是 **3167ms**。
# 排查发现两个叠加的问题：
#
#   ① 我按 Linux 的行为写了"连接被拒 → ConnectionRefusedError → 不重试"，
#      但**在 Windows 上连接被拒表现为 TimeoutError**（用满整个超时）。
#      → 平台差异，代码按 Linux 写、在 Windows 跑，静默走错分支。
#
#   ② 于是"连接失败"和"服务超时"走到了同一个分支，都重试了一次：
#         1.5s（第一次超时）+ 0.1s（退避）+ 1.5s（第二次超时）≈ 3.1s
#
# **对热路径来说 3 秒是不可接受的** —— 那比"没有记忆"的损失大得多。
#
# 所以默认不重试：
#   · 记忆是增强不是必需 —— 一次失败就降级，比等 3 秒更明智
#   · 真需要重试的场景（比如批量离线处理）可以显式传 retries=2
#
# **"默认值应该服务于最常用的场景"**，而记忆召回最常用的场景是热路径。
DEFAULT_RETRIES = 0


class RemoteMemoryStore:
    """通过 HTTP 访问远端记忆服务。

    **接口与 `MemoryStore` 完全一致**，所以主循环可以无感切换：

        store = MemoryStore(...)          # 单机模式
        store = RemoteMemoryStore(url)    # 分布式模式

    （都是鸭子类型，不需要共同基类 —— Python 里这样更灵活。
      但如果要静态检查，可以抽一个 Protocol。）

    ★ 所有方法**永不抛异常**（除了构造时的参数错误）。
      失败时返回安全默认值 + 记录统计。这是刻意的设计，见模块文档。
    """

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT,
                 retries: int = DEFAULT_RETRIES, service_name: str = "memory"):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.service_name = service_name

        # 统计（用于可观测性 + 诊断"是没命中还是服务挂了"）
        self.stats: dict[str, Any] = {
            "total_calls": 0,
            "failed_calls": 0,      # 请求失败（超时/连不上/5xx）
            "degraded_calls": 0,    # 失败后走了降级路径（当前等于 failed_calls，分开是为了将来能加"部分降级"）
            "total_latency_ms": 0.0,
            "last_error": "",
        }
        # 用 RLock 保护 stats —— 因为 Agent 可能多线程调用
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 健康

    def health(self) -> dict:
        """查询记忆服务健康状态。

        这个方法**不降级**（不像其他方法返回默认值）——
        因为它的语义就是"告诉我服务是否活着"，返回假数据没有意义。
        """
        try:
            result = self._call("GET", "/health", timeout=self.timeout)
            return result or {"status": "unknown"}
        except Exception as e:                              # noqa: BLE001
            return {"status": "unreachable", "error": str(e)[:200]}

    @property
    def healthy(self) -> bool:
        """最近一次调用的健康状态（不做网络请求，只读缓存）。

        ★ 为什么不在这里实时探测？
        因为"读状态"应该是**廉价的**。如果每次都发请求，
        调用方就不敢频繁查询，反而失去了观测价值。

        实际健康状态由每次真实调用的结果更新 —— 这叫**被动健康检查**，
        生产环境常用它替代主动探活（省资源，且反映真实链路状态）。
        """
        return self.stats["failed_calls"] == 0 or \
            self.stats["total_calls"] > 0 and \
            self.stats["failed_calls"] < self.stats["total_calls"]

    def degradation_rate(self) -> float:
        """降级率 —— 应该被监控的指标。

        如果它长期 > 0，说明记忆服务容量不足或不稳定。
        **这个指标比"记忆命中率"更能反映系统健康** ——
        因为命中率低可能是"确实没有相似记忆"，而降级率高一定是故障。
        """
        total = self.stats["total_calls"]
        return self.stats["degraded_calls"] / total if total else 0.0

    # ---------------------------------------------------------- 与本地一致的接口

    def insert(self, mem: MemoryRecord) -> bool:
        """写入记忆。失败返回 False（不抛异常）。"""
        r = self._call("POST", "/memories", body={"memory": to_dict(mem)})
        if r is None:
            return False
        return bool(r.get("inserted", False))

    def recall(self, query: str, author: str | None = None,
               tier: MemoryTier | None = MemoryTier.EPISODIC,
               top_k: int = 5, candidate_pool: int = 50,
               include_archived: bool = False) -> list[tuple[MemoryRecord, float]]:
        """召回记忆。**失败返回空列表** —— Agent 会退化成"没有记忆参考"继续工作。

        ============================================================
        ★ 返回值必须和本地 MemoryStore.recall 一致：[(记忆, 相似度), ...]
        ============================================================
        这是 P1-2 时发现的一个**接口不一致**：

            本地 MemoryStore.recall     → [(MemoryRecord, float), ...]
            远程 RemoteMemoryStore.recall → [MemoryRecord, ...]      ← 不一致！

        后果很严重：**同一段调用代码，在单机模式下能跑、切到分布式就崩。**
        这种 bug 最难查 —— 因为"本地测试全过"。

        根因是我写 remote 时凭印象对齐接口，没有真的去读本地的签名。
        **接口一致性要靠"读代码"保证，不能靠"我以为"。**

        （更彻底的做法是定义一个 Protocol 基类让两边都实现它，
          这样静态检查就能发现不一致 —— 见类文档末尾的说明。）
        """
        r = self._call("POST", "/recall", body={
            "query": query,
            "author": author,
            "tier": tier.value if tier else None,
            "top_k": top_k,
            "candidate_pool": candidate_pool,
            "include_archived": include_archived,
        })
        if r is None:
            return []

        out: list[tuple[MemoryRecord, float]] = []
        for item in r.get("memories", []):
            # 新版服务端返回 {"memory": {...}, "score": 0.87}
            if isinstance(item, dict) and "memory" in item:
                out.append((memory_from_dict(item["memory"]),
                            float(item.get("score", 0.0))))
            else:
                # 兼容旧版服务端（只返回记忆对象，没有分数）
                # ★ 向前兼容的意义：滚动升级期间新旧版本会共存，
                #   客户端不能因为服务端少一个字段就崩。
                out.append((memory_from_dict(item), 0.0))
        return out

    def find_similar(self, mem: MemoryRecord,
                     threshold: float) -> Optional[MemoryRecord]:
        """找相似记忆（用于判重）。失败返回 None —— 表示"没找到"，不会误判成"是重复"。"""
        r = self._call("POST", "/find_similar",
                       body={"memory": to_dict(mem), "threshold": threshold})
        if r is None:
            return None
        d = r.get("memory")
        return memory_from_dict(d) if d else None

    def touch_access(self, memory_id: str, used: bool) -> None:
        """记录记忆被使用。失败静默 —— 这只影响统计数据，不影响正确性。"""
        self._call("POST", "/touch", body={"memory_id": memory_id, "used": used})

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        r = self._call("GET", f"/memories/{memory_id}")
        if r is None:
            return None
        d = r.get("memory")
        return memory_from_dict(d) if d else None

    def update(self, mem: MemoryRecord) -> None:
        self._call("PUT", f"/memories/{mem.memory_id}",
                   body={"memory": to_dict(mem)})

    def delete(self, memory_id: str) -> None:
        self._call("DELETE", f"/memories/{memory_id}")

    def count(self, tier: MemoryTier | None = None) -> int:
        """记忆条数。失败返回 0。

        ★ 注意：0 和"真的没有记忆"在不同场景下语义不同。
        调用方如果需要区分，应该看 `healthy` 属性。
        """
        path = "/count" + (f"?tier={tier.value}" if tier else "")
        r = self._call("GET", path)
        return int(r.get("count", 0)) if r else 0

    def all_memories(self, tier: MemoryTier | None = None) -> list[MemoryRecord]:
        path = "/memories" + (f"?tier={tier.value}" if tier else "")
        r = self._call("GET", path)
        if r is None:
            return []
        return [memory_from_dict(d) for d in r.get("memories", [])]

    def add_conflict(self, c: PendingConflict) -> None:
        self._call("POST", "/conflicts", body={"conflict": to_dict(c)})

    def pending_conflicts(self) -> list[PendingConflict]:
        r = self._call("GET", "/conflicts")
        if r is None:
            return []
        return [conflict_from_dict(d) for d in r.get("conflicts", [])]

    def save_trace(self, trace) -> None:
        self._call("POST", "/traces", body={"trace": to_dict(trace)})

    def list_traces(self, outcome: str | None = None) -> list[dict]:
        path = "/traces" + (f"?outcome={outcome}" if outcome else "")
        r = self._call("GET", path)
        return r.get("traces", []) if r else []

    def close(self) -> None:
        """远程客户端没有本地资源要释放 —— 保留这个方法是为了接口一致。"""
        return None

    # ------------------------------------------------------------------ 内部

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout: float | None = None) -> Optional[dict]:
        """发一次 HTTP 请求，带重试，**失败返回 None 而不抛异常**。

        ============================================================
        为什么统一返回 None 而不是抛异常
        ============================================================
        因为调用方有 12 处，如果每处都要写 try/except，
        那"降级"就从"架构保证"退化成"每处都要记得写"——
        而**"每处都要记得写"的东西，一定会在某处被忘记**。

        把降级收在这一层，调用方就只需要处理 `None` / 空值。
        这是"把横切关注点收敛到一个地方"的典型应用。
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") \
            if body is not None else None
        t0 = time.time()
        last_err = ""

        with self._lock:
            self.stats["total_calls"] += 1

        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, data=data, method=method)
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(req,
                                            timeout=timeout or self.timeout) as resp:
                    raw = resp.read()
                elapsed = (time.time() - t0) * 1000
                with self._lock:
                    self.stats["total_latency_ms"] += elapsed
                return json.loads(raw.decode("utf-8")) if raw else {}

            # ★ except 的顺序必须**从具体到宽泛**。这个继承层次是：
            #
            #       OSError
            #        ├── ConnectionRefusedError
            #        └── urllib.error.URLError
            #             └── urllib.error.HTTPError      ← 是 URLError 的子类！
            #
            #   所以 HTTPError 必须写在 URLError **之前**，
            #   否则永远匹配不到（被父类截走）。
            #
            #   我在这里连错两次：第一次忘了 ConnectionRefusedError 在 OSError 之前，
            #   第二次又把 HTTPError 放到 URLError 之后。
            #   **两次都不报错，只是静默地走错分支。**
            #
            #   教训：写多个 except 时，先在心里画出继承树，
            #   然后按"叶子到根"的顺序排列。

            except ConnectionRefusedError as e:
                # 服务不在 → 立即降级，不重试
                # （重试不会让一个没启动的服务活过来，只会白等一个超时周期）
                last_err = f"ConnectionRefused: {str(e)[:80]}"
                break

            except urllib.error.HTTPError as e:
                # 5xx → 服务可能只是暂时忙，重试有意义
                # 4xx → 请求本身有问题，重试必然同样失败
                last_err = f"HTTP {e.code}"
                if 500 <= e.code < 600 and attempt < self.retries:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                break

            except urllib.error.URLError as e:
                # DNS 失败 / 连接超时等
                last_err = f"URLError: {str(e.reason)[:80]}"
                if isinstance(e.reason, ConnectionRefusedError):
                    break
                if attempt < self.retries:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                break

            except (OSError, ValueError) as e:
                # 读超时（socket.timeout 是 OSError 的子类）/ JSON 解析失败
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
                if attempt < self.retries:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                break

        # 全部失败 → 降级
        with self._lock:
            self.stats["failed_calls"] += 1
            self.stats["degraded_calls"] += 1
            self.stats["last_error"] = last_err
        return None

    def __repr__(self) -> str:
        rate = self.degradation_rate()
        return (f"<RemoteMemoryStore {self.base_url} "
                f"calls={self.stats['total_calls']} degraded={rate:.1%}>")


def make_store(settings) -> "MemoryStore | RemoteMemoryStore":
    """按配置创建记忆存储（本地或远程）。

    ★ 这是**唯一的切换点** —— 主循环和工具层不需要知道
      记忆到底在本地还是远端。

    配置方式（.env 或环境变量）：
        NW_MEMORY_SERVICE_URL=http://127.0.0.1:8765
            → 用 RemoteMemoryStore
        不配
            → 用本地 MemoryStore（默认，保持单机可用）
    """
    import os

    url = os.getenv("NW_MEMORY_SERVICE_URL", "").strip()
    if url:
        return RemoteMemoryStore(
            url,
            timeout=float(os.getenv("NW_MEMORY_TIMEOUT", DEFAULT_TIMEOUT)),
            retries=int(os.getenv("NW_MEMORY_RETRIES", DEFAULT_RETRIES)),
        )

    # 本地模式：复用调用方给的 encoder
    encoder = getattr(settings, "encoder", None)
    db = getattr(settings, "memory_db", None)
    return MemoryStore(db, encoder=encoder)
