"""
Nightwatch · 记忆服务（独立进程）

================================================================================
这个模块解决的问题
================================================================================
把 `MemoryStore` 从 Agent 进程里**搬出来，变成独立服务**。

    之前：   Agent 进程 ──直接持有──→ MemoryStore（SQLite）
    现在：   Agent 进程 ──HTTP──→ 记忆服务进程 ──→ MemoryStore

**为什么要拆？**（不是"为了显得分布式"，有实际理由）

  ① **有状态的东西不该和计算的绑在一起**
     记忆是有状态的，Agent 主循环是（或应该是）无状态的。
     绑在一起意味着：不能水平扩展 Agent（每个副本都要自己一份记忆）、
     备份记忆要停整个 Agent、记忆负载直接影响决策延迟。

  ② **故障隔离**
     记忆服务可以独立重启/升级，不影响正在处理的 issue。

  ③ **可以独立观测和伸缩**
     记忆的 QPS、P99 延迟、索引构建时间都是独立指标。
     混合在一个进程里，你无法判断"慢"是决策慢还是检索慢。

================================================================================
为什么用标准库 http.server 而不是 FastAPI
================================================================================
**保持项目零依赖。** 整个项目的设计原则是"标准库优先，生产该用什么写在注释里"。

这个模块要演示的核心是**服务边界与降级语义**，不是"用了什么框架"。
换成 FastAPI 只需要改 20 行（路由声明 + 请求模型），业务逻辑完全不动 ——
这本身就说明边界划对了。

**生产环境应该补的东西**（按重要性排序）：

  1. **并发模型**：`http.server` 是每请求一线程，高并发下会爆。
     生产用 uvicorn/gunicorn + 异步 IO，或用 Go/Rust 重写。
  2. **鉴权**：记忆服务暴露了内部数据，必须加 mTLS 或 token。
     现在**没有任何鉴权** —— 因为它只应该监听内网/本地回环。
  3. **向量索引**：现在是全表扫描 + 内存计算余弦相似度。
     数据量上万后必须换成 HNSW/IVF（faiss、milvus、pgvector）。
     **这是这个服务最先会遇到的瓶颈。**
  4. **连接池**：SQLite 单写锁在高并发下会成瓶颈，要换 Postgres + 连接池。
  5. **可观测**：每路由的延迟直方图、错误率、慢查询日志。
     现在只打了请求行日志（够调试，不够运维）。
  6. **优雅关闭**：现在是简易版（收到信号停止 accept），
     生产要"停止接收新请求 → 等待在途请求完成 → 关闭数据库"。

================================================================================
启动
================================================================================
    python scripts/run_memory_service.py --port 8765 --db data/memory.db

    # Agent 侧指向它（在 .env 里）
    NW_MEMORY_SERVICE_URL=http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from ..models import (
    MemoryRecord, MemoryTier, PendingConflict,
    conflict_from_dict, memory_from_dict, to_dict,
)
from .store import MemoryStore


class MemoryService:
    """包装 MemoryStore，提供 HTTP 语义。

    分成"服务"和"HTTP 处理器"两层，是为了：
      · 业务逻辑可以脱离 HTTP 单测（不需要起服务器）
      · 换传输层（gRPC / 消息队列）时只改 handler

    **这是"可测试性驱动设计"的一个例子** ——
    如果业务逻辑写在 handler 里，测试就必须起 HTTP 服务，慢且脆。
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self.started_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self._lock = threading.RLock()

        # 预热：检查数据库是否可用
        # ★ 这一步很重要 —— 启动时就把"数据库坏了"暴露出来，
        #   而不是等第一个请求进来才发现。
        #   **健康检查的价值在于尽早失败，而不是通过检查。**
        try:
            self.store.count()
            self.db_ok = True
            self.db_error = ""
        except Exception as e:                              # noqa: BLE001
            self.db_ok = False
            self.db_error = f"{type(e).__name__}: {str(e)[:200]}"

    # ------------------------------------------------------------------ 路由

    def handle(self, method: str, path: str, query: dict,
               body: Optional[dict]) -> tuple[int, dict]:
        """路由分发。返回 (HTTP 状态码, 响应体)。

        ★ 所有异常都在这里被捕获并转成 5xx JSON ——
        不能让异常冒到 handler 层，否则连接会断，客户端只看到一个
        语焉不详的 `ConnectionResetError`。

        **服务端的错误必须是"可读的 JSON"**，因为客户端要据此决定是否重试。
        """
        try:
            return self._route(method, path, query, body or {})
        except KeyError as e:
            return 400, {"error": f"missing field: {e}"}
        except (ValueError, TypeError) as e:
            return 400, {"error": f"bad request: {type(e).__name__}: {str(e)[:200]}"}
        except Exception as e:                              # noqa: BLE001
            # 未预期的错误 —— 记下来，返回 500（客户端会重试或降级）
            with self._lock:
                self.error_count += 1
            return 500, {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    def _route(self, method: str, path: str, query: dict,
               body: dict) -> tuple[int, dict]:
        # --- 健康检查 ---
        # ★ 健康检查**不能只返回 200**。
        #   它必须真实反映依赖组件的状态 —— 否则"服务活着但数据库连不上"
        #   这种情况会被误判为健康，客户端不会降级，结果每次调用都失败。
        #
        #   这就是"浅健康检查 vs 深健康检查"的区别：
        #     浅：进程还在吗        → 几乎没用（进程活着但不可用的情况太常见）
        #     深：依赖都正常吗      → 有用，但要控制耗时
        #   这里做深检查但很轻（就是 count 一下）。
        if path == "/health":
            uptime = time.time() - self.started_at
            return 200, {
                "status": "ok" if self.db_ok else "degraded",
                "db_ok": self.db_ok,
                "db_error": self.db_error,
                "uptime_sec": round(uptime, 1),
                "memories": self.store.count() if self.db_ok else -1,
                "requests": self.request_count,
                "errors": self.error_count,
            }

        # --- 记忆的增删改查 ---
        if path == "/memories" and method == "POST":
            mem = memory_from_dict(body["memory"])
            ok = self.store.insert(mem)
            return 200, {"inserted": ok, "memory_id": mem.memory_id}

        if path == "/memories" and method == "GET":
            tier = MemoryTier(query["tier"][0]) if "tier" in query else None
            return 200, {"memories": [to_dict(m) for m in self.store.all_memories(tier)]}

        if path.startswith("/memories/") and method == "GET":
            mid = path.rsplit("/", 1)[-1]
            mem = self.store.get(mid)
            return 200, {"memory": to_dict(mem) if mem else None}

        if path.startswith("/memories/") and method == "PUT":
            mem = memory_from_dict(body["memory"])
            self.store.update(mem)
            return 200, {"updated": True}

        if path.startswith("/memories/") and method == "DELETE":
            self.store.delete(path.rsplit("/", 1)[-1])
            return 200, {"deleted": True}

        # --- 召回（最热的路径）---
        if path == "/recall" and method == "POST":
            tier = body.get("tier")
            mems = self.store.recall(
                query=body["query"],
                author=body.get("author"),
                tier=MemoryTier(tier) if tier else MemoryTier.EPISODIC,
                top_k=int(body.get("top_k", 5)),
                candidate_pool=int(body.get("candidate_pool", 50)),
                include_archived=bool(body.get("include_archived", False)),
            )
            # ★ 注意 `recall` 返回的是 [(记忆, 相似度), ...] —— 必须解包。
            #
            #   这里踩过一个很隐蔽的坑：原来写成 `to_dict(m) for m in mems`，
            #   m 是 tuple，`to_dict` 对 tuple 调用 `asdict()` 会失败。
            #
            #   **但它一直没暴露** —— 因为演示里用哈希伪向量，召回恒为空，
            #   循环体一次都没执行。直到 P1-2 用确定性数据测出非空召回才发现。
            #
            #   教训：**"空结果"会掩盖序列化 bug。**
            #   测试召回类接口时，一定要构造"能召回出东西"的用例 ——
            #   只测"空输入返回空"等于没测。
            #
            #   分数也一并传回去：客户端需要它做阈值判断和展示排序。
            return 200, {"memories": [
                {"memory": to_dict(m), "score": round(float(s), 4)}
                for m, s in mems
            ]}

        if path == "/find_similar" and method == "POST":
            mem = memory_from_dict(body["memory"])
            found = self.store.find_similar(mem, float(body["threshold"]))
            return 200, {"memory": to_dict(found) if found else None}

        if path == "/touch" and method == "POST":
            self.store.touch_access(body["memory_id"], bool(body.get("used", False)))
            return 200, {"ok": True}

        if path == "/count" and method == "GET":
            tier = MemoryTier(query["tier"][0]) if "tier" in query else None
            return 200, {"count": self.store.count(tier)}

        # --- 冲突与轨迹 ---
        if path == "/conflicts" and method == "POST":
            self.store.add_conflict(conflict_from_dict(body["conflict"]))
            return 200, {"ok": True}

        if path == "/conflicts" and method == "GET":
            return 200, {"conflicts": [to_dict(c) for c in self.store.pending_conflicts()]}

        if path == "/traces" and method == "POST":
            self._save_trace(body["trace"])
            return 200, {"ok": True}

        if path == "/traces" and method == "GET":
            outcome = query["outcome"][0] if "outcome" in query else None
            return 200, {"traces": self.store.list_traces(outcome)}

        return 404, {"error": f"no route for {method} {path}"}

    def _save_trace(self, trace_dict: dict) -> None:
        """保存轨迹。

        ★ 为什么传 dict 而不是重建 Trace 对象？
        因为 Trace 是**只写不读**的（服务端只存，不做逻辑）。
        为了一个只写入的对象维护一套完整反序列化，是没必要的复杂度。

        **不是所有数据都要严格类型化 —— 边界上要判断"这个数据会被谁读"。**
        会被逻辑处理的数据（MemoryRecord）才需要强类型；
        只归档的数据（Trace）用 dict 更简单。
        """
        from ..models import Trace, Step, ToolCall, Outcome
        t = Trace(trace_id=trace_dict.get("trace_id", ""),
                  issue_id=trace_dict.get("issue_id", ""))
        for k, v in trace_dict.items():
            if not hasattr(t, k) or k in ("steps", "trace_id", "issue_id"):
                continue
            if k == "outcome":
                t.outcome = _safe_enum(Outcome, v, Outcome.UNFINISHED)
            else:
                setattr(t, k, v)
        t.steps = [
            Step(step=s.get("step", i),
                 thought=s.get("thought", ""),
                 action=(ToolCall(**s["action"]) if s.get("action") else None),
                 observation=s.get("observation", ""))
            for i, s in enumerate(trace_dict.get("steps", []))
        ]
        self.store.save_trace(t)


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# HTTP 处理器
# =============================================================================

class _Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器。

    ★ 注意它把业务逻辑全部委托给 `MemoryService` ——
    handler 里只有"解析 HTTP → 调服务 → 写 HTTP"三件事。
    **这是让业务逻辑可测的关键。**
    """

    service: MemoryService          # 由 make_server 注入

    protocol_version = "HTTP/1.1"   # 支持 keep-alive（HTTP/1.1 默认）
    server_version = "NightwatchMemory/1.0"

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        # 读请求体
        body = None
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 0:
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._respond(400, {"error": "invalid JSON body"})
                return

        t0 = time.time()
        with self.service._lock:
            self.service.request_count += 1
        status, payload = self.service.handle(method, parsed.path, query, body)
        self._respond(status, payload)

        # 请求日志：方法 / 路径 / 状态 / 耗时
        # ★ 日志里**必须带耗时** —— 否则你无法回答"记忆服务慢不慢"。
        #   只有状态码的日志在排查性能问题时毫无用处。
        elapsed = (time.time() - t0) * 1000
        self.log_message("%s %s -> %d (%.1fms)", method, parsed.path,
                         status, elapsed)

    def _respond(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:                       # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:                      # noqa: N802
        self._handle("POST")

    def do_PUT(self) -> None:                       # noqa: N802
        self._handle("PUT")

    def do_DELETE(self) -> None:                    # noqa: N802
        self._handle("DELETE")

    def log_message(self, fmt: str, *args: Any) -> None:
        """把日志写到 stderr，带时间戳（默认实现没有时间戳）。"""
        import sys
        sys.stderr.write("[%s] %s\n" % (
            time.strftime("%H:%M:%S"), fmt % args))
        sys.stderr.flush()


def make_server(store: MemoryStore, host: str = "127.0.0.1",
                port: int = 8765) -> tuple[ThreadingHTTPServer, MemoryService]:
    """创建 HTTP 服务器（不启动）。

    ★ 默认只监听 `127.0.0.1` —— **不是 0.0.0.0**。

    因为这个服务**没有任何鉴权**，监听 0.0.0.0 等于把内部记忆库
    暴露给整个网络。默认安全比默认方便重要 ——
    要对外暴露必须显式传 `--host 0.0.0.0` 并自行加上鉴权层。

    这是"安全默认值"的一个具体例子：
    **让危险的操作需要显式声明，而不是让安全的操作需要显式声明。**
    """
    service = MemoryService(store)
    handler = type("BoundHandler", (_Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, service


def serve_forever(server: ThreadingHTTPServer,
                  service: MemoryService) -> None:
    """启动并阻塞，处理 Ctrl-C / SIGTERM 优雅关闭。"""

    def _shutdown(signum, frame):                       # noqa: ARG001
        import sys
        sys.stderr.write("\n收到停止信号，正在优雅关闭...\n")
        # ★ 在单独线程里调 shutdown —— 因为 shutdown() 会等 serve_forever 退出，
        #   而在信号处理器（主线程）里直接调会死锁。
        #   这是个很常见的坑：**信号处理器里不要做阻塞操作。**
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    host, port = server.server_address[:2]
    import sys
    sys.stderr.write(
        f"记忆服务已启动：http://{host}:{port}\n"
        f"  健康检查：curl http://{host}:{port}/health\n"
        f"  记忆条数：{service.store.count()}\n"
        f"  按 Ctrl-C 停止\n")
    sys.stderr.flush()

    try:
        server.serve_forever()
    finally:
        server.server_close()
        sys.stderr.write("记忆服务已停止\n")
