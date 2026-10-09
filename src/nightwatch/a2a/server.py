"""
Nightwatch · A2A 服务端

================================================================================
职责：把自己的能力"发布出去"，并接收其他 Agent 的委托
================================================================================
两个端点：

    GET  /.well-known/agent-card.json   —— 能力声明（发现机制）
    POST /                              —— JSON-RPC 2.0（任务生命周期）

================================================================================
★ AgentExecutor：协议与业务之间的适配层
================================================================================
A2A 规范要求每个 Agent 实现一个 executor，把"标准协议消息"翻译成
"我这个 Agent 的具体逻辑"。这是个**适配器模式**的应用：

    A2A 协议（标准）  ←→  AgentExecutor（适配层）  ←→  具体 Agent（业务）

**为什么这层不能省？** 因为协议会变、业务会变，绑在一起就会互相拖累。
有了它，换协议版本只改适配层，换 Agent 实现也只改适配层。

本项目的 executor 是"同步执行 + 立即返回" ——
真实场景里长任务应该用**后台线程 + 状态查询**，见下面 `A2AServer` 的说明。
"""

from __future__ import annotations

import json
import signal
import threading
import time
import uuid
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from .protocol import (
    A2ATask, AgentCard, Artifact, Message, Part, PartType, TaskState,
    rpc_error, rpc_result,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class AgentExecutor(ABC):
    """A2A 要求的执行器接口 —— 协议与业务之间的接缝。

    ★ 有两个方法：`execute` 和 `cancel`。

    **`cancel` 的意义比看起来大**：它意味着"任务是可以被中止的"。
    这在普通 API 里不常见（HTTP 请求发出去就发出去了），
    但 Agent 的任务可能跑很久 —— 如果不支持取消，
    客户端发现委托错了也没法挽回，只能等它跑完（或者一直占着资源）。

    **可取消性是"长任务"的必要设计**，不是可选项。
    """

    @abstractmethod
    def execute(self, task: A2ATask, message: Message) -> None:
        """处理任务。**直接修改传入的 task 对象**（设置 state / artifacts）。

        约定：
          · 成功 → `task.state = COMPLETED` + 附加 artifacts
          · 拒绝 → `task.state = REJECTED`（不擅长这个技能时）
          · 需要更多信息 → `INPUT_REQUIRED`
          · 出错 → `FAILED` + `task.error`
        """

    def cancel(self, task: A2ATask) -> None:
        """取消任务。默认实现是"标记为已取消"。

        子类可以覆盖做真正的清理（中断线程、回滚副作用）。
        """
        task.state = TaskState.CANCELED
        task.updated_at = _now()


class A2AServer:
    """A2A 服务端：发布 Agent Card + 处理 JSON-RPC 请求。

    ============================================================
    ★ 本实现的一个重要简化：任务在"发送时同步执行"
    ============================================================
    A2A 协议是**异步优先**的：`message/send` 应该立即返回，
    任务在后台继续跑，客户端通过 `tasks/get` 轮询状态。

    本实现为了让演示可读，采用**同步执行 + 立即返回终态**。
    这对"秒级完成的任务"完全没问题，而且代码简单得多。

    **代价**：长任务会阻塞 HTTP 连接（客户端超时、并发能力差）。
    要改成真正的异步，需要：
      1. `execute` 丢到后台线程池
      2. 立刻返回 `WORKING` 状态
      3. 加任务状态存储（现在是内存 dict）
      4. 客户端用 `tasks/get` 轮询（客户端已经支持了 —— 见 client.py）

    **这个简化是刻意的、写在文档里的**，不是"忘了做"。
    """

    def __init__(self, card: AgentCard, executor: AgentExecutor):
        self.card = card
        self.executor = executor
        # 任务存储 —— 内存 dict。
        # ★ 生产要换成持久化存储（数据库/Redis），理由：
        #   ① 进程重启后任务状态不该丢（否则客户端查不到结果）
        #   ② 多副本部署时要共享任务状态
        self.tasks: dict[str, A2ATask] = {}
        self._lock = threading.RLock()
        self.started_at = time.time()
        self.stats = {"tasks_received": 0, "tasks_completed": 0,
                      "tasks_rejected": 0, "tasks_failed": 0}

    # ------------------------------------------------------------------ 入口

    def handle_rpc(self, body: dict) -> dict:
        """处理一次 JSON-RPC 调用。

        ★ 注意错误分两层：
          · JSON-RPC 层错误（方法不存在、参数错）→ 返回 `error` 对象
          · 业务层失败（任务执行失败）→ 返回 `result`，其中 status=FAILED

        **这个区分是 A2A 用 JSON-RPC 而非 REST 的价值所在**：
        "任务失败"是个正常的业务结果，不该用 HTTP 500 表达。
        客户端需要能区分"调用没成功"和"调用了但任务失败了"。
        """
        req_id = body.get("id", "")
        method = body.get("method", "")
        params = body.get("params", {}) or {}

        if body.get("jsonrpc") != "2.0":
            return rpc_error(req_id, -32600, "invalid JSON-RPC version")

        if method in ("message/send", "tasks/send"):
            return self._rpc_send(req_id, params)

        if method in ("tasks/get", "tasks/getTask"):
            return self._rpc_get(req_id, params)

        if method == "tasks/list":
            return self._rpc_list(req_id, params)

        if method == "tasks/cancel":
            return self._rpc_cancel(req_id, params)

        return rpc_error(req_id, -32601, f"method not found: {method}")

    def _rpc_send(self, req_id: str, params: dict) -> dict:
        """接收任务委托。"""
        msg = Message.from_dict(params.get("message", {}) or {})
        skill = params.get("skill", "") or msg.data().get("skill", "")

        task = A2ATask(
            task_id=params.get("id") or f"task_{uuid.uuid4().hex[:12]}",
            from_agent=params.get("fromAgent", "unknown"),
            to_agent=self.card.name,
            skill=skill,
            messages=[msg],
            metadata=params.get("metadata", {}) or {},
            created_at=_now(), updated_at=_now(),
            state=TaskState.SUBMITTED,
        )

        with self._lock:
            self.stats["tasks_received"] += 1
            self.tasks[task.task_id] = task

        # ★ 技能不匹配 → 直接拒绝
        #
        # 这是 A2A 里很容易被忽略但很重要的一步：
        # **Agent 应该能体面地拒绝它不擅长的任务。**
        # 硬着头皮做只会给一个糟糕的结果，而且浪费双方的时间和算力。
        #
        # `rejected` 是终态，客户端收到就知道"该找别人"，不用轮询等。
        if skill and not self.card.has_skill(skill):
            task.state = TaskState.REJECTED
            task.error = (f"skill '{skill}' not offered; "
                          f"available: {[s.skill_id for s in self.card.skills]}")
            task.updated_at = _now()
            with self._lock:
                self.stats["tasks_rejected"] += 1
            return rpc_result(req_id, task.to_dict())

        # 执行
        task.state = TaskState.WORKING
        try:
            self.executor.execute(task, msg)
        except Exception as e:                              # noqa: BLE001
            # ★ executor 抛异常不该让整个请求挂掉 ——
            #   要把它转成 FAILED 状态返回给客户端。
            #   因为客户端需要"知道任务失败了"，而不是"连接断了"。
            task.state = TaskState.FAILED
            task.error = f"{type(e).__name__}: {str(e)[:300]}"
        task.updated_at = _now()

        with self._lock:
            if task.state is TaskState.COMPLETED:
                self.stats["tasks_completed"] += 1
            elif task.state is TaskState.FAILED:
                self.stats["tasks_failed"] += 1

        return rpc_result(req_id, task.to_dict())

    def _rpc_get(self, req_id: str, params: dict) -> dict:
        tid = params.get("id") or params.get("taskId", "")
        with self._lock:
            task = self.tasks.get(tid)
        if not task:
            return rpc_error(req_id, -32001, f"task not found: {tid}")
        return rpc_result(req_id, task.to_dict())

    def _rpc_list(self, req_id: str, params: dict) -> dict:
        from_agent = params.get("fromAgent")
        with self._lock:
            items = [t.to_dict() for t in self.tasks.values()
                     if not from_agent or t.from_agent == from_agent]
        return rpc_result(req_id, {"tasks": items})

    def _rpc_cancel(self, req_id: str, params: dict) -> dict:
        tid = params.get("id") or params.get("taskId", "")
        with self._lock:
            task = self.tasks.get(tid)
        if not task:
            return rpc_error(req_id, -32001, f"task not found: {tid}")
        if task.state.is_terminal:
            # ★ 已经是终态就不能再取消 —— 否则客户端会困惑
            #   "它到底完成了还是被取消了"。
            #   状态机的一个基本原则：**终态不可变**。
            return rpc_error(req_id, -32002,
                             f"task already in terminal state: {task.state.value}")
        self.executor.cancel(task)
        task.updated_at = _now()
        return rpc_result(req_id, task.to_dict())


# =============================================================================
# HTTP 层
# =============================================================================

class _Handler(BaseHTTPRequestHandler):
    server_obj: A2AServer
    protocol_version = "HTTP/1.1"
    server_version = "NightwatchA2A/1.0"

    def do_GET(self) -> None:                               # noqa: N802
        path = urlparse(self.path).path
        # ★ 约定的发现路径 —— A2A 规范推荐放在 /.well-known/ 下
        #   （和 OpenID、security.txt 同一个约定，好处是"不用配置就知道去哪找"）
        if path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
            self._json(200, self.server_obj.card.to_dict())
            return
        if path == "/health":
            s = self.server_obj
            self._json(200, {"status": "ok",
                             "agent": s.card.name,
                             "uptime_sec": round(time.time() - s.started_at, 1),
                             "tasks": len(s.tasks), "stats": s.stats})
            return
        self._json(404, {"error": f"not found: {path}"})

    def do_POST(self) -> None:                              # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._json(400, rpc_error("", -32700, "parse error: invalid JSON"))
            return
        self._json(200, self.server_obj.handle_rpc(body))

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        import sys
        sys.stderr.write("[a2a %s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))
        sys.stderr.flush()


def make_a2a_server(card: AgentCard, executor: AgentExecutor,
                    host: str = "127.0.0.1",
                    port: int = 8900) -> tuple[ThreadingHTTPServer, A2AServer]:
    """创建 A2A 服务（不启动）。

    ★ 默认只监听 127.0.0.1 —— 和记忆服务同样的理由：
    **这个服务没有鉴权**，暴露到公网等于让任何人给 Agent 派活。
    默认安全 > 默认方便。
    """
    server_obj = A2AServer(card, executor)
    handler = type("BoundA2AHandler", (_Handler,), {"server_obj": server_obj})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd, server_obj


def serve_a2a(httpd: ThreadingHTTPServer, server_obj: A2AServer) -> None:
    """启动并阻塞，处理优雅关闭。"""

    def _shutdown(signum, frame):                           # noqa: ARG001
        import sys
        sys.stderr.write("\n收到停止信号，正在关闭 A2A 服务...\n")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    host, port = httpd.server_address[:2]
    import sys
    sys.stderr.write(
        f"A2A 服务已启动：http://{host}:{port}\n"
        f"  Agent: {server_obj.card.name}\n"
        f"  技能: {[s.skill_id for s in server_obj.card.skills]}\n"
        f"  Card: http://{host}:{port}/.well-known/agent-card.json\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        sys.stderr.write("A2A 服务已停止\n")
