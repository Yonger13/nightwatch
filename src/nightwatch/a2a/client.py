"""
Nightwatch · A2A 客户端

================================================================================
职责：发现远端 Agent 的能力，把任务委托出去，拿回结果
================================================================================
这是 A2A 交互的完整流程（本项目实现了全流程）：

    ① 发现     GET  /.well-known/agent-card.json   ← 读对方能不能干这活
    ② 决策     本地判断 skill 是否匹配能力声明
    ③ 委托     POST /  {method: message/send}      ← 发任务
    ④ 取回     结果直接在响应里（本项目同步执行）
                或 POST / {method: tasks/get}      ← 轮询（长任务）

================================================================================
★ 与记忆客户端的关键区别：A2A 失败**不该静默降级**
================================================================================
记忆服务挂了 → 静默降级（返回空），因为记忆是"增强"。

**但 A2A 委托失败不能静默降级。**

理由：委托是一个**业务决策**的结果。
"我把这个安全任务交给了安全 Agent" 这个动作如果失败了，
调用方**必须知道** —— 否则它会以为任务已经有人处理，
而实际上没人处理。

    记忆失败 → Agent 少一条参考，仍能独立判断   → 可以降级
    委托失败 → 任务没人做，必须有人接手         → 不能降级

**这是"依赖的失败语义"要按业务含义区分，而不是按技术类型统一处理。**

所以本模块的做法是：
  · 网络层失败 → 返回 `None` + 记录错误（不抛异常，但**绝不伪装成功**）
  · `None` 是明确的信号，调用方必须处理它（不能忽略）

**"返回 None 但记录错误" 比 "抛异常" 好** —— 因为调用方可以选择
"我自己尽力处理"或"上报失败"，而不是被迫中断整个流程。
但**前提是调用方真的检查了返回值** —— 这是接口契约，写在文档里。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .protocol import (
    A2ATask, AgentCard, Message, Part, TaskState, rpc_request,
)

DEFAULT_TIMEOUT = 10.0      # A2A 委托通常不是热路径，可以比记忆调用宽松
DEFAULT_POLL_INTERVAL = 0.5


class A2AClient:
    """与远端 A2A Agent 通信。

    一个 client 实例对应一个远端 Agent（持有它的 Card 缓存）。
    这样调用方不用每次重复发现。
    """

    def __init__(self, base_url: str, name: str = "",
                 timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.name = name or self.base_url
        self.timeout = timeout
        self._card: Optional[AgentCard] = None
        self.stats = {"discoveries": 0, "tasks_sent": 0, "failures": 0,
                      "last_error": ""}

    # ---------------------------------------------------------------- 发现

    def discover(self, force: bool = False) -> Optional[AgentCard]:
        """读取远端 Agent Card。

        ★ 缓存 Card 是有意义的：能力声明**很少变**（改能力要重启服务），
          而发现是每次委托前的动作。缓存能省掉一次往返。

          但如果对方的技能变了（比如扩容后加了新能力），
          缓存会导致"看不见新技能" —— 所以留了 `force=True` 手动刷新。

          **缓存"变化慢的数据"是合理的；但要给出失效手段。**
        """
        if self._card is not None and not force:
            return self._card

        for path in ("/.well-known/agent-card.json", "/.well-known/agent.json",
                     "/agent-card.json"):
            data = self._get(path)
            if data:
                self._card = AgentCard.from_dict(data)
                self.stats["discoveries"] += 1
                return self._card

        self._fail("discover: 所有约定路径都未找到 agent card")
        return None

    def can_handle(self, skill_id: str) -> bool:
        """检查远端是否声明了某技能。

        ★ 这一步是 A2A 的核心价值 —— **先问能不能，再决定要不要委托**。
          没有它的话，你只能"委托了才发现对方不会"，
          而那会浪费一次往返（并且对方要处理一个自己不会的任务）。
        """
        card = self.discover()
        return bool(card and card.has_skill(skill_id))

    # ---------------------------------------------------------------- 委托

    def send_task(self, skill: str, payload: dict, text: str = "",
                  from_agent: str = "unknown",
                  task_id: str = "") -> Optional[A2ATask]:
        """委托一个任务。

        返回 `None` 表示**通信失败或协议错误**（不是任务失败）。
        任务失败会体现在返回的 task.state 里（FAILED / REJECTED）——
        **这两种失败必须能区分**，因为处理方式完全不同：
          · 通信失败 → 重试、换 Agent、或自己处理
          · 任务失败 → 看 error 内容，可能是任务本身的问题
        """
        parts = []
        if text:
            parts.append(Part.of_text(text))
        parts.append(Part.of_data(payload))

        msg = Message(role="user", parts=parts, task_id=task_id)
        body = rpc_request("message/send", {
            "id": task_id,
            "message": msg.to_dict(),
            "skill": skill,
            "fromAgent": from_agent,
        })

        resp = self._post(body)
        if not resp:
            return None
        if "error" in resp:
            self._fail(f"send_task: {resp['error']}")
            return None

        self.stats["tasks_sent"] += 1
        return A2ATask.from_dict(resp["result"])

    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """查询任务状态（长任务用）。"""
        resp = self._post(rpc_request("tasks/get", {"id": task_id}))
        if not resp or "error" in resp:
            return None
        return A2ATask.from_dict(resp["result"])

    def list_tasks(self, from_agent: str = "") -> list[A2ATask]:
        params = {"fromAgent": from_agent} if from_agent else {}
        resp = self._post(rpc_request("tasks/list", params))
        if not resp or "error" in resp:
            return []
        return [A2ATask.from_dict(d) for d in resp["result"].get("tasks", [])]

    def cancel_task(self, task_id: str) -> Optional[A2ATask]:
        """取消任务。

        ★ 只有非终态的任务才能取消 —— 服务端会返回错误。
          这不是限制，是**状态机的正确性要求**：
          "已完成"的任务被"取消"会让状态语义崩掉。
        """
        resp = self._post(rpc_request("tasks/cancel", {"id": task_id}))
        if not resp or "error" in resp:
            return None
        return A2ATask.from_dict(resp["result"])

    def wait_for_completion(self, task_id: str, timeout: float = 60.0,
                            poll_interval: float = DEFAULT_POLL_INTERVAL
                            ) -> Optional[A2ATask]:
        """轮询直到任务到达终态。

        ★ 用轮询而不是流式，是因为本项目的 Card 里声明了 `streaming: false`。
          **客户端应该按对方声明的能力选择传输方式** ——
          这才是"如实声明能力"的意义：双方都知道该怎么配合。

        ★ 必须有超时。否则远端任务卡住时，客户端会无限等待 ——
          这是分布式调用最常见的死法。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self.get_task(task_id)
            if task is None:
                return None
            if task.state.is_terminal:
                return task
            time.sleep(poll_interval)
        return None      # 超时 → None（调用方自己决定怎么办）

    # ---------------------------------------------------------------- 内部

    def _get(self, path: str) -> Optional[dict]:
        try:
            req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:                                   # noqa: BLE001
            return None      # 发现失败是正常的（路径可能不存在），不记错误

    def _post(self, body: dict) -> Optional[dict]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            req = urllib.request.Request(self.base_url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            # A2A 用自定义头做版本协商（规范要求）
            req.add_header("A2A-Version", "0.2.0")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:                              # noqa: BLE001
            self._fail(f"{type(e).__name__}: {str(e)[:150]}")
            return None

    def _fail(self, msg: str) -> None:
        self.stats["failures"] += 1
        self.stats["last_error"] = msg

    def __repr__(self) -> str:
        return (f"<A2AClient {self.name} sent={self.stats['tasks_sent']} "
                f"failed={self.stats['failures']}>")
