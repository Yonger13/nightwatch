"""
Nightwatch · A2A 协议数据结构

================================================================================
A2A 是什么（以及它不是什么）
================================================================================
**A2A（Agent2Agent）是 Google 在 2025 年 4 月提出、后捐给 Linux 基金会的开放协议。**
它解决的问题是：**让不同框架、不同厂商、不同基础设施上的 Agent 能互相通信。**

    MCP 连的是「Agent ↔ 工具」      （给 Agent 装手）
    A2A 连的是「Agent ↔ Agent」     （让 Agent 之间协作）

A2A 的三个核心对象（本模块实现的）：

    Agent Card   —— 能力声明，一份 JSON，发布在 /.well-known/agent-card.json
                    客户 Agent 读它来决定"要不要把这个任务委托出去"
    Task         —— 有状态的工作单元，有明确的生命周期
                    （submitted → working → input-required → completed/failed/canceled）
    Message/Part —— 通信内容，Part 可以是 text / file / data

**A2A 刻意不定义"Agent 怎么思考"** —— 它只定义"Agent 怎么说话"。
这是它被称为"boring infrastructure"的原因，也是它能跨框架的原因。

================================================================================
★ 本项目对 A2A 的一处扩展（也是需要说清的边界）
================================================================================
**A2A 的设计原则明确包含「不暴露内部状态、记忆、工具实现」。**
原文的说法是：委托方不需要知道远端的思维链、模型、数据库或工具 ——
这种不透明性保护了实现独立性和知识产权。

所以 **"跨 Agent 共享记忆"严格来说不是 A2A 的能力**。

本项目的做法是：**在 A2A 之上加一层组织内的共享记忆**，理由是：

  场景区分：
    · A2A 原本针对的**跨组织**场景（不同公司/厂商的 Agent）
      → 记忆必须隔离，这是信任边界
    · 本项目模拟的是**同一组织内**的两个 Agent（分诊 + 安全审计）
      → 共享知识库不仅是允许的，而且是**必要的**：
        安全 Agent 发现的漏洞模式，分诊 Agent 下次就该认得

**这个区分很重要**，因为面试时会被追问"你到底遵守了 A2A 还是改了它"。
正确的回答是：**协议层完全遵守（Agent Card / Task 生命周期 / JSON-RPC），
记忆共享是应用层的扩展，不在协议边界内。**

================================================================================
实现范围的诚实说明
================================================================================
本项目实现了 A2A 的**最小可用子集**：

  ✅ Agent Card 发现（`/.well-known/agent-card.json`）
  ✅ Task 生命周期（submitted → working → completed/failed/rejected）
  ✅ JSON-RPC 2.0 消息格式
  ✅ `message/send`、`tasks/get`、`tasks/list`
  ✅ Message / Part（text 与 data 两种）
  ✅ Artifact 返回

  ❌ 流式更新（SSE）        —— Agent Card 里声明 `streaming: false`
  ❌ 推送通知（webhook）     —— 声明 `pushNotifications: false`
  ❌ 鉴权                   —— **这是个真实缺口**，见下面的安全说明
  ❌ gRPC binding

**关于"声明了不支持"这件事**：A2A 要求 Agent 在 Card 里**如实声明能力**。
声明 `streaming: false` 不是"没实现"，而是"**按协议正确地告诉对方我没实现**"——
这样客户端会走轮询而不是尝试建流。**诚实的降级优于虚假的能力声明。**
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# 协议版本 —— 放在 Agent Card 里供版本协商
PROTOCOL_VERSION = "0.2.0"


class TaskState(str, Enum):
    """A2A 定义的任务生命周期。

    ★ 注意其中两个状态是 Agent 场景特有的，普通 REST API 没有：

      `INPUT_REQUIRED` —— 远端 Agent 需要**更多信息**才能继续。
        这不是错误，是正常的协作过程（"你能提供复现步骤吗"）。
        它把"一次问答"变成了"多轮协商"。

      `AUTH_REQUIRED` —— 需要授权才能继续（比如要访问私有仓库）。
        这对应"human-in-the-loop"：等人点确认。

    **这两个状态是 A2A 区别于普通 API 的关键** ——
    它承认"Agent 的工作可能无法一次完成"，而这是真实的。
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        """终态 —— 到了这里就不会再变。

        ★ 为什么需要这个属性？
        因为它决定了客户端的**轮询策略**：
        终态就该停止轮询，否则会永远问下去（浪费且暴露设计缺陷）。
        """
        return self in (TaskState.COMPLETED, TaskState.FAILED,
                        TaskState.CANCELED, TaskState.REJECTED)

    @property
    def is_success(self) -> bool:
        return self is TaskState.COMPLETED


class PartType(str, Enum):
    TEXT = "text"
    DATA = "data"
    FILE = "file"


@dataclass
class Part:
    """消息内容的最小单元。

    ★ 为什么要有 Part 这一层，而不是直接用字符串？
    因为 Agent 之间交换的不只是文本 —— 可能是结构化数据（issue 对象）、
    文件引用、甚至表单。用 `parts[]` 让内容类型可扩展，
    而且**支持一次消息里混装多种类型**（一段说明 + 一个结构化 payload）。
    """
    type: PartType
    text: str = ""
    data: dict = field(default_factory=dict)
    file_name: str = ""
    file_uri: str = ""

    @staticmethod
    def of_text(text: str) -> "Part":
        return Part(type=PartType.TEXT, text=text)

    @staticmethod
    def of_data(data: dict) -> "Part":
        return Part(type=PartType.DATA, data=data)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type.value}
        if self.type is PartType.TEXT:
            d["text"] = self.text
        elif self.type is PartType.DATA:
            d["data"] = self.data
        elif self.type is PartType.FILE:
            d["file"] = {"name": self.file_name, "uri": self.file_uri}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Part":
        t = d.get("type", "text")
        if t == "data":
            return Part(type=PartType.DATA, data=d.get("data", {}))
        if t == "file":
            f = d.get("file", {}) or {}
            return Part(type=PartType.FILE, file_name=f.get("name", ""),
                        file_uri=f.get("uri", ""))
        return Part(type=PartType.TEXT, text=d.get("text", ""))


@dataclass
class Message:
    """一次通信回合。"""
    role: str                                  # "user"（委托方）或 "agent"（被委托方）
    parts: list[Part] = field(default_factory=list)
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    task_id: str = ""
    context_id: str = ""

    def text(self) -> str:
        """取所有 text part 拼起来（方便调用方读）。"""
        return "\n".join(p.text for p in self.parts if p.type is PartType.TEXT)

    def data(self) -> dict:
        """取第一个 data part（本项目里 payload 都是结构化的）。"""
        for p in self.parts:
            if p.type is PartType.DATA:
                return p.data
        return {}

    def to_dict(self) -> dict:
        return {"role": self.role, "messageId": self.message_id,
                "taskId": self.task_id, "contextId": self.context_id,
                "parts": [p.to_dict() for p in self.parts]}

    @staticmethod
    def from_dict(d: dict) -> "Message":
        return Message(
            role=d.get("role", "user"),
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            message_id=d.get("messageId", ""),
            task_id=d.get("taskId", ""),
            context_id=d.get("contextId", ""),
        )


@dataclass
class Artifact:
    """任务产物 —— 这是"交付物"，和"对话消息"要分开。

    ★ 区分 Message 和 Artifact 是 A2A 的一个设计要点：
      · Message 是**过程**（提问、澄清、进度）
      · Artifact 是**结果**（最终交付的东西）

    混在一起的话，客户端就无法区分"它还在问我问题"和"它已经给我答案了"。
    """
    name: str
    parts: list[Part] = field(default_factory=list)
    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict:
        return {"artifactId": self.artifact_id, "name": self.name,
                "parts": [p.to_dict() for p in self.parts]}

    @staticmethod
    def from_dict(d: dict) -> "Artifact":
        return Artifact(
            name=d.get("name", ""),
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            artifact_id=d.get("artifactId", ""),
        )


@dataclass
class A2ATask:
    """A2A 的任务对象。

    ★ 它是**有状态的、可查询的、可取消的** ——
    这是 A2A 和普通 REST 调用的根本区别：
    普通 REST 是"一次请求一次响应"，A2A 是"一个有生命周期的对象"。

    为什么这个区别重要？因为 Agent 的任务可能：
      · 跑很久（分钟到小时级）
      · 中途需要补充信息
      · 需要人工授权
      · 客户端断线后还能查结果
    """
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    context_id: str = field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:8]}")
    state: TaskState = TaskState.SUBMITTED
    from_agent: str = ""
    to_agent: str = ""
    skill: str = ""                            # 委托的是哪个技能
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    # 委托方传入的私有上下文 —— **不经过被委托方的推理，只是原样带着**
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": {"state": self.state.value, "error": self.error},
            "fromAgent": self.from_agent,
            "toAgent": self.to_agent,
            "skill": self.skill,
            "messages": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "A2ATask":
        return A2ATask(
            task_id=d.get("id", ""),
            context_id=d.get("contextId", ""),
            state=TaskState((d.get("status") or {}).get("state", "submitted")),
            error=(d.get("status") or {}).get("error", ""),
            from_agent=d.get("fromAgent", ""),
            to_agent=d.get("toAgent", ""),
            skill=d.get("skill", ""),
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            artifacts=[Artifact.from_dict(a) for a in d.get("artifacts", [])],
            metadata=d.get("metadata", {}),
            created_at=d.get("createdAt", ""),
            updated_at=d.get("updatedAt", ""),
        )


@dataclass
class AgentSkill:
    """Agent 声明的一项技能。"""
    skill_id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.skill_id, "name": self.name,
                "description": self.description, "tags": self.tags}

    @staticmethod
    def from_dict(d: dict) -> "AgentSkill":
        return AgentSkill(skill_id=d.get("id", ""), name=d.get("name", ""),
                          description=d.get("description", ""),
                          tags=d.get("tags", []))


@dataclass
class AgentCard:
    """Agent 的能力声明 —— A2A 的发现机制。

    ============================================================
    ★ 一个必须讲清的安全边界
    ============================================================
    **Agent Card 是"能力声明"，不是"安全认证"。**

    Agent 说自己能做什么，不代表：
      · 它真的做得好（可能过度宣传）
      · 它可信（可能是伪装的服务）
      · 它可以访问敏感数据

    A2A 原文特别强调：企业侧仍需要**可信注册中心、白名单、身份校验、
    策略检查、契约级治理**，才能把敏感任务委托出去。

    本项目的 Card 里**没有鉴权声明**（`securitySchemes: {}`）——
    这是一个**真实的缺口**，意味着任何人都能向这个服务发任务。
    所以演示里它只监听 127.0.0.1。

    **把这个缺口明写出来，比假装有安全机制要诚实。**
    """

    name: str
    description: str
    url: str                                   # 服务端点
    version: str = "1.0.0"
    skills: list[AgentSkill] = field(default_factory=list)
    capabilities: dict = field(default_factory=lambda: {
        # 如实声明：本项目没实现流式和推送
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    })
    default_input_modes: list[str] = field(default_factory=lambda: ["text", "data"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text", "data"])
    security_schemes: dict = field(default_factory=dict)   # ⚠ 空 = 无鉴权
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [s.to_dict() for s in self.skills],
            "securitySchemes": self.security_schemes,
        }

    @staticmethod
    def from_dict(d: dict) -> "AgentCard":
        return AgentCard(
            name=d.get("name", ""),
            description=d.get("description", ""),
            url=d.get("url", ""),
            version=d.get("version", ""),
            skills=[AgentSkill.from_dict(s) for s in d.get("skills", [])],
            capabilities=d.get("capabilities", {}),
            security_schemes=d.get("securitySchemes", {}),
            protocol_version=d.get("protocolVersion", PROTOCOL_VERSION),
        )

    def has_skill(self, skill_id: str) -> bool:
        """检查自己是否声明了某技能。

        ★ 用途：**拒绝自己不会的任务。**
        收到委托时先查这张表，不匹配就返回 `REJECTED` ——
        这比"硬着头皮做一个不擅长的任务"要好得多。

        A2A 的 `rejected` 状态存在的意义就是这个：
        **让 Agent 能体面地拒绝，而不是给一个糟糕的结果。**
        """
        return any(s.skill_id == skill_id for s in self.skills)


# =============================================================================
# JSON-RPC 2.0 辅助（A2A 的主要绑定）
# =============================================================================

def rpc_request(method: str, params: dict, req_id: str = "") -> dict:
    """构造 JSON-RPC 2.0 请求。

    ★ A2A 用 JSON-RPC 而不是 REST 的理由：
    JSON-RPC 的 `method` 字段是**自描述的**，加新方法不需要加新路由；
    而且它天然区分"传输成功"（HTTP 200）和"业务成功"（result vs error）——
    在 Agent 场景里这个区分很重要，因为"任务失败"是一个正常的业务结果，
    不该用 HTTP 500 表达。
    """
    return {"jsonrpc": "2.0", "id": req_id or uuid.uuid4().hex[:12],
            "method": method, "params": params}


def rpc_result(req_id: str, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id: str, code: int, message: str,
              data: Optional[dict] = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}
