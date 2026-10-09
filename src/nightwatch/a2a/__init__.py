"""
Nightwatch · A2A（Agent2Agent）协议实现

一个最小的 A2A 实现，用于演示 **Agent 之间的任务委托**。

    protocol.py  数据模型（AgentCard / A2ATask / Message / Part / Artifact）
    server.py    A2A 服务端（发布能力声明 + 接收委托）
    client.py    A2A 客户端（发现能力 + 委托任务 + 轮询结果）

================================================================================
A2A 与 MCP 的分层（面试常被问）
================================================================================
    MCP  → Agent 到**工具**      给 Agent 装手
    A2A  → Agent 到**Agent**     让 Agent 之间协作

两者互补，不是竞争。本项目两个都有：
    MCP 思路 → `tools/registry.py`（6 个工具 + 沙箱）
    A2A     → 本模块（委托给专门的安全审计 Agent）

================================================================================
与"共享记忆"的关系（重要边界）
================================================================================
A2A 明确要求 Agent **不暴露内部记忆**。所以"跨 Agent 共享记忆"
不是协议能力，而是本项目在应用层做的扩展（模拟同组织内共享知识库）。

协议层完全遵守 A2A（Agent Card / Task 生命周期 / JSON-RPC），
记忆共享在协议边界之外。
"""

from .client import A2AClient
from .protocol import (
    A2ATask, AgentCard, AgentSkill, Artifact, Message, Part, PartType,
    TaskState,
)
from .server import A2AServer, AgentExecutor

__all__ = [
    "A2AClient", "A2AServer", "AgentExecutor",
    "A2ATask", "AgentCard", "AgentSkill", "Artifact", "Message", "Part",
    "PartType", "TaskState",
]
