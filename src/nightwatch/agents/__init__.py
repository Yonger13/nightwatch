"""
Nightwatch · 具体 Agent 实现

本包存放"可被 A2A 委托的专家 Agent" —— 与 `loop/react.py` 的通用
TriageAgent 相对：这里的是**面向特定领域**的角色。

    security.py   安全审计 Agent（识别 CWE 类别 + 沉淀漏洞模式）

设计约定：
  · 每个专家 Agent 实现 `a2a.AgentExecutor` 接口
  · 每个专家 Agent 提供 `card(url)` 静态方法声明自己的能力
  · 通过 A2A 协议与其他 Agent 协作，**不直接函数调用** ——
    这样它才能被独立部署、独立伸缩、独立换实现
"""

from .security import SecurityAuditAgent

__all__ = ["SecurityAuditAgent"]
