"""
Nightwatch · 安全审计 Agent（A2A 的被委托方）

================================================================================
为什么需要这个 Agent
================================================================================
它存在的意义不是"功能多"，而是**作为 A2A 的被委托方**，证明：

  ① 另一个 Agent 可以不知道它的实现（模型/工具/记忆），只通过协议协作
  ② 能力分工是真实的 —— 分诊 Agent 不该自己处理安全漏洞
  ③ **共享记忆创造双向价值** —— 它写下的漏洞模式，分诊 Agent 下次就能用上

场景：pandas 仓库里出现过 `read_pickle` 的 RCE（反序列化漏洞）。
      这类 issue 的特征很明确，但不该由通用的分诊 Agent 处理 ——
      它需要专业判断（漏洞等级、影响面、修复建议）。

================================================================================
★ 与分诊 Agent 的关键区别：共享记忆的用法
================================================================================
分诊 Agent 用记忆来**参考历史分类**。
安全 Agent 用记忆来**积累漏洞模式库** —— 这是"组织知识沉淀"的体现：

    第一次遇到 pickle 反序列化漏洞 → 写入记忆（症状 + 危害 + 修复）
    第二次遇到同类                   → 召回命中 → 直接给出已知的修复方案

**共享的价值在第二次才显现**：如果两个 Agent 各有一份记忆，
安全 Agent 学到的模式，分诊 Agent 永远看不到。

（面试点：这解释了为什么"共享记忆"值得做 —— 它不是"省存储"，
  而是让组织内的知识能跨 Agent 复用。）

================================================================================
实现取舍：为什么用规则而不是 LLM
================================================================================
这个 Agent 用**关键词规则 + 记忆检索**，不调 LLM。理由：

  · 演示要**可复现**（LLM 输出有随机性，演示不稳定）
  · 安全模式的识别是**可枚举的**（CWE 类别是有限集合）
  · 让演示跑得快（不依赖 API、不加延迟）

**这不代表生产该这么做** —— 生产里安全性判断应该：
  · 用 LLM 做初筛（理解自然语言描述）
  · 用规则做兜底（已知 CWE 模式）
  · 用真实的 CVE 数据库做交叉验证
  · **必要时转人工**（安全问题的误判代价很高）

见 `docs/05` 的 F3 结论：判断密度高的任务不该关掉思考链。
安全判断恰恰是高判断密度任务 —— 所以生产实现应该开思考链。
"""

from __future__ import annotations

import re
from typing import Optional

from ..a2a.protocol import (
    A2ATask, Artifact, Message, Part, TaskState, AgentCard, AgentSkill,
)
from ..a2a.server import AgentExecutor
from ..models import MemoryRecord, MemoryTier, Outcome, Issue
from ..memory.remote import RemoteMemoryStore
from ..memory.store import MemoryStore


# 安全模式库 —— CWE 是业界通用的漏洞分类体系
#
# ★ 每条包含：匹配模式、CWE 编号、危害等级、修复建议。
#   这是**领域知识的载体** —— Agent 的"专业性"体现在这张表里，
#   而不是体现在它能说多少话。
SECURITY_PATTERNS = [
    {
        "id": "CWE-502",
        "name": "不安全反序列化",
        "patterns": ["pickle", "unpickle", "deserializ", "反序列化", "yaml.load",
                     "loads(", "marshal"],
        "severity": "high",
        "advice": ("禁止对不可信输入调用 pickle.load；"
                   "改用 JSON / 受限的 safe_load；"
                   "如必须支持，加签名校验（HMAC）后再反序列化"),
    },
    {
        "id": "CWE-78",
        "name": "命令注入",
        "patterns": ["os.system", "subprocess", "shell=true", "eval(", "exec(",
                     "命令注入", "command injection"],
        "severity": "critical",
        "advice": ("避免拼接字符串执行命令；改用参数列表传参（shell=False）；"
                   "对输入做白名单校验"),
    },
    {
        "id": "CWE-22",
        "name": "路径穿越",
        "patterns": ["path traversal", "路径穿越", "../", "..\\\\",
                     "os.path.join", "tarfile", "zipfile"],
        "severity": "high",
        "advice": ("用 os.path.realpath 校验目标路径在允许目录内；"
                   "解压时过滤包含 .. 的成员名"),
    },
    {
        "id": "CWE-89",
        "name": "SQL 注入",
        "patterns": ["sql", "select ", "insert into", "sql injection"],
        "severity": "high",
        "advice": "使用参数化查询，不要拼接 SQL 字符串",
    },
    {
        "id": "CWE-400",
        "name": "资源耗尽（拒绝服务）",
        "patterns": ["memory", "oom", "内存暴涨", "cpu", "infinite loop",
                     "死循环", "dos", "耗尽"],
        "severity": "medium",
        "advice": "加输入长度上限、超时、以及流式处理，避免一次性加载全部数据",
    },
]


class SecurityAuditAgent(AgentExecutor):
    """安全审计 Agent —— 通过 A2A 接受委托。

    ★ 它持有 `store`（共享记忆），这是"跨 Agent 共享记忆"的落点。
    """

    NAME = "security-audit-agent"
    SKILL = "security-audit"

    def __init__(self, store: "MemoryStore | RemoteMemoryStore"):
        self.store = store
        self.audited: list[dict] = []       # 审计历史（用于演示展示）

    # ------------------------------------------------------------- Agent Card

    @classmethod
    def card(cls, url: str) -> AgentCard:
        """发布自己的能力声明。

        ★ 注意 `capabilities` 里如实声明 `streaming: false` ——
        这不是"没实现"，而是"按协议正确告诉对方走轮询"。
        """
        return AgentCard(
            name=cls.NAME,
            description=("开源项目的安全漏洞审计。识别反序列化 / 命令注入 / "
                         "路径穿越 / SQL 注入 / 资源耗尽等 CWE 类别，"
                         "给出危害等级与修复建议，并沉淀漏洞模式到共享记忆。"),
            url=url,
            version="1.0.0",
            skills=[
                AgentSkill(
                    skill_id=cls.SKILL,
                    name="安全漏洞审计",
                    description=("输入一个 issue（标题 + 正文），返回漏洞类别（CWE）、"
                                 "危害等级、修复建议，以及历史上是否出现过同类漏洞。"),
                    tags=["security", "cwe", "vulnerability", "audit"],
                ),
            ],
        )

    # ---------------------------------------------------------------- 执行

    def execute(self, task: A2ATask, message: Message) -> None:
        """处理委托。

        ★ 流程刻意写成"先查记忆、再判断、后写记忆" —— 这是**记忆六环节**
          在另一个 Agent 里的复用：

            查（recall）→ 判（规则匹配）→ 写（沉淀模式）→ 回报（artifact）
        """
        payload = message.data()
        issue_dict = payload.get("issue", {}) or {}
        title = issue_dict.get("title", "")
        body = issue_dict.get("body", "")
        issue_id = issue_dict.get("issue_id", "")

        # --- ① 先查共享记忆：历史上有没有同类漏洞 ---
        # ★ 用 issue 的文本去召回，而不是用"安全"这种泛词 ——
        #   记忆检索的质量取决于查询文本的**信息量**。
        query = f"{title} {body[:200]}"
        recalled = self.store.recall(query, top_k=3, tier=MemoryTier.EPISODIC)
        # 只看本 Agent 或者说安全相关的记忆
        prior = [m for m, _ in recalled if self.SKILL in (m.symptom_tags or [])]

        # --- ② 规则匹配：识别漏洞类别 ---
        matches = self._match_patterns(f"{title} {body}")

        if not matches:
            # 没有匹配到已知模式 → 诚实地说"没发现"
            #
            # ★ 这一点很重要：**不匹配不等于安全。**
            #   如果 Agent 说"没问题"，会给人虚假的安全感。
            #   正确的表述是"未匹配到已知模式，建议人工复核"。
            task.state = TaskState.INPUT_REQUIRED
            task.error = ""
            task.messages.append(Message(
                role="agent",
                parts=[Part.of_text(
                    "未匹配到已知的漏洞模式（CWE 库）。"
                    "这不代表该 issue 安全 —— 仅代表本次规则的覆盖范围之外。"
                    "建议转人工复核或补充更多上下文。")],
            ))
            self.audited.append({"issue_id": issue_id, "verdict": "no_match"})
            return

        top = matches[0]

        # --- ③ 写共享记忆：沉淀这次发现的模式 ---
        # ★ 只在"确实是新问题"时写，避免重复记忆污染。
        #   判断依据是有没有召回高度相似的历史。
        memory_written = False
        if not prior and issue_id:
            mem = MemoryRecord(
                memory_id=f"sec_{issue_id.lstrip('#')}",
                tier=MemoryTier.EPISODIC,
                issue_id=issue_id,
                author=issue_dict.get("author", ""),
                repo=issue_dict.get("repo", ""),
                symptom=f"{top['name']}（{top['id']}）：{title[:60]}",
                symptom_tags=[self.SKILL, top["id"], top["severity"]],
                root_cause=f"匹配到模式 {top['id']} {top['name']}",
                resolution=top["advice"],
                outcome=Outcome.RESOLVED,
            )
            memory_written = self.store.insert(mem)

        # --- ④ 回报 artifact ---
        result = {
            "issue_id": issue_id,
            "verdict": "vulnerability_found",
            "primary_cwe": top["id"],
            "name": top["name"],
            "severity": top["severity"],
            "advice": top["advice"],
            "all_matches": [m["id"] for m in matches],
            "prior_occurrences": len(prior),
            "prior_issue_ids": [m.issue_id for m in prior],
            "memory_written": memory_written,
        }
        task.artifacts.append(Artifact(
            name="security-audit-report",
            parts=[Part.of_data(result),
                   Part.of_text(
                       f"安全审计结论：{top['id']} {top['name']}"
                       f"（危害等级 {top['severity']}）\n"
                       f"修复建议：{top['advice']}"
                       + (f"\n历史上出现过 {len(prior)} 次同类问题："
                          f"{', '.join(m.issue_id for m in prior)}"
                          if prior else "\n历史记忆中没有同类问题（本次已沉淀）"))],
        ))
        task.state = TaskState.COMPLETED
        self.audited.append({"issue_id": issue_id, "verdict": top["id"],
                             "memory_written": memory_written})

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _match_patterns(text: str) -> list[dict]:
        """按严重程度排序返回所有匹配的模式。

        ★ 为什么返回全部匹配而不是第一个？
        因为一个 issue 可能同时命中多个类别（比如 tarfile 解压既涉及
        路径穿越 CWE-22，也可能涉及资源耗尽 CWE-400）。
        **只报第一个会漏掉信息** —— 而且严重程度排序能确保
        最紧急的那个排在最前面。
        """
        low = text.lower()
        found = []
        for p in SECURITY_PATTERNS:
            if any(kw.lower() in low for kw in p["patterns"]):
                found.append(p)
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(found, key=lambda x: order.get(x["severity"], 9))

    def cancel(self, task: A2ATask) -> None:
        """取消：安全审计是纯计算，没有副作用要回滚。

        ★ 但**这个方法必须存在** —— 因为协议要求可取消。
          如果某个 Agent 的实现是"不可取消"的，它应该在这里明确拒绝，
          而不是假装取消成功。
        """
        task.state = TaskState.CANCELED
