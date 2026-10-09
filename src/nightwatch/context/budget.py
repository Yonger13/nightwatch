"""
Nightwatch · 上下文预算调度器

================================================================================
这个模块解决什么问题
================================================================================
大模型的上下文窗口是**硬的容量约束**，而 Agent 的任务是**开放长度**的
（一个问题可能调 5 次工具就解决，也可能调 40 次）。

这两件事放在一起，必然导致"跑着跑着上下文满了"。而多数项目的处理方式很粗暴：
    "超过 N 轮就把前面的对话删掉"
这叫滑窗（sliding window），不是压缩，它会**静默地丢掉信息**，
然后 Agent 开始失忆、重复调用同一个工具、忘记最初的约束。

================================================================================
核心设计：分桶配额 + 有顺序的驱逐
================================================================================
关键认知是：**上下文的各部分价值不对等**，不能一视同仁地删。

    System（角色定义、工具 schema）  —— 不可压缩。删了 Agent 就不知道自己是干什么的
    Task（当前目标、硬约束）          —— 不可压缩。删了 Agent 就忘了在干什么
    Memory（召回的旧经验）            —— 可压缩（按相关性截断）
    Tool Results（工具返回）          —— 优先驱逐（体积最大，且落盘后能读回来）
    History（思考-行动-观察轨迹）      —— 其次压缩

为什么工具结果排在最前面驱逐？因为它**体积最大，且最容易"真扔了也不怕"** ——
卸载不是删除，是落盘 + 在上下文里留一个指针，需要细节时可以再读回来。
这个"保留指针"的做法是这一层设计的核心。

为什么 history 排在 tools 后面？因为 history 里有"我做过什么"的因果链。
压掉它，Agent 会重复调用同一个工具（因为它不记得自己已经调过了）。

================================================================================
面试必问题：压缩丢了关键信息怎么办
================================================================================
三层兜底：
  ① 压缩时保留指针（不真删，能回读）
  ② 关键信息不参与压缩（任务目标、硬约束、已确认事实抽成结构化状态卡）
  ③ 压缩过程可观测（每轮记录压了什么、多少 token，出问题能定位到是哪次压缩）

第 ② 层是最重要的 —— 它对应"长程一致性"这个难点：
    初始指令会在上下文堆叠中被挤到中部，而注意力在中部最弱。
    所以解法不是"让模型记住"，而是**每轮把状态卡重新注入到上下文末尾**。
================================================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import (
    COMPRESS_TRIGGER_RATIO, CONTEXT_BUDGET, CONTEXT_EVICTION_ORDER,
    TOOL_RESULT_OFFLOAD_THRESHOLD,
)


# =============================================================================
# Token 估算
# =============================================================================

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    ============================================================
    为什么要自己估，而不是用 tiktoken
    ============================================================
    因为这只是**调度用的相对量**，不需要精确。误差 20% 对"该不该卸载"的判断
    几乎没有影响 —— 因为阈值本身就有缓冲（85% 触发，留 15% 余量）。

    引入一个 tokenizer 依赖的代价（包体积、模型适配、版本问题）
    大于它带来的精度收益。这是很典型的工程取舍：
    **先问"我到底需要多准"，再决定要不要引入依赖。**

    经验值：中文约 1 字 = 1 token，英文约 4 字符 = 1 token。
    """
    if not text:
        return 0
    n = len(text)
    # 统计非 ASCII 字符（中文等）占比，两类字符的 token 密度不同
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = n - non_ascii
    return int(non_ascii * 1.0 + ascii_chars / 4.0) + 1


# =============================================================================
# 数据容器
# =============================================================================

@dataclass
class Bucket:
    """一个上下文分区。"""
    name: str
    quota_ratio: float        # 占总窗口的比例
    content: list[str] = field(default_factory=list)
    compressible: bool = True
    evictable: bool = True

    def tokens(self) -> int:
        return sum(estimate_tokens(c) for c in self.content)

    def render(self) -> str:
        return "\n".join(self.content)


@dataclass
class BudgetReport:
    """一次调度的结果报告 —— 用于可观测。

    **这个报告必须能被记录下来。**
    因为"压缩丢了信息"这个问题，唯一可靠的发现方式就是事后查 trace：
    这一轮压掉了什么、从多少压到多少。没有这个记录，压缩就是黑盒。
    """
    window_size: int
    usage: dict[str, int] = field(default_factory=dict)
    quota: dict[str, int] = field(default_factory=dict)
    evicted: list[str] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)
    offloaded: list[str] = field(default_factory=list)

    @property
    def total_used(self) -> int:
        return sum(self.usage.values())

    @property
    def utilization(self) -> float:
        return self.total_used / self.window_size if self.window_size else 0.0

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "usage": self.usage,
            "quota": self.quota,
            "utilization": round(self.utilization, 4),
            "evicted": self.evicted,
            "compressed": self.compressed,
            "offloaded": self.offloaded,
            "total_used": self.total_used,
        }


# =============================================================================
# 调度器
# =============================================================================

class ContextBudgetManager:
    """上下文预算调度器。

    用法：
        ctx = ContextBudgetManager(window_size=32000)
        ctx.set_system(系统提示词)
        ctx.set_task(任务状态卡)          # 结构化，不参与压缩
        ctx.inject_memory(召回的旧经验)
        ctx.add_tool_result(工具名, 结果)  # 超阈值会自动卸载
        ctx.add_history(一步的轨迹)
        messages = ctx.build()             # 组装成 messages
    """

    def __init__(self, window_size: int = 32000,
                 budget: dict | None = None,
                 offload_dir: str | Path | None = None):
        self.window_size = window_size
        self.cfg = {**CONTEXT_BUDGET, **(budget or {})}

        # 卸载目录：工具结果超过阈值时，全文落盘到这里
        self.offload_dir = Path(offload_dir or "./data/offload")
        self.offload_dir.mkdir(parents=True, exist_ok=True)

        self.buckets: dict[str, Bucket] = {
            # 前两个不可压缩 —— 这是整条设计的底线
            "system":  Bucket("system",  self.cfg["system"],  compressible=False, evictable=False),
            "task":    Bucket("task",    self.cfg["task"],    compressible=False, evictable=False),
            "memory":  Bucket("memory",  self.cfg["memory"]),
            "tools":   Bucket("tools",   self.cfg["tools"]),
            "history": Bucket("history", self.cfg["history"]),
        }
        self.reserve_ratio = self.cfg.get("reserve", 0.05)
        self.last_report: Optional[BudgetReport] = None

    # ---------------------------------------------------------------- 写入

    def set_system(self, text: str) -> None:
        self.buckets["system"].content = [text]

    def set_task(self, state_card: dict | str) -> None:
        """设置任务状态卡。

        **这是"长程一致性"的核心机制。**
        把当前目标、硬约束、已确认事实做成一个结构化的卡，每轮重新注入。
        这样即使 history 被压缩得只剩骨架，约束还在 ——
        因为约束不在 history 里，在 task 里。
        """
        if isinstance(state_card, dict):
            text = json.dumps(state_card, ensure_ascii=False, indent=2)
        else:
            text = str(state_card)
        self.buckets["task"].content = [text]

    def inject_memory(self, memories: list[str]) -> None:
        self.buckets["memory"].content = list(memories)

    def add_tool_result(self, tool_name: str, result: Any) -> bool:
        """加入工具结果。返回是否发生了卸载。

        超阈值时：
          全文写盘 → 上下文里只放"摘要 + 文件指针"
        这个做法是从 Claude Code / Deep Agents 那套实践里学的 ——
        **别真扔，留个能回读的引用。**
        """
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, default=str)

        if estimate_tokens(text) > TOOL_RESULT_OFFLOAD_THRESHOLD:
            ref = self._offload(tool_name, text)
            # 摘要有讲究：给"头 + 尾"而不是只给头。
            # 因为工具输出的尾部经常有结论/统计，而头部是原始数据。
            head = text[:600]
            tail = text[-300:] if len(text) > 900 else ""

            # ★ 文件指针必须**单独占一行**，不要包在括号或其他符号里。
            #
            # 这是踩过的坑：最初的格式是 "...(见 offload://xxx.txt)"，
            # 结果任何"提取指针"的正则（如 offload://(\S+)）都会把右括号
            # 一起吃进去，导致读回时路径错误、文件找不到。
            # 而且这种错误是静默的 —— 系统不会报错，只是"读回来是空的"。
            #
            # 教训：**机器要解析的字段，用独立的、边界明确的格式。**
            # 不要为了排版好看而把它塞进句子里。
            preview = (f"[{tool_name}] 结果过大已卸载\n"
                       f"摘要: {head}\n"
                       f"文件指针: {ref}\n"
                       + (f"尾部: {tail}\n" if tail else ""))
            self.buckets["tools"].content.append(preview)
            return True

        self.buckets["tools"].content.append(f"[{tool_name}] {text}")
        return False

    def add_history(self, text: str) -> None:
        self.buckets["history"].content.append(text)

    # ---------------------------------------------------------------- 调度

    def build(self) -> tuple[list[dict], BudgetReport]:
        """组装 messages。会先做调度（驱逐/压缩），再渲染。"""
        report = BudgetReport(window_size=self.window_size)
        quota = {k: int(self.window_size * self.cfg[k]) for k in self.buckets}
        quota["system"] = max(quota["system"], self.buckets["system"].tokens())
        quota["task"] = max(quota["task"], self.buckets["task"].tokens())
        report.quota = quota

        # --- 第 1 步：算总占用，判断是否超窗口 ---
        # reserve 是留给模型输出的空间。**这部分绝不能动** ——
        # 如果上下文占满了，模型没有地方写答案，会返回空或者被截断。
        usable = self.window_size - int(self.window_size * self.reserve_ratio)
        while self._total_tokens() > usable:
            if not self._evict_once(report):
                break
        # --- 第 2 步：逐桶检查配额，超了先压缩 ---
        for name in CONTEXT_EVICTION_ORDER:
            b = self.buckets[name]
            if b.compressible and b.tokens() > quota[name] * COMPRESS_TRIGGER_RATIO:
                self._compress(name, report)

        report.usage = {k: b.tokens() for k, b in self.buckets.items()}
        self.last_report = report
        return self._render(), report

    def _total_tokens(self) -> int:
        return sum(b.tokens() for b in self.buckets.values())

    def _evict_once(self, report: BudgetReport) -> bool:
        """驱逐一个单元。返回是否成功驱逐。

        顺序由 CONTEXT_EVICTION_ORDER 决定（tools → history → memory）。
        每次都驱逐**最早**的那一条（FIFO）——
        因为最近的信息相关性最高。

        注意：不可驱逐的桶（system/task）永远跳过，这是硬约束。
        """
        for name in CONTEXT_EVICTION_ORDER:
            b = self.buckets[name]
            if not b.evictable or not b.content:
                continue
            if len(b.content) <= 1:
                continue      # 至少保留一条，避免桶被清空
            removed = b.content.pop(0)
            report.evicted.append(f"{name}:{removed[:40]}")
            return True
        return False

    def _compress(self, name: str, report: BudgetReport) -> None:
        """压缩一个桶。

        ============================================================
        这里是最容易出问题的地方，所以做法很保守
        ============================================================
        压缩策略：保留**第一条**（通常是最初的指令/目标）+ 最近 K 条，
        中间的部分压成一行摘要。

        为什么不压中间？等等 —— 恰恰是压中间。
        因为头和尾巴最重要：
          头：最初的目标和约束
          尾：刚刚发生的事
          中：被"挤到中部"的那部分，也正是注意力最弱的地方

        这解释了那个现象：**为什么 Agent 跑久了会忘记最初的指令** ——
        不是模型记性差，是初始指令在上下文里被挤到中部了。

        所以本项目真正的解法不是在压缩时小心，而是第 ② 层兜底：
        把约束抽出来放在 task 桶里（不参与压缩），每轮重新注入。
        """
        b = self.buckets[name]
        if len(b.content) <= 3:
            return

        head = b.content[0]
        tail = b.content[-2:]
        middle = b.content[1:-2]

        # 摘要用"计数 + 关键动作"而不是调模型 —— 调模型会让压缩本身变慢变贵，
        # 而这一层的目的只是控制体积，不是提炼信息（信息提炼在记忆层做）
        summary = f"[已压缩 {len(middle)} 条历史] 期间执行的动作: " + \
                  "、".join(self._brief(x) for x in middle[:8])
        if len(middle) > 8:
            summary += f" …等 {len(middle)} 项"

        before = b.tokens()
        b.content = [head, summary] + tail
        report.compressed.append(f"{name}:{before}→{b.tokens()}")

    @staticmethod
    def _brief(text: str) -> str:
        """提取一条记录的简短标识。"""
        t = text.strip().replace("\n", " ")
        return t[:28] + ("…" if len(t) > 28 else "")

    def _offload(self, tool_name: str, text: str) -> str:
        """把大结果写到磁盘，返回可读回的引用路径。"""
        from ..models import _hash
        fname = f"{tool_name}_{_hash(text[:200])}.txt"
        path = self.offload_dir / fname
        path.write_text(text, encoding="utf-8")
        return f"offload://{fname}"

    def read_offload(self, ref: str) -> str:
        """把卸载的内容读回来。

        这个能力必须有 —— 否则"卸载"就变成了"丢弃"，
        只是换了个好听的名字。
        """
        if not ref.startswith("offload://"):
            return ""
        path = self.offload_dir / ref.replace("offload://", "")
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _render(self) -> list[dict]:
        """渲染成 messages。

        **注意顺序**：system → task 状态卡 → memory → history → tools

        为什么 task 紧跟 system？因为它包含硬约束，
        放在前面 + 每次重建，是对抗"指令漂移"的手段。
        （更强的做法是在 history 末尾再重复一次关键约束 —— 这个可以自己加。）
        """
        msgs: list[dict] = []
        if self.buckets["system"].content:
            msgs.append({"role": "system", "content": self.buckets["system"].render()})

        if self.buckets["task"].content:
            msgs.append({"role": "system",
                         "content": "[当前任务状态]\n" + self.buckets["task"].render()})

        if self.buckets["memory"].content:
            msgs.append({"role": "system",
                         "content": "以下是与当前问题相关的历史经验（供参考，不是绝对事实）：\n"
                                    + self.buckets["memory"].render()})

        for h in self.buckets["history"].content:
            msgs.append({"role": "assistant", "content": h})

        if self.buckets["tools"].content:
            msgs.append({"role": "user",
                         "content": "[工具返回结果]\n" + self.buckets["tools"].render()})

        return msgs
