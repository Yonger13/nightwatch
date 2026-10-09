"""
Nightwatch · 四层可靠性机制

================================================================================
这个文件在应对一道具体的面试题
================================================================================
    面试官："Agent 调了三个工具开始死循环，你的异常处理写在哪？"
                                    —— 米哈游 Agent 提前批一面（这道题挂了人）

这道题为什么能挂人？因为它考的不是"你知道不知道 try/except"，
而是**"你有没有在真实系统里见过 Agent 失控"**。

没见过的人的答案通常是："我会加 try/except 捕获异常" —— 但这个答案解决不了问题，
因为死循环里的每一次调用**都是成功的**。没有异常可以捕获，Agent 就是在一个
合法的循环里反复干同一件没用的事，直到把 token 烧完。

================================================================================
四层是递进的，每一层堵上一层的漏
================================================================================

    ① 重试 Retry        —— 单次调用失败（超时、限流）
                           漏网：调用都成功，但整体在做无用功
                              ↓
    ② 熔断 Circuit      —— 某个工具反复失败（它可能已经挂了，别再浪费时间）
                           漏网：工具没失败，但 Agent 在原地打转
                              ↓
    ③ 循环检测 Loop     —— 检测到重复的 (工具, 参数) 组合，主动中断
                           漏网：没有完全重复，但是在缓慢地兜圈子
                              ↓
    ④ 轮数上限 + 兜底    —— 强制退出，并且**带着已收集的证据**转人工

================================================================================
设计要点：第 ④ 层的兜底必须是"有产出的失败"
================================================================================
这是最容易被忽略、但业务上最重要的一点：

    差的兜底：raise TimeoutError("超过最大轮数")
              → 上游收到异常，等于这次白干了，人也无从接手

    好的兜底：return {"status": "unfinished",
                       "reason": "达到 20 轮上限",
                       "evidence": {...已查到的所有信息...},
                       "suggestion": "建议人工介入"}
              → 人拿着这些证据可以接着往下查

**前者是技术的失败，后者是技术的降级。** 这两者在生产环境里差别巨大。
================================================================================
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import RELIABILITY


# =============================================================================
# ① 重试
# =============================================================================

@dataclass
class RetryPolicy:
    """指数退避重试。

    ============================================================
    关键设计：区分"可重试"和"不可重试"的错误
    ============================================================
    这是很多人漏掉的一点。不加区分的重试是有害的：

      超时 / 限流 / 连接失败  → 重试有意义（可能是暂时的）
      参数错误 / 权限不足      → 重试一万次结果一样，**快速失败才是对的**

    后者如果也重试，会让一个本该 100ms 返回的错误拖到 10 秒，
    而且占着一个并发槽位 —— 在服务化场景里这是实打实的资源浪费。
    """

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retryable_kinds: tuple[str, ...] = ("timeout", "rate_limit", "connection")

    def should_retry(self, error_kind: str, attempt: int) -> bool:
        if attempt >= self.max_retries - 1:
            return False
        return error_kind in self.retryable_kinds

    def delay_for(self, attempt: int) -> float:
        """指数退避：0.5 → 1 → 2 → 4（封顶 8）

        真实生产环境还应该加 **jitter（随机抖动）**：
            delay = base * 2**attempt * (0.5 + random()*0.5)
        原因：如果 1000 个客户端同时被限流，它们会在同一时刻一起重试，
        形成"惊群"，把服务再打挂一次。抖动把重试时间打散。
        这个点面试时说出来，能体现你有分布式系统的意识。
        """
        return min(self.base_delay * (2 ** attempt), self.max_delay)

    def execute(self, fn: Callable[[], Any], classify: Callable[[Exception], str]
                ) -> tuple[Any, Optional[Exception]]:
        """执行带重试。返回 (结果, 最终异常)。

        注意这里返回 (None, err) 而不是抛异常 ——
        让调用方决定怎么处理，因为"工具失败了"在 Agent 里是**正常情况**，
        不是需要终止流程的异常。Agent 应该能容忍工具失败并换个策略。
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return fn(), None
            except Exception as e:      # noqa: BLE001 - 这里必须捕获所有异常
                last_err = e
                kind = classify(e)
                if not self.should_retry(kind, attempt):
                    break
                time.sleep(self.delay_for(attempt))
        return None, last_err


# =============================================================================
# ② 熔断
# =============================================================================

class CircuitBreaker:
    """按工具粒度的熔断器。

    ============================================================
    为什么需要它（重试已经存在了）
    ============================================================
    重试解决"这一次失败"，熔断解决"这个工具整体已经不能用了"。

    场景：某个工具连续 3 次超时。可能是它依赖的外部服务挂了。
    这时候正确的做法不是继续重试它（每次重试都要等超时），
    而是**把它暂时摘掉**，让 Agent 用别的工具，或者直接走兜底。

    这和微服务里的熔断是同一个概念 —— **Agent 的工具层本质上就是一个
    微服务依赖图，所以后端那套可靠性模式可以直接迁移过来。**
    （面试时这个类比很有说服力：它说明你理解 Agent 工程的本质是分布式系统。）
    """

    def __init__(self, fail_threshold: int = 3, cooldown: float = 60.0):
        self.fail_threshold = fail_threshold
        self.cooldown = cooldown
        self._fails: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, tool_name: str) -> bool:
        """熔断是否生效。"""
        if tool_name not in self._opened_at:
            return False
        if time.time() - self._opened_at[tool_name] > self.cooldown:
            # 冷却结束 → 半开状态：允许放一次请求进去试探
            self._opened_at.pop(tool_name, None)
            self._fails[tool_name] = 0
            return False
        return True

    def record_success(self, tool_name: str) -> None:
        self._fails[tool_name] = 0
        self._opened_at.pop(tool_name, None)

    def record_failure(self, tool_name: str) -> bool:
        """记一次失败。返回是否触发了熔断。"""
        self._fails[tool_name] = self._fails.get(tool_name, 0) + 1
        if self._fails[tool_name] >= self.fail_threshold:
            self._opened_at[tool_name] = time.time()
            return True
        return False

    def open_tools(self) -> list[str]:
        return [t for t in self._opened_at if self.is_open(t)]


# =============================================================================
# ③ 循环检测
# =============================================================================

class LoopDetector:
    """检测 Agent 在原地打转。

    ============================================================
    判据的选择 —— 这是这一层最关键的设计决策
    ============================================================
    直觉上会想"连续调两次同一个工具就算循环"。这是错的：
    **正常业务里连续调同一个工具非常常见** ——
    比如先用关键词 A 搜一遍，觉得不对再用关键词 B 搜一遍。

    真正的循环特征是 **参数完全相同**：
        search_issues(query="timeout")
        search_issues(query="timeout")
        search_issues(query="timeout")      ← 这才是卡住了

    所以判据是归一化后的 (工具名, 参数) 三元组完全一致。

    ============================================================
    下一个问题：误判怎么办
    ============================================================
    会产生误判。正常业务里也可能出现"参数一样但确实需要再调一次"
    （比如查一个会变化的状态，第一次和第二次结果不同）。

    处理方式不是"想办法完全避免误判"，而是：
      ① 要求连续 N 次（默认 4）才判循环 —— 单次重复不触发
      ② **把误判率记录下来，作为可观测指标**
         （trace 里有 loop_detected 标记，人工复核时能发现误判）

    第 ② 点是态度问题：承认检测器不完美，但要能量化它的不完美。
    面试时这么说比"我的检测很准"可信得多。
    """

    def __init__(self, window: int = 4, max_repeats: int = 3):
        self.window = window
        self.max_repeats = max_repeats
        self._recent: deque[str] = deque(maxlen=window)

    def observe(self, tool_name: str, tool_input: dict) -> bool:
        """记录一次调用，返回是否检测到循环。"""
        key = self._signature(tool_name, tool_input)
        self._recent.append(key)

        if len(self._recent) < self.max_repeats:
            return False

        # 最近 max_repeats 次是不是完全一样？
        tail = list(self._recent)[-self.max_repeats:]
        return len(set(tail)) == 1

    @staticmethod
    def _signature(tool_name: str, tool_input: dict) -> str:
        """归一化的调用签名。

        为什么叫"归一化"？因为同样的查询意图可能写成不同的形式：
            {"query": "超时问题", "top_k": 5}
            {"top_k": 5, "query": "超时问题"}      ← 键顺序不同
            {"query": "超时问题 ", "top_k": 5}     ← 多了个空格
        这些在语义上完全是同一次调用，所以要先规范化再比对。
        """
        import json
        try:
            normalized = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            normalized = str(tool_input)
        # 去空白 + 统一大小写，消除格式差异
        normalized = "".join(normalized.split()).lower()
        return f"{tool_name}::{normalized}"

    def reset(self) -> None:
        self._recent.clear()


# =============================================================================
# ④ 兜底：有产出的失败
# =============================================================================

@dataclass
class FallbackResult:
    """达到上限时的兜底产出。

    **注意这不是异常，是一个正常的返回值。**
    这个区别就是"降级"和"失败"的区别。
    """
    status: str = "unfinished"
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    suggestion: str = "任务未完成，建议人工介入"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
        }


class ReliabilityGuard:
    """把四层打包成一个门面，主循环只跟它打交道。

    这样设计的好处：主循环的代码只关注"任务逻辑"，
    可靠性是横切关注点，不应该混在里面。
    （这就是 AOP / 中间件思想在 Agent 里的应用。）
    """

    def __init__(self, cfg: dict | None = None):
        c = {**RELIABILITY, **(cfg or {})}
        self.cfg = c
        self.retry = RetryPolicy(
            max_retries=c["max_retries"],
            base_delay=c["retry_base_delay"],
            max_delay=c["retry_max_delay"],
        )
        self.breaker = CircuitBreaker(
            fail_threshold=c["circuit_fail_threshold"],
            cooldown=c["circuit_cooldown"],
        )
        self.loop = LoopDetector(window=c["loop_window"])
        self.max_steps = c["max_steps"]

        # 统计：这些数字会进 trace，成为可观测的一部分
        self.stats = {
            "retries": 0,
            "circuit_trips": 0,
            "loops_detected": 0,
            "steps_executed": 0,
            "fallbacks": 0,
        }

    # --- 门面方法 ---

    def before_tool_call(self, tool_name: str, tool_input: dict) -> Optional[str]:
        """调工具前的检查。返回非 None 表示应该阻止调用，值是阻止原因。"""
        if self.breaker.is_open(tool_name):
            return (f"工具 {tool_name} 已熔断"
                    f"（连续失败 {self.breaker.fail_threshold} 次），"
                    f"冷却 {self.breaker.cooldown:.0f}s 内不再尝试")

        if self.loop.observe(tool_name, tool_input):
            self.stats["loops_detected"] += 1
            return (f"检测到循环：连续 {self.loop.max_repeats} 次调用 "
                    f"{tool_name} 且参数完全相同")

        return None

    def after_tool_call(self, tool_name: str, ok: bool) -> None:
        if ok:
            self.breaker.record_success(tool_name)
        else:
            if self.breaker.record_failure(tool_name):
                self.stats["circuit_trips"] += 1

    def check_step_limit(self, step: int) -> bool:
        """是否已超轮数上限。"""
        return step >= self.max_steps

    def make_fallback(self, step: int, reason: str, trace) -> FallbackResult:
        """构造有产出的兜底结果。

        **evidence 是关键** —— 把已经查到的所有信息打包带出去，
        让人能接着往下做。这是"失败"和"降级"的分界线。
        """
        self.stats["fallbacks"] += 1

        evidence = {
            "steps_executed": step,
            "tools_called": [
                {"tool": s.action.tool_name, "ok": s.action.ok}
                for s in trace.steps if s.action is not None
            ],
            "partial_findings": [
                s.observation[:200] for s in trace.steps if s.observation
            ][-3:],
            "circuit_open": self.breaker.open_tools(),
        }

        return FallbackResult(
            status="unfinished",
            reason=reason,
            evidence=evidence,
            suggestion=("建议人工介入。已收集的证据见 evidence 字段；"
                        "若为循环导致，请检查工具描述是否让模型产生了误解。"),
            confidence=0.0,
        )
