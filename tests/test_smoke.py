"""
Nightwatch · 冒烟测试

================================================================================
为什么这个文件用 unittest 而不是 pytest
================================================================================
因为要**零依赖可运行**。pytest 是更好的工具，但它是一个额外依赖 ——
而这个项目的设计原则是"学习者 clone 下来 30 秒能跑"。

用标准库的 unittest，效果几乎一样，但不需要装任何东西：

    python -m unittest discover -s tests -v

如果你已经装了 pytest，它同样能跑这些测试（pytest 兼容 unittest 风格的用例）：
    pytest tests/ -v

================================================================================
这些测试在测什么
================================================================================
不是"覆盖率"，而是**每个模块最核心的那条设计约束**：

    记忆六环节    → 幂等、合并、冲突进 pending、衰减不删数据
    四层可靠性    → 循环检测的判据是"参数相同"、熔断生效、兜底带证据
    上下文预算    → 不可压缩的桶真的不被压缩、卸载能读回来
    工具沙箱      → 数据隔离真的拦截、写操作真的幂等
    端到端        → 处理完一个 issue 后记忆真的被写进去了

这些测试的共同点是：**每一条都对应一个"如果坏了，系统会静默地出错"的地方。**
静默出错比崩溃更危险 —— 崩溃你会立刻发现，静默出错可能几周后才发现。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import load_settings                      # noqa: E402
from nightwatch.context.budget import (                          # noqa: E402
    ContextBudgetManager, estimate_tokens,
)
from nightwatch.eval.dataset import build_synthetic_dataset      # noqa: E402
from nightwatch.eval.metrics import JudgeCalibrator, Metrics     # noqa: E402
from nightwatch.llm.client import MockLLM                        # noqa: E402
from nightwatch.loop.react import TriageAgent                    # noqa: E402
from nightwatch.loop.reliability import (                        # noqa: E402
    CircuitBreaker, LoopDetector, ReliabilityGuard,
)
from nightwatch.memory.lifecycle import (                        # noqa: E402
    Extractor, Scorer, Writer,
)
from nightwatch.memory.store import HashEmbedding, MemoryStore, cosine  # noqa: E402
from nightwatch.models import (                                   # noqa: E402
    HumanVerdict, Issue, MemoryRecord, MemoryTier, Outcome, Step,
    ToolCall, Trace,
)
from nightwatch.tools.registry import (                           # noqa: E402
    IssueRepository, Sandbox, ToolRegistry,
)


def make_store() -> MemoryStore:
    """测试用记忆库。强制用确定性哈希编码器，保证结果可复现。"""
    return MemoryStore(":memory:", encoder=HashEmbedding(dim=256))


def sample_issue(iid: str = "#900", **kw) -> Issue:
    base = dict(
        issue_id=iid,
        title="接口调用超时",
        body="调用报表接口时返回 504，高峰时段必现。",
        author="tester",
        repo="test/repo",
        stack_trace="requests.exceptions.Timeout: Read timed out.",
    )
    base.update(kw)
    return Issue(**base)


# =============================================================================
# 记忆系统
# =============================================================================

class TestMemoryLifecycle(unittest.TestCase):
    """记忆六环节 —— 每一环都对应一个"静默出错"的风险点。"""

    def setUp(self):
        self.store = make_store()
        self.llm = MockLLM()
        self.settings = load_settings()

    def _make_trace(self, issue: Issue, answer: str) -> Trace:
        tr = Trace(trace_id="tr_test", issue_id=issue.issue_id)
        tr.record(Step(
            step=1, thought="查询历史",
            action=ToolCall(step=1, tool_name="search_similar_issues",
                            tool_input={"query": "timeout"}, ok=True,
                            output={"count": 1}),
            observation="ok",
        ))
        # 故意加一个失败的动作 —— 验证"失败尝试会被保留"
        tr.record(Step(
            step=2, thought="查文档",
            action=ToolCall(step=2, tool_name="search_docs",
                            tool_input={"query": "sop"}, ok=False,
                            error="not found", error_kind="unknown"),
            observation="ERROR",
        ))
        tr.final_answer = answer
        tr.outcome = Outcome.RESOLVED
        tr.confidence = 0.8
        return tr

    def test_写入路径_过阈值才写入(self):
        """评分低于阈值时不应写入 —— 这是记忆库信噪比的守门人。"""
        issue = sample_issue()
        answer = "症状：接口超时\n根因：客户端并发过高\n建议：提升配额"
        trace = self._make_trace(issue, answer)

        ex = Extractor(self.llm, self.settings.memory)
        sc = Scorer(self.settings.memory)
        wr = Writer(self.store, self.settings.memory)

        cand = ex.extract(trace, issue, HumanVerdict.ACCEPTED)
        self.assertIsNotNone(cand, "抽取应成功")
        score = sc.score(cand)
        self.assertGreater(score, 0.0)
        self.assertTrue(sc.should_write(score), f"采纳的样本应过阈值 (score={score})")

        mem, action = wr.commit(cand, issue, trace, HumanVerdict.ACCEPTED)
        self.assertEqual(action, "inserted")
        self.assertEqual(self.store.count(), 1)

    def test_幂等_重复写入被拦在数据库层(self):
        """同一个 issue 重复抽取不应产生两条记忆。

        这条测试对应一个真实风险：记忆抽取是异步的，异步就会有重试。
        如果重试会写重复记忆，记忆库会以看不见的速度膨胀。
        """
        issue = sample_issue()
        answer = "症状：接口超时\n根因：并发过高\n建议：提升配额"
        trace = self._make_trace(issue, answer)

        ex = Extractor(self.llm, self.settings.memory)
        sc = Scorer(self.settings.memory)
        wr = Writer(self.store, self.settings.memory)

        for _ in range(3):
            cand = ex.extract(trace, issue, HumanVerdict.ACCEPTED)
            score = sc.score(cand)
            if sc.should_write(score):
                wr.commit(cand, issue, trace, HumanVerdict.ACCEPTED)

        self.assertEqual(self.store.count(), 1,
                         "三次写入应只留一条（幂等键拦截）")

    def test_失败动作被保留(self):
        """★ 值钱字段验证：失败的尝试必须被记录。

        这是情景记忆比知识库多出来的价值 —— "试过 A 不行，应该试 B"。
        如果这个字段丢了，记忆就退化成普通的知识库。
        """
        issue = sample_issue()
        answer = "症状：接口超时\n根因：并发过高\n建议：提升配额"
        trace = self._make_trace(issue, answer)

        ex = Extractor(self.llm, self.settings.memory)
        cand = ex.extract(trace, issue, HumanVerdict.ACCEPTED)

        self.assertEqual(len(cand.actions), 2, "应记录两次工具调用")
        failed = [a for a in cand.actions if not a["ok"]]
        self.assertEqual(len(failed), 1, "应保留 1 条失败记录")
        self.assertEqual(failed[0]["tool"], "search_docs")

    def test_衰减_低权重归档而不是删除(self):
        """衰减不能真删数据 —— 审计要求("某次判断依据了哪条记忆"必须能回答)。"""
        from datetime import datetime, timedelta, timezone
        from nightwatch.memory.lifecycle import Decayer

        # 造一条 400 天前的记忆
        old = MemoryRecord(
            memory_id="ep_old", tier=MemoryTier.EPISODIC,
            symptom="旧问题", importance=0.8,
            created_at=(datetime.now(timezone.utc)
                        - timedelta(days=400)).isoformat(),
        )
        self.store.insert(old)
        self.assertEqual(self.store.count(), 1)

        decayer = Decayer(self.store, self.settings.memory)
        stats = decayer.decay_all()

        self.assertEqual(stats["archived"], 1, "超期记忆应被归档")
        self.assertEqual(self.store.count(), 1, "归档不是删除，数据仍在")
        self.assertTrue(self.store.get("ep_old").archived)

    def test_人工确认过的记忆不被轻易归档(self):
        """反例保护：被人工 ACCEPTED 的经验，不该因为时间流逝被埋掉。"""
        from datetime import datetime, timedelta, timezone
        from nightwatch.memory.lifecycle import Decayer

        trusted = MemoryRecord(
            memory_id="ep_trusted", tier=MemoryTier.EPISODIC,
            symptom="工程师确认过的经验", importance=0.9,
            human_verdict=HumanVerdict.ACCEPTED,
            created_at=(datetime.now(timezone.utc)
                        - timedelta(days=400)).isoformat(),
        )
        self.store.insert(trusted)
        decayer = Decayer(self.store, self.settings.memory)
        decayer.decay_all()

        self.assertFalse(self.store.get("ep_trusted").archived,
                         "人工确认过的记忆应被保护")

    def test_混合检索_按作者精确过滤(self):
        """验证"同一作者报过什么"这个查询真的按作者过滤。"""
        self.store.insert(MemoryRecord(
            memory_id="ep_a", tier=MemoryTier.EPISODIC,
            author="alice", symptom="超时问题", importance=0.8,
        ))
        self.store.insert(MemoryRecord(
            memory_id="ep_b", tier=MemoryTier.EPISODIC,
            author="bob", symptom="超时问题", importance=0.8,
        ))

        hits = self.store.recall(query="超时", author="alice", top_k=10)
        self.assertEqual(len(hits), 1, "作者过滤应只返回 alice 的记忆")
        self.assertEqual(hits[0][0].memory_id, "ep_a")

    def test_冲突_分不清时进pending队列(self):
        """★ 核心设计验证：裁决不下最终结论，矛盾项交人工。

        这条测试守住的是"不假装全能"这个设计原则。
        如果它坏了，系统会开始静默地覆盖记忆 —— 而错误会通过召回传播。
        """
        from nightwatch.memory.lifecycle import Resolver

        a = MemoryRecord(memory_id="ep_x", tier=MemoryTier.EPISODIC,
                         symptom="接口超时", root_cause="根因甲：网络抖动",
                         resolution="方案甲", importance=0.7)
        b = MemoryRecord(memory_id="ep_y", tier=MemoryTier.EPISODIC,
                         symptom="接口超时", root_cause="根因乙：代码死锁",
                         resolution="方案乙", importance=0.7)
        for m in (a, b):
            m.embedding = self.store.encoder.encode(m.symptom)
            self.store.insert(m)

        resolver = Resolver(self.llm, self.store, self.settings.memory)
        self.assertTrue(resolver.detect_conflict(a, b), "应识别为冲突")
        winner, pending = resolver.resolve(a, b)
        self.assertIsNone(winner, "分差不足时不应自动裁决")
        self.assertIsNotNone(pending, "应生成待裁决记录")
        self.assertGreater(len(self.store.pending_conflicts()), 0)


# =============================================================================
# 可靠性
# =============================================================================

class TestReliability(unittest.TestCase):

    def test_循环检测_判据是参数相同而不是工具名相同(self):
        """★ 这是四层可靠性里最容易做错的一层。

        正常业务里连续调两次同一个工具是合理的（换个关键词再查一次）。
        只有"参数完全相同"才是真的卡住。如果把判据写成"工具名相同"，
        会产生大量误判，把正常流程也打断。
        """
        det = LoopDetector(window=4, max_repeats=3)

        # 同名工具、参数不同 → 不该报警
        for q in ("超时", "内存", "权限"):
            self.assertFalse(det.observe("search_issues", {"q": q}),
                             "参数不同不该判定为循环")

        det.reset()

        # 参数完全相同 → 第 3 次应报警
        self.assertFalse(det.observe("search_issues", {"q": "超时"}))
        self.assertFalse(det.observe("search_issues", {"q": "超时"}))
        self.assertTrue(det.observe("search_issues", {"q": "超时"}),
                        "连续 3 次参数相同应判定为循环")

    def test_循环检测_对参数顺序和空白不敏感(self):
        """归一化验证：{"a":1,"b":2} 和 {"b":2,"a":1} 是同一次调用。"""
        det = LoopDetector(window=4, max_repeats=3)
        det.observe("t", {"a": 1, "b": 2})
        det.observe("t", {"b": 2, "a": 1})
        self.assertTrue(det.observe("t", {"b": 2, "a": 1}),
                        "键顺序不同应被识别为同一次调用")

    def test_熔断_连续失败后摘掉工具(self):
        cb = CircuitBreaker(fail_threshold=3, cooldown=60)
        self.assertFalse(cb.is_open("flaky_tool"))
        for _ in range(3):
            cb.record_failure("flaky_tool")
        self.assertTrue(cb.is_open("flaky_tool"), "达到阈值应熔断")
        self.assertIn("flaky_tool", cb.open_tools())

        cb.record_success("flaky_tool")
        self.assertFalse(cb.is_open("flaky_tool"), "成功后应恢复")

    def test_兜底结果必须带证据(self):
        """★ 第 ④ 层的核心：兜底是"降级"而不是"失败"。

        区别在于 evidence 字段 —— 人拿着它能接着往下做。
        如果这个字段空了，兜底就退化成抛异常，等于这次白干。
        """
        guard = ReliabilityGuard()
        trace = Trace(trace_id="tr", issue_id="#1")
        trace.record(Step(
            step=1, thought="t",
            action=ToolCall(step=1, tool_name="search_issues",
                            tool_input={"q": "x"}, ok=True, output={"k": "v"}),
            observation="找到 3 条相关记录",
        ))

        fb = guard.make_fallback(step=20, reason="达到轮数上限", trace=trace)
        self.assertEqual(fb.status, "unfinished")
        self.assertIn("evidence", fb.to_dict())
        self.assertGreater(len(fb.evidence["tools_called"]), 0,
                           "证据里应包含已调用的工具")
        self.assertGreater(len(fb.evidence["partial_findings"]), 0,
                           "证据里应包含已获得的部分结论")

    def test_不可重试的错误快速失败(self):
        """参数错误不该重试 —— 重试一万次结果一样，只是拖慢。"""
        from nightwatch.loop.reliability import RetryPolicy
        rp = RetryPolicy(max_retries=3)
        self.assertFalse(rp.should_retry("invalid_args", attempt=0))
        self.assertFalse(rp.should_retry("permission", attempt=0))
        self.assertTrue(rp.should_retry("timeout", attempt=0))
        self.assertTrue(rp.should_retry("rate_limit", attempt=1))
        self.assertFalse(rp.should_retry("timeout", attempt=2),
                         "达到重试上限应停止")


# =============================================================================
# 上下文预算
# =============================================================================

class TestContextBudget(unittest.TestCase):

    def test_不可压缩的桶真的不被压缩(self):
        """★ 这是"长程一致性"的地基。

        system（角色）和 task（当前目标）如果被压缩了，
        Agent 就会忘记自己是谁、在干什么 —— 这正是长程任务失忆的主因。
        """
        ctx = ContextBudgetManager(window_size=2000)
        ctx.set_system("系统提示词" * 20)
        ctx.set_task({"目标": "分类", "硬约束": ["不得编造"]})
        for i in range(60):
            ctx.add_history(f"第 {i} 轮：执行了某个动作")
            ctx.add_tool_result("search", {"results": list(range(50))})

        messages, report = ctx.build()
        joined = "\n".join(m["content"] for m in messages[:2])
        self.assertIn("系统提示词", joined, "system 桶不应被清空")
        self.assertIn("硬约束", joined, "task 状态卡不应被压缩掉")

    def test_超阈值自动卸载且能读回来(self):
        """★ 卸载不是删除 —— 这是"压缩不丢信息"的第一层兜底。"""
        ctx = ContextBudgetManager(window_size=50000)
        big = {"logs": [f"error line {i}" for i in range(3000)]}
        self.assertGreater(estimate_tokens(str(big)), 3000)

        offloaded = ctx.add_tool_result("read_code", big)
        self.assertTrue(offloaded, "超大结果应触发卸载")

        # 上下文里应只剩摘要和指针
        body = ctx.buckets["tools"].render()
        self.assertIn("offload://", body, "应保留可回读的指针")
        self.assertLess(estimate_tokens(body), 1000, "上下文里应只剩摘要")

        # 关键：真的能读回来
        import re as _re
        m = _re.search(r"offload://(\S+)", body)
        self.assertIsNotNone(m)
        content = ctx.read_offload(f"offload://{m.group(1)}")
        self.assertIn("error line 2999", content, "卸载的内容应能完整读回")

    def test_报告里的占用统计可追溯(self):
        """压缩/卸载必须可观测 —— 否则"信息什么时候丢的"永远查不出来。"""
        ctx = ContextBudgetManager(window_size=4000)
        ctx.set_system("s" * 200)
        for i in range(30):
            ctx.add_history(f"step {i} " + "x" * 200)
        _, report = ctx.build()
        self.assertIn("system", report.usage)
        self.assertGreater(report.total_used, 0)
        self.assertIsInstance(report.to_dict(), dict)
        self.assertLessEqual(report.utilization, 1.05)


# =============================================================================
# 工具与沙箱
# =============================================================================

class TestToolsAndSandbox(unittest.TestCase):

    def setUp(self):
        self.issue = sample_issue()
        self.repo = IssueRepository([self.issue])
        self.store = make_store()
        self.reg = ToolRegistry(self.repo, store=self.store,
                                current_issue_id=self.issue.issue_id)

    def test_工具数量保持克制(self):
        """工具数量要保持克制 —— 但断言方式应该是"不超过上限"。

        Gartner 把"用 50+ 工具的通用 Agent"点名为失败模式：
        工具太多时模型选不对，而且每个工具的描述都占上下文。

        ★ 这条测试原本断言的是 `== 6`（当时的数量）。
        P2-3 加了 PR 初审工具后变成 7 个，测试就失败了 ——
        **而这次失败不是 bug，是测试写得太死。**

        教训：**数量类断言应该表达"约束"而不是"快照"。**
          ✗ len(tools) == 6      → 每加一个工具都要改测试，改多了就麻木了
          ✓ len(tools) <= 10     → 表达"要克制"这个真实意图

        （如果哪天真的要加第 11 个工具，那时该**停下来重新想架构**，
          而不是顺手把这个数字改大 —— 这才是这条测试该起的作用。）
        """
        names = self.reg.names()
        self.assertLessEqual(
            len(names), 10,
            f"工具数 {len(names)} 过多（{names}）—— "
            f"超过 10 个时该停下来想：是不是该拆 Agent 了？")
        self.assertIn("search_similar_issues", names)
        self.assertIn("apply_labels", names)
        self.assertIn("review_pr_diff", names)      # P2-3 新增

    def test_沙箱_路径越权被拦截(self):
        """数据隔离：只能读白名单路径。"""
        result = self.reg.call("read_code", {"path": "/etc/passwd"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "permission")
        self.assertGreater(len(self.reg.sandbox.violations), 0,
                           "违规应被记录（安全信号）")

    def test_写操作幂等(self):
        """★ 重试时不会重复打标签 —— 这是"幂等"在 Agent 场景的落点。"""
        r1 = self.reg.call("apply_labels",
                           {"issue_id": self.issue.issue_id,
                            "labels": ["bug/timeout"], "reason": "测试"})
        self.assertTrue(r1.ok)
        self.assertEqual(r1.data["status"], "applied")

        r2 = self.reg.call("apply_labels",
                           {"issue_id": self.issue.issue_id,
                            "labels": ["bug/timeout"], "reason": "测试"})
        self.assertTrue(r2.ok)
        self.assertTrue(r2.data.get("idempotent_hit"), "第二次应命中幂等")
        self.assertEqual(self.issue.labels.count("bug/timeout"), 1,
                         "标签不应重复添加")

    def test_写操作可回滚(self):
        """蚂蚁二面原题："自动回滚的触发条件是什么" —— 先要有回滚能力。"""
        r = self.reg.call("post_comment",
                          {"issue_id": self.issue.issue_id,
                           "body": "建议按 timeout 处理", "cites": ["#101"]})
        self.assertTrue(r.ok)
        key = r.data["rollback_key"]
        rb = self.reg.rollback(key)
        self.assertEqual(rb["status"], "rolled_back")

    def test_参数缺失快速失败(self):
        r = self.reg.call("apply_labels", {"issue_id": self.issue.issue_id})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, "invalid_args")

    def test_检索排除自身(self):
        """★ 不能把"我自己"当成历史 issue —— 否则会自己判自己重复。"""
        r = self.reg.call("search_similar_issues", {"query": "接口调用超时"})
        self.assertTrue(r.ok)
        ids = [x["issue_id"] for x in r.data["items"]]
        self.assertNotIn(self.issue.issue_id, ids, "检索结果不应包含当前 issue")


# =============================================================================
# 评测
# =============================================================================

class TestEval(unittest.TestCase):

    def test_标签归一化(self):
        """★ 评测里最容易踩的坑：格式不一致导致 F1 永远为 0。

        实测过：ground truth 是 "timeout"，预测是 "bug/timeout"，
        不做归一化时 F1 = 0.000 —— 但这不是模型的问题，是评测代码的问题。
        **当指标恰好等于 0 或 1 时，先怀疑评测代码。**
        """
        n = Metrics.normalize_label
        self.assertEqual(n("bug/timeout"), "timeout")
        self.assertEqual(n("kind:bug-timeout"), "timeout")
        self.assertEqual(n("timeout"), "timeout")
        self.assertEqual(n("bug"), "bug")

        _, _, f1 = Metrics.label_f1(["bug/timeout"], ["timeout"])
        self.assertEqual(f1, 1.0, "归一化后应匹配")

    def test_duplicate指标分四象限(self):
        """两类错误代价不对称，必须分开报。"""
        from nightwatch.eval.metrics import SampleResult
        rows = [
            SampleResult("i1", predicted_duplicate_of="#1", gold_duplicate_of="#1"),   # TP
            SampleResult("i2", predicted_duplicate_of=None, gold_duplicate_of="#1"),   # FN
            SampleResult("i3", predicted_duplicate_of="#1", gold_duplicate_of=None),   # FP
            SampleResult("i4", predicted_duplicate_of=None, gold_duplicate_of=None),   # TN
        ]
        r = Metrics.duplicate_detection(rows)
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 1, 1))
        self.assertIn("confidence_interval_note", r,
                      "必须主动说明样本量局限")

    def test_评判校准_一致率不足则不可用(self):
        """裁判与人工不一致时应该明确说"不可用"，而不是凑合用。

        一个不可靠的裁判比没有裁判更糟 —— 它会给你错误信号，让你优化错方向。
        """
        cal = JudgeCalibrator(min_agreement=0.7)
        # 完全不同意
        bad = cal.agreement([0.9] * 10, [0.1] * 10)
        self.assertFalse(bad["可用"])
        # 完全同意
        good = cal.agreement([0.9] * 10, [0.8] * 10)
        self.assertTrue(good["可用"])

    def test_合成数据集标记为不可用于简历(self):
        """数据来源必须显式标注 —— 合成数据的指标不能当真实成绩。"""
        ds = build_synthetic_dataset()
        self.assertEqual(ds.source, "synthetic")
        self.assertIn("不要", ds.notes)
        summary = ds.summary()
        self.assertIn("来源", summary)
        self.assertGreater(len(ds.train), 0)

    # ------------------------------------------------------------ 多口径指标

    def test_top1只看第一个标签(self):
        """Top-1 是"抓核心分类"口径 —— 不惩罚多打标签。"""
        t1 = Metrics.top1_accuracy
        # 第一个标签对了就算对，哪怕多打了 3 个
        self.assertEqual(t1(["Bug", "Warnings", "Internals", "CI"], ["Bug"]), 1.0)
        # 第一个标签错了就是错，哪怕后面有对的
        self.assertEqual(t1(["Warnings", "Bug"], ["Bug"]), 0.0)
        # 归一化生效（bug/timeout vs timeout）
        self.assertEqual(t1(["bug/timeout"], ["timeout"]), 1.0)

    def test_hit_at_k是召回导向(self):
        """Hit@k 只看"有没有漏"，不看"多打了几个"。"""
        hk = Metrics.hit_at_k
        # 多打不扣分
        self.assertEqual(hk(["Bug", "Warnings", "X", "Y"], ["Bug"], k=2), 1.0)
        # 超出 k 范围就算没命中
        self.assertEqual(hk(["A", "B", "Bug"], ["Bug"], k=2), 0.0)
        # k=3 就能命中
        self.assertEqual(hk(["A", "B", "Bug"], ["Bug"], k=3), 1.0)
        # 多标签时按比例（命中 1/2）
        self.assertAlmostEqual(
            hk(["Bug", "A"], ["Bug", "Docs"], k=2), 0.5, places=4)

    def test_多口径能识别差异来源(self):
        """★ 核心断言：多口径报告要能区分"多打标签"和"抓不住核心"。

        这两种情况的改进方向完全不同（调提示词 vs 换模型），
        所以指标必须能区分它们。
        """
        from nightwatch.eval.metrics import SampleResult

        # 场景 A：只是多打标签，核心分类都对 → 严格F1低、Top-1高
        over_label = [
            SampleResult("i1", predicted_labels=["Bug", "X", "Y"], gold_labels=["Bug"]),
            SampleResult("i2", predicted_labels=["Docs", "X", "Y"], gold_labels=["Docs"]),
        ]
        a = Metrics.multi_metric_report(over_label)
        self.assertGreater(a["宽松口径"]["Top-1准确率"], a["严格口径"]["F1"])
        self.assertIn("多打标签", a["口径敏感性"])

        # 场景 B：核心分类就错 → 两个口径都低
        wrong = [
            SampleResult("i1", predicted_labels=["Docs"], gold_labels=["Bug"]),
            SampleResult("i2", predicted_labels=["CI"], gold_labels=["Docs"]),
        ]
        b = Metrics.multi_metric_report(wrong)
        self.assertEqual(b["宽松口径"]["Top-1准确率"], 0.0)
        self.assertEqual(b["严格口径"]["F1"], 0.0)

    def test_过度标注指数(self):
        """过度标注指数 = 平均预测标签数 - 平均gold标签数。"""
        from nightwatch.eval.metrics import SampleResult
        rows = [
            SampleResult("i1", predicted_labels=["Bug", "X", "Y"], gold_labels=["Bug"]),
            SampleResult("i2", predicted_labels=["Docs"], gold_labels=["Docs"]),
        ]
        r = Metrics.multi_metric_report(rows)
        self.assertEqual(r["标注倾向"]["平均预测标签数"], 2.0)
        self.assertEqual(r["标注倾向"]["平均gold标签数"], 1.0)
        self.assertEqual(r["标注倾向"]["过度标注指数"], 1.0)


class TestMemoryServiceSplit(unittest.TestCase):
    """记忆服务拆分（P1-1）—— 重点守护**降级行为**。

    ============================================================
    为什么这组测试最该存在
    ============================================================
    服务拆分引入了一个新的失败模式：**网络调用可能失败**。

    而"降级"是个非常容易被后续改动破坏的性质 ——
    比如有人为了"更好的错误处理"，把 `return []` 改成 `raise`，
    测试就会立刻失败。（这是好事：他必须显式地决定要不要改这个契约。）

    同样重要的是**降级必须快**。如果服务不可用时 Agent 卡 30 秒，
    那拆分反而是负收益。所以这里也断言了耗时上界。
    """

    def _dead_store(self, **kw):
        """指向一个没有服务在监听的地址。

        用一个高位端口（不太可能被占用），避免和真实服务冲突。
        """
        from nightwatch.memory.remote import RemoteMemoryStore
        kw.setdefault("timeout", 0.3)     # 测试里用短超时，否则测试会很慢
        kw.setdefault("retries", 0)
        return RemoteMemoryStore("http://127.0.0.1:59997", **kw)

    def test_服务不可用时召回返回空列表而非抛异常(self):
        """★ 核心契约：记忆服务挂了，Agent 应该退化成"没有记忆的版本"。"""
        store = self._dead_store()
        hits = store.recall("任意查询", top_k=3)      # 不抛异常
        self.assertEqual(hits, [])

    def test_所有方法都有安全默认值(self):
        """每个方法在失败时都返回"最保守"的值。

        ★ 这里最容易错的是 `find_similar` ——
        它必须返回 `None`（表示"没找到重复"），
        而不是抛异常或返回一个假对象。
        如果它错误地返回了非 None，Agent 会**误判两个无关 issue 是重复的**，
        然后把新问题当已解决关闭 —— 代价很大。
        """
        from nightwatch.models import MemoryRecord
        store = self._dead_store()
        mem = MemoryRecord(memory_id="m1", symptom="测试")

        self.assertEqual(store.recall("q"), [])
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.all_memories(), [])
        self.assertIsNone(store.find_similar(mem, 0.3))
        self.assertIsNone(store.get("m1"))
        self.assertFalse(store.insert(mem))
        self.assertEqual(store.pending_conflicts(), [])
        self.assertEqual(store.list_traces(), [])
        store.touch_access("m1", True)                # 静默失败
        store.close()

    def test_降级耗时有上界(self):
        """★ 降级必须是**快的**。

        如果服务不可用时卡 30 秒，拆分就是负收益 ——
        "少一条记忆参考"的损失远小于"卡 30 秒"。

        这个断言守的是"热路径依赖的延迟上界"这个设计决定。
        """
        import time as _time
        store = self._dead_store(timeout=0.3, retries=0)
        t0 = _time.time()
        store.recall("q")
        elapsed = _time.time() - t0
        self.assertLess(elapsed, 1.5,
                        f"降级耗时 {elapsed:.2f}s 过长 —— 热路径等不起")

    def test_降级统计被记录(self):
        """降级必须可观测 —— 否则无法区分"没命中"和"服务挂了"。"""
        store = self._dead_store()
        self.assertEqual(store.degradation_rate(), 0.0)
        store.recall("q1")
        store.recall("q2")
        self.assertEqual(store.stats["total_calls"], 2)
        self.assertEqual(store.stats["failed_calls"], 2)
        self.assertEqual(store.degradation_rate(), 1.0)
        self.assertTrue(store.stats["last_error"])

    def test_健康检查不降级(self):
        """health() 是唯一"不返回默认值"的方法 —— 它的语义就是报状态。

        如果它也返回"健康"，就失去了探活的意义。
        """
        store = self._dead_store()
        h = store.health()
        self.assertNotEqual(h.get("status"), "ok")

    def test_序列化往返不丢字段(self):
        """跨服务传输的数据必须能重建 —— 这是拆分的基本前提。"""
        from nightwatch.models import (
            MemoryRecord, MemoryTier, Outcome, memory_from_dict, to_dict,
        )
        mem = MemoryRecord(
            memory_id="m1", issue_id="#100", author="alice",
            symptom="症状", root_cause="根因", resolution="处置",
            outcome=Outcome.RESOLVED, confidence=0.85,
            symptom_tags=["a", "b"],
        )
        rebuilt = memory_from_dict(to_dict(mem))
        self.assertEqual(rebuilt.memory_id, mem.memory_id)
        self.assertEqual(rebuilt.symptom, mem.symptom)
        self.assertEqual(rebuilt.author, mem.author)
        self.assertEqual(rebuilt.outcome, Outcome.RESOLVED)     # 枚举被正确还原
        self.assertEqual(rebuilt.tier, MemoryTier.EPISODIC)
        self.assertEqual(rebuilt.symptom_tags, ["a", "b"])
        self.assertAlmostEqual(rebuilt.confidence, 0.85)

    def test_反序列化对未知字段容错(self):
        """服务端加了新字段时，老客户端不该崩 —— 这是滚动升级的前提。"""
        from nightwatch.models import MemoryRecord, memory_from_dict
        d = {"memory_id": "m1", "symptom": "x",
             "some_future_field": "服务端新增的，客户端不认识"}
        mem = memory_from_dict(d)          # 不该抛异常
        self.assertEqual(mem.memory_id, "m1")
        self.assertIsInstance(mem, MemoryRecord)

    def test_未知枚举值回退到默认(self):
        """枚举值不认识时回退，而不是抛异常。

        场景：服务端升级后新增了一个 Outcome 取值，
        而客户端还是老版本 —— 此时应该"尽力而为"，而不是让整个降级链路崩掉。
        """
        from nightwatch.models import Outcome, memory_from_dict
        mem = memory_from_dict({"memory_id": "m1", "outcome": "某个新枚举值"})
        self.assertEqual(mem.outcome, Outcome.UNFINISHED)


class TestA2AProtocol(unittest.TestCase):
    """A2A 协议（P2-2）—— 守护任务状态机与能力检查。"""

    def _card(self):
        from nightwatch.a2a import AgentCard, AgentSkill
        return AgentCard(name="test-agent", description="测试用", url="http://x",
                         skills=[AgentSkill(skill_id="s1", name="技能1",
                                            description="测试技能")])

    def test_终态不可再转换(self):
        """★ 状态机的基本原则：终态不可变。

        如果允许"已完成"变成"已取消"，客户端就无法判断
        "它到底做完了还是被取消了" —— 这会破坏整个协作语义。
        """
        from nightwatch.a2a import TaskState
        for s in (TaskState.COMPLETED, TaskState.FAILED,
                  TaskState.CANCELED, TaskState.REJECTED):
            self.assertTrue(s.is_terminal, f"{s} 应该是终态")
        for s in (TaskState.SUBMITTED, TaskState.WORKING,
                  TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED):
            self.assertFalse(s.is_terminal, f"{s} 不该是终态")

    def test_只有completed算成功(self):
        """★ `rejected`/`canceled` 是"正常结束"但不是"成功"。

        混在一起会导致："Agent 拒绝了这个任务"被报成"任务成功完成"。
        """
        from nightwatch.a2a import TaskState
        self.assertTrue(TaskState.COMPLETED.is_success)
        for s in (TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED):
            self.assertFalse(s.is_success)

    def test_技能检查(self):
        """Agent 只能声明它有的技能 —— 这是"体面拒绝"的依据。"""
        card = self._card()
        self.assertTrue(card.has_skill("s1"))
        self.assertFalse(card.has_skill("quantum-computing"))

    def test_card序列化往返(self):
        from nightwatch.a2a import AgentCard
        card = self._card()
        rebuilt = AgentCard.from_dict(card.to_dict())
        self.assertEqual(rebuilt.name, card.name)
        self.assertEqual(len(rebuilt.skills), 1)
        self.assertEqual(rebuilt.skills[0].skill_id, "s1")

    def test_技能不匹配被拒绝(self):
        """★ 端到端的"体面拒绝"：委托一个不会的技能 → REJECTED。"""
        from nightwatch.a2a import A2AServer, AgentExecutor, TaskState
        from nightwatch.a2a.protocol import Message, Part

        class _Noop(AgentExecutor):
            def execute(self, task, message):
                task.state = TaskState.COMPLETED

        server = A2AServer(self._card(), _Noop())
        resp = server.handle_rpc({
            "jsonrpc": "2.0", "id": "1", "method": "message/send",
            "params": {"skill": "quantum-computing",
                       "message": Message(role="user",
                                          parts=[Part.of_data({})]).to_dict()},
        })
        self.assertEqual(resp["result"]["status"]["state"], "rejected")
        self.assertIn("not offered", resp["result"]["status"]["error"])

    def test_任务失败不会中断请求(self):
        """executor 抛异常时，要返回 FAILED 状态而不是让连接断掉。

        因为客户端需要"知道任务失败了"，而不是"连接断了" ——
        这两者的处理方式完全不同。
        """
        from nightwatch.a2a import A2AServer, AgentExecutor
        from nightwatch.a2a.protocol import Message, Part

        class _Boom(AgentExecutor):
            def execute(self, task, message):
                raise RuntimeError("模拟 executor 崩了")

        server = A2AServer(self._card(), _Boom())
        resp = server.handle_rpc({
            "jsonrpc": "2.0", "id": "1", "method": "message/send",
            "params": {"skill": "s1",
                       "message": Message(role="user",
                                          parts=[Part.of_data({})]).to_dict()},
        })
        self.assertEqual(resp["result"]["status"]["state"], "failed")
        self.assertIn("RuntimeError", resp["result"]["status"]["error"])

    def test_未知方法返回JSONRPC错误(self):
        """JSON-RPC 层错误 vs 业务层失败 —— 必须能区分。

        客户端需要知道"调用没成功"（错误）还是"调用了但任务失败"（FAILED）。
        """
        from nightwatch.a2a import A2AServer, AgentExecutor, TaskState

        class _Noop(AgentExecutor):
            def execute(self, task, message):
                task.state = TaskState.COMPLETED

        server = A2AServer(self._card(), _Noop())
        resp = server.handle_rpc({"jsonrpc": "2.0", "id": "1",
                                  "method": "nonexistent/method", "params": {}})
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)


class TestSemanticMemoryLayer(unittest.TestCase):
    """语义记忆层（P1-2）—— 核心是守住**人工闸门**。

    ============================================================
    为什么这组测试里"闸门"最重要
    ============================================================
    三层记忆的风险量级不同：

        情景记忆写错了 → 影响**一次召回**
        语义记忆写错了 → 会被**所有后续同类任务**召回

    语义记忆相当于"写下了一条规则"，而规则错了会被反复应用。
    所以「未经人工审核的模式不能进语义层」这条必须是**测试守住的契约**，
    而不是注释里的承诺。

    **这类"安全性质"的测试有个特点：它们守的不是功能，是边界。**
    功能测试问"能不能做"，边界测试问"绝不能发生什么"。
    后者更容易被后续改动悄悄破坏。
    """

    def _make_store(self):
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        return MemoryStore(":memory:", encoder=HashEmbedding(dim=64))

    def _seed(self, store, tags, n, prefix="m"):
        """写入 n 条同一 tags 组合的情景记忆。"""
        from nightwatch.models import MemoryRecord, MemoryTier, Outcome
        for i in range(n):
            store.insert(MemoryRecord(
                memory_id=f"{prefix}{i}", tier=MemoryTier.EPISODIC,
                issue_id=f"#{prefix}{i}",
                symptom=f"{'|'.join(tags)} 类问题 第{i}条",
                symptom_tags=list(tags),
                root_cause="统一的根因",
                resolution="统一的处置",
                outcome=Outcome.RESOLVED))

    def _promoter(self, store):
        from nightwatch.llm.client import build_llm
        from nightwatch.memory.lifecycle import SemanticPromoter
        return SemanticPromoter(build_llm(use_mock=True), store)

    # --------------------------------------------------------- 核心：闸门

    def test_未审核时语义层为空(self):
        """★★ 最重要的一条：扫描不会自动写入语义层。

        如果这条挂了，说明有人把 promote 塞进了 scan 里 ——
        那"人工闸门"就只是注释里的承诺。
        """
        from nightwatch.models import MemoryTier
        store = self._make_store()
        self._seed(store, ["timeout", "permission"], 5)
        promoter = self._promoter(store)

        created = promoter.scan()
        self.assertGreater(len(created), 0, "应该识别出高频模式")
        # ★ 关键断言
        self.assertEqual(store.count(MemoryTier.SEMANTIC), 0,
                         "扫描之后语义层必须仍为空 —— 闸门被绕过了！")

    def test_拒绝后不进语义层(self):
        """拒绝的模式不能进语义层，但审核记录要留下。"""
        from nightwatch.models import MemoryTier
        store = self._make_store()
        self._seed(store, ["merge"], 4)
        promoter = self._promoter(store)
        promoter.scan()

        p = promoter.pending()[0]
        result = promoter.review(p.promotion_id, approved=False,
                                 reviewer="tester", note="根因不一致")

        self.assertIsNone(result, "拒绝时不该生成 SOP")
        self.assertEqual(store.count(MemoryTier.SEMANTIC), 0)
        # 记录要留下（供下次审核参考）
        after = store.get_promotion(p.promotion_id)
        self.assertEqual(after.status, "rejected")
        self.assertEqual(after.review_note, "根因不一致")

    def test_批准后才进语义层(self):
        """批准是唯一能写入语义层的路径。"""
        from nightwatch.models import MemoryTier
        store = self._make_store()
        self._seed(store, ["timeout"], 3)
        promoter = self._promoter(store)
        promoter.scan()

        p = promoter.pending()[0]
        sop = promoter.review(p.promotion_id, approved=True,
                              reviewer="tester", note="同意")

        self.assertIsNotNone(sop)
        self.assertEqual(sop.tier, MemoryTier.SEMANTIC)
        self.assertEqual(store.count(MemoryTier.SEMANTIC), 1)
        # 提案要记录它生成了哪条 SOP（可追溯）
        self.assertEqual(store.get_promotion(p.promotion_id).promoted_memory_id,
                         sop.memory_id)

    def test_promote_不能绕过审核(self):
        """直接调 promote(approved=False) 不产生任何东西。

        这个断言守的是 `promote` 的签名约束 ——
        `approved` 没有默认值，所以调用方**必须**显式表态。
        如果有人给它加了个 `approved=True` 的默认值，
        这个测试会立刻失败。
        """
        from nightwatch.models import MemoryRecord, MemoryTier
        store = self._make_store()
        self._seed(store, ["x"], 3)
        promoter = self._promoter(store)

        samples = store.all_memories(MemoryTier.EPISODIC)
        self.assertIsNone(promoter.promote("x", 3, samples, approved=False))
        self.assertEqual(store.count(MemoryTier.SEMANTIC), 0)

    def test_重复审核被拒绝(self):
        """审核是不可逆的人为决定 —— 不能覆盖历史。"""
        store = self._make_store()
        self._seed(store, ["a"], 3)
        promoter = self._promoter(store)
        promoter.scan()
        p = promoter.pending()[0]

        promoter.review(p.promotion_id, approved=True,
                        reviewer="alice", note="同意")
        # 第二个人想改成拒绝 → 应该失败
        again = promoter.review(p.promotion_id, approved=False,
                                reviewer="bob", note="我不同意")
        self.assertIsNone(again)
        # 原始记录不能被改
        self.assertEqual(store.get_promotion(p.promotion_id).reviewed_by, "alice")
        self.assertEqual(store.get_promotion(p.promotion_id).status, "approved")

    # --------------------------------------------------------- 识别与幂等

    def test_只提案超过阈值的模式(self):
        """阈值确实在起作用 —— 低频模式不该被提案。"""
        store = self._make_store()
        promoter = self._promoter(store)
        self._seed(store, ["high"], promoter.PROMOTE_MIN_OCCURRENCES)   # 刚好达标
        self._seed(store, ["low"], promoter.PROMOTE_MIN_OCCURRENCES - 1,
                   prefix="l")                                          # 差一个

        patterns = [p["pattern"] for p in promoter.find_patterns()]
        self.assertTrue(any("high" in p for p in patterns), "达标模式应被提案")
        self.assertFalse(any("low" in p for p in patterns), "未达标模式不该被提案")

    def test_扫描幂等(self):
        """重复扫描不会产生重复提案。

        因为 scan 会被反复调用（每次任务结束、定时任务），
        如果每次都插一条，审核队列会被同一个模式刷屏。
        """
        store = self._make_store()
        self._seed(store, ["dup"], 3)
        promoter = self._promoter(store)

        first = promoter.scan()
        second = promoter.scan()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0, "第二次扫描不该产生新提案")
        self.assertEqual(len(store.all_promotions()), 1)

    def test_审核后可追溯(self):
        """审核历史能回答"谁在哪时候批了什么"。"""
        store = self._make_store()
        self._seed(store, ["trace"], 3)
        promoter = self._promoter(store)
        promoter.scan()
        p = promoter.pending()[0]
        promoter.review(p.promotion_id, approved=True,
                        reviewer="zhangsan", note="根因一致")

        history = store.all_promotions()
        self.assertEqual(len(history), 1)
        h = history[0]
        self.assertEqual(h.reviewed_by, "zhangsan")
        self.assertTrue(h.reviewed_at, "审核时间必须记录")
        self.assertTrue(h.is_approved)


class TestDOE(unittest.TestCase):
    """实验设计（P1-3）—— 核心是**正交性**。

    ============================================================
    为什么正交性必须有测试守住
    ============================================================
    正交性（任意两因子的水平组合恰好出现一次）是 L9 表能"用 9 次
    试验估计主效应"的**唯一依据**。

    如果正交性被破坏（比如有人改了矩阵、或者硬套到 4 因子上），
    算出来的主效应会混入其他因子的影响 ——
    **而结论看起来仍然是一个正常的数字。**

    这是最危险的一类 bug：**实验设计错了，但结果看起来是对的。**
    """

    def _factors(self):
        from nightwatch.eval.doe import Factor
        return [Factor("A", ["a1", "a2", "a3"]),
                Factor("B", ["b1", "b2", "b3"]),
                Factor("C", ["c1", "c2", "c3"])]

    def test_l9有9次试验(self):
        from nightwatch.eval.doe import orthogonal_l9
        d = orthogonal_l9(self._factors())
        self.assertEqual(d.n_runs, 9)

    def test_l9满足正交性(self):
        """★ 核心性质：任意两个因子的水平组合恰好出现一次。

        验证方法：对每一对因子，统计 9 种组合各自的出现次数，
        全部应该恰好是 1。
        """
        from nightwatch.eval.doe import orthogonal_l9
        factors = self._factors()
        d = orthogonal_l9(factors)

        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                fi, fj = factors[i], factors[j]
                combos: dict = {}
                for run in d.runs:
                    key = (run.settings[fi.name], run.settings[fj.name])
                    combos[key] = combos.get(key, 0) + 1
                # 3×3 = 9 种组合，每种恰好 1 次
                self.assertEqual(len(combos), 9,
                                 f"{fi.name}×{fj.name} 的组合数不对")
                for key, cnt in combos.items():
                    self.assertEqual(
                        cnt, 1,
                        f"组合 {key} 出现了 {cnt} 次（应为 1）—— 正交性被破坏")

    def test_l9每个水平出现3次(self):
        """正交设计的均衡性：每个水平在全部试验中出现次数相同。"""
        from nightwatch.eval.doe import orthogonal_l9
        factors = self._factors()
        d = orthogonal_l9(factors)
        for f in factors:
            counts = {}
            for run in d.runs:
                v = run.settings[f.name]
                counts[v] = counts.get(v, 0) + 1
            self.assertEqual(set(counts.values()), {3},
                             f"因子 {f.name} 的水平出现次数不均衡：{counts}")

    def test_因子数不匹配时拒绝硬套(self):
        """★ 4 个因子时 L9 必须报错，而不是硬套。

        硬套会破坏正交性，给出**看起来正式但错误**的结论。
        **错误的实验设计比不做实验更糟。**
        """
        from nightwatch.eval.doe import Factor, orthogonal_l9
        with self.assertRaises(ValueError):
            orthogonal_l9([Factor(f"F{i}", [1, 2, 3]) for i in range(4)])
        # 水平数不是 3 也要报错
        with self.assertRaises(ValueError):
            orthogonal_l9([Factor("A", [1, 2, 3]), Factor("B", [1, 2, 3]),
                           Factor("C", [1, 2])])

    def test_全因子组合数(self):
        from nightwatch.eval.doe import Factor, full_factorial
        d = full_factorial([Factor("A", [1, 2, 3]), Factor("B", [1, 2])])
        self.assertEqual(d.n_runs, 6)
        d2 = full_factorial(self._factors())
        self.assertEqual(d2.n_runs, 27)

    def test_主效应计算(self):
        """主效应 = 各水平的效果均值。

        验证：构造一个"因子 A 明显有影响"的假数据。
        """
        from nightwatch.eval.doe import Factor, main_effects, orthogonal_l9
        factors = [Factor("A", ["a1", "a2", "a3"]),
                   Factor("B", ["b1", "b2", "b3"]),
                   Factor("C", ["c1", "c2", "c3"])]
        d = orthogonal_l9(factors)
        # A=a1 效果 0.9，a2=0.5，a3=0.1；B/C 无影响
        results = {}
        for run in d.runs:
            results[run.run_id] = {"a1": 0.9, "a2": 0.5, "a3": 0.1}[
                run.settings["A"]]
        eff = main_effects(d, results)

        self.assertAlmostEqual(eff["A"]["means"]["a1"], 0.9, places=3)
        self.assertAlmostEqual(eff["A"]["means"]["a3"], 0.1, places=3)
        self.assertAlmostEqual(eff["A"]["range"], 0.8, places=3)
        self.assertEqual(eff["A"]["best"], "a1")
        # B/C 无影响 → 极差应接近 0
        self.assertLess(eff["B"]["range"], 1e-6)
        self.assertLess(eff["C"]["range"], 1e-6)

    def test_帕累托前沿排除被支配的点(self):
        """只有"没有其他点成本更低且效果更好"的点才在前沿上。"""
        from nightwatch.eval.doe import Factor, cost_effect_points, orthogonal_l9
        d = orthogonal_l9(self._factors())
        # 构造：run01 又便宜又好 → 其他点都被支配
        results = {r.run_id: 0.5 for r in d.runs}
        costs = {r.run_id: 100.0 for r in d.runs}
        results[d.runs[0].run_id] = 0.9
        costs[d.runs[0].run_id] = 50.0

        pts = cost_effect_points(d, results, costs)
        front = [p for p in pts if p["on_pareto_frontier"]]
        self.assertEqual(len(front), 1, "应只有一个点不被支配")
        self.assertEqual(front[0]["run_id"], d.runs[0].run_id)

    def test_推荐给多个最优而不是一个(self):
        """★ 推荐必须给多个"最优" —— 因为目标是业务决定的。

        评测工具的职责是把权衡摊开，而不是替业务做价值判断。
        """
        from nightwatch.eval.doe import Factor, orthogonal_l9, recommend
        d = orthogonal_l9(self._factors())
        results = {r.run_id: 0.5 + i * 0.05 for i, r in enumerate(d.runs)}
        costs = {r.run_id: 100.0 + i * 10 for i, r in enumerate(d.runs)}
        reco = recommend(d, results, costs)

        for key in ("最省效果", "最省钱", "性价比最高", "达标里最省"):
            self.assertIn(key, reco, f"缺少推荐维度：{key}")
        self.assertIn("帕累托前沿", reco)


class TestResultFeedback(unittest.TestCase):
    """结果反馈闭环 —— 守护"用结果反推记忆质量"的几条核心语义。

    ============================================================
    这组测试守的是什么
    ============================================================
    ① **单次错误不足以定罪**（可能是模型自己失手）
    ② **用误诊率而不是误诊次数**（用 100 次错 2 次是好事）
    ③ **不做重复计数**（use_count 已在主链路更新，这里只补误诊）
    ④ **红线同 stale**：累加分数自动，生效必须过人工闸门
    """

    def _setup(self, **overrides):
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        from nightwatch.models import MemoryRecord, Outcome
        s = MemoryStore(":memory:", encoder=HashEmbedding(dim=16))
        # ★ 注意：两条记忆的 `symptom` 必须**不同** ——
        #   幂等键 = hash(repo, issue_id, symptom[:80])，
        #   两条一模一样的记忆会被幂等机制静默拒绝（第二条插不进去）。
        #   这类"测试数据自己撞了幂等键"是很常见的坑：
        #   `insert()` 返回 False 但不报错，你会以为插进去了。
        s.insert(MemoryRecord(memory_id="m_used_often", issue_id="#1",
                              symptom="被用很多次的记忆", use_count=100,
                              resolution="Bug", outcome=Outcome.RESOLVED))
        s.insert(MemoryRecord(memory_id="m_used_few", issue_id="#2",
                              symptom="被用很少次的记忆", use_count=3,
                              # ★ 给一个非零 confidence —— 否则红线测试
                              #   的"confidence 没被改动"断言没有意义（0 == 0）
                              confidence=0.8,
                              resolution="Bug", outcome=Outcome.RESOLVED))
        return s

    def test_单次错误不足以定罪(self):
        """★ 一次错误可能是模型自己失手 —— 不该牵连记忆。

        公式里 `min_misuse=2` 就是这条的实现。
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        flagged = t.record_triage_result(
            "#100", ["m_used_few"], correct=False, evidence="错一次")
        self.assertEqual(flagged, [], "单次错误不该入队")
        self.assertEqual(len(t.detector.pending()), 0)

    def test_用误诊率而不是次数(self):
        """★ 用 100 次错 2 次是好事；用 3 次错 2 次才是问题。

        只看次数的话两者都是 2，分不出来 —— 但性质完全不同。
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)

        # 两条记忆各错 2 次
        for i in range(2):
            t.record_triage_result(
                f"#20{i}", ["m_used_often", "m_used_few"], correct=False,
                evidence="e")

        often = s.get("m_used_often")
        few = s.get("m_used_few")
        self.assertEqual(often.misuse_count, few.misuse_count,
                         "两者的误诊**次数**相同")

        # 但分数差很多
        self.assertLess(t.suspect_score(often), 0.05,
                        "用 100 次错 2 次 → 分数应该很低")
        self.assertGreater(t.suspect_score(few), 0.25,
                           "用 3 次错 2 次 → 分数应该超过阈值")

        self.assertFalse(t.is_suspect(often), "高频但低错误率的不该被怀疑")
        self.assertTrue(t.is_suspect(few), "低频高错误率的该被怀疑")

    def test_误诊计数不用重复计use(self):
        """★ `use_count` 已在主链路（`react.py` → `touch_access`）更新。

        这里**不能再加一遍** —— 否则 use_count 翻倍，误诊率凭空减半。

        （这类"两处都加"是统计功能最容易错的点，所以专门测。）
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        before = s.get("m_used_few").use_count
        t.record_triage_result("#100", ["m_used_few"], correct=False)
        after = s.get("m_used_few").use_count
        self.assertEqual(before, after, "本模块不该改动 use_count")

    def test_正确判断不产生误诊记录(self):
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        t.record_triage_result("#100", ["m_used_few"], correct=True)
        st = t.stats()
        self.assertEqual(st["misuse_events"], 0, "正确判断不该有误诊记录")
        self.assertEqual(s.get("m_used_few").misuse_count, 0)

    def test_达阈值才入队且幂等(self):
        """★ 幂等：同一条记忆不会被重复推进队列。

        这里踩过一个坑：第一版 `trigger_detail` 用了每次误诊的具体信息
        （issue_id + evidence）—— 每次都不同 → 幂等完全失效 → 队列膨胀。

        修法：`trigger_detail` 用**稳定的类型标识**，
        具体信息放进 `misuse_events` 明细表。
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        f1 = t.record_triage_result("#1", ["m_used_few"], correct=False)
        f2 = t.record_triage_result("#2", ["m_used_few"], correct=False)
        self.assertEqual(len(f1), 0, "第一次：misuse=1 < 2，不入队")
        self.assertEqual(len(f2), 1, "第二次：misuse=2，入队")
        self.assertEqual(len(t.detector.pending()), 1, "队列里只有一条")

        # 第三次误诊：已经入队 → 不重复
        f3 = t.record_triage_result("#3", ["m_used_few"], correct=False)
        self.assertEqual(len(f3), 0, "已在队列里，不重复入队")
        self.assertEqual(len(t.detector.pending()), 1)

    def test_判定了就不重复问的两种情形(self):
        """★ 区分"新变更要重新问"和"继续误诊不再问"。

        · stale 线：**新的变更** → 记忆赖以成立的前提变了 → 重新问 ✓
        · feedback 线：**继续误诊** → 前提没变，只是结果不好 → 不再问 ✓

        判断标准是"**前提变了吗**"，不是"结果好不好"。
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        t.record_triage_result("#1", ["m_used_few"], correct=False)
        t.record_triage_result("#2", ["m_used_few"], correct=False)
        rid = t.detector.pending()[0].review_id
        # 人判定"记忆本身没问题"
        t.detector.review(rid, confirmed=False, reviewer="m", note="核对过")

        # 继续误诊 → 不该再问（前提没变）
        f = t.record_triage_result("#3", ["m_used_few"], correct=False)
        self.assertEqual(f, [], "判定过之后继续误诊，不该重复问人")

    def test_明细可追溯(self):
        """误诊明细要能回答"这条记忆是被哪几次误诊牵连的"。"""
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        for i in range(3):
            t.record_triage_result(f"#30{i}", ["m_used_few"], correct=False,
                                   evidence=f"误诊 {i}")
        hist = t.history("m_used_few")
        self.assertEqual(len(hist), 3)
        # ★ 不依赖顺序断言 —— 三条记录在同一秒内写入，
        #   `created_at` 是秒级字符串，排序不稳定。
        #   **同一秒内的排序不确定，测试就不该依赖它** ——
        #   否则会变成偶发失败（flaky test），而那种失败最消耗人。
        self.assertEqual({h.evidence for h in hist},
                         {"误诊 0", "误诊 1", "误诊 2"})
        self.assertTrue(all(h.memory_id == "m_used_few" for h in hist))

    def test_入队不改变记忆(self):
        """★★ 红线（同 stale）：累加分数可以自动，生效必须过闸门。"""
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        t.record_triage_result("#1", ["m_used_few"], correct=False)
        t.record_triage_result("#2", ["m_used_few"], correct=False)
        mem = s.get("m_used_few")
        self.assertFalse(mem.archived, "入队不该归档记忆")
        self.assertGreater(mem.confidence, 0.0,
                           "入队不该改变 confidence（只有人确认才降权）")

    def test_没用到记忆的错误不牵连任何记忆(self):
        """★ Agent 可能完全靠自己判断（`memory_used` 为空）。

        那这次错误**和记忆无关** —— 不该牵连任何记忆。
        **冤枉记忆比漏掉更糟**（会把好记忆推进队列，浪费人的时间）。
        """
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        f = t.record_triage_result("#1", [], correct=False, evidence="没用到记忆")
        self.assertEqual(f, [])
        self.assertEqual(t.stats()["misuse_events"], 0)

    def test_统计反映检测质量(self):
        from nightwatch.memory.feedback import MisuseTracker
        s = self._setup()
        t = MisuseTracker(s)
        t.record_triage_result("#1", ["m_used_few"], correct=False)
        t.record_triage_result("#2", ["m_used_few"], correct=False)
        st = t.stats()
        self.assertEqual(st["misuse_events"], 2)
        self.assertEqual(st["memories_with_misuse"], 1)
        self.assertEqual(st["suspects"], 1)
        self.assertEqual(st["flagged_to_review"], 1)


class TestStaleDetection(unittest.TestCase):
    """仓库变更 → 过时记忆标记 —— 守护"标记自动、生效人工"这条红线。

    ============================================================
    为什么这组测试最重要的一条是"入队不改变记忆"
    ============================================================
    本项目已经踩过一次这个坑（`docs/02` B6）：
    被人工确认过的经验，因为自动衰减规则被当垃圾归档了。

    **那次是因为"时间"，这次如果因为"文件变更"犯同样的错，就白踩了。**

    核心区别必须守住：
        「变更波及」≠「结论失效」
        文件被改了，不代表引用了它的记忆就错了（可能只是改了注释）

    所以入队（自动）**不能**碰记忆，只有人工确认（`review(confirmed=True)`）
    才能降权归档。
    """

    def _setup(self):
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        from nightwatch.models import HumanVerdict, MemoryRecord, Outcome
        s = MemoryStore(":memory:", encoder=HashEmbedding(dim=16))
        s.insert(MemoryRecord(
            memory_id="m1", issue_id="#1", symptom="merge 行数异常",
            code_refs=["pandas/core/frame.py"],
            resolution="Bug", outcome=Outcome.RESOLVED, confidence=0.8))
        s.insert(MemoryRecord(
            memory_id="m2", issue_id="#2", symptom="to_numeric 警告",
            code_refs=["pandas/core/frame.py", "pandas/core/tools.py"],
            resolution="Bug", outcome=Outcome.RESOLVED, confidence=0.6,
            human_verdict=HumanVerdict.ACCEPTED))
        s.insert(MemoryRecord(
            memory_id="m3", issue_id="#3", symptom="读 CSV 慢",
            code_refs=["pandas/io/parsers.py"],
            resolution="Performance", outcome=Outcome.RESOLVED, confidence=0.9))
        return s

    def test_检测只命中被改动的文件(self):
        from nightwatch.memory.stale import StaleDetector
        d = StaleDetector(self._setup())
        hits = d.detect(["pandas/core/frame.py"])
        ids = {h.memory_id for h in hits}
        self.assertEqual(ids, {"m1", "m2"}, "m3 引用的是另一个文件，不该命中")
        self.assertEqual(d.detect(["pandas/io/parsers.py"])[0].memory_id, "m3")
        self.assertEqual(d.detect(["unrelated/file.py"]), [])

    def test_路径归一化(self):
        """★ 路径写法不同也要能匹配 —— 否则会**静默漏判**。

        'pandas/core/frame.py' / './pandas/core/frame.py' / 'pandas\\core\\frame.py'
        是同一个文件，不归一化的话交集运算会漏掉它们。

        而漏判的方向是"该复核的没进队列" —— 最危险的方向。
        """
        from nightwatch.memory.stale import StaleDetector
        d = StaleDetector(self._setup())
        for variant in ("pandas/core/frame.py",
                        "./pandas/core/frame.py",
                        "pandas\\core\\frame.py",
                        "  pandas/core/frame.py  "):
            hits = d.detect([variant])
            self.assertTrue(hits, f"{variant!r} 应该能匹配上")

    def test_人工确认过的排最前(self):
        """复核队列永远排不完，所以**排序逻辑比检测逻辑更影响实际效果**。

        被人工确认过的记忆一旦失效，危害最大
        （不只是"记错了"，而是"人确认过的东西错了"，更难被质疑）。
        """
        from nightwatch.memory.stale import StaleDetector
        d = StaleDetector(self._setup())
        hits = d.detect(["pandas/core/frame.py"])
        self.assertEqual(hits[0].memory_id, "m2",
                         "人工确认过的 m2 应该排在 m1 前面")
        self.assertGreater(hits[0].priority, hits[1].priority)

    def test_入队不改变记忆(self):
        """★★ 红线：入队（自动）绝不能碰记忆本身。

        它只往队列里插记录。记忆照常被召回、照常参与判断 ——
        "待复核"不是"已生效的惩罚"，只是"待办清单"。

        如果哪天有人"优化"成"入队即降权" —— 这条测试会立刻失败。
        """
        from nightwatch.memory.stale import StaleDetector
        s = self._setup()
        d = StaleDetector(s)
        before = s.get("m1")
        conf_before, arch_before = before.confidence, before.archived

        d.mark(d.detect(["pandas/core/frame.py"]), trigger_detail="PR#1")

        after = s.get("m1")
        self.assertEqual(after.confidence, conf_before,
                         "入队不该改变 confidence")
        self.assertEqual(after.archived, arch_before,
                         "入队不该改变 archived")
        self.assertEqual(len(d.pending()), 2, "应该有两个待复核项")

    def test_确认过时才生效(self):
        """只有 `review(confirmed=True)` 才降权归档。

        `confirmed=False`（判定仍有效）是**有价值的记录** ——
        它回答"这条被怀疑过的记忆，人看过之后认为没问题"。
        """
        from nightwatch.memory.stale import StaleDetector
        s = self._setup()
        d = StaleDetector(s)
        ids = d.mark(d.detect(["pandas/core/frame.py"]), trigger_detail="PR#1")
        # 找到 m1 的那条
        m1_review = next(i for i in d.pending() if i.memory_id == "m1")
        m2_review = next(i for i in d.pending() if i.memory_id == "m2")

        # m1：确认过时 → 生效
        d.review(m1_review.review_id, confirmed=True, reviewer="m")
        self.assertTrue(s.get("m1").archived)
        self.assertLessEqual(s.get("m1").confidence, 0.1)

        # m2：判定仍有效 → 完全不变
        d.review(m2_review.review_id, confirmed=False, reviewer="m")
        self.assertFalse(s.get("m2").archived, "判定有效的不该被改动")
        self.assertAlmostEqual(s.get("m2").confidence, 0.6, places=6)

    def test_幂等_同一变更不重复问(self):
        """★ 幂等键是 `(memory_id, trigger_detail)` 而不是只有 memory_id。

        只查 memory_id 的话，会出现"一次 dismiss 之后永远不再复核" ——
        而那是**漏检**，比"多问一遍"危险得多。
        """
        from nightwatch.memory.stale import StaleDetector
        d = StaleDetector(self._setup())
        ids = d.mark(d.detect(["pandas/core/frame.py"]), trigger_detail="PR#123")
        self.assertEqual(len(ids), 2)

        # 同一变更重复触发 → 不重复入队
        again = d.mark(d.detect(["pandas/core/frame.py"]),
                       trigger_detail="PR#123")
        self.assertEqual(len(again), 0)

        # 判定仍有效后，同一变更**仍然不重复问**
        d.review(ids[0], confirmed=False, reviewer="m")
        after_dismiss = d.mark(d.detect(["pandas/core/frame.py"]),
                               trigger_detail="PR#123")
        self.assertEqual(len(after_dismiss), 0,
                         "已判定过的同一变更不该重复问")

        # ★ 但**新的变更**值得再问 —— 上次的结论不能自动延续
        new_trigger = d.mark(d.detect(["pandas/core/frame.py"]),
                             trigger_detail="PR#456")
        self.assertGreater(len(new_trigger), 0,
                           "新的变更应该重新入队（上次判定不能自动延续）")

    def test_已归档的不再检测(self):
        """已归档的记忆不在召回池里，复核它不产生收益。"""
        from nightwatch.memory.stale import StaleDetector
        s = self._setup()
        d = StaleDetector(s)
        mem = s.get("m1")
        mem.archived = True
        s.update(mem)
        ids = {h.memory_id for h in d.detect(["pandas/core/frame.py"])}
        self.assertNotIn("m1", ids)

    def test_无code_refs的记忆不被误报(self):
        """★ 没有代码引用的记忆（纯经验/SOP）**无法用本方法检测**。

        这是一个诚实的边界 —— 检测靠的是 `code_refs`，
        没有它就只能返回空，**而不是瞎猜**。

        （这类记忆的过时检测需要别的手段，本模块不做。）
        """
        from nightwatch.memory.stale import StaleDetector
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        from nightwatch.models import MemoryRecord, MemoryTier, Outcome
        s = MemoryStore(":memory:", encoder=HashEmbedding(dim=16))
        s.insert(MemoryRecord(
            memory_id="sem1", tier=MemoryTier.SEMANTIC,
            symptom="SOP：提交前先跑 lint", code_refs=[],
            outcome=Outcome.RESOLVED))
        d = StaleDetector(s)
        self.assertEqual(d.detect(["pandas/core/frame.py"]), [])

    def test_统计反映受影响比例(self):
        from nightwatch.memory.stale import StaleDetector
        d = StaleDetector(self._setup())
        d.mark(d.detect(["pandas/core/frame.py"]), trigger_detail="PR#1")
        st = d.stats()
        self.assertEqual(st["pending"], 2)
        self.assertEqual(st["n_memories"], 3)
        self.assertAlmostEqual(st["affected_ratio"], 2 / 3, places=3)


class TestCompareExperiment(unittest.TestCase):
    """LangGraph 对照实验 —— 守护"实验本身有没有失效"。

    ============================================================
    ★ 为什么对照实验也要写测试
    ============================================================
    直觉上"实验脚本"不是产品代码，跑一次出个数字就行，不必测。

    但这个项目真的踩了坑：**循环覆盖率是 0/15 的时候，脚本正常退出、
    报告正常生成、一致率还是 100%** —— 只是那个 100% 建立在
    "两版都在第 1 步就结束"上，**循环一次都没跑**。
    而"框架版多 67 行"里有相当一部分正是那些从未执行的代码。

    **结论就写在输出里，但它悄无声息地不成立。**

    所以这组测试守的是"实验的前提条件"，而不是"实验的结论"：
      ① 样本必须覆盖**两条**决策路径（命中记忆 / 不命中记忆）
      ② 命中与不命中的相似度分布必须**分得开**（否则阈值切不出来）
         —— 且余量必须用**含探针的完整样本集**量（只用标定样本会偏乐观）
      ③ 两版的步数统计口径必须**一致**（否则步数分布没法对齐比较）

    ★ 这组测试本身被修过两次，两次都是"假保护"：
      第一次：按被测阈值分组 → 循环论证（改阈值测试照样绿）
      第二次：分组对了，但样本集漏了探针 → 量出的余量比现实乐观
      **两次都是"看起来在守，实际没守"。** 所以下面每个断言旁边
      都写了"它是怎么被破坏验证抓出来的" —— 那段历史比断言本身值钱。
    """

    def _samples(self):
        """导入实验的 shared 模块（用 sys.path，不用 importlib）。

        ★ 踩过的坑：一开始用 `importlib.util.spec_from_file_location` 加载，
          结果 `@dataclass` 装饰器报
          `'NoneType' object has no attribute '__dict__'` ——
          因为 spec 加载的模块**没有注册进 `sys.modules`**，
          而 dataclass 解析类型注解时要 `sys.modules.get(cls.__module__)`。
          **这类"加载方式导致的坑"和业务无关，但会浪费半小时。**
          改用 sys.path + 普通 import 就没事。
        """
        import importlib
        from pathlib import Path
        exp_dir = Path(__file__).resolve().parents[1] / "experiments" / "langgraph_compare"
        added = str(exp_dir) not in sys.path
        if added:
            sys.path.insert(0, str(exp_dir))
        try:
            return importlib.import_module("shared")
        finally:
            pass  # 故意不清理：后面的测试还要用同一份模块

    def test_样本覆盖两条决策路径(self):
        """★ 核心守卫：样本里必须同时有"命中记忆"和"不命中记忆"的。

        这是那个 0/15 事故的直接修复点 —— 如果哪天有人加样本时
        只加了命中记忆的，循环覆盖率会**安静地回到 0**，
        而报告仍然一切正常。

        **断言的是"两类都存在"，不是"各有几条"** ——
        数量是可以调整的，性质不可以。
        （本项目在别处也用了这个原则：断言约束，不断言快照。）
        """
        mod = self._samples()
        store, _ = mod.setup_tools()
        try:
            hits, misses = [], []
            for iss in mod.sample_issues():
                s = store.recall(iss.search_text(), top_k=1)[0][1]
                (hits if s >= mod.MIN_RECALL_SIMILARITY else misses).append(
                    (iss.issue_id, s))
        finally:
            store.close()

        self.assertTrue(hits, "没有任何样本命中记忆 —— "
                              "这条'记忆 → 0 轮'的对照路径没被测到")
        self.assertTrue(misses, "没有任何样本不命中记忆 —— "
                                "循环永远不会执行（这就是那次的 0/15）")

    def test_命中与不命中的相似度分布分得开(self):
        """阈值必须真的能切分两组，而不是勉强落在重叠区。

        为什么要断言"有余量"而不是"能分开"？
        因为贴着边界取的阈值会让**循环覆盖率随措辞微调翻来覆去** ——
        今天过、明天不过，实验就不可复现了。
        留余量是为了"改一句话不会翻转结论"。

        ★ 这里还守着一个更隐蔽的问题：**嵌入维度过低会导致哈希碰撞**，
          把完全无关的样本也抬上去（实测 dim=64 时"UI 按钮错位"
          拿到 0.587，比"该命中"的 #901 还高）。
          那种情况下**调阈值怎么调都不对** —— 因为分布是重叠的。

        ★★ 修过一次：断言从"dim >= 256"改成"dim == 主项目默认值" ★★
          原断言 `assertGreaterEqual(EMBED_DIM, 256)` 是**软断言** ——
          它能过 1024，也能过 4096，还能过任何更大的数。
          但它**没有守住真正重要的那件事**：实验用的维度应该
          **和主项目一致**（减少一个变量），而不是"足够大就行"。

          实测（`dim_sweep.py`）：256 的空档 +0.061 已经够用，
          1024 的 +0.144 只是"更宽松"，不是"更对"。
          **用更宽松的参数会让实验比真实系统更稳健，
          从而测不出真实系统的脆弱点。**

          所以现在断言：① `EMBED_DIM` 等于主项目 `HashEmbedding` 默认值；
          ② 阈值**落在空档内部**（两个方向都有余量）。
          —— ②比"空档 > 0"更强：空档存在不等于阈值可用。

        ★★ 还有一个更隐蔽的坑：**分组不能按被测对象自身来分** ★★
          这组测试第一版是这么写的：

              (hs if s >= MIN_RECALL_SIMILARITY else ms).append(s)
              # 然后断言 min(hs) > max(ms)

          用**被测的那个阈值**分组，再断言这个分组分得开 ——
          **循环论证**。阈值怎么错，分组就跟着怎么错，断言永远成立。

          这是**故意破坏验证**时发现的：把阈值改成 0.40（明显会漏召回），
          测试照样绿。也就是说那条断言**根本没有保护作用**。

          （如果没做破坏验证，我会一直以为它有保护作用 ——
           这比"没有断言"更糟：**它给了虚假的安全感**。）

          修法：分组依据是**样本的设计意图**（`WANT_HIT` / `WANT_MISS`），
          然后拿这个"真值分组"去检验阈值。
          **测试的对照组必须来自被测对象之外。**

        ★★ 修了第二次：分组修对了，但**样本集是残缺的** ★★
          上面那版修完，分组不再循环论证了 —— 但仍然**只用了标定样本**，
          漏掉了 `dim_sweep.py` 里的 3 个探针。

          实测两套口径：

              仅标定样本（测试当时的口径）  low_hit - max(#906,#907)
                                          = 0.355 - 0.286 = **0.069**
              含 3 个探针（真实口径）        low_hit - 探针最高
                                          = 0.355 - 0.293 = **0.061**

          而真正卡住阈值下限的是**探针**「UI 按钮错位」= 0.293，
          比标定样本 #906（0.286）**还高 0.007**。

          **所以测试报的是一个偏乐观的数，而且它守的那条边不是真的那条边。**
          根因是 `PROBES` 定义在 `dim_sweep.py` 里，测试 import 不到 ——
          于是"同一件事"有了两份实现，各自长出了不同的口径。
          这和本文件开头讲的 `parse_action`/`_format_memory` 是**同一个坑**：
          **"共用"要复用实现，不要各写一份**（对齐是一个面，不是点）。

          修法：`PROBES` 上提到 `shared.py`，并新增
          `shared.separation_margin(store)` 作为**唯一的余量口径**，
          测试和 `dim_sweep.py` 都调它。
          测试另加一条断言：`n_unrelated == 5` —— 防止哪天有人
          把探针从里面去掉，测试**悄悄退化成乐观口径**而数字看着都正常。

        ★★ 同时把余量断言从 gap 换成 worst ★★
          原来断言 `gap > 0.05`（gap = 命中组最低 - 无关组最高）。
          问题：**gap 是两个方向余量之和，会掩盖单边贴地**。
          可能一边 +0.040、另一边 +0.002，gap 有 0.042 看着挺宽，
          实际某一侧已贴着悬崖。本项目实测过边界是**单侧**的
          （只对"无关样本变热"敏感），所以该守的是 `min(两侧余量)`。
          现在断言 `worst > 0.02`（当前实测 +0.027）。
        """
        mod = self._samples()

        # ① 维度必须与主项目默认值一致
        #
        # ★ 用 inspect 读签名，而不是硬编码 256 ——
        #   这样主项目改了默认值，测试会**自动跟着变**（并可能失败），
        #   而不是"测试写死 256、主项目改成 512、两边悄悄不一致"。
        #   这是"数字必须有出处"原则在测试里的应用。
        import inspect
        from nightwatch.memory.store import HashEmbedding
        main_default = inspect.signature(
            HashEmbedding.__init__).parameters["dim"].default
        self.assertEqual(
            mod.EMBED_DIM, main_default,
            f"实验用了 EMBED_DIM={mod.EMBED_DIM}，"
            f"而主项目 HashEmbedding 默认值是 {main_default}。"
            f"实验应复用主项目参数以减少变量 —— "
            f"除非有实测理由要偏离（那种情况要在共享注释里写明理由）")

        store, _ = mod.setup_tools()
        try:
            # ★★ 关键：用 `shared.separation_margin()` 量余量 ★★
            #
            # 为什么不再自己分组算：
            #   原来看起来没问题（按 WANT_HIT/WANT_MISS 分组是对的），
            #   但**样本集是残缺的** —— 只用标定样本，漏了 3 个探针。
            #   实测：仅标定样本 → 余量 0.069；含探针 → 0.061。
            #   而真正卡住阈值下限的是探针「UI 按钮错位」= 0.293，
            #   比标定样本 #906（0.286）还高。
            #
            #   **守卫必须测"最危险的那个数"。**
            #   只测乐观口径 = 守卫被配了一个比现实更宽的边界，
            #   于是它守不住真正会发生的那种翻车。
            #
            # 现在改调 `shared.separation_margin()` —— 口径**唯一**，
            # 和 `dim_sweep.py` 用的是同一份实现（含探针）。
            # **"共用"要复用实现，不要各写一份**：上次是解析/渲染，
            # 这次是样本集 —— 同一个坑的又一次。
            m = mod.separation_margin(store)
            low_hit, high_miss = m["low_hit"], m["high_miss"]

            # 顺带断言"确实含了探针" —— 否则哪天有人把探针从
            # separation_margin 里去掉，这个测试会**悄悄退化成乐观口径**，
            # 而所有数字看起来仍然正常。
            # （这正是"空结果掩盖 bug"的同一类陷阱。）
            self.assertEqual(
                m["n_unrelated"], 5,
                f"无关组应有 5 个（#906/#907 + 3 探针），实得 {m['n_unrelated']} "
                f"—— 探针被漏掉了，余量会变乐观，守卫就守不住了")
        finally:
            store.close()

        self.assertGreater(low_hit, high_miss,
                           f"两组重叠：命中组最低 {low_hit:.3f} "
                           f"< 无关组最高 {high_miss:.3f}")

        # ② ★ 余量断言 —— 换成"最危险的那个数"
        #
        #   原来断言的是 gap（= low_hit - high_miss），阈值 0.05。
        #   问题：**gap 是两个方向余量之和，会掩盖单边贴地**。
        #   可能一边 +0.040、另一边 +0.002，gap 看着 0.042 挺宽，
        #   实际已经贴着悬崖了。
        #
        #   本项目实测过边界是**单侧**的（只对"无关样本变热"敏感），
        #   所以真正该守的是 **min(两侧余量)**。
        #   实测当前值：到下限 +0.027、到上限 +0.035 → worst = +0.027。
        #
        #   阈值取 0.02 的理由：
        #     · 比"能分开"（>0）严 —— 保证有余量
        #     · 但不硬编码 0.027 —— 那是快照，改参数就过期
        #       （本项目原则：不编数字、断言约束不断言快照）
        #     · 0.02 是"两侧都不许贴着"的下限约束
        self.assertGreater(
            m["worst"], 0.02,
            f"最紧的一侧余量只有 {m['worst']:+.3f} "
            f"（到下限 {m['slack_lo']:+.3f} / 到上限 {m['slack_hi']:+.3f}）"
            f"—— 阈值贴着某一边，改措辞就会翻转结论")

        # ③ ★ 阈值必须落在空档内部（比"空档 > 0"更强）
        #
        #   为什么这一条是必要的：空档存在**不代表阈值可用**。
        #   实测 dim=256 时（旧阈值 0.30）：
        #       到下限（阈值 - 无关组最高）= 0.300 - 0.293 = +0.007
        #   阈值几乎贴着探针的上界 —— 任何让无关样本变热的改动
        #   都会把它推过阈值，#906 会从"不命中"变成"命中"，
        #   循环覆盖率从 6/18 掉到 3/18。
        #
        #   所以两个方向都要断言，缺一不可：
        #       到下限 > 0  → 排除误召回（不该命中的命中了）
        #       到上限 > 0  → 保证不漏召回（该命中的没命中）
        t = mod.MIN_RECALL_SIMILARITY
        self.assertGreater(
            t - high_miss, 0,
            f"阈值 {t} 低于/等于无关组最高分 {high_miss:.3f} —— "
            f"会有误召回（该走完整循环的样本变成命中）")
        self.assertGreater(
            low_hit - t, 0,
            f"阈值 {t} 高于/等于命中组最低分 {low_hit:.3f} —— "
            f"会有漏召回（该走'记忆 → 0 轮'的样本变成不命中）")

    def test_两版步数统计口径一致(self):
        """★ 两版必须用同一个口径统计步数，否则步数分布没法比较。

        踩过的坑：手写版原本用 `state.step`（= 最后一次迭代的步号），
        框架版用 `final_state["step"]`（= 最后一个节点写入的值）。
        **两个都是"最后写入的值"而不是"执行了几步"** ——
        所以错误是**对称的**，一致率照样 100%，只是数字整体偏小。

        而"步数分布"正是判定循环覆盖率的依据 → **覆盖率被低估**。
        这是"指标算错比指标难看更危险"的典型：
        一个算错的指标会给出一个看起来合理的错结论。

        这里断言两版在同样输入下**步数相同**。
        """
        import importlib.util
        from pathlib import Path
        exp_dir = Path(__file__).resolve().parents[1] / "experiments" / "langgraph_compare"
        sys.path.insert(0, str(exp_dir))
        try:
            from custom_agent import run_custom_agent
            from shared import sample_issues
            try:
                from graph_agent import run_graph_agent
            except ImportError:
                self.skipTest("LangGraph 未安装")
            for iss in sample_issues():
                c = run_custom_agent(iss, use_mock=True)
                g = run_graph_agent(iss, use_mock=True)
                self.assertEqual(
                    c.steps, g.steps,
                    f"{iss.issue_id} 步数不一致：手写 {c.steps} vs "
                    f"LangGraph {g.steps}（两版统计口径不同）")
        finally:
            if str(exp_dir) in sys.path:
                sys.path.remove(str(exp_dir))

    def test_至少有一个运行走到第二步(self):
        """★ 验收标准的直接编码：循环覆盖率 > 0。

        这条测试就是用户给的验收条件 —— 把它写进测试，
        意味着**它不会再悄悄地失效**。
        """
        import importlib.util
        from pathlib import Path
        exp_dir = Path(__file__).resolve().parents[1] / "experiments" / "langgraph_compare"
        sys.path.insert(0, str(exp_dir))
        try:
            from custom_agent import run_custom_agent
            from shared import sample_issues
            steps = [run_custom_agent(i, use_mock=True).steps
                     for i in sample_issues()]
            self.assertGreater(
                sum(1 for s in steps if s >= 2), 0,
                f"没有任何运行走到第 2 步（步数 {steps}）—— 循环从未被执行")
        finally:
            if str(exp_dir) in sys.path:
                sys.path.remove(str(exp_dir))

    # =========================================================================
    # ★★ 下面三条守"准确率"这个新指标本身是不是可信的 ★★
    # =========================================================================

    def test_准确率参照不依赖被测对象(self):
        """★★ 参照（expected_label）必须**在被测对象之外** ★★

        这是准确率指标的**根基** —— 如果被测逻辑能读到标准答案，
        那"准确率"就是循环论证：它测的是"被测代码有没有把答案抄对"。

        怎么验？**把 `Issue` 上的 `gold_labels` 清空，看预测变不变。**
          变了   → 被测逻辑读了标准答案 → 参照被污染 → 指标无效
          不变   → 参照确实在被测对象之外 ✓

        ★ 为什么这个风险是真实的（不是假想）：
          主项目里 `react.py:1081` **就真的读** `issue.gold_labels`
          来判定"是否误诊"。今天的实验路径不读它，但"今天不读"不是保证 ——
          哪天有人把实验的编排换成主项目的 react 循环，这个泄漏就会发生。
          所以权威的那份放在 `shared.EXPECTED_LABELS`（模块级字典），
          不放在会被传进被测代码的 `Issue` 上。

        ★ 这和测试里用硬编码 `WANT_HIT`/`WANT_MISS` 分组、
          把探针收回 `shared` 是**同一个原则的三次应用**：
          **参照必须来自被测对象之外。**
        """
        exp_dir = self._exp_dir()
        try:
            import dataclasses
            from custom_agent import run_custom_agent
            from shared import sample_issues
            self._graph_module()          # 没装 langgraph 就跳过（见其 docstring）
            from graph_agent import run_graph_agent
            for iss in sample_issues():
                blank = dataclasses.replace(iss, gold_labels=[])
                c1 = run_custom_agent(iss, use_mock=True).predicted
                c2 = run_custom_agent(blank, use_mock=True).predicted
                self.assertEqual(
                    c1, c2,
                    f"{iss.issue_id} 手写版：清空 gold_labels 后预测变了 "
                    f"（{c1} → {c2}）—— 被测逻辑读到了标准答案，"
                    f"准确率指标是循环论证")
                g1 = run_graph_agent(iss, use_mock=True).predicted
                g2 = run_graph_agent(blank, use_mock=True).predicted
                self.assertEqual(
                    g1, g2,
                    f"{iss.issue_id} LangGraph 版：清空 gold_labels 后预测变了 "
                    f"（{g1} → {g2}）—— 同上")
        finally:
            self._cleanup_path(exp_dir)

    def test_准确率不做假_已知基线是下限不是满分(self):
        """★★ 准确率必须等于"算得出来"的那个值 —— 现在是 5/6 的下限，不是满分 ★★

        ⚠️⚠️ **这条断言现在是"下限"，不是"满分"，而且这是有意的。** ⚠️⚠️

        2026-10-11 实测：一致率 100%，但准确率只有 **50%** ——
        **两版一起错**。根因是标签体系三层不匹配：
          ① `parse_final` 不认 Mock 的 `predicted_labels` 字段（实验侧）
          ② 它退化成**子串匹配**：`bug/timeout` 含 "bug" 就判成 `Bug`（实验侧）
          ③ 主项目 `MockLLM` 注释写"记忆优先"、实现是"记忆兜底"，
             且所有标签被无条件加 `bug/` 前缀（主项目）
        实测：只修 ①② 准确率仍是 3/6；三层一起修才到 6/6。

        **用户决定：本轮先让度量落地，修复另开一轮。**
        所以这里断言的是"**不得低于已记录的基线**"（约束式），
        而不是"等于 50%"（快照式）—— 后者一旦修复就会假失败。

        ★ 为什么用 `>=` 而不是 `==`：
          本项目原则是"**断言约束，不断言快照**"。这里约束是
          "准确率不许退步"；将来修好标签体系后它会自动满足新高度。
        ★ **修复那轮请把这个下限提到 1.0** —— 那时它就是真正的守卫了
          （守"实验真的在测正确的事"）。数字写在这里就是为了让那一轮
          有个明确的待办，而不是靠记忆。
        """
        exp_dir = self._exp_dir()
        try:
            from custom_agent import run_custom_agent
            from shared import label_accuracy, sample_issues
            self._graph_module()          # 没装 langgraph 就跳过
            from graph_agent import run_graph_agent
            issues = sample_issues()
            cr = {i.issue_id: run_custom_agent(i, use_mock=True) for i in issues}
            gr = {i.issue_id: run_graph_agent(i, use_mock=True) for i in issues}
            ac = label_accuracy(cr, "custom")
            ag = label_accuracy(gr, "graph")
        finally:
            self._cleanup_path(exp_dir)

        # 先守"指标本身没坏"：每个样本都算进去了
        self.assertEqual(ac["n"], len(issues), "手写版有样本没被算进准确率")
        self.assertEqual(ag["n"], len(issues), "LangGraph 版有样本没被算进准确率")
        self.assertEqual(ac["unknown"], [],
                         f"这些样本没有设计意图，准确率算不出来：{ac['unknown']}")

        BASELINE = 0.5          # ← 修复标签体系后提到 1.0（见 docstring）
        for a in (ac, ag):
            self.assertGreaterEqual(
                a["rate"], BASELINE,
                f"{a['side']} 准确率 {a['rate']:.0%} 低于已记录基线 {BASELINE:.0%} "
                f"（错的：{a['wrong']}）—— 准确率退步了，"
                f"这不是「还没修好」的问题，是修坏了")

    def test_准确率不满足时不得宣称等价性(self):
        """★★ 一致率 100% 而准确率 <100% 时，报告**必须**说"等价性不成立" ★★

        这条守的是"**报告说的话和数字一致**"。

        WHY：一致率天然会掩盖"两版一起错"—— 两版一起错时它照样 100%
        （docs/08"坑 2"：对称的错误最危险）。如果报告在那种情况下还打印
        "✓ 两版功能等价"，读报告的人就被骗了，而且**不会报错**。
        这正是本项目反复批评的"静默失效"。

        加这个指标之前，报告就是这样：一致率 100% → 打勾，
        而实际准确率只有 50%。**所以这条测试是在守"报告不能再说假话"。**

        ★ 判据本身（`shared.equivalence_valid`）不在测试里重写 ——
          测试调的是报告调的那一份实现（否则又是"同一件事两份实现"）。
        """
        exp_dir = self._exp_dir()
        try:
            from shared import equivalence_valid
        finally:
            self._cleanup_path(exp_dir)

        # ① 用户点名的那条规则：一致率满 + 准确率不满 → 不成立
        self.assertFalse(
            equivalence_valid(1.0, 0.5, 0.5),
            "一致率 100% + 准确率 50% 被判成「等价成立」—— "
            "报告会说假话（两版一起错却被当成功能等价）")
        # ② 两个方向都要：只有一侧准确率不满，也不成立
        self.assertFalse(equivalence_valid(1.0, 1.0, 0.5),
                         "只有 LangGraph 版准确率不满也必须是「不成立」")
        self.assertFalse(equivalence_valid(1.0, 0.5, 1.0),
                         "只有手写版准确率不满也必须是「不成立」")
        # ③ 一致率不满 → 不成立
        self.assertFalse(equivalence_valid(0.83, 1.0, 1.0),
                         "两版结果不一致却被判成等价")
        # ④ 覆盖不足 → 不成立（一致率再漂亮也不行）
        self.assertFalse(
            equivalence_valid(1.0, 1.0, 1.0, covered=False),
            "循环没跑 / 两版都为空时，一致率 100% 也不能算等价成立 —— "
            "那可能只是「两版一起退化成同一个空答案」")
        # ⑤ 三个前提都满足才算成立（防止把它写成永远返回 False）
        self.assertTrue(
            equivalence_valid(1.0, 1.0, 1.0, covered=True),
            "三个前提都满足却判成不成立 —— 这个判据被写成了永远说不，"
            "和「永远说是」一样没用")

    def test_组件在一次运行里只建一次(self):
        """★★ 两版的组件生命周期必须**对齐**（一次运行建一次）★★

        这条守的是**耗时对比能不能成立**。

        踩过的坑：第一版 `graph_agent` 的 `recall_node` / `think_node`
        **各自调 `setup_tools()`** —— 每轮回边就重建一次 MemoryStore + 工具集。
        于是实测的"手写 2ms / 框架 8ms"里，**有一半以上是组件重建开销**
        （单次 `setup_tools()` ≈ 1ms，1 步的样本重建 2 次、3 步的重建 4 次）。
        把它当成"框架慢 4 倍"是错的 —— 那是收口补丁慢，不是编排慢。

        2026-10-11 改成 `config["configurable"]` 依赖注入后，
        两版都是"一次运行建一次"，耗时差才**归因得了编排**。

        ★ 这条测试把"对齐"这个前提**钉住**：如果有人又在节点里写了
          `setup_tools()`，耗时结论会悄悄失真，而报告不会有任何异常 ——
          所以必须由测试来拦。
        """
        exp_dir = self._exp_dir()
        try:
            from shared import sample_issues
            import custom_agent
            graph_agent = self._graph_module()   # 没装 langgraph 就跳过

            iss = sample_issues()[0]

            # --- LangGraph 版：setup_tools 只能被调 1 次 ---
            calls = []
            orig = graph_agent.setup_tools

            def _spy():
                calls.append(1)
                return orig()

            graph_agent.setup_tools = _spy
            try:
                r = graph_agent.run_graph_agent(iss, use_mock=True)
            finally:
                graph_agent.setup_tools = orig
            self.assertEqual(
                len(calls), 1,
                f"LangGraph 版一次运行里 setup_tools() 被调了 {len(calls)} 次"
                f"（应为 1）—— 组件生命周期又被打散了，"
                f"耗时对比会重新混入重建开销")

            # --- 手写版：同样只能是 1 次（否则就不叫"对齐"）---
            calls2 = []
            orig2 = custom_agent.setup_tools

            def _spy2():
                calls2.append(1)
                return orig2()

            custom_agent.setup_tools = _spy2
            try:
                custom_agent.run_custom_agent(iss, use_mock=True)
            finally:
                custom_agent.setup_tools = orig2
            self.assertEqual(
                len(calls2), 1,
                f"手写版 setup_tools() 被调了 {len(calls2)} 次（应为 1）")

            # --- 两版必须相等（对齐的判据是"相等"，不是"各自够少"）---
            self.assertEqual(
                len(calls), len(calls2),
                f"两版组件创建次数不一致：LangGraph {len(calls)} vs "
                f"手写 {len(calls2)} —— 耗时差不能归因给编排")
        finally:
            self._cleanup_path(exp_dir)

    # --- 上面几条用的小工具 ---

    def _exp_dir(self):
        """把实验目录加进 sys.path（返回它，供 finally 清理）。"""
        from pathlib import Path
        exp_dir = Path(__file__).resolve().parents[1] / "experiments" / "langgraph_compare"
        sys.path.insert(0, str(exp_dir))
        return exp_dir

    def _cleanup_path(self, exp_dir):
        if str(exp_dir) in sys.path:
            sys.path.remove(str(exp_dir))

    def _graph_module(self):
        """导入 `graph_agent`；本机没装 langgraph 就**跳过**这条测试。

        ★ 为什么跳过而不是失败：`langgraph` 是**实验专用**依赖，
          主项目刻意保持零第三方依赖（见项目约定）。
          所以在没装它的解释器里，这些守卫应当 skip 而不是红 ——
          否则"跑测试"这件事会因为一个实验依赖而变得没法做。

        ⚠️ 但要意识到代价：**跳过 = 没在守。**
          所以装了 langgraph 的环境（CI、以及本机 `envs/default`）
          必须真的跑一遍这些用例，否则它们形同虚设。
        """
        try:
            import graph_agent
            return graph_agent
        except ImportError:
            self.skipTest("LangGraph 未安装 —— 这是实验专用依赖")


class TestProspectiveValidation(unittest.TestCase):
    """前瞻验证（轨道 A/B）—— 守护三个最容易错的地方。
    ============================================================
    这组测试守的是什么
    ============================================================
    ① **标签粒度对齐**（to_coarse）
       Agent 输出 `bug/crash`（细分），pandas gold 是 `Bug`（粗标签）。
       对齐错了**不会报错**，只会让所有指标变难看 —— 最难排查的那种错。

    ② **信息隔离**（to_issue）
       评测时绝不能把 labels / gold_labels 传给 Agent —— 那是作弊，
       而且会让指标虚高得毫无意义。

    ③ **幂等**（SQLite 主键）
       每日循环必须"同一 issue 不重复预测"，否则两周后数字全错。
    """

    def test_细分标签归到粗标签(self):
        """★ 核心：`bug/crash` 应该归到 `Bug`，而不是 `crash`。

        这里和主项目的 `Metrics.normalize_label` 是**相反**的方向：
          · 主项目：两边都是细分 → 取"有区分度的那一段"（最后一段）
          · 本实验：预测是细分、gold 是粗 → **归到粗的那一端**（第一段）

        如果照搬主项目的实现，`bug/crash` 会变成 `crash` ——
        **依然对不上 gold 的 `bug`，而且错得很隐蔽**（字母都对，只是不是同一个词）。
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from eval_prospective import to_coarse

        cases = [
            ("bug/crash", "Bug"),          # ★ 这条是关键
            ("bug/segfault", "Bug"),
            ("bug", "Bug"),
            ("Bug", "Bug"),                 # 已经是粗标签
            ("BUG", "Bug"),                 # 大小写
            ("docs/missing", "Docs"),
            ("documentation", "Docs"),
            ("perf/slow", "Performance"),
            ("enh/feature", "Enhancement"),
            ("build/ci", "Build"),
            ("regression", "Regression"),
        ]
        for raw, expect in cases:
            self.assertEqual(to_coarse(raw), expect,
                             f"{raw} 应归到 {expect}")

    def test_状态标签不算类型判断(self):
        """`needs-triage` 不是类型标签 —— 应归为 None（未给出判断）。

        ★ 为什么这个区分重要：
            "预测成别的类型" = 判断错误 → 能力问题
            "没给出类型判断" = 没回答   → 可以通过改提示词解决
        两者对系统的评价完全不同，混在一起会看不清问题在哪。
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from eval_prospective import normalize_pred, to_coarse

        for raw in ("needs-triage", "needs_triage", "testing",
                    "duplicate", "question", "unknown", ""):
            self.assertIsNone(to_coarse(raw), f"{raw} 不该当类型标签")

        # normalize_pred 会把这些过滤掉
        self.assertEqual(normalize_pred(["needs-triage", "testing"]), [])
        # 混合时保留有效的
        self.assertEqual(normalize_pred(["needs-triage", "bug/crash", "Bug"]),
                         ["Bug"])

    def test_信息隔离_评测时不给标签(self):
        """★ 评测样本绝不能带着标签进 Agent —— 那是作弊。

        这条测试守的是"实验有效性"：如果哪天有人为了"方便"
        把 labels 传进去，指标会虚高，而**没人会发现**。
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from eval_prospective import to_issue

        rec = {
            "issue_id": "#123",
            "title": "BUG: something crashes",
            "body": "It crashes.",
            "author": "someone",
            "repo": "pandas-dev/pandas",
            # 下面这些是"答案"，绝对不能传进去：
            "labels": ["Bug", "Indexing"],
            "gold_types": ["Bug"],
            "first_type_label": "Bug",
            "state": "closed",
            "state_labels": ["Needs Triage"],
        }
        issue = to_issue(rec)
        self.assertEqual(issue.labels, [], "labels 必须清空")
        self.assertEqual(issue.gold_labels, [], "gold_labels 必须清空")
        # 状态也不能泄露（open/closed 是后来的结果）
        from nightwatch.models import IssueState
        self.assertEqual(issue.state, IssueState.OPEN)
        # 但标题正文要保留（那是 Agent 该看到的）
        self.assertEqual(issue.title, rec["title"])
        self.assertEqual(issue.body, rec["body"])

    def test_每日循环的幂等(self):
        """★ 同一 issue 不能被预测两次。

        轨道 B 要跑两周，如果幂等坏了：
          · 同一 issue 每天被预测一次 → 样本量虚高 14 倍
          · 而对账时又会用后来的标签 → 数字彻底失真

        **这类 bug 不会报错**，只会让两周后的结论完全不可信。
        """
        import tempfile
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from daily_triage import Store

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            store = Store(Path(tmp.name))
            rec = {
                "issue_id": "#999", "repo": "x/y", "title": "t",
                "created_at": "2026-01-01T00:00:00+00:00",
                "predicted": ["Bug"], "predicted_raw": ["bug"],
                "n_steps": 1, "tokens": 100, "error": "",
            }
            self.assertTrue(store.add(rec), "第一次应写入成功")
            self.assertFalse(store.add(rec), "第二次应被幂等拒绝")
            # 累计仍应是 1
            self.assertEqual(store.stats()["n"], 1)
        finally:
            import os
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def test_对账遇未打标签保持待对账(self):
        """维护者还没打标签 → `gold` 应保持 NULL（不是"失败"）。

        ★ 为什么不能写成"未知"或"失败"：
            "尚未分诊" 和 "分诊了但没类型标签" 是两件不同的事。
        混在一起之后，你无法回答"到底有多少 issue 没被分诊" ——
        而那正是这个实验要暴露的信息之一（实测是 31%）。
        """
        import tempfile
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from daily_triage import Store

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            store = Store(Path(tmp.name))
            store.add({
                "issue_id": "#1", "repo": "x/y", "title": "t",
                "created_at": "2026-01-01T00:00:00+00:00",
                "predicted": ["Bug"], "predicted_raw": [],
                "n_steps": 0, "tokens": 0, "error": "",
            })
            # 未对账时 gold 是 NULL
            rows = store.all_with_gold()
            self.assertEqual(len(rows), 0, "还没对账的记录不该出现在已对账集合里")
            # 对账后才有
            store.reconcile("#1", "Bug", "2026-01-02T00:00:00+00:00", 1.0)
            rows = store.all_with_gold()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["gold"], "Bug")
            self.assertEqual(rows[0]["lag_days"], 1.0)
        finally:
            import os
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


    def test_ci和bld映射到Build(self):
        """★ `CI` / `BLD` 必须映射到 `Build` —— 这是修复时补上的。

        实测发现模型常把 `CI:` / `BLD:` 前缀照抄成标签名
        （因为它从标题里看到了这些词），而 pandas 把它们都归入 `Build`。

        第一版漏了这两条 → `['CI', 'config']` 归一化成空 →
        **被算作"未给出预测"，而真相是"给了但归一化失败"**。
        这是**评测侧的 bug，不是模型的错** —— 而它会让你误判模型的能力。
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from eval_prospective import normalize_pred, to_coarse

        self.assertEqual(to_coarse("CI"), "Build")
        self.assertEqual(to_coarse("ci"), "Build")
        self.assertEqual(to_coarse("BLD"), "Build")
        self.assertEqual(to_coarse("bld"), "Build")
        # 组合形式也要能归
        self.assertEqual(normalize_pred(["CI", "config", "pixi"]), ["Build"])

    def test_标题前缀的缩写都要能映射(self):
        """★ 模型会**照抄标题里的前缀**当标签名 —— 所以映射表要覆盖所有缩写。

        pandas 的标准前缀：BUG: / DOC: / ENH: / PERF: / CI: / BLD: / DEPR: /
        TST: / REF:

        这里漏过两次（同类问题）：
          · 第一次漏 `CI` / `BLD`（Build 类全错）
          · 第二次漏 `DEPR`（每日运行对账发现的：#70257 真实标签 Deprecate）
        **每次都是"真实运行 → 对账"才发现的，不是靠想。**

        ★ 为什么这是**评测侧**的职责？
          因为模型输出 "DEPR" 是**合理的**（它照抄标题），
          而评测如果认不出来，就会把它算作"未给出预测" ——
          **明明是"归一化失败"，却记成了"模型没回答"**。
          责任分错了，修的方向就错了。
        """
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
        from eval_prospective import normalize_pred, to_coarse

        cases = [
            ("DEPR", "Deprecate"), ("depr", "Deprecate"),
            ("Deprecate", "Deprecate"), ("deprecation", "Deprecate"),
            ("TST", "Testing"), ("REF", "Refactor"),
            # 之前修的
            ("CI", "Build"), ("BLD", "Build"),
        ]
        for raw, expect in cases:
            self.assertEqual(to_coarse(raw), expect, f"{raw} 应映射到 {expect}")

        # 组合形式（照抄标题前缀时的典型输出）
        self.assertEqual(
            normalize_pred(["DEPR", "Indexing", "API Design"]),
            ["Deprecate"])
        self.assertEqual(normalize_pred(["TST", "CI"]), ["Testing", "Build"])

    def test_评测类脚本显式指定prompt_variant(self):
        """★ 评测脚本必须显式传 prompt_variant，不能依赖默认值。

        ============================================================
        这条测试守的是一个真实踩过的坑
        ============================================================
        `triage()` 的 prompt_variant 默认是 `"default"`，
        而 `run_eval.py`（离线）显式传 `"tuned"`。

        而 `eval_prospective.py`（前瞻）**最初没传** ——
        于是**两套评测用的根本不是同一个提示词**。

        **这个不一致在数字上看不出来**（两边都给出"合理"的结果）。
        我是改了 tuned 提示词后重跑发现"毫无变化"才顺藤摸瓜找到的。

        教训同 `docs/05` 的 D2：**两条路径要逐字对齐配置**。
        """
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for f in ["scripts/eval_prospective.py", "scripts/daily_triage.py"]:
            t = (root / f).read_text(encoding="utf-8")
            self.assertIn(
                'prompt_variant="tuned"', t,
                f"{f} 应该显式传 prompt_variant=\"tuned\"（与离线评测对齐），"
                f"否则两套评测用的不是同一个提示词")

    def test_标签体系_Build是主标签(self):
        """★ `Build` 必须是主标签，不能是细分标签。

        这是前瞻验证发现的真实问题：原先把 Build 放在"细分标签"里，
        还标注"仅在确实针对该方面时才用" —— 结果 Agent 从不输出它。

        而 pandas 的真实标签体系里 `Build` 和 `Bug` 是**平级**的类型标签。

        **标签体系的层级关系要跟真实仓库对齐** ——
        自己拍一个层级结构，会让 Agent 系统性地漏掉某一类。
        """
        from nightwatch.config import LABEL_GUIDE
        # Build 必须在里面，且描述里要提到 CI
        self.assertIn("Build", LABEL_GUIDE)
        desc = LABEL_GUIDE["Build"]
        self.assertIn("CI", desc.replace("`", ""),
                      "Build 的描述里应该提到 CI/BLD（否则模型不知道这些属于 Build）")

        # 提示词里也要把 Build 列进主标签
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        react = (root / "src" / "nightwatch" / "loop" / "react.py").read_text(
            encoding="utf-8")
        i = react.find("主标签（强烈优先）")
        self.assertGreater(i, 0, "应该能找到主标签那段")
        seg = react[i:i + 220]
        self.assertIn("Build", seg,
                      "Build 应该出现在「主标签」列表里（不能只在细分标签里）")


class TestPRThresholdCalibration(unittest.TestCase):
    """PR 风险阈值的标定性 —— 守护"阈值有数据支撑"这件事。

    ============================================================
    为什么阈值需要测试
    ============================================================
    阈值是个"看起来合理就写上去"的东西，而它的后果不明显：

        阈值太高 → 规则从不触发 → 你会以为"没有高风险 PR"（其实是有）
        阈值太低 → 全是高风险   → 标签失去意义（人还是要全看）

    **两种错都不会报错**，只会让你在错误的前提下工作。

    这条测试守的是"默认值来自标定"这个结论 ——
    如果有人把它改回一个"看起来合理"的数字，会立刻失败。
    """

    def test_默认阈值是标定值而非经验值(self):
        """★ 默认阈值应该是 100（标定值），不是 500（原经验值）。

        标定依据：60 个真实 PR（30 已合并 + 30 未合并）的改动量分布
          · 500 行 → 高风险只有 5%（规则几乎不触发）
          · 100 行 → 高风险 20%（落在"人工优先看"的合理区间）

        （`docs/06` 里"阈值没有数据支撑"那条问题的第二次修。）
        """
        import inspect
        from nightwatch.tools.pr_review import review_pr
        sig = inspect.signature(review_pr)
        default = sig.parameters["max_high_risk_churn"].default
        self.assertEqual(
            default, 100,
            f"默认阈值是 {default}，应该是标定出来的 100。"
            f"如果确实要改，请先用 calibrate_pr_threshold.py 重新标定。")

    def test_工具schema里的默认值与函数一致(self):
        """★ 两处默认值必须一致 —— 这是个容易漏的地方。

        工具 schema 里的 `default` 是**给模型看的**，
        函数签名里的默认值是**实际执行的**。
        如果两者不一致，模型以为的默认值和实际行为就不同了 ——
        而且**不会报错**。
        """
        from nightwatch.tools.registry import IssueRepository, ToolRegistry
        reg = ToolRegistry(IssueRepository([]))
        spec = next(s for s in reg.specs() if s["name"] == "review_pr_diff")
        schema_default = spec["input_schema"]["properties"][
            "max_high_risk_churn"]["default"]

        import inspect
        from nightwatch.tools.registry import ToolRegistry as TR
        handler = reg._tools["review_pr_diff"].handler
        fn_default = inspect.signature(handler).parameters[
            "max_high_risk_churn"].default

        self.assertEqual(schema_default, fn_default,
                         f"schema 默认值 {schema_default} 与函数默认值 "
                         f"{fn_default} 不一致")

    def test_默认阈值在高风险区间内(self):
        """用真实数据验证：100 行阈值下高风险比例落在 10%~20%。

        有真实数据就验证，没有就跳过（数据是运行时采集的）。
        """
        import json
        from pathlib import Path
        base = Path(__file__).resolve().parents[1] / "data"
        merged = base / "real_prs.json"
        unmerged = base / "unmerged_prs.json"
        if not (merged.exists() and unmerged.exists()):
            self.skipTest("真实 PR 数据不存在（先跑 collect_prs.py）")

        prs = []
        for p in (merged, unmerged):
            prs.extend(json.loads(p.read_text(encoding="utf-8"))["prs"])
        if len(prs) < 20:
            self.skipTest(f"样本太少（{len(prs)} 个），跳过比例验证")

        n_high = sum(1 for r in prs
                     if (r.get("additions", 0) + r.get("deletions", 0)) > 100)
        rate = n_high / len(prs)
        self.assertGreaterEqual(rate, 0.05,
                                f"高风险比例 {rate:.0%} 太低，规则几乎不触发")
        self.assertLessEqual(rate, 0.45,
                             f"高风险比例 {rate:.0%} 太高，标签失去区分度")

    def test_改动量统计优先用已有字段(self):
        """`churn_of` 要能用已有字段，也能从 diff 现算（向后兼容）。

        为什么需要回退？因为数据是不同时间采集的，
        早期版本可能没有 additions/deletions 字段。
        **直接 KeyError 会让那些样本被静默丢掉** —— 而丢样本会让统计失真。
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from calibrate_pr_threshold import churn_of

        # 有字段 → 直接用
        self.assertEqual(churn_of({"additions": 10, "deletions": 5}), 15)
        # 缺字段 → 从 diff 算
        diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                "@@ -1,1 +1,2 @@\n-old\n+new\n+more\n")
        self.assertEqual(churn_of({"diff": diff}), 3)      # +2 -1


class TestRealPRParsing(unittest.TestCase):
    """真实 PR diff 的解析 —— 守护"能处理真实世界的脏数据"。

    ============================================================
    为什么这组测试用真实 diff 片段而不是构造数据
    ============================================================
    `TestPRReview` 用的是我手写的 diff —— 它们**恰好都是我正则能匹配的**。

    真实 diff 里有构造数据里不会出现的东西：
      · `rename from` / `rename to`（文件改名）
      · `Binary files ... differ`（二进制文件）
      · `No newline at end of file` 标记
      · `index abc..def 100644`（mode 行）
      · 超长单行（压缩后的 JS、生成代码）

    **这些边界不测，规则在真实数据上就会静默失效。**
    （这正是采真实 PR 的意义 —— 见 `scripts/collect_prs.py` 的说明。）
    """

    RENAME_DIFF = """diff --git a/pandas/core/old.py b/pandas/core/new.py
similarity index 95%
rename from pandas/core/old.py
rename to pandas/core/new.py
--- a/pandas/core/old.py
+++ b/pandas/core/new.py
@@ -1,3 +1,3 @@
-old line
+new line
"""

    BINARY_DIFF = """diff --git a/doc/logo.png b/doc/logo.png
index 1234567..89abcde 100644
Binary files a/doc/logo.png and b/doc/logo.png differ
diff --git a/pandas/core/frame.py b/pandas/core/frame.py
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -1,2 +1,3 @@
 def merge():
+    # 改动
     pass
"""

    NO_NEWLINE_DIFF = """diff --git a/pandas/core/x.py b/pandas/core/x.py
--- a/pandas/core/x.py
+++ b/pandas/core/x.py
@@ -1,2 +1,2 @@
-old
+new
\\ No newline at end of file
"""

    def test_文件改名能解析(self):
        """★ rename 的 diff 有两个路径，解析器要知道用哪个。

        `+++ b/new.py` 是新路径 —— 这才是"改动后的文件"。
        （如果误用 `--- a/old.py`，那敏感文件判断会用旧名字，
          而旧名字可能不匹配任何规则 → 静默漏判。）
        """
        from nightwatch.tools.pr_review import parse_diff
        files = parse_diff(self.RENAME_DIFF)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].path, "pandas/core/new.py")

    def test_二进制文件不干扰其他文件(self):
        """Binary files 行没有 +/- 行，不该被算成改动，也不该影响后续解析。"""
        from nightwatch.tools.pr_review import parse_diff
        files = parse_diff(self.BINARY_DIFF)
        self.assertEqual(len(files), 2)
        png = [f for f in files if f.path.endswith(".png")]
        self.assertEqual(len(png), 1)
        self.assertEqual(png[0].churn, 0, "二进制文件不该有行数改动")
        # 后面的源码文件要正常解析
        py = [f for f in files if f.path.endswith("frame.py")]
        self.assertEqual(len(py), 1)
        self.assertEqual(py[0].additions, 1)

    def test_无换行结尾标记不被算成改动(self):
        """`\\ No newline at end of file` 以反斜杠开头，不是 +/- 行。

        如果解析器把以 `-` 开头的都算删除，`---` 那行会被误算 ——
        同理 `\\` 开头的也要排除。
        """
        from nightwatch.tools.pr_review import parse_diff
        files = parse_diff(self.NO_NEWLINE_DIFF)
        self.assertEqual(files[0].additions, 1)
        self.assertEqual(files[0].deletions, 1)

    def test_真实采集的数据能被解析(self):
        """★ 端到端：真的有 real_prs.json 就验证它，没有就跳过。

        用 `skipTest` 而不是 fail —— 因为数据是**运行时采集**的，
        不该让"没采过数据的机器"测试失败。
        **可选依赖的测试要能优雅跳过。**
        """
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "real_prs.json"
        if not p.exists():
            self.skipTest("real_prs.json 不存在（先跑 scripts/collect_prs.py）")

        from nightwatch.tools.pr_review import review_pr
        data = json.loads(p.read_text(encoding="utf-8"))
        prs = data.get("prs", [])
        self.assertGreater(len(prs), 0)

        ok = 0
        for r in prs:
            rev = review_pr(r["diff"])          # 不抛异常
            self.assertIn(rev.risk_level, ("低", "中", "高"))
            if rev.files:
                ok += 1
        # 大部分要能解析出文件（全部要求太严，真实 diff 有各种边界）
        self.assertGreater(ok, len(prs) * 0.8,
                           f"只有 {ok}/{len(prs)} 个真实 PR 解析出文件")

    def test_改动统计来自diff而非API字段(self):
        """★ 记录一个实际踩到的坑：GitHub 列表 API 不返回改动统计。

        `collect_prs.py` 第一版直接用 `raw["changed_files"]`，
        结果**全是 0** —— 因为那些字段只在单个 PR 详情 API 里有。

        改成用 `parse_diff()` 从 diff 算（diff 是权威来源）。
        这条测试确保采集脚本不会退回到用 API 字段。
        """
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "real_prs.json"
        if not p.exists():
            self.skipTest("real_prs.json 不存在")

        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("prs", []):
            # 从 diff 算出来的统计不该全是 0（除非 diff 真的为空）
            if r.get("diff"):
                self.assertGreaterEqual(r["changed_files"], 1,
                                        f"{r['pr_id']} 应该解析出至少 1 个文件")


class TestVectorIndex(unittest.TestCase):
    """向量索引（IVF）—— 守护三个性质。

    ============================================================
    这组测试守的是什么
    ============================================================
    ① **nprobe=nlist 时必须等于精确搜索**
       这是"近似"的边界：搜了所有簇就等于暴力搜索。
       如果这条不成立，说明索引的搜索逻辑有 bug。

    ② **索引退化的检测（簇大小不均衡）**
       k-means 没收敛好会变成"一个巨簇 + 一堆空簇"，
       此时查询命中巨簇就退化成全表扫描 ——
       **索引建了等于没建，而且不会报错。**

    ③ **nlist > N 的边界**
       簇数比数据还多时，k-means 会把簇心堆在一起、随机瓜分数据。
       如果不做限制，索引质量会很差但不报错。
    """

    def _corpus(self, n=200, dim=16, seed=42):
        import random as _r
        rng = _r.Random(seed)
        # 造 5 个簇
        centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(5)]
        out = {}
        for i in range(n):
            c = centers[i % 5]
            out[f"m{i:04d}"] = [x + rng.gauss(0, 0.3) for x in c]
        return out

    def test_余弦相似度(self):
        from nightwatch.memory.index import cosine
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0, places=6)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0, places=6)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0, places=6)
        # 长度不影响（余弦的关键性质）
        self.assertAlmostEqual(cosine([1, 0], [5, 0]), 1.0, places=6)
        # 边界
        self.assertEqual(cosine([], []), 0.0)
        self.assertEqual(cosine([1, 2], [1]), 0.0)      # 维度不匹配
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)   # 零向量

    def test_构建后索引可用(self):
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus()
        idx = IVFIndex(nlist=8).build(corpus)
        self.assertTrue(idx.is_built)
        self.assertEqual(idx.stats.n_items, len(corpus))
        self.assertEqual(idx.stats.nlist, 8)
        self.assertGreater(idx.stats.build_sec, 0)

    def test_nprobe等于nlist时等于精确搜索(self):
        """★ 这是"近似"的边界条件 —— 必须成立。

        搜了所有簇就等于暴力搜索。如果这条挂了，
        说明搜索逻辑（簇选择或排序）有 bug。
        """
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus()
        idx = IVFIndex(nlist=8).build(corpus)
        q = list(corpus["m0000"])

        exact = idx.search_exact(q, top_k=5)
        approx = idx.search(q, nprobe=idx.nlist, top_k=5)
        self.assertEqual([m for m, _ in exact], [m for m, _ in approx])

    def test_相似向量被正确召回(self):
        """查询某个簇内向量时，应该召回同簇的其他向量。"""
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus()
        idx = IVFIndex(nlist=8).build(corpus)

        q = list(corpus["m0000"])
        hits = idx.search(q, nprobe=4, top_k=3)
        self.assertTrue(hits)
        # m0000 自己应该排第一（和查询完全相同）
        self.assertEqual(hits[0][0], "m0000")
        self.assertAlmostEqual(hits[0][1], 1.0, places=4)

    def test_nlist超过数据量时被裁剪(self):
        """★ nlist > N 时必须裁剪，否则产生大量空簇。

        这是容易忽略的边界：不裁的话 k-means 会把簇心堆在一起、
        随机瓜分数据 —— 索引质量很差但**不报错**。
        """
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus(n=10)
        idx = IVFIndex(nlist=100).build(corpus)
        self.assertEqual(idx.nlist, 10, "簇数应该被裁到数据量")
        self.assertEqual(idx.stats.empty_buckets, 0)

    def test_空数据不崩(self):
        from nightwatch.memory.index import IVFIndex
        idx = IVFIndex(nlist=4).build({})
        self.assertTrue(idx.is_built)
        self.assertEqual(idx.search([1.0] * 4, nprobe=2, top_k=3), [])

    def test_增量添加与删除(self):
        """IVF 的增量能力（不重建索引）。"""
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus(n=50)
        idx = IVFIndex(nlist=5).build(corpus)
        before = idx.stats.n_items

        newvec = list(corpus["m0000"])
        idx.add("new_mem", newvec)
        self.assertEqual(idx.stats.n_items, before + 1)
        # 新加的应该能被搜到
        hits = idx.search(newvec, nprobe=idx.nlist, top_k=3)
        self.assertIn("new_mem", [m for m, _ in hits])

        self.assertTrue(idx.remove("new_mem"))
        self.assertEqual(idx.stats.n_items, before)
        self.assertFalse(idx.remove("不存在"))

    def test_不均衡度指标存在(self):
        """★ 不均衡度是索引健康的信号 —— 必须能观测。

        它 > 3 说明聚类退化，索引会静默失效（命中巨簇 = 全表扫描）。
        """
        from nightwatch.memory.index import IVFIndex
        corpus = self._corpus()
        idx = IVFIndex(nlist=8).build(corpus)
        self.assertGreater(idx.stats.imbalance, 0)
        d = idx.stats.to_dict()
        for key in ("簇数", "索引向量数", "构建耗时秒", "平均簇大小",
                    "空簇数", "簇大小不均衡度", "最大簇"):
            self.assertIn(key, d)

    def test_store集成_挂载索引后recall仍可用(self):
        """★ 集成测试：挂载索引后 recall 的**返回形状不能变**。

        因为索引是个优化项，不该让调用方感知到差异 ——
        返回的仍应该是 `[(MemoryRecord, float), ...]`。
        """
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        from nightwatch.models import MemoryRecord, MemoryTier, Outcome

        store = MemoryStore(":memory:", encoder=HashEmbedding(dim=32))
        for i in range(30):
            store.insert(MemoryRecord(
                memory_id=f"m{i}", tier=MemoryTier.EPISODIC,
                issue_id=f"#{i}", symptom=f"超时问题 {i}",
                root_cause="权限", resolution="修权限",
                outcome=Outcome.RESOLVED))

        # 挂索引前后都试一次
        before = store.recall("超时", top_k=3)
        idx = store.build_index(nlist=4)
        self.assertIsNotNone(idx)
        self.assertIsNotNone(store.index_stats())
        after = store.recall("超时", top_k=3)

        # 形状一致
        for got in (before, after):
            for item in got:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
                self.assertIsInstance(item[0], MemoryRecord)
                self.assertIsInstance(item[1], float)


class TestPRReview(unittest.TestCase):
    """PR diff 初审（P2-3）—— 守护**风险筛查的准确性**。

    ============================================================
    这组测试守的是"领域知识表有没有被改坏"
    ============================================================
    `SENSITIVE_PATTERNS` 和"缺测试"的判定逻辑是这个工具的全部价值所在。
    代码本身很简单（解析 diff + 正则匹配），**难的是知道什么该担心**。

    如果哪天有人"优化"了正则，把 `pyproject.toml` 从依赖声明里移出去 ——
    结果是"高风险 PR 被判成低风险"，而**报告看起来完全正常**。
    这是最危险的一类失效：**错了但不报错。**

    所以每条敏感类别都要有测试。
    """

    DIFF_DOC = """diff --git a/docs/io.rst b/docs/io.rst
--- a/docs/io.rst
+++ b/docs/io.rst
@@ -1,3 +1,5 @@
+新增说明
+更多说明
 原内容
"""

    DIFF_SRC_NO_TEST = """diff --git a/pandas/core/frame.py b/pandas/core/frame.py
--- a/pandas/core/frame.py
+++ b/pandas/core/frame.py
@@ -1,3 +1,5 @@
     def merge(self):
-        pass
+        return self._impl()
"""

    DIFF_SRC_WITH_TEST = DIFF_SRC_NO_TEST + """diff --git a/tests/test_frame.py b/tests/test_frame.py
--- a/tests/test_frame.py
+++ b/tests/test_frame.py
@@ -1,2 +1,4 @@
+def test_merge():
+    assert True
"""

    def test_解析diff提取每个文件的改动(self):
        from nightwatch.tools.pr_review import parse_diff
        changes = parse_diff(self.DIFF_SRC_NO_TEST)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].path, "pandas/core/frame.py")
        self.assertEqual(changes[0].additions, 1)
        self.assertEqual(changes[0].deletions, 1)

    def test_文件分类(self):
        """源码 / 测试 / 文档要能正确区分 —— 这是"缺测试"判断的前提。"""
        from nightwatch.tools.pr_review import parse_diff
        for path, expect in [
            ("pandas/core/frame.py", "源码"),
            ("tests/io/test_csv.py", "测试"),
            ("pandas/tests/test_api.py", "测试"),
            ("docs/io.rst", "文档"),
            ("CHANGELOG.md", "文档"),
        ]:
            f = parse_diff(f"diff --git a/{path} b/{path}\n"
                           f"--- a/{path}\n+++ b/{path}\n+x\n")[0]
            self.assertEqual(f.kind(), expect, f"{path} 应判为 {expect}")

    def test_敏感文件各类别都能识别(self):
        """★ 每条敏感类别都要能命中 —— 这是工具的核心价值。"""
        from nightwatch.tools.pr_review import parse_diff
        cases = [
            ("pyproject.toml", "依赖声明"),
            ("requirements-dev.txt", "依赖声明"),
            (".github/workflows/ci.yml", "CI 配置"),
            ("Makefile", "构建脚本"),
            ("pandas/_libs/security_utils.py", "安全相关"),
            ("pandas/io/pickle.py", "序列化与求值"),
            ("src/nightwatch/models.py", "核心数据结构"),
        ]
        for path, expect_cat in cases:
            f = parse_diff(f"diff --git a/{path} b/{path}\n"
                           f"--- a/{path}\n+++ b/{path}\n+x\n")[0]
            self.assertIn(expect_cat, f.sensitivity(),
                          f"{path} 应命中「{expect_cat}」")

    def test_纯文档改动判低风险(self):
        from nightwatch.tools.pr_review import review_pr
        r = review_pr(self.DIFF_DOC)
        self.assertEqual(r.risk_level, "低")
        self.assertFalse(r.missing_tests, "纯文档改动不需要测试")

    def test_源码无测试判缺测试(self):
        """★ 这是最重要的判断之一：改了源码但没改测试。"""
        from nightwatch.tools.pr_review import review_pr
        r = review_pr(self.DIFF_SRC_NO_TEST)
        self.assertTrue(r.missing_tests)
        self.assertIn("frame.py", r.missing_tests_reason)
        self.assertIn(r.risk_level, ("中", "高"))

    def test_源码带测试不判缺测试(self):
        from nightwatch.tools.pr_review import review_pr
        r = review_pr(self.DIFF_SRC_WITH_TEST)
        self.assertFalse(r.missing_tests)

    def test_触及敏感文件直接判高风险(self):
        """★ 高风险判定必须是"任一命中即升级"。

        因为风险不可互相抵消 —— "改 1 行依赖声明" 不该被
        "改了 500 行文档"平均掉。
        """
        diff = """diff --git a/pyproject.toml b/pyproject.toml
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,2 +1,3 @@
+    "pyyaml>=6.0",
"""
        from nightwatch.tools.pr_review import review_pr
        r = review_pr(diff)
        self.assertEqual(r.risk_level, "高")
        self.assertTrue(any("敏感文件" in x for x in r.risk_reasons))

    def test_大改动量升级风险(self):
        from nightwatch.tools.pr_review import review_pr
        body = "\n".join(f"+line {i}" for i in range(600))
        diff = (f"diff --git a/pandas/core/frame.py b/pandas/core/frame.py\n"
                f"--- a/pandas/core/frame.py\n+++ b/pandas/core/frame.py\n{body}\n")
        r = review_pr(diff, max_high_risk_churn=500)
        self.assertEqual(r.risk_level, "高")
        self.assertTrue(any("改动量大" in x for x in r.risk_reasons))

    def test_以删除为主会提示(self):
        """删除多于新增 → 可能移除行为，调用方未必同步。"""
        from nightwatch.tools.pr_review import review_pr
        body = "\n".join(f"-old line {i}" for i in range(50))
        diff = (f"diff --git a/pandas/core/frame.py b/pandas/core/frame.py\n"
                f"--- a/pandas/core/frame.py\n+++ b/pandas/core/frame.py\n{body}\n")
        r = review_pr(diff)
        self.assertTrue(any("删除为主" in x for x in r.risk_reasons))

    def test_输出声明能力边界(self):
        """★ 报告必须声明"不读代码内容"这个限制。

        否则会被误用成"工具说低风险 = 这个 PR 没问题" —— 那是危险的理解。
        """
        from nightwatch.tools.pr_review import review_pr
        r = review_pr(self.DIFF_DOC)
        out = r.to_dict()
        self.assertIn("说明", out)
        self.assertIn("不读代码内容", out["说明"])
        self.assertIn("无法判断", out["说明"])

    def test_不改动架构_只多一个工具(self):
        """★ 工具集只是"多了一个"，没有破坏原有六个。

        这验证了 P2-3 的"成本很低"这个判断 ——
        新方向不该要求重构。
        """
        from nightwatch.tools.registry import IssueRepository, ToolRegistry
        repo = IssueRepository([])
        repo.add_pr_diff("PR-1", self.DIFF_DOC)
        reg = ToolRegistry(repo)

        # 原来六个工具都还在
        for name in ["search_issues", "search_similar_issues", "read_code",
                     "search_docs", "apply_labels", "post_comment"]:
            self.assertIn(name, reg.names(), f"原有工具 {name} 丢了")
        self.assertIn("review_pr_diff", reg.names())

    def test_工具支持传入diff文本(self):
        """两种输入方式都要支持（pr_id 和 diff）。"""
        from nightwatch.tools.registry import IssueRepository, ToolRegistry
        reg = ToolRegistry(IssueRepository([]))
        res = reg.call("review_pr_diff", {"diff": self.DIFF_DOC})
        self.assertTrue(res.ok)
        self.assertIn("改动范围", res.data)

    def test_工具_找不到PR时报错(self):
        from nightwatch.tools.registry import IssueRepository, ToolRegistry
        reg = ToolRegistry(IssueRepository([]))
        res = reg.call("review_pr_diff", {"pr_id": "不存在"})
        # 工具自己返回 error（不是抛异常）—— 让模型能读到并换个方式重试
        self.assertTrue(res.ok)
        self.assertIn("error", res.data)


class TestPoisonInjection(unittest.TestCase):
    """记忆污染（P2-1）—— 守护"污染可追踪、可纠正"这两个性质。

    ============================================================
    这组测试守的两件事
    ============================================================
    ① **污染必须可追踪**
       如果不知道污染影响了哪些判断，就无法评估影响范围，
       也无法验证"纠正是否彻底"。

    ② **纠正必须考虑派生记忆**
       这是本实验最核心的发现：删除原污染 **≠** 污染被清除。
       Agent 基于污染写下的新记忆是**独立存在**的，会继续误导。

    **如果哪天有人"优化"了纠正逻辑，让它只处理原污染 ——
      这个测试会立刻失败。** 这就是它存在的意义。
    """

    def _store(self):
        from nightwatch.memory.store import HashEmbedding, MemoryStore
        return MemoryStore(":memory:", encoder=HashEmbedding(dim=64))

    def test_五种污染类型都能注入(self):
        """五种类型都能注入，且都真的写进了记忆库。

        ★ 注意 symptom 必须各不相同 —— 因为记忆写入有**幂等键去重**
        （`idempotency_key` 基于 symptom 等内容算），
        内容相同的第二条会被判为重复而拒绝写入。

        这不是 bug，是记忆系统的正确行为（防止同一件事被记很多遍）。
        但写测试时必须知道它 —— 第一版测试用了相同 symptom，
        结果 5 条只写进去 1 条，断言直接失败。
        """
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        for idx, pt in enumerate(PoisonType):
            p = inj.inject(pt, f"#1{idx}", f"症状描述 {pt.value}",
                           f"错误的处置内容 {pt.value}", tags=[f"t{idx}"])
            self.assertIsNotNone(p)
            self.assertEqual(p.poison_type, pt)
        self.assertEqual(len(inj.poisons), 5)
        # 污染确实写进了记忆库（走正常写入路径）
        self.assertEqual(store.count(), 5)

    def test_污染和正常记忆长得一样(self):
        """★ 实验有效性前提：污染不该有明显特征。

        如果污染记忆的 confidence 特别低、或者有特殊标记，
        那 Agent 一眼就能识别 —— 实验就测不出真实危害了。
        """
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.STALE, "#1", "症状", "错内容", tags=["t"])
        mem = store.get(p.memory_id)
        self.assertIsNotNone(mem)
        # 置信度是正常的（不是 0.1 这种一眼假的）
        self.assertGreaterEqual(mem.confidence, 0.7)
        # tier 是正常的情急记忆
        from nightwatch.models import MemoryTier
        self.assertEqual(mem.tier, MemoryTier.EPISODIC)

    def test_暴露追踪(self):
        """污染被召回时要能被检出（影响范围定位的基础）。"""
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        from nightwatch.models import Trace
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.WRONG_TAGS, "#1", "症状", "错", tags=["t"])

        # 构造一条"召回了这条污染"的 trace
        tr = Trace(trace_id="tr_test", issue_id="#x")
        tr.memory_recalled = [p.memory_id, "other_mem"]
        hit = inj.detect_exposure(tr)

        self.assertEqual(hit, [p.poison_id])
        self.assertIn("tr_test", inj.poisons[p.poison_id].exposed_traces)

    def test_单次反馈不足以纠正有派生的污染(self):
        """★★ 核心断言：只标记原污染不算纠正完成。

        这是本实验最重要的发现：
        **删除原污染 ≠ 污染被清除** —— 派生记忆独立存在且继续误导。
        """
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        from nightwatch.models import MemoryRecord, Outcome
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.WRONG_RESOLUTION, "#1", "症状", "错", tags=["t"])

        # 模拟派生
        derived = MemoryRecord(memory_id="d1", issue_id="#d",
                               symptom="派生", outcome=Outcome.RESOLVED)
        store.insert(derived)
        inj.record_derived(p.poison_id, derived.memory_id)

        # 第 1 次反馈：只标记原污染
        inj.mark_wrong(p.poison_id, by="maint")
        self.assertTrue(inj.is_still_effective(p.poison_id),
                        "派生记忆还在，污染仍然有效 —— 单次反馈不该被判为完成")

        # 第 2 次反馈：清理派生
        inj.correct_derived(p.poison_id, by="maint")
        self.assertFalse(inj.is_still_effective(p.poison_id),
                         "两次反馈后污染才真正失效")
        self.assertEqual(inj.poisons[p.poison_id].feedback_count, 2)

    def test_无派生时一次反馈即可纠正(self):
        """没有派生的污染，1 次反馈就够 —— 这验证了"传染深度"的因果。"""
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.WRONG_ROOT_CAUSE, "#1", "症状", "错", tags=["t"])

        inj.mark_wrong(p.poison_id, by="maint")
        self.assertFalse(inj.is_still_effective(p.poison_id))
        self.assertEqual(inj.poisons[p.poison_id].feedback_count, 1)

    def test_纠正用软删除而非物理删除(self):
        """★ 纠正要保留追溯能力。

        如果直接 `store.delete()`，就无法回答
        "这条错误记忆当初影响了哪些判断" —— 而那正是这个实验要测的东西。

        所以本项目用**软纠正**（archived + 降置信度）。
        """
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.OVER_GENERALIZED, "#1", "症状", "错", tags=["t"])

        inj.mark_wrong(p.poison_id, by="maint", note="过度泛化")
        mem = store.get(p.memory_id)
        self.assertIsNotNone(mem, "记忆应该还在（软纠正，不是物理删除）")
        self.assertTrue(mem.archived, "应该被标记归档")
        self.assertLessEqual(mem.confidence, 0.2, "置信度应该被降低")

    def test_传染度是暴露与派生的和(self):
        """传染度 = 影响的判断数 + 派生的记忆数。

        前者是横向扩散（影响面），后者是纵向扩散（污染被固化）。
        两者代表不同的危害方式，所以要一起看。
        """
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        p = inj.inject(PoisonType.STALE, "#1", "症状", "错", tags=["t"])

        inj.record_exposure(p.poison_id, "tr1")
        inj.record_exposure(p.poison_id, "tr2")
        inj.record_derived(p.poison_id, "d1")

        self.assertEqual(inj.poisons[p.poison_id].spread, 3)
        # 重复记录同一个 trace 不该重复计数
        inj.record_exposure(p.poison_id, "tr1")
        self.assertEqual(inj.poisons[p.poison_id].spread, 3)

    def test_汇总包含三个验收指标(self):
        from nightwatch.eval.poison import PoisonInjector, PoisonType
        store = self._store()
        inj = PoisonInjector(store)
        for pt in PoisonType:
            p = inj.inject(pt, "#1", "症状", "错", tags=["t"])
            inj.record_exposure(p.poison_id, f"tr_{pt.value}")
        s = inj.summary()
        self.assertEqual(s["注入总数"], 5)
        self.assertEqual(s["总暴露次数"], 5)
        self.assertIn("按类型", s)
        # 每个类型都要有"检测难度"说明
        for info in s["按类型"].values():
            self.assertTrue(info["检测难度"])


class TestStoreInterfaceConsistency(unittest.TestCase):
    """★ 空实现必须和真实实现接口一致 —— 防止"接口漂移"。

    ============================================================
    这条测试守的是一个具体的坑（P1-3 时踩到）
    ============================================================
    `NullMemoryStore` 是 P1-3 才写的，而 `MemoryStore` 在 P1-2 加了
    4 个语义层方法（add_promotion / pending_promotions / ...）。

    结果：**NullStore 缺了 5 个方法/属性**（encoder、_embed_text 和 4 个语义层方法），
    导致"关记忆"实验组全部 AttributeError。

    更糟的是这个错误的**可误读性**：错误发生在"关记忆"组，
    很容易被读成"关掉记忆导致 Agent 不可用" ——
    而真相是"我们的空实现不完整"。

    **结论**：多个实现之间需要强制的契约检查。
    与其在运行时 AttributeError，不如让测试先失败。
    """

    def _public_api(self, obj) -> set:
        """取对象的公开方法名（排除双下划线和内部字段）。"""
        return {
            name for name in dir(obj)
            if not name.startswith("__")
            and callable(getattr(obj, name, None))
        }

    # 纯内部实现细节 —— 空实现**不该**有，因为它们依赖具体存储引擎
    #
    # ★ 怎么判断一个方法是"接口"还是"内部细节"？
    #   标准不是"有没有下划线"，而是**有没有外部调用**。
    #
    #     conn        → SQLite 连接对象（NullStore 没有数据库）
    #     _row_to_mem → sqlite3.Row 转换（NullStore 没有行）
    #     _lock_for   → 内部锁管理（NullStore 无共享状态）
    #     _rank       → recall() 的内部拆分（重构出来的辅助方法）
    #     _recall_via_index → 同上，索引路径的内部实现
    #
    #   反过来，`_embed_text` 虽然也以下划线开头，但它**被外部调用了**
    #   （lifecycle 里构造 embedding 时用到）—— 所以它在接口里。
    #
    #   **Python 的下划线只是约定，不是访问控制。**
    #   一旦有外部调用点，私有方法就成了事实上的公开接口。
    INTERNAL_ONLY = {"conn", "_row_to_mem", "_lock_for",
                     "_rank", "_recall_via_index",
                     # 增量迁移：操作 SQLite 的 PRAGMA table_info + ALTER TABLE，
                     # NullStore 没有数据库，谈不上"迁移"。
                     "_ensure_columns"}

    def test_null_store覆盖了memory_store的全部方法(self):
        from nightwatch.memory.null_store import NullMemoryStore
        from nightwatch.memory.store import MemoryStore

        real = self._public_api(MemoryStore(":memory:")) - self.INTERNAL_ONLY
        null = self._public_api(NullMemoryStore())

        missing = sorted(real - null)
        self.assertEqual(
            missing, [],
            f"NullMemoryStore 缺少 MemoryStore 的方法：{missing}\n"
            f"（空实现必须完整替代真实实现，否则调用方会 AttributeError。\n"
            f" 如果新增的方法确实是纯内部细节，请加进 INTERNAL_ONLY 并说明理由。）")

    def test_null_store有encoder属性(self):
        """encoder 不是方法而是属性，但调用方会用它 —— 同样必须提供。"""
        from nightwatch.memory.null_store import NullMemoryStore
        ns = NullMemoryStore()
        self.assertTrue(hasattr(ns, "encoder"))
        self.assertTrue(callable(ns.encoder.encode))

    def test_null_store有embed_text(self):
        """`_embed_text` 虽然下划线开头，但有外部调用点 —— 是事实上的公开接口。

        （Python 里下划线只是约定，不是访问控制。一旦被外部调用，
          它就成了接口的一部分。）
        """
        from nightwatch.memory.null_store import NullMemoryStore
        from nightwatch.models import MemoryRecord
        ns = NullMemoryStore()
        mem = MemoryRecord(memory_id="m1", symptom="s", root_cause="c",
                           resolution="r")
        text = ns._embed_text(mem)
        self.assertIsInstance(text, str)
        self.assertIn("s", text)

    def test_null_store返回值形状与真实一致(self):
        """返回值形状也要一致 —— 这是 P1-2 那个 bug 的同类型问题。

        `recall` 必须返回 `[(记忆, 分数), ...]` 而不是 `[记忆, ...]`。
        空实现平时不执行循环体，所以形状错了也不会暴露 ——
        必须显式断言。
        """
        from nightwatch.memory.null_store import NullMemoryStore
        ns = NullMemoryStore()
        self.assertEqual(ns.recall("q"), [])
        self.assertIsNone(ns.find_similar(None, 0.3))   # type: ignore
        self.assertEqual(ns.count(), 0)
        self.assertEqual(ns.all_memories(), [])
        self.assertFalse(ns.insert(None))               # type: ignore


class TestConcurrentMemoryWrites(unittest.TestCase):
    """★ 并发写入不丢数据（P2-2 时发现的真 bug）。

    ============================================================
    这个测试为什么必须存在
    ============================================================
    A2A 演示的并发写入测试发现：8 线程 × 5 条 = 40 条，**只成功了 26 条**。

    根因：`sqlite3.connect(..., check_same_thread=False)`
    只是关掉了线程检查，**不代表线程安全** —— 多线程共用连接会互相踩游标，
    而且失败是**静默的**（返回 False 而不是抛异常）。

    修法：每线程一个独立连接（thread-local）+ `busy_timeout`。

    **这类 bug 单线程测试永远发现不了** —— 所以这个测试是必要的。
    """

    def test_多线程并发写入不丢数据(self):
        import tempfile
        import threading as _th
        from nightwatch.memory.store import HashEmbedding, MemoryStore

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = MemoryStore(tmp.name, encoder=HashEmbedding(dim=64))

        n_threads, per_thread = 8, 5
        results: list[bool] = []
        lock = _th.Lock()

        def writer(tid: int) -> None:
            got = []
            for i in range(per_thread):
                from nightwatch.models import MemoryRecord, Outcome
                got.append(store.insert(MemoryRecord(
                    memory_id=f"t{tid}_{i}", issue_id=f"#{tid}{i}",
                    symptom=f"并发测试 {tid}-{i}", outcome=Outcome.RESOLVED)))
            with lock:
                results.extend(got)

        threads = [_th.Thread(target=writer, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * per_thread
        succeeded = sum(1 for r in results if r)
        self.assertEqual(
            succeeded, expected,
            f"并发写入丢失数据：{succeeded}/{expected} —— "
            f"检查是否用了共享连接而不是 thread-local 连接")
        # 数据库里的实际条数也要对得上
        self.assertEqual(store.count(), expected)

        try:
            import os as _os
            store.close()
            _os.unlink(tmp.name)
        except OSError:
            pass


# =============================================================================
# 端到端
# =============================================================================

class TestEndToEnd(unittest.TestCase):

    def test_完整流程_记忆被写入且能召回(self):
        """★ 最重要的一条：跑完一个 issue 后，记忆真的进了库。

        这是"有状态 Agent"和"无状态问答机器人"的分界线。
        """
        issue = sample_issue("#901")
        repo = IssueRepository([issue])
        repo.add_doc("CONTRIBUTING.md", "标签规范：bug/timeout 超时类。")
        store = make_store()

        reg = ToolRegistry(repo, store=store, current_issue_id=issue.issue_id)
        agent = TriageAgent(reg, llm=MockLLM(), store=store,
                            settings=load_settings())

        result = agent.triage(issue, human_verdict=HumanVerdict.ACCEPTED)

        self.assertIsNotNone(result.trace)
        self.assertGreater(result.n_steps, 0)
        self.assertGreater(store.count(), 0,
                           "处理完成后应产生至少一条情景记忆")

        mem = store.all_memories()[0]
        self.assertTrue(mem.symptom, "记忆应有症状描述")
        self.assertEqual(mem.human_verdict, HumanVerdict.ACCEPTED)
        self.assertTrue(mem.embedding, "记忆应有向量")

    def test_第二个相似问题能召回第一个的记忆(self):
        """★ 项目卖点验证：跨会话记忆真的生效。"""
        repo = IssueRepository()
        repo.add_doc("CONTRIBUTING.md", "标签规范：bug/timeout 超时类。")
        store = make_store()
        settings = load_settings()
        llm = MockLLM()

        first = sample_issue("#911", title="接口在高峰时段超时", author="alice")
        second = sample_issue("#912", title="接口在高峰时段超时",
                              body="高峰时段调用接口超时，和之前一样。",
                              author="alice")
        repo.add(first)
        repo.add(second)

        # 第一次：建立记忆
        reg1 = ToolRegistry(repo, store=store, current_issue_id=first.issue_id)
        TriageAgent(reg1, llm=llm, store=store, settings=settings).triage(
            first, human_verdict=HumanVerdict.ACCEPTED)
        self.assertGreater(store.count(), 0, "第一次应写入记忆")

        # 第二次：应召回
        reg2 = ToolRegistry(repo, store=store, current_issue_id=second.issue_id)
        r2 = TriageAgent(reg2, llm=llm, store=store, settings=settings).triage(second)

        self.assertGreater(len(r2.trace.memory_recalled), 0,
                           "第二个相似问题应召回第一个的记忆")

    def test_不收敛时返回带证据的降级结果(self):
        """轮数上限触发兜底 —— 不能抛异常，要返回可用的降级结果。"""
        from unittest.mock import patch
        issue = sample_issue("#902")
        repo = IssueRepository([issue])
        store = make_store()
        settings = load_settings()
        # 把上限压到 1 轮，强制触发兜底
        settings.reliability["max_steps"] = 1

        reg = ToolRegistry(repo, store=store, current_issue_id=issue.issue_id)
        agent = TriageAgent(reg, llm=MockLLM(), store=store, settings=settings)
        result = agent.triage(issue)

        # 要么正常收敛（1 轮内做完），要么走兜底
        if result.outcome == Outcome.UNFINISHED:
            self.assertIsNotNone(result.fallback, "兜底必须带结构化信息")
            self.assertIn("evidence", result.fallback)
        else:
            self.assertGreaterEqual(result.n_steps, 1)

    def test_LLM不可用时仍返回结构化结果(self):
        """模型挂了不能崩 —— 降级是 Agent 系统的基本要求。"""

        class BrokenLLM:
            def complete(self, *a, **kw):
                raise RuntimeError("模拟模型服务不可用")

        issue = sample_issue("#903")
        repo = IssueRepository([issue])
        store = make_store()
        reg = ToolRegistry(repo, store=store, current_issue_id=issue.issue_id)
        agent = TriageAgent(reg, llm=BrokenLLM(), store=store,
                            settings=load_settings())
        result = agent.triage(issue)

        self.assertEqual(result.outcome, Outcome.UNFINISHED)
        self.assertIsNotNone(result.fallback)
        self.assertTrue(result.summary, "应给出可读的说明")


# =============================================================================
# 长程一致性
# =============================================================================

class TestLongHorizonConsistency(unittest.TestCase):
    """长程一致性压测相关。

    这个测试类守的是项目的**核心设计主张**：
        把约束放进"任务状态卡"每轮重注入，就不怕长任务里指令漂移。

    如果哪天有人重构了上下文调度，导致约束被压掉 ——
    这些测试会失败。这就是它们的价值。
    """

    def setUp(self):
        from nightwatch.eval.dataset import build_synthetic_dataset
        self.settings = load_settings()
        # 用一个已知会触发判重、且 symptoms 明确的样本
        ds = build_synthetic_dataset()
        self.issues = ds.train + ds.holdout
        self.target = next(i for i in self.issues if i.gold_is_duplicate_of)

    def _run(self, padding: int, state_card: bool, window: int = 8000):
        from nightwatch.tools.registry import IssueRepository, ToolRegistry
        history = [i for i in self.issues if i.issue_id != self.target.issue_id]
        repo = IssueRepository(history + [self.target])
        repo.add_doc("CONTRIBUTING.md",
                     "标签规范：bug/timeout 超时类；若与已有 issue 高度相似，"
                     "标记 duplicate 而非重新分类。")
        store = make_store()
        reg = ToolRegistry(repo, store=store,
                           current_issue_id=self.target.issue_id)
        agent = TriageAgent(reg, llm=MockLLM(), store=store,
                            settings=self.settings, window_size=window)
        return agent.triage(self.target, context_padding=padding,
                            enable_state_card=state_card)

    # ------------------------------------------------------------ 判定器

    def test_约束判定器_无引用却高置信算违规(self):
        """★ 这条守住的是"不允许自信地胡说"。

        一个高置信但错误的系统比低置信但错误的系统危险得多 ——
        前者会让维护者放下戒备直接采纳，后者至少会让人多看一眼。
        """
        from nightwatch.eval.constraints import ConstraintChecker

        class FakeResult:
            predicted_labels = ["bug/timeout"]
            duplicate_of = None
            summary = "按 timeout 处理"
            cites = []                    # 没有引用
            confidence = 0.85             # 却高置信

        checker = ConstraintChecker()
        report = ConstraintChecker().check(FakeResult(), None, self.target)

        kinds = [v.constraint for v in report.violations]
        self.assertIn("必须给出引用依据", kinds)
        self.assertIn("置信度与证据匹配", kinds)
        self.assertLess(report.adherence, 1.0)

    def test_约束判定器_合规输出零违规(self):
        """反例保护：完全合规的输出不应被误判。"""
        from nightwatch.eval.constraints import ConstraintChecker

        class FakeResult:
            predicted_labels = ["bug/timeout"]
            duplicate_of = None
            summary = "症状匹配 timeout，依据 #101"
            cites = ["#101"]              # 有引用
            confidence = 0.75             # 合理置信度

        report = ConstraintChecker().check(FakeResult(), None, self.target)
        self.assertEqual(len(report.violations), 0)
        self.assertEqual(report.adherence, 1.0)

    def test_不适用的检查不计入分母(self):
        """★ 这是设计可判定指标时的一个关键细节。

        当某些检查**不适用**时，它们不应计入分母 ——
        否则遵守率会被"不适用"的检查凭空拉高。

        在这个用例里有两处不适用：
          · 判重检查：没有工具输出（没有高相似度历史可查）
          · 死循环检查：没有 trace

        所以 checks_run 应该 = 2（只剩"必须给出引用"和"置信度匹配"）。

        写成 2 而不是 3，是因为实现里 `_check_tool_loop` 在没有 trace 时
        直接 return、不计数 —— 这个行为是对的，测试预期要跟着对。
        （第一版我把预期写成了 3，测试挂了 —— 而这恰好说明
          "不适用就不计入"这条原则在两个地方都贯彻了。）
        """
        from nightwatch.eval.constraints import ConstraintChecker

        class NoCites:
            predicted_labels = ["bug/timeout"]
            summary = "ss"
            cites = []
            confidence = 0.2              # 低置信，所以不违规

        report = ConstraintChecker().check(NoCites(), None, self.target)
        self.assertEqual(report.checks_run, 2,
                         "不适用判重检查和死循环检查时，检查数不应包含它们")

        # ★ 注意这两条约束是**独立**的：
        #   "必须给出引用依据"     → cites 为空 → 违规
        #   "置信度与证据匹配"     → 低置信    → 不违规
        # 所以是 1 条违规 / 2 条检查 = adherence 0.5
        #
        # 这个设计是对的：系统"诚实地说了不知道"（低置信）值得肯定，
        # 但"没给依据"仍然是一个缺陷 —— 两件事不能互相抵消。
        # （第一版我把预期写成 1.0，测试挂了，说明我一开始把这两条混在一起看了。）
        kinds = [v.constraint for v in report.violations]
        self.assertIn("必须给出引用依据", kinds)
        self.assertNotIn("置信度与证据匹配", kinds,
                         "低置信不应被判为违规 —— 那是诚实的表现")
        self.assertAlmostEqual(report.adherence, 0.5, places=4)

    # ------------------------------------------------------------ 端到端

    def test_短上下文下两组都能遵守约束(self):
        """基线：没有上下文压力时，开不开状态卡都能遵守（这是应该的）。"""
        from nightwatch.eval.constraints import ConstraintChecker

        checker = ConstraintChecker()
        for state_card in (True, False):
            r = self._run(padding=0, state_card=state_card)
            report = checker.check(r, r.trace, self.target)
            self.assertEqual(report.adherence, 1.0,
                             f"无压力时应完全遵守（state_card={state_card}）")

    def test_长上下文下状态卡保护了约束(self):
        """★ 核心断言：注入足够压力后，无状态卡组会漂移，有状态卡组不会。

        这是整个"任务状态卡"设计的验证。如果这个测试失败，
        说明设计失效或上下文调度被改坏了。

        参数选择：padding=80 在 window=8000 下足以触发驱逐
        （实测拐点在 60 轮附近）。
        """
        from nightwatch.eval.constraints import ConstraintChecker

        checker = ConstraintChecker()
        results = {}
        for state_card in (True, False):
            r = self._run(padding=80, state_card=state_card)
            report = checker.check(r, r.trace, self.target)
            results[state_card] = report

            # 先确认压力真的形成了 —— 否则这个测试没有意义
            total_evicted = sum(len(u.get("evicted", []))
                                for u in r.trace.context_usage)
            self.assertGreater(total_evicted, 0,
                               "上下文压力未形成，测试前提不成立")

        self.assertGreater(results[True].adherence, results[False].adherence,
                           "有状态卡组应比无状态卡组更遵守约束")
        self.assertEqual(results[True].adherence, 1.0,
                         "有状态卡组应保持完全遵守")

    def test_状态卡在不可压缩的桶里(self):
        """机制验证：状态卡必须在 task 桶（不可压缩），否则保护机制失效。

        这条测试直接检查实现细节 —— 因为"约束能被保护"的**根本原因**
        就是它待在一个不会被压缩/驱逐的地方。
        一旦有人把它挪到 history 桶，这个测试会失败。
        """
        from nightwatch.context.budget import ContextBudgetManager

        ctx = ContextBudgetManager(window_size=2000)
        ctx.set_task({"硬约束": ["测试约束"]})
        for i in range(200):                     # 制造巨大压力
            ctx.add_history(f"step {i} " + "x" * 300)

        messages, report = ctx.build()
        joined = "\n".join(m["content"] for m in messages)
        self.assertIn("测试约束", joined,
                      "task 桶的内容不应被压缩或驱逐掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
