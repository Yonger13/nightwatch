#!/usr/bin/env python3
"""
Nightwatch · 一键演示

================================================================================
这个脚本演示什么
================================================================================
它要让你在 10 秒内看到这个项目**最核心的那个卖点**：

    同一个作者报了两次类似的问题，
    第二次处理时，系统记得第一次是怎么解决的 —— 更快、更准。

这是"长程记忆"从概念落到可观察现象的证明。没有这个对比，
"我实现了记忆系统"只是一句话；有了它，是一个能被看见的行为差异。

================================================================================
怎么跑
================================================================================
    cd "D:/agent program"
    python scripts/run_demo.py

不需要 API key、不需要数据库、不需要网络。
（会自动降级到 SQLite + MockLLM，见 src/nightwatch/llm/client.py 的说明）
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import 到 src 下的包（src layout 的标准处理）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import load_settings                      # noqa: E402
from nightwatch.eval.dataset import build_synthetic_dataset      # noqa: E402
from nightwatch.llm.client import build_llm                       # noqa: E402
from nightwatch.loop.react import TriageAgent                     # noqa: E402
from nightwatch.memory.store import MemoryStore                   # noqa: E402
from nightwatch.models import HumanVerdict, Issue                 # noqa: E402
from nightwatch.tools.registry import IssueRepository, ToolRegistry  # noqa: E402


# =============================================================================
# 输出美化（自己实现，不引入依赖）
# =============================================================================

class C:
    """终端颜色。不引入 rich/colorama，因为要零依赖。"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"

    @classmethod
    def disable(cls):
        for name in dir(cls):
            if name.isupper():
                setattr(cls, name, "")


def hr(title: str = "", width: int = 76) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{C.BOLD}{'─' * 2} {title} {'─' * pad}{C.RESET}")
    else:
        print(f"{C.DIM}{'─' * width}{C.RESET}")


def kv(k: str, v, color: str = "") -> None:
    print(f"  {C.DIM}{k:<22}{C.RESET}{color}{v}{C.RESET}")


# =============================================================================
# 演示
# =============================================================================

def demo_memory_effect() -> None:
    """核心演示：记忆对第二次处理的影响。

    ============================================================
    实验设计说明（这段本身也值得学）
    ============================================================
    这是一个**受控对比**，不是"跑两遍看看"：

      变量：记忆库是否已包含前一条相关经验
      控制：同一个 issue 文本的两种表述、同一模型配置、同一工具集

    为什么不用"同一个 issue 跑两次"？
    因为那会被记忆系统识别为完全相同的记忆（幂等键冲突），
    落到"跳过"路径，看不出效果。所以要**换一个措辞但语义相同的问题** ——
    这样才是在测"能不能泛化",而不是在测"能不能查到一模一样的东西"。
    """
    print(f"\n{C.BOLD}{C.CYAN}Nightwatch · 记忆效果演示{C.RESET}")
    print(f"{C.DIM}目标：展示『同一个提交者的第二个相似问题，处理得更快更准』{C.RESET}")

    settings = load_settings()
    llm = build_llm(use_mock=True)

    # --- 准备数据 ---
    # 注意这里的两个 issue 是**语义相似但表述不同**的
    first_issue = Issue(
        issue_id="#201",
        title="高峰时段调用报表接口超时",
        body=("我们的服务在每天上午 10 点调用报表接口经常返回 504，"
              "非高峰时段是正常的。已经持续一周了。"),
        author="alice",
        repo="demo/repo",
        stack_trace=("requests.exceptions.Timeout: HTTPSConnectionPool"
                     "(host='api.example.com', port=443): Read timed out."),
    )
    second_issue = Issue(
        issue_id="#202",
        title="批量导出大数据量时接口无响应",
        body=("导出超过 5000 条记录时接口会卡住最后超时，"
              "1000 条左右是能成功的。每天早上做数据同步时会遇到。"),
        author="alice",                    # ★ 同一个提交者
        repo="demo/repo",
        stack_trace=("requests.exceptions.ReadTimeout: Read timed out."
                     " (read timeout=30)"),
    )

    store = MemoryStore(":memory:")
    repo = IssueRepository([first_issue, second_issue])
    repo.add_doc("CONTRIBUTING.md",
                 "标签规范：bug/timeout 表示超时类问题；bug/permission 表示权限类；"
                 "bug/config 表示配置类；enhancement 表示功能请求。"
                 "若发现与已有 issue 高度相似，应标记为 duplicate 而不是重新分类。")

    def run_one(issue: Issue, label: str) -> dict:
        registry = ToolRegistry(repo, store=store)
        agent = TriageAgent(registry, llm=llm, store=store, settings=settings)

        hr(label)
        kv("issue", f"{issue.issue_id} {issue.title}")
        kv("提交者", issue.author)

        result = agent.triage(issue)

        kv("步数", result.n_steps, C.YELLOW)
        kv("耗时", f"{result.elapsed_ms} ms")
        kv("预测标签", result.predicted_labels or "(无)", C.GREEN)
        kv("结论", result.outcome.value,
           C.GREEN if result.outcome.value == "resolved" else C.YELLOW)

        # 显示上下文占用 —— 用于观察记忆注入对上下文的影响
        if result.trace and result.trace.context_usage:
            last = result.trace.context_usage[-1]
            kv("上下文占用", f"{last['utilization'] * 100:.1f}%  "
                             f"(工具 {last['usage'].get('tools', 0)} tk / "
                             f"记忆 {last['usage'].get('memory', 0)} tk)")

        if result.trace and result.trace.memory_recalled:
            kv("召回记忆", f"{len(result.trace.memory_recalled)} 条")

        print(f"\n  {C.DIM}处理建议：{C.RESET}{result.summary[:140]}")

        return {
            "steps": result.n_steps,
            "tokens": result.trace.total_tokens if result.trace else 0,
            "recalled": len(result.trace.memory_recalled) if result.trace else 0,
            "outcome": result.outcome.value,
        }

    # --- 第一次：记忆库为空 ---
    r1 = run_one(first_issue, "第 1 次处理 · 记忆库为空")

    # 模拟维护者采纳（这一步很重要：人工反馈会进入记忆，提升它的权重）
    print(f"\n  {C.DIM}→ 模拟维护者『采纳』该建议（人工反馈将写入记忆）{C.RESET}")
    _reconsolidate_with_feedback(first_issue, store, llm, settings,
                                 HumanVerdict.ACCEPTED)

    # 人工确认后，更新记忆库状态
    print(f"  {C.DIM}→ 记忆库现有 {store.count()} 条情景记忆{C.RESET}")

    # --- 第二次：记忆库里有前一条经验 ---
    r2 = run_one(second_issue, "第 2 次处理 · 记忆库已有前一条经验")

    # --- 对比 ---
    hr("对比结果")
    print(f"""
  {'指标':<18}{'第 1 次':>12}{'第 2 次':>12}{'变化':>12}""")
    print(f"  {'─' * 54}")

    def delta(a, b):
        if not a:
            return "—"
        d = (b - a) / a * 100
        color = C.GREEN if d < 0 else (C.RED if d > 0 else "")
        return f"{color}{d:+.0f}%{C.RESET}"

    print(f"  {'步数':<18}{r1['steps']:>12}{r2['steps']:>12}"
          f"{delta(r1['steps'], r2['steps']):>21}")
    print(f"  {'Token':<18}{r1['tokens']:>12}{r2['tokens']:>12}"
          f"{delta(r1['tokens'], r2['tokens']):>21}")
    print(f"  {'召回记忆条数':<18}{r1['recalled']:>12}{r2['recalled']:>12}")

    print(f"""
{C.DIM}  说明：
    · 第 2 次能召回第 1 次的经验，是因为两者『症状相似』且『同一提交者』。
    · 召回是两路混合的：按 author 精确匹配 + 按 symptom 语义相似 ——
      这是『同一作者报过什么』和『有没有人和他遇到一样的问题』两种意图的差别。
    · 第 2 次少走弯路，是因为**上下文里已经有可用的历史结论**，
      不需要再自己去检索一遍（注意它直接复用了记忆里的标签）。

  {C.YELLOW}但这不等于项目已经有效了 —— 三点必须说清楚：{C.RESET}
  {C.DIM}  ① 这里用的是**规则版 Mock 模型**，它的"少走弯路"是代码写死的分支，
        不是模型自己判断的结果。真实模型接入后才有真实意义。
      ② 这里用的是**合成数据**，症状描述比真实 issue 规整得多。
      ③ 所以这个演示证明的是"**链路通了**"，
        不是"**效果好了**"。效果必须用真实数据 + 真实模型跑评测才成立。

    {C.DIM}这个区分很重要：它能让你在面试时不被"你这数字是真的吗"问倒 ——
    因为你会主动先说明数据的边界。{C.RESET}""")

    # --- 记忆库体检 ---
    hr("记忆库状态")
    for mem in store.all_memories():
        kv("记忆 ID", mem.memory_id, C.CYAN)
        kv("  症状", mem.symptom[:60])
        kv("  根因", (mem.root_cause or "(未定位)")[:60])
        kv("  重要性", f"{mem.importance:.3f}")
        kv("  人工反馈", mem.human_verdict.value,
           C.GREEN if mem.human_verdict.value == "accepted" else "")
        ok_acts = [a["tool"] for a in mem.actions if a.get("ok")]
        fail_acts = [a["tool"] for a in mem.actions if not a.get("ok")]
        kv("  成功的动作", ", ".join(ok_acts) or "(无)")
        # ★ 这个字段是"值钱字段"之一，演示里特意展示出来
        kv("  失败的动作", ", ".join(fail_acts) or "(无)",
           C.RED if fail_acts else "")
        print()


def _reconsolidate_with_feedback(issue, store, llm, settings,
                                 verdict: HumanVerdict) -> None:
    """模拟"维护者采纳后，记忆被重新沉淀"。

    真实流程里，这一步是由外部事件触发的（比如 GitHub webhook 收到 label 事件）。
    演示里手动调一次，是为了让你看到 human_verdict 是怎么进记忆的 ——
    它是记忆质量的关键来源。
    """
    from nightwatch.memory.lifecycle import Extractor, Scorer, Writer
    from nightwatch.models import Outcome, Step, ToolCall, Trace

    # 构造一个"已完成的轨迹"（模拟真实处理过程）
    trace = Trace(trace_id="tr_demo_201", issue_id=issue.issue_id)
    for i, (tool, ok) in enumerate([("search_similar_issues", True),
                                    ("search_docs", True)], 1):
        trace.record(Step(
            step=i, thought=f"调用 {tool}",
            action=ToolCall(step=i, tool_name=tool, tool_input={"q": "timeout"},
                            ok=ok, output={"ok": True}),
            observation="ok",
        ))
    trace.outcome = Outcome.RESOLVED
    trace.confidence = 0.8
    trace.final_answer = (
        "症状：报表接口在高峰时段超时\n"
        "根因：客户侧并发过高触发限流（QPS 超过阈值）\n"
        "建议：提升该客户配额 + 客户端加指数退避重试"
    )
    trace.finished_at = "now"

    ex = Extractor(llm, settings.memory)
    sc = Scorer(settings.memory)
    wr = Writer(store, settings.memory)

    cand = ex.extract(trace, issue, verdict, human_edit="")
    if cand is None:
        print(f"  {C.YELLOW}(记忆抽取返回空，跳过){C.RESET}")
        return
    score = sc.score(cand)
    if sc.should_write(score):
        wr.commit(cand, issue, trace, verdict, human_edit="")
        print(f"  {C.GREEN}✓ 记忆已沉淀（重要性 {score:.3f} ≥ 阈值 "
              f"{settings.memory['write_threshold']}）{C.RESET}")
    else:
        print(f"  {C.YELLOW}✗ 未达写入阈值（{score:.3f} < "
              f"{settings.memory['write_threshold']}），被筛选掉{C.RESET}")


def demo_reliability() -> None:
    """演示四层可靠性：让 Agent 陷入循环，看它自己停下来。

    这个演示对应那道面试原题：
        "Agent 调了三个工具开始死循环，你的异常处理写在哪？"
    """
    hr("可靠性演示 · 循环检测")

    from nightwatch.loop.reliability import ReliabilityGuard

    guard = ReliabilityGuard()
    print(f"  {C.DIM}模拟连续调用同一个工具、参数完全相同：{C.RESET}\n")

    tool_name = "search_issues"
    args = {"keywords": "timeout", "limit": 10}

    for i in range(1, 7):
        blocked = guard.before_tool_call(tool_name, args)
        if blocked:
            print(f"  第 {i} 次: {C.RED}被阻止{C.RESET} — {blocked}")
            print(f"\n  {C.GREEN}✓ 循环被检测到，Agent 会被引导换策略"
                  f"（而不是继续烧 token）{C.RESET}")
            break
        print(f"  第 {i} 次: {C.GREEN}允许执行{C.RESET}")

    print(f"""
{C.DIM}  注意判据是『参数完全相同』而不是『工具名相同』：
    正常业务里连续调两次同一个工具是合理的（换个关键词再查一次），
    只有参数一模一样才是真的卡住了。

  这一步会产生误判吗？会。所以我们在 trace 里记录 loop_detected，
  人工复核时能发现误判 —— 承认检测器不完美，但要能量化它的不完美。{C.RESET}
""")


def demo_budget() -> None:
    """演示上下文预算调度：工具结果过大时自动卸载。"""
    hr("上下文预算演示 · 工具结果卸载")

    from nightwatch.context.budget import ContextBudgetManager, estimate_tokens

    ctx = ContextBudgetManager(window_size=8000)
    ctx.set_system("你是一个 issue 分类助手。" * 5)
    ctx.set_task({"目标": "分类", "硬约束": ["判重优先", "不得编造根因"]})

    # 构造一个超大的工具返回（模拟"读完一个几百行的日志文件"）
    # 为什么要造这么大？因为只有超过卸载阈值（见 config.TOOL_RESULT_OFFLOAD_THRESHOLD）
    # 才能看到卸载生效。真实场景里"读一个大日志文件"就是这个量级。
    big_result = {"logs": [f"2026-09-23 10:{i % 60:02d}:00 ERROR request failed, "
                           f"trace_id=abc{i}, upstream=api.example.com, "
                           f"retry={i % 3}, elapsed={100 + i}ms"
                           for i in range(900)]}
    raw_tokens = estimate_tokens(str(big_result))

    print(f"  {C.DIM}工具返回的原始体积：约 {raw_tokens} tokens{C.RESET}")
    offloaded = ctx.add_tool_result("read_code", big_result)

    if offloaded:
        print(f"  {C.GREEN}✓ 已自动卸载{C.RESET}（超过阈值 "
              f"{3000} tokens）")
        print(f"  {C.DIM}  上下文里只保留『摘要 + 文件指针』，"
              f"需要细节时可以读回来{C.RESET}")
    else:
        print(f"  {C.YELLOW}未触发卸载{C.RESET}")

    messages, report = ctx.build()
    print(f"\n  {C.DIM}调度后的上下文占用：{C.RESET}")
    for name, used in report.usage.items():
        quota = report.quota.get(name, 0)
        bar = "█" * int(used / max(quota, 1) * 20)
        print(f"    {name:<9} {used:>6} tk / {quota:>6} tk  {bar}")
    print(f"    {'总计':<9} {report.total_used:>6} tk  "
          f"(利用率 {report.utilization * 100:.1f}%，预留 "
          f"{int(ctx.window_size * ctx.reserve_ratio)} tk 给模型输出)")

    print(f"""
{C.DIM}  关键点：**卸载不是删除**。全文落盘了，留了指针，能读回来。
  这是『压缩不丢信息』的第一层兜底。{C.RESET}
""")


def main() -> None:
    if not sys.stdout.isatty():
        C.disable()
    demo_memory_effect()
    demo_reliability()
    demo_budget()

    hr("下一步")
    print(f"""
  1. 跑评测：      python scripts/run_eval.py
  2. 构造真实数据：python scripts/collect_issues.py --help
  3. 读设计文档：  docs/00-项目总览与学习地图.md

  {C.YELLOW}注意：以上用的是 Mock 模型和合成数据，指标仅供跑通验证。{C.RESET}
  {C.YELLOW}要写进简历的数字，必须来自真实项目 issue 队列 + 真实模型。{C.RESET}
""")


if __name__ == "__main__":
    main()
