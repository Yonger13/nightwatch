#!/usr/bin/env python3
"""
Nightwatch · 长程一致性压测

================================================================================
这个脚本验证一件具体的事
================================================================================
项目有一个核心设计主张：

    **把硬约束放进"任务状态卡"，每轮重新注入，就不怕长任务里指令漂移。**

如果不做实验，这句话永远只是个说法。这个脚本把它变成可验证的结论：

    对照组（关闭状态卡重注入）→ 约束遵守率应随上下文增长而**下降**
    实验组（开启状态卡重注入）→ 约束遵守率应**保持稳定**

如果两组没有差异，那说明这个设计**没有价值**，应该诚实地写进"已知问题"。

================================================================================
实验设计（三个关键选择）
================================================================================

**① 变量控制**

    自变量 1：上下文长度      —— 注入 0 / 5 / 15 / 40 轮前置工作
    自变量 2：状态卡重注入     —— 开 / 关
    因变量：  约束遵守率       —— 由 ConstraintChecker 规则判定

**② 用什么制造"长任务"**

不是让 Agent 真的跑很多轮（任务该几步收敛是任务本身决定的，
人为拉长会变成测别的东西），而是**注入 N 轮"前置工作记录"**。

而且注入的是**有意义的工作**，不是垃圾文本 ——
真实的长任务里，上下文被占满是由真实工作造成的。
注入垃圾会变成测"噪音干扰"，那是另一个问题。

**③ 为什么关掉状态卡就会漂移**

    开启：约束在 task 桶（不可压缩）→ 每轮重建 → 始终可见
    关闭：约束只在开头说一次，落在 history 桶 → 上下文压力下**被优先驱逐**
                                              （驱逐顺序从最早开始）

所以这个实验实际上是在测：**"约束是否可见"是否影响行为**。

================================================================================
⚠️ 这个实验能证明什么、不能证明什么
================================================================================
**能证明**：当约束从上下文中消失时，Agent 的行为会漂移；
          而任务状态卡能保证约束不消失。

**不能证明**：真实模型的漂移率是多少。
          因为这里用的是规则版 Mock —— 它"要么全遵守、要么全漂移"，
          而真实模型是概率性的、部分的遗忘。
          要测后者必须接真实模型（`--real-llm`）。

**这个区分必须写进报告。** 一个实验的边界和它的结论一样重要。

================================================================================
用法
================================================================================
    python scripts/run_longhorizon.py                 # 默认参数
    python scripts/run_longhorizon.py --runs 5        # 每格跑 5 次
    python scripts/run_longhorizon.py --real-llm      # 用真实模型（需 API key）
    python scripts/run_longhorizon.py --json out.json # 输出机器可读结果
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import load_settings                        # noqa: E402
from nightwatch.eval.constraints import (                          # noqa: E402
    ConstraintChecker, LongHorizonReport,
)
from nightwatch.eval.dataset import build_synthetic_dataset        # noqa: E402
from nightwatch.llm.client import build_llm                        # noqa: E402
from nightwatch.loop.react import TriageAgent                      # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore     # noqa: E402
from nightwatch.tools.registry import IssueRepository, ToolRegistry  # noqa: E402


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    RED = "\033[31m"; MAGENTA = "\033[35m"

    @classmethod
    def disable(cls):
        for n in dir(cls):
            if n.isupper():
                setattr(cls, n, "")


# =============================================================================
# 实验分组
# =============================================================================
# ★ 三组的设计由来（这是本项目第二个"实验设计缺陷"的修复）：
#
# 第一版只有两组（A / B）。在 Mock 上效果很好（差 66.67%），
# 但换到真实模型上几乎测不出差异 —— 两组都 100%。
#
# 根因：**system prompt 里本来就有一份约束**（"判断准则"5 条），
# 而 system 桶是**不可压缩**的。所以 B 组不是"没有约束"，
# 只是"少了一份每轮重申的强调"。
#
# 所以加了 C 组：真的把 system prompt 里的约束也拿掉，作为基线。
#
# 这三组能分离出两个贡献：
#     C→B 的差  = system prompt 的约束值多少
#     B→A 的差  = 任务状态卡的增量值多少
#
# **生产环境只用 A**，B 和 C 是实验工具。

GROUPS: dict[str, dict] = {
    "A · system有约束+状态卡": dict(enable_state_card=True,
                                    include_system_constraints=True),
    "B · system有约束+无状态卡": dict(enable_state_card=False,
                                      include_system_constraints=True),
    "C · system无约束+无状态卡": dict(enable_state_card=False,
                                      include_system_constraints=False),
}


def hr(title: str = "", width: int = 78) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{C.BOLD}── {title} {'─' * pad}{C.RESET}")
    else:
        print(f"{C.DIM}{'─' * width}{C.RESET}")


def color_for(v: float) -> str:
    if v >= 0.95:
        return C.GREEN
    if v >= 0.7:
        return C.YELLOW
    return C.RED


# =============================================================================
# 单个格子的实验
# =============================================================================

def run_cell(padding: int, group: str, group_kwargs: dict, runs: int,
             llm_factory, window_size: int, settings) -> list[dict]:
    """跑一组 (padding, 实验组) 组合，返回每次运行的结果。

    每一格独立构造环境 —— 避免格子之间互相污染
    （记忆库、工具账本都要是干净的）。
    """
    ds = build_synthetic_dataset()
    issues = ds.train + ds.holdout

    # 选一个**会触发判重**的样本，因为它能同时检验"判重优先"这条约束
    # （如果选一个没有相似历史的问题，"判重优先"检查就不适用，测不出差异）
    target = next(i for i in issues if i.gold_is_duplicate_of)
    history = [i for i in issues if i.issue_id != target.issue_id]

    checker = ConstraintChecker()
    rows = []

    for run_idx in range(runs):
        repo = IssueRepository(history + [target])
        repo.add_doc("CONTRIBUTING.md", (
            "标签规范：bug/timeout 超时类；bug/permission 权限类；"
            "bug/config 配置类；bug/dependency 依赖类；bug/crash 崩溃类；"
            "enhancement 功能请求。若与已有 issue 高度相似，标记 duplicate 而非重新分类。"))
        store = MemoryStore(":memory:", encoder=HashEmbedding(dim=256))
        reg = ToolRegistry(repo, store=store, current_issue_id=target.issue_id)

        agent = TriageAgent(reg, llm=llm_factory(), store=store,
                            settings=settings, window_size=window_size)

        result = agent.triage(target, context_padding=padding, **group_kwargs)

        report = checker.check(result, result.trace, target)

        # ---------------------------------------------------------------
        # 上下文压力证据：必须**累加所有轮**，不能只看最后一次
        # ---------------------------------------------------------------
        # 这是踩过的坑：最初取的是 `context_usage[-1]`（最后一次 build 的数据），
        # 结果显示"驱逐 0、压缩 0"，看起来像"上下文压力没形成"。
        #
        # 但真正的原因是：**信息丢失发生在第一次 build**（那时 history 最长），
        # 之后历史被压短了，后续几轮自然不再触发。
        # 只看最后一次，等于在灾难发生之后去现场，什么都看不到。
        #
        # 教训：**采集"是否发生过"的证据时，要看全过程，不能只看终态。**
        usage_list = result.trace.context_usage or []
        last = usage_list[-1] if usage_list else {}
        rows.append({
            "padding": padding,
            "group": group,
            "run": run_idx,
            "adherence": report.adherence,
            "checks": report.checks_run,
            "violations": [str(v) for v in report.violations],
            "n_steps": report.n_steps,
            "utilization": last.get("utilization", 0.0),
            "max_utilization": max((u.get("utilization", 0.0) for u in usage_list),
                                   default=0.0),
            # ★ 累加值 —— 这才是"压力是否形成"的证据
            "evicted": sum(len(u.get("evicted", [])) for u in usage_list),
            "compressed": sum(len(u.get("compressed", [])) for u in usage_list),
            "has_dup": bool(result.duplicate_of),
            "cites": len(result.cites),
            "confidence": result.confidence,
        })

    return rows


# =============================================================================
# 主流程
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Nightwatch 长程一致性压测")
    ap.add_argument("--paddings", default="0,10,30,60,100",
                    help="注入的前置工作轮数，逗号分隔（默认 0,10,30,60,100）")
    ap.add_argument("--runs", type=int, default=3, help="每格跑几次（默认 3）")
    ap.add_argument("--window", type=int, default=8000,
                    help="上下文窗口大小（故意设小，让压力真的形成；默认 8000）")
    ap.add_argument("--groups", type=int, default=3, choices=[2, 3],
                    help=("实验组数：3=完整三组对照（默认，推荐）；"
                          "2=只跑 A/B 两组（旧行为，会被 system prompt 干扰）"))
    ap.add_argument("--real-llm", action="store_true",
                    help="使用真实模型（需要 NW_LLM_API_KEY）")
    ap.add_argument("--json", help="输出机器可读结果")
    args = ap.parse_args()

    if not sys.stdout.isatty():
        C.disable()

    paddings = [int(x) for x in args.paddings.split(",") if x.strip()]
    settings = load_settings()
    use_mock = not args.real_llm
    llm_factory = (lambda: build_llm(use_mock=True)) if use_mock \
        else (lambda: build_llm(use_mock=False))

    print(f"\n{C.BOLD}{C.CYAN}Nightwatch · 长程一致性压测{C.RESET}")
    hr()
    print(f"  {C.DIM}自变量 1（上下文长度）  : {paddings} 轮前置工作")
    print(f"  自变量 2（状态卡重注入）: 开启 / 关闭  ← 这是核心设计主张的对照组")
    print(f"  因变量（约束遵守率）    : 规则判定，不依赖模型自评")
    print(f"  每格重复次数            : {args.runs}")
    print(f"  上下文窗口              : {args.window} tokens"
          f"（刻意设小，让压缩真的发生）")
    print(f"  模型                    : "
          f"{'MockLLM（规则版）' if use_mock else '真实模型'}{C.RESET}")

    if use_mock:
        print(f"""
{C.YELLOW}⚠ 本次使用规则版 Mock。{C.RESET}
{C.DIM}  它能证明"约束可见性决定行为"，但**不能**给出真实模型的漂移率。
  真实模型的漂移是概率性的、部分的，必须用 --real-llm 测。{C.RESET}""")

    # ---------------- 跑实验 ----------------
    group_names = list(GROUPS)[:args.groups]
    all_rows: list[dict] = []
    total = len(paddings) * len(group_names)
    done = 0

    for padding in paddings:
        for gname in group_names:
            done += 1
            print(f"\n  {C.DIM}[{done}/{total}] 轮数={padding:<3} {gname} ..."
                  f"{C.RESET}", end="", flush=True)
            rows = run_cell(padding, gname, GROUPS[gname], args.runs,
                            llm_factory, args.window, settings)
            all_rows.extend(rows)

            mean = statistics.mean(r["adherence"] for r in rows)
            ev = statistics.mean(r["evicted"] for r in rows)
            print(f" 遵守率 {color_for(mean)}{mean:.2%}{C.RESET}"
                  f"  驱逐 {ev:.0f}")

    # ---------------- 明细 ----------------
    hr("逐次明细")
    print(f"  {'轮数':>5}{'组':>4}{'run':>5}{'遵守率':>9}{'检查':>6}"
          f"{'步数':>6}{'占用':>8}{'驱逐':>6}{'判重':>6}{'引用':>6}{'置信':>7}")
    print(f"  {'-' * 78}")
    for r in all_rows:
        gtag = (r.get("group") or "?")[0]
        print(f"  {r['padding']:>5}{gtag:>4}"
              f"{r['run']:>5}{color_for(r['adherence'])}{r['adherence']:>9.2%}{C.RESET}"
              f"{r['checks']:>6}{r['n_steps']:>6}{r['max_utilization']:>8.1%}"
              f"{r['evicted']:>6}"
              f"{'✓' if r['has_dup'] else '✗':>6}{r['cites']:>6}"
              f"{r['confidence']:>7.2f}")

    # 违规明细（只打前 6 条，避免刷屏）
    violations = [v for r in all_rows for v in r["violations"]]
    if violations:
        hr(f"违规明细（共 {len(violations)} 条，显示前 6 条）")
        for v in violations[:6]:
            print(f"  {C.RED}·{C.RESET} {v}")

    # ---------------- 汇总对照表 ----------------
    summary = LongHorizonReport.summarize(all_rows, group_names)
    hr("★ 汇总对照表（核心结论）")

    col_w = 14
    print("\n  " + f"{'轮数':>5}" +
          "".join(f"{g[:11]:>{col_w}}" for g in group_names))
    print("  " + "-" * (5 + col_w * len(group_names)))
    for row in summary["表"]:
        line = f"  {row['轮数']:>5}"
        for g in group_names:
            v = row.get(g)
            if v is None:
                line += f"{'—':>{col_w}}"
            else:
                line += f"{color_for(v)}{v:>{col_w - 1}.2%}{C.RESET} "
        print(line)

    # 组间差异（最大压力下的对比）
    if len(group_names) >= 2 and summary["表"]:
        last_row = summary["表"][-1]
        print(f"\n  {C.BOLD}组间差异（最大压力 {last_row['轮数']} 轮时）{C.RESET}")
        base_g = group_names[-1]
        base_v = last_row.get(base_g)
        for g in group_names:
            if g == base_g:
                continue
            v = last_row.get(g)
            if v is None or base_v is None:
                continue
            d = v - base_v
            mark = f"  {C.GREEN}← 有效{C.RESET}" if d > 0.05 else ""
            print(f"    {g[:22]:<24} 比 {base_g[0]} 组 "
                  f"{C.GREEN if d > 0.05 else ''}{d:+.2%}{C.RESET}{mark}")

    # ---------------- 趋势判断 ----------------
    hr("趋势判断")
    for gname, t in summary["趋势"].items():
        color = C.GREEN if t["结论"] == "稳定" else C.RED
        print(f"  {gname[:24]:<26} 首 {t['首']:>7.2%} → 末 {t['末']:>7.2%}  "
              f"（{t['变化']:+.2%}）  {color}{t['结论']}{C.RESET}")

    print(f"\n  {C.DIM}{summary['口径']['指标']}")
    print(f"  {summary['口径']['检查数说明']}")
    print(f"  {summary['口径']['局限']}{C.RESET}")

    # ---------------- 结论 ----------------
    # ---------------- 实验有效性检查 ----------------
    # ============================================================
    # 这一步很重要，而且必须**先于结论**做
    # ============================================================
    # 第一次跑这个实验时，所有组的"驱逐/压缩"都是 0，遵守率全是 100%，
    # 看起来像是"状态卡没用"。但真正的原因是**实验没测到该测的东西** ——
    # 上下文压力不足，约束（history 的第 0 条）根本没被动过。
    #
    # 如果当时直接下结论"两组没差异"，那就是**用一个无效实验得出一个错误结论**。
    #
    # 所以这里加了一道前置检查：**如果信息丢失从未发生，实验结论不予采信。**
    # 所有对照实验都该有这个步骤 ——
    # 先确认"干预确实生效了"，再看"效果差异"。
    total_evicted = sum(r["evicted"] for r in all_rows)
    total_compressed = sum(r["compressed"] for r in all_rows)
    max_util = max((r["max_utilization"] for r in all_rows), default=0)

    if total_evicted == 0 and total_compressed == 0:
        hr("⚠ 实验有效性检查未通过")
        print(f"""
  {C.RED}本次实验的结论不予采信。{C.RESET}

  观察到的现象：
    · 累计驱逐次数 = {total_evicted}，累计压缩次数 = {total_compressed}
    · 最高上下文占用率 = {max_util:.1%}

  这意味着**上下文压力没有真正形成**，约束从未被动过 ——
  所以两组都遵守约束，不是因为"状态卡有效"，而是因为
  **这个实验根本没制造出约束会被丢掉的情况**。

  {C.YELLOW}怎么修（按成本排序）：{C.RESET}
    ① 加大 --paddings（推荐，最直接）
       python scripts/run_longhorizon.py --paddings 0,30,60,120,200
    ② 减小 --window
       python scripts/run_longhorizon.py --window 4000""")
    else:
        hr("✓ 实验有效性检查通过")
        print(f"  累计驱逐 {total_evicted} 次，累计压缩 {total_compressed} 次，"
              f"最高占用率 {max_util:.1%}")
        print(f"  {C.DIM}→ 上下文压力已形成，说明约束确实面临过被丢弃的风险{C.RESET}")

    hr("结论")
    tm = summary["趋势"]
    last_row = summary["表"][-1] if summary["表"] else {}
    max_p = last_row.get("轮数", "?")

    strongest = group_names[0]        # A 组：完整设计
    weakest = group_names[-1]         # 最后一组：保护最少
    a_v, w_v = last_row.get(strongest), last_row.get(weakest)
    a_stable = abs(tm.get(strongest, {}).get("变化", 0)) < 0.05
    w_stable = abs(tm.get(weakest, {}).get("变化", 0)) < 0.05
    mock_note = "（本次用的是 Mock）" if use_mock else "（本次用的是真实模型）"

    # 找最弱组的拐点
    # ★ 只在"它一开始是好的"的情况下才找 —— 否则像 C 组这种
    #   从第 0 轮就漂移的，"拐点"没有意义（它没有可失去的东西）。
    first_row = summary["表"][0] if summary["表"] else {}
    bp = None
    if (first_row.get(weakest) or 0) >= 0.95:
        for row in summary["表"]:
            v = row.get(weakest)
            if v is not None and v < 0.95:
                bp = row["轮数"]
                break

    print()
    if a_v is not None and w_v is not None and (a_v - w_v) > 0.05:
        bp_text = (f"\n    最弱组的拐点在 {bp} 轮附近 —— 在这之前约束还能存活，"
                   f"之后就被丢掉了。" if bp is not None else "")
        print(f"""  {C.GREEN}✓ 保护机制有效。{C.RESET}

    在最大压力（{max_p} 轮）下：
      {strongest:<26} 遵守率 {a_v:>7.2%}
      {weakest:<26} 遵守率 {w_v:>7.2%}
      差异 {a_v - w_v:+.2%}{bp_text}

  {C.BOLD}逐层贡献分解（这才是三组对照的价值）：{C.RESET}""")
        for i in range(len(group_names) - 1):
            lo, hi = group_names[i + 1], group_names[i]
            lv, hv = last_row.get(lo), last_row.get(hi)
            if lv is None or hv is None:
                continue
            gain = hv - lv
            color = C.GREEN if gain > 0.05 else C.DIM
            print(f"    {lo[0]} → {hi[0]}：{color}{gain:+.2%}{C.RESET}"
                  f"   （加上「{hi.split('·')[-1].strip()[:20]}」的贡献）")
        print(f"""
  {C.YELLOW}⚠ 结论边界（必须一起说）{C.RESET}
  {C.DIM}  · 每格样本量 = {args.runs}，波动较大，趋势仅供方向参考
    · 约束判定是规则化近似（只查「有没有」，不查「好不好」）
    · Mock 与真实模型的漂移模式不同，两者都要报告{mock_note}{C.RESET}""")

    elif a_stable and w_stable:
        print(f"""  {C.YELLOW}⚠ 没有测出差异 —— 保护最少的组在所有压力下也保持 100%。{C.RESET}

    {C.DIM}可能的原因（按可能性排序）：
      ① **保护机制不止一处。** 除了被关掉的这两项，system prompt / 工具描述里
         可能还有约束的副本 → 逐项检查是否有遗漏的约束来源。
         （这个坑本项目踩过：第一版实验只改了状态卡，忘了 system prompt 里也有，
           于是两组都 100%，看起来像「设计没用」。）
      ② 上下文压力不足 → 看"上下文压力证据"表里的累计驱逐是否为 0
      ③ 这个任务太简单，约束本来就容易被满足 —— 这也是有价值的结论

    ★ 第 ① 条是最容易犯、也最容易误判的：**同一个约束在系统里有多份副本**，
      改掉一份不改变行为，而你以为是"设计没用"。{C.RESET}""")

    else:
        print(f"""  {C.YELLOW}⚠ 结果不符合预期，需要进一步分析。{C.RESET}

    {C.DIM}按顺序检查三件事：
      1. 上下文压力是否形成（"累计驱逐"列是否为 0）
      2. 每格样本量是否足够（当前每格 {args.runs} 次，建议 ≥3）
      3. 各组之间是否真的只差一个变量（逐项 diff 两组的输入）{C.RESET}""")

    # ---------------- 上下文压力证据 ----------------
    hr("上下文压力证据")
    print(f"\n  {'轮数':>5}{'组':>4}{'峰值占用':>11}{'累计驱逐':>11}{'累计压缩':>11}")
    print(f"  {'-' * 44}")
    from collections import defaultdict
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        buckets[(r["padding"], r.get("group", "?"))].append(r)
    for (p, g), rs in sorted(buckets.items()):
        util = max(x["max_utilization"] for x in rs)
        ev = statistics.mean(x["evicted"] for x in rs)
        cm = statistics.mean(x["compressed"] for x in rs)
        print(f"  {p:>5}{g[0]:>4}{util:>11.1%}{ev:>11.0f}{cm:>11.0f}")
    print(f"""
  {C.DIM}「累计驱逐 / 累计压缩」非零，才说明上下文压力真的造成了信息丢失 ——
  这是"约束为什么会漂移"的直接证据。

  ★ 注意这两个数字是**累加所有轮**的，不是只看最后一轮。
  因为信息丢失发生在**第一次 build**（那时 history 最长），
  之后历史被压短了，后续几轮自然不再触发 ——
  只看最后一轮，等于在灾难发生之后去现场，什么都看不到。{C.RESET}""")

    # ---------------- 输出 JSON ----------------
    if args.json:
        payload = {
            "meta": {
                "paddings": paddings,
                "runs_per_cell": args.runs,
                "window_size": args.window,
                "model": "MockLLM" if use_mock else "real",
                "warning": ("Mock 模型：只能证明'约束可见性决定行为'，"
                            "不能给出真实模型漂移率" if use_mock else ""),
            },
            "summary": summary,
            "rows": all_rows,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  {C.GREEN}结果已写入 {args.json}{C.RESET}")

    hr("下一步")
    print(f"""
  1. 用真实模型重跑，看漂移是否更微妙（部分遵守而非全有全无）：
       NW_LLM_API_KEY=xxx python scripts/run_longhorizon.py --real-llm --runs 5

  2. 把结论写进简历的表达方式（注意带上边界）：
       「长程一致性压测：注入 40 轮前置工作后，
         无状态卡组的约束遵守率下降 X%，有状态卡组保持 Y%」

  3. 如果要做更严谨的版本，把"轮数"和"token 占用率"作为两个自变量分别实验 ——
     才能分清是"轮数的错"还是"长度的错"。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
