#!/usr/bin/env python3
"""
Nightwatch · 组件组合实验（P1-3）

    # 快速验证流程（Mock，秒级）
    python scripts/run_combo_experiment.py --mock --design l9 --limit 5

    # 真实模型跑 L9 正交实验（9 组 × 样本数）
    python scripts/run_combo_experiment.py --real-llm --design l9 --limit 5

    # 全因子（27 组，用于验证交互效应）
    python scripts/run_combo_experiment.py --real-llm --design full --limit 3

==============================================================================
为什么叫"组合实验"而不是"多变量实验"
==============================================================================
多变量实验是"同时改多个变量"，但组合实验的重点在**交互效应**：

    假设我们发现：
      单独测 → 强模型更好；单独测 → 开记忆更好
      那"强模型 + 开记忆"是不是最好？

    **不一定。** 因为强模型可能已经能从上下文推断出答案，
    记忆召回的噪声反而干扰它；而弱模型需要记忆补信息。

    **两个单独看都更好的选项，组合起来可能更差。**

这种"1+1 < 2"的现象就是负交互 —— 它只能通过
**让因子同时变化**才能发现。这是本实验的核心价值。

==============================================================================
实验设计（3 因子 × 3 水平）
==============================================================================
    因子 1  模型档次      small(turbo) / medium(plus) / strong(max)
    因子 2  记忆策略      关 / 只情景 / 情景+语义
    因子 3  思考链策略    全开 / 分档 / 全关

全因子 = 27 组；L9 正交 = 9 组（可估主效应，估不了交互）。

==============================================================================
成本怎么算
==============================================================================
用 **token 总数**作为成本代理，而不是钱。理由：

  · 不同账号单价不同，写死金额会误导
  · token 是**与供应商无关**的客观量
  · 真要换算成钱，读者可以用自己的单价乘

同时记录**耗时**作为第二个成本维度 ——
因为延迟在某些场景比 token 更关键（比如交互式 Agent）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import load_settings                            # noqa: E402
from nightwatch.eval.dataset import DatasetBuilder                     # noqa: E402
from nightwatch.eval.doe import (                                      # noqa: E402
    Factor, cost_effect_points, full_factorial, main_effects,
    orthogonal_l9, recommend,
)
from nightwatch.eval.metrics import Metrics                            # noqa: E402
from nightwatch.llm.client import build_llm                            # noqa: E402
from nightwatch.loop.react import TriageAgent                          # noqa: E402
from nightwatch.memory.null_store import NullMemoryStore               # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore         # noqa: E402
from nightwatch.models import MemoryTier                               # noqa: E402
from nightwatch.tools.registry import ToolRegistry                     # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C['bold']}── {t} {'─' * max(0, w - len(t) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


# =============================================================================
# 因子定义
# =============================================================================
# ★ 为什么是这三个因子？
#   它们是 Agent 系统里"改动最频繁、也最容易被想当然"的三个环节：
#     · 模型档次  —— 最容易被当成万能解药（"换更强的模型就行"）
#     · 记忆策略  —— 最容易被无条件打开（"记忆当然越多越好"）
#     · 思考链    —— 最容易一刀切（"关掉更快"）
#
#   选它们的理由是：**这三个恰好都是"单独看有道理、组合起来未必"的典型。**

MODEL_LEVELS = [
    ("small", "small（轻量档）"),
    ("medium", "medium（中档）"),
    ("strong", "strong（旗舰档）"),
]

MEMORY_LEVELS = [
    ("off", "关（NullStore）"),
    ("episodic", "只情景层"),
    ("episodic_semantic", "情景+语义层"),
]

THINKING_LEVELS = [
    ("all_on", "全开"),
    ("policy", "按任务分档"),
    ("all_off", "全关"),
]


def build_factors() -> list[Factor]:
    return [
        Factor("模型档次", [k for k, _ in MODEL_LEVELS],
               "整条链路固定在同一档（force_tier）"),
        Factor("记忆策略", [k for k, _ in MEMORY_LEVELS],
               "关闭 / 只用情景层 / 情景+语义"),
        Factor("思考链", [k for k, _ in THINKING_LEVELS],
               "全开 / 按任务分档 / 全关"),
    ]


# =============================================================================
# 按配置构造组件
# =============================================================================

def make_store(strategy: str) -> object:
    """按记忆策略构造 store。

    ★ "只情景层" vs "情景+语义层" 的差别体现在哪？
      在于**召回时是否包含语义层**。

      情景层给的是具体案例（"那次是怎么解决的"），
      语义层给的是抽象规则（"这类问题这么处理"）。

      因为语义层需要人工审核才能有内容（P1-2 的闸门），
      在没跑过扫描+审核的实验库里，语义层就是空的 ——
      所以这个因子的"情景+语义"组在**干净的实验库**上
      和"只情景"组表现相同。

      **这个限制必须写出来**，否则会得出"语义层没用"的错误结论 ——
      而真相是"实验库里还没有语义记忆"。
    """
    if strategy == "off":
        return NullMemoryStore(name="off")
    # 两种记忆策略都用同一个 store 实现，
    # 差别通过 recall 时是否带 tier 参数体现（见 run_one 的说明）
    return MemoryStore(":memory:", encoder=HashEmbedding(dim=256))


def make_llm(real: bool, model_tier: str, thinking: str):
    """按配置构造 LLM。"""
    if not real:
        return build_llm(use_mock=True)

    # 思考链策略 → disable_thinking 参数
    #   all_on   → False（强制开启）
    #   all_off  → True（强制关闭）
    #   policy   → None（按 config.THINKING_POLICY 逐任务决定）
    disable = {"all_on": False, "all_off": True, "policy": None}[thinking]
    return build_llm(use_mock=False, force_tier=model_tier,
                     disable_thinking=disable)


# =============================================================================
# 跑单个组合
# =============================================================================

def run_one(run, issues, real: bool, seed_memories: bool) -> dict:
    """跑一个实验组合，返回该组合的指标与成本。"""
    model_tier = run.settings["模型档次"]
    mem_strategy = run.settings["记忆策略"]
    thinking = run.settings["思考链"]

    settings = load_settings()
    store = make_store(mem_strategy)
    llm = make_llm(real, model_tier, thinking)

    # 预置一些情景记忆 —— 否则"开记忆"组和"关记忆"组没差别
    if seed_memories and mem_strategy != "off":
        _seed_memory(store)

    all_repo = [i for i in issues]
    f1s, top1s, hit2s = [], [], []
    tokens, steps, latencies = [], [], []
    n_pred, n_gold = [], []
    # ★ 错误收集用**局部变量**，不塞进 run.settings
    #
    #   第一版把错误写进 run.settings["_errors"]，结果 run.label() 里
    #   就多出一长串错误文本（因为 label 是遍历 settings.values() 拼的）——
    #   图表标签里全是报错，可读性完全没了。
    #
    #   教训：**"配置"和"运行结果"是两类数据，不要混在同一个容器里。**
    #   配置是输入（实验前就确定），结果是输出（实验后才产生）。
    #   混在一起会让"按配置分组""按配置画图"这类操作全部被污染。
    errors: list[str] = []

    for issue in issues:
        registry = ToolRegistry(all_repo, store=store,
                                current_issue_id=issue.issue_id)
        agent = TriageAgent(registry, llm=llm, store=store, settings=settings)
        t0 = time.time()
        try:
            r = agent.triage(issue, allowed_labels=_allowed(issues))
            tr = r.trace
            f1s.append(Metrics.label_f1(tr.predicted_labels, issue.gold_labels)[2])
            top1s.append(Metrics.top1_accuracy(tr.predicted_labels, issue.gold_labels))
            hit2s.append(Metrics.hit_at_k(tr.predicted_labels, issue.gold_labels, 2))
            tokens.append(tr.total_tokens)
            steps.append(tr.n_steps)
            n_pred.append(len(tr.predicted_labels or []))
            n_gold.append(len(issue.gold_labels or []))
        except Exception as e:                              # noqa: BLE001
            # 某个组合失败不该中断整个实验 —— 记为 0 分并继续。
            # **但要记下来**，因为"大量失败"本身就是个结论。
            f1s.append(0.0); top1s.append(0.0); hit2s.append(0.0)
            tokens.append(0); steps.append(0)
            n_pred.append(0); n_gold.append(len(issue.gold_labels or []))
            errors.append(f"{issue.issue_id}: {type(e).__name__}: {str(e)[:60]}")
        latencies.append((time.time() - t0) * 1000)

    med = lambda v: round(statistics.median(v), 4) if v else 0.0  # noqa: E731
    out = {
        "run_id": run.run_id,
        "settings": {k: v for k, v in run.settings.items()
                     if not k.startswith("_")},
        "n": len(issues),
        "f1": med(f1s),
        "f1_mean": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "top1": med(top1s),
        "hit2": med(hit2s),
        "tokens": int(sum(tokens)),
        "tokens_median": int(med(tokens)),
        "steps_median": med(steps),
        # 成本用"每条样本的平均 token"—— 比总量更可比（不同组合样本数可能不同）
        "cost_per_item": round(sum(tokens) / max(len(tokens), 1), 1),
        "latency_median_ms": int(med(latencies)),
        "over_label": round(
            (sum(n_pred) / max(len(n_pred), 1)) - (sum(n_gold) / max(len(n_gold), 1)),
            2),
        "errors": errors,
    }

    # 实验有效性检查：空 store 的调用计数
    if isinstance(store, NullMemoryStore):
        out["null_store_calls"] = store.total_calls()
    elif mem_strategy != "off":
        out["memory_count"] = store.count(MemoryTier.EPISODIC)

    if hasattr(store, "close"):
        store.close()
    return out


def _allowed(issues) -> list[str]:
    return sorted({l for i in issues for l in (i.gold_labels or [])})


def _seed_memory(store) -> None:
    """预置情景记忆。

    ★ 为什么要预置？
    因为"开记忆"这个因子的价值**只有在记忆库非空时才体现**。
    空库上开记忆 = 白开。

    这也是组件实验的一个通用前提：
    **任何"依赖某个组件的状态"的因子，都要先把那个状态准备好** ——
    否则你会得出"这个组件没用"的结论，而真相是"它没东西可用"。
    """
    from nightwatch.models import MemoryRecord, Outcome
    seeds = [
        ("s1", ["timeout", "permission"], "CI 超时且权限拒绝",
         "SA 缺 secret 读权限", "绑定 secret-reader 角色"),
        ("s2", ["timeout", "permission"], "流水线超时",
         "SA 缺 secret 读权限", "补权限并调高 timeout"),
        ("s3", ["merge", "数据量"], "merge 行数暴涨",
         "键列重复导致笛卡尔积", "先去重或加 validate"),
        ("s4", ["encoding"], "CSV 中文乱码",
         "文件是 GBK 但按 UTF-8 读", "指定 encoding='gbk'"),
        ("s5", ["memory", "oom"], "读取大文件 OOM",
         "未指定 dtype", "加 dtype + chunksize"),
    ]
    for mid, tags, symptom, cause, fix in seeds:
        store.insert(MemoryRecord(
            memory_id=mid, tier=MemoryTier.EPISODIC, issue_id=f"#{mid}",
            symptom=symptom, symptom_tags=tags, root_cause=cause,
            resolution=fix, outcome=Outcome.RESOLVED, confidence=0.8))


def _reanalyze(json_path: Path, args) -> int:
    """从已有 JSON 重新分析 —— 换成本指标、重新出图，不重跑实验。

    ★ 为什么要这个模式？
    第一次跑完（33 分钟）发现：**用 token 当成本几乎区分不出组合**
    （10628 ~ 13446，只差 1.27 倍），但**耗时差了 5.2 倍**
    （74s ~ 383s）。

    原因：`total_tokens` 是"输入 + 输出"的总和，而**输入 token 占大头**
    （同样的上下文，各组都一样），所以输出端的差异被淹没了 ——
    而输出才是配置差异真正影响的地方。

    **所以换一个成本代理就可能得出完全不同的推荐。**

    有了 `--from-json`，换指标只需几秒，而不是重跑半小时。
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]

    # 重建 design 对象（从 JSON 里的因子定义）
    factors = [Factor(f["name"], f["levels"], f.get("description", ""))
               for f in data["design"]["factors"]]
    design = (orthogonal_l9(factors) if data["design"]["kind"] == "orthogonal"
              else full_factorial(factors))

    print(f"\n{C['bold']}{C['cyan']}组件组合实验 · 重新分析{C['reset']}")
    hr()
    print(f"  来源        {json_path.name}")
    print(f"  原设计      {data['design']['name']}（{data['design']['n_runs']} 组）")
    print(f"  原始耗时    {data['config']['elapsed_sec'] / 60:.1f} 分钟")

    # --- 两种成本代理并排对比 ---
    hr("★ 两种成本代理的对比（这是重新分析的目的）")
    print(f"""
  {C['bold']}为什么换了成本指标，结论会变？{C['reset']}

    token 成本：{min(r['cost_per_item'] for r in results):.0f}
                ~ {max(r['cost_per_item'] for r in results):.0f}
                （极差 {max(r['cost_per_item'] for r in results) / max(min(r['cost_per_item'] for r in results), 1):.2f} 倍）

    耗时成本：{min(r['latency_median_ms'] for r in results) / 1000:.0f}s
                ~ {max(r['latency_median_ms'] for r in results) / 1000:.0f}s
                （极差 {max(r['latency_median_ms'] for r in results) / max(min(r['latency_median_ms'] for r in results), 1):.2f} 倍）

  {C['dim']}  total_tokens = 输入 + 输出，而**输入占大头**（各组上下文相同），
    所以输出端的差异被淹没了 —— 而输出才是配置真正影响的地方。
    **耗时更能反映"用户实际感受到的成本"。**{C['reset']}""")

    # --- 用耗时算成本-效果 ---
    hr("用耗时作为成本：主效应与推荐")

    lat_results = {r["run_id"]: r["f1"] for r in results}
    lat_costs = {r["run_id"]: r["latency_median_ms"] / 1000.0 for r in results}
    eff2 = main_effects(design, lat_results)
    reco2 = recommend(design, lat_results, lat_costs)

    # 耗时本身作为"被分析的量"：哪个因子对耗时影响最大
    eff_lat = main_effects(design, lat_costs)
    print(f"\n  {C['bold']}各因子对「耗时（秒）」的主效应{C['reset']}")
    print(f"  {'因子':<12}{'水平均值':<46}{'极差':>10}")
    print(f"  {'-' * 70}")
    for fname, info in eff_lat.items():
        means = "　".join(f"{k}={v:.0f}s" for k, v in info["means"].items())
        print(f"  {fname:<12}{means:<46}{info['range']:>9.0f}s")

    print(f"\n  {C['bold']}各因子对「F1」的主效应{C['reset']}")
    for fname, info in eff2.items():
        means = "　".join(f"{k}={v:.3f}" for k, v in info["means"].items())
        print(f"  {fname:<12}{means:<46}{info['range']:>9.3f}")

    print(f"\n  {C['bold']}推荐配置（以耗时为成本）{C['reset']}")
    for key in ("最省效果", "最省钱", "性价比最高", "达标里最省"):
        p = reco2.get(key)
        if p:
            print(f"    {key:<12} {p['label']:<42} "
                  f"耗时 {p['cost']:>6.0f}s　F1 {p['effect']:.3f}")

    # --- 出图 ---
    out_html = Path(args.html) if args.html else \
        json_path.with_suffix(".reanalyzed.html")
    render_html(design, results, eff2, reco2, out_html)
    print(f"\n  {C['green']}✓{C['reset']} 重新生成的报告 → {out_html}")
    return 0


# =============================================================================
# HTML 报告（含 Chart.js 图表）
# =============================================================================

def render_html(design, results: list[dict], effects: dict,
                reco: dict, path: Path) -> None:
    """生成成本-效果曲线 + 主效应图。"""
    pts = cost_effect_points(design, {r["run_id"]: r["f1"] for r in results},
                             {r["run_id"]: r["cost_per_item"] for r in results})
    frontier_ids = {p["run_id"] for p in pts if p["on_pareto_frontier"]}

    scatter = [{"x": p["cost"], "y": p["effect"],
                "label": p["label"], "id": p["run_id"],
                "frontier": p["on_pareto_frontier"]} for p in pts]

    # 主效应数据
    effect_data = {}
    for fname, info in effects.items():
        effect_data[fname] = {
            "labels": [str(k) for k in info["means"].keys()],
            "values": list(info["means"].values()),
            "range": round(info["range"], 4),
        }

    table_rows = "\n".join(
        f"<tr><td>{r['run_id']}</td>"
        + "".join(f"<td>{v}</td>" for v in r["settings"].values())
        + f"<td>{r['f1']:.3f}</td><td>{r['top1']:.3f}</td>"
          f"<td>{r['cost_per_item']:.0f}</td><td>{r['steps_median']}</td>"
          f"<td>{'✓' if r['run_id'] in frontier_ids else ''}</td>"
          f"<td>{len(r['errors'])}</td></tr>"
        for r in results)

    factor_names = [f.name for f in design.factors]
    head = "".join(f"<th>{n}</th>" for n in factor_names)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Nightwatch · 组件组合实验</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;
background:#0f1115;color:#e6e6e6;margin:0;padding:28px;line-height:1.65}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}}
h2{{font-size:16px;font-weight:500;margin:28px 0 10px;color:#8ab4f8}}
.sub{{color:#8a8f98;font-size:13px;margin-bottom:22px}}
.card{{background:#171a21;border:1px solid #262b35;border-radius:10px;
padding:18px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:7px 9px;text-align:left;border-bottom:1px solid #262b35}}
th{{color:#8a8f98;font-weight:500;font-size:12px}}
canvas{{max-height:320px}}
.note{{background:#1a1f2b;border-left:3px solid #8ab4f8;padding:12px 14px;
border-radius:6px;font-size:13px;margin:14px 0}}
.warn{{border-left-color:#f5a623}}
.ok{{border-left-color:#4caf50}}
code{{background:#22262f;padding:2px 6px;border-radius:4px;font-size:12.5px}}
</style></head><body>

<h1>Nightwatch · 组件组合实验</h1>
<div class="sub">{design.name}　{design.kind} 设计　{design.n_runs} 组组合
　样本 {results[0]['n'] if results else 0} 条/组</div>

<div class="note"><b>为什么做组合实验而不是单变量实验</b><br>
单独测试每个组件会得出错误结论 —— Agent 的表现来自<b>整条链路的配合</b>。
两个单独看都更好的选项，组合起来可能更差（负交互）。<br>
<span style="color:#8a8f98">{design.notes}</span></div>

<h2>成本-效果曲线</h2>
<div class="card"><canvas id="ce"></canvas>
<div class="note ok"><b>怎么读这张图</b><br>
横轴 = 每条样本的平均 token（成本代理，与供应商单价无关）<br>
纵轴 = 严格 F1 中位数（效果）<br>
<b>只有帕累托前沿上的点（实心）值得选</b> ——
其余点都存在"成本更低且效果更好"的配置。<br>
<span style="color:#8a8f98">刻意不标"最优配置"：取舍取决于业务对成本/效果的容忍度，
工具该做的是把权衡摊开，而不是替业务做决定。</span>
</div></div>

<h2>主效应（每个因子单独的影响幅度）</h2>
<div class="grid">
{''.join(f'<div class="card"><canvas id="eff{i}"></canvas>'
         f'<div class="sub" style="margin:0">极差 {info["range"]:.3f}'
         f'{"" if info["range"] >= 0.1 else "　⚠ 小于 0.1，样本量下不具统计意义"}</div></div>'
         for i, (fname, info) in enumerate(effect_data.items()))}
</div>

<h2>全部实验组</h2>
<div class="card"><table>
<tr><th>组</th>{head}<th>严格F1</th><th>Top-1</th><th>token/条</th>
<th>中位步数</th><th>帕累托</th><th>错误</th></tr>
{table_rows}
</table></div>

<h2>推荐配置</h2>
<div class="card">
<table>
<tr><th>优化目标</th><th>配置</th><th>成本</th><th>效果</th></tr>
<tr><td>效果最好</td><td>{reco.get('最省效果', {}).get('label', '—')}</td>
<td>{reco.get('最省效果', {}).get('cost', 0):.0f}</td>
<td>{reco.get('最省效果', {}).get('effect', 0):.3f}</td></tr>
<tr><td>最省钱</td><td>{reco.get('最省钱', {}).get('label', '—')}</td>
<td>{reco.get('最省钱', {}).get('cost', 0):.0f}</td>
<td>{reco.get('最省钱', {}).get('effect', 0):.3f}</td></tr>
<tr><td>性价比最高</td><td>{reco.get('性价比最高', {}).get('label', '—')}</td>
<td>{reco.get('性价比最高', {}).get('cost', 0):.0f}</td>
<td>{reco.get('性价比最高', {}).get('effect', 0):.3f}</td></tr>
<tr><td><b>效果达标里最省钱</b></td>
<td><b>{reco.get('达标里最省', {}).get('label', '—')}</b></td>
<td><b>{reco.get('达标里最省', {}).get('cost', 0):.0f}</b></td>
<td><b>{reco.get('达标里最省', {}).get('effect', 0):.3f}</b></td></tr>
</table>
<div class="note"><b>为什么给四个"最优"而不是一个</b><br>
"该多花 1 块钱换 0.01 的 F1"没有客观答案 —— 取决于业务容忍度。
给一个"最优"等于替你做了价值判断。</div>
</div>

<script>
const scatter = {json.dumps(scatter, ensure_ascii=False)};
const front = scatter.filter(p => p.frontier);
new Chart(document.getElementById('ce'), {{
  type: 'scatter',
  data: {{datasets: [
    {{label: '帕累托前沿', data: front, backgroundColor: '#4caf50',
      pointRadius: 7, pointStyle: 'rectRot'}},
    {{label: '被支配（有更优组合）', data: scatter.filter(p => !p.frontier),
      backgroundColor: '#5f6672', pointRadius: 5}},
    {{label: '理想方向', data: [{{x: Math.min(...scatter.map(p=>p.x)),
      y: Math.max(...scatter.map(p=>p.y))}},
      {{x: Math.max(...scatter.map(p=>p.x)), y: Math.min(...scatter.map(p=>p.y))}}],
      borderColor: '#8ab4f8', borderWidth: 1, borderDash: [5,5],
      pointRadius: 0, showLine: true, fill: false}}
  ]}},
  options: {{responsive: true, plugins: {{legend: {{labels: {{color:'#e6e6e6'}}}},
    tooltip: {{callbacks: {{label: c => c.raw.label + ' | 成本'
      + c.raw.x.toFixed(0) + ' | F1 ' + c.raw.y.toFixed(3)}}}}}},
    scales: {{x: {{title: {{display:true, text:'成本（token/条）', color:'#8a8f98'}},
      ticks:{{color:'#8a8f98'}}, grid:{{color:'#262b35'}}}},
      y: {{title: {{display:true, text:'严格 F1（中位数）', color:'#8a8f98'}},
      ticks:{{color:'#8a8f98'}}, grid:{{color:'#262b35'}}}}}}
}});

const effects = {json.dumps(effect_data, ensure_ascii=False)};
let i = 0;
for (const [name, d] of Object.entries(effects)) {{
  new Chart(document.getElementById('eff' + i), {{
    type: 'bar',
    data: {{labels: d.labels, datasets: [{{label: name, data: d.values,
      backgroundColor: '#4c8bf5'}}]}},
    options: {{responsive: true,
      plugins: {{legend: {{display:false}},
        title: {{display:true, text: name + ' 的主效应', color:'#e6e6e6'}}}},
      scales: {{y: {{ticks:{{color:'#8a8f98'}}, grid:{{color:'#262b35'}},
        min: 0, max: 1}}, x: {{ticks:{{color:'#8a8f98'}}, grid:{{display:false}}}}}}
  }});
  i++;
}}
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# =============================================================================
# 主流程
# =============================================================================

def main() -> int:                                          # noqa: C901
    ap = argparse.ArgumentParser(description="组件组合实验（P1-3）")
    ap.add_argument("--design", choices=["l9", "full"], default="l9",
                    help="l9=9 组正交（估主效应）/ full=27 组全因子（估交互）")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "pandas_eval.json"))
    ap.add_argument("--limit", type=int, default=5, help="每组跑几条样本")
    ap.add_argument("--real-llm", action="store_true", help="用真实模型")
    ap.add_argument("--mock", action="store_true", help="强制 Mock（快速验证流程）")
    ap.add_argument("--no-seed-memory", action="store_true",
                    help="不预置情景记忆（会削弱'开记忆'因子的效果）")
    ap.add_argument("--json", help="结果 JSON 路径")
    ap.add_argument("--html", help="报告 HTML 路径")
    ap.add_argument("--from-json", metavar="PATH",
                    help="从已有 JSON 重新分析（不重跑实验）")
    ap.add_argument("--cost-metric", choices=["tokens", "latency", "both"],
                    default="both",
                    help=("用哪个指标当成本。★ 默认 both —— 因为实测发现"
                          "两者会给出不同结论（见 render_html 的说明）"))
    args = ap.parse_args()

    # --- 从已有 JSON 重新分析（不重跑）---
    #
    # ★ 这是"数据已存"的价值：第一次跑完发现成本代理选得不好，
    #   可以直接换指标重算，而不用重跑 33 分钟的实验。
    #
    #   这正是 docs/05 F6 里那条教训的实践：
    #   **存逐条原始数据，支持未来任何角度的重算。**
    if args.from_json:
        return _reanalyze(Path(args.from_json), args)

    real = args.real_llm and not args.mock

    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 组件组合实验（P1-3）{C['reset']}")
    hr()

    # --- 数据 ---
    ds_path = Path(args.dataset)
    if ds_path.exists():
        raw = json.loads(ds_path.read_text(encoding="utf-8"))
        from nightwatch.models import Issue, IssueState
        issues = [
            Issue(issue_id=d["issue_id"], title=d["title"], body=d["body"],
                  author=d["author"], repo=d.get("repo", ""),
                  labels=d.get("labels", []), stack_trace=d.get("stack_trace", ""),
                  state=IssueState(d.get("state", "closed")),
                  gold_labels=d.get("gold_labels", []))
            for d in raw["holdout"][:args.limit]
        ]
        print(f"  数据集      {ds_path.name}（holdout 前 {len(issues)} 条）")
    else:
        ds = DatasetBuilder(seed=42).build([], source="synthetic")
        issues = ds.holdout[:args.limit] or ds.train[:args.limit]
        print(f"  {C['yellow']}⚠ 未找到真实数据集，用合成数据（结论不可用于简历）"
              f"{C['reset']}")

    # --- 设计 ---
    factors = build_factors()
    design = orthogonal_l9(factors) if args.design == "l9" \
        else full_factorial(factors)
    print(f"  实验设计    {design.name}（{design.n_runs} 组）")
    for f in factors:
        print("    因子 %s：%s" % (f.name, f.levels))
    print(f"  模型        {'真实（' + str(args.limit) + ' 条/组）' if real else 'Mock（验证流程用）'}")
    print(f"  预置记忆    {'否' if args.no_seed_memory else '是'}")

    if real:
        est = design.n_runs * len(issues) * 66 / 60
        print(f"  {C['yellow']}⚠ 预计耗时约 {est:.0f} 分钟"
              f"（按 66 秒/条估算，实际可能翻倍）{C['reset']}")

    # --- 跑 ---
    hr("开始跑组合")
    results = []
    t_start = time.time()
    for idx, run in enumerate(design.runs, 1):
        label = " / ".join(str(v) for v in run.settings.values())
        print(f"  [{idx:>2}/{design.n_runs}] {run.run_id}  {label:<46}",
              end="", flush=True)
        t0 = time.time()
        r = run_one(run, issues, real, not args.no_seed_memory)
        results.append(r)
        print(f"  {int(time.time() - t0):>5}s  "
              f"F1={C['green'] if r['f1'] > 0.6 else C['yellow']}{r['f1']:.3f}{C['reset']}  "
              f"token/条={r['cost_per_item']:.0f}")

    elapsed = time.time() - t_start

    # --- 分析 ---
    hr("主效应分析")
    eff = main_effects(design, {r["run_id"]: r["f1"] for r in results})
    print(f"\n  {'因子':<12}{'水平均值':<44}{'极差':>8}")
    print(f"  {'-' * 66}")
    for fname, info in eff.items():
        means = "　".join(f"{k}={v:.3f}" for k, v in info["means"].items())
        flag = "" if info["range"] >= 0.1 else f"  {C['dim']}(差异过小){C['reset']}"
        print(f"  {fname:<12}{means:<44}{info['range']:>8.3f}{flag}")
    print(f"""
  {C['dim']}  极差 = 该因子最好水平与最差水平的差距。
    极差 < 0.1 的因子在这个样本量下**不具统计意义** ——
    {design.kind} 设计只有 {design.n_runs - 1} 个自由度，分给 {len(factors)} 个因子后
    留给误差的自由度很少，小差异不该采信。{C['reset']}""")

    hr("推荐配置")
    reco = recommend(design, {r["run_id"]: r["f1"] for r in results},
                     {r["run_id"]: r["cost_per_item"] for r in results})
    for key in ("最省效果", "最省钱", "性价比最高", "达标里最省"):
        p = reco.get(key)
        if p:
            print(f"  {key:<12} {p['label']:<44} "
                  f"成本 {p['cost']:>6.0f}　F1 {p['effect']:.3f}")
    print(f"\n  帕累托前沿 {len(reco.get('帕累托前沿', []))} 个点"
          f"（其余都是被支配的，不值得选）")

    # --- 实验有效性检查 ---
    hr("实验有效性检查")

    # ★ Mock 模式下的效果数据**不可用于任何结论**
    #
    #   原因：MockLLM 是规则版（关键词匹配），它匹配的是合成数据的词表，
    #   对真实 pandas issue 几乎无能为力 —— 所以所有组的 F1 都会是 0，
    #   **组间差异也被压平**。
    #
    #   Mock 能验证的只有：框架能跑通、配置能生效、数据能收集、图能出。
    #   效果差异**必须**用真实模型跑。
    #
    #   这和 docs/03 里那条教训是同一个：
    #   **Mock 能验机制，不能验效果。**
    if not real:
        print(f"""  {C['yellow']}⚠ 当前是 Mock 模式 —— 效果数据不可用于结论{C['reset']}
    · MockLLM 是规则版（关键词匹配），对真实 issue 几乎无法产出正确标签
    · 所以所有组的 F1 都会接近 0，**组间差异被压平**
    · Mock 只能验证：框架跑通 / 配置生效 / 数据收集 / 图表生成
    {C['dim']}  要看效果差异，请用 --real-llm 重跑（真实模型）{C['reset']}""")

    null_runs = [r for r in results if r["settings"].get("记忆策略") == "off"]
    if null_runs:
        calls = [r.get("null_store_calls", 0) for r in null_runs]
        med_calls = statistics.median(calls) if calls else 0
        ok = med_calls > 0
        print(f"  '关记忆'组的 NullStore 调用次数（中位）：{med_calls:.0f}")
        print(f"  {C['green'] if ok else C['red']}{'✓' if ok else '✗'}{C['reset']} "
              f"{'记忆确实被调用过（说明拦截生效）' if ok else '没人调用记忆 —— 该组可能没真正生效'}")
    cached = [r for r in results if r["settings"].get("记忆策略") != "off"
              and r.get("memory_count", 0) > 0]
    print(f"  '开记忆'组有内容的：{len(cached)} 组"
          f"（每库 {cached[0]['memory_count'] if cached else 0} 条预置记忆）")
    err_runs = [r for r in results if r["errors"]]
    print(f"  有异常的组：{len(err_runs)} 组"
          + (f"　{C['yellow']}明细见 JSON{C['reset']}" if err_runs else ""))

    # --- 输出 ---
    out_json = Path(args.json) if args.json else \
        ROOT / "data" / "combo_experiment.json"
    payload = {
        "design": {"name": design.name, "kind": design.kind,
                   "n_runs": design.n_runs, "notes": design.notes,
                   "factors": [{"name": f.name, "levels": f.levels,
                                "description": f.description} for f in factors]},
        "config": {"dataset": str(ds_path), "limit": args.limit,
                   "real_llm": real, "elapsed_sec": round(elapsed, 1)},
        "results": results,
        "main_effects": {k: {"means": v["means"], "range": round(v["range"], 4),
                             "best": v["best"], "worst": v["worst"]}
                         for k, v in eff.items()},
        "recommendation": {
            k: ({"label": v["label"], "cost": v["cost"], "effect": v["effect"]}
                if isinstance(v, dict) and "label" in v else
                [{"label": x["label"], "cost": x["cost"], "effect": x["effect"]}
                 for x in v] if isinstance(v, list) else v)
            for k, v in reco.items()
        },
        "caveats": [
            f"{design.kind} 设计只有 {design.n_runs - 1} 个自由度，"
            f"留给误差的自由度有限 —— 这是筛选性实验，不是确证性实验",
            "差异小于 0.1 的结论不该采信（样本量 + 自由度都不够）",
            "成本用 token 数代理，未换算成金额（各家单价不同）",
            "'情景+语义'因子在干净实验库上与'只情景'无差别"
            "（语义层需要先跑扫描+人工审核才有内容）",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  {C['green']}✓{C['reset']} JSON → {out_json}")

    out_html = Path(args.html) if args.html else \
        ROOT / "data" / "combo_experiment.html"
    render_html(design, results, eff, reco, out_html)
    print(f"  {C['green']}✓{C['reset']} HTML 报告 → {out_html}")

    hr()
    print(f"""
  {C['bold']}验收对照{C['reset']}
    ✓ 至少 3×3 = 9 种组合各跑一遍
        —— {design.name}，共 {design.n_runs} 组
    ✓ 能回答"最强的模型是不是必要的"
        —— 见"主效应"里「模型档次」的极差；以及推荐里的"达标里最省"
    ✓ 给出成本-效果曲线
        —— {out_html.name}（含帕累托前沿）

  {C['dim']}总耗时 {elapsed / 60:.1f} 分钟{C['reset']}""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
