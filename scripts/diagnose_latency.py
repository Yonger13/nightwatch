#!/usr/bin/env python3
"""
Nightwatch · 延迟诊断：为什么 flash 比 max 还慢？

================================================================================
问题的提出
================================================================================
实测数据（同一端点、同一时段）：

    模型              极短输入的延迟
    qwen-max          570ms / 588ms / 476ms
    qwen3.8-flash    3471ms / 1593ms / 2722ms

**"flash 应该更快"这个前提，在极短输入下就已经不成立了** ——
而这恰恰说明：慢的原因**不是"计算量大"**（flash 模型更小、算得更少）。

那是什么？这个脚本要回答的就是这个问题。

================================================================================
三个必须分开的混淆因素
================================================================================
直接比"端到端延迟"是没有意义的，因为延迟里混了三样东西：

    总延迟 = 排队时间 + 预填充时间(prefill) + 解码时间(decode)

    ① **排队时间** —— 私有部署的实例是共享的，并发请求会排队。
       我之前的探测就是在**后台还有评测在跑**的时候做的，
       那 3471ms 里可能有很大一部分是排队，不是推理。
       → 变量：并发压力

    ② **预填充时间** —— 和**输入长度**成正比。输入越长，prefill 越慢。
       Agent 的上下文有 3000~8000 token，比"回复两个字"这种探测重得多。
       → 变量：prompt_tokens

    ③ **解码时间** —— 和**输出长度**成正比，这是模型"生成速度"的体现。
       如果模型带思考链（thinking），输出 token 会暴增。
       → 变量：completion_tokens

**只有把这三个分开，才能回答"到底是模型慢，还是别的原因"。**

================================================================================
关键指标：输出速率（tokens/sec）
================================================================================
    tokens_per_sec = completion_tokens / (总延迟 - 估计排队时间)

这个指标才反映"模型的生成速度"。

判断逻辑：
  · 如果 flash 的 tokens/sec **明显更高**，但总延迟更长
    → **不是模型慢，是输出太长**（比如默认开了思考链）→ 解法是限制输出
  · 如果 flash 的 tokens/sec **明显更低**
    → **是部署/资源问题**（私有部署的实例规格太小）→ 解法是扩实例
  · 如果两者 tokens/sec 接近，但 flash 的排队时间更长
    → **是并发排队问题** → 解法是错峰或加实例

**这三种情况的解法完全不同 —— 所以必须先测出是哪种。**

================================================================================
用法
================================================================================
    # 默认：对比 qwen-max 与 qwen3.8-flash
    python scripts/diagnose_latency.py

    # 指定要对比的模型（用逗号分隔）
    python scripts/diagnose_latency.py --models qwen-max,response=qwen3.8-flash

    # 测不同输入长度下的表现（定位 prefill 开销）
    python scripts/diagnose_latency.py --lengths 20,500,3000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch import config as _config          # noqa: F401  触发 .env 加载
from nightwatch.config import LLM_API_KEY, LLM_BASE_URL            # noqa: E402


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    RED = "\033[31m"

    @classmethod
    def disable(cls):
        for n in dir(cls):
            if n.isupper():
                setattr(cls, n, "")


def hr(t: str = "", w: int = 78) -> None:
    if t:
        print(f"\n{C.BOLD}── {t} {'─' * max(0, w - len(t) - 4)}{C.RESET}")
    else:
        print(f"{C.DIM}{'─' * w}{C.RESET}")


# =============================================================================
# 单次调用（带完整元数据）
# =============================================================================

def call_once(model: str, prompt: str, max_tokens: int | None = None,
              timeout: float = 180.0) -> dict:
    """调一次模型，返回**完整的耗时与 token 统计**。

    关键：必须把 `prompt_tokens` 和 `completion_tokens` 都拿到 ——
    没有它们就无法区分"是输入太长"还是"输出太长"。
    """
    import httpx

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    t0 = time.time()
    try:
        resp = httpx.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        latency = time.time() - t0

        if resp.status_code != 200:
            return {"ok": False, "model": model, "latency": latency,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        usage = data.get("usage", {}) or {}
        msg = data["choices"][0].get("message", {}) or {}
        content = msg.get("content") or ""
        # ★ 有些模型把思考过程放在 reasoning_content 里
        reasoning = msg.get("reasoning_content") or ""

        return {
            "ok": True,
            "model": data.get("model", model),
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "content_len": len(content),
            "reasoning_len": len(reasoning),      # ★ 思考链长度
            "has_reasoning": bool(reasoning),
            "content_head": content[:120].replace("\n", " "),
        }
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "model": model, "latency": time.time() - t0,
                "error": f"{type(e).__name__}: {str(e)[:150]}"}


def tokens_per_sec(c: dict) -> float:
    """输出速率（tokens/秒）—— 衡量"模型生成速度"的本质指标。

    注意这里用的是 `completion_tokens`，不是 `total_tokens`。
    因为 prefill（处理输入）和解码（生成输出）是两种不同的开销，
    混在一起算会掩盖真正的问题。
    """
    if not c.get("ok") or c["latency"] <= 0:
        return 0.0
    return c.get("completion_tokens", 0) / c["latency"]


# =============================================================================
# 实验
# =============================================================================

def exp_basic(models: list[str], repeats: int) -> dict:
    """实验 1：极短输入下的基线对比。

    这条测的是**端到端延迟**，但会同时记录 token 数 ——
    借此判断"慢"是来自输入、输出，还是纯推理。
    """
    hr("实验 1 · 极短输入基线（分离 prefill / decode）")

    prompt = "只回复两个字：正常"
    print(f"  {C.DIM}输入：{prompt!r}（极短，prefill 开销可忽略）")
    print(f"  每个模型重复 {repeats} 次，间隔 1 秒，避免自身排队{C.RESET}\n")

    results: dict[str, list[dict]] = {}
    for m in models:
        print(f"  {C.BOLD}{m}{C.RESET}")
        rows = []
        for i in range(repeats):
            c = call_once(m, prompt, max_tokens=64)
            rows.append(c)
            if not c["ok"]:
                print(f"    {C.RED}✗ {c.get('error')}{C.RESET}")
                break
            print(f"    [{i + 1}/{repeats}] {c['latency'] * 1000:>7.0f}ms  "
                  f"入={c['prompt_tokens']:>4} tok  "
                  f"出={c['completion_tokens']:>4} tok  "
                  f"速率={tokens_per_sec(c):>6.1f} tok/s"
                  + (f"  {C.YELLOW}⚠含思考链({c['reasoning_len']}字符){C.RESET}"
                     if c["has_reasoning"] else ""))
            if i < repeats - 1:
                time.sleep(1.0)
        results[m] = rows
        print()

    return results


def exp_lengths(model: str, lengths: list[int]) -> list[dict]:
    """实验 2：不同输入长度下的延迟（定位 prefill 开销）。

    如果延迟随输入长度线性上升，说明 **prefill 是主要成本** ——
    这在 Agent 场景下很关键，因为 Agent 的上下文天然很长。

    **这条实验直接对应 Agent 场景**：不是"回复两个字"那种理想输入，
    而是真实的 3000~8000 token 上下文。
    """
    hr("实验 2 · 输入长度对延迟的影响（prefill 开销）")
    print(f"  {C.DIM}模型：{model}")
    print(f"  目的：判断「慢」是否来自长上下文（Agent 的真实工况）{C.RESET}\n")

    rows = []
    for n in lengths:
        # 构造指定近似长度的输入（中文 1 字 ≈ 1 token）
        filler = "这是一个用于测试上下文长度对推理延迟影响的填充句子。" * (n // 26 + 1)
        prompt = filler[:n] + "\n\n请只回复两个字：正常"
        c = call_once(model, prompt, max_tokens=64)
        c["target_len"] = n
        rows.append(c)
        if c["ok"]:
            # 每 token 的 prefill 成本（粗略，因为含固定开销）
            per_tok = c["latency"] / max(c["prompt_tokens"], 1) * 1000
            print(f"  输入 {c['prompt_tokens']:>5} tok → "
                  f"{c['latency'] * 1000:>7.0f}ms  "
                  f"（{per_tok:.2f} ms/tok）  "
                  f"出={c['completion_tokens']} tok")
        else:
            print(f"  输入 ~{n} tok → {C.RED}✗ {c.get('error')}{C.RESET}")
        time.sleep(1.0)
    return rows


def exp_agent_like(models: list[str]) -> dict:
    """实验 3：模拟 Agent 的真实调用（长输入 + 结构化输出要求）。

    这是最能说明问题的一条 —— 因为 Agent 场景的真实特征是：
      · 输入长（system prompt + 工具 schema + 历史 + 记忆）
      · 输出要求结构化（JSON）
      · 可能触发思考链

    **和前两个实验对比，就能看出"理想输入"和"真实工况"差多少。**
    """
    hr("实验 3 · 模拟 Agent 真实调用（长输入 + JSON 输出）")

    system_like = (
        "你是开源项目的问题处理助手。你的任务是阅读一个新提交的 issue，"
        "给出分类建议和处理建议。\n\n"
        "## 判断准则\n1. 先查再判\n2. 判重优先于分类\n3. 允许不确定\n"
        "4. 不要重复调用同一个工具\n5. 给出依据\n\n"
        "## 可用工具\n" + json.dumps([
            {"name": f"tool_{i}", "description": "工具描述" * 20,
             "input_schema": {"type": "object", "properties": {
                 "a": {"type": "string", "description": "参数说明" * 10}}}}
            for i in range(6)
        ], ensure_ascii=False)[:3000]
        + "\n\n## 输出格式\n每次回复必须是一个 JSON 对象："
          '{"thought": "...", "action": "工具名或 finalize", "action_input": {}}'
    )
    prompt = system_like + "\n\n请给出你的下一步动作。"

    print(f"  {C.DIM}构造了类似 Agent 的 system prompt（约 "
          f"{len(system_like)} 字符）")
    print(f"  要求模型输出 JSON —— 这正是它在主循环里要做的事{C.RESET}\n")

    results = {}
    for m in models:
        c = call_once(m, prompt, max_tokens=512)
        results[m] = c
        if c["ok"]:
            print(f"  {C.BOLD}{m}{C.RESET}")
            print(f"    输入 {c['prompt_tokens']} tok  "
                  f"输出 {c['completion_tokens']} tok  "
                  f"延迟 {c['latency'] * 1000:.0f}ms  "
                  f"速率 {tokens_per_sec(c):.1f} tok/s")
            if c["has_reasoning"]:
                print(f"    {C.YELLOW}⚠ 检测到思考链内容"
                      f"（reasoning_content 长度 {c['reasoning_len']} 字符）{C.RESET}")
            print(f"    {C.DIM}输出开头：{c['content_head']}{C.RESET}")
        else:
            print(f"  {C.BOLD}{m}{C.RESET}  {C.RED}✗ {c.get('error')}{C.RESET}")
        print()
        time.sleep(1.0)
    return results


# =============================================================================
# 汇总分析
# =============================================================================

def summarize(basic: dict, agent_like: dict) -> None:
    hr("汇总分析")

    print(f"\n  {'模型':<22}{'中位延迟':>10}{'输出tok':>10}{'速率tok/s':>12}"
          f"{'含思考链':>10}")
    print(f"  {'-' * 66}")

    stats: dict[str, dict] = {}
    for m, rows in basic.items():
        ok = [r for r in rows if r.get("ok")]
        if not ok:
            print(f"  {m:<22}{'失败':>10}")
            continue
        lat = statistics.median(r["latency"] for r in ok) * 1000
        out = statistics.median(r["completion_tokens"] for r in ok)
        rate = statistics.median(tokens_per_sec(r) for r in ok)
        reason = any(r.get("has_reasoning") for r in ok)
        stats[m] = {"latency_ms": lat, "out_tok": out, "rate": rate,
                    "reasoning": reason}
        print(f"  {m:<22}{lat:>9.0f}ms{out:>10.0f}{rate:>12.1f}"
              f"{'是' if reason else '否':>10}")

    print(f"""
{C.BOLD}诊断逻辑（三种情况对应三种完全不同的解法）{C.RESET}

  ① 若某模型 tokens/s 明显更高但延迟更长
     → 不是模型慢，是**输出太长**（多半是默认开了思考链 thinking）
     → 解法：在请求里关掉思考（如 enable_thinking=false）或限制 max_tokens

  ② 若某模型 tokens/s 明显更低
     → **是部署/资源问题**。私有部署的实例规格由你自己配置，
       "flash 更快"的前提是**同规格部署** —— 小实例上跑 flash 当然更慢
     → 解法：查实例规格 / 并发配额，或换更大的实例

  ③ 若 tokens/s 接近但排队延迟差很多
     → **是并发排队**。同一实例上的请求会排队，
       后台有评测在跑时，前台探测测到的"延迟"大部分是排队
     → 解法：错峰、加实例、或提高并发配额

{C.YELLOW}本次数据还需要结合一个事实看：{C.RESET}
{C.DIM}  之前那组 570ms vs 3471ms 的对比，是在**后台还有评测任务在跑**时测的，
  所以里面混了排队时间 —— 那不是干净的对照。
  本次三个实验都是**空载**下跑的，结论更可信。{C.RESET}
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="延迟诊断")
    ap.add_argument("--models", default="qwen-max,qwen3.8-flash",
                    help="要对比的模型，逗号分隔（默认 qwen-max,qwen3.8-flash）")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lengths", default="20,800,3000",
                    help="输入长度梯度（实验 2）")
    ap.add_argument("--skip-lengths", action="store_true")
    args = ap.parse_args()

    if not sys.stdout.isatty():
        C.disable()

    models = [m.split("=")[-1] for m in args.models.split(",") if m.strip()]

    print(f"\n{C.BOLD}{C.CYAN}Nightwatch · 延迟诊断{C.RESET}")
    hr()
    if not LLM_API_KEY:
        print(f"  ✗ 需要 NW_LLM_API_KEY")
        return 1
    print(f"  端点   {LLM_BASE_URL}")
    print(f"  对比   {' vs '.join(models)}")
    print(f"  {C.DIM}说明：本次全部在空载下测（没有后台任务）——"
          f"这样测出的才是干净的延迟{C.RESET}")

    basic = exp_basic(models, args.repeats)

    lengths = [int(x) for x in args.lengths.split(",")]
    if not args.skip_lengths:
        exp_lengths(models[-1], lengths)

    agent_like = exp_agent_like(models)
    summarize(basic, agent_like)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
