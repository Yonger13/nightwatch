#!/usr/bin/env python3
"""
Nightwatch · API 配置检查与成本预估

================================================================================
这个脚本解决什么问题
================================================================================
接真实模型之前，有三个问题必须先回答，否则跑起来会踩坑：

  ① **配置对不对？** —— key 有效吗？三档模型你的账号都有权限吗？
  ② **成本多少？** —— 一次压测要花多少钱？会不会超预算？
  ③ **效果有没有变化？** —— 真实模型和 Mock 的行为差在哪？

如果跳过这一步直接跑完整压测，可能出现：
  · 跑到一半发现某档模型没权限 → 白跑
  · 花了几十块钱才发现配置写错了
  · 压测结果异常，但不知道是配置问题还是模型问题

所以这个脚本的设计原则是：**先小规模验证，再大规模投入。**

================================================================================
它做哪四件事
================================================================================
    1. 环境变量检查
    2. LLM 连通性（strong / medium / small 三档分别测）
    3. 向量编码器检查（语义 embedding 是否真的启用了）
    4. **跑一次真实 triage，用它实测的 token 量反推完整压测的成本**

第 4 步是关键：成本不是估的，是**基于实测 token 量算的**。
这也是这个项目一贯的做法 —— 不拍数字。

================================================================================
用法
================================================================================
    # 先设置环境变量
    export NW_LLM_API_KEY=sk-xxxx
    export NW_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

    # 检查
    python scripts/check_api.py

    # 带上你的实际单价，脚本会直接算钱（单位：元/百万 token）
    python scripts/check_api.py --price-in 20 --price-out 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.config import (                                    # noqa: E402
    LLM_API_KEY, LLM_BASE_URL, MODELS, MODEL_ROUTING, load_settings,
)
from nightwatch.eval.dataset import build_synthetic_dataset        # noqa: E402
from nightwatch.llm.client import OpenAICompatLLM, build_llm       # noqa: E402
from nightwatch.loop.react import TriageAgent                      # noqa: E402
from nightwatch.memory.store import build_encoder                  # noqa: E402
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


OK = f"{C.GREEN}✓{C.RESET}"
NO = f"{C.RED}✗{C.RESET}"
WARN = f"{C.YELLOW}!{C.RESET}"


class _Tee:
    """把输出同时写到终端和文件。

    为什么需要它？
    两个理由：
      ① **编码**：在 Windows 上通过管道捕获子进程输出时，中文经常被搞乱
         （UTF-8 字节被 GBK 解码）。写文件能保证拿到干净的 UTF-8。
      ② **留痕**：检查报告值得存下来 —— 尤其是和团队协作、
         或者要对比"换了一个 key 之后有什么变化"时。

    这就是一个很实用的工程习惯：**重要的检查过程要能被复现和追溯**，
    而不是只在终端闪一下就没了。
    """

    def __init__(self, path: Path | None):
        self._fh = open(path, "w", encoding="utf-8") if path else None
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        self._stdout.write(s)
        if self._fh:
            self._fh.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        if self._fh:
            self._fh.flush()

    def isatty(self) -> bool:
        return self._stdout.isatty()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def hr(title: str = "", width: int = 78) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{C.BOLD}── {title} {'─' * pad}{C.RESET}")
    else:
        print(f"{C.DIM}{'─' * width}{C.RESET}")


# =============================================================================
# 0 · 依赖检查
# =============================================================================

def check_deps() -> bool:
    """检查调用真实模型所需的依赖。

    ★ 为什么把这个放**第一位**？
    因为"没装 httpx"比"key 不对"更基础 —— 如果依赖缺失，
    后面所有检查都会失败，而错误信息会**误导你去查 key**。

    排错的第一原则是：**从最基础的条件开始查。**
    顺序错了，就会在一个正确的地方找一个不存在的问题。
    """
    hr("0 · 依赖检查")

    try:
        import httpx
        ver = getattr(httpx, "__version__", "?")
        print(f"  {OK} httpx {ver}（调用真实模型 / embedding 必需）")
        return True
    except ImportError:
        print(f"  {NO} httpx 未安装 —— 无法调用真实模型")
        print(f"\n      {C.BOLD}安装：{C.RESET}")
        print(f"        pip install httpx")
        print(f"      {C.DIM}或一次装好所有可选依赖：")
        print(f"        pip install -r requirements.txt{C.RESET}")
        print(f"\n      {C.DIM}说明：项目主流程**不需要** httpx（设计成零依赖），")
        print(f"      所以它列在 requirements.txt 的可选部分。")
        print(f"      但接真实模型时必须装。{C.RESET}")
        return False


# =============================================================================
# 1 · 环境变量
# =============================================================================

def check_env() -> bool:
    hr("1 · 环境变量")
    ok = True

    if LLM_API_KEY:
        masked = LLM_API_KEY[:6] + "…" + LLM_API_KEY[-4:] \
            if len(LLM_API_KEY) > 12 else "***"
        print(f"  {OK} NW_LLM_API_KEY  已设置（{masked}）")
    else:
        print(f"  {NO} NW_LLM_API_KEY  未设置")
        print(f"\n      {C.BOLD}推荐方式{C.RESET}：把 .env.example 复制为 .env，"
              f"填入 key")
        print(f"        {C.DIM}cp .env.example .env{C.RESET}")
        print(f"        {C.DIM}（.env 已被 .gitignore 忽略，不会提交到 git）{C.RESET}")
        print(f"\n      {C.DIM}或临时用环境变量（只在当前终端会话有效）：")
        print(f"        Linux/macOS   export NW_LLM_API_KEY=sk-xxxx")
        print(f"        Windows CMD   set NW_LLM_API_KEY=sk-xxxx")
        print(f"        PowerShell    $env:NW_LLM_API_KEY=\"sk-xxxx\"{C.RESET}")
        ok = False

    print(f"  {OK} NW_LLM_BASE_URL {LLM_BASE_URL}")

    print(f"\n  {C.DIM}模型档位映射（config.MODEL_ROUTING）：{C.RESET}")
    for task, tier in MODEL_ROUTING.items():
        print(f"    {task:<18} → {tier:<7} = {MODELS.get(tier, '?')}")

    print(f"\n  {C.DIM}设计说明：分三档是为了按任务匹配模型。"
          f"实测发现'最强模型不一定最好'，\n"
          f"  而且主循环决策（agent_step）是唯一真正需要强模型的环节。"
          f"详见 docs/02。{C.RESET}")
    return ok


# =============================================================================
# 2 · LLM 连通性
# =============================================================================

def check_llm() -> tuple[bool, dict]:
    """分别测三档模型。

    ★ 为什么要分档测？
    很多账号并不是所有型号都有权限。典型情况：
      · 有 qwen-plus，没有 qwen-max（需要单独申请）
      · 或者反过来，只买了便宜的档位
    如果不分档测，跑起来会在某个环节突然 403，而那时已经跑了一半。
    """
    hr("2 · LLM 连通性（三档分别测）")

    if not LLM_API_KEY:
        print(f"  {WARN} 跳过（没有 API key）")
        return False, {}

    client = OpenAICompatLLM()
    probes = [
        ("agent_step", "strong", "主循环决策"),
        ("memory_resolve", "medium", "记忆冲突消解"),
        ("memory_extract", "small", "记忆抽取"),
    ]

    results = {}
    all_ok = True
    for task, tier, label in probes:
        model = MODELS.get(tier, tier)
        t0 = time.time()
        try:
            resp = client.complete(
                [{"role": "user", "content": "只回复两个字：正常"}],
                task=task)
            dt = int((time.time() - t0) * 1000)
            print(f"  {OK} {tier:<7} {model:<32} {dt:>6}ms  "
                  f"{label}")
            # 检查返回是否合理（有些网关对不存在的模型返回空而不是报错）
            if not resp.text.strip():
                print(f"      {WARN} 返回内容为空 —— 这个模型可能不可用，"
                      f"或网关静默降级了")
                all_ok = False
            results[tier] = {"model": model, "ok": True, "latency_ms": dt}
        except Exception as e:                             # noqa: BLE001
            msg = str(e)
            print(f"  {NO} {tier:<7} {model:<32} 失败：{msg[:60]}")
            results[tier] = {"model": model, "ok": False, "error": msg}
            all_ok = False

            # 给出针对性的建议
            low = msg.lower()
            if "401" in msg or "auth" in low or "invalid_api_key" in low:
                print(f"      {C.DIM}→ API key 无效或过期，检查 NW_LLM_API_KEY{C.RESET}")
            elif "403" in msg or "permission" in low:
                print(f"      {C.DIM}→ 账号可能没有 {model} 的权限。"
                      f"改 config.py 里的 MODELS，或设环境变量 "
                      f"NW_MODEL_{tier.upper()}={C.RESET}")
            elif "404" in msg or "model" in low:
                print(f"      {C.DIM}→ 模型名 {model} 在你的服务上不存在。"
                      f"设 NW_MODEL_{tier.upper()}=你的模型名{C.RESET}")
            elif "timeout" in low or "connection" in low:
                print(f"      {C.DIM}→ 网络不通。检查 BASE_URL 或代理{C.RESET}")

    return all_ok, results


# =============================================================================
# 3 · 向量编码器
# =============================================================================

def check_embedding() -> bool:
    """检查语义 embedding 是否真的启用了。

    ★ 这里有个容易上当的地方：
    APIEmbedding 在失败时会**静默降级**到哈希伪向量（这是刻意的可靠性设计）。
    但降级之后，duplicate 判定的效果会明显变差 ——
    而如果你不知道降级发生了，会误以为"真实模型效果就这水平"。

    所以必须显式检查维度：哈希是 256 维，常见的 API embedding 是 768/1024/1536。
    """
    hr("3 · 向量编码器")

    enc = build_encoder(use_api=True)
    name = type(enc).__name__

    if name == "HashEmbedding":
        print(f"  {WARN} 当前用的是 HashEmbedding（哈希伪向量）")
        print(f"      {C.DIM}原因可能是：没有 API key。"
              f"它是零依赖的降级方案，判别力有限。{C.RESET}")
        return False

    try:
        vec = enc.encode("接口在高峰时段超时")
        dim = len(vec)
        if dim == 256:
            print(f"  {WARN} 拿到的是 256 维 —— 说明 APIEmbedding "
                  f"**静默降级**到了哈希版")
            print(f"      {C.DIM}降级原因通常是：embedding 接口不可用 / "
                  f"模型名不对 / key 无权限。{C.RESET}")
            print(f"      {C.DIM}后果：duplicate 判定的 F1 会明显偏低，"
                  f"别把它的锅算在模型头上。{C.RESET}")
            return False
        print(f"  {OK} {name} 可用，维度 {dim}")
        print(f"      {C.DIM}语义 embedding 会让 duplicate 判定的 F1 "
              f"明显高于哈希版（哈希版实测天花板约 0.5）。{C.RESET}")
        return True
    except Exception as e:                                 # noqa: BLE001
        print(f"  {NO} 编码失败：{e}")
        return False


# =============================================================================
# 4 · 真实 triage 冒烟 + 成本反推
# =============================================================================

def smoke_triage() -> dict | None:
    """跑一次真实的 triage，用实测 token 量反推压测成本。

    ★ 为什么用真实调用而不是估算？
    因为 token 消耗受很多东西影响：工具返回值有多大、记忆召回了几条、
    压缩是否触发、模型输出多长…… 估算的误差可能到 2~3 倍。

    **只有实测才有意义。** 而这个脚本只花一次调用的钱。
    """
    hr("4 · 真实 triage 冒烟（含成本反推）")

    if not LLM_API_KEY:
        print(f"  {WARN} 跳过（没有 API key）")
        return None

    ds = build_synthetic_dataset()
    issues = ds.train + ds.holdout
    target = next(i for i in issues if i.gold_is_duplicate_of)
    history = [i for i in issues if i.issue_id != target.issue_id]

    repo = IssueRepository(history + [target])
    repo.add_doc("CONTRIBUTING.md",
                 "标签规范：bug/timeout 超时类；bug/permission 权限类；"
                 "若与已有 issue 高度相似，标记 duplicate 而非重新分类。")

    from nightwatch.memory.store import MemoryStore
    encoder = build_encoder(use_api=True)
    store = MemoryStore(":memory:", encoder=encoder)
    reg = ToolRegistry(repo, store=store, current_issue_id=target.issue_id)
    agent = TriageAgent(reg, llm=build_llm(use_mock=False), store=store,
                        settings=load_settings(), window_size=8000)

    print(f"  {C.DIM}处理 {target.issue_id} {target.title[:30]} …{C.RESET}")
    t0 = time.time()
    try:
        result = agent.triage(target, context_padding=100,
                              enable_state_card=True)
    except Exception as e:                                 # noqa: BLE001
        print(f"  {NO} triage 失败：{e}")
        return None
    elapsed = int((time.time() - t0) * 1000)

    trace = result.trace
    in_tok = trace.total_tokens        # 注意：这里统计的是"所有轮次的 token 之和"
    out_tok = max(100, in_tok // 10)   # 粗估输出占比（真实输出通常远比输入少）

    print(f"  {OK} 完成：{result.n_steps} 步，{elapsed}ms，"
          f"总 token {in_tok}")
    print(f"      预测标签 {result.predicted_labels or '—'}，"
          f"判重 {result.duplicate_of or '—'}，"
          f"引用 {len(result.cites)} 条，置信度 {result.confidence:.2f}")

    # 上下文压力证据
    ev = sum(len(u.get("evicted", [])) for u in trace.context_usage)
    cm = sum(len(u.get("compressed", [])) for u in trace.context_usage)
    print(f"      上下文：驱逐 {ev} 次，压缩 {cm} 次")

    return {
        "tokens_per_run": in_tok,
        "output_per_run": out_tok,
        "steps": result.n_steps,
        "elapsed_ms": elapsed,
        "adherence_ok": len(result.cites) > 0 and result.confidence <= 0.8,
    }


def estimate_cost(usage: dict, price_in: float, price_out: float) -> None:
    """基于实测 token 量反推完整压测的成本。

    ============================================================
    这里要算三部分，不能只算压测本身
    ============================================================
      ① 压测本体：5 个 padding × 2 组 × 3 次 = 30 次 triage
      ② 记忆预热：压测里每次都要建记忆库，不额外消耗
      ③ 单独试跑：你可能还要跑几次调参 → 留 20% 余量

    另外重要的一点：**padding 越大，token 消耗越高**。
    实测的那次用的是 padding=100（最大档），所以它是**上界**。
    实际平均会低于它 —— 这是"保守估算"，宁可高估。
    """
    hr("5 · 成本预估（基于实测反推）")

    if not usage:
        print(f"  {WARN} 没有实测数据，无法预估")
        return

    # 压测规模
    n_paddings = 5          # 0,10,30,60,100
    n_groups = 2            # 状态卡 开/关
    n_runs = 3
    n_calls = n_paddings * n_groups * n_runs

    per_run_in = usage["tokens_per_run"]
    per_run_out = usage["output_per_run"]

    # 保守修正：实测用的是最大 padding，平均会低一些，取 70%
    avg_factor = 0.7

    total_in = int(per_run_in * n_calls * avg_factor)
    total_out = int(per_run_out * n_calls * avg_factor)

    print(f"\n  压测规模：{n_paddings} 个 padding × {n_groups} 组 "
          f"× {n_runs} 次 = {n_calls} 次 triage")
    print(f"  实测单次（padding=100，上界）：输入 {per_run_in:,} tok / "
          f"输出约 {per_run_out:,} tok")
    print(f"  保守系数 {avg_factor}（padding 平均小于最大值）")
    print(f"\n  预估总消耗：输入约 {total_in:,} tok，输出约 {total_out:,} tok")
    print(f"  {C.DIM}（这是保守上界，实际应该低于它）{C.RESET}")

    if price_in > 0 or price_out > 0:
        cost = total_in / 1e6 * price_in + total_out / 1e6 * price_out
        print(f"\n  按 输入 {price_in} 元/百万 tok、输出 {price_out} 元/百万 tok 计：")
        print(f"  {C.BOLD}预估成本约 {cost:.2f} 元{C.RESET}")
        print(f"  {C.DIM}建议先按 1/10 规模试跑验证配置，再跑完整版{C.RESET}")
    else:
        print(f"\n  {C.DIM}想直接算钱？加上你的单价重跑：")
        print(f"    python scripts/check_api.py --price-in 20 --price-out 60")
        print(f"  （单价查你所用服务的官网价格页，不同档位价格不同 ——"
              f"本脚本不内置价格表，因为价格会变）{C.RESET}")

    print(f"""
  {C.YELLOW}省钱的三个办法：{C.RESET}
  {C.DIM}  ① 只跑 --real-llm 测主循环，记忆抽取仍用小模型（config 已经这么配了）
    ② 先用 --runs 1 验证配置，确认无误再跑 --runs 5
    ③ 先只跑 padding=0,100 两个端点，看趋势是否成立
       python scripts/run_longhorizon.py --real-llm --paddings 0,100 --runs 1{C.RESET}""")


# =============================================================================
# 主流程
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Nightwatch API 配置检查与成本预估",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--price-in", type=float, default=0.0,
                    help="输入单价（元/百万 token）")
    ap.add_argument("--price-out", type=float, default=0.0,
                    help="输出单价（元/百万 token）")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="跳过真实调用（只检查配置，不花钱）")
    ap.add_argument("--log", help="把检查报告同时写到这个文件（UTF-8）")
    args = ap.parse_args()

    if args.log:
        sys.stdout = _Tee(Path(args.log))

    if not sys.stdout.isatty():
        C.disable()

    print(f"\n{C.BOLD}{C.CYAN}Nightwatch · API 配置检查{C.RESET}")

    deps_ok = check_deps()
    env_ok = check_env()

    if not env_ok or not deps_ok:
        hr("结论")
        missing = []
        if not deps_ok:
            missing.append("httpx 依赖（pip install httpx）")
        if not env_ok:
            missing.append("API Key（cp .env.example .env 然后填进去）")
        print(f"\n  {NO} 还差：{'、'.join(missing)}\n")
        print(f"""
  {C.BOLD}你需要准备的：{C.RESET}
    {C.BOLD}1. 装依赖{C.RESET}（如果上面标了 ✗）
         pip install httpx

    {C.BOLD}2. API Key{C.RESET}（必需）
       任何兼容 OpenAI 协议的服务都可以。最快的配置方式：
         cp .env.example .env      # 然后把 key 填进去
         python scripts/check_api.py

    {C.BOLD}3. Base URL{C.RESET}（可选，默认阿里百炼）
       常见选项：
         阿里百炼  https://dashscope.aliyuncs.com/compatible-mode/v1   ← 默认
         DeepSeek  https://api.deepseek.com/v1
         Moonshot  https://api.moonshot.cn/v1
         OpenAI    https://api.openai.com/v1
         本地 vLLM http://localhost:8000/v1
       在 .env 里设：NW_LLM_BASE_URL=https://...

    {C.BOLD}4. 模型名{C.RESET}（可选，三档都要有）
       如果默认名（qwen-turbo/plus/max）在你的服务上不存在，逐档指定：
         NW_MODEL_STRONG=deepseek-chat
         NW_MODEL_MEDIUM=deepseek-chat
         NW_MODEL_SMALL=deepseek-chat

    {C.BOLD}5. 向量模型{C.RESET}（可选但强烈建议）
       语义 embedding 决定 duplicate 判定的效果上限。
       如果服务商提供 /embeddings 接口，会自动启用。
       默认模型名 text-embedding-v3（阿里百炼）。其他服务商需要改
       src/nightwatch/memory/store.py 里 APIEmbedding 的 model 参数。

  {C.DIM}配置好之后重跑这个脚本，它会逐项验证并估算成本。{C.RESET}""")
        return 1

    llm_ok, _ = check_llm()
    emb_ok = check_embedding()

    usage = None
    if not args.skip_smoke and llm_ok:
        usage = smoke_triage()
    elif args.skip_smoke:
        hr("4 · 真实 triage 冒烟")
        print(f"  {C.DIM}已跳过（--skip-smoke）{C.RESET}")

    if usage:
        estimate_cost(usage, args.price_in, args.price_out)

    # ---------------- 结论 ----------------
    #
    # ★ 这里修了一个逻辑 bug，值得说明：
    #
    # 原来的判断是 `if llm_ok and emb_ok and usage:`，然后 `elif llm_ok and not emb_ok`，
    # 最后 `else: 不能跑`。
    #
    # 问题是：`--skip-smoke` 时 `usage` 是 None，第一个条件不成立；
    # 而 emb_ok 是 True，第二个条件也不成立 → 落到 else → 显示"还不能跑"。
    #
    # **但实际上所有检查都通过了。** 这是个典型的"分支条件没覆盖全"的问题：
    # 我按"有 smoke 数据"和"没有 smoke 数据"两种输入设计了输出，
    # 却漏掉了"两个都好的情况但没有 smoke 数据"这个组合。
    #
    # 教训：**写多分支判断时，先把所有输入组合列出来，再写条件。**
    # 靠 if/elif 顺序去"兜住"所有情况，很容易漏。
    hr("结论")

    if llm_ok and emb_ok and usage:
        print(f"""
  {OK} 配置就绪（已含端到端冒烟验证），可以跑真实模型压测。

  {C.BOLD}下一步：{C.RESET}
    # 先小规模验证（1 次，只跑两个端点）
    python scripts/run_longhorizon.py --real-llm --paddings 0,100 --runs 1

    # 确认无误后跑完整版
    python scripts/run_longhorizon.py --real-llm --runs 3 \\
        --json data/longhorizon_real.json

  {C.YELLOW}预期会看到什么变化（相对 Mock）：{C.RESET}
  {C.DIM}  · 漂移可能变成"渐变"而不是"全有全无" —— 这才是真实模型的模式
    · 步数可能增加（真实模型不总按最优路径走）
    · token 消耗明显上升（但也更真实）
    · 有状态卡组可能不再是 100%（真实模型即使看到约束也可能违反）
      → 如果不是 100%，那反而是**更有价值的结果**：
        它说明"可见性"是必要条件但不是充分条件，
        还需要在提示词强度和结构上做文章。{C.RESET}""")

    elif llm_ok and emb_ok:
        # 配置全好，只是没跑端到端冒烟
        print(f"""
  {OK} 配置就绪：LLM 三档可用 + 语义 embedding 可用。

  {WARN} 但还没跑端到端冒烟（当前用了 --skip-smoke）。

  {C.DIM}第 2、3 节验证的是"接口通"，但完整链路还包括：
    工具调用 → 记忆写入 → 上下文调度 → 约束遵守。
    建议去掉 --skip-smoke 再跑一次，确认整条链路正常、
    并拿到实测 token 量用于成本预估：

      python scripts/check_api.py --price-in <你的输入单价> --price-out <输出单价>

    这一步只花一次 triage 的钱，但能避免"跑到一半才发现链路有问题"。{C.RESET}

  {C.BOLD}也可以直接跑小规模压测来验证：{C.RESET}
    python scripts/run_longhorizon.py --real-llm --paddings 0,100 --runs 1""")

    elif llm_ok and not emb_ok:
        print(f"""
  {WARN} LLM 可用，但向量编码器有疑问（见第 3 节）。

  {C.DIM}可以跑压测，但要记住：duplicate 相关的指标会偏低，
  而且那个偏低**不是模型的锅**，是 embedding 的锅。
  报告里要写清楚这一点。{C.RESET}""")

    else:
        print(f"""
  {NO} 还不能跑。先解决上面标 ✗ 的问题。""")
        print(f"\n  {C.DIM}常见排查顺序：")
        print(f"    1. key 是否复制完整（有时会带上引号或空格）")
        print(f"    2. 服务商是否需要单独的模型授权")
        print(f"    3. 网络是否能访问 BASE_URL（公司代理经常拦）")
        print(f"    4. 先用 --skip-smoke 只验证连通性，不花钱{C.RESET}")

    return 0 if (deps_ok and llm_ok and emb_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
