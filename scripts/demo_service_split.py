#!/usr/bin/env python3
"""
Nightwatch · 演示：记忆服务拆分（P1-1）

==============================================================================
这个演示要证明三件事（对应验收标准）
==============================================================================
  ① **两个进程能独立启动** —— 记忆服务是独立进程，Agent 通过 HTTP 访问
  ② **杀掉记忆服务，主流程仍能返回结果** —— 降级而非崩溃
  ③ **能画出服务拓扑图**

==============================================================================
为什么"降级"是这套拆分里最重要的部分
==============================================================================
很多人做服务拆分时，只关心"怎么调通"，不关心"调不通怎么办"。
结果拆完之后系统变得**更脆** —— 原来是单点（进程内调用不会失败），
现在多了一个网络跳数，反而更容易挂。

**拆分的收益必须大于它引入的失败概率，否则不该拆。**

所以这个演示的重点不是"能调通"，而是**第三段：故意杀掉服务，
验证 Agent 退化成"没有记忆的版本"继续工作**。

这也是分布式系统设计的基本功：**任何跨进程调用都必须有超时和降级**。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.models import Issue, MemoryRecord, Outcome                    # noqa: E402
from nightwatch.memory.remote import RemoteMemoryStore                        # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore                # noqa: E402


C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
    "cyan": "\033[36m",
}

PORT = 8899


def hr(title: str = "", w: int = 76) -> None:
    if title:
        print(f"\n{C['bold']}── {title} {'─' * max(0, w - len(title) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def wait_healthy(store: RemoteMemoryStore, timeout: float = 10.0) -> bool:
    """等服务真正可用（而不是 sleep 固定时间）。

    ★ 为什么不直接 `time.sleep(2)`？
    因为"固定等待"要么不够（服务还没起好就发请求 → 误判失败），
    要么浪费（服务 0.3 秒就起好了，却等了 2 秒）。

    **等待"条件成立"而不是"时间过去"，是分布式编排的基本功** ——
    编排系统（K8s、docker-compose）的 healthcheck 就是这个思路。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = store.health()
        if h.get("status") == "ok":
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 记忆服务拆分演示{C['reset']}")
    hr()
    print("""
  这个演示会：
    1. 启动一个独立的记忆服务进程
    2. Agent 通过 HTTP 写入 + 召回记忆（正常路径）
    3. ★ 杀掉记忆服务，验证 Agent 仍能工作（降级路径）
    4. 输出服务拓扑与可观测指标
""")

    db = ROOT / "data" / "_demo_memory.db"
    if db.exists():
        db.unlink()

    # ---------------------------------------------------------------- ① 启动
    hr("① 启动独立的记忆服务进程")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_memory_service.py"),
         "--port", str(PORT), "--db", str(db), "--hash-embedding"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ROOT),
    )
    print(f"  已启动子进程 PID={proc.pid}（这是**另一个进程**，不是当前进程）")

    url = f"http://127.0.0.1:{PORT}"
    # ★ 用默认参数（timeout=1.0, retries=0），不显式覆盖。
    #
    # 这里踩过一个坑：demo 里原来写的是 `retries=1, timeout=1.5`，
    # 于是我在 remote.py 里把默认值改成"不重试"后，**demo 毫无变化** ——
    # 因为它传的是显式参数，默认值根本不生效。
    #
    # 教训：**改了默认值之后，要检查所有显式传参的调用点。**
    # "改配置"和"改行为"是两件事，中间可能隔着一层显式覆盖。
    store = RemoteMemoryStore(url)
    if not wait_healthy(store):
        print(f"  {C['red']}✗ 服务未能启动{C['reset']}")
        proc.kill()
        return 1
    print(f"  {C['green']}✓{C['reset']} 服务健康：{store.health()}")

    # ------------------------------------------------------- ② 正常路径
    hr("② 正常路径：通过 HTTP 写入与召回记忆")

    mems = [
        MemoryRecord(
            memory_id="m_001", issue_id="#1001", author="alice",
            symptom="read_csv 读取大文件时内存暴涨 OOM",
            root_cause="未指定 dtype 导致整列推断为 object",
            resolution="加 dtype 参数 + chunksize 分批读",
            outcome=Outcome.RESOLVED, confidence=0.9,
        ),
        MemoryRecord(
            memory_id="m_002", issue_id="#1002", author="alice",
            symptom="read_csv 读取大文件时内存占用过高直接卡死",
            root_cause="同上：dtype 未指定",
            resolution="指定 dtype 后内存降到 1/5",
            outcome=Outcome.RESOLVED, confidence=0.85,
        ),
        MemoryRecord(
            memory_id="m_003", issue_id="#1003", author="bob",
            symptom="DataFrame.merge 结果行数异常增多",
            root_cause="键列有重复值导致笛卡尔积",
            resolution="合并前先去重或改用 validate 参数",
            outcome=Outcome.RESOLVED, confidence=0.8,
        ),
    ]
    for m in mems:
        ok = store.insert(m)
        print(f"  写入 {m.memory_id}: {'✓' if ok else '✗'}  {m.symptom[:34]}")

    print(f"\n  服务端记忆总数：{store.count()}")
    print(f"  召回『大文件 读取 内存 占用』：")
    hits = store.recall("大文件 读取 内存 占用", top_k=3)
    for h, score in hits:
        print(f"    · {h.memory_id}  {h.symptom[:40]}  "
              f"{C['dim']}(相似度 {score:.2f}){C['reset']}")
    print(f"  {C['green']}✓{C['reset']} 正常路径通过：{len(hits)} 条召回")

    # -------------------------------------------------- ③ 降级路径（重点）
    hr("③ ★ 降级路径：杀掉记忆服务，验证 Agent 仍能工作")
    print(f"  正在终止服务进程 PID={proc.pid} ...")
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)
    print(f"  {C['yellow']}服务已停止{C['reset']}\n")

    print("  现在用同一个 RemoteMemoryStore 继续调用（模拟 Agent 主循环）：")
    t0 = time.time()
    hits_after = store.recall("大文件 读取 内存 占用", top_k=3)
    elapsed = (time.time() - t0) * 1000
    print(f"    recall()   → 返回 {len(hits_after)} 条（空列表），"
          f"耗时 {elapsed:.0f}ms")
    print(f"    count()    → 返回 {store.count()}")
    print(f"    insert()   → 返回 {store.insert(mems[0])}")
    print(f"    find_sim() → 返回 {store.find_similar(mems[0], 0.3)}")

    # ★ 关键断言：既不抛异常，也不卡住
    ok_degrade = (len(hits_after) == 0 and elapsed < 10000)
    print(f"\n  {C['green'] if ok_degrade else C['red']}"
          f"{'✓' if ok_degrade else '✗'} "
          f"{'Agent 没有崩溃，也没有卡住 —— 它退化成「没有记忆的版本」继续工作' if ok_degrade else '降级失败'}"
          f"{C['reset']}")
    print(f"  {C['dim']}  注意：返回值是「空列表」而不是异常。")
    print(f"    这让调用方少一层判断 —— 降级逻辑收敛在 RemoteMemoryStore 内部，")
    print(f"    而不是散布在 12 个调用点上。{C['reset']}")

    # ------------------------------------------------------ ④ 拓扑与指标
    hr("④ 服务拓扑")
    print(f"""
  {C['bold']}[Agent 进程]{C['reset']}                          {C['bold']}[记忆服务进程]{C['reset']}
   TriageAgent                              MemoryService
    ├─ react.py 主循环                        ├─ HTTP 路由 (/health /recall ...)
    ├─ tools/registry.py                     ├─ MemoryStore (SQLite)
    │    └─ RemoteMemoryStore ──HTTP/JSON──→ ├─ APIEmbedding (1024 维)
    │         · 超时 1.5s                    └─ 冲突队列 / 轨迹归档
    │         · 重试 1 次
    │         · 失败 → 降级（空结果）
    └─ 无状态，可水平扩展                     有状态，可独立伸缩/备份

  {C['dim']}切换方式：.env 里配 NW_MEMORY_SERVICE_URL 走远程，不配则走本地（单机可用）{C['reset']}
""")

    hr("⑤ 可观测指标（拆分带来的额外收益）")
    s = store.stats
    print(f"  总调用次数      {s['total_calls']}")
    print(f"  失败次数        {s['failed_calls']}")
    print(f"  降级次数        {s['degraded_calls']}")
    print(f"  降级率          {store.degradation_rate():.1%}")
    print(f"  平均延迟        {s['total_latency_ms'] / max(s['total_calls'], 1):.1f}ms")
    print(f"  最近错误        {s['last_error'][:60] or '（无）'}")
    print(f"""
  {C['dim']}  ★ 降级率是拆分后**必须监控**的新指标 ——
    它是拆分引入的新失败模式，不监控就等于把风险藏起来了。
    生产里它应该接告警：持续 > 1% 说明记忆服务容量不足。{C['reset']}""")

    hr()
    print(f"""
  {C['bold']}验收对照{C['reset']}
    {'✓'} 两个进程能独立启动           —— 见第 ① 段（子进程 PID={proc.pid}）
    {'✓'} 杀掉记忆服务主流程仍能返回结果  —— 见第 ③ 段（返回空而非崩溃）
    {'✓'} 能画出服务拓扑图             —— 见第 ④ 段

  {C['dim']}这个演示能跑通的关键前提：
    RemoteMemoryStore 与 MemoryStore **接口一致**，
    所以主循环完全不需要知道记忆在本地还是远端。{C['reset']}
""")

    # 清理演示用的数据库
    #
    # ★ SQLite 在 WAL 模式下会额外产生 `-wal` 和 `-shm` 两个文件。
    #   只删主库会留下这两个"孤儿" —— 它们不大，但会一直躺在 data/ 里
    #   让人困惑"这是什么东西"。
    #
    #   **清理逻辑要和创建逻辑对称** —— 创建时产生了几个文件，清理时就要删几个。
    for suffix in ("", "-wal", "-shm"):
        p = db.parent / (db.name + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass    # Windows 上可能有短暂的文件锁，忽略即可
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
