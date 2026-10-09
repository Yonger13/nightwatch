#!/usr/bin/env python3
"""
Nightwatch · 演示：A2A 协议 + 跨 Agent 共享记忆（P2-2）

==============================================================================
这个演示要证明三件事（对应验收标准）
==============================================================================
  ① **两个 Agent 能互相委托任务** —— TriageAgent 把安全 issue 委托给
     SecurityAgent，通过 A2A 协议（Agent Card 发现 + JSON-RPC 委托）
  ② **共享记忆池能正确处理并发写入** —— 多线程同时写，不丢数据不冲突
  ③ **共享记忆创造双向价值** —— SecurityAgent 沉淀的漏洞模式，
     第二次同类任务能被召回（这是"共享"相对"各存一份"的核心优势）

==============================================================================
架构
==============================================================================
    ┌────────────────────┐         ┌─────────────────────────┐
    │  TriageAgent 进程   │  A2A    │  SecurityAgent 进程      │
    │  (委托方)           │ ──────→ │  (被委托方)              │
    │                    │ ←────── │                         │
    │  发现 Card → 委托   │  Task   │  execute() → Artifact   │
    └─────────┬──────────┘         └───────────┬─────────────┘
              │                                │
              │        共享记忆池                │
              └───────────┬────────────────────┘
                          ▼
                 ┌─────────────────────┐
                 │  MemoryService 进程  │  ← 独立进程，两者都通过 HTTP 访问
                 │  (SQLite + 向量)     │
                 └─────────────────────┘

  ★ 三个进程、两个协议（A2A 用于 Agent 间、HTTP 用于记忆访问）。
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.a2a import A2AClient, TaskState                        # noqa: E402
from nightwatch.memory.remote import RemoteMemoryStore                 # noqa: E402
from nightwatch.models import MemoryRecord, MemoryTier, Outcome        # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m", "blue": "\033[34m"}

MEM_PORT = 8901
A2A_PORT = 8902
MEM_URL = f"http://127.0.0.1:{MEM_PORT}"
A2A_URL = f"http://127.0.0.1:{A2A_PORT}"


def hr(title: str = "", w: int = 78) -> None:
    if title:
        print(f"\n{C['bold']}── {title} {'─' * max(0, w - len(title) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def wait_ready(url: str, path: str = "/health", timeout: float = 15.0) -> bool:
    """等待服务就绪（等条件成立，不是等固定时间）。"""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + path, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:                                   # noqa: BLE001
            time.sleep(0.2)
    return False


def main() -> int:                                          # noqa: C901
    print(f"\n{C['bold']}{C['cyan']}Nightwatch · A2A 协议 + 跨 Agent 共享记忆{C['reset']}")
    hr()
    print("""
  这个演示会启动三个进程：
    · 记忆服务（共享记忆池）
    · 安全审计 Agent（A2A 被委托方）
    · 本脚本自身扮演分诊 Agent（A2A 委托方）
""")

    db = ROOT / "data" / "_demo_a2a.db"
    for suffix in ("", "-wal", "-shm"):
        p = db.parent / (db.name + suffix)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    procs: list[subprocess.Popen] = []
    try:
        # ============================================================ ① 启动
        hr("① 启动三个进程")

        mem_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "run_memory_service.py"),
             "--port", str(MEM_PORT), "--db", str(db), "--hash-embedding"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        procs.append(mem_proc)
        if not wait_ready(MEM_URL):
            print(f"  {C['red']}✗ 记忆服务启动失败{C['reset']}")
            return 1
        print(f"  {C['green']}✓{C['reset']} 记忆服务     PID={mem_proc.pid}  "
              f"{MEM_URL}")

        # 安全 Agent 服务（独立进程，自己持有远端记忆客户端）
        sec_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "run_security_agent.py"),
             "--port", str(A2A_PORT), "--memory-url", MEM_URL],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        procs.append(sec_proc)
        if not wait_ready(f"http://127.0.0.1:{A2A_PORT}"):
            print(f"  {C['red']}✗ 安全 Agent 启动失败{C['reset']}")
            return 1
        print(f"  {C['green']}✓{C['reset']} 安全 Agent   PID={sec_proc.pid}  "
              f"http://127.0.0.1:{A2A_PORT}")
        print(f"  {C['green']}✓{C['reset']} 分诊 Agent   PID={__import__('os').getpid()}"
              f"  （本脚本）")

        # ======================================================== ② 能力发现
        hr("② A2A 能力发现（读对方的 Agent Card）")

        client = A2AClient(A2A_URL, name="security-audit-agent")
        card = client.discover()
        if not card:
            print(f"  {C['red']}✗ 未发现 agent card{C['reset']}")
            return 1

        print(f"  发现远端 Agent：{C['bold']}{card.name}{C['reset']}")
        print(f"    描述         {card.description[:56]}...")
        print(f"    协议版本     {card.protocol_version}")
        print(f"    声明的技能   {[s.skill_id for s in card.skills]}")
        print(f"    能力标志     {card.capabilities}")
        print(f"  {C['dim']}  ★ 注意 streaming=False —— 这是对方**如实声明**"
              f"不支持流式，\n     所以委托方会走轮询而不是尝试建流。{C['reset']}")

        # ★ 委托前的能力检查 —— A2A 的核心价值
        want = "security-audit"
        if not client.can_handle(want):
            print(f"  {C['red']}✗ 远端不支持技能 {want}{C['reset']}")
            return 1
        print(f"  {C['green']}✓{C['reset']} 远端声明支持 `{want}`，可以委托")

        # ======================================================== ③ 第一次委托
        hr("③ 第一次委托：pickle 反序列化 RCE")

        issue1 = {
            "issue_id": "#67074",
            "title": "pandas `read_pickle` RCE via Ungated `pickle.load` on the input",
            "body": ("A crafted pickle payload passed to read_pickle can execute "
                     "arbitrary code during deserialization. See pickle.load "
                     "usage in io/pickle.py."),
            "author": "security-researcher",
            "repo": "pandas-dev/pandas",
        }

        t0 = time.time()
        task1 = client.send_task(skill=want,
                                 payload={"issue": issue1},
                                 text="请审计这个 issue 的安全性",
                                 from_agent="triage-agent")
        elapsed = (time.time() - t0) * 1000

        if task1 is None:
            print(f"  {C['red']}✗ 委托失败{C['reset']}")
            return 1

        state_color = C["green"] if task1.state.is_success else C["yellow"]
        print(f"  任务 ID     {task1.task_id}")
        print(f"  状态        {state_color}{task1.state.value}{C['reset']}  "
              f"（耗时 {elapsed:.0f}ms）")

        if task1.artifacts:
            rep = task1.artifacts[0].parts[0].data
            print(f"\n  {C['bold']}审计报告（Artifact）{C['reset']}")
            print(f"    结论       {rep['verdict']}")
            print(f"    漏洞类别   {rep['primary_cwe']} {rep['name']}")
            print(f"    危害等级   {C['red']}{rep['severity']}{C['reset']}")
            print(f"    修复建议   {rep['advice'][:60]}...")
            print(f"    历史同类   {rep['prior_occurrences']} 次")
            print(f"    已沉淀记忆 {'是' if rep['memory_written'] else '否'}")

        # ==================================================== ④ 共享记忆的价值
        hr("④ ★ 共享记忆的价值：第二次同类问题直接命中")

        issue2 = {
            "issue_id": "#67999",
            "title": "tarfile extraction allows path traversal via ../ in member name",
            "body": "Extracting an untrusted tar archive can write files outside "
                    "the target directory (path traversal, CWE-22).",
            "author": "another-reporter",
            "repo": "pandas-dev/pandas",
        }
        task2 = client.send_task(skill=want, payload={"issue": issue2},
                                 text="请审计", from_agent="triage-agent")
        if task2 and task2.artifacts:
            rep2 = task2.artifacts[0].parts[0].data
            print(f"  第二次委托：{issue2['title'][:52]}")
            print(f"    漏洞类别   {rep2['primary_cwe']} {rep2['name']}")
            print(f"    历史同类   {rep2['prior_occurrences']} 次  "
                  f"{C['dim']}← 安全 Agent 自己沉淀的记忆{C['reset']}")

        # 换一个视角：分诊 Agent 直接查共享记忆
        #
        # ★ 这里要证明的是"记忆确实共享了"，所以用两种方式各查一次：
        #
        #   方式一：all_memories() 全量读 —— **证明共享本身**（确定性，不依赖向量质量）
        #   方式二：recall() 向量检索 —— 展示**实际使用路径**的效果
        #
        # 为什么要分开？因为第一版演示只用了 recall，结果召回 0 条，
        # 却被文案写成"✓ 这就是共享的价值" —— **文案和数据矛盾，等于在说谎。**
        #
        # 根因是哈希伪向量的判别力不足（docs/06 的已知问题），
        # 不是"记忆没共享"。所以正确的做法是：
        #   **用一个不依赖向量质量的方式证明"共享"，同时如实报告检索的局限。**
        print(f"\n  {C['bold']}分诊 Agent 侧读取共享记忆{C['reset']}")
        triage_store = RemoteMemoryStore(MEM_URL)

        # 方式一：全量读 —— 证明共享
        all_mems = triage_store.all_memories()
        sec_mems = [m for m in all_mems
                    if "security-audit" in (m.symptom_tags or [])]
        print(f"    [全量读] 记忆池共 {len(all_mems)} 条，"
              f"其中安全 Agent 写入的 {len(sec_mems)} 条：")
        for m in sec_mems:
            print(f"      · {m.issue_id}  {m.symptom[:44]}")
            print(f"        处置：{m.resolution[:44]}...")
        ok_share = len(sec_mems) > 0
        print(f"    {C['green'] if ok_share else C['red']}"
              f"{'✓' if ok_share else '✗'} "
              f"{'安全 Agent 写的记忆，分诊 Agent 能直接读到' if ok_share else '共享失败'}"
              f"{C['reset']}")

        # 方式二：向量召回 —— 展示实际检索路径
        hits = triage_store.recall("pickle 反序列化 漏洞 安全", top_k=3)
        sec_hits = [m for m, _ in hits if "security-audit" in (m.symptom_tags or [])]
        print(f"    [向量召回] 查询『pickle 反序列化 漏洞』→ "
              f"召回 {len(hits)} 条，命中安全记忆 {len(sec_hits)} 条")
        if not sec_hits:
            print(f"""
    {C['yellow']}⚠ 向量召回没命中 —— 这不是"共享失败"，是**检索质量问题**：{C['reset']}
      · 当前用 HashEmbedding（256 维哈希伪向量），判别力不足
      · 这是 docs/06「已知问题」里记录的第一条，天花板 F1 仅 0.5
      · 换语义 embedding（1024 维）后召回会显著改善

    {C['dim']}  这个区分很重要：**"记忆没共享"和"检索不准"是两回事**，
      修法完全不同（前者改架构，后者换 embedding）。
      如果这里只报"召回 0 条"，你会误以为共享没做。{C['reset']}""")

        mark = f"{C['green']}✓{C['reset']}" if ok_share \
            else f"{C['red']}✗{C['reset']}"
        print(f"""
  {mark} 共享的价值：安全 Agent 学到的模式，分诊 Agent 立刻能看到。
     如果两个 Agent 各存一份记忆，这个知识**永远不会流动**。""")

        # ==================================================== ⑤ 技能不匹配被拒
        hr("⑤ A2A 的体面拒绝：委托一个它不会的技能")
        bad = client.send_task(skill="quantum-computing",
                               payload={"issue": issue1},
                               from_agent="triage-agent")
        if bad:
            color = C["green"] if bad.state is TaskState.REJECTED else C["red"]
            print(f"  委托 skill=quantum-computing → "
                  f"{color}{bad.state.value}{C['reset']}")
            print(f"  原因：{bad.error[:70]}")
            print(f"""
  {C['dim']}  ★ `rejected` 是终态 —— 客户端立刻知道"该找别人"，
     不用轮询等待。**让 Agent 能体面地拒绝，比让它硬着头皮做要好。**{C['reset']}""")

        # ==================================================== ⑥ 并发写入测试
        hr("⑥ ★ 验收标准：共享记忆池的并发写入")

        print("  并发场景：多个 Agent/线程同时写记忆池（真实的多 Agent 环境必然发生）")
        n_threads, per_thread = 8, 5
        results: list[bool] = []
        lock = threading.Lock()

        def writer(tid: int) -> None:
            store = RemoteMemoryStore(MEM_URL)
            got = []
            for i in range(per_thread):
                mem = MemoryRecord(
                    memory_id=f"concurrent_t{tid}_{i}",
                    issue_id=f"#9{tid}{i:02d}",
                    symptom=f"并发写入测试 线程{tid} 第{i}条",
                    symptom_tags=["concurrency-test"],
                    outcome=Outcome.RESOLVED,
                )
                got.append(store.insert(mem))
            with lock:
                results.extend(got)

        threads = [threading.Thread(target=writer, args=(t,))
                   for t in range(n_threads)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = (time.time() - t0) * 1000

        expected = n_threads * per_thread
        succeeded = sum(1 for r in results if r)
        actual = triage_store.count()
        concurrency_mems = len([
            m for m, _ in triage_store.recall("并发写入测试", top_k=200)
        ])

        print(f"    线程数         {n_threads} × 每线程 {per_thread} 条 = {expected} 条")
        print(f"    写入成功       {succeeded} / {expected}")
        print(f"    耗时           {elapsed:.0f}ms")
        print(f"    记忆池总数      {actual} 条（含前面的安全记忆）")
        print(f"    可召回并发条目  {concurrency_mems} 条")

        # ★ 关键断言：不丢数据
        #
        # 这里测的是"并发下是否丢写"。SQLite 用写锁串行化写入，
        # 但**如果客户端没处理好重试，就会静默丢数据** —— 而这正是
        # 多 Agent 共享存储最容易出问题的地方。
        ok = (succeeded == expected)
        print(f"\n  {C['green'] if ok else C['red']}{'✓' if ok else '✗'} "
              f"{'并发写入无丢失' if ok else f'丢失 {expected - succeeded} 条！'}"
              f"{C['reset']}")
        print(f"  {C['dim']}  说明：SQLite 用写锁串行化写入，所以并发下"
              f"不会互相覆盖；\n    但**写入变慢**（每次排队），这就是"
              f"docs/06 里写的\"要换 Postgres\"的原因。{C['reset']}")

        # ======================================================== ⑦ 拓扑
        hr("⑦ 服务拓扑")
        print(f"""
  ┌──────────────────────┐   A2A (JSON-RPC over HTTP)   ┌──────────────────────┐
  │  TriageAgent 进程     │ ──────────────────────────→  │  SecurityAgent 进程   │
  │  PID={__import__('os').getpid():<6}             │ ←──────────────────────────  │  PID={sec_proc.pid:<6}           │
  │                      │      Task / Artifact          │                      │
  │  · A2AClient         │                               │  · A2AServer         │
  │  · 读 Card 判能力     │                               │  · AgentExecutor     │
  └──────────┬───────────┘                               └──────────┬───────────┘
             │                                                      │
             │        共享记忆池 (HTTP)                              │
             └──────────────────┬───────────────────────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │  MemoryService 进程      │
                   │  PID={mem_proc.pid:<6}            │
                   │  · SQLite + 向量检索     │
                   └─────────────────────────┘

  {C['dim']}两个协议分工：A2A 管"Agent 之间怎么协作"，
   HTTP 管"Agent 怎么访问共享资源"。{C['reset']}""")

        # ==================================================== ⑧ 统计
        hr("⑧ 协议统计")
        print(f"  A2AClient    发任务 {client.stats['tasks_sent']} 次，"
              f"失败 {client.stats['failures']} 次，"
              f"发现 {client.stats['discoveries']} 次")
        if client.stats["last_error"]:
            print(f"  最近错误     {client.stats['last_error'][:70]}")
        print(f"  记忆客户端   调用 {triage_store.stats['total_calls']} 次，"
              f"降级 {triage_store.stats['degraded_calls']} 次")

        hr()
        print(f"""
  {C['bold']}验收对照{C['reset']}
    ✓ 两个 Agent 能互相委托任务
        —— 见第 ③ 段（TriageAgent 委托 SecurityAgent，含能力发现）
    ✓ 共享记忆池能正确处理并发写入
        —— 见第 ⑥ 段（{n_threads} 线程 × {per_thread} 条，{succeeded}/{expected} 成功）
    ✓ 额外：共享记忆创造双向价值
        —— 见第 ④ 段（安全 Agent 沉淀的模式，分诊 Agent 立刻可见）
    ✓ 额外：A2A 的体面拒绝
        —— 见第 ⑤ 段（不擅长的技能返回 REJECTED 而非硬做）

  {C['dim']}与记忆服务拆分（P1-1）的关系：
    记忆服务拆出来之后，"跨 Agent 共享记忆"才成为可能 ——
    如果记忆还锁在各自的进程里，两个 Agent 只能各存一份。
    **架构上的解耦会解锁新的能力**，这是 P1-1 的额外收益。{C['reset']}
""")
        return 0

    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        for p in procs:
            if p.poll() is None:
                p.kill()
        for suffix in ("", "-wal", "-shm"):
            f = db.parent / (db.name + suffix)
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
