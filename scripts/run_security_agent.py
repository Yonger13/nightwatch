#!/usr/bin/env python3
"""
Nightwatch · 启动安全审计 Agent（A2A 服务）

    python scripts/run_security_agent.py --port 8902 \\
        --memory-url http://127.0.0.1:8765

==============================================================================
它同时是一个"被委托方"和一个"记忆服务的客户端"
==============================================================================
这个脚本启动的服务有两条依赖方向，值得说清：

    上游（谁调我）：  其他 Agent 通过 A2A 委托任务给我
    下游（我调谁）：  我通过 HTTP 访问共享记忆池

**一个 Agent 可以同时是服务端和客户端** —— 这正是 A2A 里
"peer-to-peer（对等）"的含义：没有"主 Agent"和"从 Agent"，
每个 Agent 既能接活也能派活。

（对比 MCP：MCP 里永远是"host 调 server"，是主从关系。
  A2A 是平级的，这是两个协议在设计哲学上的差别。）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch import config as _config                             # noqa: F401,E402
from nightwatch.a2a.server import make_a2a_server, serve_a2a          # noqa: E402
from nightwatch.agents.security import SecurityAuditAgent             # noqa: E402
from nightwatch.memory.remote import RemoteMemoryStore                # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="启动安全审计 Agent（A2A）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址（默认 127.0.0.1；0.0.0.0 需自行加鉴权）")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--memory-url", default="",
                    help="共享记忆服务地址；不传则用本地 SQLite（单机模式）")
    ap.add_argument("--db", default=str(ROOT / "data" / "security_agent.db"),
                    help="本地模式下的 SQLite 路径")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"

    print("\nNightwatch · 安全审计 Agent")
    print("─" * 62)
    print(f"  A2A 监听    {args.host}:{args.port}")

    # ★ 记忆存储的选择顺序：优先共享（远端），退化到本地
    #
    # 为什么允许退化到本地？
    # 因为"单机也能跑"对开发者很重要 —— 起一个进程就能调试，
    # 不用先拉起记忆服务。**降低上手门槛和上生产是两件事。**
    if args.memory_url:
        store = RemoteMemoryStore(args.memory_url)
        h = store.health()
        status = h.get("status", "unknown")
        print(f"  共享记忆    {args.memory_url}  (status={status})")
        if status != "ok":
            warn = _c("⚠")
            print(f"  {warn} 记忆服务不可用 —— Agent 会降级运行"
                  f"（无历史记忆，但审计功能正常）")
    else:
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(args.db, encoder=HashEmbedding(dim=256))
        print(f"  本地记忆    {args.db}  (单机模式)")

    executor = SecurityAuditAgent(store)
    card = SecurityAuditAgent.card(url)
    httpd, server_obj = make_a2a_server(card, executor, args.host, args.port)

    print(f"  发布的技能  {[s.skill_id for s in card.skills]}")
    print(f"  Agent Card  {url}/.well-known/agent-card.json")
    print(f"  {_c('说明：本服务无鉴权，默认只监听 127.0.0.1')}")

    serve_a2a(httpd, server_obj)
    return 0


def _c(t: str) -> str:
    return f"\033[33m{t}\033[0m"


if __name__ == "__main__":
    raise SystemExit(main())
