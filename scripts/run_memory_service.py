#!/usr/bin/env python3
"""
Nightwatch · 启动记忆服务（独立进程）

    python scripts/run_memory_service.py                          # 默认 127.0.0.1:8765
    python scripts/run_memory_service.py --port 9000 --db data/mem.db
    python scripts/run_memory_service.py --host 0.0.0.0           # ⚠ 需要自行加鉴权

启动后 Agent 侧配置（.env）：
    NW_MEMORY_SERVICE_URL=http://127.0.0.1:8765

==============================================================================
这个脚本存在的意义：证明"两个进程能独立启动"
==============================================================================
它不 import 任何 Agent 主循环的东西 —— 只依赖 `nightwatch.memory`。
**依赖边界就是服务边界**：如果它能被单独 import 和启动，
说明这块代码真的解耦了。反之，如果启动它需要拖上半个项目，
那"服务拆分"就是假的。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch import config as _config                       # noqa: F401,E402
from nightwatch.memory.service import make_server, serve_forever  # noqa: E402
from nightwatch.memory.store import APIEmbedding, HashEmbedding, MemoryStore  # noqa: E402


def build_encoder(use_api: bool = True):
    """构造向量编码器。

    ★ 编码器必须和服务端**保持一致**。

    这是个容易忽略的坑：如果记忆是用 A 编码器写入的，
    却用 B 编码器读取（算相似度），那检索结果就是**噪声** ——
    而且不会报错，只是"召回不准"。

    **向量编码器是记忆库的一部分，不是调用方的参数。**
    所以它应该由服务端决定，客户端只管发文本。

    （这也是为什么远程客户端的接口里没有 encoder 参数 ——
      那是本地 MemoryStore 才需要的。）
    """
    if use_api:
        try:
            enc = APIEmbedding()
            # 探一次，确认真的可用（APIEmbedding 会静默降级，必须主动验证）
            vec = enc.encode("探针")
            if len(vec) > 256:
                print(f"  ✓ 使用 APIEmbedding（{len(vec)} 维语义向量）")
                return enc
            print(f"  ⚠ APIEmbedding 返回 {len(vec)} 维，疑似降级，改用哈希版")
        except Exception as e:                              # noqa: BLE001
            print(f"  ⚠ APIEmbedding 不可用（{type(e).__name__}），改用哈希版")
            print(f"    提示：未配置 NW_LLM_API_KEY 时这是正常的")

    print("  · 使用 HashEmbedding（256 维哈希伪向量，仅用于跑通链路）")
    return HashEmbedding(dim=256)


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 Nightwatch 记忆服务")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址（默认 127.0.0.1；0.0.0.0 需自行加鉴权）")
    ap.add_argument("--port", type=int, default=8765, help="端口（默认 8765）")
    ap.add_argument("--db", default=str(ROOT / "data" / "memory.db"),
                    help="SQLite 路径（默认 data/memory.db）")
    ap.add_argument("--hash-embedding", action="store_true",
                    help="强制用哈希向量（不调 API，用于离线演示）")
    args = ap.parse_args()

    print("\nNightwatch · 记忆服务")
    print("─" * 62)
    print(f"  数据库      {args.db}")
    print(f"  监听        {args.host}:{args.port}")

    encoder = build_encoder(use_api=not args.hash_embedding)

    # 确保目录存在（服务自己负责自己的存储）
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(args.db, encoder=encoder)
    server, service = make_server(store, args.host, args.port)

    if not service.db_ok:
        print(f"\n  ✗ 数据库不可用：{service.db_error}")
        print("    服务仍会启动，但所有请求会返回 5xx。")
        print("    这样客户端能立刻降级，而不是等超时。")
        print("    （**尽早失败**比启动时报错退出更友好 —— 因为")
        print("      启动失败会让整个编排系统反复重启，反而更难排查。）\n")

    serve_forever(server, service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
