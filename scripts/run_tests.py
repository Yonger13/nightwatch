#!/usr/bin/env python3
"""
Nightwatch · 测试入口

用法：
    python scripts/run_tests.py          # 跑全部测试
    python scripts/run_tests.py -v       # 详细输出

为什么要有这个脚本，而不直接 `python -m unittest discover`？
因为 src-layout 的 import 路径需要处理（src 下的包不在默认搜索路径里）。
把它封装成一个脚本，学习者就不用关心 sys.path 的细节了。

（如果你更喜欢标准方式，也可以：
    set PYTHONPATH=src && python -m unittest discover -s tests -v ）
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    verbosity = 2 if "-v" in sys.argv else 1

    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py",
                            top_level_dir=str(ROOT))

    if suite.countTestCases() == 0:
        print("未发现测试用例，检查 tests/ 目录")
        return 1

    print(f"\n运行 {suite.countTestCases()} 个测试用例\n" + "=" * 70)
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=False)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print(f"✓ 全部通过（{result.testsRun} 个用例）")
        return 0

    print(f"✗ 失败 {len(result.failures)} 个，错误 {len(result.errors)} 个")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
