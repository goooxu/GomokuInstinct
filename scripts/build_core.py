#!/usr/bin/env python3
"""预编译 C++ 规则内核。

    python scripts/build_core.py

正式训练前跑一次，避免多个 actor 同时首次加载时互相等待编译锁。
"""

from __future__ import annotations

import sys
import time

from gomoku_instinct.core import load_core


def main() -> int:
    started = time.time()
    core = load_core(verbose=True)
    elapsed = time.time() - started

    print()
    print(f"编译完成，耗时 {elapsed:.1f}s")
    print(f"  EMPTY/BLACK/WHITE/WALL = {core.EMPTY}/{core.BLACK}/{core.WHITE}/{core.WALL}")
    print(f"  方向 = {core.DIRECTIONS}")

    # 冒烟测试：空棋盘上没有禁手点。
    size = 15
    grid = bytearray(size * size)
    rules = core.Rules()
    fmap = rules.forbidden_map(grid, size)
    assert len(fmap) == size * size and not any(fmap), "空棋盘不应有禁手点"
    print("  冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
