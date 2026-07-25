#!/usr/bin/env python3
"""规则引擎差分测试 —— M1 的验收门槛。

在大量随机局面上，逐点比对 Python 参考实现与 C++ 实现的四项判定
（合法性/禁手/胜负/棋型计数），任何一处不一致即失败。

    python scripts/diff_test_rules.py --positions 20000 --workers 128

两份实现的做法刻意不同：Python 抽出定长列表逐格扫描，C++ 把每条线打包成
位掩码做窗口位运算。做法不同才让「结果一致」有意义。

另外还做一项与实现无关的不变性检查：规则在棋盘的八重二面体对称下必须完全不变。
这条性质不依赖任何一份实现，能独立抓出「某个方向写错了」这类 bug。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.core import load_core, make_rules  # noqa: E402
from gomoku_instinct.rules import BLACK, WHITE, Board, RenjuRules  # noqa: E402
from gomoku_instinct.rules.generate import random_position  # noqa: E402
from gomoku_instinct.rules.symmetry import (  # noqa: E402
    NUM_SYMMETRIES,
    index_map,
    transform_grid,
)

FIELD_NAMES = ("outcome", "forbidden", "fours", "open_threes", "longest_run")


def _py_judge_all(py_rules: RenjuRules, grid: bytearray, size: int, color: int):
    """Python 侧的逐点判定，压成与 C++ 相同的扁平格式便于比对。"""
    board = Board(size)
    board.grid[:] = grid
    out = bytearray(b"\xff" * (5 * size * size))
    for idx in range(size * size):
        if grid[idx] != 0:
            continue
        r, c = divmod(idx, size)
        j = py_rules.judge(board, r, c, color)
        base = 5 * idx
        out[base + 0] = int(j.outcome)
        out[base + 1] = int(j.forbidden)
        out[base + 2] = j.fours
        out[base + 3] = j.open_threes
        out[base + 4] = j.longest_run
    return out


def _describe(grid: bytearray, size: int, idx: int) -> str:
    r, c = divmod(idx, size)
    rows = []
    for rr in range(size):
        line = []
        for cc in range(size):
            v = grid[rr * size + cc]
            ch = "." if v == 0 else ("X" if v == 1 else "O")
            if rr == r and cc == c:
                ch = "*"
            line.append(ch)
        rows.append("".join(line))
    return f"落点 ({r}, {c})\n" + "\n".join(rows)


def _check_symmetry(cc_rules, grid: bytearray, size: int, color: int, maps) -> list[str]:
    """规则必须在八重对称下不变。"""
    problems = []
    base = cc_rules.judge_all(grid, size, color)
    for t in range(1, NUM_SYMMETRIES):
        tg = transform_grid(grid, size, t)
        got = cc_rules.judge_all(tg, size, color)
        imap = maps[t]
        for idx in range(size * size):
            if grid[idx] != 0:
                continue
            a = base[5 * idx : 5 * idx + 5]
            b = got[5 * imap[idx] : 5 * imap[idx] + 5]
            if a != b:
                problems.append(
                    f"对称变换 t={t} 下判定不一致：{_describe(grid, size, idx)}\n"
                    f"  原局面 {list(a)} != 变换后 {list(b)}"
                )
                return problems
    return problems


def _worker(task):
    seed, n_positions, size, kind, check_symmetry = task
    rng = random.Random(seed)

    core = load_core()
    cc_rules = core.Rules()
    py_rules = RenjuRules()

    maps = [index_map(size, t) for t in range(NUM_SYMMETRIES)] if check_symmetry else None

    stats = Counter()
    problems: list[str] = []

    for _ in range(n_positions):
        grid = random_position(rng, size, kind)
        stats["positions"] += 1

        for color in (BLACK, WHITE):
            cc_out = cc_rules.judge_all(grid, size, color)
            py_out = _py_judge_all(py_rules, grid, size, color)

            for idx in range(size * size):
                if grid[idx] != 0:
                    continue
                stats["judgments"] += 1
                a = py_out[5 * idx : 5 * idx + 5]
                b = cc_out[5 * idx : 5 * idx + 5]
                if a != b:
                    if len(problems) < 3:
                        diff = [
                            f"{name}: py={x} cc={y}"
                            for name, x, y in zip(FIELD_NAMES, a, b)
                            if x != y
                        ]
                        problems.append(
                            f"{_describe(grid, size, idx)}\n"
                            f"  色={color} 不一致字段: {'; '.join(diff)}"
                        )
                    stats["mismatches"] += 1
                    continue

                if color == BLACK and a[1] != 0:
                    stats["forbidden_total"] += 1
                    stats[f"forbidden_{a[1]}"] += 1
                if a[0] == 1:
                    stats["black_wins"] += 1
                elif a[0] == 2 and a[1] == 0:
                    stats["white_wins"] += 1

        if check_symmetry and rng.random() < 0.05:
            sym = _check_symmetry(cc_rules, grid, size, BLACK, maps)
            stats["symmetry_checks"] += 1
            if sym:
                stats["symmetry_failures"] += 1
                problems.extend(sym[:1])

    stats["depth_exceeded"] += cc_rules.depth_exceeded
    stats["depth_exceeded"] += py_rules.depth_exceeded
    stats["max_depth_cc"] = max(stats["max_depth_cc"], cc_rules.max_depth)
    stats["max_depth_py"] = max(stats["max_depth_py"], py_rules.max_depth_seen)
    return stats, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=20000, help="随机局面总数")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0 表示自动")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument(
        "--kind",
        default="mixed",
        choices=["mixed", "scatter", "cluster", "black_heavy"],
        help="局面生成分布",
    )
    ap.add_argument("--no-symmetry", action="store_true", help="跳过对称不变性检查")
    args = ap.parse_args()

    workers = args.workers or min(len(os.sched_getaffinity(0)), 128)
    per_worker = max(1, args.positions // workers)
    tasks = [
        (args.seed + i, per_worker, args.size, args.kind, not args.no_symmetry)
        for i in range(workers)
    ]

    # 先在父进程里编译/加载好内核，fork 出来的子进程直接继承，避免重复编译。
    load_core()

    print(f"局面数 {per_worker * workers}  并行度 {workers}  棋盘 {args.size}x{args.size}  分布 {args.kind}")
    started = time.time()

    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, tasks)

    elapsed = time.time() - started

    total = Counter()
    problems: list[str] = []
    max_depth_cc = 0
    max_depth_py = 0
    for stats, probs in results:
        max_depth_cc = max(max_depth_cc, stats.pop("max_depth_cc", 0))
        max_depth_py = max(max_depth_py, stats.pop("max_depth_py", 0))
        total.update(stats)
        problems.extend(probs)

    print()
    print(f"耗时 {elapsed:.1f}s")
    print(f"局面数         {total['positions']:,}")
    print(f"逐点判定次数   {total['judgments']:,}  ({total['judgments'] / max(elapsed, 1e-9):,.0f}/s)")
    print(f"其中黑方禁手   {total['forbidden_total']:,}")
    print(f"  长连禁手     {total['forbidden_1']:,}")
    print(f"  四四禁手     {total['forbidden_2']:,}")
    print(f"  三三禁手     {total['forbidden_3']:,}")
    print(f"成五（黑/白）  {total['black_wins']:,} / {total['white_wins']:,}")
    if not args.no_symmetry:
        print(f"对称不变性抽检 {total['symmetry_checks']:,} 次，失败 {total['symmetry_failures']:,}")
    print(f"实际递归深度   C++ {max_depth_cc} / Python {max_depth_py}（上限 64）")
    print(f"递归封顶次数   {total['depth_exceeded']:,}")
    print()

    failed = False

    if total["mismatches"]:
        failed = True
        print(f"两份实现不一致 {total['mismatches']:,} 处，示例：")
        for p in problems[:3]:
            print()
            print(p)
    if total["symmetry_failures"]:
        failed = True
        print(f"对称不变性失败 {total['symmetry_failures']} 处")
    if total["depth_exceeded"]:
        failed = True
        print("触发了递归深度封顶 —— 需要复核三三判定的深度上限")

    # 门槛：必须真的压到禁手逻辑，否则「零不一致」没有意义。
    if total["forbidden_total"] < 1000:
        failed = True
        print(
            f"禁手点样本仅 {total['forbidden_total']} 个，不足以证明禁手路径被覆盖；"
            "请增大 --positions 或改用 --kind black_heavy"
        )

    if failed:
        print("\n差分测试未通过")
        return 1

    print("差分测试通过：两份独立实现在全部判定上完全一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
