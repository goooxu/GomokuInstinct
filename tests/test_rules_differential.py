"""Python 参考实现与 C++ 实现的差分测试（CI 规模）。

完整规模的验收跑 scripts/diff_test_rules.py；这里跑一个几秒钟能完成的小规模版本，
用于日常回归。

两份实现刻意采用不同做法 —— Python 抽出定长列表逐格扫描，C++ 把每条线打包成
位掩码做窗口位运算 —— 做法不同，结果一致才有意义。
"""

from __future__ import annotations

import random

import pytest

from gomoku_instinct.core import load_core
from gomoku_instinct.rules import BLACK, EMPTY, WALL, WHITE, Board, RenjuRules
from gomoku_instinct.rules import constants as py_const
from gomoku_instinct.rules.generate import random_position, selfplay_positions
from gomoku_instinct.rules.symmetry import NUM_SYMMETRIES, index_map, transform_grid

SIZE = 15


@pytest.fixture(scope="module")
def core():
    return load_core()


@pytest.fixture(scope="module")
def cc_rules(core):
    return core.Rules()


@pytest.fixture(scope="module")
def py_rules():
    return RenjuRules()


def _flat_py(py_rules: RenjuRules, grid: bytearray, color: int) -> bytearray:
    board = Board(SIZE)
    board.grid[:] = grid
    out = bytearray(b"\xff" * (5 * SIZE * SIZE))
    for idx in range(SIZE * SIZE):
        if grid[idx] != EMPTY:
            continue
        r, c = divmod(idx, SIZE)
        j = py_rules.judge(board, r, c, color)
        out[5 * idx : 5 * idx + 5] = bytes(
            (int(j.outcome), int(j.forbidden), j.fours, j.open_threes, j.longest_run)
        )
    return out


def _render(grid: bytearray, idx: int) -> str:
    r, c = divmod(idx, SIZE)
    rows = []
    for rr in range(SIZE):
        cells = []
        for cc in range(SIZE):
            v = grid[rr * SIZE + cc]
            ch = "." if v == EMPTY else ("X" if v == BLACK else "O")
            cells.append("*" if (rr, cc) == (r, c) else ch)
        rows.append("".join(cells))
    return f"落点 ({r}, {c})\n" + "\n".join(rows)


# ── 编码一致性 ──────────────────────────────────────────────────────────────


def test_stone_encoding_matches(core):
    """两侧的棋子编码必须完全相同，否则后面的一致性都是假的。"""
    assert (core.EMPTY, core.BLACK, core.WHITE, core.WALL) == (EMPTY, BLACK, WHITE, WALL)


def test_direction_order_matches(core):
    assert tuple(tuple(d) for d in core.DIRECTIONS) == py_const.DIRECTIONS


# ── 差分比对 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["scatter", "cluster", "black_heavy"])
def test_differential_on_random_positions(cc_rules, py_rules, kind):
    rng = random.Random(hash(kind) & 0xFFFF)
    forbidden_seen = 0

    for _ in range(60):
        grid = random_position(rng, SIZE, kind)
        for color in (BLACK, WHITE):
            cc_out = cc_rules.judge_all(grid, SIZE, color)
            py_out = _flat_py(py_rules, grid, color)
            for idx in range(SIZE * SIZE):
                if grid[idx] != EMPTY:
                    continue
                a = bytes(py_out[5 * idx : 5 * idx + 5])
                b = bytes(cc_out[5 * idx : 5 * idx + 5])
                assert a == b, (
                    f"{_render(grid, idx)}\n色={color} python={list(a)} cpp={list(b)}"
                )
                if color == BLACK and a[1] != 0:
                    forbidden_seen += 1

    # 没压到禁手路径的话，「一致」这个结论没有说服力。
    assert forbidden_seen > 0, f"{kind} 分布没有产生任何禁手点"


def test_differential_on_selfplay_positions(cc_rules, py_rules):
    """随机合法对局中途的局面 —— 分布最接近真实自博弈。"""
    rng = random.Random(4242)
    positions = selfplay_positions(rng, SIZE, cc_rules, max_positions=60)
    assert positions

    for grid in positions:
        for color in (BLACK, WHITE):
            cc_out = cc_rules.judge_all(grid, SIZE, color)
            py_out = _flat_py(py_rules, grid, color)
            for idx in range(SIZE * SIZE):
                if grid[idx] != EMPTY:
                    continue
                a = bytes(py_out[5 * idx : 5 * idx + 5])
                b = bytes(cc_out[5 * idx : 5 * idx + 5])
                assert a == b, f"{_render(grid, idx)}\n色={color}"


# ── 与实现无关的不变性 ──────────────────────────────────────────────────────


def test_rules_are_invariant_under_dihedral_symmetry(cc_rules):
    """规则在棋盘的八重对称下必须完全不变。

    这条性质不依赖任何一份实现，因此能独立抓出「某个方向写错了」这类 bug ——
    比如把副对角当成主对角处理。
    """
    rng = random.Random(7)
    maps = [index_map(SIZE, t) for t in range(NUM_SYMMETRIES)]

    for _ in range(25):
        grid = random_position(rng, SIZE, "black_heavy")
        base = cc_rules.judge_all(grid, SIZE, BLACK)
        for t in range(1, NUM_SYMMETRIES):
            moved = cc_rules.judge_all(transform_grid(grid, SIZE, t), SIZE, BLACK)
            for idx in range(SIZE * SIZE):
                if grid[idx] != EMPTY:
                    continue
                j = maps[t][idx]
                assert base[5 * idx : 5 * idx + 5] == moved[5 * j : 5 * j + 5], (
                    f"对称变换 t={t} 下判定发生了变化\n{_render(grid, idx)}"
                )


def test_python_reference_is_also_symmetric(py_rules):
    """同一条不变性也要在 Python 参考实现上成立。"""
    rng = random.Random(11)
    maps = [index_map(SIZE, t) for t in range(NUM_SYMMETRIES)]

    for _ in range(6):
        grid = random_position(rng, SIZE, "black_heavy")
        base = _flat_py(py_rules, grid, BLACK)
        for t in range(1, NUM_SYMMETRIES):
            moved = _flat_py(py_rules, transform_grid(grid, SIZE, t), BLACK)
            for idx in range(SIZE * SIZE):
                if grid[idx] != EMPTY:
                    continue
                j = maps[t][idx]
                assert base[5 * idx : 5 * idx + 5] == moved[5 * j : 5 * j + 5]


# ── 递归深度审计 ────────────────────────────────────────────────────────────


def test_recursion_cap_is_never_reached(cc_rules, py_rules):
    """三三判定的递归深度上限不能被真的触发。

    实测在随机密集局面下递归会达到 11 层，所以上限必须显著大于 11；
    早期取 8 会让判定悄悄退化成近似而不报错。
    """
    cc_rules.reset_counters()
    rng = random.Random(99)
    for _ in range(150):
        grid = random_position(rng, SIZE, "black_heavy")
        cc_rules.judge_all(grid, SIZE, BLACK)

    assert cc_rules.depth_exceeded == 0
    assert cc_rules.max_depth < cc_rules_recursion_limit()
    assert py_rules.depth_exceeded == 0


def cc_rules_recursion_limit() -> int:
    return 64
