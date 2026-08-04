"""随机开局采样器。竞技场与网页观战共用这一份。"""

from __future__ import annotations

import random

import pytest

from gomoku_instinct.eval.opening import (
    CENTER_REGION,
    OPENING_WINDOW,
    opening_moves,
    opening_window,
)


@pytest.mark.parametrize("size", [15, 9, 5, 3])
def test_window_stays_inside_the_central_region(size):
    """窗口整个落在中央区域内。棋盘小于 9 或 5 时自动收缩，不能越界。"""
    region = min(CENTER_REGION, size)
    window = min(OPENING_WINDOW, region)
    lo, hi = (size - region) // 2, (size - region) // 2 + region - 1
    rng = random.Random(0)
    for _ in range(200):
        pts = opening_window(size, rng)
        assert len(pts) == window * window
        rows = [p // size for p in pts]
        cols = [p % size for p in pts]
        assert lo <= min(rows) and max(rows) <= hi
        assert lo <= min(cols) and max(cols) <= hi
        assert max(rows) - min(rows) == window - 1
        assert max(cols) - min(cols) == window - 1


def test_moves_are_distinct_and_share_one_window():
    """一局只取**一个**窗口 —— 散开落等于没开局。"""
    rng = random.Random(1)
    for _ in range(300):
        moves = opening_moves(15, 4, rng)
        assert len(set(moves)) == 4
        rows = [m // 15 for m in moves]
        cols = [m % 15 for m in moves]
        assert max(rows) - min(rows) < OPENING_WINDOW
        assert max(cols) - min(cols) < OPENING_WINDOW


def test_zero_plies_gives_nothing():
    assert opening_moves(15, 0, random.Random(0)) == []


def test_window_actually_moves_around():
    """窗口本身要在中央区域里移动，否则每局都从同一小块开始。"""
    rng = random.Random(2)
    corners = {min(opening_moves(15, 2, rng)) for _ in range(200)}
    assert len(corners) > 10, f"窗口几乎不动：{sorted(corners)}"


def test_same_seed_same_openings():
    """竞技场靠 seed 复现整场比赛，采样器必须是纯函数式的。"""
    a = [opening_moves(15, 3, random.Random(7)) for _ in range(3)]
    b = [opening_moves(15, 3, random.Random(7)) for _ in range(3)]
    assert a == b


def test_arena_openings_are_not_all_the_same():
    """接进竞技场之后仍然每局不同 —— 这是随机开局存在的全部理由。"""
    from gomoku_instinct.eval import RandomPlayer, play_match
    from gomoku_instinct.rules import RenjuRules

    seen = set()
    rng = random.Random(3)
    for _ in range(30):
        seen.add(tuple(opening_moves(15, 2, rng)))
    assert len(seen) > 20

    # 真跑一场，确认没把棋盘走坏
    result = play_match(RandomPlayer(seed=0), RandomPlayer(seed=1),
                        games=4, board_size=15, rules=RenjuRules(), batch=4)
    assert result.games == 4
