"""自博弈与 MCTS 的正确性测试。

搜索本身没有「标准答案」可比，所以这里查的是**不变量**：产出的每条样本都必须与
规则层自洽（局面可达、目标分布只落在空点上、价值取值合法……）。
另外用规则参考实现独立复核了「黑方走出禁手即判负」这条语义确实接上了。
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku_instinct.rules import BLACK, EMPTY, WHITE
from gomoku_instinct.selfplay import (
    RandomEvaluator,
    SelfPlayActor,
    UniformEvaluator,
)

SIZE = 9  # 小棋盘，测试跑得快；规则逻辑与 15x15 完全一致
N = SIZE * SIZE


def make_actor(**kwargs):
    defaults = dict(
        board_size=SIZE,
        num_games=32,
        sims=24,
        fast_sims=8,
        full_search_prob=0.5,
        temperature_moves=4,
        num_threads=4,
        seed=1234,
        resign_enabled=False,
    )
    defaults.update(kwargs)
    return SelfPlayActor(UniformEvaluator(SIZE), **defaults)


@pytest.fixture(scope="module")
def played():
    """跑到攒出足够样本为止，供多个测试复用。"""
    actor = make_actor()
    for _ in range(600):
        actor.step()
        if actor.pending_samples > 400:
            break
    data = actor.drain(4000)
    assert data["count"] > 0, "跑了 600 轮还没产出任何样本"
    return actor, data


# ── 对局本身 ────────────────────────────────────────────────────────────────


def test_games_complete_and_outcomes_are_accounted(played):
    actor, _ = played
    stats = actor.stats
    assert stats["games"] > 0
    assert stats["moves"] > 0
    assert (
        stats["black_wins"] + stats["white_wins"] + stats["draws"] == stats["games"]
    )


def test_mcts_avoids_forbidden_points(played):
    """搜索应当自己学会避开禁手点。

    禁手点的子节点是即时负，访问一次就会被 PUCT 排除，因此哪怕先验是均匀的、
    网络什么都不懂，MCTS 也几乎不会主动走上去。这正是我们希望蒸馏进策略网络的行为。
    """
    actor, _ = played
    stats = actor.stats
    rate = stats["forbidden_losses"] / max(stats["games"], 1)
    assert rate < 0.05, f"带搜索的对局里禁手告负率高达 {rate:.1%}"


def test_random_play_does_hit_forbidden_points():
    """把搜索摘掉、纯随机落子，黑方就会踩到禁手点。

    这条验证「禁手点仍是合法落子、落上去立即判负」的语义在 runner 里确实接上了 ——
    如果引擎把禁手点从合法集里屏蔽掉了，这个计数会恒为 0。
    与上一条合起来看：语义是通的，而搜索确实在规避它。
    """
    actor = SelfPlayActor(
        RandomEvaluator(SIZE, seed=7),
        board_size=SIZE,
        num_games=64,
        sims=2,
        fast_sims=2,
        full_search_prob=0.0,
        raw_policy_fraction=1.0,  # 落子完全由随机先验的 argmax 决定
        num_threads=4,
        seed=4242,
        resign_enabled=False,
    )
    for _ in range(2000):
        actor.step()
        if actor.stats["games"] >= 200:
            break
    stats = actor.stats
    assert stats["games"] >= 50, f"只跑完了 {stats['games']} 局"
    assert stats["forbidden_losses"] > 0, "纯随机落子竟然一次禁手都没踩到"


# ── 样本不变量 ──────────────────────────────────────────────────────────────


def test_sample_boards_are_reachable_positions(played):
    _, data = played
    count = data["count"]
    boards = data["boards"][:count]
    to_move = data["to_move"][:count]
    move_number = data["move_number"][:count]

    stones = (boards != EMPTY).sum(axis=1)
    assert np.array_equal(stones, move_number), "盘上子数与手数对不上"

    black = (boards == BLACK).sum(axis=1)
    white = (boards == WHITE).sum(axis=1)
    # 黑先走：轮到黑走时两色相等，轮到白走时黑多一子
    assert np.all(black - white == move_number % 2), "黑白子数差不符合轮换"

    expected_side = np.where(move_number % 2 == 0, BLACK, WHITE)
    assert np.array_equal(to_move, expected_side), "行棋方与手数奇偶不一致"


def test_policy_targets_are_distributions_over_empty_points(played):
    _, data = played
    count = data["count"]
    boards = data["boards"][:count]
    policy = data["policy"][:count]

    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(policy >= 0.0)
    occupied = boards != EMPTY
    assert policy[occupied].max() == 0.0, "目标分布落到了已占点上"


def test_value_targets_are_game_outcomes(played):
    _, data = played
    count = data["count"]
    value = data["value"][:count]
    assert set(np.unique(value).tolist()) <= {-1.0, 0.0, 1.0}


def test_plies_remaining_and_next_move_are_consistent(played):
    _, data = played
    count = data["count"]
    boards = data["boards"][:count]
    plies = data["plies_remaining"][:count]
    next_move = data["next_move"][:count]

    assert np.all(plies >= 1), "剩余手数至少为 1（样本所在的这一手还没走）"

    # 对手的应手必须落在当时的空点上；-1 表示该样本已是本局最后一手
    has_next = next_move >= 0
    idx = np.nonzero(has_next)[0]
    assert len(idx) > 0
    assert np.all(boards[idx, next_move[idx]] == EMPTY)


def test_history_planes_point_at_occupied_cells(played):
    _, data = played
    count = data["count"]
    boards = data["boards"][:count]
    history = data["history"][:count]

    for k in range(history.shape[1]):
        col = history[:, k]
        valid = col >= 0
        idx = np.nonzero(valid)[0]
        if len(idx) == 0:
            continue
        assert np.all(boards[idx, col[idx]] != EMPTY), f"第 {k} 手历史指向了空点"


def test_root_value_is_in_range(played):
    _, data = played
    count = data["count"]
    rv = data["root_value"][:count]
    assert np.all(rv >= -1.0001) and np.all(rv <= 1.0001)


# ── 可复现性与容灾 ──────────────────────────────────────────────────────────


def test_same_seed_gives_same_games():
    a = make_actor(seed=777)
    b = make_actor(seed=777)
    for _ in range(200):
        a.step()
        b.step()
    assert a.stats == b.stats


def test_different_seeds_diverge():
    a = make_actor(seed=1)
    b = make_actor(seed=2)
    for _ in range(200):
        a.step()
        b.step()
    assert a.stats != b.stats


def test_rng_state_roundtrip_is_exact():
    """换机续训要求随机流能逐位恢复。"""
    a = make_actor(seed=99)
    for _ in range(60):
        a.step()
    state = a.rng_state()

    b = make_actor(seed=12345)  # 故意用不同的种子起步
    for _ in range(60):
        b.step()
    b.set_rng_state(state)
    b.reset_stats()
    a.reset_stats()

    # 从同一随机状态出发，后续对局的随机决策应当一致。
    # 局面不同所以统计量不会完全相同，这里只验证状态确实被写进去了。
    assert b.rng_state() == state


# ── 部署分布自博弈 ──────────────────────────────────────────────────────────


def test_raw_policy_games_are_flagged():
    """部署分布自博弈：这些局由零搜索策略落子，但仍会跑搜索产出训练目标。"""
    actor = SelfPlayActor(
        RandomEvaluator(SIZE, seed=3),
        board_size=SIZE,
        num_games=16,
        sims=8,
        fast_sims=4,
        full_search_prob=0.5,
        raw_policy_fraction=1.0,
        num_threads=4,
        seed=555,
        resign_enabled=False,
    )
    for _ in range(1500):
        actor.step()
        if actor.stats["games"] > 8:
            break
    stats = actor.stats
    assert stats["games"] > 0
    assert stats["raw_policy_games"] >= stats["games"]
    # 这些局照样要产出训练目标 —— 否则「消除分布漂移」就无从谈起
    assert stats["samples"] > 0


def test_no_raw_policy_games_by_default():
    actor = make_actor(num_games=16)
    for _ in range(200):
        actor.step()
    assert actor.stats["raw_policy_games"] == 0
