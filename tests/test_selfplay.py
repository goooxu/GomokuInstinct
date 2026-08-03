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


# ── 对任意局面的批量搜索 ────────────────────────────────────────────────────


def _searcher_and_eval(sims=200, slots=8, size=SIZE):
    from gomoku_instinct.core import load_core

    core = load_core()
    searcher = core.BatchSearcher(
        board_size=size, sims=sims, num_slots=slots, num_threads=2
    )
    return searcher, UniformEvaluator(size)


def _run_search(searcher, evaluator, games_moves, size=SIZE):
    """把若干着法序列交给搜索器跑完，返回各自的最佳着法。"""
    n = size * size
    cap = searcher.capacity
    counts = np.zeros(cap, dtype=np.int32)
    for i, mv in enumerate(games_moves):
        counts[i] = len(mv)
    flat = np.array([m for mv in games_moves for m in mv], dtype=np.int32)
    if flat.size == 0:
        flat = np.zeros(1, dtype=np.int32)
    searcher.set_positions(flat, counts, len(games_moves))

    boards = np.zeros((cap, n), dtype=np.uint8)
    to_move = np.zeros(cap, dtype=np.uint8)
    history = np.zeros((cap, 4), dtype=np.int32)
    move_number = np.zeros(cap, dtype=np.int32)
    active = np.zeros(cap, dtype=np.uint8)

    while not searcher.done:
        searcher.collect(boards, to_move, history, move_number, active)
        policy, value = evaluator(boards, to_move, history, move_number)
        searcher.apply(
            np.ascontiguousarray(policy, np.float32),
            np.ascontiguousarray(value, np.float32),
        )
    return [int(m) for m in searcher.best_moves()[: len(games_moves)]]


def test_search_finds_the_immediate_win():
    """一步可胜的局面，搜索必须找到那一手。

    这是搜索有没有真正工作的直接检验：先验是均匀的、价值恒为 0，网络什么都不懂，
    棋力只能来自搜索本身 —— 搜到成五的子节点会立刻拿到 +1，PUCT 会把访问数全压过去。
    """
    searcher, evaluator = _searcher_and_eval(sims=300, slots=4)

    # 黑 (4,1)(4,2)(4,3)(4,4)，白在别处；黑落 (4,0) 或 (4,5) 即成五
    def cell(r, c):
        return r * SIZE + c

    moves = [
        cell(4, 1), cell(0, 0),
        cell(4, 2), cell(0, 1),
        cell(4, 3), cell(0, 2),
        cell(4, 4), cell(8, 8),
    ]
    best = _run_search(searcher, evaluator, [moves])[0]
    assert best in (cell(4, 0), cell(4, 5)), (
        f"搜索没找到成五点，选了 {divmod(best, SIZE)}"
    )


@pytest.mark.slow
def test_search_blocks_the_opponent_win():
    """对手一步可胜时必须去挡 —— 这比"找自己的杀"贵得多。

    找自己的成五点只要一层：子节点直接是终局，一次访问就拿到 +1。
    而"不挡就输"要两层：先走 X，再看到对手的成五应手。更麻烦的是 MCTS 的价值是
    **平均**的：某个根子节点被访问 80 次、其中只有 1 次探到败着时，Q 才被拉低 0.01，
    访问数几乎不会转移走。

    均匀先验下实测：300 与 6000 次模拟都挡不住，50000 次才稳定挡住。
    这个数字本身就是策略先验价值的量化——先验能把搜索预算集中到少数候选上，
    同样的算力才够看深。也正因如此，本项目"把搜索压进权重"的路线才有意义：
    一个好的策略网络等价于给搜索省下一到两个数量级的预算。
    """
    searcher, evaluator = _searcher_and_eval(sims=50000, slots=1)

    def cell(r, c):
        return r * SIZE + c

    # 白 (4,1)-(4,4) 成四，左端已被黑 (4,0) 堵死，只剩 (4,5) 一个成五点。
    # 必须是冲四而非活四：活四两端都能成五，黑方堵哪边都输，
    # 那种局面下搜索判定"怎么走都一样输"是完全正确的，测不出防守能力。
    moves = [
        cell(4, 0), cell(4, 1),
        cell(0, 0), cell(4, 2),
        cell(0, 1), cell(4, 3),
        cell(8, 8), cell(4, 4),
    ]
    best = _run_search(searcher, evaluator, [moves])[0]
    assert best == cell(4, 5), (
        f"搜索没去挡对手唯一的成五点，选了 {divmod(best, SIZE)}"
    )


def test_search_returns_legal_moves_and_marks_inactive_slots():
    searcher, evaluator = _searcher_and_eval(sims=64, slots=8)
    rng = np.random.default_rng(5)
    games_moves = []
    for _ in range(3):
        k = int(rng.integers(4, 20))
        cells = rng.choice(SIZE * SIZE, size=k, replace=False)
        games_moves.append([int(c) for c in cells])

    all_best = [int(m) for m in _searcher_best(searcher, evaluator, games_moves)]
    for mv, best in zip(games_moves, all_best):
        assert best >= 0 and best not in mv, "搜索给出了已占点"


def _searcher_best(searcher, evaluator, games_moves):
    return _run_search(searcher, evaluator, games_moves)


# ── 尾段重搜（两趟走）────────────────────────────────────────────────────────


def _run_until_samples(actor, want=300, rounds=1200):
    for _ in range(rounds):
        actor.step()
        if actor.pending_samples > want:
            break
    return actor.drain(4000)


def test_default_samples_come_from_the_whole_game(played):
    """默认（research_last_plies=0）走原来的边下边采，样本覆盖整局。

    这条比看起来重要：两趟走如果默认生效，会悄悄改掉所有既有用法 ——
    不报错，只是训练分布和目标质量都变了。
    """
    _, data = played
    plies = data["plies_remaining"][: data["count"]]
    assert plies.max() > 20, (
        f"默认配置下样本最远只到距终局 {plies.max()} 手，两趟走似乎被误开了"
    )


def test_research_covers_exactly_the_tail_window():
    """开启后，样本必须**恰好**落在最后 N 手上 —— 一个不漏、一个不多。

    这是两趟走相对"边下边采 + 事后过滤"的关键区别：后者只能收到窗口里
    恰好抽中完整搜索的那部分（约四分之一），前者是全部。
    """
    window = 6
    actor = make_actor(research_last_plies=window, seed=99)
    data = _run_until_samples(actor)
    assert data["count"] > 0, "没产出样本，测不了"

    plies = data["plies_remaining"][: data["count"]]
    assert plies.max() <= window, f"窗口 {window}，却出现距终局 {plies.max()} 手的样本"
    assert plies.min() >= 1

    # 每局应当拿满 min(窗口, 局长) 条 —— 边下边采只能拿到其中约 1/4
    stats = actor.stats
    per_game = stats["samples"] / max(1, stats["games"])
    assert per_game > window * 0.8, (
        f"每局只产出 {per_game:.1f} 条，远少于窗口 {window}，重搜没跑满"
    )


def test_research_targets_are_full_search():
    """重搜的目标必须来自满 sims 的搜索。

    访问分布的总访问数就是 sims；用 fast_sims 搜出来的分布会粗糙得多，
    而两趟走的全部意义就在于"窗口内每一手都是干净目标"。
    这里用分布的非零支撑数间接验证：满搜索会访问到更多不同的着法。
    """
    fine = make_actor(research_last_plies=6, sims=64, fast_sims=4, seed=11)
    d = _run_until_samples(fine)
    pol = d["policy"][: d["count"]]
    support = (pol > 0).sum(axis=1)
    assert support.mean() > 4, (
        f"平均只访问到 {support.mean():.1f} 个着法，不像是 64 次搜索的结果"
    )


def test_window_larger_than_the_game_takes_everything():
    """窗口大于任何一局的长度时，应当覆盖整局（从第 1 手到最后一手）。"""
    actor = make_actor(research_last_plies=10_000, seed=5)
    data = _run_until_samples(actor)
    plies = data["plies_remaining"][: data["count"]]
    # 覆盖到开局，说明窗口没有被错误地截断
    assert plies.max() > 20
