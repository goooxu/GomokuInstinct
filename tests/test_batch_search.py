"""批量收集（virtual loss）。

**这套改动最大的风险是"默认路径悄悄变了"** —— 不报错，只是搜索结果不一样了，
而自博弈和评测两条路径都用同一棵 `MctsTree`。所以这里第一条也是最重的一条，
就是钉住 `leaves=1` 与不带 virtual loss 时的行为完全一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku_instinct.core import load_core
from gomoku_instinct.rules import EMPTY

SIZE = 9
N = SIZE * SIZE


class _FixedNet:
    """固定权重的确定性打分器。均匀先验会让选点退化成"永远第一个"，测不出真实路径。"""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((N, N)) * 0.3).astype(np.float32)
        self.b = (rng.standard_normal(N) * 0.3).astype(np.float32)

    def __call__(self, boards):
        legal = boards == EMPTY
        s = boards.astype(np.float32) @ self.w + self.b
        s = np.exp(s - s.max(1, keepdims=True)) * legal
        p = (s / np.maximum(s.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        return p, np.tanh(s.sum(1) - 1).astype(np.float32)


def _run(leaves: int, sims: int = 48, slots: int = 2, seed: int = 0):
    core = load_core()
    s = core.BatchSearcher(board_size=SIZE, sims=sims, num_slots=slots,
                           num_threads=2, leaves_per_slot=leaves)
    # 与录快照时用的局面完全一致，下面那两条快照测试才对得上
    seqs = [[40, 41], [40, 41, 30, 31], [40, 41, 30, 31, 22, 50],
            [40, 41, 30, 31, 22, 50, 13, 60]][:slots]
    moves = np.array([m for q in seqs for m in q], dtype=np.int32)
    counts = np.zeros(s.capacity, dtype=np.int32)
    counts[: len(seqs)] = [len(q) for q in seqs]
    s.set_positions(moves, counts, len(seqs))

    rows = s.batch_rows
    b = np.zeros((rows, N), np.uint8)
    tm = np.zeros(rows, np.uint8)
    h = np.zeros((rows, 4), np.int32)
    mn = np.zeros(rows, np.int32)
    act = np.zeros(rows, np.uint8)
    net = _FixedNet(seed)
    leaked = []
    while not s.done:
        s.collect(b, tm, h, mn, act)
        p, v = net(b)
        s.apply(np.ascontiguousarray(p, np.float32), np.ascontiguousarray(v, np.float32))
        leaked.append(s.virtual_outstanding)
    return s, leaked


def test_batch_rows_matches_leaves():
    s, _ = _run(leaves=1)
    assert s.batch_rows == s.capacity and s.leaves_per_slot == 1
    s, _ = _run(leaves=8)
    assert s.batch_rows == s.capacity * 8 and s.leaves_per_slot == 8


def test_virtual_visits_never_leak():
    """每一轮回填之后虚拟访问都必须归零。

    泄漏不会抛异常 —— 它只会让后续下潜一直绕开某条路径，搜索悄悄变差。
    这正是第 11 章那一类，所以做成硬断言而不是"看着像对的"。
    """
    for leaves in (1, 4, 16):
        _, leaked = _run(leaves=leaves)
        assert leaked and max(leaked) == 0, f"leaves={leaves} 泄漏了 {max(leaked)}"


def test_single_leaf_path_is_deterministic():
    """同样输入跑两次，结果必须逐位相同 —— 后面的快照比对才有意义。"""
    a, _ = _run(leaves=1)
    b, _ = _run(leaves=1)
    assert np.array_equal(a.visit_counts(), b.visit_counts())
    assert np.array_equal(a.best_moves(), b.best_moves())


@pytest.mark.parametrize("leaves", [1, 2, 4, 8, 16, 32])
def test_leaves_does_not_change_total_simulations(leaves):
    """凑批不能让模拟数缩水 —— 每轮多产的叶子每一个都要算一次模拟。

    这条一开始就抓到过一个真 bug：根节点还没展开时一轮会连着下潜两次落到同一个
    叶子，第二次被判重丢弃，**但已经计入了模拟数** —— 同样的 sims 下批量比
    逐叶少算一次。修法是判重提前到记录之前，并把那次的虚拟败绩撤掉。
    """
    s, _ = _run(leaves=leaves, sims=48, slots=1)
    # 根节点第一次模拟只展开根、不产生子节点访问，所以是 sims - 1
    assert s.visit_counts()[0].sum() == 47


@pytest.mark.parametrize("leaves", [2, 4, 8, 16])
def test_batching_actually_batches(leaves):
    """每轮真的收集到了多个叶子 —— 否则加速无从谈起，而且不会有任何报错。"""
    core = load_core()
    s = core.BatchSearcher(board_size=SIZE, sims=64, num_slots=1,
                           num_threads=2, leaves_per_slot=leaves)
    s.set_positions(np.array([40, 41, 30], np.int32), np.array([3], np.int32), 1)
    rows = s.batch_rows
    b = np.zeros((rows, N), np.uint8); tm = np.zeros(rows, np.uint8)
    h = np.zeros((rows, 4), np.int32); mn = np.zeros(rows, np.int32)
    act = np.zeros(rows, np.uint8)
    net = _FixedNet(0)
    rounds = 0
    while not s.done:
        s.collect(b, tm, h, mn, act)
        p, v = net(b)
        s.apply(np.ascontiguousarray(p, np.float32), np.ascontiguousarray(v, np.float32))
        rounds += 1
    # 第一轮根节点还没展开，只能产一个叶子；其余轮次应当填满
    assert rounds <= 64 // leaves + 2, f"leaves={leaves} 只压缩到 {rounds} 轮"


def test_leaves_is_clamped_so_there_are_enough_rounds():
    """**轮数太少搜索会退化成宽度优先，而且不报任何错。**

    一轮同时取 N 个叶子，就有 N 条路径是靠虚拟败绩硬岔开的。
    实测 sims=64、leaves=16 时根节点访问从 55:1:1:1 摊成 13:10:9:8 ——
    快了，但搜的东西不一样了。所以部署侧按 sims//8 钳住。
    """
    from gomoku_instinct.cli.search_engine import SearchPlayer
    from gomoku_instinct.model import InstinctNet, ModelConfig

    model = InstinctNet(ModelConfig(size=SIZE, channels=16, blocks=1))
    for sims, want in ((16, 2), (32, 4), (64, 8), (128, 16), (400, 16)):
        p = SearchPlayer(model, SIZE, "cpu", sims=sims, leaves=16, threads=1)
        assert p.leaves == want, f"sims={sims} 钳成了 {p.leaves}，应当是 {want}"


# ── 默认路径的行为快照 ────────────────────────────────────────────────────

# 加 virtual loss 之前录下来的。**这套改动最大的风险是"默认路径悄悄变了"** ——
# 不报错，只是搜索结果不一样了，而自博弈（训练）和评测两条路径都用同一棵
# `MctsTree`。数字本身没有含义，它们只是"改动前是这样"的证据。
_SNAPSHOT_BEST = [31, 28, 28, 28]
_SNAPSHOT_ROOT_VALUES = [-0.037599, -0.108192, -0.070844, -0.016255]
_SNAPSHOT_VISITS_SUM = [47, 47, 47, 47]
_SNAPSHOT_SELFPLAY = {
    "count": 71,
    "policy_sha": "10d5d0c9d2f620d1",
    "boards_sha": "93a4dbc3858a1639",
}


def test_single_leaf_search_matches_pre_change_snapshot():
    """leaves=1 必须与加 virtual loss 之前逐位相同。

    竞技场、`search_gap.py`、技术报告里的每一个棋力数字都走这条路径。
    """
    s, _ = _run(leaves=1, sims=48, slots=4, seed=0)
    assert [int(x) for x in s.best_moves()[:4]] == _SNAPSHOT_BEST
    assert [round(float(x), 6) for x in s.root_values()[:4]] == _SNAPSHOT_ROOT_VALUES
    assert [int(r.sum()) for r in s.visit_counts()[:4]] == _SNAPSHOT_VISITS_SUM


def test_selfplay_path_matches_pre_change_snapshot():
    """自博弈一个字节都不能变 —— 它连 `BatchSearcher` 都不用，
    但和它共用 `MctsTree`。virtual loss 全为 0 时 `select_child` 的算术
    与改动前逐位相同（整数加 0、浮点减 0.0f 都是精确的），这条测试钉住这一点。
    """
    import hashlib

    from gomoku_instinct.selfplay import SelfPlayActor, UniformEvaluator

    a = SelfPlayActor(UniformEvaluator(SIZE), board_size=SIZE, num_games=16,
                      sims=24, fast_sims=8, full_search_prob=0.5,
                      temperature_moves=4, num_threads=4, seed=1234,
                      resign_enabled=False)
    for _ in range(1200):
        a.step()
        if a.pending_samples > 60:
            break
    d = a.drain(200)
    c = d["count"]
    assert c == _SNAPSHOT_SELFPLAY["count"]
    sha = lambda x: hashlib.sha256(np.ascontiguousarray(x[:c]).tobytes()).hexdigest()[:16]
    assert sha(d["policy"]) == _SNAPSHOT_SELFPLAY["policy_sha"]
    assert sha(d["boards"]) == _SNAPSHOT_SELFPLAY["boards_sha"]
