"""带 MCTS 的对手，用来测量本项目的头号指标。

零搜索策略 vs 同一份权重的 MCTS 版本，两者的 Elo 差就是「搜索还没被压进权重里的
那一部分棋力」。整个方案赌的就是把这个差压到足够小 —— 它是唯一能直接量化这件事的数。

除了搜索之外，两边用的是同一个网络、同一套输入编码、同样的确定性选点方式，
差异因此完全归于搜索本身。
"""

from __future__ import annotations

import numpy as np
import torch

from ..core import load_core
from ..model.features import NUM_HISTORY_PLANES
from ..rules import Game


class MctsPlayer:
    """对每个局面跑固定次数的 MCTS，按根节点访问数取最佳着法。"""

    name = "mcts"

    def __init__(
        self,
        evaluator,
        board_size: int,
        sims: int = 800,
        slots: int = 64,
        threads: int = 24,
        c_puct: float = 1.6,
    ) -> None:
        core = load_core()
        self.searcher = core.BatchSearcher(
            board_size=board_size,
            sims=sims,
            num_slots=slots,
            c_puct=c_puct,
            num_threads=threads,
        )
        self.evaluator = evaluator
        self.size = board_size
        self.sims = sims
        n = board_size * board_size

        self.boards = np.zeros((slots, n), dtype=np.uint8)
        self.to_move = np.zeros(slots, dtype=np.uint8)
        self.history = np.zeros((slots, NUM_HISTORY_PLANES), dtype=np.int32)
        self.move_number = np.zeros(slots, dtype=np.int32)
        self.active = np.zeros(slots, dtype=np.uint8)

    def choose_batch(self, games: list[Game]) -> list[int]:
        capacity = self.searcher.capacity
        out: list[int] = []
        for start in range(0, len(games), capacity):
            chunk = games[start : start + capacity]
            out.extend(self._search_chunk(chunk))
        return out

    def _search_chunk(self, games: list[Game]) -> list[int]:
        # 按完整着法序列载入：网络输入含最近数手的落点平面，
        # 只摆棋盘的话那几个平面会全空，测的就不是同一个输入下的表现。
        counts = np.array([len(g.history) for g in games], dtype=np.int32)
        flat = np.array(
            [m for g in games for m, _, _ in g.history], dtype=np.int32
        )
        if flat.size == 0:
            flat = np.zeros(1, dtype=np.int32)
        padded = np.zeros(self.searcher.capacity, dtype=np.int32)
        padded[: len(counts)] = counts
        self.searcher.set_positions(flat, padded, len(games))

        while not self.searcher.done:
            self.searcher.collect(
                self.boards, self.to_move, self.history, self.move_number, self.active
            )
            policy, value = self.evaluator(
                self.boards, self.to_move, self.history, self.move_number
            )
            self.searcher.apply(
                np.ascontiguousarray(policy, dtype=np.float32),
                np.ascontiguousarray(value, dtype=np.float32),
            )

        moves = self.searcher.best_moves()
        return [int(m) for m in moves[: len(games)]]
