"""竞技场里的规则基线对手。

这些**不是 AI**，只是用规则写死的启发式，用来给棋力提供一个绝对标尺 ——
Elo 是相对量，没有固定参照物就只能看出「比上一代强」，看不出「到底多强」。

它们允许使用规则（这本来就是规则基线），但不含任何棋谱知识。
"""

from __future__ import annotations

import random

from ..rules import BLACK, EMPTY, WHITE, Game
from ..rules.constants import Level, opponent


class RandomPlayer:
    """随机合法落子。棋力下限的参照物。"""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_batch(self, games: list[Game]) -> list[int]:
        return [self.rng.choice(game.legal_moves()) for game in games]


class GreedyThreatPlayer:
    """贪心威胁型基线：能赢就赢，该挡就挡，否则挑棋型分最高的点。

    优先级完全由规则导出的棋型等级决定，没有任何搜索，也没有任何调参出来的权重表。
    """

    name = "greedy_threat"

    def __init__(self, rules_core, board_size: int, seed: int = 0) -> None:
        self.core = rules_core
        self.size = board_size
        self.n = board_size * board_size
        self.rng = random.Random(seed)

    def choose_batch(self, games: list[Game]) -> list[int]:
        return [self._choose(game) for game in games]

    def _choose(self, game: Game) -> int:
        grid = bytes(game.board.grid)
        me = game.to_move
        them = opponent(me)
        mine = self.core.pattern_map(grid, self.size, me)
        theirs = self.core.pattern_map(grid, self.size, them)

        legal = game.legal_moves()
        # 执黑时避开自己的禁手点：这是规则基线，用规则天经地义
        if me == BLACK:
            forbidden = self.core.forbidden_map(grid, self.size)
            safe = [m for m in legal if not forbidden[m]]
            if safe:
                legal = safe

        def pick(predicate):
            hits = [m for m in legal if predicate(m)]
            return self.rng.choice(hits) if hits else None

        # 1. 自己能成五
        move = pick(lambda m: mine[m] == int(Level.FIVE))
        # 2. 对手能成五，挡住
        move = move or pick(lambda m: theirs[m] == int(Level.FIVE))
        # 3. 自己能成活四
        move = move or pick(lambda m: mine[m] == int(Level.OPEN_FOUR))
        # 4. 对手能成活四，挡住
        move = move or pick(lambda m: theirs[m] == int(Level.OPEN_FOUR))
        if move is not None:
            return move

        # 否则按「自己的棋型权重更高、兼顾破坏对手」打分
        center = (self.size - 1) / 2.0
        best, best_score = legal[0], -1e9
        for m in legal:
            r, c = divmod(m, self.size)
            distance = max(abs(r - center), abs(c - center))
            score = 3.0 * mine[m] + 2.0 * theirs[m] - 0.05 * distance
            score += self.rng.random() * 1e-3  # 打散完全相同的分数
            if score > best_score:
                best, best_score = m, score
        return best
