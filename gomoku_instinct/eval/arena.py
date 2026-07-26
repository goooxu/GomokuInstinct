"""竞技场：批量对局与 Elo 估计。

一条硬性约定：**模型一律以零搜索模式参赛**，与实际部署条件完全一致。
如果用 MCTS 模式来评测晋级，优化的就是错误的目标 —— 我们要的是「原始策略强」，
而不是「原始策略 + 搜索强」。

先后手强制轮换：连珠的黑白规则不对称（黑方有禁手、必须恰好五连），
不轮换的话胜率完全没有可比性。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..rules import BLACK, WHITE, ForbiddenSemantics, Game, Outcome, RenjuRules


@dataclass
class MatchResult:
    games: int = 0
    wins: int = 0  # A 方胜
    losses: int = 0
    draws: int = 0
    a_as_black_wins: int = 0
    a_as_black_games: int = 0
    a_forbidden_losses: int = 0  # A 方主动走出禁手而告负的局数
    b_forbidden_losses: int = 0
    total_plies: int = 0

    @property
    def score(self) -> float:
        """A 方得分率：胜 1 分、和 0.5 分。"""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo_diff(self) -> float:
        """由得分率反推 Elo 差。全胜/全负时给一个有限的封顶值。"""
        s = self.score
        eps = 1.0 / (2.0 * max(self.games, 1))
        s = min(max(s, eps), 1.0 - eps)
        return -400.0 * math.log10(1.0 / s - 1.0)

    @property
    def avg_plies(self) -> float:
        return self.total_plies / max(1, self.games)

    def summary(self, name_a: str = "A", name_b: str = "B") -> str:
        return (
            f"{name_a} vs {name_b}：{self.wins}胜 {self.losses}负 {self.draws}和"
            f"（得分率 {self.score:.1%}，Elo 差 {self.elo_diff:+.0f}）\n"
            f"  平均 {self.avg_plies:.0f} 手/局；"
            f"{name_a} 执黑 {self.a_as_black_games} 局胜 {self.a_as_black_wins} 局\n"
            f"  禁手告负：{name_a} {self.a_forbidden_losses} 次，"
            f"{name_b} {self.b_forbidden_losses} 次"
        )


def play_match(
    player_a,
    player_b,
    *,
    games: int,
    board_size: int,
    rules: RenjuRules | None = None,
    max_plies: int | None = None,
    batch: int = 64,
    random_opening_plies: int = 2,
    seed: int = 0,
) -> MatchResult:
    """让两个 player 对弈若干局。先后手逐局轮换。

    player 需要实现 choose_batch(list[Game]) -> list[int]。同一批里所有对局都
    轮到同一方走，因此模型侧可以一次前向吃掉整批。

    **每局开头先随机落若干子**（random_opening_plies）。两个确定性 player
    对弈时，若不这么做，每一局都是同一盘棋的重放，胜负完全由先后手决定，
    得分率会恒等于 50% —— 那是测量退化，不是势均力敌。零搜索策略与低模拟数
    MCTS 对打时尤其明显，因为后者基本就是跟着策略先验走。
    """
    rules = rules or RenjuRules()
    max_plies = max_plies or board_size * board_size
    result = MatchResult()
    rng = random.Random(seed)

    remaining = games
    game_index = 0
    while remaining > 0:
        count = min(batch, remaining)
        remaining -= count

        # 本批内逐局轮换先手
        a_is_black = [(game_index + i) % 2 == 0 for i in range(count)]
        game_index += count
        boards = [Game(board_size, rules, ForbiddenSemantics.LOSE) for _ in range(count)]
        for game in boards:
            for _ in range(random_opening_plies):
                legal = game.legal_moves()
                if not legal or game.is_terminal():
                    break
                game.play(rng.choice(legal))
        active = [i for i in range(count) if not boards[i].is_terminal()]

        while active:
            # 按「该谁走」分组，各自一次批量决策
            for player, is_a in ((player_a, True), (player_b, False)):
                group = [
                    i
                    for i in active
                    if not boards[i].is_terminal()
                    and (
                        (boards[i].to_move == BLACK) == (a_is_black[i] == is_a)
                    )
                ]
                if not group:
                    continue
                moves = player.choose_batch([boards[i] for i in group])
                for i, move in zip(group, moves):
                    boards[i].play(move)

            active = [
                i
                for i in active
                if not boards[i].is_terminal() and boards[i].num_moves < max_plies
            ]

        for i in range(count):
            game = boards[i]
            result.games += 1
            result.total_plies += game.num_moves

            if game.is_terminal() and game.outcome != Outcome.DRAW:
                winner = BLACK if game.outcome == Outcome.BLACK_WIN else WHITE
                a_won = (winner == BLACK) == a_is_black[i]
                if a_won:
                    result.wins += 1
                else:
                    result.losses += 1

                # 禁手告负：最后一手是黑方走的，却判白胜
                last_move, color, judgment = game.history[-1]
                if judgment.is_forbidden:
                    loser_is_a = (color == BLACK) == a_is_black[i]
                    if loser_is_a:
                        result.a_forbidden_losses += 1
                    else:
                        result.b_forbidden_losses += 1
            else:
                result.draws += 1

            if a_is_black[i]:
                result.a_as_black_games += 1
                if game.outcome == Outcome.BLACK_WIN:
                    result.a_as_black_wins += 1

    return result


def elo_from_score(score: float, games: int) -> float:
    eps = 1.0 / (2.0 * max(games, 1))
    score = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)
