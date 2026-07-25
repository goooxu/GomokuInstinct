"""对局状态机：把规则判定器包装成可推进、可回退的对局。"""

from __future__ import annotations

from .board import Board
from .constants import (
    BLACK,
    EMPTY,
    Forbidden,
    ForbiddenSemantics,
    Outcome,
    WHITE,
    opponent,
)
from .renju import MoveJudgment, RenjuRules


class Game:
    """一局连珠。

    禁手点的语义由 semantics 决定：

      LOSE      严格 RIF —— 禁手点仍是合法落子，黑方落上去立即判负。
                这意味着「避开禁手」是模型自己必须学会的能力。
      ILLEGAL   禁手点直接从合法落子集中移除；黑方无合法点可下时判黑负。
    """

    def __init__(
        self,
        size: int = 15,
        rules: RenjuRules | None = None,
        semantics: ForbiddenSemantics = ForbiddenSemantics.LOSE,
    ) -> None:
        self.board = Board(size)
        self.rules = rules if rules is not None else RenjuRules()
        self.semantics = semantics
        self.to_move = BLACK
        self.outcome = Outcome.ONGOING
        self.history: list[tuple[int, int, MoveJudgment]] = []  # (move, color, 判定)

    # ── 属性 ────────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return self.board.size

    @property
    def num_moves(self) -> int:
        return len(self.history)

    def is_terminal(self) -> bool:
        return self.outcome != Outcome.ONGOING

    def last_move(self) -> int | None:
        return self.history[-1][0] if self.history else None

    # ── 合法落子 ────────────────────────────────────────────────────────────
    def legal_moves(self) -> list[int]:
        if self.is_terminal():
            return []
        empties = [i for i, v in enumerate(self.board.grid) if v == EMPTY]
        if self.semantics == ForbiddenSemantics.LOSE or self.to_move == WHITE:
            return empties
        size = self.size
        return [
            i
            for i in empties
            if not self.rules.is_forbidden(self.board, *divmod(i, size))
        ]

    def legal_mask(self) -> list[bool]:
        mask = [False] * (self.size * self.size)
        for m in self.legal_moves():
            mask[m] = True
        return mask

    # ── 推进与回退 ──────────────────────────────────────────────────────────
    def play(self, move: int) -> MoveJudgment:
        if self.is_terminal():
            raise RuntimeError("对局已结束")
        size = self.size
        r, c = divmod(move, size)
        if self.board.cell(r, c) != EMPTY:
            raise ValueError(f"({r}, {c}) 不是空点")

        color = self.to_move
        judgment = self.rules.judge(self.board, r, c, color)

        if (
            self.semantics == ForbiddenSemantics.ILLEGAL
            and color == BLACK
            and judgment.is_forbidden
        ):
            raise ValueError(f"({r}, {c}) 是禁手点，在 ILLEGAL 语义下不可落子")

        self.board.set(r, c, color)
        self.history.append((move, color, judgment))

        if judgment.outcome != Outcome.ONGOING:
            self.outcome = judgment.outcome
        elif self.board.is_full():
            self.outcome = Outcome.DRAW
        else:
            self.to_move = opponent(color)
            # ILLEGAL 语义下黑方可能被逼到无处可下。
            if (
                self.semantics == ForbiddenSemantics.ILLEGAL
                and self.to_move == BLACK
                and not self.legal_moves()
            ):
                self.outcome = Outcome.WHITE_WIN

        return judgment

    def undo(self) -> None:
        if not self.history:
            raise RuntimeError("没有可回退的落子")
        move, color, _ = self.history.pop()
        r, c = divmod(move, self.size)
        self.board.set(r, c, EMPTY)
        self.to_move = color
        self.outcome = Outcome.ONGOING

    # ── 辅助 ────────────────────────────────────────────────────────────────
    def forbidden_map(self) -> list[bool]:
        """当前局面下黑方的禁手点标记。"""
        return self.rules.forbidden_map(self.board)

    def clone(self) -> "Game":
        other = Game(self.size, self.rules, self.semantics)
        other.board.grid[:] = self.board.grid
        other.to_move = self.to_move
        other.outcome = self.outcome
        other.history = list(self.history)
        return other

    @classmethod
    def from_moves(
        cls,
        moves: list[int],
        size: int = 15,
        rules: RenjuRules | None = None,
        semantics: ForbiddenSemantics = ForbiddenSemantics.LOSE,
    ) -> "Game":
        game = cls(size, rules, semantics)
        for m in moves:
            game.play(m)
        return game

    def result_for(self, color: int) -> float:
        """从 color 视角看的对局结果：胜 1.0 / 和 0.0 / 负 -1.0。"""
        if self.outcome == Outcome.DRAW:
            return 0.0
        if self.outcome == Outcome.BLACK_WIN:
            return 1.0 if color == BLACK else -1.0
        if self.outcome == Outcome.WHITE_WIN:
            return 1.0 if color == WHITE else -1.0
        raise RuntimeError("对局尚未结束")

    def __str__(self) -> str:
        return str(self.board)
