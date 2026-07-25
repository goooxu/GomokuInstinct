"""连珠规则的 Python 参考实现。

这份实现是规则的可执行规范，优先可读性；高性能版本在 csrc/ 下，
两者由 tests/test_rules_differential.py 锁定一致。
"""

from .board import Board
from .constants import (
    BLACK,
    DIRECTIONS,
    EMPTY,
    WALL,
    WHITE,
    Forbidden,
    ForbiddenSemantics,
    Level,
    Outcome,
    opponent,
)
from .game import Game
from .renju import MoveJudgment, RenjuRules

__all__ = [
    "BLACK",
    "DIRECTIONS",
    "EMPTY",
    "WALL",
    "WHITE",
    "Board",
    "Forbidden",
    "ForbiddenSemantics",
    "Game",
    "Level",
    "MoveJudgment",
    "Outcome",
    "RenjuRules",
    "opponent",
]
