"""棋盘表示（Python 参考实现）。

这一份实现优先可读性，是规则的可执行规范；性能由 csrc/ 下的 C++ 实现负责，
两者通过差分测试保持一致。
"""

from __future__ import annotations

from .constants import (
    BLACK,
    EMPTY,
    LINE_RADIUS,
    STONE_NAMES,
    WALL,
    WHITE,
)

_CHAR_TO_STONE = {
    ".": EMPTY,
    "_": EMPTY,
    "X": BLACK,
    "x": BLACK,
    "B": BLACK,
    "O": WHITE,
    "o": WHITE,
    "W": WHITE,
}


class Board:
    """一个朴素的方形棋盘。落子历史由 Game 维护，这里只管格子。"""

    __slots__ = ("size", "grid")

    def __init__(self, size: int = 15) -> None:
        if size < 5:
            raise ValueError("棋盘边长至少为 5")
        self.size = size
        self.grid = bytearray(size * size)

    # ── 基本访问 ────────────────────────────────────────────────────────────
    def index(self, r: int, c: int) -> int:
        return r * self.size + c

    def coord(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.size)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def cell(self, r: int, c: int) -> int:
        """越界返回 WALL —— 让方向扫描不必到处写边界判断。"""
        if 0 <= r < self.size and 0 <= c < self.size:
            return self.grid[r * self.size + c]
        return WALL

    def set(self, r: int, c: int, value: int) -> None:
        self.grid[r * self.size + c] = value

    def is_empty(self, r: int, c: int) -> bool:
        return self.cell(r, c) == EMPTY

    def is_full(self) -> bool:
        return EMPTY not in self.grid

    def stone_count(self) -> int:
        return sum(1 for v in self.grid if v != EMPTY)

    def empties(self) -> list[tuple[int, int]]:
        size = self.size
        return [divmod(i, size) for i, v in enumerate(self.grid) if v == EMPTY]

    # ── 方向扫描 ────────────────────────────────────────────────────────────
    def line(
        self, r: int, c: int, dr: int, dc: int, radius: int = LINE_RADIUS
    ) -> list[int]:
        """抽出以 (r, c) 为中心、沿 (dr, dc) 方向的一条直线，越界处填 WALL。

        返回长度为 2*radius+1 的列表，中心固定在下标 radius 上。棋型判定全部
        在这个一维列表上完成，不再触碰二维坐标。
        """
        return [self.cell(r + i * dr, c + i * dc) for i in range(-radius, radius + 1)]

    # ── 复制与序列化 ────────────────────────────────────────────────────────
    def copy(self) -> "Board":
        other = Board(self.size)
        other.grid[:] = self.grid
        return other

    def key(self) -> bytes:
        return bytes(self.grid)

    def to_rows(self) -> list[str]:
        size = self.size
        return [
            "".join(STONE_NAMES[self.grid[r * size + c]] for c in range(size))
            for r in range(size)
        ]

    def __str__(self) -> str:
        return "\n".join(self.to_rows())

    @classmethod
    def from_rows(cls, rows: list[str], size: int | None = None) -> "Board":
        """从字符画构造棋盘，用于测试用例。

        '.' 空点，'X' 黑子，'O' 白子。行数不足时补空行，便于只写出关心的局部。
        """
        rows = [row.strip() for row in rows if row.strip() != ""]
        width = max((len(row) for row in rows), default=0)
        if size is None:
            size = max(width, len(rows), 15)
        board = cls(size)
        for r, row in enumerate(rows):
            if r >= size:
                raise ValueError(f"字符画行数 {len(rows)} 超出棋盘边长 {size}")
            for c, ch in enumerate(row):
                if c >= size:
                    raise ValueError(f"第 {r} 行宽度超出棋盘边长 {size}")
                if ch not in _CHAR_TO_STONE:
                    raise ValueError(f"无法识别的棋盘字符 {ch!r}")
                board.set(r, c, _CHAR_TO_STONE[ch])
        return board
