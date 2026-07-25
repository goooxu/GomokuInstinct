"""终端棋盘渲染与坐标解析。

坐标沿用连珠的习惯：列用字母 A 起（跳过 I 容易混淆，这里不跳，标注清楚即可），
行用数字，**1 在底部**。所以 15x15 的天元是 H8。
"""

from __future__ import annotations

from ..rules.constants import BLACK, EMPTY, WHITE

COLUMN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

BLACK_STONE = "●"
WHITE_STONE = "○"
# 最后一手用不同字形标出，而不是加括号 —— 加括号会破坏列对齐，棋盘就不好看了
BLACK_LAST = "◆"
WHITE_LAST = "◇"
EMPTY_POINT = "·"
FORBIDDEN_MARK = "×"
CANDIDATE_MARK = "+"


class CoordinateError(ValueError):
    pass


def move_to_label(move: int, size: int) -> str:
    r, c = divmod(move, size)
    return f"{COLUMN_LETTERS[c]}{size - r}"


def label_to_move(label: str, size: int) -> int:
    text = label.strip().upper()
    if len(text) < 2:
        raise CoordinateError(f"看不懂的坐标: {label!r}")

    column = COLUMN_LETTERS.find(text[0])
    if column < 0 or column >= size:
        raise CoordinateError(f"列超出范围: {text[0]}")

    try:
        row_number = int(text[1:])
    except ValueError as exc:
        raise CoordinateError(f"看不懂的行号: {text[1:]!r}") from exc
    if not 1 <= row_number <= size:
        raise CoordinateError(f"行超出范围: {row_number}")

    return (size - row_number) * size + column


def render_board(
    grid,
    size: int,
    *,
    last_move: int | None = None,
    forbidden: list[bool] | None = None,
    highlight: set[int] | None = None,
) -> str:
    """把棋盘画成字符串。

    最后一手用实心/空心菱形标出，forbidden 里的空点标 ×（给执黑的人看），
    highlight 里的空点标 +（用于展示 AI 的候选点）。
    """
    width = len(str(size))
    header = " " * (width + 1) + " ".join(COLUMN_LETTERS[:size])
    lines = [header]

    for r in range(size):
        cells = []
        for c in range(size):
            idx = r * size + c
            value = grid[idx]
            if value == BLACK:
                ch = BLACK_LAST if idx == last_move else BLACK_STONE
            elif value == WHITE:
                ch = WHITE_LAST if idx == last_move else WHITE_STONE
            elif forbidden is not None and forbidden[idx]:
                ch = FORBIDDEN_MARK
            elif highlight is not None and idx in highlight:
                ch = CANDIDATE_MARK
            else:
                ch = EMPTY_POINT
            cells.append(ch)
        lines.append(f"{size - r:>{width}} " + " ".join(cells))

    lines.append(header)
    return "\n".join(lines)


def render_compact(grid, size: int, last_move: int | None = None) -> str:
    """紧凑渲染，每格一列，用于日志与复盘。"""
    width = len(str(size))
    header = " " * (width + 1) + COLUMN_LETTERS[:size]
    lines = [header]
    for r in range(size):
        row = []
        for c in range(size):
            idx = r * size + c
            value = grid[idx]
            row.append(
                BLACK_STONE if value == BLACK
                else WHITE_STONE if value == WHITE
                else EMPTY_POINT
            )
        lines.append(f"{size - r:>{width}} " + "".join(row))
    return "\n".join(lines)
