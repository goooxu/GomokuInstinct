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

# 光标用反显而非加括号：括号会占额外列宽、把整行对齐搞乱，而棋盘一旦错列，
# 「看错落子位置」这个本来要解决的问题反而更严重了。
INVERSE_ON = "\033[7m"
INVERSE_OFF = "\033[27m"


def move_to_label(move: int, size: int) -> str:
    r, c = divmod(move, size)
    return f"{COLUMN_LETTERS[c]}{size - r}"


def render_board(
    grid,
    size: int,
    *,
    last_move: int | None = None,
    forbidden: list[bool] | None = None,
    highlight: set[int] | None = None,
    cursor: int | None = None,
) -> str:
    """把棋盘画成字符串。

    最后一手用实心/空心菱形标出，forbidden 里的空点标 ×（给执黑的人看），
    highlight 里的空点标 +（用于展示 AI 的候选点），cursor 处反显。
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
            if idx == cursor:
                ch = f"{INVERSE_ON}{ch}{INVERSE_OFF}"
            cells.append(ch)
        lines.append(f"{size - r:>{width}} " + " ".join(cells))

    lines.append(header)
    return "\n".join(lines)
