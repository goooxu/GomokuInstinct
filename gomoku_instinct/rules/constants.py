"""规则层的基础常量。

这些取值同时被 Python 参考实现与 C++ 实现使用，两侧必须保持一致
（C++ 侧见 csrc/constants.h，差分测试会连带校验编码一致性）。
"""

from __future__ import annotations

from enum import IntEnum

# ── 棋盘取值 ────────────────────────────────────────────────────────────────
EMPTY = 0
BLACK = 1
WHITE = 2
WALL = 3  # 棋盘外，既非空点也非任何一方的子

STONE_NAMES = {EMPTY: ".", BLACK: "X", WHITE: "O", WALL: "#"}


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


# ── 方向 ────────────────────────────────────────────────────────────────────
# 只取四个方向：每条直线只算一次，反向是同一条线。
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),   # 横
    (1, 0),   # 竖
    (1, 1),   # 主对角（左上→右下）
    (1, -1),  # 副对角（右上→左下）
)

# 抽取一条直线时向两侧各取多少格。
#
# 这个半径不是随手定的：三三判定需要在根手周围 ±4 的空点上试放一子，
# 再检查该子形成的五连窗口（再 ±4），最后还要检查落点两侧的延伸是否
# 构成长连（再 ±1）。累计 4 + 4 + 1 = 9，所以半径必须至少为 9。
LINE_RADIUS = 9
LINE_LENGTH = 2 * LINE_RADIUS + 1
LINE_CENTER = LINE_RADIUS


class Outcome(IntEnum):
    """一手棋落下之后的对局状态。"""

    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3


class Forbidden(IntEnum):
    """黑方禁手类型。仅对黑方有意义。"""

    NONE = 0
    OVERLINE = 1      # 长连（≥6）
    DOUBLE_FOUR = 2   # 四四
    DOUBLE_THREE = 3  # 三三


class Level(IntEnum):
    """某个空点落子后能形成的最高棋型等级，用作辅助监督标签。

    按「最高等级」归类：一个方向若已成四，就不再计作三。
    取值必须与 csrc/constants.h 的 Level 一致。

    注意这里的「活三」用的是**非递归**判定（只看能否一手成活四），
    与三三禁手里那个递归定义不同 —— 它只是个特征标签，不参与规则判定；
    精确的禁手规则由单独的 forbidden 头去学。
    """

    NONE = 0
    CLOSED_THREE = 1   # 眠三：再一手可成四，但成不了活四
    OPEN_THREE = 2     # 活三：再一手可成活四
    FOUR = 3           # 冲四：只有一个成五点
    OPEN_FOUR = 4      # 活四：两个成五点
    FIVE = 5           # 五连
    OVERLINE = 6       # 长连（黑方为禁手；白方按成五算）


NUM_LEVELS = 7


class ForbiddenSemantics(IntEnum):
    """禁手点在引擎里的语义。"""

    LOSE = 0     # 严格 RIF：禁手点仍是合法落子，黑方落上去立即判负
    ILLEGAL = 1  # 引擎常见做法：禁手点直接从合法落子集中移除
