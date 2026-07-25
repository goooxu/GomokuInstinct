"""按 RIF 规则条文构造的禁手/非禁手用例。

这些是**规则单元测试向量**，由规则条文构造而来，不是棋谱，也不进入训练或测试数据。

每个用例都在注释里写清楚了棋型和判定理由 —— 禁手判定是整个项目唯一
「写错了整个训练信号全错」的地方，用例必须能被人肉复核。
"""

from __future__ import annotations

import pytest

from gomoku_instinct.rules import (
    BLACK,
    WHITE,
    Board,
    Forbidden,
    ForbiddenSemantics,
    Game,
    Outcome,
    RenjuRules,
)

Coord = tuple[int, int]


def mk(black: list[Coord] = (), white: list[Coord] = (), size: int = 15) -> Board:
    board = Board(size)
    for r, c in black:
        board.set(r, c, BLACK)
    for r, c in white:
        board.set(r, c, WHITE)
    return board


@pytest.fixture
def rules() -> RenjuRules:
    return RenjuRules()


# ── 成五与长连 ──────────────────────────────────────────────────────────────


def test_black_exact_five_wins(rules):
    """行 7 的 cols 3-6 已有黑子，落 col 7 恰好成五。"""
    board = mk(black=[(7, 3), (7, 4), (7, 5), (7, 6)])
    j = rules.judge(board, 7, 7, BLACK)
    assert j.outcome == Outcome.BLACK_WIN
    assert not j.is_forbidden
    assert j.longest_run == 5


def test_black_overline_is_forbidden(rules):
    """cols 3,4,5 与 7,8 已有黑子，落 col 6 把两段接成 6 连 —— 长连禁手。"""
    board = mk(black=[(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
    j = rules.judge(board, 7, 6, BLACK)
    assert j.forbidden == Forbidden.OVERLINE
    assert j.outcome == Outcome.WHITE_WIN
    assert j.longest_run == 6


def test_five_overrides_overline_in_another_direction(rules):
    """同一手在横向恰好成五、在纵向成长连 —— RIF 规定五连优先，判黑胜。"""
    board = mk(
        black=[
            (7, 3), (7, 4), (7, 5), (7, 6),          # 横向：落 (7,7) 后 cols 3-7 恰好五连
            (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),  # 纵向：落 (7,7) 后 rows 2-7 成 6 连
        ]
    )
    j = rules.judge(board, 7, 7, BLACK)
    assert j.outcome == Outcome.BLACK_WIN
    assert not j.is_forbidden


def test_five_priority_can_be_disabled():
    """关掉五连优先后，同一个局面应判长连禁手 —— 确认这条规则真的在起作用。"""
    board = mk(
        black=[
            (7, 3), (7, 4), (7, 5), (7, 6),
            (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
        ]
    )
    r = RenjuRules(five_overrides_forbidden=False)
    j = r.judge(board, 7, 7, BLACK)
    assert j.forbidden == Forbidden.OVERLINE


# ── 四四禁手 ────────────────────────────────────────────────────────────────


def test_double_four_across_two_directions(rules):
    """横向与纵向各成一个冲四（两端各被白子挡住一侧）—— 四四禁手。"""
    board = mk(
        black=[(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)],
        white=[(7, 3), (3, 7)],
    )
    j = rules.judge(board, 7, 7, BLACK)
    assert j.forbidden == Forbidden.DOUBLE_FOUR
    assert j.fours == 2


def test_same_line_double_four(rules):
    """同一条线上的双四：X.XXX.X

    落子前 cols 3,5,7,9 为黑，落 col 6 后 cols 3-9 呈 `X.XXX.X`：

        col   3 4 5 6 7 8 9
              X . X X X . X

    包含 col6 的五格窗口里有两个「四」，黑子集合分别是 {3,5,6,7} 与 {5,6,7,9}，
    两组子不同 —— 是两个独立的四，构成四四禁手。

    这个用例正是活四与同线双四的分界点：活四 `.XXXX.` 的两个成五点共用同一组
    四颗子，只算一个四；这里的两个四各有各的子，算两个。
    """
    board = mk(black=[(7, 3), (7, 5), (7, 7), (7, 9)])
    j = rules.judge(board, 7, 6, BLACK)
    assert j.forbidden == Forbidden.DOUBLE_FOUR
    assert j.fours == 2


def test_open_four_is_not_double_four(rules):
    """活四 `.XXXX.` 两端都能成五，但共用同一组四颗子，只算一个四 —— 合法。"""
    board = mk(black=[(7, 4), (7, 5), (7, 6)])
    j = rules.judge(board, 7, 7, BLACK)
    assert not j.is_forbidden
    assert j.outcome == Outcome.ONGOING
    assert j.fours == 1


def test_single_closed_four_is_legal(rules):
    """一端被白子挡住的冲四，只有一个四 —— 合法。"""
    board = mk(black=[(7, 4), (7, 5), (7, 6)], white=[(7, 3)])
    j = rules.judge(board, 7, 7, BLACK)
    assert not j.is_forbidden
    assert j.fours == 1


# ── 三三禁手 ────────────────────────────────────────────────────────────────


def test_double_three(rules):
    """横向与纵向各形成一个活三 —— 三三禁手。

    落 (7,7) 后：
      横向 cols 5,6,7 成 `..XXX..`，落 col 4 可成活四；
      纵向 rows 5,6,7 成同样的形状。
    两个方向都是真活三（造活四的那一手本身都不是禁手），因此判三三禁手。
    """
    board = mk(black=[(7, 5), (7, 6), (5, 7), (6, 7)])
    j = rules.judge(board, 7, 7, BLACK)
    assert j.forbidden == Forbidden.DOUBLE_THREE
    assert j.open_threes == 2
    assert j.outcome == Outcome.WHITE_WIN


def test_blocked_three_is_not_open_three(rules):
    """把横向那个三的一端用白子堵死，它就成不了活四，因而不是活三。

    只剩纵向一个活三 —— 不构成三三禁手。
    """
    board = mk(
        black=[(7, 5), (7, 6), (5, 7), (6, 7)],
        white=[(7, 4)],
    )
    j = rules.judge(board, 7, 7, BLACK)
    assert not j.is_forbidden
    assert j.open_threes == 1


def test_split_three_is_open_three(rules):
    """跳三 `.X.XX.` 也是活三：补上中间的空点即成活四。

    落子前 cols 5,8,9 为黑，落 col 7 后 cols 5-9 呈 `X.XX` 加空位；
    再落 col 6 即成 cols 5-9 的活四。
    """
    board = mk(black=[(7, 5), (7, 8), (5, 7), (6, 7)])
    j = rules.judge(board, 7, 7, BLACK)
    # 横向为跳三、纵向为连三，两个活三 -> 三三禁手
    assert j.forbidden == Forbidden.DOUBLE_THREE
    assert j.open_threes == 2


def test_three_plus_four_is_legal(rules):
    """四三是连珠的正常攻击手段，不是禁手（禁手只管四四与三三）。"""
    board = mk(
        black=[(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)],
        white=[(7, 3)],
    )
    j = rules.judge(board, 7, 7, BLACK)
    assert not j.is_forbidden
    assert j.fours == 1
    assert j.open_threes == 1


# ── 白方无禁手 ──────────────────────────────────────────────────────────────


def test_white_overline_wins(rules):
    """白方长连算胜。"""
    board = mk(white=[(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
    j = rules.judge(board, 7, 6, WHITE)
    assert j.outcome == Outcome.WHITE_WIN


def test_white_double_four_is_legal(rules):
    """同样的四四形状，白方走没有任何问题。"""
    board = mk(
        white=[(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)],
        black=[(7, 3), (3, 7)],
    )
    j = rules.judge(board, 7, 7, WHITE)
    assert not j.is_forbidden
    assert j.outcome == Outcome.ONGOING


def test_white_double_three_is_legal(rules):
    board = mk(white=[(7, 5), (7, 6), (5, 7), (6, 7)])
    j = rules.judge(board, 7, 7, WHITE)
    assert not j.is_forbidden


# ── 自由五子棋（关闭禁手）──────────────────────────────────────────────────


def test_freestyle_black_overline_wins():
    """关闭禁手后退化为自由五子棋，黑方长连也算胜。"""
    board = mk(black=[(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
    r = RenjuRules(forbidden_enabled=False)
    j = r.judge(board, 7, 6, BLACK)
    assert j.outcome == Outcome.BLACK_WIN
    assert not j.is_forbidden


# ── 禁手点全盘标记 ──────────────────────────────────────────────────────────


def test_forbidden_map_empty_board(rules):
    board = Board(15)
    assert not any(rules.forbidden_map(board))


def test_forbidden_map_finds_the_double_three(rules):
    board = mk(black=[(7, 5), (7, 6), (5, 7), (6, 7)])
    fmap = rules.forbidden_map(board)
    assert fmap[7 * 15 + 7]
    assert sum(fmap) >= 1


# ── 对局状态机与禁手语义 ────────────────────────────────────────────────────


def test_lose_semantics_forbidden_point_is_playable_and_loses(rules):
    """严格 RIF：禁手点仍在合法落子集里，黑方落上去立即判负。"""
    game = Game(15, rules, ForbiddenSemantics.LOSE)
    game.board.grid[:] = mk(black=[(7, 5), (7, 6), (5, 7), (6, 7)]).grid
    move = 7 * 15 + 7
    assert move in game.legal_moves()
    game.play(move)
    assert game.outcome == Outcome.WHITE_WIN


def test_illegal_semantics_removes_forbidden_point(rules):
    """ILLEGAL 语义：禁手点直接从合法落子集中移除，落上去报错。"""
    game = Game(15, rules, ForbiddenSemantics.ILLEGAL)
    game.board.grid[:] = mk(black=[(7, 5), (7, 6), (5, 7), (6, 7)]).grid
    move = 7 * 15 + 7
    assert move not in game.legal_moves()
    with pytest.raises(ValueError):
        game.play(move)


def test_game_undo_restores_state(rules):
    game = Game(15, rules)
    moves = [7 * 15 + 7, 7 * 15 + 8, 8 * 15 + 8, 6 * 15 + 6]
    for m in moves:
        game.play(m)
    snapshot = bytes(game.board.grid)
    game.play(9 * 15 + 9)
    game.undo()
    assert bytes(game.board.grid) == snapshot
    assert game.to_move == BLACK if len(moves) % 2 == 0 else WHITE
    assert game.outcome == Outcome.ONGOING


class _AllPointsForbidden(RenjuRules):
    """把所有点都判成禁手，用来单独验证状态机分支。

    「黑方所有空点都是禁手」这个局面在真实 15x15 对局中几乎不可能出现
    （活三需要周围有空间，棋盘越满越不可能成立），所以这里不去硬凑局面，
    而是直接替换判定器来覆盖这条代码路径。
    """

    def is_forbidden(self, board, r, c):  # noqa: D102
        return True


def test_white_wins_when_black_has_no_legal_move_under_illegal_semantics():
    """ILLEGAL 语义下黑方无合法点可下时判黑负。"""
    game = Game(15, _AllPointsForbidden(), ForbiddenSemantics.ILLEGAL)
    game.to_move = WHITE
    game.play(7 * 15 + 7)
    assert game.outcome == Outcome.WHITE_WIN


# ── 递归深度审计 ────────────────────────────────────────────────────────────


def test_recursion_never_hits_depth_cap(rules):
    """上面所有用例都不该触发递归封顶 —— 触发说明深度上限需要复核。"""
    for board, move in [
        (mk(black=[(7, 5), (7, 6), (5, 7), (6, 7)]), (7, 7)),
        (mk(black=[(7, 3), (7, 5), (7, 7), (7, 9)]), (7, 6)),
        (mk(black=[(7, 4), (7, 5), (7, 6)]), (7, 7)),
    ]:
        rules.judge(board, *move, BLACK)
    assert rules.depth_exceeded == 0


# ── 直线抽取 ────────────────────────────────────────────────────────────────


def test_line_extraction_pads_with_wall():
    from gomoku_instinct.rules.constants import LINE_CENTER, LINE_LENGTH, WALL

    board = mk(black=[(0, 0)])
    line = board.line(0, 0, 0, 1)
    assert len(line) == LINE_LENGTH
    assert line[LINE_CENTER] == BLACK
    assert all(v == WALL for v in line[:LINE_CENTER])  # 左侧全部越界
