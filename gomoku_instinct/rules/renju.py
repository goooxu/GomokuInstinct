"""连珠规则判定（Python 参考实现）。

这是整个项目的规则规范。所有训练信号都源自这里，因此实现优先追求「明显正确」
而非快速；高性能版本在 csrc/ 下，两者由差分测试锁定一致。

规则要点（RIF 标准，自由开局）：

  * 白方无禁手，五连及以上（含长连）即胜。
  * 黑方**恰好**五连才算胜；长连（≥6）、四四、三三均为禁手。
  * 黑方同时成五与触发禁手时，五连优先判黑胜。
  * 禁手点的语义由配置决定：严格 RIF 是「仍可落子但立即判负」。

两处容易写错的地方，单独说明：

**「四」的计数**  ——  RIF 把「四」定义为「一条五格窗口内己方占四格、余一格为空，
补上即成五」。照字面数窗口会出问题：活四 `.XXXX.` 含有两个这样的窗口
（左端补五、右端补五），会被误判成四四禁手，但活四明明是黑方的正常胜着。

真正的判据是：**两个四是否共用同一组四颗子**。

    .XXXX.     两个窗口的黑子集合都是 {1,2,3,4}   → 同一个四 → 活四，合法
    X.XXX.X    窗口黑子集合分别是 {0,2,3,4} 和 {2,3,4,6} → 两个四 → 四四禁手

所以下面按「黑子位置集合」去重来统计四的个数，而不是按窗口或按成五点。
同一组黑子若有两个成五点，那就是活四，计作一个四。

**「活三」的递归定义**  ——  一个三算活三，当且仅当存在一个空点，落子后能成活四，
**且那一手本身不是禁手**。判断那一手是否禁手又要递归地数活三，所以这是个递归定义。
实践中递归深度极少超过 3，但仍设了深度上限做兜底。
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import Board
from .constants import (
    BLACK,
    DIRECTIONS,
    EMPTY,
    LINE_CENTER,
    LINE_RADIUS,
    Forbidden,
    Outcome,
    WHITE,
)


@dataclass(frozen=True)
class MoveJudgment:
    """一手棋落下之后的判定结果。"""

    outcome: Outcome
    forbidden: Forbidden = Forbidden.NONE
    fours: int = 0
    open_threes: int = 0
    longest_run: int = 0

    @property
    def is_forbidden(self) -> bool:
        return self.forbidden != Forbidden.NONE


# ── 一维直线上的基础判定 ────────────────────────────────────────────────────
# 下面这些函数只认「一条直线 + 一个下标」，不关心二维坐标，
# 因此四个方向可以共用同一套逻辑。


def _run_length(line: list[int], idx: int, color: int) -> int:
    """经过 idx 的同色连子长度（含 idx 本身）。"""
    n = len(line)
    left = idx - 1
    while left >= 0 and line[left] == color:
        left -= 1
    right = idx + 1
    while right < n and line[right] == color:
        right += 1
    return right - left - 1


def _completes_five(line: list[int], p: int, color: int, exact: bool) -> bool:
    """在空点 p 落子后是否成五。

    exact=True（黑方）要求恰好五连：长连是禁手，不算成五，因此也不构成「四」。
    exact=False（白方）五连及以上都算。
    """
    line[p] = color
    run = _run_length(line, p, color)
    line[p] = EMPTY
    return run == 5 if exact else run >= 5


def _four_groups(
    line: list[int], idx: int, color: int, exact_five: bool
) -> dict[frozenset[int], set[int]]:
    """统计 idx 这一手在本方向上造成的「四」。

    返回 {四颗子的位置集合: {该组的成五点集合}}。按位置集合去重，是为了让活四
    的两个成五点归到同一个四上（见模块开头的说明）。
    """
    n = len(line)
    groups: dict[frozenset[int], set[int]] = {}

    # 只看包含 idx 的五格窗口 —— 我们数的是「这一手造成的四」。
    for start in range(idx - 4, idx + 1):
        if start < 0 or start + 4 >= n:
            continue
        own: list[int] = []
        gap = -1
        broken = False
        for i in range(start, start + 5):
            v = line[i]
            if v == color:
                own.append(i)
            elif v == EMPTY:
                if gap >= 0:  # 两个及以上空点，不是四
                    broken = True
                    break
                gap = i
            else:  # 对方子或墙
                broken = True
                break
        if broken or len(own) != 4:
            continue
        if not _completes_five(line, gap, color, exact_five):
            continue
        groups.setdefault(frozenset(own), set()).add(gap)

    return groups


def _count_fours(
    line: list[int], idx: int, color: int, exact_five: bool
) -> tuple[int, bool]:
    """返回 (四的个数, 是否含活四)。"""
    groups = _four_groups(line, idx, color, exact_five)
    return len(groups), any(len(points) >= 2 for points in groups.values())


# ── 规则判定器 ──────────────────────────────────────────────────────────────


class RenjuRules:
    """连珠规则判定器。

    参数对应 configs/rules.yaml；关闭全部禁手即退化为自由五子棋（≥5 连即胜），
    用于小棋盘快速验证与消融实验。
    """

    def __init__(
        self,
        *,
        forbidden_enabled: bool = True,
        overline: bool = True,
        double_four: bool = True,
        double_three: bool = True,
        five_overrides_forbidden: bool = True,
        white_overline_wins: bool = True,
        recursion_depth: int = 64,
    ) -> None:
        self.forbidden_enabled = forbidden_enabled
        self.overline = overline
        self.double_four = double_four
        self.double_three = double_three
        self.five_overrides_forbidden = five_overrides_forbidden
        self.white_overline_wins = white_overline_wins
        self.recursion_depth = recursion_depth
        # 递归封顶发生的次数。应当恒为 0；非零说明深度上限设小了。
        self.depth_exceeded = 0
        # 实际达到过的最大递归深度，用来验证上限是否合理。
        self.max_depth_seen = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "RenjuRules":
        rules = cfg.get("rules", cfg)
        forb = rules.get("forbidden", {})
        return cls(
            forbidden_enabled=forb.get("enabled", True),
            overline=forb.get("overline", True),
            double_four=forb.get("double_four", True),
            double_three=forb.get("double_three", True),
            five_overrides_forbidden=forb.get("five_overrides_forbidden", True),
            white_overline_wins=rules.get("white_overline_wins", True),
            recursion_depth=forb.get("recursion_depth", 64),
        )

    # ── 对外接口 ────────────────────────────────────────────────────────────
    def judge(self, board: Board, r: int, c: int, color: int) -> MoveJudgment:
        """判定在空点 (r, c) 落 color 子的结果。调用后棋盘保持原样。"""
        if board.cell(r, c) != EMPTY:
            raise ValueError(f"({r}, {c}) 不是空点")
        board.set(r, c, color)
        try:
            return self._judge_placed(board, r, c, color, depth=0)
        finally:
            board.set(r, c, EMPTY)

    def is_forbidden(self, board: Board, r: int, c: int) -> bool:
        """(r, c) 对黑方是否为禁手点。"""
        if not self.forbidden_enabled:
            return False
        return self.judge(board, r, c, BLACK).is_forbidden

    def forbidden_map(self, board: Board) -> list[bool]:
        """全盘禁手点标记，长度 size*size。用作辅助监督标签与人类提示。"""
        size = board.size
        out = [False] * (size * size)
        if not self.forbidden_enabled:
            return out
        for idx in range(size * size):
            if board.grid[idx] != EMPTY:
                continue
            r, c = divmod(idx, size)
            out[idx] = self.judge(board, r, c, BLACK).is_forbidden
        return out

    def judge_all(self, board: Board, color: int) -> list[MoveJudgment | None]:
        """对每个空点做一次完整判定；非空点为 None。

        与 C++ 侧 Rules.judge_all 一一对应，是差分测试的比对单位。
        """
        size = board.size
        out: list[MoveJudgment | None] = [None] * (size * size)
        for idx in range(size * size):
            if board.grid[idx] != EMPTY:
                continue
            r, c = divmod(idx, size)
            out[idx] = self.judge(board, r, c, color)
        return out

    # ── 内部实现 ────────────────────────────────────────────────────────────
    def _judge_placed(
        self, board: Board, r: int, c: int, color: int, depth: int
    ) -> MoveJudgment:
        """棋子已经放在盘上时的判定。"""
        if depth > self.max_depth_seen:
            self.max_depth_seen = depth
        lines = [board.line(r, c, dr, dc, LINE_RADIUS) for dr, dc in DIRECTIONS]
        runs = [_run_length(line, LINE_CENTER, color) for line in lines]
        longest = max(runs)

        if color == WHITE:
            wins = longest >= 5 if self.white_overline_wins else longest == 5
            return MoveJudgment(
                Outcome.WHITE_WIN if wins else Outcome.ONGOING, longest_run=longest
            )

        # 关闭禁手即退化为自由五子棋，黑白规则对称。
        if not self.forbidden_enabled:
            wins = longest >= 5 if self.white_overline_wins else longest == 5
            return MoveJudgment(
                Outcome.BLACK_WIN if wins else Outcome.ONGOING, longest_run=longest
            )

        has_five = any(run == 5 for run in runs)
        has_overline = longest >= 6

        # 五连优先：同时成五与触发禁手时判黑胜。
        if has_five and self.five_overrides_forbidden:
            return MoveJudgment(Outcome.BLACK_WIN, longest_run=longest)

        if self.overline and has_overline:
            return MoveJudgment(
                Outcome.WHITE_WIN, Forbidden.OVERLINE, longest_run=longest
            )

        total_fours = 0
        for line in lines:
            count, _ = _count_fours(line, LINE_CENTER, BLACK, exact_five=True)
            total_fours += count
        if self.double_four and total_fours >= 2:
            return MoveJudgment(
                Outcome.WHITE_WIN,
                Forbidden.DOUBLE_FOUR,
                fours=total_fours,
                longest_run=longest,
            )

        open_threes = 0
        if self.double_three:
            for line, (dr, dc) in zip(lines, DIRECTIONS):
                # 已经成四的方向按最高等级归类为四，不再计作三。
                count, _ = _count_fours(line, LINE_CENTER, BLACK, exact_five=True)
                if count:
                    continue
                if self._has_open_three(board, r, c, line, dr, dc, depth):
                    open_threes += 1
            if open_threes >= 2:
                return MoveJudgment(
                    Outcome.WHITE_WIN,
                    Forbidden.DOUBLE_THREE,
                    fours=total_fours,
                    open_threes=open_threes,
                    longest_run=longest,
                )

        if has_five:  # five_overrides_forbidden 关闭时走到这里
            return MoveJudgment(Outcome.BLACK_WIN, longest_run=longest)

        return MoveJudgment(
            Outcome.ONGOING,
            fours=total_fours,
            open_threes=open_threes,
            longest_run=longest,
        )

    def _has_open_three(
        self,
        board: Board,
        r: int,
        c: int,
        line: list[int],
        dr: int,
        dc: int,
        depth: int,
    ) -> bool:
        """本方向上是否形成活三。

        活三 = 存在一个空点，落子后在本方向成活四，且那一手本身不是禁手。
        """
        n = len(line)
        for p in range(max(0, LINE_CENTER - 4), min(n, LINE_CENTER + 5)):
            if line[p] != EMPTY:
                continue

            line[p] = BLACK
            run = _run_length(line, p, BLACK)
            _, makes_open_four = _count_fours(line, p, BLACK, exact_five=True)
            line[p] = EMPTY

            if run >= 5:
                # 该点直接成五或长连，说明原形已经是四，不是三。
                continue
            if not makes_open_four:
                continue

            if depth >= self.recursion_depth:
                # 递归封顶：按「那一手可下」处理，即认定为活三。
                # 正常对局中不会触发，计数供审计。
                self.depth_exceeded += 1
                return True

            pr = r + (p - LINE_CENTER) * dr
            pc = c + (p - LINE_CENTER) * dc
            board.set(pr, pc, BLACK)
            try:
                sub = self._judge_placed(board, pr, pc, BLACK, depth + 1)
            finally:
                board.set(pr, pc, EMPTY)

            if not sub.is_forbidden:
                return True

        return False
