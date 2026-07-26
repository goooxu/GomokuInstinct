"""光标式落子界面：方向键移动、回车落子。

15x15 上靠肉眼数格子报坐标很容易错行错列，所以默认走这个模式。
当前坐标与该点的状态（空点 / 你的禁手点 / 已占）实时显示在棋盘下方，
落子前就能确认，不必落下去才发现走错地方。

不是终端（管道、重定向）时自动退回逐行输入坐标的模式。
"""

from __future__ import annotations

import sys

from ..rules import BLACK, WHITE, Game, Outcome
from . import keyboard as kb
from .render import move_to_label, render_board

HELP_LINES = [
    "方向键 / hjkl  移动光标      回车 / 空格  落子",
    "u 悔一回合   i 让 AI 提示   s 存档   o 读档   r 认输   q 退出",
]


def _write(text: str) -> None:
    """raw 模式下换行必须带回车，否则输出会呈阶梯状。"""
    sys.stdout.write(text.replace("\n", "\r\n"))
    sys.stdout.flush()


def _clear() -> None:
    sys.stdout.write("\033[H\033[2J")


def _point_state(game: Game, move: int, forbidden: list[bool] | None) -> str:
    r, c = divmod(move, game.size)
    if not game.board.is_empty(r, c):
        return "已有子"
    if forbidden is not None and forbidden[move]:
        return "你的禁手点，落上去判负"
    return "空点"


def _describe_last(game: Game) -> str:
    if not game.history:
        return ""
    move, color, judgment = game.history[-1]
    side = "黑" if color == BLACK else "白"
    text = f"上一手 {side} {move_to_label(move, game.size)}"
    if judgment.is_forbidden:
        names = {1: "长连", 2: "四四", 3: "三三"}
        text += f"  <- {names.get(int(judgment.forbidden), '')}禁手，判负"
    return text


def _outcome_text(game: Game, human: int, opp_name: str) -> str:
    if game.outcome == Outcome.DRAW:
        return "和棋。"
    winner = BLACK if game.outcome == Outcome.BLACK_WIN else WHITE
    return f"{'黑' if winner == BLACK else '白'}方胜 —— {'你' if winner == human else opp_name}赢了。"


def run_cursor_game(game: Game, player, human: int, meta: dict, show_policy: bool) -> int:
    """光标模式的主循环。返回退出码。"""
    size = game.size
    cursor = game.last_move() if game.history else (size // 2) * size + size // 2
    message = ""
    analysis = None

    with kb.raw_mode():
        while True:
            # 每次重画都重算全盘禁手：实测约 12ms，肉眼无感，不值得为它引入
            # 「每处改动局面都要记得让缓存失效」的隐患。
            forbidden = game.forbidden_map() if game.to_move == human == BLACK else None

            _clear()
            _write(f"gomoku-instinct   你执{'黑' if human == BLACK else '白'}   "
                   f"权重 step {meta['step']:,}\n")
            _write("AI 落子完全由一次网络前向决定，不使用任何搜索。\n\n")
            _write(render_board(game.board.grid, size, last_move=game.last_move(),
                                forbidden=forbidden, cursor=cursor) + "\n")

            if forbidden is not None and any(forbidden):
                _write("× 是你的禁手点\n")
            _write(f"\n光标 {move_to_label(cursor, size)}"
                   f"（{_point_state(game, cursor, forbidden)}）\n")
            if game.history:
                _write(_describe_last(game) + "\n")
            if analysis is not None:
                _write(f"AI 自评 {analysis.value:+.3f}   候选："
                       + "  ".join(f"{move_to_label(m, size)} {p:.0%}"
                                   for m, p in analysis.top_moves) + "\n")
            if message:
                _write(f"\n{message}\n")
            _write("\n" + "\n".join(HELP_LINES) + "\n")

            if game.is_terminal():
                _write("\n" + _outcome_text(game, human, "AI") + "  按任意键退出\n")
                kb.read_key()
                return 0

            # AI 的回合
            if game.to_move != human:
                _write("\nAI 思考中……\n")
                result = player.analyze(game)
                game.play(result.move)
                analysis = result if show_policy else None
                message = f"AI 落子 {move_to_label(result.move, size)}"
                cursor = result.move
                continue

            key = kb.read_key()
            message = ""
            r, c = divmod(cursor, size)

            if key in (kb.INTERRUPT, "q"):
                _write("\n再见。\n")
                return 0
            if key == kb.UP:
                cursor = (max(0, r - 1)) * size + c
            elif key == kb.DOWN:
                cursor = (min(size - 1, r + 1)) * size + c
            elif key == kb.LEFT:
                cursor = r * size + max(0, c - 1)
            elif key == kb.RIGHT:
                cursor = r * size + min(size - 1, c + 1)
            elif key in (kb.ENTER, " "):
                if not game.board.is_empty(r, c):
                    message = "那里已经有子了。"
                else:
                    game.play(cursor)
                    analysis = None
            elif key == "u":
                for _ in range(2):
                    if game.history:
                        game.undo()
                analysis = None
                message = "已悔一回合。"
            elif key == "i":
                analysis = player.analyze(game)
                cursor = analysis.move
                message = f"AI 会走 {move_to_label(analysis.move, size)}"
            elif key == "r":
                _write("\n你认输了。\n")
                return 0
            elif key in ("s", "o"):
                message, loaded = _save_or_load(game, key)
                if loaded is not None:
                    game, analysis = loaded, None
                    cursor = game.last_move()
                    if cursor is None:
                        cursor = (size // 2) * size + size // 2
            # 其余按键忽略，避免误触


def _save_or_load(game: Game, key: str) -> tuple[str, Game | None]:
    """返回（提示信息，载入得到的新对局或 None）。

    读档不去改 game 的内部字段，而是整个换掉 —— 逐字段赋值一旦漏掉一个
    （比如 to_move），棋局会静默地走进不一致状态，比报错更难查。
    """
    from .play import load_game, save_game

    path = kb.read_line("存档文件：" if key == "s" else "读档文件：")
    if not path:
        return "已取消。", None
    try:
        if key == "s":
            save_game(game, path)
            return f"已保存到 {path}", None
        loaded = load_game(path, game.rules, game.semantics)
        return f"已载入 {path}", loaded
    except (OSError, ValueError) as exc:
        return f"失败：{exc}", None
