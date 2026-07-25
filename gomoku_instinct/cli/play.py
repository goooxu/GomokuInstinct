"""人机对战的交互循环。"""

from __future__ import annotations

import os

import torch

from ..model.loader import load_model
from ..rules import BLACK, WHITE, ForbiddenSemantics, Game, Outcome, RenjuRules
from .engine import InstinctPlayer
from .render import CoordinateError, label_to_move, move_to_label, render_board

HELP = """
可用命令：
  <坐标>    落子，例如 H8
  undo      悔一回合（连同 AI 的应手一起退回）
  hint      让 AI 说说它怎么看当前局面（不落子）
  save <文件>  保存对局
  load <文件>  载入对局
  resign    认输
  help      显示本帮助
  quit      退出
"""


def _outcome_text(outcome: Outcome, human_color: int) -> str:
    if outcome == Outcome.DRAW:
        return "和棋。"
    winner = BLACK if outcome == Outcome.BLACK_WIN else WHITE
    side = "黑" if winner == BLACK else "白"
    who = "你" if winner == human_color else "AI"
    return f"{side}方胜 —— {who}赢了。"


def _describe_last_move(game: Game) -> str:
    if not game.history:
        return ""
    move, color, judgment = game.history[-1]
    side = "黑" if color == BLACK else "白"
    label = move_to_label(move, game.size)
    text = f"{side} {label}"
    if judgment.is_forbidden:
        names = {1: "长连", 2: "四四", 3: "三三"}
        text += f"  <- {names.get(int(judgment.forbidden), '禁手')}禁手，判负"
    elif judgment.fours or judgment.open_threes:
        parts = []
        if judgment.fours:
            parts.append(f"{judgment.fours} 个四")
        if judgment.open_threes:
            parts.append(f"{judgment.open_threes} 个活三")
        text += f"  ({', '.join(parts)})"
    return text


def _print_analysis(analysis, size: int) -> None:
    print(f"  AI 评估：{analysis.value:+.3f}（+1 表示它认为自己必胜）")
    print("  候选手：", end="")
    print(
        "  ".join(
            f"{move_to_label(m, size)} {p:.1%}" for m, p in analysis.top_moves
        )
    )
    if analysis.forbidden_pred:
        risky = sorted(
            ((p, i) for i, p in enumerate(analysis.forbidden_pred) if p > 0.5),
            reverse=True,
        )[:5]
        if risky:
            print(
                "  模型判断的黑方禁手点：",
                "  ".join(f"{move_to_label(i, size)} {p:.0%}" for p, i in risky),
            )
    if analysis.masked_forbidden:
        print("  （safe-mode 生效：本次为 AI 屏蔽了禁手点）")


def save_game(game: Game, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"size {game.size}\n")
        fh.write(
            " ".join(move_to_label(m, game.size) for m, _, _ in game.history) + "\n"
        )


def load_game(path: str, rules: RenjuRules, semantics: ForbiddenSemantics) -> Game:
    with open(path) as fh:
        lines = [line.strip() for line in fh if line.strip()]
    size = 15
    moves: list[str] = []
    for line in lines:
        if line.startswith("size "):
            size = int(line.split()[1])
        else:
            moves.extend(line.split())
    game = Game(size, rules, semantics)
    for label in moves:
        game.play(label_to_move(label, size))
    return game


def run(
    checkpoint: str,
    human_color: str = "black",
    device: str = "cpu",
    show_policy: bool = False,
    safe_mode: bool = False,
    temperature: float = 0.0,
) -> int:
    model, meta = load_model(checkpoint, device)
    size = meta["board_size"]
    rules = RenjuRules()
    game = Game(size, rules, ForbiddenSemantics.LOSE)
    player = InstinctPlayer(
        model, size, device, safe_mode=safe_mode, temperature=temperature
    )

    human = BLACK if human_color.lower().startswith("b") else WHITE
    print(f"gomoku-instinct  棋盘 {size}x{size}  你执{'黑' if human == BLACK else '白'}")
    print(f"权重：{os.path.basename(meta['path'])}（训练到 step {meta['step']:,}）")
    print("AI 落子完全由一次网络前向决定，不使用任何搜索。")
    if safe_mode:
        print("safe-mode 已开启：会用规则替 AI 屏蔽禁手点。")
    print(HELP)

    while True:
        forbidden = game.forbidden_map() if game.to_move == human == BLACK else None
        print()
        print(render_board(game.board.grid, size, last_move=game.last_move(),
                           forbidden=forbidden))
        if game.history:
            print("上一手：" + _describe_last_move(game))
        if forbidden is not None and any(forbidden):
            print("× 是你的禁手点，落上去立即判负。")

        if game.is_terminal():
            print()
            print(_outcome_text(game.outcome, human))
            return 0

        if game.to_move != human:
            analysis = player.analyze(game)
            game.play(analysis.move)
            print(f"\nAI 落子 {move_to_label(analysis.move, size)}")
            if show_policy:
                _print_analysis(analysis, size)
            continue

        try:
            raw = input("你的落子> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not raw:
            continue

        command, _, argument = raw.partition(" ")
        command = command.lower()

        if command in ("quit", "exit", "q"):
            print("再见。")
            return 0
        if command == "help":
            print(HELP)
            continue
        if command == "resign":
            print("你认输了。")
            return 0
        if command == "undo":
            for _ in range(2):
                if game.history:
                    game.undo()
            continue
        if command == "hint":
            analysis = player.analyze(game)
            print(f"AI 会走 {move_to_label(analysis.move, size)}")
            _print_analysis(analysis, size)
            continue
        if command == "save":
            if not argument:
                print("用法：save <文件>")
                continue
            save_game(game, argument)
            print(f"已保存到 {argument}")
            continue
        if command == "load":
            if not argument:
                print("用法：load <文件>")
                continue
            try:
                game = load_game(argument, rules, ForbiddenSemantics.LOSE)
                print(f"已载入 {argument}")
            except (OSError, CoordinateError, ValueError) as exc:
                print(f"载入失败：{exc}")
            continue

        try:
            move = label_to_move(raw, size)
        except CoordinateError as exc:
            print(f"{exc}（输入 help 看用法）")
            continue
        if not game.board.is_empty(*divmod(move, size)):
            print("那里已经有子了。")
            continue

        game.play(move)
