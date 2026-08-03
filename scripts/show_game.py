#!/usr/bin/env python3
"""跑一局对局并打印棋谱，用于人工复盘。

    python scripts/show_game.py --run-dir runs/renju15c --opponent greedy

模型一律以零搜索模式出手（单次前向 + argmax），与实际部署一致。
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.cli.engine import InstinctPlayer  # noqa: E402
from gomoku_instinct.cli.render import move_to_label, render_board  # noqa: E402
from gomoku_instinct.core import load_core  # noqa: E402
from gomoku_instinct.eval import GreedyThreatPlayer, RandomPlayer  # noqa: E402
from gomoku_instinct.eval.mcts_player import MctsPlayer  # noqa: E402
from gomoku_instinct.model.loader import load_model  # noqa: E402
from gomoku_instinct.rules import BLACK, ForbiddenSemantics, Game, Outcome, RenjuRules  # noqa: E402
from gomoku_instinct.selfplay import ModelEvaluator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--opponent", default="greedy",
                    choices=["greedy", "random", "mcts", "self"])
    ap.add_argument("--model-plays", default="black", choices=["black", "white"])
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--sims", type=int, default=400, help="opponent=mcts 时的模拟数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, meta = load_model(args.run_dir, args.device)
    size = meta["board_size"]
    core = load_core()
    rules = RenjuRules()

    me = InstinctPlayer(model, size, args.device)
    if args.opponent == "greedy":
        opp, opp_name = GreedyThreatPlayer(core.Rules(), size, seed=args.seed), "规则基线"
    elif args.opponent == "random":
        opp, opp_name = RandomPlayer(seed=args.seed), "随机"
    elif args.opponent == "mcts":
        opp = MctsPlayer(ModelEvaluator(model, size, args.device), size,
                         sims=args.sims, slots=1, threads=16)
        opp_name = f"同权重 MCTS({args.sims})"
    else:
        opp, opp_name = me, "自己"

    model_is_black = args.model_plays == "black"
    game = Game(size, rules, ForbiddenSemantics.LOSE)

    rng = random.Random(args.seed)
    for _ in range(args.opening_plies):
        game.play(rng.choice(game.legal_moves()))

    print(f"零搜索模型（step {meta['step']:,}） 执{'黑' if model_is_black else '白'}"
          f"  vs  {opp_name}")
    print(f"开局随机落 {args.opening_plies} 子\n")

    values: list[tuple[int, float]] = []
    while not game.is_terminal():
        model_turn = (game.to_move == BLACK) == model_is_black
        if model_turn:
            analysis = me.analyze(game, top_k=3)
            values.append((game.num_moves + 1, analysis.value))
            game.play(analysis.move)
        else:
            game.play(opp.choose_batch([game])[0])
        if game.num_moves >= size * size:
            break

    # 棋谱：每行 10 手
    print("棋谱")
    labels = [move_to_label(m, size) for m, _, _ in game.history]
    for start in range(0, len(labels), 10):
        row = labels[start : start + 10]
        pairs = "  ".join(f"{start + i + 1:>3}.{lab:<4}" for i, lab in enumerate(row))
        print(f"  {pairs}")

    print()
    print(render_board(game.board.grid, size, last_move=game.last_move()))

    print()
    last_move, last_color, judgment = game.history[-1]
    side = "黑" if last_color == BLACK else "白"
    if game.outcome == Outcome.DRAW:
        verdict = "和棋"
    elif judgment.is_forbidden:
        names = {1: "长连", 2: "四四", 3: "三三"}
        verdict = f"{side}方在 {move_to_label(last_move, size)} 走出{names.get(int(judgment.forbidden), '')}禁手，判负"
    else:
        winner_is_model = (last_color == BLACK) == model_is_black
        verdict = (f"{side}方 {move_to_label(last_move, size)} 成五 —— "
                   f"{'模型' if winner_is_model else opp_name}胜")
    print(f"结果：{verdict}   共 {game.num_moves} 手")

    if values:
        shown = values[:: max(1, len(values) // 8)]
        print("模型自评（+1 表示它认为自己必胜）：")
        print("  " + "   ".join(f"{ply}手 {v:+.2f}" for ply, v in shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
