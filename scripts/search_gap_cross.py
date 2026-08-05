#!/usr/bin/env python3
"""用**同一把尺子**量两个模型的零搜索策略。

    python scripts/search_gap_cross.py --raw runs/renju15f --mcts runs/renju15c \
        --games 400 --sims 8,16,32,64

## 为什么需要这个

`search_gap.py` 量的是「零搜索 vs **它自己的** MCTS」。那个数字回答的是
"这一份权重里，搜索还贡献了多少"，是本项目的头号指标。

但它**不能用来横向比较两个模型**，因为分母各不相同：网络变强，它的 MCTS 也跟着变强。
两个模型的曲线交叉时（实测就交叉了），"谁的零搜索更接近自己的搜索版本"
有两种读法，而同权重测量分不开：

1. A 的零搜索策略确实更接近它的搜索版本（想要的）
2. A 的 MCTS 在高模拟数下没跟着网络一起变强那么多（不想要的）

把 MCTS 固定成同一份权重，分母就一样了，分子可以直接比。

代价是这时它**不再是头号指标**——固定尺子测的是"相对于某个外部对手的绝对棋力"，
而头号指标问的是"搜索还没被压进权重的那部分有多大"。两者回答不同的问题，
不要混用。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.cli.engine import InstinctPlayer  # noqa: E402
from gomoku_instinct.eval import play_match  # noqa: E402
from gomoku_instinct.eval.mcts_player import MctsPlayer  # noqa: E402
from gomoku_instinct.model.loader import load_model  # noqa: E402
from gomoku_instinct.rules import RenjuRules  # noqa: E402
from gomoku_instinct.selfplay import ModelEvaluator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="出零搜索策略的那一方")
    ap.add_argument("--mcts", required=True, help="出 MCTS 的那一方（尺子）")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", default="8,16,32,64")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opening-plies", type=int, default=2)
    args = ap.parse_args()

    raw_model, raw_meta = load_model(args.raw, args.device)
    mcts_model, mcts_meta = load_model(args.mcts, args.device)
    size = raw_meta["board_size"]
    if mcts_meta["board_size"] != size:
        raise SystemExit("两个模型的棋盘尺寸不一致，没法对局")

    raw = InstinctPlayer(raw_model, size, args.device)
    evaluator = ModelEvaluator(mcts_model, size, args.device)
    sims_list = [int(s) for s in str(args.sims).split(",")]

    print(f"零搜索 {os.path.basename(args.raw.rstrip('/'))}@{raw_meta['step']:,}"
          f"  vs  MCTS {os.path.basename(args.mcts.rstrip('/'))}@{mcts_meta['step']:,}"
          f"   每档 {args.games} 局")
    print()
    print("  sims | 零搜索得分率 |  Elo 差 | 平均手数 | 禁手告负(raw/mcts)")

    started = time.perf_counter()
    for sims in sims_list:
        searched = MctsPlayer(
            evaluator, size, sims=sims, slots=args.slots, threads=args.threads
        )
        result = play_match(
            raw,
            searched,
            games=args.games,
            board_size=size,
            rules=RenjuRules(),
            batch=min(args.slots, 64),
            random_opening_plies=args.opening_plies,
            seed=1234 + sims,
        )
        gap = -result.elo_diff
        saturated = result.wins == 0 or result.wins == result.games
        print(
            f"{sims:>6} | {result.score:>11.1%} | {gap:+7.0f}{'*' if saturated else ' '}"
            f"| {result.avg_plies:>8.0f} | "
            f"{result.a_forbidden_losses}/{result.b_forbidden_losses}"
        )

    print(f"耗时 {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
