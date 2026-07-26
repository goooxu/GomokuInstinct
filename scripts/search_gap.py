#!/usr/bin/env python3
"""测量本项目的头号指标：零搜索策略 vs 同权重 MCTS 的 Elo 差。

    python scripts/search_gap.py --run-dir runs/renju15 --games 200 --sims 800

整个方案赌的是「能把搜索压进权重里」。AlphaZero 类方法的棋力一半来自网络、
一半来自树搜索；把搜索从推理端拿掉后，原始策略通常远弱于它自己的 MCTS 版本。
这个差值就是「还没被压进权重的那部分棋力」，是唯一能直接量化这件事的数字。

两边用的是同一份权重、同一套输入编码、同样的确定性选点，差异完全归于搜索本身。
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--append-to", default=None, help="把结果追加到这个 jsonl")
    args = ap.parse_args()

    model, meta = load_model(args.run_dir, args.device)
    size = meta["board_size"]

    raw = InstinctPlayer(model, size, args.device)
    evaluator = ModelEvaluator(model, size, args.device)
    searched = MctsPlayer(
        evaluator, size, sims=args.sims, slots=args.slots, threads=args.threads
    )

    print(f"零搜索 vs MCTS({args.sims} sims)   step {meta['step']:,}   {args.games} 局")
    started = time.perf_counter()
    result = play_match(
        raw,
        searched,
        games=args.games,
        board_size=size,
        rules=RenjuRules(),
        batch=min(args.slots, 64),
    )
    elapsed = time.perf_counter() - started

    print()
    print(result.summary("raw_policy", f"mcts{args.sims}"))
    print()
    # 差值取负号：raw 打不过 mcts 时 elo_diff 为负，gap 取正数表示"落后多少"
    gap = -result.elo_diff
    print(f"搜索差距 {gap:+.0f} Elo   （越小越好；0 表示搜索已被完全压进权重）")
    print(f"耗时 {elapsed:.0f}s")

    if args.append_to:
        record = {
            "step": meta["step"],
            "sims": args.sims,
            "games": result.games,
            "raw_score": result.score,
            "search_gap_elo": gap,
            "raw_forbidden_losses": result.a_forbidden_losses,
            "mcts_forbidden_losses": result.b_forbidden_losses,
            "avg_plies": result.avg_plies,
        }
        with open(args.append_to, "a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"已记录到 {args.append_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
