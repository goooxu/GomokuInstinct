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
    ap.add_argument(
        "--sims",
        default="400",
        help="逗号分隔的多个模拟数，例如 25,50,100,200。"
        "对手强到零搜索一局不胜时 Elo 估计会被钳在上限、无法反映真实差距，"
        "所以要扫一遍找出打平的档位——那个档位就是'零搜索策略值多少次搜索'。",
    )
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opening-plies", type=int, default=2,
                help="每局开头随机落几子。两个确定性 player 对弈时必须大于 0，"
                     "否则每局都是同一盘棋的重放、胜负只由先后手决定")
    ap.add_argument("--append-to", default=None, help="把结果追加到这个 jsonl")
    args = ap.parse_args()

    model, meta = load_model(args.run_dir, args.device)
    size = meta["board_size"]

    raw = InstinctPlayer(model, size, args.device)
    evaluator = ModelEvaluator(model, size, args.device)
    sims_list = [int(s) for s in str(args.sims).split(",")]

    print(f"零搜索策略 vs 同权重 MCTS   step {meta['step']:,}   每档 {args.games} 局")
    print()
    print("  sims | 零搜索得分率 |  Elo 差 | 平均手数 | 禁手告负(raw/mcts)")

    records = []
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
        # 取负号：raw 打不过 mcts 时 elo_diff 为负，gap 取正表示"落后多少"
        gap = -result.elo_diff
        saturated = result.wins == 0 or result.wins == result.games
        print(
            f"{sims:>6} | {result.score:>11.1%} | {gap:+7.0f}{'*' if saturated else ' '}"
            f"| {result.avg_plies:>8.0f} | {result.a_forbidden_losses}/{result.b_forbidden_losses}"
        )
        records.append(
            {
                "step": meta["step"],
                "sims": sims,
                "games": result.games,
                "raw_score": result.score,
                "search_gap_elo": gap,
                "saturated": saturated,
                "raw_forbidden_losses": result.a_forbidden_losses,
                "mcts_forbidden_losses": result.b_forbidden_losses,
                "avg_plies": result.avg_plies,
                "opening_plies": args.opening_plies,
            }
        )

    print()
    if any(r["saturated"] for r in records):
        print("* 该档一局未胜（或全胜），Elo 估计被钳在测量上限，只说明差距大于这个数")

    # 零搜索策略"值多少次搜索"：得分率跨过 50% 的档位
    crossover = None
    for lo, hi in zip(records, records[1:]):
        if lo["raw_score"] >= 0.5 > hi["raw_score"]:
            crossover = (lo["sims"], hi["sims"])
            break
    if crossover:
        print(f"零搜索策略约相当于 {crossover[0]}~{crossover[1]} 次搜索")
    elif records[0]["raw_score"] < 0.5:
        print(f"零搜索策略弱于最低档（{records[0]['sims']} 次搜索）——需要再往下扫")
    else:
        print(f"零搜索策略强于最高档（{records[-1]['sims']} 次搜索）——需要再往上扫")
    print(f"耗时 {time.perf_counter() - started:.0f}s")

    if args.append_to:
        with open(args.append_to, "a") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"已记录到 {args.append_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
