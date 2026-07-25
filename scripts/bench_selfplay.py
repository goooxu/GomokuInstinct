#!/usr/bin/env python3
"""端到端自博弈吞吐基准。

    python scripts/bench_selfplay.py --games 1024 --threads 36 --seconds 60

这个数字决定训练规模：MCTS 每手要跑几百次网络评估，自博弈产量是全流程的瓶颈。
同时报告 CPU 侧（树搜索）与 GPU 侧（网络前向）各占多少时间，用来判断该往哪边加资源。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.model import InstinctNet, ModelConfig  # noqa: E402
from gomoku_instinct.selfplay import ModelEvaluator, SelfPlayActor  # noqa: E402


class TimedEvaluator:
    """包一层计时，把 GPU 评估耗时单独记出来。"""

    def __init__(self, inner):
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *args):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = self.inner(*args)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--games", type=int, default=1024, help="同时推进的对局数 = 批大小")
    ap.add_argument("--threads", type=int, default=36, help="C++ 侧树搜索线程数")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--fast-sims", type=int, default=100)
    ap.add_argument("--full-search-prob", type=float, default=0.25)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 GPU")
        return 1

    device = torch.device("cuda")
    cfg = ModelConfig(size=args.size, channels=args.channels, blocks=args.blocks)
    net = InstinctNet(cfg).to(device).to(torch.bfloat16).eval()
    print(net.parameter_summary())

    evaluator = ModelEvaluator(
        torch.compile(net) if not args.no_compile else net, args.size, device
    )
    timed = TimedEvaluator(evaluator)

    actor = SelfPlayActor(
        timed,
        board_size=args.size,
        num_games=args.games,
        sims=args.sims,
        fast_sims=args.fast_sims,
        full_search_prob=args.full_search_prob,
        num_threads=args.threads,
        seed=20260725,
    )

    print(f"对局数 {args.games}  搜索线程 {args.threads}  "
          f"sims {args.sims}/{args.fast_sims}  完整搜索比例 {args.full_search_prob}")
    print("预热中……")
    for _ in range(20):
        actor.step()
    actor.reset_stats()
    timed.seconds = 0.0
    timed.calls = 0

    started = time.perf_counter()
    steps = 0
    while time.perf_counter() - started < args.seconds:
        actor.step()
        steps += 1
    elapsed = time.perf_counter() - started

    stats = actor.stats
    evals = steps * args.games

    print()
    print(f"耗时 {elapsed:.1f}s，{steps} 轮")
    print(f"网络评估   {evals / elapsed:>12,.0f} 局面/s")
    print(f"落子       {stats['moves'] / elapsed:>12,.1f} 手/s   <- 主指标")

    # 上千局同时推进，单局要跑很久才结束；测得时间短时「局/s」严重失真，
    # 稳态产量应当由「手/s ÷ 平均每局手数」推算。
    if stats["games"] >= 20:
        avg_plies = stats["moves"] / stats["games"]
        print(f"完成对局   {stats['games'] / elapsed:>12,.2f} 局/s "
              f"（平均 {avg_plies:.0f} 手/局，{stats['games'] / elapsed * 86400:,.0f} 局/天）")
        print(f"训练样本   {stats['samples'] / elapsed:>12,.1f} 条/s")
    else:
        print(f"完成对局   仅 {stats['games']} 局，测量时长不足以给出稳态产量")
        for plies in (30, 45, 60):
            rate = stats["moves"] / elapsed / plies
            print(f"           若平均 {plies} 手/局 -> {rate:.1f} 局/s"
                  f"（{rate * 86400:,.0f} 局/天）")
    print()
    gpu_share = timed.seconds / elapsed
    print(f"GPU 前向占比 {gpu_share:6.1%}   CPU 树搜索占比 {1 - gpu_share:6.1%}")
    if gpu_share > 0.75:
        print("  -> GPU 受限：可以加搜索线程或减小模型")
    elif gpu_share < 0.35:
        print("  -> CPU 受限：可以加搜索线程，或增大同时推进的对局数")
    else:
        print("  -> 两侧较为均衡")
    print()
    print(f"黑胜 {stats['black_wins']}  白胜 {stats['white_wins']}  "
          f"和 {stats['draws']}  禁手告负 {stats['forbidden_losses']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
