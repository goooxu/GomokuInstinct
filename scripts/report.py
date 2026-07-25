#!/usr/bin/env python3
"""查看一次训练的进展。

    python scripts/report.py --run-dir runs/renju15
    python scripts/report.py --run-dir runs/renju15 --arena 200

--arena 会额外跑一轮竞技场，给出对随机与规则基线的绝对棋力标尺。
所有对局一律零搜索模式，与部署条件一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMNS = [
    ("step", "step", "{:>8.0f}"),
    ("loss/total", "总损失", "{:>8.3f}"),
    ("loss/policy", "策略", "{:>7.3f}"),
    ("loss/value", "价值", "{:>7.3f}"),
    ("policy/top1_agreement", "策略top1", "{:>8.1%}"),
    ("value/accuracy", "价值准确", "{:>8.1%}"),
    ("threat/accuracy_on_threats", "威胁识别", "{:>8.1%}"),
    ("forbidden/recall", "禁手召回", "{:>8.1%}"),
    ("buffer/size", "样本窗口", "{:>9.0f}"),
    ("buffer/reuse", "复用率", "{:>7.1f}"),
]


def show_metrics(run_dir: str, rows_to_show: int = 12) -> None:
    path = os.path.join(run_dir, "logs", "metrics.jsonl")
    if not os.path.exists(path):
        print("还没有指标记录（训练可能刚起步）")
        return
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        print("指标文件为空")
        return

    print(f"共 {len(rows)} 条记录，step {rows[0]['step']:.0f} -> {rows[-1]['step']:.0f}")
    print()
    print(" ".join(name.rjust(8) for _, name, _ in COLUMNS))
    stride = max(1, len(rows) // rows_to_show)
    for row in rows[::stride] + ([rows[-1]] if len(rows) > 1 else []):
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else " " * 8)
        print(" ".join(cells))


def show_selfplay(run_dir: str) -> None:
    """从 actor 日志里挑出最后一行统计。"""
    logs_dir = os.path.join(run_dir, "logs")
    if not os.path.isdir(logs_dir):
        return
    actor_logs = sorted(f for f in os.listdir(logs_dir) if f.startswith("actor"))
    if not actor_logs:
        return
    print()
    print("自博弈：")
    for name in actor_logs:
        with open(os.path.join(logs_dir, name)) as fh:
            lines = [line.strip() for line in fh if "手/局" in line]
        if lines:
            print("  " + lines[-1])


def run_arena(run_dir: str, games: int, device: str) -> None:
    from gomoku_instinct.core import load_core
    from gomoku_instinct.cli.engine import InstinctPlayer
    from gomoku_instinct.eval import GreedyThreatPlayer, RandomPlayer, play_match
    from gomoku_instinct.model.loader import load_model
    from gomoku_instinct.rules import RenjuRules

    model, meta = load_model(run_dir, device)
    size = meta["board_size"]
    player = InstinctPlayer(model, size, device)
    core = load_core()
    rules = RenjuRules()

    print()
    print(f"竞技场（零搜索模式，{games} 局，step {meta['step']:,}）")
    for name, opponent in (
        ("random", RandomPlayer(seed=1)),
        ("greedy_threat", GreedyThreatPlayer(core.Rules(), size, seed=1)),
    ):
        result = play_match(
            player, opponent, games=games, board_size=size, rules=rules, batch=64
        )
        print("  " + result.summary(f"model@{meta['step']}", name).replace("\n", "\n  "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arena", type=int, default=0, help="额外跑多少局竞技场")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    show_metrics(args.run_dir)
    show_selfplay(args.run_dir)
    if args.arena > 0:
        run_arena(args.run_dir, args.arena, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
