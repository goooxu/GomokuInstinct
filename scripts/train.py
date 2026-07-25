#!/usr/bin/env python3
"""训练入口。默认自动续训。

    # 从头开始（或接着上次跑）
    python scripts/train.py --run-dir runs/renju15

    # 9x9 快速验证全链路
    python scripts/train.py --run-dir runs/dev9 --board-size 9 \
        --config rules.yaml --config model_base.yaml --config train_4gpu.yaml \
        --override num_games=256 sims=64 fast_sims=16 batch_size=256 \
                   capacity=200000 min_positions_to_start=2000

启动时会自动检测 run-dir 下的最新 checkpoint 并原地续训；要从头来过加 --fresh。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.config import load_configs, trainer_config_from  # noqa: E402
from gomoku_instinct.model import ModelConfig  # noqa: E402
from gomoku_instinct.train import LossWeights, Trainer  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override 需要 key=value，收到 {item!r}")
        key, raw = item.split("=", 1)
        if raw.lower() in ("true", "false"):
            value: object = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help="可多次指定，按顺序合并；默认 rules/model_base/train_4gpu",
    )
    ap.add_argument("--board-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fresh", action="store_true", help="清空 run-dir 从头开始")
    ap.add_argument("--max-seconds", type=float, default=None, help="跑多久后自行退出")
    ap.add_argument("--override", nargs="*", default=[], help="覆盖 TrainerConfig 字段")
    args = ap.parse_args()

    configs = args.config or ["rules.yaml", "model_base.yaml", "train_4gpu.yaml"]
    cfg = load_configs(*configs)

    if args.board_size is not None:
        cfg.setdefault("rules", {})["board_size"] = args.board_size
        cfg.setdefault("model", {})["board_size"] = args.board_size

    if args.fresh and os.path.isdir(args.run_dir):
        print(f"--fresh：清空 {args.run_dir}")
        shutil.rmtree(args.run_dir)
    os.makedirs(args.run_dir, exist_ok=True)

    overrides = parse_overrides(args.override)
    trainer_cfg = trainer_config_from(cfg, **overrides)
    model_cfg = ModelConfig.from_dict(cfg)
    model_cfg.size = trainer_cfg.board_size
    loss_weights = LossWeights.from_dict(cfg)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("没有可用 GPU，退回 CPU")
        device = "cpu"
        trainer_cfg.compile = False

    trainer = Trainer(trainer_cfg, model_cfg, loss_weights, args.run_dir, device, cfg)
    print(trainer.model.parameter_summary())
    print(f"棋盘 {trainer_cfg.board_size}x{trainer_cfg.board_size}  设备 {device}")

    if trainer.resume_or_start():
        print(f"已从 checkpoint 续训：step={trainer.step}  "
              f"buffer={len(trainer.buffer):,}")
    else:
        print("未找到 checkpoint，从头开始")

    started = time.time()
    try:
        trainer.run(max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        print("\n收到中断，保存 checkpoint 后退出")
        trainer.save_checkpoint()

    print(f"结束：step={trainer.step}  用时 {time.time() - started:.0f}s")
    print(f"最新 checkpoint: {trainer.latest_checkpoint()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
