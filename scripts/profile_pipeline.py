#!/usr/bin/env python3
"""拆解训练流水线里各步骤的耗时占比。

    python scripts/profile_pipeline.py --run-dir runs/renju15c

分两侧测：

  actor 侧    collect（C++ 树下潜）/ encode+forward（GPU 评估）/ apply（展开回传）
              以及入库时的辅助标签计算与分片落盘
  trainer 侧  采样+对称增强 / 特征编码 / 前向 / 损失 / 反向 / 梯度裁剪+优化器

GPU 上的计时必须先 synchronize，否则测到的只是「把 kernel 排进队列」的时间，
异步执行会让占比完全失真。

**测的时候要把训练停掉**：四张卡都在满载时，任何计时都在测争抢而不是测本身。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.config import load_configs, trainer_config_from  # noqa: E402
from gomoku_instinct.core import load_core, make_rules  # noqa: E402
from gomoku_instinct.model import InstinctNet, ModelConfig, encode  # noqa: E402
from gomoku_instinct.model.loader import load_model  # noqa: E402
from gomoku_instinct.optim import build_optimizer, clip_grad_norm_fp32  # noqa: E402
from gomoku_instinct.rules.constants import EMPTY  # noqa: E402
from gomoku_instinct.selfplay import ModelEvaluator, SelfPlayActor  # noqa: E402
from gomoku_instinct.train import LossWeights, ReplayBuffer, compute_losses  # noqa: E402
from gomoku_instinct.train.replay import compute_labels  # noqa: E402


class Timer:
    def __init__(self, cuda: bool = True) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.cuda = cuda and torch.cuda.is_available()

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    def __call__(self, name: str):
        return _Span(self, name)

    def report(self, title: str, iters: int) -> None:
        total = sum(self.totals.values())
        print(f"\n{title}   单轮 {total / max(iters, 1) * 1e3:.2f} ms")
        print(f"  {'步骤':<26} {'占比':>7} {'单轮耗时':>12}")
        for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            print(
                f"  {name:<26} {seconds / total:>6.1%} "
                f"{seconds / max(iters, 1) * 1e3:>10.3f} ms"
            )


class _Span:
    def __init__(self, timer: Timer, name: str) -> None:
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer.sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.sync()
        self.timer.totals[self.name] += time.perf_counter() - self.start
        self.timer.counts[self.name] += 1
        return False


def profile_actor(model, cfg, device, size, iters: int, threads: int) -> None:
    timer = Timer()
    # 必须和 scripts/actor.py 一样编译。这个模型又小又深（4.44M 参数、20 个 block、
    # 每个 block 一串小算子），瓶颈是 kernel 启动开销与显存往返而不是浮点量，
    # torch.compile 把逐元素算子融掉能快一倍多。拿裸模型测出来的耗时会系统性偏高
    # 一倍以上，而且不报任何错 —— 一个专门用来测性能的工具测的却不是生产配置。
    forward = torch.compile(model) if cfg.compile else model
    evaluator = ModelEvaluator(forward, size, device)
    actor = SelfPlayActor(
        evaluator,
        board_size=size,
        num_games=cfg.num_games,
        sims=cfg.sims,
        fast_sims=cfg.fast_sims,
        full_search_prob=cfg.full_search_prob,
        raw_policy_fraction=cfg.raw_policy_fraction,
        resign_enabled=False,
        num_threads=threads,
        seed=1,
    )

    for _ in range(20):  # 预热，避开 torch.compile 与首次分配
        actor.step()

    for _ in range(iters):
        with timer("collect（C++ 树下潜）"):
            actor.runner.collect(
                actor.boards, actor.to_move, actor.history,
                actor.move_number, actor.needs_eval,
            )
        with timer("evaluate（编码+前向+回传）"):
            policy, value = evaluator(
                actor.boards, actor.to_move, actor.history, actor.move_number
            )
            policy = np.ascontiguousarray(policy, dtype=np.float32)
            value = np.ascontiguousarray(value, dtype=np.float32)
        with timer("apply（展开+回传价值）"):
            actor.runner.apply(policy, value)

    timer.report(f"actor 一轮（{cfg.num_games} 局并行，{threads} 线程）", iters)
    print(f"  注：一手棋平均需要 {cfg.full_search_prob * cfg.sims + (1 - cfg.full_search_prob) * cfg.fast_sims:.0f} 轮")


def profile_labels(rules, size: int, count: int) -> None:
    """入库时的辅助标签计算 —— 这是 CPU 侧的另一大块。"""
    rng = np.random.default_rng(0)
    boards = np.zeros((count, size * size), dtype=np.uint8)
    for i in range(count):
        cells = rng.choice(size * size, size=int(rng.integers(10, 60)), replace=False)
        for j, c in enumerate(cells):
            boards[i, c] = 1 + (j % 2)
    to_move = np.ones(count, dtype=np.uint8)

    for workers in (1, 16):
        started = time.perf_counter()
        compute_labels(boards, to_move, size, rules, workers)
        elapsed = time.perf_counter() - started
        print(
            f"  辅助标签 {workers:>2} 线程：{elapsed / count * 1e3:6.2f} ms/条  "
            f"（{count / elapsed:,.0f} 条/s）"
        )


def fill_synthetic(buffer, size: int, count: int, threat_levels: int) -> None:
    """用随机样本填满回放池，供 trainer 侧剖析。

    只在 replay 目录里还没有分片时用（比如一轮训练刚开始 —— 样本要等对局**结束**
    才产生，头十几分钟一条都没有）。trainer 每一步的耗时只取决于张量形状与 dtype，
    与数据内容无关，所以合成数据测出来的时间是有效的；不有效的是 loss 数值本身，
    而这里根本不看 loss。
    """
    rng = np.random.default_rng(0)
    n = size * size

    boards = np.zeros((count, n), np.uint8)
    for i in range(count):
        cells = rng.choice(n, size=int(rng.integers(10, 60)), replace=False)
        for j, c in enumerate(cells):
            boards[i, c] = 1 + (j % 2)

    policy = rng.random((count, n)).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)

    buffer.add({
        "boards": boards,
        "policy": policy.astype(np.float16),
        "to_move": rng.integers(1, 3, count, dtype=np.uint8),
        "history": rng.integers(-1, n, (count, 4)).astype(np.int32),
        "move_number": rng.integers(0, n, count).astype(np.int32),
        "value": rng.choice([-1.0, 0.0, 1.0], count).astype(np.float32),
        "plies_remaining": rng.integers(0, 60, count).astype(np.int32),
        "next_move": rng.integers(0, n, count).astype(np.int32),
        "root_value": rng.uniform(-1, 1, count).astype(np.float32),
        "blunder_gap": rng.uniform(0, 0.5, count).astype(np.float32),
        "threat_self": rng.integers(0, threat_levels, (count, n)).astype(np.uint8),
        "threat_opp": rng.integers(0, threat_levels, (count, n)).astype(np.uint8),
        "forbidden": (rng.random((count, n)) < 0.02).astype(np.uint8),
    })


def profile_trainer(model, cfg, model_cfg, device, size, buffer, iters: int) -> None:
    timer = Timer()
    optimizer = build_optimizer(model, {"train": {"optim": {"lr": 1e-4}}})
    weights = LossWeights()
    rng = np.random.default_rng(0)
    # 同 profile_actor：生产训练开着 compile，裸模型测出来的一步会慢一倍以上。
    # 优化器和梯度裁剪仍然作用在原始 module 上，编译只包前向。
    forward = torch.compile(model) if cfg.compile else model

    for _ in range(10):  # 预热（顺带触发编译）
        batch = buffer.sample(cfg.batch_size, device, rng)
        planes = encode(batch.boards, batch.to_move, batch.history,
                        batch.move_number, size, dtype=torch.bfloat16)
        loss, _ = compute_losses(forward(planes), batch, weights, 0,
                                 model_cfg.threat_levels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(iters):
        with timer("采样+八重对称增强"):
            batch = buffer.sample(cfg.batch_size, device, rng)
        with timer("特征编码"):
            planes = encode(batch.boards, batch.to_move, batch.history,
                            batch.move_number, size, dtype=torch.bfloat16)
        with timer("前向"):
            out = forward(planes)
        with timer("多头损失"):
            loss, _ = compute_losses(out, batch, weights, 0, model_cfg.threat_levels)
        with timer("反向"):
            loss.backward()
        with timer("梯度裁剪"):
            clip_grad_norm_fp32(list(model.parameters()), 1.0)
        with timer("优化器更新（BF16+Kahan）"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    timer.report(f"trainer 一步（batch {cfg.batch_size}）", iters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--actor-iters", type=int, default=60)
    ap.add_argument("--trainer-iters", type=int, default=40)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--buffer-samples", type=int, default=40000)
    args = ap.parse_args()

    cfgs = load_configs("rules.yaml", "model_base.yaml", "train_4gpu.yaml")
    tcfg = trainer_config_from(cfgs)
    model, meta = load_model(args.run_dir, args.device)
    size = meta["board_size"]
    model_cfg = ModelConfig(**{k: v for k, v in vars(model.cfg).items()})
    device = torch.device(args.device)

    print(f"权重 step {meta['step']:,}   棋盘 {size}x{size}   设备 {args.device}   "
          f"torch.compile {'开' if tcfg.compile else '关'}")
    print(model.parameter_summary())

    profile_actor(model, tcfg, device, size, args.actor_iters, args.threads)

    print("\n入库侧（CPU）")
    profile_labels(make_rules(cfgs), size, 400)

    print("\n装载 replay 分片以剖析 trainer……")
    buffer = ReplayBuffer(
        args.buffer_samples, size,
        shard_dir=os.path.join(args.run_dir, "replay"),
        blunder_threshold=tcfg.blunder_threshold,
        blunder_fraction=tcfg.blunder_fraction,
    )
    loaded = buffer.restore_from_shards()
    print(f"  装入 {loaded:,} 条")
    if loaded == 0:
        # 一轮训练刚开始时这里必然是 0：样本要等对局**结束**才产生，
        # 2048 局并行、每局几十手，头十几分钟一条都没有。
        # 退回合成样本，别让工具在最需要它的时候什么都不给。
        print(f"  没有分片，改用 {args.buffer_samples:,} 条合成样本"
              "（步骤耗时只取决于张量形状，与数据内容无关）")
        # 关键：先摘掉 shard_dir，否则合成数据会被当成真样本写进 replay 目录，
        # 污染正在跑的那轮训练。
        buffer.shard_dir = None
        fill_synthetic(buffer, size, args.buffer_samples, model_cfg.threat_levels)

    train_model = InstinctNet(model_cfg).to(device).to(torch.bfloat16)
    train_model.load_state_dict(model.state_dict())
    train_model.train()
    profile_trainer(train_model, tcfg, model_cfg, device, size, buffer,
                    args.trainer_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
