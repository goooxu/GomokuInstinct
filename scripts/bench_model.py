#!/usr/bin/env python3
"""InstinctNet 的 GPU 吞吐基准。

    python scripts/bench_model.py --batch 512,2048,4096

推理吞吐直接决定自博弈的产量（MCTS 每手要跑几百次评估），
所以这个数字是后续所有规模决策的依据。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.model import NUM_PLANES, InstinctNet, ModelConfig  # noqa: E402
from gomoku_instinct.optim import BF16AdamW, clip_grad_norm_fp32  # noqa: E402


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def profile_components(cfg, batch, device, iters=20, warmup=5) -> None:
    """逐个模块计时，定位瓶颈在哪。"""
    from gomoku_instinct.model.blocks import (
        DirectionalLineBlock,
        GlobalAttention,
        LineConv,
        RMSNorm2d,
        SqueezeExcite,
    )

    c, size = cfg.channels, cfg.size
    x = torch.randn(batch, c, size, size, device=device, dtype=torch.bfloat16)

    parts = {
        "RMSNorm2d": RMSNorm2d(c),
        "LineConv": LineConv(c, size, cfg.line_kernel),
        "Conv3x3": torch.nn.Conv2d(c, c, 3, padding=1, bias=False),
        "Conv1x1(2C->C)": torch.nn.Conv2d(2 * c, c, 1, bias=False),
        "SqueezeExcite": SqueezeExcite(c, cfg.se_ratio),
        "GlobalAttention": GlobalAttention(c, cfg.attn_heads),
        "整块 DirectionalLineBlock": DirectionalLineBlock(c, size, cfg.line_kernel),
    }

    print(f"逐模块耗时（batch {batch}，单次前向）")
    with torch.inference_mode():
        for name, mod in parts.items():
            mod = mod.to(device).to(torch.bfloat16)
            inp = torch.randn(
                batch, 2 * c if "2C" in name else c, size, size,
                device=device, dtype=torch.bfloat16,
            )
            ms = _time(lambda: mod(inp), iters, warmup) * 1e3
            print(f"  {name:28s} {ms:8.2f} ms")
    print()


def bench_inference(net, batch, size, device, iters=30, warmup=10) -> float:
    x = torch.randn(batch, NUM_PLANES, size, size, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(warmup):
            net(x, with_aux=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            net(x, with_aux=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return batch * iters / elapsed


def bench_train_step(net, batch, size, device, iters=20, warmup=5) -> float:
    opt = BF16AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    x = torch.randn(batch, NUM_PLANES, size, size, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, size * size, (batch,), device=device)

    def one_step():
        out = net(x)
        loss = torch.nn.functional.cross_entropy(out.policy.float(), target)
        loss = loss + out.value.float().square().mean()
        loss = loss + out.threat.float().square().mean()
        loss = loss + out.forbidden.float().square().mean()
        loss.backward()
        clip_grad_norm_fp32(list(net.parameters()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return batch * iters / elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="512,2048,4096")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--train-batch", type=int, default=1024)
    ap.add_argument("--compile", action="store_true", help="额外测一遍 torch.compile")
    ap.add_argument("--channels-last", action="store_true", help="用 channels_last 内存布局")
    ap.add_argument("--profile", action="store_true", help="逐模块计时")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 GPU")
        return 1
    device = torch.device("cuda")

    cfg = ModelConfig(size=args.size, channels=args.channels, blocks=args.blocks)
    net = InstinctNet(cfg).to(device).to(torch.bfloat16)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
    print(net.parameter_summary())
    print(f"设备 {torch.cuda.get_device_name(0)}")
    print()

    if args.profile:
        profile_components(cfg, max(int(b) for b in args.batch.split(",")), device)

    net.eval()
    print("推理吞吐（零搜索对战与 MCTS 叶子评估都走这条路径）")
    for batch in [int(b) for b in args.batch.split(",")]:
        rate = bench_inference(net, batch, args.size, device)
        # MCTS 每手 400 次评估、每局约 40 手 -> 每局约 16000 次评估
        games = rate / 16000
        print(f"  batch {batch:5d}:  {rate:10,.0f} 局面/s   ≈ {games:6.1f} 局/s（400 sims/手）")

    if args.compile:
        print()
        print("torch.compile 之后")
        compiled = torch.compile(net)
        for batch in [int(b) for b in args.batch.split(",")]:
            rate = bench_inference(compiled, batch, args.size, device)
            print(f"  batch {batch:5d}:  {rate:10,.0f} 局面/s")

    print()
    net.train()
    rate = bench_train_step(net, args.train_batch, args.size, device)
    print(f"训练吞吐  batch {args.train_batch}:  {rate:,.0f} 局面/s "
          f"（{rate / args.train_batch:.1f} step/s）")

    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    print(f"显存峰值  {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
