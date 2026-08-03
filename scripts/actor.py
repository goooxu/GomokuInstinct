#!/usr/bin/env python3
"""独立的自博弈 actor 进程：独占一张 GPU 产出训练样本。

    python scripts/actor.py --run-dir runs/renju15 --actor-id 0

它做三件事，循环往复：

  1. 盯着 checkpoints/latest，一有新权重就热加载（不需要与 trainer 握手）
  2. 跑向量化自博弈
  3. 把样本连同规则导出的辅助标签写成分片落盘，trainer 自己会扫到

分片是原子 rename 出来的，所以 trainer 扫到的一定是完整文件；
actor 与 trainer 之间除了文件系统没有任何耦合，任一侧崩了另一侧照常跑，
重启后各自从最新状态接上 —— 这正是换机续训需要的性质。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gomoku_instinct.config import load_configs, trainer_config_from  # noqa: E402
from gomoku_instinct.core import load_core, make_rules  # noqa: E402
from gomoku_instinct.model import InstinctNet, ModelConfig  # noqa: E402
from gomoku_instinct.model.loader import load_model  # noqa: E402
from gomoku_instinct.selfplay import ModelEvaluator, SelfPlayActor  # noqa: E402
from gomoku_instinct.train.replay import ReplayBuffer  # noqa: E402


def latest_checkpoint(run_dir: str) -> str | None:
    pointer = os.path.join(run_dir, "checkpoints", "latest")
    if not os.path.exists(pointer):
        return None
    with open(pointer) as fh:
        name = fh.read().strip()
    path = os.path.join(run_dir, "checkpoints", name)
    return path if os.path.exists(path) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--actor-id", type=int, default=0)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--board-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=8192)
    ap.add_argument("--reload-every-seconds", type=float, default=120.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--label-workers", type=int, default=12)
    ap.add_argument(
        "--raw-policy-fraction",
        type=float,
        default=None,
        help="部署分布自博弈比例；不指定则用配置里的值",
    )
    args = ap.parse_args()

    configs = args.config or ["rules.yaml", "model_base.yaml", "train_4gpu.yaml"]
    cfg = load_configs(*configs)
    if args.board_size is not None:
        cfg.setdefault("rules", {})["board_size"] = args.board_size
        cfg.setdefault("model", {})["board_size"] = args.board_size

    overrides = {}
    if args.games is not None:
        overrides["num_games"] = args.games
    if args.raw_policy_fraction is not None:
        overrides["raw_policy_fraction"] = args.raw_policy_fraction
    tcfg = trainer_config_from(cfg, **overrides)

    device = torch.device(args.device)
    size = tcfg.board_size
    prefix = f"actor{args.actor_id}"

    # 等 trainer 写出第一个 checkpoint。actor 不自己造初始权重，
    # 免得不同进程从不同随机初始化出发、产出的样本互相打架。
    print(f"\n===== {prefix} 启动 pid={os.getpid()} =====", flush=True)
    print(f"[{prefix}] 等待第一个 checkpoint……", flush=True)
    while latest_checkpoint(args.run_dir) is None:
        if args.max_seconds is not None:
            args.max_seconds -= 5
            if args.max_seconds <= 0:
                print(f"[{prefix}] 超时退出")
                return 1
        time.sleep(5)

    path = latest_checkpoint(args.run_dir)
    model, meta = load_model(path, device)
    print(f"[{prefix}] 载入 {os.path.basename(path)}（step {meta['step']:,}）", flush=True)

    forward = torch.compile(model) if tcfg.compile else model
    actor = SelfPlayActor(
        ModelEvaluator(forward, size, device),
        board_size=size,
        num_games=tcfg.num_games,
        sims=tcfg.sims,
        fast_sims=tcfg.fast_sims,
        full_search_prob=tcfg.full_search_prob,
        dirichlet_alpha=tcfg.dirichlet_alpha,
        dirichlet_eps=tcfg.dirichlet_eps,
        temperature=tcfg.temperature,
        temperature_moves=tcfg.temperature_moves,
        raw_policy_fraction=tcfg.raw_policy_fraction,
        resign_enabled=tcfg.resign_enabled,
        resign_threshold=tcfg.resign_threshold,
        resign_audit_fraction=tcfg.resign_audit_fraction,
        num_threads=args.threads,
        # 每个 actor 一套独立的随机流，否则几张卡会跑出一模一样的对局
        seed=tcfg.seed + 1_000_003 * (args.actor_id + 1),
    )

    rules = make_rules(cfg)
    sink = ReplayBuffer(
        capacity=max(args.shard_size * 2, 4096),
        board_size=size,
        shard_dir=os.path.join(args.run_dir, "replay"),
        shard_size=args.shard_size,
        keep_shards=0,  # 回收交给 trainer，actor 只管写
        label_workers=args.label_workers,
        shard_prefix=prefix,
    )
    # 接着磁盘上已有的分片编号往下写。不这样做的话重启后会从 0 重来，
    # 与上一次运行的分片同名——既覆盖旧数据，又会被 trainer 当成"已见过"整批跳过。
    first_index = sink.resume_shard_index()
    print(f"[{prefix}] 分片从 #{first_index} 开始写", flush=True)

    started = time.time()
    last_reload = time.time()
    last_report = time.time()
    loaded_path = path

    done_flag = os.path.join(args.run_dir, "DONE")

    while True:
        if args.max_seconds is not None and time.time() - started > args.max_seconds:
            break
        # trainer 跑满后会落这个文件。actor 是独立进程，否则无从知道训练已经结束，
        # 会继续满负荷空烧 GPU（实测发生过两次，每次白烧三张卡）。
        if os.path.exists(done_flag):
            print(f"[{prefix}] 检测到 {done_flag}，训练已结束，收摊退出", flush=True)
            break

        actor.step()

        if actor.pending_samples >= args.shard_size // 2:
            sink.add_from_drain(actor.drain(), rules)

        now = time.time()
        if now - last_reload >= args.reload_every_seconds:
            last_reload = now
            newest = latest_checkpoint(args.run_dir)
            if newest and newest != loaded_path:
                try:
                    fresh, meta = load_model(newest, device)
                    model.load_state_dict(fresh.state_dict())
                    loaded_path = newest
                    print(f"[{prefix}] 热加载 step {meta['step']:,}", flush=True)
                except (OSError, RuntimeError) as exc:
                    print(f"[{prefix}] 权重热加载失败（下轮重试）：{exc}", flush=True)

        if now - last_report >= 60.0:
            last_report = now
            stats = actor.stats
            games = stats["games"]
            # 上千局同时推进，起步阶段一局都没走完是正常的；
            # 这时不要拿 max(1, games) 去除，那会打印出「9400 手/局」这种假数字
            if games == 0:
                rates = "（尚无完成的对局）"
            else:
                # 黑白和三项都打出来：连珠规则不对称（黑必须恰好五连、还有禁手约束），
                # 只看黑胜率分不清「黑方在输」和「大量和棋」。
                rates = (
                    f"{stats['completed_plies'] / games:.0f} 手/局  "
                    f"黑{stats['black_wins']}/白{stats['white_wins']}/和{stats['draws']}  "
                    f"禁手告负 {stats['forbidden_losses'] / games:.2%}  "
                    # 认输误判率必须盯着：价值头没训起来之前认输就是在瞎判，
                    # 会让训练数据里的价值目标整体走样
                    f"认输 {stats['resigns'] / games:.1%}"
                    f"(误判 {stats['resign_false_positives'] / max(1, stats['resign_audits']):.1%})"
                )
            print(
                f"[{prefix}] 局 {games:,}  手 {stats['moves']:,}  "
                f"样本 {stats['samples']:,}  {rates}",
                flush=True,
            )

    sink.add_from_drain(actor.drain(), rules)
    sink.flush()
    print(f"[{prefix}] 退出，累计产出 {actor.stats['samples']:,} 条样本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
