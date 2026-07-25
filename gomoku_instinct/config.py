"""配置加载：把 configs/ 下的 YAML 合并成各组件的配置对象。"""

from __future__ import annotations

import os
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def load_yaml(path: str) -> dict:
    if not os.path.isabs(path):
        candidate = os.path.join(CONFIG_DIR, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_configs(*paths: str) -> dict:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    return merged


def trainer_config_from(cfg: dict, **overrides):
    """把 YAML 里的训练配置摊平成 TrainerConfig。"""
    from .train.trainer import TrainerConfig

    rules = cfg.get("rules", {})
    selfplay = cfg.get("selfplay", {})
    mcts = selfplay.get("mcts", {})
    temperature = selfplay.get("temperature", {})
    resign = selfplay.get("resign", {})
    replay = cfg.get("replay", {})
    train = cfg.get("train", {})
    optim = train.get("optim", {})
    schedule = train.get("lr_schedule", {})
    ckpt = cfg.get("checkpoint", {})
    logging_cfg = cfg.get("logging", {})

    values = dict(
        board_size=rules.get("board_size", 15),
        seed=cfg.get("run", {}).get("seed", 20260725),
        num_games=selfplay.get("games_in_flight", 1024),
        sims=mcts.get("sims", 400),
        fast_sims=mcts.get("fast_sims", 100),
        full_search_prob=mcts.get("full_search_prob", 0.25),
        dirichlet_alpha=mcts.get("dirichlet_alpha", 0.15),
        dirichlet_eps=mcts.get("dirichlet_eps", 0.25),
        temperature=temperature.get("initial", 1.0),
        temperature_moves=temperature.get("moves", 16),
        raw_policy_fraction=selfplay.get("deployment_distribution_fraction", 0.25),
        resign_enabled=resign.get("enabled", True),
        resign_threshold=resign.get("threshold", -0.92),
        resign_audit_fraction=resign.get("audit_fraction", 0.05),
        capacity=replay.get("capacity_positions", 4_000_000),
        min_positions_to_start=replay.get("min_positions_to_start", 50_000),
        shard_size=replay.get("shard_positions", 65_536),
        keep_shards=replay.get("disk_keep_shards", 400),
        batch_size=train.get("batch_size", 1024),
        max_steps=train.get("max_steps", 2_000_000),
        compile=train.get("compile", True),
        target_sample_reuse=train.get("target_sample_reuse", 4.0),
        max_train_steps_per_cycle=train.get("max_train_steps_per_cycle", 400),
        selfplay_steps_per_cycle=train.get("selfplay_steps_per_cycle", 200),
        external_selfplay=train.get("external_selfplay", False),
        lr=optim.get("lr", 2e-3),
        grad_clip=optim.get("grad_clip", 1.0),
        warmup_steps=schedule.get("warmup_steps", 2_000),
        min_lr_scale=schedule.get("min_lr_scale", 0.05),
        checkpoint_every_seconds=ckpt.get("every_seconds", 600.0),
        checkpoint_every_steps=ckpt.get("every_steps", 10_000),
        keep_last=ckpt.get("keep_last", 10),
        log_every_steps=logging_cfg.get("log_every_steps", 50),
    )
    values.update(overrides)
    return TrainerConfig(**values)
