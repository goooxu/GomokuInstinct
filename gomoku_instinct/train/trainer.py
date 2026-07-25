"""训练主循环，以及换机续训所需的全套状态保存。

开发机有使用时长限制、随时可能换机，因此这里的设计前提是**任何时刻被 kill 都能原地续训**：

* checkpoint 按**墙钟时间**触发（不只按 step），把抢占损失控制在有界范围内；
* 先写临时文件再 rename，`latest` 指针原子更新，绝不会读到半截 checkpoint；
* 保存的不只是模型权重，还有优化器状态（含 Kahan 补偿缓冲）、学习率进度、
  Python/NumPy/Torch/自博弈引擎四套随机数状态、以及 replay buffer 的分片游标；
* replay buffer 以整局为单位分片落盘，续训时按时间倒序装回最近的分片。

自博弈与训练在同一进程里交替进行。这个安排在单卡上就能跑通全链路，
多卡编排（actor 独立进程）在此之上扩展。
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..core import load_core, make_rules
from ..model import InstinctNet, ModelConfig, encode
from ..optim import BF16AdamW, build_optimizer, clip_grad_norm_fp32
from ..selfplay import ModelEvaluator, SelfPlayActor
from .losses import LossWeights, compute_losses
from .replay import ReplayBuffer

CHECKPOINT_VERSION = 1


@dataclass
class TrainerConfig:
    board_size: int = 15
    seed: int = 20260725

    # 自博弈
    num_games: int = 1024
    selfplay_threads: int = 36
    sims: int = 400
    fast_sims: int = 100
    full_search_prob: float = 0.25
    dirichlet_alpha: float = 0.15
    dirichlet_eps: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 16
    raw_policy_fraction: float = 0.25
    resign_enabled: bool = True
    resign_threshold: float = -0.92
    resign_audit_fraction: float = 0.05

    # replay
    capacity: int = 4_000_000
    min_positions_to_start: int = 50_000
    shard_size: int = 65_536
    keep_shards: int = 400
    label_workers: int = 16

    # 训练
    batch_size: int = 1024
    max_steps: int = 2_000_000
    lr: float = 2e-3
    warmup_steps: int = 2_000
    min_lr_scale: float = 0.05
    grad_clip: float = 1.0
    compile: bool = True

    # 一个周期内的节奏。
    #
    # 训练步数**不是**固定值，而是按目标样本复用率自适应算出来的：
    # 自博弈比训练贵得多（一手棋要几百次网络评估，一个训练步只吃一个 batch），
    # 固定配比极易失衡 —— 实测按固定配比跑，复用率会飙到 20 以上，
    # 也就是每条样本被反复训练二十几遍，直接进入过拟合区间。
    selfplay_steps_per_cycle: int = 200
    target_sample_reuse: float = 4.0
    max_train_steps_per_cycle: int = 400

    # 多卡编排：自博弈交给独立的 actor 进程（各占一张 GPU），
    # trainer 只扫描它们写出的分片。单卡跑通全链路时保持 False。
    external_selfplay: bool = False
    idle_sleep_seconds: float = 5.0
    # 连续这么久一条样本都没吃到就告警：这种失败是静默的，训练步数会一直停在 0
    idle_warn_seconds: float = 300.0

    # 容灾与日志
    checkpoint_every_seconds: float = 600.0
    checkpoint_every_steps: int = 10_000
    keep_last: int = 10
    log_every_steps: int = 50


def lr_at(step: int, cfg: TrainerConfig) -> float:
    """warmup 之后走 cosine 衰减。"""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine)


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model_cfg: ModelConfig,
        loss_weights: LossWeights,
        run_dir: str,
        device: torch.device | str = "cuda",
        rules_cfg: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.loss_weights = loss_weights
        self.run_dir = run_dir
        self.device = torch.device(device)

        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "replay"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        self.core = load_core()
        self.rules = make_rules(rules_cfg)

        # 必须在建模型**之前**播种：初始权重是随机初始化的产物，
        # 播种放在后面的话每次启动的初始权重都不一样，整个 run 无法复现。
        # 而且未训练网络的系统性偏向本身就随初始化而变（黑白胜率能差出几倍），
        # 早期自博弈统计量因此完全不可比。
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        # 主权重就是 BF16
        self.model = InstinctNet(model_cfg).to(self.device).to(torch.bfloat16)
        self.optimizer = build_optimizer(
            self.model, {"train": {"optim": {"lr": cfg.lr}}}
        )
        self.forward = (
            torch.compile(self.model) if cfg.compile else self.model
        )

        self.buffer = ReplayBuffer(
            cfg.capacity,
            cfg.board_size,
            shard_dir=os.path.join(run_dir, "replay"),
            shard_size=cfg.shard_size,
            keep_shards=cfg.keep_shards,
            label_workers=cfg.label_workers,
        )

        self.actor = None if cfg.external_selfplay else SelfPlayActor(
            ModelEvaluator(self.forward, cfg.board_size, self.device),
            board_size=cfg.board_size,
            num_games=cfg.num_games,
            sims=cfg.sims,
            fast_sims=cfg.fast_sims,
            full_search_prob=cfg.full_search_prob,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
            temperature=cfg.temperature,
            temperature_moves=cfg.temperature_moves,
            raw_policy_fraction=cfg.raw_policy_fraction,
            resign_enabled=cfg.resign_enabled,
            resign_threshold=cfg.resign_threshold,
            resign_audit_fraction=cfg.resign_audit_fraction,
            num_threads=cfg.selfplay_threads,
            seed=cfg.seed,
        )

        self.step = 0
        self.cycle = 0
        self.samples_seen = 0
        self.rng = np.random.default_rng(cfg.seed)
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = 0

    # ── 训练 ────────────────────────────────────────────────────────────────
    def train_step(self) -> dict[str, float]:
        lr = lr_at(self.step, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        batch = self.buffer.sample(self.cfg.batch_size, self.device, self.rng)
        planes = encode(
            batch.boards,
            batch.to_move,
            batch.history,
            batch.move_number,
            self.cfg.board_size,
            dtype=torch.bfloat16,
        )
        out = self.forward(planes)
        loss, metrics = compute_losses(
            out, batch, self.loss_weights, self.step, self.model_cfg.threat_levels
        )

        loss.backward()
        grad_norm = clip_grad_norm_fp32(
            list(self.model.parameters()), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1
        self.samples_seen += self.cfg.batch_size
        metrics["train/lr"] = lr
        metrics["train/grad_norm"] = float(grad_norm)
        return metrics

    def selfplay_cycle(self) -> int:
        if self.actor is None:
            # 多卡模式：自博弈在独立进程里跑，这里只管把新分片吃进来
            return self.buffer.ingest_new_shards()
        self.model.eval()
        self.actor.run(self.cfg.selfplay_steps_per_cycle)
        added = self.buffer.add_from_drain(self.actor.drain(), self.rules)
        self.model.train()
        return added

    def train_steps_for_cycle(self) -> int:
        """按目标样本复用率决定这一周期训练多少步。

        复用率 = 已训练样本数 ÷ 历史产出样本数。太高会过拟合、
        并且是在拿旧数据反复磨；太低则浪费了自博弈辛苦产出的样本。
        这里让它自己追着目标值走，不必手工去配「自博弈跑几轮、训练跑几步」。
        """
        allowed = self.buffer.total_added * self.cfg.target_sample_reuse
        deficit = allowed - self.samples_seen
        if deficit <= 0:
            return 0
        steps = int(deficit // self.cfg.batch_size)
        return max(0, min(steps, self.cfg.max_train_steps_per_cycle))

    def run(self, max_seconds: float | None = None) -> None:
        started = time.time()
        log_path = os.path.join(self.run_dir, "logs", "metrics.jsonl")

        # 多卡模式下 actor 要等第一个 checkpoint 才能起步，所以先落一个。
        if self.cfg.external_selfplay and self.latest_checkpoint() is None:
            self.save_checkpoint()

        last_ingest = time.time()
        warned_idle = False

        while self.step < self.cfg.max_steps:
            if max_seconds is not None and time.time() - started > max_seconds:
                break

            added = self.selfplay_cycle()
            self.cycle += 1

            if added > 0:
                last_ingest = time.time()
                warned_idle = False
            elif self.actor is None:
                # actor 还没写出新分片，别空转烧 CPU
                time.sleep(self.cfg.idle_sleep_seconds)
                # 长时间一条样本都没吃到，多半是分片扫描出了问题，而不是 actor 慢。
                # 这种情况会静默地让训练步数一直停在 0，必须主动喊出来。
                idle = time.time() - last_ingest
                if idle > self.cfg.idle_warn_seconds and not warned_idle:
                    warned_idle = True
                    shard_dir = os.path.join(self.run_dir, "replay")
                    on_disk = (
                        len([f for f in os.listdir(shard_dir) if f.endswith(".npz")])
                        if os.path.isdir(shard_dir)
                        else 0
                    )
                    print(
                        f"[trainer] 警告：已连续 {idle / 60:.0f} 分钟没有吃到任何新样本，"
                        f"而 replay 目录里有 {on_disk} 个分片。"
                        "若分片数在增长却一直吃不到，说明分片扫描有问题。",
                        flush=True,
                    )

            if len(self.buffer) >= self.cfg.min_positions_to_start:
                steps = self.train_steps_for_cycle()
                for _ in range(steps):
                    metrics = self.train_step()
                    if self.step % self.cfg.log_every_steps == 0:
                        metrics.update(self._context_metrics(added))
                        metrics["train/steps_this_cycle"] = steps
                        self._log(log_path, metrics)

            if self._should_checkpoint():
                self.save_checkpoint()

        self.save_checkpoint()

    def _context_metrics(self, added: int) -> dict[str, float]:
        base = {
            "step": self.step,
            "buffer/size": len(self.buffer),
            "buffer/total_added": self.buffer.total_added,
            "buffer/reuse": self.samples_seen / max(1, self.buffer.total_added),
            "selfplay/added_last_cycle": added,
            "optim/compensation_norm": self.optimizer.compensation_norm(),
        }
        if self.actor is None:
            return base

        stats = self.actor.stats
        games = max(1, stats["games"])
        base.update(
            {
                "selfplay/games": stats["games"],
                "selfplay/plies_per_game": stats["completed_plies"] / games,
                "selfplay/black_win_rate": stats["black_wins"] / games,
                "selfplay/forbidden_loss_rate": stats["forbidden_losses"] / games,
                "selfplay/resign_false_positive_rate": (
                    stats["resign_false_positives"] / max(1, stats["resign_audits"])
                ),
            }
        )
        return base

    @staticmethod
    def _log(path: str, metrics: dict) -> None:
        with open(path, "a") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    # ── 容灾 ────────────────────────────────────────────────────────────────
    def _should_checkpoint(self) -> bool:
        if time.time() - self._last_checkpoint_time >= self.cfg.checkpoint_every_seconds:
            return True
        return self.step - self._last_checkpoint_step >= self.cfg.checkpoint_every_steps

    def checkpoint_path(self, step: int) -> str:
        return os.path.join(self.run_dir, "checkpoints", f"step_{step:09d}.pt")

    def save_checkpoint(self) -> str:
        self.buffer.flush()
        self.buffer.write_manifest(
            os.path.join(self.run_dir, "replay", "manifest.json")
        )

        state = {
            "version": CHECKPOINT_VERSION,
            "step": self.step,
            "cycle": self.cycle,
            "samples_seen": self.samples_seen,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
            "model_cfg": vars(self.model_cfg),
            "trainer_cfg": vars(self.cfg),
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": self.rng.bit_generator.state,
                "python": random.getstate(),
                # 多卡模式下自博弈在独立进程里，各自维护自己的随机流
                "selfplay": self.actor.rng_state() if self.actor is not None else [],
            },
        }

        path = self.checkpoint_path(self.step)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # 原子替换：latest 指到的一定是完整文件

        latest = os.path.join(self.run_dir, "checkpoints", "latest")
        with open(latest + ".tmp", "w") as fh:
            fh.write(os.path.basename(path))
        os.replace(latest + ".tmp", latest)

        self._prune_checkpoints()
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = self.step
        return path

    def _prune_checkpoints(self) -> None:
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        files = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step_"))
        for name in files[: max(0, len(files) - self.cfg.keep_last)]:
            try:
                os.remove(os.path.join(ckpt_dir, name))
            except OSError:
                pass

    def latest_checkpoint(self) -> str | None:
        latest = os.path.join(self.run_dir, "checkpoints", "latest")
        if not os.path.exists(latest):
            return None
        with open(latest) as fh:
            name = fh.read().strip()
        path = os.path.join(self.run_dir, "checkpoints", name)
        return path if os.path.exists(path) else None

    def load_checkpoint(self, path: str | None = None) -> bool:
        path = path or self.latest_checkpoint()
        if path is None:
            return False

        # 一律加载到 CPU：随机数状态是 ByteTensor，被搬到 GPU 上之后
        # set_rng_state 会直接拒绝。模型与优化器各自的 load_state_dict
        # 会把张量搬到正确的设备上，不需要在这里指定 map_location。
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise RuntimeError(f"checkpoint 版本不匹配: {state.get('version')}")

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        self.cycle = state["cycle"]
        self.samples_seen = state["samples_seen"]
        self.buffer.load_state_dict(state["buffer"])

        rng = state["rng"]
        torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))

        cuda_state = rng.get("torch_cuda")
        if cuda_state and torch.cuda.is_available():
            states = [s.cpu().to(torch.uint8) for s in cuda_state]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
            else:
                # 换机后可见 GPU 数可能变了。随机数状态只影响可复现性，
                # 不影响正确性，这里明确说明而不是静默跳过。
                print(
                    f"[trainer] 可见 GPU 数由 {len(states)} 变为 "
                    f"{torch.cuda.device_count()}，跳过 CUDA 随机数状态恢复"
                )
        self.rng.bit_generator.state = rng["numpy"]
        random.setstate(rng["python"])
        if self.actor is not None and rng.get("selfplay"):
            self.actor.set_rng_state(rng["selfplay"])

        restored = self.buffer.restore_from_shards()
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_step = self.step
        return True

    def resume_or_start(self) -> bool:
        """启动时自动续训；没有 checkpoint 就从头开始。"""
        return self.load_checkpoint()
