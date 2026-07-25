"""多头损失。

主任务是策略与价值；其余几个头是**训练脚手架** —— 把本该由树搜索展开才能得到的
信息直接监督进网络。对战时没有搜索可用，这些头的作用就是让网络自己长出
「看见威胁」「知道哪里是禁手」「预判对手应手」的能力。

辅助权重按 schedule 衰减：前期它们提供密集梯度帮助起步，后期让位给主任务，
免得占用容量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..model.net import NetOutput
from ..rules.constants import EMPTY
from .replay import Batch


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    threat: float = 0.3
    forbidden: float = 0.3
    plies: float = 0.1
    reply: float = 0.2

    # 辅助头的衰减 schedule
    decay_start_step: int = 200_000
    decay_end_step: int = 400_000
    decay_final_scale: float = 0.1

    @classmethod
    def from_dict(cls, cfg: dict) -> "LossWeights":
        weights = cfg.get("loss_weights", cfg)
        return cls(
            policy=weights.get("policy", 1.0),
            value=weights.get("value", 1.0),
            threat=weights.get("aux_threat", 0.3),
            forbidden=weights.get("aux_forbidden", 0.3),
            plies=weights.get("aux_plies", 0.1),
            reply=weights.get("aux_reply", 0.2),
            decay_start_step=weights.get("aux_decay_start_step", 200_000),
            decay_end_step=weights.get("aux_decay_end_step", 400_000),
            decay_final_scale=weights.get("aux_decay_final_scale", 0.1),
        )

    def aux_scale(self, step: int) -> float:
        if step <= self.decay_start_step:
            return 1.0
        if step >= self.decay_end_step:
            return self.decay_final_scale
        span = max(1, self.decay_end_step - self.decay_start_step)
        frac = (step - self.decay_start_step) / span
        return 1.0 + frac * (self.decay_final_scale - 1.0)


def value_target_classes(value: torch.Tensor) -> torch.Tensor:
    """把 {+1, 0, -1} 的对局结果映射到三分类下标：胜 0 / 和 1 / 负 2。"""
    return torch.where(
        value > 0.5,
        torch.zeros_like(value, dtype=torch.long),
        torch.where(
            value < -0.5,
            torch.full_like(value, 2, dtype=torch.long),
            torch.ones_like(value, dtype=torch.long),
        ),
    )


def plies_bucket(plies: torch.Tensor, buckets: int) -> torch.Tensor:
    """剩余手数分桶。越接近终局分辨率越高 —— 那里才是价值需要被锐化的地方。"""
    return (plies - 1).clamp(min=0, max=buckets - 1)


def compute_losses(
    out: NetOutput,
    batch: Batch,
    weights: LossWeights,
    step: int,
    num_levels: int = 7,
) -> tuple[torch.Tensor, dict[str, float]]:
    legal = batch.boards == EMPTY  # (B, N)
    metrics: dict[str, float] = {}

    # ── 策略：与 MCTS 访问分布的交叉熵 ──────────────────────────────────────
    logits = out.policy.float().masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    # 非法点的 log 概率是 -inf，而目标概率是 0；直接相乘会得到 0 * -inf = NaN。
    # 目标在这些点上本来就是 0，置零不改变损失值，只是避开这个陷阱。
    logp = torch.where(legal, logp, torch.zeros_like(logp))
    policy_loss = -(batch.policy * logp).sum(dim=-1).mean()

    with torch.no_grad():
        agree = (logits.argmax(-1) == batch.policy.argmax(-1)).float().mean()
        metrics["policy/top1_agreement"] = agree.item()
        metrics["policy/target_entropy"] = (
            -(batch.policy.clamp_min(1e-9).log() * batch.policy).sum(-1).mean().item()
        )

    # ── 价值：胜/和/负三分类 ────────────────────────────────────────────────
    value_target = value_target_classes(batch.value)
    value_loss = F.cross_entropy(out.value.float(), value_target)
    with torch.no_grad():
        metrics["value/accuracy"] = (
            (out.value.float().argmax(-1) == value_target).float().mean().item()
        )

    total = weights.policy * policy_loss + weights.value * value_loss
    metrics["loss/policy"] = policy_loss.item()
    metrics["loss/value"] = value_loss.item()

    scale = weights.aux_scale(step)
    metrics["loss/aux_scale"] = scale

    # ── 辅助头 ──────────────────────────────────────────────────────────────
    if out.threat is not None and weights.threat > 0:
        # (B, 2, L, N) -> 对空点做逐点分类
        threat_logits = out.threat.float()
        targets = torch.stack([batch.threat_self, batch.threat_opp], dim=1)  # (B,2,N)
        mask = legal.unsqueeze(1).expand_as(targets)
        flat_logits = threat_logits.permute(0, 1, 3, 2).reshape(-1, num_levels)
        flat_target = targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        threat_loss = F.cross_entropy(
            flat_logits[flat_mask], flat_target[flat_mask]
        )
        total = total + weights.threat * scale * threat_loss
        metrics["loss/threat"] = threat_loss.item()
        with torch.no_grad():
            pred = flat_logits[flat_mask].argmax(-1)
            tgt = flat_target[flat_mask]
            metrics["threat/accuracy"] = (pred == tgt).float().mean().item()
            # 只看真正有威胁的点（活三及以上），否则会被大量 NONE 稀释
            important = tgt >= 2
            if important.any():
                metrics["threat/accuracy_on_threats"] = (
                    (pred[important] == tgt[important]).float().mean().item()
                )

    if out.forbidden is not None and weights.forbidden > 0:
        fb_logits = out.forbidden.float()
        fb_loss = F.binary_cross_entropy_with_logits(
            fb_logits[legal], batch.forbidden[legal]
        )
        total = total + weights.forbidden * scale * fb_loss
        metrics["loss/forbidden"] = fb_loss.item()
        with torch.no_grad():
            target = batch.forbidden[legal] > 0.5
            pred = fb_logits[legal] > 0
            metrics["forbidden/positive_rate"] = target.float().mean().item()
            if target.any():
                # 召回率比准确率有意义得多：禁手点极其稀疏，全判负也能有 99% 准确率
                metrics["forbidden/recall"] = (
                    pred[target].float().mean().item()
                )
            if pred.any():
                metrics["forbidden/precision"] = (
                    target[pred].float().mean().item()
                )

    if out.plies is not None and weights.plies > 0:
        buckets = out.plies.shape[-1]
        plies_loss = F.cross_entropy(
            out.plies.float(), plies_bucket(batch.plies_remaining, buckets)
        )
        total = total + weights.plies * scale * plies_loss
        metrics["loss/plies"] = plies_loss.item()

    if out.reply is not None and weights.reply > 0:
        has_reply = batch.next_move >= 0
        if has_reply.any():
            reply_logits = out.reply.float().masked_fill(~legal, float("-inf"))
            reply_loss = F.cross_entropy(
                reply_logits[has_reply], batch.next_move[has_reply]
            )
            total = total + weights.reply * scale * reply_loss
            metrics["loss/reply"] = reply_loss.item()
            with torch.no_grad():
                metrics["reply/top1"] = (
                    (reply_logits[has_reply].argmax(-1) == batch.next_move[has_reply])
                    .float()
                    .mean()
                    .item()
                )

    metrics["loss/total"] = total.item()
    return total, metrics
