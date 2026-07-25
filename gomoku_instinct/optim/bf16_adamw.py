"""主权重为 BF16 的 AdamW。

与常规混合精度（FP32 master weights + BF16 autocast）不同，这里**权重本身就是 BF16**。
这样做有个必须正面处理的问题：

BF16 只有 7 位存储尾数（含隐含位共 8 位有效位），1.0 附近相邻两个可表示数相距 2^-7。
当 `|lr*g|` 小于半个 ULP 时，`w += lr*g` 会被舍入完整吃掉。scripts/check_env.py 里
实测过：w=1、步长 1e-4 连加 100 次，**没有任何一个元素发生变化**。
训练初期梯度大还看不出来，到中后期 lr 衰减、梯度变小，更新会悄悄归零 —— 不报错，
loss 曲线就是平了。

三种处理方式：

  kahan       Kahan 补偿求和。把每步被舍掉的低位存进补偿缓冲区，下一步先补回去再加。
              确定性，可复现，默认方案。补偿缓冲区本身用 BF16 存 —— 它装的是残差，
              BF16 的宽指数范围足以表示远小于 w 的 ULP 的量。
  stochastic  随机舍入。按余数占一个 ULP 的比例随机进位，期望无偏。作对照实验用。
  none        不补偿。只用来复现「更新停滞」这一现象，不要用于正式训练。

Adam 的一阶/二阶矩保持 FP32：矩是优化器的内部状态，不是权重，
保持 FP32 不违反「主权重为 BF16」，却能避免二阶矩在长训练中失真。
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

LOW_PRECISION_DTYPES = (torch.bfloat16, torch.float16)


def stochastic_round_to_bf16(
    x: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """把 FP32 张量随机舍入到 BF16。

    做法是在 FP32 的位模式低 16 位上加一个均匀随机数再截断。截断本身总是朝零舍入，
    加噪声后「进位」的概率恰好等于被丢弃部分占一个 ULP 的比例，因此期望无偏。
    正负数都成立：截断清掉的是尾数低位，对两种符号都是朝零舍入。
    """
    if x.dtype != torch.float32:
        raise ValueError(f"随机舍入的输入需为 float32，实际 {x.dtype}")
    bits = x.view(torch.int32)
    noise = torch.randint(
        0,
        1 << 16,
        x.shape,
        dtype=torch.int32,
        device=x.device,
        generator=generator,
    )
    return (bits + noise).bitwise_and_(-65536).view(torch.float32).to(torch.bfloat16)


class BF16AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rounding: str = "kahan",
        moment_dtype: torch.dtype = torch.float32,
    ) -> None:
        if rounding not in ("kahan", "stochastic", "none"):
            raise ValueError(f"未知的舍入方式: {rounding}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas 取值非法: {betas}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rounding=rounding
        )
        super().__init__(params, defaults)
        self.moment_dtype = moment_dtype
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> None:
        """指定随机舍入用的随机源，便于 resume 时精确复现。"""
        self._generator = generator

    # ── 单步更新 ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            rounding = group["rounding"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    self._init_state(state, p, rounding)

                state["step"] += 1
                t = state["step"]

                grad = p.grad.to(self.moment_dtype)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1**t
                bias_c2 = 1.0 - beta2**t

                denom = exp_avg_sq.div(bias_c2).sqrt_().add_(eps)
                update = exp_avg.div(denom).mul_(-lr / bias_c1)

                if weight_decay != 0.0:
                    # 解耦权重衰减：不经过 Adam 的自适应缩放
                    update.add_(p.to(update.dtype), alpha=-lr * weight_decay)

                self._apply_update(p, update, state, rounding)

        return loss

    def _init_state(self, state: dict, p: torch.Tensor, rounding: str) -> None:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=self.moment_dtype)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.moment_dtype)
        # 只有低精度权重才需要补偿缓冲区
        if rounding == "kahan" and p.dtype in LOW_PRECISION_DTYPES:
            state["compensation"] = torch.zeros_like(p)

    def _apply_update(
        self, p: torch.Tensor, update: torch.Tensor, state: dict, rounding: str
    ) -> None:
        if p.dtype not in LOW_PRECISION_DTYPES:
            p.add_(update.to(p.dtype))
            return

        if rounding == "kahan":
            comp = state["compensation"]
            # 先把上一步被舍掉的低位补回来
            y = update.add(comp.to(update.dtype))
            new_p = (p.to(y.dtype) + y).to(p.dtype)
            # 实际被吃进权重的增量，与 y 的差就是这一步丢掉的部分
            applied = new_p.to(y.dtype).sub_(p.to(y.dtype))
            comp.copy_(y.sub_(applied).to(p.dtype))
            p.copy_(new_p)
        elif rounding == "stochastic":
            p.copy_(
                stochastic_round_to_bf16(
                    p.to(torch.float32) + update.to(torch.float32), self._generator
                )
            )
        else:
            p.add_(update.to(p.dtype))

    # ── 续训 ────────────────────────────────────────────────────────────────
    def load_state_dict(self, state_dict) -> None:
        """恢复优化器状态，并把各缓冲区的精度按本优化器的约定还原。

        必须覆盖基类实现：`torch.optim.Optimizer.load_state_dict` 会把所有浮点状态
        **强制转成对应参数的 dtype**。主权重是 BF16，于是 FP32 的一阶/二阶矩会在
        续训时被静默降成 BF16 —— 二阶矩一失真，更新量整体走样，而且不报任何错。
        开发机有使用时长限制、随时可能换机续训，这个坑不堵住迟早会踩。
        """
        super().load_state_dict(state_dict)

        # 基类按「保存时的参数序号 -> 当前参数」建立映射，这里照同样的顺序还原。
        saved_params = [pid for g in state_dict["param_groups"] for pid in g["params"]]
        current_params = [p for g in self.param_groups for p in g["params"]]
        id_map = dict(zip(saved_params, current_params))

        for param_id, saved in state_dict["state"].items():
            param = id_map.get(param_id)
            if param is None:
                continue
            target = self.state[param]
            for key, value in saved.items():
                if not torch.is_tensor(value):
                    continue
                dtype = param.dtype if key == "compensation" else self.moment_dtype
                target[key] = value.detach().to(device=param.device, dtype=dtype).clone()

    # ── 诊断 ────────────────────────────────────────────────────────────────
    def compensation_norm(self) -> float:
        """所有补偿缓冲区的整体 L2 范数。

        它稳定地大于 0，说明确实有更新量正在被 BF16 舍入吃掉、并被补偿机制救了回来。
        训练时把它记进指标，可以直接看出「主权重 BF16 是否已经开始拖后腿」。
        """
        total = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                comp = self.state.get(p, {}).get("compensation")
                if comp is not None:
                    total += comp.to(torch.float32).pow(2).sum().item()
        return math.sqrt(total)


def clip_grad_norm_fp32(
    parameters: Iterable[torch.Tensor], max_norm: float
) -> torch.Tensor:
    """在 FP32 下计算并裁剪梯度范数。

    BF16 梯度直接求平方和很容易在通道数多时溢出或损失精度，
    所以范数一律升到 FP32 算。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())

    device = grads[0].device
    total = torch.zeros((), dtype=torch.float32, device=device)
    for g in grads:
        total += g.to(torch.float32).pow(2).sum()
    total_norm = total.sqrt()

    if max_norm > 0:
        scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(scale.to(g.dtype))
    return total_norm


def build_optimizer(model: nn.Module, cfg: dict) -> BF16AdamW:
    """按 configs/train_*.yaml 的 train.optim 段构造优化器。

    归一化层的缩放系数与所有偏置不做权重衰减 —— 它们不是「大小需要被压住」的量。
    """
    optim_cfg = cfg.get("train", {}).get("optim", cfg)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    weight_decay = optim_cfg.get("weight_decay", 1e-4)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return BF16AdamW(
        groups,
        lr=optim_cfg.get("lr", 2e-3),
        betas=tuple(optim_cfg.get("betas", (0.9, 0.95))),
        eps=optim_cfg.get("eps", 1e-8),
        weight_decay=weight_decay,
        rounding=optim_cfg.get("rounding", "kahan"),
    )
