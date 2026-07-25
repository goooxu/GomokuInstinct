"""BF16 主权重优化器的数值测试。

核心要验证的是：主权重用 BF16 时，朴素更新会**静默停滞**，而补偿机制能救回来。
这不是理论担忧 —— 下面第一个测试就是这个现象的直接复现。
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gomoku_instinct.optim import BF16AdamW, clip_grad_norm_fp32, stochastic_round_to_bf16

LR = 1e-5
STEPS = 1000


def _run_constant_gradient(rounding: str, steps: int = STEPS, lr: float = LR):
    """对一个 BF16 参数施加恒定梯度，看权重到底动没动。

    Adam 会把更新量归一化到约 lr 的量级，所以每步的更新约为 -lr = -1e-5。
    而 w=1 处 BF16 的一个 ULP 是 2^-8 ≈ 3.9e-3 —— 单步更新比 ULP 小两个数量级。
    """
    p = nn.Parameter(torch.ones(1024, dtype=torch.bfloat16))
    opt = BF16AdamW([p], lr=lr, weight_decay=0.0, rounding=rounding)
    for _ in range(steps):
        p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad(set_to_none=False)
    return p, opt


def test_naive_bf16_update_stalls_completely():
    """不补偿时，1000 步累计 -1e-2 的更新量被舍入吃得一点不剩。

    这正是「主权重 BF16」必须补偿的直接依据：不会报错，loss 曲线就是平了。
    """
    p, _ = _run_constant_gradient("none")
    assert torch.all(p.detach() == 1.0), "朴素 BF16 更新竟然动了，需要重新评估补偿方案"


def test_kahan_compensation_recovers_the_update():
    """Kahan 补偿下，同样的 1000 步应当把权重推到约 1 - 1e-2。"""
    p, _ = _run_constant_gradient("kahan")
    expected = 1.0 - LR * STEPS
    got = p.detach().float()
    assert torch.allclose(got, torch.full_like(got, expected), atol=4e-3), (
        f"期望约 {expected}，实际 {got.mean().item()}"
    )


def test_stochastic_rounding_also_recovers_the_update():
    """随机舍入是期望无偏的，均值应当同样落在 1 - 1e-2 附近。"""
    torch.manual_seed(0)
    p, _ = _run_constant_gradient("stochastic")
    expected = 1.0 - LR * STEPS
    assert abs(p.detach().float().mean().item() - expected) < 4e-3


def test_compensation_buffer_actually_holds_something():
    """补偿缓冲区非零，说明确实有更新量正在被舍入吃掉并被救回来。"""
    _, opt = _run_constant_gradient("kahan", steps=50)
    assert opt.compensation_norm() > 0.0


# ── 与 FP32 AdamW 的轨迹对比 ────────────────────────────────────────────────


def _train_linear(param_dtype, rounding, steps=400, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    layer = nn.Linear(64, 64, bias=False)
    torch.manual_seed(seed + 1)
    target = torch.randn(64, 64)
    torch.manual_seed(seed + 2)
    x = torch.randn(256, 64)

    layer = layer.to(param_dtype)
    if param_dtype == torch.float32:
        opt = torch.optim.AdamW(
            layer.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0
        )
    else:
        opt = BF16AdamW(
            layer.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            rounding=rounding,
        )

    y = x @ target.T
    for _ in range(steps):
        pred = layer(x.to(param_dtype))
        loss = (pred.float() - y).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return layer.weight.detach().float(), loss.item()


def test_kahan_tracks_fp32_better_than_no_compensation():
    """以 FP32 AdamW 的最终权重为基准，Kahan 应当明显更接近它。"""
    ref, ref_loss = _train_linear(torch.float32, "none")
    kahan, kahan_loss = _train_linear(torch.bfloat16, "kahan")
    naive, naive_loss = _train_linear(torch.bfloat16, "none")

    dist_kahan = (kahan - ref).norm().item()
    dist_naive = (naive - ref).norm().item()

    assert dist_kahan < dist_naive, (
        f"Kahan 距 FP32 轨迹 {dist_kahan:.4f}，未补偿 {dist_naive:.4f}"
    )
    assert kahan_loss < naive_loss, f"kahan={kahan_loss:.5f} naive={naive_loss:.5f}"
    # 补偿之后应当能逼近 FP32 的收敛水平
    assert kahan_loss < ref_loss * 3.0 + 1e-6


# ── 随机舍入 ────────────────────────────────────────────────────────────────


def test_stochastic_rounding_is_unbiased():
    """随机舍入的期望应当等于原值。

    取一个落在两个相邻 BF16 之间 1/4 处的值，大量舍入后的均值应当逼近它本身，
    且只可能落在这两个数上。BF16 有 7 位存储尾数，所以 1.0 上方的间隔是 2^-7。
    """
    torch.manual_seed(1234)
    lo = 1.0
    ulp = 2.0**-7
    assert torch.tensor(lo + ulp, dtype=torch.bfloat16).float().item() == lo + ulp
    assert torch.tensor(lo + ulp / 4, dtype=torch.bfloat16).float().item() == lo

    value = lo + 0.25 * ulp
    x = torch.full((200_000,), value, dtype=torch.float32)
    rounded = stochastic_round_to_bf16(x).float()

    assert set(rounded.unique().tolist()) <= {lo, lo + ulp}
    assert abs(rounded.mean().item() - value) < ulp * 0.02
    # 进位比例应当约等于 1/4
    assert abs((rounded > lo).float().mean().item() - 0.25) < 0.01


def test_stochastic_rounding_handles_negative_values():
    torch.manual_seed(7)
    x = torch.full((100_000,), -(1.0 + 0.25 * 2.0**-8), dtype=torch.float32)
    rounded = stochastic_round_to_bf16(x).float()
    assert abs(rounded.mean().item() - x[0].item()) < 2e-4
    assert torch.all(rounded < 0)


# ── 状态保存与续训 ──────────────────────────────────────────────────────────


def test_state_dict_roundtrip_preserves_compensation():
    """续训必须连补偿缓冲区一起恢复，否则会丢掉已经攒下的低位。"""
    p = nn.Parameter(torch.ones(512, dtype=torch.bfloat16))
    opt = BF16AdamW([p], lr=LR, weight_decay=0.0, rounding="kahan")
    for _ in range(120):
        p.grad = torch.ones_like(p)
        opt.step()

    state = copy.deepcopy(opt.state_dict())
    saved_param = p.detach().clone()

    # 不中断地再跑 120 步
    for _ in range(120):
        p.grad = torch.ones_like(p)
        opt.step()
    uninterrupted = p.detach().clone()

    # 从快照恢复后再跑同样的 120 步
    p2 = nn.Parameter(saved_param.clone())
    opt2 = BF16AdamW([p2], lr=LR, weight_decay=0.0, rounding="kahan")
    opt2.load_state_dict(state)
    for _ in range(120):
        p2.grad = torch.ones_like(p2)
        opt2.step()

    assert torch.equal(p2.detach(), uninterrupted), "恢复后的轨迹与不中断的不一致"


def test_load_state_dict_keeps_moments_in_fp32():
    """基类的 load_state_dict 会把优化器状态强制转成参数 dtype。

    主权重是 BF16，若不拦住这次转换，FP32 的一阶/二阶矩在续训时会被静默降精度，
    二阶矩一失真整个更新量就走样了，而且不报错。
    """
    p = nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    opt = BF16AdamW([p], lr=LR, weight_decay=0.0, rounding="kahan")
    for _ in range(10):
        p.grad = torch.ones_like(p)
        opt.step()

    p2 = nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    opt2 = BF16AdamW([p2], lr=LR, weight_decay=0.0, rounding="kahan")
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))

    assert opt2.state[p2]["exp_avg"].dtype == torch.float32
    assert opt2.state[p2]["exp_avg_sq"].dtype == torch.float32
    assert opt2.state[p2]["compensation"].dtype == torch.bfloat16
    assert torch.equal(opt2.state[p2]["exp_avg"], opt.state[p]["exp_avg"])


def test_compensation_is_dropped_for_fp32_params():
    """FP32 参数不需要补偿缓冲区，不应白占显存。"""
    p = nn.Parameter(torch.ones(16, dtype=torch.float32))
    opt = BF16AdamW([p], lr=1e-3, rounding="kahan")
    p.grad = torch.ones_like(p)
    opt.step()
    assert "compensation" not in opt.state[p]


# ── 梯度裁剪 ────────────────────────────────────────────────────────────────


def test_clip_grad_norm_scales_in_fp32():
    p = nn.Parameter(torch.zeros(1000, dtype=torch.bfloat16))
    p.grad = torch.full_like(p, 0.5)
    before = p.grad.float().norm().item()
    total = clip_grad_norm_fp32([p], max_norm=1.0)
    assert pytest.approx(before, rel=1e-2) == total.item()
    assert p.grad.float().norm().item() <= 1.0 + 1e-2


def test_clip_grad_norm_leaves_small_grads_alone():
    p = nn.Parameter(torch.zeros(10, dtype=torch.bfloat16))
    p.grad = torch.full_like(p, 0.01)
    before = p.grad.clone()
    clip_grad_norm_fp32([p], max_norm=100.0)
    assert torch.equal(p.grad, before)
