"""InstinctNet 结构测试。

重点是 LineConv —— 「四方向直线卷积等价于同一条展平序列上四种 dilation 的
depthwise conv1d」这个等价关系是主干的核心前提，差一格就会静默地把对角线
采到别的地方去，因此拿一份朴素的二维参考实现来逐点比对。
"""

from __future__ import annotations

import pytest
import torch

from gomoku_instinct.model import (
    InstinctNet,
    LineConv,
    ModelConfig,
    NUM_PLANES,
    encode,
    legal_mask,
)
from gomoku_instinct.rules import BLACK, WHITE
from gomoku_instinct.rules.constants import DIRECTIONS

# ── LineConv 的方向正确性 ───────────────────────────────────────────────────


def _reference_line_conv(x, weight, bias, kernel, group):
    """朴素二维参考实现：直接按 (dr, dc) 在网格上取样，越界记 0。

    与 LineConv 里「补 gutter + 展平 + 带 dilation 的 conv1d」是完全不同的做法，
    因此比对结果一致才说明那套下标推导是对的。
    """
    b, c, h, w = x.shape
    radius = kernel // 2
    out = torch.zeros_like(x)
    for ch in range(c):
        dr, dc = DIRECTIONS[ch // group]
        for r in range(h):
            for col in range(w):
                acc = torch.full((b,), bias[ch].item(), dtype=x.dtype)
                for j in range(kernel):
                    off = j - radius
                    rr, cc = r + off * dr, col + off * dc
                    if 0 <= rr < h and 0 <= cc < w:
                        acc = acc + weight[ch, 0, j] * x[:, ch, rr, cc]
                out[:, ch, r, col] = acc
    return out


@pytest.mark.parametrize("kernel", [3, 5, 9])
def test_line_conv_matches_naive_reference(kernel):
    torch.manual_seed(0)
    size, channels = 7, 8
    conv = LineConv(channels, size, kernel=kernel)
    with torch.no_grad():
        conv.weight.normal_()
        conv.bias.normal_()

    x = torch.randn(2, channels, size, size)
    got = conv(x)
    want = _reference_line_conv(
        x, conv.weight, conv.bias, kernel, conv.group
    )
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_line_conv_direction_assignment():
    """通道按方向四等分：第 i 组只沿第 i 个方向采样。

    把权重设成只保留最后一个抽头，输入放一个脉冲，输出的位置就直接暴露了方向。
    """
    size, channels = 5, 4
    conv = LineConv(channels, size, kernel=3)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, 0, 2] = 1.0  # 只取 +1 步的邻居
        conv.bias.zero_()

    x = torch.zeros(1, channels, size, size)
    x[:, :, 2, 2] = 1.0
    y = conv(x)

    # out[r, c] = x[r + dr, c + dc]，所以脉冲会出现在 (2,2) - (dr,dc) 处
    expected = {0: (2, 1), 1: (1, 2), 2: (1, 1), 3: (1, 3)}
    for ch, (r, c) in expected.items():
        assert y[0, ch, r, c].item() == pytest.approx(1.0), f"通道 {ch} 方向错了"
        assert y[0, ch].abs().sum().item() == pytest.approx(1.0)


def test_line_conv_gutter_prevents_row_wraparound():
    """展平之后行末与下一行行首在一维上是相邻的；gutter 必须挡住这种串扰。

    没有 gutter 的话，横向卷积会把 (r, W-1) 的值采到 (r+1, 0) 上去。
    """
    size, channels = 5, 4
    conv = LineConv(channels, size, kernel=3)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, 0, 0] = 1.0  # 只取 -1 步的邻居
        conv.bias.zero_()

    x = torch.zeros(1, channels, size, size)
    x[:, :, 2, size - 1] = 1.0  # 放在行末
    y = conv(x)

    # 横向通道：响应本该落在 (2, size)，那是 gutter，切片后应当什么都不剩
    assert y[0, 0].abs().sum().item() == pytest.approx(0.0), "横向发生了跨行串扰"
    # 副对角 (1,-1)：out[r,c] = x[r-1, c+1]，脉冲应落在 (3, size-2)
    assert y[0, 3, 3, size - 2].item() == pytest.approx(1.0)


def test_line_conv_requires_channels_divisible_by_four():
    with pytest.raises(ValueError):
        LineConv(6, 15, kernel=9)


# ── 输入特征 ────────────────────────────────────────────────────────────────


def test_encode_shapes_and_planes():
    size = 15
    n = size * size
    boards = torch.zeros(3, n, dtype=torch.uint8)
    boards[0, 0] = BLACK
    boards[0, 1] = WHITE
    to_move = torch.tensor([BLACK, WHITE, BLACK], dtype=torch.uint8)
    history = torch.full((3, 4), -1, dtype=torch.int64)
    history[0, 0] = 1
    move_number = torch.tensor([2, 0, 0], dtype=torch.int64)

    planes = encode(boards, to_move, history, move_number, size, dtype=torch.float32)
    assert planes.shape == (3, NUM_PLANES, size, size)

    assert planes[0, 0, 0, 0] == 1.0  # 黑子
    assert planes[0, 1, 0, 1] == 1.0  # 白子
    assert planes[0, 2, 0, 0] == 0.0  # 该点非空
    assert planes[0, 3].mean() == 1.0  # 轮到黑走
    assert planes[1, 3].mean() == 0.0
    assert planes[0, 4, 0, 1] == 1.0  # 最近一手
    assert planes[0, 4].sum() == 1.0
    assert planes[1, 4].sum() == 0.0  # 无历史


def test_center_distance_plane_is_symmetric():
    """位置平面必须在八重对称下不变，否则对称数据增强就不成立了。"""
    size = 15
    boards = torch.zeros(1, size * size, dtype=torch.uint8)
    planes = encode(
        boards,
        torch.tensor([BLACK], dtype=torch.uint8),
        torch.full((1, 4), -1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        size,
        dtype=torch.float32,
    )
    plane = planes[0, 9]
    assert torch.allclose(plane, plane.flip(0))
    assert torch.allclose(plane, plane.flip(1))
    assert torch.allclose(plane, plane.T)


def test_legal_mask_keeps_forbidden_points():
    """禁手点在严格 RIF 语义下是合法落子，掩码只排除已占点。"""
    boards = torch.zeros(1, 9, dtype=torch.uint8)
    boards[0, 3] = BLACK
    mask = legal_mask(boards)
    assert mask.sum().item() == 8
    assert not mask[0, 3]


# ── 整网 ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_net():
    cfg = ModelConfig(size=9, channels=32, blocks=4, attn_every=2, attn_heads=4)
    return InstinctNet(cfg)


def test_forward_shapes(small_net):
    size = small_net.cfg.size
    x = torch.randn(2, NUM_PLANES, size, size)
    out = small_net(x)
    n = size * size
    assert out.policy.shape == (2, n)
    assert out.value.shape == (2, 3)
    assert out.threat.shape == (2, 2, small_net.cfg.threat_levels, n)
    assert out.forbidden.shape == (2, n)
    assert out.plies.shape == (2, small_net.cfg.plies_buckets)
    assert out.reply.shape == (2, n)


def test_forward_without_aux_skips_heads(small_net):
    size = small_net.cfg.size
    out = small_net(torch.randn(1, NUM_PLANES, size, size), with_aux=False)
    assert out.threat is None and out.forbidden is None
    assert out.aux_items() == {}


def test_bf16_forward_is_finite(small_net):
    net = InstinctNet(small_net.cfg).to(torch.bfloat16)
    size = net.cfg.size
    x = torch.randn(4, NUM_PLANES, size, size, dtype=torch.bfloat16)
    out = net(x)
    assert out.policy.dtype == torch.bfloat16
    assert torch.isfinite(out.policy.float()).all()
    assert torch.isfinite(out.value.float()).all()
    assert all(p.dtype == torch.bfloat16 for p in net.parameters())


def test_blocks_start_as_identity(small_net):
    """残差分支末端置零，初始时主干应当是恒等映射 —— 深网络才好起步。"""
    net = InstinctNet(small_net.cfg)
    size = net.cfg.size
    x = net.stem(torch.randn(2, NUM_PLANES, size, size))
    y = x
    for layer in net.trunk:
        y = layer(y)
    assert torch.allclose(x, y, atol=1e-6)


def test_all_parameters_receive_gradients(small_net):
    net = InstinctNet(small_net.cfg)
    size = net.cfg.size
    out = net(torch.randn(2, NUM_PLANES, size, size))
    loss = (
        out.policy.square().mean()
        + out.value.square().mean()
        + out.threat.square().mean()
        + out.forbidden.square().mean()
        + out.plies.square().mean()
        + out.reply.square().mean()
    )
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"这些参数没有梯度: {missing}"


def test_masked_logits_blocks_occupied_points(small_net):
    logits = torch.zeros(1, 9)
    legal = torch.ones(1, 9, dtype=torch.bool)
    legal[0, 4] = False
    masked = InstinctNet.masked_logits(logits, legal)
    assert masked[0, 4] == float("-inf")
    assert masked.softmax(-1)[0, 4] == 0.0


def test_value_scalar_range(small_net):
    logits = torch.tensor([[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0]])
    v = InstinctNet.value_scalar(logits)
    assert v[0] > 0.99 and v[1] < -0.99


def test_default_model_size_is_reasonable():
    net = InstinctNet(ModelConfig())
    params = net.num_parameters()
    assert 2e6 < params < 12e6, f"默认规模 {params / 1e6:.2f}M 超出预期区间"
