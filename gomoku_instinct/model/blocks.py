"""InstinctNet 的基础模块。

设计动机来自「对战时零搜索」这条约束：网络深度是唯一的前瞻深度，
所以要走「深而窄」，每个 block 都必须便宜。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm2d(nn.Module):
    """沿通道维的 RMSNorm。

    统计量在 FP32 下计算再转回原精度 —— 主权重是 BF16，归一化如果也用 BF16
    做平方和，深层网络上会明显掉精度。
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        scale = xf.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        y = xf * scale * self.weight.float().view(1, -1, 1, 1)
        return y.to(dtype)


class LineConv(nn.Module):
    """四方向直线深度卷积 —— 把「五子棋的一切都发生在直线上」写进架构。

    实现上有个便利的等价形式：把特征图右侧补 `radius` 列 gutter 再展平成一维序列后，
    四个方向恰好变成同一条序列上四种不同 dilation 的 depthwise conv1d：

        横      dilation = 1        相邻元素
        竖      dilation = Wp       跨一整行
        主对角  dilation = Wp + 1   下一行右一列
        副对角  dilation = Wp - 1   下一行左一列

    其中 Wp = size + radius。gutter 宽度恰好取核半径，是为了保证跨行采样时
    越过行末的下标一定落在补零的 gutter 里，而不会绕回下一行的真实数据。

    通道按方向四等分，每组只负责一个方向；跨方向的信息交流交给块内后面的 1x1 混合。
    这比用掩码 9x9 卷积模拟对角线省一个数量级的算力。
    """

    def __init__(self, channels: int, size: int, kernel: int = 9) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f"通道数需能被 4 整除（四个方向各一组），实际 {channels}")
        if kernel % 2 == 0:
            raise ValueError(f"核长需为奇数，实际 {kernel}")

        self.channels = channels
        self.size = size
        self.kernel = kernel
        self.radius = kernel // 2
        self.padded_width = size + self.radius
        self.group = channels // 4

        wp = self.padded_width
        self.dilations = (1, wp, wp + 1, wp - 1)

        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, std=(1.0 / kernel) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if w != self.size or h != self.size:
            raise ValueError(f"LineConv 按 {self.size}x{self.size} 构造，收到 {h}x{w}")

        wp = self.padded_width
        flat = F.pad(x, (0, self.radius)).reshape(b, c, h * wp)

        outs = []
        for i, dilation in enumerate(self.dilations):
            sl = slice(i * self.group, (i + 1) * self.group)
            outs.append(
                F.conv1d(
                    flat[:, sl],
                    self.weight[sl],
                    self.bias[sl],
                    padding=self.radius * dilation,
                    dilation=dilation,
                    groups=self.group,
                )
            )
        y = torch.cat(outs, dim=1).view(b, c, h, wp)
        return y[:, :, :, : self.size]


class SqueezeExcite(nn.Module):
    """全局门控：让整盘的局势去调制每个通道的局部响应。"""

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(8, int(channels * ratio))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class DirectionalLineBlock(nn.Module):
    """主干残差块：直线分支 + 局部形状分支，再做逐点混合与全局门控。"""

    def __init__(
        self,
        channels: int,
        size: int,
        line_kernel: int = 9,
        expansion: float = 1.0,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        hidden = max(channels, int(channels * expansion))

        self.norm = RMSNorm2d(channels)
        self.line = LineConv(channels, size, line_kernel)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.mix1 = nn.Conv2d(2 * channels, hidden, 1, bias=False)
        self.mix2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gate = SqueezeExcite(channels, se_ratio)

        # 残差分支末端置零：每个块初始为恒等映射，深网络才好训。
        nn.init.zeros_(self.mix2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = torch.cat([self.line(h), self.local(h)], dim=1)
        y = self.mix2(F.gelu(self.mix1(y)))
        return x + self.gate(y)


class GlobalAttention(nn.Module):
    """全盘自注意力。

    15x15 只有 225 个位置，全局注意力的开销与一个卷积块相当，却能直接建模
    跨越棋盘的威胁组合 —— 四三、双三这类「两处威胁互相支援」的推理，
    纯卷积要靠很多层才能间接表达，而无搜索推理恰恰最依赖这种能力。
    """

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"通道数 {channels} 不能被头数 {heads} 整除")
        self.heads = heads
        self.norm = RMSNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.heads

        # 一次性排成 (3, B, heads, N, head_dim) 并落成连续内存。
        # 这里必须连续：SDPA 的 flash 实现要求最后一维 stride 为 1，
        # 否则会悄悄退回 math 路径，把 (B, heads, N, N) 的注意力矩阵整个materialize 出来 ——
        # 实测那条路径比 flash 慢一个数量级。
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, head_dim, n)
        qkv = qkv.permute(1, 0, 2, 4, 3).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        y = F.scaled_dot_product_attention(q, k, v)
        y = y.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(y)
