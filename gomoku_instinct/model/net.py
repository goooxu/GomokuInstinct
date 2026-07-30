"""InstinctNet —— 本项目自研的连珠网络。

结构取舍全部围绕「对战时零搜索」这一条约束：

  * **深而窄**：网络深度是唯一的前瞻深度，所以宁可多堆层也不加宽。
  * **直线先验写进架构**：主干每块都有一条四方向长核直线卷积分支。
  * **全盘自注意力**：225 个位置的全局注意力很便宜，用来建模跨越棋盘的
    威胁组合（四三、双三这类需要两处威胁互相支援的推理）。
  * **多个辅助监督头**：把本该由树搜索展开才能得到的信息（棋型、禁手点、
    剩余手数、对手应手）直接作为监督目标灌进网络。这些标签全部由规则导出，
    不含任何棋谱知识。
  * **策略头全卷积**：不做展平的全连接，因此与棋盘尺寸无关，
    可以先在小棋盘上训练再迁移到 15x15。

价值与策略都以**当前行棋方**为视角。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import DirectionalLineBlock, GlobalAttention, RMSNorm2d
from ..rules.constants import NUM_LEVELS
from .features import NUM_PLANES


@dataclass
class ModelConfig:
    size: int = 15
    channels: int = 128
    blocks: int = 20
    line_kernel: int = 9
    attn_every: int = 4
    attn_heads: int = 4
    se_ratio: float = 0.25
    expansion: float = 1.0
    input_planes: int = NUM_PLANES

    threat_levels: int = NUM_LEVELS  # 与 rules.constants.Level 一致
    plies_buckets: int = 16

    use_threat_head: bool = True
    use_forbidden_head: bool = True
    use_plies_head: bool = True
    use_reply_head: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        model = cfg.get("model", cfg)
        heads = cfg.get("heads", {})
        return cls(
            size=model.get("board_size", 15),
            channels=model.get("channels", 128),
            blocks=model.get("blocks", 20),
            line_kernel=model.get("line_kernel", 9),
            attn_every=model.get("attn_every", 4),
            attn_heads=model.get("attn_heads", 4),
            se_ratio=model.get("se_ratio", 0.25),
            expansion=model.get("expansion", 1.0),
            input_planes=model.get("input_planes", NUM_PLANES),
            use_threat_head=heads.get("aux_threat", True),
            use_forbidden_head=heads.get("aux_forbidden", True),
            use_plies_head=heads.get("aux_plies", True),
            use_reply_head=heads.get("aux_reply", True),
        )


@dataclass
class NetOutput:
    """网络的全部输出。除 policy/value 外都是训练用的辅助头。"""

    policy: torch.Tensor  # (B, N) 未屏蔽的 logits
    value: torch.Tensor  # (B, 3) 胜/和/负，行棋方视角
    threat: torch.Tensor | None = None  # (B, 2, L, N) 行棋方与对方的棋型等级
    forbidden: torch.Tensor | None = None  # (B, N) 黑方禁手点 logits
    plies: torch.Tensor | None = None  # (B, buckets) 剩余手数分桶
    reply: torch.Tensor | None = None  # (B, N) 对手应手分布 logits

    def aux_items(self) -> dict[str, torch.Tensor]:
        out = {}
        for name in ("threat", "forbidden", "plies", "reply"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


class InstinctNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg.channels
        size = self.cfg.size
        self.num_cells = size * size

        self.stem = nn.Conv2d(self.cfg.input_planes, c, 3, padding=1, bias=False)

        layers: list[nn.Module] = []
        for i in range(self.cfg.blocks):
            layers.append(
                DirectionalLineBlock(
                    c,
                    size,
                    line_kernel=self.cfg.line_kernel,
                    expansion=self.cfg.expansion,
                    se_ratio=self.cfg.se_ratio,
                )
            )
            # 每隔 attn_every 块插一层全局注意力（最后一块之后不插）
            if (
                self.cfg.attn_every > 0
                and (i + 1) % self.cfg.attn_every == 0
                and i + 1 < self.cfg.blocks
            ):
                layers.append(GlobalAttention(c, self.cfg.attn_heads))
        self.trunk = nn.ModuleList(layers)

        self.final_norm = RMSNorm2d(c)

        # 策略头：1x1 卷积出一张 logit 图，无展平全连接 -> 与棋盘尺寸无关
        self.policy_head = nn.Conv2d(c, 1, 1)

        # 价值与剩余手数共享池化表征（均值 + 最大值）
        self.pool_proj = nn.Linear(2 * c, c)
        self.value_head = nn.Linear(c, 3)
        self.plies_head = (
            nn.Linear(c, self.cfg.plies_buckets) if self.cfg.use_plies_head else None
        )

        self.threat_head = (
            nn.Conv2d(c, 2 * self.cfg.threat_levels, 1)
            if self.cfg.use_threat_head
            else None
        )
        self.forbidden_head = (
            nn.Conv2d(c, 1, 1) if self.cfg.use_forbidden_head else None
        )
        self.reply_head = nn.Conv2d(c, 1, 1) if self.cfg.use_reply_head else None

    # ── 前向 ────────────────────────────────────────────────────────────────
    def forward(self, planes: torch.Tensor, with_aux: bool = True) -> NetOutput:
        """planes: (B, input_planes, size, size)"""
        x = self.stem(planes)
        for layer in self.trunk:
            x = layer(x)
        x = self.final_norm(x)

        b = x.shape[0]
        policy = self.policy_head(x).reshape(b, self.num_cells)

        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        pooled = F.gelu(self.pool_proj(pooled))
        value = self.value_head(pooled)

        out = NetOutput(policy=policy, value=value)
        if not with_aux:
            return out

        if self.threat_head is not None:
            threat = self.threat_head(x)
            out.threat = threat.reshape(
                b, 2, self.cfg.threat_levels, self.num_cells
            )
        if self.forbidden_head is not None:
            out.forbidden = self.forbidden_head(x).reshape(b, self.num_cells)
        if self.plies_head is not None:
            out.plies = self.plies_head(pooled)
        if self.reply_head is not None:
            out.reply = self.reply_head(x).reshape(b, self.num_cells)
        return out

    # ── 推理辅助 ────────────────────────────────────────────────────────────
    @staticmethod
    def masked_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """把非法点（已占点）压到负无穷。**返回的仍是 logits，不是概率。**

        注意：禁手点在严格 RIF 语义下是合法落子，因此**不**在这里屏蔽。

        这是零搜索部署路径上唯一的非模型逻辑（第 10 章），所以只留这一份定义 ——
        对战引擎的单局与批量两条路径都调它，免得哪天有人只改了其中一处。
        """
        return logits.float().masked_fill(~legal, float("-inf"))

    @staticmethod
    def value_scalar(value_logits: torch.Tensor) -> torch.Tensor:
        """把三分类价值折成 [-1, 1] 的标量：P(胜) - P(负)。"""
        p = value_logits.float().softmax(dim=-1)
        return p[..., 0] - p[..., 2]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> str:
        total = self.num_parameters()
        trunk = sum(p.numel() for p in self.trunk.parameters())
        stem = sum(p.numel() for p in self.stem.parameters())
        heads = total - trunk - stem
        return (
            f"InstinctNet  {self.cfg.blocks} blocks x {self.cfg.channels} ch  "
            f"参数 {total / 1e6:.2f}M（主干 {trunk / 1e6:.2f}M / "
            f"stem {stem / 1e3:.0f}K / 头部 {heads / 1e3:.0f}K）"
        )


def build_model(cfg: dict | ModelConfig | None = None) -> InstinctNet:
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return InstinctNet(cfg)
