"""从 checkpoint 载入模型。

checkpoint 里带着当时的 ModelConfig，所以对战与竞技场不需要再指定网络结构 ——
避免「权重与结构对不上」这种只在加载时才炸的错误。
"""

from __future__ import annotations

import os

import torch

from .net import InstinctNet, ModelConfig


def resolve_checkpoint(path: str) -> str:
    """允许直接给 run 目录：自动找到其中的 latest。"""
    if os.path.isdir(path):
        for candidate in (
            os.path.join(path, "checkpoints", "latest"),
            os.path.join(path, "latest"),
        ):
            if os.path.exists(candidate):
                with open(candidate) as fh:
                    name = fh.read().strip()
                return os.path.join(os.path.dirname(candidate), name)
        raise FileNotFoundError(f"{path} 下没有找到 checkpoint")
    return path


def load_model(
    path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[InstinctNet, dict]:
    """返回 (模型, checkpoint 元信息)。"""
    path = resolve_checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=False)

    cfg_dict = state.get("model_cfg")
    if cfg_dict is None:
        raise RuntimeError(f"{path} 里没有 model_cfg，无法确定网络结构")
    cfg = ModelConfig(**cfg_dict)

    model = InstinctNet(cfg)
    model.load_state_dict(state["model"])
    model = model.to(device).to(dtype).eval()

    meta = {
        "path": path,
        "step": state.get("step", 0),
        "cycle": state.get("cycle", 0),
        "board_size": cfg.size,
    }
    return model, meta
