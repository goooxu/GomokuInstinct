"""自博弈 actor：把 C++ 的向量化 runner 与 GPU 上的网络接起来。

一轮的节奏：

    runner.collect(...)   每局下潜一次，攒出一个批次的待评估局面
    evaluator(...)        编码特征、跑一次网络前向
    runner.apply(...)     展开叶子、回传价值，搜索次数够了就落子

批大小恒等于对局数，所以 GPU 拿到的永远是同一形状的定长批，
可以直接被 CUDA Graph 捕获。
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import torch

from ..core import load_core
from ..model import InstinctNet, encode
from ..model.features import NUM_HISTORY_PLANES
from ..rules.constants import EMPTY

HISTORY = NUM_HISTORY_PLANES


class Evaluator(Protocol):
    """给一批局面打分：返回 (policy, value)。

    policy 为 (G, N) float32，已按空点屏蔽并归一化；
    value 为 (G,) float32，行棋方视角，取值 [-1, 1]。
    """

    def __call__(
        self,
        boards: np.ndarray,
        to_move: np.ndarray,
        history: np.ndarray,
        move_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class ModelEvaluator:
    """用 InstinctNet 打分。"""

    def __init__(
        self,
        model: torch.nn.Module,
        board_size: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.size = board_size
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, boards, to_move, history, move_number):
        dev = self.device
        b = torch.from_numpy(boards).to(dev, non_blocking=True)
        t = torch.from_numpy(to_move).to(dev, non_blocking=True)
        h = torch.from_numpy(history).to(dev, non_blocking=True).to(torch.int64)
        m = torch.from_numpy(move_number).to(dev, non_blocking=True).to(torch.int64)

        planes = encode(b, t, h, m, self.size, dtype=self.dtype)
        out = self.model(planes, with_aux=False)

        legal = b == EMPTY
        logits = out.policy.float().masked_fill(~legal, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        # 整行都非法（棋盘已满）时 softmax 会出 NaN，压回 0。
        probs = torch.nan_to_num(probs, nan=0.0)
        value = InstinctNet.value_scalar(out.value)

        return (
            probs.to(torch.float32).cpu().numpy(),
            value.to(torch.float32).cpu().numpy(),
        )


class UniformEvaluator:
    """在合法点上均匀打分、价值恒为 0。

    用来在没有训练好的网络时验证搜索与规则的接线是否正确 ——
    此时 MCTS 退化为纯 rollout-free 的均匀先验搜索，仍然应当产出合法的完整对局。
    """

    def __init__(self, board_size: int) -> None:
        self.n = board_size * board_size

    def __call__(self, boards, to_move, history, move_number):
        legal = (boards == EMPTY).astype(np.float32)
        total = legal.sum(axis=1, keepdims=True)
        policy = np.divide(legal, np.maximum(total, 1.0)).astype(np.float32)
        value = np.zeros(boards.shape[0], dtype=np.float32)
        return policy, value


class RandomEvaluator:
    """在合法点上给随机先验、价值恒为 0。

    两个用途：竞技场里的随机基线；以及在测试中制造真正随机的落子 ——
    均匀先验配上 MCTS 会主动避开禁手点（禁手子节点是即时负，访问一次就被排除），
    要压到「走出禁手即判负」这条路径，就得让落子真的随机。
    """

    def __init__(self, board_size: int, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.n = board_size * board_size

    def __call__(self, boards, to_move, history, move_number):
        legal = boards == EMPTY
        scores = self.rng.random(boards.shape).astype(np.float32) * legal
        total = scores.sum(axis=1, keepdims=True)
        policy = np.divide(scores, np.maximum(total, 1e-9)).astype(np.float32)
        value = np.zeros(boards.shape[0], dtype=np.float32)
        return policy, value


class SelfPlayActor:
    """驱动一个 C++ runner 的自博弈 actor。"""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        board_size: int = 15,
        num_games: int = 1024,
        **runner_kwargs,
    ) -> None:
        core = load_core()
        self.runner = core.SelfPlayRunner(
            board_size=board_size, num_games=num_games, **runner_kwargs
        )
        self.evaluator = evaluator
        self.board_size = board_size
        self.num_games = num_games
        n = board_size * board_size

        # 预分配的主机侧缓冲区，避免每轮反复分配。
        self.boards = np.zeros((num_games, n), dtype=np.uint8)
        self.to_move = np.zeros(num_games, dtype=np.uint8)
        self.history = np.zeros((num_games, HISTORY), dtype=np.int32)
        self.move_number = np.zeros(num_games, dtype=np.int32)
        self.needs_eval = np.zeros(num_games, dtype=np.uint8)

    def step(self) -> None:
        """推进一轮：收集 -> 评估 -> 回填。"""
        self.runner.collect(
            self.boards, self.to_move, self.history, self.move_number, self.needs_eval
        )
        policy, value = self.evaluator(
            self.boards, self.to_move, self.history, self.move_number
        )
        policy = np.ascontiguousarray(policy, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)
        self.runner.apply(policy, value)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def drain(self, max_samples: int = 1 << 20) -> dict:
        return self.runner.drain(max_samples)

    @property
    def pending_samples(self) -> int:
        return self.runner.pending_samples

    @property
    def stats(self) -> dict:
        return self.runner.stats

    def reset_stats(self) -> None:
        self.runner.reset_stats()

    # ── 容灾 ────────────────────────────────────────────────────────────────
    def rng_state(self) -> list[str]:
        return self.runner.rng_state()

    def set_rng_state(self, state: list[str]) -> None:
        self.runner.set_rng_state(state)
