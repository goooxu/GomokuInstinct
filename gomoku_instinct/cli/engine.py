"""零搜索的对战引擎。

**这里是本项目「对战时不使用搜索」这条约束的落地点**：

    落子 = 一次网络前向 -> 屏蔽已占点 -> argmax

没有树搜索、没有 rollout、没有开局库，也没有任何形式的多步推演。
`--safe-mode` 之外，连禁手点都不替模型屏蔽 —— 在严格 RIF 语义下禁手点是合法落子，
避开它是模型必须自己学会的能力，替它挡掉就等于在推理时偷偷加了规则外挂。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model import InstinctNet, encode
from ..model.features import NUM_HISTORY_PLANES
from ..rules import Game
from ..rules.constants import EMPTY


@dataclass
class MoveAnalysis:
    """一次落子决策的全部依据，供 --show-policy 展示。"""

    move: int
    value: float  # 行棋方视角的胜负估计，[-1, 1]
    top_moves: list[tuple[int, float]]  # (落点, 概率)，按概率降序
    forbidden_pred: list[float]  # 模型对每个点是否为黑方禁手的预测概率
    masked_forbidden: bool  # 是否因 safe-mode 屏蔽掉了禁手点


class InstinctPlayer:
    """用 InstinctNet 下棋。单次前向，argmax，不做任何搜索。"""

    def __init__(
        self,
        model: InstinctNet,
        board_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        safe_mode: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.size = board_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.safe_mode = safe_mode
        self.temperature = temperature

    @torch.inference_mode()
    def analyze(self, game: Game, top_k: int = 5) -> MoveAnalysis:
        size = self.size
        n = size * size

        boards = torch.tensor(
            [list(game.board.grid)], dtype=torch.uint8, device=self.device
        )
        to_move = torch.tensor([game.to_move], dtype=torch.uint8, device=self.device)

        moves = [m for m, _, _ in game.history]
        history = [
            moves[-1 - k] if k < len(moves) else -1 for k in range(NUM_HISTORY_PLANES)
        ]
        history_t = torch.tensor([history], dtype=torch.int64, device=self.device)
        move_number = torch.tensor(
            [len(moves)], dtype=torch.int64, device=self.device
        )

        planes = encode(boards, to_move, history_t, move_number, size, dtype=self.dtype)
        out = self.model(planes)

        legal = boards == EMPTY
        masked_forbidden = False
        if self.safe_mode:
            # 显式开启时才用规则给模型兜底；默认关闭，见模块开头的说明。
            forbidden = game.forbidden_map()
            fb = torch.tensor([forbidden], dtype=torch.bool, device=self.device)
            if bool((legal & ~fb).any()):
                legal = legal & ~fb
                masked_forbidden = True

        logits = out.policy.float().masked_fill(~legal, float("-inf"))
        probs = torch.softmax(logits, dim=-1)[0]

        if self.temperature > 1e-6:
            sharpened = torch.softmax(logits / self.temperature, dim=-1)[0]
            move = int(torch.multinomial(sharpened, 1).item())
        else:
            move = int(torch.argmax(logits, dim=-1).item())

        k = min(top_k, int(legal.sum().item()))
        top_probs, top_idx = torch.topk(probs, k)
        top_moves = [
            (int(i), float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())
        ]

        forbidden_pred = []
        if out.forbidden is not None:
            forbidden_pred = torch.sigmoid(out.forbidden.float()[0]).tolist()

        return MoveAnalysis(
            move=move,
            value=float(InstinctNet.value_scalar(out.value)[0]),
            top_moves=top_moves,
            forbidden_pred=forbidden_pred,
            masked_forbidden=masked_forbidden,
        )

    def choose(self, game: Game) -> int:
        return self.analyze(game, top_k=1).move

    @torch.inference_mode()
    def choose_batch(self, games: list[Game]) -> list[int]:
        """一次前向吃掉整批对局。竞技场用，选点方式与单局完全一致。"""
        size = self.size
        dev = self.device

        boards = torch.tensor(
            [list(g.board.grid) for g in games], dtype=torch.uint8, device=dev
        )
        to_move = torch.tensor(
            [g.to_move for g in games], dtype=torch.uint8, device=dev
        )
        history = torch.tensor(
            [
                [
                    g.history[-1 - k][0] if k < len(g.history) else -1
                    for k in range(NUM_HISTORY_PLANES)
                ]
                for g in games
            ],
            dtype=torch.int64,
            device=dev,
        )
        move_number = torch.tensor(
            [len(g.history) for g in games], dtype=torch.int64, device=dev
        )

        planes = encode(boards, to_move, history, move_number, size, dtype=self.dtype)
        out = self.model(planes, with_aux=False)

        legal = boards == EMPTY
        if self.safe_mode:
            fb = torch.tensor(
                [g.forbidden_map() for g in games], dtype=torch.bool, device=dev
            )
            keep = legal & ~fb
            # 只在还有别的合法点时才屏蔽，避免把自己逼到无处可下
            has_any = keep.any(dim=-1, keepdim=True)
            legal = torch.where(has_any, keep, legal)

        logits = out.policy.float().masked_fill(~legal, float("-inf"))
        return logits.argmax(dim=-1).tolist()
