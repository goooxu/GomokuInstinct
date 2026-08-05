"""带搜索的对战引擎 —— **这个模块不属于项目的核心命题。**

第 1 章那五条硬约束里的第二条是「零搜索推理」，落地点在 [`engine.py`](engine.py)，
技术报告里所有棋力数字都是那条路径测出来的。

这个模块提供的是另一回事：**同一份权重配上 MCTS**。它存在的唯一理由是实用 ——
要拿这个模型去真的赢棋时，搜索是最大的一笔现成收益（实测零搜索打自己的
64 次模拟版本只有 23.8% 得分率，也就是搜索还值 +203 Elo）。

两条纪律：

1. **默认关闭。** `serve --sims 0` 是默认值，不给参数就是零搜索。
2. **开着的时候必须看得出来。** 启动横幅、网页顶栏都会标明当前模拟数 ——
   否则拿一个开着搜索的服务去测棋力，就是第 11 章那类静默失败的完美温床：
   数字变好看了，而没人知道为什么。

选点方式与 `InstinctPlayer` 的区别只有一处：那边是策略头 argmax，
这边是**根节点访问数** argmax。其余（输入编码、禁手语义、确定性）完全一致。
"""

from __future__ import annotations

import numpy as np
import torch

from ..eval.mcts_player import MctsPlayer
from ..model import InstinctNet
from ..rules import Game
from ..selfplay import ModelEvaluator
from .engine import MoveAnalysis


class SearchPlayer:
    """用 MCTS 下棋。接口与 `InstinctPlayer` 对齐，可以直接互换。"""

    def __init__(
        self,
        model: InstinctNet,
        board_size: int,
        device: torch.device | str = "cpu",
        sims: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        # **1 是最优值，不是省事的默认值。** 搜索线程池并行的维度是「槽位」——
        # 同时搜多少个不同局面 —— 而对战只有一个局面，所以多开的线程一个都用不上，
        # 只剩唤醒/休眠开销。实测 256 次模拟：1 线程 344ms、16 线程 370ms、
        # 64 线程 377ms，**越多越慢**。
        # 一轮内部的多次下潜也没法并行：每次都要看上一次留下的虚拟败绩。
        threads: int = 1,
        leaves: int = 16,
    ) -> None:
        if sims <= 0:
            raise ValueError("sims 必须为正；零搜索请直接用 InstinctPlayer")
        self.model = model
        self.size = board_size
        self.sims = sims
        # 网页版每次只搜一个局面，槽位开 1 就够。
        #
        # 关键是 `leaves`：一轮从同一棵树里取多个叶子凑成一批（virtual loss）。
        # 不这么做的话每轮只产一个叶子，GPU 每次只算一个 15×15 的输入 ——
        # 实测 CPU 只占 0.7%，其余全耗在 kernel 启动开销上，利用率约 1.7%。
        # 开到 16 之后同样的墙钟能跑约 12 倍的模拟数（拐点在 8~16，再大不涨）。
        #
        # **评测侧仍然用 leaves=1**：批量收集是近似，竞技场与 search_gap 的口径
        # 一个字都不能变，报告里的数字要保持可复算。
        # **钳住 leaves，保证轮数够。** 一轮同时取 N 个叶子，就有 N 条路径是
        # 靠虚拟败绩硬岔开的；轮数太少（sims/leaves 太小）搜索来不及收敛，
        # 直接退化成宽度优先。实测 sims=64、leaves=16（只有 4 轮）时根节点访问
        # 从 55:1:1:1 摊成 13:10:9:8 —— 快了，但搜的东西不一样了。
        #
        # 至少留 8 轮。网页那几个低档位（sims=16/32）不钳的话会整档失效，
        # 而**这种失效不报错，只是棋力悄悄变差**。
        self.leaves = max(1, min(leaves, sims // 8))
        leaves = self.leaves
        self.player = MctsPlayer(
            ModelEvaluator(model, board_size, device, dtype=dtype),
            board_size,
            sims=sims,
            slots=1,
            threads=threads,
            leaves=leaves,
        )

    def analyze(self, game: Game, top_k: int = 5) -> MoveAnalysis:
        searcher = self.player.searcher
        self.player.choose_batch([game])  # 跑完这一次搜索，结果留在树里

        visits = np.asarray(searcher.visit_counts()[0], dtype=np.float64)
        total = float(visits.sum())
        value = float(np.asarray(searcher.root_values())[0])

        if total <= 0:
            # 根节点一次都没展开（例如已是终局）。退回第一个合法点，
            # 并如实报告"没有分布可看" —— 不要伪造一个看起来正常的输出。
            legal = [i for i, v in enumerate(game.board.grid) if v == 0]
            move = legal[0] if legal else 0
            return MoveAnalysis(
                move=move, move_prob=0.0, value=value,
                top_moves=[], forbidden_pred=[], masked_forbidden=False,
            )

        probs = visits / total
        move = int(visits.argmax())
        k = min(top_k, int((visits > 0).sum()))
        order = np.argsort(-visits)[:k]
        return MoveAnalysis(
            move=move,
            move_prob=float(probs[move]),
            value=value,
            top_moves=[(int(i), float(probs[i])) for i in order],
            # 禁手预测头只用于显示，网页版根本没发到前端（第 10 章）。
            forbidden_pred=[],
            # 禁手点在树里是即时负，搜索一访问就把它排除了，不需要规则兜底。
            masked_forbidden=False,
        )

    def choose(self, game: Game) -> int:
        return self.analyze(game, top_k=1).move

    def choose_batch(self, games: list[Game]) -> list[int]:
        """竞技场用。**注意这里的槽位只有 1 个**，整批会被切成一局一局串行搜，
        比 `MctsPlayer` 直接用多槽位慢得多 —— 要批量评测请直接用后者。"""
        return self.player.choose_batch(games)
