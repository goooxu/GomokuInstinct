"""输入特征编码。

刻意只放**规则直接导出**的信息：棋子分布、行棋方、最近数手、手数、位置。
活三/冲四之类的战术棋型一律不作为输入 —— 它们改为辅助监督目标，
逼网络自己长出威胁识别能力。对战时没有搜索可用，这种「自己看出来」的能力
正是棋力的来源；喂现成的战术特征反而会让网络偷懒。

10 个平面：

    0  黑子
    1  白子
    2  空点
    3  轮到黑走（常数平面）—— 连珠规则不对称，禁手只约束黑方，必须告诉网络
    4  最近一手落点
    5  次近一手落点
    6  第三近一手落点
    7  第四近一手落点
    8  手数 / 棋盘格数（常数平面）
    9  到棋盘中心的归一化切比雪夫距离（常数平面）

这些平面在棋盘的八重二面体对称下全部同变：棋子与落点平面随坐标一起变换，
三个常数平面本身就对称。因此 8 重对称数据增强是严格合法的。

用固定的黑/白平面而非「己方/对方」平面，是因为连珠的禁手只约束黑方；
固定颜色能让禁手模式直接挂在平面 0 上，不必再由网络按行棋方做条件判断。
"""

from __future__ import annotations

import torch

from ..rules.constants import BLACK, EMPTY, WHITE

NUM_HISTORY_PLANES = 4
NUM_PLANES = 6 + NUM_HISTORY_PLANES

PLANE_BLACK = 0
PLANE_WHITE = 1
PLANE_EMPTY = 2
PLANE_SIDE_IS_BLACK = 3
PLANE_HISTORY_FIRST = 4
PLANE_MOVE_NUMBER = PLANE_HISTORY_FIRST + NUM_HISTORY_PLANES
PLANE_CENTER_DIST = PLANE_MOVE_NUMBER + 1

_CENTER_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def center_distance_plane(
    size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """到中心的归一化切比雪夫距离，(1, size, size)。

    卷积本身是平移等变的，这个平面把「离边界多远」显式告诉网络。
    边界在连珠里是实打实的信息：贴边的方向根本连不成五。
    该平面在八重对称下不变，不影响数据增强。
    """
    key = (size, device, dtype)
    cached = _CENTER_CACHE.get(key)
    if cached is not None:
        return cached

    coords = torch.arange(size, device=device, dtype=torch.float32)
    center = (size - 1) / 2.0
    d = (coords - center).abs()
    plane = torch.maximum(d[:, None], d[None, :]) / max(center, 1.0)
    plane = plane.to(dtype).unsqueeze(0)
    _CENTER_CACHE[key] = plane
    return plane


def encode(
    boards: torch.Tensor,
    to_move: torch.Tensor,
    history: torch.Tensor,
    move_number: torch.Tensor,
    size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """把局面批量编码成网络输入。

    参数
      boards       (B, N) uint8，N = size*size，取值 EMPTY/BLACK/WHITE
      to_move      (B,)   uint8，BLACK 或 WHITE
      history      (B, 4) int64，最近四手的落点下标，无则填 -1，下标 0 为最近一手
      move_number  (B,)   int64，已落子手数

    返回 (B, NUM_PLANES, size, size)
    """
    if boards.dim() != 2:
        raise ValueError(f"boards 形状应为 (B, N)，实际 {tuple(boards.shape)}")
    n = size * size
    if boards.shape[1] != n:
        raise ValueError(f"boards 第二维应为 {n}，实际 {boards.shape[1]}")

    device = boards.device
    batch = boards.shape[0]
    out = torch.zeros((batch, NUM_PLANES, n), device=device, dtype=dtype)

    out[:, PLANE_BLACK] = (boards == BLACK).to(dtype)
    out[:, PLANE_WHITE] = (boards == WHITE).to(dtype)
    out[:, PLANE_EMPTY] = (boards == EMPTY).to(dtype)

    side_is_black = (to_move == BLACK).to(dtype).view(batch, 1)
    out[:, PLANE_SIDE_IS_BLACK] = side_is_black.expand(batch, n)

    # 最近若干手：越界（-1）的位置不写入。
    hist = history[:, :NUM_HISTORY_PLANES].to(torch.int64)
    valid = hist >= 0
    safe = hist.clamp(min=0)
    for k in range(min(NUM_HISTORY_PLANES, hist.shape[1])):
        plane = out[:, PLANE_HISTORY_FIRST + k]
        plane.scatter_(1, safe[:, k : k + 1], valid[:, k : k + 1].to(dtype))

    out[:, PLANE_MOVE_NUMBER] = (move_number.to(torch.float32) / n).to(dtype).view(
        batch, 1
    ).expand(batch, n)

    out = out.view(batch, NUM_PLANES, size, size)
    out[:, PLANE_CENTER_DIST] = center_distance_plane(size, device, dtype)
    return out


def legal_mask(boards: torch.Tensor) -> torch.Tensor:
    """合法落子掩码：只排除已占点。

    禁手点在严格 RIF 语义下仍然是合法落子，因此**不**在这里屏蔽 ——
    避开禁手是模型必须自己学会的能力。
    """
    return boards == EMPTY
