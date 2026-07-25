"""棋盘的八重二面体对称。

用途有两个：训练时做免费的数据增强，以及规则测试里的不变性检查 ——
规则本身在这 8 个变换下必须完全不变，这条性质与实现方式无关，
因此能独立地抓出「某个方向写错了」这类 bug。
"""

from __future__ import annotations

NUM_SYMMETRIES = 8


def transform_rc(r: int, c: int, size: int, t: int) -> tuple[int, int]:
    """把坐标按第 t 个对称变换映射过去。t 的低 2 位是旋转次数，第 3 位是转置。"""
    if t & 4:
        r, c = c, r
    for _ in range(t & 3):
        r, c = c, size - 1 - r
    return r, c


def transform_index(idx: int, size: int, t: int) -> int:
    r, c = divmod(idx, size)
    r, c = transform_rc(r, c, size, t)
    return r * size + c


def inverse(t: int) -> int:
    """返回 t 的逆变换。"""
    if t & 4:
        return t  # 含转置的都是对合变换
    return (-t) & 3


def transform_grid(grid, size: int, t: int) -> bytearray:
    out = bytearray(len(grid))
    for idx in range(size * size):
        out[transform_index(idx, size, t)] = grid[idx]
    return out


def index_map(size: int, t: int) -> list[int]:
    """预计算的下标映射：out[i] = transform_index(i, size, t)。"""
    return [transform_index(i, size, t) for i in range(size * size)]
