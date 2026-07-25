"""差分测试与规则压测用的局面生成器。

生成的全是**随机局面**，与人类棋谱无关；它们只作为规则判定的输入，
不进入训练或测试数据。

四种分布各有侧重：

  scatter       全盘均匀撒子 —— 覆盖稀疏、互不相干的棋型
  cluster       围绕一个中心撒子 —— 制造密集交叉的棋型
  black_heavy   黑子占多数的密集局面 —— 禁手只对黑方成立，这样才能真正压到
                四四/三三/长连的判定路径；均匀随机局面里禁手点极其罕见
  selfplay      随机合法对局中途的局面 —— 分布最接近真实对局
"""

from __future__ import annotations

import random

from .constants import BLACK, EMPTY, WHITE


def scatter(rng: random.Random, size: int, n_stones: int) -> bytearray:
    """全盘均匀撒子，黑白交替。"""
    grid = bytearray(size * size)
    n_stones = min(n_stones, size * size)
    for i, cell in enumerate(rng.sample(range(size * size), n_stones)):
        grid[cell] = BLACK if i % 2 == 0 else WHITE
    return grid


def cluster(
    rng: random.Random, size: int, n_stones: int, radius: int = 4
) -> bytearray:
    """围绕随机中心撒子。"""
    return _cluster(rng, size, n_stones, radius, black_ratio=0.5)


def black_heavy(
    rng: random.Random, size: int, n_stones: int, radius: int = 4
) -> bytearray:
    """黑子占多数的密集局面，用来把禁手判定压满。"""
    return _cluster(rng, size, n_stones, radius, black_ratio=0.78)


def _cluster(
    rng: random.Random, size: int, n_stones: int, radius: int, black_ratio: float
) -> bytearray:
    grid = bytearray(size * size)
    cr = rng.randrange(size)
    cc = rng.randrange(size)
    placed = 0
    for _ in range(n_stones * 40):
        if placed >= n_stones:
            break
        r = cr + rng.randint(-radius, radius)
        c = cc + rng.randint(-radius, radius)
        if not (0 <= r < size and 0 <= c < size):
            continue
        idx = r * size + c
        if grid[idx] != EMPTY:
            continue
        grid[idx] = BLACK if rng.random() < black_ratio else WHITE
        placed += 1
    return grid


def selfplay_positions(
    rng: random.Random,
    size: int,
    rules,
    max_positions: int,
    sample_prob: float = 0.35,
) -> list[bytearray]:
    """随机合法对局，沿途采样局面。

    rules 是 C++ 侧的 Rules 对象（生成局面只图快；局面本身不偏袒任何一侧实现，
    两份实现拿到的是同一批输入）。
    """
    out: list[bytearray] = []
    grid = bytearray(size * size)
    color = BLACK
    empties = list(range(size * size))
    rng.shuffle(empties)

    while out_needed(out, max_positions):
        if not empties:
            grid = bytearray(size * size)
            color = BLACK
            empties = list(range(size * size))
            rng.shuffle(empties)
            continue

        move = empties.pop()
        if grid[move] != EMPTY:
            continue

        outcome, forbidden, _, _, _ = rules.judge(grid, size, move, color)
        grid[move] = color

        if rng.random() < sample_prob:
            out.append(bytearray(grid))

        if outcome != 0:  # 对局结束，另起一局
            grid = bytearray(size * size)
            color = BLACK
            empties = list(range(size * size))
            rng.shuffle(empties)
            continue

        color = WHITE if color == BLACK else BLACK

    return out[:max_positions]


def out_needed(out: list, target: int) -> bool:
    return len(out) < target


GENERATORS = {
    "scatter": lambda rng, size: scatter(rng, size, rng.randint(4, size * size // 3)),
    "cluster": lambda rng, size: cluster(rng, size, rng.randint(6, 40)),
    "black_heavy": lambda rng, size: black_heavy(rng, size, rng.randint(6, 34)),
}


def random_position(rng: random.Random, size: int, kind: str = "mixed") -> bytearray:
    """按名字取一个生成器产出局面；kind='mixed' 时随机挑一种。"""
    if kind == "mixed":
        kind = rng.choice(list(GENERATORS))
    if kind not in GENERATORS:
        raise ValueError(f"未知的局面生成器: {kind}")
    return GENERATORS[kind](rng, size)
