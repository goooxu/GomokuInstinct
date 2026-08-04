"""随机开局：竞技场与网页观战共用。

## 为什么需要随机开局

零搜索落子是 `argmax`、temperature = 0 —— **确定性函数**。同一个模型看到
同一个局面必然给出同一手，所以两个确定性 player 从空盘开打只有唯一一条棋路，
下一百局就是同一盘棋放一百遍。

后果在竞技场里表现得很隐蔽：胜负完全由先后手决定，而先后手是逐局轮换的，
于是得分率**恒等于 50%** —— 看起来像"势均力敌"，其实是测量退化
（第 11 章 #12）。在观战界面里表现得直白些：每局都一模一样。

## 落在哪

先在棋盘中央 `CENTER_REGION` 见方的区域里随机取一个 `OPENING_WINDOW`
见方的窗口，这一局的开局子**全落在同一个窗口内**。

一局只取**一个**窗口是关键：子散在中央区域各处的话，它们彼此之间没有任何关系，
双方接下来几十手各下各的，那不是一盘棋。

最早的实现是**全盘均匀取**，实际效果是两颗子被扔到棋盘的两个角上。
改成窗口之后，开局像个开局了。

> **注意：这个改动会让新旧对局数字不可比。** 换了开局分布就换了被测的东西 ——
> 从"任意局面下谁强"变成了"正常开局下谁强"。技术报告里 2026 年 8 月 4 日之前
> 记录的胜率、Elo 与等效模拟数，都是在全盘均匀开局下测的，
> 不要和之后的数字放进同一张表里比较。
"""

from __future__ import annotations

import random

# 中央区域边长，以及其中随机窗口的边长。棋盘小于这些值时自动收缩。
CENTER_REGION = 9
OPENING_WINDOW = 5


def opening_window(size: int, rng: random.Random) -> list[int]:
    """在中央区域里随机取一个窗口，返回窗口内所有点（按行优先的一维下标）。"""
    region = min(CENTER_REGION, size)
    window = min(OPENING_WINDOW, region)
    lo = (size - region) // 2          # 中央区域的起点
    hi = lo + region - window          # 窗口左上角的取值上界
    r0 = rng.randint(lo, hi)
    c0 = rng.randint(lo, hi)
    return [
        (r0 + dr) * size + (c0 + dc)
        for dr in range(window)
        for dc in range(window)
    ]


def opening_moves(size: int, plies: int, rng: random.Random) -> list[int]:
    """随机开局的着法序列：同一个窗口里取 `plies` 个互不相同的点。

    只保证"点互不相同、都在同一个窗口内"，不判断合法性 ——
    调用方都是在空盘上用它，任何点都是合法的。
    """
    if plies <= 0:
        return []
    pool = opening_window(size, rng)
    rng.shuffle(pool)
    return pool[:plies]
