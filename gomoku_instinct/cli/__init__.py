"""命令行工具：网页版对战入口、竞技场、checkpoint 查看。

`InstinctPlayer` 是零搜索部署的落地点，网页版与竞技场共用它（第 10 章）。
"""

from .engine import InstinctPlayer, MoveAnalysis
from .render import move_to_label, render_board

__all__ = [
    "InstinctPlayer",
    "MoveAnalysis",
    "move_to_label",
    "render_board",
]
