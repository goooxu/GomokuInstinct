"""命令行工具：人机对战、竞技场、checkpoint 查看。"""

from .engine import InstinctPlayer, MoveAnalysis
from .render import label_to_move, move_to_label, render_board

__all__ = [
    "InstinctPlayer",
    "MoveAnalysis",
    "label_to_move",
    "move_to_label",
    "render_board",
]
