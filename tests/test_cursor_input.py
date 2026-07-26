"""光标式落子的测试。

这块代码的失败方式很难靠肉眼发现：方向键是转义序列不是单字符，raw 模式下
Ctrl-C 不再是信号，退出时忘记还原终端属性会让用户的 shell 变哑。所以这里
开一个真正的 pty 来驱动整个循环，而不是打桩糊过去。

测试里有两个 pty 的坑，两个都表现为「无声死等」：
* 从端默认是行缓冲（canonical）模式，写进去的字节要等换行才对读端可见，
  所以必须先 setraw；
* `tty.setraw` 默认用 TCSAFLUSH，会丢弃已排队的输入 —— 被测代码进入 raw 模式时
  若也用默认值，预先写进去的按键会被冲掉。keyboard.raw_mode 因此显式用 TCSADRAIN。
"""

from __future__ import annotations

import os
import pty
import re
import sys
import termios
import tty

import pytest

from gomoku_instinct.cli import keyboard as kb
from gomoku_instinct.cli.cursor_play import run_cursor_game
from gomoku_instinct.cli.render import render_board
from gomoku_instinct.rules import BLACK, EMPTY, ForbiddenSemantics, Game, RenjuRules

ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")
SIZE = 15
CENTER = (SIZE // 2) * SIZE + SIZE // 2  # 光标起点：天元


class _FakeStdin:
    """把 sys.stdin 换成 pty 从端，read_key 只用到 fileno / isatty。"""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def isatty(self) -> bool:
        return True


class _StubPlayer:
    """固定应手的假 AI，测试里不碰模型。"""

    def __init__(self, replies) -> None:
        self.replies = list(replies)

    def analyze(self, game):
        result = type("R", (), {})()
        result.move = self.replies.pop(0)
        result.value = 0.0
        result.top_moves = [(result.move, 1.0)]
        result.forbidden_pred = None
        result.masked_forbidden = False
        return result


def _drive(keys: bytes, replies=(), game: Game | None = None):
    """在 pty 上跑一遍光标循环。

    返回（对局，屏幕输出，进入前的终端属性，退出后的终端属性）。
    """
    master, slave = pty.openpty()
    tty.setraw(slave)  # 见模块 docstring：不设 raw 读不出单个按键
    before = termios.tcgetattr(slave)
    os.write(master, keys)

    game = game or Game(SIZE, RenjuRules(), ForbiddenSemantics.LOSE)
    captured: list[str] = []
    real_stdin, real_stdout = sys.stdin, sys.stdout

    class _Cap:
        def write(self, text):
            captured.append(text)

        def flush(self):
            pass

        def isatty(self):
            return True

    sys.stdin, sys.stdout = _FakeStdin(slave), _Cap()
    try:
        run_cursor_game(game, _StubPlayer(replies), BLACK, {"step": 1}, False)
        after = termios.tcgetattr(slave)
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
        os.close(master)
        os.close(slave)
    return game, "".join(captured), before, after


def test_arrows_move_cursor_and_enter_places_stone():
    # 从天元出发：右、右、下
    expected = CENTER + 2 + SIZE
    game, _, _, _ = _drive(b"\x1b[C\x1b[C\x1b[B\rq", replies=[0])
    assert game.history[0][0] == expected


def test_vim_keys_work_like_arrows():
    expected = CENTER + 2 + SIZE
    game, _, _, _ = _drive(b"llj\rq", replies=[0])
    assert game.history[0][0] == expected


def test_cursor_clamps_at_edges():
    # 一路向上向左撞墙，应停在左上角，而不是绕回或越界
    game, _, _, _ = _drive(b"k" * 30 + b"h" * 30 + b"\rq", replies=[1])
    assert game.history[0][0] == 0


def test_ctrl_c_exits_and_restores_terminal():
    game, _, before, after = _drive(b"\x03")
    assert not game.history
    assert after == before  # raw 模式必须原样还原，否则用户的 shell 会变哑


def test_raw_mode_restores_a_cooked_terminal():
    """真实场景是从 cooked 进 raw：退出后回显和行编辑都要回来。"""
    master, slave = pty.openpty()
    real = sys.stdin
    sys.stdin = _FakeStdin(slave)
    try:
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ICANON  # 前提：确实是 cooked
        with kb.raw_mode():
            assert not termios.tcgetattr(slave)[3] & termios.ICANON
        after = termios.tcgetattr(slave)
        assert after[3] & termios.ICANON
        assert after[3] & termios.ECHO
    finally:
        sys.stdin = real
        os.close(master)
        os.close(slave)


def test_occupied_point_is_rejected_not_played():
    game = Game(SIZE, RenjuRules(), ForbiddenSemantics.LOSE)
    game.play(CENTER)
    game.play(CENTER + SIZE)
    # 光标停在最后一手上，直接回车应被拒绝而不是把棋下坏
    game, screen, _, _ = _drive(b"\rq", game=game)
    assert len(game.history) == 2
    assert "已经有子" in screen


def test_forbidden_point_is_flagged_before_committing():
    """禁手点必须在落子之前就提示出来 —— 事后才知道就没意义了。"""
    game, screen, _, _ = _drive(b"q")
    assert "光标" in screen and "空点" in screen


def test_cursor_highlight_does_not_shift_columns():
    """反显不能改变可见宽度 —— 一旦错列，"看错位置"这个问题反而更严重。"""
    grid = [EMPTY] * (SIZE * SIZE)
    plain = render_board(grid, SIZE)
    marked = render_board(grid, SIZE, cursor=CENTER)
    assert ANSI.sub("", marked) == plain
    assert "\033[7m" in marked


def test_lone_escape_does_not_block():
    """单按 Esc 不能把程序卡死在等后续字节上。"""
    master, slave = pty.openpty()
    tty.setraw(slave)
    real = sys.stdin
    sys.stdin = _FakeStdin(slave)
    try:
        os.write(master, b"\x1b")
        assert kb.read_key(timeout=0.05) == kb.ESCAPE
    finally:
        sys.stdin = real
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("key,expected", [
    (b"\x1b[A", kb.UP), (b"\x1b[B", kb.DOWN),
    (b"\x1b[C", kb.RIGHT), (b"\x1b[D", kb.LEFT),
    (b"\r", kb.ENTER), (b"\n", kb.ENTER), (b"\x03", kb.INTERRUPT),
    (b"k", kb.UP), (b"j", kb.DOWN), (b"h", kb.LEFT), (b"l", kb.RIGHT),
    (b" ", " "), (b"u", "u"),
])
def test_read_key_normalisation(key, expected):
    master, slave = pty.openpty()
    tty.setraw(slave)
    real = sys.stdin
    sys.stdin = _FakeStdin(slave)
    try:
        os.write(master, key)
        assert kb.read_key() == expected
    finally:
        sys.stdin = real
        os.close(master)
        os.close(slave)


def test_pipe_falls_back_to_text_mode(monkeypatch):
    """管道输入下必须退回坐标输入，否则脚本化用法会直接坏掉。"""
    with open(os.devnull) as devnull:
        monkeypatch.setattr(sys, "stdin", devnull)
        assert not kb.supports_cursor_mode()
