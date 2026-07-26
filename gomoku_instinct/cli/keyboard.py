"""终端按键读取，用于光标式落子。

15x15 的棋盘上靠肉眼数格子报坐标很容易错行错列，所以提供一个光标模式：
方向键移动、回车落子，当前坐标实时显示。

要按键即响应就得把终端切到 raw 模式（否则要等回车），这带来两个必须处理的细节：

* **方向键不是单个字符**，而是 `ESC [ A/B/C/D` 这样的转义序列。读到 ESC 之后要
  再看有没有后续字节 —— 但不能直接阻塞读，否则用户单按一下 Esc 会把程序卡住。
  这里用 select 给一个很短的超时来区分「转义序列」与「孤立的 Esc」。
* **raw 模式下 Ctrl-C 不再产生 SIGINT**，而是变成一个普通字节 0x03。
  不显式处理的话用户就退不出去了。
"""

from __future__ import annotations

import os
import select
import sys
from contextlib import contextmanager

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # 非 POSIX 终端
    _HAS_TERMIOS = False

# 归一化之后的按键名
UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
ENTER, ESCAPE, INTERRUPT = "enter", "escape", "interrupt"

_ARROWS = {"A": UP, "B": DOWN, "C": RIGHT, "D": LEFT}
# vim 手法，以及部分终端下方向键失灵时的退路
_VIM = {"k": UP, "j": DOWN, "h": LEFT, "l": RIGHT}

# 进入 raw 模式前的终端属性。read_line 需要它来临时切回可编辑的行输入 ——
# 在 raw 模式下现读当前属性再"还原"，还原到的还是 raw 模式。
_COOKED_ATTRS = None


def supports_cursor_mode() -> bool:
    """能不能用光标模式。管道输入、非 POSIX、无终端时都不行。"""
    return _HAS_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()


@contextmanager
def raw_mode():
    """把终端切到 raw 模式，退出时**一定**还原。

    不还原的话用户的 shell 会变得没有回显、没有行编辑，只能靠 reset 救回来。
    """
    if not _HAS_TERMIOS:
        yield
        return
    global _COOKED_ATTRS
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    _COOKED_ATTRS = saved
    try:
        # 必须显式指定 TCSADRAIN：tty.setraw 默认是 TCSAFLUSH，会把已经排队的
        # 输入直接丢掉 —— 进入前敲的键会凭空消失，且没有任何报错。
        tty.setraw(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        _COOKED_ATTRS = None


def read_key(timeout: float = 0.05) -> str:
    """读一个按键，返回归一化后的名字或原始字符。

    timeout 只用于判定 ESC 之后是否还有后续字节，不影响正常按键的响应速度。
    """
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1).decode("utf-8", "replace")

    if ch == "\x03":
        return INTERRUPT
    if ch in ("\r", "\n"):
        return ENTER
    if ch == "\x1b":
        # 可能是方向键的转义序列，也可能是用户单按了 Esc
        if not select.select([fd], [], [], timeout)[0]:
            return ESCAPE
        rest = os.read(fd, 2).decode("utf-8", "replace")
        if rest.startswith("[") and len(rest) > 1 and rest[1] in _ARROWS:
            return _ARROWS[rest[1]]
        return ESCAPE
    if ch in _VIM:
        return _VIM[ch]
    return ch


def read_line(prompt: str) -> str:
    """在光标模式中途需要输入一行文本（比如存档文件名）时用。

    临时退出 raw 模式，让用户能正常退格、粘贴。
    """
    if not _HAS_TERMIOS or _COOKED_ATTRS is None:
        return input(prompt)
    fd = sys.stdin.fileno()
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, _COOKED_ATTRS)
        sys.stdout.write("\r\n" + prompt)
        sys.stdout.flush()
        return sys.stdin.readline().strip()
    finally:
        tty.setraw(fd)
