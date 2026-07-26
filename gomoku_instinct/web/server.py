"""Web 版人机对战。

只用标准库的 `http.server`。这个项目至今没有 torch 之外的运行时依赖，为一个
试玩页面引入 Web 框架不划算，容器里也未必装得上。

**落子决策与 CLI 完全共用 `InstinctPlayer`**，仍然是「一次前向 → 屏蔽已占点 →
argmax」，没有搜索。换的只是界面，棋力与 `play` 子命令逐手一致。

默认只监听回环地址：这是个本地试玩工具，没有任何认证，而开发机往往是共享的。
要从别的机器访问，用 SSH 端口转发，或显式指定 `--host 0.0.0.0` 自行承担。
"""

from __future__ import annotations

import functools
import json
import os
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..cli.engine import InstinctPlayer
from ..cli.render import move_to_label
from ..model.loader import load_model
from ..rules import BLACK, WHITE, ForbiddenSemantics, Game, Outcome, RenjuRules

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 会话上限。超出后淘汰最旧的一局 —— 本地试玩工具，不做持久化。
MAX_SESSIONS = 64
MAX_BODY = 1 << 20

OUTCOME_NAMES = {
    Outcome.ONGOING: "ongoing",
    Outcome.BLACK_WIN: "black_win",
    Outcome.WHITE_WIN: "white_win",
    Outcome.DRAW: "draw",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Session:
    """一局棋。analysis 连同它是站在哪一方视角上算的一起存。

    只存 value 不存视角的话，提示（轮到人时算）与 AI 自评（轮到 AI 时算）会
    混在一起，正负号的含义正好相反，界面上没法解释。
    """

    def __init__(self, size: int, rules: RenjuRules, human: int) -> None:
        self.game = Game(size, rules, ForbiddenSemantics.LOSE)
        self.human = human
        self.analysis = None
        self.analysis_color = None
        self.resigned: int | None = None


class App:
    """服务端的全部可变状态。模型推理与会话改动都在同一把锁下进行。"""

    def __init__(self, player: InstinctPlayer, meta: dict, size: int) -> None:
        self.player = player
        self.meta = meta
        self.size = size
        self.rules = RenjuRules()
        self.lock = threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()

    def new_session(self, human: int) -> tuple[str, Session]:
        while len(self.sessions) >= MAX_SESSIONS:
            self.sessions.popitem(last=False)
        sid = secrets.token_urlsafe(12)
        session = Session(self.size, self.rules, human)
        self.sessions[sid] = session
        return sid, session

    def get(self, sid: str) -> Session | None:
        session = self.sessions.get(sid)
        if session is not None:
            self.sessions.move_to_end(sid)
        return session

    def ai_move(self, session: Session) -> None:
        """让 AI 走一手。调用方必须已持有锁。"""
        game = session.game
        if game.is_terminal() or session.resigned is not None:
            return
        analysis = self.player.analyze(game)
        session.analysis_color = game.to_move
        session.analysis = analysis
        game.play(analysis.move)

    def state(self, sid: str, session: Session) -> dict:
        game = session.game
        size = self.size

        # 禁手点只在轮到执黑的人时给 —— 与 CLI 一致。执白的人用不上，
        # 而在 AI 执黑时把它的禁手点摊开，等于替它把风险标出来了。
        forbidden = None
        if session.resigned is None and game.to_move == session.human == BLACK:
            if not game.is_terminal():
                forbidden = game.forbidden_map()

        analysis = None
        if session.analysis is not None:
            analysis = {
                "move": session.analysis.move,
                "value": session.analysis.value,
                "value_color": session.analysis_color,
                "top": [
                    {"move": m, "prob": p, "label": move_to_label(m, size)}
                    for m, p in session.analysis.top_moves
                ],
            }

        outcome = OUTCOME_NAMES[game.outcome]
        if session.resigned is not None:
            outcome = "white_win" if session.resigned == BLACK else "black_win"

        return {
            "sid": sid,
            "size": size,
            "colors": {"black": BLACK, "white": WHITE},
            "human": session.human,
            "to_move": game.to_move,
            "grid": list(game.board.grid),
            "last_move": game.last_move(),
            "outcome": outcome,
            "resigned": session.resigned,
            "forbidden": forbidden,
            "analysis": analysis,
            "step": self.meta["step"],
            "history": [
                {
                    "move": m,
                    "color": c,
                    "label": move_to_label(m, size),
                    "forbidden": int(j.forbidden) if j.is_forbidden else 0,
                }
                for m, c, j in game.history
            ],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "gomoku-instinct"
    protocol_version = "HTTP/1.1"

    def __init__(self, app: App, *args, **kwargs) -> None:
        self.app = app
        super().__init__(*args, **kwargs)

    # 默认的访问日志会把每一次落子都刷成一行，试玩时纯属噪音
    def log_message(self, fmt, *args) -> None:  # noqa: A003
        pass

    # ── 响应 ────────────────────────────────────────────────────────────
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # 页面是随代码一起改的，缓存住会让人以为改动没生效
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _static(self, name: str) -> None:
        path = os.path.normpath(os.path.join(STATIC_DIR, name.lstrip("/")))
        # 目录穿越防护：本地工具也不该能读到工作目录之外
        if not path.startswith(STATIC_DIR + os.sep) or not os.path.isfile(path):
            self._json({"error": "not found"}, 404)
            return
        with open(path, "rb") as fh:
            body = fh.read()
        ext = os.path.splitext(path)[1]
        self._send(body, CONTENT_TYPES.get(ext, "application/octet-stream"))

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _session(self, payload: dict) -> Session | None:
        session = self.app.get(str(payload.get("sid", "")))
        if session is None:
            # 410：会话没了（服务重启或被淘汰），前端据此自动开新局
            self._json({"error": "对局已失效，请开新局"}, 410)
            return None
        return session

    # ── 路由 ────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):])
        elif path == "/api/state":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            sid = ""
            for part in query.split("&"):
                if part.startswith("sid="):
                    sid = part[4:]
            with self.app.lock:
                session = self._session({"sid": sid})
                if session is not None:
                    self._json(self.app.state(sid, session))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        handlers = {
            "/api/new": self._api_new,
            "/api/move": self._api_move,
            "/api/undo": self._api_undo,
            "/api/hint": self._api_hint,
            "/api/resign": self._api_resign,
        }
        handler = handlers.get(path)
        if handler is None:
            self._json({"error": "not found"}, 404)
            return
        payload = self._body()
        with self.app.lock:
            handler(payload)

    # ── 接口 ────────────────────────────────────────────────────────────
    def _api_new(self, payload: dict) -> None:
        human = WHITE if str(payload.get("color", "black")).startswith("w") else BLACK
        sid, session = self.app.new_session(human)
        if human == WHITE:
            self.app.ai_move(session)  # AI 执黑先行
        self._json(self.app.state(sid, session))

    def _api_move(self, payload: dict) -> None:
        session = self._session(payload)
        if session is None:
            return
        sid = payload["sid"]
        game = session.game
        move = payload.get("move")

        if not isinstance(move, int) or not 0 <= move < self.app.size ** 2:
            self._json({"error": "落点不合法"}, 400)
            return
        if session.resigned is not None or game.is_terminal():
            self._json({"error": "这局已经结束了"}, 409)
            return
        if game.to_move != session.human:
            self._json({"error": "还没轮到你"}, 409)
            return
        if not game.board.is_empty(*divmod(move, self.app.size)):
            self._json({"error": "那里已经有子了"}, 409)
            return

        session.analysis = None
        session.analysis_color = None
        game.play(move)
        self.app.ai_move(session)
        self._json(self.app.state(sid, session))

    def _api_undo(self, payload: dict) -> None:
        session = self._session(payload)
        if session is None:
            return
        game = session.game
        # 退回到「又轮到人」的局面：连同 AI 的应手一起退
        for _ in range(2):
            if game.history:
                game.undo()
        session.resigned = None
        session.analysis = None
        session.analysis_color = None
        # 人执白时，退两手会退成轮到 AI，补一手回来
        if not game.is_terminal() and game.to_move != session.human:
            self.app.ai_move(session)
        self._json(self.app.state(payload["sid"], session))

    def _api_hint(self, payload: dict) -> None:
        session = self._session(payload)
        if session is None:
            return
        game = session.game
        if game.is_terminal() or session.resigned is not None:
            self._json({"error": "这局已经结束了"}, 409)
            return
        session.analysis_color = game.to_move
        session.analysis = self.app.player.analyze(game)
        self._json(self.app.state(payload["sid"], session))

    def _api_resign(self, payload: dict) -> None:
        session = self._session(payload)
        if session is None:
            return
        session.resigned = session.human
        self._json(self.app.state(payload["sid"], session))


def build_app(
    checkpoint: str,
    device: str = "cpu",
    safe_mode: bool = False,
    temperature: float = 0.0,
) -> App:
    model, meta = load_model(checkpoint, device)
    size = meta["board_size"]
    player = InstinctPlayer(
        model, size, device, safe_mode=safe_mode, temperature=temperature
    )
    return App(player, meta, size)


def serve(
    checkpoint: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cpu",
    safe_mode: bool = False,
    temperature: float = 0.0,
) -> int:
    app = build_app(checkpoint, device, safe_mode, temperature)
    httpd = ThreadingHTTPServer((host, port), functools.partial(Handler, app))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "") else host
    print(f"权重 step {app.meta['step']:,}   棋盘 {app.size}x{app.size}   设备 {device}")
    print("AI 落子完全由一次网络前向决定，不使用任何搜索。")
    if host == "0.0.0.0":
        print("警告：监听了所有网卡，同网段的人都能打开这个页面（无认证）。")
    print(f"\n打开 http://{shown}:{httpd.server_address[1]}/   Ctrl-C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n再见。")
    finally:
        httpd.server_close()
    return 0
