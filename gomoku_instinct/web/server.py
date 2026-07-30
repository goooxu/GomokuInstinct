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
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ..cli.engine import InstinctPlayer
from ..cli.render import move_to_label
from ..model.loader import load_model, resolve_checkpoint
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

# 静态文件缓存：path -> (mtime_ns, bytes)
_STATIC_CACHE: dict[str, tuple[int, bytes]] = {}

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
    """服务端的全部可变状态。会话改动在锁下进行，推理另有一条专属线程。"""

    def __init__(
        self,
        player: InstinctPlayer,
        meta: dict,
        size: int,
        sources: list[dict] | None = None,
        device: str = "cpu",
    ) -> None:
        self.player = player
        self.meta = meta
        self.size = size
        # 可在页面上切换的模型。每项 {"name": 显示名, "path": --ckpt 给的原始路径}
        self.sources = list(sources or [])
        self.active = 0
        self.device = device
        self.rules = RenjuRules()
        self.lock = threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        # 所有推理都赶到同一个长期存活的线程上跑。
        #
        # ThreadingHTTPServer 每个连接开一条新线程，而 cuBLAS/cuDNN 的句柄是**按线程**
        # 建的：在新线程上做第一次推理要重建句柄。实测同一次前向，主线程 11ms，
        # 每请求新线程 800ms —— 七十多倍，而且不报任何错，只是"点一下要等一秒"。
        # 固定一条预热过的线程之后回到 11ms。
        self._infer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gi-infer")

    def analyze(self, game: Game):
        return self._infer.submit(self.player.analyze, game).result()

    def warmup(self) -> None:
        """先在推理线程上跑一次空盘，把句柄建好，免得第一手卡一下。"""
        self.analyze(Game(self.size, self.rules, ForbiddenSemantics.LOSE))

    # ── 热加载 ──────────────────────────────────────────────────────────────
    def _reload_once(self, source: str) -> None:
        """把 run 目录里最新的权重换进来。只在推理线程上跑。"""
        path = resolve_checkpoint(source)
        if path == self.meta.get("path"):
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        # 只换权重、不换结构。结构真的变了 load_state_dict 会抛，
        # 由外层记下来并保留旧权重继续服务 —— 好过悄悄换上一个对不上的网络。
        self.player.model.load_state_dict(state["model"])
        # 整个 dict 一次性换掉：state() 读 meta 时没有加锁，重新绑定属性是原子的，
        # 逐个字段改则可能被读到一半新一半旧。
        self.meta = dict(
            self.meta,
            path=path,
            step=state.get("step", 0),
            cycle=state.get("cycle", 0),
        )
        print(f"[serve] 热加载 step {self.meta['step']:,}"
              f"（{os.path.basename(path)}）", flush=True)

    def active_source(self) -> str | None:
        if not self.sources:
            return None
        return self.sources[self.active]["path"]

    def should_follow(self) -> bool:
        """当前选中的这个源要不要跟着训练更新。

        只有 run 目录才跟；直接指到某个 .pt 文件是"我就要这一版"的意思。
        """
        source = self.active_source()
        return source is not None and os.path.isdir(source)

    def start_watching(self, interval: float) -> None:
        """盯着当前选中的 run 目录，训练每落一个新 checkpoint 就换上。

        只对 run 目录生效；直接指到某个 .pt 文件时不监视（那是"我就要这一版"的意思）。
        重新载入必须**排到推理线程上**执行，否则会和正在进行的一次前向抢同一个模型。

        每一轮都重新取一次当前选中的源 —— 页面上换了模型之后，
        该跟的是新选的那个，而不是启动时那个。
        """
        if interval <= 0:
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
                if not self.should_follow():
                    continue
                source = self.active_source()
                try:
                    self._infer.submit(self._reload_once, source).result()
                except Exception as exc:  # 换不上就继续用旧的，别把服务弄挂
                    print(f"[serve] 热加载失败（下轮重试）：{exc}", flush=True)

        threading.Thread(target=loop, daemon=True, name="gi-reload").start()

    # ── 切换模型 ────────────────────────────────────────────────────────────
    def _switch_once(self, index: int) -> None:
        """整个模型换掉。只在推理线程上跑。

        这里用 load_model 重新建一个网络，而不是像热加载那样 load_state_dict ——
        不同的 run 可能是不同的结构（层数、通道数都记在各自的 checkpoint 里），
        往现有网络里灌权重会直接对不上。
        """
        src = self.sources[index]
        model, meta = load_model(src["path"], self.device)
        if meta["board_size"] != self.size:
            raise RuntimeError(
                f"{src['name']} 是 {meta['board_size']}x{meta['board_size']} 的模型，"
                f"当前服务开在 {self.size}x{self.size}，中途换不了"
            )
        self.player.model = model
        self.active = index
        self.meta = meta
        print(f"[serve] 切到 {src['name']}   step {meta['step']:,}", flush=True)

    def switch_model(self, index: int) -> None:
        if not 0 <= index < len(self.sources):
            raise IndexError("没有这个模型")
        self._infer.submit(self._switch_once, index).result()

    def model_list(self) -> list[dict]:
        out = []
        for i, src in enumerate(self.sources):
            live = os.path.isdir(src["path"])
            step = self.meta.get("step") if i == self.active else _peek_step(src["path"])
            out.append({
                "id": i,
                "name": src["name"],
                "step": step,
                "active": i == self.active,
                # run 目录会随训练更新；直接指到 .pt 的是固定快照
                "live": live,
            })
        return out

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
        analysis = self.analyze(game)
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
                "move_prob": session.analysis.move_prob,
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
            # 当前是哪个模型，和它的 step 一起发出去。页面据此渲染下拉框，
            # 于是"显示的"和"真正在下棋的"来自同一份数据，不可能对不上。
            # 激活模型是**服务端全局状态**（不分会话），另一个标签页换了模型，
            # 这边下一次拿状态就会看到，不会各说各话。
            "model": {
                "id": self.active,
                "name": self.sources[self.active]["name"] if self.sources else "",
            },
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
        # 工作目录在 NFS 上，每次请求都重读一遍文件要几十毫秒。缓存住，
        # 但按 mtime 判断是否失效（stat 实测接近零耗时）—— 改完页面刷新即可生效，
        # 不必重启服务，省得改样式时来回等模型重新载入。
        mtime = os.stat(path).st_mtime_ns
        cached = _STATIC_CACHE.get(path)
        if cached is None or cached[0] != mtime:
            with open(path, "rb") as fh:
                cached = (mtime, fh.read())
            _STATIC_CACHE[path] = cached
        ext = os.path.splitext(path)[1]
        self._send(cached[1], CONTENT_TYPES.get(ext, "application/octet-stream"))

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
        elif path == "/api/models":
            self._json({"models": self.app.model_list()})
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
            "/api/model": self._api_model,
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

    def _api_model(self, payload: dict) -> None:
        index = payload.get("id")
        if not isinstance(index, int):
            self._json({"error": "id 必须是整数"}, 400)
            return
        try:
            self.app.switch_model(index)
        except IndexError:
            self._json({"error": "没有这个模型"}, 404)
            return
        except Exception as exc:
            # 换不上就继续用原来那个，把原因告诉页面而不是静默失败
            self._json({"error": f"切换失败：{exc}"}, 409)
            return
        self._json({"models": self.app.model_list()})

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
        session.analysis = self.app.analyze(game)
        self._json(self.app.state(payload["sid"], session))

    def _api_resign(self, payload: dict) -> None:
        session = self._session(payload)
        if session is None:
            return
        session.resigned = session.human
        self._json(self.app.state(payload["sid"], session))


def _peek_step(source: str) -> int | None:
    """从 checkpoint 的文件名里读 step，不去真的加载 53MB 权重。

    只是给页面上的下拉框标个"这个模型练到哪了"，为此把每个候选都 torch.load
    一遍太贵；文件名本来就是 step_000003088.pt 这个格式。
    """
    try:
        name = os.path.basename(resolve_checkpoint(source))
    except (OSError, ValueError):
        return None
    stem = os.path.splitext(name)[0]
    if stem.startswith("step_") and stem[5:].isdigit():
        return int(stem[5:])
    return None


def _source_entry(path: str) -> dict:
    return {"name": os.path.basename(os.path.normpath(path)) or path, "path": path}


def build_app(
    checkpoint: str | list[str],
    device: str = "cpu",
    safe_mode: bool = False,
    temperature: float = 0.0,
) -> App:
    paths = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    model, meta = load_model(paths[0], device)
    size = meta["board_size"]
    player = InstinctPlayer(
        model, size, device, safe_mode=safe_mode, temperature=temperature
    )
    return App(player, meta, size,
               sources=[_source_entry(p) for p in paths], device=device)


def serve(
    checkpoint: str | list[str],
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cpu",
    safe_mode: bool = False,
    temperature: float = 0.0,
    reload_seconds: float = 0.0,
) -> int:
    app = build_app(checkpoint, device, safe_mode, temperature)
    app.warmup()
    app.start_watching(reload_seconds)
    httpd = ThreadingHTTPServer((host, port), functools.partial(Handler, app))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "") else host
    print(f"权重 step {app.meta['step']:,}   棋盘 {app.size}x{app.size}   设备 {device}")
    print("AI 落子完全由一次网络前向决定，不使用任何搜索。")
    if len(app.sources) > 1:
        print("可切换的模型：" + "、".join(s["name"] for s in app.sources))
    if reload_seconds > 0 and os.path.isdir(app.active_source() or ""):
        print(f"每 {reload_seconds:.0f} 秒检查一次新 checkpoint，"
              "页面顶部的 step 会跟着变。")
    else:
        # 说清楚"不会变"，免得对着一个固定快照以为在看训练进度。
        print("权重固定，不会随训练更新。")
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
