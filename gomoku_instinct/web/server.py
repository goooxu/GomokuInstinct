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
import random
import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ..cli.engine import InstinctPlayer
from ..cli.render import move_to_label
from ..eval.opening import opening_moves
from ..model.loader import load_model, resolve_checkpoint
from ..rules import BLACK, WHITE, ForbiddenSemantics, Game, Outcome, RenjuRules

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 页面按概率阈值筛候选，阈值最低可以调到 1%。概率大于 1% 的点最多 99 个，
# 所以取 100 就保证**页面永远不会因为服务端截断而漏掉候选**。
# 这里不做"取前 N 个"式的静默截断：截断了却不说，看到的人会以为模型只考虑了这几个点。
MIN_CANDIDATE_THRESHOLD_PCT = 1
ANALYSIS_TOP_K = 100 // MIN_CANDIDATE_THRESHOLD_PCT

# 会话上限。超出后淘汰最旧的一局 —— 本地试玩工具，不做持久化。
MAX_SESSIONS = 64

# 部署端可选的搜索强度。**0 是默认，也就是项目的零搜索约束。**
#
# 档位是离散的，不做连续滑杆：`BatchSearcher` 的 sims 在构造时就固定了，
# 连续取值意味着为每个位置各造一个 searcher。
#
# 每手耗时 ≈ sims × 单次前向（约 11 ms）—— 单局面搜索是串行的，
# 一个槽位每轮只产一个叶子，所以这条线性关系相当准。实测（15×15，一张卡）：
# 16→0.17s  32→0.35s  64→0.70s  128→1.4s  256→2.7s  400→4.3s
SIMS_CHOICES = (0, 16, 32, 64, 128, 256, 400)

# 观战开局随机落子数。零搜索是确定性的，不随机开局的话每一局都是同一盘棋。
# 默认 2 与竞技场 `play_match(random_opening_plies=2)` 一致；
# 落点规则也共用同一份实现（`eval/opening.py`）。
DEFAULT_OPENING_PLIES = 2
MAX_OPENING_PLIES = 8

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

    def __init__(self, size: int, rules: RenjuRules, human: int,
                 model: int = 0) -> None:
        self.game = Game(size, rules, ForbiddenSemantics.LOSE)
        self.human = human
        self.analysis = None
        self.analysis_color = None
        self.resigned: int | None = None
        # 用哪个模型是**每局各自的事**，不是服务端全局的。开两个标签页就能让
        # 两个模型同时各下各的；也不会出现一个页面换了模型、另一个页面跟着变。
        self.model = model
        # 观战模式：黑白各用一个模型（可以相同）。human 置 0 表示两边都是 AI。
        # 用「颜色 → 模型下标」而不是两个字段，是为了让 ai_move 不必关心是哪种模式。
        self.models: dict[int, int] = {}
        # 这一局用多少次搜索。0 = 零搜索（项目默认）。**每局各自持有**，
        # 和模型选择一样 —— 比赛用时不定，要能中途改。
        self.sims = 0
        # 开头有多少手是随机落的（观战用，见 _api_new）。
        # 前端据此把那几颗子标出来 —— 不标的话你会把它们当成模型下的。
        self.opening_plies = 0

    def model_for(self, color: int) -> int:
        return self.models.get(color, self.model)

    @property
    def watching(self) -> bool:
        return self.human == 0


class App:
    """服务端的全部可变状态。会话改动在锁下进行，推理另有一条专属线程。"""

    def __init__(
        self,
        player: InstinctPlayer,
        meta: dict,
        size: int,
        sources: list[dict] | None = None,
        device: str = "cpu",
        safe_mode: bool = False,
        temperature: float = 0.0,
        sims: int = 0,
    ) -> None:
        self.size = size
        # 新开的对局默认用这个搜索强度（命令行 --sims）。0 = 零搜索。
        self.default_sims = sims
        # 可在页面上切换的模型。每项 {"name": 显示名, "path": --ckpt 给的原始路径}
        self.sources = list(sources or [{"name": "model", "path": None}])
        self.device = device
        self.safe_mode = safe_mode
        self.temperature = temperature
        # 按需载入的模型池。0 号是启动时就载好的那个，其余等真有人选到才载 ——
        # 挂着五个 run 却只玩其中一个时，不该为另外四个白占显存和启动时间。
        self._players: dict[int, InstinctPlayer] = {0: player}
        self._metas: dict[int, dict] = {0: meta}
        # 带搜索的 player 按 (模型下标, 模拟数) 缓存。零搜索那份始终是 _players。
        self._searchers: dict[tuple[int, int], object] = {}
        self.rules = RenjuRules()
        # 观战开局用。故意**不设种子** —— 这里要的就是每局不一样。
        self._opening_rng = random.Random()
        self.lock = threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        # 所有推理都赶到同一个长期存活的线程上跑。
        #
        # ThreadingHTTPServer 每个连接开一条新线程，而 cuBLAS/cuDNN 的句柄是**按线程**
        # 建的：在新线程上做第一次推理要重建句柄。实测同一次前向，主线程 11ms，
        # 每请求新线程 800ms —— 七十多倍，而且不报任何错，只是"点一下要等一秒"。
        # 固定一条预热过的线程之后回到 11ms。
        self._infer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gi-infer")

    # 0 号模型。保留这两个名字是因为它们在很多地方读起来更自然，
    # 语义就是"默认那个"，不再有"全局当前选中"的含义。
    @property
    def player(self) -> InstinctPlayer:
        return self._players[0]

    @property
    def meta(self) -> dict:
        return self._metas[0]

    @meta.setter
    def meta(self, value: dict) -> None:
        self._metas[0] = value

    def ensure_loaded(self, index: int) -> InstinctPlayer:
        """按需把第 index 个模型载进来。**只能在推理线程上调用。**

        载入要占 GPU，和正在进行的前向撞在一起会互相拖慢；
        推理本来就串行在一条线程上，把载入也排进去最省事。
        """
        player = self._players.get(index)
        if player is not None:
            return player
        source = self.sources[index]
        model, meta = load_model(source["path"], self.device)
        if meta["board_size"] != self.size:
            raise RuntimeError(
                f"{source['name']} 是 {meta['board_size']}x{meta['board_size']} 的模型，"
                f"当前服务开在 {self.size}x{self.size}，用不了"
            )
        player = InstinctPlayer(model, self.size, self.device,
                                safe_mode=self.safe_mode,
                                temperature=self.temperature)
        self._players[index] = player
        self._metas[index] = meta
        print(f"[serve] 载入 {source['name']}   step {meta['step']:,}", flush=True)
        return player

    def player_for(self, index: int, sims: int):
        """取这一局该用的 player。**只能在推理线程上调用。**

        sims <= 0 走零搜索（项目默认那条路径），否则走 MCTS。
        搜索 player 按 (模型, 模拟数) 缓存 —— `BatchSearcher` 的 sims
        在构造时固定，换档位就得换一个。
        """
        base = self.ensure_loaded(index)
        if sims <= 0:
            return base
        key = (index, sims)
        found = self._searchers.get(key)
        if found is None:
            from ..cli.search_engine import SearchPlayer

            found = SearchPlayer(base.model, self.size, self.device, sims=sims)
            self._searchers[key] = found
            print(f"[serve] {self.sources[index]['name']} 启用搜索 {sims} 次模拟"
                  f"（约 {sims * 0.011:.1f} 秒/手）", flush=True)
        return found

    def _analyze_on_thread(self, game: Game, index: int, sims: int):
        return self.player_for(index, sims).analyze(game, ANALYSIS_TOP_K)

    def analyze(self, game: Game, index: int = 0, sims: int = 0):
        return self._infer.submit(self._analyze_on_thread, game, index, sims).result()

    def step_of(self, index: int) -> int:
        """第 index 个模型练到哪一步了。

        **没载入时绝不能拿别的模型的 meta 顶替。** 这里原来写的是
        `self._metas.get(index, self._metas[0])` —— 一个模型还没被载入，
        页面上就显示成第一个模型的步数。观战一开局两边都还没走，白方
        renju15f（实际 495 步）就被报成了 renju15c 的 47,958 步。
        这正是 #17 那个 bug 的形状：**名字取自正确的来源，数字取自兜底的来源，
        两半拼出来的东西看着完全合理。**

        没载入就直接从盘上的 checkpoint 文件名读，和模型清单同一个来源，
        两处不会各说各话。
        """
        meta = self._metas.get(index)
        if meta is not None:
            return int(meta.get("step", 0))
        return _peek_step(self.sources[index]["path"]) or 0

    def warmup(self) -> None:
        """先在推理线程上跑一次空盘，把句柄建好，免得第一手卡一下。"""
        self.analyze(Game(self.size, self.rules, ForbiddenSemantics.LOSE))
        if self.default_sims > 0:
            # 搜索档位也预热一次：第一次构造 BatchSearcher 要分配树，
            # 不预热的话开局第一手会莫名其妙地慢。
            self.analyze(Game(self.size, self.rules, ForbiddenSemantics.LOSE),
                         0, self.default_sims)

    # ── 热加载 ──────────────────────────────────────────────────────────────
    def _reload_once(self, index: int) -> None:
        """把某个 run 目录里最新的权重换进来。只在推理线程上跑。"""
        source = self.sources[index]["path"]
        path = resolve_checkpoint(source)
        meta = self._metas[index]
        if path == meta.get("path"):
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        # 只换权重、不换结构。结构真的变了 load_state_dict 会抛，
        # 由外层记下来并保留旧权重继续服务 —— 好过悄悄换上一个对不上的网络。
        self._players[index].model.load_state_dict(state["model"])
        # 整个 dict 一次性换掉：state() 读 meta 时没有加锁，重新绑定是原子的，
        # 逐个字段改则可能被读到一半新一半旧。
        self._metas[index] = dict(
            meta, path=path, step=state.get("step", 0), cycle=state.get("cycle", 0)
        )
        print(f"[serve] {self.sources[index]['name']} 热加载 "
              f"step {self._metas[index]['step']:,}"
              f"（{os.path.basename(path)}）", flush=True)

    def followable(self) -> list[int]:
        """哪些模型要跟着训练更新。

        只跟**已经载入**的：没人选到的模型没必要为它读盘。
        也只跟 run 目录；直接指到某个 .pt 是"我就要这一版"的意思。
        """
        return [
            i for i in sorted(self._players)
            if i < len(self.sources)
            and self.sources[i]["path"]
            and os.path.isdir(self.sources[i]["path"])
        ]

    def start_watching(self, interval: float) -> None:
        """盯着已载入的 run 目录，训练每落一个新 checkpoint 就换上。

        重新载入必须**排到推理线程上**执行，否则会和正在进行的一次前向抢同一个模型。
        """
        if interval <= 0:
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
                for index in self.followable():
                    try:
                        self._infer.submit(self._reload_once, index).result()
                    except Exception as exc:  # 换不上就继续用旧的，别把服务弄挂
                        print(f"[serve] 热加载失败（下轮重试）：{exc}", flush=True)

        threading.Thread(target=loop, daemon=True, name="gi-reload").start()

    # ── 模型 ────────────────────────────────────────────────────────────────
    def use_model(self, session: Session, index: int) -> None:
        """给某一局换模型。**只影响这一局**，别的对局照旧。"""
        if not 0 <= index < len(self.sources):
            raise IndexError("没有这个模型")
        self._infer.submit(self.ensure_loaded, index).result()
        session.model = index

    def model_list(self) -> list[dict]:
        """可选模型清单。**不含"哪个在用"** —— 那是每局各自的事，
        随对局状态一起发，免得两处各说各话（见 #11 那类回写 bug）。
        """
        out = []
        for i, src in enumerate(self.sources):
            path = src["path"]
            loaded = self._metas.get(i)
            out.append({
                "id": i,
                "name": src["name"],
                "step": (loaded or {}).get("step") if loaded else _peek_step(path),
                # run 目录会随训练更新；直接指到 .pt 的是固定快照
                "live": bool(path) and os.path.isdir(path),
            })
        return out

    def new_session(self, human: int, model: int = 0,
                    models: dict[int, int] | None = None) -> tuple[str, Session]:
        for idx in [model, *(models or {}).values()]:
            if not 0 <= idx < len(self.sources):
                raise IndexError("没有这个模型")
        while len(self.sessions) >= MAX_SESSIONS:
            self.sessions.popitem(last=False)
        sid = secrets.token_urlsafe(12)
        session = Session(self.size, self.rules, human, model)
        session.sims = self.default_sims
        if models:
            session.models = dict(models)
        self.sessions[sid] = session
        return sid, session

    def get(self, sid: str) -> Session | None:
        session = self.sessions.get(sid)
        if session is not None:
            self.sessions.move_to_end(sid)
        return session

    def random_opening(self, session: Session, plies: int) -> None:
        """开局先随机落几子，让确定性的双方每局走出不同的棋。

        落点由 `eval.opening.opening_moves` 给出，**与竞技场完全同一份实现** ——
        页面上看到的开局分布，就是评测里用的那个。
        """
        game = session.game
        for move in opening_moves(self.size, plies, self._opening_rng):
            if game.is_terminal():
                break
            game.play(move)
        session.opening_plies = game.num_moves

    def ai_move(self, session: Session) -> None:
        """让 AI 走一手。调用方必须已持有锁。"""
        game = session.game
        if game.is_terminal() or session.resigned is not None:
            return
        analysis = self.analyze(game, session.model_for(game.to_move), session.sims)
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
            # 这一局用的是哪个模型、它练到哪一步 —— 都由 session 的选择算出来，
            # 页面据此渲染，"显示的"和"真正在下棋的"不可能对不上。
            "step": self.step_of(session.model),
            "model": {
                "id": session.model,
                "name": self.sources[session.model]["name"],
            },
            # 搜索强度。0 = 零搜索（项目默认）。**必须发到前端并显眼标出** ——
            # 拿一个开着搜索的服务去测棋力却不自知，正是第 11 章那类静默失败。
            "sims": session.sims,
            "sims_choices": list(SIMS_CHOICES),
            # 观战模式：两边都是 AI，界面据此切换控件并驱动单步推进
            "watching": session.watching,
            "opening": session.opening_plies,
            "seats": {
                str(c): {"id": session.model_for(c),
                         "name": self.sources[session.model_for(c)]["name"],
                         "step": self.step_of(session.model_for(c))}
                for c in (BLACK, WHITE)
            } if session.watching else None,
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
            "/api/step": self._api_step,
            "/api/sims": self._api_sims,
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
        watch = payload.get("watch")
        if watch is not None:
            # 观战：两边都是 AI。human 置 0 表示没有人类座位。
            if not isinstance(watch, dict):
                self._json({"error": "watch 必须是对象"}, 400)
                return
            try:
                models = {BLACK: int(watch["black"]), WHITE: int(watch["white"])}
            except (KeyError, TypeError, ValueError):
                self._json({"error": "watch 需要 black 与 white 两个模型 id"}, 400)
                return
            # **必须有随机开局，否则每一局都是同一盘棋的重放。**
            # 零搜索落子是 argmax、temperature = 0 —— 确定性函数，
            # 同一个模型看到同一个局面必然给出同一手，空盘开局只有唯一一条棋路。
            # 竞技场早就踩过这个坑（图鉴 #12：得分率恒等于 50%），
            # 那边的解法是每局开头随机落 2 子（`play_match` 的 random_opening_plies），
            # 这里用同一套机制，默认值也一样。
            # 设成 0 就是"每次都下同一盘"—— 想反复看同一条棋路时那才是要的。
            try:
                opening = int(watch.get("opening", DEFAULT_OPENING_PLIES))
            except (TypeError, ValueError):
                self._json({"error": "opening 必须是整数"}, 400)
                return
            if not 0 <= opening <= MAX_OPENING_PLIES:
                self._json({"error": f"opening 只能是 0~{MAX_OPENING_PLIES}"}, 400)
                return
            try:
                sid, session = self.app.new_session(0, models[BLACK], models)
            except IndexError:
                self._json({"error": "没有这个模型"}, 404)
                return
            self.app.random_opening(session, opening)
            # 随机开局之后不自动走 —— 让界面按自己的节奏推进，模型的第一手也要看得见
            self._json(self.app.state(sid, session))
            return

        human = WHITE if str(payload.get("color", "black")).startswith("w") else BLACK
        model = payload.get("model", 0)
        if not isinstance(model, int):
            self._json({"error": "model 必须是整数"}, 400)
            return
        try:
            sid, session = self.app.new_session(human, model)
        except IndexError:
            self._json({"error": "没有这个模型"}, 404)
            return
        if human == WHITE:
            self.app.ai_move(session)  # AI 执黑先行
        self._json(self.app.state(sid, session))

    def _api_step(self, payload: dict) -> None:
        """观战模式下推进一手。

        每次只走一手、由界面决定什么时候要下一手 —— 这样"每手停几秒"和
        "手动点一下走一手"是同一套机制，服务端不需要知道界面用的哪种节奏。
        """
        session = self._session(payload)
        if session is None:
            return
        if not session.watching:
            self._json({"error": "只有观战模式能用单步推进"}, 409)
            return
        self.app.ai_move(session)
        self._json(self.app.state(payload["sid"], session))

    def _api_sims(self, payload: dict) -> None:
        """改**这一局**的搜索强度。别的对局不受影响，对局中途也能改。"""
        session = self._session(payload)
        if session is None:
            return
        sims = payload.get("sims")
        if not isinstance(sims, int) or sims not in SIMS_CHOICES:
            self._json({"error": f"sims 只能取 {list(SIMS_CHOICES)}"}, 400)
            return
        session.sims = sims
        self._json(self.app.state(payload["sid"], session))

    def _api_model(self, payload: dict) -> None:
        """给**这一局**换模型。别的对局不受影响。"""
        session = self._session(payload)
        if session is None:
            return
        index = payload.get("id")
        if not isinstance(index, int):
            self._json({"error": "id 必须是整数"}, 400)
            return
        try:
            self.app.use_model(session, index)
        except IndexError:
            self._json({"error": "没有这个模型"}, 404)
            return
        except Exception as exc:
            # 换不上就继续用原来那个，把原因告诉页面而不是静默失败
            self._json({"error": f"换不了：{exc}"}, 409)
            return
        self._json(self.app.state(payload["sid"], session))

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
    sims: int = 0,
) -> App:
    paths = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    model, meta = load_model(paths[0], device)
    size = meta["board_size"]
    player = InstinctPlayer(
        model, size, device, safe_mode=safe_mode, temperature=temperature
    )
    return App(player, meta, size,
               sources=[_source_entry(p) for p in paths], device=device,
               safe_mode=safe_mode, temperature=temperature, sims=sims)


def serve(
    checkpoint: str | list[str],
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cpu",
    safe_mode: bool = False,
    temperature: float = 0.0,
    reload_seconds: float = 0.0,
    sims: int = 0,
) -> int:
    app = build_app(checkpoint, device, safe_mode, temperature, sims)
    app.warmup()
    app.start_watching(reload_seconds)
    httpd = ThreadingHTTPServer((host, port), functools.partial(Handler, app))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "") else host
    print(f"权重 step {app.meta['step']:,}   棋盘 {app.size}x{app.size}   设备 {device}")
    if sims > 0:
        print(f"** 搜索模式：每手 {sims} 次 MCTS 模拟（约 {sims * 0.011:.1f} 秒/手）**")
        print("   这**破坏了本项目「零搜索推理」那条核心约束** —— "
              "技术报告里的所有棋力数字都是零搜索口径，")
        print("   不要拿这个服务去测那些数。页面上每局可以各自改档位，0 即恢复零搜索。")
    else:
        print("AI 落子完全由一次网络前向决定，不使用任何搜索。"
              "（--sims N 可开搜索，默认关闭）")
    if len(app.sources) > 1:
        print("可选模型：" + "、".join(s["name"] for s in app.sources)
              + "　（每局各自选，开多个标签页就能同时玩多局）")
    if reload_seconds > 0 and app.followable():
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
