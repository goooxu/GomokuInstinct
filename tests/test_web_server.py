"""网页版对战服务端的测试。

用假 player 起一个真的 HTTP 服务，走真实的 HTTP 请求 —— 这些接口的价值全在
「浏览器发过来一个不合规矩的请求时会怎样」，直接调函数测不出来。
"""

from __future__ import annotations

import functools
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gomoku_instinct.cli.engine import MoveAnalysis
from gomoku_instinct.rules import BLACK, WHITE
from gomoku_instinct.web.server import App, Handler

SIZE = 15


class _StubPlayer:
    """假 AI：默认挑序号最大的空点，从右下角往左铺，不会跟人的着法搅在一起。"""

    def __init__(self, replies=None) -> None:
        self.replies = list(replies) if replies else None
        self.calls = 0
        self.threads: set[int] = set()

    def analyze(self, game, top_k: int = 5) -> MoveAnalysis:
        self.calls += 1
        self.threads.add(threading.get_ident())
        if self.replies:
            move = self.replies.pop(0)
        else:
            move = max(i for i, v in enumerate(game.board.grid) if v == 0)
        return MoveAnalysis(
            move=move,
            move_prob=0.9,
            value=0.25,
            top_moves=[(move, 0.9), ((move + 1) % (SIZE * SIZE), 0.1)],
            forbidden_pred=[],
            masked_forbidden=False,
        )


@pytest.fixture
def server():
    app = App(_StubPlayer(), {"step": 42}, SIZE)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    finally:
        httpd.shutdown()
        httpd.server_close()


def post(base: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.load(res)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def raw_get(base: str, path: str) -> tuple[int, bytes]:
    """用 http.client 是为了让路径原样发出去，不被 urllib 规范化掉。"""
    host = base.split("//", 1)[1]
    conn = http.client.HTTPConnection(host, timeout=10)
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        return res.status, res.read()
    finally:
        conn.close()


def rowcol(r: int, c: int) -> int:
    return r * SIZE + c


# ── 开局 ────────────────────────────────────────────────────────────────────
def test_new_game_as_black_waits_for_human(server):
    base, app = server
    status, s = post(base, "/api/new", {"color": "black"})
    assert status == 200
    assert s["human"] == BLACK and s["to_move"] == BLACK
    assert sum(s["grid"]) == 0
    assert app.player.calls == 0  # 执黑先行，AI 不该抢着落子


def test_new_game_as_white_lets_ai_open(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "white"})
    assert s["human"] == WHITE and s["to_move"] == WHITE
    assert s["grid"].count(BLACK) == 1 and len(s["history"]) == 1


def test_sessions_are_independent(server):
    base, _ = server
    _, a = post(base, "/api/new", {"color": "black"})
    _, b = post(base, "/api/new", {"color": "black"})
    assert a["sid"] != b["sid"]
    post(base, "/api/move", {"sid": a["sid"], "move": rowcol(7, 7)})
    _, b2 = post(base, "/api/hint", {"sid": b["sid"]})
    assert len(b2["history"]) == 0  # 另一局不受影响


# ── 落子 ────────────────────────────────────────────────────────────────────
def test_move_plays_human_then_ai(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    status, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    assert status == 200
    assert s["grid"][rowcol(7, 7)] == BLACK
    assert s["grid"].count(WHITE) == 1
    assert s["to_move"] == BLACK  # 又轮到人


def test_occupied_point_is_rejected(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    before = list(s["grid"])
    status, err = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    assert status == 409 and "已经有子" in err["error"]
    _, after = post(base, "/api/hint", {"sid": s["sid"]})
    assert after["grid"] == before  # 被拒绝的请求不能改动棋盘


@pytest.mark.parametrize("bad", [-1, SIZE * SIZE, "H8", None, 1.5])
def test_out_of_range_move_is_rejected(server, bad):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    status, _ = post(base, "/api/move", {"sid": s["sid"], "move": bad})
    assert status == 400


def test_unknown_session_returns_410(server):
    base, _ = server
    status, err = post(base, "/api/move", {"sid": "nope", "move": 0})
    assert status == 410 and "失效" in err["error"]


def test_win_is_detected_and_stops_the_game(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    for c in range(5):
        _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, c)})
    assert s["outcome"] == "black_win"
    status, _ = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(0, 0)})
    assert status == 409  # 终局后不能再落子


# ── 禁手 ────────────────────────────────────────────────────────────────────
def test_forbidden_points_given_to_black_only(server):
    base, _ = server
    _, black = post(base, "/api/new", {"color": "black"})
    assert black["forbidden"] is not None and len(black["forbidden"]) == SIZE * SIZE

    _, white = post(base, "/api/new", {"color": "white"})
    # 人执白时不给：那是 AI 的禁手点，摊开等于替它把风险标出来
    assert white["forbidden"] is None


def test_forbidden_move_loses_immediately(server):
    """禁手点是合法落子但立即判负 —— 服务端必须如实反映，不能悄悄拦下。"""
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    # 造一个黑方三三点：两条独立的活三交于 (7,7)
    for point in (rowcol(7, 5), rowcol(7, 6), rowcol(5, 7), rowcol(6, 7)):
        _, s = post(base, "/api/move", {"sid": s["sid"], "move": point})
        assert s["outcome"] == "ongoing"
    assert s["forbidden"][rowcol(7, 7)] is True

    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    assert s["outcome"] == "white_win"
    assert s["history"][-1]["forbidden"] == 3  # 三三


# ── 悔棋 / 提示 / 认输 ──────────────────────────────────────────────────────
def test_undo_returns_to_human_turn_as_black(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    _, s = post(base, "/api/undo", {"sid": s["sid"]})
    assert len(s["history"]) == 0 and s["to_move"] == BLACK


def test_undo_returns_to_human_turn_as_white(server):
    """人执白时退两手会退成轮到 AI，服务端要补一手回来，否则棋局卡死。"""
    base, _ = server
    _, s = post(base, "/api/new", {"color": "white"})
    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    assert len(s["history"]) == 3
    _, s = post(base, "/api/undo", {"sid": s["sid"]})
    assert s["to_move"] == WHITE and len(s["history"]) == 1


def test_hint_does_not_change_the_board(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, hinted = post(base, "/api/hint", {"sid": s["sid"]})
    assert hinted["grid"] == s["grid"] and hinted["history"] == []
    # 提示是站在人这一方算的，正负号含义与 AI 自评相反，必须标清楚
    assert hinted["analysis"]["value_color"] == BLACK
    assert hinted["analysis"]["top"][0]["label"]


def test_ai_analysis_is_labelled_with_its_own_side(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    assert s["analysis"]["value_color"] == WHITE


def test_played_move_probability_is_reported(server):
    """必须单独给出「实际落的那手」的概率。

    从 top_moves 里反查也能凑合，但 --temperature > 0 时抽样到的那手未必落在
    top-5 里，查不到就只能悄悄不显示 —— 又一个不报错的静默降级。
    """
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, s = post(base, "/api/move", {"sid": s["sid"], "move": rowcol(7, 7)})
    a = s["analysis"]
    assert a["move_prob"] == pytest.approx(0.9)
    assert s["grid"][a["move"]] != 0  # 这手确实已经落在盘上


def test_resign_gives_the_win_to_the_opponent(server):
    base, _ = server
    _, s = post(base, "/api/new", {"color": "black"})
    _, s = post(base, "/api/resign", {"sid": s["sid"]})
    assert s["outcome"] == "white_win" and s["resigned"] == BLACK
    status, _ = post(base, "/api/move", {"sid": s["sid"], "move": 0})
    assert status == 409


# ── 静态资源 ────────────────────────────────────────────────────────────────
def test_index_is_served(server):
    base, _ = server
    status, body = raw_get(base, "/")
    assert status == 200 and b"<canvas" in body


def test_path_traversal_is_blocked(server):
    base, _ = server
    for path in ("/static/../server.py", "/static/../../rules/game.py"):
        status, _ = raw_get(base, path)
        assert status == 404


def test_unknown_route_is_404(server):
    base, _ = server
    assert raw_get(base, "/nope")[0] == 404
    assert post(base, "/api/nope", {})[0] == 404


def test_session_cap_evicts_oldest(server):
    from gomoku_instinct.web.server import MAX_SESSIONS

    base, app = server
    for _ in range(MAX_SESSIONS + 3):
        post(base, "/api/new", {"color": "black"})
    assert len(app.sessions) <= MAX_SESSIONS


# ── 推理线程 ────────────────────────────────────────────────────────────────
def test_inference_stays_on_one_dedicated_thread(server):
    """推理必须固定在一条长期存活的线程上跑。

    ThreadingHTTPServer 每个连接开一条新线程，而 cuBLAS/cuDNN 句柄是按线程建的：
    在新线程上做第一次推理要重建句柄。实测同一次前向，主线程 11ms、每请求新线程
    800ms —— 七十多倍，却不报任何错，只表现为"点一下要等一秒"。
    """
    base, app = server
    _, s = post(base, "/api/new", {"color": "black"})
    for _ in range(6):
        post(base, "/api/hint", {"sid": s["sid"]})
    assert app.player.calls >= 6
    assert len(app.player.threads) == 1


def test_inference_thread_is_not_a_request_thread(server):
    """而且那条线程不能是某个请求自己的线程 —— 请求线程用完就没了。"""
    base, app = server
    _, s = post(base, "/api/new", {"color": "black"})
    post(base, "/api/hint", {"sid": s["sid"]})
    infer_thread = next(iter(app.player.threads))
    alive = {t.ident for t in threading.enumerate()}
    assert infer_thread in alive
    time.sleep(0.05)
    post(base, "/api/hint", {"sid": s["sid"]})
    assert app.player.threads == {infer_thread}  # 换了连接还是同一条


def test_warmup_primes_the_inference_thread(server):
    base, app = server
    app.warmup()
    assert app.player.calls == 1 and len(app.player.threads) == 1


def test_static_cache_picks_up_edits(tmp_path, server):
    """静态文件缓存必须按 mtime 失效 —— 改了页面刷新不生效比慢更难查。"""
    import os

    from gomoku_instinct.web import server as srv

    base, _ = server
    path = os.path.join(srv.STATIC_DIR, "index.html")
    original = open(path, "rb").read()
    try:
        assert raw_get(base, "/")[1] == original
        with open(path, "wb") as fh:
            fh.write(b"<canvas>changed</canvas>")
        os.utime(path, (0, 0))          # 明确改 mtime，不依赖文件系统时间精度
        assert raw_get(base, "/")[1] == b"<canvas>changed</canvas>"
    finally:
        with open(path, "wb") as fh:
            fh.write(original)
        srv._STATIC_CACHE.clear()


# ── 前端静态检查 ────────────────────────────────────────────────────────────
#
# 容器里没有 JS 运行时，前端跑不了真正的单元测试。下面两条是把已经踩过的两个
# bug 写成不变量做静态检查 —— 挡不住所有前端问题，但至少挡住这两类复发。


def _function_body(text: str, marker: str) -> str:
    """取出 marker 之后第一对大括号里的内容（按括号配对，不靠缩进猜）。"""
    start = text.index(marker)
    depth, i = 0, text.index("{", start)
    begin = i
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i]
        i += 1


def _strip_comments(text: str) -> str:
    """去掉 // 行注释 —— 不然解释 bug 的注释本身会把检查判红。"""
    import re

    return re.sub(r"//.*", "", text)


def _page() -> str:
    import os

    from gomoku_instinct.web import server as srv

    with open(os.path.join(srv.STATIC_DIR, "index.html")) as fh:
        return fh.read()


def test_render_never_writes_back_to_input_controls():
    """render() 不能回写任何输入控件。

    guard() 在发请求前会先 render() 一次。render() 里只要有一句往 <select>/<input>
    回写，用户刚做的选择就会在被读到之前被抹掉 —— 实际后果是「永远开不出白棋局」，
    而且不报任何错，点下去只是又开了一局黑棋。
    """
    import re

    body = _strip_comments(_function_body(_page(), "function render()"))
    offenders = re.findall(r'\$\("([^"]+)"\)\.(value|checked)\s*=(?!=)', body)
    assert not offenders, f"render() 回写了输入控件: {offenders}"


def test_api_error_recovery_does_not_reenter_guard():
    """api() 的失效恢复路径不能再走 guard()。

    guard() 开头就是 `if (pending) return;`，而 api() 只会在 guard() 内部被调用，
    此时 pending 必为 true。在这里调 newGame() 等于什么都没做，用户会卡死在一个
    已失效的对局上 —— 同样不报错。
    """
    body = _strip_comments(_function_body(_page(), "async function api("))
    for call in ("guard(", "newGame(", "startGame("):
        assert call not in body, f"api() 里不该调用 {call}"


def test_page_javascript_parses():
    """页面里的 JS 必须能解析通过。

    语法错误的后果是整页白屏，而服务端一切正常、没有任何报错 —— 只能靠打开浏览器
    才发现。容器里没有 JS 运行时，用纯 Python 的 esprima 只做语法检查；
    它解析到 ES2017，用了更新的语法（比如可选链 ?.）会在这里报错而不是在浏览器里。
    """
    import re

    esprima = pytest.importorskip("esprima", reason="未安装 esprima，跳过 JS 语法检查")
    script = re.search(r"<script>(.*?)</script>", _page(), re.S)
    assert script, "页面里找不到 <script> 块"
    esprima.parseScript(script.group(1))


def test_analysis_marks_are_shown_by_default():
    """模型的候选点必须默认画在棋盘上，而不是藏在开关后面。

    数量不再固定为两个：凡是概率不低于阈值的候选都要标出来，
    所以这里只认「按阈值筛」和「确实画了」这两件事。
    """
    body = _strip_comments(_function_body(_page(), "function analysisMarks()"))
    assert "threshold()" in body, "候选点没有按阈值筛"
    assert "slice(" not in body, "候选点被截断了固定条数"
    draw = _strip_comments(_function_body(_page(), "function draw()"))
    assert "analysisMarks()" in draw


def test_played_probability_is_drawn(server):
    """已落子的那颗子上要写模型给这一手的概率。"""
    draw = _strip_comments(_function_body(_page(), "function draw()"))
    assert "played.move_prob" in draw


def test_move_numbers_are_gone():
    """手数显示已经移除，相关控件和绘制都不该再留着。"""
    page = _strip_comments(_page())
    assert "showNum" not in page


def test_tiny_probabilities_do_not_render_as_zero():
    """极小的概率要显示 <1%，不能四舍五入成 0% —— 0% 读起来像"不可能"，是假的。"""
    page = _strip_comments(_page())
    assert '"<1%"' in page


def test_color_select_offers_both_sides():
    page = _page()
    assert 'value="black"' in page and 'value="white"' in page


# ── 热加载 ──────────────────────────────────────────────────────────────────
#
# 试玩服务盯着 run 目录，训练每落一个新 checkpoint 就换上，这样页面顶部的 step
# 会跟着训练走。这里最要紧的两条：换不上时必须继续用旧权重服务，
# 以及绝不能在半路把 meta 改成一半新一半旧。


class _StubModel:
    def __init__(self, fail: bool = False) -> None:
        self.loaded: list = []
        self.fail = fail

    def load_state_dict(self, state) -> None:
        if self.fail:
            raise RuntimeError("网络结构对不上")
        self.loaded.append(state)


def _make_app_with_model(model, path=None, name="run"):
    player = _StubPlayer()
    player.model = model
    return App(player, {"step": 0, "path": None}, SIZE,
               sources=[{"name": name, "path": path}])


def _write_ckpt(run_dir, step: int) -> str:
    import torch

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:09d}.pt"
    torch.save({"model": {"w": step}, "step": step, "cycle": step // 10},
               ckpt_dir / name)
    (ckpt_dir / "latest").write_text(name)
    return name


def test_reload_picks_up_new_checkpoint(tmp_path):
    model = _StubModel()
    app = _make_app_with_model(model, str(tmp_path))

    _write_ckpt(tmp_path, 100)
    app._reload_once(0)
    assert app.meta["step"] == 100
    assert model.loaded == [{"w": 100}]

    _write_ckpt(tmp_path, 250)
    app._reload_once(0)
    assert app.meta["step"] == 250
    assert len(model.loaded) == 2


def test_reload_is_noop_when_latest_unchanged(tmp_path):
    model = _StubModel()
    app = _make_app_with_model(model, str(tmp_path))

    _write_ckpt(tmp_path, 100)
    app._reload_once(0)
    app._reload_once(0)
    # 每 N 秒查一次，绝大多数时候没有新权重；不能每次都把 53MB 重读一遍
    assert len(model.loaded) == 1


def test_reload_failure_keeps_old_weights(tmp_path):
    model = _StubModel(fail=True)
    app = _make_app_with_model(model, str(tmp_path))
    app.meta = {"step": 7, "path": "旧的"}

    _write_ckpt(tmp_path, 100)
    with pytest.raises(RuntimeError):
        app._reload_once(0)

    # 关键：load 抛了以后 meta 不能被改动，否则页面会显示一个并没有装上的 step
    assert app.meta == {"step": 7, "path": "旧的"}


def test_watching_disabled_when_interval_is_zero(tmp_path):
    app = _make_app_with_model(_StubModel(), str(tmp_path))
    before = threading.active_count()
    app.start_watching(0.0)
    assert threading.active_count() == before


def test_only_loaded_run_dirs_are_followed(tmp_path):
    """只跟已载入的 run 目录。

    指到具体某个 .pt 的不跟（那是「我就要这一版」的意思）；
    还没有人选到、根本没载入的也不跟，没必要为它反复读盘。
    """
    name = _write_ckpt(tmp_path, 100)
    pinned = str(tmp_path / "checkpoints" / name)

    app = _make_app_with_model(_StubModel(), str(tmp_path))
    app.sources.append({"name": "pin", "path": pinned})
    app.sources.append({"name": "没载入", "path": str(tmp_path)})
    # 只有 0 号载入过（构造时就给了），另外两个都没有
    assert app.followable() == [0]

    app = _make_app_with_model(_StubModel(), pinned)
    assert app.followable() == []


# ── 页面上切换模型 ──────────────────────────────────────────────────────────


def test_peek_step_reads_from_filename(tmp_path):
    """列表里那个 step 是从文件名解析的，不该为了标个数字去 load 53MB 权重。"""
    from gomoku_instinct.web import server as server_mod

    _write_ckpt(tmp_path, 4321)
    assert server_mod._peek_step(str(tmp_path)) == 4321
    assert server_mod._peek_step(str(tmp_path / "不存在")) is None


def test_model_list_has_no_active_flag(tmp_path):
    """清单里**不含**「哪个在用」—— 那是每局各自的事，随对局状态一起发。

    两处各自维护同一件事，就会有一处过期；#11 号那类 bug 全是这么来的。
    """
    app = _make_app_with_model(_StubModel(), str(tmp_path), name="训练中")
    _write_ckpt(tmp_path, 900)
    pinned = str(tmp_path / "checkpoints" / "step_000000900.pt")
    app.sources.append({"name": "固定版", "path": pinned})
    app.meta = {"step": 950, "path": "x"}

    items = app.model_list()
    assert all("active" not in m for m in items)
    assert [m["live"] for m in items] == [True, False]
    # 已载入的用它自己的 meta（热加载后会比文件名新），没载入的才去看文件名
    assert items[0]["step"] == 950
    assert items[1]["step"] == 900


def _two_model_app(monkeypatch, board_size=SIZE, step=77):
    from gomoku_instinct.web import server as server_mod

    app = _make_app_with_model(_StubModel(), None, name="a")
    app.sources.append({"name": "b", "path": "B"})
    sentinel = object()
    monkeypatch.setattr(
        server_mod, "load_model",
        lambda path, device: (sentinel,
                              {"step": step, "board_size": board_size, "path": path}),
    )
    monkeypatch.setattr(server_mod, "InstinctPlayer",
                        lambda model, *a, **k: _StubPlayer())
    return app, sentinel


def test_each_session_keeps_its_own_model(monkeypatch):
    """换模型只动这一局。另一局该用哪个还用哪个。"""
    app, _ = _two_model_app(monkeypatch)
    _, one = app.new_session(BLACK)
    _, two = app.new_session(BLACK)

    app.use_model(one, 1)
    assert one.model == 1
    assert two.model == 0          # 另一局完全不受影响


def test_new_session_can_pick_a_model(monkeypatch):
    app, _ = _two_model_app(monkeypatch)
    _, session = app.new_session(BLACK, 1)
    assert session.model == 1
    with pytest.raises(IndexError):
        app.new_session(BLACK, 9)


def test_models_are_loaded_lazily(monkeypatch):
    """没人选到的模型不该被载入 —— 挂五个 run 只玩一个时不该白占显存。"""
    app, sentinel = _two_model_app(monkeypatch)
    assert sorted(app._players) == [0]

    _, session = app.new_session(BLACK)
    app.use_model(session, 1)
    assert sorted(app._players) == [0, 1]
    assert app.meta_for(1)["step"] == 77


def test_use_model_rejects_other_board_size(monkeypatch):
    app, _ = _two_model_app(monkeypatch, board_size=9)
    _, session = app.new_session(BLACK)
    with pytest.raises(RuntimeError):
        app.use_model(session, 1)
    # 换不上就保持原样，否则页面显示的和真正在下棋的不是同一个模型
    assert session.model == 0


def test_use_model_rejects_bad_index(monkeypatch):
    app, _ = _two_model_app(monkeypatch)
    _, session = app.new_session(BLACK)
    with pytest.raises(IndexError):
        app.use_model(session, 5)


def test_render_models_never_mutates_the_model_list():
    """renderModels() 不能往 MODELS 里写。

    「哪个模型在用」是服务端全局状态，MODELS 只是某一次请求的快照，
    两边短暂不同步是常态。render 里只要有一句写回去，就会留下**永久**脏值 ——
    实测表现是 renju15b 顶着 renju15 的 step 30,012 显示，刷新都不一定好。
    这和 #11 号（render() 抹掉用户刚选的颜色）是同一个模式：render 只许读、不许写。
    """
    import re

    body = _strip_comments(_function_body(_page(), "function renderModels()"))
    # 只认「赋值给 MODELS 本身 / 它的下标 / 它的元素字段」，
    # 不能把同一行后面无关的 `=` 也算进来
    offenders = re.findall(r"\bMODELS\s*(?:\[[^\]]*\])?\s*=(?!=)", body)
    offenders += re.findall(r"\bMODELS\.\w+\s*=(?!=)", body)
    offenders += re.findall(r"\b(?:m|active|entry)\.(?:step|active|name|live|id)\s*=(?!=)",
                            body)
    assert not offenders, f"renderModels() 写了模型列表: {offenders}"


def test_state_carries_the_active_model(server):
    """状态里必须带上「当前是哪个模型」。

    页面的下拉框和顶部的 step 都从这一份数据渲染，不可能各说各话。
    """
    base, app = server
    app.sources = [{"name": "a", "path": "A"}, {"name": "b", "path": "B"}]
    status, state = post(base, "/api/new", {"color": "black"})
    assert status == 200
    assert state["model"] == {"id": 0, "name": "a"}


# ── 候选阈值 ────────────────────────────────────────────────────────────────


def test_top_k_covers_the_lowest_threshold():
    """服务端一次要给够候选，不能让页面因为截断而漏掉。

    页面按概率阈值筛候选，阈值最低 1%；概率大于 1% 的点最多 99 个。
    只要 top_k 不小于这个数，任何允许的阈值都不会被服务端悄悄截断 ——
    截断了却不说，看到的人会以为模型只考虑了这几个点。
    """
    from gomoku_instinct.web import server as server_mod

    most_possible = 100 // server_mod.MIN_CANDIDATE_THRESHOLD_PCT
    assert server_mod.ANALYSIS_TOP_K >= most_possible


def test_analysis_asks_for_the_full_candidate_list():
    """走一手棋时必须按 ANALYSIS_TOP_K 要候选，不能用 analyze 的默认 5。"""
    from gomoku_instinct.rules import ForbiddenSemantics, Game
    from gomoku_instinct.web import server as server_mod

    seen = {}

    class _Recording(_StubPlayer):
        def analyze(self, game, top_k: int = 5):
            seen["top_k"] = top_k
            return super().analyze(game, top_k)

    app = App(_Recording(), {"step": 1}, SIZE)
    app.analyze(Game(SIZE, app.rules, ForbiddenSemantics.LOSE))
    assert seen["top_k"] == server_mod.ANALYSIS_TOP_K


def test_threshold_slider_bounds_match_the_server():
    """滑杆的下限不能比服务端 top_k 能覆盖的还低，否则又是静默截断。"""
    import re

    page = _page()
    lo = int(re.search(r'id="thr"[^>]*\bmin="(\d+)"', page).group(1))
    from gomoku_instinct.web import server as server_mod
    assert lo >= server_mod.MIN_CANDIDATE_THRESHOLD_PCT


def test_render_never_writes_the_threshold_input():
    """阈值滑杆是输入控件，render() 不得回写 —— 同 #11 号那条不变量。"""
    import re

    body = _strip_comments(_function_body(_page(), "function render()"))
    assert not re.findall(r'\$\("thr"\)\.value\s*=(?!=)', body)


def test_layout_direction_is_decided_by_board_size():
    """换向不能靠拍一个 CSS 断点。

    该不该上下排布取决于「哪种排法棋盘更大」，而那要同时看窗口的宽和高 ——
    CSS media query 只能看其中一个，必然在某些窗口尺寸上选错。
    """
    page = _page()
    assert "@media" not in page, "又用 media query 拍断点了"
    body = _strip_comments(_function_body(page, "function resize()"))
    assert "innerHeight" in body and "clientWidth" in body, "换向没有同时看宽和高"
    assert 'classList.toggle("stacked"' in body


def test_board_never_exceeds_its_container():
    """棋盘尺寸绝不能被抬到一个固定下限。

    原来写的是 Math.max(280, avail)：容器窄于 280px 时画布就比容器还宽，
    右边一列连同坐标被切掉 —— 窄窗口下必现，而且不报任何错。
    """
    import re

    body = _strip_comments(_function_body(_page(), "function resize()"))
    bad = re.findall(r"Math\.max\(\s*(\d+)", body)
    assert all(int(n) <= 1 for n in bad), f"棋盘尺寸被抬到了固定下限: {bad}"
