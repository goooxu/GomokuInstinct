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
