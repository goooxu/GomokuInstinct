"""gomoku-instinct 命令行入口。"""

from __future__ import annotations

import argparse
import sys

import torch


def _cmd_play(args) -> int:
    from . import play

    return play.run(
        args.ckpt,
        human_color=args.color,
        device=args.device,
        show_policy=args.show_policy,
        safe_mode=args.safe_mode,
        temperature=args.temperature,
        input_mode=args.input_mode,
    )


def _cmd_serve(args) -> int:
    from ..web import serve

    return serve(
        args.ckpt,
        host=args.host,
        port=args.port,
        device=args.device,
        safe_mode=args.safe_mode,
        temperature=args.temperature,
        reload_seconds=args.reload_seconds,
    )


def _cmd_arena(args) -> int:
    from ..core import load_core
    from ..eval import GreedyThreatPlayer, RandomPlayer, play_match
    from ..model.loader import load_model
    from ..rules import RenjuRules
    from .engine import InstinctPlayer

    core = load_core()

    def build(spec: str, size_hint: int | None):
        if spec == "random":
            return RandomPlayer(seed=args.seed), "random", size_hint
        if spec == "greedy":
            if size_hint is None:
                raise SystemExit("greedy 基线需要先由另一方确定棋盘尺寸")
            return (
                GreedyThreatPlayer(core.Rules(), size_hint, seed=args.seed),
                "greedy_threat",
                size_hint,
            )
        model, meta = load_model(spec, args.device)
        name = f"{spec}@{meta['step']}"
        return (
            InstinctPlayer(model, meta["board_size"], args.device,
                           safe_mode=args.safe_mode),
            name,
            meta["board_size"],
        )

    # 先建能确定棋盘尺寸的一方
    if args.a in ("random", "greedy"):
        player_b, name_b, size = build(args.b, None)
        player_a, name_a, _ = build(args.a, size)
    else:
        player_a, name_a, size = build(args.a, None)
        player_b, name_b, _ = build(args.b, size)

    print(f"零搜索模式对局 {args.games} 局，棋盘 {size}x{size}")
    result = play_match(
        player_a,
        player_b,
        games=args.games,
        board_size=size,
        rules=RenjuRules(),
        batch=args.batch,
    )
    print()
    print(result.summary(name_a, name_b))
    return 0


def _cmd_show(args) -> int:
    from ..model.loader import load_model

    model, meta = load_model(args.ckpt, "cpu")
    print(model.parameter_summary())
    print(f"checkpoint: {meta['path']}")
    print(f"训练步数: {meta['step']:,}   周期: {meta['cycle']:,}")
    print(f"棋盘: {meta['board_size']}x{meta['board_size']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gomoku-instinct",
        description="从零自博弈训练的连珠 AI —— 对战时仅靠单次网络前向落子，不使用任何搜索",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    play_parser = sub.add_parser("play", help="人机对战")
    play_parser.add_argument("--ckpt", required=True, help="checkpoint 文件或 run 目录")
    play_parser.add_argument("--color", default="black", choices=["black", "white"],
                             help="你执哪一方")
    play_parser.add_argument("--show-policy", action="store_true",
                             help="显示 AI 的候选手概率、价值估计与禁手点预测")
    play_parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="用规则替 AI 屏蔽禁手点。默认关闭 —— 开启后落子就不再是纯模型推理了",
    )
    play_parser.add_argument("--temperature", type=float, default=0.0,
                             help="0 表示确定性 argmax")
    play_parser.add_argument(
        "--input-mode", default="auto", choices=["auto", "cursor", "text"],
        help="cursor 用方向键选点（默认，需要真正的终端）；text 逐行输入坐标",
    )
    play_parser.set_defaults(func=_cmd_play)

    serve_parser = sub.add_parser("serve", help="启动网页版对战")
    serve_parser.add_argument(
        "--ckpt", required=True, action="append",
        help="checkpoint 文件或 run 目录。可多次指定，页面上能切换；第一个是初始模型")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="默认只监听回环地址。页面没有认证，开发机通常是共享的 —— "
             "要从别的机器访问，优先用 SSH 端口转发",
    )
    serve_parser.add_argument("--safe-mode", action="store_true",
                              help="用规则替 AI 屏蔽禁手点。默认关闭")
    serve_parser.add_argument("--temperature", type=float, default=0.0)
    serve_parser.add_argument(
        "--reload-seconds", type=float, default=0.0,
        help="每隔这么久检查一次 run 目录里有没有更新的 checkpoint，"
             "有就热加载（0 表示不检查）。只对 run 目录生效。")
    serve_parser.set_defaults(func=_cmd_serve)

    arena_parser = sub.add_parser("arena", help="模型间/对基线的批量对局与 Elo 估计")
    arena_parser.add_argument("--a", required=True,
                              help="checkpoint 路径，或 random / greedy")
    arena_parser.add_argument("--b", required=True,
                              help="checkpoint 路径，或 random / greedy")
    arena_parser.add_argument("--games", type=int, default=200)
    arena_parser.add_argument("--batch", type=int, default=64)
    arena_parser.add_argument("--seed", type=int, default=0)
    arena_parser.add_argument("--safe-mode", action="store_true")
    arena_parser.set_defaults(func=_cmd_arena)

    show_parser = sub.add_parser("show", help="查看 checkpoint 信息")
    show_parser.add_argument("--ckpt", required=True)
    show_parser.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
