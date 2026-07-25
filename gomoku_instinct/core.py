"""C++ 规则内核的加载入口。

内核用 torch.utils.cpp_extension 即时编译（PyTorch 自带 pybind11 头文件，
不需要额外依赖）。编译产物缓存在 TORCH_EXTENSIONS_DIR 指向的目录，
由 scripts/docker_run.sh 指到工作目录内，因此只在首次调用时编译一次。

多进程同时首次加载时，torch 的扩展加载器自带文件锁；但为了让 actor 启动
不必互相等待，正式训练前应先跑一次 scripts/build_core.py 预编译。
"""

from __future__ import annotations

import os
import platform
import threading

_LOCK = threading.Lock()
_CORE = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSRC = os.path.join(_REPO_ROOT, "csrc")

_SOURCES = ["geometry.cpp", "renju.cpp", "bindings.cpp"]

_BASE_CFLAGS = ["-O3", "-std=c++17", "-fvisibility=hidden", "-DNDEBUG"]


def _source_paths() -> list[str]:
    return [os.path.join(_CSRC, name) for name in _SOURCES]


def load_core(verbose: bool = False):
    """编译（或复用缓存）并返回 C++ 规则内核模块。"""
    global _CORE
    with _LOCK:
        if _CORE is not None:
            return _CORE

        from torch.utils.cpp_extension import load

        sources = _source_paths()
        missing = [p for p in sources if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"缺少 C++ 源文件: {missing}")

        # aarch64 上 -mcpu=native 能带来可观的位运算加速；若工具链不认就退回基础选项。
        attempts = [_BASE_CFLAGS]
        if platform.machine() in ("aarch64", "arm64"):
            attempts.insert(0, _BASE_CFLAGS + ["-mcpu=native"])
        elif platform.machine() in ("x86_64", "AMD64"):
            attempts.insert(0, _BASE_CFLAGS + ["-march=native"])

        last_error: Exception | None = None
        for cflags in attempts:
            try:
                _CORE = load(
                    name="gi_core",
                    sources=sources,
                    extra_cflags=cflags,
                    extra_include_paths=[_CSRC],
                    verbose=verbose,
                )
                return _CORE
            except Exception as exc:  # 编译失败则降级重试
                last_error = exc

        raise RuntimeError(f"C++ 规则内核编译失败: {last_error}")


def make_rules(cfg: dict | None = None):
    """按 configs/rules.yaml 的结构构造一个 C++ Rules 对象。"""
    core = load_core()
    if cfg is None:
        return core.Rules()
    rules = cfg.get("rules", cfg)
    forb = rules.get("forbidden", {})
    return core.Rules(
        forbidden_enabled=forb.get("enabled", True),
        overline=forb.get("overline", True),
        double_four=forb.get("double_four", True),
        double_three=forb.get("double_three", True),
        five_overrides_forbidden=forb.get("five_overrides_forbidden", True),
        white_overline_wins=rules.get("white_overline_wins", True),
        recursion_depth=forb.get("recursion_depth", 64),
    )
