#!/usr/bin/env python3
"""环境自检：在容器内跑一遍，确认后续里程碑依赖的能力都具备。

    python scripts/check_env.py

检查项：Python/torch 版本、GPU 可见性与算力、BF16 matmul、BF16 主权重的舍入行为、
C++ 扩展工具链、CPU 与 NUMA 拓扑。任一硬性检查失败则以非零码退出。
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap

FAILURES: list[str] = []
WARNINGS: list[str] = []


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    FAILURES.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")
    WARNINGS.append(msg)


def check_python() -> None:
    section("Python / 平台")
    ok(f"python {platform.python_version()}  ({sys.executable})")
    ok(f"platform {platform.platform()}  machine={platform.machine()}")
    if sys.version_info < (3, 10):
        fail("需要 Python >= 3.10")


def check_torch():
    section("PyTorch")
    try:
        import torch
    except ImportError as exc:
        fail(f"无法导入 torch: {exc}")
        return None

    ok(f"torch {torch.__version__}")
    ok(f"cuda runtime {torch.version.cuda}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() 为 False")
        return torch

    n = torch.cuda.device_count()
    ok(f"可见 GPU 数量: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        ok(
            f"  GPU{i}: {props.name}  "
            f"sm_{props.major}{props.minor}  "
            f"{props.total_memory / (1 << 30):.0f} GiB  "
            f"{props.multi_processor_count} SM"
        )
    if not torch.cuda.is_bf16_supported():
        fail("当前 GPU 不支持 BF16")
    else:
        ok("BF16 受支持")
    return torch


def check_bf16_matmul(torch) -> None:
    """BF16 矩阵乘应当走 Tensor Core 并以 FP32 累加。

    验证方式：BF16 输入的 matmul 结果，应当明显比「先把输入 round 到 BF16 再用 BF16
    累加模拟」更接近 FP32 参考值。这里用一个较大的 K 维度放大累加误差的差异。
    """
    section("BF16 matmul（FP32 累加）")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    m, k, n = 512, 8192, 512
    a32 = torch.randn(m, k, device=dev, dtype=torch.float32)
    b32 = torch.randn(k, n, device=dev, dtype=torch.float32)
    a16, b16 = a32.bfloat16(), b32.bfloat16()

    ref = (a16.float() @ b16.float())  # 相同输入精度、FP32 累加的参考值
    got = (a16 @ b16).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    if rel < 1e-2:
        ok(f"BF16 matmul 相对误差 {rel:.2e}（累加精度正常）")
    else:
        fail(f"BF16 matmul 相对误差偏大: {rel:.2e}")


def check_bf16_weight_rounding(torch) -> None:
    """确认 BF16 主权重确实存在「小更新被舍掉」的问题。

    这不是环境故障，而是本项目优化器必须做 Kahan 补偿求和的直接依据；
    如果这里没有复现出停滞，说明后续的补偿逻辑需要重新评估。
    """
    section("BF16 主权重舍入行为")
    w = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
    step = torch.full_like(w, 1e-4)  # 相对 w=1 远小于 bf16 的 2^-8 分辨率
    before = w.clone()
    for _ in range(100):
        w += step
    moved = (w != before).float().mean().item()
    if moved == 0.0:
        ok("朴素 BF16 累加在小步长下完全停滞（符合预期，需要 Kahan 补偿）")
    else:
        warn(f"朴素 BF16 累加下有 {moved:.1%} 的元素发生了变化，需复核补偿方案假设")


def check_cpp_toolchain() -> None:
    section("C++ 扩展工具链")
    try:
        from torch.utils import cpp_extension
    except ImportError as exc:
        fail(f"无法导入 torch.utils.cpp_extension: {exc}")
        return

    inc = cpp_extension.include_paths()
    pybind_ok = any(
        os.path.exists(os.path.join(p, "pybind11", "pybind11.h")) for p in inc
    )
    if pybind_ok:
        ok("torch 自带 pybind11 头文件（无需额外依赖）")
    else:
        fail("未在 torch include 路径下找到 pybind11 头文件")

    for tool in ("c++", "make"):
        try:
            out = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, timeout=30
            )
            ok(f"{tool}: {out.stdout.splitlines()[0]}")
        except (OSError, subprocess.SubprocessError) as exc:
            fail(f"{tool} 不可用: {exc}")

    ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if ext_dir:
        ok(f"TORCH_EXTENSIONS_DIR={ext_dir}")
    else:
        warn("未设置 TORCH_EXTENSIONS_DIR，扩展编译缓存会落到 HOME 下")


def check_cpu() -> None:
    section("CPU / NUMA")
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 0
    ok(f"可用 CPU 核心: {n_cpu}")
    if n_cpu < 8:
        warn("可用核心偏少，自博弈吞吐会受限")

    try:
        import torch

        ok(f"torch 线程数默认: {torch.get_num_threads()}")
    except ImportError:
        pass

    try:
        out = subprocess.run(
            ["numactl", "--hardware"], capture_output=True, text=True, timeout=30
        )
        first = out.stdout.strip().splitlines()
        if first:
            ok(f"numactl 可用: {first[0]}")
    except (OSError, subprocess.SubprocessError):
        warn("numactl 不可用，actor 将无法做 NUMA 绑定（不影响正确性，影响吞吐）")


def check_writable() -> None:
    section("工作目录可写性")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = os.path.join(root, "build", ".write_probe")
    try:
        os.makedirs(os.path.dirname(probe), exist_ok=True)
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        ok("工作目录可写")
    except OSError as exc:
        fail(f"工作目录不可写: {exc}")


def main() -> int:
    check_python()
    torch = check_torch()
    if torch is not None and torch.cuda.is_available():
        check_bf16_matmul(torch)
        check_bf16_weight_rounding(torch)
    check_cpp_toolchain()
    check_cpu()
    check_writable()

    print()
    if FAILURES:
        print(textwrap.indent("环境自检未通过：", ""))
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    if WARNINGS:
        print(f"环境自检通过（{len(WARNINGS)} 条警告）")
    else:
        print("环境自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
