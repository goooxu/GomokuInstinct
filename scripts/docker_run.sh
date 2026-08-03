#!/usr/bin/env bash
# 在容器内运行命令。容器内保持与宿主完全相同的工作目录路径。
#
#   ./scripts/docker_run.sh                 # 交互式 shell
#   ./scripts/docker_run.sh python -V       # 跑一条命令
#
# 环境变量：
#   GI_IMAGE      容器镜像（默认 nvcr.io/nvidia/pytorch:26.06-py3）
#   GI_GPUS       暴露给容器的 GPU（默认 all，可写 '"device=0,1"'）
#   GI_AS_ROOT    设为 1 则以 root 运行（默认以当前用户运行，避免产生 root 属主文件）
#   GI_NAME       给容器起名字。长任务必须起名 —— 杀掉 docker run 客户端进程
#                 并不会停掉容器，只有 docker stop <名字> 才管用。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GI_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
GPUS="${GI_GPUS:-all}"

# HOME 与各类编译缓存都放进工作目录，保证容器可重复构建，且不往工作目录之外写文件。
CONTAINER_HOME="$ROOT/runs/.container_home"
EXT_DIR="$ROOT/build/torch_extensions"
INDUCTOR_DIR="$ROOT/build/inductor_cache"
TRITON_DIR="$ROOT/build/triton_cache"
mkdir -p "$CONTAINER_HOME" "$EXT_DIR" "$INDUCTOR_DIR" "$TRITON_DIR"

# 这些缓存内部记录的是**绝对路径**（ninja 构建文件、inductor 缓存索引等）。
# 项目目录一旦被移动，旧缓存就会指向不存在的路径，而且报出来的错完全看不出根因
# —— 实测是在 torch/_inductor/remote_cache.py 里对着旧路径 makedirs 时抛
# PermissionError。所以记下缓存是为哪个根目录建的，根目录一变就整个清掉重建。
# 这个标记会被并发读写：多卡编排一次拉起四个容器，四个 docker_run.sh 几乎同时跑到这里。
# 原来的写法每次都无条件 `> $ROOT_MARKER`（先截断再写），别的进程恰好在那一瞬 cat
# 就会读到空内容、判定"根目录变了"，进而 rm -rf 掉**正在被其它容器使用的**编译缓存。
# 实际发生过一次：一个 actor 因此把缓存删到一半（rm 撞上并发写返回非零，
# 又被 set -e 直接带走），整个进程没起来，而 launch 脚本照样报告"已启动"。
#
# 两处修法：内容已经正确就**根本不写**（消除截断窗口）；真要更新时先写临时文件再
# 原子 rename。清理失败也不再中止脚本 —— 缓存清不掉最多是慢一点，不该让容器起不来。
ROOT_MARKER="$ROOT/build/.cache_root"
marker_now=""
[ -f "$ROOT_MARKER" ] && marker_now="$(cat "$ROOT_MARKER" 2>/dev/null || true)"
if [ -n "$marker_now" ] && [ "$marker_now" != "$ROOT" ]; then
  echo "检测到项目根目录已变更（原 $marker_now），清理编译缓存……" >&2
  rm -rf "$EXT_DIR" "$INDUCTOR_DIR" "$TRITON_DIR" || true
  mkdir -p "$EXT_DIR" "$INDUCTOR_DIR" "$TRITON_DIR"
fi
if [ "$marker_now" != "$ROOT" ]; then
  printf '%s\n' "$ROOT" > "$ROOT_MARKER.$$" && mv -f "$ROOT_MARKER.$$" "$ROOT_MARKER"
fi

opts=(
  --rm
  --gpus "$GPUS"
  --ipc=host
  --network=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$ROOT:$ROOT"
  -w "$ROOT"
  -e HOME="$CONTAINER_HOME"
  -e PATH="$CONTAINER_HOME/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  -e PYTHONPATH="$ROOT"
  -e TORCH_EXTENSIONS_DIR="$EXT_DIR"
  -e TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_DIR"
  -e TRITON_CACHE_DIR="$TRITON_DIR"
  -e PYTHONDONTWRITEBYTECODE=1
  # 长任务的 stdout 被重定向到日志文件时是块缓冲的，几分钟内看不到任何新输出。
  # 后果不只是"看着急"——日志是追加写的，读到的会是上一次运行留下的旧行，
  # 足以让人把一次正常的续训误判成失败。
  -e PYTHONUNBUFFERED=1
  # 容器以宿主 uid 运行，而该 uid 在容器的 /etc/passwd 里没有条目，
  # getpass.getuser() 会抛 KeyError，导致 torch._dynamo 首次导入失败、
  # 重试时又撞上「重复注册」——torch.compile 与 torch.optim 都会被带塌。
  # getpass 优先读这两个环境变量，给个固定名字即可绕开。
  -e USER=gi
  -e LOGNAME=gi
)

if [ "${GI_AS_ROOT:-0}" != "1" ]; then
  opts+=(-u "$(id -u):$(id -g)")
fi

if [ -n "${GI_NAME:-}" ]; then
  opts+=(--name "$GI_NAME")
fi

# -i 始终要加：不加的话 docker 根本不会把 stdin 接进容器，
# 管道喂输入（例如把着法喂给对战 CLI）会立刻读到 EOF。
# -t 只在真的挂着终端时加，否则后台调用会失败。
opts+=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  opts+=(-t)
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec docker run "${opts[@]}" "$IMAGE" "$@"
