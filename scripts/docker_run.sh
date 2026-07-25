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

# 只有真的挂着终端时才加 -it，否则后台/管道调用会失败。
if [ -t 0 ] && [ -t 1 ]; then
  opts+=(-it)
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec docker run "${opts[@]}" "$IMAGE" "$@"
