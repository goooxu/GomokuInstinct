#!/usr/bin/env bash
# 多卡训练编排：一张卡跑 trainer，其余卡各跑一个自博弈 actor。
#
#   ./scripts/launch_training.sh runs/renju15c          # 启动（已有进度则自动续训）
#   ./scripts/launch_training.sh runs/renju15c status   # 看谁在跑
#   ./scripts/launch_training.sh runs/renju15c stop     # 停止
#
# actor 与 trainer 之间除了文件系统没有任何耦合 —— 任一侧崩了另一侧照常跑，
# 重启后各自从最新状态接上。
#
# 注意：这里只把 CPU 核**数量**均分给各 actor，并**没有**按 NUMA 亲和做绑定
# （没有调用 numactl）。跨 socket 的内存带宽争抢因此仍然存在。
# 实测 actor 侧 96% 的时间花在 GPU 前向上、CPU 树搜索只占 3.3%，
# 所以这件事目前不是瓶颈；真要优化应先做 CUDA Graph。
#
# 进程管理走 **docker 容器名**，不走 pid：杀掉 docker run 客户端进程并不会停掉
# 容器，只按 pid 管理会导致每次「停止再启动」都叠加一层，多个 trainer 争抢
# checkpoint、多个同名 actor 互相覆盖分片，而且从进程列表上不容易看出来。
#
# 环境变量：
#   GI_BOARD_SIZE   棋盘边长（默认 15）
#   GI_TRAINER_GPU  trainer 用哪张卡（默认 0）
#   GI_ACTOR_GPUS   actor 用哪些卡（默认 "1 2 3"）
#   GI_ACTOR_THREADS 每个 actor 的搜索线程数（默认按核数自动均分）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:?用法: launch_training.sh <run-dir> [start|stop|status]}"
ACTION="${2:-start}"

BOARD_SIZE="${GI_BOARD_SIZE:-15}"
TRAINER_GPU="${GI_TRAINER_GPU:-0}"
read -r -a ACTOR_GPUS <<< "${GI_ACTOR_GPUS:-1 2 3}"

# 容器名前缀由 run 目录派生，不同 run 互不干扰
TAG="gi_$(echo "$RUN_DIR" | tr '/.' '__')"

mkdir -p "$ROOT/$RUN_DIR/logs"

running_containers() {
  docker ps --filter "name=^${TAG}_" --format '{{.Names}}'
}

stop_all() {
  local names
  names="$(docker ps -a --filter "name=^${TAG}_" --format '{{.Names}}')"
  if [ -z "$names" ]; then
    echo "没有属于 $RUN_DIR 的容器在跑。"
    return
  fi
  echo "停止：$(echo "$names" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  docker rm -f $names >/dev/null
  echo "已停止。checkpoint 与分片都在磁盘上，重新启动即可续训。"
}

case "$ACTION" in
  stop)
    stop_all
    exit 0
    ;;
  status)
    echo "run-dir $RUN_DIR"
    names="$(running_containers)"
    if [ -z "$names" ]; then
      echo "  没有容器在跑"
    else
      docker ps --filter "name=^${TAG}_" \
        --format '  {{.Names}}  运行 {{.RunningFor}}  {{.Status}}'
    fi
    exit 0
    ;;
esac

if [ -n "$(running_containers)" ]; then
  echo "已有属于 $RUN_DIR 的容器在跑："
  running_containers | sed 's/^/  /'
  echo "先执行：$0 $RUN_DIR stop"
  exit 1
fi

TOTAL_CORES="$(nproc)"
# 给 trainer 留一部分核做数据整理，其余均分给 actor
TRAINER_CORES=$(( TOTAL_CORES / 8 ))
ACTOR_THREADS="${GI_ACTOR_THREADS:-$(( (TOTAL_CORES - TRAINER_CORES) / ${#ACTOR_GPUS[@]} ))}"

echo "run-dir       $RUN_DIR"
echo "棋盘          ${BOARD_SIZE}x${BOARD_SIZE}"
echo "trainer GPU   $TRAINER_GPU"
echo "actor GPU     ${ACTOR_GPUS[*]}（每个 $ACTOR_THREADS 个搜索线程）"
echo

launch() {
  local name="$1"; shift
  local gpu="$1"; shift
  local log="$ROOT/$RUN_DIR/logs/${name}.log"
  echo "启动 $name（GPU $gpu）-> logs/${name}.log"
  GI_GPUS="device=${gpu}" GI_NAME="${TAG}_${name}" \
    nohup "$ROOT/scripts/docker_run.sh" "$@" >>"$log" 2>&1 &
  disown
}

launch trainer "$TRAINER_GPU" \
  python scripts/train.py \
    --run-dir "$RUN_DIR" \
    --board-size "$BOARD_SIZE" \
    --device cuda \
    --override external_selfplay=true

# 等 trainer 落出第一个 checkpoint，actor 才有权重可用
echo "等待 trainer 写出第一个 checkpoint……"
for _ in $(seq 1 120); do
  [ -f "$ROOT/$RUN_DIR/checkpoints/latest" ] && break
  sleep 5
done

for i in "${!ACTOR_GPUS[@]}"; do
  launch "actor${i}" "${ACTOR_GPUS[$i]}" \
    python scripts/actor.py \
      --run-dir "$RUN_DIR" \
      --actor-id "$i" \
      --board-size "$BOARD_SIZE" \
      --threads "$ACTOR_THREADS" \
      --device cuda
done

sleep 3
echo
echo "在跑的容器："
docker ps --filter "name=^${TAG}_" --format '  {{.Names}}' || true
echo
echo "查看进度：python scripts/report.py --run-dir $RUN_DIR"
echo "停止：    $0 $RUN_DIR stop"
