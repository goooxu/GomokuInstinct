#!/usr/bin/env bash
# 多卡训练编排：一张卡跑 trainer，其余卡各跑一个自博弈 actor。
#
#   ./scripts/launch_training.sh runs/renju15
#   ./scripts/launch_training.sh runs/renju15 stop
#
# 分工按 GPU 与 CPU 的 NUMA 亲和来切：同一 socket 上的 GPU 与 CPU 核绑在一起，
# 避免跨 socket 抢内存带宽。actor 与 trainer 之间除了文件系统没有任何耦合 ——
# 任一侧崩了另一侧照常跑，重启后各自从最新状态接上。
#
# 环境变量：
#   GI_BOARD_SIZE   棋盘边长（默认 15）
#   GI_TRAINER_GPU  trainer 用哪张卡（默认 0）
#   GI_ACTOR_GPUS   actor 用哪些卡（默认 "1 2 3"）
#   GI_ACTOR_THREADS 每个 actor 的搜索线程数（默认按核数自动均分）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:?用法: launch_training.sh <run-dir> [stop]}"
ACTION="${2:-start}"

BOARD_SIZE="${GI_BOARD_SIZE:-15}"
TRAINER_GPU="${GI_TRAINER_GPU:-0}"
read -r -a ACTOR_GPUS <<< "${GI_ACTOR_GPUS:-1 2 3}"

mkdir -p "$ROOT/$RUN_DIR/logs" "$ROOT/$RUN_DIR/pids"
PID_DIR="$ROOT/$RUN_DIR/pids"

stop_all() {
  local stopped=0
  for pidfile in "$PID_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "停止 $(basename "$pidfile" .pid) (pid $pid)"
      kill "$pid" 2>/dev/null || true
      stopped=$((stopped + 1))
    fi
    rm -f "$pidfile"
  done
  echo "已停止 $stopped 个进程。checkpoint 与分片都在磁盘上，重新启动即可续训。"
}

if [ "$ACTION" = "stop" ]; then
  stop_all
  exit 0
fi

# 已经在跑就不要重复拉起
for pidfile in "$PID_DIR"/*.pid; do
  [ -e "$pidfile" ] || continue
  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "已有进程在跑（$(basename "$pidfile" .pid)）。先执行 stop。"
    exit 1
  fi
done

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
  GI_GPUS="device=${gpu}" nohup "$ROOT/scripts/docker_run.sh" "$@" \
    >>"$log" 2>&1 &
  echo $! > "$PID_DIR/${name}.pid"
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

echo
echo "全部启动完毕。查看进度："
echo "  tail -f $RUN_DIR/logs/actor0.log"
echo "  python scripts/report.py --run-dir $RUN_DIR"
echo "停止：./scripts/launch_training.sh $RUN_DIR stop"
