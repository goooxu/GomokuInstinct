#!/usr/bin/env bash
# 容器一次性初始化：装上镜像里没有的少量依赖。
#
#   ./scripts/docker_run.sh scripts/setup_container.sh
#
# 依赖装到 $HOME/.local（docker_run.sh 已把 HOME 指向工作目录内），
# 因此只需跑一次，后续容器直接复用，不会污染镜像也不会写到工作目录之外。
set -euo pipefail

echo "== 安装开发依赖到 user site =="
python -m pip install --user --no-warn-script-location \
    pytest \
    pyyaml

echo
echo "== 版本确认 =="
python - <<'PY'
import importlib

for name in ("numpy", "yaml", "pytest", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:8s} {getattr(mod, '__version__', '?')}")
    except ImportError as exc:
        print(f"  {name:8s} 缺失: {exc}")
PY

echo
echo "初始化完成。"
