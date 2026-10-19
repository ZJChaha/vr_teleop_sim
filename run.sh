#!/usr/bin/env bash
# ============================================================================
# 启动 VR 遥操作（MuJoCo 仿真 + WebXR）
#
#     ./run.sh                     # 正常启动（会先问：要不要采集？任务描述？）
#     VR_TELEOP_DS=0 ./run.sh      # 只看遥操作，不采集数据
#
# 解释器查找顺序:
#   1. VR_TELEOP_PY 环境变量
#   2. 仓库内 .venv/bin/python
#   3. 系统 python3（需已装 mujoco / aiohttp / numpy / pillow）
#
# 换机器人配置:
#   VR_TELEOP_PROFILE=robot_profile_myarm ./run.sh
# ============================================================================
set -u
cd "$(dirname "$0")" || exit 1

PY="${VR_TELEOP_PY:-}"
if [ -z "$PY" ]; then
    for c in "$PWD/.venv/bin/python" "$PWD/venv/bin/python"; do
        [ -x "$c" ] && { PY="$c"; break; }
    done
fi
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"; fi

if [ -z "$PY" ]; then
    echo "[run.sh] 找不到 python。请先建虚拟环境："
    echo "         python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# 依赖自检（缺什么直接说清楚，别等 ImportError 堆栈）
"$PY" - <<'EOF' || exit 1
import importlib, sys
missing = [m for m in ("mujoco", "numpy", "aiohttp", "PIL") if not importlib.util.find_spec(m)]
if missing:
    print("[run.sh] 缺少依赖: " + ", ".join(missing))
    print("         安装: %s -m pip install -r requirements.txt" % sys.executable)
    sys.exit(1)
EOF

exec "$PY" -u server.py "$@"
