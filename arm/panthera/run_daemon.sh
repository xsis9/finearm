#!/usr/bin/env bash
# ── Panthera 机械臂 daemon 启动(驱动层,与采集/推理解耦后共用的入口)──
# 2026-11 从 data/collect/panthera-lerobot-collect/run_panthera_lerobot_daemon.sh
# 迁入 deploy/arm/;旧路径保留同名薄 wrapper 转发到这里,老命令不破。
set -euo pipefail

ARM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"     # = deploy/arm/panthera
DEPLOY_ROOT="$(dirname "$(dirname -- "${ARM_ROOT}")")"             # = deploy
COLLECT_TOOL="${DEPLOY_ROOT}/data/collect/panthera-lerobot-collect"

# SDK(机械臂资产)随驱动放在本目录 sdk/ 下;克隆/校验仍由采集工具
# scripts/init_workspace.sh 负责。
export PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${ARM_ROOT}/sdk/Panthera-HT_SDK}"

if [[ -n "${PANTHERA_PYTHON:-}" ]]; then
  if [[ ! -x "${PANTHERA_PYTHON}" ]]; then
    echo "error: PANTHERA_PYTHON is not executable: ${PANTHERA_PYTHON}" >&2
    exit 2
  fi
  PYTHON="${PANTHERA_PYTHON}"
elif [[ -x "${ARM_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ARM_ROOT}/.venv/bin/python"
elif [[ -x "${COLLECT_TOOL}/.venv-lerobot/bin/python" ]]; then
  PYTHON="${COLLECT_TOOL}/.venv-lerobot/bin/python"
elif [[ -x "${COLLECT_TOOL}/.venv/bin/python" ]]; then
  PYTHON="${COLLECT_TOOL}/.venv/bin/python"
else
  echo "error: 未找到可用的 python 环境" >&2
  echo "hint: 先运行 ${COLLECT_TOOL}/scripts/init_workspace.sh --install 创建采集环境" >&2
  echo "      (或建 ${ARM_ROOT}/.venv —— daemon 本体只依赖 SDK 轮子)" >&2
  exit 2
fi

PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

# ROS setup scripts can prepend their own packages to PYTHONPATH. The daemon
# should import this repository and the active Python environment instead.
PYTHONPATH=""
export PYTHONPATH

daemon_args=()
live=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      live=1
      shift
      ;;
    --mock)
      live=0
      shift
      ;;
    --hz)
      if [[ $# -lt 2 ]]; then
        echo "error: --hz requires a value" >&2
        exit 2
      fi
      daemon_args+=(--loop-hz "$2")
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage:
  ./run_daemon.sh [--live] [--hz HZ] [daemon options]

Examples:
  ./run_daemon.sh --mock --hz 1000
  ./run_daemon.sh --live --hz 1000

Friendly flags:
  --live      connect to Panthera hardware; omitted defaults to mock mode
  --mock      run without hardware
  --hz HZ     translated to panthera-control-daemon --loop-hz HZ

Daemon options passed through:
  --socket PATH
  --sdk-root PATH
  --no-leader        follower hardware only (Quest/policy teleop); teleop mode rejected
  --leader-config PATH
  --follower-config PATH
  --initial-control-mode MODE  start mode: hold|teleop|policy|freedrive (default hold)
  --no-park          disable the exit park move (live mode ramps back to the startup pose before braking)
  --park-joints J    park target 'j1,...,j6' (default: startup pose)
  --park-move-s S    park ramp duration (default 4)
  --no-leader-park   disable the leader exit park (live mode ramps the leader to zero joints before release)
  --leader-park-move-s S  leader park ramp duration (default 4)
USAGE
      exit 0
      ;;
    *)
      daemon_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${live}" -eq 0 ]]; then
  daemon_args=(--mock "${daemon_args[@]}")
fi

cd "${ARM_ROOT}"
exec "${PYTHON}" -m panthera_control.cli "${daemon_args[@]}"
