#!/usr/bin/env bash
# ── 机械臂驱动环境初始化(envs/arm):建本目录 .venv 并安装 SDK 轮子 + SDK Python 依赖 ──
# daemon 本体只依赖标准库 + SDK(Panthera_lib 从 SDK scripts 目录加载,
# 底层电机库来自 motor_whl 轮子,Python 层需 pyyaml/numpy/scipy/pinocchio),
# 与采集环境的 lerobot 完全解耦。
# SDK 属机械臂资产,位于 <repo>/arm/panthera/sdk/Panthera-HT_SDK;克隆/校验由采集
# 插件的 scripts/init_workspace.sh 负责。
set -euo pipefail

ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"       # = <repo>/envs/arm
REPO_ROOT="$(dirname "$(dirname -- "${ENV_DIR}")")"                 # = 仓库根
ARM_ROOT="${REPO_ROOT}/arm/panthera"
SDK_ROOT="${PANTHERA_SDK_ROOT:-${ARM_ROOT}/sdk/Panthera-HT_SDK}"
VENV="${ENV_DIR}/.venv"
WHEEL_PATTERN="hightorque_robot-*-cp312-cp312-linux_x86_64.whl"

if [[ ! -d "${SDK_ROOT}/panthera_python" ]]; then
  echo "error: 未找到 SDK: ${SDK_ROOT}" >&2
  echo "hint: 先运行 ${REPO_ROOT}/plugins/panthera-lerobot-collect/scripts/init_workspace.sh --install" >&2
  exit 2
fi

shopt -s nullglob
wheels=("${SDK_ROOT}"/panthera_python/motor_whl/${WHEEL_PATTERN})
shopt -u nullglob
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "error: expected exactly one ${WHEEL_PATTERN} under ${SDK_ROOT}; found ${#wheels[@]}" >&2
  exit 2
fi

python_candidate="${PYTHON_BIN:-$(command -v python3.12 || command -v python3)}"
if ! "${python_candidate}" -c 'import platform, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and platform.machine() == "x86_64" else 1)'; then
  echo "error: 驱动需要 Python 3.12 x86_64(轮子 ABI 限制): ${python_candidate}" >&2
  exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${python_candidate}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install "${wheels[0]}"
# Panthera_lib 的 Python 依赖(版本对齐采集环境 .venv-lerobot;numpy 已随轮子装好)
"${VENV}/bin/python" -m pip install pyyaml==6.0.3 scipy==1.18.1 pin==4.1.0

echo
echo "机械臂驱动环境就绪: ${VENV}/bin/python"
echo "启动 daemon(仓库根执行): bash arm/panthera/run_daemon.sh --live --hz 1000 ...(自动优先使用本 venv)"
