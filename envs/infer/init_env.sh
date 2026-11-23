#!/usr/bin/env bash
# ── 推理环境初始化(envs/infer)──
# 创建本目录 .venv 并安装:
#   · openpi-client —— 直接用 <repo>/models/openpi/packages/openpi-client 本地安装
#   · numpy / opencv-python —— 相机取流用
set -euo pipefail

ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"       # = <repo>/envs/infer
REPO_ROOT="$(dirname "$(dirname -- "${ENV_DIR}")")"                 # = 仓库根
OPENPI_CLIENT_DIR="${REPO_ROOT}/models/openpi/packages/openpi-client"

if [[ ! -f "${OPENPI_CLIENT_DIR}/pyproject.toml" ]]; then
  echo "error: 未找到 ${OPENPI_CLIENT_DIR}" >&2
  echo "hint: models/openpi 需先就位(仓库自带该模型仓库)" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${ENV_DIR}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${ENV_DIR}/requirements.txt"
"${VENV}/bin/pip" install "${OPENPI_CLIENT_DIR}"

echo
echo "推理环境就绪(pi RTC / lingbot ws 共用): ${VENV}/bin/python"
echo "机械臂 daemon(仓库根执行): bash arm/panthera/run_daemon.sh --live --no-leader --hz 1000"
