#!/usr/bin/env bash
# ── 推理环境初始化(deploy/infer)──
# 创建本目录 .venv 并安装:
#   · openpi-client —— 直接用 deploy/models/openpi/packages/openpi-client 本地安装
#   · numpy / opencv-python —— 相机取流用
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"     # = deploy/infer
DEPLOY_ROOT="$(dirname -- "${REPO_ROOT}")"                          # = deploy
OPENPI_CLIENT_DIR="${DEPLOY_ROOT}/models/openpi/packages/openpi-client"

if [[ ! -f "${OPENPI_CLIENT_DIR}/pyproject.toml" ]]; then
  echo "error: 未找到 ${OPENPI_CLIENT_DIR}" >&2
  echo "hint: deploy/models/openpi 需先就位(deploy 项目自带该模型仓库)" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${REPO_ROOT}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${REPO_ROOT}/requirements.txt"
"${VENV}/bin/pip" install "${OPENPI_CLIENT_DIR}"

echo
echo "推理环境就绪(pi RTC / lingbot ws 共用): ${VENV}/bin/python"
echo "机械臂 daemon 用采集工具启动: deploy/data/collect/panthera-lerobot-collect/run_panthera_lerobot_daemon.sh --live --no-leader --hz 1000"
