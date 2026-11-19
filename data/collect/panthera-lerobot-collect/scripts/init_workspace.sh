#!/usr/bin/env bash
set -euo pipefail

readonly SDK_REVISION="7f704d90cd3187327a587e9a5c355a2e1f95f987"
readonly WHEEL_PATTERN="hightorque_robot-*-cp312-cp312-linux_x86_64.whl"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 数据域根 = <deploy>/data(本工具位于 <deploy>/data/collect/panthera-lerobot-collect)。
# 数据集/日志/本机标定配置落在数据域内,与 deploy 其他目录及旧项目
# ~/panthera、project-Embodied_AGV 完全隔离;各变量可被环境变量覆盖。
# SDK 属机械臂资产,统一放 <deploy>/arm/panthera/sdk;克隆与 revision 锁定仍由本脚本负责。
DATA_ROOT="$(dirname "$(dirname -- "${REPO_ROOT}")")"
DEPLOY_ROOT="$(dirname -- "${DATA_ROOT}")"
PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${DEPLOY_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"
PANTHERA_SDK_URL="${PANTHERA_SDK_URL:-https://github.com/HighTorque-Robotics/Panthera-HT_SDK.git}"
PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${DATA_ROOT}/datasets}"
PANTHERA_PYTHON="${PANTHERA_PYTHON:-${REPO_ROOT}/.venv-lerobot/bin/python}"

mode="check"
case "${1:-}" in
  ""|--check) ;;
  --install) mode="install" ;;
  --help|-h)
    echo "Usage: $0 [--check|--install]"
    echo "  --check    verify the pinned SDK and matching wheel without changing the workspace"
    echo "  --install  create workspace directories/venv and install project dependencies"
    exit 0
    ;;
  *) echo "error: unknown option: $1" >&2; exit 2 ;;
esac

if [[ ! -d "${PANTHERA_SDK_ROOT}/.git" ]]; then
  if [[ "${mode}" != "install" ]]; then
    echo "error: clone Panthera-HT_SDK into PANTHERA_SDK_ROOT first: ${PANTHERA_SDK_ROOT}" >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "${PANTHERA_SDK_ROOT}")"
  git clone "${PANTHERA_SDK_URL}" "${PANTHERA_SDK_ROOT}"
  git -C "${PANTHERA_SDK_ROOT}" checkout --detach "${SDK_REVISION}"
fi

sdk_head="$(git -C "${PANTHERA_SDK_ROOT}" rev-parse HEAD)"
if [[ "${sdk_head}" != "${SDK_REVISION}" ]]; then
  echo "error: SDK revision ${sdk_head} does not match required ${SDK_REVISION}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PANTHERA_SDK_ROOT}" status --porcelain)" ]]; then
  echo "error: SDK worktree must be clean" >&2
  exit 2
fi

shopt -s nullglob
wheels=("${PANTHERA_SDK_ROOT}"/panthera_python/motor_whl/${WHEEL_PATTERN})
shopt -u nullglob
if [[ -x "${PANTHERA_PYTHON}" ]]; then
  python_candidate="${PANTHERA_PYTHON}"
else
  python_candidate="$(command -v python3.12 || true)"
fi
if [[ -z "${python_candidate}" ]]; then
  echo "error: PANTHERA_PYTHON is absent and no python3.12 bootstrap interpreter was found" >&2
  exit 2
fi
if ! "${python_candidate}" -c 'import platform, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and platform.machine() == "x86_64" else 1)'; then
  echo "error: Panthera requires Python 3.12 on x86_64: ${python_candidate}" >&2
  exit 2
fi
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "error: expected exactly one ${WHEEL_PATTERN} under the pinned SDK; found ${#wheels[@]}" >&2
  exit 2
fi
wheel="${wheels[0]}"

echo "data root: ${DATA_ROOT}"
echo "sdk: ${PANTHERA_SDK_ROOT} (${SDK_REVISION})"
echo "wheel: ${wheel}"
echo "python: ${PANTHERA_PYTHON}"

if [[ "${mode}" == "check" ]]; then
  exit 0
fi

mkdir -p \
  "${PANTHERA_DATA_ROOT}/lerobot" \
  "${DATA_ROOT}/logs/daemon" "${DATA_ROOT}/logs/diagnostics" \
  "$(dirname -- "${PANTHERA_PYTHON}")"

if [[ ! -x "${PANTHERA_PYTHON}" ]]; then
  "${python_candidate}" -m venv "$(dirname -- "$(dirname -- "${PANTHERA_PYTHON}")")"
fi

"${PANTHERA_PYTHON}" -m pip install --upgrade pip
"${PANTHERA_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-lerobot.txt"
"${PANTHERA_PYTHON}" -m pip install "${wheel}"
echo "workspace initialization complete"
