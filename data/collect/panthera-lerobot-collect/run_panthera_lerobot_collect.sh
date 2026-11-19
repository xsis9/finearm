#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本工具位于 <deploy>/data/collect/panthera-lerobot-collect → 数据域根 = <deploy>/data。
# 数据集、日志等落在本数据域内,与 deploy 其他目录(train/infer/models)及旧项目
# ~/panthera、project-Embodied_AGV 完全隔离;可用环境变量逐项覆盖。
# SDK 属机械臂资产,默认在 <deploy>/arm/panthera/sdk。
DATA_ROOT="$(dirname "$(dirname "${ROOT_DIR}")")"
DEPLOY_ROOT="$(dirname -- "${DATA_ROOT}")"

export PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${DEPLOY_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"
export PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${DATA_ROOT}/datasets}"

SUPPORTED_LEROBOT_VERSION="0.5.1"

python_has_lerobot_runtime() {
  local python_path="$1"
  local record_path
  record_path="$(dirname "${python_path}")/lerobot-record"
  [[ -x "${python_path}" && -x "${record_path}" ]] || return 1
  PYTHONNOUSERSITE=1 "${python_path}" -c \
    'import importlib, importlib.metadata as m, sys; importlib.import_module("lerobot"); sys.exit(0 if m.version("lerobot") == "0.5.1" else 1)' \
    >/dev/null 2>&1
}

if [[ -n "${PANTHERA_PYTHON:-}" ]]; then
  if [[ ! -x "${PANTHERA_PYTHON}" ]]; then
    echo "error: PANTHERA_PYTHON is not executable: ${PANTHERA_PYTHON}" >&2
    exit 2
  fi
  if [[ ! -x "$(dirname "${PANTHERA_PYTHON}")/lerobot-record" ]]; then
    echo "error: PANTHERA_PYTHON must have an executable sibling lerobot-record: ${PANTHERA_PYTHON}" >&2
    exit 2
  fi
  if ! python_has_lerobot_runtime "${PANTHERA_PYTHON}"; then
    echo "error: PANTHERA_PYTHON must provide lerobot==${SUPPORTED_LEROBOT_VERSION}: ${PANTHERA_PYTHON}" >&2
    exit 2
  fi
  PYTHON="${PANTHERA_PYTHON}"
else
  # 只认本工具自己的 venv(init_workspace.sh --install 创建),不回退外部环境。
  candidates=(
    "${ROOT_DIR}/.venv-lerobot/bin/python"
    "${ROOT_DIR}/.venv/bin/python"
  )
  PYTHON=""
  for candidate in "${candidates[@]}"; do
    if python_has_lerobot_runtime "${candidate}"; then
      PYTHON="${candidate}"
      break
    fi
  done
  if [[ -z "${PYTHON}" ]]; then
    echo "error: 未找到本工具的采集环境(需 python + lerobot-record + lerobot==${SUPPORTED_LEROBOT_VERSION})" >&2
    echo "hint: 先运行 ./scripts/init_workspace.sh --install 创建 ${ROOT_DIR}/.venv-lerobot" >&2
    exit 2
  fi
fi

PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

PATH="$(dirname "${PYTHON}"):${PATH}"
export PATH

# ROS setup scripts can prepend their own packages to PYTHONPATH. The collect
# wrapper should import this repository and the active Python environment only.
PYTHONPATH="${ROOT_DIR}"
export PYTHONPATH

collect_args=()
live=0
live_seen=0
mock_seen=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      if [[ "${mock_seen}" -eq 1 ]]; then
        echo "error: --live and --mock cannot be used together" >&2
        exit 2
      fi
      live_seen=1
      live=1
      shift
      ;;
    --mock)
      if [[ "${live_seen}" -eq 1 ]]; then
        echo "error: --live and --mock cannot be used together" >&2
        exit 2
      fi
      mock_seen=1
      live=0
      shift
      ;;
    --hz)
      if [[ $# -lt 2 ]]; then
        echo "error: --hz requires a value" >&2
        exit 2
      fi
      collect_args+=(--hz "$2")
      shift 2
      ;;
    --socket)
      if [[ $# -lt 2 ]]; then
        echo "error: --socket requires a value" >&2
        exit 2
      fi
      collect_args+=(--socket "$2")
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage:
  ./run_panthera_lerobot_collect.sh [--live] [collection options]

Examples:
  ./run_panthera_lerobot_collect.sh --mock --auto-start --no-cameras --repo-id local/test
  ./run_panthera_lerobot_collect.sh --live --select-cameras --repo-id local/panthera_demo --task "pick cube"

Friendly flags:
  --live        connect to Panthera hardware; omitted defaults to mock mode
  --mock        run without hardware
  --hz HZ       Panthera daemon control loop rate
  --socket PATH Unix socket shared by daemon, robot, and teleop adapters

Common collection options:
  --repo-id ID
  --task TEXT
  --episodes N
  --episode-seconds S
  --reset-seconds S
  --fps FPS
  --camera NAME=SOURCE
  --select-cameras
  --auto-start
  --no-status-ui
  --voice-prompts
  --print-command

All other arguments are passed to panthera-lr-collect.
USAGE
      exit 0
      ;;
    *)
      collect_args+=("$1")
      shift
      ;;
  esac
done

# ── 相机预检(仅 --live;LEROBOT_COLLECT_SKIP_CAMERA_PREFLIGHT=1 可跳过) ──
# ① 双相机状态恢复:调 deploy/utils/lock_cameras.sh(采集 & 推理共用的唯一权威
#    副本)——恢复到参考数据集 bolt_test2 采集时刻的确切参数(wrist 手动曝光
#    36/gain125,front 全 auto),逐项设置并读回验证,任一不一致即拒采,保证训练
#    与推理的相机外观一致。UVC 控制重插/断电后复位,故每次采集前执行。
# ② 双开 fps 门禁:<29fps 或丢帧直接拒采,避免带病采集(兼作 auto 模式热身)。
camera_preflight() {
  local list wrist="" front=""
  list="$(v4l2-ctl --list-devices 2>/dev/null)" || { echo "warn: v4l2-ctl 不可用,跳过相机预检" >&2; return 0; }
  wrist="$(awk '/HD USB CAMERA/{f=1;next} f&&/\/dev\/video/{print $1;exit}' <<<"${list}")"
  front="$(awk '/MX Brio/{f=1;next} f&&/\/dev\/video/{print $1;exit}' <<<"${list}")"
  if [[ -z "${wrist}" ]]; then
    echo "warn: 未找到 wrist 相机(HD USB CAMERA),跳过预检——若本次要用相机请先排查接线" >&2
    return 0
  fi
  echo "[preflight] wrist=${wrist} front=${front:-未找到}"
  # 双相机锁定+验证(与推理端共用 deploy/utils/lock_cameras.sh 档案)
  local cam_lock="${PANTHERA_CAM_LOCK:-$(dirname "${DATA_ROOT}")/utils/lock_cameras.sh}"
  if [[ ! -f "${cam_lock}" ]]; then
    echo "error: 相机锁定脚本不存在: ${cam_lock}(deploy/utils/lock_cameras.sh;可用 PANTHERA_CAM_LOCK 指定)" >&2
    exit 2
  fi
  if ! bash "${cam_lock}"; then
    echo "error: 相机参数锁定/验证失败——采集必须与推理使用相同相机参数,先修复相机再采集" >&2
    exit 2
  fi
  local lat_tool="${HOME}/camera_latency_test/v4l2_latency"
  if [[ ! -x "${lat_tool}" ]]; then
    echo "warn: 缺 ${lat_tool},跳过 fps 门禁" >&2
    return 0
  fi
  local out pids=() f fail=0
  out="$(mktemp -d)"
  "${lat_tool}" "${wrist}" 640 480 MJPG 30 120 30 >"${out}/wrist.txt" 2>&1 & pids+=($!)
  if [[ -n "${front}" ]]; then
    "${lat_tool}" "${front}" 640 480 MJPG 30 120 30 >"${out}/front.txt" 2>&1 & pids+=($!)
  fi
  for p in "${pids[@]}"; do wait "${p}" || true; done
  for f in "${out}"/*.txt; do
    local name fps drops
    name="$(basename "${f}" .txt)"
    fps="$(grep -oP 'actual frame rate: \K[0-9.]+' "${f}" | head -1 || true)"
    drops="$(grep -oP 'dropped/duplicated sequence steps: \K[0-9]+' "${f}" | head -1 || true)"
    echo "[preflight] ${name}: ${fps:-无输出}fps drops=${drops:-?}"
    if [[ -z "${fps}" ]] || awk -v v="${fps}" 'BEGIN{exit !(v<29)}' || [[ "${drops:-1}" != "0" ]]; then
      echo "error: ${name} 预检不达标(需 ≥29fps 且零丢帧)。常见原因:曝光未锁/USB 带宽争抢/相机卡死(重插)" >&2
      fail=1
    fi
  done
  rm -rf "${out}"
  [[ "${fail}" -eq 0 ]] || exit 2
  echo "[preflight] ✓ 相机预检通过"
}

if [[ "${live}" -eq 1 ]]; then
  if [[ "${LEROBOT_COLLECT_SKIP_CAMERA_PREFLIGHT:-0}" != "1" ]]; then
    camera_preflight
  fi
  collect_args=(--live "${collect_args[@]}")
else
  collect_args=(--mock "${collect_args[@]}")
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" -m panthera_lerobot_control.collect_cli "${collect_args[@]}"
