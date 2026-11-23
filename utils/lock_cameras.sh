#!/usr/bin/env bash
# ── Panthera 双相机状态恢复 + 读回验证(采集 & 推理共用,唯一事实来源)──
# 本脚本位于仓库根 utils/,是采集(plugins/panthera-lerobot-collect)
# 与推理(src/finearm/infer/robot_interface.py)共用的唯一权威副本——相机档案变更只改这里。
# 目标状态 = 采集 bolt_test2(2026-11-07 14:48,参考数据集)时两台相机的
# 确切参数。run_panthera_lerobot_collect.sh(采集)与
# src/finearm/infer/robot_interface.py(推理)启动时会自动调用本脚本:
# 逐项设置、逐项读回验证,任一不匹配自动重试一次,仍失败则退出非 0。
#
# 用法:
#   bash lock_cameras.sh           # 设置 + 读回验证(每次采集/推理前跑)
#   bash lock_cameras.sh --check   # 只读回验证、不设置(巡检用)
#   PANTHERA_CAM_LOCK_SKIP=1       # 完全跳过(仅调试)
#
# 状态档案(bolt_test2 采集时刻,2026-11-07 14:45-14:48 标定/确认):
#   wrist(HD USB CAMERA):
#     - 手动曝光 36(3.6ms)+ gain 125:该相机曝光按粗档量化(≤39/42–78/≥90),
#       36 档把白桌面高光溢出从 7.9% 压到 0.02%,gain 补回旧 78 档亮度(88≈93);
#       锁手动是因自动曝光暗光掉到 ~25fps 且延迟 +8ms(2026-11-06 实测)
#     - 白平衡 auto(与 bolt_test2 一致;手动 4500K 也几乎复现 auto,但以参考状态为准)
#   front(MX Brio):全部 auto(与 bolt_test2 一致)——AE 亮度 127.8/0% 溢出、
#     AWB episode 内稳定 ±0.3%、AF 锁定后 lapvar≈12000(与定焦 96/数据集同水平)。
#     注意 auto 模式开流后需 ~10s 收敛(2026-11-07 实测:从手动焦距切回 AF 偶尔
#     >8s 才合焦,收敛前清晰度可低至 ~90):采集脚本锁完即跑 120 帧 fps 门禁
#     (部分热身),正式录制的确认/准备阶段足以补足;推理端 go_home 阶段同理。
#     若推理实测 front 长期模糊,改回定焦:front 参数换成 focus_automatic_continuous=0
#     focus_absolute=96(2026-11-07 实测与数据集清晰度一致)。
#   设置顺序有依赖(勿调整):wrist manual 模式须先于曝光时间设置。
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
if [[ "${PANTHERA_CAM_LOCK_SKIP:-0}" == "1" ]]; then
  echo "[cam-lock] PANTHERA_CAM_LOCK_SKIP=1,跳过相机状态恢复"
  exit 0
fi

WRIST_NAME="HD USB CAMERA"
FRONT_NAME="MX Brio"

WRIST_PARAMS=(auto_exposure=1 exposure_time_absolute=36 gain=125
              white_balance_automatic=1)
FRONT_PARAMS=(auto_exposure=3 gain=0 white_balance_automatic=1
              focus_automatic_continuous=1)

find_dev() { awk -v n="$1" 'index($0, n){f=1; next} f && /\/dev\/video/{print $1; exit}' \
              <<<"$(v4l2-ctl --list-devices 2>/dev/null)"; }

get_val() { v4l2-ctl -d "$1" --get-ctrl="$2" 2>/dev/null \
              | awk -F': ' '{print $2}' | awk '{print $1}'; }

apply() {
  local dev="$1"; shift
  local p
  for p in "$@"; do
    v4l2-ctl -d "${dev}" --set-ctrl="${p}" 2>/dev/null || true  # 失败交给验证兜底
  done
}

verify() {
  local dev="$1" name="$2"; shift 2
  local p k want got bad=0
  for p in "$@"; do
    k="${p%%=*}"; want="${p#*=}"; got="$(get_val "${dev}" "${k}")"
    if [[ "${got}" == "${want}" ]]; then
      echo "[cam-lock]   ✓ ${name} ${k}=${got}"
    else
      echo "[cam-lock]   ✗ ${name} ${k}: 期望=${want} 实际=${got:-读取失败}" >&2
      bad=1
    fi
  done
  return "${bad}"
}

lock_one() {
  local name="$1"; shift
  local params=("$@")
  local dev; dev="$(find_dev "${name}")"
  if [[ -z "${dev}" ]]; then
    echo "[cam-lock] ⚠ 未找到相机 ${name},跳过(若本次要用请先排查接线)" >&2
    return 0
  fi
  echo "[cam-lock] ${name} @ ${dev}$([[ ${CHECK_ONLY} -eq 1 ]] && echo ' [--check 只验证]')"
  if [[ ${CHECK_ONLY} -eq 0 ]]; then
    apply "${dev}" "${params[@]}"
    sleep 0.2
  fi
  if ! verify "${dev}" "${name}" "${params[@]}"; then
    if [[ ${CHECK_ONLY} -eq 1 ]]; then
      echo "[cam-lock] error: ${name} 参数与档案不一致(--check 不自动修复)" >&2
      return 1
    fi
    echo "[cam-lock] 首次验证有偏差,重试一次..." >&2
    apply "${dev}" "${params[@]}"
    sleep 0.3
    verify "${dev}" "${name}" "${params[@]}" || { echo "[cam-lock] error: ${name} 参数锁定失败" >&2; return 1; }
  fi
}

fail=0
lock_one "${WRIST_NAME}" "${WRIST_PARAMS[@]}" || fail=1
lock_one "${FRONT_NAME}" "${FRONT_PARAMS[@]}" || fail=1
if [[ "${fail}" -eq 0 ]]; then
  echo "[cam-lock] ✓ 双相机已恢复 bolt_test2 参考状态(front auto 模式开流 ~3-5s 后收敛)"
else
  echo "[cam-lock] ✗ 存在不一致,拒绝继续(采集/推理前必须修复)" >&2
fi
exit "${fail}"
