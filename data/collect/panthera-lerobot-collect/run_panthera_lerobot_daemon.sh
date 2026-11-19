#!/usr/bin/env bash
# 兼容 wrapper:daemon 已迁至 deploy/arm/panthera/run_daemon.sh(2026-11),本脚本仅转发,
# 老命令/文档/SOP 不变。参数原样透传。
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../../arm/panthera/run_daemon.sh" "$@"
