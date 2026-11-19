"""兼容 shim:驱动层真身已迁 src/finearm/arms(Phase 2)。

本包(arm/panthera/panthera_control)只做转发:import 本包任一模块会
先把仓库 src/ 引导进 sys.path,再给出 finearm.arms 的同名对象。
旧 import 路径 `panthera_control.*` 与包级导出继续有效,存量代码
(daemon 启动脚本、采集包 shim、infer/robot_interface、campaign)零改动。
回退 = git revert Phase 2(shim 反转即回退)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# finearm.core bootstrap (same pattern as Phase 1 wiring).
try:
    import finearm.arms  # noqa: F401
except ImportError:
    _deploy_src = Path(__file__).resolve().parents[3] / "src"
    if str(_deploy_src) not in sys.path:
        sys.path.insert(0, str(_deploy_src))

from finearm.arms.daemon import (  # noqa: E402,F401
    ControlClient,
    ControlClientError,
    ControlConfig,
    ControlDaemon,
    ControlLoopError,
    DaemonDisconnectedError,
    MalformedResponseError,
    StaleSnapshotError,
)

# Legacy name (pre-split). Same class object; old except-clauses keep working.
PantheraControlError = ControlLoopError


def main() -> int:
    from .cli import main as cli_main

    return cli_main()


__all__ = [
    "ControlClient",
    "ControlClientError",
    "ControlConfig",
    "ControlDaemon",
    "ControlLoopError",
    "DaemonDisconnectedError",
    "MalformedResponseError",
    "PantheraControlError",
    "StaleSnapshotError",
    "main",
]
