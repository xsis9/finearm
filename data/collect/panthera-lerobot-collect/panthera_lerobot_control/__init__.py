"""兼容 shim:驱动层真身 = src/finearm/arms(2026-11 抽离,Phase 2 再迁 finearm)。

本包现在只保留采集域自己的模块(collect_cli / sample_coordinator),
驱动层经由本目录同名 shim → arm/panthera/panthera_control(再转发)到达
finearm.arms;旧 import 路径 `panthera_lerobot_control.*` 继续有效。
新代码请直接使用 `finearm.arms.*`。
"""
import pathlib
import sys

_ARM = pathlib.Path(__file__).resolve().parents[4] / "arm" / "panthera"
if str(_ARM) not in sys.path:
    sys.path.insert(0, str(_ARM))

from panthera_control.controller import ControlConfig, ControlDaemon, PantheraControlError  # noqa: E402,F401
from panthera_control.ipc import (  # noqa: E402,F401
    ControlClient,
    ControlClientError,
    DaemonDisconnectedError,
    MalformedResponseError,
    StaleSnapshotError,
)


def main() -> int:
    from panthera_control.cli import main as cli_main

    return cli_main()


__all__ = [
    "ControlClient",
    "ControlClientError",
    "ControlConfig",
    "ControlDaemon",
    "DaemonDisconnectedError",
    "MalformedResponseError",
    "PantheraControlError",
    "StaleSnapshotError",
    "main",
]
