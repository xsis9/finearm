"""Panthera 机械臂控制服务(驱动层)。

daemon(1000Hz 控制环、hold/policy/teleop/freedrive 模式、安全兜底)+ IPC 客户端。
被两类客户端共用:数据采集(deploy/data/collect)与推理(deploy/infer)。
2026-11 从采集工具 panthera_lerobot_control 抽离,旧包名保留兼容 shim。
"""

from .controller import ControlConfig, ControlDaemon, PantheraControlError
from .ipc import (
    ControlClient,
    ControlClientError,
    DaemonDisconnectedError,
    MalformedResponseError,
    StaleSnapshotError,
)


def main() -> int:
    from .cli import main as cli_main

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
