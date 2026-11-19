"""Arm-neutral control daemon: controller loop, IPC protocol, CLI.

``finearm-daemon --arm <id>`` runs this daemon against the named arm plugin
(``finearm/arms/<id>/``). Pure standard library: the daemon itself has no
numpy/yaml dependency; vendor SDKs are loaded lazily by arm adapters.
"""

from .adapters import AdapterError, ArmState, MockRobotAdapter, RobotAdapter
from .controller import ControlConfig, ControlDaemon, ControlLoopError
from .gains import ControlGains
from .ipc import (
    ControlClient,
    ControlClientError,
    ControlServer,
    DaemonDisconnectedError,
    MalformedResponseError,
    StaleSnapshotError,
)

# Legacy alias: the daemon predates the arm-neutral split under this name.
PantheraControlError = ControlLoopError


def main() -> int:
    from .cli import main as cli_main

    return cli_main()


__all__ = [
    "AdapterError",
    "ArmState",
    "ControlClient",
    "ControlClientError",
    "ControlConfig",
    "ControlDaemon",
    "ControlGains",
    "ControlLoopError",
    "ControlServer",
    "DaemonDisconnectedError",
    "MalformedResponseError",
    "MockRobotAdapter",
    "PantheraControlError",
    "RobotAdapter",
    "StaleSnapshotError",
    "main",
]
