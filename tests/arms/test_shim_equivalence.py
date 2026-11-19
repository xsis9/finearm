"""Legacy shim equivalence: panthera_control.* must behave exactly as before.

The campaign entry points (run_daemon.sh → python -m panthera_control.cli,
infer/robot_interface.py, the collect package shims) import through
``panthera_control``; this suite locks that surface to the finearm.arms
implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_DIR = _REPO_ROOT / "arm" / "panthera"
if str(_ARM_DIR) not in sys.path:
    sys.path.insert(0, str(_ARM_DIR))

import panthera_control  # noqa: E402
import panthera_control.adapters as legacy_adapters  # noqa: E402
import panthera_control.constants as legacy_constants  # noqa: E402
import panthera_control.ipc as legacy_ipc  # noqa: E402
from finearm.arms.daemon import controller as real_controller  # noqa: E402
from finearm.arms.daemon import ipc as real_ipc  # noqa: E402


def test_constants_values_unchanged() -> None:
    assert legacy_constants.DEFAULT_SOCKET_PATH == "/tmp/panthera_lerobot_control.sock"
    assert legacy_constants.JOINT_COUNT == 6
    assert legacy_constants.KP == [10.0, 21.0, 21.0, 16.0, 13.0, 1.0]
    assert legacy_constants.KD == [1.0, 2.0, 2.0, 0.9, 0.8, 0.1]
    assert legacy_constants.GRIPPER_KP == 4.0
    assert legacy_constants.GRIPPER_KD == 0.4
    assert legacy_constants.FC == [0.15, 0.12, 0.12, 0.12, 0.04, 0.04]
    assert legacy_constants.FV == [0.05, 0.05, 0.05, 0.03, 0.02, 0.02]
    assert legacy_constants.VEL_THRESHOLD == 0.02
    assert legacy_constants.TOR_THRESHOLD == [0.5, 1.0, 1.0, 0.5, 0.3, 0.3]
    assert legacy_constants.TAU_LIMIT == [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]
    assert legacy_constants.DEFAULT_JOINT_MIN == [-3.14159] * 6
    assert legacy_constants.DEFAULT_JOINT_MAX == [3.14159] * 6
    assert legacy_constants.DEFAULT_GRIPPER_MIN == 0.0
    assert legacy_constants.DEFAULT_GRIPPER_MAX == 2.0


def test_make_snapshot_layout_unchanged() -> None:
    snapshot = legacy_constants.make_snapshot(
        follower_positions=[0.1] * 6,
        follower_velocities=[0.0] * 6,
        follower_gripper_position=0.5,
        follower_gripper_velocity=0.1,
        target_positions=[0.2] * 6,
        target_gripper_position=0.6,
        diagnostics={"mode": "mock"},
        timestamp=1.0,
        sequence=3,
        loop_timing={"target_period_s": 0.001},
    )
    assert len(snapshot["observation.state"]) == 14
    assert len(snapshot["action"]) == 7
    assert snapshot["observation.state"][6] == 0.5  # gripper position after 6 joints
    assert snapshot["observation.state"][13] == 0.1  # gripper velocity last
    assert snapshot["action"] == [0.2] * 6 + [0.6]
    assert legacy_constants.zero_joints() == [0.0] * 6


def test_legacy_error_name_is_alias() -> None:
    assert panthera_control.PantheraControlError is real_controller.ControlLoopError
    assert real_controller.PantheraControlError is real_controller.ControlLoopError


def test_ipc_and_adapter_modules_are_the_real_ones() -> None:
    assert legacy_ipc is real_ipc
    assert legacy_ipc.ControlClient is real_ipc.ControlClient
    # Default dims stay the legacy 14/7 for no-argument clients.
    client = legacy_ipc.ControlClient("/tmp/x.sock")
    assert client.state_dim == 14 and client.action_dim == 7
    for name in ("make_panthera_pair", "make_follower_only", "ensure_panthera_serial_access",
                 "PantheraRobotAdapter", "MockRobotAdapter", "AdapterError"):
        assert hasattr(legacy_adapters, name), name


def test_control_config_defaults_unchanged() -> None:
    cfg = panthera_control.ControlConfig()
    assert cfg.arm_id == "panthera_ht"
    assert cfg.socket_path == "/tmp/panthera_lerobot_control.sock"
    assert cfg.joint_min == [-3.14159] * 6
    assert cfg.joint_max == [3.14159] * 6
    assert cfg.gripper_min == 0.0 and cfg.gripper_max == 2.0
    assert cfg.policy_mirror_leader_kp == [0.3] * 6
    assert cfg.policy_mirror_leader_kd == [0.03] * 6
    assert cfg.leader_gravity_gain == [1.0] * 6
    assert list(cfg.gains.follower_kp) == legacy_constants.KP
    assert cfg.gains.gripper_kp == 4.0


def test_cli_entrypoint_forward_exists() -> None:
    from panthera_control.cli import main

    assert callable(main)
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_infer_robot_interface_import_path_works() -> None:
    """infer/robot_interface.py does `from panthera_control.ipc import ControlClient`."""
    from panthera_control.ipc import ControlClient  # noqa: F401
