"""Panthera robot adapter layer: serial access, SDK discovery, hardware adapter.

Moved from ``arm/panthera/panthera_control/adapters.py`` (Phase 2). Changes:
dimensions come from this package's PROFILE, the repo root is discovered via
``finearm.core`` (works from any clone location), and the calibration default
prefers ``configs/arms/panthera_ht/`` when present (opt-in; absent = the
legacy SDK-tree robot_param default, behavior unchanged).
"""

from __future__ import annotations

import getpass
import glob
import grp
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from ...core import paths as core_paths
from ..daemon.adapters import AdapterError, ArmState, MockRobotAdapter, RobotAdapter
from . import PROFILE

JOINT_COUNT = PROFILE.joint_count


def _discover_panthera_serial_ports(
    by_id_dir: Path = Path("/dev/serial/by-id"),
    tty_glob: str = "/dev/ttyACM*",
) -> list[Path]:
    ports: list[Path] = []
    if by_id_dir.exists():
        for path in sorted(by_id_dir.glob("*Livelybot*")):
            try:
                ports.append(path.resolve(strict=False))
            except OSError:
                ports.append(path)
    if ports:
        return ports
    return [Path(path) for path in sorted(glob.glob(tty_glob))]


def _missing_panthera_serial_access(
    paths: Optional[list[Path]] = None,
    *,
    access_fn: Callable[[str | bytes | os.PathLike, int], bool] = os.access,
) -> list[str]:
    candidates = paths if paths is not None else _discover_panthera_serial_ports()
    return [str(path) for path in candidates if not access_fn(path, os.R_OK | os.W_OK)]


def _validate_panthera_serial_access(
    paths: Optional[list[Path]] = None,
    *,
    access_fn: Callable[[str | bytes | os.PathLike, int], bool] = os.access,
) -> None:
    missing = _missing_panthera_serial_access(paths, access_fn=access_fn)
    if not missing:
        return
    raise AdapterError(_serial_access_error_message(missing, user=getpass.getuser()))


def ensure_panthera_serial_access(
    paths: Optional[list[Path]] = None,
    *,
    access_fn: Callable[[str | bytes | os.PathLike, int], bool] = os.access,
    runner: Callable[..., object] = subprocess.run,
    stderr=None,
    user: Optional[str] = None,
) -> None:
    missing = _missing_panthera_serial_access(paths, access_fn=access_fn)
    if not missing:
        return

    stream = sys.stderr if stderr is None else stderr
    current_user = user or getpass.getuser()
    command = ["sudo", "setfacl", "-m", f"u:{current_user}:rw", *missing]
    print("Panthera live control needs temporary read/write permission for these serial ports:", file=stream)
    for path in missing:
        print(f"  - {path}", file=stream)
    excluded_media = "cam" + "era"
    excluded_storage = "data" + "set"
    print(
        "Scope: Panthera SDK uses these Livelybot/Panthera control-board ports to communicate "
        f"with the leader/follower arm motor boards. This does not grant {excluded_media}, network, "
        f"{excluded_storage}, or broad filesystem access; it only adds an ACL for the listed device "
        "nodes until they are unplugged/recreated or the system reboots.",
        file=stream,
    )
    print("About to run: " + " ".join(command), file=stream)
    print("Enter your sudo password if prompted.", file=stream)
    result = runner(command, check=False)
    if getattr(result, "returncode", 0) != 0:
        raise AdapterError(
            "Failed to grant Panthera serial permissions. "
            + _serial_access_error_message(missing, user=current_user)
        )

    still_missing = _missing_panthera_serial_access(paths, access_fn=access_fn)
    if still_missing:
        raise AdapterError(
            "Panthera serial permissions were requested, but these ports are still inaccessible: "
            + ", ".join(still_missing)
        )


def _serial_access_error_message(missing: list[str], *, user: str) -> str:
    try:
        dialout_gid = grp.getgrnam("dialout").gr_gid
        in_dialout = dialout_gid in os.getgroups()
    except KeyError:
        in_dialout = False
    detail = "current user is not in the dialout group" if not in_dialout else "current process cannot read/write these ports"
    return (
        "Panthera serial ports are not accessible: "
        + ", ".join(missing)
        + f" ({detail}). Run: sudo usermod -aG dialout {user} && log out/in; "
        + "for this session, you can also run: sudo setfacl -m u:"
        + user
        + ":rw "
        + " ".join(missing)
    )


def _finite_values(values, *, label: str, length: int) -> list[float]:
    try:
        result = [float(value) for value in values]
        if len(result) != length:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterError(f"{label} must contain {length} numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise AdapterError(f"{label} must contain only finite values")
    return result


def _finite_scalar(value, *, label: str) -> float:
    if isinstance(value, bool):
        raise AdapterError(f"{label} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterError(f"{label} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise AdapterError(f"{label} must be a finite numeric value")
    return result


class PantheraRobotAdapter(RobotAdapter):
    def __init__(self, robot):
        self.robot = robot
        self.motor_count = int(getattr(robot, "motor_count", JOINT_COUNT))

    def read_state(self) -> ArmState:
        gripper_torque = 0.0
        if hasattr(self.robot, "get_current_torque_gripper"):
            gripper_torque = self.robot.get_current_torque_gripper()
        elif hasattr(self.robot, "get_current_state_gripper"):
            gripper_state = self.robot.get_current_state_gripper()
            gripper_torque = getattr(gripper_state, "torque", 0.0)
        return ArmState(
            positions=_finite_values(self.robot.get_current_pos(), label="positions", length=JOINT_COUNT),
            velocities=_finite_values(self.robot.get_current_vel(), label="velocities", length=JOINT_COUNT),
            torques=_finite_values(self.robot.get_current_torque(), label="torques", length=JOINT_COUNT),
            gravity=_finite_values(self.robot.get_Gravity(), label="gravity", length=JOINT_COUNT),
            gripper_position=_finite_scalar(self.robot.get_current_pos_gripper(), label="gripper_position"),
            gripper_velocity=_finite_scalar(self.robot.get_current_vel_gripper(), label="gripper_velocity"),
            gripper_torque=_finite_scalar(gripper_torque, label="gripper_torque"),
        )

    def friction(self, velocities, fc, fv, velocity_threshold):
        return _finite_values(
            self.robot.get_friction_compensation(
                list(velocities),
                list(fc),
                list(fv),
                float(velocity_threshold),
            ),
            label="friction_compensation",
            length=JOINT_COUNT,
        )

    def command_arm(self, positions, velocities, torques, kp, kd) -> None:
        self.robot.pos_vel_tqe_kp_kd(list(positions), list(velocities), list(torques), list(kp), list(kd))

    def command_gripper(self, position, velocity, torque, kp, kd) -> None:
        self.robot.gripper_control_MIT(float(position), float(velocity), float(torque), float(kp), float(kd))

    def stop(self) -> None:
        brake = getattr(self.robot, "set_brake", None)
        if callable(brake):
            brake()


def make_mock_pair():
    return MockRobotAdapter("leader", joint_count=JOINT_COUNT), MockRobotAdapter("follower", joint_count=JOINT_COUNT)


def make_panthera_pair(
    *,
    sdk_root: Optional[str | Path] = None,
    leader_config: Optional[str | Path] = None,
    follower_config: Optional[str | Path] = None,
    factory: Optional[Callable[[str], object]] = None,
):
    robot_factory, root = _prepare_panthera_factory(sdk_root=sdk_root, factory=factory)
    leader_path = _resolve_calibration_path(leader_config, "Leader") if leader_config else _default_calibration("Leader", root)
    follower_path = _resolve_calibration_path(follower_config, "Follower") if follower_config else _default_calibration("Follower", root)
    return PantheraRobotAdapter(robot_factory(str(leader_path))), PantheraRobotAdapter(robot_factory(str(follower_path)))


def make_follower_only(
    *,
    sdk_root: Optional[str | Path] = None,
    follower_config: Optional[str | Path] = None,
    factory: Optional[Callable[[str], object]] = None,
):
    """Follower-only pair: the leader slot is a mock stand-in.

    Used by Quest-style teleoperation where a controller drives the follower
    through policy actions and no leader arm hardware is connected. Teleop
    mode is meaningless in this configuration and is rejected by the daemon.
    """
    robot_factory, root = _prepare_panthera_factory(sdk_root=sdk_root, factory=factory)
    follower_path = _resolve_calibration_path(follower_config, "Follower") if follower_config else _default_calibration("Follower", root)
    return MockRobotAdapter("leader", joint_count=JOINT_COUNT), PantheraRobotAdapter(robot_factory(str(follower_path)))


def _prepare_panthera_factory(
    *,
    sdk_root: Optional[str | Path] = None,
    factory: Optional[Callable[[str], object]] = None,
):
    if factory is None:
        _validate_panthera_serial_access()
    if sdk_root:
        # Accept both the SDK repo root and its panthera_python directory,
        # matching the env-var discovery path (_discover_default_sdk_root).
        root = _normalize_sdk_python_root(_resolve_project_path(sdk_root))
    else:
        root = _discover_default_sdk_root()
    scripts_dir = root / "scripts"
    robot_factory = factory or _load_factory(scripts_dir)
    return robot_factory, root


def _repo_root() -> Path:
    """finearm checkout root (core discovery; independent of clone location)."""
    root = core_paths.find_repo_root()
    if root is not None:
        return root
    return Path(__file__).resolve().parents[4]  # src/finearm/arms/panthera → repo root


def _resolve_project_path(path: str | Path) -> Path:
    """把 CLI 参数/环境变量里的路径解析成可移植的绝对路径。

    repo 是项目根:相对路径一律优先按仓库根解析——命令可从任意 CWD、
    任意克隆位置发起;仓库根下不存在时再按 CWD 解析;绝对路径原样返回。
    项目内不得依赖任何本机硬编码路径。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    anchored = _repo_root() / p
    if anchored.exists():
        return anchored
    return p.resolve()


def _default_calibration(name: str, sdk_python_root: Path) -> Path:
    """Calibration YAML default: configs/arms/<arm>/ when present, else the SDK tree.

    ``configs/arms/panthera_ht/{Leader,Follower}.yaml`` is the user-editable
    home (opt-in: absent = legacy SDK-tree robot_param default, unchanged).
    """
    repo_copy = _repo_root() / "configs" / "arms" / PROFILE.arm_id / f"{name}.yaml"
    if repo_copy.is_file():
        return repo_copy
    return sdk_python_root / "robot_param" / f"{name}.yaml"


def _resolve_calibration_path(path: str | Path, name: str) -> Path:
    return _resolve_project_path(path)


def _deploy_sdk_python_root() -> Path:
    """SDK default location: <repo>/arm/panthera/sdk/Panthera-HT_SDK/panthera_python."""
    return _repo_root() / "arm" / "panthera" / "sdk" / "Panthera-HT_SDK" / "panthera_python"


def _discover_default_sdk_root(cwd: Optional[str | Path] = None, package_file: Optional[str | Path] = None) -> Path:
    start = Path(cwd) if cwd is not None else Path.cwd()

    env_root = os.environ.get("PANTHERA_SDK_ROOT")
    if env_root:
        return _normalize_sdk_python_root(_resolve_project_path(env_root))

    # configs/local.jsonc "sdk_root" (layer 3 of the Phase 1 config model).
    # Only the explicit override — never the layer-4 default — so workspace
    # walk-up discovery below keeps its original priority.
    from ...core import config as core_config

    local_sdk = core_config.local_overrides().get("sdk_root")
    if local_sdk is not None:
        return _normalize_sdk_python_root(_resolve_project_path(local_sdk))

    # 先按注入的 cwd/package 祖先遍历(可发现任意外置 SDK 布局),
    # 都没有再落到仓库默认 SDK 位置(<repo>/arm/panthera/sdk)。
    candidate_bases = [start, *start.parents]
    package_path = Path(package_file).resolve() if package_file is not None else Path(__file__).resolve()
    candidate_bases.extend([package_path.parent, *package_path.parents])

    seen = set()
    for base in candidate_bases:
        candidate = base / "Panthera-HT_SDK" / "panthera_python"
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if (candidate / "scripts").exists() and (candidate / "robot_param").exists():
            return candidate

    return _deploy_sdk_python_root()


def _normalize_sdk_python_root(path: Path) -> Path:
    if (path / "panthera_python").exists():
        return path / "panthera_python"
    return path


def _load_factory(scripts_dir: Path):
    if not scripts_dir.exists():
        raise AdapterError(f"Panthera SDK scripts path not found: {scripts_dir}")
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from Panthera_lib import Panthera
    except ImportError as exc:
        raise AdapterError(f"Failed to import Panthera_lib from {scripts_dir}") from exc
    return Panthera


__all__ = [
    "AdapterError",
    "ArmState",
    "MockRobotAdapter",
    "PantheraRobotAdapter",
    "RobotAdapter",
    "ensure_panthera_serial_access",
    "make_follower_only",
    "make_mock_pair",
    "make_panthera_pair",
]
