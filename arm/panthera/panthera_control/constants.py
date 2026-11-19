"""Shared control constants and snapshot schema helpers."""

from __future__ import annotations

DEFAULT_SOCKET_PATH = "/tmp/panthera_lerobot_control.sock"
JOINT_COUNT = 6

KP = [10.0, 21.0, 21.0, 16.0, 13.0, 1.0]
KD = [1.0, 2.0, 2.0, 0.9, 0.8, 0.1]
GRIPPER_KP = 4.0
GRIPPER_KD = 0.4
FC = [0.15, 0.12, 0.12, 0.12, 0.04, 0.04]
FV = [0.05, 0.05, 0.05, 0.03, 0.02, 0.02]
VEL_THRESHOLD = 0.02
TOR_THRESHOLD = [0.5, 1.0, 1.0, 0.5, 0.3, 0.3]
TAU_LIMIT = [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]

DEFAULT_JOINT_MIN = [-3.14159] * JOINT_COUNT
DEFAULT_JOINT_MAX = [3.14159] * JOINT_COUNT
DEFAULT_GRIPPER_MIN = 0.0
DEFAULT_GRIPPER_MAX = 2.0


def as_floats(values, length: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length:
        raise ValueError(f"expected {length} values, got {len(result)}")
    return result


def clamp(values, lower, upper) -> list[float]:
    return [min(max(float(value), float(lo)), float(hi)) for value, lo, hi in zip(values, lower, upper)]


def clip(values, limits) -> list[float]:
    return [min(max(float(value), -float(limit)), float(limit)) for value, limit in zip(values, limits)]


def add(left, right) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def sub(left, right) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def threshold(values, limits) -> list[float]:
    result = []
    for value, limit in zip(values, limits):
        numeric = float(value)
        result.append(0.0 if abs(numeric) < float(limit) else numeric)
    return result


def zero_joints() -> list[float]:
    return [0.0] * JOINT_COUNT


def make_snapshot(
    *,
    follower_positions,
    follower_velocities,
    follower_gripper_position,
    follower_gripper_velocity,
    target_positions,
    target_gripper_position,
    diagnostics,
    timestamp,
    sequence,
    loop_timing,
) -> dict:
    state = (
        as_floats(follower_positions, JOINT_COUNT)
        + [float(follower_gripper_position)]
        + as_floats(follower_velocities, JOINT_COUNT)
        + [float(follower_gripper_velocity)]
    )
    action = as_floats(target_positions, JOINT_COUNT) + [float(target_gripper_position)]
    return {
        "observation.state": state,
        "action": action,
        "diagnostics": dict(diagnostics),
        "timestamp": float(timestamp),
        "sequence": int(sequence),
        "loop_timing": dict(loop_timing),
    }
