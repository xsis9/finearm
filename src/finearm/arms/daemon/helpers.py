"""Arm-neutral snapshot math helpers (successor of panthera_control.constants helpers).

All dimension-sensitive helpers take the joint count explicitly; defaults
reference the shipped panthera_ht arm only where backward compatibility with
the legacy no-argument call sites requires it.
"""

from __future__ import annotations

# Legacy default = the shipped panthera_ht arm.
DEFAULT_JOINT_COUNT = 6


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


def zero_joints(joint_count: int = DEFAULT_JOINT_COUNT) -> list[float]:
    return [0.0] * joint_count


def make_snapshot(
    *,
    joint_count: int,
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
    """Assemble the IPC snapshot; layout = profile.state_feature_names order.

    ``observation.state`` = positions(j) + gripper + velocities(j) + gripper velocity;
    ``action`` = target positions(j) + target gripper.
    """
    state = (
        as_floats(follower_positions, joint_count)
        + [float(follower_gripper_position)]
        + as_floats(follower_velocities, joint_count)
        + [float(follower_gripper_velocity)]
    )
    action = as_floats(target_positions, joint_count) + [float(target_gripper_position)]
    return {
        "observation.state": state,
        "action": action,
        "diagnostics": dict(diagnostics),
        "timestamp": float(timestamp),
        "sequence": int(sequence),
        "loop_timing": dict(loop_timing),
    }


__all__ = [
    "add",
    "as_floats",
    "clamp",
    "clip",
    "make_snapshot",
    "sub",
    "threshold",
    "zero_joints",
]
