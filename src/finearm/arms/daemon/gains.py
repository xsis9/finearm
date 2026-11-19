"""Arm-neutral control gains container.

Values are per-arm (see ``finearm/arms/<arm>/gains.py``); the daemon loop
only consumes them through :class:`ControlGains`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlGains:
    follower_kp: tuple[float, ...]
    follower_kd: tuple[float, ...]
    gripper_kp: float
    gripper_kd: float
    friction_fc: tuple[float, ...]
    friction_fv: tuple[float, ...]
    velocity_threshold: float
    torque_threshold: tuple[float, ...]
    torque_limit: tuple[float, ...]
    gripper_friction_velocity_threshold: float
    gripper_friction_torque: float

    def __post_init__(self) -> None:
        joints = len(self.follower_kp)
        if joints == 0:
            raise ValueError("ControlGains: follower_kp must not be empty")
        for name in ("follower_kd", "friction_fc", "friction_fv", "torque_threshold", "torque_limit"):
            if len(getattr(self, name)) != joints:
                raise ValueError(f"ControlGains: {name} must match follower_kp length {joints}")


__all__ = ["ControlGains"]
