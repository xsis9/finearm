"""Panthera-HT control gains (moved verbatim from panthera_control/constants.py)."""

from __future__ import annotations

from ..daemon.gains import ControlGains

GAINS = ControlGains(
    follower_kp=(10.0, 21.0, 21.0, 16.0, 13.0, 1.0),
    follower_kd=(1.0, 2.0, 2.0, 0.9, 0.8, 0.1),
    gripper_kp=4.0,
    gripper_kd=0.4,
    friction_fc=(0.15, 0.12, 0.12, 0.12, 0.04, 0.04),
    friction_fv=(0.05, 0.05, 0.05, 0.03, 0.02, 0.02),
    velocity_threshold=0.02,
    torque_threshold=(0.5, 1.0, 1.0, 0.5, 0.3, 0.3),
    torque_limit=(15.0, 30.0, 30.0, 15.0, 5.0, 5.0),
    gripper_friction_velocity_threshold=0.15,
    gripper_friction_torque=0.06,
)

__all__ = ["GAINS"]
