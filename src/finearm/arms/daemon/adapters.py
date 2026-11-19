"""Arm-neutral adapter contract and the mock adapter.

``RobotAdapter`` is the five-method hardware seam the control loop drives
(read_state / friction / command_arm / command_gripper / stop). Real adapters
live in ``finearm/arms/<arm>/adapter.py``; the mock here is used by ``--mock``
runs and by mock-only demo arms.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

# Default joint count when an adapter does not declare one: the shipped
# panthera_ht arm (backward-compatible legacy value).
DEFAULT_JOINT_COUNT = 6


class AdapterError(RuntimeError):
    pass


@dataclass
class ArmState:
    positions: list[float]
    velocities: list[float]
    torques: list[float]
    gravity: list[float]
    gripper_position: float
    gripper_velocity: float
    gripper_torque: float


class RobotAdapter:
    motor_count = DEFAULT_JOINT_COUNT

    def read_state(self) -> ArmState:
        raise NotImplementedError

    def friction(self, velocities, fc, fv, velocity_threshold):
        raise NotImplementedError

    def command_arm(self, positions, velocities, torques, kp, kd) -> None:
        raise NotImplementedError

    def command_gripper(self, position, velocity, torque, kp, kd) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class MockRobotAdapter(RobotAdapter):
    """Deterministic software stand-in: the leader sweeps sinusoids, the follower servo-follows."""

    def __init__(self, role: str, joint_count: int = DEFAULT_JOINT_COUNT):
        self.role = role
        self.motor_count = int(joint_count)
        self.positions = [0.0] * self.motor_count
        self.velocities = [0.0] * self.motor_count
        self.torques = [0.0] * self.motor_count
        self.gravity = [0.0] * self.motor_count
        self.gripper_position = 0.5 if role == "leader" else 0.0
        self.gripper_velocity = 0.0
        self.gripper_torque = 0.0
        self._started = time.monotonic()
        self._last_update = self._started
        self._stopped = False

    def read_state(self) -> ArmState:
        now = time.monotonic()
        dt = max(now - self._last_update, 0.0)
        self._last_update = now
        if self.role == "leader" and not self._stopped:
            t = now - self._started
            next_positions = [0.15 * math.sin(t + i * 0.1) for i in range(self.motor_count)]
            self.velocities = [
                (next_position - position) / dt if dt > 0.0 else 0.0
                for next_position, position in zip(next_positions, self.positions)
            ]
            self.positions = next_positions
            self.gripper_position = 0.8 + 0.2 * math.sin(t)
            self.gripper_velocity = 0.2 * math.cos(t)
        return ArmState(
            positions=list(self.positions),
            velocities=list(self.velocities),
            torques=list(self.torques),
            gravity=list(self.gravity),
            gripper_position=float(self.gripper_position),
            gripper_velocity=float(self.gripper_velocity),
            gripper_torque=float(self.gripper_torque),
        )

    def friction(self, velocities, fc, fv, velocity_threshold):
        result = []
        for velocity, static, viscous in zip(velocities, fc, fv):
            velocity = float(velocity)
            sign = 0.0
            if abs(velocity) >= float(velocity_threshold):
                sign = 1.0 if velocity > 0.0 else -1.0
            result.append(sign * float(static) + float(viscous) * velocity)
        return result

    def command_arm(self, positions, velocities, torques, kp, kd) -> None:
        if self._stopped:
            return
        self.torques = list(torques)
        if self.role == "follower":
            previous = self.positions
            self.positions = list(positions)
            self.velocities = list(velocities)
            if not any(abs(value) > 0.0 for value in self.velocities):
                self.velocities = [target - current for target, current in zip(self.positions, previous)]

    def command_gripper(self, position, velocity, torque, kp, kd) -> None:
        if self._stopped:
            return
        self.gripper_torque = float(torque)
        if self.role == "follower":
            self.gripper_position = float(position)
            self.gripper_velocity = float(velocity)

    def stop(self) -> None:
        self._stopped = True


__all__ = [
    "AdapterError",
    "ArmState",
    "MockRobotAdapter",
    "RobotAdapter",
]
