"""ArmProfile: the per-arm data declaration that single-sources the snapshot schema.

Before Phase 2 the 14-dim state / 7-dim action schema was hardcoded in four
places (constants, IPC validation, both LeRobot plugins, collect resume
validation). Every consumer now derives its dimensions and feature names from
one ArmProfile declaration per arm (REFACTOR_PLAN.md §3a).

Snapshot layout convention (asserted in :class:`ArmProfile`):

- ``observation.state`` = ``state_feature_names`` in order:
  joint positions, gripper position, joint velocities, gripper velocity.
- ``action`` = ``action_feature_names`` in order:
  joint target positions, gripper target.

Pure standard library: profiles are imported by the daemon, the LeRobot
plugins, and the collect CLI alike, in environments with no numpy/yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JointRange:
    min: float
    max: float


@dataclass(frozen=True)
class ArmProfile:
    """Declarative facts about one arm. Order tuples define the data schema."""

    arm_id: str
    joint_names: tuple[str, ...]
    state_feature_names: tuple[str, ...]
    action_feature_names: tuple[str, ...]
    socket_path: str
    gripper_range: JointRange = field(default_factory=lambda: JointRange(0.0, 2.0))
    joint_limits: tuple[JointRange, ...] | None = None
    default_home: str | None = None  # "j1,...,jN,gripper" park target

    def __post_init__(self) -> None:
        joints = len(self.joint_names)
        if joints == 0:
            raise ValueError(f"{self.arm_id}: joint_names must not be empty")
        if len(set(self.joint_names)) != joints:
            raise ValueError(f"{self.arm_id}: joint_names must be unique")
        expected_state = 2 * (joints + 1)
        if len(self.state_feature_names) != expected_state:
            raise ValueError(
                f"{self.arm_id}: state_feature_names must contain {expected_state} entries "
                f"(positions+gripper+velocities+gripper velocity), got {len(self.state_feature_names)}"
            )
        if len(self.action_feature_names) != joints + 1:
            raise ValueError(
                f"{self.arm_id}: action_feature_names must contain {joints + 1} entries "
                f"(joint targets+gripper), got {len(self.action_feature_names)}"
            )
        if self.gripper_range.min >= self.gripper_range.max:
            raise ValueError(f"{self.arm_id}: gripper_range must satisfy min < max")
        if self.joint_limits is not None:
            if len(self.joint_limits) != joints:
                raise ValueError(f"{self.arm_id}: joint_limits must match joint_names length")
            for joint, limits in zip(self.joint_names, self.joint_limits):
                if limits.min >= limits.max:
                    raise ValueError(f"{self.arm_id}: joint {joint} limits must satisfy min < max")

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return len(self.state_feature_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_feature_names)

    def handshake(self) -> dict:
        """Serializable summary attached to IPC status responses (additive)."""
        return {
            "arm_id": self.arm_id,
            "joint_count": self.joint_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "joint_names": list(self.joint_names),
            "state_feature_names": list(self.state_feature_names),
            "action_feature_names": list(self.action_feature_names),
        }


__all__ = ["ArmProfile", "JointRange"]
