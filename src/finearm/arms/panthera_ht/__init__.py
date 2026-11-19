"""Panthera-HT arm plugin: profile, gains, and adapter factories.

This directory is the single source of truth for everything panthera-specific
the framework consumes. Adding a new arm = adding a sibling directory with
the same declarations (see ``finearm/arms/mock_twojoint`` for a minimal
example); no other code changes.
"""

from __future__ import annotations

from ..daemon.gains import ControlGains
from ..profile import ArmProfile, JointRange
from .gains import GAINS

ARM_ID = "panthera_ht"

# Snapshot schema (order defines observation.state / action layout):
# positions(6) + gripper + velocities(6) + gripper velocity; targets(6) + gripper.
PROFILE = ArmProfile(
    arm_id=ARM_ID,
    joint_names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"),
    state_feature_names=(
        "follower_joint_1.pos",
        "follower_joint_2.pos",
        "follower_joint_3.pos",
        "follower_joint_4.pos",
        "follower_joint_5.pos",
        "follower_joint_6.pos",
        "follower_gripper.pos",
        "follower_joint_1.vel",
        "follower_joint_2.vel",
        "follower_joint_3.vel",
        "follower_joint_4.vel",
        "follower_joint_5.vel",
        "follower_joint_6.vel",
        "follower_gripper.vel",
    ),
    action_feature_names=(
        "target_joint_1.pos",
        "target_joint_2.pos",
        "target_joint_3.pos",
        "target_joint_4.pos",
        "target_joint_5.pos",
        "target_joint_6.pos",
        "target_gripper.pos",
    ),
    socket_path="/tmp/panthera_lerobot_control.sock",
    gripper_range=JointRange(0.0, 2.0),
    # Conservative symmetric limits, verbatim from the legacy constants
    # (DEFAULT_JOINT_MIN/MAX = ±3.14159, not ±π).
    joint_limits=tuple(JointRange(-3.14159, 3.14159) for _ in range(6)),
    default_home=None,  # live mode parks to the startup pose
)


def make_mock_pair():
    from ..daemon.adapters import MockRobotAdapter

    return MockRobotAdapter("leader", joint_count=PROFILE.joint_count), MockRobotAdapter(
        "follower", joint_count=PROFILE.joint_count
    )


def make_pair(*, sdk_root=None, leader_config=None, follower_config=None, factory=None):
    from .adapter import make_panthera_pair

    return make_panthera_pair(
        sdk_root=sdk_root, leader_config=leader_config, follower_config=follower_config, factory=factory
    )


def make_follower_only(*, sdk_root=None, follower_config=None, factory=None):
    from .adapter import make_follower_only as _make

    return _make(sdk_root=sdk_root, follower_config=follower_config, factory=factory)


def ensure_serial_access():
    from .adapter import ensure_panthera_serial_access

    return ensure_panthera_serial_access()


__all__ = [
    "ARM_ID",
    "ControlGains",
    "GAINS",
    "PROFILE",
    "ensure_serial_access",
    "make_follower_only",
    "make_mock_pair",
    "make_pair",
]
