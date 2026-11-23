"""Mock two-joint demo arm: the extensibility reference.

A complete arm plugin with zero hardware: 2 joints + gripper → 6-dim state /
3-dim action. It exists to prove that adding one directory under
``finearm/arms/`` is enough for the daemon, IPC, and clients to pick up a new
arm with different dimensions (the archived refactor plan (git history) §10.4) — no other code
changes. Use it as the skeleton for a real new arm: replace the mock
factories with a real adapter (see ``finearm/arms/panthera/adapter.py``).
"""

from __future__ import annotations

from ..daemon.gains import ControlGains
from ..profile import ArmProfile, JointRange

ARM_ID = "mock_twojoint"

PROFILE = ArmProfile(
    arm_id=ARM_ID,
    joint_names=("joint_a", "joint_b"),
    state_feature_names=(
        "follower_joint_a.pos",
        "follower_joint_b.pos",
        "follower_gripper.pos",
        "follower_joint_a.vel",
        "follower_joint_b.vel",
        "follower_gripper.vel",
    ),
    action_feature_names=(
        "target_joint_a.pos",
        "target_joint_b.pos",
        "target_gripper.pos",
    ),
    socket_path="/tmp/finearm_mock_twojoint.sock",
    gripper_range=JointRange(0.0, 1.0),
    joint_limits=(JointRange(-2.5, 2.5), JointRange(-2.0, 2.0)),
    default_home="0,0,0.5",
)

GAINS = ControlGains(
    follower_kp=(8.0, 8.0),
    follower_kd=(1.0, 1.0),
    gripper_kp=2.0,
    gripper_kd=0.2,
    friction_fc=(0.1, 0.1),
    friction_fv=(0.02, 0.02),
    velocity_threshold=0.02,
    torque_threshold=(0.5, 0.5),
    torque_limit=(10.0, 10.0),
    gripper_friction_velocity_threshold=0.15,
    gripper_friction_torque=0.05,
)


def make_mock_pair():
    from ..daemon.adapters import MockRobotAdapter

    return MockRobotAdapter("leader", joint_count=PROFILE.joint_count), MockRobotAdapter(
        "follower", joint_count=PROFILE.joint_count
    )


def make_pair(*, sdk_root=None, leader_config=None, follower_config=None, factory=None):
    raise NotImplementedError("mock_twojoint is a mock-only demo arm; run the daemon with --mock")


def make_follower_only(*, sdk_root=None, follower_config=None, factory=None):
    raise NotImplementedError("mock_twojoint is a mock-only demo arm; run the daemon with --mock")


__all__ = ["ARM_ID", "ControlGains", "GAINS", "PROFILE", "make_follower_only", "make_mock_pair", "make_pair"]
