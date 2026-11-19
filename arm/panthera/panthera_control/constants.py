"""兼容 shim:常量真身 = finearm.arms.panthera 的 PROFILE/GAINS + finearm.arms.daemon.helpers。

数值与旧 panthera_control/constants.py 逐位一致(Phase 2 迁移,行为等价);
snapshot schema 现在由 ArmProfile 单点声明。
"""

from __future__ import annotations

from finearm.arms.daemon import helpers as _h
from finearm.arms.panthera_ht import GAINS as _GAINS
from finearm.arms.panthera_ht import PROFILE as _PROFILE

DEFAULT_SOCKET_PATH = _PROFILE.socket_path
JOINT_COUNT = _PROFILE.joint_count

KP = list(_GAINS.follower_kp)
KD = list(_GAINS.follower_kd)
GRIPPER_KP = _GAINS.gripper_kp
GRIPPER_KD = _GAINS.gripper_kd
FC = list(_GAINS.friction_fc)
FV = list(_GAINS.friction_fv)
VEL_THRESHOLD = _GAINS.velocity_threshold
TOR_THRESHOLD = list(_GAINS.torque_threshold)
TAU_LIMIT = list(_GAINS.torque_limit)

DEFAULT_JOINT_MIN = [limits.min for limits in _PROFILE.joint_limits]
DEFAULT_JOINT_MAX = [limits.max for limits in _PROFILE.joint_limits]
DEFAULT_GRIPPER_MIN = _PROFILE.gripper_range.min
DEFAULT_GRIPPER_MAX = _PROFILE.gripper_range.max


as_floats = _h.as_floats
clamp = _h.clamp
clip = _h.clip
add = _h.add
sub = _h.sub
threshold = _h.threshold


def zero_joints() -> list[float]:
    return _h.zero_joints(JOINT_COUNT)


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
    return _h.make_snapshot(
        joint_count=JOINT_COUNT,
        follower_positions=follower_positions,
        follower_velocities=follower_velocities,
        follower_gripper_position=follower_gripper_position,
        follower_gripper_velocity=follower_gripper_velocity,
        target_positions=target_positions,
        target_gripper_position=target_gripper_position,
        diagnostics=diagnostics,
        timestamp=timestamp,
        sequence=sequence,
        loop_timing=loop_timing,
    )
