"""ArmProfile contract tests + panthera_ht golden schema lock.

The 14/7 names below are copied verbatim from the pre-Phase-2 hardcoded
tuples (constants/plugins/collect_cli). If this test fails, someone changed
the dataset schema — that breaks LeRobot dataset compatibility.
"""

from __future__ import annotations

import pytest

from finearm.arms import load_profile
from finearm.arms.profile import ArmProfile, JointRange

LEGACY_OBSERVATION_KEYS = (
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
)

LEGACY_ACTION_KEYS = (
    "target_joint_1.pos",
    "target_joint_2.pos",
    "target_joint_3.pos",
    "target_joint_4.pos",
    "target_joint_5.pos",
    "target_joint_6.pos",
    "target_gripper.pos",
)


def test_panthera_profile_matches_legacy_schema() -> None:
    profile = load_profile("panthera_ht")
    assert profile.state_feature_names == LEGACY_OBSERVATION_KEYS
    assert profile.action_feature_names == LEGACY_ACTION_KEYS
    assert profile.joint_count == 6
    assert profile.state_dim == 14
    assert profile.action_dim == 7
    assert profile.socket_path == "/tmp/panthera_lerobot_control.sock"
    assert profile.gripper_range == JointRange(0.0, 2.0)
    # Verbatim legacy DEFAULT_JOINT_MIN/MAX (±3.14159, not ±π).
    assert [r.min for r in profile.joint_limits] == [-3.14159] * 6
    assert [r.max for r in profile.joint_limits] == [3.14159] * 6


class TestArmProfileContract:
    def _make(
        self,
        joints: int = 2,
        state_extra: int = 0,
        action_extra: int = 0,
        joint_names=None,
        state_feature_names=None,
        action_feature_names=None,
        **kwargs,
    ) -> ArmProfile:
        return ArmProfile(
            arm_id="test_arm",
            joint_names=joint_names if joint_names is not None else tuple(f"joint_{i}" for i in range(joints)),
            state_feature_names=(
                state_feature_names
                if state_feature_names is not None
                else tuple(f"s{i}" for i in range(2 * (joints + 1) + state_extra))
            ),
            action_feature_names=(
                action_feature_names
                if action_feature_names is not None
                else tuple(f"a{i}" for i in range(joints + 1 + action_extra))
            ),
            socket_path="/tmp/test.sock",
            **kwargs,
        )

    def test_valid_profile_dims(self) -> None:
        profile = self._make(joints=3)
        assert profile.joint_count == 3
        assert profile.state_dim == 8
        assert profile.action_dim == 4

    def test_wrong_state_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="state_feature_names"):
            self._make(state_extra=1)

    def test_wrong_action_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="action_feature_names"):
            self._make(action_extra=1)

    def test_duplicate_joint_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            self._make(joints=2, joint_names=("a", "a"))

    def test_empty_joints_rejected(self) -> None:
        with pytest.raises(ValueError, match="joint_names"):
            self._make(joints=0, state_feature_names=("s0", "s1"), action_feature_names=("a0",))

    def test_bad_gripper_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="gripper_range"):
            self._make(gripper_range=JointRange(2.0, 2.0))

    def test_joint_limits_length_must_match(self) -> None:
        with pytest.raises(ValueError, match="joint_limits"):
            self._make(joint_limits=(JointRange(-1.0, 1.0),) * 3)

    def test_handshake_payload(self) -> None:
        profile = self._make(joints=1)
        payload = profile.handshake()
        assert payload["arm_id"] == "test_arm"
        assert payload["joint_count"] == 1
        assert payload["state_dim"] == 4
        assert payload["action_dim"] == 2
        assert payload["state_feature_names"] == list(profile.state_feature_names)
        assert payload["action_feature_names"] == list(profile.action_feature_names)
