"""Arm plugin discovery: directory scan, extensibility, error paths."""

from __future__ import annotations

import pytest

from finearm.arms import available, load_arm, load_profile
from finearm.core.registry import RegistryError


def test_available_lists_arm_plugins_not_infra() -> None:
    arms = available()
    assert "panthera_ht" in arms
    assert "mock_twojoint" in arms
    assert "daemon" not in arms
    assert "profile" not in arms


def test_load_arm_returns_plugin_with_declarations() -> None:
    plugin = load_arm("panthera_ht")
    assert plugin.ARM_ID == "panthera_ht"
    assert hasattr(plugin, "PROFILE")
    assert hasattr(plugin, "GAINS")
    assert callable(plugin.make_pair)
    assert callable(plugin.make_follower_only)
    assert callable(plugin.make_mock_pair)


def test_load_profile() -> None:
    assert load_profile("mock_twojoint").arm_id == "mock_twojoint"


def test_unknown_arm_rejected() -> None:
    with pytest.raises(RegistryError, match="Arm plugin not importable"):
        load_arm("no_such_arm")


def test_infra_module_not_an_arm() -> None:
    with pytest.raises(RegistryError, match="Unknown arm plugin"):
        load_arm("daemon")


def test_extensibility_one_directory_different_dims() -> None:
    """The extensibility contract: a new arm is one directory, different dims work."""
    profile = load_profile("mock_twojoint")
    assert profile.joint_count == 2
    assert profile.state_dim == 6
    assert profile.action_dim == 3
    leader, follower = load_arm("mock_twojoint").make_mock_pair()
    assert leader.motor_count == 2
    assert follower.motor_count == 2
    state = leader.read_state()
    assert len(state.positions) == 2
