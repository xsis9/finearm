"""Mock daemon round-trips: dimensions and handshake flow from the arm profile.

No hardware, no network: the daemon runs in-process on mock adapters over a
real Unix socket. Both the shipped arm (14/7) and the mock two-joint demo arm
(6/3) go through the same daemon + IPC code paths.
"""

from __future__ import annotations

import time

import pytest

from finearm.arms import load_profile
from finearm.arms.daemon import ControlClient, ControlConfig, ControlDaemon


@pytest.fixture
def daemon(tmp_path, request):
    arm_id = getattr(request, "param", "panthera_ht")
    profile = load_profile(arm_id)
    socket_path = str(tmp_path / f"{arm_id}.sock")
    daemon = ControlDaemon(
        ControlConfig(arm_id=arm_id, mock=True, socket_path=socket_path, loop_hz=250.0)
    )
    daemon.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if daemon.latest_snapshot().get("sequence", 0) >= 1:
            break
        time.sleep(0.01)
    yield daemon, socket_path, profile
    daemon.stop()


def _client(profile, socket_path) -> ControlClient:
    return ControlClient.from_profile(profile, socket_path=socket_path)


@pytest.mark.parametrize("daemon", ["panthera_ht", "mock_twojoint"], indirect=True)
class TestBothArms:
    def test_snapshot_dims_match_profile(self, daemon) -> None:
        d, sock, profile = daemon
        snapshot = _client(profile, sock).snapshot(max_age_s=2.0)
        assert len(snapshot["observation.state"]) == profile.state_dim
        assert len(snapshot["action"]) == profile.action_dim
        assert snapshot["sequence"] >= 1

    def test_status_handshake(self, daemon) -> None:
        d, sock, profile = daemon
        status = _client(profile, sock).status()
        assert status["running"] is True
        assert status["profile"]["arm_id"] == profile.arm_id
        assert status["profile"]["state_dim"] == profile.state_dim
        assert status["profile"]["action_dim"] == profile.action_dim
        assert status["profile"]["state_feature_names"] == list(profile.state_feature_names)

    def test_policy_action_roundtrip(self, daemon) -> None:
        d, sock, profile = daemon
        client = _client(profile, sock)
        assert client.set_mode("policy")["control_mode"] == "policy"
        action = [0.05] * profile.action_dim
        assert client.send_policy_action(action)["control_mode"] == "policy"

    def test_policy_action_wrong_dim_rejected(self, daemon) -> None:
        d, sock, profile = daemon
        client = _client(profile, sock)
        with pytest.raises(ValueError, match=f"must contain {profile.action_dim} values"):
            client.send_policy_action([0.1] * (profile.action_dim + 1))


def test_legacy_default_client_validates_panthera_dims(tmp_path) -> None:
    """A no-argument ControlClient keeps the legacy 14/7 validation defaults."""
    socket_path = str(tmp_path / "legacy.sock")
    daemon = ControlDaemon(ControlConfig(mock=True, socket_path=socket_path, loop_hz=250.0))
    daemon.start()
    try:
        legacy_client = ControlClient(socket_path)
        snapshot = legacy_client.snapshot(max_age_s=2.0)
        assert len(snapshot["observation.state"]) == 14
    finally:
        daemon.stop()


def test_status_response_is_backward_compatible_superset() -> None:
    """The handshake is additive: every pre-Phase-2 status key is still present."""
    legacy_keys = {
        "running", "mode", "control_mode", "holding", "sequence", "timestamp",
        "last_error", "socket_path", "policy_action_age_s", "policy_sequence", "policy_stale",
    }
    daemon = ControlDaemon(ControlConfig(mock=True))
    status = daemon.status()
    assert legacy_keys <= set(status)
