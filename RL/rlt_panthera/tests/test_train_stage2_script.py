"""Script-level tests for train_stage2.py (console gating, config, limits).

Runs CPU-only, no server / no robot: the console manager is fed a fake stdin
stream, and parse helpers are exercised directly.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_stage2  # noqa: E402


class FakeStdin:
    def __init__(self, lines: list[str] | None = None):
        self._lines = list(lines or [])

    def push(self, line: str) -> None:
        self._lines.append(line)

    def readline(self):
        if self._lines:
            return self._lines.pop(0) + "\n"
        # Block-ish: the console loop sleeps and re-reads on empty input.
        import time
        time.sleep(0.01)
        return "\n"


@pytest.fixture()
def console_factory():
    made = []

    def make(lines, **kwargs):
        manager = train_stage2.ConsoleInputManager(stream=FakeStdin(lines), **kwargs)
        made.append(manager)
        return manager

    yield make
    for manager in made:
        manager.stop()


def _wait(flag_check, timeout=2.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if flag_check():
            return True
        time.sleep(0.01)
    return False


def test_a_requires_critical_press_c_first(console_factory):
    train_stage2.ACTOR_ENABLED = False
    train_stage2.CRITICAL_PHASE_ACTIVE = False
    stream = FakeStdin(["a"])
    console = train_stage2.ConsoleInputManager(stream=stream)
    console.start()
    try:
        import time
        time.sleep(0.3)  # let the lone 'a' be consumed (and rejected)
        assert not train_stage2.ACTOR_ENABLED, "'a' outside critical must not enable"
        stream.push("c")
        assert _wait(lambda: train_stage2.CRITICAL_PHASE_ACTIVE)
        stream.push("a")
        assert _wait(lambda: train_stage2.ACTOR_ENABLED), "'a' inside critical enables the actor"
    finally:
        console.stop()
        train_stage2.ACTOR_ENABLED = False
        train_stage2.CRITICAL_PHASE_ACTIVE = False


def test_second_a_disables_actor_inside_critical(console_factory):
    train_stage2.ACTOR_ENABLED = False
    train_stage2.CRITICAL_PHASE_ACTIVE = True
    console = console_factory(["a", "a"])
    console.start()
    assert _wait(lambda: train_stage2.ACTOR_ENABLED)
    assert _wait(lambda: not train_stage2.ACTOR_ENABLED)
    train_stage2.CRITICAL_PHASE_ACTIVE = False


def test_auto_critical_allows_a_without_c(console_factory):
    train_stage2.ACTOR_ENABLED = False
    train_stage2.CRITICAL_PHASE_ACTIVE = False
    console = console_factory(["a"], auto_critical=True)
    console.start()
    assert _wait(lambda: train_stage2.ACTOR_ENABLED)
    train_stage2.ACTOR_ENABLED = False


def test_reward_prompt_parses_y_n_d_q(console_factory):
    console = console_factory(["x", "y"])
    console.start()
    assert console.prompt_episode_reward() == (1.0, True, False)
    train_stage2.SHUTDOWN = False

    console2 = console_factory(["d"])
    console2.start()
    assert console2.prompt_episode_reward() == (0.0, True, True)

    console3 = console_factory(["q"])
    console3.start()
    assert console3.prompt_episode_reward() == (0.0, False, False)


def test_strip_json_comments_preserves_strings():
    text = '{ // comment\n"a": "http://x", /* block\ncomment */ "b": 1, }'
    stripped = train_stage2._strip_json_comments(text)
    assert "// comment" not in stripped
    assert "http://x" in stripped, "URLs inside strings must survive comment stripping"


def test_parse_relative_limit_scalar_and_map():
    names = train_stage2.PANTHERA_JOINT_NAMES
    scalar = train_stage2.parse_relative_limit(0.05, names)
    assert scalar is not None and set(scalar) == set(names)
    assert all(v == 0.05 for v in scalar.values())

    payload = json.dumps({"target_joint_1": 0.1, "target_gripper": 0.02})
    mapping = train_stage2.parse_relative_limit(payload, names)
    assert mapping["target_joint_1"] == 0.1 and mapping["target_gripper"] == 0.02

    assert train_stage2.parse_relative_limit(None, names) is None
    assert train_stage2.parse_relative_limit("null", names) is None
    with pytest.raises(ValueError):
        train_stage2.parse_relative_limit(-1, names)
    with pytest.raises(ValueError):
        train_stage2.parse_relative_limit(json.dumps({"nope": 1.0}), names)


def test_config_jsonc_loads_and_rejects_unknown(tmp_path):
    config = {
        "server_port": 8123,
        "warmup_episodes": 3,
        "actor_max_relative_target": {"target_joint_1": 0.1},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config) + "\n// trailing comment")
    sys.argv = ["train_stage2.py", "--config", str(path),
                "--rlt_checkpoint", "x", "--stats_path", "y", "--warmup_episodes", "7"]
    args = train_stage2.parse_args()
    assert args.server_port == 8123
    assert args.warmup_episodes == 7, "CLI overrides config"
    assert args.actor_max_relative_target == json.dumps({"target_joint_1": 0.1})

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nonexistent_key": 1}))
    sys.argv = ["train_stage2.py", "--config", str(bad),
                "--rlt_checkpoint", "x", "--stats_path", "y"]
    with pytest.raises(SystemExit):
        train_stage2.parse_args()


def test_load_robot_dummy():
    robot = train_stage2.load_robot("robot.robot_interface", dummy=True)
    obs = robot.get_observation()
    assert obs["state"].shape == (7,)
    assert obs["base_image"].shape[2] == 3
    out = robot.send_action(obs["state"])
    assert out.shape == (7,)
