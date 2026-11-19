"""Stop-and-go executor: window semantics, pacing, clamping, stats."""

from __future__ import annotations

import numpy as np

from finearm.infer.adapters import load as load_adapter
from finearm.infer.executors import StopAndGoExecutor


class RecordingRobot:
    def __init__(self):
        self.actions = []
        self.obs_calls = 0

    def get_obs(self, style="openpi"):
        self.obs_calls += 1
        return {"style": style}

    def send_action(self, a7):
        self.actions.append(np.asarray(a7, dtype=np.float64).copy())

    def close(self):
        pass


def _run(n_action_steps, max_steps, **kw):
    adapter = load_adapter("mock", host=None)
    robot = RecordingRobot()
    executor = StopAndGoExecutor(
        adapter, n_action_steps=n_action_steps, ctrl_hz=1000.0, **kw)
    stats = executor.run(robot, "task", max_steps=max_steps)
    return robot, stats


def test_window_size_ten_executes_exact_steps() -> None:
    robot, stats = _run(10, 45)
    assert stats.steps == 45
    assert stats.switches == 5          # 45 / 10 → 5 replans
    assert len(robot.actions) == 45


def test_zero_executes_whole_chunk() -> None:
    robot, stats = _run(0, 65)
    assert stats.steps == 65
    assert stats.switches == 3          # chunks are 30 long: 30 + 30 + 5


def test_reference_freshness_one_obs_per_window() -> None:
    robot, _ = _run(10, 45)
    assert robot.obs_calls == 5 + 0     # exactly one observation per replan


def test_pos_limits_enforced() -> None:
    lo = np.full(7, -0.05)
    hi = np.full(7, 0.05)
    robot, stats = _run(10, 40, pos_limits=(lo, hi))
    for action in robot.actions:
        assert np.all(action <= 0.05 + 1e-12)
        assert np.all(action >= -0.05 - 1e-12)
    assert stats.steps == 40


def test_vel_limits_enforced() -> None:
    robot, _ = _run(0, 60, vel_limits=np.full(7, 0.02))
    actions = robot.actions
    for prev, cur in zip(actions, actions[1:]):
        assert np.all(np.abs(cur - prev) <= 0.02 + 1e-12)


def test_max_steps_cuts_window_short() -> None:
    robot, stats = _run(10, 25)
    assert stats.steps == 25
    assert stats.switches == 3          # windows of 10, 10, 5(cut)
