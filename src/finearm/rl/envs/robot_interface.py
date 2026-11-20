"""RobotInterface contract for Stage 2 online RL (execution machine, 4060).

Moved verbatim from ``RL/rlt_panthera/robot/robot_interface.py`` in
Phase 5a (that path is now a shim).

The acquisition machine already has a COMPLETE RobotInterface implementation
(panthera LeRobot robot + HIL). Port it into ``robot/panthera_robot.py`` and
adapt it to this contract — the Stage-2 loop below talks to hardware ONLY
through these methods, so nothing else in ``train_stage2.py`` needs to change.

Contract (all actions/absolute joint targets, 7-dim: 6 joints + gripper):

    connect() -> None
        Open buses / cameras / SDK. Raise on failure (the run aborts).
    disconnect() -> None
        Best-effort cleanup. Never raises.
    get_observation() -> dict
        {"state": (7,) float32 absolute joint positions,
         "base_image": (H, W, 3) uint8 RGB front camera,
         "wrist_image": (H, W, 3) uint8 RGB wrist camera}
        Camera resolution is arbitrary (the server resizes to 224²); the
        acquisition-stream resolution (480×640) is typical.
    send_action(target: (7,) float) -> (7,) float
        Command one absolute joint target; return the target ACTUALLY
        commanded (post driver clamp), same shape. Must raise (not return
        garbage) on communication failure — the loop retries 3× then faults.
    poll_human() -> tuple[bool, np.ndarray | None]
        HIL probe, called every control step OUTSIDE human control:
        (True, None)  → takeover requested this step (operator grabbed /
                        teaching-mode trigger / SDK intervention flag);
        (False, None) → no human activity.
        Once the loop enters human control it calls ``latest_human_action()``
        instead; takeover requests raised during human control are ignored.
    enter_human_control() -> bool
        Switch the hardware into human control ('h' key or poll_human trigger).
        Return False to refuse the takeover (the request stays pending).
        The interface is responsible for takeover SMOOTHING (entering teleop
        from the follower's current pose); the loop applies no offset.
    exit_human_control() -> None
        Return to policy control ('s' key / toggle back). Best-effort; called
        once when the loop leaves human control, before the next replan.
    latest_human_action() -> np.ndarray | None
        CURRENT human command (7,) ABSOLUTE targets while the operator
        controls the arm (leader stream / teaching pendant / SDK), or None
        if no fresh sample arrived since the last call (the window simply
        ends early, like the reference's stale-leader path).
    human_finished() -> bool
        True when the operator has released / remained still long enough to
        resume the policy. Default False (manual 's'/'h' exit only) — return
        True only if the hardware can genuinely detect operator release.
    reset() -> bool
        Move the arm to the episode start pose (closed-loop, your SDK's
        homing/teach pose). True on success.
    estop() -> None
        Immediate safe stop. Idempotent; called from the 'e' key path.

Notes:
- All methods are called from the MAIN control thread only (no internal
  locking required), except disconnect/estop which may come from any thread.
- Units: radians for joints, meters (or the SDK gripper unit) for the gripper
  — must match the p3 dataset action space (stats.json quantiles live there).
- ``DummyRobot`` below implements the contract for ``--dry_run`` / benching.
"""

from __future__ import annotations

import time

import numpy as np


class RobotInterface:
    """Abstract contract; subclass (or duck-type) with the real implementation."""

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_observation(self) -> dict:
        raise NotImplementedError

    def send_action(self, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def poll_human(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def enter_human_control(self) -> bool:
        return False

    def exit_human_control(self) -> None:
        return None

    def latest_human_action(self) -> np.ndarray | None:
        return None

    def human_finished(self) -> bool:
        return False

    def reset(self) -> bool:
        return False

    def estop(self) -> None:
        return None


class DummyRobot(RobotInterface):
    """No-hardware implementation for --dry_run and pipeline benching.

    Emits a deterministic slow sine wave around the p3 neutral pose so the
    full Stage-2 loop (server calls, replay, learner) can run end-to-end on
    a desk. poll_human never triggers; human paths stay unexercised.
    """

    def __init__(self, control_hz: float = 30.0):
        self._t0 = time.monotonic()
        self._dt = 1.0 / control_hz
        self._steps = 0
        self._neutral = np.array([0.0, -0.5, 1.2, 0.0, 0.6, 0.0, 0.5], dtype=np.float32)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> dict:
        t = self._steps * self._dt
        phase = np.sin(t * 0.5)
        state = self._neutral + 0.05 * phase * np.arange(1, 8, dtype=np.float32) / 7.0
        rng = np.random.default_rng(self._steps)
        return {
            "state": state.astype(np.float32),
            "base_image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        }

    def send_action(self, target: np.ndarray) -> np.ndarray:
        self._steps += 1
        return np.asarray(target, dtype=np.float32).copy()
