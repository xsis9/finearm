"""Execution-runtime state moved out of replay (Phase 5b, the archived refactor plan (git history) §五).

``RerunVisualizationManager`` and ``TakeoverState`` are loop/execution
concerns (visualization cadence, HIL takeover bookkeeping), not replay
machinery; they moved here so ``finearm.rl.replay`` owns only what becomes
transitions. ``finearm.rl.replay`` re-exports both for the shim surface
(pickle/journal compatibility unchanged — these types appear in deferred
runtime state, not in journals).

Moved verbatim from ``finearm/rl/replay.py``.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

class RerunVisualizationManager:
    """Best-effort, non-blocking Rerun logging for live robot data."""

    def __init__(self, log_data, compress_images: bool, fps: float):
        self._log_data = log_data
        self._compress_images = compress_images
        self._period_s = 1.0 / fps
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._action: dict | None = None
        self._running = False
        self._disabled = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_values(values: dict) -> dict:
        copied = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().numpy().copy()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                copied[key] = value
        return copied

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rerun-visualization", daemon=True)
        self._thread.start()

    def publish_observation(self, observation: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._observation = self._copy_values(observation)
            self._condition.notify()

    def publish_action(self, action: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._action = self._copy_values(action)
            self._condition.notify()

    def _run(self) -> None:
        next_log_time = 0.0
        while self._running:
            with self._condition:
                while self._running and self._observation is None and self._action is None:
                    self._condition.wait(timeout=0.1)
                if not self._running:
                    return
                wait_s = next_log_time - time.monotonic()
                if wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue
                observation, action = self._observation, self._action
                self._observation, self._action = None, None
            try:
                self._log_data(
                    observation=observation,
                    action=action,
                    compress_images=self._compress_images,
                )
                next_log_time = time.monotonic() + self._period_s
            except Exception as exc:
                self._disabled = True
                logger.warning("Rerun visualization disabled after logging failure: %s", exc)
                return

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.5)



class TakeoverState:
    """Small coordinator for lossless takeover requests and leader sample ordering."""

    pending: bool = False
    last_leader_sequence: int = -1

    def latch(self, requested: bool = True) -> None:
        self.pending = self.pending or bool(requested)

    def handoff_finished(self, succeeded: bool) -> None:
        if succeeded:
            self.pending = False

    def accept_leader_sequence(self, sequence: int) -> bool:
        if sequence <= self.last_leader_sequence:
            return False
        self.last_leader_sequence = sequence
        return True




__all__ = ["RerunVisualizationManager", "TakeoverState"]
