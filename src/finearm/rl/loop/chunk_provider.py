"""ASAC-style pipelined chunk provider for rtc_mode.

Extracted verbatim from ``RL/rlt_panthera/scripts/train_stage2.py`` in
Phase 5a (class body unchanged; ``train_stage2`` now imports it from here).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

from finearm.rl.serving.client import VLAClient  # noqa: F401 — annotation only

logger = logging.getLogger(__name__)


class PipelinedChunkProvider:
    """Port of the deploy rtc_client.RTCRunner execution pattern (2026-11-13,
    user-validated on this task: continuous motion, higher success than
    stop-and-go).

    At most ONE in-flight serve request; the main thread triggers a replan
    when the current chunk has ≤ k steps remaining (k adaptive from RTT p95),
    so the round trip hides behind continued execution. Each result carries
    the TRIGGER-TIME tuple (chunk, z_rl, state) — kept internally consistent
    (≈ RTT-stale relative to the window that consumes it; the replay stride /
    terminal boundaries still materialize from exact boundary observations on
    the background connection, unaffected).
    """

    def __init__(self, vla: "VLAClient", *, control_hz: float, k: int | None = None):
        self._vla = vla
        self._k_user = k
        self._hz = control_hz
        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._payload: dict | None = None       # {"obs", "state"} set by main thread
        self._result: dict | None = None        # {"chunk", "z_rl", "state"}
        self._inflight = False
        self._rtts: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rtc-provider", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._trigger.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def inflight(self) -> bool:
        with self._lock:
            return self._inflight

    def k_eff(self, horizon: int) -> int:
        """Adaptive k = ceil(p95_RTT × hz) + 2, clamped to [3, horizon-5]."""
        cap = max(3, (horizon or 30) - 5)
        if self._k_user is not None:
            return self._k_user
        with self._lock:
            rtts = list(self._rtts)
        if not rtts:
            return 5
        k = int(np.clip(np.ceil(np.percentile(rtts, 95) * self._hz) + 2, 3, cap))
        if len(rtts) < 3:
            return min(max(5, k), cap)
        return k

    def trigger_replan(self, obs: dict, state: torch.Tensor) -> None:
        """Main thread: publish the latest observation and wake the infer loop."""
        with self._lock:
            if self._inflight:
                return
            self._payload = {"obs": obs, "state": state.detach().clone()}
        self._trigger.set()

    def take_result(self) -> dict | None:
        with self._lock:
            result = self._result
            self._result = None
        return result

    def sync_fetch(self, obs: dict, state: torch.Tensor) -> dict:
        """Blocking fetch from the MAIN thread (episode seed / post-HIL reseed).
        Waits for any in-flight request first so the control connection is
        never used concurrently."""
        while True:
            with self._lock:
                if not self._inflight:
                    self._payload = None
                    self._result = None
                    break
            time.sleep(0.01)
        t0 = time.perf_counter()
        actions, z_rl = self._vla.infer(obs)
        with self._lock:
            self._rtts.append(time.perf_counter() - t0)
            self._rtts = self._rtts[-30:]
        return {"chunk": actions.detach(), "z_rl": z_rl.detach(),
                "state": state.detach().clone()}

    def _loop(self) -> None:
        while self._running:
            if not self._trigger.wait(timeout=0.5):
                continue
            self._trigger.clear()
            with self._lock:
                payload = self._payload
                self._payload = None
                if payload is None or self._inflight:
                    continue
                self._inflight = True
            t0 = time.perf_counter()
            try:
                actions, z_rl = self._vla.infer(payload["obs"])
                rtt = time.perf_counter() - t0
                with self._lock:
                    self._result = {"chunk": actions.detach(), "z_rl": z_rl.detach(),
                                    "state": payload["state"]}
                    self._rtts.append(rtt)
                    self._rtts = self._rtts[-30:]
                    self._inflight = False
            except Exception as exc:  # noqa: BLE001
                logger.error("[rtc] pipelined inference failed: %s", exc)
                with self._lock:
                    self._inflight = False
