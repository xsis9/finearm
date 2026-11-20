"""Console key state + stdin ownership for the Stage-2 loop (5b extraction).

Extracted verbatim from ``train_stage2.py`` in Phase 5b: the module-level
key state lives HERE (single definition — the loop, infer_stage2, and tests
all read/write these attributes; nothing imports them by value anymore),
and :class:`ConsoleInputManager` owns stdin in one thread so safety keys
and reward prompts never race.

Signal handlers register explicitly via :func:`register_signal_handlers`
(called from main) instead of at import time, so importing the training
module as a library no longer hijacks SIGINT.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# ── Global console state (single stdin thread owns all writes) ───────────────
SHUTDOWN = False
EMERGENCY_STOP = False
ACTOR_ENABLED = False
SKIP_EPISODE = False
FINISH_EPISODE = False
CRITICAL_PHASE_ACTIVE = False
CRITICAL_MANUAL_OVERRIDE_AT = 0.0
HIL_FORCE_RESUME = False
HIL_TOGGLE_REQUESTED = False
POSITION_TOGGLE_REQUESTED = False
POSITIONING_STATE = "off"


class ConsoleInputManager:
    """Own stdin in one thread so safety keys and reward prompts never race.

    Ported verbatim from the reference (SO-101 leader comments dropped); see
    the reference docstring for the auto-critical 'a' semantics."""

    def __init__(self, stream=None, *, auto_critical: bool = False):
        self._stream = stream if stream is not None else sys.stdin
        self._responses: deque[str] = deque()
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_toggle_time = 0.0
        self._auto_critical = bool(auto_critical)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _loop(self) -> None:
        global EMERGENCY_STOP, SHUTDOWN, ACTOR_ENABLED, SKIP_EPISODE, FINISH_EPISODE
        global CRITICAL_PHASE_ACTIVE, HIL_FORCE_RESUME, HIL_TOGGLE_REQUESTED
        global CRITICAL_MANUAL_OVERRIDE_AT, POSITION_TOGGLE_REQUESTED
        while self._running and not SHUTDOWN:
            try:
                line = self._stream.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.05)
                continue
            value = line.strip().lower()
            if value.startswith("e"):
                EMERGENCY_STOP = True
                logger.warning("[KEY] EMERGENCY STOP requested (press 'e')")
                print("[KEY] EMERGENCY STOP", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("q"):
                SHUTDOWN = True
                logger.info("[KEY] Graceful quit requested (press 'q')")
                print("[KEY] Quit requested", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("a"):
                if ACTOR_ENABLED:
                    ACTOR_ENABLED = False
                    logger.info("[KEY] Actor mode OFF (press 'a' to re-enable)")
                    print("[KEY] Actor OFF", flush=True)
                elif CRITICAL_PHASE_ACTIVE or self._auto_critical:
                    ACTOR_ENABLED = True
                    logger.info("[KEY] Actor mode ON")
                    print("[KEY] Actor ON", flush=True)
                else:
                    logger.info("[KEY] Actor mode requires the critical phase: press 'c' first, then 'a'")
                    print("[KEY] Actor needs the critical phase (press 'c' first)", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("g"):
                SKIP_EPISODE = True
                print("[KEY] Skip episode", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("f"):
                FINISH_EPISODE = True
                print("[KEY] Finish episode early", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("c"):
                CRITICAL_PHASE_ACTIVE = not CRITICAL_PHASE_ACTIVE
                CRITICAL_MANUAL_OVERRIDE_AT = time.time()
                print(f"[KEY] Critical {'ON' if CRITICAL_PHASE_ACTIVE else 'OFF'}", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("s"):
                if not HIL_FORCE_RESUME:
                    HIL_FORCE_RESUME = True
                    print("[KEY] HIL force-resume", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("h"):
                now = time.monotonic()
                if now - self._last_toggle_time >= 1.0:
                    self._last_toggle_time = now
                    HIL_TOGGLE_REQUESTED = True
                    print("[KEY] HIL toggle", flush=True)
                else:
                    print("[KEY] HIL toggle debounced", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("p"):
                POSITION_TOGGLE_REQUESTED = True
                print(f"[KEY] Positioning toggle [state={POSITIONING_STATE}]", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            with self._condition:
                self._responses.append(value)
                self._condition.notify_all()

    def prompt_episode_reward(self) -> tuple[float, bool, bool]:
        print("\nWas this episode successful? [y/n/d(iscard)/q(uit)]: ", end="", flush=True)
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return 1.0, True, False
            if response in ("n", "no", "0"):
                return 0.0, True, False
            if response in ("d", "discard"):
                return 0.0, True, True
            if response in ("q", "quit"):
                return 0.0, False, False
            print("  Please enter 'y', 'n', 'd', or 'q': ", end="", flush=True)
        return 0.0, False, False

    def prompt_hil_resolution(self) -> str:
        print("\nHuman control paused. [y] success / [n] failure / [r] resume / [q] quit: ",
              end="", flush=True)
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return "success"
            if response in ("n", "no", "0"):
                return "failure"
            if response in ("r", "resume"):
                return "resume"
            if response in ("q", "quit"):
                return "quit"
            print("  Please enter 'y', 'n', 'r', or 'q': ", end="", flush=True)
        return "quit"


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


def register_signal_handlers() -> None:
    """Install graceful-shutdown handlers (SIGINT/SIGTERM → SHUTDOWN)."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


__all__ = [
    "ConsoleInputManager", "signal_handler", "register_signal_handlers",
    "SHUTDOWN", "EMERGENCY_STOP", "ACTOR_ENABLED", "SKIP_EPISODE",
    "FINISH_EPISODE", "CRITICAL_PHASE_ACTIVE", "CRITICAL_MANUAL_OVERRIDE_AT",
    "HIL_FORCE_RESUME", "HIL_TOGGLE_REQUESTED", "POSITION_TOGGLE_REQUESTED",
    "POSITIONING_STATE",
]

