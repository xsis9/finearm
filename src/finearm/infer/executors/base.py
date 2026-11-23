"""ChunkExecutor base: the executor contract + shared action math.

The align/clamp/blend helpers are module-level so ``finearm.rl.loop`` can
reuse them in Phase 5 (the archived refactor plan (git history) §4). Executors differ only in HOW
they interleave observation, inference, and execution — safety clamping is
identical everywhere and lives only here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExecutorStats:
    """Return value of ``ChunkExecutor.run``."""

    steps: int = 0
    switches: int = 0                     # chunk switches (RTC) / replans (stop-and-go)
    rtt_p50_ms: float | None = None
    rtt_p95_ms: float | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        rtts = "" if self.rtt_p50_ms is None else (
            f"RTT p50={self.rtt_p50_ms:.0f}ms p95={self.rtt_p95_ms:.0f}ms"
        )
        return f"{self.steps} 步 / {self.switches} 次切换; {rtts}".strip("; ")


def clamp_action(prev, target, pos_limits, vel_limits) -> np.ndarray:
    """Clip a target against joint limits and per-step velocity limits.

    (Explicit clamping is managed ONLY here and in the executors' use of it.)
    """
    a = np.asarray(target, dtype=np.float64).copy()
    if pos_limits is not None:
        lo, hi = pos_limits
        a = np.clip(a, lo, hi)
    if prev is not None and vel_limits is not None:
        a = np.clip(a, prev - vel_limits, prev + vel_limits)
    return a


def align_index(new_chunk, ref_action) -> int:
    """Index in ``new_chunk`` closest to ``ref_action`` (first 6 joints L2, gripper excluded)."""
    d = np.linalg.norm(new_chunk[:, :6] - np.asarray(ref_action)[:6], axis=1)
    return int(np.argmin(d))


def blend_transition(old_chunk, old_idx, new_chunk, j, blend):
    """Linear-blend handover between chunks (RTC chunk switch).

    Returns ``(pending, start)``: ``pending`` = up to ``blend`` interpolated
    actions covering old[j, j+blend) → new[j, j+blend); ``start`` = the index
    to resume at inside the new chunk (blend steps already covered, avoiding
    a backward jump).
    """
    pending = []
    for s in range(blend):
        if old_idx + s < len(old_chunk):
            alpha = (s + 1) / (blend + 1)
            pending.append(
                (1 - alpha) * old_chunk[old_idx + s]
                + alpha * new_chunk[min(j + s, len(new_chunk) - 1)])
    start = min(j + blend, len(new_chunk) - 1) if blend > 0 else j
    return pending, start


class ChunkExecutor:
    """Common executor interface; adapters provide inference, robots execute.

    Subclasses implement :meth:`run`. ``pos_limits``/``vel_limits`` semantics
    are shared: (lo7, hi7) joint bounds and (7,) max displacement per control
    step; ``None`` disables the respective clamp.
    """

    name = "?"

    def __init__(self, adapter, *, ctrl_hz=30.0, pos_limits=None, vel_limits=None,
                 dry_run=False, verbose=True):
        self.adapter = adapter
        self.ctrl_hz = ctrl_hz
        self.pos_limits = pos_limits
        self.vel_limits = vel_limits
        self.dry_run = dry_run
        self.verbose = verbose

    def run(self, robot, task, max_steps=None) -> ExecutorStats:
        raise NotImplementedError


__all__ = [
    "ChunkExecutor",
    "ExecutorStats",
    "align_index",
    "blend_transition",
    "clamp_action",
]
