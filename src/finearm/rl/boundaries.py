"""WindowReceipt / BoundaryCollector — the core Stage-2 seam (the archived refactor plan, git history
§五 5b).

The episode loop OWNS execution; WHAT becomes a replay transition is the
collector's decision. Splitting those two responsibilities is what lets the
rtc/stop-and-go executors, the HIL paths, and future DAgger-style collectors
share one loop without duplicating boundary logic.

Red line (tests/rl/test_boundary_equivalence.py): the rtc-mode transition
stream must stay byte-identical while this seam is cut. The first concrete
implementation below owns the episode-end commit path verbatim as it lived
in ``train_stage2.main``; window/takeover receipts wire in with the
EpisodeRunner extraction (next 5b step) — the Protocol is the contract both
sides build against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from finearm.rl.replay import Transition, order_episode_transitions


@dataclass
class WindowReceipt:
    """One executed window, handed to the collector at its boundary.

    ``reference_age_steps``: 0 = embeddings fresh at the boundary (exact
    boundary snapshot); >0 = RTC-style staleness (trigger-time tuple kept
    internally consistent).
    """

    window_index: int
    executed_actions: torch.Tensor
    reference_chunk: torch.Tensor
    reference_embeddings: dict[str, torch.Tensor]
    reference_age_steps: int = 0
    boundary_obs: dict | None = None
    source: str = "policy"  # "policy" | "human" | "mixed"


@dataclass
class _EpisodeCommit:
    """Inputs of the episode-end commit path (extracted from main verbatim)."""

    transitions: list[Transition] = field(default_factory=list)


class BoundaryCollector(Protocol):
    """The seam contract (target wiring: EpisodeRunner, 5b-2)."""

    def on_window(self, receipt: WindowReceipt) -> None: ...

    def on_takeover(self, obs: dict) -> None: ...

    def on_terminal(self, obs: dict, success: bool) -> None: ...

    def materialize_pending(self, service) -> None: ...

    def collect_episode(self) -> list[Transition]: ...


class ReplayBoundaryCollector:
    """First implementation: owns ordering, terminal reward stamping, and the
    durable journal append (disk-first) at episode end.

    Bodies are verbatim from ``train_stage2.main``'s episode-end block; the
    golden journal locks them byte-for-byte.
    """

    def __init__(self, replay) -> None:
        self._replay = replay
        self._pending = _EpisodeCommit()

    # ── commit path (verbatim extraction) ────────────────────────────────
    def commit_episode(
        self,
        episode_transitions: list[Transition],
        *,
        reward: float,
        faulted: bool = False,
        interrupted: bool = False,
    ) -> int:
        episode_transitions[:] = order_episode_transitions(episode_transitions)
        if episode_transitions:
            terminal_indices = [
                i for i, t in enumerate(episode_transitions) if t.done
            ]
            if not terminal_indices and not interrupted and not faulted:
                episode_transitions[-1].done = True
                terminal_indices = [len(episode_transitions) - 1]
            for idx in terminal_indices:
                terminal = episode_transitions[idx]
                terminal.rewards[terminal.executed_steps - 1] = reward
        for transition in episode_transitions:
            self._replay.add_transition(transition)
        return len(episode_transitions)

    # ── Protocol surface (wired by the EpisodeRunner step) ───────────────
    def on_window(self, receipt: WindowReceipt) -> None:
        self._pending.transitions  # pragma: no cover - wired in 5b-2

    def on_takeover(self, obs: dict) -> None:  # pragma: no cover - 5b-2
        return None

    def on_terminal(self, obs: dict, success: bool) -> None:  # pragma: no cover - 5b-2
        return None

    def materialize_pending(self, service) -> None:  # pragma: no cover - 5b-2
        return None

    def collect_episode(self) -> list[Transition]:  # pragma: no cover - 5b-2
        return []


__all__ = ["BoundaryCollector", "WindowReceipt", "ReplayBoundaryCollector"]
