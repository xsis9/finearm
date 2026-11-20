"""Collector contract: one collection method per plugin directory.

A Collector owns a whole recording session lifecycle against an arm. The
built-in ``teleop`` collector wraps the LeRobot-native teleoperation flow;
``dagger`` (policy-with-human-correction) is the planned second method — its
building blocks (daemon policy mode, ``submit_policy_action``, freedrive
takeover, sample_coordinator same-tick pairing) already exist; implementing
it is expected to be one new directory plus a PolicySource (see dagger/README).
"""

from __future__ import annotations

import argparse
from typing import Any, Protocol, runtime_checkable


class CollectError(RuntimeError):
    """Collection failed (preflight, daemon, recorder, or validation)."""


@runtime_checkable
class PolicySource(Protocol):
    """DAgger attach point: where policy actions come from.

    Provided by finearm.infer executors or finearm.rl serving; DAgger reuses
    them instead of talking to servers directly.
    """

    def infer(self, obs) -> tuple[Any, dict[str, Any]]:
        ...


class Collector(Protocol):
    """One collection method (directory-scanned; ``method_id`` = directory name)."""

    method_id: str

    def setup(self, args: argparse.Namespace) -> None:
        """Validate preconditions (runtime present, cameras reachable, ...)."""
        ...

    def run_session(self, args: argparse.Namespace) -> int:
        """Run one recording session; return the process exit code."""
        ...


__all__ = ["CollectError", "Collector", "PolicySource"]
