"""DAgger collection — interface stub only (not implemented; Phase 4 scope).

Design notes live in README.md next to this file. Implementing DAgger is
expected to be: this directory's Collector driving the follower via the
daemon's policy mode with actions from a :class:`..base.PolicySource`
(finearm.infer executor or finearm.rl serving), while the operator corrects
through the leader (freedrive takeover already exists daemon-side) and the
sample_coordinator pairs policy/human snapshots on the same tick.
"""

method_id = "dagger"


class DAggerCollector:
    """Placeholder: raises with guidance until implemented."""

    method_id = method_id

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "DAgger collection is not implemented yet (Phase 4 ships the seam: "
            "finearm.data.collector.base.PolicySource + this registry slot)"
        )

    def setup(self, args):  # pragma: no cover - stub
        raise NotImplementedError

    def run_session(self, args):  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["DAggerCollector", "method_id"]
