"""Quantile (q01/q99) normalization for the Panthera-HT absolute action space.

Moved from ``RL/rlt_panthera/rlt/norm.py`` in Phase 5a (that path is now a
shim). The on-disk stats format is owned by ``finearm.core.norm`` (pure
stdlib, Phase 1 contract — identical validation semantics); this module
keeps only the tensor wrappers.

Changes vs the legacy module (both intentional, Phase 5a):
- ``PANTHERA_BOLT_P3_STATS_PATH`` (a ``/root/xsx`` machine-absolute constant)
  is dropped: it had no consumers repo-wide (verified 2026-11-19); the real
  stats file always arrives via ``--stats_path`` / config.
- Parsing/validation delegates to ``finearm.core.norm.load_quantile_pairs``
  instead of a numpy duplicate of the same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from finearm.core.norm import load_quantile_pairs


def quantile_normalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def quantile_unnormalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    return (tensor + 1.0) * denom / 2.0 + q01


@dataclass
class QuantileStats:
    """Validated q01/q99 pairs for action (exact dim) and state (sliced dim)."""

    action_q01: torch.Tensor          # (action_dim,)
    action_q99: torch.Tensor          # (action_dim,)
    state_q01: torch.Tensor           # (state_dim,)
    state_q99: torch.Tensor           # (state_dim,)
    action_dim: int
    state_dim: int
    source_path: str = ""

    # ── action space (absolute joint targets) ─────────────────────
    def normalize_action(self, tensor: torch.Tensor) -> torch.Tensor:
        return quantile_normalize(tensor, self.action_q01, self.action_q99)

    def unnormalize_action(self, tensor: torch.Tensor) -> torch.Tensor:
        return quantile_unnormalize(tensor, self.action_q01, self.action_q99)

    # ── proprio (first state_dim of observation.state: positions) ─
    def normalize_state(self, tensor: torch.Tensor) -> torch.Tensor:
        return quantile_normalize(tensor, self.state_q01, self.state_q99)

    def unnormalize_state(self, tensor: torch.Tensor) -> torch.Tensor:
        return quantile_unnormalize(tensor, self.state_q01, self.state_q99)


def load_quantile_stats(
    stats_path: str | Path,
    *,
    action_dim: int = 7,
    state_dim: int = 7,
) -> QuantileStats:
    """Load and validate q01/q99 normalization stats for Stage 2.

    Accepts the LeRobot dataset ``meta/stats.json`` format
    (``action`` / ``observation.state`` entries with q01/q99 lists) or the
    openpi checkpoint ``assets/norm_stats.json`` format (``action`` /
    ``state``). State quantiles wider than ``state_dim`` are sliced to their
    first ``state_dim`` entries (positions). Rejects relative-action stats.
    """
    path = Path(stats_path).expanduser()
    pairs = load_quantile_pairs(path, action_dim=action_dim, state_dim=state_dim)
    return QuantileStats(
        action_q01=torch.as_tensor(pairs.action_q01, dtype=torch.float32),
        action_q99=torch.as_tensor(pairs.action_q99, dtype=torch.float32),
        state_q01=torch.as_tensor(pairs.state_q01, dtype=torch.float32),
        state_q99=torch.as_tensor(pairs.state_q99, dtype=torch.float32),
        action_dim=pairs.action_dim,
        state_dim=pairs.state_dim,
        source_path=str(path.resolve()),
    )


__all__ = [
    "QuantileStats",
    "quantile_normalize",
    "quantile_unnormalize",
    "load_quantile_stats",
]
