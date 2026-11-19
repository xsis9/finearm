"""Quantile (q01/q99) normalization for the Panthera-HT absolute action space.

Ported from RL-Token-Pi05-open ``scripts/rlt_core.py`` /
``train_rlt_stage2_pi05.py`` (``quantile_normalize`` / ``quantile_unnormalize``
/ ``_validate_quantiles``), with the "exactly 6D" validation relaxed:

- action quantiles must be exactly ``action_dim`` (7 for Panthera)
- state quantiles may be the full LeRobot ``observation.state`` width (14 =
  7 positions + 7 velocities); the first ``state_dim`` (7, positions) are
  used — the same slice openpi's panthera_policy feeds the VLA
  (``observation/state[..., :7]``).

Stats source: ``panthera_bolt_p3/meta/stats.json`` (LeRobot v3 dataset stats,
absolute joint space). The frozen p3 checkpoint serves absolute joint targets,
so RL normalization never round-trips openpi's internal delta/quantile
convention — it is transparent to the VLA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Canonical stats file for the panthera_bolt_p3 dataset (same-origin chain:
# base p3/2000 → p3 embeddings → stage2 config must all use THESE quantiles).
PANTHERA_BOLT_P3_STATS_PATH = "/root/xsx/data/panthera_bolt_p3/meta/stats.json"

# LeRobot stats key aliases: dataset meta/stats.json uses "observation.state";
# openpi checkpoint norm_stats.json uses "state".
_STATE_KEY_ALIASES = ("observation.state", "state")


def quantile_normalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def quantile_unnormalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    return (tensor + 1.0) * denom / 2.0 + q01


def _load_json(path: Path) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_quantile_pair(
    key: str,
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    min_dim: int,
    exact: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one (q01, q99) pair; return the validated (possibly sliced) arrays."""
    if exact:
        if q01.shape != (min_dim,) or q99.shape != (min_dim,):
            raise ValueError(f"{key} quantiles must each be exactly {min_dim}D")
    else:
        if q01.ndim != 1 or q99.ndim != 1 or q01.shape[0] < min_dim or q99.shape[0] < min_dim:
            raise ValueError(f"{key} quantiles must each be at least {min_dim}D")
        q01 = q01[:min_dim]
        q99 = q99[:min_dim]
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError(f"{key} quantiles must be finite")
    if not np.all(q99 > q01):
        raise ValueError(f"{key} requires q99 > q01 elementwise")
    return q01, q99


def _extract_quantiles(stats: dict, key: str, aliases: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    for name in (key, *aliases):
        entry = stats.get(name)
        if isinstance(entry, dict) and "q01" in entry and "q99" in entry:
            return (
                np.asarray(entry["q01"], dtype=np.float32),
                np.asarray(entry["q99"], dtype=np.float32),
            )
    raise ValueError(f"Normalization stats missing {key!r} (looked for {key}/{aliases}) with q01/q99")


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
    if not path.is_file():
        raise FileNotFoundError(f"Stats path does not exist: {path}")
    raw = _load_json(path)
    if raw.get("use_relative_actions") is True:
        raise ValueError(
            "Stage 2 requires absolute joint-target actions; use_relative_actions must be false"
        )

    action_q01, action_q99 = _extract_quantiles(raw, "action", ())
    action_q01, action_q99 = _validate_quantile_pair(
        "action", action_q01, action_q99, min_dim=action_dim, exact=True,
    )
    state_q01, state_q99 = _extract_quantiles(raw, "observation.state", _STATE_KEY_ALIASES[1:])
    state_q01, state_q99 = _validate_quantile_pair(
        "observation.state", state_q01, state_q99, min_dim=state_dim, exact=False,
    )

    return QuantileStats(
        action_q01=torch.as_tensor(action_q01, dtype=torch.float32),
        action_q99=torch.as_tensor(action_q99, dtype=torch.float32),
        state_q01=torch.as_tensor(state_q01, dtype=torch.float32),
        state_q99=torch.as_tensor(state_q99, dtype=torch.float32),
        action_dim=action_dim,
        state_dim=state_dim,
        source_path=str(path.resolve()),
    )


__all__ = [
    "PANTHERA_BOLT_P3_STATS_PATH",
    "QuantileStats",
    "quantile_normalize",
    "quantile_unnormalize",
    "load_quantile_stats",
]
