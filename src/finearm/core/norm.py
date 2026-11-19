"""Quantile (q01/q99) normalization stats: on-disk format contract.

Pure standard library, so any environment — including the 4090 openpi venv —
can load and validate a stats file. Tensor wrappers stay in the domain
packages (``RL/rlt_panthera/rlt/norm.py`` today; ``finearm.rl`` after Phase
5a): this module owns the format so the contract has exactly one definition.

Accepted formats (same as ``rlt/norm.py``):

- LeRobot dataset ``meta/stats.json``: ``action`` / ``observation.state``
  entries with ``q01``/``q99`` lists.
- openpi checkpoint ``assets/norm_stats.json``: ``action`` / ``state``.

Action quantiles must be exactly ``action_dim``; state quantiles at least
``state_dim`` and are sliced to their first ``state_dim`` entries (positions).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# LeRobot stats key aliases: dataset meta/stats.json uses "observation.state";
# openpi checkpoint norm_stats.json uses "state".
_STATE_KEY_ALIASES = ("observation.state", "state")


@dataclass(frozen=True)
class QuantilePairs:
    """Validated q01/q99 plain-float pairs for action (exact) and state (sliced)."""

    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    state_q01: tuple[float, ...]
    state_q99: tuple[float, ...]
    action_dim: int
    state_dim: int
    source_path: str = ""


def _as_floats(key: str, values: object) -> list[float]:
    try:
        return [float(v) for v in values]  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} quantiles must be numeric") from exc


def _extract_quantiles(
    stats: dict, key: str, aliases: tuple[str, ...]
) -> tuple[list[float], list[float]]:
    for name in (key, *aliases):
        entry = stats.get(name)
        if isinstance(entry, dict) and "q01" in entry and "q99" in entry:
            return _as_floats(name, entry["q01"]), _as_floats(name, entry["q99"])
    raise ValueError(f"Normalization stats missing {key!r} (looked for {key}/{aliases}) with q01/q99")


def _validate_quantile_pair(
    key: str,
    q01: list[float],
    q99: list[float],
    *,
    min_dim: int,
    exact: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate one (q01, q99) pair; return the validated (possibly sliced) tuples."""
    if exact:
        if len(q01) != min_dim or len(q99) != min_dim:
            raise ValueError(f"{key} quantiles must each be exactly {min_dim}D")
    else:
        if len(q01) < min_dim or len(q99) < min_dim:
            raise ValueError(f"{key} quantiles must each be at least {min_dim}D")
        q01, q99 = q01[:min_dim], q99[:min_dim]
    if not all(math.isfinite(v) for v in (*q01, *q99)):
        raise ValueError(f"{key} quantiles must be finite")
    if not all(b > a for a, b in zip(q01, q99)):
        raise ValueError(f"{key} requires q99 > q01 elementwise")
    return tuple(q01), tuple(q99)


def load_quantile_pairs(
    stats_path: Path | str,
    *,
    action_dim: int = 7,
    state_dim: int = 7,
) -> QuantilePairs:
    """Load and validate q01/q99 stats; returns plain floats (no tensors).

    Rejects relative-action stats (Stage 2 / RL require absolute joint
    targets).
    """
    path = Path(stats_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Stats path does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if raw.get("use_relative_actions") is True:
        raise ValueError(
            "Stage 2 requires absolute joint-target actions; use_relative_actions must be false"
        )

    action_q01, action_q99 = _extract_quantiles(raw, "action", ())
    action_q01, action_q99 = _validate_quantile_pair(
        "action", action_q01, action_q99, min_dim=action_dim, exact=True
    )
    state_q01, state_q99 = _extract_quantiles(raw, "observation.state", _STATE_KEY_ALIASES[1:])
    state_q01, state_q99 = _validate_quantile_pair(
        "observation.state", state_q01, state_q99, min_dim=state_dim, exact=False
    )

    return QuantilePairs(
        action_q01=action_q01,
        action_q99=action_q99,
        state_q01=state_q01,
        state_q99=state_q99,
        action_dim=action_dim,
        state_dim=state_dim,
        source_path=str(path.resolve()),
    )


__all__ = ["QuantilePairs", "load_quantile_pairs"]
