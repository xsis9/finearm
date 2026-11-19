"""Tests for finearm.core.norm: stats format contract (pure stdlib).

Validation semantics must stay equivalent to ``RL/rlt_panthera/rlt/norm.py``
(the active implementation until Phase 5a switches it onto this module).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finearm.core import norm


def _stats_file(tmp_path: Path, stats: dict) -> Path:
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(stats))
    return p


def _valid_stats() -> dict:
    return {
        "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "observation.state": {"q01": [0.0] * 14, "q99": [2.0] * 14},
    }


def test_load_slices_state_to_state_dim(tmp_path: Path) -> None:
    pairs = norm.load_quantile_pairs(_stats_file(tmp_path, _valid_stats()))
    assert pairs.action_dim == 7
    assert pairs.state_dim == 7
    assert pairs.action_q01 == (0.0,) * 7
    assert pairs.action_q99 == (1.0,) * 7
    assert pairs.state_q01 == (0.0,) * 7
    assert pairs.state_q99 == (2.0,) * 7


def test_source_path_is_resolved_absolute(tmp_path: Path) -> None:
    p = _stats_file(tmp_path, _valid_stats())
    assert norm.load_quantile_pairs(p).source_path == str(p.resolve())


def test_openpi_state_alias(tmp_path: Path) -> None:
    stats = {
        "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
    }
    pairs = norm.load_quantile_pairs(_stats_file(tmp_path, stats))
    assert pairs.state_q99 == (1.0,) * 7


def test_action_dim_must_be_exact(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["action"] = {"q01": [0.0] * 6, "q99": [1.0] * 6}
    with pytest.raises(ValueError, match="exactly 7"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_state_dim_must_be_at_least_min(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["observation.state"] = {"q01": [0.0] * 5, "q99": [1.0] * 5}
    with pytest.raises(ValueError, match="at least 7"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_q99_must_exceed_q01(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["action"] = {"q01": [0.0] * 7, "q99": [0.0] * 7}
    with pytest.raises(ValueError, match="q99 > q01"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_non_finite_rejected(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["action"] = {"q01": [float("nan")] * 7, "q99": [1.0] * 7}
    with pytest.raises(ValueError, match="finite"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_non_numeric_rejected(tmp_path: Path) -> None:
    stats = _valid_stats()
    stats["action"] = {"q01": ["x"] * 7, "q99": [1.0] * 7}
    with pytest.raises(ValueError, match="numeric"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_relative_actions_rejected(tmp_path: Path) -> None:
    stats = _valid_stats() | {"use_relative_actions": True}
    with pytest.raises(ValueError, match="absolute"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_missing_action_key(tmp_path: Path) -> None:
    stats = {"observation.state": {"q01": [0.0] * 14, "q99": [1.0] * 14}}
    with pytest.raises(ValueError, match="action"):
        norm.load_quantile_pairs(_stats_file(tmp_path, stats))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        norm.load_quantile_pairs(tmp_path / "nope.json")
