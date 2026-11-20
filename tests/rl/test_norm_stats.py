"""Quantile normalization tests: math round-trip, 7-dim validation, LeRobot
stats.json loading (14-dim state sliced to 7), and the real panthera_bolt_p3
stats file when present.
"""

import json
from pathlib import Path

import pytest
import torch

from rlt.norm import (
    load_quantile_stats,
    quantile_normalize,
    quantile_unnormalize,
)

# Optional real-stats integration: set FINEARM_REAL_STATS_PATH to a
# panthera_bolt_p3 meta/stats.json to exercise the loader on real data
# (the dropped PANTHERA_BOLT_P3_STATS_PATH constant was machine-absolute;
# Phase 5a made the path an explicit environment opt-in instead).
import os

_REAL_STATS_PATH = os.environ.get("FINEARM_REAL_STATS_PATH")


def test_quantile_round_trip_identity():
    q01 = torch.tensor([0.0, -1.0])
    q99 = torch.tensor([2.0, 1.0])
    x = torch.tensor([[0.5, 0.0], [1.5, -0.5]])
    normalized = quantile_normalize(x, q01, q99)
    # in-range values map into [-1, 1]; the 1.5 (< q99=2.0, first column) row is in range
    assert torch.all(normalized <= 1.0 + 1e-6) and torch.all(normalized >= -1.0 - 1e-6)
    torch.testing.assert_close(quantile_unnormalize(normalized, q01, q99), x)


def test_out_of_range_values_exceed_unit_interval():
    q01 = torch.tensor([0.0])
    q99 = torch.tensor([1.0])
    below = quantile_normalize(torch.tensor([-1.0]), q01, q99)
    above = quantile_normalize(torch.tensor([2.0]), q01, q99)
    assert float(below) < -1.0 and float(above) > 1.0


def test_zero_width_quantile_does_not_divide_by_zero():
    q01 = torch.tensor([1.0])
    q99 = torch.tensor([1.0])  # degenerate (rejected by loader, tolerated by math)
    out = quantile_normalize(torch.tensor([1.0]), q01, q99)
    assert torch.isfinite(out).all()


def _stats_file(tmp_path, action_width=7, state_width=14, relative=False, **overrides):
    payload = {
        "action": {
            "q01": [0.0] * action_width,
            "q99": [1.0] * action_width,
        },
        "observation.state": {
            "q01": [0.0] * state_width,
            "q99": [2.0] * state_width,
        },
    }
    if relative:
        payload["use_relative_actions"] = True
    payload.update(overrides)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_quantile_stats_slices_full_state_width(tmp_path):
    stats = load_quantile_stats(_stats_file(tmp_path), action_dim=7, state_dim=7)
    assert tuple(stats.action_q01.shape) == (7,)
    assert tuple(stats.state_q01.shape) == (7,)  # 14 -> first 7 (positions)
    assert stats.state_q99.tolist() == [2.0] * 7
    # helpers round-trip through the loaded quantiles
    action = torch.rand(2, 10, 7)
    torch.testing.assert_close(
        stats.unnormalize_action(stats.normalize_action(action)), action, rtol=1e-5, atol=1e-6,
    )
    state = torch.rand(2, 7)
    torch.testing.assert_close(
        stats.unnormalize_state(stats.normalize_state(state)), state, rtol=1e-5, atol=1e-6,
    )


def test_load_accepts_openpi_state_key_alias(tmp_path):
    payload = {
        "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},  # openpi norm_stats key
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(payload))
    stats = load_quantile_stats(path)
    assert tuple(stats.state_q01.shape) == (7,)


def test_rejects_wrong_action_width(tmp_path):
    # The reference repo validated "exactly 6D"; this port validates "exactly 7D".
    with pytest.raises(ValueError, match="exactly 7D"):
        load_quantile_stats(_stats_file(tmp_path, action_width=6))


def test_rejects_short_state(tmp_path):
    with pytest.raises(ValueError, match="at least 7D"):
        load_quantile_stats(_stats_file(tmp_path, state_width=5))


def test_rejects_q99_not_greater_than_q01(tmp_path):
    path = _stats_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["action"]["q99"] = [0.0] * 7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="q99 > q01"):
        load_quantile_stats(path)


def test_rejects_non_finite_quantiles(tmp_path):
    path = _stats_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["action"]["q01"][3] = None
    path.write_text(json.dumps(payload))
    # Phase 5a: parsing lives in finearm.core.norm — a None entry now fails
# coercion ("must be numeric") instead of numpy's silent None→nan
# ("must be finite"). Both are ValueError; type is the contract.
    with pytest.raises(ValueError, match="finite|numeric"):
        load_quantile_stats(path)


def test_rejects_relative_actions(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        load_quantile_stats(_stats_file(tmp_path, relative=True))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_quantile_stats(tmp_path / "nope.json")


def _p3_stats_available() -> bool:
    if not _REAL_STATS_PATH:
        return False
    # Untraversable prefixes raise OSError, which means "not present".
    try:
        return Path(_REAL_STATS_PATH).is_file()
    except OSError:
        return False


@pytest.mark.skipif(
    not _p3_stats_available(),
    reason="FINEARM_REAL_STATS_PATH not set or file missing",
)
def test_real_panthera_p3_stats_load():
    """Integration: the actual panthera_bolt_p3 meta/stats.json (absolute space)
    loads with exact 7-dim action quantiles and 14->7 state slicing."""
    stats = load_quantile_stats(_REAL_STATS_PATH)
    assert tuple(stats.action_q01.shape) == (7,)
    assert tuple(stats.state_q01.shape) == (7,)
    assert torch.all(stats.action_q99 > stats.action_q01)
    assert torch.all(stats.state_q99 > stats.state_q01)
    # Absolute joint space sanity: positions inside the dataset range normalize
    # into (roughly) [-1, 1].
    mid = (stats.action_q01 + stats.action_q99) / 2
    assert torch.all(quantile_normalize(mid, stats.action_q01, stats.action_q99).abs() < 0.1)
