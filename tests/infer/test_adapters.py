"""Adapter registry + shim equivalence for the infer domain."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from finearm.infer.adapters import available, load, mock_obs

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registry_lists_shipped_adapters() -> None:
    names = available()
    assert "mock" in names
    assert "pi05" in names
    assert "lingbot" in names
    assert "__init__" not in names


def test_mock_adapter_deterministic_by_call_index() -> None:
    adapter = load("mock", host=None)
    adapter.reset()
    c0 = adapter.infer({})
    c0_again = adapter.infer({})  # call 1 now
    adapter.reset()
    d0 = adapter.infer({})
    assert c0.shape == (30, 7)
    np.testing.assert_array_equal(c0, d0)          # reset restores call 0
    assert not np.array_equal(c0, c0_again)        # successive calls differ


def test_mock_obs_styles() -> None:
    obs = mock_obs("openpi", seed=1)
    assert obs["observation/image"].shape == (224, 224, 3)
    assert obs["observation/state"].shape == (7,)
    raw = mock_obs("raw", seed=1)
    assert raw["front"].shape == (480, 640, 3)
    assert raw["state"].shape == (14,)


def test_lingbot_repo_resolution_or_skip() -> None:
    checkout = REPO_ROOT / "models" / "lingbot-vla-v2"
    if not checkout.is_dir():
        return  # fresh clone without the gitignored model checkout — resolution raises
    import finearm.infer.adapters.lingbot as lingbot

    assert lingbot._REPO == checkout
