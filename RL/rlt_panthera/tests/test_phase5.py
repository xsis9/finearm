"""Phase 5 tests: critical-classifier labeling, metrics plotting, HIL contract."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_metrics  # noqa: E402
import train_critical_classifier as tcc  # noqa: E402


def _write_stream(path, entries):
    with open(path, "wb") as f:
        for entry in entries:
            pickle.dump(entry, f)


def test_expand_labels_segments_and_gaps(tmp_path):
    entries = [
        {"kind": "sample", "episode": 1, "step": 0, "z_rl": np.zeros(8, np.float32),
         "proprio": np.zeros(7, np.float32)},
        {"kind": "toggle", "episode": 1, "step": 10, "critical": True},
        {"kind": "sample", "episode": 1, "step": 12, "z_rl": np.ones(8, np.float32),
         "proprio": np.ones(7, np.float32)},
        {"kind": "toggle", "episode": 1, "step": 30, "critical": False},
        {"kind": "sample", "episode": 1, "step": 35, "z_rl": np.full(8, 2.0, np.float32),
         "proprio": np.full(7, 2.0, np.float32)},
        # Episode 2 has no toggles: every sample is negative.
        {"kind": "sample", "episode": 2, "step": 5, "z_rl": np.full(8, 3.0, np.float32),
         "proprio": np.full(7, 3.0, np.float32)},
    ]
    path = tmp_path / "critical_samples.pkl"
    _write_stream(path, entries)

    z, p, y, eps = tcc.expand_labels(tcc.load_stream(path))
    assert list(y) == [0.0, 1.0, 0.0, 0.0]
    assert list(eps) == [1, 1, 1, 2]
    assert z.shape == (4, 8) and p.shape == (4, 7)


def test_expand_labels_requires_toggles(tmp_path):
    path = tmp_path / "critical_samples.pkl"
    _write_stream(path, [{"kind": "sample", "episode": 1, "step": 0,
                          "z_rl": np.zeros(8, np.float32), "proprio": np.zeros(7)}])
    with pytest.raises(ValueError):
        tcc.expand_labels(tcc.load_stream(path))


def test_classifier_trains_and_saves(tmp_path):
    # Synthetic: critical ⟺ z_rl[0] > 0.
    entries = []
    for step in range(60):
        critical = step % 40 < 20
        entries.append({"kind": "sample", "episode": step // 30, "step": step,
                        "z_rl": np.array([1.5 if critical else -1.5] + [0.0] * 15, np.float32),
                        "proprio": np.zeros(7, np.float32)})
    for episode in range(2):
        entries.append({"kind": "toggle", "episode": episode, "step": 0, "critical": True})
        entries.append({"kind": "toggle", "episode": episode, "step": 20, "critical": False})
    path = tmp_path / "samples.pkl"
    _write_stream(path, entries)

    out = tmp_path / "classifier.pt"
    tcc.main.__wrapped__ = None  # not used; call internals via CLI below
    import subprocess, os
    env = dict(os.environ)
    cmd = [sys.executable, "-I", str(REPO_ROOT / "scripts" / "train_critical_classifier.py"),
           "--samples", str(path), "--output", str(out),
           "--epochs", "300", "--lr", "3e-3"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr[-2000:]

    import torch
    from rlt.critical import CriticalPhaseDetector
    detector = CriticalPhaseDetector(str(out), "cpu")
    assert detector.z_dim == 16
    # A clearly-critical z should drive the probability up, non-critical down.
    p_pos = detector._probabilities  # noqa: SLF001
    detector.update(np.array([1.5] + [0.0] * 15, np.float32), np.zeros(7, np.float32))
    prob_pos = detector.last_probability
    detector.reset()
    detector.update(np.array([-1.5] + [0.0] * 15, np.float32), np.zeros(7, np.float32))
    prob_neg = detector.last_probability
    assert prob_pos > prob_neg
    assert prob_pos > 0.7 and prob_neg < 0.3


def test_plot_metrics_parses_log(tmp_path):
    log = tmp_path / "rl_metrics.log"
    log.write_text(
        "# updates\tepisode\tbuffer\tq1_mean\tq1_max\tq2_max\ttd_mean\ttd_max\ttd_err\tactorQ\tbc\tdelta\tactor_on\n"
        "20\t1\t24\t0.11\t0.9\t0.1\t0.4\t0.9\t0.02\t\t\t\t0\n"
        "40\t1\t48\t0.13\t0.91\t0.12\t0.42\t0.9\t0.02\t0.5\t0.01\t0\t1\n"
        "# episode 1 | reward=1.0 | transitions=24 | interventions=3 | success_last20=1.00 | avg_int_last20=3.0 | buffer=24 (human=0) | updates=40\n"
    )
    episodes, rows = [], []
    header = None
    for line in log.read_text().splitlines():
        if line.startswith("# updates"):
            header = line.lstrip("# ").split("\t")
            continue
        if line.startswith("#"):
            entry = plot_metrics.parse_episode_line(line)
            if entry:
                entry["index"] = len(episodes) + 1
                episodes.append(entry)
        else:
            parts = line.split("\t")
            if header is None:
                header = parts
            else:
                rows.append({k: plot_metrics._f(v) for k, v in zip(header, parts)})
    assert episodes[0]["reward"] == 1.0 and episodes[0]["transitions"] == 24.0
    assert rows[0]["q1_mean"] == 0.11 and rows[0]["actorQ"] is None
    assert rows[1]["actorQ"] == 0.5


def test_hil_contract_hooks_exist():
    from robot.robot_interface import DummyRobot, RobotInterface

    for attr in ("poll_human", "enter_human_control", "exit_human_control",
                 "latest_human_action", "human_finished"):
        assert hasattr(RobotInterface, attr)
    dummy = DummyRobot()
    assert dummy.enter_human_control() is False
    assert dummy.exit_human_control() is None
    assert dummy.human_finished() is False  # manual exit only by default
