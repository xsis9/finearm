"""Critical-phase classifier / detector tests (ported from
tests/test_critical_classifier.py; classifier/behavior portions only — the
recorder lives in the Phase-5 training script)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from finearm.rl.algorithms.rltoken.critical import CriticalPhaseClassifier, CriticalPhaseDetector


def test_classifier_output_shape():
    model = CriticalPhaseClassifier(z_dim=8, proprio_dim=7, hidden_dim=16)
    z = torch.randn(4, 8)
    p = torch.randn(4, 7)
    logits = model(z, p)
    assert tuple(logits.shape) == (4, 2)


def test_classifier_full_size_panthera_input():
    """Real dims: z_rl 2048 + proprio 7."""
    model = CriticalPhaseClassifier(z_dim=2048, proprio_dim=7)
    logits = model(torch.randn(2, 2048), torch.randn(2, 7))
    assert tuple(logits.shape) == (2, 2)


def _save_checkpoint(directory, model, p_train, z_dim, proprio_dim):
    checkpoint_path = Path(directory) / "cls.pt"
    torch.save({
        "classifier_state_dict": model.state_dict(),
        "proprio_mean": p_train.mean(0).numpy().astype(np.float32),
        "proprio_std": p_train.std(0).numpy().astype(np.float32) + 1e-6,
        "z_dim": z_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }, checkpoint_path)
    return checkpoint_path


def test_detector_loads_checkpoint_and_applies_hysteresis():
    torch.manual_seed(0)  # classifier training below is seed-sensitive
    z_dim, proprio_dim = 4, 2
    model = CriticalPhaseClassifier(z_dim=z_dim, proprio_dim=proprio_dim, hidden_dim=8)
    # Train a trivially separable classifier: z[:, 0] > 0 -> critical.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    z_train = torch.cat([torch.randn(64, z_dim) + 1.0, torch.randn(64, z_dim) - 1.0])
    p_train = torch.randn(128, proprio_dim)
    labels = torch.cat([torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(z_train, p_train), labels)
        loss.backward()
        optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = _save_checkpoint(directory, model, p_train, z_dim, proprio_dim)
        detector = CriticalPhaseDetector(
            str(checkpoint_path), "cpu",
            threshold_on=0.7, threshold_off=0.3, smooth_steps=5,
        )
        # Positive z -> on after a few updates; negative z -> off.
        state = None
        for _ in range(10):
            candidate = detector.update(
                np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )
            if candidate is not None:
                state = candidate
        assert state is True
        detector.reset()
        state = None
        for _ in range(10):
            candidate = detector.update(
                np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )
            if candidate is not None:
                state = candidate
        assert state is False


def test_detector_debounce_holds_on_through_brief_dip():
    """Once ON, a single below-off sample inside min_on_chunks must not flip OFF."""
    torch.manual_seed(1)
    z_dim, proprio_dim = 4, 2
    model = CriticalPhaseClassifier(z_dim=z_dim, proprio_dim=proprio_dim, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    z_train = torch.cat([torch.randn(64, z_dim) + 1.0, torch.randn(64, z_dim) - 1.0])
    p_train = torch.randn(128, proprio_dim)
    labels = torch.cat([torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(z_train, p_train), labels)
        loss.backward()
        optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = _save_checkpoint(directory, model, p_train, z_dim, proprio_dim)
        detector = CriticalPhaseDetector(
            str(checkpoint_path), "cpu",
            threshold_on=0.7, threshold_off=0.3, smooth_steps=5, min_on_chunks=3,
        )
        positive = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        negative = np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        proprio = np.array([0.0, 0.0], dtype=np.float32)
        assert detector.update(positive, proprio) is True
        # immediate dip: debounced (returns None), still ON
        assert detector.update(negative, proprio) is None


def test_detector_rejects_wrong_z_dim():
    with tempfile.TemporaryDirectory() as directory:
        model = CriticalPhaseClassifier(z_dim=4, proprio_dim=2, hidden_dim=8)
        checkpoint_path = _save_checkpoint(
            directory, model, torch.randn(8, 2), z_dim=4, proprio_dim=2,
        )
        detector = CriticalPhaseDetector(str(checkpoint_path), "cpu")
        with pytest.raises(ValueError):
            detector.update(np.zeros(8, dtype=np.float32), np.zeros(2, dtype=np.float32))
