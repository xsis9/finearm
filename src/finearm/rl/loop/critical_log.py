"""Critical-phase logging (Stage-1.5 classifier training data) — 5b move.

``CriticalPhaseRecorder`` and ``save_critical_transition`` moved verbatim
from ``train_stage2.py`` (Phase 5b): both the training entry and the
episode runner need them; a separate module breaks the import cycle.
"""

from __future__ import annotations

import csv
import logging
import pickle
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CriticalPhaseRecorder:
    """Append-only pickle stream of {kind: sample|toggle} dicts (ported)."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._f = open(self._path, "ab")

    def record_sample(self, episode, step, z_rl, proprio) -> None:
        pickle.dump({
            "kind": "sample", "episode": int(episode), "step": int(step),
            "t": time.time(), "z_rl": np.asarray(z_rl, dtype=np.float32),
            "proprio": np.asarray(proprio, dtype=np.float32),
        }, self._f, protocol=pickle.HIGHEST_PROTOCOL)
        self._f.flush()

    def record_toggle(self, episode, step, critical) -> None:
        pickle.dump({
            "kind": "toggle", "episode": int(episode), "step": int(step),
            "t": time.time(), "critical": bool(critical),
        }, self._f, protocol=pickle.HIGHEST_PROTOCOL)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except (ValueError, OSError):
            pass

    @property
    def path(self) -> Path:
        return self._path


def save_critical_transition(frames_dir, camera_names, *, episode, step, direction,
                             source, prob, obs) -> None:
    """Persist one critical on/off transition (CSV row + camera jpgs)."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    events_path = frames_dir / "critical_events.csv"
    if not events_path.exists():
        with open(events_path, "w", newline="") as file:
            csv.writer(file).writerow(["episode", "step", "direction", "source", "prob", "wall_time"])
    with open(events_path, "a", newline="") as file:
        csv.writer(file).writerow([
            int(episode), int(step), direction, source,
            "" if prob is None else f"{prob:.3f}",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
    try:
        import cv2 as _cv2
    except ImportError:
        return
    for cam in camera_names:
        image = obs.get(cam) if isinstance(obs, dict) else None
        if image is None:
            continue
        fname = f"ep{int(episode):03d}_step{int(step):05d}_{direction}_{source}_{cam}.jpg"
        try:
            frame = np.asarray(image)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(str(frames_dir / fname), frame)
        except Exception as exc:  # noqa: BLE001 - saving is best-effort
            logger.warning("Critical frame save failed (%s): %s", fname, exc)




__all__ = ["CriticalPhaseRecorder", "save_critical_transition"]
