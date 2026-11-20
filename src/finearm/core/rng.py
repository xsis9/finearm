"""Python RNG state capture/restore — pure standard library.

The python-random piece of the Stage-2 RNG snapshot; numpy/torch/cuda
capture composes on top in ``finearm.rl.loop.rng``. Split out so the
contract (state objects are opaque, restore is best-effort) has one home.
"""

from __future__ import annotations

import random


def capture_python_rng():
    return random.getstate()


def restore_python_rng(state) -> None:
    random.setstate(state)


__all__ = ["capture_python_rng", "restore_python_rng"]
