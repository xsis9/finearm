"""Full RNG snapshot (python + numpy + torch + cuda) — 5b extraction.

Moved verbatim from ``finearm.rl.algorithms.rltoken.learner``; the
python-random piece delegates to ``finearm.core.rng``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from finearm.core.rng import capture_python_rng, restore_python_rng

logger = logging.getLogger(__name__)


def capture_rng_state() -> dict:
    return {
        "python": capture_python_rng(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    # Byte-tensor RNG states must be on CPU: torch.load(map_location="cuda")
    # would otherwise move them to the GPU and torch.set_rng_state raises
    # "RNG state must be a torch.ByteTensor". RNG exactness is best-effort only.
    try:
        restore_python_rng(state["python"])
        np.random.set_state(state["numpy"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("python/numpy RNG state could not be restored: %s", exc)
    torch_state = state.get("torch")
    if torch_state is not None:
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().to("cpu")
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch RNG state could not be restored: %s", exc)
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([
                s.detach().to("cpu") if torch.is_tensor(s) else s for s in cuda_states
            ])
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA RNG state could not be restored: %s", exc)


__all__ = ["capture_rng_state", "restore_rng_state"]
