"""Stage-2 checkpoint schema + save/restore machinery (5b extraction).

Moved from ``finearm.rl.algorithms.rltoken.learner`` in Phase 5b: the
learner keeps only the training algorithm; durability mechanics come from
``finearm.core.io`` (atomic replace, fail-closed output dirs). Bodies are
verbatim unless noted; ``prepare_output_dir`` delegates to the core
primitive with identical messages.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from finearm.core.io import atomic_save, ensure_empty_output_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
ACTOR_CONTRACT = "paper_full_output_v1"
RL_CHUNK_LENGTH = 10


def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_stage2_checkpoint(**payload) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "resume_supported": True,
        "actor_contract": ACTOR_CONTRACT,
        "rl_chunk_length": RL_CHUNK_LENGTH,
        **payload,
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    atomic_save(path, lambda tmp: torch.save(payload, tmp))


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def prepare_output_dir(path: Path, *, dry_run: bool) -> None:
    """Fail closed on prior contents; dry-run performs no filesystem mutation."""
    ensure_empty_output_dir(path, create=not dry_run)


def prepare_resume_checkpoint(path: Path, output_dir: Path, *, dry_run: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"--resume_from checkpoint does not exist: {path}")
    if path.parent.resolve() == output_dir.resolve():
        raise ValueError("--output_dir must be a new directory, not the resume checkpoint directory.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict", "actor_target_state_dict", "critic_state_dict", "critic_target_state_dict",
        "actor_optimizer", "critic_optimizer", "episode", "episode_rewards", "episode_interventions",
        "total_updates", "rng_state", "replay_rng_state", "replay_journal_bytes", "replay_entries",
        "runtime_provenance",
    }
    missing = sorted(required - set(checkpoint))
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or not checkpoint.get("resume_supported")
        or checkpoint.get("actor_contract") != ACTOR_CONTRACT
        or checkpoint.get("rl_chunk_length") != RL_CHUNK_LENGTH
        or missing
    ):
        raise ValueError(
            "Checkpoint is not a complete paper-aligned full-output Stage 2 checkpoint; start a new run. "
            f"Missing: {missing}"
        )
    source_journal = path.parent / "replay_journal.pkl"
    if not source_journal.is_file():
        raise FileNotFoundError(f"Resume checkpoint requires replay journal: {source_journal}")
    journal_bytes = checkpoint["replay_journal_bytes"]
    if not isinstance(journal_bytes, int) or journal_bytes < 0 or source_journal.stat().st_size < journal_bytes:
        raise ValueError("Resume checkpoint replay journal boundary is invalid or source journal is truncated.")
    if not dry_run:
        destination_journal = output_dir / "replay_journal.pkl"
        with source_journal.open("rb") as source, destination_journal.open("wb") as destination:
            remaining = journal_bytes
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("Resume source journal ended before the checkpoint boundary.")
                destination.write(block)
                remaining -= len(block)
    return checkpoint


__all__ = [
    "SCHEMA_VERSION", "ACTOR_CONTRACT", "RL_CHUNK_LENGTH",
    "canonical_digest", "build_stage2_checkpoint", "atomic_torch_save",
    "move_optimizer_state_to_device", "prepare_output_dir",
    "prepare_resume_checkpoint",
]
