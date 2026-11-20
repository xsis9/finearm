"""Atomic file-write and output-dir primitives — pure standard library.

Extracted from the Stage-2 learner in Phase 5b so the durability mechanics
(temp file + ``os.replace`` crash atomicity, fsync, fail-closed output dirs)
have exactly one definition, reusable by any domain (replay journal appends,
checkpoints, metrics dumps). Torch-aware wrappers stay in
``finearm.rl.loop.checkpoint``; this module never imports numpy/torch.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def fsync_file(f) -> None:
    """Flush and fsync an open binary file (durable append)."""
    f.flush()
    os.fsync(f.fileno())


def atomic_save(path: Path, writer) -> None:
    """Crash-atomic save: ``writer(tmp_path)`` then ``os.replace`` onto path.

    A crash leaves at most the tmp file (cleaned on the next attempt or by
    the finally block); the destination is never a partial write.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_empty_output_dir(path: Path, *, create: bool) -> None:
    """Fail closed on prior contents; ``create=False`` mutates nothing."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--output_dir is not a directory: {path}")
        contents = list(path.iterdir())
        if contents:
            raise FileExistsError(
                f"--output_dir already contains {len(contents)} item(s): {path}. "
                "Stage 2 resume is unsupported; choose a new empty directory."
            )
    if create:
        path.mkdir(parents=True, exist_ok=True)


__all__ = ["atomic_save", "ensure_empty_output_dir", "fsync_file"]
