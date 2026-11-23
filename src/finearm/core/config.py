"""Four-layer configuration resolution, JSONC loading, and local overrides.

Layer model (the archived refactor plan (git history) §7, confirmed 2026-11-19):

1. explicit CLI arguments (the caller's argparse)  — highest
2. environment variables (explicit names per key)
3. ``configs/local.jsonc`` — machine-local persistence, gitignored
   (template: ``configs/local.example.jsonc``)
4. repo-relative built-in defaults (:mod:`finearm.core.paths`) — lowest

No machine identity is stored anywhere: no hostname matching, no per-machine
profiles. A fresh clone works with layer 4 anywhere; each machine customizes
with layers 3-1 only on that machine.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_COMMENT_PATTERN = re.compile(r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.DOTALL | re.MULTILINE)


def strip_json_comments(text: str) -> str:
    """Remove ``//`` line and ``/* */`` block comments, preserving strings.

    Same semantics as ``train_stage2._strip_json_comments``: URLs inside JSON
    strings survive, and newlines inside removed matches are preserved so
    error line numbers stay meaningful.
    """

    def repl(match: re.Match) -> str:
        s = match.group(0)
        return " " if s.startswith("/") else s.replace("\n", "\n ")

    return re.sub(_COMMENT_PATTERN, repl, text)


def load_jsonc(path: Path | str) -> dict:
    """Load a JSONC file (JSON + comments); it must contain a JSON object."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    try:
        value = json.loads(strip_json_comments(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONC in {p}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {p}")
    return value


def local_config_path(repo: Path | None = None) -> Path | None:
    """Path to ``<repo>/configs/local.jsonc`` (may not exist); None w/o repo."""
    root = repo if repo is not None else _default_repo()
    return None if root is None else root / "configs" / "local.jsonc"


def local_overrides(repo: Path | None = None) -> dict:
    """Layer 3: read ``<repo>/configs/local.jsonc``; ``{}`` when absent."""
    path = local_config_path(repo)
    if path is None or not path.is_file():
        return {}
    return load_jsonc(path)


def resolve(
    key: str,
    *,
    cli: object | None = None,
    env: str | None = None,
    overrides: dict | None = None,
    repo: Path | None = None,
) -> object | None:
    """Resolve ``key`` over layers 1-3 (CLI > env > local.jsonc); None if unset.

    ``env`` is an environment variable name; set-but-empty counts as unset.
    ``overrides`` replaces the local.jsonc layer when given (used by tests);
    ``None`` reads the file whenever a repo root is known. Layer 4 defaults
    are the caller's job (see :func:`finearm.core.paths.resolve_path`).
    """
    if cli is not None:
        return cli
    if env is not None:
        value = os.environ.get(env)
        if value:
            return value
    if overrides is None:
        overrides = local_overrides(repo)
    return overrides.get(key)


def _default_repo() -> Path | None:
    # Local import: finearm.core.paths imports this module at top level.
    from .paths import find_repo_root

    return find_repo_root()


__all__ = [
    "load_jsonc",
    "local_config_path",
    "local_overrides",
    "resolve",
    "strip_json_comments",
]
