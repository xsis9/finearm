"""Repository-root discovery and built-in repo-relative default paths.

Layer 4 of the four-layer configuration model (REFACTOR_PLAN.md §7):
CLI arguments > environment variables > ``configs/local.jsonc`` >
repo-relative defaults (this module). The repo stores no machine identity:
any clone on any machine works out of the box and defaults follow the
checkout location.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from . import config

#: Environment override for the repo root. Checked first, so an environment
#: where finearm is pip-installed can still resolve a separate checkout.
ROOT_ENV = "FINEARM_ROOT"

#: Built-in defaults (layer 4). Relative entries resolve against the repo
#: root; absolute entries are used as-is.
DEFAULT_PATHS: dict[str, str] = {
    "dataset_root": "data/datasets",
    "sdk_root": "arm/panthera/sdk",
    "acquisition_interface": "infer/robot_interface.py",
    "socket_dir": "/tmp",
}

def _is_repo_root(candidate: Path) -> bool:
    """True when ``candidate/pyproject.toml`` declares ``name = "finearm"``."""
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return data.get("project", {}).get("name") == "finearm"


def find_repo_root(start: Path | str | None = None) -> Path | None:
    """Locate the finearm repository root.

    Resolution order: ``FINEARM_ROOT`` env var > walk up from ``start``
    (default: this file's location) to the first directory whose
    ``pyproject.toml`` declares ``name = "finearm"``. Returns ``None`` when
    no root is found (e.g. finearm installed into a venv without a checkout).
    """
    env_root = os.environ.get(ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser()
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _is_repo_root(candidate):
            return candidate
    return None


def repo_root() -> Path | None:
    """Repo root of the checkout this module belongs to (``None`` if none)."""
    return find_repo_root()


def default_path(key: str, repo: Path | None = None) -> Path | None:
    """Built-in default for ``key`` (layer 4 only; no env/local.jsonc lookup).

    Returns ``None`` for unknown keys, and for relative defaults when no repo
    root can be anchored (``repo`` given or discovered).
    """
    raw = DEFAULT_PATHS.get(key)
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    root = repo if repo is not None else find_repo_root()
    if root is None:
        return None
    return root / path


def resolve_path(
    key: str,
    *,
    cli: str | Path | None = None,
    env: str | None = None,
    overrides: dict | None = None,
    repo: Path | None = None,
) -> Path | None:
    """Resolve ``key`` through the four layers; ``None`` when all are unset.

    ``cli``: explicit caller value (highest). ``env``: environment variable
    name (set-but-empty counts as unset). ``overrides``: local.jsonc content;
    ``None`` reads ``<repo>/configs/local.jsonc`` when a repo root is known.
    The final layer is the built-in default for ``key`` (:data:`DEFAULT_PATHS`).
    """
    root = repo if repo is not None else find_repo_root()
    value = config.resolve(key, cli=cli, env=env, overrides=overrides, repo=root)
    if value is None:
        return default_path(key, repo=root)
    return Path(value).expanduser()


__all__ = [
    "DEFAULT_PATHS",
    "ROOT_ENV",
    "default_path",
    "find_repo_root",
    "repo_root",
    "resolve_path",
]
