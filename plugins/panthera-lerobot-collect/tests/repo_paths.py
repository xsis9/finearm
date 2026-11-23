"""Marker-based repo-root discovery for plugin tests.

Mirrors ``finearm.core.paths``: walk up from this file until a pyproject.toml
declares ``name = "finearm"``. Tests anchor every repo-relative path on this
instead of counting ``parents[N]`` levels, so the plugin directory can be
relocated without the tests silently re-anchoring to the wrong tree.
"""
from __future__ import annotations

import tomllib
from pathlib import Path


def repo_root(start: Path | str | None = None) -> Path:
    """Locate the finearm repository root above ``start`` (default: here)."""
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if data.get("project", {}).get("name") == "finearm":
            return candidate
    raise AssertionError(f"finearm repo root not found above {current}")
