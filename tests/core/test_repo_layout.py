"""Repo-layout invariants (added in the 2026-11 restructure).

Two properties, both born from real incidents:

1. Every repo-relative ``DEFAULT_PATHS`` entry anchors inside the tracked
   tree. A stale default (``acquisition_interface`` kept pointing at
   ``infer/robot_interface.py`` after the src-layout move) made
   ``PantheraStage2Robot()`` raise FileNotFoundError on any machine
   without an override — this test would have caught it.
2. Directories removed by restructurings (the old ``deploy/`` tree,
   ``data/collect/``) are not referenced from tracked text files —
   stale paths in operator docs and scripts are how restructurings
   break machines silently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from finearm.core import paths

#: Substrings that must not appear in tracked text files. Built without
#: literals so this file itself never matches.
STALE_ROOTS = ("deploy" + "/", "data" + "/collect")

#: Directory names never scanned (vendored upstream, runtime products,
#: local venvs).
SKIP_DIRS = {
    ".git", "models", "__pycache__", ".pytest_cache", ".ruff_cache",
    "datasets", "logs", "checkpoints", "policy_records", "sdk",
}

SCAN_SUFFIXES = {
    ".md", ".sh", ".py", ".yml", ".yaml", ".toml", ".json", ".jsonc",
    ".example", ".cfg",
}

#: The migration plan itself (talks about old paths by definition) and
#: test files (they legitimately mention stale strings as guard
#: arguments, e.g. assertNotIn).
SCAN_EXEMPT_NAMES = {"RESTRUCTURE_PLAN.md"}
SCAN_EXEMPT_PREFIXES = ("test_", "conftest")


def _repo() -> Path:
    root = paths.repo_root()
    if root is None:
        pytest.skip("no finearm checkout anchored (package installed elsewhere)")
    return root


class TestDefaultPathsAnchor:
    def test_relative_defaults_stay_inside_tracked_tree(self) -> None:
        repo = _repo()
        for key, raw in paths.DEFAULT_PATHS.items():
            p = Path(raw)
            if p.is_absolute():
                continue
            top = repo / p.parts[0]
            assert top.is_dir(), (
                f"DEFAULT_PATHS[{key}] = {raw!r}: top-level {p.parts[0]!r} "
                "missing from the checkout — dangling repo-path default"
            )
            if p.suffix:
                # A default that names a FILE must point at tracked code.
                assert (repo / p).is_file(), (
                    f"DEFAULT_PATHS[{key}] = {raw!r}: file missing from the checkout"
                )

    def test_acquisition_backend_has_no_repo_path_default(self) -> None:
        # Its default is the finearm.infer.robot_interface package module;
        # a repo-path default here dangles silently after restructures.
        assert "acquisition_interface" not in paths.DEFAULT_PATHS


class TestNoStaleRootReferences:
    def test_removed_directories_are_not_referenced(self) -> None:
        repo = _repo()
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(repo):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".venv")
            ]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix not in SCAN_SUFFIXES:
                    continue
                if name in SCAN_EXEMPT_NAMES or name.startswith(SCAN_EXEMPT_PREFIXES):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for stale in STALE_ROOTS:
                    if stale in text:
                        hits.append(f"{path.relative_to(repo)}: {stale!r}")
        assert not hits, "stale root references in tracked files:\n" + "\n".join(hits)
