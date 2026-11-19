"""Shared fixtures: a fake finearm checkout with the standard default layout."""

from __future__ import annotations

from pathlib import Path

import pytest

PYPROJECT = '[project]\nname = "finearm"\nversion = "0.1.0"\n'


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "finearm-checkout"
    (repo / "configs").mkdir(parents=True)
    (repo / "data" / "datasets").mkdir(parents=True)
    (repo / "arm" / "panthera" / "sdk").mkdir(parents=True)
    (repo / "infer").mkdir(parents=True)
    (repo / "infer" / "robot_interface.py").write_text("# acquisition interface\n")
    (repo / "pyproject.toml").write_text(PYPROJECT)
    return repo
