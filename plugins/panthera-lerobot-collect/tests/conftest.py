"""Package test bootstrap: make the repo's finearm package importable.

The collector implementation lives in <repo>/src/finearm/data; this package
keeps the LeRobot plugins, wrapper scripts, and their tests. The repo root is
found by marker (pyproject ``name = "finearm"``), not by counting directory
levels — see repo_paths.py.
"""
import sys

from repo_paths import repo_root

_repo_src = repo_root() / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))
