"""Package test bootstrap: make the repo's finearm package importable.

The collector implementation lives in <repo>/src/finearm/data; this package
keeps the LeRobot plugins, wrapper scripts, and their tests.
"""
import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[3] / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))
