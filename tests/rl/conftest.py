"""tests/rl — Stage-2 RL suite (migrated from RL/rlt_panthera/tests in 5b).

Runs from the repo root with the infer venv (torch + numpy + pytest). The
legacy import roots stay on sys.path so tests exercising the shim chain
(rlt.*, robot.*, scripts) keep working until Phase 6 removes the shims.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (
    REPO_ROOT / "src",
    REPO_ROOT / "RL" / "rlt_panthera",
    REPO_ROOT / "RL" / "rlt_panthera" / "scripts",
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
