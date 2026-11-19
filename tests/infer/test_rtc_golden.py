"""Golden comparison: the migrated RTC executor must reproduce the pre-Phase-3
rtc_client.py byte-for-byte.

The golden file was captured from the ORIGINAL implementation (2026-11-19,
before any Phase 3 edit) with the deterministic mock adapter:
``rtc_client.py --model mock --mock --host none --k 5 --blend 2 --max-steps 300``.
Filtering mirrors the capture (the adapter banner and first-chunk line carry
version-specific text). If this test fails, the RTC loop math changed —
that is a behavior regression for the live deployment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden" / "rtc_mock_300.txt"

ARGS = [
    "--model", "mock", "--mock", "--host", "none",
    "--k", "5", "--blend", "2", "--max-steps", "300",
]
FILTER = ("[RTC] 首", "适配器")


def _run() -> str:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "infer" / "rtc_client.py"), *ARGS],
        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines()
             if not any(token in line for token in FILTER)]
    return "\n".join(lines) + "\n"


def test_rtc_matches_pre_refactor_golden() -> None:
    golden = GOLDEN.read_text()
    current = _run()
    assert current == golden, (
        "RTC executor output diverged from the pre-Phase-3 golden baseline"
    )


def test_rtc_run_is_deterministic() -> None:
    assert _run() == _run()
