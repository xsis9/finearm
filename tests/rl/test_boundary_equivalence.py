"""Phase 5b red line: rtc-mode transition stream is refactor-invariant.

Two layers (design confirmed with the user 2026-11-20):
1. Seeded-scenario golden: the full train_stage2 loop runs offline
   (DummyRobot + DeterministicVLA + scripted stdin + --vla_only + fixed k);
   the replay journal must stay byte-identical across the interface
   refactor. Run with FINEARM_GOLDEN_FREEZE=1 to (re)write the golden file
   — only legitimate when the implementation semantics intentionally
   changed and the change was reviewed against this test.
2. Real-journal schema lock: the v3_1 campaign journal (machine-local,
   env-gated) defines the Transition record contract — field names, dtypes,
   stride-2 shapes — independent of implementation.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from scenarios import (
    GOLDEN_DIR,
    golden_path,
    run_scenario,
    scenario_inputs_available,
)

pytestmark = pytest.mark.skipif(
    not scenario_inputs_available(),
    reason="needs FINEARM_STAGE1_CKPT + FINEARM_REAL_STATS_PATH (machine-local inputs)",
)


def test_rtc_golden_equivalence(tmp_path):
    name = "rtc_basic"
    result = run_scenario(name, tmp_path / "run")

    golden = golden_path(name)
    if os.environ.get("FINEARM_GOLDEN_FREEZE") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(result["journal_bytes"])
        digest = hashlib.sha256(result["journal_bytes"]).hexdigest()
        (golden.parent / "sha256").write_text(digest + "\n")
        pytest.skip(f"golden frozen ({len(result['records'])} records, sha256 {digest[:12]})")

    assert golden.is_file(), (
        "golden journal missing — freeze first with FINEARM_GOLDEN_FREEZE=1 "
        "on the pre-refactor implementation"
    )
    expected = golden.read_bytes()
    if result["journal_bytes"] == expected:
        return
    # Byte mismatch: diagnose field-wise before failing.
    want = _load_all(expected)
    got = result["records"]
    assert len(want) == len(got), (
        f"record count changed: golden {len(want)} vs now {len(got)}"
    )
    for i, (w, g) in enumerate(zip(want, got)):
        assert set(w) == set(g), f"record {i}: field set changed"
        for key in w:
            if isinstance(w[key], np.ndarray):
                assert w[key].dtype == g[key].dtype, f"record {i}.{key}: dtype"
                assert w[key].shape == g[key].shape, f"record {i}.{key}: shape"
                assert np.array_equal(w[key], g[key]), f"record {i}.{key}: values"
            else:
                assert w[key] == g[key], f"record {i}.{key}"
    pytest.fail("journals differ byte-wise but match field-wise — "
                "pickle nondeterminism (investigate before accepting)")


def _load_all(payload: bytes):
    records = []
    import io
    f = io.BytesIO(payload)
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            return records


def test_real_journal_schema_lock():
    """v3_1 campaign journal defines the Transition record contract."""
    journal = Path(os.environ.get(
        "FINEARM_GOLDEN_JOURNAL",
        "RL/rlt_panthera/checkpoints/stage2_v3_1/replay_journal.pkl",
    ))
    if not journal.is_file():
        pytest.skip("campaign journal not present")
    records = _load_all(journal.read_bytes())
    assert records, "journal must not be empty"
    expected_fields = {
        "z_rl", "proprio", "ref_chunk", "action_chunk", "rewards", "done",
        "next_z_rl", "next_proprio", "next_ref_chunk", "source",
        "source_chunk", "valid_mask", "executed_steps", "collection_phase_id",
        "success", "intervention_flag", "episode_id", "step_id",
    }
    for i, r in enumerate(records[:50]):
        assert set(r) == expected_fields, f"record {i}: schema drift"
        assert r["action_chunk"].shape[0] == 10, "RL window = 10 (contract)"
    # executed window rule: each transition carries the executed step count
    # of its window (rtc: up to the 10-step RL chunk; stop-and-go: ≤ n steps)
    executed = int(records[0]["executed_steps"])
    assert 1 <= executed <= 10
