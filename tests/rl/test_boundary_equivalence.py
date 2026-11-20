"""Stage-2 regression gate for the rtc transition stream (Phase 5b).

Evolved 2026-11-20 with the user's direction: the RL framework targets
PLUGGABLE post-training algorithms (rltoken is the first, reference
RL/RL-Token-Pi05-open); byte-preservation of the legacy implementation is
NOT a contract. Three layers:

1. Determinism: the seeded offline scenario (DummyRobot + DeterministicVLA
   + scripted stdin + --vla_only + fixed k) must produce a byte-identical
   journal across runs OF THE SAME implementation — async materialization
   interleaving included.
2. Structural equivalence vs the frozen golden: record count, episode/step
   ids, sources, executed steps, done/success/rewards, valid masks — the
   learning-signal contract — must match exactly.
3. Value baseline: tensor values compared against the frozen journal.
   Value drift with intact structure means the boundary-materialization
   timing shifted; intentional changes (algorithm/interface work) are
   accepted by re-freezing with FINEARM_GOLDEN_FREEZE=1 after review.

Plus the real-journal schema lock (v3_1 campaign journal, env-gated).
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
    run_a = run_scenario(name, tmp_path / "a")
    run_b = run_scenario(name, tmp_path / "b")

    # Layer 1: same-implementation determinism (async timing included).
    assert run_a["journal_bytes"] == run_b["journal_bytes"], (
        "scenario is nondeterministic across runs — async materialization "
        "interleaving must stay stable"
    )

    golden = golden_path(name)
    if os.environ.get("FINEARM_GOLDEN_FREEZE") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(run_a["journal_bytes"])
        digest = hashlib.sha256(run_a["journal_bytes"]).hexdigest()
        (golden.parent / "sha256").write_text(digest + "\n")
        pytest.skip(f"golden frozen ({len(run_a['records'])} records, sha256 {digest[:12]})")
    assert golden.is_file(), "golden missing — freeze with FINEARM_GOLDEN_FREEZE=1"

    want = _load_all(golden.read_bytes())
    got = run_a["records"]

    # Layer 2: structural contract — must hold across ANY refactor/algorithm
    # change unless the learning signal itself is being redesigned.
    assert len(want) == len(got), (
        f"record count changed: golden {len(want)} vs now {len(got)}"
    )
    meta_keys = ("episode_id", "step_id", "source", "source_chunk",
                 "executed_steps", "collection_phase_id", "done", "success",
                 "intervention_flag", "valid_mask", "rewards")
    for i, (w, g) in enumerate(zip(want, got)):
        assert set(w) == set(g), f"record {i}: field set changed"
        for key in meta_keys:
            assert np.array_equal(w[key], g[key]), (
                f"record {i}.{key}: structural contract violated"
            )

    # Layer 3: value baseline — drift here = materialization timing shift;
    # accept intentionally via FINEARM_GOLDEN_FREEZE=1 after review.
    if run_a["journal_bytes"] == golden.read_bytes():
        return
    value_keys = ("z_rl", "next_z_rl", "proprio", "next_proprio",
                  "ref_chunk", "next_ref_chunk", "action_chunk")
    drifted = {
        key: sum(1 for w, g in zip(want, got) if not np.array_equal(w[key], g[key]))
        for key in value_keys
    }
    pytest.fail(
        f"journal value drift (structure intact): { {k: v for k, v in drifted.items() if v} }. "
        "If this change is intentional, re-freeze with FINEARM_GOLDEN_FREEZE=1."
    )


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
