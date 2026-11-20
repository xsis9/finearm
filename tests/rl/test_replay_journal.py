"""Replay journal / stride-2 / transition-construction tests.

Ported from RL-Token-Pi05-open ``tests/test_rlt_core_journal.py`` and the
replay/stride-2 portions of ``tests/test_train_rlt_stage2_pi05.py`` (journal
restore semantics, disk-first writes, build_transition semantics, stride-2
window layout, per-step reference alignment, warmup journal schema), adapted
to 7-dim actions and Panthera joint names.
"""

import io
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from finearm.rl.replay import (
    COLLECTION_PHASE_WARMUP,
    PANTHERA_JOINT_NAMES,
    BoundaryContext,
    DeferredBoundarySnapshot,
    DeferredExecutionWindow,
    ExecutionWindow,
    Pi05ActionCache,
    ReplayBuffer,
    ReplayManager,
    StepRecord,
    TransitionSource,
    action_order_indices,
    build_stride2_transitions,
    build_transition,
    materialize_deferred_windows,
    make_vla_reference_chunk,
    order_episode_transitions,
    reorder_action_tensor,
    subsample_vla_actions,
    write_warmup_interval_records,
)

HORIZON = 3
ACTION_DIM = 2  # tiny dims for transition-machinery tests (7-dim shape tests live in test_heads_contract)


def _boundary(step, horizon=HORIZON, action_dim=ACTION_DIM):
    return BoundaryContext(
        z_rl=torch.tensor([[1.0, 2.0]]),
        proprio=torch.tensor([0.0, 0.0]),
        ref_chunk=torch.randn(1, horizon, action_dim),
        collection_phase="online",
        episode_id=1,
        step_id=step,
    )


def _transition():
    window = ExecutionWindow(_boundary(step=1))
    window.append(torch.tensor([0.1, 0.2]), int(TransitionSource.BASE))
    return build_transition(
        window, _boundary(step=2), horizon=HORIZON, action_dim=ACTION_DIM,
    )


# ── ReplayManager journal semantics ──────────────────────────────────────────


def test_replay_manager_does_not_restore_unless_explicit():
    transition = _transition()
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        with open(journal, "wb") as stream:
            pickle.dump(transition.to_numpy(), stream)
        buffer = ReplayBuffer(8)
        ReplayManager(buffer, str(journal))
        assert len(buffer) == 0


def test_replay_manager_restores_entries_when_explicit():
    transition = _transition()
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        with open(journal, "wb") as stream:
            pickle.dump(transition.to_numpy(), stream)
        buffer = ReplayBuffer(8)
        manager = ReplayManager(buffer, str(journal), restore=True)
        assert len(buffer) == 1
        assert manager.stats()["adds_total"] == 1


def test_replay_manager_restore_empty_journal_is_benign():
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        journal.write_bytes(b"")
        buffer = ReplayBuffer(8)
        manager = ReplayManager(buffer, str(journal), restore=True)
        assert len(buffer) == 0
        assert manager.stats()["adds_total"] == 0


def test_replay_manager_restore_truncated_tail_keeps_prior_entries():
    transition = _transition()
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        with open(journal, "wb") as stream:
            pickle.dump(transition.to_numpy(), stream)
            pickle.dump(transition.to_numpy(), stream)
        data = journal.read_bytes()
        journal.write_bytes(data[:len(data) - 17])  # cut inside the 2nd record
        buffer = ReplayBuffer(8)
        manager = ReplayManager(buffer, str(journal), restore=True)
        assert len(buffer) == 1
        assert manager.stats()["adds_total"] == 1


def test_replay_manager_restore_legacy_schema_fails_loudly():
    # A first-record failure (legacy schema) must propagate, not be
    # swallowed into a misleading "restored count" warning.
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        with open(journal, "wb") as stream:
            pickle.dump({"legacy": True}, stream)
        buffer = ReplayBuffer(8)
        with pytest.raises(ValueError):
            ReplayManager(buffer, str(journal), restore=True)
        assert len(buffer) == 0


def test_replay_add_is_disk_first():
    events = []
    transition = _transition()
    with tempfile.TemporaryDirectory() as root:
        journal = Path(root) / "replay.pkl"
        buffer = ReplayBuffer(8)
        original_add = buffer.add

        def checked_add(value):
            events.append(journal.exists() and journal.stat().st_size > 0)
            return original_add(value)

        buffer.add = checked_add
        manager = ReplayManager(buffer, str(journal))
        manager.add_transition(transition)
        assert events == [True]


def test_buffer_sample_batch_shapes_and_dtypes():
    buffer = ReplayBuffer(64, seed=0)
    for step in range(6):
        window = ExecutionWindow(_boundary(step=step))
        for k in range(HORIZON):
            window.append(torch.full((ACTION_DIM,), 0.1 * (step + k)), int(TransitionSource.BASE))
        buffer.add(build_transition(window, _boundary(step=step + 1), horizon=HORIZON, action_dim=ACTION_DIM))
    batch = buffer.sample(4)
    assert set(batch) == set(_transition().to_numpy())
    assert tuple(batch["z_rl"].shape) == (4, 2)
    assert tuple(batch["ref_chunk"].shape) == (4, HORIZON, ACTION_DIM)
    assert tuple(batch["done"].shape) == (4, 1)          # done gets a trailing dim
    assert tuple(batch["valid_mask"].shape) == (4, HORIZON)
    assert batch["z_rl"].dtype == torch.float32          # fp16 storage upcast on sample


# ── build_transition semantics ───────────────────────────────────────────────


def test_execution_window_materializes_partial_valid_mask_and_actual_human_actions():
    boundary = _boundary(step=3)
    window = ExecutionWindow(boundary)
    window.append(torch.tensor([0.5, 0.6]), int(TransitionSource.HUMAN), True)
    window.append(torch.tensor([0.7, 0.8]), int(TransitionSource.HUMAN), True)
    transition = build_transition(
        window, _boundary(step=4), horizon=HORIZON, action_dim=ACTION_DIM,
    )
    assert transition.executed_steps == 2
    assert transition.valid_mask.tolist() == [True, True, False]
    assert not transition.done
    assert transition.rewards.tolist() == [0.0, 0.0, 0.0]
    # Algorithm 1: intervention actions replace the VLA reference in replay.
    torch.testing.assert_close(
        torch.from_numpy(transition.action_chunk[:2]),
        torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    )
    torch.testing.assert_close(
        torch.from_numpy(transition.ref_chunk[:2]),
        torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    )


def test_reward_and_done_only_apply_at_real_terminal():
    window = ExecutionWindow(_boundary(step=1))
    window.append(torch.tensor([0.1, 0.2]), int(TransitionSource.BASE))
    transition = build_transition(
        window, _boundary(step=2), horizon=HORIZON, action_dim=ACTION_DIM,
        reward=1.0, terminal=True,
    )
    assert transition.done
    assert transition.rewards.tolist() == [1.0, 0.0, 0.0]


def test_deferred_windows_keep_partial_policy_and_separate_human_chunks():
    policy_start = _boundary(step=1)
    takeover = DeferredBoundarySnapshot({}, torch.zeros(2), 2)
    human_end = DeferredBoundarySnapshot({}, torch.ones(2), 3)
    calls = []

    def materialize(snapshot):
        calls.append(snapshot.step_id)
        return _boundary(step=snapshot.step_id)

    deferred = [
        DeferredExecutionWindow(
            policy_start, takeover,
            [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
            [int(TransitionSource.BASE)] * 2,
            [True, True],
        ),
        DeferredExecutionWindow(
            takeover, human_end,
            [torch.tensor([0.5, 0.6])],
            [int(TransitionSource.HUMAN)],
            [True],
        ),
    ]
    transitions = materialize_deferred_windows(
        deferred, materialize, horizon=HORIZON, action_dim=ACTION_DIM,
    )
    assert calls == [2, 3]
    assert [t.executed_steps for t in transitions] == [2, 1]
    assert transitions[0].valid_mask.tolist() == [True, True, False]
    assert transitions[1].valid_mask.tolist() == [True, False, False]
    assert transitions[0].source_chunk.tolist() == [0, 0, 0]
    assert transitions[1].source_chunk.tolist() == [2, 0, 0]
    assert transitions[1].step_id == 2


def test_deferred_transitions_are_restored_to_temporal_order():
    later = _transition()
    later.step_id = 8
    earlier = _transition()
    earlier.step_id = 3
    ordered = order_episode_transitions([later, earlier])
    assert [item.step_id for item in ordered] == [3, 8]


# ── stride-2 window construction ─────────────────────────────────────────────


def _make_records(n, action_dim, human_index=None):
    records = []
    for i in range(n):
        is_human = human_index is not None and i in human_index
        records.append(StepRecord(
            action=torch.arange(action_dim, dtype=torch.float32) + float(i),
            source=(
                int(TransitionSource.HUMAN) if is_human
                else int(TransitionSource.BASE)
            ),
            intervention=is_human,
            step_id=100 + i,
            ref_row=None if is_human else (
                torch.arange(action_dim, dtype=torch.float32) + 100.0 + float(i)
            ),
        ))
    return records


def _make_snapshots(n, stride, state_dim=6):
    snapshots = {}
    for step in range(0, n + 1, stride):
        snapshots[step] = DeferredBoundarySnapshot(
            observation={}, state=torch.zeros(state_dim), step_id=100 + step,
        )
    return snapshots


def _materialize(snapshot, horizon, action_dim):
    # Deliberately stale zero reference: per-step ref_row alignment must
    # overwrite it for non-HUMAN rows.
    return BoundaryContext(
        z_rl=torch.zeros(1, 2),
        proprio=torch.zeros(6),
        ref_chunk=torch.zeros(1, horizon, action_dim),
        collection_phase="online",
        episode_id=1,
        step_id=snapshot.step_id,
    )


def test_build_stride2_transitions_emits_overlapping_windows():
    horizon, action_dim, stride = 4, 2, 2
    records = _make_records(8, action_dim)
    snapshots = _make_snapshots(8, stride)  # boundary snapshots at steps 0,2,4,6,8
    terminal = DeferredBoundarySnapshot({}, torch.zeros(6), 100 + 8)

    transitions = build_stride2_transitions(
        records, snapshots, terminal,
        horizon=horizon, action_dim=action_dim, stride=stride,
        materialize_snapshot=lambda s: _materialize(s, horizon, action_dim),
    )
    assert [t.step_id for t in transitions] == [100, 102, 104]
    assert [t.executed_steps for t in transitions] == [4, 4, 4]


class _StrideBuilder:
    """Mirror of the reference Stride2ShortEpisodeTest helpers."""

    def _build(self, n_records, horizon, action_dim, stride, human_index=None):
        records = _make_records(n_records, action_dim, human_index=human_index)
        snapshots = _make_snapshots(n_records, stride)
        terminal = DeferredBoundarySnapshot({}, torch.zeros(6), 100 + n_records)
        return build_stride2_transitions(
            records, snapshots, terminal,
            horizon=horizon, action_dim=action_dim, stride=stride,
            materialize_snapshot=lambda s: _materialize(s, horizon, action_dim),
        )


def test_short_episode_keeps_every_stride_start():
    transitions = _StrideBuilder()._build(n_records=9, horizon=10, action_dim=6, stride=2)
    assert [t.step_id - 100 for t in transitions] == [0, 2, 4, 6, 8]
    assert [t.executed_steps for t in transitions] == [9, 7, 5, 3, 1]
    assert [t.done for t in transitions] == [True, True, True, True, True]


def test_exact_horizon_episode_emits_one_complete_window():
    transitions = _StrideBuilder()._build(n_records=10, horizon=10, action_dim=6, stride=2)
    assert [t.step_id - 100 for t in transitions] == [0]
    assert [t.executed_steps for t in transitions] == [10]
    assert transitions[0].done


def test_longer_episode_keeps_complete_and_one_terminal_tail():
    transitions = _StrideBuilder()._build(n_records=13, horizon=10, action_dim=6, stride=2)
    assert [t.step_id - 100 for t in transitions] == [0, 2, 12]
    assert [t.executed_steps for t in transitions] == [10, 10, 1]
    assert [t.done for t in transitions] == [False, False, True]


def test_every_row_uses_its_own_step_reference():
    # 12 steps, horizon 10, stride 2 -> windows [0:10] and [2:12]; the
    # latter crosses the re-plan boundary at step 10 (rows 8..9).
    transitions = _StrideBuilder()._build(n_records=12, horizon=10, action_dim=2, stride=2)
    assert [t.step_id - 100 for t in transitions] == [0, 2]
    for transition, start in ((transitions[0], 0), (transitions[1], 2)):
        for row in range(10):
            expected = torch.tensor([100.0 + start + row, 101.0 + start + row])
            assert torch.allclose(torch.as_tensor(transition.ref_chunk[row]), expected), (
                f"window start {start} row {row} reference mismatch"
            )


def test_human_rows_keep_executed_action_replacement():
    # Human step at index 2: its reference must stay the executed action
    # (Algorithm 1 intervention replacement), not the per-step ref_row.
    transitions = _StrideBuilder()._build(
        n_records=6, horizon=4, action_dim=2, stride=2, human_index={2},
    )
    window = transitions[0]  # [0:4]
    assert window.source_chunk[2] == int(TransitionSource.HUMAN)
    torch.testing.assert_close(
        torch.as_tensor(window.ref_chunk[2]),
        torch.as_tensor(window.action_chunk[2]),
    )
    # Non-human rows still use their own step references.
    for row in (0, 1, 3):
        expected = torch.tensor([100.0 + row, 101.0 + row])
        assert torch.allclose(torch.as_tensor(window.ref_chunk[row]), expected)


# ── warmup journal writer schema (7-dim) ─────────────────────────────────────

RECORD_FIELDS = [
    "z_rl", "proprio", "ref_chunk", "action_chunk", "rewards", "done",
    "next_z_rl", "next_proprio", "next_ref_chunk", "source", "source_chunk",
    "valid_mask", "executed_steps", "collection_phase_id", "success",
    "intervention_flag", "episode_id", "step_id",
]


def _make_warmup_fixture():
    """Episode 0, annotated frames 10..24 (15 frames), window 10 / stride 2, 7-dim."""
    frame_data = {}
    for frame in range(10, 25):
        frame_data[(0, frame)] = (
            np.full(4, float(frame), dtype=np.float16),          # z_rl
            np.arange(7, dtype=np.float32) + frame,              # proprio
            np.full(7, 0.5 * frame, dtype=np.float32),           # action
        )
    return frame_data


def test_writes_stride2_windows_with_full_schema_7dim():
    frame_data = _make_warmup_fixture()
    out = io.BytesIO()
    n = write_warmup_interval_records(
        out, ep=0, frames_in_interval=list(range(10, 25)),
        frame_data=frame_data, horizon=10, stride=2, action_dim=7,
    )
    # 15 frames -> windows at pos 0,2,...,14 (8 windows)
    assert n == 8

    out.seek(0)
    records = []
    while True:
        try:
            records.append(pickle.load(out))
        except EOFError:
            break
    assert len(records) == 8

    for rec in records:
        assert list(rec) == RECORD_FIELDS, list(rec)
        assert rec["z_rl"].dtype == np.float16 and rec["z_rl"].shape == (4,)
        assert rec["proprio"].dtype == np.float32 and rec["proprio"].shape == (7,)
        for key in ("ref_chunk", "action_chunk", "next_ref_chunk"):
            assert rec[key].dtype == np.float16 and rec[key].shape == (10, 7)
        assert rec["rewards"].dtype == np.float32 and rec["rewards"].shape == (10,)
        assert rec["done"] == np.asarray(False, dtype=np.bool_)
        assert rec["source"] == np.asarray(TransitionSource.BASE, dtype=np.uint8)
        assert rec["source_chunk"].dtype == np.uint8 and rec["source_chunk"].shape == (10,)
        assert rec["collection_phase_id"] == np.asarray(COLLECTION_PHASE_WARMUP, dtype=np.uint8)
        assert rec["success"] == np.asarray(0, dtype=np.int8)
        assert rec["intervention_flag"] == np.asarray(False, dtype=np.bool_)
        assert rec["episode_id"] == np.asarray(0, dtype=np.int32)


def test_warmup_window_content_semantics_7dim():
    frame_data = _make_warmup_fixture()
    out = io.BytesIO()
    write_warmup_interval_records(
        out, ep=0, frames_in_interval=list(range(10, 25)),
        frame_data=frame_data, horizon=10, stride=2, action_dim=7,
    )
    out.seek(0)
    records = [pickle.load(out) for _ in range(2)]  # pos 0 (frames 10-19) & pos 2 (12-21)
    r0, r1 = records
    assert r0["executed_steps"] == np.asarray(10, dtype=np.int16)
    assert r0["valid_mask"].all()
    assert r0["step_id"] == np.asarray(10, dtype=np.int32)
    # first window: next_ref_chunk = frames 20..24 (actions 0.5*frame) then zero padding
    expected_next = np.full((5, 7), 0.5 * np.arange(20, 25, dtype=np.float32)[:, None],
                            dtype=np.float16)
    assert np.allclose(r0["next_ref_chunk"][:5], expected_next)
    assert (r0["next_ref_chunk"][5:] == 0).all()
    # z_rl of window start == z of frame 10; next_z_rl == z of the frame right
    # after the window (frame 20 for window 0), clamped to the interval tail
    assert r0["z_rl"][0] == np.float16(10.0)
    assert r0["next_z_rl"][0] == np.float16(20.0)
    # second window starts at frame 12; its next window (22..24) has 3 real frames
    assert r1["step_id"] == np.asarray(12, dtype=np.int32)
    assert (r1["next_ref_chunk"][:3] != 0).all() and (r1["next_ref_chunk"][3:] == 0).all()


# ── VLA action cache & reference chunk helpers ───────────────────────────────


def test_action_cache_takes_a_full_execution_horizon():
    cache = Pi05ActionCache()
    actions = torch.tensor([[[0.1, 9.0], [0.2, 8.0], [0.3, 7.0], [0.4, 6.0]]])
    embedding = torch.tensor([[4.0]])
    cache.refresh(actions, embedding, action_dim=1)

    result, start_index, result_embedding = cache.take(2)

    torch.testing.assert_close(result, torch.tensor([[[0.1], [0.2]]]))
    assert start_index == 0
    assert result_embedding is embedding
    torch.testing.assert_close(cache.remaining_raw_actions(), torch.tensor([[[0.3, 7.0], [0.4, 6.0]]]))
    assert cache.exhausted() is False


def test_action_cache_rejects_partial_execution_horizon():
    cache = Pi05ActionCache()
    cache.refresh(torch.zeros(1, 3, 1), torch.zeros(1, 1), action_dim=1)
    with pytest.raises(ValueError):
        cache.take(4)


def test_reference_chunk_indexing_stays_in_model_space():
    actions = torch.tensor([[[0.25, -0.75], [0.50, -0.50], [0.75, -0.25]]])
    result = subsample_vla_actions(actions, action_dim=2, stride=2, num_steps=2)
    torch.testing.assert_close(result, torch.tensor([[[0.25, -0.75], [0.75, -0.25]]]))

    actions = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]])
    result = make_vla_reference_chunk(
        actions, action_dim=1, start_index=1, stride=2, num_steps=3,
    )
    torch.testing.assert_close(result, torch.tensor([[[1.0], [3.0], [5.0]]]))


# ── Panthera joint-name ordering (fail-closed reorder) ───────────────────────


def test_identity_order_is_preserved():
    indices = action_order_indices(
        [f"{name}.pos" for name in PANTHERA_JOINT_NAMES],
        list(PANTHERA_JOINT_NAMES),
    )
    assert indices == tuple(range(7))
    action = torch.randn(2, 10, 7)
    torch.testing.assert_close(
        reorder_action_tensor(
            action,
            source_names=[f"{name}.pos" for name in PANTHERA_JOINT_NAMES],
            target_names=list(PANTHERA_JOINT_NAMES),
        ),
        action,
    )


def test_policy_action_is_reordered_to_robot_feature_order():
    shuffled = [
        "target_gripper.pos", "target_joint_3", "action.target_joint_1.pos",
        "target_joint_2.pos", "target_joint_5.pos", "target_joint_4",
        "target_joint_6.pos",
    ]
    indices = action_order_indices(shuffled, list(PANTHERA_JOINT_NAMES))
    assert indices == (2, 3, 1, 5, 4, 6, 0)
    action = torch.arange(14, dtype=torch.float32).reshape(1, 2, 7)
    reordered = reorder_action_tensor(action, source_names=shuffled, target_names=list(PANTHERA_JOINT_NAMES))
    torch.testing.assert_close(reordered[0, 0], torch.tensor([2.0, 3.0, 1.0, 5.0, 4.0, 6.0, 0.0]))


@pytest.mark.parametrize("bad", [
    ["target_joint_1"] * 7,                                   # duplicates
    [f"target_joint_{i}" for i in range(1, 8)] + ["extra"],   # wrong length handled by caller
    [f"target_joint_{i}" for i in range(2, 8)] + ["bogus"],   # unknown joint
])
def test_invalid_robot_order_is_rejected(bad):
    with pytest.raises(ValueError):
        action_order_indices(list(PANTHERA_JOINT_NAMES), bad)
