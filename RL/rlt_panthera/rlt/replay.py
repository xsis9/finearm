"""Replay machinery for RL-Token Stage 2 (Panthera-HT port).

Ported from RL-Token-Pi05-open ``scripts/rlt_core.py`` and the runtime core
of ``scripts/train_rlt_stage2_pi05.py``; the algorithm is byte-equivalent,
with two Panthera changes:

- ``PANTHERA_JOINT_NAMES`` replaces ``SO101_JOINT_NAMES`` (7 joints:
  ``target_joint_1..6`` + ``target_gripper`` — the canonical order of the
  ``panthera_bolt_p3`` dataset ``action`` feature and of the frozen p3
  checkpoint's served absolute joint targets).
- All shapes are parameterized by ``action_dim`` (7) / ``state_dim`` (7);
  nothing here hard-codes 6.

Contents:
- TransitionSource / Transition / ReplayBuffer / ReplayManager   journal+fsync+schema
- ExecutionWindow / BoundaryContext / StepRecord / ...           boundary snapshots
- build_transition / build_stride2_transitions / ...             stride-2 replay construction
- write_warmup_interval_records                                  warmup journal writer
- Pi05ActionCache / make_pi05_reference_chunk                    VLA chunk helpers
- RerunVisualizationManager / TakeoverState / clone_observation  Stage-2 runtime support
"""

from __future__ import annotations

import enum
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Panthera-HT action order used by the p3 dataset and the served checkpoints.
# Keep this explicit because LeRobot feature dictionaries are not an ordering
# contract: a robot/dataset can legally expose the same names in another order.
PANTHERA_JOINT_NAMES = (
    "target_joint_1",
    "target_joint_2",
    "target_joint_3",
    "target_joint_4",
    "target_joint_5",
    "target_joint_6",
    "target_gripper",
)


def _canonical_joint_name(name: str) -> str:
    """Normalize LeRobot action feature names to bare Panthera joint names."""
    value = str(name).strip()
    if value.startswith("action."):
        value = value[len("action."):]
    if value.endswith(".pos"):
        value = value[:-4]
    return value


def action_order_indices(
    source_names,
    target_names,
    allowed_joints: tuple[str, ...] = PANTHERA_JOINT_NAMES,
) -> tuple[int, ...]:
    """Return indices that reorder policy outputs into robot feature order.

    ``source_names`` describes the order used by the trained policy, while
    ``target_names`` is the order expected by the robot/dataset boundary.
    Duplicate, missing, or unknown joints fail closed before any hardware move.
    """
    source = tuple(_canonical_joint_name(name) for name in source_names)
    target = tuple(_canonical_joint_name(name) for name in target_names)
    allowed = set(allowed_joints)
    if len(source) != len(target):
        raise ValueError(f"Action order length mismatch: source={len(source)} target={len(target)}")
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError(f"Action order contains duplicate joints: source={source}, target={target}")
    if set(source) != allowed:
        raise ValueError(
            f"Policy action order must contain Panthera joints {allowed_joints}, got {source}"
        )
    if set(target) != allowed:
        raise ValueError(
            f"Robot action order must contain Panthera joints {allowed_joints}, got {target}"
        )
    positions = {name: index for index, name in enumerate(source)}
    return tuple(positions[name] for name in target)


def reorder_action_tensor(
    action: torch.Tensor,
    *,
    source_names,
    target_names,
    allowed_joints: tuple[str, ...] = PANTHERA_JOINT_NAMES,
) -> torch.Tensor:
    """Reorder the final action dimension without changing batch/chunk axes."""
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"action must be a torch.Tensor, got {type(action).__name__}")
    indices = action_order_indices(source_names, target_names, allowed_joints)
    if action.shape[-1] != len(indices):
        raise ValueError(
            f"Action tensor last dimension {action.shape[-1]} does not match action order {len(indices)}"
        )
    index = torch.tensor(indices, dtype=torch.long, device=action.device)
    return action.index_select(-1, index)


# ── Replay collection-phase ids (shared by Transition machinery) ──────────
COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


class RerunVisualizationManager:
    """Best-effort, non-blocking Rerun logging for live robot data."""

    def __init__(self, log_data, compress_images: bool, fps: float):
        self._log_data = log_data
        self._compress_images = compress_images
        self._period_s = 1.0 / fps
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._action: dict | None = None
        self._running = False
        self._disabled = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_values(values: dict) -> dict:
        copied = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().numpy().copy()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                copied[key] = value
        return copied

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rerun-visualization", daemon=True)
        self._thread.start()

    def publish_observation(self, observation: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._observation = self._copy_values(observation)
            self._condition.notify()

    def publish_action(self, action: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._action = self._copy_values(action)
            self._condition.notify()

    def _run(self) -> None:
        next_log_time = 0.0
        while self._running:
            with self._condition:
                while self._running and self._observation is None and self._action is None:
                    self._condition.wait(timeout=0.1)
                if not self._running:
                    return
                wait_s = next_log_time - time.monotonic()
                if wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue
                observation, action = self._observation, self._action
                self._observation, self._action = None, None
            try:
                self._log_data(
                    observation=observation,
                    action=action,
                    compress_images=self._compress_images,
                )
                next_log_time = time.monotonic() + self._period_s
            except Exception as exc:
                self._disabled = True
                logger.warning("Rerun visualization disabled after logging failure: %s", exc)
                return

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.5)


class TransitionSource(enum.IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def collection_phase_to_id(phase: str) -> int:
    name = str(phase).split(":", 1)[0].lower()
    if name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def _ensure_array(value, *, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass
class Transition:
    z_rl: np.ndarray           # (rlt_hidden_dim,)
    proprio: np.ndarray        # (state_dim,)
    ref_chunk: np.ndarray      # (n_action_steps_rl, action_dim)
    action_chunk: np.ndarray   # (n_action_steps_rl, action_dim)
    rewards: np.ndarray        # (n_action_steps_rl,)
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int
    source_chunk: np.ndarray   # uint8 (n_action_steps_rl,)
    valid_mask: np.ndarray     # bool (n_action_steps_rl,)
    executed_steps: int
    collection_phase: str
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "z_rl": _ensure_array(self.z_rl, dtype=np.float16),
            "proprio": _ensure_array(self.proprio, dtype=np.float32),
            "ref_chunk": _ensure_array(self.ref_chunk, dtype=np.float16),
            "action_chunk": _ensure_array(self.action_chunk, dtype=np.float16),
            "rewards": _ensure_array(self.rewards, dtype=np.float32),
            "done": _ensure_array(self.done, dtype=np.bool_),
            "next_z_rl": _ensure_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _ensure_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _ensure_array(self.next_ref_chunk, dtype=np.float16),
            "source": _ensure_array(self.source, dtype=np.uint8),
            "source_chunk": _ensure_array(self.source_chunk, dtype=np.uint8),
            "valid_mask": _ensure_array(self.valid_mask, dtype=np.bool_),
            "executed_steps": _ensure_array(self.executed_steps, dtype=np.int16),
            "collection_phase_id": _ensure_array(
                collection_phase_to_id(self.collection_phase), dtype=np.uint8,
            ),
            "success": _ensure_array(self.success, dtype=np.int8),
            "intervention_flag": _ensure_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _ensure_array(self.episode_id, dtype=np.int32),
            "step_id": _ensure_array(self.step_id, dtype=np.int32),
        }


@dataclass
class Pi05ActionCache:
    """Full VLA raw chunks plus the model-space actions used by RLT.

    ``raw_actions`` is the (1, chunk_size, action_dim) absolute-target chunk
    served by serve_rlt (chunk_size=30 for the p3 base); ``take(n)`` hands out
    contiguous execution horizons (n=10 per RL window, 3 windows per VLA call).
    """

    raw_actions: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    z_rl: torch.Tensor | None = None
    next_index: int = 0

    def exhausted(self) -> bool:
        return self.raw_actions is None or self.next_index >= self.raw_actions.shape[1]

    def remaining_raw_actions(self) -> torch.Tensor | None:
        if self.exhausted() or self.raw_actions is None:
            return None
        return self.raw_actions[:, self.next_index:].clone()

    def refresh(self, raw_actions: torch.Tensor, z_rl: torch.Tensor, action_dim: int) -> None:
        self.raw_actions = raw_actions
        self.actions = raw_actions[:, :, :action_dim]
        self.z_rl = z_rl
        self.next_index = 0

    def clear(self) -> None:
        self.raw_actions = None
        self.actions = None
        self.z_rl = None
        self.next_index = 0

    def take(self, count: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if self.exhausted() or self.actions is None or self.z_rl is None:
            raise RuntimeError("VLA action cache is empty.")
        if count < 1 or self.next_index + count > self.actions.shape[1]:
            raise ValueError("Requested VLA action horizon is not available in the cache.")
        action_index = self.next_index
        result = self.actions[:, action_index : action_index + count]
        self.next_index += count
        return result, action_index, self.z_rl


def make_vla_reference_chunk(
    actions: torch.Tensor,
    *,
    action_dim: int,
    start_index: int,
    num_steps: int,
    stride: int = 1,
) -> torch.Tensor:
    """Build a normalized RLT reference horizon from a raw VLA action chunk.

    Paper Eq. (5) / Algorithm 1 define the reference as ``a~1:C`` — the first C
    contiguous VLA actions. Stride-2 subsampling applies only to replay
    transition collection, never to the reference chunk, so ``stride`` defaults
    to 1 (contiguous) and is kept only for tests of the stride helper.

    (Named ``make_vla_reference_chunk`` in this port; the reference repo calls
    it ``make_pi05_reference_chunk`` — same function.)
    """
    indices = start_index + torch.arange(num_steps, device=actions.device) * stride
    indices = indices.clamp_max(actions.shape[1] - 1)
    return actions.index_select(1, indices)[:, :, :action_dim]


def subsample_vla_actions(
    actions: torch.Tensor,
    *,
    action_dim: int,
    stride: int,
    num_steps: int,
) -> torch.Tensor:
    """Build an RLT reference horizon starting at the first VLA action."""
    return make_vla_reference_chunk(
        actions,
        action_dim=action_dim,
        start_index=0,
        stride=stride,
        num_steps=num_steps,
    )


# Compatibility aliases (reference-repo names).
make_pi05_reference_chunk = make_vla_reference_chunk
subsample_pi05_actions = subsample_vla_actions


@dataclass
class BoundaryContext:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor
    collection_phase: str
    episode_id: int
    step_id: int


@dataclass
class ExecutionWindow:
    """Actions actually accepted by hardware under one observation boundary."""

    boundary: BoundaryContext
    actions: list[torch.Tensor] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)

    def append(self, actual_action: torch.Tensor, source: int, intervention: bool = False) -> None:
        if actual_action.ndim != 1:
            raise ValueError(f"Executed action must be rank 1, got {tuple(actual_action.shape)}")
        self.actions.append(actual_action.detach().clone())
        self.sources.append(int(source))
        self.intervention_flags.append(bool(intervention))

    @property
    def executed_steps(self) -> int:
        return len(self.actions)


def _resolve_chunk_source(sources: list[int], intervention_flags: list[bool]) -> tuple[int, bool]:
    intervention = any(intervention_flags)
    source_set = set(sources)
    has_human = int(TransitionSource.HUMAN) in source_set
    has_policy = any(s in source_set for s in (
        int(TransitionSource.BASE), int(TransitionSource.RL), int(TransitionSource.MIXED),
    ))
    if int(TransitionSource.MIXED) in source_set or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(sources[0]), intervention


@dataclass
class DeferredBoundarySnapshot:
    """Observation captured at takeover; expensive boundary inference is deferred."""

    observation: dict
    state: torch.Tensor
    step_id: int
    raw_actions: torch.Tensor | None = None
    raw_action_index: int = 0


@dataclass
class TakeoverState:
    """Small coordinator for lossless takeover requests and leader sample ordering."""

    pending: bool = False
    last_leader_sequence: int = -1

    def latch(self, requested: bool = True) -> None:
        self.pending = self.pending or bool(requested)

    def handoff_finished(self, succeeded: bool) -> None:
        if succeeded:
            self.pending = False

    def accept_leader_sequence(self, sequence: int) -> bool:
        if sequence <= self.last_leader_sequence:
            return False
        self.last_leader_sequence = sequence
        return True


@dataclass
class DeferredExecutionWindow:
    """Executed actions whose start/end contexts will be inferred after HIL."""

    start: BoundaryContext | DeferredBoundarySnapshot
    end: BoundaryContext | DeferredBoundarySnapshot
    actions: list[torch.Tensor]
    sources: list[int]
    intervention_flags: list[bool]


@dataclass
class StepRecord:
    """One executed control step inside the critical-phase replay segment."""

    action: torch.Tensor   # normalized actual action (action_dim,)
    source: int            # TransitionSource value
    intervention: bool
    step_id: int           # absolute episode control-step id
    ref_row: torch.Tensor | None = None  # VLA reference row for this step (normalized); None for HUMAN steps


def clone_observation(obs: dict) -> dict:
    """Detach camera/state payloads so a deferred boundary cannot mutate underneath us."""
    import copy as _copy

    result = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            result[key] = value.copy()
        else:
            result[key] = _copy.deepcopy(value)
    return result


def materialize_deferred_windows(
    deferred: list[DeferredExecutionWindow],
    materialize_boundary,
    *,
    horizon: int,
    action_dim: int,
) -> list[Transition]:
    """Infer each captured boundary once and build one transition per real window."""
    cache: dict[int, BoundaryContext] = {}

    def resolve(value: BoundaryContext | DeferredBoundarySnapshot) -> BoundaryContext:
        if isinstance(value, BoundaryContext):
            return value
        key = id(value)
        if key not in cache:
            cache[key] = materialize_boundary(value)
        return cache[key]

    transitions = []
    for item in deferred:
        window = ExecutionWindow(resolve(item.start))
        for action, source, intervention in zip(
            item.actions, item.sources, item.intervention_flags, strict=True,
        ):
            window.append(action, source, intervention)
        transitions.append(build_transition(
            window, resolve(item.end), horizon=horizon, action_dim=action_dim,
        ))
    return transitions


def build_transition(
    window: ExecutionWindow,
    next_boundary: BoundaryContext,
    *,
    horizon: int,
    action_dim: int,
    reward: float = 0.0,
    terminal: bool = False,
) -> Transition:
    """Materialize a replay row; only real terminal boundaries receive reward/done."""
    if window.executed_steps < 1 or window.executed_steps > horizon:
        raise ValueError(f"Execution window has {window.executed_steps} actions; expected 1..{horizon}")
    if len(window.sources) != window.executed_steps:
        raise ValueError("Execution window source count does not match action count")
    actions = torch.zeros(horizon, action_dim, dtype=torch.float32)
    stacked = torch.stack(window.actions, dim=0).cpu()
    if stacked.shape[1] != action_dim:
        raise ValueError(f"Executed action dim {stacked.shape[1]} != expected {action_dim}")
    actions[:window.executed_steps] = stacked
    refs = window.boundary.ref_chunk.detach().cpu().squeeze(0).float().clone()
    if refs.shape != (horizon, action_dim):
        raise ValueError(f"Reference chunk shape {tuple(refs.shape)} != {(horizon, action_dim)}")
    source_chunk = np.full(horizon, int(TransitionSource.BASE), dtype=np.uint8)
    source_chunk[:window.executed_steps] = np.asarray(window.sources, dtype=np.uint8)
    # Algorithm 1: an intervention replaces the VLA reference in replay.
    for index, source in enumerate(window.sources):
        if source == int(TransitionSource.HUMAN):
            refs[index] = actions[index]
    valid_mask = np.zeros(horizon, dtype=np.bool_)
    valid_mask[:window.executed_steps] = True
    rewards = np.zeros(horizon, dtype=np.float32)
    if terminal:
        rewards[window.executed_steps - 1] = float(reward)
    source, intervention = _resolve_chunk_source(window.sources, window.intervention_flags)
    return Transition(
        z_rl=window.boundary.z_rl.detach().cpu().numpy().squeeze(),
        proprio=window.boundary.proprio.detach().cpu().numpy(),
        ref_chunk=refs.numpy(),
        action_chunk=actions.numpy(),
        rewards=rewards,
        done=bool(terminal),
        next_z_rl=next_boundary.z_rl.detach().cpu().numpy().squeeze(),
        next_proprio=next_boundary.proprio.detach().cpu().numpy(),
        next_ref_chunk=next_boundary.ref_chunk.detach().cpu().numpy().squeeze(),
        source=source,
        source_chunk=source_chunk,
        valid_mask=valid_mask,
        executed_steps=window.executed_steps,
        collection_phase=window.boundary.collection_phase,
        intervention_flag=intervention,
        episode_id=window.boundary.episode_id,
        step_id=window.boundary.step_id,
    )


def order_episode_transitions(transitions: list[Transition]) -> list[Transition]:
    """Restore temporal order after deferred HIL boundary materialization."""
    return sorted(transitions, key=lambda item: (item.episode_id, item.step_id))


def build_stride2_transitions(
    step_records: list[StepRecord],
    stride_snapshots: dict[int, DeferredBoundarySnapshot],
    terminal_snapshot: DeferredBoundarySnapshot | None,
    *,
    horizon: int,
    action_dim: int,
    stride: int,
    materialize_snapshot,
) -> list[Transition]:
    """Build stride-2 overlapping replay transitions from a continuous step log.

    This follows the paper's subsampling rule: for stride=2 and chunk length C,
    save transitions <x0, a0:C>, <x2, a2:C+2>, <x4, a4:C+4>, ... The start/end
    observations are taken from ``stride_snapshots``; intermediate actions come
    from ``step_records``. Complete windows are always emitted. If the episode
    ends before a full window, the last partial tail is emitted as a padded
    terminal window. When the critical phase itself is shorter than one horizon
    (fewer executed steps than ``horizon``), every stride start is padded with
    the terminal observation so no executed control step is dropped from replay.
    """
    if stride < 1 or horizon < 1:
        return []
    records = sorted(step_records, key=lambda item: item.step_id)
    if not records:
        return []

    boundary_cache: dict[int, BoundaryContext] = {}

    def resolve(snapshot: DeferredBoundarySnapshot) -> BoundaryContext:
        key = id(snapshot)
        if key not in boundary_cache:
            boundary_cache[key] = materialize_snapshot(snapshot)
        return boundary_cache[key]

    transitions: list[Transition] = []
    short_episode = len(records) < horizon
    last_possible_start = len(records) - 1
    for start_idx in range(0, len(records), stride):
        if start_idx > last_possible_start:
            break
        start_snapshot = stride_snapshots.get(start_idx)
        if start_snapshot is None:
            continue

        end_idx = start_idx + horizon
        if end_idx < len(records):
            end_snapshot = stride_snapshots.get(end_idx)
            if end_snapshot is None:
                continue
        elif end_idx == len(records) and terminal_snapshot is not None:
            # Complete final window ends at the terminal observation.
            end_snapshot = terminal_snapshot
        elif terminal_snapshot is not None and (
            short_episode or start_idx + stride > len(records)
        ):
            # Short episode (< horizon steps): pad every stride start with the
            # terminal observation. Otherwise only the last partial tail is
            # emitted as one padded terminal window.
            end_snapshot = terminal_snapshot
        else:
            continue

        window = ExecutionWindow(resolve(start_snapshot))
        for record in records[start_idx:end_idx]:
            window.append(record.action, record.source, record.intervention)
        if window.executed_steps == 0:
            continue
        ends_at_terminal = terminal_snapshot is not None and end_snapshot is terminal_snapshot
        transition = build_transition(
            window,
            resolve(end_snapshot),
            horizon=horizon,
            action_dim=action_dim,
            terminal=ends_at_terminal,
        )
        # Per-step reference alignment: a stride window can cross a re-plan
        # boundary (replan every C steps), so its tail rows were executed under
        # the NEXT chunk's VLA reference, not the window-start boundary's.
        # Overwrite each non-HUMAN row with the reference of the chunk that
        # actually covered that step; HUMAN rows keep the executed-action
        # replacement from build_transition (Algorithm 1: intervention replaces
        # the VLA reference).
        for index, record in enumerate(records[start_idx:end_idx]):
            if record.ref_row is not None and record.source != int(TransitionSource.HUMAN):
                transition.ref_chunk[index] = record.ref_row.detach().cpu().numpy()
        transitions.append(transition)
    return transitions


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        sample_strategy: str = "stratified",
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
    ):
        self.capacity = capacity
        self._sample_strategy = sample_strategy
        self._recent_episode_window = recent_episode_window
        self._recent_online_ratio = recent_online_ratio
        self._warmup_demo_ratio = warmup_demo_ratio
        self._human_intervention_ratio = human_intervention_ratio
        self._storage: dict[str, np.ndarray] | None = None
        self._position = 0
        self._size = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    def _initialize_storage(self, record: dict[str, np.ndarray]) -> None:
        self._storage = {}
        for key, value in record.items():
            value = _ensure_array(value)
            self._storage[key] = np.empty((self.capacity, *value.shape), dtype=value.dtype)

    def add(self, transition: Transition) -> None:
        record = transition.to_numpy()
        with self._lock:
            if self._storage is None:
                self._initialize_storage(record)
            for key, value in record.items():
                expected = self._storage[key].shape[1:]
                if value.shape != expected:
                    raise ValueError(f"{key} shape {value.shape} != expected {expected}")
                self._storage[key][self._position] = value
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        with self._lock:
            batch_size = min(batch_size, self._size)
            if batch_size == 0:
                return {}
            if self._sample_strategy == "stratified":
                indices = self._sample_stratified_indices(batch_size)
            else:
                indices = self._sample_uniform_indices(batch_size)
            raw = {k: v[indices].copy() for k, v in self._storage.items()}
        result = {}
        for k, v in raw.items():
            if v.dtype == np.float16:
                result[k] = torch.from_numpy(v.astype(np.float32))
            elif v.dtype == np.bool_:
                tensor = torch.from_numpy(v.astype(np.float32))
                result[k] = tensor.unsqueeze(-1) if k == "done" else tensor
            else:
                result[k] = torch.from_numpy(v.astype(np.float32))
        return result

    def _sample_uniform_indices(self, batch_size: int) -> np.ndarray:
        return self._rng.integers(0, self._size, size=batch_size)

    def _sample_stratified_indices(self, batch_size: int) -> np.ndarray:
        phase = self._storage["collection_phase_id"][:self._size]
        episode_id = self._storage["episode_id"][:self._size]
        source = self._storage["source"][:self._size]
        source_chunk = self._storage["source_chunk"][:self._size]
        intervention = self._storage["intervention_flag"][:self._size]
        all_idx = np.arange(self._size, dtype=np.int64)

        max_ep = int(np.max(episode_id)) if episode_id.size else -1
        recent_start = max_ep - max(self._recent_episode_window, 1) + 1

        recent_pool = np.flatnonzero(
            (phase == COLLECTION_PHASE_ONLINE) & (episode_id >= recent_start)
        )
        warmup_pool = np.flatnonzero(phase == COLLECTION_PHASE_WARMUP)
        human_pool = np.flatnonzero(
            intervention
            | (source == int(TransitionSource.HUMAN))
            | (source == int(TransitionSource.MIXED))
            | np.any(source_chunk == int(TransitionSource.HUMAN), axis=1)
            | np.any(source_chunk == int(TransitionSource.MIXED), axis=1)
        )

        n_recent = int(round(batch_size * self._recent_online_ratio))
        n_warmup = int(round(batch_size * self._warmup_demo_ratio))
        n_human = int(round(batch_size * self._human_intervention_ratio))
        n_uniform = max(batch_size - n_recent - n_warmup - n_human, 0)

        parts = [
            self._pick(recent_pool, n_recent),
            self._pick(warmup_pool, n_warmup),
            self._pick(human_pool, n_human),
            self._pick(all_idx, n_uniform),
        ]
        result = np.concatenate([p for p in parts if p.size > 0])
        if result.size < batch_size:
            result = np.concatenate([result, self._pick(all_idx, batch_size - result.size)])
        self._rng.shuffle(result)
        return result[:batch_size]

    def _pick(self, pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.empty((0,), dtype=np.int64)
        return self._rng.choice(pool, size=min(count, pool.size), replace=True)

    @property
    def intervention_count(self) -> int:
        if self._storage is None or self._size == 0:
            return 0
        return int(np.sum(self._storage["intervention_flag"][:self._size]))

    @property
    def human_count(self) -> int:
        if self._storage is None or self._size == 0:
            return 0
        source = self._storage["source"][:self._size]
        return int(np.sum(
            (source == int(TransitionSource.HUMAN)) | (source == int(TransitionSource.MIXED))
        ))

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "position": self._position,
            "sample_strategy": self._sample_strategy,
            "intervention_count": self.intervention_count,
        }

    def __len__(self):
        return self._size


class ReplayManager:
    """Append-only replay journal. Restore is explicit; writes are disk-first."""

    def __init__(self, buffer: ReplayBuffer, journal_path: str, *, restore: bool = False):
        self._buffer = buffer
        self._journal_path = journal_path
        self._lock = threading.Lock()
        self._adds_total = 0
        os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
        if restore:
            self._restore_from_journal()

    def add_transition(self, transition: Transition) -> None:
        record = transition.to_numpy()
        with self._lock:
            with open(self._journal_path, "ab") as f:
                pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            try:
                self._buffer.add(transition)
            except Exception:
                logger.exception(
                    "Replay memory add failed after durable journal append; aborting to avoid divergence."
                )
                raise
            self._adds_total += 1

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self._buffer.sample(batch_size)

    def stats(self) -> dict:
        base = self._buffer.stats()
        base["adds_total"] = self._adds_total
        base["journal_path"] = self._journal_path
        return base

    def _restore_from_journal(self) -> None:
        if not os.path.exists(self._journal_path):
            return
        restored = 0
        try:
            with open(self._journal_path, "rb") as f:
                while True:
                    try:
                        record = pickle.load(f)
                    except EOFError:
                        break
                    if "valid_mask" not in record or "executed_steps" not in record:
                        raise ValueError(
                            "Replay journal uses the legacy one-control-step schema; "
                            "start with a new --output_dir to avoid mixing misaligned transitions."
                        )
                    t = Transition(
                        z_rl=record["z_rl"].astype(np.float32),
                        proprio=record["proprio"].astype(np.float32),
                        ref_chunk=record["ref_chunk"].astype(np.float32),
                        action_chunk=record["action_chunk"].astype(np.float32),
                        rewards=record["rewards"].astype(np.float32),
                        done=bool(record["done"]),
                        next_z_rl=record["next_z_rl"].astype(np.float32),
                        next_proprio=record["next_proprio"].astype(np.float32),
                        next_ref_chunk=record["next_ref_chunk"].astype(np.float32),
                        source=int(record["source"]),
                        source_chunk=record["source_chunk"].astype(np.uint8),
                        valid_mask=record["valid_mask"].astype(np.bool_),
                        executed_steps=int(record["executed_steps"]),
                        collection_phase="warmup" if record["collection_phase_id"] == COLLECTION_PHASE_WARMUP else "online",
                        success=int(record["success"]),
                        intervention_flag=bool(record["intervention_flag"]),
                        episode_id=int(record["episode_id"]),
                        step_id=int(record["step_id"]),
                    )
                    self._buffer.add(t)
                    restored += 1
        # EOFError means the stream ended cleanly on a record boundary (including
        # an empty file), which is a normal exit. For any other exception: a crash
        # can only truncate the last written record, so if some records were
        # already restored, stop and keep them and let the caller's replay_entries
        # count check report the mismatch. If the very first record fails
        # (restored == 0) the cause is not tail truncation but a schema mismatch or
        # a code defect (e.g. a missing constant), which must propagate verbatim --
        # otherwise it masquerades as a "count mismatch" and misleads debugging.
        except EOFError:
            pass
        except Exception as exc:
            if restored == 0:
                raise
            logger.warning("Journal restore stopped at %d entries: %s", restored, exc)
        self._adds_total = restored
        if restored:
            logger.info("Restored %d transitions from journal %s", restored, self._journal_path)


def write_warmup_interval_records(
    out_file,
    ep: int,
    frames_in_interval: list,
    frame_data: dict,
    *,
    horizon: int,
    stride: int,
    action_dim: int,
) -> int:
    """Build stride-2 replay windows inside one annotated interval and write warmup
    journal records (pickle stream, schema identical to Stage-2 online transitions
    with ``collection_phase_id=WARMUP``).

    ``frame_data`` maps ``(episode, frame) -> (z_rl fp16, proprio fp32, action fp32)``.
    Returns the number of records written.
    """
    f = out_file  # verbatim body references ``f``
    n_records = 0
    for pos in range(0, len(frames_in_interval), stride):
        window_frames = frames_in_interval[pos:pos + horizon]
        if len(window_frames) < 1:
            break
        t0f = window_frames[0]
        t1f = frames_in_interval[min(pos + horizon, len(frames_in_interval) - 1)]
        z0, p0, _ = frame_data[(ep, t0f)]
        z1, p1, _ = frame_data[(ep, t1f)]

        action_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        next_ref_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        valid_mask = np.zeros(horizon, dtype=np.bool_)
        executed = 0
        for i, fr in enumerate(window_frames):
            action_chunk[i] = frame_data[(ep, fr)][2]
            valid_mask[i] = True
            executed += 1
        # next window's demo actions as next_ref_chunk (may be all-zero padding)
        next_frames = frames_in_interval[pos + horizon:pos + 2 * horizon]
        for i, fr in enumerate(next_frames):
            next_ref_chunk[i] = frame_data[(ep, fr)][2]

        record = {
            "z_rl": z0.astype(np.float16),
            "proprio": p0.astype(np.float32),
            "ref_chunk": action_chunk.astype(np.float16),
            "action_chunk": action_chunk.astype(np.float16),
            "rewards": np.zeros(horizon, dtype=np.float32),
            "done": np.asarray(False, dtype=np.bool_),
            "next_z_rl": z1.astype(np.float16),
            "next_proprio": p1.astype(np.float32),
            "next_ref_chunk": next_ref_chunk.astype(np.float16),
            "source": np.asarray(TransitionSource.BASE, dtype=np.uint8),
            "source_chunk": np.full(horizon, TransitionSource.BASE, dtype=np.uint8),
            "valid_mask": valid_mask,
            "executed_steps": np.asarray(executed, dtype=np.int16),
            "collection_phase_id": np.asarray(COLLECTION_PHASE_WARMUP, dtype=np.uint8),
            "success": np.asarray(0, dtype=np.int8),
            "intervention_flag": np.asarray(False, dtype=np.bool_),
            "episode_id": np.asarray(ep, dtype=np.int32),
            "step_id": np.asarray(t0f, dtype=np.int32),
        }
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_records += 1
    return n_records


__all__ = [
    "PANTHERA_JOINT_NAMES",
    "action_order_indices",
    "reorder_action_tensor",
    "RerunVisualizationManager",
    "TransitionSource",
    "collection_phase_to_id",
    "COLLECTION_PHASE_UNKNOWN",
    "COLLECTION_PHASE_WARMUP",
    "COLLECTION_PHASE_ONLINE",
    "_ensure_array",
    "Transition",
    "Pi05ActionCache",
    "make_vla_reference_chunk",
    "subsample_vla_actions",
    "make_pi05_reference_chunk",
    "subsample_pi05_actions",
    "BoundaryContext",
    "ExecutionWindow",
    "DeferredBoundarySnapshot",
    "DeferredExecutionWindow",
    "TakeoverState",
    "StepRecord",
    "clone_observation",
    "materialize_deferred_windows",
    "build_transition",
    "order_episode_transitions",
    "build_stride2_transitions",
    "ReplayBuffer",
    "ReplayManager",
    "write_warmup_interval_records",
]
