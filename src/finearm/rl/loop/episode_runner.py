"""EpisodeRunner — the Stage-2 episode loop behind protocol seams (5b).

Extracted verbatim from ``train_stage2.main`` in Phase 5b: the loop body is
unchanged (golden-locked byte-for-byte by tests/rl/test_boundary_equivalence.py);
what changed is ownership — ``main`` keeps SETUP and TEARDOWN (clients,
robot, replay, checkpoint machinery, thread lifecycle), the runner owns
EXECUTION of episodes.

Design notes:

- ``__init__`` takes the loop's free variables as a context; ``run()``
  unpacks them into locals so the extracted body stays verbatim.
- ``get_total_updates`` is a LIVE reader (closure over main's counter,
  mutated by the async training thread via nonlocal) — the loop reads it at
  the actor-activation gate and in logs; a plain value copy would freeze
  the gate at its entry-time value.
- Window/takeover/terminal receipts (``boundaries.BoundaryCollector``)
  wire into this loop as the next 5b/5c step; today the collector owns the
  episode-end commit path, invoked via ``boundary_collector``.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

from finearm.rl.algorithms.rltoken.learner import sample_fixed_std_action
from finearm.rl.loop import console
from finearm.rl.loop.critical_log import save_critical_transition
from finearm.rl.replay import (
    BoundaryContext,
    DeferredBoundarySnapshot,
    DeferredExecutionWindow,
    ExecutionWindow,
    PANTHERA_JOINT_NAMES,
    Pi05ActionCache,
    StepRecord,
    TakeoverState,
    TransitionSource,
    build_stride2_transitions,
    build_transition,
    make_vla_reference_chunk,
    materialize_deferred_windows,
)

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Owns episode execution; setup/teardown stay with the entry point."""

    def __init__(self, **ctx) -> None:
        missing = [k for k in ("action_dim", "actor", "actor_relative_limit", "append_metrics_line", "args", "boundary_collector", "buffer", "capture_boundary_snapshot", "console_mgr", "critical_detector", "critical_frames_dir", "critical_recorder", "episode_interventions", "episode_rewards", "last_valid_target", "normalize_action", "normalize_state", "observation_fault_reason", "observation_faulted", "output_dir", "read_robot_observation", "rlt_config", "robot", "robot_faulted", "rtc_provider", "save_stage2_checkpoint", "send_validated_action", "start_episode", "state_dim", "target_dt", "training_lock", "unnormalize_action", "validate_state", "vla", "get_total_updates") if k not in ctx]
        if missing:
            raise TypeError("EpisodeRunner missing context: %s" % missing)
        self._ctx = ctx

    def run(self) -> None:
        ctx = self._ctx
        action_dim = ctx["action_dim"]
        actor = ctx["actor"]
        actor_relative_limit = ctx["actor_relative_limit"]
        append_metrics_line = ctx["append_metrics_line"]
        args = ctx["args"]
        boundary_collector = ctx["boundary_collector"]
        buffer = ctx["buffer"]
        capture_boundary_snapshot = ctx["capture_boundary_snapshot"]
        console_mgr = ctx["console_mgr"]
        critical_detector = ctx["critical_detector"]
        critical_frames_dir = ctx["critical_frames_dir"]
        critical_recorder = ctx["critical_recorder"]
        episode_interventions = ctx["episode_interventions"]
        episode_rewards = ctx["episode_rewards"]
        last_valid_target = ctx["last_valid_target"]
        normalize_action = ctx["normalize_action"]
        normalize_state = ctx["normalize_state"]
        observation_fault_reason = ctx["observation_fault_reason"]
        observation_faulted = ctx["observation_faulted"]
        output_dir = ctx["output_dir"]
        read_robot_observation = ctx["read_robot_observation"]
        rlt_config = ctx["rlt_config"]
        robot = ctx["robot"]
        robot_faulted = ctx["robot_faulted"]
        rtc_provider = ctx["rtc_provider"]
        save_stage2_checkpoint = ctx["save_stage2_checkpoint"]
        send_validated_action = ctx["send_validated_action"]
        start_episode = ctx["start_episode"]
        state_dim = ctx["state_dim"]
        target_dt = ctx["target_dt"]
        training_lock = ctx["training_lock"]
        unnormalize_action = ctx["unnormalize_action"]
        validate_state = ctx["validate_state"]
        vla = ctx["vla"]
        get_total_updates = ctx["get_total_updates"]
        def infer_boundary(obs: dict, state: torch.Tensor, step_id: int, *,
                           raw_actions: torch.Tensor | None = None,
                           raw_action_index: int = 0) -> BoundaryContext:
            """Background materialization: fresh z_rl via the background server
            connection; reference chunk sliced from the cached ABSOLUTE chunk."""
            if vla is None:
                raise RuntimeError("no VLA client")
            fresh = vla.infer_boundary(obs)
            fresh_actions = fresh.actions
            z_rl = fresh.embeddings["z_rl"]  # algorithm side (REQUIRED_EMBEDDINGS)
            if raw_actions is None:
                raw_actions = fresh_actions
                raw_action_index = 0
            ref_chunk = normalize_action(make_vla_reference_chunk(
                raw_actions, action_dim=action_dim,
                start_index=raw_action_index, num_steps=rlt_config.n_action_steps_rl,
            ))
            return BoundaryContext(
                z_rl=z_rl.detach(),
                proprio=normalize_state(state)[:state_dim].detach(),
                ref_chunk=ref_chunk.detach(),
                collection_phase=collection_phase,
                episode_id=episode,
                step_id=step_id,
            )

        for episode in range(start_episode, args.max_episodes + 1):
            if console.SHUTDOWN or console.EMERGENCY_STOP:
                break
            print(f"\nEpisode {episode}/{args.max_episodes}", flush=True)
            logger.info("Episode %d | buffer=%d (human=%d) | updates=%d",
                        episode, len(buffer), buffer.human_count, get_total_updates())

            console.SKIP_EPISODE = False
            console.FINISH_EPISODE = False
            console.CRITICAL_PHASE_ACTIVE = False
            # NOTE: console.ACTOR_ENABLED deliberately persists across episodes
            # (reference-project behavior): press 'a' once and the actor
            # takes over every critical phase until 'a' is pressed again.
            episode_actor_executed = False
            _last_critical_seen = False
            _last_z_rl: np.ndarray | None = None
            _auto_fired_once = False
            _critical_sustained_steps = 0
            episode_transitions = []
            intervention_steps = 0
            source_counts = {s: 0 for s in TransitionSource}
            human_active = False

            collection_phase = "warmup" if episode <= args.warmup_episodes else "online"

            action_cache = Pi05ActionCache()
            # rtc_mode per-episode pipelined-chunk state (tuple = trigger-time obs).
            rtc_chunk: torch.Tensor | None = None
            rtc_cursor = 0
            rtc_z_rl = None
            rtc_state = None
            rtc_last_cmd = None
            rtc_blends: list[torch.Tensor] = []
            window_rows: list[int] = []
            takeover = TakeoverState()
            pending_policy_window: ExecutionWindow | None = None
            deferred_windows: list[DeferredExecutionWindow] = []
            human_boundary: DeferredBoundarySnapshot | None = None
            collect_hil_data = True
            hil_materialization_thread = None
            hil_materialized = None
            hil_materialization_error = None

            actor_clamp_steps = 0
            actor_clamp_max_delta = 0.0
            actor_ema_state = None

            episode_step_counter = 0
            critical_step_counter: int | None = None
            step_records: list[StepRecord] = []
            stride_snapshots: dict[int, DeferredBoundarySnapshot] = {}
            terminal_snapshot: DeferredBoundarySnapshot | None = None
            replay_stride = rlt_config.action_stride
            stride_pending: list[DeferredBoundarySnapshot] = []
            stride_materialized: dict[int, BoundaryContext] = {}
            stride_mt_lock = threading.Lock()
            stride_mt_thread = None
            stride_mt_error: Exception | None = None

            def materialize_snapshot(snapshot: DeferredBoundarySnapshot) -> BoundaryContext:
                key = id(snapshot)
                if key in stride_materialized:
                    return stride_materialized[key]
                return infer_boundary(
                    snapshot.observation, snapshot.state, snapshot.step_id,
                    raw_actions=snapshot.raw_actions,
                    raw_action_index=snapshot.raw_action_index,
                )

            def start_stride_materialization() -> None:
                nonlocal stride_mt_thread
                if stride_mt_thread is not None and stride_mt_thread.is_alive():
                    return

                def worker():
                    nonlocal stride_mt_error
                    idle = 0
                    try:
                        while True:
                            with stride_mt_lock:
                                snap = stride_pending.pop(0) if stride_pending else None
                                if snap is None:
                                    idle += 1
                                    if idle >= 50:
                                        break
                            if snap is None:
                                time.sleep(0.03)
                                continue
                            idle = 0
                            key = id(snap)
                            if key not in stride_materialized:
                                stride_materialized[key] = infer_boundary(
                                    snap.observation, snap.state, snap.step_id,
                                    raw_actions=snap.raw_actions,
                                    raw_action_index=snap.raw_action_index,
                                )
                    except Exception as exc:  # noqa: BLE001
                        stride_mt_error = exc

                stride_mt_thread = threading.Thread(target=worker, daemon=True)
                stride_mt_thread.start()

            def flush_stride_materialization() -> None:
                nonlocal stride_mt_thread, stride_mt_error
                if stride_mt_thread is not None:
                    stride_mt_thread.join(timeout=120.0)
                    stride_mt_thread = None
                while True:
                    with stride_mt_lock:
                        if not stride_pending:
                            break
                        snap = stride_pending.pop(0)
                    key = id(snap)
                    if key not in stride_materialized:
                        try:
                            stride_materialized[key] = infer_boundary(
                                snap.observation, snap.state, snap.step_id,
                                raw_actions=snap.raw_actions,
                                raw_action_index=snap.raw_action_index,
                            )
                        except Exception as exc:  # noqa: BLE001
                            stride_mt_error = exc
                if stride_mt_error is not None:
                    logger.warning("Stride materialization errors: %s", stride_mt_error)
                    stride_mt_error = None

            def start_hil_materialization() -> None:
                nonlocal hil_materialization_thread, hil_materialized, hil_materialization_error
                if hil_materialization_thread is not None or not deferred_windows:
                    return
                batch = list(deferred_windows)
                deferred_windows.clear()
                hil_materialized = None
                hil_materialization_error = None

                def worker():
                    nonlocal hil_materialized, hil_materialization_error
                    try:
                        hil_materialized = materialize_deferred_windows(
                            batch, materialize_snapshot,
                            horizon=rlt_config.n_action_steps_rl, action_dim=action_dim,
                        )
                    except Exception as exc:  # noqa: BLE001
                        hil_materialization_error = exc

                hil_materialization_thread = threading.Thread(target=worker, daemon=True)
                hil_materialization_thread.start()

            def flush_deferred_hil_windows(*, wait: bool = True) -> bool:
                nonlocal collect_hil_data, hil_materialization_thread
                start_hil_materialization()
                if hil_materialization_thread is None:
                    return collect_hil_data
                if wait:
                    hil_materialization_thread.join()
                elif hil_materialization_thread.is_alive():
                    return False
                hil_materialization_thread = None
                if hil_materialization_error is not None:
                    collect_hil_data = False
                    episode_transitions[:] = [t for t in episode_transitions
                                              if not t.intervention_flag]
                    logger.error("HIL materialization failed; no HIL data this episode: %s",
                                 hil_materialization_error)
                    deferred_windows.clear()
                    return False
                episode_transitions.extend(hil_materialized or [])
                if deferred_windows:
                    start_hil_materialization()
                    if not wait:
                        return False
                    return flush_deferred_hil_windows(wait=True)
                return True

            # ── Pre-episode manual positioning (NOT HIL, never replayed) ──
            if args.manual_positioning:
                console.POSITIONING_STATE = "waiting"
                if not console.POSITION_TOGGLE_REQUESTED:
                    logger.info("Episode %d: press 'p' to position manually, 'p' again to start.",
                                episode)
                    while not console.SHUTDOWN and not console.EMERGENCY_STOP and not console.POSITION_TOGGLE_REQUESTED:
                        time.sleep(0.05)
                    if console.SHUTDOWN or console.EMERGENCY_STOP:
                        break
                console.POSITION_TOGGLE_REQUESTED = False
                console.POSITIONING_STATE = "active"
                positioning_active = True
                logger.info("Positioning active (human controls the arm); press 'p' to start the episode.")
                while positioning_active and not console.SHUTDOWN and not console.EMERGENCY_STOP:
                    if console.POSITION_TOGGLE_REQUESTED:
                        console.POSITION_TOGGLE_REQUESTED = False
                        positioning_active = False
                        break
                    human_cmd = robot.latest_human_action()
                    if human_cmd is not None:
                        send_validated_action(torch.as_tensor(np.asarray(human_cmd, dtype=np.float32)),
                                              "POSITION")
                    time.sleep(target_dt)
                console.POSITIONING_STATE = "off"
                if console.SHUTDOWN or console.EMERGENCY_STOP:
                    break
                logger.info("Positioning finished; starting the VLA episode.")

            steps_per_block = rlt_config.n_action_steps_rl
            if args.actor_execution_steps > 0:
                steps_per_block = min(args.actor_execution_steps, rlt_config.n_action_steps_rl)
            max_rollout_blocks = args.steps_per_episode // steps_per_block
            if max_rollout_blocks < 1:
                raise ValueError("--steps_per_episode must be at least one window")

            for step in range(max_rollout_blocks):
                if console.SHUTDOWN or console.EMERGENCY_STOP or console.SKIP_EPISODE or console.FINISH_EPISODE:
                    break
                step_start = time.time()

                # ── HIL entry (interface handles takeover smoothing) ──
                takeover_requested, _ = robot.poll_human()
                takeover.latch(takeover_requested or console.HIL_TOGGLE_REQUESTED)
                if not human_active and (takeover.pending or console.HIL_TOGGLE_REQUESTED):
                    console.HIL_TOGGLE_REQUESTED = False
                    snapshot = capture_boundary_snapshot("HIL entry", episode_step_counter)
                    if snapshot is None:
                        break
                    if pending_policy_window is not None:
                        deferred_windows.append(DeferredExecutionWindow(
                            start=pending_policy_window.boundary, end=snapshot,
                            actions=list(pending_policy_window.actions),
                            sources=list(pending_policy_window.sources),
                            intervention_flags=list(pending_policy_window.intervention_flags),
                        ))
                        pending_policy_window = None
                    if not robot.enter_human_control():
                        logger.warning("HIL entry refused by the robot interface; staying pending.")
                        takeover.handoff_finished(False)
                    else:
                        human_active = True
                        human_boundary = snapshot
                        action_cache.clear()
                        rtc_chunk = None
                        takeover.handoff_finished(True)
                        start_hil_materialization()
                        logger.info("HIL active (hardware human control; 's'/'h' to resume policy)")

                force_resume = console.HIL_FORCE_RESUME or console.HIL_TOGGLE_REQUESTED
                if force_resume:
                    console.HIL_FORCE_RESUME = False
                    console.HIL_TOGGLE_REQUESTED = False
                    logger.info("HIL force-resume by operator ('s'/'h'); exiting human control.")
                if human_active and (robot.human_finished() or force_resume):
                    if flush_deferred_hil_windows(wait=True):
                        logger.info("HIL exited; policy replans and resumes.")
                    else:
                        logger.warning("HIL materialization failed; forcing policy resume.")
                    human_active = False
                    human_boundary = None
                    action_cache.clear()
                    rtc_chunk = None
                    robot.exit_human_control()

                # ── HUMAN direct-follow: one window per chunk ──
                if human_active:
                    start_snapshot = human_boundary
                    if start_snapshot is None:
                        start_snapshot = capture_boundary_snapshot("HIL chunk start",
                                                                   episode_step_counter)
                        human_boundary = start_snapshot
                    if start_snapshot is None:
                        break
                    if console.CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                        direction = "on" if console.CRITICAL_PHASE_ACTIVE else "off"
                        if critical_recorder is not None:
                            critical_recorder.record_toggle(episode, episode_step_counter,
                                                            console.CRITICAL_PHASE_ACTIVE)
                        if critical_frames_dir is not None:
                            save_critical_transition(
                                critical_frames_dir, ("base_image", "wrist_image"),
                                episode=episode, step=episode_step_counter,
                                direction=direction, source="manual", prob=None,
                                obs=start_snapshot.observation)
                        if direction == "off" and terminal_snapshot is None:
                            terminal_snapshot = capture_boundary_snapshot(
                                "critical terminal (off)", episode_step_counter)
                            if terminal_snapshot is not None:
                                with stride_mt_lock:
                                    stride_pending.append(terminal_snapshot)
                                start_stride_materialization()
                        _last_critical_seen = console.CRITICAL_PHASE_ACTIVE

                    human_actions = []
                    for _ in range(rlt_config.n_action_steps_rl):
                        human_cmd = robot.latest_human_action()
                        if human_cmd is None:
                            logger.warning("No fresh human sample; ending HUMAN chunk early.")
                            break
                        target_deg = torch.as_tensor(np.asarray(human_cmd, dtype=np.float32))
                        if console.CRITICAL_PHASE_ACTIVE:
                            if critical_step_counter is None:
                                critical_step_counter = 0
                            if critical_step_counter % replay_stride == 0:
                                if critical_step_counter not in stride_snapshots:
                                    stride_snapshot = capture_boundary_snapshot(
                                        "stride boundary", episode_step_counter)
                                    if stride_snapshot is not None:
                                        stride_snapshots[critical_step_counter] = stride_snapshot
                                        with stride_mt_lock:
                                            stride_pending.append(stride_snapshot)
                                        start_stride_materialization()
                        actual = send_validated_action(target_deg, "HUMAN")
                        if actual is None:
                            console.SHUTDOWN = True
                            break
                        normalized = normalize_action(actual.unsqueeze(0)).squeeze(0)
                        human_actions.append(normalized)
                        intervention_steps += 1
                        if console.CRITICAL_PHASE_ACTIVE:
                            if critical_step_counter is None:
                                critical_step_counter = 0
                            step_records.append(StepRecord(
                                action=normalized, source=int(TransitionSource.HUMAN),
                                intervention=True, step_id=episode_step_counter))
                            critical_step_counter += 1
                        episode_step_counter += 1
                        remaining = target_dt - (time.time() - step_start)
                        if remaining > 0:
                            time.sleep(remaining)
                        step_start = time.time()

                    if console.SHUTDOWN or not human_actions:
                        break
                    end_snapshot = capture_boundary_snapshot("HIL chunk boundary",
                                                             episode_step_counter)
                    if end_snapshot is None:
                        break
                    if collect_hil_data and console.CRITICAL_PHASE_ACTIVE:
                        deferred_windows.append(DeferredExecutionWindow(
                            start=start_snapshot, end=end_snapshot, actions=human_actions,
                            sources=[int(TransitionSource.HUMAN)] * len(human_actions),
                            intervention_flags=[True] * len(human_actions),
                        ))
                    human_boundary = end_snapshot
                    source_counts[TransitionSource.HUMAN] += len(human_actions)
                    continue

                # ── Autonomous boundary: server inference ──
                obs = read_robot_observation("autonomous pre-action")
                if obs is None:
                    break
                auto_triggered = False
                try:
                    state_tensor = validate_state(obs["state"])
                    # Early auto-critical detection with the previous z_rl.
                    _auto_armed = (
                        critical_detector is not None
                        and episode_step_counter >= args.auto_critical_start_delay_steps
                        and not (args.auto_critical_once and _auto_fired_once
                                 and not console.CRITICAL_PHASE_ACTIVE)
                        and time.time() - console.CRITICAL_MANUAL_OVERRIDE_AT >= args.auto_critical_override_cooldown
                    )
                    if _auto_armed and _last_z_rl is not None:
                        desired = critical_detector.update(
                            _last_z_rl, normalize_state(state_tensor)[:state_dim].numpy())
                        if desired is not None and desired != console.CRITICAL_PHASE_ACTIVE:
                            console.CRITICAL_PHASE_ACTIVE = desired
                            auto_triggered = True
                            if desired:
                                _auto_fired_once = True
                            logger.info("[KEY] Auto critical phase %s (early, P=%.3f)",
                                        "STARTED" if desired else "STOPPED",
                                        critical_detector.last_probability)
                    if args.rtc_mode:
                        # ── rtc_mode: pipelined chunks, swap at window boundary ──
                        fresh = rtc_provider.take_result()
                        if fresh is not None:
                            new_chunk = fresh["inference"].actions
                            j = 0
                            if rtc_last_cmd is not None:
                                diff = new_chunk[0, :, :6].cpu() - rtc_last_cmd[:6].cpu()
                                j = int(torch.argmin(torch.linalg.norm(diff, dim=1)))
                            rtc_blends = []
                            for s in range(args.rtc_blend):
                                alpha = (s + 1) / (args.rtc_blend + 1)
                                new_side = new_chunk[0, min(j + s, new_chunk.shape[1] - 1)].cpu()
                                old_side = (rtc_last_cmd if rtc_last_cmd is not None
                                            else new_side)
                                rtc_blends.append((1 - alpha) * old_side + alpha * new_side)
                            rtc_chunk = new_chunk
                            rtc_cursor = (min(j + len(rtc_blends), rtc_chunk.shape[1] - 1)
                                          if args.rtc_blend > 0 else j)
                            rtc_z_rl = fresh["inference"].embeddings["z_rl"]
                            rtc_state = fresh["state"]
                        if rtc_chunk is None:
                            # Episode seed / post-HIL reseed: synchronous fetch.
                            seeded = rtc_provider.sync_fetch(obs, state_tensor)
                            rtc_chunk = seeded["inference"].actions
                            rtc_z_rl = seeded["inference"].embeddings["z_rl"]
                            rtc_state = seeded["state"]
                            rtc_cursor = 0
                            rtc_blends = []
                        raw_actions = rtc_chunk
                        z_rl = rtc_z_rl
                        # Build this window's ABSOLUTE actions + per-step chunk rows.
                        window_abs, window_rows = [], []
                        starved_steps = 0
                        for s in range(rlt_config.n_action_steps_rl):
                            if s < len(rtc_blends):
                                window_abs.append(rtc_blends[s])
                                window_rows.append(min(j + s, rtc_chunk.shape[1] - 1)
                                                   if fresh is not None else rtc_cursor)
                            elif rtc_cursor < rtc_chunk.shape[1]:
                                window_abs.append(rtc_chunk[0, rtc_cursor].cpu())
                                window_rows.append(rtc_cursor)
                                rtc_cursor += 1
                            else:
                                hold = (rtc_last_cmd if rtc_last_cmd is not None
                                        else rtc_chunk[0, -1].cpu())
                                window_abs.append(hold)
                                window_rows.append(rtc_chunk.shape[1] - 1)
                                starved_steps += 1
                        action_index = window_rows[0] if window_rows else 0
                        vla_actions = torch.stack(window_abs).unsqueeze(0)   # (1,10,7) ABSOLUTE
                        rtc_blends = []   # consumed into window_abs; drop so the
                        # next window without a fresh chunk cannot reuse them
                        # Post-window trigger (covers BOTH the just-swapped chunk
                        # landing low after alignment AND normal exhaustion):
                        # fires with the fresh boundary obs while this window
                        # executes, so the next chunk is ready at the boundary.
                        remaining = rtc_chunk.shape[1] - rtc_cursor
                        if (not rtc_provider.inflight
                                and remaining <= rtc_provider.k_eff(rtc_chunk.shape[1])):
                            try:
                                rtc_provider.trigger_replan(obs, state_tensor)
                            except ValueError:
                                pass
                        if starved_steps:
                            # Fixed in 5b: the second %d arg was missing —
                            # every starvation warning misformatted (and raised
                            # under pytest's logging capture).
                            logger.warning("[rtc] chunk starved for %d/%d steps this window "
                                           "(network jitter); held last target.",
                                           starved_steps, rlt_config.n_action_steps_rl)
                    elif args.replan_every_window:
                        action_cache.clear()
                        rtc_chunk = None
                        raw_actions, z_rl = vla.infer(obs)      # (1,30,7) ABSOLUTE
                        action_cache.refresh(raw_actions.detach(), z_rl.detach(), action_dim)
                        vla_actions, action_index, _ = action_cache.take(
                            rlt_config.n_action_steps_rl)
                    else:
                        needs_replan = (
                            action_cache.raw_actions is None
                            or action_cache.next_index + rlt_config.n_action_steps_rl
                            > action_cache.raw_actions.shape[1]
                        )
                        if needs_replan:
                            raw_actions, z_rl = vla.infer(obs)
                            action_cache.refresh(raw_actions.detach(), z_rl.detach(), action_dim)
                        else:
                            _, z_rl = vla.infer(obs)
                            raw_actions = action_cache.raw_actions
                        vla_actions, action_index, _ = action_cache.take(
                            rlt_config.n_action_steps_rl)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Autonomous boundary inference failed: %s", exc)
                    console.SHUTDOWN = True
                    break
                _last_z_rl = z_rl.detach().cpu().numpy()

                # ── Critical bookkeeping (post-inference auto fallback) ──
                if (_auto_armed and critical_detector is not None
                        and time.time() - console.CRITICAL_MANUAL_OVERRIDE_AT
                        >= args.auto_critical_override_cooldown):
                    desired = critical_detector.update(
                        _last_z_rl, normalize_state(state_tensor)[:state_dim].numpy())
                    if desired is not None and desired != console.CRITICAL_PHASE_ACTIVE:
                        console.CRITICAL_PHASE_ACTIVE = desired
                        auto_triggered = True
                        if desired:
                            _auto_fired_once = True
                        logger.info("[KEY] Auto critical phase %s (P=%.3f)",
                                    "STARTED" if desired else "STOPPED",
                                    critical_detector.last_probability)
                if console.CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                    direction = "on" if console.CRITICAL_PHASE_ACTIVE else "off"
                    toggle_source = "auto" if auto_triggered else "manual"
                    if critical_recorder is not None:
                        critical_recorder.record_toggle(episode, episode_step_counter,
                                                        console.CRITICAL_PHASE_ACTIVE)
                    if critical_frames_dir is not None:
                        save_critical_transition(
                            critical_frames_dir, ("base_image", "wrist_image"),
                            episode=episode, step=episode_step_counter,
                            direction=direction, source=toggle_source,
                            prob=critical_detector.last_probability if auto_triggered else None,
                            obs=obs)
                        logger.info("[KEY] Critical transition: ep%d step%d %s",
                                    episode, episode_step_counter, direction)
                    if direction == "off" and terminal_snapshot is None:
                        terminal_snapshot = capture_boundary_snapshot(
                            "critical terminal (off)", episode_step_counter)
                        if terminal_snapshot is not None:
                            with stride_mt_lock:
                                stride_pending.append(terminal_snapshot)
                            start_stride_materialization()
                            logger.info("[KEY] Terminal observation anchored to critical OFF: "
                                        "ep%d step%d", episode, episode_step_counter)
                    _last_critical_seen = console.CRITICAL_PHASE_ACTIVE
                if critical_recorder is not None:
                    critical_recorder.record_sample(
                        episode, episode_step_counter, _last_z_rl,
                        normalize_state(state_tensor)[:state_dim].numpy())

                # ── Boundary + previous-window transition ──
                ref_actions = normalize_action(make_vla_reference_chunk(
                    raw_actions, action_dim=action_dim,
                    start_index=action_index, num_steps=rlt_config.n_action_steps_rl,
                ))
                boundary = BoundaryContext(
                    z_rl=z_rl.detach(),
                    proprio=normalize_state(rtc_state if args.rtc_mode else state_tensor)[:state_dim].detach(),
                    ref_chunk=ref_actions.detach(),
                    collection_phase=collection_phase,
                    episode_id=episode, step_id=episode_step_counter,
                )
                if pending_policy_window is not None and console.CRITICAL_PHASE_ACTIVE:
                    episode_transitions.append(build_transition(
                        pending_policy_window, boundary,
                        horizon=rlt_config.n_action_steps_rl, action_dim=action_dim))
                    pending_policy_window = None

                # ── Actor gating (VLA vs RL) ──
                _critical_sustained_steps = (
                    _critical_sustained_steps + rlt_config.n_action_steps_rl
                    if console.CRITICAL_PHASE_ACTIVE else 0
                )
                if (
                    not console.CRITICAL_PHASE_ACTIVE
                    or _critical_sustained_steps < args.actor_critical_delay_steps
                    or args.vla_only
                    or episode <= args.warmup_episodes
                    or get_total_updates() < args.actor_start_updates
                    or not console.ACTOR_ENABLED
                ):
                    requested_abs = vla_actions                    # (1,10,7) ABSOLUTE
                    action_source = "VLA"
                    current_source = int(TransitionSource.BASE)
                    actor_ema_state = None
                else:
                    with torch.no_grad():
                        with training_lock:
                            action_mean, action_std = actor(
                                z_rl.cpu(), boundary.proprio.unsqueeze(0), ref_actions)
                            if args.actor_execute_mean:
                                requested_norm = action_mean
                                if args.actor_noise_std > 0:
                                    requested_norm = requested_norm + (
                                        torch.randn_like(requested_norm) * args.actor_noise_std)
                            else:
                                requested_norm = sample_fixed_std_action(action_mean, action_std)
                            if args.actor_execution_scale != 1.0:
                                requested_norm = requested_norm * args.actor_execution_scale
                    # Execution-side smoothness filter (execution-only; the
                    # journal still records executed actions and the pure VLA
                    # ref_chunk, so BC/Q learning targets are unchanged).
                    if args.actor_blend_beta != 1.0:
                        # Blend toward the VLA reference chunk (the BC target)
                        # in normalized space; quantile normalize is affine, so
                        # the convex combination commutes with unnormalize.
                        requested_norm = (
                            args.actor_blend_beta * requested_norm
                            + (1.0 - args.actor_blend_beta)
                            * ref_actions.to(requested_norm.dtype))
                    requested_abs = unnormalize_action(requested_norm)
                    if args.actor_ema_alpha != 1.0:
                        # Per-step EMA in absolute-target space; state carries
                        # across consecutive actor chunks and re-initializes
                        # from the last executed target on each (re)takeover.
                        prev = actor_ema_state
                        if prev is None:
                            prev = (last_valid_target.clone()
                                    if last_valid_target is not None
                                    else requested_abs[0, 0].clone())
                        prev = prev.to(requested_abs.device, requested_abs.dtype)
                        filtered = torch.empty_like(requested_abs)
                        for k in range(requested_abs.shape[1]):
                            prev = (args.actor_ema_alpha * requested_abs[0, k]
                                    + (1.0 - args.actor_ema_alpha) * prev)
                            filtered[0, k] = prev
                        requested_abs = filtered
                        actor_ema_state = prev.clone()
                    action_source = "ACTOR"
                    current_source = int(TransitionSource.RL)
                    episode_actor_executed = True

                window = ExecutionWindow(boundary)
                takeover_during_chunk = False
                steps_to_execute = rlt_config.n_action_steps_rl
                if action_source == "ACTOR" and args.actor_execution_steps > 0:
                    steps_to_execute = min(args.actor_execution_steps, rlt_config.n_action_steps_rl)

                for action_offset in range(steps_to_execute):
                    takeover_requested, _ = robot.poll_human()
                    takeover.latch(takeover_requested)
                    if takeover.pending:
                        takeover_during_chunk = True
                        break
                    if console.CRITICAL_PHASE_ACTIVE:
                        if critical_step_counter is None:
                            critical_step_counter = 0
                        if critical_step_counter % replay_stride == 0:
                            if critical_step_counter not in stride_snapshots:
                                stride_snapshot = capture_boundary_snapshot(
                                    "stride boundary", episode_step_counter,
                                    raw_actions=raw_actions,
                                    raw_action_index=(window_rows[action_offset]
                                                      if args.rtc_mode
                                                      else action_index + action_offset),
                                )
                                if stride_snapshot is not None:
                                    stride_snapshots[critical_step_counter] = stride_snapshot
                                    with stride_mt_lock:
                                        stride_pending.append(stride_snapshot)
                                    start_stride_materialization()
                    target_deg = requested_abs[0, action_offset].float()
                    # ACTOR-only per-step clamp (radians, relative to last target).
                    if (action_source == "ACTOR" and actor_relative_limit is not None
                            and last_valid_target is not None):
                        ref = last_valid_target.to(target_deg.device)
                        clamped = target_deg.clone()
                        for j, name in enumerate(PANTHERA_JOINT_NAMES):
                            lim = actor_relative_limit.get(name)
                            if lim is not None:
                                clamped[j] = torch.clamp(target_deg[j], ref[j] - lim, ref[j] + lim)
                        if not torch.equal(clamped, target_deg):
                            actor_clamp_steps += 1
                            actor_clamp_max_delta = max(
                                actor_clamp_max_delta, float((clamped - target_deg).abs().max()))
                        target_deg = clamped
                    actual = send_validated_action(target_deg, action_source)
                    if actual is None:
                        console.SHUTDOWN = True
                        break
                    normalized = normalize_action(actual.unsqueeze(0)).squeeze(0)
                    window.append(normalized, current_source)
                    if console.CRITICAL_PHASE_ACTIVE:
                        if critical_step_counter is None:
                            critical_step_counter = 0
                        row = (window_rows[action_offset] if args.rtc_mode
                               else action_index + action_offset)
                        step_records.append(StepRecord(
                            action=normalized, source=current_source, intervention=False,
                            step_id=episode_step_counter,
                            ref_row=(normalize_action(raw_actions[
                                :, min(row, raw_actions.shape[1] - 1), :action_dim]).squeeze(0)
                                if raw_actions is not None else None),
                        ))
                        critical_step_counter += 1
                    episode_step_counter += 1
                    remaining = target_dt - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
                    step_start = time.time()
                if console.SHUTDOWN:
                    break
                source_counts[TransitionSource(current_source)] += window.executed_steps
                if (action_source == "ACTOR" and args.actor_execution_steps > 0
                        and window.executed_steps < rlt_config.n_action_steps_rl):
                    action_cache.clear()
                    rtc_chunk = None

                if takeover_during_chunk:
                    end_snapshot = capture_boundary_snapshot("mid-chunk takeover boundary",
                                                             episode_step_counter)
                    if end_snapshot is None:
                        break
                    if window.executed_steps and console.CRITICAL_PHASE_ACTIVE:
                        deferred_windows.append(DeferredExecutionWindow(
                            start=window.boundary, end=end_snapshot,
                            actions=list(window.actions), sources=list(window.sources),
                            intervention_flags=list(window.intervention_flags),
                        ))
                    human_boundary = end_snapshot
                    continue

                if window.executed_steps < steps_to_execute:
                    break
                pending_policy_window = window if console.CRITICAL_PHASE_ACTIVE else None

                if args.rtc_mode:
                    rtc_last_cmd = last_valid_target

                if step % 10 == 0:
                    logger.info("  block %s/%s | %s | z_rl|=%.1f | interventions=%d",
                                step, max_rollout_blocks, action_source,
                                float(np.linalg.norm(_last_z_rl)), intervention_steps)
            # ── end of rollout blocks ─────────────────────────────────────

            if not console.SHUTDOWN and not console.EMERGENCY_STOP:
                human_active = False
                takeover = TakeoverState()

            flush_deferred_hil_windows()

            if terminal_snapshot is None and step_records and not console.SHUTDOWN \
                    and not console.EMERGENCY_STOP and not robot_faulted:
                terminal_snapshot = capture_boundary_snapshot("terminal transition",
                                                              episode_step_counter)
                if terminal_snapshot is not None:
                    with stride_mt_lock:
                        stride_pending.append(terminal_snapshot)
                    start_stride_materialization()

            terminal_pending_window = None
            if pending_policy_window is not None and not console.SHUTDOWN and not console.EMERGENCY_STOP \
                    and not robot_faulted:
                terminal_pending_window = pending_policy_window
                pending_policy_window = None

            if observation_faulted:
                logger.error("Discarding %d transitions after observation fault: %s",
                             len(episode_transitions), observation_fault_reason)
                episode_transitions.clear()
                break

            if console.SKIP_EPISODE:
                console.SKIP_EPISODE = False
                logger.info("Episode skipped by user ('g').")
                episode_transitions.clear()
                # A skipped episode still counts toward the episode counter, so
                # a save_every boundary landing on it must save a checkpoint
                # (weights + journal are a consistent pair at this point).
                if episode % args.save_every == 0:
                    ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
                    save_stage2_checkpoint(ckpt_path, episode)
                    logger.info("  Saved checkpoint (skipped episode): %s", ckpt_path)
                if not console.EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
                    robot.reset()
                continue
            if console.FINISH_EPISODE:
                console.FINISH_EPISODE = False
                logger.info("Episode %d finished early ('f'); keeping %d transition(s).",
                            episode, len(episode_transitions))

            reward, should_continue, discard_episode = console_mgr.prompt_episode_reward()
            if not should_continue:
                logger.info("User requested quit.")
                break
            if discard_episode:
                logger.info("Discarded episode %d with %d transitions.",
                            episode, len(episode_transitions))
                episode_transitions.clear()
                if not console.EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
                    robot.reset()
                continue

            if step_records and replay_stride > 1:
                flush_stride_materialization()

            if terminal_pending_window is not None and terminal_snapshot is not None:
                try:
                    terminal_boundary = materialize_snapshot(terminal_snapshot)
                    episode_transitions.append(build_transition(
                        terminal_pending_window, terminal_boundary,
                        horizon=rlt_config.n_action_steps_rl, action_dim=action_dim,
                        terminal=True))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not finalize last autonomous horizon: %s", exc)

            if step_records and replay_stride > 1:
                existing_ids = {t.step_id for t in episode_transitions}
                try:
                    stride_transitions = build_stride2_transitions(
                        step_records, stride_snapshots, terminal_snapshot,
                        horizon=rlt_config.n_action_steps_rl, action_dim=action_dim,
                        stride=replay_stride, materialize_snapshot=materialize_snapshot)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Stride-2 replay failed; keeping block transitions: %s", exc)
                    stride_transitions = []
                added = 0
                for transition in stride_transitions:
                    if transition.step_id not in existing_ids:
                        episode_transitions.append(transition)
                        existing_ids.add(transition.step_id)
                        added += 1
                if added:
                    logger.info("  Stride-2 replay: +%d overlapping (total %d).",
                                added, len(episode_transitions))

            # Boundary seam (5b): ordering, terminal reward stamping, and the
            # durable journal append live in the collector now — golden-locked.
            boundary_collector.commit_episode(
                episode_transitions,
                reward=reward,
                faulted=robot_faulted,
                interrupted=console.SHUTDOWN or console.EMERGENCY_STOP,
            )

            episode_rewards.append(reward)
            episode_interventions.append(intervention_steps)
            completed_episode = episode
            last_actor_active = 1 if episode_actor_executed else 0

            if not console.EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
                robot.reset()

            recent = episode_rewards[-20:]
            success_rate = sum(recent) / len(recent)
            logger.info(
                "  Episode %d | reward=%s | interventions=%d | success(last20)=%.1f%% | "
                "buffer=%d (human=%d) | b/rl/h/m=%d/%d/%d/%d | clamp=%d(Δ%.3f) | updates=%d",
                episode, reward, intervention_steps, 100.0 * success_rate, len(buffer),
                buffer.human_count,
                source_counts[TransitionSource.BASE], source_counts[TransitionSource.RL],
                source_counts[TransitionSource.HUMAN], source_counts[TransitionSource.MIXED],
                actor_clamp_steps, actor_clamp_max_delta, get_total_updates())
            append_metrics_line(
                f"# episode {episode} | reward={reward} | transitions={len(episode_transitions)} | "
                f"interventions={intervention_steps} | success_last20={success_rate:.2f} | "
                f"buffer={len(buffer)} (human={buffer.human_count}) | updates={get_total_updates()}")

            if episode % args.save_every == 0:
                ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
                save_stage2_checkpoint(ckpt_path, episode)
                logger.info("  Saved checkpoint: %s", ckpt_path)
