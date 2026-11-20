#!/usr/bin/env python3
"""RLT Stage 2: online RL on the Panthera-HT arm (π0.5 via serve_rlt).

Moved from ``RL/rlt_panthera/scripts/train_stage2.py`` in Phase 5a (that
path is now a shim). PipelinedChunkProvider and VLAClient were extracted
to ``finearm.rl.loop.chunk_provider`` / ``finearm.rl.serving.client``;
everything else — including the console state and CriticalPhaseRecorder —
stays here until the Phase 5b interface work.

Execution machine (4060, robot-attached). Port of the reference
``train_rlt_stage2_pi05.py`` for the openpi-JAX + websocket architecture:

  - the frozen π0.5 + Stage-1 encoder run on the 4090 (``serve_rlt.py``); this
    process only talks websocket (openpi-client, pure python + torch learner
    on CPU). Two connections: one for the control path, one for background
    boundary materialization (stride-2 / terminal / HIL snapshots), so replay
    inference never stalls control.
  - hardware is accessed ONLY through ``robot/robot_interface.py``
    (the acquisition machine's implementation adapted to that contract);
    ``--dry_run`` uses ``DummyRobot`` and requires no hardware.
  - actions are ABSOLUTE 7-dim joint targets end-to-end: serve returns (30,7)
    absolute, the replay/learner space is quantile-normalized (stats.json).

Timing reality (measured Phase 3): one serve call ≈ RTT (p99 ~0.6 s through
the ssh tunnel). With ``--replan_every_window`` (paper Algorithm 1: reference
~ pi_vla(s_t)) the robot runs stop-and-go: 0.33 s of motion per window, then
waits one RTT for the next plan — the reference project shipped this same
cadence with 1-5 s in-process diffusion stalls. Replay materialization is
fully async (background connection + threads), hidden behind execution and
the y/n prompt.

Keys (console thread): 'e' estop · 'q' quit · 'c' critical phase toggle ·
'a' actor on/off (needs critical) · 'g' skip episode · 'f' finish early ·
'y/n/d/q' terminal reward · 's' HIL force-resume · 'h' HIL toggle ·
'p' pre-episode manual positioning.

Usage:
  # dry-run preflight (no hardware, no server):
  python -m finearm.rl.train_stage2 --config configs/stage2_panthera.json --dry_run
  # vla_only baseline (20 episodes, success-rate stats only):
  python -m finearm.rl.train_stage2 --config configs/stage2_panthera.json --vla_only
  # online RL:
  python -m finearm.rl.train_stage2 --config configs/stage2_panthera.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import logging
import math
import pickle
import random
import re
import shutil
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Four-layer path resolution (CLI > env > configs/local.jsonc > repo-relative
# defaults) — Phase 1 wiring, direct import since this file lives in the
# finearm package.
from finearm.core import paths as _core_paths

from finearm.rl.algorithms.rltoken import REQUIRED_EMBEDDINGS
from finearm.rl.algorithms.rltoken.critical import CriticalPhaseDetector
from finearm.rl.algorithms.rltoken.learner import (
    perform_training_update,
    sample_fixed_std_action,
)
from finearm.rl.loop.checkpoint import (
    atomic_torch_save,
    build_stage2_checkpoint,
    canonical_digest,
    move_optimizer_state_to_device,
    prepare_output_dir,
    prepare_resume_checkpoint,
)
from finearm.rl.loop.rng import capture_rng_state, restore_rng_state
from finearm.rl.algorithms.rltoken.networks import (
    RLTChunkActor,
    RLTConfig,
    RLTTwinCritic,
    RLTokenEncoder,
)
from finearm.rl.serving.client import VLAClient
from finearm.rl.loop import console
from finearm.rl.loop.chunk_provider import PipelinedChunkProvider
from finearm.rl.loop.critical_log import CriticalPhaseRecorder, save_critical_transition
from finearm.rl.loop.episode_runner import EpisodeRunner
from finearm.rl.loop.console import (
    ConsoleInputManager,
    register_signal_handlers,
)
from finearm.rl.boundaries import ReplayBoundaryCollector
from finearm.rl.norm import load_quantile_stats
from finearm.rl.replay import (
    PANTHERA_JOINT_NAMES,
    BoundaryContext,
    DeferredBoundarySnapshot,
    DeferredExecutionWindow,
    ExecutionWindow,
    Pi05ActionCache,
    ReplayBuffer,
    ReplayManager,
    StepRecord,
    TakeoverState,
    TransitionSource,
    build_stride2_transitions,
    build_transition,
    clone_observation,
    materialize_deferred_windows,
    make_vla_reference_chunk,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Config / args ────────────────────────────────────────────────────────────
def _strip_json_comments(text: str) -> str:
    def repl(match):
        s = match.group(0)
        return " " if s.startswith("/") else s.replace("\n", "\n ")
    pattern = re.compile(r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.DOTALL | re.MULTILINE)
    return re.sub(pattern, repl, text)


def parse_relative_limit(value, joint_names) -> dict[str, float] | None:
    """JSON scalar or per-joint map → {joint: positive limit} (radians)."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        limit = float(value)
    else:
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, (int, float)):
            limit = float(parsed)
        elif isinstance(parsed, dict):
            out = {}
            for key, val in parsed.items():
                if float(val) <= 0:
                    raise ValueError(f"relative limit must be positive: {key}={val}")
                out[str(key)] = float(val)
            unknown = set(out) - set(joint_names)
            if unknown:
                raise ValueError(f"unknown joints in relative limit: {sorted(unknown)}")
            return out
        else:
            raise ValueError(f"cannot parse relative target limit: {value!r}")
    if limit <= 0:
        raise ValueError("relative limit must be positive")
    return {name: limit for name in joint_names}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RLT Stage 2: online RL on Panthera-HT (π0.5 via serve_rlt)")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config; explicit CLI arguments override")
    # serve / model
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the bolt and place it on the nut")
    parser.add_argument("--rlt_checkpoint", required=True)
    parser.add_argument("--stats_path", default=None,
                        help="VLA q01/q99 stats json; fallback chain: this flag > "
                             "--config value > FINEARM_STATS_PATH > configs/local.jsonc")
    # robot
    parser.add_argument("--robot_module", default="finearm.rl.envs.robot_interface",
                        help="module providing RobotInterface (adapted from the acquisition machine)")
    # episode / control
    parser.add_argument("--control_fps", type=float, default=30.0)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--warmup_episodes", type=int, default=15)
    parser.add_argument("--reset_between_episodes", action="store_true")
    parser.add_argument("--manual_positioning", action=argparse.BooleanOptionalAction, default=False)
    # modes
    parser.add_argument("--dry_run", action="store_true",
                        help="Preflight: build everything, no hardware, no server calls, exit")
    parser.add_argument("--dummy_robot", action="store_true",
                        help="Use DummyRobot (no hardware) with a real server connection")
    parser.add_argument("--connect_only", action="store_true",
                        help="Safety bring-up check: connect robot (NO homing if "
                             "PANTHERA_RTC_HOME=\"\"), read one observation, one server "
                             "infer, print contract stats, disconnect. No arm motion.")
    parser.add_argument("--vla_only", action="store_true",
                        help="No actor execution / RL updates; success-rate baseline only")
    # RL
    parser.add_argument("--replan_every_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc_mode", action="store_true",
                        help="ASAC-style pipelined chunks (port of the deploy rtc_client): "
                             "continuous motion — replan triggered k steps before the current "
                             "chunk runs out (k adaptive from RTT p95), swap at window "
                             "boundaries with alignment+blend. Reference/z_rl tuples are "
                             "trigger-time (≈RTT-stale); stop-and-go default stays "
                             "paper-faithful (reference ~ pi_vla(s_t)).")
    parser.add_argument("--rtc_k", type=int, default=None,
                        help="Fixed trigger threshold in steps (default: adaptive from RTT p95)")
    parser.add_argument("--rtc_blend", type=int, default=2,
                        help="Linear blend steps at chunk swap (0 = hard switch)")
    parser.add_argument("--actor_start_updates", type=int, default=1000)
    parser.add_argument("--actor_critical_delay_steps", type=int, default=15)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--utd_ratio", type=int, default=5)
    parser.add_argument("--critic_per_actor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--warmup_bc_weight", type=float, default=10.0)
    parser.add_argument("--warmup_q_weight", type=float, default=0.0)
    parser.add_argument("--online_bc_weight", type=float, default=10.0)
    parser.add_argument("--online_q_weight", type=float, default=1.0)
    parser.add_argument("--delta_weight", type=float, default=0.0)
    parser.add_argument("--td_target_clamp", type=float, default=2.0,
                        help="Clamp the critic TD target to [-v, +v]. With terminal "
                             "0/1 rewards the true Q range is [0, 1], so 1.0 bounds "
                             "overestimation; the legacy default 2.0 permits Q to "
                             "inflate to twice the true maximum.")
    parser.add_argument("--learner_warmup_updates", type=int, default=0)
    parser.add_argument("--policy_fixed_std", type=float, default=0.05)
    parser.add_argument("--reference_dropout", type=float, default=0.5)
    parser.add_argument("--actor_hidden_dim", type=int, default=256)
    parser.add_argument("--actor_num_layers", type=int, default=2)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_layers", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    # actor execution
    parser.add_argument("--actor_execute_mean", action="store_true")
    parser.add_argument("--actor_noise_std", type=float, default=0.0)
    parser.add_argument("--actor_execution_scale", type=float, default=1.0)
    parser.add_argument("--actor_execution_steps", type=int, default=0)
    parser.add_argument("--actor_max_relative_target", type=str, default=None)
    parser.add_argument("--actor_blend_beta", type=float, default=1.0,
                        help="Blend the actor chunk toward the VLA reference chunk "
                             "(the BC target) at execution time: 1.0=off, "
                             "0.5 => executed = 0.5*actor + 0.5*vla_ref.")
    parser.add_argument("--actor_ema_alpha", type=float, default=1.0,
                        help="Per-step EMA on the actor chunk in absolute-target space: "
                             "1.0=off, 0.3 => heavy smoothing (~3-step lag at 30 fps).")
    # robustness / logging
    parser.add_argument("--observation_retries", type=int, default=3)
    parser.add_argument("--observation_retry_delay_s", type=float, default=0.25)
    parser.add_argument("--save_debug_obs", action="store_true")
    parser.add_argument("--output_dir", default="checkpoints/stage2_panthera_v1")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--restore_replay", default=None)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--metrics_log", default=None)
    parser.add_argument("--metrics_log_every", type=int, default=20)
    parser.add_argument("--metrics_log_window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    # critical phase
    parser.add_argument("--auto_critical", action="store_true")
    parser.add_argument("--critical_classifier", default=None)
    parser.add_argument("--auto_critical_threshold_on", type=float, default=0.75)
    parser.add_argument("--auto_critical_threshold_off", type=float, default=0.45)
    parser.add_argument("--auto_critical_smooth_steps", type=int, default=5)
    parser.add_argument("--auto_critical_min_on_chunks", type=int, default=2)
    parser.add_argument("--auto_critical_start_delay_steps", type=int, default=60)
    parser.add_argument("--auto_critical_once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_critical_override_cooldown", type=float, default=3.0)
    parser.add_argument("--critical_record_path", default=None)
    parser.add_argument("--critical_frames_dir", default=None)
    parser.add_argument("--save_critical_frames", action=argparse.BooleanOptionalAction, default=True)

    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default=None)
    config_args, _ = config_probe.parse_known_args()
    if config_args.config:
        config_path = Path(config_args.config).expanduser()
        with open(config_path, encoding="utf-8") as file:
            config_values = json.loads(_strip_json_comments(file.read()))
        if not isinstance(config_values, dict):
            parser.error(f"Config must contain a JSON object: {config_path}")
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(config_values) - set(actions))
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        for key in ("actor_max_relative_target",):
            if config_values.get(key) is not None and not isinstance(config_values[key], str):
                config_values[key] = json.dumps(config_values[key])
        if config_values.get("critical_classifier") is None:
            config_values["critical_classifier"] = ""
        parser.set_defaults(**config_values)
        for key in config_values:
            actions[key].required = False
    args = parser.parse_args()
    # --stats_path is no longer argparse-required: resolve through the four
    # layers (CLI/config > FINEARM_STATS_PATH > configs/local.jsonc), then
    # fail with guidance. (Phase 1; rollback = restore required=True.)
    if not args.stats_path:
        resolved = (
            _core_paths.resolve_path("stats_path", env="FINEARM_STATS_PATH")
            if _core_paths is not None
            else None
        )
        if resolved is None:
            parser.error(
                '--stats_path is required: pass --stats_path, set "stats_path" in the '
                "--config JSON, or add it to configs/local.jsonc"
            )
        args.stats_path = str(resolved)
    return args


def load_robot(module_name: str, *, dummy: bool):
    """Import the RobotInterface implementation named by --robot_module."""
    from finearm.rl.envs.robot_interface import DummyRobot, RobotInterface

    if dummy:
        return DummyRobot()
    module = importlib.import_module(module_name)
    candidate = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and issubclass(obj, RobotInterface)
                and obj not in (RobotInterface, DummyRobot)):
            candidate = obj
            break
    if candidate is None:
        raise ValueError(
            f"{module_name} defines no RobotInterface subclass; implement one "
            "(see robot/robot_interface.py contract)"
        )
    return candidate()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    register_signal_handlers()

    args = parse_args()
    if args.control_fps <= 0:
        raise ValueError("--control_fps must be positive")
    if args.steps_per_episode < RLTConfig().n_action_steps_rl:
        raise ValueError("--steps_per_episode must be at least one RL window (10)")
    if args.delta_weight < 0 or not math.isfinite(args.delta_weight):
        raise ValueError("--delta_weight must be >= 0")
    if args.auto_critical and not args.critical_classifier:
        raise ValueError("--auto_critical requires --critical_classifier")
    if not 0.0 < args.auto_critical_threshold_off < args.auto_critical_threshold_on < 1.0:
        raise ValueError("auto-critical thresholds must satisfy 0 < off < on < 1")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("--reference_dropout must be in [0, 1]")
    if args.actor_execution_steps < 0 or args.actor_execution_steps > 10:
        raise ValueError("--actor_execution_steps must be in [0, 10]")
    if not (0.0 < args.actor_blend_beta <= 1.0):
        raise ValueError("--actor_blend_beta must be in (0, 1]")
    if not (0.0 < args.actor_ema_alpha <= 1.0):
        raise ValueError("--actor_ema_alpha must be in (0, 1]")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ── Output dir / resume (fail-closed, dry_run mutates nothing) ─────────
    output_dir = Path(args.output_dir).expanduser()
    resume_source = Path(args.resume_from).expanduser() if args.resume_from else None
    resume_checkpoint = None
    if resume_source is not None:
        prepare_resume_checkpoint(resume_source, output_dir, dry_run=True)
    prepare_output_dir(output_dir, dry_run=args.dry_run)
    if resume_source is not None:
        resume_checkpoint = prepare_resume_checkpoint(resume_source, output_dir, dry_run=args.dry_run)

    if not args.dry_run:
        try:
            handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logging.getLogger().addHandler(handler)
            for h in list(logging.getLogger().handlers):
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                    h.setLevel(logging.WARNING)
        except OSError as exc:
            logger.warning("Could not open train.log: %s", exc)

    # ── Stage-1 encoder (frozen; on the SERVER — loaded here only for the
    #    provenance/architecture check and reference-residual logging) ──────
    rlt_config = RLTConfig(mode="online_rl")
    rlt_path = Path(args.rlt_checkpoint).expanduser()
    if not rlt_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {rlt_path}")
    ckpt = torch.load(rlt_path, map_location="cpu", weights_only=False)
    if ckpt.get("rlt_architecture") != "paper_v1" or not ckpt.get("image_only", False):
        raise ValueError("Stage 2 requires a paper_v1 image-only Stage-1 checkpoint")
    saved_config = ckpt.get("config", {})
    for field_name in ("vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers", "rlt_num_heads"):
        if field_name in saved_config and saved_config[field_name] != getattr(rlt_config, field_name):
            raise ValueError(
                f"Stage-1 checkpoint {field_name}={saved_config[field_name]} != rlt config "
                f"{getattr(rlt_config, field_name)}"
            )
    stage1_step = int(ckpt.get("step", -1))
    encoder = RLTokenEncoder(rlt_config)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()  # kept on CPU; only used for the dry-run forward check
    del encoder  # z_rl always comes from the server

    # ── Quantile stats (absolute joint space → normalized replay space) ────
    stats = load_quantile_stats(args.stats_path)
    action_dim = rlt_config.action_dim
    state_dim = rlt_config.state_dim

    def normalize_state(t: torch.Tensor) -> torch.Tensor:
        return stats.normalize_state(t)

    def normalize_action(t: torch.Tensor) -> torch.Tensor:
        return stats.normalize_action(t)

    def unnormalize_action(t: torch.Tensor) -> torch.Tensor:
        return stats.unnormalize_action(t)

    # ── Actor + critic + targets (learner runs here, on CPU by default) ────
    actor = RLTChunkActor(rlt_config, hidden_dim=args.actor_hidden_dim,
                          num_layers=args.actor_num_layers).to(device)
    actor.fixed_std = args.policy_fixed_std
    actor_target = copy.deepcopy(actor).to(device)
    for p in actor_target.parameters():
        p.requires_grad_(False)
    critic = RLTTwinCritic(rlt_config, hidden_dim=args.critic_hidden_dim,
                           num_layers=args.critic_num_layers).to(device)
    critic_target = RLTTwinCritic(rlt_config, hidden_dim=args.critic_hidden_dim,
                                  num_layers=args.critic_num_layers).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    # ── Provenance digest (四件套) ─────────────────────────────────────────
    runtime_provenance = {
        "server": f"{args.server_host}:{args.server_port}",
        "rlt_checkpoint": str(rlt_path.resolve()),
        "stage1_step": stage1_step,
        "stats_path": str(Path(args.stats_path).expanduser().resolve()),
        "task": args.task,
        "rl_chunk_length": rlt_config.n_action_steps_rl,
        "action_stride": rlt_config.action_stride,
        "reference_dropout": args.reference_dropout,
        "actor_hidden_dim": args.actor_hidden_dim,
        "actor_num_layers": args.actor_num_layers,
        "critic_hidden_dim": args.critic_hidden_dim,
        "critic_num_layers": args.critic_num_layers,
    }
    runtime_provenance["digest"] = canonical_digest(runtime_provenance)
    if resume_checkpoint is not None and resume_checkpoint.get("runtime_provenance") != runtime_provenance:
        raise ValueError("Resume checkpoint runtime provenance does not match this invocation")

    if resume_checkpoint is not None:
        actor.load_state_dict(resume_checkpoint["actor_state_dict"], strict=True)
        actor_target.load_state_dict(resume_checkpoint["actor_target_state_dict"], strict=True)
        critic.load_state_dict(resume_checkpoint["critic_state_dict"], strict=True)
        critic_target.load_state_dict(resume_checkpoint["critic_target_state_dict"], strict=True)
        actor_optimizer.load_state_dict(resume_checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(resume_checkpoint["critic_optimizer"])
        move_optimizer_state_to_device(actor_optimizer, device)
        move_optimizer_state_to_device(critic_optimizer, device)

    # ── Replay buffer + journal ────────────────────────────────────────────
    buffer = ReplayBuffer(args.buffer_capacity, seed=args.seed, sample_strategy="uniform",
                          recent_episode_window=20, recent_online_ratio=0.4,
                          warmup_demo_ratio=0.3, human_intervention_ratio=0.2)
    restore_replay = args.restore_replay.strip() if isinstance(args.restore_replay, str) else args.restore_replay
    restore_replay = restore_replay or None
    replay = None if args.dry_run else ReplayManager(
        buffer, journal_path=str(output_dir / "replay_journal.pkl"),
        restore=resume_checkpoint is not None or restore_replay is not None,
    )
    if restore_replay is not None and resume_checkpoint is None:
        restore_source = Path(restore_replay).expanduser()
        if not restore_source.is_file():
            raise FileNotFoundError(f"--restore_replay journal does not exist: {restore_source}")
        if replay is not None:
            shutil.copyfile(restore_source, Path(replay._journal_path))
            replay._restore_from_journal()
    boundary_collector = ReplayBoundaryCollector(replay) if replay is not None else None
    if replay is not None and resume_checkpoint is not None:
        restored = replay.stats()["adds_total"]
        if restored != resume_checkpoint["replay_entries"]:
            raise ValueError(
                f"Restored replay {restored} entries != checkpoint replay_entries="
                f"{resume_checkpoint['replay_entries']}"
            )
        buffer._rng.bit_generator.state = resume_checkpoint["replay_rng_state"]

    # ── VLA client + same-origin handshake ────────────────────────────────
    vla: VLAClient | None = None
    if not args.dry_run:
        logger.info("Connecting to serve_rlt at %s:%s ...", args.server_host, args.server_port)
        vla = VLAClient(args.server_host, args.server_port, args.task,
                        required_embeddings=REQUIRED_EMBEDDINGS)
        vla.check_same_origin(task=args.task, stage1_step=stage1_step)
        logger.info("serve_rlt metadata: %s",
                    {k: vla.metadata.get(k) for k in
                     ("server", "config_name", "checkpoint_dir", "stage1_checkpoint",
                      "stage1_step", "task_prompt", "fp16_roundtrip", "action_shape")})

    # ── Robot ─────────────────────────────────────────────────────────────
    robot = load_robot(args.robot_module, dummy=args.dry_run or args.dummy_robot)
    logger.info("Robot: %s", type(robot).__name__)
    if args.dry_run:
        logger.info("Dry-run preflight OK: stats/nets/replay validated; no server, no hardware.")
        return
    robot.connect()

    console_mgr = ConsoleInputManager(auto_critical=bool(args.auto_critical))

    rtc_provider = None
    if args.rtc_mode:
        rtc_provider = PipelinedChunkProvider(vla, control_hz=args.control_fps, k=args.rtc_k)
        rtc_provider.start()
        logger.info("[rtc] pipelined chunk provider started (k=%s, blend=%d)",
                    args.rtc_k or "adaptive", args.rtc_blend)
    console_mgr.start()

    actor_relative_limit = parse_relative_limit(args.actor_max_relative_target, PANTHERA_JOINT_NAMES)
    if args.actor_blend_beta != 1.0 or args.actor_ema_alpha != 1.0:
        logger.info("Actor execution filter enabled: blend_beta=%.2f ema_alpha=%.2f "
                    "(executed = EMA(beta*actor + (1-beta)*vla_ref), applied pre-clamp)",
                    args.actor_blend_beta, args.actor_ema_alpha)

    robot_faulted = False
    observation_faulted = False
    observation_fault_reason: str | None = None
    last_valid_target: torch.Tensor | None = None
    actor_ema_state: torch.Tensor | None = None

    def read_robot_observation(context: str) -> dict | None:
        nonlocal observation_faulted, observation_fault_reason, robot_faulted
        if observation_faulted:
            return None
        try:
            return robot.get_observation()
        except Exception as exc:  # noqa: BLE001 - robot SDKs raise broadly
            for attempt in range(1, args.observation_retries + 1):
                time.sleep(args.observation_retry_delay_s)
                try:
                    observation = robot.get_observation()
                    logger.warning("Transient observation fault recovered after %d retries (%s): %s",
                                   attempt, context, exc)
                    return observation
                except Exception as retry_exc:  # noqa: BLE001
                    exc = retry_exc
            observation_faulted = True
            observation_fault_reason = f"{context}: {exc}"
            robot_faulted = True
            console.SHUTDOWN = True
            logger.error("FATAL observation fault (%s); stopping without further commands.",
                         observation_fault_reason)
            return None

    def validate_state(state) -> torch.Tensor:
        values = torch.as_tensor(np.asarray(state, dtype=np.float32))
        if tuple(values.shape) != (state_dim,):
            raise ValueError(f"observation state must be ({state_dim},), got {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise ValueError(f"non-finite state: {values.tolist()}")
        return values

    def send_validated_action(target_deg: torch.Tensor, source: str) -> torch.Tensor | None:
        nonlocal last_valid_target, robot_faulted
        if robot_faulted or observation_faulted or console.EMERGENCY_STOP:
            return None
        values = target_deg.detach().float().cpu().numpy()
        if values.shape != (action_dim,) or not np.isfinite(values).all():
            robot_faulted = True
            logger.error("%s target invalid: %s", source, values.tolist())
            return None
        for retry in range(3):
            try:
                actual = robot.send_action(values)
                actual_tensor = validate_state(actual)
                last_valid_target = actual_tensor.clone()
                max_delta = float((actual_tensor - torch.as_tensor(values)).abs().max())
                if max_delta > 1e-4:
                    logger.warning("%s action clamped; max requested-vs-sent delta %.4f.",
                                   source, max_delta)
                return actual_tensor
            except Exception as exc:  # noqa: BLE001
                if retry == 2:
                    robot_faulted = True
                    logger.error("Robot command fault while sending %s: %s", source, exc)
                    return None
                time.sleep(0.05)
        return None

    def capture_boundary_snapshot(context: str, step_id: int,
                                  *, raw_actions: torch.Tensor | None = None,
                                  raw_action_index: int = 0) -> DeferredBoundarySnapshot | None:
        obs = read_robot_observation(context)
        if obs is None:
            return None
        try:
            state = validate_state(obs["state"])
        except ValueError as exc:
            logger.error("Invalid observation at %s: %s", context, exc)
            return None
        snapshot = DeferredBoundarySnapshot(clone_observation(obs), state, step_id)
        if raw_actions is not None:
            snapshot.raw_actions = raw_actions.detach().cpu().clone()
            snapshot.raw_action_index = raw_action_index
        return snapshot

    target_dt = 1.0 / args.control_fps

    if args.connect_only:
        obs = robot.get_observation()
        state = validate_state(obs["state"])
        t0 = time.time()
        probe = vla.infer(obs)
        actions = probe.actions
        z_rl = probe.embeddings["z_rl"]  # algorithm side (REQUIRED_EMBEDDINGS)
        infer_ms = (time.time() - t0) * 1000.0
        logger.info(
            "connect_only OK | state=%s | images %s/%s | server %.0fms | actions %s "
            "in [%.3f, %.3f] | z_rl(2048) norm=%.2f | finite=%s",
            [round(float(v), 3) for v in state.tolist()],
            tuple(np.asarray(obs["base_image"]).shape), tuple(np.asarray(obs["wrist_image"]).shape),
            infer_ms, tuple(actions.shape), float(actions.min()), float(actions.max()),
            float(z_rl.norm()), bool(torch.isfinite(actions).all() and torch.isfinite(z_rl).all()),
        )
        print("\nconnect_only: 全链路 OK（未动臂）。", flush=True)
        robot.disconnect()
        return

    # ── Async learner (paper Algorithm 1: rollouts and learning overlap) ───
    training_lock = threading.Lock()
    training_stop = threading.Event()
    total_updates = 0 if resume_checkpoint is None else int(resume_checkpoint["total_updates"])

    def training_update() -> bool:
        nonlocal total_updates
        if args.vla_only or console.SHUTDOWN or console.EMERGENCY_STOP:
            return False
        with training_lock:
            if len(buffer) < args.batch_size:
                return False
            batch = replay.sample_batch(args.batch_size)
            if not batch:
                return False
            metrics = perform_training_update(
                actor=actor, actor_target=actor_target,
                critic=critic, critic_target=critic_target,
                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                batch=batch, total_updates=total_updates,
                discount=args.discount, target_tau=args.target_tau,
                warmup_bc_weight=args.warmup_bc_weight, warmup_q_weight=args.warmup_q_weight,
                online_bc_weight=args.online_bc_weight, online_q_weight=args.online_q_weight,
                delta_weight=args.delta_weight, critic_per_actor=args.critic_per_actor,
                reference_dropout=args.reference_dropout,
                td_target_clamp=args.td_target_clamp,
                learner_warmup_updates=args.learner_warmup_updates,
                unnormalize_action=unnormalize_action if args.delta_weight else None,
                device=device,
            )
            metrics_window.append(metrics)
            total_updates += 1
            return True

    def training_worker() -> None:
        while not training_stop.is_set() and not console.SHUTDOWN and not console.EMERGENCY_STOP:
            if args.vla_only or completed_episode < args.warmup_episodes:
                time.sleep(0.05)
                continue
            if len(buffer) < args.batch_size:
                time.sleep(0.05)
                continue
            if total_updates >= args.utd_ratio * replay.stats()["adds_total"]:
                time.sleep(0.01)
                continue
            try:
                if training_update() and total_updates % args.metrics_log_every == 0:
                    write_metrics_line()
                time.sleep(0.001)
            except Exception:  # noqa: BLE001
                logger.exception("Async training update failed; continuing")
                time.sleep(0.1)

    # ── Metrics log (tail-friendly TSV + '#' episode summaries) ────────────
    metrics_log_path = Path(args.metrics_log) if args.metrics_log else output_dir / "rl_metrics.log"
    metrics_log_lock = threading.Lock()
    metrics_window: deque[dict] = deque(maxlen=args.metrics_log_window)
    last_actor_active = 0

    def append_metrics_line(line: str) -> None:
        try:
            with metrics_log_lock:
                with open(metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
        except OSError as exc:
            logger.warning("Metrics log write failed: %s", exc)

    try:
        if not metrics_log_path.exists() or metrics_log_path.stat().st_size == 0:
            append_metrics_line(
                "# updates\tepisode\tbuffer\tq1_mean\tq1_max\tq2_max\ttd_mean\ttd_max\t"
                "td_err\tactorQ\tbc\tdelta\tactor_on")
    except OSError:
        pass

    def write_metrics_line() -> None:
        if not metrics_window:
            return

        def mean(key: str, default=float("nan")) -> float:
            vals = [m.get(key, default) for m in metrics_window
                    if isinstance(m.get(key), (int, float)) and math.isfinite(m.get(key, default))]
            return sum(vals) / len(vals) if vals else float("nan")

        def fmt(value: float, digits: int = 4) -> str:
            return f"{value:.{digits}f}" if math.isfinite(value) else ""

        append_metrics_line(
            f"{total_updates}\t{completed_episode + 1}\t{len(buffer)}\t"
            f"{fmt(mean('q1_mean'))}\t{fmt(mean('q1_max'))}\t{fmt(mean('q2_max'))}\t"
            f"{fmt(mean('td_target_mean'))}\t{fmt(mean('td_target_max'))}\t"
            f"{fmt(mean('critic_loss'), 5)}\t{fmt(mean('actor_q'))}\t"
            f"{fmt(mean('bc_penalty'), 5)}\t{fmt(mean('delta_penalty'), 5)}\t{last_actor_active}"
        )

    def save_stage2_checkpoint(path: Path, ep: int) -> None:
        journal_path = output_dir / "replay_journal.pkl"
        journal_bytes = journal_path.stat().st_size if journal_path.exists() else 0
        with training_lock:
            payload = build_stage2_checkpoint(
                episode=ep,
                actor_state_dict=actor.state_dict(),
                actor_target_state_dict=actor_target.state_dict(),
                critic_state_dict=critic.state_dict(),
                critic_target_state_dict=critic_target.state_dict(),
                actor_optimizer=actor_optimizer.state_dict(),
                critic_optimizer=critic_optimizer.state_dict(),
                episode_rewards=episode_rewards,
                episode_interventions=episode_interventions,
                total_updates=total_updates,
                runtime_provenance=runtime_provenance,
                rng_state=capture_rng_state(),
                replay_rng_state=copy.deepcopy(buffer._rng.bit_generator.state),
                replay_journal_bytes=journal_bytes,
                replay_entries=replay.stats()["adds_total"],
            )
        atomic_torch_save(payload, path)

    episode_rewards = [] if resume_checkpoint is None else list(resume_checkpoint["episode_rewards"])
    episode_interventions = ([] if resume_checkpoint is None
                             else list(resume_checkpoint["episode_interventions"]))
    start_episode = 1 if resume_checkpoint is None else int(resume_checkpoint["episode"]) + 1
    completed_episode = start_episode - 1
    if resume_checkpoint is not None:
        restore_rng_state(resume_checkpoint["rng_state"])

    training_thread = None
    if not args.vla_only:
        training_thread = threading.Thread(target=training_worker, name="rlt-async-trainer",
                                           daemon=True)
        training_thread.start()
        logger.info("Async training worker started (UTD=%d).", args.utd_ratio)

    critical_recorder = (CriticalPhaseRecorder(args.critical_record_path or
                                               output_dir / "critical_samples.pkl"))
    critical_frames_dir = None
    if args.save_critical_frames:
        critical_frames_dir = (Path(args.critical_frames_dir).expanduser()
                               if args.critical_frames_dir else output_dir / "critical_frames")
        critical_frames_dir.mkdir(parents=True, exist_ok=True)
    critical_detector = None
    if args.auto_critical:
        critical_detector = CriticalPhaseDetector(
            args.critical_classifier, device,
            threshold_on=args.auto_critical_threshold_on,
            threshold_off=args.auto_critical_threshold_off,
            smooth_steps=args.auto_critical_smooth_steps,
            min_on_chunks=args.auto_critical_min_on_chunks,
        )

    logger.info("Starting Stage 2: %d episodes | warmup %d | window %d steps @ %.0f fps | server %s:%s",
                args.max_episodes, args.warmup_episodes, rlt_config.n_action_steps_rl,
                args.control_fps, args.server_host, args.server_port)
    logger.info("Keys: e=estop q=quit c=critical a=actor g=skip f=finish h=HIL s=force-resume p=positioning")

    def get_total_updates() -> int:
        """Live read of the training counter (async thread mutates it via nonlocal)."""
        return total_updates

    try:
        # ── Episode execution (5b): the loop lives in EpisodeRunner; the
        # body is verbatim, golden-locked. Setup/teardown stay here.
        runner = EpisodeRunner(
            action_dim=action_dim,
            actor=actor,
            actor_relative_limit=actor_relative_limit,
            append_metrics_line=append_metrics_line,
            args=args,
            boundary_collector=boundary_collector,
            buffer=buffer,
            capture_boundary_snapshot=capture_boundary_snapshot,
            console_mgr=console_mgr,
            critical_detector=critical_detector,
            critical_frames_dir=critical_frames_dir,
            critical_recorder=critical_recorder,
            episode_interventions=episode_interventions,
            episode_rewards=episode_rewards,
            last_valid_target=last_valid_target,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            observation_fault_reason=observation_fault_reason,
            observation_faulted=observation_faulted,
            output_dir=output_dir,
            read_robot_observation=read_robot_observation,
            rlt_config=rlt_config,
            robot=robot,
            robot_faulted=robot_faulted,
            rtc_provider=rtc_provider,
            save_stage2_checkpoint=save_stage2_checkpoint,
            send_validated_action=send_validated_action,
            start_episode=start_episode,
            state_dim=state_dim,
            target_dt=target_dt,
            training_lock=training_lock,
            unnormalize_action=unnormalize_action,
            validate_state=validate_state,
            vla=vla,
            get_total_updates=get_total_updates,
        )
        runner.run()

        # ── cleanup ─────────────────────────────────────────────────────
        if training_thread is not None:
            training_stop.set()
            training_thread.join(timeout=5.0)
        if not (args.dry_run):
            # final_checkpoint.pt intentionally not saved: periodic
            # checkpoint_ep{N}.pt (save_every) are the only resume points.
            with open(output_dir / "training_log.json", "w") as f:
                json.dump({
                    "episode_rewards": episode_rewards,
                    "episode_interventions": episode_interventions,
                    "total_updates": total_updates,
                }, f, indent=2)
        recent = episode_rewards[-20:]
        logger.info("Training complete! episodes=%d updates=%d success(last20)=%.0f%%",
                    len(episode_rewards), total_updates,
                    100.0 * sum(recent) / max(1, len(recent)))
        print(f"\nTraining complete! Episodes: {len(episode_rewards)} | "
              f"Updates: {total_updates} | Success (last 20): "
              f"{sum(recent) / max(1, len(recent)):.0%}", flush=True)
    finally:
        if rtc_provider is not None:
            rtc_provider.stop()
        if training_thread is not None:
            training_stop.set()
            try:
                training_thread.join(timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Async trainer cleanup failed: %s", exc)
        console_mgr.stop()
        if critical_recorder is not None:
            critical_recorder.close()
        if console.EMERGENCY_STOP:
            robot.estop()
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Robot disconnect failed: %s", exc)


if __name__ == "__main__":
    main()
