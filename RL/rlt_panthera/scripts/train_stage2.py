#!/usr/bin/env python3
"""RLT Stage 2: online RL on the Panthera-HT arm (π0.5 via serve_rlt).

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
  python scripts/train_stage2.py --config configs/stage2_panthera.json --dry_run
  # vla_only baseline (20 episodes, success-rate stats only):
  python scripts/train_stage2.py --config configs/stage2_panthera.json --vla_only
  # online RL:
  python scripts/train_stage2.py --config configs/stage2_panthera.json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# finearm.core bootstrap (Phase 1; rollback = delete this block). Provides the
# four-layer path resolution (CLI > env > configs/local.jsonc > repo-relative
# defaults). Without finearm, legacy behavior stands unchanged.
try:
    from finearm.core import paths as _core_paths
except ImportError:
    try:
        _deploy_src = REPO_ROOT.parent.parent / "src"
        if str(_deploy_src) not in sys.path:
            sys.path.insert(0, str(_deploy_src))
        from finearm.core import paths as _core_paths
    except ImportError:
        _core_paths = None

from rlt.critical import CriticalPhaseDetector
from rlt.learner import (
    atomic_torch_save,
    build_stage2_checkpoint,
    canonical_digest,
    capture_rng_state,
    move_optimizer_state_to_device,
    perform_training_update,
    prepare_output_dir,
    prepare_resume_checkpoint,
    restore_rng_state,
    sample_fixed_std_action,
)
from rlt.networks import RLTChunkActor, RLTConfig, RLTTwinCritic, RLTokenEncoder
from rlt.norm import load_quantile_stats
from rlt.replay import (
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
    order_episode_transitions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical Panthera action order (must match rlt.replay's canonical order and
# the p3 dataset action feature order: 6 joints + gripper).
PANTHERA_JOINT_NAMES = (
    "target_joint_1", "target_joint_2", "target_joint_3",
    "target_joint_4", "target_joint_5", "target_joint_6", "target_gripper",
)

# ── Global console state (single stdin thread owns all writes) ───────────────
SHUTDOWN = False
EMERGENCY_STOP = False
ACTOR_ENABLED = False
SKIP_EPISODE = False
FINISH_EPISODE = False
CRITICAL_PHASE_ACTIVE = False
CRITICAL_MANUAL_OVERRIDE_AT = 0.0
HIL_FORCE_RESUME = False
HIL_TOGGLE_REQUESTED = False
POSITION_TOGGLE_REQUESTED = False
POSITIONING_STATE = "off"


class ConsoleInputManager:
    """Own stdin in one thread so safety keys and reward prompts never race.

    Ported verbatim from the reference (SO-101 leader comments dropped); see
    the reference docstring for the auto-critical 'a' semantics."""

    def __init__(self, stream=None, *, auto_critical: bool = False):
        self._stream = stream if stream is not None else sys.stdin
        self._responses: deque[str] = deque()
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_toggle_time = 0.0
        self._auto_critical = bool(auto_critical)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _loop(self) -> None:
        global EMERGENCY_STOP, SHUTDOWN, ACTOR_ENABLED, SKIP_EPISODE, FINISH_EPISODE
        global CRITICAL_PHASE_ACTIVE, HIL_FORCE_RESUME, HIL_TOGGLE_REQUESTED
        global CRITICAL_MANUAL_OVERRIDE_AT, POSITION_TOGGLE_REQUESTED
        while self._running and not SHUTDOWN:
            try:
                line = self._stream.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.05)
                continue
            value = line.strip().lower()
            if value.startswith("e"):
                EMERGENCY_STOP = True
                logger.warning("[KEY] EMERGENCY STOP requested (press 'e')")
                print("[KEY] EMERGENCY STOP", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("q"):
                SHUTDOWN = True
                logger.info("[KEY] Graceful quit requested (press 'q')")
                print("[KEY] Quit requested", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("a"):
                if ACTOR_ENABLED:
                    ACTOR_ENABLED = False
                    logger.info("[KEY] Actor mode OFF (press 'a' to re-enable)")
                    print("[KEY] Actor OFF", flush=True)
                elif CRITICAL_PHASE_ACTIVE or self._auto_critical:
                    ACTOR_ENABLED = True
                    logger.info("[KEY] Actor mode ON")
                    print("[KEY] Actor ON", flush=True)
                else:
                    logger.info("[KEY] Actor mode requires the critical phase: press 'c' first, then 'a'")
                    print("[KEY] Actor needs the critical phase (press 'c' first)", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("g"):
                SKIP_EPISODE = True
                print("[KEY] Skip episode", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("f"):
                FINISH_EPISODE = True
                print("[KEY] Finish episode early", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("c"):
                CRITICAL_PHASE_ACTIVE = not CRITICAL_PHASE_ACTIVE
                CRITICAL_MANUAL_OVERRIDE_AT = time.time()
                print(f"[KEY] Critical {'ON' if CRITICAL_PHASE_ACTIVE else 'OFF'}", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("s"):
                if not HIL_FORCE_RESUME:
                    HIL_FORCE_RESUME = True
                    print("[KEY] HIL force-resume", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("h"):
                now = time.monotonic()
                if now - self._last_toggle_time >= 1.0:
                    self._last_toggle_time = now
                    HIL_TOGGLE_REQUESTED = True
                    print("[KEY] HIL toggle", flush=True)
                else:
                    print("[KEY] HIL toggle debounced", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("p"):
                POSITION_TOGGLE_REQUESTED = True
                print(f"[KEY] Positioning toggle [state={POSITIONING_STATE}]", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            with self._condition:
                self._responses.append(value)
                self._condition.notify_all()

    def prompt_episode_reward(self) -> tuple[float, bool, bool]:
        print("\nWas this episode successful? [y/n/d(iscard)/q(uit)]: ", end="", flush=True)
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return 1.0, True, False
            if response in ("n", "no", "0"):
                return 0.0, True, False
            if response in ("d", "discard"):
                return 0.0, True, True
            if response in ("q", "quit"):
                return 0.0, False, False
            print("  Please enter 'y', 'n', 'd', or 'q': ", end="", flush=True)
        return 0.0, False, False

    def prompt_hil_resolution(self) -> str:
        print("\nHuman control paused. [y] success / [n] failure / [r] resume / [q] quit: ",
              end="", flush=True)
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return "success"
            if response in ("n", "no", "0"):
                return "failure"
            if response in ("r", "resume"):
                return "resume"
            if response in ("q", "quit"):
                return "quit"
            print("  Please enter 'y', 'n', 'r', or 'q': ", end="", flush=True)
        return "quit"


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ── rtc_mode: ASAC-style pipelined chunk provider ───────────────────────────
class PipelinedChunkProvider:
    """Port of the deploy rtc_client.RTCRunner execution pattern (2026-11-13,
    user-validated on this task: continuous motion, higher success than
    stop-and-go).

    At most ONE in-flight serve request; the main thread triggers a replan
    when the current chunk has ≤ k steps remaining (k adaptive from RTT p95),
    so the round trip hides behind continued execution. Each result carries
    the TRIGGER-TIME tuple (chunk, z_rl, state) — kept internally consistent
    (≈ RTT-stale relative to the window that consumes it; the replay stride /
    terminal boundaries still materialize from exact boundary observations on
    the background connection, unaffected).
    """

    def __init__(self, vla: "VLAClient", *, control_hz: float, k: int | None = None):
        self._vla = vla
        self._k_user = k
        self._hz = control_hz
        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._payload: dict | None = None       # {"obs", "state"} set by main thread
        self._result: dict | None = None        # {"chunk", "z_rl", "state"}
        self._inflight = False
        self._rtts: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rtc-provider", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._trigger.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def inflight(self) -> bool:
        with self._lock:
            return self._inflight

    def k_eff(self, horizon: int) -> int:
        """Adaptive k = ceil(p95_RTT × hz) + 2, clamped to [3, horizon-5]."""
        cap = max(3, (horizon or 30) - 5)
        if self._k_user is not None:
            return self._k_user
        with self._lock:
            rtts = list(self._rtts)
        if not rtts:
            return 5
        k = int(np.clip(np.ceil(np.percentile(rtts, 95) * self._hz) + 2, 3, cap))
        if len(rtts) < 3:
            return min(max(5, k), cap)
        return k

    def trigger_replan(self, obs: dict, state: torch.Tensor) -> None:
        """Main thread: publish the latest observation and wake the infer loop."""
        with self._lock:
            if self._inflight:
                return
            self._payload = {"obs": obs, "state": state.detach().clone()}
        self._trigger.set()

    def take_result(self) -> dict | None:
        with self._lock:
            result = self._result
            self._result = None
        return result

    def sync_fetch(self, obs: dict, state: torch.Tensor) -> dict:
        """Blocking fetch from the MAIN thread (episode seed / post-HIL reseed).
        Waits for any in-flight request first so the control connection is
        never used concurrently."""
        while True:
            with self._lock:
                if not self._inflight:
                    self._payload = None
                    self._result = None
                    break
            time.sleep(0.01)
        t0 = time.perf_counter()
        actions, z_rl = self._vla.infer(obs)
        with self._lock:
            self._rtts.append(time.perf_counter() - t0)
            self._rtts = self._rtts[-30:]
        return {"chunk": actions.detach(), "z_rl": z_rl.detach(),
                "state": state.detach().clone()}

    def _loop(self) -> None:
        while self._running:
            if not self._trigger.wait(timeout=0.5):
                continue
            self._trigger.clear()
            with self._lock:
                payload = self._payload
                self._payload = None
                if payload is None or self._inflight:
                    continue
                self._inflight = True
            t0 = time.perf_counter()
            try:
                actions, z_rl = self._vla.infer(payload["obs"])
                rtt = time.perf_counter() - t0
                with self._lock:
                    self._result = {"chunk": actions.detach(), "z_rl": z_rl.detach(),
                                    "state": payload["state"]}
                    self._rtts.append(rtt)
                    self._rtts = self._rtts[-30:]
                    self._inflight = False
            except Exception as exc:  # noqa: BLE001
                logger.error("[rtc] pipelined inference failed: %s", exc)
                with self._lock:
                    self._inflight = False


# ── Critical-phase logging (Phase 5 classifier training data) ────────────────
class CriticalPhaseRecorder:
    """Append-only pickle stream of {kind: sample|toggle} dicts (ported)."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._f = open(self._path, "ab")

    def record_sample(self, episode, step, z_rl, proprio) -> None:
        pickle.dump({
            "kind": "sample", "episode": int(episode), "step": int(step),
            "t": time.time(), "z_rl": np.asarray(z_rl, dtype=np.float32),
            "proprio": np.asarray(proprio, dtype=np.float32),
        }, self._f, protocol=pickle.HIGHEST_PROTOCOL)
        self._f.flush()

    def record_toggle(self, episode, step, critical) -> None:
        pickle.dump({
            "kind": "toggle", "episode": int(episode), "step": int(step),
            "t": time.time(), "critical": bool(critical),
        }, self._f, protocol=pickle.HIGHEST_PROTOCOL)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except (ValueError, OSError):
            pass

    @property
    def path(self) -> Path:
        return self._path


def save_critical_transition(frames_dir, camera_names, *, episode, step, direction,
                             source, prob, obs) -> None:
    """Persist one critical on/off transition (CSV row + camera jpgs)."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    events_path = frames_dir / "critical_events.csv"
    if not events_path.exists():
        with open(events_path, "w", newline="") as file:
            csv.writer(file).writerow(["episode", "step", "direction", "source", "prob", "wall_time"])
    with open(events_path, "a", newline="") as file:
        csv.writer(file).writerow([
            int(episode), int(step), direction, source,
            "" if prob is None else f"{prob:.3f}",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
    try:
        import cv2 as _cv2
    except ImportError:
        return
    for cam in camera_names:
        image = obs.get(cam) if isinstance(obs, dict) else None
        if image is None:
            continue
        fname = f"ep{int(episode):03d}_step{int(step):05d}_{direction}_{source}_{cam}.jpg"
        try:
            frame = np.asarray(image)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(str(frames_dir / fname), frame)
        except Exception as exc:  # noqa: BLE001 - saving is best-effort
            logger.warning("Critical frame save failed (%s): %s", fname, exc)


# ── serve_rlt client (control + background connections) ──────────────────────
class VLAClient:
    """Two websocket connections to serve_rlt: control path and background
    boundary materialization (never blocks control; server handles each
    connection independently)."""

    def __init__(self, host: str, port: int, task: str):
        from openpi_client import websocket_client_policy

        self._task = task
        self._policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._bg_policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self._policy.get_server_metadata()

    def check_same_origin(self, *, task: str, stage1_step: int | None = None) -> None:
        meta = self.metadata
        if meta.get("server") != "serve_rlt":
            raise ValueError(f"server metadata is not serve_rlt: {meta.get('server')!r}")
        if meta.get("task_prompt") != task:
            raise ValueError(
                f"server task {meta.get('task_prompt')!r} != config task {task!r} (四件套不一致)"
            )
        if stage1_step is not None and meta.get("stage1_step") is not None:
            if int(meta["stage1_step"]) != int(stage1_step):
                raise ValueError(
                    f"server stage1_step {meta['stage1_step']} != local checkpoint step {stage1_step}"
                )

    def _obs_payload(self, obs: dict) -> dict:
        return {
            "observation/state": np.asarray(obs["state"], dtype=np.float32),
            "observation/image": np.asarray(obs["base_image"]),
            "observation/wrist_image": np.asarray(obs["wrist_image"]),
            "prompt": self._task,
        }

    def infer(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Control path: returns (raw_actions (1,30,7) ABSOLUTE, z_rl (2048,))."""
        out = self._policy.infer(self._obs_payload(obs))
        actions = torch.as_tensor(np.asarray(out["actions"]), dtype=torch.float32).unsqueeze(0)
        z_rl = torch.as_tensor(np.asarray(out["z_rl"]), dtype=torch.float32)
        return actions, z_rl

    def infer_boundary(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Background path (same result; separate connection)."""
        out = self._bg_policy.infer(self._obs_payload(obs))
        actions = torch.as_tensor(np.asarray(out["actions"]), dtype=torch.float32).unsqueeze(0)
        z_rl = torch.as_tensor(np.asarray(out["z_rl"]), dtype=torch.float32)
        return actions, z_rl


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
    parser.add_argument("--robot_module", default="robot.robot_interface",
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
    from robot.robot_interface import DummyRobot, RobotInterface

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
    global SHUTDOWN, EMERGENCY_STOP, SKIP_EPISODE, FINISH_EPISODE, CRITICAL_PHASE_ACTIVE
    global HIL_FORCE_RESUME, HIL_TOGGLE_REQUESTED, POSITION_TOGGLE_REQUESTED, POSITIONING_STATE
    global ACTOR_ENABLED

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
        vla = VLAClient(args.server_host, args.server_port, args.task)
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

    console = ConsoleInputManager(auto_critical=bool(args.auto_critical))

    rtc_provider = None
    if args.rtc_mode:
        rtc_provider = PipelinedChunkProvider(vla, control_hz=args.control_fps, k=args.rtc_k)
        rtc_provider.start()
        logger.info("[rtc] pipelined chunk provider started (k=%s, blend=%d)",
                    args.rtc_k or "adaptive", args.rtc_blend)
    console.start()

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
        global SHUTDOWN
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
            SHUTDOWN = True
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
        if robot_faulted or observation_faulted or EMERGENCY_STOP:
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
        actions, z_rl = vla.infer(obs)
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

    def infer_boundary(obs: dict, state: torch.Tensor, step_id: int, *,
                       raw_actions: torch.Tensor | None = None,
                       raw_action_index: int = 0) -> BoundaryContext:
        """Background materialization: fresh z_rl via the background server
        connection; reference chunk sliced from the cached ABSOLUTE chunk."""
        if vla is None:
            raise RuntimeError("no VLA client")
        fresh_actions, z_rl = vla.infer_boundary(obs)
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

    # ── Async learner (paper Algorithm 1: rollouts and learning overlap) ───
    training_lock = threading.Lock()
    training_stop = threading.Event()
    total_updates = 0 if resume_checkpoint is None else int(resume_checkpoint["total_updates"])

    def training_update() -> bool:
        nonlocal total_updates
        if args.vla_only or SHUTDOWN or EMERGENCY_STOP:
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
        while not training_stop.is_set() and not SHUTDOWN and not EMERGENCY_STOP:
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

    try:
        for episode in range(start_episode, args.max_episodes + 1):
            if SHUTDOWN or EMERGENCY_STOP:
                break
            print(f"\nEpisode {episode}/{args.max_episodes}", flush=True)
            logger.info("Episode %d | buffer=%d (human=%d) | updates=%d",
                        episode, len(buffer), buffer.human_count, total_updates)

            SKIP_EPISODE = False
            FINISH_EPISODE = False
            CRITICAL_PHASE_ACTIVE = False
            # NOTE: ACTOR_ENABLED deliberately persists across episodes
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
                POSITIONING_STATE = "waiting"
                if not POSITION_TOGGLE_REQUESTED:
                    logger.info("Episode %d: press 'p' to position manually, 'p' again to start.",
                                episode)
                    while not SHUTDOWN and not EMERGENCY_STOP and not POSITION_TOGGLE_REQUESTED:
                        time.sleep(0.05)
                    if SHUTDOWN or EMERGENCY_STOP:
                        break
                POSITION_TOGGLE_REQUESTED = False
                POSITIONING_STATE = "active"
                positioning_active = True
                logger.info("Positioning active (human controls the arm); press 'p' to start the episode.")
                while positioning_active and not SHUTDOWN and not EMERGENCY_STOP:
                    if POSITION_TOGGLE_REQUESTED:
                        POSITION_TOGGLE_REQUESTED = False
                        positioning_active = False
                        break
                    human_cmd = robot.latest_human_action()
                    if human_cmd is not None:
                        send_validated_action(torch.as_tensor(np.asarray(human_cmd, dtype=np.float32)),
                                              "POSITION")
                    time.sleep(target_dt)
                POSITIONING_STATE = "off"
                if SHUTDOWN or EMERGENCY_STOP:
                    break
                logger.info("Positioning finished; starting the VLA episode.")

            steps_per_block = rlt_config.n_action_steps_rl
            if args.actor_execution_steps > 0:
                steps_per_block = min(args.actor_execution_steps, rlt_config.n_action_steps_rl)
            max_rollout_blocks = args.steps_per_episode // steps_per_block
            if max_rollout_blocks < 1:
                raise ValueError("--steps_per_episode must be at least one window")

            for step in range(max_rollout_blocks):
                if SHUTDOWN or EMERGENCY_STOP or SKIP_EPISODE or FINISH_EPISODE:
                    break
                step_start = time.time()

                # ── HIL entry (interface handles takeover smoothing) ──
                takeover_requested, _ = robot.poll_human()
                takeover.latch(takeover_requested or HIL_TOGGLE_REQUESTED)
                if not human_active and (takeover.pending or HIL_TOGGLE_REQUESTED):
                    HIL_TOGGLE_REQUESTED = False
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

                force_resume = HIL_FORCE_RESUME or HIL_TOGGLE_REQUESTED
                if force_resume:
                    HIL_FORCE_RESUME = False
                    HIL_TOGGLE_REQUESTED = False
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
                    if CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                        direction = "on" if CRITICAL_PHASE_ACTIVE else "off"
                        if critical_recorder is not None:
                            critical_recorder.record_toggle(episode, episode_step_counter,
                                                            CRITICAL_PHASE_ACTIVE)
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
                        _last_critical_seen = CRITICAL_PHASE_ACTIVE

                    human_actions = []
                    for _ in range(rlt_config.n_action_steps_rl):
                        human_cmd = robot.latest_human_action()
                        if human_cmd is None:
                            logger.warning("No fresh human sample; ending HUMAN chunk early.")
                            break
                        target_deg = torch.as_tensor(np.asarray(human_cmd, dtype=np.float32))
                        if CRITICAL_PHASE_ACTIVE:
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
                            SHUTDOWN = True
                            break
                        normalized = normalize_action(actual.unsqueeze(0)).squeeze(0)
                        human_actions.append(normalized)
                        intervention_steps += 1
                        if CRITICAL_PHASE_ACTIVE:
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

                    if SHUTDOWN or not human_actions:
                        break
                    end_snapshot = capture_boundary_snapshot("HIL chunk boundary",
                                                             episode_step_counter)
                    if end_snapshot is None:
                        break
                    if collect_hil_data and CRITICAL_PHASE_ACTIVE:
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
                                 and not CRITICAL_PHASE_ACTIVE)
                        and time.time() - CRITICAL_MANUAL_OVERRIDE_AT >= args.auto_critical_override_cooldown
                    )
                    if _auto_armed and _last_z_rl is not None:
                        desired = critical_detector.update(
                            _last_z_rl, normalize_state(state_tensor)[:state_dim].numpy())
                        if desired is not None and desired != CRITICAL_PHASE_ACTIVE:
                            CRITICAL_PHASE_ACTIVE = desired
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
                            new_chunk = fresh["chunk"]
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
                            rtc_z_rl = fresh["z_rl"]
                            rtc_state = fresh["state"]
                        if rtc_chunk is None:
                            # Episode seed / post-HIL reseed: synchronous fetch.
                            seeded = rtc_provider.sync_fetch(obs, state_tensor)
                            rtc_chunk, rtc_z_rl, rtc_state = (seeded["chunk"], seeded["z_rl"],
                                                              seeded["state"])
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
                            logger.warning("[rtc] chunk starved for %d/%d steps this window "
                                           "(network jitter); held last target.", starved_steps)
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
                    SHUTDOWN = True
                    break
                _last_z_rl = z_rl.detach().cpu().numpy()

                # ── Critical bookkeeping (post-inference auto fallback) ──
                if (_auto_armed and critical_detector is not None
                        and time.time() - CRITICAL_MANUAL_OVERRIDE_AT
                        >= args.auto_critical_override_cooldown):
                    desired = critical_detector.update(
                        _last_z_rl, normalize_state(state_tensor)[:state_dim].numpy())
                    if desired is not None and desired != CRITICAL_PHASE_ACTIVE:
                        CRITICAL_PHASE_ACTIVE = desired
                        auto_triggered = True
                        if desired:
                            _auto_fired_once = True
                        logger.info("[KEY] Auto critical phase %s (P=%.3f)",
                                    "STARTED" if desired else "STOPPED",
                                    critical_detector.last_probability)
                if CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                    direction = "on" if CRITICAL_PHASE_ACTIVE else "off"
                    toggle_source = "auto" if auto_triggered else "manual"
                    if critical_recorder is not None:
                        critical_recorder.record_toggle(episode, episode_step_counter,
                                                        CRITICAL_PHASE_ACTIVE)
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
                    _last_critical_seen = CRITICAL_PHASE_ACTIVE
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
                if pending_policy_window is not None and CRITICAL_PHASE_ACTIVE:
                    episode_transitions.append(build_transition(
                        pending_policy_window, boundary,
                        horizon=rlt_config.n_action_steps_rl, action_dim=action_dim))
                    pending_policy_window = None

                # ── Actor gating (VLA vs RL) ──
                _critical_sustained_steps = (
                    _critical_sustained_steps + rlt_config.n_action_steps_rl
                    if CRITICAL_PHASE_ACTIVE else 0
                )
                if (
                    not CRITICAL_PHASE_ACTIVE
                    or _critical_sustained_steps < args.actor_critical_delay_steps
                    or args.vla_only
                    or episode <= args.warmup_episodes
                    or total_updates < args.actor_start_updates
                    or not ACTOR_ENABLED
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
                    if CRITICAL_PHASE_ACTIVE:
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
                        SHUTDOWN = True
                        break
                    normalized = normalize_action(actual.unsqueeze(0)).squeeze(0)
                    window.append(normalized, current_source)
                    if CRITICAL_PHASE_ACTIVE:
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
                if SHUTDOWN:
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
                    if window.executed_steps and CRITICAL_PHASE_ACTIVE:
                        deferred_windows.append(DeferredExecutionWindow(
                            start=window.boundary, end=end_snapshot,
                            actions=list(window.actions), sources=list(window.sources),
                            intervention_flags=list(window.intervention_flags),
                        ))
                    human_boundary = end_snapshot
                    continue

                if window.executed_steps < steps_to_execute:
                    break
                pending_policy_window = window if CRITICAL_PHASE_ACTIVE else None

                if args.rtc_mode:
                    rtc_last_cmd = last_valid_target

                if step % 10 == 0:
                    logger.info("  block %s/%s | %s | z_rl|=%.1f | interventions=%d",
                                step, max_rollout_blocks, action_source,
                                float(np.linalg.norm(_last_z_rl)), intervention_steps)
            # ── end of rollout blocks ─────────────────────────────────────

            if not SHUTDOWN and not EMERGENCY_STOP:
                human_active = False
                takeover = TakeoverState()

            flush_deferred_hil_windows()

            if terminal_snapshot is None and step_records and not SHUTDOWN \
                    and not EMERGENCY_STOP and not robot_faulted:
                terminal_snapshot = capture_boundary_snapshot("terminal transition",
                                                              episode_step_counter)
                if terminal_snapshot is not None:
                    with stride_mt_lock:
                        stride_pending.append(terminal_snapshot)
                    start_stride_materialization()

            terminal_pending_window = None
            if pending_policy_window is not None and not SHUTDOWN and not EMERGENCY_STOP \
                    and not robot_faulted:
                terminal_pending_window = pending_policy_window
                pending_policy_window = None

            if observation_faulted:
                logger.error("Discarding %d transitions after observation fault: %s",
                             len(episode_transitions), observation_fault_reason)
                episode_transitions.clear()
                break

            if SKIP_EPISODE:
                SKIP_EPISODE = False
                logger.info("Episode skipped by user ('g').")
                episode_transitions.clear()
                # A skipped episode still counts toward the episode counter, so
                # a save_every boundary landing on it must save a checkpoint
                # (weights + journal are a consistent pair at this point).
                if episode % args.save_every == 0:
                    ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
                    save_stage2_checkpoint(ckpt_path, episode)
                    logger.info("  Saved checkpoint (skipped episode): %s", ckpt_path)
                if not EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
                    robot.reset()
                continue
            if FINISH_EPISODE:
                FINISH_EPISODE = False
                logger.info("Episode %d finished early ('f'); keeping %d transition(s).",
                            episode, len(episode_transitions))

            reward, should_continue, discard_episode = console.prompt_episode_reward()
            if not should_continue:
                logger.info("User requested quit.")
                break
            if discard_episode:
                logger.info("Discarded episode %d with %d transitions.",
                            episode, len(episode_transitions))
                episode_transitions.clear()
                if not EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
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

            episode_transitions[:] = order_episode_transitions(episode_transitions)
            if episode_transitions:
                terminal_indices = [i for i, t in enumerate(episode_transitions) if t.done]
                if not terminal_indices and not SHUTDOWN and not EMERGENCY_STOP \
                        and not robot_faulted:
                    episode_transitions[-1].done = True
                    terminal_indices = [len(episode_transitions) - 1]
                for idx in terminal_indices:
                    terminal = episode_transitions[idx]
                    terminal.rewards[terminal.executed_steps - 1] = reward
            for transition in episode_transitions:
                replay.add_transition(transition)

            episode_rewards.append(reward)
            episode_interventions.append(intervention_steps)
            completed_episode = episode
            last_actor_active = 1 if episode_actor_executed else 0

            if not EMERGENCY_STOP and not robot_faulted and args.reset_between_episodes:
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
                actor_clamp_steps, actor_clamp_max_delta, total_updates)
            append_metrics_line(
                f"# episode {episode} | reward={reward} | transitions={len(episode_transitions)} | "
                f"interventions={intervention_steps} | success_last20={success_rate:.2f} | "
                f"buffer={len(buffer)} (human={buffer.human_count}) | updates={total_updates}")

            if episode % args.save_every == 0:
                ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
                save_stage2_checkpoint(ckpt_path, episode)
                logger.info("  Saved checkpoint: %s", ckpt_path)

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
        console.stop()
        if critical_recorder is not None:
            critical_recorder.close()
        if EMERGENCY_STOP:
            robot.estop()
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Robot disconnect failed: %s", exc)


if __name__ == "__main__":
    main()
