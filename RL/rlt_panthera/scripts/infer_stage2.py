#!/usr/bin/env python3
"""Stage-2 deployment inference (Phase 5): frozen VLA + trained RL actor, NO learning.

Runs on the execution machine against serve_rlt. Loads a Stage-2 checkpoint
(--stage2_checkpoint, from training), executes:

  - VLA windows until the critical phase (manual 'c' by default; optional
    auto-detector with a CONSERVATIVE threshold pair, PLAN §五);
  - inside the critical phase with 'a' enabled: the actor's MEAN action plus
    small exploration noise (--actor_noise_std, default 0.02), full 10-step
    chunk execution;
  - no replay, no learner, no reward prompts. Optional per-episode y/n
    scoring (--score) and raw rollout recording (--record DIR).

Usage:
  python scripts/infer_stage2.py --config configs/stage2_panthera.json \
      --stage2_checkpoint <dir>/final_checkpoint.pt [--auto_critical \
      --critical_classifier critical_classifier.pt] [--score] [--record rollouts]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rlt.critical import CriticalPhaseDetector  # noqa: E402
from rlt.networks import RLTChunkActor, RLTConfig  # noqa: E402
from rlt.norm import load_quantile_stats  # noqa: E402
from rlt.replay import make_vla_reference_chunk  # noqa: E402
import train_stage2 as t2  # noqa: E402  (console globals + helpers)
from train_stage2 import ConsoleInputManager, VLAClient, load_robot  # noqa: E402
import signal  # noqa: E402

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="JSON config (same keys as train_stage2)")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the bolt and place it on the nut")
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--robot_module", default="robot.panthera_robot")
    parser.add_argument("--control_fps", type=float, default=30.0)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--actor_noise_std", type=float, default=0.02)
    parser.add_argument("--reset_between_episodes", action="store_true")
    parser.add_argument("--auto_critical", action="store_true")
    parser.add_argument("--critical_classifier", default=None)
    # Conservative deployment thresholds (PLAN §五: 保守 auto-critical 阈值).
    parser.add_argument("--auto_critical_threshold_on", type=float, default=0.8)
    parser.add_argument("--auto_critical_threshold_off", type=float, default=0.4)
    parser.add_argument("--auto_critical_smooth_steps", type=int, default=5)
    parser.add_argument("--auto_critical_min_on_chunks", type=int, default=3)
    parser.add_argument("--auto_critical_start_delay_steps", type=int, default=60)
    parser.add_argument("--score", action="store_true", help="y/n per episode (counting only)")
    parser.add_argument("--record", default="", help="Directory for raw rollout pickles")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        config = json.loads(t2._strip_json_comments(Path(args.config).read_text()))
        defaults = {k: v for k, v in config.items()
                    if k in {"server_host", "server_port", "task", "stats_path",
                             "robot_module", "control_fps", "steps_per_episode",
                             "reset_between_episodes"}}
        for key, value in defaults.items():
            setattr(args, key, value)

    device = torch.device(args.device)
    rlt_config = RLTConfig(mode="inference")
    ckpt = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    actor = RLTChunkActor(rlt_config).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    logger.info("Stage-2 actor loaded (episode %d, updates %d)",
                ckpt.get("episode", -1), ckpt.get("total_updates", -1))

    stats = load_quantile_stats(args.stats_path)
    vla = VLAClient(args.server_host, args.server_port, args.task)
    vla.check_same_origin(task=args.task)

    detector = None
    if args.auto_critical:
        if not args.critical_classifier:
            raise ValueError("--auto_critical requires --critical_classifier")
        detector = CriticalPhaseDetector(
            args.critical_classifier, device,
            threshold_on=args.auto_critical_threshold_on,
            threshold_off=args.auto_critical_threshold_off,
            smooth_steps=args.auto_critical_smooth_steps,
            min_on_chunks=args.auto_critical_min_on_chunks,
        )

    robot = load_robot(args.robot_module, dummy=False)
    robot.connect()
    console = ConsoleInputManager(auto_critical=bool(args.auto_critical))
    console.start()

    record_dir = Path(args.record).expanduser() if args.record else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    target_dt = 1.0 / args.control_fps
    scores: list[bool] = []
    window = rlt_config.n_action_steps_rl

    try:
        for episode in range(1, args.max_episodes + 1):
            if t2.SHUTDOWN or t2.EMERGENCY_STOP:
                break
            print(f"\n[deploy] Episode {episode}/{args.max_episodes}", flush=True)
            CRITICAL = False
            ACTOR = False
            _last_z = None
            step_counter = 0
            rollout = {"states": [], "actions": [], "sources": [], "z_rl_norms": []} if record_dir else None

            for block in range(args.steps_per_episode // window):
                if t2.SHUTDOWN or t2.EMERGENCY_STOP or t2.FINISH_EPISODE:
                    t2.FINISH_EPISODE = False
                    break
                step_start = time.time()
                obs = robot.get_observation()
                state = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32))
                raw_actions, z_rl = vla.infer(obs)
                _last_z = z_rl.numpy()

                if detector is not None and step_counter >= args.auto_critical_start_delay_steps \
                        and _last_z is not None:
                    desired = detector.update(
                        _last_z, stats.normalize_state(state)[:rlt_config.state_dim].numpy())
                    if desired is not None:
                        CRITICAL = desired
                        logger.info("[deploy] auto critical %s (P=%.3f)",
                                    "ON" if desired else "OFF", detector.last_probability)

                # Manual 'c' always wins over the auto-detector (PLAN §五).
                if t2.CRITICAL_PHASE_ACTIVE != CRITICAL:
                    CRITICAL = t2.CRITICAL_PHASE_ACTIVE
                ACTOR = t2.ACTOR_ENABLED

                ref_norm = stats.normalize_action(make_vla_reference_chunk(
                    raw_actions, action_dim=rlt_config.action_dim,
                    start_index=0, num_steps=window))
                if not CRITICAL or not ACTOR:
                    requested = stats.normalize_action(raw_actions[:, :window])
                    source = "VLA"
                else:
                    with torch.no_grad():
                        mean, action_std = actor(z_rl, stats.normalize_state(state)[:rlt_config.state_dim].unsqueeze(0),
                                                 ref_norm)
                        requested = mean + torch.randn_like(mean) * args.actor_noise_std
                    source = "ACTOR"

                targets = stats.unnormalize_action(requested)[0]  # (window, 7) absolute
                for offset in range(window):
                    robot.send_action(targets[offset].numpy())
                    if rollout is not None:
                        rollout["states"].append(state.numpy().tolist())
                        rollout["actions"].append(targets[offset].numpy().tolist())
                        rollout["sources"].append(source)
                        rollout["z_rl_norms"].append(float(np.linalg.norm(_last_z)))
                    step_counter += 1
                    rest = target_dt - (time.time() - step_start)
                    if rest > 0:
                        time.sleep(rest)
                    step_start = time.time()
                if block % 10 == 0:
                    logger.info("[deploy] block %d | %s | steps %d", block, source, step_counter)

            if record_dir and rollout and rollout["actions"]:
                path = record_dir / f"rollout_ep{episode:03d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(rollout, f)
                logger.info("[deploy] rollout saved: %s", path)
            if args.score:
                reward, cont, _ = console.prompt_episode_reward()
                if not cont:
                    break
                scores.append(reward > 0)
                logger.info("[deploy] success rate: %.0f%% (%d/%d)",
                            100.0 * sum(scores) / len(scores), sum(scores), len(scores))
            if args.reset_between_episodes:
                robot.reset()
    finally:
        console.stop()
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("robot disconnect failed: %s", exc)
    logger.info("[deploy] done. episodes=%d", len(scores) or args.max_episodes)


if __name__ == "__main__":
    main()
