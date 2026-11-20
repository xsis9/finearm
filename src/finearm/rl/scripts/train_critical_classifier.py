#!/usr/bin/env python3
"""Train the critical-phase classifier (Phase 5) from Stage-2 logs.

Input: the append-only pickle stream written by train_stage2.py
(<output_dir>/critical_samples.pkl): ``sample`` entries carry (z_rl, proprio)
per autonomous chunk; ``toggle`` entries carry the operator's manual 'c'
boundaries (or auto-detector toggles — manual always wins by construction).

Labels are expanded per episode: samples with on_step <= step < off_step are
positive (critical), everything else negative. Training is episode-wise
split (held-out episodes measure generalization, not memorization).

Output checkpoint loads directly into rlt.critical.CriticalPhaseDetector:
{z_dim, proprio_dim, hidden_dim, num_layers, classifier_state_dict,
 proprio_mean, proprio_std} + training report JSON.

Usage (any machine; CPU is fine):
  python scripts/train_critical_classifier.py \
      --samples <stage2_output>/critical_samples.pkl \
      --output checkpoints/critical_classifier.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Moved from RL/rlt_panthera/scripts/train_critical_classifier.py in Phase 5a.

from finearm.rl.algorithms.rltoken.critical import CriticalPhaseClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, help="critical_samples.pkl from Stage 2")
    parser.add_argument("--output", required=True, help="classifier checkpoint path (.pt)")
    parser.add_argument("--report", default="", help="metrics report JSON (default: alongside output)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="BCE positive-class weight (raise if critical frames are rare)")
    parser.add_argument("--val_episodes_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_stream(path: Path):
    entries = []
    with open(path, "rb") as f:
        while True:
            try:
                entries.append(pickle.load(f))
            except EOFError:
                break
    return entries


def expand_labels(entries):
    """Episode-wise label expansion from toggle boundaries.

    Returns (z, proprio, labels, episode_ids). Samples at step within
    [on_step, off_step) of each critical segment are positive.
    """
    toggles = defaultdict(list)  # episode -> [(step, critical)]
    samples = []
    for entry in entries:
        if entry.get("kind") == "toggle":
            toggles[int(entry["episode"])].append((int(entry["step"]), bool(entry["critical"])))
        elif entry.get("kind") == "sample":
            samples.append(entry)
    for episode in toggles:
        toggles[episode].sort()

    z_list, p_list, y_list, ep_list = [], [], [], []
    labeled = 0
    for sample in samples:
        episode, step = int(sample["episode"]), int(sample["step"])
        segments = toggles.get(episode, [])
        label = 0
        # Segment [on_step_i, off_step_i): the LAST 'on' at or before this
        # step whose matching 'off' has not yet happened.
        active = False
        for seg_step, seg_critical in segments:
            if seg_step > step:
                break
            active = seg_critical
        label = int(active)
        z_list.append(np.asarray(sample["z_rl"], dtype=np.float32))
        p_list.append(np.asarray(sample["proprio"], dtype=np.float32))
        y_list.append(label)
        ep_list.append(episode)
        labeled += label
    if not samples:
        raise ValueError("no 'sample' entries in the pickle stream")
    if not toggles:
        raise ValueError("no 'toggle' entries — the operator never pressed 'c'; "
                         "no supervision to train on")
    z = np.stack(z_list)
    p = np.stack(p_list)
    y = np.asarray(y_list, dtype=np.float32)
    eps = np.asarray(ep_list)
    logger.info("Samples: %d (%d critical, %.1f%%) across %d episodes; toggle episodes: %d",
                len(y), labeled, 100.0 * labeled / max(1, len(y)),
                len(set(eps.tolist())), len(toggles))
    return z, p, y, eps


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    z, p, y, eps = expand_labels(load_stream(Path(args.samples)))
    device = torch.device("cpu")
    z_dim, proprio_dim = z.shape[1], p.shape[1]

    # Episode-wise split.
    episodes = sorted(set(eps.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * args.val_episodes_frac)) if len(episodes) > 1 else 0
    val_eps = set(episodes[:n_val])
    train_mask = ~np.isin(eps, list(val_eps))
    val_mask = ~train_mask
    if train_mask.sum() == 0 or (n_val and val_mask.sum() == 0):
        raise ValueError(f"degenerate split: {train_mask.sum()} train / {val_mask.sum()} val samples")
    logger.info("Split: %d train / %d val samples (val episodes: %s)",
                int(train_mask.sum()), int(val_mask.sum()),
                sorted(val_eps) if val_eps else "none")

    proprio_mean = torch.tensor(p[train_mask].mean(axis=0), dtype=torch.float32)
    proprio_std = torch.tensor(p[train_mask].std(axis=0), dtype=torch.float32).clamp_min(1e-6)

    z_t = torch.tensor(z, dtype=torch.float32)
    p_t = (torch.tensor(p, dtype=torch.float32) - proprio_mean) / proprio_std
    y_t = torch.tensor(y, dtype=torch.float32)

    model = CriticalPhaseClassifier(z_dim=z_dim, proprio_dim=proprio_dim,
                                    hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32)

    idx_train = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = idx_train[torch.randperm(len(idx_train))]
        losses = []
        for start in range(0, len(perm), args.batch_size):
            batch = perm[start:start + args.batch_size]
            logits = model(z_t[batch], p_t[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1], y_t[batch], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs:
            val_stats = evaluate(model, z_t, p_t, y_t, val_mask)
            logger.info("epoch %3d | train BCE %.4f | val %s",
                        epoch, sum(losses) / len(losses),
                        {k: round(v, 4) for k, v in val_stats.items()})
            if val_stats["auc"] > best_val_auc:
                best_val_auc = val_stats["auc"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_stats = evaluate(model, z_t, p_t, y_t, val_mask)
    train_stats = evaluate(model, z_t, p_t, y_t, train_mask)
    report = {
        "samples": {"total": int(len(y)), "critical": int(y.sum()),
                    "train": int(train_mask.sum()), "val": int(val_mask.sum())},
        "dims": {"z_dim": z_dim, "proprio_dim": proprio_dim,
                 "hidden_dim": args.hidden_dim, "num_layers": args.num_layers},
        "train": train_stats,
        "val": val_stats,
        "val_episodes": sorted(val_eps),
        "epochs": args.epochs,
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "z_dim": z_dim, "proprio_dim": proprio_dim,
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "classifier_state_dict": model.state_dict(),
        "proprio_mean": proprio_mean.numpy(), "proprio_std": proprio_std.numpy(),
        "val_stats": val_stats,
    }, output)
    report_path = Path(args.report).expanduser() if args.report else output.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved classifier: %s | report: %s", output, report_path)
    logger.info("Held-out separation (PLAN §八 P5 target: in≥0.92, out<0.1): "
                "P(crit|critical)=%.3f  P(crit|non)=%.3f  AUC=%.3f",
                val_stats["mean_prob_pos"], val_stats["mean_prob_neg"], val_stats["auc"])


@torch.no_grad()
def evaluate(model, z_t, p_t, y_t, mask):
    if mask.sum() == 0:
        return {"auc": float("nan"), "mean_prob_pos": float("nan"),
                "mean_prob_neg": float("nan"), "accuracy": float("nan")}
    model.eval()
    probs = torch.softmax(model(z_t[mask], p_t[mask]), dim=-1)[:, 1]
    labels = y_t[mask]
    pos, neg = labels > 0.5, labels <= 0.5
    # Simple AUC by rank (Mann-Whitney).
    order = torch.argsort(probs)
    ranks = torch.empty_like(probs)
    ranks[order] = torch.arange(1, len(probs) + 1, dtype=probs.dtype)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    auc = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg)) \
        if n_pos and n_neg else float("nan")
    preds = (probs > 0.5).float()
    return {
        "auc": auc,
        "mean_prob_pos": float(probs[pos].mean()) if n_pos else float("nan"),
        "mean_prob_neg": float(probs[neg].mean()) if n_neg else float("nan"),
        "accuracy": float((preds == labels).float().mean()),
    }


if __name__ == "__main__":
    main()
