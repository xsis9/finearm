#!/usr/bin/env python3
"""Post-hoc z_rl sanity report for a trained Stage 1 checkpoint.

Complements train_stage1.py's built-in report by reading episode boundaries
from LeRobot **v3** datasets (``meta/episodes/chunk-*/file-*.parquet``; v2
``meta/episodes.jsonl`` also supported), which the in-training
``episode_boundaries()`` helper does not understand. This adds the
cross-episode variance ratio to the effective-rank / per-dim-std report
(PLAN.md §八 P2) — useful to re-verify any checkpoint without retraining.

Read-only w.r.t. inputs; writes one JSON report.

Usage (A100 training machine):
  /root/xsx/openpi/.venv/bin/python scripts/report_zrl_sanity.py \
      --checkpoint data/stage1_v1/best_checkpoint.pt \
      --precomputed_path data/prefix_v1 \
      --output data/stage1_v1/z_rl_sanity_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_stage1 import load_cache, z_rl_sanity_report  # noqa: E402

from rlt.networks import RLTConfig, RLTokenDecoder, RLTokenEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="z_rl sanity report (with cross-episode variance)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--precomputed_path", type=str, required=True,
                        help="Precomputed embeddings dir (or comma-separated shards).")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def episode_lengths_v3(dataset_repo_id: str) -> list[int] | None:
    """Frame counts per episode from a LeRobot v3 dataset (parquet episodes)."""
    episodes_dir = Path("/root/xsx/data") / dataset_repo_id / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return None
    files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        return None
    import pyarrow.parquet as pq

    lengths: list[int] = []
    for f in files:
        table = pq.read_table(f, columns=["episode_index", "length"])
        for ep, length in zip(table.column("episode_index").to_pylist(),
                              table.column("length").to_pylist()):
            lengths.append((int(ep), int(length)))
    lengths.sort()  # by episode_index, parquet chunks are usually already sorted
    return [length for _, length in lengths] or None


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = {k: ckpt.get(k) for k in ("step", "loss", "best_loss", "rlt_architecture", "image_only")}
    logger.info("Checkpoint: %s | %s", args.checkpoint, meta)
    if ckpt.get("rlt_architecture") != "paper_v1" or not ckpt.get("image_only", False):
        raise ValueError("Not a paper_v1 image-only Stage 1 checkpoint.")

    config = RLTConfig(mode="rlt_training")
    if ckpt.get("config") and ckpt["config"].get("vlm_hidden_dim") != config.vlm_hidden_dim:
        raise ValueError("Checkpoint vlm_hidden_dim mismatch.")
    encoder = RLTokenEncoder(config).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    shard_paths = [Path(p.strip()) for p in str(args.precomputed_path).split(",") if p.strip()]
    # load_cache validates shard sizes/keys; decoder only used for its pos_embed limit.
    dataset, _, _ = load_cache(shard_paths, config, RLTokenDecoder(config))

    repo_id = json.loads((shard_paths[0] / "meta.json").read_text()).get("dataset_repo_id", "")
    lengths = episode_lengths_v3(repo_id) if repo_id else None
    if lengths is None:
        logger.warning("No episode metadata found for %r; cross-episode ratio will be skipped.", repo_id)
    else:
        logger.info("Episodes: %d, total frames: %d", len(lengths), sum(lengths))

    report = z_rl_sanity_report(
        encoder, dataset, device,
        n_samples=args.n_samples, batch_size=args.batch_size,
        use_amp=False, amp_dtype=torch.float32,
        episode_lengths=lengths,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_step"] = int(ckpt.get("step", -1))
    report["checkpoint_loss"] = float(ckpt.get("loss", float("nan")))

    output = Path(args.output) if args.output else Path(args.checkpoint).with_name("z_rl_sanity_full.json")
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written: %s", output)
    logger.info("effective_rank=%.1f / %d, cross_episode_var_ratio_median=%s",
                report["effective_rank"], report["z_dim"],
                report.get("cross_episode_var_ratio_median", "n/a"))


if __name__ == "__main__":
    main()
