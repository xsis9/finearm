#!/usr/bin/env python3
"""Stage 1: offline RLT encoder-decoder training on precomputed prefix embeddings.

Port of RL-Token-Pi05-open ``scripts/train_rlt_stage1_pi05.py`` FAST mmap mode
(legacy slow mode dropped: the frozen π0.5 forward lives in
``precompute_prefix.py`` on the openpi/JAX side; this script never imports
openpi). Networks come from the local ``rlt`` package (2048-dim, paper_v1).

Objective (paper Eq. (1)-(2)):
  - append one learned <rl> token to the frozen image-only prefix tokens and
    take the final encoder position → z_rl
  - teacher-forced causal reconstruction from [z_rl, stopgrad(z_1:i-1)]
  - masked reconstruction MSE over valid token positions

Defaults follow the reference: 30k steps, batch 256, AdamW lr 1e-4 (wd 1e-5),
200-step warmup + cosine, grad clip 1.0, AMP bf16 (pass --use_amp).
Post-training it writes a z_rl sanity report (effective rank / per-dim std /
cross-episode variance) as PLAN.md §八 P2 requires.

Usage (queued on train-queue):
  /root/xsx/openpi/.venv/bin/python scripts/train_stage1.py \
      --precomputed_path data/prefix_v1 --output_dir data/stage1_v1 \
      --steps 30000 --batch_size 256 --use_amp --wait-gpu-free-mib 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Moved from RL/rlt_panthera/scripts/train_stage1.py in Phase 5a (shim remains there).

from finearm.rl.algorithms.rltoken.networks import (
    RLTConfig,
    RLTokenDecoder,
    RLTokenEncoder,
    rlt_reconstruction_loss,
)  

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Stage 1 (fast mmap mode): encoder-decoder training")
    parser.add_argument("--precomputed_path", type=str, required=True,
                        help="Precomputed embeddings dir (or comma-separated shard dirs).")
    parser.add_argument("--output_dir", type=str, default="data/stage1_v1")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Effective batch size (optimizer-level). With --grad_accum N the "
                             "DataLoader uses batch_size/N per micro-batch and gradients are "
                             "accumulated — exact for all-True masks (equal masked-mean "
                             "denominators), which this cache guarantees.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation micro-batches per optimizer update "
                             "(memory workaround: e.g. --batch_size 256 --grad_accum 2).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--allow_schedule_change", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one finite forward/backward check, then exit.")
    parser.add_argument("--sanity_samples", type=int, default=2048,
                        help="Frames sampled for the post-training z_rl sanity report.")
    parser.add_argument("--wait-gpu-free-mib", type=int, default=0,
                        help="Block until a GPU has at least this many MiB free (before torch CUDA init).")
    return parser.parse_args()


def pick_and_wait_gpu(min_free_mib: int) -> int:
    """stdlib-only GPU gate (same helper as precompute_prefix.py)."""
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        gpus = []
        for line in result.stdout.strip().splitlines():
            index, free = (part.strip() for part in line.split(","))
            gpus.append((int(index), int(free)))
        chosen, free = max(gpus, key=lambda pair: pair[1])
        if free >= min_free_mib:
            break
        logging.info("waiting for GPU with >= %d MiB free (current free: %s)...", min_free_mib, gpus)
        time.sleep(60)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    return chosen


def atomic_torch_save(obj, path: Path):
    """Write a torch checkpoint atomically in the destination directory."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_args(args):
    for name in ("steps", "batch_size", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.grad_accum < 1 or args.batch_size % args.grad_accum != 0:
        raise ValueError("--grad_accum must divide --batch_size exactly")
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        raise ValueError("--warmup_steps must be between 0 and --steps")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    try:
        requested = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device {args.device!r}") from exc
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda")
    return requested


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore RNG state, tolerating device-mapped byte tensors (best-effort)."""
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("python/numpy RNG state could not be restored: %s", exc)
    torch_state = state.get("torch")
    if torch_state is not None:
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().to("cpu")
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch RNG state could not be restored: %s", exc)
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([
                s.detach().to("cpu") if torch.is_tensor(s) else s for s in cuda_states
            ])
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA RNG state could not be restored: %s", exc)


class MMapPrefixDataset(Dataset):
    """Lazily reads prefix embeddings and masks from mmap-backed cache roots.

    Ported verbatim from the reference repo: multi-shard support (comma
    separated ``--precomputed_path``), per-worker mmap opening, pickle-safe.
    """

    def __init__(self, roots, counts, seq_len: int, hidden_dim: int):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if isinstance(counts, int):
            counts = [counts]
        if len(roots) != len(counts):
            raise ValueError(f"roots/counts length mismatch: {len(roots)} vs {len(counts)}")
        self.roots = [Path(r) for r in roots]
        self.counts = [int(c) for c in counts]
        self.n_samples = int(sum(self.counts))
        if self.n_samples <= 0:
            raise ValueError("multi-shard cache contains zero samples")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self._offsets: list[int] = []
        acc = 0
        for c in self.counts:
            self._offsets.append(acc)
            acc += c
        self._prefix = None
        self._mask = None

    def _open(self):
        # Open independently in every DataLoader process; never pickle live mmap handles.
        if self._prefix is None:
            self._prefix, self._mask = [], []
            for root, n in zip(self.roots, self.counts):
                self._prefix.append(np.memmap(
                    str(Path(root) / "prefix_out.mmap"), dtype=np.float16, mode="r",
                    shape=(n, self.seq_len, self.hidden_dim),
                ))
                self._mask.append(np.memmap(
                    str(Path(root) / "prefix_mask.mmap"), dtype=np.bool_, mode="r",
                    shape=(n, self.seq_len),
                ))

    def __len__(self):
        return self.n_samples

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.n_samples
        if not 0 <= index < self.n_samples:
            raise IndexError(index)
        for shard in range(len(self.roots) - 1, -1, -1):
            if index >= self._offsets[shard]:
                return shard, index - self._offsets[shard]
        raise IndexError(index)

    def __getitem__(self, index):
        self._open()
        shard, local = self._locate(index)
        # Copies are batch-sized/lazy and make the read-only mmap safe for torch tensors.
        prefix = torch.from_numpy(np.array(self._prefix[shard][local], copy=True))
        mask = torch.from_numpy(np.array(self._mask[shard][local], copy=True))
        if not mask.any():
            raise ValueError(f"Cache sample {index} has no valid image tokens.")
        if not torch.isfinite(prefix).all():
            raise ValueError(f"Cache sample {index} contains NaN or Inf prefix embeddings.")
        return prefix, mask

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prefix"] = None
        state["_mask"] = None
        return state


def load_cache(shard_paths: list[Path], rlt_config: RLTConfig, decoder: RLTokenDecoder):
    """Validate schema v2 shards and return (dataset, seq_len, vlm_dim). Ported
    from the reference fast-mode loader (same checks, same error messages)."""
    meta_path = shard_paths[0] / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {shard_paths[0]}. Run precompute_prefix.py first.")
    with open(meta_path) as f:
        meta = json.load(f)
    raw_version = meta.get("schema_version", meta.get("format_version", meta.get("version")))
    if raw_version is None:
        raw_version = 1 if (shard_paths[0] / "actions.mmap").exists() else 2
    cache_version = int(raw_version)
    if cache_version == 2:
        if meta.get("complete") is not True:
            raise ValueError("v2 embedding cache is incomplete; rerun precomputation")
        if meta.get("image_only") is not True:
            raise ValueError("v2 cache must contain image-only Stage 1 embeddings")
        if meta.get("embedding_stage") != "post_transformer":
            raise ValueError("v2 cache must contain post-transformer prefix embeddings")
        if meta.get("language_positions_dropped") is not True:
            raise ValueError("v2 cache must drop language token positions")
    if cache_version not in (1, 2):
        raise ValueError(f"Unsupported embedding cache version {cache_version}; expected v1 or v2")
    seq_len = int(meta["seq_len"])
    vlm_dim = int(meta["vlm_hidden_dim"])
    if int(meta["n_samples"]) <= 0 or seq_len <= 0 or vlm_dim <= 0:
        raise ValueError("Cache dimensions must all be positive")
    if not meta.get("has_mask", False):
        raise ValueError("Paper-aligned Stage 1 requires prefix_mask.mmap; regenerate the cache.")
    if meta.get("dtype", "float16") != "float16":
        raise ValueError("Only float16 prefix_out.mmap caches are supported")
    if vlm_dim != rlt_config.vlm_hidden_dim:
        raise ValueError(f"Cache VLM dim {vlm_dim} does not match RLT config {rlt_config.vlm_hidden_dim}.")
    if seq_len > decoder.pos_embed.shape[1]:
        raise ValueError(f"Cache sequence length {seq_len} exceeds decoder limit {decoder.pos_embed.shape[1]}.")

    shard_counts: list[int] = []
    for sp in shard_paths:
        sp_meta_path = sp / "meta.json"
        if not sp_meta_path.exists():
            raise FileNotFoundError(f"shard is missing meta.json: {sp}")
        with open(sp_meta_path) as f:
            sp_meta = json.load(f)
        sp_n = int(sp_meta["n_samples"])
        for key, expect in (("seq_len", seq_len), ("vlm_hidden_dim", vlm_dim)):
            if int(sp_meta.get(key, -1)) != expect:
                raise ValueError(
                    f"shard {sp} has {key}={sp_meta.get(key)}, which differs from the first shard ({expect})"
                )
        if sp_meta.get("dtype", "float16") != "float16":
            raise ValueError(f"shard {sp} dtype is not float16")
        if sp_meta.get("has_mask", False) is not True:
            raise ValueError(f"shard {sp} is missing prefix_mask.mmap (has_mask=False)")
        if sp_meta.get("complete") is not True:
            raise ValueError(f"shard {sp} is incomplete (complete != true)")
        for filename, expected_size in (
            ("prefix_out.mmap", sp_n * seq_len * vlm_dim * np.dtype("float16").itemsize),
            ("prefix_mask.mmap", sp_n * seq_len * np.dtype("bool").itemsize),
        ):
            p = sp / filename
            if not p.exists() or p.stat().st_size != expected_size:
                actual = p.stat().st_size if p.exists() else "missing"
                raise ValueError(f"Invalid cache file {p}: expected {expected_size} bytes, got {actual}.")
        shard_counts.append(sp_n)
        logger.info("  shard %s: %d samples", sp.name, sp_n)

    dataset = MMapPrefixDataset(shard_paths, shard_counts, seq_len, vlm_dim)
    logger.info("Using lazy mmap cache v%d: %d shard(s) / %d samples, seq_len=%d, vlm_hidden_dim=%d",
                cache_version, len(shard_paths), len(dataset), seq_len, vlm_dim)
    return dataset, seq_len, vlm_dim


def episode_boundaries(dataset_repo_id: str) -> list[int] | None:
    """Best-effort frame→episode cumulative lengths from the LeRobot dataset."""
    meta = Path("/root/xsx/data") / dataset_repo_id / "meta" / "episodes.jsonl"
    if not meta.is_file():
        return None
    lengths = []
    with open(meta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            lengths.append(int(entry.get("length", 0)))
    return lengths or None


@torch.no_grad()
def z_rl_sanity_report(encoder, dataset, device, *, n_samples, batch_size, use_amp, amp_dtype,
                       episode_lengths=None):
    """PLAN.md §八 P2: z_rl must not collapse — report effective rank, per-dim
    std and (best-effort) cross-episode variance ratio."""
    n = min(n_samples, len(dataset))
    idx = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)
    encoder.eval()
    zs = []
    for i in range(0, n, batch_size):
        items = [dataset[int(j)] for j in idx[i:i + batch_size]]
        prefix = torch.stack([item[0] for item in items]).to(device).float()
        mask = torch.stack([item[1] for item in items]).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z = encoder(prefix, mask=mask)
        zs.append(z.float().cpu())
    z = torch.cat(zs)                       # (n, d)
    zc = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc)           # singular values
    p = sv / sv.sum().clamp_min(1e-12)
    erank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    std = zc.std(dim=0)
    report = {
        "n_samples": int(n),
        "z_dim": int(z.shape[1]),
        "mean_norm": float(z.norm(dim=1).mean()),
        "per_dim_std_min": float(std.min()),
        "per_dim_std_median": float(std.median()),
        "per_dim_std_max": float(std.max()),
        "effective_rank": erank,
    }
    if episode_lengths:
        # Map global frame index -> episode id (frames are stored in dataset order).
        bounds = np.cumsum([0] + episode_lengths)
        ep_of = np.searchsorted(bounds, idx, side="right") - 1
        ep_ids, counts = np.unique(ep_of, return_counts=True)
        if len(ep_ids) > 1:
            total_var = zc.var(dim=0)
            between = torch.zeros(z.shape[1])
            for ep, cnt in zip(ep_ids.tolist(), counts.tolist()):
                mask_ep = torch.as_tensor(ep_of == ep)
                between += cnt * (z[mask_ep].mean(dim=0) - z.mean(dim=0)).pow(2)
            between /= len(idx)
            report["episodes"] = int(len(ep_ids))
            report["cross_episode_var_ratio_median"] = float((between / total_var.clamp_min(1e-12)).median())
    encoder.train()
    return report


def main():
    args = parse_args()
    if args.wait_gpu_free_mib > 0 and args.device.startswith("cuda"):
        chosen = pick_and_wait_gpu(args.wait_gpu_free_mib)
        logger.info("Using GPU %d", chosen)
    device = validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        if device.index is None:
            # CUDA_VISIBLE_DEVICES (set by the GPU gate) maps in-process index 0
            # to the chosen card; torch.cuda.set_device needs an explicit index.
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

    # ── RLT encoder + decoder (rlt package, paper_v1, 2048-dim) ─────────
    rlt_config = RLTConfig(mode="rlt_training")
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info("RLT Encoder: %s params, Decoder: %s params", f"{enc_params:,}", f"{dec_params:,}")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    def lr_schedule(update_index):
        # LambdaLR evaluates index 0 at construction. Keep the first optimizer
        # update non-zero, then apply warmup followed by cosine decay.
        update_number = update_index + 1
        if args.warmup_steps > 0 and update_number <= args.warmup_steps:
            return update_number / args.warmup_steps
        cosine_updates = max(1, args.steps - args.warmup_steps)
        progress = min(1.0, max(0.0, (update_number - args.warmup_steps - 1) / cosine_updates))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    effective_args = vars(args).copy()
    effective_args["device"] = str(device)

    from dataclasses import asdict

    def save_checkpoint(step, loss_val, scheduler_=None, scaler=None, loss_history=None, best_loss=None):
        ckpt = {
            "step": step,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(rlt_config),
            "rlt_architecture": rlt_config.rlt_architecture,
            "image_only": True,
            "loss": loss_val,
            "effective_args": effective_args,
            "schedule_config": {
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "total_steps": args.steps,
            },
            "rng_state": capture_rng_state(),
            "resume_exact": False,
            "resume_note": "DataLoader sampler position is not checkpointed; resume is not exact.",
        }
        if scheduler_ is not None:
            ckpt["scheduler_state_dict"] = scheduler_.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if best_loss is not None:
            ckpt["best_loss"] = best_loss
        if loss_history is not None:
            ckpt["loss_history"] = loss_history

        improved = best_loss is not None and loss_val < best_loss
        if improved:
            best_loss = loss_val
        if best_loss is not None:
            ckpt["best_loss"] = best_loss

        ckpt_path = output_dir / f"checkpoint_step{step}.pt"
        atomic_torch_save(ckpt, ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)
        if improved:
            atomic_torch_save(ckpt, output_dir / "best_checkpoint.pt")
            logger.info("New best loss: %.4f", best_loss)
        return best_loss

    # ── Data: precomputed mmap cache (schema v2) ──────────────────────────
    shard_paths = [Path(p.strip()) for p in str(args.precomputed_path).split(",") if p.strip()]
    if not shard_paths:
        raise ValueError("--precomputed_path is empty")
    dataset, seq_len, vlm_dim = load_cache(shard_paths, rlt_config, decoder)
    repo_id = json.loads((shard_paths[0] / "meta.json").read_text()).get("dataset_repo_id", "")
    ep_lengths = episode_boundaries(repo_id) if repo_id else None

    micro_size = args.batch_size // args.grad_accum
    dataloader = DataLoader(
        dataset, batch_size=micro_size, shuffle=True,
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = iter(dataloader)

    def get_batch():
        nonlocal data_iter
        try:
            prefix_out, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            prefix_out, mask = next(data_iter)
        return (
            prefix_out.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
            mask.to(device=device, dtype=torch.bool, non_blocking=device.type == "cuda"),
        )

    # ── Mixed precision (optional) ───────────────────────────────────────
    use_amp = args.use_amp
    amp_dtype = None
    use_scaler = False
    if use_amp:
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.bfloat16
        use_scaler = amp_dtype == torch.float16
        logger.info("Using AMP on %s (%s, GradScaler=%s)", device.type, amp_dtype, use_scaler)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    if args.grad_accum > 1:
        logger.info("Gradient accumulation: %d micro-batches of %d (effective batch %d)",
                    args.grad_accum, micro_size, args.batch_size)
        if use_scaler:
            raise ValueError("--grad_accum > 1 is only supported with bf16 AMP (no GradScaler)")

    # ── Resume (optional) ────────────────────────────────────────────────
    start_step = 1
    loss_history: list[float] = []
    best_loss = float("inf")
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        if ckpt.get("rlt_architecture") != rlt_config.rlt_architecture or not ckpt.get("image_only", False):
            raise ValueError("Checkpoint is not a paper_v1 image-only Stage 1 checkpoint.")
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        saved_schedule = ckpt.get("schedule_config")
        if saved_schedule is None:
            legacy_args = ckpt.get("effective_args", {})
            if all(key in legacy_args for key in ("lr", "warmup_steps", "steps")):
                saved_schedule = {"lr": legacy_args["lr"], "warmup_steps": legacy_args["warmup_steps"],
                                  "total_steps": legacy_args["steps"]}
        requested_schedule = {"lr": args.lr, "warmup_steps": args.warmup_steps, "total_steps": args.steps}
        if saved_schedule is None:
            logger.warning("Legacy checkpoint has no schedule metadata; assuming the schedule matches.")
        elif saved_schedule != requested_schedule:
            if not args.allow_schedule_change:
                raise ValueError(
                    "Cannot resume with a different LR schedule. "
                    f"Checkpoint={saved_schedule}, requested={requested_schedule}. "
                    "Pass --allow_schedule_change."
                )
            logger.warning("LR schedule CHANGED with --allow_schedule_change (warm restart).")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        schedule_matches = saved_schedule == requested_schedule
        if "scheduler_state_dict" in ckpt and schedule_matches:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif start_step > 1 and schedule_matches:
            raise ValueError("Resume checkpoint is missing scheduler_state_dict")
        if "scaler_state_dict" in ckpt and use_scaler:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "loss_history" in ckpt:
            loss_history = ckpt["loss_history"]
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
        logger.info("Resumed at step %d, best_loss=%.4f", start_step, best_loss)

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("Starting training for %d steps (from step %d)...", args.steps, start_step)
    start_time = time.time()

    encoder.train()
    decoder.train()

    updates_this_process = 0
    for step in range(start_step, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        # One optimizer update = grad_accum micro-batches. The cache masks are
        # all-True (512 valid tokens every frame), so averaging the per-batch
        # masked-mean losses is exactly the full-batch masked mean.
        loss_sum = 0.0
        valid_tokens = 0
        for _ in range(args.grad_accum):
            prefix_out, prefix_mask = get_batch()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _ = rlt_reconstruction_loss(encoder, decoder, prefix_out, mask=prefix_mask)
            if args.grad_accum > 1:
                (loss / args.grad_accum).backward()
            else:
                scaler.scale(loss).backward()
            loss_sum += float(loss.detach()) / args.grad_accum
            valid_tokens += int(prefix_mask.sum().item())

        if args.grad_accum > 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not math.isfinite(loss_sum) or not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite Stage 1 update at step {step}: loss={loss_sum}, grad_norm={grad_norm}"
            )
        if args.dry_run:
            logger.info("Dry run passed: loss=%.6f, grad_norm=%.6f, valid_tokens=%d",
                        loss_sum, float(grad_norm), valid_tokens)
            return
        if args.grad_accum > 1:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        updates_this_process += 1

        loss_val = loss_sum
        loss_history.append(loss_val)

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            updates_per_sec = updates_this_process / max(elapsed, 1e-9)
            logger.info("Step %d/%d | Loss: %.4f | LR: %.2e | %.1f updates/s",
                        step, args.steps, loss_val, scheduler.get_last_lr()[0], updates_per_sec)

        if step % args.save_every == 0 or step == args.steps:
            best_loss = save_checkpoint(
                step, loss_val,
                scheduler_=scheduler, scaler=scaler if use_scaler else None,
                loss_history=loss_history, best_loss=best_loss,
            )

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    # ── Post-training z_rl sanity (PLAN.md §八 P2) ───────────────────────
    try:
        report = z_rl_sanity_report(
            encoder, dataset, device,
            n_samples=args.sanity_samples, batch_size=micro_size,
            use_amp=use_amp, amp_dtype=amp_dtype or torch.bfloat16,
            episode_lengths=ep_lengths,
        )
        with open(output_dir / "z_rl_sanity.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("z_rl sanity: %s", report)
        if report["effective_rank"] < 0.05 * report["z_dim"]:  # gross collapse tripwire
            logger.warning("z_rl effective rank is very low — representation may be collapsed!")
    except Exception as exc:  # noqa: BLE001
        logger.warning("z_rl sanity report failed (non-fatal): %s", exc)

    total_time = time.time() - start_time
    logger.info("Training complete! %d steps in %.1fs (%.1fmin)", args.steps, total_time, total_time / 60)
    logger.info("Best loss: %.4f", best_loss)
    logger.info("Checkpoints saved to: %s", output_dir)


if __name__ == "__main__":
    main()
