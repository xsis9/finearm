#!/usr/bin/env python3
"""Pre-compute π0.5 image-only post-transformer prefix embeddings for RLT Stage 1.

openpi JAX port of RL-Token-Pi05-open ``scripts/precompute_pi05_embeddings.py``
(schema v2 output, byte-compatible for the Stage-1 fast-mmap trainer).

Pipeline (identical to training/serving distribution — PLAN.md §六.4):
  dataset(LeRobot v3) → repack → InjectDefaultPrompt → PantheraInputs
  → Normalize(ckpt assets norm_stats) → model_transforms(Resize/Tokenize/Pad)
  → Observation → frozen π0.5 prefix forward

The prefix forward is an EXTERNAL jitted function (zero openpi source changes)
that reproduces the prefix half of ``pi0.py compute_loss`` exactly:
image tokens cannot attend to the action suffix in the block-attention
pattern, and prefix ``positions`` are the cumsum over the prefix mask — so
``llm([prefix_tokens, None])`` equals compute_loss's ``prefix_out`` (both at
train=False, i.e. no augmentation — same as serving).

Image-only slice: SigLIP emits 256 tokens per camera; panthera_policy feeds 3
cameras (``right_wrist_0_rgb`` zero-padded with mask=False) → 768 image-token
positions. Only the 512 VALID tokens (base + wrist) are stored: the masked
right_wrist rows have fully-masked attention rows (uniform softmax over
big_neg) and therefore mix in action-suffix values — garbage that the mask
would only screen downstream. Language + discrete-state tokens (which pi05
appends to the prompt) are dropped by slicing to the image block.

Output (schema v2, matches the reference repo):
  prefix_out.mmap  fp16 (N, 512, 2048)   ~61 GB for 29,036 frames
  prefix_mask.mmap bool (N, 512)         (all True for this base)
  meta.json        {schema_version: 2, image_only: true, embedding_stage:
                    post_transformer, language_positions_dropped: true,
                    complete: true, seq_len, vlm_hidden_dim, ...provenance}

Validation (--validate k, default 4): k random frames are re-run through a
full prefix+suffix forward (the compute_loss path with dummy actions) and the
prefix outputs compared — must agree within bf16 tolerance; also asserts every
frame has exactly the expected number of valid image tokens (512).

Usage (queued on train-queue after lingbotvla):
  /root/xsx/openpi/.venv/bin/python scripts/precompute_prefix.py \
      --config-name pi05_panthera_p3 \
      --checkpoint-dir /root/xsx/openpi/checkpoints/pi05_panthera_p3/panthera_p3/2000 \
      --output-dir data/prefix_v1 \
      --batch-size 32 --wait-gpu-free-mib 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# The same-origin chain (PLAN.md §二.5) fixes this task string; it must match
# training/serving. InjectDefaultPrompt only fills missing prompts (the dataset
# already carries the task text), so this is a safety net.
DEFAULT_TASK_PROMPT = "pick up the bolt and place it on the nut"

# pi0's IMAGE_KEYS order — images enter the prefix in this order.
EXPECTED_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TOKENS_PER_CAMERA = 256  # SigLIP So400m/14 @ 224² → 16×16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="pi05_panthera_p3")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Debug: only process the first N frames (still writes a complete meta).")
    parser.add_argument("--validate", type=int, default=4,
                        help="Number of random frames to re-check against the full compute_loss path.")
    parser.add_argument("--wait-gpu-free-mib", type=int, default=0,
                        help="Block until a GPU has at least this many MiB free (picks the freest GPU). "
                             "0 disables waiting. Set BEFORE jax init; enables XLA no-preallocate.")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Pin to a specific GPU index (still subject to --wait-gpu-free-mib).")
    parser.add_argument("--jax-platform", default=None,
                        help="Force JAX platform (e.g. 'cpu' for smoke tests). Sets JAX_PLATFORMS env.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Frame-index modulo sharding (stage-1 samples frames independently). "
                             "Default 1 = full dataset in one output dir.")
    return parser.parse_args()


def pick_and_wait_gpu(min_free_mib: int, prefer_index: int | None) -> int:
    """stdlib-only GPU gate: wait until some GPU has >= min_free_mib free.

    Sets CUDA_VISIBLE_DEVICES (and XLA no-preallocate) BEFORE jax is imported
    anywhere, then returns the chosen index.
    """
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
        if prefer_index is not None:
            free = next((f for i, f in gpus if i == prefer_index), -1)
            if free >= min_free_mib:
                chosen = prefer_index
                break
        else:
            index, free = max(gpus, key=lambda pair: pair[1])
            if free >= min_free_mib:
                chosen = index
                break
        logging.info("waiting for GPU with >= %d MiB free (current free: %s)...", min_free_mib, gpus)
        time.sleep(60)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return chosen


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.jax_platform:
        os.environ["JAX_PLATFORMS"] = args.jax_platform
    if args.wait_gpu_free_mib > 0 or args.gpu_index is not None:
        chosen = pick_and_wait_gpu(max(args.wait_gpu_free_mib, 1), args.gpu_index)
        logging.info("Using GPU %d", chosen)

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index/--shard_count out of range")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # ── Heavy imports AFTER env/CUDA setup ────────────────────────────────
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import openpi.models.model as _model
    import openpi.models.pi0 as _pi0
    import openpi.transforms as _transforms
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    logging.info("jax %s | devices: %s", jax.__version__, jax.devices())

    train_config = _config.get_config(args.config_name)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint_dir}/params")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Input transform stack: identical lines to create_trained_policy ──
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    input_transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(args.task_prompt),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    logging.info(
        "Input stack: repack(%d) + prompt + data(%d) + normalize + model(%d) | asset_id=%s",
        len(data_config.repack_transforms.inputs), len(data_config.data_transforms.inputs),
        len(data_config.model_transforms.inputs), data_config.asset_id,
    )

    # ── Dataset (raw items; transforms run in the main process so the jax
    #    resize/tokenize kernels never run inside DataLoader workers) ─────
    dataset = _data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model,
    )
    indices = list(range(len(dataset)))
    if args.shard_count > 1:
        indices = indices[args.shard_index::args.shard_count]
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,  # mmap rows must follow dataset order
        num_workers=args.num_workers,
        collate_fn=lambda items: items,  # keep per-item dicts; transform in main
    )
    n_total = len(subset)
    if n_total == 0:
        raise ValueError("No frames selected (empty shard / max-frames)")
    logging.info("Dataset %s: %d frames | %d batches of %d", data_config.repo_id, n_total,
                 (n_total + args.batch_size - 1) // args.batch_size, args.batch_size)

    # ── Frozen model + external jitted prefix forward (openpi untouched) ──
    logging.info("Loading frozen model from %s (bf16)...", checkpoint_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    graphdef, state = nnx.split(model)

    def make_forwards(graphdef):
        """Build jitted (prefix_only, full) forwards closing over a graphdef.

        prefix_only reproduces the prefix half of ``pi0.py compute_loss`` (and
        of ``sample_actions``' kv-cache prefill): image tokens cannot attend to
        the action suffix in the block-attention pattern and prefix positions
        are the cumsum over the prefix mask.
        """

        def _prefix_impl(state: nnx.State, obs: _model.Observation):
            model = nnx.merge(graphdef, state)
            obs = _model.preprocess_observation(None, obs, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
            prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
            )
            return prefix_out, prefix_mask

        def _full_impl(state: nnx.State, obs: _model.Observation, actions: _model.Actions):
            model = nnx.merge(graphdef, state)
            obs = _model.preprocess_observation(None, obs, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                obs, actions, jnp.full((obs.state.shape[0],), 0.5),
            )
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, _), _ = model.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, prefix_mask

        return jax.jit(_prefix_impl), jax.jit(_full_impl)

    prefix_forward, full_forward = make_forwards(graphdef)

    # ── Transform helpers ─────────────────────────────────────────────────
    def to_numpy(item: dict) -> dict:
        converted = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            converted[key] = value
        return converted

    def make_obs(items: list[dict]) -> tuple[_model.Observation, dict]:
        transformed = [input_transform(to_numpy(item)) for item in items]
        batch = {}
        def stack_leaf(leafs):
            return np.stack([np.asarray(leaf) for leaf in leafs], axis=0)
        for key in transformed[0]:
            value = transformed[0][key]
            if isinstance(value, dict):
                batch[key] = {subkey: stack_leaf([t[key][subkey] for t in transformed])
                              for subkey in value}
            elif isinstance(value, str):
                batch[key] = value
            else:
                batch[key] = stack_leaf([t[key] for t in transformed])
        batched = jax.tree.map(jnp.asarray, batch)
        prompt_len = int(batched["tokenized_prompt"].shape[1])
        return _model.Observation.from_dict(batched), batched, prompt_len

    # ── mmap writers (schema v2) ──────────────────────────────────────────
    prefix_path = output_dir / "prefix_out.mmap"
    mask_path = output_dir / "prefix_mask.mmap"
    meta_path = output_dir / "meta.json"
    n_image_tokens = TOKENS_PER_CAMERA * len(EXPECTED_IMAGE_KEYS)
    # Only the two real cameras are stored (base_0_rgb + left_wrist_0_rgb =
    # the first 512 image positions, contiguous). The masked right_wrist rows
    # are NOT stored: their attention rows are fully masked (image_mask=False)
    # → softmax over an all-big_neg row is uniform, so they mix in the action
    # suffix's values (harmless downstream — the mask screened them — but
    # storing them wastes 30 GiB and keeps suffix-dependent garbage).
    n_stored_tokens = TOKENS_PER_CAMERA * 2
    stored_cameras = EXPECTED_IMAGE_KEYS[:2]
    expected_mask_pattern = np.array([True] * n_stored_tokens + [False] * TOKENS_PER_CAMERA)
    provenance = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "dataset_repo_id": data_config.repo_id,
        "task_prompt": args.task_prompt,
        "image_keys": list(EXPECTED_IMAGE_KEYS),
        "stored_cameras": list(stored_cameras),
        "dropped_cameras": list(EXPECTED_IMAGE_KEYS[2:]),
        "tokens_per_camera": TOKENS_PER_CAMERA,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "jax_devices": [str(d) for d in jax.devices()],
    }

    prefix_mmap = None
    mask_mmap = None
    seq_len = vlm_dim = None

    def open_mmaps(s: int, d: int) -> None:
        nonlocal prefix_mmap, mask_mmap, seq_len, vlm_dim
        seq_len, vlm_dim = s, d
        logging.info("mmap: (%d, %d, %d) fp16 = %.1f GiB + mask (%d, %d)",
                     n_total, s, d, n_total * s * d * 2 / 2**30, n_total, s)
        prefix_mmap = np.memmap(prefix_path, dtype="float16", mode="w+", shape=(n_total, s, d))
        mask_mmap = np.memmap(mask_path, dtype="bool", mode="w+", shape=(n_total, s))
        _atomic_write_json(meta_path, {
            "schema_version": 2,
            "complete": False,
            "n_samples": n_total,
            "has_mask": True,
            "dtype": "float16",
            "image_only": True,
            "embedding_stage": "post_transformer",
            "language_positions_dropped": True,
            **provenance,
        })

    # ── Main loop ─────────────────────────────────────────────────────────
    t0 = time.time()
    written = 0
    valid_counts: dict[int, int] = {}
    prompt_lens: set[int] = set()
    sample_cache: dict[int, tuple] = {}  # frame index -> (obs, batched) for validation

    for batch_idx, items in enumerate(loader):
        obs, batched, prompt_len = make_obs(items)
        prompt_lens.add(prompt_len)
        if tuple(obs.images.keys()) != EXPECTED_IMAGE_KEYS:
            raise ValueError(f"Unexpected image key order: {list(obs.images.keys())}")

        prefix_out, prefix_mask = prefix_forward(state, obs)
        prefix_out = np.asarray(jax.device_get(prefix_out))  # (b, L, 2048) bf16→fp32 host
        prefix_mask = np.asarray(jax.device_get(prefix_mask))

        if prefix_out.shape[1] < n_image_tokens:
            raise ValueError(f"Prefix too short for image slice: {prefix_out.shape[1]} < {n_image_tokens}")
        # The image-token block must be [base(256, valid), wrist(256, valid),
        # right_wrist(256, masked)] — anything else breaks the [:512] storage slice.
        full_image_mask = prefix_mask[:, :n_image_tokens].astype(bool)
        if not (full_image_mask == expected_mask_pattern).all():
            bad = int((full_image_mask != expected_mask_pattern).any(axis=1).sum())
            raise ValueError(
                f"Batch {batch_idx}: unexpected image mask pattern in {bad}/{full_image_mask.shape[0]} "
                f"frames (expected {expected_mask_pattern.sum()} valid then masked)"
            )
        img_out = prefix_out[:, :n_stored_tokens].astype(np.float16)
        img_mask = full_image_mask[:, :n_stored_tokens]

        if not np.isfinite(img_out.astype(np.float32)).all():
            raise ValueError(f"Batch {batch_idx}: non-finite embeddings (check fp16 overflow)")
        counts = img_mask.sum(axis=1).astype(int)
        for count in counts.tolist():
            valid_counts[count] = valid_counts.get(count, 0) + 1
        if (counts != n_stored_tokens).any():
            raise ValueError(f"Batch {batch_idx}: expected {n_stored_tokens} valid tokens/frame, got {counts.tolist()}")

        if prefix_mmap is None:
            open_mmaps(img_out.shape[1], img_out.shape[2])
        elif img_out.shape[1:] != (seq_len, vlm_dim):
            raise ValueError(f"Batch {batch_idx}: unstable shape {img_out.shape[1:]} != {(seq_len, vlm_dim)}")

        b = img_out.shape[0]
        end = written + b
        if end > n_total:
            raise ValueError(f"Batch {batch_idx}: write overflow {end} > {n_total}")
        prefix_mmap[written:end] = img_out
        mask_mmap[written:end] = img_mask
        prefix_mmap.flush()
        mask_mmap.flush()

        if len(sample_cache) < args.validate:
            for i in range(b):
                if len(sample_cache) >= args.validate:
                    break
                sample_cache[written + i] = (obs_replace_batch(obs, i), tree_slice(batched, i))
        written = end

        if (batch_idx + 1) % 20 == 0 or end == n_total:
            elapsed = time.time() - t0
            rate = written / max(elapsed, 1e-6)
            logging.info("  %d/%d frames | %.1f frames/s | ETA %.0fs", written, n_total, rate,
                         (n_total - written) / max(rate, 1e-6))

    if prefix_mmap is None:
        raise RuntimeError("DataLoader produced no batches")
    if written != n_total:
        raise ValueError(f"Write count mismatch: {written} != {n_total}")
    prefix_mmap.flush()
    mask_mmap.flush()
    del prefix_mmap, mask_mmap

    # ── Validation ────────────────────────────────────────────────────────
    # Claim under test: the stored prefix-only embeddings are the compute_loss
    # prefix_out. The prefix half of the joint forward is mathematically
    # independent of the action suffix (masked keys get exp→exactly 0
    # probabilities; prefix positions are the prefix-mask cumsum), but in bf16
    # the fused kernels reassociate differently across sequence lengths
    # (768 vs ~998), producing up to ~18%-relative noise on Gemma's outlier
    # dims. Therefore:
    #   HARD (semantic, fp32): prefix-only ≡ full(zeros), and the full forward's
    #     prefix is invariant to suffix values.
    #   INFO (bf16): magnitude of the kernel noise prefix-only vs full, and the
    #     stored bf16 vs the fp32 reference ("bf16 容差" per PLAN.md §八 P1).
    if args.validate > 0 and sample_cache:
        import dataclasses as _dc

        rng = np.random.default_rng(0)

        # ── fp32 reference model ─────────────────────────────────────────
        logging.info("Loading fp32 model copy for equivalence check...")
        model_f32 = _dc.replace(train_config.model, dtype="float32").load(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
        )
        graphdef_f32, state_f32 = nnx.split(model_f32)
        prefix_forward_f32, full_forward_f32 = make_forwards(graphdef_f32)

        stored = np.memmap(prefix_path, dtype="float16", mode="r",
                           shape=(n_total, seq_len, vlm_dim))

        for frame_index, (obs_one, batched_one) in sample_cache.items():
            zeros = jnp.zeros_like(batched_one["actions"])
            random_actions = jnp.asarray(rng.normal(size=zeros.shape), dtype=jnp.float32)

            def _f32(array):
                return np.asarray(jax.device_get(array))[0, :n_stored_tokens].astype(np.float64)

            def _cos(a, b):
                denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
                return float(np.dot(a.ravel(), b.ravel()) / max(denom, 1e-12))

            # (1) fp32 semantic equivalence: prefix-only == full compute_loss path.
            a_f32 = _f32(prefix_forward_f32(state_f32, obs_one)[0])
            b_f32 = _f32(full_forward_f32(state_f32, obs_one, zeros)[0])
            max_abs = float(np.abs(a_f32 - b_f32).max())
            cosine = _cos(a_f32, b_f32)
            logging.info("  frame %d fp32 prefix-only vs full: max|a|=%.1f max|Δ|=%.6f cosine=%.9f",
                         frame_index, float(np.abs(a_f32).max()), max_abs, cosine)
            if max_abs > 0.5 or cosine < 0.9999:
                raise ValueError(
                    f"Frame {frame_index} fp32 mismatch vs compute_loss path: "
                    f"max|Δ|={max_abs:.4f} cosine={cosine:.7f}"
                )

            # (2) fp32 suffix causality: prefix must be invariant to suffix values.
            c_f32 = _f32(full_forward_f32(state_f32, obs_one, random_actions)[0])
            leak = float(np.abs(b_f32 - c_f32).max())
            logging.info("  frame %d fp32 suffix-causality: max|Δ|=%.8f (expect 0)", frame_index, leak)
            if leak > 1e-4:
                raise ValueError(
                    f"Frame {frame_index}: fp32 prefix depends on the action suffix (max|Δ|={leak})"
                )

            # (3) bf16 kernel noise (informational).
            a_bf16 = _f32(prefix_forward(state, obs_one)[0])
            b_bf16 = _f32(full_forward(state, obs_one, zeros)[0])
            c_bf16 = _f32(full_forward(state, obs_one, random_actions)[0])
            logging.info("  frame %d bf16 prefix-only vs full: max|Δ|=%.4f cosine=%.7f",
                         frame_index, float(np.abs(a_bf16 - b_bf16).max()), _cos(a_bf16, b_bf16))
            logging.info("  frame %d bf16 suffix sensitivity: max|Δ|=%.4f cosine=%.7f",
                         frame_index, float(np.abs(b_bf16 - c_bf16).max()), _cos(b_bf16, c_bf16))

            # (4) stored fp16 embeddings vs the fp32 reference (informational).
            s = np.asarray(stored[frame_index], dtype=np.float64)
            logging.info("  frame %d stored-fp16 vs fp32: max|Δ|=%.4f cosine=%.7f",
                         frame_index, float(np.abs(s - a_f32).max()), _cos(s, a_f32))

        del stored, model_f32, state_f32, graphdef_f32

    # ── Finalize meta (schema v2 complete) ───────────────────────────────
    _atomic_write_json(meta_path, {
        "schema_version": 2,
        "complete": True,
        "n_samples": n_total,
        "has_mask": True,
        "dtype": "float16",
        "image_only": True,
        "embedding_stage": "post_transformer",
        "language_positions_dropped": True,
        "seq_len": seq_len,
        "vlm_hidden_dim": vlm_dim,
        "tokens_per_camera": TOKENS_PER_CAMERA,
        "expected_valid_tokens": n_stored_tokens,
        "valid_token_counts": {str(k): v for k, v in valid_counts.items()},
        "prompt_lengths": sorted(prompt_lens),
        "source_frame_indices": {"first": indices[0], "last": indices[-1]},
        **provenance,
    })
    total = time.time() - t0
    logging.info("Done: %d frames in %.0fs (%.1f frames/s)", written, total, written / max(total, 1e-6))
    logging.info("  %s (%.1f GiB)", prefix_path, n_total * seq_len * vlm_dim * 2 / 2**30)
    logging.info("  %s", mask_path)
    logging.info("  %s (complete=true, valid tokens: %s)", meta_path, valid_counts)


def obs_replace_batch(obs, i: int):
    """Extract sample i from a batched Observation (for size-1 validation)."""
    import jax.tree_util

    return jax.tree.map(lambda leaf: leaf[i:i + 1], obs)


def tree_slice(tree, i: int):
    import jax

    return jax.tree.map(lambda leaf: leaf[i:i + 1] if hasattr(leaf, "shape") else leaf, tree)


if __name__ == "__main__":
    main()
