#!/usr/bin/env python3
"""serve_rlt: frozen π0.5 policy server + RL Token z_rl (Phase 3).

Runs on the **4090 inference server** (same LAN as the execution machine).
Extends the standard openpi serving stack — zero openpi source changes — so
every response carries, besides the usual absolute-joint action chunk, the
Stage-1 RLT z_rl vector that Stage 2 consumes:

    request (msgpack over websocket, openpi-client compatible):
        observation/state        (≥7,) float — absolute joint positions
        observation/image        (H,W,3) uint8 — front camera
        observation/wrist_image  (H,W,3) uint8 — wrist camera
        prompt                   str (optional; same-origin default injected)
    response:
        actions  (30, 7) float — absolute joint targets (unchanged stack)
        z_rl     (2048,) float32 — frozen Stage-1 encoder on current obs
        state    pass-through + policy_timing{infer_ms, z_rl_ms} + server_timing

z_rl path (same distribution as data/prefix_v1 → data/stage1_v1):
    input transforms (== create_trained_policy stack)
    → jitted prefix-only π0.5 forward (external, openpi untouched)
    → first 512 image-token rows (base+wrist; masked right_wrist dropped)
    → fp16 round-trip (Stage-1 mmap storage quantization)
    → frozen Stage-1 RLTokenEncoder (torch) → z_rl.

Same-origin guard (PLAN.md §二.5 四件套): stage-1 checkpoint, prefix_v1 meta
and this server's base checkpoint/config/prompt must agree; mismatches abort
unless --allow_origin_mismatch.

Startup warmup runs one random-example infer (JIT compile + end-to-end path
check) so the first robot request never pays compile time.

Usage (4090 server; see README §4090):
  /path/to/openpi/.venv/bin/python scripts/serve_rlt.py \
      --checkpoint-dir /path/to/checkpoints/pi05_panthera_p3/panthera_p3/2000 \
      --stage1-checkpoint /path/to/data/stage1_v1/best_checkpoint.pt \
      --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

logger = logging.getLogger(__name__)


def _early_env() -> None:
    """Set GPU env BEFORE any jax/torch import.

    Module-level import of openpi.policies.policy pulls in jax; CUDA device
    pinning and the no-preallocate flag must land first. Both serve_rlt and
    verify_serve_consistency (which imports this module) expose --gpu-index,
    so a light sys.argv scan covers both entry points; programmatic callers
    can also set the env vars themselves before importing.
    """
    if "--gpu-index" in sys.argv:
        try:
            i = sys.argv.index("--gpu-index")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(int(sys.argv[i + 1])))
        except (IndexError, ValueError):
            pass
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_early_env()

# Heavy imports (jax/torch/openpi) — AFTER _early_env().
import openpi.policies.policy as _policy  # noqa: E402

import openpi_prefix  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="pi05_panthera_p3")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Frozen π0.5 checkpoint dir (…/panthera_p3/2000).")
    parser.add_argument("--stage1-checkpoint", required=True,
                        help="Stage-1 RLT checkpoint (best_checkpoint.pt).")
    parser.add_argument("--prefix-meta", default=str(REPO_ROOT / "data/prefix_v1/meta.json"),
                        help="meta.json of the embeddings the stage-1 ckpt was trained on "
                             "(same-origin guard; missing file → warning only).")
    parser.add_argument("--task-prompt", default=None,
                        help="Default: DEFAULT_TASK_PROMPT from openpi_prefix (fixed by 四件套).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-device", default="cuda",
                        help="Device for the torch encoder ('cuda' → resolved to an explicit "
                             "index, 'cpu' for debugging).")
    parser.add_argument("--no-fp16-roundtrip", action="store_true",
                        help="Skip the fp16 round-trip (NOT recommended: stage1 was trained on "
                             "fp16-quantized embeddings).")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--allow-origin-mismatch", action="store_true")
    parser.add_argument("--record", default="", help="Optional dir to record every infer (debug).")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Pin CUDA device BEFORE jax/torch init (sets CUDA_VISIBLE_DEVICES).")
    return parser.parse_args()


def ckpt_tail(path: str | Path, n: int = 3) -> str:
    """Layout-independent checkpoint identity: last n path components
    (…/pi05_panthera_p3/panthera_p3/2000 — comparable across machines)."""
    parts = Path(path).expanduser().resolve().parts
    return "/".join(parts[-n:]) if len(parts) >= n else str(path)


def load_stage1_encoder(stage1_path: Path, torch_device):
    """Load + validate the frozen Stage-1 encoder (paper_v1, image-only)."""
    import torch

    from rlt.networks import RLTConfig, RLTokenEncoder

    ckpt = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if ckpt.get("rlt_architecture") != "paper_v1":
        raise ValueError(f"stage1 ckpt architecture {ckpt.get('rlt_architecture')!r} != 'paper_v1'")
    if not ckpt.get("image_only", False):
        raise ValueError("stage1 ckpt is not image-only")
    config = RLTConfig(mode="online_rl")  # dims only; mode never switches heads here
    if int(ckpt["config"]["vlm_hidden_dim"]) != config.vlm_hidden_dim:
        raise ValueError("stage1 ckpt vlm_hidden_dim mismatch vs rlt package")
    encoder = RLTokenEncoder(config)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(torch_device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, ckpt


def check_same_origin(args, checkpoint_dir: Path, stage1_ckpt: dict, task_prompt: str) -> list[str]:
    """Machine-layout-independent 四件套 checks. Returns human-readable problems."""
    import numpy as _np  # noqa: F401  (marker: no heavy deps needed below)

    problems: list[str] = []
    meta_path = Path(args.prefix_meta).expanduser()
    if not meta_path.is_file():
        logger.warning("prefix meta not found (%s); same-origin chain not machine-checked", meta_path)
        return problems
    meta = json.loads(meta_path.read_text())

    if meta.get("config_name") != args.config_name:
        problems.append(f"config_name: meta={meta.get('config_name')!r} vs serve={args.config_name!r}")
    if meta.get("task_prompt") != task_prompt:
        problems.append(f"task_prompt: meta={meta.get('task_prompt')!r} vs serve={task_prompt!r}")
    if ckpt_tail(meta.get("checkpoint_dir", "")) != ckpt_tail(checkpoint_dir):
        problems.append(
            f"base checkpoint: meta={ckpt_tail(meta.get('checkpoint_dir', ''))!r} vs "
            f"serve={ckpt_tail(checkpoint_dir)!r}"
        )
    for key, expected in (("seq_len", 512), ("vlm_hidden_dim", 2048), ("complete", True),
                          ("image_only", True)):
        if meta.get(key) != expected:
            problems.append(f"meta[{key!r}]={meta.get(key)!r} != {expected!r}")
    # stage1 ckpt ← embeddings dir linkage (by dir name; layouts differ per machine)
    trained_on = str(stage1_ckpt.get("effective_args", {}).get("precomputed_path", ""))
    if trained_on and Path(trained_on).name != meta_path.parent.name:
        problems.append(
            f"stage1 was trained on embeddings {trained_on!r} but prefix meta is "
            f"{meta_path.parent.name!r}"
        )
    return problems


def build_rltpolicy(args):
    """Construct the RLTPolicy. Heavy: imports jax/torch/openpi and loads both
    models. Reused by verify_serve_consistency.py so the verified path IS the
    served path. Returns (policy, metadata)."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    import openpi.models.model as _model
    from openpi.training import config as _config

    from rlt.networks import RLTConfig  # noqa: F401  (config sanity inside loader)

    if args.gpu_index is not None:
        # Normally already set by _early_env(); keep for programmatic callers.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))

    task_prompt = args.task_prompt or openpi_prefix.DEFAULT_TASK_PROMPT
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint_dir}/params")
    stage1_path = Path(args.stage1_checkpoint).expanduser()
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_path}")

    # ── Stage-1 encoder + same-origin guard (cheap, fail fast) ───────────
    torch_device = torch.device(args.torch_device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("cuda requested for torch but unavailable; falling back to cpu")
            torch_device = torch.device("cpu")
        else:
            torch_device = torch.device("cuda", torch.cuda.current_device())  # explicit index
    logger.info("torch encoder device: %s", torch_device)
    encoder, stage1_ckpt = load_stage1_encoder(stage1_path, torch_device)

    problems = check_same_origin(args, checkpoint_dir, stage1_ckpt, task_prompt)
    if problems:
        for p in problems:
            logger.error("origin mismatch: %s", p)
        if not args.allow_origin_mismatch:
            raise SystemExit("Same-origin check failed (四件套). Fix the mismatch or pass "
                             "--allow_origin_mismatch.")
        logger.warning("CONTINUING despite origin mismatch (--allow-origin-mismatch)")

    # ── Frozen π0.5 + transform stack (== create_trained_policy) ─────────
    train_config = _config.get_config(args.config_name)
    logger.info("Loading frozen model from %s (bf16)...", checkpoint_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    graphdef, state = nnx.split(model)
    prefix_forward = openpi_prefix.make_prefix_forward(graphdef)
    input_transform, output_transform, data_config = openpi_prefix.build_policy_stack(
        train_config, checkpoint_dir, task_prompt,
    )
    logger.info("Input stack ready | asset_id=%s | prompt=%r", data_config.asset_id, task_prompt)

    metadata = {
        **(train_config.policy_metadata or {}),
        "server": "serve_rlt",
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "stage1_checkpoint": str(stage1_path.resolve()),
        "stage1_step": int(stage1_ckpt.get("step", -1)),
        "stage1_best_loss": float(stage1_ckpt.get("best_loss", float("nan"))),
        "task_prompt": task_prompt,
        "fp16_roundtrip": not args.no_fp16_roundtrip,
        "action_shape": [train_config.model.action_horizon, 7],
        "z_rl_dim": 2048,
    }

    policy = RLTPolicy(
        model,
        prefix_forward=prefix_forward,
        prefix_state=state,
        encoder=encoder,
        torch_device=torch_device,
        fp16_roundtrip=not args.no_fp16_roundtrip,
        transforms=[input_transform],
        output_transforms=[output_transform],
        metadata=metadata,
    )
    return policy, metadata


class RLTPolicy(_policy.Policy):
    """Policy + z_rl: subclasses the standard Policy so actions flow through
    the exact create_trained_policy path, while the prefix embedding rides
    along on the SAME transformed observation (single transform pass)."""

    def __init__(self, model, *, prefix_forward, prefix_state, encoder, torch_device,
                 fp16_roundtrip=True, **policy_kwargs):
        super().__init__(model, **policy_kwargs)
        self._prefix_forward = prefix_forward
        self._prefix_state = prefix_state
        self._encoder = encoder
        self._torch_device = torch_device
        self._fp16_roundtrip = fp16_roundtrip

    def infer(self, obs: dict, *, noise=None) -> dict:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch

        import openpi_prefix
        from openpi.models import model as _model

        # ── identical flow to Policy.infer (JAX branch) ──────────────────
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)

        # ── z_rl hook: prefix-only forward → image block → torch encoder ─
        z_t0 = time.monotonic()
        prefix_out, prefix_mask = self._prefix_forward(self._prefix_state, observation)
        img, img_mask = openpi_prefix.image_block(prefix_out, prefix_mask)
        img = openpi_prefix.to_stage1_dtype(img, self._fp16_roundtrip)
        img_np = np.asarray(jax.device_get(img))                    # (1, 512, 2048) fp32
        mask_np = np.asarray(jax.device_get(img_mask))              # (1, 512) bool
        with torch.inference_mode():
            z = self._encoder(
                torch.from_numpy(img_np).to(self._torch_device),
                mask=torch.from_numpy(mask_np).to(self._torch_device),
            )
        z_rl = z[0].detach().float().cpu().numpy().astype(np.float32)
        z_rl_ms = (time.monotonic() - z_t0) * 1000.0

        # ── actions: untouched standard path ─────────────────────────────
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise
        a_t0 = time.monotonic()
        actions = self._sample_actions(rng, observation, **sample_kwargs)
        actions_ms = (time.monotonic() - a_t0) * 1000.0

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["z_rl"] = z_rl
        outputs["policy_timing"] = {"infer_ms": actions_ms, "z_rl_ms": z_rl_ms}
        return outputs


def warmup(policy, metadata: dict) -> None:
    """One random-example infer: JIT-compiles both jitted paths and proves the
    end-to-end request/response contract before any robot connects."""
    import numpy as np

    from openpi.policies.panthera_policy import make_panthera_example

    logger.info("Warmup infer (JIT compile + contract check)...")
    t0 = time.monotonic()
    out = policy.infer(make_panthera_example())
    logger.info("Warmup done in %.1fs", time.monotonic() - t0)

    actions, z_rl = out["actions"], out["z_rl"]
    expected_shape = tuple(metadata["action_shape"])
    if actions.shape != expected_shape:
        raise ValueError(f"warmup actions shape {actions.shape} != {expected_shape}")
    if z_rl.shape != (metadata["z_rl_dim"],):
        raise ValueError(f"warmup z_rl shape {z_rl.shape} != ({metadata['z_rl_dim']},)")
    if not np.isfinite(actions).all() or not np.isfinite(z_rl).all():
        raise ValueError("warmup produced non-finite values")
    logger.info("Warmup OK | actions %s in [%.3f, %.3f] | z_rl(2048) norm=%.2f | %s",
                actions.shape, float(actions.min()), float(actions.max()),
                float(np.linalg.norm(z_rl)), out["policy_timing"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    policy, metadata = build_rltpolicy(args)

    if args.record:
        from openpi.policies.policy import PolicyRecorder

        policy = PolicyRecorder(policy, args.record)

    if not args.no_warmup:
        warmup(policy, metadata)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "?"
    logger.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    logger.info("metadata: %s", json.dumps({k: v for k, v in metadata.items() if k != "steps"},
                                           default=str))

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
