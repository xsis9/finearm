"""Shared openpi-side helpers for the serve / verify scripts (Phase 3+).

This module deliberately lives in ``scripts/`` (NOT in ``rlt/``): the ``rlt``
package stays pure torch + numpy with zero openpi/JAX imports, so it can be
copied to the execution machine (4060) as-is.

Two pieces are shared by ``serve_rlt.py`` and ``verify_serve_consistency.py``:

* ``build_policy_stack`` — replicates ``policy_config.create_trained_policy``'s
  transform composition line-for-line (same 5-layer input / 4-layer output
  stack), so the serve path is the same distribution as training and as the
  Phase-1 precompute path.
* ``make_prefix_forward`` — the EXTERNAL jitted prefix-only π0.5 forward.
  Identical math to ``precompute_prefix.py::make_forwards._prefix_impl``
  (that script is finished+validated and intentionally not refactored; keep
  the two in sync by hand if one changes).
"""

from __future__ import annotations

import numpy as np

# Same-origin task string (PLAN.md §二.5) — fixed by the four-piece chain.
DEFAULT_TASK_PROMPT = "pick up the bolt and place it on the nut"

# pi0's IMAGE_KEYS order — images enter the prefix in this order.
EXPECTED_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TOKENS_PER_CAMERA = 256  # SigLIP So400m/14 @ 224² → 16×16
N_IMAGE_TOKENS = TOKENS_PER_CAMERA * len(EXPECTED_IMAGE_KEYS)   # 768
N_STORED_TOKENS = TOKENS_PER_CAMERA * 2                          # 512 (base + wrist)

# The image-token block must be [base(256, valid), wrist(256, valid),
# right_wrist(256, masked)] — anything else breaks the [:512] slice that
# Stage 1 was trained on (see data/prefix_v1/meta.json stored_cameras).
EXPECTED_IMAGE_MASK = np.array([True] * N_STORED_TOKENS + [False] * TOKENS_PER_CAMERA)


def build_policy_stack(train_config, checkpoint_dir, task_prompt: str):
    """Return (input_transform, output_transform, data_config) — the SERVING
    composition (what policy_config.create_trained_policy builds for the
    websocket server): NO repack layer, because clients already send openpi
    convention keys (observation/image, observation/wrist_image,
    observation/state, prompt).

    precompute_prefix.py instead reads raw LeRobot items and prepends the
    config's repack_transforms (observation.images.front → observation/image);
    the repack output equals the client payload, so both paths feed
    PantheraInputs the same dicts — same distribution.
    """
    import openpi.transforms as _transforms
    from openpi.training import checkpoints as _checkpoints

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    input_transform = _transforms.compose([
        _transforms.InjectDefaultPrompt(task_prompt),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    output_transform = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    return input_transform, output_transform, data_config


def make_prefix_forward(graphdef):
    """Build the jitted prefix-only forward closing over a graphdef.

    Reproduces the prefix half of ``pi0.py compute_loss`` / ``sample_actions``
    prefill: image tokens cannot attend to the action suffix under the
    block-attention pattern, and prefix ``positions`` are the cumsum over the
    prefix mask — validated against the full compute_loss path in Phase 1
    (fp32 cosine ≈ 1.0, suffix causality exactly 0).
    """

    def _prefix_impl(state, obs):
        import flax.nnx as nnx

        import openpi.models.model as _model
        import openpi.models.pi0 as _pi0
        import jax.numpy as jnp

        model = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
        )
        return prefix_out, prefix_mask

    import jax

    return jax.jit(_prefix_impl)


def image_block(prefix_out, prefix_mask):
    """Slice + hard-validate the image-token block of a prefix forward output.

    Returns (img (b, 512, ·) jnp, mask (b, 512) jnp). Raises if the observed
    mask pattern deviates from EXPECTED_IMAGE_MASK (e.g. a third real camera
    appeared, or key order changed) — Stage-1's [:512] slice is only correct
    for the frozen pattern.
    """
    import jax

    if prefix_out.shape[1] < N_IMAGE_TOKENS:
        raise ValueError(f"Prefix too short for image slice: {prefix_out.shape[1]} < {N_IMAGE_TOKENS}")
    full_image_mask = prefix_mask[:, :N_IMAGE_TOKENS].astype(bool)
    ok = bool(jax.device_get((full_image_mask == EXPECTED_IMAGE_MASK).all()))
    if not ok:
        bad = int(jax.device_get((full_image_mask != EXPECTED_IMAGE_MASK).any(axis=1).sum()))
        raise ValueError(
            f"Unexpected image mask pattern in {bad}/{full_image_mask.shape[0]} frames — "
            f"expected {N_STORED_TOKENS} valid (base+wrist) then {TOKENS_PER_CAMERA} masked "
            f"(right_wrist). The [:512] Stage-1 slice is invalid for this input."
        )
    return prefix_out[:, :N_STORED_TOKENS], prefix_mask[:, :N_STORED_TOKENS]


def to_stage1_dtype(img, fp16_roundtrip: bool = True):
    """Match Stage-1's input quantization.

    The mmap cache stores fp16 (written from the bf16 JAX forward); Stage 1
    read those fp16 values upcast to fp32. The serve path recomputes bf16 in
    memory — round-tripping through fp16 reproduces the stored values up to
    bf16 kernel noise, keeping offline/online z_rl in the same distribution.
    """
    import jax.numpy as jnp

    if fp16_roundtrip:
        return img.astype(jnp.float16).astype(jnp.float32)
    return img.astype(jnp.float32)
