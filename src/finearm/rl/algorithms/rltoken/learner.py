"""Stage-2 learner core for RL-Token (Panthera-HT port).

Moved verbatim from ``RL/rlt_panthera/rlt/learner.py`` in Phase 5a (that
path is now a shim); the core/io + core/rng extraction lands in Phase 5b.

Ported from RL-Token-Pi05-open ``scripts/train_rlt_stage2_pi05.py`` (the
``perform_training_update`` closure, TD3 helpers and the Stage-2 checkpoint
schema). The update math is byte-equivalent to the reference; the closure was
extracted into a module-level function with explicit arguments so it can be
driven by the online training loop (Phase 4) and unit-tested (Phase 0).

Highlights (paper Algorithm 1):
- critic update every call, actor update every ``critic_per_actor`` calls
- TD3 target from the TARGET actor (fixed-std sampling) and TARGET twin critic
  q_min, discounted chunk return + bootstrapping, TD target clamp [-c, c]
- source-aware BC: HUMAN/MIXED rows regress to executed actions, others to the
  VLA reference chunk; reference dropout on the actor input
- polyak update of actor/critic targets after each actor update
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from finearm.rl.replay import TransitionSource

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TD3-style helpers
# ─────────────────────────────────────────────────────────────────────────────


def sample_fixed_std_action(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Sample a training-only action chunk from the actor's fixed-std Gaussian."""
    return mean + std * torch.randn_like(mean)


def polyak_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.mul_(1 - tau).add_(parameter.data, alpha=tau)


def discounted_chunk_return(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    discount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discounted rewards and bootstrap discount for executed action rows."""
    valid_mask = valid_mask.to(dtype=rewards.dtype)
    steps = torch.arange(rewards.shape[-1], device=rewards.device, dtype=rewards.dtype)
    weights = torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype).pow(steps)
    returns = (rewards * valid_mask * weights).sum(dim=-1, keepdim=True)
    executed_steps = valid_mask.sum(dim=-1, keepdim=True)
    bootstrap_discount = torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype).pow(executed_steps)
    return returns, bootstrap_discount


@contextmanager
def frozen_parameters(module: torch.nn.Module):
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(module.parameters(), states):
            parameter.requires_grad_(state)


# ─────────────────────────────────────────────────────────────────────────────
# One learner update (critic every call; actor every critic_per_actor calls)
# ─────────────────────────────────────────────────────────────────────────────


def perform_training_update(
    *,
    actor: torch.nn.Module,
    actor_target: torch.nn.Module,
    critic: torch.nn.Module,
    critic_target: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    total_updates: int,
    discount: float = 0.99,
    target_tau: float = 0.005,
    warmup_bc_weight: float = 10.0,
    warmup_q_weight: float = 0.0,
    online_bc_weight: float = 10.0,
    online_q_weight: float = 1.0,
    delta_weight: float = 0.0,
    critic_per_actor: int = 2,
    reference_dropout: float = 0.5,
    td_target_clamp: float = 2.0,
    critic_grad_max_norm: float = 10.0,
    learner_warmup_updates: int = 0,
    unnormalize_action=None,
    device: torch.device | str | None = None,
) -> dict[str, float | bool]:
    """Run one critic update and (every ``critic_per_actor`` updates) one actor update.

    ``total_updates`` is the caller's persistent counter (number of updates
    performed so far); the caller increments it after each call. Weight
    scheduling switches from ``warmup_*`` to ``online_*`` at
    ``learner_warmup_updates``.

    ``unnormalize_action`` (callable Tensor->Tensor, absolute joint space) is
    only required when ``delta_weight > 0`` (smoothness term in absolute
    space). The reference implementation always computes the delta penalty and
    multiplies it by 0; computing it lazily is mathematically identical for
    weight 0 and keeps the function usable without normalization stats.

    Returns a flat metrics dict (floats, plus ``actor_updated`` flag).
    """
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    q_weight = (
        warmup_q_weight if total_updates < learner_warmup_updates
        else online_q_weight
    )
    bc_weight = (
        warmup_bc_weight if total_updates < learner_warmup_updates
        else online_bc_weight
    )
    # Populated only on actor-update steps; None elsewhere.
    actor_q: torch.Tensor | None = None
    bc_penalty: torch.Tensor | None = None
    delta_penalty: torch.Tensor | None = None

    # ── Critic update ───────────────────────────────────────────
    with torch.no_grad():
        next_action_mean, next_action_std = actor_target(
            batch["next_z_rl"], batch["next_proprio"],
            batch["next_ref_chunk"],
        )
        next_action = sample_fixed_std_action(next_action_mean, next_action_std)
        target_q = critic_target.q_min(
            batch["next_z_rl"], batch["next_proprio"], next_action
        )
        chunk_return, bootstrap_discount = discounted_chunk_return(
            batch["rewards"], batch["valid_mask"], discount
        )
        td_target = (
            chunk_return
            + bootstrap_discount * (1.0 - batch["done"]) * target_q
        )
        # Stability guardrail: rewards are terminal 0/1, so the true
        # Q range is ~[-1, 1]; clip the TD target so a diverging
        # target critic can never bootstrap Q values into the
        # hundreds (deadly-triad overestimation spiral).
        td_target = td_target.clamp(min=-td_target_clamp, max=td_target_clamp)
    q1, q2 = critic(batch["z_rl"], batch["proprio"], batch["action_chunk"])
    critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
    rew_mask = batch["rewards"].amax(dim=-1) > 0.0

    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    # Stability guardrail: bound critic gradient norm so a spike in
    # TD error cannot destabilize the twin critics in one step.
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=critic_grad_max_norm)
    critic_optimizer.step()

    metrics: dict[str, float | bool] = {
        "critic_loss": float(critic_loss.detach()),
        "q1_mean": float(q1.mean()),
        "q2_mean": float(q2.mean()),
        "q1_max": float(q1.max()),
        "q2_max": float(q2.max()),
        "td_target_mean": float(td_target.mean()),
        "td_target_max": float(td_target.max()),
    }

    # ── Actor update (paper: two critic updates per actor update) ──
    if total_updates % critic_per_actor == 0:
        actor_ref_chunk = batch["ref_chunk"]
        if reference_dropout > 0.0:
            drop_ref = (
                torch.rand(batch["ref_chunk"].shape[0], device=batch["ref_chunk"].device)
                < reference_dropout
            )
            actor_ref_chunk = torch.where(
                drop_ref.view(-1, 1, 1),
                torch.zeros_like(batch["ref_chunk"]),
                batch["ref_chunk"],
            )
        action_mean, action_std = actor(
            batch["z_rl"], batch["proprio"],
            actor_ref_chunk, training=True,
        )
        sampled_action = sample_fixed_std_action(action_mean, action_std)

        with frozen_parameters(critic):
            actor_q = critic.q_min(
                batch["z_rl"].detach(), batch["proprio"], sampled_action
            ).mean()

        source_chunk = batch["source_chunk"]
        human_mask = (
            (source_chunk == int(TransitionSource.HUMAN))
            | (source_chunk == int(TransitionSource.MIXED))
        )
        valid_mask_f = batch["valid_mask"].float()

        bc_target = torch.where(
            human_mask.unsqueeze(-1), batch["action_chunk"], batch["ref_chunk"]
        )
        bc_error = (sampled_action - bc_target).pow(2).mean(dim=-1)
        bc_penalty = (bc_error * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)

        # Smoothness term is kept in code but disabled by default
        # for strict paper alignment (delta_weight = 0).
        if delta_weight != 0.0:
            if unnormalize_action is None:
                raise ValueError("delta_weight > 0 requires unnormalize_action (absolute space)")
            ad = action_mean.shape[-1]
            pred_absolute = unnormalize_action(action_mean[..., :ad])
            target_absolute = unnormalize_action(bc_target[..., :ad])
            pred_delta = pred_absolute[:, 1:] - pred_absolute[:, :-1]
            target_delta = target_absolute[:, 1:] - target_absolute[:, :-1]
            valid_pairs = valid_mask_f[:, 1:] * valid_mask_f[:, :-1]
            delta_error = (pred_delta - target_delta).pow(2).mean(dim=-1)
            delta_penalty = (delta_error * valid_pairs).sum() / valid_pairs.sum().clamp_min(1.0)
        else:
            delta_penalty = torch.zeros((), device=action_mean.device)

        actor_loss = (
            bc_weight * bc_penalty
            - q_weight * actor_q
            + delta_weight * delta_penalty
        )

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()

        polyak_update(actor, actor_target, target_tau)
        polyak_update(critic, critic_target, target_tau)

        metrics.update({
            "actor_updated": True,
            "actor_loss": float(actor_loss.detach()),
            "actor_q": float(actor_q.detach()),
            "bc_penalty": float(bc_penalty.detach()),
            "delta_penalty": float(delta_penalty.detach()),
            "bc_weight": float(bc_weight),
            "q_weight": float(q_weight),
        })
    else:
        metrics["actor_updated"] = False

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 checkpoint schema (v3, paper_full_output_v1 contract)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 3
ACTOR_CONTRACT = "paper_full_output_v1"
RL_CHUNK_LENGTH = 10


def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_stage2_checkpoint(**payload) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "resume_supported": True,
        "actor_contract": ACTOR_CONTRACT,
        "rl_chunk_length": RL_CHUNK_LENGTH,
        **payload,
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    # Byte-tensor RNG states must be on CPU: torch.load(map_location="cuda")
    # would otherwise move them to the GPU and torch.set_rng_state raises
    # "RNG state must be a torch.ByteTensor". RNG exactness is best-effort only.
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


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def prepare_output_dir(path: Path, *, dry_run: bool) -> None:
    """Fail closed on prior contents; dry-run performs no filesystem mutation."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--output_dir is not a directory: {path}")
        contents = list(path.iterdir())
        if contents:
            raise FileExistsError(
                f"--output_dir already contains {len(contents)} item(s): {path}. "
                "Stage 2 resume is unsupported; choose a new empty directory."
            )
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def prepare_resume_checkpoint(path: Path, output_dir: Path, *, dry_run: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"--resume_from checkpoint does not exist: {path}")
    if path.parent.resolve() == output_dir.resolve():
        raise ValueError("--output_dir must be a new directory, not the resume checkpoint directory.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict", "actor_target_state_dict", "critic_state_dict", "critic_target_state_dict",
        "actor_optimizer", "critic_optimizer", "episode", "episode_rewards", "episode_interventions",
        "total_updates", "rng_state", "replay_rng_state", "replay_journal_bytes", "replay_entries",
        "runtime_provenance",
    }
    missing = sorted(required - set(checkpoint))
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or not checkpoint.get("resume_supported")
        or checkpoint.get("actor_contract") != ACTOR_CONTRACT
        or checkpoint.get("rl_chunk_length") != RL_CHUNK_LENGTH
        or missing
    ):
        raise ValueError(
            "Checkpoint is not a complete paper-aligned full-output Stage 2 checkpoint; start a new run. "
            f"Missing: {missing}"
        )
    source_journal = path.parent / "replay_journal.pkl"
    if not source_journal.is_file():
        raise FileNotFoundError(f"Resume checkpoint requires replay journal: {source_journal}")
    journal_bytes = checkpoint["replay_journal_bytes"]
    if not isinstance(journal_bytes, int) or journal_bytes < 0 or source_journal.stat().st_size < journal_bytes:
        raise ValueError("Resume checkpoint replay journal boundary is invalid or source journal is truncated.")
    if not dry_run:
        destination_journal = output_dir / "replay_journal.pkl"
        with source_journal.open("rb") as source, destination_journal.open("wb") as destination:
            remaining = journal_bytes
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("Resume source journal ended before the checkpoint boundary.")
                destination.write(block)
                remaining -= len(block)
    return checkpoint


__all__ = [
    "sample_fixed_std_action",
    "polyak_update",
    "discounted_chunk_return",
    "frozen_parameters",
    "perform_training_update",
    "SCHEMA_VERSION",
    "ACTOR_CONTRACT",
    "RL_CHUNK_LENGTH",
    "canonical_digest",
    "build_stage2_checkpoint",
    "atomic_torch_save",
    "capture_rng_state",
    "restore_rng_state",
    "move_optimizer_state_to_device",
    "prepare_output_dir",
    "prepare_resume_checkpoint",
]
