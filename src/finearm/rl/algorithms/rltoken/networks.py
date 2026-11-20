"""RL Token networks for the Panthera-HT π0.5 stack (openpi JAX base).

Moved verbatim from ``RL/rlt_panthera/rlt/networks.py`` in Phase 5a
(that path is now a shim).

Ported from RL-Token-Pi05-open
(`lerobot/src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py`, classes
RLTokenEncoder / RLTokenDecoder / RLTChunkActor / RLTTwinCritic) with the
lerobot ``PreTrainedConfig`` machinery replaced by a plain dataclass and the
SO-101 6-dim action space re-parameterized for Panthera-HT:

- state_dim = action_dim = 7 (6 joints + gripper), absolute joint targets
- VLA horizon chunk_size = 30 (openpi p3 checkpoint serves 30-step chunks)
- RL chunk n_action_steps_rl = 10 (locked into the Stage-2 checkpoint
  contract ``paper_full_output_v1``; keep 10 unless restarting training)

The forward math is ported verbatim; only the configuration container,
defaults, and dimension validation bounds changed. The frozen VLA itself is
NOT wrapped here — prefix embeddings are extracted on the JAX/openpi side
(precompute / serve_rlt) and only the (B, L, 2048) image-token tensors plus
the mask reach these modules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (plain dataclass — no lerobot dependency)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RLTConfig:
    """RL Token configuration — Panthera-HT (7-dim absolute joint space).

    Defaults follow the RL Token paper / RL-Token-Pi05-open trained contract
    (``paper_full_output_v1``) except for the dimension changes noted in the
    module docstring. RL hyperparameter defaults mirror the reference
    Stage-2 argparse defaults (BC=10 / Q=1 online, warmup Q=0, τ=0.005,
    UTD=5, batch 256, reference_dropout 0.5).
    """

    # ── RLT encoder-decoder (trained contract: keep dims aligned with the
    #    Stage 1 checkpoints produced by scripts/train_stage1.py) ──────────
    rlt_architecture: str = "paper_v1"
    vlm_hidden_dim: int = 2048      # PaliGemma gemma_2b prefix token dim (openpi π0.5)
    rlt_hidden_dim: int = 2048      # z_rl dimension (match VLA dim, per paper)
    rlt_encoder_layers: int = 4
    rlt_decoder_layers: int = 4
    rlt_num_heads: int = 8
    rlt_dropout: float = 0.1

    # ── Paper full-output actor MLP ──────────────────────────────────────
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2

    # ── Twin critic MLP ──────────────────────────────────────────────────
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    # Fixed exploration std of the Gaussian actor (paper: small fixed std)
    policy_fixed_std: float = 0.05

    # ── Panthera-HT action space (matches the frozen p3 checkpoint) ──────
    chunk_size: int = 30            # VLA horizon: openpi p3 serves (30, 7) chunks
    action_dim: int = 7             # 6 joints + gripper (absolute joint targets)
    state_dim: int = 7              # first 7 of observation.state (positions)
    n_action_steps_rl: int = 10     # RL chunk executed per policy call (=checkpoint contract)
    action_stride: int = 2          # replay-transition subsampling stride (stride-2)
    # Canonical Panthera action feature names (dataset order); declared so
    # export/rollout boundaries can validate/reorder before touching hardware.
    action_feature_names: list[str] | None = field(default=None)
    # Export-contract fields (descriptive round-trip fields, mirroring the
    # reference repo so saved configs load back unchanged).
    actor_critic_style: str = "paper_mlp"
    actor_residual_scale: float = 0.0

    # ── RL hyperparameters (Stage 2 learner; see rlt/learner.py) ─────────
    discount: float = 0.99
    target_tau: float = 0.005
    warmup_bc_weight: float = 10.0
    warmup_q_weight: float = 0.0
    online_bc_weight: float = 10.0
    online_q_weight: float = 1.0
    delta_weight: float = 0.0       # smoothness term kept but disabled (paper alignment)
    critic_per_actor: int = 2       # critic updates per actor update
    utd_ratio: int = 5              # gradient updates per collected transition
    batch_size: int = 256
    reference_dropout: float = 0.5
    td_target_clamp: float = 2.0    # TD target clamp [-c, c] (stability guardrail)
    critic_grad_max_norm: float = 10.0

    # ── Deployment behavior (auto-critical gating; Phase 4/5) ────────────
    deploy_auto_critical: bool = False
    critical_classifier_path: str = ""      # CriticalPhaseDetector .pt checkpoint
    auto_critical_threshold_on: float = 0.75
    auto_critical_threshold_off: float = 0.45
    auto_critical_smooth_steps: int = 8
    auto_critical_min_on_chunks: int = 3
    auto_critical_start_delay_steps: int = 0   # control steps before detection arms
    auto_critical_once: bool = True            # only one auto ON per episode
    auto_critical_override_cooldown: float = 3.0  # s after a manual 'c'
    actor_critical_delay_steps: int = 15       # sustained critical steps before actor takes over
    deploy_vla_only: bool = False              # never let the actor take over

    # ── Training mode ────────────────────────────────────────────────────
    mode: str = "rlt_training"

    def __post_init__(self):
        if self.mode not in ("rlt_training", "online_rl", "inference"):
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: rlt_training, online_rl, inference"
            )
        if self.rlt_architecture != "paper_v1":
            raise ValueError(
                f"Unsupported rlt_architecture '{self.rlt_architecture}'. Expected 'paper_v1'."
            )
        if self.rlt_hidden_dim % self.rlt_num_heads != 0:
            raise ValueError("rlt_hidden_dim must be divisible by rlt_num_heads")
        if self.action_dim <= 0 or self.state_dim <= 0:
            raise ValueError("action_dim and state_dim must be positive")
        if self.n_action_steps_rl < 1:
            raise ValueError("n_action_steps_rl must be at least 1")
        if self.action_stride < 1:
            raise ValueError("action_stride must be at least 1")
        if self.actor_num_layers < 1 or self.critic_num_layers < 1:
            raise ValueError("actor_num_layers and critic_num_layers must be at least 1")
        if self.actor_hidden_dim <= 0 or self.critic_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim and critic_hidden_dim must be positive")
        if self.policy_fixed_std <= 0:
            raise ValueError("policy_fixed_std must be positive")
        if self.deploy_auto_critical:
            if not self.critical_classifier_path:
                raise ValueError(
                    "deploy_auto_critical requires critical_classifier_path "
                    "(a trained CriticalPhaseDetector .pt checkpoint)"
                )
            if not (0.0 <= self.auto_critical_threshold_off < self.auto_critical_threshold_on <= 1.0):
                raise ValueError(
                    "auto_critical thresholds must satisfy 0 <= off < on <= 1 "
                    f"(got off={self.auto_critical_threshold_off}, on={self.auto_critical_threshold_on})"
                )
            if self.auto_critical_smooth_steps < 1 or self.auto_critical_min_on_chunks < 0:
                raise ValueError("auto_critical_smooth_steps/min_on_chunks out of range")
            if self.auto_critical_override_cooldown < 0:
                raise ValueError("auto_critical_override_cooldown must be >= 0")


# ─────────────────────────────────────────────────────────────────────────────
# RLT Encoder: append one learned RL token to the frozen VLA token sequence
# ─────────────────────────────────────────────────────────────────────────────


class RLTokenEncoder(nn.Module):
    """Paper Eq. (1): encode ``[z_1, ..., z_M, e_rl]`` and return position M+1."""

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim
        self.vlm_proj = (
            nn.Linear(config.vlm_hidden_dim, d)
            if config.vlm_hidden_dim != d
            else nn.Identity()
        )
        self.rl_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.rlt_encoder_layers,
            norm=nn.LayerNorm(d),
        )

    def forward(self, vlm_embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return the encoded appended RL token.

        Args:
            vlm_embeddings: ``(B, L, vlm_hidden_dim)`` frozen VLA tokens.
            mask: ``(B, L)`` boolean validity mask. The appended RL token is
                always valid.
        """
        if vlm_embeddings.ndim != 3:
            raise ValueError(
                f"vlm_embeddings must have shape (B, L, D), got {tuple(vlm_embeddings.shape)}"
            )
        batch_size, seq_len, _ = vlm_embeddings.shape
        tokens = self.vlm_proj(vlm_embeddings)
        rl_token = self.rl_token.expand(batch_size, -1, -1)
        augmented = torch.cat([tokens, rl_token], dim=1)

        # mask: True=valid → src_key_padding_mask: True=IGNORE (PyTorch convention)
        key_padding_mask = None
        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
                )
            rl_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([~mask.bool(), rl_valid], dim=1)

        encoded = self.transformer(augmented, src_key_padding_mask=key_padding_mask)
        return encoded[:, -1]


# ─────────────────────────────────────────────────────────────────────────────
# RLT Decoder: teacher-forced causal reconstruction from z_rl
# ─────────────────────────────────────────────────────────────────────────────


class RLTokenDecoder(nn.Module):
    """Paper Eq. (2): predict z_i from ``[z_rl, stopgrad(z_1:i-1)]``."""

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim
        self.target_proj = (
            nn.Linear(config.vlm_hidden_dim, d)
            if config.vlm_hidden_dim != d
            else nn.Identity()
        )
        self.z_proj = nn.Linear(d, d) if config.vlm_hidden_dim != d else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, 2048, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=config.rlt_decoder_layers,
            norm=nn.LayerNorm(d),
        )
        self.pred_head = (
            nn.Linear(d, config.vlm_hidden_dim)
            if d != config.vlm_hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        z_rl: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Teacher-force stopped-gradient targets and reconstruct the full sequence."""
        if targets.ndim != 3:
            raise ValueError(f"targets must have shape (B, L, D), got {tuple(targets.shape)}")
        batch_size, target_len, _ = targets.shape
        if z_rl.shape[0] != batch_size:
            raise ValueError("z_rl and targets must have the same batch size")
        if target_len > self.pos_embed.shape[1]:
            raise ValueError(
                f"target length {target_len} exceeds decoder limit {self.pos_embed.shape[1]}"
            )

        shifted = targets.detach()[:, :-1]
        decoder_inputs = torch.cat(
            [self.z_proj(z_rl).unsqueeze(1), self.target_proj(shifted)],
            dim=1,
        )
        decoder_inputs = decoder_inputs + self.pos_embed[:, :target_len]

        causal_mask = torch.triu(
            torch.ones(target_len, target_len, dtype=torch.bool, device=targets.device),
            diagonal=1,
        )
        # mask: True=valid → src_key_padding_mask: True=IGNORE
        key_padding_mask = None
        if mask is not None:
            if mask.shape != (batch_size, target_len):
                raise ValueError(
                    f"mask must have shape {(batch_size, target_len)}, got {tuple(mask.shape)}"
                )
            # Position 0 contains z_rl (always valid); position i>0 contains target i-1.
            input_invalid = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device),
                 ~mask[:, :-1].bool()],
                dim=1,
            )
            key_padding_mask = input_invalid

        decoded = self.decoder(
            decoder_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return self.pred_head(decoded)


# ─────────────────────────────────────────────────────────────────────────────
# Actor/Critic MLP: paper full-output heads (z_rl + state + ref/action chunk)
# ─────────────────────────────────────────────────────────────────────────────


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    """Plain ReLU MLP: ``[Linear, ReLU] * (num_layers - 1) + Linear(output_dim)``.

    The exact module layout matches the RL-Token-Pi05 Stage-2 trained
    checkpoints (state dict keys ``net.*`` for the actor, ``q1.*``/``q2.*``
    for the critic), so ``load_state_dict(..., strict=True)`` works on them.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class RLTChunkActor(nn.Module):
    """Paper full-output Gaussian chunk actor: (z_rl, proprio, ref_chunk) -> action mean.

    Lightweight MLP matching the RL-Token-Pi05 Stage-2 trained architecture
    (``actor_contract = paper_full_output_v1``): concatenate ``z_rl``,
    proprioception and the flattened reference chunk, then
    ``[Linear, ReLU] * (num_layers - 1) + Linear(chunk * action_dim)``.
    Returns a Gaussian with fixed std. Always full output — no residual mode
    (``is_residual`` stays False, per the paper contract).
    """

    def __init__(
        self,
        config: RLTConfig,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # ``getattr`` with the paper defaults keeps duck-typed configs working
        # (tests construct heads from SimpleNamespace-style instances).
        self.hidden_dim = int(
            hidden_dim if hidden_dim is not None else getattr(config, "actor_hidden_dim", 256)
        )
        self.num_layers = int(
            num_layers if num_layers is not None else getattr(config, "actor_num_layers", 2)
        )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")

        z_dim = int(getattr(config, "rlt_hidden_dim", 2048))
        state_dim = int(getattr(config, "state_dim", 7))
        action_dim = int(getattr(config, "action_dim", 7))
        chunk = int(getattr(config, "n_action_steps_rl", 10))

        self.action_dim = action_dim
        self.chunk = chunk
        self.fixed_std = float(getattr(config, "policy_fixed_std", 0.05))
        self.is_residual = False  # paper contract: full output, never residual

        input_dim = z_dim + state_dim + chunk * action_dim
        output_dim = chunk * action_dim
        self.net = _build_mlp(input_dim, self.hidden_dim, self.num_layers, output_dim)

    def _flatten_chunk(self, chunk: Tensor, batch: int) -> Tensor:
        """Flatten a (B, chunk, action_dim) / (B, chunk*action_dim) / (chunk, action_dim)
        tensor to (B, chunk*action_dim), validating every dimension so an MLP input
        mismatch fails with a clear message instead of an opaque matmul error."""
        flat = self.chunk * self.action_dim
        if chunk.ndim == 3:
            flat_chunk = chunk.reshape(chunk.shape[0], -1)
        elif chunk.ndim == 2 and chunk.shape[1] == flat:
            flat_chunk = chunk
        elif chunk.ndim == 2:
            # (chunk, action_dim) without a batch dimension: treat as one sample.
            flat_chunk = chunk.reshape(1, -1)
        else:
            raise ValueError(
                f"ref_chunk must have rank 2 or 3, got rank {chunk.ndim} with shape {tuple(chunk.shape)}"
            )
        if flat_chunk.shape[-1] != flat:
            raise ValueError(
                f"ref_chunk last dim {flat_chunk.shape[-1]} != expected {flat} "
                f"(chunk={self.chunk}, action_dim={self.action_dim})"
            )
        if flat_chunk.shape[0] != batch:
            raise ValueError(
                f"ref_chunk batch {flat_chunk.shape[0]} != input batch {batch}; "
                "add a batch dimension to ref_chunk"
            )
        return flat_chunk

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        ref_flat = self._flatten_chunk(ref_chunk, z_rl.shape[0])
        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mean_flat = self.net(x)
        mean = mean_flat.reshape(-1, self.chunk, self.action_dim)
        std = torch.full_like(mean, self.fixed_std)
        return mean, std


class RLTTwinCritic(nn.Module):
    """Twin critic for full chunk actions: (z_rl, proprio, action_chunk) -> Q1, Q2.

    Same lightweight MLP family as the actor (paper-aligned, keys ``q1.*`` /
    ``q2.*`` matching the RL-Token-Pi05 Stage-2 trained checkpoints).
    """

    def __init__(
        self,
        config: RLTConfig,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = int(
            hidden_dim if hidden_dim is not None else getattr(config, "critic_hidden_dim", 256)
        )
        self.num_layers = int(
            num_layers if num_layers is not None else getattr(config, "critic_num_layers", 2)
        )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")

        z_dim = int(getattr(config, "rlt_hidden_dim", 2048))
        state_dim = int(getattr(config, "state_dim", 7))
        action_dim = int(getattr(config, "action_dim", 7))
        chunk = int(getattr(config, "n_action_steps_rl", 10))

        self.action_dim = action_dim
        self.chunk = chunk
        input_dim = z_dim + state_dim + chunk * action_dim
        self.q1 = _build_mlp(input_dim, self.hidden_dim, self.num_layers, 1)
        self.q2 = _build_mlp(input_dim, self.hidden_dim, self.num_layers, 1)

    def _flatten_chunk(self, chunk: Tensor, batch: int) -> Tensor:
        """Flatten a (B, chunk, action_dim) / (B, chunk*action_dim) / (chunk, action_dim)
        tensor to (B, chunk*action_dim), validating every dimension so an MLP input
        mismatch fails with a clear message instead of an opaque matmul error."""
        flat = self.chunk * self.action_dim
        if chunk.ndim == 3:
            flat_chunk = chunk.reshape(chunk.shape[0], -1)
        elif chunk.ndim == 2 and chunk.shape[1] == flat:
            flat_chunk = chunk
        elif chunk.ndim == 2:
            # (chunk, action_dim) without a batch dimension: treat as one sample.
            flat_chunk = chunk.reshape(1, -1)
        else:
            raise ValueError(
                f"action_chunk must have rank 2 or 3, got rank {chunk.ndim} with shape {tuple(chunk.shape)}"
            )
        if flat_chunk.shape[-1] != flat:
            raise ValueError(
                f"action_chunk last dim {flat_chunk.shape[-1]} != expected {flat} "
                f"(chunk={self.chunk}, action_dim={self.action_dim})"
            )
        if flat_chunk.shape[0] != batch:
            raise ValueError(
                f"action_chunk batch {flat_chunk.shape[0]} != input batch {batch}; "
                "add a batch dimension to action_chunk"
            )
        return flat_chunk

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        action_chunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_flat = self._flatten_chunk(action_chunk, z_rl.shape[0])
        x = torch.cat([z_rl, proprio, action_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        action_chunk: Tensor,
    ) -> Tensor:
        q1, q2 = self(z_rl, proprio, action_chunk)
        return torch.minimum(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 masked reconstruction loss (paper Eq. (2); per-token MSE with mask)
# ─────────────────────────────────────────────────────────────────────────────


def rlt_reconstruction_loss(
    encoder: RLTokenEncoder,
    decoder: RLTokenDecoder,
    vlm_embeddings: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Stage 1 forward: encoder → z_rl → decoder → masked reconstruction loss.

    Per paper Eq. (2): stop-grad target, per-token MSE averaged over valid
    (unmasked) positions only. Returns ``(loss, z_rl)``.
    """
    # Detach targets (stop-gradient as per RLT paper Eq. 2)
    vlm_target = vlm_embeddings.detach()

    # Encode Eq. (1): append e_rl and take the final encoder position.
    z_rl = encoder(vlm_target, mask=mask)

    # Decode Eq. (2): [z_rl, stopgrad(z_1:M-1)] with causal attention.
    vlm_recon = decoder(z_rl, vlm_target, mask=mask)

    # Masked reconstruction loss
    sq_error = (vlm_recon - vlm_target).pow(2)  # (B, L, D)
    if mask is not None:
        # Mean over valid (unmasked) positions only
        mask_expanded = mask.float().unsqueeze(-1)  # (B, L, 1)
        loss = (sq_error * mask_expanded).sum() / (mask_expanded.sum() * vlm_target.shape[-1]).clamp(min=1.0)
    else:
        loss = sq_error.mean()

    return loss, z_rl


def clone_target_network(module: nn.Module) -> nn.Module:
    """Deep-copy a network and freeze it (target actor/critic convention)."""
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad = False
    return target


__all__ = [
    "RLTConfig",
    "RLTokenEncoder",
    "RLTokenDecoder",
    "RLTChunkActor",
    "RLTTwinCritic",
    "rlt_reconstruction_loss",
    "clone_target_network",
]
