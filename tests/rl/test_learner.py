"""Learner core tests: TD3 helpers, checkpoint schema, and an end-to-end smoke
test wiring ReplayBuffer.sample straight into perform_training_update with
tiny 7-dim-style networks (the same batch format Stage 2 will produce).
"""

import copy
import tempfile
from pathlib import Path

import pytest
import torch

from finearm.rl.algorithms.rltoken.learner import (
    ACTOR_CONTRACT,
    RL_CHUNK_LENGTH,
    SCHEMA_VERSION,
    atomic_torch_save,
    build_stage2_checkpoint,
    discounted_chunk_return,
    frozen_parameters,
    perform_training_update,
    polyak_update,
    prepare_output_dir,
    prepare_resume_checkpoint,
    sample_fixed_std_action,
)
from finearm.rl.algorithms.rltoken.networks import RLTConfig, RLTChunkActor, RLTTwinCritic, clone_target_network
from finearm.rl.replay import (
    BoundaryContext,
    ExecutionWindow,
    ReplayBuffer,
    TransitionSource,
    build_transition,
)


# ── TD3 helpers ──────────────────────────────────────────────────────────────


def test_polyak_update_moves_target_toward_source():
    source = torch.nn.Linear(1, 1, bias=False)
    target = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(4.0)
        target.weight.fill_(0.0)

    polyak_update(source, target, 0.25)

    torch.testing.assert_close(target.weight, torch.tensor([[1.0]]))


def test_fixed_std_sample_is_stochastic():
    torch.manual_seed(7)
    mean = torch.zeros(1, 2, 1)
    std = torch.full_like(mean, 0.25)

    sampled = sample_fixed_std_action(mean, std)

    assert not torch.equal(sampled, mean)
    torch.manual_seed(7)
    torch.testing.assert_close(sampled, sample_fixed_std_action(mean, std))


def test_discounted_chunk_return_values():
    # reward at step 0, executed 2 of 3 rows: return = 1 * γ^0 = 1, bootstrap = γ^2
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 0.0]])
    returns, bootstrap = discounted_chunk_return(rewards, valid, 0.5)
    torch.testing.assert_close(returns, torch.tensor([[1.0]]))
    torch.testing.assert_close(bootstrap, torch.tensor([[0.25]]))

    # reward at step 1, all rows executed: return = γ^1, bootstrap = γ^3
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0]])
    returns, bootstrap = discounted_chunk_return(rewards, valid, 0.5)
    torch.testing.assert_close(returns, torch.tensor([[0.5]]))
    torch.testing.assert_close(bootstrap, torch.tensor([[0.125]]))


def test_frozen_parameters_preserves_action_gradient_without_critic_grads():
    critic = torch.nn.Linear(2, 1)
    action = torch.tensor([[1.0, 2.0]], requires_grad=True)
    with frozen_parameters(critic):
        critic(action).sum().backward()
    assert action.grad is not None
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert all(parameter.requires_grad for parameter in critic.parameters())


# ── Checkpoint schema ────────────────────────────────────────────────────────


def test_checkpoint_builder_uses_paper_full_output_contract():
    checkpoint = build_stage2_checkpoint(total_updates=2)
    assert checkpoint["resume_supported"]
    assert checkpoint["schema_version"] == SCHEMA_VERSION == 3
    assert checkpoint["actor_contract"] == ACTOR_CONTRACT == "paper_full_output_v1"
    assert checkpoint["rl_chunk_length"] == RL_CHUNK_LENGTH == 10
    assert checkpoint["total_updates"] == 2


def test_atomic_save_round_trip(tmp_path):
    payload = {"tensor": torch.ones(3), "value": 7}
    target = tmp_path / "ckpt.pt"
    atomic_torch_save(payload, target)
    loaded = torch.load(target, weights_only=True)
    torch.testing.assert_close(loaded["tensor"], torch.ones(3))
    assert loaded["value"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_prepare_output_dir_preflight(tmp_path):
    fresh = tmp_path / "new-output"
    prepare_output_dir(fresh, dry_run=True)
    assert not fresh.exists()
    prepare_output_dir(fresh, dry_run=False)
    assert fresh.is_dir()
    (fresh / "prior.pt").write_bytes(b"prior")
    with pytest.raises(FileExistsError):
        prepare_output_dir(fresh, dry_run=False)  # now non-empty


def test_prepare_resume_checkpoint_rejects_foreign_schema(tmp_path):
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    ckpt = source_dir / "resume.pt"
    ckpt.write_bytes(b"")
    # Missing schema keys -> rejected before touching torch.load internals.
    with pytest.raises((ValueError, EOFError)):
        prepare_resume_checkpoint(ckpt, tmp_path / "out", dry_run=True)


# ── perform_training_update end-to-end (tiny nets + real replay batch) ───────

TINY = RLTConfig(
    rlt_hidden_dim=8,
    vlm_hidden_dim=8,
    rlt_num_heads=2,
    rlt_encoder_layers=1,
    rlt_decoder_layers=1,
    state_dim=3,
    action_dim=3,
    n_action_steps_rl=4,
    chunk_size=12,
    actor_hidden_dim=16,
    critic_hidden_dim=16,
    mode="online_rl",
)


def _tiny_batch(n=6, horizon=4, action_dim=3, z_dim=8):
    """Build a replay-style batch via the real transition machinery."""
    buffer = ReplayBuffer(64, seed=0)
    for step in range(n):
        boundary = BoundaryContext(
            z_rl=torch.randn(1, z_dim),
            proprio=torch.randn(action_dim),
            ref_chunk=torch.randn(1, horizon, action_dim),
            collection_phase="online",
            episode_id=step // 3,
            step_id=step,
        )
        window = ExecutionWindow(boundary)
        for k in range(horizon):
            source = int(TransitionSource.HUMAN) if k == 0 and step % 3 == 0 else int(TransitionSource.BASE)
            window.append(torch.randn(action_dim), source, source == int(TransitionSource.HUMAN))
        next_boundary = BoundaryContext(
            z_rl=torch.randn(1, z_dim),
            proprio=torch.randn(action_dim),
            ref_chunk=torch.randn(1, horizon, action_dim),
            collection_phase="online",
            episode_id=step // 3,
            step_id=step + 1,
        )
        transition = build_transition(
            window, next_boundary, horizon=horizon, action_dim=action_dim,
            reward=1.0 if step == n - 1 else 0.0, terminal=step == n - 1,
        )
        buffer.add(transition)
    return buffer.sample(n)


def test_perform_training_update_runs_and_reports_metrics():
    torch.manual_seed(0)
    actor = RLTChunkActor(TINY)
    actor_target = clone_target_network(actor)
    critic = RLTTwinCritic(TINY)
    critic_target = clone_target_network(critic)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    batch = _tiny_batch()

    before_actor = copy.deepcopy(actor.state_dict())
    before_critic = copy.deepcopy(critic.state_dict())
    target_before = copy.deepcopy(critic_target.state_dict())

    metrics = perform_training_update(
        actor=actor, actor_target=actor_target,
        critic=critic, critic_target=critic_target,
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
        batch=batch, total_updates=0,
        discount=0.99, target_tau=0.005,
        critic_per_actor=2, reference_dropout=0.5,
    )

    # total_updates=0 % 2 == 0 -> actor update ran on the first call
    assert metrics["actor_updated"] is True
    for key in ("critic_loss", "q1_mean", "q2_mean", "td_target_mean", "actor_loss",
                "actor_q", "bc_penalty", "bc_weight", "q_weight"):
        assert key in metrics and metrics[key] == metrics[key]  # present and not NaN
    # online weights (learner_warmup_updates=0 -> online regime)
    assert metrics["bc_weight"] == 10.0 and metrics["q_weight"] == 1.0
    # TD target clamped into [-2, 2]
    assert metrics["td_target_mean"] <= 2.0
    # parameters actually moved; targets moved by polyak only
    assert not torch.equal(actor.net[0].weight, before_actor["net.0.weight"])
    assert not torch.equal(critic.q1[0].weight, before_critic["q1.0.weight"])
    moved = (critic_target.q1[0].weight - target_before["q1.0.weight"]).abs().max()
    assert 0.0 < float(moved) < float((critic.q1[0].weight - before_critic["q1.0.weight"]).abs().max())


def test_actor_update_cadence_follows_critic_per_actor():
    torch.manual_seed(1)

    def fresh():
        actor = RLTChunkActor(TINY)
        critic = RLTTwinCritic(TINY)
        return (
            actor, clone_target_network(actor),
            critic, clone_target_network(critic),
            torch.optim.AdamW(actor.parameters(), lr=1e-3),
            torch.optim.AdamW(critic.parameters(), lr=1e-3),
        )

    batch = _tiny_batch()
    # total_updates 0 -> actor updates; 1 -> critic only; 2 -> actor again.
    for total, expected in ((0, True), (1, False), (2, True)):
        actor, actor_target, critic, critic_target, actor_opt, critic_opt = fresh()
        metrics = perform_training_update(
            actor=actor, actor_target=actor_target,
            critic=critic, critic_target=critic_target,
            actor_optimizer=actor_opt, critic_optimizer=critic_opt,
            batch=batch, total_updates=total, critic_per_actor=2,
            reference_dropout=0.0,
        )
        assert metrics["actor_updated"] is expected


def test_warmup_weight_schedule_switches():
    torch.manual_seed(2)
    actor = RLTChunkActor(TINY)
    critic = RLTTwinCritic(TINY)
    batch = _tiny_batch()
    metrics = perform_training_update(
        actor=actor, actor_target=clone_target_network(actor),
        critic=critic, critic_target=clone_target_network(critic),
        actor_optimizer=torch.optim.AdamW(actor.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-3),
        batch=batch, total_updates=4, learner_warmup_updates=5,
        warmup_bc_weight=10.0, warmup_q_weight=0.0,
        online_bc_weight=10.0, online_q_weight=1.0,
        reference_dropout=0.0,
    )
    assert metrics["actor_updated"] is True
    assert metrics["bc_weight"] == 10.0 and metrics["q_weight"] == 0.0  # still warmup


def test_delta_weight_requires_unnormalize():
    torch.manual_seed(3)
    actor = RLTChunkActor(TINY)
    critic = RLTTwinCritic(TINY)
    batch = _tiny_batch()
    with pytest.raises(ValueError, match="unnormalize_action"):
        perform_training_update(
            actor=actor, actor_target=clone_target_network(actor),
            critic=critic, critic_target=clone_target_network(critic),
            actor_optimizer=torch.optim.AdamW(actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-3),
            batch=batch, total_updates=0, delta_weight=1.0, reference_dropout=0.0,
        )


def test_delta_penalty_runs_with_unnormalize():
    torch.manual_seed(4)
    actor = RLTChunkActor(TINY)
    critic = RLTTwinCritic(TINY)

    def unnormalize(tensor):
        return tensor * 2.0  # arbitrary affine stand-in

    batch = _tiny_batch()
    metrics = perform_training_update(
        actor=actor, actor_target=clone_target_network(actor),
        critic=critic, critic_target=clone_target_network(critic),
        actor_optimizer=torch.optim.AdamW(actor.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-3),
        batch=batch, total_updates=0, delta_weight=0.1,
        unnormalize_action=unnormalize, reference_dropout=0.0,
    )
    assert metrics["actor_updated"] is True
    assert metrics["delta_penalty"] >= 0.0
