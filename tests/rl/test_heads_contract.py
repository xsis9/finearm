"""Actor/critic head contract tests (ported from tests/test_pi05_rlt_paper_heads.py).

Pins the state-dict layout that makes trained checkpoints loadable:
``net.0/net.2`` for the actor, ``q1.*/q2.*`` for the critic
(``actor_contract = paper_full_output_v1``). Adapted to the Panthera 7-dim
action space and extended with full-size shape checks (2048-dim z_rl).
"""

import pytest
import torch

from finearm.rl.algorithms.rltoken.networks import RLTConfig, RLTChunkActor, RLTTwinCritic, RLTokenDecoder, RLTokenEncoder


def make_config(**overrides) -> RLTConfig:
    values = {
        "rlt_hidden_dim": 32,  # z_rl
        "state_dim": 7,
        "action_dim": 7,
        "n_action_steps_rl": 10,
        "actor_hidden_dim": 16,
        "actor_num_layers": 2,
        "critic_hidden_dim": 16,
        "critic_num_layers": 2,
        "policy_fixed_std": 0.05,
        "mode": "online_rl",
    }
    values.update(overrides)
    return RLTConfig(**values)


def test_paper_actor_state_dict_layout_matches_trained_contract():
    actor = RLTChunkActor(make_config())
    keys = list(actor.state_dict())
    # num_layers=2 -> Linear(0), ReLU(1), Linear(2): only net.0/net.2 have params.
    assert keys == ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"]
    assert actor.is_residual is False
    # net.0 input = z(32) + state(7) + chunk(10*7); net.2 output = chunk(10) * action(7)
    assert tuple(actor.net[0].weight.shape) == (16, 32 + 7 + 70)
    assert tuple(actor.net[2].weight.shape) == (70, 16)


def test_paper_critic_state_dict_layout_matches_trained_contract():
    critic = RLTTwinCritic(make_config())
    keys = list(critic.state_dict())
    assert keys == [
        "q1.0.weight", "q1.0.bias", "q1.2.weight", "q1.2.bias",
        "q2.0.weight", "q2.0.bias", "q2.2.weight", "q2.2.bias",
    ]


def test_paper_actor_forward_shapes_and_fixed_std():
    cfg = make_config()
    actor = RLTChunkActor(cfg).eval()
    z = torch.randn(2, 32)
    state = torch.randn(2, 7)
    ref = torch.randn(2, 10, 7)
    with torch.no_grad():
        mean, std = actor(z, state, ref)
    assert tuple(mean.shape) == (2, 10, 7)
    assert torch.allclose(std, torch.full_like(std, cfg.policy_fixed_std))
    # explicit overrides (mirrors legacy script construction) still work
    actor2 = RLTChunkActor(make_config(), hidden_dim=24, num_layers=3)
    assert actor2.hidden_dim == 24 and actor2.num_layers == 3
    assert tuple(actor2.net[0].weight.shape) == (24, 32 + 7 + 70)


def test_paper_critic_forward_and_q_min():
    critic = RLTTwinCritic(make_config()).eval()
    z = torch.randn(2, 32)
    state = torch.randn(2, 7)
    actions = torch.randn(2, 10, 7)
    with torch.no_grad():
        q1, q2 = critic(z, state, actions)
        qmin = critic.q_min(z, state, actions)
    assert tuple(q1.shape) == (2, 1)
    assert torch.allclose(qmin, torch.minimum(q1, q2))


def test_legacy_style_singleton_inputs_accepted():
    """Training code sometimes feeds (chunk, action_dim) refs without a batch dim."""
    actor = RLTChunkActor(make_config()).eval()
    z = torch.randn(32)
    state = torch.randn(7)
    ref = torch.randn(10, 7)
    with torch.no_grad():
        mean, _ = actor(z, state, ref)
    assert tuple(mean.shape) == (1, 10, 7)


def test_misaligned_chunks_fail_with_clear_errors():
    actor = RLTChunkActor(make_config()).eval()
    critic = RLTTwinCritic(make_config()).eval()
    z = torch.randn(2, 32)
    state = torch.randn(2, 7)
    with pytest.raises(ValueError, match="ref_chunk last dim"):
        actor(z, state, torch.randn(2, 10, 6))       # 6-dim ref on a 7-dim actor
    with pytest.raises(ValueError, match="action_chunk last dim"):
        critic(z, state, torch.randn(2, 9, 7))       # wrong chunk length


def test_config_validates_paper_hyperparameters():
    with pytest.raises(ValueError):
        make_config(actor_num_layers=0)
    with pytest.raises(ValueError):
        make_config(actor_hidden_dim=0)
    with pytest.raises(ValueError):
        make_config(policy_fixed_std=0.0)
    with pytest.raises(ValueError):
        make_config(n_action_steps_rl=0)
    with pytest.raises(ValueError):
        make_config(action_stride=0)
    with pytest.raises(ValueError):
        make_config(rlt_hidden_dim=33, rlt_num_heads=8)  # not divisible by heads
    with pytest.raises(ValueError):
        make_config(mode="bogus")


def test_config_defaults_match_trained_contract():
    """Defaults must stay aligned with the paper checkpoints (2048-dim z_rl)
    and the Panthera p3 base (7-dim actions, VLA horizon 30)."""
    cfg = RLTConfig(mode="online_rl")
    assert cfg.rlt_hidden_dim == 2048
    assert cfg.vlm_hidden_dim == 2048
    assert cfg.rlt_encoder_layers == 4
    assert cfg.rlt_decoder_layers == 4
    assert cfg.rlt_num_heads == 8
    assert (cfg.state_dim, cfg.action_dim) == (7, 7)
    assert cfg.chunk_size == 30
    assert cfg.n_action_steps_rl == 10
    assert cfg.action_stride == 2
    assert cfg.policy_fixed_std == 0.05
    assert cfg.target_tau == 0.005
    assert cfg.online_bc_weight == 10.0
    assert cfg.online_q_weight == 1.0
    assert cfg.reference_dropout == 0.5


def test_full_size_actor_shape_7dim():
    """Real-dimension shape check: z(2048) + proprio(7) + ref(10*7) = 2125 -> 70."""
    cfg = RLTConfig(mode="online_rl")
    actor = RLTChunkActor(cfg)
    critic = RLTTwinCritic(cfg)
    assert tuple(actor.net[0].weight.shape) == (256, 2048 + 7 + 70)
    assert tuple(actor.net[-1].weight.shape) == (70, 256)
    assert tuple(critic.q1[0].weight.shape) == (256, 2048 + 7 + 70)
    z = torch.randn(2, 2048)
    state = torch.randn(2, 7)
    ref = torch.randn(2, 10, 7)
    with torch.no_grad():
        mean, std = actor(z, state, ref)
        q1, q2 = critic(z, state, ref)
    assert tuple(mean.shape) == (2, 10, 7)
    assert tuple(q1.shape) == (2, 1)


def test_full_size_encoder_decoder_shape():
    """Encoder consumes (B, 512, 2048) image tokens (2 valid cameras x 256);
    decoder reconstructs the same sequence from z_rl."""
    cfg = RLTConfig()
    encoder = RLTokenEncoder(cfg).eval()
    decoder = RLTokenDecoder(cfg).eval()
    tokens = torch.randn(2, 512, 2048)
    mask = torch.ones(2, 512, dtype=torch.bool)
    with torch.no_grad():
        z_rl = encoder(tokens, mask)
        recon = decoder(z_rl, tokens, mask)
    assert tuple(z_rl.shape) == (2, 2048)
    assert tuple(recon.shape) == (2, 512, 2048)
