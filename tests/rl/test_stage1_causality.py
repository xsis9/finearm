"""Stage 1 causality tests (ported from tests/test_pi05_rlt_paper_stage1.py).

Pins the paper's Stage-1 contract on the ported encoder/decoder:
- the encoder appends the RL token and ignores masked (invalid) inputs
- the decoder is strictly teacher-forced and causal: prediction i can never
  read target i or any future target; changing a past target must affect
  later predictions
- teacher-forcing targets receive no gradient (stop-grad, paper Eq. (2))
- the masked reconstruction loss averages over valid positions only
"""

import torch

from rlt.networks import (
    RLTConfig,
    RLTokenDecoder,
    RLTokenEncoder,
    rlt_reconstruction_loss,
)


def make_config(**overrides):
    values = {
        "vlm_hidden_dim": 8,
        "rlt_hidden_dim": 8,
        "rlt_num_heads": 2,
        "rlt_encoder_layers": 1,
        "rlt_decoder_layers": 1,
        "rlt_dropout": 0.0,
    }
    values.update(overrides)
    return RLTConfig(**values)


def test_encoder_appends_rl_token_and_ignores_masked_inputs():
    torch.manual_seed(0)
    encoder = RLTokenEncoder(make_config()).eval()
    embeddings = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    changed = embeddings.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 100

    with torch.no_grad():
        actual = encoder(embeddings, mask)
        perturbed = encoder(changed, mask)

    assert actual.shape == (2, 8)
    torch.testing.assert_close(actual, perturbed)


def test_encoder_rejects_invalid_shapes():
    encoder = RLTokenEncoder(make_config())

    with torch.no_grad():
        try:
            encoder(torch.randn(2, 8))
        except ValueError as error:
            assert "(B, L, D)" in str(error)
        else:
            raise AssertionError("Expected invalid embedding rank to fail")

        try:
            encoder(torch.randn(2, 4, 8), torch.ones(2, 3, dtype=torch.bool))
        except ValueError as error:
            assert "mask must have shape" in str(error)
        else:
            raise AssertionError("Expected invalid mask shape to fail")


def test_decoder_is_strictly_teacher_forced_and_causal():
    torch.manual_seed(1)
    decoder = RLTokenDecoder(make_config()).eval()
    z_rl = torch.randn(1, 8)
    targets = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)

    with torch.no_grad():
        baseline = decoder(z_rl, targets, mask)

        changed_current = targets.clone()
        changed_current[:, 2] += 50
        current_result = decoder(z_rl, changed_current, mask)

        changed_future = targets.clone()
        changed_future[:, 4] -= 50
        future_result = decoder(z_rl, changed_future, mask)

        changed_past = targets.clone()
        changed_past[:, 1] += 50
        past_result = decoder(z_rl, changed_past, mask)

    # Prediction i cannot read target i or any future target.
    torch.testing.assert_close(baseline[:, :3], current_result[:, :3])
    torch.testing.assert_close(baseline[:, :5], future_result[:, :5])
    # Target 1 is shifted into decoder input position 2 and can affect prediction 2+.
    assert not torch.allclose(baseline[:, 2:], past_result[:, 2:])


def test_decoder_masks_teacher_tokens_from_later_valid_predictions():
    torch.manual_seed(4)
    decoder = RLTokenDecoder(make_config()).eval()
    z_rl = torch.randn(1, 8)
    targets = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, False, True, True, True]])

    changed = targets.clone()
    changed[:, 1] += 100

    with torch.no_grad():
        baseline = decoder(z_rl, targets, mask)
        perturbed = decoder(z_rl, changed, mask)

    # target[1] is shifted to decoder input position 2, but that input position
    # is padding-masked and therefore cannot contaminate later valid outputs.
    torch.testing.assert_close(baseline[:, 2:], perturbed[:, 2:], rtol=1e-4, atol=2e-5)


def test_decoder_stops_gradient_through_teacher_forcing_targets():
    torch.manual_seed(5)
    decoder = RLTokenDecoder(make_config())
    z_rl = torch.randn(1, 8, requires_grad=True)
    targets = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)

    reconstruction = decoder(z_rl, targets, mask)
    reconstruction.square().mean().backward()

    assert z_rl.grad is not None and torch.count_nonzero(z_rl.grad)
    assert targets.grad is None


def test_masked_reconstruction_loss_uses_only_valid_elements():
    predictions = torch.tensor([[[1.0, 3.0], [100.0, 100.0], [5.0, 9.0]]])
    targets = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 3.0]]])
    mask = torch.tensor([[True, False, True]])
    expanded = mask.to(predictions.dtype).unsqueeze(-1)

    loss = ((predictions - targets).square() * expanded).sum() / (
        expanded.sum() * targets.shape[-1]
    )

    # Valid squared errors are [1, 4] and [16, 36].
    torch.testing.assert_close(loss, torch.tensor((1 + 4 + 16 + 36) / 4))


def test_paper_stage1_tiny_batch_overfits():
    torch.manual_seed(3)
    config = make_config()
    encoder = RLTokenEncoder(config)
    decoder = RLTokenDecoder(config)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-3)
    targets = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    losses = []

    for _ in range(100):
        loss, _ = rlt_reconstruction_loss(encoder, decoder, targets, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1


def test_paper_stage1_has_finite_gradients():
    torch.manual_seed(2)
    config = make_config()
    encoder = RLTokenEncoder(config)
    decoder = RLTokenDecoder(config)
    targets = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    loss, _ = rlt_reconstruction_loss(encoder, decoder, targets, mask)
    loss.backward()

    assert torch.isfinite(loss)
    for module in (encoder, decoder):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
        assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
