# sft · fine-tuning (SFT)

[English](sft.md) · [中文](sft_zh.md)

## Environment

The upstream environment is reproduced exactly by the committed `uv.lock`:

```bash
cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, once
uv sync                                            # reproduce the training env

export LEROBOT_HOME=<dir-containing-the-datasets>  # lerobot resolves local datasets
```

## Training (π0.5, panthera)

Do these things in order:

1. **Smoke run**: 10 steps with the `pi05_panthera_p3` config — proves weights + data + env before the full run;
2. **Full run**: same config, full length.

For the concrete commands (train.py usage, config overrides, norm-stats
computation, GPU-memory settings) refer to the official README:
[`models/openpi/README.md`](../models/openpi/README.md), section
*Fine-Tuning Base Models on Your Own Data* — and check it first whenever
training misbehaves.

Checkpoints land under `checkpoints/pi05_panthera_p3/<exp_name>/`.
Per-checkpoint auto-evaluation: `bash scripts/watch_panthera_p3.sh`
(separate terminal). Loss plot:
`uv run python scripts/plot_panthera_p3_loss.py <train.log>`.
