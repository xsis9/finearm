# sft · fine-tuning (SFT)

[English](sft.md) · [中文](sft_zh.md)

This directory holds the model trees: the vendored **modified openpi**
(`openpi/`, the runnable training/serving tree) and the heavy local
assets that never enter git (base weights, checkpoints, datasets — see
`.gitignore`). The lingbot stack stays external.

## Handoff contract (dataset → checkpoint)

What makes a collected dataset trainable and the resulting checkpoint
servable:

1. the dataset directory (LeRobot v3) is the sole training input;
2. the **task string** recorded at collection must equal the training
   prompt verbatim — it is checked again at the serve/infer handshake;
3. any post-training must use the quantile stats of the same dataset
   (`meta/stats.json`) — the same-origin chain (dataset → stats →
   checkpoint → served task) is validated end-to-end at startup;
4. the trained checkpoint + its task string are what serving loads.

## Reproduce the training environment

The upstream env is reproduced exactly by the committed `uv.lock`:

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, once
uv sync                                            # reproduce the training env

# heavy assets (not in git) — copy once from wherever they live:
#   pi0.5 base weights → checkpoints/pi05_base/          (referenced tree-relative)
#   datasets (panthera_bolt_p3 …) → any dir, then:
export LEROBOT_HOME=<dir-containing-the-datasets>  # lerobot resolves local datasets
```

## Train (π0.5, panthera)

```bash
# smoke (10 steps — proves weights + data + env before the full run):
uv run python scripts/train.py --config pi05_panthera_p3 num_train_steps=10 exp_name=smoke
# full run:
uv run python scripts/train.py --config pi05_panthera_p3
```

Checkpoints land under `checkpoints/pi05_panthera_p3/<exp_name>/`.
Per-checkpoint auto-evaluation: `bash scripts/watch_panthera_p3.sh`
(separate terminal). Loss plot:
`uv run python scripts/plot_panthera_p3_loss.py <train.log>`.

## Vendored-tree provenance

`openpi/` = upstream `215abfb` + the panthera patch (`panthera_policy`,
`pi05_panthera_p3` config, lerobot-0.5.1/numpy-2/py3.12 compat, eval
scripts). Exact delta: [`third_party/patches/openpi-panthera.patch`](../third_party/patches/openpi-panthera.patch).
Rebase onto a newer upstream: checkout upstream, apply the patch.

Next step — serving the checkpoint on the GPU box:
[`infer.md`](infer.md).
