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

Do these four things in order:

1. **Adapt the arm**: write the policy adapter for your arm, modeled on
   [`panthera_policy.py`](../models/openpi/src/openpi/policies/panthera_policy.py)
   (already written for the panthera arm);
2. **Data config**: write `class LeRobotPantheraDataConfig(DataConfigFactory)`
   ([`config.py:361`](../models/openpi/src/openpi/training/config.py) — already
   written for the panthera arm);
3. **Training config**: in the `CONFIGS` list of the same
   [`config.py`](../models/openpi/src/openpi/training/config.py), add an entry
   modeled on `pi05_panthera` — chiefly three fields: `repo_id` (dataset id),
   `weight_loader` (base-weight loading), and `num_train_steps`;
4. **Train** (from `models/openpi/`):

   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_panthera --exp-name=my_experiment --overwrite --no-wandb-enabled
   ```

   `--no-wandb-enabled` disables wandb logging (`wandb_enabled` defaults on,
   unreachable from mainland China); `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
   lets JAX use up to 90% of GPU memory.

For command details and training problems refer to the official README:
[`models/openpi/README.md`](../models/openpi/README.md).

Checkpoints land under `checkpoints/<config>/<exp_name>/`. Per-checkpoint
auto-evaluation: `bash scripts/watch_panthera_p3.sh` (separate terminal).
Loss plot: `uv run python scripts/plot_panthera_p3_loss.py <train.log>`.
