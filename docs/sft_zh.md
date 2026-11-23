# sft · 微调(SFT)

[English](sft.md) · [中文](sft_zh.md)

## 环境

上游环境由入库的 `uv.lock` 精确复现:

```bash
cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv,一次性
uv sync                                            # 复现训练环境

export LEROBOT_HOME=<数据集所在目录>                # lerobot 按此解析本地数据集
```

## 训练(π0.5,panthera)

按顺序做这四件事:

1. **适配机械臂**:参考
   [`panthera_policy.py`](../models/openpi/src/openpi/policies/panthera_policy.py)
   写机械臂的策略适配(panthera 臂已写好);
2. **写数据配置**:写 `class LeRobotPantheraDataConfig(DataConfigFactory)`
   ([`config.py:361`](../models/openpi/src/openpi/training/config.py),panthera 臂已写好);
3. **写训练配置**:在同一
   [`config.py`](../models/openpi/src/openpi/training/config.py) 的 `CONFIGS`
   列表里仿照 `pi05_panthera` 写——主要是三处:`repo_id`(数据集 id)、
   `weight_loader`(基线权重加载)、`num_train_steps`(训练步数);
4. **开训**(在 `models/openpi/` 下):

   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_panthera --exp-name=my_experiment --overwrite --no-wandb-enabled
   ```

   `--no-wandb-enabled` 关闭 wandb 日志(`wandb_enabled` 默认开,国内访问不了);
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 让 JAX 可用 90% 显存。

命令细节与训练问题参考官方 README:
[`models/openpi/README.md`](../models/openpi/README.md)。

checkpoint 落在 `checkpoints/<配置名>/<exp_name>/`。
