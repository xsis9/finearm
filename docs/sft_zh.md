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

按顺序做这几件事:

1. **冒烟**:用 `pi05_panthera_p3` 配置跑 10 步——先证明 权重 + 数据 + 环境 全链通;
2. **全量训练**:同一配置跑全量。

具体命令(train.py 用法、配置覆盖、norm-stats 计算、显存设置等)参考官方
README:[`models/openpi/README.md`](../models/openpi/README.md) 的
*Fine-Tuning Base Models on Your Own Data* 一节;训练遇到问题也先查它。

checkpoint 落在 `checkpoints/pi05_panthera_p3/<exp_name>/`。逐 checkpoint
自动评估:另开终端 `bash scripts/watch_panthera_p3.sh`;loss 曲线:
`uv run python scripts/plot_panthera_p3_loss.py <train.log>`。
