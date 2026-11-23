# sft · 微调(SFT)

[English](sft.md) · [中文](sft_zh.md)

本目录承载模型树:vendored 的 **修改版 openpi**(`openpi/`,可直接训练/服务
的树)与不入 git 的重资产(基线权重、checkpoint、数据集——见 `.gitignore`)。
lingbot 栈仍在外部。

## 交接契约(数据集 → checkpoint)

让采集数据可训、产物 checkpoint 可服务的契约:

1. 数据集目录(LeRobot v3)是唯一训练输入;
2. 采集时记录的 **task 字符串**必须与训练 prompt 逐字一致——serve/infer
   握手时会再校验;
3. 任何后训练必须用同一数据集的分位统计(`meta/stats.json`)——同源链
   (数据集 → stats → checkpoint → 服务的 task)启动时端到端校验;
4. 训练出的 checkpoint + 其 task 字符串就是服务加载的内容。

## 复现训练环境

上游环境由入库的 `uv.lock` 精确复现:

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv,一次性
uv sync                                            # 复现训练环境

# 重资产(不入 git)——从现存位置拷一次:
#   π0.5 基线权重 → checkpoints/pi05_base/          (config 按树相对路径引用)
#   数据集(panthera_bolt_p3 …)→ 任意目录,然后:
export LEROBOT_HOME=<数据集所在目录>                # lerobot 按此解析本地数据集
```

## 训练(π0.5,panthera)

```bash
# 冒烟(10 步——先证明 权重+数据+环境 全链通再跑全量):
uv run python scripts/train.py --config pi05_panthera_p3 num_train_steps=10 exp_name=smoke
# 全量:
uv run python scripts/train.py --config pi05_panthera_p3
```

checkpoint 落在 `checkpoints/pi05_panthera_p3/<exp_name>/`。逐 checkpoint
自动评估:另开终端 `bash scripts/watch_panthera_p3.sh`;loss 曲线:
`uv run python scripts/plot_panthera_p3_loss.py <train.log>`。

## vendored 树溯源

`openpi/` = 上游 `215abfb` + panthera 补丁(`panthera_policy`、
`pi05_panthera_p3` 配置、lerobot-0.5.1/numpy-2/py3.12 兼容、评估脚本)。
精确差异:[`third_party/patches/openpi-panthera.patch`](../third_party/patches/openpi-panthera.patch)。
跟上游更新:checkout 上游后应用补丁。

下一步——在 GPU 机上服务 checkpoint:
[`infer_zh.md`](infer_zh.md)。
