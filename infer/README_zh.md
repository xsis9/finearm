# infer · 推理环境

[English](README.md) · [中文](README_zh.md)

推理客户端环境(openpi-client venv)。客户端**代码**在 `src/finearm/infer/`
——模型适配器、分块执行器(`finearm-infer --exec-mode {rtc,stop-and-go}`)、
采集后端(`finearm.infer.robot_interface`:daemon IPC + 相机)。

## 环境搭建(执行机,一次性)

```bash
./infer/init_env.sh                                  # 建 openpi-client 环境(或复用现有)
infer/.venv/bin/pip install -e . --no-deps           # 装 finearm 命令行入口
```

## 三方分工

| 角色 | 机器 | 命令 |
|---|---|---|
| 策略服务 | GPU 机(4090) | `uv run scripts/serve_policy.py …`(见下;用 `finearm/models/openpi` vendored 树) |
| 隧道 | 执行机 | `ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>` |
| 推理客户端 | 执行机 | `finearm-infer …`(下面安全阶梯) |

### 起服务(GPU 机)—— pi05

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync

# 从训练机搬训练产物:
rsync -av <user>@<训练机>:<openpi树>/checkpoints/pi05_panthera*/<exp>/ checkpoints/

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera_p3 \
    --policy.dir checkpoints/pi05_panthera_p3/<exp>/<step>
# 默认端口 8000;调试加 --record 记录每条请求的 obs/动作。
```

### 起服务 —— lingbot(外部栈,端口 8006)

```bash
cd <lingbot-vla-v2 仓库根>            # 必须仓库根:config 按相对路径解析
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/微调ckpt/hf_ckpt --use_bf16 true --use_compile true \
  --use_length 25 --port 8006
```

lingbot 核对清单:checkpoint 必须是含 training config 的 `hf_ckpt` 目录;
`--use_length` ≤ 训练 action horizon;`configs/robot_configs/panthera.yaml`
与 `assets/norm_stats/panthera.json` 在位且 norm-stats 与训练同一份。

### 推理(执行机)—— 安全阶梯

```bash
cd <finearm 仓库根> && source infer/.venv/bin/activate

# 一级 · 无硬件(mock 适配器,无服务):
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
# 二级 · 真相机 + 真 daemon,action 只打印不下发:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --dry-run --duration 60
# 三级 · 低速真控,断档即停:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30
# 四级 · 正常:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --task "<训练 prompt>"
```

执行机预检:

```bash
python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs   # 相机/时延/图像方向
PYTHONPATH=src python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1   # RTT + 建议 k
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--task` | **必须与训练 prompt 逐字一致**(握手校验) |
| `--home` | 归位位姿(7 值);`--home` > `PANTHERA_RTC_HOME` > 适配器默认;空串跳过归位 |
| `--exec-mode` | `rtc`(chunk 流水线)或 `stop-and-go`(分窗) |
| `--k/--blend/--starve` | replan 余量 / 混合步数 / 断档策略(`hold` 或 `stop`) |
| `--pos-lo/--pos-hi/--vel-limit` | 显式限幅(臂侧只保留逐拍跳变兜底) |
| `PANTHERA_CAM_FRONT/WRIST` | 相机源(默认 `/dev/video4`、`/dev/video2`) |

## 新增模型适配器

在 `src/finearm/infer/adapters/` 加一个文件(文件名 = `--model` 参数值,
不要下划线开头),约 50 行,继承 `AdapterBase`。执行器/控制环/安全逻辑
模型无关;适配器只声明连接方式、默认端口、观测风格,以及 infer/reset 到
服务端协议的映射。现成例子:`pi05.py`、`lingbot.py`、`mock.py`。RTC 对
服务端零改动——每连接任一时刻最多一个在途请求。
