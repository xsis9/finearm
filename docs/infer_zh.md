# infer · 服务与推理

[English](infer.md) · [中文](infer_zh.md)


## 环境搭建(执行机,一次性)

```bash
./envs/infer/init_env.sh                                  # 建 openpi-client 环境(或复用现有)
envs/infer/.venv/bin/pip install -e . --no-deps           # 装 finearm 命令行入口
```

## 三方分工

| 角色 | 机器 | 命令 |
|---|---|---|
| 策略服务 | GPU 机(4090) | `uv run scripts/serve_policy.py …`(见下) |
| 推理客户端 | 执行机 | `finearm-infer …`(下面安全阶梯) |

### 起服务(GPU 机)—— pi05

```bash
cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync

# 从训练机搬checkpoint到推理机

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera \
    --policy.dir checkpoints/pi05_panthera/<exp>/<step>
# 默认端口 8000;调试加 --record 记录每条请求的 obs/动作。
```

### 推理(执行机)—— 安全阶梯

执行机预检:

```bash
bash utils/lock_cameras.sh                              # 永远第一步:相机参数必须与训练/推理一致
python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs   # 相机/时延/图像方向
PYTHONPATH=src python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1   # RTT + 建议 k
```

相机锁定不只是一次性预检:采集后端(`finearm.infer.robot_interface`)启动时
会自动执行 `utils/lock_cameras.sh`——与采集端共用的同一份权威档案。UVC
相机重插/断电后参数会复位,参数不一致会让观测分布悄悄偏离训练态。
`PANTHERA_CAM_LOCK` 覆盖脚本路径;`PANTHERA_CAM_LOCK_SKIP=1` 跳过(仅调试)。

```bash
cd <finearm 仓库根> && source envs/infer/.venv/bin/activate

# 这里由于公司4090服务器用的docker,无法直接访问，所以先在执行机ssh连接推理机，然后在执行机使用127.0.0.1访问推理机
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host> 

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

## 常用参数

| 参数 | 说明 |
|---|---|
| `--task` | **必须与训练 prompt 逐字一致**(握手校验) |
| `--home` | 归位位姿(7 值);`--home` > `PANTHERA_RTC_HOME` > 适配器默认;空串跳过归位 |
| `--exec-mode` | `rtc`(chunk 流水线)或 `stop-and-go`(分窗) |
| `--k/--blend/--starve` | replan 余量 / 混合步数 / 断档策略(`hold` 或 `stop`) |
| `--pos-lo/--pos-hi/--vel-limit` | 显式限幅(臂侧只保留逐拍跳变兜底) |
| `PANTHERA_CAM_FRONT/WRIST` | 相机源(默认 `/dev/video4`、`/dev/video2`) |
| `PANTHERA_CAM_LOCK/SKIP` | 相机锁定脚本路径覆盖 / `=1` 跳过(仅调试) |

## 新增模型适配器

在 `src/finearm/infer/adapters/` 加一个文件(文件名 = `--model` 参数值,
不要下划线开头),约 50 行,继承 `AdapterBase`。执行器/控制环/安全逻辑
模型无关;适配器只声明连接方式、默认端口、观测风格,以及 infer/reset 到
服务端协议的映射。现成例子:`pi05.py`、`lingbot.py`、`mock.py`。
