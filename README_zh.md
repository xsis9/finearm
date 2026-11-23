# finearm

[English](README.md) · [中文](README_zh.md)

VLA 微调框架:遥操作数据采集、微调、推理、以及强化学习后训练。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `arm/panthera/` | 验证过的机械臂资产:厂商 SDK(gitignored)、标定(`robot_param/`)、daemon 启动器(`run_daemon.sh`;驱动环境在 `envs/arm/`) |
| `plugins/panthera-lerobot-collect/` | LeRobot 采集栈:robot/teleoperator 插件、启动 wrapper、操作文档 |
| `envs/` | 全部环境定义与 venv:`infer/`(openpi-client)、`collect/`(LeRobot 0.5.1)、`arm/`(驱动) |
| `RL/rlt_panthera/` | RL 运行产物:campaign checkpoints/data(gitignored)与 stage-2 配置;代码在 `src/finearm/rl/` |
| `src/finearm/core/` | 零依赖核心:四层配置解析、仓库路径、插件注册表、norm-stats 契约 |
| `src/finearm/arms/` | 臂插件:`ArmProfile` schema 声明 + 臂中立控制 daemon(`finearm-daemon --arm <id>`);`panthera_ht`(真臂)与 `mock_twojoint`(扩展性示例) |
| `src/finearm/infer/` | 模型适配器(pi05 / lingbot / mock)+ 分块执行器:`RTCExecutor`(流水线)与 `StopAndGoExecutor`;`finearm-infer --exec-mode {rtc,stop-and-go}`;采集后端 `robot_interface`(daemon IPC + 相机) |
| `src/finearm/rl/` | Stage-2 在线 RL:`rltoken` 算法(networks/critical/learner)、replay + journal、边界接缝(`WindowReceipt`/`BoundaryCollector`)、`EpisodeRunner`、serving(serve_rlt + client)、envs(RobotInterface 契约 + panthera 后端)、离线流水线 |
| `src/finearm/data/` | 采集方法即插件:`teleop`(LeRobot 原生,`--arm` 参数化)与 `dagger` 占位;`finearm-collect --method teleop` |
| `models/openpi/` | vendored 的 openpi 修改版(上游 `215abfb` + panthera 补丁:`panthera_policy`、`pi05_panthera_p3` 配置、lerobot-0.5.1/numpy-2/py3.12 兼容、评估脚本);溯源见 `third_party/` |
| `configs/` | 机器/臂/模型配置(权威副本) |
| `third_party/` | vendored 依赖的上游基线记录 |
| `docs/` | 工作笔记与硬件手册 |

## 任意机器可运行

仓库**不含任何机器身份**——没有主机名、没有绝对路径、没有按机器的档案。
克隆到任意位置,内置默认值跟随检出目录。按机器定制用三层覆盖,优先级从高到低:

1. **CLI 参数**(每次调用,如 `--stats_path`)
2. **环境变量**(如 `PANTHERA_ACQ_IFACE`、`PANTHERA_DATA_ROOT`、`FINEARM_STATS_PATH`)
3. **`configs/local.jsonc`**——机器本地持久化,gitignored
   (从 [`configs/local.example.jsonc`](configs/local.example.jsonc) 复制)

第 4 层(最低)是 `finearm.core.paths` 解析的仓库相对默认值:
`data/datasets`(数据集枢纽)、`arm/panthera/sdk`(厂商 SDK)、
`finearm/infer/robot_interface`(采集后端)、`/tmp`(daemon socket 目录)。
包安装在他处时,`FINEARM_ROOT` 指向某个检出。

### 拓扑

- **单机(本地推理)**:一台带 CUDA GPU 的 Linux 机器(π0.5 bf16 服务需
  ≥8 GB 显存)同时跑推理服务、臂 daemon、采集/RL 客户端——一切默认
  `127.0.0.1`。
- **执行机 + 推理服务机分离**:接臂的执行机跑 daemon/采集/RL;GPU 机服务
  策略。ssh 转发端口(`ssh -N -L 8000:127.0.0.1:8000 <gpu-host>`)后
  `server_host` 保持 `127.0.0.1`——客户端完全无感。

`finearm.core` 只需 Python ≥ 3.11 标准库;更重的依赖在各域环境里
(见各域 README)。

### 臂 daemon

控制 daemon 臂中立;每条臂是一个插件目录,声明自己的 schema、增益与适配器:

```bash
finearm-daemon --arm panthera_ht --mock
```

加一条臂 = 加 `src/finearm/arms/<arm>/`(最小骨架见 `mock_twojoint/`);
daemon、IPC 与客户端自动适应不同维度。

## 全流程:采集 → 微调 → 服务 → 推理

一次性准备——把包(纯 Python,零新依赖)装进需要命令行入口的环境:

```bash
# 采集环境:                            # 推理环境:
cd <仓库根> && <采集venv>/bin/pip install -e . --no-deps
                                       <推理venv>/bin/pip install -e . --no-deps   # (同样,在 <仓库根> 下)
```

| 环节 | 规范入口 | 环境 |
|---|---|---|
| 臂 daemon | `finearm-daemon --arm panthera_ht --live` | 臂 venv 或任意装了厂商 SDK 的 venv |
| 采集 | `finearm-collect --method teleop --live …` | 采集 venv(LeRobot 0.5.1) |
| 微调 | 外部——训练机的 openpi / lingbot 栈 | 见下方交接契约 |
| 服务 | GPU 机上的 openpi / lingbot serve(π0.5 bf16 需 ≥8 GB) | GPU 机 |
| 推理 | `finearm-infer --model <name> --exec-mode rtc …` | 推理 venv(openpi-client) |

采集启动 wrapper(`run_panthera_lerobot_collect.sh`)与 daemon 启动器
(`run_daemon.sh`)仍是推荐的实操入口——两者现在都直接驱动 `finearm` 模块。

### 1 · 臂 daemon

```bash
# 先台架检查(无硬件):
finearm-daemon --arm panthera_ht --mock --loop-hz 1000
# 实机(标定显式传——robot_param 说明见 arm/panthera/README.md):
bash arm/panthera/run_daemon.sh --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml
```

实机安全起步顺序:daemon 以 **hold** 起步 → 读遥测、手动活动主臂确认每个
关节通道跟随 → 才切 `teleop`/`policy`。采集与推理会复用已运行的 daemon
(探测要求模式 ∈ {teleop, policy, freedrive});停止时杀 **python** PID
(不是 bash 包装进程)。

### 2 · 采集(teleop)

```bash
bash utils/lock_cameras.sh        # 永远第一步:相机参数必须与训练/推理一致

source envs/collect/.venv-lerobot/bin/activate
# (预检要求 lerobot-record 在 PATH 上——先激活采集 venv,或把它的 bin/
#  目录前插到 PATH;只靠绝对路径调命令行入口是不够的)

# 规范入口(与启动 wrapper 同一套 flag):
finearm-collect --method teleop --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml \
    --select-cameras \
    --repo-id <namespace>/<dataset> --task "<任务指令>" \
    --episodes 1 --episode-seconds 30 --reset-seconds 60 \
    --fps 30 --hz 1000 --camera-fourcc MJPG
# 或用启动 wrapper(flag 相同,代管 venv):
#   ./plugins/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live …
```

数据集落在 `data/datasets/lerobot/<namespace>/<repo-id>/<时间戳>/`
(可用 `PANTHERA_DATA_ROOT` 覆盖)。产出为 **LeRobot 原生 v3**——直接可训,
无需转换。权威操作手册(三种采集方式、参数表、批次 SOP):
[`plugins/panthera-lerobot-collect/docs/quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md)。

### 3 · 微调(外部交接)

SFT/微调在训练机上跑(修改版 openpi 源码已 vendored 在 `models/openpi/`——
与上游的精确差异见 `third_party/patches/`;lingbot 栈仍在外部)。
让采集数据可训、产物 checkpoint 可服务的交接契约:

1. 数据集目录(LeRobot v3)是唯一训练输入;
2. 采集时记录的 **task 字符串**必须与训练 prompt 逐字一致——serve/infer
   握手时会再校验;
3. 任何后训练必须用同一数据集的分位统计(`meta/stats.json`)——同源链
   (数据集 → stats → checkpoint → 服务的 task)启动时端到端校验;
4. 训练出的 checkpoint + 其 task 字符串就是第 4 步服务的内容。

#### SFT 快速上手(π0.5,用本仓库)

vendored 的 `models/openpi` 就是可训练的树(训练机或任意 CUDA 机器;上游
环境由入库的 `uv.lock` 精确复现):

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv,一次性
uv sync                                            # 复现训练环境

# 重资产(不入 git)——从现存位置拷一次:
#   π0.5 基线权重 → checkpoints/pi05_base/          (config 按树相对路径引用)
#   数据集(panthera_bolt_p3 …)→ 任意目录,然后:
export LEROBOT_HOME=<数据集所在目录>                # lerobot 按此解析本地数据集

# 冒烟(10 步——先证明 权重+数据+环境 全链通再跑全量):
uv run python scripts/train.py --config pi05_panthera_p3 num_train_steps=10 exp_name=smoke
# 全量:
uv run python scripts/train.py --config pi05_panthera_p3
```

checkpoint 落在 `checkpoints/pi05_panthera_p3/<exp_name>/`。逐 checkpoint
自动评估:另开终端 `bash scripts/watch_panthera_p3.sh`;loss 曲线:
`uv run python scripts/plot_panthera_p3_loss.py <train.log>`。跟上游更新:
checkout 上游后应用
[`third_party/patches/openpi-panthera.patch`](third_party/patches/openpi-panthera.patch)。

### 4 · 服务(GPU 机)与隧道

#### 服务快速上手(π0.5,用本仓库)

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv,一次性
uv sync                                            # 复现服务环境

# 搬训练产物(从训练机):
rsync -av <user>@<训练机>:<openpi树>/checkpoints/pi05_panthera_p3/<exp>/ \
    checkpoints/pi05_panthera_p3/<exp>/

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera \
    --policy.dir checkpoints/pi05_panthera/panthera_v2/10000
# 默认端口 8000;调试可加 --record(把每条请求的 obs/动作记到
# policy_records/,事后核对输入)。norm-stats assets 随仓库自带。
```

在 GPU 机上起策略服务(快速上手见上;各模型细节:
[`envs/infer/README.md`](envs/infer/README.md)——pi05 端口 8000,lingbot 8006),
然后在执行机转发端口并保持 `--host 127.0.0.1`:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>
```

### 5 · 推理(安全阶梯——逐级放行)

在**执行机**(接机械臂的那台)、仓库根目录、**推理环境**里运行
(openpi-client + finearm 命令行入口——即上面一次性准备里装过的那个):

```bash
source envs/infer/.venv/bin/activate     # 还没装 → envs/infer/.venv/bin/pip install -e . --no-deps

# 一级 · 无硬件,全客户端栈(mock 适配器,无服务):
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
# 二级 · 真相机 + 真 daemon,action 只打印不下发:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --dry-run --duration 60
# 三级 · 低速真控;断档即停:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30
# 四级 · 正常运行:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --task "<训练 prompt>"
```

`--task` 必须与训练 prompt 逐字一致。`--exec-mode stop-and-go` 跑分窗
替代模式(无 chunk 流水线)。执行机预检:
`python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs`
(相机、时延、图像方向)与
`python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1`
(RTT 分位 + 建议 k;顺带预热服务端首次 jit)。

### 实机验证清单

1. 向采集与推理 venv 各执行 `pip install -e . --no-deps`;三个命令行入口
   `--help` 全部可用。
2. `finearm-daemon --arm panthera_ht --mock` + 一级推理——无硬件。
3. 实机 daemon:握手打印 `panthera_ht 14/7`;遥测跟随主臂;停-启一轮。
4. `bash utils/lock_cameras.sh`;采集**一**条 episode;检查数据集
   (`fps=30`、帧数 = 秒数 × 30、`robot_type=panthera_ht_follower`)。
5. 服务 + 隧道;`bench_rtt` 给出 p95 与建议 k。
6. 二级(`--dry-run`)再三级(低速、`--starve stop`)推理。
7. 可选 RL 预检(无服务、无硬件):
   `python -m finearm.rl.train_stage2 --rlt_checkpoint <ckpt> --stats_path <stats> --dry_run --output_dir <空目录>`。

### 故障排查(已知坑)

- **主臂读数 999 / 无数据**——标定选错:实机必须显式传
  `--leader-config/--follower-config`
  (见 `configs/arms/panthera_ht/README.md`)。
- **daemon 杀不掉**——杀 python PID,不是 bash 包装;从臂断电会使 SDK
  重连循环中止(exit 134)——恢复供电后重启 daemon。
- **相机**——每批采集前都跑 `lock_cameras.sh`;源默认 `/dev/video4`(front)、
  `/dev/video2`(wrist),用 `PANTHERA_CAM_FRONT/WRIST` 覆盖。
- **推理握手失败**——task 字符串不一致或同源链断裂
  (stats ↔ checkpoint ↔ 数据集);报错会指名出错字段。

## 测试

```bash
# Core 套件(纯 stdlib,无需安装)
python -m pytest tests/core/

# Arms 套件(profiles、daemon mock 往返)
python -m pytest tests/arms/

# Infer 套件(执行器数学、RTC golden 对拍、stop-and-go)
python -m pytest tests/infer/

# Collect 套件(采集器注册表、dagger 占位、--arm 参数化)
python -m pytest tests/data/

# RL 套件(需 torch、numpy、matplotlib、pytest;golden 与真输入测试
# 按环境变量门控,缺机器本地输入时自动 skip)
python -m pytest tests/rl/

# 数据采集套件(需 lerobot==0.5.1、pytest)
cd plugins/panthera-lerobot-collect && python -m pytest tests/
```

CI 在每次 push 跑 lint 加全部六个套件(`.github/workflows/ci.yml`)。
CI 与测试套件永不触碰硬件。

## 许可

MIT——见 [LICENSE](LICENSE)。
