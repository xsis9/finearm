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
| `models/` | 微调(SFT):vendored 的修改版 openpi 树(上游 `215abfb` + panthera 补丁;溯源见 `third_party/`)+ 微调文档([`models/README_zh.md`](models/README_zh.md)) |
| `configs/` | 机器/臂/模型配置(权威副本) |
| `third_party/` | vendored 依赖的上游基线记录 |
| `docs/` | 工作笔记与硬件手册 |

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

## 全流程:采集 → 微调 → 服务 → 推理 → RL

每个环节的完整命令在环节自己的文档里——本页是地图。各环境按 `envs/` 下
各 README 一次性搭好;finearm 命令行入口装进 venv 的方式:
`<venv>/bin/pip install -e . --no-deps`(仓库根执行)。

| 环节 | 入口 | 环境 | 完整文档 |
|---|---|---|---|
| 臂 daemon | `bash arm/panthera/run_daemon.sh --live …`(或 `finearm-daemon --arm panthera_ht`) | `envs/arm/`(或任意装了厂商 SDK 的 venv) | [`arm/panthera/README_zh.md`](arm/panthera/README_zh.md) |
| 采集(teleop) | `finearm-collect --method teleop --live …` | `envs/collect/`(LeRobot 0.5.1) | [`data/README_zh.md`](data/README_zh.md) · 操作手册 [`quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md) |
| 微调(SFT) | `models/openpi/` 下 `uv run python scripts/train.py --config pi05_panthera_p3` | GPU 机(`uv sync`) | [`models/README_zh.md`](models/README_zh.md) |
| 服务 | `models/openpi/` 下 `uv run scripts/serve_policy.py …` | GPU 机(π0.5 bf16 需 ≥8 GB) | [`envs/infer/README_zh.md`](envs/infer/README_zh.md) |
| 推理 | `finearm-infer --model <name> --exec-mode rtc …` | 执行机的 `envs/infer/` | 安全阶梯见 [`envs/infer/README_zh.md`](envs/infer/README_zh.md) |
| RL 后训练 | `python -m finearm.rl.train_stage2 …` | `envs/infer/`(+ torch) | [`RL/rlt_panthera/README.md`](RL/rlt_panthera/README.md) |

启动 wrapper(`run_daemon.sh`、`run_panthera_lerobot_collect.sh`)仍是
推荐的实操入口——它们代管 venv 选择。数据集落 `data/datasets/`
(LeRobot 原生 v3,直接可训);数据集 → stats → checkpoint → task 的
交接契约见 [`models/README_zh.md`](models/README_zh.md)。

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
