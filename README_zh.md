# finearm

[English](README.md) · [中文](README_zh.md)

VLA 微调框架:遥操作数据采集、微调、推理、以及强化学习后训练。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `arm/panthera/` | 验证过的机械臂资产 |
| `plugins/panthera-lerobot-collect/` | LeRobot 采集栈:robot/teleoperator 插件、启动 wrapper、操作文档 |
| `envs/` | 全部环境定义与 venv:`infer/`(openpi-client)、`collect/`(LeRobot 0.5.1)、`arm/`(驱动) |
| `RL/rlt_panthera/` | RL 运行产物 |
| `src/finearm/core/` | 零依赖核心:四层配置解析、仓库路径、插件注册表、norm-stats 契约 |
| `src/finearm/arms/` | 机械臂 |
| `src/finearm/infer/` | `RTCExecutor`(流水线)与 `StopAndGoExecutor`|
| `src/finearm/rl/` | vla强化学习后训练 |
| `src/finearm/data/` | 数据采集|
| `models/` | 微调(SFT):vendored 的修改版 openpi 树|
| `configs/` | 机器/臂/模型配置(权威副本) |
| `third_party/` | vendored 依赖的上游基线记录 |
| `docs/` | 笔记 |

### 拓扑

- **单机(本地推理)**:一台带 CUDA GPU 的 Linux 机器(π0.5 bf16 服务需
  ≥8 GB 显存)同时跑推理服务、臂 daemon、采集/RL 客户端。
- **执行机 + 推理服务机分离**:接臂的执行机跑 daemon/采集/RL;GPU 机跑服务
  策略。


## 全流程:采集 → 微调 → 服务 → 推理 → RL

| 环节 | 环境 | 完整文档 |
|---|---|---|
| 臂 daemon | `envs/arm/`(或任意装了厂商 SDK 的 venv) | [`docs/daemon_zh.md`](docs/daemon_zh.md) |
| 采集(teleop) | `envs/collect/`(LeRobot 0.5.1) | [`docs/collect_zh.md`](docs/collect_zh.md) · 操作手册 [`quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md) |
| 微调(SFT) | GPU 机(`uv sync`) | [`docs/sft_zh.md`](docs/sft_zh.md) |
| 服务 | GPU 机(π0.5 bf16 需 ≥8 GB) | [`docs/infer_zh.md`](docs/infer_zh.md) |
| 推理 | 执行机的 `envs/infer/` | 安全阶梯见 [`docs/infer_zh.md`](docs/infer_zh.md) |
| RL 后训练 | `envs/infer/`(+ torch) | [`docs/rl.md`](docs/rl.md) |

## 接入新机械臂

1. **daemon 插件**:建 `src/finearm/arms/<新臂>/`,声明 `ARM_ID`、`PROFILE`
   (维度/限位/特征名)、`GAINS`、`make_pair`/`make_follower_only`/`make_mock_pair`
   (对接厂商 SDK)。骨架抄 `mock_twojoint/`,实体抄 `panthera_ht/`。
   放入目录即被自动发现,`finearm-daemon --arm <新臂> --mock` 台架验证。
2. **采集插件**:仿 `plugins/panthera-lerobot-collect/` 里的
   `lerobot_robot_panthera_ht`(从臂)与 `lerobot_teleoperator_panthera_ht`(主臂)
   各写一个包,**发行名必须与插件包名一致**。
   采集入口:`finearm-collect --method teleop --arm <新臂>`(必须显式 `--arm`,
   缺省走 panthera)。
3. **推理后端**:仿 `src/finearm/infer/robot_interface.py` 写新臂版本,
   `PANTHERA_ACQ_IFACE` 指向它(或写进 `configs/local.jsonc` 的
   `acquisition_interface`)。
4. **训练配置**:按 [`docs/sft_zh.md`](docs/sft_zh.md) 四步
   (policy 适配 → DataConfig → CONFIGS 条目 → 开训)。
5. **相机档案**:新臂相机与 panthera 不同时,写自己的参数锁定脚本,
   `PANTHERA_CAM_LOCK` 指向它。

> ⚠️ **注意**  ：新机械臂的自由度数以及关节空间是否需要适配，需要自己检验

## 许可

MIT——见 [LICENSE](LICENSE)。
