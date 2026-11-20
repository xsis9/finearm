# data · 数据域

[English](README.md) · [中文](README_zh.md)

遥操作数据采集与其产出:LeRobot 采集栈(插件 + 启动器)、数据集枢纽
(`data/datasets/`)与日志。采集**代码**在 `src/finearm/data/`(入口
`finearm-collect`)。产出数据集为 **LeRobot 原生 v3**——直接可训,无需转换。

## 布局

```text
data/
├── collect/panthera-lerobot-collect/   # 采集栈:LeRobot 插件、启动器、文档、测试
│   └── .venv-lerobot/                  # 采集环境(LeRobot 0.5.1)
├── datasets/                           # 数据集落这里:lerobot/<ns>/<repo-id>/<时间戳>/
└── logs/
```

## 首次搭建

```bash
# 采集环境(LeRobot 0.5.1 + 插件 + SDK 轮子;顺带克隆/校验厂商 SDK):
./data/collect/panthera-lerobot-collect/scripts/init_workspace.sh --install
./data/collect/panthera-lerobot-collect/scripts/init_workspace.sh --check     # 校验

# daemon 驱动环境(见 arm/panthera/README_zh.md):
./arm/panthera/init_env.sh

# 采集环境装 finearm 命令行入口(一次性):
data/collect/panthera-lerobot-collect/.venv-lerobot/bin/pip install -e . --no-deps
```

## 采集(主从臂 teleop)

```bash
bash utils/lock_cameras.sh        # 每批次第一步:相机参数必须与训练/推理一致

source data/collect/panthera-lerobot-collect/.venv-lerobot/bin/activate
# (预检要求 lerobot-record 在 PATH 上——必须激活采集 venv)

finearm-collect --method teleop --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml \
    --select-cameras \
    --repo-id <namespace>/<dataset> --task "<任务指令>" \
    --episodes 1 --episode-seconds 30 --reset-seconds 60 \
    --fps 30 --hz 1000 --camera-fourcc MJPG

# 或用启动 wrapper(flag 相同,代管 venv):
#   ./data/collect/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live …
```

其余方式——freedrive 手动示教、Quest3S 遥操作——的操作手册:
[`collect/panthera-lerobot-collect/docs/quest2ros_setup.md`](collect/panthera-lerobot-collect/docs/quest2ros_setup.md)
(参数表、定标值、批次 SOP)。

## 数据集产出

- 位置:`data/datasets/lerobot/<namespace>/<repo-id>/<时间戳>/`
  (可用 `PANTHERA_DATA_ROOT` 覆盖)
- 试采后核对:`meta/info.json` → `fps=30`、`total_frames = 秒数 × 30`、
  `robot_type=panthera_ht_follower`
- `--task` 字符串随数据集存档,必须与训练 prompt 逐字一致
  (serve/infer 握手时会再校验)。

## 已知坑

- 预检报 `lerobot-record not found`——采集 venv 不在 PATH:先激活
  (见上),不要只靠绝对路径调命令行入口。
- 主臂读数 999 / 无数据——标定选错:实机必须显式传
  `--leader-config/--follower-config`。
- DAgger 采集是注册表占位(`finearm.data.collector.dagger`);调用即提示
  未实现。
