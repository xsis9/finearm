# data · 数据域(Panthera-HT 数据采集与数据集)

`deploy/data/` 是 deploy 项目的**数据域**:数据采集工具、采集产出的数据集、
本机标定配置与日志全部在这里。工具驱动 Panthera-HT 机械臂,产出 LeRobot 原生
格式数据集(模仿学习训练直接可用,无需转换)。

**路径约定**:deploy 是项目根——命令从 deploy 目录发起,文档中的相对路径一律
按 deploy 根解析(CLI 的 `--leader-config/--follower-config/--sdk-root` 等同样
锚定 deploy 根,从任意 CWD 发起均可),代码内路径按 `__file__` 相对定位;
全项目无本机硬编码,克隆到任意机器可直接使用。

- **操作手册(权威,先读这个)**:`collect/panthera-lerobot-collect/docs/quest2ros_setup.md`
  ——三种采集方式、参数表、定标值、批次 SOP、常见问题速查
- 采集链路原理文档:`collect/panthera-lerobot-collect/docs/lerobot_native_collection.md`

## ⚠️ 采集前第一步:固定相机参数

```bash
bash deploy/utils/lock_cameras.sh
```

- 脚本把两台相机恢复到参考数据集 **bolt_test2** 采集时刻的确切参数
  (wrist 手动曝光 36/gain125、front 全 auto),逐项设置并**读回验证**,
  任一不一致自动重试,仍失败则拒绝继续
- **训练与推理的相机外观必须一致**——该脚本是采集与推理
  (`deploy/infer`)共用的唯一权威副本,放在 `deploy/utils/`,相机档案变更只改这一处
- UVC 相机重插/断电后参数会复位,所以**每次开采集前都要跑**;
  `run_panthera_lerobot_collect.sh --live` 的相机预检会自动再调一遍(双保险),
  也可单独 `--check` 只验证不设置
- `PANTHERA_CAM_LOCK=路径` 可覆盖脚本位置;`PANTHERA_CAM_LOCK_SKIP=1` 跳过(仅调试)

## 数据域布局(deploy/data)

```text
deploy/data/
├── README.md                          # 本文件(数据域总览)
├── collect/panthera-lerobot-collect/  # 采集工具(代码 + 自己的 .venv-lerobot + docs + tests)
├── datasets/                          # 采集数据集落这里(lerobot/<namespace>/<repo-id>/<时间戳>/)
└── logs/                              # daemon/diagnostics 日志

机械臂相关资产(SDK、robot_param 本机标定)统一在 <deploy>/arm/panthera 下,不在本数据域。
```

## 三种采集方式

| 方式 | 终端数 | 一句话 |
|---|---|---|
| ① 主从臂 teleop(推荐) | 1 | 手持主臂,从臂 1000Hz 跟随,wrapper 自动拉起 daemon |
| ② 手动示教 freedrive | 2 | 手拖臂 + 手捏夹爪 |
| ③ Quest3S 遥操作 | 4 | endpoint + daemon + bridge + collect |

```bash
cd deploy   # 项目根;下面所有相对路径一律按 deploy 根解析(从任意 CWD 发起均可)

# 方式①:主从臂(1 个终端)
./data/collect/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live \
  --leader-config arm/panthera/robot_param/Leader.yaml \
  --follower-config arm/panthera/robot_param/Follower.yaml \
  --select-cameras \
  --repo-id bolt/test3 --task "pick up the bolt and place it on the nut" \
  --episodes 1 --episode-seconds 30 --reset-seconds 60 \
  --fps 30 --hz 1000 \
  --teleop-gripper-squeeze-boost 1.0 --camera-fourcc MJPG
```

(`--leader-config/--follower-config/--sdk-root` 等相对路径一律按 deploy 根解析,
无本机硬编码;`--sdk-root` 缺省即 `arm/panthera/sdk/Panthera-HT_SDK`,无需再传;
方式②③的逐终端命令、参数含义、安全须知见操作手册 quest2ros_setup.md。)

## 首次环境搭建

```bash
cd deploy
./data/collect/panthera-lerobot-collect/scripts/init_workspace.sh --install
                                       # 建采集 venv .venv-lerobot + 装 LeRobot 0.5.1 + SDK 电机 wheel
                                       # (SDK 缺失时自动从 GitHub clone 并锁 revision,落在 deploy/arm/panthera/sdk)
./data/collect/panthera-lerobot-collect/scripts/init_workspace.sh --check   # 校验
./arm/panthera/init_env.sh                      # 建机械臂驱动环境 arm/.venv(python3.12 + SDK 轮子 + SDK Python 依赖)
                                       # (daemon 用,与采集/推理环境三者解耦;缺失时会回落采集 venv)
```

- **零配置**:wrapper 不 `source .env` 也能跑——SDK/数据/venv 位置全部按目录结构自动定位。
  手册中直接引用 `"${PANTHERA_PYTHON}"` / `"${PANTHERA_SDK_ROOT}"` 的片段,按手册
  `set -a; source .env; set +a` 即可(机器 `.env` 在工具目录内,可选文件,发布不含)。
- venv 只认工具内 `.venv-lerobot`,**不会**回退到外部环境;找不到时报错并提示先跑 install。
- 相机预检依赖 `v4l2-ctl`(`sudo apt install v4l-utils`);fps 门禁工具
  `~/camera_latency_test/v4l2_latency` 缺失时自动跳过(仅警告)。

## 采集工具目录结构(collect/panthera-lerobot-collect)

```text
run_panthera_lerobot_collect.sh   # 采集 wrapper(preflight + 相机锁定 + 拉起 lerobot-record)
run_panthera_lerobot_daemon.sh    # daemon wrapper → 转发 deploy/arm/panthera/run_daemon.sh(驱动层权威入口)
quest_bridge.py                   # Quest3S → daemon 遥操作桥(IK/FK,方式③)
panthera_lerobot_control/         # 核心:daemon(controller/cli/ipc/adapters)+ 采集编排(collect_cli)
lerobot_robot_panthera_ht/        # LeRobot Robot 插件(panthera_ht_follower,14 维 state + 相机)
lerobot_teleoperator_panthera_ht/ # LeRobot Teleoperator 插件(panthera_ht_leader,7 维 action)
scripts/init_workspace.sh         # 环境初始化(SDK revision + wheel + .venv-lerobot)
docs/quest2ros_setup.md           # 操作手册(三种方式,权威文档)
docs/lerobot_native_collection.md # 原生采集链路手册
tests/                            # 单测(python3 -m unittest discover -s tests)
```

## 与旧项目 / 其他域的关系

- **与 `project-Embodied_AGV` 零依赖**:代码、配置、venv、数据全部自包含;旧项目仅作为
  历史出处存在。旧数据集仍在 `~/panthera/data/lerobot`(归旧项目/训练用),本工具新采的数据
  一律落 `<deploy>/data/datasets/lerobot/`。
- **机械臂相关资产统一在 `deploy/arm/panthera/`**(2026-11 抽离并归拢):daemon 由
  `deploy/arm/run_daemon.sh` 启动、用独立的 `arm/.venv`;SDK 实体在 `arm/panthera/sdk/`,
  本机标定在 `arm/panthera/robot_param/`;工具内旧命令 `./run_panthera_lerobot_daemon.sh`
  等价转发,深手册中的旧写法继续有效。
- **相机参数档案**:`deploy/utils/lock_cameras.sh` 是采集与推理共用的**唯一权威副本**
  (采集 wrapper 与推理端启动时都自动调用)。
- **HG-Dagger 纠错采集方式重整中**(2026-11-12 清空旧脚本,`data/collect/Dagger/` 已删除);
  机械臂/IPC 基建不受影响,重整后的采集方式另行文档化。
- Quest 链路的 ROS 侧(`ros_tcp_endpoint`、quest2ros App)在 `~/quest_ws`,不在本目录,
  见手册第三大部分 3.5。

## 验证(无硬件)

```bash
# 终端1:mock daemon(驱动层)
./arm/panthera/run_daemon.sh --mock --hz 1000
# 终端2:检查将执行的采集命令(不动硬件;--dataset.root 应指向 deploy/data/datasets)
./data/collect/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --mock --no-cameras \
  --repo-id local/smoke --task "mock smoke test" \
  --episodes 1 --episode-seconds 2 --reset-seconds 0 --print-command

# 单测
cd data/collect/panthera-lerobot-collect && python3 -m unittest discover -s tests -v
```
