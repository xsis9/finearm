# Panthera-HT LeRobot 原生采集使用手册

> 本文档属于独立数据采集工具 `panthera-lerobot-collect`(2026-11-10 从
> `project-Embodied_AGV` 抽离)。SmolVLA/pi0.5 训练与安全 policy inference
> 仍在主项目 `~/internship/baiqin/project-Embodied_AGV`,见其
> `docs/smolvla_training_inference.md`。

这份文档是 Panthera-HT 当前推荐的数据采集入口。核心原则是把实时控制和数据集工程拆开：

- Panthera daemon 只负责 1000 Hz leader/follower 底层控制、状态快照和 IPC。
- LeRobot 负责相机、episode 流程、dataset 保存、视频编码、resume 和训练兼容性。
- 本文聚焦 teleoperation collection(policy inference 见主项目)。

旧的 `panthera_teleop_recorder.py`(legacy/debug 工具,离线格式验证、Rerun 调试)保留在主项目 `project-Embodied_AGV`;本工具只包含 LeRobot-native 采集链路。

## 当前已验证配置

以下配置于 2026-07-10 在当前工作站完成实机验收，是本文正式命令的基线：

```bash
set -a
source .env
set +a
export FRONT_CAMERA=/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0
export WRIST_CAMERA=/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0
```

- Python 3.12、LeRobot 0.5.1 和 Panthera SDK v4.6.0 可用。
- HOLD-only 和 10 秒 teleop 均无 `last_error`，控制 loop 接近 1000 Hz。
- 双相机 10 秒短采得到 `298/300` 帧；两路视频均为 640x480、30 FPS、9.93 秒。
- `observation.state` 为 14 维，`action` 为 7 维，全部为有限值。
- front/wrist 构图已人工确认，结束后 confirmed HOLD、SDK brake、进程和 socket 清理正常。

`/dev/videoN` 会随重插或重启变化，两个节点也可能属于同一物理设备。正式采集使用上面的 `by-id` 路径；更换相机后重新执行相机枚举和短采验收。

软件 HOLD 和 `Ctrl+C` 不是硬件急停。正式运行前必须明确可触达的物理断电方式，并清空机械臂和夹爪工作区。

## 正式采集快速开始

首次采集不要传 `--dataset-root`，wrapper 会创建唯一 timestamp root：

```bash
cd plugins/panthera-lerobot-collect

set -a
source .env
set +a
export FRONT_CAMERA=/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0
export WRIST_CAMERA=/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0

./run_panthera_lerobot_collect.sh --live \
  --camera "front=${FRONT_CAMERA}" \
  --camera "wrist=${WRIST_CAMERA}" \
  --repo-id local/pick_red_cube_into_bowl \
  --task "pick up the red cube and place it in the bowl" \
  --episodes 50 \
  --episode-seconds 60 \
  --reset-seconds 60 \
  --fps 30 \
  --hz 1000 \
  --sdk-root "${PANTHERA_SDK_ROOT}"
```

正式命令有意不使用 `--auto-start`、`--no-status-ui` 或 `--no-release-prompt`。preflight 在 daemon 前检查环境、dataset、相机实际读帧和串口；Preview 阶段确认现场安全后再开始录制。失败 episode 按左方向键 re-record，不要用 `Esc` 丢弃，因为 partial episode 可能被保存。

结束时 wrapper 等待新 snapshot 确认 `holding=true`、`control_mode=hold`。owned daemon 只有在 confirmed HOLD 后才提示扶住机械臂并停止；外部 daemon 默认保持 HOLD 并继续运行。

fresh root 格式：

```text
${PANTHERA_DATA_ROOT}/lerobot/<namespace>/<name>/YYYYMMDD_HHMMSS_ffffff_p<PID>_c<counter>
```

### 续录已有数据集

只有明确的 `--resume --dataset-root` 才会 append。下面示例追加 10 个 episode：

```bash
export DATASET_ROOT=${PANTHERA_DATA_ROOT}/lerobot/local/pick_red_cube_into_bowl/具体时间戳目录

./run_panthera_lerobot_collect.sh --live \
  --camera "front=${FRONT_CAMERA}" \
  --camera "wrist=${WRIST_CAMERA}" \
  --repo-id local/pick_red_cube_into_bowl \
  --task "pick up the red cube and place it in the bowl" \
  --episodes 10 \
  --episode-seconds 60 \
  --reset-seconds 60 \
  --fps 30 \
  --hz 1000 \
  --sdk-root "${PANTHERA_SDK_ROOT}" \
  --resume \
  --dataset-root "${DATASET_ROOT}"
```

resume 在机械臂启动前严格检查 root/meta 可写性、robot type、FPS、14/7 feature dtype/整数 shape/names、唯一 task 和 Arrow integer `task_index=0`。`repo-id`、`task` 和 `fps` 必须与原数据集完全一致；否则创建新的 fresh dataset。

## 目录

1. [当前已验证配置](#当前已验证配置)
2. [正式采集快速开始](#正式采集快速开始)
3. [整体架构](#整体架构)
4. [环境安装](#环境安装)
5. [硬件启动前检查](#硬件启动前检查)
6. [Mock smoke test](#mock-smoke-test)
7. [采集命令详解](#采集命令详解)
8. [停止和异常处理](#停止和异常处理)
9. [验收 checklist](#验收-checklist)
10. [常见问题](#常见问题)
11. [开发验证命令](#开发验证命令)

## 整体架构

运行时分成两个主要进程：

```text
Terminal A
  panthera-control-daemon
    - 连接 Panthera leader/follower
    - 运行 1000 Hz teleop control loop
    - 维护最新 observation/action snapshot
    - 通过 Unix socket 提供 status/snapshot/hold/set_mode/policy_action/stop

Terminal B
  lerobot-record
    - 加载 panthera_ht_follower robot plugin
    - 加载 panthera_ht_leader teleoperator plugin
    - 读取 daemon snapshot
    - 管理相机、episode、dataset、视频编码和 resume
```

daemon 不做这些事情：

- 不打开相机。
- 不创建 episode。
- 不写 LeRobotDataset。
- 不编码视频。
- 不启动 Rerun。
- 不导入 `panthera_teleop_recorder.py`。

LeRobot 不直接控制 Panthera follower。采集循环里的 `robot.send_action(action)` 只做 action 校验并原样返回；实际 follower 控制已经由 daemon 的 1000 Hz loop 完成。

## 代码入口

主要文件：

```text
panthera_lerobot_control/
  cli.py              # daemon CLI
  controller.py       # 控制 loop 和生命周期
  adapters.py         # Panthera SDK/mock adapter
  ipc.py              # Unix socket JSON-lines server/client
  constants.py        # state/action schema 和控制参数
  collect_cli.py      # 一条命令采集编排器，只负责 preflight/launch/hold

lerobot_robot_panthera_ht/
  __init__.py         # LeRobot Robot plugin: panthera_ht_follower

lerobot_teleoperator_panthera_ht/
  __init__.py         # LeRobot Teleoperator plugin: panthera_ht_leader

quest_bridge.py       # Quest3S 遥操作桥(第三种采集方式,见 quest2ros_setup.md)
run_panthera_lerobot_collect.sh
scripts/
  init_workspace.sh   # SDK + venv + LeRobot 0.5.1 环境初始化
  lock_cameras.sh     # 双相机参数锁定/验证(与推理端共用同一档案)
requirements-lerobot.txt
docs/
tests/
```

LeRobot 类型名：

```text
robot.type  = panthera_ht_follower
teleop.type = panthera_ht_leader
```

默认 socket：

```text
/tmp/panthera_lerobot_control.sock
```

## 环境安装

建议使用干净虚拟环境，避免 ROS、系统 Python、旧 recorder 环境和 LeRobot 环境互相污染。

```bash
cd plugins/panthera-lerobot-collect
./scripts/init_workspace.sh --install
```

采集链路按 `lerobot==0.5.1` 验证并固定版本。launcher 只会选择同时包含可执行 `python`、同环境 `lerobot-record` 和 LeRobot 0.5.1 的虚拟环境；显式 `PANTHERA_PYTHON` 不满足这三项时直接失败，不会静默回退。

初始化脚本会先验证独立 SDK 仓库的固定 revision，再选择与 Python 3.12 和 x86_64 匹配的 `hightorque_robot` wheel；`requirements-lerobot.txt` 不包含任何本机绝对路径。

安装后检查插件能被导入：

```bash
python - <<'PY'
from lerobot_robot_panthera_ht import PantheraHTRobotConfig
from lerobot_teleoperator_panthera_ht import PantheraHTTeleoperatorConfig

print(PantheraHTRobotConfig().type)
print(PantheraHTTeleoperatorConfig().type)
PY
```

期望输出：

```text
panthera_ht_follower
panthera_ht_leader
```

## 硬件启动前检查

每次连接实机前先检查：

```bash
ls /dev/ttyACM*
stat -c '%A %U %G %n' /dev/ttyACM*
```

如果串口权限不对，临时处理：

```bash
sudo chmod a+rw /dev/ttyACM*
```

永久处理是把当前用户加入 `dialout` 组，然后注销并重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

如果 collect wrapper 启动 daemon 后报 socket timeout，它会打印 `Daemon log: /tmp/panthera_lerobot_daemon_....log`，并在 daemon 退出或超时时显示日志尾部。优先看日志里的串口、SDK 或 YAML 错误。

确认：

- Leader 主臂和 Follower 从臂串口/serial 配置没有换。
- `${PANTHERA_SDK_ROOT}/panthera_python/robot_param/Leader.yaml` 和 `Follower.yaml` 匹配当前硬件。
- 两个机械臂工作空间清空。
- 急停、断电、Ctrl+C 退出路径明确。
- 相机视角固定，能覆盖完整任务过程。
- 先完成 [Mock smoke test](#mock-smoke-test)，再接实机。

## Mock smoke test

Mock 模式不连接机械臂，用来验证 daemon、IPC 和 LeRobot plugin 基本链路。

Terminal A 启动 daemon：

```bash
bash arm/panthera/run_daemon.sh --mock --hz 1000    # 仓库根执行
```

Terminal B 检查 status 和 snapshot：

```bash
cd plugins/panthera-lerobot-collect
python - <<'PY'
from panthera_lerobot_control import ControlClient

client = ControlClient('/tmp/panthera_lerobot_control.sock')
status = client.status()
snapshot = client.snapshot(max_age_s=0.5)

print(status)
print(len(snapshot['observation.state']), len(snapshot['action']))
PY
```

期望：

```text
mode: mock
running: True
observation.state length: 14
action length: 7
```

再检查 LeRobot adapter 能读取 daemon：

```bash
python - <<'PY'
from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
from lerobot_teleoperator_panthera_ht import PantheraHTTeleoperator, PantheraHTTeleoperatorConfig

robot = PantheraHTRobot(PantheraHTRobotConfig())
teleop = PantheraHTTeleoperator(PantheraHTTeleoperatorConfig())
robot.connect()
teleop.connect()

obs = robot.get_observation()
action = teleop.get_action()

print(len(obs), len(action), robot.cameras)
PY
```

期望输出：

```text
14 7 {}
```

## 采集命令详解

正式采集使用前面的[正式采集快速开始](#正式采集快速开始)。需要更换或重新识别相机时，可用 GUI 交互选择：

```bash
cd plugins/panthera-lerobot-collect
./run_panthera_lerobot_collect.sh --live \
  --select-cameras \
  --repo-id local/panthera_smolvla_demo \
  --task "pick up the red cube and place it in the bowl" \
  --episodes 50 \
  --episode-seconds 60 \
  --reset-seconds 60 \
  --fps 30 \
  --hz 1000
```

这条命令做的事情：

- 使用通过 LeRobot 0.5.1 校验的项目 Python 环境，清理 ROS 注入的 `PYTHONPATH`。
- 在串口权限变更和 daemon 启动前完成 repo、dataset root、resume metadata 和相机配置预检。
- 对每个配置相机执行有界实际读帧；选择结果保存到 `.panthera_camera_selection.json`。两个角色不能解析到同一物理相机。
- 如果 socket 上没有现成 daemon，自动启动 `panthera-control-daemon`。
- 启动 `lerobot-record` 前检查 `status` 和 fresh `snapshot`，确认 daemon mode、sequence 和 socket 正常。
- 默认进入 Preview；这时 daemon 已经在 1000 Hz teleop，但 LeRobot 还没开始写 episode。确认机械臂和相机后按 Enter 或 GUI 的 `Start recording`。
- 正式进入 `lerobot-record` 后会打开常驻 Tk 状态指示器，明确显示 `RECORDING`、`RESET`、`STOPPING`、`DONE` 等阶段。它只做状态提示，不接管 LeRobot 原生快捷键。
- LeRobot 的文字转语音默认关闭，wrapper 根据阶段日志发出更清晰的终端蜂鸣：进入 episode 两声，进入 reset 或停止时一声。需要保留 LeRobot 语音时加 `--voice-prompts`。
- `lerobot-record` 退出或被中断后，先发送 IPC `hold`，提示用户扶住机械臂；如果 daemon 是本命令启动的，再按 Enter 后发送 `stop`。如果是外部已有 daemon，则默认只 hold、不自动 stop。
- 未显式指定 `--dataset-root` 的 fresh run 使用 `PANTHERA_DATA_ROOT/lerobot/<namespace>/<name>/YYYYMMDD_HHMMSS_ffffff_p<PID>_c<counter>`；微秒、进程 ID 和进程内计数避免并发启动选中同一路径，已有碰撞仍追加 `_001`、`_002`。
- 显式 fresh root 已存在时直接拒绝。只有明确提供 `--resume --dataset-root PATH`，且 root/meta/metadata 可写、robot type、FPS、严格整数 shape、14/7 schema 和单任务 metadata 全部匹配时才允许续录。

如果已经知道相机路径，可以不用 GUI：

```bash
./run_panthera_lerobot_collect.sh --live \
  --camera front=/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0 \
  --camera wrist=/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0 \
  --repo-id local/panthera_smolvla_demo \
  --task "pick up the red cube and place it in the bowl"
```

只做 mock/state smoke test，不接相机和机械臂：

```bash
./run_panthera_lerobot_collect.sh --mock --auto-start --no-cameras \
  --repo-id local/panthera_mock_smoke \
  --episodes 1 \
  --episode-seconds 5 \
  --reset-seconds 0
```

想查看底层 LeRobot 命令但不启动任何进程：

```bash
./run_panthera_lerobot_collect.sh --mock --print-command \
  --camera front=/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0 \
  --camera wrist=/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0 \
  --repo-id local/panthera_smolvla_demo
```

常用参数：

```text
--live / --mock                   连接真实硬件或使用 mock；默认 mock
--hz 1000                         daemon 控制 loop 目标频率
--socket PATH                     daemon、robot 和 teleop 共用 Unix socket
--sdk-root PATH                   Panthera SDK panthera_python 路径
--leader-config / --follower-config PATH  覆盖默认硬件 YAML
--camera NAME=SOURCE              显式相机映射，可重复
--select-cameras                  GUI 选择并缓存相机；不能和 --print-command/--no-gui 同用
--camera-test-timeout-s 3         单次相机预检的最长等待时间
--image-width / --image-height    LeRobot 相机分辨率，默认 640x480
--no-cameras                      state/action only，仅用于 mock 或诊断
--auto-start                      跳过 Preview，直接启动 lerobot-record
--no-gui / --no-status-ui         禁用 Preview GUI 或录制状态窗口
--voice-prompts                   保留 LeRobot 原生文字转语音提示
--print-command                   只打印 argv，不启动相机、daemon 或 recorder
--dataset-root PATH               fresh 时必须不存在；resume 时必须已存在且匹配
--resume                          明确续录；必须同时给 --dataset-root
--push-to-hub                     完成后上传数据集；默认只保存在本地
--timeout-s 1                     IPC 请求超时，并传给 robot/teleop plugin
--stale-timeout-s 0.2             snapshot 最大允许年龄
--startup-timeout-s 5             daemon readiness 等待时间
--stop-external-daemon            允许停止启动前已经存在的 daemon
--no-release-prompt               confirmed HOLD 后不等待 Enter，自动 stop owned daemon
--record-graceful-timeout-s 30    recorder SIGINT 优雅退出窗口
--record-terminate-timeout-s 5    recorder SIGTERM 退出窗口
--lerobot-arg ARG                 追加非受保护的 lerobot-record 参数
```

`--no-cameras` 只建议用于 mock 或诊断。正式视觉模仿学习数据必须带相机。

`--lerobot-arg` 不允许覆盖 wrapper 管理的 `--robot.*`、`--teleop.*`、`--dataset.*` 或 `--resume`，避免绕过安全和数据校验。

## 实机启动 daemon

Terminal A：

```bash
bash arm/panthera/run_daemon.sh --live --hz 1000    # 仓库根执行
```

`--live` 表示连接真实 Panthera 硬件。不写 `--live` 时 wrapper 默认加 `--mock`，避免误启动硬件。

`--hz 1000` 会翻译成 daemon CLI 的 `--loop-hz 1000`。

常用参数：

```text
--live                         连接真实硬件
--mock                         不连接硬件，使用 mock adapter
--hz 1000                      控制 loop 目标频率，转成 --loop-hz
--socket PATH                  覆盖 Unix socket 路径
--sdk-root PATH                覆盖 Panthera SDK panthera_python 路径
--leader-config PATH           覆盖 Leader.yaml
--follower-config PATH         覆盖 Follower.yaml
```

示例：

```bash
bash arm/panthera/run_daemon.sh --live --hz 1000 \
  --socket /tmp/panthera_lerobot_control.sock \
  --sdk-root ${PANTHERA_SDK_ROOT}/panthera_python
```

daemon 启动后，可以在另一个终端查询：

```bash
python - <<'PY'
from panthera_lerobot_control import ControlClient

client = ControlClient('/tmp/panthera_lerobot_control.sock')
print(client.status())
PY
```

正常时应看到：

```text
running: True
mode: live
sequence: 持续增长
last_error: None
```

## 运行 LeRobot record

这是透明的底层手动路径，主要用于调试 collect wrapper 或复现 LeRobot 官方参数。日常采集优先使用上一节的一条命令入口。

Terminal B，在 daemon 已经运行后执行：

```bash
cd plugins/panthera-lerobot-collect
lerobot-record \
  --robot.type=panthera_ht_follower \
  --robot.id=panthera_follower \
  --robot.socket_path=/tmp/panthera_lerobot_control.sock \
  --robot.timeout_s=1 \
  --robot.stale_timeout_s=0.2 \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0","width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":"/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0","width":640,"height":480,"fps":30}}' \
  --teleop.type=panthera_ht_leader \
  --teleop.id=panthera_leader \
  --teleop.socket_path=/tmp/panthera_lerobot_control.sock \
  --teleop.timeout_s=1 \
  --teleop.stale_timeout_s=0.2 \
  --dataset.repo_id=local/panthera_smolvla_demo \
  --dataset.root=${PANTHERA_DATA_ROOT}/lerobot/local/panthera_smolvla_demo/20260710_143205_123456_p4242_c000000 \
  --dataset.push_to_hub=False \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=60 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.single_task="pick up the red cube and place it in the bowl" \
  --dataset.fps=30 \
  --resume=false
```

参数说明：

```text
--robot.type
  使用 Panthera follower state adapter。

--robot.socket_path
  robot adapter 连接 daemon 的 socket。必须和 daemon --socket 一致。

--resume
  默认使用 false 开始 timestamped fresh 数据集。wrapper 续录必须使用 `--resume --dataset-root PATH`；root/meta/info/tasks 必须可写，且已有目录的 robot type、FPS、14/7 feature names/整数 shape/dtype、Arrow integer task_index 和唯一 task 必须完全匹配。

--robot.cameras
  严格 JSON camera config。wrapper 在串口和 daemon 前先有界读帧，Panthera robot plugin 在每次 observation 中读取并返回相机帧。

--teleop.type
  使用 Panthera leader-derived action adapter。

--teleop.socket_path
  teleoperator adapter 连接 daemon 的 socket。必须和 daemon --socket 一致。

--dataset.repo_id
  数据集 ID。离线本地采集可用 local/...。

--dataset.push_to_hub=False
  本地调试和首次实机建议关闭上传。

--dataset.num_episodes
  采集 episode 数。正式任务建议从 50 条开始。

--dataset.episode_time_s / --dataset.reset_time_s
  每条 episode 和 reset 窗口时长。

--dataset.streaming_encoding=true / --dataset.encoder_threads=2
  让 LeRobot 负责视频流式编码，避免采集结束后堆积太多编码工作。

--dataset.fps=30
  LeRobot 采样频率和相机目标帧率。
```

本地可以调整：

- `/dev/v4l/by-id/usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0`、`/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0` 按实际相机映射修改。
- `width`、`height`、`fps` 按 USB 带宽和任务需要修改。
- `--dataset.repo_id` 按本机命名规范修改。
- `--dataset.single_task` 必须和采集动作一致。

不要随意修改：

- `--robot.type=panthera_ht_follower`
- `--teleop.type=panthera_ht_leader`
- robot/teleop 的 `socket_path`，除非 daemon 也同步改了 `--socket`。

## 采集流程建议

首次实机建议按这个顺序：

1. 不接 LeRobot，只启动 daemon：

   ```bash
   bash arm/panthera/run_daemon.sh --live --hz 1000
   ```

   观察 2 分钟，确认主从臂控制稳定。

2. 查询 status：

   ```bash
   python - <<'PY'
   from panthera_lerobot_control import ControlClient
   print(ControlClient('/tmp/panthera_lerobot_control.sock').status())
   PY
   ```

3. 做短采集：

   ```text
   --dataset.num_episodes=2
   --dataset.episode_time_s=10
   --dataset.reset_time_s=10
   ```

4. 检查 LeRobot dataset：

   - episode 数是否正确。
   - front/wrist 视频是否清晰稳定。
   - `observation.state` 是否 14 维。
   - `action` 是否 7 维。
   - action/state 曲线是否连续，没有明显跳变。

5. 再切回正式采集：

   ```text
   --dataset.num_episodes=50
   --dataset.episode_time_s=60
   --dataset.reset_time_s=60
   ```

采集中如果当前 episode 任务失败、遮挡严重或相机画面异常，应在该 episode 提交前按左方向键触发 LeRobot 的 **re-record**，清空当前 episode buffer 后重录。

不要把 `Esc` 当作“丢弃当前 episode”：LeRobot 0.5.1 的正常停止路径可能保存当前 partial episode。需要停止整个 run 时，先用左方向键重录/清空失败 episode，再在安全时机停止；不要把失败样本混进成功任务描述。

## Snapshot schema

daemon 的 `snapshot` 响应是 JSON object，核心字段：

```text
observation.state: 14 floats
  follower_joint_1.pos
  follower_joint_2.pos
  follower_joint_3.pos
  follower_joint_4.pos
  follower_joint_5.pos
  follower_joint_6.pos
  follower_gripper.pos
  follower_joint_1.vel
  follower_joint_2.vel
  follower_joint_3.vel
  follower_joint_4.vel
  follower_joint_5.vel
  follower_joint_6.vel
  follower_gripper.vel

action: 7 floats
  target_joint_1.pos
  target_joint_2.pos
  target_joint_3.pos
  target_joint_4.pos
  target_joint_5.pos
  target_joint_6.pos
  target_gripper.pos

timestamp: monotonic timestamp
sequence: increasing integer
diagnostics: mode, holding, torque feedback, commanded torque, last_error
loop_timing: target_period_s, last_period_s, mean_period_s
```

`observation.state` 是 follower 当前状态；`action` 是 leader 映射后发给 follower 的目标位置。重力补偿、摩擦补偿和力矩限幅只作为 daemon 内部控制和 diagnostics，不作为 LeRobot 监督动作。

LeRobot robot adapter 返回的是命名后的 flat dict：

```text
{
  "follower_joint_1.pos": ...,
  ...
  "follower_gripper.vel": ...
}
```

LeRobot teleoperator adapter 返回：

```text
{
  "target_joint_1.pos": ...,
  ...
  "target_gripper.pos": ...
}
```

## IPC requests

IPC 使用 Unix socket JSON-lines。默认 socket 可用 wrapper 的 `--socket` 覆盖。

请求格式：

```json
{"request": "status"}
{"request": "snapshot"}
{"request": "hold"}
{"request": "stop"}
```

响应格式：

```json
{"ok": true, "type": "status", "status": {...}}
{"ok": true, "type": "snapshot", "snapshot": {...}}
{"ok": true, "type": "hold", "status": {...}}
{"ok": true, "type": "stop"}
```

未知请求、超时、过大的 request line 或非法 JSON 会返回：

```json
{"ok": false, "error": "..."}
```

手动发 stop：

```bash
python - <<'PY'
from panthera_lerobot_control import ControlClient
print(ControlClient('/tmp/panthera_lerobot_control.sock').stop())
PY
```

手动发 hold：

```bash
python - <<'PY'
from panthera_lerobot_control import ControlClient
print(ControlClient('/tmp/panthera_lerobot_control.sock').hold())
PY
```

## 停止和异常处理

正常停止顺序：

1. 让 `lerobot-record` 正常结束当前 episode。
2. wrapper 先向 daemon 发送 IPC `hold`。
3. wrapper 等待 daemon `status` 报告 `holding=True`，并确认 sequence 在 hold 后更新。
4. 只有 confirmed hold 后，wrapper 才会按提示或配置发送 IPC `stop`。
5. daemon 收到 `stop` 后会再次进入 hold，等待运行中的 control loop 产生 holding tick；停止 loop 后，再同步发送一次 follower 当前姿态 hold command，最后才 stop adapter。

LeRobot 退出时：

- collect wrapper 启动的 owned daemon：wrapper 请求 HOLD，等待更新后的 holding snapshot，提示扶住机械臂后再发送 stop。
- 已经存在的 external daemon：wrapper 默认只请求 HOLD，不擅自停止；只有显式 `--stop-external-daemon` 才允许 stop。
- 如果 HOLD 未确认，wrapper 跳过自动 stop、返回非零安全错误，并把 daemon 留在运行状态；先物理处理现场，再检查 PID、dataset root、daemon 日志和 `last_error`。
- recorder 退出不等于电机断电。confirmed HOLD 依赖 daemon、串口和控制 loop；真正急停仍使用物理断电。

daemon 退出时：

- LeRobot adapter 下一次读取 snapshot/action 会收到 socket 断开或 stale snapshot 错误。
- record 流程应停止，而不是继续写入伪数据。
- 当前 Panthera SDK 仍可能在 `set_brake()` 或对象析构时进入 brake 路径；本项目的安全边界是先保持 active hold，再进入 SDK stop/brake。SDK 配置里的 `motor_timeout_ms: 0` 表示无指令自动阻尼 fallback 关闭，实机修改该配置前必须单独验证。

daemon 控制 loop 内部出错时：

- `last_error` 会写入 diagnostics/status。
- daemon 会设置 holding 状态。
- follower target 会尽量切到当前位置保持，速度目标为 0。

IPC `hold` 请求：

- 设置 holding 状态。
- 后续 target 采用 follower 当前状态。
- 不等价于完全停机；仍应按现场安全流程确认机械臂状态。

## 验收 checklist

### Mock 验收

- `bash arm/panthera/run_daemon.sh --mock --hz 1000`(仓库根)能启动。
- `ControlClient.status()` 返回 `mode=mock`、`running=True`。
- `ControlClient.snapshot()` 返回 14 维 `observation.state` 和 7 维 `action`。
- `PantheraHTRobot.get_observation()` 返回 14 个命名 key。
- `PantheraHTTeleoperator.get_action()` 返回 7 个命名 key。

### 实机 daemon 验收

- [x] HOLD-only 连续运行 10 秒，sequence 持续增长，`last_error` 为空。
- [x] 10 秒 teleop 跟踪稳定，最大观测跟踪误差约 0.0251 rad。
- [x] 结束时 confirmed HOLD，SDK brake，进程和 socket 无残留。
- [ ] 长时间正式批采前，再做一次不少于 2 分钟的现场 soak test。

### LeRobot 短采验收

- [x] 完成 1 条 10 秒 front/wrist 实机 episode。
- [x] 生成 298 帧 state/action 和两路 298 帧视频，目标 300 帧，约 0.67% 帧差。
- [x] state/action 全部有限，shape 分别为 `(298, 14)` 和 `(298, 7)`。
- [x] 两路视频均为 640x480、30 FPS、9.93 秒，构图和角色命名已人工确认。
- [x] 采集后 confirmed HOLD，daemon、recorder 和 socket 均无残留。
- [ ] 正式批采前演练一次中断路径，确认 re-record/partial-episode 处理符合现场流程。

### 正式采集前

- 任务语言固定、明确。
- 每个任务变化有足够重复样本。
- 成功样本和失败样本不要混在同一个任务描述下。
- 每次正式采集使用明确的 dataset repo_id。

## 常见问题

### `lerobot-record` 找不到 `panthera_ht_follower`

确认当前环境安装了本仓库：

```bash
python -m pip install -e .
```

再检查：

```bash
python - <<'PY'
import lerobot_robot_panthera_ht
import lerobot_teleoperator_panthera_ht
from lerobot_robot_panthera_ht import PantheraHTRobotConfig
from lerobot_teleoperator_panthera_ht import PantheraHTTeleoperatorConfig
print(PantheraHTRobotConfig().type)
print(PantheraHTTeleoperatorConfig().type)
PY
```

### adapter 报 `not connected`

这是正常保护。LeRobot 应先调用 `connect()`。手动脚本测试时也需要：

```python
robot.connect()
teleop.connect()
```

### adapter 报 stale snapshot

常见原因：

- daemon 没启动。
- socket 路径不一致。
- daemon 控制 loop 出错并停止更新 snapshot。
- 系统负载过高，snapshot 年龄超过 `stale_timeout_s`。

检查：

```bash
python - <<'PY'
from panthera_lerobot_control import ControlClient
client = ControlClient('/tmp/panthera_lerobot_control.sock')
print(client.status())
print(client.snapshot())
PY
```

### socket path already served

说明已有 daemon 正在占用 socket。先确认是否已有进程：

```bash
python - <<'PY'
from panthera_lerobot_control import ControlClient
print(ControlClient('/tmp/panthera_lerobot_control.sock').status())
PY
```

如果确认是旧进程，先停止旧 daemon，再启动新的。

### `A module that was compiled using NumPy 1.x` / `cv2` 导入失败

这是 Python 环境混用了 user-site 包导致的 NumPy/OpenCV ABI 冲突。collect launcher 只选择带同环境 `lerobot-record` 且版本为 LeRobot 0.5.1 的环境，并设置 `PYTHONNOUSERSITE=1`，避免从 `~/.local` 导入错误版本。

如果仍然遇到这个错误，先确认 wrapper 实际使用的 Python：

```bash
bash -x ./run_panthera_lerobot_collect.sh --mock --print-command --no-cameras --repo-id local/check
```

正常应看到类似：

```text
PYTHON=${PANTHERA_PYTHON}
PYTHONNOUSERSITE=1
```

再检查 venv 内的 NumPy/OpenCV：

```bash
${PANTHERA_PYTHON} - <<'PY'
import numpy, cv2
print(numpy.__version__, numpy.__file__)
print(cv2.__version__, cv2.__file__)
PY
```

### 相机打不开

相机由 LeRobot 负责。先单独确认设备路径：

```bash
ls -l /dev/video*
ls -l /dev/v4l/by-id/
```

当前验证路径见[当前已验证配置](#当前已验证配置)。`video-index0` 是视频采集节点，`video-index1` 通常是 metadata 节点。

不要仅凭 `/dev/videoN` 编号分配 front/wrist；先用 `v4l2-ctl --list-devices` 确认它们属于不同物理设备，再各读取一帧。更换相机后重新完成 10 秒短采并人工确认构图。

### 采集时 follower 还在动，但 LeRobot 已退出

daemon 和 LeRobot 是独立进程，但 collect wrapper 会先请求 hold，并且只有在更新后的 snapshot 确认 `holding=true`、`control_mode=hold` 后才停止自有 daemon。外部 daemon 默认保持 hold 并继续运行，便于用户扶稳机械臂后再决定是否 stop。

如果未看到 confirmed hold，按告警处理，不要假设 follower 已保持。优先扶住机械臂、检查 daemon `last_error` 和日志，再决定是否急停或终止 daemon。

## 训练和推理

本工具只负责采集。采集产出的 LeRobot 数据集(在 `${PANTHERA_DATA_ROOT}/lerobot/` 下)
供主项目 `~/internship/baiqin/project-Embodied_AGV` 训练(SmolVLA/pi0.5 微调)与
安全 policy inference,完整说明见主项目 `docs/smolvla_training_inference.md`。

## 开发验证命令

运行采集链路测试：

```bash
python3 -m unittest \
  tests/test_panthera_lerobot_control.py \
  tests/test_panthera_lerobot_collect.py \
  tests/test_lerobot_panthera_plugins.py \
  tests/test_collect_scripts.py
```

语法检查：

```bash
python3 -m py_compile \
  panthera_lerobot_control/*.py \
  lerobot_robot_panthera_ht/*.py \
  lerobot_teleoperator_panthera_ht/*.py \
  tests/test_panthera_lerobot_control.py \
  tests/test_panthera_lerobot_collect.py \
  tests/test_lerobot_panthera_plugins.py \
  tests/test_collect_scripts.py
```

脚本检查：

```bash
bash -n arm/panthera/run_daemon.sh
```
