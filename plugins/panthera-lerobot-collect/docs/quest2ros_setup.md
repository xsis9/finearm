# Panthera-HT 机械臂 · LeRobot 数据采集手册(主从臂 / 手动示教 / Quest3S 遥操作)

> 目标:通过**主从臂(Leader-Follower)**、**手动示教(freedrive)**或 **Quest3S 右手柄遥操作**
> 驱动 Panthera-HT 机械臂,用本工具的采集链路直接产出 **LeRobot 格式数据集**(用于模仿学习训练)。
>
> 机器:AU(Ubuntu 22.04,conda base 自动激活)· ROS2 Humble · 工作区 `~/quest_ws` · 本工具 `plugins/panthera-lerobot-collect`
>
> 📌 本手册是独立采集工具 **panthera-lerobot-collect** 的操作文档
> (2026-11-10 从 project-Embodied_AGV 抽离,采集链路代码已全部包含在本目录,
> 训练/推理仍在主项目)。后续维护以本副本为准。

---

## 总览

### 三种采集方式对比

| | 方式一 · 主从臂(teleop) | 方式二 · 手动示教(freedrive) | 方式三 · Quest3S 遥操作 |
|---|---|---|---|
| 操作方式 | 手持主臂(Leader),从臂 1000Hz 跟随 | 直接用手拖臂 + 手捏夹爪 | 戴头显,右手柄遥控 |
| 终端数 | **1 个**(wrapper 自动拉起 daemon) | 2 个(daemon + collect) | 4 个(endpoint + daemon + bridge + collect) |
| 硬件依赖 | **Leader + Follower 双臂**都要上电连接 | 仅 follower 臂 | 仅 follower 臂 + Quest3S |
| 依赖软件 | 无(项目原生路径) | 无(不需要 Quest/ROS) | quest_ws + quest2ros App |
| 力反馈 | 从臂夹爪力矩反馈到主臂夹爪(捏到物体手上有感) | 全部直接手感 | 无 |
| 适用 | 官方推荐的正式批量采集 | 快速示教、验证任务设计 | 无主臂时的批量采集 |
| 实现状态 | ✅ 项目原生支持(2026-07-10 实机验收基线) | ✅ 单测+mock 过,实机验收待做 | ✅ 2026-10-23 实机验收通过 |

### 共用架构(三种方式在 daemon 汇合)

```
方式一:主臂(Leader)──1000Hz 关节复制──────────────────┐
                                                        │
方式二:人手 ──拖动──▶ follower 机械臂 ◀──零增益+重力/摩擦前馈─┤
                                                        │
方式三:Quest3S 右手柄 (quest2ros App)                   │
    │ TCP (Quest 主动连 192.168.110.14:10000)           │
    ▼                                                   │
ros_tcp_endpoint (本机, ~/quest_ws) ──▶ /q2r_right_hand_pose
    │                                                   │
    ▼                                                   │
quest_bridge.py(锚定式相对控制 + IK → 7 维关节目标)     │
    │ Unix socket IPC ── policy action ──▶ ┌────────────┴───────────────────┐
    │                                      │ panthera-control-daemon (1000Hz,│
    │                                      │ 唯一写硬件实体)                 │
    │                                      │ hold / teleop / policy /        │
    │                                      │ freedrive(2026-10-26 新增)      │
    │                                      │ 关节限幅 / 0.25rad/tick 跳变限制 │
    │                                      │ / 0.25s 新鲜度超时自动 HOLD      │
    │                                      └──────────┬─────────────────────┘
    │                                                 ▼
    └────────────── 同时 ─────────────▶ lerobot-record(相机 + 14 维 state + 7 维 action)
                                                      │
                                                      ▼
                                       LeRobot 数据集(原生格式,无需转换)
```

---

# 第一大部分 · 主从臂采集(Leader-Follower 遥操作)【项目原生路径】

> daemon 的 teleop 模式即为此设计:同时连接 Leader + Follower 双臂,1000Hz 把主臂关节
> 目标复制给从臂。这是项目官方推荐、验收最充分的采集入口
> (详见 `docs/lerobot_native_collection.md`,2026-07-10 实机验收基线)。

## 1.1 开始真机采集(1 个终端即可)

> **采集前第一步——固定相机参数**(三种采集方式通用,必须先跑):
>
> ```bash
> bash utils/lock_cameras.sh
> ```
>
> 恢复到参考数据集 bolt_test2 采集时刻的确切参数(wrist 手动曝光 36/gain125、
> front 全 auto)并逐项读回验证,保证采集/训练/推理相机外观一致。UVC 相机重插/
> 断电后参数复位,**每次开采集前都要跑**;`--live` 采集的相机预检会自动再调一遍
> (双保险)。该脚本位于仓库根 `utils/`,是采集与推理共用的唯一权威副本。

没有外部 daemon 时,采集 wrapper 会**自动拉起**自己的 teleop daemon(owned daemon),
所以一条命令即可开始:

```bash
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
./run_panthera_lerobot_collect.sh --live \
  --leader-config arm/panthera/robot_param/Leader.yaml \
  --follower-config arm/panthera/robot_param/Follower.yaml \
  --select-cameras \
  --repo-id local/bolt_test1 \
  --task "pick up the bolt and place it on the nut" \
  --episodes 1 --episode-seconds 60 --reset-seconds 60 \
  --fps 30 --hz 1000 \
  --sdk-root "${PANTHERA_SDK_ROOT}" \
  --teleop-gripper-squeeze-boost 1.0 \
  --camera-fourcc MJPG
  
```

> ⚠️ **本机必带 `--leader-config` / `--follower-config`**(指向 `arm/panthera/robot_param/`
> 的本地配置,serial_id 已按本机串口布局改对,见 1.4)。不带会报
> `Motor connection disconnected`——SDK 默认的 serial_id 1/2 只适配原验收工作站的口布局。
> ⚠️ **teleop 启动瞬间从臂会跳到主臂当前位形**——开录前把主臂摆到接近从臂的位姿
> (或手扶从臂),避免首帧大幅跳动。

流程:wrapper 检查到无外部 daemon → 自动 spawn `teleop` 模式 daemon(日志打印
`Starting Panthera daemon: ... --initial-control-mode teleop`)→ preflight(串口/相机/双臂
状态)→ **Preview 阶段人工确认现场安全** → 按 Enter 开录。

显式指定相机(替代 `--select-cameras`,固定工位推荐;`/dev/videoN` 会漂移,用 by-id 路径):

```bash
v4l2-ctl --list-devices     # 找到两个相机的 by-id 路径
./run_panthera_lerobot_collect.sh --live \
  --camera "front=/dev/v4l/by-id/usb-XXXX-video-index0" \
  --camera "wrist=/dev/v4l/by-id/usb-YYYY-video-index0" \
  --camera-fourcc MJPG \
  --camera-warmup-s 5 \
  ...其余参数同上
```

> ⚠️ **两个相机挂同一 USB 控制器时必须加 `--camera-fourcc MJPG`**(2026-10-27 实测):
> 未压缩 YUYV 640x480@30 两路 ≈36MB/s,超出 USB2 实际带宽,第二路相机
> `lerobot-record` 启动即 `read failed (status=False)` 崩掉(预检是逐个串行读帧,
> 所以预检能过、正式录制才炸)。MJPG 压缩后带宽降一个数量级,双路并发正常。
> ⚠️ **双相机并发初始化时首帧可能 >1s**(2026-10-27 实测 Brio 并发首帧 1.32s),
> LeRobot 默认 `warmup_s=1` 会报 `Timed out waiting for frame`。
> `--camera-warmup-s` 默认已放宽到 5s(只加长等首帧的超时上限,不影响正常启动速度),
> 一般无需显式传参。

**命令参数含义**(本表是三种方式共用的 collect 参数权威参考):

| 参数 | 含义 |
|---|---|
| `--live` | 连真实硬件(不加则 mock,不碰机械臂) |
| `--select-cameras` | 开关参数,**不带值**;preflight 弹 GUI 勾选相机(两个就勾两个)。显式指定用 `--camera NAME=SOURCE`(可重复传);查设备用 `v4l2-ctl --list-devices` |
| `--repo-id local/xxx` | 数据集名,决定保存目录 |
| `--task "..."` | 任务描述文本,写进数据集元数据(训练时的语言指令) |
| `--episodes N` | 采集多少集 |
| `--episode-seconds S` | 每集录制时长(秒),期间录制数据 |
| `--reset-seconds S` | **集间重置窗口,不录数据**——用于复位场景、摆臂回起点;`--episodes 1` 时可设 0 |
| `--fps 30` | 采样帧率(state/action 与视频对齐) |
| `--hz 1000` | daemon 控制环频率 |
| `--sdk-root` | SDK 路径(= `${PANTHERA_SDK_ROOT}`) |
| `--no-cameras` | 不录视频(快速验证链路用) |
| `--print-command` | 只打印将执行的 lerobot-record 命令不执行(与 `--select-cameras` 互斥) |
| `--resume --dataset-root DIR` | 续录已有数据集(见 1.5) |
| `--teleop-gripper-squeeze-boost X` | **夹持加力**(2026-10-26 新增):从臂夹爪被物体挡住、主臂继续捏时,自动加深虚拟目标增大夹持力;空夹不生效。0=关(默认),建议 0.5~1.0 起试 |
| `--teleop-gripper-squeeze-max X` | 加力封顶(默认 0.4,虚拟目标最多低于夹爪下限的量) |
| `--teleop-gripper-feedback-scale X` | 主臂夹爪力反馈强度(默认 0.5;调小主臂更轻、能捏得更深) |
| `--voice-prompts` | 保留 LeRobot 原生语音播报(默认关闭,改为终端蜂鸣:进 episode 两声、reset/停止一声) |

**数据保存位置**:`data/datasets/lerobot/local/leader_teach_pick_cube/<YYYYMMDD_HHMMSS_ffffff_p<PID>_c<计数>/`
(即 `${PANTHERA_DATA_ROOT}/lerobot/<namespace>/<name>/<时间戳目录>/`),结构:

```
├── meta/     # info.json、episodes、tasks 等元数据
├── data/     # parquet 分块(state 14 维 + action 7 维)
└── videos/   # 各相机文件夹的 mp4
```

**主从臂的数据语义**:`action` = daemon 限幅后的从臂关节目标(即主臂位姿),
`observation.state` = 从臂 14 维实时状态;前置/腕部相机随录制编码为视频。

**定标/记录起始位姿**(用主臂把从臂摆到目标初始位姿后执行;采集与部署必须用同一
起始位姿——模仿学习无探索能力,0 位姿开局=分布外。标准:臂在工作区后上方、夹爪张开、
front 相机里垫子/螺栓/螺母完整可见、腕部相机能看到工作区):

```bash
# 终端1 · daemon 以 teleop 模式运行(仓库根执行;用主臂带动从臂摆位姿,启动瞬间从臂跳到主臂位形,注意安全)
# (daemon 自动定位 SDK/venv,无需 source .env;确有覆盖项时可先 set -a; source plugins/panthera-lerobot-collect/.env; set +a)
bash arm/panthera/run_daemon.sh --live \
  --leader-config arm/panthera/robot_param/Leader.yaml \
  --follower-config arm/panthera/robot_param/Follower.yaml \
  --hz 1000 --initial-control-mode teleop
```

```bash
# 终端2 · 锁住当前位姿并读出关节向量(先切 hold 再读数,松开主臂不会漂)
cd plugins/panthera-lerobot-collect && set -a; source .env; set +a
"${PANTHERA_PYTHON}" -c "
from panthera_lerobot_control import ControlClient
c = ControlClient('/tmp/panthera_lerobot_control.sock')
print(c.set_mode('hold'))                      # 锁住当前位姿;切回 teleop 可继续用主臂带动
state = c.snapshot()['observation.state']
print('14 维 state:', [round(v, 4) for v in state])
print('初始位姿 6 关节:', [round(v, 4) for v in state[:6]])
print('夹爪:', round(state[6], 4))
"
```

> `observation.state` 前 7 维 = 6 关节 + 夹爪(与 action 语义一致);读 2~3 次确认数值
> 无漂移后把关节向量记档,部署初始化复用同一向量。

**本机定标结果(2026-10-27 实测,螺栓→螺母任务起始位姿)**:

| 项 | 值 |
|---|---|
| 初始位姿 6 关节 | `[0.0641, 0.781, 1.0801, -1.0882, 0.0314, 0.0113]` |
| 夹爪(张开) | `1.5576` |

**部署起始位姿(2026-10-27 依据批 1 实采数据统计,120+1 集)**:批 1 采集时起始位姿
未严格复位(与定标 HOME 偏差 mean j2/j3 约 0.1~0.3 rad),故**部署时应从训练起始
分布的中心启动**,而非定标 HOME:

| 项 | 值 |
|---|---|
| 部署起始位姿 6 关节(mean) | `[0.0503, 0.6404, 0.7899, -0.9262, -0.0057, 0.0301]` |
| 夹爪(张开) | `1.57` |
| 起始分布 std | 约 0.07~0.15 rad/关节 |

> ⚠️ 批 2(高质量)及后续批次采集时请用 1.2 节监控脚本严格复位(±0.05 rad 内),
> 减小起始分布方差;严格复位后部署起始位姿可回归定标 HOME。

## 1.2 操作节奏与停止

- 每集开始(终端蜂鸣两声)→ **手持主臂操作**,从臂实时跟随;主臂夹爪开合对应从臂夹爪,
  从臂夹爪的力矩会反馈到主臂夹爪(捏到物体手上有感觉)
- 每集结束(蜂鸣一声)→ reset 窗口(不录):复位场景、**双臂摆回起始位姿(1.1 节定标值)** → 下一集自动开始

**reset 复位校验**(可选但强烈推荐;另开一个终端挂着,teleop 下从臂跟随主臂,
把主臂带回 HOME、6 关节偏差全部进入 ±0.05 rad 时终端响铃并显示 HOME OK):

```bash
cd plugins/panthera-lerobot-collect && set -a; source .env; set +a
"${PANTHERA_PYTHON}" -c "
import time
from panthera_lerobot_control import ControlClient
HOME = [0.0641, 0.781, 1.0801, -1.0882, 0.0314, 0.0113]
TOL = 0.05
c = ControlClient('/tmp/panthera_lerobot_control.sock')
was_ok = False
while True:
    s = c.snapshot(max_age_s=0.5)['observation.state'][:6]
    err = [s[i] - HOME[i] for i in range(6)]
    ok = all(abs(e) < TOL for e in err)
    line = ' '.join(f'{e:+.2f}' for e in err) + ('   HOME OK' if ok else '          ')
    print('\r' + line, end='\a\n' if ok and not was_ok else '', flush=True)
    was_ok = ok
    time.sleep(0.2)
"
```
- **失败的 episode 按左方向键 re-record,不要用 Esc 丢弃**(partial episode 可能已被保存)
- 正式命令有意不用 `--auto-start`/`--no-status-ui`/`--no-release-prompt`,保留 Preview
  人工确认环节

停止:采集终端 Ctrl+C → wrapper 等待新 snapshot 确认 `holding=true` + `control_mode=hold`
→ owned daemon 提示"扶住机械臂"后按 Enter → daemon 退出时**先从臂缓动归位(~4s)再主臂缓动回 0 关节(~4s)**
(2026-10-27 新增 leader park:主臂原本退出即失去重力补偿直接摔落,现以软增益 kp=0.3/kd=0.03
+ 重力/摩擦前馈沿余弦缓动折叠到零位后再释放;`--no-leader-park` 可关,`--leader-park-move-s` 调时长)
→ 抱闸。

## 1.3 原理

- daemon(不带 `--no-leader`)同时连接 Leader + Follower 双臂串口,teleop 模式下
  1000Hz 循环:读双臂状态 → 主臂关节角 clamp 后作为从臂目标 → 下发从臂位置命令
- **主臂悬浮**:daemon 对主臂施加"重力+摩擦补偿"力矩(零位置增益),主臂轻若浮臂,
  长时间操作不累
- 从臂目标经过关节限幅 + 0.25rad/tick 跳变限制;控制环异常自动 fail-safe hold
- 快照 `action` 字段 = 限幅后的从臂目标 = 主臂位姿 → 录制即示教

## 1.4 硬件前提与安全

- **Leader + Follower 双臂都要上电、USB 连接**(控制板 USB 供电只能应答,
  **电机需要动力电源**——"板子在线+电机全断"就是没开总电/急停)。
  缺主臂会报 Leader 的 `Motor connection disconnected` + 999 限位告警
  (那种场景请用第二/三大部分的单臂方式,daemon 加 `--no-leader`)
- **主从角色由 serial_id 指定**(2026-10-26 实测破译):motor_param 里
  `serial_id: N` → 直接打开 `/dev/ttyACM(N-1)`,与运行时的 Port 列表顺序无关。
  本机布局:主臂板 = ttyACM7-13(CAN 口 ttyACM7 → **serial_id 8**),
  从臂板 = ttyACM0-6(CAN 口 ttyACM0 → **serial_id 1**)。
  SDK 自带配置写死 serial_id 1/2(2→ttyACM1 是从臂板的空接口,必然报 disconnected),
  只适配原验收工作站的口布局——**本机必须用本地配置**(放在 `arm/panthera/robot_param/`,
  由 Leader.yaml/Follower.yaml 指向 `motor_param/*_au.yaml`,URDF 用绝对路径指向 SDK;
  不改 SDK 仓库本体,重装 SDK 不受影响)
- **每次双臂上电先做 wiggle 确认角色**:轻抬悬浮无阻力的那只 = 主臂,另一只应实时跟随
- ttyACM 编号随插口/重启变化。两块板序列号相同(2024051701),`/dev/serial/by-id/`
  只能看到一套链接(冲突),用 `/dev/serial/by-path/` 区分;换口后需重验角色
- 串口权限:`ls -l /dev/serial/by-id/`;不可访问时 `sudo usermod -aG dialout $USER` 后重新登录
- 相机用 `/dev/v4l/by-id/` 路径;重插/重启后 `/dev/videoN` 会漂移,by-id 不变
- **软件 HOLD 和 Ctrl+C 不是硬件急停**。正式运行前必须明确可触达的物理断电方式,
  并清空机械臂和夹爪工作区
- 首次采集不要传 `--dataset-root`,wrapper 会创建唯一 timestamp root

## 1.5 续录已有数据集

只有明确的 `--resume --dataset-root` 才会追加(否则总是新建 fresh root):

```bash
export DATASET_ROOT=data/datasets/lerobot/local/leader_teach_pick_cube/<具体时间戳目录>

./run_panthera_lerobot_collect.sh --live \
  --camera "front=..." --camera "wrist=..." \
  --repo-id local/leader_teach_pick_cube \
  --task "pick up the cube with right arm" \
  --episodes 10 --episode-seconds 60 --reset-seconds 60 --fps 30 --hz 1000 \
  --sdk-root "${PANTHERA_SDK_ROOT}" \
  --resume --dataset-root "${DATASET_ROOT}"
```

resume 在机械臂启动前严格校验原数据集:`repo-id`、`task`、`fps`、feature 维度/类型必须
与原数据集完全一致,否则创建新数据集。

## 1.6 螺栓→螺母任务 · 采集流程 SOP(2026-10-27 定稿,用于 pi0.5 全量微调)

### 1.6.1 采集前固定项(全部批次通用,一次定死不再动)

| 项目 | 固定内容 |
|---|---|
| 相机 | front 机位定稿后**地上+桌面贴胶带标记**,整个数据集不再动;wrist 相机固定;线缆捆扎不会被碰动 |
| 命令 | 同一命令模板,只改 `--episodes`/`--resume`;必带 `--camera-fourcc MJPG` |
| task 文本 | `"pick up the bolt and place it on the nut"`,所有批次一致(=推理时语言指令) |
| fps/分辨率 | 30fps / 640×480,永不变 |
| 起始位姿 | 每集开始从臂、主臂都回同一初始位形(reset 窗口完成;定标值见 1.1 节"本机定标结果",reset 复位校验命令见 1.2 节) |
| 抓取策略 | 统一竖直下抓,全数据集一致 |

> 相机机位经验(2026-10-27 实测定稿):俯角 40–55° 斜上方、光轴对准垫子中心、
> 避开臂体方向(臂只占画面边缘 <10%)、垫子占画面 30–40%、背景带(顶部 15–20%)
> 略乱无妨但**不能有会动的东西**。Logitech 相机带重力传感器自动 180° 翻转,
> 俯角过大时会触发图像倒置——翻转安装(滚转 180°)可利用该特性,但**绝不能把机位
> 卡在翻转阈值附近**(图像时正时倒是数据灾难),定稿前连拍数帧确认方向稳定。

### 1.6.2 总体结构与批次计划(2026-10-27 定稿:基础量 + 高质量 + 多背景 + DAgger)

```
阶段A:批1(基础量 120 条,内嵌 3 种背景) + 批2(高质量 40 条)
   ↓ 合并训练 → pi0.5 v1 → 固定机位 rollout 评估
阶段B:批4(DAgger 纠错数据 30~50 条,必须有 v1 模型才能采)
   ↓ v1 + 批1 + 批2 + 批4 再训练 → v2 → 评估(不够再采一轮)
```

> **DAgger 必须放在最后**:它需要跑着训好的模型、人只在模型出错时接管纠错——没有
> v1 模型就无错可纠。这也是 1/2/4 分开建数据集的原因:批 4 是"模型分布之外"的数据,
> 混在基础数据里会稀释训练信号,且采集时机完全不同。背景变化(第 3 类)嵌在批 1 里。

**批 1 · 基础数据(repo-id: `local/bolt_p1_base`)→ 120 条,三个子批分时段采**

| 子批 | 条数 | 螺栓 | 螺母 | 背景 |
|---|---|---|---|---|
| 1a | 40 | 垫中央 3×3 随机,朝向 0/45/90/135° | 固定标记点 | **背景 A**(当前整洁状态) |
| 1b | 40 | 全域含边角,朝向加 180°/270° | 固定 | **背景 B**(垫外放纸盒+挪线缆) |
| 1c | 40 | 全域随机 | 固定点 ±3cm | **背景 C**(换深色桌布/挪杂物) |

- 三种背景差异要**肉眼明显**(物体位置/颜色/光照至少两处不同);1a/1b/1c 各占一个时段
  (如上午/下午/晚上),光照增广顺手就有
- 操作正常速度即可,失败 re-record

**批 2 · 高质量数据(repo-id: `local/bolt_p2_quality`)→ 40 条,批 1 采完人手热了再采**

- 背景 A、螺母 ±5cm 随机、螺栓全域随机
- **标准最严**:动作放慢 30%,对准段再慢;每集必须一次完美完成,任何犹豫/微调/回退就重录
- 目的:给模型看"天花板",强化最终放置精度段的分布

**批 4 · DAgger 纠错数据(repo-id: `local/bolt_p4_dagger`)→ 30~50 条,v1 训好之后采**

1. 以 policy 模式跑 v1 模型驱动从臂 + 正常录采集
2. 人持主臂在旁盯着:**模型一旦走错(伸错方向/对不准/要碰倒螺母)立即接管纠正**,
   纠回正轨后放开让模型继续,直到任务完成
3. 纠错段的 action 来自主臂 = 高价值监督信号,专补模型"会犯错的区域"(DAgger 核心收益)
4. 一集纠错 1~3 次都正常,照样保留;模型一路做对的也保留(说明该区域已掌握)
- ⚠️ 前置技术项:需打通 daemon 的"policy ↔ teleop 秒级切换"(ControlClient.set_mode
  已支持,录制的是 daemon 限幅后 action,切换不影响录制);批 1/2 训练跑通后接此工具

**通用约束**
- 另预留 3–5 个螺栓位置(所有批次**故意不采**)作部署评测用,检验真泛化
- 背景扰动约束:干扰物只放操作垫以外;每条 episode 内背景纹丝不动(变动只在集间 reset 做);
  **不放与目标相似的东西**(小金属圆柱、黄铜色零件、螺母状物体)
- 每个 repo-id 首次新建,续录加 `--resume --dataset-root`;训练时多数据集合并喂入

### 1.6.3 每条 episode 的操作节奏

1. **reset 窗口(60s)做三件事**:摆螺栓到本轮随机位置、双臂回起始位形、确认垫上无杂物
2. 蜂鸣两声开录 → **主臂缓慢、匀速、一次到位**:接近→减速→抓→提起→平移→对准→下放→松爪
3. 放置判定:螺栓确实落在螺母上/立住才算成功
4. 任何失败(没抓住、中途掉、放偏、碰倒螺母)→ **左方向键 re-record,不要将就录完**
5. 蜂鸣一声 → reset 窗口重复第 1 步

### 1.6.4 操作注意事项(直接影响数据质量)

**动作质量**
- 慢而稳,最后 5cm 对准段放慢一倍;全程连续,不犹豫、不中途撤回重试——错误恢复动作会被模型学走
- 夹爪加力已开(squeeze-boost 1.0),主臂有反馈即可,不必死捏

**一致性**
- 操作者站位固定(相机背后,不入镜);避不开就每次站同一位置
- `--episode-seconds 30` 够用,动作节奏固定使每集时长自然一致
- 螺栓只放预定义区域内,不越界到垫外

**链路安全**
- teleop 启动瞬间从臂跳到主臂位形——开录前主臂先摆到接近从臂的姿态
- 采集中别碰相机/USB 线;中途断了按"常见问题速查"排障;续录用 `--resume --dataset-root`,参数必须与原数据集完全一致
- 停止流程:采集终端 Ctrl+C → 确认 `holding=true` → 托臂 → Enter

### 1.6.5 每批采完的验证(5 分钟)

```bash
# 目录下确认 episode 数、两个相机的视频、parquet
ls data/datasets/lerobot/local/<repo-id>/<时间戳>/videos/*/ | head
# 抽看 2-3 集视频:方向正、关键瞬间无遮挡、画面稳定
```

判定标准:批 1+批 2 合并微调出 v1 后,在固定机位 rollout 20 次,**成功率 ≥60%** 进入
DAgger 阶段;**<30%** 先回来排查数据/视角/训练配置问题,不要盲目堆量或急着采批 4。

---

# 第二大部分 · 手动示教采集(freedrive 模式)【✅ 2026-10-26 实现,单测+mock 通过】

## 2.1 开始真机采集(2 个终端)

**终端1 —— daemon 以 freedrive 模式启动**(⚠️ 启动瞬间臂变悬浮,先托住臂!):

```bash
# 仓库根执行(daemon 自动定位 SDK/venv,无需 source .env)
bash arm/panthera/run_daemon.sh --live --no-leader --hz 1000 --initial-control-mode freedrive
```

**终端2 —— 采集**(preflight 会校验 daemon 状态,freedrive 已放行):

```bash
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
./run_panthera_lerobot_collect.sh --live \
  --select-cameras \
  --repo-id local/hand_teach_demo \
  --task "pick up the cube" \
  --episodes 10 --episode-seconds 30 --reset-seconds 30 \
  --fps 30 --hz 1000
```

**命令参数含义**:collect 侧参数与第一大部分 1.1 的表格完全相同,不重复;
daemon 侧差异见下表(此方式必须先手动起 daemon,所以是 2 个终端):

| daemon 参数 | 含义 |
|---|---|
| `--no-leader` | 单臂运行(虚拟 Leader 占位)。不加会报 Leader 的 `Motor connection disconnected` + 999 限位告警 |
| `--initial-control-mode freedrive` | daemon 从第一个 tick 起进入重力补偿悬浮模式 |
| `--no-park` / `--park-joints "j1,...,j6"` | 退出归位控制(默认 Ctrl+C 时从示教位姿平滑归位 ~4s 再抱闸) |

**数据保存位置**:`data/datasets/lerobot/local/hand_teach_demo/<时间戳目录>/`,
目录结构同 1.1。**freedrive 的数据语义**:`action` 恒等于示教位姿(daemon 每帧把
target 设为 follower 实时位姿),`observation.state` = follower 14 维实时状态。

## 2.2 操作节奏与停止顺序

1. 托住臂 → 终端1 启动 daemon(臂悬浮)→ 终端2 启动 collect → preflight 检查 → Preview 确认相机画面 → 开录
2. 每集录制窗口内:**用手拖动臂 + 手捏夹爪**完成任务(动作从容一点,数据质量更好)
3. 每集结束 → reset 窗口(不录):复位场景 + 用手把臂拖回起始位姿 → 下一集自动开始
4. 采集中途 daemon 一直保持 freedrive,拖臂随时可行;需要臂锁住不动时切 hold(见 2.6)

停止顺序:采集终端 Ctrl+C(自动 hold,臂变硬锁住)→ 确认 `holding=true` → 停 daemon
(默认自动归位 park ~4s,期间忽略 Ctrl+C)。

## 2.3 原理

**为什么通电后直接掰不动**:机械臂通电使能后,关节处于闭环位置控制(KP/KD 增益锁位),
手掰 = 和 1000Hz 位置控制器角力,又卡又慢;断电则抱闸锁死。
手拖示教需要**重力补偿(float)模式**:kp=kd=0(关闭位置环)+ 前馈"重力+摩擦"补偿力矩,
手臂悬浮可随手移动,夹爪同样零增益可手捏。

**为什么不能直接跑 SDK 的重力补偿脚本**
(`${PANTHERA_SDK_ROOT}/panthera_python/scripts/2_gravity_friction_compensation_control.py`,
SDK 位于 `arm/panthera/sdk/Panthera-HT_SDK`):
它会绕过 daemon 独占串口(冲突),且采集链路读不到状态。正确做法是 freedrive 作为
daemon 的第四种模式——daemon 的 `_step` 本来每 tick 就在算 `follower_torque = 重力+摩擦补偿`
(原用于位置环前馈),freedrive 只是把同一份力矩以零增益下发,与 SDK 示例的
`pos_vel_tqe_kp_kd(0,0,G,0,0)` 完全等价。

**录制如何工作**:freedrive 模式下 daemon 每帧把 `target = follower 实时位姿`,而快照的
`action` 字段 = target(7 维)→ **录下来的 action 恒等于示教位姿**,采集链路
(robot 插件、collect_cli)零改动直接复用。

## 2.4 实现改动清单(2026-10-26)

| 文件 | 改动 |
|---|---|
| `panthera_lerobot_control/controller.py` | ① `CONTROL_MODES` 加入 `"freedrive"`;② `_step` 新增 freedrive 分支:target 跟随 follower 实时位姿、holding=False;③ 下发命令处新增分支:臂 `command_arm(target, 0, 重力+摩擦力矩, kp=0, kd=0)`、夹爪 `command_gripper(target, 0, 0, 0, 0)`(手捏可动) |
| `panthera_lerobot_control/ipc.py` | `ControlClient.set_mode` 白名单加入 `"freedrive"` |
| `panthera_lerobot_control/cli.py` | `--initial-control-mode` 选项加入 `freedrive`(daemon 可直接以 freedrive 启动) |
| `panthera_lerobot_control/collect_cli.py` | `_validate_daemon_status` 的 control_mode 白名单从 `{teleop, policy}` 放宽为 `{teleop, policy, freedrive}` |
| `run_panthera_lerobot_daemon.sh` | help 文本补 `--initial-control-mode` 透传说明(无逻辑改动) |
| `tests/test_panthera_lerobot_control.py` | 新增 3 测试:freedrive 零增益+重力前馈+action==示教位姿;freedrive→hold 恢复 KP/KD;IPC set_mode("freedrive") 验收 |
| `tests/test_panthera_lerobot_collect.py` | 新增 1 测试:采集接受 freedrive 模式的外部 daemon |

验证:全量单测 332 个(新增 4 个),失败数与改动前基线完全一致(仅 2 个存量
环境问题:whiptail 加载错误、smolvla 测试硬编码 `/home/jiaming` 路径)。
mock 冒烟:daemon `--initial-control-mode freedrive` 启动 → status/snapshot 为
freedrive、holding=False、`action == observation.state[:7]` → `set_mode("hold")` 正常切回。

## 2.5 安全机制(全部复用现有保护)

- **异常 fail-safe**:控制环任何异常 → 自动 hold(KP/KD 恢复)锁在当前位姿,臂不会失重坠落
- **采集结束**:collect 正常收尾自动 hold 并提示托臂后再停 daemon
- **daemon 退出**:Ctrl+C 默认从示教位姿平滑归位(park)再抱闸,不想归位用 `--no-park`
- **模式随时可切**:`set_mode("hold"/"policy"/"freedrive")` 即时生效
- freedrive 不需要 leader,`--no-leader` 照常使用;力矩有 TAU_LIMIT 限幅

## 2.6 模式切换速查

```bash
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
# 切 freedrive(悬浮,可手拖) / 切 hold(锁位):
"${PANTHERA_PYTHON}" -c "from panthera_lerobot_control import ControlClient; \
print(ControlClient('/tmp/panthera_lerobot_control.sock').set_mode('freedrive'))"
```

## 2.7 注意事项 + 实机验收清单

- **切进 freedrive 的瞬间臂即失重**:务必先托住,尤其末端伸展较远时
- 重力补偿不完美(夹爪自重、模型误差),松手后臂可能极缓慢漂移,属正常;停留示教时建议切 `hold` 锁住
- 手拖速度别太猛(力矩限幅 TAU_LIMIT,猛烈甩动可能触发异常 fail-safe hold)
- ⬜ **实机验收待做**:托臂启动 → 缓慢全行程拖动 → 手捏夹爪 → 切 hold/切回 freedrive →
  跑 1 个短 episode(`--episodes 1 --episode-seconds 20 --reset-seconds 0`)验证数据落盘

---

# 第三大部分 · Quest3S 遥操作采集【✅ 2026-10-23 实机验收通过】

## 3.1 开始真机采集(4 个终端)

**终端1 —— endpoint**(保持运行;一次只能跑一个,重复启动会端口冲突/数据错乱):

```bash
source ~/quest_ws/install/setup.bash
ros2 launch ros_tcp_endpoint endpoint.py
```

Quest3S:戴上头显 → 打开 quest2ros App → IP `192.168.110.14`,端口 `10000` → **Apply**。
看到 `RegisterPublisher(q2r_right_hand_pose ...) OK` 即连接成功。

**终端2 —— daemon**:

```bash
bash arm/panthera/run_daemon.sh --live --no-leader --hz 1000
# Ctrl+C 停止时自动平滑归位到启动位形(~4s)再抱闸;--no-park 关闭 / --park-joints 指定目标
```

**终端3 —— quest bridge**:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/quest_ws/install/setup.bash
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
"${PANTHERA_PYTHON}" quest_bridge.py --side right \
    --socket /tmp/panthera_lerobot_control.sock \
    --gripper-mode trigger
# 启动时询问起始位姿:[1] 归位到俯视桌面初始位姿 [2] 保持当前位姿
```

**终端4 —— 采集**(preflight:串口/相机/daemon 状态检查,Preview 确认后开录):

```bash
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
./run_panthera_lerobot_collect.sh --live \
  --select-cameras \
  --repo-id local/quest_pick_cube \
  --task "pick up the cube with right arm" \
  --episodes 50 --episode-seconds 60 --reset-seconds 60 \
  --fps 30 --hz 1000 \
  --sdk-root "${PANTHERA_SDK_ROOT}"
```

**命令参数含义**:collect 侧参数与第一大部分 1.1 的表格完全相同,此处不重复;
daemon 侧除**不加** `--initial-control-mode freedrive` 外与第二大部分 2.1 相同
(bridge 会在按 A 时自行 `set_mode("policy")`)。

**数据保存位置**:`data/datasets/lerobot/local/quest_pick_cube/<时间戳目录>/`
(结构同 1.1)。**Quest 方式的数据语义**:`action` = bridge IK 算出、daemon 限幅后的
从臂关节目标,`observation.state` = follower 14 维实时状态。

**操作节奏**:每个 episode 内戴头显按住 A 键锚定 → 手柄操作 → 完成任务;摘头显即可
(追踪暂停 → 数据断流 0.5s → bridge 视为松键 → daemon 自动 HOLD)。

**停止流程(顺序不可乱)**:采集终端 Ctrl+C → 等待 `holding=true` 确认 → 停 bridge →
最后停 daemon(归位期间约 4s 忽略 Ctrl+C)→ 停 endpoint。

## 3.2 采集前自检:Quest 世界系(30 秒,必做)

Quest 的世界坐标系决定了"手柄移动"如何映射到"机械臂移动",**采集前必须自检**:

```bash
# 任意终端(不用开 daemon/bridge)
source ~/quest_ws/install/setup.bash
ros2 topic echo /q2r_right_hand_pose
```

戴上头显,站到标定时的朝向(面朝机械臂"前方",基座 X 指向的一侧),手柄举胸前,依次:

- 前推 20cm → `position.x` 应**变大**
- 左移 20cm → `position.y` 应**变大**
- 上移 20cm → `position.z` 应**变大**

三者都对 → 直接开采集。判定口诀:**z 永远不会错**(重力锁定);只有 x/y 会乱。

**x/y 错乱时两种修法:**

- **方案 A · 头显重新居中**:站到想要的面朝方向,按住右手柄 Meta 键约 2 秒 →"重置视图"→
  重新自检(有的 App 用 Guardian 空间,重置视角可能无效,不对就用方案 B)。
  **采集过程中千万不要居中**,会当场转掉坐标系
- **方案 B · 软件重标定(最可靠)**:实测世界系朝向后改 bridge 参数——
  `--axis-map` 第 i 位填"推基座第 i 轴正方向时哪个世界分量在变",
  `--axis-signs` 第 i 位填该分量的变化方向(增=+1,减=-1)。
  例:面朝机械臂前方推手柄 y 在减、左移 x 在增 → `--axis-map "y,x,z" --axis-signs "-1,-1,1"`

**防复发**:地上贴胶带标记站位,每次站同一位置同一朝向。

## 3.3 底层原理:头显与手柄的分工

**追踪层**:头显用摄像头+IMU 做 inside-out SLAM 定位自己,建立世界系(原点/偏航由
初始化/重新居中决定,z 由重力对齐永远向上,**固定在房间里、不跟身体转**)。
手柄自己只有 IMU,**姿态永远有效,但位置必须被头显摄像头看到**——手柄出视野
(举太低/放身后)→ 位置冻结漂移、姿态正常 → 表现为"平移不动但拧手腕好使"。

**数据层**:quest2ros App 周期读右手柄世界系位姿(7 个数)发到 endpoint,
发布 `/q2r_right_hand_pose`(~36Hz)。

**控制层**:quest_bridge.py(按住 A 锚定时):

```
Δ = 手柄当前位置 − 锚定时手柄位置            ← 世界系位移增量
Δ_基座 = M @ Δ                                ← M = 固定的 axis-map/signs 矩阵
目标末端 = 锚定时末端FK + Δ_基座 × scale      ← clip 到 max-reach
手腕姿态 = 锚定时末端姿态 × 手柄旋转增量       ← 手柄姿态系,不受站位影响
IK(目标) → 7 维关节目标 → daemon policy 模式 → 电机
```

要点:用**位移增量**而非绝对位置(人站哪、手举多高无所谓);矩阵 M 按标定时的
"你的朝向 vs 机械臂朝向"写死,换站位/重新居中后需按 3.2 重新自检。

## 3.4 quest_bridge.py 参数速查

操作速查:**按住A=锚定跟随;平移手柄=末端平移;拧手腕=j6夹爪角度;腕上下弯=前后倾;
腕左右弯=j5左右摆;扳机=夹爪开合;松开A=臂停住保持**。

| 参数 | 说明 |
|---|---|
| `--side right` | 右手柄 |
| `--socket` | daemon IPC socket 路径 |
| `--gripper-mode trigger` | 扳机模拟量(不扣=开、扣紧=合、中间可停;提着物体别松)。`toggle` 备用未实测 |
| `--scale` | 手柄→臂平移缩放(默认 1.0,精细操作可 0.5) |
| `--max-reach` | 位移限幅(默认 0.3m) |
| `--dry-run` | 只算 IK + 打日志,不发 action(安全测试第 2 级) |
| `--sim-input` | 合成输入,无需 Quest/ROS(mock 测试) |
| `--verbose` | 打印 action 与 1Hz status 行 |
| `--no-rotation` | 锁定末端姿态,只平移+扳机(精细抓取更稳) |
| `--start-pose ask\|home\|current` | 起始位姿选择(默认 ask 交互询问) |
| `--home-joints j1,...,j6` | 自定义初始位姿(默认 `[0, 0.90, 0.90, -0.95, 0, 0]`,实机手调定稿) |
| `--axis-map x,y,z` / `--axis-signs 1,1,1` | 平移映射(默认已实机标定,勿改;错乱时按 3.2 方案 B 重标) |
| `--wrist-map x,y,z` | 腕部姿态映射(默认已实机标定:拧=j6、左右弯=j5) |

**单臂说明**:本方案只用 Follower 一只臂——Quest 手柄取代了主从示教里 Leader 臂的角色。
daemon 原设计必须同时连 Leader+Follower 双臂,已加 `--no-leader` 参数支持单臂运行
(2026-10-23,虚拟 Leader 占位,policy/hold/freedrive 模式不受影响,teleop 模式被拒绝)。

## 3.5 一次性环境搭建(装好就不用再动)

### 3.5.1 quest_ws 工作区【✅ 2026-10-22 验收通过】

已装好并实测通过:位姿(位置+四元数)、扳机模拟量(0~1)、A/B 键、握键、摇杆,~36Hz。

每次使用的启动/验证命令见 3.1 终端1;辅助验证:

```bash
source ~/quest_ws/install/setup.bash
ros2 run q2r2_bringup CheckTCPconnection     # 每秒打印 pose/inputs/twist
ros2 topic echo /q2r_right_hand_inputs       # 扳机/按键
ros2 topic hz /q2r_right_hand_pose           # 频率(~36Hz)
```

关键使用守则(踩坑记录):

1. **一段时间不戴头显连接会断**——Quest 接近传感器检测到离头会暂停追踪,App 只发全零。
2. 遇到 Python 报错先 `conda deactivate` 再跑(anaconda 干扰 ROS2)。
3. 编译(colcon build)必须用系统 Python,命令:
   ```bash
   env -i HOME="$HOME" USER="$USER" TERM=xterm \
     PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
     bash -c 'source /opt/ros/humble/setup.bash && source ~/quest_ws/install/setup.bash \
       && cd ~/quest_ws && colcon build'
   ```
4. 本机 Wi-Fi IP(当前 `192.168.110.14`)是 DHCP 动态的,变了要同步改:
   `~/quest_ws/src/ros_tcp_communication/launch/endpoint.py` 的 `ROS_IP` + Quest App 里的 IP。
   查当前 IP:`ip -4 addr show wlp4s0`
5. 若 App 连上即断、无限重连 → 已修复(syscommand `\0` 问题);若数据乱码 → 已修复(CDR 解析)。
   原始字节日志(排障用):`~/.ros/log/<最新时间戳>/launch.log`

已做的代码修复(都在 `~/quest_ws/src/ros_tcp_communication/`,已编译生效):

| 文件 | 修复 |
|---|---|
| `ros_tcp_endpoint/server.py` | syscommand JSON 末尾 `\0` 终止符导致断连 → `rstrip("\x00")` |
| `ros_tcp_endpoint/ros_msg_converter.py` | 重写为标准 ROS2 CDR 解析(pose: 头4B+时间戳8B+字符串,7 double 按 8 对齐;inputs: 2 bool + 4 float32,`press_index`@16,`press_middle`@20) |
| `ros_tcp_endpoint/client.py` | DEBUG print 加 `flush=True` |

### 3.5.2 项目环境(.env + SDK + LeRobot venv)【✅ 2026-10-23 验收通过】

```bash
cd plugins/panthera-lerobot-collect
cp .env.example .env          # 可选:默认零配置即可跑(SDK/数据/venv 均自动定位)
set -a; source .env; set +a

python3.12 --version          # 没有则: sudo apt install -y python3.12 python3.12-venv
./scripts/init_workspace.sh --install    # 自动拉 SDK + 建 Python 3.12 venv + 装 LeRobot 0.5.1
./scripts/init_workspace.sh --check      # 完成后校验
```

> SDK 已改从 GitHub 官方源拉取(`https://github.com/HighTorque-Robotics/Panthera-HT_SDK.git`),
> 锁定 revision `7f704d9`,无需 Gitea 公钥。

无硬件 smoke test(不动机械臂;三种方式通用):

```bash
# 终端1:mock daemon(仓库根执行)
bash arm/panthera/run_daemon.sh --mock --hz 1000
# 终端2:检查将要执行的采集命令(插件目录执行)
cd plugins/panthera-lerobot-collect
./run_panthera_lerobot_collect.sh --mock --no-cameras \
  --repo-id local/panthera_quest_smoke --task "mock smoke test" \
  --episodes 1 --episode-seconds 2 --reset-seconds 0 --print-command
```

> 实测补充:去掉 `--print-command` 才真正录制,数据在 `data/datasets/lerobot/local/panthera_quest_smoke`。
> 若报 `Daemon control_mode is 'hold'`,是上次残留 daemon 占了 socket:
> `pkill -f panthera_lerobot_control.cli && rm -f /tmp/panthera_lerobot_control.sock` 后重跑。

## 3.6 安全测试阶梯(Quest 链路逐级验证)【✅ 2026-10-23 三级全部通过】

```bash
# ========== 第1级:mock 全链路【✅ 已通过】==========
# 终端1:mock daemon(不动硬件;仓库根执行)
bash arm/panthera/run_daemon.sh --mock --hz 1000

# 终端2:bridge --sim-input(合成手柄输入,无需 Quest/ROS endpoint)
conda deactivate
cd plugins/panthera-lerobot-collect
set -a; source .env; set +a
"${PANTHERA_PYTHON}" quest_bridge.py --mock --sim-input --verbose
# 验证点:看到 "anchored"、status 行里 last_action 在变、
#         Ctrl+C 后 "requested daemon hold"、daemon 进入 holding=true

# ========== 第2级:实机 daemon + 桥只算日志不写(验证 IK 数值合理)【✅ 已通过】==========
# 终端1:endpoint(见 3.1)+ Quest 戴头显连接
# 终端2:实机 daemon(仓库根执行:bash arm/panthera/run_daemon.sh --live --no-leader --hz 1000)
# 终端3:bridge --dry-run(读真 follower 关节、算 IK、打日志,不发 action)
"${PANTHERA_PYTHON}" quest_bridge.py --side right --dry-run --verbose
# 操作:戴头显按住 A → 小幅移动手柄 → 观察 action 数值连续、无频繁 IK failed
# 验证点:pose_rx/inputs_rx 持续增长(~36Hz);last_action 与手柄移动方向一致

# ========== 第3级:实机低速真控(手扶机械臂、工作区清空、急停可达)【✅ 已通过】==========
# 同第2级,但去掉 --dry-run,首测加 --scale 0.5 --max-reach 0.15:
"${PANTHERA_PYTHON}" quest_bridge.py --side right --scale 0.5 --max-reach 0.15
# 操作:按 A 锚定 → 小幅移动手柄 → 观察跟随 → 扣/松扳机看夹爪 → 松开 A → 确认 HOLD
# 任何时候 Ctrl+C 退出 bridge,daemon 都会自动 hold
```

## 3.7 开发记录(备查)

### quest_bridge.py 功能注记【✅ 2026-10-23 实机验收通过】

> 位置:项目顶层 `plugins/panthera-lerobot-collect/quest_bridge.py`
> (没有放进 ros2_ws —— 项目规定 ROS 栈禁止直写 daemon)

- 订阅 `/q2r_right_hand_pose` 和 `/q2r_right_hand_inputs`:**实现方式**为不经 rclpy
  (rclpy 只有 Humble/Python3.10,项目 venv 是 3.12),而是起 `ros2 topic echo` 子进程
  (系统 Python 跑)+ 增量 YAML 解析,bridge 本体留在 3.12
- 锚定式相对控制:按住 A 键(`button_lower`)锚定;目标 = FK锚点 + 平移增量 + 旋转增量;
  松开 A 停发 → daemon 0.25s 超时自动 HOLD
- 方向映射(2026-10-23 实机标定):手柄系 x=前后、y=左右、z=上下;基座系 X=前后、
  Y=左右、Z=上下 → 恒等映射已设默认。**教训:手柄/基座系都需实测标定,别靠猜**
- 腕部姿态映射(实机标定):旋转增量在手自身坐标系,映射到夹爪自身坐标系施加(恒等已设默认)
- IK/FK:**不实例化 `Panthera()`**(其构造器会开串口,与 daemon 冲突),直接用 pinocchio
  加载 URDF 纯计算;DLS + 步长限幅 + 关节限位 clamp + 多起点重试,以 follower 当前关节为种子
- IK 无解:保留上一帧 + 限频告警(daemon 侧超时/限幅双保险)
- 优雅退出:Ctrl+C/SIGTERM → 主动下发 hold;Quest 数据断流 0.5s 视为松键
- 已验证:IK 往返 3/3 收敛(perr<3mm)、小位移 50/50 求解 0.2ms/次;
  mock 全链路 110 个 policy action 被 daemon 接收
- 实机验证全部通过:平移六方向、腕部三轴、夹爪扳机、初始位姿归位、daemon 退出归位+抱闸

### collect_cli 放宽 policy 模式【✅ 2026-10-23】

`_validate_daemon_status` 的 control_mode 校验从只接受 `teleop` 放宽为
`{"teleop", "policy"}`(2026-10-26 又加入 `freedrive`,见 2.4),否则 Quest 桥以
policy 模式驱动时采集会直接报错。同步更新了单测。

### 遗留观察项

SDK 偶发 `IO Exception (9): Bad file descriptor ... Aborted (core dumped)`
(C++ 串口库崩溃,Python 层无法捕获)。若发生在退出时且 `motor brake` 已打印,臂已安全,
仅 socket 残留(重启自动清理);若运行中频发,查 USB 线/供电。

---

## 常见问题速查

| 现象 | 处置 |
|---|---|
| **手柄平移不动,但拧手腕正常** | **手柄出了头显摄像头视野**(位置追踪失效、IMU 姿态仍有效)。手柄举到胸前、保持视野内,看 bridge status 的 `hand(xyz)` 是否恢复变化 |
| **手柄上下反向** | 先怀疑手柄出视野在漂移(z 轴由重力锁定**永不反**),别急着改 `--axis-signs` |
| 手柄前后/左右方向乱 | Quest 世界系偏航变了(重新居中/换站位),按 3.2 方案 A/B 处理 |
| 采集听不到蜂鸣 | 蜂鸣=终端响铃字符 `\a`,需终端开启"响铃":GNOME Terminal 首选项勾选;仍无声试 `gsettings set org.gnome.desktop.sound theme-name 'freedesktop'`;或改用 `--voice-prompts` |
| App 连接后数据全零 | 戴上头显!离头暂停追踪 |
| 手柄按键/位姿全零但 pose_rx 在涨 | 头显没戴头上(接近传感器暂停追踪,App 发全零消息) |
| App 连上即断循环重连 | 确认跑的是修复后的 endpoint(3.5.1);查终端1报错 |
| `ros2: command not found` | `source ~/quest_ws/install/setup.bash` |
| Python 相关报错 | `conda deactivate` 后重跑 |
| 端口 10000 被占 | `pgrep -af default_server_endpoin` 查残留,杀掉后重启 |
| IP 变了 | 改 endpoint.py 的 ROS_IP + App 里重填,重编译不必要(launch 参数) |
| 串口不可访问 | `ls -l /dev/serial/by-id/`;`sudo usermod -aG dialout $USER` 后重新登录 |
| daemon 频繁 HOLD | 看 bridge 的 IK 无解告警 / 手柄数据是否断流(超时 0.25s 即 HOLD,属正常保护) |
| 某方向手柄↔臂反向 | `--axis-signs` 翻对应位;若方向串位(左右↔前后)改 `--axis-map`(默认已实机标定,勿改) |
| 手腕动作对应错关节 | `--wrist-map` 调顺序(默认已实机标定:拧=j6、左右弯=j5) |
| 通电后手掰不动臂 | 正常(闭环位置控制锁位)。手拖要用 freedrive 模式(第二大部分) |
| 999 限位告警刷屏 | 电机状态读不到:机械臂没上电/急停(999=无数据哨兵值);或单臂采集没加 `--no-leader` |
| 双臂报 `Motor connection disconnected`(7 个全断) | ① 臂没上动力电(控制板 USB 电只够应答);② serial_id 指错口——本机必须带 `--leader-config/--follower-config`(见 1.4);定位用 lsof 看 daemon 实际打开的口 |
| daemon 杀不死/重启报串口错误或段错误 | C 库对普通信号响应慢,残留进程占串口:`pkill -9 -f panthera_lerobot_control.cli` 后重跑 |
| 主从角色反了(抬从臂想当主臂) | 角色由 serial_id 决定不由外观决定(见 1.4);对调两根 USB 线或改本地 motor_param 的 serial_id |
| `IO Exception (9): Bad file descriptor` 后 core dumped | SDK C++ 串口库崩溃。退出时且已打印 motor brake → 无害;运行中频发 → 查 USB/供电 |
| 相机 `read failed (status=False)` / `read thread is not running`,且单测该相机正常 | USB 带宽不足(多相机同控制器、未压缩 YUYV)。加 `--camera-fourcc MJPG`;验证:`ffmpeg` 双路 YUYV 并发报 `No space left on device` 即坐实。或把相机分插不同 USB 控制器 |
| 相机 `Timed out waiting for frame ... after 1000 ms` | 双相机并发初始化首帧 >1s(Brio 实测 1.32s),超过 LeRobot 默认 warmup 1s。`--camera-warmup-s` 默认 5s 已放宽,旧命令需补该参数或重跑 |
| `Segmentation fault`(daemon 启动时) | 机械臂没上电,SDK 空串口列表崩溃;上电后重试 |

---

## 当前进度清单

- [x] ROS2 Humble 安装
- [x] quest_ws 工作区(quest2ros / q2r2_bringup / ros_tcp_endpoint)
- [x] endpoint 三处修复 + Quest3S 实机验收(位姿/扳机/按键)
- [x] 项目 .env + init_workspace.sh(SDK 已改从 GitHub 官方源拉取,2026-10-23)
- [x] mock smoke test(lerobot-record 60 样本落盘,2026-10-23)
- [x] quest_bridge.py 编写 + mock 全链路验证(2026-10-23)
- [x] collect_cli.py 放宽 policy 模式 + 单测全过(2026-10-23,存量 2 处环境问题与本次无关)
- [x] daemon 单臂支持 `--no-leader`(2026-10-23,虚拟 Leader 占位,teleop 被拒)
- [x] daemon 退出自动归位 park-on-exit(2026-10-23,实机验证)
- [x] 安全测试阶梯三级全部通过(2026-10-23 实机验收)
- [x] 方向/腕部映射实机标定(axis-map/wrist-map 恒等,已设默认)
- [x] 初始位姿实机手调定稿 + 启动时交互选择归位(2026-10-23)
- [x] 夹爪模式定型 trigger(2026-10-23 实机验证)
- [x] daemon freedrive 手动示教模式 + 采集放行(2026-10-26,单测 332 过、mock 冒烟过,实机验收待做,见第二大部分)
- [ ] 主从臂(Leader-Follower)在本机实机采集验证(第一大部分;项目原生路径,2026-07-10 已在工作站验收过基线)
- [ ] freedrive 实机验收(第二大部分 2.7 清单)
- [ ] Quest 正式采集 + 数据集验证(第三大部分 3.1)
- [ ] 全量单测 + git 提交同步公司远端

## 下一步操作(按序执行)

1. **主从臂实机采集验证**(第一大部分 1.1,双臂上电后跑 1 个短 episode 验证链路与数据落盘)
2. **freedrive 实机验收**(按第二大部分 2.7 清单,托臂启动→拖动→捏夹爪→跑 1 个短 episode)
3. **正式采集**:主从臂用第一大部分 1.1(1 个终端);手动示教用第二大部分 2.1(2 个终端);
   Quest 遥操作用第三大部分 3.1(4 个终端:endpoint → daemon → bridge → collect;
   每个 episode 从归一位姿开始,数据一致性好)
4. 采集完成后验证数据集(`${PANTHERA_DATA_ROOT}/lerobot/local/<repo-id>/<时间戳>/`,
   检查 episode 数、parquet/视频文件)
5. 跑全量单测,把这批改动(quest_bridge.py、collect_cli、controller/cli/ipc、
   init_workspace.sh、测试)提交并同步公司 Gitea 远端(中文提交信息)
