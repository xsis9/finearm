# arm/panthera · Panthera-HT 机械臂驱动层

> **代码已上移(Phase 2)**:daemon 控制器/IPC/适配器真身在 `src/finearm/arms/`
> (规范入口 `finearm-daemon --arm panthera_ht`)。本目录保留**驱动资产与启动
> 环境的权威位置**:SDK 实体、本机标定 `robot_param/`、`init_env.sh`/`run_daemon.sh`;
> `panthera_control/` 下五个模块是转发 shim(旧 import 路径照常工作)。


Panthera-HT 机械臂控制服务(daemon)与 IPC 客户端。**被数据采集
(`deploy/data/collect`)与推理(`deploy/infer`)共用**——2026-11 从采集工具
`panthera_lerobot_control` 包抽离,deploy/arm 按臂分目录(未来多臂各占一格),本目录是 Panthera-HT 的驱动权威位置。

| 文件 | 说明 |
|---|---|
| `init_env.sh` | 环境初始化:建本目录 `.venv` 并安装 SDK 轮子(motor_whl)+ SDK Python 依赖(pyyaml/scipy/pinocchio;需 python3.12 x86_64),与采集的 lerobot 环境解耦 |
| `run_daemon.sh` | daemon 启动 wrapper(`--live/--mock/--hz` + daemon 参数透传);自动优先用本目录 `.venv` |
| `sdk/Panthera-HT_SDK/` | 机械臂 SDK 实体(revision 7f704d9;含 motor_whl 轮子、URDF;由采集工具 init 脚本克隆/校验) |
| `robot_param/` | 本机标定配置(Leader/Follower.yaml + motor_param;serial_id 按本机串口布局改对,URDF 相对引用 `../sdk/...`) |
| `panthera_control/ipc.py` | IPC 客户端/服务端(ControlClient/ControlServer,纯标准库)——采集、推理、Dagger 都只用它 |
| `panthera_control/controller.py` | daemon 主体:1000Hz 控制环、hold/policy/teleop/freedrive 模式、跳变限制与新鲜度兜底 |
| `panthera_control/adapters.py` | 硬件适配:SDK(Panthera_lib)懒加载、mock 适配器、串口权限检查 |
| `panthera_control/constants.py` | 控制参数(KP/KD/限位)与 snapshot schema 帮助函数 |
| `panthera_control/cli.py` | daemon 命令行入口(`python -m panthera_control.cli`) |

## 客户端怎么用

```python
import sys
sys.path.insert(0, "<deploy>/arm/panthera")
from panthera_control.ipc import ControlClient

client = ControlClient("/tmp/panthera_lerobot_control.sock")
client.status()                          # daemon 状态
snap = client.snapshot(max_age_s=0.25)   # {"observation.state": [14], ...}
client.set_mode("policy")                # hold | policy | teleop | freedrive
client.send_policy_action([7 维绝对关节目标])  # 只收 list/tuple,先 .tolist()
client.hold()
```

## 启动 daemon

```bash
deploy/arm/panthera/run_daemon.sh --live --hz 1000 --no-leader --initial-control-mode hold
# 旧命令继续有效(薄 wrapper 转发):
# deploy/data/collect/panthera-lerobot-collect/run_panthera_lerobot_daemon.sh ...(转发 arm/panthera)
```

## 依赖与兼容

- **daemon 环境独立**:`.venv`(init_env.sh 创建)= python3.12 + SDK 轮子 + SDK Python
  依赖(pyyaml/scipy/pinocchio);没建时会回落到采集工具的 `.venv-lerobot`。
  本目录集中全部机械臂资产:SDK 在 `sdk/Panthera-HT_SDK`(PANTHERA_SDK_ROOT 默认值,
  克隆/校验由采集工具的 `scripts/init_workspace.sh` 负责);本机标定在
  `robot_param/`。
- **旧 import 路径不破**:采集工具内 `panthera_lerobot_control.*` 是转发 shim;
  外部新代码请直接用 `panthera_control.*`
- socket 路径与消息协议未变:`/tmp/panthera_lerobot_control.sock`
- 驱动层测试(经 shim)仍在采集工具 `tests/test_panthera_lerobot_control.py`
