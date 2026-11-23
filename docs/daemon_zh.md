# daemon · Panthera-HT 驱动资产

[English](daemon.md) · [中文](daemon_zh.md)

本机机械臂资产目录。daemon **代码**在 `src/finearm/arms/`(规范入口
`finearm-daemon --arm panthera_ht`;本目录的 `run_daemon.sh` 直接驱动它)。
本目录是厂商 SDK 与本机标定的权威位置;驱动**环境**在 `envs/arm/`
(仓库规则:所有 venv 一律在 `envs/` 下)。

| 路径 | 用途 |
|---|---|
| `run_daemon.sh` | daemon 启动器(`--live/--mock/--hz` + daemon 参数透传);优先用 `envs/arm/.venv`,缺失时回落采集 venv |
| `sdk/Panthera-HT_SDK/` | 厂商 SDK 实体(gitignored;由采集工具的 `init_workspace.sh` 克隆/校验) |
| `robot_param/` | **本机标定**(Leader/Follower.yaml + motor_param) |

## 首次搭建

```bash
./envs/arm/init_env.sh              # 驱动 venv(缺失时 daemon 回落到采集 venv)
```

## daemon 生命周期

```bash
# 台架检查(无硬件):
bash arm/panthera/run_daemon.sh --mock --hz 1000

# 实机——标定必须显式传(缺省会解析到 SDK 树副本,串口重排的机器上不对):
bash arm/panthera/run_daemon.sh --live --hz 1000 \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml
# 单臂变体(policy/Quest 场景):加 --no-leader
```

安全起步顺序:daemon 以 **hold** 起步 → 读遥测、手动活动主臂确认每个关节
通道跟随 → 才切模式。采集/推理客户端会**复用**已运行的 daemon(探测接受
模式 ∈ {teleop, policy, freedrive})。

停止时杀 **python** PID(不是 bash 包装):

```bash
ps aux | grep finearm.arms.daemon | grep -v grep   # 找 python PID
kill <PID>
```

## 客户端访问 daemon(任意环境)

```python
from finearm.arms.daemon.ipc import ControlClient   # src 在 sys.path,或已装 finearm

client = ControlClient("/tmp/panthera_lerobot_control.sock")
client.status()                            # daemon 状态 + profile 握手
snap = client.snapshot(max_age_s=0.25)     # {"observation.state": [14 维], ...}
client.set_mode("policy")                  # hold | policy | teleop | freedrive
client.send_policy_action([7 维绝对关节目标])       # 只收 list/tuple(先 .tolist())
client.hold()
```

## 标定说明

- `robot_param/` 是机器本地配置,作为工作副本入库;serial id 必须与本机
  实际串口布局一致。
- 仓库根的 `configs/arms/panthera_ht/` 提供可选覆盖 YAML——解析顺序:
  CLI `--leader-config/--follower-config` > `configs/arms/panthera_ht/*.yaml`
  > SDK 树副本。
