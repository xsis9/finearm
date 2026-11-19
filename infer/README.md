# 推理端(通用 VLA 实时推理 · 适配器架构)

一套 RTC 客户端跑所有模型:执行器/控制环/安全逻辑模型无关(`rtc_client.py`),
模型差异只存在 `adapters/<name>.py` 适配器里,`--model` 选择;机械臂/相机后端
`robot_interface.py` 共用。**新增模型 = 写一个约 50 行的适配器,其他文件都不动。**

> 方法来源:ASAC — "Real-Time Execution of Action Chunking Flow Policies"
> (Black et al., arXiv:2409.01833)。pi05 链路 2026-11-02 实机三级验证全通;
> RTC 骨架的四处理修正记录在 `rtc_client.py` docstring。
> RTC 服务端说明:单连接足够(服务器逐条串行处理,客户端任一时刻最多 1 个在途
> 请求);服务端不需要为 RTC 做任何改动。

**路径约定**:下文命令默认在仓库根目录执行(克隆到任意位置均可);
所有代码内的路径都按 `__file__` 相对定位,不依赖克隆位置。

## 第三方依赖(克隆后先就位,均不入 git)

| 位置 | 用途 | 获取 |
|---|---|---|
| `models/openpi/` | `init_env.sh` 从中本地安装 openpi-client;robot_interface 的 resize 实现也依赖 | `github.com/Physical-Intelligence/openpi` |
| `models/lingbot-vla-v2/` | lingbot 适配器的 websocket 客户端 + panthera robot config/norm stats 参考副本 | 该模型官方仓库 |
| `arm/panthera/sdk/` | Panthera-HT SDK(URDF/wheel),采集工具 `scripts/init_workspace.sh` 自动安装 | `github.com/HighTorque-Robotics/Panthera-HT_SDK` |

缺它们时 `init_env.sh` / 适配器 / 采集工具会明确报错提示。

---

## 推理一个模型:流程与详细命令

### 情况 A · 适配器已存在(现有:`pi05` / `lingbot`)——直接推理

**第 1 步 · 4090 推理机:启动推理服务**(按模型选一,终端保持运行)

```bash
# pi05(openpi/JAX;服务器侧无需任何改动,serve_policy 本身就是 websocket 服务)
cd <4090 上的 openpi 仓库>
uv run scripts/serve_policy.py \
    policy.config=pi05_panthera_p3 \
    policy.dir=checkpoints/pi05_panthera_p3/panthera_p3/2000
#    调试可加 record=True(记录每条请求的 obs/动作,事后核对输入)

# lingbot(lingbot-vla-v2/PyTorch;启动前核对见下「lingbot-vla-2 关键信息」)
# ⚠ 必须 cd 到仓库根再启动:robot config / norm_stats 都按相对路径解析,
#    cwd 不对 → 客户端 reset 时报 FileNotFoundError: configs/robot_configs/panthera.yaml
cd <4090 上的 lingbot-vla-v2 仓库>   # 必须 cd 到仓库根(见下)
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/微调ckpt/hf_ckpt --use_bf16 true --use_compile true \
  --use_length 25 --port 8006
```

**第 2 步 · AU 执行机 · 终端 1:锁相机 + daemon**

```bash
# ① 固定相机参数(推理前必做;恢复 bolt_test2 采集时刻的曝光/增益/WB 并读回验证,
#    观测外观必须与训练数据一致,否则输入分布漂移、成功率直接崩。
#    UVC 重插/断电会复位,所以每次都要跑;robot_interface 启动时还会自动再调,双保险)
bash utils/lock_cameras.sh
#    PANTHERA_CAM_LOCK=路径 覆盖脚本位置;PANTHERA_CAM_LOCK_SKIP=1 跳过(仅调试)

# ② 机械臂 daemon(单臂、policy 就绪;robot_interface 连它的 socket)
bash arm/panthera/run_daemon.sh --live --no-leader --hz 1000
#    (首次先跑一次 arm/init_env.sh 建驱动环境;保持运行,停止时自动缓动归位再抱闸。
#     旧路径 data/collect/panthera-lerobot-collect/run_panthera_lerobot_daemon.sh 等价)
```

**第 3 步 · AU 执行机 · 终端 2:ssh 隧道**(把 4090 端口映射到本机,之后 `--host` 一律 127.0.0.1)

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<4090-IP>    # pi05(端口 8000)
ssh -N -L 8006:127.0.0.1:8006 <user>@<4090-IP>    # lingbot(端口 8006)
```

**第 4 步 · AU 执行机 · 终端 3:自检 → 基准 → 预热**

```bash
cd infer
PY=.venv/bin/python          # 没有就先跑一次 ./init_env.sh(见「目录结构」)

# 自检(不动臂):相机取流、图像方向/曝光、观测时延;肉眼看 /tmp/rtc_obs 两张图
"${PY}" robot_interface.py --selftest --dump-obs /tmp/rtc_obs

# RTT 基准(按适配器):p50/p95 + 建议 k;兼预热——第一条含服务端 jit/compile
# (10~60s)已被剔除,别让真机第一次推理等编译
"${PY}" bench_rtt.py --model <name> --host 127.0.0.1
#    p95=100ms → k≈5;200ms → k≈8;300ms → k≈11(客户端自适应,通常不用手动)
```

**第 5 步 · 推理(安全阶梯,逐级放行)**

```bash
# 一级 · 不接硬件空跑:验证适配器/返回解析/chunk 切换(仍连 4090 服务)
"${PY}" rtc_client.py --model <name> --host 127.0.0.1 --mock --max-steps 300
#      要同时走完整 daemon IPC:终端 1 改用 mock daemon
#      (./run_panthera_lerobot_daemon.sh --mock --hz 1000)+ 下一级用 --mock-images

# 二级 · 实机 dry-run:真相机 + 真 daemon;归位会实际运动,推理只打印不下发
"${PY}" rtc_client.py --model <name> --host 127.0.0.1 --dry-run --duration 60
#      或连 mock daemon + 全零图(不动真臂):--mock-images

# 三级 · 实机低速真控:≈18°/s 起步,断档即停;工作区清空、急停伸手可及
"${PY}" rtc_client.py --model <name> --host 127.0.0.1 --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30

# 四级 · 正常推理:限幅逐步放宽到标定值或去掉
"${PY}" rtc_client.py --model <name> --host 127.0.0.1 --task "<训练prompt原文>"
```

常用参数速查:

| 参数 | 说明 |
|---|---|
| `--task` | 任务指令,**必须与训练 prompt 逐字一致** |
| `--home` | 归位位姿(7 值);优先级:`--home` > 环境变量 `PANTHERA_RTC_HOME` > 适配器 `DEFAULT_HOME`(各模型训练起始位姿,已记录在适配器里);空串跳过归位 |
| `PANTHERA_CAM_FRONT/WRIST` | 相机源,默认 /dev/video4、/dev/video2 |
| `--k/--blend/--starve/--hz` | 见下方「调参表」 |
| `--pos-lo/--pos-hi/--vel-limit` | 显式限幅(限幅只在这一处管理,臂侧只留 0.25rad/tick 兜底) |

### 情况 B · 新模型 —— 先写一个适配器(~50 行),再走情况 A

**1 · 拷骨架建 `adapters/<name>.py`**(文件名 = `--model` 参数值;不要下划线开头):

```python
"""<模型名> 适配器。服务端:<怎么起、在哪台机器>。"""
import numpy as np
from . import AdapterBase


class Adapter(AdapterBase):
    NAME = "<name>"
    DEFAULT_PORT = <端口>
    OBS_STYLE = "openpi"   # 或 "raw",见下方约定
    DEFAULT_HOME = "j1,j2,j3,j4,j5,j6,gripper"  # 该模型训练数据的起始位姿

    def __init__(self, host, port=None):
        # 连接该模型的推理服务;import 放这里(延迟),保证模块 import 无副作用
        raise NotImplementedError

    def reset(self):
        pass  # 需要握手(如 lingbot 发 robo_name)就写在这里

    def build_obs(self, obs, task):
        # obs 是 robot.get_obs(self.OBS_STYLE) 的结果,映射成该模型的请求键名
        raise NotImplementedError

    def infer(self, payload):
        # 发请求 → 解析返回 → (T,7) 绝对关节目标;维度/finite 校验在这里做
        raise NotImplementedError
```

**2 · 离线自测**(不连服务器、不起机械臂):

```bash
cd infer
.venv/bin/python - <<'EOF'
from adapters import mock_obs
from adapters.<name> import Adapter
a = Adapter.__new__(Adapter)            # 不触发 __init__(不连接)
p = a.build_obs(mock_obs(Adapter.OBS_STYLE), "test")
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in p.items()})
EOF
```

**3 · 4090 起服务 → 情况 A 第 2~5 步**(bench_rtt 看 RTT/返回形状 → 阶梯)。

适配器约定(`rtc_client.py` 执行器的假设,必须满足):

- **OBS_STYLE 二选一**:`"openpi"`(224×224 预 resize + state 前 7 维,客户端预处理)
  / `"raw"`(原始 640×480 + 完整 14 维 state,后处理交给服务端);
  两种 `robot_interface.py` 都已支持,适配器只声明用哪种
- **动作 = (T,7) 绝对关节目标**,顺序/单位与训练数据 `action` 一致;
  归一化/逆变换在服务端完成;chunk 时间步 = 1/30s(与控制频率一致)
- **DEFAULT_HOME** = 该模型训练数据起始位姿(训练集 episode 0 首帧 state 前 7 维),
  起始位姿不一致会让"开局就漂移"的坑复发
- **import 无副作用**(连接延迟到 `__init__`):注册表扫描/离线自测只 import 不实例化
- 需要 robot config / norm stats 等服务端文件的模型,核对项写进本 README 对应小节

## 目录结构

| 路径 | 说明 |
|---|---|
| `rtc_client.py` | **通用 RTC 推理客户端**:30Hz 主循环 + 异步推理线程 + chunk 对齐切换/blend/自适应 k/显式限幅;`--model` 选适配器 |
| `bench_rtt.py` | 通用 RTT 基准(按适配器),输出 p50/p95 与建议 k |
| `adapters/` | **模型适配器,唯一模型相关层**(现:pi05 / lingbot);`__init__.py` 是注册表 |
| `robot_interface.py` | 机械臂/相机接口(共用):daemon IPC、双相机 MJPG 取流(断流重开/帧率自检)、锁相机、go_home 归位、`mock_images`、`--selftest` 自检;`get_obs(style)` 双风格 |
| `init_env.sh` + `requirements.txt` | 环境初始化:建本目录 `.venv`(openpi-client 从 `models/openpi` 本地安装,仓库自包含,拷走即用) |
| `.venv/` | 推理环境(所有模型共用) |

## lingbot-vla-2 关键信息(适配器 `lingbot.py` 对应的服务端)

### 启动前核对(4090,一次性)

- 仓库位置:4090 上克隆的 lingbot-vla-v2(相对路径解析要求从仓库根启动,见上)
- 微调 ckpt 是**含 training config 的 `hf_ckpt` 目录**(不是裸 safetensors);
  `--use_length` 必须 ≤ 训练 action horizon
- 仓库里必须已有 `configs/robot_configs/panthera.yaml` 和 `assets/norm_stats/panthera.json`,
  且 **norm_stats 与训练同一份**(norm_stats 是全量训练数据合并统计的;
  训练/部署不同份 → 动作幅度异常)。缺了就从执行机拷:
  `scp models/lingbot-vla-v2/{configs/robot_configs/panthera.yaml,assets/norm_stats/panthera.json} <user>@<4090-IP>:<仓库路径>/<对应目录>/`
- `export QWEN3VL_PATH` 指向 Qwen3-VL-4B-Instruct(base 测试时装过)
- 客户端 reset 时服务端报 `FileNotFoundError: configs/robot_configs/panthera.yaml`
  = 文件缺失或服务端 cwd 不在仓库根(2026-11-11 实测踩过)

### 语义与坑

- **显存**:4090 24G 必须 bf16(权重+激活 ~14-16GB),fp32 光权重就装不下;
  `--use_compile true` 首次请求编译约 1 分钟(先跑 bench_rtt 预热,正常现象)
- **协议**:连接后适配器自动 `reset("panthera")` → 服务端加载 panthera.yaml 构建
  FeatureTransform;obs 发数据集原始特征名(`observation.state` 14 维 /
  `observation.images.front` / `.wrist` / `task`)。返回 key = **服务端那份 robot config
  里的动作名**:4090 训练 config 是单块 `action` (T,7)(2026-11-11 实测);AU 参考副本
  (`models/lingbot-vla-v2`)的 config 是拆分式 `action.arm.position`
  + `action.effector.position`。适配器两种都兼容。**训练 config 是权威——不要拿
  AU 的 config 覆盖 4090 的**(ckpt 按它训的,服务端逆变换也按它做)
- **图像**:发原始 640×480,服务端自己 resize 到 256×256;客户端不翻转不裁剪
  (采集存的就是相机原始帧)。相机:front = MX Brio `/dev/video4`,
  wrist = 4K_USB_CAMERA `/dev/video2`
- **RTC**:服务端不自带(串行阻塞循环;`chunk_ret=False` 模式下中途观测被忽略),
  但默认 `chunk_ret=True` 语义与 RTC 执行器完全兼容,**服务端不用改**,
  客户端直接 `rtc_client.py --model lingbot`
- **输出噪声(2026-11-11 静态实测,机械臂静止同一 obs 连发 3 次)**:
  ① 采样是随机的——模型动作头 flow-matching 每次推理重抽噪声
  (`modeling_lingbot_vla_v2.py:1190` `torch.randn`),同 obs 重叠段分歧 0.5~0.6 rad,
  服务端无 seed 参数;② chunk 内部高频振荡——步方向交替率 0.66、步增量峰值
  66 mrad/步。表现 = 机械臂发抖。**现有旋钮缓解**:`--vel-limit`(限速压尖峰)
  + `--blend 3`(更长接缝过渡);根治要动模型侧(去噪步数/固定采样 generator/
  微调数据动作平滑度/fp32 对比,官方提示 bf16 影响动作平滑度)
- **action chunk = 50 步**(训练 horizon;pi05 是 30)。服务端 `--use_length` 截断返回
  (当前 25);RTC 下建议起服务用 `--use_length 50` 用满 chunk——切换更少、抗 RTT 抖动
  (执行器对 chunk 长度无假设,k 上限自动随实际长度适配:pi05→25,lingbot→45)
- RTT 实测 ~130ms/chunk(4090D 本地 bf16+compile;走 ssh 隧道 p50≈325ms);
  `--task` 必须与训练 prompt 逐字一致;
  归位位姿 = 训练数据起始(适配器 DEFAULT_HOME 已记录)

## RTC 原理(为什么需要)

**传统 chunk 执行**:执行完整 chunk(如 25 步 ≈ 0.83s)→ 发 obs 阻塞等推理(RTT)
→ 新 chunk 从头执行。有效延迟 = chunk 时长 + RTT ≈ 1s,且新 chunk 基于过时观测。

**RTC 两个组件**(全在客户端):
1. **连续推理**:不等当前 chunk 执行完,在**还剩 k 步**时就用**最新观测**异步发起
   下一次推理。推理耗时被这 k 步的执行时间(k×33ms)掩盖。
   要求 k ≥ RTT×30,否则 chunk 断档;有效相位滞后 ≈ k×33ms。
2. **chunk 对齐切换**:新 chunk 到达时,机械臂已推进到旧 chunk 某位置,不能从第 0 步
   执行(会"倒带")。在新 chunk 中找与"下一个要执行的动作"最接近的索引,从那里接续,
   并用 blend 步线性混合消除跳变。

```
时间 ──────────────────────────────────────────────▶
执行线程 30Hz:  A0 A1 ... A[30-k] B[j] B[j+1] ...
                     │        └── 新 chunk 从对齐索引 j 接续
                     └── 剩 k 步时: 取最新 obs 异步请求 B
推理线程:            obs ──── RTT ────▶ B 就绪(在 A 执行完前)
```

## 调参表

| 参数 | 含义 | 建议 |
|---|---|---|
| `k` | 剩 k 步时触发重规划 | 自适应(默认)。固定时 = ceil(p95_RTT×30)+2。上限自动随 chunk 长度(pi05 30→25;lingbot 50→45)。k 越小动作越新鲜,但必须 ≥ RTT 折算步数,否则断档 |
| `--hz` | 控制频率 | **30**(必须与采数 fps 一致,否则动作播放速度错) |
| `--blend` | 切换混合步数 | 2~3。动作有跳变/顿挫 → 加大;动作迟滞 → 减小 |
| `--starve` | 断档策略 | 调试期 `hold`;正式跑建议 `stop`(断档=异常,停比僵住安全) |
| `--vel-limit` | 每步限速 | 显式写在客户端;同时核对臂控制器侧限制(限幅要一处管理) |

k 与滞后/断档的关系:k=5 → 滞后 ~165ms;k=10 → ~330ms。RTT 允许时 k 尽量小。

## 安全检查清单(每次上真机前)

- [ ] `bench_rtt.py` 跑过,p95 RTT 无明显抖动(抖动大 → 先解决网络,RTC 救不了网络)
- [ ] 限幅只在这一处:客户端 `--pos-lo/--pos-hi/--vel-limit`;臂控制器侧旧限制已核对/取消
- [ ] `robot_interface.py --selftest --dump-obs` 看过图像:与采数同相机、同分辨率、不镜像不裁剪;state 前 7 维顺序/单位与训练数据一致;action 单位一致
- [ ] `--task` 字符串与训练完全一致
- [ ] 归位位姿 = 该模型训练数据起始位姿(适配器 DEFAULT_HOME 已记录,确认没被环境变量覆盖错)
- [ ] 第一次跑用 `--starve stop` + 小 `--vel-limit`
- [ ] 手边有 E-stop;Ctrl-C 后 `close()` 里让臂安全停止;软件 HOLD 不是硬件急停,全程保持物理急停可达

## 故障排查

| 现象 | 原因 | 处置 |
|---|---|---|
| "chunk 耗尽"警告刷屏 | k 小于 RTT 折算步数,或网络抖动 | k 自适应下仍出现 → 查网络;或 `--k` 手动加大 |
| 切换极频繁 + 偶发断档(hold 桥接) | RTT 相对 chunk 时长过大(服务端 `--use_length` 把 chunk 截短了) | lingbot 用 `--use_length 50` 起服务用满 chunk(≤训练 horizon);pi05=30 不截 |
| 动作跳变/顿挫 | 对齐索引抖动 | `--blend` 加到 3;检查 state 反馈是实测值(不是上次指令值) |
| 机械臂持续发抖(lingbot) | 模型输出高频振荡 + 随机采样(见 lingbot 小节实测) | `--vel-limit` 限速 + `--blend 3`;根治在模型侧 |
| 动作整体滞后、越走越偏 | 控制频率≠30Hz;或 obs 采集慢 | 确认 `--hz 30`;get_obs 用最新帧,不要排队帧 |
| 首条请求等很久 | 服务端 jit/compile 预热 | 正常;启动后先跑 `bench_rtt.py` 预热 |
| 推理失败/KeyError | 图像格式、state 维度、返回 key 与服务端不符 | 对照适配器 build_obs/infer 与该模型服务端协议(各模型小节) |
| RTT 忽大忽小 | 4090 上有其他任务抢占 | 独占 GPU;确认没有别人在跑训练 |
| lingbot 服务器起不来(显存不足) | 非 bf16 | 确认 `--use_bf16 true`;关 `--use_compile` 省编译期显存 |
| lingbot 动作整体正确但幅度异常 | 4090 侧 norm_stats 与训练不是同一份 | 核对 `assets/norm_stats/panthera.json` |
| lingbot 动作方向合理但任务做不对 | `--task` 与训练 prompt 不一致;或场景摆放/机位与训练数据差太远 | 逐字核对 prompt;按训练 SOP 摆放 |
| lingbot 起步就往错方向走 | 起始位姿 ≠ 训练起始 | `--home` 用训练集 episode 0 首帧 state 前 7 维 |
| 归位永远到不了 0.03 rad | 腕关节带载静态下垂 ~0.09 rad | 容差 0.1(robot_interface 默认),不要再调小 |
| 相机 `VIDIOC_DQBUF: No such device` | USB 断连 | `dmesg` 确认,换 USB 口/带供电 hub,关 usbcore autosuspend |
| 相机协商失败(分辨率/帧率不对) | 相机残留坏配置 | 重新插拔,`v4l2-ctl --list-devices` 核对后重跑 |

## 相关

- 采集(训练数据):`data/collect/panthera-lerobot-collect/`
- 相机参数档案(唯一权威):`utils/lock_cameras.sh`
- lingbot-vla-v2 仓库(本地参考副本):`models/lingbot-vla-v2/`
