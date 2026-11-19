# finearm 框架化重构计划

## Context

`/home/xsx/internship/baiqin/deploy` 是具身智能(VLA π0.5)机械臂微调 + RL 后训练项目,要重构为标准、鲁棒、可扩展的开源 GitHub 框架 **finearm**。当前痛点(已探索验证):

- **deploy 根不是 git 仓库,自研 ~20k 行代码零版本控制**(最大单点风险);panthera 专属配置散落在 vendored 仓库的未跟踪文件里(lingbot 的 panthera.yaml/json)
- 无根 README/pyproject/tests/CI;train/ 空目录;utils/ 仅 1 个脚本;层间依赖全靠 sys.path.insert + 三机器绝对路径硬编码
- snapshot schema 14/7 维硬编码 4 处(constants/ipc/两个 LeRobot 插件/续采校验)——换臂要同步改 4 处
- train_stage2.py 2067 行单文件混"RL框架+算法+panthera环境"三类代码
- 已有两套互不统一的动态加载约定(infer adapters 目录扫描、RL --robot_module)
- RTC 执行器逻辑双份(infer/rtc_client.py 与 train_stage2 PipelinedChunkProvider)
- `infer_stage2.py:53-54` import 期 NameError 真实 bug(signal_handler 未导入)

正面资产(重构的基础,不推倒):arm daemon 架构(IPC 协议臂无关)、RobotAdapter 基类、data 采集鲁棒性(预检/续采校验/fail-safe)、infer adapters"文件名即注册表"、RL 侧 robot_interface 契约分层、RL 86 + data 153 个现测试。

## 已确认决策(用户)

1. **渐进式迁移**:RL stage2 真机 campaign 进行中(续点 v3_2_2/ep60),现有入口(train_stage2.py、infer_stage2.py、serve_rlt.py、rtc_client.py、panthera-lr-collect)迁移期保持可用,逐模块切换
2. **pyproject + uv**;3. **包名/仓库名 finearm**;4. **MIT**;5. **README 英文主体+中文版,其余文档英文**

## 设计原则

1. campaign 期间 4060 执行机 checkout 钉在 `campaign/v3` 分支(tag),框架工作全在 main,**只在 campaign 断点快进合并+冒烟**;`RL/rlt_panthera/checkpoints/`、`data/` 产物目录物理位置永不移动
2. 迁移三步曲(已被本仓库 data 域 shim 验证):真身移入新包 → 旧路径变转发 shim → 切换日后删 shim
3. `finearm.core` 零第三方依赖(纯标准库),可装进任何 venv(含 4090 openpi venv)——"部分坏了不影响整体"的根基
4. 插件发现用**目录扫描 + lazy import**(加文件即接入,与 infer/adapters 现状一致),不用 entry points(需打包安装才能发现)
5. 每阶段一个 PR,回退 = git revert 该阶段 PR;**删旧入口的切换日判据**:新入口连续 2 次真机会话无回退需求

## 一、目标目录树(终态)

```
deploy/                                  # git 仓库根
├── pyproject.toml  uv.lock  README.md  README_zh.md  LICENSE  CONTRIBUTING.md
├── .github/workflows/ci.yml  .gitignore(扩充)
├── src/finearm/
│   ├── core/          # 纯标准库:registry.py(目录扫描)/config.py(JSONC+机器档案)/paths.py/norm.py/io.py/rng.py
│   ├── arms/
│   │   ├── __init__.py      # load_profile()/load_arm()/available()
│   │   ├── profile.py       # ArmProfile 纯数据声明(见下)
│   │   ├── daemon/          # 臂中立 daemon:controller 循环骨架 + ipc + cli
│   │   └── panthera/        # 新臂 = 新增此层级目录,别处零改动:adapter.py/gains.py/lerobot_backends.py
│   ├── infer/
│   │   ├── adapters/        # 原样迁入(pi05.py/lingbot.py)+ AdapterBase,文件名即注册表
│   │   ├── executors/       # base.py / rtc.py / stop_and_go.py
│   │   ├── camera.py  cli.py
│   ├── data/
│   │   ├── collector/       # Collector 接口 + teleop/(现 collect_cli 主流程)
│   │   ├── dagger/          # 仅接口桩 + 设计文档(不实现)
│   │   └── sample_coordinator.py  cli.py
│   └── rl/
│       ├── loop/            # 框架:episode_runner / console / metrics / checkpoint / chunk_provider
│       ├── replay/          # TransitionSource / ReplayManager / journal / stride-2
│       ├── boundaries.py    # WindowReceipt / BoundaryCollector 协议(核心接缝,见下)
│       ├── algorithms/rltoken/   # networks/learner/critical(纯算法)  + recap/ 占位
│       ├── envs/            # robot_interface.py(契约)+ panthera/robot.py
│       ├── serving/         # client.py(VLAService)+ serve_rlt.py
│       └── scripts/         # infer_deploy(infer_stage2 后继)、precompute、plot 等
├── plugins/
│   ├── lerobot_robot_panthera_ht/        # 顶层模块(LeRobot 按发行名=模块名发现,不可并入命名空间)
│   └── lerobot_teleoperator_panthera_ht/
├── configs/
│   ├── machines/{a100,4090,4060,localhost}.json   # 机器事实:host/port/路径/socket/相机设备
│   ├── arms/panthera_ht.json
│   ├── models/lingbot/{panthera.yaml,norm_stats/panthera.json}  # 自 vendored 未跟踪文件收编(canonical)
│   ├── rl/stage2_panthera.jsonc          # 仅超参
│   └── collect/*.jsonc
├── third_party/
│   ├── versions.json      # 上游基线 commit 记录(openpi 同步用)
│   └── patches/*.patch
├── scripts/fetch_models.sh  envs.sh  sync_4060.sh
├── train/                 # SFT 配方(不复制模型源码):openpi_pi05/ lingbot_v2/ 各=README+config+launch wrapper
├── tests/{core,arms,infer,data,rl}/
├── docs/
├── models/
│   ├── openpi/            # 用户修改版源码入库(含 panthera_policy;checkpoints/.venv 等产物 gitignored)
│   └── lingbot-vla-v2/    # gitignored,fetch_models.sh 可选布置
├── arm/panthera/{sdk,LiteArm-A1}    # gitignored
└── RL/rlt_panthera/{checkpoints,data}  # gitignored 产物;代码迁完删
```

## 二、包与环境(uv 多环境,不合并)

核实:三个 venv 全是 Python 3.12.2,分歧在包不在解释器(lerobot==0.5.1 钉死、torch 版本差、openpi-client 本地 numpy 补丁)。**单 pyproject + extras + `UV_PROJECT_ENVIRONMENT` 多环境**:

```toml
[project]
name = "finearm"; requires-python = ">=3.11"; dependencies = []   # core 纯标准库
[project.optional-dependencies]
robot = ["numpy>=2,<2.3", "pyyaml>=6"]
data  = ["lerobot==0.5.1", "opencv-python>=4.8"]
infer = ["numpy>=2,<2.3", "opencv-python>=4.8", "websockets>=13", "msgpack>=1.1"]
rl    = ["torch>=2.2", "numpy>=2,<2.3", "websockets>=13", "msgpack>=1.1"]
dev   = ["pytest>=8", "ruff>=0.8"]
[tool.uv.sources]
openpi-client = { path = "models/openpi/packages/openpi-client" }
[project.scripts]
finearm-daemon = "finearm.arms.daemon.cli:main"
finearm-infer = "finearm.infer.cli:main"
finearm-collect = "finearm.data.cli:main"
finearm-train-stage2 = "finearm.rl.scripts.train_stage2:main"
```

`scripts/envs.sh`:`UV_PROJECT_ENVIRONMENT=envs/{robot,data,infer} uv sync --extra …`(infer 环境同时 `--extra rl`)。一个 lock、三个环境。4090 serve 侧 `uv pip install --no-deps -e .` 只装纯 py 的 finearm。迁移期现有三个 venv 原样保留为真机环境,新环境逐域接管后再删。**B 计划**(uv 单锁解析冲突时):plugins+data 拆独立 uv project,Phase 0 即在 CI 验证 lock 可解析。

## 三、统一注册表与四类插件协议

`finearm/core/registry.py`:`scan_names(package)`(列名不 import)+ `load_attr(package, name, attr)`。三套旧约定收敛为 `--arm <id>` / `--algorithm <id>` / `--model <id>` / `--method <id>`,模块:类语法保留为逃生口。

**(a) 机械臂 ArmProfile**——snapshot schema 从 4 处硬编码变每臂声明一次。**SDK 与标定的归属**:src/finearm/arms/<臂>/ 只放我们写的代码(adapter 五方法 + ArmProfile 声明 + 增益表);厂商 SDK 留在外部(gitignored,如 arm/panthera/sdk),由 `sdk_root` 配置指向——adapter 不感知 SDK 位置,新臂自带 SDK 放哪都行;本机标定(robot_param/*.yaml)归 configs/arms/,属用户可改配置:
```python
@dataclass(frozen=True)
class ArmProfile:
    arm_id: str; joint_names: tuple[str, ...]
    state_feature_names: tuple[str, ...]   # 14 项,顺序即 snapshot["observation.state"]
    action_feature_names: tuple[str, ...]  # 7 项,顺序即 snapshot["action"]
    socket_path: str; default_home: str | None; joint_limits: tuple[Range, ...] | None
```
每臂目录 `__init__.py` 声明 `ARM_ID/PROFILE/make_pair/make_follower_only`。四处消费切换:make_snapshot 按 PROFILE 组装;ipc 校验长度取自 profile + IPC 握手新增 profile 应答(向后兼容);两个 LeRobot 插件 keys 参数化;collect_cli 续采校验同源。daemon 泛化 `finearm-daemon --arm panthera_ht`;panthera 的 KP/KD/摩擦移 `arms/panthera/gains.py`(现 constants.py)。controller.py 控制环/模式机/IPC 协议保持臂中立(现状已中立)。

**(b) 模型 adapter**:保持现 AdapterBase 契约不变(NAME/DEFAULT_PORT/OBS_STYLE/DEFAULT_HOME/reset/build_obs/infer,已验证于 infer/adapters/__init__.py:28-62),仅迁入 + lazy load。

**(c) RL 算法**:`algorithms/<name>/__init__.py` 声明 `ALGORITHM_ID`、`REQUIRED_EMBEDDINGS: tuple[str,...]`(rltoken → ("z_rl",))、`build(config, spec)`。

**(d) 采集方式 Collector**(DAgger 预留,只立契约):
```python
class PolicySource(Protocol):           # DAgger 挂点:策略从哪来(由 finearm.infer / rl.serving 提供)
    def infer(self, obs) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...
class Collector:
    method_id: str                       # "teleop" / "dagger"(目录扫描注册)
    def setup(self, ctx: CollectContext) -> None: ...
    def run_session(self, ctx: CollectContext) -> int: ...
```
teleop/ 收编现 collect_cli 主流程;dagger/ 占位——依赖的基建(daemon policy 模式 + submit_policy_action + freedrive 接管 + sample_coordinator 同拍配对)已全部存在,将来 DAgger ≈ 一个新 collector 文件 + PolicySource 复用。

## 四、infer:stop-and-go 执行器

只动 infer 域,adapters 零改动。现 RTCRunner(rtc_client.py:39)抽为执行器策略,`--exec-mode {rtc,stop-and-go}`:
```python
class ChunkExecutor:
    name = "?"
    def __init__(self, adapter, *, ctrl_hz=30.0, pos_limits=None, vel_limits=None, dry_run=False, verbose=True)
    def run(self, robot, task: str, max_steps: int | None = None) -> ExecutorStats
class StopAndGoExecutor(ChunkExecutor):   # 取观测→阻塞推理→执行 n_action_steps 步→循环
    name = "stop-and-go"                   # 参照 train_stage2 --replan_every_window 语义
```
实现路径:RTCRunner 整体搬迁不改逻辑(为 RTCExecutor),StopAndGo 独立新文件,共享只有 _clamp;对齐/混合函数提为模块级供 rl/loop 复用。旧 rtc_client.py 变 shim(默认 exec-mode=rtc,行为等价)。

## 五、RL 三分(先搬家后拆分)

**5a 纯搬移**(文件映射,不改逻辑):networks/critical → algorithms/rltoken;learner.py 拆出 core/io.py+core/rng.py(atomic_torch_save/capture_rng_state 等出狱);norm.py 去 `/root/xsx` 路径 → core/norm.py;replay.py 拆三(core/io.py 的 journal/fsync、rl/replay/ 的 TransitionSource/stride-2、rl/loop/runtime.py 的 RerunVisual/TakeoverState);PANTHERA_JOINT_NAMES 双份定义(replay.py:41-49、train_stage2.py:106-109)归一至 ArmProfile;train_stage2 的 ConsoleInputManager(:125-292)→ rl/loop/console.py、PipelinedChunkProvider(:294-410)→ chunk_provider.py、VLAClient(:479-528)→ serving/client.py、metrics/ckpt → loop/;robot/panthera_robot.py 删 `_ACQ_CANDIDATES`(:64-68)改 import finearm.infer;**infer_stage2.py 修 signal_handler bug(:53-54)并解耦 `import train_stage2`(:44-45)**。

**5b 接口化**:EpisodeRunner 经协议回调;RLTConfig 拆四类(算法结构留 rltoken / TrainConfig / ExecutionConfig / ArmProfile,machine config);**核心接缝——WindowReceipt/BoundaryCollector**:
```python
@dataclass
class WindowReceipt:
    window_index: int; executed_actions: Tensor; reference_chunk: Tensor
    reference_embeddings: dict[str, Tensor]; reference_age_steps: int  # 0=边界新鲜;>0=RTC 陈旧
    boundary_obs: Observation | None; source: str  # "policy"|"human"|"mixed"
class BoundaryCollector(Protocol):
    def on_window(self, receipt) -> None; def on_takeover(self, obs) -> None
    def on_terminal(self, obs, success) -> None
    def materialize_pending(self, service) -> None      # 现 DeferredBoundarySnapshot 后台物化
    def collect_episode(self) -> list[Transition]       # stride-2 规则归算法侧
```
**红线**:rtc 模式产出 transition 必须与现 train_stage2 逐字段一致(用现 journal 做 golden 对拍测试 `tests/rl/test_boundary_equivalence.py`);rltoken 默认沿用 trigger-time 语义,不悄悄改变学习信号。VLAClient 泛化:`infer() -> VLAInference(actions, embeddings: dict)`(键由算法声明,握手校验),serve_rlt 按 REQUIRED_EMBEDDINGS 装配,框架不见 z_rl。

**5c 算法注册**:`--algorithm rltoken` 目录扫描;recap/ 占位 README 说明契约。

## 六、train/ 与 models/(不合并;models = openpi 修改版入库,train = 臂参数化微调流水线)

**models/openpi:用户修改版直接入库**(2026-11-18 决定):
- 入库 = 用户在训练机修改过的 openpi 源码(**含 panthera_policy、pi05_panthera_p3 配置;本地 models/openpi 是原版,入库前先从训练机同步修改版作为 canonical**),保留上游 LICENSE(Apache-2.0 允许修改再分发);third_party/versions.json 记录官方基线 commit,便于日后对 diff/同步上游
- 不入库:checkpoints/.venv/训练数据/wandb 产物(.gitignore 排除);venv 环境不可迁移,由 uv lock 固化复现
- 收益:纠缠点⑩(serve_rlt 依赖 4090 仓库外 fork openpi)消解,4090 直接用仓库内源码 serve;clone 即含主线模型;原 third_party/openpi-client 副本方案取消,uv source 直指 models/openpi/packages/openpi-client
- lingbot 不上传:lingbot-vla-v2 保持 gitignored,fetch_models.sh 仅剩它(可选用,lingbot adapter 保留作第二模型线示例)

**train/ = "任意已接入臂 × 模型"微调流水线**(目标:用户 clone 后在自己指定的机械臂上微调):
```
finearm-train openpi_pi05 --arm panthera_ht --dataset <repo_id>   # 用户一条命令
  = prepare_data.py   # LeRobot 数据集校验 + norm stats 计算(维度/键名自 ArmProfile)
  → make_config.py    # 模板 + ArmProfile + 数据集 meta → 自动生成该臂该数据集训练配置
  → launch.sh         # 调 models/openpi/scripts/train.py + 生成的 config(训练机)
```
- 不重写训练器:openpi train.py 已成熟,我们只做"数据准备 + 配置生成 + 启动 + checkpoint 对接 infer/rl"这层 glue
- 新臂微调 = 接入 arms 插件(一次性)→ data/ 采集 → 上面一条命令;关节名/维度/限位/相机键由 ArmProfile 单一来源自动注入,用户不手改模型配置
- train/{openpi_pi05,lingbot_v2}/ 各含 README + 配置模板 + prepare/make/launch 三件套
- RL/RL-Token-Pi05-open(纯参考已移植完)Phase 6 移出

## 七、配置体系:保持 argparse+JSONC,加机器档案层

不引 hydra/YAML 新机制(train_stage2.py:565-698 的 JSONC→set_defaults→CLI 覆盖已实现且有测试覆盖,推广之)。`core/config.py`:`load_jsonc/machine_config(优先级 --machine > FINEARM_MACHINE > hostname 匹配 > localhost)/apply_overrides(多层 set_defaults,未知键报错)`。机器事实(server_host/stats_path/dataset_root/socket/相机设备/sdk_root)只进 configs/machines/*.json;实验配置只放超参;消灭三处绝对路径(norm.py:30、configs/stage2_panthera.json:15、_ACQ_CANDIDATES)。

**配置与插件的边界**:改参数(机器/路径/超参/臂增益标定)只动 configs/,零代码;新增能力(新机械臂/新模型/新算法/新采集方式)= 新增插件文件,不是配置——每臂硬件协议不同必须写 adapter 代码,这是物理下限。

## 八、git 与 CI

- Phase 0 第一件事 `git init`;.gitignore 扩充:models/、arm/panthera/{sdk,LiteArm-A1}、RL/rlt_panthera/{checkpoints,data}/、RL/RL-Token-Pi05-open/、data/{datasets,logs}/、envs/、**/.venv*
- CI(GitHub Actions,纯 CPU/mock,硬件永不进 CI)按域分 job:lint(ruff)/ rl(86+新增对拍测试)/ data(153 测试)/ core+arms(daemon/ipc/profile)/ infer(adapters mock + 两执行器 mock 循环)
- README 三层:根(一屏:架构图、quickstart=clone+fetch_models+envs.sh+三条命令、各域链接)+ 各域一篇(自现 infer/data README 收编)+ docs/(quest2ros 手册等正式位置);英文主体 + README_zh.md

## 九、分阶段实施

| Phase | 内容 | 工作量 | 验证 | 切换/回退 |
|---|---|---|---|---|
| **0** git init+骨架 | git init、.gitignore(lingbot/SDK/checkpoints/数据/venv 排除)、**openpi 修改版入库(先从训练机同步含 panthera_policy 的 canonical 副本)**、pyproject(空 finearm+dev extra)、ruff、README 骨架、CI(lint+现三组测试按目录 job)、versions.json 记录 openpi 基线 | 半天,零行为变化 | CI 全绿;train_stage2 `--dry_run`、`panthera-lr-collect --help`、rtc_client `--mock`(原 venv)不受影响 | 删新增文件即回退,零风险 |
| **1** core+配置 | core/{registry,config,paths,norm}.py、configs/machines/*;train_stage2/collect_cli 路径解析改走 core.paths(行为等价),不碰主逻辑 | 1 天 | RL 86+data 153 测试;4060 断点 `--dry_run` 对拍 | revert;campaign 机可不合并 |
| **2** arms 域 | ArmProfile+daemon 泛化+snapshot schema 单点化(§三 a)+IPC profile 握手(向后兼容)+panthera_control 真身迁 src、原位变 shim+LeRobot 插件 keys 参数化+collect_cli 硬编码(collect_cli.py:568/588/619)改 profile 驱动 | 2-3 天 | data 全测试+mock 采集+真机 5-episode 采集+daemon 重启复用;RL connect_only 冒烟 | 切换日=采集批次间隙;shim 反转即回退 |
| **3** infer 域 | executors 抽取+StopAndGoExecutor+finearm-infer CLI+camera 收编;rtc_client 变 shim(默认 rtc);adapters 迁入(lingbot 的 sys.path 探测改 core.paths) | 1-2 天 | `--mock --exec-mode rtc` 与旧 rtc_client 输出对拍;`--exec-mode stop-and-go --mock`;bench_rtt;真机 RTC 一段+stop-and-go 一段(用户验收新功能) | shim 反转 |
| **4** data 域 | Collector 接口+TeleopCollector 收编+`--arm` 参数化+dagger/ 桩 | 1-2 天 | 153 测试+mock 采集+真机一批次;旧命令别名到新 CLI | 同上 |
| **5** RL 域 | **5a** 纯搬移+shim(1 天);**5b** 接口化拆 train_stage2+WindowReceipt/BoundaryCollector+golden 对拍+修 infer_stage2 bug+解耦(2-3 天);**5c** 算法注册+serve_rlt embedding 声明化+recap 占位(1 天) | 4-6 天 | 5a:86 测试+`--dry_run`+断点跑 1 episode 旧命令等价;5b:对拍测试锁死 rtc transition 逐字段一致+`--dry_run/--connect_only/--dummy_robot` 全通 | **切换日=campaign v3 收官后 v4 开局前**;5a/5b/5c 各自独立 PR,campaign 机可只合 5a |
| **6** 收尾 | train/ 流水线、fetch_models.sh(lingbot 可选)、openpi 修改版入库收整、删全部 shim 与 RL-Token-Pi05-open、README/docs 定稿、tag v1.0.0 | 1-2 天 | **全新 clone→envs.sh(openpi 已在树内,lingbot 按需 fetch)→三条 mock 链路 + finearm-train make_config 干跑(clone 即用验收)**;真机全链路 smoke | — |

## 十、风险清单

| 风险 | 缓解 |
|---|---|
| campaign 中断(v3_2_2 续点依赖现入口+journal 格式) | 4060 钉 tag 不合并;5b 前 golden 对拍;checkpoints/journal 目录不动;RL 最后切 |
| uv 单锁解析冲突(lerobot==0.5.1 × torch × openpi-client) | B 计划拆独立 project;Phase 0 CI 即验 lock |
| LeRobot 插件按"发行名=模块名"发现 | 插件保持顶层模块(thin shim→finearm),发行名不变 |
| daemon schema 变更破坏旧客户端 | IPC 加 schema_version+profile 握手,缺省按 panthera 兼容,一个发布周期后才硬校验 |
| journal/ckpt schema 演进破坏 resume | journal 版本字段只读兼容;RLTConfig 保留兼容视图(stage2_v3_2_2 resume 必须能读);对拍进 CI |
| rtc 语义被"顺手改对"(reference 陈旧度)造成学习信号漂移 | WindowReceipt 显式 age;rltoken 默认 trigger-time 语义;对拍锁死 |
| 4090 serve 环境不受控 | finearm `--no-deps` 纯 py 安装;serve_rlt 不 import 重依赖 |
| 与"部署三件套"(明日计划③,infer_stage2 混合执行)冲突 | 三件套在 campaign 分支上做(旧代码);框架化 Phase 5b 吸收其成果进新结构 |

## 关键文件(实施主战场)

- `RL/rlt_panthera/scripts/train_stage2.py`(2067 行,RL 拆分主战场)
- `RL/rlt_panthera/rlt/replay.py`(981 行,对拍基准)
- `arm/panthera/panthera_control/{constants,ipc,adapters}.py`(ArmProfile 单点化)
- `data/collect/panthera-lerobot-collect/panthera_lerobot_control/collect_cli.py`(1892 行,Collector 收编)
- `infer/rtc_client.py`(执行器母本)+ `infer/adapters/__init__.py`(注册表范本,保持不动)
- `RL/rlt_panthera/scripts/infer_stage2.py`(:53-54 bug,5b 修)

## 验证方式(总)

1. 每阶段:对应域 pytest 全绿 + CI 绿 + 旧入口原 venv 冒烟(`--dry_run`/`--help`/`--mock`)
2. 行为等价三对拍:RTCExecutor vs 旧 rtc_client mock 输出;Phase 5a 断点 1 episode 旧命令经 shim;Phase 5b golden journal 对拍
3. 终态验收:全新 clone → fetch_models.sh → envs.sh → finearm-daemon/infer/collect 三条 mock 链路 + 真机全链路 smoke
4. 扩展性验收(框架目标):mock 第二臂(实现 ArmProfile+五方法)只加一个目录即可被 daemon/collect/infer 识别——作为 arms 域的测试用例固化
