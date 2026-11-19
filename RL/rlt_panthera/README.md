# rlt_panthera

RL Token 在线强化学习框架（Panthera-HT + openpi π0.5 JAX 底座）。总计划见
`/root/xsx/RL/PLAN.md`。

本仓库从 [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open)
（lerobot PyTorch + SO-101）移植纯算法层，**零 openpi/lerobot 依赖**（仅
torch + numpy）。关键参数化差异：

| | 参考仓库 | 本仓库 |
|---|---|---|
| 动作空间 | 6 维（SO-101 5 关节+夹爪） | **7 维**（Panthera 6 关节+夹爪，绝对关节目标） |
| VLA horizon | 50（lerobot π0.5） | **30**（openpi p3 checkpoint） |
| RL chunk | 10（checkpoint 契约锁死） | 10（不变，`paper_full_output_v1`） |
| 关节名 | `SO101_JOINT_NAMES` | `PANTHERA_JOINT_NAMES`（`target_joint_1..6` + `target_gripper`） |
| quantile 校验 | 恰好 6 维 | action 恰好 **7 维**；state 接受 LeRobot 14 维并切前 7（位置），与 openpi `panthera_policy` 的 `observation/state[..., :7]` 同切片 |

## 布局

```
rlt/
├── networks.py   # RLTConfig + RLTokenEncoder/Decoder, RLTChunkActor(2125→70), RLTTwinCritic
├── critical.py   # CriticalPhaseClassifier/Detector（Stage 1.5，照搬）
├── replay.py     # Transition/ReplayManager(journal+fsync+schema)/stride-2/动作缓存/关节序
├── learner.py    # perform_training_update(TD3+BC), polyak, discounted_chunk_return, ckpt schema v3
└── norm.py       # q01/q99 归一化（panthera_bolt_p3/meta/stats.json，绝对空间）
scripts/          # precompute_prefix / train_stage1 / report_zrl_sanity / openpi_prefix + serve_rlt / verify_serve_consistency / bench_rtt（Phase 3）
tests/            # 移植单测 + 7 维形状测试（Phase 0 验证：pytest 全绿）
data/             # prefix_v1（预计算 embeddings）/ stage1_v1（stage1 checkpoints）——A100 机产物，勿入 git
```

## 测试

```bash
/root/xsx/openpi/.venv/bin/python -m pytest tests/ -q
```

（任何带 torch≥2.5 / numpy 的 Python ≥3.11 均可；无需 GPU。）

## 三机部署总览（单仓库，三端共用本仓库）

| 机器 | 硬件 / 网络 | 角色 | 用到的组件 |
|---|---|---|---|
| **A100 训练机** | 双 A100，公网高延迟，**临时可用** | 仅离线：Phase 1 预计算 + Phase 2 stage1 训练 | `scripts/precompute_prefix.py`、`scripts/train_stage1.py`、`scripts/report_zrl_sanity.py`；数据 `data/prefix_v1`、`data/stage1_v1` |
| **4090 服务器** | 与执行机同局域网 | 在线推理：`serve_rlt`（冻结 VLA + stage1 encoder → `actions(30,7)` + `z_rl(2048)`） | `scripts/serve_rlt.py`（Phase 3）、openpi 运行环境、p3/2000 checkpoint、stage1 `best_checkpoint.pt` |
| **执行机** | 4060 Ubuntu，接 Panthera-HT | stage2 在线 RL：主循环 + 异步 learner + RobotInterface | `scripts/train_stage2.py`（Phase 4）、`rlt/` 纯 torch 包、openpi-client（纯 python） |

- A100 机**不承担任何在线角色**；所有产物（embeddings / checkpoint）由 A100 机拷贝/rsync 到 4090 与执行机后离线运行。

### 4090 服务器（docker 容器）：部署与启动 serve_rlt（Phase 3）

实际部署形态：4090 侧跑在 **docker 容器**内（路径如 `/data/xsx/RL/rlt_panthera`），8000 端口
无法对局域网开放 → 执行机通过 **SSH 隧道**访问（见执行机小节）。

一次性准备（从 A100 机同步，约 5GB）：

```bash
# A100 机上执行（仓库代码 + stage1 checkpoint；prefix_v1 的 57GB mmap 不需要去 4090）
rsync -av --exclude 'data/prefix_v1' --exclude 'data/stage1_v1/checkpoint_step*' \
  /root/xsx/RL/rlt_panthera/  root@<4090-ip>:/data/xsx/RL/rlt_panthera/
# 底座 checkpoint（若 4090 上还没有）
rsync -av /root/xsx/openpi/checkpoints/pi05_panthera_p3/  root@<4090-ip>:/data/xsx/openpi/checkpoints/
# 975 字节的 meta.json 单独补上（恢复同源四件套校验；缺它只是跳过校验，不影响运行）
ssh root@<4090-ip> "mkdir -p /data/xsx/RL/rlt_panthera/data/prefix_v1"
rsync -av /root/xsx/RL/rlt_panthera/data/prefix_v1/meta.json \
  root@<4090-ip>:/data/xsx/RL/rlt_panthera/data/prefix_v1/
```

4090 容器内需要 openpi 运行环境（venv 含 jax/torch/openpi/openpi-client）。启动：

```bash
openpi/.venv/bin/python RL/rlt_panthera/scripts/serve_rlt.py \
    --checkpoint-dir /data/xsx/openpi/checkpoints/pi05_panthera_p3/panthera_p3/2000 \
    --stage1-checkpoint /data/xsx/RL/rlt_panthera/data/stage1_v1/best_checkpoint.pt \
    --port 8000
# 启动序列：加载 π0.5 bf16（~1-2min）→ warmup JIT+契约自检（~35s）→ 监听 0.0.0.0:8000。
# 同源四件套不匹配会拒绝启动（--allow-origin-mismatch 可强行跳过，不建议）。
```

请求/响应契约（openpi-client 兼容）：
`observation/state(≥7)` + `observation/image(H,W,3 uint8)` + `observation/wrist_image` + 可选 `prompt`
→ `{"actions": (30,7) 绝对关节目标, "z_rl": (2048,) float32, "policy_timing": {infer_ms, z_rl_ms}}`。

实测参考（2026-11-13，SSH 隧道 300 请求）：服务端 infer p50 60.7 / z_rl p50 67.7（合计 ~128ms，
与 A100 几乎一致）；RTT p50 312.7 / p99 587.5 / max 2077.7（隧道+打包 ~184ms）。

### 执行机（4060）：Stage 2 在线 RL（Phase 4）

```bash
# 0) 一次性：同步仓库（A100 机执行）+ 装依赖（openpi-client + torch-cpu + numpy）
rsync -av --exclude 'data/prefix_v1' --exclude 'data/stage1_v1/checkpoint_step*' \
  /root/xsx/RL/rlt_panthera/  <user>@<4060>:~/rlt_panthera/
pip install openpi-client   # torch CPU 版随 rlt 包需要（learner 2×256 MLP 跑 CPU 足够）

# 1) 开隧道（保持窗口常开；4090 容器内 serve_rlt 已启动）
ssh -c aes128-gcm@openssh.com -N -L 8000:127.0.0.1:8000 root@192.168.110.46

# 2) 接真机前检查：robot/panthera_robot.py 适配层已就绪（包装采集机
#    deploy/infer/robot_interface.py，默认自动在 ~/internship/baiqin/deploy/infer/
#    找到它；可用 PANTHERA_ACQ_IFACE 指定路径）。要点：
#    · 观测走 raw 640×480 帧（服务端 resize 与训练同分布；勿用 224 预缩放）
#    · v1 未接 HIL 人工流（daemon 无遥操作接口），'h'/'s' 键暂为 no-op
#    · PANTHERA_MOCK_IMAGES=1 可不接相机联调（arm daemon 仍需在跑）
/home/xsx/internship/baiqin/deploy/infer/.venv/bin/python scripts/train_stage2.py --config configs/stage2_panthera.json --dry_run

# 3) 上机顺序（安全协议，逐步推进）：
/home/xsx/internship/baiqin/deploy/infer/.venv/bin/python scripts/train_stage2.py --config configs/stage2_panthera.json --vla_only --max_episodes 10
/home/xsx/internship/baiqin/deploy/infer/.venv/bin/python scripts/train_stage2.py --config configs/stage2_panthera.json            # warmup 15 集 → 'c'+'a' 小规模 actor
```

**rtc_mode（连续运动，推荐先对比再选）**：`--rtc_mode` 启用 deploy rtc_client 同款流水线执行——
剩 k 步（k 按 RTT p95 自适应）时用当前观测异步重规划、窗边界对齐+混合切换，机械臂连续运动。
reference/z_rl 元组为触发时刻（≈RTT 前移）；stride-2/terminal replay 边界仍按真实边界观测物化。
默认关（stop-and-go，论文 Algorithm 1 原版语义）。用 vla_only 各跑 20 集对比成功率后决定训练节奏：

```bash
/home/xsx/internship/baiqin/deploy/infer/.venv/bin/python scripts/train_stage2.py --config configs/stage2_panthera.json --rtc_mode --vla_only --max_episodes 20
```

键位：`e` 急停 · `q` 退出 · `c` 关键段开关 · `a` actor 开关（需先 `c`）· `g` 跳集 ·
`f` 提前结束 · `h` HIL 切换（daemon→teleop 主臂接管；`PANTHERA_HIL_MODE=freedrive` 为拖动示教）·
`s` HIL 交还（daemon→policy，无缝）· `p` 集前手动摆位 · `y/n/d/q` 终端 reward。
时序：每窗（10 步@30fps=0.33s）一次 serve 调用，stop-and-go 节奏（动作 0.33s + 一个 RTT 等待）；
replay materialization 与 learner 全部后台线程，不占控制路径。
产出：`<output_dir>/replay_journal.pkl`、`checkpoint_ep<N>.pt`（schema v3）、`rl_metrics.log`（`tail -f`）、`train.log`、`critical_samples.pkl`（Phase 5 分类器训练数据）。

### Phase 5：critical 分类器训练 + 部署推理（任意机器，CPU 可）

```bash
# 1) 用 Stage-2 攒的人工 'c' 边界训练分类器（跑若干集后）：
python3 scripts/train_critical_classifier.py \
    --samples <stage2_output>/critical_samples.pkl \
    --output checkpoints/critical_classifier.pt
# 看 report 里的 held-out 分离度（目标：区间内≥0.92 / 区间外<0.1，PLAN §八 P5）

# 2) 训练曲线绘图：
python3 scripts/plot_metrics.py --log <stage2_output>/rl_metrics.log

# 3) 部署推理（无学习；actor mean + 噪声 0.02，保守 auto-critical 阈值）：
python3 scripts/infer_stage2.py --config configs/stage2_panthera.json \
    --stage2_checkpoint <stage2_output>/final_checkpoint.pt \
    [--auto_critical --critical_classifier checkpoints/critical_classifier.pt] \
    [--score] [--record rollouts]
```

### 执行机（4060）：SSH 隧道 + RTT/联调

```bash
# RTT 基准 / 联调（openpi-client 纯 python，无需 torch/jax）
python3 ~/rlt_panthera/scripts/bench_rtt.py --host 127.0.0.1 --port 8000 --requests 300
```

延迟判定口径（PLAN.md Phase 3）：warmup/vla_only 预算 1000ms；critical RL 窗口 stop-and-go
（0.33s 动作 + RTT 等待 ~0.3-0.6s）——参考项目默认模式同为此节奏（其进程内扩散 1-5s）。

## 状态

- **Phase 0（纯 torch 组件移植）：完成**（2026-11-12，75 测试全绿）
- **Phase 1（JAX prefix 预计算）：完成**（2026-11-12，`data/prefix_v1`：29036 帧 × 512 token × 2048 维 fp16，56.7GiB，与 compute_loss 数值等价已验证）
- **Phase 2（Stage 1 训练）：完成**（2026-11-12/13，batch 128 × 30k 步 ≈ 12h；loss 1.51→0.063；z_rl 有效秩 1230.9/2048、跨集方差比 0.104，未塌缩；`data/stage1_v1/best_checkpoint.pt` = step 30000）
- **Phase 3（serve_rlt + 双机联调）：完成**（2026-11-13；A100 16 帧一致性门禁全过 z_rl cos≥0.9962/幂等 0/敏感性 0.90；4090 docker 部署 + SSH 隧道实测 RTT p99 587.5ms；报告 `data/prefix_v1/serve_consistency_report.json`）
- **Phase 4（Stage 2 在线 RL）：代码完成 + 真机适配 + HIL 接入 + 冒烟通过**（2026-11-13；88 测试全绿；`--connect_only` 安全联检模式；HIL=daemon teleop/freedrive 模式切换，`'h'`/`'s'` 键）。**待做**：4060 部署 → vla_only 基线 → 上机训练
- **Phase 5（critical 分类器 + 部署推理）：代码完成**（2026-11-13；train_critical_classifier / infer_stage2 / plot_metrics）

## 实验日志（2026-11-16，Stage 2 真机训练首日）

### 跑过的 run（checkpoints/ 下，全部经过 journal 不变量校验：奖励挂载/mask/转移链/checkpoint↔journal 字节级一致）

| run | 内容 | 结果 |
|---|---|---|
| `stage2_panthera_v1` | 2 集冒烟 | 数据结构全对；发现 j6 编码器通道恒定、act==ref（无回读）两个观察项 |
| `stage2_panthera_v1_1` | 11 集（9 有效） | 8/10=80% 基线；stride 网格奇偶/补尾窗/chain 双采样等"怪象"全部溯源为设计路径 |
| `stage2_panthera_v1_2` | 20 集，ep17 首次 `a` | 发现 `a` 全局粘滞 bug；actor 晃动严重（当时默认 Q=1/BC=10，bc 平台 0.003，critic 过拟合 td_err≈1e-4） |
| `stage2_panthera_v1_3` | resume 至 ep60 | ep35-40 好转（70%，限幅 Δ 从 0.214 降到 0.04），**ep54-60 发散**：actorQ−q1_mean→+0.32、bc→0.0043、q1_max 顶 2.0 钳、Δ11rad 脏指令、七连败 → 回滚 ep40 |
| `stage2_v2_1` | 全新纯 `c` 采集 19 集 | 68% 基线；`c` 时机一致性改善（ON 360-440±20 步）；新权重 0.3/20 下 actorQ−q1_mean≈−0.003、q1_max 0.96，学习器健康 |
| `stage2_v2_2` | resume 起 `a`，35 集止 | 学习器仍健康（无剥削），但 **actor 接管 11 集中 7 集几秒内碰掉螺栓**；VLA-only 2/6 属基线波动。照片证据：ep40 结束帧组件被放倒在桌面上（用户随后复位） |

### 诊断结论

1. **发散机理（v1_3）**：小 buffer + critic 过拟合 → actor 剥削幻觉 Q 峰（actorQ−q1_mean 拉大、bc 上升）。红线：`actorQ−q1_mean>0.1`、`bc>0.004↑`、clamp Δ>0.5、q1_max 钉 2.0。
2. **当前瓶颈（v2_2）不是学习器，是 actor 模仿精度**：bc 平台 ≈0.003（每维 RMSE≈0.055 归一化）大于插入容差；z_rl 帧间噪声（相邻 cos 0.84-0.93）放大 chunk 间不一致。VLA 用完整视觉骨干重规划不怕，MLP actor 只见噪声摘要。
3. 已知观察项：j6（index 5）编码器恒定（两台位/全部 run，命令有波动但不响应——待查机械/编码器，需对照 p3 数据集 j6 统计）；`action_chunk==ref_chunk` 逐位相等（daemon 无回读，v1 已知限制，夹紧不可见）；`success` 字段恒 0（未使用）。

### 今日代码/配置改动

- `scripts/train_stage2.py`：**每集开始自动 `ACTOR_ENABLED=False`**（打印 `[KEY] Actor auto-off at episode start`）——修复 `a` 跨集粘滞；
- `deploy/infer/robot_interface.py`：`_go_home` 插值 `move_s` **6.0s→2.5s**（`__init__` 的 `home_move_s` 与 `_go_home` 默认值两处）；归位从最坏 14s 降到 ~10.5s；
- 训练配方：`--online_q_weight 0.3 --online_bc_weight 20`（防剥削）、`--actor_max_relative_target` 按关节（臂 0.05、**夹爪 0.25**——标量 0.05 会掐慢夹爪）、`--save_every 10`。

### 操作规约（已与操作者对齐）

- `c`=关键段开关（ON/OFF toggle，OFF 锚定 terminal+奖励）；`a`=actor 执行许可（每集需重按，critical 内才有效）；事故处置 `a→失败→c→f→n→复位台面` 是正确流程；
- `c` 的界标固定在对准段开始（ON 360-440 步区间内保持一致）；结果确定即 `c`-off；卡死早收（c→f→n）；
- VLA 集：只按 `c`；actor 集：`c` 后立刻 `a` 让它跑完整段（不要中途抢收——5s 脉冲实为事故处置，正常放量时不这么用）；
- 台面复位是手动的：**建议胶带标记螺母/螺栓标准摆放位**（复位可重复性影响 VLA 成功率、`c` 一致性、后续分类器）。

### 下一步（明日继续）

1. 以 VLA 采集为主 ~20-30 集（每 5 集 1 个 actor 试水集），buffer 推向 5k+，盯 `bc` 是否破 0.0025——破=数据量问题（继续）；死钉 0.003=MLP 容量/z_rl 噪声上限；
2. actor 试水集配保险：关节限幅收 0.03、可试 `--actor_execution_steps 5`；放量标准：clamp 触发率<40%、Δmax≤0.1、能走完整段不碰掉螺栓；
3. 攒够后 Phase 5：`train_critical_classifier.py`（目标区间内 P≥0.92/区间外<0.1）→ `infer_stage2 --score` 与 VLA 基线对比；
4. 待查：j6 通道问题对照 p3 数据集统计。

## 实验日志（2026-11-17，Stage 2 第二日：v2 收尾 → v3 重启 → 根因修复）

### 跑过的 run（全部经过 journal 不变量校验）

| run | 配置要点 | 结果 |
|---|---|---|
| `stage2_v2_3` | resume v2_1/ep20（v2_2 数据弃用），限幅标量 0.05 | ep21-40 共 20 集 55%；actor 试水 4 次 Δ 0.172→0.056、clamp% 13→6 达标，但全部 5-10s 手动切回未走完整段 |
| `stage2_v2_4` | **残差滤波版**（β=0.5 混 VLA ref + EMA 0.3，当日实现） | 滤波完全兑现（每步\|Δ\| 0.042→0.0111、翻转 61%→41%、clamp 0-3 步）；但手感≈纯 VLA → 判定为参考项目已废弃的 residual 模式，**撤** |
| `stage2_v2_5` | 无滤波对照（操作者自跑），限幅 0.03 削峰 | 8/10=80%；确认无滤波也可赢集 |
| `stage2_v2_6` | **execute_mean 首次**（参考项目消抖正道）+ noise 0.02 + 逐关节 JSON 限幅 | ~55%；q1_max 0.97→1.85 爬升引出后续根因 |
| `stage2_v3_1` | **全新起点**（stage2 权重+buffer 重置）+ 论文权重 10/1 | 18 集 **78%**（最好档）；q1_max 0.97 无通胀；ep20 = 首个完整段成功 |
| `stage2_v3_2` 第一场 | 中途换螺栓/螺母位置 | VLA 泛化良好（p3 示教位置随机），**失败全在按 a 后**：actor/critic 小 MLP 对新位置关键段 OOD（0/7）→ 场次废弃，教训保留 |
| `stage2_v3_2` 第二场 | 从 ep20 重启；**限幅 0.02 + noise 0.01**；位置 A(ep21-30)/B(ep31-40) 轮换 | 20 集 65%；**换位无崩塌**（B 位 6/10）——0.02 限幅兼作 OOD 保护；actor 收敛拐点（每步\|Δ\| 0.048→0.024）；但 q1_max 爬到 1.82 |
| `stage2_v3_3` | 论文权重 10/1 继续 | **红线触发**：actorQ−q1 +0.118、bc 抬 0.0056、q1_max 1.89、成功率 60→50% |
| `stage2_v3_4` | 回 0.3/20（从 ep50 续） | **未止血**：actorQ−q1 涨到 +0.196、2/10 → 证明 actor 侧权重压不住 critic 自身通胀 |
| `stage2_v3_2_1` | **回退 v3_2/ep40 + 0.3/20 + `--td_target_clamp 1.0`** | **恢复成功**：q1_max 一千 updates 内 1.82→0.98；成功率回 60-65% |
| `stage2_v3_2_2` 旧 | 同配方 | 4/10；**skip 保存 bug 受害**：ep60 被 `g` 跳过且为唯一 save 边界 → 该场零 checkpoint，数据作废 |
| `stage2_v3_2_2` 新 | 从 v3_2_1/ep50 续 | **7/10=70%，重启后最佳**；checkpoint_ep60 正常保存；actor 窗破 3k |

### 诊断结论（当日三大根因，均已验证）

1. **bc"钉死 0.003"归因更正**：bc 误差算在采样后动作上，σ²=policy_fixed_std²=0.0025 是不可约地板；真实均值模仿误差 = bc−0.0025（v2_3→v3_2 从 0.001 降到 0.0004，**数据一直在起作用**）。此前"MLP 容量上限"结论作废。
2. **actor 晃动主因 = 执行时采样噪声（std 0.05）**，不是策略本身。参考项目正解 `--actor_execute_mean`（均值执行，"execution smooth like the VLA"）+ `--actor_noise_std` 小探索；残差混合（executed = ref + β·修正）是参考项目**硬废弃**的模式（`actor_residual_scale` 强制 0，paper-aligned 要求全量输出）。
3. **跨链 Q 通胀公共根因 = `td_target_clamp=2.0`**（rlt/learner.py:104，原未暴露恒用默认）：本任务奖励 0/1 terminal 无负奖励，真值域 Q∈[0,1]，旧钳位允许估值到真值两倍——v1_3 钉 2.0、v2_6 1.85、v3_3/4 1.89 全部爬向钳位。**修复：钳 1.0**（`--td_target_clamp 1.0` 透传），q1_max 千次 updates 内归位 0.98。附：论文权重 10/1 在本任务两度复发（v1_3、v3_3），**0.3/20 为实证必要配方**；钳位 1.0 后 actorQ−q1 指标语义改变（结构性正偏），健康信号改为**成功率趋势 + bc + Δ 野值**。
4. **泛化**：VLA/stage1 已带位置多样性（p3 示教随机摆位），过拟合的只是 actor/critic 小头 → **位置多样性进 RL 采集即可**（每 ~10 集换位、`c` 按真实对准点、0.02 限幅护 OOD），无需回示教阶段。

### 代码改动（train_stage2.py）

- 新增 `--actor_blend_beta/--actor_ema_alpha`（执行侧残差滤波；**已被否决**，默认关，仅留代码）；
- **撤每集 actor auto-off**（恢复参考项目跨集保持：`a` 按一次，每个关键段接管直到再按）；
- 撤退出时的 final_checkpoint.pt 保存（resume 只用 `checkpoint_ep{N}.pt`）；
- 新增 `--td_target_clamp` 透传（默认 2.0 兼容旧行为，本任务跑 1.0）；
- **修复 skip 保存 bug**：`g` 跳过的集落在 save_every 边界时也保存 checkpoint（v3_2_2 旧场因此零 checkpoint）。

### 操作规约更新

- **`c` 只在"真成/真废"时按**——中途用 `c` 交接会给失败状态错发奖励 1；段内换执行者用 `a`（或 `h` 人接管，人的修正直接替换 BC 参照）。救援成功流程：`a`-off → VLA 段内完成 → `c` → `f` → `y`；
- 位置轮换：每 ~10 集一个胶带固定位，集内不换；`c` 界标按语义（真实对准开始）不按固定步数；
- y 判定口径现阶段宽松（插进去就算），actor 平滑后再收紧到"稳慢准"。

### 部署设计（当日定稿，待实现）

**混合执行模式**：VLA 接近段用生产 RTCRunner 流水线（`deploy/infer/rtc_client.py` 模式，单在途请求 + 提前 k 步触发，连续运动；train_stage2 内已有移植 `PipelinedChunkProvider`）；按 `c`+`a` 切换后 actor 段用**同步 stop-and-go**（新鲜 ref + 均值 + 逐关节限幅）；出关键段切回流水线。

正确性论证：actor 的三个输入（z_rl/proprio/ref_chunk）全部来自切换后同步回路的新鲜推理 = **与训练分布完全对称**（训练全程 rtc_mode=false）；VLA 段节奏只影响到达时状态（二阶差异，流畅到达更稳）；训练测得的成功率本来就含 stop-and-go 节奏，数字可迁移；切换的一次 RTT 停顿落在对准起点，对本任务无害甚至有利。

待实现（infer_stage2.py 三件套）：① VLA 段接 `PipelinedChunkProvider`；② actor 段同步执行 + `--actor_max_relative_target` 限幅（当前推理脚本无限幅）；③ 进出关键段的切换逻辑（进：丢弃在途 chunk；出：恢复流水线）。之后做对照实验：同 checkpoint，VLA-only（不按 a）vs RL（按 a），`--score --actor_noise_std 0`，每位置 ≥10 集，`--record` 存档。

### 下一步（2026-11-18 继续）

1. 续跑 campaign：从 `stage2_v3_2_2/checkpoint_ep60.pt` 起（建议 `--save_every 5`），标准配方 = execute_mean + noise 0.01 + 逐关节限幅 0.02/夹爪 0.25 + `--online_q_weight 0.3 --online_bc_weight 20 --td_target_clamp 1.0`；位置 A/B/C 轮换，目标成功率稳定 70%+、actor 窗推向 5k；
2. 分类器：用历史 `critical_samples.pkl` 训练（先确认 `train_critical_classifier.py` 输入接口），目标区间内 ≥0.92 / 区间外 ≤0.1；达标后训练/部署挂 `--auto_critical`；
3. 部署：完成 infer_stage2 三件套移植 → VLA/RL 对照实验；
4. 盯防项：Δ 野值（如 v3_2_2 旧场 ep58 的 1.745 rad 单步请求，被 clamp 拦住；复现则深挖）。

## 实验日志（2026-11-18，Stage 2 第三日：campaign 续跑 + 换位 OOD + 接管跳变分析 + 外部调研）

### 跑过的 run

| run | 经过 | 结果 |
|---|---|---|
| `stage2_v3_2_3` | 从 v3_2_2/ep60 续（标准配方）；**ep67 前操作者对调了螺栓/螺母位置** | ep61-66 = 4/6（last20 0.70 持平）；ep67-70 **连败 4 集**（last20 0.70→0.55），ep67 关键段挣扎 370 步（正常 3 倍） |
| `stage2_v3_2_4` | 从 v3_2_3/ep70 续，新位置继续采 | ep71-80：新位置累计 5/14，最近 10 集 5/10，last20 底 0.40→0.45 回升中；**actor 窗 ~5.4k 破 5k**；buffer 7100；ep75/80 存档 |

### 换位下滑诊断（结论：预期内 OOD，非策略退化）

离线指标全程干净（q1_max 0.96-0.97 未钉 1.0、bc 0.0029-0.0030、Δ≤0.151、actorQ−q1 正常区间）+ z_rl 范数钉 46.6-46.7（基座对位置泛化良好）→ 与 v3_2 第一场同机制：**只有 actor/critic 小头对新几何 OOD**，解法就是继续喂新位置数据。外部复现证据（见下）也确认换位下跌是该方法已知行为（他们换机器人起始位 90%→33%，物体仍在分布内）。操作侧照旧：失败临界段救援（`a`-off→VLA 完成→`c`→`f`→`y`，或 `h`），h/m 窗至今为 0。

### 接管跳变实测（journal 扫描，299 个 VLA→RL 边界）

- **接管第一步 = VLA 正常步长的 3-17×**（j5 17×、j6 10×、j1 9×、gripper 7×；j6 含已知命令摆动水分）；更根本的是 **actor 稳态步长本就是 VLA 的 4-5×**（0.017-0.043 vs 0.003-0.010 归一化），边界相对 actor 稳态仅 1.1-2.4×——"跳变感"= 速度不连续 + Q 修正第一步兑现 + gripper 限幅 0.25 过松（归一化 max 0.42）。
- **论文不设边界平滑**（paper-aligned 即 delta_weight=0）：论文的解法是结构性的——actor 均值被 BC 强锚在 VLA reference 上（execute_mean 时 "smooth like the VLA"），Q 修正越大接管跳变越大，属方法的内在性质。参考项目工具箱：`--delta_weight 0.02`（学习侧平滑，约束 w×delta ≤ 1/3×Q 项；复现者也独立加了同型的 jerk 正则）、`--actor_execution_scale`、`--actor_execution_steps`；无边界插值（唯一 smooth align 是人手接管 leader 臂对齐）。blend/EMA = 已废弃残差模式，不用。
- **修复方案（下次 resume 上，均不在 provenance）**：`--delta_weight 0.02` + gripper 限幅 0.25→0.05；若 j1/j5 边界仍明显臂 0.02→0.01。
- 扫描脚本 `/tmp/scan_boundary.py`（/tmp 会清）：journal pickle 流重建每集执行动作序列（按 episode_id/step_id 排序、executed_steps 前缀、valid_mask），在 source_chunk 0→1 处对比相邻执行动作，与 VLA/RL 稳态步长对照；注意窗口接缝会使个别 max 虚高，看均值/p95。

### 机制 FAQ（当日答疑，记档备查）

- **失败集（按 n，r=0）数据完全有效且必要**：稀疏奖励下成功/失败集唯一区别是 terminal 一步，TD 沿链回传（γ=0.99）压低整条失败路径的 Q——没有失败数据 Q 会通胀（数据偏科版）。BC 目标是 ref_chunk，与奖励无关，失败窗口照样做锚定。唯一纪律：n 的判定必须真实。
- **actor chunk = 10 步**（stage-1 契约锁死）；重规划每 10 步一次。名义 0.33s/块（30fps），**实测 ~0.6s/块（~17Hz）**，VLA/ACTOR 同节奏 → 瓶颈在执行机/推理链路。每次 replan 第一步都是一个接缝，VLA→RL 接管是最大的一次。
- Δ 列含义确认：`(clamped − requested)` = 被削掉的请求超额量，不是步长。

### 外部调研（官方/社区/博客）

参考仓库 [Zhengsw03/RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open) issues 为 0，无社区案例。最有价值的是复现笔记 [Research Notes From Reproducing RL Token](https://villekuosmanen.medium.com/research-notes-from-reproducing-rl-token-f375ecfd3c28)（2026-04，复现 [arXiv:2604.23073](https://arxiv.org/abs/2604.23073)）：

1. **"稳定提升后突然连败、暂时回落到 60%"**——与我们 70→55 同型，列为未解决开放问题（怀疑 lr 偏高）；**周期性回落是方法已知行为**。
2. **方差是第一杀手**：固定最难构型 1 小时 0→90%；随机摆位 plateau ~40%；"RLT 是精调动作的，不是发明新动作的"，MLP 不擅多模态。换机器人起始位 90%→33%。
3. **提速手段**：网络 256×2 学得慢，512×3 明显快（第 4 层更快但不稳）；UTD 5→10 明显提速；BC 权重核心 trade-off（重=慢、轻=快而乱）；gripper 单独降 BC 权重学得更快；自加 jerk 正则。
4. **reward hacking 是他们最大痛点**（Q 尖峰+危险动作），靠 4 critic + 梯度裁剪减半缓解；我们的钳位体系（td_target_clamp 1.0 + per-step 限幅）未出现此问题。
5. **基座质量设上限**："RL expert 救不了差的冻结 VLA，不如把时间花在干预数据上"。

### 候选行动（按代价排序，均待执行）

1. **基线对照实验先行**：部署三件套移植后同 checkpoint 跑 VLA-only vs RL——"效果好不好"必须对基线度量（复现者是 0% 基线 → 90%；我们 VLA-only 未知）；
2. **UTD 5→10**：不在 provenance，resume 可改，零风险提速；
3. **权重朝论文方向重试**（0.3/20 是钳位 bug 年代的保守值，根因已修；复现者核心经验"BC 太重学得慢"）：先 0.5/15 或 1/10 试水，红线照旧盯防；
4. **网络 256×2→512×3**：`--restore_replay` 已确认移植（train_stage2.py:656），可在现有 journal 上换新头重训免重采；前三条不够再上；
5. **delta_weight 0.02 + gripper 限幅 0.05**（接管跳变修复，见上节）；
6. h/m 救援采集持续。
