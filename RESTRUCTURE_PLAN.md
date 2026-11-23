# RESTRUCTURE_PLAN · 目录重构迁移方案

> 状态:待执行(v2 · 已并入执行前评审修订)· 创建:2026-11-23 · 修订:2026-11-23
> 目标:让每条目录轴只剩一个含义 —— **代码在 `src` + `plugins`,环境在 `envs`,机器资产在 `arm`,运行产物在 `data`**。
> 原则:纯机械搬家 + 路径引用更新,不改任何运行时行为;每个 Phase 结束 CI 全绿、可独立 revert。

## 0 · 现状问题

| 问题 | 现状 |
|---|---|
| 命名撞车 | `data/` vs `src/finearm/data/`、`infer/` vs `src/finearm/infer/` 同名不同物 |
| `data/` 里放代码 | `data/collect/panthera-lerobot-collect/` 是带 pyproject/tests/venv 的插件包,却藏在数据目录下 |
| 环境定义散落三处、约定不一 | `infer/{init_env.sh,requirements.txt}`、插件内 `requirements-lerobot.txt`、`arm/panthera/init_env.sh` |
| 遗留 `deploy/` 前缀注释 | 多个脚本/文档注释仍写 `deploy/data/...`(上一轮重组的残迹) |
| 活跃 bug(已验证) | `core/paths.py` 的 `acquisition_interface` 默认值 `infer/robot_interface.py` 指向不存在的文件 → 无覆盖时 `PantheraStage2Robot()` 构造**直接 `FileNotFoundError`**(`rl/envs/panthera/robot.py:88`),并非静默回退;真机现在能跑只是因为设了 `PANTHERA_ACQ_IFACE`/local.jsonc。修法见 Phase 3.2(**删默认条目**,非改指向) |
| `.gitignore` 埋雷 | 第 6 行遗留 `/envs/`(pre-refactor 产物目录)会**吞掉整个新 `envs/`** —— `git check-ignore` 已验证三条 env 定义全部命中;不删则 Phase 2 的文件进不了 git、CI checkout 上没有 envs/(Phase 2.0 处理) |
| 位置推导脆弱 | wrapper 全靠 dirname 数层数找仓库根、插件测试靠 `ROOT.parents[N]`(每次搬家全碎);已被 deploy/ 改名烧过一次,本次必须同步重锚(Phase 1.6;根治见 §6.1) |

## 1 · 目标布局

```text
finearm/
├── src/finearm/                  # 全部代码(不动)
├── configs/                      # 配置(不动)
├── envs/                         # ★ 新:所有环境定义集中一处(脚本 + requirements + 文档 + venv)
│   ├── infer/                    #   ← infer/{init_env.sh, requirements.txt, README.md, README_zh.md}
│   │   └── .venv/                #   推理环境(openpi-client)
│   ├── collect/                  #   ← 插件中的 requirements-lerobot.txt 拆出 + 新 README
│   │   └── .venv-lerobot/        #   采集环境(LeRobot 0.5.1)
│   └── arm/                      #   ← arm/panthera/init_env.sh
│       └── .venv/                #   驱动环境(SDK 轮子)
├── plugins/                      # ★ 新:依赖宇宙独立于 src 的代码包
│   └── panthera-lerobot-collect/ #   ← data/collect/panthera-lerobot-collect 整目录 git mv
├── arm/panthera/                 # 保留:run_daemon.sh、robot_param/、sdk/(gitignored)
├── data/                         # ★ 变纯:只剩 datasets/、logs/(均 gitignored)+ 一个 README
├── models/  third_party/  docs/  tests/  utils/  RL/   # 不动
```

**venv 规则(新):** 所有 venv 一律放 `envs/<domain>/` 下,名字保留语义(`.venv` / `.venv-lerobot`)。
找环境只看 `envs/`,找 venv 不再需要知道使用它的代码在哪。

**命名备注(定案):** 顶层定名 `envs/` 而非 `venvs/`。曾顾虑与 `src/finearm/rl/envs/`
(RL 环境适配器包)语义撞轴,但两者不在同一层级(顶层轴 vs src 内子包),且该目录除
venv 外还有 requirements/init 脚本/README,`venvs/` 反而窄了。若团队后续仍介意,
整体改名是一次机械替换,不影响本计划结构。

**不动的部分(明确排除):** `src/finearm/**`、`models/openpi/**`(vendored,排除在 lint/清扫范围外)、
`finearm.core.paths` 的 repo-relative 解析机制、四层配置模型。

## 2 · 旧 → 新路径对照表

| 旧 | 新 |
|---|---|
| `infer/init_env.sh` / `requirements.txt` / `README*.md` | `envs/infer/` 同名文件 |
| `infer/.venv/` | `envs/infer/.venv/` |
| `data/collect/panthera-lerobot-collect/` | `plugins/panthera-lerobot-collect/` |
| `data/collect/panthera-lerobot-collect/requirements-lerobot.txt` | `envs/collect/requirements-lerobot.txt`(拆出,env 定义归 envs) |
| `data/collect/panthera-lerobot-collect/.venv-lerobot/` | `envs/collect/.venv-lerobot/` |
| `arm/panthera/init_env.sh` | `envs/arm/init_env.sh` |
| `arm/panthera/.venv/` | `envs/arm/.venv/` |
| `data/datasets/`、`data/logs/` | 不变(只是 `data/` 里不再有别的东西) |
| `arm/panthera/sdk/`、`robot_param/`、`run_daemon.sh` | 不变 |
| `data/collect/panthera-lerobot-collect/docs/quest2ros_setup.md` | `plugins/panthera-lerobot-collect/docs/quest2ros_setup.md`(随目录走) |
| `data/collect/.../run_panthera_lerobot_daemon.sh` | **删除**(转发 shim,与"不留 shim"原则矛盾;文档 ~17 处引用改指 `arm/panthera/run_daemon.sh`,见 Phase 1.4) |
| `.env.example` | 随插件目录走;注释里 `<deploy>/…` 段同步重写(Phase 1.8) |

## 3 · 分阶段执行

### Phase 1 · 插件搬家:`data/collect/` → `plugins/`

影响面最大的一个 Phase(脚本里的自定位路径推导全要改)。

1. **git mv**(保留历史):
   ```bash
   mkdir plugins
   git mv data/collect/panthera-lerobot-collect plugins/panthera-lerobot-collect
   rmdir data/collect
   ```
   ⚠️ `git mv` 会把目录内未跟踪的 `.venv-lerobot/` 一起在磁盘上搬走 —— venv 不可重定位
   (pip/lerobot-record 的 shebang 硬编码旧绝对路径),搬完即坏;采集机处理见本 Phase 末"真机窗口"。
2. **`plugins/panthera-lerobot-collect/run_panthera_lerobot_collect.sh`** — 自定位推导改一层:
   插件现在位于 `<repo>/plugins/<name>`,repo 根 = 上两级:
   ```bash
   REPO_ROOT="$(dirname "$(dirname "${ROOT_DIR}")")"        # 原 DATA_ROOT+DEPLOY_ROOT 两步推导,合并
   export PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${REPO_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"
   export PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${REPO_ROOT}/data/datasets}"
   ```
   **除两条 export 外,同脚本另有两处引用必须同步改**(漏一处即 `set -u` 下 live 采集预检直接崩):
   - **行 170**(相机预检):`cam_lock` 默认值 `$(dirname "${DATA_ROOT}")/utils/lock_cameras.sh`
     → `${REPO_ROOT}/utils/lock_cameras.sh`
   - **行 69**:`PYTHONPATH="${ROOT_DIR}/src"` 本就指向不存在的 `<plugin>/src`(死代码,现靠
     venv 内 editable install 才工作)→ 改 `${REPO_ROOT}/src`(裸 checkout 不 pip install 也能跑)
   venv 查找路径同步改(见 Phase 2 前本 Phase 先维持插件内 `.venv-lerobot`,Phase 2 再切到 `envs/collect/`)。
   顺带清除注释/报错文案里的 `<deploy>/...` 旧前缀(行 154/169/172)。
3. **`plugins/.../scripts/init_workspace.sh`** — 同样改推导;变量改名 `REPO_ROOT` → `PLUGIN_ROOT`
   (它原来指插件目录,现在 repo 根也出现了,两个"root"并存必须区分):
   ```bash
   PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
   REPO_ROOT="$(dirname "$(dirname "${PLUGIN_ROOT}")")"
   PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${REPO_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"
   PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${REPO_ROOT}/data/datasets}"
   ```
   `requirements-lerobot.txt` 暂留插件内(Phase 2 拆走),路径变量同步改;顶部注释(行 8-11)
   的"数据域"叙述改写为 plugins 定位。
4. **删除转发 shim `run_panthera_lerobot_daemon.sh`**:上一轮重组遗留、与本计划"不留 shim"
   原则矛盾,且其 `../../../arm/...` 相对推导搬家后恰好错一层(会指到仓库外)——与其修,不如删。
   `git rm plugins/panthera-lerobot-collect/run_panthera_lerobot_daemon.sh`;同步:
   - `docs/quest2ros_setup.md`(~10 处)与 `docs/lerobot_native_collection.md`(~7 处)的
     `./run_panthera_lerobot_daemon.sh …` → `bash arm/panthera/run_daemon.sh …`(与主 README
     canonical 入口统一,消除双入口歧义)
   - `tests/test_collect_scripts.py:27-31/177/303` 的 shim 存在性/转发断言删除或改指新入口
5. **`arm/panthera/run_daemon.sh`**:
   ```bash
   REPO_ROOT="$(dirname "$(dirname "${ARM_ROOT}")")"        # 原 DEPLOY_ROOT 同值,改名以对齐新语义
   COLLECT_TOOL="${REPO_ROOT}/plugins/panthera-lerobot-collect"
   ```
   顶部注释里的旧路径(`data/collect/...`)更新。
6. **`plugins/.../tests/` 重锚(本 Phase CI 全绿的前提,初版漏项)**:
   `test_collect_scripts.py` 的 `ROOT.parents[N]` 位置推导在插件上移一层后全部错位
   (`parents[2]` 指到仓库外、`parents[1]` 从 `data/` 变成仓库根),且多处断言锁定旧字符串:
   - `tests/conftest.py` 增加标记式寻根 `repo_root()`(向上找 `name = "finearm"` 的 pyproject,
     与 `finearm.core.paths` 同逻辑);conftest:9 的 `parents[3] / "src"` 一并换掉
     (当前恰好同深不碎,但别再赌层数)
   - 行 16/36/202/218/238 的 `parents[2]` → `repo_root()`;行 192 `parents[1]/"datasets"` →
     `repo_root()/"data"/"datasets"`;行 237 `parents[1]/"README.md"` →
     `repo_root()/"data"/"README.md"`;行 238 `parents[2]/"infer"` → `repo_root()/"infer"`
     (Phase 2 后随 `envs/infer` 再跟一次)
   - 断言更新:行 211-213 的 `DATA_ROOT`/`DEPLOY_ROOT` 推导串断言 → 新 `REPO_ROOT` 推导串;
     **行 359 现断言手册必须包含 `deploy/data/datasets/lerobot/`(正是 Phase 3 要清除的
     旧串),反转为 `data/datasets/lerobot/` + `assertNotIn("deploy/")`**
7. **`.github/workflows/ci.yml`** `data-tests` job:
   ```yaml
   - run: pip install -e plugins/panthera-lerobot-collect --no-deps
   - name: Data-collection suite (panthera-lerobot-collect)
     working-directory: plugins/panthera-lerobot-collect
   ```
8. **文档**(同 PR 内改完,不留旧路径):
   - `README.md` / `README_zh.md`:布局表第 19 行、采集命令里的三处路径(122/136/143 行附近)、测试一节(293 行)
   - `plugins/.../docs/quest2ros_setup.md`:约 15 处 `deploy/data/collect/panthera-lerobot-collect` → `plugins/panthera-lerobot-collect`(操作员手册,**必须与代码同 PR 更新**);另 151/288/396 行的 `deploy/data/datasets/…` → `data/datasets/…`
   - `plugins/.../docs/lerobot_native_collection.md`:同样清扫
   - **`.env.example`(行 3-10)**:整段 `<deploy>/…` 注释重写(操作员会 source 此文件;初版漏项)
   - `data/README.md` / `README_zh.md`:删掉 collect 段落(Phase 3 整体重写,本 Phase 先改指向)

**真机窗口(Phase 1 → Phase 2):** 采集机 pull Phase 1 后 `.venv-lerobot` 已随目录搬坏,
而 venv 查找路径要到 Phase 2 才切 `envs/collect/`。推荐:采集机在两 Phase 之间**暂停采集**,
两个 Phase 都 pull 后按 Phase 4 流程一次重建(lerobot 环境重建要几分钟,能省一次是一次);
若必须中间采集,`./plugins/.../scripts/init_workspace.sh --install` 在插件内新位置重建即可,
Phase 2 合入后再建一次。

### Phase 2 · 环境集中:`envs/`

0. **先删 `.gitignore:6` 的 `/envs/` 行(硬前置 · blocker)**:pre-refactor 产物目录的
   遗留忽略,会吞掉整个新 `envs/` —— `git check-ignore` 已验证
   `envs/infer/init_env.sh`、`envs/collect/requirements-lerobot.txt`、`envs/arm/init_env.sh`
   全部命中。不删则 env 定义进不了 git、CI checkout 上没有 envs/、真机 pull 后全断。
   删除后全局 `.venv*/` 通配仍覆盖 `envs/**/.venv*`。落地验证:`git check-ignore
   envs/infer/init_env.sh` 无输出、`git status` 显示 envs/ 下为 renamed。
1. **建目录并搬文件**:
   ```bash
   mkdir -p envs/collect envs/arm
   git mv infer/init_env.sh infer/requirements.txt infer/README.md infer/README_zh.md envs/infer/
   git mv plugins/panthera-lerobot-collect/requirements-lerobot.txt envs/collect/
   git mv arm/panthera/init_env.sh envs/arm/init_env.sh
   rm -rf infer/.venv        # 仅本机存在时;venv 未跟踪且不可搬,直接删(真机另有重建流程)
   rmdir infer               # 注意:.venv 不先删则 rmdir 失败(目录非空)
   ```
2. **`envs/infer/init_env.sh`**:脚本位于 `envs/infer/`,repo 根 = 上两级;
   `VENV` 改为 `envs/infer/.venv`(原 `${REPO_ROOT}/.venv` 即本目录 `.venv`,语义不变);
   末尾 echo 里的旧路径(`deploy/data/collect/...`)更新为 `arm/panthera/run_daemon.sh`。
3. **`envs/collect/README.md` + `README_zh.md`**(新建,短):说明 LeRobot 0.5.1 环境、
   `requirements-lerobot.txt`、venv 位置;`init_workspace.sh --install` 仍是唯一入口。
4. **`plugins/.../scripts/init_workspace.sh`**:`requirements-lerobot.txt` 路径改
   `${REPO_ROOT}/envs/collect/requirements-lerobot.txt`;venv `PANTHERA_PYTHON` 默认值改
   `${REPO_ROOT}/envs/collect/.venv-lerobot`。
5. **venv 落点切换**(三个脚本统一):
   - `run_panthera_lerobot_collect.sh`:`.venv-lerobot` 查找 → `envs/collect/.venv-lerobot`
   - `arm/panthera/run_daemon.sh`:fallback 顺序 → `envs/arm/.venv` → `envs/collect/.venv-lerobot`
   - `envs/arm/init_env.sh`:`VENV` → `envs/arm/.venv`;SDK 路径指回
     `${REPO_ROOT}/arm/panthera/sdk/Panthera-HT_SDK`;末尾 hint 改 `plugins/...`
   - 插件自身 `.gitignore` 的 `/.venv-lerobot/` 死条目删除(venv 已不在插件内)
6. **`envs/arm/README.md`**(新建,短):指向 `arm/panthera/README.md`(SDK/标定文档不动)。
7. **文档**:`README.md`/`README_zh.md` 中 `source infer/.venv/bin/activate` →
   `source envs/infer/.venv/bin/activate`;`data/README*` 中 venv 路径同步;
   `envs/infer/README*.md` 内部自引用路径同步。
8. **`tests/`**:本 Phase 不涉及 src 代码,六个 suite 应全绿;若 `tests/data` 断言插件内
   `requirements-lerobot.txt` 存在,改为断言 `envs/collect/` 下存在
   (`plugins/.../tests/test_collect_scripts.py:134` 同样检查,随插件一起改)。

### Phase 3 · `data/` 变纯 + 修活跃 bug + 清扫

1. **重写 `data/README.md` / `README_zh.md`**:只讲产物(datasets/、logs/、`PANTHERA_DATA_ROOT`、
   LeRobot v3 格式、验证方法),采集操作指向 `plugins/.../docs/quest2ros_setup.md`。
2. **修 `acquisition_interface` 默认值 —— 删条目,不是改指向**(已验证:无覆盖时
   `resolve_path` 返回 `<repo>/infer/robot_interface.py`,该文件不存在,`robot.py:88`
   直接 `FileNotFoundError` —— 并非"永远走 env/local 覆盖"的静默回退,真机能跑只是
   因为设了覆盖):
   - `src/finearm/core/paths.py:27`:**删除** `"acquisition_interface"` 整条。删后无覆盖 →
     `resolve_path` 返回 `None` → `robot.py:98` 走 `from finearm.infer import robot_interface`
     包导入 —— 这正是 `robot.py` docstring 与 README 四层配置段声明的默认行为。
     (改指向 `src/finearm/infer/robot_interface.py` 也可用,但会经 `spec_from_file_location`
     加载出第二个模块实例、与包模块并存,各自持有相机/daemon 状态 —— 没必要。)
   - 同步:`tests/core/test_paths.py:36-39` 删对应用例;`src/finearm/rl/envs/panthera/robot.py:47-49`
     docstring;`configs/local.example.jsonc:16` 注释;`README.md:44-46` / `README_zh.md:37`
     四层配置段(默认清单去掉 acquisition 项,注明默认=包模块)。
   跑 `python -m pytest tests/rl/ tests/core/` 确认无断言依赖旧值。
3. **新增 `tests/core/test_repo_layout.py`(把一次性 grep 门槛变成永久 CI 门槛)**:
   - 断言每条 repo-relative `DEFAULT_PATHS` 在当前 checkout 真实存在 —— 这一条能当场
     抓住本项修复的悬空默认这类 bug;
   - 断言已删除/改名的目录不再被引用(`data/collect`、顶层 `infer/`、`deploy/` 前缀
     出现在 README/docs/workflow 即红)。
4. **清扫遗留注释**:全仓(除 `models/openpi/`)grep `deploy/data`、`deploy/infer`、
   `deploy/arm`、`REFACTOR_PLAN`,逐条更新或删除。已知位置:
   - `utils/lock_cameras.sh` 顶部注释(3 处)
   - `pyproject.toml:23`(REFACTOR_PLAN 注释)
   - `RL/rlt_panthera/README.md:112-125`(机器本地绝对路径 `~/.../deploy/infer/.venv` —
     改成 repo-relative 写法 `envs/infer/.venv/bin/python`)
   - `src/finearm/infer/robot_interface.py`、`src/finearm/rl/envs/robot_interface.py`
     docstring 中的历史出处说明(保留,属历史记录;只把"现在在哪"的描述改准)
5. **`.gitignore`**:`/envs/` 遗留行已在 Phase 2.0 删除(初版"无需改"的判断有误,
   见 §0);`/data/datasets/`、`/data/logs/` 保持。可加一行注释注明 envs 下 venv 靠
   `.venv*/` 通配不入库。

### Phase 4 · 验证

**仓库内(每 Phase 收尾 + 最终全量):**
```bash
ruff check .
python -m pytest tests/core/ tests/arms/ tests/infer/ tests/data/
python -m pytest tests/rl/                                  # 需 torch 环境
cd plugins/panthera-lerobot-collect && python -m pytest tests/

# envs/ 必须已被跟踪(Phase 2.0 的 gitignore 修复未被回退)
test -z "$(git check-ignore envs/infer/init_env.sh envs/collect/requirements-lerobot.txt)" \
  || echo "WARN: envs/ 仍被 .gitignore 吞!"

# 残留路径扫描(预期:仅历史出处类 docstring 命中)
grep -rn "data/collect\|deploy/data\|deploy/infer\|deploy/arm" \
     --exclude-dir=.git --exclude-dir=models .
grep -rn "source infer/\|infer/\.venv" --exclude-dir=.git --exclude-dir=models .
```

**无硬件冒烟(任意机器):**
```bash
finearm-daemon --arm panthera_ht --mock --loop-hz 1000
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
finearm-collect --help
envs/infer/init_env.sh --help 2>/dev/null || bash -n envs/infer/init_env.sh   # 语法检查
bash -n plugins/panthera-lerobot-collect/run_panthera_lerobot_collect.sh
bash -n plugins/panthera-lerobot-collect/scripts/init_workspace.sh
bash -n arm/panthera/run_daemon.sh
```

**真机(executor box / 采集机)迁移 —— 注意 venv 不可搬:**

venv 不是可重定位的(activate 脚本和 shebang 硬编码绝对路径),`mv` 旧 venv 到新路径会静默损坏。
正确做法是重建(脚本幂等、依赖已 pin,代价低):

```bash
git pull
rm -rf infer/.venv data/collect/panthera-lerobot-collect/.venv-lerobot arm/panthera/.venv  # 旧 venv 已失效
./envs/infer/init_env.sh   && envs/infer/.venv/bin/pip install -e . --no-deps
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --install                  # 建 envs/collect/.venv-lerobot + SDK 校验
envs/collect/.venv-lerobot/bin/pip install -e . --no-deps
./envs/arm/init_env.sh
# 验证:三个 console entry --help 全通;README「真机验证清单」7 步走一遍
```

**回滚:** 每个 Phase 一个独立 PR,`git revert` 即可;无数据迁移(`data/` 下产物路径未变)。

## 4 · 风险与对策

| 风险 | 对策 |
|---|---|
| 操作员手册(quest2ros_setup.md)路径失效,采集机上的人按旧文档操作 | Phase 1 同 PR 更新;PR 描述里放旧→新路径对照表;通知操作员重新 pull |
| 脚本自定位推导改错层(多减/少减一级 dirname) | Phase 3 冒烟清单逐个 `bash -n` + repo 根干跑 `--check`;根治靠 §6.1 标记寻根(后续 PR) |
| 新 `envs/` 被遗留 `/envs/` 忽略规则整体吞掉 | Phase 2.0 同 PR 删行 + `git check-ignore` 验证(§0 已验证风险真实存在) |
| 插件测试 `parents[N]` 错位、旧串断言(如 :359 锁定 `deploy/data/datasets`) | Phase 1.6 标记式重锚 + 断言反转;CI 当场红,不会带病合入 |
| Phase 1→2 之间采集机 venv 搬坏 | Phase 1"真机窗口":推荐暂停采集、两 Phase 合入后一次重建 |
| 真机 venv 路径失效 | 上文重建流程;init 脚本依赖全部 pin,可复现 |
| git mv 后文件历史断裂感 | 全部用 `git mv`,`git log --follow` 可追;PR 单一目的、不夹带逻辑改动 |
| 漏改某个引用 | Phase 3 的两条 grep 扫描为发布门槛;`models/openpi` 明确豁免(vendored 上游) |
| `acquisition_interface` 默认值修复引入行为变化 | 已验证:当前无覆盖即 `FileNotFoundError`(非静默回退),真机靠 env/local 覆盖掩盖;删默认后无覆盖 → 包导入,与 docstring/README 声明一致;`tests/rl`、`tests/core` 把关 |

## 5 · 明确不做的事

- 不把 `plugins/panthera-lerobot-collect` 合进 `src/finearm` — LeRobot 0.5.1 依赖宇宙隔离是刻意的。
- 不合并三个 venv — 依赖冲突(openpi-client vs lerobot vs SDK 轮子)。
- 不给旧路径留 wrapper shim — 小团队、文档同 PR 更新,双路径并存只会重新制造歧义。
- 不动 `finearm.core.paths` 的解析机制与四层配置模型 — repo-relative 默认值随 checkout 走是本仓库最重要的设计。
- 不在搬家 PR 里引入 `utils/repo_root.sh` 标记寻根、shellcheck、doctor 等行为/工具改动 — 归入 §6 后续加固,保持搬家 PR 纯机械、可独立 revert。

## 6 · 后续加固(搬家完成后,各自独立小 PR;不阻塞本计划)

1. **`utils/repo_root.sh` 标记式寻根**:仓库现有两套寻根机制 —— `finearm.core.paths` 用
   pyproject 标记(稳),bash 全靠 dirname 数层数(已被目录重组烧过两次)。写一个 ~10 行的
   walk-up(向上找 `name = "finearm"` 的 pyproject),所有 wrapper `source` 它,推导一律从
   `REPO_ROOT` 出发。消灭"数错层"这一整类风险(§4 风险表的根治项)。
2. **CI 加 shellcheck**(lint job 一行):`bash -n` 只查语法;`set -u` 未定义变量
   (Phase 1 修的 `cam_lock` 那类)shellcheck 可静态抓出。存量告警可先 `-S error` 起步。
3. **`finearm-doctor` 体检命令**:一条命令检查三个环境(lerobot==0.5.1、openpi-client 可
   import、SDK wheel + revision pin)、三个 console entry 是否已装、标定文件存在、相机可达
   (无硬件安全项)。现在这些检查散在 `init_workspace --check`、采集 preflight、README 七步
   清单里;多机协作(executor / GPU / 采集机)下这是易用性性价比最高的一项。
4. **README 加"环境 × 机器"矩阵**:三 venv × 三机(executor box / GPU box / 采集机)一张表,
   新成员不用脑补拓扑。
