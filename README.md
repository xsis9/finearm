# finearm

[English](README.md) · [中文](README_zh.md)

Robot-arm VLA fine-tuning framework: teleop data collection, real-time chunked
inference, and RL post-training on top of pre-trained VLA policies
(pi0.5 / Panthera-HT arm).

The `finearm` package owns all code; the surrounding directories hold
per-machine assets (SDK, calibration), environments, data, and the LeRobot
plugin packages. Adding an arm, a model adapter, or an RL algorithm = adding
one plugin directory, nothing else changes.

## Repository layout

| Path | Contents |
|---|---|
| `arm/panthera/` | Per-machine arm assets: vendor SDK (gitignored), calibration (`robot_param/`), daemon launcher (`run_daemon.sh`; driver env in `envs/arm/`) |
| `plugins/panthera-lerobot-collect/` | LeRobot collection stack: robot/teleoperator plugins, launcher wrapper, operator docs |
| `envs/` | All environment definitions + venvs: `infer/` (openpi-client), `collect/` (LeRobot 0.5.1), `arm/` (driver) |
| `RL/rlt_panthera/` | RL run artifacts: campaign checkpoints/data (gitignored) and the stage-2 config; the code lives in `src/finearm/rl/` |
| `src/finearm/core/` | Zero-dependency core: four-layer config resolution, repo paths, plugin registry, norm-stats contract |
| `src/finearm/arms/` | Arm plugins: `ArmProfile` schema declaration + arm-neutral control daemon (`finearm-daemon --arm <id>`); `panthera_ht` (real) and `mock_twojoint` (extensibility demo) |
| `src/finearm/infer/` | Model adapters (pi05 / lingbot / mock) + chunk executors: `RTCExecutor` (pipelined) and `StopAndGoExecutor`; `finearm-infer --exec-mode {rtc,stop-and-go}`; acquisition `robot_interface` (daemon IPC + cameras) |
| `src/finearm/rl/` | Stage-2 online RL: `rltoken` algorithm (networks/critical/learner), replay + journal, boundary seam (`WindowReceipt`/`BoundaryCollector`), `EpisodeRunner`, serving (serve_rlt + client), envs (RobotInterface contract + panthera backend), offline pipelines |
| `src/finearm/data/` | Collection methods as plugins: `teleop` (LeRobot-native, `--arm` parameterized) and the `dagger` stub; `finearm-collect --method teleop` |
| `models/openpi/` | Vendored modified openpi (upstream `215abfb` + panthera patch: `panthera_policy`, `pi05_panthera_p3` config, lerobot-0.5.1/numpy-2/py3.12 compat, eval scripts); provenance in `third_party/` |
| `configs/` | Machine/arm/model configuration (canonical copies) |
| `third_party/` | Upstream baseline records for vendored dependencies |
| `doc/` | Working notes and hardware manuals |

## Running on any machine

The repository stores **no machine identity** — no hostnames, no absolute
paths, no per-machine profiles. Clone it anywhere and the built-in defaults
follow the checkout. Per-machine customization uses three override layers,
highest first:

1. **CLI flags** (per invocation, e.g. `--stats_path`)
2. **Environment variables** (e.g. `PANTHERA_ACQ_IFACE`, `PANTHERA_DATA_ROOT`, `FINEARM_STATS_PATH`)
3. **`configs/local.jsonc`** — machine-local persistence, gitignored
   (copy from [`configs/local.example.jsonc`](configs/local.example.jsonc))

Layer 4 (lowest) is repo-relative defaults resolved by `finearm.core.paths`:
`data/datasets` (dataset hub), `arm/panthera/sdk` (vendor SDK), `/tmp`
(daemon socket dir); the acquisition backend defaults to the
`finearm.infer.robot_interface` package module. `FINEARM_ROOT` points at a
checkout from an environment where the package is installed elsewhere.

### Topologies

- **Single machine (local inference)**: one Linux box with a CUDA GPU
  (≥ 8 GB VRAM to serve π0.5 in bf16) runs the inference server, the arm
  daemon, and the collection/RL clients together — everything defaults to
  `127.0.0.1`.
- **Split executor + inference server**: the arm-attached executor box runs
  daemon/collection/RL; a GPU box serves the policy. Forward the port over
  ssh (`ssh -N -L 8000:127.0.0.1:8000 <gpu-host>`) and keep
  `server_host 127.0.0.1` — clients cannot tell the difference.

`finearm.core` needs only a Python ≥ 3.11 standard library; heavier
dependencies live in the per-domain environments (see each domain's README).

### Arm daemon

The control daemon is arm-neutral; each arm is one plugin directory declaring
its schema, gains, and adapters:

```bash
finearm-daemon --arm panthera_ht --mock
```

Adding an arm = adding `src/finearm/arms/<arm>/` (see `mock_twojoint/` for a
minimal skeleton); daemon, IPC, and clients pick up different dimensions
automatically.

## The pipeline: collect → fine-tune → serve → infer

One-time setup — install the package (pure Python, no new dependencies)
into the environments that need the console entries:

```bash
# collect env:                          # infer env:
cd <repo> && <collect-venv>/bin/pip install -e . --no-deps
                                       <infer-venv>/bin/pip install -e . --no-deps   # (same, from <repo>)
```

| Stage | Canonical entry | Environment |
|---|---|---|
| Arm daemon | `finearm-daemon --arm panthera_ht --live` | arm venv or any venv with the vendor SDK |
| Collect | `finearm-collect --method teleop --live …` | collect venv (LeRobot 0.5.1) |
| Fine-tune | external — training machine's openpi / lingbot stack | see handoff below |
| Serve | openpi / lingbot serve on the GPU box (≥8 GB for π0.5 bf16) | GPU box |
| Infer | `finearm-infer --model <name> --exec-mode rtc …` | infer venv (openpi-client) |

The collection launcher wrapper (`run_panthera_lerobot_collect.sh`) and the
daemon launcher (`run_daemon.sh`) remain the recommended operational
entries — both now drive the `finearm` modules directly.

### 1 · Arm daemon

```bash
# bench check first (no hardware):
finearm-daemon --arm panthera_ht --mock --loop-hz 1000
# live (explicit calibration — see arm/panthera/README.md for robot_param):
bash arm/panthera/run_daemon.sh --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml
```

Safe-start order on real hardware: daemon comes up in **hold** → read
telemetry and confirm every joint channel tracks (move the leader by hand)
→ only then switch to `teleop`/`policy`. Collect and infer reuse an
already-running daemon (probe requires mode ∈ {teleop, policy, freedrive});
to stop it, kill the **python** PID (not the bash wrapper).

### 2 · Collect (teleop)

```bash
bash utils/lock_cameras.sh        # FIRST, always: camera params must match training/inference

source envs/collect/.venv-lerobot/bin/activate
# (preflight requires lerobot-record on PATH — activate the collect venv or
#  prepend its bin/ dir; invoking the console entry by absolute path alone
#  is not enough)

# canonical entry (same flags as the legacy wrapper):
finearm-collect --method teleop --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml \
    --select-cameras \
    --repo-id <namespace>/<dataset> --task "<task prompt>" \
    --episodes 1 --episode-seconds 30 --reset-seconds 60 \
    --fps 30 --hz 1000 --camera-fourcc MJPG
# or the launcher wrapper (same flags, manages the venv for you):
#   ./plugins/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live …
```

Datasets land under `data/datasets/lerobot/<namespace>/<repo-id>/<timestamp>/`
(override: `PANTHERA_DATA_ROOT`). Output is **LeRobot-native v3** — directly
trainable, no conversion. The authoritative operator manual (three collection
methods, parameter tables, batch SOP) is
[`plugins/panthera-lerobot-collect/docs/quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md).

### 3 · Fine-tune (external handoff)

SFT/fine-tuning runs on the training machine (the modified openpi source
is vendored at `models/openpi/` — see `third_party/patches/` for the exact
delta vs upstream; the lingbot stack stays external). The handoff
contract that makes a collected dataset trainable and the resulting
checkpoint servable:

1. the dataset directory (LeRobot v3) is the sole training input;
2. the **task string** recorded at collection must equal the training
   prompt verbatim — it is checked again at the serve/infer handshake;
3. any post-training must use the quantile stats of the same dataset
   (`meta/stats.json`) — the same-origin chain (dataset → stats →
   checkpoint → served task) is validated end-to-end at startup;
4. the trained checkpoint + its task string are what stage 4 serves.

#### SFT quickstart (π0.5, from this repo)

The vendored `models/openpi` is the runnable training tree (training machine
or any CUDA box; the upstream env is reproduced exactly by the committed
`uv.lock`):

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, once
uv sync                                            # reproduce the training env

# heavy assets (not in git) — copy once from wherever they live:
#   pi0.5 base weights → checkpoints/pi05_base/          (referenced tree-relative)
#   datasets (panthera_bolt_p3 …) → any dir, then:
export LEROBOT_HOME=<dir-containing-the-datasets>  # lerobot resolves local datasets

# smoke (10 steps — proves weights + data + env before the full run):
uv run python scripts/train.py --config pi05_panthera_p3 num_train_steps=10 exp_name=smoke
# full run:
uv run python scripts/train.py --config pi05_panthera_p3
```

Checkpoints land under `checkpoints/pi05_panthera_p3/<exp_name>/`. Per-checkpoint
auto-evaluation: `bash scripts/watch_panthera_p3.sh` (separate terminal); loss
plot: `uv run python scripts/plot_panthera_p3_loss.py <train.log>`. Rebase onto
a newer upstream: checkout upstream, apply
[`third_party/patches/openpi-panthera.patch`](third_party/patches/openpi-panthera.patch).

### 4 · Serve (GPU box) and tunnel

#### Serve quickstart (π0.5, from this repo)

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, once
uv sync                                            # reproduce the serving env

# bring the trained checkpoint (from the training machine):
rsync -av <user>@<trainer>:<openpi-tree>/checkpoints/pi05_panthera_p3/<exp>/ \
    checkpoints/pi05_panthera_p3/<exp>/

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera_p3 \
    --policy.dir checkpoints/pi05_panthera_p3/<exp>/<step>
# serves on port 8000; add --record while debugging (logs every request's
# obs/actions under policy_records/). Norm-stats assets ship with the repo.
```

Start the policy server on the GPU box (quickstart above; per-model details:
[`envs/infer/README.md`](envs/infer/README.md) — pi05 port 8000, lingbot 8006), then
from the executor box forward the port and keep `--host 127.0.0.1`:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>
```

### 5 · Infer (safety ladder — climb one level at a time)

Runs on the **executor box** (the machine attached to the arm), from the
repo root, in the **infer environment** (openpi-client + the finearm console
entries — same one-time `pip install -e . --no-deps` as the setup above):

```bash
source envs/infer/.venv/bin/activate     # not installed yet → envs/infer/.venv/bin/pip install -e . --no-deps

# level 1 · no hardware, full client stack (mock adapter, no server):
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
# level 2 · real cameras + daemon, actions printed not sent:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --dry-run --duration 60
# level 3 · real control at low speed; stop-on-starve:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30
# level 4 · normal operation:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --task "<training prompt>"
```

`--task` must match the training prompt verbatim. `--exec-mode stop-and-go`
runs the windowed alternative (no chunk pipelining). Pre-flight on the
executor box: `python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs`
(cameras, latency, image orientation) and `python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1`
(RTT percentiles + suggested k; also warms the server's first jit).

### Real-machine verification checklist

1. `pip install -e . --no-deps` into the collect and infer venvs; `--help`
   works for all three console entries.
2. `finearm-daemon --arm panthera_ht --mock` + level-1 infer — no hardware.
3. Live daemon: handshake prints `panthera_ht 14/7`; telemetry tracks the
   leader; stop-and-start again.
4. `bash utils/lock_cameras.sh`; collect **one** episode; inspect the
   dataset (`fps=30`, frames = seconds × 30, `robot_type=panthera_ht_follower`).
5. Serve + tunnel; `bench_rtt` shows p95 and suggests k.
6. Level-2 (`--dry-run`) then level-3 (slow, `--starve stop`) inference.
7. Optional RL preflight (no server, no hardware):
   `python -m finearm.rl.train_stage2 --rlt_checkpoint <ckpt> --stats_path <stats> --dry_run --output_dir <empty-dir>`.

### Troubleshooting (known traps)

- **Leader shows 999 / no data** — wrong calibration: always pass
  `--leader-config/--follower-config` explicitly on the real machine
  (see `configs/arms/panthera_ht/README.md`).
- **Daemon won't die** — kill the python PID, not the bash wrapper;
  a follower power loss makes the SDK reconnect-loop abort (exit 134) —
  restore power, restart the daemon.
- **Cameras** — run `lock_cameras.sh` before EVERY collection batch;
  sources default to `/dev/video4` (front) and `/dev/video2` (wrist),
  override via `PANTHERA_CAM_FRONT/WRIST`.
- **Handshake failures at infer** — task string mismatch or broken
  same-origin chain (stats ↔ checkpoint ↔ dataset); the error names the
  offending field.

## Testing

```bash
# Core suite (pure stdlib, no install needed)
python -m pytest tests/core/

# Arms suite (profiles, daemon mock round-trip)
python -m pytest tests/arms/

# Infer suite (executor math, RTC golden comparison vs pre-refactor, stop-and-go)
python -m pytest tests/infer/

# Collect suite (collector registry, dagger stub, --arm parameterization)
python -m pytest tests/data/

# RL suite (requires: torch, numpy, matplotlib, pytest; golden harness and
# real-input tests are env-gated and skip without machine-local inputs)
python -m pytest tests/rl/

# Data-collection suite (requires: lerobot==0.5.1, pytest)
cd plugins/panthera-lerobot-collect && python -m pytest tests/
```

CI runs lint plus all six suites on every push (`.github/workflows/ci.yml`).
Hardware is never touched by CI or by the test suites.

## License

MIT — see [LICENSE](LICENSE).
