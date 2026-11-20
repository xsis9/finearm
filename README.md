# finearm

Robot-arm VLA fine-tuning framework: teleop data collection, real-time chunked
inference, and RL post-training on top of pre-trained VLA policies
(pi0.5 / Panthera-HT arm).

> **Status: early-stage restructuring.** The codebase is being migrated from a
> deployment tree into the `finearm` package layout (see `REFACTOR_PLAN.md`).
> During migration the existing entry points keep working unchanged.

## Repository layout

| Path | Contents |
|---|---|
| `arm/panthera/` | Vendor SDK, per-machine calibration (`robot_param/`), daemon launcher (`run_daemon.sh`); the `panthera_control.*` modules are forwarding shims to `src/finearm/arms` |
| `data/collect/panthera-lerobot-collect/` | LeRobot teleop collection stack (plugins, wrapper, docs); `collect_cli.py` is a forwarding shim to `src/finearm/data` |
| `infer/` | Inference environment; `rtc_client.py` / `robot_interface.py` are forwarding shims to `src/finearm/infer` |
| `RL/rlt_panthera/` | Online RL post-training (RL-Token): training loop, replay, learner, serving |
| `src/finearm/core/` | Zero-dependency core: four-layer config resolution, repo paths, plugin registry, norm-stats contract |
| `src/finearm/arms/` | Arm plugins: `ArmProfile` schema declaration + arm-neutral control daemon (`finearm-daemon --arm <id>`); `panthera_ht` (real) and `mock_twojoint` (extensibility demo) |
| `src/finearm/infer/` | Model adapters (pi05 / lingbot / mock) + chunk executors: `RTCExecutor` (pipelined) and `StopAndGoExecutor`; `finearm-infer --exec-mode {rtc,stop-and-go}`; acquisition `robot_interface` (daemon IPC + cameras) |
| `src/finearm/rl/` | Stage-2 online RL: `rltoken` algorithm (networks/critical/learner), replay + journal, serving (serve_rlt + client), envs (RobotInterface contract + panthera backend), offline pipelines; `RL/rlt_panthera` keeps shim entry points for campaign continuity |
| `src/finearm/data/` | Collection methods as plugins: `teleop` (LeRobot-native, `--arm` parameterized) and the `dagger` stub; `finearm-collect --method teleop` |
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
`data/datasets` (dataset hub), `arm/panthera/sdk` (vendor SDK),
`infer/robot_interface.py` (acquisition interface), `/tmp` (daemon socket
dir). `FINEARM_ROOT` points at a checkout from an environment where the
package is installed elsewhere.

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
finearm-daemon --arm panthera_ht --mock   # or: python -m panthera_control.cli --mock (legacy path)
```

Adding an arm = adding `src/finearm/arms/<arm>/` (see `mock_twojoint/` for a
minimal skeleton); daemon, IPC, and clients pick up different dimensions
automatically. The legacy `panthera_control.*` import paths keep working as
forwarding shims during the migration.

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

Every legacy entry keeps working as a forwarding shim — `run_panthera_lerobot_collect.sh`,
`python -m panthera_lerobot_control.collect_cli`, `infer/rtc_client.py`,
`python -m panthera_control.cli` — operator muscle memory and existing
scripts are unaffected.

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

# canonical entry (same flags as the legacy wrapper):
finearm-collect --method teleop --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml \
    --select-cameras \
    --repo-id <namespace>/<dataset> --task "<task prompt>" \
    --episodes 1 --episode-seconds 30 --reset-seconds 60 \
    --fps 30 --hz 1000 --camera-fourcc MJPG
# unchanged legacy path:
#   ./data/collect/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live …
```

Datasets land under `data/datasets/lerobot/<namespace>/<repo-id>/<timestamp>/`
(override: `PANTHERA_DATA_ROOT`). Output is **LeRobot-native v3** — directly
trainable, no conversion. The authoritative operator manual (three collection
methods, parameter tables, batch SOP) is
[`data/collect/panthera-lerobot-collect/docs/quest2ros_setup.md`](data/collect/panthera-lerobot-collect/docs/quest2ros_setup.md).

### 3 · Fine-tune (external handoff)

SFT/fine-tuning runs on the training machine's model stacks (openpi for
π0.5, the lingbot stack for lingbot) — not in this repo. The handoff
contract that makes a collected dataset trainable and the resulting
checkpoint servable:

1. the dataset directory (LeRobot v3) is the sole training input;
2. the **task string** recorded at collection must equal the training
   prompt verbatim — it is checked again at the serve/infer handshake;
3. any post-training must use the quantile stats of the same dataset
   (`meta/stats.json`) — the same-origin chain (dataset → stats →
   checkpoint → served task) is validated end-to-end at startup;
4. the trained checkpoint + its task string are what stage 4 serves.

### 4 · Serve (GPU box) and tunnel

Start the policy server on the GPU box (per-model instructions:
[`infer/README.md`](infer/README.md) — pi05 port 8000, lingbot 8006), then
from the executor box forward the port and keep `--host 127.0.0.1`:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>
```

### 5 · Infer (safety ladder — climb one level at a time)

```bash
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

# Arms suite (profiles, mock daemon round-trip, legacy-shim equivalence)
python -m pytest tests/arms/

# Infer suite (executor math, RTC golden comparison vs pre-refactor, stop-and-go)
python -m pytest tests/infer/

# Collect suite (collector registry, dagger stub, --arm parameterization)
python -m pytest tests/data/

# RL suite (requires: torch, numpy, matplotlib, pytest; golden harness and
# real-input tests are env-gated and skip without machine-local inputs)
python -m pytest tests/rl/

# Data-collection suite (requires: lerobot==0.5.1, pytest)
cd data/collect/panthera-lerobot-collect && python -m pytest tests/
```

CI runs lint plus all six suites on every push (`.github/workflows/ci.yml`).
Hardware is never touched by CI or by the test suites.

## License

MIT — see [LICENSE](LICENSE).
