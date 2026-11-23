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
| `models/` | Fine-tuning (SFT): the vendored modified openpi tree (upstream `215abfb` + panthera patch; provenance in `third_party/`) + the SFT how-to ([`models/README.md`](models/README.md)) |
| `configs/` | Machine/arm/model configuration (canonical copies) |
| `third_party/` | Upstream baseline records for vendored dependencies |
| `docs/` | Working notes and hardware manuals |

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

Each stage's full commands live in that stage's own doc — this page is the
map. Environments are built once per machine from the READMEs under `envs/`;
the `finearm` console entries install into a venv with
`<venv>/bin/pip install -e . --no-deps` (run from the repo root).

| Stage | Entry | Environment | Full how-to |
|---|---|---|---|
| Arm daemon | `bash arm/panthera/run_daemon.sh --live …` (or `finearm-daemon --arm panthera_ht`) | `envs/arm/` (or any venv with the vendor SDK) | [`arm/panthera/README.md`](arm/panthera/README.md) |
| Collect (teleop) | `finearm-collect --method teleop --live …` | `envs/collect/` (LeRobot 0.5.1) | [`data/README.md`](data/README.md) · operator manual [`quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md) |
| Fine-tune (SFT) | `uv run python scripts/train.py --config pi05_panthera_p3` in `models/openpi/` | GPU box (`uv sync`) | [`models/README.md`](models/README.md) |
| Serve | `uv run scripts/serve_policy.py …` in `models/openpi/` | GPU box (≥8 GB VRAM for π0.5 bf16) | [`envs/infer/README.md`](envs/infer/README.md) |
| Infer | `finearm-infer --model <name> --exec-mode rtc …` | `envs/infer/` on the executor box | safety ladder in [`envs/infer/README.md`](envs/infer/README.md) |
| RL post-training | `python -m finearm.rl.train_stage2 …` | `envs/infer/` (+ torch) | [`RL/rlt_panthera/README.md`](RL/rlt_panthera/README.md) |

The launcher wrappers (`run_daemon.sh`, `run_panthera_lerobot_collect.sh`)
remain the recommended operational entries — they manage venv selection.
Datasets land under `data/datasets/` (LeRobot-native v3, directly trainable);
the dataset → stats → checkpoint → task handoff contract is documented in
[`models/README.md`](models/README.md).

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
