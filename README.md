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
| `arm/panthera/panthera_control/` | Arm-neutral control daemon: controller loop, IPC protocol, robot adapters |
| `data/collect/panthera-lerobot-collect/` | LeRobot teleop data collection (robot + teleoperator plugins, CLI, tests) |
| `infer/` | Real-time chunked inference client (RTC), model adapters, arm/camera interface |
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

# RL suite (requires: torch, numpy, pytest)
cd RL/rlt_panthera && python -m pytest

# Data-collection suite (requires: lerobot==0.5.1, pytest)
cd data/collect/panthera-lerobot-collect && python -m pytest tests/
```

CI runs lint plus all six suites on every push (`.github/workflows/ci.yml`).
Hardware is never touched by CI or by the test suites.

## License

MIT — see [LICENSE](LICENSE).
