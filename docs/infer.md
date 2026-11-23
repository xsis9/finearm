# infer · serving # infer · inference environment inference

[English](infer.md) · [中文](infer_zh.md)

The inference client environment (openpi-client venv). The client CODE
lives in `src/finearm/infer/` — model adapters, chunk executors
(`finearm-infer --exec-mode {rtc,stop-and-go}`), and the acquisition
backend (`finearm.infer.robot_interface`: daemon IPC + cameras).

## Setup (executor box, one-time)

```bash
./envs/infer/init_env.sh                                  # env with openpi-client (or reuse an existing one)
envs/infer/.venv/bin/pip install -e . --no-deps           # finearm console entries
```

## The three parties

| Party | Machine | Command |
|---|---|---|
| Policy server | GPU box (4090) | `uv run scripts/serve_policy.py …` (see below; vendored tree at `finearm/models/openpi`) |
| Tunnel | executor box | `ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>` |
| Inference client | executor box | `finearm-infer …` (ladder below) |

### Serve (GPU box) — pi05

```bash
git clone git@github.com:xsis9/finearm.git && cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync

# bring the trained checkpoint from the training machine:
rsync -av <user>@<trainer>:<openpi-tree>/checkpoints/pi05_panthera*/<exp>/ checkpoints/

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera_p3 \
    --policy.dir checkpoints/pi05_panthera_p3/<exp>/<step>
# port 8000 (default); add --record to log every request's obs/actions.
```

### Serve — lingbot (external stack, port 8006)

```bash
cd <lingbot-vla-v2 repo root>          # MUST be the repo root: configs resolve relatively
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/finetuned/hf_ckpt --use_bf16 true --use_compile true \
  --use_length 25 --port 8006
```

lingbot checklist: the checkpoint must be the `hf_ckpt` dir that still
contains its training config; `--use_length` ≤ the training action horizon;
`configs/robot_configs/panthera.yaml` + `assets/norm_stats/panthera.json`
must be present and norm-stats must be the same file used in training.

### Infer (executor box) — safety ladder

```bash
cd <finearm repo root> && source envs/infer/.venv/bin/activate

# level 1 · no hardware (mock adapter, no server):
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
# level 2 · real cameras + daemon, actions printed not sent:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --dry-run --duration 60
# level 3 · real control, low speed, stop-on-starve:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30
# level 4 · normal:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --task "<training prompt>"
```

Pre-flight on the executor box:

```bash
bash utils/lock_cameras.sh                              # FIRST: camera params must match training/inference
python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs   # cameras/latency/orientation
PYTHONPATH=src python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1   # RTT + suggested k
```

The camera lock is not just manual pre-flight: the acquisition backend
(`finearm.infer.robot_interface`) auto-runs `utils/lock_cameras.sh` at
startup against the same authoritative profile collection verifies
against — UVC params reset on replug/power-cycle, so a mismatched camera
silently shifts the observation distribution vs training. Override the
script with `PANTHERA_CAM_LOCK`; skip (debug only) with
`PANTHERA_CAM_LOCK_SKIP=1`.

## Key parameters

| Parameter | Notes |
|---|---|
| `--task` | MUST equal the training prompt verbatim (handshake-checked) |
| `--home` | home pose (7 values); `--home` > `PANTHERA_RTC_HOME` > adapter default; empty string skips homing |
| `--exec-mode` | `rtc` (pipelined chunks) or `stop-and-go` (windowed) |
| `--k/--blend/--starve` | replan margin / blend steps / starvation policy (`hold` or `stop`) |
| `--pos-lo/--pos-hi/--vel-limit` | explicit limits (the arm side only keeps a per-tick jump guard) |
| `PANTHERA_CAM_FRONT/WRIST` | camera sources (default `/dev/video4`, `/dev/video2`) |
| `PANTHERA_CAM_LOCK/SKIP` | camera-lock script override / `=1` to skip (debug only) |

## Adding a model adapter

One file in `src/finearm/infer/adapters/` (filename = `--model` value, no
leading underscore), ~50 lines, subclassing `AdapterBase`. The executor,
control loop, and safety logic are model-agnostic; the adapter only
declares the connection, default port, observation style, and how
infer/reset map to the server protocol. Existing examples: `pi05.py`,
`lingbot.py`, `mock.py`. The server needs no changes for RTC — at most one
in-flight request at a time per connection.
