# infer · serving & inference

[English](infer.md) · [中文](infer_zh.md)


## Setup (executor box, one-time)

```bash
./envs/infer/init_env.sh                                  # env with openpi-client (or reuse an existing one)
envs/infer/.venv/bin/pip install -e . --no-deps           # finearm console entries
```

## Division of labor

| Party | Machine | Command |
|---|---|---|
| Policy server | GPU box (4090) | `uv run scripts/serve_policy.py …` (below) |
| Inference client | executor box | `finearm-infer …` (safety ladder below) |

### Serve (GPU box) — pi05

```bash
cd finearm/models/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync

# bring the checkpoint over from the training machine to the inference machine

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_panthera \
    --policy.dir checkpoints/pi05_panthera/<exp>/<step>
# default port 8000; add --record while debugging (logs every request's obs/actions)
```

### Infer (executor box) — safety ladder

Pre-flight on the executor box:

```bash
bash utils/lock_cameras.sh                              # FIRST, always: camera params must match training/inference
python -m finearm.infer.robot_interface --selftest --dump-obs /tmp/rtc_obs   # cameras/latency/orientation
PYTHONPATH=src python -m finearm.rl.scripts.bench_rtt --model <name> --host 127.0.0.1   # RTT + suggested k
```

The camera lock is not just one-time pre-flight: the acquisition backend
(`finearm.infer.robot_interface`) auto-runs `utils/lock_cameras.sh` at
startup — the same authoritative profile the collection side uses. UVC
params reset on replug/power-cycle; a mismatch silently shifts the
observation distribution vs training. `PANTHERA_CAM_LOCK` overrides the
script path; `PANTHERA_CAM_LOCK_SKIP=1` skips it (debug only).

```bash
cd <finearm repo root> && source envs/infer/.venv/bin/activate

# The company's 4090 server runs docker and is not directly reachable, so
# first ssh-tunnel from the executor box to the inference server, then
# reach it via 127.0.0.1 from the executor box:
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-host>

# level 1 · no hardware (mock adapter, no server):
finearm-infer --model mock --host 127.0.0.1 --exec-mode rtc --mock --max-steps 300
# level 2 · real cameras + daemon, actions printed not sent:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --dry-run --duration 60
# level 3 · real control at low speed, stop-on-starve:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --starve stop \
    --vel-limit 0.01,0.01,0.01,0.01,0.01,0.01,0.005 --duration 30
# level 4 · normal:
finearm-infer --model pi05 --host 127.0.0.1 --exec-mode rtc --task "<training prompt>"
```

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
`lingbot.py`, `mock.py`.
