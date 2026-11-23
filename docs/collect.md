# collect · data collection

[English](collect.md) · [中文](collect_zh.md)

Data-collection **outputs**: the dataset hub (`data/datasets/`) and logs.
The collection stack (LeRobot plugins, launcher, operator manual) lives in
`plugins/panthera-lerobot-collect/`; the collector CODE lives in
`src/finearm/data/` (entry `finearm-collect`). Output datasets are
**LeRobot-native v3** — directly trainable, no conversion.

## Layout

```text
data/
├── datasets/                           # datasets land here: lerobot/<ns>/<repo-id>/<timestamp>/
└── logs/                               # daemon / diagnostics logs
```

## First-time setup

```bash
# collect env (LeRobot 0.5.1 + plugins + SDK wheel; also clones/verifies the vendor SDK):
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --install
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --check     # verify

# driver env for the daemon (see docs/daemon.md):
./envs/arm/init_env.sh

# finearm console entries into the collect env (one-time):
envs/collect/.venv-lerobot/bin/pip install -e . --no-deps
```

## Collecting (leader-follower teleop)

```bash
bash utils/lock_cameras.sh        # FIRST, every batch: camera params must match training/inference

source envs/collect/.venv-lerobot/bin/activate
# (preflight requires lerobot-record on PATH — the venv must be activated)

finearm-collect --method teleop --live \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml \
    --select-cameras \
    --repo-id <namespace>/<dataset> --task "<task prompt>" \
    --episodes 1 --episode-seconds 30 --reset-seconds 60 \
    --fps 30 --hz 1000 --camera-fourcc MJPG

# or the launcher wrapper (same flags, manages the venv):
#   ./plugins/panthera-lerobot-collect/run_panthera_lerobot_collect.sh --live …
```

Other methods — freedrive hand-teaching, Quest3S teleop — are documented
step-by-step in the operator manual:
[`plugins/panthera-lerobot-collect/docs/quest2ros_setup.md`](../plugins/panthera-lerobot-collect/docs/quest2ros_setup.md)
(parameter tables, calibration values, batch SOP).

## Dataset output

- Location: `data/datasets/lerobot/<namespace>/<repo-id>/<timestamp>/`
  (override: `PANTHERA_DATA_ROOT`)
- Verify after a test episode:
  `meta/info.json` → `fps=30`, `total_frames = seconds × 30`,
  `robot_type=panthera_ht_follower`
- The `--task` string is stored with the dataset and must equal the training
  prompt verbatim (checked again at the serve/infer handshake).

## Known traps

- `lerobot-record not found` at preflight — the collect venv is not on
  PATH: activate it (see above), don't call the console entry by absolute
  path alone.
- Leader reads 999 / no data — wrong calibration: always pass
  `--leader-config/--follower-config` explicitly on the real machine.
- DAgger collection is a registry placeholder (`finearm.data.collector.dagger`);
  invoking it raises with guidance.
