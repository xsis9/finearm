# daemon · Panthera-HT driver assets

[English](daemon.md) · [中文](daemon_zh.md)

Per-machine arm assets. The daemon CODE lives in `src/finearm/arms/`
(canonical entry `finearm-daemon --arm panthera_ht`; `run_daemon.sh` here
drives it directly). This directory is the authoritative home for the
vendor SDK and per-machine calibration; the driver ENVIRONMENT lives in
`envs/arm/` (repo rule: all venvs under `envs/`).

| Path | Purpose |
|---|---|
| `run_daemon.sh` | Daemon launcher (`--live/--mock/--hz` + passthrough daemon flags); prefers `envs/arm/.venv`, falls back to the collect venv |
| `sdk/Panthera-HT_SDK/` | Vendor SDK tree (gitignored; cloned/verified by the collect tool's `init_workspace.sh`) |
| `robot_param/` | **Per-machine calibration** (Leader/Follower.yaml + motor_param) |

## First-time setup

```bash
./envs/arm/init_env.sh              # driver venv (falls back to the collect venv if absent)
```

## Daemon lifecycle

```bash
# bench check (no hardware):
bash arm/panthera/run_daemon.sh --mock --hz 1000

# live — ALWAYS pass the calibration explicitly (defaults resolve to the SDK
# tree copy, which is wrong for a machine with re-mapped serial ids):
bash arm/panthera/run_daemon.sh --live --hz 1000 \
    --leader-config arm/panthera/robot_param/Leader.yaml \
    --follower-config arm/panthera/robot_param/Follower.yaml
# single-arm variant (policy/Quest setups): add --no-leader
```

Safe-start order: the daemon comes up in **hold** → read telemetry and
confirm every joint channel tracks (move the leader by hand) → only then
switch modes. Collect/infer clients REUSE a running daemon (probe accepts
mode ∈ {teleop, policy, freedrive}).

To stop it, kill the **python** PID (not the bash wrapper):

```bash
ps aux | grep finearm.arms.daemon | grep -v grep   # find the python PID
kill <PID>
```

## Talking to the daemon (any client)

```python
from finearm.arms.daemon.ipc import ControlClient   # src on sys.path, or finearm installed

client = ControlClient("/tmp/panthera_lerobot_control.sock")
client.status()                            # daemon status + profile handshake
snap = client.snapshot(max_age_s=0.25)     # {"observation.state": [14 dims], ...}
client.set_mode("policy")                  # hold | policy | teleop | freedrive
client.send_policy_action([7-dim absolute targets])   # list/tuple only (.tolist() first)
client.hold()
```

## Calibration notes

- `robot_param/` is machine-local and git-tracked as the working copy;
  serial ids must match the machine's actual bus layout.
- `configs/arms/panthera_ht/` (repo root) offers opt-in override YAMLs —
  resolution order: CLI `--leader-config/--follower-config` >
  `configs/arms/panthera_ht/*.yaml` > SDK-tree copy.

## Known traps

- **Leader reads 999 / no data** — wrong calibration: pass
  `--leader-config/--follower-config` explicitly.
- **Follower power loss** → SDK reconnect loop aborts (exit 134): restore
  power, restart the daemon.
