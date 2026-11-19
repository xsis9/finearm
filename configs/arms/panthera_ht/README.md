# panthera_ht — per-arm configuration home

This directory is the user-editable home for panthera_ht machine-side
configuration (REFACTOR_PLAN.md §3a).

## Calibration YAMLs (opt-in override)

If you place `Leader.yaml` / `Follower.yaml` here, the daemon prefers them
over the SDK-tree copies (`<sdk_root>/robot_param/*.yaml`). Resolution order:

1. `--leader-config` / `--follower-config` CLI flags
2. `configs/arms/panthera_ht/{Leader,Follower}.yaml` (this directory, if present)
3. SDK tree default `<sdk_root>/robot_param/{Leader,Follower}.yaml`

The files are NOT moved automatically: the two existing copies (tracked
`arm/panthera/robot_param/` and the SDK-tree working copy) differ in their
relative `param_file`/URDF paths, so copying blindly changes what the live
arm reads. To adopt this home, copy a file, adjust its relative paths to this
directory, and verify with a `--mock` daemon round before live use.

Snapshot schema, gains, and dimensions are NOT configured here — they are
declared once in `src/finearm/arms/panthera_ht/` (code = new capability,
configs = tunable facts).
