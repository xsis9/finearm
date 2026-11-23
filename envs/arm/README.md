# envs/arm · arm-driver environment

The daemon driver venv (`.venv/`): python 3.12 + the vendor SDK wheel +
its Python deps (pyyaml/scipy/pinocchio). Built by
[`init_env.sh`](init_env.sh) in this directory.

The machine assets it drives (vendor SDK, calibration `robot_param/`,
daemon launcher) stay in [`arm/panthera/`](../../arm/panthera/) — see
[docs/daemon.md](../../docs/daemon.md).
