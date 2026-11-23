# envs/collect · collection environment

The LeRobot **0.5.1** venv (`.venv-lerobot/`) used by the collection
stack. Pinned dependencies: [`requirements-lerobot.txt`](requirements-lerobot.txt)
in this directory.

One entry point builds it (also clones/verifies the vendor SDK and
installs the plugins):

```bash
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --install
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --check
```

Repo rule: every venv lives under `envs/<domain>/` — find the
environment without knowing which code uses it. The venv is NOT
relocatable; after moving this directory, rebuild via the entry point
above.
