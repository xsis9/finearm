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
| `configs/` | Machine/arm/model configuration (canonical copies) |
| `third_party/` | Upstream baseline records for vendored dependencies |
| `doc/` | Working notes and hardware manuals |

## Testing

```bash
# RL suite (requires: torch, numpy, pytest)
cd RL/rlt_panthera && python -m pytest

# Data-collection suite (requires: lerobot==0.5.1, pytest)
cd data/collect/panthera-lerobot-collect && python -m pytest tests/
```

CI runs lint plus both suites on every push (`.github/workflows/ci.yml`).
Hardware is never touched by CI or by the test suites.

## License

MIT — see [LICENSE](LICENSE).
