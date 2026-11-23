# finearm

[English](README.md) · [中文](README_zh.md)

VLA fine-tuning framework: teleop data collection, fine-tuning, inference,
and RL post-training.

## Repository layout

| Path | Contents |
|---|---|
| `arm/panthera/` | Verified arm assets |
| `plugins/panthera-lerobot-collect/` | LeRobot collection stack: robot/teleoperator plugins, launcher wrapper, operator docs |
| `envs/` | All environment definitions + venvs: `infer/` (openpi-client), `collect/` (LeRobot 0.5.1), `arm/` (driver) |
| `RL/rlt_panthera/` | RL run artifacts |
| `src/finearm/core/` | Zero-dependency core: four-layer config resolution, repo paths, plugin registry, norm-stats contract |
| `src/finearm/arms/` | Arms |
| `src/finearm/infer/` | `RTCExecutor` (pipelined) and `StopAndGoExecutor` |
| `src/finearm/rl/` | VLA RL post-training |
| `src/finearm/data/` | Data collection |
| `models/` | Fine-tuning (SFT): vendored modified openpi tree |
| `configs/` | Machine/arm/model configuration (canonical copies) |
| `third_party/` | Upstream baseline records for vendored dependencies |
| `docs/` | Notes |

### Topologies

- **Single machine (local inference)**: one Linux box with a CUDA GPU
  (≥ 8 GB VRAM to serve π0.5 in bf16) runs the inference server, the arm
  daemon, and the collection/RL clients.
- **Split executor + inference server**: the arm-attached executor box runs
  daemon/collection/RL; the GPU box serves the policy.


## The pipeline: collect → fine-tune → serve → infer → RL

| Stage | Environment | Full how-to |
|---|---|---|
| Arm daemon | `envs/arm/` (or any venv with the vendor SDK) | [`arm/panthera/README.md`](arm/panthera/README.md) |
| Collect (teleop) | `envs/collect/` (LeRobot 0.5.1) | [`data/README.md`](data/README.md) · operator manual [`quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md) |
| Fine-tune (SFT) | GPU box (`uv sync`) | [`models/README.md`](models/README.md) |
| Serve | GPU box (≥8 GB VRAM for π0.5 bf16) | [`envs/infer/README.md`](envs/infer/README.md) |
| Infer | `envs/infer/` on the executor box | safety ladder in [`envs/infer/README.md`](envs/infer/README.md) |
| RL post-training | `envs/infer/` (+ torch) | [`RL/rlt_panthera/README.md`](RL/rlt_panthera/README.md) |

## License

MIT — see [LICENSE](LICENSE).
