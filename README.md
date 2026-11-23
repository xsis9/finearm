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
| Arm daemon | `envs/arm/` (or any venv with the vendor SDK) | [`docs/daemon.md`](docs/daemon.md) |
| Collect (teleop) | `envs/collect/` (LeRobot 0.5.1) | [`docs/collect.md`](docs/collect.md) · operator manual [`quest2ros_setup.md`](plugins/panthera-lerobot-collect/docs/quest2ros_setup.md) |
| Fine-tune (SFT) | GPU box (`uv sync`) | [`docs/sft.md`](docs/sft.md) |
| Serve | GPU box (≥8 GB VRAM for π0.5 bf16) | [`docs/infer.md`](docs/infer.md) |
| Infer | `envs/infer/` on the executor box | safety ladder in [`docs/infer.md`](docs/infer.md) |
| RL post-training | `envs/infer/` (+ torch) | [`docs/rl.md`](docs/rl.md) |

## Adding a new arm

Do these in order — every step has an existing template to copy from:

1. **Daemon plugin**: create `src/finearm/arms/<arm>/` declaring `ARM_ID`,
   `PROFILE` (dims/limits/feature names), `GAINS`, and
   `make_pair`/`make_follower_only`/`make_mock_pair` (vendor-SDK adapters).
   Copy `mock_twojoint/` for the skeleton, `panthera_ht/` for a real one.
   Dropped in, it is auto-discovered; bench-check with
   `finearm-daemon --arm <arm> --mock`.
2. **Collection plugins**: model two packages on
   `plugins/panthera-lerobot-collect/`'s `lerobot_robot_panthera_ht`
   (follower) and `lerobot_teleoperator_panthera_ht` (leader) — the
   distribution name MUST equal the plugin package name.
   Collect via `finearm-collect --method teleop --arm <arm>` (`--arm` is
   required; the default is panthera).
3. **Inference backend**: write your arm's version of
   `src/finearm/infer/robot_interface.py` and point `PANTHERA_ACQ_IFACE`
   at it (or set `acquisition_interface` in `configs/local.jsonc`).
4. **Training config**: follow the four steps in [`docs/sft_zh.md`](docs/sft_zh.md)
   (policy adapter → DataConfig → CONFIGS entry → train).
5. **Camera profile**: if the new arm's cameras differ from panthera's,
   write your own parameter-lock script and point `PANTHERA_CAM_LOCK` at it.

> ⚠️ **Watch out — these mistakes never raise an error; they just silently
> make the policy worse:**
>
> - **Joint order**: collection / training / inference must number and
>   direction-match every joint identically;
> - **Gripper range**: the normalization interval (e.g. 0~1.6) identical
>   in all three places;
> - **Home pose**: collection and inference must use the same home
>   (the dataset's start pose);
> - **Same-distribution chain**: identical camera params at training and
>   inference (lock script), norm-stats from the same dataset, `--task`
>   verbatim equal to the training prompt.

## License

MIT — see [LICENSE](LICENSE).
