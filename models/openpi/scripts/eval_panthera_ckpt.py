"""离线验证 panthera checkpoint:在真实示教帧上跑策略,与示教动作对比。

用法(GPU 被训练占用,强制 CPU 推理):
  cd <openpi-tree> && uv run python scripts/eval_panthera_ckpt.py \
      --ckpt checkpoints/pi05_panthera/panthera_v1/10000

指标:
  - err_first : 预测第 1 步动作 vs 示教第 1 步(部署时真正执行的那步)的 L2
  - err_chunk : 30 步 chunk 平均 L2
  - hold 基线: "保持当前关节角不动"(delta=0)的对应误差,模型必须显著小于它才算学到了东西
  - 物理单位: 前 6 维 rad,夹爪 0~1
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 必须在 import jax 之前

import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pi05_panthera/panthera_v1/10000")
    ap.add_argument("--config", default="pi05_panthera")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--frames-per-ep", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="../data/panthera_bolt")
    ap.add_argument(
        "--episodes-file",
        default=None,
        help="split json (如 panthera_bolt_split.json):给定则只用其中的 val_episodes 测试",
    )
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    horizon = cfg.model.action_horizon

    print(f"[eval] loading policy from {args.ckpt} (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    print(f"[eval] policy loaded in {time.time()-t0:.0f}s", flush=True)

    root_kwargs = {"root": args.data_root}
    episodes_kwargs = {}
    if args.episodes_file:
        split = json.loads(open(args.episodes_file).read())
        episodes_kwargs["episodes"] = split["val_episodes"]
    meta = LeRobotDatasetMetadata(cfg.data.repo_id, **root_kwargs)
    ds = LeRobotDataset(
        cfg.data.repo_id,
        delta_timestamps={"action": [t / meta.fps for t in range(horizon)]},
        video_backend="pyav",
        **root_kwargs,
        **episodes_kwargs,
    )
    tasks = {int(v): k for k, v in meta.tasks["task_index"].items()}

    rng = np.random.default_rng(args.seed)
    if args.episodes_file:
        ep_ids = split["val_episodes"]
        if args.num_episodes < len(ep_ids):
            ep_ids = sorted(rng.choice(ep_ids, size=args.num_episodes, replace=False).tolist())
    else:
        num_eps = len(ds.meta.episodes)
        ep_ids = sorted(rng.choice(num_eps, size=args.num_episodes, replace=False).tolist())
    print(f"[eval] dataset: @ {meta.fps}fps, horizon={horizon}; sampling eps {ep_ids} (len={len(ds)})", flush=True)

    rows = []
    per_dim_abs_err = []
    for ep in ep_ids:
        ep_len = ds.meta.episodes[ep]["length"]
        ts = np.linspace(5, max(6, ep_len - horizon - 5), args.frames_per_ep).astype(int)
        for t in ts:
            idx = int(ds.meta.episodes[ep]["dataset_from_index"]) + int(t)
            item = ds[idx]
            state14 = np.asarray(item["observation.state"])
            obs = {
                "observation/image": np.asarray(item["observation.images.front"]),
                "observation/wrist_image": np.asarray(item["observation.images.wrist"]),
                "observation/state": state14,
                "prompt": tasks[int(item["task_index"])],
            }
            gt = np.asarray(item["action"])[:horizon]  # (horizon, 7) 绝对目标
            t1 = time.time()
            out = policy.infer(obs)
            pred = out["actions"]  # (horizon, 7) 绝对目标(已反归一化、反delta)
            infer_s = time.time() - t1

            state7 = state14[:7]
            err = pred[:horizon] - gt
            hold_err = state7[None, :] - gt  # 基线: 保持不动

            def l2_first(e):
                return float(np.linalg.norm(e[0, :6])), float(abs(e[0, 6]))

            def l2_mean(e):
                return float(np.mean(np.linalg.norm(e[:, :6], axis=1))), float(np.mean(np.abs(e[:, 6])))

            f_arm, f_grip = l2_first(err)
            m_arm, m_grip = l2_mean(err)
            hf_arm, hf_grip = l2_first(hold_err)
            hm_arm, hm_grip = l2_mean(hold_err)
            delta0 = pred[0] - state7

            rows.append((ep, int(t), f_arm, f_grip, m_arm, m_grip, hf_arm, hf_grip, hm_arm, infer_s))
            per_dim_abs_err.append(np.mean(np.abs(err), axis=0))
            print(
                f"[eval] ep{ep:03d} t{int(t):04d} | first-step err: arm {f_arm:.4f}rad grip {f_grip:.3f}"
                f" | chunk-mean: arm {m_arm:.4f} grip {m_grip:.3f}"
                f" | hold-baseline first/chunk: {hf_arm:.4f}/{hm_arm:.4f}rad"
                f" | |delta0| arm {np.linalg.norm(delta0[:6]):.4f}rad | {infer_s:.1f}s/帧",
                flush=True,
            )

    print("\n===== 汇总 =====", flush=True)
    arr = np.array([r[2:11] for r in rows], dtype=float)  # f_arm f_grip m_arm m_grip hf_arm hf_grip hm_arm
    names = [
        "first-step arm (rad)", "first-step grip",
        "chunk-mean arm (rad)", "chunk-mean grip",
        "hold 基线 first arm", "hold 基线 first grip", "hold 基线 chunk arm",
    ]
    for name, v in zip(names, arr.mean(axis=0)):
        print(f"  mean {name:26s} = {v:.4f}", flush=True)
    print(f"  模型/基线 (first-step arm): {arr[:,0].mean()/arr[:,4].mean():.2f}x  (越小越好,显著<1 说明学到了动作)", flush=True)
    print(f"  模型/基线 (chunk arm)    : {arr[:,2].mean()/arr[:,6].mean():.2f}x", flush=True)
    dim = np.mean(per_dim_abs_err, axis=0)
    print("  各维平均|误差| (j1..j6, grip):", " ".join(f"{x:.4f}" for x in dim), flush=True)


if __name__ == "__main__":
    main()
