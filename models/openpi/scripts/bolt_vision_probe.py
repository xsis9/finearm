"""定位诊断:固定 state,只换不同 episode 的图像(螺栓位置随机),
看模型预测是否随图像变化并指向该 episode 的真实目标。

判据:
  d_match  = ||pred(配对图像) - GT||   (模型用图定位螺栓的能力)
  d_cross  = ||pred(错配图像) - GT||   (换成别的螺栓位置的图)
  spread_pred vs spread_GT             (预测随图变化的幅度 vs 真实目标差异)
若 d_match << d_cross → 视觉定位学到了 → 真机问题是部署侧(输入偏移/反馈/时序)。
若 d_match ≈ d_cross → 模型没用图定位 → 训练侧欠拟合(64集不够/训练不足)。
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import numpy as np


def parse_image(x):
    im = np.asarray(x)
    if im.dtype != np.uint8:
        im = (255 * np.clip(im, 0, 1)).astype(np.uint8)
    if im.shape[0] == 3:
        im = np.moveaxis(im, 0, -1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pi05_panthera_p3/panthera_p3/2000")
    ap.add_argument("--config", default="pi05_panthera_p3")
    ap.add_argument("--repo-id", default="panthera_bolt_p3_val")
    ap.add_argument("--t", type=int, default=10, help="每条 episode 取第 t 帧")
    ap.add_argument("--own-state", action="store_true", help="用各 episode 自己的 state(中途相位诊断)")
    ap.add_argument("--save-frames", default=None, help="存样例帧到此目录(front/wrist PNG)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    horizon = cfg.model.action_horizon
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)

    ds = LeRobotDataset(args.repo_id, delta_timestamps={"action": [i / 30.0 for i in range(horizon)]}, video_backend="pyav")
    tasks = {int(v): k for k, v in ds.meta.tasks["task_index"].items()}
    num_eps = len(ds.meta.episodes)

    obs_list, gt_list = [], []
    for ep in range(num_eps):
        idx = int(ds.meta.episodes[ep]["dataset_from_index"]) + args.t
        item = ds[idx]
        obs_list.append({
            "observation/image": parse_image(item["observation.images.front"]),
            "observation/wrist_image": parse_image(item["observation.images.wrist"]),
            "observation/state": np.asarray(item["observation.state"]),
            "prompt": tasks[int(item["task_index"])],
        })
        gt_list.append(np.asarray(item["action"])[:horizon])
        print(f"[probe] ep{ep:02d} 取帧 {args.t} done", flush=True)

    # 统一 state: 用所有 episode 第 t 帧的平均(消除 state 差异,只留图像变量); --own-state 时保留各自 state
    state_mean = np.mean([o["observation/state"] for o in obs_list], axis=0)

    if args.save_frames:
        import PIL.Image
        os.makedirs(args.save_frames, exist_ok=True)
        for i in [0, E0 if (E0 := len(obs_list)) > 0 else 0][:1]:
            PIL.Image.fromarray(obs_list[i]["observation/image"]).save(f"{args.save_frames}/ep{i}_front_t{args.t}.png")
            PIL.Image.fromarray(obs_list[i]["observation/wrist_image"]).save(f"{args.save_frames}/ep{i}_wrist_t{args.t}.png")
        print(f"[probe] 样例帧已存 {args.save_frames}", flush=True)

    preds = []
    for i, o in enumerate(obs_list):
        o = dict(o)
        if not args.own_state:
            o["observation/state"] = state_mean
        pred = np.asarray(policy.infer(o)["actions"])  # (horizon,7) 绝对目标
        preds.append(pred)
        print(f"[probe] infer ep{i:02d}: pred[0,:6]={np.round(pred[0,:6],3)}", flush=True)
    preds = np.array(preds)  # (E,horizon,7)
    gts = np.array(gt_list)

    def adist(x, y):  # 前6关节,首步与整段两种口径
        return np.linalg.norm(x[:6] - y[:6]), np.mean(np.linalg.norm(x[:, :6] - y[:, :6], axis=1))

    E = len(preds)
    d_match_first, d_match_chunk, d_cross_first, d_cross_chunk = [], [], [], []
    for e in range(E):
        f, c = adist(preds[e], gts[e]); d_match_first.append(f); d_match_chunk.append(c)
        others = [o for o in range(E) if o != e]
        fs, cs = [], []
        for o in others:
            f, c = adist(preds[o], gts[e]); fs.append(f); cs.append(c)
        d_cross_first.append(np.mean(fs)); d_cross_chunk.append(np.mean(cs))

    spread_pred = np.mean([np.linalg.norm(preds[e][0, :6] - np.mean(preds[:, 0, :6], axis=0)) for e in range(E)])
    spread_gt = np.mean([np.linalg.norm(gts[e][0, :6] - np.mean(gts[:, 0, :6], axis=0)) for e in range(E)])

    print("\n======== 诊断结果(rad) ========")
    print(f"配对图像  d_match: first={np.mean(d_match_first):.4f}  chunk={np.mean(d_match_chunk):.4f}")
    print(f"错配图像  d_cross: first={np.mean(d_cross_first):.4f}  chunk={np.mean(d_cross_chunk):.4f}")
    print(f"预测跨episode 散布 spread_pred(first)={spread_pred:.4f}")
    print(f"真值跨episode 散布 spread_GT  (first)={spread_gt:.4f}")
    print(f"视觉增益 = d_cross - d_match: first={np.mean(d_cross_first)-np.mean(d_match_first):+.4f} chunk={np.mean(d_cross_chunk)-np.mean(d_match_chunk):+.4f}")
    print("解读: 增益显著>0且spread_pred接近spread_GT → 视觉定位有效(部署侧问题);")
    print("      增益≈0且spread_pred<<spread_GT → 模型没用图像定位(训练侧欠拟合)")


if __name__ == "__main__":
    main()
