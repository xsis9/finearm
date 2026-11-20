"""相机扰动敏感性离线测试:量化 front / wrist 图像偏移多少会让动作崩掉。

用途:回答"推理时腕相机被挪动了几厘米/几度,是不是真机失败主因"。
做法:取训练集示教帧,baseline 推理后,对某一路相机图像施加平移/旋转/全黑,
     测量首步执行动作的变化 ‖Δpred[:6]‖(rad),并以"示教动作本身的大小"为尺度参照。

用法(CPU,GPU 被训练占用):
  cd <openpi-tree> && uv run python scripts/perturb_panthera_ckpt.py \
      --ckpt checkpoints/pi05_panthera/panthera_v2/20000
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 必须在 import jax 之前

import argparse
import time

import cv2
import numpy as np


def shift_img(img: np.ndarray, dx: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0))


def rot_img(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pi05_panthera/panthera_v2/20000")
    ap.add_argument("--config", default="pi05_panthera")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--data-root", default="../data/panthera_bolt")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    print(f"[perturb] loading policy from {args.ckpt} (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    print(f"[perturb] policy loaded in {time.time()-t0:.0f}s", flush=True)

    meta = LeRobotDatasetMetadata(cfg.data.repo_id, root=args.data_root)
    ds = LeRobotDataset(cfg.data.repo_id, video_backend="pyav", root=args.data_root)
    tasks = {int(v): k for k, v in meta.tasks["task_index"].items()}

    ep_len = ds.meta.episodes[args.episode]["length"]
    idx0 = int(ds.meta.episodes[args.episode]["dataset_from_index"])
    # 取接近抓取的两个时刻(40% / 65%),最能反映视觉定位
    frames = [idx0 + int(ep_len * f) for f in (0.40, 0.65)]

    # 扰动表: (名称, 对哪路相机, 变换函数)
    w, f = "wrist", "front"
    perts = [
        (f"{w} shift +4px",  w, lambda im: shift_img(im, 4)),
        (f"{w} shift +8px",  w, lambda im: shift_img(im, 8)),
        (f"{w} shift +16px", w, lambda im: shift_img(im, 16)),
        (f"{w} shift +32px", w, lambda im: shift_img(im, 32)),
        (f"{w} rot 2deg",    w, lambda im: rot_img(im, 2)),
        (f"{w} rot 5deg",    w, lambda im: rot_img(im, 5)),
        (f"{w} rot 10deg",   w, lambda im: rot_img(im, 10)),
        (f"{w} 全黑",         w, lambda im: np.zeros_like(im)),
        (f"{f} shift +8px",  f, lambda im: shift_img(im, 8)),
        (f"{f} shift +16px", f, lambda im: shift_img(im, 16)),
        (f"{f} shift +32px", f, lambda im: shift_img(im, 32)),
        (f"{f} rot 5deg",    f, lambda im: rot_img(im, 5)),
        (f"{f} rot 10deg",   f, lambda im: rot_img(im, 10)),
        (f"{f} 全黑",         f, lambda im: np.zeros_like(im)),
    ]

    all_rows = []
    for idx in frames:
        item = ds[idx]
        state14 = np.asarray(item["observation.state"])
        front = np.asarray(item["observation.images.front"])
        wrist = np.asarray(item["observation.images.wrist"])
        if front.dtype != np.uint8:  # parse 成 HWC uint8(与 PantheraInputs._parse_image 一致)
            front = (255 * np.clip(front, 0, 1)).astype(np.uint8)
            front = np.moveaxis(front, 0, -1) if front.shape[0] == 3 else front
            wrist = (255 * np.clip(wrist, 0, 1)).astype(np.uint8)
            wrist = np.moveaxis(wrist, 0, -1) if wrist.shape[0] == 3 else wrist
        prompt = tasks[int(item["task_index"])]
        gt0 = np.asarray(item["action"])[:6]
        state7 = state14[:7]

        def infer(front_img, wrist_img, pmt=prompt):
            obs = {
                "observation/image": front_img,
                "observation/wrist_image": wrist_img,
                "observation/state": state14,
                "prompt": pmt,
            }
            return policy.infer(obs)["actions"]  # (horizon,7)

        t1 = time.time()
        base = infer(front, wrist)
        base_arm = base[0, :6]
        # 尺度参照: 示教首步动作幅度 & 预测相对当前状态的运动幅度
        gt_mag = float(np.linalg.norm(gt0 - state7[:6]))
        pred_mag = float(np.linalg.norm(base_arm - state7[:6]))
        print(f"\n[perturb] ep{args.episode} frame idx {idx} (t={idx-idx0}) prompt={prompt!r}", flush=True)
        print(f"  baseline: |gt首步|={gt_mag:.4f}rad |pred首步-状态|={pred_mag:.4f}rad "
              f"pred0={np.round(base_arm,3).tolist()}", flush=True)

        rows = []
        for name, which, fn in perts + [("prompt 改错(螺丝)", None, None)]:
            pmt = prompt
            if which == w:
                fi, wi = front, fn(wrist)
            elif which == f:
                fi, wi = fn(front), wrist
            else:  # prompt ablation
                fi, wi = front, wrist
                pmt = "pick up the screw and place it on the nut"
            pred = infer(fi, wi, pmt)
            d_arm = float(np.linalg.norm(pred[0, :6] - base_arm))
            d_grip = float(abs(pred[0, 6] - base[0, 6]))
            rows.append((name, d_arm, d_grip))
            print(f"  {name:22s} -> Δ首步 arm {d_arm:.4f}rad  grip {d_grip:.3f}", flush=True)
        all_rows.append((gt_mag, pred_mag, rows))

    print("\n===== 汇总(两帧平均) =====", flush=True)
    print(f"  尺度参照: |示教首步|≈{np.mean([r[0] for r in all_rows]):.4f}rad, "
          f"|pred-状态|≈{np.mean([r[1] for r in all_rows]):.4f}rad", flush=True)
    names = [r[2][0][0] for r in all_rows[:1]]
    for i, name in enumerate(names):
        d = np.mean([r[2][i][1] for r in all_rows])
        dg = np.mean([r[2][i][2] for r in all_rows])
        print(f"  {name:22s} Δ首步 arm {d:.4f}rad  grip {dg:.3f}", flush=True)
    print("  注: 图像 640px 宽,policy 内缩到 224px → 16px≈缩后5.6px;"
          "60-70°hFOV 下 16px≈1.5-1.9°视角≈30cm深处约0.8-1cm视差", flush=True)


if __name__ == "__main__":
    main()
