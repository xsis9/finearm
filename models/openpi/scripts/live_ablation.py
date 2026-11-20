"""用真机现场图 vs 训练图做 2x2 消融:哪路相机的差异真正影响动作输出。

four combos: (train_front, train_wrist) / (live_front, train_wrist) /
             (train_front, live_wrist) / (live_front, live_wrist)
state 用 ep0 初始 state(与两图位姿最接近的可用参照),prompt 用数据集任务文本。
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time

import cv2
import numpy as np

CKPT = "checkpoints/pi05_panthera/panthera_v2/20000"


def bgr_to_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_panthera")
    print("[live] loading policy (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"[live] loaded {time.time()-t0:.0f}s", flush=True)

    ds = LeRobotDataset("panthera_bolt", root="../data/panthera_bolt", video_backend="pyav")
    idx0 = int(ds.meta.episodes[0]["dataset_from_index"])
    prompt = "pick up the bolt and place it on the nut"

    lf = bgr_to_rgb("images/live_front.jpg")
    lw = bgr_to_rgb("images/live_wrist.jpg")

    def parse(item, key):
        im = np.asarray(item[key])
        if im.dtype != np.uint8:
            im = (255 * np.clip(im, 0, 1)).astype(np.uint8)
        if im.shape[0] == 3:
            im = np.moveaxis(im, 0, -1)
        return im

    for t in (0, 20, 60):  # t=0 初始 / ~0.7s / ~2s(伸手初期)
        item = ds[idx0 + t]
        state14 = np.asarray(item["observation.state"])
        tf, tw = parse(item, "observation.images.front"), parse(item, "observation.images.wrist")
        gt0 = np.asarray(item["action"])[:6]

        def infer(fi, wi):
            obs = {"observation/image": fi, "observation/wrist_image": wi,
                   "observation/state": state14, "prompt": prompt}
            return policy.infer(obs)["actions"]

        combos = [
            ("trainF + trainW (基准)", tf, tw),
            ("LIVE_F + trainW", lf, tw),
            ("trainF + LIVE_W", tf, lw),
            ("LIVE_F + LIVE_W", lf, lw),
            ("trainF全黑 + trainW", np.zeros_like(tf), tw),
        ]
        print(f"\n[live] ep0 t={t}  state[:7]={np.round(state14[:7],3).tolist()}", flush=True)
        print(f"  示教首步 |gt-状态| = {float(np.linalg.norm(gt0 - state14[:6])):.4f}rad", flush=True)
        base = None
        for name, fi, wi in combos:
            pred = infer(fi, wi)
            a = pred[0, :6]
            if base is None:
                base = a
            d = float(np.linalg.norm(a - base))
            print(f"  {name:24s} pred0={np.round(a,3).tolist()} grip={pred[0,6]:.3f}"
                  f" | Δ vs 基准 = {d:.4f}rad", flush=True)
    print("  尺度: 示教首步≈0.03-0.14rad; Δ 超过 0.05rad 即为实质性改变", flush=True)


if __name__ == "__main__":
    main()
