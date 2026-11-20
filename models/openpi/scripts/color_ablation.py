"""颜色通道/亮度消融:检测推理客户端若把 BGR 当 RGB 发送,策略输出会偏多少。

t=0 / 20 / 60 三时刻:基准(train 原图) vs R/B 通道互换 vs 亮度±30%。
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time

import numpy as np

CKPT = "checkpoints/pi05_panthera/panthera_v2/20000"


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_panthera")
    print("[color] loading policy (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"[color] loaded {time.time()-t0:.0f}s", flush=True)

    ds = LeRobotDataset("panthera_bolt", root="../data/panthera_bolt", video_backend="pyav")
    idx0 = int(ds.meta.episodes[0]["dataset_from_index"])
    prompt = "pick up the bolt and place it on the nut"

    def parse(item, key):
        im = np.asarray(item[key])
        if im.dtype != np.uint8:
            im = (255 * np.clip(im, 0, 1)).astype(np.uint8)
        if im.shape[0] == 3:
            im = np.moveaxis(im, 0, -1)
        return im

    for t in (0, 20, 60):
        item = ds[idx0 + t]
        state14 = np.asarray(item["observation.state"])
        tf, tw = parse(item, "observation.images.front"), parse(item, "observation.images.wrist")

        def infer(fi, wi):
            obs = {"observation/image": fi, "observation/wrist_image": wi,
                   "observation/state": state14, "prompt": prompt}
            return policy.infer(obs)["actions"]

        def swap_rb(im):
            return im[..., ::-1].copy()

        def bright(im, k):
            return np.clip(im.astype(int) * k, 0, 255).astype(np.uint8)

        rows = [
            ("基准 原图", tf, tw),
            ("R/B 互换(两路)", swap_rb(tf), swap_rb(tw)),
            ("仅 front R/B 互换", swap_rb(tf), tw),
            ("亮度 x1.3", bright(tf, 1.3), bright(tw, 1.3)),
            ("亮度 x0.7", bright(tf, 0.7), bright(tw, 0.7)),
        ]
        print(f"\n[color] ep0 t={t}", flush=True)
        base = base_grip = None
        for name, fi, wi in rows:
            pred = infer(fi, wi)
            a = pred[0, :6]
            if base is None:
                base, base_grip = a, pred[0, 6]
            d = float(np.linalg.norm(a - base))
            print(f"  {name:20s} Δ vs 基准 = {d:.4f}rad  Δgrip {abs(pred[0,6]-base_grip):.3f}", flush=True)
    print("  尺度: Δ > 0.05rad 为实质性改变", flush=True)


if __name__ == "__main__":
    main()
