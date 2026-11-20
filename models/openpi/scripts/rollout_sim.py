"""闭环 rollout 判别实验:策略自己喂自己,从不同起点出发是否收敛到同一终点。

判别两种假设:
  A. 策略漏斗  -> 不同起点终点几乎相同(训练/数据问题,BC 复合误差)
  B. 部署 state 坏 -> 离线闭环从不同起点终点不同(问题在客户端发的 state)

rollout: s <- a[0](瞬时跟踪假设,绝对关节目标), 图像用该集 t0 帧固定
(已证明策略不看图,图像内容不影响结论)。
另附"state 卡死"模拟:首 chunk 的末端构型(客户端若发常量 state,机器人收到的就是它)。
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time

import numpy as np

CKPT = "checkpoints/pi05_panthera/panthera_v2/20000"
N_STEPS = 70


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_panthera")
    print("[rollout] loading policy (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"[rollout] loaded {time.time()-t0:.0f}s", flush=True)

    ds = LeRobotDataset("panthera_bolt", root="../data/panthera_bolt", video_backend="pyav")
    prompt = "pick up the bolt and place it on the nut"

    def parse(item, key):
        im = np.asarray(item[key])
        if im.dtype != np.uint8:
            im = (255 * np.clip(im, 0, 1)).astype(np.uint8)
        if im.shape[0] == 3:
            im = np.moveaxis(im, 0, -1)
        return im

    def ep_start(ep):
        item = ds[int(ds.meta.episodes[ep]["dataset_from_index"])]
        s14 = np.asarray(item["observation.state"])
        return s14, parse(item, "observation.images.front"), parse(item, "observation.images.wrist")

    def infer_chunk(s14, tf, tw):
        obs = {"observation/image": tf, "observation/wrist_image": tw,
               "observation/state": s14, "prompt": prompt}
        return policy.infer(obs)["actions"]  # (30,7)

    def rollout(name, s14, tf, tw):
        s7 = s14[:7].copy()
        print(f"\n[rollout] {name} 起点 s7={np.round(s7,3).tolist()}", flush=True)
        t0 = time.time()
        for t in range(N_STEPS):
            s14_feed = np.concatenate([s7, np.zeros(7)])
            chunk = infer_chunk(s14_feed, tf, tw)
            s7 = chunk[0]  # 瞬时跟踪:state <- 首步绝对目标
            if t % 10 == 0 or t == N_STEPS - 1:
                print(f"  t{t:3d} ({t/30:.1f}s) s7={np.round(s7,3).tolist()} ({(time.time()-t0)/(t+1):.1f}s/步)",
                      flush=True)
        return s7

    # 三个起点:ep0 起点 / ep0 起点扰动joint1+0.15rad(=换初始位姿) / ep60 起点(不同集)
    sA, fA, wA = ep_start(0)
    sB = sA.copy(); sB[1] += 0.15
    sC, fC, wC = ep_start(60)

    eA = rollout("A: ep0 起点", sA, fA, wA)
    eB = rollout("B: ep0 起点扰动 j2+0.15", sB, fA, wA)
    eC = rollout("C: ep60 起点", sC, fC, wC)

    # state 卡死模拟:客户端若一直发 ep0 起点 state,首个 chunk 的轨迹终点
    chunk = infer_chunk(sA, fA, wA)
    print(f"\n[state卡死] 首 chunk 30 步轨迹末端: {np.round(chunk[-1][:7],3).tolist()}", flush=True)

    print("\n===== 终点对比 (前6关节) =====", flush=True)
    print(f"  ‖A - B‖ = {np.linalg.norm(eA[:6]-eB[:6]):.4f} rad  (同起点+扰动,小=漏斗/大=状态敏感)", flush=True)
    print(f"  ‖A - C‖ = {np.linalg.norm(eA[:6]-eC[:6]):.4f} rad  (不同集起点,应大;小=漏斗)", flush=True)
    print(f"  A 终点: {np.round(eA,3).tolist()}", flush=True)
    print(f"  B 终点: {np.round(eB,3).tolist()}", flush=True)
    print(f"  C 终点: {np.round(eC,3).tolist()}", flush=True)
    print("  注: 若真机最终停在某个关节构型,可直接与上面候选构型对号", flush=True)


if __name__ == "__main__":
    main()
