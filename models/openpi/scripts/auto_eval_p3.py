"""panthera_p3 checkpoint 自动评估:每个 checkpoint 在验证集上跑三项检查(CPU 推理,不占训练 GPU)。

  ① held-out 回放  : 真实示教帧上预测 vs 示教动作(err_first/err_chunk + hold 基线)
  ② 黑图敏感性     : 双路相机涂黑 vs 原图的 Δaction(策略是否真的在看图)
  ③ 闭环 rollout   : 自喂 state 走 N 步,终点散布 + 从起点的最大偏移(是否发散)

输出: 一行 JSON 追加到 --out。flags 里所有阈值均为"临时",用 step 1000 的结果校准后再收紧。

用法:
  cd <openpi-tree> && uv run python scripts/auto_eval_p3.py \
      --ckpt checkpoints/panthera_p3/1000 --step 1000
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 必须在 import jax 之前

import argparse
import datetime as _dt
import json
import time

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, default=None, help="记录用;缺省从 ckpt 路径推断")
    ap.add_argument("--config", default="pi05_panthera_p3")
    ap.add_argument("--repo-id", default="panthera_bolt_p3_val", help="评估数据集(默认验证集),与训练 repo 解耦")
    ap.add_argument("--data-root", default=None,
                    help="数据集目录本身(非父目录!);缺省走 ~/.cache/huggingface/lerobot/{repo_id} 软链")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--frames-per-ep", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=70)
    ap.add_argument("--rollout-perturb", type=float, default=0.15)
    ap.add_argument("--out", default="checkpoints/pi05_panthera_p3/panthera_p3/auto_eval/results.jsonl")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    step = args.step if args.step is not None else int(os.path.basename(os.path.abspath(args.ckpt)))
    cfg = _config.get_config(args.config)
    horizon = cfg.model.action_horizon

    print(f"[auto_eval] step={step} loading policy from {args.ckpt} (CPU)...", flush=True)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    print(f"[auto_eval] policy loaded in {time.time() - t0:.0f}s", flush=True)

    root_kwargs = {"root": args.data_root} if args.data_root else {}
    meta = LeRobotDatasetMetadata(args.repo_id, **root_kwargs)
    ds = LeRobotDataset(
        args.repo_id,
        delta_timestamps={"action": [t / meta.fps for t in range(horizon)]},
        video_backend="pyav",
        **root_kwargs,
    )
    tasks = {int(v): k for k, v in meta.tasks["task_index"].items()}

    rng = np.random.default_rng(args.seed)
    num_eps = len(ds.meta.episodes)
    ep_ids = sorted(rng.choice(num_eps, size=min(args.num_episodes, num_eps), replace=False).tolist())
    print(f"[auto_eval] repo={args.repo_id} eps={ep_ids} horizon={horizon} len={len(ds)}", flush=True)

    def infer(front, wrist, state14, prompt):
        obs = {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": state14,
            "prompt": prompt,
        }
        return np.asarray(policy.infer(obs)["actions"])  # (horizon, 7) 绝对目标

    # ---------- ① 回放 + ② 黑图敏感性(同一批帧,省一遍视频解码) ----------
    rep = {"first_arm": [], "first_grip": [], "chunk_arm": [], "chunk_grip": [],
           "hold_first_arm": [], "hold_chunk_arm": []}
    vis = {"black_first_arm": [], "black_first_grip": [], "black_chunk_arm": []}
    for ep in ep_ids:
        ep_len = ds.meta.episodes[ep]["length"]
        ts = np.linspace(5, max(6, ep_len - horizon - 5), args.frames_per_ep).astype(int)
        for t in ts:
            idx = int(ds.meta.episodes[ep]["dataset_from_index"]) + int(t)
            item = ds[idx]
            state14 = np.asarray(item["observation.state"])
            front = parse_image(item["observation.images.front"])
            wrist = parse_image(item["observation.images.wrist"])
            prompt = tasks[int(item["task_index"])]
            gt = np.asarray(item["action"])[:horizon]
            state7 = state14[:7]

            pred = infer(front, wrist, state14, prompt)
            err = pred - gt
            hold = state7[None, :] - gt

            rep["first_arm"].append(float(np.linalg.norm(err[0, :6])))
            rep["first_grip"].append(float(abs(err[0, 6])))
            rep["chunk_arm"].append(float(np.mean(np.linalg.norm(err[:, :6], axis=1))))
            rep["chunk_grip"].append(float(np.mean(np.abs(err[:, 6]))))
            rep["hold_first_arm"].append(float(np.linalg.norm(hold[0, :6])))
            rep["hold_chunk_arm"].append(float(np.mean(np.linalg.norm(hold[:, :6], axis=1))))

            black = infer(np.zeros_like(front), np.zeros_like(wrist), state14, prompt)
            d = black - pred
            vis["black_first_arm"].append(float(np.linalg.norm(d[0, :6])))
            vis["black_first_grip"].append(float(abs(d[0, 6])))
            vis["black_chunk_arm"].append(float(np.mean(np.linalg.norm(d[:, :6], axis=1))))

            print(
                f"[auto_eval] ep{ep:03d} t{int(t):04d} | replay first {rep['first_arm'][-1]:.4f}rad"
                f" (hold {rep['hold_first_arm'][-1]:.4f}) | black-Δ first {vis['black_first_arm'][-1]:.4f}rad",
                flush=True,
            )

    m = lambda k: float(np.mean(rep[k]))
    vm = lambda k: float(np.mean(vis[k]))
    replay_summary = {k: round(m(k), 5) for k in rep}
    replay_summary["ratio_first"] = round(m("first_arm") / max(m("hold_first_arm"), 1e-9), 3)
    replay_summary["ratio_chunk"] = round(m("chunk_arm") / max(m("hold_chunk_arm"), 1e-9), 3)
    vision_summary = {k: round(vm(k), 5) for k in vis}

    # ---------- ③ 闭环 rollout(策略自喂 state,瞬时跟踪假设) ----------
    def ep_start(ep):
        item = ds[int(ds.meta.episodes[ep]["dataset_from_index"])]
        return (
            np.asarray(item["observation.state"]),
            parse_image(item["observation.images.front"]),
            parse_image(item["observation.images.wrist"]),
            tasks[int(item["task_index"])],
        )

    def rollout(s14, front, wrist, prompt, tag=""):
        s7 = np.asarray(s14)[:7].copy()
        start = s7.copy()
        max_exc = 0.0
        for i in range(args.rollout_steps):
            chunk = infer(front, wrist, np.concatenate([s7, np.zeros(7)]), prompt)
            s7 = np.asarray(chunk[0])[:7].copy()  # 瞬时跟踪: state <- 首步绝对目标
            max_exc = max(max_exc, float(np.max(np.abs(s7[:6] - start[:6]))))
            if (i + 1) % 10 == 0:  # 每 10 步打印进度,便于实时观测
                print(f"[auto_eval] rollout {tag} {i + 1}/{args.rollout_steps} max_exc={max_exc:.3f}", flush=True)
        return s7, max_exc

    ep_a, ep_c = ep_ids[0], ep_ids[-1]
    s_a, f_a, w_a, p_a = ep_start(ep_a)
    s_b = s_a.copy()
    s_b[1] += args.rollout_perturb
    s_c, f_c, w_c, p_c = ep_start(ep_c)

    print(f"[auto_eval] rollout: A=ep{ep_a}, B=A扰动j2+{args.rollout_perturb}rad, C=ep{ep_c},"
          f" {args.rollout_steps}步 x3 ...", flush=True)
    t0 = time.time()
    e_a, exc_a = rollout(s_a, f_a, w_a, p_a, tag="A")
    e_b, _ = rollout(s_b, f_a, w_a, p_a, tag="B")
    e_c, exc_c = rollout(s_c, f_c, w_c, p_c, tag="C")
    rollout_summary = {
        "final_AB": round(float(np.linalg.norm(e_a[:6] - e_b[:6])), 4),
        "final_AC": round(float(np.linalg.norm(e_a[:6] - e_c[:6])), 4),
        "max_excursion_A": round(exc_a, 4),
        "max_excursion_C": round(exc_c, 4),
        "final_A": [round(float(x), 3) for x in e_a[:6]],
        "final_B": [round(float(x), 3) for x in e_b[:6]],
        "final_C": [round(float(x), 3) for x in e_c[:6]],
    }
    print(f"[auto_eval] rollout done in {time.time() - t0:.0f}s", flush=True)

    # ---------- 临时通过标记(用 step 1000 的结果校准后收紧) ----------
    flags = {
        "replay_learn": replay_summary["ratio_first"] < 0.5,
        "vision_black": vision_summary["black_first_arm"] > 0.05,
        "rollout_no_runaway": rollout_summary["max_excursion_A"] < 1.0,
        "rollout_state_sensitive": rollout_summary["final_AB"] > 0.1,
        "_note": "临时阈值,待 step1000 校准",
    }

    record = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "ckpt": os.path.abspath(args.ckpt),
        "config": args.config,
        "repo_id": args.repo_id,
        "n_frames": len(rep["first_arm"]),
        "replay": replay_summary,
        "vision": vision_summary,
        "rollout": rollout_summary,
        "flags": flags,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n[auto_eval] ===== 汇总 =====", flush=True)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
