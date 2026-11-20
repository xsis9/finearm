#!/usr/bin/env python3
"""RTT bench: execution machine ↔ serve_rlt (Phase 3, PLAN.md §八 P3).

A plain openpi-client websocket client that hammers the server with
panthera-shaped observations and reports round-trip latency percentiles, plus
the server-side timings reported in each response. 30 fps control needs
p99 < 50 ms (PLAN.md §七).

Runs anywhere with `openpi_client` installed (the execution machine has it —
pure python). Random obs by default (shapes/values do not affect latency:
images resize to the same token grid, prompt is fixed → same jitted shapes);
use --image-size/--state-dim to match the real camera.

Usage (execution machine, or A100 localhost smoke):
  python3 scripts/bench_rtt.py --host <4090-ip> --port 8000 --requests 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_PROMPT = "pick up the bolt and place it on the nut"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--image-size", type=int, nargs=2, default=(480, 640),
                        help="H W of the synthetic cameras (default 480 640, like the dataset).")
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--save", default="", help="Optional JSON results path.")
    return parser.parse_args()


def make_obs(args, rng):
    import numpy as np

    h, w = args.image_size
    return {
        "observation/state": rng.random(args.state_dim).astype(np.float32),
        "observation/image": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "prompt": args.prompt,
    }


def main() -> None:
    args = parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    print(f"connected | server metadata: "
          f"{ {k: meta.get(k) for k in ('server', 'config_name', 'stage1_step', 'fp16_roundtrip')} }")

    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(args.warmup):
        client.infer(make_obs(args, rng))

    rtts, server_infer, server_z = [], [], []
    for _ in range(args.requests):
        obs = make_obs(args, rng)
        t0 = time.monotonic()
        out = client.infer(obs)
        rtts.append((time.monotonic() - t0) * 1000)
        timing = out.get("policy_timing", {})
        if "infer_ms" in timing:
            server_infer.append(float(timing["infer_ms"]))
        if "z_rl_ms" in timing:
            server_z.append(float(timing["z_rl_ms"]))
        if "z_rl" not in out or "actions" not in out:
            raise ValueError(f"response missing keys: {list(out)}")

    def stats(xs):
        xs = np.asarray(xs)
        return {k: round(float(v), 2) for k, v in {
            "mean": xs.mean(), "p50": np.percentile(xs, 50), "p95": np.percentile(xs, 95),
            "p99": np.percentile(xs, 99), "max": xs.max(),
        }.items()}

    results = {
        "host": args.host, "port": args.port, "requests": args.requests,
        "rtt_ms": stats(rtts),
        "server_infer_ms": stats(server_infer) if server_infer else None,
        "server_z_rl_ms": stats(server_z) if server_z else None,
        "actions_shape": list(out["actions"].shape),
        "z_rl_shape": list(out["z_rl"].shape),
        "p99_ok": stats(rtts)["p99"] < 50.0,
    }
    print(json.dumps(results, indent=2))
    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
