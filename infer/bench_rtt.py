#!/usr/bin/env python3
"""通用推理 RTT 基准(按适配器,模型无关):给 RTC 的 k 建议值。

用法:
  .venv/bin/python bench_rtt.py --model pi05    --host 127.0.0.1
  .venv/bin/python bench_rtt.py --model lingbot --host 127.0.0.1
"""
import argparse
import time

import numpy as np

from adapters import load as load_adapter, mock_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--task", default="pick up the bolt and place it on the nut")
    args = ap.parse_args()

    adapter = load_adapter(args.model, host=args.host, port=args.port)
    adapter.reset()
    payload = adapter.build_obs(mock_obs(adapter.OBS_STYLE), args.task)

    rtts = []
    for i in range(args.n):
        t0 = time.perf_counter()
        chunk = adapter.infer(payload)
        rtt = time.perf_counter() - t0
        rtts.append(rtt)
        chunk = np.asarray(chunk)
        print(f"[{i+1}/{args.n}] rtt={rtt*1000:.1f}ms  chunk={chunk.shape}")
    rtts = np.array(rtts[1:])  # 第一条是 jit/compile 预热,剔除
    p50, p95 = np.percentile(rtts, 50) * 1000, np.percentile(rtts, 95) * 1000
    k = int(np.ceil(np.percentile(rtts, 95) * 30)) + 2
    print(f"\n{adapter.NAME}: p50={p50:.0f}ms  p95={p95:.0f}ms  max={rtts.max()*1000:.0f}ms")
    print(f"建议 RTC 参数: --k {k}(自适应模式下客户端也会自动定)")


if __name__ == "__main__":
    main()
