"""Stop-and-go executor: fresh reference every window (train_stage2 --replan_every_window semantics).

Observe → blocking inference → execute ``n_action_steps`` steps → repeat.
No pipelining: every executed step belongs to a chunk inferred from an
observation taken at the window start, which makes the reference maximally
fresh at the cost of one RTT stall per window. Intended for the actor
segment of the hybrid deployment mode (infer/README.md) and for debugging.
"""

from __future__ import annotations

import time

import numpy as np

from .base import ChunkExecutor, ExecutorStats, clamp_action


class StopAndGoExecutor(ChunkExecutor):
    """Synchronous chunked executor."""

    name = "stop-and-go"

    def __init__(self, adapter, *, n_action_steps=10, ctrl_hz=30.0,
                 pos_limits=None, vel_limits=None, dry_run=False, verbose=True):
        """
        n_action_steps : steps executed per inference; 0 or None = the whole chunk
        """
        super().__init__(adapter, ctrl_hz=ctrl_hz, pos_limits=pos_limits,
                         vel_limits=vel_limits, dry_run=dry_run, verbose=verbose)
        self.n_action_steps = n_action_steps

    def run(self, robot, task, max_steps=None) -> ExecutorStats:
        step = 0
        replans = 0
        rtts = []
        prev_cmd = None
        tick = 1.0 / self.ctrl_hz

        try:
            while max_steps is None or step < max_steps:
                # ① observe → blocking inference (the stall lands at the window start)
                payload = self.adapter.build_obs(
                    robot.get_obs(self.adapter.OBS_STYLE), task)
                t0 = time.perf_counter()
                chunk = np.asarray(self.adapter.infer(payload), dtype=np.float64)
                rtts.append(time.perf_counter() - t0)
                replans += 1
                if replans == 1:
                    print(f"[SAG] 首 chunk 就绪(horizon={len(chunk)} 步,"
                          f" 每窗执行 {self._take(len(chunk))} 步)")

                # ② execute the window at ctrl_hz
                next_t = time.perf_counter()
                for i in range(self._take(len(chunk))):
                    if max_steps is not None and step >= max_steps:
                        break
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += tick
                    if next_t < now - 2 * tick:
                        next_t = time.perf_counter() + tick

                    cmd = clamp_action(prev_cmd, chunk[i], self.pos_limits, self.vel_limits)
                    if self.dry_run:
                        if step % 30 == 0:
                            print(f"[{step:5d}] action(dry-run)={np.round(cmd, 4).tolist()}")
                    else:
                        robot.send_action(cmd)
                    prev_cmd = cmd
                    step += 1
                if self.verbose:
                    print(f"[SAG] 窗口 {replans} 完成(累计 {step} 步)")
        except KeyboardInterrupt:
            print("\n[SAG] 中断")
        finally:
            robot.close()
            arr = np.array(rtts[1:]) if len(rtts) > 1 else np.array([])
            stats = ExecutorStats(
                steps=step,
                switches=replans,
                rtt_p50_ms=float(np.percentile(arr, 50) * 1000) if len(arr) else None,
                rtt_p95_ms=float(np.percentile(arr, 95) * 1000) if len(arr) else None,
            )
            if len(arr):
                print(f"[SAG] 共 {stats.steps} 步 / {stats.switches} 次规划; "
                      f"RTT p50={stats.rtt_p50_ms:.0f}ms p95={stats.rtt_p95_ms:.0f}ms")
        return stats

    def _take(self, chunk_len: int) -> int:
        if not self.n_action_steps:
            return chunk_len
        return min(int(self.n_action_steps), chunk_len)


__all__ = ["StopAndGoExecutor"]
