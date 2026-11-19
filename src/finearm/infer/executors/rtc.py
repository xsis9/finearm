"""RTC executor (Real-Time Chunking / ASAC style) — moved verbatim from rtc_client.RTCRunner.

One inference in flight at most, triggered ``k`` steps before the current
chunk runs out; chunk switches are alignment + linear-blend handovers. The
loop/align/blend/clamp math is unchanged from the pre-Phase-3 implementation
(locked by the golden comparison test in tests/infer/).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .base import ChunkExecutor, ExecutorStats, align_index, blend_transition, clamp_action


class RTCExecutor(ChunkExecutor):
    """RTC executor driving a robot backend + model adapter."""

    name = "rtc"

    def __init__(self, adapter, *, k=None, ctrl_hz=30.0, blend=2, starve="hold",
                 pos_limits=None, vel_limits=None, dry_run=False, verbose=True):
        """
        k          : trigger replanning when the current chunk has k steps left (ASAC
                     semantics, NOT every k steps); None = adaptive (RTT p95 based, recommended)
        ctrl_hz    : control rate, must match the training-data fps
        blend      : linear blend steps at chunk switches (0 = hard switch)
        starve     : chunk exhausted with no successor yet: "hold" last action / "stop"
        dry_run    : print actions instead of sending (the home move still moves the arm)
        """
        super().__init__(adapter, ctrl_hz=ctrl_hz, pos_limits=pos_limits,
                         vel_limits=vel_limits, dry_run=dry_run, verbose=verbose)
        self.k_user = k
        self.blend = blend
        self.starve = starve

        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._latest_payload = None  # filled by the main thread at trigger time (post-build_obs full request)
        self._next_chunk = None      # (T,7) produced by the inference thread
        self._inflight = False
        self._rtts = []
        self._running = False
        self._thread = None
        self._horizon = None         # current chunk length (model-dependent, e.g. 30 / 50)

    def _infer_loop(self):
        """Inference thread: wait for trigger → infer with the latest payload → publish the
        new chunk; at most one request in flight at any time."""
        while self._running:
            if not self._trigger.wait(timeout=0.5):
                continue
            self._trigger.clear()
            with self._lock:
                payload = self._latest_payload
                if payload is None or self._inflight:
                    continue
                self._inflight = True
            t0 = time.perf_counter()
            try:
                chunk = self.adapter.infer(payload)  # (T,7) absolute joint targets
                rtt = time.perf_counter() - t0
                with self._lock:
                    self._next_chunk = np.asarray(chunk, dtype=np.float64)
                    self._rtts.append(rtt)
                    self._rtts = self._rtts[-30:]
                    self._inflight = False
            except Exception as e:  # noqa: BLE001
                print(f"[RTC] 推理失败: {e}")
                with self._lock:
                    self._inflight = False

    def _k_eff(self, horizon=None) -> int:
        """Adaptive k = ceil(p95_RTT × hz) + 2, lower bound 3, upper bound horizon-5."""
        cap = max(3, (horizon or 30) - 5)
        with self._lock:
            rtts = list(self._rtts)
        if self.k_user is not None:
            return self.k_user
        if not rtts:
            return 5
        p95 = np.percentile(rtts, 95)
        k = int(np.clip(np.ceil(p95 * self.ctrl_hz) + 2, 3, cap))
        if len(rtts) < 3:
            return min(max(5, k), cap)
        return k

    def run(self, robot, task, max_steps=None) -> ExecutorStats:
        """Main control loop; Ctrl-C or max_steps exits safely (robot.close() → HOLD)."""
        self._running = True
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

        # initial chunk (synchronous inference; the first request includes the
        # server's jit/compile warmup, so a longer latency is normal)
        payload0 = self.adapter.build_obs(
            robot.get_obs(self.adapter.OBS_STYLE), task)
        t0 = time.perf_counter()
        chunk = np.asarray(self.adapter.infer(payload0), dtype=np.float64)
        self._horizon = len(chunk)
        print(f"[RTC] 首 chunk 就绪({time.perf_counter()-t0:.1f}s, 含服务端预热;"
              f" horizon={self._horizon} 步)")
        with self._lock:
            self._rtts.append(time.perf_counter() - t0)

        idx = 0                    # position inside the current chunk
        pending_blend = []         # transition action queue after a switch
        prev_cmd = None
        starve_cnt = 0
        switch_cnt = 0
        step = 0
        tick = 1.0 / self.ctrl_hz
        next_t = time.perf_counter()

        try:
            while max_steps is None or step < max_steps:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                next_t += tick
                if next_t < now - 2 * tick:  # more than 2 ticks behind → reset the metronome
                    next_t = time.perf_counter() + tick

                # ① new chunk available: align + blend transition, then switch
                with self._lock:
                    nc = self._next_chunk
                    self._next_chunk = None
                if nc is not None:
                    ref = chunk[min(idx, len(chunk) - 1)]
                    j = align_index(nc, ref)
                    pending_blend, start = blend_transition(chunk, idx, nc, j, self.blend)
                    # blend steps cover the new chunk's [j, j+blend); resume at
                    # j+blend to avoid jumping back
                    chunk, idx = nc, start
                    self._horizon = len(nc)
                    switch_cnt += 1
                    if self.verbose:
                        print(f"[RTC] 切换 chunk(第{switch_cnt}次):对齐索引 j={j}/{len(nc)},续跑于 {start}")

                # ② pick this step's action: transition → chunk → starvation policy
                if pending_blend:
                    raw = pending_blend.pop(0)
                elif idx < len(chunk):
                    raw = chunk[idx]
                    idx += 1
                else:
                    starve_cnt += 1
                    if starve_cnt % 30 == 1:
                        print(f"[RTC] 警告:chunk 耗尽(k 偏小或网络抖动) "
                              f"starve={self.starve}, 已发生 {starve_cnt} 步")
                    if self.starve == "hold":
                        raw = chunk[-1]
                    else:
                        print("[RTC] 停止(chunk 断档)")
                        break

                cmd = clamp_action(prev_cmd, raw, self.pos_limits, self.vel_limits)
                if self.dry_run:
                    if step % 30 == 0:
                        print(f"[{step:5d}] action(dry-run)={np.round(cmd, 4).tolist()}")
                else:
                    robot.send_action(cmd)
                prev_cmd = cmd
                step += 1

                # ③ with k steps left, trigger the next inference with fresh obs
                remaining = len(chunk) - idx
                k_eff = self._k_eff(self._horizon)
                with self._lock:
                    inflight = self._inflight
                if remaining <= k_eff and not inflight:
                    payload = self.adapter.build_obs(
                        robot.get_obs(self.adapter.OBS_STYLE), task)
                    with self._lock:
                        self._latest_payload = payload
                    self._trigger.set()
        except KeyboardInterrupt:
            print("\n[RTC] 中断")
        finally:
            self._running = False
            self._thread.join(timeout=1.0)
            robot.close()
            with self._lock:
                rtts = np.array(self._rtts[1:]) if len(self._rtts) > 1 else np.array([])
            stats = ExecutorStats(
                steps=step,
                switches=switch_cnt,
                rtt_p50_ms=float(np.percentile(rtts, 50) * 1000) if len(rtts) else None,
                rtt_p95_ms=float(np.percentile(rtts, 95) * 1000) if len(rtts) else None,
            )
            if len(rtts):
                print(f"[RTC] 共 {stats.steps} 步 / {stats.switches} 次切换; "
                      f"RTT p50={stats.rtt_p50_ms:.0f}ms "
                      f"p95={stats.rtt_p95_ms:.0f}ms "
                      f"k_eff={self._k_eff(self._horizon)}")
        return stats


__all__ = ["RTCExecutor"]
