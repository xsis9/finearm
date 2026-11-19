#!/usr/bin/env python3
"""通用 RTC(Real-Time Chunking / ASAC 风格)推理客户端"""

import argparse
import importlib
import os
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from adapters import available as adapters_available, load as load_adapter, mock_obs  # noqa: E402


class MockRobot:
    """假机械臂(--mock 空跑用):不需要 daemon/相机,观测来自 mock_obs。"""

    def __init__(self):
        self.n = 0
        self._obs_n = 0

    def get_obs(self, style="openpi"):
        self._obs_n += 1
        return mock_obs(style, seed=self._obs_n)

    def send_action(self, a7):
        self.n += 1
        if self.n % 30 == 0:
            print(f"[mock] step {self.n}: {np.round(a7, 3)}")

    def close(self):
        print(f"[mock] 共执行 {self.n} 步")


class RTCRunner:
    """RTC 执行器:对 robot(机械臂后端)+ adapter(模型适配器)驱动整段推理-执行循环。"""
    def __init__(self, robot, adapter, task,
                 k=None, ctrl_hz=30.0, blend=2,
                 pos_limits=None, vel_limits=None,
                 starve="hold", dry_run=False, verbose=True):
        """
        robot      : get_obs(style) / send_action(a7) / close(),如 robot_interface
        adapter    : adapters/<name>.Adapter(reset/build_obs/infer/OBS_STYLE)
        task       : 任务指令(训练 prompt 原文,适配器放进对应键)
        k          : 当前 chunk 还剩 k 步时触发重规划(ASAC 语义,不是每隔 k 步);
                     None = 自适应(按 RTT p95 估计,建议)
        ctrl_hz    : 控制频率,必须与训练数据 fps 一致
        blend      : 新旧 chunk 切换处的线性混合步数(0=硬切换)
        pos_limits : (lo7, hi7) 关节限位,None=不限
        vel_limits : (7,) 每控制步最大位移(rad/step),None=不限
        starve     : chunk 耗尽且新 chunk 未到时:"hold" 保持最后动作 / "stop" 停止
        dry_run    : True 时只打印不下发(验证推理链路;归位仍会实际运动)
        """
        self.robot = robot
        self.adapter = adapter
        self.task = task
        self.k_user = k
        self.ctrl_hz = ctrl_hz
        self.blend = blend
        self.pos_limits = pos_limits
        self.vel_limits = vel_limits
        self.starve = starve
        self.dry_run = dry_run
        self.verbose = verbose

        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._latest_payload = None  # 触发时由主线程填(build_obs 之后的完整请求)
        self._next_chunk = None      # 推理线程产出的 (T,7)
        self._inflight = False
        self._rtts = []
        self._running = False
        self._thread = None
        self._horizon = None         # 当前 chunk 长度(随模型不同,如 30 / 50)

    def _infer_loop(self):
        """推理线程:等触发信号 → 用最新 payload 推理 → 发布新 chunk;任一时刻最多一个在途请求。"""
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
                chunk = self.adapter.infer(payload)  # (T,7) 绝对关节目标
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
        """自适应 k = ceil(p95_RTT × hz) + 2,下限 3,上限 horizon-5"""
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

    def _clamp(self, prev, target):
        """按关节限位与每步限速裁剪目标(显式限幅只在这一处管理)。"""
        a = np.asarray(target, dtype=np.float64).copy()
        if self.pos_limits is not None:
            lo, hi = self.pos_limits
            a = np.clip(a, lo, hi)
        if prev is not None and self.vel_limits is not None:
            a = np.clip(a, prev - self.vel_limits, prev + self.vel_limits)
        return a

    @staticmethod
    def _align(new_chunk, ref_action):
        """在新 chunk 中找与 ref_action 最接近的索引(前 6 关节 L2,夹爪不参与)。"""
        d = np.linalg.norm(new_chunk[:, :6] - np.asarray(ref_action)[:6], axis=1)
        return int(np.argmin(d))

    def run(self, max_steps=None):
        """主控制循环;Ctrl-C 或 max_steps 结束时安全退出(robot.close() → HOLD)。"""
        self._running = True
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

        # 初始 chunk(同步推理;首条含服务端 jit/compile 预热,耗时较长属正常)
        payload0 = self.adapter.build_obs(
            self.robot.get_obs(self.adapter.OBS_STYLE), self.task)
        t0 = time.perf_counter()
        chunk = np.asarray(self.adapter.infer(payload0), dtype=np.float64)
        self._horizon = len(chunk)
        print(f"[RTC] 首 chunk 就绪({time.perf_counter()-t0:.1f}s, 含服务端预热;"
              f" horizon={self._horizon} 步)")
        with self._lock:
            self._rtts.append(time.perf_counter() - t0)

        idx = 0                    # 当前执行到 chunk 的位置
        pending_blend = []         # 切换后的过渡动作队列
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
                if next_t < now - 2 * tick:  # 落后超过 2 tick 则重置节拍
                    next_t = time.perf_counter() + tick

                # ① 若有新 chunk:对齐 + blend 过渡后切换
                with self._lock:
                    nc = self._next_chunk
                    self._next_chunk = None
                if nc is not None:
                    ref = chunk[min(idx, len(chunk) - 1)]
                    j = self._align(nc, ref)
                    old_i = idx
                    pending_blend = []
                    for s in range(self.blend):
                        if old_i + s < len(chunk):
                            alpha = (s + 1) / (self.blend + 1)
                            pending_blend.append(
                                (1 - alpha) * chunk[old_i + s]
                                + alpha * nc[min(j + s, len(nc) - 1)])
                    # blend 步已覆盖新 chunk 的 [j, j+blend),从 j+blend 续跑避免回跳
                    start = min(j + self.blend, len(nc) - 1) if self.blend > 0 else j
                    chunk, idx = nc, start
                    self._horizon = len(nc)
                    switch_cnt += 1
                    if self.verbose:
                        print(f"[RTC] 切换 chunk(第{switch_cnt}次):对齐索引 j={j}/{len(nc)},续跑于 {start}")

                # ② 取本步动作:过渡步 → chunk 步 → 断档处理
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

                cmd = self._clamp(prev_cmd, raw)
                if self.dry_run:
                    if step % 30 == 0:
                        print(f"[{step:5d}] action(dry-run)={np.round(cmd, 4).tolist()}")
                else:
                    self.robot.send_action(cmd)
                prev_cmd = cmd
                step += 1

                # ③ 剩 k 步时,用最新观测异步触发下一次推理
                remaining = len(chunk) - idx
                k_eff = self._k_eff(self._horizon)
                with self._lock:
                    inflight = self._inflight
                if remaining <= k_eff and not inflight:
                    payload = self.adapter.build_obs(
                        self.robot.get_obs(self.adapter.OBS_STYLE), self.task)
                    with self._lock:
                        self._latest_payload = payload
                    self._trigger.set()
        except KeyboardInterrupt:
            print("\n[RTC] 中断")
        finally:
            self._running = False
            self._thread.join(timeout=1.0)
            self.robot.close()
            with self._lock:
                rtts = np.array(self._rtts[1:]) if len(self._rtts) > 1 else np.array([])
            if len(rtts):
                print(f"[RTC] 共 {step} 步 / {switch_cnt} 次切换; "
                      f"RTT p50={np.percentile(rtts,50)*1000:.0f}ms "
                      f"p95={np.percentile(rtts,95)*1000:.0f}ms "
                      f"k_eff={self._k_eff(self._horizon)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=adapters_available(),
                    help="模型适配器(adapters/ 目录;新增模型见 README「情况 B · 新模型」)")
    ap.add_argument("--host", required=True, help="推理服务器 IP(走 ssh 隧道时 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="缺省用适配器默认端口")
    ap.add_argument("--task", default="pick up the bolt and place it on the nut",
                    help="任务指令;必须与训练 prompt 逐字一致")
    ap.add_argument("--k", type=int, default=None, help="固定 k;缺省自适应(按 RTT p95)")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--blend", type=int, default=2)
    ap.add_argument("--starve", choices=["hold", "stop"], default="hold")
    ap.add_argument("--arm", default="robot_interface:RobotInterface",
                    help="机械臂后端 模块:类;默认本目录 robot_interface.py")
    ap.add_argument("--home", default=None,
                    help="归位位姿(7 个逗号分隔值);优先级:本参数 > PANTHERA_RTC_HOME "
                         "> 适配器默认;空字符串跳过归位")
    ap.add_argument("--mock", action="store_true",
                    help="完全不接硬件空跑(不连 daemon/相机,假观测+假下发)")
    ap.add_argument("--mock-images", action="store_true",
                    help="连 daemon 但相机用全零图(配 mock daemon 可走完整 IPC 链路,不动真臂)")
    ap.add_argument("--dry-run", action="store_true",
                    help="归位后只打印 action 不下发(验证推理链路;归位阶段会实际运动)")
    ap.add_argument("--duration", type=float, default=None, help="运行时长(秒),到点自动停")
    ap.add_argument("--max-steps", type=int, default=None, help="最大步数(优先于 --duration)")
    # 显式限幅(daemon 侧仅保留逐拍跳变限制作为兜底,不承担常规限速)
    ap.add_argument("--pos-lo", default=None, help="7 个数,逗号分隔")
    ap.add_argument("--pos-hi", default=None)
    ap.add_argument("--vel-limit", default=None,
                    help="每步最大位移,如 0.02,0.02,0.02,0.02,0.02,0.02,0.01")
    args = ap.parse_args()

    adapter = load_adapter(args.model, host=args.host, port=args.port)
    adapter.reset()
    print(f"[RTC] 适配器 {adapter.NAME} 就绪(port {args.port or adapter.DEFAULT_PORT}, "
          f"obs_style={adapter.OBS_STYLE})")

    if args.mock:
        robot = MockRobot()
    else:
        mod_name, cls_name = args.arm.split(":")
        home = args.home if args.home is not None else \
            os.environ.get("PANTHERA_RTC_HOME", "") or adapter.DEFAULT_HOME
        robot = getattr(importlib.import_module(mod_name), cls_name)(
            home=home,
            img_sizes=(224,) if adapter.OBS_STYLE == "openpi" else (),
            mock_images=args.mock_images)
        if args.dry_run:
            print("*** DRY-RUN: action 只打印不下发(归位已实际运动)***")
        else:
            print("*** POLICY MODE ON: action 将下发 ***")

    pos_limits = None
    if args.pos_lo and args.pos_hi:
        pos_limits = (
            np.array([float(x) for x in args.pos_lo.split(",")]),
            np.array([float(x) for x in args.pos_hi.split(",")]),
        )
    vel_limits = None
    if args.vel_limit:
        vel_limits = np.array([float(x) for x in args.vel_limit.split(",")])

    max_steps = args.max_steps
    if max_steps is None and args.duration is not None:
        max_steps = int(args.duration * args.hz)

    RTCRunner(robot, adapter, args.task,
              k=args.k, ctrl_hz=args.hz, blend=args.blend,
              pos_limits=pos_limits, vel_limits=vel_limits,
              starve=args.starve, dry_run=args.dry_run).run(max_steps=max_steps)


if __name__ == "__main__":
    main()
