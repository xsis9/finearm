"""Panthera-HT 机械臂/相机后端 —— deploy/infer 各推理客户端共用。

经机械臂驱动层(deploy/arm 的 daemon IPC,默认 /tmp/panthera_lerobot_control.sock)
读状态、下发动作;
双相机由后台线程取流,只保留最新帧。构造时自动完成:锁相机参数 → 开流 → 归位 → 进 policy 模式。

环境变量(均可省略):
  PANTHERA_CAM_FRONT / PANTHERA_CAM_WRIST  相机源,默认 /dev/video4、/dev/video2
  PANTHERA_RTC_HOME    默认归位位姿 "j1,...,j6,gripper"(显式 home 参数优先于它;置空跳过归位)
  PANTHERA_CAM_LOCK    lock_cameras.sh 路径(默认按本文件位置定位到 deploy/utils/ 下)
  PANTHERA_CAM_LOCK_SKIP=1               跳过相机锁定(仅调试)
"""
import os
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
from openpi_client import image_tools

_ARM = pathlib.Path(__file__).resolve().parents[1] / "arm" / "panthera"
sys.path.insert(0, str(_ARM))
from panthera_control.ipc import ControlClient  # noqa: E402

SOCKET_PATH = "/tmp/panthera_lerobot_control.sock"

CAM_LOCK_SCRIPT = os.environ.get(
    "PANTHERA_CAM_LOCK",
    str(pathlib.Path(__file__).resolve().parents[1] / "utils" / "lock_cameras.sh"),
)
DEFAULT_HOME = "0,0,0,0,0,0,1.8"  # 默认归位位姿


def _lock_cameras():
    """执行相机参数锁定脚本;失败即中止(观测参数与训练不一致,推理没有意义)。"""
    if os.environ.get("PANTHERA_CAM_LOCK_SKIP", "0") == "1":
        print("[cam-lock] PANTHERA_CAM_LOCK_SKIP=1,跳过相机锁定(仅调试)")
        return
    if not pathlib.Path(CAM_LOCK_SCRIPT).exists():
        raise SystemExit(f"相机锁定脚本不存在: {CAM_LOCK_SCRIPT}")
    r = subprocess.run(["bash", CAM_LOCK_SCRIPT], check=False)
    if r.returncode != 0:
        raise SystemExit(f"相机锁定失败(exit={r.returncode}),推理观测必须与采集同参数")


class CameraGrabber:
    """单相机后台取流线程:持续 read 只保留最新一帧(BGR→RGB,并按 sizes 预 resize)。

    latest(size) 无阻塞、帧龄 ≤ 1/fps:latest(None) 返回原始 640×480 帧,
    latest(s) 返回预 resize 好的 s×s 帧(未配置的尺寸退回原始帧)。
    读帧连续失败自动重开相机(最多 3 次),仍失败则标记失联,latest() 抛错。
    """

    def __init__(self, src: str, name: str, sizes=()):
        self.src, self.name = src, name
        self._sizes = tuple(s for s in sizes if s)  # 需要预 resize 的尺寸,如 (224,)
        self._lock = threading.Lock()
        self._frames = {}      # {None: 原始帧, 224: resize 帧, ...} 最新一帧
        self._t = None         # 该帧的 monotonic 时间戳
        self._running = True
        self._dead = False
        self._reopen = 0
        self._warn_t = 0.0
        self.cap = self._open_checked()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{name}", daemon=True)
        self._thread.start()

    def _open(self):
        """按固定格式开流:显式 V4L2 后端 + MJPG + 640×480@30 + 缓冲 ≥2。

        显式 V4L2:FFMPEG 后端会忽略 FOURCC 设置;显式 MJPG:两路 YUYV 会超出
        USB2 带宽;缓冲 ≥2:部分相机固件在单缓冲下帧率减半。
        """
        import cv2
        cap = cv2.VideoCapture(self.src if not str(self.src).isdigit() else int(self.src),
                               cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开相机: {self.src}")
        return cap

    def _open_checked(self):
        """开流并实测分辨率/帧率;不达标直接报错(格式协商失败常见于相机残留坏配置)。"""
        import cv2
        cap = self._open()
        cap.read()  # 触发实际格式协商
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t0, n = time.time(), 0
        while time.time() - t0 < 1.0:
            if cap.read()[0]:
                n += 1
        fps = n / (time.time() - t0)
        print(f"[cam] {self.name} @{self.src}: {w}x{h} 实测 {fps:.0f} fps")
        if (w, h) != (640, 480) or fps < 20:
            cap.release()
            raise RuntimeError(
                f"相机 {self.src} 协商失败({w}x{h}@{fps:.0f}fps,需 640x480>=20fps);"
                "重新插拔相机、v4l2-ctl --list-devices 核对后重试")
        return cap

    def _loop(self):
        fails = 0
        while self._running:
            ok, f = self.cap.read()  # 阻塞等帧并持续排空 → 帧永远新鲜
            if not ok:
                fails += 1
                if fails >= 30 and self._reopen < 3:  # 连续 ~1s 失败 → 重开
                    self._reopen += 1
                    print(f"[cam] {self.name} 读帧连续失败,重开({self._reopen}/3)")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    try:
                        self.cap = self._open()
                    except Exception as e:
                        print(f"[cam] {self.name} 重开失败: {e}")
                        time.sleep(1.0)
                    fails = 0
                    continue
                if fails >= 30 and self._reopen >= 3:
                    self._dead = True
                    print(f"[cam] {self.name} 重开 3 次仍失败,标记失联")
                    return
                time.sleep(0.005)
                continue
            fails = 0
            rgb = np.ascontiguousarray(f[:, :, ::-1])  # BGR→RGB,不翻转不裁剪
            frames = {None: rgb}
            for s in self._sizes:
                frames[s] = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, s, s))
            with self._lock:
                self._frames = frames
                self._t = time.monotonic()

    def latest(self, size=None, max_age_s: float = 0.3, first_frame_wait_s: float = 2.0) -> np.ndarray:
        """返回最新帧;首帧未出时最多等 first_frame_wait_s(相机开流预热),超时报错。"""
        deadline = time.monotonic() + first_frame_wait_s
        while True:
            with self._lock:
                frames, t = self._frames, self._t
            frame = frames.get(size, frames.get(None)) if frames else None
            if frame is not None:
                break
            if self._dead:
                raise RuntimeError(f"相机 {self.name} 已失联")
            if time.monotonic() > deadline:
                raise RuntimeError(f"相机 {self.name} 尚未出帧(已等 {first_frame_wait_s}s)")
            time.sleep(0.01)
        age = time.monotonic() - t
        if age > max_age_s and time.monotonic() - self._warn_t > 1.0:
            self._warn_t = time.monotonic()
            print(f"[cam] 警告:{self.name} 帧龄 {age*1000:.0f}ms(>{max_age_s*1000:.0f}ms)")
        return frame

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


class MockGrabber:
    """假相机(mock_images 模式):latest() 返回全零图,不需要相机硬件,用于链路联调。"""

    def __init__(self, name: str):
        self.name = name

    def latest(self, size=None, max_age_s: float = 0.3, first_frame_wait_s: float = 2.0) -> np.ndarray:
        if size is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return np.zeros((size, size, 3), dtype=np.uint8)

    def close(self):
        pass


class RobotInterface:
    """推理客户端的机械臂后端,约定接口:get_obs(style) / send_action(a7) / close()。

    参数:
      cam_front / cam_wrist : 相机源,默认取 PANTHERA_CAM_FRONT/WRIST,再默认
                              /dev/video4、/dev/video2
      home                  : 归位位姿 "j1,...,j6,gripper"(7 个逗号分隔值);
                              "env"(默认)取 PANTHERA_RTC_HOME;空字符串跳过归位
      img_sizes             : 取流线程预 resize 的尺寸;(224,) 供 openpi 风格观测,
                              () 用原始帧(服务端自行 resize 的模型)
      mock_images           : True 时不接真实相机,观测用全零图
      home_tol / home_timeout_s / home_move_s : 归位到位容差 / 等待超时 / 插值时长
    """

    def __init__(self, cam_front=None, cam_wrist=None, home="env",
                 img_sizes=(224,), mock_images=False,
                 home_tol=0.1, home_timeout_s=8.0, home_move_s=2.5):
        cam_front = cam_front or os.environ.get("PANTHERA_CAM_FRONT", "/dev/video4")
        cam_wrist = cam_wrist or os.environ.get("PANTHERA_CAM_WRIST", "/dev/video2")
        if home == "env":
            home = os.environ.get("PANTHERA_RTC_HOME", DEFAULT_HOME)

        self.client = ControlClient(SOCKET_PATH)
        print("daemon status:", self.client.status())

        if mock_images:
            print("[cam] mock_images 模式:不接真实相机,观测用全零图")
            self._front = MockGrabber("front")
            self._wrist = MockGrabber("wrist")
        else:
            _lock_cameras()
            self._front = CameraGrabber(cam_front, "front", sizes=img_sizes)
            self._wrist = CameraGrabber(cam_wrist, "wrist", sizes=img_sizes)

        if home:
            hv = np.array([float(v) for v in home.split(",")], dtype=np.float64)
            if hv.shape != (7,):
                raise SystemExit("home 需要 7 个逗号分隔值(6 关节 + 夹爪)")
            self._go_home(hv, tol=home_tol, timeout_s=home_timeout_s, move_s=home_move_s)
        else:
            print("跳过归位(home 为空)")
        # 归位已在 policy 模式;重复设置无副作用
        self.client.set_mode("policy")

    def _go_home(self, home, tol=0.1, timeout_s=8.0, move_s=2.5, fps=30.0):
        """policy 模式下线性插值到 home(速度由 move_s 控制),到位判定容差 tol。

        腕部带载存在静态下垂,容差不宜小于 0.1 rad,否则可能永远等不到位。
        """
        self.client.set_mode("policy")
        snap = self.client.snapshot(max_age_s=0.25)
        start = np.asarray(snap["observation.state"], dtype=np.float64)[:7].copy()
        total = float(np.abs(home[:6] - start[:6]).max())
        print(f"*** 归位中: {np.round(home, 4).tolist()} (最大行程 {total:.3f} rad, {move_s:.1f}s) ***")
        period = 1.0 / fps
        n_ramp = max(1, int(move_s * fps))
        for i in range(1, n_ramp + 1):
            t0 = time.time()
            target = start + (home - start) * (i / n_ramp)
            self.client.send_policy_action(target.tolist())
            rest = period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            t0 = time.time()
            cur = np.asarray(self.client.snapshot(max_age_s=0.25)["observation.state"],
                             dtype=np.float64)[:6]
            self.client.send_policy_action(home.tolist())
            err = float(np.abs(cur - home[:6]).max())
            print(f"\r归位误差 max={err:.4f} rad", end="", flush=True)
            if err < tol:
                print(" — 已到位")
                return
            rest = period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
        print(" — 归位超时,继续执行")

    def get_obs(self, style: str = "openpi") -> dict:
        """最新观测,毫秒级返回(帧龄 ≤ 1/fps,不阻塞控制环)。

        style="openpi":224×224 预 resize 图 + state 前 7 维
            {"observation/image", "observation/wrist_image", "observation/state"}
        style="raw":原始 640×480 帧 + 完整 14 维 state(数据集语义)
            {"front", "wrist", "state"}
        """
        if style == "openpi":
            return {
                "observation/image": self._front.latest(224),        # 224×224 RGB uint8
                "observation/wrist_image": self._wrist.latest(224),
                "observation/state": np.asarray(                     # 前 7 维:6 关节 + 夹爪
                    self.client.snapshot(max_age_s=0.25)["observation.state"],
                    dtype=np.float32)[:7],
            }
        if style == "raw":
            return {
                "front": self._front.latest(),      # 640×480 RGB uint8 原始帧
                "wrist": self._wrist.latest(),
                "state": np.asarray(                # 14 维:6 关节 + 夹爪 + 7 速度,原样
                    self.client.snapshot(max_age_s=0.25)["observation.state"],
                    dtype=np.float32),
            }
        raise ValueError(f"未知 style: {style}(可选 openpi | raw)")

    def send_action(self, a7: np.ndarray):
        """下发 7 维绝对关节目标(6 关节 + 夹爪),单位/顺序与训练数据 action 一致。"""
        a = np.asarray(a7, dtype=np.float64).ravel()
        if a.shape[0] < 7 or not np.all(np.isfinite(a[:7])):
            raise ValueError(f"非法 action: {a}")
        self.client.send_policy_action(a[:7].tolist())

    def close(self):
        """切回 HOLD 并释放相机;推理客户端在 finally 中调用。"""
        try:
            self.client.hold()
            print("[arm] 已切回 HOLD")
        except Exception as e:
            print(f"[arm] hold 失败: {e}")
        self._front.close()
        self._wrist.close()


def _selftest(args):
    """不动臂,只验证观测链路:取流格式、get_obs 时延、图像方向/曝光。"""
    iface = RobotInterface(home="")  # 不归位
    try:
        ts = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            obs = iface.get_obs()
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000
        print(f"\nget_obs() {args.n} 次: mean={ts.mean():.2f}ms max={ts.max():.2f}ms"
              f"({'达标' if ts.max() < 5 else '超时!检查相机线程'};要求 <5ms)")
        print("state7 =", np.round(obs["observation/state"], 4).tolist())
        if args.dump_obs:
            import cv2
            for key, name in [("observation/image", "front"), ("observation/wrist_image", "wrist")]:
                p = pathlib.Path(args.dump_obs) / f"obs_{name}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), np.asarray(obs[key])[:, :, ::-1])
                print("已存", p)
    finally:
        iface.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RobotInterface 自检(不动臂)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dump-obs", default=None, help="目录:存一张 front/wrist 图肉眼核对")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("请用 --selftest 运行(自检模式)")
    _selftest(a)
