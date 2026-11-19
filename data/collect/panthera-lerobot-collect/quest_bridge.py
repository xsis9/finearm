#!/usr/bin/env python3
"""Quest3S teleoperation bridge for the Panthera follower arm.

Subscribes to the quest2ros topics (/q2r_<side>_hand_pose, /q2r_<side>_hand_inputs),
converts anchored relative hand motion into 6-joint + gripper policy actions with
pinocchio FK/IK, and streams them to the Panthera control daemon over its Unix
socket IPC in "policy" mode.

The bridge never talks to hardware directly: the daemon stays the single writer.
Kinematics are computed from the SDK URDF (no serial connection is opened).

ROS subscription is done through `ros2 topic echo` subprocesses so this script
can keep running under the project's Python 3.12 environment (rclpy is only
available for the system Python 3.10).
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import pinocchio as pin
    from scipy.spatial.transform import Rotation as Rotation
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"error: quest_bridge requires pinocchio, scipy, numpy, PyYAML: {exc}", file=sys.stderr)
    raise SystemExit(2)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panthera_lerobot_control.ipc import ControlClient, ControlClientError  # noqa: E402

DEFAULT_JOINT_COUNT = 6
DEFAULT_GRIPPER_RANGE = (0.0, 1.6)
SIM_INPUT_HZ = 36.0
# Table-overview home pose: j1=j5=0, tool tilted ~26 degrees (j6 sets the
# tilt), end effector ~0.33m in front of the base. Fine-tune with --home-joints.
DEFAULT_HOME_JOINTS = (0.0, 0.90, 0.90, -0.95, 0.0, 0.0)


class BridgeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# ROS input via `ros2 topic echo` subprocesses
# ---------------------------------------------------------------------------


class EchoSubscriber:
    """Continuously parse YAML documents from `ros2 topic echo <topic>`."""

    def __init__(self, topic: str, parser, *, side: str, verbose: bool = False):
        self.topic = topic
        self.parser = parser
        self.latest: Optional[dict] = None
        self.received_count = 0
        self.last_rx_time: Optional[float] = None
        self._lock = threading.Lock()
        self._verbose = verbose
        self._side = side
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        command = ["ros2", "topic", "echo", self.topic]
        if self._verbose:
            print(f"[bridge] starting: {shlex.join(command)}", file=sys.stderr)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                "ros2 CLI not found; source /opt/ros/humble/setup.bash and "
                "~/quest_ws/install/setup.bash before running the bridge"
            ) from exc
        self._thread = threading.Thread(target=self._read_loop, name=f"echo-{self._side}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read_loop(self) -> None:
        buffer: list[str] = []
        while not self._stop.is_set():
            line = self._process.stdout.readline()
            if not line:
                break
            if line.strip() == "---":
                document = "".join(buffer)
                buffer = []
                if not document.strip():
                    continue
                try:
                    message = yaml.safe_load(document)
                    parsed = self.parser(message)
                except Exception:
                    continue
                if parsed is None:
                    continue
                with self._lock:
                    self.latest = parsed
                    self.received_count += 1
                    self.last_rx_time = time.monotonic()
            else:
                buffer.append(line)

    def snapshot_latest(self) -> Optional[dict]:
        with self._lock:
            return None if self.latest is None else dict(self.latest)


def _parse_pose_message(message: dict) -> Optional[dict]:
    if not isinstance(message, dict):
        return None
    pose = message.get("pose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    try:
        values = [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    quaternion = values[3:]
    if abs(np.linalg.norm(quaternion)) < 1e-9:
        return None
    return {"position": values[:3], "quaternion_xyzw": quaternion}


def _parse_inputs_message(message: dict) -> Optional[dict]:
    if not isinstance(message, dict):
        return None
    try:
        values = {
            "button_upper": bool(message.get("button_upper", False)),
            "button_lower": bool(message.get("button_lower", False)),
            "press_index": float(message.get("press_index", 0.0)),
        }
    except (TypeError, ValueError):
        return None
    if not math.isfinite(values["press_index"]):
        return None
    values["press_index"] = min(max(values["press_index"], 0.0), 1.0)
    return values


# ---------------------------------------------------------------------------
# Kinematics: pinocchio model loaded directly from the SDK URDF (no hardware)
# ---------------------------------------------------------------------------


class ArmKinematics:
    """Position/rotation FK and damped-least-squares IK from the SDK URDF."""

    def __init__(self, follower_config: Path):
        follower_config = Path(follower_config)
        if not follower_config.exists():
            raise BridgeError(f"follower config not found: {follower_config}")
        with open(follower_config, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        config_dir = follower_config.parent
        urdf_path = (config_dir / config["urdf"]["file_path"]).resolve()
        if not urdf_path.exists():
            raise BridgeError(f"URDF not found: {urdf_path}")

        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.joint_names = list(config["kinematics"]["joint_names"])
        self.joint_count = len(self.joint_names)

        limits = config.get("robot", {}).get("joint_limits", {})
        self.lower = np.array(limits.get("lower", [-math.pi] * self.joint_count), dtype=float)
        self.upper = np.array(limits.get("upper", [math.pi] * self.joint_count), dtype=float)

        gripper_limits = config.get("robot", {}).get("gripper_limits", {})
        self.gripper_min = float(gripper_limits.get("lower", DEFAULT_GRIPPER_RANGE[0]))
        self.gripper_max = float(gripper_limits.get("upper", DEFAULT_GRIPPER_RANGE[1]))

        eef_link = config["urdf"]["end_effector_link"]
        if not self.model.existFrame(eef_link):
            raise BridgeError(f"end effector frame {eef_link!r} not found in URDF")
        self.eef_frame_id = self.model.getFrameId(eef_link)

        self._q_indices = []
        self._v_indices = []
        for name in self.joint_names:
            if not self.model.existJointName(name):
                raise BridgeError(f"joint {name!r} from config not found in URDF")
            joint_id = self.model.getJointId(name)
            self._q_indices.append(self.model.joints[joint_id].idx_q)
            self._v_indices.append(self.model.joints[joint_id].idx_v)

    def _pack_q(self, joint_values) -> np.ndarray:
        q = pin.neutral(self.model)
        for index, value in zip(self._q_indices, joint_values):
            q[index] = value
        return q

    def _unpack_q(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[index] for index in self._q_indices])

    def clamp_joints(self, values) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(values, dtype=float), self.lower), self.upper)

    def fk(self, joint_values) -> dict:
        q = self._pack_q(joint_values)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.eef_frame_id]
        return {
            "position": placement.translation.copy(),
            "rotation": placement.rotation.copy(),
        }

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian_full = pin.computeFrameJacobian(
            self.model, self.data, q, self.eef_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        jacobian = np.zeros((6, self.joint_count))
        for column, index in enumerate(self._v_indices):
            jacobian[:, column] = jacobian_full[:, index]
        return jacobian

    def solve_ik(
        self,
        target_position,
        target_rotation: Optional[np.ndarray],
        init_q,
        *,
        max_iter: int = 200,
        pos_eps: float = 3e-3,
        rot_eps: float = 2e-2,
        damping: float = 1e-2,
        max_step: float = 0.2,
        rotation_gain: float = 0.5,
        attempts: int = 4,
    ) -> Optional[np.ndarray]:
        """Damped least squares IK seeded at init_q; returns joints or None.

        Runs a short descent from init_q (motion continuity), then falls back
        to mid-range/random restarts if the seed gets stuck against joint
        limits. Returns the best solution found within tolerance.
        """
        target_position = np.asarray(target_position, dtype=float)
        use_rotation = target_rotation is not None
        if use_rotation:
            target_rotation = np.asarray(target_rotation, dtype=float)
        seeds = [self.clamp_joints(init_q)]
        rng = np.random.default_rng(0)
        mid = (self.lower + self.upper) / 2.0
        for attempt in range(max(0, attempts - 1)):
            if attempt == 0:
                seeds.append(mid)
            else:
                seeds.append(self.lower + rng.random(self.joint_count) * (self.upper - self.lower))
        best_joints: Optional[np.ndarray] = None
        best_cost = math.inf
        for seed in seeds:
            joints, converged, cost = self._ik_descent(
                target_position,
                target_rotation if use_rotation else None,
                seed,
                max_iter=max_iter,
                pos_eps=pos_eps,
                rot_eps=rot_eps,
                damping=damping,
                max_step=max_step,
                rotation_gain=rotation_gain,
            )
            if cost < best_cost:
                best_cost = cost
                best_joints = joints
            if converged:
                return self.clamp_joints(joints)
        if best_cost < (pos_eps + 0.1 * rot_eps):
            return self.clamp_joints(best_joints)
        return None

    def _ik_descent(
        self,
        target_position: np.ndarray,
        target_rotation: Optional[np.ndarray],
        seed: np.ndarray,
        *,
        max_iter: int,
        pos_eps: float,
        rot_eps: float,
        damping: float,
        max_step: float,
        rotation_gain: float,
    ):
        joints = np.array(seed, dtype=float)
        best_joints = joints
        best_cost = math.inf
        for _ in range(max_iter):
            q = self._pack_q(joints)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            placement = self.data.oMf[self.eef_frame_id]
            position_error = target_position - placement.translation
            if target_rotation is not None:
                rotation_error = pin.log3(target_rotation @ placement.rotation.T) * rotation_gain
            else:
                rotation_error = np.zeros(3)
            pos_norm = float(np.linalg.norm(position_error))
            rot_norm = float(np.linalg.norm(rotation_error))
            cost = pos_norm + 0.2 * rot_norm
            if cost < best_cost:
                best_cost = cost
                best_joints = joints
            if pos_norm < pos_eps and rot_norm < rot_eps:
                return joints, True, best_cost
            error = np.hstack([position_error, rotation_error])
            jacobian = self._jacobian(q)
            jjt = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
            try:
                delta = jacobian.T @ np.linalg.solve(jjt, error)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            step = float(np.linalg.norm(delta))
            if step > max_step:
                delta *= max_step / step
            joints = self.clamp_joints(joints + delta)
        return best_joints, False, best_cost


# ---------------------------------------------------------------------------
# Bridge control logic
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    controller_position: np.ndarray
    controller_rotation: np.ndarray
    fk_position: np.ndarray
    fk_rotation: np.ndarray


class QuestBridge:
    def __init__(self, args):
        self.args = args
        self.client = ControlClient(args.socket, timeout_s=args.timeout_s)
        self.kinematics = ArmKinematics(_resolve_follower_config(args))
        self.pose_sub: Optional[EchoSubscriber] = None
        self.inputs_sub: Optional[EchoSubscriber] = None
        self.anchor: Optional[Anchor] = None
        self.previous_button_lower = False
        self.previous_button_upper = False
        self.gripper_toggle_open = False
        self.last_action: Optional[list[float]] = None
        self.last_ik_warning = 0.0
        self.last_status_print = 0.0
        self.stop_requested = False

    # -- daemon helpers -----------------------------------------------------

    def daemon_status(self) -> dict:
        return self.client.status()

    def follower_joints(self) -> list[float]:
        snapshot = self.client.snapshot(max_age_s=self.args.snapshot_max_age_s)
        state = snapshot["observation.state"]
        return [float(value) for value in state[: self.kinematics.joint_count]]

    # -- input sources ------------------------------------------------------

    def start_inputs(self) -> None:
        if self.args.sim_input:
            return
        side = self.args.side
        self.pose_sub = EchoSubscriber(
            f"/q2r_{side}_hand_pose", _parse_pose_message, side=side, verbose=self.args.verbose
        )
        self.inputs_sub = EchoSubscriber(
            f"/q2r_{side}_hand_inputs", _parse_inputs_message, side=side, verbose=self.args.verbose
        )
        self.pose_sub.start()
        self.inputs_sub.start()

    def stop_inputs(self) -> None:
        for subscriber in (self.pose_sub, self.inputs_sub):
            if subscriber is not None:
                subscriber.stop()

    def latest_inputs(self) -> dict:
        if self.args.sim_input:
            return _simulated_inputs()
        snapshot = self.inputs_sub.snapshot_latest() if self.inputs_sub else None
        if snapshot is None:
            return {"button_upper": False, "button_lower": False, "press_index": 0.0}
        if self.inputs_sub.last_rx_time is not None:
            age = time.monotonic() - self.inputs_sub.last_rx_time
            if age > self.args.input_timeout_s:
                return {"button_upper": False, "button_lower": False, "press_index": 0.0}
        return snapshot

    def latest_hand_pose(self) -> Optional[dict]:
        if self.args.sim_input:
            return _simulated_hand_pose()
        if self.pose_sub is None:
            return None
        snapshot = self.pose_sub.snapshot_latest()
        if snapshot is None or self.pose_sub.last_rx_time is None:
            return None
        if time.monotonic() - self.pose_sub.last_rx_time > self.args.input_timeout_s:
            return None
        return snapshot

    # -- anchored relative control ------------------------------------------

    def _update_anchor(self, hand_pose: dict, joints: list[float]) -> None:
        fk = self.kinematics.fk(joints)
        self.anchor = Anchor(
            controller_position=np.array(hand_pose["position"], dtype=float),
            controller_rotation=_quaternion_to_matrix(hand_pose["quaternion_xyzw"]),
            fk_position=fk["position"],
            fk_rotation=fk["rotation"],
        )

    def _frame_matrix(self) -> np.ndarray:
        """Quest world frame -> robot base frame: permutation (axis-map) + signs.

        Measured on hardware (2026-10-23): the quest2ros App frame is
        x=forward, y=left, z=up; the robot base frame (teleop-verified) is
        X=forward/back, Y=left/right, Z=up/down - so the default map is the
        identity permutation and only signs may need flipping per setup.
        """
        axes = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        permutation = np.array([axes[name] for name in self.args.axis_map], dtype=float)
        signs = np.array(self.args.axis_signs, dtype=float)
        return signs[:, None] * permutation

    def _wrist_matrix(self) -> np.ndarray:
        """Hand-local frame -> tool-local frame axis correspondence.

        Hardware-verified (2026-10-23, tool pointing down): joint5 rotates about
        the vertical axis (gripper swings left/right) and joint6 about the
        forward axis (gripper tilt angle). With the hand-local pointing axis
        mapped to the tool's forward-rotation axis (identity by default), a
        wrist twist drives joint6, wrist pitch tilts fore/aft, and wrist yaw
        drives joint5.
        """
        axes = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        return np.array([axes[name] for name in self.args.wrist_map], dtype=float)

    def _target_pose(self, hand_pose: dict) -> tuple[np.ndarray, np.ndarray]:
        position = np.array(hand_pose["position"], dtype=float)
        rotation = _quaternion_to_matrix(hand_pose["quaternion_xyzw"])
        assert self.anchor is not None
        # Translation: Quest world frame -> robot base frame (A @ delta).
        transform = self._frame_matrix()
        delta_position = transform @ ((position - self.anchor.controller_position) * self.args.scale)
        reach = float(np.linalg.norm(delta_position))
        if reach > self.args.max_reach:
            delta_position *= self.args.max_reach / reach
        target_position = self.anchor.fk_position + delta_position
        if self.args.no_rotation:
            target_rotation = self.anchor.fk_rotation.copy()
            return target_position, target_rotation
        # Rotation: express the wrist delta in the hand's own anchor-local
        # frame, map it to the tool's local frame (C @ R @ C^T) and apply it
        # on the right so it acts about the tool's own axes: a wrist twist
        # rolls the tool instead of orbiting it about a world axis.
        hand_local_delta = self.anchor.controller_rotation.T @ rotation
        tool_local_delta = self._wrist_matrix() @ hand_local_delta @ self._wrist_matrix().T
        target_rotation = self.anchor.fk_rotation @ tool_local_delta
        return target_position, target_rotation

    def _gripper_target(self, inputs: dict) -> float:
        kin = self.kinematics
        if self.args.gripper_mode == "toggle":
            if inputs["button_upper"] and not self.previous_button_upper:
                self.gripper_toggle_open = not self.gripper_toggle_open
            return kin.gripper_max if self.gripper_toggle_open else kin.gripper_min
        trigger = inputs["press_index"]
        return kin.gripper_max - trigger * (kin.gripper_max - kin.gripper_min)

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_ik_warning >= self.args.warn_interval_s:
            self.last_ik_warning = now
            print(f"[bridge] warning: {message}", file=sys.stderr)

    # -- startup pose selection ------------------------------------------------

    def _resolve_start_pose(self) -> bool:
        """Ask whether to move to the home pose first; True = move to home."""
        choice = self.args.start_pose
        if choice == "current":
            return False
        if choice == "ask":
            if not sys.stdin.isatty():
                return False
            home = self._home_joints()
            fk = self.kinematics.fk(home)
            print(
                "[bridge] home pose (looks down at the tabletop): joints="
                + np.array2string(home, precision=2)
                + f" ee={np.array2string(fk['position'], precision=2)}",
                file=sys.stderr,
            )
            while True:
                try:
                    answer = input("起始位姿: [1] 归位到俯视桌面位姿  [2] 保持当前位姿 (回车=2): ").strip()
                except (EOFError, KeyboardInterrupt):
                    return False
                if answer in ("", "2"):
                    return False
                if answer == "1":
                    return True
        return True

    def _home_joints(self) -> np.ndarray:
        return self.kinematics.clamp_joints(self.args.home_joints)

    def _move_to_home(self) -> None:
        """Ramp the follower to the home pose through policy actions, then hold."""
        home = [float(value) for value in self._home_joints()]
        gripper = self.kinematics.gripper_max
        if self.args.dry_run:
            print(f"[bridge] dry-run: would move to home joints {np.round(home, 3).tolist()}", file=sys.stderr)
            return
        print(
            f"[bridge] moving to home pose in {self.args.home_countdown_s:.0f}s - "
            "keep the workspace clear, stay ready to press Ctrl+C",
            file=sys.stderr,
        )
        for remaining in range(int(self.args.home_countdown_s), 0, -1):
            print(f"[bridge] {remaining}...", file=sys.stderr)
            time.sleep(1.0)
        start = self.follower_joints()
        duration = max(0.5, float(self.args.home_move_s))
        steps = max(1, int(duration * self.args.hz))
        period = 1.0 / float(self.args.hz)
        self.client.set_mode("policy")
        for step in range(1, steps + 1):
            eased = 0.5 - 0.5 * math.cos(math.pi * step / steps)
            action = [
                current + (target - current) * eased for current, target in zip(start, home)
            ] + [gripper]
            self.client.send_policy_action(action)
            time.sleep(period)
        self.client.hold()
        print(
            "[bridge] home pose reached (follower holding); hold A on the controller to start teleoperation",
            file=sys.stderr,
        )

    # -- main loop ------------------------------------------------------------

    def run(self) -> int:
        status = self.daemon_status()
        if not status.get("running"):
            raise BridgeError("daemon is not running")
        mode = status.get("mode")
        expected_mode = "mock" if self.args.mock else "live"
        if mode != expected_mode:
            raise BridgeError(f"daemon mode is {mode!r}, expected {expected_mode!r}")
        print(
            f"[bridge] daemon ready: mode={mode} control_mode={status.get('control_mode')} "
            f"socket={self.args.socket}",
            file=sys.stderr,
        )

        if self._resolve_start_pose():
            self._move_to_home()

        self.start_inputs()
        print(
            f"[bridge] running: side={self.args.side} hz={self.args.hz} scale={self.args.scale} "
            f"max_reach={self.args.max_reach} gripper={self.args.gripper_mode} "
            f"axis_map={self.args.axis_map} axis_signs={self.args.axis_signs}"
            + (" sim_input" if self.args.sim_input else "")
            + (" dry_run" if self.args.dry_run else ""),
            file=sys.stderr,
        )
        print(
            "[bridge] hold the anchor button (button_lower; physically the A button on Quest) "
            "to anchor and drive; squeeze the index trigger to close the gripper; release to hold.",
            file=sys.stderr,
        )

        period = 1.0 / float(self.args.hz)
        next_tick = time.monotonic()

        def _request_stop(_signum, _frame):
            self.stop_requested = True

        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_stop)
        try:
            while not self.stop_requested:
                now = time.monotonic()
                self._tick()
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print("\n[bridge] interrupted", file=sys.stderr)
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
            self._shutdown()
        return 0

    def _tick(self) -> None:
        inputs = self.latest_inputs()
        hand_pose = self.latest_hand_pose()
        now = time.monotonic()

        button = inputs["button_lower"]
        if button and not self.previous_button_lower:
            if hand_pose is None:
                self._warn("anchor requested but no hand pose yet; ignoring")
            else:
                joints = self.follower_joints()
                self._update_anchor(hand_pose, joints)
                if not self.args.dry_run:
                    self.client.set_mode("policy")
                print(
                    "[bridge] anchored: fk_pos="
                    + np.array2string(self.anchor.fk_position, precision=3),
                    file=sys.stderr,
                )
        if not button and self.previous_button_lower:
            self.anchor = None
            print("[bridge] released; daemon will hold after action timeout", file=sys.stderr)
        self.previous_button_lower = button
        self.previous_button_upper = inputs["button_upper"]

        if self.anchor is not None and hand_pose is not None:
            target_position, target_rotation = self._target_pose(hand_pose)
            joints = self.follower_joints()
            solution = self.kinematics.solve_ik(
                target_position,
                target_rotation,
                init_q=joints,
                max_iter=self.args.ik_max_iter,
            )
            if solution is None:
                self._warn("IK failed; keeping previous target (daemon holds)")
            else:
                action = [float(value) for value in solution] + [self._gripper_target(inputs)]
                self.last_action = action
                if self.args.dry_run:
                    if self.args.verbose:
                        print(
                            "[bridge] action(dry-run)="
                            + np.array2string(np.array(action), precision=3),
                            file=sys.stderr,
                        )
                else:
                    self.client.send_policy_action(action)

        if now - self.last_status_print >= 1.0:
            self.last_status_print = now
            self._print_status(inputs)

    def _print_status(self, inputs: dict) -> None:
        pose_rx = self.pose_sub.received_count if self.pose_sub else 0
        inputs_rx = self.inputs_sub.received_count if self.inputs_sub else 0
        anchor_state = "anchored" if self.anchor is not None else "idle"
        hand_pose = self.latest_hand_pose()
        hand = (
            " ".join(f"{value:+.2f}" for value in hand_pose["position"])
            if hand_pose is not None
            else "-"
        )
        print(
            f"[bridge] status: {anchor_state} pose_rx={pose_rx} inputs_rx={inputs_rx} "
            f"B={int(inputs['button_lower'])} trigger={inputs['press_index']:.2f} "
            f"hand(xyz)=[{hand}] "
            f"last_action={' '.join(f'{v:.2f}' for v in self.last_action) if self.last_action else '-'}",
            file=sys.stderr,
        )

    def _shutdown(self) -> None:
        self.stop_inputs()
        if self.args.dry_run:
            return
        try:
            self.client.hold()
            print("[bridge] requested daemon hold", file=sys.stderr)
        except ControlClientError as exc:
            print(f"[bridge] warning: could not request hold: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Simulated input (level-1 mock testing without the Quest / ROS stack)
# ---------------------------------------------------------------------------


def _simulated_hand_pose() -> dict:
    t = time.monotonic()
    amplitude = 0.05
    return {
        "position": [0.2 + amplitude * math.sin(t * 0.8), 0.0, 0.1 + amplitude * math.sin(t * 0.5)],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _simulated_inputs() -> dict:
    t = time.monotonic()
    return {
        "button_lower": True,
        "button_upper": False,
        "press_index": 0.5 + 0.5 * math.sin(t * 0.7),
    }


# ---------------------------------------------------------------------------
# Helpers / CLI
# ---------------------------------------------------------------------------


def _quaternion_to_matrix(quaternion_xyzw) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=float)).as_matrix()


def _parse_axis_signs(value) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value).split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("axis signs must contain 3 comma-separated values")
    signs = tuple(float(part) for part in parts)
    if any(sign not in (1.0, -1.0) for sign in signs):
        raise argparse.ArgumentTypeError("axis signs must be 1 or -1")
    return signs


def _parse_axis_map(value) -> tuple[str, str, str]:
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip().lower() for part in value]
    else:
        parts = [part.strip().lower() for part in str(value).split(",")]
    if len(parts) != 3 or any(part not in ("x", "y", "z") for part in parts) or len(set(parts)) != 3:
        raise argparse.ArgumentTypeError("axis map must be a permutation of 'x,y,z', e.g. 'y,z,x'")
    return tuple(parts)


def _parse_joint_values(value) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value).split(",")
    if len(parts) != DEFAULT_JOINT_COUNT:
        raise argparse.ArgumentTypeError(f"expected {DEFAULT_JOINT_COUNT} comma-separated joint values")
    joints = tuple(float(part) for part in parts)
    if any(not math.isfinite(joint) for joint in joints):
        raise argparse.ArgumentTypeError("joint values must be finite")
    return joints


def _resolve_follower_config(args) -> Path:
    deploy_root = REPO_ROOT.parents[2]
    if args.follower_config:
        p = Path(args.follower_config)
        if not p.is_absolute():
            anchored = deploy_root / p
            if anchored.exists():
                return anchored
        return p
    # 标定配置属 arm 域(<deploy>/arm/panthera/robot_param,serial_id 已按本机串口布局改对),
    # 没有才退回 SDK 自带的出厂配置(serial_id 1/2,仅适配原验收工作站)。
    panthera_root = deploy_root / "arm" / "panthera"
    local_config = panthera_root / "robot_param" / "Follower.yaml"
    if local_config.exists():
        return local_config
    sdk_root = args.sdk_root or os.environ.get("PANTHERA_SDK_ROOT") or str(panthera_root / "sdk" / "Panthera-HT_SDK")
    candidate = Path(sdk_root)
    if (candidate / "panthera_python").exists():
        candidate = candidate / "panthera_python"
    return candidate / "robot_param" / "Follower.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quest3S -> Panthera policy-mode teleoperation bridge")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--socket", default="/tmp/panthera_lerobot_control.sock")
    parser.add_argument("--hz", type=float, default=50.0, help="action send rate")
    parser.add_argument("--timeout-s", type=float, default=1.0, help="daemon IPC timeout")
    parser.add_argument("--snapshot-max-age-s", type=float, default=0.2)
    parser.add_argument("--input-timeout-s", type=float, default=0.5, help="stale Quest input threshold")
    parser.add_argument("--scale", type=float, default=1.0, help="translation scale controller->arm")
    parser.add_argument(
        "--axis-map",
        type=_parse_axis_map,
        default=("x", "y", "z"),
        help="which Quest axis feeds each robot axis 'X,Y,Z'. Both frames measured on hardware: "
        "Quest x=forward/y=left/z=up; robot X=forward/Y=left/Z=up. Default is the identity "
        "'x,y,z' - only change for a differently mounted setup",
    )
    parser.add_argument(
        "--wrist-map",
        type=_parse_axis_map,
        default=("x", "y", "z"),
        help="which hand-local axis feeds each tool-local rotation axis 'x,y,z'. Default "
        "identity: wrist twist -> joint6 (gripper tilt), wrist pitch -> fore/aft tilt, "
        "wrist yaw -> joint5 (gripper swings left/right)",
    )
    parser.add_argument(
        "--axis-signs",
        type=lambda value: _parse_axis_signs(value),
        default=(1.0, 1.0, 1.0),
        help="per-robot-axis sign flip 'X,Y,Z' applied after --axis-map; flip an axis when that "
        "robot direction is inverted",
    )
    parser.add_argument("--max-reach", type=float, default=0.30, help="clamp target displacement from anchor (m)")
    parser.add_argument("--gripper-mode", choices=("trigger", "toggle"), default="trigger")
    parser.add_argument("--ik-max-iter", type=int, default=60)
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="keep end-effector orientation fixed at the anchor pose; hand rotation is ignored "
        "(easier for pick-and-place: translation + trigger only)",
    )
    parser.add_argument(
        "--home-joints",
        type=_parse_joint_values,
        default=DEFAULT_HOME_JOINTS,
        help="home pose joints 'j1,...,j6': tool looks straight down at the tabletop, "
        "end effector ~0.35m in front of the base, gripper open",
    )
    parser.add_argument("--start-pose", choices=("ask", "home", "current"), default="ask",
                        help="startup pose: ask (interactive, default), home, or current")
    parser.add_argument("--home-move-s", type=float, default=4.0, help="seconds for the home move ramp")
    parser.add_argument("--home-countdown-s", type=int, default=3, help="countdown before the home move")
    parser.add_argument("--warn-interval-s", type=float, default=2.0)
    parser.add_argument("--sdk-root", default=None)
    parser.add_argument("--follower-config", default=None)
    parser.add_argument("--mock", action="store_true", help="expect a mock daemon")
    parser.add_argument("--sim-input", action="store_true", help="synthetic hand input; no ROS required")
    parser.add_argument("--dry-run", action="store_true", help="compute actions and log without sending")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bridge = QuestBridge(args)
    return bridge.run()


if __name__ == "__main__":
    raise SystemExit(main())
