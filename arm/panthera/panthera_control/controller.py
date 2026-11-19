"""Control loop and daemon lifecycle."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import constants as c
from .adapters import AdapterError, make_follower_only, make_mock_pair, make_panthera_pair
from .ipc import ControlServer


class PantheraControlError(RuntimeError):
    pass


CONTROL_MODES = {"teleop", "policy", "hold", "freedrive"}


@dataclass
class ControlConfig:
    mock: bool = False
    no_leader: bool = False
    socket_path: str = c.DEFAULT_SOCKET_PATH
    loop_hz: float = 1000.0
    sdk_root: Optional[str | Path] = None
    leader_config: Optional[str | Path] = None
    follower_config: Optional[str | Path] = None
    joint_min: list[float] = field(default_factory=lambda: list(c.DEFAULT_JOINT_MIN))
    joint_max: list[float] = field(default_factory=lambda: list(c.DEFAULT_JOINT_MAX))
    gripper_min: float = c.DEFAULT_GRIPPER_MIN
    gripper_max: float = c.DEFAULT_GRIPPER_MAX
    policy_action_timeout_s: float = 0.25
    initial_control_mode: str = "teleop"
    max_target_lead_rad: float = 0.25
    max_gripper_lead: float = 0.4
    teleop_target_lead_rad: float = 0.25
    policy_mirror_leader: bool = False
    policy_mirror_max_step_rad: float = 0.02
    policy_mirror_max_gripper_step: float = 0.04
    policy_mirror_leader_kp: list[float] = field(default_factory=lambda: [0.3] * c.JOINT_COUNT)
    policy_mirror_leader_kd: list[float] = field(default_factory=lambda: [0.03] * c.JOINT_COUNT)
    policy_mirror_gripper_kp: float = 0.4
    policy_mirror_gripper_kd: float = 0.04
    teleop_gripper_feedback_scale: float = 0.5
    teleop_gripper_squeeze_boost: float = 0.0
    teleop_gripper_squeeze_max: float = 0.4
    leader_gravity_gain: list[float] = field(default_factory=lambda: [1.0] * c.JOINT_COUNT)

    def __post_init__(self) -> None:
        self.loop_hz = _positive_finite(self.loop_hz, "loop_hz")
        if self.initial_control_mode not in CONTROL_MODES:
            raise ValueError(f"unknown initial control mode: {self.initial_control_mode!r}")
        if self.no_leader and self.initial_control_mode == "teleop":
            raise ValueError("no_leader mode cannot start in teleop control mode")
        self.policy_action_timeout_s = _positive_finite(self.policy_action_timeout_s, "policy_action_timeout_s")
        self.max_target_lead_rad = _positive_finite(self.max_target_lead_rad, "max_target_lead_rad")
        self.max_gripper_lead = _positive_finite(self.max_gripper_lead, "max_gripper_lead")
        self.teleop_target_lead_rad = _positive_finite(self.teleop_target_lead_rad, "teleop_target_lead_rad")
        self.policy_mirror_max_step_rad = _positive_finite(self.policy_mirror_max_step_rad, "policy_mirror_max_step_rad")
        self.policy_mirror_max_gripper_step = _positive_finite(
            self.policy_mirror_max_gripper_step,
            "policy_mirror_max_gripper_step",
        )
        self.policy_mirror_leader_kp = _finite_vector(
            self.policy_mirror_leader_kp,
            c.JOINT_COUNT,
            "policy_mirror_leader_kp",
        )
        self.policy_mirror_leader_kd = _finite_vector(
            self.policy_mirror_leader_kd,
            c.JOINT_COUNT,
            "policy_mirror_leader_kd",
        )
        self.policy_mirror_gripper_kp = _non_negative_finite(self.policy_mirror_gripper_kp, "policy_mirror_gripper_kp")
        self.policy_mirror_gripper_kd = _non_negative_finite(self.policy_mirror_gripper_kd, "policy_mirror_gripper_kd")
        self.teleop_gripper_feedback_scale = _non_negative_finite(
            self.teleop_gripper_feedback_scale,
            "teleop_gripper_feedback_scale",
        )
        self.teleop_gripper_squeeze_boost = _non_negative_finite(
            self.teleop_gripper_squeeze_boost,
            "teleop_gripper_squeeze_boost",
        )
        self.teleop_gripper_squeeze_max = _positive_finite(
            self.teleop_gripper_squeeze_max,
            "teleop_gripper_squeeze_max",
        )
        self.leader_gravity_gain = _finite_vector(
            self.leader_gravity_gain,
            c.JOINT_COUNT,
            "leader_gravity_gain",
        )


class ControlDaemon:
    def __init__(self, config: Optional[ControlConfig] = None, adapter_factory: Optional[Callable] = None):
        self.config = config or ControlConfig()
        self._adapter_factory = adapter_factory
        self._leader = None
        self._follower = None
        self._loop_thread = None
        self._server = None
        self._stop_event = threading.Event()
        self._hold_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._control_mode = self.config.initial_control_mode
        self._policy_action = None
        self._policy_action_timestamp = None
        self._policy_action_is_seed = False
        self._policy_sequence = 0
        self._teleop_cmd_positions = None  # teleop 指令限速的上一拍指令(None=进 teleop 时重新播种)
        self._sequence = 0
        self._last_valid_follower_state = None
        self._last_error = None
        self._leader_park: Optional[dict] = None
        self._leader_park_done = threading.Event()
        self._snapshot = c.make_snapshot(
            follower_positions=c.zero_joints(),
            follower_velocities=c.zero_joints(),
            follower_gripper_position=0.0,
            follower_gripper_velocity=0.0,
            target_positions=c.zero_joints(),
            target_gripper_position=0.0,
            diagnostics={
                "mode": self.mode,
                "control_mode": self.control_mode,
                "holding": False,
                "policy_stale": False,
                "policy_action_age_s": None,
            },
            timestamp=time.monotonic(),
            sequence=0,
            loop_timing={"target_period_s": self.period_s, "last_period_s": 0.0, "mean_period_s": 0.0},
        )

    @property
    def mode(self) -> str:
        return "mock" if self.config.mock else "live"

    @property
    def control_mode(self) -> str:
        with self._control_lock:
            return self._control_mode

    @property
    def period_s(self) -> float:
        if self.config.loop_hz <= 0.0:
            raise PantheraControlError("loop_hz must be positive")
        return 1.0 / float(self.config.loop_hz)

    def start(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._leader, self._follower = self._make_adapters()
        self._stop_event.clear()
        self._server = ControlServer(self.config.socket_path, self)
        try:
            self._server.start()
        except Exception:
            self._stop_adapters()
            self._server = None
            raise
        self._loop_thread = threading.Thread(target=self._run_loop, name="panthera-control-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._prepare_hold_for_stop()
        self._stop_event.set()
        if self._server:
            self._server.stop()
            self._server = None
        if self._loop_thread:
            if threading.current_thread() is not self._loop_thread:
                self._loop_thread.join(timeout=2.0)
            self._loop_thread = None
        self._apply_final_hold_before_adapter_stop()
        self._stop_adapters()

    def _prepare_hold_for_stop(self) -> None:
        if self._leader is None or self._follower is None:
            return
        with self._snapshot_lock:
            start_sequence = int(self._snapshot.get("sequence", 0))
        try:
            self.hold()
        except Exception:
            self._hold_event.set()
        loop_thread = self._loop_thread
        if loop_thread is None or not loop_thread.is_alive() or threading.current_thread() is loop_thread:
            return
        deadline = time.monotonic() + max(0.05, min(0.25, self.period_s * 5.0))
        sleep_s = min(max(self.period_s, 0.001), 0.01)
        while time.monotonic() < deadline:
            snapshot = self.latest_snapshot()
            diagnostics = snapshot.get("diagnostics", {})
            if snapshot.get("sequence", 0) > start_sequence and diagnostics.get("holding", False):
                return
            time.sleep(sleep_s)

    def _apply_final_hold_before_adapter_stop(self) -> None:
        if self._leader is None or self._follower is None:
            return
        now = time.monotonic()
        self._apply_hold(now, 0.0, 0.0)

    def _stop_adapters(self) -> None:
        for robot in (self._leader, self._follower):
            if robot:
                try:
                    robot.stop()
                except Exception:
                    pass
        self._leader = None
        self._follower = None

    def hold(self) -> None:
        self.set_control_mode("hold")

    def park_leader(self, move_s: float = 4.0, target: Optional[list] = None) -> bool:
        """Ramp the leader arm to ``target`` joints with soft position control.

        Used on daemon exit so the leader (normally floating on gravity
        compensation) folds back to a compact pose before the motors release,
        instead of collapsing under gravity. Blocks until the ramp completes.
        """
        if self._leader is None or self.config.no_leader:
            return False
        park_target = list(target) if target is not None else c.zero_joints()
        if len(park_target) != c.JOINT_COUNT:
            raise PantheraControlError("leader park target requires exactly JOINT_COUNT joint values")
        duration = max(0.05, float(move_s))
        self._leader_park_done.clear()
        self._leader_park = {
            "target": [float(value) for value in park_target],
            "duration": duration,
            "start": None,
            "t0": None,
        }
        completed = self._leader_park_done.wait(timeout=duration * 2.0 + 1.0)
        self._leader_park = None
        return completed

    def set_control_mode(self, mode: str) -> None:
        if mode not in CONTROL_MODES:
            raise PantheraControlError(f"unknown control mode: {mode!r}")
        if mode == "teleop" and self.config.no_leader:
            raise PantheraControlError("teleop control mode requires leader hardware; daemon runs in no_leader mode")
        seed_policy_action = None
        if mode == "policy":
            with self._snapshot_lock:
                state = self._snapshot.get("observation.state", [])
            if isinstance(state, list) and len(state) >= c.JOINT_COUNT + 1:
                try:
                    seed_policy_action = _validate_policy_action(state[: c.JOINT_COUNT + 1])
                except PantheraControlError:
                    seed_policy_action = None
        with self._control_lock:
            self._control_mode = mode
            self._teleop_cmd_positions = None  # 模式切换后重进 teleop 需重新播种起步位姿
            if mode == "policy":
                self._policy_action = seed_policy_action
                if seed_policy_action is None:
                    self._policy_action_timestamp = None
                    self._policy_action_is_seed = False
                else:
                    self._policy_action_timestamp = time.monotonic()
                    self._policy_action_is_seed = True
                    self._policy_sequence += 1
        if mode == "hold":
            self._hold_event.set()
        else:
            self._hold_event.clear()

    def submit_policy_action(self, action) -> None:
        values = _validate_policy_action(action)
        with self._control_lock:
            self._policy_action = values
            self._policy_action_timestamp = time.monotonic()
            self._policy_action_is_seed = False
            self._policy_sequence += 1

    def latest_snapshot(self) -> dict:
        with self._snapshot_lock:
            return _deep_copy_snapshot(self._snapshot)

    def status(self) -> dict:
        snapshot = self.latest_snapshot()
        policy_status = self._policy_status(time.monotonic())
        diagnostics = snapshot.get("diagnostics", {})
        return {
            "running": bool(self._loop_thread and self._loop_thread.is_alive()),
            "mode": self.mode,
            "control_mode": self.control_mode,
            "holding": diagnostics.get("holding") is True,
            "sequence": snapshot["sequence"],
            "timestamp": snapshot["timestamp"],
            "last_error": self._last_error,
            "socket_path": self.config.socket_path,
            "policy_action_age_s": policy_status["age_s"],
            "policy_sequence": policy_status["sequence"],
            "policy_stale": policy_status["stale"],
        }

    def _make_adapters(self):
        if self._adapter_factory:
            return self._adapter_factory()
        if self.config.mock:
            return make_mock_pair()
        try:
            if self.config.no_leader:
                return make_follower_only(
                    sdk_root=self.config.sdk_root,
                    follower_config=self.config.follower_config,
                )
            return make_panthera_pair(
                sdk_root=self.config.sdk_root,
                leader_config=self.config.leader_config,
                follower_config=self.config.follower_config,
            )
        except AdapterError as exc:
            raise PantheraControlError(str(exc)) from exc

    def _run_loop(self) -> None:
        next_tick = time.monotonic()
        last_tick = next_tick
        mean_period = 0.0
        count = 0
        while not self._stop_event.is_set():
            now = time.monotonic()
            last_period = now - last_tick
            last_tick = now
            try:
                self._step(now, last_period, mean_period)
            except Exception as exc:
                self._append_last_error(exc)
                self._hold_event.set()
                self._apply_hold(now, last_period, mean_period)
            count += 1
            mean_period += (last_period - mean_period) / count
            next_tick += self.period_s
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def _step(self, now: float, last_period: float, mean_period: float) -> None:
        leader = self._leader.read_state()
        follower = self._follower.read_state()
        self._last_valid_follower_state = copy.deepcopy(follower)
        teleop_positions = c.clamp(leader.positions, self.config.joint_min, self.config.joint_max)
        teleop_gripper = min(max(float(leader.gripper_position), self.config.gripper_min), self.config.gripper_max)

        leader_friction = self._leader.friction(leader.velocities, c.FC, c.FV, c.VEL_THRESHOLD)
        follower_friction = self._follower.friction(follower.velocities, c.FC, c.FV, c.VEL_THRESHOLD)
        torque_feedback = c.threshold(c.sub(follower.torques, follower.gravity), c.TOR_THRESHOLD)
        leader_torque = c.clip(c.add(leader.gravity, leader_friction), c.TAU_LIMIT)
        follower_torque = c.clip(c.add(follower.gravity, follower_friction), c.TAU_LIMIT)
        gripper_torque = self._gripper_friction(leader.gripper_velocity) - (
            follower.gripper_torque * self.config.teleop_gripper_feedback_scale
        )

        control = self._policy_status(now)
        control_mode = self.control_mode
        policy_stale = False
        policy_target_lead_limited = False
        policy_mirror_leader = False
        policy_mirror_limited = False
        teleop_target_limited = False
        policy_action_age_s = control["age_s"]
        if self._hold_event.is_set() or control_mode == "hold":
            target_positions = follower.positions
            target_gripper = follower.gripper_position
            target_velocities = c.zero_joints()
            target_gripper_velocity = 0.0
            holding = True
        elif control_mode == "policy":
            if control["action"] is None or control["stale"]:
                target_positions = follower.positions
                target_gripper = follower.gripper_position
                target_velocities = c.zero_joints()
                target_gripper_velocity = 0.0
                holding = True
                policy_stale = True
            else:
                desired_positions = c.clamp(control["action"][: c.JOINT_COUNT], self.config.joint_min, self.config.joint_max)
                desired_gripper = min(
                    max(float(control["action"][c.JOINT_COUNT]), self.config.gripper_min),
                    self.config.gripper_max,
                )
                target_positions, joints_limited = _limit_target_lead(
                    desired_positions,
                    follower.positions,
                    self.config.max_target_lead_rad,
                )
                target_gripper, gripper_limited = _limit_scalar_lead(
                    desired_gripper,
                    follower.gripper_position,
                    self.config.max_gripper_lead,
                )
                policy_target_lead_limited = joints_limited or gripper_limited
                target_velocities = c.zero_joints()
                target_gripper_velocity = 0.0
                holding = False
        elif control_mode == "freedrive":
            # Hand-teaching: zero position/velocity gains, feed forward gravity
            # and friction compensation only. Targets track the live follower
            # pose so the recorded action reflects the taught configuration.
            target_positions = follower.positions
            target_gripper = follower.gripper_position
            target_velocities = c.zero_joints()
            target_gripper_velocity = 0.0
            holding = False
        else:
            target_gripper = teleop_gripper
            # 接管平滑(修正版):对"指令信号"限速——每 tick 指令最多前进
            # teleop_target_lead_rad(相对上一拍指令,而非从臂实测位姿)。
            # ⚠ 不能相对从臂位姿限幅(2026-11-09 第一版教训):那会把伺服
            # 保持力矩封顶在 kp×限幅(0.008→0.17N·m,小于重力残差),从臂
            # 直接垂下去且永远追不上主臂。指令限速则保持力矩不受限
            # (稳态指令=主臂位姿),接管分叉时从臂按 ≤限幅/tick 的速度追上。
            if self._teleop_cmd_positions is None:
                self._teleop_cmd_positions = list(follower.positions)  # 进 teleop 时从从臂当前位姿起步
            target_positions, teleop_target_limited = _limit_target_lead(
                teleop_positions,
                self._teleop_cmd_positions,
                self.config.teleop_target_lead_rad,
            )
            self._teleop_cmd_positions = list(target_positions)
            # Squeeze boost: when the follower gripper is blocked open by an
            # object while the operator keeps squeezing (leader target below
            # the follower pose), deepen the virtual target so the position
            # error - and therefore the grip force through kp - grows. With an
            # empty gripper the follower reaches the target, the closing error
            # is zero and no extra force is applied. The overshoot is capped so
            # the virtual target never drops more than squeeze_max below the
            # gripper lower limit.
            if self.config.teleop_gripper_squeeze_boost > 0.0:
                closing_error = max(0.0, float(follower.gripper_position) - target_gripper)
                if closing_error > 0.0:
                    overshoot = min(
                        self.config.teleop_gripper_squeeze_boost * closing_error,
                        self.config.teleop_gripper_squeeze_max,
                    )
                    target_gripper = max(
                        target_gripper - overshoot,
                        self.config.gripper_min - self.config.teleop_gripper_squeeze_max,
                    )
            target_velocities = leader.velocities
            target_gripper_velocity = leader.gripper_velocity
            holding = False

        leader_arm_position = c.zero_joints()
        leader_arm_velocity = c.zero_joints()
        leader_arm_torque = c.zero_joints()
        leader_arm_kp = c.zero_joints()
        leader_arm_kd = c.zero_joints()
        if control_mode == "teleop" and not holding:
            leader_arm_torque = leader_torque
            leader_gripper_position = 1.5
            leader_gripper_torque = gripper_torque
            leader_gripper_kp = 0.2
            leader_gripper_kd = 0.02
        elif control_mode == "policy" and self.config.policy_mirror_leader and not holding and not control["seed"]:
            # 直接把 policy 下发目标同时发给主臂(与从臂同一指令、各自伺服),
            # 不做"从臂实测位姿→主臂"的级联(弹簧+摩擦导致主臂带不动,2026-11-09 实测)。
            # 步距限幅只用于平滑接入,限制的是目标相对主臂当前位置的跳变。
            leader_arm_position, joints_limited = _limit_target_lead(
                target_positions,
                leader.positions,
                self.config.policy_mirror_max_step_rad,
            )
            leader_gripper_position, gripper_limited = _limit_scalar_lead(
                target_gripper,
                leader.gripper_position,
                self.config.policy_mirror_max_gripper_step,
            )
            leader_arm_velocity = c.zero_joints()  # 与从臂一致:policy 模式速度目标=0,靠位置伺服
            leader_arm_kp = c.as_floats(self.config.policy_mirror_leader_kp, c.JOINT_COUNT)
            leader_arm_kd = c.as_floats(self.config.policy_mirror_leader_kd, c.JOINT_COUNT)
            leader_gripper_torque = 0.0
            leader_gripper_kp = float(self.config.policy_mirror_gripper_kp)
            leader_gripper_kd = float(self.config.policy_mirror_gripper_kd)
            policy_mirror_leader = True
            policy_mirror_limited = joints_limited or gripper_limited
        else:
            leader_gripper_position = leader.gripper_position
            leader_gripper_torque = 0.0
            leader_gripper_kp = 0.0
            leader_gripper_kd = 0.0

        if self._leader_park is not None:
            # Exit ramp for the leader arm: soft position control (the same
            # gains as policy mirroring) plus the gravity/friction feedforward
            # computed above, easing along a cosine from the pose captured at
            # ramp start to the park target. Once the ramp completes the arm
            # sits at a compact pose and the normal stop path (neutralize +
            # adapters stop / brake) can release the motors safely.
            park = self._leader_park
            if park["start"] is None:
                park["start"] = list(leader.positions)
                park["t0"] = now
            progress = min(1.0, (now - park["t0"]) / park["duration"])
            eased = 0.5 - 0.5 * math.cos(math.pi * progress)
            leader_arm_position = [
                start + (goal - start) * eased
                for start, goal in zip(park["start"], park["target"])
            ]
            leader_arm_velocity = c.zero_joints()
            leader_arm_torque = leader_torque
            leader_arm_kp = c.as_floats(self.config.policy_mirror_leader_kp, c.JOINT_COUNT)
            leader_arm_kd = c.as_floats(self.config.policy_mirror_leader_kd, c.JOINT_COUNT)
            leader_gripper_torque = 0.0
            leader_gripper_kp = 0.0
            leader_gripper_kd = 0.0
            if progress >= 1.0:
                self._leader_park = None
                self._leader_park_done.set()

        self._leader.command_arm(leader_arm_position, leader_arm_velocity, leader_arm_torque, leader_arm_kp, leader_arm_kd)
        freedrive = control_mode == "freedrive" and not holding
        if freedrive:
            # Zero kp/kd disables the position loop; the arm floats on the
            # gravity+friction feedforward and can be moved by hand. The
            # gripper is commanded limp as well so it can be squeezed by hand.
            self._follower.command_arm(target_positions, c.zero_joints(), follower_torque, c.zero_joints(), c.zero_joints())
            self._follower.command_gripper(target_gripper, 0.0, 0.0, 0.0, 0.0)
        else:
            self._follower.command_arm(target_positions, target_velocities, follower_torque, c.KP, c.KD)
            self._follower.command_gripper(target_gripper, target_gripper_velocity, 0.0, c.GRIPPER_KP, c.GRIPPER_KD)
        self._leader.command_gripper(leader_gripper_position, 0.0, leader_gripper_torque, leader_gripper_kp, leader_gripper_kd)

        self._store_snapshot(
            follower=follower,
            target_positions=target_positions,
            target_gripper=target_gripper,
            now=now,
            last_period=last_period,
            mean_period=mean_period,
            diagnostics={
                "mode": self.mode,
                "control_mode": control_mode,
                "holding": holding,
                "policy_stale": policy_stale,
                "policy_target_lead_limited": policy_target_lead_limited,
                "policy_mirror_leader": policy_mirror_leader,
                "policy_mirror_limited": policy_mirror_limited,
                "policy_action_age_s": policy_action_age_s,
                "policy_action_is_seed": control["seed"],
                "policy_sequence": control["sequence"],
                "torque_feedback": torque_feedback,
                "leader_torque": leader_torque,
                "follower_torque": follower_torque,
                "gripper_torque": gripper_torque,
                # 主臂遥测(2026-11-09 加):位姿/下发目标/重力前馈,供主从分叉诊断
                "leader_positions": list(leader.positions),
                "leader_gripper_position": float(leader.gripper_position),
                "leader_cmd_positions": list(leader_arm_position),
                "leader_cmd_gripper": float(leader_gripper_position),
                "leader_gravity": list(leader.gravity),
                "teleop_target_limited": teleop_target_limited,
                "last_error": self._last_error,
            },
        )

    def _apply_hold(self, now: float, last_period: float, mean_period: float) -> bool:
        if self._leader_park is None:
            self._neutralize_leader_best_effort()
        follower = None
        try:
            follower = self._follower.read_state()
            self._last_valid_follower_state = copy.deepcopy(follower)
        except Exception as exc:
            self._append_last_error(exc)
            if self._last_valid_follower_state is not None:
                follower = copy.deepcopy(self._last_valid_follower_state)

        if follower is None:
            self._append_last_error(PantheraControlError("no valid follower pose is available for fail-safe hold"))
            return False

        try:
            target_positions = c.clamp(follower.positions, self.config.joint_min, self.config.joint_max)
            target_gripper = min(max(float(follower.gripper_position), self.config.gripper_min), self.config.gripper_max)
            self._follower.command_arm(target_positions, c.zero_joints(), c.zero_joints(), c.KP, c.KD)
            self._follower.command_gripper(target_gripper, 0.0, 0.0, c.GRIPPER_KP, c.GRIPPER_KD)
        except Exception as exc:
            self._append_last_error(exc)
            return False

        policy_status = self._policy_status(now)
        self._store_snapshot(
            follower=follower,
            target_positions=target_positions,
            target_gripper=target_gripper,
            now=now,
            last_period=last_period,
            mean_period=mean_period,
            diagnostics={
                "mode": self.mode,
                "control_mode": self.control_mode,
                "holding": True,
                "policy_stale": policy_status["stale"],
                "policy_action_age_s": policy_status["age_s"],
                "policy_action_is_seed": policy_status["seed"],
                "policy_mirror_leader": False,
                "policy_mirror_limited": False,
                "policy_sequence": policy_status["sequence"],
                "last_error": self._last_error,
            },
        )
        return True

    def _append_last_error(self, exc: Exception) -> None:
        message = repr(exc)
        if self._last_error is None:
            self._last_error = message
        elif message not in self._last_error:
            self._last_error = f"{self._last_error}; fail-safe hold: {message}"
        with self._snapshot_lock:
            diagnostics = self._snapshot.setdefault("diagnostics", {})
            diagnostics["last_error"] = self._last_error

    def _neutralize_leader_best_effort(self) -> None:
        if self._leader is None:
            return
        gripper_position = 0.0
        try:
            gripper_position = float(self._leader.read_state().gripper_position)
        except Exception as exc:
            self._append_last_error(exc)
        try:
            self._leader.command_arm(c.zero_joints(), c.zero_joints(), c.zero_joints(), c.zero_joints(), c.zero_joints())
        except Exception as exc:
            self._append_last_error(exc)
        try:
            self._leader.command_gripper(gripper_position, 0.0, 0.0, 0.0, 0.0)
        except Exception as exc:
            self._append_last_error(exc)

    def _store_snapshot(self, *, follower, target_positions, target_gripper, now, last_period, mean_period, diagnostics):
        self._sequence += 1
        snapshot = c.make_snapshot(
            follower_positions=follower.positions,
            follower_velocities=follower.velocities,
            follower_gripper_position=follower.gripper_position,
            follower_gripper_velocity=follower.gripper_velocity,
            target_positions=target_positions,
            target_gripper_position=target_gripper,
            diagnostics=diagnostics,
            timestamp=now,
            sequence=self._sequence,
            loop_timing={
                "target_period_s": self.period_s,
                "last_period_s": float(last_period),
                "mean_period_s": float(mean_period),
            },
        )
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _policy_status(self, now: float) -> dict:
        with self._control_lock:
            action = None if self._policy_action is None else list(self._policy_action)
            timestamp = self._policy_action_timestamp
            sequence = self._policy_sequence
            seed = self._policy_action_is_seed
        if timestamp is None:
            return {"action": action, "age_s": None, "sequence": sequence, "stale": True, "seed": seed}
        age_s = max(0.0, float(now) - float(timestamp))
        timeout_s = _positive_finite(self.config.policy_action_timeout_s, "policy_action_timeout_s")
        stale = age_s > timeout_s
        return {"action": action, "age_s": age_s, "sequence": sequence, "stale": stale, "seed": seed}

    def _gripper_friction(self, velocity: float) -> float:
        velocity = float(velocity)
        if abs(velocity) < 0.15:
            return 0.0
        return (1.0 if velocity > 0.0 else -1.0) * 0.06


def _limit_target_lead(values, current_values, max_lead: float) -> tuple[list[float], bool]:
    limited = []
    clipped = False
    for value, current in zip(values, current_values):
        bounded, was_clipped = _limit_scalar_lead(value, current, max_lead)
        limited.append(bounded)
        clipped = clipped or was_clipped
    return limited, clipped


def _limit_scalar_lead(value: float, current: float, max_lead: float) -> tuple[float, bool]:
    value = float(value)
    current = float(current)
    lead = float(max_lead)
    if not math.isfinite(lead) or lead <= 0.0:
        raise PantheraControlError("max lead must be finite and positive")
    delta = value - current
    if abs(delta) <= lead:
        return value, False
    return current + math.copysign(lead, delta), True


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _non_negative_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _finite_vector(values, length: int, name: str) -> list[float]:
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if len(parsed) != length:
        raise ValueError(f"{name} must contain {length} values")
    if any(not math.isfinite(value) or value < 0.0 for value in parsed):
        raise ValueError(f"{name} values must be finite and non-negative")
    return parsed


def _validate_policy_action(action) -> list[float]:
    if not isinstance(action, (list, tuple)):
        raise PantheraControlError("policy action must be a list or tuple")
    if len(action) != c.JOINT_COUNT + 1:
        raise PantheraControlError(f"policy action must contain {c.JOINT_COUNT + 1} values")
    values = []
    for value in action:
        if isinstance(value, bool):
            raise PantheraControlError("policy action values must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PantheraControlError("policy action values must be numeric") from exc
        if not math.isfinite(numeric):
            raise PantheraControlError("policy action values must be finite")
        values.append(numeric)
    return values


def _deep_copy_snapshot(snapshot: dict) -> dict:
    return copy.deepcopy(snapshot)
