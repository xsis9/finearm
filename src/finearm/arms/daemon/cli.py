"""Arm-neutral CLI entry point for the control daemon (``finearm-daemon``).

``--arm <id>`` selects the arm plugin (directory scan under
``finearm/arms/``); dimensions, socket default, and the serial-access
preflight come from that plugin. All control flags are inherited unchanged
from the panthera_control CLI (same names, same defaults).
"""

from __future__ import annotations

import argparse
import math
import signal
import time

from .. import DEFAULT_ARM, available, load_arm
from .controller import ControlConfig, ControlDaemon


def build_parser(arm: str = DEFAULT_ARM, prog: str = "finearm-daemon") -> argparse.ArgumentParser:
    plugin = load_arm(arm)
    profile = plugin.PROFILE
    joint_count = profile.joint_count

    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--arm", default=arm, choices=[arm, *sorted(set(available()) - {arm})],
                        help="arm plugin id (directory under finearm/arms/)")
    parser.add_argument("--mock", action="store_true", help="run without arm hardware")
    parser.add_argument(
        "--no-leader",
        action="store_true",
        help="run with follower hardware only (virtual leader); teleop control mode is rejected",
    )
    parser.add_argument("--socket", default=profile.socket_path, help="Unix socket path")
    parser.add_argument("--loop-hz", type=float, default=1000.0, help="control loop rate")
    parser.add_argument("--sdk-root", default=None, help="vendor SDK root path")
    parser.add_argument("--leader-config", default=None, help="leader YAML path")
    parser.add_argument("--follower-config", default=None, help="follower YAML path")
    parser.add_argument("--policy-action-timeout-s", type=_positive_float, default=0.25, help="policy action freshness threshold")
    parser.add_argument(
        "--initial-control-mode",
        choices=("hold", "teleop", "policy", "freedrive"),
        default="hold",
        help="control mode used from the first hardware loop tick",
    )
    parser.add_argument("--policy-mirror-leader", action="store_true", help="mirror POLICY TARGETS to the leader directly (both arms servo the same commanded target; not follower-pose cascading)")
    parser.add_argument("--policy-mirror-max-step-rad", type=_positive_float, default=0.02, help="max leader joint target step away from the leader's current pose per tick (engagement smoothing + torque cap ≈ kp×this)")
    parser.add_argument("--policy-mirror-max-gripper-step", type=_positive_float, default=0.04, help="max leader gripper target step per control tick")
    parser.add_argument(
        "--policy-mirror-leader-kp",
        type=lambda value: _gain_values(value, joint_count),
        default=None,
        help="leader joint servo kp, single value or N comma-separated. Default 0.3 is "
        "guidance-only and cannot move a real arm. Recommended = follower gains.",
    )
    parser.add_argument(
        "--policy-mirror-leader-kd",
        type=lambda value: _gain_values(value, joint_count),
        default=None,
        help="leader joint servo kd, single value or N comma-separated.",
    )
    parser.add_argument("--policy-mirror-gripper-kp", type=_non_negative_float, default=None, help="leader gripper servo kp")
    parser.add_argument("--policy-mirror-gripper-kd", type=_non_negative_float, default=None, help="leader gripper servo kd")
    parser.add_argument(
        "--teleop-target-lead-rad",
        type=_positive_float,
        default=0.25,
        help="teleop: max COMMAND ramp step per tick (vs the previous command, not the follower "
        "pose — pose-referenced limiting caps holding torque and the follower sags). "
        "0.008 = 8 rad/s at 1kHz: takeover divergence catches up smoothly instead of snapping; "
        "steady-state command equals the leader pose so holding force is unaffected",
    )
    parser.add_argument(
        "--no-park",
        action="store_true",
        help="skip the exit park move (live mode parks back to the startup pose before braking)",
    )
    parser.add_argument(
        "--park-joints",
        type=lambda value: _joint_values(value, joint_count),
        default=None,
        help="override the park target with explicit 'j1,...,jN' instead of the startup pose",
    )
    parser.add_argument("--park-move-s", type=_positive_float, default=4.0, help="seconds for the exit park ramp")
    parser.add_argument(
        "--no-leader-park",
        action="store_true",
        help="skip the leader exit park move (live mode ramps the leader arm to zero joints with soft gains before the motors release, so the floating leader does not collapse)",
    )
    parser.add_argument("--leader-park-move-s", type=_positive_float, default=4.0, help="seconds for the leader exit park ramp")
    parser.add_argument(
        "--teleop-gripper-feedback-scale",
        type=_non_negative_float,
        default=0.5,
        help="teleop: how much follower gripper torque is fed back to the leader gripper (smaller = easier to squeeze)",
    )
    parser.add_argument(
        "--teleop-gripper-squeeze-boost",
        type=_non_negative_float,
        default=0.0,
        help="teleop: extra closing force when the follower gripper is blocked by an object "
        "(0 disables; try 0.5-1.0; grows grip force with how far the leader keeps squeezing)",
    )
    parser.add_argument(
        "--teleop-gripper-squeeze-max",
        type=_positive_float,
        default=0.4,
        help="teleop: cap on the squeeze-boost virtual target overshoot below the gripper lower limit",
    )
    return parser


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _joint_values(value: str, joint_count: int) -> list[float]:
    parts = str(value).split(",")
    if len(parts) != joint_count:
        raise argparse.ArgumentTypeError(f"expected {joint_count} comma-separated joint values")
    joints = [float(part) for part in parts]
    if any(not math.isfinite(joint) for joint in joints):
        raise argparse.ArgumentTypeError("joint values must be finite")
    return joints


def _gain_values(value: str, joint_count: int) -> list[float]:
    """单个数值(×N)或 N 个逗号分隔值,非负有限。"""
    parts = [float(part) for part in str(value).split(",")]
    if len(parts) == 1:
        parts = parts * joint_count
    if len(parts) != joint_count:
        raise argparse.ArgumentTypeError(f"expected 1 or {joint_count} comma-separated values")
    if any(not math.isfinite(gain) or gain < 0.0 for gain in parts):
        raise argparse.ArgumentTypeError("gain values must be finite and non-negative")
    return parts


def _startup_park_target(daemon: ControlDaemon) -> list[float] | None:
    """Capture the follower pose right after startup for the exit park move."""
    joint_count = daemon.joint_count
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        snapshot = daemon.latest_snapshot()
        if snapshot.get("sequence", 0) >= 1:
            state = [float(value) for value in snapshot["observation.state"][: joint_count + 1]]
            if all(abs(value) < 500.0 for value in state):  # reject the 999 "no data" sentinel
                return state
            return None
        time.sleep(0.01)
    return None


def plan_park_actions(start: list[float], target: list[float], gripper: float, steps: int) -> list[list[float]]:
    """Cosine-eased joint ramp from start to target (N joints + gripper)."""
    if len(start) != len(target):
        raise ValueError("park plan requires matching joint counts")
    steps = max(1, int(steps))
    actions = []
    for step in range(1, steps + 1):
        eased = 0.5 - 0.5 * math.cos(math.pi * step / steps)
        actions.append(
            [current + (goal - current) * eased for current, goal in zip(start, target)] + [float(gripper)]
        )
    return actions


def _park_before_stop(daemon: ControlDaemon, park_target: list[float], move_s: float) -> bool:
    """Ramp the follower back to park_target through policy actions. Best effort."""
    joint_count = daemon.joint_count
    snapshot = daemon.latest_snapshot()
    state = [float(value) for value in snapshot["observation.state"][: joint_count + 1]]
    if any(abs(value) >= 500.0 for value in state):
        print("park skipped: no valid follower state available")
        return False
    start, gripper = state[:joint_count], state[joint_count]
    gripper_target = park_target[joint_count] if len(park_target) > joint_count else gripper
    print(
        "parking to joints "
        f"{[round(value, 3) for value in park_target[:joint_count]]} (~{move_s:.0f}s); "
        "Ctrl+C is ignored while parking"
    )
    actions = plan_park_actions(start, park_target[:joint_count], gripper_target, steps=int(move_s * 50))
    daemon.set_control_mode("policy")
    for action in actions:
        daemon.submit_policy_action(action)
        time.sleep(0.02)
    return True


def main(argv: list[str] | None = None, *, prog: str = "finearm-daemon", default_arm: str = DEFAULT_ARM) -> int:
    # Probe --arm before building the full parser (joint-count validators and
    # the socket default come from the selected arm's profile).
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--arm", default=default_arm)
    probe_args, _ = probe.parse_known_args(argv)
    args = build_parser(arm=probe_args.arm, prog=prog).parse_args(argv)

    plugin = load_arm(args.arm)
    daemon = ControlDaemon(
        ControlConfig(
            arm_id=args.arm,
            mock=args.mock,
            no_leader=args.no_leader,
            socket_path=args.socket,
            loop_hz=args.loop_hz,
            sdk_root=args.sdk_root,
            leader_config=args.leader_config,
            follower_config=args.follower_config,
            policy_action_timeout_s=args.policy_action_timeout_s,
            initial_control_mode=args.initial_control_mode,
            policy_mirror_leader=args.policy_mirror_leader,
            policy_mirror_max_step_rad=args.policy_mirror_max_step_rad,
            policy_mirror_max_gripper_step=args.policy_mirror_max_gripper_step,
            **({"policy_mirror_leader_kp": args.policy_mirror_leader_kp} if args.policy_mirror_leader_kp is not None else {}),
            **({"policy_mirror_leader_kd": args.policy_mirror_leader_kd} if args.policy_mirror_leader_kd is not None else {}),
            **({"policy_mirror_gripper_kp": args.policy_mirror_gripper_kp} if args.policy_mirror_gripper_kp is not None else {}),
            **({"policy_mirror_gripper_kd": args.policy_mirror_gripper_kd} if args.policy_mirror_gripper_kd is not None else {}),
            teleop_target_lead_rad=args.teleop_target_lead_rad,
            teleop_gripper_feedback_scale=args.teleop_gripper_feedback_scale,
            teleop_gripper_squeeze_boost=args.teleop_gripper_squeeze_boost,
            teleop_gripper_squeeze_max=args.teleop_gripper_squeeze_max,
        )
    )
    stopping = False
    installed_handlers = []
    park_target: list[float] | None = None

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous = signal.getsignal(signum)
            signal.signal(signum, request_stop)
            installed_handlers.append((signum, previous))
        ensure_serial_access = getattr(plugin, "ensure_serial_access", None)
        if not args.mock and callable(ensure_serial_access):
            ensure_serial_access()
        daemon.start()
        if not args.mock and not args.no_park:
            if args.park_joints:
                park_target = list(args.park_joints) + [0.0]
            else:
                park_target = _startup_park_target(daemon)
                if park_target is None:
                    print("warning: follower state unavailable at startup; exit park is disabled")
        while not stopping:
            time.sleep(0.2)
    finally:
        try:
            if park_target is not None and not args.no_park:
                try:
                    _park_before_stop(daemon, park_target, args.park_move_s)
                except Exception as exc:
                    print(f"warning: exit park failed ({exc!r}); stopping directly")
            if not args.mock and not args.no_park and not args.no_leader_park and not args.no_leader:
                try:
                    print(
                        f"parking leader arm to zero joints (~{args.leader_park_move_s:.0f}s); "
                        "keep hands clear of the leader arm"
                    )
                    if not daemon.park_leader(args.leader_park_move_s):
                        print("warning: leader park did not complete in time")
                except Exception as exc:
                    print(f"warning: leader park failed ({exc!r})")
            daemon.stop()
        finally:
            for signum, previous in reversed(installed_handlers):
                signal.signal(signum, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
