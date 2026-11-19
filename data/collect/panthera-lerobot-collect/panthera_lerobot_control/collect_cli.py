"""One-command LeRobot collection wrapper for Panthera HT."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import shutil
import queue
import re
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

from .adapters import AdapterError, ensure_panthera_serial_access
from .constants import DEFAULT_SOCKET_PATH
from .ipc import ControlClient, ControlClientError


DEFAULT_CAMERA_ROLE_NAMES = ("front", "wrist")
CAMERA_SELECTION_PATH = Path(".panthera_camera_selection.json")
DEFAULT_REPO_ID = "local/panthera_smolvla_demo"
DEFAULT_TASK = "panthera teleoperation"
DEFAULT_CAMERA_TEST_TIMEOUT_S = 3.0
SUPPORTED_LEROBOT_VERSION = "0.5.1"
PANTHERA_OBSERVATION_NAMES = (
    "follower_joint_1.pos", "follower_joint_2.pos", "follower_joint_3.pos",
    "follower_joint_4.pos", "follower_joint_5.pos", "follower_joint_6.pos",
    "follower_gripper.pos", "follower_joint_1.vel", "follower_joint_2.vel",
    "follower_joint_3.vel", "follower_joint_4.vel", "follower_joint_5.vel",
    "follower_joint_6.vel", "follower_gripper.vel",
)
PANTHERA_ACTION_NAMES = (
    "target_joint_1.pos", "target_joint_2.pos", "target_joint_3.pos",
    "target_joint_4.pos", "target_joint_5.pos", "target_joint_6.pos",
    "target_gripper.pos",
)
def _default_data_root() -> Path:
    """Dataset hub root: <deploy>/data/datasets (two levels above this tool dir)."""
    return Path(__file__).resolve().parents[3] / "datasets"


_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DATASET_ROOT_COUNTER = 0
_DATASET_ROOT_COUNTER_LOCK = threading.Lock()



class CollectError(RuntimeError):
    pass


def run_collection_preflight(args: argparse.Namespace, *, require_runtime: bool = True) -> None:
    _validate_lerobot_passthrough_args(args.lerobot_arg)
    if not str(args.repo_id).strip():
        raise CollectError("--repo-id must not be empty")
    if not str(args.task).strip():
        raise CollectError("--task must not be empty")
    if args.print_command and args.select_cameras:
        raise CollectError("--print-command cannot be combined with --select-cameras; pass explicit --camera values instead")
    if args.select_cameras and args.no_gui:
        raise CollectError("--select-cameras requires the GUI; pass explicit --camera NAME=SOURCE values instead.")
    _prepare_collection_dataset(args)
    if not require_runtime:
        return

    record_bin = str(args.record_bin)
    resolved_record = shutil.which(record_bin)
    if resolved_record is None:
        raise CollectError(f"LeRobot recorder is not executable or was not found: {record_bin}")
    record_path = Path(resolved_record)
    if not record_path.is_file() or not os.access(record_path, os.X_OK):
        raise CollectError(f"LeRobot recorder is not an executable file: {record_path}")

    try:
        importlib.import_module("lerobot")
    except Exception as exc:
        raise CollectError(
            f"LeRobot {SUPPORTED_LEROBOT_VERSION} cannot be imported in the active Python environment"
        ) from exc

    try:
        lerobot_version = metadata.version("lerobot")
    except metadata.PackageNotFoundError as exc:
        raise CollectError(
            f"LeRobot {SUPPORTED_LEROBOT_VERSION} is required in the active Python environment"
        ) from exc
    if lerobot_version != SUPPORTED_LEROBOT_VERSION:
        raise CollectError(
            f"LeRobot {SUPPORTED_LEROBOT_VERSION} is required, found {lerobot_version}"
        )
    if args.live and sys.version_info[:2] != (3, 12):
        raise CollectError("Live Panthera collection requires Python 3.12 for the hardware SDK wheel")


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: str


@dataclass(frozen=True)
class CameraDevice:
    label: str
    source: str
    resolved: str


def _validate_camera_specs(camera_specs: Sequence[CameraSpec]) -> tuple[CameraSpec, ...]:
    validated = tuple(camera_specs)
    seen_names = set()
    seen_sources = {}
    for spec in validated:
        if not isinstance(spec, CameraSpec):
            raise CollectError("Camera entries must be CameraSpec values")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", spec.name):
            raise CollectError(
                f"Camera name {spec.name!r} must start with a letter and contain only letters, numbers or '_'"
            )
        if not str(spec.source).strip():
            raise CollectError(f"Camera source for {spec.name!r} must not be empty")
        if spec.name in seen_names:
            raise CollectError(f"Duplicate camera name: {spec.name}")
        seen_names.add(spec.name)
        source_key = _canonical_camera_source(spec.source)
        previous_name = seen_sources.get(source_key)
        if previous_name is not None:
            raise CollectError(
                f"Camera roles {previous_name!r} and {spec.name!r} resolve to the same source: {source_key}"
            )
        seen_sources[source_key] = spec.name
    return validated


def _canonical_camera_source(source: str) -> str:
    value = str(source).strip()
    parsed_source = _camera_source(value)
    if isinstance(parsed_source, int):
        value = f"/dev/video{parsed_source}"
    return str(Path(value).expanduser().resolve(strict=False))


@dataclass(frozen=True)
class _RecordStatusUpdate:
    phase: str
    detail: str
    beep_count: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panthera-lr-collect",
        description="Start Panthera control daemon and run LeRobot-native teleoperation collection.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="connect to Panthera hardware")
    mode.add_argument("--mock", action="store_false", dest="live", help="run without Panthera hardware")
    parser.set_defaults(live=False)

    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id")
    parser.add_argument("--task", default=DEFAULT_TASK, help="single task text stored in the dataset")
    parser.add_argument("--episodes", type=_positive_int, default=50, help="number of episodes")
    parser.add_argument("--episode-seconds", type=_positive_float, default=60.0, help="seconds per episode")
    parser.add_argument("--reset-seconds", type=_non_negative_float, default=60.0, help="seconds between episodes")
    parser.add_argument("--fps", type=_positive_int, default=30, help="LeRobot sampling and camera FPS")
    parser.add_argument("--hz", type=_positive_float, default=1000.0, help="Panthera daemon control loop rate")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path shared by daemon and adapters")
    parser.add_argument("--robot-id", default="panthera_follower", help="LeRobot robot id")
    parser.add_argument("--teleop-id", default="panthera_leader", help="LeRobot teleoperator id")
    parser.add_argument("--record-bin", default="lerobot-record", help="LeRobot record executable")
    parser.add_argument("--image-width", type=_positive_int, default=640, help="camera width passed to LeRobot")
    parser.add_argument("--image-height", type=_positive_int, default=480, help="camera height passed to LeRobot")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=SOURCE", help="camera mapping")
    parser.add_argument("--camera-fourcc", default=None, help="V4L2 pixel format fourcc for LeRobot OpenCV cameras (e.g. MJPG); needed when multiple uncompressed streams exceed USB bandwidth")
    parser.add_argument("--camera-warmup-s", type=_positive_int, default=5, help="seconds LeRobot waits for the first camera frame (first frame can take >1s when several cameras initialize concurrently on one USB controller)")
    parser.add_argument("--select-cameras", action="store_true", help="choose cameras in a preflight GUI")
    parser.add_argument("--camera-test-timeout-s", type=_positive_float, default=DEFAULT_CAMERA_TEST_TIMEOUT_S, help="seconds to wait for camera preview frame reads")
    parser.add_argument("--no-cameras", action="store_true", help="run state/action collection without cameras")
    parser.add_argument("--auto-start", action="store_true", help="start lerobot-record without Preview")
    parser.add_argument("--no-gui", action="store_true", help="disable optional Tk preview/camera GUIs")
    parser.add_argument("--no-status-ui", action="store_true", help="disable the persistent Tk recording status indicator")
    parser.add_argument("--voice-prompts", action="store_true", help="keep LeRobot text-to-speech prompts instead of Panthera bell prompts only")
    parser.add_argument("--print-command", action="store_true", help="print lerobot-record command and exit")
    parser.add_argument("--no-release-prompt", action="store_true", help="do not wait for Enter before stopping own daemon")
    parser.add_argument("--stop-external-daemon", action="store_true", help="also stop a daemon that was already running")
    parser.add_argument("--push-to-hub", action="store_true", help="set --dataset.push_to_hub=True")
    parser.add_argument("--dataset-root", default=None, help="local LeRobot dataset root; required with --resume")
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument("--resume", action="store_true", default=False, help="resume an existing local dataset; requires --dataset-root")
    resume_mode.add_argument("--no-resume", action="store_false", dest="resume", help="start a fresh recording/evaluation dataset")
    parser.add_argument(
        "--no-streaming-encoding",
        action="store_true",
        help="set --dataset.streaming_encoding=false",
    )
    parser.add_argument("--encoder-threads", type=_positive_int, default=2, help="LeRobot video encoder threads")
    parser.add_argument("--timeout-s", type=_positive_float, default=1.0, help="IPC request timeout")
    parser.add_argument("--stale-timeout-s", type=_positive_float, default=0.2, help="fresh snapshot threshold")
    parser.add_argument("--record-graceful-timeout-s", type=_positive_float, default=30.0, help="seconds for recorder SIGINT shutdown")
    parser.add_argument("--record-terminate-timeout-s", type=_positive_float, default=5.0, help="seconds for recorder SIGTERM shutdown")
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=5.0, help="daemon startup timeout")
    parser.add_argument("--policy-mirror-leader", action="store_true", help="mirror follower pose to leader during policy control")
    parser.add_argument("--policy-mirror-max-step-rad", type=_positive_float, default=0.02, help="max leader mirror joint target step per daemon control tick")
    parser.add_argument("--policy-mirror-max-gripper-step", type=_positive_float, default=0.04, help="max leader mirror gripper target step per daemon control tick")
    parser.add_argument("--sdk-root", default=None, help="Panthera SDK panthera_python path")
    parser.add_argument("--leader-config", default=None, help="leader YAML path")
    parser.add_argument("--follower-config", default=None, help="follower YAML path")
    parser.add_argument(
        "--teleop-gripper-feedback-scale",
        type=_non_negative_float,
        default=0.5,
        help="teleop: follower gripper torque fed back to the leader gripper (smaller = easier to squeeze)",
    )
    parser.add_argument(
        "--teleop-gripper-squeeze-boost",
        type=_non_negative_float,
        default=0.0,
        help="teleop: extra closing force when the follower gripper is blocked by an object (0 disables; try 0.5-1.0)",
    )
    parser.add_argument(
        "--teleop-gripper-squeeze-max",
        type=_positive_float,
        default=0.4,
        help="teleop: cap on the squeeze-boost virtual target overshoot",
    )
    parser.add_argument(
        "--lerobot-arg",
        action="append",
        default=[],
        help="extra raw argument appended to lerobot-record; repeat as needed",
    )
    return parser


def collect(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any | None = None,
    stderr: Any | None = None,
    stdin: Any | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    record_runner: Callable[..., int] | None = None,
    client_factory: Callable[..., ControlClient] = ControlClient,
    status_ui_factory: Callable[[Any], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    stdin = sys.stdin if stdin is None else stdin
    args = build_parser().parse_args(argv)
    using_default_record_runner = record_runner is None
    record_runner = _default_record_runner if using_default_record_runner else record_runner
    daemon_process = None
    daemon_started = False
    daemon_log = None
    daemon_log_path = None
    client = None
    status_ui = None
    result_code = None
    should_cleanup = False
    daemon_stop_requested = False

    try:
        expected_mode = "live" if args.live else "mock"
        run_collection_preflight(args, require_runtime=using_default_record_runner)
        camera_specs = resolve_camera_specs(args)
        record_command = build_lerobot_record_command(args, camera_specs)
        if args.print_command:
            print(shlex.join(record_command), file=stdout)
            return 0
        _preflight_camera_reads(camera_specs, args=args)

        external_status = _probe_existing_daemon(args, expected_mode, client_factory)
        if external_status is None:
            ensure_live_serial_permissions(args, stderr=stderr)

        child_env = build_child_env()
        if external_status is None:
            daemon_command = build_daemon_command(args)
            print(f"Starting Panthera daemon: {shlex.join(daemon_command)}", file=stderr)
            daemon_log_path = _daemon_log_path()
            daemon_log = open(daemon_log_path, "w", encoding="utf-8", buffering=1)
            print(f"Daemon log: {daemon_log_path}", file=stderr)
            try:
                daemon_process = popen_factory(
                    daemon_command,
                    env=child_env,
                    stdout=daemon_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                raise CollectError(f"Failed to start Panthera daemon: {exc}") from exc
            daemon_started = True
        else:
            print(f"Using existing Panthera daemon on {args.socket}", file=stderr)

        client = wait_for_daemon_ready(
            args,
            expected_mode,
            client_factory=client_factory,
            sleep=sleep,
            stderr=stderr,
            daemon_process=daemon_process,
            daemon_log_path=daemon_log_path,
        )
        should_cleanup = True

        if not args.auto_start:
            wait_for_preview_start(args, stdin=stdin, stdout=stdout, stderr=stderr, sleep=sleep)

        status_ui = _start_record_status_ui(args, stderr=stderr, status_ui_factory=status_ui_factory)
        if status_ui is not None:
            status_ui.set_status("STARTING", "Launching LeRobot record")

        print(f"Starting LeRobot record: {shlex.join(record_command)}", file=stderr)
        if using_default_record_runner:
            return_code = int(
                record_runner(
                    record_command,
                    env=child_env,
                    graceful_timeout_s=args.record_graceful_timeout_s,
                    terminate_timeout_s=args.record_terminate_timeout_s,
                    status_ui=status_ui,
                    stdout=stdout,
                    stderr=stderr,
                    sleep=sleep,
                )
            )
        else:
            beep(2, stdout=stdout)
            return_code = int(record_runner(record_command, env=child_env))
            beep(1, stdout=stdout)
        if status_ui is not None:
            phase = "DONE" if return_code == 0 else "ERROR"
            status_ui.set_status(phase, f"LeRobot record exited with code {return_code}")
            status_ui.pump()
            sleep(0.5)
        result_code = return_code
    except KeyboardInterrupt:
        if status_ui is not None:
            status_ui.set_status("STOPPING", "Interrupted; holding Panthera")
        print("\nCollection interrupted.", file=stderr)
        result_code = 130
    except CollectError as exc:
        print(f"error: {exc}", file=stderr)
        result_code = 2
    finally:
        if status_ui is not None:
            status_ui.shutdown()
        if should_cleanup and client is not None:
            daemon_stop_requested = _hold_and_release(
                client,
                own_daemon=daemon_started,
                args=args,
                stdin=stdin,
                stderr=stderr,
            )
        if daemon_process is not None and (daemon_stop_requested or not should_cleanup):
            _wait_or_terminate_daemon(daemon_process, stderr=stderr)
        if result_code == 0 and daemon_started and should_cleanup and not daemon_stop_requested:
            dataset_root = getattr(args, "_resolved_dataset_root", None) or args.dataset_root or "<unknown>"
            daemon_pid = getattr(daemon_process, "pid", "unknown")
            print(
                "error: safe hold was not confirmed; owned daemon left running "
                f"(pid={daemon_pid}). Dataset root: {dataset_root}",
                file=stderr,
            )
            result_code = 2


        if daemon_log is not None:
            try:
                daemon_log.close()
            except Exception:
                pass

    if result_code is None:
        raise RuntimeError("collection exited without a result code")
    return result_code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        with _translated_termination_signals():
            return collect(argv)
    except KeyboardInterrupt:
        return 130


@contextmanager
def _translated_termination_signals():
    installed = []
    interruption_requested = False

    def interrupt_collection(_signum, _frame):
        nonlocal interruption_requested
        if interruption_requested:
            return
        interruption_requested = True
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous = signal.getsignal(signum)
            signal.signal(signum, interrupt_collection)
            installed.append((signum, previous))
        yield
    finally:
        for signum, previous in reversed(installed):
            signal.signal(signum, previous)


def parse_camera_specs(values: Sequence[str]) -> tuple[CameraSpec, ...]:
    specs = []
    seen_names = set()
    for value in values:
        if "=" not in value:
            raise CollectError(f"Camera spec must be NAME=SOURCE, got {value!r}.")
        name, source = value.split("=", 1)
        name = name.strip()
        source = source.strip()
        if not name or not source:
            raise CollectError(f"Camera spec must be NAME=SOURCE, got {value!r}.")
        if "/" in name or "." in name:
            raise CollectError("Camera name must not contain '/' or '.'.")
        if name in seen_names:
            raise CollectError(f"Duplicate camera name: {name}")
        seen_names.add(name)
        specs.append(CameraSpec(name=name, source=source))
    return _validate_camera_specs(specs)


def load_camera_selection(path: Path | None = None) -> tuple[CameraSpec, ...]:
    selection_path = CAMERA_SELECTION_PATH if path is None else path
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectError(f"Invalid camera selection cache {selection_path}: {exc}") from exc
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
        raise CollectError(
            f"Invalid camera selection cache {selection_path}: expected an object with version 1"
        )
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        raise CollectError(f"Invalid camera selection cache {selection_path}: 'cameras' must be a list")
    specs = []
    for index, item in enumerate(cameras):
        if not isinstance(item, dict):
            raise CollectError(
                f"Invalid camera selection cache {selection_path}: cameras[{index}] must be an object"
            )
        name = item.get("name")
        source = item.get("source")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(source, str)
            or not source.strip()
        ):
            raise CollectError(
                f"Invalid camera selection cache {selection_path}: cameras[{index}] requires non-empty string name and source"
            )
        specs.append(CameraSpec(name=name.strip(), source=source.strip()))
    try:
        return _validate_camera_specs(specs)
    except CollectError as exc:
        raise CollectError(f"Invalid camera selection cache {selection_path}: {exc}") from exc


def save_camera_selection(camera_specs: Sequence[CameraSpec], path: Path | None = None) -> None:
    camera_specs = _validate_camera_specs(camera_specs)
    selection_path = CAMERA_SELECTION_PATH if path is None else path
    payload = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cameras": [{"name": spec.name, "source": spec.source} for spec in camera_specs],
    }
    selection_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def resolve_camera_specs(args: argparse.Namespace) -> tuple[CameraSpec, ...]:
    if args.no_cameras and (args.camera or args.select_cameras):
        raise CollectError("--no-cameras cannot be combined with --camera or --select-cameras")
    explicit_specs = parse_camera_specs(args.camera)
    if args.no_cameras:
        return ()
    if args.select_cameras:
        if args.no_gui:
            raise CollectError("--select-cameras requires the GUI; remove --no-gui or pass explicit --camera values.")
        role_names = tuple(spec.name for spec in explicit_specs) or DEFAULT_CAMERA_ROLE_NAMES
        return _validate_camera_specs(select_camera_specs_interactively(explicit_specs, role_names=role_names, args=args))
    if explicit_specs:
        return explicit_specs
    cached = load_camera_selection()
    if cached:
        return cached
    raise CollectError("No cameras configured. Pass --select-cameras, --camera NAME=SOURCE, or --no-cameras.")


def build_lerobot_camera_config(
    camera_specs: Sequence[CameraSpec],
    *,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None = None,
    warmup_s: int | None = None,
) -> str:
    payload = {}
    for spec in _validate_camera_specs(camera_specs):
        payload[spec.name] = {
            "type": "opencv",
            "index_or_path": _camera_source(spec.source),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
        }
        if fourcc:
            payload[spec.name]["fourcc"] = fourcc
        if warmup_s is not None:
            payload[spec.name]["warmup_s"] = int(warmup_s)
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _validate_lerobot_passthrough_args(values: Sequence[str]) -> None:
    for value in values:
        option = str(value).strip()
        managed_namespace = any(
            option == f"--{name}"
            or option.startswith(f"--{name}.")
            or option.startswith(f"--{name}=")
            for name in ("robot", "teleop", "dataset")
        )
        managed_resume = option == "--resume" or option.startswith("--resume=")
        if managed_namespace or managed_resume:
            raise CollectError(
                f"--lerobot-arg cannot override Panthera-managed option {option!r}; use the wrapper option instead"
            )


def build_lerobot_record_command(args: argparse.Namespace, camera_specs: Sequence[CameraSpec]) -> list[str]:
    _validate_lerobot_passthrough_args(args.lerobot_arg)
    dataset_root = _prepare_collection_dataset(args)
    command = [
        args.record_bin,
        "--robot.type=panthera_ht_follower",
        f"--robot.id={args.robot_id}",
        f"--robot.socket_path={args.socket}",
        f"--robot.timeout_s={_format_number(args.timeout_s)}",
        f"--robot.stale_timeout_s={_format_number(args.stale_timeout_s)}",
    ]
    if camera_specs:
        command.append(
            "--robot.cameras="
            + build_lerobot_camera_config(
                camera_specs,
                width=args.image_width,
                height=args.image_height,
                fps=args.fps,
                fourcc=args.camera_fourcc,
                warmup_s=args.camera_warmup_s,
            )
        )
    command.extend(
        [
            "--teleop.type=panthera_ht_leader",
            f"--teleop.id={args.teleop_id}",
            f"--teleop.socket_path={args.socket}",
            f"--teleop.timeout_s={_format_number(args.timeout_s)}",
            f"--teleop.stale_timeout_s={_format_number(args.stale_timeout_s)}",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.push_to_hub={_bool_title(args.push_to_hub)}",
            f"--play_sounds={_bool_lower(args.voice_prompts)}",
            f"--dataset.num_episodes={args.episodes}",
            f"--dataset.episode_time_s={float(args.episode_seconds)}",
            f"--dataset.reset_time_s={float(args.reset_seconds)}",
            f"--dataset.streaming_encoding={_bool_lower(not args.no_streaming_encoding)}",
            f"--dataset.encoder_threads={args.encoder_threads}",
            f"--dataset.fps={args.fps}",
        ]
    )
    _append_dataset_root_and_resume_args(
        command,
        args,
        default_root=dataset_root,
    )
    if args.task:
        command.append(f"--dataset.single_task={args.task}")
    command.extend(args.lerobot_arg)
    return command


def build_daemon_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "panthera_lerobot_control.cli",
        "--socket",
        args.socket,
        "--loop-hz",
        _format_number(args.hz),
        "--initial-control-mode",
        "teleop",
    ]
    if not args.live:
        command.append("--mock")
    if getattr(args, "policy_mirror_leader", False):
        command.append("--policy-mirror-leader")
        for flag, value in (
            ("--policy-mirror-max-step-rad", getattr(args, "policy_mirror_max_step_rad", None)),
            ("--policy-mirror-max-gripper-step", getattr(args, "policy_mirror_max_gripper_step", None)),
        ):
            if value is not None:
                command.extend([flag, _format_number(value)])
    for flag, value in (
        ("--sdk-root", args.sdk_root),
        ("--leader-config", args.leader_config),
        ("--follower-config", args.follower_config),
    ):
        if value:
            command.extend([flag, str(value)])
    for flag, attr, default in (
        ("--teleop-gripper-feedback-scale", "teleop_gripper_feedback_scale", 0.5),
        ("--teleop-gripper-squeeze-boost", "teleop_gripper_squeeze_boost", 0.0),
        ("--teleop-gripper-squeeze-max", "teleop_gripper_squeeze_max", 0.4),
    ):
        value = getattr(args, attr, None)
        if value is not None and float(value) != default:
            command.extend([flag, _format_number(float(value))])
    return command


def wait_for_daemon_ready(
    args: argparse.Namespace,
    expected_mode: str,
    *,
    client_factory: Callable[..., ControlClient] = ControlClient,
    sleep: Callable[[float], None] = time.sleep,
    stderr: Any | None = None,
    daemon_process: Any | None = None,
    daemon_log_path: str | None = None,
) -> ControlClient:
    stderr = sys.stderr if stderr is None else stderr
    client = client_factory(args.socket, timeout_s=args.timeout_s)
    deadline = time.monotonic() + float(args.startup_timeout_s)
    last_sequence = None
    last_error = None
    while time.monotonic() < deadline:
        if daemon_process is not None:
            return_code = daemon_process.poll()
            if return_code is not None:
                log_tail = _read_log_tail(daemon_log_path)
                detail = f"\nDaemon log tail:\n{log_tail}" if log_tail else ""
                raise CollectError(f"Panthera daemon exited before it became ready (code {return_code}).{detail}")
        try:
            status = client.status()
            _validate_daemon_status(status, expected_mode)
            snapshot = client.snapshot(max_age_s=args.stale_timeout_s)
            sequence = int(snapshot["sequence"])
            if last_sequence is not None and sequence > last_sequence:
                print(f"Daemon ready: mode={status.get('mode')} sequence={sequence}", file=stderr)
                return client
            last_sequence = sequence
        except Exception as exc:
            last_error = exc
        sleep(0.05)
    reason = f": {last_error}" if last_error else ""
    log_tail = _read_log_tail(daemon_log_path)
    detail = f"\nDaemon log tail:\n{log_tail}" if log_tail else ""
    raise CollectError(f"Timed out waiting for daemon on {args.socket}{reason}{detail}")


def wait_for_preview_start(
    args: argparse.Namespace,
    *,
    stdin: Any,
    stdout: Any,
    stderr: Any,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    ui = None
    if not args.no_gui:
        ui = _PreviewStartUI(stderr=stderr).start()
    if ui is None and not _is_tty(stdin):
        raise CollectError("Preview manual start requires a TTY or GUI; pass --auto-start for unattended use.")
    print("Preview is active. Move the robot and check cameras, then press Enter to start recording.", file=stdout)
    try:
        while True:
            if ui is not None:
                ui.pump()
                if ui.start_requested:
                    return
            if _stdin_requested(stdin):
                return
            sleep(0.05)
    finally:
        if ui is not None:
            ui.shutdown()


def select_camera_specs_interactively(
    current_specs: Sequence[CameraSpec],
    *,
    role_names: Sequence[str],
    args: argparse.Namespace,
) -> tuple[CameraSpec, ...]:
    devices = list_camera_devices()
    if not devices:
        raise CollectError("No camera devices found under /dev/v4l/by-id or /dev/video*.")
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        raise CollectError("Camera selection requires Tkinter. Pass explicit --camera NAME=SOURCE values instead.") from exc

    label_to_device = {device.label: device for device in devices}
    labels = list(label_to_device)
    root = tk.Tk()
    root.title("Select Panthera LeRobot Cameras")
    root.resizable(False, False)
    result: dict[str, tuple[CameraSpec, ...] | None] = {"specs": None}

    tk.Label(root, text="Select LeRobot cameras before starting the daemon", font=("TkDefaultFont", 13, "bold")).grid(
        row=0,
        column=0,
        columnspan=2,
        padx=12,
        pady=(12, 8),
        sticky="w",
    )
    selected_vars = {}
    assigned = set()
    for offset, role_name in enumerate(role_names):
        row = offset + 1
        tk.Label(root, text=f"{role_name}:").grid(row=row, column=0, padx=(12, 8), pady=4, sticky="e")
        default_label = _default_camera_label(role_name, devices, current_specs, assigned)
        selected_vars[role_name] = tk.StringVar(value=default_label)
        combo = ttk.Combobox(root, textvariable=selected_vars[role_name], values=labels, state="readonly", width=88)
        combo.grid(row=row, column=1, padx=(0, 12), pady=4, sticky="w")

    status_var = tk.StringVar(value="")
    status_row = len(role_names) + 1
    tk.Label(root, textvariable=status_var, wraplength=720, justify="left").grid(
        row=status_row,
        column=0,
        columnspan=2,
        padx=12,
        pady=(6, 4),
        sticky="w",
    )

    def selected_specs() -> tuple[CameraSpec, ...]:
        specs = []
        for role_name in role_names:
            device = label_to_device[selected_vars[role_name].get()]
            specs.append(CameraSpec(name=role_name, source=device.source))
        return tuple(specs)

    def test_selected() -> None:
        try:
            preview_camera_specs(selected_specs(), args=args, parent=root)
        except Exception as exc:
            status_var.set(f"Test failed: {exc}")
            messagebox.showerror("Camera test failed", str(exc), parent=root)
            return
        status_var.set("Camera preview OK")

    def use_selected() -> None:
        result["specs"] = selected_specs()
        save_camera_selection(result["specs"])
        root.destroy()

    def cancel() -> None:
        root.destroy()

    buttons = tk.Frame(root)
    buttons.grid(row=status_row + 1, column=0, columnspan=2, padx=12, pady=(4, 12), sticky="e")
    tk.Button(buttons, text="Test cameras", command=test_selected).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Use selected", command=use_selected).pack(side="left", padx=(0, 8))
    tk.Button(buttons, text="Cancel", command=cancel).pack(side="left")
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    if result["specs"] is None:
        raise CollectError("Camera selection cancelled.")
    return result["specs"]


def list_camera_devices(by_id_dir: Path = Path("/dev/v4l/by-id"), dev_dir: Path = Path("/dev")) -> tuple[CameraDevice, ...]:
    devices = []
    if by_id_dir.exists():
        for path in sorted(by_id_dir.glob("*video-index0")):
            try:
                resolved = str(path.resolve(strict=False))
            except OSError:
                resolved = str(path)
            devices.append(CameraDevice(label=f"{path.name} ({resolved})", source=str(path), resolved=resolved))
    if devices:
        return tuple(devices)
    for path in sorted(dev_dir.glob("video*")):
        devices.append(CameraDevice(label=f"{path.name} ({path})", source=str(path), resolved=str(path)))
    return tuple(devices)


def preview_camera_specs(camera_specs: Sequence[CameraSpec], *, args: argparse.Namespace, parent: Any | None = None) -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise CollectError("Camera preview requires Tkinter.") from exc
    images = _read_camera_preview_images_with_timeout(
        camera_specs,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fps=int(args.fps),
        timeout_s=float(getattr(args, "camera_test_timeout_s", DEFAULT_CAMERA_TEST_TIMEOUT_S)),
    )
    if not images:
        raise CollectError("No camera frames available for preview.")
    _show_preview_images(images, parent=parent, tk=tk)


def _preflight_camera_reads(camera_specs: Sequence[CameraSpec], *, args: argparse.Namespace) -> None:
    if not camera_specs:
        return
    images = _read_camera_preview_images_with_timeout(
        camera_specs,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fps=int(args.fps),
        timeout_s=float(getattr(args, "camera_test_timeout_s", DEFAULT_CAMERA_TEST_TIMEOUT_S)),
    )
    expected = {spec.name for spec in camera_specs}
    missing = expected - set(images)
    if missing:
        raise CollectError(
            f"Camera preflight did not return frames for: {', '.join(sorted(missing))}"
        )

def _read_camera_preview_images_with_timeout(
    camera_specs: Sequence[CameraSpec],
    *,
    image_width: int,
    image_height: int,
    fps: int,
    timeout_s: float,
    process_factory: Callable[..., Any] | None = None,
    pipe_factory: Callable[[], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0:
        return _read_camera_preview_images(camera_specs, image_width=image_width, image_height=image_height, fps=fps)

    ctx = mp.get_context("fork")
    if process_factory is None:
        process_factory = ctx.Process
    if pipe_factory is None:
        pipe_factory = ctx.Pipe

    parent_conn, child_conn = pipe_factory()
    payload = [(spec.name, spec.source) for spec in camera_specs]
    process = process_factory(
        target=_camera_preview_worker,
        args=(child_conn, payload, image_width, image_height, fps),
        daemon=True,
    )
    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if parent_conn.poll(min(remaining, 0.05)):
                status, data = parent_conn.recv()
                process.join(timeout=0.5)
                if status == "ok":
                    return data
                raise CollectError(str(data))
            if getattr(process, "exitcode", None) is not None:
                if parent_conn.poll():
                    status, data = parent_conn.recv()
                    if status == "ok":
                        return data
                    raise CollectError(str(data))
                raise CollectError(f"Camera preview worker exited with status {process.exitcode}")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            child_conn.close()
        except Exception:
            pass
        _terminate_preview_process(process)

    raise CollectError(
        f"Timed out reading camera preview after {_format_number(timeout_s)}s. "
        "This usually means the selected USB camera node is stuck; try another /dev/video device or unplug/replug it."
    )


def _terminate_preview_process(process: Any) -> None:
    try:
        process.join(timeout=0.5)
    except Exception:
        pass
    try:
        alive = process.is_alive()
    except Exception:
        alive = True
    if alive:
        try:
            process.terminate()
            process.join(timeout=0.5)
        except Exception:
            pass
    try:
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
    except Exception:
        pass


def _camera_preview_worker(conn: Any, specs_payload: Sequence[tuple[str, str]], image_width: int, image_height: int, fps: int) -> None:
    try:
        specs = tuple(CameraSpec(name=name, source=source) for name, source in specs_payload)
        images = _read_camera_preview_images(specs, image_width=image_width, image_height=image_height, fps=fps)
        conn.send(("ok", images))
    except BaseException as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _read_camera_preview_images(
    camera_specs: Sequence[CameraSpec],
    *,
    image_width: int,
    image_height: int,
    fps: int,
) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:
        raise CollectError("Camera preview requires OpenCV.") from exc
    images = {}
    for spec in camera_specs:
        capture = cv2.VideoCapture(_camera_source(spec.source))
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            ok, image = capture.read()
            if not ok or image is None:
                raise CollectError(f"Failed to read camera {spec.name} from {spec.source}")
            images[spec.name] = image
        finally:
            capture.release()
    return images


def ensure_live_serial_permissions(args: argparse.Namespace, *, stderr: Any) -> None:
    if not getattr(args, "live", False):
        return
    try:
        ensure_panthera_serial_access(stderr=stderr)
    except AdapterError as exc:
        raise CollectError(str(exc)) from exc


def _current_local_datetime() -> datetime:
    return datetime.now().astimezone()

def _next_dataset_root_counter() -> int:
    global _DATASET_ROOT_COUNTER
    with _DATASET_ROOT_COUNTER_LOCK:
        value = _DATASET_ROOT_COUNTER
        _DATASET_ROOT_COUNTER += 1
    return value



def _validated_repo_segments(repo_id: str) -> tuple[str, str]:
    parts = str(repo_id).split("/")
    if len(parts) != 2 or any(not part or not _REPO_SEGMENT_RE.fullmatch(part) for part in parts):
        raise CollectError(
            "--repo-id must be exactly namespace/name using letters, numbers, '.', '_' or '-'"
        )
    if any(part in {".", ".."} for part in parts):
        raise CollectError("--repo-id must not contain '.' or '..' path segments")
    return parts[0], parts[1]


def resolve_collection_dataset_root(
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
    path_exists: Callable[[Any], bool] = os.path.lexists,
) -> Path:
    namespace, name = _validated_repo_segments(args.repo_id)
    explicit_root = getattr(args, "dataset_root", None)
    resume = bool(getattr(args, "resume", False))

    if resume and not explicit_root:
        raise CollectError("--resume requires --dataset-root because LeRobot cannot resume with root=None")
    if explicit_root:
        root = Path(explicit_root).expanduser()
        exists = path_exists(root)
        if resume:
            if not exists or not root.is_dir():
                raise CollectError(f"--resume requires an existing dataset directory: {root}")
            _validate_resume_dataset_writable(root)
            return root
        if exists:
            raise CollectError(
                f"Fresh dataset root already exists: {root}. Use --resume --dataset-root only after validating it."
            )
        _validate_dataset_parent(root)
        return root

    data_root = Path(os.environ.get("PANTHERA_DATA_ROOT", _default_data_root())).expanduser()
    timestamp = (now or _current_local_datetime()).strftime("%Y%m%d_%H%M%S_%f")
    process_id = os.getpid()
    counter = _next_dataset_root_counter()
    base = data_root / "lerobot" / namespace / name / f"{timestamp}_p{process_id}_c{counter:06d}"
    candidate = base
    suffix = 1
    while path_exists(candidate):
        candidate = Path(f"{base}_{suffix:03d}")
        suffix += 1
    _validate_dataset_parent(candidate)
    return candidate
def _validate_resume_dataset_writable(root: Path) -> None:
    meta_dir = root / "meta"
    checks = (
        (root, os.W_OK | os.X_OK),
        (meta_dir, os.W_OK | os.X_OK),
        (meta_dir / "info.json", os.W_OK),
        (meta_dir / "tasks.parquet", os.W_OK),
    )
    for path, mode in checks:
        if not os.path.lexists(path):
            continue
        if not os.access(path, mode):
            raise CollectError(f"Resume dataset path is not writable: {path}")




def _validate_dataset_parent(root: Path) -> None:
    parent = root.parent
    while not os.path.lexists(parent):
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent
    if not parent.is_dir():
        raise CollectError(f"Dataset parent is not a directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise CollectError(f"Dataset parent is not writable: {parent}")


def _prepare_collection_dataset(args: argparse.Namespace) -> Path:
    cached = getattr(args, "_resolved_dataset_root", None)
    if cached is not None:
        return Path(cached)
    root = resolve_collection_dataset_root(args)
    if bool(getattr(args, "resume", False)):
        validate_resume_dataset(root, args)
    args._resolved_dataset_root = str(root)
    return root


def validate_resume_dataset(root: Path, args: argparse.Namespace) -> None:
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectError(f"Cannot validate resume metadata at {info_path}: {exc}") from exc
    if not isinstance(info, dict):
        raise CollectError(f"Resume metadata must be a JSON object: {info_path}")
    if info.get("robot_type") != "panthera_ht_follower":
        raise CollectError("Resume dataset robot_type must be 'panthera_ht_follower'")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps != int(args.fps):
        raise CollectError(f"Resume dataset fps must equal requested fps={args.fps}")
    total_tasks = info.get("total_tasks")
    if type(total_tasks) is not int or total_tasks != 1:
        raise CollectError("Resume dataset must contain exactly one task")

    features = info.get("features")
    if not isinstance(features, dict):
        raise CollectError("Resume dataset info.json is missing features")
    _validate_resume_feature(
        features,
        "observation.state",
        shape=[14],
        names=list(PANTHERA_OBSERVATION_NAMES),
    )
    _validate_resume_feature(
        features,
        "action",
        shape=[7],
        names=list(PANTHERA_ACTION_NAMES),
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        task_table = pq.read_table(tasks_path, columns=["task_index", "task"])
    except Exception as exc:
        raise CollectError(f"Cannot validate resume task metadata at {tasks_path}: {exc}") from exc

    task_index_type = task_table.schema.field("task_index").type
    if not pa.types.is_integer(task_index_type):
        raise CollectError("Resume dataset task_index must use an Arrow integer type")
    task_indices = task_table.column("task_index").to_pylist()
    if (
        len(task_indices) != 1
        or type(task_indices[0]) is not int
        or task_indices[0] != 0
    ):
        raise CollectError("Resume dataset task_index must contain exactly integer 0")
    tasks = task_table.column("task").to_pylist()
    if tasks != [str(args.task)]:
        raise CollectError(
            f"Resume dataset task must exactly match {args.task!r} with task_index 0"
        )


def _validate_resume_feature(
    features: dict[str, Any],
    key: str,
    *,
    shape: list[int],
    names: list[str],
) -> None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise CollectError(f"Resume dataset is missing feature {key!r}")
    actual_shape = feature.get("shape")
    if (
        feature.get("dtype") != "float32"
        or not isinstance(actual_shape, list)
        or any(type(dimension) is not int for dimension in actual_shape)
        or actual_shape != shape
        or feature.get("names") != names
    ):
        raise CollectError(
            f"Resume dataset feature {key!r} does not match the Panthera dtype/shape/name contract"
        )


def _append_dataset_root_and_resume_args(
    command: list[str],
    args: argparse.Namespace,
    *,
    default_root: Path | None = None,
) -> None:
    dataset_root = getattr(args, "_resolved_dataset_root", None)
    if dataset_root is None:
        dataset_root = getattr(args, "dataset_root", None)
    resume = bool(getattr(args, "resume", False))
    if resume and not dataset_root:
        raise CollectError("--resume requires --dataset-root because LeRobot cannot resume with root=None")
    if dataset_root:
        command.append(f"--dataset.root={Path(dataset_root)}")
    elif default_root is not None:
        command.append(f"--dataset.root={default_root}")
    command.append(f"--resume={_bool_lower(resume)}")


def _default_collection_dataset_root(repo_id: str | None) -> Path:
    namespace, name = _validated_repo_segments(repo_id or DEFAULT_REPO_ID)
    data_root = Path(os.environ.get("PANTHERA_DATA_ROOT", _default_data_root()))
    return data_root / "lerobot" / namespace / name


def build_child_env(project_root: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    root = str(project_root or Path.cwd())
    env["PYTHONPATH"] = root
    return env


def beep(count: int = 1, *, stdout: Any | None = None) -> None:
    stream = sys.stdout if stdout is None else stdout
    try:
        stream.write("\a" * max(0, int(count)))
        stream.flush()
    except Exception:
        pass


def _probe_existing_daemon(
    args: argparse.Namespace,
    expected_mode: str,
    client_factory: Callable[..., ControlClient],
) -> dict[str, Any] | None:
    try:
        status = client_factory(args.socket, timeout_s=args.timeout_s).status()
    except ControlClientError:
        return None
    _validate_daemon_status(status, expected_mode)
    return status


def _validate_daemon_status(status: dict[str, Any], expected_mode: str) -> None:
    if not status.get("running"):
        raise CollectError("Daemon status reports running=False")
    actual_mode = status.get("mode")
    if actual_mode != expected_mode:
        raise CollectError(f"Daemon mode is {actual_mode!r}, expected {expected_mode!r}")
    control_mode = status.get("control_mode")
    if control_mode not in ("teleop", "policy", "freedrive"):
        raise CollectError(
            f"Daemon control_mode is {control_mode!r}; LeRobot collection requires 'teleop', 'policy' or 'freedrive'"
        )
    if status.get("last_error"):
        raise CollectError(f"Daemon reports last_error: {status['last_error']}")


def _hold_and_release(
    client: ControlClient,
    *,
    own_daemon: bool,
    args: argparse.Namespace,
    stdin: Any,
    stderr: Any,
) -> bool:
    hold_confirmed = False
    try:
        hold_status = client.hold()
        hold_confirmed = _wait_for_confirmed_hold(
            client,
            hold_status=hold_status,
            timeout_s=getattr(args, "stale_timeout_s", 0.25),
        )
        if hold_confirmed:
            print("Daemon confirmed holding follower position. Support the arm before stopping the daemon into brake mode.", file=stderr)
        else:
            print("warning: daemon did not confirm holding follower position; stop request skipped", file=stderr)
    except Exception as exc:
        print(f"warning: failed to confirm hold request: {exc}", file=stderr)
    should_stop = own_daemon or args.stop_external_daemon
    if not should_stop:
        print("External daemon left running; stop it manually when collection is finished.", file=stderr)
        return False
    if not hold_confirmed:
        print("Daemon left running because safe hold was not confirmed.", file=stderr)
        return False
    if not args.no_release_prompt and _is_tty(stdin):
        print("Press Enter to stop the daemon after the arm is supported; Panthera motors will be left in brake mode when available.", file=stderr)
        try:
            stdin.readline()
        except Exception:
            pass
    try:
        client.stop()
        return True
    except Exception as exc:
        print(f"warning: failed to stop daemon through IPC: {exc}", file=stderr)
        return bool(own_daemon and hold_confirmed)


def _wait_for_confirmed_hold(client: ControlClient, *, hold_status: dict[str, Any], timeout_s: float) -> bool:
    start_sequence = _status_sequence(hold_status)
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        timeout = 0.25
    if not math.isfinite(timeout) or timeout <= 0.0:
        timeout = 0.25
    deadline = time.monotonic() + max(0.05, timeout)
    sleep_s = min(max(timeout / 10.0, 0.001), 0.01)
    if start_sequence is None:
        return False
    while time.monotonic() < deadline:
        try:
            snapshot = client.snapshot(max_age_s=timeout)
            sequence = _status_sequence(snapshot)
            diagnostics = snapshot.get("diagnostics", {})
            if (
                sequence is not None
                and sequence > start_sequence
                and diagnostics.get("holding") is True
                and diagnostics.get("control_mode") == "hold"
            ):
                return True
        except ControlClientError:
            pass
        time.sleep(sleep_s)
    return False


def _status_sequence(status: dict[str, Any]) -> int | None:
    sequence = status.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        return None
    return sequence


def _wait_or_terminate_daemon(process: Any, *, stderr: Any) -> None:
    try:
        process.wait(timeout=2.0)
        return
    except Exception:
        pass
    try:
        process.terminate()
        process.wait(timeout=1.0)
        return
    except Exception as exc:
        print(f"warning: failed to terminate daemon process: {exc}", file=stderr)
    try:
        process.kill()
        process.wait(timeout=1.0)
    except Exception as exc:
        print(f"warning: failed to kill daemon process: {exc}", file=stderr)


def _daemon_log_path() -> str:
    with tempfile.NamedTemporaryFile(prefix="panthera_lerobot_daemon_", suffix=".log", delete=False) as tmp:
        return tmp.name


def _read_log_tail(path: str | None, *, max_chars: int = 4000) -> str:
    if not path:
        return ""
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return data[-max_chars:].strip()


def _record_status_update_from_line(line: str) -> _RecordStatusUpdate | None:
    match = re.search(r"\bRecording episode\s+(\d+)\b", line)
    if match:
        return _RecordStatusUpdate("RECORDING", f"Episode {match.group(1)}", beep_count=2)
    if "Reset the environment" in line:
        return _RecordStatusUpdate("RESET", "Prepare the scene for the next episode", beep_count=1)
    if "Re-record episode" in line or "Left arrow key pressed" in line:
        return _RecordStatusUpdate("RERECORD", "Rerecording the current episode", beep_count=2)
    if "Right arrow key pressed" in line:
        return _RecordStatusUpdate("ADVANCING", "Finishing the current phase", beep_count=0)
    if "Escape key pressed" in line or "Stop recording" in line:
        return _RecordStatusUpdate("STOPPING", "Finalizing dataset and stopping recording", beep_count=1)
    if re.search(r"\bExiting\b", line):
        return _RecordStatusUpdate("DONE", "LeRobot record finished", beep_count=0)
    return None


class _RecordStatusTracker:
    def __init__(self, *, status_ui: Any | None, beep_func: Callable[[int], None]):
        self.status_ui = status_ui
        self.beep_func = beep_func

    def process_line(self, line: str) -> _RecordStatusUpdate | None:
        update = _record_status_update_from_line(line)
        if update is None:
            return None
        if self.status_ui is not None:
            self.status_ui.set_status(update.phase, update.detail)
        if update.beep_count > 0:
            self.beep_func(update.beep_count)
        return update


def _start_record_status_ui(
    args: argparse.Namespace,
    *,
    stderr: Any,
    status_ui_factory: Callable[[Any], Any] | None = None,
) -> Any | None:
    if args.no_gui or args.no_status_ui:
        return None
    factory = status_ui_factory or (lambda stream: _RecordStatusUI(stderr=stream))
    try:
        ui = factory(stderr)
        return ui.start()
    except Exception as exc:
        print(f"warning: recording status GUI unavailable: {exc}", file=stderr)
        return None


class _RecordStatusUI:
    _COLORS = {
        "STARTING": "#2563eb",
        "RECORDING": "#dc2626",
        "RESET": "#d97706",
        "RERECORD": "#7c3aed",
        "ADVANCING": "#0891b2",
        "STOPPING": "#475569",
        "DONE": "#16a34a",
        "ERROR": "#b91c1c",
    }

    def __init__(self, *, stderr: Any):
        self.stderr = stderr
        self.root = None
        self.phase_label = None
        self.detail_label = None

    def start(self):
        try:
            import tkinter as tk
        except Exception as exc:
            print(f"warning: recording status GUI unavailable: {exc}", file=self.stderr)
            return None
        try:
            self.root = tk.Tk()
            self.root.title("Panthera LeRobot Status")
            self.root.resizable(False, False)
            try:
                self.root.attributes("-topmost", True)
            except Exception:
                pass
            tk.Label(
                self.root,
                text="Panthera LeRobot",
                font=("TkDefaultFont", 13, "bold"),
            ).pack(padx=14, pady=(14, 8), fill="x")
            self.phase_label = tk.Label(
                self.root,
                text="STARTING",
                width=20,
                height=2,
                font=("TkDefaultFont", 24, "bold"),
                bg=self._COLORS["STARTING"],
                fg="white",
            )
            self.phase_label.pack(padx=14, pady=(0, 8), fill="x")
            self.detail_label = tk.Label(
                self.root,
                text="Launching LeRobot record",
                width=44,
                height=2,
                font=("TkDefaultFont", 11),
                justify="center",
                wraplength=360,
            )
            self.detail_label.pack(padx=14, pady=(0, 8), fill="x")
            tk.Label(
                self.root,
                text="Keys: -> next phase    <- rerecord    Esc stop",
                font=("TkDefaultFont", 10),
                fg="#334155",
            ).pack(padx=14, pady=(0, 14), fill="x")
            self.pump()
            return self
        except Exception as exc:
            print(f"warning: recording status GUI unavailable: {exc}", file=self.stderr)
            self.root = None
            return None

    def set_status(self, phase: str, detail: str) -> None:
        if self.root is None:
            return
        color = self._COLORS.get(phase, "#475569")
        try:
            if self.phase_label is not None:
                self.phase_label.configure(text=phase, bg=color)
            if self.detail_label is not None:
                self.detail_label.configure(text=detail)
            self.root.lift()
        except Exception as exc:
            print(f"warning: recording status GUI stopped: {exc}", file=self.stderr)
            self.root = None

    def pump(self) -> None:
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception as exc:
            print(f"warning: recording status GUI stopped: {exc}", file=self.stderr)
            self.root = None

    def shutdown(self) -> None:
        if self.root is None:
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None


def _enqueue_process_output(stream: Any, output_queue: queue.Queue[Any]) -> None:
    try:
        for line in stream:
            output_queue.put(line)
    finally:
        output_queue.put(None)


def _drain_record_output(
    output_queue: queue.Queue[Any],
    *,
    tracker: _RecordStatusTracker,
    stream: Any,
) -> bool:
    stream_done = False
    while True:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            stream_done = True
            continue
        try:
            stream.write(item)
            stream.flush()
        except Exception:
            pass
        tracker.process_line(str(item))
    return stream_done


def _default_record_runner(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    status_ui: Any | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    graceful_timeout_s: float = 30.0,
    terminate_timeout_s: float = 5.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    killpg: Callable[[int, int], None] = os.killpg,
) -> int:
    output_stream = sys.stderr if stderr is None else stderr
    beep_stream = sys.stdout if stdout is None else stdout
    process = None
    reader = None
    try:
        try:
            process = popen_factory(
                list(command),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise CollectError(f"Failed to run lerobot-record: {exc}") from exc

        output_queue: queue.Queue[Any] = queue.Queue()
        if process.stdout is None:
            return int(process.wait())

        reader = threading.Thread(
            target=_enqueue_process_output,
            args=(process.stdout, output_queue),
            daemon=True,
        )
        reader.start()
        tracker = _RecordStatusTracker(
            status_ui=status_ui,
            beep_func=lambda count: beep(count, stdout=beep_stream),
        )
        stream_done = False
        while True:
            stream_done = _drain_record_output(output_queue, tracker=tracker, stream=output_stream) or stream_done
            return_code = process.poll()
            if status_ui is not None:
                status_ui.pump()
            if return_code is not None and stream_done:
                break
            sleep(0.05)
        reader.join(timeout=1.0)
        _drain_record_output(output_queue, tracker=tracker, stream=output_stream)
        return int(process.returncode if process.returncode is not None else return_code)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt) and status_ui is not None:
            status_ui.set_status("STOPPING", "Interrupted; forwarding stop to LeRobot")
        if process is not None:
            _shutdown_record_process_group(
                process,
                graceful_timeout_s=graceful_timeout_s,
                terminate_timeout_s=terminate_timeout_s,
                killpg=killpg,
            )
        if reader is not None:
            reader.join(timeout=1.0)
        raise


def _shutdown_record_process_group(
    process: Any,
    *,
    graceful_timeout_s: float,
    terminate_timeout_s: float,
    killpg: Callable[[int, int], None] = os.killpg,
) -> None:
    if process.poll() is not None:
        _reap_record_process(process)
        return

    if not _send_record_group_signal(process, signal.SIGINT, killpg):
        _reap_record_process(process)
        return
    if _wait_for_record_process(process, graceful_timeout_s):
        return

    if process.poll() is None:
        _send_record_group_signal(process, signal.SIGTERM, killpg)
    if _wait_for_record_process(process, terminate_timeout_s):
        return

    if process.poll() is None:
        _send_record_group_signal(process, signal.SIGKILL, killpg)
    _reap_record_process(process)


def _send_record_group_signal(process: Any, signum: int, killpg: Callable[[int, int], None]) -> bool:
    try:
        killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False


def _wait_for_record_process(process: Any, timeout_s: float) -> bool:
    try:
        process.wait(timeout=float(timeout_s))
        return True
    except subprocess.TimeoutExpired:
        return False
    except KeyboardInterrupt:
        return False


def _reap_record_process(process: Any) -> None:
    while True:
        try:
            process.wait()
            return
        except KeyboardInterrupt:
            continue


def _stdin_requested(stdin: Any) -> bool:
    if not _is_tty(stdin):
        return False
    try:
        readable, _, _ = select.select([stdin], [], [], 0.0)
    except Exception:
        return False
    if not readable:
        return False
    try:
        stdin.readline()
    except Exception:
        return False
    return True


def _is_tty(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _default_camera_label(
    role_name: str,
    devices: Sequence[CameraDevice],
    current_specs: Sequence[CameraSpec],
    assigned_labels: set[str],
) -> str:
    current_source = next((spec.source for spec in current_specs if spec.name == role_name), None)
    if current_source is not None:
        for device in devices:
            if device.source == current_source or device.resolved == current_source:
                assigned_labels.add(device.label)
                return device.label
    for device in devices:
        if device.label not in assigned_labels:
            assigned_labels.add(device.label)
            return device.label
    return devices[0].label


def _camera_source(value: str) -> int | str:
    normalized = str(value).strip()
    try:
        number = int(normalized)
    except ValueError:
        return value
    if number < 0:
        raise CollectError(f"Camera index must be non-negative, got {value!r}")
    return number


def _show_preview_images(images: dict[str, Any], *, parent: Any | None, tk: Any) -> None:
    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title("Panthera LeRobot Camera Preview")
    root.resizable(False, False)
    tk.Label(root, text="Camera preview", font=("TkDefaultFont", 13, "bold")).grid(
        row=0,
        column=0,
        columnspan=max(1, len(images)),
        padx=12,
        pady=(12, 6),
        sticky="w",
    )
    image_refs = []
    for column, (camera_name, image) in enumerate(images.items()):
        tk.Label(root, text=camera_name).grid(row=1, column=column, padx=12, pady=(0, 4), sticky="w")
        photo = _tk_photo_image_from_bgr(tk, image)
        image_refs.append(photo)
        tk.Label(root, image=photo).grid(row=2, column=column, padx=12, pady=(0, 8))
    tk.Button(root, text="Close", command=root.destroy).grid(
        row=3,
        column=0,
        columnspan=max(1, len(images)),
        padx=12,
        pady=(0, 12),
        sticky="e",
    )
    root._panthera_image_refs = image_refs
    if parent is not None:
        root.transient(parent)
        root.grab_set()
    root.wait_window()


def _tk_photo_image_from_bgr(tk: Any, image: Any, *, max_width: int = 640, max_height: int = 480) -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise CollectError("Camera preview requires NumPy.") from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CollectError(f"Unsupported preview image shape: {getattr(array, 'shape', None)}")
    array = array[:, :, :3][:, :, ::-1]
    height, width = int(array.shape[0]), int(array.shape[1])
    stride = max(1, int(math.ceil(max(width / max_width, height / max_height))))
    if stride > 1:
        array = array[::stride, ::stride, :]
    array = np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
    height, width = int(array.shape[0]), int(array.shape[1])
    ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + array.tobytes()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="panthera_lerobot_camera_preview_", suffix=".ppm", delete=False) as tmp:
            tmp.write(ppm)
            temp_path = tmp.name
        return tk.PhotoImage(file=temp_path)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class _PreviewStartUI:
    def __init__(self, *, stderr: Any):
        self.stderr = stderr
        self.root = None
        self.start_requested = False

    def start(self):
        try:
            import tkinter as tk
        except Exception as exc:
            print(f"warning: Preview GUI unavailable: {exc}", file=self.stderr)
            return None
        try:
            self.root = tk.Tk()
            self.root.title("Panthera LeRobot Collect")
            self.root.resizable(False, False)
            tk.Label(
                self.root,
                text="Preview",
                width=34,
                height=2,
                font=("TkDefaultFont", 15, "bold"),
                bg="#4f46e5",
                fg="white",
            ).pack(padx=12, pady=(12, 8), fill="x")
            tk.Button(self.root, text="Start recording", command=self._request_start).pack(
                padx=12,
                pady=(0, 12),
                fill="x",
            )
            self.root.bind("<Return>", lambda _event: self._request_start())
            return self
        except Exception as exc:
            print(f"warning: Preview GUI unavailable: {exc}", file=self.stderr)
            self.root = None
            return None

    def pump(self) -> None:
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception as exc:
            print(f"warning: Preview GUI stopped: {exc}", file=self.stderr)
            self.root = None

    def shutdown(self) -> None:
        if self.root is None:
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None

    def _request_start(self) -> None:
        self.start_requested = True


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _format_number(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def _bool_lower(value: bool) -> str:
    return "true" if value else "false"


def _bool_title(value: bool) -> str:
    return "True" if value else "False"


if __name__ == "__main__":
    raise SystemExit(main())
