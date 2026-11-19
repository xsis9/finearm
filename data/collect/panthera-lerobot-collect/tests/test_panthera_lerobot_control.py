import inspect
import math
import os
import io
import socket
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from panthera_lerobot_control.adapters import (
    AdapterError,
    ArmState,
    MockRobotAdapter,
    PantheraRobotAdapter,
    make_follower_only,
    make_panthera_pair,
    _discover_default_sdk_root,
    _validate_panthera_serial_access,
    ensure_panthera_serial_access,
)
from panthera_lerobot_control.ipc import ControlServer
from panthera_lerobot_control import cli as control_cli
from panthera_lerobot_control import (
    ControlClient,
    ControlConfig,
    ControlDaemon,
    DaemonDisconnectedError,
    MalformedResponseError,
    StaleSnapshotError,
)


class PantheraLeRobotControlTests(unittest.TestCase):
    def test_package_metadata_declares_panthera_control_daemon_script(self):
        project_root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

        self.assertEqual(
            metadata["project"]["scripts"]["panthera-control-daemon"],
            "panthera_lerobot_control:main",
        )

    def test_daemon_cli_defaults_to_hold_mode(self):
        args = control_cli.build_parser().parse_args(["--mock"])

        self.assertEqual(args.initial_control_mode, "hold")

    def test_cli_module_entrypoint_starts_mock_daemon_until_terminated(self):
        socket_path = self._socket_path()
        project_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)
        env["PYTHONNOUSERSITE"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "panthera_lerobot_control.cli",
                "--mock",
                "--socket",
                socket_path,
                "--loop-hz",
                "20",
            ],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1.0)
                    self.fail(f"daemon CLI exited early with {process.returncode}: stdout={stdout!r} stderr={stderr!r}")
                if Path(socket_path).exists():
                    break
                time.sleep(0.02)
            else:
                self.fail("daemon CLI did not create its IPC socket")

            client = ControlClient(socket_path)
            self.assertEqual(client.status()["mode"], "mock")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def test_daemon_cli_signal_handlers_defer_stop_and_restore_previous_handlers(self):
        managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

        for triggered_signal in managed_signals:
            with self.subTest(triggered_signal=triggered_signal):
                events = []
                installed = {}
                previous = {signum: object() for signum in managed_signals}
                signal_calls = []

                class FakeDaemon:
                    def __init__(self, _config):
                        pass

                    def start(self):
                        events.append("start")

                    def stop(self):
                        events.append("stop")

                def fake_signal(signum, handler):
                    signal_calls.append((signum, handler))
                    if handler is not previous[signum]:
                        installed[signum] = handler

                def fake_sleep(_seconds):
                    self.assertEqual(events, ["start"])
                    handler = installed.get(triggered_signal)
                    self.assertIsNotNone(handler)
                    handler(triggered_signal, None)
                    self.assertEqual(events, ["start"])

                with mock.patch.object(control_cli, "ControlDaemon", FakeDaemon), mock.patch.object(
                    control_cli.signal, "getsignal", side_effect=previous.__getitem__
                ), mock.patch.object(
                    control_cli.signal, "signal", side_effect=fake_signal
                ), mock.patch.object(
                    control_cli.time, "sleep", side_effect=fake_sleep
                ):
                    rc = control_cli.main(["--mock"])

                self.assertEqual(rc, 0)
                self.assertEqual(events, ["start", "stop"])
                for signum in managed_signals:
                    self.assertIn((signum, previous[signum]), signal_calls)

    def test_panthera_pair_uses_explicit_sdk_root_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk_root = Path(tmp) / "custom_sdk" / "panthera_python"
            created_paths = []

            make_panthera_pair(sdk_root=sdk_root, factory=created_paths.append)

        self.assertEqual(
            created_paths,
            [
                str(sdk_root / "robot_param" / "Leader.yaml"),
                str(sdk_root / "robot_param" / "Follower.yaml"),
            ],
        )

    def test_panthera_pair_accepts_sdk_repo_root_as_explicit_sdk_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk_repo = Path(tmp) / "Panthera-HT_SDK"
            python_root = sdk_repo / "panthera_python"
            python_root.mkdir(parents=True)
            created_paths = []

            make_panthera_pair(sdk_root=sdk_repo, factory=created_paths.append)

        self.assertEqual(
            created_paths,
            [
                str(python_root / "robot_param" / "Leader.yaml"),
                str(python_root / "robot_param" / "Follower.yaml"),
            ],
        )

    def test_config_paths_resolve_against_deploy_root_from_any_cwd(self):
        # deploy 是项目根:相对 leader/follower 配置路径按 <deploy> 根解析,
        # 与发起命令的 CWD 无关(项目可克隆到任意位置、从任意目录启动);
        # <deploy> 下不存在时回退按 CWD 解析;绝对路径原样。
        created_paths = []
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cwd_only.yaml").write_text("x", encoding="utf-8")
            os.chdir(tmp)
            try:
                make_panthera_pair(
                    leader_config="arm/panthera/robot_param/Leader.yaml",
                    follower_config="arm/panthera/robot_param/Follower.yaml",
                    factory=created_paths.append,
                )
                make_follower_only(follower_config="cwd_only.yaml", factory=created_paths.append)
            finally:
                os.chdir(old_cwd)
        deploy_root = Path(__file__).resolve().parents[4]
        self.assertEqual(
            created_paths[:2],
            [
                str(deploy_root / "arm" / "panthera" / "robot_param" / "Leader.yaml"),
                str(deploy_root / "arm" / "panthera" / "robot_param" / "Follower.yaml"),
            ],
        )
        for p in created_paths[:2]:
            self.assertTrue(Path(p).exists(), f"{p} should exist under deploy root")
        # deploy 根下没有 cwd_only.yaml → 回退 CWD 相对(临时目录),证明两级查找顺序
        self.assertEqual(created_paths[2], str(Path(tmp) / "cwd_only.yaml"))

    def test_follower_only_constructs_single_follower_with_mock_leader(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk_root = Path(tmp) / "custom_sdk" / "panthera_python"
            created_paths = []

            leader, follower = make_follower_only(sdk_root=sdk_root, factory=created_paths.append)

        self.assertEqual(created_paths, [str(sdk_root / "robot_param" / "Follower.yaml")])
        self.assertIsInstance(leader, MockRobotAdapter)
        self.assertEqual(leader.role, "leader")
        self.assertIsInstance(follower, PantheraRobotAdapter)

    def test_no_leader_config_rejects_teleop_initial_control_mode(self):
        with self.assertRaisesRegex(ValueError, "no_leader"):
            ControlConfig(no_leader=True, initial_control_mode="teleop")

    def test_no_leader_daemon_rejects_teleop_control_mode(self):
        daemon = ControlDaemon(ControlConfig(no_leader=True, initial_control_mode="hold"))

        with self.assertRaisesRegex(Exception, "teleop"):
            daemon.set_control_mode("teleop")

        daemon.set_control_mode("policy")
        self.assertEqual(daemon.control_mode, "policy")

    def test_daemon_cli_parses_no_leader_flag(self):
        args = control_cli.build_parser().parse_args(["--no-leader"])

        self.assertTrue(args.no_leader)
        self.assertFalse(args.mock)

    def test_daemon_cli_parses_park_options(self):
        args = control_cli.build_parser().parse_args(["--park-joints", "0.1,0.2,0.3,0.4,0.5,0.6", "--park-move-s", "2"])

        self.assertFalse(args.no_park)
        self.assertEqual(args.park_joints, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(args.park_move_s, 2.0)

        disabled = control_cli.build_parser().parse_args(["--no-park"])
        self.assertTrue(disabled.no_park)
        self.assertIsNone(disabled.park_joints)

    def test_plan_park_actions_ramps_from_start_to_target(self):
        start = [0.0] * 6
        target = [1.0] * 6

        actions = control_cli.plan_park_actions(start, target, gripper=0.5, steps=10)

        self.assertEqual(len(actions), 10)
        self.assertTrue(all(len(action) == 7 for action in actions))
        self.assertTrue(all(action[6] == 0.5 for action in actions))
        self.assertTrue(all(a <= b for a, b in zip(actions[0], actions[1])))
        self.assertAlmostEqual(actions[-1][0], 1.0)
        mid = actions[len(actions) // 2][0]
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

        with self.assertRaises(ValueError):
            control_cli.plan_park_actions([0.0] * 3, target, gripper=0.5, steps=10)


    def test_panthera_adapter_stop_prefers_brake_over_stop(self):
        robot = BrakeStopRobot()
        adapter = PantheraRobotAdapter(robot)

        adapter.stop()

        self.assertEqual(robot.calls, ["set_brake"])

    def test_panthera_adapter_stop_does_not_send_stop_when_brake_binding_is_missing(self):
        robot = StopOnlyRobot()
        adapter = PantheraRobotAdapter(robot)

        adapter.stop()

        self.assertEqual(robot.calls, [])

    def test_panthera_adapter_rejects_non_finite_sdk_state(self):
        invalid_fields = {
            "positions": [float("nan")] + [0.0] * 5,
            "velocities": [float("inf")] + [0.0] * 5,
            "torques": [float("-inf")] + [0.0] * 5,
            "gravity": [float("nan")] + [0.0] * 5,
            "gripper_position": float("inf"),
            "gripper_velocity": float("nan"),
            "gripper_torque": float("-inf"),
        }

        for field, value in invalid_fields.items():
            with self.subTest(field=field):
                robot = FakeSdkStateRobot()
                setattr(robot, field, value)

                with self.assertRaisesRegex(AdapterError, field):
                    PantheraRobotAdapter(robot).read_state()

    def test_panthera_adapter_rejects_non_finite_friction_compensation(self):
        robot = FakeSdkStateRobot()
        robot.friction_compensation = [float("nan")] + [0.0] * 5

        with self.assertRaisesRegex(AdapterError, "friction_compensation"):
            PantheraRobotAdapter(robot).friction(
                [0.0] * 6, [0.0] * 6, [0.0] * 6, 0.1
            )

    def test_live_serial_preflight_reports_missing_tty_permissions(self):
        with self.assertRaises(AdapterError) as ctx:
            _validate_panthera_serial_access(
                paths=[Path("/dev/ttyACM1"), Path("/dev/ttyACM2")],
                access_fn=lambda _path, _mode: False,
            )

        message = str(ctx.exception)
        self.assertIn("/dev/ttyACM1", message)
        self.assertIn("dialout", message)
        self.assertIn("sudo usermod -aG dialout", message)

    def test_live_serial_permission_request_describes_scope_and_uses_setfacl(self):
        stderr = io.StringIO()
        commands = []
        accessible = {"ok": False}

        def access_fn(_path, _mode):
            return accessible["ok"]

        def runner(command, check=False):
            commands.append(list(command))
            accessible["ok"] = True

            class Result:
                returncode = 0

            return Result()

        ensure_panthera_serial_access(
            paths=[Path("/dev/ttyACM1"), Path("/dev/ttyACM2")],
            access_fn=access_fn,
            runner=runner,
            stderr=stderr,
            user="jiaming",
        )

        self.assertEqual(
            commands,
            [["sudo", "setfacl", "-m", "u:jiaming:rw", "/dev/ttyACM1", "/dev/ttyACM2"]],
        )
        text = stderr.getvalue()
        self.assertIn("temporary read/write", text)
        self.assertIn("Panthera SDK", text)
        self.assertIn("/dev/ttyACM1", text)
        self.assertIn("does not grant camera", text)

    def test_default_sdk_discovery_can_search_injected_workspace_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "Panthera-HT"
            sdk_root = workspace / "Panthera-HT_SDK" / "panthera_python"
            (sdk_root / "scripts").mkdir(parents=True)
            (sdk_root / "robot_param").mkdir()
            worktree = workspace / ".worktrees" / "lerobot-native-control"
            package_file = worktree / "panthera_lerobot_control" / "adapters.py"
            package_file.parent.mkdir(parents=True)
            package_file.write_text("# fake package path")

            self.assertEqual(
                _discover_default_sdk_root(cwd=worktree, package_file=package_file),
                sdk_root,
            )

    def test_default_sdk_discovery_accepts_panthera_sdk_root_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk_root = Path(tmp) / "Panthera-HT_SDK" / "panthera_python"
            (sdk_root / "scripts").mkdir(parents=True)
            (sdk_root / "robot_param").mkdir()
            old_value = os.environ.get("PANTHERA_SDK_ROOT")
            os.environ["PANTHERA_SDK_ROOT"] = str(sdk_root.parent)
            try:
                self.assertEqual(
                    _discover_default_sdk_root(
                        cwd=Path("/__panthera_missing_workspace__"),
                        package_file=Path("/__panthera_missing_package__/adapters.py"),
                    ),
                    sdk_root,
                )
            finally:
                if old_value is None:
                    os.environ.pop("PANTHERA_SDK_ROOT", None)
                else:
                    os.environ["PANTHERA_SDK_ROOT"] = old_value

    def test_small_nonzero_gripper_torque_is_sent_to_leader(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.7,
                gripper_velocity=0.2,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.1,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = follower

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.gripper_commands[-1][2], 0.06)

    def test_policy_mode_does_not_feed_follower_gripper_torque_to_leader(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.7,
                gripper_velocity=0.2,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.1,
                gripper_velocity=0.0,
                gripper_torque=0.8,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        daemon.submit_policy_action([0.0] * 7)

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][2], [0.0] * 6)
        self.assertEqual(leader.gripper_commands[-1][2], 0.0)
        self.assertEqual(leader.gripper_commands[-1][3], 0.0)
        self.assertEqual(leader.gripper_commands[-1][4], 0.0)

    def test_leader_park_ramps_leader_to_target_with_soft_gains(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.2, -0.2, 0.4, -0.4, 0.1, -0.1],
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.05] * 6,
                gripper_position=0.7,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.0,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                policy_mirror_leader_kp=[0.3] * 6,
                policy_mirror_leader_kd=[0.03] * 6,
            )
        )
        daemon._leader = leader
        daemon._follower = follower

        # Ramp start: the commanded target equals the captured leader pose.
        daemon._leader_park = {"target": [0.0] * 6, "duration": 0.2, "start": None, "t0": None}
        daemon._step(time.monotonic(), 0.001, 0.001)
        self.assertIsNotNone(daemon._leader_park)
        self.assertEqual(leader.arm_commands[-1][0], [0.2, -0.2, 0.4, -0.4, 0.1, -0.1])
        self.assertEqual(leader.arm_commands[-1][3], [0.3] * 6)
        self.assertEqual(leader.arm_commands[-1][4], [0.03] * 6)
        # Gravity feedforward is kept during the ramp.
        self.assertEqual(leader.arm_commands[-1][2], [0.05] * 6)

        # Ramp end: the commanded target reaches the park target and the done event fires.
        park = daemon._leader_park
        park["t0"] = time.monotonic() - park["duration"] - 0.01
        daemon._leader_park_done.clear()
        daemon._step(time.monotonic(), 0.001, 0.001)
        self.assertIsNone(daemon._leader_park)
        self.assertTrue(daemon._leader_park_done.is_set())
        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)

    def test_park_leader_rejected_without_leader_hardware(self):
        daemon = ControlDaemon(
            ControlConfig(mock=True, no_leader=True, initial_control_mode="hold", socket_path=self._socket_path())
        )
        self.assertFalse(daemon.park_leader(0.1))

    def test_policy_mode_mirrors_policy_target_to_leader_with_rate_limit(self):
        # 2026-11-09 起镜像语义 = 把 policy 下发目标同发给主臂(从臂位姿级联
        # 因弹簧+摩擦带不动,实测弃用,见 controller.py policy-mirror 分支注释)。
        # 本测试提交与 follower 位姿同值的 policy 目标,断言镜像与步距限幅。
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.2,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.2, -0.2, 0.05, -0.05, 0.01, -0.01],
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=1.0,
                gripper_velocity=0.0,
                gripper_torque=0.8,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                policy_mirror_leader=True,
                policy_mirror_max_step_rad=0.05,
                policy_mirror_max_gripper_step=0.2,
                policy_mirror_leader_kp=[0.3] * 6,
                policy_mirror_leader_kd=[0.03] * 6,
                policy_mirror_gripper_kp=0.4,
                policy_mirror_gripper_kd=0.04,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        daemon.submit_policy_action([0.2, -0.2, 0.05, -0.05, 0.01, -0.01, 1.0])

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][0], [0.05, -0.05, 0.05, -0.05, 0.01, -0.01])
        self.assertEqual(leader.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][2], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.3] * 6)
        self.assertEqual(leader.arm_commands[-1][4], [0.03] * 6)
        self.assertAlmostEqual(leader.gripper_commands[-1][0], 0.4)
        self.assertEqual(leader.gripper_commands[-1][2], 0.0)
        self.assertEqual(leader.gripper_commands[-1][3], 0.4)
        self.assertEqual(leader.gripper_commands[-1][4], 0.04)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_mirror_leader"])
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_mirror_limited"])

    def test_policy_mirror_is_default_off_and_waits_for_real_policy_action(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.2,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.2, -0.2, 0.05, -0.05, 0.01, -0.01],
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=1.0,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )

        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                policy_mirror_leader=True,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.0] * 6)
        self.assertEqual(leader.gripper_commands[-1][0], leader.state.gripper_position)
        self.assertEqual(leader.gripper_commands[-1][3], 0.0)
        self.assertFalse(daemon.latest_snapshot()["diagnostics"]["policy_mirror_leader"])

        daemon.submit_policy_action([0.0] * 7)
        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_mirror_leader"])

        default_off = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        default_off._leader = leader
        default_off._follower = follower
        default_off.set_control_mode("policy")
        default_off.submit_policy_action([0.0] * 7)

        default_off._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertFalse(default_off.latest_snapshot()["diagnostics"]["policy_mirror_leader"])

    def test_daemon_seeds_policy_action_from_current_follower_state(self):
        from panthera_lerobot_control import constants as c

        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._snapshot = c.make_snapshot(
            follower_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            follower_velocities=[0.0] * 6,
            follower_gripper_position=1.7,
            follower_gripper_velocity=0.0,
            target_positions=[9.0] * 6,
            target_gripper_position=9.0,
            diagnostics={},
            timestamp=1.0,
            sequence=1,
            loop_timing={},
        )

        daemon.set_control_mode("policy")

        self.assertEqual(daemon.control_mode, "policy")
        self.assertEqual(daemon._policy_action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.7])
        self.assertTrue(daemon._policy_action_is_seed)
        self.assertIsNotNone(daemon._policy_action_timestamp)

    def test_daemon_rejects_non_finite_policy_seed_snapshot(self):
        from panthera_lerobot_control import constants as c

        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._snapshot = c.make_snapshot(
            follower_positions=[0.1, float("nan"), 0.3, 0.4, 0.5, 0.6],
            follower_velocities=[0.0] * 6,
            follower_gripper_position=1.7,
            follower_gripper_velocity=0.0,
            target_positions=[9.0] * 6,
            target_gripper_position=9.0,
            diagnostics={},
            timestamp=1.0,
            sequence=1,
            loop_timing={},
        )
        daemon.submit_policy_action([0.0] * 7)

        daemon.set_control_mode("policy")

        self.assertEqual(daemon.control_mode, "policy")
        self.assertIsNone(daemon._policy_action)
        self.assertIsNone(daemon._policy_action_timestamp)
        self.assertFalse(daemon._policy_action_is_seed)

    def test_policy_mode_limits_target_lead_from_current_follower_state(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.0,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.2,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                max_target_lead_rad=0.05,
                max_gripper_lead=0.1,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        daemon.submit_policy_action([0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 1.5])

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(follower.arm_commands[-1][0], [0.05, -0.05, 0.05, -0.05, 0.05, -0.05])
        self.assertAlmostEqual(follower.gripper_commands[-1][0], 0.3)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_target_lead_limited"])

    def test_ipc_server_times_out_idle_request_clients(self):
        server = ControlServer(self._socket_path(), DummyDaemon(), read_timeout_s=0.05)
        server.start()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.5)
                client.connect(server.socket_path)
                raw = client.recv(4096)
        finally:
            server.stop()

        self.assertIn(b"timed out", raw)

    def test_ipc_server_rejects_oversized_request_lines(self):
        server = ControlServer(self._socket_path(), DummyDaemon(), max_request_bytes=32)
        server.start()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.5)
                client.connect(server.socket_path)
                client.sendall(b"{" + b"x" * 64 + b"}\n")
                raw = client.recv(4096)
        finally:
            server.stop()

        self.assertIn(b"too large", raw)

    def test_ipc_server_refuses_existing_regular_file_socket_path(self):
        path = self._socket_path()
        Path(path).write_text("not a socket")
        server = ControlServer(path, DummyDaemon())
        try:
            with self.assertRaises(RuntimeError):
                server.start()
            self.assertEqual(Path(path).read_text(), "not a socket")
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def test_ipc_server_refuses_active_socket_path(self):
        path = self._socket_path()
        active = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        active.bind(path)
        active.listen(1)
        server = ControlServer(path, DummyDaemon())
        try:
            with self.assertRaises(RuntimeError):
                server.start()
        finally:
            active.close()
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def test_ipc_server_cleans_up_stale_socket_path(self):
        path = self._socket_path()
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(path)
        stale.close()
        server = ControlServer(path, DummyDaemon())
        try:
            server.start()
            self.assertTrue(Path(path).exists())
        finally:
            server.stop()

    def test_latest_snapshot_returns_deep_copy_of_nested_values(self):
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._snapshot["diagnostics"]["nested"] = ["original"]

        snapshot = daemon.latest_snapshot()
        snapshot["diagnostics"]["nested"].append("mutated")

        self.assertEqual(daemon.latest_snapshot()["diagnostics"]["nested"], ["original"])

    def test_start_stops_created_adapters_when_ipc_startup_fails(self):
        path = self._socket_path()
        Path(path).write_text("not a socket")
        leader = LifecycleRobotAdapter()
        follower = LifecycleRobotAdapter()
        daemon = ControlDaemon(
            ControlConfig(mock=True, socket_path=path),
            adapter_factory=lambda: (leader, follower),
        )

        try:
            with self.assertRaises(RuntimeError):
                daemon.start()
            self.assertTrue(leader.stopped)
            self.assertTrue(follower.stopped)
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def test_snapshot_schema_has_complete_numeric_observation_state_and_action(self):
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        try:
            daemon.start()
            snapshot = self._wait_for_snapshot(daemon)
        finally:
            daemon.stop()

        self.assertIn("observation.state", snapshot)
        self.assertIn("action", snapshot)
        self.assertEqual(len(snapshot["observation.state"]), 14)
        self.assertEqual(len(snapshot["action"]), 7)
        for value in snapshot["observation.state"] + snapshot["action"]:
            self.assertIsInstance(value, (int, float))
            self.assertTrue(math.isfinite(value))

        self.assertIn("diagnostics", snapshot)
        self.assertIn("timestamp", snapshot)
        self.assertIn("sequence", snapshot)
        self.assertIn("loop_timing", snapshot)

    def test_fake_daemon_supports_status_and_snapshot_over_ipc(self):
        socket_path = self._socket_path()
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=socket_path))
        try:
            daemon.start()
            client = ControlClient(socket_path)
            status = client.status()
            snapshot = client.snapshot()
        finally:
            daemon.stop()

        self.assertEqual(status["mode"], "mock")
        self.assertTrue(status["running"])
        self.assertGreaterEqual(status["sequence"], 0)
        self.assertEqual(len(snapshot["observation.state"]), 14)
        self.assertEqual(len(snapshot["action"]), 7)

    def test_ipc_client_can_switch_control_mode_and_submit_policy_action(self):
        daemon = DummyDaemon()
        server = ControlServer(self._socket_path(), daemon)
        server.start()
        try:
            client = ControlClient(server.socket_path)

            status = client.set_mode("policy")
            action_status = client.send_policy_action([float(i) for i in range(7)])
        finally:
            server.stop()

        self.assertEqual(daemon.control_mode, "policy")
        self.assertEqual(daemon.policy_actions, [[float(i) for i in range(7)]])
        self.assertEqual(status["control_mode"], "policy")
        self.assertEqual(action_status["policy_sequence"], 1)

    def test_policy_mode_uses_fresh_policy_action_and_clamps_targets(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.1] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.7,
                gripper_velocity=0.2,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.1,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                joint_min=[-1.0] * 6,
                joint_max=[1.0] * 6,
                gripper_min=0.0,
                gripper_max=1.0,
                max_target_lead_rad=10.0,
                max_gripper_lead=10.0,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        daemon.submit_policy_action([2.0, -2.0, 0.5, 0.4, 0.3, 0.2, 1.5])

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(follower.arm_commands[-1][0], [1.0, -1.0, 0.5, 0.4, 0.3, 0.2])
        self.assertEqual(follower.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(follower.gripper_commands[-1][0], 1.0)
        snapshot = daemon.latest_snapshot()
        self.assertEqual(snapshot["diagnostics"]["control_mode"], "policy")
        self.assertFalse(snapshot["diagnostics"]["policy_stale"])

    def test_policy_mode_holds_when_action_is_missing_or_stale(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.9] * 6,
                velocities=[0.1] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.9,
                gripper_velocity=0.2,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.2, 0.1, 0.0, -0.1, -0.2, -0.3],
                velocities=[0.4] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.3,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                policy_action_timeout_s=0.01,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        with daemon._control_lock:
            daemon._policy_action = None
            daemon._policy_action_timestamp = None

        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(follower.arm_commands[-1][0], follower.state.positions)
        self.assertEqual(follower.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(follower.gripper_commands[-1][0], follower.state.gripper_position)
        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.0] * 6)
        self.assertEqual(leader.gripper_commands[-1][0], leader.state.gripper_position)
        self.assertEqual(leader.gripper_commands[-1][3], 0.0)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_stale"])

        daemon.submit_policy_action([0.0] * 7)
        time.sleep(0.02)
        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertEqual(follower.arm_commands[-1][0], follower.state.positions)
        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.0] * 6)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["policy_stale"])

    def test_freedrive_mode_commands_zero_gains_with_gravity_feedforward(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.1] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.7,
                gripper_velocity=0.2,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[1.0] * 6,
                gripper_position=0.5,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                initial_control_mode="freedrive",
            )
        )
        daemon._leader = leader
        daemon._follower = follower

        daemon._step(time.monotonic(), 0.001, 0.001)

        positions, velocities, torques, kp, kd = follower.arm_commands[-1]
        self.assertEqual(positions, follower.state.positions)
        self.assertEqual(velocities, [0.0] * 6)
        self.assertEqual(torques, [1.0] * 6)
        self.assertEqual(kp, [0.0] * 6)
        self.assertEqual(kd, [0.0] * 6)
        gripper_position, _gripper_velocity, gripper_torque, gripper_kp, gripper_kd = follower.gripper_commands[-1]
        self.assertEqual(gripper_position, follower.state.gripper_position)
        self.assertEqual(gripper_torque, 0.0)
        self.assertEqual(gripper_kp, 0.0)
        self.assertEqual(gripper_kd, 0.0)
        snapshot = daemon.latest_snapshot()
        self.assertEqual(snapshot["diagnostics"]["control_mode"], "freedrive")
        self.assertFalse(snapshot["diagnostics"]["holding"])
        self.assertEqual(snapshot["action"], follower.state.positions + [follower.state.gripper_position])

    def test_freedrive_mode_restores_position_gains_on_hold(self):
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.1] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[1.0] * 6,
                gripper_position=0.5,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                initial_control_mode="freedrive",
            )
        )
        daemon._leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.0] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.5,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon._follower = follower

        daemon._step(time.monotonic(), 0.001, 0.001)
        self.assertEqual(follower.arm_commands[-1][3], [0.0] * 6)

        daemon.hold()
        daemon._step(time.monotonic(), 0.001, 0.001)

        self.assertTrue(all(value > 0.0 for value in follower.arm_commands[-1][3]))
        self.assertTrue(all(value > 0.0 for value in follower.arm_commands[-1][4]))
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["holding"])

    def test_ipc_client_accepts_freedrive_control_mode(self):
        daemon = DummyDaemon()
        server = ControlServer(self._socket_path(), daemon)
        server.start()
        try:
            client = ControlClient(server.socket_path)

            status = client.set_mode("freedrive")
        finally:
            server.stop()

        self.assertEqual(daemon.control_mode, "freedrive")
        self.assertEqual(status["control_mode"], "freedrive")

    def test_teleop_squeeze_boost_deepens_blocked_gripper_target(self):
        cases = [
            # (follower_gripper, leader_gripper, boost, expected_target)
            (0.6, 0.2, 1.0, -0.2),   # blocked open: deepen by the closing error
            (1.5, 0.0, 2.0, -0.4),   # overshoot capped at squeeze_max below the limit
            (0.2, 0.2, 1.0, 0.2),    # gripper reaches the target: no extra force
            (0.6, 0.2, 0.0, 0.2),    # boost disabled: passthrough
        ]
        for follower_gripper, leader_gripper, boost, expected in cases:
            with self.subTest(follower=follower_gripper, leader=leader_gripper, boost=boost):
                leader = RecordingRobotAdapter(
                    ArmState(
                        positions=[0.0] * 6,
                        velocities=[0.0] * 6,
                        torques=[0.0] * 6,
                        gravity=[0.0] * 6,
                        gripper_position=leader_gripper,
                        gripper_velocity=0.0,
                        gripper_torque=0.0,
                    )
                )
                follower = RecordingRobotAdapter(
                    ArmState(
                        positions=[0.0] * 6,
                        velocities=[0.0] * 6,
                        torques=[0.0] * 6,
                        gravity=[0.0] * 6,
                        gripper_position=follower_gripper,
                        gripper_velocity=0.0,
                        gripper_torque=0.0,
                    )
                )
                daemon = ControlDaemon(
                    ControlConfig(
                        mock=True,
                        socket_path=self._socket_path(),
                        initial_control_mode="teleop",
                        teleop_gripper_squeeze_boost=boost,
                    )
                )
                daemon._leader = leader
                daemon._follower = follower

                daemon._step(time.monotonic(), 0.001, 0.001)

                self.assertAlmostEqual(follower.gripper_commands[-1][0], expected, places=6)

    def test_teleop_gripper_feedback_scale_controls_leader_gripper_torque(self):
        for scale in (0.5, 0.25):
            with self.subTest(scale=scale):
                leader = RecordingRobotAdapter(
                    ArmState(
                        positions=[0.0] * 6,
                        velocities=[0.0] * 6,
                        torques=[0.0] * 6,
                        gravity=[0.0] * 6,
                        gripper_position=1.0,
                        gripper_velocity=0.0,
                        gripper_torque=0.0,
                    )
                )
                follower = RecordingRobotAdapter(
                    ArmState(
                        positions=[0.0] * 6,
                        velocities=[0.0] * 6,
                        torques=[0.0] * 6,
                        gravity=[0.0] * 6,
                        gripper_position=1.0,
                        gripper_velocity=0.0,
                        gripper_torque=2.0,
                    )
                )
                daemon = ControlDaemon(
                    ControlConfig(
                        mock=True,
                        socket_path=self._socket_path(),
                        initial_control_mode="teleop",
                        teleop_gripper_feedback_scale=scale,
                    )
                )
                daemon._leader = leader
                daemon._follower = follower

                daemon._step(time.monotonic(), 0.001, 0.001)

                self.assertAlmostEqual(leader.gripper_commands[-1][2], -2.0 * scale, places=6)

    def test_apply_hold_neutralizes_leader_even_when_follower_read_fails(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.3] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.6,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = FailingReadRobotAdapter()

        daemon._apply_hold(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][2], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][4], [0.0] * 6)
        self.assertEqual(leader.gripper_commands[-1], (leader.state.gripper_position, 0.0, 0.0, 0.0, 0.0))
        self.assertIn("read failed", daemon._last_error)

    def test_apply_hold_uses_last_valid_follower_pose_when_read_fails(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.3] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.6,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = FailAfterValidRobotAdapter(
            ArmState(
                positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = follower
        daemon._step(time.monotonic(), 0.001, 0.001)
        baseline_sequence = daemon.latest_snapshot()["sequence"]

        follower.fail_reads = True
        daemon._last_error = "primary follower read failure"
        daemon.set_control_mode("hold")
        held = daemon._apply_hold(time.monotonic(), 0.001, 0.001)

        snapshot = daemon.latest_snapshot()
        self.assertTrue(held)
        self.assertGreater(snapshot["sequence"], baseline_sequence)
        self.assertTrue(snapshot["diagnostics"]["holding"])
        self.assertEqual(snapshot["diagnostics"]["control_mode"], "hold")
        self.assertEqual(follower.arm_commands[-1][0], follower.state.positions)
        self.assertEqual(follower.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(follower.gripper_commands[-1][0], follower.state.gripper_position)
        self.assertEqual(follower.gripper_commands[-1][1], 0.0)
        self.assertIn("primary follower read failure", daemon._last_error)
        self.assertIn("read failed", daemon._last_error)

    def test_apply_hold_does_not_confirm_when_cached_pose_command_fails(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.3] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.6,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = FailAfterValidRobotAdapter(
            ArmState(
                positions=[0.1] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = follower
        daemon._step(time.monotonic(), 0.001, 0.001)
        baseline_snapshot = daemon.latest_snapshot()

        follower.fail_reads = True
        follower.fail_commands = True
        daemon._last_error = "primary follower read failure"
        daemon.set_control_mode("hold")
        held = daemon._apply_hold(time.monotonic(), 0.001, 0.001)

        snapshot = daemon.latest_snapshot()
        self.assertFalse(held)
        self.assertEqual(snapshot["sequence"], baseline_snapshot["sequence"])
        self.assertEqual(snapshot["timestamp"], baseline_snapshot["timestamp"])
        self.assertEqual(
            snapshot["diagnostics"]["holding"],
            baseline_snapshot["diagnostics"]["holding"],
        )
        self.assertIn("fail-safe hold command failed", snapshot["diagnostics"]["last_error"])
        self.assertFalse(daemon.status()["holding"])
        self.assertIn("primary follower read failure", daemon._last_error)
        self.assertIn("read failed", daemon._last_error)
        self.assertIn("fail-safe hold command failed", daemon._last_error)

    def test_run_loop_latches_errors_across_control_cycles(self):
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path(), loop_hz=1000.0))
        errors = [RuntimeError("first control failure"), RuntimeError("second control failure")]

        def fail_step(_now, _last_period, _mean_period):
            error = errors.pop(0)
            if not errors:
                daemon._stop_event.set()
            raise error

        daemon._step = fail_step
        daemon._apply_hold = lambda *_args: False

        daemon._run_loop()

        self.assertIn("first control failure", daemon._last_error)
        self.assertIn("second control failure", daemon._last_error)

    def test_apply_hold_records_leader_neutralization_failures(self):
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.1] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = FailingNeutralizeLeaderAdapter()
        daemon._follower = follower
        daemon._last_error = "primary follower failure"
        daemon.set_control_mode("hold")

        held = daemon._apply_hold(time.monotonic(), 0.001, 0.001)

        self.assertTrue(held)
        self.assertIn("primary follower failure", daemon._last_error)
        self.assertIn("leader read failed", daemon._last_error)
        self.assertIn("leader neutralization failed", daemon._last_error)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["holding"])

    def test_apply_hold_neutralizes_leader_after_error(self):
        leader = RecordingRobotAdapter(
            ArmState(
                positions=[0.3] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.6,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        follower = RecordingRobotAdapter(
            ArmState(
                positions=[0.1] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            )
        )
        daemon = ControlDaemon(
            ControlConfig(
                mock=True,
                socket_path=self._socket_path(),
                policy_mirror_leader=True,
            )
        )
        daemon._leader = leader
        daemon._follower = follower
        daemon.set_control_mode("policy")
        daemon.submit_policy_action([0.0] * 7)
        daemon._step(time.monotonic(), 0.001, 0.001)

        daemon._apply_hold(time.monotonic(), 0.001, 0.001)

        self.assertEqual(leader.arm_commands[-1][0], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][1], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][2], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][3], [0.0] * 6)
        self.assertEqual(leader.arm_commands[-1][4], [0.0] * 6)
        self.assertEqual(leader.gripper_commands[-1], (leader.state.gripper_position, 0.0, 0.0, 0.0, 0.0))
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["holding"])

    def test_stop_applies_synchronous_hold_before_stopping_adapters(self):
        events = []
        leader = OrderedRecordingRobotAdapter(
            ArmState(
                positions=[0.3] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.6,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            ),
            "leader",
            events,
        )
        follower = OrderedRecordingRobotAdapter(
            ArmState(
                positions=[0.1] * 6,
                velocities=[0.0] * 6,
                torques=[0.0] * 6,
                gravity=[0.0] * 6,
                gripper_position=0.4,
                gripper_velocity=0.0,
                gripper_torque=0.0,
            ),
            "follower",
            events,
        )
        daemon = ControlDaemon(ControlConfig(mock=True, socket_path=self._socket_path()))
        daemon._leader = leader
        daemon._follower = follower

        daemon.stop()

        self.assertIn("leader:command_arm", events)
        self.assertIn("follower:command_arm", events)
        self.assertLess(events.index("follower:command_arm"), events.index("leader:stop"))
        self.assertLess(events.index("follower:command_arm"), events.index("follower:stop"))
        self.assertEqual(follower.arm_commands[-1][0], follower.state.positions)
        self.assertEqual(follower.arm_commands[-1][1], [0.0] * 6)
        self.assertTrue(daemon.latest_snapshot()["diagnostics"]["holding"])

    def test_control_config_rejects_non_finite_or_non_positive_safety_values(self):
        invalid_kwargs = [
            {"loop_hz": 0.0},
            {"policy_action_timeout_s": 0.0},
            {"policy_action_timeout_s": float("nan")},
            {"policy_action_timeout_s": float("inf")},
            {"max_target_lead_rad": 0.0},
            {"max_target_lead_rad": -0.1},
            {"max_gripper_lead": 0.0},
            {"max_gripper_lead": float("nan")},
            {"policy_mirror_max_step_rad": 0.0},
            {"policy_mirror_max_gripper_step": float("inf")},
            {"policy_mirror_leader_kp": [0.3] * 5},
            {"policy_mirror_leader_kd": [0.03] * 5},
            {"policy_mirror_leader_kp": [float("nan")] * 6},
            {"policy_mirror_gripper_kp": float("inf")},
            {"policy_mirror_gripper_kd": -0.1},
        ]

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ControlConfig(mock=True, socket_path=self._socket_path(), **kwargs)

    def test_client_handles_normal_and_stale_snapshot(self):
        snapshot = {
            "type": "snapshot",
            "ok": True,
            "snapshot": {
                "observation.state": [0.0] * 14,
                "action": [1.0] * 7,
                "timestamp": time.monotonic(),
                "sequence": 5,
                "diagnostics": {},
                "loop_timing": {},
            },
        }
        with ScriptedSocketServer([snapshot]) as server:
            client = ControlClient(server.socket_path)
            self.assertEqual(client.snapshot(max_age_s=1.0)["sequence"], 5)

        stale = dict(snapshot)
        stale["snapshot"] = dict(snapshot["snapshot"])
        stale["snapshot"]["timestamp"] = time.monotonic() - 10.0
        with ScriptedSocketServer([stale]) as server:
            client = ControlClient(server.socket_path)
            with self.assertRaises(StaleSnapshotError):
                client.snapshot(max_age_s=0.01)

    def test_client_handles_malformed_daemon_response_and_disconnect(self):
        with ScriptedSocketServer(["not-json\n"]) as server:
            client = ControlClient(server.socket_path)
            with self.assertRaises(MalformedResponseError):
                client.snapshot()

        with ScriptedSocketServer([b""]) as server:
            client = ControlClient(server.socket_path)
            with self.assertRaises(DaemonDisconnectedError):
                client.snapshot()

    def test_client_rejects_non_finite_or_invalid_snapshot_metadata(self):
        base = {
            "observation.state": [0.0] * 14,
            "action": [1.0] * 7,
            "timestamp": time.monotonic(),
            "sequence": 5,
            "diagnostics": {},
            "loop_timing": {},
        }
        mutations = [
            ("observation.state", 0, float("nan")),
            ("action", 0, float("inf")),
            ("timestamp", None, float("nan")),
            ("sequence", None, True),
            ("sequence", None, 1.5),
            ("sequence", None, -1),
        ]

        for field, index, value in mutations:
            with self.subTest(field=field, value=value):
                snapshot = dict(base)
                if index is None:
                    snapshot[field] = value
                else:
                    snapshot[field] = list(base[field])
                    snapshot[field][index] = value
                response = {"type": "snapshot", "ok": True, "snapshot": snapshot}

                with ScriptedSocketServer([response]) as server:
                    with self.assertRaises(MalformedResponseError):
                        ControlClient(server.socket_path).snapshot()

    def test_mock_daemon_runs_briefly_at_high_rate_and_produces_fresh_snapshot(self):
        daemon = ControlDaemon(
            ControlConfig(mock=True, socket_path=self._socket_path(), loop_hz=1000.0)
        )
        try:
            daemon.start()
            time.sleep(0.03)
            snapshot = daemon.latest_snapshot()
        finally:
            daemon.stop()

        self.assertGreater(snapshot["sequence"], 5)
        self.assertLess(time.monotonic() - snapshot["timestamp"], 0.1)
        self.assertLess(snapshot["loop_timing"]["target_period_s"], 0.002)

    def test_no_camera_dataset_video_or_rerun_logic_appears_in_daemon_core(self):
        import panthera_lerobot_control

        package_dir = Path(inspect.getfile(panthera_lerobot_control)).parent
        core_files = [
            "adapters.py",
            "cli.py",
            "constants.py",
            "controller.py",
            "ipc.py",
        ]
        source = "\n".join((package_dir / name).read_text() for name in core_files)

        forbidden = ("camera", "dataset", "video", "rerun", "panthera_teleop_recorder")
        for token in forbidden:
            self.assertNotIn(token, source.lower())

    def _socket_path(self):
        return str(Path(tempfile.gettempdir()) / f"panthera-test-{os.getpid()}-{time.time_ns()}.sock")

    def _wait_for_snapshot(self, daemon, timeout_s=1.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = daemon.latest_snapshot()
            if snapshot["sequence"] > 0:
                return snapshot
            time.sleep(0.005)
        self.fail("daemon did not produce a snapshot")


class BrakeStopRobot:
    motor_count = 6

    def __init__(self):
        self.calls = []

    def set_brake(self):
        self.calls.append("set_brake")

    def set_stop(self):
        self.calls.append("set_stop")


class StopOnlyRobot:
    motor_count = 6

    def __init__(self):
        self.calls = []

    def set_stop(self):
        self.calls.append("set_stop")


class FakeSdkStateRobot:
    motor_count = 6

    def __init__(self):
        self.positions = [0.0] * 6
        self.velocities = [0.0] * 6
        self.torques = [0.0] * 6
        self.gravity = [0.0] * 6
        self.gripper_position = 0.0
        self.gripper_velocity = 0.0
        self.gripper_torque = 0.0
        self.friction_compensation = [0.0] * 6

    def get_current_pos(self):
        return self.positions

    def get_current_vel(self):
        return self.velocities

    def get_current_torque(self):
        return self.torques

    def get_Gravity(self):
        return self.gravity

    def get_current_pos_gripper(self):
        return self.gripper_position

    def get_current_vel_gripper(self):
        return self.gripper_velocity

    def get_current_torque_gripper(self):
        return self.gripper_torque

    def get_friction_compensation(self, velocities, fc, fv, velocity_threshold):
        return self.friction_compensation


class FailingNeutralizeLeaderAdapter:
    def read_state(self):
        raise RuntimeError("leader read failed")

    def command_arm(self, positions, velocities, torques, kp, kd):
        raise RuntimeError("leader neutralization failed")


class LifecycleRobotAdapter:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class RecordingRobotAdapter:
    def __init__(self, state):
        self.state = state
        self.arm_commands = []
        self.gripper_commands = []

    def read_state(self):
        return self.state

    def friction(self, velocities, fc, fv, velocity_threshold):
        return [0.0] * 6

    def command_arm(self, positions, velocities, torques, kp, kd):
        self.arm_commands.append((list(positions), list(velocities), list(torques), list(kp), list(kd)))

    def command_gripper(self, position, velocity, torque, kp, kd):
        self.gripper_commands.append((position, velocity, torque, kp, kd))

    def stop(self):
        pass


class OrderedRecordingRobotAdapter(RecordingRobotAdapter):
    def __init__(self, state, name, events):
        super().__init__(state)
        self.name = name
        self.events = events

    def command_arm(self, positions, velocities, torques, kp, kd):
        self.events.append(f"{self.name}:command_arm")
        super().command_arm(positions, velocities, torques, kp, kd)

    def command_gripper(self, position, velocity, torque, kp, kd):
        self.events.append(f"{self.name}:command_gripper")
        super().command_gripper(position, velocity, torque, kp, kd)

    def stop(self):
        self.events.append(f"{self.name}:stop")


class FailAfterValidRobotAdapter(RecordingRobotAdapter):
    def __init__(self, state):
        super().__init__(state)
        self.fail_reads = False
        self.fail_commands = False

    def read_state(self):
        if self.fail_reads:
            raise RuntimeError("read failed")
        return super().read_state()

    def command_arm(self, positions, velocities, torques, kp, kd):
        if self.fail_commands:
            raise RuntimeError("fail-safe hold command failed")
        super().command_arm(positions, velocities, torques, kp, kd)

    def command_gripper(self, position, velocity, torque, kp, kd):
        if self.fail_commands:
            raise RuntimeError("fail-safe hold command failed")
        super().command_gripper(position, velocity, torque, kp, kd)


class FailingReadRobotAdapter:
    def read_state(self):
        raise RuntimeError("read failed")

    def command_arm(self, positions, velocities, torques, kp, kd):
        raise RuntimeError("command_arm should not be reached")

    def command_gripper(self, position, velocity, torque, kp, kd):
        raise RuntimeError("command_gripper should not be reached")

    def stop(self):
        pass


class DummyDaemon:
    def __init__(self):
        self.control_mode = "teleop"
        self.policy_actions = []

    def status(self):
        return {
            "running": True,
            "control_mode": self.control_mode,
            "policy_sequence": len(self.policy_actions),
        }

    def latest_snapshot(self):
        return {
            "observation.state": [0.0] * 14,
            "action": [0.0] * 7,
            "timestamp": time.monotonic(),
            "sequence": 1,
            "diagnostics": {},
            "loop_timing": {},
        }

    def hold(self):
        self.control_mode = "hold"

    def set_control_mode(self, mode):
        self.control_mode = mode

    def submit_policy_action(self, action):
        self.policy_actions.append(list(action))

    def stop(self):
        pass


class ScriptedSocketServer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.socket_path = str(Path(tempfile.gettempdir()) / f"panthera-scripted-{os.getpid()}-{time.time_ns()}.sock")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._thread = None

    def __enter__(self):
        self._sock.bind(self.socket_path)
        self._sock.listen(1)

        import threading

        self._thread = threading.Thread(target=self._serve_once, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sock.close()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _serve_once(self):
        conn, _ = self._sock.accept()
        with conn:
            conn.recv(4096)
            response = self.responses.pop(0)
            if isinstance(response, bytes):
                if response:
                    conn.sendall(response)
                return
            if isinstance(response, str):
                conn.sendall(response.encode("utf-8"))
                return
            import json

            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
