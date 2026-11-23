import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import finearm.data.collector.teleop as collect_cli


class PantheraLeRobotCollectTests(unittest.TestCase):
    def test_camera_specs_round_trip_through_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            specs = collect_cli.parse_camera_specs(["front=/dev/video0", "wrist=/dev/video2"])

            collect_cli.save_camera_selection(specs, path=path)
            loaded = collect_cli.load_camera_selection(path=path)

        self.assertEqual(specs, loaded)

    def test_camera_cache_rejects_any_malformed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cameras": [
                            {"name": "front", "source": "/dev/video0"},
                            {"name": "wrist"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(collect_cli.CollectError, "camera selection cache"):
                collect_cli.load_camera_selection(path=path)

    def test_camera_config_is_structured_json_and_preserves_numeric_sources(self):
        config = collect_cli.build_lerobot_camera_config(
            (
                collect_cli.CameraSpec(name="front", source="0"),
                collect_cli.CameraSpec(name="wrist", source="/dev/video2"),
            ),
            width=640,
            height=480,
            fps=30,
        )

        self.assertEqual(
            json.loads(config),
            {
                "front": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                "wrist": {
                    "type": "opencv",
                    "index_or_path": "/dev/video2",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            },
        )

    def test_camera_config_includes_fourcc_when_provided(self):
        config = collect_cli.build_lerobot_camera_config(
            (collect_cli.CameraSpec(name="front", source="0"),),
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            warmup_s=5,
        )

        self.assertEqual(
            json.loads(config),
            {
                "front": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "fourcc": "MJPG",
                    "warmup_s": 5,
                },
            },
        )

    def test_camera_specs_reject_invalid_names_and_duplicate_physical_sources(self):
        invalid_values = [
            ["front-camera=/dev/video0"],
            ["1front=/dev/video0"],
            ["front=0", "wrist=/dev/video0"],
            ["front=0", "wrist=+0"],
            ["front=0", "wrist=-0"],
            ["front=-1"],
            ["front=/dev/video0", "wrist=/dev/video0"],
        ]

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(collect_cli.CollectError):
                    collect_cli.parse_camera_specs(values)

    def test_collection_daemon_explicitly_starts_in_teleop_mode(self):
        args = collect_cli.build_parser().parse_args(["--mock", "--no-cameras"])

        command = collect_cli.build_daemon_command(args)

        self.assertIn("--initial-control-mode", command)
        index = command.index("--initial-control-mode")
        self.assertEqual(command[index + 1], "teleop")

    def test_build_lerobot_record_command_includes_panthera_plugins_and_cameras(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--auto-start",
                "--camera",
                "front=/dev/video0",
                "--camera",
                "wrist=/dev/video2",
                "--repo-id",
                "local/demo",
                "--task",
                "pick cube",
                "--episodes",
                "2",
                "--episode-seconds",
                "10",
                "--reset-seconds",
                "5",
                "--fps",
                "30",
            ]
        )
        specs = collect_cli.parse_camera_specs(args.camera)

        fixed_now = datetime(2026, 7, 10, 14, 32, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ), mock.patch.object(
            collect_cli,
            "_current_local_datetime",
            return_value=fixed_now,
        ):
            command = collect_cli.build_lerobot_record_command(args, specs)
            expected_root_parent = Path(tmp) / "lerobot" / "local" / "demo"
        joined = " ".join(command)

        self.assertEqual(command[0], "lerobot-record")
        self.assertIn("--robot.type=panthera_ht_follower", command)
        self.assertIn("--teleop.type=panthera_ht_leader", command)
        self.assertIn("--dataset.repo_id=local/demo", command)
        self.assertIn("--dataset.num_episodes=2", command)
        self.assertIn("--dataset.episode_time_s=10.0", command)
        self.assertIn("--dataset.reset_time_s=5.0", command)
        self.assertIn("--dataset.fps=30", command)
        self.assertNotIn("--fps=30", command)
        self.assertIn("--resume=false", command)
        self.assertNotIn("--resume=true", command)
        dataset_root_arg = next(arg for arg in command if arg.startswith("--dataset.root="))
        dataset_root = Path(dataset_root_arg.split("=", 1)[1])
        self.assertEqual(dataset_root.parent, expected_root_parent)
        self.assertRegex(dataset_root.name, r"^20260710_143205_000000_p\d+_c\d{6}$")
        self.assertIn("--dataset.single_task=pick cube", command)
        self.assertIn("front", joined)
        self.assertIn("/dev/video0", joined)
        self.assertIn("wrist", joined)
        self.assertIn("/dev/video2", joined)

    def test_build_record_command_propagates_ipc_timeout_and_freshness_to_both_plugins(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--no-cameras",
                "--timeout-s",
                "0.7",
                "--stale-timeout-s",
                "0.15",
            ]
        )

        command = collect_cli.build_lerobot_record_command(args, ())

        self.assertIn("--robot.timeout_s=0.7", command)
        self.assertIn("--robot.stale_timeout_s=0.15", command)
        self.assertIn("--teleop.timeout_s=0.7", command)
        self.assertIn("--teleop.stale_timeout_s=0.15", command)

    def test_build_record_command_rejects_lerobot_args_that_override_managed_contracts(self):
        protected_args = (
            "--robot.socket_path=/tmp/override.sock",
            "--teleop.type=other_teleoperator",
            "--dataset.root=/tmp/override-dataset",
            "--resume=true",
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ):
            for raw_arg in protected_args:
                with self.subTest(raw_arg=raw_arg):
                    args = collect_cli.build_parser().parse_args(
                        ["--mock", "--no-cameras", f"--lerobot-arg={raw_arg}"]
                    )
                    with self.assertRaisesRegex(collect_cli.CollectError, "--lerobot-arg"):
                        collect_cli.build_lerobot_record_command(args, ())

    def test_build_record_command_keeps_unmanaged_lerobot_args(self):
        args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--lerobot-arg=--display_data=true"]
        )

        command = collect_cli.build_lerobot_record_command(args, ())

        self.assertIn("--display_data=true", command)

    def test_missing_record_executable_fails_preflight_before_serial_or_daemon(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            collect_cli,
            "ensure_live_serial_permissions",
            side_effect=lambda *args, **kwargs: events.append("serial"),
        ):
            rc = collect_cli.collect(
                [
                    "--live",
                    "--print-command",
                    "--no-cameras",
                    "--record-bin",
                    str(Path(tmp) / "missing-lerobot-record"),
                ],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                popen_factory=lambda *_args, **_kwargs: events.append("popen") or self.fail("daemon should not start"),
            )

        self.assertEqual(rc, 2)
        self.assertEqual(events, [])

    def test_runtime_preflight_imports_lerobot_package(self):
        args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--record-bin", sys.executable]
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ), mock.patch.object(
            collect_cli.metadata,
            "version",
            return_value=collect_cli.SUPPORTED_LEROBOT_VERSION,
        ), mock.patch(
            "importlib.import_module"
        ) as import_module:
            collect_cli.run_collection_preflight(args)

        import_module.assert_called_once_with("lerobot")

    def test_print_command_rejects_interactive_camera_selection(self):
        args = collect_cli.build_parser().parse_args(
            ["--mock", "--print-command", "--select-cameras"]
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ):
            with self.assertRaisesRegex(collect_cli.CollectError, "--print-command"):
                collect_cli.run_collection_preflight(args, require_runtime=False)

    def test_build_lerobot_record_command_disables_unclear_voice_prompts_by_default(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
            ]
        )

        command = collect_cli.build_lerobot_record_command(args, ())

        self.assertIn("--play_sounds=false", command)

    def test_build_lerobot_record_command_can_keep_voice_prompts(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--voice-prompts",
            ]
        )

        command = collect_cli.build_lerobot_record_command(args, ())

        self.assertIn("--play_sounds=true", command)

    def test_record_status_tracker_updates_phase_and_beeps_from_lerobot_logs(self):
        ui = FakeRecordStatusUI()
        beeps = []
        tracker = collect_cli._RecordStatusTracker(
            status_ui=ui,
            beep_func=lambda count: beeps.append(count),
        )

        tracker.process_line("INFO 2026-07-03 ls/utils.py:142 Recording episode 3\n")
        tracker.process_line("INFO 2026-07-03 ls/utils.py:142 Reset the environment\n")
        tracker.process_line("INFO 2026-07-03 ls/utils.py:142 Stop recording\n")

        self.assertEqual(
            ui.updates,
            [
                ("RECORDING", "Episode 3"),
                ("RESET", "Prepare the scene for the next episode"),
                ("STOPPING", "Finalizing dataset and stopping recording"),
            ],
        )
        self.assertEqual(beeps, [2, 1, 1])

    def test_record_runner_interrupts_isolated_process_group_then_escalates_once(self):
        process = InterruptingRecordProcess()
        popen_calls = []
        group_signals = []

        def popen_factory(command, **kwargs):
            popen_calls.append((command, kwargs))
            return process

        def interrupt_sleep(_seconds):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            collect_cli._default_record_runner(
                ["lerobot-record", "--robot.type=panthera_ht_follower"],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                sleep=interrupt_sleep,
                graceful_timeout_s=30.0,
                terminate_timeout_s=5.0,
                popen_factory=popen_factory,
                killpg=lambda pid, sig: group_signals.append((pid, sig)),
            )

        self.assertIs(popen_calls[0][1]["start_new_session"], True)
        self.assertEqual(
            group_signals,
            [
                (process.pid, signal.SIGINT),
                (process.pid, signal.SIGTERM),
                (process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(process.wait_calls, [30.0, 5.0, None])

    def test_record_runner_owns_process_immediately_after_popen_returns(self):
        process = InterruptingRecordProcess()
        group_signals = []

        with mock.patch.object(collect_cli.queue, "Queue", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                collect_cli._default_record_runner(
                    ["lerobot-record", "--robot.type=panthera_ht_follower"],
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                    graceful_timeout_s=30.0,
                    terminate_timeout_s=5.0,
                    popen_factory=lambda *_args, **_kwargs: process,
                    killpg=lambda pid, sig: group_signals.append((pid, sig)),
                )

        self.assertEqual(
            group_signals,
            [
                (process.pid, signal.SIGINT),
                (process.pid, signal.SIGTERM),
                (process.pid, signal.SIGKILL),
            ],
        )

    def test_record_shutdown_timeouts_are_finite_positive_parser_options(self):
        args = collect_cli.build_parser().parse_args(["--mock", "--no-cameras"])

        self.assertEqual(args.record_graceful_timeout_s, 30.0)
        self.assertEqual(args.record_terminate_timeout_s, 5.0)
        for flag, value in (
            ("--record-graceful-timeout-s", "nan"),
            ("--record-graceful-timeout-s", "0"),
            ("--record-terminate-timeout-s", "inf"),
            ("--record-terminate-timeout-s", "-1"),
        ):
            with self.subTest(flag=flag, value=value):
                with self.assertRaises(SystemExit):
                    collect_cli.build_parser().parse_args(["--mock", "--no-cameras", flag, value])

    def test_collect_main_translates_term_and_hup_and_restores_handlers(self):
        termination_signals = (signal.SIGTERM, signal.SIGHUP)

        for triggered_signal in termination_signals:
            with self.subTest(triggered_signal=triggered_signal):
                previous = {signum: object() for signum in termination_signals}
                installed = {}
                signal_calls = []

                def fake_signal(signum, handler):
                    signal_calls.append((signum, handler))
                    if handler is not previous[signum]:
                        installed[signum] = handler

                def fake_collect(_argv):
                    handler = installed.get(triggered_signal)
                    self.assertIsNotNone(handler)
                    try:
                        handler(triggered_signal, None)
                    except KeyboardInterrupt:
                        return 130
                    self.fail("termination signal handler did not interrupt collection")

                with mock.patch.object(collect_cli.signal, "getsignal", side_effect=previous.__getitem__), mock.patch.object(
                    collect_cli.signal, "signal", side_effect=fake_signal
                ), mock.patch.object(collect_cli, "collect", side_effect=fake_collect):
                    rc = collect_cli.main([])

                self.assertEqual(rc, 130)
                for signum in termination_signals:
                    self.assertIn((signum, previous[signum]), signal_calls)

    def test_collect_termination_handlers_interrupt_only_once_during_cleanup(self):
        previous = {signum: object() for signum in (signal.SIGTERM, signal.SIGHUP)}
        installed = {}

        def fake_signal(signum, handler):
            if handler is not previous[signum]:
                installed[signum] = handler

        with mock.patch.object(
            collect_cli.signal,
            "getsignal",
            side_effect=previous.__getitem__,
        ), mock.patch.object(
            collect_cli.signal,
            "signal",
            side_effect=fake_signal,
        ):
            with collect_cli._translated_termination_signals():
                with self.assertRaises(KeyboardInterrupt):
                    installed[signal.SIGTERM](signal.SIGTERM, None)
                second_interrupted = False
                try:
                    installed[signal.SIGHUP](signal.SIGHUP, None)
                except KeyboardInterrupt:
                    second_interrupted = True
                self.assertFalse(second_interrupted)

    def test_collect_starts_record_status_ui_for_recording_lifecycle(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        popen_calls = []
        record_calls = []
        ui = FakeRecordStatusUI()

        def client_factory(socket_path, timeout_s=1.0):
            return FakeClient(socket_path, mode="mock", active_after=len(popen_calls) > 0)

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--no-release-prompt",
            ],
            stdout=stdout,
            stderr=stderr,
            popen_factory=lambda command, **kwargs: popen_calls.append((command, kwargs)) or FakeProcess(),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=client_factory,
            status_ui_factory=lambda stderr: ui,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertTrue(ui.started)
        self.assertIn(("STARTING", "Launching LeRobot record"), ui.updates)
        self.assertIn(("DONE", "LeRobot record exited with code 0"), ui.updates)
        self.assertTrue(ui.shutdown_called)

    def test_collect_daemon_command_can_enable_policy_leader_mirror(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--no-cameras",
                "--policy-mirror-leader",
                "--policy-mirror-max-step-rad",
                "0.03",
                "--policy-mirror-max-gripper-step",
                "0.08",
            ]
        )

        command = collect_cli.build_daemon_command(args)

        self.assertIn("--policy-mirror-leader", command)
        self.assertIn("--policy-mirror-max-step-rad", command)
        self.assertIn("0.03", command)
        self.assertIn("--policy-mirror-max-gripper-step", command)
        self.assertIn("0.08", command)

    def test_collect_cli_rejects_non_finite_or_non_positive_policy_mirror_args(self):
        invalid_args = [
            ("--policy-mirror-max-step-rad", "nan"),
            ("--policy-mirror-max-step-rad", "0"),
            ("--policy-mirror-max-gripper-step", "inf"),
            ("--policy-mirror-max-gripper-step", "-0.01"),
        ]

        for flag, value in invalid_args:
            with self.subTest(flag=flag, value=value):
                with self.assertRaises(SystemExit):
                    collect_cli.build_parser().parse_args(["--mock", "--no-cameras", flag, value])

    def test_implicit_fresh_dataset_root_preserves_namespace_and_adds_timestamp(self):
        args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--repo-id", "team/demo"]
        )
        now = datetime(2026, 7, 10, 14, 32, 5, 123456, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ):
            root = collect_cli.resolve_collection_dataset_root(args, now=now)

        self.assertEqual(root.parent, Path(tmp) / "lerobot" / "team" / "demo")
        self.assertRegex(
            root.name,
            r"^20260710_143205_123456_p\d+_c\d{6}$",
        )

    def test_implicit_fresh_dataset_roots_are_unique_for_the_same_instant(self):
        first_args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--repo-id", "local/demo"]
        )
        second_args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--repo-id", "local/demo"]
        )
        now = datetime(2026, 7, 10, 14, 32, 5, 123456, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ):
            first = collect_cli.resolve_collection_dataset_root(first_args, now=now)
            second = collect_cli.resolve_collection_dataset_root(second_args, now=now)

        self.assertEqual(first.parent, second.parent)
        self.assertNotEqual(first, second)

    def test_implicit_fresh_dataset_root_appends_suffix_on_timestamp_collision(self):
        args = collect_cli.build_parser().parse_args(
            ["--mock", "--no-cameras", "--repo-id", "local/demo"]
        )
        now = datetime(2026, 7, 10, 14, 32, 5, 123456, tzinfo=timezone.utc)
        checks = []

        def collide_twice(path):
            checks.append(Path(path))
            return len(checks) <= 2

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"PANTHERA_DATA_ROOT": tmp},
        ):
            root = collect_cli.resolve_collection_dataset_root(
                args,
                now=now,
                path_exists=collide_twice,
            )

        self.assertEqual(len(checks), 3)
        self.assertEqual(root, Path(f"{checks[0]}_002"))

    def test_collection_dataset_root_rejects_unsafe_repo_ids(self):
        for repo_id in ("", "local", "local/../demo", "local//demo", "/demo", "local/demo/extra"):
            with self.subTest(repo_id=repo_id):
                args = collect_cli.build_parser().parse_args(
                    ["--mock", "--no-cameras", "--repo-id", repo_id]
                )
                with self.assertRaises(collect_cli.CollectError):
                    collect_cli.resolve_collection_dataset_root(args)

    def test_explicit_existing_root_requires_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = collect_cli.build_parser().parse_args(
                [
                    "--mock",
                    "--no-cameras",
                    "--repo-id",
                    "local/demo",
                    "--dataset-root",
                    tmp,
                ]
            )

            with self.assertRaisesRegex(collect_cli.CollectError, "--resume"):
                collect_cli.resolve_collection_dataset_root(args)

    def test_collect_resume_requires_explicit_dataset_root(self):
        args = collect_cli.build_parser().parse_args(
            [
                "--mock",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--resume",
            ]
        )

        with self.assertRaises(collect_cli.CollectError) as ctx:
            collect_cli.build_lerobot_record_command(args, ())

        self.assertIn("--resume requires --dataset-root", str(ctx.exception))

    def test_collect_resume_passes_dataset_root_to_lerobot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_valid_resume_metadata(root, task="pick cube", fps=30)
            args = collect_cli.build_parser().parse_args(
                [
                    "--mock",
                    "--no-cameras",
                    "--repo-id",
                    "local/demo",
                    "--dataset-root",
                    str(root),
                    "--task",
                    "pick cube",
                    "--fps",
                    "30",
                    "--resume",
                ]
            )

            command = collect_cli.build_lerobot_record_command(args, ())

        self.assertIn(f"--dataset.root={root}", command)
        self.assertIn("--resume=true", command)

    def test_collect_resume_rejects_task_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_valid_resume_metadata(root, task="pick cube", fps=30)
            args = collect_cli.build_parser().parse_args(
                [
                    "--mock",
                    "--no-cameras",
                    "--repo-id",
                    "local/demo",
                    "--dataset-root",
                    str(root),
                    "--task",
                    "place cube",
                    "--resume",
                ]
            )

            with self.assertRaisesRegex(collect_cli.CollectError, "task"):
                collect_cli.build_lerobot_record_command(args, ())

    def test_resume_requires_writable_root_meta_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_valid_resume_metadata(root, task="pick cube", fps=30)
            args = collect_cli.build_parser().parse_args(
                [
                    "--mock",
                    "--no-cameras",
                    "--repo-id",
                    "local/demo",
                    "--dataset-root",
                    str(root),
                    "--task",
                    "pick cube",
                    "--resume",
                ]
            )
            protected_paths = (
                root,
                root / "meta",
                root / "meta" / "info.json",
                root / "meta" / "tasks.parquet",
            )

            for protected_path in protected_paths:
                with self.subTest(path=protected_path), mock.patch.object(
                    collect_cli.os,
                    "access",
                    side_effect=lambda path, _mode, denied=protected_path: Path(path) != denied,
                ):
                    with self.assertRaisesRegex(collect_cli.CollectError, "writable"):
                        collect_cli.resolve_collection_dataset_root(args)

    def test_resume_rejects_non_integer_info_schema_values(self):
        cases = (
            (("total_tasks",), 1.0),
            (("features", "observation.state", "shape"), [14.0]),
            (("features", "action", "shape"), [7.0]),
        )

        for key_path, invalid_value in cases:
            with self.subTest(key_path=key_path), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "dataset"
                _write_valid_resume_metadata(root, task="pick cube", fps=30)
                info_path = root / "meta" / "info.json"
                info = json.loads(info_path.read_text(encoding="utf-8"))
                target = info
                for key in key_path[:-1]:
                    target = target[key]
                target[key_path[-1]] = invalid_value
                info_path.write_text(json.dumps(info), encoding="utf-8")
                args = collect_cli.build_parser().parse_args(
                    [
                        "--mock",
                        "--no-cameras",
                        "--repo-id",
                        "local/demo",
                        "--dataset-root",
                        str(root),
                        "--task",
                        "pick cube",
                        "--resume",
                    ]
                )

                with self.assertRaises(collect_cli.CollectError):
                    collect_cli.build_lerobot_record_command(args, ())

    def test_resume_rejects_non_integer_task_index_arrow_types(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        for invalid_index in (False, 0.0):
            with self.subTest(task_index=invalid_index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "dataset"
                _write_valid_resume_metadata(root, task="pick cube", fps=30)
                pq.write_table(
                    pa.table({"task_index": [invalid_index], "task": ["pick cube"]}),
                    root / "meta" / "tasks.parquet",
                )
                args = collect_cli.build_parser().parse_args(
                    [
                        "--mock",
                        "--no-cameras",
                        "--repo-id",
                        "local/demo",
                        "--dataset-root",
                        str(root),
                        "--task",
                        "pick cube",
                        "--resume",
                    ]
                )

                with self.assertRaisesRegex(collect_cli.CollectError, "task_index"):
                    collect_cli.build_lerobot_record_command(args, ())

    def test_select_cameras_requires_gui(self):
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            rc = collect_cli.collect(
                [
                    "--mock",
                    "--select-cameras",
                    "--no-gui",
                    "--repo-id",
                    "local/demo",
                    "--socket",
                    str(Path(tmp) / "panthera-test.sock"),
                ],
                stderr=stderr,
            )

        self.assertEqual(rc, 2)
        self.assertIn("--select-cameras requires the GUI", stderr.getvalue())

    def test_camera_preview_worker_timeout_does_not_block_caller(self):
        process = FakePreviewProcess()
        started = time.monotonic()

        with self.assertRaises(collect_cli.CollectError) as ctx:
            collect_cli._read_camera_preview_images_with_timeout(
                [collect_cli.CameraSpec(name="front", source="/dev/video4")],
                image_width=640,
                image_height=480,
                fps=30,
                timeout_s=0.01,
                process_factory=lambda *_args, **_kwargs: process,
                pipe_factory=lambda: (FakePreviewConnection(), FakePreviewConnection()),
            )

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIn("Timed out", str(ctx.exception))
        self.assertTrue(process.terminated)

    def test_camera_preview_worker_is_reaped_even_after_success_payload(self):
        process = FakePreviewProcess()

        images = collect_cli._read_camera_preview_images_with_timeout(
            [collect_cli.CameraSpec(name="front", source="/dev/video4")],
            image_width=640,
            image_height=480,
            fps=30,
            timeout_s=0.1,
            process_factory=lambda *_args, **_kwargs: process,
            pipe_factory=lambda: (
                SuccessfulPreviewConnection({"front": "frame"}),
                ClosingPreviewConnection(),
            ),
        )

        self.assertEqual(images, {"front": "frame"})
        self.assertTrue(process.terminated or process.killed)

    def test_camera_preview_worker_is_reaped_when_start_is_interrupted(self):
        class InterruptingStartProcess(FakePreviewProcess):
            def start(self):
                raise KeyboardInterrupt

        process = InterruptingStartProcess()

        with self.assertRaises(KeyboardInterrupt):
            collect_cli._read_camera_preview_images_with_timeout(
                [collect_cli.CameraSpec(name="front", source="/dev/video4")],
                image_width=640,
                image_height=480,
                fps=30,
                timeout_s=0.1,
                process_factory=lambda *_args, **_kwargs: process,
                pipe_factory=lambda: (FakePreviewConnection(), FakePreviewConnection()),
            )

        self.assertTrue(process.terminated or process.killed)

    def test_print_command_does_not_start_daemon_or_record_process(self):
        stdout = io.StringIO()
        popen_calls = []
        record_calls = []

        rc = collect_cli.collect(
            [
                "--mock",
                "--print-command",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--task",
                "pick cube",
            ],
            stdout=stdout,
            popen_factory=lambda *args, **kwargs: popen_calls.append((args, kwargs)),
            record_runner=lambda *args, **kwargs: record_calls.append((args, kwargs)),
        )

        self.assertEqual(rc, 0)
        self.assertIn("lerobot-record", stdout.getvalue())
        self.assertIn("--robot.type=panthera_ht_follower", stdout.getvalue())
        self.assertEqual(popen_calls, [])
        self.assertEqual(record_calls, [])

    def test_live_collect_resolves_cameras_before_requesting_serial_permissions(self):
        events = []
        original_resolve = collect_cli.resolve_camera_specs
        original_ensure = collect_cli.ensure_panthera_serial_access

        def resolve_camera_specs(_args):
            events.append("cameras")
            raise collect_cli.CollectError("stop after camera resolution")

        try:
            collect_cli.resolve_camera_specs = resolve_camera_specs
            collect_cli.ensure_panthera_serial_access = lambda stderr=None: events.append("serial")

            rc = collect_cli.collect(
                [
                    "--live",
                    "--no-cameras",
                    "--repo-id",
                    "local/demo",
                ],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                popen_factory=lambda *_args, **_kwargs: self.fail("daemon should not start"),
                record_runner=lambda *_args, **_kwargs: self.fail("record should not start"),
                client_factory=lambda socket_path, timeout_s=1.0: FakeClient(
                    socket_path,
                    mode="live",
                    active_after=False,
                ),
            )
        finally:
            collect_cli.resolve_camera_specs = original_resolve
            collect_cli.ensure_panthera_serial_access = original_ensure

        self.assertEqual(rc, 2)
        self.assertEqual(events, ["cameras"])

    def test_camera_read_preflight_runs_before_daemon_probe_or_serial_permissions(self):
        events = []

        def fail_camera_read(*args, **kwargs):
            events.append("camera_read")
            raise collect_cli.CollectError("camera read failed")

        with mock.patch.object(
            collect_cli,
            "_read_camera_preview_images_with_timeout",
            side_effect=fail_camera_read,
        ), mock.patch.object(
            collect_cli,
            "ensure_live_serial_permissions",
            side_effect=lambda *args, **kwargs: events.append("serial"),
        ):
            rc = collect_cli.collect(
                [
                    "--live",
                    "--auto-start",
                    "--camera",
                    "front=/dev/video0",
                    "--repo-id",
                    "local/demo",
                ],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                record_runner=lambda *_args, **_kwargs: 0,
                client_factory=lambda *_args, **_kwargs: events.append("client")
                or (_ for _ in ()).throw(collect_cli.CollectError("daemon probe reached")),
                popen_factory=lambda *_args, **_kwargs: events.append("popen") or FakeProcess(),
            )

        self.assertEqual(rc, 2)
        self.assertEqual(events, ["camera_read"])

    def test_collect_starts_mock_daemon_runs_record_then_holds_and_stops(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        popen_calls = []
        record_calls = []
        clients = []

        def client_factory(socket_path, timeout_s=1.0):
            client = FakeClient(socket_path, mode="mock", active_after=len(popen_calls) > 0)
            clients.append(client)
            return client

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--task",
                "pick cube",
                "--hz",
                "500",
                "--no-release-prompt",
            ],
            stdout=stdout,
            stderr=stderr,
            popen_factory=lambda command, **kwargs: popen_calls.append((command, kwargs)) or FakeProcess(),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(popen_calls), 1)
        self.assertIn("--mock", popen_calls[0][0])
        self.assertIn("--loop-hz", popen_calls[0][0])
        self.assertIn("500", popen_calls[0][0])
        self.assertIn("stdout", popen_calls[0][1])
        self.assertIn("stderr", popen_calls[0][1])
        self.assertIs(popen_calls[0][1].get("start_new_session"), True)
        self.assertEqual(len(record_calls), 1)
        self.assertIn("--dataset.repo_id=local/demo", record_calls[0][0])
        self.assertTrue(any(client.hold_called for client in clients))
        self.assertTrue(any(client.stop_called for client in clients))

    def test_hold_and_release_waits_for_new_confirmed_hold_snapshot_before_stop(self):
        args = collect_cli.build_parser().parse_args(["--mock", "--no-cameras", "--no-release-prompt"])
        client = DelayedSnapshotHoldingClient()

        stopped = collect_cli._hold_and_release(
            client,
            own_daemon=True,
            args=args,
            stdin=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertTrue(stopped)
        self.assertIn("snapshot:12:holding", client.events)
        self.assertLess(client.events.index("snapshot:12:holding"), client.events.index("stop"))

    def test_hold_and_release_does_not_stop_without_new_confirmed_hold_snapshot(self):
        args = collect_cli.build_parser().parse_args(["--mock", "--no-cameras", "--no-release-prompt"])
        args.stale_timeout_s = 0.001
        client = NeverConfirmedHoldClient()

        stopped = collect_cli._hold_and_release(
            client,
            own_daemon=True,
            args=args,
            stdin=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertFalse(stopped)
        self.assertFalse(client.stop_called)

    def test_owned_daemon_unconfirmed_hold_changes_success_to_safety_error_without_killing(self):
        process = RunningProcess()
        popen_calls = []
        clients = []
        stderr = io.StringIO()

        def client_factory(socket_path, timeout_s=1.0):
            client = UnconfirmedHoldClient(
                socket_path,
                mode="mock",
                active_after=bool(popen_calls),
            )
            clients.append(client)
            return client

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/unconfirmed-hold",
                "--stale-timeout-s",
                "0.001",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
            popen_factory=lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
            record_runner=lambda *_args, **_kwargs: 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 2)
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)
        self.assertFalse(any(client.stop_called for client in clients))
        self.assertIn("safe hold", stderr.getvalue())

    def test_confirmed_hold_rejects_non_integer_baseline_sequence(self):
        invalid_sequences = [None, True, 10.5, "10", -1]

        for sequence in invalid_sequences:
            with self.subTest(sequence=sequence):
                client = DelayedSnapshotHoldingClient()

                confirmed = collect_cli._wait_for_confirmed_hold(
                    client,
                    hold_status={"holding": True, "sequence": sequence},
                    timeout_s=0.001,
                )

                self.assertFalse(confirmed)

    def test_collect_refuses_daemon_mode_mismatch_before_recording(self):
        record_calls = []
        stderr = io.StringIO()

        rc = collect_cli.collect(
            ["--mock", "--auto-start", "--no-cameras", "--repo-id", "local/demo"],
            stdout=io.StringIO(),
            stderr=stderr,
            popen_factory=lambda *_args, **_kwargs: self.fail("mode mismatch should not start a new daemon"),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=lambda socket_path, timeout_s=1.0: FakeClient(socket_path, mode="live", active_after=True),
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 2)
        self.assertEqual(record_calls, [])
        self.assertIn("expected 'mock'", stderr.getvalue())

    def test_collect_refuses_external_daemon_that_is_not_in_teleop_control_mode(self):
        for control_mode in ("hold",):
            with self.subTest(control_mode=control_mode):
                record_calls = []
                stderr = io.StringIO()

                rc = collect_cli.collect(
                    [
                        "--mock",
                        "--auto-start",
                        "--no-cameras",
                        "--repo-id",
                        "local/demo",
                        "--no-release-prompt",
                    ],
                    stdout=io.StringIO(),
                    stderr=stderr,
                    popen_factory=lambda *_args, **_kwargs: self.fail("external daemon should not start a process"),
                    record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
                    client_factory=lambda socket_path, timeout_s=1.0, mode=control_mode: FakeClient(
                        socket_path,
                        mode="mock",
                        active_after=True,
                        control_mode=mode,
                    ),
                    sleep=lambda _seconds: None,
                )

                self.assertEqual(rc, 2)
                self.assertEqual(record_calls, [])
                self.assertIn("teleop", stderr.getvalue())

    def test_collect_accepts_external_daemon_in_policy_control_mode(self):
        record_calls = []
        clients = []

        def client_factory(socket_path, timeout_s=1.0):
            client = FakeClient(socket_path, mode="mock", active_after=True, control_mode="policy")
            clients.append(client)
            return client

        stderr = io.StringIO()
        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
            popen_factory=lambda *_args, **_kwargs: self.fail("external daemon should not start a new process"),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(record_calls), 1)
        self.assertIn("Using existing Panthera daemon", stderr.getvalue())
        self.assertTrue(any(client.hold_called for client in clients))

    def test_collect_accepts_external_daemon_in_freedrive_control_mode(self):
        record_calls = []
        clients = []

        def client_factory(socket_path, timeout_s=1.0):
            client = FakeClient(socket_path, mode="mock", active_after=True, control_mode="freedrive")
            clients.append(client)
            return client

        stderr = io.StringIO()
        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
            popen_factory=lambda *_args, **_kwargs: self.fail("external daemon should not start a new process"),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(record_calls), 1)
        self.assertIn("Using existing Panthera daemon", stderr.getvalue())
        self.assertTrue(any(client.hold_called for client in clients))

    def test_collect_uses_external_daemon_without_auto_stopping_it(self):
        record_calls = []
        clients = []

        def client_factory(socket_path, timeout_s=1.0):
            client = FakeClient(socket_path, mode="mock", active_after=True)
            clients.append(client)
            return client

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            popen_factory=lambda *_args, **_kwargs: self.fail("external daemon should not start a new process"),
            record_runner=lambda command, env=None: record_calls.append((command, env)) or 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(record_calls), 1)
        self.assertTrue(any(client.hold_called for client in clients))
        self.assertFalse(any(client.stop_called for client in clients))

    def test_collect_terminates_owned_daemon_when_readiness_times_out(self):
        process = RunningProcess()

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--startup-timeout-s",
                "0.001",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            popen_factory=lambda *_args, **_kwargs: process,
            record_runner=lambda *_args, **_kwargs: self.fail("record should not start"),
            client_factory=lambda socket_path, timeout_s=1.0: FakeClient(
                socket_path,
                mode="mock",
                active_after=False,
            ),
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 2)
        self.assertTrue(process.terminated or process.killed)

    def test_collect_reaps_owned_daemon_when_ipc_stop_fails_after_confirmed_hold(self):
        process = RunningProcess()
        popen_calls = []

        def client_factory(socket_path, timeout_s=1.0):
            return FakeClient(
                socket_path,
                mode="mock",
                active_after=bool(popen_calls),
                fail_stop=True,
            )

        rc = collect_cli.collect(
            [
                "--mock",
                "--auto-start",
                "--no-cameras",
                "--repo-id",
                "local/demo",
                "--no-release-prompt",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            popen_factory=lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
            record_runner=lambda *_args, **_kwargs: 0,
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(rc, 0)
        self.assertTrue(process.terminated or process.killed)


class DelayedSnapshotHoldingClient:
    def __init__(self):
        self.events = []
        self.hold_called = False
        self.stop_called = False
        self._snapshots = [
            self._snapshot(10, holding=True, control_mode="hold"),
            self._snapshot(11, holding=False, control_mode="hold"),
            self._snapshot(12, holding=True, control_mode="hold"),
        ]

    def hold(self):
        self.hold_called = True
        self.events.append("hold")
        return {"holding": True, "sequence": 10}

    def status(self):
        self.events.append("status:holding")
        return {"holding": True, "sequence": 10}

    def snapshot(self, max_age_s=None):
        snapshot = self._snapshots.pop(0) if len(self._snapshots) > 1 else self._snapshots[0]
        holding = snapshot["diagnostics"]["holding"]
        self.events.append(f"snapshot:{snapshot['sequence']}:{'holding' if holding else 'not_holding'}")
        return snapshot

    def stop(self):
        self.stop_called = True
        self.events.append("stop")
        return True

    @staticmethod
    def _snapshot(sequence, *, holding, control_mode):
        return {
            "observation.state": [0.0] * 14,
            "action": [0.0] * 7,
            "timestamp": time.monotonic(),
            "sequence": sequence,
            "diagnostics": {"holding": holding, "control_mode": control_mode},
            "loop_timing": {},
        }


class NeverConfirmedHoldClient(DelayedSnapshotHoldingClient):
    def __init__(self):
        super().__init__()
        self._snapshots = [self._snapshot(10, holding=True, control_mode="hold")]


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class RunningProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if not self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("panthera-daemon", timeout)
        return 0


class InterruptingRecordProcess:
    pid = 4321

    def __init__(self):
        self.stdout = iter(())
        self.returncode = None
        self.wait_calls = []

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if len(self.wait_calls) <= 2:
            raise subprocess.TimeoutExpired("lerobot-record", timeout)
        self.returncode = -signal.SIGKILL
        return self.returncode


class FakeRecordStatusUI:
    def __init__(self):
        self.started = False
        self.shutdown_called = False
        self.updates = []

    def start(self):
        self.started = True
        return self

    def set_status(self, phase, detail):
        self.updates.append((phase, detail))

    def pump(self):
        pass

    def shutdown(self):
        self.shutdown_called = True


class FakeClient:
    def __init__(self, socket_path, *, mode, active_after, control_mode="teleop", fail_stop=False):
        self.socket_path = socket_path
        self.mode = mode
        self.active_after = active_after
        self.control_mode = control_mode
        self.fail_stop = fail_stop
        self.sequence = 0
        self.hold_called = False
        self.stop_called = False

    def status(self):
        if not self.active_after:
            raise collect_cli.ControlClientError("not connected")
        return {
            "running": True,
            "mode": self.mode,
            "holding": self.hold_called,
            "control_mode": self.control_mode,
            "sequence": self.sequence,
            "timestamp": time.monotonic(),
            "last_error": None,
            "socket_path": self.socket_path,
        }

    def snapshot(self, max_age_s=None):
        if not self.active_after:
            raise collect_cli.ControlClientError("not connected")
        self.sequence += 1
        return {
            "observation.state": [0.0] * 14,
            "action": [0.0] * 7,
            "timestamp": time.monotonic(),
            "sequence": self.sequence,
            "diagnostics": {
                "holding": self.hold_called,
                "control_mode": "hold" if self.hold_called else self.control_mode,
            },
            "loop_timing": {},
        }

    def hold(self):
        self.hold_called = True
        self.control_mode = "hold"
        return self.status()

    def stop(self):
        self.stop_called = True
        if self.fail_stop:
            raise collect_cli.ControlClientError("stop IPC failed")
        return True


class UnconfirmedHoldClient(FakeClient):
    def snapshot(self, max_age_s=None):
        if not self.active_after:
            raise collect_cli.ControlClientError("not connected")
        if not self.hold_called:
            self.sequence += 1
        return {
            "observation.state": [0.0] * 14,
            "action": [0.0] * 7,
            "timestamp": time.monotonic(),
            "sequence": self.sequence,
            "diagnostics": {
                "holding": self.hold_called,
                "control_mode": "hold" if self.hold_called else self.control_mode,
            },
            "loop_timing": {},
        }


def _write_valid_resume_metadata(root: Path, *, task: str, fps: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "robot_type": "panthera_ht_follower",
        "fps": fps,
        "total_tasks": 1,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": list(collect_cli.PANTHERA_OBSERVATION_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": list(collect_cli.PANTHERA_ACTION_NAMES),
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": [0], "task": [task]}),
        meta_dir / "tasks.parquet",
    )


if __name__ == "__main__":
    unittest.main()

class FakePreviewConnection:
    def poll(self, timeout=None):
        if timeout:
            time.sleep(min(timeout, 0.02))
        return False

    def close(self):
        pass


class SuccessfulPreviewConnection:
    def __init__(self, images):
        self.images = images

    def poll(self, timeout=None):
        return True

    def recv(self):
        return "ok", self.images

    def close(self):
        pass


class ClosingPreviewConnection:
    def close(self):
        pass


class FakePreviewProcess:
    exitcode = None

    def __init__(self):
        self.terminated = False
        self.killed = False

    def start(self):
        pass

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return True
