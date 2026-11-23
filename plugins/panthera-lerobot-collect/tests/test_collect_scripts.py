import os
import re
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path

from repo_paths import repo_root

ROOT = Path(__file__).resolve().parents[1]
REPO = repo_root()


class CollectScriptsTests(unittest.TestCase):
    def test_run_script_translates_friendly_live_and_hz_flags(self):
        # daemon 启动脚本(唯一入口)在 <repo>/arm/panthera/run_daemon.sh;
        # 2026-11 目录重组后插件内不再保留转发 shim。
        script = REPO / "arm" / "panthera" / "run_daemon.sh"
        text = script.read_text()
        self.assertTrue(script.stat().st_mode & 0o111, "daemon launcher should be executable")

        self.assertIn("--live", text)
        self.assertIn("--hz", text)
        self.assertIn("--loop-hz", text)
        self.assertIn("finearm.arms.daemon.cli", text)
        self.assertIn(".venv/bin/python", text)
        self.assertRegex(text, r"(^|\n)PYTHONPATH=\"?\"?", "script should clear ROS-injected PYTHONPATH")

        self.assertFalse((ROOT / "run_panthera_lerobot_daemon.sh").exists(),
                         "legacy forwarding shim should be gone — arm/panthera/run_daemon.sh is the only entry")

    def test_collect_script_runs_collect_cli_and_exposes_friendly_flags(self):
        script = ROOT / "run_panthera_lerobot_collect.sh"
        text = script.read_text()
        mode = script.stat().st_mode
        cli_text = (REPO / "src" / "finearm" / "data" / "collector" / "teleop.py").read_text()

        self.assertTrue(mode & 0o111, "collect launcher should be executable")
        self.assertIn("--live", text)
        self.assertIn("--mock", text)
        self.assertIn("--hz", text)
        self.assertIn("--socket", text)
        self.assertIn("finearm.data.cli", text)
        self.assertIn("PANTHERA_PYTHON", text)
        self.assertIn(".venv/bin/python", text)
        self.assertIn(".venv-lerobot/bin/python", text)
        self.assertIn("PYTHONNOUSERSITE=1", text)
        self.assertIn('PYTHONPATH="${REPO_ROOT}/src"', text)
        self.assertIn("lerobot-record", cli_text)

    def test_collect_script_rejects_python_without_sibling_lerobot_record(self):
        script = ROOT / "run_panthera_lerobot_collect.sh"
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            env = dict(os.environ)
            env["PANTHERA_PYTHON"] = str(python)

            result = subprocess.run(
                [str(script), "--mock", "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=5.0,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("lerobot-record", result.stderr)

    def test_collect_script_requires_lerobot_to_be_importable(self):
        script = ROOT / "run_panthera_lerobot_collect.sh"
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            record = bin_dir / "lerobot-record"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-c\" && \"${2:-}\" == *'import_module(\"lerobot\")'* ]]; then\n"
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            record.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            record.chmod(0o755)
            env = dict(os.environ)
            env["PANTHERA_PYTHON"] = str(python)

            result = subprocess.run(
                [str(script), "--mock", "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=5.0,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_collect_script_rejects_contradictory_live_and_mock_flags(self):
        script = ROOT / "run_panthera_lerobot_collect.sh"
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            record = bin_dir / "lerobot-record"
            python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            record.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            record.chmod(0o755)
            env = dict(os.environ)
            env["PANTHERA_PYTHON"] = str(python)

            result = subprocess.run(
                [str(script), "--mock", "--live", "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=5.0,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--live", result.stderr)
        self.assertIn("--mock", result.stderr)

    def test_lerobot_requirements_leave_local_panthera_wheel_to_bootstrap(self):
        text = (ROOT / "requirements-lerobot.txt").read_text().lower()

        self.assertRegex(text, r"(^|\n)lerobot==0\.5\.1($|\n)")
        self.assertIn("init_workspace.sh", text)
        self.assertIn("hightorque_robot", text)
        self.assertNotIn("file://", text)
        self.assertNotIn("/home/jiaming", text)

    def test_workspace_template_and_bootstrap_are_portable_and_pin_sdk(self):
        template = (ROOT / ".env.example").read_text()
        script = ROOT / "scripts" / "init_workspace.sh"
        script_text = script.read_text()

        # .env 全部可选;覆盖项仍在模板中说明。
        for name in (
            "PANTHERA_SDK_ROOT",
            "PANTHERA_DATA_ROOT",
            "PANTHERA_PYTHON",
        ):
            self.assertIn(name, template)
        # 本插件位于 <repo>/plugins/panthera-lerobot-collect,仓库根 = 上两级
        # (PLUGIN_ROOT→REPO_ROOT 两级推导);数据集默认落 <repo>/data/datasets;
        # SDK/标定属机械臂资产,统一在 <repo>/arm/panthera 下。
        # venv 固定为本插件内 .venv-lerobot(Phase 2 起迁 envs/collect/)。
        self.assertIn('PLUGIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"', script_text)
        self.assertIn('REPO_ROOT="$(dirname "$(dirname -- "${PLUGIN_ROOT}")")"', script_text)
        self.assertIn('PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${REPO_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"', script_text)
        self.assertIn('PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${REPO_ROOT}/data/datasets}"', script_text)
        self.assertIn('PANTHERA_PYTHON="${PANTHERA_PYTHON:-${PLUGIN_ROOT}/.venv-lerobot/bin/python}"', script_text)

        self.assertTrue(script.stat().st_mode & 0o111)
        self.assertIn("7f704d90cd3187327a587e9a5c355a2e1f95f987", script_text)
        self.assertIn("cp312-cp312-linux_x86_64.whl", script_text)
        self.assertIn('python_candidate="${PANTHERA_PYTHON}"', script_text)
        self.assertIn("requirements-lerobot.txt", script_text)
        self.assertNotIn("/home/jiaming", script_text)
        syntax = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_tool_has_no_references_to_legacy_panthera_workspace(self):
        # 数据域自包含:入口脚本/初始化脚本/核心代码不得再引用 ~/panthera 或
        # PANTHERA_ROOT(旧项目/旧工作区),保证 deploy 独立发布。
        for path in (
            ROOT / "run_panthera_lerobot_collect.sh",
            ROOT / "scripts" / "init_workspace.sh",
            ROOT / "quest_bridge.py",
        ):
            text = path.read_text()
            self.assertNotIn("PANTHERA_ROOT", text, f"{path.name} still references PANTHERA_ROOT")
            self.assertNotIn('"/panthera"', text, f"{path.name} still references ~/panthera")
            self.assertNotIn("panthera\" / \"sdk", text, f"{path.name} still references ~/panthera/sdk")
            self.assertNotIn("panthera\" / \"data", text, f"{path.name} still references ~/panthera/data")

    def test_collect_cli_default_dataset_root_lands_in_data_hub(self):
        # 未设 PANTHERA_DATA_ROOT 时,数据集默认落 <repo>/data/datasets。
        sys.path.insert(0, str(ROOT))
        try:
            from finearm.data.collector.teleop import _default_data_root
            expected = REPO / "data" / "datasets"
            self.assertEqual(_default_data_root(), expected)
        finally:
            sys.path.remove(str(ROOT))

    def test_collect_wrapper_exports_data_hub_defaults(self):
        # wrapper 在未设置环境变量时,数据集默认指 <repo>/data/datasets,
        # SDK(机械臂资产)默认指 <repo>/arm/panthera/sdk。
        script = ROOT / "run_panthera_lerobot_collect.sh"
        text = script.read_text()
        daemon_text = (REPO / "arm" / "panthera" / "run_daemon.sh").read_text()

        self.assertIn('REPO_ROOT="$(dirname "$(dirname -- "${ROOT_DIR}")")"', text)
        self.assertIn('export PANTHERA_SDK_ROOT="${PANTHERA_SDK_ROOT:-${REPO_ROOT}/arm/panthera/sdk/Panthera-HT_SDK}"', text)
        self.assertIn('export PANTHERA_DATA_ROOT="${PANTHERA_DATA_ROOT:-${REPO_ROOT}/data/datasets}"', text)
        self.assertIn('cam_lock="${PANTHERA_CAM_LOCK:-${REPO_ROOT}/utils/lock_cameras.sh}"', text)
        # daemon(arm/panthera)的 SDK 默认同指 arm/panthera/sdk
        self.assertIn('PANTHERA_SDK_ROOT:-${ARM_ROOT}/sdk/Panthera-HT_SDK', daemon_text)
        # venv 只认本插件/驱动域内 venv,不回退外部环境。
        self.assertNotIn("envs/lerobot", text)
        self.assertNotIn("envs/lerobot", daemon_text)

    def test_camera_lock_script_lives_in_utils_and_wrapper_calls_it(self):
        # utils/lock_cameras.sh(仓库根)是采集与推理共用的唯一权威副本;
        # 本工具不再保留自己的副本(防漂移),wrapper 预检按 PANTHERA_CAM_LOCK/
        # utils/ 顺序调用。
        script = REPO / "utils" / "lock_cameras.sh"
        text = script.read_text()

        self.assertTrue(script.stat().st_mode & 0o111)
        self.assertIn("HD USB CAMERA", text)
        self.assertIn("MX Brio", text)
        self.assertIn("exposure_time_absolute=36", text)
        self.assertIn("唯一权威副本", text)
        syntax = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        self.assertFalse((ROOT / "scripts" / "lock_cameras.sh").exists(),
                         "工具内不应再有 lock_cameras.sh 副本(权威副本在 utils/)")
        wrapper = (ROOT / "run_panthera_lerobot_collect.sh").read_text()
        self.assertIn('PANTHERA_CAM_LOCK:-${REPO_ROOT}/utils/lock_cameras.sh', wrapper)

    def test_readmes_direct_users_to_lock_cameras_first(self):
        # 采集(data/)与推理(infer/;Phase 2 起 envs/infer/)的 README——
        # 中英双份——都必须写明:相机参数必须与训练/推理一致,并给出
        # bash utils/lock_cameras.sh(仓库根执行)。说明文字按各自语言
        # 匹配实质,不绑定单一中文文案(README 已是中英双份)。
        infer_dir = next(
            (d for d in (REPO / "infer", REPO / "envs" / "infer") if (d / "README.md").is_file()),
            None,
        )
        self.assertIsNotNone(infer_dir, "infer/ 下应有 README.md(Phase 2 起为 envs/infer/)")

        for readme, phrase in (
            (REPO / "data" / "README.md", "camera params must match"),
            (REPO / "data" / "README_zh.md", "相机参数必须与训练/推理一致"),
            (infer_dir / "README.md", "camera params must match"),
            (infer_dir / "README_zh.md", "相机参数必须与训练/推理一致"),
        ):
            text = readme.read_text()
            self.assertIn("utils/lock_cameras.sh", text, f"{readme} missing camera-lock step")
            self.assertIn(phrase, text, f"{readme} should explain locking camera params first")
            self.assertIn("bash utils/lock_cameras.sh", text, f"{readme} should show the runnable command")

    def test_pyproject_declares_collection_packages_and_scripts_only(self):
        import tomllib

        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

        # LeRobot 按已安装发行名 import 同名模块发现插件:发行名必须与某个实际
        # 插件包名一致,否则 lerobot-record 报 --robot.type invalid choice
        # (2026-11-10 实机踩坑:发行名起成 lerobot_robot_panthera_collect 时插件
        # 注册失败,真机采集在 lerobot-record 参数解析处崩掉)。
        dist_name = metadata["project"]["name"]
        packages = set(metadata["tool"]["setuptools"]["packages"])
        self.assertIn(dist_name, packages, "发行名必须是 packages 之一的同名插件包")

        # 只包含采集链路包;训练/推理包不属于本工具。
        self.assertEqual(
            set(metadata["tool"]["setuptools"]["packages"]),
            {
                "panthera_lerobot_control",
                "lerobot_robot_panthera_ht",
                "lerobot_teleoperator_panthera_ht",
            },
        )

    def test_docs_contain_official_lerobot_record_command_and_architecture(self):
        text = (ROOT / "docs" / "lerobot_native_collection.md").read_text()

        self.assertIn("lerobot-record", text)
        required_record_flags = [
            "--robot.type=panthera_ht_follower",
            "--robot.id=panthera_follower",
            "--robot.socket_path=/tmp/panthera_lerobot_control.sock",
            "--robot.cameras=",
            "front",
            "wrist",
            "--teleop.type=panthera_ht_leader",
            "--teleop.id=panthera_leader",
            "--teleop.socket_path=/tmp/panthera_lerobot_control.sock",
            "--dataset.repo_id=local/panthera_smolvla_demo",
            "--dataset.push_to_hub=False",
            "--dataset.num_episodes=50",
            "--dataset.episode_time_s=60",
            "--dataset.reset_time_s=60",
            "--dataset.streaming_encoding=true",
            "--dataset.encoder_threads=2",
        ]
        for flag in required_record_flags:
            self.assertIn(flag, text)
        self.assertNotIn("上传策略和具体 episode/reset 参数按安装的 LeRobot 版本", text)
        self.assertIn("./run_panthera_lerobot_collect.sh --live", text)
        self.assertIn("--select-cameras", text)
        self.assertIn("--print-command", text)
        self.assertIn("bash arm/panthera/run_daemon.sh --live --hz 1000", text)
        self.assertIn("1000 Hz", text)
        self.assertIn("teleoperation", text.lower())
        self.assertIn("policy inference", text.lower())
        self.assertIn("observation.state", text)
        self.assertIn("hold", text)
        self.assertIn("stop", text)
        self.assertIn("lerobot==0.5.1", text)
        self.assertIn("YYYYMMDD_HHMMSS", text)
        self.assertIn("左方向键", text)
        self.assertIn("Esc", text)

    def test_native_collection_runbook_matches_verified_hardware_profile(self):
        text = (ROOT / "docs" / "lerobot_native_collection.md").read_text()

        required = [
            "source .env",
            "PANTHERA_PYTHON",
            "PANTHERA_SDK_ROOT",
            "PANTHERA_DATA_ROOT",
            "usb-SN0002_1080P_USB_Camera_44434000_P030C01_SN0002-video-index0",
            "usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0",
            "--repo-id local/pick_red_cube_into_bowl",
            '--task "pick up the red cube and place it in the bowl"',
            "--resume",
            "--dataset-root",
            "298/300",
            "物理断电",
            "owned daemon",
        ]
        for value in required:
            self.assertIn(value, text)

        stale_formal_examples = [
            "--camera front=/dev/video0",
            "--camera wrist=/dev/video2",
        ]
        for value in stale_formal_examples:
            self.assertNotIn(value, text)

    def test_quest2ros_manual_points_at_this_project_root(self):
        text = (ROOT / "docs" / "quest2ros_setup.md").read_text()

        self.assertIn("panthera-lerobot-collect", text)
        self.assertNotIn("cd ~/internship/baiqin/project-Embodied_AGV", text)
        self.assertIn("quest_bridge.py --side right", text)
        self.assertIn("--initial-control-mode freedrive", text)
        self.assertIn("--leader-config", text)

    def test_manual_uses_data_hub_paths_for_configs_and_datasets(self):
        # 手册命令的相对路径一律仓库根锚定(arm/panthera/robot_param、data/datasets),
        # 由 adapters 的项目路径解析器保证任意 CWD 可用;不再指 ~/panthera,
        # 也不再残留 deploy/ 前缀(2026-11 目录重组)。
        text = (ROOT / "docs" / "quest2ros_setup.md").read_text()

        self.assertIn("--leader-config arm/panthera/robot_param/Leader.yaml", text)
        self.assertIn("--follower-config arm/panthera/robot_param/Follower.yaml", text)
        self.assertIn("data/datasets/lerobot/", text)
        self.assertNotIn("deploy/", text)
        self.assertNotIn("~/panthera/robot_param", text)
        self.assertNotIn("~/panthera/data", text)


if __name__ == "__main__":
    unittest.main()
