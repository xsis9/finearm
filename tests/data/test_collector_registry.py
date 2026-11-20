"""Collector plugin registry + dagger stub + teleop --arm parameterization."""

from __future__ import annotations

from pathlib import Path

import pytest

from finearm.data.collector import available, load
from finearm.core.registry import RegistryError


def test_available_lists_methods_not_infra() -> None:
    methods = available()
    assert "teleop" in methods
    assert "dagger" in methods
    assert "base" not in methods


def test_load_validates_method_id() -> None:
    teleop = load("teleop")
    assert teleop.method_id == "teleop"
    assert callable(teleop.main)


def test_unknown_method_rejected() -> None:
    with pytest.raises(RegistryError):
        load("no_such_method")


def test_dagger_stub_present_but_raises() -> None:
    dagger = load("dagger")
    assert dagger.method_id == "dagger"
    with pytest.raises(NotImplementedError):
        dagger.DAggerCollector()


class TestArmParameterization:
    """--arm derives the record command's plugin types and resume checks."""

    @staticmethod
    def _args(argv):
        from finearm.data.collector.teleop import build_parser

        return build_parser().parse_args(argv)

    def test_default_arm_is_panthera_ht(self) -> None:
        args = self._args(["--mock"])
        assert args.arm == "panthera_ht"

    def test_record_command_derives_plugin_types_from_arm(self) -> None:
        from finearm.data.collector.teleop import build_lerobot_record_command

        args = self._args(
            ["--mock", "--arm", "mock_twojoint", "--repo-id", "x/y",
             "--episodes", "1", "--episode-seconds", "5", "--reset-seconds", "1",
             "--no-cameras", "--record-bin", "lerobot-record"]
        )
        args._resolved_dataset_root = "/tmp/ds"
        command = build_lerobot_record_command(args, camera_specs=())
        assert "--robot.type=mock_twojoint_follower" in command
        assert "--teleop.type=mock_twojoint_leader" in command

    def test_default_panthera_types_unchanged(self) -> None:
        from finearm.data.collector.teleop import build_lerobot_record_command

        args = self._args(
            ["--mock", "--repo-id", "x/y", "--episodes", "1",
             "--episode-seconds", "5", "--reset-seconds", "1", "--no-cameras"]
        )
        args._resolved_dataset_root = "/tmp/ds"
        command = build_lerobot_record_command(args, camera_specs=())
        assert "--robot.type=panthera_ht_follower" in command
        assert "--teleop.type=panthera_ht_leader" in command
        assert "--robot.id=panthera_follower" in command
        assert "--teleop.id=panthera_leader" in command

    def test_resume_robot_type_check_is_arm_derived(self, tmp_path, monkeypatch) -> None:
        from finearm.data.collector.teleop import CollectError, validate_resume_dataset

        def make_info(robot_type: str) -> Path:
            import json

            root = tmp_path / robot_type.replace("/", "_")
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"robot_type": robot_type, "fps": 30, "total_tasks": 1,
                            "features": {}}))
            return root

        args = self._args(["--mock", "--fps", "30", "--arm", "panthera_ht"])
        with pytest.raises(CollectError, match="panthera_ht_follower"):
            validate_resume_dataset(make_info("other_arm_follower"), args)

    def test_serial_permissions_name_importable_for_tests(self) -> None:
        from finearm.data.collector import teleop

        assert callable(teleop.ensure_panthera_serial_access)
