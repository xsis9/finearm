"""Tests for finearm.core.paths: repo-root discovery and four-layer path resolution."""

from __future__ import annotations

from pathlib import Path

from finearm.core import paths


class TestFindRepoRoot:
    def test_walks_up_to_marker(self, fake_repo: Path) -> None:
        deep = fake_repo / "RL" / "rlt_panthera" / "scripts"
        deep.mkdir(parents=True)
        assert paths.find_repo_root(deep) == fake_repo

    def test_none_without_marker(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
        deep = tmp_path / "a"
        deep.mkdir()
        assert paths.find_repo_root(deep) is None

    def test_env_override_wins(self, fake_repo: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(paths.ROOT_ENV, str(fake_repo))
        assert paths.find_repo_root(tmp_path) == fake_repo

    def test_accepts_file_start(self, fake_repo: Path) -> None:
        marker_file = fake_repo / "configs" / "x.json"
        marker_file.write_text("{}")
        assert paths.find_repo_root(marker_file) == fake_repo


class TestDefaultPath:
    def test_relative_defaults_anchor_at_repo(self, fake_repo: Path) -> None:
        assert paths.default_path("dataset_root", repo=fake_repo) == fake_repo / "data" / "datasets"
        assert paths.default_path("sdk_root", repo=fake_repo) == fake_repo / "arm" / "panthera" / "sdk"

    def test_acquisition_interface_has_no_repo_path_default(self, fake_repo: Path) -> None:
        # The acquisition backend defaults to the finearm.infer.robot_interface
        # package module — a repo-path default here once dangled after the
        # src-layout move (2026-11). It must stay out of DEFAULT_PATHS.
        assert "acquisition_interface" not in paths.DEFAULT_PATHS
        assert paths.default_path("acquisition_interface", repo=fake_repo) is None

    def test_absolute_default_needs_no_repo(self) -> None:
        assert paths.default_path("socket_dir") == Path("/tmp")

    def test_unknown_key_is_none(self, fake_repo: Path) -> None:
        assert paths.default_path("stats_path", repo=fake_repo) is None

    def test_relative_default_without_repo_is_none(self, monkeypatch) -> None:
        monkeypatch.delenv(paths.ROOT_ENV, raising=False)
        monkeypatch.setattr(paths, "find_repo_root", lambda start=None: None)
        assert paths.default_path("dataset_root") is None


class TestResolvePath:
    def test_layer4_default(self, fake_repo: Path) -> None:
        assert paths.resolve_path("dataset_root", repo=fake_repo) == fake_repo / "data" / "datasets"

    def test_layer3_local_jsonc(self, fake_repo: Path) -> None:
        (fake_repo / "configs" / "local.jsonc").write_text('{"dataset_root": "/data/hub"}')
        assert paths.resolve_path("dataset_root", repo=fake_repo) == Path("/data/hub")

    def test_layer2_env(self, fake_repo: Path, monkeypatch) -> None:
        (fake_repo / "configs" / "local.jsonc").write_text('{"dataset_root": "/data/hub"}')
        monkeypatch.setenv("FINEARM_T_PATH", "/env/root")
        assert (
            paths.resolve_path("dataset_root", env="FINEARM_T_PATH", repo=fake_repo)
            == Path("/env/root")
        )

    def test_layer1_cli_wins(self, fake_repo: Path, monkeypatch) -> None:
        monkeypatch.setenv("FINEARM_T_PATH", "/env/root")
        assert (
            paths.resolve_path(
                "dataset_root", cli="/cli/root", env="FINEARM_T_PATH", repo=fake_repo
            )
            == Path("/cli/root")
        )

    def test_env_set_but_empty_is_unset(self, fake_repo: Path, monkeypatch) -> None:
        (fake_repo / "configs" / "local.jsonc").write_text('{"dataset_root": "/data/hub"}')
        monkeypatch.setenv("FINEARM_T_EMPTY", "")
        assert (
            paths.resolve_path("dataset_root", env="FINEARM_T_EMPTY", repo=fake_repo)
            == Path("/data/hub")
        )

    def test_key_without_default(self, fake_repo: Path) -> None:
        assert paths.resolve_path("stats_path", repo=fake_repo) is None
        assert paths.resolve_path("stats_path", cli="/s.json", repo=fake_repo) == Path("/s.json")

    def test_local_jsonc_null_falls_through_to_default(self, fake_repo: Path) -> None:
        (fake_repo / "configs" / "local.jsonc").write_text('{"dataset_root": null}')
        assert paths.resolve_path("dataset_root", repo=fake_repo) == fake_repo / "data" / "datasets"

    def test_without_repo_root_only_absolute_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv(paths.ROOT_ENV, raising=False)
        monkeypatch.setattr(paths, "find_repo_root", lambda start=None: None)
        assert paths.resolve_path("socket_dir") == Path("/tmp")
        assert paths.resolve_path("dataset_root") is None
