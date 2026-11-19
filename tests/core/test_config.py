"""Tests for finearm.core.config: JSONC loading and layer 1-3 resolution."""

from __future__ import annotations

import json

import pytest

from finearm.core import config


class TestStripJsonComments:
    def test_line_and_block_comments(self) -> None:
        out = config.strip_json_comments('{\n  // line\n  "a": /* block */ 1,\n  "b": 2\n}')
        assert json.loads(out) == {"a": 1, "b": 2}

    def test_urls_in_strings_survive(self) -> None:
        out = config.strip_json_comments('{"u": "https://host/path", // c\n "b": 2}')
        data = json.loads(out)
        assert data["u"] == "https://host/path"
        assert data["b"] == 2


class TestLoadJsonc:
    def test_comments_and_null_values(self, fake_repo) -> None:
        p = fake_repo / "configs" / "local.jsonc"
        p.write_text(
            '{\n  // machine-local overrides\n  "stats_path": "/tmp/s.json",\n  "dataset_root": null\n}\n'
        )
        assert config.load_jsonc(p) == {"stats_path": "/tmp/s.json", "dataset_root": None}

    def test_invalid_json_raises_value_error(self, tmp_path) -> None:
        p = tmp_path / "bad.jsonc"
        p.write_text("{ not json }")
        with pytest.raises(ValueError, match="Invalid JSONC"):
            config.load_jsonc(p)

    def test_non_object_raises_value_error(self, tmp_path) -> None:
        p = tmp_path / "arr.jsonc"
        p.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            config.load_jsonc(p)


class TestLocalOverrides:
    def test_missing_file_is_empty(self, fake_repo) -> None:
        assert config.local_overrides(fake_repo) == {}

    def test_present_file(self, fake_repo) -> None:
        (fake_repo / "configs" / "local.jsonc").write_text('{"sdk_root": "/opt/sdk"}')
        assert config.local_overrides(fake_repo) == {"sdk_root": "/opt/sdk"}

    def test_none_repo_uses_discovery(self, monkeypatch, fake_repo) -> None:
        from finearm.core import paths

        monkeypatch.setattr(paths, "find_repo_root", lambda start=None: fake_repo)
        (fake_repo / "configs" / "local.jsonc").write_text('{"sdk_root": "/opt/sdk"}')
        assert config.local_overrides(None) == {"sdk_root": "/opt/sdk"}

    def test_none_repo_without_discovery_is_empty(self, monkeypatch) -> None:
        from finearm.core import paths

        monkeypatch.delenv(paths.ROOT_ENV, raising=False)
        monkeypatch.setattr(paths, "find_repo_root", lambda start=None: None)
        assert config.local_config_path(None) is None
        assert config.local_overrides(None) == {}


class TestResolve:
    def test_layer_order(self, monkeypatch) -> None:
        monkeypatch.setenv("FINEARM_T_RESOLVE", "env-value")
        assert config.resolve("k", cli="cli-value", env="FINEARM_T_RESOLVE", overrides={}) == "cli-value"
        assert config.resolve("k", env="FINEARM_T_RESOLVE", overrides={}) == "env-value"
        assert config.resolve("k", env="FINEARM_T_RESOLVE", overrides={"k": "local"}) == "env-value"
        assert config.resolve("k", overrides={"k": "local"}) == "local"
        assert config.resolve("k", overrides={}) is None
        assert config.resolve("k", overrides={"k": None}) is None

    def test_env_empty_counts_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv("FINEARM_T_RESOLVE", "")
        assert config.resolve("k", env="FINEARM_T_RESOLVE", overrides={"k": "local"}) == "local"

    def test_no_env_layer_without_name(self, monkeypatch) -> None:
        monkeypatch.setenv("k", "stray-env-value")
        assert config.resolve("k", overrides={"k": "local"}) == "local"
