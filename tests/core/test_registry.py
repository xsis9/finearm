"""Tests for finearm.core.registry: directory-scan plugin discovery."""

from __future__ import annotations

import pytest

from finearm.core import registry


@pytest.fixture
def plugin_pkg(tmp_path, monkeypatch):
    pkg = tmp_path / "fake_plugins"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text("VALUE = 'a'\n")
    (pkg / "beta.py").write_text("OTHER = 'b'\n")
    (pkg / "_private.py").write_text("VALUE = 'p'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "fake_plugins"


def test_scan_names_lists_public_modules(plugin_pkg) -> None:
    assert registry.scan_names(plugin_pkg) == ["alpha", "beta"]


def test_scan_names_unknown_package() -> None:
    with pytest.raises(registry.RegistryError, match="not importable"):
        registry.scan_names("finearm_no_such_package")


def test_scan_names_rejects_non_package(tmp_path, monkeypatch) -> None:
    mod = tmp_path / "single_module.py"
    mod.write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(registry.RegistryError, match="no __path__"):
        registry.scan_names("single_module")


def test_load_attr_lazy_import(plugin_pkg) -> None:
    assert registry.load_attr(plugin_pkg, "alpha", "VALUE") == "a"


def test_load_attr_missing_declaration(plugin_pkg) -> None:
    with pytest.raises(registry.RegistryError, match="VALUE"):
        registry.load_attr(plugin_pkg, "beta", "VALUE")


def test_load_attr_unknown_plugin(plugin_pkg) -> None:
    with pytest.raises(registry.RegistryError, match="not importable"):
        registry.load_attr(plugin_pkg, "gamma", "VALUE")
