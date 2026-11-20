"""Collection method registry: every directory here is a Collector plugin."""

from __future__ import annotations

import importlib

from ...core.registry import RegistryError, scan_names
from .base import CollectError, Collector, PolicySource

#: Non-collector infrastructure under this package (excluded from scans).
_INFRA_MODULES = frozenset({"base"})


def available() -> list[str]:
    """Collector method ids (directory names)."""
    return sorted(set(scan_names(__package__)) - _INFRA_MODULES)


def load(method: str):
    """Import a collector plugin module and validate its ``method_id``."""
    if method in _INFRA_MODULES or not method.isidentifier():
        raise RegistryError(f"Unknown collector method: {method!r}")
    try:
        module = importlib.import_module(f"{__package__}.{method}")
    except ImportError as exc:
        raise RegistryError(f"Collector method not importable: {method!r}") from exc
    if getattr(module, "method_id", None) != method:
        raise RegistryError(
            f"Collector plugin {method!r} declares method_id="
            f"{getattr(module, 'method_id', None)!r} (must match the directory name)"
        )
    return module


__all__ = ["CollectError", "Collector", "PolicySource", "available", "load"]
