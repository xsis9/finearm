"""Directory-scan plugin registry: drop a module in, it is discovered.

Conventions (mirrors ``infer/adapters`` — "the filename is the registry"):

- A plugin package (e.g. ``finearm.infer.adapters``) is a regular package
  whose submodules/subpackages are plugins.
- Names starting with ``_`` are private infrastructure and skipped.
- :func:`scan_names` lists names **without importing** the plugins;
  :func:`load_attr` imports one plugin lazily and fetches its declaration
  attribute (``ADAPTER`` / ``ALGORITHM_ID`` / ``PROFILE`` / ...).

No entry points: plugin discovery must work from an uninstalled checkout
(the archived refactor plan (git history) design principle 4).
"""

from __future__ import annotations

import importlib
import pkgutil


class RegistryError(RuntimeError):
    """Plugin lookup failed (unknown package/plugin or missing declaration)."""


def scan_names(package: str) -> list[str]:
    """Sorted plugin names inside ``package``, without importing the plugins."""
    try:
        pkg = importlib.import_module(package)
    except ImportError as exc:
        raise RegistryError(f"Plugin package not importable: {package!r}") from exc
    search_paths = getattr(pkg, "__path__", None)
    if search_paths is None:
        raise RegistryError(f"Not a package (no __path__): {package!r}")
    names = [
        name
        for _, name, _is_pkg in pkgutil.iter_modules(search_paths)
        if not name.startswith("_")
    ]
    return sorted(names)


def load_attr(package: str, name: str, attr: str):
    """Import ``package.name`` lazily and return its ``attr`` declaration."""
    module_name = f"{package}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RegistryError(f"Plugin not importable: {module_name!r}") from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        raise RegistryError(f"Plugin {module_name!r} has no attribute {attr!r}") from None


__all__ = ["RegistryError", "load_attr", "scan_names"]
