"""Arm plugin discovery: every directory under ``finearm/arms/`` is an arm.

An arm plugin package declares ``ARM_ID``, ``PROFILE`` (:class:`ArmProfile`),
``GAINS`` (:class:`~finearm.arms.daemon.ControlGains`), and the adapter
factories ``make_pair`` / ``make_follower_only`` / ``make_mock_pair``
(the archived refactor plan (git history) §3a). Discovery is a directory scan (no entry points, no
installation required) — drop a directory in and it is available.
"""

from __future__ import annotations

import importlib

from ..core.registry import RegistryError, scan_names
from .profile import ArmProfile, JointRange

#: Arm used when a config/CLI does not name one (the shipped arm).
DEFAULT_ARM = "panthera_ht"

#: Non-arm infrastructure living under finearm/arms/ (excluded from scans).
_INFRA_MODULES = frozenset({"daemon", "profile"})


def available() -> list[str]:
    """Sorted arm plugin ids discoverable in this checkout."""
    return sorted(set(scan_names(__name__)) - _INFRA_MODULES)


def load_arm(arm_id: str):
    """Import an arm plugin and validate its declarations."""
    if arm_id in _INFRA_MODULES or not arm_id.isidentifier():
        raise RegistryError(f"Unknown arm plugin: {arm_id!r}")
    try:
        plugin = importlib.import_module(f"{__name__}.{arm_id}")
    except ImportError as exc:
        raise RegistryError(f"Arm plugin not importable: {arm_id!r}") from exc
    missing = [name for name in ("ARM_ID", "PROFILE", "make_mock_pair") if not hasattr(plugin, name)]
    if missing:
        raise RegistryError(f"Arm plugin {arm_id!r} is missing declarations: {', '.join(missing)}")
    profile = plugin.PROFILE
    if not isinstance(profile, ArmProfile):
        raise RegistryError(f"Arm plugin {arm_id!r}: PROFILE must be an ArmProfile")
    return plugin


def load_profile(arm_id: str) -> ArmProfile:
    """The ArmProfile of one arm."""
    return load_arm(arm_id).PROFILE


__all__ = ["ArmProfile", "DEFAULT_ARM", "JointRange", "available", "load_arm", "load_profile"]
