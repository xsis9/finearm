"""LeRobot teleoperator plugin for Panthera HT leader-derived actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from finearm.arms.daemon.ipc import ControlClient
from finearm.arms.panthera_ht import PROFILE as _P
DEFAULT_SOCKET_PATH = _P.socket_path
from panthera_lerobot_control.sample_coordinator import clear_snapshot, consume_snapshot

from ._compat import Teleoperator, TeleoperatorConfig

TELEOPERATOR_TYPE = "panthera_ht_leader"


def _register_config(base_config, type_name: str, config_cls):
    try:
        registered = base_config.register_subclass(type_name)(config_cls)
    except ValueError:
        registry = getattr(base_config, "_choice_registry", None)
        existing = registry.get(type_name) if isinstance(registry, dict) else None
        if (
            existing is not None
            and getattr(existing, "__module__", None) == config_cls.__module__
            and getattr(existing, "__name__", None) == config_cls.__name__
        ):
            registry[type_name] = config_cls
            registered = config_cls
        else:
            raise
    _bind_choice_registry(base_config, type_name, registered)
    return registered


def _bind_choice_registry(base_config, type_name: str, config_cls):
    registry = getattr(base_config, "_choice_registry", None)
    if isinstance(registry, dict):
        stable_registry = dict(registry)
        stable_registry[type_name] = config_cls
        config_cls._choice_registry = stable_registry
    config_cls._registered_choice_name = type_name


# Feature keys come from the arm profile (Phase 2 single-sourcing); the
# literals are the no-finearm fallback (identical values, die in Phase 4).
try:
    from finearm.arms.panthera_ht import PROFILE as _ARM_PROFILE

    ACTION_KEYS = _ARM_PROFILE.action_feature_names
except ImportError:  # pragma: no cover - legacy fallback outside the repo tree
    ACTION_KEYS = (
        "target_joint_1.pos",
        "target_joint_2.pos",
        "target_joint_3.pos",
        "target_joint_4.pos",
        "target_joint_5.pos",
        "target_joint_6.pos",
        "target_gripper.pos",
    )


@dataclass
class PantheraHTTeleoperatorConfig(TeleoperatorConfig):
    socket_path: str = DEFAULT_SOCKET_PATH
    timeout_s: float = 1.0
    stale_timeout_s: float = 0.25


PantheraHTTeleoperatorConfig = _register_config(TeleoperatorConfig, TELEOPERATOR_TYPE, PantheraHTTeleoperatorConfig)


class PantheraHTTeleoperator(Teleoperator):
    """Socket-backed LeRobot teleoperator adapter for Panthera HT actions."""

    config_class = PantheraHTTeleoperatorConfig
    name = TELEOPERATOR_TYPE

    def __init__(self, config: PantheraHTTeleoperatorConfig, client=None):
        _init_base(self, config, PantheraHTTeleoperator)
        self.config = config
        self._client = client
        self._connected = False

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        clear_snapshot(self.config.socket_path)
        if self._client is None:
            self._client = ControlClient(self.config.socket_path, timeout_s=self.config.timeout_s)
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, float]:
        client = self._require_client()
        snapshot = consume_snapshot(self.config.socket_path)
        if snapshot is None:
            snapshot = client.snapshot(max_age_s=self.config.stale_timeout_s)
        _raise_for_daemon_error(snapshot)
        return _map_values(snapshot["action"], ACTION_KEYS, "action")

    def send_feedback(self, feedback) -> None:
        return None

    def disconnect(self) -> None:
        clear_snapshot(self.config.socket_path)
        self._connected = False

    def _require_client(self):
        if not self._connected:
            raise RuntimeError("PantheraHTTeleoperator is not connected")
        if self._client is None:
            raise RuntimeError("PantheraHTTeleoperator has no control client")
        return self._client


def _init_base(instance, config, cls) -> None:
    try:
        super(cls, instance).__init__(config)
    except TypeError:
        super(cls, instance).__init__()


def _raise_for_daemon_error(snapshot: dict) -> None:
    diagnostics = snapshot.get("diagnostics", {})
    last_error = diagnostics.get("last_error") if isinstance(diagnostics, dict) else None
    if last_error:
        raise RuntimeError(f"Panthera daemon reported a latched control error: {last_error}")


def _map_values(values, keys, label: str) -> dict[str, float]:
    if not isinstance(values, list) or len(values) != len(keys):
        raise ValueError(f"{label} must contain {len(keys)} values")
    mapped = {}
    for key, value in zip(keys, values):
        if not _is_number(value):
            raise TypeError(f"{label} value for {key!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} value for {key!r} must be finite")
        mapped[key] = number
    return mapped


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


__all__ = [
    "ACTION_KEYS",
    "PantheraHTTeleoperator",
    "PantheraHTTeleoperatorConfig",
]
