"""LeRobot robot plugin for Panthera HT follower state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from panthera_lerobot_control import ControlClient
from panthera_lerobot_control.constants import DEFAULT_SOCKET_PATH
from panthera_lerobot_control.sample_coordinator import clear_snapshot, publish_snapshot

try:
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs
except ImportError:  # pragma: no cover - exercised by compatibility tests with fake LeRobot modules
    CameraConfig = object
    make_cameras_from_configs = None

from ._compat import Robot, RobotConfig

ROBOT_TYPE = "panthera_ht_follower"

OBSERVATION_KEYS = (
    "follower_joint_1.pos",
    "follower_joint_2.pos",
    "follower_joint_3.pos",
    "follower_joint_4.pos",
    "follower_joint_5.pos",
    "follower_joint_6.pos",
    "follower_gripper.pos",
    "follower_joint_1.vel",
    "follower_joint_2.vel",
    "follower_joint_3.vel",
    "follower_joint_4.vel",
    "follower_joint_5.vel",
    "follower_joint_6.vel",
    "follower_gripper.vel",
)


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
class PantheraHTRobotConfig(RobotConfig):
    socket_path: str = DEFAULT_SOCKET_PATH
    timeout_s: float = 1.0
    stale_timeout_s: float = 0.25
    send_action_mode: str = "validate_only"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cameras and all(_has_camera_runtime_fields(config) for config in self.cameras.values()):
            parent_post_init = getattr(super(), "__post_init__", None)
            if callable(parent_post_init):
                parent_post_init()


PantheraHTRobotConfig = _register_config(RobotConfig, ROBOT_TYPE, PantheraHTRobotConfig)


class PantheraHTRobot(Robot):
    """Socket-backed, read-only LeRobot adapter for Panthera HT follower state."""

    config_class = PantheraHTRobotConfig
    name = ROBOT_TYPE

    def __init__(self, config: PantheraHTRobotConfig, client=None):
        _init_base(self, config, PantheraHTRobot)
        self.config = config
        self._client = client
        self._connected = False
        self.cameras = _make_camera_runtimes(config.cameras)

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**{key: float for key in OBSERVATION_KEYS}, **_camera_features(self.cameras)}

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self._connected and all(_camera_is_connected(camera) for camera in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        clear_snapshot(self.config.socket_path)
        if self._client is None:
            self._client = ControlClient(self.config.socket_path, timeout_s=self.config.timeout_s)
        for camera in self.cameras.values():
            _connect_camera(camera)
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, object]:
        clear_snapshot(self.config.socket_path)
        snapshot = self._require_client().snapshot(max_age_s=self.config.stale_timeout_s)
        _raise_for_daemon_error(snapshot)
        observation: dict[str, object] = _map_values(snapshot["observation.state"], OBSERVATION_KEYS, "observation.state")
        publish_snapshot(self.config.socket_path, snapshot)
        try:
            for camera_key, camera in self.cameras.items():
                observation[camera_key] = _read_camera(camera)
        except BaseException:
            clear_snapshot(self.config.socket_path)
            raise
        return observation

    def send_action(self, action):
        _validate_action_dict(action)
        if self.config.send_action_mode == "validate_only":
            return action
        if self.config.send_action_mode == "policy_ipc":
            self._require_client().send_policy_action(_action_values(action))
            return action
        raise ValueError(f"unknown send_action_mode: {self.config.send_action_mode!r}")

    def disconnect(self) -> None:
        clear_snapshot(self.config.socket_path)
        for camera in self.cameras.values():
            _disconnect_camera(camera)
        self._connected = False

    def _require_client(self):
        if not self._connected:
            raise RuntimeError("PantheraHTRobot is not connected")
        if self._client is None:
            raise RuntimeError("PantheraHTRobot has no control client")
        return self._client


def _make_camera_runtimes(cameras: dict[str, CameraConfig]) -> dict:
    if not cameras:
        return {}
    if all(_has_camera_config_fields(config) for config in cameras.values()):
        if make_cameras_from_configs is None:
            raise RuntimeError("LeRobot camera runtime is unavailable")
        return make_cameras_from_configs(cameras)
    return cameras


def _camera_features(cameras: dict) -> dict[str, tuple[int, int, int]]:
    features = {}
    for key, camera in cameras.items():
        height = getattr(camera, "height", None) or getattr(getattr(camera, "config", None), "height", None)
        width = getattr(camera, "width", None) or getattr(getattr(camera, "config", None), "width", None)
        if height is not None and width is not None:
            features[key] = (int(height), int(width), 3)
    return features


def _camera_is_connected(camera) -> bool:
    return bool(getattr(camera, "is_connected", True))


def _connect_camera(camera) -> None:
    connect = getattr(camera, "connect", None)
    if callable(connect) and not _camera_is_connected(camera):
        connect()


def _read_camera(camera):
    read_latest = getattr(camera, "read_latest", None)
    if callable(read_latest):
        return read_latest()
    read = getattr(camera, "read", None)
    if callable(read):
        return read()
    raise RuntimeError(f"camera {camera!r} does not provide read_latest() or read()")


def _disconnect_camera(camera) -> None:
    disconnect = getattr(camera, "disconnect", None)
    if callable(disconnect):
        disconnect()


def _has_camera_config_fields(config) -> bool:
    return hasattr(config, "type") and all(hasattr(config, attr) for attr in ("width", "height", "fps"))


def _has_camera_runtime_fields(config) -> bool:
    return all(hasattr(config, attr) for attr in ("width", "height", "fps"))


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


def _action_values(action) -> list[float]:
    return [float(action[key]) for key in ACTION_KEYS]


def _validate_action_dict(action) -> None:
    if not isinstance(action, dict):
        raise TypeError("action must be a dictionary")
    expected = set(ACTION_KEYS)
    actual = set(action)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        raise ValueError("; ".join(parts))
    for key in ACTION_KEYS:
        if not _is_number(action[key]):
            raise TypeError(f"action value for {key!r} must be numeric")
        if not math.isfinite(float(action[key])):
            raise ValueError(f"action value for {key!r} must be finite")


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


__all__ = [
    "ACTION_KEYS",
    "OBSERVATION_KEYS",
    "PantheraHTRobot",
    "PantheraHTRobotConfig",
]

# LeRobot discovers third-party packages by distribution/package name. Importing
# this package must register the paired teleoperator config as well.
import lerobot_teleoperator_panthera_ht as _panthera_ht_teleoperator_plugin
