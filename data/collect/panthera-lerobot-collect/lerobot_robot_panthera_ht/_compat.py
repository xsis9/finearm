"""Small LeRobot compatibility layer for optional third-party installs."""

from __future__ import annotations

from dataclasses import dataclass


def _import_first(candidates):
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


_RealRobotConfig = _import_first(
    [
        ("lerobot.robots.config", "RobotConfig"),
        ("lerobot.common.robots.configs", "RobotConfig"),
        ("lerobot.common.robot_devices.robots.configs", "RobotConfig"),
        ("lerobot.common.robots", "RobotConfig"),
    ]
)
_RealRobot = _import_first(
    [
        ("lerobot.robots.robot", "Robot"),
        ("lerobot.common.robots.robot", "Robot"),
        ("lerobot.common.robot_devices.robots.robot", "Robot"),
        ("lerobot.common.robots", "Robot"),
    ]
)


class _FallbackChoiceRegistry:
    _choice_registry = {}

    @classmethod
    def register_subclass(cls, name):
        def decorator(subclass):
            registry = dict(getattr(cls, "_choice_registry", {}))
            registry[name] = subclass
            cls._choice_registry = registry
            subclass._choice_registry = registry
            subclass._registered_choice_name = name
            return subclass

        return decorator

    @classmethod
    def get_choice_class(cls, name):
        return cls._choice_registry[name]


@dataclass
class _FallbackRobotConfig(_FallbackChoiceRegistry):
    @property
    def type(self) -> str:
        return self.__class__._registered_choice_name


class _FallbackRobot:
    def __init__(self, config=None):
        pass


RobotConfig = _RealRobotConfig or _FallbackRobotConfig
Robot = _RealRobot or _FallbackRobot
