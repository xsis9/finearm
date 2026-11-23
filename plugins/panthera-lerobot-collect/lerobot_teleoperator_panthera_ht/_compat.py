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


_RealTeleoperatorConfig = _import_first(
    [
        ("lerobot.teleoperators.config", "TeleoperatorConfig"),
        ("lerobot.common.teleoperators.configs", "TeleoperatorConfig"),
        ("lerobot.common.robot_devices.teleoperators.configs", "TeleoperatorConfig"),
        ("lerobot.common.teleoperators", "TeleoperatorConfig"),
    ]
)
_RealTeleoperator = _import_first(
    [
        ("lerobot.teleoperators.teleoperator", "Teleoperator"),
        ("lerobot.common.teleoperators.teleoperator", "Teleoperator"),
        ("lerobot.common.robot_devices.teleoperators.teleoperator", "Teleoperator"),
        ("lerobot.common.teleoperators", "Teleoperator"),
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
class _FallbackTeleoperatorConfig(_FallbackChoiceRegistry):
    @property
    def type(self) -> str:
        return self.__class__._registered_choice_name


class _FallbackTeleoperator:
    def __init__(self, config=None):
        pass


TeleoperatorConfig = _RealTeleoperatorConfig or _FallbackTeleoperatorConfig
Teleoperator = _RealTeleoperator or _FallbackTeleoperator
