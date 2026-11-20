"""Stage-2 robot environments: the RobotInterface contract + backends."""

from .robot_interface import DummyRobot, RobotInterface

__all__ = ["RobotInterface", "DummyRobot"]
