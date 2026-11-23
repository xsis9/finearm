import dataclasses
import importlib
import sys
import time
import tomllib
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

from finearm.arms.daemon import DaemonDisconnectedError, StaleSnapshotError


OBSERVATION_KEYS = [
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
]

ACTION_KEYS = [
    "target_joint_1.pos",
    "target_joint_2.pos",
    "target_joint_3.pos",
    "target_joint_4.pos",
    "target_joint_5.pos",
    "target_joint_6.pos",
    "target_gripper.pos",
]


class LeRobotPantheraPluginTests(unittest.TestCase):
    def test_robot_config_declares_panthera_follower_type(self):
        from lerobot_robot_panthera_ht import PantheraHTRobotConfig

        self.assertEqual(PantheraHTRobotConfig().type, "panthera_ht_follower")
        self.assert_registered_type(PantheraHTRobotConfig, "panthera_ht_follower")
        self.assertNotIn("type", {field.name for field in dataclasses.fields(PantheraHTRobotConfig)})

    def test_teleoperator_config_declares_panthera_leader_type(self):
        from lerobot_teleoperator_panthera_ht import PantheraHTTeleoperatorConfig

        self.assertEqual(PantheraHTTeleoperatorConfig().type, "panthera_ht_leader")
        self.assert_registered_type(PantheraHTTeleoperatorConfig, "panthera_ht_leader")
        self.assertNotIn(
            "type",
            {field.name for field in dataclasses.fields(PantheraHTTeleoperatorConfig)},
        )

    def test_robot_compat_prefers_current_upstream_lerobot_paths(self):
        fake_config = make_fake_choice_config("RobotConfig")

        class FakeRobot:
            def __init__(self, config):
                self.base_init_config = config
                self.base_init_called = True

        modules = {
            "lerobot": types.ModuleType("lerobot"),
            "lerobot.robots": types.ModuleType("lerobot.robots"),
            "lerobot.robots.config": module_with(RobotConfig=fake_config),
            "lerobot.robots.robot": module_with(Robot=FakeRobot),
        }

        with isolated_modules(
            modules,
            "lerobot_robot_panthera_ht",
            "lerobot_robot_panthera_ht._compat",
        ):
            plugin = importlib.import_module("lerobot_robot_panthera_ht")

            self.assertTrue(issubclass(plugin.PantheraHTRobotConfig, fake_config))
            self.assertTrue(issubclass(plugin.PantheraHTRobot, FakeRobot))
            self.assertIs(
                fake_config.get_choice_class("panthera_ht_follower"),
                plugin.PantheraHTRobotConfig,
            )
            self.assertEqual(plugin.PantheraHTRobotConfig().type, "panthera_ht_follower")
            self.assertNotIn(
                "type",
                {field.name for field in dataclasses.fields(plugin.PantheraHTRobotConfig)},
            )
            config = plugin.PantheraHTRobotConfig()
            robot = plugin.PantheraHTRobot(config, client=FakeClient())
            self.assertIs(robot.base_init_config, config)
            self.assertTrue(robot.base_init_called)

    def test_teleoperator_compat_prefers_current_upstream_lerobot_paths(self):
        fake_config = make_fake_choice_config("TeleoperatorConfig")

        class FakeTeleoperator:
            def __init__(self, config):
                self.base_init_config = config
                self.base_init_called = True

        modules = {
            "lerobot": types.ModuleType("lerobot"),
            "lerobot.teleoperators": types.ModuleType("lerobot.teleoperators"),
            "lerobot.teleoperators.config": module_with(TeleoperatorConfig=fake_config),
            "lerobot.teleoperators.teleoperator": module_with(Teleoperator=FakeTeleoperator),
        }

        with isolated_modules(
            modules,
            "lerobot_teleoperator_panthera_ht",
            "lerobot_teleoperator_panthera_ht._compat",
        ):
            plugin = importlib.import_module("lerobot_teleoperator_panthera_ht")

            self.assertTrue(issubclass(plugin.PantheraHTTeleoperatorConfig, fake_config))
            self.assertTrue(issubclass(plugin.PantheraHTTeleoperator, FakeTeleoperator))
            self.assertIs(
                fake_config.get_choice_class("panthera_ht_leader"),
                plugin.PantheraHTTeleoperatorConfig,
            )
            self.assertEqual(plugin.PantheraHTTeleoperatorConfig().type, "panthera_ht_leader")
            self.assertNotIn(
                "type",
                {field.name for field in dataclasses.fields(plugin.PantheraHTTeleoperatorConfig)},
            )
            config = plugin.PantheraHTTeleoperatorConfig()
            teleop = plugin.PantheraHTTeleoperator(config, client=FakeClient())
            self.assertIs(teleop.base_init_config, config)
            self.assertTrue(teleop.base_init_called)

    def test_pyproject_name_uses_lerobot_plugin_prefix(self):
        project_root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

        self.assertTrue(
            metadata["project"]["name"].startswith(("lerobot_robot_", "lerobot-robot-"))
        )

    def test_importing_robot_plugin_also_registers_teleoperator_plugin(self):
        with isolated_modules(
            {},
            "lerobot_robot_panthera_ht",
            "lerobot_robot_panthera_ht._compat",
            "lerobot_teleoperator_panthera_ht",
            "lerobot_teleoperator_panthera_ht._compat",
        ):
            importlib.import_module("lerobot_robot_panthera_ht")
            self.assertIn("lerobot_teleoperator_panthera_ht", sys.modules)
            teleop = importlib.import_module("lerobot_teleoperator_panthera_ht")

            self.assertEqual(teleop.PantheraHTTeleoperatorConfig().type, "panthera_ht_leader")
            self.assert_registered_type(
                teleop.PantheraHTTeleoperatorConfig,
                "panthera_ht_leader",
            )

    def test_robot_config_decodes_lerobot_camera_configs_with_draccus(self):
        import draccus
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot_robot_panthera_ht import PantheraHTRobotConfig

        config = draccus.decode(
            PantheraHTRobotConfig,
            {
                "cameras": {
                    "front": {
                        "type": "opencv",
                        "index_or_path": "/dev/video0",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                },
            },
        )

        self.assertIsInstance(config.cameras["front"], OpenCVCameraConfig)
        self.assertEqual(config.cameras["front"].index_or_path, Path("/dev/video0"))
        self.assertEqual(config.cameras["front"].width, 640)
        self.assertEqual(config.cameras["front"].height, 480)
        self.assertEqual(config.cameras["front"].fps, 30)

    def test_robot_config_accepts_external_camera_configs_without_opening_devices(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        fake_camera = object()
        cameras = {"front": fake_camera}
        config = PantheraHTRobotConfig(cameras=cameras)
        robot = PantheraHTRobot(config, client=FakeClient())

        self.assertIs(config.cameras, cameras)
        self.assertIs(robot.cameras, cameras)
        self.assertIs(robot.cameras["front"], fake_camera)

    def test_robot_exposes_flat_float_observation_and_action_features(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        robot = PantheraHTRobot(PantheraHTRobotConfig(), client=FakeClient())

        self.assertEqual(robot.cameras, {})
        self.assertEqual(list(robot.observation_features), OBSERVATION_KEYS)
        self.assertEqual(list(robot.action_features), ACTION_KEYS)
        self.assertEqual(robot.observation_features, {key: float for key in OBSERVATION_KEYS})
        self.assertEqual(robot.action_features, {key: float for key in ACTION_KEYS})

    def test_teleoperator_exposes_flat_float_action_features_and_empty_feedback(self):
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        teleop = PantheraHTTeleoperator(PantheraHTTeleoperatorConfig(), client=FakeClient())

        self.assertEqual(list(teleop.action_features), ACTION_KEYS)
        self.assertEqual(teleop.action_features, {key: float for key in ACTION_KEYS})
        self.assertEqual(teleop.feedback_features, {})

    def test_robot_get_observation_maps_daemon_state_to_named_keys(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        client = FakeClient(state=[float(i) for i in range(14)])
        robot = PantheraHTRobot(PantheraHTRobotConfig(stale_timeout_s=0.25), client=client)
        robot.connect()

        self.assertEqual(
            robot.get_observation(),
            {key: float(i) for i, key in enumerate(OBSERVATION_KEYS)},
        )
        self.assertEqual(client.max_age_s_values, [0.25])

    def test_robot_get_observation_includes_camera_frames(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        frame = object()
        camera = FakeCamera(frame)
        config = PantheraHTRobotConfig(cameras={"front": camera}, stale_timeout_s=0.25)
        robot = PantheraHTRobot(config, client=FakeClient(state=[float(i) for i in range(14)]))

        self.assertIn("front", robot.observation_features)
        self.assertEqual(robot.observation_features["front"], (480, 640, 3))

        robot.connect()
        observation = robot.get_observation()

        self.assertIs(observation["front"], frame)
        self.assertTrue(camera.connected)
        self.assertEqual(camera.reads, 1)

        robot.disconnect()
        self.assertFalse(camera.connected)

    def test_teleoperator_get_action_maps_daemon_action_to_named_keys(self):
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        client = FakeClient(action=[10.0 + i for i in range(7)])
        teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(stale_timeout_s=0.5),
            client=client,
        )
        teleop.connect()

        self.assertEqual(
            teleop.get_action(),
            {key: 10.0 + i for i, key in enumerate(ACTION_KEYS)},
        )
        self.assertEqual(client.max_age_s_values, [0.5])

    def test_robot_observation_and_teleoperator_action_share_one_daemon_snapshot(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        client = SequencedFakeClient()
        socket_path = "/tmp/panthera-paired-test.sock"
        robot = PantheraHTRobot(
            PantheraHTRobotConfig(socket_path=socket_path, stale_timeout_s=0.25),
            client=client,
        )
        teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(socket_path=socket_path, stale_timeout_s=0.25),
            client=client,
        )
        robot.connect()
        teleop.connect()

        observation = robot.get_observation()
        action = teleop.get_action()

        self.assertEqual(client.snapshot_calls, 1)
        self.assertEqual(observation[OBSERVATION_KEYS[0]], 1.0)
        self.assertEqual(action[ACTION_KEYS[0]], 101.0)

        next_action = teleop.get_action()

        self.assertEqual(client.snapshot_calls, 2)
        self.assertEqual(next_action[ACTION_KEYS[0]], 102.0)

    def test_plugins_reject_non_finite_observation_action_and_send_action(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        robot = PantheraHTRobot(
            PantheraHTRobotConfig(socket_path="/tmp/panthera-finite-robot.sock"),
            client=FakeClient(state=[float("nan")] + [0.0] * 13),
        )
        robot.connect()
        with self.assertRaises(ValueError):
            robot.get_observation()

        teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(socket_path="/tmp/panthera-finite-teleop.sock"),
            client=FakeClient(action=[float("inf")] + [0.0] * 6),
        )
        teleop.connect()
        with self.assertRaises(ValueError):
            teleop.get_action()

        policy_client = FakeClient()
        policy_robot = PantheraHTRobot(
            PantheraHTRobotConfig(send_action_mode="policy_ipc"),
            client=policy_client,
        )
        policy_robot.connect()
        invalid_action = {key: float(index) for index, key in enumerate(ACTION_KEYS)}
        invalid_action[ACTION_KEYS[0]] = float("-inf")
        with self.assertRaises(ValueError):
            policy_robot.send_action(invalid_action)
        self.assertEqual(policy_client.policy_actions, [])

    def test_plugins_stop_collection_when_daemon_snapshot_reports_latched_error(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        diagnostics = {"last_error": "RuntimeError('follower read failed')", "holding": True}
        robot = PantheraHTRobot(
            PantheraHTRobotConfig(socket_path="/tmp/panthera-error-robot.sock"),
            client=FakeClient(diagnostics=diagnostics),
        )
        robot.connect()
        with self.assertRaisesRegex(RuntimeError, "follower read failed"):
            robot.get_observation()

        teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(socket_path="/tmp/panthera-error-teleop.sock"),
            client=FakeClient(diagnostics=diagnostics),
        )
        teleop.connect()
        with self.assertRaisesRegex(RuntimeError, "follower read failed"):
            teleop.get_action()

    def test_robot_send_action_returns_exact_input_unchanged_and_validates_keys(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        robot = PantheraHTRobot(PantheraHTRobotConfig(), client=FakeClient())
        action = {key: float(i) for i, key in enumerate(ACTION_KEYS)}

        self.assertIs(robot.send_action(action), action)

        missing = dict(action)
        missing.pop(ACTION_KEYS[-1])
        with self.assertRaises(ValueError):
            robot.send_action(missing)

        extra = dict(action)
        extra["unexpected"] = 0.0
        with self.assertRaises(ValueError):
            robot.send_action(extra)

        wrong_type = dict(action)
        wrong_type[ACTION_KEYS[0]] = "0.0"
        with self.assertRaises(TypeError):
            robot.send_action(wrong_type)

    def test_robot_send_action_can_forward_policy_action_over_ipc(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig

        client = FakeClient()
        robot = PantheraHTRobot(
            PantheraHTRobotConfig(send_action_mode="policy_ipc"),
            client=client,
        )
        robot.connect()
        action = {key: float(i) for i, key in enumerate(ACTION_KEYS)}

        self.assertIs(robot.send_action(action), action)

        self.assertEqual(client.policy_actions, [[float(i) for i in range(7)]])

    def test_robot_and_teleoperator_surface_stale_and_disconnect_errors(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        stale_robot = PantheraHTRobot(
            PantheraHTRobotConfig(stale_timeout_s=0.01),
            client=FakeClient(exc=StaleSnapshotError("snapshot is stale")),
        )
        stale_robot.connect()
        with self.assertRaises(StaleSnapshotError):
            stale_robot.get_observation()

        disconnected_teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(stale_timeout_s=0.01),
            client=FakeClient(exc=DaemonDisconnectedError("disconnected")),
        )
        disconnected_teleop.connect()
        with self.assertRaises(DaemonDisconnectedError):
            disconnected_teleop.get_action()

    def test_robot_and_teleoperator_reject_reads_while_disconnected(self):
        from lerobot_robot_panthera_ht import PantheraHTRobot, PantheraHTRobotConfig
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        robot = PantheraHTRobot(PantheraHTRobotConfig(), client=FakeClient())
        with self.assertRaises(RuntimeError):
            robot.get_observation()
        robot.connect()
        robot.disconnect()
        with self.assertRaises(RuntimeError):
            robot.get_observation()

        teleop = PantheraHTTeleoperator(PantheraHTTeleoperatorConfig(), client=FakeClient())
        with self.assertRaises(RuntimeError):
            teleop.get_action()
        teleop.connect()
        teleop.disconnect()
        with self.assertRaises(RuntimeError):
            teleop.get_action()

    def test_disconnected_teleoperator_does_not_consume_paired_snapshot(self):
        from panthera_lerobot_control.sample_coordinator import (
            clear_snapshot,
            consume_snapshot,
            publish_snapshot,
        )
        from lerobot_teleoperator_panthera_ht import (
            PantheraHTTeleoperator,
            PantheraHTTeleoperatorConfig,
        )

        socket_path = "/tmp/panthera-disconnected-paired-test.sock"
        publish_snapshot(socket_path, FakeClient().snapshot())
        paired_teleop = PantheraHTTeleoperator(
            PantheraHTTeleoperatorConfig(socket_path=socket_path),
            client=FakeClient(),
        )
        try:
            with self.assertRaises(RuntimeError):
                paired_teleop.get_action()
            self.assertIsNotNone(consume_snapshot(socket_path))
        finally:
            clear_snapshot(socket_path)

    def assert_registered_type(self, config_cls, expected_type):
        registry = getattr(config_cls, "_choice_registry", None)
        if isinstance(registry, dict):
            self.assertIs(registry[expected_type], config_cls)
            return

        for attr in ("get_choice_class", "get_subclass", "by_name"):
            lookup = getattr(config_cls, attr, None)
            if callable(lookup):
                try:
                    self.assertIs(lookup(expected_type), config_cls)
                    return
                except Exception:
                    pass

        self.assertEqual(config_cls().type, expected_type)


class FakeCamera:
    height = 480
    width = 640
    fps = 30

    def __init__(self, frame):
        self.frame = frame
        self.connected = False
        self.reads = 0

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def read_latest(self):
        self.reads += 1
        return self.frame

    def disconnect(self):
        self.connected = False


class FakeClient:
    def __init__(self, state=None, action=None, exc=None, diagnostics=None):
        self.state = list(state) if state is not None else [0.0] * 14
        self.action = list(action) if action is not None else [0.0] * 7
        self.exc = exc
        self.diagnostics = dict(diagnostics or {})
        self.max_age_s_values = []
        self.policy_actions = []

    def send_policy_action(self, action):
        self.policy_actions.append(list(action))
        return {"policy_sequence": len(self.policy_actions)}

    def snapshot(self, max_age_s=None):
        self.max_age_s_values.append(max_age_s)
        if self.exc is not None:
            raise self.exc
        return {
            "observation.state": self.state,
            "action": self.action,
            "timestamp": time.monotonic(),
            "sequence": 1,
            "diagnostics": self.diagnostics,
            "loop_timing": {},
        }


class SequencedFakeClient:
    def __init__(self):
        self.snapshot_calls = 0

    def snapshot(self, max_age_s=None):
        self.snapshot_calls += 1
        sequence = self.snapshot_calls
        return {
            "observation.state": [float(sequence)] * 14,
            "action": [float(100 + sequence)] * 7,
            "timestamp": time.monotonic(),
            "sequence": sequence,
            "diagnostics": {},
            "loop_timing": {},
        }

    def send_policy_action(self, action):
        raise AssertionError("policy action should not be used in snapshot pairing test")


def module_with(**attrs):
    module = types.ModuleType("fake_lerobot_module")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def make_fake_choice_config(name):
    class FakeChoiceConfig:
        _choice_registry = {}

        @classmethod
        def register_subclass(cls, choice_name):
            def decorator(subclass):
                cls._choice_registry[choice_name] = subclass
                subclass._registered_choice_name = choice_name
                return subclass

            return decorator

        @classmethod
        def get_choice_class(cls, choice_name):
            return cls._choice_registry[choice_name]

        @property
        def type(self):
            return self.__class__._registered_choice_name

    FakeChoiceConfig.__name__ = name
    return FakeChoiceConfig


@contextmanager
def isolated_modules(fake_modules, *module_names):
    saved = {}
    prefixes = tuple(name + "." for name in module_names)
    for name in list(sys.modules):
        if name in module_names or name.startswith(prefixes) or name in fake_modules:
            saved[name] = sys.modules.pop(name)
    sys.modules.update(fake_modules)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name in module_names or name.startswith(prefixes) or name in fake_modules:
                sys.modules.pop(name)
        sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
