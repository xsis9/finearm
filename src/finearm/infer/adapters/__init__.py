"""Model adapter registry + shared tools (loaded by name from the CLI / bench).

Adapter contract (read before writing a new one, see infer/README.md):

  NAME          adapter name (= filename; selected via --model)
  DEFAULT_PORT  default inference-server port
  OBS_STYLE     observation style, decides how the robot backend builds obs:
                  "openpi" → get_obs("openpi"): 224×224 pre-resized images + first 7 state dims
                  "raw"    → get_obs("raw"): raw 640×480 frames + full 14-dim state
  DEFAULT_HOME  pre-inference home pose ("j1,...,j6,gripper", 7 values; "" = none).
                Priority: finearm-infer --home > PANTHERA_RTC_HOME > this default

  __init__(host, port=None)   connect to the inference server (port defaults to DEFAULT_PORT)
  reset()                     post-connect handshake (pass if unneeded; lingbot sends robo_name)
  build_obs(obs, task)        get_obs(OBS_STYLE) result → this model's request dict
  infer(payload) -> (T,7)     send request, parse reply into absolute joint-target chunks
                              (6 joints + gripper, rows = timesteps; value validation lives here)

Adapter modules must NOT connect at import time (registry scans / unit tests
import without instantiating).
"""

import importlib

from ...core.registry import RegistryError, scan_names


class AdapterBase:
    """Adapter base class: interface stubs + common defaults.

    Subclasses must override NAME/DEFAULT_PORT/OBS_STYLE and
    __init__/build_obs/infer; reset as needed.
    """

    NAME = "?"
    DEFAULT_PORT = 0
    OBS_STYLE = "openpi"
    DEFAULT_HOME = ""

    def __init__(self, host, port=None):
        raise NotImplementedError

    def reset(self):
        pass

    def build_obs(self, obs, task):
        raise NotImplementedError

    def infer(self, payload):
        raise NotImplementedError


def available():
    """Registered adapter names (module names under this package, excluding _-prefixed)."""
    return scan_names(__package__)


def load(name, host, port=None):
    """Load and instantiate an adapter by name (connects to the server here)."""
    if name not in available():
        raise SystemExit(
            f"未知适配器 {name!r};可用: {', '.join(available())}"
            f"(在 finearm/infer/adapters/ 目录下,新增适配器见 infer/README.md)"
        )
    mod = importlib.import_module(f"{__package__}.{name}")
    return mod.Adapter(host=host, port=port)


def mock_obs(style, seed=0):
    """Deterministic fake observations (--mock dry runs / bench_rtt; no hardware)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    if style == "openpi":
        img = lambda: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)  # noqa: E731
        return {
            "observation/image": img(),
            "observation/wrist_image": img(),
            "observation/state": np.array([0.05, 0.3, 1.2, -0.5, 0.0, 0.1, 0.8], np.float32),
        }
    if style == "raw":
        img = lambda: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)  # noqa: E731
        return {
            "front": img(),
            "wrist": img(),
            "state": np.concatenate([
                np.array([0.05, 0.3, 1.2, -0.5, 0.0, 0.1, 0.8], np.float32),
                np.zeros(7, np.float32)]),
        }
    raise ValueError(f"未知 style: {style}(openpi | raw)")


__all__ = ["AdapterBase", "RegistryError", "available", "load", "mock_obs"]
