"""Mock adapter: deterministic local chunk generation, no server.

Usage: --mock dry-run comparisons, offline CI tests, chain smoke runs.
Chunks depend only on the call index (not wall-clock/thread timing),
keeping runs reproducible: call t returns (30,7) where step i joint j =
i/100 + t/10 (gripper halved).
"""
from . import AdapterBase


class Adapter(AdapterBase):
    NAME = "mock"
    DEFAULT_PORT = 0
    OBS_STYLE = "openpi"
    DEFAULT_HOME = "0,0,0,0,0,0,1.8"

    def __init__(self, host=None, port=None):
        self._n = 0

    def reset(self):
        self._n = 0

    def build_obs(self, obs, task):
        return {"obs": obs, "prompt": task, "call": self._n}

    def infer(self, payload):
        import numpy as np

        t = self._n
        self._n += 1
        ramp = (np.arange(30, dtype=np.float64) / 100.0 + t / 10.0)[:, None]
        chunk = np.repeat(ramp, 7, axis=1)
        chunk[:, 6] *= 0.5
        return chunk
