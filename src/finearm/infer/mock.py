"""Mock robot backend for ``--mock`` dry runs: no daemon, no cameras."""

import numpy as np

from .adapters import mock_obs


class MockRobot:
    """Fake arm (--mock): observations come from mock_obs, actions are counted."""

    def __init__(self):
        self.n = 0
        self._obs_n = 0

    def get_obs(self, style="openpi"):
        self._obs_n += 1
        return mock_obs(style, seed=self._obs_n)

    def send_action(self, a7):
        self.n += 1
        if self.n % 30 == 0:
            print(f"[mock] step {self.n}: {np.round(a7, 3)}")

    def close(self):
        print(f"[mock] 共执行 {self.n} 步")


__all__ = ["MockRobot"]
