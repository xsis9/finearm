"""pi0.5 / openpi adapter.

Server (4090): openpi repo `scripts/serve_policy.py` (JAX), default port 8000.
Observations use openpi-style keys; the reply's single "actions" key IS the (T,7) chunk.
"""
import numpy as np

from . import AdapterBase


class Adapter(AdapterBase):
    NAME = "pi05"
    DEFAULT_PORT = 8000
    OBS_STYLE = "openpi"                       # 224 pre-resized + first 7 state dims
    # model action chunk = 30 steps (≈1s @30Hz; lingbot is 50)
    DEFAULT_HOME = "0,0,0,0,0,0,1.8"           # p3 training start pose ≈ all zeros

    def __init__(self, host, port=None):
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port or self.DEFAULT_PORT)

    def reset(self):
        pass  # openpi server needs no handshake

    def build_obs(self, obs, task):
        d = dict(obs)
        d["prompt"] = task
        return d

    def infer(self, payload):
        return np.asarray(self.client.infer(payload)["actions"], dtype=np.float64)
