"""pi0.5 / openpi 适配器。

服务端(4090):openpi 仓库 `scripts/serve_policy.py`(JAX),默认端口 8000。
obs 用 openpi 约定键;返回单键 "actions" 直接是 (T,7)。
"""
import numpy as np

from . import AdapterBase


class Adapter(AdapterBase):
    NAME = "pi05"
    DEFAULT_PORT = 8000
    OBS_STYLE = "openpi"                       # 224 预 resize + state 前 7 维
    # 模型 action chunk = 30 步(≈1s @30Hz;lingbot 是 50)
    DEFAULT_HOME = "0,0,0,0,0,0,1.8"           # p3 训练起始位姿≈全零

    def __init__(self, host, port=None):
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port or self.DEFAULT_PORT)

    def reset(self):
        pass  # openpi 服务端无需握手

    def build_obs(self, obs, task):
        d = dict(obs)
        d["prompt"] = task
        return d

    def infer(self, payload):
        return np.asarray(self.client.infer(payload)["actions"], dtype=np.float64)
