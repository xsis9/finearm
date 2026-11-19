"""LingBot-VLA 2.0 适配器。

服务端(4090):lingbot-vla-v2 仓库 `python -m deploy.lingbot_vla_v2_policy`
(PyTorch/bf16),默认端口 8006。观测发**数据集原始特征名**(14 维 state +
原始 640×480 帧,服务端自己 resize 到 256×256);返回按 robot config 的
原始动作名分 key,这里拼回 (T,7)。服务端启动命令与关键坑见
../README.md「lingbot-vla-2 关键信息」。
"""
import pathlib
import sys

import numpy as np

from . import AdapterBase

# lingbot-vla-v2 仓库(deploy 自包含:adapters/ 的上两级是 deploy/)
_REPO = pathlib.Path(__file__).resolve().parents[2] / "models" / "lingbot-vla-v2"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class Adapter(AdapterBase):
    NAME = "lingbot"
    DEFAULT_PORT = 8006
    OBS_STYLE = "raw"                          # 原始帧 + 14 维 state;resize 在服务端
    # 模型 action chunk = 50 步(训练 horizon;pi05 是 30)。服务端 --use_length 会
    # 截断返回(当前 25);RTC 建议起服务时 --use_length 50 用满 chunk(切换更少、
    # 抗 RTT 抖动)。执行器对 chunk 长度无假设,k 上限自动随实际长度适配。
    DEFAULT_HOME = "0,0,0,0,0,0,1.5576"        # 微调训练起始位姿≈全零(2026-11-11 按训练数据修正)

    def __init__(self, host, port=None):
        from deploy.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=host, port=port or self.DEFAULT_PORT)

    def reset(self):
        # 服务端按 robo_name 加载 configs/robot_configs/panthera.yaml
        # 并构建 FeatureTransform(subtract_state 逆变换在服务端做)
        self.client.reset("panthera")

    def build_obs(self, obs, task):
        return {
            "observation.state": obs["state"],          # 14 维,daemon snapshot 原样
            "observation.images.front": obs["front"],   # HWC uint8 RGB 原始帧
            "observation.images.wrist": obs["wrist"],
            "task": task,
        }

    def infer(self, payload):
        res = self.client.infer(payload)
        # 返回 key = 服务端 robot config 里的动作名,两种风格都兼容:
        #   单块式(4090 训练 config,2026-11-11 实测):{"action": (T,7)} 直接用
        #   拆分式(AU 参考副本 config):arm/effector 分 key,拼回 (T,7)
        if "action" in res:
            chunk = np.asarray(res["action"], dtype=np.float64)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
        else:
            arm = np.asarray(res["action.arm.position"], dtype=np.float64).reshape(-1)
            eff = np.asarray(res["action.effector.position"], dtype=np.float64).reshape(-1)
            t_len = min(arm.size // 6, eff.size)
            if t_len < 1:
                raise RuntimeError(f"action 返回异常: keys={list(res)} arm={arm.shape} eff={eff.shape}")
            chunk = np.concatenate(
                [arm[: t_len * 6].reshape(t_len, 6), eff[:t_len].reshape(t_len, 1)], axis=1)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < 1:
            raise RuntimeError(f"action 返回异常形状: {chunk.shape}(keys={list(res)})")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError(f"action 数值异常: {chunk}")
        return chunk
