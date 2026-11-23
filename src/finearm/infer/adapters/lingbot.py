"""LingBot-VLA 2.0 adapter.

Server (4090): lingbot-vla-v2 repo `python -m deploy.lingbot_vla_v2_policy`
(PyTorch/bf16), default port 8006. Observations use the dataset's raw feature
names (14-dim state + raw 640×480 frames; the server resizes to 256×256);
replies come back keyed by the robot config's raw action names and are
reassembled here into (T,7). Server startup and pitfalls: envs/infer/README.md.
"""
import sys

import numpy as np

from . import AdapterBase


def _lingbot_repo_root():
    """lingbot-vla-v2 checkout location: <repo>/models/lingbot-vla-v2 (gitignored).

    Resolved through finearm.core (works from any clone location; FINEARM_ROOT
    env override applies). Raises with guidance when the checkout is missing.
    """
    from ...core import paths as core_paths

    repo = core_paths.find_repo_root()
    if repo is None:
        raise RuntimeError(
            "lingbot adapter: finearm repo root not found and no lingbot path is importable; "
            "run from a checkout or set FINEARM_ROOT"
        )
    lingbot = repo / "models" / "lingbot-vla-v2"
    if not lingbot.is_dir():
        raise RuntimeError(
            f"lingbot-vla-v2 checkout not found at {lingbot} "
            "(gitignored; see fetch_models instructions in envs/infer/README.md)"
        )
    return lingbot


_REPO = _lingbot_repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class Adapter(AdapterBase):
    NAME = "lingbot"
    DEFAULT_PORT = 8006
    OBS_STYLE = "raw"                          # raw frames + 14-dim state; resize server-side
    # model action chunk = 50 steps (training horizon; pi05 is 30). The server's
    # --use_length truncates the reply (currently 25); for RTC prefer starting the
    # server with --use_length 50 to use the full chunk (fewer switches, more RTT
    # jitter tolerance). Executors make no chunk-length assumption; the k cap
    # adapts to the actual length.
    DEFAULT_HOME = "0,0,0,0,0,0,1.5576"        # finetune start pose ≈ all zeros (2026-11-11, per training data)

    def __init__(self, host, port=None):
        from deploy.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=host, port=port or self.DEFAULT_PORT)

    def reset(self):
        # server loads configs/robot_configs/panthera.yaml by robo_name and
        # builds FeatureTransform (inverse subtract_state happens server-side)
        self.client.reset("panthera")

    def build_obs(self, obs, task):
        return {
            "observation.state": obs["state"],          # 14 dims, daemon snapshot as-is
            "observation.images.front": obs["front"],   # HWC uint8 RGB raw frame
            "observation.images.wrist": obs["wrist"],
            "task": task,
        }

    def infer(self, payload):
        res = self.client.infer(payload)
        # reply keys = the server robot config's action names; both styles work:
        #   single-block (4090 training config, verified 2026-11-11): {"action": (T,7)} used directly
        #   split (AU reference copy config): arm/effector keyed separately, reassembled
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
