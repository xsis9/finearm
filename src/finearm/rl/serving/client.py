"""serve_rlt websocket client (control + background connections).

Extracted verbatim from ``RL/rlt_panthera/scripts/train_stage2.py`` in
Phase 5a (class body unchanged; ``train_stage2`` now imports it from here).
"""

from __future__ import annotations

import numpy as np
import torch


class VLAClient:
    """Two websocket connections to serve_rlt: control path and background
    boundary materialization (never blocks control; server handles each
    connection independently)."""

    def __init__(self, host: str, port: int, task: str):
        from openpi_client import websocket_client_policy

        self._task = task
        self._policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._bg_policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self._policy.get_server_metadata()

    def check_same_origin(self, *, task: str, stage1_step: int | None = None) -> None:
        meta = self.metadata
        if meta.get("server") != "serve_rlt":
            raise ValueError(f"server metadata is not serve_rlt: {meta.get('server')!r}")
        if meta.get("task_prompt") != task:
            raise ValueError(
                f"server task {meta.get('task_prompt')!r} != config task {task!r} (四件套不一致)"
            )
        if stage1_step is not None and meta.get("stage1_step") is not None:
            if int(meta["stage1_step"]) != int(stage1_step):
                raise ValueError(
                    f"server stage1_step {meta['stage1_step']} != local checkpoint step {stage1_step}"
                )

    def _obs_payload(self, obs: dict) -> dict:
        return {
            "observation/state": np.asarray(obs["state"], dtype=np.float32),
            "observation/image": np.asarray(obs["base_image"]),
            "observation/wrist_image": np.asarray(obs["wrist_image"]),
            "prompt": self._task,
        }

    def infer(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Control path: returns (raw_actions (1,30,7) ABSOLUTE, z_rl (2048,))."""
        out = self._policy.infer(self._obs_payload(obs))
        actions = torch.as_tensor(np.asarray(out["actions"]), dtype=torch.float32).unsqueeze(0)
        z_rl = torch.as_tensor(np.asarray(out["z_rl"]), dtype=torch.float32)
        return actions, z_rl

    def infer_boundary(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Background path (same result; separate connection)."""
        out = self._bg_policy.infer(self._obs_payload(obs))
        actions = torch.as_tensor(np.asarray(out["actions"]), dtype=torch.float32).unsqueeze(0)
        z_rl = torch.as_tensor(np.asarray(out["z_rl"]), dtype=torch.float32)
        return actions, z_rl
