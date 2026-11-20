"""serve_rlt websocket client (control + background connections).

Extracted verbatim from ``RL/rlt_panthera/scripts/train_stage2.py`` in
Phase 5a (class body unchanged; ``train_stage2`` now imports it from here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VLAInference:
    """One serve response: actions + algorithm-declared embeddings.

    The loop/framework treats ``embeddings`` as an opaque mapping (it never
    names keys); the algorithm side fetches what it declared via
    ``REQUIRED_EMBEDDINGS``. Server-side assembly per declaration is 5c.
    """

    actions: torch.Tensor                 # (1, horizon, action_dim) ABSOLUTE
    embeddings: dict[str, torch.Tensor]


class VLAClient:
    """Two websocket connections to serve_rlt: control path and background
    boundary materialization (never blocks control; server handles each
    connection independently)."""

    def __init__(self, host: str, port: int, task: str,
                 required_embeddings: tuple[str, ...] = ("z_rl",)):
        from openpi_client import websocket_client_policy

        self._task = task
        self._required_embeddings = tuple(required_embeddings)
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

    def _wrap(self, out: dict) -> VLAInference:
        """Wire dict → VLAInference; every non-action key becomes an embedding.

        Validates the algorithm-declared keys (handshake against the actual
        payload — until 5c, serve_rlt metadata does not yet declare them).
        """
        actions = torch.as_tensor(np.asarray(out["actions"]), dtype=torch.float32).unsqueeze(0)
        embeddings = {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32)
            for key, value in out.items() if key != "actions"
        }
        missing = [k for k in self._required_embeddings if k not in embeddings]
        if missing:
            raise ValueError(
                f"serve response is missing required embeddings {missing} "
                f"(got {sorted(embeddings)}); algorithm/server mismatch"
            )
        return VLAInference(actions=actions, embeddings=embeddings)

    def infer(self, obs: dict) -> VLAInference:
        """Control path: raw actions (1,30,7) ABSOLUTE + embeddings dict."""
        return self._wrap(self._policy.infer(self._obs_payload(obs)))

    def infer_boundary(self, obs: dict) -> VLAInference:
        """Background path (same result; separate connection)."""
        return self._wrap(self._bg_policy.infer(self._obs_payload(obs)))
