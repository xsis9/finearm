"""Deterministic offline scenarios for the Phase 5b red line (the archived refactor plan, git history
§五): rtc-mode transitions produced by the episode loop must be reproducible
field-for-field across the interface refactor.

The scenario drives the REAL train_stage2 main() with:
- DummyRobot (--dummy_robot): step-count-deterministic observations
- DeterministicVLA (patched over VLAClient): seeded per-call streams
- scripted stdin: episode reward prompts answered deterministically
- --vla_only: no async learner thread (thread interleaving would make
  byte-level comparison meaningless; the collection/boundary path — the
  BoundaryCollector seam — is what this locks, and it runs identically with
  or without the learner)
- fixed --rtc_k: kills the RTT-adaptive replan boundary

Machine-local inputs (stage-1 checkpoint, p3 stats) are env-gated:
FINEARM_STAGE1_CKPT / FINEARM_REAL_STATS_PATH. The golden test skips where
they are absent (CI); it is the pre-/post-refactor equivalence gate on the
machine that has them.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from unittest import mock

import torch

from finearm.rl.serving.client import VLAInference

REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_ARGV = {
    # --auto_critical + fixture classifier: the seeded DeterministicVLA z_rl
    # stream drives a deterministic critical phase (the fixture bias forces
    # prob≈1, so the whole episode is recorded — step_records/stride-2 only
    # fill while CRITICAL_PHASE_ACTIVE, which is exactly the boundary path
    # this golden locks).
    "rtc_basic": [
        "--rtc_mode", "--rtc_k", "4",
        "--dummy_robot", "--vla_only",
        "--max_episodes", "2", "--steps_per_episode", "40",
        "--warmup_episodes", "99",
        "--auto_critical",
        "--auto_critical_start_delay_steps", "0",
        "--auto_critical_smooth_steps", "1",
        "--auto_critical_min_on_chunks", "1",
    ],
}

SCENARIO_STDIN = {"rtc_basic": "n\nn\n"}


def make_classifier_fixture(path: Path) -> Path:
    """Always-critical classifier checkpoint (fixed seed, biased output)."""
    from finearm.rl.algorithms.rltoken.critical import CriticalPhaseClassifier

    torch.manual_seed(1234)
    # Dims must match CriticalPhaseDetector's fixed construction (hidden 256,
    # 2 layers); the checkpoint only carries z_dim/proprio_dim.
    clf = CriticalPhaseClassifier(z_dim=2048, proprio_dim=7, hidden_dim=256, num_layers=2)
    with torch.no_grad():
        head = clf.net[-1]
        if hasattr(head, "bias") and head.bias is not None:
            head.bias[1] = 10.0  # class-1 logit +10 → P(critical) ≈ 1
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "z_dim": 2048,
            "proprio_dim": 7,
            "classifier_state_dict": clf.state_dict(),
            "proprio_mean": torch.zeros(7),
            "proprio_std": torch.ones(7),
        },
        path,
    )
    return path


class DeterministicVLA:
    """Drop-in VLAClient replacement with per-call deterministic streams."""

    def __init__(self, host: str, port: int, task: str):
        from finearm.rl.algorithms.rltoken.networks import RLTConfig

        self._task = task
        self.metadata = {
            "server": "serve_rlt",
            "task_prompt": task,
            "stage1_step": None,
        }
        cfg = RLTConfig()
        self._horizon = cfg.chunk_size
        self._action_dim = cfg.action_dim
        self._g_control = torch.Generator().manual_seed(7)
        self._g_boundary = torch.Generator().manual_seed(8)
        self.calls_control = 0
        self.calls_boundary = 0

    def check_same_origin(self, *, task, stage1_step=None) -> None:
        assert self.metadata["task_prompt"] == task

    def _chunk(self, generator) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.rand(
            (1, self._horizon, self._action_dim), generator=generator
        ) * 0.2 - 0.1
        z_rl = torch.rand((2048,), generator=generator)
        return actions, z_rl

    def infer(self, obs: dict):
        self.calls_control += 1
        actions, z_rl = self._chunk(self._g_control)
        return VLAInference(actions=actions, embeddings={"z_rl": z_rl})

    def infer_boundary(self, obs: dict):
        self.calls_boundary += 1
        actions, z_rl = self._chunk(self._g_boundary)
        return VLAInference(actions=actions, embeddings={"z_rl": z_rl})


def scenario_inputs_available() -> bool:
    ckpt = os.environ.get("FINEARM_STAGE1_CKPT")
    stats = os.environ.get("FINEARM_REAL_STATS_PATH")
    return bool(ckpt and stats and Path(ckpt).is_file() and Path(stats).is_file())


def run_scenario(name: str, output_dir: Path) -> dict:
    """Run one scenario end-to-end; returns journal info for comparison."""
    import finearm.rl.train_stage2 as t2

    argv = [
        "train_stage2", *SCENARIO_ARGV[name],
        "--rlt_checkpoint", os.environ["FINEARM_STAGE1_CKPT"],
        "--stats_path", os.environ["FINEARM_REAL_STATS_PATH"],
        "--output_dir", str(output_dir),
    ]
    if name == "rtc_basic":
        argv += [
            "--critical_classifier",
            str(make_classifier_fixture(Path(output_dir).parent / "clf.pt")),
        ]

    stdin = io.StringIO(SCENARIO_STDIN[name])
    vla_instances: list[DeterministicVLA] = []

    def make_vla(host, port, task, required_embeddings=("z_rl",)):
        assert "z_rl" in required_embeddings, required_embeddings
        vla = DeterministicVLA(host, port, task)
        vla_instances.append(vla)
        return vla

    with mock.patch.object(t2, "VLAClient", make_vla), \
            mock.patch("sys.stdin", stdin), \
            mock.patch("sys.argv", argv):
        t2.main()

    journal_path = Path(output_dir) / "replay_journal.pkl"
    payload = journal_path.read_bytes()
    records = []
    with open(journal_path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
    return {
        "journal_bytes": payload,
        "records": records,
        "vla": vla_instances[0] if vla_instances else None,
    }


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / name / "journal.pkl"
