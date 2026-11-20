"""Phase 5a shim contracts: legacy import paths alias the finearm.rl modules.

Campaign continuity depends on this: journals and checkpoints pickle classes
by module path (``rlt.replay.Transition`` …); unpickling imports the shim,
which must yield the *same* class object the new code uses. A failure here
means a resume-from-journal or checkpoint-restore regression, not a test
flake.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import paths (src/, RL/rlt_panthera, its scripts/) come from tests/rl/conftest.py.

import rlt.critical  # noqa: E402
import rlt.learner  # noqa: E402
import rlt.networks  # noqa: E402
import rlt.norm  # noqa: E402
import rlt.replay  # noqa: E402
import robot.panthera_robot  # noqa: E402
import robot.robot_interface  # noqa: E402
import train_stage2  # noqa: E402


def test_rlt_module_identity() -> None:
    from finearm.rl import norm as new_norm
    from finearm.rl import replay as new_replay
    from finearm.rl.algorithms.rltoken import (
        critical as new_critical,
        learner as new_learner,
        networks as new_networks,
    )

    assert rlt.networks is new_networks
    assert rlt.critical is new_critical
    assert rlt.learner is new_learner
    assert rlt.replay is new_replay
    assert rlt.norm is new_norm


def test_pickled_class_identity() -> None:
    """Pickle round-trips through the legacy path resolve to the same class."""
    import pickle

    from finearm.rl import replay as new_replay

    for cls in (rlt.replay.Transition, rlt.replay.ExecutionWindow, rlt.replay.StepRecord):
        assert cls is getattr(new_replay, cls.__name__)
        restored = pickle.loads(pickle.dumps(cls))
        assert restored is cls


def test_robot_and_script_identity() -> None:
    from finearm.rl import train_stage2 as new_t2
    from finearm.rl.envs import robot_interface as new_contract
    from finearm.rl.envs.panthera import robot as new_panthera

    assert robot.robot_interface is new_contract
    assert robot.panthera_robot is new_panthera
    assert train_stage2 is new_t2
    # Names the console/loop code is known to reach through the old path:
    for name in ("ConsoleInputManager", "PipelinedChunkProvider", "VLAClient",
                 "load_robot", "PANTHERA_JOINT_NAMES", "signal_handler"):
        assert hasattr(train_stage2, name), name


def test_joint_names_single_sourced() -> None:
    """PANTHERA_JOINT_NAMES derives from the panthera_ht ArmProfile."""
    from finearm.arms.panthera_ht import PROFILE

    expected = tuple(
        n[: -len(".pos")] if n.endswith(".pos") else n
        for n in PROFILE.action_feature_names
    )
    assert rlt.replay.PANTHERA_JOINT_NAMES == expected
