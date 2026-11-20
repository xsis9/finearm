"""rltoken — pure torch RL Token components for the Panthera-HT π0.5 stack.

Moved verbatim from ``RL/rlt_panthera/rlt`` (Phase 5a; that path is now a
shim). This package has zero openpi / lerobot imports: only torch + numpy.
Dimension parameterization is state_dim = action_dim = 7 (6 joints +
gripper), RL chunk 10, VLA horizon 30.

Modules:
- networks: RLTConfig + RLTokenEncoder/Decoder, RLTChunkActor, RLTTwinCritic
- critical: CriticalPhaseClassifier / CriticalPhaseDetector (Stage 1.5)
- learner: perform_training_update (TD3+BC) + Stage-2 checkpoint schema
"""

from . import critical, learner, networks  # noqa: F401

__all__ = ["networks", "critical", "learner"]
