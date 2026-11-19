"""rlt — pure torch RL Token components for the Panthera-HT π0.5 stack.

Ported from RL-Token-Pi05-open (github.com/Zhengsw03/RL-Token-Pi05-open) for
the openpi JAX base (see /root/xsx/RL/PLAN.md). This package has zero openpi /
lerobot imports: only torch + numpy. Dimension parameterization is
state_dim = action_dim = 7 (6 joints + gripper), RL chunk 10, VLA horizon 30.

Modules:
- networks: RLTConfig + RLTokenEncoder/Decoder, RLTChunkActor, RLTTwinCritic
- critical: CriticalPhaseClassifier / CriticalPhaseDetector (Stage 1.5)
- replay:  Transition/ReplayManager journal + stride-2 construction
- learner: perform_training_update (TD3+BC) + Stage-2 checkpoint schema
- norm:    q01/q99 quantile normalization for the absolute joint space
"""

from . import critical, learner, networks, norm, replay  # noqa: F401

__all__ = ["networks", "critical", "replay", "learner", "norm"]
