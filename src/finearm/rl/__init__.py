"""finearm.rl — Stage-2 online RL domain (Phase 5a moved from RL/rlt_panthera).

Layout (REFACTOR_PLAN.md §一/§五):
- ``algorithms/rltoken``: pure torch RL-Token components (networks, critical,
  learner) — zero openpi/lerobot imports.
- ``replay``: journal + stride-2 transition construction.
- ``norm``: tensor wrappers over the ``finearm.core.norm`` stats contract.
- ``loop``: execution-loop pieces (chunk provider; console/runner land in 5b).
- ``serving``: serve_rlt client + server.
- ``envs``: RobotInterface contract + DummyRobot and the panthera backend.
- ``scripts``: offline pipelines (stage-1, precompute, plot, …).
- ``train_stage2``: the online-RL entry point (pre-5b shape).

Legacy import paths (``rlt.*``, ``robot.*``, ``scripts/train_stage2.py`` …)
keep working via shims in ``RL/rlt_panthera`` so campaign journals, configs
(``robot_module``), and operator history stay valid.
"""
