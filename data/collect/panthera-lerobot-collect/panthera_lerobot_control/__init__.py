"""Collection-domain helpers (daemon-layer code lives in finearm.arms).

``sample_coordinator`` pairs one robot observation with the action from the
same daemon snapshot (used by the LeRobot plugins). The legacy daemon and
collector modules that used to live here moved to ``finearm.arms`` /
``finearm.data`` in the restructuring; their old import paths were removed
with the shims (git history keeps them).
"""
