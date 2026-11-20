"""兼容 shim:rlt 包真身 = finearm.rl(Phase 5a 搬移)。

networks/critical/learner → finearm.rl.algorithms.rltoken;
replay → finearm.rl.replay;norm → finearm.rl.norm。子模块 shim 逐个转发,
sys.modules 别名保证 pickle 类路径(旧 journal/checkpoint)继续可解析。
"""

import sys
from pathlib import Path

try:
    import finearm.rl  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[3] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    import finearm.rl  # noqa: F401

from finearm.rl.algorithms import rltoken as _rltoken
from finearm.rl import norm as _norm
from finearm.rl import replay as _replay

critical = _rltoken.critical
learner = _rltoken.learner
networks = _rltoken.networks
norm = _norm
replay = _replay

__all__ = ["networks", "critical", "replay", "learner", "norm"]
