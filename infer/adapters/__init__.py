"""兼容 shim:适配器注册表真身 = finearm.infer.adapters(Phase 3 迁移)。

经 sys.modules 别名转发:任何 import 路径(含 `import adapters.pi05`)拿到的是
finearm 包里的同一个模块对象;bench_rtt.py / 存量脚本零改动。
"""

import sys
from pathlib import Path

try:
    import finearm.infer.adapters  # noqa: F401
except ImportError:
    _deploy_src = Path(__file__).resolve().parents[2] / "src"
    if str(_deploy_src) not in sys.path:
        sys.path.insert(0, str(_deploy_src))

import finearm.infer.adapters as _real

sys.modules[__name__] = _real
