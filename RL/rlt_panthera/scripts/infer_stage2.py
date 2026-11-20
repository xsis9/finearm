"""兼容 shim:Stage-2 部署推理真身 = finearm.rl.scripts.infer_stage2(Phase 5a 搬移)。

import 使用时经 sys.modules 别名转发:任何 import 路径拿到同一个模块对象,
全部名称(含下划线私有名)可用——旧 journal/checkpoint 的 pickle 类路径
(rlt.replay.* 等)由此保持可解析;以脚本方式运行时由守卫直接转发入口。
campaign 配置(robot_module)、历史命令、bash history 均无需改动。
"""

import sys
from pathlib import Path

try:
    import finearm.rl.scripts.infer_stage2  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[3] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    import finearm.rl.scripts.infer_stage2  # noqa: F401

_real = finearm.rl.scripts.infer_stage2

if __name__ == "__main__":
    raise SystemExit(_real.main())
else:
    sys.modules[__name__] = _real
