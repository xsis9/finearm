"""兼容 shim:Stage-2 RobotInterface 契约真身 = finearm.rl.envs.robot_interface(Phase 5a 搬移)。

import 使用时经 sys.modules 别名转发(库模块,无脚本入口);旧代码的
import 路径与下划线私有名全部照常可用。
"""

import sys
from pathlib import Path

try:
    import finearm.rl.envs.robot_interface  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[3] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    import finearm.rl.envs.robot_interface  # noqa: F401

sys.modules[__name__] = finearm.rl.envs.robot_interface
