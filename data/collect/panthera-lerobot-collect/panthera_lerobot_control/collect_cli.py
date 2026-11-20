"""兼容 shim:采集真身 = finearm.data.collector.teleop(Phase 4 迁移)。

真身承担完整采集流:预检 → daemon 生命周期 → 相机预检 → lerobot-record
桥接 → 清理(--arm 参数化,默认行为与旧版逐字节一致,print-command 对拍锁定)。

import 使用时经 sys.modules 别名转发(任何 import 路径拿到同一个模块对象,
全部名称含下划线私有名可用);以 `python -m panthera_lerobot_control.collect_cli`
运行时由守卫直接转发入口。console 入口 `panthera-lr-collect` 照常工作。
"""

import sys
from pathlib import Path

try:
    import finearm.data.collector.teleop  # noqa: F401
except ImportError:
    _deploy_src = Path(__file__).resolve().parents[4] / "src"
    if str(_deploy_src) not in sys.path:
        sys.path.insert(0, str(_deploy_src))

import finearm.data.collector.teleop as _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
else:
    sys.modules[__name__] = _real
