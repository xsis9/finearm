"""兼容 shim:采集 RobotInterface 真身 = finearm.infer.robot_interface(Phase 5a)。

import 使用时经 sys.modules 别名转发(任何 import 路径拿到同一个模块对象,
全部名称含下划线私有名可用);`python infer/robot_interface.py --selftest`
自检入口由守卫等价转发。panthera_robot(PANTHERA_ACQ_IFACE/local.jsonc
覆盖)与 finearm-infer --arm 默认值均继续经本文件或包内真身工作。
"""

import sys
from pathlib import Path

try:
    import finearm.infer.robot_interface  # noqa: F401
except ImportError:
    _src = Path(__file__).resolve().parents[1] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import finearm.infer.robot_interface  # noqa: F401

_real = finearm.infer.robot_interface

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="RobotInterface 自检(不动臂)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dump-obs", default=None, help="目录:存一张 front/wrist 图肉眼核对")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("请用 --selftest 运行(自检模式)")
    _real._selftest(a)
else:
    sys.modules[__name__] = _real
