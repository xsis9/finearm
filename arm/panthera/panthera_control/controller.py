"""兼容 shim:控制环真身 = finearm.arms.daemon.controller(Phase 2)。

经 sys.modules 别名转发:任何 import 路径拿到的是同一个模块对象,
全部名称(含旧异常名 PantheraControlError,本体已带别名)都可用。
"""

import sys

import finearm.arms.daemon.controller as _real

sys.modules[__name__] = _real
