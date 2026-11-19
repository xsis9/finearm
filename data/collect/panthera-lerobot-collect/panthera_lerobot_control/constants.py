"""兼容 shim:本模块即 deploy/arm/panthera_control/constants(驱动层,2026-11 抽离)。

经 sys.modules 别名转发:任何 import 路径拿到的是同一个模块对象,
全部名称(含测试使用的下划线私有名)都可用。
"""
import sys

import panthera_control.constants as _real

sys.modules[__name__] = _real
