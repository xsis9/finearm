"""兼容 shim:本模块即 deploy/arm/panthera_control/controller(驱动层,2026-11 抽离)。"""
import sys

import panthera_control.controller as _real

sys.modules[__name__] = _real
