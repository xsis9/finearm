"""兼容 shim:本模块即 deploy/arm/panthera_control/cli(驱动层,2026-11 抽离)。

以 `python -m panthera_lerobot_control.cli` 运行时转发到真实入口。
"""
import sys

import panthera_control.cli as _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
sys.modules[__name__] = _real
