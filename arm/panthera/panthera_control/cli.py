"""兼容 shim:CLI 真身 = finearm.arms.daemon.cli(Phase 2)。

import 使用时经 sys.modules 别名转发(build_parser/plan_park_actions/main
及测试 patch 的 signal/time 属性都在真身模块上);以
`python -m panthera_control.cli` 运行时由守卫直接转发入口。
默认 --arm panthera_ht,--help 的 prog 名保持旧值。
"""

import sys

import finearm.arms.daemon.cli as _real

if __name__ == "__main__":
    raise SystemExit(_real.main(prog="panthera-control-daemon", default_arm="panthera_ht"))
else:
    sys.modules[__name__] = _real
