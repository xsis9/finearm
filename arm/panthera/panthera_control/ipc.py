"""兼容 shim:IPC 真身 = finearm.arms.daemon.ipc(Phase 2)。

经 sys.modules 别名转发:任何 import 路径拿到的是同一个模块对象,
全部名称(含测试使用的下划线私有名)都可用。维度校验参数化后默认值
仍为 panthera 的 14/7,行为等价;status 应答新增 profile 握手块(增量)。
"""

import sys

import finearm.arms.daemon.ipc as _real

sys.modules[__name__] = _real
