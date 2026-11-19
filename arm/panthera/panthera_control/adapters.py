"""兼容 shim:适配层真身 = finearm.arms.panthera.adapter(Phase 2)。

经 sys.modules 别名转发:任何 import 路径拿到的是同一个模块对象,
全部名称(含测试使用的下划线私有名)都可用。基类/Mock 适配器在
finearm.arms.daemon.adapters,一并经由本模块再导出。
"""

import sys

import finearm.arms.panthera_ht.adapter as _real

sys.modules[__name__] = _real
