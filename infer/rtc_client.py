#!/usr/bin/env python3
"""兼容 shim:RTC 推理客户端真身 = finearm.infer.cli(Phase 3 迁移)。

默认 --exec-mode rtc,行为与旧 rtc_client.py 等价(golden 对拍锁定);
新增 --exec-mode stop-and-go / --n-action-steps 为增量参数。
以 `python rtc_client.py` 运行时直接转发入口。
"""

import sys
from pathlib import Path

try:
    import finearm.infer.cli  # noqa: F401
except ImportError:
    _deploy_src = Path(__file__).resolve().parents[1] / "src"
    if str(_deploy_src) not in sys.path:
        sys.path.insert(0, str(_deploy_src))


def main():
    from finearm.infer.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
