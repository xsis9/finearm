"""``finearm-collect`` CLI: run a collection method (default: teleop).

Successor of ``panthera-lr-collect`` (which keeps working through the
``panthera_lerobot_control.collect_cli`` shim). All teleop flags pass
through; ``--method`` selects the collector plugin.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from .collector import available, load

    args_list = list(sys.argv[1:] if argv is None else argv)
    method = "teleop"
    filtered = []
    i = 0
    while i < len(args_list):
        token = args_list[i]
        if token == "--method":
            i += 1
            method = args_list[i]
        elif token.startswith("--method="):
            method = token.split("=", 1)[1]
        else:
            filtered.append(token)
        i += 1

    if method not in available():
        print(f"未知采集方式 {method!r};可用: {', '.join(available())}", file=sys.stderr)
        return 2
    module = load(method)
    return module.main(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
