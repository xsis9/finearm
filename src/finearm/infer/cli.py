"""``finearm-infer`` CLI: RTC / stop-and-go chunked inference against an arm backend.

Successor of ``infer/rtc_client.py`` (which forwards here as a shim, default
``--exec-mode rtc``). Flags are inherited unchanged; new flags are additive.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

from ..core import paths as core_paths
from .adapters import available as adapters_available, load as load_adapter
from .executors import RTCExecutor, StopAndGoExecutor
from .mock import MockRobot


def _infer_dir_on_path() -> None:
    """Make the repo's infer/ importable for the default robot backend module."""
    repo = core_paths.find_repo_root()
    if repo is not None:
        infer_dir = str(repo / "infer")
        if infer_dir not in sys.path:
            sys.path.insert(0, infer_dir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RTC / stop-and-go 推理执行客户端", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exec-mode", choices=["rtc", "stop-and-go"], default="rtc",
                    help="执行器:rtc=流水线单在途请求(默认,连续运动);"
                         "stop-and-go=每窗阻塞推理新鲜参考(混合部署 actor 段语义)")
    ap.add_argument("--model", required=True, choices=adapters_available(),
                    help="模型适配器(finearm/infer/adapters/ 目录;新增模型见 infer/README.md)")
    ap.add_argument("--host", required=True, help="推理服务器 IP(走 ssh 隧道时 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="缺省用适配器默认端口")
    ap.add_argument("--task", default="pick up the bolt and place it on the nut",
                    help="任务指令;必须与训练 prompt 逐字一致")
    # rtc-only
    ap.add_argument("--k", type=int, default=None, help="固定 k;缺省自适应(按 RTT p95)[rtc]")
    ap.add_argument("--blend", type=int, default=2, help="chunk 切换线性混合步数 [rtc]")
    ap.add_argument("--starve", choices=["hold", "stop"], default="hold", help="chunk 断档策略 [rtc]")
    # stop-and-go only
    ap.add_argument("--n-action-steps", type=int, default=10,
                    help="每窗执行步数;0=整 chunk [stop-and-go]")
    # shared
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--arm", default="robot_interface:RobotInterface",
                    help="机械臂后端 模块:类;默认 infer/robot_interface.py"
                         "(Phase 5a 起 shim 转发到 finearm.infer.robot_interface)")
    ap.add_argument("--home", default=None,
                    help="归位位姿(7 个逗号分隔值);优先级:本参数 > PANTHERA_RTC_HOME "
                         "> 适配器默认;空字符串跳过归位")
    ap.add_argument("--mock", action="store_true",
                    help="完全不接硬件空跑(不连 daemon/相机,假观测+假下发)")
    ap.add_argument("--mock-images", action="store_true",
                    help="连 daemon 但相机用全零图(配 mock daemon 可走完整 IPC 链路,不动真臂)")
    ap.add_argument("--dry-run", action="store_true",
                    help="归位后只打印 action 不下发(验证推理链路;归位阶段会实际运动)")
    ap.add_argument("--duration", type=float, default=None, help="运行时长(秒),到点自动停")
    ap.add_argument("--max-steps", type=int, default=None, help="最大步数(优先于 --duration)")
    # 显式限幅(daemon 侧仅保留逐拍跳变限制作为兜底,不承担常规限速)
    ap.add_argument("--pos-lo", default=None, help="7 个数,逗号分隔")
    ap.add_argument("--pos-hi", default=None)
    ap.add_argument("--vel-limit", default=None,
                    help="每步最大位移,如 0.02,0.02,0.02,0.02,0.02,0.02,0.01")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    adapter = load_adapter(args.model, host=args.host, port=args.port)
    adapter.reset()
    print(f"[infer] 适配器 {adapter.NAME} 就绪(port {args.port or adapter.DEFAULT_PORT}, "
          f"obs_style={adapter.OBS_STYLE}, exec={args.exec_mode})")

    if args.mock:
        robot = MockRobot()
    else:
        _infer_dir_on_path()
        mod_name, cls_name = args.arm.split(":")
        home = args.home if args.home is not None else \
            os.environ.get("PANTHERA_RTC_HOME", "") or adapter.DEFAULT_HOME
        robot = getattr(importlib.import_module(mod_name), cls_name)(
            home=home,
            img_sizes=(224,) if adapter.OBS_STYLE == "openpi" else (),
            mock_images=args.mock_images)
        if args.dry_run:
            print("*** DRY-RUN: action 只打印不下发(归位已实际运动)***")
        else:
            print("*** POLICY MODE ON: action 将下发 ***")

    pos_limits = None
    if args.pos_lo and args.pos_hi:
        pos_limits = (
            np.array([float(x) for x in args.pos_lo.split(",")]),
            np.array([float(x) for x in args.pos_hi.split(",")]),
        )
    vel_limits = None
    if args.vel_limit:
        vel_limits = np.array([float(x) for x in args.vel_limit.split(",")])

    max_steps = args.max_steps
    if max_steps is None and args.duration is not None:
        max_steps = int(args.duration * args.hz)

    if args.exec_mode == "rtc":
        executor = RTCExecutor(
            adapter, k=args.k, ctrl_hz=args.hz, blend=args.blend, starve=args.starve,
            pos_limits=pos_limits, vel_limits=vel_limits, dry_run=args.dry_run)
    else:
        executor = StopAndGoExecutor(
            adapter, n_action_steps=args.n_action_steps, ctrl_hz=args.hz,
            pos_limits=pos_limits, vel_limits=vel_limits, dry_run=args.dry_run)
    executor.run(robot, args.task, max_steps=max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
