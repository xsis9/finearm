# envs/arm · 机械臂驱动环境

daemon 驱动 venv(`.venv/`):python 3.12 + 厂商 SDK 轮子及其 Python 依赖
(pyyaml/scipy/pinocchio)。由本目录的 [`init_env.sh`](init_env.sh) 创建。

它驱动的机器资产(厂商 SDK、标定 `robot_param/`、daemon 启动器)仍在
[`arm/panthera/`](../../arm/panthera/)——见
[docs/daemon_zh.md](../../docs/daemon_zh.md)。
