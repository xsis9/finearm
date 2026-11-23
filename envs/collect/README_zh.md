# envs/collect · 采集环境

采集栈使用的 LeRobot **0.5.1** venv(`.venv-lerobot/`)。锁定依赖:本目录的
[`requirements-lerobot.txt`](requirements-lerobot.txt)。

唯一入口(顺带克隆/校验厂商 SDK、安装插件):

```bash
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --install
./plugins/panthera-lerobot-collect/scripts/init_workspace.sh --check
```

仓库规则:所有 venv 一律放 `envs/<域>/`——找环境不需要知道用它的代码在哪。
venv 不可搬;目录挪动后用上面的入口重建。
