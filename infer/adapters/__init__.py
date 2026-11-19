"""模型适配器注册表 + 共用工具(rtc_client.py / bench_rtt.py 通过名字加载)。

适配器接口约定(写新适配器前必读,详见 ../README.md「新增模型:写一个适配器」):

  NAME          适配器名(与文件名一致,--model 参数用它选)
  DEFAULT_PORT  默认端口
  OBS_STYLE     观测风格,决定机械臂后端怎么取观测:
                  "openpi" → get_obs("openpi"):224×224 预 resize 图 + state 前 7 维
                  "raw"    → get_obs("raw"):原始 640×480 帧 + 完整 14 维 state
  DEFAULT_HOME  推理前归位位姿("j1,...,j6,gripper" 7 值;空串=不归位)。
                优先级:rtc_client --home > 环境变量 PANTHERA_RTC_HOME > 此默认值

  __init__(host, port=None)   建立与推理服务的连接(端口缺省用 DEFAULT_PORT)
  reset()                     连接后握手(无需则 pass;lingbot 要发 robo_name)
  build_obs(obs, task)        get_obs(OBS_STYLE) 的结果 → 该模型的推理请求 dict
  infer(payload) -> (T,7)     发请求并把返回解析成绝对关节目标 chunk
                              (6 关节 + 夹爪,行=时间步;数值校验在这里做)

适配器文件里**不要**在 import 时连接服务(延迟到 __init__),保证
`import adapters.<name>` 无副作用(注册表扫描/单元测试都只 import 不实例化)。
"""
import importlib
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent


class AdapterBase:
    """适配器基类:接口占位 + 公共默认值。子类必须覆盖 NAME/DEFAULT_PORT/OBS_STYLE
    与 __init__/build_obs/infer;reset 按需。"""

    NAME = "?"
    DEFAULT_PORT = 0
    OBS_STYLE = "openpi"
    DEFAULT_HOME = ""

    def __init__(self, host, port=None):
        raise NotImplementedError

    def reset(self):
        pass

    def build_obs(self, obs, task):
        raise NotImplementedError

    def infer(self, payload):
        raise NotImplementedError


def available():
    """已注册的适配器名(adapters/ 下的 .py 文件名,排除 __init__/_ 前缀)。"""
    return sorted(p.stem for p in _HERE.glob("*.py")
                  if p.stem != "__init__" and not p.stem.startswith("_"))


def load(name, host, port=None):
    """按名字加载并实例化适配器(此时才连接推理服务)。"""
    if name not in available():
        raise SystemExit(f"未知适配器 {name!r};可用: {', '.join(available())}"
                         f"(在 adapters/ 目录下,新增适配器见 ../README.md)")
    mod = importlib.import_module(f"{__name__}.{name}")
    return mod.Adapter(host=host, port=port)


def mock_obs(style, seed=0):
    """生成指定风格的假观测(--mock 空跑 / bench_rtt 用,不需要任何硬件)。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    if style == "openpi":
        img = lambda: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)  # noqa: E731
        return {
            "observation/image": img(),
            "observation/wrist_image": img(),
            "observation/state": np.array([0.05, 0.3, 1.2, -0.5, 0.0, 0.1, 0.8], np.float32),
        }
    if style == "raw":
        img = lambda: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)  # noqa: E731
        return {
            "front": img(),
            "wrist": img(),
            "state": np.concatenate([
                np.array([0.05, 0.3, 1.2, -0.5, 0.0, 0.1, 0.8], np.float32),
                np.zeros(7, np.float32)]),
        }
    raise ValueError(f"未知 style: {style}(openpi | raw)")
