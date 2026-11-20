import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_panthera_example() -> dict:
    """随机输入样例,供 compute_norm_stats / 服务端预热使用。"""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the bolt and place it on the nut",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PantheraInputs(transforms.DataTransformFn):
    """训练与推理共用的输入变换。输出 dict 的 key 不要改。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            # 训练数据 14 维 / 推理客户端 7 维,统一在此截前 7 维
            "state": np.asarray(data["observation/state"])[..., :7],
            "image": {
                "base_0_rgb": base_image,                        # front 全局相机
                "left_wrist_0_rgb": wrist_image,                 # 腕相机
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 无右腕 → 补零
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PantheraOutputs(transforms.DataTransformFn):
    """推理输出变换:截取 7 维(服务端已反归一化、反 delta)。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
