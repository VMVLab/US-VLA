import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur_example() -> dict:
    return {
        "state": np.random.rand(6),
        "images": {
            "side_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "ultrasound_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "scan the left liver standard plane",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class URInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    EXPECTED_CAMERAS: ClassVar[tuple[str, str, str]] = ("side_camera", "wrist_camera", "ultrasound_camera")

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            state = np.asarray(data["state"])
        elif "observation/state" in data:
            state = np.asarray(data["observation/state"])
        else:
            raise KeyError("Missing state input for UR policy")

        if "images" in data:
            side_image = _parse_image(data["images"].get(self.EXPECTED_CAMERAS[0]))
            wrist_image = _parse_image(data["images"].get(self.EXPECTED_CAMERAS[1]))
            ultrasound_image = _parse_image(data["images"].get(self.EXPECTED_CAMERAS[2]))
        else:
            side_image = _parse_image(data.get("observation/side_camera", data.get("observation/image")))
            wrist_image = _parse_image(data.get("observation/wrist_camera"))
            ultrasound_image = _parse_image(data.get("observation/ultrasound_camera"))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": side_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": ultrasound_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class UROutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}


@dataclasses.dataclass(frozen=True)
class URInputsWithoutUS(transforms.DataTransformFn):
    model_type: _model.ModelType
    EXPECTED_CAMERAS: ClassVar[tuple[str, str]] = ("side_camera", "wrist_camera")

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            state = np.asarray(data["state"])
        elif "observation/state" in data:
            state = np.asarray(data["observation/state"])
        else:
            raise KeyError("Missing state input for UR policy")

        if "images" in data:
            side_image = _parse_image(data["images"].get(self.EXPECTED_CAMERAS[0]))
            wrist_image = _parse_image(data["images"].get(self.EXPECTED_CAMERAS[1]))
        else:
            side_image = _parse_image(data.get("observation/side_camera", data.get("observation/image")))
            wrist_image = _parse_image(data.get("observation/wrist_camera"))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": side_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(side_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class UROutputsWithoutUS(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
