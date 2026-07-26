import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    usfm_weights_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        result = _merge_params(
            loaded_params,
            params,
            missing_regex=".*(lora|usfm_encoder|USFMEncoder|ultrasound_fusion|UltrasoundFusion).*",
        )

        if self.usfm_weights_path is not None:
            if pathlib.Path(self.usfm_weights_path).exists():
                logger.info(f"Loading USFM pretrained weights from: {self.usfm_weights_path}")
                result = USFMWeightLoader(usfm_weights_path=self.usfm_weights_path).load(result)
            else:
                logger.warning(f"USFM weights path not found: {self.usfm_weights_path}. Using random initialization.")

        return result


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class USFMWeightLoader(WeightLoader):
    usfm_weights_path: str
    backbone_prefix: str = "backbone."

    def load(self, params: at.Params) -> at.Params:
        weights_path = download.maybe_download(self.usfm_weights_path)
        logger.info(f"Loading USFM weights from {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        usfm_params = self._convert_pytorch_to_flax(state_dict)

        return _merge_params(usfm_params, params, missing_regex=".*")

    def _convert_pytorch_to_flax(self, state_dict: dict) -> at.Params:
        flax_params = {}
        prefix = self.backbone_prefix

        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue

            flax_key = self._map_key(key[len(prefix) :])
            if flax_key is None:
                continue

            np_value = self._transpose_if_needed(flax_key, value.detach().cpu().numpy())
            self._set_nested(flax_params, flax_key, np_value)

        return flax_params

    def _map_key(self, pt_key: str) -> str | None:
        if pt_key == "patch_embed.proj.weight":
            return "USFMEncoder/patch_embed/proj/kernel"
        if pt_key == "patch_embed.proj.bias":
            return "USFMEncoder/patch_embed/proj/bias"

        if pt_key == "pos_embed":
            return "USFMEncoder/pos_embed"
        if pt_key == "cls_token":
            return "USFMEncoder/cls_token"

        if pt_key == "norm.weight":
            return "USFMEncoder/norm/scale"
        if pt_key == "norm.bias":
            return "USFMEncoder/norm/bias"

        block_match = re.match(r"blocks\.(\d+)\.(.+)", pt_key)
        if block_match:
            block_idx = block_match.group(1)
            rest = block_match.group(2)

            component_map = {
                "norm1.weight": "norm1/scale",
                "norm1.bias": "norm1/bias",
                "attn.qkv.weight": "attn/qkv/kernel",
                "attn.qkv.bias": "attn/qkv/bias",
                "attn.proj.weight": "attn/proj/kernel",
                "attn.proj.bias": "attn/proj/bias",
                "norm2.weight": "norm2/scale",
                "norm2.bias": "norm2/bias",
                "mlp.fc1.weight": "mlp/fc1/kernel",
                "mlp.fc1.bias": "mlp/fc1/bias",
                "mlp.fc2.weight": "mlp/fc2/kernel",
                "mlp.fc2.bias": "mlp/fc2/bias",
            }

            if rest in component_map:
                return f"USFMEncoder/blocks_{block_idx}/{component_map[rest]}"

        return None

    def _transpose_if_needed(self, flax_key: str, value: np.ndarray) -> np.ndarray:
        if "patch_embed" in flax_key and "kernel" in flax_key:
            return np.transpose(value, (2, 3, 1, 0))

        if "kernel" in flax_key and value.ndim == 2:
            return np.transpose(value, (1, 0))

        return value

    def _set_nested(self, d: dict, key_path: str, value: np.ndarray):
        keys = key_path.split("/")
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
