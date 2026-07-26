# Copyright 2026 US-VLA Authors.
#
# JAX/Flax implementation of the USFM (Ultrasound Foundation Model) encoder,
# compatible with the PyTorch pretrained weights released at
# https://github.com/openmedlab/USFM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.training.sharding as sharding


@dataclasses.dataclass
class USFMConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_class_token: bool = True
    dtype: str = "float32"

    target_num_patches: int = 256

    @property
    def mlp_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


class PatchEmbed(nn.Module):
    embed_dim: int = 768
    patch_size: int = 16
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype,
            name="proj",
        )(x)
        b, h, w, c = x.shape
        return jnp.reshape(x, (b, h * w, c))


class Attention(nn.Module):
    dim: int
    num_heads: int = 12
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        b, n, c = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim**-0.5

        qkv = nn.Dense(
            features=self.dim * 3,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            name="qkv",
        )(x)
        qkv = jnp.reshape(qkv, (b, n, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        x = attn @ v
        x = jnp.swapaxes(x, 1, 2)
        x = jnp.reshape(x, (b, n, c))

        x = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            name="proj",
        )(x)
        return nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)


class MlpBlock(nn.Module):
    hidden_features: int
    out_features: int | None = None
    drop: float = 0.0
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        out_features = self.out_features or x.shape[-1]

        x = nn.Dense(
            features=self.hidden_features,
            dtype=self.dtype,
            name="fc1",
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        x = nn.Dense(
            features=out_features,
            dtype=self.dtype,
            name="fc2",
        )(x)
        return nn.Dropout(rate=self.drop)(x, deterministic=deterministic)


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    dtype: str = "float32"

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype, name="norm1")(x)
        x = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            dtype=self.dtype,
            name="attn",
        )(x, deterministic=deterministic)
        x = residual + x

        residual = x
        x = nn.LayerNorm(dtype=self.dtype, name="norm2")(x)
        x = MlpBlock(
            hidden_features=int(self.dim * self.mlp_ratio),
            drop=self.drop,
            dtype=self.dtype,
            name="mlp",
        )(x, deterministic=deterministic)
        return residual + x


class USFMEncoder(nn.Module):
    config: USFMConfig = dataclasses.field(default_factory=USFMConfig)
    output_dim: int | None = None

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        config = self.config
        deterministic = not train
        dtype = jnp.dtype(config.dtype)

        x = jnp.asarray(x, dtype=jnp.float32)

        x = PatchEmbed(
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            dtype=config.dtype,
            name="patch_embed",
        )(x)

        b, num_patches, _ = x.shape

        if config.use_class_token:
            cls_token = self.param(
                "cls_token",
                nn.initializers.zeros,
                (1, 1, config.embed_dim),
            )
            cls_tokens = jnp.tile(cls_token, (b, 1, 1))
            x = jnp.concatenate([cls_tokens, x], axis=1)
            num_patches += 1

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, config.embed_dim),
        )
        x = x + pos_embed
        x = nn.Dropout(rate=config.dropout)(x, deterministic=deterministic)

        x = x.astype(dtype)

        for i in range(config.depth):
            x = sharding.activation_sharding_constraint(x)
            x = TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                drop=config.dropout,
                attn_drop=config.attention_dropout,
                dtype=config.dtype,
                name=f"blocks_{i}",
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=config.dtype, name="norm")(x)

        if config.use_class_token:
            x = x[:, 1:, :]

        grid_size = config.grid_size
        target_grid = int(config.target_num_patches**0.5)

        if grid_size * grid_size != x.shape[1]:
            raise ValueError(f"Expected {grid_size * grid_size} patches, got {x.shape[1]}")

        x = jnp.reshape(x, (b, grid_size, grid_size, -1))
        x = jax.image.resize(
            x,
            shape=(b, target_grid, target_grid, x.shape[-1]),
            method="bilinear",
        )
        x = jnp.reshape(x, (b, target_grid * target_grid, -1))

        if self.output_dim is not None:
            x = nn.Dense(
                features=self.output_dim,
                dtype=config.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name="proj_out",
            )(x)

        return x
