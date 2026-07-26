# Copyright 2026 US-VLA Authors.
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
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

Dtype = Any


@dataclasses.dataclass(frozen=True)
class UltrasoundFusionConfig:
    hidden_dim: int = 2048
    num_heads: int = 8
    out_dim: int = 1024
    ffn_dim: int | None = None
    dropout_rate: float = 0.1
    num_layers: int = 1
    dtype: str = "bfloat16"

    @property
    def ffn_hidden_dim(self) -> int:
        return self.ffn_dim if self.ffn_dim is not None else 2 * self.hidden_dim

    @property
    def jax_dtype(self):
        if self.dtype == "bfloat16":
            return jnp.bfloat16
        if self.dtype == "float16":
            return jnp.float16
        return jnp.float32


class CrossAttentionBlock(nn.Module):
    num_heads: int
    dtype: Dtype = jnp.bfloat16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        query_input: jnp.ndarray,
        kv_input: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        query_norm = nn.LayerNorm(dtype=self.dtype, name="query_norm")(query_input)
        kv_norm = nn.LayerNorm(dtype=self.dtype, name="kv_norm")(kv_input)

        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            kernel_init=nn.initializers.xavier_uniform(),
            name="cross_attn",
        )(query_norm, kv_norm)

        return query_input + attn_output


class FFNBlock(nn.Module):
    hidden_dim: int
    out_dim: int
    dtype: Dtype = jnp.bfloat16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        residual = x

        x = nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(x)
        x = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ffn_up",
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            features=self.out_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ffn_down",
        )(x)

        return residual + x


class UltrasoundFusionBlock(nn.Module):
    config: UltrasoundFusionConfig

    @nn.compact
    def __call__(
        self,
        vlm_features: jnp.ndarray,
        ultrasound_features: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        config = self.config
        dtype = config.jax_dtype

        x = vlm_features

        for i in range(config.num_layers):
            x = CrossAttentionBlock(
                num_heads=config.num_heads,
                dtype=dtype,
                dropout_rate=config.dropout_rate,
                name=f"cross_attn_layer_{i}",
            )(x, ultrasound_features, deterministic=deterministic)

            x = FFNBlock(
                hidden_dim=config.ffn_hidden_dim,
                out_dim=config.hidden_dim,
                dtype=dtype,
                dropout_rate=config.dropout_rate,
                name=f"ffn_layer_{i}",
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=dtype, name="final_norm")(x)

        return nn.Dense(
            features=config.out_dim,
            dtype=dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output_proj",
        )(x)


class UltrasoundFusionModule(nn.Module):
    config: UltrasoundFusionConfig

    @nn.compact
    def __call__(
        self,
        vlm_features: jnp.ndarray,
        ultrasound_features: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        return UltrasoundFusionBlock(
            config=self.config,
            name="fusion_block",
        )(vlm_features, ultrasound_features, deterministic=deterministic)
