import flax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any


class LinearSelfAttention(nn.Module):
    """Linear-time self-attention using an ELU+1 feature map."""
    embed_dim: int
    num_heads: int
    dtype: Any = jnp.float32
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = self.embed_dim // self.num_heads

        # Project inputs to queries, keys and values.
        queries = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            name="query",
        )(x)
        keys = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            name="key",
        )(x)
        values = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            name="value",
        )(x)

        # Positive feature map for linear attention.
        feature_map = lambda t: nn.elu(t) + 1
        queries_phi = feature_map(queries)
        keys_phi = feature_map(keys)

        # Compute denominator and numerator in linear attention form.
        keys_sum = keys_phi.sum(axis=-3)
        kv = jnp.einsum("...lhd,...lhm->...hdm", keys_phi, values)

        normalizer = jnp.einsum("...lhd,...hd->...lh", queries_phi, keys_sum)
        normalizer = normalizer + self.epsilon

        attention = jnp.einsum("...lhd,...hdm->...lhm", queries_phi, kv)
        attention = attention / normalizer[..., None]

        attention = attention.reshape(*attention.shape[:-2], self.embed_dim)
        return nn.Dense(self.embed_dim, dtype=self.dtype)(attention)


class LinearTransformerEncoderLayer(nn.Module):
    """Single linear transformer encoder block with pre-LN layout."""
    embed_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        residual = x
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = LinearSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dtype=self.dtype,
        )(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        x = residual + y

        residual = x
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(self.mlp_dim, dtype=self.dtype)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.embed_dim, dtype=self.dtype)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return residual + y


class LinearTransformerEncoderStack(nn.Module):
    """Simple stack of linear transformer encoder layers."""
    num_layers: int
    embed_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        for _ in range(self.num_layers):
            x = LinearTransformerEncoderLayer(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
        return x
