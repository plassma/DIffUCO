import flax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder block with pre-LN layout."""
    embed_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        residual = x
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dtype=self.dtype,
        )(y, deterministic=deterministic)
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

class TransformerEncoderStack(nn.Module):
    """Simple stack of transformer encoder layers."""
    num_layers: int
    embed_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
        return x