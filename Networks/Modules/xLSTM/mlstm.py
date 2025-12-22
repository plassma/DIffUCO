from functools import partial

import jax
import flax.linen as nn
from jax import jit
from jax import numpy as jnp
from jax.nn import initializers as init
from typing import Any
from Networks.Modules.xLSTM.utils import causal_conv, normalize, linspace_init, centered_uniform_init

__all__ = ["Block", "Layer", "parallel", "sequential", "recurrence", "scan", "chunk_wise", "chunked_recurrence", "mLSTMStack"]


@partial(jit, static_argnames="causal")
def parallel(q, k, v, s_i, s_f, eps=1e-6, causal=True):
    """
    Parallel implementation of the mLSTM recurrence.

    Parameters
    ----------
    q : (..., S, H, D) array
    k : (..., S, H, D) array
    v : (..., S, H, D) array
    s_i : (..., S, H) array
    s_f : (..., S, H) array

    Returns
    -------
    h : (..., S, H, D) array
    """
    s_i = jnp.moveaxis(s_i, -2, -1)
    s_f = jnp.moveaxis(s_f, -2, -1)
    mask = jnp.tri(s_i.shape[-1], s_f.shape[-1], dtype=jnp.bool) if causal else jnp.ones((s_i.shape[-1], s_f.shape[-1]), dtype=jnp.bool)
    log_f = jnp.expand_dims(jax.nn.log_sigmoid(s_f), axis=-1)
    log_f_mat = jnp.repeat(log_f, s_f.shape[-1], axis=-1)
    log_prod_f_mat = jnp.cumsum(jnp.tril(log_f_mat, k=-1), axis=-2) if causal else jnp.cumsum(log_f_mat, axis=-2)
    f_mat = log_prod_f_mat + jnp.expand_dims(s_i, axis=-2)
    log_scale = jnp.where(mask, f_mat, -jnp.inf)
    max_log_scale = jnp.max(log_scale, axis=-1, keepdims=True)

    k_scale = 1. / k.shape[-1] ** .5
    qk = jnp.einsum("...shd,...thd->...hst", q, k * k_scale)
    c = jnp.exp(log_scale - max_log_scale) * qk
    n = jnp.abs(jnp.sum(c, axis=-1, keepdims=True))
    _n = jnp.maximum(jnp.abs(n), jnp.exp(-max_log_scale) + eps)
    return jnp.einsum("...hst,...thd->...shd", c / _n, v)


@partial(jit, static_argnames=("chunk_size", "causal"))
def chunk_wise(q, k, v, s_i, s_f, state=None, chunk_size=64, eps=1e-6, causal=True):
    """
    Chunk-wise parallel implementation of the mLSTM recurrence.

    Parameters
    ----------
    q : (..., S, H, D) array
    k : (..., S, H, D) array
    v : (..., S, H, D) array
    log_i : (..., S, H) array
    log_f : (..., S, H) array
    state : tuple, optional
        State of the recurrence containing:
         - c : (..., H, D, D) array
         - n : (..., H, D) array
         - m : (..., H) array

    Returns
    -------
    h : (..., S, H, D) array
    """
    if state is None:
        n0 = jnp.zeros_like(k[..., 0, :, :])
        c0 = jnp.zeros(n0.shape + (n0.shape[-1], ), dtype=n0.dtype)
        m0 = jnp.full(n0.shape[:-1], -jnp.inf, dtype=n0.dtype)
        state = (c0, n0, m0)

    sections = list(range(chunk_size, q.shape[-3], chunk_size))
    qs = jnp.split(q, sections, axis=-3)
    qs = jnp.split(q, sections, axis=-3)
    ks = jnp.split(k, sections, axis=-3)
    vs = jnp.split(v, sections, axis=-3)
    s_is = jnp.split(s_i, sections, axis=-2)
    s_fs = jnp.split(s_f, sections, axis=-2)

    # TODO: write as scan? (to reduce JIT compilation times)
    h = []
    for qi, ki, vi, s_ii, s_fi in zip(qs, ks, vs, s_is, s_fs):
        hi, state = chunked_recurrence(qi, ki, vi, s_ii, s_fi, state, eps=eps, causal=causal)
        h.append(hi)
    
    return jnp.concatenate(h, axis=-3)


@jit
def chunked_recurrence(q, k, v, s_i, s_f, state, eps=1e-6, causal=True):
    # parallel gating matrix computation
    s_i = jnp.moveaxis(s_i, -2, -1)
    s_f = jnp.moveaxis(s_f, -2, -1)
    mask = jnp.tri(s_i.shape[-1], s_f.shape[-1], dtype=jnp.bool) if causal else jnp.ones((s_i.shape[-1], s_f.shape[-1]), dtype=jnp.bool)
    log_f = jax.nn.log_sigmoid(s_f)
    log_f_mat = jnp.repeat(jnp.expand_dims(log_f, axis=-1), s_f.shape[-1], axis=-1)
    log_prod_f_mat = jnp.cumsum(jnp.tril(log_f_mat, k=-1), axis=-2) if causal else jnp.cumsum(log_f_mat, axis=-2)
    log_prod_f = log_f[..., :1] + log_prod_f_mat[..., 0]
    f_mat = log_prod_f_mat + jnp.expand_dims(s_i, axis=-2)
    log_scale = jnp.where(mask, f_mat, -jnp.inf)
    max_log_scale = jnp.max(log_scale, axis=-1, keepdims=True)
    d = jnp.exp(log_scale - max_log_scale)

    # recurrent gate computations
    c_prev, n_prev, m_prev = state
    m_and_f = jnp.expand_dims(m_prev, axis=-1) + log_prod_f
    m_all = jnp.maximum(max_log_scale[..., 0], m_and_f)
    i_all = jnp.exp(max_log_scale[..., 0] - m_all)
    f_all = jnp.exp(m_and_f - m_all)
    i, f = i_all[..., -1:], f_all[..., -1:]

    # recurrent state computations
    k = jnp.moveaxis(k, -3, -2)
    q = jnp.moveaxis(q, -3, -2)
    v = jnp.moveaxis(v, -3, -2)
    k = k / k.shape[-1] ** .5
    k_gates = k * jnp.expand_dims(d[..., -1, :], axis=-1)
    vk = jnp.einsum("...hsi,...hso->...hio", v, k_gates)
    c = jnp.expand_dims(f, axis=-1) * c_prev + jnp.expand_dims(i, axis=-1) * vk
    n = f * n_prev + i * jnp.sum(k_gates, axis=-2)
    m = m_all[..., -1]

    # parallel output computations
    q_igate = jnp.expand_dims(i_all, axis=-1) * q
    e_par = d * jnp.einsum("...hsd,...htd->...hst", q_igate, k)
    n_par = jnp.sum(e_par, axis=-1, keepdims=True)

    # recurrent output computations
    q_fgate = jnp.expand_dims(f_all, axis=-1) * q
    c_rec = jnp.einsum("...hsi,...hoi->...hso", q_fgate, c_prev)
    n_rec = jnp.einsum("...hsd,...hd->...hs", q_fgate, n_prev)

    # combine results
    n_both = n_par + jnp.expand_dims(n_rec, axis=-1)
    _n = jnp.maximum(jnp.abs(n_both), jnp.exp(-jnp.expand_dims(m_all, axis=-1)) + eps)
    h = (c_rec + jnp.einsum("...hst,...htd->...hsd", e_par, v)) / _n
    return jnp.moveaxis(h, -2, -3), (c, n, m)


@jit
def sequential(q, k, v, s_i, s_f, state=None, eps=1e-6):
    """
    Sequential implementation of the mLSTM recurrence.

    Parameters
    ----------
    q : (..., S, H, D) array
    k : (..., S, H, D) array
    v : (..., S, H, D) array
    log_i : (..., S, H) array
    log_f : (..., S, H) array
    state : tuple, optional
        State of the recurrence containing:
         - c : (..., S, H, D, D) array
         - n : (..., S, H, D) array
         - m : (..., S, H) array

    Returns
    -------
    h : (..., S, H, D) array
    """
    if state is None:
        n0 = jnp.zeros_like(k[..., 0, :, :])
        c0 = jnp.zeros(n0.shape + (n0.shape[-1], ), dtype=n0.dtype)
        m0 = jnp.full(n0.shape[:-1], -jnp.inf, dtype=n0.dtype)
        state = (c0, n0, m0)

    qs = jnp.unstack(q, axis=-3)
    ks = jnp.unstack(k, axis=-3)
    vs = jnp.unstack(v, axis=-3)
    s_is = jnp.unstack(s_i, axis=-2)
    s_fs = jnp.unstack(s_f, axis=-2)

    h = []
    for qi, ki, vi, s_ii, s_fi in zip(qs, ks, vs, s_is, s_fs):
        hi, state = recurrence(qi, ki, vi, s_ii, s_fi, state, eps=eps)
        h.append(hi)
    
    return jnp.stack(h, axis=-3)

@jit
def scan(q, k, v, s_i, s_f, state=None, eps=1e-6):
    """ Alternative to sequential that compiles faster, but runs slower. """
    if state is None:
        n0 = jnp.zeros_like(k[..., 0, :, :])
        c0 = jnp.zeros(n0.shape + (n0.shape[-1], ), dtype=n0.dtype)
        m0 = jnp.full(n0.shape[:-1], -jnp.inf, dtype=n0.dtype)
        state = (c0, n0, m0)
    
    _, out = jax.lax.scan(
        lambda state, inputs: recurrence(*inputs, state)[::-1],
        init=state,
        xs=(
            jnp.moveaxis(q, -3, 0), jnp.moveaxis(k, -3, 0), jnp.moveaxis(v, -3, 0),
            jnp.moveaxis(s_i, -2, 0), jnp.moveaxis(s_f, -2, 0),
        )
    )
    return jnp.moveaxis(out, 0, -3)


@jit
def recurrence(q, k, v, s_i, s_f, state, eps=1e-6):
    """
    Implementation of a single step in the mLSTM recurrence.

    Parameters
    ----------
    q : (..., H, D) array
    k : (..., H, D) array
    v : (..., H, D) array
    s_i : (..., H) array
    s_f : (..., H) array
    state : tuple
        State for the current step containing
         - c_prev : (..., H, D, D) array
         - n_prev : (..., H, D) array
         - m_prev : (..., H) array

    Returns
    -------
    h : (..., H, D) array
    c : (..., H, D, D) array
    n : (..., H, D) array
    m : (..., H) array
    """
    c_prev, n_prev, m_prev = state
    m_prev = m_prev + jax.nn.log_sigmoid(s_f)
    m = jnp.maximum(s_i, m_prev)
    i, f = jnp.exp(s_i - m), jnp.exp(m_prev - m)
    k = k / k.shape[-1] ** .5

    c_axes = (f.ndim, f.ndim + 1)
    c = jnp.expand_dims(f, axis=c_axes) * c_prev 
    c = c + jnp.expand_dims(i, axis=c_axes) * jnp.einsum("...i,...o->...io", v, k)
    n = jnp.expand_dims(f, axis=-1) * n_prev + jnp.expand_dims(i, axis=-1) * k
    _h = jnp.einsum("...i,...oi->...o", q, c)
    _n = jnp.maximum(jnp.abs(jnp.einsum("...i,...i->...", n, q)), jnp.exp(-m) + eps)
    h = _h / jnp.expand_dims(_n, axis=-1)
    return h, (c, n, m)


class Layer:
    """
    mLSTM layer.

    Combination of mLSTM recurrence and required weights.
    This can be used as a replacement for typical attention layers.
    """

    def __init__(self, num_heads: int, head_dim: int,
                 num_proj_heads: int | None = None, proj_head_dim: int | None = None):
        if num_proj_heads is None:
            num_proj_heads = head_dim
        if proj_head_dim is None:
            proj_head_dim = num_heads * head_dim // num_proj_heads
        if proj_head_dim * num_proj_heads != num_heads * head_dim:
            msg1 = f"projection dimensions ({num_proj_heads} * {proj_head_dim})"
            msg2 = f"attention dimmensions ({num_heads} * {head_dim})"
            raise ValueError(f"{msg1} do not match {msg2}")

        self.q_weight = jnp.empty((num_proj_heads, proj_head_dim, proj_head_dim))
        self.k_weight = jnp.empty((num_proj_heads, proj_head_dim, proj_head_dim))
        self.v_weight = jnp.empty((num_proj_heads, proj_head_dim, proj_head_dim))

        self.i_weight = jnp.empty((num_heads, num_proj_heads, 3 * proj_head_dim))
        self.i_bias = jnp.empty(num_heads)
        self.f_weight = jnp.empty((num_heads, num_proj_heads, 3 * proj_head_dim))
        self.f_bias = jnp.empty(num_heads)

    @property
    def num_heads(self):
        return self.i_weight.shape[0]

    @property
    def head_dim(self):
        return self.num_proj_heads * self.proj_head_dim // self.num_heads

    @property
    def num_proj_heads(self):
        return self.q_weight.shape[0]

    @property
    def proj_head_dim(self):
        return self.q_weight.shape[1]

    def reset_parameters(self, key,
        q_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        k_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        v_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        i_weight_init=init.zeros,
        i_bias_init=init.normal(0.1),
        f_weight_init=init.zeros,
        f_bias_init=linspace_init(3., 6.),
    ):
        keys = jax.random.split(key, 10)
        self.q_weight = q_weight_init(keys[0], self.q_weight.shape, self.q_weight.dtype)
        self.k_weight = k_weight_init(keys[2], self.k_weight.shape, self.k_weight.dtype)
        self.v_weight = v_weight_init(keys[4], self.v_weight.shape, self.v_weight.dtype)

        self.i_weight = i_weight_init(keys[6], self.i_weight.shape, self.i_weight.dtype)
        self.f_weight = f_weight_init(keys[7], self.f_weight.shape, self.f_weight.dtype)
        self.i_bias = i_bias_init(keys[8], self.i_bias.shape, self.i_bias.dtype)
        self.f_bias = f_bias_init(keys[9], self.f_bias.shape, self.f_bias.dtype)

    def __call__(self, x_q, x_k, x_v, *, causal=True):
        _x_q = jnp.reshape(x_q, (*x_q.shape[:-1], self.num_proj_heads, -1))
        _x_k = jnp.reshape(x_k, (*x_k.shape[:-1], self.num_proj_heads, -1))
        _x_v = jnp.reshape(x_v, (*x_v.shape[:-1], self.num_proj_heads, -1))
        q = jnp.einsum("...hi,hoi->...ho", _x_q, self.q_weight)
        k = jnp.einsum("...hi,hoi->...ho", _x_k, self.k_weight)
        v = jnp.einsum("...hi,hoi->...ho", _x_v, self.v_weight)

        qkv = jnp.concatenate([q, k, v], axis=-1)
        s_i = jnp.einsum("...hi,ohi->...o", qkv, self.i_weight) + self.i_bias
        s_f = jnp.einsum("...hi,ohi->...o", qkv, self.f_weight) + self.f_bias
        
        _q = jnp.reshape(q, (*x_q.shape[:-1], self.num_heads, -1))
        _k = jnp.reshape(k, (*x_k.shape[:-1], self.num_heads, -1))
        _v = jnp.reshape(v, (*x_v.shape[:-1], self.num_heads, -1))
        return parallel(_q, _k, _v, s_i, s_f, causal=causal)


class Block:
    """
    mLSTM block.

    Implementation of mLSTM block with skip-connections and normalization.
    Dropout layers have not been included (i.e. this is an inference model).
    """

    def __init__(
        self, in_dim: int, num_heads: int,
        num_proj_heads: int | None= None,
        proj_head_dim: int = 4,
        kernel_size: int = 4,
        causal: bool = True,
    ):
        if num_proj_heads is None:
            num_proj_heads = in_dim // 2

        hid_dim = num_proj_heads * proj_head_dim
        assert hid_dim % num_heads == 0, "num_heads must divide num_proj_heads * proj_head_dim"
        self.in_norm_weight = jnp.empty((in_dim))
        self.up_weight = jnp.empty((2 * hid_dim, in_dim))
        self.down_weight = jnp.empty((in_dim, hid_dim))
        self.conv_weight = jnp.empty((hid_dim, 1, kernel_size))
        self.conv_bias = jnp.empty(hid_dim)

        self.mlstm_layer = Layer(num_heads, hid_dim // num_heads,
                                 num_proj_heads, proj_head_dim)

        self.norm_weight = jnp.empty(hid_dim)
        self.skip_weight = jnp.empty(hid_dim)
        self.causal = causal
        self.reset_parameters(jax.random.key(2024))

    def reset_parameters(self, key,
        in_norm_weight_init=init.zeros,
        up_weight_init=init.variance_scaling(1 / 5, "fan_out", "normal", in_axis=1, out_axis=0),
        down_weight_init=init.variance_scaling(4, "fan_out", "normal", in_axis=1, out_axis=0),
        conv_weight_init=init.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=1, out_axis=0),
        conv_bias_init=centered_uniform_init(1 / 2),
        q_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        k_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        v_weight_init=init.variance_scaling(2 / 5, "fan_out", "normal", in_axis=2, out_axis=(0, 1)),
        i_weight_init=init.zeros,
        i_bias_init=init.normal(0.1),
        f_weight_init=init.zeros,
        f_bias_init=linspace_init(3., 6.),
        skip_weight_init=init.ones,
        norm_weight_init=init.zeros,
    ):
        keys = jax.random.split(key, 8)
        self.in_norm_weight = in_norm_weight_init(keys[-1], self.in_norm_weight.shape, self.in_norm_weight.dtype)
        self.up_weight = up_weight_init(keys[0], self.up_weight.shape, self.up_weight.dtype)
        self.down_weight = down_weight_init(keys[1], self.down_weight.shape, self.down_weight.dtype)
        self.conv_weight = conv_weight_init(keys[2], self.conv_weight.shape, self.conv_weight.dtype)
        self.conv_bias = conv_bias_init(keys[3], self.conv_bias.shape, self.conv_bias.dtype)

        self.mlstm_layer.reset_parameters(keys[4],
            q_weight_init, k_weight_init, v_weight_init,
            i_weight_init, i_bias_init, f_weight_init, f_bias_init
        )

        self.skip_weight = skip_weight_init(keys[5], self.skip_weight.shape, self.skip_weight.dtype)
        self.norm_weight = norm_weight_init(keys[6], self.norm_weight.shape, self.norm_weight.dtype)
    
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        x_norm = (1. + self.in_norm_weight) * normalize(x)
        s = jnp.einsum("...i,oi->...o", x_norm, self.up_weight)
        x1, s_o = jnp.split(s, 2, axis=-1)
        s2 = causal_conv(x1, self.conv_weight, self.conv_bias, causal=self.causal)
        x2 = jax.nn.silu(s2)

        h_tilde = self.mlstm_layer(x2, x2, x1, causal=self.causal)

        # group-norm, skip-connections and output gate
        h_norm = jnp.reshape(normalize(h_tilde, axis=-1), (*batch_shape, -1))
        h_skip = (1. + self.norm_weight) * h_norm + self.skip_weight * x2
        h = h_skip * jax.nn.silu(s_o)
        out = jnp.einsum("...i,oi->...o", h, self.down_weight)
        return out + x
    
class mLSTMStack(nn.Module):
    """Stack of encoder layers using linear attention throughout."""
    num_layers: int
    embed_dim: int
    num_heads: int
    causal: bool = False

    @nn.compact
    def __call__(self, x,):
        for _ in range(self.num_layers):
            x = Block(
                in_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_proj_heads=None,
                causal=self.causal,
            )(x)
        return x
