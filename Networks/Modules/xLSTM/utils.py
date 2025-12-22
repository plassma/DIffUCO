from jax import lax
import jax
import jax.numpy as jnp


def normalize(x, axis=-1, eps=1e-5):
    """Simple normalization: zero mean, unit variance along the given axis."""
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=axis, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def linspace_init(start, stop):
    """Initializer that linearly spaces values between start and stop."""
    def _init(key, shape, dtype=jnp.float_):
        del key
        base = jnp.linspace(start, stop, shape[-1], dtype=dtype)
        return jnp.broadcast_to(base, shape)
    return _init


def centered_uniform_init(scale):
    """Initializer sampling uniformly around zero with the given half-width."""
    def _init(key, shape, dtype=jnp.float_):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return _init


def causal_conv(x, w, b, *, causal=True):
    """
    Depth-wise 1D convolution over the sequence dimension with optional causal padding.

    Parameters
    ----------
    x : (..., S, H) array
        Input sequence with feature dimension last.
    w : (H, 1, K) array
        Convolution weights; one kernel per feature channel.
    b : (H,) array
        Bias per feature channel.
    causal : bool, default True
        If True, pad on the left only (causal). If False, use symmetric padding.

    Returns
    -------
    (..., S, H) array
    """
    x_ch_first = jnp.moveaxis(x, -2, -1)  # (..., H, S)
    k = w.shape[-1]
    if causal:
        padding = ((k - 1, 0),)  # pad time dimension on the left only
    else:
        left = k // 2
        padding = ((left, k - 1 - left),)

    y = lax.conv_general_dilated(
        x_ch_first,
        w,
        window_strides=(1,),
        padding=padding,
        dimension_numbers=("NCT", "OIT", "NCT"),
        feature_group_count=x_ch_first.shape[-2],  # depth-wise
    )
    y = y + b[..., None]
    return jnp.moveaxis(y, -2, -1)
