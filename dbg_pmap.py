import jax
import jax.numpy as jnp

print("Devices:", jax.devices(), flush=True)

@jax.pmap
def f(x):
    return x + jax.lax.psum(x, axis_name='i')

x = jnp.ones((jax.device_count(), 4), jnp.float32)
print("Input shape:", x.shape, flush=True)
y = f(x)
print("Output:", y, flush=True)