import jax
import jax.numpy as jnp
from functools import partial

from .BaseTrainer import repeat_along_nodes

vmap_repeat_along_nodes = jax.vmap(repeat_along_nodes, in_axes=(0, 0, 0))


@partial(jax.jit, static_argnums=())
def select_time_indices(graph, data_buffer_dict, rand_diff_steps, rand_states, key):
    """Gather slices from the PPO buffer at random time/basis indices."""
    n_graphs = data_buffer_dict["policies"].shape[-2]
    n_nodes = data_buffer_dict["states"].shape[-3]
    n_devices = data_buffer_dict["policies"].shape[0]

    D_mat = jnp.arange(0, n_devices)[:, None, None, None]
    graph_mat = jnp.arange(0, n_graphs)[None, None, :, None]
    node_mat = jnp.arange(0, n_nodes)[None, None, :, None]

    rand_diff_steps_original = rand_diff_steps
    rand_diff_steps = jnp.transpose(rand_diff_steps, (0, -1, -3, -2))
    rand_states = jnp.transpose(rand_states, (0, 2, 1))

    rand_diff_steps_per_node = vmap_repeat_along_nodes(graph.nodes, graph.n_node, jnp.swapaxes(rand_diff_steps, 1, 2))
    rand_diff_steps_per_node = jnp.swapaxes(rand_diff_steps_per_node, 1, 2)

    out_dict = {}
    for dict_key, tensor in data_buffer_dict.items():
        if dict_key in ("states", "actions", "rand_node_features"):
            el = tensor[D_mat, rand_diff_steps_per_node, node_mat, rand_states[..., None, :]]
        else:
            el = tensor[D_mat, rand_diff_steps, graph_mat, rand_states[..., None, :]]

        el = jnp.swapaxes(el, 2, 3)
        el = jnp.reshape(el, (el.shape[0], el.shape[1] * el.shape[2]) + el.shape[3:])
        out_dict[dict_key] = el

    rand_diff_steps_original = jnp.swapaxes(rand_diff_steps_original, -1, -2)
    rand_diff_steps_resh = jnp.reshape(
        rand_diff_steps_original,
        (rand_diff_steps_original.shape[0], rand_diff_steps_original.shape[1], rand_diff_steps_original.shape[2] * rand_diff_steps_original.shape[3], 1),
    )
    out_dict["time_index"] = jnp.swapaxes(rand_diff_steps_resh, 1, 2)

    rand_diff_steps_per_node = jnp.swapaxes(rand_diff_steps_per_node, -1, -2)
    rand_diff_steps_per_node = jnp.reshape(
        rand_diff_steps_per_node,
        (rand_diff_steps_per_node.shape[0], rand_diff_steps_per_node.shape[1] * rand_diff_steps_per_node.shape[2], rand_diff_steps_per_node.shape[3], 1),
    )
    out_dict["time_index_per_node"] = rand_diff_steps_per_node
    return out_dict, key
