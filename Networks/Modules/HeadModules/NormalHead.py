import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ProbMLP
from functools import partial
import flax  # Remove unused import
from typing import Any

# --- grouped_log_softmax implementation ---
def grouped_log_softmax(logits: jnp.ndarray, jraph_graph_list: dict, n_groups: int = 66) -> jnp.ndarray: # todo plassma: n_groups is hardcoded for now
    group_ids = jraph_graph_list["graphs"][0].graph.globals["group_ids"]
    # num_segments = jraph_graph_list["n_groups"][0]#jnp.max(group_ids) + 1
    max_per_group = jax.ops.segment_max(logits[:, :, 0].squeeze(-1), group_ids, n_groups)
    shifted = logits[:, :, 0].squeeze(-1) - max_per_group[group_ids]
    exp_shifted = jnp.exp(shifted)
    sum_exp_per_group = jax.ops.segment_sum(exp_shifted, group_ids, n_groups)
    logsumexp_per_group = jnp.log(sum_exp_per_group)
    log_probs = (shifted - logsumexp_per_group[group_ids])[..., None]
    log_probs = jnp.where(jraph_graph_list["graphs"][0].graph.globals["group_ids"] == 0, 0, log_probs[..., 0])[..., None]
    return log_probs

def group_ids_from_graph(jraph_graph_list):
    graph = jraph_graph_list["graphs"][0]
    senders = graph.senders.astype(jnp.int32)
    receivers = graph.receivers.astype(jnp.int32)
    globals_ = graph.globals.astype(jnp.int32)
    sender_types = globals_[senders]
    receiver_types = globals_[receivers]
    n_edges = senders.shape[0]
    # Assign group ids: -1 for non-mutually-exclusive edges
    group_ids = -jnp.ones(n_edges, dtype=jnp.int32)
    receivers_int = jnp.asarray(receivers, dtype=jnp.int32)

    # Cabinet-room: group by cabinet (receiver type 1, sender type 0)
    mask_cabinet_room = jnp.asarray((sender_types == 0) & (receiver_types == 1), dtype=bool)
    group_ids = jnp.where(mask_cabinet_room, receivers_int, group_ids)
    # Thing-cabinet: group by thing (sender type 2, receiver type 1)
    mask_thing_cabinet = jnp.asarray((sender_types == 1) & (receiver_types == 2), dtype=bool)
    group_ids = jnp.where(mask_thing_cabinet, receivers_int, group_ids)
    # Room-person: group by room (sender type 0, receiver type 3)
    mask_room_person = jnp.asarray((sender_types == 0) & (receiver_types == 3), dtype=bool)
    group_ids = jnp.where(mask_room_person, receivers_int, group_ids)
    return group_ids

class NormalHeadModule(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: Any
    def setup(self):
        self.mode = "abnormal"

        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype = self.dtype, mode = self.mode)

        self.n_groups = None

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self, jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        spin_logits = self.probMLP(x)
        if self.mode == "normal":
            out_dict["spin_logits"] = spin_logits
            return out_dict
        
        # For categorical variables, we need to output raw logits, not log probabilities
        # The groupwise_sample function will handle the normalization
        raw_logits = jnp.where(jraph_graph_list["graphs"][0].graph.globals["group_ids"] == 0, 0, spin_logits[..., 0])[..., None]
        out_dict["spin_logits"] = raw_logits
        return out_dict
    


class TransformerHead(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: Any

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype= self.dtype)

    @partial(nn.jit, static_argnums=0)
    def __call__(self,jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        x_aggr = jnp.mean(x, axis = -2, keepdims=True)
        rep_x_aggr = jnp.repeat(x_aggr, x.shape[-2], axis = -2)
        x = jnp.concatenate([x, rep_x_aggr], axis = -1)
        spin_logits = self.probMLP(x)[:,0,...]


        spin_logits = jnp.reshape(spin_logits, (spin_logits.shape[0]*spin_logits.shape[1],) + (1,spin_logits.shape[-1]))

        out_dict["spin_logits"] = spin_logits
        return out_dict