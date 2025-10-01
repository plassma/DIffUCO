import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial

from Networks.Modules import get_GNN_model


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

def gumbel_keys(key, shape):
    """Sample Gumbel noise."""
    u = jax.random.uniform(key, shape=shape, minval=1e-6, maxval=1.0)
    return -jnp.log(-jnp.log(u))


def groupwise_sample(key, logits, group_ids, num_segments=66, temp=1.0):
    """Groupwise sampling from logits via the Gumbel-max trick.
    Assumes logits are already groupwise log_softmax-normalized.
    Adds temperature scaling to logits before sampling.
    """
    if len(logits.shape) == 2:
        logits = logits[..., None]
    if key is None:
        key = jax.random.PRNGKey(0)

    gumbel_noise = gumbel_keys(key, logits.shape)
    noisy_logits = logits / temp + gumbel_noise

    max_per_group = jax.ops.segment_max(noisy_logits, group_ids, num_segments=num_segments)
    is_max = (noisy_logits == max_per_group[group_ids])
    is_max = jnp.where((group_ids == 0)[..., None, None], 1, is_max)
    return is_max.astype(jnp.int32)

class DummyEdgeMLP(nn.Module):
    def setup(self):
        layers = [nn.Dense(features=128), jax.nn.relu, nn.Dense(features=64,), jax.nn.relu, nn.Dense(features=64,), jax.nn.relu, nn.Dense(features=626)]
        self.mlp = nn.Sequential(layers)


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)
    


class DiffModel(nn.Module):
    """
    Policy Network
    """

    n_features_list_prob: np.ndarray

    n_features_list_nodes: np.ndarray
    n_features_list_edges: np.ndarray
    n_features_list_messages: np.ndarray

    n_features_list_encode: np.ndarray
    n_features_list_decode: np.ndarray

    n_diffusion_steps: int
    n_message_passes: int
    edge_updates: bool
    problem_type: str

    time_encoding: str
    n_diff_steps: int
    embedding_dim: int = 32
    message_passing_weight_tied: bool = True
    linear_message_passing: bool = True
    n_bernoulli_features: int = 1
    mean_aggr: bool = False
    EncoderModel: str = "normal"
    n_random_node_features: int = 5
    train_mode: str = "REINFORCE"
    graph_norm: bool = False
    bfloat16: bool = False
    dataset_name: str = "None"
    mode: str = "edge"

    def setup(self):
        if self.bfloat16 == False:
            dtype = jnp.float32
        else:
            dtype = jnp.bfloat16

        GNNModel, HeadModel = get_GNN_model(self.EncoderModel, self.train_mode)
        if self.EncoderModel != "UNet":
            self.encode_process_decode = GNNModel(
                dtype=dtype,
                n_features_list_nodes=self.n_features_list_nodes,
                n_features_list_edges=self.n_features_list_edges,
                n_features_list_messages=self.n_features_list_messages,
                n_features_list_encode=self.n_features_list_encode,
                n_features_list_decode=self.n_features_list_decode,
                edge_updates=self.edge_updates,
                n_message_passes=self.n_message_passes,
                weight_tied=self.message_passing_weight_tied,
                linear_message_passing=self.linear_message_passing,
                mean_aggr=self.mean_aggr,
                graph_norm=self.graph_norm,
                mode=self.mode,
            )
        else:
            import re

            def extract_integer(input_string):
                # Use a regular expression to find digits at the end of the string
                match = re.search(r"\d+$", input_string)
                if match:
                    return int(match.group())
                else:
                    return None

            size = extract_integer(self.dataset_name)

            self.encode_process_decode = GNNModel(
                size=size,
                features=self.n_features_list_nodes[0],
                n_layers=self.n_message_passes,
                mode=self.mode,
            )

        self.HeadModel = HeadModel(
            n_features_list_prob=self.n_features_list_prob, dtype=dtype
        )

        self.__vmap_get_log_probs = jax.vmap(
            self.__get_log_prob, in_axes=(0, None, None), out_axes=(0)
        )
        self.vamp_get_sinusoidal_positional_encoding = jax.vmap(
            get_sinusoidal_positional_encoding, in_axes=(0, None, None)
        )
        ### TODO random node feature key is different during eval and sample, force them to be the same?
        self.time_embedder = nn.Embed(
            num_embeddings=200, features=self.n_random_node_features // 2
        )

        self.edge_embedder = nn.Embed(num_embeddings=626, features=64)
        self.node_embedder = nn.Embed(num_embeddings=250, features=64)

        self.edge_state_embedder = nn.Embed(num_embeddings=2, features=64)

        self.dummy_edge_mlp = DummyEdgeMLP()

    @flax.linen.jit
    def __call__(
        self, jraph_graph_list, X_prev, rand_edge_features_in, t_idx_per_node, key
    ):
        out_dict = {}
        t_embeddings = self.time_embedder(t_idx_per_node.astype(int)[0])
        pred = self.dummy_edge_mlp(jnp.concatenate((X_prev.T, t_embeddings), 1)).T
        out_dict["spin_logits"] = grouped_log_softmax(pred[:, None], jraph_graph_list, jraph_graph_list["graphs"][0].meta["n_groups"])
        # embeddings[:, None, :] shape: (num_edges/nodes, 1, embedding_dim)
        out_dict["rand_node_features"] = (
            rand_edge_features_in  # (11551, 1, 2) | (3151, 1, 2)
        )
        return out_dict, key

    # @partial(flax.linen.jit, static_argnums=0)
    def get_graph_info(self, jraph_graph_list):
        first_graph = jraph_graph_list["graphs"][0].graph
        nodes = first_graph.nodes
        edges = first_graph.edges
        n_node = first_graph.n_node
        n_edge = first_graph.n_edge
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        total_edges = jax.tree_util.tree_leaves(edges)[0].shape[0]
        node_graph_idx = jnp.repeat(
            graph_idx, n_node, axis=0, total_repeat_length=total_nodes
        )
        edge_graph_idx = jnp.repeat(
            graph_idx, n_edge, axis=0, total_repeat_length=total_edges
        )
        if self.mode == "edge":
            return edge_graph_idx, n_graph, n_edge
        else:
            return node_graph_idx, n_graph, n_node

    @partial(flax.linen.jit, static_argnums=0)
    def reinit_rand_nodes(self, X_t, key):
        key, subkey = jax.random.split(key)
        rand_nodes = jax.random.uniform(
            subkey, shape=(X_t.shape[0], self.n_random_node_features)
        )

        return rand_nodes, key

    @partial(flax.linen.jit, static_argnums=(0,))
    def _add_random_nodes_and_time_index(self, X_t, rand_nodes, t_idx_per_node):
        if self.time_encoding == "one_hot":
            X_embed = jax.nn.one_hot(
                jnp.squeeze(t_idx_per_node, axis=-1), num_classes=self.n_diffusion_steps
            )
        else:
            X_embed = self.vamp_get_sinusoidal_positional_encoding(
                jnp.squeeze(t_idx_per_node, axis=-1),
                self.embedding_dim,
                self.n_diff_steps,
            )

        X_one_hot = jax.nn.one_hot(X_t[..., 0], num_classes=self.n_bernoulli_features)

        X_input = jnp.concatenate([X_one_hot, X_embed, rand_nodes], axis=-1)
        return X_input

    @partial(flax.linen.jit, static_argnums=0)
    def make_one_step(self, params, jraph_graph_list, X_prev, t_idx_per_node, key, temp=1.0):
        rand_nodes, key = self.reinit_rand_nodes(X_prev, key)

        out_dict, key = self.apply(
            params, jraph_graph_list, X_prev, rand_nodes, t_idx_per_node, key
        )

        spin_logits = out_dict["spin_logits"]

        node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

        X_next, spin_log_probs, key = self.sample_from_model(
            spin_logits, jraph_graph_list["graphs"][0], key, temp
        )

        group0_per_graph = jax.ops.segment_sum((jraph_graph_list["graphs"][0].graph.globals["group_ids"] == 0).astype(jnp.int32), node_graph_idx,n_graph,)
        n_node -= group0_per_graph
        graph_log_prob = jax.lax.stop_gradient(
            jnp.exp(
                (
                    self.__get_log_prob(spin_log_probs[..., 0], node_graph_idx, n_graph)
                    / (n_node)
                )[:-1]
            )
        )

        #out_dict["pred_entropy"] = - spin_logits * jnp.exp(spin_logits)
        out_dict["X_next"] = X_next  # [3151, 1]
        out_dict["spin_log_probs"] = spin_log_probs  # [3151, 1]
        out_dict["state_log_probs"] = self.__get_log_prob(
            spin_log_probs[..., 0], node_graph_idx, n_graph
        )  # 31
        out_dict["graph_log_prob"] = graph_log_prob  # 30
        return out_dict, key

    @partial(flax.linen.jit, static_argnums=0)
    def unbiased_last_step(
        self, params, jraph_graph_list, X_prev, t_idx, key, eps=0.01
    ):
        assert False
        rand_nodes, key = self.reinit_rand_nodes(X_prev, key)
        out_dict, key = self.apply(
            params, jraph_graph_list, rand_nodes, X_prev, t_idx, key
        )

        spin_logits = out_dict["spin_logits"]
        group_ids = out_dict.get("group_ids", None)

        j_graphs = jraph_graph_list["graphs"][0]
        key, subkey = jax.random.split(key)

        sampled_p = jax.random.uniform(key, shape=(j_graphs.n_node.shape[0],))

        nodes = j_graphs.nodes
        n_node = j_graphs.n_node
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        graph_sampled_p = jnp.repeat(
            sampled_p, n_node, axis=0, total_repeat_length=total_nodes
        )
        graph_sampled_p = graph_sampled_p[:, None]

        X_next_model, spin_log_probs_model, key = self.sample_from_model(
            spin_logits, group_ids, key
        )
        X_next_uniform, one_hot_state, log_p_uniform_density, key = self.sample_prior(
            j_graphs, spin_logits.shape[1], key
        )
        X_next_uniform = X_next_uniform[..., 0]
        log_p_uniform = jnp.sum(log_p_uniform_density * one_hot_state, axis=-1)[..., 0]

        X_next = jnp.where(graph_sampled_p < eps, X_next_uniform, X_next_model)

        concat_spin_log_probs = jnp.concatenate(
            [spin_log_probs_model[None, ...], log_p_uniform[None, ...]], axis=0
        )
        weights = jnp.concatenate(
            [
                (1 - eps) * jnp.ones_like(spin_log_probs_model)[None, ...],
                eps * jnp.ones_like(spin_log_probs_model)[None, ...],
            ],
            axis=0,
        )
        spin_log_probs = jax.scipy.special.logsumexp(
            concat_spin_log_probs, axis=0, b=weights
        )

        node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

        graph_log_prob = jax.lax.stop_gradient(
            jnp.exp(
                (
                    self.__get_log_prob(
                        jnp.sum(spin_log_probs, axis=-1), node_graph_idx, n_graph
                    )
                    / (n_node)
                )[:-1]
            )
        )
        return X_next, spin_log_probs, spin_logits, graph_log_prob, key

    @partial(flax.linen.jit, static_argnums=0)
    def sample_from_model(self, spin_logits, graph, key, temp=1.0):
        key, subkey = jax.random.split(key)

        if spin_logits.shape[-1] == 2:
            X_next = jax.random.categorical(
                key=subkey, logits=spin_logits, axis=-1, shape=spin_logits.shape[:-1]
            )
            assert False
        else:
            if len(spin_logits.shape) == 2:
                spin_logits = spin_logits[..., None]
            X_next = groupwise_sample(subkey, spin_logits, graph.graph.globals["group_ids"], graph.meta["n_groups"], temp)
            # X_next = X_next[:, 0, 0]

        one_hot_state = jax.nn.one_hot(X_next[..., 0], num_classes=self.n_bernoulli_features)

        # X_next = jnp.expand_dims(X_next, axis = -1)
        spin_log_probs = jnp.sum(spin_logits * X_next, axis=-1)

        # print("Diff model model samples", X_next.shape, one_hot_state.shape)
        return X_next[..., 0], spin_log_probs, key  # [N, 1], [N, 1]

    @partial(flax.linen.jit, static_argnums=0)
    def calc_log_q(
        self, params, jraph_graph_list, X_prev, rand_nodes, X_next, t_idx_per_node, key
    ):
        out_dict, key = self.apply(
            params, jraph_graph_list, X_prev, rand_nodes, t_idx_per_node, key
        )

        spin_logits = out_dict["spin_logits"]
        node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

        one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
        # X_next = jnp.expand_dims(X_next, axis = -1)
        spin_log_probs = jnp.sum(spin_logits * X_next, axis=-1)
        # print(X_next.shape, X_next, jnp.exp(spin_log_probs))
        X_next_log_prob = self.__get_log_prob(
            spin_log_probs, node_graph_idx, n_graph
        )

        # graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
        # print("average prob T:0", jnp.mean(graph_log_prob))
        out_dict["state_log_probs"] = X_next_log_prob
        out_dict["spin_log_probs"] = spin_log_probs
        return out_dict, key

    @partial(flax.linen.jit, static_argnums=0)
    def calc_log_q_T(self, j_graph, X_T):
        """

        :param j_graph:
        :param X_T: shape =  (batched_graph_nodes, n_states, 1)
        :return:
        """
        j_graph = j_graph.graph
        shape = X_T.shape#[0:-1]
        log_p_uniform = self._get_prior(shape, j_graph)

        log_p_X_T_per_node = jnp.sum(log_p_uniform * X_T, axis=-1)

        nodes = j_graph.nodes
        edges = j_graph.edges
        n_node = j_graph.n_node
        n_edge = j_graph.n_edge
        n_graph = j_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        total_edges = jax.tree_util.tree_leaves(edges)[0].shape[0]
        edge_graph_idx = jnp.repeat(
            graph_idx, n_edge, axis=0, total_repeat_length=total_edges
        )

        log_p_X_T = self.__get_log_prob(log_p_X_T_per_node, edge_graph_idx, n_graph)

        # graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
        # print("average prob 0", jnp.mean(graph_log_prob))
        return log_p_X_T

    @partial(flax.linen.jit, static_argnums=(0, 2))
    def sample_prior(self, j_graph, N_basis_states, key):
        if self.mode == "edge":
            edges = j_graph.graph.edges
            shape = (edges.shape[0], N_basis_states, 1)
        else:
            nodes = j_graph.graph.nodes  # todo plassma: edge mode
            shape = (nodes.shape[0], N_basis_states, 1)

        key, subkey = jax.random.split(key)
        log_p_uniform = self._get_prior(shape, j_graph.graph) # todo plassma: contains inf

        #X_prev = jax.random.categorical(
        #    key=subkey, logits=log_p_uniform, axis=-1, shape=log_p_uniform.shape[:-1]
        #)
        X_prev = groupwise_sample(subkey, log_p_uniform, j_graph.graph.globals["group_ids"], j_graph.meta["n_groups"])

        one_hot_state = jax.nn.one_hot(X_prev, num_classes=self.n_bernoulli_features)
        return X_prev, one_hot_state, log_p_uniform, key

    @partial(flax.linen.jit, static_argnums=(0, 2))
    def sample_prior_w_probs(self, j_graph, N_basis_states, key):
        X_T, one_hot_state, log_p_uniform, key = self.sample_prior(
            j_graph, N_basis_states, key
        )
        log_p_X_T = self.calc_log_q_T(j_graph, X_T)
        return X_T, log_p_X_T, one_hot_state, log_p_uniform, key

    @partial(flax.linen.jit, static_argnums=(0, 1))
    def _get_prior(self, shape, j_graph):
        log_p_uniform = jnp.log(jnp.ones(shape) / j_graph.globals["group_counts"].reshape((shape[0],) + (1,) * (len(shape) - 1)))
        log_p_uniform = jnp.where((j_graph.globals["group_ids"] == 0)[..., None, None], 0, log_p_uniform)
        return log_p_uniform

    # @partial(flax.linen.jit, static_argnums=(0,-1))
    def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
        log_probs = self.__global_graph_aggr(spin_log_probs, node_graph_idx, n_graph)
        return log_probs

    # @partial(flax.linen.jit, static_argnums=(0,-1))
    def __global_graph_aggr(self, feature, node_graph_idx, n_graph):
        aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
        return aggr_feature


def get_sinusoidal_positional_encoding(timestep, embedding_dim, max_position):
    """
    Create a sinusoidal positional encoding as described in the
    "Attention is All You Need" paper.

    Args:
        timestep (int): The current time step.
        embedding_dim (int): The dimensionality of the encoding.

    Returns:
        A 1D tensor of shape (embedding_dim,) representing the
        positional encoding for the given timestep.
    """
    position = timestep
    div_term = jnp.exp(
        np.arange(0, embedding_dim, 2) * (-jnp.log(max_position) / embedding_dim)
    )
    return jnp.concatenate(
        [jnp.sin(position * div_term), jnp.cos(position * div_term)], axis=-1
    )
