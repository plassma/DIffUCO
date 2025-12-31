from .BaseNoise import BaseNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial


class OrdinalNoiseDistr(BaseNoiseDistr):
    def __init__(self, config):
        super().__init__(config)
        self.n_bernoilli_features = config["n_bernoulli_features"]
        self.ordinal_sigma = config.get("ordinal_sigma", 1.0)

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        return -T * L_entropy + T * L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        return -(noise_distr_step - entropy_reward)

    def _transition_log_probs(self, x_prev, beta_t):
        """Return log transition probabilities for a single ordinal value."""
        num_states = self.n_bernoilli_features
        states = jnp.arange(num_states)
        sigma = self.ordinal_sigma
        x_prev = jnp.asarray(x_prev, dtype=jnp.int32)

        safe_change_mass = jnp.where(num_states <= 1, 0.0, beta_t)
        stay_prob = jnp.where(num_states <= 1, 1.0, 1 - safe_change_mass)

        distances = states - x_prev
        weights = jnp.exp(-0.5 * (distances / sigma) ** 2)
        weights = weights.at[x_prev].set(0.0)

        weight_sum = jnp.sum(weights)
        safe_denominator = jnp.where(weight_sum > 0, weight_sum, 1.0)
        normalized_weights = jnp.where(
            weight_sum > 0,
            weights / safe_denominator,
            jnp.ones_like(weights) / jnp.maximum(num_states - 1, 1),
        )

        change_probs = safe_change_mass * normalized_weights
        transition_probs = change_probs.at[x_prev].set(stay_prob)
        transition_probs = jnp.clip(transition_probs, a_min=1e-12)

        return jnp.log(transition_probs)

    def _log_transition_probabilities(self, X_prev, X_next, beta_t):
        """
        Calculate log transition probabilities for an ordinal variable where nearby states are more likely.
        A discretized Gaussian centered at X_prev distributes the total change mass beta_t across other states.
        """
        def log_prob_single(x_prev, x_next):
            log_transition_probs = self._transition_log_probs(x_prev, beta_t)
            return log_transition_probs[x_next]

        vmapped_log_prob = jax.vmap(log_prob_single)
        flat_log = vmapped_log_prob(X_prev.reshape(-1), X_next.reshape(-1))
        return flat_log.reshape(X_prev.shape)

    @partial(jax.jit, static_argnums=(0,))
    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        gamma_t = self.get_gamma_t(t_idx)
        beta_t = 2 * gamma_t
        log_p_i = self._log_transition_probabilities(X_prev, X_next, beta_t)

        noise_per_node = jnp.sum(log_p_i, axis=-1)
        log_p_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        return log_p_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, model_step_idx, node_gr_idx, T):
        gamma_t = self.beta_arr[model_step_idx]
        beta_t = 2 * gamma_t
        probs_next = jax.nn.softmax(spin_logits_next, axis=-1)

        X_prev_flat = X_prev[..., 0].reshape(-1)
        probs_next_flat = probs_next.reshape(-1, probs_next.shape[-1])

        def expected_log_prob(x_prev, prob_next):
            log_transition_probs = self._transition_log_probs(x_prev, beta_t)
            return jnp.sum(prob_next * log_transition_probs)

        vmapped_expected_log_prob = jax.vmap(expected_log_prob)
        expected_log = vmapped_expected_log_prob(X_prev_flat, probs_next_flat)
        expected_log = expected_log.reshape(X_prev[..., 0].shape)

        noise_per_node = expected_log
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        return T * noise_per_graph, jnp.sum(log_p_prev_per_node, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        raise ValueError("relaxed does not make sense here")

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        gamma_t = self.beta_arr[model_step_idx]
        reward_idx = model_step_idx
        beta_t = 2 * gamma_t

        log_p_i = self._log_transition_probabilities(X_prev, X_next, beta_t)

        noise_per_node = jnp.sum(log_p_i, axis=-1)
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        noise_step_value = -T * noise_per_graph
        noise_rewards_arr = noise_rewards_arr.at[reward_idx].set(noise_rewards_arr[reward_idx] - noise_step_value)
        return noise_rewards_arr

    def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
        log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
        return log_probs
