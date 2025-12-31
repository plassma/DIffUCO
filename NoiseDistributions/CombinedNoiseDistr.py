from NoiseDistributions.OrdinalNoise import OrdinalNoiseDistr
from .BaseNoise import BaseNoiseDistr
from .AnnealedNoise import AnnealedNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial

class CombinedNoiseDistr(BaseNoiseDistr):

    def __init__(self, config):
        super().__init__(config)
        ### TODO init AnnealedNoiseDistr
        self.AnnealedNoiseDist = AnnealedNoiseDistr(self.config)
        self.OrdinalNoise = OrdinalNoiseDistr(self.config)
        ### TODO init CatNoiseDistr

        self.n_bernoilli_features = config["n_bernoulli_features"]

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        return -T * L_entropy + L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        return -(noise_distr_step - entropy_reward)

    @partial(jax.jit, static_argnums=(0,))
    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        log_p_T_0_ord = self.OrdinalNoise.get_log_p_T_0(jraph_graph, X_prev, X_next, t_idx, T)
        log_p_T_0_anneal = self.AnnealedNoiseDist.get_log_p_T_0(jraph_graph, X_prev, X_next, t_idx, T)
        res = log_p_T_0_ord + log_p_T_0_anneal
        return res

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, t_idx, node_gr_idx, T):
        ordinal_noise_energy, _ = self.OrdinalNoise.calc_noise_loss(jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, t_idx, node_gr_idx, T)
        annealed_noise_energy, _ = self.AnnealedNoiseDist.calc_noise_loss(jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, t_idx, node_gr_idx, T)
        combined_noise_energy = ordinal_noise_energy + annealed_noise_energy
        log_p_sum = jnp.sum(log_p_prev_per_node, axis=0)
        return combined_noise_energy, log_p_sum

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        raise ValueError("I think this function is stale")

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        ### TODO exchange beta_t with time_step?
        noise_rewards_arr = self.OrdinalNoise.calc_noise_step(jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr)
        noise_rewards_arr = self.AnnealedNoiseDist.calc_noise_step(jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr)
        return noise_rewards_arr

    def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
        log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
        return log_probs
