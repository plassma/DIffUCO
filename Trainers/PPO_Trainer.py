import jax.numpy as jnp
import numpy as np
from functools import partial
import jax

from HCProblem_test import aug_solution_jax, sample_permutations

from .BaseTrainer import Base
from .ppo_utils import select_time_indices
import time
import optax
from utils import MovingAverages
from jax import nn
### TODO use RL environments to make it possible to project solutions onto feasible solutions!

class PPO(Base):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        super(PPO, self).__init__(config, EnergyClass, NoiseClass, model)
        self.N_basis_states = self.config["N_basis_states"]
        self.n_bernoulli_features = self.config["n_bernoulli_features"]
        self.n_diffusion_steps = self.config["n_diffusion_steps"]
        self.time_horizon = self.n_diffusion_steps

        ### TODO add these things to config
        k = self.config["TD_k"] # 3
        lam = np.exp(-np.log(k)/self.time_horizon) # float(np.round((1 - 1 / self.time_horizon), decimals=3))
        #lam = float(np.round((1 - 1 / self.time_horizon), decimals=3))
        self.lam = lam
        self.gamma = 1.
        self.inner_loop_steps = self.config["inner_loop_steps"]
        self.clip_value = self.config["clip_value"] # 0.2
        self.n_diff_bs = min([self.config["minib_diff_steps"], self.n_diffusion_steps])
        self.n_state_bs = min([self.config["minib_basis_states"], self.N_basis_states])
        self.inner_update_steps = int(self.n_diffusion_steps*self.N_basis_states/(self.n_diff_bs*self.n_state_bs))*self.inner_loop_steps
        self.c1 = self.config["value_weighting"] # 0.65
        self.proj_method = self.config["proj_method"]
        self.sample_multiplier = self.config.get("sample_multiplier", 1)

        self.calc_noise_step = self.NoiseDistrClass.calc_noise_step

        self.pmap_calc_traces = jax.pmap(self._calc_traces, in_axes=(0,0))
        self.pmap_environment_steps = jax.pmap(lambda a,b,c,d,e,f: self._environment_steps_scan(a,b,c,d,e,f, "train"), in_axes=(0, 0, 0, None, None, 0))
        self.pmap_environment_steps_force_samples = jax.pmap(lambda a,b,c,d,e,f,g: self._environment_steps_scan_force_samples(a,b,c,d,e,f,g, "train"), in_axes=(0, 0, 0, 0, None, None, 0))
        self._environment_steps_scan_eval = lambda a,b,c,d,e,f: self._environment_steps_scan(a,b,c,d,e,f, "eval")

        self.PPO_loss_grad = jax.jit(jax.value_and_grad(self.PPO_loss, has_aux=True))
        self.pmap_PPO_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, 0,0), axis_name="device")
        self.vmapped_calc_log_q = jax.vmap(self.model.calc_log_q, in_axes = (None, None, 0, 0, 0, 0, 0))

        self.alpha = self.config["mov_average"]
        ### TODO moving averages should ideally be loaded from checkpoint
        self.MovingAverageClass = MovingAverages.MovingAverage(self.alpha, self.alpha)

        self.n_diff_batches = self.n_diffusion_steps // self.n_diff_bs
        self.n_state_batches = self.N_basis_states // self.n_state_bs
        self.n_devices = len(jax.devices())
        self._init_index_arrays()

        self.best_buffer = None

    #@partial(jax.jit, static_argnums=(0,))
    def _init_index_arrays(self):
        self.n_graphs = self.config["n_graphs"] + 1 #int(self.config["batch_size"]/self.n_devices) + 1 todo plassma: hardcoded for now
        diff_step_arr = jnp.arange(0,self.n_diffusion_steps)
        Nb_arr = jnp.repeat(diff_step_arr[None, ...], self.N_basis_states, axis=0)
        Gb_Nb_arr = jnp.repeat(Nb_arr[None, ...], self.n_graphs, axis=0) #Nb_arr#

        self.Db_Gb_Nb_arr = jnp.repeat(Gb_Nb_arr[None, ...], self.n_devices, axis = 0)

        basis_state_arr = jnp.arange(0, self.N_basis_states)
        self.Db_basis_state_arr = jnp.repeat(basis_state_arr[None, ...], self.n_devices, axis=0)
        self.Db_basis_n_diff_state_arr = jnp.repeat(self.Db_basis_state_arr[...,None], self.n_diffusion_steps, axis=-1)
        print("TODO also add random permutation along graph batch size!")

    @partial(jax.jit, static_argnums=(0,))
    def _shuffle_index_array(self, key):
        key, subkey = jax.random.split(key)
        perm_diff_array = jax.random.permutation(subkey, self.Db_Gb_Nb_arr, axis=-1, independent=True)

        key, subkey = jax.random.split(key)
        perm_state_array = jax.random.permutation(subkey, self.Db_basis_n_diff_state_arr, axis=-2, independent=True)

        return perm_diff_array, perm_state_array, key

    @partial(jax.jit, static_argnums=(0,2,3))
    def _split_arrays(self, arr, n_splits, axis ):
        arr_list = jnp.split(arr, n_splits, axis=axis)
        return arr_list

    @partial(jax.jit, static_argnums=(0,))
    def loss_backward(self, params, opt_state, graphs, batch_dict, key):
        (loss, (log_dict, key)), grad = self.PPO_loss_grad(params, graphs, batch_dict, key)
        grad = jax.lax.pmean(grad, axis_name='device')
        params, opt_state = self.__update_params(params, grad, opt_state)
        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def __update_params(self, params, grads, opt_state):
        grad_update, opt_state = self.opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def sample(self, params, graphs, energy_graph_batch, T, key):
        (log_dict, _) =  self._environment_steps_scan_eval(params, graphs, energy_graph_batch, T, self.ownership_weight, key)
        loss = 0.
        ### TODO log reverse KL here?
        return loss, (log_dict, _)

    def sort_databuffer_by_value(self, databuffer: dict, keep_top: int = -1, shuffle=False, key=None):
        databuffer = jax.tree_util.tree_map(lambda x: x.copy(), databuffer)
        indices = jnp.argsort(-databuffer["energy_rewards"][0, 0])#jnp.argsort(databuffer["rewards"][0, -1, 0])
        if keep_top > 0:
            indices = indices[:keep_top]
        if shuffle:
            indices = jax.random.permutation(key, indices)
        for key in ["actions", "policies", "rewards", "states", "values", "bin_sequence"]:#"bin_sequence"
            databuffer[key] = databuffer[key][:, :, :, indices]
        databuffer["energy_rewards"] = databuffer["energy_rewards"][:, :, indices]
        return databuffer
    
    def merge_databuffers(self, A: dict, B: dict, shuffle=False, key=None):
        # Use the size of the sample axis (basis states) as weight
        sample_size_A = int(A["RL"]["energy_rewards"].shape[-1])
        sample_size_B = int(B["RL"]["energy_rewards"].shape[-1])
        total_samples = sample_size_A + sample_size_B
        sample_dims = {sample_size_A, sample_size_B}

        def _weighted_average(a, b):
            return (a * sample_size_A + b * sample_size_B) / total_samples

        def _find_sample_axis(a_shape, b_shape):
            if len(a_shape) != len(b_shape):
                return None
            candidate_axes = []
            for idx, (dim_a, dim_b) in enumerate(zip(a_shape, b_shape)):
                if (dim_a in sample_dims or dim_b in sample_dims) and all(
                        a_shape[j] == b_shape[j] for j in range(len(a_shape)) if j != idx):
                    candidate_axes.append(idx)
            return candidate_axes[-1] if candidate_axes else None

        def _merge_value(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                merged_dict = {}
                for k in set(a.keys()).union(b.keys()):
                    if k in a and k in b:
                        merged_dict[k] = _merge_value(a[k], b[k])
                    elif k in a:
                        merged_dict[k] = a[k]
                    else:
                        merged_dict[k] = b[k]
                return merged_dict

            if hasattr(a, "shape") and hasattr(b, "shape"):
                sample_axis = _find_sample_axis(a.shape, b.shape)
                if sample_axis is not None:
                    return jnp.concatenate([a, b], axis=sample_axis)
                if a.shape == b.shape:
                    try:
                        return _weighted_average(a, b)
                    except Exception:
                        return a
                try:
                    return _weighted_average(jnp.asarray(a), jnp.asarray(b))
                except Exception:
                    return a

            try:
                return _weighted_average(jnp.asarray(a), jnp.asarray(b))
            except Exception:
                return b

        merged = _merge_value(A, B)

        if shuffle:
            n_samples = merged["RL"]["energy_rewards"].shape[-1]
            perm_indices = jax.random.permutation(key, n_samples)

            def _find_axis_with_dim(shape, dim):
                axes = [i for i, d in enumerate(shape) if d == dim]
                return axes[-1] if axes else None

            def _shuffle(value):
                if isinstance(value, dict):
                    return {k: _shuffle(v) for k, v in value.items()}
                if hasattr(value, "shape"):
                    axis = _find_axis_with_dim(value.shape, n_samples)
                    if axis is not None and value.ndim > 1:
                        return jnp.take(value, perm_indices, axis=axis)
                return value

            merged = _shuffle(merged)

        return merged

    def train_step(self, params, opt_state, graphs, energy_graph_batch, T, key):

        start_env_step_time = time.time()

        out_dict = None

        for i in range(self.sample_multiplier): # self.sample_multiplier
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, num=len(jax.devices()))
            temp_out_dict, _ = self.pmap_environment_steps(params, graphs, energy_graph_batch, T, self.ownership_weight, batched_key)
            temp_out_dict["RL"]["bin_sequence"] = temp_out_dict["bin_sequence"]
            if out_dict is None:
                out_dict = temp_out_dict
            else:
                key, subkey = jax.random.split(key)
                out_dict = self.merge_databuffers(out_dict, temp_out_dict, shuffle=False and i == self.sample_multiplier - 1, key=subkey)
        
        # out_dict["RL"] = self.sort_databuffer_by_value(out_dict["RL"], keep_top=self.N_basis_states, shuffle=True, key=subkey)
        
        #permutations = sample_permutations(subkey, out_dict["RL"]["states"].shape[-2], graphs["graphs"][0].meta["rooms"])

        #trajectories = out_dict["bin_sequence"]
        #aug_trajectories = trajectories.astype(jnp.int32)#aug_solution_jax(trajectories, permutations, graphs["graphs"][0])
        #out_dict, _ = self.pmap_environment_steps_force_samples(params, graphs, energy_graph_batch, aug_trajectories, T, self.ownership_weight, batched_key,)
        
        #if self.best_buffer is not None:
        #    trajectories = self.best_buffer["bin_sequence"].astype(jnp.int32)
        #    best_out_dict, _ = self.pmap_environment_steps_force_samples(params, graphs, energy_graph_batch, trajectories, T, self.ownership_weight, batched_key,)
        #    best_out_dict["RL"]["bin_sequence"] = best_out_dict["bin_sequence"]
        #    key, subkey = jax.random.split(key)
        #    out_dict = self.merge_databuffers(out_dict, best_out_dict, shuffle=True, key=subkey)
        #self.best_buffer = self.sort_databuffer_by_value(out_dict["RL"], keep_top=int(self.N_basis_states * 0.3), shuffle=False)

        del out_dict["RL"]["energy_rewards"]

        #node_gr_idx = jnp.repeat(jnp.arange(graphs["graphs"][0].graph.n_node.shape[1]), graphs["graphs"][0].graph.n_node[0], axis=0, total_repeat_length=graphs["graphs"][0].graph.n_node.sum())
        #states_before_aug = flatten_states(out_dict["RL"]["states"])
        #energy_before_aug = self.vmapped_relaxed_energy(graphs["graphs"][0], states_before_aug, node_gr_idx)
        #out_dict["RL"]["states"] = aug_solution_jax(out_dict["RL"]["states"], permutations, graphs["graphs"][0])
        #states_after_aug = flatten_states(out_dict["RL"]["states"])
        #energy_after_aug = self.vmapped_relaxed_energy(graphs["graphs"][0], states_after_aug, node_gr_idx)
        #out_dict["RL"]["actions"] = aug_solution_jax(out_dict["RL"]["actions"], permutations, graphs["graphs"][0])
        end_env_step_time = time.time()

        overall_env_step_time = end_env_step_time - start_env_step_time


        if(False):
            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: x[0], energy_graph_batch)
            graphs_ = jax.tree_util.tree_map(lambda x: x[0], graphs["graphs"][0])
            params_no = jax.tree_util.tree_map(lambda x: x[0], params)

            s_cast = time.time()
            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), energy_graph_batch_copy)
            graphs_ = jax.tree_util.tree_map(lambda x: jnp.array(x), graphs_)
            e_cast = time.time()

            graph_list = {"graphs": [graphs_]}

            start_env_step_time = time.time()
            _, _ = self._environment_steps(params_no, graph_list, energy_graph_batch_copy, T, key)
            end_env_step_time = time.time()

            overall_env_step_time_2 = end_env_step_time - start_env_step_time
            print("pmap compare", overall_env_step_time_2, overall_env_step_time + e_cast - s_cast , e_cast-s_cast)

            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: x[0],energy_graph_batch)
            node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch_copy)
            # relaxed_energies_per_graph, _, Hb_per_graph = self.vmapped_relaxed_energy(energy_graph_batch_copy, out_dict["best_X_0"][0], node_gr_idx)
            #
            # Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
            #
            # print(jnp.any(out_dict["energies"]["HA"][0] == relaxed_energies_per_graph[:-1]))
            # print("Hb", out_dict["energies"]["Hb"], float(Hb))

            #print(Hb_per_graph[:-1])

            vmap_debug = jax.vmap(self.EnergyClass.calculate_CE_debug, in_axes=(None, 1, None))
            vmap_debug = self.vmapped_energy_CE

            reps = [2,4,6,8,10]
            for rep in reps:
                X_in = out_dict["X_0"][0,:, 0:rep]
                print("X_in", X_in.shape)

                s = time.time()
                _ = vmap_debug(energy_graph_batch_copy, X_in,node_gr_idx)
                e = time.time()

                print("time_needed", e-s, "reps", rep)

            # best_X_0, _, HB_per_graph_recal, HB_per_graph = vmap_debug(energy_graph_batch_copy, out_dict["X_0"][0], node_gr_idx)
            # print(jnp.mean(HB_per_graph_recal), jnp.mean(HB_per_graph), jnp.mean(out_dict["energies"]["Hb"]))
            # if (jnp.sum(HB_per_graph_recal) != jnp.sum(HB_per_graph)):
            #     print(jnp.sum(HB_per_graph_recal, jnp.sum(HB_per_graph)))
            #     raise ValueError("not equal")


            # relaxed_energies_per_graph, Hb_per_node, Hb_per_graph = self.vmapped_relaxed_energy(energy_graph_batch_copy,
            #                                                                           best_X_0,
            #                                                                           node_gr_idx)
            # Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
            #
            # print("is equal",jnp.mean(1*(best_X_0 == out_dict["best_X_0"][0])))
            # print("Hb")
            # print(Hb_per_graph[:-1])
            # print(Hb_per_graph[:-1].shape)
            # print(out_dict["X_0"].shape, best_X_0.shape, out_dict["X_0"][0].shape, "here")
            # print(relaxed_energies_per_graph.shape, Hb_per_node.shape, Hb_per_graph.shape)
            #
            # if(float(Hb) != 0):
            #     raise ValueError("")
        out_dict = self._calculate_advantages(out_dict)

        start_backprob_time = time.time()
        (loss, (log_dict, key)), params, opt_state = self._update_policy(params, opt_state, graphs, out_dict["RL"], key)
        end_backprob_time = time.time()
        # out_dict = jax.tree_util.tree_map(lambda x: np.array(x), out_dict)
        # self.__init_Dataloader(out_dict)
        overall_backprob_time = end_backprob_time - start_backprob_time

        for dict_key in log_dict["Losses"].keys():
            log_dict["Losses"][dict_key] = np.mean(log_dict["Losses"][dict_key])

        # for dict_key in log_dict["Losses"].keys():
        #     out_dict["Losses"][dict_key] = log_dict["Losses"][dict_key]
        out_dict.update(log_dict)

        time_dict = {"time": {"buffer_backprob": overall_backprob_time, "env_steps": overall_env_step_time}}
        out_dict.update(time_dict)

        return params, opt_state, loss, (out_dict, energy_graph_batch, key)

    #@partial(jax.jit, static_argnums=(0,))
    def _update_policy(self, params, opt_state, graphs, RL_buffer, key):

        log_dict = {"Losses": {"overall_loss": [], "critic_loss": [], "actor_loss": [], "max_ratios": [], "min_ratios": [], "mean_ratios": [], "perc_clipped": []}}

        for i in range(self.inner_loop_steps):
            perm_diff_array, perm_state_array, key = self._shuffle_index_array(key)
            split_diff_array_list = self._split_arrays(perm_diff_array, self.n_diff_batches, -1)
            split_state_array_list = self._split_arrays(perm_state_array, self.n_diff_batches, -1)

            for split_diff_arr, split_state_arr in zip(split_diff_array_list, split_state_array_list):
                split_split_diff_arr_list = self._split_arrays(split_diff_arr, self.n_state_batches, -2)
                split_split_state_arr_list = self._split_arrays(split_state_arr, self.n_state_batches, -2)
                for split_split_diff_arr, split_split_state_arr in zip(split_split_diff_arr_list, split_split_state_arr_list):
                    #batch_dict, key = select_time_indices(graphs["graphs"][0], RL_buffer, split_split_diff_arr, split_split_state_arr, key)
                    #key, subkey = jax.random.split(key)
                    #batched_key = jax.random.split(subkey, num=len(jax.devices()))
                    #(loss, (loss_dict, _)), params, opt_state = self.pmap_PPO_loss_backward(params, opt_state, graphs, batch_dict, batched_key)
                    (loss, (loss_dict, key)), params, opt_state = self.loop_inner(params, opt_state, graphs, RL_buffer, key, split_split_diff_arr, split_split_state_arr)

                    for dict_key in log_dict["Losses"].keys():
                        log_dict["Losses"][dict_key].append(loss_dict[dict_key])

        return (loss, (log_dict, key)), params, opt_state


    @partial(jax.jit, static_argnums=(0,))
    def loop_inner(self, params, opt_state, graphs, RL_buffer, key, split_diff_arr, split_state_arr):
        batch_dict, key = select_time_indices(graphs["graphs"][0].graph, RL_buffer, split_diff_arr, split_state_arr, key)
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))
        (loss, (loss_dict, _)), params, opt_state = self.pmap_PPO_loss_backward(params, opt_state, graphs, batch_dict, batched_key)
        return (loss, (loss_dict, key)), params, opt_state


    @partial(jax.jit, static_argnums=0, static_argnames=("deterministic", "force_samples"))
    def scan_body(self, scan_dict, y, deterministic: bool, force_samples: bool = False):
        i = scan_dict["step"]
        T = scan_dict["T"]
        params = scan_dict["params"]
        X_prev = scan_dict["X_prev"]
        X_next = scan_dict["Xs_over_different_steps"][i + 1]
        graphs = scan_dict["graphs"]
        node_gr_idx = scan_dict["node_gr_idx"]
        energy_graph_batch = scan_dict["energy_graph_batch"]

        energy_per_node = jnp.zeros(X_prev.shape[:-1]) #self.vmapped_relaxed_energy(energy_graph_batch, X_prev, node_gr_idx)[2] # (116, 20, 1)

        model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
        model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1), dtype=jnp.int16)

        key, subkey = jax.random.split(scan_dict["key"])
        scan_dict["key"] = key

        batched_key = jax.random.split(subkey, num=X_prev.shape[1])

        if force_samples:
            out_dict, _ = self._vmapped_make_one_step_force_samples(
                params,
                graphs,
                X_prev,
                X_next,
                energy_per_node,
                model_step_idx_per_node,
                batched_key,
                scan_dict["step"],
            )
        else:
            out_dict, _ = self.vmapped_make_one_step(
                params,
                graphs,
                X_prev,
                energy_per_node,
                model_step_idx_per_node,
                batched_key,
                deterministic,
                scan_dict["step"],
            )

        X_next = out_dict["X_next"]
        spin_log_probs = out_dict["spin_log_probs"]
        state_log_probs = out_dict["state_log_probs"]
        spin_logits_next = out_dict["spin_logits"]
        graph_log_prob = out_dict["graph_log_prob"]
        Values = out_dict["Values"]
        #rand_node_features = out_dict["rand_node_features"]

        #scan_dict["rand_node_features_diff_steps"] = scan_dict["rand_node_features_diff_steps"].at[i].set(rand_node_features)

        entropy_step = self._get_entropy_step(energy_graph_batch, state_log_probs, node_gr_idx)
        ### TODO is this still correct for annealed noise distr? Anneled reward should be given to step i-1?!
        scan_dict["noise_rewards"] = self._get_noise_distr_step(energy_graph_batch, X_prev, X_next, model_step_idx, node_gr_idx, T, scan_dict["noise_rewards"])

        X_prev = X_next
        scan_dict["Xs_over_different_steps"] = scan_dict["Xs_over_different_steps"].at[i + 1].set(X_next)

        scan_dict["entropy_rewards"] = scan_dict["entropy_rewards"].at[i].set(T * entropy_step)

        scan_dict["log_policies"] = scan_dict["log_policies"].at[i].set(state_log_probs)
        scan_dict["Values_over_diff_steps"] = scan_dict["Values_over_diff_steps"].at[i].set(Values)

        average_probs = jnp.mean(graph_log_prob)
        scan_dict["prob_over_diff_steps"] = scan_dict["prob_over_diff_steps"].at[i + 1].set(average_probs)

        scan_dict["X_prev"] = X_prev
        scan_dict["step"] += 1

        out_dict = {}
        out_dict["spin_log_probs"] = spin_log_probs
        out_dict["spin_logits_next"] = spin_logits_next
        return scan_dict, out_dict

    @partial(jax.jit, static_argnums=(0,7))
    def _environment_steps_scan(self, params, graphs, energy_graph_batch, T, ownership_floor, key, mode):
        ### TDOD cahnge rewards to non exact expectation rewards
        print("scan function is being jitted")
        if(mode == "train"):
            N_basis_states = self.N_basis_states // self.sample_multiplier
        else:
            N_basis_states = self.N_test_basis_states
        deterministic = mode != "train"

        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             N_basis_states,
                                                                                             key)

        n_graphs = energy_graph_batch.n_node.shape[0]
        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)

        log_policies = jnp.zeros((overall_diffusion_steps, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        Xs_over_different_steps = jnp.zeros((overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,), dtype=jnp.float32)
        noise_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        entropy_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        Values_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        rand_node_features_diff_steps = jnp.zeros(
            (overall_diffusion_steps, X_prev.shape[0], X_prev.shape[1], self.n_random_node_features), dtype=jnp.float32)

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(jnp.exp((jax.ops.segment_sum((log_p_uniform * one_hot_state).sum(-1), node_gr_idx, n_graphs) / energy_graph_batch.n_node[..., None])[:-1]).mean())
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)


        
        scan_dict = {"log_policies": log_policies, "Xs_over_different_steps": Xs_over_different_steps, "prob_over_diff_steps": prob_over_diff_steps, "noise_rewards": noise_rewards, "entropy_rewards": entropy_rewards,
                    "Values_over_diff_steps": Values_over_diff_steps, "rand_node_features_diff_steps":rand_node_features_diff_steps,
                    "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev, "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T}

        scan_fn = partial(self.scan_body, deterministic=deterministic)
        scan_dict, out_dict_list = jax.lax.scan(scan_fn, scan_dict, None, length = overall_diffusion_steps)


        key = scan_dict["key"]
        spin_log_probs = out_dict_list["spin_log_probs"][-1]
        spin_logits_next = out_dict_list["spin_logits_next"][-1]

        solution_spins_last_step = (nn.one_hot(graphs["graphs"][0].graph.globals["solution_nodes"], num_classes=graphs["graphs"][0].meta["cabinets"])[:, None, None] * jax.lax.stop_gradient(spin_logits_next)).sum(-1)
        solution_prob_last_step = jax.ops.segment_sum(solution_spins_last_step, node_gr_idx, graphs["graphs"][0].graph.n_node.shape[0])[:-1]
        solution_prob_min_factor = jax.ops.segment_min(solution_spins_last_step, node_gr_idx, graphs["graphs"][0].graph.n_node.shape[0])[:-1]

        X_next = scan_dict["X_prev"]#
        energy_step, Hb, best_X_0, key, energy_dict = self._get_energy_step(energy_graph_batch, X_next, node_gr_idx, ownership_floor, key)
        energy_reward = -energy_step

        noise_rewards = scan_dict["noise_rewards"]
        entropy_rewards = scan_dict["entropy_rewards"]
        combined_reward = self.NoiseDistrClass.calculate_noise_distr_reward(-noise_rewards, entropy_rewards)

        rewards = combined_reward
        rewards = rewards.at[-1].set(rewards[-1] + energy_reward)

        graph_energies = energy_step[:-1]
        graph_energies = graph_energies[..., None]
        X_0 = X_next

        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]
        rand_node_features_diff_steps = scan_dict["rand_node_features_diff_steps"]
        Values_over_diff_steps = scan_dict["Values_over_diff_steps"]
        log_policies = scan_dict["log_policies"]
        entropy_rewards = scan_dict["entropy_rewards"]
        prob_over_diff_steps = scan_dict["prob_over_diff_steps"]
        noise_rewards = scan_dict["noise_rewards"]

        log_q_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_p_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        log_q_0_T = log_q_0_T.at[1:].set(log_policies)
        log_p_0_T = log_p_0_T.at[:-1].set(-1/(T* 10**-6)*noise_rewards)
        log_p_0_T = log_p_0_T.at[-1].set(1/(T* 10**-6)*energy_step)

        forward_KL = self._forward_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])
        reverse_KL = self._reverse_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])

        x_axis = jnp.arange(0, overall_diffusion_steps)
        log_dict = {"RL": {"states": Xs_over_different_steps[0:-1], "rand_node_features": rand_node_features_diff_steps,
                           "actions": Xs_over_different_steps[1:], "policies": log_policies, "rewards": rewards,
                           "values": Values_over_diff_steps, "energy_rewards": energy_reward[:-1],},
                    "Losses": {"forward_KL": forward_KL, "reverse_KL": reverse_KL, "noise_rewards": jnp.mean(jnp.sum(noise_rewards[0:, :-1], axis=0)),
                               "energy_rewards": jnp.mean(energy_reward[:-1]),
                               "neg_entropy_rewards": jnp.mean(jnp.sum(entropy_rewards[0:, :-1], axis=0)),
                               "overall_rewards": jnp.mean(jnp.sum(rewards[0:, :-1], axis=0))},
                    "metrics": {"energies": graph_energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "graph_mean_energies": energy_reward, "free_energies": 0., "solution_prob_mean": solution_prob_last_step.mean(), "solution_prob_min": solution_prob_min_factor.mean()},
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "overall_rewards_over_diff_steps": {"x_values": x_axis,
                                                                    "y_values": jnp.mean(
                                                                        jnp.mean(rewards[:, :-1], axis=-1), axis=-1)},
                                "entropy_rewards": {"x_values": x_axis,
                                                    "y_values": jnp.mean(jnp.mean(entropy_rewards[:, :-1], axis=-1),
                                                                         axis=-1)},
                                "noise_rewards": {"x_values": x_axis,
                                                  "y_values": jnp.mean(jnp.mean(noise_rewards[:, :-1], axis=-1),
                                                                       axis=-1)}
                                },
                    "energies": {"HA": graph_energies, "Hb": Hb, **energy_dict},
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "best_X_0": best_X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    }
        return log_dict, key
    
    @partial(jax.jit, static_argnums=(0,8))
    def _environment_steps_scan_force_samples(self, params, graphs, energy_graph_batch, Xs_over_different_steps, T, ownership_floor, key, mode):
        ### TDOD cahnge rewards to non exact expectation rewards
        print("scan function is being jitted")
        N_basis_states = Xs_over_different_steps.shape[2]
        deterministic = mode != "train"

        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             N_basis_states,
                                                                                             key)
        X_prev = Xs_over_different_steps[0]

        n_graphs = energy_graph_batch.n_node.shape[0]

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)

        log_policies = jnp.zeros((overall_diffusion_steps, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,), dtype=jnp.float32)
        noise_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        entropy_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        Values_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        rand_node_features_diff_steps = jnp.zeros(
            (overall_diffusion_steps, X_prev.shape[0], X_prev.shape[1], self.n_random_node_features), dtype=jnp.float32)

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(jnp.exp((jax.ops.segment_sum((log_p_uniform * one_hot_state).sum(-1), node_gr_idx, n_graphs) / energy_graph_batch.n_node[..., None])[:-1]).mean())
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)


        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        scan_dict = {"log_policies": log_policies, "Xs_over_different_steps": Xs_over_different_steps, "prob_over_diff_steps": prob_over_diff_steps, "noise_rewards": noise_rewards, "entropy_rewards": entropy_rewards,
                    "Values_over_diff_steps": Values_over_diff_steps, "rand_node_features_diff_steps":rand_node_features_diff_steps,
                    "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev, "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T}

        scan_fn = partial(self.scan_body, force_samples=True, deterministic=deterministic)
        scan_dict, out_dict_list = jax.lax.scan(scan_fn, scan_dict, None, length = overall_diffusion_steps)


        key = scan_dict["key"]
        spin_log_probs = out_dict_list["spin_log_probs"][-1]
        spin_logits_next = out_dict_list["spin_logits_next"][-1]

        solution_spins_last_step = (nn.one_hot(graphs["graphs"][0].graph.globals["solution_nodes"], num_classes=graphs["graphs"][0].meta["cabinets"])[:, None, None] * jax.lax.stop_gradient(spin_logits_next)).sum(-1)
        solution_prob_last_step = jax.ops.segment_sum(solution_spins_last_step, node_gr_idx, graphs["graphs"][0].graph.n_node.shape[0])[:-1]
        solution_prob_min_factor = jax.ops.segment_min(solution_spins_last_step, node_gr_idx, graphs["graphs"][0].graph.n_node.shape[0])[:-1]

        X_next = scan_dict["X_prev"]#
        energy_step, Hb, best_X_0, key, energy_dict = self._get_energy_step(energy_graph_batch, X_next, node_gr_idx, ownership_floor, key)
        energy_reward = -energy_step

        noise_rewards = scan_dict["noise_rewards"]
        entropy_rewards = scan_dict["entropy_rewards"]
        combined_reward = self.NoiseDistrClass.calculate_noise_distr_reward(-noise_rewards, entropy_rewards)

        rewards = combined_reward
        rewards = rewards.at[-1].set(rewards[-1] + energy_reward)

        graph_energies = energy_step[:-1]
        graph_energies = graph_energies[..., None]
        X_0 = X_next

        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]
        rand_node_features_diff_steps = scan_dict["rand_node_features_diff_steps"]
        Values_over_diff_steps = scan_dict["Values_over_diff_steps"]
        log_policies = scan_dict["log_policies"]
        entropy_rewards = scan_dict["entropy_rewards"]
        prob_over_diff_steps = scan_dict["prob_over_diff_steps"]
        noise_rewards = scan_dict["noise_rewards"]

        log_q_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_p_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        log_q_0_T = log_q_0_T.at[1:].set(log_policies)
        log_p_0_T = log_p_0_T.at[:-1].set(-1/(T* 10**-6)*noise_rewards)
        log_p_0_T = log_p_0_T.at[-1].set(1/(T* 10**-6)*energy_step)

        forward_KL = self._forward_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])
        reverse_KL = self._reverse_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])

        x_axis = jnp.arange(0, overall_diffusion_steps)
        log_dict = {"RL": {"states": Xs_over_different_steps[0:-1], "rand_node_features": rand_node_features_diff_steps,
                           "actions": Xs_over_different_steps[1:], "policies": log_policies, "rewards": rewards,
                           "values": Values_over_diff_steps, "energy_rewards": energy_reward[:-1],},
                    "Losses": {"forward_KL": forward_KL, "reverse_KL": reverse_KL, "noise_rewards": jnp.mean(jnp.sum(noise_rewards[0:, :-1], axis=0)),
                               "energy_rewards": jnp.mean(energy_reward[:-1]),
                               "neg_entropy_rewards": jnp.mean(jnp.sum(entropy_rewards[0:, :-1], axis=0)),
                               "overall_rewards": jnp.mean(jnp.sum(rewards[0:, :-1], axis=0))},
                    "metrics": {"energies": graph_energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "graph_mean_energies": energy_reward, "free_energies": 0., "solution_prob_mean": solution_prob_last_step.mean(), "solution_prob_min": solution_prob_min_factor.mean()},
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "overall_rewards_over_diff_steps": {"x_values": x_axis,
                                                                    "y_values": jnp.mean(
                                                                        jnp.mean(rewards[:, :-1], axis=-1), axis=-1)},
                                "entropy_rewards": {"x_values": x_axis,
                                                    "y_values": jnp.mean(jnp.mean(entropy_rewards[:, :-1], axis=-1),
                                                                         axis=-1)},
                                "noise_rewards": {"x_values": x_axis,
                                                  "y_values": jnp.mean(jnp.mean(noise_rewards[:, :-1], axis=-1),
                                                                       axis=-1)}
                                },
                    "energies": {"HA": graph_energies, "Hb": Hb, **energy_dict},
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "best_X_0": best_X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    }
        return log_dict, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_noise_distr_step(self, jraph_graph, X_prev, X_next, t_idx, node_gr_idx, T,  noise_rewards_arr):
        noise_rewards_arr = self.calc_noise_step( jraph_graph, X_prev, X_next, t_idx, node_gr_idx, T, noise_rewards_arr)
        return noise_rewards_arr

    @partial(jax.jit, static_argnums=(0,))
    def _get_noise_distr_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        Noise_Energy_per_graph = self.calc_noise_step_relaxed( jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx)
        return (-1)*Noise_Energy_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def _get_entropy_step(self, jraph_graph, state_log_probs, node_gr_idx):
        return -state_log_probs

    @partial(jax.jit, static_argnums=(0,))
    def _get_entropy_step_relaxed(self, jraph_graph, spin_logits, node_gr_idx):
        assert False
        log_probs_down = spin_logits[..., 0]
        log_probs_up = spin_logits[..., 1]
        probs_up = jnp.exp(log_probs_up)
        probs_down = jnp.exp(log_probs_down)

        entropy_term_1 = -probs_up * log_probs_up
        entropy_term_2 = -probs_down * log_probs_down
        entropy_term_per_node = entropy_term_1 + entropy_term_2

        n_graph = jraph_graph.n_node.shape[0]
        relaxed_entropies_per_graph = jax.ops.segment_sum(jnp.sum(entropy_term_per_node, axis=-1, keepdims=True),
                                                          node_gr_idx, n_graph)
        return relaxed_entropies_per_graph[...,0]

    @partial(jax.jit, static_argnums=(0,))
    def _get_energy_step(self, jraph_graph, X_0, node_gr_idx, ownership_floor, key):
        if(self.proj_method == "feasible"):
            best_X_0, Hb, relaxed_energies_per_graph = self.vmapped_energy_feasible(jraph_graph, X_0)
            Hb = jnp.mean(jnp.abs(Hb))
        elif(self.proj_method == "CE"):
            best_X_0, relaxed_energies_per_graph, Hb_per_graph = self.vmapped_energy_CE(jraph_graph, X_0, node_gr_idx)
            Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
        else:
            relaxed_energies_per_graph, energy_dict, Hb_per_graph = self.vmapped_relaxed_energy(jraph_graph, X_0, node_gr_idx, ownership_floor)
            best_X_0 = X_0
            Hb = jnp.mean(jnp.abs(Hb_per_graph)[:-1])
            energy_dict = {k: v[:-1] for k, v in energy_dict.items()}

        return relaxed_energies_per_graph[...,0], Hb, best_X_0, key, energy_dict

    @partial(jax.jit, static_argnums=(0,))
    def _get_energy_reward_relaxed(self, jraph_graph, spin_logits, node_gr_idx):
        relaxed_energies_per_graph, _, _ = self.vmapped_relaxed_energy_for_Loss(
            jraph_graph, spin_logits, node_gr_idx, self.ownership_weight
        )
        return relaxed_energies_per_graph[...,0]



    ### TODO add scanner
    @partial(jax.jit, static_argnums=(0,))
    def _calc_traces(self, values, rewards):
        max_steps = self.n_diffusion_steps
        advantage = jnp.zeros_like(values)
        for t in range(max_steps):
            idx = max_steps - t - 1
            delta = rewards[idx] + self.gamma * values[idx + 1] - values[idx]
            advantage = advantage.at[idx].set(delta + self.gamma * self.lam * advantage[idx + 1])

        value_target = (advantage + values)[0:max_steps]
        return value_target, advantage[0:max_steps]

    def _calculate_advantages(self, log_dict):

        rewards = log_dict["RL"]["rewards"]
        reduced_rewards = rewards[:,:,:-1]

        mov_average_reward, mov_std_reward =  self.MovingAverageClass.update_mov_averages(reduced_rewards)
        normed_rewards = self.MovingAverageClass.calculate_average(rewards, mov_average_reward, mov_std_reward)

        log_dict["RL"]["normed_rewards"] = normed_rewards
        log_dict["energies"]["normed_rewards"] = jnp.swapaxes(normed_rewards[:,:,:-1], 1, 2)
        log_dict["energies"]["reduced_rewards"] = jnp.swapaxes(reduced_rewards, 1, 2)
        log_dict["energies"]["mov_average_reward"] = mov_average_reward
        log_dict["energies"]["mov_std_reward"] = mov_std_reward

        value_target, advantages = self.pmap_calc_traces(log_dict["RL"]["values"], log_dict["RL"]["normed_rewards"])
        normed_advantages = self._normalize_advantages(advantages)
        log_dict["RL"]["value_target"] = value_target
        log_dict["RL"]["advantages"] = normed_advantages
        return log_dict

    @partial(jax.jit, static_argnums=(0,))
    def _normalize_advantages(self, advantages):
        unpadded_adv = advantages[:,:,:-1]
        normed_advantages = (advantages - jnp.mean(unpadded_adv))/(jnp.std(unpadded_adv)+10**-10)
        return normed_advantages

    def get_loss(self, params, jraph_graph_list, batch_dict, key):
        return self.PPO_loss(params, jraph_graph_list, batch_dict, key)

    @partial(jax.jit, static_argnums=(0,))
    def PPO_loss(self, params, jraph_graph_list, batch_dict, key):
        Sb_Hb_Nb_A_k = batch_dict["advantages"]
        Sb_Hb_Nb_X_prev = batch_dict["states"]
        Sb_Hb_Nb_rand_node_features = batch_dict["rand_node_features"]
        Sb_Hb_Nb_X_next = batch_dict["actions"]

        Sb_Nb_t_idx_per_node = batch_dict["time_index_per_node"]
        Sb_Hb_Nb_state_log_probs = batch_dict["policies"]
        Sb_Hb_Nb_rtg = batch_dict["value_target"]

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=Sb_Hb_Nb_A_k.shape[0])

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(jraph_graph_list["graphs"][0])
        #energy_per_node = self.vmapped_relaxed_energy(jraph_graph_list["graphs"][0], Sb_Hb_Nb_X_prev.swapaxes(0,1).astype(jnp.int32), node_gr_idx)[2]
        #out_dict, _ = self.vmapped_calc_log_q(params, jraph_graph_list, Sb_Hb_Nb_X_prev, energy_per_node.T, Sb_Hb_Nb_X_next, Sb_Nb_t_idx_per_node, batched_key)
        out_dict, _ = self.vmapped_calc_log_q(params, jraph_graph_list, Sb_Hb_Nb_X_prev, Sb_Hb_Nb_rand_node_features, Sb_Hb_Nb_X_next, Sb_Nb_t_idx_per_node, batched_key)

        out_values = out_dict["Values"]
        state_log_probs = out_dict["state_log_probs"]

        ratios = jnp.exp(state_log_probs - Sb_Hb_Nb_state_log_probs)

        surr1 = ratios * Sb_Hb_Nb_A_k
        ### TODO replace it with KL diergence?
        surr2 = jnp.clip(ratios, 1 - self.clip_value, 1 + self.clip_value) * Sb_Hb_Nb_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)[:,:-1]))
        critic_loss = jnp.mean((out_values - Sb_Hb_Nb_rtg)[:,:-1] ** 2)
        overall_loss = (1-self.c1) * actor_loss + self.c1 *critic_loss

        max_ratios = jnp.max(ratios)
        min_ratios = jnp.min(ratios)
        mean_ratios = jnp.mean(ratios)
        clip_fraction = jnp.mean(1*(ratios < 1-self.clip_value)+1*(ratios > 1+self.clip_value))


        loss_dict = {"actor_loss": actor_loss, "critic_loss": critic_loss, "overall_loss": overall_loss,
                     "max_ratios": max_ratios, "min_ratios": min_ratios, "mean_ratios": mean_ratios, "perc_clipped": clip_fraction}
        return overall_loss, (loss_dict, key)
