import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial
from Networks.Modules.Transformer.TransformerEncoderStack import TransformerEncoderStack
from Networks.Modules.Transformer.LinearTransformerEncoderStack import LinearTransformerEncoderStack
from Networks.Modules.xLSTM.mlstm import mLSTMStack
from house_config import prior_logits_for_graph

from Networks.Modules import get_GNN_model
from Networks.Modules.HeadModules.RLHead import global_graph_aggr
from Networks.Modules.MLPModules.MLPs import ValueMLP
from house_config.utils import CABINETS, ROOMS, THINGS

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
	dim_energy_per_node: int = 1
	train_mode: str = "REINFORCE"
	graph_norm: bool = False
	bfloat16: bool = False
	dataset_name: str = "None"
	augment_rooms: bool = False
	node_emb_type: str = "neighbor"
	node_transformer_num_layers: int = 4
	node_transformer_num_heads: int = 4
	node_transformer_dropout_rate: float = 0.05
	transformer_type: str = "linear"  # "linear" or "standard"



	def setup(self):
		if(self.bfloat16 == False):
			dtype = jnp.float32
		else:
			dtype = jnp.bfloat16

		GNNModel, HeadModel = get_GNN_model(self.EncoderModel, self.train_mode)
		if(self.EncoderModel != "UNet"):
			self.encode_process_decode = GNNModel(dtype = dtype, n_features_list_nodes=self.n_features_list_nodes,
																 n_features_list_edges=self.n_features_list_edges,
																 n_features_list_messages=self.n_features_list_messages,
																 n_features_list_encode=self.n_features_list_encode,
																 n_features_list_decode=self.n_features_list_decode,
																 edge_updates=self.edge_updates,
																 n_message_passes=self.n_message_passes,
																 weight_tied=self.message_passing_weight_tied,
																 linear_message_passing=self.linear_message_passing,
																 mean_aggr = self.mean_aggr,
																 graph_norm = self.graph_norm)
		else:
			import re
			def extract_integer(input_string):
				# Use a regular expression to find digits at the end of the string
				match = re.search(r'\d+$', input_string)
				if match:
					return int(match.group())
				else:
					return None
			size = extract_integer(self.dataset_name)

			self.encode_process_decode = GNNModel(size = size, features=self.n_features_list_nodes[0],
																 n_layers=self.n_message_passes
																 )
		self.feature_proj = nn.Sequential([
			nn.Dense(features=self.embedding_dim * 4, dtype=dtype),
			nn.gelu,
			nn.Dense(features=self.embedding_dim, dtype=dtype),
			nn.LayerNorm(dtype=dtype)
		])

		self.node_transformer = TransformerEncoderStack(
	            num_layers=self.node_transformer_num_layers,
	            embed_dim=self.embedding_dim,
	            mlp_dim=self.embedding_dim * 4,
	            num_heads=self.node_transformer_num_heads,
	            dropout_rate=self.node_transformer_dropout_rate,
	            dtype=dtype,
	        )
		
		self.linear_node_transformer = LinearTransformerEncoderStack(
	            num_layers=self.node_transformer_num_layers,
	            embed_dim=self.embedding_dim,
	            mlp_dim=self.embedding_dim * 4,
	            num_heads=self.node_transformer_num_heads,
	            dropout_rate=self.node_transformer_dropout_rate,
	            dtype=dtype,
	        )
		
		self.mLSTMStack = mLSTMStack(
			num_layers=self.node_transformer_num_layers,
			embed_dim=self.embedding_dim,
			num_heads=self.node_transformer_num_heads,
			causal=False,
		)

		self.HeadModel = HeadModel(n_features_list_prob=self.n_features_list_prob, dtype = dtype)

		self.__vmap_get_log_probs = jax.vmap(self.__get_log_prob, in_axes=(0, None, None), out_axes=(0))
		self.vmap_get_sinusoidal_positional_encoding = jax.vmap(get_sinusoidal_positional_encoding, in_axes=(0, None))
		### TODO random node feature key is different during eval and sample, force them to be the same?

		self.W_k = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.W_q = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.W_v = nn.Sequential([nn.Dense(features=self.embedding_dim * 2, dtype=dtype), nn.gelu, nn.Dense(features=self.embedding_dim, dtype=dtype)])
		self.value_mlp = ValueMLP(n_features_list=[self.embedding_dim * 4, self.embedding_dim * 2, 1], dtype=dtype)

		self.time_step_emb = nn.Embed(num_embeddings=self.n_diffusion_steps, features=8, dtype=dtype)
		self.node_type_emb = nn.Embed(num_embeddings=5, features=8, dtype=dtype)
		self.nth_of_type_emb = nn.Embed(num_embeddings=500, features=32, dtype=dtype)
		self.connection_emb = [
			nn.Embed(num_embeddings=500, features=self.embedding_dim, dtype=dtype), # rooms -> things
			nn.Embed(num_embeddings=50, features=self.embedding_dim, dtype=dtype), # cabinets -> rooms
			nn.Embed(num_embeddings=100, features=self.embedding_dim, dtype=dtype), # things -> cabinets
			nn.Embed(num_embeddings=50, features=self.embedding_dim, dtype=dtype)  # things -> persons
		]


	@partial(flax.linen.jit, static_argnums=(0,), static_argnames=("deterministic",))
	def __call__(self, jraph_graph_list, X_prev, energy_per_node, t_idx_per_node, key, *, deterministic: bool = True, pred_type: int = 0):
		X_prev = X_prev.astype(jnp.int32)
		X_prev_emb, node_types_emb, nth_of_type_emb, key = self.embed_nodes(X_prev, energy_per_node, t_idx_per_node, jraph_graph_list["graphs"][0], key)
		X_prev_emb = jnp.concatenate([X_prev_emb], axis = -1)

		#pred_type_emb = self.pred_type_emb(jnp.full(X_prev.shape[0], pred_type, dtype=jnp.int32))
		#embeddings = self.encode_process_decode(jraph_graph_list, X_prev)
		# Choose between linear attention and standard transformer based on configuration
		if self.transformer_type == "linear":
			embeddings = self.linear_node_transformer(self.feature_proj(X_prev_emb), deterministic=deterministic)  # deterministic=deterministic
		elif self.transformer_type == "standard":
			embeddings = self.node_transformer(self.feature_proj(X_prev_emb))  # deterministic=deterministic
		elif self.transformer_type == "mlstm":
			embeddings = self.mLSTMStack(self.feature_proj(X_prev_emb)[None])[0]
		else:
			raise ValueError(f"Unknown transformer_type: {self.transformer_type}")

		embeddings = jnp.concat([embeddings, node_types_emb, nth_of_type_emb], axis = -1)

		bernoulli_embeddings = jnp.repeat(embeddings[:, jnp.newaxis, :], 1, axis = -2)
		embeddings_aranged_for_nodes = embeddings[jraph_graph_list["graphs"][0].globals["neighbours_per_node"]]

		queries = self.W_q(embeddings)
		keys = self.W_k(embeddings_aranged_for_nodes)
		values = self.W_v(embeddings_aranged_for_nodes)

		scores = jnp.einsum('nd,ncd->nc', queries, keys) / jnp.sqrt(queries.shape[-1])
		out_dict = {}
		out_dict = self.HeadModel(jraph_graph_list, bernoulli_embeddings, out_dict)
		#out_dict["spin_logits"].shape (71, 1, 2)
		#out_dict["Values"].shape (2,)

		mask = self.get_mask(jraph_graph_list["graphs"][0])
		mask_bool = mask.astype(bool)
		neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
		masked_scores = jnp.where(mask_bool, scores, neg_inf)
		all_masked = jnp.all(~mask_bool, axis=-1, keepdims=True)
		safe_scores = jnp.where(all_masked, jnp.zeros_like(masked_scores), masked_scores)
		spin_logits = jax.nn.log_softmax(safe_scores, axis=-1)
		spin_logits = jnp.where(mask_bool, spin_logits, neg_inf)
		spin_logits = jnp.where(all_masked, neg_inf, spin_logits)[:, None]
		#spin_logits = self._undo_spin_logits_permutation(spin_logits, mapping)

		#fix ownership logits constant
		start_ownerships = jraph_graph_list["graphs"][0].meta["offset_things_persons"]
		end_ownerships = start_ownerships + jraph_graph_list["graphs"][0].meta["things"]
		spin_logits = spin_logits.at[start_ownerships:end_ownerships].set(neg_inf)
		rows = jnp.arange(start_ownerships, end_ownerships)
		cols = X_prev[start_ownerships:end_ownerships, 0].astype(jnp.int32)
		spin_logits = spin_logits.at[rows, 0, cols].set(0.0)

		attn_weights = jnp.exp(spin_logits[:, 0, :])
		attn_weights = attn_weights / jnp.clip(jnp.sum(attn_weights, axis=-1, keepdims=True), a_min=1e-9)
		attn_out = jnp.einsum('nc,ncd->nd', attn_weights, values)
		
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)
		value_emb = global_graph_aggr(attn_out[:, None], node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])
		Values = self.value_mlp(value_emb)[..., 0, 0]


		#spin_logits = jax.nn.log_softmax(scores + (1 - self.get_mask(jraph_graph_list["graphs"][0])) * -1e9)[:, None]
		out_dict = {
			"spin_logits": spin_logits,
			"Values": Values
		}
		return out_dict, key

	#@partial(flax.linen.jit, static_argnums=0)
	def get_graph_info(self, jraph_graph_list):
		first_graph = jraph_graph_list["graphs"][0]
		nodes = first_graph.nodes
		n_node = first_graph.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
		return node_graph_idx, n_graph, n_node

	@partial(flax.linen.jit, static_argnums=0)
	def reinit_rand_nodes(self, X_t,  key):
		key, subkey = jax.random.split(key)
		rand_nodes = jax.random.uniform(subkey, shape=(X_t.shape[0], self.dim_energy_per_node))

		return rand_nodes, key

	def get_connections_per_node(self, X_t, jraph_graph):
		connections_per_node = jnp.zeros_like(X_t)
		connections_per_node = connections_per_node.at[jraph_graph.meta["offset_rooms"]: jraph_graph.meta["offset_rooms"] + jraph_graph.meta["rooms"], 0].add(jnp.bincount(X_t[jraph_graph.meta["offset_cabinets"]: jraph_graph.meta["offset_cabinets"] + jraph_graph.meta["cabinets"], 0], length=jraph_graph.meta["rooms"]))
		connections_per_node = connections_per_node.at[jraph_graph.meta["offset_cabinets"]: jraph_graph.meta["offset_cabinets"] + jraph_graph.meta["cabinets"], 0].add(things_per_cabinet:=jnp.bincount(X_t[jraph_graph.meta["offset_things_cabinets"]: jraph_graph.meta["offset_things_cabinets"] + jraph_graph.meta["things"], 0], length=jraph_graph.meta["cabinets"]))
		connections_per_node = connections_per_node.at[jraph_graph.meta["offset_things_cabinets"]: jraph_graph.meta["offset_things_cabinets"] + jraph_graph.meta["things"], 0].add(things_per_cabinet[X_t[jnp.arange(jraph_graph.meta["things"]) + jraph_graph.meta["offset_things_cabinets"], 0]])
		
		return connections_per_node

	@partial(flax.linen.jit, static_argnums=(0,))
	def embed_nodes(self, X_t, energy_per_node, t_idx_per_node, jraph_graph, key):
		dtype = jnp.bfloat16 if self.bfloat16 else jnp.float32
		t_idx = jnp.squeeze(t_idx_per_node, axis=-1)
		t_idx = jnp.clip(t_idx, 0, self.n_diff_steps - 1)
		
		if(self.time_encoding == "one_hot"):
			T_embed = jax.nn.one_hot(t_idx, num_classes=self.n_diffusion_steps)
		elif (self.time_encoding == "learned"):
			T_embed = self.time_step_emb(t_idx.astype(jnp.int32)).astype(dtype)
		else:
			T_embed = self.vmap_get_sinusoidal_positional_encoding(t_idx, self.embedding_dim // 4).astype(dtype)
		
		#n_nodes_connected_to_node = self.get_connections_per_node(X_t, jraph_graph)
		#n_nodes_connected_to_node_emb = self.vamp_get_sinusoidal_positional_encoding(n_nodes_connected_to_node[:, 0], 8, 512).astype(dtype)

		node_types_emb = self.node_type_emb(jraph_graph.globals["node_types"])#self.vmap_get_sinusoidal_positional_encoding(node_types, self.embedding_dim // 4).astype(dtype) # todo plassma: max position hardcoded for now

		nth_of_type_emb = self.vmap_get_sinusoidal_positional_encoding(jraph_graph.globals["nth_of_type"], 16).astype(dtype) # self.nth_of_type_emb(jraph_graph.globals["nth_of_type"])

		energy_per_node_emb = self.vmap_get_sinusoidal_positional_encoding((energy_per_node.squeeze()).astype(jnp.int32), 8).astype(dtype)
		
		X_input = jnp.concatenate([T_embed, node_types_emb, nth_of_type_emb, ], axis=-1) # we don't need rand_nodes! # energy_per_node_emb

		if self.node_emb_type == "one_hot":
			X_emb = jax.nn.one_hot(X_t[..., 0], num_classes=jraph_graph.meta["cabinets"])
		elif self.node_emb_type == "neighbor":
			idx = jnp.array(jraph_graph.globals["neighbours_per_node"])[jnp.arange(X_t.shape[0]), X_t[:, 0].astype(jnp.int32)]
			X_emb = X_input[idx]
		elif self.node_emb_type == "learned":
			X_emb = jnp.zeros((X_t.shape[0], self.embedding_dim + 16), dtype=dtype)
			slices = [slice(jraph_graph.meta["offset_rooms"], jraph_graph.meta["offset_rooms"] + jraph_graph.meta["rooms"]), 
					  slice(jraph_graph.meta["offset_cabinets"], jraph_graph.meta["offset_cabinets"] + jraph_graph.meta["cabinets"]), 
					  slice(jraph_graph.meta["offset_things_cabinets"], jraph_graph.meta["offset_things_cabinets"] + jraph_graph.meta["things"]), 
					  slice(jraph_graph.meta["offset_things_persons"], jraph_graph.meta["offset_things_persons"] + jraph_graph.meta["persons"])]
			for i in range(4):
				X_emb = X_emb.at[slices[i]].set(jnp.concatenate([self.connection_emb[i](X_t[slices[i], 0].astype(jnp.int32)), self.vmap_get_sinusoidal_positional_encoding(X_t[slices[i], 0], 16)], axis=-1))
		else:
			X_emb = self.vmap_get_sinusoidal_positional_encoding(X_t[..., 0], self.embedding_dim)

		X_emb = X_emb.astype(dtype)
		X_input = jnp.concatenate([X_input, X_emb, ], axis=-1).astype(dtype) # energy_per_node_emb

		
		return X_input, node_types_emb, nth_of_type_emb, key

	@partial(flax.linen.jit, static_argnums=0, static_argnames=("deterministic",))
	def make_one_step(self,params ,jraph_graph_list, X_prev, energy_per_node, t_idx_per_node, key, deterministic: bool = True, step: int = 0):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)

		rngs = None
		if not deterministic:
			key, dropout_key = jax.random.split(key)
			rngs = {"dropout": dropout_key}
		
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		out_dict, key = self.apply(params, jraph_graph_list, X_prev, energy_per_node, t_idx_per_node, key, deterministic=deterministic, rngs=rngs)
		X_next, spin_log_probs, key = self.sample_from_model(out_dict["spin_logits"], jraph_graph_list, key)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node))[:-1]))
		out_dict["X_next"] = X_next
		out_dict["spin_log_probs"] = spin_log_probs
		out_dict["state_log_probs"] = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)
		out_dict["graph_log_prob"] = graph_log_prob
		return out_dict, key
	
	@partial(flax.linen.jit, static_argnums=0)
	def make_one_step_force_samples(self,params ,jraph_graph_list, X_prev, X_next, energy_per_node, t_idx_per_node, key, step: int = 0):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)

		
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		out_dict, key = self.apply(params, jraph_graph_list, X_prev, energy_per_node, t_idx_per_node, key, deterministic=True)
		X_next, spin_log_probs = self.force_sample_from_model(out_dict["spin_logits"], X_next)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node))[:-1]))
		out_dict["X_next"] = X_next
		out_dict["spin_log_probs"] = spin_log_probs
		out_dict["state_log_probs"] = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)
		out_dict["graph_log_prob"] = graph_log_prob
		return out_dict, key
	
	@partial(flax.linen.jit, static_argnums=0)
	def unbiased_last_step(self,params ,jraph_graph_list, X_prev, t_idx, key, eps = 0.01):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)
		out_dict, key = self.apply(params, jraph_graph_list, rand_nodes, X_prev, t_idx, key)

		spin_logits = out_dict["spin_logits"]
		j_graphs = jraph_graph_list["graphs"][0]
		key, subkey = jax.random.split(key)

		sampled_p = jax.random.uniform(key, shape =  (j_graphs.n_node.shape[0],))

		nodes = j_graphs.nodes
		n_node = j_graphs.n_node
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		graph_sampled_p = jnp.repeat(sampled_p, n_node, axis=0, total_repeat_length=total_nodes)
		graph_sampled_p = graph_sampled_p[:, None]

		X_next_model, spin_log_probs_model, key = self.sample_from_model(spin_logits, key)
		X_next_uniform, one_hot_state, log_p_uniform_density, key = self.sample_prior(j_graphs, spin_logits.shape[1],  key)
		X_next_uniform = X_next_uniform[...,0]
		log_p_uniform = jnp.sum(log_p_uniform_density * one_hot_state, axis=-1)[...,0]

		X_next = jnp.where(graph_sampled_p < eps, X_next_uniform, X_next_model)

		concat_spin_log_probs = jnp.concatenate([spin_log_probs_model[None,...], log_p_uniform[None, ...]], axis = 0)
		weights =  jnp.concatenate([(1-eps)*jnp.ones_like(spin_log_probs_model)[None,...], eps*jnp.ones_like(spin_log_probs_model)[None, ...]], axis = 0)
		spin_log_probs = jax.scipy.special.logsumexp(concat_spin_log_probs, axis = 0, b = weights)

		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(jnp.sum(spin_log_probs, axis = -1), node_graph_idx, n_graph)/(n_node))[:-1]))
		return X_next, spin_log_probs, spin_logits, graph_log_prob, key

	@partial(flax.linen.jit, static_argnums=0)
	def sample_from_model(self, spin_logits, jraph_graph_list, key):
		key, subkey = jax.random.split(key)
		X_next = jax.random.categorical(key=subkey,
											   logits=spin_logits,
											   axis=-1,
											   shape=spin_logits.shape[:-1])


		one_hot_state = jax.nn.one_hot(X_next, num_classes=jraph_graph_list["graphs"][0].meta["cabinets"])
		#X_next = jnp.expand_dims(X_next, axis = -1)
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)

		#print("Diff model model samples", X_next.shape, one_hot_state.shape)
		return X_next, spin_log_probs, key
	@partial(flax.linen.jit, static_argnums=0)
	def force_sample_from_model(self, spin_logits, X_next):
		one_hot_state = jax.nn.one_hot(X_next, num_classes=spin_logits.shape[-1])
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		return X_next, spin_log_probs
	
	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q(self, params, jraph_graph_list, X_prev, energy_per_node, X_next, t_idx_per_node, key):
		out_dict, key = self.apply(params, jraph_graph_list, X_prev, energy_per_node, t_idx_per_node, key)

		spin_logits = out_dict["spin_logits"]
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
		#X_next = jnp.expand_dims(X_next, axis = -1)
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		#print(X_next.shape, X_next, jnp.exp(spin_log_probs))
		X_next_log_prob = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)

		# graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
		# print("average prob T:0", jnp.mean(graph_log_prob))
		out_dict["state_log_probs"] = X_next_log_prob
		out_dict["spin_log_probs"] = spin_log_probs
		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q_T(self, j_graph, X_T):
		'''

		:param j_graph:
		:param X_T: shape =  (batched_graph_nodes, n_states, 1)
		:return:
		'''

		shape = X_T.shape
		log_p_uniform = self._get_prior(shape, j_graph, soft=True)

		one_hot_state = jax.nn.one_hot(X_T[..., 0], num_classes=j_graph.meta["cabinets"])
		log_p_X_T_per_node = jnp.sum(log_p_uniform * one_hot_state, axis=-1)


		nodes = j_graph.nodes
		n_node = j_graph.n_node
		n_graph = j_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		log_p_X_T = self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)

		# graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
		# print("average prob 0", jnp.mean(graph_log_prob))
		return log_p_X_T

	def sample_prior(self, j_graph, N_basis_states, key):
		nodes = j_graph.nodes
		shape = (nodes.shape[0], N_basis_states, 1)
		key, subkey = jax.random.split(key)
		log_p_uniform = self._get_prior(shape, j_graph)

		X_prev = jax.random.categorical(key=subkey,
										logits=log_p_uniform,
										axis=-1,
										shape=log_p_uniform.shape[:-1])[..., None]

		one_hot_state = jax.nn.one_hot(X_prev[..., 0], num_classes=j_graph.meta["cabinets"])
		return X_prev, one_hot_state, log_p_uniform, key

	def sample_prior_w_probs(self, j_graph, N_basis_states, key):
		X_T, one_hot_state, log_p_uniform, key = self.sample_prior(j_graph, N_basis_states, key)
		log_p_X_T = self.calc_log_q_T(j_graph, X_T)
		return X_T, log_p_X_T, one_hot_state, log_p_uniform, key

	@partial(flax.linen.jit, static_argnums=(0,))
	def get_mask(self, meta_graph):
		mask = (jnp.arange(meta_graph.meta["cabinets"])[None, :] < meta_graph.graph.globals["classes_per_node"][:, None])
		return mask * 1.0

	@partial(flax.linen.jit, static_argnums=(0,1))
	def _get_prior(self, shape, meta_graph, soft=False):
		base_logits = prior_logits_for_graph(meta_graph, soft=soft)
		target_shape = shape[:-1] + (base_logits.shape[-1],)
		return jnp.broadcast_to(base_logits[:, None, :], target_shape)

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
		log_probs = self.__global_graph_aggr(spin_log_probs, node_graph_idx, n_graph)
		return log_probs

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __global_graph_aggr(self, feature, node_graph_idx, n_graph):
		aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
		return aggr_feature


def get_sinusoidal_positional_encoding(timestep, embedding_dim, max_position=10000.0):
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
	dtype = jnp.float32
	position = jnp.asarray(timestep, dtype=dtype)
	max_position_safe = jnp.maximum(jnp.asarray(max_position, dtype=dtype), dtype(1.0))
	div_term = jnp.exp(jnp.arange(0, embedding_dim, 2, dtype=dtype) * (-jnp.log(max_position_safe) / embedding_dim))
	angles = position[..., None] * div_term
	return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
