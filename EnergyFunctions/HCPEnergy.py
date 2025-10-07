from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import igraph as ig
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import plot_graph
from Networks.DiffModel import groupwise_sample

def show_graph(graph_batch, bins):
    edges = [(graph_batch["graphs"][0].senders[0,i], graph_batch["graphs"][0].receivers[0,i]) for i,e in enumerate(bins) if e and graph_batch["graphs"][0].senders[0,i] != graph_batch["graphs"][0].receivers[0,i]]
    graph = ig.Graph(edges=edges)
    plot_graph(graph, graph_batch["graphs"][0].globals[0])

ROOMS = 0
CABINETS = 1
THINGS = 2
PERSONS = 3

def repeat_to_gr_idx(counts: list[int]):
    return jnp.repeat(jnp.arange(len(counts)), jnp.array(counts), axis=0, total_repeat_length=sum(counts))


@jax.jit
def step_window_squared_cost(cabinets_x_things: jnp.ndarray) -> jnp.ndarray:
    """
    cabinets_x_things: [C, T]
      - columns can be one-hot (discrete) or soft (sum≈1).
    Returns scalar cost.
    """
    C, T = cabinets_x_things.shape

    t = jnp.arange(T)  # [T]

    # Step interval per thing (even partition of things across cabinets)
    lower = (t * C) // T                                   # floor(t*C/T)        [T]
    upper = jnp.ceil((t + 1) * C / T).astype(int) - 1      # ceil((t+1)C/T)-1   [T]

    # Broadcast cabinet indices
    c = jnp.arange(C)[:, None]  # [C,1]

    # Squared distance outside the interval; zero inside
    left_violation  = jnp.clip(lower[None, :] - c, a_min=0)      # [C,T]
    right_violation = jnp.clip(c - upper[None, :], a_min=0)      # [C,T]

    penalty = left_violation**2 + right_violation**2              # [C,T]

    # Weight by assignment (hard or soft) and sum
    return jnp.sum(cabinets_x_things * penalty)

class HCPEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        self.vmapped_argmax_energy = jax.vmap(self.calculate_Energy, in_axes=(None, 1, None))
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, meta_graph, sample, node_gr_idx, A = 1., B = 1.2):
        '''
        This method assumes that no edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''
        H_graph = meta_graph.graph
        edge_gr_idx = node_gr_idx[H_graph.senders]
        n_graph = max(H_graph.n_node.shape)

        sample = sample.squeeze()

        #solution_hamming = jnp.array([jnp.sum(sample != H_graph.globals["solution"]), 0])[:, None]
        #return solution_hamming, {"solution_hamming": solution_hamming}, solution_hamming
        #dummy_easy_energy = (sample * meta_graph.graph.globals["nth_of_group"] * (meta_graph.graph.globals["group_ids"] != 0).astype(jnp.int32))
        #dummy_easy_energy = jax.ops.segment_sum(dummy_easy_energy, edge_gr_idx, n_graph)[..., None]
        #return dummy_easy_energy, {"dummy_easy_energy": dummy_easy_energy}, dummy_easy_energy # todo: not even this very easy dummy energy can be optimized to 0 -> is NN/Head broken?´
        
        n_node = max(H_graph.nodes.shape)
        node_types = H_graph.globals["node_types"].squeeze() # faulty shape: (1, N)

        # sample = H_graph.globals["solution"]

        senders = H_graph.senders.squeeze()
        receivers = H_graph.receivers.squeeze()


        thing_index = jnp.where((node_types == THINGS), H_graph.globals["nth_of_type"].squeeze(), -1)[receivers]
        cabinet_index = jnp.where((node_types == CABINETS), H_graph.globals["nth_of_type"].squeeze(), -1)[senders]
        cabinets_x_things = jnp.zeros((meta_graph.meta["cabinets"] + 1, meta_graph.meta["things"] + 1))
        cabinets_x_things = cabinets_x_things.at[cabinet_index, thing_index].add(sample)
        cabinets_x_things = cabinets_x_things[:-1, :-1]

        energy_things_per_cabinet = jnp.array([((jax.nn.relu(cabinets_x_things.sum(1) - 5))**2).sum(),0])[:, None]

        #thing_idx = jnp.arange(meta_graph.meta["things"])
        #cabinet_idx = jnp.arange(meta_graph.meta["cabinets"])

        #thing_idx = (thing_idx[None, :].repeat(meta_graph.meta["cabinets"], axis=0))
        #cabinet_idx = (cabinet_idx[:, None].repeat(meta_graph.meta["things"], axis=1))

        #diff = thing_idx - cabinet_idx

        # aux_things_per_cabinet = jnp.array([(cabinets_x_things.sum(1) ** 2).sum(),0])[:, None]


        #order_violations old:
        #prefix_incl = jax.lax.cumsum(cabinets_x_things, axis=1)
        #prefix_excl = jnp.pad(prefix_incl[:, :-1], ((0, 0), (1, 0))) > 0
        #violations_mask = jnp.logical_and(cabinets_x_things[:-1, :].astype(jnp.bool), prefix_excl[1:, :])
        #order_violations = jnp.array([violations_mask.sum(), 0])[:, None]


        # Per row, count how many items have index < t (exclusive prefix count across things)
        # (jax.lax.cumsum has no 'exclusive' kwarg, so we subtract x to make it exclusive)
        row_prefix_lt = jnp.cumsum(cabinets_x_things, axis=1) - cabinets_x_things  # shape [C, T]

       # Compute cumulative sum over later cabinets: flip -> cumsum -> flip -> subtract own row
        rev_cumsum = jnp.flip(jnp.cumsum(jnp.flip(row_prefix_lt, axis=0), axis=0), axis=0)
        #later_row_prefix_lt = 1. * (rev_cumsum - row_prefix_lt).astype(bool)
        later_row_prefix_lt = rev_cumsum - row_prefix_lt

        # For every (i,t) that is actually present, add how many later-cabinet items have index < t
        order_violations = jnp.sum(cabinets_x_things * later_row_prefix_lt)
        order_violations = jnp.array([order_violations, 0])[:, None]

        #order_violations = step_window_squared_cost(cabinets_x_things)
        #order_violations = jnp.array([order_violations, 0])[:, None]

        
        

        owners_of_things_edges = jnp.where((node_types[senders] == THINGS) & (node_types[receivers] == PERSONS) & sample, receivers, 0) # faulty senders shape: (1, N)
        owners_of_things_nodes = jax.ops.segment_max(owners_of_things_edges, H_graph.senders, n_node)
        owners_of_things_nodes = jnp.where(owners_of_things_nodes < 0, 0, owners_of_things_nodes)

        owners_of_rooms_edges = jnp.where((node_types[receivers] == PERSONS) & (node_types[senders] == ROOMS) & sample, receivers, 0)
        owners_of_rooms_nodes = jax.ops.segment_max(owners_of_rooms_edges, H_graph.senders, n_node)
        owners_of_rooms_nodes = jnp.where(owners_of_rooms_nodes < 0, 0, owners_of_rooms_nodes)

        owners_of_cabinets_edges = jnp.where((node_types[senders] == ROOMS) & (node_types[receivers] == CABINETS) & sample, owners_of_rooms_nodes[senders], 0)
        owners_of_cabinets_nodes = jax.ops.segment_max(owners_of_cabinets_edges, H_graph.receivers, n_node)
        owners_of_cabinets_nodes = jnp.where(owners_of_cabinets_nodes < 0, 0, owners_of_cabinets_nodes)

        persons_of_things_edges = jnp.where((node_types[senders] == CABINETS) & (node_types[receivers] == THINGS) & sample, owners_of_cabinets_nodes[senders], 0)
        persons_of_things_nodes = jax.ops.segment_max(persons_of_things_edges, H_graph.receivers, n_node)
        persons_of_things_nodes = jnp.where(persons_of_things_nodes < 0, 0, persons_of_things_nodes)

        energy_ownerships = jax.ops.segment_sum((owners_of_things_nodes != persons_of_things_nodes).astype(jnp.float32), node_gr_idx, n_graph)[..., None]

        energy = energy_ownerships + order_violations + energy_things_per_cabinet# todo: can order violations be formulated more monotonically?

        return energy, {"energy_ownerships": energy_ownerships, "order_violations": order_violations, "things_per_cabinet": energy_things_per_cabinet}, energy


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        assert False
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)
        

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, key=None, temp=1.): # logits shape: (626, 1)
       return self.calculate_Energy(H_graph["graphs"][0], logits, node_gr_idx)
        