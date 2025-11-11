from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import igraph as ig
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import plot_graph
from Networks.DiffModel import groupwise_sample


ROOMS = 0
CABINETS = 1
THINGS = 2
PERSONS = 3



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
        room_index = jnp.where((node_types == ROOMS), H_graph.globals["nth_of_type"].squeeze(), -1)[senders]
        cabinet_receivers_index = jnp.where((node_types == CABINETS), H_graph.globals["nth_of_type"].squeeze(), -1)[receivers]
        cabinets_x_things = jnp.zeros((meta_graph.meta["cabinets"] + 1, meta_graph.meta["things"] + 1))
        cabinets_x_things = cabinets_x_things.at[cabinet_index, thing_index].add(sample)
        cabinets_x_things = cabinets_x_things[:-1, :-1]

        cabinet_gr_idx = jnp.repeat(jnp.arange(len(meta_graph.meta["cabinets_concat"])), jnp.array(meta_graph.meta["cabinets_concat"]), total_repeat_length=meta_graph.meta["cabinets"])
        energy_things_per_cabinet = jax.ops.segment_sum(jnp.abs(cabinets_x_things.sum(1) - 5), cabinet_gr_idx, len(meta_graph.meta["cabinets_concat"]) + 1)

        rooms_x_cabinets = jnp.zeros((meta_graph.meta["rooms"] + 1, meta_graph.meta["cabinets"] + 1))
        rooms_x_cabinets = rooms_x_cabinets.at[room_index, cabinet_receivers_index].add(sample)
        rooms_x_cabinets = rooms_x_cabinets[:-1, :-1]

        rooms_gr_idx = jnp.repeat(jnp.arange(len(meta_graph.meta["rooms_concat"])), jnp.array(meta_graph.meta["rooms_concat"]), total_repeat_length=meta_graph.meta["rooms"])
        cabinets_per_room = jax.ops.segment_sum(jnp.abs(rooms_x_cabinets.sum(1) - 2), rooms_gr_idx, len(meta_graph.meta["rooms_concat"]) + 1)



        # Per row, count how many items have index < t (exclusive prefix count across things)
        # (jax.lax.cumsum has no 'exclusive' kwarg, so we subtract x to make it exclusive)
        row_prefix_lt = jnp.cumsum(cabinets_x_things, axis=1) - cabinets_x_things  # shape [C, T]

       # Compute cumulative sum over later cabinets: flip -> cumsum -> flip -> subtract own row
        rev_cumsum = jnp.flip(jnp.cumsum(jnp.flip(row_prefix_lt, axis=0), axis=0), axis=0)
        #later_row_prefix_lt = 1. * (rev_cumsum - row_prefix_lt).astype(bool)
        later_row_prefix_lt = rev_cumsum - row_prefix_lt

        # For every (i,t) that is actually present, add how many later-cabinet items have index < t
        order_violations = (cabinets_x_things * later_row_prefix_lt).sum(-1)
        order_violations = jax.ops.segment_sum(order_violations, cabinet_gr_idx, len(meta_graph.meta["cabinets_concat"]) + 1)

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

        energy_ownerships = jax.ops.segment_sum((owners_of_things_nodes != persons_of_things_nodes).astype(jnp.float32), node_gr_idx, n_graph)

        things_per_graph = jnp.array(meta_graph.meta["things_concat"] + [1])
        rooms_per_graph = jnp.array(meta_graph.meta["rooms_concat"] + [1])

        c = 1

        #energy_ownerships = jnp.where(((c * order_violations / H_graph.n_edge) + c * energy_things_per_cabinet / things_per_graph + c * cabinets_per_room / rooms_per_graph) == 0, energy_ownerships, meta_graph.meta["things"])

        energy = (energy_ownerships / things_per_graph + (c * order_violations / H_graph.n_edge) + c * energy_things_per_cabinet / things_per_graph + c * cabinets_per_room / rooms_per_graph)
        
        return energy, {"energy_ownerships": energy_ownerships, "order_violations": order_violations, "things_per_cabinet": energy_things_per_cabinet, "cabinets_per_room": cabinets_per_room}, energy


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        assert False
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)
        

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, key=None, temp=1.): # logits shape: (626, 1)
       return self.calculate_Energy(H_graph["graphs"][0], logits, node_gr_idx)
        
