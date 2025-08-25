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

class HCPEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, sample, node_gr_idx, A = 1., B = 1.2):
        '''
        This method assumes that no edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''
        edge_gr_idx = node_gr_idx[H_graph.senders]
        n_graph = max(H_graph.n_node.shape)
        n_node = max(H_graph.nodes.shape)
        node_types = H_graph.globals["node_types"].squeeze() # faulty shape: (1, N)

        # sample = H_graph.globals["solution"]

        senders = H_graph.senders.squeeze()
        receivers = H_graph.receivers.squeeze()


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

        energy = jax.ops.segment_sum((owners_of_things_nodes != persons_of_things_nodes).astype(jnp.float32), node_gr_idx, n_graph)[..., None] + 0.0001

        return energy, {"dict_energy": energy}, energy


    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_logits(self, meta_graph, logits, node_gr_idx, A = 1., B = 1.2):
        '''
        Calculate energy directly from logits without sampling.
        Replaces segment_max operations with weighted sums for continuous logits.
        '''
        H_graph = meta_graph.graph
        edge_gr_idx = node_gr_idx[H_graph.senders]
        n_graph = max(H_graph.n_node.shape)
        n_node = max(H_graph.nodes.shape)
        node_types = H_graph.globals["node_types"].squeeze()

        senders = H_graph.senders.squeeze()
        receivers = H_graph.receivers.squeeze()
        
        # Convert logits to probabilities
        probs = jnp.exp(logits)
        probs_per_group = jax.ops.segment_sum(probs, H_graph.globals["group_ids"], meta_graph.meta["n_groups"])
        

        owners_of_things = jnp.zeros((meta_graph.meta["persons"] + 1, meta_graph.meta["things"] + 1))

        person_index = jnp.where((node_types == PERSONS), H_graph.globals["nth_of_type"], -1)[receivers]
        thing_index = jnp.where((node_types == THINGS), H_graph.globals["nth_of_type"], -1)[senders]
        owners_of_things = owners_of_things.at[person_index, thing_index].add(probs)[:-1, :-1]

        owners_of_rooms = jnp.zeros((meta_graph.meta["persons"] + 1,meta_graph.meta["rooms"] + 1))

        room_index = jnp.where((node_types == ROOMS), H_graph.globals["nth_of_type"], -1)[senders]
        owners_of_rooms = owners_of_rooms.at[person_index, room_index].add(probs)

        rooms_x_cabinets = jnp.zeros((meta_graph.meta["rooms"] + 1, meta_graph.meta["cabinets"] + 1))
        cabinet_index = jnp.where((node_types == CABINETS), H_graph.globals["nth_of_type"], -1)[receivers]
        rooms_x_cabinets = rooms_x_cabinets.at[room_index, cabinet_index].add(probs)
        owners_of_cabinets = owners_of_rooms[:-1, :-1] @ rooms_x_cabinets[:-1, :-1]

        thing_index = jnp.where((node_types == THINGS), H_graph.globals["nth_of_type"], -1)[receivers]
        cabinet_index = jnp.where((node_types == CABINETS), H_graph.globals["nth_of_type"], -1)[senders]
        cabinets_x_things = jnp.zeros((meta_graph.meta["cabinets"] + 1, meta_graph.meta["things"] + 1))
        cabinets_x_things = cabinets_x_things.at[cabinet_index, thing_index].add(probs)
        owners_of_things_in_cabinets = owners_of_cabinets @ cabinets_x_things[:-1, :-1]

        argmax_per_cabinet = cabinets_x_things[:-1, :-1].argmax(-1)
        max_per_cabinet = cabinets_x_things[:-1, :-1].max(-1)

        argmax_per_cabinet_padded = jnp.pad(argmax_per_cabinet, 1, mode="constant", constant_values=0)

        loss_order_violations = ((argmax_per_cabinet_padded[1:] >= argmax_per_cabinet_padded[:-1])[:-1] * max_per_cabinet).sum()
        loss_things_per_cabinet = jax.nn.relu(cabinets_x_things[:-1, :-1].sum(-1) - 6).sum()
        loss_cabinets_per_room = jax.nn.relu(rooms_x_cabinets[:-1, :-1].sum(-1) - 5).sum()
        loss_owners_of_things = ((owners_of_things_in_cabinets - owners_of_things) ** 2).sum()
    
        energy = jnp.array([loss_owners_of_things + \
                loss_things_per_cabinet + \
                loss_cabinets_per_room + \
                loss_order_violations, 0])[:, None] # less than 5 cabinets per room
        
        def pad_up(x):
            return jnp.tile(x, 2)[..., None]

        sample = groupwise_sample(None, logits[..., None],H_graph.globals["group_ids"], meta_graph.meta["n_groups"])
        return energy, {"energy": energy, "argmax_energy": self.calculate_Energy(H_graph, sample.squeeze(), node_gr_idx)[0], "loss_things_per_cabinet": pad_up(loss_things_per_cabinet), "loss_cabinets_per_room": pad_up(loss_cabinets_per_room), "loss_order_violations": pad_up(loss_order_violations), "loss_owners_of_things": pad_up(loss_owners_of_things)}, energy

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, key=None):
        if logits.dtype == jnp.float32:
            return self.calculate_Energy_logits(H_graph["graphs"][0], logits[:, 0], node_gr_idx)
        else:
            return self.calculate_Energy(H_graph["graphs"][0].graph, logits[:, 0], node_gr_idx)
        