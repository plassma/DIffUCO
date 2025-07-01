from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

ROOMS = 0
CABINETS = 1
THINGS = 2
PERSONS = 3

class HCPEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
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
        n_graph = H_graph.n_node.shape[0]
        #nodes = H_graph.nodes
        # get things of cabinets (edges)
        bins = jnp.squeeze(bins)

        cabinets_to_things = jnp.where((H_graph.globals[H_graph.senders] == CABINETS) & (H_graph.globals[H_graph.receivers] == THINGS), bins, 0) # todo: use bins instead of H_graph.edges?
        n_connections_cabinets_to_things = jax.ops.segment_sum(cabinets_to_things, H_graph.senders, H_graph.nodes.shape[0])
        loss_cabinets_to_things = jnp.where((H_graph.globals == CABINETS), (1 - n_connections_cabinets_to_things) ** 2, 0)
        loss_cabinets_to_things = jnp.reshape(loss_cabinets_to_things, (loss_cabinets_to_things.shape[0], 1))

        rooms_to_cabinets = jnp.where((H_graph.globals[H_graph.senders] == ROOMS) & (H_graph.globals[H_graph.receivers] == CABINETS), bins, 0)
        n_connections_rooms_to_cabinets = jax.ops.segment_sum(rooms_to_cabinets, H_graph.senders, H_graph.nodes.shape[0])
        loss_rooms_to_cabinets = jnp.where((H_graph.globals == ROOMS), (1 - n_connections_rooms_to_cabinets) ** 2, 0)
        loss_rooms_to_cabinets = jnp.reshape(loss_rooms_to_cabinets, (loss_rooms_to_cabinets.shape[0], 1))

        rooms_to_persons = jnp.where((H_graph.globals[H_graph.senders] == PERSONS) & (H_graph.globals[H_graph.receivers] == ROOMS), bins, 0)
        n_connections_rooms_to_persons = jax.ops.segment_sum(rooms_to_persons, H_graph.receivers, H_graph.nodes.shape[0])
        loss_rooms_to_persons = jnp.where((H_graph.globals == ROOMS), (1 - n_connections_rooms_to_persons) ** 2, 0).reshape((H_graph.nodes.shape[0], 1))

        things_to_person = jnp.where((H_graph.globals[H_graph.senders] == THINGS) & (H_graph.globals[H_graph.receivers] == PERSONS), bins, 1)
        loss_things_to_person = jax.ops.segment_sum((1-things_to_person) ** 2, edge_gr_idx, n_graph)
        
        person_of_thing = jnp.where((H_graph.globals[H_graph.senders] == THINGS) & (H_graph.globals[H_graph.receivers] == PERSONS) & bins, H_graph.globals[H_graph.receivers], -1)

        person_of_room = jnp.where((H_graph.globals[H_graph.senders] == PERSONS) & (H_graph.globals[H_graph.receivers] == ROOMS) & bins, H_graph.globals[H_graph.senders], -1)
        person_of_cabinet = jnp.where((H_graph.globals[H_graph.senders] == ROOMS) & (H_graph.globals[H_graph.receivers] == CABINETS) & bins, person_of_room, -1)
        person_of_thing_in_cabinet = jnp.where((H_graph.globals[H_graph.senders] == CABINETS) & (H_graph.globals[H_graph.receivers] == THINGS) & bins, person_of_cabinet, -1)

        loss_thing_in_owned_cabinet = jax.ops.segment_sum((person_of_thing_in_cabinet != person_of_thing).astype(int), edge_gr_idx, n_graph)

        Energy = jnp.reshape(jax.ops.segment_sum(loss_cabinets_to_things + loss_rooms_to_cabinets + loss_rooms_to_persons, node_gr_idx, n_graph), (n_graph, 1)) + jnp.reshape(loss_things_to_person + loss_thing_in_owned_cabinet, (n_graph, 1)) + 0.0001

        
        #jax.debug.print("Energy {}", jnp.mean(Energy))
        # HB_per_graph = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
        return Energy , Energy, Energy#violations_per_node, HB_per_graph  # shapes: [n_graph, 1], [n_node <!--probably edge needed here-->, 1], [n_graph, 1]


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        #p = jnp.exp(logits[...,1]) # 3151, 1 != 11151, 1, 2
        p = logits[..., 1]
        return self.calculate_Energy(H_graph, p, node_gr_idx)
