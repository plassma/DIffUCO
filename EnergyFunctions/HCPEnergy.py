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
        # fragen:
        # globals -> metadata
        # padding
        # differentiable? (glaub nein)
        # shapes?
        # FOL -> jax?
        # total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        # total_num_edges = jax.tree_util.tree_leaves(edges)[0].shape[0]
        # raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        # Energy_messages = (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])

        # print("Energy_per_graph", Energy.shape)
        #HA_per_node = - A * raveled_bins
        # HB_per_node = B * ( jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes))
        #HB_per_node = B * ( jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_edges))

        #violations_per_node = 0.5*(HB_per_node + jax.ops.segment_sum(Energy_messages, H_graph.senders, total_num_nodes))
        #violations_per_node = 0.5*(HB_per_node + jax.ops.segment_sum(Energy_messages, H_graph.senders, total_num_edges))

        Energy = jnp.reshape(jax.ops.segment_sum(loss_cabinets_to_things, node_gr_idx, n_graph), (n_graph, 1)) + 0.0001
        jax.debug.print("Energy {}", jnp.mean(Energy))
        # HB_per_graph = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
        return Energy , Energy, Energy#violations_per_node, HB_per_graph  # shapes: [n_graph, 1], [n_node <!--probably edge needed here-->, 1], [n_graph, 1]


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        p = jnp.exp(logits[...,1]) # 3151, 1 != 11151, 1, 2
        return self.calculate_Energy(H_graph, p, node_gr_idx)
