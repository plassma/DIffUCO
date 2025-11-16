from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp

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

        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = max(jax.tree_util.tree_leaves(nodes)[0].shape)
        node_types = H_graph.globals["node_types"].squeeze()
        senders = H_graph.senders.squeeze()
        receivers = H_graph.receivers.squeeze()
        bins = bins.squeeze()

        edges_person_thing = ((node_types[receivers] == THINGS) & (node_types[senders] == PERSONS))
        
        

        gt_owners_of_things = jnp.full((total_num_nodes,), -1, dtype=jnp.int32).at[jnp.where(edges_person_thing, receivers, -1)].set(jnp.arange(total_num_nodes)[jnp.where(edges_person_thing, senders, -1)])
        owners_of_rooms = jnp.full((total_num_nodes,), -1, dtype=jnp.int32).at[jnp.where((node_types == ROOMS), jnp.arange(total_num_nodes), -1)].set(bins)
        owners_of_cabinets = jnp.full((total_num_nodes,), -1, dtype=jnp.int32).at[jnp.where((node_types == CABINETS), jnp.arange(total_num_nodes), -1)].set(owners_of_rooms[bins])
        bins_cabinets = jnp.where((node_types == THINGS), bins + H_graph.meta["offset_cabinets"], -1)
        owners_of_things = jnp.full((total_num_nodes,), -1, dtype=jnp.int32).at[jnp.where((node_types == THINGS), jnp.arange(total_num_nodes), -1)].set(owners_of_cabinets[bins_cabinets] + H_graph.meta["offset_persons"])

        things_per_cabinet = jnp.zeros((H_graph.meta["cabinets"] + 1,)).at[jnp.where((node_types == THINGS), bins, -1)].add(1)[:-1]
        cabinets_per_room = jnp.zeros((H_graph.meta["rooms"] + 1,)).at[jnp.where((node_types == CABINETS), owners_of_rooms[bins], -1)].add(1)[:-1]

        energy_ownerships = jax.ops.segment_sum((owners_of_things != gt_owners_of_things).astype(jnp.float32), node_gr_idx, n_graph)[..., None]
        energy_things_per_cabinet = jnp.array([jnp.abs(things_per_cabinet - 5).sum(),  0])[..., None]
        energy_cabinets_per_room = jnp.array([jnp.abs(cabinets_per_room - 2).sum(), 0])[..., None]

        cabinets_of_things = jnp.where(node_types == THINGS, bins, -1)
        energy_order_violations = jnp.where((cabinets_of_things[1:] != -1) & (cabinets_of_things[:-1] > cabinets_of_things[1:]), 1, 0)
        
        energy_order_violations = jnp.array([energy_order_violations.sum(), 0])[..., None]

        Energy = energy_ownerships + energy_things_per_cabinet + energy_cabinets_per_room + energy_order_violations
        
        return Energy, {"energy_ownerships": energy_ownerships, "energy_order_violations": energy_order_violations, "energy_things_per_cabinet": energy_things_per_cabinet, "energy_cabinets_per_room": energy_cabinets_per_room}, Energy


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        return self.calculate_Energy(H_graph, logits, node_gr_idx)
