from functools import partial

import jax
import jax.numpy as jnp

from house_config import (
    build_house_arrays,
    calculate_capacity_counts,
    calculate_order_violations,
    calculate_owner_assignments,
)
from .BaseEnergy import BaseEnergyClass

THINGS_PER_CABINET_TARGET = 5
CABINETS_PER_ROOM_TARGET = 2

def expected_num_inversions(nodes: int) -> int:
    return (nodes - 5) * (nodes - 1) // 4

class HCPEnergyClass(BaseEnergyClass):
    """Energy function for the House Configuration Problem (HCP)."""

    def __init__(self, config):
        super().__init__(config)
        self.energy_weights = config.get("energy_weights", {
            "energy_ownerships": 1.0,
            "energy_things_per_cabinet": 1.0,
            "energy_cabinets_per_room": 1.0,
            "energy_order_violations": 3.0,
            "asymmetric": False,
            "exp_cabinets_things_rooms": 2,
        })

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, meta_graph, bins, node_gr_idx, A=1.0, B=1.2):
        """Evaluate hard constraints for the given assignment."""
        del A, B  # Unused legacy parameters kept for API compatibility.

        n_graph = int(meta_graph.n_node.shape[0])
        bins = bins.squeeze()
        house_arrays = build_house_arrays(meta_graph)

        owner_info = calculate_owner_assignments(house_arrays, bins, meta_graph.meta)
        capacity_counts = calculate_capacity_counts(
            house_arrays, bins, meta_graph.meta, owner_info["owners_of_rooms"]
        )

        exp_cabinets_things_rooms = self.energy_weights.get("exp_cabinets_things_rooms", 2)

        if self.energy_weights["asymmetric"]:
            energy_ownerships = jnp.array([((owner_info["owners_of_things"] != owner_info["gt_owners_of_things"]) * (owner_info["gt_owners_of_things"] + 1)).sum() * 2 / meta_graph.meta["persons"], 0]).astype(jnp.float32)[..., None]
        else:
            energy_ownerships = jnp.array([(owner_info["owners_of_things"] != owner_info["gt_owners_of_things"]).sum(), 0]).astype(jnp.float32)[..., None]

        if self.energy_weights["asymmetric"]:
            things_penalty = jnp.abs((capacity_counts["things_per_cabinet"] - THINGS_PER_CABINET_TARGET) * jnp.arange(meta_graph.meta["cabinets"]) * 2 / meta_graph.meta["cabinets"]).sum() #jnp.abs((capacity_counts["things_per_cabinet"] - THINGS_PER_CABINET_TARGET)).sum()
        else:
            things_penalty = jnp.abs((capacity_counts["things_per_cabinet"] - THINGS_PER_CABINET_TARGET) ** exp_cabinets_things_rooms).sum()

        energy_things_per_cabinet = jnp.array([things_penalty, 0.0], dtype=jnp.float32)[..., None]

        if self.energy_weights["asymmetric"]:
            cabinets_penalty = jnp.abs((capacity_counts["cabinets_per_room"] - CABINETS_PER_ROOM_TARGET) * jnp.arange(meta_graph.meta["rooms"]) * 2 / meta_graph.meta["rooms"]).sum() #jnp.abs((capacity_counts["cabinets_per_room"] - CABINETS_PER_ROOM_TARGET)).sum()
        else:
            cabinets_penalty = jnp.abs((capacity_counts["cabinets_per_room"] - CABINETS_PER_ROOM_TARGET) ** exp_cabinets_things_rooms).sum()
        energy_cabinets_per_room = jnp.array([cabinets_penalty, 0.0], dtype=jnp.float32)[..., None]

        order_violations_things = calculate_order_violations(meta_graph, bins, "things")
        energy_order_violations_things = jnp.array([order_violations_things, 0.0], dtype=jnp.float32)[..., None]

        order_violations_cabinets = calculate_order_violations(meta_graph, bins, "cabinets")
        energy_order_violations_cabinets = jnp.array([order_violations_cabinets, 0.0], dtype=jnp.float32)[..., None]

        order_violations_rooms = calculate_order_violations(meta_graph, bins, "rooms")
        energy_order_violations_rooms = jnp.array([order_violations_rooms, 0.0], dtype=jnp.float32)[..., None]

        total_energy = (
            self.energy_weights["energy_ownerships"] * energy_ownerships / meta_graph.meta["things"] + 
            self.energy_weights["energy_things_per_cabinet"] * energy_things_per_cabinet / meta_graph.meta["things"] + 
            self.energy_weights["energy_cabinets_per_room"] * energy_cabinets_per_room / meta_graph.meta["cabinets"] + 
            jnp.sqrt(self.energy_weights["energy_order_violations"] * energy_order_violations_things / expected_num_inversions(meta_graph.meta["things"])) #+
            #jnp.sqrt(self.energy_weights["energy_order_violations"] * energy_order_violations_cabinets / expected_num_inversions(meta_graph.meta["cabinets"])) +
            #jnp.sqrt(self.energy_weights["energy_order_violations"] * energy_order_violations_rooms / expected_num_inversions(meta_graph.meta["rooms"]))
        )
        breakdown = {
            "energy_ownerships": energy_ownerships,
            "energy_things_per_cabinet": energy_things_per_cabinet,
            "energy_cabinets_per_room": energy_cabinets_per_room,
            "energy_order_violations": energy_order_violations_things,
            "weighted_ownerships": self.energy_weights["energy_ownerships"] * energy_ownerships / meta_graph.meta["things"],
            "weighted_things_per_cabinet": self.energy_weights["energy_things_per_cabinet"] * energy_things_per_cabinet / meta_graph.meta["things"],
            "weighted_cabinets_per_room": self.energy_weights["energy_cabinets_per_room"] * energy_cabinets_per_room / meta_graph.meta["cabinets"] ,
            "weighted_order_violations": self.energy_weights["energy_order_violations"] * energy_order_violations_things / expected_num_inversions(meta_graph.meta["things"]),
        }
        return total_energy, breakdown, total_energy

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, bins, node_gr_idx, A=A, B=B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        return self.calculate_Energy(H_graph, logits, node_gr_idx)
