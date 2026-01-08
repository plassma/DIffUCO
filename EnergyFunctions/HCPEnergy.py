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
        self.energy_weights = config["energy_weights"]

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, meta_graph, bins, node_gr_idx, A=1.0, B=1.2):
        """Evaluate hard constraints for the given assignment.

        Parameter `A` acts as an annealing weight on the ownership term
        (A=1.0 preserves the vanilla objective).
        """

        n_graph = int(meta_graph.n_node.shape[0])
        bins = bins.squeeze()
        house_arrays = build_house_arrays(meta_graph)

        owner_info = calculate_owner_assignments(house_arrays, bins, meta_graph.meta)
        capacity_counts = calculate_capacity_counts(
            house_arrays, bins, meta_graph.meta, owner_info["owners_of_rooms"]
        )

        exp_cabinets_things_rooms = self.energy_weights.get("exp_cabinets_things_rooms", 2)

        ownership_violations = (owner_info["owners_of_things"] != owner_info["gt_owners_of_things"]).sum()
        ownership_weight = jnp.asarray(A, dtype=jnp.float32)
        energy_ownerships = jnp.array([ownership_violations, 0.0], dtype=jnp.float32)[..., None]

        violations_per_cabinet = jnp.abs((capacity_counts["things_per_cabinet"] - THINGS_PER_CABINET_TARGET) ** exp_cabinets_things_rooms)
        things_penalty = violations_per_cabinet.sum()

        energy_things_per_cabinet = jnp.array([things_penalty, 0.0], dtype=jnp.float32)[..., None]

        violations_per_room = jnp.abs((capacity_counts["cabinets_per_room"] - CABINETS_PER_ROOM_TARGET) ** exp_cabinets_things_rooms)
        cabinets_penalty = violations_per_room.sum()
        energy_cabinets_per_room = jnp.array([cabinets_penalty, 0.0], dtype=jnp.float32)[..., None]

        order_violations_things = calculate_order_violations(meta_graph, bins, "things", include_severity=self.energy_weights["order_severity"])
        energy_order_violations_things = jnp.array([order_violations_things.sum(), 0.0], dtype=jnp.float32)[..., None]

        order_violations_cabinets = calculate_order_violations(meta_graph, bins, "cabinets", include_severity=self.energy_weights["order_severity"])
        energy_order_violations_cabinets = jnp.array([order_violations_cabinets.sum(), 0.0], dtype=jnp.float32)[..., None]

        order_violations_rooms = calculate_order_violations(meta_graph, bins, "rooms", include_severity=self.energy_weights["order_severity"])
        energy_order_violations_rooms = jnp.array([order_violations_rooms.sum(), 0.0], dtype=jnp.float32)[..., None]

        energy_per_node = jnp.pad(
            jnp.concat([violations_per_room, violations_per_cabinet, order_violations_things.sum(0)], axis=0),
            (0, max(0, bins.size - jnp.concat([violations_per_room, violations_per_cabinet, order_violations_things.sum(0)], axis=0).size))
        ).astype(jnp.float32)

        total_energy = (
            self.energy_weights["energy_ownerships"] * ownership_weight * (energy_ownerships / meta_graph.meta["things"]) + 
            self.energy_weights["energy_things_per_cabinet"] * energy_things_per_cabinet / meta_graph.meta["things"] + 
            self.energy_weights["energy_cabinets_per_room"] * energy_cabinets_per_room / meta_graph.meta["cabinets"] + 
            jnp.sqrt(self.energy_weights["energy_order_violations"] *  energy_order_violations_things / expected_num_inversions(meta_graph.meta["things"])) +
            jnp.sqrt(self.energy_weights["energy_order_violations_cabinets"] *  energy_order_violations_cabinets / meta_graph.meta["cabinets"] ** 2)+
            jnp.sqrt(self.energy_weights["energy_order_violations_rooms"] * energy_order_violations_rooms / meta_graph.meta["rooms"] ** 2)
        )
        breakdown = {
            "energy_ownerships": energy_ownerships,
            "energy_things_per_cabinet": energy_things_per_cabinet,
            "energy_cabinets_per_room": energy_cabinets_per_room,
            "energy_order_violations_things": energy_order_violations_things,
            "energy_order_violations_cabinets": energy_order_violations_cabinets,
            "energy_order_violations_rooms": energy_order_violations_rooms,
            "weighted_ownerships": self.energy_weights["energy_ownerships"] * ownership_weight * (energy_ownerships / meta_graph.meta["things"]),
            "weighted_things_per_cabinet": self.energy_weights["energy_things_per_cabinet"] * energy_things_per_cabinet / meta_graph.meta["things"],
            "weighted_cabinets_per_room": self.energy_weights["energy_cabinets_per_room"] * energy_cabinets_per_room / meta_graph.meta["cabinets"],
            "weighted_order_violations_things": jnp.sqrt(self.energy_weights["energy_order_violations"] *  energy_order_violations_things / expected_num_inversions(meta_graph.meta["things"])),
            "weighted_order_violations_cabinets": jnp.sqrt(self.energy_weights["energy_order_violations_cabinets"] *  energy_order_violations_cabinets / meta_graph.meta["cabinets"] ** 2),
            "weighted_order_violations_rooms": jnp.sqrt(self.energy_weights["energy_order_violations_rooms"] * energy_order_violations_rooms / meta_graph.meta["rooms"] ** 2),
        }
        return total_energy, breakdown, energy_per_node

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, bins, node_gr_idx, A=A, B=B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, logits, node_gr_idx, A=A, B=B)
