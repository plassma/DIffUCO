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


class HCPEnergyClass(BaseEnergyClass):
    """Energy function for the House Configuration Problem (HCP)."""

    def __init__(self, config):
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        """Evaluate hard constraints for the given assignment."""
        del A, B  # Unused legacy parameters kept for API compatibility.

        n_graph = int(H_graph.n_node.shape[0])
        bins = bins.squeeze()
        house_arrays = build_house_arrays(H_graph)

        owner_info = calculate_owner_assignments(house_arrays, bins, H_graph.meta)
        capacity_counts = calculate_capacity_counts(
            house_arrays, bins, H_graph.meta, owner_info["owners_of_rooms"]
        )

        energy_ownerships = jax.ops.segment_sum(
            (owner_info["owners_of_things"] != owner_info["gt_owners_of_things"]).astype(jnp.float32),
            node_gr_idx,
            n_graph,
        )[..., None]

        things_penalty = jnp.abs(capacity_counts["things_per_cabinet"] - THINGS_PER_CABINET_TARGET).sum()
        energy_things_per_cabinet = jnp.array([things_penalty, 0.0], dtype=jnp.float32)[..., None]

        cabinets_penalty = jnp.abs(capacity_counts["cabinets_per_room"] - CABINETS_PER_ROOM_TARGET).sum()
        energy_cabinets_per_room = jnp.array([cabinets_penalty, 0.0], dtype=jnp.float32)[..., None]

        order_violation_count = calculate_order_violations(house_arrays, bins)
        energy_order_violations = jnp.array([order_violation_count, 0.0], dtype=jnp.float32)[..., None]

        total_energy = (
            energy_ownerships * 0 + energy_things_per_cabinet * 0.01 + energy_cabinets_per_room * 0 + energy_order_violations
        )
        breakdown = {
            "energy_ownerships": energy_ownerships,
            "energy_things_per_cabinet": energy_things_per_cabinet,
            "energy_cabinets_per_room": energy_cabinets_per_room,
            "energy_order_violations": energy_order_violations,
        }
        return total_energy, breakdown, total_energy

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A=1.0, B=1.2):
        return self.calculate_Energy(H_graph, bins, node_gr_idx, A=A, B=B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        return self.calculate_Energy(H_graph, logits, node_gr_idx)
