"""Shared helpers for the House Configuration Problem.

Utilities in this package are reused by the energy functions, dataset loading
code, JAX models, and debugging scripts.  They are implemented with jax.numpy
so that callers can safely use them inside jit/pmap regions.
"""

from .utils import (
    CABINETS,
    PERSONS,
    ROOMS,
    THINGS,
    build_house_arrays,
    calculate_capacity_counts,
    calculate_order_violations,
    calculate_owner_assignments,
    compute_node_graph_indices,
    pad_energy_graph,
    prior_logits_for_graph,
    sample_prior_state,
)

__all__ = [
    "ROOMS",
    "CABINETS",
    "THINGS",
    "PERSONS",
    "build_house_arrays",
    "calculate_owner_assignments",
    "calculate_capacity_counts",
    "calculate_order_violations",
    "compute_node_graph_indices",
    "prior_logits_for_graph",
    "sample_prior_state",
    "pad_energy_graph",
]
