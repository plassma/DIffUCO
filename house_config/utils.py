"""Utility functions shared by the House Configuration Problem code paths."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from GraphWithMeta import GraphWithMeta
from jraph_utils import pmap_graph_list_better

ROOMS = 0
CABINETS = 1
THINGS = 2
PERSONS = 3


def build_house_arrays(H_graph) -> Dict[str, jnp.ndarray]:
    """Extract frequently used arrays from a padded HCP graph."""
    node_types = jnp.asarray(H_graph.globals["node_types"]).squeeze()
    senders = jnp.asarray(H_graph.senders).squeeze()
    receivers = jnp.asarray(H_graph.receivers).squeeze()
    total_nodes = int(jax.tree_util.tree_leaves(H_graph.nodes.squeeze())[0].shape[0])
    n_graph = int(H_graph.n_node.squeeze().shape[0])
    return {
        "node_types": node_types,
        "senders": senders,
        "receivers": receivers,
        "total_nodes": total_nodes,
        "n_graph": n_graph,
    }


def _owner_template(total_nodes: int) -> jnp.ndarray:
    return jnp.full((total_nodes,), -1, dtype=jnp.int32)

def _owner_template_cont(bins: int) -> jnp.ndarray:
    return jnp.zeros_like(bins)


def calculate_owner_assignments(
    house_arrays: Dict[str, jnp.ndarray], bins: jnp.ndarray, meta: Dict[str, int]
) -> Dict[str, jnp.ndarray]:
    """Compute ground-truth and predicted owners for rooms/cabinets/things."""
    node_types = house_arrays["node_types"]
    senders = house_arrays["senders"]
    receivers = house_arrays["receivers"]
    total_nodes = house_arrays["total_nodes"]

    bins = bins.squeeze()

    #person_thing_edges = (node_types[receivers] == THINGS) & (node_types[senders] == PERSONS)
    total_idx = jnp.arange(total_nodes, dtype=jnp.int32)

    gt_owners_of_things = bins[meta["offset_things_persons"] : meta["offset_things_persons"]+ meta["things"]]
    owners_of_rooms = _owner_template(total_nodes).at[
        jnp.where(node_types == ROOMS, total_idx, -1)
    ].set(bins)

    owners_of_cabinets = _owner_template(total_nodes).at[
        jnp.where(node_types == CABINETS, total_idx, -1)
    ].set(owners_of_rooms[bins])

    bins_cabinets = jnp.where(node_types == THINGS, bins + meta["offset_cabinets"], -1)
    owners_of_things = _owner_template(total_nodes).at[
        jnp.where(node_types == THINGS, total_idx, -1)
    ].set(owners_of_cabinets[bins_cabinets])[meta["offset_things_cabinets"]:meta["offset_things_cabinets"] + meta["things"]]

    return {
        "gt_owners_of_things": gt_owners_of_things,
        "owners_of_rooms": owners_of_rooms,
        "owners_of_cabinets": owners_of_cabinets,
        "owners_of_things": owners_of_things,
    }

def calculate_owner_assignments_continuous(
    house_arrays: Dict[str, jnp.ndarray], bins: jnp.ndarray, meta: Dict[str, int]
) -> Dict[str, jnp.ndarray]:
    """Compute ground-truth and predicted owners for rooms/cabinets/things."""
    node_types = house_arrays["node_types"]
    senders = house_arrays["senders"]
    receivers = house_arrays["receivers"]
    total_nodes = house_arrays["total_nodes"]

    bins = bins.squeeze()

    #person_thing_edges = (node_types[receivers] == THINGS) & (node_types[senders] == PERSONS)
    total_idx = jnp.arange(total_nodes, dtype=jnp.int32)

    gt_owners_of_things = bins[meta["offset_things_persons"] : meta["offset_things_persons"]+ meta["things"]]
    owners_of_rooms = _owner_template_cont(bins).at[
        jnp.where(node_types == ROOMS, total_idx, -1)
    ].set(bins)

    owners_of_cabinets = _owner_template(total_nodes).at[
        jnp.where(node_types == CABINETS, total_idx, -1)
    ].set(owners_of_rooms[bins])

    bins_cabinets = jnp.where(node_types == THINGS, bins + meta["offset_cabinets"], -1)
    owners_of_things = _owner_template(total_nodes).at[
        jnp.where(node_types == THINGS, total_idx, -1)
    ].set(owners_of_cabinets[bins_cabinets])[meta["offset_things_cabinets"]:meta["offset_things_cabinets"] + meta["things"]]

    return {
        "gt_owners_of_things": gt_owners_of_things,
        "owners_of_rooms": owners_of_rooms,
        "owners_of_cabinets": owners_of_cabinets,
        "owners_of_things": owners_of_things,
    }


def calculate_capacity_counts(
    house_arrays: Dict[str, jnp.ndarray],
    bins: jnp.ndarray,
    meta: Dict[str, int],
    owners_of_rooms: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Count things per cabinet and cabinets per room for penalty terms."""
    node_types = house_arrays["node_types"]
    things_per_cabinet = (
        jnp.zeros((meta["cabinets"] + 1,), dtype=jnp.int32)
        .at[jnp.where(node_types == THINGS, bins, -1)]
        .add(1)
    )[:-1]

    cabinets_per_room = (
        jnp.zeros((meta["rooms"] + 1,), dtype=jnp.int32)
        .at[jnp.where(node_types == CABINETS, bins, -1)]
        .add(1)
    )[:-1]

    return {
        "things_per_cabinet": things_per_cabinet,
        "cabinets_per_room": cabinets_per_room,
    }


def calculate_order_violations(meta_graph, bins: jnp.ndarray, what="things", include_severity=True) -> jnp.ndarray:
    """Return severity-weighted inversions in `bins` (weight = x_i - x_j), JIT-compatible."""
    if what == "things":
        offset = meta_graph.meta["offset_things_cabinets"]
        n = meta_graph.meta["things"]
    elif what == "cabinets":
        offset = meta_graph.meta["offset_cabinets"]
        n = meta_graph.meta["cabinets"]
    elif what == "rooms":
        offset = meta_graph.meta["offset_rooms"]
        n = meta_graph.meta["rooms"]
    else:
        raise ValueError(f"What is '{what}' not recognized for order violations.")
    x = bins[offset : offset + n]   # shape (n,)
    n = x.shape[0]

    idx = jnp.arange(n)
    
    i = idx[:, None]
    j = idx[None, :]

    inversion_mask = (i < j) & (x[:, None] > x[None, :])
    severity = x[:, None] - x[None, :]  # positive where mask is True

    if not include_severity:
        severity = jnp.ones_like(severity)

    return jnp.where(inversion_mask, severity, 0)


def compute_node_graph_indices(graph) -> Tuple[jnp.ndarray, int, int]:
    """Return node->graph indices along with helper counts."""
    n_node = jnp.asarray(graph.n_node)
    if n_node.ndim == 2:
        n_graph = int(n_node.shape[1])
        n_node = n_node[0]
    else:
        n_graph = int(n_node.shape[0])
    graph_idx = jnp.arange(n_graph)
    total_nodes = int(jnp.sum(n_node))
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
    return node_graph_idx, n_graph, total_nodes


def prior_logits_for_graph(meta_graph, soft: bool = False):
    """Compute log probs for the per-node categorical prior."""
    meta = getattr(meta_graph, "meta", {})
    classes_per_node = jnp.asarray(meta_graph.globals["classes_per_node"])
    num_classes = int(meta.get("cabinets", classes_per_node.shape[-1]))
    mask = (jnp.arange(num_classes)[None, :] < classes_per_node[:, None]).astype(jnp.float32)
    probs = mask / jnp.maximum(classes_per_node, 1)[:, None]
    eps = jnp.where(soft, 1e-10, 0.0)
    min_val = eps

    #fix ownership logits constant
    start_ownerships = meta["offset_things_persons"]
    end_ownerships = start_ownerships + meta["things"]
    probs = probs.at[start_ownerships:end_ownerships].set(0.0)
    rows = jnp.arange(start_ownerships, end_ownerships)
    probs = probs.at[rows, meta_graph.globals["owners_of_things"][:meta["things"]]].set(1.0)
    return jnp.log(jnp.clip(probs * mask + eps, a_min=min_val, a_max=None))


def sample_prior_state(meta_graph, key, soft: bool = False):
    """Sample a discrete assignment from the per-node prior."""
    logits = prior_logits_for_graph(meta_graph, soft=soft)
    key, subkey = jax.random.split(key)
    sample = jax.random.categorical(subkey, logits=logits, axis=-1)
    return sample.astype(jnp.int32), logits, key


def pad_energy_graph(graph_with_meta, dataset_statistics):
    """Pad a single energy graph using stored dataset statistics."""
    padded = pmap_graph_list_better([graph_with_meta], dataset_statistics)
    graph = padded.graph if isinstance(padded, GraphWithMeta) else padded
    graph = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf) if hasattr(leaf, "shape") else leaf, graph)
    if isinstance(padded, GraphWithMeta):
        return GraphWithMeta(graph=graph, meta=padded.meta)
    return graph
