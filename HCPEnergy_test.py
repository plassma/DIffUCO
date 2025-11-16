from EnergyFunctions import HCPEnergyClass
import jax.numpy as jnp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
import jraph
import jraph_utils
import numpy as np
from jraph_utils import pmap_graph_list_better
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import plot
import argparse

from Data.LoadGraphDataset import SolutionDatasetLoader
import os

if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True,)# precision=4

    parser = argparse.ArgumentParser(
        description="Run HCPEnergyClass.calculate_Energy_loss on a graph loaded from the dataset."
    )
    parser.add_argument("--dataset", default="HCP_dummy", help="Dataset name passed to SolutionDatasetLoader.")
    parser.add_argument("--problem", default="HCP", help="Problem name passed to SolutionDatasetLoader.")
    parser.add_argument("--mode", default="train", choices=("train", "val", "test"), help="Which dataset split to draw from.")
    parser.add_argument("--seed", type=int, default=123, help="Dataset random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for the DataLoader.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the stored instance to load (mapped via config['use_sample']).")
    parser.add_argument("--n-bernoulli-features", type=int, default=10, help="Value forwarded to HCPEnergyClass config.")
    parser.add_argument("--n-diffusion-steps", type=int, default=0, help="Minimal config knob required by SolutionDatasetLoader.")
    parser.add_argument("--n-basis-states", type=int, default=1, help="Minimal config knob required by SolutionDatasetLoader.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed used when sampling from the prior.")
    args = parser.parse_args()

    def _compute_node_graph_idx(graph):
        n_node = jnp.asarray(graph.n_node)
        n_graph = int(n_node.shape[1])
        graph_idx = jnp.arange(n_graph)
        total_nodes = int(jnp.sum(n_node))
        return jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    def _build_dataset_statistics(dataloader):
        if dataloader is None:
            raise RuntimeError("Dataset statistics unavailable because the dataloader is None.")

        required_attrs = (
            "smallest_n_nodes_energy_graph",
            "largest_n_nodes_energy_graph",
            "smallest_n_edges_energy_graph",
            "largest_n_edges_energy_graph",
        )
        missing = [attr for attr in required_attrs if not hasattr(dataloader, attr)]
        if missing:
            raise AttributeError(f"Missing dataset statistics on dataloader: {missing}")

        n_devices = max(len(jax.devices()), 1)
        grid_num = max(1, min(7, int(12 * args.batch_size / (30 * n_devices)) or 1))
        edge_grid_factor = 2

        return {
            "grid_num": grid_num,
            "edge_grid_factor": edge_grid_factor,
            "min_nodes": getattr(dataloader, "smallest_n_nodes_energy_graph"),
            "max_nodes": getattr(dataloader, "largest_n_nodes_energy_graph"),
            "min_edges": getattr(dataloader, "smallest_n_edges_energy_graph"),
            "max_edges": getattr(dataloader, "largest_n_edges_energy_graph"),
        }

    def _pad_energy_graph(graph, dataset_stats):
        return pmap_graph_list_better([graph], dataset_stats)

    def _sample_prior(meta_graph, key):
        num_classes = int(meta_graph.meta.get("cabinets", 1))
        classes_per_node = jnp.asarray(meta_graph.graph.globals["classes_per_node"])
        mask = (jnp.arange(num_classes)[None, :] < classes_per_node[:, None]).astype(jnp.float32)
        probs = mask / jnp.maximum(classes_per_node, 1)[:, None]
        logits = jnp.log(jnp.clip(probs, a_min=1e-9, a_max=1.0))
        return jax.random.categorical(key, logits=logits).astype(jnp.float32)

    jax.config.update("jax_disable_jit", True)

    base_config = {
        "n_bernoulli_features": args.n_bernoulli_features,
        "n_diffusion_steps": args.n_diffusion_steps,
        "N_basis_states": args.n_basis_states,
        "use_sample": args.sample_idx,
    }

    energy_fn = HCPEnergyClass(base_config)
    dataset_loader = SolutionDatasetLoader(
        config=base_config,
        dataset=args.dataset,
        problem=args.problem,
        batch_size=args.batch_size,
        relaxed=False,
        seed=args.seed,
        mode=args.mode,
    )

    dataloader_train, dataloader_test, dataloader_val, _ = dataset_loader.dataloaders()
    dataloader = {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test}[args.mode]
    if dataloader is None:
        raise RuntimeError(f"No dataloader initialized for mode={args.mode}.")

    dataset_stats = _build_dataset_statistics(dataloader)

    batch = next(iter(dataloader))
    graph_with_meta = batch["energy_graph"][0]
    key = jax.random.PRNGKey(args.sample_seed)
    raw_sample = _sample_prior(graph_with_meta, key)
    
    padded_energy_graph = _pad_energy_graph(graph_with_meta, dataset_stats)
    energy_graph = jax.tree_map(
        lambda leaf: jnp.asarray(leaf) if hasattr(leaf, "shape") else leaf,
        padded_energy_graph,
    )
    node_gr_idx = _compute_node_graph_idx(energy_graph)

    total_nodes = jax.tree_util.tree_leaves(energy_graph.nodes)[0].shape[1]
    pad_amount = total_nodes - raw_sample.shape[0]
    if pad_amount < 0:
        raise ValueError("Padded graph has fewer nodes than the sampled state length.")
    if pad_amount > 0:
        raw_sample = jnp.pad(raw_sample, (0, pad_amount))

    node_types = np.array(energy_graph.globals["node_types"].squeeze())
    plot(None, node_types[:-1], "solution_nodes.png", solution_nodes=energy_graph.globals["solution_nodes"].squeeze()[:-1])

    raw_sample = raw_sample.astype(np.int32).squeeze()
    prior_sample = np.full((total_nodes,), -1)
    prior_sample[10:56] = raw_sample[10:56]
    plot(None, node_types, "prior_nodes.png", solution_nodes=prior_sample)

    energy_fn.calculate_Energy(energy_graph, energy_graph.globals["solution_nodes"].squeeze(), node_gr_idx)
    energy, violations, hb = energy_fn.calculate_Energy(energy_graph, raw_sample.astype(jnp.int32), node_gr_idx)
    print(f"Loaded dataset='{args.dataset}' mode='{args.mode}' sample={args.sample_idx}")
    print("Sampled state energy per graph:", energy)
    print("Violations per node:", violations)
    print("HB per graph:", hb)
