from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
import networkx as nx
from typing import Optional

from .BaseDatasetGenerator import BaseDatasetGenerator
from DatasetCreator.jraph_utils import utils as jutils
from tqdm import tqdm
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from GraphWithMeta import GraphWithMeta
from matplotlib.colors import ListedColormap

def to_shape(a, shape, pad_value=-1):
	a = np.array(a)
	z = np.full(shape, pad_value)
	z[:a.shape[0], :a.shape[1]] = a
	return z

@dataclass
class HCProblem:

	rooms: int
	cabinets: int
	things: int
	persons: int
	persons_permutation: Optional[list[int]] = None
	OFFSET_ROOMS: int = field(init=False)
	OFFSET_CABINETS: int = field(init=False)
	OFFSET_THINGS_CABINETS: int = field(init=False)
	OFFSET_THINGS_PERSONS: int = field(init=False)
	OFFSET_PER_NODE_TYPE: list[int] = field(init=False)
	N_NODES: int = field(init=False)


	# Fields for the properties
	globals: dict[str, np.ndarray] = field(init=False)
	node_types: np.ndarray = field(init=False)
	classes_per_node: np.ndarray = field(init=False)
	solution_edges: tuple[list[tuple[int, int]], list[int]] = field(init=False)

	def __post_init__(self):
		# Existing initialization code
		self.OFFSET_ROOMS = 0
		self.OFFSET_CABINETS = self.OFFSET_ROOMS + self.rooms
		self.OFFSET_THINGS_CABINETS = self.OFFSET_CABINETS + self.cabinets
		self.OFFSET_THINGS_PERSONS = self.OFFSET_THINGS_CABINETS + self.things
		self.N_NODES = self.OFFSET_THINGS_PERSONS
		self.OFFSET_PER_NODE_TYPE = np.array([self.OFFSET_THINGS_PERSONS, self.OFFSET_ROOMS, self.OFFSET_CABINETS, self.OFFSET_THINGS_PERSONS])

		self.persons_range = self.persons_permutation if self.persons_permutation else list(range(self.persons))

		self.edges_RxC = [(r + self.OFFSET_ROOMS, c + self.OFFSET_CABINETS) for r in range(self.rooms) for c in range(self.cabinets)]
		self.edges_CxT = [(c + self.OFFSET_CABINETS, t + self.OFFSET_THINGS_CABINETS) for c in range(self.cabinets) for t in range(self.things)]
		self.edges_PxR = [(p + self.OFFSET_THINGS_PERSONS, r + self.OFFSET_ROOMS) for p in range(self.persons) for r in range(self.rooms)]
		self.edges_TxP = [(t + p * 10 + self.OFFSET_THINGS_CABINETS, i + self.OFFSET_THINGS_PERSONS) for i, p in enumerate(self.persons_range) for t in range(10)]  # default ownership - 10 things per person
		self.all_edges = sorted(self.edges_RxC + self.edges_CxT + self.edges_PxR + self.edges_TxP)

		self.igraph = ig.Graph(n=self.OFFSET_THINGS_PERSONS + self.things, edges=self.all_edges)

		# Initialize the new fields
		self.node_types = self._initialize_node_types()
		self.classes_per_node = self._initialize_classes_per_node()
		self.solution_edges, self.solution_bin, self.solution_nodes = self._initialize_solution()
		self.neighbours_per_node = self._init_neighbours_per_node()
		self.nth_of_type = self._init_nth_of_type()

		self.globals = {"node_types": self.node_types, "solution_bin": self.solution_bin, "classes_per_node": self.classes_per_node, "neighbours_per_node": self.neighbours_per_node, "solution_nodes": self.solution_nodes,
				  "offset_per_node_type": self.OFFSET_PER_NODE_TYPE[self.node_types], "nth_of_type": self.nth_of_type, "owners_of_things": self.solution_nodes[self.OFFSET_THINGS_PERSONS:self.OFFSET_THINGS_PERSONS + self.things]}

	def _init_neighbours_per_node(self) -> list[list[int]]:
		rooms = np.where(self.node_types == 0)[0]
		cabinets = np.where(self.node_types == 1)[0]
		things_to_cabinets = np.where(self.node_types == 2)[0]
		things_to_persons = np.where(self.node_types == 3)[0]

		max_neighbours = max(self.rooms, self.cabinets, self.persons, 1)

		neighbours = [
			to_shape([[self.OFFSET_THINGS_PERSONS + p for p in range(self.persons)] for _ in rooms], (len(rooms), max_neighbours)),
			to_shape([[self.OFFSET_ROOMS + r for r in range(self.rooms)] for _ in cabinets], (len(cabinets), max_neighbours)),
			to_shape([[self.OFFSET_CABINETS + c for c in range(self.cabinets)] for _ in things_to_cabinets], (len(things_to_cabinets), max_neighbours)),
			to_shape([[self.OFFSET_THINGS_PERSONS + p for p in range(self.persons)] for _ in things_to_persons], (len(things_to_persons), max_neighbours)),
		]

		return np.concatenate(neighbours, 0)
	
	def _init_nth_of_type(self) -> np.ndarray:
		nth_of_type = np.zeros(len(self.node_types), dtype=int)
		type_counts = {t: 0 for t in np.unique(self.node_types)}
		for i, node_type in enumerate(self.node_types):
			nth_of_type[i] = type_counts[node_type]
			type_counts[node_type] += 1
		return nth_of_type

	def _initialize_node_types(self) -> np.ndarray:
		return np.array(
			[0] * self.rooms +
			[1] * self.cabinets +
			[2] * self.things +
			[3] * self.things
		)

	def _initialize_classes_per_node(self) -> np.ndarray:
		classes_per_node_type = np.array([
			self.persons,  # rooms connected to persons
			self.rooms,    # cabinets connected to rooms
			self.cabinets, # things connected to cabinets
			self.persons   # things connected to persons
		])
		return classes_per_node_type[self.node_types]

	def _initialize_solution(self) -> tuple[list[tuple[int, int]], list[int]]:
		rooms = np.where(self.node_types == 0)[0]
		cabinets = np.where(self.node_types == 1)[0]
		things_to_cabinets = np.where(self.node_types == 2)[0]
		things_to_persons = np.where(self.node_types == 3)[0]
		persons = np.array(np.argsort(self.persons_range)) + self.OFFSET_THINGS_PERSONS #np.arange(self.OFFSET_THINGS_PERSONS, self.OFFSET_THINGS_PERSONS + self.persons)

		max_node_index = max([max(edge) for edge in self.all_edges]) + 1 if len(self.all_edges) > 0 else 0
		n_nodes = max(len(self.node_types), self.OFFSET_THINGS_PERSONS + self.persons, max_node_index)

		solution_nodes = np.full(n_nodes, -1, dtype=int)
		edges = list(self.edges_TxP)

		thing_to_cabinet_order = {node: idx for idx, node in enumerate(things_to_cabinets)}
		thing_to_person_order = {node: idx for idx, node in enumerate(things_to_persons)}

		for sender, receiver in self.edges_TxP:
			person_class = receiver - self.OFFSET_THINGS_PERSONS
			if sender in thing_to_person_order:
				solution_nodes[sender] = person_class
			elif sender in thing_to_cabinet_order:
				idx = thing_to_cabinet_order[sender]
				if idx < len(things_to_persons):
					solution_nodes[things_to_persons[idx]] = person_class

		for idx, thing in enumerate(things_to_cabinets):
			if len(cabinets) == 0:
				break
			target_cabinet = cabinets[idx // 5 % len(cabinets)]
			solution_nodes[thing] = idx // 5
			edges.append((target_cabinet, thing))

		for idx, cabinet in enumerate(cabinets):
			if len(rooms) == 0:
				break
			target_room = rooms[idx // 2 % len(rooms)]
			solution_nodes[cabinet] = idx // 2
			edges.append((target_room, cabinet))

		for room, person in zip(rooms, persons):
			solution_nodes[room] = person - self.OFFSET_THINGS_PERSONS
			edges.append((person, room))

		edges.sort()
		
		j = 0
		bin_solution = []
		for edge in self.all_edges:
			if j < len(edges) and edge == edges[j]:
				bin_solution.append(1)
				j += 1
			else:
				bin_solution.append(0)
		assert j == len(edges)

		return np.array(edges), np.array(bin_solution), solution_nodes
	
	def plot(self, target, include_legend=False):
		return plot(self.igraph, self.node_types, target, include_legend)
		
	@property
	def meta_graph(problem) -> GraphWithMeta:
		H_graph = jutils.from_igraph_to_jgraph(problem.igraph)
		H_graph = H_graph._replace(globals=problem.globals)

		return GraphWithMeta(graph=H_graph, meta={"rooms": problem.rooms, "cabinets": problem.cabinets, "things": problem.things, "persons": problem.persons, "id": -1,
											"offset_rooms": problem.OFFSET_ROOMS, "offset_cabinets": problem.OFFSET_CABINETS, "offset_things_cabinets": problem.OFFSET_THINGS_CABINETS, "offset_things_persons": problem.OFFSET_THINGS_PERSONS, "n_nodes": problem.N_NODES})
	
def plot(igraph, node_types, target, include_legend=False, bin_solution_edge=None, solution_nodes = None, verbose=False, meta_graph=None):
	if solution_nodes is not None:
		solution_nodes = solution_nodes[(node_types if node_types is not None else meta_graph.globals["node_types"][0]) >= 0]
		offset_node_types = meta_graph.globals["offset_per_node_type"]
		while offset_node_types.ndim > 1:
			offset_node_types = offset_node_types[0]
		while node_types.ndim > 1:
			node_types = node_types[0]
		edges = [(i if i < meta_graph.meta["n_nodes"] else i - meta_graph.meta["things"], int(j) + offset_node_types[i]) for i, j in enumerate(solution_nodes) if int(j) != -1]
		plot_graph = ig.Graph(edges=edges)
	elif bin_solution_edge is not None:
		edges = igraph.get_edgelist()
		edges = [(a, b) for i, (a, b) in enumerate(edges) if bin_solution_edge is None or bin_solution_edge[i]]
		plot_graph = ig.Graph(edges=edges)
	else:
		plot_graph = igraph.copy()
	edges = plot_graph.get_edgelist()

	vertex_label_appendix = []
	vertex_color_appendix = []


	mismatches = -1

	if verbose:
		things = [i for i in range(len(node_types)) if node_types[i] == 2]
		rooms = [i for i in range(len(node_types)) if node_types[i] == 0]
		cabinets = [i for i in range(len(node_types)) if node_types[i] == 1]


		owners_of_things = {}

		for thing in things:
			connections_to_persons = sum(1 for e in edges if e[0] == thing and node_types[e[1]] == 3)
			connections_to_cabinets = sum(1 for e in edges if e[1] == thing and node_types[e[0]] == 1)
			if verbose:
				print(f"Thing {thing} has {connections_to_persons} connections to persons and {connections_to_cabinets} connections to cabinets")
			owners_of_things[thing] = [e[1] for e in edges if e[0] == thing and node_types[e[1]] == 3][0]
		
		owners_of_rooms = {}

		for room in rooms:
			connections_to_persons = sum(1 for e in edges if e[0] == room and node_types[e[1]] == 3)
			connections_to_cabinets = sum(1 for e in edges if e[0] == room and node_types[e[1]] == 1)
			if verbose:
				print(f"Room {room} has {connections_to_persons} connections to persons and {connections_to_cabinets} connections to cabinets")
			owners_of_rooms[room] = [e[1] for e in edges if e[0] == room and node_types[e[1]] == 3][0]

		owners_of_cabinets = {}

		for cabinet in cabinets:
			owners_of_cabinets[cabinet] = [owners_of_rooms[e[0]] for e in edges if e[1] == cabinet and node_types[e[0]] == 0][0]

		holders_of_things = {}

		for thing in things:
			holders_of_things[thing] = [owners_of_cabinets[e[0]] for e in edges if e[1] == thing and node_types[e[0]] == 1][0]

		mismatches = sum(1 for k in owners_of_things if owners_of_things[k] != holders_of_things[k])


	
		print(f"Mismatches: {mismatches}")
	
	node_types = node_types[:len(plot_graph.vs)]

	if include_legend:
		plot_graph.add_vertices(4)
		vertex_color_appendix = [VERTEX_COLORS[t] for t in list(VERTEX_LABELS.keys())[:-1]]
		vertex_label_appendix = [VERTEX_LABELS[t] for t in list(VERTEX_LABELS.keys())[:-1]]

	edge_colors = ["black" if node_types[e[0]] == 2 and node_types[e[1]] == 3 else "grey" for e in edges]
	edge_widths = [3 if node_types[e[0]] == 2 and node_types[e[1]] == 3 else 2 for e in edges]

	vertex_colors = [VERTEX_COLORS[t] for t in node_types] + vertex_color_appendix #+ list(VERTEX_LABELS.keys())[:-1]
	counts_per_node_type = defaultdict(int)
	vertex_labels = []
	for i, t in enumerate(node_types):
		label = counts_per_node_type[t]
		vertex_labels.append(label)
		counts_per_node_type[t] += 1
	vertex_labels = vertex_labels + vertex_label_appendix
	#vertex_labels = [i - meta_graph.globals["offset_per_node_type"].squeeze()[i] for i, t in enumerate(node_types)] + vertex_label_appendix#+ list(VERTEX_LABELS.values())[:-1]
	ig.plot(plot_graph, vertex_label=vertex_labels, target=target, vertex_color=vertex_colors,edge_color=edge_colors, edge_width=edge_widths) # vertex_color=vertex_colors,edge_color=edge_colors

	return mismatches

def plot_graph_flat(
    value_arr,
    type_arr=None,
    type_to_color={0: "red", 1: "yellow", 2: "cyan", 3: "green", -1: "gray"},
    box_size=1,      # inches per cell (controls box size)
    text_ratio=0.5,    # fraction of cell height used for font size
    clip_after_per_type=None,  # max number of consecutive nodes of the same type to show
    target=None
):
    """
    Plot an integer array as colored boxes with numbers inside.

    Parameters
    ----------
    value_arr : array-like of ints, shape (H, W) or (N,)
        The integers to show as text inside the boxes.
    type_arr : array-like of ints, optional
        Same shape as value_arr. Determines the color of each box
        (e.g. node types). If None, value_arr is used.
    type_to_color : dict[int, str] or list/tuple of color strings, optional
        - If dict: maps each type value -> color (name or hex).
        - If list/tuple: used in order of sorted unique types.
          (len must be >= number of unique types)
    box_size : float, optional
        Size of each cell in inches (both width & height).
    text_ratio : float, optional
        Font size as fraction of cell height (0–1).
    clip_after_per_type : int or None, optional
        If set, only the first N consecutive nodes of the same type are shown.
        Additional nodes of that consecutive type are collapsed into a single
        placeholder box containing "...".
    """
    value_arr = np.asarray(value_arr)

    # Allow 1D arrays -> single row
    if value_arr.ndim == 1:
        value_arr = value_arr[np.newaxis, :]

    if type_arr is None:
        type_arr = value_arr.copy()
    else:
        type_arr = np.asarray(type_arr)
        if type_arr.ndim == 1:
            type_arr = type_arr[np.newaxis, :]
        if type_arr.shape != value_arr.shape:
            raise ValueError("type_arr must have same shape as value_arr")

    if clip_after_per_type is not None:
        if clip_after_per_type < 0:
            raise ValueError("clip_after_per_type must be non-negative")

        values_flat = value_arr.reshape(-1).tolist()
        types_flat = type_arr.reshape(-1).tolist()

        clipped_values = []
        clipped_types = []

        idx = 0
        while idx < len(values_flat):
            current_type = types_flat[idx]
            run_end = idx + 1
            while run_end < len(values_flat) and types_flat[run_end] == current_type:
                run_end += 1

            run_len = run_end - idx
            keep = min(run_len, clip_after_per_type)

            if keep:
                clipped_values.extend(values_flat[idx:idx + keep])
                clipped_types.extend(types_flat[idx:idx + keep])

            if run_len > clip_after_per_type:
                clipped_values.append("...")
                clipped_types.append(-1)

            idx = run_end

        value_arr = np.asarray(clipped_values, dtype=object)[np.newaxis, :]
        type_arr = np.asarray(clipped_types, dtype=int)[np.newaxis, :]

    H, W = value_arr.shape

    # Unique node types and mapping to 0..K-1 for colormap
    unique_types, inv = np.unique(type_arr, return_inverse=True)
    inv = inv.reshape(H, W)
    K = len(unique_types)

    # Build colors for each type
    if type_to_color is None:
        # default: use a discrete colormap
        cmap_base = plt.get_cmap("tab20", K)
        colors = [cmap_base(i) for i in range(K)]
    else:
        if isinstance(type_to_color, dict):
            colors = []
            for t in unique_types:
                if t in type_to_color:
                    colors.append(type_to_color[t])
                elif t == -1:
                    colors.append("gray")
                else:
                    raise KeyError(f"No color specified for type {t}")
        else:
            # assume list/tuple matching the number of unique types
            if len(type_to_color) < K:
                raise ValueError("Not enough colors for unique types")
            colors = list(type_to_color[:K])

    cmap = ListedColormap(colors)

    # Figure size so each cell has box_size inches
    figsize = (W * box_size, H * box_size)
    fig, ax = plt.subplots(figsize=figsize)

    # Show colored boxes
    ax.imshow(inv, cmap=cmap, interpolation="nearest", aspect="equal")

    # Draw grid lines around cells
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)
    ax.tick_params(
        which="both",
        bottom=False, left=False,
        labelbottom=False, labelleft=False
    )

    # Compute a reasonable font size from cell size
    # 1 inch = 72 points; each cell is box_size inches.
    fontsize = 72 * box_size * text_ratio

    # Put the integers as text in each cell
    for (i, j), v in np.ndenumerate(value_arr):
        ax.text(
            j, i, str(v),
            ha="center", va="center",
            color="black",
            fontsize=fontsize,
        )

    plt.tight_layout()

    if target is not None:
        plt.savefig(target)
    else:
        plt.show()

	


##
# output:
# RxC cabinet <-> room
# CxT thing <-> cabinet
# PxR room <-> person

#DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5), HCProblem(10, 20, 100, 10),
#			   HCProblem(15, 30, 150, 15)]

DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5),
				 HCProblem(10, 20, 100, 10),
				 HCProblem(15, 30, 150, 15),
				 HCProblem(50, 100, 500, 50)]

#DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5, perm) for perm in permutations(range(5))]

VERTEX_LABELS = {0: "R", 1: "C", 2: "T", 3: "P", -1: "_"}
VERTEX_COLORS = {0: "red", 1: "yellow", 2: "cyan", 3: "green", -1: "gray"}

class HCPDatasetGenerator(BaseDatasetGenerator):
	"""
	Class for generating datasets for the House Configuration Problem
	"""
	def __init__(self, config):
		super().__init__(config)


		self.graph_config = {
			"n_train": 1,
			"n_val": 1,
			"n_test": 1,
					}

		print(f'\nGenerating HCP {self.mode} dataset "{self.dataset_name}" with {self.graph_config[f"n_{self.mode}"]} instances!\n')

	def generate_dataset(self):
		"""
		Generate HCP instances for the dataset
		"""
		solutions = {
			"Energies": [],
			"H_graphs": [],
			"gs_bins": [],
			"graph_sizes": [],
			"densities": [],
			"runtimes": [],
			"upperBoundEnergies": [],
			"compl_H_graphs": [],
		}
		edges, nodes = 0, 0
		for idx, problem in enumerate(DUMMY_SAMPLES):
			problem.plot(f"input_sample_{idx}_input.png", include_legend=False)
			g = problem.igraph

			globals = problem.globals
			bin_solution = problem.solution_bin

			plot(g, globals["node_types"], f"input_sample_{idx}_solution.png", include_legend=False, bin_solution_edge=bin_solution)

			H_graph, density, graph_size = self.igraph_to_jraph(g)
			H_graph = H_graph._replace(globals=globals)


			#Energy, boundEnergy, solution, runtime, H_graph_compl = self.solve_graph(H_graph, g)
			Energy, boundEnergy, solution, runtime, compl_H_graph = self.solve_graph(H_graph,g)

			#H_graph = GraphWithMeta(graph=H_graph, meta={"rooms": problem.rooms, "cabinets": problem.cabinets, "things": problem.things, "persons": problem.persons, "id": idx,
		#										"offset_rooms": problem.OFFSET_ROOMS, "offset_cabinets": problem.OFFSET_CABINETS, "offset_things": problem.OFFSET_THINGS_CABINETS, "offset_persons": problem.OFFSET_THINGS_PERSONS})
			
			H_graph = problem.meta_graph


			solutions["Energies"].append(Energy + 0.0001)
			solutions["H_graphs"].append(H_graph)
			solutions["gs_bins"].append(solution)
			solutions["graph_sizes"].append(graph_size)
			solutions["densities"].append(density)
			solutions["runtimes"].append(runtime)
			solutions["upperBoundEnergies"].append(boundEnergy + 0.0001)
			solutions["compl_H_graphs"].append(compl_H_graph)

			indexed_solution_dict = {}
			for key in solutions.keys():
				if len(solutions[key]) > 0:
					indexed_solution_dict[key] = solutions[key][idx]
			self.save_instance_solution(indexed_solution_dict, idx)
		self.save_solutions(solutions)
