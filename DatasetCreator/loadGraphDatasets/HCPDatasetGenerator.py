from dataclasses import dataclass, field
import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from DatasetCreator.jraph_utils import utils as jutils
from tqdm import tqdm
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from GraphWithMeta import GraphWithMeta

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
	things_of_person: list[list[int]]
	OFFSET_ROOMS: int = field(init=False)
	OFFSET_CABINETS: int = field(init=False)
	OFFSET_THINGS: int = field(init=False)
	OFFSET_PERSONS: int = field(init=False)
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
		self.OFFSET_THINGS = self.OFFSET_CABINETS + self.cabinets
		self.OFFSET_PERSONS = self.OFFSET_THINGS + self.things
		self.N_NODES = self.OFFSET_PERSONS + self.persons
		self.OFFSET_PER_NODE_TYPE = np.array([self.OFFSET_PERSONS, self.OFFSET_ROOMS, self.OFFSET_CABINETS, -1])

		self.edges_RxC = [(r + self.OFFSET_ROOMS, c + self.OFFSET_CABINETS) for r in range(self.rooms) for c in range(self.cabinets)]
		self.edges_CxT = [(c + self.OFFSET_CABINETS, t + self.OFFSET_THINGS) for c in range(self.cabinets) for t in range(self.things)]
		self.edges_PxR = [(p + self.OFFSET_PERSONS, r + self.OFFSET_ROOMS) for p in range(self.persons) for r in range(self.rooms)]
		self.edges_TxP = [(t + p * 10 + self.OFFSET_THINGS, p + self.OFFSET_PERSONS) for p in range(self.persons) for t in range(10)]  # default ownership - 10 things per person
		self.all_edges = sorted(self.edges_RxC + self.edges_CxT + self.edges_PxR + self.edges_TxP)

		self.igraph = ig.Graph(n=self.OFFSET_PERSONS, edges=self.all_edges)

		# Initialize the new fields
		self.node_types = self._initialize_node_types()
		self.classes_per_node = self._initialize_classes_per_node()
		self.solution_edges, self.solution_bin, self.solution_nodes = self._initialize_solution()
		self.neighbours_per_node = self._init_neighbours_per_node()
		self.nth_of_type = self._init_nth_of_type()

		self.globals = {"node_types": self.node_types, "solution_bin": self.solution_bin, "classes_per_node": self.classes_per_node, "neighbours_per_node": self.neighbours_per_node, "solution_nodes": self.solution_nodes,
				  "offset_per_node_type": self.OFFSET_PER_NODE_TYPE[self.node_types], "nth_of_type": self.nth_of_type}

	def _init_neighbours_per_node(self) -> list[list[int]]:
		neighbours = [to_shape([[self.OFFSET_PERSONS + p for p in range(self.persons)] for _ in range(self.rooms)], (self.rooms, self.cabinets)),
				to_shape([[self.OFFSET_ROOMS + r for r in range(self.rooms)] for _ in range(self.cabinets)], (self.cabinets, self.cabinets)),
				to_shape([[self.OFFSET_CABINETS + c for c in range(self.cabinets)] for _ in range(self.things)], (self.things, self.cabinets)),
				to_shape([self.persons * [-1]], (self.persons, self.cabinets))]
		
		return np.concatenate(neighbours, 0)
	
	def _init_nth_of_type(self) -> np.ndarray:
		nth_of_type = np.zeros(len(self.node_types), dtype=int)
		type_counts = {}
		for i, node_type in enumerate(self.node_types):
			nth_of_type[i] = type_counts.get(node_type, 0)
			type_counts[node_type] = type_counts.get(node_type, 0) + 1
		return nth_of_type

	def _initialize_node_types(self) -> np.ndarray:
		return np.array(
			[0] * self.rooms +
			[1] * self.cabinets +
			[2] * self.things +
			[3] * self.persons
		)

	def _initialize_classes_per_node(self) -> np.ndarray:
		classes_per_node_type = np.array([
			self.persons,  # rooms connected to persons
			self.rooms,    # cabinets connected to rooms
			self.cabinets, # things connected to cabinets
			1              # persons connected to things
		])
		return classes_per_node_type[self.node_types]

	def _initialize_solution(self) -> tuple[list[tuple[int, int]], list[int]]:
		edges = list(self.edges_TxP)

		solution_nodes = np.full(self.N_NODES, -1, dtype=int)
		rooms = [r for r in range(self.rooms)]
		cabinets = [c + self.OFFSET_CABINETS for c in range(self.cabinets)]
		things = [t + self.OFFSET_THINGS for t in range(self.things)]
		persons = [p + self.OFFSET_PERSONS for p in range(self.persons)]

		ci = 0
		for thing in things:
			solution_nodes[thing] = ci // 5
			edges.append((cabinets[ci // 5], thing))
			ci += 1
		ri = 0
		for c in cabinets:
			solution_nodes[c] = ri // 2 
			edges.append((rooms[ri // 2], c))
			ri += 1

		for r, p in zip(rooms, persons):
			solution_nodes[r] = p - self.OFFSET_PERSONS
			edges.append((p, r))

		edges.sort()
		
		j = 0
		bin_solution = []
		for i in range(len(self.all_edges)):
			if j < len(edges) and self.all_edges[i] == edges[j]:
				bin_solution.append(1)
				j += 1
			else:
				bin_solution.append(0)
		assert j == len(edges)

		return np.array(edges), np.array(bin_solution), solution_nodes
	
	def plot(self, target, include_legend=False):
		return plot(self.igraph, self.node_types, target, include_legend)
	
def plot(igraph, node_types, target, include_legend=False, bin_solution_edge=None, solution_nodes = None, verbose=False, meta_graph=None):
	if solution_nodes is not None:
		offset_node_types = meta_graph.globals["offset_per_node_type"].squeeze()
		edges = [(i, int(j) + offset_node_types[i]) for i, j in enumerate(solution_nodes[:meta_graph.meta["offset_persons"]]) if int(j) != -1]
		things = [i for i in range(len(node_types)) if node_types[i] == 2]
		persons = [i for i in range(len(node_types)) if node_types[i] == 3]
		for i, t in enumerate(things):
			edges.append((t, persons[i//10]))
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


	if include_legend:
		plot_graph.add_vertices(4)
		vertex_color_appendix = [VERTEX_COLORS[t] for t in list(VERTEX_LABELS.keys())[:-1]]
		vertex_label_appendix = [VERTEX_LABELS[t] for t in list(VERTEX_LABELS.keys())[:-1]]

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

	edge_colors = ["black" if node_types[e[0]] == 2 and node_types[e[1]] == 3 else "grey" for e in edges]
	edge_widths = [3 if node_types[e[0]] == 2 and node_types[e[1]] == 3 else 2 for e in edges]

	vertex_colors = [VERTEX_COLORS[t] for t in node_types] + vertex_color_appendix #+ list(VERTEX_LABELS.keys())[:-1]
	vertex_labels = [i for i, t in enumerate(node_types)] + vertex_label_appendix#+ list(VERTEX_LABELS.values())[:-1]
	ig.plot(plot_graph, vertex_label=vertex_labels, target=target, vertex_color=vertex_colors,edge_color=edge_colors, edge_width=edge_widths) # vertex_color=vertex_colors,edge_color=edge_colors

	return mismatches

	


##
# output:
# RxC cabinet <-> room
# CxT thing <-> cabinet
# PxR room <-> person

#DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5, [[p * 10 + i for i in range(10)] for p in range(5)]), HCProblem(10, 20, 100, 10, [[p * 10 + i for i in range(10)] for p in range(10)]),
#			   HCProblem(15, 30, 150, 15, [[p * 10 + i for i in range(10)] for p in range(15)])]

DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5, [[p * 10 + i for i in range(10)] for p in range(5)]),
				 HCProblem(10, 20, 100, 10, [[p * 10 + i for i in range(10)] for p in range(10)]),
				 HCProblem(15, 30, 150, 15, [[p * 10 + i for i in range(10)] for p in range(15)])]

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

			H_graph = GraphWithMeta(graph=H_graph, meta={"rooms": problem.rooms, "cabinets": problem.cabinets, "things": problem.things, "persons": problem.persons, "id": idx,
												"offset_rooms": problem.OFFSET_ROOMS, "offset_cabinets": problem.OFFSET_CABINETS, "offset_things": problem.OFFSET_THINGS, "offset_persons": problem.OFFSET_PERSONS})


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




