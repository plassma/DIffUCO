from dataclasses import dataclass, field
import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from DatasetCreator.jraph_utils import utils as jutils
from tqdm import tqdm
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from utils.GraphWithMeta import GraphWithMeta

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
	N_NODES: int = field(init=False)

	def __post_init__(self):
		self.OFFSET_ROOMS = 0
		self.OFFSET_CABINETS = self.OFFSET_ROOMS + self.rooms
		self.OFFSET_THINGS = self.OFFSET_CABINETS + self.cabinets
		self.OFFSET_PERSONS = self.OFFSET_THINGS + self.things
		self.N_NODES = self.OFFSET_PERSONS + self.persons

		self.edges_RxC = [(r + self.OFFSET_ROOMS, c + self.OFFSET_CABINETS) for r in range(self.rooms) for c in range(self.cabinets)]
		self.edges_CxT = [(c + self.OFFSET_CABINETS, t + self.OFFSET_THINGS) for c in range(self.cabinets) for t in range(self.things)]
		self.edges_PxR = [(p + self.OFFSET_PERSONS, r + self.OFFSET_ROOMS) for p in range(self.persons) for r in range(self.rooms)]
		self.edges_TxP = [(t + p * 10 + self.OFFSET_THINGS, p + self.OFFSET_PERSONS) for p in range(self.persons) for t in range(10)] # default ownership - 10 things per person
		self.all_edges = sorted(self.edges_RxC + self.edges_CxT + self.edges_PxR + self.edges_TxP)

		self.igraph = ig.Graph(n=self.N_NODES, edges=self.all_edges)


	@property
	def globals(self) -> dict[str, np.ndarray]:
		return {"node_types": self.node_types, "solution": self.solution_edges}
	
	@property
	def node_types(self) -> np.ndarray:
		return np.array(
            [0] * self.rooms +
            [1] * self.cabinets +
            [2] * self.things +
            [3] * self.persons
        )
	
	@property
	def solution_edges(self):
		edges = list(self.edges_TxP)

		rooms = [r for r in range(self.rooms)]
		cabinets = [c + self.OFFSET_CABINETS for c in range(self.cabinets)]
		things = [t + self.OFFSET_THINGS for t in range(self.things)]
		persons = [p + self.OFFSET_PERSONS for p in range(self.persons)]

		ci = 0
		for thing in things:
			edges.append((cabinets[ci // 5], thing))
			ci += 1
		ri = 0
		for c in cabinets:
			edges.append((rooms[ri // 2], c))
			ri += 1

		for r, p in zip(rooms, persons):
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

		return edges, bin_solution
	
	def plot(self, target, include_legend=False):
		return plot(self.igraph, self.node_types, target, include_legend)
	
def plot(igraph, node_types, target, include_legend=False, bin_solution=None, verbose=False):
	edges = igraph.get_edgelist()
	edges = [(a, b) for i, (a, b) in enumerate(edges) if not bin_solution or bin_solution[i]]
	plot_graph = ig.Graph(edges=edges)

	vertex_label_appendix = []
	vertex_color_appendix = []


	if include_legend:
		plot_graph.add_vertices(4)
		vertex_color_appendix = [VERTEX_COLORS[t] for t in list(VERTEX_LABELS.keys())[:-1]]
		vertex_label_appendix = [VERTEX_LABELS[t] for t in list(VERTEX_LABELS.keys())[:-1]]

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


	if verbose:
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
			edges, bin_solution = problem.solution_edges

			#plot(g, globals["node_types"], f"input_sample_{idx}_solution.png", include_legend=False, bin_solution=bin_solution)

			H_graph, density, graph_size = self.igraph_to_jraph(g)
			H_graph = H_graph._replace(globals=globals)


			#Energy, boundEnergy, solution, runtime, H_graph_compl = self.solve_graph(H_graph, g)
			Energy, boundEnergy, solution, runtime, compl_H_graph = self.solve_graph(H_graph,g)

			H_graph = GraphWithMeta(graph=H_graph, meta={"rooms": problem.rooms, "cabinets": problem.cabinets, "things": problem.things, "persons": problem.persons, "id": idx})


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




