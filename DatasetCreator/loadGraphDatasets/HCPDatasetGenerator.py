from dataclasses import dataclass
import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from DatasetCreator.jraph_utils import utils as jutils
from tqdm import tqdm
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from graph_with_metadata import GraphWithMeta

@dataclass
class HCProblem:

	rooms: int
	cabinets: int
	things: int
	persons: int
	things_of_person: list[list[int]]
	
	MAX_ROOMS: int = 15
	MAX_CABINETS: int = 30
	MAX_THINGS: int = 150
	MAX_PERSONS: int = 15
	
	OFFSET_ROOMS = 0
	OFFSET_CABINETS = MAX_ROOMS
	OFFSET_THINGS = OFFSET_CABINETS + MAX_CABINETS
	OFFSET_PERSONS = OFFSET_THINGS + MAX_THINGS

	N_NODES = OFFSET_PERSONS + MAX_PERSONS

	@property
	def igraph(self):
		
		edges_RxC = [[r + HCProblem.OFFSET_ROOMS, c + HCProblem.OFFSET_CABINETS] for r in range(self.rooms) for c in range(self.cabinets)]
		edges_CxT = [[c + HCProblem.OFFSET_CABINETS, t + HCProblem.OFFSET_THINGS] for c in range(self.cabinets) for t in range(self.things)]
		edges_PxR = [[p + HCProblem.OFFSET_PERSONS, r + HCProblem.OFFSET_ROOMS] for p in range(self.persons) for r in range(self.rooms)]
		edges_TxP = [[t + p * 10 + HCProblem.OFFSET_THINGS, p + HCProblem.OFFSET_PERSONS] for p in range(self.persons) for t in range(10) ]
		all_edges = edges_RxC + edges_CxT + edges_PxR + edges_TxP
		
		return ig.Graph(n=HCProblem.N_NODES, edges=all_edges)
	
	
	@property
	def globals(self) -> dict[str, np.ndarray]:
		node_types = np.array([-1] * HCProblem.N_NODES)
		node_types[HCProblem.OFFSET_ROOMS: HCProblem.OFFSET_ROOMS + self.rooms] = 0
		node_types[HCProblem.OFFSET_CABINETS: HCProblem.OFFSET_CABINETS + self.cabinets] = 1
		node_types[HCProblem.OFFSET_THINGS: HCProblem.OFFSET_THINGS + self.things] = 2
		node_types[HCProblem.OFFSET_PERSONS: HCProblem.OFFSET_PERSONS + self.persons] = 3

		igraph = self.igraph
		senders, receivers = zip(*igraph.get_edgelist())
		senders = np.array(senders)
		receivers = np.array(receivers)

		sender_types = node_types[senders]
		receiver_types = node_types[receivers]
		n_edges = senders.shape[0]
		# Assign group ids: -1 for non-mutually-exclusive edges
		group_ids = -np.ones(n_edges, dtype=np.int32)
		receivers_int = np.asarray(receivers, dtype=np.int32)
		senders_int = np.asarray(senders, dtype=np.int32)

		# Cabinet-room: group by cabinet (receiver type 1, sender type 0)
		mask_cabinet_room = np.asarray((sender_types == 0) & (receiver_types == 1), dtype=bool)
		group_ids = np.where(mask_cabinet_room, receivers_int, group_ids)
		# Thing-cabinet: group by thing (sender type 2, receiver type 1)
		mask_thing_cabinet = np.asarray((sender_types == 1) & (receiver_types == 2), dtype=bool)
		group_ids = np.where(mask_thing_cabinet, receivers_int, group_ids)
		# Room-person: group by room (sender type 0, receiver type 3)
		mask_room_person = np.asarray((sender_types == 0) & (receiver_types == 3), dtype=bool)
		group_ids = np.where(mask_room_person, senders_int, group_ids)

		return {"node_types": node_types, "group_ids": group_ids,} #  "rooms": self.rooms, "cabinets": self.cabinets, "things": self.things, "persons": self.persons


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

def plot_graph(igraph, globals, n):
	omit_nodes = (globals == -1).cumsum()
	globals_compact = globals[np.where(globals > -1)]
	edges = igraph.get_edgelist()
	edges = [(a - omit_nodes[a], b - omit_nodes[b]) for a, b in edges]
	plot_graph = ig.Graph(edges=edges)

	things = [i for i in range(len(globals_compact)) if globals_compact[i] == 2]
	rooms = [i for i in range(len(globals_compact)) if globals_compact[i] == 0]
	cabinets = [i for i in range(len(globals_compact)) if globals_compact[i] == 1]


	owners_of_things = {}

	for thing in things:
		connections_to_persons = sum(1 for e in edges if e[0] == thing and globals_compact[e[1]] == 3)
		connections_to_cabinets = sum(1 for e in edges if e[1] == thing and globals_compact[e[0]] == 1)
		#print(f"Thing {thing} has {connections_to_persons} connections to persons and {connections_to_cabinets} connections to cabinets")
		owners_of_things[thing] = [e[1] for e in edges if e[0] == thing and globals_compact[e[1]] == 3][0]
	
	owners_of_rooms = {}

	for room in rooms:
		connections_to_persons = sum(1 for e in edges if e[0] == room and globals_compact[e[1]] == 3)
		connections_to_cabinets = sum(1 for e in edges if e[0] == room and globals_compact[e[1]] == 1)
		#print(f"Room {room} has {connections_to_persons} connections to persons and {connections_to_cabinets} connections to cabinets")
		owners_of_rooms[room] = [e[1] for e in edges if e[0] == room and globals_compact[e[1]] == 3][0]

	owners_of_cabinets = {}

	for cabinet in cabinets:
		owners_of_cabinets[cabinet] = [owners_of_rooms[e[0]] for e in edges if e[1] == cabinet and globals_compact[e[0]] == 0][0]

	holders_of_things = {}

	for thing in things:
		holders_of_things[thing] = [owners_of_cabinets[e[0]] for e in edges if e[1] == thing and globals_compact[e[0]] == 1][0]

	mismatches = sum(1 for k in owners_of_things if owners_of_things[k] != holders_of_things[k])

	print(f"Mismatches: {mismatches}")

	edge_colors = ["red" if globals_compact[e[1]] == 2 and globals_compact[e[0]] == 1 else "black" for e in edges]
	vertex_colors = ["black" if globals_compact[i] != 2 else ("red" if owners_of_things[i] != holders_of_things[i] else "green") for i in range(len(globals_compact))]
	ig.plot(plot_graph, vertex_label=[VERTEX_LABELS[t] for i, t in enumerate(globals_compact)], target=f"plot_{n}.png",) # vertex_color=vertex_colors,edge_color=edge_colors

	return mismatches

def solve_plot_graph(igraph, globals):
	omit_nodes = (globals == -1).cumsum()
	globals_compact = globals[np.where(globals > -1)]
	edges = igraph.get_edgelist()
	edges = [(a - omit_nodes[a], b - omit_nodes[b]) for a, b in edges]

	edges = [e for e in edges if globals_compact[e[0]] == 2 and globals_compact[e[1]] == 3]

	things = [i for i in range(len(globals_compact)) if globals_compact[i] == 2]
	cabinets = [i for i in range(len(globals_compact)) if globals_compact[i] == 1]
	persons = [i for i in range(len(globals_compact)) if globals_compact[i] == 3]
	rooms = [i for i in range(len(globals_compact)) if globals_compact[i] == 0]
	
	ci = 0
	for thing in things:
		edges.append((thing, cabinets[ci // 5]))
		ci += 1
	ri = 0
	for c in cabinets:
		edges.append((c, rooms[ri // 2]))
		ri += 1

	for r, p in zip(rooms, persons):
		edges.append((r, p))

	plot_graph = ig.Graph(edges=edges)

	#things = [i for i in range(len(globals_compact)) if globals_compact[i] == 2]

	#for thing in things:
	#	connections_to_persons = sum(1 for e in edges if e[0] == thing and globals_compact[e[1]] == 3)
	#	connections_to_cabinets = sum(1 for e in edges if e[1] == thing and globals_compact[e[0]] == 1)
	#	print(f"Thing {thing} has {connections_to_persons} connections to persons and {connections_to_cabinets} connections to cabinets")
	edge_colors = ["red" if globals_compact[e[1]] == 2 and globals_compact[e[0]] == 1 else "black" for e in edges]
	ig.plot(plot_graph, vertex_label=[VERTEX_LABELS[t] for i, t in enumerate(globals_compact)], target="plot.png",) # edge_color=edge_colors

def solve_graph(igraph, globals):
	edges = igraph.get_edgelist()

	edges = [e for e in edges if globals[e[0]] == 2 and globals[e[1]] == 3]

	things = [i for i in range(len(globals)) if globals[i] == 2]
	cabinets = [i for i in range(len(globals)) if globals[i] == 1]
	persons = [i for i in range(len(globals)) if globals[i] == 3]
	rooms = [i for i in range(len(globals)) if globals[i] == 0]

	ci = 0
	for thing in things:
		edges.append((cabinets[ci // 5], thing))
		ci += 1
	ri = 0
	for c in cabinets:
		edges.append((rooms[ri // 2], c))
		ri += 1

	for r, p in zip(rooms, persons):
		edges.append((r, p))

	return edges

def group_ids_from_graph(jraph_graph_list): # todo: can be deleted
    graph = jraph_graph_list["graphs"][0]
    senders = graph.senders.astype(np.int32)
    receivers = graph.receivers.astype(np.int32)
    globals_ = graph.globals.astype(np.int32)
    sender_types = globals_[senders]
    receiver_types = globals_[receivers]
    n_edges = senders.shape[0]
    # Assign group ids: -1 for non-mutually-exclusive edges
    group_ids = -np.ones(n_edges, dtype=np.int32)
    receivers_int = np.asarray(receivers, dtype=np.int32)

    # Cabinet-room: group by cabinet (receiver type 1, sender type 0)
    mask_cabinet_room = np.asarray((sender_types == 0) & (receiver_types == 1), dtype=bool)
    group_ids = np.where(mask_cabinet_room, receivers_int, group_ids)
    # Thing-cabinet: group by thing (sender type 2, receiver type 1)
    mask_thing_cabinet = np.asarray((sender_types == 1) & (receiver_types == 2), dtype=bool)
    group_ids = np.where(mask_thing_cabinet, receivers_int, group_ids)
    # Room-person: group by room (sender type 0, receiver type 3)
    mask_room_person = np.asarray((sender_types == 0) & (receiver_types == 3), dtype=bool)
    group_ids = np.where(mask_room_person, receivers_int, group_ids)
    return group_ids

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
			g = problem.igraph
			#edges += g.ecount()
			#nodes += g.vcount()
			globals = problem.globals

			edges = solve_graph(g, globals["node_types"])
			globals["solution"] = np.array([1 if e in set(edges) else 0 for e in g.get_edgelist()])

			H_graph, density, graph_size = self.igraph_to_jraph(g, double_edges=False, globals=globals)


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





