from dataclasses import dataclass
import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from DatasetCreator.jraph_utils import utils as jutils
from tqdm import tqdm
import numpy as np
import igraph as ig

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
	
	@staticmethod
	def get_node_type(node_idx: int):
		return sum([node_idx >= x for x in [HCProblem.OFFSET_ROOMS, HCProblem.OFFSET_CABINETS, HCProblem.OFFSET_THINGS, HCProblem.OFFSET_PERSONS]]) - 1
	
	@property
	def globals(self) -> list[int]:
		ans = np.array([-1] * HCProblem.N_NODES)
		ans[HCProblem.OFFSET_ROOMS: HCProblem.OFFSET_ROOMS + self.rooms] = 0
		ans[HCProblem.OFFSET_CABINETS: HCProblem.OFFSET_CABINETS + self.cabinets] = 1
		ans[HCProblem.OFFSET_THINGS: HCProblem.OFFSET_THINGS + self.things] = 2
		ans[HCProblem.OFFSET_PERSONS: HCProblem.OFFSET_PERSONS + self.persons] = 3
		return ans


##
# output:
# RxC cabinet <-> room
# CxT thing <-> cabinet
# PxR room <-> person

DUMMY_SAMPLES = [HCProblem(5, 10, 50, 5, [[p * 10 + i for i in range(10)] for p in range(5)]), HCProblem(10, 20, 100, 10, [[p * 10 + i for i in range(10)] for p in range(10)]),
			   HCProblem(15, 30, 150, 15, [[p * 10 + i for i in range(10)] for p in range(15)])]



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

		for idx, problem in enumerate(DUMMY_SAMPLES):
			g = problem.igraph

			globals = problem.globals

			H_graph, density, graph_size = self.igraph_to_jraph(g, double_edges=False, globals=globals)

			#Energy, boundEnergy, solution, runtime, H_graph_compl = self.solve_graph(H_graph, g)

			Energy, boundEnergy, solution, runtime, compl_H_graph = self.solve_graph(H_graph,g)


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





