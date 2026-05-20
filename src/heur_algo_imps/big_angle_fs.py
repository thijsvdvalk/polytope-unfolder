from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq


class BigAnglefs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        graph = polytope.neigh_graph.copy()

        for facet_1, facet_2 in graph.edges():
            angle = compute_dihedral_angle(polytope.normals[facet_1], polytope.normals[facet_2])
            graph[facet_1][facet_2]['dihedral_angle'] = angle

        return nx.maximum_spanning_tree(graph, weight='dihedral_angle')

def compute_dihedral_angle(n1, n2):
    cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)
