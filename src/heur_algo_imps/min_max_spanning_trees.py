from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm, HeuristicConfig, MSTConfig
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np


class MinSpanningTree(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope, config: MSTConfig) -> nx.Graph:
        assert polytope.edge_weights_initialized

        g = polytope.neigh_graph.copy()

        a, b, c = config.dihedral_angle, config.shared_face_area, config.centroids_distance
        for u, v in g.edges():
            g[u][v]['weight'] = g[u][v]['dihedral_angle'] * a + g[u][v]['shared_face_area'] * b + g[u][v]['centroids_distance'] * c

        return nx.minimum_spanning_tree(g, weight='weight')
