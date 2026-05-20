from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq
from .big_area_fs import compute_area

class SmallAreafs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        graph = polytope.neigh_graph.copy()

        for facet_1, facet_2 in graph.edges():
            shared = list(set(polytope.simplices[facet_1]) & set(polytope.simplices[facet_2]))
            area = compute_area(polytope.points[shared])
            graph[facet_1][facet_2]['ridge_area'] = area

        return nx.minimum_spanning_tree(graph, weight='ridge_area')



