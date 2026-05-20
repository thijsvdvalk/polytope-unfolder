from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq


class BigAreafs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        graph = polytope.neigh_graph.copy()

        for facet_1, facet_2 in graph.edges():
            shared = list(set(polytope.simplices[facet_1]) & set(polytope.simplices[facet_2]))
            area = compute_area(polytope.points[shared])
            graph[facet_1][facet_2]['ridge_area'] = area

        return nx.maximum_spanning_tree(graph, weight='ridge_area')



def compute_area(triangle: NDArray[np.float64]):
    # triangle: (3, 4) array, 3 vertices each with 4 coordinates
    a, b, c = triangle
    ab = b - a
    ac = c - a
    # Area = 0.5 * sqrt(|ab|^2 * |ac|^2 - (ab . ac)^2)
    # This is from the identity |u x v|^2 = |u|^2|v|^2 - (u.v)^2
    area = 0.5 * np.sqrt(np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac)**2)
    return area
