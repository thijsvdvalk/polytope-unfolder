from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq

class BigVolfs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        vols = np.zeros(len(polytope.simplices)) 
        for i, simp in enumerate(polytope.simplices):
            vols[i] = tetrahedron_volume_4d(polytope.points[simp])

        biggest_i = np.argmax(vols)
        
        visited = set()
        pq = [(-vols[biggest_i], biggest_i, None)]
        st = nx.Graph()

        while pq:
            _, node, parent = hq.heappop(pq)
            if node in visited: 
                continue

            visited.add(node)

            if parent is not None:
                st.add_edge(parent, node)

            for neighbor in polytope.neigh_graph.neighbors(node):
                if neighbor not in visited:
                    hq.heappush(pq, (-vols[neighbor], neighbor, node))

        return st


def tetrahedron_volume_4d(verts):
    a, b, c, d = verts
    # 3 edge vectors, each of length 4
    mat = np.array([a - d, b - d, c - d])  # shape (3, 4)
    # Gram matrix: dot products of all edge vector pairs
    gram = mat @ mat.T                      # shape (3, 3)
    return np.sqrt(abs(np.linalg.det(gram))) / 6
