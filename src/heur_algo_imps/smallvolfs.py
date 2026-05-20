from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq
from .bigvolfs import tetrahedron_volume_4d

class SmallVolfs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        vols = np.zeros(len(polytope.simplices)) 
        for i, simp in enumerate(polytope.simplices):
            vols[i] = tetrahedron_volume_4d(polytope.points[simp])

        smallest_i = np.argmin(vols)
        
        visited = set()
        pq = [(vols[smallest_i], smallest_i, None)]
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


