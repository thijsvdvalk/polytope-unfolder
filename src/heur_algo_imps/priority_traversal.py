from numpy.typing import NDArray
from polytope_core.heur_algo_abc import HeuristicAlgorithm, PriorityTraversalConfig
import networkx as nx
from polytope_core.polytope import Polytope
import numpy as np
import heapq as hq


class PriorityTraversal(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope, config: PriorityTraversalConfig) -> nx.Graph:
        g = polytope.neigh_graph
        max = -1 if config.max_is_priority else False
        a, b = config.volume, config.aspect_ratio

        min_node: tuple[int, float] | None = None
        for node, data in g.nodes(data=True):
            data['weight'] = (data['volume'] * a + data['aspect_ratio'] * b) * max
            if min_node is None:
                min_node = node, data['weight']
            elif data['weight'] < min_node[1]:
                min_node = node, data['weight']


        visited = set()
        pq = [(min_node[1], min_node[0], None)]
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
                    hq.heappush(pq, (g.nodes[neighbor]['weight'], neighbor, node))

        return st

