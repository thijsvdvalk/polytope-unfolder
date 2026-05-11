from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope


class Bfs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        return nx.bfs_tree(polytope.neigh_graph, source=0).to_undirected()
