from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope


class Dfs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope, source: int = 0) -> nx.Graph:
        return nx.dfs_tree(polytope.neigh_graph, source=source).to_undirected()
