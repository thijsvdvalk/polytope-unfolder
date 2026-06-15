from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope


class Dfs(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope, config) -> nx.Graph:
        return nx.dfs_tree(polytope.neigh_graph, source=0).to_undirected()

    @classmethod
    def id(cls):
        return "DFS()"
