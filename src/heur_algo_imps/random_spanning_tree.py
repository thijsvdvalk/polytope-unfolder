from polytope_core.heur_algo_abc import HeuristicAlgorithm
import networkx as nx
from polytope_core.polytope import Polytope


class RandomSpanningTree(HeuristicAlgorithm):
    @classmethod
    def _spanning_tree(cls, polytope: Polytope, config) -> nx.Graph:
        return nx.random_spanning_tree(polytope.neigh_graph)
