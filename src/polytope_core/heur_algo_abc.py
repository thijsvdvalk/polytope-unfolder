from abc import ABC, abstractmethod
from polytope_core.polytope import Polytope
import networkx as nx


class HeuristicAlgorithm(ABC):
    @classmethod
    def spanning_tree(cls, polytope: Polytope) -> nx.Graph:
        st = cls._spanning_tree(polytope)
        ng = polytope.neigh_graph

        assert set(st.nodes) == set(ng.nodes)
        assert nx.is_connected(st)
        assert st.number_of_edges() == ng.number_of_nodes() - 1
        assert all(ng.has_edge(u, v) for u, v in st.edges())

        return st

    @classmethod
    @abstractmethod
    def _spanning_tree(cls, polytope: Polytope) -> nx.Graph: ...
