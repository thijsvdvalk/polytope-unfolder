from abc import ABC, abstractmethod
from polytope_core.polytope import Polytope
import networkx as nx
from dataclasses import dataclass

class HeuristicAlgorithm(ABC):
    @classmethod
    def spanning_tree(cls, polytope: Polytope, config: HeuristicConfig) -> nx.Graph:
        st = cls._spanning_tree(polytope, config)
        ng = polytope.neigh_graph

        assert set(st.nodes) == set(ng.nodes)
        assert nx.is_connected(st)
        assert st.number_of_edges() == ng.number_of_nodes() - 1
        assert all(ng.has_edge(u, v) for u, v in st.edges())

        return st

    @classmethod
    @abstractmethod
    def _spanning_tree(cls, polytope: Polytope, config: HeuristicConfig) -> nx.Graph: ...



@dataclass
class HeuristicConfig:
    pass  # base, empty

@dataclass
class MSTConfig(HeuristicConfig):
    dihedral_angle: float 
    shared_face_area: float 
    centroids_distance: float

    def __str__(self):
        return f"MSTConfig: a={self.dihedral_angle}, b={self.shared_face_area}, c={self.centroids_distance}"


@dataclass
class PriorityTraversalConfig(HeuristicConfig):
    volume: float
    aspect_ratio: float

    def __str__(self):
        return f"PriorityTraversalConfig: a={self.volume}, b={self.aspect_ratio}"
