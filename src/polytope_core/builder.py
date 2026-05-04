from .polytope import Polytope
import json
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from numpy.typing import NDArray

class PolytopeBuilder:
    @staticmethod
    def serialize(polytope: Polytope, filepath: str) -> None:
        data = {
            "points": polytope.points.tolist(),
            "simplices": polytope.simplices.tolist(),
            "normals": polytope.normals.tolist(),
            "neighbors": nx.to_dict_of_lists(polytope.neigh_graph),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def deserialize(filepath: str) -> Polytope:
        with open(filepath, "r") as f:
            data = json.load(f)

        neigh_graph = nx.from_dict_of_lists(data["neighbors"])

        return Polytope(
            points=np.array(data["points"], dtype=np.float64),
            simplices=np.array(data["simplices"], dtype=np.intp),
            normals=np.array(data["normals"], dtype=np.float64),
            neigh_graph=neigh_graph,
        )

    @staticmethod
    def random(num_points = 20):
        points = np.random.randn(num_points, 4)
        hull = ConvexHull(points)

        points_on_hull = points[hull.vertices] 

        lookup = np.zeros(num_points + 1, dtype=int)
        lookup[hull.vertices] = np.arange(len(hull.vertices))
        new_simplices = lookup[hull.simplices]

        neigh_graph = _build_neigh_graph(hull.neighbors)
        return Polytope(points_on_hull, new_simplices, hull.equations[:, :4], neigh_graph)

    @staticmethod
    def simplex4():
        vertices = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.float64)

        hull = ConvexHull(vertices)
        neigh_graph = _build_neigh_graph(hull.neighbors)
        return Polytope(vertices, hull.simplices, hull.equations[:, :4], neigh_graph)


def _build_neigh_graph(neighbors: NDArray[np.integer]) -> nx.Graph:
    g = nx.Graph()
    for node, neighs in enumerate(neighbors):
        g.add_node(int(node))
        for neigh in neighs:
            g.add_edge(int(node), int(neigh))
    return g




