from .polytope import Polytope
import json
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


class PolytopeBuilder:
    @staticmethod
    def serialize(polytope: Polytope, filepath: Path) -> None:
        data = {
            "points": polytope.points.tolist(),
            "simplices": polytope.simplices.tolist(),
            "normals": polytope.normals.tolist(),
            "neighbors": nx.to_dict_of_lists(polytope.neigh_graph),
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def deserialize(filepath: Path) -> Polytope:
        with open(filepath, "r") as f:
            data = json.load(f)

        neighbours = data["neighbors"]
        neighbours_ints = {int(k): v for k, v in neighbours.items()}
        neigh_graph = nx.from_dict_of_lists(neighbours_ints)

        return Polytope(
            points=np.array(data["points"], dtype=np.float64),
            simplices=np.array(data["simplices"], dtype=np.intp),
            normals=np.array(data["normals"], dtype=np.float64),
            neigh_graph=neigh_graph,
        )

    @staticmethod
    def random_normal(num_points: int) -> Polytope:
        points = np.random.randn(num_points, 4)
        return polytope_from_hull_vertices(points)

    @staticmethod
    def random_uniform(num_points: int) -> Polytope:
        points = np.random.uniform(-1, 1, (num_points, 4))
        return polytope_from_hull_vertices(points)

    @staticmethod
    def random_exponential(num_points: int) -> Polytope:
        points = np.random.exponential(scale=1, size=(num_points, 4))
        return polytope_from_hull_vertices(points)

    @staticmethod
    def uniform_on_hypersphere(num_points: int) -> Polytope:
        points = np.random.normal(0, 1, size=(num_points, 4))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return polytope_from_hull_vertices(points)

    @staticmethod
    def simplex4():
        vertices = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        return polytope_from_hull_vertices(vertices)

    @staticmethod
    def orthoplex4():
        vertices = np.array(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ]
        )
        return polytope_from_hull_vertices(vertices)


def _build_neigh_graph(neighbors: NDArray[np.integer]) -> nx.Graph:
    g = nx.Graph()
    for node, neighs in enumerate(neighbors):
        g.add_node(int(node))
        for neigh in neighs:
            g.add_edge(int(node), int(neigh))
    return g


def _remap_simplices(
    vertices: NDArray[np.float64],
    hull_vertices: NDArray[np.integer],
    hull_simplices: NDArray[np.integer],
) -> NDArray[np.integer]:
    lookup = np.zeros(len(vertices), dtype=int)
    lookup[hull_vertices] = np.arange(len(hull_vertices))
    return lookup[hull_simplices]


def polytope_from_hull_vertices(vertices: NDArray[np.float64]) -> Polytope:
    hull = ConvexHull(vertices)
    neigh_graph = _build_neigh_graph(hull.neighbors)

    if len(vertices) > len(hull.vertices):
        points_on_hull = vertices[hull.vertices]
        new_simplices = _remap_simplices(vertices, hull.vertices, hull.simplices)
        return Polytope(
            points_on_hull, new_simplices, hull.equations[:, :4], neigh_graph
        )

    return Polytope(vertices, hull.simplices, hull.equations[:, :4], neigh_graph)
