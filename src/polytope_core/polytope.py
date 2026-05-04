from __future__ import annotations
from scipy.spatial import ConvexHull
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import json
from dataclasses import dataclass
from itertools import combinations
from .overlap import overlaps

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

w = np.array([0, 0, 0, 1])


@dataclass(frozen=True)
class Polytope:
    points: NDArray[np.float64]
    simplices: NDArray[np.integer]
    normals: NDArray[np.float64]
    neigh_graph: nx.Graph

    def __post_init__(self):
        self.points.flags.writeable = False
        self.simplices.flags.writeable = False
        self.normals.flags.writeable = False


    def unfold(self, traversal: list[tuple[int, int]]) -> Net:
        cells: list[Cell | None] = [None] * len(self.simplices)
        first = traversal[0][0]
        cells[first] = compute_first_cell(
            self.points[self.simplices[first]],
            self.normals[first],
        )
        for prev, cur in traversal:
            cells[cur] = compute_new_cell(
                _assert_cell(cells[prev]),
                self.points[self.simplices[cur]],
                self.normals[cur],
            )
        return Net(self, traversal, cells)

@dataclass
class Net:
    polytope: Polytope
    traversal: list[tuple[int, int]]
    cells: list[Cell | None]

    def overlaps(self) -> bool:
        traversal_set = set(self.traversal)
        n = len(self.cells)
        to_check = [x for x in list(combinations(range(n), 2)) if x not in traversal_set]
        
        return any(
            overlaps(self[i].net_vertices_3d, self[j].net_vertices_3d)
            for i, j in to_check
        )

    def __getitem__(self, idx: int) -> Cell:
        return _assert_cell(self.cells[idx])


def find_shared_vertices(
    verts_a: NDArray[np.float64],
    verts_b: NDArray[np.float64],
    atol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns (shared_in_a, shared_in_b) — matched rows between two vertex arrays."""
    shared_a, shared_b = [], []
    for i, va in enumerate(verts_a):
        for j, vb in enumerate(verts_b):
            if np.allclose(va, vb, atol=atol):
                shared_a.append(verts_a[i])
                shared_b.append(verts_b[j])
    return np.array(shared_a), np.array(shared_b)



@dataclass(frozen=True)
class Cell:
    poly_vertices: NDArray[np.float64]  # shape (4, 4) original 4D positions
    net_vertices: NDArray[np.float64]   # shape (4, 4) placed positions (w=0)

    @property
    def net_vertices_3d(self) -> NDArray[np.float64]:
        return self.net_vertices[:, :3]

    def shared_with(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns poly vertices shared with another cell, shape (3, 4)."""
        shared, _ = find_shared_vertices(self.poly_vertices, other)
        return shared

    def free_from(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns poly vertices in other that are not in self, shape (1, 4)."""
        return np.array([
            v for v in other
            if not any(np.allclose(v, pv, atol=1e-10) for pv in self.poly_vertices)
        ])

    def net_positions_of(self, poly_verts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Given poly vertices, return their corresponding net positions."""
        _, shared_self = find_shared_vertices(poly_verts, self.poly_vertices)
        indices = [
            i for v in shared_self
            for i, pv in enumerate(self.poly_vertices)
            if np.allclose(v, pv, atol=1e-10)
        ]
        return self.net_vertices[indices]

    def assert_3D(self):
        np.testing.assert_allclose(self.net_vertices[:, 3], 0, atol=1e-10)

    def assert_placed_correct(self, other: Cell):
        for i, v in enumerate(self.poly_vertices):
            for j, pv in enumerate(other.poly_vertices):
                if np.allclose(v, pv, atol=1e-10):
                    np.testing.assert_allclose(
                        self.net_vertices[i], other.net_vertices[j], atol=1e-10
                    )
                    break


def compute_first_cell(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> Cell:
    R = rotation_matrix(normal, w)
    points_trans = points - points[0]
    net_points = points_trans @ R.T
    return Cell(poly_vertices=points, net_vertices=net_points)


def compute_new_cell(
    prev_cell: Cell,
    cur_points: NDArray[np.float64],
    cur_normal: NDArray[np.float64],
) -> Cell:
    shared_poly = prev_cell.shared_with(cur_points)
    shared_net = prev_cell.net_positions_of(shared_poly)

    shared_poly_centroid = shared_poly.mean(axis=0)
    shared_net_centroid = shared_net.mean(axis=0)

    shared_poly_centered = shared_poly - shared_poly_centroid
    shared_net_centered = shared_net - shared_net_centroid

    src = np.vstack([shared_poly_centered, cur_normal])
    tgt = np.vstack([shared_net_centered, w])

    R = rotation_matrix(src, tgt)

    new_net_verts = (cur_points - shared_poly_centroid) @ R.T + shared_net_centroid
    new_cell = Cell(poly_vertices=cur_points, net_vertices=new_net_verts)

    new_cell.assert_3D()
    new_cell.assert_placed_correct(prev_cell)

    return new_cell

def _assert_cell(cell: Cell | None) -> Cell:
    assert cell is not None
    return cell

def rotation_matrix(src: NDArray[np.float64], tgt: NDArray[np.float64]) -> NDArray[np.float64]:
    src = np.atleast_2d(src)
    tgt = np.atleast_2d(tgt)
    
    assert src.shape == tgt.shape, "src and tgt must have same shape"
    assert not np.allclose(src, 0), "src is zero"
    assert not np.allclose(tgt, 0), "tgt is zero"
    
    M = tgt.T @ src
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    return R

# def tet_faces(verts):
#     # all combinations of 3 vertices from 4 = the 4 triangular faces
#     return [verts[list(f)] for f in combinations(range(4), 3)]
#
# def plot_tets(*tets):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     colors = matplotlib.cm.tab10.colors  # 10 distinct colors
#
#     for verts, color in zip(tets, colors):
#         faces = tet_faces(verts)
#         poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='black')
#         ax.add_collection3d(poly)
#
#     all_verts = np.vstack(tets)
#     for setter, coord in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], all_verts.T):
#         setter(coord.min(), coord.max())
#
#     ax.set_aspect('equal')
#
#     plt.show()
