from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from dataclasses import dataclass
from itertools import combinations
from .overlap import overlaps


W = np.array([0, 0, 0, 1])
SNAP_TOL = 1e-8


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
        nx.freeze(self.neigh_graph)

    def unfold_from_spanning_tree(self, spanning_tree: nx.Graph) -> Net:
        traversal = traversal_from_spanning_tree(spanning_tree)
        return self._unfold(traversal)

    def _unfold(self, traversal: list[tuple[int, int]]) -> Net:
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

        cells_not_none = [_assert_cell(cell) for cell in cells]
        return Net(self, traversal, cells_not_none)


@dataclass(frozen=True)
class Net:
    polytope: Polytope
    traversal: list[tuple[int, int]]
    cells: list[Cell]

    def overlaps(self) -> bool:
        traversal_set = set(self.traversal)
        n = len(self.cells)
        to_check = [
            x for x in list(combinations(range(n), 2)) if x not in traversal_set
        ]

        return any(
            overlaps(self[i].net_vertices_3d, self[j].net_vertices_3d)
            for i, j in to_check
        )

    def __getitem__(self, idx: int) -> Cell:
        return self.cells[idx]


def find_shared_vertices(
    verts_a: NDArray[np.float64],
    verts_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns (shared_in_a, shared_in_b) — matched rows between two vertex arrays."""
    shared_a, shared_b = [], []
    for i, va in enumerate(verts_a):
        for j, vb in enumerate(verts_b):
            if np.all(va == vb):
                shared_a.append(verts_a[i])
                shared_b.append(verts_b[j])
    return np.array(shared_a), np.array(shared_b)


@dataclass(frozen=True)
class Cell:
    poly_vertices: NDArray[np.float64]  # shape (4, 4) original 4D positions
    net_vertices: NDArray[np.float64]  # shape (4, 4) placed positions (w=0)

    @property
    def net_vertices_3d(self) -> NDArray[np.float64]:
        return self.net_vertices[:, :3]

    def shared_with(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns poly vertices shared with another cell, shape (3, 4)."""
        shared, _ = find_shared_vertices(self.poly_vertices, other)
        return shared

    def net_positions_of(self, poly_verts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Given poly vertices, return their corresponding net positions."""
        indices = [
            i
            for v in poly_verts
            for i, pv in enumerate(self.poly_vertices)
            if np.all(v == pv)
        ]
        return self.net_vertices[indices]

    def assert_3D_and_snap(self):
        np.testing.assert_allclose(self.net_vertices[:, 3], 0, atol=SNAP_TOL)
        self.net_vertices[:, 3] = 0

    def assert_placed_correct_and_snap(self, other: Cell):
        for i, v in enumerate(self.poly_vertices):
            for j, pv in enumerate(other.poly_vertices):
                if np.allclose(v, pv, atol=SNAP_TOL):
                    np.testing.assert_allclose(
                        self.net_vertices[i], other.net_vertices[j], atol=SNAP_TOL
                    )
                    self.net_vertices[i] = other.net_vertices[j]  # snap
                    break

    def freeze(self):
        self.poly_vertices.flags.writeable = False
        self.net_vertices.flags.writeable = False


def compute_first_cell(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> Cell:
    R = rotation_matrix(normal, W)
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
    tgt = np.vstack([shared_net_centered, W])

    R = rotation_matrix(src, tgt)

    new_net_verts = (cur_points - shared_poly_centroid) @ R.T + shared_net_centroid
    new_cell = Cell(poly_vertices=cur_points, net_vertices=new_net_verts)

    new_cell.assert_3D_and_snap()
    new_cell.assert_placed_correct_and_snap(prev_cell)
    new_cell.freeze()

    return new_cell


def _assert_cell(cell: Cell | None) -> Cell:
    assert cell is not None
    return cell


def traversal_from_spanning_tree(spanning_tree: nx.Graph) -> list[tuple[int, int]]:
    source = list(spanning_tree.nodes)[0]
    return list(nx.bfs_edges(spanning_tree, source=source))


def rotation_matrix(
    src: NDArray[np.float64], tgt: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Finds rotation matrix that best aligns src onto tgt."""
    src = np.atleast_2d(src)
    tgt = np.atleast_2d(tgt)

    assert src.shape == tgt.shape, "src and tgt must have same shape"
    assert not np.allclose(src, 0), "src is zero"
    assert not np.allclose(tgt, 0), "tgt is zero"

    M = tgt.T @ src
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    # If it is a reflection, convert it to rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Assert matrix is actually rotation matrix
    np.testing.assert_allclose(np.linalg.det(R), 1, atol=SNAP_TOL)

    return R
