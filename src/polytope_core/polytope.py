from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from dataclasses import dataclass
import bisect

W = np.array([0, 0, 0, 1])
SNAP_TOL = 1e-8

@dataclass
class Polytope:
    points: NDArray[np.float64]
    simplices: NDArray[np.integer]
    normals: NDArray[np.float64]
    neigh_graph: nx.Graph
    edge_weights_initialized: bool = False 
    node_weights_initialized: bool = False 

    def __post_init__(self):
        self.points.flags.writeable = False
        self.simplices.flags.writeable = False
        self.normals.flags.writeable = False
        nx.freeze(self.neigh_graph)

    def overlap_free_unfolding(self, spanning_tree: nx.Graph) -> bool:
        traversal = traversal_from_spanning_tree(spanning_tree)
        return self._overlap_free_unfolding_from_traversal(traversal)


    def _overlap_free_unfolding_from_traversal(self, traversal: list[tuple[int, int]]) -> bool:
        cells: list[Cell | None] = [None] * len(self.simplices)
        first = traversal[0][0]

        cells_placed = set()
        cells_sorted = []  # list of (bbox_min_x, cell_index), kept sorted

        cells[first] = compute_first_cell(
            self.points[self.simplices[first]],
            self.normals[first],
        )

        cells_placed.add(first)
        cells_sorted.append((cells[first]._bbox_min[0], first))

        for prev, cur in traversal:
            new_cell = compute_new_cell(
                _assert_cell(cells[prev]),
                self.points[self.simplices[cur]],
                self.normals[cur],
            )

            cells[cur] = new_cell

            new_min_x = new_cell._bbox_min[0]
            new_max_x = new_cell._bbox_max[0]

            # Insert into sorted list
            bisect.insort(cells_sorted, (new_min_x, cur))
            # Now need to check with all the already placed cells. 
        
                    # Sweep: only check cells whose bbox_min_x <= new_max_x
            for min_x, cell_idx in cells_sorted:
                if min_x > new_max_x:
                    break
                if cell_idx == cur or cell_idx == prev:
                    continue
                if new_cell.overlaps_with(cells[cell_idx]):
                    return False

            # for cell in cells_placed:
            #     if cell != prev and new_cell.overlaps_with(cells[cell]):
            #         return False

            cells_placed.add(cur)

        return True

    def init_edge_weigths(self):
        g = self.neigh_graph
        for u,v in g.edges():
            g[u][v]['dihedral_angle'] = compute_dihedral_angle(self.normals[u], self.normals[v])
            shared_simp = list(set(self.simplices[u]) & set(self.simplices[v]))
            g[u][v]['shared_face_area'] = compute_shared_face_area(self.points[shared_simp])
            g[u][v]['centroids_distance'] = compute_centroids_distance(self.points[self.simplices[u]], self.points[self.simplices[v]])
        
        # Normalize
        for prop in ['dihedral_angle', 'shared_face_area', 'centroids_distance']:
            values = np.array([g[u][v][prop] for u, v in g.edges()])
            min, max = values.min(), values.max()
            for u, v in g.edges():
                g[u][v][prop] = (g[u][v][prop] - min) / (max - min) if max != min else 0.0
        
        self.edge_weights_initialized = True

    def init_node_weigths(self):
        g = self.neigh_graph
        for node, data in g.nodes(data=True):
            data['volume'] = compute_volume(self.points[self.simplices[node]])
            data['aspect_ratio'] = compute_aspect_ratio(self.points[self.simplices[node]])
        
        for prop in ['volume', 'aspect_ratio']:
            values = np.array([g.nodes[n][prop] for n in g.nodes()])
            min, max = values.min(), values.max()
            for n in g.nodes():
                g.nodes[n][prop] = (g.nodes[n][prop] - min) / (max - min) if max != min else 0.0


        self.node_weights_initialized = True

@dataclass(frozen=True)
class Net:
    polytope: Polytope
    traversal: list[tuple[int, int]]
    cells: list[Cell]



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


@dataclass()
class Cell:
    poly_vertices: NDArray[np.float64]  # shape (4, 4) original 4D positions
    net_vertices: NDArray[np.float64]  # shape (4, 3) placed positions (w=0)
    

    def __post_init__(self):
        verts = self.net_vertices[:, :3]
        self._net_vertices_3d = verts
        self._face_normals = face_normals(verts)
        self._edges = edges(verts)
        self._bbox_min = verts.min(axis=0)
        self._bbox_max = verts.max(axis=0)
        

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

    def overlaps_with(self, other: Cell) -> bool:
        if not self._bboxes_overlap(other):
            return False

        verts_a = self._net_vertices_3d
        verts_b = other._net_vertices_3d

        # Face normals first — most likely to separate, cheap early exit
        for axis in self._face_normals:
            if separates(axis, verts_a, verts_b):
                return False

        for axis in other._face_normals:
            if separates(axis, verts_a, verts_b):
                return False

        # All 36 edge cross products at once
        ea = self._edges[:, np.newaxis, :]  # (6, 1, 3)
        eb = other._edges[np.newaxis, :, :]  # (1, 6, 3)
        axes = np.cross(ea, eb).reshape(-1, 3)  # (36, 3)

        # Filter degenerate axes
        axes = np.array([a for a in axes if not np.allclose(a, 0)])

        # Project all axes at once
        dots_a = verts_a @ axes.T  # (4, n_axes)
        dots_b = verts_b @ axes.T  # (4, n_axes)

        TOL = 1000 * np.finfo(np.float64).eps

        sep = ((dots_a.max(axis=0) <= dots_b.min(axis=0) + TOL) |
               (dots_b.max(axis=0) <= dots_a.min(axis=0) + TOL))

        return not sep.any()

    def _bboxes_overlap(self, other: Cell) -> np.bool:
        return np.all(self._bbox_max >= other._bbox_min) and np.all(other._bbox_max >= self._bbox_min)


def compute_first_cell(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> Cell:
    R = rotation_matrix(normal, W)
    points_trans = points - points[0] # we translate, because we need the hyperplane to include the origin to make the rotation valid.
    net_points = points_trans @ R.T
    assert_3D_and_snap(net_points)
    cell = Cell(poly_vertices=points, net_vertices=net_points)
    return cell


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

    new_net_verts = (cur_points - shared_poly_centroid) @ R.T + shared_net_centroid # again we translate to the origin to make the rotation valid.

    assert_3D_and_snap(new_net_verts)
    assert_placed_correct_and_snap(cur_points, new_net_verts, prev_cell)
    return Cell(cur_points, new_net_verts)


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



def assert_3D_and_snap(net_verts):
    np.testing.assert_allclose(net_verts[:, 3], 0, atol=SNAP_TOL)
    net_verts[:, 3] = 0

def assert_placed_correct_and_snap(poly_vertices, net_vertices, other: Cell):
    for i, v in enumerate(poly_vertices):
        for j, pv in enumerate(other.poly_vertices):
            if np.allclose(v, pv, atol=SNAP_TOL):
                np.testing.assert_allclose(
                    net_vertices[i], other.net_vertices[j], atol=SNAP_TOL
                )
                net_vertices[i] = other.net_vertices[j]  # snap
                break


def project(vertices, axis):
    """Project all vertices onto axis, return [min, max]."""
    dots = vertices @ axis
    return dots.min(), dots.max()


def separates(axis, verts_a, verts_b):
    tol = 1000 * np.finfo(np.float64).eps
    min_a, max_a = project(verts_a, axis)
    min_b, max_b = project(verts_b, axis)

    return (
        max_a <= min_b + tol or max_b <= min_a + tol
    )  

def face_normals(verts: NDArray) -> NDArray:
    # face vertex indices
    i = np.array([0, 0, 0, 1])
    j = np.array([1, 1, 2, 2])
    k = np.array([2, 3, 3, 3])
    e1 = verts[j] - verts[i]  # (4, 3)
    e2 = verts[k] - verts[i]  # (4, 3)
    return np.cross(e1, e2)    # (4, 3)

def edges(verts: NDArray) -> NDArray:
    i = np.array([0, 0, 0, 1, 1, 2])
    j = np.array([1, 2, 3, 2, 3, 3])
    return verts[j] - verts[i]  # (6, 3)


def compute_dihedral_angle(n1, n2):
    cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def compute_shared_face_area(triangle: NDArray[np.float64]):
    # triangle: (3, 4) array, 3 vertices each with 4 coordinates
    a, b, c = triangle
    ab = b - a
    ac = c - a
    # Area = 0.5 * sqrt(|ab|^2 * |ac|^2 - (ab . ac)^2)
    # This is from the identity |u x v|^2 = |u|^2|v|^2 - (u.v)^2
    area = 0.5 * np.sqrt(np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac)**2)
    return area

def compute_centroids_distance(tet1: NDArray[np.float64], tet2: NDArray[np.float64]) -> np.float64:
    centroid1 = tet1.mean(axis=0)
    centroid2 = tet2.mean(axis=0)
    return np.float64(np.linalg.norm(centroid1 - centroid2))

def compute_volume(verts: NDArray[np.float64]):
    a, b, c, d = verts
    # 3 edge vectors, each of length 4
    mat = np.array([a - d, b - d, c - d])  # shape (3, 4)
    # Gram matrix: dot products of all edge vector pairs
    gram = mat @ mat.T                      # shape (3, 3)
    return np.sqrt(abs(np.linalg.det(gram))) / 6


def compute_aspect_ratio(verts: NDArray[np.float64]):
    # tet: (4, 4) array of 4 vertices in 4D
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(np.linalg.norm(verts[i] - verts[j]))
    return max(edges) / min(edges)
