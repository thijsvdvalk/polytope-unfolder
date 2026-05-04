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

class Polytope:
    def __init__(self, 
                 points: NDArray[np.float64], 
                 simplices: NDArray[np.integer], 
                 normals: NDArray[np.float64], 
                 neighbors: NDArray[np.integer],
                 ) -> None:
        # gotta store the facets, the store neighbour graph
        
        self.points = points
        self.simplices = simplices
        self.neighbors = neighbors
        self.normals = normals
        
        if self.neighbors is not None:
            self.neigh_graph = nx.Graph()
            for node, neighs in enumerate(self.neighbors):
                self.neigh_graph.add_node(int(node))
                for neigh in neighs:
                    self.neigh_graph.add_edge(int(node), int(neigh))


    def serialize(self, filepath: str) -> None:
        data = {
            "points": self.points.tolist(),
            "simplices": self.simplices.tolist(),
            "normals": self.normals.tolist(),
            "neighbors": self.neighbors.tolist() if self.neighbors is not None else None
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


    @classmethod
    def deserialize(cls, filepath: str) -> "Polytope":
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        points = np.array(data["points"], dtype=np.float64)
        simplices = np.array(data["simplices"], dtype=np.intp)
        normals = np.array(data["normals"], dtype=np.float64)
        neighbors = np.array(data["neighbors"], dtype=np.intp)
        
        return cls(points, simplices, normals, neighbors)


    # def unfold(self, traversal: list[tuple[int, int]]) -> Net:
    #     net = Net(self)
    #     net.place_first_cell(traversal[0][0])
    #
    #     for prev, cur in traversal:
    #         net.place_new_cell(prev, cur)
            
            # prev_cell = net.cells[prev]
            # cur_points = self.points[self.simplices[cur]]
            #
            # assert prev_cell is not None
            #
            # # calculate the free point of cur
            # shared_poly = prev_cell.shared_with(cur_points)
            # free_poly = prev_cell.free_from(cur_points)
            # shared_net  = prev_cell.net_positions_of(shared_poly)
            # print(f"{shared_poly=}")
            # print(f"{free_poly=}")
            # print(f"{shared_net=}")
            # shared_poly_centroid = shared_poly.mean(axis=0)
            # shared_net_centroid = shared_net.mean(axis=0)
            #
            # shared_poly_centered = shared_poly - shared_poly_centroid
            # shared_net_centered  = shared_net  - shared_net_centroid
            #
            # cur_normal = self.normals[cur][:4]
            # src = np.vstack([shared_poly_centered, cur_normal])
            # tgt = np.vstack([shared_net_centered,  w])
            #
            # R = rotation_matrix(src, tgt)
            #
            # new_net_verts = (cur_points - shared_poly_centroid) @ R.T + shared_net_centroid
            # new_cell = Cell(poly_vertices=cur_points, net_vertices=new_net_verts)
            # np.testing.assert_allclose(new_cell.net_vertices[:, 3], 0, atol=1e-10)
            #
            # for i, v in enumerate(new_cell.poly_vertices):
            #     for j, pv in enumerate(prev_cell.poly_vertices):
            #         if np.allclose(v, pv, atol=1e-10):
            #             new_cell.net_vertices[i] = prev_cell.net_vertices[j]
            #             break
            #
            # net.cells[cur] = new_cell
            #
        # return net

@dataclass
class Net:
    polytope: Polytope
    traversal: list[tuple[int, int]]

    def __post_init__(self):
        self.cells: list[Cell | None] = [None] * len(self.polytope.simplices)

        self.place_first_cell(self.traversal[0][0])
        
        for prev, cur in self.traversal:
            self.place_new_cell(prev, cur)


    def place_first_cell(self, node_index: int):
        points = self.polytope.points[self.polytope.simplices[node_index]]
        normal = self.polytope.normals[node_index][:4]
        R = rotation_matrix(normal, w)
        points_trans = points - points[0]
        net_points = (points_trans) @ R.T
        self.cells[node_index] = Cell(poly_vertices=points, net_vertices=net_points)
   

    def place_new_cell(self, prev_idx: int, cur_idx: int):
        prev_cell = self.get_cell(prev_idx)

        cur_points = self.polytope.points[self.polytope.simplices[cur_idx]]
        
        # calculate the free point of cur
        shared_poly = prev_cell.shared_with(cur_points)
        free_poly = prev_cell.free_from(cur_points)
        shared_net  = prev_cell.net_positions_of(shared_poly)
        
        shared_poly_centroid = shared_poly.mean(axis=0)
        shared_net_centroid = shared_net.mean(axis=0)

        shared_poly_centered = shared_poly - shared_poly_centroid
        shared_net_centered  = shared_net  - shared_net_centroid
    
        cur_normal = self.polytope.normals[cur_idx][:4]
        src = np.vstack([shared_poly_centered, cur_normal])
        tgt = np.vstack([shared_net_centered,  w])

        R = rotation_matrix(src, tgt)

        new_net_verts = (cur_points - shared_poly_centroid) @ R.T + shared_net_centroid
        new_cell = Cell(poly_vertices=cur_points, net_vertices=new_net_verts)

        new_cell.assert_3D()
        new_cell.assert_placed_correct(prev_cell)

        # np.testing.assert_allclose(new_cell.net_vertices[:, 3], 0, atol=1e-10)
        
        # for i, v in enumerate(new_cell.poly_vertices):
        #     for j, pv in enumerate(prev_cell.poly_vertices):
        #         if np.allclose(v, pv, atol=1e-10):
        #             new_cell.net_vertices[i] = prev_cell.net_vertices[j]
        #             break
        
        self.cells[cur_idx] = new_cell


    def overlaps(self) -> bool:
        traversal_set = set(self.traversal)
        n = len(self.cells)
        to_check = [x for x in list(combinations(range(n), 2)) if x not in traversal_set]
        print(to_check)

        plot_tets(*[cell.net_vertices_3d for cell in self.cells]) 
        for i, j in to_check:
            if overlaps(self.get_cell(i).net_vertices_3d, self.get_cell(j).net_vertices_3d):
                return True
        
        for cell in self.cells:
            print(cell)

        return False

    def get_cell(self, idx: int) -> Cell:
        cell = self.cells[idx]
        assert cell is not None
        return cell

@dataclass
class Cell:
    poly_vertices: NDArray[np.float64]  # shape (4, 4) original 4D positions
    net_vertices: NDArray[np.float64]   # shape (4, 4) placed positions (w=0)

    @property
    def net_vertices_3d(self):
        return self.net_vertices[:, :3] 

    def shared_with(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns poly vertices shared with another cell, shape (3, 4)."""
        return np.array([
            v for v in other
            if any(np.allclose(v, pv, atol=1e-10) for pv in self.poly_vertices)
        ])

    def free_from(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns poly vertices in other that are not in self, shape (1, 4)."""
        return np.array([
            v for v in other
            if not any(np.allclose(v, pv, atol=1e-10) for pv in self.poly_vertices)
        ])

    def net_positions_of(self, poly_verts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Given poly vertices, return their corresponding net positions."""
        return np.array([
            self.net_vertices[i]
            for v in poly_verts
            for i, pv in enumerate(self.poly_vertices)
            if np.allclose(v, pv, atol=1e-10)
        ])

    def assert_3D(self):
        np.testing.assert_allclose(self.net_vertices[:, 3], 0, atol=1e-10)

    def assert_placed_correct(self, other: Cell):
        for i, v in enumerate(self.poly_vertices):
            for j, pv in enumerate(other.poly_vertices):
                if np.allclose(v, pv, atol=1e-10):
                    np.testing.assert_allclose(self.net_vertices[i], other.net_vertices[j], atol=1e-10)
                    break



class PolytopeBuilder:
    @staticmethod
    def random(num_points = 20):
        points = np.random.randn(num_points, 4)
        hull = ConvexHull(points)

        points_on_hull = points[hull.vertices] 

        lookup = np.zeros(num_points + 1, dtype=int)
        lookup[hull.vertices] = np.arange(len(hull.vertices))
        print(lookup)
        new_simplices = lookup[hull.simplices]

        print(hull.neighbors)

        return Polytope(points_on_hull, new_simplices, hull.equations, hull.neighbors)

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
        return Polytope(vertices, hull.simplices, hull.equations, hull.neighbors)


def rotation_matrix(src: NDArray[np.float64], tgt: NDArray[np.float64]):
    src = np.atleast_2d(src)
    tgt = np.atleast_2d(tgt)
    
    assert src.shape == tgt.shape, "src and tgt must have same shape"
    
    # Check for degenerate input
    assert not np.allclose(src, 0), "src is zero"
    assert not np.allclose(tgt, 0), "tgt is zero"
    
    M = tgt.T @ src
    
    # Check rank - if low rank the result may be unreliable
    # rank = np.linalg.matrix_rank(M)
    # if rank < src.shape[1]:
    #     print(f"Warning: M is rank {rank} but shape {src.shape[1]}, result may be unreliable")
    
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    return R

def tet_faces(verts):
    # all combinations of 3 vertices from 4 = the 4 triangular faces
    return [verts[list(f)] for f in combinations(range(4), 3)]

def plot_tets(*tets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = matplotlib.cm.tab10.colors  # 10 distinct colors
    
    for verts, color in zip(tets, colors):
        faces = tet_faces(verts)
        poly = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='black')
        ax.add_collection3d(poly)
    
    all_verts = np.vstack(tets)
    for setter, coord in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], all_verts.T):
        setter(coord.min(), coord.max())

    ax.set_aspect('equal')
    
    plt.show()
