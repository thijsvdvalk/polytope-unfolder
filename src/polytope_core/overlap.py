import numpy as np
from numpy.typing import NDArray

"""
Seperating Axis Theorem.
"""


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
    )  # Have the equals such that touching is allowed


def overlaps(tet_a: NDArray[np.float64], tet_b: NDArray[np.float64]) -> bool:
    """Check if two tetrahedra overlap using SAT."""

    # get face normals
    def face_normals(verts):
        normals = []
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        for i, j, k in faces:
            e1 = verts[j] - verts[i]
            e2 = verts[k] - verts[i]
            normals.append(np.cross(e1, e2))
        return normals

    # get edges
    def edges(verts):
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        return [verts[j] - verts[i] for i, j in pairs]

    axes = []
    axes += face_normals(tet_a)
    axes += face_normals(tet_b)

    # edge cross products
    for e1 in edges(tet_a):
        for e2 in edges(tet_b):
            axes.append(np.cross(e1, e2))

    for axis in axes:
        if np.allclose(axis, 0):
            continue  # skip degenerate axes
        if separates(axis, tet_a, tet_b):
            return False  # found a separating axis → no overlap

    return True  # no separating axis found → overlap
