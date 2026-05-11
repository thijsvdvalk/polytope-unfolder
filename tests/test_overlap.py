import numpy as np
from numpy.typing import NDArray
from itertools import combinations
from hypothesis.extra.numpy import arrays
from hypothesis import given, assume, strategies as st
from polytope_core.overlap import overlaps

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def diameter(tet: NDArray[np.float64]) -> np.float64:
    return np.float64(max(np.linalg.norm(tet[i] - tet[j]) for i, j in combinations(range(4), 2)))

def random_tet():
    return arrays(
        np.float64, (4,3),
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)
    ).filter(lambda x: np.linalg.matrix_rank(x - x[0], tol=1e-6) == 3) # almost flat tets are filtered out

def random_direction():
    return arrays(
        np.float64, (3,), 
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)
    ).filter(lambda x: np.linalg.norm(x) > 1e-6)

def three_random_points():
    return arrays(
        np.float64, (3, 3), 
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)
    )

@given(random_tet(), random_direction())
def test_not_overlapping_by_big_translation(tet_1, direction):
    direction = direction / np.linalg.norm(direction)  # normalize
    tet_2 = tet_1 + direction * (diameter(tet_1) + 0.1)

    assert not overlaps(tet_1, tet_2)

@given(random_tet(), three_random_points())
def test_overlapping_by_point_other_tet_at_centroid(tet_1, three_points):
    centroid = tet_1.mean(axis=0)
    tet_2 = np.vstack([centroid, three_points]) 

    # Make sure that it spans 3d with the centroid, tol is such that almost flat tets are filtered out
    assume(np.linalg.matrix_rank(tet_2 - tet_2[0], tol=1e-6) == 3)

    assert overlaps(tet_1, tet_2)


@given(random_tet(), arrays(np.float64, (3,), elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False)))
def test_shared_face_no_overlap(tet_1, free_2):
    shared = tet_1[:3]
    free_1 = tet_1[3]

    v0, v1, v2 = shared
    normal = np.cross(v1 - v0, v2 - v0)

    assume(np.linalg.norm(normal) > 1e-6)  # shared face must not be degenerate

    # free points must be on opposite sides of the shared face
    side_1 = np.dot(free_1 - v0, normal)
    side_2 = np.dot(free_2 - v0, normal)
    assume(side_1 * side_2 < 0)

    # free_2 must not be coplanar with shared face
    assume(abs(side_2) > 1e-6)

    tet_2 = np.vstack([shared, free_2])
    assert not overlaps(tet_1, tet_2)

# @given(random_tet())
# def test_shared_edge_no_overlap(tet_1):
#     shared = tet_1[:2]
#     free_1a, free_1b = tet_1[2], tet_1[3]
#
#     edge = shared[1] - shared[0]
#     assume(np.linalg.norm(edge) > 1e-6)
#
#     perp = np.array([1, 0, 0]) if abs(edge[0]) < 0.9 else np.array([0, 1, 0])
#     normal = np.cross(edge, perp)
#     normal = normal / np.linalg.norm(normal)
#
#     assume(abs(np.dot(free_1a - shared[0], normal)) > 1e-6)
#     assume(abs(np.dot(free_1b - shared[0], normal)) > 1e-6)
#
#     # reflect free points through the plane to get opposite side
#     def reflect(p):
#         d = np.dot(p - shared[0], normal)
#         return p - 2 * d * normal
#
#     free_2a = reflect(free_1a)
#     free_2b = reflect(free_1b)
#
#     tet_2 = np.vstack([shared, free_2a, free_2b])
#     assume(np.linalg.matrix_rank(tet_2 - tet_2[0], tol=1e-6) == 3)
#
#     assert not overlaps(tet_1, tet_2)

def construct_tet(tet_1, free_2):
    shared = tet_1[:3]
    free_1 = tet_1[3]

    v0, v1, v2 = shared
    normal = np.cross(v1 - v0, v2 - v0)

    # assume(np.linalg.norm(normal) > 1e-6)  # shared face must not be degenerate
    assert np.linalg.norm(normal) > 1e-6  # shared face must not be degenerate

    # free points must be on opposite sides of the shared face
    side_1 = np.dot(free_1 - v0, normal)
    side_2 = np.dot(free_2 - v0, normal)
    assert side_1 * side_2 < 0
    # assume(side_1 * side_2 < 0)

    # free_2 must not be coplanar with shared face
    # assume(abs(side_2) > 1e-6)
    assert abs(side_2) > 1e-6

    tet_2 = np.vstack([shared, free_2])
    return tet_2



