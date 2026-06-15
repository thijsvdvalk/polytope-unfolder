import numpy as np
from numpy.typing import NDArray
from itertools import combinations
from hypothesis.extra.numpy import arrays
from hypothesis import given, assume, strategies as st
from fp_error_consts import DEGENERATE_TOL
from polytope_core.polytope import Cell

def diameter(tet: NDArray[np.float64]) -> np.float64:
    return np.float64(
        max(np.linalg.norm(tet[i] - tet[j]) for i, j in combinations(range(4), 2))
    )


def random_tet():
    return arrays(
        np.float64,
        (4, 3),
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    ).filter(
        lambda x: np.linalg.matrix_rank(x - x[0], tol=DEGENERATE_TOL) == 3
    )  # almost flat tets are filtered out


def random_direction():
    return arrays(
        np.float64,
        (3,),
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    ).filter(lambda x: np.linalg.norm(x) > DEGENERATE_TOL)


def three_random_points():
    return arrays(
        np.float64,
        (3, 3),
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    )


@given(random_tet(), random_direction())
def test_not_overlapping_by_big_translation(tet_1, direction):
    """Translating a copy of the tet by more than its diameter should yield no overlap."""
    direction = direction / np.linalg.norm(direction)
    tet_2 = tet_1 + direction * (diameter(tet_1) + 0.1)
    cell_1 = Cell(np.zeros(4), tet_1)
    cell_2 = Cell(np.zeros(4), tet_2)

    assert not cell_2.overlaps_with(cell_1)
    assert not cell_1.overlaps_with(cell_2)

@given(random_tet(), three_random_points())
def test_overlapping_by_point_other_tet_at_centroid(tet_1, three_points):
    """Having one vertice of tet_2 at the center of tet_1 should yield overlap."""
    centroid = tet_1.mean(axis=0)
    tet_2 = np.vstack([centroid, three_points])
    assume(np.linalg.matrix_rank(tet_2 - tet_2[0], tol=DEGENERATE_TOL) == 3)
    cell_1 = Cell(np.zeros((4, 4)), np.hstack([tet_1, np.zeros((4, 1))]))
    cell_2 = Cell(np.zeros((4, 4)), np.hstack([tet_2, np.zeros((4, 1))]))

    assert cell_1.overlaps_with(cell_2)
    assert cell_2.overlaps_with(cell_1)

@given(
    random_tet(),
    arrays(
        np.float64,
        (3,),
        elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    ),
)
def test_shared_face_no_overlap(tet_1, free_2):
    """Two tets that share a face and have their free points on opposite sides of the face should yield no overlap."""
    shared = tet_1[:3]
    free_1 = tet_1[3]
    v0, v1, v2 = shared
    normal = np.cross(v1 - v0, v2 - v0)
    assume(np.linalg.norm(normal) > DEGENERATE_TOL)
    side_1 = np.dot(free_1 - v0, normal)
    side_2 = np.dot(free_2 - v0, normal)
    assume(side_1 * side_2 < 0)
    assume(abs(side_2) > DEGENERATE_TOL)
    tet_2 = np.vstack([shared, free_2])
    cell_1 = Cell(np.zeros((4, 4)), np.hstack([tet_1, np.zeros((4, 1))]))
    cell_2 = Cell(np.zeros((4, 4)), np.hstack([tet_2, np.zeros((4, 1))]))

    assert not cell_1.overlaps_with(cell_2)
    assert not cell_2.overlaps_with(cell_1)

# @given(random_tet(), three_random_points())
# def test_overlapping_by_point_other_tet_at_centroid(tet_1, three_points):
#     """Having one vertice of tet_2 at the center of tet_1 should yield overlap."""
#     centroid = tet_1.mean(axis=0)
#     tet_2 = np.vstack([centroid, three_points])
#
#     assume(np.linalg.matrix_rank(tet_2 - tet_2[0], tol=DEGENERATE_TOL) == 3)
#
#     assert overlaps(tet_1, tet_2)
#
#
# @given(
#     random_tet(),
#     arrays(
#         np.float64,
#         (3,),
#         elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
#     ),
# )
# def test_shared_face_no_overlap(tet_1, free_2):
#     """Two tets that share a face and have their free points on opposite sides of the face should yield no overlap."""
#     shared = tet_1[:3]
#     free_1 = tet_1[3]
#
#     v0, v1, v2 = shared
#     normal = np.cross(v1 - v0, v2 - v0)
#
#     assume(
#         np.linalg.norm(normal) > DEGENERATE_TOL
#     )  # shared face must not be degenerate
#
#     # free points must be on opposite sides of the shared face
#     side_1 = np.dot(free_1 - v0, normal)
#     side_2 = np.dot(free_2 - v0, normal)
#     assume(side_1 * side_2 < 0)
#
#     # free_2 must not be coplanar with shared face
#     assume(abs(side_2) > DEGENERATE_TOL)
#
#     tet_2 = np.vstack([shared, free_2])
#     assert not overlaps(tet_1, tet_2)
