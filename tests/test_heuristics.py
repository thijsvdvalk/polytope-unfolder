import numpy as np
from numpy.testing import assert_allclose
from polytope_core.polytope import (
    compute_aspect_ratio,
    compute_centroids_distance,
    compute_dihedral_angle,
    compute_shared_face_area,
    compute_volume,
)


class TestDihedralAngle:
    def test_parallel_normals(self):
        # same direction -> angle = 0
        n1 = np.array([1, 0, 0, 0], dtype=float)
        n2 = np.array([1, 0, 0, 0], dtype=float)
        assert_allclose(compute_dihedral_angle(n1, n2), 0.0, atol=1e-10)

    def test_antiparallel_normals(self):
        # opposite direction -> angle = pi
        n1 = np.array([1, 0, 0, 0], dtype=float)
        n2 = np.array([-1, 0, 0, 0], dtype=float)
        assert_allclose(compute_dihedral_angle(n1, n2), np.pi, atol=1e-10)

    def test_perpendicular_normals(self):
        # orthogonal -> angle = pi/2
        n1 = np.array([1, 0, 0, 0], dtype=float)
        n2 = np.array([0, 1, 0, 0], dtype=float)
        assert_allclose(compute_dihedral_angle(n1, n2), np.pi / 2, atol=1e-10)

    def test_unnormalized_normals(self):
        # should still work with unnormalized vectors
        n1 = np.array([2, 0, 0, 0], dtype=float)
        n2 = np.array([0, 5, 0, 0], dtype=float)
        assert_allclose(compute_dihedral_angle(n1, n2), np.pi / 2, atol=1e-10)

    def test_numerical_stability(self):
        # slightly over 1 due to floating point -> should not return nan
        n1 = np.array([1, 0, 0, 0], dtype=float)
        n2 = np.array([1, 1e-17, 0, 0], dtype=float)
        result = compute_dihedral_angle(n1, n2)
        assert not np.isnan(result)


class TestSharedFaceArea:
    def test_unit_right_triangle(self):
        # right triangle with legs of length 1 -> area = 0.5
        triangle = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_shared_face_area(triangle), 0.5, atol=1e-10)

    def test_equilateral_triangle(self):
        # equilateral triangle with side length 1 -> area = sqrt(3)/4
        triangle = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0.5, np.sqrt(3) / 2, 0, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_shared_face_area(triangle), np.sqrt(3) / 4, atol=1e-10)

    def test_degenerate_triangle(self):
        # collinear points -> area = 0
        triangle = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 0, 0, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_shared_face_area(triangle), 0.0, atol=1e-10)

    def test_triangle_in_4d(self):
        # triangle living in 4D space, not just a subspace
        triangle = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        assert_allclose(compute_shared_face_area(triangle), 0.5, atol=1e-10)


class TestCentroidsDistance:
    def test_same_tet(self):
        tet = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float
        )
        assert_allclose(compute_centroids_distance(tet, tet), 0.0, atol=1e-10)

    def test_known_distance(self):
        tet1 = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float
        )
        # shift tet1 by [1,0,0,0] -> centroids differ by 1
        tet2 = tet1 + np.array([1, 0, 0, 0])
        assert_allclose(compute_centroids_distance(tet1, tet2), 1.0, atol=1e-10)

    def test_symmetry(self):
        tet1 = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float
        )
        tet2 = np.array(
            [[2, 0, 0, 0], [3, 0, 0, 0], [2, 1, 0, 0], [2, 0, 1, 0]], dtype=float
        )
        assert_allclose(
            compute_centroids_distance(tet1, tet2),
            compute_centroids_distance(tet2, tet1),
            atol=1e-10,
        )


class TestVolume:
    def test_regular_tet(self):
        # standard 3D tet embedded in 4D, volume = 1/6
        verts = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_volume(verts), 1 / 6, atol=1e-10)

    def test_degenerate_tet(self):
        # coplanar points -> volume = 0
        verts = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0.5, 0.5, 0, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_volume(verts), 0.0, atol=1e-10)

    def test_scaled_tet(self):
        # scaling by k scales volume by k^3
        verts = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        k = 2.0
        assert_allclose(
            compute_volume(verts * k), compute_volume(verts) * k**3, atol=1e-10
        )


class TestAspectRatio:
    def test_regular_tet(self):
        # all edges equal -> aspect ratio = 1
        verts = np.array(
            [
                [1, 1, 1, 0],
                [1, -1, -1, 0],
                [-1, 1, -1, 0],
                [-1, -1, 1, 0],
            ],
            dtype=float,
        )
        assert_allclose(compute_aspect_ratio(verts), 1.0, atol=1e-10)

    def test_aspect_ratio_gte_1(self):
        # aspect ratio always >= 1
        verts = np.random.rand(4, 4)
        assert compute_aspect_ratio(verts) >= 1.0

    def test_flat_tet(self):
        # very flat tet -> high aspect ratio
        verts = np.array(
            [
                [0, 0, 0, 0],
                [100, 0, 0, 0],
                [0, 0.01, 0, 0],
                [0, 0, 0.01, 0],
            ],
            dtype=float,
        )
        assert compute_aspect_ratio(verts) > 10
