from polytope_core.polytope import rotation_matrix
from hypothesis.extra.numpy import arrays
from hypothesis import given, strategies as st
import numpy as np
from fp_error_consts import DEGENERATE_TOL, ERROR_ATOL


def valid_nd_array(n):
    return arrays(
        np.float64,
        (n, 4),
        elements=st.floats(
            min_value=-1, max_value=1, allow_nan=False, allow_infinity=False
        ),
    ).filter(
        lambda x: np.linalg.matrix_rank(x, tol=DEGENERATE_TOL) == n
        and not np.allclose(x, 0)
    )


@given(st.one_of(valid_nd_array(1), valid_nd_array(3), valid_nd_array(4)))
def test_aligns_src_to_tgt_property(src):
    """For any src, rotating it and passing both in should recover the rotation."""
    R_true, _ = np.linalg.qr(np.random.randn(4, 4))
    if np.linalg.det(R_true) < 0:
        R_true[:, -1] *= -1
    tgt = src @ R_true.T

    R = rotation_matrix(src, tgt)
    np.testing.assert_allclose(src @ R.T, tgt, atol=ERROR_ATOL)
