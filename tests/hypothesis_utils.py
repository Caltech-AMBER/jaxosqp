import numpy as np
import scipy.sparse as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ######### #
# UTILITIES #
# ######### #


@st.composite
def unitary_matrices(draw, *, n: int = 3):
    """Strategy for generating a random unitary matrix.

    See: mathoverflow.net/questions/333187/random-unitary-matrices

    Parameters
    ----------
    draw
        A function that draws from a strategy.
    n : int, default=3
        Dimensions of the unitary matrix.
    """
    # drawing and orthonormalizing a matrix to return a unitary matrix
    _U = draw(
        arrays(
            np.float64,
            (n, n),
            elements=st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
            fill=st.nothing(),
        )
    )
    U, _ = np.linalg.qr(_U)
    return U


@st.composite
def psd_matrices(draw, *, n: int = 3):
    """Strategy for generating a PSD matrix.

    Uses
        P = U @ D @ U.T,
    where D is a diagonal matrix with non-negative diagonal entries and U is unitary.
    """
    # draws a unitary matrix and a diagonal matrix to construct a PSD matrix
    U = draw(unitary_matrices(n=n))
    _D = draw(
        arrays(
            np.float64,
            n,
            elements=st.floats(
                min_value=1e-6,
                max_value=1e6,  # help to prevent overflow
                allow_nan=False,
                allow_infinity=False,
            ),
            fill=st.nothing(),
        )
    )
    D = np.diag(_D)
    return U @ D @ U.T


@st.composite
def lower_upper_bounds(draw, *, n: int = 3):
    """Strategy for generating lower and upper bounds."""
    # drawing random bounds and making sure the lower value is in the first column
    # OSQP sets 1e30 to be inf
    lu = draw(
        arrays(
            np.float64,
            (n, 2),
            elements=st.floats(allow_nan=False, min_value=-1e30, max_value=1e30),
            fill=st.nothing(),
        )
    )
    ind_inverted = lu[:, 0] > lu[:, 1]
    lu[ind_inverted, :] = lu[ind_inverted, ::-1]

    # eps=1e-30 in the OSQP code
    l = lu[:, 0]
    u = lu[:, 1] + 1e-6
    return l, u


@st.composite
def qp_random1(draw, *, n: int = 3, m: int = 3):
    """Strategy for generating QP data.

    Parameters
    ----------
    n : int, default=3
        The number of decision variables.
    m : int, default=3
        The number of constraints.
    """
    P = draw(psd_matrices(n=n))
    q = draw(
        arrays(
            np.float64,
            n,
            elements=st.floats(
                min_value=-1e10,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            ),
            fill=st.nothing(),
        )
    )
    A = draw(
        arrays(
            np.float64,
            (m, n),
            elements=st.floats(
                min_value=-1e10,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            ),
            fill=st.nothing(),
        )
    )
    l, u = draw(lower_upper_bounds(n=m))
    return P, q, A, l, u


@st.composite
def qp_random1_dims(
    draw, *, n_range: tuple[int, int] = (1, 10), m_range: tuple[int, int] = (1, 10)
):
    """Strategy for generating QP data over a range of data dimensions.

    Parameters
    ----------
    n_range : tuple[int, int], default=(1, 10)
        The range of dimensions for the decision variable dimension.
    m_range : tuple[int, int], default=(1, 10)
        The range for the number of constraints.
    """
    n = draw(st.integers(min_value=n_range[0], max_value=n_range[1]))
    m = draw(st.integers(min_value=m_range[0], max_value=m_range[1]))
    data = draw(qp_random1(n=n, m=m))
    return data


def _qp_random2(n: int = 3, density=0.15):
    """Generates a random QP using the same procedure as the OSQP paper.

    https://github.com/osqp/osqp_benchmarks/blob/master/problem_classes/random_qp.py
    """
    m = int(10 * n)
    P = sp.random(n, n, density=density, data_rvs=np.random.randn, format="csc")
    P = (P @ P.T).tocsc() + 1e-02 * sp.eye(n)
    q = np.random.randn(n)
    _A = sp.random(m, n, density=density, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)  # fictitious solution
    delta = np.random.rand(m)  # to get inequality
    _u = _A @ v + delta
    _l = -1e30 * np.ones(m)  # u - np.random.rand(m)

    # TODO(ahl): get rid of this once pre-solving on A is implemented
    # checking for rows of all 0s and eliminating them
    ind_all_zeros = np.array(
        np.all(_A.todense() == 0.0, axis=1)
    ).flatten()  # inds of rows that are all 0s
    A = np.delete(_A.todense(), ind_all_zeros, axis=0)
    l = np.delete(_l, ind_all_zeros)
    u = np.delete(_u, ind_all_zeros)

    return P.todense(), q, A, l, u


@st.composite
def qp_random2_dims(draw, *, n_range: tuple[int, int] = (1, 10), density: float = 0.15):
    """Strategy for generating a random QP over some range of dimensions.

    Parameters
    ----------
    n_range : tuple[int, int], default=(1, 10)
        The range of dimensions for the decision variable dimension.
    density : float, default=0.15
        The fraction of elements that are nonzero in P and A.
    """
    n = draw(st.integers(min_value=n_range[0], max_value=n_range[1]))
    data = _qp_random2(n=n, density=density)
    return data


# ############# #
# UTILITY TESTS #
# ############# #


@given(unitary_matrices(n=5))
def test_unitary_strategy(U: np.ndarray):
    """Tests that the unitary matrix strategy returns valid unitary matrices."""
    assert np.allclose(U @ U.T, np.eye(5))
    assert np.allclose(U.T @ U, np.eye(5))


@given(psd_matrices(n=5))
def test_psd_strategy(P: np.ndarray):
    """Tests that the PSD strategy returns valid PSD matrices."""
    assert np.allclose(P, P.T)  # symmetry check
    assert np.all(np.linalg.eig(P)[0] >= 0.0)  # non-negative eigval check


@given(lower_upper_bounds(n=10))
def test_bounds(lu: tuple[np.ndarray, np.ndarray]):
    """Tests that the lower and upper bound strategy returns valid bounds."""
    l = lu[0]
    u = lu[1]
    assert np.all(l <= u)
