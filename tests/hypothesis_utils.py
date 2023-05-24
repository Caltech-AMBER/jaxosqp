import numpy as np
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
    lu = draw(
        arrays(
            np.float64,
            (n, 2),
            elements=st.floats(allow_nan=False, min_value=-1e10, max_value=1e10),
            fill=st.nothing(),
        )
    )
    ind_inverted = lu[:, 0] > lu[:, 1]
    lu[ind_inverted, :] = lu[ind_inverted, ::-1]

    l = lu[:, 0]
    u = lu[:, 1]
    return l, u


@st.composite
def qp_data(draw, *, n: int = 3, m: int = 3):
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
def qp_data_dims(
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
    data = draw(qp_data(n=n, m=m))
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
