import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap
from hypothesis import given, strategies as st

from jaxosqp import osqp

jax.config.update("jax_enable_x64", True)

# ######### #
# UTILITIES #
# ######### #

@jit
def outer(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Outer product of jax arrays."""
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[-1] == b.shape[0]
    return a @ b.T

# ##### #
# TESTS #
# ##### #

@given(
    st.integers(min_value=1, max_value=10),  # batch size
    st.integers(min_value=1, max_value=10),  # problem dimension
    st.integers(min_value=1, max_value=10),  # number of constraints
)
def test_qp_convergence(B: int, n: int, m: int):
    """Tests the correctness of a batch of QPs.

    Parameters
    ----------
    B : int
        The batch size.
    n : int
        The dimension of the decision variable.
    m : int
        The number of constraints.
    """
    # batched utility
    outer_batched = vmap(outer)

    # random problem data
    key = random.PRNGKey(0)
    _P = random.normal(key, (B, n, n))
    P = outer_batched(_P, _P)  # (B, n, n)

    key, subkey = random.split(key)
    q = random.normal(subkey, (B, n))

    key, subkey = random.split(key)
    A = random.normal(subkey, (B, m, n))

    key, subkey = random.split(key)
    l = -random.uniform(subkey, (B, m))

    key, subkey = random.split(key)
    u = random.uniform(subkey, (B, m))

    # solving a batched jax OSQP problem instance
    prob_jax, data, state = osqp.OSQPProblem.from_data(P, q, A, l, u)
    state = vmap(prob_jax.solve)(data, state)
    opt_primal_jax = state.x
    opt_dual_jax = state.y

    # [DEBUG]
    # print("JAX")
    # print(P)
    # print(q)
    # print(A)
    # print(l)
    # print(u)

    # solving cvxpy problem instances serially
    opt_primal_cvx = []
    opt_dual_cvx = []
    for b in range(B):
        P_np = np.array(P[b, ...])
        q_np = np.array(q[b, ...])
        A_np = np.array(A[b, ...])
        l_np = np.array(l[b, ...])
        u_np = np.array(u[b, ...])

        # [DEBUG]
        # print("CVX")
        # print(P_np)
        # print(q_np)
        # print(A_np)
        # print(l_np)
        # print(u_np)

        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(
                (1 / 2) * cp.quad_form(x, P_np) + q_np.T @ x),
                [A_np @ x <= u_np, l_np <= A_np @ x],
            )
        prob.solve(solver=cp.OSQP)
        opt_primal_cvx.append(x.value)
        _opt_dual_cvx = np.concatenate(
            (
                prob.constraints[0].dual_value,
                prob.constraints[1].dual_value,
            ),
            axis=-1,
        )
        opt_dual_cvx.append(_opt_dual_cvx)
    opt_primal_cvx = np.stack(opt_primal_cvx)
    opt_dual_cvx = np.stack(opt_dual_cvx)

    # checking closeness
    assert np.allclose(np.array(opt_primal_jax), opt_primal_cvx, rtol=1e-3, atol=1e-3)
    assert np.allclose(np.array(opt_dual_jax), opt_dual_cvx, rtol=1e-3, atol=1e-3)
