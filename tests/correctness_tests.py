import types

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from jax import jit

from jaxosqp import osqp

from .hypothesis_utils import qp_data_dims

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


# ############ #
# SOLVER TESTS #
# ############ #


@given(qp_data_dims())
@settings(deadline=None)
def test_qp_convergence(data: types.GenericAlias(tuple, (np.ndarray,) * 5)):
    """Tests the correctness of the QP solve.

    Parameters
    ----------
    data : tuple[np.ndarray] * 5
        The QP problem data.
        data[0] = P
        data[1] = q
        data[2] = A
        data[3] = l
        data[4] = u
    """
    # unpacking data
    P = data[0]
    q = data[1]
    A = data[2]
    l = data[3]
    u = data[4]

    # solving a jax OSQP problem instance
    # TODO(ahl): once the constructor is fixed to not assume batched inputs, unbatch these data arrays and also the optimization outputs. also, remove the vmap call.
    # TODO(ahl): once the state API stabilizes, change these tests
    P_jax = jnp.array(P)[None, ...]
    q_jax = jnp.array(q)[None, ...]
    A_jax = jnp.array(A)[None, ...]
    l_jax = jnp.array(l)[None, ...]
    u_jax = jnp.array(u)[None, ...]
    prob_jax, data, state = osqp.OSQPProblem.from_data(
        P_jax, q_jax, A_jax, l_jax, u_jax
    )
    state = jax.vmap(prob_jax.solve)(data, state)  # (iters, data, info)
    info = state[-1]

    # solving a cvxpy problem instance
    n = q.shape[0]
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
        [A @ x <= u, l <= A @ x],
    )
    prob.solve(solver=cp.OSQP)

    # checking that impl converges iff cvxpy converges
    converged = info.converged.item()
    # if not converged:
    #     breakpoint()
    assert converged == (prob.status not in ["infeasible", "unbounded"])

    # checking closeness of primal/dual optima
    if converged:
        opt_primal_jax = info.x[0, ...]
        opt_dual_jax = info.y[0, ...]

        opt_primal_cvx = x.value
        opt_dual_cvx = np.max(
            (prob.constraints[0].dual_value, prob.constraints[1].dual_value),
            axis=0,
        )

        assert np.allclose(
            np.array(opt_primal_jax), opt_primal_cvx, rtol=1e-3, atol=1e-3
        )
        assert np.allclose(np.array(opt_dual_jax), opt_dual_cvx, rtol=1e-3, atol=1e-3)
