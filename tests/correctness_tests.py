import types

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings

from jaxosqp import osqp

from .hypothesis_utils import qp_random2_dims

# from hypothesis import given, settings, strategies as st


# @given(st.one_of(qp_random1_dims(), qp_random2_dims()))
@given(qp_random2_dims(n_range=(1, 10), density=0.15))
@settings(deadline=None)
def test_qp_convergence1(data: types.GenericAlias(tuple, (np.ndarray,) * 5)):
    """Tests the correctness of the QP solve. Generates random QPs from 2 strategies.

    (1) A strategy defined by us;
    (2) A strategy defined by the OSQP authors.
    See the hypothesis_utils.py file for details.

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
        P_jax, q_jax, A_jax, l_jax, u_jax, config=osqp.OSQPConfig(max_iters=10000)
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
    prob.solve(
        solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, verbose=True
    )  # OSQP default tols

    # checking that impl converges iff cvxpy converges
    converged_jax = info.converged.item()
    converged_cvx = prob.status not in ["infeasible", "unbounded"]
    assert converged_jax == converged_cvx

    # # checking closeness of primal/dual optima
    # if converged_jax:
    #     opt_primal_jax = np.array(info.x[0, ...])
    #     opt_dual_jax = np.array(info.y[0, ...])

    #     opt_primal_cvx = x.value
    #     opt_dual_cvx = np.max(
    #         (prob.constraints[0].dual_value, prob.constraints[1].dual_value),
    #         axis=0,
    #     )

    #     opt_value_jax = opt_primal_jax @ P @ opt_primal_jax + q @ opt_primal_jax
    #     opt_value_cvx = opt_primal_cvx @ P @ opt_primal_cvx + q @ opt_primal_cvx

    #     u_vio_jax = u - A @ opt_primal_jax
    #     l_vio_jax = A @ opt_primal_jax - l
    #     u_vio_cvx = u - A @ opt_primal_cvx
    #     l_vio_cvx = A @ opt_primal_cvx - l

    #     # assert np.allclose(opt_primal_jax, opt_primal_cvx, rtol=1e-3, atol=1e-3)
    #     # assert np.allclose(opt_dual_jax, opt_dual_cvx, rtol=1e-3, atol=1e-3)
    #     if not np.all(u_vio_jax >= 0.0):
    #         breakpoint()
    #     assert np.all(u_vio_jax >= 0.0)
    #     assert np.all(l_vio_jax >= 0.0)
    #     assert np.all(u_vio_cvx >= 0.0)
    #     assert np.all(l_vio_cvx >= 0.0)
    #     assert np.allclose(opt_value_jax, opt_value_cvx)
