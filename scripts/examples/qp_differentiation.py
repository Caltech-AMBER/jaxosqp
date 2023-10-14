import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd, vmap
from jax.tree_util import tree_flatten

from jaxosqp import osqp


# creating a loss and a jacobian function wrt all the inputs
@jit
def loss(P, q, A, l, u):
    """Toy loss that sums the optimal solution."""
    prob, data, state = osqp.OSQPProblem.from_data(P, q, A, l, u)
    state = vmap(prob.solve)(data, state)
    return jnp.sum(state[-1].x)
jac_loss = jit(jacfwd(loss, argnums=(0, 1, 2, 3, 4)))

if __name__ == "__main__":
    # [NOTE] roughly the problem size needed for PFC bound with D=4
    B = 16  # batch size
    n = 17  # num decision vars
    m = 14  # num constraints

    # P = jnp.tile(jnp.diag(jnp.arange(n, dtype=jnp.float32), (B, 1, 1))  # case: QP
    P = jnp.zeros((B, n, n))  # case: LP

    # q = jnp.tile(jnp.arange(n, dtype=jnp.float32), (B, 1))
    # A = jnp.tile(jnp.arange(m * n, dtype=jnp.float32).reshape((m, n)), (B, 1, 1))
    # l = jnp.tile(-jnp.arange(m, dtype=jnp.float32), (B, 1))
    # u = jnp.tile(jnp.arange(m, dtype=jnp.float32), (B, 1))
    q = jnp.array(np.random.randn(B, n))
    A = jnp.array(np.random.randn(B, m, n))
    l = -jnp.array(np.random.rand(B, m))
    u = jnp.array(np.random.rand(B, m))

    # pre-compiling
    loss(P, q, A, l, u)
    jac_loss(P, q, A, l, u)

    # timing the loss and jacobian computations
    t0 = time.time()
    val = loss(P, q, A, l, u).block_until_ready()
    t1 = time.time()
    print(f"Avg. time to compute loss: {(t1 - t0) / B}")  # 4.88e-5

    t0 = time.time()
    # first element of list of Jacobians, done this way to block correctly, since
    # it only returns after all elements in the output of jac_loss are computed.
    # for non-timing, `jac_loss(P, q, A, l, u)` is sufficient.
    Jval = tree_flatten(jac_loss(P, q, A, l, u))[0][0].block_until_ready()
    t1 = time.time()
    print(f"Avg. time to compute Jacobian of loss: {(t1 - t0) / B}")  # 7.42e-3
