import time

import jax
import jax.numpy as jnp
import numpy
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
    # TODO(ahl): once batching not required for config, unbatch this example
    # generating simple problem data
    B = 100  # batch size
    P = jnp.tile(jnp.diag(jnp.array([1.0, 2.0, 3.0])), (B, 1, 1))
    q = jnp.tile(jnp.array([1.0, 2.0, 3.0]), (B, 1))
    A = jnp.tile(jnp.arange(6, dtype=jnp.float32).reshape((2, 3)), (B, 1, 1))
    l = jnp.tile(jnp.array([-1.0, -2.0]), (B, 1))
    u = jnp.tile(jnp.array([1.0, 2.0]), (B, 1))

    # pre-compiling
    loss(P, q, A, l, u)
    jac_loss(P, q, A, l, u)

    # timing the loss and jacobian computations
    t0 = time.time()
    val = loss(P, q, A, l, u).block_until_ready()
    t1 = time.time()
    print(f"Avg. time to compute loss: {(t1 - t0) / B}")

    t0 = time.time()
    # first element of list of Jacobians, done this way to block correctly, since
    # it only returns after all elements in the output of jac_loss are computed.
    # for non-timing, `jac_loss(P, q, A, l, u)` is sufficient.
    Jval = tree_flatten(jac_loss(P, q, A, l, u))[0][0].block_until_ready()
    t1 = time.time()
    print(f"Avg. time to compute Jacobian of loss: {(t1 - t0) / B}")
