import jax.numpy as jnp

OSQP_MAX_SCALING = 1e4
OSQP_MIN_SCALING = 1e-4


def vcat(*args):
    return jnp.concatenate(args, axis=0)

def hcat(*args):
    return jnp.concatenate(args, axis=1)

def linf_norm(x, **args):
    return jnp.linalg.norm(x, ord=jnp.inf, **args)

def limit_scaling(x):
    """Limit the scaling factors within the bounds of the OSQP C implementation

    Values below OSQP_MIN_SCALING are set to 1.0 while values above OSQP_MAX_SCALING are clipped
    """
    x = jnp.where(x < OSQP_MIN_SCALING, 1.0, x)
    return jnp.where(x > OSQP_MAX_SCALING, OSQP_MAX_SCALING, x)
