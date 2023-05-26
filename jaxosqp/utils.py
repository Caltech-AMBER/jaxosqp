import jax.numpy as jnp

def vcat(*args):
	return jnp.concatenate(args, axis=0)

def hcat(*args):
	return jnp.concatenate(args, axis=1)

def linf_norm(x, **args):
	return jnp.linalg.norm(x, ord=jnp.inf, **args)