import jax.numpy as jnp

def vcat(*args):
	return jnp.concatenate(args, axis=0)

def hcat(*args):
	return jnp.concatenate(args, axis=1)