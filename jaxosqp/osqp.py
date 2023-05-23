import dataclasses
import jax_dataclasses as jdc
import jax
import jax.numpy as jnp

from functools import partial
from jaxosqp import utils
from jax import tree_util

@jdc.pytree_dataclass
class OSQPConfig:
	"""
	Config object for OSQP solver.
	"""
	alpha: jdc.Static[float] = 1.6
	"""Relaxation parameter, in (0, 2)."""
	rho_bar: jdc.Static[float] = 1e-1
	"""Initial penalty parameter."""
	sigma: jdc.Static[float] = 1e-6
	"""Cost modification to ensure P > 0."""
	eps_abs: jdc.Static[float] = 1e-3
	"""Absolute tolerance for convergence."""
	eps_rel: jdc.Static[float] = 1e-3
	"""Relative tolerance for convergence."""
	eps_pinf: jdc.Static[float] = 1e-4
	"""Tolerance for primal infeasibility."""
	eps_dinf: jdc.Static[float] = 1e-4
	"""Tolerance for dual infeasibility."""
	max_iters: jdc.Static[int] = 4000
	"""Maximum number of iterations to run during solve."""
	term_check_iters: jdc.Static[int] = 25
	"""Frequency of termination check."""
	rho_update_iters: jdc.Static[int] = 100
	"""Frequency of rho update."""

@jdc.pytree_dataclass
class OSQPData:
	"""
	Stores data for OSQP problem. 
	"""
	P: jnp.ndarray
	"""Cost matrix (problem parameter)."""
	q: jnp.ndarray
	"""Cost vector (problem parameter)."""
	A: jnp.ndarray
	"""Constraint matrix (problem parameter)."""
	l: jnp.ndarray
	"""Lower bound on inequality constraints (problem parameter)."""
	u: jnp.ndarray
	"""Upper bound on inequality constraints (problem parameter)."""

@jdc.pytree_dataclass
class OSQPState:
	"""
	Internal state for the OSQP solver.
	"""
	x: jnp.ndarray
	""" Container for primal variables."""
	z: jnp.ndarray
	""" Projected constraint values. """
	chi: jnp.ndarray
	""" Auxilliary copy of x for ADMM."""
	zeta: jnp.ndarray
	""" Auxilliary copy of z for ADMM."""
	y: jnp.ndarray
	""" Lagrange multipliers for Ax = z."""
	dx: jnp.ndarray
	""" Last step taken in x."""
	dy: jnp.ndarray
	"""Last step taken in y."""
	converged: jnp.ndarray
	"""Flags for convergence."""
	primal_infeas: jnp.ndarray
	"""Flags for primal infeasibility."""
	dual_infeas: jnp.ndarray
	"""Flags for dual infeasibility."""
	kkt_mat: jnp.ndarray
	"""Full, dense KKT matrix used for ADMM step computation."""
	kkt_q: jnp.ndarray
	"""Q factor of KKT matrix."""
	kkt_r: jnp.ndarray
	"""R factor of KKT matrix."""
	rho: jnp.ndarray
	"""ADMM penalty weights."""

@jdc.pytree_dataclass
class OSQPProblem:
	"""
	Top-level for a batched QP problem.
	"""
	n: jdc.Static[int]
	"""Number of decision variables."""
	m: jdc.Static[int]
	"""Number of constraints."""
	B: jdc.Static[int]
	"""Batch size."""
	config: OSQPConfig
	"""Config parameters for problem."""

	@classmethod
	def from_data(cls, P, q, A, l, u, config=OSQPConfig()):
		"""
		Main constructor for OSQPProblems. Takes in problem data 
		(cost terms + affine constraints) and returns a problem wrapper
		as well as data + state objects initialized correctly.
		"""

		# Extract shapes from constraint matrix.
		assert len(A.shape) == 3
		B, m, n = A.shape

		# Create OSQPProblem object.
		prob = cls(n, m, B, config)
		
		data, state = prob.build_data_state(P, q, A, l, u)

		return prob, data, state

	def build_data_state(self, P, q, A, l, u):
		"""
		Helper method to wrap problem data + create initial solver state.

		Ensures problem data are shaped correctly + computes initial KKT matrix + QR factors.
		"""

		# Pull out shape vars for convenience. 
		B, m, n = self.B, self.m, self.n

		# Reshape data to ensure correct shapes. 
		P = P.reshape(B, n, n)
		q = q.reshape(B, n)
		l = l.reshape(B, m)
		u = u.reshape(B, m)

		# Allocate storage for solver data.
		x = jnp.zeros((B, n))
		z = jnp.zeros((B, m))
		chi = jnp.zeros((B, n))
		zeta = jnp.zeros((B, m))
		y = jnp.zeros((B, m))
		dx = jnp.zeros((B, n))
		dy = jnp.zeros((B, m))

		# Allocate problem status flags.
		converged = jnp.zeros(B).astype(bool)
		primal_infeas = jnp.zeros(B).astype(bool)
		dual_infeas = jnp.zeros(B).astype(bool)

		# Setup penalty weights.
		rho = jnp.where(l == u, 1e3 * self.config.rho_bar, self.config.rho_bar)

		# Compute initial KKT matrix.
		kkt_mat, kkt_q, kkt_r = build_kkt(P, A, rho, self.config.sigma)

		# Wrap problem data in OSQPData object. 
		data = OSQPData(P, q, A, l, u)

		return data, OSQPState(x, z, chi, zeta, y, dx, dy, converged, primal_infeas, dual_infeas, kkt_mat, kkt_q, kkt_r, rho)

	def step(self, data, state):
		"""
		Main step of OSQP algorithm; computes one ADMM step on the problem.
		"""
		# Compute residuals vector.
		r = state.kkt_q.transpose(0, 2, 1) @ utils.hcat(self.config.sigma * state.x - data.q, state.z - (1/state.rho) * state.y)[:, :, None]
		r = r.squeeze()

		# Solve KKT system (in QR form).
		r = jax.vmap(jax.scipy.linalg.solve_triangular)(state.kkt_r, r)

		# Pull out ADMM vars corresponding to x, z. 
		chi = r[:, :self.n]
		zeta = state.z + (1/state.rho) * (r[:, self.n:] - state.y)

		# Update x, z with a filtered step.
		new_x = self.config.alpha * chi + (1 - self.config.alpha) * state.x
		new_z = self.config.alpha * zeta + (1 - self.config.alpha) * state.z + (1/state.rho) * state.y

		# Compute new Lagrange multipliers.
		new_y = state.rho * (new_z - jnp.clip(new_z, data.l, data.u))

		# Note: jdc dataclasses are immutable; use this helper to copy them. 
		with jdc.copy_and_mutate(state, validate=False) as new_state:

			new_state.chi = chi
			new_state.zeta = zeta

			new_state.dx = new_x - state.x
			new_state.dy = new_y - state.y

			new_state.x = new_x
			new_state.y = new_y
			new_state.z = jnp.clip(new_z, data.l, data.u)

		return new_state

	def solve_inner(self, val):
		"""
		Helper function for jax.lax.while_loop; encodes main logic of OSQP solve loop. 
		"""

		# Unpack `val` tuple. 
		ii, data, state = val

		# Take ADMM step. 
		state = self.step(data, state)

		# If we should check termintaion this iteration, update term. vars.
		state = jax.lax.cond(
			jnp.mod(ii, self.config.term_check_iters) == 0, 
			lambda val: self.check_converged(*val), 
			lambda val: state, (data, state))

		state = jax.lax.cond(
			jnp.mod(ii, self.config.term_check_iters) == 0, 
			lambda val: self.check_primal_infeas(*val), 
			lambda val: state, (data, state))

		state = jax.lax.cond(
			jnp.mod(ii, self.config.term_check_iters) == 0, 
			lambda val: self.check_dual_infeas(*val), 
			lambda val: state, (data, state))

		# If we should scale the penalty this iteration, update penalty. 
		state = jax.lax.cond(
			jnp.mod(ii, self.config.rho_update_iters) == 0,
			lambda val: self.update_rho(*val),
			lambda val: state,
			(data, state))

		return (ii+1, data, state)

	def solve_cond_fun(self, val):
		"""
		Termination condition for solve loop. Stops if we hit max iters or all problems terminate.
		"""
		ii, data, state = val

		is_finished = jnp.logical_or(state.converged, state.primal_infeas)
		is_finished = jnp.logical_or(is_finished, state.dual_infeas)

		return jnp.logical_and(ii < self.config.max_iters, jnp.logical_not(jnp.all(is_finished)))

	@jax.jit
	def solve(self, data, state):
		"""
		Top-level (jit'd) solve function; returns state of last solver iter.
		"""
		return jax.lax.while_loop(self.solve_cond_fun, self.solve_inner, (0, data, state))[-1]

	def check_converged(self, data, state):
		"""
		Checks optimality of each subproblem -- (26) in OSQP paper. 

		Returns a state with updated `converged` variables. 
		"""

		# Compute some matrix-vector products we'll need. 
		Ax = (data.A @ state.x[:, :, None]).squeeze()
		Px = (data.P @ state.x[:, :, None]).squeeze()
		Aty = (data.A.transpose(0, 2, 1) @ state.y[:, :, None]).squeeze()

		# Compute primal and dual residuals.
		r_prim = Ax - state.z
		r_dual = Px + data.q + Aty

		# Compute l-inf norms of a bunch of quantities we need.
		prim_norm = jnp.linalg.norm(r_prim, ord=jnp.inf, axis=-1)
		dual_norm = jnp.linalg.norm(r_dual, ord=jnp.inf, axis=-1)

		Ax_norm = jnp.linalg.norm(Ax, ord=jnp.inf, axis=-1)
		Px_norm = jnp.linalg.norm(Px, ord=jnp.inf, axis=-1)
		Aty_norm = jnp.linalg.norm(Aty, ord=jnp.inf, axis=-1)
		z_norm = jnp.linalg.norm(state.z, ord=jnp.inf, axis=-1)
		q_norm = jnp.linalg.norm(data.q, ord=jnp.inf, axis=-1)

		# Compute the epsilon (combined abs / rel tolerance) needed for convergence. 
		eps_prim = self.config.eps_abs + self.config.eps_rel * jnp.maximum(Ax_norm, z_norm)
		eps_dual = self.config.eps_abs + self.config.eps_rel * jnp.maximum(jnp.maximum(Px_norm, Aty_norm), q_norm)

		# Converged if primal and dual residuals are below desired tol. 
		prim_converged = prim_norm <= eps_prim
		dual_converged = dual_norm <= eps_dual

		with jdc.copy_and_mutate(state) as new_state:
			new_state.converged = jnp.logical_and(prim_converged, dual_converged)

		return new_state


	def check_primal_infeas(self, data, state):
		"""
		Check primal infeasibility of each subproblem. Returns a state with updated primal_infeas variables. 
		"""

		# Precompute infinity norms of dy, A * dy.
		dy_inf = jnp.linalg.norm(state.dy, ord=jnp.inf, axis=-1)
		Ady_inf = jnp.linalg.norm((data.A.transpose(0, 2, 1) @ state.dy[:, :, None]).squeeze(), ord=jnp.inf, axis=-1)

		# Compute constraint residual terms.
		const_residuals = jnp.sum(data.u * jnp.maximum(0., state.dy) + data.l * jnp.minimum(0., state.dy), axis=-1)

		# Problem is primal infeasible if both inequalities hold.
		nullspace = Ady_inf <= self.config.eps_pinf * dy_inf
		slackness = const_residuals <= self.config.eps_pinf * dy_inf

		with jdc.copy_and_mutate(state) as new_state:
			new_state.primal_infeas = jnp.logical_and(nullspace, slackness)

		return new_state

	def check_dual_infeas(self, data, state):
		"""
		Checks problems for dual infeasibility. Returns a state with updated `dual_infeas` variables. 
		"""
		# Compute some matrix-vector products we'll need. 
		Pdx = (data.P @ state.dx[:, :, None]).squeeze()
		Adx = (data.A @ state.dx[:, :, None]).squeeze()
		qdx = jnp.sum(data.q * state.dx, axis=-1)

		# Compute some l-inf norms we'll need. 
		dx_norm = jnp.linalg.norm(state.dx, ord=jnp.inf, axis=-1)
		Pdx_norm = jnp.linalg.norm(Pdx, ord=jnp.inf, axis=-1)
		qdx_norm = jnp.linalg.norm(qdx, ord=jnp.inf, axis=-1)

		# Enforce upper / lower bounds only on constraints where l/u are finite. 
		lower_bound = jnp.where(jnp.isinf(data.l), -self.config.eps_dinf * dx_norm[:, None], -jnp.inf)
		upper_bound = jnp.where(jnp.isinf(data.u), self.config.eps_dinf * dx_norm[:, None], jnp.inf)

		dual_infeas = Pdx_norm <= self.config.eps_dinf * dx_norm
		dual_infeas = jnp.logical_and(dual_infeas, qdx <= self.config.eps_dinf * dx_norm)
		dual_infeas = jnp.logical_and(dual_infeas, jnp.all(lower_bound <= Adx, axis=-1))
		dual_infeas = jnp.logical_and(dual_infeas, jnp.all(Adx <= upper_bound, axis=-1))

		with jdc.copy_and_mutate(state) as new_state:
			new_state.dual_infeas = dual_infeas

		return new_state

	def update_rho(self, data, state):
		"""
		Update the penalty parameters rho according to Sec. 5.2.
		"""

		# TODO(pculbert): implement penalty scheduling. 
		return state


def build_kkt(P, A, rho, sigma):
	"""
	Helper function to build the OSQP KKT system (and QR factorize it).
	"""
	kkt_mat = jax.vmap(build_single_kkt, in_axes=(0, 0, 0, None))(P, A, rho, sigma)
	kkt_q, kkt_r = jnp.linalg.qr(kkt_mat)

	return kkt_mat, kkt_q, kkt_r

def build_single_kkt(P, A, rho, sigma):
	"""
	Function to build a single OSQP KKT system (vmap this for batches).
	"""
	return utils.vcat(
		utils.hcat(P + sigma * jnp.eye(P.shape[0]), A.T),
		utils.hcat(A, -jnp.diag(1/rho)))
