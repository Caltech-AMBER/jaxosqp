import dataclasses
import jax_dataclasses as jdc
import jax
import jax.numpy as jnp

from functools import partial
from jaxosqp import utils
from typing import Tuple

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
	scaling_iters: jdc.Static[int] = 10
	"""Number of iters to run problem scaling."""
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

	If problem scaling is not used, D, E, c will be all ones -- otherwise
	these will have non-trivial values and problem data is scaled accordingly. 
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
	D: jnp.ndarray
	"""Scaling matrix w/r/t primal variables."""
	E: jnp.ndarray
	"""Scaling matrix w/r/t dual variables."""
	c: jnp.ndarray
	"""Scaling vector for cost function."""

@jdc.pytree_dataclass
class OSQPState:
	"""
	Internal state for the OSQP solver.
	"""
	x: jnp.ndarray
	""" Container for primal variables."""
	z: jnp.ndarray
	""" Projected constraint values. """
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
	kkt_lu: Tuple[jnp.ndarray, jnp.ndarray]
	"""LU factorization of the KKT matrix; a tuple of permuations / data for each matrix."""
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

		Ensures problem data are shaped correctly + computes initial KKT matrix + LU factors.
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
		y = jnp.zeros((B, m))
		dx = jnp.zeros((B, n))
		dy = jnp.zeros((B, m))

		# Allocate problem status flags.
		converged = jnp.zeros(B).astype(bool)
		primal_infeas = jnp.zeros(B).astype(bool)
		dual_infeas = jnp.zeros(B).astype(bool)

		# Initialize scaling parameters.
		D = jnp.ones((B, n))
		E = jnp.ones((B, m))
		c = jnp.ones((B))

		# Wrap problem data in OSQPData object. 
		data = OSQPData(P, q, A, l, u, D, E, c)

		data = jax.vmap(self.scale_problem)(data)

		# Setup penalty weights.
		rho = jnp.where(l == u, 1e3 * self.config.rho_bar, self.config.rho_bar)

		# Build KKT matrix and LU factorize it. 
		kkt_mat, kkt_lu = build_kkt(data.P, data.A, rho, self.config.sigma)

		return data, OSQPState(x, z, y, dx, dy, converged, primal_infeas, dual_infeas, kkt_mat, kkt_lu, rho)

	def step(self, data, state):
		"""
		Main step of OSQP algorithm; computes one ADMM step on the problem.
		"""
		# Compute residuals vector.
		# r = utils.vcat(self.config.sigma * state.x - data.q, state.z - jnp.diag(1/state.rho) @ state.y)
		r = jnp.concatenate((self.config.sigma * state.x - data.q, state.z - (1/state.rho) * state.y), axis=0)

		# Solve the KKT system in LU form. 
		kkt_sol = jax.scipy.linalg.lu_solve(state.kkt_lu, r)

		# Pull out ADMM vars corresponding to x, z. 
		chi = kkt_sol[:self.n]
		zeta = state.z + (1 / state.rho) * (kkt_sol[self.n:] - state.y)

		# Update x, z with a filtered step.ν
		new_x = self.config.alpha * chi + (1 - self.config.alpha) * state.x
		new_z = self.config.alpha * zeta + (1 - self.config.alpha) * state.z + (1/state.rho) * state.y
		new_z = jnp.clip(new_z,data.l, data.u)

		# Compute new Lagrange multipliers.
		new_y = state.y + state.rho * (self.config.alpha * zeta + (1 - self.config.alpha) * state.z - new_z)

		# Note: jdc dataclasses are immutable; use this helper to copy them. 
		with jdc.copy_and_mutate(state) as new_state:

			new_state.dx = new_x - state.x
			new_state.dy = new_y - state.y

			new_state.x = new_x
			new_state.y = new_y
			new_state.z = new_z

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

		return jnp.logical_and(ii < self.config.max_iters, jnp.logical_not(is_finished))

	# @jax.jit
	def solve(self, data, state):
		"""
		Top-level (jit'd) solve function; returns state of last solver iter.
		""" 
		state = self.scale_vars(data, state)
		iters, data, state = jax.lax.while_loop(self.solve_cond_fun, self.solve_inner, (0, data, state))
		state = self.unscale_vars(data, state)

		return iters, data, state

	def check_converged(self, data, state):
		"""
		Checks optimality of each subproblem -- (26) in OSQP paper. 

		Returns a state with updated `converged` variables. 
		"""

		# Compute some matrix-vector products we'll need. 

		# Compute primal and dual residuals.
		r_prim = (1 / data.E) * (data.A @ state.x - state.z)
		r_dual = (1 / data.c) * (1 / data.D) * (data.P @ state.x + data.q + data.A.T @ state.y)

		# Compute l-inf norms of a bunch of quantities we need.
		prim_norm = utils.linf_norm(r_prim)
		dual_norm = utils.linf_norm(r_dual)

		# Compute l-inf norms for primal tolerance. 
		Ax_norm = utils.linf_norm((1 / data.E) * (data.A @ state.x))
		z_norm = utils.linf_norm((1 / data.E) * state.z)

		# Compute l-inf norms for dual lolerance
		Px_norm = utils.linf_norm((1 / data.D) * (data.P @ state.x))
		Aty_norm = utils.linf_norm((1 / data.D) * (data.A.T @ state.y))
		q_norm = utils.linf_norm(jnp.diag(1 / data.D) @ data.q)

		# Compute the epsilon (combined abs / rel tolerance) needed for convergence. 
		eps_prim = self.config.eps_abs + self.config.eps_rel * jnp.maximum(Ax_norm, z_norm)
		eps_dual = self.config.eps_abs + self.config.eps_rel * (1 / data.c) * jnp.maximum(jnp.maximum(Px_norm, Aty_norm), q_norm)

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
		dy_inf = utils.linf_norm(jnp.diag(data.E) @ state.dy)
		Ady_inf = utils.linf_norm(jnp.diag(1 / data.D) @ (data.A.T @ state.dy))

		# Compute constraint residual terms.
		const_residuals = jnp.dot(data.u, jnp.maximum(0., state.dy)) + jnp.dot(data.l, jnp.minimum(0., state.dy))

		# Problem is primal infeasible if both inequalities hold.
		nullspace = Ady_inf <= self.config.eps_pinf * dy_inf
		slackness = const_residuals <= self.config.eps_pinf * dy_inf
		is_nonzero = dy_inf >= jnp.finfo(state.x.dtype).eps

		with jdc.copy_and_mutate(state) as new_state:
			new_state.primal_infeas = jnp.logical_and(jnp.logical_and(nullspace, slackness), is_nonzero)

		return new_state

	def check_dual_infeas(self, data, state):
		"""
		Checks problems for dual infeasibility. Returns a state with updated `dual_infeas` variables. 
		"""
		# Compute some matrix-vector products we'll need. 
		Pdx = jnp.diag(1 / data.D) @ data.P @ state.dx
		Adx = jnp.diag(1 / data.E) @ data.A @ state.dx
		qdx = jnp.dot(data.q, state.dx)

		# Compute some l-inf norms we'll need. 
		dx_norm = utils.linf_norm(jnp.diag(1 / data.D) @ state.dx)
		Pdx_norm = utils.linf_norm(Pdx)

		# Enforce upper / lower bounds only on constraints where l/u are finite. 
		lower_bound = jnp.where(jnp.isinf(data.l), -self.config.eps_dinf * dx_norm, -jnp.inf)
		upper_bound = jnp.where(jnp.isinf(data.u), self.config.eps_dinf * dx_norm, jnp.inf)

		dual_infeas = Pdx_norm <= self.config.eps_dinf * data.c * dx_norm
		dual_infeas = jnp.logical_and(dual_infeas, qdx <= self.config.eps_dinf * data.c * dx_norm)
		dual_infeas = jnp.logical_and(dual_infeas, jnp.all(lower_bound <= Adx))
		dual_infeas = jnp.logical_and(dual_infeas, jnp.all(Adx <= upper_bound))

		with jdc.copy_and_mutate(state) as new_state:
			new_state.dual_infeas = dual_infeas

		return new_state

	def scale_problem(self, data):
		"""Implements Ruiz equilibriation for problem data (Algorithm 2). Returns modified data object with
			non-trivial D, E, c to approximately equally scale problem data."""

		def cond_fun(val):
			"""
			Condition for stopping the while loop -- max iters. 
			"""
			ii, data = val
			return ii < self.config.scaling_iters

		def body_fun(val):
			"""
			Main body of while loop. 
			"""
			ii, data = val

			# Compute delta terms by looking at l-inf norms of the "data matrix" M (32).
			delta_d = 1 / jnp.sqrt(1e-8 + utils.linf_norm(utils.vcat(data.P, data.A), axis=0))
			delta_e = 1 / jnp.sqrt(1e-8 + utils.linf_norm(data.A.T, axis=0))

			# Update problem data with new scaling. 
			P = data.c * jnp.diag(delta_d) @ data.P @ jnp.diag(delta_d)
			q = data.c * jnp.diag(delta_d) @ data.q
			A = jnp.diag(delta_e) @ data.A @ jnp.diag(delta_d)
			l = jnp.diag(delta_e) @ data.l
			u = jnp.diag(delta_e) @ data.u

			# Compute new cost scaling term.
			gamma = 1 / jnp.maximum(jnp.mean(utils.linf_norm(P, axis=0)), utils.linf_norm(q))

			# Update cost parameters with new scaling.
			P = gamma * P
			q = gamma * q

			c = gamma * data.c

			# Return new OSQPData object with scaled parameters. 
			with jdc.copy_and_mutate(data) as new_data:
				new_data.P = P
				new_data.q = q
				new_data.A = A
				new_data.l = l
				new_data.u = u

				new_data.D = delta_d * data.D
				new_data.E = delta_e * data.E
				new_data.c = c

			return (ii+1, new_data)

		return jax.lax.while_loop(cond_fun, body_fun, (0, data))[-1]

	def update_rho(self, data, state):
		"""
		Update the penalty parameters rho according to Sec. 5.2.
		"""

		return state

	def unscale_vars(self, data, state):
		with jdc.copy_and_mutate(state) as new_state:
			new_state.x = jnp.diag(data.D) @ state.x
			new_state.y = (1 / data.c) * jnp.diag(data.E) @ state.y

		return new_state

	def scale_vars(self, data, state):
		with jdc.copy_and_mutate(state) as new_state:
			new_state.x = jnp.diag(1 / data.D) @ state.x
			new_state.y = data.c * jnp.diag(1 / data.E) @ state.y

		return new_state


def build_kkt(P, A, rho, sigma):
	"""
	Helper function to build the OSQP KKT system (and LU factorize it).
	"""
	kkt_mat = jax.vmap(build_single_kkt, in_axes=(0, 0, 0, None))(P, A, rho, sigma)
	kkt_lu = jax.scipy.linalg.lu_factor(kkt_mat)

	return kkt_mat, kkt_lu

def build_single_kkt(P, A, rho, sigma):
	"""
	Function to build a single OSQP KKT system (vmap this for batches).
	"""

	return utils.vcat(
		utils.hcat(P + sigma * jnp.eye(P.shape[0]), A.T),
		utils.hcat(A, -jnp.diag(1/rho)))
