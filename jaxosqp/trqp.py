# TODO(pculbert): check imports.
import dataclasses
import jax_dataclasses as jdc
import jax
import jax.numpy as jnp

from functools import partial
from jaxosqp import osqp, utils
from typing import Tuple

@jdc.pytree_dataclass
class TRQPData(osqp.OSQPData):
    """
    Stores data for trust region QP. 

    This extends the OSQPData dataclass from osqp -- trust region-specific fields
    are added to the end of the constructor. 
    """
    delta: jnp.ndarray
    """Norm bound for trust region constraint."""
    
@jdc.pytree_dataclass
class TRQPState(osqp.OSQPState):
    """
    Internal state for the trust region QP. 
    """
    w: jnp.ndarray
    """ Lagrange multipliers for ||x|| <= delta."""
    dz: jnp.ndarray
    """ Last step taken in z."""

@jdc.pytree_dataclass
class TRQPProblem(osqp.OSQPProblem):
    """
    Top-level for a batched trust-region QP.
    """
    n: jdc.Static[int]
    """Number of decision variables."""
    m: jdc.Static[int]
    """Number of constraints."""
    config: osqp.OSQPConfig
    """Config parameters for problem."""

    @classmethod
    def from_data(cls, P, q, A, l, u, delta=jnp.inf, config=osqp.OSQPConfig(sigma=1e-1)):
        """
        Main constructor for OSQPProblems. Takes in problem data 
        (cost terms + affine constraints) and returns a problem wrapper
        as well as data + state objects initialized correctly.
        """

        # Extract shapes from constraint matrix.
        assert len(A.shape) == 2
        m, n = A.shape

        # Create OSQPProblem object.
        prob = cls(n, m, config)
        
        data, state = prob.build_data_state(P, q, A, l, u, delta)

        return prob, data, state

    def build_data_state(self, P, q, A, l, u, delta):
        """
        Helper method to wrap problem data + create initial solver state.

        Ensures problem data are shaped correctly + computes initial KKT matrix + LU factors.
        """

        # Pull out shape vars for convenience. 
        m, n = self.m, self.n

        # Allocate storage for solver data.
        x = jnp.zeros((n,))
        z = jnp.zeros((m,))
        w = jnp.zeros((n,))
        y = jnp.zeros((m,))
        dx = jnp.zeros((n,))
        dz = jnp.zeros((m,))
        dy = jnp.zeros((m,))

        # Allocate problem status flags.
        converged = jnp.zeros(()).astype(bool)
        primal_infeas = jnp.zeros(()).astype(bool)
        dual_infeas = jnp.zeros(()).astype(bool)

        # Initialize scaling parameters.
        D = jnp.ones((n,))
        E = jnp.ones((m,))
        c = jnp.ones(())

        # Wrap problem data in OSQPData object. 
        data = TRQPData(P, q, A, l, u, D, E, c, delta)

        data = self.scale_problem(data)

        # Setup penalty weights.
        rho = jnp.where(l == u, 1e3 * self.config.rho_bar, self.config.rho_bar)

        # Build KKT matrix and LU factorize it. 
        kkt_mat, kkt_lu = osqp.build_kkt(data.P, data.A, rho, self.config.sigma)

        return data, TRQPState(x, z, y, dx, dy, converged, primal_infeas, dual_infeas, kkt_mat, kkt_lu, rho, w, dz)
        
    def step(self, data, state):
        """
        Main step of OSQP algorithm; computes one ADMM step on the problem.
        """
        # Compute residuals vector.
        # r = utils.vcat(self.config.sigma * state.x - data.q, state.z - jnp.diag(1/state.rho) @ state.y)
        r = jnp.concatenate((self.config.sigma * state.x - data.q - state.w, state.z - (1/state.rho) * state.y), axis=0)

        # Solve the KKT system in LU form. 
        kkt_sol = jax.scipy.linalg.lu_solve(state.kkt_lu, r)

        # Pull out ADMM vars corresponding to x, z. 
        chi = kkt_sol[:self.n]
        zeta = state.z + (1 / state.rho) * (kkt_sol[self.n:] - state.y)

        # Update x, z with a filtered step.
        new_x = self.config.alpha * chi + (1 - self.config.alpha) * state.x
        new_x = new_x * jnp.minimum(1., data.delta / (jnp.finfo(state.x.dtype).eps + jnp.linalg.norm(data.D * new_x)))
        
        new_z = self.config.alpha * zeta + (1 - self.config.alpha) * state.z + (1/state.rho) * state.y
        new_z = jnp.clip(new_z,data.l, data.u)

        # Compute new Lagrange multipliers.
        new_w = state.w + self.config.sigma * (self.config.alpha * chi + (1 - self.config.alpha) * state.x - new_x)
        new_y = state.y + state.rho * (self.config.alpha * zeta + (1 - self.config.alpha) * state.z - new_z)

        # Note: jdc dataclasses are immutable; use this helper to copy them. 
        with jdc.copy_and_mutate(state) as new_state:

            new_state.dx = new_x - state.x
            new_state.dz = new_z - state.z
            new_state.dy = new_y - state.y

            new_state.w = new_w
            new_state.x = new_x
            new_state.y = new_y
            new_state.z = new_z

        return new_state

    def check_converged(self, data, state):
        """
        Checks optimality of each subproblem -- (26) in OSQP paper. 

        Returns a state with updated `converged` variables. 
        """

        # Compute primal and dual residuals.
        r_prim = (1 / data.E) * (data.A @ state.x - state.z)
        r_dual = utils.vcat(self.config.sigma * (data.D) * state.dx, state.rho * (data.E) * state.dz)
        
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
