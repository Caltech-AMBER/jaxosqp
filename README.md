# jaxosqp

[OSQP](https://osqp.org/), but in `jax`. 

### Installation

First, clone the repo
``` git clone https://github.com/pculbertson/jaxosqp ```

Then, install the package locally using

``` pip install -e . ``` 

### Usage

OSQP solves quadratic programs of the form:

$$\begin{align} 
\min_x & \quad \frac{1}{2} x^T P x + q^T x \\
\text{s.t. } & \quad \ell \leq A x \leq u. 
\end{align}$$ 

This solver is built to solve batches of QPs in this form, assuming the problem data `P, q, A, l, u` are stored as `jnp.array`s of the appropriate size.

For example, we can generate random batches of QP data (the first OSQP benchmark): 
```
import jax
import jax.numpy as jnp

from jax import random
from jax.experimental import sparse
from jaxosqp import osqp

# Generate some random problem data.
B = 100
n = 100
m = 2*n

outer = lambda A: A @ A.T + 1e-2 * jnp.eye(n)
key = random.PRNGKey(208)
key, subkey = random.split(key)
P = sparse.random_bcoo(key, (B, n, n), nse=0.15, generator=random.normal).todense()
P = jax.vmap(outer)(P) # Ensure P >= 0.
key, subkey = random.split(key)
q = random.normal(subkey, (B, n))
key, subkey = random.split(key)

# Add some random constraints (a la OSQP benchmarks
A = sparse.random_bcoo(subkey, (B, m, n), nse=0.15, generator=random.normal).todense()
key, subkey = random.split(key)
l = -random.uniform(subkey, (B, m))
key, subkey = random.split(key)
u = random.uniform(subkey, (B, m))
```

Next, we create an OSQP problem instance:
```prob, data, state = osqp.OSQPProblem.from_data(P, q, A, l, u)```

In short, `prob` is a top-level container for the problem config; `data` is a container for the problem params `P, q, A, l, u`,
and `state` holds the internal variables used during the solve.

We can solve our problem by running:
```iters, data, state = prob.solve(data, state)```

The optimal primal solution will be stored in `state.x`, and the optimal Lagrange multipliers will be stored in `state.y`.

The problem statuses are stored in `state.converged`, `state.primal_infeas`, and `state.dual_infeas`; the solver stops when every problem in the batch has hit a termination condition. 

### Project Roadmap

There are a number of features/improvements needed to make this package production-ready. 

Grouping the big goals by topic:

1. Performance:
    - [x] Make `solve()` call `vmap` across batch dimension to decouple problems.
    - [ ] ~Support sparse (i.e., `sparse.BCOO`) `P, A, kkt_mat`.~ (Sparse routines inefficient in CUDA).
    - [ ] Implement `OSQPProblem.update_rho()`.
    - [x] Profile code + check for bottlenecks.
    - [ ] Ensure we're not losing too much performance with `jdc.copy_and_mutate()`.
    - [ ] Multi-GPU support. 

2. Testing / benchmarking:
    - [ ] Write unit tests for internal solver methods:
        - [ ] `OSQPProblem.check_convergence()`
        - [ ] `OSQPProblem.check_primal_infeas()`
        - [ ] `OSQPProblem.check_dual_infeas()`
        - [ ] `OSQPProblem.step()`

    - [ ] Write top-level unit tests for correctness of QP solves (against Gurobi, OSQP on CPU):
        - [ ] Random ineq. QP
        - [ ] Random eq. QP
        - [ ] Optimal control

    - [ ] Implement timing benchmarking against existing QP / LP solvers.

3. Convenience / QoL:
    - [ ] Method to mutate `OSQPData` to change just one field (and update `converged, primal_infeas, dual_infeas` of OSQPProblem).
    - [ ] Use `vmap` to construct a batch of problems, instead of hardcoding size of `A` into factory method.
