# ABOUTME: Test suite for HMC sampler core functionality.
# ABOUTME: Tests log probability functions, gradients, leapfrog integrator, and sampling.

import jax
import jax.numpy as jnp
from hmc.sampler import (
    log_prior,
    log_likelihood,
    log_posterior,
    grad_log_posterior,
    leapfrog_step,
    leapfrog,
)


def test_log_prior_shape():
    """Prior accepts parameter dict and returns a scalar."""
    params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}
    result = log_prior(params)
    assert jnp.ndim(result) == 0, "log_prior should return a scalar"


def test_log_likelihood_shape():
    """Likelihood accepts params dict, x, y arrays and returns a scalar."""
    params = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])
    result = log_likelihood(params, x, y)
    assert jnp.ndim(result) == 0, "log_likelihood should return a scalar"


def test_log_posterior_shape():
    """Posterior combines prior and likelihood correctly."""
    params = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])
    result = log_posterior(params, x, y)
    assert jnp.ndim(result) == 0, "log_posterior should return a scalar"


def test_log_prior_values():
    """Check specific log prior values at zero parameters."""
    params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}
    result = log_prior(params)

    # At zero, each Normal(0, sigma) evaluates to -log(sigma * sqrt(2*pi))
    # For m and b: Normal(0, 10) -> -log(10 * sqrt(2*pi))
    # For log_sigma: Normal(0, 2) -> -log(2 * sqrt(2*pi))
    expected = (
        -jnp.log(10.0 * jnp.sqrt(2 * jnp.pi)) +  # m
        -jnp.log(10.0 * jnp.sqrt(2 * jnp.pi)) +  # b
        -jnp.log(2.0 * jnp.sqrt(2 * jnp.pi))     # log_sigma
    )

    assert jnp.isclose(result, expected, rtol=1e-5), \
        f"log_prior at zeros should be {expected}, got {result}"


def numerical_gradient(f, params, epsilon=1e-4):
    """Compute numerical gradient using finite differences.

    Args:
        f: Function that takes params dict and returns scalar
        params: Dict with parameter values
        epsilon: Step size for finite differences

    Returns:
        Dict with same structure as params containing gradients
    """
    grad = {}
    for key in params:
        # Create new dicts to avoid any mutation issues
        params_plus = {k: v for k, v in params.items()}
        params_minus = {k: v for k, v in params.items()}

        params_plus[key] = params[key] + epsilon
        params_minus[key] = params[key] - epsilon

        grad[key] = (f(params_plus) - f(params_minus)) / (2 * epsilon)

    return grad


def test_grad_log_posterior_exists():
    """Verify we can compute gradients of log_posterior."""
    params = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    # Should not raise an error
    grad = grad_log_posterior(params, x, y)
    assert grad is not None, "grad_log_posterior should return a gradient"


def test_grad_log_posterior_shape():
    """Gradient should return dict with same keys as params."""
    params = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    grad = grad_log_posterior(params, x, y)

    assert isinstance(grad, dict), "Gradient should be a dict"
    assert set(grad.keys()) == set(params.keys()), \
        f"Gradient keys {grad.keys()} should match params keys {params.keys()}"

    # Each gradient should be a scalar
    for key in grad:
        assert jnp.ndim(grad[key]) == 0, f"Gradient for {key} should be scalar"


def test_grad_log_posterior_numerical():
    """Compare autodiff gradient to numerical gradient."""
    # Test at multiple parameter values
    test_cases = [
        {"m": 1.0, "b": 0.5, "log_sigma": -1.0},
        {"m": 2.5, "b": -1.0, "log_sigma": 0.0},
        {"m": 0.0, "b": 0.0, "log_sigma": -0.5},
    ]

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.array([2.1, 4.0, 5.9, 8.2, 10.1])

    for params in test_cases:
        # Autodiff gradient
        grad_auto = grad_log_posterior(params, x, y)

        # Numerical gradient
        def f(p):
            return log_posterior(p, x, y)
        grad_num = numerical_gradient(f, params, epsilon=1e-4)

        # Compare with reasonable tolerance for numerical approximation
        # With epsilon=1e-4, we achieve ~0.03% relative error
        for key in params:
            assert jnp.isclose(grad_auto[key], grad_num[key], rtol=1e-3, atol=0.5), \
                f"Gradient for {key} at params {params}: autodiff={grad_auto[key]}, numerical={grad_num[key]}"


def test_leapfrog_step_shapes():
    """Verify leapfrog output shapes match input shapes."""
    q = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    p = {"m": 0.1, "b": -0.2, "log_sigma": 0.05}
    epsilon = 0.01
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    # Create gradient function bound to data
    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    q_new, p_new = leapfrog_step(q, p, epsilon, grad_fn)

    # Check that outputs are dicts with same keys
    assert isinstance(q_new, dict), "q_new should be a dict"
    assert isinstance(p_new, dict), "p_new should be a dict"
    assert set(q_new.keys()) == set(q.keys()), "q_new should have same keys as q"
    assert set(p_new.keys()) == set(p.keys()), "p_new should have same keys as p"

    # Check that all values are scalars
    for key in q_new:
        assert jnp.ndim(q_new[key]) == 0, f"q_new[{key}] should be scalar"
        assert jnp.ndim(p_new[key]) == 0, f"p_new[{key}] should be scalar"


def test_leapfrog_step_deterministic():
    """Same inputs should produce same outputs."""
    q = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    p = {"m": 0.1, "b": -0.2, "log_sigma": 0.05}
    epsilon = 0.01
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    # Run twice with same inputs
    q_new1, p_new1 = leapfrog_step(q, p, epsilon, grad_fn)
    q_new2, p_new2 = leapfrog_step(q, p, epsilon, grad_fn)

    # Results should be identical
    for key in q:
        assert jnp.allclose(q_new1[key], q_new2[key]), \
            f"q_new[{key}] should be deterministic"
        assert jnp.allclose(p_new1[key], p_new2[key]), \
            f"p_new[{key}] should be deterministic"


def test_leapfrog_step_modifies_position():
    """Position should change after leapfrog step."""
    q = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    p = {"m": 0.1, "b": -0.2, "log_sigma": 0.05}
    epsilon = 0.01
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    q_new, p_new = leapfrog_step(q, p, epsilon, grad_fn)

    # At least one position component should change (typically all will)
    position_changed = False
    for key in q:
        if not jnp.allclose(q[key], q_new[key]):
            position_changed = True
            break

    assert position_changed, "Position should change after leapfrog step"


def test_leapfrog_step_modifies_momentum():
    """Momentum should change after leapfrog step."""
    q = {"m": 1.0, "b": 0.5, "log_sigma": -1.0}
    p = {"m": 0.1, "b": -0.2, "log_sigma": 0.05}
    epsilon = 0.01
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.5, 2.5, 3.5])

    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    q_new, p_new = leapfrog_step(q, p, epsilon, grad_fn)

    # At least one momentum component should change (typically all will)
    momentum_changed = False
    for key in p:
        if not jnp.allclose(p[key], p_new[key]):
            momentum_changed = True
            break

    assert momentum_changed, "Momentum should change after leapfrog step"


def hamiltonian(q, p, log_prob_fn):
    """Compute Hamiltonian (total energy) for HMC.

    H(q, p) = -log_prob(q) + 0.5 * ||p||^2

    Args:
        q: Position (dict with parameter values)
        p: Momentum (dict with same structure as q)
        log_prob_fn: Function that computes log probability at q

    Returns:
        Scalar Hamiltonian value
    """
    # Potential energy: -log_prob(q)
    potential = -log_prob_fn(q)

    # Kinetic energy: 0.5 * sum(p_i^2)
    kinetic = 0.5 * sum(p[key]**2 for key in p)

    return potential + kinetic


def test_leapfrog_energy_conservation():
    """Verify leapfrog integrator approximately conserves energy."""
    # Use a simple quadratic potential for testing
    # This makes energy conservation easier to verify
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.array([2.1, 4.0, 5.9, 8.2, 10.1])

    # Initial position and momentum
    q = {"m": 2.0, "b": 0.0, "log_sigma": -0.5}
    p = {"m": 0.3, "b": 0.1, "log_sigma": 0.05}

    # Small step size for better energy conservation
    epsilon = 0.001
    n_steps = 10

    # Define log probability function and gradient
    def log_prob(params):
        return log_posterior(params, x, y)

    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    # Compute initial energy
    H_initial = hamiltonian(q, p, log_prob)

    # Run multiple leapfrog steps
    q_current, p_current = q, p
    for _ in range(n_steps):
        q_current, p_current = leapfrog_step(q_current, p_current, epsilon, grad_fn)

    # Compute final energy
    H_final = hamiltonian(q_current, p_current, log_prob)

    # Energy should be approximately conserved
    # With small step size (0.001) and few steps (10), error should be very small
    # Leapfrog is second-order accurate, so error scales as O(epsilon^2)
    energy_diff = jnp.abs(H_final - H_initial)

    # For epsilon=0.001, expect error < 0.01
    assert energy_diff < 0.01, \
        f"Energy conservation violated: |H_final - H_initial| = {energy_diff} (initial: {H_initial}, final: {H_final})"


def test_leapfrog_trajectory():
    """Verify full leapfrog trajectory runs multiple steps correctly."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.array([2.1, 4.0, 5.9, 8.2, 10.1])

    # Initial position and momentum
    q = {"m": 2.0, "b": 0.0, "log_sigma": -0.5}
    p = {"m": 0.3, "b": 0.1, "log_sigma": 0.05}

    epsilon = 0.01
    n_steps = 20

    def grad_fn(params):
        return grad_log_posterior(params, x, y)

    # Run full trajectory
    q_final, p_final = leapfrog(q, p, epsilon, n_steps, grad_fn)

    # Check outputs are dicts with correct keys
    assert isinstance(q_final, dict), "q_final should be a dict"
    assert isinstance(p_final, dict), "p_final should be a dict"
    assert set(q_final.keys()) == set(q.keys()), "q_final should have same keys as q"
    assert set(p_final.keys()) == set(p.keys()), "p_final should have same keys as p"

    # Check values are scalars
    for key in q_final:
        assert jnp.ndim(q_final[key]) == 0, f"q_final[{key}] should be scalar"
        assert jnp.ndim(p_final[key]) == 0, f"p_final[{key}] should be scalar"

    # Verify position and momentum have changed
    position_changed = any(not jnp.allclose(q[key], q_final[key]) for key in q)
    momentum_changed = any(not jnp.allclose(p[key], p_final[key]) for key in p)

    assert position_changed, "Position should change after trajectory"
    assert momentum_changed, "Momentum should change after trajectory"

    # Verify the trajectory is equivalent to running leapfrog_step n_steps times
    q_manual, p_manual = q, p
    for _ in range(n_steps):
        q_manual, p_manual = leapfrog_step(q_manual, p_manual, epsilon, grad_fn)

    # Results should match
    for key in q:
        assert jnp.allclose(q_final[key], q_manual[key], rtol=1e-10), \
            f"q_final[{key}] should match manual iteration"
        assert jnp.allclose(p_final[key], p_manual[key], rtol=1e-10), \
            f"p_final[{key}] should match manual iteration"
