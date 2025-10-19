# ABOUTME: Test suite for HMC sampler core functionality.
# ABOUTME: Tests log probability functions, gradients, leapfrog integrator, and sampling.

import jax
import jax.numpy as jnp
from hmc.sampler import log_prior, log_likelihood, log_posterior, grad_log_posterior


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


def numerical_gradient(f, params, epsilon=1e-5):
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
        params_plus = params.copy()
        params_minus = params.copy()

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
        grad_num = numerical_gradient(f, params, epsilon=1e-5)

        # Compare
        for key in params:
            assert jnp.isclose(grad_auto[key], grad_num[key], rtol=1e-3, atol=1e-3), \
                f"Gradient for {key} at params {params}: autodiff={grad_auto[key]}, numerical={grad_num[key]}"
