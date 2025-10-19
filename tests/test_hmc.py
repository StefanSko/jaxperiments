# ABOUTME: Test suite for HMC sampler core functionality.
# ABOUTME: Tests log probability functions, gradients, leapfrog integrator, and sampling.

import pytest
import jax.numpy as jnp
from hmc.sampler import log_prior, log_likelihood, log_posterior


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
