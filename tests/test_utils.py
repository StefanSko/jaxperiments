# ABOUTME: Tests for HMC utility functions including data generation.

import jax.numpy as jnp
import jax.random as random
import pytest
from hmc.utils import generate_regression_data


def test_generate_regression_data_shape():
    """Test that generate_regression_data returns arrays of correct shape."""
    key = random.PRNGKey(0)
    n_samples = 100
    m_true = 2.5
    b_true = 1.0
    sigma_true = 0.5

    x, y = generate_regression_data(key, n_samples, m_true, b_true, sigma_true)

    assert x.shape == (n_samples,), f"Expected x shape ({n_samples},), got {x.shape}"
    assert y.shape == (n_samples,), f"Expected y shape ({n_samples},), got {y.shape}"


def test_generate_regression_data_reproducible():
    """Test that same key gives same data."""
    key = random.PRNGKey(42)
    n_samples = 50
    m_true = 3.0
    b_true = -1.5
    sigma_true = 0.8

    x1, y1 = generate_regression_data(key, n_samples, m_true, b_true, sigma_true)
    x2, y2 = generate_regression_data(key, n_samples, m_true, b_true, sigma_true)

    assert jnp.allclose(x1, x2), "Same key should produce identical x values"
    assert jnp.allclose(y1, y2), "Same key should produce identical y values"


def test_generate_regression_data_different_seeds():
    """Test that different keys give different data."""
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(1)
    n_samples = 50
    m_true = 2.0
    b_true = 1.0
    sigma_true = 0.5

    x1, y1 = generate_regression_data(key1, n_samples, m_true, b_true, sigma_true)
    x2, y2 = generate_regression_data(key2, n_samples, m_true, b_true, sigma_true)

    assert not jnp.allclose(x1, x2), "Different keys should produce different x values"
    assert not jnp.allclose(y1, y2), "Different keys should produce different y values"
