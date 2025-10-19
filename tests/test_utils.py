# ABOUTME: Tests for HMC utility functions including data generation.

import jax.numpy as jnp
import jax.random as random
import pytest
import matplotlib.pyplot as plt
from hmc.utils import generate_regression_data, DEFAULT_HMC_CONFIG, get_hmc_config, plot_trace


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


def test_default_config_values():
    """Test that DEFAULT_HMC_CONFIG contains expected values."""
    assert DEFAULT_HMC_CONFIG["epsilon"] == 0.01
    assert DEFAULT_HMC_CONFIG["n_steps"] == 20
    assert DEFAULT_HMC_CONFIG["n_warmup"] == 500
    assert DEFAULT_HMC_CONFIG["n_samples"] == 1000


def test_config_overrides():
    """Test that get_hmc_config properly overrides default values."""
    # Test with no overrides
    config = get_hmc_config()
    assert config == DEFAULT_HMC_CONFIG
    assert config is not DEFAULT_HMC_CONFIG  # Should be a copy

    # Test with single override
    config = get_hmc_config(epsilon=0.05)
    assert config["epsilon"] == 0.05
    assert config["n_steps"] == 20  # Other values unchanged
    assert config["n_warmup"] == 500
    assert config["n_samples"] == 1000

    # Test with multiple overrides
    config = get_hmc_config(epsilon=0.02, n_steps=50, n_samples=2000)
    assert config["epsilon"] == 0.02
    assert config["n_steps"] == 50
    assert config["n_warmup"] == 500  # Unchanged
    assert config["n_samples"] == 2000

    # Test that original DEFAULT_HMC_CONFIG is not modified
    assert DEFAULT_HMC_CONFIG["epsilon"] == 0.01
    assert DEFAULT_HMC_CONFIG["n_steps"] == 20


def test_plot_trace_creates_figure():
    """Test that plot_trace creates a figure with correct subplots."""
    # Create sample data
    n_samples = 100
    samples = {
        "m": jnp.linspace(0, 1, n_samples),
        "b": jnp.linspace(1, 2, n_samples),
        "log_sigma": jnp.linspace(-1, 0, n_samples),
    }
    param_names = ["m", "b", "log_sigma"]

    # Create plot
    fig = plot_trace(samples, param_names, title="Test Trace Plot")

    # Verify figure is created
    assert fig is not None
    assert isinstance(fig, plt.Figure)

    # Verify correct number of subplots
    axes = fig.get_axes()
    assert len(axes) == len(param_names), f"Expected {len(param_names)} subplots, got {len(axes)}"

    # Verify each subplot has correct labels
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        assert ax.get_ylabel() == param_name, f"Expected ylabel '{param_name}', got '{ax.get_ylabel()}'"
        assert ax.get_xlabel() == "Iteration", f"Expected xlabel 'Iteration', got '{ax.get_xlabel()}'"

    # Clean up
    plt.close(fig)
