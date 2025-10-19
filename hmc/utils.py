# ABOUTME: Utility functions for HMC sampler including data generation and configuration.
# ABOUTME: Contains helpers for creating synthetic regression data and managing HMC settings.

import jax.numpy as jnp
import jax.random as random


# Default HMC hyperparameters
DEFAULT_HMC_CONFIG = {
    "epsilon": 0.01,
    "n_steps": 20,
    "n_warmup": 500,
    "n_samples": 1000,
}


def get_hmc_config(**overrides):
    """Get HMC configuration with optional overrides.

    Args:
        **overrides: Keyword arguments to override default config values

    Returns:
        Dictionary with HMC configuration
    """
    config = DEFAULT_HMC_CONFIG.copy()
    config.update(overrides)
    return config


def generate_regression_data(key, n_samples, m_true, b_true, sigma_true, x_range=(0, 10)):
    """Generate synthetic linear regression data.

    Args:
        key: JAX random key
        n_samples: Number of data points to generate
        m_true: True slope
        b_true: True intercept
        sigma_true: True noise standard deviation
        x_range: Tuple of (min, max) for x values

    Returns:
        Tuple of (x, y) arrays
    """
    key_x, key_noise = random.split(key)

    # Generate x uniformly in x_range
    x = random.uniform(key_x, shape=(n_samples,), minval=x_range[0], maxval=x_range[1])

    # Generate y = m_true * x + b_true + noise
    noise = random.normal(key_noise, shape=(n_samples,)) * sigma_true
    y = m_true * x + b_true + noise

    return x, y
