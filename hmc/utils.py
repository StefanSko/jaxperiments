# ABOUTME: Utility functions for HMC sampler including data generation and configuration.
# ABOUTME: Contains helpers for creating synthetic regression data and managing HMC settings.

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt


# Default HMC hyperparameters
DEFAULT_HMC_CONFIG = {
    "epsilon": 0.01,        # Step size for leapfrog integration
    "n_steps": 20,          # Number of leapfrog steps per HMC iteration
    "n_warmup": 500,        # Number of warmup samples to discard (user must handle)
    "n_samples": 1000,      # Number of post-warmup samples to keep
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


def plot_trace(samples, param_names, title="Trace Plot"):
    """Create trace plots for MCMC samples.

    Args:
        samples: Dictionary of parameter arrays with shape (n_samples,) each
        param_names: List of parameter names to plot (keys in samples dict)
        title: Overall title for the plot

    Returns:
        matplotlib Figure object
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params))

    # Handle single parameter case
    if n_params == 1:
        axes = [axes]

    for i, param_name in enumerate(param_names):
        axes[i].plot(samples[param_name])
        axes[i].set_ylabel(param_name)
        axes[i].set_xlabel("Iteration")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    return fig
