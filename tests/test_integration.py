# ABOUTME: Integration tests for the complete HMC sampling pipeline.
# ABOUTME: Tests end-to-end workflow from data generation through sampling to validation.

import jax.numpy as jnp
import jax.random as random
from hmc.sampler import hmc_sample, log_posterior
from hmc.utils import generate_regression_data, get_hmc_config


def test_full_regression_sampling():
    """Test the complete HMC sampling pipeline on synthetic regression data."""
    # Set up known parameters
    m_true = 2.5
    b_true = 1.0
    sigma_true = 0.5

    # Generate synthetic data
    key = random.PRNGKey(42)
    key_data, key_sample = random.split(key)
    x, y = generate_regression_data(key_data, n_samples=200,
                                    m_true=m_true, b_true=b_true,
                                    sigma_true=sigma_true)

    # Set up HMC configuration
    # Use moderate epsilon for stable sampling with good initialization
    n_warmup = 100
    n_samples = 200
    config = get_hmc_config(epsilon=0.001, n_steps=15)

    # Initial parameters - use reasonable starting values to avoid extreme gradients
    initial_q = {"m": 2.0, "b": 0.5, "log_sigma": -0.5}

    # Create log probability function with data
    def log_prob_fn(q):
        return log_posterior(q, x, y)

    # Run HMC sampler (returns all samples including warmup)
    all_samples = hmc_sample(
        key_sample,
        initial_q,
        n_samples=n_warmup + n_samples,
        epsilon=config["epsilon"],
        n_steps=config["n_steps"],
        log_prob_fn=log_prob_fn
    )

    # Verify all samples have correct structure and shapes
    total_samples = n_warmup + n_samples
    assert "m" in all_samples, "Samples should contain 'm'"
    assert "b" in all_samples, "Samples should contain 'b'"
    assert "log_sigma" in all_samples, "Samples should contain 'log_sigma'"
    assert all_samples["m"].shape == (total_samples,), f"Expected m shape ({total_samples},), got {all_samples['m'].shape}"
    assert all_samples["b"].shape == (total_samples,), f"Expected b shape ({total_samples},), got {all_samples['b'].shape}"
    assert all_samples["log_sigma"].shape == (total_samples,), f"Expected log_sigma shape ({total_samples},), got {all_samples['log_sigma'].shape}"

    # Discard warmup samples (first n_warmup samples)
    m_samples = all_samples["m"][n_warmup:]
    b_samples = all_samples["b"][n_warmup:]
    log_sigma_samples = all_samples["log_sigma"][n_warmup:]

    # Verify samples vary (not all identical)
    assert jnp.std(m_samples) > 0, "m samples should vary"
    assert jnp.std(b_samples) > 0, "b samples should vary"
    assert jnp.std(log_sigma_samples) > 0, "log_sigma samples should vary"

    # Verify mean is reasonably close to true values
    # Use generous tolerances since this is a small test
    m_mean = jnp.mean(m_samples)
    b_mean = jnp.mean(b_samples)
    sigma_mean = jnp.exp(jnp.mean(log_sigma_samples))

    assert jnp.abs(m_mean - m_true) < 1.0, f"m mean {m_mean} should be within 1.0 of true value {m_true}"
    assert jnp.abs(b_mean - b_true) < 1.0, f"b mean {b_mean} should be within 1.0 of true value {b_true}"
    assert jnp.abs(sigma_mean - sigma_true) < 1.0, f"sigma mean {sigma_mean} should be within 1.0 of true value {sigma_true}"

    # Verify samples explore parameter space (check range)
    m_range = jnp.max(m_samples) - jnp.min(m_samples)
    b_range = jnp.max(b_samples) - jnp.min(b_samples)

    assert m_range > 0.01, f"m samples should explore parameter space, range: {m_range}"
    assert b_range > 0.01, f"b samples should explore parameter space, range: {b_range}"
