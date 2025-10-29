# ABOUTME: Deliberately buggy HMC implementation that mishandles random keys.
# ABOUTME: Used for research demonstration of why proper randomness is essential.

import jax
import jax.numpy as jnp

from hmc.sampler import log_posterior, grad_log_posterior, leapfrog, hmc_step


def hmc_sample_buggy_reuse_key(key, initial_q, n_samples, epsilon, n_steps, log_prob_fn):
    """Buggy HMC sampler that REUSES the same key without splitting.

    This demonstrates what happens when you don't follow JAX best practices
    for random key management. The algorithm will:
    - Reuse the same random numbers for momentum sampling
    - Reuse the same random numbers for Metropolis acceptance
    - Result in high autocorrelation and poor mixing

    DO NOT USE THIS IN PRODUCTION. This is intentionally broken for educational purposes.

    Args:
        key: JAX random key for sampling
        initial_q: Initial position (dict with parameter values)
        n_samples: Number of samples to draw
        epsilon: Step size for leapfrog integration
        n_steps: Number of leapfrog steps per HMC step
        log_prob_fn: Function that computes log probability at position q

    Returns:
        Dict of arrays, where each array has shape (n_samples,) containing
        the sampled values for each parameter
    """
    # Pre-compute gradient function once for efficiency
    grad_log_prob_fn = jax.grad(log_prob_fn)

    def scan_fn(carry, scan_key):
        """Single iteration of HMC for jax.lax.scan."""
        q_current = carry
        q_new, accepted = hmc_step(scan_key, q_current, epsilon, n_steps, log_prob_fn, grad_log_prob_fn)
        return q_new, q_new

    # BUG: Don't split keys - just reuse the same key for all iterations!
    # This is the critical mistake that breaks MCMC
    keys = jnp.array([key] * n_samples)  # All iterations get SAME key

    # Run scan to collect all samples
    _, samples_dict = jax.lax.scan(scan_fn, initial_q, keys)

    return samples_dict


def hmc_sample_buggy_no_split(key, initial_q, n_samples, epsilon, n_steps, log_prob_fn):
    """Buggy HMC sampler that splits key once then reuses subkeys.

    This is a more subtle bug: we split once but then reuse the same pattern.

    DO NOT USE THIS IN PRODUCTION. This is intentionally broken for educational purposes.

    Args:
        key: JAX random key for sampling
        initial_q: Initial position (dict with parameter values)
        n_samples: Number of samples to draw
        epsilon: Step size for leapfrog integration
        n_steps: Number of leapfrog steps per HMC step
        log_prob_fn: Function that computes log probability at position q

    Returns:
        Dict of arrays, where each array has shape (n_samples,) containing
        the sampled values for each parameter
    """
    # Pre-compute gradient function once for efficiency
    grad_log_prob_fn = jax.grad(log_prob_fn)

    def scan_fn(carry, scan_key):
        """Single iteration of HMC for jax.lax.scan."""
        q_current = carry
        q_new, accepted = hmc_step(scan_key, q_current, epsilon, n_steps, log_prob_fn, grad_log_prob_fn)
        return q_new, q_new

    # BUG: Split once, then tile to reuse the same pattern
    key1, key2 = jax.random.split(key)
    keys = jnp.tile(key1, (n_samples, 1))  # Reuse key1 for all iterations

    # Run scan to collect all samples
    _, samples_dict = jax.lax.scan(scan_fn, initial_q, keys)

    return samples_dict
