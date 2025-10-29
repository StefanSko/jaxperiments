# ABOUTME: Deliberately buggy HMC implementation that mishandles random keys.
# ABOUTME: Used for research demonstration of why proper randomness is essential.

import jax
import jax.numpy as jnp
import jax.random as random

from hmc.sampler import log_posterior, grad_log_posterior, leapfrog, hmc_step


def hmc_step_buggy_internal_reset(q, epsilon, n_steps, log_prob_fn, grad_log_prob_fn):
    """Buggy HMC step that internally resets key to PRNGKey(0).

    This simulates the common beginner mistake of creating a new key inside
    a function instead of passing it as a parameter. This was the actual bug
    in the Laplacean project (commit 21e2e91).

    When called in a loop, every iteration uses identical random numbers,
    completely breaking the MCMC sampling.

    DO NOT USE THIS IN PRODUCTION. This is intentionally broken for educational purposes.

    Args:
        q: Current position (dict with parameter values)
        epsilon: Step size for leapfrog integration
        n_steps: Number of leapfrog steps per HMC step
        log_prob_fn: Function that computes log probability at position q
        grad_log_prob_fn: Function that computes gradient of log probability

    Returns:
        Tuple (q_new, accepted) where:
            - q_new is the new position (dict)
            - accepted is a boolean indicating if proposal was accepted
    """
    # BUG: Create a fresh key with seed 0 every time!
    # This means every call to this function uses IDENTICAL random numbers
    key = random.PRNGKey(0)  # ⚠️ THE BUG!

    # Split key for momentum sampling and acceptance
    key_momentum, key_accept = random.split(key)

    # Sample momentum from standard normal
    param_names = sorted(q.keys())
    param_keys = random.split(key_momentum, len(param_names))
    p = {name: random.normal(param_keys[i], shape=()) for i, name in enumerate(param_names)}

    # Compute initial energy
    log_prob_current = log_prob_fn(q)
    kinetic_current = 0.5 * jnp.sum(jnp.array([p[k]**2 for k in p]))
    H_current = -log_prob_current + kinetic_current

    # Run leapfrog to get proposal
    q_proposal, p_proposal = leapfrog(q, p, epsilon, n_steps, grad_log_prob_fn)

    # Compute proposal energy
    log_prob_proposal = log_prob_fn(q_proposal)
    kinetic_proposal = 0.5 * jnp.sum(jnp.array([p_proposal[k]**2 for k in p_proposal]))
    H_proposal = -log_prob_proposal + kinetic_proposal

    # Metropolis acceptance
    log_accept_prob = H_current - H_proposal
    log_u = jnp.log(random.uniform(key_accept))

    accepted = log_accept_prob > log_u

    # Return accepted proposal or current position
    q_new = jax.tree.map(lambda prop, curr: jnp.where(accepted, prop, curr), q_proposal, q)

    return q_new, accepted


def hmc_sample_buggy_internal_reset(initial_q, n_samples, epsilon, n_steps, log_prob_fn):
    """Buggy HMC sampler that uses scan but still has the PRNGKey(0) bug.

    This replicates the actual bug pattern: each HMC step internally creates
    PRNGKey(0), causing all iterations to use identical randomness.

    Even though we use jax.lax.scan for efficiency, the bug still manifests
    because the key is created inside the scanned function.

    DO NOT USE THIS IN PRODUCTION. This is intentionally broken for educational purposes.

    Args:
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

    def scan_fn(carry, _):
        """Single iteration of buggy HMC for jax.lax.scan.

        Note: The second argument (_) would normally be a key, but we ignore it
        because the bug creates PRNGKey(0) inside hmc_step_buggy_internal_reset.
        """
        q_current = carry
        # BUG: hmc_step_buggy_internal_reset creates PRNGKey(0) internally!
        q_new, accepted = hmc_step_buggy_internal_reset(
            q_current, epsilon, n_steps, log_prob_fn, grad_log_prob_fn
        )
        return q_new, q_new

    # Run scan - note we pass None as the array to scan over since we don't use it
    _, samples_dict = jax.lax.scan(scan_fn, initial_q, None, length=n_samples)

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
