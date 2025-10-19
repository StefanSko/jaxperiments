# ABOUTME: Core HMC sampling functions including log probability and gradient computations.
# ABOUTME: Implements leapfrog integrator and Metropolis-Hastings acceptance for HMC.

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def log_prior(params):
    """Compute log prior probability for regression parameters.

    Uses weakly informative priors:
    - m ~ Normal(0, 10)
    - b ~ Normal(0, 10)
    - log_sigma ~ Normal(0, 2)

    Args:
        params: Dict with keys "m", "b", "log_sigma"

    Returns:
        Scalar log prior probability
    """
    log_p_m = norm.logpdf(params["m"], loc=0.0, scale=10.0)
    log_p_b = norm.logpdf(params["b"], loc=0.0, scale=10.0)
    log_p_log_sigma = norm.logpdf(params["log_sigma"], loc=0.0, scale=2.0)

    return log_p_m + log_p_b + log_p_log_sigma


def log_likelihood(params, x, y):
    """Compute log likelihood for linear regression.

    Model: y ~ Normal(m*x + b, exp(log_sigma))

    Args:
        params: Dict with keys "m", "b", "log_sigma"
        x: Array of input values
        y: Array of observed output values

    Returns:
        Scalar log likelihood
    """
    m = params["m"]
    b = params["b"]
    sigma = jnp.exp(params["log_sigma"])

    y_pred = m * x + b
    log_lik = jnp.sum(norm.logpdf(y, loc=y_pred, scale=sigma))

    return log_lik


def log_posterior(params, x, y):
    """Compute log posterior probability (prior + likelihood).

    Args:
        params: Dict with keys "m", "b", "log_sigma"
        x: Array of input values
        y: Array of observed output values

    Returns:
        Scalar log posterior probability
    """
    return log_prior(params) + log_likelihood(params, x, y)


# Define gradient function at module level for efficiency
_grad_log_posterior_fn = jax.grad(log_posterior)


def grad_log_posterior(params, x, y):
    """Compute gradient of log posterior with respect to parameters.

    Uses JAX automatic differentiation to compute gradients.

    Args:
        params: Dict with keys "m", "b", "log_sigma"
        x: Array of input values
        y: Array of observed output values

    Returns:
        Dict with same structure as params containing gradients
    """
    return _grad_log_posterior_fn(params, x, y)


def leapfrog_step(q, p, epsilon, grad_log_prob_fn):
    """Perform a single leapfrog integration step.

    Implements the standard leapfrog (Störmer-Verlet) integrator:
    1. Half step for momentum: p = p + (epsilon/2) * grad_log_prob(q)
    2. Full step for position: q = q + epsilon * p
    3. Half step for momentum: p = p + (epsilon/2) * grad_log_prob(q)

    Args:
        q: Current position (dict with parameter values)
        p: Current momentum (dict with same structure as q)
        epsilon: Step size for integration
        grad_log_prob_fn: Function that computes gradient of log probability

    Returns:
        Tuple (q_new, p_new) with updated position and momentum
    """
    # Half step for momentum
    grad = grad_log_prob_fn(q)
    p_half = {key: p[key] + 0.5 * epsilon * grad[key] for key in p}

    # Full step for position
    q_new = {key: q[key] + epsilon * p_half[key] for key in q}

    # Half step for momentum
    grad_new = grad_log_prob_fn(q_new)
    p_new = {key: p_half[key] + 0.5 * epsilon * grad_new[key] for key in p_half}

    return q_new, p_new
