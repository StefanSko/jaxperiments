#!/usr/bin/env python3
"""Quick test: Can we get decent HMC mixing with better parameters?"""

import jax.numpy as jnp
import jax.random as random
import numpy as np

from hmc.sampler import hmc_sample, log_posterior
from hmc.utils import generate_regression_data

# Generate data
data_key = random.PRNGKey(42)
m_true, b_true, sigma_true = 2.5, 1.0, 0.5
n_data = 1500
x, y = generate_regression_data(data_key, n_data, m_true, b_true, sigma_true)

def log_prob_fn(params):
    return log_posterior(params, x, y)

initial_params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}

# Test different epsilon values
epsilons = [0.001, 0.01, 0.05, 0.1]
n_steps = 15
n_samples = 500

print("Testing HMC with different step sizes:")
print("=" * 60)

for eps in epsilons:
    key = random.PRNGKey(0)
    samples = hmc_sample(key, initial_params, n_samples, eps, n_steps, log_prob_fn)

    # Compute simple autocorrelation at lag 1
    m_samples = np.array(samples['m'])
    m_centered = m_samples - np.mean(m_samples)
    autocorr = np.corrcoef(m_centered[:-1], m_centered[1:])[0, 1]

    # Compute rough ESS
    variance = np.var(m_samples)
    ess_approx = n_samples / (1 + 2 * max(0, autocorr))

    print(f"\nεpsilon = {eps:6.3f}:")
    print(f"  Mean m: {np.mean(m_samples):.4f} (true: {m_true})")
    print(f"  Std m:  {np.std(m_samples):.4f}")
    print(f"  Lag-1 autocorr: {autocorr:.4f}")
    print(f"  Approx ESS: {ess_approx:.1f} / {n_samples} ({100*ess_approx/n_samples:.1f}%)")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("If all epsilons show terrible mixing (<10% ESS), our HMC is broken.")
print("If larger epsilon shows good mixing (>30% ESS), we just need better tuning.")
