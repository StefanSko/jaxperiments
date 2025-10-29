#!/usr/bin/env python3
"""
Research validation script: Does fixed seed break MCMC sampling?

This script empirically tests whether using a fixed random seed in HMC
introduces problematic autocorrelation compared to proper random seeding.
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import time

from hmc.sampler import hmc_sample, log_posterior
from hmc.utils import generate_regression_data


def compute_autocorrelation(samples, max_lag=50):
    """Compute autocorrelation function for a 1D array of samples."""
    n = len(samples)
    samples_centered = samples - np.mean(samples)
    variance = np.var(samples)

    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.mean(samples_centered[:-lag] * samples_centered[lag:]) / variance

    return autocorr


def compute_effective_sample_size(samples, max_lag=50):
    """Compute effective sample size using autocorrelation.

    ESS = N / (1 + 2 * sum(autocorr[k])) for k > 0 until autocorr becomes negative.
    """
    autocorr = compute_autocorrelation(samples, max_lag)

    # Sum positive autocorrelations
    sum_autocorr = 0.0
    for k in range(1, len(autocorr)):
        if autocorr[k] <= 0:
            break
        sum_autocorr += autocorr[k]

    n = len(samples)
    ess = n / (1.0 + 2.0 * sum_autocorr)

    return ess


def main():
    print("=" * 70)
    print("RESEARCH VALIDATION: Does Fixed Seed Break MCMC Sampling?")
    print("=" * 70)

    # Generate synthetic data
    print("\n[1/4] Generating synthetic regression data...")
    data_key = random.PRNGKey(42)
    m_true, b_true, sigma_true = 2.5, 1.0, 0.5
    n_data = 1500
    x, y = generate_regression_data(data_key, n_data, m_true, b_true, sigma_true)
    print(f"  Generated {n_data} data points")
    print(f"  True parameters: m={m_true}, b={b_true}, σ={sigma_true}")

    # Configure HMC
    epsilon = 0.001
    n_steps = 15
    n_samples = 2000  # More samples for better statistics
    initial_params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}

    def log_prob_fn(params):
        return log_posterior(params, x, y)

    print(f"\n[2/4] Running HMC with FIXED seed (key=0)...")
    print(f"  Config: epsilon={epsilon}, n_steps={n_steps}, n_samples={n_samples}")
    fixed_key = random.PRNGKey(0)
    samples_fixed = hmc_sample(
        fixed_key, initial_params, n_samples, epsilon, n_steps, log_prob_fn
    )
    print(f"  ✓ Completed {n_samples} samples")

    print(f"\n[3/4] Running HMC with RANDOM seed...")
    random_seed = int(time.time() * 1000) % (2**32)
    random_key = random.PRNGKey(random_seed)
    samples_random = hmc_sample(
        random_key, initial_params, n_samples, epsilon, n_steps, log_prob_fn
    )
    print(f"  ✓ Completed {n_samples} samples (seed={random_seed})")

    print(f"\n[4/4] Computing autocorrelation and ESS...")
    print("\n" + "=" * 70)
    print("RESULTS: Autocorrelation Analysis")
    print("=" * 70)

    param_names = ["m", "b", "log_sigma"]
    max_lag = 50

    for param_name in param_names:
        print(f"\nParameter: {param_name}")
        print("-" * 70)

        # Convert JAX arrays to numpy
        fixed_samples = np.array(samples_fixed[param_name])
        random_samples = np.array(samples_random[param_name])

        # Compute autocorrelation at specific lags
        autocorr_fixed = compute_autocorrelation(fixed_samples, max_lag)
        autocorr_random = compute_autocorrelation(random_samples, max_lag)

        print(f"  Autocorrelation (Fixed Seed):")
        print(f"    Lag 1:  {autocorr_fixed[1]:.4f}")
        print(f"    Lag 5:  {autocorr_fixed[5]:.4f}")
        print(f"    Lag 10: {autocorr_fixed[10]:.4f}")
        print(f"    Lag 20: {autocorr_fixed[20]:.4f}")

        print(f"  Autocorrelation (Random Seed):")
        print(f"    Lag 1:  {autocorr_random[1]:.4f}")
        print(f"    Lag 5:  {autocorr_random[5]:.4f}")
        print(f"    Lag 10: {autocorr_random[10]:.4f}")
        print(f"    Lag 20: {autocorr_random[20]:.4f}")

        # Compute ESS
        ess_fixed = compute_effective_sample_size(fixed_samples, max_lag)
        ess_random = compute_effective_sample_size(random_samples, max_lag)

        print(f"  Effective Sample Size:")
        print(f"    Fixed Seed:  {ess_fixed:.1f} / {n_samples} ({100*ess_fixed/n_samples:.1f}%)")
        print(f"    Random Seed: {ess_random:.1f} / {n_samples} ({100*ess_random/n_samples:.1f}%)")

        # Compare
        ratio = autocorr_fixed[1] / autocorr_random[1] if autocorr_random[1] != 0 else float('inf')
        print(f"  Ratio (Fixed/Random) at lag 1: {ratio:.2f}x")

        if ess_fixed < ess_random * 0.8:
            print(f"  ⚠️  Fixed seed has {100*(1 - ess_fixed/ess_random):.1f}% LOWER ESS")
        elif ess_fixed > ess_random * 1.2:
            print(f"  ⚠️  Fixed seed has {100*(ess_fixed/ess_random - 1):.1f}% HIGHER ESS")
        else:
            print(f"  ℹ️  ESS is similar between fixed and random seed")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Overall assessment
    all_fixed_autocorr = []
    all_random_autocorr = []
    all_ess_fixed = []
    all_ess_random = []

    for param_name in param_names:
        fixed_samples = np.array(samples_fixed[param_name])
        random_samples = np.array(samples_random[param_name])

        autocorr_fixed = compute_autocorrelation(fixed_samples, max_lag)
        autocorr_random = compute_autocorrelation(random_samples, max_lag)
        all_fixed_autocorr.append(autocorr_fixed[1])
        all_random_autocorr.append(autocorr_random[1])

        ess_fixed = compute_effective_sample_size(fixed_samples, max_lag)
        ess_random = compute_effective_sample_size(random_samples, max_lag)
        all_ess_fixed.append(ess_fixed)
        all_ess_random.append(ess_random)

    avg_autocorr_fixed = np.mean(all_fixed_autocorr)
    avg_autocorr_random = np.mean(all_random_autocorr)
    avg_ess_fixed = np.mean(all_ess_fixed)
    avg_ess_random = np.mean(all_ess_random)

    print(f"\nAverage lag-1 autocorrelation:")
    print(f"  Fixed seed:  {avg_autocorr_fixed:.4f}")
    print(f"  Random seed: {avg_autocorr_random:.4f}")
    print(f"  Difference:  {avg_autocorr_fixed - avg_autocorr_random:.4f}")

    print(f"\nAverage effective sample size:")
    print(f"  Fixed seed:  {avg_ess_fixed:.1f} / {n_samples} ({100*avg_ess_fixed/n_samples:.1f}%)")
    print(f"  Random seed: {avg_ess_random:.1f} / {n_samples} ({100*avg_ess_random/n_samples:.1f}%)")
    print(f"  Difference:  {avg_ess_fixed - avg_ess_random:.1f}")

    if avg_autocorr_fixed > avg_autocorr_random * 1.2:
        print("\n✅ HYPOTHESIS CONFIRMED: Fixed seed shows HIGHER autocorrelation")
        print("   This demonstrates that fixed seeds break MCMC sampling!")
    elif avg_autocorr_fixed < avg_autocorr_random * 0.8:
        print("\n❌ HYPOTHESIS REJECTED: Fixed seed shows LOWER autocorrelation")
        print("   This contradicts the expected behavior!")
    else:
        print("\n⚠️  INCONCLUSIVE: Autocorrelation is similar between fixed and random")
        print("   May need more samples or different configuration to see effect")

    if avg_ess_fixed < avg_ess_random * 0.8:
        print("✅ HYPOTHESIS CONFIRMED: Fixed seed has LOWER effective sample size")
        print("   This confirms that fixed seeds reduce sampling efficiency!")
    elif avg_ess_fixed > avg_ess_random * 1.2:
        print("❌ HYPOTHESIS REJECTED: Fixed seed has HIGHER effective sample size")
        print("   This contradicts the expected behavior!")
    else:
        print("⚠️  INCONCLUSIVE: ESS is similar between fixed and random")
        print("   May need more samples or different configuration to see effect")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
