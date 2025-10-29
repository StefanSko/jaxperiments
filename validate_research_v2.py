#!/usr/bin/env python3
"""
Research validation script: Is proper randomization essential for MCMC convergence?

This script empirically tests whether proper random key handling is critical for
HMC (Hamiltonian Monte Carlo) to work correctly.

HYPOTHESIS: MCMC algorithms fundamentally require proper randomness management.
Improper handling (reusing keys, not splitting) breaks the algorithm.

IMPLICATION: If randomness is so essential that core inference algorithms break
without it, why do we blindly fix seeds everywhere else in data science?
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np

from hmc.sampler import hmc_sample, log_posterior
from hmc.sampler_buggy import hmc_sample_buggy_reuse_key
from hmc.utils import generate_regression_data


def compute_autocorrelation(samples, max_lag=50):
    """Compute autocorrelation function for a 1D array of samples."""
    n = len(samples)
    samples_centered = samples - np.mean(samples)
    variance = np.var(samples)

    if variance == 0:
        return np.ones(max_lag + 1)

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


def analyze_samples(samples_dict, param_names, max_lag=50):
    """Analyze MCMC samples for autocorrelation and ESS."""
    results = {}

    for param_name in param_names:
        samples_array = np.array(samples_dict[param_name])

        autocorr = compute_autocorrelation(samples_array, max_lag)
        ess = compute_effective_sample_size(samples_array, max_lag)

        results[param_name] = {
            'mean': np.mean(samples_array),
            'std': np.std(samples_array),
            'autocorr_lag1': autocorr[1],
            'autocorr_lag5': autocorr[5],
            'autocorr_lag10': autocorr[10],
            'ess': ess,
            'ess_percent': 100 * ess / len(samples_array),
            'samples': samples_array
        }

    return results


def main():
    print("=" * 80)
    print("RESEARCH VALIDATION: Is Proper Randomization Essential for MCMC?")
    print("=" * 80)
    print()
    print("HYPOTHESIS: MCMC algorithms require proper random key management to converge.")
    print("Improper handling (reusing keys) breaks the algorithm completely.")
    print()

    # Generate synthetic data
    print("[1/4] Generating synthetic regression data...")
    data_key = random.PRNGKey(42)
    m_true, b_true, sigma_true = 2.5, 1.0, 0.5
    n_data = 1500
    x, y = generate_regression_data(data_key, n_data, m_true, b_true, sigma_true)
    print(f"  Generated {n_data} data points")
    print(f"  True parameters: m={m_true}, b={b_true}, σ={sigma_true}")
    print()

    # Configure HMC
    epsilon = 0.001
    n_steps = 15
    n_samples = 1000
    initial_params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}

    def log_prob_fn(params):
        return log_posterior(params, x, y)

    # Run CORRECT HMC (proper key splitting)
    print("[2/4] Running CORRECT HMC (proper random key splitting)...")
    print(f"  Config: epsilon={epsilon}, n_steps={n_steps}, n_samples={n_samples}")
    correct_key = random.PRNGKey(0)
    samples_correct = hmc_sample(
        correct_key, initial_params, n_samples, epsilon, n_steps, log_prob_fn
    )
    print(f"  ✓ Completed {n_samples} samples")
    print()

    # Run BUGGY HMC (reuses same key)
    print("[3/4] Running BUGGY HMC (reuses same key without splitting)...")
    print(f"  Config: epsilon={epsilon}, n_steps={n_steps}, n_samples={n_samples}")
    print(f"  ⚠️  WARNING: This implementation intentionally mishandles random keys!")
    buggy_key = random.PRNGKey(0)
    samples_buggy = hmc_sample_buggy_reuse_key(
        buggy_key, initial_params, n_samples, epsilon, n_steps, log_prob_fn
    )
    print(f"  ✓ Completed {n_samples} samples")
    print()

    # Analyze results
    print("[4/4] Analyzing samples and computing diagnostics...")
    print()
    param_names = ["m", "b", "log_sigma"]
    max_lag = 50

    results_correct = analyze_samples(samples_correct, param_names, max_lag)
    results_buggy = analyze_samples(samples_buggy, param_names, max_lag)

    # Print results
    print("=" * 80)
    print("RESULTS: Comparison of Correct vs Buggy HMC")
    print("=" * 80)
    print()

    for param_name in param_names:
        print(f"Parameter: {param_name} (true value: ", end="")
        if param_name == "m":
            print(f"{m_true})")
        elif param_name == "b":
            print(f"{b_true})")
        else:
            print(f"{np.log(sigma_true):.4f})")
        print("-" * 80)

        rc = results_correct[param_name]
        rb = results_buggy[param_name]

        print(f"  Posterior Mean:")
        print(f"    Correct HMC: {rc['mean']:8.4f}  (std: {rc['std']:.4f})")
        print(f"    Buggy HMC:   {rb['mean']:8.4f}  (std: {rb['std']:.4f})")
        print()

        print(f"  Autocorrelation:")
        print(f"    Lag 1:  Correct={rc['autocorr_lag1']:.4f}  Buggy={rb['autocorr_lag1']:.4f}  Δ={rb['autocorr_lag1']-rc['autocorr_lag1']:+.4f}")
        print(f"    Lag 5:  Correct={rc['autocorr_lag5']:.4f}  Buggy={rb['autocorr_lag5']:.4f}  Δ={rb['autocorr_lag5']-rc['autocorr_lag5']:+.4f}")
        print(f"    Lag 10: Correct={rc['autocorr_lag10']:.4f}  Buggy={rb['autocorr_lag10']:.4f}  Δ={rb['autocorr_lag10']-rc['autocorr_lag10']:+.4f}")
        print()

        print(f"  Effective Sample Size (ESS):")
        print(f"    Correct HMC: {rc['ess']:6.1f} / {n_samples} ({rc['ess_percent']:5.1f}%)")
        print(f"    Buggy HMC:   {rb['ess']:6.1f} / {n_samples} ({rb['ess_percent']:5.1f}%)")

        if rb['ess'] < rc['ess'] * 0.5:
            degradation = 100 * (1 - rb['ess'] / rc['ess'])
            print(f"    ⚠️  Buggy version has {degradation:.1f}% LOWER ESS!")
        print()

    # Overall conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    # Compute averages
    avg_autocorr_correct = np.mean([results_correct[p]['autocorr_lag1'] for p in param_names])
    avg_autocorr_buggy = np.mean([results_buggy[p]['autocorr_lag1'] for p in param_names])
    avg_ess_correct = np.mean([results_correct[p]['ess_percent'] for p in param_names])
    avg_ess_buggy = np.mean([results_buggy[p]['ess_percent'] for p in param_names])

    print(f"Average lag-1 autocorrelation:")
    print(f"  Correct HMC: {avg_autocorr_correct:.4f}")
    print(f"  Buggy HMC:   {avg_autocorr_buggy:.4f}")
    print(f"  Difference:  {avg_autocorr_buggy - avg_autocorr_correct:+.4f}")
    print()

    print(f"Average effective sample size:")
    print(f"  Correct HMC: {avg_ess_correct:.1f}%")
    print(f"  Buggy HMC:   {avg_ess_buggy:.1f}%")
    print(f"  Difference:  {avg_ess_buggy - avg_ess_correct:+.1f}%")
    print()

    # Verdict
    if avg_autocorr_buggy > avg_autocorr_correct * 1.5 or avg_ess_buggy < avg_ess_correct * 0.5:
        print("✅ HYPOTHESIS CONFIRMED!")
        print()
        print("The buggy implementation (improper key handling) shows significantly")
        print("worse MCMC diagnostics:")
        if avg_autocorr_buggy > avg_autocorr_correct * 1.5:
            print(f"  • {(avg_autocorr_buggy/avg_autocorr_correct - 1)*100:.0f}% HIGHER autocorrelation")
        if avg_ess_buggy < avg_ess_correct * 0.5:
            print(f"  • {(1 - avg_ess_buggy/avg_ess_correct)*100:.0f}% LOWER effective sample size")
        print()
        print("KEY LESSON:")
        print("Proper randomness management is ESSENTIAL for MCMC algorithms to work.")
        print("You can't just 'fix a seed and forget it' - you must understand and")
        print("implement randomness correctly, or your inference will be broken.")
        print()
        print("IMPLICATION FOR DATA SCIENCE:")
        print("If randomness is so critical that core statistical algorithms break")
        print("without proper handling, why do we blindly fix seeds everywhere else?")
        print("Randomness is a FEATURE, not a bug!")
    else:
        print("⚠️  INCONCLUSIVE or UNEXPECTED RESULT")
        print()
        print("The buggy implementation doesn't show significantly worse performance.")
        print("This may indicate:")
        print("  • The bug isn't severe enough with these parameters")
        print("  • Need more samples to see the effect")
        print("  • The correct implementation also has issues")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
