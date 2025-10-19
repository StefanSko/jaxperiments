#!/usr/bin/env -S uv run
# /// script
# dependencies = ["jax", "jaxlib", "matplotlib", "numpy"]
# ///

# ABOUTME: Standalone script demonstrating HMC sampling with fixed vs random seeds.
# ABOUTME: Compares autocorrelation effects between the two seeding strategies.

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path to import hmc package
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from hmc.sampler import hmc_sample, log_posterior
from hmc.utils import generate_regression_data, get_hmc_config, plot_trace


def main():
    """Run HMC sampling comparison between fixed and random seeds."""
    parser = argparse.ArgumentParser(
        description="Compare HMC sampling with fixed vs random seeds"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1500,
        help="Number of data points for regression (default: 1500)",
    )
    parser.add_argument(
        "--m_true", type=float, default=2.5, help="True slope (default: 2.5)"
    )
    parser.add_argument(
        "--b_true", type=float, default=1.0, help="True intercept (default: 1.0)"
    )
    parser.add_argument(
        "--sigma_true",
        type=float,
        default=0.5,
        help="True noise std dev (default: 0.5)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hmc_comparison.png",
        help="Output path for comparison plot (default: hmc_comparison.png)",
    )
    parser.add_argument(
        "--n_hmc_samples",
        type=int,
        default=1000,
        help="Number of HMC samples to draw (default: 1000)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.001, help="HMC step size (default: 0.001)"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=15,
        help="Number of leapfrog steps (default: 15)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HMC Sampling Comparison: Fixed vs Random Seed")
    print("=" * 60)

    # Generate synthetic regression data
    print("\n[1/5] Generating synthetic regression data...")
    data_key = random.PRNGKey(42)
    x, y = generate_regression_data(
        data_key, args.n_samples, args.m_true, args.b_true, args.sigma_true
    )
    print(f"  Generated {args.n_samples} data points")
    print(f"  True parameters: m={args.m_true}, b={args.b_true}, σ={args.sigma_true}")

    # Get HMC configuration
    config = get_hmc_config(
        epsilon=args.epsilon, n_steps=args.n_steps, n_samples=args.n_hmc_samples
    )
    print(f"\n[2/5] HMC Configuration:")
    print(f"  Step size (ε): {config['epsilon']}")
    print(f"  Leapfrog steps: {config['n_steps']}")
    print(f"  Samples to draw: {config['n_samples']}")

    # Define log probability function with data
    def log_prob_fn(params):
        return log_posterior(params, x, y)

    # Initial parameters
    initial_params = {"m": 0.0, "b": 0.0, "log_sigma": 0.0}

    # Run HMC with fixed seed
    print("\n[3/5] Running HMC with FIXED seed (key=0)...")
    fixed_key = random.PRNGKey(0)
    samples_fixed = hmc_sample(
        fixed_key,
        initial_params,
        config["n_samples"],
        config["epsilon"],
        config["n_steps"],
        log_prob_fn,
    )
    print(f"  Completed {config['n_samples']} samples")
    print(f"  Final m: {samples_fixed['m'][-1]:.4f}")
    print(f"  Final b: {samples_fixed['b'][-1]:.4f}")

    # Run HMC with random seed (based on current time)
    print("\n[4/5] Running HMC with RANDOM seed...")
    random_seed = int(time.time() * 1000) % (2**32)
    random_key = random.PRNGKey(random_seed)
    samples_random = hmc_sample(
        random_key,
        initial_params,
        config["n_samples"],
        config["epsilon"],
        config["n_steps"],
        log_prob_fn,
    )
    print(f"  Completed {config['n_samples']} samples (seed={random_seed})")
    print(f"  Final m: {samples_random['m'][-1]:.4f}")
    print(f"  Final b: {samples_random['b'][-1]:.4f}")

    # Create comparison plots
    print("\n[5/5] Generating comparison plots...")
    param_names = ["m", "b", "log_sigma"]
    n_params = len(param_names)

    # Create figure with 2 rows (fixed seed top, random seed bottom)
    fig, axes = plt.subplots(2, n_params, figsize=(15, 8))

    # Plot fixed seed traces (top row)
    for i, param_name in enumerate(param_names):
        axes[0, i].plot(samples_fixed[param_name], alpha=0.7, linewidth=1)
        axes[0, i].set_ylabel(param_name)
        axes[0, i].set_xlabel("Iteration")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_title(f"{param_name} (Fixed Seed)")

    # Plot random seed traces (bottom row)
    for i, param_name in enumerate(param_names):
        axes[1, i].plot(samples_random[param_name], alpha=0.7, linewidth=1)
        axes[1, i].set_ylabel(param_name)
        axes[1, i].set_xlabel("Iteration")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_title(f"{param_name} (Random Seed)")

    # Add overall title
    fig.suptitle("HMC Trace Plots: Fixed vs Random Seed Comparison", fontsize=14, y=0.995)
    fig.tight_layout()

    # Save figure
    fig.savefig(args.output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved comparison plot to: {args.output_path}")
    print(f"  Fixed seed samples: {len(samples_fixed['m'])} points")
    print(f"  Random seed samples: {len(samples_random['m'])} points")

    print("\n" + "=" * 60)
    print("Sampling complete!")
    print(f"View the results at: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
