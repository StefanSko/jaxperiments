# JAXperiments TDD Implementation Plan

## High-Level Blueprint

### Phase 1: Project Setup & Infrastructure
- Initialize UV project with dependencies
- Set up directory structure
- Configure testing framework

### Phase 2: Core Mathematical Components (TDD)
- Implement log probability functions (prior + likelihood)
- Implement gradient computation with JAX autodiff
- Verify gradients against numerical approximations

### Phase 3: Leapfrog Integrator (TDD)
- Implement single leapfrog step
- Test energy conservation properties
- Implement full leapfrog trajectory

### Phase 4: HMC Sampler Core (TDD)
- Implement single HMC step (propose + accept/reject)
- Implement full HMC chain sampling
- Test basic sampling properties

### Phase 5: Integration & Utilities
- Data generation utilities
- Parameter extraction/processing
- Configuration management

### Phase 6: Demonstration Components
- Command-line script
- Jupyter notebook
- Visualization functions

---

## Detailed Step-by-Step Prompts

### Step 1: Project Initialization

**Context**: Starting fresh with UV project setup for JAX-based HMC sampler.

**Prompt**:
```
Initialize a new UV Python project for the JAXperiments HMC sampler. Create the following:

1. Run `uv init` to create pyproject.toml
2. Add dependencies: jax, jaxlib, numpy, matplotlib, pytest, jupyter
3. Create the directory structure:
   - hmc/ (with __init__.py)
   - tests/ (with __init__.py) 
   - notebooks/
   - scripts/

4. In hmc/__init__.py, add the ABOUTME comment explaining this is an HMC sampler package
5. In tests/__init__.py, add minimal ABOUTME comment

Verify the setup works by running `uv run python -c "import jax; print(jax.__version__)"`
```

**Expected outcome**: Working UV project with all dependencies installed and directory structure in place.

---

### Step 2: Test Infrastructure for Log Probability

**Context**: Before implementing the log probability function, write tests that define its expected behavior.

**Prompt**:
```
Using TDD, write tests for the log probability function in tests/test_hmc.py. The log probability should combine:
- Prior: Normal(0, 10) for m and b, Normal(0, 2) for log_sigma
- Likelihood: Linear regression y ~ Normal(m*x + b, exp(log_sigma))

Write test cases that verify:
1. test_log_prior_shape: Prior accepts parameters as a dict {"m": float, "b": float, "log_sigma": float} and returns a scalar
2. test_log_likelihood_shape: Likelihood accepts params dict, x array, y array and returns a scalar
3. test_log_posterior_shape: Posterior combines prior and likelihood correctly
4. test_log_prior_values: Check specific values (e.g., at params all zeros)

Don't implement the functions yet - just write the tests. They should import from hmc.sampler (which doesn't exist yet).

Use pytest and JAX numpy (jax.numpy as jnp).
```

**Expected outcome**: Test file with 4 failing tests that specify the interface for log probability functions.

---

### Step 3: Implement Log Probability Functions

**Context**: Now implement the functions to make the tests pass.

**Prompt**:
```
Implement the log probability functions in hmc/sampler.py to make the tests pass.

Create these functions:
1. log_prior(params): Takes dict with keys "m", "b", "log_sigma", returns log probability under weakly informative priors
2. log_likelihood(params, x, y): Linear regression likelihood  
3. log_posterior(params, x, y): Combines prior and likelihood

Use jax.scipy.stats.norm.logpdf for the Normal distributions.
Add the ABOUTME comment at the top explaining this file contains HMC sampling functions.

Run the tests with `uv run pytest tests/test_hmc.py -v` and verify they all pass.
```

**Expected outcome**: All 4 tests passing. Core probability functions working correctly.

---

### Step 4: Test Infrastructure for Gradients

**Context**: Write tests that verify gradient computation against numerical gradients.

**Prompt**:
```
Add gradient tests to tests/test_hmc.py. We need to verify that JAX autodiff produces correct gradients.

Write these test cases:
1. test_grad_log_posterior_exists: Verify we can compute gradients of log_posterior
2. test_grad_log_posterior_shape: Gradient should return dict with same keys as params
3. test_grad_log_posterior_numerical: Compare autodiff gradient to numerical gradient
   - Use jax.grad to get autodiff gradient
   - Implement simple finite differences for numerical gradient (epsilon=1e-5)
   - Check they match within tolerance (1e-3)
   - Test at a few different parameter values

Don't implement grad_log_posterior yet - just the tests.
```

**Expected outcome**: 3 new failing tests for gradient computation.

---

### Step 5: Implement Gradient Function

**Context**: Implement gradient computation using JAX autodiff.

**Prompt**:
```
Add gradient computation to hmc/sampler.py to make the gradient tests pass.

Implement:
1. grad_log_posterior(params, x, y): Returns gradients of log_posterior with respect to params
   - Use jax.grad to differentiate log_posterior
   - Handle the dict structure properly (JAX supports this natively)

Add a helper for numerical gradients in tests (for comparison):
2. In tests/test_hmc.py, implement numerical_gradient(f, params, epsilon=1e-5)

Run `uv run pytest tests/test_hmc.py -v` and ensure all tests pass.
```

**Expected outcome**: All gradient tests passing. Autodiff working correctly for dictionary parameters.

---

### Step 6: Test Infrastructure for Leapfrog Step

**Context**: The leapfrog integrator is critical for HMC. Write tests for a single leapfrog step.

**Prompt**:
```
Add leapfrog integrator tests to tests/test_hmc.py.

The leapfrog step should:
- Take current position (q), momentum (p), step size (epsilon), gradient function
- Return new position and momentum after one step
- Use the standard leapfrog update equations

Write these tests:
1. test_leapfrog_step_shapes: Verify output shapes match input shapes
2. test_leapfrog_step_deterministic: Same inputs produce same outputs
3. test_leapfrog_step_modifies_position: Position should change after step
4. test_leapfrog_step_modifies_momentum: Momentum should change after step

Don't implement leapfrog_step yet.
```

**Expected outcome**: 4 new failing tests for leapfrog integrator.

---

### Step 7: Implement Single Leapfrog Step

**Context**: Implement the leapfrog step to make tests pass.

**Prompt**:
```
Implement the leapfrog step in hmc/sampler.py.

Add function:
- leapfrog_step(q, p, epsilon, grad_log_prob_fn): 
  - q and p are dicts with same structure as params
  - Implements standard leapfrog equations:
    - Half step for momentum: p = p + (epsilon/2) * grad_log_prob(q)
    - Full step for position: q = q + epsilon * p  
    - Half step for momentum: p = p + (epsilon/2) * grad_log_prob(q)
  - Returns (q_new, p_new)

Use JAX operations to ensure everything is differentiable.

Run `uv run pytest tests/test_hmc.py -v` to verify.
```

**Expected outcome**: Leapfrog step tests passing.

---

### Step 8: Test Leapfrog Energy Conservation

**Context**: A key property of leapfrog is approximate energy conservation. Test this.

**Prompt**:
```
Add an energy conservation test to tests/test_hmc.py.

The Hamiltonian (energy) is: H(q, p) = -log_posterior(q) + 0.5 * ||p||^2

Write test:
- test_leapfrog_energy_conservation:
  - Compute initial energy
  - Run multiple leapfrog steps (e.g., 10 steps)
  - Compute final energy
  - Verify energy is approximately conserved (within tolerance based on step size)
  - Use a simple problem (quadratic) for testing

Implement a helper:
- hamiltonian(q, p, log_prob_fn): Computes total energy

Run tests to see this fail (we'll need to refine leapfrog if needed).
```

**Expected outcome**: Energy conservation test in place (may need adjustment of tolerances).

---

### Step 9: Implement Full Leapfrog Trajectory

**Context**: Extend to multiple leapfrog steps.

**Prompt**:
```
Add a function for full leapfrog trajectory in hmc/sampler.py:

- leapfrog(q, p, epsilon, n_steps, grad_log_prob_fn):
  - Takes initial q, p
  - Applies leapfrog_step n_steps times
  - Returns final (q, p)

Add test in tests/test_hmc.py:
- test_leapfrog_trajectory: Verify n_steps iterations produces expected behavior

Run `uv run pytest tests/test_hmc.py -v`.
```

**Expected outcome**: Full leapfrog trajectory working, all tests passing.

---

### Step 10: Test Infrastructure for HMC Step

**Context**: Now build the HMC sampling step (propose + Metropolis accept/reject).

**Prompt**:
```
Add tests for a single HMC step in tests/test_hmc.py.

An HMC step should:
- Sample momentum from standard normal
- Run leapfrog trajectory  
- Compute acceptance probability via Metropolis
- Accept or reject proposal

Write tests:
1. test_hmc_step_returns_state: Returns new position and acceptance info
2. test_hmc_step_accepts_better_states: High probability states should be accepted
3. test_hmc_step_rejects_worse_states: Low probability states should be rejected with some frequency
4. test_hmc_step_uses_random_key: Different JAX random keys give different results

Don't implement yet. HMC step should take a JAX random key.
```

**Expected outcome**: 4 failing tests for HMC step.

---

### Step 11: Implement HMC Step

**Context**: Implement single HMC step with Metropolis acceptance.

**Prompt**:
```
Implement HMC step in hmc/sampler.py:

- hmc_step(key, q, epsilon, n_steps, log_prob_fn):
  - Split key for momentum sampling
  - Sample p ~ Normal(0, I)
  - Compute initial energy
  - Run leapfrog to get proposal
  - Compute proposal energy
  - Metropolis acceptance: accept if rand < exp(H_current - H_proposal)
  - Return (q_new, accepted) where accepted is boolean

Use jax.random for key splitting and sampling.

Run `uv run pytest tests/test_hmc.py -v`.
```

**Expected outcome**: HMC step tests passing.

---

### Step 12: Test Full HMC Chain

**Context**: Test sampling a full chain of HMC samples.

**Prompt**:
```
Add tests for full chain sampling in tests/test_hmc.py:

- hmc_sample(key, initial_q, n_samples, epsilon, n_steps, log_prob_fn):
  - Returns array of samples

Tests:
1. test_hmc_sample_shape: Verify output shape is (n_samples, n_params)
2. test_hmc_sample_different_seeds: Different keys give different samples
3. test_hmc_sample_same_seed: Same key gives identical samples
4. test_hmc_sample_simple_posterior: For a simple known distribution, verify samples are reasonable

Don't implement yet.
```

**Expected outcome**: 4 failing tests for chain sampling.

---

### Step 13: Implement HMC Chain Sampling

**Context**: Implement the full sampling loop.

**Prompt**:
```
Implement full chain sampling in hmc/sampler.py:

- hmc_sample(key, initial_q, n_samples, epsilon, n_steps, log_prob_fn):
  - Use jax.lax.scan for efficient looping
  - Collect all samples
  - Return as dictionary of arrays

Add helper:
- samples_to_array(samples_dict): Convert dict of arrays to single array
- array_to_samples(array, param_names): Convert back

Run `uv run pytest tests/test_hmc.py -v`.
```

**Expected outcome**: All HMC chain tests passing. Core sampler complete.

---

### Step 14: Data Generation Utilities

**Context**: Create utilities for generating synthetic regression data.

**Prompt**:
```
Create hmc/utils.py with data generation functions.

Implement:
- generate_regression_data(key, n_samples, m_true, b_true, sigma_true, x_range=(0, 10)):
  - Generates x uniformly in x_range
  - Generates y = m_true * x + b_true + noise
  - Returns (x, y)

Add ABOUTME comment.

Write tests in tests/test_utils.py:
1. test_generate_regression_data_shape
2. test_generate_regression_data_reproducible: Same key gives same data
3. test_generate_regression_data_different_seeds: Different keys give different data

Run tests.
```

**Expected outcome**: Data generation utilities working with tests.

---

### Step 15: Configuration and Defaults

**Context**: Set up default HMC hyperparameters.

**Prompt**:
```
Add configuration to hmc/utils.py:

Create:
- DEFAULT_HMC_CONFIG: Dict with reasonable defaults
  - epsilon: 0.01
  - n_steps: 20  
  - n_warmup: 500
  - n_samples: 1000

- get_hmc_config(**overrides): Returns config dict with overrides applied

Add tests in tests/test_utils.py:
1. test_default_config_values
2. test_config_overrides

Run tests.
```

**Expected outcome**: Configuration management in place.

---

### Step 16: Integration Test - Full Pipeline

**Context**: Test the complete sampling pipeline on synthetic data.

**Prompt**:
```
Add integration test in tests/test_integration.py:

- test_full_regression_sampling:
  - Generate synthetic data with known parameters
  - Run HMC sampler
  - Verify samples have reasonable properties:
    - Mean close to true values
    - Samples vary (not all identical)
    - Chain explores parameter space

This test brings together all components.

Run `uv run pytest tests/test_integration.py -v`.
```

**Expected outcome**: Full pipeline working end-to-end.

---

### Step 17: Visualization Utilities

**Context**: Create plotting functions for trace plots.

**Prompt**:
```
Add visualization to hmc/utils.py:

Implement:
- plot_trace(samples, param_names, title="Trace Plot"):
  - Creates subplot for each parameter
  - Plots sample values vs iteration
  - Returns matplotlib figure

Add test in tests/test_utils.py:
- test_plot_trace_creates_figure: Verify figure is created with correct subplots

Run tests.
```

**Expected outcome**: Trace plotting functionality working.

---

### Step 18: Command-Line Script - Part 1 (Core Logic)

**Context**: Build the standalone script that demonstrates fixed vs random seed.

**Prompt**:
```
Create scripts/run_comparison.py as a UV script with PEP 723 metadata.

Add shebang and metadata:
```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["jax", "jaxlib", "matplotlib", "numpy"]
# ///
```

Implement:
1. main() function that:
   - Generates synthetic data
   - Runs HMC with fixed seed (key = 0)
   - Runs HMC with random seed (key = current time or random)
   - Calls plotting function (to be wired next)

2. Argument parsing for:
   - n_samples
   - m_true, b_true, sigma_true
   - output_path

Don't add plotting yet - just print confirmation messages.

Test by running: `./scripts/run_comparison.py`
```

**Expected outcome**: Executable script that runs sampling (no plots yet).

---

### Step 19: Command-Line Script - Part 2 (Plotting & Output)

**Context**: Wire in the visualization and save plots.

**Prompt**:
```
Update scripts/run_comparison.py to:

1. Import plot_trace from hmc.utils
2. Create comparison plot:
   - 2 rows of subplots
   - Row 1: Fixed seed traces
   - Row 2: Random seed traces
3. Save figure to output_path (default: "hmc_comparison.png")
4. Add proper labels and titles

Test by running: `./scripts/run_comparison.py --output_path test_output.png`

Verify the output image exists and shows clear differences.
```

**Expected outcome**: Complete working script that generates comparison plots.

---

### Step 20: Jupyter Notebook - Setup and Data

**Context**: Create interactive notebook for demonstration.

**Prompt**:
```
Create notebooks/demonstration.ipynb with these cells:

1. Title and introduction markdown
2. Imports cell
3. Generate synthetic data cell with visualization of x,y scatter
4. Display true parameters

Keep it simple for now - just data generation and basic plotting.

Test by running: `uv run jupyter notebook` and executing cells.
```

**Expected outcome**: Notebook with data generation working.

---

### Step 21: Jupyter Notebook - HMC Sampling & Research Validation

**Context**: Add HMC sampling to notebook and validate the research hypothesis that fixed seeds break MCMC sampling.

**Research Question**: Does using a fixed random seed in HMC sampling introduce problematic autocorrelation and poor mixing that defeats the purpose of MCMC sampling? This demonstrates why blindly fixing seeds for "reproducibility" can be harmful in stochastic algorithms.

**Prompt**:
```
Add to notebooks/demonstration.ipynb:

5. Cell: Run HMC with fixed seed
   - Set config
   - Run sampler
   - Print acceptance rate

6. Cell: Run HMC with random seed
   - Same config
   - Run sampler
   - Print acceptance rate

7. Cell: Plot both trace plots side by side

8. Cell: Research Validation - Compute and compare:
   - Autocorrelation for each parameter (lag 1, 5, 10)
   - Effective sample size (ESS) using standard formula
   - Sample variance/standard deviation
   - Visual comparison showing fixed seed has higher autocorrelation
   - Statistical evidence that fixed seed produces inferior samples

9. Cell: Interpretation
   - Explain findings
   - Discuss implications for reproducibility vs. correctness
   - Highlight why MCMC requires proper randomness

Test the notebook end-to-end and verify that fixed seed shows measurably worse autocorrelation.
```

**Expected outcome**: Complete notebook showing the autocorrelation effect WITH quantitative validation that fixed seed breaks MCMC sampling.

---

### Step 22: Documentation and README

**Context**: Add minimal project documentation.

**Prompt**:
```
Create README.md with:

1. Project title and description
2. Installation instructions using UV
3. Usage examples:
   - Running tests
   - Running command-line script
   - Opening notebook
4. Brief explanation of what the project demonstrates

Keep it concise and practical.

No need to test this - just create the file.
```

**Expected outcome**: Complete README for the project.

---

### Step 23: Final Integration and Cleanup

**Context**: Ensure everything works together and clean up any issues.

**Prompt**:
```
Final checks:

1. Run full test suite: `uv run pytest -v`
2. Run the command-line script: `./scripts/run_comparison.py`
3. Execute the notebook end-to-end
4. Verify all imports work correctly
5. Check that all files have ABOUTME comments where required
6. Remove any unused imports or dead code

Fix any issues that arise.

Commit all changes with message: "feat: complete HMC sampler implementation"
```

**Expected outcome**: Fully working project with all components integrated and tested.

---

## Summary

This plan breaks down the HMC sampler into 23 incremental steps:
- Steps 1-5: Project setup and probability functions
- Steps 6-9: Leapfrog integrator
- Steps 10-13: HMC sampling core
- Steps 14-16: Utilities and integration
- Steps 17-19: Command-line interface
- Steps 20-21: Jupyter notebook with research validation
- Steps 22-23: Documentation and polish

Each step is small enough to implement safely with tests, but substantial enough to make meaningful progress. Every step builds on previous work with no orphaned code.

## Research Validation

**Core Hypothesis**: Using a fixed random seed in MCMC sampling (specifically HMC) produces samples with higher autocorrelation and poorer mixing compared to proper random seeding. This demonstrates why blindly fixing random seeds for "reproducibility" can produce scientifically invalid results in stochastic algorithms.

**Validation Metrics**:
- Autocorrelation function at various lags
- Effective Sample Size (ESS)
- Sample variance and exploration
- Visual trace plot comparison

**Expected Finding**: Fixed seed samples will show significantly higher autocorrelation and lower effective sample size, demonstrating that the fixed seed breaks the MCMC algorithm's ability to properly explore the posterior distribution.
