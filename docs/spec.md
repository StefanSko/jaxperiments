# JAXperiments: HMC Sampler Specification

## Project Overview
Build a small HMC (Hamiltonian Monte Carlo) sampler with JAX for illustrative purposes. The goal is to sample parameters for a simple linear regression and demonstrate the negative effects of using a fixed random seed on the autocorrelation of samples drawn.

## Regression Problem
- **Model**: Linear regression y = mx + b
- **Dataset size**: 1,000-2,000 samples
- **Synthetic data**: Generated with configurable true parameters (m_true, b_true, σ_true)

## Parameters to Infer
1. Slope (m)
2. Intercept (b)
3. Log noise standard deviation (log(σ))

## Prior Distributions
All parameters use weakly informative normal priors:
- m ~ Normal(mean=0, std=10)
- b ~ Normal(mean=0, std=10)
- log(σ) ~ Normal(mean=0, std=2)

## HMC Configuration
Use reasonable defaults for:
- Number of leapfrog steps per iteration
- Step size (ε) for leapfrog integrator
- Number of warmup/burn-in samples
- Number of post-warmup samples to keep

Implementation should be moderately efficient using JAX.

## Demonstration
Compare two sampling runs:
1. One chain with proper random seeding
2. One chain with fixed random seed

Visualize the comparison using trace plots to show autocorrelation effects.

## Output Components

### 1. Core HMC Code
- Modular Python package structure
- HMC sampler implementation in JAX
- Utility functions

### 2. Jupyter Notebook
- Interactive demonstration
- Inline plots and explanations
- Shows fixed vs. random seed comparison

### 3. Command-Line Script
- Standalone UV script with shebang
- Executable from command line
- Saves comparison plots (trace plots)
- Uses PEP 723 inline script metadata

## Testing Strategy
Unit tests for core HMC algorithm (TDD approach):
1. Leapfrog integrator correctness (energy conservation)
2. Gradient calculations (vs. numerical gradients)

## Technical Stack
- **Language**: Python
- **Numerical framework**: JAX
- **Plotting**: Matplotlib
- **Package management**: UV
- **Testing**: pytest (via `uv run pytest`)

## Project Structure
```
jaxperiments/
├── hmc/
│   ├── __init__.py
│   ├── sampler.py
│   └── utils.py
├── tests/
│   └── test_hmc.py
├── notebooks/
│   └── demonstration.ipynb
├── scripts/
│   └── run_comparison.py
├── pyproject.toml
└── README.md
```

## Documentation
Minimal documentation approach:
- Brief ABOUTME comments at top of each file
- Docstrings for key functions only
- Code should be self-explanatory

## Deliverables
1. Working HMC sampler implemented in JAX
2. Unit tests for leapfrog integrator and gradients
3. Jupyter notebook with demonstration
4. Standalone UV script for command-line execution
5. Trace plots showing autocorrelation effects (fixed vs. random seed)
