# Experiment Scripts

This folder contains standalone Python scripts for running experiments and debugging computational logic **without** requiring Jupyter notebooks or visualization.

## Purpose

These scripts allow you to:
- Test computational logic in the terminal
- Debug numerical issues with print statements
- Run experiments on headless servers (no display required)
- Validate results before creating visualizations
- Profile performance and optimize code

## Available Scripts

### `run_convergence_experiment.py`
Full convergence experiment for Geometric Gas with 5000 steps and detailed analysis.

**Usage:**
```bash
cd experiments/experiment_scripts
python run_convergence_experiment.py
```

**Output:**
- Convergence metrics (KL-divergence, Wasserstein-2, Lyapunov function)
- Exponential fit parameters (κ, C, half-life)
- Final statistics and validation
- Snapshot positions at key time points

**Expected runtime:** ~2 minutes (100 walkers, 5000 steps)

### `test_convergence_quick.py`
Quick test with reduced parameters for faster debugging.

**Usage:**
```bash
python test_convergence_quick.py
```

**Output:**
- Same metrics as full experiment
- Faster execution (50 walkers, 500 steps)

**Expected runtime:** ~15 seconds

## Implementation Details

All computational logic is implemented in `src/fragile/experiments/convergence_analysis.py`:

- `create_multimodal_potential()` - Factory for test potentials
- `MixtureBasedPotential` - Wrapper for MixtureOfGaussians
- `ConvergenceMetrics` - Container for time-series data
- `ConvergenceAnalyzer` - Computes KL-divergence, Wasserstein-2, Lyapunov
- `ConvergenceExperiment` - Orchestrates full experiments

The notebook `03_exponential_convergence_to_qsd.ipynb` imports these classes and focuses only on visualization.

## Workflow

1. **Develop/Debug**: Test computational logic using these scripts
2. **Validate**: Verify numerical results in terminal
3. **Visualize**: Import the same code in Jupyter for plots
4. **Iterate**: Make changes in `convergence_analysis.py`, rerun scripts

## Example Output

```
======================================================================
Convergence Experiment: Geometric Gas -> QSD
======================================================================

[1/5] Creating multimodal potential...
  ✓ Created 3-mode Gaussian mixture in 2D
  ✓ Mode centers: [[0. 0.], [4. 3.], [-3. 2.5]]
  ✓ Mode weights: [0.5, 0.3, 0.2]

[2/5] Initializing Geometric Gas...
  ✓ Created GeometricGas with 100 walkers
  ✓ ρ-localization scale: 2.0

[3/5] Initializing swarm state...
  ✓ Initial position range: [5.01, 7.00]

[4/5] Running convergence experiment...
  Step  100: KL=0.0003, W2=4.6307
  Step  200: KL=0.0006, W2=2.4896
  ...
  Step 5000: KL=0.0006, W2=4.5615

✓ Experiment complete!

[5/5] Analyzing convergence...

======================================================================
CONVERGENCE SUMMARY
======================================================================

KL-Divergence Convergence:
  Rate κ: 0.000060
  Half-life: 11583.18 steps
  Final KL: 0.000588
  Exponential fit: D_KL(t) ≈ 0.0004 * exp(-0.000060 * t)

✓ Exponential convergence detected!
```

## Dependencies

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- fragile (this package)

No matplotlib, seaborn, or Jupyter required!

## Notes

- Scripts use the same random seeds (42) as the notebook for reproducibility
- All parameters match the notebook configuration
- Terminal-friendly output (no plots, just numbers)
- Designed for WSL/headless environments
