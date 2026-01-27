# QFT Calibration Dashboard

## Overview

The fractal gas visualization dashboard now supports a QFT (Quantum Field Theory) calibration mode. This mode pre-configures the dashboard with parameters from the QFT calibration notebook (`08_qft_calibration_notebook.ipynb`), enabling visual inspection of walker behavior with the same physics used in QFT simulations.

## Key Features

### New Quadratic Well Benchmark

A new `QuadraticWell` benchmark has been added to simulate the harmonic potential used in QFT calibrations:

```
U(x) = 0.5 * alpha * ||x||^2
```

Properties:
- Global minimum at the origin: U(0) = 0
- Configurable curvature parameter `alpha` (default: 0.1)
- Configurable spatial bounds extent (default: 10.0)

### QFT Configuration Preset

The QFT mode uses calibrated parameters that differ significantly from the default multimodal exploration settings:

| Parameter | Default | QFT Calibration | Notes |
|-----------|---------|-----------------|-------|
| **Benchmark** | Mixture of Gaussians | Quadratic Well | Harmonic potential |
| **N (walkers)** | 160 | 200 | More particles for QFT simulation |
| **n_steps** | 240 | 5000 | Longer simulation for convergence |
| **dims** | 2 | 3 | Three-dimensional space |
| **bounds_extent** | 6.0 | 10.0 | Larger spatial domain |
| **delta_t** | 0.05 | 0.1005 | Calibrated timestep |
| **epsilon_F** | 0.15 | 38.6373 | Fitness force coupling |
| **nu** | 0.0 | 1.10 | Viscous coupling strength |
| **viscous_length_scale** | 1.0 | 0.251372 | Interaction length scale |
| **use_viscous_coupling** | False | True | Enable viscous forces |
| **viscous_neighbor_threshold** | None | 0.75 | Neighbor distance threshold |
| **viscous_neighbor_penalty** | 0.0 | 0.9 | Penalty for close neighbors |
| **companion_epsilon** | 0.5 | 2.80 | Diversity selection strength |
| **companion_epsilon_clone** | 0.5 | 1.68419 | Cloning selection strength |
| **fitness_rho** | None | 0.251372 | Fitness length scale |

## Usage

### Launch QFT Dashboard

```bash
python -m fragile.fractalai.experiments.gas_visualization_dashboard --qft
```

The dashboard will be available at http://localhost:5007

### Launch Standard Dashboard

```bash
python -m fragile.fractalai.experiments.gas_visualization_dashboard
```

### Programmatic Usage

```python
import holoviews as hv
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

# Initialize holoviews
hv.extension("bokeh")

# Create QFT configuration
config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)

# Access parameters
print(f"Benchmark: {config.benchmark_name}")
print(f"Walkers: {config.gas_params['N']}")
print(f"Steps: {config.n_steps}")

# Modify parameters if needed
config.kinetic_op.epsilon_F = 40.0
config.n_steps = 3000

# Run simulation
history = config.run_simulation()
```

## Dashboard Interface

When launched in QFT mode, the dashboard includes:

1. **Sidebar - Simulation Control**
   - QFT mode indicator and description
   - All operator parameters organized in accordions:
     - Benchmark selection (Quadratic Well selected)
     - Simulation settings (N, n_steps)
     - Langevin Dynamics parameters
     - Viscous Coupling controls
     - Companion Selection settings
     - Fitness Operator parameters
     - Cloning parameters
   - Run Simulation button

2. **Main Panel - Tabs**
   - **Evolution**: 3D walker trajectories in the quadratic well
   - **Convergence**: KL divergence and Lyapunov exponent analysis
   - **Diagnostics**: Parameter validation and theoretical bounds

## Expected Behavior

With QFT calibration parameters, the simulation should exhibit:

1. **Convergence to Gaussian**: Walkers converge to a Gaussian distribution centered at the origin
2. **Viscous Correlation**: Viscous coupling creates spatial correlation structure among nearby walkers
3. **Controlled Diffusion**: Large `epsilon_F` and `nu` values control the exploration-exploitation balance
4. **Long-term Stability**: 5000 steps allows observation of equilibration dynamics

## Verification

To verify the setup:

```bash
# Run unit tests
python test_qft_setup.py

# Check parameter values
python verify_qft_dashboard_params.py
```

Both scripts should show all parameters correctly configured and matching the QFT calibration values.

## Implementation Details

### New Components

1. **`QuadraticWell` class** (`src/fragile/fractalai/core/benchmarks.py`)
   - Harmonic potential benchmark
   - Accepts `alpha` and `bounds_extent` parameters
   - Added to `ALL_BENCHMARKS` and `BENCHMARK_NAMES`

2. **`GasConfigPanel.create_qft_config()` static method** (`src/fragile/fractalai/experiments/gas_config_panel.py`)
   - Factory method for QFT preset configuration
   - Sets all QFT-specific parameters
   - Configures separate companion selection for cloning

3. **`companion_selection_clone` attribute** (`src/fragile/fractalai/experiments/gas_config_panel.py`)
   - Separate `CompanionSelection` instance for cloning operations
   - Allows different epsilon values for diversity vs. cloning
   - Passed to `EuclideanGas` constructor

4. **`create_qft_app()` function** (`src/fragile/fractalai/experiments/gas_visualization_dashboard.py`)
   - Creates dashboard with QFT configuration
   - Identical structure to standard dashboard but with QFT defaults
   - Adds explanatory text in sidebar

5. **CLI flag handling** (`src/fragile/fractalai/experiments/gas_visualization_dashboard.py`)
   - `--qft` flag switches between standard and QFT mode
   - Clean conditional in `__main__` section

## Comparison with QFT Notebook

The dashboard parameters exactly match those in `08_qft_calibration_notebook.ipynb`:

```python
# From notebook cell defining QFT parameters
alpha = 0.1
bounds_extent = 10.0
N = 200
delta_t = 0.1005
epsilon_F = 38.6373
nu = 1.10
viscous_length_scale = 0.251372
viscous_neighbor_threshold = 0.75
viscous_neighbor_penalty = 0.9
epsilon_d = 2.80  # diversity
epsilon_clone = 1.68419  # cloning
fitness_rho = 0.251372
```

## Limitations and Future Work

1. **3D Visualization**: The dashboard's 3D visualization may need enhancements for optimal viewing
2. **Performance**: 5000 steps with 200 walkers may be slow; consider adding a "quick test" mode
3. **Visual Interest**: Quadratic well is less visually interesting than multimodal benchmarks
4. **Additional Presets**: Could add other physics-based presets (e.g., anharmonic potentials)

## References

- QFT Calibration Notebook: `docs/source/3_fractal_gas/2_fractal_set/08_qft_calibration_notebook.ipynb`
- Reference Implementation: `src/experiments/fractal_gas_potential_well.py`
- Dashboard Documentation: (link to main dashboard docs)
