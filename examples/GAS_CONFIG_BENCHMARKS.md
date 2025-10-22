# GasConfig with Integrated Benchmark Selector

The `GasConfig` dashboard now includes an integrated benchmark selector, eliminating the need to manually create potential functions.

## What's New

### Integrated Benchmark Selection

`GasConfig` now automatically creates potentials from a dropdown selection of 9 benchmark functions:
- No need to create potentials manually
- Dynamic switching between benchmarks
- Configurable benchmark-specific parameters
- Fully backward compatible

### Available Benchmarks

| Benchmark | Description | Configurable Parameters |
|-----------|-------------|------------------------|
| **Mixture of Gaussians** | Multimodal Gaussian mixture | n_gaussians, seed |
| **Sphere** | Simple quadratic potential | None |
| **Rastrigin** | Highly multimodal landscape | None |
| **EggHolder** | Complex with deep minimum | None |
| **Styblinski-Tang** | Multimodal with local minima | None |
| **Holder Table** | Multiple global minima | None |
| **Easom** | Flat with sharp minimum | None |
| **Lennard-Jones** | Molecular cluster potential | n_atoms |
| **Constant (Zero)** | Flat potential (pure diffusion) | None |

## New Workflow

### Before (Old Way)

```python
# Had to create potential manually
from fragile.experiments.convergence_analysis import create_multimodal_potential

potential, _ = create_multimodal_potential(dims=2, n_gaussians=3)
config = GasConfig(potential=potential, dims=2)
```

### After (New Way)

```python
# No manual potential creation needed!
config = GasConfig(dims=2)  # Uses default "Mixture of Gaussians"
config.benchmark_name = "Rastrigin"  # Switch benchmark dynamically
```

## Usage Examples

### 1. Interactive Dashboard

```python
import holoviews as hv
import panel as pn
hv.extension('bokeh')
pn.extension()

from fragile.experiments.gas_config_dashboard import GasConfig

# Create config without providing potential
config = GasConfig(dims=2)

# Launch interactive dashboard with benchmark selector
dashboard = config.panel()
dashboard.show()
```

The dashboard includes:
- **Potential Function** section (new) with benchmark dropdown
- Dynamic parameter controls for selected benchmark
- All existing simulation parameters
- Run button to execute simulation

### 2. Programmatic Configuration

```python
from fragile.experiments.gas_config_dashboard import GasConfig

# Create config
config = GasConfig(dims=2, N=100, n_steps=100)

# Select and configure benchmark
config.benchmark_name = "Mixture of Gaussians"
config.n_gaussians = 5
config.benchmark_seed = 42

# Run simulation
history = config.run_simulation()
print(f"Completed: {history.n_steps} steps")
```

### 3. Dynamic Benchmark Switching

```python
config = GasConfig(dims=2)

# Switch between benchmarks
for name in ["Sphere", "Rastrigin", "Constant (Zero)"]:
    config.benchmark_name = name
    history = config.run_simulation()
    print(f"{name}: {history.n_alive[-1].item()} walkers survived")
```

### 4. Backward Compatibility

Old code still works! You can still provide explicit potentials:

```python
from fragile.core.benchmarks import Sphere

# Explicit potential (old workflow)
sphere = Sphere(dims=2)
config = GasConfig(potential=sphere, dims=2)

# Still works perfectly!
history = config.run_simulation()
```

## Benchmark-Specific Configuration

### Mixture of Gaussians

Configure the number of Gaussian modes and random seed:

```python
config = GasConfig(dims=2)
config.benchmark_name = "Mixture of Gaussians"
config.n_gaussians = 7  # Number of modes
config.benchmark_seed = 123  # For reproducibility
```

### Lennard-Jones

Configure the number of atoms:

```python
config = GasConfig(dims=3*10)  # 3 dims per atom
config.benchmark_name = "Lennard-Jones"
config.n_atoms = 10
```

### Other Benchmarks

No additional configuration needed - just select and run:

```python
config = GasConfig(dims=2)
config.benchmark_name = "Rastrigin"
# Ready to run!
```

## UI Changes

The `panel()` method now returns an accordion with an additional section:

### Accordion Sections

1. **Potential Function** (NEW)
   - Benchmark dropdown selector
   - Dynamic parameters (e.g., n_gaussians for MoG)
   - Status display showing current benchmark

2. **General** (existing)
   - N, n_steps, enable_cloning, enable_kinetic

3. **Langevin Dynamics** (existing)
   - gamma, beta, delta_t, forces, diffusion

4. **Cloning & Selection** (existing)
   - Fitness parameters, companion selection

5. **Initialization** (existing)
   - init_offset, init_spread, bounds_extent

## Technical Details

### Automatic Wrapping

The `GasConfig` automatically wraps benchmark objects in `PotentialParams` for compatibility with `EuclideanGas`:

```python
# Internal wrapping (automatic)
class BenchmarkPotential(PotentialParams):
    benchmark_obj: object

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self.benchmark_obj(x)
```

This ensures:
- Compatibility with EuclideanGas Pydantic validation
- Consistent interface across all benchmarks
- No user intervention required

### Dynamic Updates

Benchmark parameters are watched and trigger automatic potential updates:

```python
# Changing benchmark_name updates the potential immediately
config.benchmark_name = "Rastrigin"

# Changing n_gaussians updates MixtureOfGaussians configuration
config.n_gaussians = 5

# Status display shows current configuration
```

## Demo Scripts

Try the demo scripts:

```bash
# Interactive dashboard
python examples/gas_config_with_benchmarks_demo.py basic

# Programmatic control
python examples/gas_config_with_benchmarks_demo.py programmatic

# Benchmark comparison
python examples/gas_config_with_benchmarks_demo.py comparison
```

## Migration Guide

### Migrating Existing Code

**Old code:**
```python
from fragile.experiments.convergence_analysis import create_multimodal_potential
potential, _ = create_multimodal_potential(dims=2, n_gaussians=3)
config = GasConfig(potential=potential, dims=2)
```

**New code (Option 1 - Use integrated selector):**
```python
config = GasConfig(dims=2)
config.benchmark_name = "Mixture of Gaussians"
config.n_gaussians = 3
```

**New code (Option 2 - Keep explicit potential):**
```python
# Old code continues to work unchanged!
from fragile.experiments.convergence_analysis import create_multimodal_potential
potential, _ = create_multimodal_potential(dims=2, n_gaussians=3)
config = GasConfig(potential=potential, dims=2)  # Still works
```

### Integration with SwarmExplorer

`SwarmExplorer` still requires explicit potential, background, and mode_points. Use `prepare_benchmark_for_explorer`:

```python
from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.interactive_euclidean_gas import SwarmExplorer

benchmark, background, mode_points = prepare_benchmark_for_explorer(
    'Rastrigin',
    dims=2,
)

explorer = SwarmExplorer(
    potential=benchmark,
    background=background,
    mode_points=mode_points,
    dims=2,
)
```

## API Reference

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark_name` | ObjectSelector | "Mixture of Gaussians" | Selected benchmark |
| `n_gaussians` | Integer | 3 | Number of Gaussian modes (MoG) |
| `benchmark_seed` | Integer | 42 | Random seed (MoG) |
| `n_atoms` | Integer | 10 | Number of atoms (Lennard-Jones) |

### Modified Signature

```python
def __init__(
    self,
    potential: object | None = None,  # Now optional!
    dims: int = 2,
    **params
)
```

### New Methods

- `_update_benchmark()` - Creates potential from benchmark parameters
- `_on_benchmark_change(*_)` - Handles benchmark parameter changes

## Benefits

1. **Simplified Workflow**: No manual potential creation
2. **Quick Experimentation**: Switch benchmarks with a dropdown
3. **Reproducibility**: Explicit seed control for stochastic benchmarks
4. **Backward Compatible**: Existing code continues to work
5. **Integrated UI**: All configuration in one place
6. **Type Safe**: Automatic PotentialParams wrapping

## Notes

- Benchmark selection updates potential immediately
- Changes to benchmark parameters trigger potential recreation
- All benchmarks are automatically wrapped in PotentialParams
- Backward compatibility maintained for explicit potentials
- Status pane shows current benchmark configuration
