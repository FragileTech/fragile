# Benchmark Selector Dashboard

The `benchmarks.py` module now includes an interactive dashboard for exploring and visualizing benchmark potential functions.

## Features

### 1. **BenchmarkSelector** - Interactive Dashboard

A Panel-based dashboard for selecting and configuring benchmark functions with real-time visualization.

**Available Benchmarks:**
- **Sphere** - Simple quadratic potential
- **Rastrigin** - Highly multimodal with many local minima
- **EggHolder** - Complex landscape with deep global minimum
- **Styblinski-Tang** - Multimodal with many local minima
- **Holder Table** - Multiple global minima
- **Easom** - Flat landscape with single sharp minimum
- **Mixture of Gaussians** - Configurable multimodal distribution
- **Lennard-Jones** - Molecular cluster potential
- **Constant (Zero)** - Flat potential for testing pure diffusion

**Features:**
- Real-time potential landscape visualization
- Configurable spatial dimension (2D-10D)
- Adjustable bounds and resolution
- Benchmark-specific parameters:
  - **Mixture of Gaussians**: Number of modes, random seed
  - **Lennard-Jones**: Number of atoms

### 2. **Helper Functions**

#### `create_benchmark_background(benchmark, bounds_range, resolution)`
Generate HoloViews background visualization for any benchmark function.

#### `prepare_benchmark_for_explorer(benchmark_name, dims, bounds_range, **kwargs)`
One-stop function to prepare a benchmark for use with SwarmExplorer.

**Returns:** `(benchmark, background, mode_points)` tuple ready for SwarmExplorer.

### 3. **Constant Potential**

New `Constant` benchmark class that returns U(x) = 0 everywhere, useful for:
- Testing pure diffusion dynamics
- Baseline comparisons
- Debugging cloning mechanisms without potential forces

## Usage Examples

### Standalone Benchmark Selector

```python
import holoviews as hv
import panel as pn
hv.extension('bokeh')
pn.extension()

from fragile.core.benchmarks import BenchmarkSelector

# Create and launch dashboard
selector = BenchmarkSelector()
dashboard = selector.panel()
dashboard.show()

# Access current benchmark
benchmark = selector.get_benchmark()
background = selector.get_background()
mode_points = selector.get_mode_points()
```

### Integration with SwarmExplorer

```python
from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.interactive_euclidean_gas import SwarmExplorer

# Prepare benchmark
benchmark, background, mode_points = prepare_benchmark_for_explorer(
    'Rastrigin',
    dims=2,
    bounds_range=(-5.12, 5.12),
)

# Create explorer
explorer = SwarmExplorer(
    potential=benchmark,
    background=background,
    mode_points=mode_points,
    dims=2,
    N=100,
    n_steps=100,
)

# Launch dashboard
explorer.panel().show()
```

### Mixture of Gaussians Configuration

```python
# Custom Mixture of Gaussians
benchmark, background, mode_points = prepare_benchmark_for_explorer(
    'Mixture of Gaussians',
    dims=2,
    n_gaussians=5,
    seed=42,
    bounds_range=(-8.0, 8.0),
)

# Access mixture info
info = benchmark.get_component_info()
print(f"Centers: {info['centers']}")
print(f"Weights: {info['weights']}")
print(f"Stds: {info['stds']}")
```

### Constant Potential (Pure Diffusion)

```python
# Test pure diffusion without potential forces
benchmark, background, mode_points = prepare_benchmark_for_explorer(
    'Constant (Zero)',
    dims=2,
)

explorer = SwarmExplorer(
    potential=benchmark,
    background=background,
    mode_points=mode_points,
    dims=2,
    use_potential_force=False,  # No forces
    enable_kinetic=True,          # Only diffusion
)
```

## Demo Script

Run the included demo script to explore different modes:

```bash
# Standalone benchmark selector
python examples/benchmark_selector_demo.py selector

# SwarmExplorer with benchmark selection
python examples/benchmark_selector_demo.py explorer

# Side-by-side benchmark comparison
python examples/benchmark_selector_demo.py compare
```

## API Reference

### BenchmarkSelector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark_name` | str | "Mixture of Gaussians" | Selected benchmark |
| `dims` | int | 2 | Spatial dimension (2-10) |
| `bounds_extent` | float | 6.0 | Spatial bounds ±extent |
| `resolution` | int | 200 | Background grid resolution |
| `n_gaussians` | int | 3 | Number of Gaussian modes (MoG) |
| `seed` | int | 42 | Random seed (MoG) |
| `n_atoms` | int | 10 | Number of atoms (Lennard-Jones) |

### BenchmarkSelector Methods

- `get_benchmark()` → Get current benchmark instance
- `get_background()` → Get background visualization
- `get_mode_points()` → Get mode point markers
- `panel()` → Create Panel dashboard

## Visualization Guidelines

All visualizations follow the HoloViz stack conventions:
- **2D plots**: HoloViews with Bokeh backend
- **Interactive controls**: Panel widgets
- **Reactive updates**: Automatic refresh on parameter changes

## Integration with Existing Code

The new dashboard is fully compatible with existing code:

```python
# Old way (still works)
from fragile.experiments.convergence_analysis import create_multimodal_potential

potential, target_mixture = create_multimodal_potential(dims=2, n_gaussians=3)

# New way (with visualization)
from fragile.core.benchmarks import prepare_benchmark_for_explorer

benchmark, background, mode_points = prepare_benchmark_for_explorer(
    'Mixture of Gaussians',
    dims=2,
    n_gaussians=3,
)

# benchmark is compatible with existing potential interface
```

## Notes

- **2D visualization only**: Background density plots are only generated for 2D benchmarks
- **Higher dimensions**: For dims > 2, the dashboard shows a placeholder
- **Performance**: Reduce `resolution` for faster updates with high-resolution grids
- **Benchmark bounds**: Some benchmarks have fixed bounds (e.g., EggHolder: [-512, 512])
