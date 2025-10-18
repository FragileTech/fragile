# Fragile Shaolin Visualization Library - Complete Index

This directory exploration documents the comprehensive visualization library available in `src/fragile/shaolin/`.

## Quick Navigation

- **New to Shaolin?** Start with [SHAOLIN_SUMMARY.txt](SHAOLIN_SUMMARY.txt)
- **Need detailed reference?** Read [SHAOLIN_EXPLORATION.md](SHAOLIN_EXPLORATION.md)
- **Want code examples?** See [SHAOLIN_EXAMPLES.md](SHAOLIN_EXAMPLES.md)

## Three-Part Documentation

### 1. SHAOLIN_SUMMARY.txt (Quick Reference)
- **Length**: Concise, scannable
- **Content**: Components overview, patterns, best practices
- **Use When**: You need a quick lookup, starting point, or decision guide
- **Key Sections**:
  - Core components overview
  - Key patterns to reuse
  - Data format requirements
  - Import patterns
  - Best practices
  - Performance metrics
  - Example use cases

### 2. SHAOLIN_EXPLORATION.md (Detailed Reference)
- **Length**: Comprehensive (16 sections)
- **Content**: Deep dive into each component, architecture, data flow
- **Use When**: Understanding how things work, implementing complex features
- **Key Sections**:
  1. File structure and overview
  2. Streaming plot base classes
  3. All plot types (10+)
  4. Gas algorithm visualization
  5. Interactive DataFrame
  6. Dimension mapping widgets
  7. Parameter selectors
  8. Colormap utilities
  9. Control widgets
  10. Patterns and best practices
  11. Backend configuration
  12. Import patterns
  13. Stream types
  14. Common customizations
  15. Data format requirements
  16. Performance considerations

### 3. SHAOLIN_EXAMPLES.md (Code Examples)
- **Length**: 16 complete examples
- **Content**: Copy-paste ready code snippets
- **Use When**: Building visualizations, learning by example
- **Covers**:
  1. Basic setup
  2. Real-time scatter
  3. Position + velocity overlay
  4. Metric tracking
  5. Multi-metric dashboard
  6. Gas algorithm
  7. Boundary-aware viz
  8. Parameter configuration
  9. Interactive exploration
  10. Dimension mapping
  11. Colormap selection
  12. Algorithm runner
  13. Categorical scatter
  14. Histograms
  15. Energy landscapes
  16. Full dashboard example
  - Plus 7 common patterns
  - Plus troubleshooting guide

---

## Module-by-Module Guide

### Core Streaming Framework (`stream_plots.py` - 988 lines)

**Foundation Class**: `StreamingPlot`
- All other plots inherit from this
- Handles Bokeh/Plotly backend switching
- Manages Pipe (stateless) and Buffer (stateful) streams
- Implements `send(data)` for streaming updates

**Key Plot Classes**:
| Class | Type | Use Case |
|-------|------|----------|
| `Scatter` | Points | Walker positions |
| `VectorField` | Arrows | Velocities/forces |
| `Curve` | Line | Metrics over time |
| `Histogram` | Bars | Distributions |
| `Landscape2D` | Heatmap | Energy surface |
| `Table` | DataFrame | Data inspection |
| `Bivariate` | KDE | 2D densities |
| `QuadMesh` | Grid | Heatmaps |

**Reference**: [SHAOLIN_EXPLORATION.md - Section 2-3](SHAOLIN_EXPLORATION.md#2-streaming-plot-base-classes)

---

### Gas Algorithm Visualization (`gas_viz.py` - 330 lines)

**Main Classes**:
1. **`GasVisualization`** - Canonical Gas algorithm visualization
   - Real-time position scatter
   - Velocity vector field overlay
   - Alive/cloned walker tracking
   - Complete Panel layout

2. **`BoundaryGasVisualization`** - Extends with boundary checking
   - Color-coded in/out of bounds
   - Automatic detection
   - Only in-bounds velocity vectors

**Example**:
```python
from fragile.shaolin.gas_viz import GasVisualization
viz = GasVisualization(bounds=bounds, track_alive=True)
viz.update(state, n_alive=count)
layout = viz.create_layout()
```

**Reference**: [SHAOLIN_EXPLORATION.md - Section 4](SHAOLIN_EXPLORATION.md#4-gas-algorithm-visualization)

---

### Interactive DataFrame (`dataframe.py` - 179 lines)

**Main Class**: `InteractiveDataFrame`
- Exploratory data visualization
- Column selection for axes
- Dynamic dimension mapping (size, color, alpha)
- Tap event callbacks
- Interactive sizing

**Features**:
- Hover tooltips
- Dimension transforms (log, invert, rank)
- Point selection
- Layout control

**Example**:
```python
from fragile.shaolin.dataframe import InteractiveDataFrame
idf = InteractiveDataFrame(df, default_x_col="x", default_y_col="y")
layout = pn.Column(idf.layout(), idf.view())
```

**Reference**: [SHAOLIN_EXPLORATION.md - Section 5](SHAOLIN_EXPLORATION.md#5-interactive-dataframe-visualization)

---

### Dimension Mapping (`dimension_mapper.py` - 415 lines)

**Core Concept**: Map DataFrame columns to visual properties

**Classes**:
| Class | Property | Range | Features |
|-------|----------|-------|----------|
| `SizeDim` | Marker size | 0-25 | Point sizing |
| `ColorDim` | Color | 0-1 | 200+ colormaps |
| `AlphaDim` | Transparency | 0-1 | Opacity control |
| `LineWidthDim` | Line width | 0-6 | Stroke sizing |

**Base Class**: `DimensionMapper`
- Column selection widget
- Statistics (min/max/std)
- Transforms (log, invert, rank)
- Auto UI visibility

**Container**: `Dimensions`
- Groups multiple mappers
- Automatic layout
- Creates streaming dict for DynamicMap

**Example**:
```python
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim
dims = Dimensions(df, size=SizeDim, color=ColorDim)
dmap = hv.DynamicMap(plot_func, streams=dims.streams)
```

**Reference**: [SHAOLIN_EXPLORATION.md - Section 6](SHAOLIN_EXPLORATION.md#6-dimension-mapping-widgets)

---

### Parameter Selectors

#### EuclideanGasParams (`euclidean_gas_params.py` - 150+ lines)

Interactive dashboard for configuring Euclidean Gas:
- Swarm parameters (N, d, device, dtype)
- Langevin dynamics (gamma, beta, delta_t, integrator)
- Cloning mechanism (sigma_x, epsilon, lambda_alg, restitution)
- Benchmark selection

**Example**:
```python
from fragile.shaolin import EuclideanGasParamSelector
selector = EuclideanGasParamSelector()
dashboard = pn.Column(selector.swarm_section, selector.langevin_section, ...)
```

#### AdaptiveGasParams (`adaptive_gas_params.py` - 150+ lines)

Extended parameters for adaptive gas:
- All Euclidean Gas parameters
- Plus adaptive mechanisms
- Viscous coupling
- Regularized Hessian diffusion

**Reference**: [SHAOLIN_EXPLORATION.md - Section 7](SHAOLIN_EXPLORATION.md#7-parameter-selection-dashboards)

---

### Colormap Utilities (`colormaps.py` - 232 lines)

**Main Class**: `ColorMap`
- 200+ colormap selection
- Autocomplete search
- Visual swatch display
- Interactive picker

**Available Sources**:
- Matplotlib: viridis, plasma, inferno, magma, twilight, jet, etc.
- Colorcet: perceptually uniform colormaps

**Example**:
```python
from fragile.shaolin.colormaps import ColorMap
cmap = ColorMap(default="viridis")
layout = pn.Column(cmap.view())
```

**Reference**: [SHAOLIN_EXPLORATION.md - Section 8](SHAOLIN_EXPLORATION.md#8-colormap-utilities)

---

### Algorithm Control (`control.py` - 100 lines)

**Main Class**: `FaiRunner`
- Play/Pause/Step buttons
- Progress bar
- Reset button
- Sleep time control
- Summary statistics table

**Features**:
- Periodic execution
- Plot updates on each step
- Status indicators

**Reference**: [SHAOLIN_EXPLORATION.md - Section 9](SHAOLIN_EXPLORATION.md#9-control-widgets)

---

## Common Workflows

### Workflow 1: Visualize Gas Algorithm Optimization
```
1. Create GasVisualization with bounds
2. Run algorithm in loop
3. Update visualization each step
4. Call create_layout() to get Panel pane
5. Serve with pn.serve() or notebook display
```
**Reference**: [SHAOLIN_EXAMPLES.md - Example 6](SHAOLIN_EXAMPLES.md#6-gas-algorithm-visualization)

### Workflow 2: Exploratory Data Analysis
```
1. Create InteractiveDataFrame with results DataFrame
2. Select x/y columns
3. Map size/color/alpha to other columns
4. Click points to inspect
5. Use bind_tap() for custom callbacks
```
**Reference**: [SHAOLIN_EXAMPLES.md - Example 9](SHAOLIN_EXAMPLES.md#9-interactive-dataframe-exploration)

### Workflow 3: Multi-Metric Dashboard
```
1. Create Curve for each metric
2. Overlay with + operator
3. Stream updates in optimization loop
4. Combine with scatter in pn.Row/Column
```
**Reference**: [SHAOLIN_EXAMPLES.md - Example 5](SHAOLIN_EXAMPLES.md#5-multi-metric-dashboard)

### Workflow 4: Parameter Configuration
```
1. Use EuclideanGasParamSelector
2. Display dashboard with pn.serve()
3. Extract parameters with get_params()
4. Create algorithm with params
5. Run optimization with visualization
```
**Reference**: [SHAOLIN_EXAMPLES.md - Example 8](SHAOLIN_EXAMPLES.md#8-interactive-parameter-configuration)

---

## Quick Implementation Checklist

### For Scatter Plots
- [x] Import Scatter
- [x] Create DataFrame with "x", "y" columns
- [x] Create Scatter instance
- [x] Use .send(df) to stream updates
- [x] Access .plot attribute for Panel display

### For Overlaying Plots
- [x] Create multiple plot objects
- [x] Use `*` operator: plot1.plot * plot2.plot
- [x] Use `.opts()` for styling
- [x] Display in pn.pane.HoloViews()

### For Time Series
- [x] Import Curve
- [x] Set buffer_length (10000 typical)
- [x] Use Buffer stream (automatic)
- [x] Send individual data points
- [x] Access .plot for display

### For Interactive Exploration
- [x] Import InteractiveDataFrame
- [x] Pass DataFrame
- [x] Set default_x_col, default_y_col
- [x] Use .layout() for widgets
- [x] Use .view() for plot
- [x] Use .bind_tap() for callbacks

---

## Performance Benchmarks

| Plot Type | Max Points | Notes |
|-----------|-----------|-------|
| Scatter | 1000-2000 | More: downsample |
| VectorField | 500-1000 | Each is a segment |
| Curve (Buffer) | 10000 | Rolling window |
| Histogram | Unlimited | Auto-binned |
| Landscape2D | 100-200 | Interpolation cost |

**Update Frequency**: Every 10-100 steps (not every step)

**Browser**: Works with standard modern browsers

---

## Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| Plot not updating | Call `.send(df)` with proper DataFrame format |
| Slow rendering | Downsample to <5000 points |
| Memory growth | Set reasonable buffer_length |
| Wrong colors | Check column exists in DataFrame |
| Missing hover info | Add columns to DataFrame |

See [SHAOLIN_EXAMPLES.md - Troubleshooting](SHAOLIN_EXAMPLES.md#troubleshooting) for details.

---

## File Structure Overview

```
src/fragile/shaolin/
├── __init__.py                  # Exports main classes
├── stream_plots.py              # BASE: All plot types (KEY FILE)
├── gas_viz.py                   # Gas-specific visualization
├── dataframe.py                 # Interactive exploration
├── dimension_mapper.py          # Dynamic property mapping
├── euclidean_gas_params.py      # Parameter selector
├── adaptive_gas_params.py       # Adaptive parameters
├── colormaps.py                 # Colormap utilities
├── control.py                   # Algorithm runner UI
├── atari_gas_panel.py          # Atari integration
├── streaming_fai.py            # FAI streaming
├── graph.py                    # Graph visualizations
├── utils.py                    # Utilities
└── version.py                  # Version
```

**Start Here**: `stream_plots.py` (foundation)
**Key Pattern**: `gas_viz.py` (practical usage)
**Advanced**: `dataframe.py` (complex example)

---

## Related Documentation

- **Algorithm Implementation**: See `src/fragile/euclidean_gas.py`
- **Benchmark Functions**: See `src/fragile/benchmarks.py`
- **Mathematical Framework**: See `docs/source/` markdown files
- **Testing**: See `tests/test_geometric_gas.py` for visualization tests

---

## Contributing

When adding new visualizations:
1. Inherit from `StreamingPlot`
2. Implement `get_default_data()`, `preprocess_data()`
3. Set appropriate `default_opts`, `bokeh_opts`
4. Document in docstrings
5. Add example to SHAOLIN_EXAMPLES.md

---

## References

- **HoloViews**: https://holoviews.org/
- **Panel**: https://panel.holoviz.org/
- **Bokeh**: https://docs.bokeh.org/
- **hvPlot**: https://hvplot.holoviz.org/

---

## Document Generation

These documentation files were generated via systematic exploration of:
- File structure analysis
- Class hierarchy inspection
- Method signature review
- Code pattern identification
- Example extraction

Generated: 2024
Scope: Complete `src/fragile/shaolin/` directory

