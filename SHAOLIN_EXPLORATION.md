# Fragile Shaolin Visualization Exploration Report

## Overview

The `src/fragile/shaolin/` directory contains a comprehensive visualization library built on the **HoloViz stack** (HoloViews + hvPlot with Bokeh/Plotly backends) and **Panel** for interactive dashboards. This is the canonical visualization framework for the Fragile Gas algorithms.

---

## 1. Core Visualization Architecture

### File Structure
```
src/fragile/shaolin/
├── __init__.py                 # Exports parameter selectors
├── stream_plots.py             # Streaming plot base classes (KEY FILE)
├── gas_viz.py                  # Gas algorithm visualizations
├── dataframe.py                # Interactive DataFrame visualization
├── dimension_mapper.py         # Dynamic dimension mapping widgets
├── euclidean_gas_params.py     # Interactive parameter selector
├── adaptive_gas_params.py      # Adaptive gas parameter selector
├── colormaps.py                # Colormap utilities
├── control.py                  # Algorithm runner controls
├── atari_gas_panel.py          # Atari environment integration
├── streaming_fai.py            # Streaming FAI visualizations
├── graph.py                    # Graph visualizations
├── utils.py                    # Utility functions
└── version.py                  # Version info
```

---

## 2. Streaming Plot Base Classes (stream_plots.py)

### Core Abstraction: `StreamingPlot`

The foundation of all visualizations. Combines HoloViews with streaming data updates:

```python
class StreamingPlot:
    """Represents a holoviews plot updated with streamed data."""
    
    # Key methods:
    - init_stream()          # Set up data stream (Pipe or Buffer)
    - init_plot()            # Initialize DynamicMap with streaming
    - send(data)             # Stream new data to plot
    - opts(**kwargs)         # Configure plot appearance
    - preprocess_data(data)  # Transform data before visualization
```

**Key Features:**
- Backend-aware options (Bokeh vs Matplotlib)
- Dynamic plot updates without recreation
- Normalized axes across all plots
- Support for both `Pipe` and `Buffer` streaming

---

### 3. Available Plot Types

#### 3.1 **Scatter** - Position scatter plots
```python
Scatter(
    data=pd.DataFrame({"x": [], "y": []}),
    bokeh_opts={
        "color": "blue",
        "size": 8,
        "alpha": 0.7,
        "height": 600,
        "width": 600,
        "tools": ["hover", "box_zoom", "wheel_zoom", "reset"],
    }
)
```

**Features:**
- Hover tooltips for data inspection
- Color mapping from DataFrame columns
- Interactive box/wheel zoom
- Customizable size and alpha

#### 3.2 **VectorField** - Velocity/force visualization
```python
VectorField(
    data=pd.DataFrame({"x0": [], "y0": [], "x1": [], "y1": []}),
    scale=0.5,
    bokeh_opts={
        "line_color": "cyan",
        "line_width": 2,
        "alpha": 0.5,
    }
)
```

**Features:**
- Uses HoloViews `Segments` internally
- Segment data format: (x0, y0) -> (x1, y1)
- Velocity scaling for visualization
- Transparent overlay on scatter plots

#### 3.3 **Curve** - Time series / live tracking
```python
Curve(
    data=pd.DataFrame({"step": [], "alive": []}),
    buffer_length=10000,  # Keep last N points
    bokeh_opts={
        "color": "green",
        "line_width": 3,
        "height": 300,
        "width": 800,
    }
)
```

**Features:**
- Uses `Buffer` stream for rolling window
- Configurable buffer length
- Hover tool for value inspection
- Line color and width customization

#### 3.4 **Histogram** - Distribution visualization
```python
Histogram(
    data=None,
    n_bins=20,
    bokeh_opts={"tools": ["hover"]}
)
```

**Features:**
- Automatic bin computation
- Handles inf/nan values
- Customizable bin count
- Hover tooltips with bin info

#### 3.5 **Landscape2D** - Energy landscape with contours
```python
Landscape2D(
    data=None,
    contours=True,
)
```

**Features:**
- 2D interpolation using `griddata`
- QuadMesh + Contours overlay
- Interactive colormap selection
- Scatter overlay with original points

#### 3.6 **Additional Types**
- `Table` - DataFrame display
- `Bivariate` - 2D kernel density estimation
- `RGB` / `Image` - Image data
- `QuadMesh` - Heat map grids
- `Div` - HTML content

---

## 4. Gas Algorithm Visualization (gas_viz.py)

### **GasVisualization** - Canonical Euclidean Gas visualization

```python
class GasVisualization:
    """Displays swarm dynamics in real-time."""
    
    def __init__(
        self,
        bounds=None,                    # Optional Bounds object
        position_color="blue",          # Walker position color
        velocity_color="cyan",          # Velocity vector color
        velocity_scale=0.5,             # Arrow scaling
        plot_size=600,                  # Plot dimensions
        track_alive=True                # Monitor alive/dead walkers
    ):
    
    def update(self, state: SwarmState, n_alive: int | None = None):
        """Update with new swarm state."""
    
    def create_layout():
        """Build complete Panel layout."""
```

**Layout Components:**
1. **Position Scatter**: Walker locations (blue by default)
2. **Velocity Field**: Velocity vectors overlaid (cyan by default)
3. **Alive/Cloned Curve**: Time series of walker counts
   - Green solid line: Alive walkers
   - Red dashed line: Cloned/dead walkers

**Example Usage:**
```python
from fragile.euclidean_gas import EuclideanGas, SwarmState
from fragile.shaolin.gas_viz import GasVisualization

viz = GasVisualization(plot_size=600, track_alive=True)

# In optimization loop:
state = gas.step()
viz.update(state, n_alive=n_alive_walkers)
layout = viz.create_layout()
```

---

### **BoundaryGasVisualization** - Boundary-aware visualization

Extends `GasVisualization` to color-code boundary violations:

```python
class BoundaryGasVisualization(GasVisualization):
    def __init__(
        self,
        bounds,                         # Bounds object (required)
        in_bounds_color="blue",         # In-bounds walker color
        out_bounds_color="red",         # Out-of-bounds walker color
        velocity_color="cyan",
        velocity_scale=0.5,
        plot_size=600
    ):
    
    def update(self, state: SwarmState):
        """Automatically detects boundary violations."""
```

**Features:**
- Automatic boundary checking
- Color-coded particle states
- Only shows velocity vectors for in-bounds walkers
- Tracks alive count separately

---

## 5. Interactive DataFrame Visualization (dataframe.py)

### **InteractiveDataFrame** - Exploratory data visualization

Combines dimensional mapping with interactive tap events:

```python
class InteractiveDataFrame(param.Parameterized):
    def __init__(
        self,
        df: pd.DataFrame,
        ignore_cols=None,               # Columns to exclude
        n_cols=3,                       # Widget layout columns
        default_x_col="x",
        default_y_col="y"
    ):
```

**Features:**
1. **Dimension Selection**: Choose x/y axes from DataFrame columns
2. **Dynamic Styling**:
   - Size mapping (via `SizeDim`)
   - Color mapping (via `ColorDim`)
   - Transparency (via `AlphaDim`)
3. **Dimension Transforms**:
   - Log scale
   - Invert
   - Rank
4. **Tap Events**: Click points to trigger callbacks
5. **Plot Sizing**: Interactive height/width controls

**Architecture:**
```
InteractiveDataFrame
├── df_dims: Dimensions (size, color, alpha mappers)
├── sel_x/sel_y: Column selectors
├── dmap: HoloViews DynamicMap (streams-aware)
├── tap_stream: Tap event handler
└── layout(): Panel UI
```

**Example Usage:**
```python
from fragile.shaolin.dataframe import InteractiveDataFrame, SizeDim, ColorDim

idf = InteractiveDataFrame(
    df_results,
    default_x_col="iteration",
    default_y_col="reward"
)

# Bind custom callback to tap events
def on_tap(index, df):
    print(f"Clicked point {index}: {df.iloc[index]}")

callback = idf.bind_tap(on_tap)
panel_layout = pn.Column(idf.view(), callback)
```

---

## 6. Dimension Mapping Widgets (dimension_mapper.py)

### **DimensionMapper** - Base class for dynamic dimension mapping

Maps DataFrame columns to visual properties:

```python
class DimensionMapper(param.Parameterized):
    """Base class for dynamic dimension mapping."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        name: str,                      # Widget name
        default_value: float,           # Default when no column selected
        value_range: tuple,             # Output range (min, max) or (min, max, step)
        ignore_cols=(),
        resolution=100,
        default_range=None,
        ignore_string_cols=True,
        epsilon=1e-5
    ):
```

**Key Methods:**
- `get_value()` - Compute dimension value from selected column
- `panel()` - Create Panel widget UI
- `update_col_values()` - Recompute statistics on column change
- `update_ui()` - Show/hide widgets based on selection

**Transformations Available:**
- Log scale (skips boolean columns)
- Invert (flip colors/sizes)
- Rank (sort values)

---

### **Specialized Dimension Mappers**

#### **SizeDim** - Map to point/marker size
```python
SizeDim(
    df,
    name="size",
    default=8,                         # Default marker size
    value_range=(0, 25),              # Output range
    default_range=(1, 10)             # Widget range
)
```

#### **ColorDim** - Map to color using colormaps
```python
ColorDim(
    df,
    name="color",
    default="#30a2da",                # Default color
    value_range=(0.0, 1.0),          # Colormap range
)
```

Features:
- Colormap selector widget (100+ colormaps from colorcet + matplotlib)
- Auto-update colormap display
- Boolean column support

#### **AlphaDim** - Map to transparency
```python
AlphaDim(
    df,
    name="alpha",
    default=1.0,
    value_range=(0.0, 1.0),
    default_range=(0.1, 1.0)
)
```

#### **LineWidthDim** - Map to line width (for plots)
```python
LineWidthDim(
    df,
    name="line_width",
    default=2.0,
    value_range=(0.0, 6.0)
)
```

---

### **Dimensions** - Multi-dimensional mapping container

Groups multiple mappers with automatic widget layout:

```python
dims = Dimensions(
    df,
    n_cols=3,  # Widgets per row
    size=SizeDim,
    color=ColorDim,
    alpha=AlphaDim,
    line_width=LineWidthDim
)

# Access mappers
dims.dimensions["size"].value      # Current size value
dims.dimensions["color"].value     # Current color value
dims.streams                        # Dict for DynamicMap streaming
```

---

## 7. Parameter Selection Dashboards

### **EuclideanGasParamSelector** - Interactive Gas configuration

```python
class EuclideanGasParamSelector(param.Parameterized):
    """Panel dashboard for Euclidean Gas parameter configuration."""
    
    # Swarm parameters
    n_walkers: int
    dimensions: int
    device: {"cpu", "cuda"}
    dtype: {"float32", "float64"}
    
    # Langevin dynamics
    gamma: float              # Friction coefficient
    beta: float               # Inverse temperature
    delta_t: float            # Time step
    integrator: str           # {"baoab", "aboba", "babo", "obab"}
    
    # Cloning mechanism
    sigma_x: float            # Collision radius
    epsilon: float            # Companion selection range
    lambda_alg: float         # Cloning parameter
    alpha_restitution: float  # Collision elasticity
    use_inelastic_collision: bool
    
    # Benchmark
    benchmark_type: str       # Optimization function
```

**Usage:**
```python
import panel as pn
from fragile.shaolin import EuclideanGasParamSelector

pn.extension()
selector = EuclideanGasParamSelector()
params = selector.get_params()  # Get configured parameters
gas = EuclideanGas(params)
```

---

### **AdaptiveGasParamSelector** - Extended adaptive gas parameters

Similar to EuclideanGasParamSelector but with additional sections for:
- Adaptive force parameters
- Viscous coupling configuration
- Regularized Hessian diffusion

---

## 8. Colormap Utilities (colormaps.py)

### **ColorMap** - Interactive colormap selector

```python
class ColorMap(param.Parameterized):
    """Widget for interactive colormap selection."""
    
    all_cmaps: dict    # 200+ colormaps from matplotlib + colorcet
    value: object      # Currently selected colormap
```

**Available Colormaps:**
- Matplotlib: viridis, plasma, inferno, magma, twilight, etc.
- Colorcet: perceptually uniform colormaps
- Total: 200+ options

**Features:**
- Autocomplete search
- Visual swatch display
- Interactive selection

---

## 9. Control Widgets (control.py)

### **FaiRunner** - Algorithm execution controller

```python
class FaiRunner(param.Parameterized):
    """Controller for FAI algorithm execution with UI."""
    
    def __init__(self, fai, n_steps, plot=None):
```

**UI Components:**
- Play/Pause/Step buttons
- Progress bar
- Reset button
- Sleep time control
- Summary statistics table

**Features:**
- Periodic execution via `pn.state.add_periodic_callback()`
- Plot updates on each step
- Progress tracking
- Status indicators (success/danger)

---

## 10. Patterns and Best Practices

### Pattern 1: Basic Scatter Plot

```python
import pandas as pd
from fragile.shaolin.stream_plots import Scatter

# Create scatter plot
scatter = Scatter(
    data=pd.DataFrame({"x": [], "y": []}),
    bokeh_opts={
        "color": "blue",
        "size": 8,
        "alpha": 0.7,
        "height": 600,
        "width": 600,
        "tools": ["hover", "box_zoom", "wheel_zoom", "reset"],
    }
)

# Stream updates
new_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
scatter.send(new_data)

# Get HoloViews plot for Panel display
plot_pane = pn.pane.HoloViews(scatter.plot)
```

### Pattern 2: Overlaying Multiple Plots

```python
from fragile.shaolin.stream_plots import Scatter, VectorField

position = Scatter(data=pos_df, ...)
velocity = VectorField(data=vel_df, ...)

# Overlay plots using * operator
combined = (position.plot * velocity.plot).opts(
    title="Gas Dynamics",
    xlabel="x₁",
    ylabel="x₂"
)
```

### Pattern 3: Streaming with Curve Buffer

```python
from fragile.shaolin.stream_plots import Curve

# Curve uses Buffer for rolling window
alive_tracker = Curve(
    data=pd.DataFrame({"step": [], "count": []}),
    buffer_length=1000,  # Keep last 1000 points
    bokeh_opts={"color": "green", "line_width": 3}
)

# Stream individual data points
for step in range(n_steps):
    new_point = pd.DataFrame({"step": [step], "count": [n_alive]})
    alive_tracker.send(new_point)
```

### Pattern 4: Interactive Dimension Mapping

```python
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim
import holoviews as hv

# Create dimension mappers
dims = Dimensions(
    df,
    n_cols=2,
    size=(SizeDim, {"value_range": (1, 15)}),
    color=ColorDim
)

# Create dynamic plot with streaming
def plot_func(data, size, color):
    return hv.Scatter(data, kdims=['x'], vdims=['y']).opts(
        size=size,
        color=color
    )

dmap = hv.DynamicMap(plot_func, streams=dims.streams)
```

### Pattern 5: Gas Algorithm Visualization

```python
from fragile.euclidean_gas import EuclideanGas
from fragile.shaolin.gas_viz import GasVisualization

# Initialize
gas = EuclideanGas(params)
viz = GasVisualization(
    bounds=bounds,
    position_color="blue",
    velocity_color="cyan"
)

# Optimization loop
for step in range(n_steps):
    state = gas.step()
    viz.update(state, n_alive=count_alive(state))

# Create Panel layout
layout = viz.create_layout()
pn.serve(layout)
```

---

## 11. Backend Configuration

### Bokeh (2D - Default)

```python
import holoviews as hv
hv.extension('bokeh')  # Already default in shaolin

# Bokeh-specific options
{
    "height": 600,
    "width": 800,
    "tools": ["hover", "box_zoom", "wheel_zoom", "reset"],
    "bgcolor": "lightgray",
    "colorbar": True
}
```

### Plotly (3D - Alternative)

```python
import holoviews as hv
hv.extension('plotly')

# Plotly-specific options
{
    "height": 600,
    "width": 800,
    "camera_angle": (25, 45, 100)  # For 3D plots
}
```

### matplotlib (Legacy - Avoid)

The codebase supports matplotlib options but **DO NOT USE in new code**. Always use Bokeh/Plotly.

---

## 12. Key Import Patterns

### Correct Imports

```python
# Main visualization
from fragile.shaolin.stream_plots import Scatter, Curve, VectorField, Landscape2D

# Gas-specific
from fragile.shaolin.gas_viz import GasVisualization, BoundaryGasVisualization

# Interactive components
from fragile.shaolin.dataframe import InteractiveDataFrame
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim

# Parameter selectors
from fragile.shaolin import EuclideanGasParamSelector, AdaptiveGasParamSelector

# Panel integration
import panel as pn
pn.extension()

# HoloViews base
import holoviews as hv
hv.extension('bokeh')
```

---

## 13. Stream Types

### Pipe Stream (Stateless)
```python
from holoviews.streams import Pipe

stream = Pipe(data=initial_df)
stream.send(new_df)  # Replaces entire data
```
**Use Case**: Points/events where order doesn't matter

### Buffer Stream (Stateful)
```python
from holoviews.streams import Buffer

stream = Buffer(data=initial_df, length=1000)
stream.send(new_row_df)  # Appends, maintains rolling window
```
**Use Case**: Time series, cumulative tracking

---

## 14. Common Customizations

### Changing Colors

```python
scatter.opts(color="red")  # Solid color
scatter.opts(color="column_name")  # Map to column
```

### Adding Hover Tooltips

```python
from bokeh.models import HoverTool

hover = HoverTool(tooltips=[
    ("Index", "$index"),
    ("Position", "(@x, @y)"),
    ("Reward", "@reward")
])

scatter.opts(tools=[hover])
```

### Axis Limits

```python
plot = plot.opts(xlim=(0, 10), ylim=(0, 5))
```

### Legend and Titles

```python
plot = plot.opts(
    title="Gas Algorithm Step 100",
    xlabel="x-coordinate",
    ylabel="y-coordinate",
    legend_position="top_right"
)
```

---

## 15. Data Format Requirements

### For Scatter

```python
pd.DataFrame({
    "x": [x1, x2, ...],      # Required: x-coordinate
    "y": [y1, y2, ...],      # Required: y-coordinate
    "color": [...],          # Optional: color values
    "size": [...],           # Optional: size values
    "alpha": [...]           # Optional: transparency
})
```

### For VectorField (Segments)

```python
pd.DataFrame({
    "x0": [x_start_1, ...],  # Arrow start x
    "y0": [y_start_1, ...],  # Arrow start y
    "x1": [x_end_1, ...],    # Arrow end x
    "y1": [y_end_1, ...]     # Arrow end y
})
```

### For Curve

```python
pd.DataFrame({
    "step": [0, 1, 2, ...],  # X-axis (time)
    "value": [v1, v2, v3, ...]  # Y-axis (metric)
})
```

---

## 16. Performance Considerations

1. **Buffer Length**: For live data, limit `Curve` buffer to prevent memory issues
   ```python
   Curve(..., buffer_length=10000)  # Keep last 10k points
   ```

2. **Update Frequency**: Too frequent updates can overwhelm browser
   ```python
   # In optimization loop, update every N steps
   if step % 10 == 0:
       viz.update(state)
   ```

3. **Point Limit**: Many scattered points can slow rendering
   ```python
   # Consider downsampling
   if len(positions) > 5000:
       positions = positions[::downsample_factor]
   ```

4. **Color Mapping**: Categorical columns are faster than continuous

---

## Summary of Key Files and Functions

| File | Key Class/Function | Purpose |
|------|-------------------|---------|
| `stream_plots.py` | `StreamingPlot`, `Scatter`, `Curve`, `VectorField`, `Landscape2D` | Base streaming visualization framework |
| `gas_viz.py` | `GasVisualization`, `BoundaryGasVisualization` | Canonical Gas algorithm visualization |
| `dataframe.py` | `InteractiveDataFrame` | Exploratory DataFrame visualization with tap events |
| `dimension_mapper.py` | `DimensionMapper`, `SizeDim`, `ColorDim`, `AlphaDim`, `Dimensions` | Dynamic visual property mapping |
| `euclidean_gas_params.py` | `EuclideanGasParamSelector` | Interactive parameter configuration |
| `colormaps.py` | `ColorMap` | 200+ colormap selection widget |
| `control.py` | `FaiRunner` | Algorithm execution UI controller |

---

## Recommended Reading Order

1. **Start here**: `stream_plots.py` - Understand `StreamingPlot` base class
2. **Then**: `gas_viz.py` - See practical Gas visualization usage
3. **Then**: `dimension_mapper.py` - Learn interactive styling patterns
4. **Reference**: `dataframe.py` - Complex interactive example
5. **Use as needed**: Parameter selectors and control widgets

