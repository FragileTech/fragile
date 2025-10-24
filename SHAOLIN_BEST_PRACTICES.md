# Best Practices: Shaolin Interactive Visualization Library

**Purpose**: This document teaches how to use the Shaolin module (`src/fragile/shaolin/`) for building interactive, streaming visualizations of algorithm data using HoloViews and Panel.

---

## Table of Contents

1. [Introduction & Philosophy](#1-introduction--philosophy)
2. [Core Architecture](#2-core-architecture)
3. [DimensionMapper: Column → Visual Property](#3-dimensionmapper-column--visual-property)
4. [StreamingPlot: Real-Time Updates](#4-streamingplot-real-time-updates)
5. [Plot Type Reference](#5-plot-type-reference)
6. [Interactive Visualizations](#6-interactive-visualizations)
7. [Control Widgets](#7-control-widgets)
8. [Practical Examples](#8-practical-examples)
9. [Advanced Patterns](#9-advanced-patterns)
10. [Best Practices Checklist](#10-best-practices-checklist)

---

## 1. Introduction & Philosophy

### What is Shaolin?

**Shaolin** is an interactive visualization library built specifically for the Fragile Gas algorithms. It provides:

- **Real-time streaming plots** that update as your algorithm runs
- **Interactive dimension mapping**: Map DataFrame columns to visual properties (size, color, alpha, etc.)
- **Pre-built dashboards** for common visualization patterns (DataFrames, graphs, algorithm states)
- **Tap event handling** for interactive exploration

### Core Technology Stack

```
┌─────────────────────┐
│   Your Algorithm    │  ← Produces DataFrames/states
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Shaolin Layer     │  ← Maps data → visual properties
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   HoloViews Plots   │  ← Declarative plotting
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Panel Dashboard   │  ← Interactive UI with widgets
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Bokeh/Plotly       │  ← Backend rendering
└─────────────────────┘
```

### Key Philosophy

**"DataFrame columns are the source of truth"** - Shaolin automatically generates UI controls for mapping any column to any visual property, with transformations applied on-the-fly.

**Example**:
```python
# Instead of hardcoding:
plt.scatter(df['x'], df['y'], s=df['reward'], c=df['distance'])

# Shaolin lets users choose dynamically:
viz = InteractiveDataFrame(df)
# User selects: x=position_x, y=position_y, size=reward, color=distance
# User can switch to: x=velocity, y=energy, size=distance, color=reward
# All through UI widgets - no code changes
```

---

## 2. Core Architecture

### Three Core Patterns

Shaolin is built around three complementary patterns:

#### Pattern 1: **DimensionMapper**
Maps DataFrame columns to visual properties (size, color, alpha, line_width) with user controls.

```python
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim, AlphaDim

dims = Dimensions(
    dataframe,
    n_cols=3,          # Layout: 3 widgets per row
    size=SizeDim,      # Map column → point size
    color=ColorDim,    # Map column → color
    alpha=AlphaDim,    # Map column → transparency
)

# Auto-generated stream dict for HoloViews
streams = dims.streams  # {'size': <param>, 'color': <param>, 'cmap': <param>, ...}
```

#### Pattern 2: **StreamingPlot**
Base class for real-time plots that accept streaming data.

```python
from fragile.shaolin.stream_plots import Scatter

plot = Scatter(data=initial_df)

# Later, stream new data
plot.send(updated_df)  # Plot updates automatically

# Render in Panel
dashboard = pn.pane.HoloViews(plot.plot)
```

#### Pattern 3: **Interactive Classes**
Full-featured dashboard components with dimension mapping + tap events.

```python
from fragile.shaolin.dataframe import InteractiveDataFrame

viz = InteractiveDataFrame(df, n_cols=3)

# Automatic UI generation:
# - Column selectors for x/y axes
# - Dimension mappers for size/color/alpha
# - Tap event handling
# - Width/height sliders

dashboard = viz.__panel__()
dashboard.show()  # Opens browser
```

---

## 3. DimensionMapper: Column → Visual Property

### The DimensionMapper Class

**Purpose**: Map a DataFrame column to a visual property with interactive controls.

#### Core Components

Each `DimensionMapper` provides:

1. **Column selector**: Dropdown to choose which column to map
2. **Range slider**: Control output range (e.g., size: 1-10, alpha: 0-1)
3. **Transformation buttons**: Invert, Log scale, Rank
4. **Default value**: Used when no column is selected

#### Workflow

```
User selects column "reward"
         ↓
Extract values: [0.1, 5.2, 2.3, ...]
         ↓
Apply transformation (e.g., Log scale)
         ↓
Normalize to [0, 1]
         ↓
Apply "Invert" if selected: 1 - value
         ↓
Map to output range [1, 10] for size
         ↓
Result: [1.2, 8.5, 4.1, ...]
```

### Specialized DimensionMappers

#### **SizeDim** - Point/marker size

```python
from fragile.shaolin.dimension_mapper import SizeDim

size_mapper = SizeDim(
    df,
    name="size",
    default=8,                    # Default point size
    value_range=(0, 25),          # Output range
    default_range=(1, 10),        # Initial slider range
)

# Use in plot
scatter = hv.Scatter(df, kdims=['x', 'y']).opts(size=size_mapper.value)
```

**Use case**: Emphasize points by some metric (reward, importance, error).

#### **ColorDim** - Point/marker color

```python
from fragile.shaolin.dimension_mapper import ColorDim

color_mapper = ColorDim(
    df,
    name="color",
    default="#30a2da",            # Default color (hex)
    value_range=(0.0, 1.0),       # Normalized range for colormap
)

# Includes colormap selection widget
scatter = hv.Scatter(df, kdims=['x', 'y']).opts(
    color=color_mapper.value,
    cmap=color_mapper.cmap,
    colorbar=True,
)
```

**Special features**:
- Integrated colormap selector (matplotlib + colorcet)
- Boolean column handling (categorizes True/False)
- Auto-hides range slider (uses full colormap range)

#### **AlphaDim** - Transparency

```python
from fragile.shaolin.dimension_mapper import AlphaDim

alpha_mapper = AlphaDim(
    df,
    name="alpha",
    default=1.0,                  # Fully opaque
    value_range=(0.0, 1.0),
    default_range=(0.1, 1.0),
)

scatter = hv.Scatter(df, kdims=['x', 'y']).opts(alpha=alpha_mapper.value)
```

**Use case**: De-emphasize low-confidence or old data points.

#### **LineWidthDim** - Line thickness (for graphs/segments)

```python
from fragile.shaolin.dimension_mapper import LineWidthDim

line_mapper = LineWidthDim(
    df,
    name="line_width",
    default=2.0,
    value_range=(0.0, 6.0),
    default_range=(0.5, 3.0),
)

segments = hv.Segments(df, kdims=['x0', 'y0', 'x1', 'y1']).opts(
    line_width=line_mapper.value
)
```

**Use case**: Visualize edge weights, connection strength.

### Transformations

All DimensionMappers support three transformations:

#### **Invert**
Flip the normalized values: `value → 1 - value`

**Use case**: "Smaller is better" metrics (error, cost).

```python
# Without invert: high error → large size
# With invert: high error → small size
```

#### **Log Scale**
Apply logarithm before normalization: `value → log(value + ε)`

**Use case**: Wide dynamic range (1e-5 to 1e5).

```python
# Without log: Most points clustered at bottom
# With log: Even distribution across scale
```

#### **Rank**
Replace values with their rank: `[2.1, 5.3, 1.2] → [1, 2, 0]`

**Use case**: Ordinal relationships more important than absolute values.

### The Dimensions Container

The `Dimensions` class manages multiple DimensionMappers:

```python
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim, AlphaDim

dims = Dimensions(
    df,
    n_cols=3,                    # Widget layout: 3 per row
    size=SizeDim,                # Default config
    color=ColorDim,
    alpha=AlphaDim,
    # Custom config with tuple:
    line_width=(LineWidthDim, {"default": 1.5, "value_range": (0, 5)}),
)

# Auto-generated streams for HoloViews DynamicMap
streams = dims.streams
# {
#   'size': <param.value>,
#   'color': <param.value>,
#   'cmap': <param.cmap>,        # Auto-added for ColorDim
#   'alpha': <param.value>,
#   'line_width': <param.value>
# }

# Render all widgets
widget_panel = dims.panel()
```

**Key feature**: Automatically sorts widgets by priority (Size → Alpha → Color).

---

## 4. StreamingPlot: Real-Time Updates

### The StreamingPlot Base Class

**Purpose**: Wrap HoloViews plots to accept streaming data updates.

#### Core Concept

```python
# Static plot
plot = hv.Scatter(df)

# Streaming plot
from fragile.shaolin.stream_plots import Scatter

streaming_plot = Scatter(data=df)

# Update data
for new_df in data_stream:
    streaming_plot.send(new_df)  # Plot updates in real-time
```

### Stream Types

#### **Pipe** (default)
Replace entire dataset each update.

```python
from holoviews.streams import Pipe

pipe = Pipe(data=initial_df)
# Later: pipe.send(new_df)  # Complete replacement
```

**Use case**: Full state snapshots (e.g., swarm positions at each iteration).

#### **Buffer**
Append data with sliding window.

```python
from holoviews.streams import Buffer

buffer = Buffer(data=initial_df, length=1000)  # Keep last 1000 points
# Later: buffer.send(new_row)  # Append, trim oldest
```

**Use case**: Time series (e.g., reward history, loss curves).

### Backend Agnostic

StreamingPlot supports both Bokeh and Matplotlib backends:

```python
plot = Scatter(data=df, bokeh_opts={'size': 5}, mpl_opts={'s': 20})

# Bokeh backend
import holoviews as hv
hv.extension('bokeh')
plot.opts(width=600, height=400)  # Uses bokeh_opts

# Matplotlib backend
hv.extension('matplotlib')
plot.opts(width=6, height=4)  # Uses mpl_opts
```

---

## 5. Plot Type Reference

### Complete Catalog

| Plot Type | Best For | Stream Type | Key Options |
|-----------|----------|-------------|-------------|
| **Curve** | Time series | Buffer | `buffer_length`, `data_names` |
| **Histogram** | Distributions | Pipe | `n_bins`, `xlim` |
| **Scatter** | Point clouds | Pipe | `size`, `color`, `alpha` |
| **QuadMesh** | Heatmaps | Pipe | `n_points`, `cmap` |
| **QuadMeshContours** | Contoured heatmaps | Pipe | `n_points`, `levels` |
| **Landscape2D** | Interpolated surfaces | Pipe | `n_points`, `contours` |
| **RGB** | Images (3-channel) | Pipe | `xaxis`, `yaxis` |
| **Image** | Images (1-channel) | Pipe | `xaxis`, `yaxis` |
| **VectorField** | Arrows/segments | Pipe | `scale`, `line_color`, `line_width` |
| **Bivariate** | 2D density | Pipe | `bins`, `cmap` |
| **Table** | Data tables | Pipe | `width`, `height` |
| **Div** | Text/HTML | Pipe | `width`, `height` |

### Detailed Usage

#### **Curve** - Time Series

```python
from fragile.shaolin.stream_plots import Curve
import pandas as pd

# Initialize
curve = Curve(
    data=pd.DataFrame({'iteration': [], 'reward': []}),
    buffer_length=1000,   # Keep last 1000 points
    index=False,
    data_names=('iteration', 'reward'),
)

# Stream data
for i, reward in enumerate(rewards):
    curve.send(pd.DataFrame({'iteration': [i], 'reward': [reward]}))

# Render
dashboard = pn.pane.HoloViews(curve.plot)
```

**Use case**: Reward curves, loss tracking, convergence monitoring.

#### **Histogram** - Distributions

```python
from fragile.shaolin.stream_plots import Histogram
import numpy as np

hist = Histogram(
    data=np.random.randn(100),
    n_bins=30,
)

# Update with new data
hist.send(np.random.randn(200))  # Distribution updates

# Options
hist.opts(ylabel='Frequency', xlabel='Value', framewise=True)
```

**Use case**: Parameter distributions, error histograms.

#### **Scatter** - Point Clouds

```python
from fragile.shaolin.stream_plots import Scatter

scatter = Scatter(
    data=pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
)

# With dimension mapping
scatter.send(pd.DataFrame({
    'x': walker_positions[:, 0],
    'y': walker_positions[:, 1],
    'reward': rewards,
}))

scatter.opts(
    size=5,
    alpha=0.7,
    color='reward',
    cmap='viridis',
    colorbar=True,
)
```

**Use case**: Walker positions, state space exploration.

#### **QuadMesh** - Heatmaps

```python
from fragile.shaolin.stream_plots import QuadMesh

qmesh = QuadMesh(
    data=None,  # Will auto-generate default
    n_points=50,  # Grid resolution
)

# Send (x, y, z) data - will interpolate to grid
x = walker_positions[:, 0]
y = walker_positions[:, 1]
z = energies

qmesh.send((x, y, z))

qmesh.opts(
    cmap='plasma',
    colorbar=True,
    bgcolor='lightgray',
)
```

**Use case**: Energy landscapes, density maps.

#### **QuadMeshContours** - Contoured Heatmaps

```python
from fragile.shaolin.stream_plots import QuadMeshContours

contour = QuadMeshContours(
    data=None,
    n_points=50,
    levels=16,  # Number of contour lines
)

contour.send((x, y, z))

contour.opts(
    cmap='viridis',
    line_color='black',
    alpha=0.9,
)
```

**Use case**: Potential functions, fitness landscapes with topography.

#### **Landscape2D** - Complete Interpolated Visualization

Combines QuadMesh + Contours + Scatter for full landscape view.

```python
from fragile.shaolin.stream_plots import Landscape2D

landscape = Landscape2D(
    data=None,
    contours=True,   # Include contour lines
    n_points=50,
)

# Send walker data
landscape.send((x, y, z))

# Internally creates:
# - QuadMesh for color-coded surface
# - Contours for topography
# - Scatter for original data points
```

**Use case**: Complete state space visualization.

#### **VectorField** - Velocity/Force Arrows

```python
from fragile.shaolin.stream_plots import VectorField

vectors = VectorField(
    data=pd.DataFrame({'x0': [], 'y0': [], 'x1': [], 'y1': []}),
    scale=0.5,  # Arrow scaling factor
)

# Send segment data (start and end points)
x0, y0 = walker_positions[:, 0], walker_positions[:, 1]
x1, y1 = x0 + velocities[:, 0] * scale, y0 + velocities[:, 1] * scale

vectors.send(pd.DataFrame({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}))

vectors.opts(
    line_color='cyan',
    line_width=2,
    alpha=0.6,
)
```

**Use case**: Velocity fields, force vectors, gradient visualization.

**Pattern for overlay**:
```python
# Overlay vectors on scatter plot
scatter = Scatter(data=walker_df)
vectors = VectorField(data=vector_df)

overlay = scatter.plot * vectors.plot
pn.pane.HoloViews(overlay)
```

#### **RGB** - Image Rendering

```python
from fragile.shaolin.stream_plots import RGB

rgb = RGB(data=None)

# Send RGB array [height, width, 3]
image = env.render()  # e.g., (210, 160, 3)
rgb.send(image)

rgb.opts(xaxis=None, yaxis=None)  # Hide axes
```

**Use case**: Environment rendering (Atari, robotic vision).

**Example from montezuma.py**:
```python
# Render game frames
rgb_plot = RGB(data=fai.rgb[0])

for step in range(n_steps):
    fai.step_tree()
    best_walker_ix = fai.cum_reward.argmax()
    rgb_plot.send(fai.rgb[best_walker_ix])
```

#### **Table** - Data Display

```python
from fragile.shaolin.stream_plots import Table

table = Table(data=pd.DataFrame({'metric': ['reward', 'steps'], 'value': [100, 50]}))

# Update metrics
summary = pd.DataFrame(fai.summary(), index=[0])
table.send(summary)

table.opts(width=400, height=200)
```

**Use case**: Summary statistics, hyperparameter display.

---

## 6. Interactive Visualizations

### InteractiveDataFrame

**Purpose**: Turn any DataFrame into a full interactive dashboard.

#### Basic Usage

```python
from fragile.shaolin.dataframe import InteractiveDataFrame

df = pd.DataFrame({
    'x': walker_positions[:, 0],
    'y': walker_positions[:, 1],
    'reward': rewards,
    'distance': distances,
    'is_alive': alive_mask,
})

viz = InteractiveDataFrame(
    df,
    n_cols=3,                    # Widget layout
    default_x_col='x',
    default_y_col='y',
    ignore_cols=('states',),     # Hide these columns
)

dashboard = viz.__panel__()
dashboard.show()
```

#### What You Get

1. **Scatter plot** of data with hover tooltips
2. **X/Y axis selectors** - switch between any numeric columns
3. **Dimension mappers** - size, color, alpha controls
4. **Width/height sliders** - resize plot
5. **Tap events** - click points to trigger actions

#### Updating Data

```python
# Method 1: Update via pipe
viz.pipe.send(new_df)

# Method 2: Direct property access
viz.pipe.data = new_df
```

#### Auto-Generated UI

```
┌─────────────────────────────────────────────┐
│  [X: x ▼] [Y: y ▼] [Height: ——●——] [Width] │
│                                             │
│  Size Mapping:                              │
│  [Column: reward ▼] [Range: ●——●] [Default]│
│  [ ] Invert  [ ] Log  [ ] Rank              │
│                                             │
│  Color Mapping:                             │
│  [Column: distance ▼] [Colormap: viridis ▼]│
│  [ ] Invert  [ ] Log  [ ] Rank              │
│                                             │
│  Alpha Mapping:                             │
│  [Column: None ▼] [Default: 1.0]            │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│                                             │
│          ●  ●    ● ●                        │
│        ●    ●  ●   ●  ●                     │
│      ●  ●      ●      ●  ●                  │
│    ●      ●  ●   ●  ●      ●                │
│  ●    ●     ●       ●    ●    ●             │
│    ●    ●      ●  ●    ●    ●               │
│      ●    ●  ●       ●    ●                 │
│        ●    ●    ●      ●                   │
│          ●    ●    ●  ●                     │
│                                             │
└─────────────────────────────────────────────┘
```

### InteractiveFai

**Purpose**: Specialized visualization for Fractal AI tree structures.

**From streaming_fai.py**:

```python
from fragile.shaolin.streaming_fai import InteractiveFai

fai = FractalTree(...)  # Your algorithm instance

viz = InteractiveFai(
    fai,
    n_cols=3,
    default_x_col='x',
    default_y_col='y',
    ignore_cols=('states',),
)

# Includes:
# - Graph edges connecting parent-child walkers
# - Dimension mapping for nodes
# - Summary table (iteration, best_reward, etc.)
# - Benchmark target marker (red star)

dashboard = viz.__panel__()
dashboard.show()
```

#### Additional Features

**Summary Table**: Auto-generated from `fai.summary()`:
```python
def summary(self):
    return {
        "iteration": self.iteration,
        "leaf_nodes": self.is_leaf.sum().item(),
        "best_reward": self.cum_reward.max().item(),
        "will_clone": self.will_clone.sum().item(),
        "n_walkers": self.n_walkers,
    }
```

**Graph Visualization**: Edges show parent-child relationships:
```python
def plot_edges(df, **kwargs):
    parent, x, y = df["parent"].values, df["x"].values, df["y"].values
    segments = x[parent], y[parent], x, y  # (x_start, y_start, x_end, y_end)
    return hv.Segments(segments).opts(**kwargs)
```

### InteractiveGraph

**Purpose**: NetworkX graph visualization with dimension mapping.

```python
from fragile.shaolin.graph import InteractiveGraph, nodes_as_df, edges_as_df
import networkx as nx

# Create graph
G = nx.DiGraph()
G.add_nodes_from([(i, {'reward': r, 'depth': d}) for i, r, d in walker_data])
G.add_edges_from([(parent[i], i, {'weight': w}) for i, w in edge_data])

# Layout
pos = nx.spring_layout(G)  # or create_graphviz_layout(G)

# Convert to DataFrames
df_nodes = nodes_as_df(G, pos)
df_edges = edges_as_df(G)

# Interactive viz
viz = InteractiveGraph(
    df_nodes,
    df_edges,
    n_cols=3,
)

# Dimension mapping for BOTH nodes and edges
# node_size, node_color, node_alpha
# edge_color, edge_alpha, edge_line_width

dashboard = viz.panel()
dashboard.show()
```

**Key difference from InteractiveDataFrame**: Supports separate dimension mapping for nodes vs edges.

---

## 7. Control Widgets

### FaiRunner - Play/Pause Controls

**Purpose**: Control algorithm execution with UI buttons.

**From control.py**:

```python
from fragile.shaolin.control import FaiRunner

runner = FaiRunner(
    fai=your_algorithm,
    n_steps=1000,
    plot=your_streaming_plot,  # Optional: auto-update on each step
)

control_panel = runner.__panel__()

# Provides:
# - Play button: Run continuously
# - Pause button: Stop execution
# - Step button: Execute one step
# - Reset button: Reset to initial state
# - Progress bar: Track completion
# - Sleep slider: Control update speed
# - Summary table: Live metrics
```

#### Integration Example

```python
import panel as pn
from fragile.shaolin.control import FaiRunner
from fragile.shaolin.streaming_fai import InteractiveFai

# Setup
fai = FractalTree(...)
viz = InteractiveFai(fai)

# Create streaming plot
from fragile.shaolin.stream_plots import Scatter
scatter = Scatter(data=pd.DataFrame(fai.to_dict()))

# Link runner to plot
runner = FaiRunner(fai, n_steps=1000, plot=viz)

# Compose dashboard
dashboard = pn.Column(
    pn.pane.Markdown("## Fractal AI Explorer"),
    runner.__panel__(),        # Controls at top
    viz.__panel__(),           # Visualization below
)

dashboard.show()
```

#### Periodic Callback

FaiRunner uses Panel's periodic callback for smooth updates:

```python
def __panel__(self):
    pn.state.add_periodic_callback(self.run, period=1)  # Check every 1ms
    # ...

def run(self):
    if not self.is_running:
        return
    self.fai.step_tree()
    self.curr_step += 1
    self.progress.value = self.curr_step
    summary = pd.DataFrame(self.fai.summary(), index=[0])
    self.table.value = summary
    if self.plot is not None:
        self.plot.send(self.fai)  # Update visualization
    time.sleep(self.sleep_val.value)
```

**Key pattern**: Set `is_running` flag from button callbacks, check flag in periodic callback.

---

## 8. Practical Examples

### Example 1: Basic Streaming Scatter Plot

**Goal**: Visualize walker positions updating in real-time.

```python
import panel as pn
import pandas as pd
from fragile.shaolin.stream_plots import Scatter

# Initialize
scatter = Scatter(data=pd.DataFrame({'x': [], 'y': []}))

scatter.opts(
    width=600,
    height=400,
    size=5,
    alpha=0.7,
    color='blue',
)

# Dashboard
dashboard = pn.pane.HoloViews(scatter.plot)
dashboard.show()

# Simulation loop
for iteration in range(100):
    walkers.step()
    df = pd.DataFrame({
        'x': walkers.positions[:, 0],
        'y': walkers.positions[:, 1],
    })
    scatter.send(df)
    time.sleep(0.1)
```

### Example 2: Multi-Plot Dashboard

**Goal**: Show positions + reward curve + energy landscape.

```python
import panel as pn
from fragile.shaolin.stream_plots import Scatter, Curve, QuadMesh

# Three plots
positions = Scatter(data=pd.DataFrame({'x': [], 'y': []}))
reward_curve = Curve(
    data=pd.DataFrame({'iteration': [], 'reward': []}),
    buffer_length=500,
)
energy = QuadMesh(n_points=30)

# Style
positions.opts(width=400, height=400, color='reward', cmap='viridis', colorbar=True)
reward_curve.opts(width=400, height=200, xlabel='Iteration', ylabel='Best Reward')
energy.opts(width=400, height=400, cmap='plasma', colorbar=True)

# Layout
dashboard = pn.Column(
    pn.pane.Markdown("## Swarm Optimization"),
    pn.Row(
        pn.pane.HoloViews(positions.plot),
        pn.pane.HoloViews(energy.plot),
    ),
    pn.pane.HoloViews(reward_curve.plot),
)

dashboard.show()

# Update loop
for i in range(100):
    swarm.step()

    # Update positions
    positions.send(pd.DataFrame({
        'x': swarm.x[:, 0],
        'y': swarm.x[:, 1],
        'reward': swarm.reward,
    }))

    # Update reward curve
    reward_curve.send(pd.DataFrame({
        'iteration': [i],
        'reward': [swarm.reward.max()],
    }))

    # Update energy landscape
    energy.send((
        swarm.x[:, 0],
        swarm.x[:, 1],
        swarm.potential,
    ))
```

### Example 3: Interactive DataFrame with Tap Events

**Goal**: Click walkers to inspect their state.

```python
from fragile.shaolin.dataframe import InteractiveDataFrame
import panel as pn

# Create visualization
df = pd.DataFrame({
    'x': walkers.x[:, 0],
    'y': walkers.x[:, 1],
    'reward': walkers.reward,
    'distance': walkers.distance,
    'id': range(len(walkers.x)),
})

viz = InteractiveDataFrame(df, n_cols=3)

# Define tap handler
def on_walker_click(ix, df):
    walker_id = df.loc[ix, 'id']
    walker_reward = df.loc[ix, 'reward']
    return pn.pane.Markdown(f"""
    ### Walker {walker_id}
    - **Reward**: {walker_reward:.2f}
    - **Position**: ({df.loc[ix, 'x']:.2f}, {df.loc[ix, 'y']:.2f})
    - **Distance**: {df.loc[ix, 'distance']:.2f}
    """)

# Bind tap event
info_panel = viz.bind_tap(on_walker_click)

# Layout
dashboard = pn.Row(
    viz.__panel__(),
    pn.Column(
        pn.pane.Markdown("## Walker Info"),
        info_panel,
    ),
)

dashboard.show()
```

### Example 4: Montezuma's Revenge Visualization

**From montezuma.py**:

```python
from fragile.montezuma import FractalTree
from fragile.shaolin.stream_plots import RGB, QuadMesh
import panel as pn

# Initialize Fractal Tree
fai = FractalTree(
    max_walkers=1000,
    env=montezuma_env,
    start_walkers=100,
    min_leafs=100,
)

# Plots
rgb_plot = RGB(data=fai.rgb[0])  # Game screen
visits_plot = QuadMesh(n_points=50)  # State visitation heatmap

rgb_plot.opts(xaxis=None, yaxis=None, width=320, height=420)
visits_plot.opts(cmap='hot', colorbar=True, width=400, height=400)

# Dashboard
dashboard = pn.Column(
    pn.pane.Markdown("## Montezuma's Revenge Explorer"),
    pn.Row(
        pn.pane.HoloViews(rgb_plot.plot),
        pn.pane.HoloViews(visits_plot.plot),
    ),
)

dashboard.show()

# Exploration loop
fai.reset()
for step in range(1000):
    fai.step_tree()

    # Update RGB (best walker's view)
    best_ix = fai.cum_reward.argmax()
    rgb_plot.send(fai.rgb[best_ix])

    # Update visits heatmap (aggregated exploration)
    visits = fai.calculate_visits_reward()
    visits_plot.send((
        fai.observ[:, 0],   # x positions
        fai.observ[:, 1],   # y positions
        visits,             # visit counts
    ))
```

**Key techniques**:
- `aggregate_visits()`: Downsample visit counts for visualization
- `calculate_visits_reward()`: Compute exploration bonus
- Overlaying exploration map with current game screen

### Example 5: Force Vector Visualization

**Goal**: Show force fields acting on walkers.

```python
from fragile.shaolin.stream_plots import Scatter, VectorField
import panel as pn
import numpy as np

# Data
positions = gas.x  # [N, 2]
forces = gas.calculate_force()  # [N, 2]

# Scatter for positions
scatter = Scatter(data=pd.DataFrame({'x': positions[:, 0], 'y': positions[:, 1]}))
scatter.opts(size=8, alpha=0.7, color='blue')

# VectorField for forces
scale = 0.1  # Adjust arrow length
vectors = VectorField(
    data=pd.DataFrame({
        'x0': positions[:, 0],
        'y0': positions[:, 1],
        'x1': positions[:, 0] + forces[:, 0] * scale,
        'y1': positions[:, 1] + forces[:, 1] * scale,
    }),
    scale=scale,
)
vectors.opts(line_color='red', line_width=2, alpha=0.5)

# Overlay
overlay = scatter.plot * vectors.plot
dashboard = pn.pane.HoloViews(overlay)
dashboard.show()

# Update loop
for step in range(100):
    gas.step()

    positions = gas.x
    forces = gas.calculate_force()

    scatter.send(pd.DataFrame({'x': positions[:, 0], 'y': positions[:, 1]}))
    vectors.send(pd.DataFrame({
        'x0': positions[:, 0],
        'y0': positions[:, 1],
        'x1': positions[:, 0] + forces[:, 0] * scale,
        'y1': positions[:, 1] + forces[:, 1] * scale,
    }))
```

---

## 9. Advanced Patterns

### Pattern 1: Synchronized Multi-View Updates

**Use Case**: Update multiple linked plots from single data source.

```python
from holoviews.streams import Pipe
import pandas as pd
import panel as pn

# Shared data stream
data_pipe = Pipe(data=initial_df)

# Create plots sharing the stream
from fragile.shaolin.stream_plots import Scatter, Histogram

scatter = Scatter(stream=data_pipe)
hist_x = Histogram(stream=data_pipe)
hist_y = Histogram(stream=data_pipe)

# Custom plot functions
def plot_scatter(data):
    return hv.Scatter(data, kdims=['x'], vdims=['y']).opts(width=400, height=400)

def plot_hist_x(data):
    return hv.Histogram(np.histogram(data['x'].values, bins=30)).opts(width=400)

def plot_hist_y(data):
    return hv.Histogram(np.histogram(data['y'].values, bins=30)).opts(width=400)

# DynamicMaps
scatter_map = hv.DynamicMap(plot_scatter, streams=[data_pipe])
hist_x_map = hv.DynamicMap(plot_hist_x, streams=[data_pipe])
hist_y_map = hv.DynamicMap(plot_hist_y, streams=[data_pipe])

# Layout
dashboard = pn.Column(
    pn.pane.HoloViews(scatter_map),
    pn.Row(
        pn.pane.HoloViews(hist_x_map),
        pn.pane.HoloViews(hist_y_map),
    ),
)

# Single update → all plots refresh
data_pipe.send(new_df)
```

### Pattern 2: Custom DimensionMapper

**Use Case**: Special transformation not covered by standard options.

```python
from fragile.shaolin.dimension_mapper import DimensionMapper
import panel as pn

class CustomDim(DimensionMapper):
    def __init__(self, df, name="custom", **kwargs):
        super().__init__(df, name, **kwargs)

        # Add custom transformation button
        self.custom_transform = pn.widgets.Checkbox(
            name="Custom Transform",
            value=False,
        )

    @param.depends("custom_transform.value", watch=True)
    def get_value(self):
        # Standard processing
        super().get_value()

        # Apply custom transform
        if self.custom_transform.value:
            # Example: Sigmoidal transformation
            self.value = 1 / (1 + np.exp(-self.value))

    def panel(self):
        base_panel = super().panel()
        return pn.Column(base_panel, self.custom_transform)

# Usage
dims = Dimensions(
    df,
    n_cols=3,
    custom=CustomDim,
)
```

### Pattern 3: Tap Event with State Update

**Use Case**: Click point → update algorithm → refresh visualization.

```python
from fragile.shaolin.dataframe import InteractiveDataFrame
import panel as pn

viz = InteractiveDataFrame(df)

# State to track
selected_walker = pn.widgets.IntInput(name="Selected Walker", value=-1)

def on_tap(ix, df):
    walker_id = df.loc[ix, 'id']
    selected_walker.value = walker_id

    # Trigger algorithm action
    algorithm.focus_on_walker(walker_id)

    # Update visualization
    new_df = algorithm.get_state_df()
    viz.pipe.send(new_df)

    return pn.pane.Markdown(f"Selected walker {walker_id}")

info = viz.bind_tap(on_tap)

dashboard = pn.Column(
    viz.__panel__(),
    selected_walker,
    info,
)
```

### Pattern 4: Custom Colormap Integration

**Use Case**: Domain-specific color schemes.

```python
from fragile.shaolin.colormaps import ColorMap
import panel as pn

# Create custom colormap
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list(
    'custom',
    ['red', 'yellow', 'green'],  # Low → High
)

# Add to ColorMap widget
colormap_widget = ColorMap(default='viridis')
colormap_widget.all_cmaps['my_custom'] = custom_cmap

# Use in dimension mapping
from fragile.shaolin.dimension_mapper import ColorDim

color_dim = ColorDim(df, name='color')
color_dim.colormap_widget = colormap_widget

# Bind to plot
scatter = hv.Scatter(df, kdims=['x', 'y']).opts(
    color=color_dim.value,
    cmap=color_dim.cmap,
)
```

### Pattern 5: Real-Time Plot Overlays

**Use Case**: Combine multiple plot types for rich visualizations.

```python
from fragile.shaolin.stream_plots import QuadMesh, Scatter, VectorField
import panel as pn

# Layer 1: Energy landscape (QuadMesh)
landscape = QuadMesh(n_points=50)
landscape.opts(cmap='viridis', alpha=0.5, colorbar=True)

# Layer 2: Walker positions (Scatter)
positions = Scatter()
positions.opts(size=8, color='red', alpha=0.8)

# Layer 3: Velocity vectors (VectorField)
velocities = VectorField(scale=0.5)
velocities.opts(line_color='cyan', line_width=2, alpha=0.6)

# Overlay
overlay = landscape.plot * positions.plot * velocities.plot

# Dashboard
dashboard = pn.pane.HoloViews(overlay)
dashboard.show()

# Update all layers
for step in range(100):
    swarm.step()

    landscape.send((swarm.x[:, 0], swarm.x[:, 1], swarm.potential))
    positions.send(pd.DataFrame({'x': swarm.x[:, 0], 'y': swarm.x[:, 1]}))

    v_scale = 0.1
    velocities.send(pd.DataFrame({
        'x0': swarm.x[:, 0],
        'y0': swarm.x[:, 1],
        'x1': swarm.x[:, 0] + swarm.v[:, 0] * v_scale,
        'y1': swarm.x[:, 1] + swarm.v[:, 1] * v_scale,
    }))
```

---

## 10. Best Practices Checklist

### When to Use Shaolin

✅ **Good Fit**:
- Real-time algorithm visualization
- Exploratory data analysis of algorithm states
- Interactive parameter tuning
- Debugging swarm/population-based algorithms
- Comparing multiple runs

❌ **Not Ideal**:
- Static paper figures (use matplotlib directly)
- Extremely high-frequency updates (> 100 Hz)
- 3D visualizations (limited HoloViews support - use Plotly extension)

### Streaming vs Static Plots

| Criterion | Use StreamingPlot | Use Static HoloViews |
|-----------|-------------------|----------------------|
| Data changes during execution | ✅ | ❌ |
| Need play/pause controls | ✅ | ❌ |
| One-time visualization | ❌ | ✅ |
| Publication figures | ❌ | ✅ |
| Interactive exploration | ✅ | ✅ (with DynamicMap) |

### Performance Considerations

#### ✅ **Do**:
- **Downsample large datasets**: `df.sample(1000)` before sending
- **Use Buffer for time series**: Automatic windowing
- **Batch updates**: Update once per iteration, not per walker
- **Appropriate resolution**: QuadMesh n_points=30-50 usually sufficient
- **Hide unused columns**: `ignore_cols=('states', 'metadata')`

#### ❌ **Don't**:
- Send updates > 60 Hz (bottleneck is rendering, not data)
- Keep all history in Pipe (use Buffer with limit)
- Create new plots each frame (reuse and `.send()`)
- Render invisible plots (use `visible=False` in Panel)

### Code Organization

```python
# ✅ GOOD: Separate concerns
class MyAlgorithm:
    def step(self):
        # Pure algorithm logic
        pass

    def to_dict(self):
        # Export state as DataFrame-ready dict
        return {'x': self.positions[:, 0], ...}

    def summary(self):
        # Export summary metrics
        return {'iteration': self.i, 'best_reward': self.reward.max()}

class MyViz:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.scatter = Scatter(data=pd.DataFrame(algorithm.to_dict()))

    def update(self):
        df = pd.DataFrame(self.algo.to_dict())
        self.scatter.send(df)

# ❌ BAD: Mixed concerns
class MyAlgorithm:
    def step(self):
        # ... algorithm logic ...

        # Visualization logic embedded
        self.scatter.send(pd.DataFrame({...}))
        self.update_ui()
```

### Debugging Tips

#### Issue: Plot not updating

```python
# Check 1: Is data changing?
print(df.head())

# Check 2: Is send() being called?
def send_with_log(data):
    print(f"Sending data: {len(data)} rows")
    plot.send(data)

# Check 3: Is stream connected?
print(plot.data_stream)  # Should not be None

# Check 4: Backend loaded?
import holoviews as hv
print(hv.Store.current_backend)  # Should be 'bokeh' or 'matplotlib'
```

#### Issue: Slow rendering

```python
# Check 1: How much data?
print(len(df))  # If > 10k, downsample

# Check 2: Update frequency
import time
start = time.time()
plot.send(df)
print(f"Send took {time.time() - start:.3f}s")  # Should be < 0.1s

# Fix: Reduce update rate
if iteration % 10 == 0:  # Update every 10 steps
    plot.send(df)

# Fix: Downsample
plot.send(df.sample(min(1000, len(df))))
```

#### Issue: Dimension mapper shows no columns

```python
# Check: Are columns numeric?
print(df.dtypes)

# Fix: Convert to numeric
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')

# Check: String columns are ignored
mapper = SizeDim(df, ignore_string_cols=True)
print(mapper.valid_cols)  # Should exclude string columns
```

### Common Patterns Summary

#### **Pattern 1**: Algorithm + Streaming Plot
```python
scatter = Scatter(data=df)
for step in range(n_steps):
    algo.step()
    scatter.send(pd.DataFrame(algo.to_dict()))
```

#### **Pattern 2**: Algorithm + Interactive Dashboard
```python
viz = InteractiveDataFrame(algo.to_df())
dashboard = viz.__panel__()
dashboard.show()

# Later updates
viz.pipe.send(algo.to_df())
```

#### **Pattern 3**: Algorithm + Controls
```python
runner = FaiRunner(algo, n_steps=1000, plot=viz)
dashboard = pn.Column(runner.__panel__(), viz.__panel__())
```

#### **Pattern 4**: Multi-Plot Composition
```python
plots = [
    Scatter(data=df).opts(...),
    Curve(data=history).opts(...),
    QuadMesh(data=None).opts(...),
]

dashboard = pn.Column(
    *[pn.pane.HoloViews(p.plot) for p in plots]
)
```

#### **Pattern 5**: Tap Events
```python
viz = InteractiveDataFrame(df)

def on_click(ix, df):
    # Process click
    return pn.pane.Markdown(f"Clicked {ix}")

info = viz.bind_tap(on_click)
dashboard = pn.Row(viz.__panel__(), info)
```

---

## Summary

### Key Takeaways

1. **DimensionMapper** = Column → Visual Property with UI controls
2. **StreamingPlot** = Real-time plot updates via Pipe/Buffer
3. **Interactive Classes** = Full dashboards (DataFrame/Graph/Fai)
4. **Separation of Concerns** = Algorithm logic ≠ Visualization logic
5. **Composition** = Build complex dashboards from simple components

### Workflow Template

```python
# 1. Run algorithm
algo = MyAlgorithm()

# 2. Export data
df = pd.DataFrame(algo.to_dict())

# 3. Create visualization
viz = InteractiveDataFrame(df, n_cols=3)

# 4. (Optional) Add controls
runner = FaiRunner(algo, n_steps=1000, plot=viz)

# 5. Compose dashboard
dashboard = pn.Column(
    pn.pane.Markdown("## My Algorithm"),
    runner.__panel__(),
    viz.__panel__(),
)

# 6. Serve
dashboard.show()  # Opens browser

# 7. Update loop (if not using FaiRunner)
for step in range(n_steps):
    algo.step()
    viz.pipe.send(pd.DataFrame(algo.to_dict()))
```

### Further Reading

- **HoloViews Docs**: https://holoviews.org/
- **Panel Docs**: https://panel.holoviz.org/
- **Bokeh Docs**: https://docs.bokeh.org/
- **Example Notebooks**: See `notebooks/` in Fragile repo
