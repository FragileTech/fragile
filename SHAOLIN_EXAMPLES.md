# Fragile Shaolin - Quick Reference Code Examples

## 1. Basic Setup

```python
import panel as pn
import holoviews as hv
import pandas as pd
from fragile.shaolin.stream_plots import Scatter, Curve, VectorField

# Initialize Panel and HoloViews
pn.extension()
hv.extension('bokeh')
```

---

## 2. Real-Time Position Scatter Plot

```python
import numpy as np
from fragile.shaolin.stream_plots import Scatter

# Initialize scatter plot
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

# Stream updates in optimization loop
for step in range(100):
    x = np.random.randn(50)
    y = np.random.randn(50)
    data = pd.DataFrame({"x": x, "y": y})
    scatter.send(data)

# Display
layout = pn.pane.HoloViews(scatter.plot)
layout.servable()
```

---

## 3. Overlaying Position and Velocity

```python
from fragile.shaolin.stream_plots import Scatter, VectorField

# Initialize both plots
position_plot = Scatter(
    data=pd.DataFrame({"x": [], "y": []}),
    bokeh_opts={"color": "blue", "size": 8, "alpha": 0.7, "height": 600, "width": 600}
)

velocity_plot = VectorField(
    data=pd.DataFrame({"x0": [], "y0": [], "x1": [], "y1": []}),
    scale=0.1,
    bokeh_opts={"line_color": "cyan", "line_width": 2, "alpha": 0.5}
)

# Overlay plots
combined = (position_plot.plot * velocity_plot.plot).opts(
    title="Particle Swarm Dynamics",
    xlabel="x-coordinate",
    ylabel="y-coordinate"
)

# Stream updates
for step in range(100):
    # Update positions
    x = np.random.randn(50)
    y = np.random.randn(50)
    pos_data = pd.DataFrame({"x": x, "y": y})
    position_plot.send(pos_data)
    
    # Update velocities
    vx = np.random.randn(50) * 0.1
    vy = np.random.randn(50) * 0.1
    vel_data = pd.DataFrame({
        "x0": x,
        "y0": y,
        "x1": x + vx,
        "y1": y + vy
    })
    velocity_plot.send(vel_data)

layout = pn.pane.HoloViews(combined)
layout.servable()
```

---

## 4. Tracking Metrics Over Time

```python
from fragile.shaolin.stream_plots import Curve

# Create curve for tracking alive walkers
alive_curve = Curve(
    data=pd.DataFrame({"step": [], "count": []}),
    buffer_length=1000,
    bokeh_opts={
        "color": "green",
        "line_width": 3,
        "height": 300,
        "width": 800,
        "ylabel": "Alive Walkers",
        "xlabel": "Step"
    }
)

# Stream data points
for step in range(100):
    n_alive = 50 - step // 2  # Simulated decay
    data = pd.DataFrame({"step": [step], "count": [n_alive]})
    alive_curve.send(data)

layout = pn.pane.HoloViews(alive_curve.plot)
layout.servable()
```

---

## 5. Multi-Metric Dashboard

```python
from fragile.shaolin.stream_plots import Curve

# Create multiple curves
alive_curve = Curve(
    data=pd.DataFrame({"step": [], "alive": []}),
    buffer_length=500,
    bokeh_opts={"color": "green", "line_width": 3}
)

reward_curve = Curve(
    data=pd.DataFrame({"step": [], "reward": []}),
    buffer_length=500,
    bokeh_opts={"color": "orange", "line_width": 3}
)

# Overlay curves
metrics = (alive_curve.plot + reward_curve.plot).cols(1)

# Build dashboard
dashboard = pn.Column(
    pn.pane.Markdown("# Algorithm Metrics"),
    metrics
)

# Stream data
for step in range(100):
    alive_data = pd.DataFrame({"step": [step], "alive": [50 - step // 2]})
    reward_data = pd.DataFrame({"step": [step], "reward": [np.sin(step / 10)]})
    
    alive_curve.send(alive_data)
    reward_curve.send(reward_data)

dashboard.servable()
```

---

## 6. Gas Algorithm Visualization

```python
from fragile.euclidean_gas import EuclideanGas, EuclideanGasParams
from fragile.shaolin.gas_viz import GasVisualization
from fragile.benchmarks import Sphere

# Setup algorithm
benchmark = Sphere(d=2)
params = EuclideanGasParams(
    N=50,
    d=2,
    potential=benchmark,
    langevin__gamma=1.0,
    langevin__beta=2.0,
    langevin__delta_t=0.1,
    cloning__sigma_x=0.5
)
gas = EuclideanGas(params)

# Setup visualization
viz = GasVisualization(
    bounds=benchmark.bounds,
    position_color="blue",
    velocity_color="cyan",
    plot_size=600,
    track_alive=True
)

# Run optimization loop
for step in range(50):
    state = gas.step()
    n_alive = (benchmark.bounds.contains(state.x)).sum().item()
    viz.update(state, n_alive=n_alive)

# Display
layout = viz.create_layout()
layout.servable()
```

---

## 7. Boundary-Aware Visualization

```python
from fragile.shaolin.gas_viz import BoundaryGasVisualization

# Setup boundary visualization
viz = BoundaryGasVisualization(
    bounds=benchmark.bounds,
    in_bounds_color="blue",
    out_bounds_color="red",
    velocity_color="cyan",
    plot_size=600
)

# Run with automatic boundary detection
for step in range(50):
    state = gas.step()
    viz.update(state)  # Automatically detects in/out of bounds

layout = viz.create_layout()
layout.servable()
```

---

## 8. Interactive Parameter Configuration

```python
import panel as pn
from fragile.shaolin import EuclideanGasParamSelector

pn.extension()

# Create parameter selector
selector = EuclideanGasParamSelector()

# Access parameters
print(f"N walkers: {selector.n_walkers}")
print(f"Gamma: {selector.gamma}")

# Build dashboard
dashboard = pn.Column(
    pn.pane.Markdown("# Euclidean Gas Configuration"),
    selector.swarm_section,
    selector.langevin_section,
    selector.cloning_section,
    selector.benchmark_section
)

dashboard.servable()
```

---

## 9. Interactive DataFrame Exploration

```python
from fragile.shaolin.dataframe import InteractiveDataFrame
import numpy as np

# Create sample data
df = pd.DataFrame({
    "x": np.random.randn(100),
    "y": np.random.randn(100),
    "reward": np.random.rand(100),
    "iteration": np.arange(100)
})

# Create interactive viewer
idf = InteractiveDataFrame(
    df,
    default_x_col="iteration",
    default_y_col="reward",
    n_cols=3
)

# Display
dashboard = pn.Column(
    pn.pane.Markdown("# Interactive Data Explorer"),
    idf.layout(),
    idf.view()
)

dashboard.servable()
```

---

## 10. Dynamic Dimension Mapping

```python
from fragile.shaolin.dimension_mapper import Dimensions, SizeDim, ColorDim, AlphaDim

# Create dimension mappers
dims = Dimensions(
    df,
    n_cols=2,
    size=(SizeDim, {"value_range": (1, 15), "default": 5}),
    color=ColorDim,
    alpha=AlphaDim
)

# Build widget panel
widget_panel = pn.Column(
    pn.pane.Markdown("## Visual Property Mapping"),
    dims.panel()
)

# Display
widget_panel.servable()
```

---

## 11. Colormap Selector Widget

```python
from fragile.shaolin.colormaps import ColorMap

# Create colormap selector
cmap_widget = ColorMap(default="viridis")

# Get selected colormap
selected = cmap_widget.value

# Display
layout = pn.Column(
    pn.pane.Markdown("# Select Colormap"),
    cmap_widget.view()
)

layout.servable()
```

---

## 12. Algorithm Runner with Controls

```python
from fragile.shaolin.control import FaiRunner

# Setup algorithm and visualization
gas = EuclideanGas(params)
viz = GasVisualization()

# Create runner with controls
runner = FaiRunner(gas, n_steps=100, plot=viz)

# Display dashboard
dashboard = pn.Column(
    pn.pane.Markdown("# Algorithm Control Panel"),
    runner
)

dashboard.servable()
```

---

## 13. Colored Scatter by Category

```python
from fragile.shaolin.stream_plots import Scatter

# Create scatter with category colors
scatter = Scatter(
    data=pd.DataFrame({
        "x": [],
        "y": [],
        "category": []
    }),
    bokeh_opts={
        "height": 600,
        "width": 600,
        "tools": ["hover"]
    }
)

# Stream categorical data
for step in range(100):
    x = np.random.randn(50)
    y = np.random.randn(50)
    cat = np.random.choice(["A", "B", "C"], 50)
    
    data = pd.DataFrame({"x": x, "y": y, "category": cat})
    scatter.send(data)

# Update plot with color mapping
scatter.opts(color="category", cmap={"A": "blue", "B": "red", "C": "green"})

layout = pn.pane.HoloViews(scatter.plot)
layout.servable()
```

---

## 14. Histogram Distribution Visualization

```python
from fragile.shaolin.stream_plots import Histogram

# Create histogram
hist = Histogram(
    data=None,
    n_bins=30,
    bokeh_opts={
        "height": 400,
        "width": 600,
        "ylabel": "Frequency"
    }
)

# Stream data
for step in range(10):
    # Generate random data
    values = np.random.normal(0, 1, 1000)
    hist.send(values)

layout = pn.pane.HoloViews(hist.plot)
layout.servable()
```

---

## 15. Energy Landscape Visualization

```python
from fragile.shaolin.stream_plots import Landscape2D

# Create landscape plot
landscape = Landscape2D(
    data=None,
    contours=True,
    n_points=50
)

# Stream data points
for step in range(10):
    x = np.random.uniform(-5, 5, 100)
    y = np.random.uniform(-5, 5, 100)
    z = np.sin(x) * np.cos(y)  # Example energy function
    
    landscape.send((x, y, z))

layout = pn.pane.HoloViews(landscape.plot)
layout.servable()
```

---

## 16. Full Dashboard Example

```python
import panel as pn
from fragile.euclidean_gas import EuclideanGas, EuclideanGasParams
from fragile.shaolin.gas_viz import GasVisualization
from fragile.shaolin.stream_plots import Curve
from fragile.benchmarks import Rastrigin

pn.extension()

# Setup
benchmark = Rastrigin(d=2)
params = EuclideanGasParams(
    N=100, d=2, potential=benchmark,
    langevin__gamma=1.0, langevin__beta=2.0,
    cloning__sigma_x=0.5
)
gas = EuclideanGas(params)

# Visualizations
gas_viz = GasVisualization(bounds=benchmark.bounds)
best_reward = Curve(
    data=pd.DataFrame({"step": [], "best": []}),
    buffer_length=500,
    bokeh_opts={"color": "purple", "line_width": 2}
)

# Build dashboard
dashboard = pn.Column(
    pn.pane.Markdown("# Rastrigin Optimization with Euclidean Gas"),
    pn.Row(
        pn.pane.HoloViews(gas_viz.position_stream.plot * gas_viz.velocity_stream.plot),
        pn.pane.HoloViews(best_reward.plot)
    )
)

# Run optimization
best_values = []
for step in range(100):
    state = gas.step()
    n_alive = (benchmark.bounds.contains(state.x)).sum().item()
    
    gas_viz.update(state, n_alive=n_alive)
    
    best_reward_val = state.reward.min().item()
    best_values.append(best_reward_val)
    
    reward_data = pd.DataFrame({
        "step": [step],
        "best": [best_reward_val]
    })
    best_reward.send(reward_data)

dashboard.servable()
```

---

## Common Patterns

### Pattern: Adaptive Update Frequency
```python
# Update visualization every N steps to avoid browser overload
UPDATE_EVERY = 10

for step in range(1000):
    state = gas.step()
    
    if step % UPDATE_EVERY == 0:
        viz.update(state)
        print(f"Step {step}: updated visualization")
```

### Pattern: Tap Event Callback
```python
from fragile.shaolin.dataframe import InteractiveDataFrame

idf = InteractiveDataFrame(df)

def on_point_tap(index, dataframe):
    selected_row = dataframe.iloc[index]
    print(f"Clicked: {selected_row}")

callback = idf.bind_tap(on_point_tap)
panel = pn.Column(idf.view(), callback)
```

### Pattern: Multi-Dimensional Streaming
```python
# Stream multiple related plots together
position = Scatter(...)
velocity = VectorField(...)

# Sync updates
for step in range(100):
    pos_data = compute_positions(step)
    vel_data = compute_velocities(step)
    
    position.send(pos_data)
    velocity.send(vel_data)
```

### Pattern: Conditional Styling
```python
# Update colors based on conditions
scatter.opts(
    color="reward",  # Column name for continuous color
    cmap="viridis"   # Colormap
)

# Or discrete mapping
scatter.opts(
    color="status",  # Column with categorical values
    cmap={"active": "green", "inactive": "red"}
)
```

---

## Troubleshooting

### Issue: Plot not updating
```python
# Make sure to call send() with proper DataFrame format
data = pd.DataFrame({"x": x_values, "y": y_values})
scatter.send(data)  # Don't forget this!
```

### Issue: Too many points slowing down rendering
```python
# Downsample before sending
if len(positions) > 5000:
    indices = np.random.choice(len(positions), 5000, replace=False)
    positions = positions[indices]
```

### Issue: Memory leak with streaming
```python
# Set reasonable buffer length for time series
curve = Curve(data=df, buffer_length=10000)  # Keep only last 10k points
```

