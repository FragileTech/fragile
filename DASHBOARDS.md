# Dashboard Usage Guide

All dashboards now run **without automatically opening a browser**. You can manually navigate to the provided URL in your browser.

## Available Dashboards

### 1. Gas Visualization Dashboard
**File:** `src/fragile/experiments/gas_visualization_dashboard.py`

```bash
python -m fragile.experiments.gas_visualization_dashboard
```

**URL:** http://localhost:5007

**Features:**
- Real-time Gas simulation visualization
- Interactive parameter controls
- Multiple visualization modes (scatter, heatmap, trajectories)
- Vector field overlays (velocity, force)


### 2. Fluid Dynamics Dashboard
**File:** `src/fragile/experiments/fluid_dynamics_dashboard.py`

```bash
python -m fragile.experiments.fluid_dynamics_dashboard
```

**URL:** http://localhost:5007

**Features:**
- Fluid simulation presets (Taylor-Green, Lid-Driven Cavity, Kelvin-Helmholtz)
- Velocity and vorticity field visualization
- Stream function and divergence analysis
- Conservation law validation
- Taylor-Green analytical comparison


### 3. Voronoi Evolution Viewer
**File:** `src/fragile/experiments/voronoi_evolution_viewer.py`

```bash
python -m fragile.experiments.voronoi_evolution_viewer
```

**URL:** http://localhost:5006

**Features:**
- Voronoi tessellation evolution
- Cell-based swarm dynamics
- Adaptive partitioning visualization


## Programmatic Usage

All dashboards can also be used programmatically:

```python
import holoviews as hv
hv.extension('bokeh')

# Gas Visualization Dashboard
from fragile.experiments.gas_visualization_dashboard import create_app
app = create_app()
app.show(port=5007, open=False)  # Doesn't auto-open browser

# Fluid Dynamics Dashboard
from fragile.experiments.fluid_dynamics_dashboard import create_fluid_dashboard
dashboard = create_fluid_dashboard()
dashboard.show(port=5008, open=False)

# Voronoi Evolution Viewer
from fragile.experiments.voronoi_evolution_viewer import create_app
viewer = create_app()
viewer.show(port=5009, open=False)
```

## Configuration

### Disable Browser Auto-Open
All dashboards now include `open=False` by default. To re-enable auto-opening:

```python
app.show(port=5007, open=True)  # Will open browser automatically
```

### Custom Ports
Each dashboard can run on a custom port:

```python
app.show(port=YOUR_PORT, open=False)
```

### Jupyter Notebook Integration
To display dashboards inline in Jupyter notebooks:

```python
import panel as pn
pn.extension()

app = create_app()
app  # Display inline
```

## Notes

- Dashboards use Panel (built on Bokeh) for interactive visualization
- Default ports: 5006 (Voronoi), 5007 (Gas/Fluid)
- Multiple dashboards can run simultaneously on different ports
- Press `Ctrl+C` to stop the server
