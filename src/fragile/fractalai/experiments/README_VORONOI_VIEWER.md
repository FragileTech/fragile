# Voronoi Tessellation Evolution Viewer

Interactive dashboard for visualizing how Voronoi tessellations evolve during EuclideanGas 2D simulations.

## Features

### Domain Bounds
- **Visualization Domain**: [-6, 6]² (matching `prepare_background` default)
- **Plot Axis Limits**: xlim and ylim are fixed to (-6, 6) to show only the valid domain
- **Bounds Filtering**: Only walkers with positions inside the domain are included in tessellation
- **Guaranteed**: All Voronoi cell centers are within the visualization bounds

### Left Panel: Simulation Configuration
- **General Parameters**: Number of walkers (N), simulation steps
- **Langevin Dynamics**: Friction (γ), temperature (β), time step (Δt), forces
- **Cloning & Selection**: Jitter (σ_x), algorithmic distance weight (λ_alg), restitution
- **Initialization**: Initial positions, velocity scale, spatial bounds

### Center Panel: Voronoi Tessellation Visualization
- **Background**: Potential landscape (multimodal Gaussian mixture)
- **Voronoi Cells**: Spatial tessellation showing walker territories
- **Walkers**: Individual walker positions
- **Playback Controls**:
  - **Play/Pause**: Watch tessellation evolution in real-time
  - **Time Slider**: Manually navigate to specific timesteps
  - **Speed Control**: Adjust playback speed (1-30 fps)

### Visualization Controls
- **Show Walkers**: Toggle walker position markers (Voronoi cell centers / Delaunay vertices)
- **Show Voronoi Edges**: Toggle Voronoi tessellation boundary lines (black solid lines)
- **Show Voronoi Vertices**: Toggle Voronoi vertex markers (orange points, NOT cell centers)
- **Show Delaunay Dual**: Toggle Delaunay triangulation edges (blue dashed lines)
- **Color by**: Radio button selection for coloring mode
  - **None**: Default cyan walkers, no coloring
  - **Fitness**: Color walkers and cells by fitness potential values (coolwarm colormap)
  - **Ricci Curvature**: Color Voronoi cell polygons AND walker points by Ricci scalar (RdBu_r colormap)
    - Red: Negative curvature (hyperbolic geometry)
    - White: Zero curvature (flat geometry)
    - Blue: Positive curvature (spherical geometry)

### Statistics Panel
Displays real-time metrics:
- Number of Voronoi cells
- Neighbor statistics (avg, min, max)
- Scutoid classification (prisms vs scutoids)
- **Ricci scalar curvature** (mean, std, min, max)

## Usage

### Launch the Dashboard

```bash
# From project root
python -m fragile.experiments.voronoi_evolution_viewer
```

The dashboard will open in your browser at `http://localhost:5006`

### Workflow

1. **Configure Parameters** (left panel)
   - Adjust simulation parameters as desired
   - Recommended defaults work well for initial exploration

2. **Run Simulation**
   - Click "Run Simulation" button
   - Wait for completion message

3. **Explore Evolution**
   - **Play Mode**: Click play button to watch tessellation evolve automatically
   - **Manual Mode**: Use time slider to move through specific timesteps
   - **Speed Adjustment**: Use speed slider to control playback rate (1-30 fps)
   - Toggle visualization options to see different aspects
   - Check statistics panel for quantitative metrics

4. **Iterate**
   - Modify parameters and run again
   - Compare different configurations

## Mathematical Framework

The viewer builds Voronoi tessellations from `RunHistory` data using the new `ScutoidHistory2D` API:

```python
from fragile.core import create_scutoid_history

# After simulation completes
scutoid_history = create_scutoid_history(run_history)
scutoid_history.build_tessellation()
scutoid_history.compute_ricci_scalars()
```

### Voronoi Tessellation

At each timestep `t`, the spatial domain is partitioned into Voronoi cells:

```
Vor_i(t) = {x ∈ X : d(x, x_i) ≤ d(x, x_j) for all j ≠ i}
```

Where `x_i(t)` are walker positions.

### Delaunay Triangulation (Voronoi Dual)

The Delaunay triangulation is the **geometric dual** of the Voronoi tessellation:

- **Voronoi**: Partitions space into regions based on nearest walker
  - Vertices: Points equidistant from 3+ walkers
  - Edges: Boundaries between walker territories
  - Cells: Regions closer to walker `i` than any other

- **Delaunay**: Connects walkers whose Voronoi cells share an edge
  - Vertices: Walker positions (Voronoi cell centers)
  - Edges: Connect walkers with adjacent Voronoi cells (shown as blue dashed lines)
  - Triangles: Formed by walkers whose Voronoi vertices meet

**Duality Property**: If walkers `i` and `j` have Voronoi cells sharing an edge, then there exists a Delaunay edge connecting `x_i` and `x_j`.

This dual structure is visualized when "Show Delaunay Dual" is enabled.

### Scutoid Cells

Between consecutive timesteps, scutoids connect Voronoi cells:
- **Prisms**: No neighbor topology change (same neighbors at top/bottom)
- **Scutoids**: Neighbor topology changes (cloning events)

### Ricci Scalar Curvature

The viewer computes discrete Ricci scalar using the **deficit angle method** from Regge calculus:

```
R(x_i) = δ_i / (C(2) * Perimeter_i)
```

Where:
- `δ_i = 2π - Σ θ_triangle`: Deficit angle at vertex i
- `C(2) = 2π`: 2D dimension constant
- `θ_triangle`: Interior angle in each incident Delaunay triangle

This provides a discrete approximation to the continuum Ricci scalar of the emergent Riemannian geometry.

## Implementation Details

### Architecture

```python
class VoronoiEvolutionViewer:
    def __init__(self, dims=2):
        # Initialize components
        self.gas_config = GasConfig(...)  # Parameter panel
        self.scutoid_history = None       # Tessellation data
        self.time_slider = ...             # Time control

    def _on_simulation_complete(self, history):
        # Build tessellation from RunHistory
        self.scutoid_history = create_scutoid_history(history)
        self.scutoid_history.build_tessellation()
        self.scutoid_history.compute_ricci_scalars()

    def _update_plot(self):
        # Render Voronoi tessellation at current timestep
        cells = self.scutoid_history.voronoi_cells[t_idx]
        # ... draw edges, walkers, centers
```

### Data Flow

1. **User configures parameters** → `GasConfig`
2. **Run simulation** → `EuclideanGas.run()` → `RunHistory`
3. **Build tessellation** → `ScutoidHistory2D.build_tessellation()`
4. **Compute curvature** → `ScutoidHistory2D.compute_ricci_scalars()`
5. **Visualize** → Extract Voronoi cells at timestep → Render with HoloViews

## Dependencies

- `holoviews` (with Bokeh backend): Interactive plots
- `panel`: Dashboard framework
- `scipy.spatial`: Voronoi/Delaunay computation
- `fragile.core`: ScutoidHistory2D, RunHistory
- `fragile.experiments`: GasConfig, prepare_background

## Limitations

- **2D only**: Currently only supports 2D spatial domains
- **Bounded cells**: Infinite Voronoi cells (at boundaries) are skipped
- **Performance**: Large N (>500 walkers) may slow down rendering
- **Domain bounds**: Fixed to [-6, 6]² (walkers outside this domain are excluded from tessellation)

## Future Enhancements

- [ ] 3D Voronoi visualization (using Plotly backend)
- [ ] Animation mode (auto-play through timesteps)
- [ ] Export tessellation data (save Voronoi cells, statistics)
- [ ] Curvature heatmap overlay
- [ ] Cloning event markers (show which cells underwent topology changes)
- [ ] Comparison mode (show two simulations side-by-side)

## References

- Mathematical framework: `old_docs/source/14_scutoid_geometry_framework.md`
- Ricci scalar computation: §5.4.1 (Deficit Angle Convergence)
- Scutoid implementation: `src/fragile/core/scutoids.py`
