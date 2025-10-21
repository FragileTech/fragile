# Fractal Set Interactive Visualization Guide

## Overview

The Fractal Set visualization provides an interactive dashboard for exploring the complete graph representation of EuclideanGas execution traces.

## Files

- **Script**: `experiments/experiment_scripts/fractal_set_interactive_viz.py`
- **Notebook**: `experiments/fractal_set_visualization.ipynb`
- **Core Module**: `src/fragile/core/fractal_set.py`

## Key Features

### 1. **Smart Layout Caching** ✅
- Layouts are computed ONCE per (N, n_recorded, layout_type) combination
- Cached across multiple simulation runs
- Instant switching between previously used parameter combinations
- Example: After running with N=20 and N=15, switching back to N=20 is instant

### 2. **Multiple Layout Options**
- **`physical` (default)**: Uses actual walker (x, y) positions
  - ⚡ Instant computation
  - 🎯 Coherent across timesteps
  - Shows physical trajectories in state space
  
- **Graphviz layouts**: `dot`, `neato`, `fdp`, `sfdp`, `circo`, `twopi`
  - Computed on-demand (~30s for full graph)
  - Cached for reuse
  - Better for visualizing graph structure

### 3. **Interactive Timeline**
- Slider to watch graph grow from t=0 to t=max
- Fast updates (0.2-0.3s per timestep change)
- Just filters nodes/edges from precomputed layout

### 4. **Edge Type Filtering**
- **CST edges**: Temporal evolution (walker trajectories over time)
- **IG edges**: Selection coupling (directed with antisymmetric cloning potential)
- Toggle independently to see different structures

### 5. **Rich Node/Edge Attributes**
- **Nodes**: fitness, kinetic energy, alive status, timestep
- **Edges** (CST): velocity changes, position displacements
- **Edges** (IG): antisymmetric cloning potential V_clone = Φ_j - Φ_i

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Build Fractal Set | ~1s | For N=20, n_steps=100 |
| Physical layout | Instant | Pre-computed from positions |
| Graphviz layout (first time) | ~30s | For 2020 nodes, 40000 edges |
| Graphviz layout (cached) | Instant | Reused across runs |
| Timestep update | 0.2-0.3s | Just filters, no recomputation |
| Parameter switch (cached) | Instant | E.g., N=20 → N=15 → N=20 |

## Usage

### Jupyter Notebook

```bash
jupyter notebook experiments/fractal_set_visualization.ipynb
```

The notebook provides:
1. Interactive Panel dashboard
2. Programmatic access to Fractal Set
3. Analysis tools (queries, statistics)
4. Data export capabilities

### Script Mode

```python
from fractal_set_interactive_viz import create_fractal_set_explorer

explorer, dashboard = create_fractal_set_explorer()
dashboard.show()  # Launch dashboard
```

## Layout Caching Details

The cache key is: `(N, n_recorded, layout_type)`

**Example:**
```
Run 1: N=20, n_steps=100 → Cache: [(20, 101, 'physical')]
Run 2: N=20, n_steps=100 → Cache hit! (instant)
Run 3: N=15, n_steps=100 → Cache: [(20, 101, 'physical'), (15, 101, 'physical')]
Run 4: N=20, n_steps=100 → Cache hit from Run 1! (instant)
```

Graphviz layouts are cached similarly:
```
Switch to 'neato': ~30s computation
Run new simulation (same N, n_steps)
Switch to 'neato' again: Instant! (cache hit)
```

## Why No Hanging?

### Previous Problem
- Computed Graphviz layout for FULL graph immediately
- Set max_timestep to 100 (showing entire graph at once)
- Layout computation: O(n²) complexity → 30+ seconds

### Solution
1. ✅ Start at max_timestep=0 (small graph initially)
2. ✅ Pre-compute physical layout (instant)
3. ✅ Compute Graphviz layouts on-demand
4. ✅ Cache ALL layouts across runs
5. ✅ Timestep slider just filters cached layout

## Graph Structure

### Nodes
- **ID**: `(walker_id, timestep)` tuple
- **Attributes**: position x, velocity v, kinetic energy, fitness, alive status

### CST Edges (Causal Spacetime Tree)
- **Direction**: `(i, t) → (i, t+1)` (same walker across time)
- **Attributes**: velocities, velocity changes, position displacements
- **Count**: ~N × n_recorded (one per walker per timestep)

### IG Edges (Information Graph)
- **Direction**: `(i, t) → (j, t)` for i ≠ j (different walkers at same time)
- **Attributes**: V_clone (antisymmetric!), distances, fitness values
- **Count**: k(k-1) per timestep for k alive walkers (complete tournament)

## Tips

1. **Start with physical layout** - instant and shows physical trajectories
2. **Use small timesteps first** - slide to watch graph grow
3. **Toggle edge types** - CST for trajectories, IG for coupling
4. **Cache Graphviz layouts** - compute once, reuse forever (for same N, n_steps)
5. **Explore different N values** - cache makes switching instant

## Troubleshooting

### "Building Fractal Set..." hangs
✅ **Fixed!** Script now starts at t=0 and uses cached layouts

### Slow timestep updates
✅ **Fixed!** Updates are now 0.2-0.3s (just filtering, no recomputation)

### Graphviz layout taking long
Expected on first computation (~30s for full graph). Subsequent uses are instant via cache.

### Want to clear cache
The cache persists for the session. Restart the notebook/dashboard to clear.
