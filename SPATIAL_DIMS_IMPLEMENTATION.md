# Spatial Dimensions Configuration Implementation

## Overview

Implemented configurable spatial dimensions (2D or 3D spatial) for the QFT dashboard, with Euclidean time always added as an additional dimension. This makes debugging and visualization easier while maintaining full analysis capabilities.

## Key Changes

### 1. User-Facing UI

**Location**: Dashboard sidebar, Simulation section (before benchmark selection)

**New Control**: Spatial dimension selector with two options:
- **2D spatial + time (3D total)** - Easier to visualize and debug
- **3D spatial + time (4D total)** - Default, original behavior

**Benefits**:
- Easier visualization in 2D mode (2 spatial + time/MC axes in 3D viewer)
- Faster Voronoi computations (O(N^(d/2)) scaling)
- Simpler debugging of swarm dynamics

### 2. Architecture Changes

#### A. Core Configuration (`gas_config_panel.py`)

Added `spatial_dims` parameter to `GasConfigPanel`:
```python
spatial_dims = param.ObjectSelector(
    default=3,
    objects=[2, 3],
    doc="Number of spatial dimensions (Euclidean time added as extra)"
)
```

Updated `create_qft_config()` to accept `spatial_dims` parameter.

#### B. Simulation Configuration (`simulation.py`)

Updated `PotentialWellConfig` to auto-compute total dimensions:
```python
@dataclass
class PotentialWellConfig:
    spatial_dims: int = 3  # Number of spatial dimensions
    dims: int = field(init=False)  # Computed as spatial_dims + 1

    def __post_init__(self):
        self.dims = self.spatial_dims + 1  # Always add Euclidean time
```

**Example**:
- `spatial_dims=2` → `dims=3` (2 spatial + 1 time)
- `spatial_dims=3` → `dims=4` (3 spatial + 1 time)

#### C. Voronoi Tessellation (`euclidean_gas.py`, `benchmarks.py`)

Updated `compute_voronoi_tessellation()` calls to exclude time dimension:
```python
# For QFT mode (d>=3), use d-1 spatial dims (exclude Euclidean time)
spatial_dims = self.d - 1 if self.d >= 3 else None

voronoi_data = compute_voronoi_tessellation(
    positions=state.x,
    alive=alive_mask,
    bounds=self.bounds,
    spatial_dims=spatial_dims,  # Exclude time from neighbor calculation
    ...
)
```

**Why**: Voronoi neighbors should be based on spatial proximity only, not temporal separation.

#### D. Baryon Filtering (`correlator_channels.py`, `dashboard.py`)

Added automatic filtering of baryon channels in 2D mode:
```python
def compute_all_channels(
    history,
    channels=None,
    config=None,
    spatial_dims=None,  # NEW parameter
):
    # Filter out baryon channels in 2D mode (they require d=3)
    if spatial_dims is not None and spatial_dims < 3:
        channels = [ch for ch in channels if ch not in {"nucleon"}]
```

**Reason**: Baryon operators require 3D color space (SU(3) color symmetry).

### 3. Feature Compatibility

#### Fully Supported in 2D Mode

✅ **Electroweak Tab** - All electroweak correlators work
✅ **Strong Force Tab (partial)**:
  - Meson channels (quark-antiquark pairs) ✓
  - Glueball channels (field configurations) ✓
  - ~~Baryon channels~~ (auto-disabled in 2D)
✅ **Voronoi tessellations** - 2D Voronoi diagrams, neighbor graphs, cell volumes
✅ **Visualization** - 3D viewer with 2 spatial + time/MC axes
✅ **All benchmarks** - Including Voronoi-based benchmarks

#### Not Supported in 2D Mode

❌ **Baryon observables** - Require 3D spatial (SU(3) color)
  - Nucleon channel automatically filtered
  - No computation attempted (prevents errors)

### 4. Dimension Constraints

**Voronoi Requirement**: Minimum 2 spatial dimensions
- ✅ 2D spatial + 1 time = 3D total (supported)
- ✅ 3D spatial + 1 time = 4D total (supported)
- ❌ 1D spatial + 1 time = 2D total (NOT supported - Voronoi fails)

**Why**: `scipy.spatial.Voronoi` (Qhull backend) requires minimum 2D for tessellations.

## Usage

### From Dashboard

1. Open QFT dashboard
2. Navigate to sidebar → Simulation section
3. Select spatial dimensions:
   - "2D spatial + time (3D total)" for easier debugging
   - "3D spatial + time (4D total)" for full physics
4. Run simulation
5. Analyze results:
   - Electroweak tab: ✓ All features work
   - Strong Force tab: ✓ Mesons and glueballs work (no baryons in 2D)

### From Code

```python
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

# Create 2D spatial configuration
config_2d = GasConfigPanel.create_qft_config(spatial_dims=2)

# Create 3D spatial configuration (default)
config_3d = GasConfigPanel.create_qft_config(spatial_dims=3)
```

### Simulation Configuration

```python
from fragile.fractalai.qft.simulation import PotentialWellConfig

# 2D spatial + 1 time = 3D total
config = PotentialWellConfig(
    spatial_dims=2,  # dims auto-computed as 3
    n_walkers=200,
    n_steps=300,
)

# 3D spatial + 1 time = 4D total
config = PotentialWellConfig(
    spatial_dims=3,  # dims auto-computed as 4
    n_walkers=200,
    n_steps=300,
)
```

## Testing

Run the verification script to test all features:

```bash
python test_spatial_dims_simple.py
```

**Tests**:
1. ✓ PotentialWellConfig auto-computes dims = spatial_dims + 1
2. ✓ Baryon channels filtered in 2D mode
3. ✓ Voronoi tessellation works in 2D, 3D, and QFT mode (3D spatial)

## Technical Details

### Data Flow

**Before** (hardcoded 4D):
```
GasConfigPanel(dims=3)
  → Simulation(dims=4)
    → Voronoi(all 4 dims) ❌ includes time in neighbors
      → Analysis(time_dim=3)
```

**After** (configurable):
```
GasConfigPanel(spatial_dims={2,3})
  → Simulation(dims=spatial_dims+1)  # 3 or 4
    → Voronoi(spatial_dims={2,3}) ✓ excludes time
      → Analysis(time_dim=spatial_dims, auto-detect)
```

### Files Modified

| File | Changes |
|------|---------|
| `gas_config_panel.py` | Added `spatial_dims` parameter and selector |
| `simulation.py` | Added `spatial_dims` field, auto-compute dims |
| `dashboard.py` | Updated config creation, dimension mapping |
| `euclidean_gas.py` | Pass `spatial_dims` to Voronoi calls |
| `benchmarks.py` | Pass `spatial_dims=None` (use all provided dims) |
| `correlator_channels.py` | Filter baryon channels in 2D mode |

### Backward Compatibility

✅ **Fully backward compatible**:
- Default behavior unchanged (3D spatial + time = 4D total)
- Old code using `dims` parameter still works
- `create_qft_config(dims=3)` deprecated but functional

## Performance Impact

**2D Mode Benefits**:
- Faster Voronoi: O(N^1) vs O(N^(3/2))
- Smaller data: 25% reduction in position storage
- Faster visualization: 2D projections easier to render

**No Performance Penalty**:
- 3D mode unchanged
- No overhead when not using 2D mode
- Dimension checks are simple integer comparisons

## Future Enhancements

Potential improvements:
- Add 2D-specific particle observables (e.g., vortex detection)
- Optimize 2D correlator computations
- Add 2D-specific visualizations (contour plots, density fields)
- Support mixed dimensionality (e.g., 2D spatial + 2D internal)

## Notes

- The time dimension is always Euclidean (not Minkowski)
- Time evolves like spatial coordinates during dynamics
- Analysis can reinterpret any dimension as "time" via dimension mapping
- Voronoi neighbors computed on spatial dimensions only (critical for QFT)
