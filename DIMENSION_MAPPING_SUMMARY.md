# 3D Visualization Dimension Mapping Implementation

## Summary

Successfully implemented interactive dimension mapping controls for the dashboard's 3D particle viewer. Users can now map any spatial dimension (dim_0, dim_1, dim_2, dim_3 for 4D) or Monte Carlo time to the plot axes (X, Y, Z) and color channel.

## What Was Implemented

### 1. New Parameters
Added to `SwarmConvergence3D` class:
- `x_axis_dim`: Select dimension for X axis (default: "dim_0")
- `y_axis_dim`: Select dimension for Y axis (default: "dim_1")
- `z_axis_dim`: Select dimension for Z axis (default: "dim_2")
- Updated `color_metric`: Now includes spatial dimensions and MC time options

### 2. Dynamic Dimension Options
The available dimension options automatically adapt based on the simulation data:
- **3D simulations**: Shows dim_0, dim_1, dim_2, mc_time
- **4D simulations**: Shows dim_0, dim_1, dim_2, dim_3, mc_time

### 3. Helper Methods Added

#### `_extract_dimension(dim_spec, frame, positions_all, alive)`
Extracts coordinate values based on dimension specification:
- Spatial dimensions (dim_0, dim_1, etc.): Returns position data
- MC time: Returns frame index (constant for all walkers)

#### `_get_color_values(frame, positions_all, alive)`
Extracts color values with support for:
- Spatial dimensions (dim_0, dim_1, etc.)
- MC time
- Existing metrics (fitness, reward, radius, constant)

#### `_get_axis_ranges(frame)`
Determines appropriate axis ranges:
- Spatial dimensions: Uses bounds_extent (default ±10)
- MC time: Uses [0, n_recorded-1]

#### `_axis_label(dim_spec)`
Generates human-readable axis labels:
- "dim_0" → "Dimension 0 (X)"
- "dim_1" → "Dimension 1 (Y)"
- "dim_2" → "Dimension 2 (Z)"
- "dim_3" → "Dimension 3 (T)"
- "mc_time" → "Monte Carlo Time (frame)"

#### `_build_delaunay_trace_mapped(frame, positions_all, alive, positions_mapped)`
Renders Delaunay edges using mapped coordinates to ensure edges appear correctly in transformed space.

### 4. Updated Core Methods

#### `set_history(history)`
- Detects simulation dimensionality (history.d)
- Updates available dimension options dynamically
- Resets invalid selections to sensible defaults
- Displays dimension count in status

#### `_make_figure(frame)`
- Uses dimension mapping for all coordinate extraction
- Applies mapped coordinates to scatter plot
- Updates axis labels to show dimension names
- Adjusts axis ranges based on dimension type
- Passes mapped coordinates to Delaunay edge renderer

### 5. UI Enhancements
- Added dimension mapping dropdowns at the top of the control panel
- Added informational alert explaining dimension mapping
- Customized widget labels ("X Axis", "Y Axis", "Z Axis", "Color By")
- Controls automatically update when history is loaded

## How to Use

### Basic Usage

1. **Load or run a simulation** in the dashboard
2. **Navigate to the "Simulation" tab** to see the 3D particle viewer
3. **Use the dimension mapping dropdowns** at the top of the controls:
   - **X Axis**: Choose which dimension maps to the X axis
   - **Y Axis**: Choose which dimension maps to the Y axis
   - **Z Axis**: Choose which dimension maps to the Z axis
   - **Color By**: Choose how particles are colored

### Use Cases

#### Use Case 1: View 4D Spatial Distribution
**Setup:**
- Run a 4D simulation (dims=4)
- X → dim_0 (x)
- Y → dim_1 (y)
- Z → dim_3 (t, Euclidean time)
- Color → dim_2 (z)

**Result:** See XY spatial distribution across Euclidean time slices, colored by Z position.

#### Use Case 2: Track Temporal Evolution
**Setup:**
- X → dim_0 (x)
- Y → dim_1 (y)
- Z → mc_time
- Color → fitness

**Result:** See spatial distribution with time on Z axis, colored by fitness. Animation shows vertical "stack" building up as Monte Carlo time progresses.

#### Use Case 3: View All Time Slices
**Setup:**
- X → dim_0 (x)
- Y → dim_1 (y)
- Z → mc_time
- Color → mc_time
- Advance to the last frame

**Result:** See all walker positions stacked in time dimension, creating a trajectory visualization.

#### Use Case 4: Focus on Specific Dimension
**Setup:**
- X → dim_3 (t, Euclidean time)
- Y → dim_3 (same dimension)
- Z → mc_time
- Color → fitness

**Result:** Creates a 2D view projecting Euclidean time vs Monte Carlo time, useful for understanding temporal correlations.

## Technical Details

### Dimension Types

1. **Spatial Dimensions (dim_0, dim_1, dim_2, dim_3)**
   - Extracted from `history.x_final[frame, :, dim_idx]`
   - Uses bounds_extent for axis ranges
   - Available dimensions depend on simulation configuration

2. **Monte Carlo Time (mc_time)**
   - Constant value (frame index) for all walkers at a given frame
   - Range: [0, n_recorded-1]
   - Useful for visualizing temporal evolution

### Axis Range Handling

- **Spatial dimensions**: Fixed range [-bounds_extent, +bounds_extent] when `fix_axes=True`
- **MC time**: Fixed range [0, n_recorded-1]
- **Auto mode**: When `fix_axes=False`, uses Plotly's automatic scaling

### Delaunay Edge Rendering

- Edges are computed based on original spatial positions
- Edges are rendered in the mapped coordinate space
- This ensures connectivity appears correct even when viewing non-standard projections

### Parameter Reactivity

All dimension mapping parameters use Panel's `param.watch` system:
- Changes trigger immediate plot updates
- No need to reload data or restart the viewer
- Smooth interactive experience

## Testing

Comprehensive tests verify:
- ✓ Dimension extraction for spatial dims and MC time
- ✓ Dynamic dimension options update (3D vs 4D)
- ✓ Color value extraction with dimension support
- ✓ Axis range calculation
- ✓ Axis label generation
- ✓ Figure creation with various mappings

Run tests with:
```bash
python test_dimension_mapping.py
```

## Files Modified

- **src/fragile/fractalai/qft/dashboard.py**
  - Added dimension mapping parameters (lines 62-80)
  - Updated `set_history()` method (lines 152-185)
  - Added helper methods (lines ~280-400)
  - Modified `_make_figure()` method (lines ~490-570)
  - Added `_build_delaunay_trace_mapped()` method
  - Updated `panel()` method (lines ~578-620)

## Future Enhancements

Potential improvements for future development:

1. **Velocity Mapping**: Add support for mapping velocity dimensions from `history.v_before_clone`
2. **Computed Dimensions**: Add derived quantities (velocity magnitude, kinetic energy, angular momentum)
3. **2D Projections**: Option to collapse to 2D view for high-dimensional data
4. **Time Trails**: Show past positions as fading trails connecting frames
5. **Multi-Frame View**: Display multiple time slices simultaneously with different colors
6. **Custom Expressions**: Allow users to define custom dimension expressions (e.g., "dim_0 + dim_1")

## Notes

- The default mapping (dim_0, dim_1, dim_2 → X, Y, Z) preserves backward compatibility
- All existing functionality remains unchanged when using defaults
- The feature gracefully handles edge cases (invalid dimensions, missing data)
- Performance impact is minimal as coordinate extraction is vectorized with NumPy
