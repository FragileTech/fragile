# Dashboard Slider UnsetValueError Fix - Implementation Summary

## Problem Fixed

Fixed a critical bug where the fractal gas visualization dashboard would crash on launch with:
```
UnsetValueError: Slider(id='p1606', ...).value doesn't have a value set
```

This occurred when launching the standard dashboard (non-QFT mode) due to three widgets being created without initial values.

## Root Cause

Three widgets in operator `__panel__()` methods were created without initial `value` parameters:

1. **`viscous_neighbor_threshold`** (KineticOperator) - EditableFloatSlider without value
2. **`viscous_degree_cap`** (KineticOperator) - EditableFloatSlider without value
3. **`rho`** (FitnessOperator) - FloatInput without value

These parameters have `default=None` and `allow_None=True` in their param definitions, but the Panel/Bokeh widgets require explicit values even if `None`.

## Changes Made

### 1. Fixed Widget Definitions

**File: `src/fragile/fractalai/core/kinetic_operator.py`**

Added `"value": None` to two widgets:

- Line 310: `viscous_neighbor_threshold` EditableFloatSlider
- Line 326: `viscous_degree_cap` EditableFloatSlider

**File: `src/fragile/fractalai/core/fitness.py`**

Added `"value": None` to one widget:

- Line 633: `rho` FloatInput

### 2. Created Comprehensive Test Suite

**File: `tests/test_dashboard_launch.py`** (new)

Created 11 tests covering:
- Standard and QFT dashboard creation
- Dashboard serialization (simulates browser connection)
- Individual operator panel widget initialization
- GasConfigPanel in both standard and QFT modes
- Comprehensive widget value verification

**File: `tests/integration/test_dashboard_server.py`** (new)

Created integration tests that:
- Actually start the Bokeh server
- Verify HTTP connections work
- Test both standard and QFT modes

## Verification Results

### Unit Tests
```bash
uv run pytest tests/test_dashboard_launch.py -v
```

**Result:** ✅ All 11 tests pass

Tests verify:
- ✓ test_standard_dashboard_creates_without_error
- ✓ test_qft_dashboard_creates_without_error
- ✓ test_standard_dashboard_serializes
- ✓ test_qft_dashboard_serializes
- ✓ test_kinetic_operator_panel_widgets_have_values
- ✓ test_fitness_operator_panel_widgets_have_values
- ✓ test_companion_selection_panel_widgets_have_values
- ✓ test_clone_operator_panel_widgets_have_values
- ✓ test_gas_config_panel_standard_mode
- ✓ test_gas_config_panel_qft_mode
- ✓ test_all_widgets_have_initial_values

### Manual Verification
```bash
# Standard dashboard
uv run -m fragile.fractalai.experiments.gas_visualization_dashboard
```
✅ Launches successfully without UnsetValueError

```bash
# QFT dashboard
uv run -m fragile.fractalai.experiments.gas_visualization_dashboard --qft
```
✅ Launches successfully without UnsetValueError

## Why This Fix Works

**Before:**
```python
"viscous_neighbor_threshold": {
    "type": pn.widgets.EditableFloatSlider,
    # ... other params ...
    # NO "value" KEY - Bokeh creates slider with Unset value
},
```

When Panel renders this widget:
1. Creates EditableFloatSlider instance
2. Doesn't set `.value` property (because no `"value"` key in dict)
3. Bokeh tries to serialize slider to JSON for browser
4. Encounters `.value` property that's Unset (not even `None`, but literally uninitialized)
5. Raises `UnsetValueError`

**After:**
```python
"viscous_neighbor_threshold": {
    "type": pn.widgets.EditableFloatSlider,
    # ... other params ...
    "value": None,  # ✓ Explicitly set value to None
},
```

Now:
1. Creates EditableFloatSlider instance
2. Sets `.value = None` (explicitly)
3. Bokeh serializes with `value: null` in JSON
4. ✓ Success - `None` is a valid value for `allow_None=True` parameters

## Why These Parameters Allow None

These are **optional advanced features**:

1. **`viscous_neighbor_threshold`**: Optional threshold for viscous coupling. If `None`, all neighbors contribute. If set (e.g., 0.75), only "strong" neighbors contribute.

2. **`viscous_degree_cap`**: Optional cap on viscous degree. If `None`, unlimited neighbors. If set, saturates multi-neighbor coupling.

3. **`rho`**: Localization scale for fitness. If `None`, uses global (mean-field) statistics. If finite, uses local statistics.

Setting them to `None` is valid and intentional - it means "use default behavior".

## Pattern for New Widgets

When adding new optional parameters with Panel widgets, always include:

```python
"my_optional_param": {
    "type": pn.widgets.SomeWidget,
    "value": None,  # ← CRITICAL for optional params with allow_None=True
    # ... other config ...
},
```

Or if the param has a non-None default:

```python
"my_param": {
    "type": pn.widgets.SomeWidget,
    "value": 1.0,  # ← Use the param's default value
    # ... other config ...
},
```

## Future Prevention

The test suite in `tests/test_dashboard_launch.py` will catch this issue automatically:
- Any new widget without a proper `value` parameter will fail serialization tests
- Tests run on every PR/commit
- Prevents regression

## Success Criteria - All Met ✅

- ✅ Dashboard launches without UnsetValueError (both standard and QFT modes)
- ✅ All three problematic widgets have `"value": None` added
- ✅ Comprehensive test suite catches widget initialization errors
- ✅ All tests pass
- ✅ Manual verification confirms fix works
- ✅ Documentation created for widget creation pattern
