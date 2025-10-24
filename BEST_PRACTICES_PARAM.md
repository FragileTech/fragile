# Best Practices: Param + Panel for Dashboard Configuration

**Purpose**: This document teaches how to design dataclasses that map seamlessly to interactive UI dashboards using Panel and Param, following the patterns established in `dashboard_core_example.py` and `liquid_trader.py`.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Base Architecture: The PanelModel Pattern](#2-base-architecture-the-PanelModel-pattern)
3. [Parameter Management](#3-parameter-management)
4. [Widget Configuration Patterns](#4-widget-configuration-patterns)
5. [Layout Composition](#5-layout-composition)
6. [Serialization & Hydra Integration](#6-serialization--hydra-integration)
7. [Advanced Patterns](#7-advanced-patterns)
8. [Constants & Consistency](#8-constants--consistency)
9. [Step-by-Step Examples](#9-step-by-step-examples)
10. [Decision Trees](#10-decision-trees)

---

## 1. Core Philosophy

### Why This Pattern Works

The `PanelModel` pattern succeeds because it achieves **complete separation of concerns**:

```
┌─────────────────────┐
│  Business Logic     │  ← Pure computation parameters
│  (what to compute)  │     Example: beta=0.5, gamma=1.0
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Configuration API  │  ← Properties for organization
│  (how to organize)  │     Example: dict_param_names, widgets
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  UI Rendering       │  ← Panel widgets & layout
│  (how to display)   │     Example: __panel__(), FloatInput
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Serialization      │  ← Hydra-compatible dict
│  (how to persist)   │     Example: to_dict(), instantiate()
└─────────────────────┘
```

### Key Benefits

1. **Single Source of Truth**: Parameters defined once with `param.Parameter`
2. **Type Safety**: Param provides validation (bounds, types, constraints)
3. **Reactive UI**: Changes in UI automatically update Python objects
4. **Bidirectional Binding**: Changes in code update UI via `param.watch()`
5. **Composability**: Nest configurations arbitrarily deep
6. **Hydra Integration**: Serialize to/from YAML configs seamlessly
7. **No Boilerplate**: `__panel__()` auto-generates UI from parameter definitions

---

## 2. Base Architecture: The PanelModel Pattern

### Essential Structure

```python
import param
import panel as pn
from typing import Any

class PanelModel(param.Parameterized):
    # ============================================================
    # INTERNAL PARAMETERS (UI configuration, prefixed with _)
    # ============================================================
    _max_widget_width = param.Integer(
        default=800, bounds=(0, None), doc="Max widget width"
    )
    _n_widget_columns = param.Integer(
        default=2, bounds=(1, None), doc="Number of widget columns"
    )
    _target_ = param.String(doc="Target class for instantiation")

    # ============================================================
    # BUSINESS LOGIC PARAMETERS (public, no _ prefix)
    # ============================================================
    # (defined in subclasses)

    # ============================================================
    # PROPERTY API: Organize and filter parameters
    # ============================================================

    @property
    def dict_param_names(self) -> list[str]:
        """Names of parameters to serialize (excludes UI config)."""
        return list(
            set(list(self.param))
            - {"_max_widget_width", "_n_widget_columns", "name"}
        )

    @property
    def widgets(self) -> dict[str, dict]:
        """Widget configurations for parameters."""
        return {}  # Override in subclasses

    @property
    def widget_parameters(self) -> list[str]:
        """Parameters to display in UI (excludes _target_, etc.)."""
        ignore_params = {"_target_", "_max_widget_width", "_n_widget_columns"}
        return list(set(self.dict_param_names) - ignore_params)

    @property
    def default_layout(self):
        """Define default layout (grid-based)."""
        def new_class(cls, **kwargs):
            return type(type(cls).__name__, (cls,), kwargs)

        names = list(self.widget_parameters)
        ncols = (
            self._n_widget_columns
            if len(names) > self._n_widget_columns
            else len(names)
        )
        return new_class(
            pn.GridBox, ncols=ncols, max_width=self._max_widget_width
        )

    # ============================================================
    # SERIALIZATION: to_dict / set_values / instantiate
    # ============================================================

    def to_dict(self, parameters=None) -> dict[str, Any]:
        """Convert to dictionary (Hydra-compatible)."""
        if parameters is None:
            parameters = self.dict_param_names
        return {
            param: (
                getattr(self, param).to_dict()
                if hasattr(getattr(self, param), "to_dict")
                else getattr(self, param)
            )
            for param in parameters
        }

    def set_values(self, d, parameters=None) -> None:
        """Set values from dictionary."""
        if parameters is None:
            parameters = self.dict_param_names
        for key in parameters:
            if key in d and hasattr(getattr(self, key), "set_values"):
                getattr(self, key).set_values(d[key])
            elif key in d:
                setattr(self, key, d[key])

    def instantiate(self, **kwargs):
        """Instantiate the target class with current parameters."""
        d = self.to_dict()

        def _is_config(k):
            is_dashboard_conf = isinstance(getattr(self, k), PanelModel)
            is_list_of_conf = isinstance(getattr(self, k), list) and all(
                isinstance(item, PanelModel) for item in getattr(self, k)
            )
            return is_dashboard_conf or is_list_of_conf

        config_params = {
            k: (
                getattr(self, k).instantiate(**kwargs)
                if not isinstance(getattr(self, k), list)
                else [item.instantiate(**kwargs) for item in getattr(self, k)]
            )
            for k in d.keys()
            if _is_config(k)
        }
        raw_params = {
            k: v for k, v in d.items() if not _is_config(k) and k != "_target_"
        }
        all_params = {
            **instantiate(raw_params, _convert_="all"),
            **config_params,
        }
        if "_target_" in d:
            return get_object(self._target_)(**all_params)
        return all_params

    # ============================================================
    # RENDERING: __panel__() for UI
    # ============================================================

    def __panel__(self, parameters=None):
        """Render parameters in Panel UI."""
        if parameters is None:
            parameters = self.widget_parameters
        return pn.Param(
            self,
            show_name=False,
            parameters=parameters,
            widgets=self.process_widgets(self.widgets),
            default_layout=self.default_layout,
        )

    @classmethod
    def process_widgets(cls, widgets):
        """Ensure consistent widget naming and structure."""
        new_widgets = {}
        for k, v in widgets.items():
            widget_data = v if isinstance(v, dict) else {"type": v}
            if "name" not in widget_data:
                widget_data["name"] = k.replace("_", " ").capitalize()
            new_widgets[k] = widget_data
        return new_widgets
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Prefix internal params with `_`** | Clear visual separation between business logic and UI config |
| **Properties, not methods** | More Pythonic, enables lazy computation |
| **Recursive `to_dict()`** | Handles nested configs automatically |
| **`widget_parameters` property** | Easy to customize which params appear in UI |
| **`process_widgets()` classmethod** | Ensures consistent naming even with minimal config |
| **`default_layout` as property** | Can be overridden per-class without breaking subclasses |

---

## 3. Parameter Management

### Naming Conventions

```python
class MyConfig(PanelModel):
    # ✅ GOOD: Public business logic parameters (no prefix)
    temperature = param.Number(default=1.0, bounds=(0, 10))
    learning_rate = param.Number(default=0.01, bounds=(0, 1))

    # ✅ GOOD: Internal UI configuration (underscore prefix)
    _n_widget_columns = param.Integer(default=3)
    _max_widget_width = param.Integer(default=600)

    # ✅ GOOD: Special Hydra parameter
    _target_ = param.String(default="my_module.MyClass")

    # ❌ BAD: Mixing naming conventions
    _temperature = param.Number(...)  # Don't prefix business logic
    uiColumns = param.Integer(...)     # Don't use camelCase for internal
```

### Documentation Standards

**Rule**: Every parameter needs a `doc` string. For nested configs, include the class name prefix.

```python
class TradeMomentConfig(PanelModel):
    analysis_window_ms = param.Number(
        default=10.0,
        doc="TradeMomentConfig: Analysis window size in milliseconds",
        # ↑ Prefix helps when debugging or inspecting nested configs
    )

    minimal_trade_moment_distance = param.String(
        default=None,
        doc="Minimum distance between trade moments",
        # ↑ Clear description of what this parameter controls
    )
```

### Parameter Types & Common Patterns

| Param Type | Use Case | Example |
|------------|----------|---------|
| `param.Number` | Floats, can have bounds | `beta = param.Number(default=0.5, bounds=(0, 1))` |
| `param.Integer` | Integers, can have bounds | `N = param.Integer(default=100, bounds=(10, None))` |
| `param.String` | Text, paths, enum-like | `mode = param.String(default="train")` |
| `param.Boolean` | Flags, toggles | `enable_logging = param.Boolean(default=True)` |
| `param.Date` | Timestamps | `start_time = param.Date(default=pd.Timestamp("2024-01-01"))` |
| `param.Filename` | File paths | `config_file = param.Filename(default="./config.yaml")` |
| `param.Path` | Directory paths | `output_dir = param.Path(default="./outputs")` |
| `param.ClassSelector` | Nested configs | `optimizer = param.ClassSelector(class_=OptimizerConfig)` |
| `param.List` | Lists of items | `horizons = param.List(default=[1, 5, 10], item_type=int)` |

### Nested Configuration Pattern

```python
class OptimizerConfig(PanelModel):
    learning_rate = param.Number(default=0.01)
    momentum = param.Number(default=0.9)
    _target_ = param.String(default="torch.optim.SGD")

class TrainingConfig(PanelModel):
    batch_size = param.Integer(default=32)

    # ✅ Nested config with ClassSelector
    optimizer = param.ClassSelector(
        class_=OptimizerConfig,
        default=None,
        doc="Optimizer configuration"
    )
    _target_ = param.String(default="my_module.Trainer")

    def __init__(self, optimizer=None, **params):
        # ✅ Initialize nested config in __init__
        if optimizer is None:
            optimizer = OptimizerConfig()
        super().__init__(optimizer=optimizer, **params)
```

**Why this works**:
- `ClassSelector` ensures type safety
- Initialization in `__init__` provides default instance
- Recursive `to_dict()` handles serialization automatically
- `__panel__()` can render nested config with `self.optimizer.__panel__()`

---

## 4. Widget Configuration Patterns

### The `widgets` Property

The `widgets` property maps parameter names to widget specifications:

```python
class MyConfig(PanelModel):
    temperature = param.Number(default=1.0, bounds=(0, 10))
    mode = param.String(default="train")
    enable_cache = param.Boolean(default=True)

    @property
    def widgets(self):
        return {
            "temperature": {
                "type": pn.widgets.FloatInput,
                "width": 150,
                "format": "0.00",  # Two decimal places
                "step": 0.1,
                "start": 0.0,
            },
            "mode": {
                "type": pn.widgets.Select,
                "options": ["train", "eval", "test"],
                "width": 120,
            },
            "enable_cache": {
                "type": pn.widgets.Toggle,
                "width": 100,
            },
        }
```

### Widget Type Reference

| Widget Type | Best For | Key Options | Example |
|-------------|----------|-------------|---------|
| `pn.widgets.FloatInput` | Numeric input (floats) | `format`, `step`, `start`, `width` | Temperature, learning rate |
| `pn.widgets.IntInput` | Numeric input (ints) | `step`, `start`, `end`, `width` | Batch size, epochs |
| `pn.widgets.EditableFloatSlider` | Ranged floats with visual feedback | `start`, `end`, `step`, `width` | Proportions (0-1), weights |
| `pn.widgets.Toggle` | Boolean flags | `width`, `button_type` | Enable/disable features |
| `pn.widgets.Select` | Enum-like choices | `options`, `width` | Mode selection, optimizer type |
| `pn.widgets.RadioButtonGroup` | Exclusive choices with emphasis | `options`, `button_type` | Critical mode switches |
| `pn.widgets.DatetimePicker` | Timestamps | `width` | Start/end times |
| `pn.widgets.TextInput` | Short strings | `placeholder`, `width` | Names, labels |
| `pn.widgets.TextAreaInput` | Multi-line text | `placeholder`, `height`, `width` | Descriptions, code snippets |
| `pn.widgets.LiteralInput` | Python literals (lists, dicts) | `width` | Lists of values |

### Format Strings for Numeric Widgets

```python
"format": "0.00"      # Two decimal places: 3.14
"format": "0.0000"    # Four decimal places: 3.1416
"format": "0."        # Integer display: 42
"format": "0.00%"     # Percentage: 75.50%
"format": "0,0"       # Thousands separator: 1,000
"format": "0.0a"      # Short format: 1.2k, 3.5M
```

### Width Constants

**Establish global constants for consistency**:

```python
# At module level
INPUT_WIDTH = 95       # Standard input field width
SLIDER_WIDTH = 200     # Standard slider width

class MyConfig(PanelModel):
    @property
    def widgets(self):
        return {
            "learning_rate": {
                "type": pn.widgets.FloatInput,
                "width": INPUT_WIDTH,  # ✅ Use constant
            },
            "momentum": {
                "type": pn.widgets.EditableFloatSlider,
                "width": SLIDER_WIDTH,  # ✅ Use constant
            },
            "description": {
                "type": pn.widgets.TextAreaInput,
                "width": int(INPUT_WIDTH * 4),  # ✅ Scale from constant
            },
        }
```

### Complex Widget Configuration Example

From `liquid_trader.py`:

```python
class PriceNewsTicks(PanelModel):
    input_freq_hz = param.Number(
        default=3.0,
        doc="Input frequency in Hz",
        bounds=(0.001, None),
        step=0.01,
    )

    decay_rate_news_proportion = param.Number(
        default=0.1,
        step=0.1,
        bounds=(0.01, 1),
        doc="Proportion remaining after time period for news memory",
    )

    @property
    def widgets(self):
        return {
            "input_freq_hz": {
                "type": pn.widgets.FloatInput,
                "width": int(INPUT_WIDTH * 1.15),
                "start": 0.001,
                "step": 0.01,
                "name": "Input frequency (Hz)",  # ✅ Custom display name
                "format": "0.00",
            },
            "decay_rate_news_proportion": {
                "type": pn.widgets.EditableFloatSlider,
                "start": 0.01,
                "end": 1.0,
                "step": 0.1,
                "name": "Proportion remaining",  # ✅ Shorter, clearer name
                "width": SLIDER_WIDTH,
            },
        }
```

**Key Patterns**:
- **Custom names**: Override parameter name for better UI labels
- **Scaled widths**: Use `int(INPUT_WIDTH * factor)` for proportional sizing
- **Explicit bounds**: Duplicate `bounds` from param definition in widget for clarity
- **Format strings**: Match precision to use case (Hz needs 2 decimals)

---

## 5. Layout Composition

### Basic Layouts

Panel provides three main layout containers:

```python
# Column: Vertical stacking
pn.Column(widget1, widget2, widget3)

# Row: Horizontal arrangement
pn.Row(widget1, widget2, widget3)

# GridBox: Grid with automatic wrapping
pn.GridBox(widget1, widget2, widget3, ncols=2, max_width=800)
```

### The `__panel__()` Method

Default implementation auto-generates grid layout:

```python
def __panel__(self, parameters=None):
    """Basic rendering - uses GridBox with default_layout."""
    if parameters is None:
        parameters = self.widget_parameters
    return pn.Param(
        self,
        show_name=False,
        parameters=parameters,
        widgets=self.process_widgets(self.widgets),
        default_layout=self.default_layout,  # GridBox with ncols
    )
```

### Custom Layouts with Markdown Headers

```python
class MyConfig(PanelModel):
    # ... parameter definitions ...

    def __panel__(self, parameters=None):
        """Custom layout with sections."""
        return pn.Column(
            pn.pane.Markdown("### Core Parameters"),
            super().__panel__(parameters=["learning_rate", "batch_size"]),

            pn.pane.Markdown("### Advanced Options"),
            super().__panel__(parameters=["momentum", "weight_decay"]),
        )
```

### Card-Based Organization

Cards provide collapsible sections:

```python
def __panel__(self, parameters=None):
    return pn.Column(
        pn.Card(
            super().__panel__(parameters=["warmup_start", "train_start"]),
            title="Training Time Period",
            collapsed=False,  # Expanded by default
            width=250,
        ),
        pn.Card(
            super().__panel__(parameters=["test_start", "test_end"]),
            title="Testing Time Period",
            collapsed=True,   # Collapsed by default
            width=250,
        ),
    )
```

### Multi-Column Layouts

From `liquid_trader.py`:

```python
def create_inputs_and_moments_panel(self):
    """Two-column layout for related configs."""
    return pn.Row(
        pn.Column(
            pn.pane.Markdown("### Input params"),
            self.input_data.__panel__(),
            pn.pane.Markdown("### Trade moments params"),
            self.trade_moment.__panel__(),
        ),
        self.trade_moment.detector.__panel__(),
    )
```

### Nested Config Rendering

```python
class ParentConfig(PanelModel):
    child1 = param.ClassSelector(class_=ChildConfig1)
    child2 = param.ClassSelector(class_=ChildConfig2)

    def __panel__(self, parameters=None):
        return pn.Column(
            pn.pane.Markdown("## Parent Configuration"),

            # Render parent's own parameters
            super().__panel__(parameters=["parent_param1", "parent_param2"]),

            pn.pane.Markdown("### Child Config 1"),
            self.child1.__panel__(),  # ✅ Recursive rendering

            pn.pane.Markdown("### Child Config 2"),
            self.child2.__panel__(),
        )
```

---

## 6. Serialization & Hydra Integration

### The Serialization Triplet

Every `PanelModel` provides three methods:

```python
# 1. to_dict() - Python object → Dictionary
config = MyConfig(learning_rate=0.01, batch_size=32)
d = config.to_dict()
# → {"_target_": "my_module.MyClass", "learning_rate": 0.01, "batch_size": 32}

# 2. set_values() - Dictionary → Python object
config2 = MyConfig()
config2.set_values(d)
# config2.learning_rate == 0.01

# 3. instantiate() - Dictionary → Actual class instance
instance = config.instantiate()
# → Returns my_module.MyClass(learning_rate=0.01, batch_size=32)
```

### Hydra `_target_` Convention

The `_target_` parameter specifies the fully qualified class name:

```python
class OptimizerConfig(PanelModel):
    learning_rate = param.Number(default=0.01)
    momentum = param.Number(default=0.9)

    _target_ = param.String(default="torch.optim.SGD")
    # ↑ Hydra will instantiate torch.optim.SGD(**params)
```

When serialized:

```yaml
optimizer:
  _target_: torch.optim.SGD
  learning_rate: 0.01
  momentum: 0.9
```

### Special Type Serialization

Some Python types need special handling for Hydra:

#### Paths

```python
def to_dict(self, parameters=None):
    data = super().to_dict(parameters)
    data["output_folder"] = {
        "_target_": "pathlib.Path",
        "_args_": [self.output_folder],
    }
    return data
```

#### Datetimes

```python
def to_dict(self, parameters=None):
    datetime_format = "%Y-%m-%d %H:%M:%S"
    return {
        "_target_": self._target_,
        "start_time": self.start_time.strftime(datetime_format),
        # Hydra will parse string back to datetime via target class
    }
```

#### Timedeltas

```python
def to_dict(self, parameters=None):
    return {
        "_target_": self._target_,
        "delay": {
            "_target_": "pandas.Timedelta",
            "milliseconds": self.delay,
        },
        "horizons": [
            {"_target_": "pandas.Timedelta", "seconds": h}
            for h in self.horizons
        ],
    }
```

### `set_values()` with Special Types

Must handle both dict format (from Hydra) and direct values:

```python
def set_values(self, d, parameters=None):
    super().set_values(d, parameters=parameters)

    # Handle Path serialization
    if "output_folder" in d:
        if isinstance(d["output_folder"], dict):
            # From Hydra: {"_target_": "pathlib.Path", "_args_": ["/path"]}
            self.output_folder = d["output_folder"]["_args_"][0]
        else:
            # Direct value: "/path"
            self.output_folder = d["output_folder"]

    # Handle Timedelta serialization
    if "delay" in d:
        if isinstance(d["delay"], dict):
            # From Hydra: {"_target_": "pandas.Timedelta", "milliseconds": 100}
            self.delay = d["delay"]["milliseconds"]
        else:
            # Direct value: 100 (or pd.Timedelta object)
            self.delay = d["delay"]
```

### `instantiate()` Method

Handles nested configs recursively:

```python
def instantiate(self, **kwargs):
    d = self.to_dict()

    # Identify which params are PanelModel instances
    def _is_config(k):
        is_dashboard_conf = isinstance(getattr(self, k), PanelModel)
        is_list_of_conf = isinstance(getattr(self, k), list) and all(
            isinstance(item, PanelModel) for item in getattr(self, k)
        )
        return is_dashboard_conf or is_list_of_conf

    # Recursively instantiate nested configs
    config_params = {
        k: (
            getattr(self, k).instantiate(**kwargs)
            if not isinstance(getattr(self, k), list)
            else [item.instantiate(**kwargs) for item in getattr(self, k)]
        )
        for k in d.keys()
        if _is_config(k)
    }

    # Regular params (no nested configs)
    raw_params = {
        k: v for k, v in d.items() if not _is_config(k) and k != "_target_"
    }

    # Combine and instantiate via Hydra
    all_params = {
        **instantiate(raw_params, _convert_="all"),
        **config_params,
    }

    if "_target_" in d:
        return get_object(self._target_)(**all_params)
    return all_params
```

---

## 7. Advanced Patterns

### Conditional UI (Mode-Based Visibility)

**Use Case**: Show/hide parameter groups based on a mode selection.

```python
class PriceNewsTicks(PanelModel):
    mode = param.String(default="direct")

    # Parameters visible only in "fuzzy" modes
    decay_rate_news_proportion = param.Number(default=0.1)
    decay_rate_price_proportion = param.Number(default=0.1)

    def __init__(self, **params):
        super().__init__(**params)

        # Create mode toggle
        self._mode_toggle = pn.widgets.RadioButtonGroup(
            name="Mode",
            options={
                "Direct": "direct",
                "Fuzzy News": "fuzzy_news",
                "Fuzzy All": "fuzzy_all",
            },
            value=self.mode,
            button_type="primary",
        )

        # Watch for mode changes
        self._mode_toggle.param.watch(self._update_mode, "value")

    def _update_mode(self, event=None):
        """Update visibility based on mode."""
        if event is not None:
            self.mode = event.new

        if self.mode == "direct":
            self.price_decay.visible = False
            self.news_decay.visible = False
        elif self.mode == "fuzzy_news":
            self.price_decay.visible = False
            self.news_decay.visible = True
        elif self.mode == "fuzzy_all":
            self.price_decay.visible = True
            self.news_decay.visible = True

    def __panel__(self, parameters=None):
        # Create parameter groups
        self.news_decay = pn.Column(
            pn.pane.Markdown("### News Memory Decay"),
            super().__panel__(parameters=["decay_rate_news_proportion"]),
        )

        self.price_decay = pn.Column(
            pn.pane.Markdown("### Price Memory Decay"),
            super().__panel__(parameters=["decay_rate_price_proportion"]),
        )

        # Initialize visibility
        self._update_mode()

        return pn.Column(
            self._mode_toggle,
            self.news_decay,
            self.price_decay,
        )
```

**Key Technique**: Store Panel components as instance attributes (`self.news_decay`, `self.price_decay`) so `_update_mode()` can control their visibility.

### Custom File Selector Widget

**Use Case**: Need a file browser with clean UI and bidirectional binding.

```python
class FileInputWidget(param.Parameterized):
    """Widget for file selection with nice UI."""

    file_path = param.String(default="", doc="Path to the selected file")
    widget_name = param.String(default="File", doc="Name of the widget")

    def __init__(
        self,
        ref,                # Reference to parent object
        ref_name,           # Attribute name on parent to update
        widget_name="File",
        only_files=False,
        file_path=None,
        file_pattern="*",
        **params,
    ):
        self._ref = ref
        self._ref_name = ref_name

        # Initialize file path
        if file_path is None:
            file_path = str(getattr(ref, ref_name))
        else:
            file_path = str(file_path)
            setattr(ref, ref_name, file_path)

        super().__init__(
            widget_name=widget_name, file_path=file_path, **params
        )

        # Create file selector
        self.file_selector = pn.widgets.FileSelector(
            only_files=only_files,
            value=[file_path],
            file_pattern=file_pattern,
            directory="/data",  # Root directory
        )

        # Watch for changes
        self.file_selector.param.watch(self._update_file_path, "value")

        # Create header
        self._header = pn.Row(
            pn.pane.Markdown(f"### {self.widget_name}: {self.file_path}"),
        )

    def _update_file_path(self, event):
        """Bidirectional binding: UI → Python object."""
        if event.new and len(event.new) > 0:
            new_value = str(event.new[0])
            self.file_path = new_value
            self._header[0].object = f"### {self.widget_name}: {self.file_path}"
            setattr(self._ref, self._ref_name, new_value)  # Update parent

    def __panel__(self):
        """Render as collapsible card."""
        return pn.Card(
            self.file_selector,
            header=self._header,
            collapsed=True,  # Collapsed by default to save space
        )


# Usage in PanelModel
class MyConfig(PanelModel):
    data_file = param.Filename(default="./data.csv")

    def __init__(self, **params):
        super().__init__(**params)
        self.file_widget = FileInputWidget(
            ref=self,
            ref_name="data_file",
            widget_name="Data File",
            file_pattern="*.csv",
        )

    def __panel__(self, parameters=None):
        return pn.Column(
            self.file_widget,
            super().__panel__(),
        )
```

### Lists of Nested Configs

**Use Case**: Multiple items of the same config type (e.g., list of features).

```python
class FeatureConfig(PanelModel):
    window_ms = param.Number(default=100.0)
    _target_ = param.String(default="my_module.Feature")

class FeaturesConfig(PanelModel):
    features = param.List(
        default=None,
        item_type=PanelModel,
        doc="List of features"
    )
    _target_ = param.String(default="my_module.FeatureCollection")

    def __init__(self, features=None, **params):
        # Initialize with default features
        if features is None:
            features = [
                FeatureConfig(window_ms=50.0),
                FeatureConfig(window_ms=100.0),
                FeatureConfig(window_ms=200.0),
            ]
        super().__init__(features=features, **params)

    def to_dict(self, parameters=None):
        return {
            "_target_": self._target_,
            "features": [f.to_dict() for f in self.features],
        }

    def set_values(self, d, parameters=None):
        super().set_values(d, parameters=parameters)
        if "features" in d:
            for feature, feat_dict in zip(self.features, d["features"]):
                if isinstance(feature, PanelModel):
                    feature.set_values(feat_dict)

    def __panel__(self, parameters=None):
        return pn.Column(
            pn.pane.Markdown("### Features Configuration"),
            pn.Row(
                *[feature.__panel__() for feature in self.features]
            ),
        )
```

### Reactive Parameter Updates

**Use Case**: Update UI when Python values change programmatically.

```python
class MyConfig(PanelModel):
    learning_rate = param.Number(default=0.01)
    momentum = param.Number(default=0.9)

    def __init__(self, **params):
        super().__init__(**params)

        # Watch for learning rate changes
        self.param.watch(self._on_lr_change, "learning_rate")

    def _on_lr_change(self, event):
        """Auto-adjust momentum when learning rate changes."""
        if event.new > 0.1:
            self.momentum = 0.95  # High LR → high momentum
        else:
            self.momentum = 0.9   # Low LR → normal momentum
```

---

## 8. Constants & Consistency

### Global Constants (Define Once)

```python
# At module level (top of file)
INPUT_WIDTH = 95       # Standard input field
SLIDER_WIDTH = 200     # Standard slider
BUTTON_WIDTH = 120     # Standard button
TEXT_AREA_HEIGHT = 80  # Standard text area height

# Use throughout widget definitions
@property
def widgets(self):
    return {
        "param1": {"type": pn.widgets.FloatInput, "width": INPUT_WIDTH},
        "param2": {"type": pn.widgets.EditableFloatSlider, "width": SLIDER_WIDTH},
    }
```

### Naming Conventions Summary

| Item | Convention | Example |
|------|------------|---------|
| **Business logic params** | `snake_case`, no prefix | `learning_rate`, `batch_size` |
| **Internal UI params** | `snake_case`, `_` prefix | `_n_widget_columns`, `_max_widget_width` |
| **Hydra target param** | Special: `_target_` | `_target_ = "my_module.MyClass"` |
| **Widget config keys** | Match param names exactly | `"learning_rate": {...}` |
| **Display names** | Title case, spaces OK | `"name": "Learning Rate"` |
| **Instance attributes for UI** | `_` prefix, descriptive | `self._mode_toggle`, `self._header` |

### Docstring Standards

```python
class MyConfig(PanelModel):
    learning_rate = param.Number(
        default=0.01,
        bounds=(0, 1),
        step=0.001,
        doc="MyConfig: Learning rate for gradient descent"
        #   ^^^^^^^^  Class prefix for nested configs
        #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Clear description
    )
```

**Template**:
```
"{ClassName}: {clear description of what this controls}"
```

---

## 9. Step-by-Step Examples

### Example 1: Simple Configuration (Start Here)

**Goal**: Create a config for a simple machine learning optimizer.

```python
import param
import panel as pn
from dashboard_core_example import PanelModel, INPUT_WIDTH, SLIDER_WIDTH

# Step 1: Define your config class
class OptimizerConfig(PanelModel):
    """Configuration for gradient descent optimizer."""

    # Step 2: Define business logic parameters
    learning_rate = param.Number(
        default=0.01,
        bounds=(0.0001, 1.0),
        step=0.001,
        doc="OptimizerConfig: Step size for gradient descent",
    )

    momentum = param.Number(
        default=0.9,
        bounds=(0.0, 1.0),
        step=0.01,
        doc="OptimizerConfig: Momentum coefficient",
    )

    weight_decay = param.Number(
        default=0.0001,
        bounds=(0.0, 1.0),
        step=0.0001,
        doc="OptimizerConfig: L2 regularization strength",
    )

    use_nesterov = param.Boolean(
        default=True,
        doc="OptimizerConfig: Use Nesterov momentum",
    )

    # Step 3: Set Hydra target
    _target_ = param.String(default="torch.optim.SGD")

    # Step 4: Configure widgets
    @property
    def widgets(self):
        return {
            "learning_rate": {
                "type": pn.widgets.EditableFloatSlider,
                "start": 0.0001,
                "end": 1.0,
                "step": 0.001,
                "format": "0.0000",
                "width": SLIDER_WIDTH,
                "name": "Learning Rate",
            },
            "momentum": {
                "type": pn.widgets.EditableFloatSlider,
                "start": 0.0,
                "end": 1.0,
                "step": 0.01,
                "format": "0.00",
                "width": SLIDER_WIDTH,
            },
            "weight_decay": {
                "type": pn.widgets.FloatInput,
                "format": "0.0000",
                "width": INPUT_WIDTH,
                "name": "Weight Decay",
            },
            "use_nesterov": {
                "type": pn.widgets.Toggle,
                "width": INPUT_WIDTH,
                "name": "Nesterov Momentum",
            },
        }

# Step 5: Use it
config = OptimizerConfig()
config.learning_rate = 0.001
config.momentum = 0.95

# Render UI
dashboard = pn.Column(
    pn.pane.Markdown("## Optimizer Configuration"),
    config.__panel__(),
)
dashboard.show()  # Opens browser

# Serialize to dict
d = config.to_dict()
# {'_target_': 'torch.optim.SGD', 'learning_rate': 0.001, 'momentum': 0.95, ...}

# Instantiate actual optimizer
optimizer = config.instantiate(params=model.parameters())
# Returns torch.optim.SGD(params, lr=0.001, momentum=0.95, ...)
```

### Example 2: Nested Configuration

**Goal**: Training config with nested optimizer config.

```python
class TrainingConfig(PanelModel):
    """Configuration for model training."""

    # Own parameters
    num_epochs = param.Integer(
        default=100,
        bounds=(1, None),
        doc="TrainingConfig: Number of training epochs",
    )

    batch_size = param.Integer(
        default=32,
        bounds=(1, None),
        doc="TrainingConfig: Batch size for training",
    )

    # Nested config
    optimizer = param.ClassSelector(
        class_=OptimizerConfig,
        default=None,
        doc="TrainingConfig: Optimizer configuration",
    )

    _target_ = param.String(default="my_module.Trainer")
    _n_widget_columns = 2  # Custom grid width

    def __init__(self, optimizer=None, **params):
        if optimizer is None:
            optimizer = OptimizerConfig()
        super().__init__(optimizer=optimizer, **params)

    @property
    def widgets(self):
        return {
            "num_epochs": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
            },
            "batch_size": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
            },
        }

    def __panel__(self, parameters=None):
        """Custom layout with sections."""
        return pn.Column(
            pn.pane.Markdown("## Training Configuration"),

            pn.pane.Markdown("### Training Parameters"),
            super().__panel__(parameters=["num_epochs", "batch_size"]),

            pn.pane.Markdown("### Optimizer"),
            self.optimizer.__panel__(),  # Nested rendering
        )

# Usage
config = TrainingConfig()
config.num_epochs = 200
config.optimizer.learning_rate = 0.001

# Serialize (handles nesting automatically)
d = config.to_dict()
# {
#   '_target_': 'my_module.Trainer',
#   'num_epochs': 200,
#   'batch_size': 32,
#   'optimizer': {
#     '_target_': 'torch.optim.SGD',
#     'learning_rate': 0.001,
#     ...
#   }
# }

# Instantiate (recursively instantiates optimizer)
trainer = config.instantiate(model=my_model)
# Returns my_module.Trainer(
#   num_epochs=200,
#   batch_size=32,
#   optimizer=torch.optim.SGD(...),
#   model=my_model
# )
```

### Example 3: Conditional UI (Mode-Based)

**Goal**: Config with different parameter groups based on mode selection.

```python
class DataConfig(PanelModel):
    """Configuration for data loading."""

    mode = param.String(
        default="simple",
        doc="DataConfig: Loading mode (simple, advanced)",
    )

    # Always visible
    batch_size = param.Integer(
        default=32,
        doc="DataConfig: Batch size",
    )

    # Visible only in "advanced" mode
    num_workers = param.Integer(
        default=4,
        doc="DataConfig: Number of worker threads",
    )

    prefetch_factor = param.Integer(
        default=2,
        doc="DataConfig: Batches to prefetch per worker",
    )

    _target_ = param.String(default="my_module.DataLoader")

    def __init__(self, **params):
        super().__init__(**params)

        # Create mode toggle
        self._mode_toggle = pn.widgets.RadioButtonGroup(
            name="Mode",
            options={"Simple": "simple", "Advanced": "advanced"},
            value=self.mode,
            button_type="primary",
        )
        self._mode_toggle.param.watch(self._update_mode, "value")

    def _update_mode(self, event=None):
        """Show/hide advanced options."""
        if event is not None:
            self.mode = event.new

        if self.mode == "simple":
            self.advanced_panel.visible = False
        else:
            self.advanced_panel.visible = True

    @property
    def widgets(self):
        return {
            "batch_size": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
            },
            "num_workers": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
            },
            "prefetch_factor": {
                "type": pn.widgets.IntInput,
                "width": INPUT_WIDTH,
            },
        }

    def __panel__(self, parameters=None):
        # Basic parameters
        basic_panel = pn.Column(
            pn.pane.Markdown("### Basic Settings"),
            super().__panel__(parameters=["batch_size"]),
        )

        # Advanced parameters (stored as instance attr for visibility control)
        self.advanced_panel = pn.Column(
            pn.pane.Markdown("### Advanced Settings"),
            super().__panel__(parameters=["num_workers", "prefetch_factor"]),
        )

        # Initialize visibility
        self._update_mode()

        return pn.Column(
            self._mode_toggle,
            basic_panel,
            self.advanced_panel,
        )

# Usage
config = DataConfig()
dashboard = config.__panel__()
dashboard.show()

# Toggle to "Advanced" → advanced_panel becomes visible
# Changes to mode in UI automatically update config.mode
```

---

## 10. Decision Trees

### When to Use Which Widget?

```
Need to input a number?
├─ Float?
│  ├─ Need visual slider with range?
│  │  └─ Use EditableFloatSlider (e.g., proportions 0-1)
│  └─ Just need input field?
│     └─ Use FloatInput (e.g., learning rate, temperature)
└─ Integer?
   ├─ Small range (< 20 options)?
   │  └─ Use Select with integer options
   └─ Large range?
      └─ Use IntInput (e.g., batch size, epochs)

Need to toggle a flag?
└─ Use Toggle (Boolean params)

Need to select from options?
├─ 2-4 options, visually important?
│  └─ Use RadioButtonGroup (e.g., mode selection)
├─ Many options (> 4)?
│  └─ Use Select (dropdown)
└─ Multiple selections allowed?
   └─ Use MultiChoice

Need to input text?
├─ Short (< 50 chars)?
│  └─ Use TextInput (e.g., names, labels)
├─ Multi-line?
│  └─ Use TextAreaInput (e.g., descriptions)
└─ Python literal (list, dict)?
   └─ Use LiteralInput

Need to select a file/folder?
├─ Just a path string?
│  └─ Use Filename or Path param with TextInput
└─ Need file browser UI?
   └─ Use FileInputWidget (custom widget example above)

Need to input a date/time?
└─ Use DatetimePicker (DateTime params)
```

### When to Override `__panel__()`?

```
Default GridBox layout sufficient?
├─ Yes → Don't override (use default)
└─ No → Override __panel__()
   ├─ Need sections with headers?
   │  └─ Use Column with Markdown headers
   ├─ Need collapsible sections?
   │  └─ Use Card containers
   ├─ Need multi-column layout?
   │  └─ Use Row with Columns
   ├─ Need conditional visibility?
   │  └─ Store panels as instance attrs, control .visible
   └─ Need nested config rendering?
      └─ Call nested_config.__panel__() in your layout
```

### When to Create a Custom Widget?

```
Need special UI behavior?
├─ Standard param + widget works?
│  └─ No custom widget needed
├─ Need bidirectional binding to parent?
│  └─ Use FileInputWidget pattern (ref + ref_name)
├─ Need complex composition?
│  └─ Create Parameterized widget with __panel__()
└─ Need custom event handling?
   └─ Use param.watch() callbacks
```

### How to Organize Parameters?

```
How many parameters?
├─ < 5 parameters?
│  └─ Use default GridBox with ncols=2
├─ 5-10 parameters?
│  ├─ Logical groups exist?
│  │  └─ Override __panel__() with sections
│  └─ No clear groups?
│     └─ Use default GridBox with ncols=3
└─ > 10 parameters?
   ├─ Can be grouped?
   │  ├─ Groups roughly equal size?
   │  │  └─ Use Row with Columns (multi-column)
   │  └─ Groups different importance?
   │     └─ Use Cards (collapsible)
   └─ Can be split by mode?
      └─ Use conditional visibility pattern
```

---

## Summary: The Complete Pattern Checklist

When creating a new `PanelModel`:

- [ ] **1. Inherit from PanelModel**
  - [ ] Define business logic parameters (no `_` prefix)
  - [ ] Set `_target_` string for Hydra
  - [ ] Optionally set `_n_widget_columns` and `_max_widget_width`

- [ ] **2. Document all parameters**
  - [ ] Every param has a `doc` string
  - [ ] Doc strings include class name prefix for nested configs
  - [ ] Bounds and constraints specified in param definition

- [ ] **3. Define `widgets` property**
  - [ ] Map param names to widget configs
  - [ ] Use global constants for widths (INPUT_WIDTH, SLIDER_WIDTH)
  - [ ] Specify formats for numeric widgets ("0.00", etc.)
  - [ ] Override display names if needed

- [ ] **4. Handle nested configs** (if applicable)
  - [ ] Use `param.ClassSelector` for nested configs
  - [ ] Initialize nested configs in `__init__`
  - [ ] Handle recursive serialization in `to_dict()`
  - [ ] Handle recursive deserialization in `set_values()`

- [ ] **5. Custom layout** (if needed)
  - [ ] Override `__panel__()` method
  - [ ] Use Markdown headers for sections
  - [ ] Compose with Column/Row/Card
  - [ ] Call `super().__panel__(parameters=[...])` for groups
  - [ ] Call `nested_config.__panel__()` for nested configs

- [ ] **6. Special type handling** (if applicable)
  - [ ] Handle Path/Filename serialization in `to_dict()`
  - [ ] Handle datetime/Timedelta serialization in `to_dict()`
  - [ ] Handle deserialization from both dict and direct values in `set_values()`

- [ ] **7. Conditional UI** (if applicable)
  - [ ] Create mode toggle widget in `__init__`
  - [ ] Store Panel components as instance attributes
  - [ ] Implement `_update_mode()` callback
  - [ ] Watch mode changes with `param.watch()`

- [ ] **8. Test the complete cycle**
  - [ ] Render UI: `config.__panel__().show()`
  - [ ] Modify values in UI
  - [ ] Serialize: `d = config.to_dict()`
  - [ ] Deserialize: `config2.set_values(d)`
  - [ ] Instantiate: `instance = config.instantiate()`

---

## Anti-Patterns to Avoid

### ❌ Mixing Business Logic with UI Configuration

```python
# BAD: UI width in public parameter name
class BadConfig(PanelModel):
    learning_rate_width = param.Integer(default=100)  # ❌ UI concern

# GOOD: Separate concerns
class GoodConfig(PanelModel):
    learning_rate = param.Number(default=0.01)  # ✅ Business logic

    @property
    def widgets(self):
        return {
            "learning_rate": {"width": 100}  # ✅ UI concern
        }
```

### ❌ Hardcoding Widths Everywhere

```python
# BAD: Magic numbers everywhere
@property
def widgets(self):
    return {
        "param1": {"width": 95},   # ❌
        "param2": {"width": 100},  # ❌
        "param3": {"width": 95},   # ❌
    }

# GOOD: Use constants
INPUT_WIDTH = 95

@property
def widgets(self):
    return {
        "param1": {"width": INPUT_WIDTH},        # ✅
        "param2": {"width": int(INPUT_WIDTH * 1.05)},  # ✅
        "param3": {"width": INPUT_WIDTH},        # ✅
    }
```

### ❌ Forgetting to Initialize Nested Configs

```python
# BAD: ClassSelector with no default
class BadConfig(PanelModel):
    optimizer = param.ClassSelector(class_=OptimizerConfig)  # ❌ None by default

# GOOD: Initialize in __init__
class GoodConfig(PanelModel):
    optimizer = param.ClassSelector(class_=OptimizerConfig, default=None)

    def __init__(self, optimizer=None, **params):
        if optimizer is None:
            optimizer = OptimizerConfig()  # ✅ Provide default
        super().__init__(optimizer=optimizer, **params)
```

### ❌ Not Handling Special Type Serialization

```python
# BAD: Serializing Path directly
def to_dict(self):
    return {"output_folder": self.output_folder}  # ❌ Path object not JSON-serializable

# GOOD: Convert to Hydra-compatible dict
def to_dict(self):
    return {
        "output_folder": {
            "_target_": "pathlib.Path",
            "_args_": [str(self.output_folder)],  # ✅
        }
    }
```

### ❌ Ignoring Visibility State in Conditional UI

```python
# BAD: Creating new panels every time
def _update_mode(self, event):
    if self.mode == "advanced":
        self.advanced_panel = pn.Column(...)  # ❌ Loses state

# GOOD: Store panel once, toggle visibility
def __panel__(self):
    self.advanced_panel = pn.Column(...)  # ✅ Create once
    self._update_mode()  # Initialize visibility
    return pn.Column(..., self.advanced_panel)

def _update_mode(self, event):
    self.advanced_panel.visible = (self.mode == "advanced")  # ✅ Toggle
```

---

## Conclusion

This pattern succeeds because it achieves:

1. **Composability**: Configs nest arbitrarily deep
2. **Type Safety**: Param validates at assignment time
3. **Reactivity**: UI updates propagate automatically
4. **Serializability**: Clean Hydra integration
5. **Readability**: Properties make the API self-documenting
6. **Maintainability**: Change param definition → UI updates automatically

**Golden Rule**: Define your parameters once with Param, configure widgets once in the `widgets` property, and let the framework handle the rest.
