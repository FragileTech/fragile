# Algorithm Development - Quick Start

Copy-paste ready commands for developing Gas algorithm variants.

---

## Quick Setup

### Create new algorithm files
```bash
# Algorithm implementation
touch src/fragile/my_algorithm.py

# Parameters
touch src/fragile/my_algorithm_parameters.py

# Tests
touch tests/test_my_algorithm.py

# Visualization
touch src/fragile/shaolin/my_algorithm_viz.py
```

---

## Parameter Definition

### Pydantic parameters template
```python
# src/fragile/my_algorithm_parameters.py

from pydantic import BaseModel, Field

class MyAlgorithmParams(BaseModel):
    """Parameters for MyAlgorithm."""

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Dimensionality")
    gamma: float = Field(gt=0, default=0.5, description="Friction coefficient")
    tau: float = Field(gt=0, default=0.01, description="Time step")

    class Config:
        frozen = True
        extra = "forbid"
```

---

## Algorithm Implementation

### Basic algorithm template
```python
# src/fragile/my_algorithm.py

import torch
from fragile.euclidean_gas import EuclideanGas, SwarmState
from fragile.my_algorithm_parameters import MyAlgorithmParams


class MyAlgorithm(EuclideanGas):
    """My new algorithm variant.

    Mathematical specification: See docs/source/.../XX_my_algorithm.md
    """

    def __init__(self, params: MyAlgorithmParams, device: str = "cpu"):
        super().__init__(params, device)
        # Additional initialization

    def step(self, state: SwarmState) -> SwarmState:
        """Single algorithm step."""
        # Apply base operations
        state = super().step(state)

        # Apply new mechanism
        state = self._apply_new_mechanism(state)

        return state

    def _apply_new_mechanism(self, state: SwarmState) -> SwarmState:
        """Apply new mechanism."""
        # Your implementation
        return state
```

---

## Testing

### Basic test template
```python
# tests/test_my_algorithm.py

import pytest
import torch
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


@pytest.fixture
def algorithm():
    """Create algorithm instance."""
    params = MyAlgorithmParams(N=10, d=2)
    return MyAlgorithm(params)


def test_initialization(algorithm):
    """Test algorithm initializes correctly."""
    assert algorithm.params.N == 10
    assert algorithm.params.d == 2


def test_step_shape_preservation(algorithm):
    """Test that step preserves tensor shapes."""
    state = algorithm.initialize()
    new_state = algorithm.step(state)

    assert new_state.x.shape == (10, 2)
    assert new_state.v.shape == (10, 2)


def test_convergence(algorithm):
    """Test convergence on simple problem."""
    state = algorithm.initialize()

    for _ in range(100):
        state = algorithm.step(state)

    # Check convergence
    mean_pos = state.x.mean(dim=0)
    assert torch.norm(mean_pos) < 1.0  # Reasonable bound
```

### Run tests
```bash
# Run all tests
pytest tests/test_my_algorithm.py -v

# With coverage
pytest tests/test_my_algorithm.py --cov=src/fragile/my_algorithm --cov-report=term-missing

# Specific test
pytest tests/test_my_algorithm.py::test_convergence -v
```

---

## Visualization

### Basic 2D visualization
```python
# src/fragile/shaolin/my_algorithm_viz.py

import holoviews as hv
from fragile.my_algorithm import MyAlgorithm

hv.extension('bokeh')


def visualize_swarm(algorithm, n_steps=100):
    """Visualize swarm evolution."""
    state = algorithm.initialize()
    history = []

    # Record history
    for step in range(n_steps):
        state = algorithm.step(state)
        history.append({
            'positions': state.x.clone().cpu().numpy(),
            'rewards': state.reward.clone().cpu().numpy(),
        })

    # Create animation
    def plot_frame(frame_idx):
        data = history[frame_idx]
        positions = data['positions']
        rewards = data['rewards']

        return hv.Scatter(
            (positions[:, 0], positions[:, 1], rewards),
            kdims=['x', 'y'],
            vdims=['reward']
        ).opts(
            color='reward',
            cmap='viridis',
            size=8,
            width=600,
            height=600,
            title=f'Step {frame_idx}',
            colorbar=True,
        )

    dmap = hv.DynamicMap(plot_frame, kdims=['frame'])
    dmap = dmap.redim.values(frame=list(range(n_steps)))

    return dmap
```

---

## Usage Example

### Complete example
```python
# examples/my_algorithm_example.py

from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


def main():
    """Run optimization."""

    # Create parameters
    params = MyAlgorithmParams(
        N=50,
        d=2,
        gamma=0.5,
        tau=0.01,
    )

    # Create algorithm
    algo = MyAlgorithm(params)

    # Initialize
    state = algo.initialize()

    # Run
    print("Step | Mean Reward | Std Reward")
    print("-" * 40)

    for step in range(500):
        state = algo.step(state)

        if step % 50 == 0:
            print(f"{step:4d} | {state.mean_reward:11.4f} | {state.std_reward:10.4f}")


if __name__ == "__main__":
    main()
```

### Run example
```bash
python examples/my_algorithm_example.py
```

---

## Common Vectorization Patterns

### Per-walker operations
```python
# Add velocity to position (vectorized)
state.x = state.x + state.v * dt  # [N, d] + [N, d] * scalar

# Compute per-walker energies
energies = torch.sum(state.x ** 2, dim=1)  # [N]

# Apply function element-wise
rewards = torch.exp(-state.potential)  # [N]
```

### Masked operations
```python
# Filter by alive mask
alive_x = state.x[state.alive_mask]  # [N_alive, d]

# Update only alive walkers
state.x[state.alive_mask] += delta

# Compute statistics over alive walkers
mean_alive = state.x[state.alive_mask].mean(dim=0)  # [d]
```

### Pairwise operations
```python
# Compute all pairwise differences
x_expanded = state.x.unsqueeze(1)  # [N, 1, d]
y_expanded = state.x.unsqueeze(0)  # [1, N, d]
diffs = x_expanded - y_expanded     # [N, N, d]

# Pairwise distances
distances = torch.norm(diffs, dim=2)  # [N, N]
```

---

## Code Quality Checks

### Format and lint
```bash
# Format code
make style

# Check style
make check

# Type checking
make typing

# All checks
make lint
```

### Run all tests
```bash
# All tests
make test

# Without coverage (faster)
make no-cov

# With debugging
make debug
```

---

## Documentation

### Build docs
```bash
# Build documentation
make build-docs

# Serve locally
make serve-docs

# Open browser to http://localhost:8000
```

### Write mathematical spec
```bash
# Create spec document
vim docs/source/1_euclidean_gas/XX_my_algorithm.md

# Format math
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/XX_my_algorithm.md

# Build and verify
make build-docs
```

---

## Interactive Dashboard

### Panel dashboard template
```python
import panel as pn
import holoviews as hv


def create_dashboard():
    """Create interactive dashboard."""

    # Widgets
    N_widget = pn.widgets.IntSlider(name='N', start=10, end=200, value=50)
    gamma_widget = pn.widgets.FloatSlider(name='gamma', start=0.1, end=2.0, value=0.5)
    run_button = pn.widgets.Button(name='Run', button_type='primary')

    # Plot pane
    plot_pane = pn.pane.HoloViews()

    # Callback
    def run_simulation(event):
        params = MyAlgorithmParams(N=N_widget.value, d=2, gamma=gamma_widget.value)
        algo = MyAlgorithm(params)
        viz = visualize_swarm(algo, n_steps=100)
        plot_pane.object = viz

    run_button.on_click(run_simulation)

    # Layout
    sidebar = pn.Column('# Parameters', N_widget, gamma_widget, run_button)
    return pn.Row(sidebar, plot_pane)


if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard.servable()
```

### Run dashboard
```bash
panel serve src/fragile/shaolin/my_algorithm_viz.py
# Open browser to http://localhost:5006
```

---

## Complete Workflow

### From idea to tested algorithm

```bash
# 1. Document math spec
vim docs/source/1_euclidean_gas/XX_my_algorithm.md
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/XX_my_algorithm.md

# 2. Create files
touch src/fragile/my_algorithm.py
touch src/fragile/my_algorithm_parameters.py
touch tests/test_my_algorithm.py

# 3. Implement (see templates above)
vim src/fragile/my_algorithm_parameters.py
vim src/fragile/my_algorithm.py

# 4. Test
pytest tests/test_my_algorithm.py -v

# 5. Visualize
touch src/fragile/shaolin/my_algorithm_viz.py
# (Add visualization code)

# 6. Run checks
make lint
make test

# 7. Build docs
make build-docs
```

---

## Quick Reference

### Shape conventions
| Shape | Meaning |
|-------|---------|
| `[N]` | Per-walker scalars |
| `[N, d]` | Per-walker vectors |
| `[N, d, d]` | Per-walker matrices |
| `[N, N]` | Pairwise scalars |
| `[N, N, d]` | Pairwise vectors |

### Notation mapping
| Math | Code |
|------|------|
| $\gamma$ | `gamma` |
| $\tau$ | `tau` |
| $\sigma$ | `sigma` |
| $N$ | `N` |
| $d$ | `d` |
| $x_i$ | `state.x[i]` or `state.x` (vectorized) |
| $\mathcal{X}$ | State space |

### Test commands
```bash
pytest tests/test_my_algorithm.py -v          # Run all tests
pytest tests/test_my_algorithm.py::test_name  # Specific test
pytest --cov=src/fragile/my_algorithm          # With coverage
make test                                       # All project tests
```

---

**Time estimates:**
- Parameter definition: ~10 min
- Basic implementation: ~1-2 hours
- Testing: ~30 min
- Visualization: ~30 min
- Documentation: ~30 min
- **Total**: ~3-4 hours for complete algorithm

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
