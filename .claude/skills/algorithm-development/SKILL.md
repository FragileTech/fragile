---
name: algorithm-development
description: Develop, test, and visualize Gas algorithm variants following project conventions. Use when implementing new algorithms, adding features, testing correctness, or creating interactive visualizations.
---

# Algorithm Development Skill

## Purpose

Complete workflow for developing, testing, and visualizing Gas algorithm variants while following Fragile project conventions and maintaining code quality.

**Input**: Algorithm idea, mathematical specification, or feature request
**Output**: Tested, visualized, and documented algorithm implementation
**Pipeline**: Design → Implement → Test → Visualize → Document

---

## Project Architecture

### Algorithm Hierarchy

```
BaseAlgorithm (abstract)
    ↓
EuclideanGas (src/fragile/euclidean_gas.py)
    ├─ Langevin dynamics (BAOAB integrator)
    ├─ Cloning operator
    └─ Virtual reward mechanism
    ↓
AdaptiveGas (src/fragile/adaptive_gas.py)
    ├─ Inherits all EuclideanGas features
    ├─ + Adaptive force (mean-field fitness)
    ├─ + Viscous coupling
    └─ + Regularized Hessian diffusion
```

### Key Design Principles

1. **Vectorization First**: All operations use PyTorch tensors, first dimension is N (number of walkers)
2. **Parameter Validation**: All parameters use Pydantic models with strict validation
3. **Mathematical Rigor**: Code notation matches markdown documentation exactly
4. **Testability**: Every component has comprehensive tests
5. **Visualization**: HoloViz stack (HoloViews + Bokeh/Plotly) for all visualizations

---

## Core Data Structures

### SwarmState

**Location**: `src/fragile/euclidean_gas.py`

```python
@dataclass
class SwarmState:
    """Vectorized swarm state representation."""

    # Required fields
    x: Tensor  # Positions [N, d]
    v: Tensor  # Velocities [N, d]

    # Per-walker metadata
    reward: Tensor  # Fitness values [N]
    potential: Tensor  # Potential energy [N]
    virtual_reward: Tensor  # Virtual rewards [N]
    cum_reward: Tensor  # Cumulative rewards [N]

    # Swarm statistics (cached)
    mean_reward: float
    std_reward: float

    # Alive/dead tracking
    alive_mask: Tensor  # Boolean mask [N]
```

**Shape conventions:**
- `[N]`: Per-walker scalars
- `[N, d]`: Per-walker vectors
- `[N, d, d]`: Per-walker matrices

### Parameters

**Location**: `src/fragile/gas_parameters.py`

```python
class EuclideanGasParams(BaseModel):
    """Euclidean Gas parameters with validation."""

    N: int = Field(gt=0)  # Number of walkers
    d: int = Field(gt=0)  # Dimensionality

    # Potential function
    potential: PotentialFunction

    # Nested parameter groups
    langevin: LangevinParams
    cloning: CloningParams
```

**Validation patterns:**
- Use `Field(gt=0)` for positive values
- Use `Field(ge=0)` for non-negative values
- Provide clear docstrings matching mathematical notation

---

## Complete Development Workflow

### Stage 1: Design and Specification

**Purpose**: Define algorithm mathematically before coding

#### Step 1.1: Mathematical Specification

**Document in markdown first:**

```bash
# Create specification document
vim docs/source/1_euclidean_gas/XX_your_algorithm.md
```

**Template:**
```markdown
# Your Algorithm Name

## Mathematical Formulation

### State Space

The algorithm operates on state space $(\mathcal{X}, d_\mathcal{X})$ where...

### Dynamics

The update equation is:

$$
dx = f(x) dt + \sigma dW_t
$$

where:
- $f(x)$: Drift term
- $\sigma$: Diffusion coefficient

### Operators

**Operator 1: Name**

$$
\Psi_1: \mathcal{S}_N \to \mathcal{S}_N
$$

[Define mathematically]

### Parameters

| Symbol | Name | Type | Range | Default |
|--------|------|------|-------|---------|
| $\gamma$ | Friction | float | $> 0$ | 0.5 |
| $\tau$ | Time step | float | $> 0$ | 0.01 |
```

**Use mathematical-writing skill** to ensure correctness.

#### Step 1.2: Define Interface

**Identify required methods:**

```python
# What methods does your algorithm need?

# Core update loop
def step(self, state: SwarmState) -> SwarmState:
    """Single algorithm step."""
    ...

# Operator methods
def _apply_dynamics(self, state: SwarmState) -> SwarmState:
    """Apply physical dynamics."""
    ...

def _apply_selection(self, state: SwarmState) -> SwarmState:
    """Apply selection mechanism."""
    ...
```

#### Step 1.3: Determine Base Class

**Decision tree:**

```
Does algorithm use Langevin dynamics + cloning?
    Yes → Inherit from EuclideanGas
    No → Is it a Gas variant?
        Yes → Inherit from BaseGas (if exists)
        No → Create new base class
```

**Example:**
```python
# New adaptive variant
class MyAdaptiveGas(EuclideanGas):
    """Extends EuclideanGas with new adaptive mechanism."""
    ...

# Completely new approach
class MyNovelAlgorithm:
    """Novel optimization algorithm."""
    ...
```

---

### Stage 2: Implementation

**Purpose**: Write code following project conventions

#### Step 2.1: Set Up File Structure

```bash
# Create algorithm file
touch src/fragile/my_algorithm.py

# Create parameter file (if needed)
touch src/fragile/my_algorithm_parameters.py

# Create test file
touch tests/test_my_algorithm.py

# Create visualization file (optional)
touch src/fragile/shaolin/my_algorithm_viz.py
```

#### Step 2.2: Define Parameters

**Template** (`my_algorithm_parameters.py`):

```python
from pydantic import BaseModel, Field

class MyAlgorithmParams(BaseModel):
    """Parameters for MyAlgorithm.

    Mathematical specification: See docs/source/.../XX_my_algorithm.md

    Attributes:
        N: Number of walkers
        d: Dimensionality
        gamma: Friction coefficient (matches $\\gamma$ in docs)
        tau: Time step (matches $\\tau$ in docs)
    """

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Dimensionality")
    gamma: float = Field(gt=0, default=0.5, description="Friction coefficient")
    tau: float = Field(gt=0, default=0.01, description="Time step")

    class Config:
        """Pydantic config."""
        frozen = True  # Immutable after creation
        extra = "forbid"  # No extra fields allowed
```

**Key points:**
- Use `Field` for validation
- Provide clear descriptions
- Reference mathematical documentation
- Match notation (γ → gamma, τ → tau)

#### Step 2.3: Implement Algorithm

**Template** (`my_algorithm.py`):

```python
import torch
from torch import Tensor
from typing import Optional

from fragile.euclidean_gas import EuclideanGas, SwarmState
from fragile.my_algorithm_parameters import MyAlgorithmParams


class MyAlgorithm(EuclideanGas):
    """My new algorithm variant.

    Extends EuclideanGas with [describe extension].

    Mathematical specification: See docs/source/.../XX_my_algorithm.md

    Attributes:
        params: Algorithm parameters
        device: PyTorch device
    """

    def __init__(
        self,
        params: MyAlgorithmParams,
        device: str = "cpu"
    ):
        """Initialize algorithm.

        Args:
            params: Algorithm parameters
            device: PyTorch device ("cpu" or "cuda")
        """
        # Initialize base class
        super().__init__(params, device)

        # Additional initialization
        self.my_param = params.my_param

    def step(self, state: SwarmState) -> SwarmState:
        """Single algorithm step.

        Implements the update equation from docs.

        Args:
            state: Current swarm state

        Returns:
            Updated swarm state
        """
        # Apply base operations (Langevin, cloning)
        state = super().step(state)

        # Apply new mechanism
        state = self._apply_new_mechanism(state)

        return state

    def _apply_new_mechanism(self, state: SwarmState) -> SwarmState:
        """Apply new adaptive mechanism.

        Mathematical specification: See docs, Equation (X.Y)

        Args:
            state: Current swarm state

        Returns:
            Updated swarm state
        """
        # Extract state
        x = state.x  # [N, d]
        v = state.v  # [N, d]

        # Compute update
        # [Your implementation matching mathematical spec]

        # Update state
        state.x = x
        state.v = v

        return state
```

**Code conventions:**
- Type hints for all parameters and returns
- Docstrings with Args/Returns sections
- Reference mathematical documentation
- Vectorized operations (no loops over walkers)

#### Step 2.4: Follow Vectorization Patterns

**Key patterns:**

```python
# Pattern 1: Per-walker operations
# Shape: [N, d] → [N, d]
new_x = state.x + velocity * dt  # Vectorized addition

# Pattern 2: Per-walker scalars
# Shape: [N] → [N]
energies = torch.sum(state.x ** 2, dim=1)  # Sum over dimensions

# Pattern 3: Pairwise operations
# Shape: [N, d] → [N, N, d]
differences = state.x.unsqueeze(1) - state.x.unsqueeze(0)  # All pairs

# Pattern 4: Masked operations
# Shape: [N, d] with mask [N]
alive_positions = state.x[state.alive_mask]  # Filter by mask
```

**Common mistakes to avoid:**
```python
# ❌ WRONG: Loop over walkers
for i in range(N):
    state.x[i] = update(state.x[i])

# ✅ CORRECT: Vectorized operation
state.x = update(state.x)
```

---

### Stage 3: Testing

**Purpose**: Ensure correctness through comprehensive tests

#### Step 3.1: Unit Tests

**Template** (`tests/test_my_algorithm.py`):

```python
import pytest
import torch
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


@pytest.fixture
def simple_params():
    """Create simple test parameters."""
    return MyAlgorithmParams(
        N=10,
        d=2,
        gamma=0.5,
        tau=0.01,
    )


@pytest.fixture
def algorithm(simple_params):
    """Create algorithm instance."""
    return MyAlgorithm(simple_params)


def test_initialization(algorithm):
    """Test algorithm initializes correctly."""
    assert algorithm.params.N == 10
    assert algorithm.params.d == 2
    assert algorithm.device == "cpu"


def test_step_shape_preservation(algorithm):
    """Test that step preserves tensor shapes."""
    # Create initial state
    state = algorithm.initialize()

    # Apply step
    new_state = algorithm.step(state)

    # Check shapes
    assert new_state.x.shape == (10, 2)
    assert new_state.v.shape == (10, 2)
    assert new_state.reward.shape == (10,)


def test_energy_conservation(algorithm):
    """Test physical property: energy bounds."""
    state = algorithm.initialize()

    # Run for several steps
    for _ in range(100):
        state = algorithm.step(state)

    # Check energy bounds
    total_energy = compute_energy(state)
    assert total_energy < 1000  # Should not explode


def test_convergence_simple_problem(algorithm):
    """Test convergence on simple optimization problem."""
    # Define simple quadratic potential
    # x^2 has minimum at x=0

    state = algorithm.initialize()

    # Run algorithm
    for _ in range(1000):
        state = algorithm.step(state)

    # Check convergence
    mean_position = state.x.mean(dim=0)
    assert torch.norm(mean_position) < 0.1  # Close to minimum
```

**Test categories:**
- **Initialization**: Correct setup
- **Shape preservation**: Tensor shapes maintained
- **Physical properties**: Energy, momentum conservation
- **Convergence**: Works on simple problems
- **Edge cases**: N=1, d=1, boundary conditions

#### Step 3.2: Run Tests

```bash
# Run all tests for your algorithm
pytest tests/test_my_algorithm.py -v

# Run with coverage
pytest tests/test_my_algorithm.py --cov=src/fragile/my_algorithm --cov-report=term-missing

# Run specific test
pytest tests/test_my_algorithm.py::test_convergence_simple_problem -v
```

#### Step 3.3: Comparison Tests

**Compare with baseline:**

```python
def test_compare_with_euclidean_gas():
    """Compare with baseline EuclideanGas."""
    from fragile.euclidean_gas import EuclideanGas

    # Same parameters
    params = MyAlgorithmParams(N=50, d=2)

    # Initialize both
    my_algo = MyAlgorithm(params)
    baseline = EuclideanGas(params)

    # Run both
    my_state = my_algo.initialize()
    baseline_state = baseline.initialize()

    for _ in range(500):
        my_state = my_algo.step(my_state)
        baseline_state = baseline.step(baseline_state)

    # Compare performance
    my_reward = my_state.mean_reward
    baseline_reward = baseline_state.mean_reward

    # Your algorithm should be better (or at least competitive)
    assert my_reward >= baseline_reward * 0.9  # Within 10%
```

---

### Stage 4: Visualization

**Purpose**: Create interactive visualizations using HoloViz

#### Step 4.1: Basic Visualization

**Template** (`src/fragile/shaolin/my_algorithm_viz.py`):

```python
import holoviews as hv
import panel as pn
from fragile.my_algorithm import MyAlgorithm

hv.extension('bokeh')  # 2D visualizations


def visualize_swarm(algorithm: MyAlgorithm, n_steps: int = 100):
    """Visualize swarm evolution.

    Args:
        algorithm: Algorithm instance
        n_steps: Number of steps to visualize

    Returns:
        HoloViews DynamicMap for animation
    """
    # Initialize
    state = algorithm.initialize()
    history = []

    # Run and record
    for step in range(n_steps):
        state = algorithm.step(state)
        history.append({
            'positions': state.x.clone().cpu().numpy(),
            'rewards': state.reward.clone().cpu().numpy(),
            'step': step,
        })

    # Create visualization
    def plot_frame(frame_idx):
        """Plot single frame."""
        data = history[frame_idx]
        positions = data['positions']
        rewards = data['rewards']

        # Create scatter plot
        scatter = hv.Scatter(
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

        return scatter

    # Create animation
    dmap = hv.DynamicMap(plot_frame, kdims=['frame'])
    dmap = dmap.redim.values(frame=list(range(n_steps)))

    return dmap
```

#### Step 4.2: Interactive Dashboard

**Create Panel dashboard:**

```python
def create_dashboard():
    """Create interactive dashboard for algorithm exploration."""

    # Parameters widgets
    N_widget = pn.widgets.IntSlider(name='N (walkers)', start=10, end=200, value=50)
    gamma_widget = pn.widgets.FloatSlider(name='gamma (friction)', start=0.1, end=2.0, value=0.5)

    # Control widgets
    run_button = pn.widgets.Button(name='Run Simulation', button_type='primary')
    reset_button = pn.widgets.Button(name='Reset', button_type='warning')

    # Visualization pane
    plot_pane = pn.pane.HoloViews(hv.Curve([]))

    # Callback
    def run_simulation(event):
        """Run simulation with current parameters."""
        params = MyAlgorithmParams(
            N=N_widget.value,
            d=2,
            gamma=gamma_widget.value,
        )
        algorithm = MyAlgorithm(params)
        viz = visualize_swarm(algorithm, n_steps=100)
        plot_pane.object = viz

    run_button.on_click(run_simulation)

    # Layout
    sidebar = pn.Column(
        '# Algorithm Parameters',
        N_widget,
        gamma_widget,
        run_button,
        reset_button,
    )

    return pn.Row(sidebar, plot_pane)


# Launch dashboard
if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard.servable()  # Make servable with panel serve
```

**Run dashboard:**
```bash
panel serve src/fragile/shaolin/my_algorithm_viz.py
# Open browser to http://localhost:5006
```

#### Step 4.3: 3D Visualization

**For 3D problems:**

```python
import holoviews as hv
hv.extension('plotly')  # 3D requires Plotly backend


def visualize_3d_swarm(algorithm, n_steps=100):
    """Visualize 3D swarm."""
    # Run simulation (same as 2D)
    state = algorithm.initialize()
    history = []

    for step in range(n_steps):
        state = algorithm.step(state)
        history.append({
            'positions': state.x.clone().cpu().numpy(),
            'rewards': state.reward.clone().cpu().numpy(),
        })

    # Create 3D scatter
    def plot_3d_frame(frame_idx):
        data = history[frame_idx]
        pos = data['positions']
        rewards = data['rewards']

        scatter3d = hv.Scatter3D(
            (pos[:, 0], pos[:, 1], pos[:, 2], rewards),
            kdims=['x', 'y', 'z'],
            vdims=['reward']
        ).opts(
            color='reward',
            cmap='viridis',
            size=4,
            width=800,
            height=800,
        )

        return scatter3d

    dmap = hv.DynamicMap(plot_3d_frame, kdims=['frame'])
    dmap = dmap.redim.values(frame=list(range(n_steps)))

    return dmap
```

---

### Stage 5: Documentation

**Purpose**: Document algorithm for users and developers

#### Step 5.1: Code Documentation

**Docstring requirements:**

```python
class MyAlgorithm(EuclideanGas):
    """One-line summary.

    Extended description explaining:
    - What the algorithm does
    - How it extends/differs from base class
    - Key mathematical ideas

    Mathematical specification: See docs/source/.../XX_my_algorithm.md

    Attributes:
        params: Algorithm parameters
        device: PyTorch device
        [other attributes]

    Examples:
        >>> params = MyAlgorithmParams(N=50, d=2)
        >>> algo = MyAlgorithm(params)
        >>> state = algo.initialize()
        >>> for _ in range(100):
        ...     state = algo.step(state)
        >>> print(state.mean_reward)
    """
```

#### Step 5.2: Usage Examples

**Create example script:**

```python
# examples/my_algorithm_example.py

"""Example usage of MyAlgorithm."""

import torch
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


def main():
    """Run simple optimization problem."""

    # Define parameters
    params = MyAlgorithmParams(
        N=50,           # 50 walkers
        d=2,            # 2D problem
        gamma=0.5,      # Moderate friction
        tau=0.01,       # Small time step
    )

    # Create algorithm
    algorithm = MyAlgorithm(params, device="cpu")

    # Initialize swarm
    state = algorithm.initialize()

    # Run optimization
    print("Step | Mean Reward | Std Reward")
    print("-" * 40)

    for step in range(500):
        state = algorithm.step(state)

        if step % 50 == 0:
            print(f"{step:4d} | {state.mean_reward:11.4f} | {state.std_reward:10.4f}")

    print(f"\nFinal position: {state.x.mean(dim=0)}")


if __name__ == "__main__":
    main()
```

#### Step 5.3: Mathematical Documentation

**Update markdown document:**

```bash
vim docs/source/1_euclidean_gas/XX_my_algorithm.md
```

**Add implementation section:**
```markdown
## Implementation

### Code Structure

The algorithm is implemented in `src/fragile/my_algorithm.py` as class `MyAlgorithm`.

**Inheritance**: `MyAlgorithm` extends {prf:ref}`obj-euclidean-gas`.

### Key Methods

**step(state) → state**
Implements Algorithm X from §Y.Z.

**_apply_new_mechanism(state) → state**
Implements Equation (X.Y).

### Usage Example

See `examples/my_algorithm_example.py` for complete usage.

```python
from fragile.my_algorithm import MyAlgorithm

params = MyAlgorithmParams(N=50, d=2)
algo = MyAlgorithm(params)
state = algo.initialize()

for _ in range(100):
    state = algo.step(state)
```

### Performance

Tested on [benchmark problems]:
- Rastrigin: Convergence in ~500 steps
- Rosenbrock: Convergence in ~1000 steps
```

---

## Integration Points

### With Mathematical-Writing

**Document algorithm before implementing:**

1. Write mathematical specification (mathematical-writing)
2. Get dual-review for correctness
3. Implement algorithm following spec
4. Test implementation matches math

### With Framework-Consistency

**Ensure algorithm matches framework:**

1. Implement algorithm (algorithm-development)
2. Check notation consistency (framework-consistency)
3. Verify parameter definitions match
4. Validate axiom usage

### With Proof-Validation

**Prove algorithmic properties:**

1. Implement algorithm
2. Formulate convergence theorem
3. Develop proof (proof-validation)
4. Implement corresponding test

---

## Best Practices

### 1. Document Math First

Always write mathematical specification before coding.

**Workflow:**
```
Math spec → Review → Code → Test → Visualize
```

### 2. Maintain Vectorization

Never loop over walkers:

```python
# ❌ BAD
for i in range(N):
    x[i] += v[i] * dt

# ✅ GOOD
x += v * dt
```

### 3. Match Notation

Code variables should match mathematical symbols:

| Math | Code |
|------|------|
| $\gamma$ | `gamma` |
| $\tau$ | `tau` |
| $\mathcal{X}$ | `X` (state space) |
| $x_i$ | `x[i]` or `x` (vectorized) |

### 4. Use Type Hints

Always provide type hints:

```python
def step(self, state: SwarmState) -> SwarmState:
    ...
```

### 5. Test Early and Often

Write tests alongside implementation:

```
Implement operator → Write test → Run test → Repeat
```

### 6. Visualize for Understanding

Create visualizations to debug and understand:

```python
# During development, visualize intermediate steps
viz = visualize_swarm(algorithm, n_steps=50)
viz  # Inspect in Jupyter notebook
```

---

## Quality Checklist

Before merging algorithm:

**Code Quality:**
- [ ] Type hints for all functions
- [ ] Comprehensive docstrings
- [ ] No loops over walkers (vectorized)
- [ ] Follows project conventions

**Testing:**
- [ ] Unit tests pass
- [ ] Coverage > 90%
- [ ] Tests physical properties
- [ ] Tests convergence

**Documentation:**
- [ ] Mathematical specification written
- [ ] Usage examples provided
- [ ] Visualization created
- [ ] Integration documented

**Framework Consistency:**
- [ ] Notation matches glossary
- [ ] Parameters validated with Pydantic
- [ ] References framework axioms
- [ ] No contradictions

---

## Related Documentation

- **CLAUDE.md**: Project coding standards
- **Mathematical-Writing Skill**: Document mathematical specs
- **Framework-Consistency Skill**: Verify notation consistency
- **Proof-Validation Skill**: Prove algorithmic properties
- **HoloViz Documentation**: https://holoviz.org/
- **PyTorch Documentation**: https://pytorch.org/docs/

---

## Version History

- **v1.0.0** (2025-10-28): Initial algorithm-development skill
  - Complete development workflow
  - Testing guidelines
  - Visualization patterns
  - Integration with other skills

---

**Next**: See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands.
