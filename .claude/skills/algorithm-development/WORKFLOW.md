# Algorithm Development - Complete Workflow

Detailed step-by-step procedures for developing Gas algorithm variants.

---

## Prerequisites

- ✅ Python environment with fragile package installed
- ✅ Understanding of project architecture (CLAUDE.md)
- ✅ Mathematical specification of algorithm
- ✅ Familiarity with PyTorch basics

---

## Complete Development Pipeline

### Stage 0: Mathematical Design

**Purpose**: Define algorithm rigorously before coding

#### Step 0.1: Write Mathematical Specification

```bash
# Create specification document
vim docs/source/1_euclidean_gas/XX_my_algorithm.md
```

**Required sections:**
1. State space definition
2. Update equations
3. Operator definitions
4. Parameter specifications
5. Convergence properties (if known)

#### Step 0.2: Dual-Review Mathematical Spec

Submit to Gemini 2.5 Pro + Codex for review:
- Mathematical rigor
- Consistency with framework
- Notation correctness

#### Step 0.3: Define Algorithm Interface

**Design methods needed:**
```python
# Core update loop
step(state) → state

# Initialization
initialize() → state

# Operators (if modular)
_apply_operator_1(state) → state
_apply_operator_2(state) → state
```

---

### Stage 1: Parameter Implementation

**Purpose**: Define and validate algorithm parameters

#### Step 1.1: Create Parameter File

```bash
touch src/fragile/my_algorithm_parameters.py
vim src/fragile/my_algorithm_parameters.py
```

#### Step 1.2: Define Parameters with Pydantic

```python
from pydantic import BaseModel, Field

class MyAlgorithmParams(BaseModel):
    """Parameters for MyAlgorithm.

    Mathematical specification: docs/source/.../XX_my_algorithm.md

    Attributes:
        N: Number of walkers (matches $N$ in docs)
        d: Dimensionality (matches $d$ in docs)
        gamma: Friction coefficient (matches $\\gamma$ in docs)
        tau: Time step (matches $\\tau$ in docs)
    """

    N: int = Field(gt=0, description="Number of walkers")
    d: int = Field(gt=0, description="Dimensionality")

    gamma: float = Field(gt=0, default=0.5, description="Friction coefficient")
    tau: float = Field(gt=0, default=0.01, description="Time step")

    # Add algorithm-specific parameters
    my_param: float = Field(gt=0, default=1.0, description="My parameter")

    class Config:
        frozen = True  # Immutable
        extra = "forbid"  # No extra fields
```

#### Step 1.3: Test Parameter Validation

```python
# Test in Python REPL
>>> from my_algorithm_parameters import MyAlgorithmParams
>>>
>>> # Valid parameters
>>> params = MyAlgorithmParams(N=50, d=2)
>>> print(params)
>>>
>>> # Invalid parameters (should raise error)
>>> params = MyAlgorithmParams(N=-1, d=2)  # N must be > 0
ValidationError...
```

---

### Stage 2: Algorithm Implementation

**Purpose**: Implement algorithm following specification

#### Step 2.1: Create Algorithm File

```bash
touch src/fragile/my_algorithm.py
vim src/fragile/my_algorithm.py
```

#### Step 2.2: Implement Base Structure

```python
import torch
from torch import Tensor
from typing import Optional

from fragile.euclidean_gas import EuclideanGas, SwarmState
from fragile.my_algorithm_parameters import MyAlgorithmParams


class MyAlgorithm(EuclideanGas):
    """My new algorithm variant.

    Extends EuclideanGas with [describe extension].

    Mathematical specification: docs/source/.../XX_my_algorithm.md

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
            device: PyTorch device
        """
        super().__init__(params, device)
        # Additional initialization

    def step(self, state: SwarmState) -> SwarmState:
        """Single algorithm step.

        Args:
            state: Current swarm state

        Returns:
            Updated swarm state
        """
        # Implementation
        ...
```

#### Step 2.3: Implement Core Methods

**step() method:**
```python
def step(self, state: SwarmState) -> SwarmState:
    """Single algorithm step."""

    # Option 1: Extend base class
    state = super().step(state)  # Langevin + cloning
    state = self._apply_new_mechanism(state)

    # Option 2: Complete override
    # state = self._apply_all_operations(state)

    return state
```

**Helper methods:**
```python
def _apply_new_mechanism(self, state: SwarmState) -> SwarmState:
    """Apply new adaptive mechanism.

    Mathematical specification: Equation (X.Y) in docs.

    Args:
        state: Current swarm state

    Returns:
        Updated swarm state
    """
    # Extract state (shape: [N, d])
    x = state.x
    v = state.v

    # Compute update (vectorized)
    # Example: Adaptive force based on mean
    x_mean = x.mean(dim=0)  # [d]
    force = x_mean.unsqueeze(0) - x  # [N, d]

    # Update velocities
    v = v + force * self.params.tau

    # Update positions
    x = x + v * self.params.tau

    # Update state
    state.x = x
    state.v = v

    return state
```

#### Step 2.4: Follow Vectorization Patterns

**Key patterns:**

**Pattern 1: Element-wise operations**
```python
# ✅ CORRECT: Vectorized
new_x = state.x + state.v * dt  # [N, d] + [N, d] * scalar

# ❌ WRONG: Loop over walkers
for i in range(N):
    new_x[i] = state.x[i] + state.v[i] * dt
```

**Pattern 2: Reduction operations**
```python
# Mean over walkers
x_mean = state.x.mean(dim=0)  # [N, d] → [d]

# Sum over dimensions
energies = torch.sum(state.x ** 2, dim=1)  # [N, d] → [N]
```

**Pattern 3: Broadcasting**
```python
# Broadcast scalar to all walkers
tau_vec = self.params.tau * torch.ones(N, device=self.device)  # [N]

# Or use implicit broadcasting
x = x + v * self.params.tau  # tau broadcasts automatically
```

---

### Stage 3: Testing

**Purpose**: Ensure correctness through comprehensive tests

#### Step 3.1: Create Test File

```bash
touch tests/test_my_algorithm.py
vim tests/test_my_algorithm.py
```

#### Step 3.2: Write Unit Tests

```python
import pytest
import torch
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


@pytest.fixture
def simple_params():
    """Create simple test parameters."""
    return MyAlgorithmParams(N=10, d=2)


@pytest.fixture
def algorithm(simple_params):
    """Create algorithm instance."""
    return MyAlgorithm(simple_params)


# Test 1: Initialization
def test_initialization(algorithm):
    """Test algorithm initializes correctly."""
    assert algorithm.params.N == 10
    assert algorithm.params.d == 2
    assert algorithm.device == "cpu"

    # Test can create state
    state = algorithm.initialize()
    assert state.x.shape == (10, 2)
    assert state.v.shape == (10, 2)


# Test 2: Shape preservation
def test_step_shape_preservation(algorithm):
    """Test that step preserves tensor shapes."""
    state = algorithm.initialize()
    new_state = algorithm.step(state)

    assert new_state.x.shape == state.x.shape
    assert new_state.v.shape == state.v.shape
    assert new_state.reward.shape == state.reward.shape


# Test 3: Multiple steps
def test_multiple_steps(algorithm):
    """Test algorithm runs for multiple steps."""
    state = algorithm.initialize()

    for _ in range(10):
        state = algorithm.step(state)

    # Should not crash
    assert state is not None


# Test 4: Physical properties
def test_energy_bounds(algorithm):
    """Test energy remains bounded."""
    state = algorithm.initialize()

    max_energy = 0
    for _ in range(100):
        state = algorithm.step(state)

        # Compute total kinetic + potential energy
        kinetic = 0.5 * torch.sum(state.v ** 2)
        potential = torch.sum(state.potential)
        total_energy = kinetic + potential

        max_energy = max(max_energy, total_energy.item())

    # Energy should not explode
    assert max_energy < 1e6


# Test 5: Convergence
def test_convergence_simple_problem(algorithm):
    """Test convergence on simple quadratic potential."""
    # Potential: x^2, minimum at x=0

    state = algorithm.initialize()

    # Run for many steps
    for _ in range(500):
        state = algorithm.step(state)

    # Check convergence to minimum
    mean_pos = state.x.mean(dim=0)
    distance_to_min = torch.norm(mean_pos)

    assert distance_to_min < 0.5  # Should be near minimum
```

#### Step 3.3: Run Tests

```bash
# Run all tests
pytest tests/test_my_algorithm.py -v

# Run specific test
pytest tests/test_my_algorithm.py::test_convergence_simple_problem -v

# With coverage
pytest tests/test_my_algorithm.py --cov=src/fragile/my_algorithm --cov-report=term-missing

# Check coverage
# Target: > 90%
```

#### Step 3.4: Add Comparison Tests

```python
def test_compare_with_baseline():
    """Compare performance with EuclideanGas baseline."""
    from fragile.euclidean_gas import EuclideanGas

    params = MyAlgorithmParams(N=50, d=2)

    # Create both algorithms
    my_algo = MyAlgorithm(params)
    baseline = EuclideanGas(params)

    # Run both
    my_state = my_algo.initialize()
    baseline_state = baseline.initialize()

    for _ in range(200):
        my_state = my_algo.step(my_state)
        baseline_state = baseline.step(baseline_state)

    # Compare final rewards
    my_reward = my_state.mean_reward
    baseline_reward = baseline_state.mean_reward

    print(f"My algorithm: {my_reward:.4f}")
    print(f"Baseline: {baseline_reward:.4f}")

    # Your algorithm should be competitive
    # (adjust threshold based on expected performance)
    assert my_reward >= baseline_reward * 0.8  # Within 20%
```

---

### Stage 4: Visualization

**Purpose**: Create interactive visualizations

#### Step 4.1: Create Visualization File

```bash
touch src/fragile/shaolin/my_algorithm_viz.py
vim src/fragile/shaolin/my_algorithm_viz.py
```

#### Step 4.2: Implement Basic Visualization

```python
import holoviews as hv
import numpy as np
from fragile.my_algorithm import MyAlgorithm

hv.extension('bokeh')


def visualize_swarm(algorithm, n_steps=100):
    """Visualize swarm evolution."""

    # Run simulation and record
    state = algorithm.initialize()
    history = []

    for step in range(n_steps):
        state = algorithm.step(state)
        history.append({
            'positions': state.x.clone().cpu().numpy(),
            'rewards': state.reward.clone().cpu().numpy(),
            'step': step,
        })

    # Create animation
    def plot_frame(frame_idx):
        data = history[frame_idx]
        positions = data['positions']
        rewards = data['rewards']

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
            tools=['hover'],
        )

        return scatter

    # Create DynamicMap
    dmap = hv.DynamicMap(plot_frame, kdims=['frame'])
    dmap = dmap.redim.values(frame=list(range(n_steps)))

    return dmap
```

#### Step 4.3: Test Visualization

```python
# Test in Jupyter notebook or script
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams
from fragile.shaolin.my_algorithm_viz import visualize_swarm

# Create algorithm
params = MyAlgorithmParams(N=50, d=2)
algo = MyAlgorithm(params)

# Visualize
viz = visualize_swarm(algo, n_steps=100)
viz  # Display in notebook

# Or save
hv.save(viz, 'my_algorithm_animation.html')
```

---

### Stage 5: Documentation

**Purpose**: Document algorithm for users and developers

#### Step 5.1: Update Mathematical Document

```bash
vim docs/source/1_euclidean_gas/XX_my_algorithm.md
```

**Add implementation section:**
```markdown
## Implementation

### Code Structure

Implemented in `src/fragile/my_algorithm.py` as class `MyAlgorithm`.

**Inheritance**: Extends {prf:ref}`obj-euclidean-gas`.

### Key Methods

**step(state) → state**

Implements Algorithm X from §Y.Z.

**_apply_new_mechanism(state) → state**

Implements Equation (X.Y).

### Usage

```python
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams

params = MyAlgorithmParams(N=50, d=2)
algo = MyAlgorithm(params)

state = algo.initialize()
for _ in range(100):
    state = algo.step(state)

print(f"Final reward: {state.mean_reward}")
```

### Performance

Tested on standard benchmarks:
- Rastrigin (d=2): Converges in ~500 steps
- Rosenbrock (d=2): Converges in ~1000 steps
```

#### Step 5.2: Create Usage Example

```bash
touch examples/my_algorithm_example.py
vim examples/my_algorithm_example.py
```

```python
"""Example usage of MyAlgorithm."""

from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams


def main():
    """Run simple optimization."""

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

    # Run optimization
    print("Step | Mean Reward | Std Reward")
    print("-" * 40)

    for step in range(500):
        state = algo.step(state)

        if step % 50 == 0:
            print(f"{step:4d} | {state.mean_reward:11.4f} | {state.std_reward:10.4f}")

    print(f"\nFinal position: {state.x.mean(dim=0)}")


if __name__ == "__main__":
    main()
```

---

## Final Quality Checks

### Run All Checks

```bash
# 1. Format code
make style

# 2. Check style compliance
make check

# 3. Type checking
make typing

# 4. Run tests
make test

# 5. Build documentation
make build-docs

# 6. Serve documentation
make serve-docs
```

### Verify Checklist

- [ ] Mathematical spec written and reviewed
- [ ] Parameters defined with Pydantic
- [ ] Algorithm implemented with type hints
- [ ] All tests pass (coverage > 90%)
- [ ] Visualization created
- [ ] Documentation updated
- [ ] Example script works
- [ ] Code formatted (ruff)
- [ ] No type errors (mypy)

---

## Integration and Deployment

### Add to Package

Update `src/fragile/__init__.py`:

```python
from fragile.my_algorithm import MyAlgorithm
from fragile.my_algorithm_parameters import MyAlgorithmParams

__all__ = [
    ...,
    "MyAlgorithm",
    "MyAlgorithmParams",
]
```

### Update Tests

Add to test suite:
```python
# tests/__init__.py
# (No changes needed - pytest auto-discovers)
```

### Document in README

Update main README if this is a major algorithm.

---

## Time Estimates

| Stage | Time | Notes |
|-------|------|-------|
| Mathematical design | ~1-2 hours | Critical foundation |
| Parameter implementation | ~30 min | Simple with Pydantic |
| Algorithm implementation | ~2-4 hours | Depends on complexity |
| Testing | ~1-2 hours | Comprehensive tests |
| Visualization | ~1 hour | Basic visualization |
| Documentation | ~1 hour | Usage examples |
| Quality checks | ~30 min | Automated tools |
| **Total** | **~6-11 hours** | For complete algorithm |

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
