# Algorithm Development - Troubleshooting

Common issues and solutions when developing Gas algorithms.

---

## Implementation Issues

### Issue: Shape mismatch errors

**Symptoms**: `RuntimeError: The size of tensor a (10) must match the size of tensor b (2)`

**Cause**: Tensor dimensions don't align

**Solution**: Check shapes at each step
```python
# Debug shapes
print(f"x shape: {x.shape}")  # Expected: [N, d]
print(f"v shape: {v.shape}")  # Expected: [N, d]

# Common mistake: Missing dimension
force = x.mean(dim=0)  # [d]
x = x + force  # Error! [N, d] + [d] needs broadcasting

# Fix: Add dimension
force = x.mean(dim=0).unsqueeze(0)  # [1, d]
x = x + force  # Works: [N, d] + [1, d] broadcasts
```

---

### Issue: Gradients not detaching

**Symptoms**: Memory grows over time, eventually OOM

**Cause**: PyTorch tracking gradients unnecessarily

**Solution**: Use `.clone().detach()` or `with torch.no_grad()`
```python
# ❌ WRONG: Accumulates gradient history
history.append({'positions': state.x})

# ✅ CORRECT: Detach from computation graph
history.append({'positions': state.x.clone().detach().cpu()})

# Or wrap in no_grad context
with torch.no_grad():
    positions = state.x.clone().cpu().numpy()
```

---

### Issue: Vectorization breaks for edge cases

**Symptoms**: Works for N>1 but fails for N=1

**Cause**: Operations that expect multiple elements

**Solution**: Handle N=1 case explicitly
```python
# Example: Mean operation
if state.x.shape[0] == 1:
    x_mean = state.x[0]  # [d]
else:
    x_mean = state.x.mean(dim=0)  # [d]

# Or use keepdim
x_mean = state.x.mean(dim=0, keepdim=True)  # Always [1, d]
```

---

### Issue: NaN or Inf values appear

**Symptoms**: `RuntimeError: Function 'XXX' returned nan values` or algorithm diverges

**Cause**: Numerical instability

**Solution 1 - Add epsilon for stability**:
```python
# ❌ WRONG: Division by zero
normalized = x / x.norm(dim=1).unsqueeze(1)

# ✅ CORRECT: Add epsilon
eps = 1e-8
normalized = x / (x.norm(dim=1).unsqueeze(1) + eps)
```

**Solution 2 - Clamp values**:
```python
# Clamp to safe range
energies = torch.clamp(energies, min=-100, max=100)
```

**Solution 3 - Check for NaNs**:
```python
# Debug NaNs
if torch.isnan(state.x).any():
    print("NaN detected in positions!")
    import pdb; pdb.set_trace()
```

---

## Testing Issues

### Issue: Tests fail intermittently

**Symptoms**: Test passes sometimes, fails other times

**Cause**: Random initialization or stochastic dynamics

**Solution**: Set random seed
```python
@pytest.fixture
def algorithm():
    """Create algorithm with fixed seed."""
    torch.manual_seed(42)
    params = MyAlgorithmParams(N=10, d=2)
    return MyAlgorithm(params)
```

---

### Issue: Convergence test fails

**Symptoms**: `assert distance_to_min < 0.5` fails

**Cause**: Algorithm needs more steps or different parameters

**Solution 1 - More steps**:
```python
# Increase iterations
for _ in range(1000):  # Was 500
    state = algorithm.step(state)
```

**Solution 2 - Adjust parameters**:
```python
# Use parameters conducive to convergence
params = MyAlgorithmParams(
    N=100,  # More walkers
    d=2,
    gamma=0.8,  # Higher friction (faster convergence)
    tau=0.005,  # Smaller time step (more stable)
)
```

**Solution 3 - Relax tolerance**:
```python
# Relax bound
assert distance_to_min < 1.0  # Was 0.5
```

---

### Issue: Coverage too low

**Symptoms**: Coverage report shows < 90%

**Cause**: Missing tests for some code paths

**Solution**: Check coverage report
```bash
pytest tests/test_my_algorithm.py --cov=src/fragile/my_algorithm --cov-report=html

# Open htmlcov/index.html to see uncovered lines

# Add tests for missing lines
```

---

## Visualization Issues

### Issue: Visualization shows blank plot

**Symptoms**: Plot renders but shows nothing

**Cause**: Data not in correct format or empty

**Solution**: Debug data
```python
def plot_frame(frame_idx):
    data = history[frame_idx]

    # Debug
    print(f"Frame {frame_idx}")
    print(f"Positions shape: {data['positions'].shape}")
    print(f"Positions range: {data['positions'].min()}, {data['positions'].max()}")

    # Check if data is empty
    if len(data['positions']) == 0:
        return hv.Text(0, 0, 'No data')

    # Create plot...
```

---

### Issue: Animation doesn't update

**Symptoms**: Shows first frame but doesn't animate

**Cause**: DynamicMap not configured correctly

**Solution**: Check frame range
```python
# ❌ WRONG: Missing redim.values
dmap = hv.DynamicMap(plot_frame, kdims=['frame'])

# ✅ CORRECT: Specify frame values
dmap = hv.DynamicMap(plot_frame, kdims=['frame'])
dmap = dmap.redim.values(frame=list(range(n_steps)))
```

---

### Issue: Matplotlib used instead of HoloViews

**Symptoms**: Static plots instead of interactive

**Cause**: Wrong import

**Solution**: Use HoloViz stack
```python
# ❌ WRONG: Don't use matplotlib
import matplotlib.pyplot as plt

# ✅ CORRECT: Use HoloViews
import holoviews as hv
hv.extension('bokeh')  # 2D
# or
hv.extension('plotly')  # 3D
```

---

## Parameter Issues

### Issue: Pydantic validation fails

**Symptoms**: `ValidationError: 1 validation error for MyAlgorithmParams`

**Cause**: Parameter doesn't meet constraints

**Solution**: Check Field constraints
```python
# Error message shows which field failed
ValidationError: N must be greater than 0

# Fix: Provide valid value
params = MyAlgorithmParams(N=50)  # Not N=-1
```

---

### Issue: Parameters are mutable

**Symptoms**: Parameters change after initialization

**Cause**: `frozen=False` in Config

**Solution**: Freeze parameters
```python
class MyAlgorithmParams(BaseModel):
    ...

    class Config:
        frozen = True  # ← Add this
        extra = "forbid"
```

---

## Performance Issues

### Issue: Algorithm is slow

**Symptoms**: Takes minutes for 100 steps

**Cause**: Not vectorized, using loops

**Solution**: Vectorize operations
```python
# ❌ SLOW: Loop over walkers
for i in range(N):
    state.x[i] = state.x[i] + state.v[i] * dt

# ✅ FAST: Vectorized
state.x = state.x + state.v * dt
```

---

### Issue: Memory usage grows

**Symptoms**: Memory increases over time, eventual OOM

**Cause**: Not detaching tensors or accumulating gradients

**Solution**: Detach and cleanup
```python
# Detach from computation graph
history.append({
    'positions': state.x.clone().detach().cpu(),
    'rewards': state.reward.clone().detach().cpu(),
})

# Or use no_grad
with torch.no_grad():
    state = algorithm.step(state)
```

---

## Documentation Issues

### Issue: Math rendering broken

**Symptoms**: Raw LaTeX visible in built docs

**Cause**: Missing blank line before `$$`

**Solution**: Run formatting tool
```bash
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/XX_my_algorithm.md
```

---

### Issue: Cross-reference broken

**Symptoms**: Link shows as `?` or not clickable

**Cause**: Referenced label doesn't exist

**Solution**: Check glossary
```bash
# Find correct label
cat docs/glossary.md | grep "euclidean-gas"

# Update reference
{prf:ref}`obj-euclidean-gas`  # Correct
```

---

## Integration Issues

### Issue: Import fails

**Symptoms**: `ModuleNotFoundError: No module named 'fragile.my_algorithm'`

**Cause**: Package not installed or in wrong location

**Solution**: Reinstall package
```bash
# Reinstall in editable mode
pip install -e .

# Or use UV
uv sync
```

---

### Issue: Type hints not recognized

**Symptoms**: mypy reports type errors

**Cause**: Missing type hints or wrong types

**Solution**: Add type hints
```python
# ❌ WRONG: No type hints
def step(self, state):
    return state

# ✅ CORRECT: Full type hints
def step(self, state: SwarmState) -> SwarmState:
    return state
```

---

## Debugging Tips

### Enable debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use IPython debugger
import ipdb; ipdb.set_trace()
```

### Inspect shapes

```python
# Print all shapes
def debug_shapes(state: SwarmState):
    print(f"x: {state.x.shape}")
    print(f"v: {state.v.shape}")
    print(f"reward: {state.reward.shape}")
    print(f"alive_mask: {state.alive_mask.shape}")

# Call before operations
debug_shapes(state)
```

### Check for common issues

```python
# Check for NaNs
assert not torch.isnan(state.x).any(), "NaN in positions"

# Check for Infs
assert not torch.isinf(state.x).any(), "Inf in positions"

# Check shapes
assert state.x.shape == (N, d), f"Wrong shape: {state.x.shape}"

# Check bounds
assert torch.all(state.x >= -10) and torch.all(state.x <= 10), "Positions out of bounds"
```

---

## Getting Help

If issues persist:

1. **Check CLAUDE.md**: Project coding standards
2. **Check algorithm hierarchy**: Understand base classes
3. **Read existing algorithms**: `euclidean_gas.py`, `adaptive_gas.py`
4. **Run examples**: Test existing algorithms first
5. **Ask for help**: Provide error message and minimal reproducible example

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- **CLAUDE.md**: Project conventions
- **PyTorch Documentation**: https://pytorch.org/docs/
