# Experiment Framework: Modular Design

## Overview

The experiment framework separates **computational logic** from **visualization**, enabling:
- Terminal debugging without Jupyter
- Code reuse across multiple notebooks
- Independent testing of numerical methods
- Faster iteration during development

## Architecture

```
src/fragile/experiments/
├── __init__.py                    # Module exports
└── convergence_analysis.py        # All computational logic

experiments/experiment_scripts/
├── README.md                       # Documentation
├── run_convergence_experiment.py  # Full experiment (5000 steps)
└── test_convergence_quick.py      # Quick test (500 steps)

experiments/
├── 03_exponential_convergence_to_qsd.ipynb  # Notebook (visualization only)
└── 03_exponential_convergence_to_qsd_old.ipynb  # Original (backup)
```

## Core Components

### 1. Potential Creation
```python
from fragile.experiments import create_multimodal_potential

potential, target_mixture = create_multimodal_potential(
    dims=2,
    n_gaussians=3,
    seed=42
)
```

### 2. Convergence Analysis
```python
from fragile.experiments import ConvergenceAnalyzer, ConvergenceMetrics

analyzer = ConvergenceAnalyzer(
    target_mixture=target_mixture,
    target_centers=centers,
    target_weights=weights
)

# Analyze single state
metrics_dict = analyzer.analyze_state(state, time=100)
# Returns: {time, kl_divergence, wasserstein_distance, lyapunov_value, ...}
```

### 3. Full Experiment Runner
```python
from fragile.experiments import ConvergenceExperiment

experiment = ConvergenceExperiment(
    gas=gas,
    analyzer=analyzer,
    save_snapshots_at=[0, 100, 500, 1000, 5000]
)

metrics, snapshots = experiment.run(
    n_steps=5000,
    x_init=x_init,
    v_init=v_init,
    measure_every=10,
    verbose=True
)
```

### 4. Convergence Summary
```python
summary = experiment.get_convergence_summary()
# Returns: {
#   'kl_convergence_rate': κ,
#   'kl_half_life': t_1/2,
#   'final_kl': D_KL(T),
#   'w2_convergence_rate': κ_W2,
#   'lyapunov_decay_rate': κ_lyap,
#   ...
# }
```

## Workflow

### Development Workflow
1. **Write computational logic** in `convergence_analysis.py`
2. **Test in terminal** using `experiment_scripts/test_convergence_quick.py`
3. **Debug with prints** and verify numerical correctness
4. **Run full experiment** with `run_convergence_experiment.py`
5. **Import in notebook** for visualization only

### Example: Terminal Testing
```bash
cd experiments/experiment_scripts
python test_convergence_quick.py
```

Output:
```
Quick convergence test (500 steps)
============================================================

[1/4] Creating potential...
  ✓ Created 3-mode mixture

[4/4] Running 500 steps...
  Step   100: KL=0.0006, W2=3.7180
  Step   500: KL=0.0005, W2=4.9917

✓ Experiment complete!

Exponential fit:
  κ (rate): 0.000088
  Half-life: inf steps
```

### Example: Notebook Usage
```python
# In Jupyter notebook (visualization only)
from fragile.experiments import (
    create_multimodal_potential,
    ConvergenceExperiment,
    ConvergenceAnalyzer,
)

# Create potential (one line!)
potential, target_mixture = create_multimodal_potential(dims=2, n_gaussians=3, seed=42)

# Run experiment (computational logic)
experiment = ConvergenceExperiment(gas, analyzer, save_snapshots_at=[...])
metrics, snapshots = experiment.run(n_steps=5000, verbose=True)

# Visualize results (notebook-specific code)
import matplotlib.pyplot as plt
plt.semilogy(metrics.time, metrics.kl_divergence)
plt.show()
```

## Benefits

### 1. Terminal Debugging
- No need to restart Jupyter kernel
- Fast print-based debugging
- Works in WSL/headless environments
- Easy to profile performance

### 2. Code Reuse
- Same code in multiple notebooks
- DRY (Don't Repeat Yourself)
- Single source of truth
- Easy to maintain

### 3. Independent Testing
- Unit tests for computational logic
- Verify numerical correctness
- Regression testing
- CI/CD integration

### 4. Separation of Concerns
- Computational logic: `convergence_analysis.py`
- Visualization: Jupyter notebooks
- Clear boundaries
- Easy to understand

## Testing

### Quick Test (15 seconds)
```bash
python test_convergence_quick.py
```
- 50 walkers
- 500 steps
- Validates basic functionality

### Full Test (2 minutes)
```bash
python run_convergence_experiment.py
```
- 100 walkers
- 5000 steps
- Complete convergence analysis

### Unit Tests (future)
```bash
pytest tests/test_convergence_analysis.py
```
- Test individual components
- Mock GeometricGas
- Verify metrics computation

## Implementation Details

### ConvergenceMetrics Dataclass
```python
@dataclass
class ConvergenceMetrics:
    time: list[int]
    kl_divergence: list[float]
    wasserstein_distance: list[float]
    lyapunov_value: list[float]
    mean_position: list[np.ndarray]
    variance_position: list[float]
    
    def get_valid_kl_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Filter out inf/nan values."""
        ...
    
    def fit_exponential_decay(self, metric: str, fit_start_time: int):
        """Fit y = C * exp(-κ * t) to metric."""
        ...
```

### ConvergenceAnalyzer Methods
```python
class ConvergenceAnalyzer:
    def compute_kl_divergence_kde(self, samples, n_grid=1000) -> float:
        """KL(empirical || target) using KDE."""
        
    def compute_wasserstein_distance(self, samples) -> float:
        """W_2(empirical, target) using mean proxy."""
        
    def compute_lyapunov_function(self, state) -> float:
        """V_total = Var(x) + Var(v)."""
```

### ConvergenceExperiment Orchestration
```python
class ConvergenceExperiment:
    def run(self, n_steps, x_init=None, v_init=None, 
            measure_every=10, verbose=True):
        """Run full experiment with metric tracking."""
        
        # Initialize
        state = self.gas.initialize_state(x_init, v_init)
        
        # Main loop
        for step in range(n_steps):
            _, state = self.gas.step(state)
            
            if (step + 1) % measure_every == 0:
                metrics_dict = self.analyzer.analyze_state(state, step + 1)
                self.metrics.add_snapshot(**metrics_dict)
        
        return self.metrics, self.snapshots
```

## Future Extensions

### 1. Parameter Sweeps
```python
def parameter_sweep(param_name, values, base_params):
    """Sweep parameter and collect convergence rates."""
    results = []
    for val in values:
        params = update_param(base_params, param_name, val)
        experiment = ConvergenceExperiment(...)
        metrics, _ = experiment.run(...)
        results.append(metrics.fit_exponential_decay())
    return results
```

### 2. Comparison Studies
```python
def compare_algorithms(algorithms, potential):
    """Compare Euclidean vs Geometric Gas."""
    results = {}
    for name, gas_class in algorithms.items():
        gas = gas_class(params)
        experiment = ConvergenceExperiment(gas, ...)
        results[name] = experiment.run(...)
    return results
```

### 3. Adaptive Timestep
```python
def adaptive_timestep_experiment(gas, target_kl=0.01):
    """Run until KL < threshold."""
    while metrics.kl_divergence[-1] > target_kl:
        _, state = gas.step(state)
        # Track metrics...
    return metrics, total_steps
```

## Lessons Learned

1. **Separate computational logic from visualization early**
   - Makes debugging much easier
   - Enables terminal-based testing
   - Facilitates code reuse

2. **Use dataclasses for metrics**
   - Clear structure
   - Type hints
   - Easy to extend

3. **Factory functions for common setups**
   - `create_multimodal_potential()` saves boilerplate
   - Consistent defaults
   - Easy to customize

4. **Progressive enhancement**
   - Start with simple scripts
   - Add analysis methods
   - Create experiment runners
   - Build visualization on top

## References

- Main implementation: `src/fragile/experiments/convergence_analysis.py`
- Scripts: `experiments/experiment_scripts/`
- Notebook: `experiments/03_exponential_convergence_to_qsd.ipynb`
- Documentation: This file

---

**Created**: 2025-10-18  
**Purpose**: Document modular experiment framework design
