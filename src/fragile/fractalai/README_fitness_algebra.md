# Symbolic Fitness Algebra Module

## Overview

The `fitness_algebra.py` module provides a complete **symbolic algebra framework** for the emergent geometry of the Adaptive Gas, implementing all mathematical expressions from **Chapter 9** of `docs/source/08_emergent_geometry.md`.

## Purpose

This module enables:

1. **Verification** of algebraic derivations through symbolic computation
2. **Manipulation** of geometric quantities using SymPy
3. **Export** to LaTeX for documentation
4. **Generation** of optimized numerical functions
5. **Exploration** of parameter dependencies

## Installation

The module requires SymPy:

```bash
pip install sympy
```

## Quick Start

```python
from fragile.fitness_algebra import FitnessPotential, EmergentMetric

# Create fitness potential (3D, 3 walkers)
fitness = FitnessPotential(dim=3, num_walkers=3)

# Compute fitness potential
V_fit = fitness.fitness_potential_sigmoid()

# Compute metric
metric = EmergentMetric(fitness)
H = metric.hessian_V_fit(V_fit)
g = metric.metric_tensor(H)

# Compute volume element
vol = metric.volume_element(g)
```

## Module Structure

### Core Classes

#### `FitnessPotential`
Implements the fitness potential pipeline (Chapter 9.2):
- Localization weights: `localization_weights()`
- Localized moments: `localized_mean()`, `localized_variance()`
- Z-score: `z_score()`
- Fitness potential: `fitness_potential_sigmoid()`

#### `EmergentMetric`
Implements metric tensor computation (Chapters 9.3-9.5):
- Hessian: `hessian_V_fit(V_fit)`
- Metric: `metric_tensor(H)`
- Volume element: `volume_element(g)` or `volume_element_3d_explicit(H)`

#### `ChristoffelSymbols`
Computes Christoffel symbols (Chapter 9.7.7):
- Full computation: `compute(g, simplify_result=False)`
- Individual component: `get_component(Gamma, a, b, c)`

#### `GeodesicEquation`
Symbolic geodesic equation (Chapter 9.7.8):
- Equation: `equation(Gamma, component)`

### Helper Classes

#### `MeasurementFunction`
Abstract measurement function d: ℝ^d → ℝ
- Gradient: `gradient()`
- Hessian: `hessian()`

#### `RescaleFunction`
Rescale functions g_A: ℝ → [0, A]
- Sigmoid: `sigmoid(z, A)`
- Derivatives: `sigmoid_derivative(z, A, order)`

### Utility Functions

- `create_algorithmic_parameters()` — All parameters from Chapter 9.6.1
- `create_state_variables(dim)` — State space coordinates
- `export_to_latex(expr, name)` — Export to LaTeX
- `generate_numerical_function(expr, vars)` — Create numerical function

## Examples

### Example 1: Verify Sigmoid Properties

```python
from sympy import symbols, simplify
from fragile.fitness_algebra import RescaleFunction

z, A = symbols('z A', positive=True)
g = RescaleFunction.sigmoid(z, A)

# Check bounds
assert g.subs(z, 0) == A/2
assert g.limit(z, float('inf')) == A
assert g.limit(z, float('-inf')) == 0

# Check derivative at z=0
g_prime = RescaleFunction.sigmoid_derivative(z, A, order=1)
assert simplify(g_prime.subs(z, 0)) == A/4
```

### Example 2: Compute 2D Metric

```python
from sympy import symbols, Matrix, pprint
from fragile.fitness_algebra import EmergentMetric, FitnessPotential

# Create test Hessian
h11, h12, h22 = symbols('h11 h12 h22', real=True)
H = Matrix([[h11, h12], [h12, h22]])

# Compute metric
fitness = FitnessPotential(dim=2, num_walkers=2)
metric = EmergentMetric(fitness)
g = metric.metric_tensor(H)

pprint(g)
# Output:
# ⎡ε_Σ + h₁₁     h₁₂   ⎤
# ⎢                    ⎥
# ⎣   h₁₂     ε_Σ + h₂₂⎦
```

### Example 3: 3D Volume Element

```python
from sympy import symbols, Matrix, simplify
from fragile.fitness_algebra import EmergentMetric, FitnessPotential

# Diagonal Hessian
h11, h22, h33 = symbols('h11 h22 h33', real=True)
H = Matrix([[h11, 0, 0], [0, h22, 0], [0, 0, h33]])

fitness = FitnessPotential(dim=3, num_walkers=2)
metric = EmergentMetric(fitness)

# Explicit 3D formula (Chapter 9.7.6)
vol = metric.volume_element_3d_explicit(H)

# Should be sqrt((h11+ε)(h22+ε)(h33+ε))
epsilon_Sigma = fitness.params['epsilon_Sigma']
expected = sqrt((h11 + epsilon_Sigma) *
                (h22 + epsilon_Sigma) *
                (h33 + epsilon_Sigma))

assert simplify(vol - expected) == 0
```

### Example 4: Export to LaTeX

```python
from sympy import symbols, Matrix
from fragile.fitness_algebra import export_to_latex

h11, h12, h22 = symbols('h_{11} h_{12} h_{22}')
epsilon_Sigma = symbols('epsilon_Sigma', positive=True)

g = Matrix([[h11 + epsilon_Sigma, h12],
            [h12, h22 + epsilon_Sigma]])

latex = export_to_latex(g, name='g')
print(latex)
# Output: g = \left[\begin{matrix}...
```

## Demonstrations

Run the comprehensive demonstration:

```bash
python examples/symbolic_geometry_demo.py
```

This demonstrates:
1. Sigmoid rescale function
2. Localization weights
3. 2D metric tensor
4. 3D volume element
5. LaTeX export
6. Hessian decomposition

## Testing

Run the test suite:

```bash
pytest tests/test_fitness_algebra.py -v
```

Tests verify:
- Parameter creation
- Rescale function properties
- Localization weight normalization
- Metric regularization
- Volume element formulas
- 2D and 3D cases

## Performance Notes

### Computational Complexity

The symbolic computations can be **very expensive** for:
- Large number of walkers (k > 5)
- High dimensions (d > 3)
- Full simplification of complex expressions

**Recommendations:**
1. Use **small dimensions** (2D or 3D) for symbolic work
2. Use **few walkers** (k ≤ 3) for full pipeline
3. **Avoid automatic simplification** for large expressions
4. Use **numerical evaluation** (lambdify) for production

### Example Timings

On a modern CPU:
- 2D fitness potential, k=2: ~1 second
- 3D fitness potential, k=2: ~5 seconds
- 3D metric tensor computation: ~10 seconds
- 3D Christoffel symbols (all): ~several minutes
- Full 3D with k=5: **too slow** (hours)

## API Reference

### Class Hierarchy

```
MeasurementFunction
  ├─ gradient()
  └─ hessian()

RescaleFunction (static methods)
  ├─ sigmoid(z, A)
  └─ sigmoid_derivative(z, A, order)

FitnessPotential
  ├─ localization_kernel(x_i, x_j, rho)
  ├─ localization_weights()
  ├─ localized_mean(weights?)
  ├─ localized_variance(weights?, mu?)
  ├─ z_score(weights?)
  └─ fitness_potential_sigmoid(weights?)

EmergentMetric
  ├─ gradient_V_fit(V_fit)
  ├─ hessian_V_fit(V_fit)
  ├─ metric_tensor(H)
  ├─ volume_element(g)
  └─ volume_element_3d_explicit(H)

ChristoffelSymbols
  ├─ compute(g, simplify_result=False)
  └─ get_component(Gamma, a, b, c)

GeodesicEquation
  └─ equation(Gamma, component)
```

### Parameter List

All algorithmic parameters from Chapter 9.6.1:

```python
params = create_algorithmic_parameters()

# Measurement: rho, kappa_var_min, A, epsilon_d
# Diffusion: epsilon_Sigma, sigma_v, delta
# Adaptive: epsilon_F, nu
# Kinetic: gamma, tau, lambda_v, lambda_alg
# Cloning: alpha, beta, eta, epsilon_rescale
# Confinement: kappa_conf
# Lyapunov: alpha_x, alpha_v, alpha_D, alpha_R
```

## Mathematical Correspondence

This module implements formulas from:

| Chapter Section | Implementation |
|:---|:---|
| 9.2: Fitness Potential | `FitnessPotential` class |
| 9.3: Hessian via Chain Rule | `EmergentMetric.hessian_V_fit()` |
| 9.4: Regularized Metric | `EmergentMetric.metric_tensor()` |
| 9.5: Geodesics | `GeodesicEquation` class |
| 9.6: Complete Pipeline | All parameter dependencies |
| 9.7.1-9.7.2: 3D Setup | `dim=3` examples |
| 9.7.3: Gradient | `gradient_V_fit()` |
| 9.7.4: Hessian 3×3 | Matrix operations |
| 9.7.5: Metric 3×3 | `metric_tensor()` |
| 9.7.6: Volume Element | `volume_element_3d_explicit()` |
| 9.7.7: Christoffel Symbols | `ChristoffelSymbols.compute()` |
| 9.7.8: Geodesic Equation | `GeodesicEquation.equation()` |

## Limitations

1. **Symbolic only**: No numerical integration of geodesics
2. **Gaussian kernel only**: Other kernels require modification
3. **Sigmoid rescale**: Other rescales require adding methods
4. **No visualization**: Use SymPy plotting or export to numerical
5. **Memory intensive**: Large expressions can exhaust RAM

## Future Enhancements

Possible extensions:
- [ ] Numerical geodesic integration
- [ ] Riemannian curvature tensors
- [ ] Scalar curvature computation
- [ ] Sectional curvature
- [ ] Connection to information geometry
- [ ] Parallel transport
- [ ] Exponential map
- [ ] Other kernels (Epanechnikov, etc.)
- [ ] Visualization utilities

## References

- **Chapter 9** of `docs/source/08_emergent_geometry.md`
- SymPy documentation: https://docs.sympy.org/
- Riemannian geometry: Lee, *Introduction to Riemannian Manifolds*
- Information geometry: Amari, *Information Geometry and Its Applications*

## License

Same as the Fragile project.

## Author

Generated by Claude (Anthropic) based on Chapter 9 specifications.
Date: 2025-10-09
