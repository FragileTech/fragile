# Math Errors Report

Scope: Reviewed core mathematical code modules and theory utilities first (`src/fragile/core/*`, `src/fragile/fractalai/core/*`, `src/fragile/fractalai/theory/*`, `src/fragile/theory/*`). Findings below focus on mathematical definitions, dimensional consistency, logical/proof alignment, and type/dimension mismatches that affect correctness. Line numbers refer to current file contents.

---

## src/fragile/core/layers/gauge.py

- **Lines 207-210 -- SU(2) chiral projector uses a non-Pauli sigma_2 matrix**
  - **Severity:** major
  - **Issue:** `sigma_2` is defined as `[[0, -1], [1, 0]]`, which is real and squares to `-I`. The Pauli sigma_2 for SU(2) is `[[0, -i], [i, 0]]`, which is Hermitian and squares to `+I`. Using the real antisymmetric matrix breaks the Pauli algebra and makes the projector `P = (I + n * sigma) / 2` non-idempotent (`P^2 != P`). This is not a valid SU(2) chiral projector.
  - **Suggested fix:** Use complex tensors and define sigma_2 with imaginary entries (e.g., `[[0, -1j], [1j, 0]]`). Ensure `psi_doublet` and related tensors are complex-valued. If a real-valued 2x2 representation is intended, rename and adjust the algebra to a real Clifford/SO(2) analog instead of SU(2).

---

## src/fragile/fractalai/core/benchmarks.py

- **Lines 46-49 -- Rosenbrock function drops the last coordinate**
  - **Severity:** major
  - **Issue:** The Rosenbrock implementation uses `x[:, :-2]` and `x[:, 1:-1]`, which omits the last dimension. For 2D inputs this yields an empty sum and returns `0` for all inputs. The canonical Rosenbrock sum is `sum_{i=1}^{d-1} [100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]`.
  - **Suggested fix:** Use `x[:, :-1]` and `x[:, 1:]` in both terms:
    ```python
    return 100 * torch.sum((x[:, 1:] - x[:, :-1] ** 2) ** 2, dim=1) + torch.sum(
        (x[:, :-1] - 1) ** 2, dim=1
    )
    ```

---

## src/fragile/fractalai/core/scutoids.py

- **Lines 563-617 + 885-913 -- Ricci scalar uses boundary length instead of cell area (dimension mismatch)**
  - **Severity:** major
  - **Issue:** `compute_ricci_scalars()` uses `R = delta / (C(d) * Vol(dV))` and `_compute_boundary_volumes()` returns perimeter for 2D cells. In 2D Regge calculus, curvature at a vertex scales as `R = 2 * delta / A`, where `A` is the dual cell area (not perimeter). Using boundary length gives the wrong units and magnitude.
  - **Suggested fix:** Compute Voronoi polygon area for 2D cells (and dual volume for general d), then use `R = 2 * delta / A` (or update the constant to match the intended normalization). If a boundary-based surrogate is intended, document it as such rather than calling it Ricci scalar.

- **Lines 1249-1257 -- Deficit angles computed with 2*pi for boundary vertices**
  - **Severity:** major
  - **Issue:** The deficit angle is always computed as `delta = 2*pi - sum(theta)`. For boundary vertices (cells on the convex hull or unbounded Voronoi regions), the correct flat reference angle is `pi`, not `2*pi`. This overestimates curvature near boundaries.
  - **Suggested fix:** Detect boundary vertices (e.g., cells with `-1` in region, or Delaunay hull vertices) and use `delta = pi - sum(theta)`, or exclude those vertices from curvature statistics.

- **Lines 900-903 -- Unbounded Voronoi cells assigned boundary volume = 1.0 (arbitrary)**
  - **Severity:** minor
  - **Issue:** When a cell has no vertices (unbounded region), the boundary volume is hard-coded to `1.0`. Combined with the curvature formula, this yields arbitrary Ricci values for boundary/unbounded cells.
  - **Suggested fix:** Mark these cells as invalid (NaN) and skip them in curvature statistics, or clip the Voronoi diagram to the domain bounds and compute finite boundary measures.

- **Lines 345-353 and 1088-1094 + 1191-1194 -- Time stamps ignore `recorded_steps`**
  - **Severity:** major
  - **Issue:** Time values are set as `t_idx * record_every`, which is wrong for the final recorded step when `n_steps` is not a multiple of `record_every`. This mis-stamps scutoid time intervals and any time-dependent geometry.
  - **Suggested fix:** Use `history.recorded_steps[t_idx]` (and multiply by `history.delta_t` if physical time is intended) instead of `t_idx * record_every`.

---

## src/fragile/fractalai/theory/qsd_variance.py

- **Lines 188-194 -- Potential force disabled while claiming QSD of confining potential**
  - **Severity:** major
  - **Issue:** `KineticOperator` is configured with `use_potential_force=False` even though the experiment narrative assumes a confining quadratic potential. This means the simulated dynamics are not the Langevin process with potential `U(x)`, so the sampled distribution is not the QSD described in the theory.
  - **Suggested fix:** Set `use_potential_force=True` for QSD experiments tied to the confining potential, or explicitly state that the experiment measures a different (cloning-only) dynamics.

- **Lines 202-206 -- CloneOperator initialized with unsupported parameters**
  - **Severity:** major
  - **Issue:** `CloneOperator` is called with `alpha_reward`, `lambda_alg`, and `alpha_rest`, which are not in its constructor. This is a parameter mismatch that will raise at runtime and prevents the experiment from running.
  - **Suggested fix:** Replace with the valid parameters: `p_max`, `epsilon_clone`, `sigma_x`, and `alpha_restitution`.

- **Lines 92-105 -- D_max taken from sample diameter, not domain diameter**
  - **Severity:** major
  - **Issue:** The phase-space packing lemma uses `D_max` as a fixed domain diameter (or known support bound). The code computes `D_max` as the maximum sample pairwise distance, which underestimates the true bound and inflates the variance ratio `Var_h / D_max^2`. This biases conclusions toward the high-variance regime.
  - **Suggested fix:** Use the known bound from `TorchBounds` (e.g., `sqrt(sum_i (high_i-low_i)^2)`) and an explicit velocity bound for the phase-space diameter, or rename the ratio as empirical and adjust the lemma accordingly.

---

## src/fragile/fractalai/theory/qsd_variance_sweep.py

- **Lines 294-299 -- CloneOperator initialized with unsupported parameters**
  - **Severity:** major
  - **Issue:** Same mismatch as in `qsd_variance.py`; `CloneOperator` does not accept `alpha_reward`, `lambda_alg`, or `alpha_rest`.
  - **Suggested fix:** Use `p_max`, `epsilon_clone`, `sigma_x`, and `alpha_restitution`.

---

## Summary of Highest-Risk Issues

- **Major correctness errors:** Rosenbrock benchmark formula, SU(2) chiral projector algebra, scutoid curvature normalization (boundary vs area), QSD experiments using non-matching dynamics and invalid CloneOperator parameters.
- **Geometry/proof alignment risks:** Boundary handling in Regge deficit angles and arbitrary boundary volume defaults can distort curvature results and invalidate comparisons with the Volume 3 curvature claims.

If you want, I can propose concrete patches for each item or add small validation tests (e.g., Rosenbrock correctness at known optima, projector idempotency checks, curvature dimension checks).
