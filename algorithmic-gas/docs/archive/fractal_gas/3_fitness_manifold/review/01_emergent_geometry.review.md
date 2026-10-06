# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/01_emergent_geometry.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/01_emergent_geometry.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-adaptive-diffusion-tensor-latent: Sigma_reg = (H + epsilon_Sigma I)^(-1/2) and D_reg = (H + epsilon_Sigma I)^(-1).
  - thm-uniform-ellipticity-latent: c_min I <= D_reg <= c_max I under spectral bounds.
  - lem-geometric-drift-latent: geometric drift term that yields the Riemannian invariant measure.
  - lem-operator-lipschitz-inv-sqrt-latent: inverse square-root Lipschitz bound on SPD matrices.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 1
- Notes: 0
- Primary themes: Geometric quadrature error bounds need explicit smoothness and cell-size assumptions.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-004 | sec-fan-triangulation, sec-tetrahedral-decomposition | Minor | Scope restriction | Error orders for Riemannian area/volume need explicit smoothness and size assumptions on g. |

## Detailed findings

### [E-004] Riemannian area/volume quadrature errors lack explicit smoothness assumptions
- Location: sec-fan-triangulation and sec-tetrahedral-decomposition
- Severity: Minor
- Type: Scope restriction
- Claim (paraphrase): Fan triangulation yields O(diam(C)^2) error for smooth g; tetrahedral volume formula uses g at the centroid without stated error conditions.
- Why this is an error in the framework: The stated error orders require C^2 (or higher) regularity of g and a small-diameter assumption on the cell; these are not explicitly stated in the algorithm definitions.
- Impact on downstream results: Minor, but readers may treat the error claims as unconditional.
- Fix guidance (step-by-step):
  1. Add a hypothesis line to each algorithm: "Assume g is C^2 with bounded derivatives on the cell and diam(C) is small."
  2. State the error bound in terms of a specific derivative norm (e.g., ||nabla g|| or ||nabla^2 g||).
  3. Note that the centroid approximation is first-order in cell diameter.
- Required new assumptions/permits (if any): None beyond regularity and size conditions already used elsewhere (Safe Harbor).
- Framework-first proof sketch for the fix: Taylor expand g around the centroid; the leading error is controlled by the second derivative and the cell diameter.
- Validation plan: Check the error scaling on a known curved metric (e.g., small perturbation of Euclidean).

## Scope restrictions and clarifications
- The Riemannian quadrature error claims require explicit C^2 (or higher) bounds on g and a small-cell regime.

## Proposed edits (optional)
- Annotate the fan triangulation and tetrahedral formulas with C^2 and small-diameter assumptions.

## Open questions
- Should the quadrature error assumptions be consolidated into a single labeled lemma to reuse across the tessellation section?
