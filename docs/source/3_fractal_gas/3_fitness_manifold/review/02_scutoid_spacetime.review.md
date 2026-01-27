# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-adaptive-diffusion-tensor-latent (emergent metric g = H + eps I)
  - def-fractal-set-sde (continuous SDE evolution of walkers)
  - def-fractal-set-cst-edges / def-fractal-set-cloning-score (CST edges and cloning events)
  - thm-fractal-adaptive-sprinkling (QSD sampling / local Poisson approximation)

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 0
- Minor: 0
- Notes: 0
- Primary themes: remaining lower-bound validity; resolved CST-compatible slab metric, cloning cell consistency, boundary correspondence regularity, prism argument, oriented counting, curvature note

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-006 | Thm. thm-omega-n-lower-bound | Major | Invalid inference | Output-size argument does not yield update lower bound for dynamic maintenance |

## Detailed findings

### [E-006] Output-size lower bound does not apply to dynamic update
- Location: Theorem thm-omega-n-lower-bound (lower bound proof)
- Severity: Major
- Type: Invalid inference (secondary: Conceptual)
- Claim (paraphrase): Any algorithm that updates a Voronoi/Delaunay complex after arbitrary point movements must take Omega(|DT|) time; therefore the online update is optimal.
- Why this is an error in the framework: The proof uses an output-size argument, which only applies when the full tessellation (all simplices and adjacency) must be explicitly output each step. Dynamic maintenance can update internal structures without touching every simplex when the combinatorial structure is unchanged. The rotation example only forces updating vertex coordinates (O(N)), not Omega(|DT|) when |DT| is superlinear.
- Impact on downstream results: The claimed optimality of the online algorithm is overstated; the lower bound does not justify it in the dynamic-update model.
- Fix guidance (step-by-step):
  1. Restate the theorem as an output-size lower bound for explicit reporting/serialization of the tessellation, not for dynamic maintenance.
  2. If optimality is desired, prove a lower bound that matches the update model used (e.g., Omega(N) for arbitrary coordinate updates, or Omega(k) for the number of affected simplices).
  3. Adjust the conclusion and Key Takeaways to reflect the refined lower bound.
- Required new assumptions/permits (if any): A framework lemma establishing a dynamic lower bound under the CST update model.
- Framework-first proof sketch for the fix: For explicit-output models, the size of the reported complex is Theta(|DT|), so any algorithm must spend Omega(|DT|) time to emit it. For dynamic models, lower bounds should be tied to the number of vertices or affected simplices rather than output size.
- Validation plan: Ensure the revised theorem matches the computational model used in Algorithm alg-online-triangulation-update and in the TLDR optimality claim.

## Scope restrictions and clarifications
- The O(p_clone N log N) term requires maintaining a spatial index; otherwise point location is O(N^{1/d}) (as noted in Jump-and-Walk).
- Geodesic uniqueness for ruled surfaces requires convex normal neighborhoods (or Safe Harbor) in the slab metric $\bar{G}_k$.

## Proposed edits (optional)
- Restate the lower-bound theorem to match the dynamic-update model (or restrict it to explicit-output settings).

## Open questions
- Should the lower-bound theorem be reframed as an output-size bound, or do we want a stronger dynamic-update lower bound in this chapter?

## Resolved findings (post-update)
- E-001: Spacetime geodesics now defined via the CST-compatible slab metrics
  (Definitions {prf:ref}`def-scutoid-slab-metric` and {prf:ref}`def-scutoid-path-length`).
- E-002: Scutoid cells reindexed by walker ID, consistent with CST edges and theorem statements.
- E-003: Boundary correspondence now assumes rectifiability and finite positive measure in Safe Harbor regions.
- E-004: Prism obstruction now uses a face-preserving bijection argument instead of a generic homeomorphism claim.
- E-005: Curvature note now presents two independent proof routes (analytic Gevrey-1 bounds and hypostructure metatheorem chain) without mixing.
- E-007: K_total interpretation clarified as oriented neighbor changes with factor-of-two note.
