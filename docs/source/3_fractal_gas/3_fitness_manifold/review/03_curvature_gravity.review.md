# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/03_curvature_gravity.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/03_curvature_gravity.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-affine-connection: Levi-Civita connection from the emergent metric g.
  - def-holonomy: parallel transport and holonomy around loops.
  - def-regularity-conditions: spacing and smoothness conditions for Voronoi cells.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 1
- Notes: 0
- Primary themes: topology-change claims need boundary assumptions; notation for plaquette area is ambiguous.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | sec-curvature-topology (thm-curvature-topology, prop-curvature-change-2d) | Moderate | Scope restriction | Euler characteristic arguments need compactness/boundary conditions. |
| E-002 | def-scutoid-plaquette | Minor | Notation conflict | Constant c and bivector notation are undefined/ambiguous. |

## Detailed findings

### [E-001] Euler characteristic change under cloning needs boundary/compactness hypotheses
- Location: sec-curvature-topology (thm-curvature-topology and prop-curvature-change-2d)
- Severity: Moderate
- Type: Scope restriction (secondary: Invalid inference)
- Claim (paraphrase): Cloning changes face counts in a way constrained by Euler characteristic, yielding fixed curvature jumps.
- Why this is an error in the framework: Euler characteristic identities require compactness or explicit boundary corrections. A local Voronoi update can preserve chi, especially in infinite or periodic tessellations. The argument needs a precise topological operation and boundary condition.
- Impact on downstream results: The O(1) curvature change claim may fail outside the stated boundary regime.
- Fix guidance (step-by-step):
  1. Specify the topology (compact without boundary or periodic) and the exact Voronoi update rule.
  2. Compute Delta V, Delta E, Delta F for that update and state Delta chi explicitly.
  3. If boundaries are present, include the boundary term in Gauss-Bonnet.
- Required new assumptions/permits (if any): A permit characterizing Voronoi updates and their Euler characteristic changes under the CST update model.
- Framework-first proof sketch for the fix: Enumerate the local combinatorial changes induced by inserting/removing a site and compute the Euler characteristic change in a finite complex with stated boundary conditions.
- Validation plan: Compare with explicit 2D Voronoi updates under periodic boundary conditions.

### [E-002] Plaquette area and bivector notation are ambiguous
- Location: def-scutoid-plaquette
- Severity: Minor
- Type: Notation conflict
- Claim (paraphrase): A_Pi approx ell * c * dt and curvature contraction uses T^c T^d.
- Why this is an error in the framework: The constant c is undefined in this chapter, and T^c T^d reads as a symmetric product rather than an antisymmetric bivector. This obscures the meaning of the holonomy contraction.
- Impact on downstream results: Minor, but it weakens clarity in the key dictionary formula.
- Fix guidance (step-by-step):
  1. Define c (e.g., algorithmic speed scale) or remove it from the area estimate.
  2. Replace T^c T^d with an explicit bivector T^{cd} = T^c S^d - T^d S^c.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Pure notation change; no proof required.
- Validation plan: Check the dimensional consistency of A_Pi and the contraction R^a_{bcd} V^b T^{cd}.

## Proposed edits (optional)
- Add explicit hypotheses for the Euler characteristic arguments (compactness or periodic boundaries).

## Open questions
