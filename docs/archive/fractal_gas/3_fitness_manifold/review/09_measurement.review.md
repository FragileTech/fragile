# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/09_measurement.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/09_measurement.md
- Review date: 2026-02-02
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - {prf:ref}`def-pierced-neighbor-graph`
  - {prf:ref}`def-scutoid-field-transport`
  - {prf:ref}`def-fractal-set-gauge-connection`

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 0
- Notes: 1
- Primary themes: Coordinate-dependent probes; explicit scope of geometric distance to probe sets.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| N-001 | Section 1–3 (Probe + geodesic phase) | Note | Scope restriction | Hyperplane/line probes are coordinate-defined; geodesic distance is therefore chart-dependent unless declared as such. |

## Detailed findings

### [N-001] Coordinate-defined probes in curved metric
- Location: Section 1 (Hyperplane/Line Probe) and Section 3 (Geodesic Phase Proxies)
- Severity: Note
- Type: Scope restriction
- Claim (paraphrase): Hyperplane/line probes are defined in coordinates; geodesic distance to these sets is used as a phase.
- Why this is an issue in the framework: The emergent metric is generally curved; a Euclidean hyperplane/line in coordinates is not an intrinsic geodesic submanifold. The definition is valid but is chart-dependent. The document should make this dependence explicit to avoid implying coordinate invariance.
- Impact on downstream results: None if interpreted as a coordinate-level measurement operator. Potential confusion if readers interpret the probe as an intrinsic geometric object.
- Fix guidance (step-by-step):
  1. Add a short scope clause that probes are defined in a chosen coordinate chart of the latent space.
  2. Clarify that $\mathrm{dist}_g(x, H)$ is computed to the coordinate-defined set $H$.
  3. (Optional) Add a remark that an intrinsic probe would require a geodesic hypersurface definition.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Not needed (clarification only).
- Validation plan: Check that the stated probe/phase definitions remain unchanged; confirm no downstream definitions require coordinate-free invariance.

## Scope restrictions and clarifications
- Probes are defined in a coordinate chart; geodesic distances are computed to those coordinate-defined sets.
- Path-independent transport requires the explicit flatness condition in {prf:ref}`thm-path-independent-reference-transport`.

## Proposed edits (optional)
- Add a one-line scope sentence in Section 1 clarifying coordinate dependence of $H(n,b)$ and $L(p,u)$.

## Open questions
- Should $\chi(S)$ be specified by existing gauge data (e.g., fixed basis per node) or left as a free measurement choice?
- Do you want an intrinsic (geodesic) probe definition to complement the coordinate probe?
