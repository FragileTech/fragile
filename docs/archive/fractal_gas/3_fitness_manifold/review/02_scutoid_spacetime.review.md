# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-voronoi-tessellation-time-t: geodesic Voronoi cells in the emergent metric g.
  - def-scutoid-slab-metric: slab metric G_k and Riemannianized slab metric G_bar_k.
  - def-scutoid-cell: scutoid cell construction from bottom/top faces and ruled lateral faces.

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 2
- Minor: 1
- Notes: 0
- Primary themes: geodesic Voronoi constructions need convex-normal restrictions; boundary correspondence requires equal interface measures; output-size lower bound does not apply to dynamic updates; complexity claims require explicit index/linear-size assumptions.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | def-voronoi-tessellation-time-t (properties 1-3) | Moderate | Scope restriction | Partition/convexity claims require convex normal neighborhoods or Safe Harbor assumptions.
| E-002 | def-boundary-correspondence-map (existence of measure-preserving bijection) | Moderate | Proof gap / omission | Measure-preserving map requires equal interface measures and additional regularity assumptions.
| E-003 | thm-omega-n-lower-bound (lower bound proof) | Major | Invalid inference | Output-size argument does not yield a dynamic-update lower bound.
| E-004 | sec-complexity-analysis, sec-jump-and-walk | Minor | Scope restriction | O(N) amortized cost assumes spatial index and linear-size Delaunay complex; not stated as required hypotheses.

## Detailed findings

### [E-001] Voronoi partition and convexity need explicit convex-normal hypotheses
- Location: def-voronoi-tessellation-time-t (properties 1-3)
- Severity: Moderate
- Type: Scope restriction
- Claim (paraphrase): The geodesic Voronoi cells form a partition and are closed; under Hadamard/CAT(0) geometry they are geodesically convex.
- Why this is an error in the framework: The emergent metric g is defined from H + epsilon_Sigma I, but global nonpositive curvature and unique geodesics are not established. Without a convex normal neighborhood or Safe Harbor restrictions, the geodesic distance can be non-unique and Voronoi cells can be disconnected.
- Impact on downstream results: The Delaunay nerve and scutoid construction rely on well-defined, contractible Voronoi cells. Without these hypotheses, the cell complex may not be a triangulation and the scutoid cell boundaries can be ill-posed.
- Fix guidance (step-by-step):
  1. State that all Voronoi constructions are restricted to convex normal neighborhoods or Safe Harbor windows where geodesics are unique.
  2. Add a short lemma: on each such window, geodesic Voronoi cells are closed and contractible.
  3. In global statements, treat the Delaunay structure as a general cell complex rather than a triangulation.
- Required new assumptions/permits (if any): A permit that Safe Harbor windows admit unique geodesics and contractible Voronoi cells.
- Framework-first proof sketch for the fix: In a convex normal neighborhood, distance to each site is smooth and strictly convex along geodesics, yielding connected Voronoi cells and contractible intersections.
- Validation plan: Confirm that later references to Delaunay triangulation explicitly mention the convex-normal restriction.

### [E-002] Boundary correspondence needs equal measures and explicit regularity
- Location: def-boundary-correspondence-map (existence of measure-preserving correspondence)
- Severity: Moderate
- Type: Proof gap / omission
- Claim (paraphrase): For shared neighbors, there exists a measure-preserving bijection between interfaces, yielding ruled lateral faces.
- Why this is an error in the framework: A measure-preserving bijection exists only when the two interfaces have equal total measure and are standard Borel spaces. The definition does not state equality of Hausdorff measures between top and bottom interfaces, nor the regularity required for the induced measures to be comparable.
- Impact on downstream results: The ruled lateral faces may be ill-defined for shared neighbors if interface measures differ; the scutoid boundary construction is not fully specified.
- Fix guidance (step-by-step):
  1. Add an explicit hypothesis: the shared interfaces have equal (d-1)-dimensional measure.
  2. If equality does not hold, replace the bijection with an optimal transport plan or a measure-preserving map between normalized measures.
  3. State the required rectifiability and absolute continuity assumptions under Safe Harbor.
- Required new assumptions/permits (if any): A permit for equal-measure interfaces under the update rule, or a framework lemma allowing transport plans instead of bijections.
- Framework-first proof sketch for the fix: Use normalized measures on interfaces and construct a transport map (or plan) in a convex normal neighborhood; define the ruled face via geodesic interpolation of the transport plan.
- Validation plan: Check a sample cloning event where interface measures change to confirm the revised definition still produces a well-defined lateral face.

### [E-003] Output-size lower bound does not apply to dynamic update
- Location: thm-omega-n-lower-bound (lower bound proof)
- Severity: Major
- Type: Invalid inference (secondary: Conceptual)
- Claim (paraphrase): Any algorithm updating a Voronoi/Delaunay complex after arbitrary movements must take Omega(|DT|) time; therefore the online algorithm is optimal.
- Why this is an error in the framework: The proof argues from output size, which only applies when the full complex is explicitly output each step. Dynamic maintenance can update internal structures without touching every simplex when the combinatorial structure is unchanged. The rotation example forces O(N) coordinate updates, not Omega(|DT|) when |DT| is superlinear.
- Impact on downstream results: The optimality claim is overstated; the lower bound does not match the update model used by Algorithm alg-online-triangulation-update.
- Fix guidance (step-by-step):
  1. Restate the theorem as an output-size lower bound for explicit reporting/serialization.
  2. If dynamic optimality is desired, prove a lower bound tied to the number of affected simplices or vertex updates.
  3. Update the TLDR and Key Takeaways to match the refined bound.
- Required new assumptions/permits (if any): A framework lemma defining the dynamic-update cost model and its lower bounds.
- Framework-first proof sketch for the fix: Show that any algorithm that explicitly reports all simplices must spend Omega(|DT|); for dynamic maintenance, a lower bound Omega(k) follows from the number of affected simplices in the conflict region.
- Validation plan: Check that the revised bound aligns with the complexity expression in sec-complexity-analysis.

### [E-004] Amortized O(N) cost needs explicit index and linear-size assumptions
- Location: sec-complexity-analysis and sec-jump-and-walk
- Severity: Minor
- Type: Scope restriction
- Claim (paraphrase): The amortized per-timestep cost is O(N) when p_clone << 1/log N.
- Why this is an error in the framework: The stated bound uses O(log N) point-location and O(1) conflict-region size in expectation, which require (a) a spatial index and (b) linear-size Delaunay complex under quasi-uniform sampling in bounded curvature. These are stated informally but not as explicit hypotheses.
- Impact on downstream results: Minor; the complexity statement is accurate only under these additional assumptions.
- Fix guidance (step-by-step):
  1. Add an explicit assumption list for the O(N) amortized claim (index present, |DT| = Theta(N), quasi-uniform sampling).
  2. In the jump-and-walk section, link the O(log N) cost to the presence of the index.
- Required new assumptions/permits (if any): None beyond those already mentioned; just elevate them to explicit hypotheses.
- Framework-first proof sketch for the fix: Under quasi-uniform sampling, expected conflict size is O(1); with an index, point location is O(log N), giving O(N + p_clone N log N).
- Validation plan: Cross-check the TLDR and theorem statements to ensure they carry the same hypothesis list.

## Scope restrictions and clarifications
- Scutoid cell construction assumes convex normal neighborhoods so that slab geodesics are unique.
- The Delaunay complex is a triangulation only under contractibility of Voronoi cells; otherwise treat it as a nerve complex.

## Proposed edits (optional)
- Add a brief "Safe Harbor" hypothesis box before def-voronoi-tessellation-time-t.
- Replace the measure-preserving bijection with a transport-plan statement when interface measures differ.
- Reframe the lower-bound theorem as an explicit-output bound.

## Open questions
- Should the scutoid construction allow transport plans (non-bijective) when interface measures differ?
- Do we want a dynamic-update lower bound in the CST model, or is an output-size bound sufficient?