# Mathematical Review: 04_gas_variants.md

## Metadata

- Reviewed file: [04_gas_variants.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/1_the_algorithm/04_gas_variants.md).
- Review date: 2026-10-02.
- Reviewer: Codex, integration review with independent finite-law and spectral reviewers.
- Scope: New recorded color/geometry definitions and proofs in Sections 4.3–4.5, updated Section 4.2 theorem ledger, and their implementation correspondence. This is not a new audit of every older variant chapter.
- Framework anchors:
  - `def-variant-viscous-euclidean`: exact eligible-count/row Gaussian forces, physical positions, and both B evaluation stages.
  - `def-slc-parameter-register`: existing donor, gate, component collision, jitter, noise, cap and terminal boundary conventions.
  - `thm-sm-su3-emergence`: the componentwise force-amplitude/velocity-phase color map and its non-linear spatial transformation law.
  - `def-tessellation-rust-representation` and the existing `GeometryPipelineConfig`, `MetricKind`, `WeightMode`, volume and curvature enums: the actual observation recipes.
  - `prop-fractal-set-analytic-transfer`: lossless encoding transports an already established law and its form, not inequalities for new independent coordinates.

## Executive summary

- Critical: 0 active.
- Major: 0 active.
- Moderate: 0 active; four implementation/scope mismatches were corrected.
- Minor: 0 active.
- Notes: Finite nontriviality and passive-law transfer have the exact scope below.
- Primary themes: Preserve the executed kernel; distinguish a passive full-slot observation from a geometry stage that filters eligible sites and writes fields back; pair color with the same force-evaluation velocity; do not confuse spatial covariance with a fundamental color representation.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Recorded geometry register and pipeline correspondence | Moderate, resolved | Algorithm mismatch | Numerical distance and normalization floors were initially described as free controls rather than fixed implementation constants. |
| E-002 | Earlier theorem-status prose and comparison | Moderate, resolved | Scope restriction / miswording | Statements that no viscous QSD or mean-field result exists became stale after the coupled proofs were supplied. |
| E-003 | Covariance spectral formula | Moderate, resolved | Algorithm mismatch | The native pseudo-inverse threshold is retained explicitly, rather than replaced by an ordinary inverse on every unbounded cloud. |
| E-004 | Observation execution register and payload scope | Moderate, resolved | Algorithm mismatch / scope restriction | The actual edge budget and failure outcome are retained; a native error has no invented graph or metric payload. |
| N-001 | B2 color validity and finite color variance | Note | Scope clarification | Nonzero force and a nonconstant finite invariant are not continuum gauge curvature or a local Yang–Mills identification. |

## Detailed findings

### [E-001] Exact existing pipeline controls

- Location: `def-variant-recorded-color-geometry`, `rem-variant-recorded-geometry-config`.
- Severity: Moderate, resolved.
- Type: Algorithm mismatch; secondary: parameter inconsistency.
- Original mismatch: The displayed curvature distance floor and row-weight denominator were registered as arbitrary positive settings. The implementation fixes the squared-distance floor and additive distance floor to `1e-8`, and the normalization floor to `1e-12`.
- Impact: A proof for arbitrary floors would not be the stated numerical observation instrument. Likewise, the covariance formula needs the existing absolute-ridge enum, not the default relative-to-trace setting.
- Implemented fix:
  1. Register the fixed implementation constants explicitly, with their recorded coordinate units.
  2. Identify `RidgeScale::Absolute`, `MetricPolicy::Clipped`, the configured lower and upper eigenvalue clamps, and the same determinant floor in both volume and curvature.
  3. Distinguish `observe_recorded_geometry` from `GeometryStageConfig::refresh`: the former calls the existing evaluation routine with the all-slot mask and cannot mutate the swarm or random stream.
- Required new assumptions: None. These are existing enum choices and constants for an observation instrument, not new controls or a changed force.
- Framework-first verification: On each finite graph, the ridge covariance, spectral clip, determinant density, row normalization and curvature difference are exactly the displayed operations. Spectral bounds yield the volume bounds. The row sums are at most one, and the conformal logarithm has oscillation at most one sixth of the determinant log ratio; multiplying by four yields the stated curvature bound.
- Validation: The native tests check all retained slots, reject loss of a spatial coordinate, and compare seeded step outputs and checkpoint bytes with and without passive observation.

### [E-003] Preserve the existing spectral threshold

- Location: The metric equation in `def-variant-recorded-color-geometry` and its configuration correspondence.
- Severity: Moderate, resolved.
- Type: Algorithm mismatch.
- Original mismatch: A positive ridge makes the covariance mathematically invertible, but the native implementation still replaces any eigenvalue at or below `3 * machine_epsilon * largest_absolute_eigenvalue` by zero before the metric clamp. An unbounded cloud can trigger this branch despite a positive ridge.
- Implemented fix: Use the exact existing thresholded pseudo-inverse in the displayed equation; declare the fixed `f64`/`f32` epsilons and repair flags. The ordinary inverse is identified only on the branch where every eigenvalue passes the threshold, without assuming that branch globally.
- Required new assumptions: None. No tolerance or dynamical parameter is changed.
- Framework-first verification: This is precisely `inverse_covariance` followed by `pinv_clamped` in the existing metric implementation. Spectral thresholds are Borel; the subsequent clamps give the same proved metric/volume/curvature bounds. Orthogonal conjugation preserves eigenvalues and the threshold, so the stated exact-arithmetic spatial covariance remains valid.
- Validation: Existing geometry tests exercise repaired/indefinite spectra. Finite-precision evaluation error remains separate from the exact operation with the recorded threshold.

### [E-002] Correct the proved-result ledger without transferring a different kernel

- Location: Section 4.2 and the convergence-status comparison row.
- Severity: Moderate, resolved.
- Type: Scope restriction; secondary: miswording.
- Original mismatch: A blanket absence of all viscous QSD and mean-field results contradicted Chapters 18 and 19 after their addition.
- Implemented fix: Name the actual proved subset: quadratic/capped/terminal-box finite-population QSD and entropy, population-independent marginal QSD tails, and fixed-horizon population consistency for both existing normalizations. The row proof derives local degree positivity from the actual target moments and preserves the finite self-exclusion. Retain the missing uniform stationary estimates, quantitative row-normalized B2 concentration, and metric/Boris/adaptive-noise cases as unresolved.
- Required new assumptions: None. The reference parameter tuple and coercivity margins are evaluated in Chapter 18. The original selection-consistency inputs remain visible in Chapter 19.
- Validation: All referenced proof labels resolve within the published two-volume TOC. No theorem about the passive recorded variant is presented as a theorem about the Einstein–Hilbert or Geometric Gas.

### [E-004] Retain geometry budgets and failure outcomes

- Location: The execution parameter register, pipeline correspondence and `prop-variant-passive-geometry-bounds`.
- Severity: Moderate, resolved.
- Type: Algorithm mismatch; secondary: scope restriction.
- Original mismatch: The geometry payload bounds were phrased as if every finite configuration returned a payload, although the actual `max_edges` check or configured failure policy can return an error. The edge budget was not named in the parameter tuple.
- Implemented fix: Retain the complete `GeometryPipelineConfig`, the supplied `max_edges` and optional previous-cell-volume data. Treat its actual success/error result as the observation. State metric/volume/curvature inequalities only for successful evaluations, with no invented fallback, increased budget or missing-data imputation. Keep `FailurePolicy::EmptyGraph` only on its native branch; in particular an edge-budget error remains an error.
- Required new assumptions: None. This removes an implicit success assumption and does not change either gas or observer execution.
- Validation: A native regression test calls the existing observer with an insufficient edge budget and verifies the error and unchanged input. The earlier seeded-step/checkpoint comparison continues to verify noninterference.

### [N-001] Native color validity and finite invariant nontriviality

- Location: `lem-variant-viscous-consensus`, `thm-variant-b2-color-nondegeneracy`, `thm-variant-finite-color-variance`.
- Verification:
  1. Pairing symmetric Gaussian terms gives the exact count and degree-weighted row dissipation identities. Vanishing on every row forces consensus; vanishing on one row need not.
  2. Conditional on the entire post-collision array, the OU velocities have full Gaussian density. The actual B2 positions depend on those same velocities. Substitution gives a nontrivial real-analytic force numerator; its zero set is null. Conditioning on the B2 positions as if they were independent would be incorrect and is not used.
  3. At the two explicit velocity arrays, both force directions are parallel or antiparallel to the chosen vector. The actual color map yields squared contractions `1` and `9/25` for either normalization. The force Lipschitz bounds keep them separated on the two displayed product balls.
  4. Gaussian density-times-volume bounds give positive masses. The independent-copy variance identity gives the stated `64/625` product lower bound. The final independent position noise supplies explicit positive survival masses without altering the terminal rule.
- Required new assumptions: None beyond the configured positive noise/viscosity and declared physical phase calibration of this instrument.
- Validation: Independent numerical tests evaluate both centers using the same B2 positions, velocities and exact Gaussian force formula. The theorem retains its dependence on the complete finite arrays and does not claim a population-uniform lower bound.

## Scope restrictions and clarifications

- A resting or collision-consensus B1 cloud can have zero color force. The exact/numerical validity masks remain part of the record.
- Full-slot Delaunay general position follows at B2 from the nondegenerate position density. Initial and B1 degeneracies retain the existing duplicate/rank policies.
- The canonical cube has signed coordinate-permutation symmetry, not full active orthogonal invariance.
- A linear common-frame color-basis change preserves the invariant contractions. A physical spatial rotation through the componentwise phase map is generally not that linear action.
- State decoration transports the existing state Dirichlet form. A record carrying intermediate noises does not acquire a path-space LSI from this identity.
- Exact-arithmetic readout bounds do not remove floating-point spectral repair, tolerance or precision data from a numerical run.

## Proposed edits

- All identified implementation and ledger corrections are incorporated.
- Preserve the finite/native versus continuum/physical distinction in downstream citations.

## Open questions

- Population-uniform stationary dependence and a suitable full-law continuous/discrete entropy form remain unresolved.
- Same-record geometric consistency and local non-Abelian connection/action identification are separate proofs.
- The explicit finite color variance bound can decrease with population and input arrays; a positive limiting gauge fluctuation needs additional proved estimates, not a new assumed lower bound.

---

## Adversarial algorithm/readout audit — 2026-10-02

### Metadata and independently verified coverage

- Reviewer: Codex algorithm/readout reviewer, independent second pass. The preceding review is preserved as history; its conclusions were not used as evidence for this pass.
- Request: Identify a detached comparison object, an unverified new premise, or a changed algorithm. This pass made **no edits to core code, definitions, theorem statements, proofs, or explanatory blocks**. The only write is this appended report.
- Framework: `def-gas-variant`, `def-variant-euclidean`, `def-variant-viscous-euclidean`, `def-slc-parameter-register`, `def-eg-baoab-canonical`, `def-fractal-set-record-coverage`, `thm-sm-su3-emergence`, `def-sm-color-alignment`, and the actual constructor/component/evaluation call paths below. No external theorem or new analytic premise was imported.
- Full linear document coverage: all of `04_gas_variants.md`, including the older Einstein–Hilbert, Geometric, Latent, Environment, comparison and measurement sections, not only the newly inserted formal results.
- Selected linear dependency coverage: the Fractal Set force/SDE and complete-record definitions, reconstruction proofs and analytic-transfer proposition (`01_fractal_set.md`, Sections 3.3–3.4, 4.4–5.1, 6, and the transfer proposition in 7); Standard Model color encoding, direct-law definition, all four alignment conventions, direct contractions and instantiated-record transition (`04_standard_model.md`, the relevant parts of Sections 2–3 and the recorded-transition section); the full parameter/noise/cap register in `06a_structural_landscape_convergence.md`; the parameter/reference/numerical-scope and final readout sections of Chapter 18; the population parameter register, both force maps, finite self-exclusion and final scope statements of Chapter 19. **This is not a linear review of every line of the 2,198-line Fractal Set or 4,406-line Standard Model chapter, nor an independent re-proof of all Chapter 18/19 estimates.** The separate finite-law reviewer audits their remaining proof/selection dependencies.
- Native source coverage: the complete Euclidean, viscous and Einstein–Hilbert constructors; donor enum/default and independent weighted-draw paths; fitness orientation/global regularization/positive map; clone decision, accepted-component identification and the component-rotation interface; dense recorded force and full native BAOAB/noise/cap/terminal path; native recording configuration, step-limit guard and extra force-record memory guard; the complete geometry-pipeline composition/configuration, relevant geometry-stage scheduling/filtering, metric/covariance/threshold/clamp, weight normalization, conformal curvature and tessellation failure/budget paths; the all-slot observer; native color-vector/alignment routines and stage-record assembly.
- Python source coverage: the actual donor-star collision, `EuclideanGas.step` selection and alive-mask entry path, kinetic force/graph-weight modifiers, plain/Boris B branches, OU coefficients and complete `KineticOperator.apply`; Fractal Set CST and sampled IG force construction; `compute_color_states_batch`; and the complete standalone `coupled_gas_diagnostics.py` with its focused tests. This verifies the relevant call paths, not every optional dashboard, geometric estimator, environment adapter or experimental setting.
- Validation: the focused Python diagnostic suite passed independently (`15 passed`). This review's Rust checks were source-only. The integration reviewer separately reports a successful 20-test Rust compatibility run (4 observer, 5 QFT execution and 11 tessellation geometry tests) using installed Rust 1.93 with `--offline --ignore-rust-version`; this is not verification with the project's required Rust 1.95 toolchain. That remaining toolchain scope is separate from mathematical correctness.

### Executive summary

- Critical: **0 identified in the newly added passive-record/color constructions in the verified coverage**. This is not a certificate for the entire Volume II chain.
- Major: **1 active older implementation-totality claim** (A-001).
- Moderate: **2 active implementation/scope issues** (A-002 and A-003).
- Minor: **3 scope/naming ambiguities** (A-004–A-006).
- The new estimates concern the existing **real-coordinate canonical viscous specialization**. They do not prove the generic Python graph-viscous/adaptive update, the Einstein–Hilbert consumed-geometry update, or the Geometric SDE. An explicit passive observation can inherit a proved state law; it cannot discharge those changed-kernel obligations.
- No added numerical cutoff, changed force, new geometry-feedback kernel, different velocity cap, altered donor law, or ordered-star substitution was found in the new canonical force/map definitions and observer implementation inspected here. Bounds may use analysis regions; these regions do not reject draws or change an update.

### Error log

| ID | Location | Severity | Type | Finding |
|---|---|---|---|---|
| A-001 | `prop-variant-eh-identities`, Item 6 and its proof; Section 5.2 explanation | Major | Algorithm mismatch / scope restriction | The universal finite-input geometry/survival assertion does not describe the native error/budget/precision behavior. |
| A-002 | `def-variant-latent`, closing implementation sentence | Moderate | Algorithm mismatch | The cited Python core is not established to implement the entire declared chart-metric latent tuple. |
| A-003 | Passive recording interpretation versus the finite-budget native archival API | Moderate | Scope restriction | Native archival limits can reject an update even though recording does not change a successful numerical trajectory. |
| A-004 | `prop-variant-passive-geometry-bounds` versus generic observer signature | Minor | Scope restriction / miswording | The bounded-payload result must be read as the displayed absolute-ridge/clamped readout, not arbitrary accepted pipeline enums/defaults. |
| A-005 | Euclidean component 11 and native reward-direction table | Minor | Notation conflict / definition mismatch | The mathematical oriented reward `R=-U` must not be passed as the native raw objective under `Minimize`. |
| A-006 | Native viscous reference constructor comment | Minor | Miswording | Its initial B1 force is exactly zero, so “B kicks record a non-zero force” cannot apply to every kick. |

### [A-001] The older Einstein–Hilbert totality assertion is too strong

- Exact document location: `04_gas_variants.md:1271`, `:1291`, and `:1299`; also downstream uses of `M=N` in the measurement discussion.
- Claim: Every map is finite-valued on every finite input, the geometry exists on every finite configuration, and consequently every slot remains alive at every step.
- Native evidence: `tessellation/degenerate.rs:28–48` defaults to `FailurePolicy::Error`. `tessellate` validates the domain, invokes the mesh routine, and then rejects a lifted-edge count exceeding `max_edges` (`:220–249`). `EmptyGraph` catches the mesh-error branch only; it does not catch the later edge-budget error or pipeline/field errors. `GeometryStageConfig::refresh` additionally enforces memory and rejects nonfinite output fields (`tessellation/stage.rs:178–218`). The pipeline propagates errors from site construction, tessellation, optional cells, metric, volume, weights and curvature. These paths are used by the actual Einstein–Hilbert constructor; duplicate/rank handling is not a guarantee that all paths succeed.
- Exact source-validated counterexample (not a claimed executed Rust test): take `N=11` finite `f32` slots, all three-component positions and velocities zero, `GasConfig::einstein_hilbert(0.33, 0.002)`, `GeometryReward::default()`, and `ZeroPotential`; retain the default memory budget and set the existing **execution** budget `max_batch_elements=99`.
  1. `GasBuilder::build` prepares the geometry fields before validating the gas (`engine.rs:356–362`). Its largest field is the ambient diffusion tensor, with `11*3*3=99` elements. Position/velocity fields have 33 elements; scalar geometry fields have 11. The positive batch limit therefore passes the field and one-companion-size checks in `GasConfig::validate`. All numerical gas parameters and finite fields are valid, and the default memory budget is ample for this small population.
  2. The engine's initial geometry refresh passes `max_edges=max_batch_elements/2=49` (`engine.rs:461–476`). All 11 projected sites coincide. The default `DuplicatePolicy::LiftCliques` preserves one site with a group of size 11. `open_mesh` returns a valid disconnected one-site mesh before attempting a nontrivial triangulation (`degenerate.rs:117–120`).
  3. `lifted_edge_count` adds the within-group clique count `11*10/2=55` (`degenerate.rs:203–213`). Since `55>49`, the later budget check returns `GasError::Capability` with “above the budget” (`:245–249`). It is outside the `EmptyGraph` catch, so selecting that existing policy would not rescue this budget error either.
  4. The initial `GasBuilder::build` refresh therefore fails before motion, despite finite valid input data. No altered force, geometric estimator or stochastic innovation is needed to obtain the counterexample.
  Unbounded finite-precision coordinates can also overflow intermediate covariance, force or field arithmetic. A successful finite exact-arithmetic formula is distinct from a total native execution API.
- Impact: The unrestricted native `M=N always` interpretation and blanket availability/survival claims are unsupported. This does **not** invalidate the new viscous QSD proofs, whose real-coordinate kernel and different terminal-box components are explicitly named. Nor does an execution error itself prove a row was physically killed: failure and population death must remain distinct outcomes.
- Minimal fix, without adding a premise or changing the gas:
  1. Separate the ideal finite-formula identity on an executed successful step from the native API's possible failure result.
  2. Remove the claim that duplicate/rank rules guarantee success for every finite input; retain configured budget/failure/precision outcomes.
  3. Restrict downstream `M=N` uses to successful exact finite updates with all initial slots alive, rather than deriving unconditional runtime immortality.
- Required new assumptions: **None for that correction**. Proving total native execution would require additional work and would not follow from the existing duplicate/rank construction.
- Validation: Check the actual EH refresh call at each reward boundary, an insufficient budget with finite duplicate sites, and propagation of a native error without silently selecting a new fallback.

### [A-002] The complete latent tuple is not the cited Python transition

- Exact document location: `04_gas_variants.md:1421`, immediately after the detailed latent component table.
- Claim: The Python core is the reference implementation of the tuple containing a chart metric `G`, momentum `p=Gv`, chart/metric-aware drift, reward one-form, metric cap, and the declared latent viscous/Boris dynamics.
- Source evidence: The inspected `KineticOperator.apply` directly updates Cartesian arrays by `x += (dt/2)*v`; `psi_v` is the Euclidean radial cap, not a `G`-norm cap; `_compute_viscous_force` reads a supplied neighbor graph and optional volume/degree/threshold modifiers; clone collision writes overlapping donor stars from frozen inputs, without the canonical shared Haar component rotation. The Python wrapper can use a reward one-form and optional adaptive diffusion, but those options do not establish the complete `G`-momentum/chart transition described in the latent definition.
- Impact: This sentence can make the reader apply a theorem to the wrong program configuration. The new Chapters 18/19 do not make that transfer: they name the native-canonical complete Gaussian force and component-collision model. The discrepancy is an existing cross-implementation assertion, not a new algorithm introduced by the present work.
- Minimal fix:
  1. Identify which portions of the Python implementation realize which latent specializations.
  2. Do not present the whole chart-metric tuple as implemented until a constructor/call-path correspondence is exhibited.
  3. Keep the existing native canonical and Python graph/adaptive profiles separate when citing the new results.
- Required new assumptions: **None**; use an accurate implementation scope. Do not retrofit `G`, a new cap or a new force merely to obtain correspondence.
- Validation: A literal component-by-component mapping, including drift, viscosity, OU amplitude, cap and boundary schedule, not the shared name “BAOAB.”

### [A-003] Mathematical passive recording is not an unlimited native archive

- Locations: `thm-variant-passive-record-transfer` and its implementation interpretation; `engine.rs:646–695`, `kinetic.rs:252–270`, and `tracking.rs:51–67`.
- Formal claim that **passes**: A kernel `M` with marginal `P` preserves projected paths. The standalone geometry helper is a read-only call; it consumes no random address, writes no population field, and retains its actual `Ok`/`Err` outcome. Its existing budget error therefore requires no invented graph.
- Runtime boundary: Native `start_recording` enables field and influence buffers. Before a later update, the engine checks `archive.steps.len() < max_steps` and otherwise returns an error. Dense force recording budgets an additional `N(N-1)*256` influence bytes. A finite recording or memory budget may reject an update that would otherwise execute. The archive default has a finite 256-step horizon. Recording failure is not a different viscous force, but it is a different *execution-success event*.
- Impact: A statement about the real-coordinate instrument remains correct. An unconditional claim that enabling the bounded native archival API preserves an arbitrary entire run, including whether steps execute, is not established by the seed/checkpoint noninterference test. No bit-level invariant/QSD transfer follows from that test.
- Minimal fix:
  1. Name the mathematical instrument in the exact marginal-law statement.
  2. Describe the native comparison as noninterference on common successful executions within the declared recording limits.
  3. Keep archive/observer failure outcomes and recording coverage in the report/header; do not reinterpret failed transactions as successfully recorded gas transitions.
- Required new assumptions: **None for the clarification**. An unbounded archive or enlarged memory budget is not supplied as a repair.
- Validation: Tests at a reached `max_steps` boundary and a tight influence-record memory budget would verify the execution boundary. Existing trajectory tests verify only the common-success branch.

### [A-004] Successful payload bounds belong to the displayed readout

- Locations: `04_gas_variants.md:455–458`, `:599`, `:627–656`; `observe_recorded_geometry` accepts an arbitrary validated `GeometryPipelineConfig`.
- Verified positive result: For the displayed **absolute-ridge**, lower/upper-clamped neighbor-covariance metric, determinant volume, normalized two weight families and conformal-Laplacian curvature, the formulas use the actual fixed `1e-8` distance floors, `1e-12` row floor and `3*epsilon_T` pseudo-inverse threshold. The spectral/volume/subunit-row/curvature bounds follow on successful evaluations without confining the physical sites.
- Ambiguity: The definition also retains the complete generic pipeline and allows other domains/enum arms; the helper itself accepts identity, observation-field, Voronoi and other valid estimators. The default metric uses **relative-to-trace ridge and no upper clamp**. Its outputs cannot simply be bounded by the absolute `g_-`, `g_+` constants in the displayed theorem. For example, a valid identity readout has eigenvalue 1 regardless of unrelated displayed clamp values. The generic helper test using `MetricKind::Identity` is a noninterference test, not a test of the covariance formula.
- Minimal fix: Say “for the displayed covariance/conformal readout” in the payload-inequality statement, while retaining measurability/native-error recording for the generic configured instrument. No additional algorithm parameter or hypothesis is needed.
- Validation: Check a default relative-scale pipeline and an identity pipeline are not cited as satisfying the displayed absolute-clamp bounds without their own actual parameter interpretation.

### [A-005] Distinguish raw objective from oriented reward

- Locations: Euclidean component 11 (`04_gas_variants.md:172`) and its correspondence table; `fitness.rs:10–17`, `:370–382`; `benchmarks/src/lib.rs:155–210` and the reference constructors.
- Evidence: Native `BenchmarkModel` emits `U(x)` for the quadratic objective; `ObjectiveDirection::Minimize` converts it to `-U(x)` before standardization. The mathematical fitness reward of the reference instance is therefore correctly `R=-U`. If the table's mathematical `R=-U` were passed directly as the native raw `RewardBatch` while keeping `Minimize`, the engine would orient it back to `+U` and selection would change.
- Impact: The reference proof is not using the wrong sign—the actual benchmark path agrees after orientation. The naming of the tuple's reward source as “raw reward” and the later `R=-U` row is potentially misleading for a custom caller.
- Minimal fix: State which symbol denotes the raw provider value and which denotes the oriented value used by the mathematics; retain both existing APIs and the same direction bit.
- Required new assumptions: **None**. This is a naming/contract correction, not a request to change the reward.
- Validation: Compare `RewardBatch.raw`, `FitnessBatch.oriented_reward` and the reference theoretical `R` on the same fixed coordinates.

### [A-006] Not every native reference kick has nonzero force

- Location: `algorithmic-gas/crates/benchmarks/src/lib.rs:433–436`, the `RunConfig::viscous_euclidean` comment.
- Claim: Its B kicks record a nonzero viscous force.
- Counterexample: The reference initial velocities are all zero. A frozen component collision preserves that consensus, so the very first B1 viscous force is exactly zero. The analytic almost-sure conclusion concerns the later B2 stage with nondegenerate ideal OU noise, not B1 or every finite-precision row.
- Impact: The new formal color theorems correctly retain the B1 validity mask, but this older code comment overstates the constructor guarantee.
- Minimal fix: Describe the recorded viscous force as a color input with stage/validity masks, and name B2 when citing its exact-Gaussian almost-sure validity theorem. No algorithm or parameter change is needed.
- Validation: At the existing initial state, inspect the first B1 force record and its norm; keep it distinct from the later B2 record.

### Positive checks and non-discharge boundaries

1. **Force and selection identity.** The native constructor differs from Euclidean only in `qft.viscosity`. It uses physical unsquashed pair distances, the configured bandwidth, and either eligible-count or nonself row mass. The force is frozen once per B stage. Canonical revival precedes kinetics and terminal-only classification leaves all slots eligible during both kicks. Weighted donor pool indices are not walker indices; the native frame assembler resolves them through source tables. Independent role streams, no history, one current donor, self exclusion with singleton convention, every-step gate, frozen donor positions and shared component rotations remain explicit rather than being replaced with a convenient selection model.
2. **Exact cap and OU.** The native scaled arithmetic implements `V*v/(V+|v|)`, not a hard projection and not a `tanh` cap. The OU factor is `b_O*sqrt((1-exp(-2*gamma*h))/(2*gamma))`, with the `gamma=0` branch; final position noise is `sigma_x*sqrt(h)`. Native code places position diffusion before the velocity-only cap. These two operations commute as terminal state maps in this specialization; their individual recording-stage labels must still retain the executed order.
3. **No exact-Gaussian claim for finite bits.** `random.rs` uses addressed finite-bit uniforms and Box–Muller evaluation. Native dense viscosity exponentiates raw Gaussian weights; a computed row mass of zero is skipped. Real Gaussian full support, analytic positivity, joint densities, null-force events and the smooth row ratio therefore describe the book's existing mathematical kernel, not a proved bit-level law. Chapter 18 explicitly declares this distinction. A comparison-error theorem quantifying floating-point effects was not proved here and is not inferred from the distinction.
4. **B2 validity is the correct stage, not the default Python history field.** Native records retain `force_input_velocity` with each actual kick force and versions/coverage. The new analytic argument substitutes the same OU variables into B2 positions and velocities rather than incorrectly conditioning on positions as if independent. Python `KineticOperator.apply` stores its reported force at B1; `compute_color_states_batch` normally pairs it with pre-clone velocity (reference-offset alignment) or, with the declared convention, post-clone velocity. It does not reconstruct a missing B2 sample. The preceding-kick, matched-kick and stage-native observables are distinct maps.
5. **Finite color variance does not supply the needed continuum gauge sector.** The new observable is the exact all-slot, stage-native `|c_1^dagger c_2|^2` in a chosen common three-component basis, with the exact zero-force mask. Its two velocity patterns use actual B2 positions and the actual force for either normalization. It is not automatically the primary pre-clone companion-contracted Standard Model observable, which has generation/clone/alive masks and potentially a positive threshold. Its positive finite bound is conditional-array and population dependent. It is not a local non-Abelian connection, nonzero curvature, gauge-sector mass gap, or continuum-uniform variance estimate. Those boundaries are stated, so the theorem must not be marked as discharging those stronger obligations.
6. **Passive geometry is not a geometry-feedback variant.** The new helper calls the existing pipeline on all retained slots; native `GeometryStageConfig::refresh` instead filters by `eligible(include_truncated)`, writes geometry fields and supplies graph forces. EH additionally drops the last coordinate in dimension at least three and consumes its graph. The recorded viscous construction retains three spatial coordinates and history time `nh`. These are different observation/dynamic/time choices. Neither a bounded estimator nor transported QSD proves continuum metric consistency or an Einstein–Hilbert first variation.
7. **Lossless transport is conditional on actual coverage, not extra physics.** The Fractal Set contract covers direct indexed anchors, evaluation payloads, missing marks and the declared raw-array decoder. Sampled Python IG force contributions do not sum to a full dense applied force; the full evaluated CST/history field is a separate payload. Encoding transports a law/form already proved for that covered record. It does not create omitted B2 evaluations, full diffusion off-diagonals, a path-space LSI from a state LSI, or a new independent field equation.
8. **Population limit scope.** The inspected Chapter 19 maps retain the same B1 and B2 empirical laws, exact row self subtraction `a_emp-1/N`, actual cap and final marks. Their positive result is finite-horizon and for the unchanged quadratic reference landscape with the existing normalization choice. A derived local degree floor uses target moments; it is not an algorithmic position cutoff or an assumed global degree comparison. The generic force-profile register can report an infinite bound rather than silently changing a force. Stationary attraction, uniform joint functional inequalities, row B2 quantitative concentration and shrinking-scale geometry consistency remain explicitly open. Selection consistency and the full proof are audited by the separate finite-law reviewer, not certified merely by this call-path check.

### Open checks and recommended integration

- Integrate A-001–A-006 as **scope/contract findings**, with their older/new provenance retained. None authorizes a core fix under this review-only request.
- Do not relabel the mathematical-kernel proof as a finite-precision execution theorem, the finite color statistic as a gauge-curvature theorem, or the passive spatial instrument as the Geometric/EH consumed-geometry kernel.
- The observer's four Rust tests have executed in the integration reviewer's 20-test compatibility run described above. Verification with the required Rust 1.95 toolchain remains open; neither that compatibility run nor the source-level noninterference argument certifies every accepted geometry recipe against the displayed covariance bounds (A-004).
- Broader completeness remains unverified: this audit did not inspect every old potential/reward landscape proof, every Geometric/latent premise, all Standard Model operator/spectral constructions, all graph/tessellation backends, or every optional Python execution setting. Its coverage is enumerated above so that an integration report cannot convert a targeted audit into a blanket claim that “everything is proved.”
