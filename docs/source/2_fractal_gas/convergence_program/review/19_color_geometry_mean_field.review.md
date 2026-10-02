# Mathematical Review: 19_color_geometry_mean_field.md

## Metadata

- Reviewed file: [19_color_geometry_mean_field.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/19_color_geometry_mean_field.md)
- Review date: 2026-10-02
- Reviewer: Codex, native spectral and independent population-proof reviewer
- Scope: Linear review of all definitions, estimates, proofs and retained conjectures in Chapter 19; verification against the actual canonical component-collision theorems in Chapters 08 and 09. Chapter 18 was read for its compatible finite-population smoothing, QSD and readout results.
- Framework anchors (definitions/axioms/permits):
  - `def-slc-parameter-register` and `def-slc-profiles`: complete configured parameter tuple, actual noise variances, velocity cap and component-collision bound.
  - `def-variant-viscous-euclidean` and `def-variant-recorded-color-geometry`: eligible-count force, evaluation at the two actual kinetic stages, fixed component-Haar collision and passive observation instrument.
  - `lem-mean-field-measurement-consistency`, `def-mean-field-rooted-collision` and `thm-mean-field-one-step-consistency` in Chapter 08: positive alive mass, sampled fitness and its moment requirements, finite rooted collision law, weak empirical post-collision convergence for deterministic or random arrays.
  - `lem-chaos-innovation-variance` in Chapter 09: resampling variance inequality for independent innovations, with dependence of the output retained.
  - `def-chaos-ordered-star-population-map`: ordered collisions require a persistent priority-decorated population law and are distinct from the permutation-equivariant canonical Haar kernel.
  - `thm-chaos-conditioned-propagation`: whole-path survival conditioning is distinct from individually renormalizing intermediate transitions.

## Executive summary

- Critical: 0.
- Major: 0.
- Moderate: 0 active; the tagged-input scope issue and cemetery-proof omission were repaired during this review.
- Minor: 0 active; the scaling-register clarification was repaired.
- Notes: 3 active scope records (A19-001–A19-003), alongside 4 verified proof blocks; the final block covers the existing row-normalized configuration.
- Primary themes: Preserve the distinction between empirical-law convergence and fixed-label path convergence; make extinction handling explicit; complete the physical-to-dimensionless parameter ledger. The kinetic force, moment, coupled-innovation variance and bounded-cap transport estimates passed the mathematical checks described below.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | `cor-cg-mf-full-update`, descriptor paragraph, source lines 484–492 at initial review | Moderate, resolved | Scope restriction; proof gap | Empirical convergence alone does not establish convergence of arbitrary fixed tagged input variables or their kinetic paths; the current chapter now explicitly restricts the claim. |
| E-002 | `def-cg-mf-parameter-register`, nondimensionalization paragraph | Minor, resolved | Dimensional mismatch / ambiguity | The expanded register now gives thermostat, primitive position-noise/jitter, metric, feature, width, diversity and color conversions. |
| E-003 | `cor-cg-mf-full-update`, absorbing extension and induction | Moderate, resolved | Definition mismatch; proof gap | The empirical law after extinction and the finite-horizon probability of using that extension were initially unspecified; the explicit correction now proves their irrelevance. |
| A19-001 | Population map and native RNG | Note | Scope restriction | The limit concerns the canonical exact-noise family, not a limit theorem for addressed finite-bit pseudorandom execution. |
| A19-002 | `cor-cg-mf-full-update` | Note | Conditional input scope | Initial-law measurement/convergence premises are retained; their regeneration at every later finite horizon is derived from this unchanged reference. |
| A19-003 | Whole-horizon conditioning and recorded observations | Note | Scope restriction | A vanishing conditioning cost does not supply an otherwise unproved joint temporal tagged-history or growing-graph limit. |

## Detailed findings

### [E-001] The tagged-descriptor assertion exceeds the empirical theorem

- Location: `cor-cg-mf-full-update`, descriptor paragraph following the explicitly exchangeable fixed-row statement; proof's final continuous-mapping sentence.
- Severity: Moderate, resolved by the author's subsequent restriction.
- Type: Scope restriction; secondary: proof gap / omission.
- Claim (paraphrase): Every bounded continuous descriptor of a fixed finite collection of tagged variables inside one kinetic update has a limiting law under the empirical convergence hypotheses of the corollary.
- Why this is an error in the framework: The collision-consistency theorem controls a uniformly sampled root and the empirical distribution. It does not control a distinguished label in a nonexchangeable input array. One exceptional tagged row can vary with population size while the empirical law converges to the same deterministic limit. Rejection of its cloning proposal can retain that exceptional row. The subsequent kinetic noise does not generally erase all information about its input. The preceding fixed-row product-law conclusion correctly requires exchangeability, but the descriptor paragraph does not explicitly retain that restriction. Moreover, final-time marginal chaos alone does not supply the joint law of a tagged row's intermediate B1/O/B2 variables and marks.
- Impact on downstream results: The empirical population limit remains valid. Applying the paragraph to recorded tagged paths, stage-specific color contractions or tagged source-response descriptors would exceed the proved statement.
- Fix guidance (step-by-step):
  1. Remove the blanket tagged-history assertion, as authorized by the parent task, and retain the final-time fixed-row consequence under the already stated exchangeability condition.
  2. Retain the statement that a graph/color readout needs continuity at the limiting law and that an increasing-row graph requires a separate estimate.
  3. Leave tagged multi-stage history convergence as a separately identified proof obligation unless its existing hypotheses and coupling are verified explicitly.
- Required new assumptions/permits: None for the recommended deletion/restriction. A stronger fixed-label result would require initial tagged-law convergence or exchangeability plus a joint tagged-stage proof; this review does not add either assumption to the model.
- Framework-first proof sketch for a future stronger result: For the fixed canonical Haar kernel, combine the rooted finite-component convergence with conditional fresh Gaussian marks. Couple the two force evaluations using the empirical force-stability lemma. Keep the same tagged Gaussian innovations through all stages and show the resulting whole tagged path converges. For ordered donor-star collisions, retain priorities in both the input law and limiting recursion.
- Validation plan: Check that no remaining theorem or roadmap statement claims a tagged history, shrinking-bandwidth graph or curvature limit from one-time empirical convergence alone.
- Resolution checked: The current chapter explicitly confines fixed-row chaos to completed states, identifies the exceptional-row obstruction, retains ordered priorities, and leaves intermediate tagged paths and increasing-row graphs as separate estimates. Its proof no longer invokes unproved tagged histories.

### [E-002] Complete the primitive unit conversion

- Location: `def-cg-mf-parameter-register`, nondimensionalization paragraph.
- Severity: Minor, resolved by the expanded primitive parameter conversion.
- Type: Dimensional mismatch / ambiguity.
- Claim (paraphrase): All parameter budgets use the stated dimensionless coordinates; the displayed list rescales the integrated OU amplitude, position amplitude, timestep, force, viscosity and bandwidth.
- Why this needs clarification: The moment and variance formulas are dimensionally coherent when their arguments are already dimensionless. However, `c_h` and `q` are subsequently computed from primitive parameters `gamma` and `b_O`, and `s` from `sigma_x`. Their physical-to-dimensionless transformations are not listed. A reader can therefore combine the dimensionless step with physical primitive coefficients despite the intended declaration.
- Impact on downstream results: This does not invalidate the algebraic estimates evaluated with consistently dimensionless inputs. It affects parameterized reproducibility and physical-time calibration.
- Fix guidance (step-by-step):
  1. State explicitly that every primitive parameter in the tuple is expressed in the selected units before the budgets are evaluated.
  2. Include the thermostat/noise transformations: `gamma = t_* gamma_phys`, `b_O = t_*^(3/2) b_O_phys / ell_*`, `sigma_x = sqrt(t_*) sigma_x_phys / ell_*`, and `sigma_J = sigma_J_phys / ell_*`.
  3. If donor widths, feature radii and diversity regularizers are supplied in physical units, list their corresponding conversions or identify the complete parameter conversion map. The phase-space metric weight must transform consistently with the position/velocity feature scales.
- Required new assumptions/permits: None. These are changes of units for the existing parameters.
- Framework-first proof sketch for the fix: Substitute the conversions into the actual OU variance and final position variance. They give `q = t_* q_phys / ell_*` and `s = s_phys / ell_*`, exactly the integrated-amplitude transformations already stated. With position and velocity feature scalings, the donor metric and widths retain the same Gaussian exponent.
- Validation plan: Evaluate one physical parameter tuple and its converted tuple; verify that rescaling the actual finite update commutes with the nondimensional update, and that all scalar budgets have dimensionless arguments.
- Resolution checked: The author added thermostat, primitive final-position noise and clone-jitter, feature-radius, phase-space metric weight, donor-width, diversity-standardizer, reward-unit and color-phase conversions. Substitution into the integrated noise formulas gives exactly the stated dimensionless amplitudes.

### [E-003] Cemetery observations and survival conditioning, repaired

- Location: `cor-cg-mf-full-update`, new absorbing-extension declaration and proof, source lines 460–477 and 508–535 at review time.
- Severity: Moderate, resolved.
- Type: Definition mismatch; secondary: proof gap / omission.
- Original claim (paraphrase): Iterate the population map finitely many times because its limiting alive mass stays positive.
- Original gap: A finite swarm can be extinct with positive probability even when the limiting law has positive alive mass. Its next donor denominator then vanishes and the process stops. A mass-one empirical law after that stopping event needs an explicit observation convention; positive limiting mass also needs to be converted into vanishing finite-horizon extinction probability.
- Implemented correction: The chapter now uses the actual empirical law on survival and a fixed cemetery observation afterward, without a restart. The proof simultaneously inducts on empirical convergence and survival probability. Conditional on previous survival, output alive-fraction convergence to a positive number forces the next extinction probability to zero. A finite sum gives vanishing extinction probability through a fixed horizon. Splitting bounded path tests over survival and extinction proves the whole-horizon conditioning cost.
- Required new assumptions/permits: None beyond the corollary's existing positive initial alive mass and the canonical nonempty terminal domain. No canonical zero-viscosity extinction rate is imported.
- Validation plan: Preserve the distinction between conditioning once on survival through the entire horizon and renormalizing each individual transition. Do not infer a growing-horizon rate from this qualitative finite-horizon induction.

## Verified proof blocks

1. **Force stability and moments.** The Gaussian gradient maximum is `exp(-1/2)/rho`. Splitting against a copy of the actual coupling gives the four target-velocity/position terms in the displayed estimate; Hölder bounds them by the target fourth moment times the transport error. Symmetry and bounded count-normalized row/column sums give the all-order force bound. The post-collision speed bound uses retained capped velocities from every component, including revived rows. Bounded donor positions followed by the prescribed Gaussian jitter give all-order post-collision position moments without assuming compact physical output support.

2. **Conditional variance after coupled B2.** Replacing one OU draw changes its own A2 position by `a Delta u` and changes every second-kick row through the viscous sum. The rowwise influence estimates sum to the stated coefficients `B_phi` and `C_phi`. The fourth-moment bound `A_4` retains uncapped OU velocities. Cauchy–Schwarz and the independent-coordinate resampling inequality yield the displayed `1/N` budget, with no false claim that the coupled outputs are independent. Integrating final position noise smooths the terminal mark through total variation of translated Gaussians.

3. **The W4 output upgrade.** At B1, bounded post-collision velocities allow the W4 force comparison. The OU stage retains independent fresh innovations conditioned on the whole array. At B2, fourth-moment transport controls the force difference in L2 while the same coupling retains W4 position control. Final position noise retains weak convergence and the uniform higher position moment. The bounded velocity cap upgrades its L2 transport cost to fourth-order cost. The terminal-mark mismatch vanishes because Gaussian position convolution gives zero mass to the stated boundary. The argument needs the same coupled positions and velocities through the stages; componentwise marginals by themselves would not suffice.

4. **Existing row normalization without a new degree hypothesis.** The finite force keeps the exact denominator `a_empirical(x_i) - 1/N`. The target fourth position moment supplies at least half its mass inside the explicit radius `S_eta`, so the Gaussian degree has the displayed positive lower bound on each analysis ball. The degree-error L2 norm and count-numerator L2 norm follow from the same actual W4 coupling. Tail exclusion, the derived local denominator bound and Chebyshev give the stated force-error probability estimate. The bounded collision velocities make B1 force errors uniformly bounded, upgrading convergence in probability to fourth transport cost. At B2 the proof needs only convergence in coupling probability: the completed velocity cap supplies bounded fourth cost, while B2 never changes positions. The fresh-noise lemma then applies to the capped joint array with the proved higher position moments. No uniform uncapped B2 row-moment bound, positive global degree, or maximum-over-N Gaussian estimate is imported.

## Scope restrictions and clarifications

- The fixed-horizon results concern the fixed canonical Haar-component gas with either existing Gaussian normalization. Count-normalized force-moment and conditional-concentration statements retain their narrower scope. The force is evaluated at the actual B1 and B2 populations in every case.
- Positive population-limit conclusions use the unchanged quadratic reference force, whose Lipschitz and growth profiles are evaluated explicitly. Bounds for other configured forces use their actual profiles and do not certify a divergent profile. The configured positive noise amplitudes and terminal box are checked directly; no landscape is modified or new regularity premise imposed to obtain a positive result.
- The ordered donor-star implementation has a separate priority-decorated population map. Its physical marginal is not automatically a closed permutation-equivariant recursion.
- All-order post-collision moments hold on the bounded terminal donor domain because every dead row is revived from an alive donor before jitter. They are not a stationary confinement theorem on all of physical space.
- Fixed-horizon convergence and finite-N QSD uniqueness do not establish population-uniform stationary chaos, full-gradient LSI, shrinking-bandwidth geometry or continuum Yang–Mills action identification. The final conjectures correctly retain those obligations.
- Chapter 18's terminal nonzero-force result should not be applied to the B1 force without verification. At zero restitution a connected collision component can have consensus velocities. The actual B2 force has a direct analytic-Gaussian nonzero proof with the existing positive OU noise; this correction was sent to that chapter's author.

## Proposed edits

- Implemented: removed the blanket tagged multi-stage descriptor assertion and its unsupported final proof sentence.
- Implemented: completed the primitive noise and feature unit conversions while retaining the configured parameter tuple.
- Keep the repaired cemetery extension and qualitative whole-horizon conditioning argument.

## Open questions

- A joint tagged-stage proof for the actual selected gas remains separate if later chapters require tagged history observables.
- The graph/geometry readout's increasing-population and shrinking-bandwidth consistency estimates remain the explicit geometric conjecture.
- A finite edge spectral gap for a declared recorded matrix does not supply a population-uniform gap or identify that operator with the native physical gauge evolution.

## Adversarial re-audit — 2026-10-02

### Coverage and framework

Every formal block and retained conjecture in the current 971-line Chapter 19 was read linearly. The audit then traced the exact selection/measurement/rooted-component proofs consumed from Chapter 08, the relevant canonical/ordered-star and innovation/survival-conditioning distinctions in Chapter 09, the actual force/cap/collision parameter register in Chapter 06a, and Chapter 18's compatible smoothing and marginal-moment scope. This is not an exhaustive review of the 18,901-line Chapter 06a, the 4,313-line Chapter 09, all continuous-time results in Chapter 15, or the full Volume II book. No theorem is marked established merely because a supporting file has a plausible title.

Native transition evidence was checked in `kinetic.rs`, `cloning.rs`, `fitness.rs`, `donor.rs`, `noise.rs`, `random.rs`, `engine.rs`, both Euclidean variant constructors, and the reference initializer in `benchmarks/src/lib.rs`. These checks establish which analytical kernel is represented and which numerical comparison remains outside the theorem; they do not replace mathematical hypotheses by simulation output. Only this review artifact was edited during this pass.

### Exact finite force and local row-ratio audit

- At each force stage all rows have been revived and remain eligible until the final boundary classification. The count force is the stage empirical integral with a zero self numerator. The row force removes exactly the self degree 1/N, retaining `a_LN(x_i)−1/N`. It is not the empirical ratio with its self degree included.
- For N=1 the configured force is zero. For every finite real-coordinate array with N≥2, the nonself degree is strictly positive. There is no denominator floor or modified row force in the proof. Numerically zero raw weight sums are handled by a separate native underflow branch, not by this local lemma.
- The local lower target degree is derived, not assumed: its fourth position moment puts more than half the target mass inside `S_eta=(1+2M_x^4)^(1/4)`, and the positive Gaussian kernel gives `b_R` on each chosen target ball. The 1/N self correction is included in `e_N`. The count-numerator L2 bound uses the same actual W4 coupling.
- The probability estimate excludes large **target** position/velocity and degree error, then bounds the ratio on that remaining set. The order of limits is N→∞ at fixed R,H, followed by R,H→∞. There is no maximum-over-N Gaussian event, global spatial cutoff or global degree lower bound.
- At B1, bounded collision velocities make the row-force difference bounded by 4νR_c, so convergence in coupling probability upgrades to fourth transport cost. After OU/A2, target fourth moments justify the local row lemma. At B2, only force convergence in coupling probability is required; the completed cap then gives bounded fourth cost. No count-only uncapped row Lp contraction or B2 variance theorem is imported.

### Count, noise and transport checks

The Gaussian slope is exactly exp(−1/2)/ρ. Count-kernel row and column mass bounds give the all-order force estimate, while the unbounded-velocity transport comparison uses the target fourth velocity moment rather than a maximum speed at B2. The p>4 budgets remain uniform in N after compulsory revival from the bounded donor domain and prescribed clone jitter.

The fresh-noise lemma conditions on the entire entering array. Its output rows are independent at that stage because their newly added innovations are independent; this claim is not extended past coupled B2. Weak convergence plus the uniform higher position moment and bounded completed velocities yields the joint fourth-order transport upgrade. The terminal-mark step uses the limiting boundary-null position law, not a pointwise coupling assertion at the box boundary.

In the count variance proof, resampling one OU draw changes its own A2 position and the force on every B2 row. Splitting kernel and velocity changes against the original uncapped velocities gives the displayed B_phi and C_phi. The Gaussian second/fourth resampling moments, Jensen estimate for the average speed and Cauchy–Schwarz produce the stated coefficient. No independence of the post-B2 velocities is used. This is variance conditional on the complete cloning output; it does not control the separate variance generated by measurement, donor choice and component collision, and no row-normalized quantitative B2 concentration is concluded.

### Selection-law applicability at every fixed horizon

The imported collision map is the unchanged component-Haar map `J`, not a generic cloning approximation. Chapter 08 freezes sampled fitness, uses independent recipient donor/gate draws, retains dead incoming vertices and their old velocities, and applies one shared rotation per accepted component. Native literal cloning reads frozen donor positions; native component rotations read the old component velocities. No ordered write is substituted for this map. The Python ordered-star map remains priority-decorated and cannot inherit canonical exchangeability automatically.

The regeneration check is specific to the quadratic bounded-terminal reference:

| Required input to the next selection theorem | Derivation from the preceding unchanged update |
|---|---|
| Positive limiting alive mass | The configured positive final position Gaussian assigns positive mass to the nonempty interior of D from every finite intermediate position. |
| Terminally consistent marks; zero boundary mass | The actual completed mark is 1_D(x+); Gaussian convolution assigns zero mass to ∂D. |
| Alive reward mean and second moment convergence | R=−|x|²/2 is continuous and bounded on the eligible terminal box; with the explicit mark coordinate, these empirical observables are bounded continuous on the completed-state subspace. |
| Velocity envelope | The unchanged final cap bounds every retained velocity, including dead rows, by V. |
| Required moment/UI input | The preceding kinetic position budgets have an order p>4. At the following clone stage all positions reset to bounded eligible donor positions, possibly plus Gaussian jitter, and all collision velocities are bounded by R_c. |
| Positive measurement/donor denominators | The unchanged squashed comparison features give a fixed positive Gaussian lower weight; positive limiting alive mass gives the source theorem's denominator lower bound. |
| Continuous standardized sampled fitness and acceptance | The existing positive regularizers/floors and clipped positive-part gate keep the exact source formulas continuous, including at ties. |

Conditioning on survival through n changes these input convergences and uniform moment expectations by at most the already vanishing exceptional probability / reciprocal survival probability. The latter tends to one for each fixed n. Thus the random-input version of the existing selection theorem applies inductively; no new positive-mass or confinement assumption is inserted at later horizons. The simultaneous induction then proves that each next extinction event has vanishing probability. The cemetery observation is not used as a revival donor.

### Detailed scope records

#### [A19-001] Exact-noise analytical family, not finite-bit limit

The analytical population map has independent exact Gaussian innovations and exact Haar component rotations. Native `RandomStream` instead uses a fixed seed and addressed finite-bit uniforms/Box–Muller/QR calculations. It is not a product of independent continuous random variables when interpreted as the bit-level execution law. This distinction is inherited from the canonical source and explicitly retained in the row theorem and Chapter 18. The force/stage order agrees, but no quantitative numerical-law comparison or asymptotic theorem for that addressed RNG is supplied. Native population/memory guards also bound runnable instances; the mathematical N→∞ result concerns the identified analytical family. No force or noise algorithm is changed by this audit.

#### [A19-002] Retained initial premises and reference applicability

The full-update corollary still requires an initial deterministic limiting law, positive alive mass, empirical convergence and the source measurement/moment conditions. These are existing data-dependent initial premises, not universally discharged facts for arbitrary initial arrays. They should not be omitted in a completion summary. For the canonical mathematical interpretation of the native preset initialization—independent uniform positions in [-1,1]^3 and zero velocities—they follow directly from the law of large numbers and bounded input/reward moments, with initial alive mass one. The actual fixed-seed finite-bit initializer does not thereby receive a mathematical i.i.d. law. Subsequent selection hypotheses are derived as in the table above. A general landscape with an infinite Lipschitz profile is not certified merely because the register allows extended-real profile values.

#### [A19-003] Survival comparison does not prove an absent tagged path limit

The exact bounded-test estimate for conditioning once on survival through T is valid because its exceptional probability tends to zero. It establishes that any already proved finite-horizon limit is unchanged by this conditioning. The corollary proves the empirical laws at each completed time (and their finite tuple, since all limits are deterministic), and completed-state finite-row chaos under exchangeability. It does **not** prove joint temporal tagged trajectories or the intermediate B1/O/B2/color history. Empirical convergence alone cannot control an exceptional deterministic label. Likewise, a deterministic current-state geometry decoration, an incoming one-step record and an accumulated history are different readout objects. A shrinking-bandwidth/increasing-row graph or its first variations needs the separate estimates already retained as conjectural.

### Adversarial disposition

The three historical errors remain resolved in the current source. No new active Critical, Major, Moderate or Minor flaw was found in its stated finite-horizon analytical conclusions. In particular the row normalization, regenerated selection inputs and whole-horizon extinction comparison do not conceal a new degree, cutoff, weak-selection or moment assumption. This is not a discharge of the numerical RNG comparison, population-uniform stationary chaos/full-law LSI, joint tagged-history convergence, continuum geometry or native physical gauge operator. Those remain separate obligations, with the same algorithm and no proposed new premise.
