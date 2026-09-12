# Interactive experiments for Part VI: fields and algorithmic QFT

:::{div} feynman-prose
Start with the walkers. Let them choose companions, copy, move, and receive noise. Then measure the field you want to understand from those executed events. A correlation, a curvature tensor, or a graph cut becomes useful when we can point to the states that produced it and ask which operation explains its change.

All 66 Part VI workbenches execute the Rust Euclidean Gas. Rust computes the scientific measurements and comparisons; the browser displays the resulting arrays. Most workbenches measure a recorded trajectory. VI-15, VI-30, and VI-31 execute paired full runs, while VI-19, VI-22, and VI-45 use independent continuations of a complete checkpoint. These are different sampling experiments, and the distinction matters when interpreting uncertainty.
:::

(sec-partvi-experiments-use)=
## Reading and running a workbench

:::{div} feynman-prose
Begin with the executed configuration and the measurement definition. Changing donor memory changes the dynamics. Changing a spectral fitting window changes how the same kind of trajectory is summarized. Changing the angle of a local basis changes coordinates. The controls should be read in that order: first what generated the data, then what was measured, then how the proposed description was tested.

The shared Rust registry defines the controls and their defaults. Part VI uses three-dimensional, CPU f64 runs with a quadratic objective unless its resolved request specifies otherwise. Common controls include population, timestep, donor memory, viscosity, and innovation law. The result exports the resolved request and actual run configurations, so there is no need to infer the law from the appearance of a plot. Geometry workbenches install the conditional fitness metric used by their executed run.

A session advances complete gas updates in bounded batches and records event identities. Temporal fits wait for enough recorded frames. The complete run and analysis evidence can be exported and reanalyzed through Rust. Extinction, empty support, and an inconclusive fit have distinct meanings; inspect their counts and reasons instead of interpreting an absent curve as a zero measurement.
:::

:::{prf:definition} Part VI measurement and evidence contract
:label: def-partvi-measurement-evidence

A workbench result specifies an executed Euclidean Gas configuration, its recorded states or continuation protocol, an observable and its normalization, and the numerical comparison applied to that observable. Its evidence consists of the resolved request, run configurations, retained archives, and continuation replay evidence when applicable.

An **executed measurement** is computed from these states or transitions. An **observable construction** applies a declared map to them, such as a covariance quotient, ray connection, or graph projection. An **identity comparison** checks a mathematical relation for that declared object. A **predictive hypothesis** uses one set of observations or an analytic conditional law to predict another specified measurement; its residual and sampling unit are part of the result.

All four are measurements or calculations on actual algorithm runs. An identity for a constructed observable does not by itself identify that observable with the chapter's proposed physical field law.
:::

(sec-partvi-experiments-algorithm-law)=
## The field theory follows the complete update

:::{div} feynman-prose
Imagine stopping the gas just before an update. Positions and velocities alone may not tell us what happens next: the retained donor records, eligibility, and provider state also matter. Independent continuations must start with this complete state. Otherwise we would be testing a different conditional law while giving it the same field name.

Cloning also explains why an average over walkers cannot be analyzed as though each walker fluctuated alone. Several recipients can inherit a common source. Their increments become correlated, and those correlations contribute to the fluctuation of the swarm average. The complete kernel keeps those effects in the calculation.
:::

:::{prf:definition} Conditional field increments
:label: def-partvi-conditional-field-increments

Let $R_n$ denote the complete algorithm state and $K$ its executed one-step kernel. For integrable real observables $F,G$, define

$$
D_F(R)=KF(R)-F(R),\qquad
Q_{FG}(R)=K(FG)(R)-KF(R)KG(R),
$$

whenever the required moments exist. A continuation experiment estimates these quantities, or their stated multi-step counterparts, conditional on one complete checkpoint. Its independent sampling unit is a continuation. A paired experiment uses matched random addresses within each baseline/transformed pair and independent seed streams across pairs; uncertainty is computed across pairs.

An empirical time average along one run is a different sampling law. Its frames, overlapping lag windows, and walkers are not counted as independent continuation replicas. Chronological holdout tests prediction on a later segment while retaining this dependence.
:::

The complete-state field hierarchy is developed in {prf:ref}`prop-ym-transient-algorithm-fields`. The per-experiment placements below identify the related chapter formulas; the measurement contract states which quantities are actually compared.

(sec-partvi-experiments-measured-outcomes)=
## Correlations, stress, and predictive comparisons

:::{div} feynman-prose
A correlation asks whether a measured pattern resembles its later self. A swinging pattern can become anticorrelated and then return. Its magnitude can therefore fall near zero and rebound even while its envelope decays. VI-36 retains the signed complex correlation, tests both complex exponential and damped-oscillation candidates, and inspects fitting-window stability and later prediction error. An inconclusive fit remains a useful result: it tells us that the chosen short spectral description has not resolved the measured behavior.

The frame average and source-pair measurements answer different questions. The frame average includes correlations between different walkers. The source-pair statistic tracks particular source relationships and uses a lag-dependent valid-pair count. Shared cloning histories make that distinction substantial. Keep both conventions visible when comparing curves.
:::

:::{prf:definition} Frame and source-pair correlation normalization
:label: def-partvi-correlation-normalization

For a fixed-capacity population of $N$ slots, let $o_i(R)$ be a local observable with the declared zero extension outside its support. The collective frame observable is

$$
F(R)=\frac1N\sum_{i=1}^{N}o_i(R).
$$

Its two-time product contains the full double sum $N^{-2}\sum_{i,j}\overline{o_i(R_n)}o_j(R_{n+\ell})$. A source-frozen pair readout instead averages its declared matched products over the valid pairs at that lag. Its source identities, denominator, and centering convention must be retained separately. These observables are not equated by changing one constant normalization.

The positive-transfer interpretation has the additional hypotheses in {prf:ref}`cor-effective-twistor-positive-transfer`. Passing a finite-sample spectral fitting diagnostic does not establish those hypotheses.
:::

:::{div} feynman-prose
For a proposed stress equation, the useful comparison is prediction on independent data. The research evaluation of a two-coefficient spatial curvature/stress relation gives local held-out RMSE 51,363, versus 31,600 for a constant predictor and 32,372 for shuffled training. The global improvement over a constant is only 0.56%. Those results do not support that fitted relation as a useful predictor of the sampled curvature.

VI-45 measures conditional metric evolution, and VI-51 measures the mechanical contributions that might explain it. Neither inserts the desired stress equation into its data. A successful description must connect the actual field drift and fluctuations to the measured mechanical state, including cloning, transport, noise, and the conditional fitness context. The research comparison and its sampling definitions are recorded in the [physics verification report](https://github.com/FragileTech/fragile/blob/main/algorithmic-gas/VERIFICATION.md).
:::

(sec-partvi-experiments-lattice)=
## Recorded fields and exterior constructions

:::{div} feynman-prose
Begin with finite objects whose entries can all be traced to a run: a clone probability, a covariance matrix, or an interaction face. Their exact identities make useful checks of the recording and analysis code.
:::

(sec-partvi-experiment-01)=
### VI-01 — Oriented transport on recorded triangles

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-fg-lqft-wilson-loops`.

**Rust measurement and comparison:** Normalize the recorded phase-space rays x+i v at Fractal Set events, form unit Hermitian-overlap phases, and multiply them around resolved interaction triangles. Compare reversed traversal with complex conjugation and local rephasing with the unchanged loop.

:::{div} feynman-prose
Change the local frame angle while watching the loop residual. This U(1) connection is constructed from actual event data; near-orthogonal overlaps have explicitly missing support. The experiment tests its orientation and basis identities.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-02)=
### VI-02 — Executed clone gates and their innovations

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-fermions`.

**Rust measurement and comparison:** Reconstruct each ordinary gate probability from the consumed source fitness, donor fitness, saturation, and clipping. Compare recorded probability and realized acceptance, and accumulate the centered gate innovation with predictable variance p(1-p).

:::{div} feynman-prose
Change donor memory or the timestep and follow the resulting decisions. Deterministic revivals have separate counts. The residual distinguishes an incorrect probability reconstruction from the random variation of correctly sampled gates.
:::

(sec-partvi-experiment-03)=
### VI-03 — Exterior algebra of measured frame observables

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-lqft-recorded-fermionic-reconstruction`.

**Rust measurement and comparison:** Center complete-frame phase-space means, diagonalize their empirical Gram matrix, and construct creation and annihilation matrices on its retained positive eigenspace. Compare their anticommutator with the retained Gram matrix and report discarded covariance.

:::{div} feynman-prose
Increase the retained mode count and inspect the covariance rank. A null direction supplies no occupation mode; the Fock dimension is 2 to the retained rank. The empirical law samples whole recorded frames, preserving within-swarm dependence.
:::

**Analysis or protocol controls:** `modes` (default `3`). The common engine controls remain active.

(sec-partvi-experiment-04)=
### VI-04 — Measured transitions and CAR channels

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-record-car-channel`.

**Rust measurement and comparison:** Estimate source, target, and cross covariances from a training prefix, whitening the two marginals separately. Construct the fermionic channel associated with the measured contraction and check its algebra independently of later cross-moment errors.

:::{div} feynman-prose
Change the lag and retained mode count. A channel can satisfy its exact matrix identities while describing later data poorly. Separate marginal whitening makes transient changes of the swarm distribution visible.
:::

**Analysis or protocol controls:** `modes` (default `2`), `lag` (default `1`). The common engine controls remain active.

(sec-partvi-experiment-05)=
### VI-05 — Regional covariance and locality

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-record-locality-defect`.

**Rust measurement and comparison:** Use paired pre-clone frames containing eligible walkers on both sides of the spatial split. Their regional velocity means determine a normalized cross-covariance, which is compared with an explicit two-mode mixed CAR anticommutator.

:::{div} feynman-prose
Inspect regional counts alongside covariance. Shared cloning histories can correlate distant regions, and nearly empty regions supply little information. The locality measurement belongs to these regional observables and their measured covariance.
:::

(sec-partvi-experiment-06)=
### VI-06 — Products and empirical replica wedges

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-product-obstruction`.

**Rust measurement and comparison:** Evaluate centered whole-frame observables on all pairs of draws from their finite empirical record law. Compare the normalized squared two-record determinant sum with the corresponding Gram determinant, alongside the ordinary same-frame product.

:::{div} feynman-prose
Vary the retained Gram modes and distinguish linear dependence from a small same-frame product. Independent draws from the empirical law are a mathematical construction over recorded states; they do not turn successive gas frames into independent dynamical runs.
:::

**Analysis or protocol controls:** `modes` (default `6`). The common engine controls remain active.

(sec-partvi-experiment-07)=
### VI-07 — Finite-step weak-generator samples

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-scalar-fields`.

**Rust measurement and comparison:** Measure linear and quadratic position increments on slots eligible at both endpoints. Compare the direct quadratic increment with the sum of 2 x times the displacement and the squared displacement, divided by the executed timestep.

:::{div} feynman-prose
Change the timestep and inspect the finite-step remainder that a first-derivative approximation would omit. Cloning displacements remain in the measurement. These realized increments become conditional generator estimates only after averaging the appropriate continuation law.
:::

(sec-partvi-experiments-standard)=
## Color observables and predictive channels

:::{div} feynman-prose
A field coordinate is a measured function of the algorithm state. Compare its coordinate identities first, then ask whether its transition statistics carry enough information to predict later frames.
:::

(sec-partvi-experiment-08)=
### VI-08 — Force-based colors from executed stages

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-direct-observables`.

**Rust measurement and comparison:** Build normalized color observables from recorded force and velocity at their designated kinetic stage. Export their validity masks, norms, and associated event identities.

:::{div} feynman-prose
Change viscosity or the innovation law and observe how the input vectors and valid colors change. A threshold defines the observable's support; excluded records must remain visible in the counts.
:::

(sec-partvi-experiment-09)=
### VI-09 — Invariant coordinates of measured colors

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-direct-orbit-isomorphism`.

**Rust measurement and comparison:** Compute inner products, determinant-based quantities, and invariant coordinates from recorded colors. Apply the configured unitary basis transformation and compare the resulting invariant residuals.

:::{div} feynman-prose
Change the basis angle while keeping the archived fields fixed. This separates coordinate changes from changes of the algorithm. The algebra characterizes the recorded color vectors without asserting a dynamical SU(3) transport law.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-10)=
### VI-10 — Companion doublets with immutable sources

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`def-sm-direct-companion-doublet`.

**Rust measurement and comparison:** Resolve each recorded companion reference to its immutable source and construct the associated color doublet and overlap measurements. Keep source validity and historical provenance with the result.

:::{div} feynman-prose
Increase donor memory and inspect how many observables use retained source records. A present slot and a historical donor at that slot can have different colors; the measured doublet must use the donor consumed by the update.
:::

(sec-partvi-experiment-11)=
### VI-11 — Phase-space ray holonomy

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`prop-sm-attribution-holonomy-defect`.

**Rust measurement and comparison:** Construct normalized phase-space rays at event vertices and multiply their overlap phases around actual interaction triangles. Report loop phases, reversal residuals, and local rephasing residuals.

:::{div} feynman-prose
Inspect which triangles have resolved nonzero overlaps. The measured object is the ray-induced U(1) connection. Its holonomy is an explicit coordinate construction on the executed interaction graph.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-12)=
### VI-12 — Compressed channels and memory

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-direct-channel-memory`.

**Rust measurement and comparison:** Partition the chosen frame descriptor using training data, estimate transition laws, and compare compressed evolution with retained-channel predictions and chronological holdout scores.

:::{div} feynman-prose
Change the number of bins, descriptor, and lag. Donor memory and omitted fields can return information to the observable. A multi-step defect measures the cost of discarding that information.
:::

**Analysis or protocol controls:** `bins` (default `4`), `lag` (default `2`), `descriptor` (default `"mean_speed_squared"`). The common engine controls remain active.

(sec-partvi-experiment-13)=
### VI-13 — Predictive partition refinement

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-predictive-partition-convergence`.

**Rust measurement and comparison:** Fit partitions and transition estimates using the training prefix, then evaluate prediction losses on later recorded frames. Report support and temporal prediction discrepancies as the partition changes.

:::{div} feynman-prose
Increase the bin count gradually. More bins retain detail but also leave fewer observations in each cell. The useful resolution is decided by prediction on held-out data, rather than by a visually finer histogram.
:::

**Analysis or protocol controls:** `bins` (default `4`), `lag` (default `2`), `descriptor` (default `"mean_speed_squared"`). The common engine controls remain active.

(sec-partvi-experiment-14)=
### VI-14 — Literal clone writes and population balances

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`prop-sm-implemented-collision-increments`.

**Rust measurement and comparison:** Reconstruct accepted copying and subsequent recorded transformations using their actual donors and stages. Compare measured increments with the complete write ledger and identify conditions under which paired contributions cancel.

:::{div} feynman-prose
Follow a recipient and donor through a gate. Literal copying need not give the donor an opposite impulse. Keep copying, restitution, and kinetic contributions separate when testing a proposed conservation law.
:::

(sec-partvi-experiment-15)=
### VI-15 — Complete-engine parity and color conjugation

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-coupling-matching`.

**Rust measurement and comparison:** Execute independently seeded baseline/transformed pairs with matched random addresses. Reflect positions, velocities, and symmetric innovations; compare positions, velocities, clone decisions, donor sets, and eligibility after complete updates.

:::{div} feynman-prose
The even default objective predicts parity covariance. Pair counts control across-pair uncertainty; shared addresses reduce within-pair variation. Force-based colors transform with the associated conjugation convention. A nonzero loop phase alone is not a test of complete-algorithm CP violation.
:::

**Analysis or protocol controls:** `replicas` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-16)=
### VI-16 — Generating functions of recorded observables

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-wilson-loops`.

**Rust measurement and comparison:** Form a finite empirical generating function from bounded terminal-position descriptors of recorded steps. Compare direct source derivatives with finite differences and check normalization at zero source; retain execution-action components separately.

:::{div} feynman-prose
The source weights the retained observations. It does not alter subsequent clone decisions or the transition law. Compare this statistical construction with the dynamical intervention in VI-19.
:::

(sec-partvi-experiments-yang-mills)=
## Path laws, response, and symmetry

:::{div} feynman-prose
A basis change, a change of statistical weights, and a physical intervention are different experiments. The following workbenches expose their inputs and compare the appropriate outputs of the complete gas.
:::

(sec-partvi-experiment-17)=
### VI-17 — Executed path-action accounting

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-first-principles`.

**Rust measurement and comparison:** Evaluate recorded companion outcomes, clone gates, revivals, and innovations under their executed conditional laws. Plot additive negative-log contributions, with the probability carrier and coverage stated for each component.

:::{div} feynman-prose
Change the innovation law and inspect the resulting action terms. Low-rank Gaussian and uniform innovations require their latent-coordinate carrier. Deterministic copying is a state map, so its output must not be assigned an invented full-dimensional Gaussian density.
:::

(sec-partvi-experiment-18)=
### VI-18 — Noise moments from actual factors

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-kinetic-metric-correspondence`.

**Rust measurement and comparison:** Read each eligible recorded diffusion factor and raw innovation. Compute second and fourth moments using the factor rank, source mean, and Gaussian or standardized-uniform fourth cumulant, and compare with executed samples.

:::{div} feynman-prose
Switch between Gaussian and uniform innovations. Their covariance can agree while fourth moments differ. Raw innovation moments precede the thermostat's physical damping and timestep scale, which VI-48 measures.
:::

(sec-partvi-experiment-19)=
### VI-19 — Gaussian source response through complete updates

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-native-source-response`.

**Rust measurement and comparison:** Fork a complete checkpoint into independent continuation groups. Shift one addressed Gaussian innovation in direct reruns and compare their final observable with likelihood-weighted baseline reruns, retaining all subsequent selection and kinetic operations.

:::{div} feynman-prose
Increase replicas and vary source strength or horizon. Each continuation is one sampling unit. Threshold crossings and changed clone decisions contribute to the physical response; the likelihood identity must include their downstream consequences.
:::

**Analysis or protocol controls:** `replicas` (default `32`), `horizon` (default `4`), `theta` (default `0.25`). The common engine controls remain active.

(sec-partvi-experiment-20)=
### VI-20 — Recorded transition information

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-metric-force-sources`.

**Rust measurement and comparison:** Bin whole-frame mean position using a training-only interval, count actual consecutive transitions, and compare the joint table with its transpose using an explicit pseudocount. Report held-out prediction and supported-row contraction diagnostics.

:::{div} feynman-prose
Change the number of bins and examine occupied rows. This KL measures a coarse empirical time asymmetry. Complete-path entropy production requires a separately specified forward and reverse path law.
:::

**Analysis or protocol controls:** `bins` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-21)=
### VI-21 — Empirical geometry-fiber disintegration

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-native-geometry-fiber-action`.

**Rust measurement and comparison:** Build a joint histogram of bounded mean-position and accepted-clone-fraction descriptors. Compute marginal and conditional distributions from the same counts and check normalization and action decomposition on supported cells.

:::{div} feynman-prose
Inspect an empty marginal bin: it has no conditional empirical law. The decomposition is exact for the recorded histogram, while the histogram's ability to characterize future behavior requires a separate prediction test.
:::

(sec-partvi-experiment-22)=
### VI-22 — Conditional stochastic balance

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-noether`.

**Rust measurement and comparison:** Compare independent continuation groups for the full observable increment. At each actual thermostat input, also compute analytic conditional momentum and energy moments from the configured damping, diffusion factor, innovation law, and source.

:::{div} feynman-prose
Accumulate centered thermostat increments and their predictable variance within each continuation, then compare across independent continuations. This retains the dependence of stages and walkers while testing the stochastic balance under the complete algorithm.
:::

**Analysis or protocol controls:** `replicas` (default `32`), `horizon` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-23)=
### VI-23 — Local phase Ward identity on recorded loops

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`lem-ym-discrete-ward`.

**Rust measurement and comparison:** Rephase the measured event rays independently and compare the product of transformed overlap links with the measured loop value. Report the local phase-covariance residual.

:::{div} feynman-prose
Change the angle control and watch individual links change while the loop remains invariant. This is the Ward identity of the explicitly constructed U(1) ray connection; the gas path action must be examined separately.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-24)=
### VI-24 — Measured loop action and phase variation

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-continuum`.

**Rust measurement and comparison:** Evaluate 1 minus the real part of the holonomy on actual recorded triangles. Perturb one edge phase and compare a centered finite difference with the imaginary part of the measured loop.

:::{div} feynman-prose
The loop data come from the executed gas, and the action is a stated observable of those data. Agreement checks its variation formula; matching that observable to a Yang-Mills transition density is a further hypothesis.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-25)=
### VI-25 — Empirical descriptor contraction

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-constants`.

**Rust measurement and comparison:** Estimate the mean-position descriptor channel from training transitions and compute its supported-row Dobrushin coefficient. Evaluate later transitions with a held-out Brier score.

:::{div} feynman-prose
Vary the bins and inspect row support. A contraction of this finite empirical channel describes this descriptor and sample; it is not automatically the spectral gap of the full gas with donor memory.
:::

**Analysis or protocol controls:** `bins` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-26)=
### VI-26 — Cloning and restitution mechanical ledger

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-complete-fluctuation-equations`.

**Rust measurement and comparison:** Measure all-slot unit-mass momentum and kinetic energy at the recorded cloning and restitution stages. Compare direct changes with their stage reconstruction.

:::{div} feynman-prose
Vary memory and inspect donors whose values are copied into recipients. A nonzero total increment can be an accurately accounted algorithmic transfer. The experiment tests the ledger before assigning a conservation interpretation.
:::

(sec-partvi-experiment-27)=
### VI-27 — Eligibility and conditional population moments

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-qsd-window-relaxation`.

**Rust measurement and comparison:** Plot actual eligible fraction, its net increments, and the mean squared radius conditioned on eligible walkers in each recorded frame.

:::{div} feynman-prose
Watch the two curves together when eligibility changes. The fraction of eligible slots is distinct from the survival probability of independent killed trajectories. Quasi-stationarity requires a stability comparison for the relevant conditional ensemble.
:::

(sec-partvi-experiment-28)=
### VI-28 — Empirical temporal reflection spectrum

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-native-labeled-color-reflection-sign`.

**Rust measurement and comparison:** Build a centered past/future bilinear matrix from consecutive same-epoch frame windows, symmetrize it explicitly, and display its eigenvalues and window count.

:::{div} feynman-prose
Change the number of modes and repeat seeds. A negative empirical eigenvalue identifies a quantity needing independent-run uncertainty; ordinary covariance positivity does not impose positivity on this different bilinear form.
:::

**Analysis or protocol controls:** `modes` (default `3`). The common engine controls remain active.

(sec-partvi-experiment-29)=
### VI-29 — Oriented boundaries of recorded interaction faces

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-native-scalar-face-evaluation`.

**Rust measurement and comparison:** Use actual Fractal Set triangles and their boundary-edge references to form oriented phase-space-ray products. Check orientation reversal and local phase covariance on resolved faces.

:::{div} feynman-prose
Inspect the triangle vertices, including historical sources. The loop is tied to the recorded face, and unresolved source or overlap counts describe exactly which faces contribute.
:::

**Analysis or protocol controls:** `angle` (default `0.7`). The common engine controls remain active.

(sec-partvi-experiment-30)=
### VI-30 — Complete-engine translation covariance and response

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-anchored-regulator-translation-test`.

**Rust measurement and comparison:** Run baseline/transformed pairs. In translated-system mode shift the population, reward optimum, force potential, and box boundaries together; in fixed-objective mode shift only initial positions.

:::{div} feynman-prose
Compare full-trajectory residuals, gates, donor choices, and eligibility. The first protocol tests coordinate covariance. The second measures a real displacement response under the unchanged objective. Independent pairs supply the reported uncertainty.
:::

**Analysis or protocol controls:** `amplitude` (default `0.5`), `replicas` (default `4`), `translation_mode` (default `"translated_system"`). The common engine controls remain active.

(sec-partvi-experiment-31)=
### VI-31 — Propagation of a local intervention

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-qft-axioms-verification`.

**Rust measurement and comparison:** Perturb one initial slot coordinate in one member of each independently seeded pair, then execute all donor choices, cloning, history, force, and noise operations. Measure mean response, RMS difference, and the fraction of affected slots.

:::{div} feynman-prose
Change the source slot, intervention size, and replica count. The footprint includes changes in the selection decisions themselves. It tests influence in the actual algorithm rather than influence through a frozen interaction network.
:::

**Analysis or protocol controls:** `source_slot` (default `0`), `amplitude` (default `0.1`), `replicas` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-32)=
### VI-32 — Predictive algorithmic field channels

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-algorithmic-qft-synthesis`.

**Rust measurement and comparison:** Fit channels for measured whole-frame descriptors, evaluate later predictions, and compare compressed multi-step evolution with the retained description. Keep training partitions and lag conventions explicit.

:::{div} feynman-prose
Vary the descriptor, bins, and lag. A field theory earns its predictive interpretation by carrying enough information through these tests. Exact channel algebra alone cannot settle the temporal closure question.
:::

**Analysis or protocol controls:** `bins` (default `4`), `lag` (default `2`), `descriptor` (default `"mean_speed_squared"`). The common engine controls remain active.

(sec-partvi-experiments-twistors)=
## Twistor observables and temporal spectra

:::{div} feynman-prose
Follow the source identities before fitting a decay rate. The triplets define the field, the aggregation defines the correlation, and the fitting test decides whether a compact temporal description works.
:::

(sec-partvi-experiment-33)=
### VI-33 — Velocity-derived null bispinors

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-twistor-spinor-conventions`.

**Rust measurement and comparison:** Embed three recorded velocity components as the explicitly null four-vector (norm of v, v), form its two-by-two Hermitian bispinor, and compare its determinant with zero.

:::{div} feynman-prose
Follow the velocity norm and determinant across actual walker records. Nullness is part of this readout definition. A particle mass or dispersion law requires a separate dynamical measurement, so this construction does not estimate a mass.
:::

(sec-partvi-experiment-34)=
### VI-34 — Twistor observables from recorded triplets

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-fractal-set-twistorization`.

**Rust measurement and comparison:** Resolve actual companion triplets, construct their twistor operators, and retain validity, source identities, and their geometric invariants with the measurements.

:::{div} feynman-prose
Increase donor memory and inspect the triplet support. The same algebra can be evaluated on different triplets, but the algorithm chooses which source records are present in each field value.
:::

(sec-partvi-experiment-35)=
### VI-35 — Source-frozen twistor lag tracking

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-twistor-vs-euclidean`.

**Rust measurement and comparison:** Freeze each source triplet, resolve its sink readout at the requested lag, and retain validity counts and event identities for the source-frozen correlation.

:::{div} feynman-prose
Watch how valid-pair counts change with lag. Recomputing an unrelated donor triplet at the sink answers a different question. This measurement uses the source-frozen convention explicitly.
:::

(sec-partvi-experiment-36)=
### VI-36 — Signed correlations and spectral fit diagnostics

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {prf:ref}`thm-effective-twistor-spectral-meaning`.

**Rust measurement and comparison:** Compute complex phase-space or twistor correlations with the selected frame or source-pair aggregation. Fit complex exponentials and damped-oscillation candidates on training lags; report trimmed-window stability, held-out complex errors, and a zero-predictor comparison.

:::{div} feynman-prose
Move both ends of the fit window. A dip and rebound in correlation magnitude can accompany a sign change without exponential growth. Inconclusive candidates retain their diagnostic curves but do not supply an identified growth rate or mass; chronological holdout alone supplies no independent-run confidence interval.
:::

**Analysis or protocol controls:** `readout` (default `"twistor"`), `aggregation` (default `"frame_mean"`), `channels` (default `1`), `max_lag` (default `6`). The common engine controls remain active.

(sec-partvi-experiments-gravity)=
## Conditional metric geometry and cloud evolution

:::{div} feynman-prose
The metric is reconstructed from the fitness calculation consumed by the update. Geometric probes use that measured field, while population-volume experiments follow actual copying, motion, and eligibility changes.
:::

(sec-partvi-experiment-37)=
### VI-37 — Metric transport along executed displacements

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-tessellation-to-curvature`.

**Rust measurement and comparison:** Reconstruct the selected conditional fitness metric from its executed donor context. Integrate its Levi-Civita transport along segments joining recorded positions and compare coarse and refined transport and metric-norm residuals.

:::{div} feynman-prose
Increase transport resolution while holding the recorded path and conditional context fixed. This measures spatial transport in a frozen conditional field. Moving the walker and changing the field are separate operations, combined explicitly in VI-45.
:::

**Analysis or protocol controls:** `walker` (default `0`), `record` (default `0`), `transport_steps` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-38)=
### VI-38 — Holonomy along a recorded-point loop

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-riemann-scutoid-dictionary`.

**Rust measurement and comparison:** Construct a closed path from the selected recorded displacement and a neighboring recorded point. Integrate conditional-metric transport at two resolutions and measure loop departure from the identity and norm preservation.

:::{div} feynman-prose
Refine the integration and inspect both errors. The loop endpoints come from the gas; the joining segments define the geometric probe. Holonomy is measured directly without imposing a constant-curvature loop-area formula.
:::

**Analysis or protocol controls:** `walker` (default `0`), `record` (default `0`), `transport_steps` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-39)=
### VI-39 — Curvature of the executed fitness metric

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-algorithmic-curvature-computation`.

**Rust measurement and comparison:** Reconstruct conditional fitness derivatives using the actual donor and normalization context, then compute the configured metric, connection, and curvature tensors. Independently estimate curvature from finite differences of the metric, starting at spacing `0.002` and allowing at most twelve halvings. Select the spacing from successive numerical scalar-curvature and Ricci-tensor estimates, without using the packed analytic curvature in the stopping rule. Display the numerical refinement curve, selected spacing, analytic comparison, and `converged`, `underresolved`, or `unsupported` status.

:::{div} feynman-prose
Vary the metric regularizer and inspect its spectrum and the refinement curve. As the probe spacing shrinks, the numerical scalar curvature and every Ricci component must stabilize for two consecutive refinements before the comparison is marked converged. The analytic answer does not choose where this process stops. An underresolved result means the allowed refinements did not settle the comparison. A spectral clipping threshold may make a stencil unsupported even though the metric itself remains defined; the experiment retains that support information. All these derivatives concern the stated conditional context.
:::

**Analysis or protocol controls:** `epsilon` (default `3`). The common engine controls remain active.

(sec-partvi-experiment-40)=
### VI-40 — Connection of a measured conditional metric

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-discrete-connection`.

**Rust measurement and comparison:** Compute centered metric derivatives at two finite-difference scales in the executed conditional context. Form Christoffel coefficients and compare metric-compatibility and derivative-refinement residuals.

:::{div} feynman-prose
Reduce the difference step gradually. A small algebraic compatibility residual alone does not establish an accurate derivative; the refinement discrepancy exposes cancellation and clipping-threshold crossings.
:::

**Analysis or protocol controls:** `walker` (default `0`), `record` (default `0`), `difference_step` (default `0.0001`). The common engine controls remain active.

(sec-partvi-experiment-41)=
### VI-41 — Finite-step cloud expansion balance

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-raychaudhuri`.

**Rust measurement and comparison:** Define volume by the square root of the determinant of regularized eligible-position covariance. Compare its endpoint log-volume increment with the sum of recorded stage increments.

:::{div} feynman-prose
Vary the covariance ridge and follow cloning and kinetic contributions separately. The telescope is a finite-step identity for this measured cloud volume; its agreement does not impose a Lorentzian Raychaudhuri equation.
:::

**Analysis or protocol controls:** `covariance_ridge` (default `1e-06`). The common engine controls remain active.

(sec-partvi-experiment-42)=
### VI-42 — Population volume and its stage budget

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-discrete-raychaudhuri`.

**Rust measurement and comparison:** Plot the regularized covariance length scale together with direct log-volume increments and their executed-stage reconstruction. Keep eligible counts in the budget.

:::{div} feynman-prose
A copied population can become narrow without the trajectories defining an invertible material flow. Inspect stage contributions before identifying population covariance volume with the volume of a transported material element.
:::

**Analysis or protocol controls:** `covariance_ridge` (default `1e-06`). The common engine controls remain active.

(sec-partvi-experiment-43)=
### VI-43 — Measured cloud focusing

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-focusing-theorem`.

**Rust measurement and comparison:** Measure theta as the finite-step log-volume increment divided by timestep, then display its finite difference plus theta squared divided by dimension.

:::{div} feynman-prose
Inspect this residual instead of imposing a Riccati equation on the plot. Cloning, boundaries, and finite-step effects remain in the measured expansion. The covariance ridge sets a finite floor when the cloud degenerates.
:::

**Analysis or protocol controls:** `covariance_ridge` (default `1e-06`). The common engine controls remain active.

(sec-partvi-experiment-44)=
### VI-44 — Topology of the recorded Fractal Set

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-curvature-topology`.

**Rust measurement and comparison:** Build the Fractal Set from executed events and edges, count connected components and graph cycle rank, and compare V minus E with components minus cycle rank. Report resolved edges and interaction triangles.

:::{div} feynman-prose
Inspect historical source coverage and event identity. This is an exact graph calculation on the recorded structure. A spacetime triangulation or curvature-integral relation requires additional geometric data beyond these graph counts.
:::

(sec-partvi-experiments-fields)=
## Metric fluctuations and mechanical balances

:::{div} feynman-prose
A material observer can see a changing field because the field changes, because cloning moves the probe, and because the probe then travels through the field. The stage ledgers let us measure those contributions together.
:::

(sec-partvi-experiment-45)=
### VI-45 — Conditional material metric drift and covariance

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-algorithmic-metric-evolution`.

**Rust measurement and comparison:** Fork a complete checkpoint into calibration and validation groups. Measure the six packed metric components at the material or fixed probe, and decompose increments into conditional-field change, clone-induced probe displacement, and subsequent motion.

:::{div} feynman-prose
Compare group means and the full covariance, including cross terms between contributions. Every continuation executes the complete law. The recorded support convention, unavailable outcomes, and extinction counts define the field whose conditional statistics are being tested.
:::

**Analysis or protocol controls:** `readout` (default `"material"`), `replicas` (default `32`), `horizon` (default `1`). The common engine controls remain active.

(sec-partvi-experiment-46)=
### VI-46 — Executed virial and dilation work

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-elastic-pressure`.

**Rust measurement and comparison:** For each recorded stage transition, compare the direct change in the eligible zero-extended x dot v sum with x dot delta-v plus delta-x dot the final velocity.

:::{div} feynman-prose
Inspect the ledger while changing timestep or viscosity. This discrete product rule retains finite increments and eligibility changes. Interpreting its terms as deformation pressure needs a specified deformation protocol and volume convention.
:::

(sec-partvi-experiment-47)=
### VI-47 — Density modes of the executed population

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-linearized-dynamics`.

**Rust measurement and comparison:** Compute the real part, imaginary part, and magnitude of the empirical Fourier mode from actual positions at the selected wavenumber. Export fixed-capacity finite-step increments with their eligibility convention.

:::{div} feynman-prose
Change wavenumber and timestep and inspect phase as well as magnitude. The displayed mode uses the eligible count, while the increment ledger uses fixed capacity and zero extension; the two normalizations answer different questions when eligibility changes.
:::

**Analysis or protocol controls:** `wave_number` (default `1`). The common engine controls remain active.

(sec-partvi-experiment-48)=
### VI-48 — Conditional BAOAB diffusion and heating

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-algorithmic-balance-laws`.

**Rust measurement and comparison:** Use the actual A1 input, O-stage damping, diffusion factor, and innovation law to predict conditional momentum and kinetic-energy increments. Compare with recorded O-stage realizations and their stage coverage.

:::{div} feynman-prose
Switch innovation law or timestep and inspect the predicted scale. This measurement includes physical integration factors, complementing the raw-noise moment comparison in VI-18.
:::

(sec-partvi-experiment-49)=
### VI-49 — Velocity-mode energy and Parseval balance

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-radiation-pressure`.

**Rust measurement and comparison:** Apply an orthonormal Fourier transform over the explicitly ordered eligible walker slots. Compare summed mode energies with direct unit-mass kinetic energy at each frame.

:::{div} feynman-prose
Inspect the slot order in the exported result. This basis gives a complete energy decomposition but is not a spatial wavenumber basis, so its mode index alone does not define a dispersion relation.
:::

(sec-partvi-experiment-50)=
### VI-50 — Measured cloud and kinetic scales

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-pressure-regimes`.

**Rust measurement and comparison:** Measure position covariance radius, RMS speed, centered velocity variance, and the corresponding crossing-time diagnostic from eligible records.

:::{div} feynman-prose
Compare their changes under viscosity and noise controls. These empirical scales characterize the actual run. An equation of state or pressure crossover must make an additional prediction for these measurements.
:::

(sec-partvi-experiment-51)=
### VI-51 — Complete mechanical balances and stress ingredients

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-stress-energy-tensor`.

**Rust measurement and comparison:** Resolve cloning, restitution, jitter, total-force work, thermostat, transport, boundary, and eligibility contributions using recorded stages. Compare exact balance ledgers with direct changes and retain the quantities needed for stress diagnostics.

:::{div} feynman-prose
Use the complete force, including viscosity, in kick work. Compare these mechanical measurements with VI-45 metric evolution before proposing a stress relation. A fitted spatial curvature relation must predict independent data as well as the complete balances permit.
:::

(sec-partvi-experiments-holography)=
## Executed graphs, entropy, and response

:::{div} feynman-prose
An interaction graph records the choices actually made. Its cuts and entropies have precise finite definitions. These measurements provide concrete quantities against which a proposed geometric or thermodynamic relation can be tested.
:::

(sec-partvi-experiment-52)=
### VI-52 — Cuts by executed interaction channel

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ig-entropy`.

**Rust measurement and comparison:** Count selected directed edges crossing nested slot partitions, separating distance, cloning, and their historical channels. Project retained event edges to slot labels with unit capacity per executed selection.

:::{div} feynman-prose
Move the partition and inspect each channel. Slot projection is part of the observable definition; its edge count is not a spatial area. Historical identity remains available in the underlying Fractal Set.
:::

**Analysis or protocol controls:** `partition_fraction` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-53)=
### VI-53 — Empirical transition time asymmetry

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-algorithmic-thermodynamic-response`.

**Rust measurement and comparison:** Build a training-prefix whole-frame descriptor table with explicit symmetric pseudocounts and compare it with its transpose through KL contributions.

:::{div} feynman-prose
Change the pseudocount and bin count to inspect sensitivity to sparse cells. This is an empirical descriptor statistic. The complete execution likelihood in VI-17 supplies different information and cannot be inferred from this coarse table alone.
:::

**Analysis or protocol controls:** `bins` (default `4`), `pseudocount` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-54)=
### VI-54 — Recorded interaction perimeter scan

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-gamma-convergence`.

**Rust measurement and comparison:** Scan nested slot regions and count directed executed interactions crossing each partition. Compare cut differences with the incident-edge first variation.

:::{div} feynman-prose
Change population and donor memory and retain both run configurations. The cut depends on actual selected edges and the chosen slot regions. A continuum spatial perimeter law is a hypothesis requiring a corresponding geometric comparison.
:::

**Analysis or protocol controls:** `partition_fraction` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-55)=
### VI-55 — Observed companion diversity and support

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-area-law`.

**Rust measurement and comparison:** Count selected target slots for each source slot, form empirical target frequencies, and compare their entropy with the logarithm of observed support.

:::{div} feynman-prose
Increase the recording budget or memory and inspect support growth. Entropy is bounded by log support for the empirical distribution. Dependent selection frequencies are not the exact conditional probabilities of the sampler.
:::

(sec-partvi-experiment-56)=
### VI-56 — First variation of an executed graph cut

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-first-law`.

**Rust measurement and comparison:** Add one slot to a nested partition and compare its measured directed-cut increment with the signed count of incident selected edges.

:::{div} feynman-prose
Inspect a slot with many historical interactions. Each entering or leaving edge contributes explicitly, making the discrete variation a check of the graph representation and its orientation conventions.
:::

**Analysis or protocol controls:** `partition_fraction` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-57)=
### VI-57 — Measured energy distribution and an exponential hypothesis

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-qsd-thermal`.

**Rust measurement and comparison:** Fit an exponential energy-bin hypothesis using the training mean kinetic energy. Compare it with the chronological held-out energy histogram, assigning the complete upper tail to the final bin.

:::{div} feynman-prose
Change bins, noise, and viscosity and inspect total variation. The exponential is a proposed distribution fitted to actual data; it is not automatically the configured gas's Gibbs or quasi-stationary law.
:::

**Analysis or protocol controls:** `bins` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-58)=
### VI-58 — Empirical source susceptibility

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-algorithmic-thermodynamic-response`.

**Rust measurement and comparison:** Exponentially tilt retained whole-frame mean-position observations, compute the tilted mean and variance, and compare the derivative of the mean with that variance by finite differences.

:::{div} feynman-prose
Change the source and inspect which observations gain weight. This exact finite-empirical-law identity changes statistical weights. Use VI-19 when the question concerns a source that changes the subsequent algorithm itself.
:::

**Analysis or protocol controls:** `source` (default `0.3`). The common engine controls remain active.

(sec-partvi-experiment-59)=
### VI-59 — Recorded ancestry depth

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ads-cft`.

**Rust measurement and comparison:** Follow actual ancestry and persistence edges and plot event counts by maximum retained ancestral depth. Report the deepest recorded chain.

:::{div} feynman-prose
Increase the run length and donor memory and inspect the resulting genealogy. Depth starts at the retained history boundary; it measures ancestry coverage without assigning a spacetime horizon or AdS radius.
:::

(sec-partvi-experiment-60)=
### VI-60 — Maximum flow and minimum cut of selected interactions

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ads-cft`.

**Rust measurement and comparison:** Project executed selected-edge counts to a directed capacity graph. Compute maximum flow between the selected source and sink slots, recover the minimum partition, and compare its crossing capacity with the flow.

:::{div} feynman-prose
Change source and sink and inspect the edges supplying capacity. Max-flow/min-cut is an exact finite graph identity here. Equating this capacity to quantum entropy requires an independently defined quantum state and entropy measurement.
:::

**Analysis or protocol controls:** `source_slot` (default `0`), `sink_slot` (default `4`). The common engine controls remain active.

(sec-partvi-experiments-cosmology)=
## Stationarity, scales, and dynamical closure

:::{div} feynman-prose
Measure the distribution and the cloud scale before assigning an equilibrium or expansion law. A useful reduced theory must predict how those measurements change when the algorithm configuration changes.
:::

(sec-partvi-experiment-61)=
### VI-61 — Recorded distribution stationarity

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-three-vacuum-energies`.

**Rust measurement and comparison:** Fix position histogram bins using the training prefix, then plot frame entropy, distance to the pooled training distribution, and successive-frame distance. Retain eligible counts.

:::{div} feynman-prose
Extend the trajectory and compare separate seeds. Slow drift or small frame differences are observations to quantify; the experiment does not insert a stationary distribution or substitute eligible-slot fraction for trajectory survival probability.
:::

**Analysis or protocol controls:** `bins` (default `4`). The common engine controls remain active.

(sec-partvi-experiment-62)=
### VI-62 — Measured fitness curvature scale

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-uv-regime`.

**Rust measurement and comparison:** Reconstruct actual conditional fitness curvature, report scalar curvature and its absolute scalar-derived radius, and compare Ricci with scalar curvature times the metric divided by dimension.

:::{div} feynman-prose
Inspect the Ricci isotropy residual alongside the radius. A scalar radius compresses one tensor contraction; it does not establish constant curvature, an AdS geometry, or a cosmological constant.
:::

**Analysis or protocol controls:** `walker` (default `0`), `record` (default `0`). The common engine controls remain active.

(sec-partvi-experiment-63)=
### VI-63 — Scale evolution of the executed cloud

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-raychaudhuri-expansion`.

**Rust measurement and comparison:** Measure the covariance volume scale and its finite-step log-volume expansion from recorded eligible positions, with the covariance ridge and stage budget exported.

:::{div} feynman-prose
Change viscosity or memory and compare the resulting histories. The scale is computed from the swarm at each time. Its evolution is measured rather than generated by an imposed cosmological expansion equation.
:::

**Analysis or protocol controls:** `covariance_ridge` (default `1e-06`). The common engine controls remain active.

(sec-partvi-experiment-64)=
### VI-64 — Information in executed region-to-region interactions

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-closure-theory`.

**Rust measurement and comparison:** Form the joint distribution of source and target region labels using actual selected-edge counts. Compute marginal and joint entropies, conditional target entropy, mutual information, and the information chain-rule residual.

:::{div} feynman-prose
Move the partition and inspect the four joint frequencies. This classical empirical edge statistic characterizes the selected interaction pattern; predictive information across time needs a time-indexed observable comparison.
:::

**Analysis or protocol controls:** `partition_fraction` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-65)=
### VI-65 — Held-out empirical field closure

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-closure-theory`.

**Rust measurement and comparison:** Fit a pseudocount-regularized transition matrix for whole-frame mean position on the training prefix. Compare one-step and iterated two-step predictions with a training-marginal baseline on the chronological suffix.

:::{div} feynman-prose
Change bins and pseudocount and compare the losses. A successful one-step fit need not retain the donor history or unresolved fields needed for two-step prediction. Empty held-out support remains an explicit coverage outcome.
:::

**Analysis or protocol controls:** `bins` (default `4`), `pseudocount` (default `0.5`). The common engine controls remain active.

(sec-partvi-experiment-66)=
### VI-66 — Dimensionless scales measured from the gas

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-cosmological-observations`.

**Rust measurement and comparison:** Measure an RMS covariance length and speed, form their crossing time, and report timestep times speed divided by length alongside eligible fraction. Export scale degeneracy when a required denominator vanishes.

:::{div} feynman-prose
Compare configurations in these measured units while retaining their raw quantities. These are empirical gas scales; no external particle-mass anchor or physical cosmological scale enters their construction.
:::

(sec-partvi-experiments-native)=
## Native execution, evidence, and stress tests

:::{div} feynman-prose
Use the native runner when you want to repeat a browser experiment with a longer recording budget or inspect every exported array. The scientific calculation is shared: the result is still tied to the executed gas configuration and its retained evidence. Start with a reproduced request, then change one input and compare measurements whose definitions remain fixed.

A stress test should explain what failed. An exact clone or stage identity can expose a tracking error. A derivative-refinement discrepancy can expose a poorly resolved clipped metric. A held-out prediction loss can expose a field description that has forgotten essential state. These failures call for different changes to the library, and their diagnostic outputs should remain visible.
:::

### Shared request and result

The Rust `LectureRequest` contains `id`, `seed`, `parameters`, and `steps`. The registry resolves default controls and rejects unknown controls or out-of-range values. `LectureSession` executes the request in bounded batches and exports a snapshot and `ExperimentEvidence`. The evidence contains resolved configurations and archives, plus continuation replay evidence for VI-19, VI-22, and VI-45. Rust reanalysis consumes that evidence rather than trusting imported plot coordinates.

A scientific result contains its title, measurement definition, metrics, plotted series, notes, and structured details. Source identities, normalization, sampling unit, training intervals, coverage, and fit status belong to the measurement. A finite numerical identity, empirical fit, and unsupported physical interpretation must not share a generic validation claim.

### Execute and reanalyze the same experiment

From the repository's `algorithmic-gas` directory, run the registered twistor correlation experiment with seed 7 and 96 updates:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  VI-36 7 96 > /tmp/partvi-twistor-run.json
```

The output contains both the snapshot and its evidence. Extract the evidence and reanalyze it through the same Rust measurement path:

```bash
python3 -c 'import json; p=json.load(open("/tmp/partvi-twistor-run.json")); json.dump(p["evidence"],open("/tmp/partvi-twistor-evidence.json","w"))'
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  analyze /tmp/partvi-twistor-evidence.json > /tmp/partvi-twistor-reanalysis.json
```

Use `gas-lecture catalog` to inspect the implemented control specifications. The shared session API accepts explicit parameter values for native sweeps as well as the browser controls. Save the complete request and evidence with each comparison.

### Validation criteria

- **Execution and tracking:** each default session must execute gas updates and compute its declared measured outputs; source identities, stages, and event graphs must resolve under the stated recording budget.
- **Exact identities:** gate reconstruction, stage telescopes, matrix identities, finite empirical differentiation, graph cuts, and energy decompositions require numerical residual checks against independently implemented expressions where available.
- **Numerical geometry:** vary finite-difference and transport resolution while holding the recorded conditional context fixed; retain clipping and support diagnostics.
- **Statistics and prediction:** use independent continuations or independent run pairs for their uncertainty estimates. Chronological training and holdout remain dependent trajectory segments. Compare fitted models with their documented baselines and inspect sample support.
- **Stress and replay:** exercise seeds, engine controls, history, noise laws, protocol controls, extinction, and degenerate observables. Reanalysis must reproduce measurements from evidence without advancing or altering the simulation state.

:::{div} feynman-prose
The validation target is a correctly executed and correctly characterized algorithm. A proposed field equation can fail its prediction test while the experiment succeeds in measuring that failure accurately.
:::
