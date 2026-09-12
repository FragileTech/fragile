# Volume II, Part V: interactive Fractal Set experiments

(sec-partv-experiments-purpose)=
## 1. Teaching sequence and numerical contract

:::{div} feynman-prose
Follow one walker through an update. Its position moves, its fitness is recomputed, and it may copy material from a donor whose state belongs to another recorded time. The Fractal Set records these different relations explicitly. Once we can recover what happened, geometry becomes a calculation on that history.

Every experiment in this guide runs the planar Euclidean Gas through the compiled Rust session. Rust also constructs the graphs, cells, derivatives, statistics and plot arrays. The browser supplies controls and draws the returned results. A prediction is evaluated against the same executed configuration and the same observable as its measurement.

The default experiments use a quadratic objective, 32 walkers, timestep 0.04, donor memory 2, viscosity 0.15 and Gaussian innovations; V-11 starts with 16 walkers. Shared controls vary population, timestep, memory, viscosity and innovation law. V-05–V-16 and V-20 execute the adaptive fitness-metric provider. Metric regularization therefore changes their actual dynamics as well as their measured geometry. The request and the complete executed configuration accompany the result.

For V-01–V-04, scientific measurements use the full recorded event graph. The scene shows the last eight recorded transitions and all their selected edges, including the actual earlier endpoints of incoming historical relations. Its time window and displayed or omitted counts are explicit. Scrubbing the retained display frames revisits earlier scene windows; the exported archive retains the complete recorded history for reconstruction.

The scene has two spatial coordinates and a physical-time coordinate. Camera rotation changes the view. A chart time-scale control, where present, changes the explicitly defined geometric comparison. These are different operations. Likewise, a finite-step identity, a reconstruction residual and a fitted empirical relation have different meanings; the individual descriptions below identify which comparison is being made.
:::

:::{div} feynman-added
| IDs | Lecture chapter | Measured object |
|---|---|---|
| V-01–V-04 | [The Fractal Set](../2_fractal_gas/2_fractal_set/01_fractal_set.md) | Executed event relations, scalar reconstruction and tangent turning |
| V-05–V-08 | [Emergent Geometry](../2_fractal_gas/3_fitness_manifold/01_emergent_geometry.md) | Executed conditional Hessian, metric, thermostat moments and interacting spread |
| V-09–V-12 | [Scutoid Spacetime](../2_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md) | Cells, stage interface changes, slot slabs and mesh maintenance |
| V-13–V-16 | [Continuum Discharge](../2_fractal_gas/convergence_program/16_continuum_discharge.md) | Conditional derivatives, empirical spatial kernels and held-out fitness prediction |
| V-17–V-20 | [Causal Set Theory](../2_fractal_gas/2_fractal_set/02_causal_set_theory.md) | Recorded orders, regional occupancy, correlation dimension and executed curvature |
:::

(sec-partv-experiments-record)=
## 2. Record the interaction before interpreting it

### V-01 — A slot is not an ancestor

:::{div} feynman-prose
**Where and why.** Place at `sec-nodes`. Which events inherit this walker’s material, and which follow its numerical slot? Follow an event through the executed graph. Cloning replaces a recipient’s material while preserving its numerical slot, so the two reachability curves answer different questions.

**Display and experiment.** Inspect the recent recorded event scene and choose a slot. The scientific query uses the full archive graph: it starts from a recorded event of that slot and plots the number of material descendants and causal-slot future events on each recorded step. The scene window restricts drawing, not the reachability calculation. Increase donor memory and inspect sources on earlier slices. Event addresses retain epoch, step, population version, slot and generation.

**Why the curves differ.** A slot is storage that survives copying; material ancestry follows the donor used by an accepted replacement. The generation counter counts replacements of the recipient slot. It is neither the donor’s generation nor genealogical depth. Graph queries use immutable source identities. A source outside recording coverage raises a coverage error, because looking up its current slot would answer the wrong question.
:::

### V-02 — Build the three-edge interaction

:::{div} feynman-prose
**Where and why.** Place at `sec-fractal-set`. Do the recorded interaction triangles close? Read CST, IG and IA directly from the archive. Closing a triangle checks that its events, orientations and recorded displacements describe the same interaction.

**Display and experiment.** Inspect CST, interaction and attribution edges alongside triangle counts, channels per triangle and displacement closure residuals. Follow a same-frame triangle around its three edges, then compare an interaction involving historical memory. The graph keeps relation kinds and recorded times explicit.

**Prediction and measurement.** With the recorded boundary orientation, the displacement residual is the norm of `u + v - w`; the integer graph also evaluates the boundary-of-boundary identity. These checks complement one another: graph incidence can be correct while the coordinate attached to an event is wrong. Self interactions and zero displacements are retained according to the graph’s conventions; their existence does not create a nondegenerate geometric face.
:::

### V-03 — Reconstruct the run from its record

:::{div} feynman-prose
**Where and why.** Place at `sec-reconstruction`. Can scalar addresses and spinor coordinates recover the executed stages and edges? Reconstruction uses every recorded stage and actual edge displacement. The scalar and Spin(2) residuals test two distinct representations of the same executed run.

**Display and experiment.** Plot the maximum scalar reconstruction error for every recorded stage, and separately plot the Spin(2) decoding error of actual edge displacements. Follow a step with copying through the scene. Repeat the analysis after archive serialization and restoration.

**Prediction and measurement.** Scalar addresses recover the stored numerical components. For a planar displacement represented by a complex number, the Spin(2) codec stores a square root whose square decodes the vector. This card applies the codec to recorded edges. Its roundtrip residual checks representation and tracking; it does not measure a dynamical spin precession. Stage names and event versions identify which actual state supplies each component.
:::

### V-04 — Turn along recorded edge tangents

:::{div} feynman-prose
**Where and why.** Place at `sec-guarantees`. How much does the direction turn around an actual interaction triangle? Compute oriented angles between successive nonzero recorded displacements. This is a connection defined from planar tangents; its closure does not identify an independent gauge field.

**Display and experiment.** Plot the angles of nonzero recorded edge displacements and their support counts. For a triangle whose three boundary tangents are supported, plot the summed oriented turn and its phase closure residual. Change donor memory or timestep to obtain different executed interactions, then inspect how their directions turn.

**Prediction and measurement.** For successive tangent vectors, the turn is computed with the signed `atan2(cross, dot)` angle. The boundary tangents are `u`, `v`, and `-w`. A nondegenerate closed planar triangle has total turning equal to an integer multiple of two pi, so its complex turning phase returns to one. The readout measures this geometric identity on actual interaction displacements.

This connection is defined from planar tangent directions. It does not supply a separate SU(2) gauge field or identify a Wilson action for the gas. A zero tangent has no direction; unsupported triangle turns are omitted while measured edge orientations and support remain visible. The loop discussion in the chapter motivates the question, but this experiment’s observable is tangent turning.
:::

(sec-partv-experiments-metric)=
## 3. Make the metric an observable calculation

### V-05 — From fitness curvature to a metric

:::{div} feynman-prose
**Where and why.** Place at `sec-adaptive-diffusion-tensor`. Does the executed Hessian agree with differences of the same conditional fitness? Freeze the actual donor and population context, perturb the recorded O input, and compare scalar differences with the analytic Hessian used by the metric provider.

**Display and experiment.** Show the mean matrix norms of the executed raw fitness Hessian and regularized metric. For a selected available slot, compare its recorded Hessian with central differences of scalar fitness at the actual O-stage input. The result retains every matrix, query point, slot, difference step and coverage mask. If the selected slot lacks an evaluation, the audit reports the available slot it uses.

**Prediction and measurement.** The scalar evaluations reconstruct the executed conditional field, including its sampled donor context and population normalizers. They use the A1 input whose population version matches the O-stage field evaluation. The audit computes differences at a chosen step and at half that step, reporting both the discrepancy from the analytic Hessian and the refinement discrepancy.

Decrease the difference step to study truncation error, then look for the scale at which floating-point cancellation competes with refinement. Change metric regularization and rerun to observe the feedback through actual adaptive noise. The objective Hessian and the conditional fitness Hessian are distinct quantities; the provider’s recorded values determine which one defines this metric.
:::

### V-06 — Predict the executed thermostat increment

:::{div} feynman-prose
**Where and why.** Place at `sec-equivalence-principle`. Does the actual O-stage noise factor predict the energy change? Each update supplies its own conditional prediction. Compare realized energy increments with that prediction without treating changing walker covariances as identical samples.

**Display and experiment.** Plot the realized O-stage energy increment and its conditional prediction on each executed step. The result also supplies conditional total-momentum covariance and the number of eligible noise rows. Vary timestep, metric regularization or innovation law and rerun.

**Prediction and measurement.** For an O update written as damped input velocity plus a centered innovation with factor B, the covariance contribution is B times its transpose, with the actual thermal scale included. The conditional energy prediction combines that second moment with the damped input energy. The Rust measurement reads the executed factor and stage data rather than averaging incompatible matrices from different steps.

A realized increment need not equal its conditional mean. The difference is the fluctuation we want to inspect. This card contains one interacting trajectory, so walker rows and successive steps do not provide independent-replica error bars. Gaussian and standardized-uniform innovations share their specified second moments while differing in higher moments.
:::

### V-07 — Coordinate area versus geometric area

:::{div} feynman-prose
**Where and why.** Place at `sec-riemannian-volume`. How does a recorded metric change cell areas? Partition actual walker sites using the first available recorded O-stage metric as a constant local chart. Compare coordinate and geometric cell areas.

**Display and experiment.** Construct clipped metric Voronoi cells from the final eligible walker positions. Plot coordinate area and geometric area for every eligible site, with the partition closure residual. The scene displays the cells and their neighbors. Rerun with changed regularization or population and inspect the resulting partition.

**Prediction and measurement.** The first available executed O-stage metric is frozen across this local chart. Its observation rectangle encloses the actual final cloud with a recorded geometric margin. For this constant chart metric, each geometric area is its coordinate area times the square root of the determinant. The total coordinate area should equal the rectangle’s area.

This is a local-chart construction derived from an executed metric sample. It is not spatial quadrature of a varying metric determinant. Duplicates preserve walker identities while the geometric ownership convention assigns their shared region once. Site indices in area arrays and original slot identities in the scene serve different indexing roles.
:::

### V-08 — Measure interacting relaxation

:::{div} feynman-prose
**Where and why.** Place at `sec-hypocoercivity`. How does population spread evolve while cloning and the thermostat act? Follow empirical position variances with denominator N and compare thermostat increments with their conditional predictions. Cloning, donor memory and viscosity remain active.

**Display and experiment.** Follow the two empirical coordinate variances of the actual quadratic-objective gas, alongside its realized and conditionally predicted O-stage energy increments. Change viscosity, donor memory or innovation law while keeping the seed visible. Watch how a cloning event can alter population spread before the next thermostat increment.

**Measured quantity.** At a final recorded population of n eligible walkers, the plotted variance is the centered sum of squares divided by n. It is the spread of that empirical population. Cloning makes walker values dependent, and its update changes which material they represent. Dividing by n minus one would not in itself produce an unbiased estimate of an independent equilibrium ensemble.

The exact conditional thermostat comparison still applies at each O stage. A stationary covariance or relaxation rate of the complete interacting gas requires a separate dynamical prediction. This card does not attach an independent harmonic-oscillator Lyapunov curve to an interacting trajectory. Its useful experiment is to observe how empirical spread and the measured thermostat budget respond to actual algorithm controls.
:::

(sec-partv-experiments-cells)=
## 4. Follow interfaces and cell volumes through operations

### V-09 — Cells and their dual graph

:::{div} feynman-prose
**Where and why.** Place at `sec-time-varying-voronoi`. Which recorded walkers share a positive-length cell interface? Construct cells from eligible recorded sites and an executed metric. Preserve slot identities when eligibility changes and inspect how geometry creates the neighbor graph.

**Display and experiment.** Display final eligible sites, clipped metric cells and the dual neighbor graph, together with the degree of each cell. The chart uses a recorded O-stage metric held constant across the observation rectangle. Increase population size or change the dynamics and examine which cells gain neighbors.

**Prediction and measurement.** Positive-length shared interfaces define the neighbor graph. A point at which several cells meet is a different geometric object. Duplicate sites retain their slot labels; the ownership convention avoids counting the same area twice. Partition closure checks complete area accounting. A Delaunay triangulation can choose a diagonal in a co-circular configuration even when that diagonal has no positive-length dual interface; the cell graph’s definition remains explicit.
:::

### V-10 — Which operation changed the neighbors?

:::{div} feynman-prose
**Where and why.** Place at `sec-cloning-topological-transitions`. Did the clone stage or subsequent kinetics alter these interfaces? Compare before, post-clone and final meshes with one common chart per step. A changed edge is counted once, including when eligibility changes.

**Display and experiment.** For every recorded update, compare the before, validated post-clone and final cell graphs. The plot separates the unique interfaces changed across the clone stage from those changed across the kinetic stage. Inspect a step with accepted copies, then follow its subsequent motion.

**Prediction and measurement.** Each step uses one metric and one rectangle enclosing all three point sets. Neighbor edges are canonical unordered pairs of original slot IDs. The symmetric difference of two such edge sets counts each changed interface once. Eligibility changes must not renumber those slots: an edge belonging to slots 2 and 7 cannot become an edge between rows 0 and 1 merely because other walkers died. The exported interface records preserve the slot sets and edge sets for all three stages.
:::

### V-11 — Sweep the cells through time

:::{div} feynman-prose
**Where and why.** Place at `sec-scutoid-cell-definition`. What volume belongs to cells following the recorded numerical slots? Use common eligible slots at before, post-clone and final stages. Reconstruct a finite-resolution slab from these endpoints and retain clone jumps at one physical time.

**Display and experiment.** Build a slab from the final recorded update’s before, post-clone and final positions. Track only slots eligible at all three stages, and display that support explicitly. Before and post-clone frames share one time; the final frame lies one algorithm timestep later. The scene shows endpoints, clone replacement edges, kinetic motion and extracted cell boundary faces. Increase slab refinement to inspect the geometry calculation.

**Prediction and measurement.** The common tetrahedral partition clips interpolated nearest-site scores inside one observation slab. Its volume accounting is shared by all cells. Cloning replacement has zero duration; the reconstruction introduces no positive-duration path through the recipient’s jump. Motion between recorded endpoints is a declared geometric interpolation, not an additional engine trajectory.

A closed partition can still approximate individual moving boundaries poorly at coarse resolution. Compare refinement separately from total closure. At least two slots must remain eligible across all three stages. The result identifies excluded support rather than silently matching a different collection of walkers. Scene owners refer back to the actual common slot IDs.
:::

### V-12 — Maintain the mesh and count its activity

:::{div} feynman-prose
**Where and why.** Place at `sec-dynamic-delaunay-algorithm`. Does maintaining a mesh agree with rebuilding it from the same recorded sites? Measure geometry operations on up to eight recorded frames with a fixed common-slot set. Keep interface counts separate from maintenance diagnostics.

**Display and experiment.** Analyze up to eight final recorded frames. Retain the slots eligible throughout that window and compute their triangulation maintenance diagnostics. Plot the number of metric-cell interfaces in each retained frame; inspect operation counts and independent reconstruction comparisons in the result details.

**Prediction and measurement.** Maintaining a triangulation through removals and insertions should agree with a full reconstruction under matching degeneracy conventions. This tests numerical geometry on the moving, cloning population. Common-slot selection keeps vertex identities aligned when eligibility changes. Both methods use the same frozen recorded chart metric. Interface counts, affected vertices and maintenance operations answer different questions; one of them cannot be relabeled as another or interpreted as a complexity exponent from a single short trajectory.
:::

### V-13 — Differentiate the executed conditional field

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-regularity`. Which changes belong to a frozen field and which come from the next algorithm step? Keep the O-stage inputs and donor references fixed for differentiation. Then compare with actual slot motion across updates, where cloning and new companions can change the stratum.

**Display and experiment.** Inspect the executed Hessian norm and the scalar finite-difference audit used in V-05. Add the mean displacement of slots present both before and after an update. Change the finite-difference step while keeping the recorded conditional context fixed, then compare different actual updates.

**Prediction and measurement.** A conditional derivative holds the donor references, eligibility and other context fixed while moving the query point. The next algorithm update can change all of these as well as the point itself. Smoothness within one conditional stratum therefore does not predict that a time series of Hessians must be smooth across cloning or companion changes.

The selected-slot audit reports the actual O-stage query and its matching population version. Historical donor coordinates are reconstructed from their immutable source records. Derivative support and spectral-clipping thresholds remain explicit. The measured between-step slot displacement includes possible cloning replacement and is not a pure differential motion along one fixed field.
:::

(sec-partv-experiments-continuum)=
## 5. Test sampling, operators and geometric statistics

### V-14 — Normalize a kernel on the recorded population

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-sampling`. Does the spatial averaging operator preserve a constant field? Build weights from actual pre-clone positions. Row normalization defines an empirical smoother; it does not by itself turn the sampling law into geometric volume.

**Display and experiment.** On the final update’s actual pre-clone eligible positions, build Gaussian weights at the selected bandwidth and normalize each row. Plot every row sum against one. Inspect the returned matrix, then compare narrow and broad bandwidths on the same kind of clustered population.

**Prediction and measurement.** Each row divides positive weights by their sum, including the self weight. Applying that matrix to a constant field reproduces the constant. A narrow row emphasizes nearby recorded sites; a broad row averages more of the population.

This checks an empirical averaging operator. Row normalization does not estimate the unknown sampling density or transform the empirical measure into Riemannian volume. Cloning-generated concentrations are part of the measured cloud and therefore influence its averaging weights. The volume and sampling discussion at the lecture placement supplies the motivation for asking exactly which measure an operator uses.
:::

### V-15 — Measure temporal and spatial fitness differences

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-kernels`. How do slot-time changes compare with spatially averaged fitness differences? Compare a three-frame second slot-time difference with a row-normalized spatial kernel generator. Cloning jumps remain in the temporal term.

**Display and experiment.** Use recorded pre-clone fitness as a scalar field. Plot its three-frame second difference along numerical slots and its spatial kernel generator on the final pre-clone population. At least three recorded frames are required. Change bandwidth to alter the spatial averaging scale and timestep to alter the executed dynamics and time denominator.

**Measured quantity.** The spatial term averages the difference between neighboring fitness and the selected slot’s fitness, using V-14’s row-normalized Gaussian weights, then divides by bandwidth squared. The temporal term takes the current fitness minus twice the preceding fitness plus the earlier fitness, divided by timestep squared. It follows slots eligible on all three frames.

These are two explicitly defined discrete terms. Fitness changes with population normalization, companions and cloning; the temporal term includes those changes. Agreement with a continuum wave equation would require an independently derived balance, not a visual resemblance between two curves. The experiment exposes the actual terms that such a derivation would need to explain.
:::

### V-16 — Predict held-out fitness across bandwidths

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-variance`. Which spatial bandwidth predicts later recorded fitness most accurately? Train a spatial kernel predictor on the first half of the trajectory and score the second half. Dependence and changing donor context remain part of the prediction problem.

**Display and experiment.** Split the recorded trajectory chronologically. Use pre-clone positions and fitness from the first half to predict fitness at positions in the second half with Gaussian kernel regression. Plot held-out root mean square error at half, equal and double the selected bandwidth. The result records the training and held-out frame counts.

**Prediction and measurement.** Every scored target comes from a later frame excluded from fitting. The reported error is computed directly from those targets. Compare widths, increase the run length and repeat with another seed. A narrower bandwidth need not perform better: it can leave very little effective support, while a wider one can smooth away spatial variation.

The predictor uses position, although actual fitness also depends on the changing population and donor context. Its error measures the usefulness of that restricted description. A chronological split prevents fitting to the scored values but does not make the two halves independent. The card does not attach independent-replica standard errors or a universal bandwidth exponent to these dependent observations.
:::

### V-17 — Recorded order and chart light cones

:::{div} feynman-prose
**Where and why.** Place at `sec-cst-geometric-comparison`. How often do causal-slot reachability and a chosen chart cone agree? Apply both relations to the same recorded events. The cone speed defines the comparison geometry and does not modify the gas transition.

**Display and experiment.** Apply the recorded causal-slot reachability relation and an explicit flat-chart cone comparison to the same bounded set of archived events. The plot counts pairs admitted by both relations, by the recorded order alone and by the chart cone alone. Change the chart cone speed to see which geometric comparisons change.

**Prediction and measurement.** The chart cone uses actual event positions and physical times with its selected speed. The recorded relation comes from the event graph. Cone speed is an analysis control, not a speed limit imposed on the Euclidean Gas. Increasing it admits more chart-comparable pairs on a fixed event set; it does not alter graph reachability. The exported comparison identifies the actual nodes used. Pair disagreement quantifies this particular comparison and does not change the recorded relation into a Lorentzian causal law.
:::

### V-18 — Count walkers in an observation region

:::{div} feynman-prose
**Where and why.** Place at `sec-faithful-discretization`. How does regional occupancy change during the executed run? Count actual final-stage walkers inside a centered square at each step. Shared ancestry and temporal dependence remain in the counts; no Poisson count law is imposed.

**Display and experiment.** Count final-stage eligible walkers inside the centered square whose half-width is the region control. Plot the count over recorded steps and its density per coordinate area. Increase the square size on a fixed run, then vary donor memory or viscosity and rerun to study altered occupancy.

**Prediction and measurement.** The square has coordinate area four times the half-width squared. The density is the observed count divided by that area. Enlarging nested squares cannot lower the count for a fixed frame, although the area-normalized density can decrease.

These observations are interacting populations connected by ancestry and time. A Poisson or independent-binomial variance is not imposed. Nor does a regional count alone determine geometric volume without a sampling-density relation. This card measures the occupancy and normalization that a proposed count-to-volume law would have to explain.
:::

### V-19 — Measure correlation dimension across scales

:::{div} feynman-prose
**Where and why.** Place at `sec-cst-dimension-curvature`. How does the fraction of nearby recorded event pairs change with radius? Measure Euclidean pair distances in (x, y, c t). The result is a correlation slope of recorded events, with time scaling, shared ancestry and saturation visible in its interpretation.

**Display and experiment.** Form the bounded set of recorded event coordinates `(x, y, c t)`, where c is the chart time-scale control. Count all unordered pairs whose Euclidean distance is at most each of twelve radii, starting at 0.05 and multiplying by 1.5. Plot the resulting pair fraction and its logarithmic slope between consecutive positive fractions.

**Measured quantity.** The denominator is the total number of unordered pairs in the selected event set. The correlation integral is nondecreasing and lies between zero and one. A scale range with an approximately constant slope is evidence of that finite-range scaling of the recorded cloud. At large radii the fraction saturates and the slope falls; at very small radii finite samples and coincident positions can dominate.

The chart has three coordinates, but the measured slope need not be three. Temporal sampling, confinement, cloning and shared ancestry shape the actual distribution. Changing c rescales time relative to space in the distance calculation. This is correlation dimension in an explicit Euclidean chart, not inversion of a Lorentzian comparable-pair fraction. The pair observations are dependent, so the plotted slope has no independent-pair confidence band.
:::

### V-20 — Read curvature from the executed metric

:::{div} feynman-prose
**Where and why.** Place at `sec-physical-consequences`. Does recorded scalar curvature agree with contracting the recorded Ricci tensor? Read the metric, Ricci tensor and scalar curvature from executed O stages. The summary plots mean absolute scalar curvature; signed values and coverage remain in the result details.

**Display and experiment.** Read the metric, Ricci tensor and scalar curvature recorded by the adaptive provider at actual O stages. Plot the average absolute scalar curvature over available walkers, and the largest residual between scalar curvature and the inverse-metric Ricci contraction on each step. Signed scalar values and availability masks remain in the result details. Vary metric regularization and rerun to see its effect through the executed dynamics.

**Prediction and measurement.** Scalar curvature equals the contraction of Ricci with the inverse metric at a supported evaluation. The analyzer computes that contraction directly from the archived two-by-two matrices and compares it with the separately recorded scalar readout. This checks consistency of the tensor fields and indexing; it is not an independent derivation of their complete differential geometry.

Curvature requires derivatives beyond those defining the metric itself. At a spectral-clipping branch boundary, the corresponding derivative readout can be unavailable. Retain that coverage information. A mean absolute curvature plot measures magnitude and cannot establish a sign; inspect the signed records when asking whether curvature is positive or negative. No imposed constant-curvature sampling manifold supplies the displayed data.
:::

(sec-partv-experiments-runtime)=
## 6. Rust sessions, provenance and reproducible interpretation

:::{div} feynman-prose
The shared lecture registry defines experiment IDs, control defaults and admitted values. A request contains `id`, `seed`, `parameters` and the recorded step budget. Rust resolves that request, builds the Euclidean Gas configuration, starts recording and advances the session. The same scientific analyzer consumes the archive in native and compiled execution.

The core library owns tracking, immutable donor sources, Fractal Set relations and reusable geometry. The benchmarks crate owns the executed lecture protocol and Part V archive measurements. Results include plots, metrics, detailed numerical records, the executed gas configuration and the origin `executed_algorithm_archive`. V-01–V-04 return full-archive graph counts in a compact summary; their plot measurements use that full graph. The display payload contains the recent scene window rather than a duplicate serialization of the complete graph. Exported archives preserve the event data needed to reconstruct it. Scenes use recorded positions with actual slot ownership; generated cell surfaces are explicitly reconstructed geometry on those positions.

An analysis may require particular support: three frames for a temporal second difference, common eligible slots for a slab, and available metric derivatives for curvature. Missing coverage must remain visible or produce a specific capability result. Historical sources outside the recording interval cannot be replaced by current values from the same slot. Invalid controls and malformed archives are errors rather than an invitation to run another model.

The derivative audit independently reevaluates scalar fitness from the archive context. Its finite differences are a diagnostic of the analytic derivative used by the engine. They do not replace that production derivative. The plotting layer receives the numerical arrays and performs no fitness, geometry, dynamics or statistical calculation.
:::

(sec-partv-experiments-audit)=
## 7. Validation and the meaning of a residual

:::{div} feynman-prose
The native Part V integration check runs an actual planar gas, applies all twenty analyzers to its archive and checks that each result supplies finite plotted values and recorded scene nodes. Cell experiments additionally supply cell faces. The test verifies that analysis leaves the archive byte-for-byte unchanged. This is a concrete check that observation does not modify the run being studied.

Focused checks compare V-05 and V-13 recorded Hessians with independent scalar differences, reanalyze serialized archives for deterministic geometry and curvature results, and exercise absorbing-boundary eligibility changes to verify original slot labels in interface records. The slab check verifies that its displayed owners match the common tracked slots. These checks pass in the native test configurations; browser integration and broad control sweeps have their own acceptance results.

Different residuals answer different questions. Triangle closure and scalar reconstruction test identities and addresses. Partition closure tests complete volume accounting; refinement tests the accuracy of individual reconstructed cells. The thermostat comparison tests a conditional expectation, whose realized residual fluctuates. The chronological fitness score tests a restricted predictor on later observations. A Ricci contraction checks internal tensor consistency. None of these readouts should borrow another one’s label or uncertainty model.

All twenty cards analyze one interacting trajectory per request. Shared ancestry, interactions and successive recorded times remain part of that data. Where independent-run uncertainty is needed, repeat complete sessions with independently addressed seeds and keep the run as the sampling unit. More walkers or more event pairs alone do not establish that independence.

The teaching sequence therefore keeps the question attached to the measured object: which material was inherited, which cell changed, which conditional noise law applied, which spatial predictor worked, and which scaling appeared in the recorded events. Each answer can be inspected back to the configuration and archive that produced it.
:::
