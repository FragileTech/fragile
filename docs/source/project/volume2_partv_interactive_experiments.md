# Volume II, Part V: interactive Fractal Set experiments

(sec-partv-experiments-purpose)=
## 1. Teaching sequence and numerical contract

:::{div} feynman-prose
A cloud of moving points becomes much more useful when we can ask what happened to one particular point. Did its slot move continuously? Did it inherit another walker's state? Which measurements influenced its next update? Part V turns those questions into a recorded structure, then uses that structure to study geometry and continuum observables.

The twenty experiments below follow that progression. V-01–V-04 establish identities, edges, reconstruction and transport. V-05–V-08 turn a declared fitness field into a metric and test the resulting covariance. V-09–V-12 follow cells and interfaces through recorded updates. V-13–V-16 connect smooth fields, sampling weights and calibrated kernels. V-17–V-20 compare recorded order with a specified spacetime geometry and measure counts, dimension and curvature.

The main display uses two spatial coordinates and a separate time axis. Rotating a scene changes its camera, never its metric, clock or causal comparisons. The engine runs inside the existing worker through the compiled Rust/WASM library. Reference calculations also run in Rust. JavaScript supplies controls, rendering and labels; it does not introduce a second scientific implementation.

This is an author-facing implementation and teaching document. Lecture placement follows `lecture/placements.json`; the five chapters are those actually listed under Part V in the Theory TOC. Each experiment states its numerical model, prediction, measurement and a sequence of changes that a reader can reproduce. The finite-reference experiments use independent replicas where their standard errors require independent observations.
:::

:::{div} feynman-added
| IDs | Lecture chapter | Numerical role |
|---|---|---|
| V-01–V-04 | [The Fractal Set](../2_fractal_gas/2_fractal_set/01_fractal_set.md) | Actual event archive and graph; scalar reconstruction; independent Spin(2) and supplied-connection identities |
| V-05–V-08 | [Emergent Geometry](../2_fractal_gas/3_fitness_manifold/01_emergent_geometry.md) | Exact conditional derivatives, spectral regularization, frozen OU samples and actual harmonic BAOAB |
| V-09–V-12 | [Scutoid Spacetime](../2_fractal_gas/3_fitness_manifold/02_scutoid_spacetime.md) | Native geometric queries on actual recorded sites; moving-cell reference partition |
| V-13–V-16 | [Continuum Discharge](../2_fractal_gas/convergence_program/16_continuum_discharge.md) | Smooth conditional field, known-density integration, calibrated signed kernel and Monte Carlo bandwidth sweep |
| V-17–V-20 | [Causal Set Theory](../2_fractal_gas/2_fractal_set/02_causal_set_theory.md) | Recorded/geometric order comparison; explicitly specified independent count, dimension and curvature models |
:::

(sec-partv-experiments-record)=
## 2. Record the interaction before interpreting it

### V-01 — A slot is not an ancestor

:::{div} feynman-prose
**Where and why.** Place at `sec-nodes`, beside the node and clone-source definitions. The reader needs two different notions of continuity: a persistent numerical slot and the material history inherited from a donor. Color alone cannot carry both meanings reliably.

**Display and controls.** Show the current position cloud, a space–time event graph, and counts separated by relation type. Event inspection identifies epoch, step, slot, generation and population version. Start with 64 walkers and seed 7; offer 16, 32 and 64 walkers, donor-history windows 0, 2 and 4, and ordinary, revival and singleton initial populations. The camera shows a bounded recent window while the archive retains every recorded microstep. V-01, V-02 and V-17 reuse that archive when walker count, donor-history setting, initial-population scenario and seed remain compatible; switching the explanation does not require rerunning the experiment.

**Experiment.** Advance one step and identify an accepted clone. Follow its recipient slot backward along CST and its inherited state backward along ancestry. Repeat for a rejected proposal. Select revival to start with an ineligible subset, then singleton to follow copies from the lone eligible donor. Enable historical donors and find an ancestry source on an earlier slice. A source outside the retained recording coverage appears as an unresolved earlier source, not as a current walker with the same slot number.

**Prediction and measurement.** Accepted copies use the recorded donor reference; rejected copies preserve the recipient's ancestry. The generation counter increments when that recipient is replaced; it is not the donor's generation or a genealogical depth. Count each relation separately and use an exact checkpoint replay to compare event identities.
:::

### V-02 — Build the three-edge interaction

:::{div} feynman-prose
**Where and why.** Place at `sec-fractal-set`, next to the interaction-triangle definition. Three directed edges tell a complete story: a source-time walker advances, its update refers back to a companion, and the source-time companion relation closes the loop.

**Display and controls.** Retain V-01's walker, history and initial-population controls. Overlay CST, distance IG, cloning IG, their IA relations and highlighted triangle faces. Keep historical interactions visibly separate from same-time interactions. Self draws remain in the record but do not create a degenerate triangle.

**Experiment.** Select one same-frame interaction and traverse its boundary in the displayed orientation. Hide one edge type and identify the missing explanation. Compare distance and cloning companions when they coincide and when they differ. Enable historical donors and inspect why the corresponding record has a different time pattern.

**Prediction and measurement.** For a triangle with vertices `(before recipient, after recipient, donor)`, the oriented boundary is CST + IA − IG. Its vertex boundary cancels exactly: the discrete identity is `boundary_squared_zero()`. The graph contains integer addresses, so topology checks use exact identities rather than geometric proximity. Channel annotations must preserve whether the same geometric triangle records one or both companion roles.
:::

### V-03 — Reconstruct the run from its record

:::{div} feynman-prose
**Where and why.** Place at `sec-reconstruction`, with a return link to the Spin(2) discussion under `sec-frame-invariance`. This experiment separates two concrete tasks: retrieving recorded numerical components and decoding a geometric representation of a vector.

**Display and controls.** Advance an actual 64-walker recorded run and reconstruct its post-kinetic positions from indexed scalar component addresses. Display reconstructed coordinates, component count and maximum absolute residual. Alongside it, show the independent native Spin(2) experiment; its rotation control ranges from zero to four pi.

**Experiment.** Inspect a step containing copying and compare its recorded before, post-clone and final coordinates. Restore a checkpoint and repeat the reconstruction query. In the spinor panel, rotate continuously through two pi and then four pi without re-encoding the vector at each angle.

**Prediction and measurement.** The scalar decoder returns the recorded components at the requested stage. For the two-dimensional codec, a vector is the complex square of its spinor. A continuously rotated spinor changes sign at two pi and returns at four pi, while the decoded vector has already returned after two pi. The native roundtrip and equivariance tests include zero, the negative-real branch, and very large and small finite vectors. The component reconstruction and the Spin(2) identity have separate residuals; they are separate operations.
:::

### V-04 — Triangles, plaquettes, and transport

:::{div} feynman-prose
**Where and why.** The catalog places this at `sec-guarantees`, following the chapter's interaction-triangle and Wilson-loop discussion. Link back to Sections 5.4–5.7. It teaches what changes under a local frame transformation and what can be compared without choosing a frame.

**Display and controls.** Use the connection-angle control from zero to four pi. The native reference assigns oriented U(1) phases and noncommuting SU(2) transports to a small loop. Display original and transformed edge data, U(1) loop real/imaginary components, the normalized SU(2) Wilson trace and the invariance residual.

**Experiment.** Begin with zero connection angle, where the loop product is the identity. Increase the angle, inspect the noncommuting SU(2) factors, and apply different gauges at the vertices. Reverse an edge only with the inverse transport. When explaining two triangles as a plaquette, transport the second triangle to the first triangle's basepoint before multiplication.

**Prediction and measurement.** With `U_ij -> G_i U_ij G_j^-1`, a closed-loop product changes by conjugation and its trace remains unchanged. The tests multiply the actual supplied factors and their transformed counterparts. This experiment studies an explicitly supplied connection; a generic moving swarm does not supply that connection merely by producing an adjacency graph.
:::

(sec-partv-experiments-metric)=
## 3. Make the metric an observable calculation

### V-05 — From fitness curvature to a metric

:::{div} feynman-prose
**Where and why.** Place at `sec-adaptive-diffusion-tensor`. The key prediction concerns the Hessian of a specified conditional fitness field, not the raw objective Hessian. Showing the intermediate matrices makes that distinction visible.

**Display and controls.** Take frozen coordinates and distance companions from a recorded engine step, then declare the conditional teaching profile used for the query: quadratic objective, global standardizers with `sigma_min=0.1`, logistic channels `2/(1+exp(-z))+10^-6`, unit channel powers and position-only distance. These profile parameters are part of the displayed experiment; archived positions alone do not identify the archived engine fitness. Query walker zero along a horizontal line. Show its fitness, exact two-coordinate Hessian, regularized metric and eigenvalues. Controls include 16–64 walkers, query position, distance regularization, metric regularization and strict versus clipped spectral policy. The native derivative API also supports the declared smooth objective presets and fitness exponents.

**Experiment.** Move only the query position. Then increase the distance regularizer and compare the curvature near a companion. Change the metric regularizer while leaving the fitness field fixed. Finally select strict regularization in a configuration with a negative Hessian eigenvalue and inspect the positive-margin admission condition.

**Prediction and measurement.** Second-order forward jets differentiate the full supported fitness calculation, including its population normalization. Companion source coordinates, other walker coordinates, velocities and alive membership belong to the frozen context. The strict policy uses `g=H+epsilon I`; the clipped policy additionally raises metric eigenvalues to the declared floor. Plot raw and regularized spectra separately. Derivative checks compare exact analytical fixtures and independent finite-difference diagnostics; finite differences are not the production Hessian.
:::

### V-06 — Predict the next noise cloud

:::{div} feynman-prose
**Where and why.** Place at `sec-equivalence-principle`. An ellipse is a prediction about many random increments, so the natural experiment freezes the coefficients and draws many independent increments before moving the walkers again.

**Display and controls.** By default, run the actual adaptive BAOAB gas and read slot zero's metric from its archived O-stage evaluation. Draw an independent native innovation ensemble with that exact metric, next to predicted and measured covariance matrices. A separate constant-metric source offers anisotropy from 1 to 8. Both modes expose duration 0.01–0.2 and independent innovations 128–4096. Friction is 1 and temperature is 0.4.

**Experiment.** First select the constant source, start with the identity metric, and increase one eigenvalue to predict the narrower velocity direction. Switch to the sampled-fitness source and inspect the actual O-stage Hessian/metric before comparing its independent innovation cloud. Increase the number of innovations without changing the selected metric. Then change the duration and predict the overall scale using the exact exponential factor.

**Prediction and measurement.** Conditional covariance is `T(1-exp(-2 gamma h)) g^-1`. The noise factor is an amplitude; its covariance is the factor times its transpose. The finite covariance discrepancy is measured from independent frozen O-step draws. A separate matrix divides each covariance discrepancy by its Gaussian sampling standard error, and a metric reports the largest absolute discrepancy in standard-error units. This panel isolates that conditional law. The adaptive provider evaluates the metric at the actual O-stage position and archives it with the raw Hessian, inverse square root, evaluation coverage and actual noise factor. Its context freezes the pre-clone sampled field; a revived recipient uses a separately identified donor-field extension. The independent ensemble estimates conditional covariance at that one recorded evaluation; it does not pool unequal covariance matrices from a changing trajectory.
:::

### V-07 — Coordinate area versus geometric volume

:::{div} feynman-prose
**Where and why.** Place at `sec-riemannian-volume`. Readers can directly see why equal coordinate areas need not represent equal geometric volumes under a different metric.

**Display and controls.** Compute clipped metric Voronoi cells on the current recorded sites in the square `[-3,3]^2`. Show their polygons, coordinate areas and geometric volumes. Use the walker-count and anisotropy controls, with an explicit constant metric for the geometric query.

**Experiment.** Freeze the sites, begin at the identity, and increase one metric eigenvalue. Predict both effects: bisectors can move because the metric changes distance, and every coordinate area acquires the common factor `sqrt(det g)` when interpreted as geometric volume. Advance the actual gas and recompute the partition.

**Prediction and measurement.** Coordinate areas sum to 36 for this clipped square. With constant metric, geometric volumes sum to `36 sqrt(det g)`. The closure residual concerns a complete partition of the observation window; cells at its boundary are clipped cells. Coincident walkers retain their identities but share geometric ownership according to the recorded duplicate convention. This default is a constant-metric integration example, not a quadrature of a spatially varying determinant.
:::

### V-08 — Relaxation in anisotropic geometry

:::{div} feynman-prose
**Where and why.** Place at `sec-hypocoercivity`. It extends the successful harmonic relaxation check from the earlier parts to a full position–velocity covariance with anisotropic noise.

**Display and controls.** Run the actual quadratic-potential BAOAB engine with neutral selection, 64–256 independent walkers, timestep 0.04 and a constant anisotropic factor. Plot the unbiased position sample variance against the exact discrete transient and its pointwise 1.96-standard-error bands, with the discrete and continuous stationary references shown separately. Show the full covariance matrices in coordinate order `(x,y,vx,vy)`.

**Experiment.** Begin with isotropic noise, then increase anisotropy. Watch the transient separately from the stationary prediction. Compare the two covariance matrices; they solve different equations at a finite timestep. Re-run with independent seeds or a larger population to distinguish ensemble variation from the equilibrium shift.

**Prediction and measurement.** The native reference constructs the actual BAOAB transition matrix and innovation covariance, then solves its discrete Lyapunov equation. It also solves the continuous Lyapunov equation for the declared linear SDE. The displayed residual checks the matrix equation itself. Noncommuting stiffness and diffusion matrices require the full covariance solve; a componentwise inverse-stiffness shortcut is not generally sufficient. The exact transient begins from the actual independent uniform initialization. Its sample-variance uncertainty propagates the initial fourth cumulant as well as the covariance; replacing that initial distribution by a Gaussian would give the wrong early-time band. The measured variance divides its centered sum of squares by `N-1`, matching the population covariance in expectation. The stationary matrices remain the long-time references.
:::

(sec-partv-experiments-cells)=
## 4. Follow interfaces and cell volumes through operations

### V-09 — Cells and their dual graph

:::{div} feynman-prose
**Where and why.** Place at `sec-time-varying-voronoi`. This supplies the geometric object used by the next three experiments.

**Display and controls.** Draw clipped metric Voronoi cells, sites and unique shared interfaces. Offer 16–64 walkers, metric anisotropy, two coordinates for moving slot zero, and constant versus variable distance geometry. The variable mode uses grid refinements 8, 16 and 24. The underlying gas remains a separately configured run; a geometric-query metric changes the interpretation of its sites unless an adaptive run is explicitly selected.

**Experiment.** Move slot zero across a bisector and inspect the changed neighbors. Place it on another site and inspect the duplicate record. Use a symmetric four-site fixture to distinguish a positive-length interface from a point where several cells meet. Increase anisotropy and predict the transformed distance comparison. In variable mode, compare the native nearest-site graph-distance ownership grids as refinement increases. The metric field is explicitly `diag(1+a x²/9,1+a y²/9)`. Constant-metric adjacency and volume plots remain labeled as comparisons.

**Prediction and measurement.** Cells cover the clipped region and their interiors do not overlap. Exact duplicates are not moved by artificial jitter. A representative owns their common cell, and duplicate entries retain their slot identity. Geometric shared interfaces and the deterministic Delaunay triangulation have different conventions in co-circular cases: a triangulation may select a diagonal whose dual Voronoi interface has zero length. The experiment counts the requested object explicitly. Variable geometry uses a refining directional metric graph whose physical edge scale shrinks and whose directional resolution increases; its ownership image is a numerical distance partition rather than an exact finite polygon construction.
:::

### V-10 — Which operation changed the neighbors?

:::{div} feynman-prose
**Where and why.** Place at `sec-cloning-topological-transitions`. A before/after movie can conceal whether copying or subsequent kinetic motion changed an interface.

**Display and controls.** Compare native meshes at recorded pre-clone, validated post-clone and final positions. Display changed unique interfaces for the clone stage and kinetic stage separately. Retain walker count and anisotropy controls, with an active/off copying selector. The off setting sets both fitness exponents to zero so ordinary selection produces no accepted copies.

**Experiment.** Advance a single recorded step. Find a slot that copied and inspect its old and new cells. Compare the post-clone and final partitions to isolate subsequent motion. Switch copying off and repeat: the clone-stage interface difference should vanish unless a separately recorded transform or boundary operation changed positions.

**Prediction and measurement.** Interface changes are symmetric differences of canonical undirected edge sets. One lost interface contributes once, even though it touches two cells. The stage names come from the archive, not from interpolating between end-of-step clouds. If the requested validated stage is unavailable, its status must remain explicit; reusing the initial cloud would hide the missing observation.
:::

### V-11 — Sweep the cells through time

:::{div} feynman-prose
**Where and why.** Place at `sec-scutoid-cell-definition`. A swept cell is a volume in two space dimensions plus time; it is not obtained by pairing polygon vertices that happen to have similar indices.

**Display and controls.** Render extracted cell boundary surfaces in a shared space–time partition, removing internal tetrahedral faces from the visible boundary mesh. Offer 8, 16 and 32 sites and refinement levels 2, 4, 6 and 8, subject to the native work limit. Keep the spatial metric and observation bounds visible. Clone jumps have separate before and after caps at the same physical time. Per-slot bars show the computed geometric three-volume of the recorded slab. The scene uses vertical display factor 25 to make thin slabs inspectable; that factor does not enter the volume calculation.

**Experiment.** First use static sites and compare each swept volume with its cell area times elapsed time. Next allow motion and increase refinement. Finally inspect a copying event: there is no fictitious positive-duration tube connecting the recipient's old position to its copied position.

**Prediction and measurement.** A common Freudenthal tetrahedral grid partitions the observation slab. The algorithm samples nearest-site scores at shared vertices and clips their affine interpolants inside each tetrahedron. All cell pieces therefore use a compatible partition. If two affine score fields agree throughout a tetrahedron, its full-dimensional ownership goes to the smaller slot index; treating equality as inside both candidates would count a positive volume twice. Equality along an ordinary shared face retains the common boundary. The closure residual compares the sum with the observation slab's volume. Refinement addresses the approximation of moving score boundaries; it is separate from closure. Coordinate and geometric volume measures are named separately when the spatial determinant is nonunit. The physical-time slider clips intersected faces and edges at its selected time, retaining their visible portions; it does not hide a whole face because one vertex lies later. Camera rotation and vertical display scaling leave the clipping time and native volumes unchanged.
:::

### V-12 — Maintain the mesh and count its activity

:::{div} feynman-prose
**Where and why.** Place at `sec-dynamic-delaunay-algorithm`. The goal is to make a mesh update measurable: which unique edges changed, how many cells were incident to those changes, and how much geometric work was performed?

**Display and controls.** Use a short sequence of recorded point clouds and native triangulation queries. Display current adjacency, insertion/update diagnostics and edge changes. Offer 16–128 walkers and the metric anisotropy control. Keep the clipped Voronoi picture beside the triangulation diagnostics so their domains and degeneracies can be compared.

**Experiment.** Observe a quiet step, then a step with several accepted copies. Compare the count of changed unique interfaces with the number of affected cells. Repeat from the same archive to verify deterministic topology. Increase the population while keeping the fixture and requested operation consistent before interpreting runtime or operation-count scaling.

**Prediction and measurement.** One interface is shared by two cells, so those counts answer different questions. Native Spade triangulations compare maintained vertex removal/insertion updates with an independent full reconstruction at every frame. The output reports operation counts, slot-search visits, unique changed edges, incident changes, Euler counts and adjacency disagreements. Robust predicates and deterministic duplicate handling make the comparison reproducible. These are the measured maintenance operations; they are not an invented edge-flip count or a complexity exponent inferred from one run.
:::

### V-13 — Smooth fields within a fixed stratum

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-regularity`. A smooth conditional field is obtained after declaring which discrete choices are held fixed.

**Display and controls.** Reuse the exact jet and query display from V-05, including its explicitly declared conditional teaching profile on archived coordinates. The global scale floor is 0.1, the logistic map is `2/(1+exp(-z))+10^-6`, both channel powers are one, and distances use positions only. Show fitness along the query path and its Hessian. Expose distance and metric regularizers and a stratum selector: recorded, new companions, or half alive. The selected alive mask and companions remain fixed during each sweep.

**Experiment.** Move the query through the location of its frozen companion with a positive distance floor. Compare two different floors. Then select new companions or half alive and compare the resulting field. These changes create a new conditional stratum rather than another point on the previous smooth curve.

**Prediction and measurement.** The regularized distance and supported smooth fitness pipeline have finite derivatives on the declared stratum. Population means and scales still contribute derivatives. The native API rejects unsupported nonsmooth configurations instead of supplying an unrelated finite-difference Hessian. Metric eigenvalue clipping has its own threshold regularity; a smooth raw fitness Hessian and a clipped matrix function must be labeled separately.
:::

(sec-partv-experiments-continuum)=
## 5. Test sampling, operators and geometric statistics

### V-14 — Normalize the sampled geometry

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-sampling`. When a cloud is denser on one side, an ordinary average estimates an integral against that cloud's law. This experiment supplies the sampling density exactly so the correction can be checked independently.

**Display and controls.** Sample the cube `[-1,1]^3` with known density `p(t,x,y)=(1+c x)/8`. Offer density contrast 0–0.9, 64–16,384 samples and 8–64 independent replicas. Plot corrected and uncorrected replica estimates, their means and standard errors. Display effective sample size and its fraction of the nominal count.

**Experiment.** Start with zero contrast, where all importance weights are equal. Increase the contrast while keeping the integrand fixed. Compare the systematic shift in the uncorrected estimate with the corrected estimate's remaining sampling fluctuation. Increase the sample count, then inspect how strongly unequal weights reduce effective sample size.

**Prediction and measurement.** For `f=1+x²+2y²+t²/2+0.3x`, the volume integral is `52/3`. The uncorrected volume-scaled sample average has expectation `52/3+0.8c`; the importance estimator averages `f/p`. Effective sample size is `(sum w)²/sum w²` with `w=1/p`. The density is a known input here; estimating an unknown swarm density is a different statistical problem.
:::

### V-15 — Build a signed wave-operator kernel

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-kernels`. The temporal and spatial second moments of a wave operator have opposite signs. A signed kernel exposes that requirement directly.

**Display and controls.** Show the kernel coefficients, positive and negative masses, and independent verification of zeroth and second moments. The compact support is `|t|<1`, `r<|t|`, with symmetric past and future contributions. The implemented laboratory-frame basis is `b(t,r)(c0+c1 t²+c2 r²)`, where `b=(1-t²)²(1-r²/t²)²` on the support. Its time cutoff and basis are explicitly frame dependent.

**Experiment.** Compare the positive and negative portions and predict the sign of the temporal second moment. Inspect the calibration matrix and its admitted rank. Apply the resulting operator to temporal and spatial quadratic fields. Compare calibration and verification outputs rather than using the calibration grid itself as the only accuracy check.

**Prediction and measurement.** The required moments are zero mass, temporal second moment −2, and each spatial second moment +2. Odd and cross moments vanish by symmetry. Native code solves the actual three-by-three moment system with pivot/rank checks, using 16-point tensor Gauss–Legendre quadrature, and verifies with 24-point quadrature. This is a concrete admitted kernel for the chapter's conditional operator; it is not advertised as a pure proper-time radial kernel.
:::

### V-16 — Find the useful bandwidth

:::{div} feynman-prose
**Where and why.** Place at `sec-continuum-variance`. A smaller bandwidth reduces approximation bias while admitting fewer nearby samples. The reader should see both effects on the same explicitly manufactured field.

**Display and controls.** Offer 64–16,384 samples, 8–64 independent replicas and starting bandwidth 0.1–0.8. Sweep seven bandwidths down by factors of square-root two. Plot independent quadrature expectation and bias, Monte Carlo mean with empirical and independently predicted standard errors, single-run variance and mean squared error alongside their deterministic predictions, mean support count and zero-support replica fraction. Keep axes with different units separate in the presentation.

**Experiment.** Begin with `f=t²+x²+2y²+0.2t⁴+0.1x⁴`, whose wave operator at the origin is 4. Shrink the bandwidth at fixed sample count. Then increase sample count and repeat. Inspect the finest windows: a set of empty supports can produce zero observed variance while failing to resolve the operator; its zero-support fraction reveals what happened, while the independently integrated variance remains positive and supplies the missing uncertainty scale.

**Prediction and measurement.** The native estimator samples a fixed cube containing all swept supports and computes `sum K(z/eps)(f(z)-f(0))/p(z)/(N eps^5)`. Here spacetime dimension is three. Replicas are independent; bandwidths within a replica reuse the same draws. Deterministic polynomial bias has order two. Empirical MSE is computed directly against 4, while the Monte Carlo mean is compared with the independent finite-bandwidth quadrature expectation. The deterministic variance integrates the square of the actual single-draw estimator and subtracts the squared finite-bandwidth mean, then divides by N. The MSE reference adds squared bias against 4; dividing variance by the replica count gives the ensemble-SEM reference. Since this field has zero gradient at the origin, its first nonzero Taylor term is quadratic, giving the sharper leading variance `1/(N eps^3)`. The chapter's general `1/(N eps^5)` upper bound in three spacetime dimensions remains valid, but is not the sharp rate of this example. Measured replica variance remains a separate curve; no reference curve is fitted to it.
:::

### V-17 — Recorded order and geometric light cones

:::{div} feynman-prose
**Where and why.** Place at `sec-cst-geometric-comparison`. Two relations can be displayed on the same events without assuming they are identical.

**Display and controls.** Overlay actual CST edges/reachability and geometric light-cone comparability. Retain walker, donor-history and ordinary/revival/singleton initial-population controls, and offer light-cone speed 0.25–4. Event details identify the physical timestep, the declared spacetime metric and any purely visual vertical exaggeration.

**Experiment.** Follow a CST path for one slot, then inspect a geometrically comparable pair on different slots. Observe how a copying jump changes spatial positions while CST continues to use the slot relation. Hide ancestry and IA edges before querying the recorded causal order. Changing the camera must not change either result.

**Prediction and measurement.** CST reachability follows strictly increasing recorded steps and uses only CST edges. A flat geometric comparison additionally requires positive physical time difference and spatial distance no larger than the declared light speed times that difference. The BAOAB teaching run uses physical timestep 0.04; a display-height scale is not a substitute clock. The native comparison reports pairs in both relations, CST only and the geometric relation only, on the same retained finite events. CST reachability can traverse the full archive even when the display selects only recent events.
:::

### V-18 — Count events to measure volume

:::{div} feynman-prose
**Where and why.** Place at `sec-faithful-discretization`. The variance of a count depends on how the points were sampled, even when the expected count is the same.

**Display and controls.** Choose Poisson or fixed-count sampling, the expected/total population and independent replicas. Display total and quarter-region counts, their sample means and variances, and the total Fano factor. Means carry independent-replica SEM; sample variances carry the exact uncertainty obtained from the Poisson or binomial fourth central moment.

**Experiment.** Use a fixed total N first. Its total variance is exactly zero, while a quarter-region count fluctuates. Switch to a Poisson total with mean N and compare both variances. Increase replica count to stabilize the variance estimate; a single total count does not estimate a variance.

**Prediction and measurement.** A Poisson total has mean and variance N, and independent thinning to a quarter-region gives mean and variance N/4. For fixed N, the region count is binomial, with mean N/4 and variance 3N/16. The Poisson sampler uses exponential waiting times to avoid numerical underflow at large means. This reference count law is named explicitly; it is not inferred from the fixed-size interacting engine population.
:::

### V-19 — Estimate dimension from comparable pairs

:::{div} feynman-prose
**Where and why.** Place at `sec-cst-dimension-curvature`. A dimension estimator is useful only after specifying the sampling region and the normalization of its pair count.

**Display and controls.** Draw independent uniform-volume events in a Minkowski Alexandrov interval. Offer spacetime dimensions 2, 3 and 4, 64–512 samples, 8–64 independent replicas and density contrast (zero by default). The sample ceiling keeps every browser control combination inside the native all-pairs budget. Display the event cloud, comparable-pair fraction, inferred dimension and replica variation. Captions identify the selected spacetime dimension and its spatial dimension separately. The scene shows every spatial coordinate in dimensions two and three; for spacetime dimension four it displays a projection of the three spatial coordinates, while comparability still uses all coordinates.

**Experiment.** Start with dimension three and increase sample count. Change the dimension while holding the interval duration and count convention fixed. Compare fluctuations across independent intervals rather than treating all pairs in one interval as independent Bernoulli observations. Then bias the spatial density by the known factor `1+2c x`, and compare the ordinary pair fraction with its pair-weight correction.

**Prediction and measurement.** The ratio is comparable unordered pairs divided by `binomial(N,2)`. Its references are 1/2, 8/35 and 1/10 for spacetime dimensions 2, 3 and 4. Native code samples the interval with its correct time-slice volume density, counts the actual comparable pairs and numerically inverts the gamma-function relation. Standard error for the mean fraction comes from independent replicas. For a nonuniform interval cloud, use `w_i=1/(1+2c x_i)` and the ratio `sum_(i<j) w_i w_j 1_comparable / sum_(i<j) w_i w_j`. This self-normalized pair ratio targets the uniform-volume fraction asymptotically; its finite-sample ratio bias is distinct from replica standard error. The unweighted fraction and node effective sample size remain visible. An inverse outside the admitted dimension bracket is unresolved. The inverse series preserves each original replica index when an earlier inverse is unavailable. Coverage metadata reports requested replicas, replicas containing pairs and resolved inversions, so the fraction and dimension views still refer to the same runs.
:::

### V-20 — Curvature, calibration, and amplified noise

:::{div} feynman-prose
**Where and why.** Place at `sec-physical-consequences`, immediately following the chapter's dimension/curvature statistics. The aim is to reveal both the curvature signal and the cost of extracting it from a small relative volume change.

**Display and controls.** Use the product spacetime with metric `-dt²+dr²+S_K(r)² dtheta²`, where the spatial surface has constant curvature K and scalar spacetime curvature is `R=2K`. Offer K from −0.5 to 0.5, interval duration/support scale 0.1–0.8 through the bandwidth control, 64–16,384 samples and 8–64 independent replicas. Show the calibrated kernel curvature estimate with its empirical standard error, the continuum value `R=2K`, the finite-radius quadrature expectation and pointwise bands of 1.96 independently predicted standard errors. Support counts and compact curvature action accompany the comparison. A separate midpoint-union estimator provides an independently sampled comparison on the same product family.

**Experiment.** Start at K=0 and compare the sampled kernel volume with its independently calibrated flat value. Choose positive and negative curvature that were not used for calibration. Reduce the support scale without increasing the sample count and watch the amplified uncertainty. Then increase samples and compare the weighted kernel integral with its finite-radius quadrature prediction. Repeat with the midpoint statistic, whose flat fraction is 1/4.

**Kernel prediction and measurement.** On `J={|u|<1,r<|u|}`, use squared proper distance `s²=u²-r²` and `K_R=(1-s²)^3`. At physical scale epsilon, the product-volume Jacobian is `j=S_K(epsilon r)/(epsilon r)`. Independent flat-domain quadrature gives `flat=187pi/630` and `M_R=-(1/12) integral_J K_R r² dζ=-359pi/41580`. The estimator is `(mean 8*1_J*K_R*j-flat)/(epsilon² M_R)`, using uniform cube samples with known density 1/8. It targets `R=2K` at small support scales. A distinct, higher-order quadrature measures finite-radius bias at the requested K. The compact action multiplies the scalar-curvature estimate by the independently computed geometric volume of that support. Samples in the 3D scene belong to the kernel support. Sampling occurs in a normalized cube and its physical image shrinks with epsilon, keeping support-hit probability `pi/12`. Consequently this protocol has leading variance `1/(N epsilon^4)`, rather than the fixed-density `1/(N epsilon^7)` scale in the three-dimensional theorem. The native calculation supplies an independent variance prediction, `[8 integral_J (K_R j)^2 - (integral_J K_R j)^2]/(N epsilon^4 M_R^2)`, and the corresponding ensemble SEM. Both measured and predicted uncertainty remain available when a finite replica ensemble gives an unusually quiet realization.

**Midpoint comparison.** The interval volume is `4pi integral_0^(tau/2) (tau/2-r) S_K(r) dr`. The union of the two known-midpoint subintervals has relative volume `2V_K(tau/2)/V_K(tau)`. Independent Simpson quadrature at K=±0.001 calibrates its derivative with respect to R. The small-interval coefficient is `3 tau²/2560`. Samples use the actual product-volume density via rejection from a flat interval. Finite-radius bias is measured by quadrature separately from Monte Carlo error. The midpoint statistic and kernel statistic use separate addressed random streams and separate calibrations. Both use the product metric and its actual scalar curvature `R=2K`.
:::

(sec-partv-experiments-runtime)=
## 6. Native contracts, validation and reproducible interpretation

### Rust computation and WASM delivery

:::{div} feynman-prose
The numerical implementation is split by responsibility. `tracking` owns the observational archive and stage/noise payloads. `fractal_set` constructs event relations and interaction cells. `partv_geometry` owns conditional jets, spectral matrix functions, clipped Voronoi geometry, spacetime partitions, triangulation and linear Gaussian references. `partv_analysis` owns the independent sampling and operator experiments. The browser worker calls these Rust computations and renders their returned arrays.

The Part V analysis API is a serde request/response boundary. A request specifies `kind`, `seed`, `samples`, `replicas`, `bandwidth`, `curvature`, `density_contrast`, `phase`, `spacetime_dimension` and `count_model`; omitted fields use documented defaults. Kinds are `spin2`, `transport`, `kernel`, `manufactured`, `integration`, `counts`, `dimension` and `curvature`. Responses contain named metrics with optional reference and standard error, named x/y series, point arrays and model metadata. Missing uncertainty is represented as absent, not numerical zero.

The native analysis defaults are seed 7, 256 samples, 32 replicas, bandwidth 0.4, curvature 0.2, contrast 0 (the integration card explicitly selects 0.5), phase 0.7, dimension three and Poisson counts. Independent-reference requests are bounded by sample and replica capacities; dimension requests additionally bound all-pairs work. Product-curvature analysis is restricted to dimension three and identifies its conditional fixed-count volume samples explicitly. Unsupported dimensions and invalid configurations return a capability/configuration result.

Geometry has a separate tagged request because its inputs include arrays of actual coordinates, metric matrices and ordered frames. The supported smooth derivative family uses global regularized statistics, logistic channels, one declared same-frame distance companion and analytical objectives. The objective potential driving B steps and the scalar fitness whose Hessian defines an adaptive metric have separate provider identities. No interface silently turns one into the other.
:::

### Tracking identity, stages and reconstruction

:::{div} feynman-prose
Recording is opt-in and bounded. It retains each committed microstep even when a WASM call advances several steps. It does not draw additional simulation random numbers. A failed, cancelled or extinct transition does not append a partial graph. An external population replacement starts a new recording epoch rather than pretending that unrelated states are a continuous trajectory.

A donor source retains frame, slot, generation and population version. A recipient generation counts that slot's accepted replacements. The archive stores the alive mask and fitness before cloning, actual companion source tables, accepted/rejected choices, revival status, donor fitness used in the decision and stage-qualified numerical data. Historical donor fitness is stored as used in that decision, since rescoring may differ from its original archived value.

The stage sequence includes pre-clone evaluation, literal copying, transformed offspring, validated post-clone state, BAOAB substages and final evaluation. The validated post-clone state is the source for clone-versus-kinetic geometry comparisons. Boundaries can act between kinetic substages; dead rows do not continue moving. Built-in noise payloads retain the actual transformed innovation, its raw innovation and factor. Custom providers must explicitly supply the extra evaluation metadata required by their reconstruction claim.

CST follows eligible source-time slots; ancestry follows copied material; IG records a sampled same-time interaction; IA points from effect to its recorded source. Historical donor relations carry distinct kinds. Scalar reconstruction uses explicit component addresses and stage anchors. Geometric vector encoding records its Spin(2) convention. No missing historical component is reconstructed from a current slot with the same integer label.

Checkpoint version three includes the optional archive. Version-two migration starts a labeled anchor at the restored state. That migration preserves numerical continuation while identifying the earlier recording coverage as unavailable. Replay tests compare event identities and numerical arrays within the selected execution profile.
:::

### What is checked, and what a residual measures

:::{div} feynman-prose
The independent analysis suite contains twenty-one passing native tests. They cover every reference mode, seeded repeatability, Spin(2) roundtrip and lifted rotation, U(1)/SU(2) gauge identities, independent signed-kernel moment verification, second-order polynomial bias, known-density integration, effective sample size, Poisson versus fixed-count laws, the correctly normalized Myrheim–Meyer references, and independently calibrated product-curvature volume and kernel moments. Held-out positive and negative curvatures validate the kernel normalization, and nonuniform pair sampling exercises the pair-weight correction. The manufactured-field tests also verify that increasing N changes the measured variance and that displayed MSE equals the direct replica average of squared error.

The tests distinguish three kinds of comparison. Algebraic identities use tight floating-point residuals or exact integer topology. Deterministic discretizations compare independent resolutions or quadrature rules. Statistical comparisons use independently replicated estimates and their measured standard errors. For example, each pair inside one Alexandrov interval shares sampled events with other pairs; independent interval replicas provide the uncertainty unit.

Tracking and geometry acceptance includes clone swaps and repeated donors, revival and historical sources, replay and failed-step atomicity, malformed archive validation, exact duplicate sites, co-circular/collinear geometry, partition closure, stationary covariance residuals and strict/clipped metric behavior. Integrated browser acceptance confirms that all twenty Part V controls produce the native request intended by their labels, boundary or capacity outcomes remain legible, and displayed scientific units match the calculation.

A seed-7 native reference audit with 256 samples and 32 replicas gives a useful reading example. The dimension fraction was `0.229201 ± 0.002680` (one standard error), against `8/35=0.228571`; the inverted dimension was `2.99660`. Known-density integration at contrast 0.5 returned `17.4182 ± 0.0799`, against `17.3333`. At bandwidth 0.4, the deterministic manufactured-field operator was `3.95953846`, approaching its zero-bandwidth value 4. The Monte Carlo mean was `0.8448 ± 4.1571`, with only 4.5 support points per replica on average: the uncertainty is visible because the local sample is small.

The curvature audit separates deterministic calibration from sampling just as sharply. At K=0.2 and support scale 0.4, the kernel quadrature estimate was `0.399659` against `R=0.4`; the sampled estimate was `-4.1384 ± 5.0506`. The midpoint quadrature estimate was `0.400065`, while its sampled estimate was `31.25 ± 28.47`. These are concrete examples for the sample-size controls: both calibrated finite-radius references are close to the target, and the displayed replica uncertainty reveals how much the corresponding stochastic estimates still fluctuate.

The audited implementation passes 125 Rust workspace tests, 111 JavaScript lecture checks (84 existing and 27 Part V), and 98 documentation tests. The JavaScript coverage includes control endpoints, batched checkpoint behavior, partial scene clipping, declared fitness profiles and finite-sample uncertainty comparisons. The updated compiled build passes browser initialization, stepping, reset and control checks for all 62 lecture demos, together with dedicated scene, import, shared-archive and mobile checks. The standalone WASM contract suite also passes.

The audited incremental Sphinx build succeeds with four warnings: failed network lookups for two DOIs and their dependent tooltip-rendering warnings. These concern external reference retrieval rather than the mathematical content. The initial published-site checks covered all 62 embeds, Expert Mode, lazy loading and project prefixes; those checks remain identified as initial-build results. The audited Rust workspace passes Clippy for all targets with warnings denied, and its WASM release build passes. The initial WASM WebGPU configuration check also passed. Adaptive sampled-fitness runs retain their explicit CPU/WASM f64 execution requirement; a WebGPU build check does not imply that provider supports WebGPU execution.

A deterministic reference agreement, an actual engine measurement and a Monte Carlo ensemble estimate retain their own labels in the published cards. This lets readers judge the numerical comparison directly from the experiment they have just run.
:::

(sec-partv-experiments-audit)=
## 7. Audit findings and corrected comparisons

:::{div} feynman-prose
The audit starts with quantities that can be checked independently. A partition must account for the observation volume exactly once. A covariance measurement must use the same normalization and elapsed time as its prediction. An estimator's uncertainty must follow the law that actually generated its samples. These checks found errors in geometry, numerical representation, data presentation and uncertainty reporting. The conditional mathematical predictions survived the independent calculations; none of the corrections below requires replacing a valid theorem or changing its upper bound to fit a plotted curve.
:::

### Positive-volume ties and partial spacetime slices

:::{div} feynman-prose
The original spacetime clipping code admitted equality on both sides of every candidate comparison. This is harmless on a shared face, which has zero three-volume. It fails when two interpolated score fields agree throughout a tetrahedron: both candidates then claim its entire positive volume. A concrete regression starts two sites at `(0,0)`, moves them to `(-0.5,0.5)` and `(0.5,-0.5)`, and uses the square `[-1,1]^2` for one unit of time. The old computed total was `5.333333`, although the observation slab has volume 4. The corrected code recognizes equality of the affine fields throughout the tetrahedron and gives ownership to the smaller slot index. The total is now 4. This corrects the discrete partition; the lecture's requirement that cells partition the slab was the independent check that exposed the error.

Closure and geometric accuracy answer different questions. Once closure is correct, individual moving-cell volumes should still improve under refinement. In the moving-site regression, the sum of absolute cell-volume errors at resolutions 1, 2, 4 and 8 is `1.92537`, `1.23938`, `0.357427` and `0.0902453`. The total volume remains 4 throughout. A zero closure residual therefore verifies complete accounting, while the refinement sequence measures how well the separate cell shapes approximate the moving boundaries.

The viewer also discarded whole faces or edges when part of them extended beyond the time slider. That made an intermediate slice appear to lose geometry before its selected physical time. It now intersects crossing faces and edges with the selected time plane and displays the portion on the visible side. The displayed cut therefore corresponds to a physical time, while rotation and vertical exaggeration remain view controls.
:::

### Metric conditioning, source connections and declared fitness

:::{div} feynman-prose
A mathematically positive eigenvalue can disappear if it is computed as the difference of nearly equal large floating-point numbers. For `diag(10^20,1)`, the old subtraction-based eigenvalue calculation could return zero for the second eigenvalue, falsely rejecting a strictly positive metric or distorting its inverse. The corrected symmetric two-by-two calculation uses scaled, stable formulas and preserves the diagonal axes. The regression recovers eigenvalues `10^20` and 1 and verifies the associated matrix functions. This is a numerical correction to the implementation of the positive-definite metric, rather than a change to the metric condition in the theory.

The variable-metric graph had a separate inconsistency: ordinary graph edges used segment-midpoint metric quadrature, but a source away from a grid vertex attached to the graph using the endpoint metric. The discrepancy is measurable in one dimension. For `g_xx(x)=1+8x`, a source at `x=0.5` and an endpoint at `x=1`, the exact segment length is `1.318305`. Endpoint evaluation returned 1.5. Using the same midpoint quadrature as the graph returns `1.322876`. The finite segment still has quadrature error, but the source connection now follows the same rule as the rest of the graph and participates in refinement consistently.

V-05 and V-13 also needed a more exact statement of what was differentiated. Recorded coordinates and companions were inputs to a separately specified smooth teaching profile. They did not make every numerical fitness parameter identical to the run that produced those coordinates. The cards and this document now state the conditional profile explicitly: global standardizer floor 0.1, logistic channels `2/(1+exp(-z))+10^-6`, unit powers, a quadratic objective and position-only distance. The Hessian is the exact Hessian of that declared conditional field. V-06 separately tests the metric recorded at the actual adaptive O stage.
:::

### Harmonic transients and the variance denominator

:::{div} feynman-prose
V-08 originally divided the centered squared positions by N and compared the result with a population covariance. For independent walkers, that measured quantity has expectation `(N-1)/N` times the population variance. At N=32 this creates a systematic 3.125 percent deficit before any dynamics is considered. The corrected card uses the unbiased denominator `N-1`.

The initial population also has a transient. It starts from independent uniform positions and zero velocity, so its early covariance should be compared with the propagated initial covariance, rather than either stationary Lyapunov solution. The native reference now propagates the full discrete transition and its innovation covariance. It also propagates the uniform distribution's fourth cumulant, which contributes to the exact uncertainty of the sample variance. A Gaussian approximation would discard that contribution precisely where the initial distribution matters most.

The harmonic regression uses stiffness `diag(2,3)`, metric `diag(3,1)`, temperature 0.4, friction 1, timestep 0.04, independent initial positions uniform on `[-1.8,1.8]` in each coordinate and initial velocity zero. An independent ensemble of 512 runs with 32 walkers at time 0.96 gave mean position sample variance `0.208685`, against the exact transient prediction `0.207616`. The standard deviation of the sample variance across runs was `0.039841`, against its prediction `0.039810`. These are uncertainties of one run's sample variance; the SEM of the 512-run mean is smaller by the square root of 512. The corrected card shows the exact transient and its pointwise uncertainty bands, with the stationary references retained for the long-time comparison.
:::

### Empty neighborhoods, variance rates and original replica identities

:::{div} feynman-prose
In V-16, small support and finite sampling can make every observed replica return zero. An empirical variance calculation then also returns zero. Treating that as a resolved uncertainty interval would suggest high precision exactly when no local information was collected. The audit adds an independent integration of the square of the single-draw estimator. Its variance, MSE and SEM remain positive even when all observed supports are empty. The empirical curves and zero-support fraction remain visible, so the student can see both the realized experiment and its sampling risk.

For the chosen manufactured field, the gradient at the origin is zero. Its leading difference is quadratic, and the sharp variance scale is therefore `1/(N epsilon^3)` in three spacetime dimensions. The chapter's more general `1/(N epsilon^5)` upper bound allows a nonzero gradient and remains valid. At bandwidth 1 with 128 replicas, N=1024 gave observed single-run variance `6.61668` against independently integrated variance `7.52143`; N=4096 gave `1.57694` against `1.88036`. The reference decreases exactly fourfold when N increases fourfold. The corresponding ensemble means were `3.52184` and `3.56649`, against finite-bandwidth expectation `3.74712`, with predicted SEMs `0.242407` and `0.121204`. The difference between `3.74712` and the limiting wave operator 4 is deterministic bias, already measured by the separate bandwidth plot.

:::

:::{div} feynman-prose
V-20 has a different sampling protocol. Its physical cube shrinks with epsilon, so the probability of entering the double cone stays `pi/12`. Dividing a sampled volume correction by `epsilon^2` produces variance proportional to `1/(N epsilon^4)`. At zero curvature, N=4096 and 128 replicas, epsilon 1 gave observed variance `1.17599` against exact variance `1.33592`. Epsilon 0.5 gave `18.81584` against `21.37468`: the predicted factor is sixteen. At epsilon 1 the mean estimated scalar curvature was `0.047958`, with predicted ensemble SEM `0.102161`, consistent with the flat value zero. The fixed-density theorem has an additional shrinking-support factor and its `1/(N epsilon^7)` upper bound describes that different experiment.

:::

:::{div} feynman-prose
V-19's unresolved inverse handling contained a data-identification bug. After skipping a replica whose ordering fraction could not be inverted within the admitted dimension interval, the old dimension series renumbered the later successful inversions. Its plotted points then referred to different runs from the fraction series. The corrected series retains original replica indices and reports coverage. Dimension-dependent captions also distinguish spacetime dimensions 2, 3 and 4 from their spatial coordinates and identify the projection used in dimension four. The calibrated ordering fractions themselves remain `1/2`, `8/35` and `1/10`.

The count card now supplies uncertainty for its comparisons as well. Mean error bars use independent-replica SEM. Error bars on the unbiased sample variance use the Poisson or binomial fourth central moment, allowing a finite ensemble's variance discrepancy to be assessed on its own statistical scale. For example, the compiled seed-7 fixed-count check with N=1024 and 128 replicas gives region sample variance `230.378`, against the binomial value 192. Its independently predicted uncertainty is `24.0904`, so the visible 20 percent excess is approximately 1.59 standard deviations of the sample-variance estimator, rather than a changed binomial law.
:::

### Larger-sample checks through the compiled browser API

:::{div} feynman-prose
The following seed-7 results come from the actual WASM analysis entry point. Integration, manufactured-operator and curvature rows use 16,384 samples in each of 64 independent replicas; dimension rows use mean count 512 and 32 independent Poisson interval replicas. The manufactured and curvature bandwidth is 0.8. This checks the compiled numerical path as well as the independent native fixtures.

For the operator and curvature rows, compare the sampled mean first with the finite-bandwidth expectation. The difference between that expectation and the continuum value is the deterministic bias; the difference between the sampled mean and the expectation is sampling variation. Holding bandwidth fixed while increasing the sample count reduces the second difference without removing the first.
:::

:::{div} feynman-added
| Experiment | Sampled mean | Exact or finite-bandwidth expectation | Ensemble SEM |
|---|---:|---:|---:|
| Density-corrected integral, contrast 0.9 | 17.32318 | 17.33333 | 0.014864 |
| Ordering fraction, spacetime dimension 2 | 0.500993 | 0.500000 | 0.002661 |
| Ordering fraction, spacetime dimension 3 | 0.229401 | 0.228571 | 0.002307 |
| Ordering fraction, spacetime dimension 4 | 0.100073 | 0.100000 | 0.001793 |
| Manufactured wave operator | 3.75612 | 3.83815 | 0.122403 |
| Kernel scalar curvature, K=−0.5 | −0.93987 | −1.00858 | 0.115092 predicted |
| Kernel scalar curvature, K=0 | 0.06990 | 0 | 0.112873 predicted |
| Kernel scalar curvature, K=0.5 | 1.06256 | 0.991501 | 0.110721 predicted |
:::

:::{div} feynman-prose
The corresponding inferred spacetime dimensions are `1.99735`, `2.99552` and `3.99913`. Their fractions are each within one ensemble SEM of the calibration. The manufactured field has continuum wave operator 4, so its finite-bandwidth bias here is approximately −0.161846. Its sampled mean differs from the finite-bandwidth expectation by about −0.08204, smaller than one measured SEM. For curvature, the continuum values are −1, 0 and 1; the finite-bandwidth biases are about −0.00858, 0 and −0.00850. Each sampled curvature mean is within one independently predicted ensemble SEM of its finite-bandwidth expectation. The card displays all three quantities separately so that visible bias and visible random fluctuation do not share one unexplained residual.
:::

### Vector reconstruction, edge attributes and archive integrity

:::{div} feynman-prose
The Fractal Set Spin(2) codec originally formed a small component by subtracting the vector's x coordinate from its norm. Near the positive x axis those numbers can round to the same value: for `(1,10^-9)`, the transverse component was lost. The corrected codec computes the larger spinor component stably and obtains the smaller one from the product identity `2uv=y`. It therefore retains the transverse information needed to reconstruct the vector. The independent analysis codec already used this stable construction; the archive codec now receives its own near-axis regressions.

The graph audit also found that available displacement attributes were absent from some instantaneous-interaction and historical-source relations. The corrected construction attaches source-minus-recipient position displacement and its Spin(2) encoding when the actual recorded source is available. IA relations retain their separate displacement semantics. Historical geometry uses the historical source coordinates, rather than substituting the current occupant of the donor slot. Missing coverage remains an explicit absence.

Archive import previously accepted a missing validated `post_clone` stage, duplicate stage names, altered intermediate generations, a historical source version inconsistent with its recorded population, and nonfinite donor fitness. These cases could change reconstruction while leaving many individual arrays plausible. Import now requires mandatory stages in order, unique stage labels, monotone versions, consistent incarnation counters, exact covered source version/generation/eligibility and positive finite donor scores. At the anchor-to-first-step and final-to-next-step boundaries, the same event identity and observation version cannot carry contradictory coordinates, field schemas or opaque state. Recipient generations must agree and versions cannot decrease. A declared extraction that advances the observation version remains valid; refreshed reward provenance or validity is not confused with a changed coordinate record. Together these checks make reconstruction a check of a coherent record, including which historical source each selected relation actually identifies.
:::
