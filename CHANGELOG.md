# Changelog

- Add opt-in Optimization Lab covariance overlays for CMA-ES/BIPOP and same-swarm Fractal estimators, learned field/drift arrows, source-captured movement arrows, 2D/3D and higher-dimensional projections, and geometry-aware recording replay.

- Add live fractal boundary handling choices: no repair, periodic wrapping, and libcmaes boundary repair. Apply the shared mapping across all six adapters, defer planner changes safely, and retain legacy periodic settings and recordings.

- Restore complete valid elite rows after every Wave movement step so the next clone always has donors, including when all proposals leave bounds. Keep evaluations and recorded movement unchanged; publish corrected population metrics and preserve older Optimization recording imports.

- Add a 20D population benchmark with million-evaluation caps, known-minimum target stopping, resumable results, and explicit native precision checks; reject FP64 fractal runs while their internal state remains FP32.

- Add experimental cloning-guided perturbations across all six fractal optimizers: local selected-population geometry, bounded fitness drift, live controls, Euclidean kick/direct modes, and planner-safe model freezing. Preserve existing perturbations and older recording imports.

- Replace adaptive fractal path-based scalar growth with bounded objective-percentile scales. Add live minimum/maximum controls, capped broad moves, normalized donor-difference moves, and focused rounds that respect user bounds; retain covariance geometry across scale edits and ignore legacy archived scale multipliers.

- Add experimental adaptive fractal exploration across the six swarm optimizers, Euclidean kick/direct modes, live multi-run controls, basin archives, and a reproducible benchmark pilot. Preserve legacy seeded tests; record refinement provenance and document experimental limitations.

- Add live Optimization Lab tuning for perturbations, swarm parameters, populations, and evaluation budgets, preserving run state and recording changes at application boundaries.

- Run Arcade trajectory playback in an independent emulator while sampling continues, with fixed recordings, isolated failures, and a bounded share of the combined memory budget.

- Add an Arcade trajectory player with best/selected walker ancestry, scrubbing, playback controls, direct Graph snapshot rendering, and Wave/planner replay from the initial state.

- Add Arcade worker selection (Auto or 1–20), persisted 1–8 GiB combined WebAssembly budgets, growing wasm32 heaps, direct donor stepping, bounded Graph history, and runtime cleanup on reset.

- Add an optional shared-prefix threshold to Graph in the LLM, Arcade, and Optimization labs. Frozen ancestry is archived outside the active population, shown in green, and retained for trajectory reconstruction.

- Add generated fixed-target Xent games to LLM Lab, with opposite surprise objectives, target likelihood scoring during Fractal sampling, best-prefix retention, paired token-budget benchmarks, portable game settings and offline scoring evidence.

- Correct the algorithmic spectroscopy against the eighty-defect audit of the reference pipeline and raise `SPECTROSCOPY_VERSION` to 2. A momentum mode now projects the connected element, so an additive convention on the observable no longer changes the rate and mode 0 of both phases is refused; the orientation of a score-directed pair and the column order of a score-ordered triplet are frozen with the element at its source frame and reapplied at every sink, so no rate carries the score decorrelation of the sink frame, and the two arms whose propagator that identity makes a duplicate of another channel's keep only their frame mean; the score gradient divides each companion difference by its own length and is a gradient again; a frame whose weighted score dispersion vanishes has no average and carries the weight zero; the generalized eigenvalue problem follows a fixed vector solved once at a reference lag instead of taking the largest eigenvalue at each lag, and a basis of two channels that measure one observable is refused; a fit under an active singular-value floor reports the rank it actually used as its degrees of freedom; and every analysed channel publishes what its resampling does not cover, including that a delete-one jackknife removes time origins and not observations, so beyond the block its error is a lower bound. The emergent metric's ridge and eigenvalue bounds are multiples of the displacement covariance scale, so the metric scales with the cloud again; an eigenvalue a clamp or a sign repair moved is recorded and refused under a strict policy instead of leaving an indefinite matrix positive; and a tessellation carried over by a schedule that does not refresh every step is stamped with its staleness, declined by the scale gating and checked on decode, so an archive written before the stamp cannot read as fresh.
- Add an algorithmic spectroscopy subsystem to the Rust Algorithmic Gas: meson, vector, baryon, glueball, tensor, Dirac, electroweak, chirality and twistor operators, plus injected custom ones, evaluated on the recorded companion topology of sites, distance pairs, cloning pairs and triplets, with every documented definition kept as an explicit arm. A measurement streams short recorded chunks through an accumulator and retains operator series, source-frozen propagator moments and coverage; a pure, repeatable analysis forms connected correlators, block-resampled errors, effective masses, AIC window scans, multi-exponential fits with channel-agnostic scale-free priors and a prior-dominance diagnostic, joint group fits, GEVP levels, geodesic multiscale copies, a graph smoothing diagnostic, coupling scales and a reference comparison labelled a hypothesis mapping. Every variant can be measured: capabilities derived from the gas configuration make the channels of an unrecorded quantity unavailable with a reason instead of failing the run, an exchange-odd frame mean that cancels exactly on a mutual pairing falls back to the source-frozen propagator, an envelope that never enters a correlator is non-correlatable by type, and the fitted number is reported as the decay rate of the algorithm-time autocorrelation, which is a mass only under a positive self-adjoint transfer representation. Add the resumable spectroscopy session with re-analyzable CBOR evidence, the `gas-spectroscopy` runner and the wasm bindings that share its request, report schema and f64-only CPU precision rule. The subsystem runs end to end: a pipeline test measures Einstein–Hilbert, viscous Euclidean and Euclidean archives and fits rates on them, a session test shows that a longer run needs no more engine memory because one chunk is recorded at a time, a CLI test drives every subcommand, and a parity test checks the operators, the statistics and the window scan against exported fixtures of the Python reference.

- Name the gas variants and register them: the Fractal Gas is the family, the Algorithmic Gas is the engine and never a variant, and the Euclidean, Viscous Euclidean, Einstein–Hilbert, Geometric, Latent Fractal and Environment gases are the registry entries, book-only ones marked unimplemented. Add `algorithmic_gas::variants` with each variant's title, book label, summary, reference instance and configuration constructor, move `GasConfig::euclidean` and `GasConfig::einstein_hilbert` into it behind unchanged public paths and wire names, add `GasConfig::viscous_euclidean` for the Gaussian-kernel viscous force the colour channels need, and add `RunConfig::euclidean`, `RunConfig::variant` and `gas-benchmark --variant NAME` with `--einstein-hilbert` kept as an alias.

- Add the Volume 2 chapter "Variants of the Fractal Gas", which fixes this vocabulary, writes each variant as the twelve components of the step operator, and states variant by variant which theorems of the volume apply and which hypotheses are still owed.

- Add the QFT Simulator page to the Algorithmic Gas section beside the Lab and the Lectures. It configures a variant, streams a recorded run in the compiled engine, and shows coverage, correlators, effective rates, fit diagnostics and the reference comparison in five tabs. JavaScript performs no science: every number comes from the wasm spectroscopy API, undefined values stay gaps, availability reasons and notes are printed verbatim, a rate is never relabelled a mass, and a test scans the page modules for arithmetic. A node test pins the page contract and a Playwright test drives the compiled engine in a real browser, from configuring a variant through streaming a run, re-analyzing it without advancing the gas, exporting the evidence and re-importing it to the identical report.

- Add the Python bridge and fixture tooling for the spectroscopy parity tests: an exporter that builds prepared channel data from seeded synthetic tensors, runs the real Python operators and writes reference cases for a non-involutive topology, a mutual pairing and a masked population, each naming the function and line that produced every value. Correct the archive-to-history converter so that colour phase and force share one velocity through a declared history convention and the cloning scores are the unclipped, ungated scores.

- Require alive cloning companions for every Python Euclidean Gas selection scheme, restrict selection calculations to survivors, fix single-survivor revival in pairing methods, and reject invalid custom donors before cloning.

- Fix Rust elite reinjection after each completed step: elites still move normally, then the updated best-ever bank is copied back into the first population slots with refreshed geometry.
- Add historical elite walkers to the Rust Algorithmic Gas, with transactional retention, checkpoint continuation and recorded reinjection. The Euclidean Gas Lab exposes the elite count and starts fresh runs with two elites; existing configurations retain zero.

- Fix CI checkpoint-version expectations for geometry schema 5 and update mobile browser tests to open Settings and interaction modes through Menu and Tools.

- Give the Control Lab environment most of the mobile screen, with compact run controls and on-demand tools, settings, and replay panels; preserve desktop layout and recording state across responsive changes.

- Review the Rust Algorithmic Gas against its architecture specification. Selection: greedy Gaussian matching takes the specified ascending pivot order, with the random permutation as the explicit `pivot: "random"` law; kernel-weighted rows sampled without replacement are in draw order; an unmatched walker is a topology error instead of a zero diversity measurement; a disabled fitness channel is not evaluated. Recorded-run analyses (clone field balance, VI-02) use the executed acceptance law through `CloneDecision::acceptance_probability`, including the cloning period, and the atomic mean-field reference rejects gated cloning. Tessellation: exactly flat swarms are exactly rank deficient, collinear planar input is an error instead of an empty graph, the finite-difference Hessian uses the unequal-spacing stencil, and periodic wraps cannot land on the upper face. Robustness: checked Part V work budget on 32-bit targets, overflow-free velocity-cap prediction, float population products, archive byte accounting that includes recorded noise geometry, structurally compared jet spaces, browser seeds bounded by exact JavaScript integers. Efficiency: allocation-free pair tiles and local statistics, recording buffers moved into the archive, single-pass jet powers. Lecture sessions recompute their measurements on restore, matched-time step counts do not overshoot exact multiples, and across-run statistics are unbiased or undefined. `gas-benchmark` reports first-step and steady-state timing separately and keeps the committed archive when a step fails.

- Add tessellation geometry to the Rust Algorithmic Gas: planar (Spade) and native exact-predicate 3D Delaunay with coincident-walker and rank-deficient handling, clip-box and periodic domains, dual Voronoi cells, and swappable metric, volume-element, edge-weight, curvature (conformal Laplacian and quadratic fit, Regge deficits in 2D and 3D, Voronoi proxies) and reward-allocation components, bit-identical in serial and Rayon-parallel execution. Add an engine geometry stage with clone-carried observation fields and a stage schedule, graph viscosity with Boris curl rotation, periodic cloning, `GasConfig::einstein_hilbert`, a geometry reward and point start in `RunConfig`, archive recording (`--record`, optional tessellation graph) in `gas-benchmark`, Python reference fixtures and a `RunArchive` to `RunHistory` converter. Checkpoint format 5.

- Add 3D spatial and landscape population views to the Euclidean Gas Lab (surface with contours, slices, companion links, trails, atom viewer) beside the 2D projection, and bring its objectives to parity with the Optimization Lab: nine more classic tensor-graph objectives with analytic gradients and a pure-Rust port of the 24 COCO BBOB functions validated against COCO 2.8.2, served to the browser and the CLI from one Rust catalog.

- Restore the original combined mean-field proof target, retain additive drift offsets, and repair full-slot moment and velocity-weight accounting in the cloning inputs.

- Complete geometric coverage for the averaged Keystone estimate and certify unequal-fitness structural expansion with exact interval bounds and independent full Rust engine comparisons.

- Derive measurement-averaged Keystone cluster pressure and complete signed cloning balances; verify a population-uniform structural-pressure family and independent full-engine measurement averages.

- Repair Keystone donor flux with moving-barycenter identities, signed geometric-cluster bounds, canonical barycenter concentration, and component-average cancellation for revival. Prove bounded transport smoothing for the actual nonlinear BAOAB/cap update and verify the estimates with Rust execution.

- Recover the mean-field proof sources and add reviewed exact geometric-cluster coupling balances, quantitative full-step bias and variance estimates, survival-aware entropy identities, finite-population entropy convergence, and independent Rust conditional-fluctuation checks.

- Derive the canonical Euclidean Gas fixed-step population law and propagation of chaos from frozen weighted sampling and shared component collisions. Implement component Haar rotations, scheduled weighted revival, final position diffusion and smooth capping; add exact collision covariances, independent rooted population integration, Part III diagnostics and independently seeded validation. Repair cloning drift with exact reset bounds and prove finite-population QSD convergence for the actual kernel.

- Add coupled Taylor sensitivity and exact companion-law averaging to IV-07, with production fitness comparisons, adaptive sampled accuracy windows, independent coefficient-resolution checks, and reward-landscape controls.

- Validate complete field-equation recording coverage, preserve legitimate boundary extinction, distinguish field accounting from prediction checks, and test full covariance with exact quadrature, clone-gate enumeration and independent metric-active ensembles.

- Derive the gas field equations from explicit donor, cloning and BAOAB primitives, including spatial sources/fluxes, anisotropic stress, conditional metric evolution and transient memory. Add Rust Fourier-field predictions and interactive clone/thermostat validation with immutable historical sources.

- Execute all 128 Volume II demos through shared native/WASM Rust sessions, with recorded algorithm measurements, configured predictions, validated evidence replay, tracked geometry, continuation protocols, and control stress tests.

- Derive transient field drift/covariance from complete algorithm states; measure historical-source fields, material metric increments, source-aware noise moments and stage balances, with explicit per-experiment provenance and independent validation.
- Add 66 compiled Part VI workbenches with Rust/WASM calculation APIs, native `gas-physics` runs and sweeps, source-aware archive readouts, conditional trace likelihoods, held-out predictions, independent response and metric replicas, formula lookup, and a detailed lecture experiment guide.

- Derive and implement finite-step Algorithmic Gas geometry and balances: packed general-dimensional fitness jets, efficient 3D curvature, exact global-statistics caching and local derivatives, spectral clipping, backend batch contractions, independent checkpoint replicas, thermostat fluctuations, stationary response/Fisher references, and reproducible research sweeps.

- Audit and correct Part V experiments: stabilize metric eigenvalues and Spin(2) reconstruction, prevent overlapping spacetime cells, align geodesic quadrature, preserve archive and replica identities, and add exact transient and sampling-error predictions with independent numerical regressions.

- Add 20 Part V lecture experiments with durable Rust walker archives, stage and lineage provenance, scalar reconstruction, adaptive O-stage geometry, Voronoi and spacetime cells, causal-order comparisons, and independently calibrated continuum statistics. Include archive import/export, interactive scene controls, and 20 computed chapter posters.

- Align Volume II lecture experiments with their theoretical predictions: correct BAOAB pseudocode and alive-conditioned mixtures, use theorem-specific entropy and drift diagnostics, calibrate sampling uncertainty, resolve misleading spectral and density-ratio displays, and exercise survival and population comparisons with independent replicas.

- Add 42 interactive Volume II lecture experiments with a persistent WASM worker, authentic cloning and BAOAB traces, mathematical reference models, seed replay, SVG/JSON exports, and lazy chapter embeds with computed posters. Add validated population fixtures and independent reward/potential controls to the Rust browser API.

- Make GAS (2017) with adaptive local Gaussian proposals the Optimization Lab default, and fix narrow landing-page navigation so the full Pages bundle can deploy.

- Correct Algorithmic Gas distance reduction, local statistics, linear uniform sampling, donor availability, boundary timing, extinction handling and provenance. Validate tensor/checkpoint imports, introduce checkpoint format 2, enforce safe-Rust lints and add resource guards, regression tests, dependency audits and fuzz/Miri checks.

- Add the independent Algorithmic Gas Rust workspace and Euclidean Gas Lab, with modular observations/rewards, separate donor roles, CPU/WASM f32/f64, explicit hybrid GPU adapters, stochastic and Langevin kinetics, anisotropic noise, transactional cloning, and replay checkpoints. Existing Python and C++ engines are unchanged.

- Make beam-style length normalization with α = 0.6 the default LLM generation reward while preserving explicit objectives and legacy recording imports.

- Add a visual FragileTech home page introducing Arcade, Control, Optimization, and LLM labs; preserve Arcade at `arcade.html` and update lab navigation.

- Add pairwise LLM answer ranking with Davidson Elo ratings, uncertainty, adaptive comparisons, independent method audits, held-out validation, optional model/human audits, durable request budgets and recovery, and portable version 2 comparison reports with offline CLI processing.

- Add negative mean Xent as the LLM Lab default, adjustable beam-style length normalization, and optional mean XED maximization/minimization through Together. Share cumulative utility with native Wave/Graph, preserve offline scoring provenance, and report scoring usage separately.

- Move judge grading, scores and trace comparisons into a dedicated LLM Evaluation tab. Add direct Gemini Flash evaluation and resolve its moving alias to a concrete model before selecting an active structured-output provider.

- Default optional LLM benchmark grading to OpenRouter's latest Gemini Flash alias while keeping judge and generation settings independent.

- Reward tandem checkpoint proximity with radius / (radius + mean agent distance), including cleared agents waiting for the team. This positive per-frame score replaces signed checkpoint progress; crossing bonuses and synchronization remain unchanged.

- Add a full-width LLM Benchmark tab with linked archived/retained sampling comparisons, trial-weighted statistics, diversity and work curves, pinned trace inspection, optional independent-model grading, offline comparison reports, and CLI report processing.

- Highlight the shared tandem checkpoint in amber with a crossed-agent count, and flash crossings green in live views and forward replay; seeking and reduced-motion preferences suppress transient feedback.

- Preserve LLM EOS traces while recycling terminal walker slots; stop at the initial-population EOS target or shared generated-token budget and retain partial results at run limits. Add version 3 progress recordings with legacy imports. Fix provider continuation markers and verify three consecutive prose chunks before accepting a route. Default to Qwen 3.5 35B A3B with verified Alibaba partial mode; keep DeepSeek selectable.

- Synchronize tandem checkpoints across agents, with default progress weight 1 and checkpoint bonus 30 normalized by team size. Agents wait for the team before advancing to the next ordered checkpoint.

- Add LLM benchmarks comparing Wave or Graph with independent populations, matched generated-token budgets, and temperature-zero answers. Save repeated trials in browser storage or checkpointed CLI journals, with portable archives, offline processing, and explicit retry of unfinished methods.

- Raise the Tandem flight formation reward default to 50 and expose its full 0–100 range in reward controls and scene validation.

- Add an LLM Analysis tab with recorded generation trees, stored Graph views, metric coloring, independent playback, token and decision inspectors, branch comparison, and PNG export. Capture opt-in pre-cloning diagnostics without changing seeded algorithms; export version 2 recordings while retaining offline version 1 inspection.

- Draw formation links between every scored agent pair, with reusable dashed segments colored by pair quality and a matching legend in both visual styles.

- Add right-drag camera rotation and tilt to the Control Lab, preserving pan, zoom, agent following, and replay; reset controls restore the initial scene view.

- Reward formation flight with the product of pairwise target-distance scores each physics frame. Add validated per-pair distance overrides and editor reference handling; default to formation, squared travel, and collision terms with optional checkpoint rewards.

- Increase the default Control Lab wall-collision penalty to 100 per touching vehicle per physics frame; explicit custom penalties and wall-death settings remain unchanged.

- Add an OpenRouter LLM Lab with shared native Wave/Graph token environments, conditional token likelihood objectives, selectable embedding context, and portable trace recordings. Introduce a shared configurable L2/cosine observation-distance API, retaining L2 defaults for existing labs.

- Unify Arcade, Control Lab, and Optimization Lab navigation as compact controls
  with shared spacing, hover, focus, and current-page states.

- Tighten the Control Lab's default whole-arena camera to the collision boundary,
  with a centered 5% margin that adapts to the viewport and flight projection.

- Make solo and collaborative flight mining playable with upright 24 N rockets, recoverable wall contacts, catch/target-focused rewards, and a 32-action, 6-frame, 4-elite planner preset. Preserve explicit controller tuning when switching scenes.

- Consolidate native Wave, Graph, FMC, Jump Wave, and Euclidean Gas in the application-independent `fractal-gas-web/src/fractal/` library. Arcade, Control Lab, and optimization retain their batch storage and action policies; Lab Jump Wave decisions now run in C++. Reuse numerical scratch and preserve direct packed donor-to-output physics stepping. Standard comparison algorithms and Python/Torch research implementations remain separate.
- Introduce Control Lab checkpoint version 2 with shared population, elite, RNG, history, and incremental planner state; reject previous checkpoints without migration. Advance the optimization replay engine identifier to `fgopt-4`. Stable cumulative-reward elite ties, complete elite metadata, common lifecycle ordering, and terminal bookkeeping can change historical seeded trajectories; replay remains deterministic within the new backend/build.
- Keep Arcade WASM builds from overwriting Lab's standalone module, and allow fractional benchmark targets so the default target does not block applying controller settings.

- Dynamically distribute Control Lab physics futures in small batches across worker threads, reducing idle time when collision workloads differ while preserving planner settings and deterministic results. Keep static scheduling for other backends.

- Speed up collision-heavy flight with conservative wall filtering, reusable geometry, and substep-local resting contacts. Count continuing wall contacts once per substep; old recordings remain viewable, while re-simulated trajectories can differ.

- Dismiss the episode-ended notice on the next click or timeline drag so replay remains unobstructed.

- Add configurable wall-collision penalties and optional whole-run death in every Control Lab environment. Wall penalties default to 2 per touching vehicle per physics frame, including older scenes; shipped presets default to nonlethal walls.

- Add persistent dropped asteroids with inner delivery and outer re-hook release zones.

- Enable retained rocks in Collaborative mining with a longer cooperative planning horizon and delivery/detachment/re-hook regression coverage; the existing toggle still restores respawning.

- Add persistent weighted harvesting hooks, live hook mass controls, and approach/catch/movement rewards.
- Make the default asteroid masses light enough for solo rocket towing at 1× in flight mode.

## Optimization Lab

- Add pinned libcmaes Active and BIPOP-active CMA-ES to native and WASM builds, with bounded double-precision candidates, deterministic restart streams, complete-generation budgets, and recorded optimizer status.

- Enable adaptive local covariance perturbations in GAS (2017), learning from accepted jumps while preserving retry scaling and existing defaults.

- Add local covariance Gaussian perturbations for Wave, FMC, and Wave Jump, with bounded local learning and frozen planner geometry for action replay.

- Add GAS (2017) with objective-adaptive Gaussian jumps, independently switchable tabu memory and bounded L-BFGS-B searches, and shared recording/evaluation-budget support.

- Make history recording opt-in; keep only the live frame by default and enable replay and exports when recording is selected.

- Integrate all 24 COCO BBOB functions from pinned upstream COCO 2.8.2 C source, including shifted and rotated instances, official dimensions, and reference minima.
- Add evaluation budgets, convergence by evaluations, and IOHanalyzer custom CSV export; count optimization queries and finite-difference forces while excluding visualization samples.

- Reuse the shared FMC and Wave Jump planners in Optimization Lab alongside Wave, Graph, and Euclidean Gas; expose planning horizons and the committed position.
- Add extensible perturbation strategies with zero-mean Gaussian noise and configurable standard deviation, plus a uniform strategy; share the controls and recording settings across algorithms.
- Default to minimization and add maximization, with consistent fitness, potential forces, and best-objective metrics. Preserve visual replay of earlier recordings.

- Run browser checks in a pinned Playwright/Mesa image under Xvfb with software WebGL and report startup errors immediately instead of timing out.
- Add a standalone C++/WebAssembly Optimization Lab with Wave, Graph, and core Euclidean Gas, 13 benchmark objectives, full-dimensional 3D views, molecular inspection, and portable replay.
- Share emulator-independent numerical/swarm targets with Control Lab and the arcade engine; add native, WASM, Python parity, and browser checks.
- Correct adjacent-coordinate Rosenbrock evaluation and the general Python kinetic operator's plain BAOAB half-kick branch. The Boris branch retains its existing behavior.


Unreleased
----------

* Restore Python test collection after learning and QFT API changes, repair encoder unpacking, coupling gathers and diagonal diffusion shapes, avoid uninitialized experiment logging, correct finite-difference validity and periodic Voronoi facet indexing, and add a locked CPU Python CI job with headless dashboard tests.

* Wait for the simulation worker's pause acknowledgement before checking that the Lab clock stays stopped in CI, and make diagnostic experiment screenshots opt-in to avoid software WebGL capture timeouts.

* Add FMC and Jump Wave to the NES, Atari and Genesis arcade, with committed gameplay, configurable search horizons, shared-path or full-path execution, and deterministic planner/replay tests.

* Repair Lab CI renderer fixtures and browser regressions for paused startup, staged settings, recording branches and Drive mode; discover all Lab contract tests and run workspace regressions on build/config changes.

* Default new Lab runs to Wave Jump with 128 walkers and a horizon of 64.

* Show vehicle resource loads as growing cargo piles with intake and unloading transfers, visible held/picked/delivered counts and loading status; remove permanently full harvester cargo while preserving animation-off updates and shared rendering budgets.

* Strengthen Lab world silhouettes, reactor and docking rings, scenery housings and refinery loading machinery; separate painted panels, tinted glass and rough mineral surfaces while retaining existing asset budgets and animation attachments.

* Lower the default Collaborative mining rock mass to 0.24 kg so a single upright rocket can lift it at normal thrust in gravity mode.

* Refine Lab world props and both refineries with shaped structural housings, clearer capture hardware, fractured mineral surfaces, worn industrial finishes and refreshed Blender sources, within the preceding per-asset rendering budgets.

* Explain gravity-mode rock lifting with a thrust-versus-weight readout and a lighter-flight-load control that respects staged thrust settings; verify floor takeoff and heavy-load limits in the physics engine.

* Redesign the Lab with a persistent simulation workspace, guided task selection, staged settings, continuous manual driving, automatic run preservation, replay branches, contextual inspection, and structured controller comparisons.

* Refine both Lab vehicle collections with stronger canopy, armor, seating and cab profiles, plus aged metal and clearer panel finishes using existing geometry and texture budgets.

* Apply Lab rewards mid-run without resetting the world, camera, or recording, and retain reward-change boundaries for replay and saved runs.

* Allow Rockets, Drones, Karts and Harvesters in every Lab environment and racing track, preserving world setup when switching the fleet and remembering each environment’s selection during the session.

* Add static actuator cues that remain visible with animations off, optional signed action guides relative to configured limits, and catalog-based Workshop sliders with neutral/max presets and independent-thruster previews.

* Add per-agent-type action multiplier controls from 0× to 10× for independently scaling every built-in actuator degree of freedom, including rocket thrust and torque.

* Preserve per-agent actuator multipliers when applying mining rock settings, so a 10× rocket thrust and 0.01× rock mass take effect together.

* Add auto-detected and explicitly configurable rocket/drone flight mode with downward gravity, a sidebar checkbox, and side-on/overhead camera views.

* Add optional vehicle and world animations with a persistent reduced-motion-aware switch, paused idle motion, and Workshop preview controls.

* Make shared-prefix execution the default for new Wave Jump searches while preserving explicit full-path settings and legacy in-flight checkpoints.

* Derive the exact commutation criterion for the existing even quantum regional algebras, including low-dimensional vacuum sectors and the equivalent native descriptor-independence test in the nondegenerate case.
* Verify the recorded Wilson multiplication commutator by expanding its original holonomy traces, weights, and full-face masks before extending to bounded cylinders and local von Neumann algebras.
* Remove the failed pair-covariance counterexample from the HK argument and explicitly track multiplication locality and CAR commutators through their respective existing record transports.

* Add optional Wave Jump shared-prefix execution with bounded search extension and single-action fallback.

* Evaluate native gauge locality coefficients by signed history cancellation and full-face tuple products, retaining original weights, masks, and selected laws.

* Reward hooked rock travel by distinct-rock distance, with an adjustable weight and no credit for respawn teleportation or duplicate tow hooks.

* Extend the Lab rock-weight control down to 0.01×, allowing rocks to be one hundred times lighter while retaining the 10× upper bound.

* Expose mining hook stiffness, allow rock sizes down to 0.1×, and prevent false target-progress reward when tow hooks break.

* Mark unapplied Lab reward-slider edits explicitly and verify that applying a zero movement reward preserves the running world.

* Prefer alive final paths in Wave Jump; when every final walker is dead, execute only the first action of the highest-scoring path before replanning.

* Remove alternate equilibrium/OS/Galerkin dynamics and independent field-model additions from the algorithmic QFT chapters; retain native recorded operators, laws, sources, and bounds, and update dependent documentation.

* Express the smaller physical-gauge descriptor's projected source covariance through its native polynomial Gram matrices and source responses, with projection-error and occupation bounds.

* Prove exact disjoint-source covariance cancellation from the native likelihood and occupation identities, to all derivative orders, retaining the descriptor projection and survival normalization formulas.

* Enable left-drag camera panning in the Lab, preserve click selection and scene editing, and add Reset view.

* Derive the inherited-history covariance of regional readouts as exact native-kernel integrals, with update contributions and predictable-variance bounds retaining survival conditioning; carry both covariance terms into the full even-observable commutator bound.

* Add Wave Jump to the Lab: execute the highest-reward final FMC trajectory between searches, with pause/resume, trajectory checkpoints and experiment comparisons.

* Expose Lab reward and diversity sliders with a shared Apply settings action that preserves the world; enable the mean per-frame squared-distance vehicle bonus by default.

* Bring Lab vehicle, world and refinery shapes and surfaces closer to their concept sheets; freeze the preceding 24 runtime packs as geometry, draw-batch, texture-memory and download-size regression budgets.

* Refine all 94 styled Lab assets with distinct mechanical assemblies, exposed refinery conveyors, mineral chambers and capture claws; reduce vehicle triangle counts within their existing rendering budgets, preserve shared collision envelopes, and add complete background regeneration commands.

* Correct the locality analysis to use squashing before both position transports; derive the exact capped kinetic support and remove the mismatched equilibrium-density obstruction.

* Fix Lab native-control help markup and label targeting in Firefox; add Firefox and Chromium dropdown, checkbox, button, and tooltip regression coverage.

* Fix Lab builds by exporting the geometry helpers used by mining controls; default to 8 worker threads, 12 action frames, and horizon 32.

* Add mining rock-size controls and a 1–20 rock count for asteroid harvesting, with clear placement and replenishment after delivery.

* Add adjustable Lab reward terms and an optional per-frame squared-distance vehicle reward; applying weights preserves the live world, replans, and starts a new recording.

* Set Lab defaults to 8 worker threads, 12 action frames, and horizon 32.

* Allow choosing 1–128 vehicles in every Lab environment, with clear spawn positions and preserved cargo, obstacles, and automatic tow hooks.

* Refine all eight Lab vehicle designs with concept-inspired armor panels, baked fasteners and wear, aged metals, darker glazing, and adjusted drone and hopper shapes; enforce their previous geometry, draw-batch, texture-memory and download budgets.

* Lighten the Lab's playable arena floors and racing tracks with matte slate and warm stone colors, improving agent and planning-trace visibility against dark out-of-bounds areas in both styles.

* Respawn delivered mining rocks at seeded random clear positions throughout the playable map, avoiding walls, holes, active bodies, and delivery bases while preserving exact replay.

* Give each control-browser suite its own CI timeout and preview-server lifecycle, retain server diagnostics, and avoid repeating racing physics and every screenshot on tablet.

* Restore missing racing-guide screenshots and fix the Pages smoke test for the five-vehicle Ants default; use smaller planning budgets and retain large-fleet coverage in native tests.

* Refine vehicle silhouettes against the concept sheets with rounded pressure hulls, integrated drone armor, detailed cockpit and intake assemblies, corrected outward normals, and shared brushed-metal maps within the existing LOD budgets.

* Default Ants & Drops to five vehicles and add five-drop cargo tanks to harvesters and drones, full-load bonuses, gradual rewarded refinery unloading, replayable cargo state, and concept-based steampunk/futuristic Blender refineries.

* Add illustrated tutorials for all six Lab tasks and six racing circuits, a complete scene-editor course and JSON reference, downloadable workshop scenes, and reproducible screenshot and example checks. Let editor pointer actions pass through the status overlay so tether placement can reach the second body.

* Refine both Blender asset styles with vehicle suspension and machinery, detailed dock assemblies, corrected ore normals and UVs, branching mineral textures, and zoom-dependent ore detail while preserving native collisions.

* Give Collaborative mining one high-mass, high-drag rock that a solo rocket can move slowly and two can haul faster; immediately replenish it after every delivery.

* Fix Pages deployment by isolating documentation route tests from package-wide fixtures, so the docs-only build does not require installing the simulation package.

* Build 42 world asset types in both futuristic and steampunk styles with shared collision envelopes, distinct machinery, packed Blender/GLB sources, instanced scenery and pickups, replayable effects, and an expanded asset workshop.

* Add five video-traced historical kart circuits alongside Violet Circuit, under one Racing task with a difficulty-ordered track selector, geometry previews, source references, validated obstacles and checkpoints, and two-lap driving and replay regression tests.

* Add sixteen futuristic and steampunk world-asset concept sheets, a paired visual library with original downloads and generation prompts, and narrow-screen style-control checks.

* Add configurable Ants & Drops fleets of 1–128 harvesters or drones, with clear starting positions and verified continuous seeded drop replenishment and replay.

* Add eight concept-based Blender vehicle models with detailed and crowd LODs, matching scene props, a persistent futuristic/steampunk lab selector, and a model workshop with Blender/GLB downloads; preserve live state and replay while switching styles.

* Automatically install or activate Emscripten for `make control-web`, isolate local documentation dependencies, and serve the portal, books, and simulator together with working local routes and a fast `make docs-serve` preview.

* Split the documentation into a `/docs/` portal, the two-volume lectures under `/docs/theory/`, and an independently searchable control-lab guide under `/docs/lab/`, with scoped sidebars and compatibility redirects.

* Link the lab and arcade navigation to the published laboratory user guide, keep the guide accessible on small screens, and verify all seven guide pages in the Pages deployment bundle.

* Run control-engine Python integration checks in an isolated environment with the existing gas's complete import dependencies, and stop the browser test server before CI cleanup to avoid a package-cache lock hang.

* Build and verify a separate GitHub Pages lab artifact with both WebAssembly engines, test project-path service-worker isolation and serial fallback, restore missing book navigation sources, and fix control CI build/dependency failures.

* Add a six-page control-laboratory user guide covering setup, every control, scene editing, replay and storage, experiments, diagnostics, and engine extension contracts, with browser screenshots and worked examples.

* Support up to 64 control-engine execution threads in the browser and Python, prewarm the selected browser pool, and synchronize empty batch partitions safely.

* Align the control lab with the Fragile Tech documentation identity and add the Violet Circuit kart environment, ordered lap checkpoints, replayable lap progress, and extensible environment renderers and score readouts.

* Add iCEM and MPPI control-lab baselines with generic parameter controls, batched rollouts, warm starts, exact search checkpoints, and measured simulator-work comparisons.

* Add generic actuator, world-extension and controller registries; kart and independent-thruster dynamics; exact planner checkpoints; reproducible benchmark suites and fork comparisons; compressed persistent replay with events; multi-selection, templates and unit-bearing editor controls; and native/browser performance and physics inspection.

* Add reusable agent archetypes, distinct kart/rocket/drone models, declarative visual kits, mixed crowd instancing, and complete world-motion recording with playback, exact seeking and continuation in the control laboratory.

* Add a custom C++ continuous-control engine with parallel world batches, compact complete-state snapshots, native Python bindings, Wave/FMC planning, optional cloning-aware exploration trees, and a browser laboratory with five editable cyberpunk scenarios and original 3D assets.

* Make the full `make physics` workflow run end to end on the default configuration: per-group Bayesian fits, exact origin statistics for FFT correlators, float64 weighted AIC window fits with a signal-to-noise cut, working prior seeding and data-scaled amplitude priors, electroweak lag units and frame alignment, geodesic edge lengths for multiscale scale selection, deterministic curl estimation, and dashboard fixes (frame-restricted geodesic cache, chunked distance statistics, consistent AIC time units, stale-output clearing). See reviews/physics_fixes.md (second round).

* Retain twistor triplet lag statistics for measured mass-fit covariance and exclude unselected channels from covariance preparation.

* Correct physics integration, cloning records, duplicate-particle geometry, analysis timing, parity observables, multiscale fitting, measured covariance, and dashboard invalidation. See reviews/physics_fixes.md for numerical compatibility notes.

* Reorganize Fractal Gas as Volume 2, promote analytical proofs to main chapters, and remove the retired framework volume.
* Restrict mathematical downloads to published chapters and preserve migrated Fractal Gas URLs.

* Clarify the binding-field remark on beta-function sign dependence and confinement justification.

0.2.0 (2024-10-10)
------------------

* First release on PyPI.

- Replace cloning-guided perturbations with bounded signed clone-score fields and adaptive covariance of comparison vectors across all six fractal optimizers. Remove lineage gating and log-fitness regression; retain compatible geometry on degenerate evidence and keep live scale bounds separate.
