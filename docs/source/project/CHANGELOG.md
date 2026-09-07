# Changelog

Unreleased
----------

* Add optional Wave Jump shared-prefix execution with bounded search extension and single-action fallback.

* Evaluate native gauge locality coefficients by signed history cancellation and full-face tuple products, retaining original weights, masks, and selected laws.

* Reward hooked rock travel by distinct-rock distance, with an adjustable weight and no credit for respawn teleportation or duplicate tow hooks.

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

* Default Ants & Drops to five vehicles and add five-drop cargo tanks to harvesters and drones, full-load bonuses, gradual rewarded refinery unloading, replayable cargo state, and concept-based steampunk/futuristic Blender refineries.

* Refine vehicle silhouettes against the concept sheets with rounded pressure hulls, integrated drone armor, detailed cockpit and intake assemblies, corrected outward normals, and shared brushed-metal maps within the existing LOD budgets.

* Add illustrated tutorials for all six Lab tasks and six racing circuits, a complete scene-editor course and JSON reference, downloadable workshop scenes, and reproducible screenshot and example checks. Let editor pointer actions pass through the status overlay so tether placement can reach the second body.

* Refine both Blender asset styles with vehicle suspension and machinery, detailed dock assemblies, corrected ore normals and UVs, branching mineral textures, and zoom-dependent ore detail while preserving native collisions.

* Give Collaborative mining one high-mass, high-drag rock that a solo rocket can move slowly and two can haul faster; immediately replenish it after every delivery.

* Build 42 world asset types in both futuristic and steampunk styles with shared collision envelopes, distinct machinery, packed Blender/GLB sources, instanced scenery and pickups, replayable effects, and an expanded asset workshop.

* Add five video-traced historical kart circuits alongside Violet Circuit, under one Racing task with a difficulty-ordered track selector, geometry previews, source references, validated obstacles and checkpoints, and two-lap driving and replay regression tests.

* Add sixteen futuristic and steampunk world-asset concept sheets, a paired visual library with original downloads and generation prompts, and narrow-screen style-control checks.

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

* Reorganize Fractal Gas as Volume 2, promote analytical proofs to main chapters, and remove the retired framework volume.
* Restrict mathematical downloads to published chapters and preserve migrated Fractal Gas URLs.

* Clarify the binding-field remark on beta-function sign dependence and confinement justification.

0.2.0 (2024-10-10)
------------------

* First release on PyPI.
