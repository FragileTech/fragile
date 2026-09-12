# Lecture experiment interface

The Rust registry in `algorithmic-gas/src/lecture_experiments.json` defines all
128 experiments, their controls, placement metadata and teaching descriptions.
`build-partvi-index.mjs` generates the browser registry. The asset builder checks
that it matches the compiled catalog.

`LectureRequest` contains `id`, `seed`, `steps` and `parameters`. Resolution rejects
unknown controls, unsupported options, out-of-range values and fractional counts.
Each scientific control change creates a fresh Rust session.

`LectureSession` owns the actual gas runs, complete recording, bounded ensembles,
measurements and predictions. Native `gas-lecture` and WASM `LectureExperiment`
call this implementation. Its `advance` operation executes at most 32 updates per
run; the browser requests eight. Snapshots contain actual completed-step counts,
per-run budgets, terminal status, measurements and eligible walker positions.
Until a required sample window is collected, the host displays the executed swarm.

JavaScript adapts Rust result arrays to SVG charts and scene controls. It performs
no random sampling, algorithm updates, scientific statistics or theory fitting.
`run-model.js` owns the WASM object and frees it on disposal; `worker.js` serializes
requests. Controls, display rotation, clipping and export are host responsibilities.

Evidence exports contain the resolved request, run configurations and validated
archives. Import recomputes measurements in Rust. Continuation experiments also
retain a complete checkpoint, future random schedules, per-run fingerprints and
bounded embedded archives; replay validates them against newly executed futures.
A result-only JSON object is not execution evidence. Session checkpoints preserve
all runs and budgets and reconstruct the appropriate configured providers.

Scenes use stable node IDs and `[x,y,time]` chart positions, named edge layers and
ordered face vertices. Rotation and vertical scale change only the display. The
viewer supports time clipping, layers, cell selection, lineage tracing, keyboard
controls and SVG export. The host retains eight complex scene frames or 80 simpler
frames; scientific recording has an independent explicit Rust memory budget.
The scrubber revisits those retained views, including their earlier scene windows.

V-01–V-04 compute scientific measurements on the full archive graph and return
its complete counts in a compact graph summary. Their scene selects the last
eight recorded transitions and all selected-window edges, retaining incoming
historical endpoints at their recorded coordinates. Explicit window bounds and
displayed/omitted counts describe rendering coverage. The result carries no
duplicate full graph JSON. Complete archive exports preserve the data required
to reconstruct the graph and recompute all full-history measurements.

Every registered experiment must execute through the native default and control
stress suites and the compiled browser tests. Numerical oracles belong in Rust
tests; lecture measurements always originate in executed Euclidean Gas runs.
