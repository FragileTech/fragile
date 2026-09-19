# QFT Simulator

The third page of the Algorithmic Gas section (Lab · Lectures · QFT Simulator).
It runs a gas variant in the compiled Rust engine, accumulates channel
correlators on the recorded run, and shows the decay rates Rust fits to them.
Open `euclidean-gas/qft.html` through the local Lab server.

## The contract: JavaScript does no science

The rules of [`../lecture/CONTRACT.md`](../lecture/CONTRACT.md) apply unchanged.
JavaScript performs no random sampling, algorithm updates, scientific
statistics or fitting. Every number on the page originates in the wasm class
`SpectroscopyExperiment` or in the free functions `spectroscopy_defaults`,
`spectroscopy_capabilities`, `spectroscopy_analyze` and `spectroscopy_archive`.
The host adapts Rust arrays to SVG charts, tables and controls. Concretely:

- An undefined value (`None` in Rust, `null` in JSON) stays a gap in a line and
  an empty table cell. It is never drawn or printed as zero.
- A channel's availability and the reason it is unavailable are Rust strings
  and are shown verbatim. So are `report.notes` and every other `notes` array.
- The fitted quantity keeps the name Rust gives it (`MassEstimate.quantity`:
  "decay rate of the algorithm-time autocorrelation"). The host never relabels
  a rate as a particle mass. The reference comparison carries Rust's label
  "hypothesis mapping"; the channel → reference assignment is a user input.
- Analysis parameters the user does not set (fit windows, priors, SVD cut,
  stability grid, GEVP `t0` and cut, reference table) are left as Rust sent them
  or omitted so that Rust applies its defaults.
- The only arithmetic is drawing geometry: the lag axis `lag · time_step` in
  the unit Rust reports, the two edges `value ± error` of a dashed error band
  (`plots.js` has no band primitive), and the width of the progress bar. Tables,
  including the numeric fallback of every chart, print the Rust `value` and
  `error`, never a band edge.

`tests/euclidean-gas/qft-simulator.test.mjs` scans the four modules for
`Math.random`, exponentials, logarithms, roots and powers.

## Architecture

| File          | Role                                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../qft.html` | Static skeleton: masthead, a five-tab `tablist` and the panel containers.                                                                                                                                                                                             |
| `worker.js`   | `createWasmEngine(load)` adapts the wasm API to thirteen async methods; `createDispatcher(engine)` maps message types to them; `createQueue` serializes requests; `serve(scope, engine)` binds a worker scope; `createClient(worker)` is the main-thread promise RPC. |
| `model.js`    | `createSession(engine)` (create → bounded `advance` loop → snapshot, pause/resume, re-analysis, evidence, checkpoint, imports) and the pure functions that map Rust JSON to chart descriptors and tables.                                                             |
| `views.js`    | String-returning HTML builders (channel checklist, chart cards with SVG export and numeric fallback, tables, notes, analysis and mapping controls) and `initTabs`.                                                                                                    |
| `main.js`     | State, events and rendering of the active tab.                                                                                                                                                                                                                        |
| `style.css`   | Workbench rules on top of `../lecture/style.css`, whose dark tokens and chart, table and toolbar rules the page reuses.                                                                                                                                               |

Charts are drawn by `../lecture/plots.js` (`chartSVG`, `legendHTML`), which this
page does not modify.

The engine is injectable at both seams. `createDispatcher` and `createSession`
accept any object with the thirteen methods, and `createWasmEngine` accepts a
module loader. The tests pass a fake engine with canned Rust-shaped JSON; the
fake exists only under `tests/`.

### Tabs

1. **Setup** — variant selector from `defaults().variants` (book-only variants
   are listed disabled), walkers, dimensions, steps,
   replicas, seed, colour source and alignment, time axis, and the channel
   checklist grouped by family. Every control change asks
   `spectroscopy_capabilities(request)` again; the probe requests every catalog
   channel so that Rust reports on all of them.
2. **Run** — progress, walker cloud, coverage counters, live `C(τ)` and
   effective-rate charts for the selected channels, pause/resume, step,
   checkpoint save/restore, evidence export/import and run-archive import.
3. **Correlators** — per-channel `C(τ)` with error band, effective rate, and the
   analysis controls: estimator (automatic, frame average, source-frozen,
   Euclidean time), resampling, block size, replicas, time unit, connected part.
4. **Rates & fits** — fit method, stability scan, GEVP; one diagnostics table
   (χ², dof, Q, window, effective block, τ_int, prior dominance, rejection and
   no-signal reasons), fit windows, joint fits, GEVP levels.
5. **Physics comparison** — hypothesis mapping and anchor controls, reference,
   prediction, ratio and anchor-spread tables, calibration, couplings.

Controls of tabs 3–5 call `analyze` only. They never advance or re-create the
session. After an evidence or archive import there is no session; `analyze`
then re-runs `spectroscopy_analyze` or `spectroscopy_archive` on the retained
bytes.

## Message protocol

Main thread → worker: `{ id, type, payload }`. Worker → main thread:
`{ id, result }` or `{ id, error }` (the Rust error text). Requests are
answered one at a time in arrival order. `Uint8Array` results are transferred.

| `type`            | `payload`                                    | Rust call                                                | `result`             |
| ----------------- | -------------------------------------------- | -------------------------------------------------------- | -------------------- |
| `defaults`        | —                                            | `spectroscopy_defaults()`                                | defaults JSON        |
| `capabilities`    | request                                      | `spectroscopy_capabilities(json)`                        | capabilities JSON    |
| `create`          | request                                      | `SpectroscopyExperiment.create(json)`, then `snapshot()` | snapshot             |
| `advance`         | `{ steps }`, integer 1–64                    | `advance(steps)`                                         | snapshot             |
| `snapshot`        | —                                            | `snapshot()`                                             | snapshot             |
| `analyze`         | `AnalysisConfig`                             | `analyze(json)`, or the retained import (see above)      | `SpectroscopyReport` |
| `presentation`    | `AnalysisConfig`                             | `presentation(json)`                                     | `[ExperimentResult]` |
| `evidence`        | —                                            | `evidence()`                                             | CBOR bytes           |
| `checkpoint`      | —                                            | `checkpoint()`                                           | CBOR bytes           |
| `restore`         | CBOR bytes                                   | `SpectroscopyExperiment.restore(bytes)`                  | snapshot             |
| `import_evidence` | `{ bytes, analysis }`                        | `spectroscopy_analyze(bytes, json)`                      | `SpectroscopyReport` |
| `import_archive`  | `{ bytes, config: {measurement, analysis} }` | `spectroscopy_archive(json, bytes)`                      | `SpectroscopyReport` |
| `dispose`         | —                                            | `free()`                                                 | `null`               |

Creating, restoring or importing frees the previous session. A failed import
leaves the live session untouched.

### JSON shapes the host reads

The report is `SpectroscopyReport` of
`algorithmic-gas/crates/algorithmic-gas/src/physics/spectroscopy/report.rs`, and
the analysis is `AnalysisConfig` of `config.rs`. The runner payloads are those
of `algorithmic-gas/crates/benchmarks/src/spectroscopy.rs`:

```text
defaults      { schema_version, request: SpectroscopyRequest,
                variants: [{ name, title, implemented, request }],
                catalog: [CatalogEntry], reference: ReferenceTable }
CatalogEntry  { id, spec: ChannelSpec, kind, family, standard, availability,
                descriptor: { definition, book_label, spatial_parity, note },
                exchange, requirements: { records, dimension }, components,
                correlatable, assignment }
capabilities  { capabilities: Capabilities,
                channels: [{ id, availability: { status, reason? } }], chunk }
snapshot      SessionSnapshot { schema_version, step, steps, done, chunk,
                replicas: [{ seed, step, frames, segments, terminal }],
                capabilities, calibration,
                walkers: { dimension, positions: flat [N·d], eligible: [N] },
                channels: [LiveChannel { id, availability, coverage,
                           correlator: [f64|null], effective_mass: [f64|null] }],
                notes }
presentation  [ExperimentResult { title, plots: [{ title, x_label, y_label,
                series: [{ name, kind, points }] }], metrics, notes }]
```

A catalog row is one (specification, element kind) pair and its `id` is the
channel id (`meson/scalar/standard/distance`). Rust measures specifications, so
the request carries each ticked `spec` once; the analysis is then restricted to
the ticked rows through `AnalysisConfig.channels`. Imported evidence and
restored checkpoints are analysed over every channel they hold. Live channels
carry no lag axis and no errors: the array index is the lag in frames.

The request the host sends is the request Rust resolved for the chosen variant
(`variants[i].request`, or `defaults.request`) with `variant`, `run.walkers`,
`run.dimensions`, `steps`, `replicas`, `seed`, `measurement.color`,
`measurement.time` and `measurement.channels` overridden. Choosing a variant
restarts the form from that variant's request. A snapshot carries no request,
so a restored checkpoint can be continued and analysed but not restarted.

## Build, run and test

From the repository root:

```bash
npm --prefix fractal-gas-web run build:euclidean-gas
uv run --no-project python fractal-gas-web/tools/serve-control.py --port 8770
npm --prefix fractal-gas-web run test:euclidean-qft-simulator
```

The page needs a wasm bundle that exports the spectroscopy bindings; with an
older bundle it reports that the engine is unavailable. The node test needs no
wasm and no browser.
