# QFT Simulator

The third page of the Algorithmic Gas section (Lab · Lectures · QFT Simulator).
It runs a gas variant in the compiled Rust engine, accumulates channel
correlators on the recorded run, and shows the decay rates Rust fits to them.
Open `euclidean-gas/qft.html` through the local Lab server.

The host and the Rust it calls are both complete: `make algorithmic-gas-web`
produces a bundle in which `spectroscopy_defaults`, `spectroscopy_capabilities`
and the whole `SpectroscopyExperiment` run, analyse and plot a real session.
When a channel, a fit, a group or a basis has nothing to report, Rust says so
in its own words and the page shows that sentence instead of a number: nothing
on the page is ever faked.

## Naming

The section, the engine and its three pages are "Algorithmic Gas": the Lab
(`index.html`), the Lectures (`lecture.html`) and this QFT Simulator
(`qft.html`). "Euclidean Gas" names a _variant_ — one gas configuration offered
in the Setup tab next to Viscous Euclidean Gas, Einstein–Hilbert Gas and the
book-only entries — and is therefore correct in the variant selector, in
`defaults().variants[i].title` and wherever a run is described by its
configuration. The directory, the URL prefix `euclidean-gas/`, the npm scripts
(`build:euclidean-gas`, `test:euclidean-qft-simulator`), the CI artifact
`euclidean-gas-lab` and the deploy path `site/euclidean-gas` keep the old
spelling on purpose: CI and the published URLs depend on them, so only
human-visible text carries the section name.

## The contract: JavaScript derives no number

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
- A measured channel the live estimator cannot report (`LiveChannel.estimator`
  is null, as for an exchange-odd operator that cancels on a mutual pairing)
  keeps its coverage row with an empty estimator, and its `LiveChannel.note` is
  listed under the session's notes. It is never dropped without a reason.
- The fitted quantity keeps the name Rust gives it (`MassEstimate.quantity`:
  "decay rate of the algorithm-time autocorrelation"). The host never relabels
  a rate as a particle mass. The reference comparison carries Rust's label
  "hypothesis mapping"; the channel → reference assignment is a user input.
- Analysis parameters the user does not set (fit windows, priors, SVD cut,
  stability grid, GEVP `t0` and cut, reference table) are left as Rust sent them
  or omitted so that Rust applies its defaults.
- The host never sends a configuration Rust would refuse. The only bound it has
  to mirror is `GEVP_BASIS` (2..=16 channels, from `GevpBasis::validate` in
  `spectroscopy/config.rs`): outside it the GEVP control is disabled, says which
  bound was missed and no basis is requested. `spectroscopy_defaults()`
  publishing that bound would remove the copy.
- The host performs no arithmetic on a reported number at all. Every plotted
  point comes from `spectroscopy/presentation.rs`: the lag axis is already
  `lag · time_step` in the unit Rust reports, and the two dashed edges of an
  error band are the series Rust names `"<name> + error"` and
  `"<name> - error"` (`plots.js` has no band primitive). The page only selects
  a result by title, joins the consecutive runs Rust split at an undefined
  point into one curve with a `[null, null]` gap, and assigns a colour. Tables,
  including the numeric fallback of every chart, print the Rust `value` and
  `error` and leave the band edges out. The progress bar's width is the only
  number the page still computes, and it plots nothing.

`tests/euclidean-gas/qft-simulator.test.mjs` scans the four modules
(`main.js`, `model.js`, `views.js`, `worker.js`) for `Math.random`,
`crypto.getRandomValues`, exponentials, logarithms, roots, powers and the `**`
operator, and for the words "particle mass" and "hadron mass". A host change
that needs any of them is a contract violation: the computation belongs in
`physics/spectroscopy/` and the result belongs in the report.

It also asserts that no source builds a band edge of its own (`value ± error`):
the tables assert that the drawn points are identical to the series
`presentation()` returned.

Charts therefore need an analysis AND its presentation. `analyze()` asks for
both with the same `AnalysisConfig`, which analyses the measurement twice in
Rust; a binding that turned an existing report into plots without re-analysing
would halve that cost.

## Architecture

| File          | Role                                                                                                                                                                                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../qft.html` | Static skeleton: masthead, a five-tab `tablist` and the panel containers.                                                                                                                                                                                                                    |
| `worker.js`   | `createWasmEngine(load)` adapts the wasm API to the fourteen async methods of `MESSAGE_TYPES`; `createDispatcher(engine)` maps message types to them; `createQueue` serializes requests; `serve(scope, engine)` binds a worker scope; `createClient(worker)` is the main-thread promise RPC. |
| `model.js`    | `createSession(engine)` (create → bounded `advance` loop → snapshot, pause/resume, re-analysis, evidence, checkpoint, imports) and the pure functions that map Rust JSON to chart descriptors and tables.                                                                                    |
| `views.js`    | String-returning HTML builders (channel checklist, chart cards with SVG export and numeric fallback, tables, notes, analysis and mapping controls) and `initTabs`.                                                                                                                           |
| `main.js`     | State, events and rendering of the active tab.                                                                                                                                                                                                                                               |
| `style.css`   | Workbench rules on top of `../lecture/style.css`, whose dark tokens and chart, table and toolbar rules the page reuses.                                                                                                                                                                      |

Charts are drawn by `../lecture/plots.js` (`chartSVG`, `legendHTML`), which this
page does not modify.

The engine is injectable at both seams. `createDispatcher` and `createSession`
accept any object with the methods of `MESSAGE_TYPES`, and `createWasmEngine`
accepts a module loader. The tests pass a fake engine that replays payloads
recorded from the compiled bundle, so they check the shapes Rust really emits;
the fake exists only under `tests/`.

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
   Every other plot of the analysis is a gallery behind the "Every Rust
   presentation plot" button: an analysis of twenty channels carries some fifty
   plots, and building them all on every control change is a second of layout
   for charts nobody asked for.
4. **Rates & fits** — fit method, stability scan, GEVP; one diagnostics table
   (χ², dof, Q, window, effective block, τ_int, prior dominance, rejection and
   no-signal reasons), fit windows, joint fits, GEVP levels.
5. **Physics comparison** — hypothesis mapping and anchor controls, reference,
   prediction, ratio and anchor-spread tables, calibration, couplings.

Controls of tabs 3–5 call `analyze` and `presentation` only. They never advance
or re-create the session. After an evidence or archive import there is no
session; both calls then run on the retained bytes through
`spectroscopy_analyze` / `spectroscopy_presentation` and `spectroscopy_archive`
/ `spectroscopy_archive_presentation`, so an import draws the same charts as a
live session. A restored checkpoint is analysed as soon as it is adopted.

## Message protocol

Main thread → worker: `{ id, type, payload }`. Worker → main thread:
`{ id, result }` or `{ id, error }` (the Rust error text). Requests are
answered one at a time in arrival order. `Uint8Array` results are transferred.

| `type`            | `payload`                                    | Rust call                                                | `result`              |
| ----------------- | -------------------------------------------- | -------------------------------------------------------- | --------------------- |
| `defaults`        | —                                            | `spectroscopy_defaults()`                                | defaults JSON         |
| `capabilities`    | request                                      | `spectroscopy_capabilities(json)`                        | capabilities JSON     |
| `create`          | request                                      | `SpectroscopyExperiment.create(json)`, then `snapshot()` | snapshot              |
| `advance`         | `{ steps }`, integer 1–64                    | `advance(steps)`                                         | snapshot              |
| `snapshot`        | —                                            | `snapshot()`                                             | snapshot              |
| `request`         | —                                            | `request()`                                              | `SpectroscopyRequest` |
| `analyze`         | `AnalysisConfig`                             | `analyze(json)`, or the retained import (see above)      | `SpectroscopyReport`  |
| `presentation`    | `AnalysisConfig`                             | `presentation(json)`, or the import's own binding        | `[ExperimentResult]`  |
| `evidence`        | —                                            | `evidence()`                                             | CBOR bytes            |
| `checkpoint`      | —                                            | `checkpoint()`                                           | CBOR bytes            |
| `restore`         | CBOR bytes                                   | `SpectroscopyExperiment.restore(bytes)`                  | snapshot              |
| `import_evidence` | `{ bytes, analysis }`                        | `spectroscopy_analyze(bytes, json)`                      | `SpectroscopyReport`  |
| `import_archive`  | `{ bytes, config: {measurement, analysis} }` | `spectroscopy_archive(json, bytes)`                      | `SpectroscopyReport`  |
| `dispose`         | —                                            | `free()`                                                 | `null`                |

Creating, restoring or importing frees the previous session. A failed import
leaves the live session untouched.

`advance`, `snapshot`, `request`, `evidence` and `checkpoint` need a live
session and are rejected with "Create or restore a spectroscopy session first."
without one. `defaults`, `capabilities`, `create`, `restore`, `import_evidence`
and `import_archive` never need one, and `analyze` and `presentation` fall back
to the retained import bytes through `spectroscopy_analyze` /
`spectroscopy_presentation` and `spectroscopy_archive` /
`spectroscopy_archive_presentation`. The dispatcher rejects an unknown `type` and a `steps` budget
outside 1–64 before the engine is asked, so a malformed message can never reach
Rust.

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
                signature: null | { requires: { records, dimension },
                  components, exchange, correlatable, normalization,
                  degenerate, propagatable, auxiliary,
                  descriptor: { definition, book_label, spatial_parity, note } },
                assignment }
capabilities  { capabilities: Capabilities,
                channels: [{ id, spec, availability: { status, reason? },
                             requested }], chunk }
snapshot      SessionSnapshot { schema_version, step, steps, done, chunk,
                replicas: [{ seed, step, frames, segments, terminal }],
                capabilities, calibration,
                walkers: { dimension, positions: flat [N·d], eligible: [N] },
                channels: [LiveChannel { id, availability, coverage, estimator,
                           correlator: [f64|null], effective_mass: [f64|null],
                           note }],
                notes }
presentation  [ExperimentResult { experiment, title, model,
                metrics: [{ label, value, unit }],
                plots: [{ title, x_label, y_label,
                          series: [{ name, kind, points }] }],
                notes, details: { calculation_origin, precision,
                                  schema_version, request } }]
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

## Run it locally

From the repository root, build the wasm engine once and then serve the web
tree:

```bash
make algorithmic-gas-web          # npm ci + npm run build:euclidean-gas
make algorithmic-gas-lab          # serve-control.py on ALGORITHMIC_GAS_PORT (8770)
```

Then open <http://127.0.0.1:8770/euclidean-gas/qft.html>; the server prints the
section's address as "Algorithmic Gas laboratory". The equivalent without the
Makefile is:

```bash
npm --prefix fractal-gas-web run build:euclidean-gas
uv run --no-project python fractal-gas-web/tools/serve-control.py --port 8770
```

`build:euclidean-gas` compiles `algorithmic-gas-wasm` for
`wasm32-unknown-unknown` twice (`engine/cpu`, `engine/webgpu`) and runs
`wasm-bindgen`, whose version is pinned: without `wasm-bindgen` 0.2.114 on
`PATH` (or in `WASM_BINDGEN`) the build stops and prints the `cargo install`
line. Pass `--cpu-only` for the CPU profile alone. The page needs a bundle that
exports the spectroscopy bindings; with an older bundle the masthead status
reports that the engine is unavailable, and nothing on the page is faked.

## Tests

```bash
npm --prefix fractal-gas-web run test:euclidean-qft-simulator
```

That is `node --test tests/euclidean-gas/qft-simulator.test.mjs`. It needs no
wasm, no server and no browser: it drives `worker.js` and `model.js` through
the fake engine of `tests/euclidean-gas/qft-simulator-support.mjs`, whose canned
payloads are shaped like the Rust JSON. It covers the worker protocol
(routing, rejection, ordering, error text, binary transfer, session freeing),
the session state machine (bounded advance loop, pause/resume, re-analysis
without advancing, imports without a session), the request and analysis
adapters, the rendering rules (gaps instead of zeros, verbatim Rust reasons and
notes, band edges from Rust's `value` and `error`, SVG export and numeric
fallback), the module scan of the contract above, and the tab and section
navigation, including the page titles "Algorithmic Gas · QFT Simulator" and
"Algorithmic Gas Lab · Fragile".

Run a single test with `node --test --test-name-pattern "<text>"` on the same
file. When the fake engine's payloads and the real ones drift, the test is the
place to fix: change `qft-simulator-support.mjs` to match Rust, never the page
to match the fake.

The compiled engine is exercised in a real browser by

```bash
make algorithmic-gas-lab   # serves the lab on 8770
npm --prefix fractal-gas-web run test:euclidean-qft-simulator-browser
```

That is `node tests/euclidean-gas/qft-simulator-browser.mjs`, Playwright driving
the system Chrome (`channel: "chrome"`) against `LECTURE_BASE_URL`
(`http://127.0.0.1:8770` by default). It loads the page against the real wasm,
checks that the variant selector is the Rust registry with the book-only
variants disabled, that the canonical Euclidean Gas disables the colour
channels with the engine's own sentences while the fitness and electroweak
channels stay available, runs 96 Einstein–Hilbert steps on 24 walkers and two
channels, and asserts that changing the subtraction re-analyses the same
measured frames — new numbers, same step, same replicas. Screenshots and a
summary land in `outputs/qft-simulator-review/`.

Related suites: `npm run test:euclidean-qft` (the lecture QFT experiments) and
`npm run test:euclidean-gas` (the engine bindings), both under
`fractal-gas-web`.
