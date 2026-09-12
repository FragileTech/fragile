# Volume II lecture experiments

This directory implements the 42 experiment IDs in
[the visualization plan](../../../../docs/source/project/volume2_interactive_visualization_plan.md).
Open `euclidean-gas/lecture.html?demo=I-01` through the local Lab server.
Append `&embed=1` for the compact lecture view.

## Run and verify

From the repository root:

    npm --prefix fractal-gas-web run build:euclidean-gas
    npm --prefix fractal-gas-web run test:euclidean-gas
    npm --prefix fractal-gas-web run test:euclidean-lectures
    uv run --no-project python fractal-gas-web/tools/serve-control.py --port 8770
    npm --prefix fractal-gas-web run test:euclidean-lectures-browser

The Rust build requires the pinned Rust toolchain and wasm-bindgen documented in
`algorithmic-gas/README.md`. Generated CPU/WebGPU bundles remain build artifacts.
The build also regenerates the 42 SVG posters and chapter manifest.
To refresh only those assets using an existing compiled CPU bundle:

    npm --prefix fractal-gas-web run build:euclidean-lectures

Build the book with `make docs`. Its Sphinx extension inserts two experiments in
each of the first 21 Volume II chapters according to `placements.json`. Captions
are reviewed in `captions.json`; posters and the generated manifest are committed
under `docs/_static_theory/gas-demos/`, so a documentation-only build needs no Rust.
Section placements are validated against actual Markdown blocks; changing a
target heading without updating the placement fails the documentation build.

## Implementation map

| Family                 | IDs           | Module           |
| ---------------------- | ------------- | ---------------- |
| Foundations            | I-01–I-10     | `foundations.js` |
| Convergence            | II-01–II-08   | `convergence.js` |
| Mean-field limits      | III-01–III-08 | `convergence.js` |
| Entropy and regularity | IV-01–IV-16   | `entropy.js`     |

Every descriptor implements the interface in `CONTRACT.md`. The interface
separates scientific computation from DOM rendering. Exact reference models run
in the same worker as WASM experiments and name their method in the displayed
caption. The WASM views use the compiled Rust engine, including validated
population fixtures and opt-in authentic cloning/BAOAB traces.

`worker.js` serializes commands, imports and initializes the CPU WASM module once
per view, and retains it across parameter resets. Initialization does not perform
an unreported scientific step. Each model frees its owned runs on replacement;
the worker also tracks and frees them on disposal or failed initialization.
The baseline is CPU WASM, with no WebGPU requirement.

`main.js` starts paused and maintains at most 80 display frames. Family models
bound their own scalar histories and experiment horizons. Run pauses when the
tab or embedded figure becomes hidden. The lecture page keeps one loaded iframe
at a time and removes its worker when closed or hidden by Expert Mode.

Scientific controls reset the seed experiment; Step advances one bounded work
unit defined by the selected model. A step can mean one engine update, one
independent frozen-population repetition, or a reference-model time increment.
The displayed message and axes identify that unit. Some algebraic experiments
are complete immediately and respond to parameter changes.

JSON export includes the selected display frame, its replay tick, seed,
parameters, and model-provided experiment metadata. Import recreates that seed
and runs the same number of steps. Hardware timing fields naturally vary between
replays. JSON imports are limited to 20 MB and 2000 replay steps. IV-16 also
exports its first run's binary engine checkpoint. Each plot exports standalone
SVG; the numerical table and JSON expose plotted values.

## Scope of the first implementation

All 42 IDs have working computations, controls, and plots. The proposal is a
broader design catalog: this implementation uses compact views rather than every
proposed tab, drag gesture, parameter sweep, and overlay. It does not yet include
cross-demo checkpoint import, a browser archive of long ensembles, live WebGPU
selection within the lecture host, or a general continuum/PDE solver. Exact
Gaussian, finite-state, graph, and analytic reference calculations provide the
corresponding lecture experiments. They are identified in their views.

The tests check mathematical identities and real WASM behavior, all descriptor
defaults and control endpoints, deterministic reset, finite outputs, section
placement, published URL paths, desktop/mobile interaction, export/replay,
and the one-iframe lifecycle. Production deployment uses the existing Euclidean
Gas artifact and Theory build; no separate hosting service is required.
