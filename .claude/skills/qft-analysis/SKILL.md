---
name: qft-analysis
description: Measure and analyze QFT channel spectroscopy of Algorithmic Gas runs through the native Rust route (the gas-spectroscopy CLI and SpectroscopySession) - channel correlators, decay rates, fit and coverage diagnostics, and a hypothesis-labelled Standard Model comparison - and re-analyze stored evidence or recorded run archives without running the gas again. Use when analyzing QFT simulation results, extracting particle masses or spectra, comparing channels with Standard Model values, or validating a calibration. Also documents the legacy Python RunHistory route and why its strong-sector masses are unreliable.
allowed-tools: Bash(cargo:*), Bash(uv:*), Read, Write, Glob
argument-hint: "[--request FILE] [--variant NAME] [--steps N] [--replicas N] [--channels LIST] [--evidence FILE] [--archive GLOB] [--analysis FILE] [--output DIR] [--legacy-python]"
disable-model-invocation: false
---

# QFT Analysis Skill

Drive the Rust spectroscopy subsystem of the Algorithmic Gas: measure gauge-theory
operators on a recorded run, fit their correlators, and report the result with its
diagnostics. Work from `/home/guillem/fragile/algorithmic-gas` unless told otherwise.

## What the numbers are

The fitted quantity is the **decay rate of the algorithm-time autocorrelation**. It is a
mass only under a positive self-adjoint transfer representation; the gas is not
reversible, so complex or oscillating modes are expected and reject the exponential
model instead of producing a number. Every report repeats this in `notes[0]`.

Three rules follow, and they are not negotiable:

1. Never relabel a rate as a particle mass. The channel → particle assignment is an
   input hypothesis (`AnalysisConfig.assignments`), carried through the report with the
   label `hypothesis mapping`.
2. Never choose a channel, a fit window, a prior or an anchor because the result agrees
   better with a Standard Model value. That is the audited defect D2.
3. Never print an undefined value as zero. `null` in the JSON stays an empty cell or a
   gap in a curve.

## Route selection

**Native (default).** The Rust subsystem in
`crates/algorithmic-gas/src/physics/spectroscopy` plus
`crates/benchmarks/src/spectroscopy.rs`, driven by the `gas-spectroscopy` binary. It is
f64 throughout, streams the recording in bounded chunks, declares per-channel
availability instead of returning a number it cannot justify, and pins the corrections
to every confirmed defect of the Python pipeline.

**Legacy Python.** Only when the user explicitly asks for it, or when the input is an
existing `RunHistory` `.pt` file that cannot be re-run. See
[Legacy Python route](#legacy-python-route) — its strong-sector masses are not
trustworthy.

## Native workflow

Spectroscopy analyses an f64 run on the CPU backend and refuses anything else before it
builds a gas, so keep `run.gas.precision` at `"f64"` and `run.gas.backend` at `"cpu"`.

### Step 1 — Orient

Ask the subsystem what it can do before writing a request:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-spectroscopy -- defaults --output /tmp/spectroscopy-defaults.json
```

The document holds the default request, one default request per implemented variant of
the variants registry, the channel catalog with the records each channel requires and
its availability under the default request, and the reference table. `variants` prints
the registry on its own. Read the catalog before selecting channels: a channel whose
records the chosen variant does not write is unavailable, and the reason says why.

### Step 2 — Build the request

A `SpectroscopyRequest` is `{variant, run, steps, replicas, seed, chunk, spectroscopy}`;
`run` is an ordinary `RunConfig` and is authoritative, `variant` only labels it. Missing
fields take the defaults, so `{}` is the Einstein–Hilbert reference instance with 200
walkers in f64, 2000 steps, four replicas, seed 7, chunk 16 and the standard channel set
(`warmup: 16`, `max_lag: 80`).

```json
{
  "variant": "einstein_hilbert",
  "steps": 512,
  "replicas": 4,
  "seed": 7,
  "chunk": 16
}
```

Constraints to respect when the user asks for something else:

- `run.gas.precision` must be `"f64"`. There is no precision fallback.
- Replicas 1..32, steps 1..10000000, chunk 1..32. Replica `r` runs with
  `seed + r * 104729`; `run.gas.seed` is ignored.
- Under dense viscosity (`qft.viscosity`) the chunk must be 1..4: that configuration
  records `2 N (N - 1)` influence rows per step.
- Fewer than four replicas makes the reported errors block-resampling errors inside one
  run, not replica standard errors. The report says so; repeat it to the user.
- Colour channels need a non-zero viscous force. `euclidean` has none, so choose
  `viscous_euclidean` or `einstein_hilbert`, or set an explicit
  `ColorSource::RecordedField`.

### Step 3 — Run and keep the evidence

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-spectroscopy -- \
  run /tmp/spectroscopy-request.json --output /tmp/spectroscopy-report.json --evidence /tmp/spectroscopy-evidence.cbor
```

Always write the evidence. It is the CBOR `{schema_version, request, configs,
measurements}` that makes every later analysis free.

### Step 4 — Re-analyze without running the gas

Resampling, fit windows, the estimator, the reference table, the anchors and the channel
selection are analysis parameters. Changing any of them is a re-analysis of the same
measurement:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-spectroscopy -- \
  analyze /tmp/spectroscopy-evidence.cbor /tmp/analysis.json --output /tmp/spectroscopy-narrow.json
```

Never re-run the gas to change a fit window. If the user wants several windows, priors
or estimators compared, run once and analyze many times.

### Step 5 — Measure an existing archive

A run already recorded by the benchmark runner can be measured directly:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-benchmark -- \
  --variant einstein_hilbert --precision f64 --steps 750 --record /tmp/run.cbor --record-graph
cargo run --release -p algorithmic-gas-benchmarks --bin gas-spectroscopy -- \
  archive /tmp/spectroscopy-request.json /tmp/run.cbor --output /tmp/spectroscopy-report.json
```

Pass one archive per replica. The archive must be an f64 run — the Einstein–Hilbert
preset runs in `Precision::F32` unless `--precision f64` says otherwise, and spectroscopy
refuses an f32 archive rather than promoting it. `--record-graph` is needed for the
graph-based diagnostics (geodesic scales, smoothing).

### Step 6 — Read the report before reporting it

Read in this order and do not skip ahead:

1. `notes` — the interpretation note, the replica-count note and every measurement note.
2. `capabilities` and `calibration` — what the run could support at all.
3. Per channel: `availability` first. An unavailable channel has a reason string; print
   it verbatim and do not substitute a number from elsewhere.
4. `coverage` — frames, valid elements, masked counts. Thin coverage invalidates a fit
   before its χ² does.
5. `estimator` — `frame_mean`, `source_frozen` or `euclidean_time`. An exchange-odd
   channel on a mutual pairing legitimately falls back to the source-frozen propagator;
   say which estimator produced each number.
6. `fits` / `mass` — with χ², dof, Q, the window, the effective block, τ_int and the
   prior-dominance diagnostic. A prior-dominated ground state or a below-threshold
   signal-to-noise reports **no rate**; report that as the result, not as a failure to
   be worked around.
7. `groups`, `gevp`, `comparison`, `couplings`, `flow`.

In the coupling report keep the three kinds of number apart: a **scale** is configured or
fixed in the warm-up, a **proxy** is a book formula on those scales and is never compared
with a Standard Model coupling, and a **target** is a Standard Model input re-expressed
as a gas parameter. The Standard Model map is an inversion from reference inputs to gas
parameters, never a measurement of them — do not present α_em, sin²θ_W or α_s as outputs.

The graph smoothing diagnostic (`flow`) sets no length scale. Do not read a `w0` or any
other scale off its roughness curve.

### Step 7 — Summarize

Produce one channel table (id, availability or its reason, estimator, coverage, rate ±
error, χ²/dof and Q, prior dominance), the comparison table if assignments were given
(always labelled a hypothesis mapping), and the notes verbatim. Write the report file
only where the user asked; otherwise show the tables and cite the JSON paths.

## Current interface status

Every subcommand and every library entry point listed above is implemented and runs end to
end. What a request still cannot ask for: a run in f32 or on a GPU backend (refused, not
promoted), a momentum projection on a non-periodic box, and the book-only variants
`geometric`, `latent` and `environment`, which the registry marks unimplemented. Channels
whose inputs a variant does not record come back `unavailable` with a sentence; that is
the normal outcome, not a failure, and the sentence is what to report. Never synthesize the
JSON a command would have printed, and never fall back to the Python route silently.

## Library entry points

For work inside the workspace rather than through the CLI:

- `algorithmic_gas::physics::spectroscopy::measure_archive(&MeasurementConfig, &RunArchive<f64>)`
  and `measure_archive_with(.., Extensions)` for injected operators or field sources.
- `algorithmic_gas::physics::spectroscopy::analyze(&[Measurement], &AnalysisConfig)`,
  and `select_estimator` for a live view.
- `algorithmic_gas_benchmarks::spectroscopy::{SpectroscopySession, analyze_evidence, analyze_archive}`.
- `spectroscopy::presentation::present` for the plot and metric descriptors the SVG
  adapters render.
- The browser equivalent is `/euclidean-gas/qft.html` over the same Rust through
  `crates/wasm/src/spectroscopy_bindings.rs`.

Measurements combine only when their fingerprints agree — measurement configuration, gas
configuration with the seed cleared, resolved capabilities, injected component identities
— with equal population size and schema version. Do not attempt to merge measurements of
different configurations; the analysis rejects it.

## Legacy Python route

`src/experiments/calibrate_fractal_gas_qft.py` (Standard Model constants → algorithmic
parameters) and `src/experiments/analyze_fractal_gas_qft.py` (masses and observables from
a `RunHistory` `.pt` file), with reference constants in
`src/experiments/constants_check.py`. Anchors: electron `0.000510998950`, Z `91.1876`,
tau `1.77686`, or a custom mass and label.

```bash
uv run python src/experiments/calibrate_fractal_gas_qft.py \
  --history-path outputs/<run>_history.pt --m-gev 91.1876 --scale-label Z_Boson --run-id <id> --qsd-iter 4
uv run python src/experiments/analyze_fractal_gas_qft.py \
  --history-path outputs/<run>_history.pt --analysis-id <id> \
  --compute-particles --build-fractal-set --particle-operators "baryon,meson,glueball" \
  --use-connected --use-local-fields
```

Outputs land in `outputs/qft_calibration/<run_id>_calibration.json` and
`outputs/fractal_gas_potential_well_analysis/<analysis_id>_metrics.json`.

**State these limitations whenever you report a number from this route.** The operator
audit is recorded in
[`algorithmic-gas/QFT_VALIDATION.md`](../../../algorithmic-gas/QFT_VALIDATION.md),
section "Algorithmic spectroscopy: scope, parity exclusions and open conventions", with
the fixture-level exclusion list in
[`algorithmic-gas/crates/algorithmic-gas/tests/fixtures/qft/README.md`](../../../algorithmic-gas/crates/algorithmic-gas/tests/fixtures/qft/README.md).
The two findings that make the strong-sector masses unreliable:

- **D1 — exchange-odd channels cancel.** The Fisher–Yates pairing is an involution
  (`c(i)=j`, `c(j)=i`), and `z_ji = conj(z_ij)`, so a frame mean over both orientations
  of an exchange-odd quantity vanishes exactly. The pseudoscalar (`Im z_ij`), the vector
  (`Re z_ij · (x_j − x_i)`) and the imaginary part of every U(1) channel are float32
  roundoff — measured `1.3e-8` and `1.5e-8` against `0.14` and `0.35` for the scalar and
  axial channels. A fit on them returns a rate of the noise. The pion and rho numbers of
  this pipeline are therefore not measurements of anything.
- **D2 — PDG-tuned priors.** The per-channel `dE_ground` priors (π 0.024, σ 0.09, ρ 0.13,
  N 0.16, G 0.29) reproduce PDG ratios by construction (π/N 0.150 against 0.149). Any
  agreement with the Standard Model that this route reports is built into the prior and
  is not evidence.

Further confirmed defects that affect its error bars and its GEVP levels: D3 (GEVP
bootstrap shuffles time points independently), D6 (block size hard-coded to 10 and a
jackknife that reuses the full-sample mean; nominal 68 % intervals cover 0.41), D4 (the
SU(2) and mixed electroweak amplitudes are identically zero because `epsilon_clone`
doubles as the Gaussian range) and D8 (vector, axial and tensor components averaged
before correlating, which is not rotation invariant). Rust pins the corrected behaviour
for all of them.

Do not "fix" the Python pipeline as part of an analysis task. If a user needs corrected
numbers, re-run the measurement through the native route.

## Troubleshooting

| Symptom | Cause and action |
|---|---|
| Every colour channel unavailable, "viscous force is identically zero" | The variant has no viscous force. Use `viscous_euclidean` or `einstein_hilbert`, or configure `qft.viscosity` / `qft.graph_viscosity`, or select an explicit `RecordedField` colour source. |
| "spectroscopy requires an f64 recorded run" | The archive or request is f32. Set `run.gas.precision` to `"f64"` and record again. |
| "momentum projection needs a periodic box" | A momentum-projected glueball needs a periodic boundary. Drop the projection or change the boundary. |
| "no Euclidean-time axis declared" | The Euclidean-time estimator needs a geometry projection that drops a coordinate. Use another estimator or another variant. |
| "exchange-odd operator cancels on a mutual pairing" | Expected on an involutive pairing. The source-frozen propagator is the estimator to use; a non-involutive companion law would restore the frame mean. |
| "Dense viscosity requires chunk 1..4" | Lower `chunk`. |
| "measurements of different configurations ... cannot be combined" | The fingerprints differ. Re-measure the replicas with one configuration. |
| "runs-as-samples requires at least 8 replicas" | Use `pooled_blocks`, or raise `replicas`. |
| "analysis windows exceed the measured lag range" | Raise `measurement.max_lag` and re-measure, or narrow the fit window. |
| "spectroscopy analyses an f64 CPU run" | The request asks for f32 or a GPU backend. Set `run.gas.precision` to `"f64"` and `run.gas.backend` to `"cpu"`. |

## References

- `algorithmic-gas/README.md`, section "Algorithmic spectroscopy" — the subsystem, the
  channel families and their requirements, and the CLI.
- `algorithmic-gas/QFT_VALIDATION.md` — validated identities, parity exclusions and the
  conventions still open.
- `docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md` — the normative operator
  definitions; `09_qft_calibration.md` — channel knobs and mass plateaus;
  `1_the_algorithm/04_gas_variants.md` — which variant supports what.
