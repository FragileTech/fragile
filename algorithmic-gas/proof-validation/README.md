# Convergence chapter validation

The native `gas-convergence` runner inventories convergence chapters 1–3 and
checks the existing Rust operators using recorded stage boundaries. It preserves
the Gaussian companion law, sampled fitness, component collisions, scheduled
revival, and terminal boundary. Auxiliary uniform-companion and Hermite-patch
identities run as separately scoped fixtures.

The empirical standardization diagnostics use probability-law Wasserstein error.
The corrected full map has gain `1 / sigma_min` and squared-error gain
`1 / sigma_min²`, independent of population size. Independent native measurement
error relative to its actual conditional mixture has the decreasing bound
`D² / (2 sigma_min² sqrt(k))`. Population probes retain the Gaussian donor weights,
the realized global normalizer, and optimal scalar transport. Summed row errors
are retained only as measurement baselines.

The `decay` configuration controls a separate paired trajectory matrix. By default
it uses populations 4/16/64, dimensions 1/2, eight independent pairs, and 128 complete
updates per pair across quadratic, sphere and Rastrigin cases. Its chapter-theory
error is optimal transport between the normalized alive empirical laws after
revival and kinetics. Dead coordinates awaiting revival cannot dilute the initial
error or enter this observable. Full marked all-slot transport, barycenter and
structural error remain separate diagnostics. Quadratic linear controls have a discrete Lyapunov matrix
and contraction factor computed before simulation; independent noise has an
explicit floor. Canonical selected, capped and killed cases retain their own
empirical rate and uncertainty without borrowing the linear control theorem.
Set `decay` to `null` to skip these auxiliary trajectories, or supply its complete
configuration to change the populations, dimensions, seed pairs and fixed fit window.

`shared_decay` repeats that matrix with a native common-stream coupling. Before
each paired update, both swarms are put into intrinsic representatives through
checkpoint restoration, gathering every state field together. The shared Gaussian,
uniform and categorical Gumbel innovations preserve both native marginal laws.
This route requires the declared memory-free presets, with elites, donor history,
geometry and native recording disabled. It gives a separate comparison of initial
state error; independent-run error retains its finite-population sampling floor.
The primary coupling diagnostic uses the physical hypocoercive transport of the
alive laws, with the chapter's normalization. The auxiliary marked all-slot
diagnostic adds a status penalty and retains its separate scope.

`gas-decay` runs this trajectory suite independently. The explicitly named
`rastrigin_diffusion_calibration` profile accepts `calibration_position_diffusion`;
it preserves all other native parameters. The reference setting 1.25 at `h=.04`
gives a per-step positional Gaussian standard deviation equal to one quarter of
the Rastrigin well spacing. Calibration configurations are in `decay-calibration/`.
The `rastrigin_cloning_calibration` profile changes only
`calibration_jitter_amplitude`, leaving kinetic position diffusion fixed. Both
sweeps preserve baseline seed addresses and fixed fit windows. Their empirical
decay remains a diagnostic, with sampling floors and extinction outcomes retained.
The baseline already decreases in all canonical alive-law endpoint comparisons;
these sweeps compare parameters and do not justify changing the canonical preset.

`gas-decay analyze REPORT.json` recomputes rates and endpoint comparisons from
recorded native trajectories. This supports correcting an observable without
resampling the engine. `--require-decrease` checks mean endpoint decline and the
full configured seed count; it does not assert per-step monotonicity or a theorem.
The original eight seeds and the separate 32 fresh seeds are distinguished in
the calibration summary.

From `algorithmic-gas/`:

```sh
cargo run --offline -p algorithmic-gas-benchmarks --bin gas-convergence -- config --output proof-validation/config.json
cargo run --offline -p algorithmic-gas-benchmarks --bin gas-convergence -- run proof-validation/config.json --output outputs/convergence/report.json --strict
cargo run --offline -p algorithmic-gas-benchmarks --bin gas-convergence -- inventory --output outputs/convergence/inventory.json
cargo test --offline -p algorithmic-gas-benchmarks --test convergence_framework --test convergence_coefficients --test convergence_continuity --test convergence_decay --test convergence_smoothing --test convergence_kinetic --test convergence_lyapunov --test convergence_cloning --test convergence_selection --test convergence_boundary --test convergence_validation
```

The default matrix has 96 runs: two populations, four landscapes, six parameter
profiles, and two independent seeds. Each run completes 24 steps, and every
recorded step is diagnosed. Increase `walkers`, `dimensions`, `seeds`, `steps`,
`kinetic_samples`, or `cloning_samples` in the JSON. The `runs` field accepts exact
`RunConfig` objects to sweep any engine parameter; a nonempty list replaces the
Cartesian matrix. Unknown fields, profiles, invalid shapes and numerical inputs
are rejected.

Profiles cover dispersed and collapsed starts; reward/diversity exponents and
gate saturation; denominator, positivity and distance floors; clone epsilon;
timestep, friction, both diffusions, cap, jitter and restitution; and separate
measurement and cloning bandwidths. These are finite probes of parameter space.
They do not establish a global optimal tuning rule.

The JSON retains configurations, seeds, all trajectories, exact conditional
cloning predictions and residuals, first/final stage diagnostics, source-linked
kinetic constants and independent Monte Carlo experiments. Keystone constants
are a separate calculation for the fixed canonical quadratic reference on
`(-2,2)^d`, with the physical reward modulus `2*sqrt(d)`; they are not assigned
to other landscapes or altered profiles. Extreme rates and population thresholds
are stored as natural logarithms to avoid silent underflow or overflow.

`--strict` fails after writing the complete report if an exact check, statistical
diagnostic, or engine execution fails. An extinction is a recorded outcome of
the killed chain. Statistical checks retain standard errors and a specified
threshold; acceptance does not imply proof of a universal statement.

From the repository root, regenerate inventories after changing the chapters:

```sh
uv run python algorithmic-gas/proof-validation/generate_inventory.py
uv run python algorithmic-gas/proof-validation/generate_cloning_inventory.py
uv run python algorithmic-gas/proof-validation/summarize_validation.py algorithmic-gas/outputs/convergence/report.json --output algorithmic-gas/outputs/convergence/summary
uv run python algorithmic-gas/proof-validation/summarize_decay_calibration.py algorithmic-gas/outputs/convergence/calibration
```

Inventories retain source hashes, exact statement text, source lines, every
displayed expression and relevant inline quantity (including formulas spanning
multiple lines), hypotheses, and sampling-law
scope. Every formal item receives a disposition. An item without a matching
implemented diagnostic remains explicitly unevaluated; inventory membership
does not count as validation. Topological claims, global infima, unspecified
prefactors and existential hypotheses remain analytical obligations.

The repaired Sasaki distance estimate separates positional change under a fixed
companion law from structural change. Its measurement-support term is not a
permanent death penalty. Revival diagnostics examine the copied live-donor
coordinates before the next kinetic boundary. The suite also preserves the
critical `h=2` kinetic degeneracy and the exact four-walker positive variance
drift, so it cannot incorrectly validate unconditional monotone contraction.

The complete-step reset experiments include retained dead coordinates of
magnitude `1e3`, `1e6`, and `1e9`. Mandatory revival overwrites them using current
live sources before jitter, collision, and kinetics. The computed `M_x`, `M`,
integrable logarithmic barrier norm, and `M_b` use the eligible source domain;
they impose no bound on those dead positions. Independent completed swarms
check the all-slot moment and the exact Gaussian position moment conditional
on the recorded post-cloning stage. These ensembles reuse `cloning_samples`.
One ensemble begins with a single survivor. The optimal-transport diagnostics
use alive-normalized centered measures, and the corrected coercivity lemma
retains the minimum over couplings; a supplied admissible coupling gives an
upper bound. Unequal alive counts and independently permuted copies are
separate regressions. Storage indices carry no intrinsic walker identity.

Additional fixtures enumerate nonempty uniform supports, native/reference noise
moments and concentration coefficients, finite transport and status coupling,
geometric clusters, rescale/log-gap constants, stability thresholds, retained
selection pressure, and complete measurement patterns. The weighted drift
assembly fixtures use supplied finite conservative and killed operators;
they validate the coefficient algebra under those hypotheses and do not assign
their rates to the gas engine. Conditional companion frequencies reuse
`kinetic_samples`; each report records its sample count and sampling scope.

Boundary smoothing computes the corrected ball/heat density and perimeter
prefactors across dimensions and noise scales. Transfer to the normalized swarm
metric includes `sqrt(N)` and an explicit inverse-distance factor for the
projection. Generic threshold-revival probes retain their sufficient parameter
condition separately from the native unconditional dead-recipient branch.

`extended.json` runs 648 swarms over populations 4/16/64, dimensions 1/2/4,
three seeds, four landscapes and six profiles, with 48 steps per run. Use
`--release` for this matrix. The summary script writes a scientific trajectory
plot, a compact Markdown summary, and a `constants.json` ledger retaining each
value's original scope and report path, including logarithmic rates.

## Individual source equations

`gas-estimates` exercises separately quoted estimates in chapters 1–3. A theorem
label does not grant credit to its other formulas. Each record retains the exact
source expression, fixture, hypotheses, measured comparisons and scope. The
coverage catalog lists every equation independently; unmatched source quotes,
failed hypotheses and missing estimates prevent a completion claim.

```sh
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-estimates -- --samples 1024 --output outputs/convergence/estimates.json --strict --require-complete
cargo test --offline -p algorithmic-gas-benchmarks --test convergence_estimates --test convergence_estimates_algebra --test convergence_estimates_contracts --test convergence_estimates_scalar --test convergence_estimates_cloning_scalar --test convergence_estimates_framework --test convergence_estimates_kinetic --test convergence_estimates_cloning
```

The runner writes its evidence before returning an error. `--strict` checks
numerical comparisons, recorded hypotheses and exact source binding.
`--chapters 1,2` selects specific chapters. If one phase fails, the report keeps
the other completed phases and records the error; either strict flag rejects it.
`--require-complete` additionally rejects every required expression without
individual executed evidence, including unexecuted analytic obligations. A
finite experiment remains scoped to its recorded hypotheses.

The catalog retains the chapter 2 TLDR's explicit reference to chapter 8's
mean-field recurrence as an unevaluated external result. It cannot receive
finite-population numerical credit or stand in for this chapter's coupled
Markov transition. Every locally asserted bound and constant remains required.

Schema version 2 stores input fixtures, numerical comparisons and comparison
sets once. Each formula record's `inputs.fixture`, `checks.set`, and
`hypothesis_checks.set` reference those tables without discarding any values.
The comparison ledger uses these same shared tables, so its formula records
retain source ownership without repeating large input arrays for every check.
From the repository root, generate a compact ledger and the exact missing list:

```sh
uv run python algorithmic-gas/proof-validation/summarize_estimates.py algorithmic-gas/outputs/convergence/estimates.json --output algorithmic-gas/outputs/convergence/estimates-summary
```

The exact conditional-law fixtures include native sampled normalization and
clipping, accepted component collision energy, partial alive support and
singleton revival. Large Keystone thresholds use logarithms and compressed
event-law calculations. Those calculations retain their population hypotheses
and are distinct from native finite-swarm simulations. Directed integer
interval certificates retain every weighted measurement pattern in the chapter's
numerical expansion examples.

Smooth-barrier checks use the stated bump-integral cutoff on a declared ball
collar, with transition derivatives and a geometric boundary sequence. Native
uniform and Gaussian innovation checks retain their statistical tolerances.
State-space contracts compare normalized marked transport against exhaustive
permutation minima and verify the auxiliary continuous Langevin generator on
polynomial moments; the native discrete BAOAB has separate chapter 2 evidence.
The phase-space transport checks include unequal alive counts, exact barycentric
decomposition and positive-definite cross weights close to their coercivity
boundary. Boundary reward margins pass through the actual native Global,
logistic and product-fitness APIs. Conditional safe-population calculations
retain their independent-death stage and Gaussian tail hypotheses.

## Complete Chapter 3 validation

Run the source-expression gate and the independent measured-inequality matrix separately:

```sh
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-estimates -- --chapters 3 --samples 1024 --output outputs/convergence/chapter03-complete.json --strict --require-complete
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-cloning-validation -- --samples 1024 --output outputs/convergence/chapter03-empirical.json --strict
uv run python algorithmic-gas/proof-validation/summarize_estimates.py algorithmic-gas/outputs/convergence/chapter03-complete.json --output algorithmic-gas/outputs/convergence/chapter03-complete
uv run python algorithmic-gas/proof-validation/summarize_cloning_empirical.py algorithmic-gas/outputs/convergence/chapter03-empirical.json --output algorithmic-gas/outputs/convergence/chapter03-empirical --plot
```

Cargo commands run from `algorithmic-gas`; Python summary commands run from the
repository root. `--compact --samples 64` provides a small empirical smoke run.
The complete matrix uses populations 4/16/64, dimensions 1/2, four landscapes,
and canonical, changed-selection, half-alive and singleton-alive proposals.
Each replicate uses two independently seeded native engines. The report retains
actual configurations, entering masks, sampled fitness, acceptance probabilities,
component energy, measured moments, predictions, residuals and standard errors.
Conditional means and upper bounds remain distinct. A rejected equality or
upper bound fails strict mode after saving the evidence.

The balanced-cloud experiments compare measured error-weighted activity directly
against the proved canonical Keystone coefficient, using normalized empirical
laws and no intrinsic walker identities. The general cases compare positional
reset, barycenter covariance, component velocity dissipation, boundary integrals,
transport expansion and the complete two-swarm weighted proposal drift. The
boundary observable is the declared global C2 function `phi(x)=|x|^2`; zero
conditional pressure realizations are retained and do not certify a positive
state-uniform boundary rate. Proposal drift does not certify external
stationary-law or survival-conditioned QSD results referenced by the chapter.

## Chapters 4–6 from retained experiments

`gas-stored-validation` reads the existing extended matrix, independent/shared
decay trajectories and Chapter 3 proposal replicas. It never advances an engine.
The next convergence chapters are Wasserstein control, kinetic contraction and
convergence; the separate single-particle foundation chapter is outside this
sequence.

From the repository root, inventory the current source first:

```sh
python3 algorithmic-gas/proof-validation/generate_stored_inventories.py
```

From `algorithmic-gas`, run all three analyses or select `--chapter 4`, `5` or `6`:

```sh
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-stored-validation -- --chapter all --output outputs/convergence/chapters04-06-stored.json --strict
```

From the repository root, write source- and input-hashed summaries and plots:

```sh
uv run python algorithmic-gas/proof-validation/summarize_stored_validation.py algorithmic-gas/outputs/convergence/chapters04-06-stored.json --output algorithmic-gas/outputs/convergence/chapters04-06-stored --plot
```

Predictions are reconstructed from retained configurations and inputs. The
analysis distinguishes finite native moment bounds, exact geometric identities,
certified quadratic controls and empirical selected/killed trajectories. Stored
sample standard errors are retained; linear Gaussian quadratic observables use
their exact analytic variance across independent seed pairs. Embedded copies of
the decay suites do not count as additional independent experiments.

`--strict` rejects numerical discrepancies and invalid source bindings after
saving the evidence. `--require-complete` additionally rejects missing required
source expressions. A passed finite comparison credits only its complete quoted
expression, not all formulas under its theorem label. Missing capped-stage
coupling costs, exact-SDE/timestep references, survival-conditioned laws,
minorization, entropy and functional-inequality certificates stay explicit.
The canonical singleton continuation and a theorem's below-two cemetery
convention describe different killed kernels.

## Einstein–Hilbert centered operator estimates

`gas-eh-proof` extends the centered variance and signed-drift strategy to the
native Einstein–Hilbert preset. It integrates uniform mutual matchings with
both edge-inclusion probabilities, including their covariance contribution.
It retains both graph/Boris kicks, their shared frozen weights, the actual
second-kick/noise correlation, and the period-20 cloning clock. The formulas
and hypotheses are proved in `27_einstein_hilbert_bounds.md`, labels
`thm-eh-centered-cloning`, `thm-eh-centered-transition`,
`cor-eh-period-budget`, and `thm-eh-marked-kinetic-limit`.

From `algorithmic-gas/`:

```sh
cargo test --offline -p algorithmic-gas-benchmarks --test convergence_einstein_hilbert --test einstein_hilbert_gas
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-eh-proof -- proof-validation/einstein-hilbert.json outputs/convergence/einstein-hilbert/report.json
cargo run --offline --release -p algorithmic-gas-benchmarks --bin gas-eh-proof -- proof-validation/einstein-hilbert-long.json outputs/convergence/einstein-hilbert/long-report.json
```

From the repository root:

```sh
uv run python algorithmic-gas/proof-validation/summarize_einstein_hilbert.py algorithmic-gas/outputs/convergence/einstein-hilbert/report.json algorithmic-gas/outputs/convergence/einstein-hilbert/long-report.json --output algorithmic-gas/proof-validation/einstein-hilbert-results --figure docs/source/2_fractal_gas/convergence_program/figures/eh_operator_validation.png
```

The retained sweep covers 40,800 native steps and 768 checkpoint replicas,
independent within each ensemble, including N=500 at the reference settings. Each replica subtracts
its own exact conditional expectation; uncertainty is estimated across
independent completed replicas, not across dependent walkers. Exhaustive
matching regressions verify the covariance algebra separately. The numerical
report distinguishes successful operator checks from the unresolved uniform
long-time drift and evolving-neighborhood closure. The detailed JSON reports
live under the ignored `outputs/` directory; configurations, compact results,
provenance and the plot remain with the source.

`gas-eh-stationarity` retains the signed position–velocity term through every
20-step cycle and stores centered position samples, velocities, final curvature,
volume and reward fields, and the graph consumed by the kinetic stages. Its
cycle residual is the sum of two successive martingale differences, with the
conditional cloning variance bounded by `EH.M4a` and the OU variance by `EH.K3`.
The complete time-sampling estimate is `cor-eh-signed-cycle`; a largest observed
moment is not substituted for that theorem's expected moment bounds.

From `algorithmic-gas/`, the longer experiments are reproduced by:

```sh
cargo build --offline --release -p algorithmic-gas-benchmarks --bin gas-eh-stationarity
for n in 32 128 500; do
  for seed in 7 1729 991; do
    target/release/gas-eh-stationarity "$n" "$seed" 50000 500 "outputs/convergence/einstein-hilbert/stationarity/n${n}-s${seed}.json"
  done
done
for seed in 7 1729 991; do
  target/release/gas-eh-stationarity 32 "$seed" 500000 5000 "outputs/convergence/einstein-hilbert/extended/n32-s${seed}.json"
done
```

From the repository root, summarize each common-horizon ensemble:

```sh
uv run python algorithmic-gas/proof-validation/summarize_eh_stationarity.py algorithmic-gas/outputs/convergence/einstein-hilbert/stationarity --output algorithmic-gas/proof-validation/einstein-hilbert-results --figure docs/source/2_fractal_gas/convergence_program/figures/eh_stationarity_validation.png
uv run python algorithmic-gas/proof-validation/summarize_eh_stationarity.py algorithmic-gas/outputs/convergence/einstein-hilbert/extended --output algorithmic-gas/proof-validation/einstein-hilbert-results/extended --figure docs/source/2_fractal_gas/convergence_program/figures/eh_extended_validation.png
```

The distribution comparison uses 51 fixed projection directions and equal
quarter-horizon windows. RMS-normalized shape is an additional diagnostic,
kept distinct from the unscaled centered distribution. Curvature, volume and
reward quantiles pool dependent time snapshots only for descriptive comparisons;
the plot's spread and energy bands show the range across the three seeds, not
confidence intervals. `lem-eh-native-curvature-balance` proves the actual
curvature balance and its scale transformation, checked against native geometry.

## Fresh native experiments for Chapters 4–6

The experiment runner extends the stored analyses with actual paired cloning
outputs, dense viscous reference trajectories, capped kinetic contraction,
exact-SDE/timestep references, terminal mismatch events and independent
survival-conditioned ensembles. The primary distances minimize over swarm
permutations and normalize by population size. Stage coupling assignments are
saved as representatives of a probability coupling; they do not introduce
persistent walker labels. Mandatory revival discards dead positions.

From `algorithmic-gas/`:

```sh
cargo test --offline -p algorithmic-gas-benchmarks --test convergence_experiments --test convergence_experiments_chapter04 --test convergence_experiments_chapter05 --test convergence_experiments_chapter06
cargo build --offline --release -p algorithmic-gas-benchmarks --bin gas-proof-experiments
uv run --no-sync python proof-validation/generate_stored_inventories.py
target/release/gas-proof-experiments run proof-validation/experiment-configs/chapters04-06-smoke.json --chapter all --output outputs/convergence/chapters04-06-experiments/smoke --strict
```

For the full matrix, run the three commands concurrently in separate terminals:

```sh
target/release/gas-proof-experiments run proof-validation/experiment-configs/chapters04-06-full.json --chapter 4 --output outputs/convergence/chapters04-06-experiments/full/chapter04 --strict
target/release/gas-proof-experiments run proof-validation/experiment-configs/chapters04-06-full.json --chapter 5 --output outputs/convergence/chapters04-06-experiments/full/chapter05 --strict
target/release/gas-proof-experiments run proof-validation/experiment-configs/chapters04-06-full.json --chapter 6 --output outputs/convergence/chapters04-06-experiments/full/chapter06 --strict
uv run --no-sync python proof-validation/summarize_experiments.py outputs/convergence/chapters04-06-experiments/full/chapter04 outputs/convergence/chapters04-06-experiments/full/chapter05 outputs/convergence/chapters04-06-experiments/full/chapter06 --output outputs/convergence/chapters04-06-experiments/full/summary
```

Every output directory must be empty. Use a new path for a new run. Native
`RunArchive<f64>` and resumable `Checkpoint<f64>` records use lossless gzip CBOR;
raw experiment plans and exact Brownian reference inputs use gzip JSON.
An append/fsync journal commits each completed chunk. Index snapshots every
256 chunks and at completion avoid quadratic rewrites in large ensembles.
Opening an interrupted dataset replays the journal, retaining completed chunks.
SHA256 checksums, native configurations, provider identifiers, seed rules,
source inventories, complete theorem text and implementation snapshots are
retained for later independent analysis. Full recording includes intermediate
populations, native donor/component plans, evaluated forces and all noise draws.

```sh
target/release/gas-proof-experiments verify outputs/convergence/chapters04-06-experiments/full/chapter04 --deep
target/release/gas-proof-experiments reanalyze outputs/convergence/chapters04-06-experiments/full/chapter04 --output outputs/convergence/chapters04-06-experiments/reanalyzed-chapter04
```

`verify --deep` checks checksums, decompresses and validates every native record.
`reanalyze` recomputes native conditional cloning identities and moments from
saved stage data without advancing an engine. For additional experiments, use
`ArchiveStore::load_archive`, `load_checkpoint` and `load_json`. Restoring a
checkpoint into the same configured native engine preserves its step-addressed
random stream and donor memory; a regression compares the next native transition
before and after compressed checkpoint restoration.

The exact source coverage ledger retains all expressions not checked by the
finite experiments. Empirical law decline does not identify a QSD eigenvalue,
a joint-law LSI constant, a global sensitivity constant or a population-uniform
minorization coefficient. The Chapter 6 analytic Gaussian density and barrier
specializations state their domains explicitly; their finite-N positivity
certificate is separate from a uniform mixing rate. Passing comparisons and
remaining analytic obligations are both retained in the reports.

For quadratic timestep errors below ordinary Monte Carlo resolution, the native
Gaussian cubature experiment integrates every quadratic observable exactly:

```sh
cargo build --offline --release -p algorithmic-gas-benchmarks --bin gas-kinetic-cubature
target/release/gas-kinetic-cubature outputs/convergence/kinetic-cubature
```

The 96 deterministic weighted nodes integrate the 48-dimensional shared Gaussian
input through native uncapped BAOAB at steps 0.04, 0.02 and 0.01. Native archives,
Brownian bridge inputs, node weights and checkpoints are retained. The comparison
has zero sampling uncertainty for second moments and squared strong errors.
The fixed-cap kernel continues to use its separately proved discrete bound.
Use `summarize_experiments.py --cubature-review PATH` to include a reviewed,
checksum-bound cubature report alongside the three chapter datasets.

The completed 2026-10-04 full matrix is described in
[Chapters 4–6 results](chapters04-06-full-results.md), with detailed constants,
measured rates, exact empirical-law transport and the reusable data catalog.

The structural landscape extension keeps dimension, regional force/reward
profiles, slow-zone transfer, Gaussian excursions and unbounded tails explicit:

```sh
cargo build --offline --release -p algorithmic-gas-benchmarks --bin gas-structural-landscape --bin gas-structural-mixing
target/release/gas-structural-mixing FRESH_REGISTER.json [PRIMITIVES.json]
target/release/gas-structural-landscape CONFIG.json EMPTY_DATASET
target/release/gas-structural-landscape CONFIG.json EMPTY_SELECTED_DATASET selected
uv run --no-sync python proof-validation/rastrigin_regional_interval.py --output FRESH_CERTIFICATE.json
uv run --no-sync python proof-validation/summarize_structural_landscape.py DATASET EMPTY_DERIVED_DIRECTORY
```

`convergence_structural_rates` composes proved regional discrepancy inequalities,
reports the decay rate and additive error floor, and rejects missing or failed
closure conditions. An empirical transition matrix does not supply these
inequalities. `convergence_structural_mixing` evaluates Chapter 6a's full
primitive weighted/unweighted registers and their kernel/profile gates.
`convergence_structural_tails` retains the full Gaussian and polynomial tails;
cutoffs are analysis parameters. Actual nonquadratic native trajectories retain
each force query, stage, noise draw and checkpoint. In selected mode, the kinetic
bound starts at the actual state prepared by copying and complete collisions.
Error decline, error growth, local residence, and analytic hypothesis failures
remain distinct measurements. The global bounded-force perturbation is a rate
to an explicit remainder; a within-well rate is conditional on both kick queries.

The existing root-core landscape parameterization is evaluated directly in Rust,
including actual jitter, diffusion, collision cap and viscosity normalization:

```sh
cargo build --offline --release -p algorithmic-gas-benchmarks --bin gas-landscape-phase --bin gas-landscape-phase-probes
target/release/gas-landscape-phase FIXTURES.json RETAINED_FRAMES.json FRESH_REVIEW.json
target/release/gas-landscape-phase-probes EMPTY_DATASET 32
uv run --no-sync python proof-validation/landscape_phase_interval.py crates/benchmarks/fixtures/landscape-phase-python.json FRESH_INTERVAL_CERTIFICATE.json
```

`regional_bound` retains the original source formula.
`tightened_regional_bound` uses exact Gaussian variance and a finite analytic
root enclosure. Its multiplier describes one-step source variance; iteration
requires the stated source-pressure and phase-flux estimates.
`RegionalErrorBudget::optimize_weights` searches positive regional weights and
then verifies the resulting column bound, retaining the conversion back to
unweighted error. Uniformity in N is a hypothesis on the full regional system.

The [completed landscape validation](landscape-tightening.md) links the five new
deeply verified datasets, complete local nonlinear audit, exact empirical-law
endpoints and all regional probe measurements. Derived analyses advance no
native engine. The dataset catalog retains checksums for reports, execution
provenance, historical executed sources and the current reviewed proof snapshot.


## Completion experiments and global tail estimates

`gas-landscape-axioms` evaluates the actual native Rastrigin gradient along
long segments in dimensions 1, 2, 4 and 8. Its global bounded-force certificate
uses `M_d=20*pi*sqrt(d)`, `L_grad=40*pi*sqrt(3*d)` and
`kappa_grad=400*pi^2*d`, independently of population size. The immutable
[gradient dataset](../outputs/convergence/estimate-completion-20261004/nonquadratic-gradient-axiom/report.json)
contains 144 segment cases, 576 comparisons and 1,179,792 native gradient queries.
All comparisons pass; these queries are distinct from complete engine updates.

The coordinate-variance refinement proves `L_grad=sqrt(d)` and
`kappa_grad>=1207`, with the lower constant independent of dimension and
population. Run `target/release/gas-landscape-axioms EMPTY_DIR short` to evaluate
that certificate. The separate [short-segment dataset](../outputs/convergence/estimate-completion-20261004/nonquadratic-short-gradient-axiom/report.json)
contains another 144 cases and 1,179,792 native gradient evaluations; all 576
comparisons pass. Both certificates retain their complete unbounded-domain
proofs and exact segment integrals.

The native regional runner now accepts an optional exact JSON parameter profile:

```sh
target/release/gas-structural-landscape MATRIX.json EMPTY_DIR selected NATIVE_PARAMETERS.json
python3 proof-validation/run_completion_parameters.py
```

Required profile fields are timestep, friction, velocity cap, position diffusion,
clone jitter, restitution, reward/diversity exponents and cloning bandwidth.
The committed seven-profile matrix retains every actual native stage, innovation,
checkpoint, configuration and checksum. Physical-time rates use each run's actual
timestep. The weak-selection profile has positive exponents `1e-5` and cloning
bandwidth 3 on an unbounded conservative state space.

[Section 20](../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md#sec-slc-global-selected-moments)
proves its normalized moment and tail rate, including the actual selected-source
incoming load and finite donor history. The closure is checked from global
feature, fitness and force envelopes; sampled moments are measurements of the
bound rather than replacements for those hypotheses. The complete law-mixing
rate retains its regional transfer conditions.

The [completion catalog](estimate-completion.md) links every chapter inventory,
the strict source-expression reports, the seven parameter profiles, empirical
error-decrease measurements and the statement-level analytic obligations.
The profiles add 64,512 actual native updates and 96,768 comparisons with no
bound violations; their 8,204 lossless archives are deeply verified. Exact
selected-source integration checks every donor/gate contribution independently
of whether a rare clone event occurred in the sampled trajectory.

[Chapter 5 completion results](../outputs/convergence/chapter05-completion-20261004/results.md)
contain 2,760 source-scoped checks over 11,520 retained native left stages,
with separate per-step independent-seed uncertainty, native transient moments
and joint Gaussian/cap density bounds.

## Chapters 7–9: equilibrium models, marked laws and population rates

The [Chapter 7–9 catalog](chapters07-09-results.md) links the individual source
expression tables, actual native experiments and separate reference-law checks.
`generate_population_inventories.py` retains every expression and its exact
chapter source. Global QSD existence, limiting laws and stationary functional
inequalities retain their stated analytic hypotheses.

```bash
cargo build --offline --release -p algorithmic-gas-benchmarks \
  --bin gas-chapter07-completion --bin gas-chapter07-poisson \
  --bin gas-chapter08-completion --bin gas-chapter09-completion \
  --bin gas-native-kinetic-resonance --bin gas-population-independent-review \
  --bin gas-native-horizon-variance --bin gas-native-measurement-replacement
target/release/gas-chapter07-completion EMPTY_OUTPUT COMPLETE_NATIVE_DATASET
target/release/gas-chapter07-poisson EMPTY_OUTPUT
target/release/gas-chapter08-completion EMPTY_OUTPUT 64 512
target/release/gas-chapter09-completion EMPTY_OUTPUT COMPLETE_CHAPTER08_DATASET
target/release/gas-native-kinetic-resonance EMPTY_OUTPUT
target/release/gas-population-independent-review COMPLETE_NATIVE_DATASET EMPTY_OUTPUT
target/release/gas-native-horizon-variance EMPTY_OUTPUT \
  WEAK_SELECTION_DATASET FRICTION_2_DATASET CAP_JITTER_DIFFUSION_DATASET
target/release/gas-native-measurement-replacement COMPLETE_CHAPTER08_DATASET EMPTY_OUTPUT 512
uv run --no-sync python proof-validation/population_scaling_review.py \
  COMPLETE_CHAPTER08_DATASET EMPTY_OUTPUT --bootstrap 2000
```

The native mean-field suite varies dimension 1/2/4 and population 8/32/128 on
quadratic, Rastrigin well/saddle/tail and absorbing/revival cases. Each seed is
one independent complete native update from the same complete prepared input.
Independent rooted laws sample measurement marks before the nonlinear fitness
map, actual weighted incoming components, shared rotations, jitter and complete
kinetics. Root capacities cause an error rather than conditional truncation.

Conditional variance, bias and MSE are compared with their own bounds and
independent sampling uncertainty. `population_scaling_review.py` resamples whole
observable vectors and stores every bootstrap address, measured exponent and
reference sampling floor. Probability normalization is used throughout;
storage addresses serve only replay and recording.

## Chapters 10–12: entropy, mass/transport and exchangeability

The [Chapter 10–12 catalog](chapters10-12-results.md) links the complete
source-expression tables and retained experiments. These chapters contain
61 formal items and 369 expressions. A numerical binding identifies its own
formula and operands; definitions, global hypotheses and limiting statements
have separate dispositions.

```bash
uv run --no-sync python proof-validation/generate_entropy_inventories.py
cargo build --offline --release -p algorithmic-gas-benchmarks \
  --bin gas-chapter10-completion --bin gas-chapter11-completion \
  --bin gas-chapter12-completion --bin gas-entropy-native-trajectories \
  --bin gas-chapter10-landscape-matched
target/release/gas-chapter10-completion EMPTY_CHAPTER10_OUTPUT
target/release/gas-chapter10-landscape-matched EMPTY_MATCHED_REFERENCE_OUTPUT
target/release/gas-entropy-native-trajectories EMPTY_NATIVE_OUTPUT 24 32
target/release/gas-chapter11-completion EMPTY_CHAPTER11_OUTPUT COMPLETE_NATIVE_DATASET
target/release/gas-chapter12-completion EMPTY_CHAPTER12_OUTPUT COMPLETE_CHAPTER08_DATASET 8
uv run --no-sync python proof-validation/chapter10_completion_ledger.py \
  COMPLETE_CHAPTER10_OUTPUT EMPTY_CHAPTER10_LEDGER
uv run --no-sync python proof-validation/chapter11_completion_ledger.py \
  COMPLETE_CHAPTER11_OUTPUT EMPTY_CHAPTER11_LEDGER --bootstrap 1000 \
  --primary-native-horizon 32
uv run --no-sync python proof-validation/chapter11_exact_marginal_transport.py \
  COMPLETE_CHAPTER11_OUTPUT EMPTY_POOLED_TRANSPORT_OUTPUT --bootstrap 200
uv run --no-sync python proof-validation/chapter11_nonquadratic_transfer.py \
  COMPLETE_MATCHED_REFERENCE_OUTPUT EMPTY_MATCHED_TRANSFER_OUTPUT
uv run --no-sync python proof-validation/chapter12_completion_ledger.py \
  COMPLETE_CHAPTER12_OUTPUT EMPTY_CHAPTER12_LEDGER \
  --nonquadratic-reference COMPLETE_CHAPTER10_OUTPUT
uv run --no-sync python proof-validation/verify_entropy_archives.py \
  COMPLETE_EXPERIMENT_ROOT NEW_AUDIT_JSON
```

The native trajectory matrix uses six quadratic/Rastrigin well, saddle, tail
and absorbing/revival profiles, dimensions 1/2/4, populations 8/32/128 and two
distinct initial laws. Each side has 24 independently seeded trajectories of
up to 32 complete native updates. Complete physical states, actual masks,
source graphs, shared component rotations, forces and Gaussian innovations
are retained. Extinct trajectories have explicit cemetery continuation;
the killing transition is reconstructed from the first zero alive mass.

Chapter 10 calculates entropy and Fisher terms for identified continuous
reference densities and nonquadratic Gibbs-relative densities. The separable
nonconvex LSI estimate pays the one-coordinate perturbation cost before
tensorizing across dimension and population. Chapter 11 measures native
alive mass, exact projected one-dimensional transport and explicitly declared
finite-grid Hellinger laws, with uncertainty from whole trajectories.
Chapter 12 transports all realized random fields when replaying permutations,
then checks physical population outputs and exact empirical-mixture sampling.

These experiments retain their generator, law and normalization. Atomic
empirical measures, finite-grid pushforwards, continuous Gibbs laws and the
full native configuration-space law have different entropies. Native QSD,
LSI and infinite-limit claims retain the analytic hypotheses in the chapters.
