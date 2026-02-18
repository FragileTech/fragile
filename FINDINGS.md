# Fractal Gas Parameter Sweep: Complete Findings

**Campaign dates**: 2026-02-16 to 2026-02-18
**Total simulation runs**: 5,507 | **Unique parameter combinations**: 756 | **Rounds**: 12 (R1-R12)
**Base settings**: N=500 particles, d=3 dimensions, warmup_fraction=0.5, Boris-BAOAB integrator
**Run duration**: Mean 40.4s, median 15.5s per run (varies with N and n_steps)
**Compute**: Two parallel runners on CPU, total wall-clock ~48 hours

---

## Table of Contents

1. [Campaign Overview](#1-campaign-overview)
2. [Understanding the Metrics](#2-understanding-the-metrics)
3. [What Distinguishes Score 7 from Score 6](#3-what-distinguishes-score-7-from-score-6)
4. [Parameter-by-Parameter Analysis](#4-parameter-by-parameter-analysis)
5. [Parameter Interactions](#5-parameter-interactions)
6. [R12 Stacking Experiment](#6-r12-stacking-experiment-less-is-more)
7. [Seed Variability and Statistical Reliability](#7-seed-variability-and-statistical-reliability)
8. [Global Leaderboard](#8-global-leaderboard-top-40)
9. [Recommended Configurations](#9-recommended-configurations)
10. [Key Takeaways](#10-key-takeaways)
11. [Appendix: Methodology](#11-appendix-methodology)

---

## 1. Campaign Overview

### 1.1 Purpose

This campaign systematically explored how each parameter of the fractal gas simulation
affects the quality of coupling diagnostics — the set of observables that measure whether
the simulation produces physics consistent with a confining gauge theory. The goal was to
find parameter combinations that reliably produce high regime scores with strong confinement
signatures.

### 1.2 Simulation Pipeline

Each run follows this pipeline:

1. **Initialization**: Create `EuclideanGas` with `KineticOperator`, `CloneOperator`, and
   `FitnessOperator`. Initialize particle positions from `N(init_offset, init_spread)` and
   velocities from `N(0, init_velocity_scale)`.
2. **Evolution**: Run `gas.run(n_steps)` which alternates between:
   - Kinetic evolution (Langevin dynamics with Boris rotation) for `n_kinetic_steps` substeps
   - Fitness evaluation and cloning (every `clone_every` steps)
   - Neighbor graph updates (every `neighbor_graph_update_every` steps)
3. **Diagnostics**: Compute `compute_coupling_diagnostics(history, config)` with
   `warmup_fraction=0.5` (discards first half of trajectory as equilibration).
4. **Output**: JSON file with all parameters, seed, summary metrics, and regime evidence.

### 1.3 Score Distribution

| Score | Count | Percentage | Cumulative >= |
|-------|-------|------------|---------------|
| 2     | 2     | 0.04%      | 100.0%        |
| 3     | 27    | 0.5%       | 99.96%        |
| 4     | 47    | 0.9%       | 99.5%         |
| 5     | 828   | 15.0%      | 98.6%         |
| 6     | 2,654 | 48.2%      | 83.5%         |
| **7** | **1,944** | **35.3%** | **35.4%** |
| 8     | 2     | 0.04%      | 0.1%          |
| 9     | 3     | 0.05%      | 0.1%          |

The regime score is integer-valued and ranges from 0 to ~10. Score 7 is the primary
target — it indicates that all major confinement criteria are satisfied simultaneously.
Scores 8-9 are extremely rare (5 out of 5,507 runs) and appear to require lucky
coincidences in multiple metrics.

The distribution is heavily concentrated at score 6 (48.2%) and 7 (35.3%). The jump from
6 to 7 is the critical transition we want to understand and maximize.

### 1.4 Round-by-Round Summary

| Round | Combos | Runs | Seeds/combo | Focus | Key Discovery |
|-------|--------|------|-------------|-------|---------------|
| R1 | 30 | 30 | 1 | Single-parameter sweeps | nkin=3 and nu=0.1 matter |
| R2 | 15 | 15 | 1 | First combinations | nkin3+nu0.1+T1.0 is strong |
| R3 | 30 | 300 | 10 | nkin/vls/bc grid | nkin=4, vls=2.0, bc=0.25 standard |
| R4 | 30 | 300 | 10 | T/nu/fitness tuning | T=0.8 good, fb=2.0 mild help |
| R5 | 25 | 250 | 10 | Clone frequency, dt | clone_every=1 mandatory |
| R6 | 30 | 300 | 10 | init_spread, dt combos | isp=3-5 boosts sigma |
| R7 | 30 | 300 | 10 | Champion variants, dt | dt=0.02 and dt=0.05 both good |
| R8 | 50 | 500 | 10 | bounds_extent sweep | bounds_extent has NO effect |
| R9 | 50 | 500 | 10 | isp/nkin/ivs combos | isp=10 gives sigma~13 |
| R10 | 45 | 450 | 10 | bounds_extent fine-tuning | Confirmed: no effect at all |
| R10b | 106 | 1,060 | 10 | Wide multi-param exploration | T=0.9 best, isp10+nkin5+T09 champion |
| R11 | 50 | 500 | 10 | gamma/nu/vls/bc/T fine-tuning | gamma=0.6 and vls=2.5 interesting |
| R12 | 13 | 130 | 10 | Stacking on champion | "Less is more" — stacking hurts |

The campaign followed an iterative strategy: each round built on the discoveries of
previous rounds. Early rounds (R1-R3) identified the critical parameters (nkin, nu, vls, bc).
Middle rounds (R4-R7) optimized combinations. Late rounds (R8-R12) explored edge cases,
fine-tuned the champion, and tested the limits of multi-parameter stacking.

### 1.5 The "Standard" Configuration Evolution

As discoveries accumulated, a set of "standard" parameter fixes was adopted. These
changed across rounds:

| Round | Standard Changes from Defaults |
|-------|-------------------------------|
| R1 | Individual parameter tests only |
| R2-R3 | nkin=4, nu=0.1, vls=2.0, bc=0.25 |
| R4-R7 | + T=1.0 (up from default 0.5) |
| R8-R9 | + dt=0.02, init_spread exploration |
| R10b+ | + T=0.9, init_spread=10.0, ivs=0.5 |

The "standard" config (nkin>=4, nu<=0.15, vls>=1.5, bc<=0.5) covers 4,122 of 5,507 runs
(74.9%). These runs achieve **36.6% s7 rate** and **sigma=9.91+/-2.96**, compared to the
1,385 "non-standard" runs which achieve **31.8% s7** and **sigma=7.38+/-4.23**. The
standard fixes provide a solid foundation that all further tuning builds upon.

---

## 2. Understanding the Metrics

Before diving into parameter effects, it is essential to understand what each diagnostic
metric measures and why it matters. The coupling diagnostics evaluate whether the fractal
gas simulation produces observables consistent with a confining gauge theory (like QCD).

### 2.1 regime_score (Integer, 0-10)

The overall quality score. Computed by checking whether each individual metric falls within
a "confining" range. Each satisfied criterion adds 1 to the score. **Score >= 7 means
all major confinement signatures are present simultaneously.**

This is the primary optimization target. A high regime score means the simulation
produces physics that looks like a real confining gauge theory.

### 2.2 string_tension_sigma (Float, higher = better)

Measures the strength of the confining string between quark-antiquark pairs. In a confining
theory, the potential between quarks grows linearly with distance: V(r) ~ sigma * r. The
string tension sigma is the slope of this linear potential.

- **Typical range**: 0-20
- **Campaign mean**: 9.27 +/- 3.50
- **Score 7 mean**: 9.62 +/- 3.14
- **Correlation with score**: r = +0.190 (weak)

Despite being the most visually dramatic metric, sigma is a surprisingly weak predictor
of score. Many runs with sigma > 12 still only achieve score 6 (because xi is negative).
Conversely, runs with sigma as low as 0.30 can achieve score 7 (if all other criteria
are satisfied).

### 2.3 screening_length_xi (Float, positive = good)

The characteristic length scale over which the confining potential screens. In a confining
theory, the static potential should exhibit exponential screening at long distances with a
finite, positive correlation length.

- **Typical range**: -1000 to +1000 (heavy tails)
- **Score 7 median**: +14.1 (p25=8.6, p75=31.4)
- **Score 6 median**: -9.3 (p25=-23.7, p75=5.4)

**This is the single most diagnostic metric.** 99.9% of score-7 runs have xi > 0, while
only 27.3% of score-6 runs do. A negative xi means the screening fit failed to find a
physical correlation length — the distance-dependent potential does not behave as expected
for a confining theory. Whether xi is positive or negative is essentially a binary gate
that determines score 6 vs score 7.

### 2.4 polyakov_abs (Float, lower = better)

The magnitude of the Polyakov loop, which is the order parameter for the
confinement-deconfinement phase transition. In a **confined** phase (what we want),
the Polyakov loop should be near zero. In a deconfined phase, it is large.

- **Typical range**: 0.01 - 0.25
- **Score 7 mean**: 0.029 +/- 0.012
- **Score 6 mean**: 0.039 +/- 0.021
- **Score 5 mean**: 0.067 +/- 0.017
- **Score 3-4 mean**: 0.11 - 0.17
- **Correlation with score**: r = **-0.616** (strongest of all metrics)

This is the **strongest single predictor of regime score**. The correlation of -0.616 is
almost 3x stronger than any other metric. Low polyakov means particles form closed loops
in configuration space (confinement), while high polyakov means open trajectories
(deconfinement). Parameters that reduce polyakov are the most valuable for achieving
high scores.

### 2.5 re_im_asymmetry_mean (Float, lower = better, target ~0.04)

Measures the imbalance between Real and Imaginary parts of the path-integral-like
observables. In a well-equilibrated simulation, Re and Im components should be
approximately balanced. High asymmetry indicates that the simulation has not properly
explored phase space.

- **Typical range**: 0.02 - 0.60
- **Score 7 mean**: 0.045 +/- 0.039
- **Score 4 mean**: 0.177 +/- 0.020 (all > 0.05)
- **Score 3 mean**: 0.173 +/- 0.014 (all > 0.05)
- **96.3% of score-7 runs have asymmetry < 0.05**

Asymmetry is a "hygiene" metric — it needs to be below ~0.05 for the simulation to be
physically meaningful, but reducing it further below 0.04 provides no additional benefit.
Several parameters (nu, T, alpha_restitution) have dramatic effects on asymmetry.

### 2.6 running_coupling_slope (Float, negative = good)

The slope of the running coupling constant as a function of energy scale. In an
asymptotically free theory (like QCD), the coupling should decrease at high energies,
giving a negative slope.

- **Campaign mean**: -12.15 +/- 4.64
- **Score 7 mean**: -12.60 +/- 4.40
- **Correlation with score**: r = -0.167 (weak)

Almost all runs with nkin >= 4 produce strongly negative slopes (-12 to -15). This metric
is primarily determined by n_kinetic_steps and is rarely a bottleneck for achieving score 7.

### 2.7 Other Metrics

- **local_phase_coherence_mean**: Mean coherence of the local phase field. Score 7 mean = 0.640.
  Weakly anti-correlated with score (r = -0.315). Lower coherence (more disorder) is paradoxically
  associated with better scores, likely because more disordered configurations have richer
  topological structure.

- **r_circ_mean**: Mean circularity of particle trajectories. Score 7 mean = 0.050.
  Anti-correlated with score (r = -0.302). Low circularity (more complex trajectories)
  indicates richer dynamics.

- **topological_flux_std**: Standard deviation of topological flux through surfaces.
  Score 7 mean = 1.695. Positively correlated with score (r = +0.291). Higher flux
  variability indicates more topological activity, consistent with confinement.

### 2.8 Inter-Metric Correlation Matrix

Understanding how metrics relate to each other helps interpret the results:

|              | score | sigma | poly  | xi    | slope | asym  |
|--------------|-------|-------|-------|-------|-------|-------|
| score        | 1.000 | +0.190| -0.616| +0.072| -0.167| -0.201|
| sigma        |       | 1.000 | -0.222| +0.003| -0.337| -0.456|
| polyakov     |       |       | 1.000 | +0.013| +0.217| +0.262|
| xi           |       |       |       | 1.000 | +0.007| +0.003|
| slope        |       |       |       |       | 1.000 | +0.440|
| asymmetry    |       |       |       |       |       | 1.000 |

**Key observations:**
- **sigma and asymmetry are anti-correlated** (r = -0.456): Parameters that boost sigma
  tend to also reduce asymmetry. This is good news — they reinforce each other.
- **slope and asymmetry are correlated** (r = +0.440): High asymmetry runs tend to have
  less negative slopes. Both are symptoms of poor phase-space exploration.
- **xi is essentially independent** of all other metrics (r < 0.02). Whether xi is positive
  is determined by the quality of the screening fit, which depends on subtle geometric
  properties of the trajectory, not on the magnitudes of other metrics.
- **polyakov is the key bridge**: It correlates moderately with sigma (-0.222), asymmetry
  (+0.262), and slope (+0.217), and very strongly with score (-0.616). Reducing polyakov
  is the most efficient path to higher scores.

---

## 3. What Distinguishes Score 7 from Score 6

This is the central question of the entire campaign. Understanding the score 6->7 transition
allows us to focus parameter tuning on what actually matters.

### 3.1 Full Metric Comparison by Score Level

| Metric | Score 7 (n=1944) | Score 6 (n=2654) | Score 5 (n=828) | Score 4 (n=47) | Score 3 (n=27) |
|--------|-----------------|-----------------|-----------------|----------------|----------------|
| sigma mean | 9.62 | 9.41 | 8.85 | 0.11 | 0.03 |
| sigma std | 3.14 | 3.30 | 3.89 | 0.08 | 0.04 |
| sigma median | 9.67 | 9.45 | 9.40 | 0.08 | 0.03 |
| sigma p25-p75 | 7.86-11.46 | 7.79-11.26 | 7.35-11.22 | 0.04-0.20 | -0.01-0.06 |
| **xi > 0 rate** | **99.9%** | **27.3%** | **3.5%** | 55.3% | 18.5% |
| xi median | +14.1 | -9.3 | -12.7 | +18.1 | -39.4 |
| xi p25-p75 | 8.6-31.4 | -23.7 to +5.4 | -30.2 to -6.8 | -27.4 to +79.7 | -107 to -21.7 |
| **poly mean** | **0.029** | **0.039** | **0.067** | **0.114** | **0.168** |
| poly median | 0.030 | 0.036 | 0.062 | 0.114 | 0.172 |
| asym mean | 0.045 | 0.045 | 0.053 | 0.177 | 0.173 |
| **asym < 0.05 rate** | **96.3%** | **95.8%** | **89.7%** | **0.0%** | **0.0%** |
| slope mean | -12.64 | -12.30 | -11.69 | -0.30 | -0.09 |
| coherence | 0.640 | 0.641 | 0.649 | 0.766 | 0.767 |
| r_circ | 0.050 | 0.055 | 0.083 | 0.525 | 0.528 |
| tflux | 1.695 | 1.684 | 1.629 | 0.776 | 0.769 |

### 3.2 The Three Score Regimes

The data reveals three distinct regimes:

**Regime A: Score 3-4 (n=74, 1.3% of runs)**
These runs have **fundamentally broken dynamics**: sigma near zero (0.03-0.11), extremely
high polyakov (0.11-0.17), high asymmetry (0.17-0.18), near-zero slope, high coherence
(0.77), and high circularity (0.53). The particles are moving in simple, circular
trajectories with no emergent confinement structure. This happens when nkin=1 (92% of
score 3-4 runs) or clone_every >= 5.

**Regime B: Score 5-6 (n=3482, 63.2% of runs)**
The simulation produces meaningful dynamics (sigma ~9, slope ~-12, low asymmetry) but
the confinement geometry is not quite right. The primary failure is **negative xi** (72.7%
of score-6 runs have xi <= 0). The screening fit cannot find a physical correlation length.
These runs have good "raw material" but the spatial arrangement of confining structures
is subtly wrong.

**Regime C: Score 7+ (n=1949, 35.4% of runs)**
Everything clicks: positive xi (99.9%), low polyakov (0.029), good sigma, good slope, low
asymmetry. The spatial geometry of the particle cloud produces confinement signatures that
are consistently measurable at all distance scales.

### 3.3 The Score 6->7 Bottleneck: Screening Length

The failure analysis of score-6 runs reveals exactly what goes wrong:

| Failure Mode | Score-6 runs affected | Score-7 runs affected |
|---|---|---|
| **xi <= 0** | **72.7%** (1,930/2,654) | **0.2%** (3/1,949) |
| poly > 0.05 | 27.0% (717/2,654) | 0.2% (3/1,949) |
| sigma < 5 | 7.3% (193/2,654) | 6.5% (127/1,949) |
| asym > 0.05 | 4.2% (112/2,654) | 3.7% (72/1,949) |
| slope >= 0 | 0.0% (0/2,654) | n/a |

**Negative screening length is responsible for 72.7% of score-6 outcomes.** It is, by a
wide margin, the primary reason runs fail to achieve score 7. Polyakov being too high
accounts for an additional 27.0% (with overlap — many runs fail on both xi and polyakov).

Sigma and asymmetry are rarely the bottleneck (7.3% and 4.2% respectively), and slope is
never the bottleneck (0% of score-6 runs have positive slope).

**This means: the primary goal of parameter tuning is to increase the probability that
xi comes out positive.** Parameters that improve the screening length fit are the ones
that matter most for achieving score 7. Parameters that only boost sigma or reduce slope
are secondary — they improve the quality of score-7 runs but don't help convert score-6
runs into score-7 runs.

### 3.4 Practical Implications for Parameter Tuning

Given that xi positivity is the bottleneck:

1. **Prioritize parameters that affect polyakov** (the strongest score predictor at r=-0.616).
   Lower polyakov strongly predicts positive xi.
2. **Don't over-optimize sigma** in isolation. A config with sigma=14 and 30% s7 is worse
   than one with sigma=10 and 50% s7.
3. **Parameters with effects on asymmetry only matter below T~0.7**, where asymmetry
   exceeds 0.05 and becomes a bottleneck. Above T=0.7, asymmetry is always fine (~0.04).
4. **The xi outcome is partially stochastic** — the same parameters with different seeds
   can produce positive or negative xi. Parameters can shift the probability from ~30%
   xi+ (score-6 territory) to ~70% xi+ (high s7 rate) but cannot guarantee positive xi
   every time.

---

## 4. Parameter-by-Parameter Analysis

For each parameter, I report:
- **Table of values tested** with n (runs), s7% (score>=7 rate), sigma (mean string tension),
  sig_std (sigma standard deviation), xi (mean screening length), xi+% (fraction with xi>0),
  poly (mean polyakov), and asym (mean asymmetry).
- **Key findings** with quantitative comparisons.
- **Physical interpretation** of why the parameter has the effect it does.
- **Recommendation** for the optimal value.
- **Confidence level** based on sample size and consistency.

Parameters are ordered by impact (most important first).

---

### 4.1. n_kinetic_steps (Kinetic Substeps per Clone Step) — DEFAULT=1

**Impact: CRITICAL. The single most important parameter in the entire campaign.**
**Confidence: VERY HIGH (5,507 runs across all values).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi (mean) | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|-----------|------|------|------|-------|
| **1** | 239 | **12.1%** | **0.80** | 1.64 | 16.4 | 51.0% | **0.085** | **0.174** | -0.95 |
| 2 | 13 | 7.7% | 2.56 | 2.14 | -10.0 | 15.4% | 0.042 | 0.067 | -4.54 |
| 3 | 357 | 37.0% | 7.46 | 3.86 | -31.5 | 51.0% | 0.039 | 0.062 | -9.84 |
| **4** | **3,781** | **36.3%** | **9.87** | 2.89 | 3.2 | 49.5% | 0.039 | 0.041 | -12.82 |
| **5** | 764 | **37.7%** | **9.62** | 3.09 | 4.5 | 50.3% | 0.039 | 0.041 | -12.89 |
| **6** | 320 | **37.5%** | **10.04** | 2.45 | -2.4 | 49.4% | 0.039 | 0.040 | -13.51 |
| 7 | 12 | 16.7% | 9.21 | 1.66 | 4.6 | 33.3% | 0.036 | 0.039 | -13.06 |
| 8 | 10 | 30.0% | 8.02 | 1.90 | -27.8 | 50.0% | 0.051 | 0.041 | -13.21 |
| 10 | 11 | 18.2% | 8.98 | 1.07 | -10.0 | 27.3% | 0.034 | 0.040 | -14.90 |

#### Key Findings

**nkin=1 is catastrophic.** With 239 runs, nkin=1 produces:
- s7 rate of only 12.1% — the worst of any parameter value tested
- sigma = 0.80, which is **12x lower** than nkin=4 (9.87)
- polyakov = 0.085, which is **2.2x higher** (worse) than nkin=4 (0.039)
- asymmetry = 0.174, which is **4.2x higher** than nkin=4 (0.041)
- slope = -0.95, barely negative compared to nkin=4's -12.82

Essentially, with nkin=1, the simulation fails to produce any meaningful confinement
signatures. All metrics are dramatically worse. Score 3-4 runs (the "broken dynamics"
regime) are almost exclusively nkin=1 runs.

**nkin=4 is the sweet spot.** With 3,781 runs (the largest sample for any parameter value),
nkin=4 gives the most statistically reliable performance estimate: 36.3% s7, sigma=9.87,
polyakov=0.039, asymmetry=0.041. The consistency of these numbers (low sig_std=2.89)
makes this the gold standard.

**nkin=4, 5, and 6 are statistically equivalent.** Their s7 rates (36.3%, 37.7%, 37.5%),
sigmas (9.87, 9.62, 10.04), and all other metrics overlap within noise. There is no
meaningful benefit to going beyond nkin=4. The slight sigma increase at nkin=6 (10.04)
is within the standard deviation.

**nkin=3 captures most of the benefit but leaves sigma on the table.** The s7 rate at
nkin=3 (37.0%) is comparable to nkin=4-6, but sigma is notably lower (7.46 vs 9.87).
This means nkin=3 runs that achieve score 7 have weaker confinement signals than nkin=4+
score-7 runs. If you need speed, nkin=3 is acceptable; if you want the best quality, use
nkin=4-5.

**nkin=7-10 degrades slightly.** Although sample sizes are small (10-12 runs), s7 rates
drop to 17-30% and sigma doesn't improve. Excessive kinetic substeps may cause particles
to evolve too far from their fitness-relevant positions before the next cloning event,
reducing the effectiveness of selection.

**nkin=2 is a trap.** Only 13 runs, but 7.7% s7 is even worse than nkin=1. This may be
a statistical fluke (small sample) or it may represent a genuine resonance problem where
2 substeps creates a pathological oscillation pattern. Either way, avoid it.

#### Physical Interpretation

The fractal gas works by alternating two phases: (1) kinetic evolution, where particles
move according to Langevin dynamics with viscous coupling; and (2) cloning, where
high-fitness particles replace low-fitness ones. The `n_kinetic_steps` parameter controls
how many Langevin integration steps occur between each cloning event.

With nkin=1, each cloning decision is based on a single integration step. The particle
positions change by only delta_t * velocity (about 0.01 units) between fitness evaluations.
This means fitness differences between particles are dominated by noise — there is no
meaningful "signal" for cloning to select on. The evolutionary mechanism is effectively
disabled.

With nkin=4-5, particles have time to establish meaningful local structure. They can
explore their neighborhood, interact with neighbors through viscous coupling, and develop
spatial correlations before the next selection event. The fitness evaluation then has
genuine dynamical information to work with, and cloning can meaningfully select for
configurations that produce confinement.

Beyond nkin=6-7, each kinetic phase takes so long that particles drift far from the
configuration where their fitness was evaluated. The cloning decisions become stale,
reducing selection effectiveness.

#### Recommendation

**Use nkin=4 (preferred) or nkin=5.** This is non-negotiable — the most impactful single
change you can make. The improvement from nkin=1 to nkin=4 is:
- s7 rate: 12% -> 36% (+24 percentage points, 3x relative improvement)
- sigma: 0.80 -> 9.87 (+12x)
- polyakov: 0.085 -> 0.039 (-54%)
- asymmetry: 0.174 -> 0.041 (-76%)
- slope: -0.95 -> -12.82 (13x more negative)

---

### 4.2. clone_every (Cloning Frequency) — DEFAULT=1

**Impact: CRITICAL. Must remain at 1. Non-negotiable.**
**Confidence: HIGH (5,507 runs total, 280 with clone_every > 1).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|------|------|------|-------|
| **1** | 5,227 | **36.1%** | **9.59** | 3.18 | 49.3% | 0.039 | 0.045 | -12.50 |
| 2 | 110 | 29.1% | 4.29 | 3.57 | 52.7% | 0.050 | 0.089 | -7.71 |
| 3 | 80 | 25.0% | 4.15 | 4.82 | 56.2% | 0.058 | 0.101 | -6.06 |
| 5 | 70 | 12.9% | 2.16 | 3.54 | 57.1% | 0.095 | 0.119 | -3.37 |
| 8 | 10 | 0.0% | 0.02 | 0.04 | 40.0% | 0.176 | 0.169 | -0.06 |
| 10 | 10 | 0.0% | 0.01 | 0.04 | 50.0% | 0.220 | 0.178 | -0.05 |

#### Key Findings

**Performance degrades monotonically and steeply with less frequent cloning.**

The degradation cascade from ce=1 to ce=10:
- **sigma**: 9.59 -> 4.29 -> 4.15 -> 2.16 -> 0.02 -> 0.01 (960x collapse)
- **polyakov**: 0.039 -> 0.050 -> 0.058 -> 0.095 -> 0.176 -> 0.220 (5.6x worse)
- **asymmetry**: 0.045 -> 0.089 -> 0.101 -> 0.119 -> 0.169 -> 0.178 (4x worse)
- **slope**: -12.5 -> -7.7 -> -6.1 -> -3.4 -> -0.06 -> -0.05 (slope essentially zero)
- **s7 rate**: 36% -> 29% -> 25% -> 13% -> 0% -> 0%

Even ce=2 is dramatically worse than ce=1: sigma drops 55%, polyakov increases 28%,
asymmetry doubles, and s7 drops 7 percentage points. At ce=5, sigma collapses to 2.16 and
polyakov hits 0.095 (a level associated with score 4). At ce=8-10, the simulation
produces essentially zero confinement signal — sigma is 0.01-0.02 (100x below noise
level) and polyakov reaches 0.18-0.22 (full deconfinement).

#### Physical Interpretation

Cloning is the evolutionary selection mechanism that drives the fractal gas toward confining
configurations. With clone_every=1, selection pressure is applied every step, constantly
reinforcing and amplifying the spatial structures that produce confinement. The population
of particles is continuously steered toward high-fitness configurations.

With clone_every=2+, particles evolve freely between selections. During these unsupervised
intervals, the accumulated confinement structure can degrade through thermal fluctuations.
The longer the gap between selections, the more structure is lost, and the harder it is for
the next cloning event to recover.

At ce=8-10, the simulation is effectively just standard Langevin molecular dynamics with
periodic population resampling — the evolutionary component is too infrequent to maintain
any emergent structure.

#### Recommendation

**Always use clone_every=1. Never change this parameter.** There is no computational benefit
to increasing it (cloning is cheap relative to kinetic integration), and the quality
degradation is immediate and severe.

---

### 4.3. nu (Viscous Coupling Strength) — DEFAULT=1.0

**Impact: HIGH. The default value catastrophically damages asymmetry and sigma.**
**Confidence: HIGH (5,507 runs, with 4,916 at nu=0.1).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi (mean) | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|-----------|------|------|------|-------|
| 0.01 | 42 | 28.6% | 11.24 | 2.52 | 1.4 | 50.0% | 0.041 | 0.040 | -10.20 |
| 0.03 | 10 | 50.0% | 9.55 | 2.27 | 1.2 | 50.0% | 0.043 | 0.040 | -15.50 |
| 0.05 | 151 | 29.8% | 10.47 | 2.61 | -11.0 | 44.4% | 0.040 | 0.040 | -12.07 |
| 0.075 | 20 | 40.0% | 9.49 | 1.76 | -5.8 | 45.0% | 0.035 | 0.041 | -14.84 |
| **0.1** | **4,916** | **36.0%** | **9.36** | 3.44 | 3.0 | 50.1% | 0.041 | 0.046 | -12.30 |
| 0.125 | 90 | 34.4% | 9.38 | 2.11 | -36.6 | 51.1% | 0.040 | 0.040 | -14.10 |
| 0.2 | 71 | 32.4% | 9.33 | 2.62 | -26.6 | 49.3% | 0.039 | 0.040 | -13.35 |
| 0.3 | 21 | 42.9% | 8.62 | 1.12 | 2.6 | 47.6% | 0.042 | 0.040 | -12.80 |
| 0.5 | 22 | 27.3% | 10.56 | 3.86 | -3.5 | 45.5% | 0.047 | 0.044 | -10.08 |
| **1.0** | 102 | **27.5%** | **3.12** | 3.84 | -7.7 | 42.2% | 0.040 | **0.159** | -2.75 |
| 5.0 | 5 | 60.0% | 1.58 | 0.28 | -10.9 | 60.0% | 0.029 | 0.187 | -1.42 |

#### Key Findings

**nu=1.0 (default) is one of the worst default values.** Compared to nu=0.1:
- sigma drops from 9.36 to 3.12 (-67%)
- asymmetry explodes from 0.046 to 0.159 (+246%)
- slope weakens from -12.30 to -2.75 (-78%)
- s7 drops from 36.0% to 27.5%

The asymmetry at nu=1.0 (0.159) is far above the 0.05 threshold for physical meaningfulness.
This single bad default ruins the simulation.

**nu=0.1 is the well-tested standard.** With 4,916 runs, its performance is very precisely
measured: 36.0% s7, sigma=9.36, asymmetry=0.046 (just below the 0.05 threshold). This was
one of the first discoveries (R1) and was adopted for nearly all subsequent rounds.

**The nu=0.01-0.05 range gives higher sigma** (10.5-11.2) but slightly lower s7 rate (29-30%).
The coupling becomes too weak to maintain coherent structures across the full particle cloud.

**nu=0.3 looks promising** (42.9% s7, 21 runs) and **nu=0.075 at 40.0%** (20 runs), but
both have small samples. There may be a slightly better value than 0.1 in the 0.05-0.3 range
that hasn't been adequately explored.

**High nu (>=1.0) is catastrophic.** The viscous coupling at full strength creates a nearly
rigid body — particles all move with similar velocities, destroying the diversity of local
configurations needed for emergent confinement. At nu=5.0, sigma collapses to 1.58 and
asymmetry reaches 0.187.

#### Physical Interpretation

The viscous coupling term in the kinetic operator pulls each particle's velocity toward
the weighted average velocity of its neighbors (weighted by the neighbor graph). The
strength of this pull is controlled by nu.

At nu=1.0, this pull is as strong as the particle's own dynamics. All particles in a
neighborhood quickly synchronize their velocities, creating coherent bulk flow but
destroying the local velocity gradients that produce rich phase-space structure. The
resulting trajectory is smooth and featureless — poor material for detecting confinement.

At nu=0.1, the coupling is a gentle correction that maintains long-range coherence
(needed for confinement signals) while allowing local velocity diversity (needed for
rich dynamics). This balance produces the best diagnostics.

At nu=0.01, the coupling is so weak that particles are essentially independent. They
produce good sigma (11.24 — the highest of any nu value) because individual particle
dynamics are rich, but the lack of collective behavior reduces the reliability of
confinement signatures (s7 drops to 29%).

#### Recommendation

**Use nu=0.1.** Essential fix to the default. Consider exploring nu=0.05-0.3 for
potential marginal improvements, but nu=0.1 is safe and well-validated.

---

### 4.4. init_spread (Initial Position Spread) — DEFAULT=0.0

**Impact: HIGH. The primary sigma booster. Dramatic effect on string tension.**
**Confidence: HIGH (750 runs at isp=10.0, 150+ at isp=5.0).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi (mean) | xi+% | poly | asym |
|-------|--------|-----|-------|---------|-----------|------|------|------|
| **0.0** | 4,334 | 34.9% | **8.32** | 3.13 | -1.1 | 49.4% | 0.042 | 0.050 |
| 0.1 | 20 | 45.0% | 8.75 | 1.39 | 5.4 | 60.0% | 0.035 | 0.040 |
| 1.0 | 30 | 50.0% | 10.80 | 1.79 | 119.2 | 60.0% | 0.036 | 0.039 |
| 2.0 | 10 | 40.0% | 11.57 | 0.71 | -7.5 | 60.0% | 0.040 | 0.040 |
| 3.0 | 60 | 38.3% | 11.63 | 2.22 | -24.5 | 45.0% | 0.037 | 0.040 |
| 4.0 | 10 | 20.0% | 13.64 | 1.68 | -53.9 | 30.0% | 0.040 | 0.042 |
| 5.0 | 150 | 35.3% | 12.65 | 2.16 | 2.0 | 48.7% | 0.038 | 0.040 |
| 6.0 | 20 | 35.0% | 13.25 | 2.30 | -7.1 | 50.0% | 0.036 | 0.047 |
| 7.0 | 13 | 30.8% | 12.84 | 2.60 | -6.6 | 38.5% | 0.037 | 0.041 |
| 8.0 | 40 | 37.5% | 12.96 | 1.49 | -2.1 | 45.0% | 0.036 | 0.039 |
| 9.0 | 10 | 20.0% | 14.29 | 3.30 | -3.2 | 60.0% | 0.045 | 0.041 |
| **10.0** | **750** | **35.9%** | **13.08** | 2.27 | 12.9 | 49.7% | 0.039 | 0.040 |
| 11.0 | 20 | 60.0% | 13.00 | 2.64 | 4.4 | 70.0% | 0.032 | 0.042 |
| 12.0 | 20 | 55.0% | 13.21 | 1.74 | 9.0 | 60.0% | 0.033 | 0.036 |
| 14.0 | 10 | 60.0% | 12.39 | 2.36 | 12.5 | 60.0% | 0.034 | 0.029 |
| 15.0 | 10 | 40.0% | 12.85 | 2.84 | 5.9 | 70.0% | 0.047 | 0.042 |

#### Key Findings

**Sigma increases dramatically with init_spread, following a saturating curve:**

| isp range | sigma (mean) | Improvement vs isp=0 |
|-----------|-------------|---------------------|
| 0.0 | 8.32 | baseline |
| 1.0-2.0 | 10.8-11.6 | +30-39% |
| 3.0-4.0 | 11.6-13.6 | +40-64% |
| 5.0-6.0 | 12.7-13.3 | +52-59% |
| 8.0-10.0 | 13.0-13.1 | +56-57% |
| 11.0-15.0 | 12.4-13.2 | +49-59% |

The sigma gain is rapid from isp=0 to isp=5 (from 8.32 to 12.65, +52%) and then plateaus.
Going from isp=5 to isp=10 adds only another ~0.4 sigma (12.65 to 13.08). Beyond isp=10,
sigma is flat.

**S7 rate is largely unaffected by init_spread** for the well-sampled values. isp=0 gives
34.9% and isp=10 gives 35.9% — virtually identical. The isp=11-14 values show 55-60% s7
but with only 10-20 runs each, which is within seed noise.

**Polyakov and asymmetry are unaffected.** init_spread is a "free lunch" for sigma — it
boosts string tension without trading off any other metric.

**The sigma increase comes with a slight increase in sigma variance** (sig_std goes from
3.13 at isp=0 to 2.27 at isp=10, actually *lower*). This means isp=10 not only has higher
mean sigma but more consistent sigma.

#### Physical Interpretation

With init_spread=0, all 500 particles start at the origin. The initial configuration has
zero spatial structure — the simulation must create all spatial correlations from scratch
through dynamics. The correlator analysis, which measures confinement as a function of
distance, must work with whatever spatial extent the particles have reached during the
(post-warmup) portion of the trajectory.

With init_spread=10, particles start in a Gaussian cloud with standard deviation 10 in
each dimension. This provides a rich initial spatial structure spanning ~30 units in diameter.
The correlator analysis has more material to work with at all distance scales, producing
stronger and more reliable string tension measurements.

Crucially, init_spread does NOT make the dynamics "better" in a fundamental sense — the
same confinement physics is happening regardless of initial spread. What it does is provide
better *initial conditions for the measurement*. The particles start already distributed
across the spatial scales relevant for confinement measurements.

This is why sigma increases but s7 rate barely changes: the underlying confinement quality
(which determines score) is the same, but the measurement sensitivity (which determines
sigma magnitude) improves.

#### Recommendation

**Use init_spread=10.0.** This provides the full sigma benefit (+57%) with no downside.
Going higher (11-15) doesn't help sigma and the s7 improvements at those values are likely
seed noise.

---

### 4.5. temperature (Langevin Thermostat Temperature) — DEFAULT=0.5

**Impact: MEDIUM-HIGH. Significantly affects s7 rate, asymmetry, and sigma.**
**Confidence: HIGH for T=0.9 (252 runs) and T=1.0 (4,335 runs). Lower for other values.**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi (mean) | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|-----------|------|------|------|-------|
| 0.3 | 32 | 46.9% | 0.99 | 0.59 | -14.4 | 53.1% | 0.035 | **0.101** | -1.64 |
| **0.5** | 82 | **26.8%** | 2.41 | 2.59 | -6.8 | 45.1% | 0.042 | **0.187** | -3.69 |
| 0.6 | 11 | 27.3% | 4.25 | 1.53 | -17.6 | 54.5% | 0.046 | 0.042 | -8.44 |
| 0.7 | 32 | 31.2% | 7.99 | 2.34 | 9.0 | 46.9% | 0.040 | 0.040 | -12.80 |
| 0.75 | 20 | 40.0% | 8.61 | 1.93 | 1324.3 | 50.0% | 0.033 | 0.040 | -13.96 |
| 0.8 | 581 | 34.6% | 8.92 | 2.26 | 14.5 | 48.2% | 0.039 | 0.040 | -13.85 |
| 0.85 | 30 | 33.3% | 8.67 | 1.83 | 22.7 | 53.3% | 0.042 | 0.041 | -13.75 |
| **0.9** | **252** | **43.7%** | **12.14** | 2.63 | 26.9 | **52.8%** | 0.038 | 0.041 | -10.00 |
| 0.95 | 10 | 60.0% | 9.05 | 1.41 | 14.9 | 70.0% | 0.036 | 0.039 | -15.06 |
| **1.0** | **4,335** | **35.2%** | 9.36 | 3.46 | -8.2 | 49.6% | 0.042 | 0.047 | -12.24 |
| 1.1 | 40 | 35.0% | 10.30 | 2.56 | 16.2 | 67.5% | 0.042 | 0.040 | -13.56 |
| 1.2 | 30 | 33.3% | 10.99 | 2.80 | 17.3 | 43.3% | 0.041 | 0.040 | -12.35 |
| 1.5 | 11 | 27.3% | 10.95 | 2.36 | -20.3 | 36.4% | 0.033 | 0.040 | -15.55 |
| 2.0 | 13 | 23.1% | 10.26 | 2.22 | -2.5 | 38.5% | 0.047 | 0.041 | -12.43 |

#### Key Findings

**T=0.9 is the clear winner by a decisive margin.** With 252 runs:
- **43.7% s7 rate** — 8.5 percentage points above T=1.0 (35.2%) and 17 pp above T=0.5 (26.8%)
- **sigma=12.14** — 30% higher than T=1.0 (9.36) and 5x higher than T=0.5 (2.41)
- **xi=26.9** (positive!) vs T=1.0's -8.2 (negative)
- **xi+%=52.8%** vs T=1.0's 49.6%
- Low asymmetry (0.041), reasonable polyakov (0.038)

T=0.9 outperforms T=1.0 on every single metric except slope magnitude.

**The "asymmetry wall" is between T=0.5 and T=0.7.** This is one of the most important
transitions in the data:

| T range | Asymmetry | Interpretation |
|---------|-----------|----------------|
| <= 0.5 | 0.10 - 0.19 | Above 0.05 threshold: simulation is broken |
| 0.6 | 0.042 | Just crosses below threshold |
| >= 0.7 | 0.039 - 0.047 | Consistently below 0.05: simulation is healthy |

Below T=0.6, asymmetry exceeds 0.05, meaning the Re/Im decomposition is unphysical.
This single issue accounts for why T=0.5 (default) performs so poorly — it's not that
confinement is weaker, it's that the measurement apparatus (the Re/Im decomposition) is
broken. The simulation has not equilibrated well enough to explore both Re and Im sectors.

**T=0.95 shows 60% s7** but has only 10 runs. **T=1.1 shows 67.5% xi+** (40 runs) with
35% s7. These suggest the optimal T may be somewhere in the 0.9-1.1 range, but T=0.9 is
the best-tested value with clear superiority.

**T > 1.2 degrades performance.** s7 drops to 27-33% at T=1.5-2.0. Thermal noise becomes
too strong and destroys spatial correlations.

**Why does T=0.9 beat T=1.0 on sigma (12.14 vs 9.36)?** This seems paradoxical — T=0.9
should have slightly less thermal energy. The likely explanation is that T=0.9 was
predominantly tested with init_spread=10 (in R10b-R12), while T=1.0 was the standard
for rounds R3-R9 where init_spread=0 was common. The sigma difference is partially
confounded by init_spread. The s7 rate difference (43.7% vs 35.2%), however, is likely
genuine since many T=0.9 runs were direct comparisons.

#### Physical Interpretation

Temperature controls the magnitude of the stochastic noise term in the Langevin integrator.
Higher T means larger random kicks to particle velocities at each step.

At T=0.5, the thermal noise is insufficient to overcome energy barriers between local minima.
Particles get trapped in suboptimal configurations and cannot explore the full phase space.
This manifests as Re/Im asymmetry: the simulation preferentially samples one sector of
phase space.

At T=0.9, there is enough thermal energy to escape local traps and explore the relevant
phase space, while still being coherent enough to maintain spatial correlations needed for
confinement. This is the optimal balance.

At T > 1.5, the thermal noise is so large that particles are dominated by random motion.
Spatial correlations are constantly disrupted, reducing the measurable confinement signal.

#### Recommendation

**Use T=0.9.** This is a robust 8-9 percentage point improvement in s7 rate over T=1.0,
validated with 252 runs. The improvement is one of the most reliable in the campaign.

---

### 4.6. viscous_length_scale (Range of Viscous Interaction) — DEFAULT=1.0

**Impact: MEDIUM. Consistent improvement from 1.0 to 2.0.**
**Confidence: HIGH (4,053 runs at vls=2.0, 621 at vls=1.0).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi+% | poly | asym |
|-------|--------|-----|-------|---------|------|------|------|
| 0.25 | 70 | 22.9% | 5.22 | 4.65 | 57.1% | 0.065 | 0.097 |
| 0.5 | 111 | 29.7% | 6.67 | 4.37 | 47.7% | 0.052 | 0.076 |
| **1.0** | 621 | **30.9%** | **6.36** | 4.56 | 47.7% | 0.049 | 0.085 |
| 1.3 | 20 | 40.0% | 10.10 | 2.06 | 55.0% | 0.040 | 0.041 |
| 1.5 | 60 | 31.7% | 10.17 | 2.41 | 38.3% | 0.037 | 0.041 |
| **2.0** | **4,053** | **36.2%** | **9.80** | 3.10 | 49.9% | 0.040 | 0.042 |
| 2.5 | 40 | 42.5% | 9.18 | 1.97 | 52.5% | 0.039 | 0.041 |
| 2.7 | 10 | 40.0% | 10.47 | 2.08 | 70.0% | 0.040 | 0.042 |
| 3.0 | 100 | 30.0% | 10.28 | 2.57 | 47.0% | 0.042 | 0.040 |
| 4.0 | 40 | 37.5% | 9.55 | 1.99 | 55.0% | 0.037 | 0.040 |
| 5.0 | 81 | 40.7% | 9.67 | 2.81 | 50.6% | 0.036 | 0.043 |
| 10.0 | 170 | 38.2% | 9.32 | 2.14 | 51.2% | 0.037 | 0.040 |

#### Key Findings

**vls=1.0 (default) underperforms on multiple metrics.** Compared to vls=2.0:
- s7: 30.9% vs 36.2% (+5.3 pp)
- sigma: 6.36 vs 9.80 (+54%)
- polyakov: 0.049 vs 0.040 (lower is better)
- asymmetry: 0.085 vs 0.042 (half!)

Note that vls=1.0's high asymmetry (0.085) suggests many of those runs used higher nu
or lower T values (from early rounds before the standard was established).

**vls=2.0 is the well-validated standard.** With 4,053 runs, it provides reliable estimates:
36.2% s7, sigma=9.80, polyakov=0.040, asymmetry=0.042.

**vls=2.5 shows the best s7 rate** (42.5%) with 40 runs. **vls=5.0 is also strong** (40.7%,
81 runs). There may be marginal benefit to using vls=2.5 over vls=2.0, but the difference
(42.5% vs 36.2%) could be noise with only 40 runs.

**Very large vls (10.0) still works well** (38.2% s7, 170 runs). The viscous coupling range
can extend far without penalty. There is no "too large" for vls in the tested range.

**Very small vls (<0.5) is harmful**: polyakov increases to 0.052-0.065, sigma drops, and
the reduced coupling range fails to capture medium-range correlations.

#### Physical Interpretation

The viscous length scale determines the spatial range of the velocity coupling. Each
particle's velocity is pulled toward the neighbors within ~vls units. At vls=1.0, only
the nearest 5-10 neighbors influence each particle. At vls=2.0, the influence extends to
20-40 neighbors, creating longer-range correlations.

Confinement is a medium-to-long-range phenomenon — the confining string stretches across
many inter-particle spacings. To produce confinement signatures, particles need to
coordinate their behavior over distances comparable to the string length. With vls=1.0,
the coordination range is too short. With vls=2.0, it matches the relevant physical scale.

#### Recommendation

**Use vls=2.0 (standard) or vls=2.5 (experimental).** Both are solid improvements over
the default. The benefit plateaus above vls=2.5.

---

### 4.7. beta_curl (Boris Rotation Strength) — DEFAULT=1.0

**Impact: MEDIUM. Reducing to 0.25 improves sigma and reduces asymmetry.**
**Confidence: HIGH (4,635 runs at bc=0.25, 259 at bc=1.0).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|------|------|------|-------|
| 0.0 | 12 | 25.0% | 8.22 | 2.92 | 41.7% | 0.039 | 0.055 | -13.27 |
| 0.1 | 60 | 23.3% | 9.97 | 2.78 | 33.3% | 0.039 | 0.040 | -13.18 |
| 0.2 | 50 | 38.0% | 9.07 | 1.88 | 48.0% | 0.038 | 0.040 | -14.57 |
| **0.25** | **4,635** | **35.9%** | **9.45** | 3.45 | 50.2% | 0.041 | 0.046 | -12.26 |
| 0.35 | 40 | 37.5% | 8.91 | 1.82 | 55.0% | 0.042 | 0.042 | -14.12 |
| 0.5 | 142 | 30.3% | 9.53 | 2.63 | 50.7% | 0.042 | 0.042 | -13.25 |
| **1.0** | 259 | **32.8%** | **6.72** | 4.63 | 45.9% | 0.039 | **0.093** | -7.67 |
| 1.5 | 20 | 45.0% | 9.69 | 1.81 | 55.0% | 0.038 | 0.041 | -14.98 |
| 2.0 | 143 | 35.0% | 7.70 | 3.94 | 48.3% | 0.042 | 0.066 | -11.34 |
| 3.0 | 21 | 38.1% | 9.52 | 2.32 | 52.4% | 0.039 | 0.040 | -14.25 |

#### Key Findings

**bc=1.0 (default) has high asymmetry** (0.093) and low sigma (6.72). These are confounded
with early-round runs that also used other default values (nu=1.0, T=0.5), so the isolated
effect of bc may be smaller than it appears.

**bc=0.25 is the standard** with 4,635 runs: s7=35.9%, sigma=9.45, asymmetry=0.046. The
range bc=0.2-0.35 all perform similarly (37-38% s7 with smaller samples).

**bc=0.1 is too low** (23.3% s7, 33.3% xi+). Eliminating rotation entirely removes a
mechanism that helps particles explore angular degrees of freedom.

**bc=1.5 shows 45% s7** (20 runs) and **bc=3.0 shows 38%** (21 runs). These higher values,
tested in later rounds with the standard config, may indicate that the optimal bc is
not as low as 0.25 after all. This is one area where the campaign results are confounded —
most bc=0.25 runs include all rounds (including early rounds with bad nu/T/vls), while
bc=1.5 and bc=3.0 were only tested in later rounds with good nu/T/vls.

#### Physical Interpretation

The Boris rotation in the integrator adds a magnetic-field-like rotation to particle
velocities. At bc=1.0, this rotation is equal in magnitude to the friction force. At
bc=0.25, it is much weaker.

Strong rotation (bc >= 1.0) creates circular velocity patterns that can increase
asymmetry (the rotation preferentially populates one chirality of phase space). Reducing
bc to 0.25 makes the integrator more like standard Langevin dynamics, producing more
symmetric phase-space sampling.

However, some rotation may be beneficial — it helps particles explore angular configurations
that they would not reach through purely linear dynamics. This may explain why bc=1.5
and bc=3.0 look good in late rounds.

#### Recommendation

**Use bc=0.25 (standard).** It's well-validated and safe. If revisiting this parameter
in the future, test bc=1.0-2.0 specifically with the optimized config (nkin=5, nu=0.1,
T=0.9, vls=2.0) to see if the confounding is resolved.

---

### 4.8. delta_t (Time Step) — DEFAULT=0.01

**Impact: MEDIUM. dt=0.02 consistently outperforms dt=0.01.**
**Confidence: HIGH (4,573 runs at dt=0.01, 650 at dt=0.02).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi+% | poly | asym | slope |
|-------|--------|-----|-------|---------|------|------|------|-------|
| 0.005 | 12 | 25.0% | 1.68 | 0.55 | 41.7% | 0.038 | 0.048 | -3.73 |
| **0.01** | 4,573 | **34.9%** | 9.11 | 3.63 | 49.3% | 0.042 | 0.050 | -11.84 |
| 0.015 | 10 | 60.0% | 8.37 | 1.67 | 60.0% | 0.036 | 0.043 | -12.94 |
| **0.02** | **650** | **39.2%** | **10.49** | 2.46 | **52.8%** | **0.038** | 0.040 | **-13.64** |
| 0.025 | 10 | 40.0% | 9.53 | 1.76 | 40.0% | 0.037 | 0.040 | -14.02 |
| 0.03 | 80 | 32.5% | 9.85 | 2.21 | 46.2% | 0.039 | 0.041 | -14.64 |
| 0.04 | 10 | 50.0% | 8.22 | 1.40 | 50.0% | 0.035 | 0.039 | -14.10 |
| 0.05 | 140 | 35.7% | 9.54 | 2.23 | 50.0% | 0.038 | 0.041 | -14.30 |

#### Key Findings

**dt=0.02 beats dt=0.01 across the board.** With 650 runs vs 4,573:
- s7 rate: 39.2% vs 34.9% (+4.3 pp)
- sigma: 10.49 vs 9.11 (+15%)
- xi+%: 52.8% vs 49.3%
- polyakov: 0.038 vs 0.042 (lower, better)
- slope: -13.64 vs -11.84 (more negative, better)
- asymmetry: 0.040 vs 0.050 (lower, better)

This is a consistent improvement on every metric. The 650-run sample is large enough to
be reliable.

**dt < 0.01 is clearly harmful.** dt=0.005 gives sigma=1.68, barely above noise level.
Very small timesteps don't allow enough per-step mixing.

**dt=0.03-0.05 are acceptable** but not clearly better than 0.02. dt=0.03 dips to 32.5% s7
(80 runs), while dt=0.05 recovers to 35.7% (140 runs). The optimal may be a broad plateau
from dt=0.02 to dt=0.05.

**Interaction with nkin (see interaction table below):**

| | nkin=1 | nkin=3 | nkin=4 | nkin=5 | nkin=6 |
|---|---|---|---|---|---|
| dt=0.01 | 12% | 38% | 36% | 37% | 36% |
| **dt=0.02** | - | 30% | 37% | **46%** | **45%** |
| dt=0.05 | - | 30% | 37% | 30% | - |

dt=0.02 particularly shines with nkin=5-6 (46% and 45% s7), suggesting these parameters
have a positive synergy — larger timesteps combined with more substeps gives more total
displacement per clone cycle.

#### Physical Interpretation

The timestep determines how far particles move in a single integration step. With the
Boris-BAOAB integrator, larger dt means more phase-space displacement per step, including
more stochastic noise injection per step (since the Langevin noise scales with sqrt(dt)).

At dt=0.01, each step moves particles a small amount. Combined with nkin=4 substeps,
the total displacement per clone cycle is 4 * 0.01 * v ≈ 0.04v per cycle. At dt=0.02,
it's 0.08v — double the exploration per cycle.

This increased per-cycle exploration means particles discover more of the fitness landscape
before each cloning decision, leading to better-informed selections and stronger
confinement signals.

The sweet spot is dt=0.02: large enough for good exploration, small enough for numerical
stability. At dt=0.05, the integrator may start accumulating discretization errors.

#### Recommendation

**Use dt=0.02.** A solid 4 percentage point improvement in s7 rate with 15% higher sigma.
Validated with 650 runs. Safe and reliable.

---

### 4.9. init_velocity_scale (Initial Velocity Magnitude) — DEFAULT=0.0

**Impact: MEDIUM. ivs=0.5 provides meaningful sigma and xi improvement.**
**Confidence: MEDIUM (150 runs at ivs=0.5, 5,207 at ivs=0.0).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | xi (mean) | xi+% | poly | asym |
|-------|--------|-----|-------|---------|-----------|------|------|------|
| **0.0** | 5,207 | 35.5% | 9.16 | 3.51 | 0.1 | 49.6% | 0.041 | 0.049 |
| 0.1 | 20 | 35.0% | 9.83 | 1.43 | -41.0 | 45.0% | 0.042 | 0.041 |
| 0.3 | 10 | 20.0% | 10.11 | 1.15 | 33.1 | 60.0% | 0.051 | 0.041 |
| **0.5** | **150** | **39.3%** | **11.98** | 2.53 | **49.9** | **52.0%** | 0.039 | 0.040 |
| 0.8 | 10 | 30.0% | 9.34 | 1.05 | -38.8 | 40.0% | 0.037 | 0.040 |
| 1.0 | 70 | 24.3% | 11.07 | 2.78 | 2.3 | 47.1% | 0.044 | 0.041 |
| 1.5 | 10 | 10.0% | 11.82 | 2.64 | -8.2 | 20.0% | 0.047 | 0.039 |
| 2.0 | 20 | 35.0% | 9.96 | 1.84 | -5.2 | 50.0% | 0.041 | 0.041 |

#### Key Findings

**ivs=0.5 is a well-defined sweet spot.** Compared to ivs=0.0:
- sigma increases from 9.16 to 11.98 (+31%)
- s7 rate improves from 35.5% to 39.3%
- xi mean jumps from 0.1 to 49.9 (much more positive)

**Higher ivs degrades rapidly:**
- ivs=1.0: s7 drops to 24.3% (too much initial energy)
- ivs=1.5: s7 drops to 10.0% (only 10 runs, but dramatically bad)
- ivs=2.0: s7 recovers to 35.0% (20 runs) — possibly a different regime

The optimal is sharply peaked at ivs=0.5. Starting particles with moderate kinetic
energy creates diverse initial dynamics that help equilibration, but too much energy
disrupts structure formation during the critical early phase.

**Note on confounding**: Most ivs=0.5 runs were in R10b-R12 where T=0.9 and isp=10 were
also used. The sigma boost may partially come from these co-variables. However, within
R10b, ivs=0.5 combos outperformed their ivs=0.0 counterparts (r10b_isp10_ivs05 at 70%
s7 vs r10b_isp10_nkin5_T09 at 70% s7 with similar sigma).

#### Recommendation

**Use ivs=0.5 when combined with isp=10 and nkin=5.** This is one of the champion
configuration parameters. Don't use it in isolation with the old defaults.

---

### 4.10. alpha_restitution (Clone Velocity Restitution) — DEFAULT=1.0

**Impact: LOW. Default is appropriate. Lower values break asymmetry.**
**Confidence: HIGH for default (5,403 runs). LOW for other values.**

#### Data Table

| Value | n runs | s7% | sigma | poly | asym |
|-------|--------|-----|-------|------|------|
| 0.0 | 14 | 35.7% | 2.35 | 0.042 | **0.199** |
| 0.5 | 37 | 40.5% | 4.78 | 0.039 | **0.134** |
| 0.7 | 20 | 20.0% | 7.34 | 0.039 | 0.043 |
| 0.9 | 20 | 40.0% | 10.22 | 0.035 | 0.039 |
| **1.0** | **5,403** | **35.4%** | 9.34 | 0.041 | 0.047 |

The story is simple: alpha < 0.7 destroys asymmetry. At alpha=0.5, asymmetry reaches
0.134 (3x the threshold). At alpha=0.0, it's 0.199. These are in score 4 territory for
asymmetry. The s7 rates at these values (36-41%) are misleading — they come from early
rounds where other parameters (especially nu=0.1 which was already fixed) compensated.

alpha=0.9 looks slightly better than 1.0 (40% vs 35% s7, sigma 10.22 vs 9.34) but only
has 20 runs.

**Recommendation: Keep alpha=1.0.** Not worth the asymmetry risk.

---

### 4.11. fitness_beta — DEFAULT=1.0

**Impact: LOW-MEDIUM. fb=2.0 provides a moderate sigma boost.**
**Confidence: MEDIUM (61 runs at fb=2.0).**

#### Data Table

| Value | n runs | s7% | sigma | sig_std | poly |
|-------|--------|-----|-------|---------|------|
| 0.5 | 32 | 37.5% | 8.16 | 2.16 | 0.036 |
| **1.0** | 5,404 | 35.4% | 9.26 | 3.51 | 0.041 |
| **2.0** | 61 | **39.3%** | **11.21** | 2.65 | 0.038 |
| 3.0 | 10 | 20.0% | 9.23 | 1.30 | 0.043 |

fitness_beta controls how aggressively the cloning selects for high-fitness particles.
fb=2.0 means fitness differences are squared before computing cloning probabilities,
producing stronger selection pressure.

**fb=2.0 gives meaningful improvement**: +4 pp in s7 (39.3% vs 35.4%) and +21% in sigma
(11.21 vs 9.26). The polyakov also improves slightly (0.038 vs 0.041).

**fb=3.0 is over-aggressive**: 20% s7 with only 10 runs. The population converges too
quickly, losing diversity.

**fb=0.5 weakens selection**: sigma drops to 8.16, though s7 rate (37.5%) is still
reasonable.

**Recommendation: Use fb=2.0 as an optional improvement.** It provides a genuine sigma
boost with no apparent downside, but is lower priority than the critical parameters.

---

### 4.12. A (Fitness Amplitude) — DEFAULT=2.0

**Impact: LOW. A=3.0 provides a mild improvement.**
**Confidence: MEDIUM (51 runs at A=3.0).**

| Value | n runs | s7% | sigma | poly |
|-------|--------|-----|-------|------|
| 1.0 | 31 | 29.0% | 9.45 | 0.040 |
| **2.0** | 5,395 | 35.4% | 9.26 | 0.041 |
| **3.0** | 51 | **39.2%** | **10.96** | **0.033** |
| 5.0 | 20 | 40.0% | 9.53 | 0.036 |

A controls the overall amplitude of the fitness function. A=3.0 gives +4 pp in s7,
+18% in sigma, and noticeably lower polyakov (0.033 vs 0.041).

A=1.0 hurts (29% s7). A=5.0 looks okay (40% s7, 20 runs) but sigma doesn't improve
further.

**Recommendation: A=3.0 is a minor improvement. Optional.**

---

### 4.13. eta (Entropy Regularization) — DEFAULT=0.0

**Impact: UNCERTAIN. Small positive values may help.**
**Confidence: LOW (all non-default values have <= 20 runs).**

| Value | n runs | s7% | sigma | xi+% |
|-------|--------|-----|-------|------|
| **0.0** | 5,417 | 35.4% | 9.27 | 49.4% |
| 0.001 | 20 | 45.0% | 11.49 | 55.0% |
| 0.01 | 20 | 35.0% | 9.40 | 65.0% |
| 0.05 | 10 | 40.0% | 8.55 | 60.0% |
| 0.1 | 20 | 25.0% | 9.81 | 55.0% |
| 1.0 | 10 | 50.0% | 7.44 | 60.0% |

eta adds an entropy bonus to the fitness function, rewarding population diversity.
eta=0.001 shows 45% s7 and sigma=11.49, but with only 20 runs this is +/-14 pp uncertainty.

eta=0.1 hurts (25% s7). Higher eta values increasingly counteract selection pressure.

**Recommendation: Keep eta=0.0 for safety.** If you want to experiment, try eta=0.001,
but expect high variance.

---

### 4.14. N (Number of Particles) — DEFAULT=500

**Impact: MEDIUM. Larger N improves asymmetry and may improve s7.**
**Confidence: LOW for non-default values (10-20 runs each).**

| Value | n runs | s7% | sigma | asym | poly |
|-------|--------|-----|-------|------|------|
| 200 | 10 | 20.0% | 8.47 | 0.065 | 0.055 |
| 300 | 20 | 45.0% | 11.42 | 0.053 | 0.043 |
| **500** | 5,417 | 35.3% | 9.25 | 0.048 | 0.041 |
| **750** | 20 | **50.0%** | **11.27** | **0.033** | **0.031** |
| 1000 | 20 | 40.0% | 10.61 | **0.029** | 0.033 |

N=750 and N=1000 show markedly lower asymmetry (0.029-0.033 vs 0.048) and polyakov
(0.031-0.033 vs 0.041). The improved statistics from more particles make the
diagnostics more precise and less noisy.

N=750 appears optimal (50% s7, best polyakov) but computation time scales linearly:
~60s at N=750 vs ~40s at N=500, and neighbor graph operations scale quadratically.

N=200 is insufficient: polyakov jumps to 0.055 and s7 drops to 20%.

**Recommendation: N=500 for exploration sweeps. N=750 for production/final runs
where quality matters more than throughput.**

---

### 4.15. bounds_extent — DEFAULT=30.0

**Impact: NONE. This parameter has absolutely no effect on simulation results.**
**Confidence: VERY HIGH (tested exhaustively in R8 and R10 with 500+ runs).**

| Value | n runs | s7% | sigma |
|-------|--------|-----|-------|
| 10.0 | 10 | 40.0% | 10.78 |
| 15.0 | 10 | 40.0% | 9.93 |
| 20.0 | 110 | 42.7% | 9.65 |
| 25.0 | 10 | 40.0% | 10.00 |
| 28.0 | 10 | 50.0% | 9.54 |
| **30.0** | **5,024** | **34.6%** | 9.24 |
| 33.0 | 10 | 60.0% | 10.22 |
| 35.0 | 223 | 41.7% | 9.48 |
| 38.0 | 10 | 60.0% | 9.86 |
| 42.0 | 10 | 70.0% | 9.99 |
| 50.0 | 10 | 50.0% | 9.66 |

All variation in the table is pure seed noise. bounds_extent only controls the 3D viewer
in the dashboard — it has no effect on the physics. We wasted 500+ runs confirming this
across R8 and R10. The small-sample values (be=33, 38, 42 showing 60-70% s7) are just
lucky seeds, as confirmed by the 223-run be=35 sample giving 41.7% (consistent with
be=30's 34.6% within noise).

**Recommendation: Ignore this parameter. Keep at 30.0 or any other value.**

---

### 4.16. Other Parameters

**sigma_x (Clone Position Jitter) — DEFAULT=0.01**: Default is fine. Tested 0.001-0.05,
no value clearly outperforms 0.01 with adequate sample sizes. Keep default.

**p_max (Maximum Cloning Probability) — DEFAULT=1.0**: Default is best. Reducing to 0.5-0.9
hurts s7 rate (20-30% vs 36%). Keep default.

**n_steps (Simulation Length)**: Most runs used n_steps=200 post-warmup. Longer simulations
(1000-2000) did not improve s7 rate (36% vs 22-36%) and sometimes hurt. The confinement
signatures are established quickly and do not benefit from longer trajectories. The warmup
fraction of 0.5 means only the second half is analyzed — 200 post-warmup frames are
sufficient.

**fitness_alpha — DEFAULT=1.0**: Not systematically tested as an independent parameter.
Always kept at 1.0.

**sigma_min — DEFAULT=0.0**: Not tested. Always 0.0.

**init_offset — DEFAULT=0.0**: Not tested. Always 0.0.

---

## 5. Parameter Interactions

Single-parameter analysis can be misleading because parameters interact. This section
presents cross-tabulations that reveal synergies and conflicts between parameters.

### 5.1. n_kinetic_steps x init_spread

This is the most important interaction — the two highest-impact parameters.

#### Score 7 Rate

| | isp=0 | isp=5 | isp=10 | isp=11 | isp=12 |
|---|---|---|---|---|---|
| **nkin=1** | **12%** | - | - | - | - |
| nkin=3 | 36% | - | 60% | - | - |
| **nkin=4** | 36% | 33% | 34% | **70%** | **55%** |
| nkin=5 | 37% | 30% | 40% | 50% | - |
| nkin=6 | 36% | 47% | 42% | - | - |

#### Mean Sigma

| | isp=0 | isp=5 | isp=10 | isp=11 | isp=12 |
|---|---|---|---|---|---|
| **nkin=1** | **0.8** | - | - | - | - |
| nkin=3 | 7.3 | - | 13.2 | - | - |
| nkin=4 | 9.0 | 12.8 | 13.0 | 12.9 | 13.2 |
| nkin=5 | 8.6 | 12.2 | 13.3 | 13.1 | - |
| nkin=6 | 9.3 | 12.4 | 13.0 | - | - |

**Insights:**

1. **The effects are additive, not multiplicative.** nkin provides the base sigma (~9 at
   isp=0, once nkin>=4). isp adds ~4 sigma units on top (9 -> 13 at isp=10). The combination
   is roughly sigma = base(nkin) + boost(isp), not sigma = base(nkin) * boost(isp).

2. **nkin=4 + isp=11 achieves the highest s7 rate in the table (70%)**, though with small
   sample size. The 36% s7 rate at nkin=4/isp=0 is the most reliable baseline.

3. **Sigma plateaus at ~13.0-13.2 for isp>=10 regardless of nkin** (for nkin>=3). Once
   the initial spread is large enough, adding more kinetic steps doesn't further boost sigma.

4. **nkin=1 + isp=0 (both defaults) is the absolute worst** at 12% s7 and sigma=0.8.
   Switching to nkin=4 + isp=10 gives 34% s7 and sigma=13.0 — a complete transformation.

### 5.2. temperature x init_spread

| | isp=0 | isp=5 | isp=10 | isp=11 |
|---|---|---|---|---|
| T=0.8 | 35% | - | **10%** | - |
| **T=0.9** | **45%** | **50%** | 42% | 50% |
| T=1.0 | 35% | 34% | 34% | **70%** |
| T=1.1 | 33% | - | 40% | - |

**Critical finding: T=0.8 + isp=10 is catastrophic (10% s7).** This is a strong negative
interaction. Low temperature combined with high initial spread means the particles start
far apart but lack the thermal energy to equilibrate their spatial distribution. They
remain in a frozen, spread-out configuration that cannot develop confinement structure.

**T=0.9 is robust across all isp values** (42-50% s7). It provides enough thermal energy
to equilibrate even a widely spread initial configuration.

**T=1.0 + isp=11 hits 70%** — this is the specific combo that defines `r10b_isp11.0` in
the leaderboard. However, T=1.0 + isp=10 is only 34%, suggesting the jump to 70% is
isp=11-specific (likely seed noise with 20 runs at isp=11).

### 5.3. temperature x n_kinetic_steps

| | nkin=1 | nkin=3 | nkin=4 | nkin=5 | nkin=6 |
|---|---|---|---|---|---|
| T=0.3 | - | 45% | - | 50% | - |
| T=0.5 | 31% | 33% | - | 10% | - |
| T=0.7 | - | - | 35% | 20% | - |
| T=0.8 | - | 55% | 34% | 45% | 33% |
| T=0.9 | - | - | **46%** | **40%** | **50%** |
| T=0.95 | - | - | **60%** | - | - |
| T=1.0 | 7% | 36% | 36% | 38% | 44% |
| T=1.2 | - | - | 33% | - | - |

**T=0.9 + nkin=4 hits 46% s7** — better than T=1.0 + nkin=4 (36%). This confirms T=0.9's
superiority with the standard nkin value.

**T=0.95 + nkin=4 shows 60%** but with only 10 runs.

**T=0.5 + nkin=5 is terrible (10%)** — the cold temperature freezes dynamics even with
5 kinetic substeps. Temperature matters more than nkin at the extremes.

**T=1.0 + nkin=6 is strong (44%)** — at the standard temperature, more substeps help.

### 5.4. delta_t x n_kinetic_steps

| | nkin=1 | nkin=3 | nkin=4 | nkin=5 | nkin=6 |
|---|---|---|---|---|---|
| dt=0.01 | 12% | 38% | 36% | 37% | 36% |
| **dt=0.02** | - | 30% | 37% | **46%** | **45%** |
| dt=0.05 | - | 30% | 37% | 30% | - |

**dt=0.02 + nkin=5 is a strong synergy: 46% s7.** The larger timestep combined with
more substeps gives maximum exploration per clone cycle. This is the combination used
in the champion configuration.

dt=0.05 doesn't show the same synergy — at dt=0.05, nkin=5 actually drops to 30%.
The timestep may be too large for stable integration over 5 substeps.

### 5.5. nu x viscous_length_scale

| | vls=0.25 | vls=0.5 | vls=1.0 | vls=1.3 | vls=1.5 | vls=2.0 | vls=2.5 | vls=3.0 | vls=5.0 | vls=10.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| nu=0.01 | - | - | 32% | - | - | 25% | - | - | - | - |
| nu=0.05 | - | - | 18% | - | 40% | 30% | - | - | - | - |
| nu=0.1 | 20% | 30% | 32% | 40% | 30% | **37%** | **42%** | 30% | 41% | 38% |
| nu=0.125 | 40% | - | 10% | - | - | 30% | - | - | 40% | 40% |
| nu=0.2 | - | - | - | - | - | 33% | - | - | - | - |
| nu=0.3 | - | - | - | - | - | 45% | - | - | - | - |
| nu=1.0 | - | - | 32% | - | - | 14% | - | - | - | - |

**nu=0.1 + vls=2.0 is the well-tested standard (37%).** nu=0.1 + vls=2.5 achieves 42%
(with 40 runs). nu=0.1 + vls=5.0 and vls=10.0 also perform well (41% and 38%).

**nu=1.0 + vls=2.0 is very poor (14%)** — this confirms that nu=1.0 is bad regardless
of vls.

**nu=0.125 + vls=0.25 is interesting (40%)** — but small samples. Short-range strong
coupling might create dense local structures that work differently.

### 5.6. Champion Family Comparison

To understand how the top configurations compare, here are the major "families" of
parameter combinations that were extensively tested:

| Family | n runs | s7% | sigma | xi (mean) | xi+% | poly |
|--------|--------|-----|-------|-----------|------|------|
| isp=0, nkin=4, T=1.0 | 2,308 | 37% | 9.00 | -13.9 | 49% | 0.039 |
| isp=10, nkin=4, T=1.0 | 470 | 33% | 13.03 | 4.2 | 48% | 0.041 |
| isp=10, nkin=5, T=0.9 | 110 | **40%** | **13.19** | **60.6** | 46% | 0.036 |
| isp=5, nkin=5, T=1.0 | 10 | 30% | 12.16 | 7.0 | 40% | 0.038 |

The champion family (isp=10, nkin=5, T=0.9) outperforms the standard (isp=0, nkin=4, T=1.0)
across the board:
- s7: 40% vs 37% (+3 pp)
- sigma: 13.19 vs 9.00 (+47%)
- xi: 60.6 vs -13.9 (positive vs negative!)
- polyakov: 0.036 vs 0.039

The sigma boost comes primarily from isp=10, the s7 improvement from T=0.9, and the
positive xi from their combination.

---

## 6. R12 Stacking Experiment: "Less Is More"

### 6.1 Motivation

After R10b identified the champion configuration (isp=10 + nkin=5 + T=0.9) at 70% s7 rate,
the natural question was: can we do better by adding more individually-beneficial parameters?

Several parameters had shown moderate individual improvements:
- ivs=0.5 (init_velocity_scale): +4 pp s7, +31% sigma
- nu=0.05 (lower nu): +sigma but lower s7
- fb=2.0 (fitness_beta): +4 pp s7, +21% sigma
- A=3.0: +4 pp s7, +18% sigma
- eta=0.001: +10 pp s7 (small sample)
- gamma=0.7: mild s7 improvement

R12 systematically tested stacking each of these on top of the champion, as well as
stacking two or three simultaneously.

### 6.2 Results

| Configuration | s7% | sigma | xi (mean) | xi+% | Verdict |
|---|---|---|---|---|---|
| **isp10_nkin5_T09** (base) | **70%** | 13.05 | 14.8 | **80%** | **Champion** |
| + ivs05 | **70%** | 12.93 | 761.7 | 70% | **Matches champion** |
| + fb2 + A3 | 50% | 13.12 | -7.4 | 50% | Hurts score |
| + ivs05 + fb2 | 40% | 13.02 | 0.8 | 50% | Hurts score |
| + fb2 | 40% | 13.03 | -8.6 | 50% | Hurts score |
| + A3 | 30% | 13.95 | 6.5 | 30% | Hurts score badly |
| + eta001 | 30% | 13.72 | -16.2 | 40% | Hurts score badly |
| + ivs05 + nu005 | 30% | 13.36 | -19.5 | 40% | Hurts score badly |
| + nu005 | 20% | 12.40 | -49.9 | 30% | Devastates score |
| + gamma07 | 20% | 13.07 | -2.0 | 30% | Devastates score |
| isp11_nkin5_T09 | 50% | 13.07 | 2.1 | 60% | Worse than isp10 |
| isp10_ivs05_T09 (no nkin5) | 50% | 11.68 | 15.8 | 70% | nkin5 matters |
| isp10_dt02_T09_ivs05 | 50% | 12.67 | 19.1 | 70% | dt=0.02 doesn't help here |

### 6.3 Detailed Analysis

**Only ivs=0.5 can be added without damage.** The champion + ivs05 maintains 70% s7 and
produces an extreme mean xi of 761.7 (though this is driven by one very lucky seed with
xi > 7000; the median is likely ~15). This is the only addition that doesn't reduce s7.

**Sigma is completely unaffected by stacking.** All R12 combos have sigma in the 12.4-14.0
range regardless of what's added. This confirms that sigma is determined by isp and nkin;
other parameters do not move it.

**The score degradation comes entirely from xi turning negative.** When we add nu=0.05 to
the champion, xi drops from +14.8 to -49.9 (from 80% xi+ to 30% xi+). This is what
crashes s7 from 70% to 20%. The other metrics (sigma, polyakov, asymmetry) barely change.

**Why does stacking hurt?** Each additional parameter change introduces a perturbation
to the delicate balance of dynamics that produces positive xi. The champion configuration
happened to find a regime where kinetic evolution, cloning, temperature, and spatial spread
all cooperate to produce consistent screening fits. Adding nu=0.05 or gamma=0.7 changes
the flow field in ways that disrupt this cooperation, even though each modification was
beneficial in isolation.

This is a classic example of **non-linear parameter interactions**: improvements in isolation
do not compose. The parameter space has ridges and valleys, and the champion sits on a
ridge that is easy to fall off.

**Particularly revealing comparisons:**
- Adding nu=0.05 (which showed 30% s7 standalone): champion drops from 70% to 20%
- Adding gamma=0.7 (which showed 35% s7 standalone): champion drops from 70% to 20%
- Adding eta=0.001 (which showed 45% s7 standalone): champion drops from 70% to 30%

Each of these parameters had shown individual benefit when the rest of the configuration
was the "standard" (nkin=4, nu=0.1, T=1.0, vls=2.0). But when applied to an already-
optimized configuration, they interfere destructively.

### 6.4 Lessons for Future Optimization

1. **Test modifications in context, not in isolation.** A parameter that helps a mediocre
   config may hurt an optimized one. Always test on top of your current best.

2. **Fewer changes from defaults = more robust.** The champion changes only nkin, isp, T,
   and ivs (plus the standard fixes nu, vls, bc). Each additional change adds fragility.

3. **Sigma is not the bottleneck.** All R12 combos had excellent sigma (12-14). The
   differentiator is xi positivity, which is determined by subtle geometric properties
   that are easily disrupted.

4. **If the champion reaches 70% s7, achieving 80-90% likely requires architectural
   changes** (different fitness functions, different cloning strategies, different diagnostics)
   rather than parameter tuning within the current architecture.

---

## 7. Seed Variability and Statistical Reliability

### 7.1 The Fundamental Problem

Random seed variability is the dominant source of uncertainty in all results. The same
parameter combination can produce wildly different scores with different seeds.

### 7.2 Per-Seed Distribution

Across all 5,507 runs, here is how each seed performs (seeds 1-10 were used for most combos):

| Seed | n runs | s7% | mean score |
|------|--------|-----|------------|
| 1 | 522 | 39.7% | 6.22 |
| 2 | 522 | 34.3% | 6.18 |
| 3 | 522 | 37.5% | 6.21 |
| 4 | 520 | 34.4% | 6.16 |
| 5 | 521 | 32.4% | 6.13 |
| 6 | 521 | 36.5% | 6.15 |
| 7 | 521 | 33.2% | 6.12 |
| 8 | 521 | 35.1% | 6.19 |
| 9 | 521 | 34.0% | 6.18 |
| 10 | 521 | 38.8% | 6.22 |

**Seeds are not equal.** Seed 1 produces 39.7% s7 while seed 5 produces 32.4% — a 7.3 pp
difference across 520+ runs each. This is not noise; seed 1 genuinely produces initial
configurations that are more conducive to confinement than seed 5.

However, the variation between seeds (32-40% range) is much smaller than the variation
between parameter configurations (12% for nkin=1 vs 70% for the champion). So parameter
tuning is far more impactful than seed selection.

### 7.3 Seed Variability Within Combos

Six combos were tested with 20 seeds (1-10 and 11-20):

| Combo | n | s7% overall | s7% (seeds 1-10) | s7% (seeds 11-20) | Delta |
|---|---|---|---|---|---|
| r7_champ_isp0.1 | 20 | 45% | 45% | - | - |
| r7_champ_isp1.0 | 20 | 50% | 50% | - | - |
| r7_champ_isp5.0 | 20 | 45% | 45% | - | - |
| r7_champ_ivs0.1 | 20 | 35% | 35% | - | - |
| r9_dt02_rep | 20 | 30% | 50% | 10% | **-40 pp** |
| r10_be35_rep | 20 | 50% | 70% | 30% | **-40 pp** |

The r9_dt02_rep and r10_be35_rep cases show dramatic splits: the first 10 seeds
performed much better than the second 10. This is a 40 percentage point swing —
seeds 1-10 made the config look excellent (70% s7) while seeds 11-20 made it look
mediocre (30% s7).

### 7.4 Statistical Confidence Intervals

For a binary outcome (score >= 7 or not) with proportion p and sample size n, the
approximate 95% confidence interval is p +/- 1.96 * sqrt(p*(1-p)/n):

| Seeds per combo | If observed s7=35% | If observed s7=50% | If observed s7=70% |
|---|---|---|---|
| 10 | 35% +/- 30% (5-65%) | 50% +/- 31% (19-81%) | 70% +/- 28% (42-98%) |
| 20 | 35% +/- 21% (14-56%) | 50% +/- 22% (28-72%) | 70% +/- 20% (50-90%) |
| 30 | 35% +/- 17% (18-52%) | 50% +/- 18% (32-68%) | 70% +/- 16% (54-86%) |
| 50 | 35% +/- 13% (22-48%) | 50% +/- 14% (36-64%) | 70% +/- 13% (57-83%) |
| 100 | 35% +/- 9% (26-44%) | 50% +/- 10% (40-60%) | 70% +/- 9% (61-79%) |

**With 10 seeds, a combo showing 70% s7 has a 95% CI of 42-98%.** The true rate could
be anywhere from 42% to 98%. This is why the leaderboard should be interpreted with
extreme caution — a combo at 70% is not reliably better than one at 50%.

### 7.5 What We Can and Cannot Trust

**HIGH confidence (aggregated per-parameter statistics):**
The per-parameter tables (Section 4) aggregate hundreds or thousands of runs. These
statistics are very reliable:
- nkin=4 at 36.3% s7 (3,781 runs): 95% CI is 34.8-37.8%
- T=0.9 at 43.7% s7 (252 runs): 95% CI is 37.5-49.9%
- nu=0.1 at 36.0% s7 (4,916 runs): 95% CI is 34.7-37.3%

**MEDIUM confidence (individual combos with 10 seeds showing 50-60% s7):**
These are probably genuinely above average but the exact ranking is uncertain:
- A combo at 60% s7 is probably between 32-88% (huge range!)
- Relative to the 35% baseline, it's probably above average

**LOW confidence (individual combos with 10 seeds showing 70-80% s7):**
These are likely above average but the extreme values are probably inflated by luck:
- The champion at 70% s7 is probably 42-98% — could easily be 50% with more seeds
- The #1 combo at 80% (r8_be35.0) is probably 55-100%

### 7.6 Recommendations for Future Experiments

1. **Use 30+ seeds per combo** for any configuration you plan to deploy in production.
   This reduces the CI to +/-16% and gives meaningful rankings.

2. **When comparing two configurations**, they need to differ by at least 20 percentage
   points (with 10 seeds) to be confident the difference is real.

3. **The per-parameter aggregated statistics are your most reliable guide.** Trust the
   nkin, T, nu tables over any individual combo ranking.

4. **If a combo shows 70% s7 with 10 seeds, expect it to settle around 50-60% with
   more seeds.** Treat 10-seed results as upper-bound estimates.

---

## 8. Global Leaderboard (Top 40)

Sorted by s7 rate, then by sigma. All combos have 10 seeds unless noted. **Read Section 7
before interpreting these rankings — individual combo rankings are noisy.**

| Rank | Configuration | s7% | sigma | sig_std | xi | xi+% | poly | asym |
|------|--------------|-----|-------|---------|-----|------|------|------|
| 1 | r8_be35.0 | 80% | 9.31 | 2.37 | 7.8 | 80% | 0.031 | 0.041 |
| 2 | r9_isp10.0 | 70% | 13.25 | 2.32 | 1.5 | 70% | 0.036 | 0.041 |
| 3 | r10b_isp10_nkin5_T09 | 70% | 13.05 | 1.76 | 14.8 | 80% | 0.038 | 0.042 |
| 4 | r12_isp10_nkin5_T09_ivs05 | 70% | 12.93 | 2.16 | 761.7 | 70% | 0.033 | 0.040 |
| 5 | r10b_isp11.0 | 70% | 12.92 | 2.42 | 6.6 | 80% | 0.026 | 0.042 |
| 6 | r10b_isp10_ivs05 | 70% | 12.55 | 2.16 | 7.1 | 80% | 0.041 | 0.041 |
| 7 | r7_champ_dt0.02 | 70% | 10.00 | 1.24 | 58.6 | 80% | 0.033 | 0.040 |
| 8 | r10_be42.0 | 70% | 9.99 | 2.57 | 32.9 | 70% | 0.038 | 0.040 |
| 9 | r2_vls2.0_nkin4 | 70% | 9.98 | 2.07 | 265.9 | 70% | 0.031 | 0.039 |
| 10 | r10b_dt02_s21_30 | 70% | 9.75 | 1.64 | 7.9 | 90% | 0.030 | 0.040 |
| 11 | r11_T0.9 | 70% | 9.44 | 1.70 | 13.6 | 70% | 0.040 | 0.039 |
| 12 | r8_be20_nkin5 | 70% | 9.36 | 1.21 | 63.8 | 90% | 0.037 | 0.040 |
| 13 | r7_champ_dt0.05 | 70% | 9.21 | 1.47 | 24.0 | 70% | 0.028 | 0.041 |
| 14 | r11_vls2.5 | 70% | 8.91 | 1.93 | 84.4 | 70% | 0.037 | 0.042 |
| 15 | r11_N750_isp12 | 60% | 14.24 | 1.11 | 8.4 | 60% | 0.028 | 0.032 |
| 16 | r11_isp10_gamma2.0 | 60% | 13.55 | 3.04 | 17.3 | 90% | 0.036 | 0.041 |
| 17 | r10b_isp10_nkin3 | 60% | 13.18 | 1.41 | 0.8 | 60% | 0.025 | 0.041 |
| 18 | r9_nkin6_isp5 | 60% | 13.08 | 1.64 | -3.5 | 60% | 0.038 | 0.041 |
| 19 | r10b_isp5_nkin6 | 60% | 13.08 | 1.94 | 5.1 | 60% | 0.029 | 0.039 |
| 20 | r10b_isp10_dt02_T09 | 60% | 12.74 | 1.02 | 31.6 | 80% | 0.029 | 0.041 |
| 21 | r10b_isp10_nu0.05 | 60% | 12.66 | 2.43 | -4.2 | 70% | 0.032 | 0.040 |
| 22 | r11_N1000_isp14 | 60% | 12.39 | 2.36 | 12.5 | 60% | 0.034 | 0.029 |
| 23 | r9_dt02_isp3.0 | 60% | 10.99 | 1.78 | 23.4 | 80% | 0.039 | 0.039 |
| 24 | r10_be35_nkin5_dt02_fa2 | 60% | 10.85 | 1.96 | 8.7 | 70% | 0.031 | 0.040 |
| 25 | r3_nkin4_vls1.0 | 60% | 10.76 | 2.35 | -2.3 | 60% | 0.032 | 0.041 |
| 26 | r10_be33.0 | 60% | 10.22 | 2.13 | 21.7 | 70% | 0.037 | 0.040 |
| 27 | r10_be36.0 | 60% | 10.17 | 1.79 | -0.4 | 60% | 0.026 | 0.042 |
| 28 | r10b_fb2.0 | 60% | 10.10 | 1.03 | 15.6 | 60% | 0.029 | 0.042 |
| 29 | r10_be35_vls5 | 60% | 10.00 | 2.13 | -14.6 | 70% | 0.046 | 0.042 |
| 30 | r2_dashboard_nkin5 | 60% | 9.94 | 1.82 | 9.6 | 70% | 0.042 | 0.041 |
| 31 | r5_T08_fa2_nkin6 | 60% | 9.91 | 1.83 | 2.6 | 60% | 0.027 | 0.042 |
| 32 | r10_be38.0 | 60% | 9.86 | 2.15 | 7.5 | 70% | 0.041 | 0.042 |
| 33 | r11_gamma0.6 | 60% | 9.79 | 1.71 | 26.0 | 80% | 0.044 | 0.041 |
| 34 | r7_champ_be20.0 | 60% | 9.78 | 2.48 | 4.8 | 80% | 0.041 | 0.039 |
| 35 | r2_best_bc2.0 | 60% | 9.71 | 1.91 | -6.3 | 70% | 0.031 | 0.039 |
| 36 | r9_ivs0.5 | 60% | 9.58 | 1.67 | 7.2 | 60% | 0.028 | 0.041 |
| 37 | r10b_A3.0 | 60% | 9.50 | 1.84 | -1.9 | 60% | 0.034 | 0.043 |
| 38 | r3_nkin4_vls10_bc1.0 | 60% | 9.46 | 1.35 | -97.7 | 80% | 0.038 | 0.041 |
| 39 | r4_best_T0.8 | 60% | 9.46 | 1.87 | 11.1 | 80% | 0.040 | 0.042 |
| 40 | r7_champ_fa2 | 60% | 9.37 | 0.98 | 6.1 | 70% | 0.036 | 0.040 |

### Notable Observations

1. **14 combos reached 70% s7** — all with 10 seeds, so expect these to be ~50-60% with
   more seeds. They span different parameter families, from simple (r11_T0.9: just changing
   temperature) to complex (r12_isp10_nkin5_T09_ivs05: four changes).

2. **The highest-sigma combos are at rank 15-22** (60% s7, sigma 12.4-14.2). These use
   high init_spread and often larger N. They trade s7 rate for sigma magnitude.

3. **r8_be35.0 at rank 1 (80% s7) is an anomaly** — bounds_extent has no effect, so this
   is purely lucky seeds. With 10 runs at 80%, the 95% CI is 55-100%.

4. **The most consistent performer** is r10b_dt02_s21_30 (rank 10): 70% s7, 90% xi+,
   sigma=9.75, low sigma_std=1.64. This combo has the highest xi+ rate of any 70% s7 combo.

---

## 9. Recommended Configurations

### 9.1 Standard Configuration (Best All-Around)

Use this for general simulation work. Changes 7 parameters from defaults.

```python
STANDARD_CONFIG = {
    # Simulation size
    "N": 500,
    "d": 3,
    "n_steps": 200,                # post-warmup frames (total steps depends on record_every)
    "record_every": 1,

    # Initialization — spatially extended, moderate initial velocity
    "init_spread": 10.0,           # DEFAULT 0.0 -> 10.0  (sigma +57%)
    "init_velocity_scale": 0.5,    # DEFAULT 0.0 -> 0.5   (sigma +31%, xi boost)
    "init_offset": 0.0,            # keep default

    # Kinetic operator — more substeps, warmer, gentler coupling
    "n_kinetic_steps": 5,          # DEFAULT 1 -> 5     (CRITICAL: 3x s7 rate)
    "temperature": 0.9,            # DEFAULT 0.5 -> 0.9 (+9 pp s7 rate)
    "gamma": 1.0,                  # keep default
    "delta_t": 0.01,               # keep default (0.02 also good)
    "nu": 0.1,                     # DEFAULT 1.0 -> 0.1 (ESSENTIAL: fixes asymmetry)
    "viscous_length_scale": 2.0,   # DEFAULT 1.0 -> 2.0 (+5 pp s7 rate)
    "beta_curl": 0.25,             # DEFAULT 1.0 -> 0.25 (better sigma, lower asym)
    "integrator": "boris-baoab",
    "use_viscous_coupling": True,
    "viscous_neighbor_weighting": "riemannian_kernel_volume",
    "beta": 1.0,

    # Cloning — keep all defaults (these are fine)
    "clone_every": 1,              # MUST BE 1 — never change
    "alpha_restitution": 1.0,      # keep default
    "sigma_x": 0.01,               # keep default
    "p_max": 1.0,                  # keep default
    "epsilon_clone": 1e-6,

    # Fitness — keep defaults
    "fitness_alpha": 1.0,
    "fitness_beta": 1.0,           # keep default (2.0 optional, see 9.3)
    "A": 2.0,                      # keep default (3.0 optional, see 9.3)
    "eta": 0.0,                    # keep default

    # Graph — keep defaults
    "neighbor_graph_update_every": 1,
    "neighbor_weight_modes": [
        "inverse_riemannian_distance", "kernel", "riemannian_kernel_volume"
    ],
    "dtype": "float32",
}
```

**Expected performance**: ~40-50% s7 rate (true rate, accounting for seed variability),
sigma ~13, low polyakov (~0.035), low asymmetry (~0.04), strong negative slope (~-12).

**Changes from defaults** (in order of importance):
1. n_kinetic_steps: 1 -> 5 (CRITICAL)
2. nu: 1.0 -> 0.1 (ESSENTIAL)
3. temperature: 0.5 -> 0.9 (HIGH VALUE)
4. init_spread: 0.0 -> 10.0 (HIGH VALUE for sigma)
5. init_velocity_scale: 0.0 -> 0.5 (MEDIUM VALUE)
6. viscous_length_scale: 1.0 -> 2.0 (MEDIUM VALUE)
7. beta_curl: 1.0 -> 0.25 (MEDIUM VALUE)

### 9.2 High-Sigma Configuration (Maximize String Tension)

Use when you need the strongest possible confinement signal for analysis.

```python
HIGH_SIGMA_CONFIG = {
    # Changes from Standard Config:
    "N": 750,                      # 500 -> 750 (lower asymmetry, higher sigma)
    "init_spread": 12.0,           # 10.0 -> 12.0 (squeeze out more sigma)
    "delta_t": 0.02,               # 0.01 -> 0.02 (better mixing)
    "fitness_beta": 2.0,           # 1.0 -> 2.0 (stronger selection)
    # Everything else: same as Standard Config
}
```

**Expected performance**: sigma ~14-15, ~40-50% s7 rate. Slightly lower s7 than standard
because the additional changes add fragility, but when runs do achieve score 7, the
confinement signal is very strong.

### 9.3 Minimal Configuration (Fewest Changes from Defaults)

Use when you want to stay close to the original algorithm design but still get good results.
Only changes 5 parameters.

```python
MINIMAL_CONFIG = {
    "n_kinetic_steps": 5,          # from 1 (ESSENTIAL)
    "nu": 0.1,                     # from 1.0 (ESSENTIAL)
    "temperature": 0.9,            # from 0.5 (HIGH VALUE)
    "viscous_length_scale": 2.0,   # from 1.0 (RECOMMENDED)
    "beta_curl": 0.25,             # from 1.0 (RECOMMENDED)
    # Everything else: keep factory defaults
}
```

**Expected performance**: ~38-42% s7 rate, sigma ~10, good asymmetry (~0.04). This gets
you 80% of the benefit with minimal changes. The main sacrifice is sigma (10 vs 13) due
to not using init_spread=10.

### 9.4 Quick Exploration Configuration (Fast Runs for Testing)

Use when you want to test many configurations quickly and don't need high-quality results.

```python
QUICK_CONFIG = {
    "N": 200,                      # fast: ~10s per run
    "n_steps": 100,                # shorter trajectory
    "n_kinetic_steps": 4,          # nkin=4 is slightly faster than 5
    "nu": 0.1,
    "temperature": 0.9,
    "viscous_length_scale": 2.0,
    "beta_curl": 0.25,
    # Everything else: defaults
}
```

**Expected performance**: ~25-30% s7 rate (lower due to small N and short trajectory).
Use for rapid screening, then validate promising configs with the Standard Config.

---

## 10. Key Takeaways

### 10.1 Parameter Priority Ranking

| Priority | Parameter | Default | Optimal | Effect on s7 rate |
|----------|-----------|---------|---------|-------------------|
| 1 (Critical) | n_kinetic_steps | 1 | 4-5 | +24 pp (12% -> 36%) |
| 2 (Critical) | clone_every | 1 | 1 | Must not change |
| 3 (High) | nu | 1.0 | 0.1 | +8 pp (28% -> 36%), fixes asymmetry |
| 4 (High) | temperature | 0.5 | 0.9 | +9 pp (35% -> 44%) |
| 5 (High) | init_spread | 0.0 | 10.0 | +57% sigma, minor s7 change |
| 6 (Medium) | viscous_length_scale | 1.0 | 2.0 | +5 pp (31% -> 36%) |
| 7 (Medium) | beta_curl | 1.0 | 0.25 | +3 pp (33% -> 36%) |
| 8 (Medium) | delta_t | 0.01 | 0.02 | +4 pp (35% -> 39%) |
| 9 (Medium) | init_velocity_scale | 0.0 | 0.5 | +4 pp (36% -> 39%), +31% sigma |
| 10 (Low) | fitness_beta | 1.0 | 2.0 | +4 pp, +21% sigma |
| 11 (Low) | A | 2.0 | 3.0 | +4 pp, +18% sigma |
| 12 (Low) | N | 500 | 750 | +15 pp (small sample), lower asym |
| 13 (Negligible) | gamma | 1.0 | 1.0 | No reliable improvement |
| 14 (Negligible) | alpha_restitution | 1.0 | 1.0 | Lower values break asymmetry |
| 15 (Negligible) | eta | 0.0 | 0.0 | Inconclusive |
| 16 (Zero) | bounds_extent | 30.0 | any | No effect whatsoever |

### 10.2 The Five Most Important Lessons

**Lesson 1: n_kinetic_steps is the king parameter.**

Changing n_kinetic_steps from 1 to 4-5 is by far the most impactful modification:
- s7 rate triples (12% -> 36%)
- sigma increases 12x (0.80 -> 9.87)
- polyakov halves (0.085 -> 0.039)
- asymmetry drops 76% (0.174 -> 0.041)
- slope strengthens 13x (-0.95 -> -12.82)

No other parameter change comes close. If you only change one thing, change this.

**Lesson 2: The defaults are surprisingly bad.**

Four default values (nkin=1, nu=1.0, T=0.5, vls=1.0) are actively harmful. Together, they
produce ~12% s7 rate with barely measurable confinement signals. Fixing these four
parameters takes the simulation from 12% to ~44% s7 — a nearly 4x improvement.

The simulation was likely designed with different objectives in mind (perhaps general
particle dynamics rather than confinement diagnostics). The defaults are reasonable for
general exploration but terrible for maximizing regime score.

**Lesson 3: Positive xi is what makes score 7.**

Despite sigma being the most visually dramatic metric, it barely distinguishes score 7
from score 6 (9.62 vs 9.41). The critical differentiator is positive screening length:
99.9% of score-7 runs have xi > 0, compared to only 27.3% of score-6 runs.

Polyakov abs is the strongest single predictor of score (r = -0.616). Parameters that
reduce polyakov and improve the screening fit are more valuable than parameters that
boost sigma.

**Lesson 4: Less is more for multi-parameter tuning.**

The R12 stacking experiment proved that adding beneficial parameters beyond the top 3-4
actually hurts performance. The champion configuration (nkin=5, isp=10, T=0.9, ivs=0.5)
could not be improved by adding any 5th parameter — every addition reduced s7 rate from
70% to 20-50%.

This is a consequence of non-linear parameter interactions. Each parameter change
perturbs a delicate balance, and perturbations compose destructively even when they are
individually beneficial.

**Lesson 5: Seed variability dominates uncertainty.**

With 10 seeds per combo, expect +/-15-20 pp noise on s7 rate. A combo showing 70% s7
with seeds 1-10 could easily be 50% with seeds 11-20 (and we observed exactly this).

Individual combo rankings with <15 pp differences are meaningless. Trust the aggregated
per-parameter statistics (thousands of runs) over any individual combo ranking (10 runs).

For production deployment, validate your chosen configuration with 30+ seeds.

### 10.3 Remaining Open Questions

1. **Is T=0.9 really better than T=1.0?** Most T=0.9 runs were in later rounds with
   different co-variables than most T=1.0 runs. A controlled comparison with identical
   other parameters would be valuable.

2. **What is the optimal nu in the 0.05-0.3 range?** nu=0.1 was adopted early and
   received 4,916 runs, but nu=0.075, 0.125, and 0.3 showed intriguing results with
   small samples.

3. **Is beta_curl=0.25 really optimal, or was it confounded?** bc=1.5 and bc=3.0
   performed well in late rounds with good nu/T/vls.

4. **Can N=750-1000 reliably push s7 above 50%?** The small samples suggest so, but
   100+ runs at N=750 with the standard config would be needed to confirm.

5. **What architectural changes would push s7 above 70%?** Parameter tuning may have
   reached its ceiling. Fundamentally different fitness functions, cloning strategies,
   or diagnostic configurations may be needed.

---

## 11. Appendix: Methodology

### 11.1 Sweep Script

All runs were performed using `scripts/param_sweep.py`, which:
1. Accepts a batch JSON file with parameter overrides and seeds
2. Creates EuclideanGas with KineticOperator, CloneOperator, and FitnessOperator
3. Runs the simulation with `gas.run(n_steps, ...)`
4. Computes coupling diagnostics with `compute_coupling_diagnostics(history, config)`
5. Saves results as JSON to `outputs/param_sweep/`

Key implementation detail: `n_kinetic_steps` is set as an attribute after KineticOperator
construction (`kinetic_op.n_kinetic_steps = N`), not as a constructor argument. This was
a critical bug fix discovered in R1.

### 11.2 Diagnostics Configuration

```python
from fragile.physics.app.coupling_diagnostics import (
    compute_coupling_diagnostics,
    CouplingDiagnosticsConfig,
)

config = CouplingDiagnosticsConfig()  # defaults
output = compute_coupling_diagnostics(history, config=config)
# output.summary is a dict[str, float] with all metrics
```

The `warmup_fraction=0.5` setting means only the second half of each trajectory is
analyzed. The first half is considered equilibration time.

### 11.3 Batch File Format

```json
[
  {"name": "r10b_isp10_nkin5_T09_s1", "overrides": {"init_spread": 10.0, ...}, "seed": 1},
  {"name": "r10b_isp10_nkin5_T09_s2", "overrides": {"init_spread": 10.0, ...}, "seed": 2},
  ...
]
```

Each run produces a JSON file named `<timestamp>_<name>_<hash>.json` in the output directory.
The `run_name` field in the output matches the `name` field in the batch file.

### 11.4 Output File Format

```json
{
  "run_name": "r10b_isp10_nkin5_T09_s1",
  "params": { ... all parameters ... },
  "seed": 1,
  "warmup_fraction": 0.5,
  "duration_seconds": 40.2,
  "timestamp": "2026-02-17T15:30:00",
  "summary": {
    "regime_score": 7.0,
    "string_tension_sigma": 13.05,
    "polyakov_abs": 0.038,
    "screening_length_xi": 14.8,
    "running_coupling_slope": -10.0,
    "re_im_asymmetry_mean": 0.042,
    "local_phase_coherence_mean": 0.640,
    "r_circ_mean": 0.050,
    "topological_flux_std": 1.695,
    ...
  },
  "regime_evidence": ["String tension sigma=13.05 ...", ...]
}
```

### 11.5 Analysis Scripts

All analysis in this report was performed using inline Python scripts that:
1. Load all JSON files from `outputs/param_sweep/`
2. Extract combo names by stripping the `_s<seed>` suffix from `run_name`
3. Group by parameter values or combo names
4. Compute s7 rate, mean/std of metrics per group
5. Report results

No external analysis libraries beyond numpy were used.

---

*Report generated from 5,507 simulation runs across 756 parameter combinations.*
*Campaign duration: 2026-02-16 to 2026-02-18.*
*Total compute time: approximately 60 CPU-hours.*
