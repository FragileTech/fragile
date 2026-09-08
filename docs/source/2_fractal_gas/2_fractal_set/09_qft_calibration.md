# QFT Calibration: Channel Knobs and Mass Plateaus

:::{div} feynman-prose
**TLDR.** Calibration begins with a precisely defined observable, its recorded
frame and mask, and the correlator that the analysis actually computes. A
stable fitted exponential supplies an operational channel decay scale.
Identification with a physical particle requires the corresponding theoretical
model and independent validation.

This chapter connects the Fractal Set operators to the calibration code and
separates simulation parameters from analysis parameters. Some proposed
channels vanish identically under their current masks; some names denote
scalar phase proxies. The exact implementation identities below determine
which signals can be fitted and what those fits measure.

Prerequisites: {doc}`01_fractal_set`, {doc}`03_lattice_qft`,
{doc}`04_standard_model`, and {doc}`05_yang_mills_noether`.
Recorded experiments and the calibration notebook are in
{doc}`06_empirical_validation`, {doc}`07_qft_calibration_report`, and
{doc}`08_qft_calibration_notebook`.
:::

(sec-qft-calibration-correlators)=
## From correlators to mass plateaus

Channel masses are extracted from Euclidean correlators. The Schwinger functions are the Euclidean
correlators of the theory ({prf:ref}`def-euclidean-correlator-fg`). For practical calibration we
measure two-point correlators and their connected variants ({prf:ref}`def-two-point-connected`),
then fit a single-exponential decay in Euclidean time to identify a mass plateau.

The link to theory is the correlation length relation {prf:ref}`def-correlation-length` and the
mass scale hierarchy {prf:ref}`thm-mass-scales`. A stable exponential decay corresponds to a stable
correlation length, which is the operational definition of a channel mass in the analysis pipeline.

Implementation note: the generic correlator utilities and effective-mass extraction live in
`src/fragile/physics/new_channels/correlator_channels.py` and
`src/fragile/physics/aic/correlator_channels.py`. The active electroweak dashboard route then
assembles electroweak-specific operators and fit inputs in
`src/fragile/physics/app/electroweak_correlators.py` and
`src/fragile/physics/app/electroweak_mass_tab.py`.

(sec-qft-calibration-couplings)=
## Couplings and interaction ranges

:::{div} feynman-prose
The following coupling assignments use the normalization conventions and hypotheses of the cited results. They organize parameter comparisons within those models. A scalar dashboard phase or a channel name alone does not identify a matrix-valued gauge transport or its physical coupling.
:::



$$
g_1^2 = \frac{\hbar_{\text{eff}}}{\epsilon_d^2}\,\mathcal{N}_1(T,d)
$$
({prf:ref}`thm-sm-g1-coupling`)

$$
g_2^2 = \frac{2\hbar_{\text{eff}}}{\epsilon_c^2}\,\frac{C_2(2)}{C_2(d)}
$$
({prf:ref}`thm-sm-g2-coupling`)

$$
g_d^2 = \frac{\nu^2}{\hbar_{\text{eff}}^2}\,\frac{d(d^2-1)}{12}\,\langle K_{\text{visc}}^2\rangle_{\text{QSD}}
$$
({prf:ref}`thm-sm-g3-coupling`)

$$
e_{\text{fitness}}^2 = \frac{m}{\epsilon_F}
$$
({prf:ref}`thm-u1-coupling-constant`)

:::{div} feynman-prose
With the other factors fixed, these expressions give the displayed inverse-range and amplitude scalings. In a new simulation, changing a parameter can also change the QSD statistics, including the kernel average. The response of a fitted channel mass must therefore be measured; it does not follow from a prefactor alone.
:::



The scale conventions and separation regime in {prf:ref}`thm-mass-scales` are:

$$
m_{\text{clone}} = 1/\epsilon_c,\quad
m_{\text{MF}} = 1/\rho,\quad
m_{\text{gap}} = \hbar_{\text{eff}}\lambda_{\text{gap}},\quad
m_{\text{friction}} = \gamma.
$$

:::{div} feynman-prose
These characteristic scales describe the regime of the cited model. Applying a result that assumes their hierarchy requires checking that hierarchy. The actual channel decay also depends on the observable and its overlap with the evolving modes.
:::



(sec-qft-calibration-channels)=
## Channel sensitivity map (theory to knobs)

Channel operators are built from Fractal Set ingredients:

- Companion kernels and algorithmic distance ({prf:ref}`def-fractal-set-companion-kernel`,
  {prf:ref}`def-fractal-set-alg-distance`).
- Two-channel fitness and cloning score ({prf:ref}`def-fractal-set-two-channel-fitness`,
  {prf:ref}`def-fractal-set-cloning-score`).
- Viscous coupling and color state ({prf:ref}`def-fractal-set-viscous-force`,
  {prf:ref}`thm-sm-su3-emergence`).
- Gauge loops for glueball channels ({prf:ref}`def-fractal-set-plaquette`,
  {prf:ref}`def-fractal-set-wilson-loop`).

The table below summarizes which knobs primarily move which channel families. The suggested directions are sweep hypotheses. Their signs and magnitudes require validation for the chosen observable, generating run, and fit window.

| Channel family | Fractal Set ingredient | Primary knobs | Expected qualitative effect (operational) |
| --- | --- | --- | --- |
| Meson / pseudoscalar (color bilinear) | Color state from viscous force ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, $\gamma$, $\beta$, $\Delta t$ | Shorter $\rho$ or larger $\nu$ increases color coupling, typically shortening correlators (heavier masses). |
| Baryon / nucleon (color determinant) | SU(3) invariant of three color vectors ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, neighbor selection | Trilinear color invariants are sensitive to color coherence; adjust $\nu$ and $\rho$ first. |
| Glueball / gauge channel | Gauge field strength and Wilson loops ({prf:ref}`def-fractal-set-viscous-force`, {prf:ref}`def-fractal-set-wilson-loop`) | $\nu$, $\rho$ | Stronger viscous coupling or shorter $\rho$ tends to increase glueball mass scales. |
| Cloning/diversity-dominated channels | Companion kernel + cloning score ({prf:ref}`def-fractal-set-companion-kernel`, {prf:ref}`def-fractal-set-cloning-score`) | $\epsilon_c$, $\epsilon_d$, $\lambda_{\text{alg}}$, $\epsilon_{\text{clone}}$, $p_{\max}$ | Decreasing $\epsilon_c$ or $\epsilon_d$ strengthens the corresponding coupling and can shift correlator decay. |
| Fitness/U(1) phase channels | Phase potential and fitness coupling ({prf:ref}`def-fractal-set-phase-potential`, {prf:ref}`thm-u1-coupling-constant`) | $\epsilon_F$, fitness weights $(\alpha,\beta)$ | Larger $\epsilon_F$ weakens the fitness coupling, softening phase-driven oscillations. |

(sec-qft-calibration-channel-derivations)=
## Channel operators and calibration parameters

The electroweak correlator and mass tabs wired by `src/fragile/physics/app/dashboard.py` report
**Extracted Masses** from Euclidean two-point correlators. The generic correlator machinery lives in
`src/fragile/physics/new_channels/correlator_channels.py` and
`src/fragile/physics/aic/correlator_channels.py`, while the electroweak-specific assembly happens
in `src/fragile/physics/app/electroweak_correlators.py` and
`src/fragile/physics/app/electroweak_mass_tab.py`. For any channel operator $O_\chi$,

$$
C_\chi(\tau) = \langle O_\chi(\tau)\,O_\chi(0)\rangle_{\text{conn}}
$$
({prf:ref}`def-euclidean-correlator-fg`, {prf:ref}`def-two-point-connected`).

For a channel with a nonzero leading exponential contribution in the selected regime, write the asymptotic form and effective-mass estimator as:

$$
C_\chi(\tau) \sim Z_\chi e^{-m_\chi \tau},
\qquad
m_\chi(\tau) = -\frac{1}{\Delta \tau}\log\frac{C_\chi(\tau+\Delta\tau)}{C_\chi(\tau)},
\qquad
\xi_\chi = \frac{1}{m_\chi}
$$

using the correlation-length definition {prf:ref}`def-correlation-length` and the mass-scale
hierarchy {prf:ref}`thm-mass-scales`. The AIC-weighted plateau in the Channels tab is an
implementation of this $m_\chi(\tau)$ extraction, so its output depends on the operator, sampling, and fit window. The operator formulas identify parameters to investigate; they do not establish universal monotonic tuning rules.

:::{div} feynman-prose
For a fixed correlator sequence indexed by frame lag, changing the assigned
time unit rescales every fitted decay rate by the inverse factor. Changing
$\Delta t$ in a new simulation changes its transition kernel, while changing
recording stride changes the sampled data. Neither operation is merely a
change of units. Compare dynamical sweeps at controlled time resolution and
check discretization and recording effects separately.

The scale hierarchy is a hypothesis of the corresponding continuum or
spectral model. Use it when applying those results, alongside checks of the
observed fit stability and uncertainty.
:::

Implementation note: the active electroweak route exposes analysis knobs such as `h_eff`,
`mass`, `ell0`, `ell0_method`, `max_lag`, `use_connected`, and the Bayesian fit settings in the
mass tab. These belong to the **measurement** map, not the underlying swarm dynamics. The first
three appear directly in the color-state and spinor constructions
({prf:ref}`thm-sm-su3-emergence`, {prf:ref}`def-lqft-chiral-projectors`), so set them consistently
with the run. Changing them can move extracted masses without changing the simulation itself.

The table below makes the knob-to-parameter correspondence explicit.

| Knob (symbol) | Algorithm parameter name | Location (code) | Role in calibration |
| --- | --- | --- | --- |
| $\nu$ | `nu` | `KineticOperator` (`src/fragile/physics/fractal_gas/kinetic_operator.py`) | Viscous coupling strength (color/gauge sector) |
| $\rho$ | `viscous_length_scale` | `KineticOperator` | Localization range of viscous kernel |
| $\gamma$ | `gamma` | `KineticOperator` | Friction mass scale ($m_{\text{friction}}$) |
| $\beta$ | `beta` | `KineticOperator` | Inverse temperature (noise scale) |
| $\Delta t$ | `delta_t` | `KineticOperator` | Integrator step; changes the dynamics when rerunning |
| $\epsilon_F$ | `epsilon_F` | `KineticOperator` | Fitness/U(1) coupling scale |
| $\epsilon_c$ | `companion_selection_clone.epsilon` | `RunHistory.params` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Clone-companion interaction range used by electroweak operators |
| $\epsilon_d$ | `companion_selection.epsilon` | `RunHistory.params` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Diversity-companion interaction range used by electroweak operators |
| $\lambda_{\text{alg}}$ | `lambda_alg` | Fixed inside `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | Pinned to `0.0` in the active electroweak pipeline |
| $\epsilon_{\text{clone}}$ | `epsilon_clone` | `ElectroweakCorrelatorSettings` in `src/fragile/physics/app/electroweak_correlators.py`, falling back to `CloneOperator` (`src/fragile/physics/fractal_gas/cloning.py`) via `RunHistory.params` | Cloning-score regularization entering SU(2) and chirality operators |
| $p_{\max}$ | `p_max` | `CloneOperator` (`src/fragile/physics/fractal_gas/cloning.py`) | Max cloning probability in the recorded dynamics |
| Fitness weights $(\alpha,\beta,\eta,A,\rho)$ | `alpha`, `beta`, `eta`, `A`, `rho` | `FitnessOperator` (`src/fragile/physics/fractal_gas/fitness.py`) | Fitness coupling shape |
| $\hbar_{\text{eff}}$ | `h_eff` | `ElectroweakCorrelatorSettings` (`src/fragile/physics/app/electroweak_correlators.py`) | Measurement: phase scale for chirality and Dirac-spinor operators |
| $m$ (phase mass) | `mass` | `ElectroweakCorrelatorSettings` | Measurement: color-state phase factor in the spinor path |
| $\ell_0$ | `ell0` | `ElectroweakCorrelatorSettings` | Measurement: color-state length scale in the spinor path |
| $\ell_0$ method | `ell0_method` | `ElectroweakCorrelatorSettings` | Measurement: automatic estimator for the spinor path when `ell0` is blank |
| Connected correlator | `use_connected` | `ElectroweakCorrelatorSettings` | Measurement: connected vs raw $C(t)$ |
| Max lag | `max_lag` | `ElectroweakCorrelatorSettings` | Measurement: correlator window length |
| Warmup fraction | `warmup_fraction` | `ElectroweakCorrelatorSettings` | Measurement: drop transient steps |
| Covariance / prior fit controls | `covariance_method`, `nexp`, `tmin`, `tmax`, `svdcut`, `use_log_dE`, `use_fastfit_seeding`, `effective_mass_method`, `include_multiscale` | `ElectroweakMassSettings` (`src/fragile/physics/app/electroweak_mass_tab.py`) | Bayesian mass-extraction and plateau-fitting controls |

The older electroweak UI in `src/fragile/physics/app/electroweak.py` retains additional knobs such
as `knn_k`, `knn_sample`, and `window_widths_spec`. Those belong to that legacy/alternate
interface, not to the active `dashboard.py` route documented in this chapter.

Below, each channel is tied to its operator, the Fractal Set ingredients that define it, and the
parameters that control its correlator decay.

:::{div} feynman-prose
A normalization can remove an apparent tuning parameter from a fixed frame.
In the displayed color encoding, multiplying every component of a nonzero
viscous force vector by the same positive number leaves its normalized color
state unchanged, with the phase held fixed. Varying viscous strength in a new
run can still change the trajectory. Distinguish that dynamical effect from
recomputing an observable on the same recorded frame.
:::

### Scalar channel (σ, $0^{++}$)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger color coupling → shorter correlator → heavier $m_\sigma$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter localization → heavier $m_\sigma$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster velocity relaxation → heavier $m_\sigma$ (keep hierarchy). |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → weaker phase winding → slightly lighter $m_\sigma$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\sigma$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\sigma$. |

**Operator (bilinear color scalar):**

$$
O_{\sigma}(t) = \langle \bar{\psi}_i \psi_j \rangle
\;\propto\; \sum_a \left(c_i^{(a)}\right)^* c_j^{(a)}.
$$

The color state $c_i$ is built from the viscous force and momentum-phase encoding
({prf:ref}`thm-sm-su3-emergence`):

$$
\tilde{c}_i^{(\alpha)} = F_\alpha^{(\text{visc})}(i)\,
\exp\!\left(i\,p_i^{(\alpha)}\ell_0/\hbar_{\text{eff}}\right),
\quad
c_i^{(\alpha)} = \frac{\tilde{c}_i^{(\alpha)}}{\|\tilde{c}_i\|}.
$$

Therefore the scalar correlator is controlled by the viscous force
({prf:ref}`def-fractal-set-viscous-force`) and the $SU(d)$ coupling
({prf:ref}`thm-sm-g3-coupling`), with the mean-field range $\rho$ and friction $\gamma$ setting the
dominant decay scales ({prf:ref}`thm-mass-scales`).

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen the viscous coupling and shorten the scalar
  correlation length (heavier scalar mass).
- Decrease $\nu$ or increase $\rho$ to soften the coupling and lengthen the plateau (lighter
  scalar mass).
- Increasing $\gamma$ raises $m_{\text{friction}}$ and typically shortens scalar plateaus; keep the
  hierarchy $m_{\text{friction}} \ll m_{\text{gap}}$ intact.

### Pseudoscalar channel (π, $0^{-+}$)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase dispersion → lighter $m_\pi$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → more phase winding → heavier $m_\pi$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → more phase winding → heavier $m_\pi$. |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → lifts overall meson scale → heavier $m_\pi$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter coupling → heavier $m_\pi$. |

**Operator (bilinear with $\gamma_5$ projection):**

$$
O_{\pi}(t) = \langle \bar{\psi}_i \gamma_5 \psi_j \rangle
\;\propto\; \sum_a \left(c_i^{(a)}\right)^* (\gamma_5)_{aa}\,c_j^{(a)}.
$$

Because $\gamma_5$ alternates signs across components, the pseudoscalar channel is **phase
sensitive**: it responds directly to the momentum-phase factor
$\exp(i\,p_i^{(\alpha)}\ell_0/\hbar_{\text{eff}})$ in the color state
({prf:ref}`thm-sm-su3-emergence`). This is the cleanest knob for splitting scalar vs.
pseudoscalar masses **without** changing the overall color coupling.

**Sweep hypotheses to check:**
- Increase $m$ or $\ell_0$, or decrease $\hbar_{\text{eff}}$, to increase phase winding and shorten
  the pseudoscalar correlation length (heavier pseudoscalar).
- Decrease $m$ or $\ell_0$, or increase $\hbar_{\text{eff}}$, to reduce phase dispersion (lighter
  pseudoscalar).
- Sweep $\nu$ and $\rho$ to measure whether the scalar and pseudoscalar scales move together
  through their dependence on the viscous-force coupling.

### Vector channel (ρ, $1^{--}$)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger alignment → heavier $m_\rho$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter alignment → heavier $m_\rho$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster decay of coherent modes → heavier $m_\rho$. |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase winding → slightly lighter $m_\rho$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\rho$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\rho$. |

**Operator (bilinear with $\gamma_\mu$ projection):**

$$
O_{\rho}(t) = \langle \bar{\psi}_i \gamma_\mu \psi_j \rangle
\;\propto\; \frac{1}{d}\sum_\mu \sum_{a,b} \left(c_i^{(a)}\right)^* (\gamma_\mu)_{ab}\,c_j^{(b)}.
$$

The vector projection emphasizes **directional coherence** in the color state, which is driven by
velocity alignment in the viscous force ({prf:ref}`def-fractal-set-viscous-force`) and damped by
friction ($m_{\text{friction}}=\gamma$; {prf:ref}`thm-mass-scales`).

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen alignment and shorten the vector correlator
  (heavier vector mass).
- Increase $\gamma$ to speed velocity relaxation, which typically shortens vector plateaus.
- Keep $\Delta t$ fixed when comparing vector masses across runs (see the time-scale normalization
  rule above).

### Nucleon channel (baryon, color determinant)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → tighter color coherence → heavier $m_N$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → stronger local binding → heavier $m_N$. |
| Clone temperature | $\epsilon_c$ | Recorded in `RunHistory.params["companion_selection_clone"]["epsilon"]` | Decrease $\epsilon_c$ in the generating run → stronger clone locality in the recorded companion graph → typically heavier $m_N$. |
| Diversity temperature | $\epsilon_d$ | Recorded in `RunHistory.params["companion_selection"]["epsilon"]` | Decrease $\epsilon_d$ in the generating run → tighter distance locality → typically heavier $m_N$. |
| Alg. distance weight | $\lambda_{\text{alg}}$ | Recorded run parameter when present | Larger $\lambda_{\text{alg}}$ strengthens velocity-weighted locality in the generating run and can raise $m_N$. |
| Pair selection | — | `CompanionCorrelatorSettings.pair_selection` | Measurement: choose distance pairs, clone pairs, or both when building local triplets; this changes the estimator, not the recorded dynamics. |
| Multiscale locality | — | `CompanionCorrelatorSettings.n_scales`, `kernel_type`, `edge_weight_mode` | Measurement: changes neighborhood weighting and plateau stability for baryon correlators without changing the run itself. |

**Operator (trilinear color invariant):**

$$
O_{N}(t) = \det\!\big[c_i, c_j, c_k\big]
$$

This channel is an $SU(3)$-invariant trilinear built from the same color state
({prf:ref}`thm-sm-su3-emergence`). It probes **three-body color coherence**, which depends both on
the viscous coupling (for color alignment) and on the companion/IG structure that determines which
triplets are local ({prf:ref}`def-fractal-set-companion-kernel`,
{prf:ref}`def-fractal-set-cloning-score`).

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to tighten color coherence and increase nucleon masses.
- Adjust $\epsilon_c$, $\epsilon_d$, and $\lambda_{\text{alg}}$ to modify local companion structure
  and triplet availability; this changes baryon plateaus without altering the color definition.
- Implementation constraint: the nucleon channel requires $d=3$ and at least two neighbors; if the
  Channels tab reports `n/a`, verify that the run dimension is three and that neighbor sampling is
  adequate.

### Glueball channel ($0^{++}$, gauge sector)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger force fluctuations → heavier $m_G$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → shorter-range force → heavier $m_G$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster damping → heavier $m_G$. |

**Operator (force-norm gauge observable):**

$$
O_{G}(t) = \sum_i \left\|F^{(\text{visc})}(i,t)\right\|^2,
$$

which is the local gauge-field strength proxy derived from the viscous force
({prf:ref}`def-fractal-set-viscous-force`) and tied to Wilson-loop observables
({prf:ref}`def-fractal-set-plaquette`, {prf:ref}`def-fractal-set-wilson-loop`).
The glueball correlator therefore tracks how quickly the force magnitude decorrelates under the
viscous coupling.

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen gauge-field fluctuations and shorten the glueball
  correlator (heavier glueball mass).
- Use $\gamma$ only to fine-tune decay speed while preserving the mass-scale hierarchy.

### Empirical calibration status (zero-reward baseline)

The baseline QFT calibration runs in `QFT_CALIBRATION_REPORT.txt` (zero reward, viscosity-only,
200 walkers, 300 steps, Channels-tab analysis) show a **tradeoff** between the target ratios
$R_{\rho\pi}=5.5$ and $R_{N\pi}=6.7$:

- **Closest $R_{\rho\pi}$**: $\;R_{\rho\pi}\approx 5.437$ (thr=0.9, pen=1.1, $\beta=0.5$), but
  $R_{N\pi}\approx 0.592$ (nucleon suppressed).
- **Closest $R_{N\pi}$**: $\;R_{N\pi}\approx 6.171$ (weak\_potential\_fit1\_aniso\_stable2), but
  $R_{\rho\pi}\approx 3.186$ (rho too light).
- **Nucleon\_abs2** can raise $R_{N\pi}$ (≈7.55) but collapses $\pi$ and explodes $R_{\rho\pi}$.
- **Threshold sensitivity**: high neighbor thresholds (≈0.9) are the only tested lever that moves
  $R_{\rho\pi}$ near target, but they suppress $R_{N\pi}$ in the baseline.
- **Numerical stability**: curl + anisotropic diffusion runs are currently unstable (NaN noise at
  step 1), so those results are not admissible for calibration.

**Empirical conclusion.** Within the current viscosity-only baseline and neighbor-threshold/penalty
parameter space, no configuration achieves both ratios within the ±2% tolerance. High companion
thresholds move $R_{\rho\pi}$ toward target but suppress $R_{N\pi}$; stable anisotropic settings
recover $R_{N\pi}$ but leave $R_{\rho\pi}$ low. These findings are measurement-based and do not
override the theoretical ratio-sieve constraints below; they instead flag where the current
baseline fails to realize the target point.

(sec-qft-calibration-electroweak)=
## Electroweak dashboard calibration

:::{div} feynman-prose
The dashboard combines three observable families: labels derived from recorded
cloning roles, projected Dirac-spinor bilinears, and legacy scalar-phase or
doublet proxies. They share correlator utilities but have different algebra
and masks. The two realization propositions below specify the recorded frames,
normalizations, and scalar factors actually used by the implementation.

Start with those definitions before interpreting a fitted mass. In particular,
a channel that is identically zero has no exponential amplitude to fit, and
a scalar phase applied to a bilinear does not acquire matrix-valued gauge
transport merely through its channel name.
:::

Implementation note: the active electroweak correlator path is
`src/fragile/physics/electroweak/electroweak_channels.py`, with chirality classification in
`src/fragile/physics/electroweak/chirality.py`, projector-based spinor operators in
`src/fragile/physics/electroweak/electroweak_spinors.py`, the channel-selection UI in
`src/fragile/physics/app/electroweak_correlators.py`, the mass-extraction tab in
`src/fragile/physics/app/electroweak_mass_tab.py`, and the top-level tab wiring in
`src/fragile/physics/app/dashboard.py`.

### Walker-role chirality observables

The baseline electroweak matter observables are defined from the recorded clone events. At each
frame, alive walkers are partitioned into

$$
\Delta_t,\qquad \mathrm{SR}_t,\qquad \mathrm{WR}_t,\qquad \mathrm{P}_t,
$$

exactly as in {prf:ref}`def-sm-walker-role-partition`, with left- and right-handed sectors

$$
L_t = \Delta_t \cup \mathrm{SR}_t,
\qquad
R_t = \mathrm{WR}_t \cup \mathrm{P}_t.
$$

The chirality label is

$$
\chi_i(t)=
\begin{cases}
+1,& i\in L_t,\\
-1,& i\in R_t,\\
0,& i\notin A_t.
\end{cases}
$$

Writing $N$ for the recorded walker count per frame, the dashboard-computed chirality channels are
then

$$
\chi_{\mathrm{mean}}(t)=\frac{1}{N}\sum_{i=1}^{N}\chi_i(t),
\qquad
f_L(t)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}_{\{i\in L_t\}},
$$

$$
f_{\Delta\to R}(t)=\frac{1}{|\Delta_t|}
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}},
$$

and the complex left-right transfer observable

$$
M_{LR}(t)=
\frac{1}{N_{\Delta\to R}(t)}
\sum_{\substack{i\in\Delta_t\\c_c(i,t)\in R_t}}
\exp\!\left(i\frac{F_{c_c(i,t)}(t)-F_i(t)}{\hbar_{\mathrm{eff}}}\right).
$$

Dead walkers contribute $0$ to $\chi_i$, so the averages above are taken over the full recorded
walker count exactly as in the implementation. The conventions are
$f_{\Delta\to R}(t)=0$ when $|\Delta_t|=0$ and $M_{LR}(t)=0$ when
$N_{\Delta\to R}(t)=0$.

Operationally, the electroweak correlator tab exposes these as `chi_mean`, `left_fraction`,
`lr_fraction`, and `lr_coupling_mag`. 

:::{div} feynman-prose
Under the same-frame role partition, every alive target of a cloning walker
belongs to the left set. Thus the intersection defining the right-target
transfer is empty: `lr_fraction` and `lr_coupling_mag` are exactly zero under
the proposition's conventions. Their zero correlators contain no mass signal.
The nonzero role observables remain diagnostics of the recorded population;
identifying them with a physical chiral interaction requires additional model
structure.
:::



:::{prf:proposition} Current Chirality-Channel Realization
:label: prop-qft-ew-chirality-realization

**Rigor Class:** F (Implementation-Exact)

Let

$$
t \in \{t_{\mathrm{start}},\dots,t_{\mathrm{end}}-1\},
\qquad
t_{\mathrm{start}}=\max(1,\lfloor n_{\mathrm{recorded}}\,f_{\mathrm{warm}}\rfloor),
\qquad
t_{\mathrm{end}}=\max(t_{\mathrm{start}}+1,\lfloor n_{\mathrm{recorded}}\,f_{\mathrm{end}}\rfloor).
$$

For each such frame, let $\chi_i(t)$, $L_t$, $R_t$, and $\Delta_t$ be the walker-role chirality
objects of {prf:ref}`def-sm-walker-chirality`, computed from the recorded slices
`will_clone[t-1]`, `companions_clone[t-1]`, `fitness[t-1]`, and `alive_mask[t-1]`. Then the
implemented chirality channels in `src/fragile/physics/electroweak/electroweak_channels.py` are
exactly

$$
\mathrm{chi\_mean}(t)=\frac{1}{N}\sum_{i=1}^{N}\chi_i(t),
\qquad
\mathrm{left\_fraction}(t)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}_{\{i\in L_t\}},
$$

$$
\mathrm{lr\_fraction}(t)=
\frac{1}{\max(|\Delta_t|,1)}
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}},
$$

$$
\mathrm{lr\_coupling\_mag}(t)=
\left|
\frac{1}{\max(N_{\Delta\to R}(t),1)}
\sum_{\substack{i\in\Delta_t\\c_c(i,t)\in R_t}}
\exp\!\left(i\frac{F_{c_c(i,t)}(t)-F_i(t)}{h_{\mathrm{eff}}}\right)
\right|,
$$

where

$$
N_{\Delta\to R}(t):=
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}}.
$$

The same-frame partition also gives the exact identities

$$
N_{\Delta\to R}(t)=0,\qquad
\mathrm{lr\_fraction}(t)=\mathrm{lr\_coupling\_mag}(t)=0,\qquad
\mathrm{chi\_mean}(t)=2\,\mathrm{left\_fraction}(t)-|A_t|/N.
$$

These follow for the recorded companion-role definitions of
{prf:ref}`prop-sm-walker-role-partition`; selecting cloning frames preserves
them.

If `cloning_frames_only=True`, these four series are further restricted to the subfamily of frames
with at least one cloning event.
:::

:::{prf:proof}
By {prf:ref}`prop-sm-walker-role-partition`, a clone companion of a
walker in $\Delta_t$ lies in the left role set, so the cross mask is empty.
The conventions in the displayed denominators then give both zero channels.
Since $A_t=L_t\sqcup R_t$ and dead walkers have chirality zero,
$N^{-1}(|L_t|-|R_t|)=2|L_t|/N-|A_t|/N$.

In `_compute_chirality_series`, the recorded tensors are sliced on
`[t_start-1:t_end-1]` and passed to `classify_walkers_vectorized`. By construction of that helper,
`classification.chi` is the tensor $\chi_i(t)$ with dead walkers assigned the value $0$, and
`classification.left_handed` is the indicator of $L_t$. The assignments

$$
\texttt{series["chi_mean"]} = \texttt{classification.chi.mean(dim=1)},
\qquad
\texttt{series["left_fraction"]} =
\texttt{classification.left_handed.float().mean(dim=1)}
$$

therefore produce the two averages above over the full recorded walker count $N$.

Next, the code forms `comp_idx = companions_clone.clamp(0,N-1)`,
`comp_is_right = gather(classification.right_handed, comp_idx)`, and
`cross_mask = classification.delta & comp_is_right`. Hence `cross_mask[t,i]` is true exactly when
$i\in\Delta_t$ and $c_c(i,t)\in R_t$. The lines

$$
\texttt{delta_count = delta_mask.float().sum(dim=1).clamp(min=1)},
\qquad
\texttt{cross_count = cross_mask.float().sum(dim=1)}
$$

give $\max(|\Delta_t|,1)$ and $N_{\Delta\to R}(t)$ respectively, so
`series["lr_fraction"] = cross_count / delta_count` is exactly the stated formula with the
zero-delta convention built in.

For the phase-transfer channel, the code computes
`phase = (comp_fitness - fitness) / h_eff`,
`phase_exp = exp(1j * phase)`, and

$$
\texttt{lr_complex}
=
\frac{
\sum_i e^{i(F_{c_c(i,t)}-F_i(t))/h_{\mathrm{eff}}}\,\mathbf{1}_{\{i\in\Delta_t,\,
c_c(i,t)\in R_t\}}
}{
\max(N_{\Delta\to R}(t),1)
}.
$$

Taking `abs()` yields the displayed $\mathrm{lr\_coupling\_mag}(t)$. Finally, if
`cloning_frames_only=True`, the code restricts all four series to
`frame_has_cloning = will_clone.any(dim=1)`, which is exactly the stated frame filter. $\square$
:::

### Dirac-spinor electroweak operator layer

The second electroweak layer maps recorded color states to four-component
vectors $\psi_i \in \mathbb{C}^4$ using the implemented map in
{prf:ref}`prop-qft-ew-spinor-realization`. Its matrix bilinears use the
chiral projectors from {prf:ref}`def-lqft-chiral-projectors`. The implementation
constructs the following measurement channels:

$$
J_L^\mu = \bar\psi\gamma^\mu P_L\psi,
\qquad
J_R^\mu = \bar\psi\gamma^\mu P_R\psi,
\qquad
J_V^\mu = \bar\psi\gamma^\mu\psi,
$$

$$
O_L=\bar\psi P_L\psi,
\qquad
O_R=\bar\psi P_R\psi,
\qquad
O_{LR}=\bar\psi P_L\psi\ \text{on }L\!\to\!R\text{ pairs}.
$$

Multiplication by the implemented scalar phases gives channels carrying the following historical labels:

$$
J_{U(1)}^\mu,\qquad J_{L,U(1)}^\mu,\qquad J_{L,SU(2)}^\mu,\qquad J_{R,SU(2)}^\mu.
$$

In code, these are recorded as real bilinears,
$\operatorname{Re}(\bar\psi_i\Gamma P\psi_j)$ and
$\operatorname{Re}(U_{ij}\bar\psi_i\Gamma P\psi_j)$, before time correlators are constructed.

In code, these appear as
`j_vector_L`,
`j_vector_R`,
`j_vector_V`,
`o_scalar_L`,
`o_scalar_R`,
`j_vector_walkerL`,
`j_vector_walkerR`,
`j_vector_L_walkerL`,
`j_vector_R_walkerR`,
`o_yukawa_LR`,
`o_yukawa_RL`,
`j_vector_u1`,
`j_vector_L_u1`,
`j_vector_L_su2`,
`j_vector_R_su2`,
`parity_violation_dirac`,
and `parity_violation_walker`.

The implementation interprets these channels as follows:

- `j_vector_L_su2`: left-current bilinear multiplied by the scalar returned by `compute_su2_gauge_link`, used as a W-like proxy.
- `j_vector_u1`: vector bilinear multiplied by the fitness-difference phase, used as a photon-like proxy.
- `j_vector_L_u1`: left-current bilinear multiplied by that phase, used as a neutral-current proxy.
- `o_yukawa_LR`: cross-chirality scalar bilinear, used as the Dirac/Yukawa mass proxy.
- `parity_violation_dirac` and `parity_violation_walker`: asymmetry diagnostics comparing left and
  right sectors at the projector and walker-role levels.

:::{div} feynman-prose
The routine named `compute_su2_gauge_link` returns one complex number of unit
modulus. Its phase uses an absolute fitness difference, so reversing the edge
leaves it unchanged. The determinant and orientation calculations in the next
proposition explain why it is a scalar modulation rather than an implemented
$SU(2)$ connection. The bilinear and parity diagnostics can still be computed
and compared under their stated definitions.
:::



:::{prf:proposition} Current Dirac-Spinor Realization
:label: prop-qft-ew-spinor-realization

**Rigor Class:** F (Implementation-Exact)

Assume $d=3$ so that the color states admit the implemented map
$c_i(t)\mapsto \psi_i(t)\in\mathbb{C}^4$. For each retained frame $t$ and walker index $i$, let

$$
j=c_d(i,t)
$$

be the recorded distance companion, let $\chi_i(t)\in\{+1,-1,0\}$ be the walker-role chirality
computed from the clone companion data, and define the validity mask

$$
V_t(i):=
\mathbf{1}_{\{\mathrm{color\_valid}_i(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{color\_valid}_j(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{alive}_i(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{alive}_j(t)\}}
\cdot
\mathbf{1}_{\{j\neq i\}}.
$$

Let the pair classes be

$$
LL_t=\{i:V_t(i)=1,\ \chi_i(t)>0,\ \chi_j(t)>0\},
\qquad
RR_t=\{i:V_t(i)=1,\ \chi_i(t)<0,\ \chi_j(t)<0\},
$$

$$
LR_t=\{i:V_t(i)=1,\ \chi_i(t)>0,\ \chi_j(t)<0\},
\qquad
RL_t=\{i:V_t(i)=1,\ \chi_i(t)<0,\ \chi_j(t)>0\}.
$$

With unit edge weights, define for any mask $M_t\subseteq\{1,\dots,N\}$ and any pair observable
$B_t(i)$

$$
\operatorname{Avg}_{M_t}[B]
:=
\frac{
\sum_{i=1}^{N}\mathbf{1}_{\{i\in M_t\}}\,B_t(i)
}{
\max(|M_t|,10^{-12})
}.
$$

Further define the real bilinears

$$
B_{\Gamma,P}(i,t):=
\operatorname{Re}\!\bigl(\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
$$

$$
B_{\Gamma,P}^{U(1)}(i,t):=
\operatorname{Re}\!\bigl(U_{ij}^{(1)}(t)\,\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
\qquad
U_{ij}^{(1)}(t)=
\exp\!\left(i\frac{F_j(t)-F_i(t)}{h_{\mathrm{eff}}}\right),
$$

$$
B_{\Gamma,P}^{SU(2)}(i,t):=
\operatorname{Re}\!\bigl(U_{ij}^{(2)}(t)\,\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
\qquad
U_{ij}^{(2)}(t)=
\exp\!\left(
i\,
\frac{|F_j(t)-F_i(t)|}{|F_j(t)-F_i(t)|+\epsilon_{\mathrm{clone}}}
\cdot
\frac{\pi}{2h_{\mathrm{eff}}}
\right).
$$

Here $U_{ij}^{(2)}$ is the scalar phase returned by
`compute_su2_gauge_link`, as distinguished in
{prf:ref}`thm-sm-ew-operator-layers`. Its absolute fitness difference makes
$U_{ji}^{(2)}=U_{ij}^{(2)}$, whereas inverse-oriented transport would require
$U_{ji}^{(2)}=(U_{ij}^{(2)})^{-1}$. Multiplication by this scalar is the
implemented bilinear modulation; it is not a matrix-valued $SU(2)$ link.
Indeed, representing it as $U_{ij}^{(2)}I_2$ gives determinant
$(U_{ij}^{(2)})^2$, which is generally not one.

Then the current Dirac-spinor pipeline computes exactly the operator series

$$
j_{\mathrm{vector},L}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}\right],
\qquad
j_{\mathrm{vector},R}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}\right],
$$

$$
j_{\mathrm{vector},V}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
\qquad
o_{\mathrm{scalar},L}(t)=\operatorname{Avg}_{V_t}[B_{I,P_L}],
\qquad
o_{\mathrm{scalar},R}(t)=\operatorname{Avg}_{V_t}[B_{I,P_R}],
$$

$$
j_{\mathrm{vector},\mathrm{walkerL}}(t)=
\operatorname{Avg}_{LL_t\cup LR_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
\qquad
j_{\mathrm{vector},\mathrm{walkerR}}(t)=
\operatorname{Avg}_{RR_t\cup RL_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
$$

$$
j_{\mathrm{vector},L,\mathrm{walkerL}}(t)=
\operatorname{Avg}_{LL_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}\right],
\qquad
j_{\mathrm{vector},R,\mathrm{walkerR}}(t)=
\operatorname{Avg}_{RR_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}\right],
$$

$$
o_{\mathrm{yukawa},LR}(t)=\operatorname{Avg}_{LR_t}[B_{I,P_L}],
\qquad
o_{\mathrm{yukawa},RL}(t)=\operatorname{Avg}_{RL_t}[B_{I,P_R}],
$$

$$
j_{\mathrm{vector},U(1)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}^{U(1)}\right],
\qquad
j_{\mathrm{vector},L,U(1)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}^{U(1)}\right],
$$

$$
j_{\mathrm{vector},L,SU(2)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}^{SU(2)}\right],
\qquad
j_{\mathrm{vector},R,SU(2)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}^{SU(2)}\right],
$$

and the parity diagnostics

$$
\mathrm{pv}_{\mathrm{dirac}}(t)=
\frac{j_{\mathrm{vector},L}(t)^2-j_{\mathrm{vector},R}(t)^2}
{j_{\mathrm{vector},L}(t)^2+j_{\mathrm{vector},R}(t)^2+\varepsilon_{\mathrm{pv}}},
$$

$$
\mathrm{pv}_{\mathrm{walker}}(t)=
\frac{j_{\mathrm{vector},\mathrm{walkerL}}(t)^2-j_{\mathrm{vector},\mathrm{walkerR}}(t)^2}
{j_{\mathrm{vector},\mathrm{walkerL}}(t)^2+j_{\mathrm{vector},\mathrm{walkerR}}(t)^2+\varepsilon_{\mathrm{pv}}},
\qquad
\varepsilon_{\mathrm{pv}}=10^{-30}.
$$

The same routine also records the pair-count diagnostics

$$
n_{\mathrm{valid}}(t)=|V_t|,
\qquad
n_{LL}(t)=|LL_t|,
\qquad
n_{RR}(t)=|RR_t|,
\qquad
n_{LR}(t)=|LR_t|.
$$
:::

:::{prf:proof}
The helper `_compute_dirac_spinor_channels` first resolves the retained frame interval, computes
color states on that interval, reads the clone companions for walker classification, and reads the
distance companions for the spinor pairing. It then sets
`sample_indices = [0,\dots,N-1]` and `neighbor_indices = companions_distance.unsqueeze(-1)`, so
the pair for walker $i$ is exactly $(i,c_d(i,t))$.

Inside `compute_electroweak_spinor_operators`, the validity mask is
`valid = v_i & v_j & (first_nb != sample_indices)`, with `v_i` and `v_j` requiring both color
validity and `alive`. This is precisely $V_t(i)$. The chirality masks `both_L`, `both_R`,
`cross_LR`, and `cross_RL` are exactly the four sets $LL_t$, $RR_t$, $LR_t$, and $RL_t$ above.

The helper `_compute_chiral_bilinear` builds the matrix
$M=\gamma^0\Gamma$ and, when present, right-multiplies by the chiral projector $P_L$ or $P_R$.
It evaluates $\psi_i^\dagger M\psi_j$, multiplies by the requested gauge link if present, and
returns `bilinear.real.float()`. Hence every recorded spinor operator is the real part of the
corresponding complex bilinear.

The helper `_vector_current` sums the three spatial gamma-matrix bilinears and divides by $3$;
`_scalar_op` uses $\Gamma=I$. Both helpers average over the requested mask through `_avg`, whose
denominator is the masked weight sum clamped below by $10^{-12}$. In the active dashboard
`sample_edge_weights` is not supplied, so all weights are $1$ and `_avg` becomes the stated masked
arithmetic mean. The named assignments in the function body are exactly the displayed formulas for
`j_vector_L`, `j_vector_R`, `j_vector_V`, `o_scalar_L`, `o_scalar_R`,
`j_vector_walkerL`, `j_vector_walkerR`, `j_vector_L_walkerL`,
`j_vector_R_walkerR`, `o_yukawa_LR`, `o_yukawa_RL`,
`j_vector_u1`, `j_vector_L_u1`, `j_vector_L_su2`, and `j_vector_R_su2`.
The `_count` helper simultaneously returns the displayed cardinalities
`n_valid_pairs`, `n_valid_pairs_LL`, `n_valid_pairs_RR`, and `n_valid_pairs_LR`.

Finally, the function squares the already averaged current series and inserts them into the two
rational expressions defining `parity_violation_dirac` and `parity_violation_walker`, with the
regularizer `eps_pv = 1e-30`. This proves the claim. $\square$
:::

### Legacy phase/doublet proxy construction

The older U(1)/SU(2) phase and doublet channels are retained for continuity, comparison, and gauge
coherence diagnostics. They remain valid observables, but they should be read as a legacy proxy
family rather than the primary electroweak matter-sector story.

:::{prf:theorem} Active Electroweak Mass-Fit Domain
:label: thm-qft-ew-active-pipeline

**Rigor Class:** F (Implementation-Exact)

In the current dashboard pipeline, the electroweak mass fitter acts only on correlator keys
present in `state["electroweak_correlator_output"].correlators`. Consequently, the fitted
electroweak masses are extracted only from the user-selected legacy electroweak channels together
with the user-selected chirality channels and, when enabled, the user-selected Dirac-spinor
channels. No additional clustering observable or latent-dimension proxy enters the mass fit unless
it has first been materialized as a correlator key in that pipeline result.
:::

:::{prf:proof}
The electroweak correlator tab first collects the user-selected channel names from the U(1), SU(2),
mixed, symmetry-breaking, parity-velocity, and chirality selectors. It passes that list to
`compute_electroweak_channels(history, channels=selected_channels, config=cfg)`, converts the
output to a `PipelineResult`, and stores it as `state["electroweak_correlator_output"]`.

If `enable_dirac_spinors=True`, the helper `_compute_dirac_spinor_channels` iterates only over the
user-selected entries of the Dirac-spinor selector. For each selected key `ch_name` that matches a
field of `ElectroweakSpinorOutput`, it inserts exactly two objects into the same `PipelineResult`:
the operator time series `result.operators[ch_name]` and its FFT correlator
`result.correlators[ch_name]`. No unselected spinor key is inserted.

The electroweak mass tab then reads
`pipeline_result = state["electroweak_correlator_output"]` and forms channel groups solely from
`list(pipeline_result.correlators.keys())`. The widget selectors in that tab can only remove keys
from those groups; they cannot introduce new ones. After this filtering, the code calls
`extract_masses(pipeline_result, config)`. Therefore the fit domain is exactly the set of retained
correlator keys already present in `pipeline_result.correlators`.

In particular, the mass fitter has no direct access to any independent Higgs-clustering observable,
to any latent-dimension label, or to any undocumented diagnostic outside the stored correlator map.
Only realized correlator channels are fitted. For the identically zero
same-frame channels proved in {prf:ref}`prop-qft-ew-chirality-realization`,
the exact correlator contains no nonzero exponential signal from which a
mass can be identified. Availability of a channel key does not alter that
algebraic fact. $\square$
:::

Let $c_d(i)$ be the **distance** companion and $c_c(i)$ the **clone** companion of walker $i$. The
legacy U(1) and SU(2) phases are constructed from the fitness differences as

$$
\phi_i^{(U1)} = -\frac{F_{c_d(i)} - F_i}{\hbar_{\text{eff}}}, \qquad
\phi_i^{(SU2)} = \frac{F_{c_c(i)} - F_i}{(F_i + \epsilon_{\text{clone}})\,\hbar_{\text{eff}}}.
$$

The companion-localized amplitudes use the algorithmic distance
({prf:ref}`def-fractal-set-companion-kernel`):

$$
D_{d,i}^2 = \|x_i - x_{c_d(i)}\|^2 + \lambda_{\text{alg}}\|v_i - v_{c_d(i)}\|^2,
\qquad
D_{c,i}^2 = \|x_i - x_{c_c(i)}\|^2 + \lambda_{\text{alg}}\|v_i - v_{c_c(i)}\|^2,
$$

$$
w_{d,i} = \exp\!\left(-\frac{D_{d,i}^2}{2\epsilon_d^2}\right), \qquad
w_{c,i} = \exp\!\left(-\frac{D_{c,i}^2}{2\epsilon_c^2}\right),
$$

and amplitudes $A_{d,i}=\sqrt{w_{d,i}}$, $A_{c,i}=\sqrt{w_{c,i}}$. The dashboard computes
correlators from these complex phase series and extracts masses using the same effective-mass
relation and correlation-length definition {prf:ref}`def-correlation-length`.

The dashboard proxies are computed from phase dispersion (a diagnostic for phase coherence, not a
direct measurement of the physical couplings in {doc}`07_qft_calibration_report`):

$$
g_1^{\text{proxy}} = \operatorname{std}(\phi^{(U1)}), \qquad
g_2^{\text{proxy}} = \operatorname{std}(\phi^{(SU2)}),
$$

$$
\sin^2\theta_W^{\text{proxy}} = \frac{(g_1^{\text{proxy}})^2}{(g_1^{\text{proxy}})^2+(g_2^{\text{proxy}})^2},
\qquad
\tan\theta_W^{\text{proxy}} = \frac{g_1^{\text{proxy}}}{g_2^{\text{proxy}}}.
$$

The coupling estimates displayed for this proxy family follow directly from Volume 2:

$$
g_1^{\text{est}} = \sqrt{\frac{\hbar_{\text{eff}}}{\epsilon_d^2}}, \qquad
g_2^{\text{est}} = \sqrt{\frac{2\hbar_{\text{eff}}}{\epsilon_c^2}\frac{C_2(2)}{C_2(d)}}.
$$

**Calibration cross-check.** The dashboard label `g1_est (N1=1)` corresponds to the simplified
$\mathcal{N}_1(T,d)=1$ normalization. To compare with the calibration report, rescale via
$g_1 = g_1^{\text{est}}\sqrt{\mathcal{N}_1(T,d)}$ and use the report's $g_2$ directly.

**Measurement note.** The active `electroweak_channels.py` path resolves $\epsilon_d$ and
$\epsilon_c$ from `RunHistory.params`, keeps $\lambda_{\text{alg}}=0$ in that path, and uses
$\hbar_{\text{eff}}$ and $\epsilon_{\text{clone}}$ as the main analysis-level controls. The
projector-based spinor path additionally uses `mass`, `ell0`, and `ell0_method` when constructing
color states and Dirac spinors. The UI merges chirality, spinor, and legacy proxy correlators into
a single electroweak result object before passing them to the mass-extraction tab.

**Legacy proxy reference mapping.**

| Electroweak channel | Proxy reference (GeV) | Dashboard mapping |
| --- | --- | --- |
| `u1_phase` | 0.000511 | electron |
| `u1_dressed` | 0.105658 | muon |
| `su2_phase` | 80.379 | $W$ boson |
| `su2_doublet` | 91.1876 | $Z$ boson |
| `ew_mixed` | 1.77686 | tau |

These references are dashboard anchors for visual comparison of the legacy proxy masses. The
chirality and projector layers are not constrained to this five-channel mapping; the mass tab fits
whatever electroweak channels are selected. The calibration inversion in
{doc}`07_qft_calibration_report` uses measured couplings
$(\alpha_{\text{em}}, \sin^2\theta_W, \alpha_s)$ at a chosen scale instead of these proxy masses.

### Legacy U(1) phase channel (`u1_phase`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → typically smaller phase winding → often lighter $m_{u1}$. |
| Distance companion selection | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → tighter locality → often heavier $m_{u1}$ (validate by sweep). |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Time step | $\Delta t$ | `KineticOperator.delta_t` | Changes the generating dynamics; relabeling a fixed analysis time unit instead rescales rates uniformly. |

**Operator (U(1) phase mean):**

$$
O_{u1}(t) = \left\langle e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Increasing $\hbar_{\text{eff}}$ tends to reduce phase dispersion and lengthen the correlator.
- Decreasing $\epsilon_d$ in the generating run typically tightens companion locality and shortens
  the correlator; confirm empirically.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting and can shorten the correlator. The active dashboard route keeps this term off.
- Fitness coupling parameters ($\epsilon_F$, fitness weights) can shift the fitness differences and
  therefore the U(1) phase spread.

### Legacy U(1) dressed channel (`u1_dressed`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Distance temperature | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → sharper amplitude localization → often heavier $m_{u1,d}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{u1,d}$. |

**Operator (amplitude-weighted U(1) phase):**

$$
O_{u1,d}(t) = \left\langle A_{d,i}\,e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}},
\qquad A_{d,i}=\sqrt{w_{d,i}}.
$$

**Sweep hypotheses to check:**
- Use $\epsilon_d$ from the recorded run to control the locality of the U(1) amplitude envelope;
  tighter locality often shortens the plateau.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting to the same envelope. The active dashboard route keeps this term fixed at zero.
- Use $\hbar_{\text{eff}}$ to control the overall phase winding without changing locality.

### Legacy SU(2) phase channel (`su2_phase`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{su2}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → smaller score → often lighter $m_{su2}$. |
| Clone companion selection | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → tighter clone locality → often heavier $m_{su2}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier SU(2) proxy. |

**Operator (SU(2) phase mean):**

$$
O_{su2}(t) = \left\langle e^{i\phi_i^{(SU2)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Decreasing $\epsilon_{\text{clone}}$ tends to increase the phase score magnitude and shorten the
  correlator (confirm by sweep).
- Decreasing $\epsilon_c$ in the generating run typically tightens clone pairing and increases
  $m_{su2}$.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it can
  also raise the proxy mass. The active dashboard route keeps this term off.
- Adjust $\hbar_{\text{eff}}$ to rescale phase winding without changing clone topology.

### Legacy SU(2) doublet channel (`su2_doublet`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Clone temperature | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → tighter pairing → often heavier $m_{su2,d}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → often lighter $m_{su2,d}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier SU(2) doublet proxy. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{su2,d}$. |

**Operator (clone-paired doublet):**

$$
O_{su2,d}(t) = \left\langle A_{c,i}e^{i\phi_i^{(SU2)}(t)}
+ A_{c,c(i)}e^{i\phi_{c(i)}^{(SU2)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Tightening clone locality (smaller $\epsilon_c$) often sharpens the doublet and shortens the
  plateau; verify with sweeps.
- Use $\epsilon_{\text{clone}}$ to regulate phase-score magnitude without changing the pairing
  graph.

### Legacy mixed electroweak channel (`ew_mixed`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| U(1) locality | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → often heavier $m_{\text{EW}}$. |
| SU(2) locality | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → often heavier $m_{\text{EW}}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → often lighter $m_{\text{EW}}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier mixed proxy. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{\text{EW}}$. |

**Operator (U(1) × SU(2) phase product):**

$$
O_{\text{EW}}(t) = \left\langle A_{d,i}A_{c,i}\,e^{i(\phi_i^{(U1)}(t)+\phi_i^{(SU2)}(t))}
\right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Use $\epsilon_d$ and $\epsilon_c$ to control the relative U(1) vs. SU(2) localization; the mixed
  channel is typically the most sensitive to simultaneous changes in both.
- Adjust $\hbar_{\text{eff}}$ and $\epsilon_{\text{clone}}$ to shift phase winding without changing
  the companion graphs; validate shifts with the Electroweak tab fits.

### Legacy empirical tuning status (QFT baseline)

The electroweak tuning runs in `electroweak_tuning_report.md` (zero reward, viscosity-only,
analysis-level Electroweak tab) support the theoretical mapping for **coupling estimates** but not
for **mass ratios**:

- **Couplings**: adjusting $\epsilon_d$ and $\epsilon_c$ moves $g_1^{\text{est}}$ and
  $g_2^{\text{est}}$ as predicted, and defaults land within $\sim$3.5% of the $M_Z$ targets.
- **Ratios**: observed proxy ratios remain $\mathcal{O}(1)$ across analysis-level sweeps. The best
  $m_{\text{su2\_doublet}}/m_{\text{u1\_dressed}}$ achieved $\sim 5.83$ (target $\sim 863$), and
  $m_{\text{u1\_phase}}/m_{\text{u1\_dressed}}$ stays orders of magnitude above the observed
  electron/muon ratio.
- **Interpretation**: the electroweak channels are therefore **phase-coherence diagnostics** in the
  current baseline, not a calibrated reproduction of electroweak mass hierarchies. This aligns with
  the coupling inversion workflow in {doc}`07_qft_calibration_report`, which calibrates couplings
  directly rather than through proxy mass ratios.

(sec-qft-calibration-ratio-sieve)=
## Ratio-sieve theorems (symbolic constraints)

Define the Channels-tab mass ratios (symbolic targets):

$$
R_{\sigma\pi} := \frac{m_\sigma}{m_\pi}, \qquad
R_{\rho\pi} := \frac{m_\rho}{m_\pi}, \qquad
R_{G\pi} := \frac{m_G}{m_\pi}, \qquad
R_{N\pi} := \frac{m_N}{m_\pi}.
$$

For the current calibration targets, fix

$$
R_{\rho\pi} = 5.5, \qquad R_{N\pi} = 6.7,
$$
and treat $R_{\sigma\pi}, R_{G\pi}$ as symbolic until anchored by data.

:::{div} feynman-prose
These numbers are chosen calibration targets. The following algebra supplies necessary constraints within a specified scale and coupling model. It does not establish that every selected dashboard channel has an asymptotic mass, or that satisfying the constraints reproduces the targets.
:::



:::{prf:theorem} Ratio invariance under relabeling a fixed time unit
:label: thm-qft-ratio-rescale

Fix the correlator values $C_\chi[n]$ at integer lags and assign a time unit
$\Delta\tau>0$ per lag. Wherever the effective mass is defined,

$$
m_\chi[n]=-\frac1{\Delta\tau}\log\frac{C_\chi[n+1]}{C_\chi[n]}.
$$

Replacing only the assigned unit by $s\Delta\tau$, $s>0$, sends
$m_\chi[n]$ to $m_\chi[n]/s$ and leaves ratios unchanged.
:::

:::{prf:proof}
The correlator quotient remains fixed while the prefactor is divided by $s$.
The common factor cancels in a ratio. A new generating time step or recording
stride can change the correlator sequence itself and is outside this
unit-relabeling statement.
:::

:::{div} feynman-prose
Changing only the unit label cannot tune ratios. Hold the integrator step and recording settings controlled during parameter sweeps, and treat changes to either as changes to the experiment.
:::



:::{prf:corollary} Dimensionless Reduction of Ratio Dependence
:label: cor-qft-ratio-dimensionless

In a model covariant under its declared changes of units, a mass ratio is a function of dimensionless inputs. The combinations in {prf:ref}`thm-dimensionless-ratios` and the coupling conventions provide the following useful coordinates.
The displayed combinations provide reduced coordinates for the declared scale model:

$$
(\sigma_{\text{sep}}, \eta_{\text{time}}, \kappa; \; g_1, g_2, g_3; \; N, d; \; \phi),
$$
where $\phi := m\ell_0/\hbar_{\text{eff}}$ is the phase-winding combination from
{prf:ref}`thm-sm-su3-emergence`.
:::

:::{prf:proof}
{prf:ref}`thm-dimensionless-ratios` enumerates the fundamental dimensionless
ratios built from $(m,\tau,\rho,\epsilon_c)$; the gauge couplings are themselves dimensionless
({prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`), and
the displayed phase factor contains $\phi$. Other dimensionless regularizers, operator settings, or kernel parameters must be included if they vary; dimensionlessness alone does not make this coordinate list exhaustive.
$\square$
:::

:::{prf:theorem} A clustering envelope bounds an asymptotic decay exponent
:label: thm-qft-channel-gap-bound

Suppose the specified channel correlator has a clustering bound
$|C_\chi(t)|\leq A e^{-m_{\mathrm{gap}}t}$ with $A<\infty$, and its
nonzero asymptotic exponential rate $m_\chi$ exists. Then
$m_\chi\geq m_{\mathrm{gap}}$. In particular, if
$C_\chi(t)=Z_\chi e^{-m_\chi t}(1+o(1))$ with $Z_\chi\ne0$, the result
applies. When the model identifies
$m_{\mathrm{gap}}=\hbar_{\mathrm{eff}}\lambda_{\mathrm{gap}}$, this is the
corresponding lower bound in that normalization.
:::

:::{prf:proof}
At times with $C_\chi(t)\ne0$, take logarithms of the envelope:

$$
-\frac1t\log|C_\chi(t)|\geq m_{\mathrm{gap}}-\frac{\log A}{t}.
$$

Taking the lower limit proves the claim. For the stated leading exponential,
the left side tends to $m_\chi$. An envelope alone does not bound every
finite-lag logarithmic ratio; a fitted plateau estimates an asymptotic rate
only with control of competing contributions and fit error.
:::

:::{prf:corollary} Ratio-Driven Bounds on $\lambda_{\text{gap}}, \eta_{\text{time}}, \kappa$
:label: cor-qft-ratio-gap-bounds

Let

$$
m_{\min} := \min(m_\pi, m_\sigma, m_\rho, m_G, m_N)
       = m_\pi \cdot \min(1, R_{\sigma\pi}, R_{\rho\pi}, R_{G\pi}, R_{N\pi}).
$$
Then

$$
\lambda_{\text{gap}} \leq \frac{m_{\min}}{\hbar_{\text{eff}}},
\qquad
\eta_{\text{time}} = \tau \lambda_{\text{gap}} \leq \tau \frac{m_{\min}}{\hbar_{\text{eff}}},
\qquad
\kappa = \frac{1}{\rho \hbar_{\text{eff}} \lambda_{\text{gap}}} \geq \frac{1}{\rho m_{\min}}.
$$

These are necessary bounds when the chosen channels possess the asymptotic rates and common clustering envelope of the preceding theorem. Applying them to fitted values must retain fit and approximation uncertainty.
:::

:::{prf:proof}
Apply the preceding theorem to every included nonzero asymptotic channel rate and take their minimum. Divide by $\hbar_{\mathrm{eff}}>0$, multiply by $\tau>0$, and invert the positive inequality for $\rho\hbar_{\mathrm{eff}}\lambda_{\mathrm{gap}}$. These operations give the three bounds.
:::


:::{prf:corollary} Explicit Pruning Bounds for $R_{\rho\pi}=5.5$, $R_{N\pi}=6.7$
:label: cor-qft-ratio-numeric-bounds

With the fixed targets $R_{\rho\pi}=5.5$ and $R_{N\pi}=6.7$,

$$
m_\rho = 5.5\,m_\pi, \qquad m_N = 6.7\,m_\pi,
$$
and

$$
m_{\min} = m_\pi \cdot \min(1, R_{\sigma\pi}, R_{G\pi})
$$
because both $5.5$ and $6.7$ exceed $1$. Therefore the ratio-sieve bounds become

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\eta_{\text{time}} \leq \tau \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\kappa \geq \frac{1}{\rho\,m_\pi\,\min(1, R_{\sigma\pi}, R_{G\pi})}.
$$
In particular, if future calibration anchors give $R_{\sigma\pi} \geq 1$ and
$R_{G\pi} \geq 1$, then

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}, \qquad
\eta_{\text{time}} \leq \tau \frac{m_\pi}{\hbar_{\text{eff}}}, \qquad
\kappa \geq \frac{1}{\rho\,m_\pi}.
$$
:::

:::{prf:proof}
Substitute $m_\rho=5.5m_\pi$ and $m_N=6.7m_\pi$ into the minimum. Since both multipliers exceed one, neither lowers the minimum. Apply the previous corollary and simplify; if the remaining ratios are also at least one, the minimum multiplier is one.
:::


:::{prf:definition} Candidate constraints in the declared calibration model
:label: cor-qft-parameter-sieve

Within the declared hierarchy and coupling model, define the algebraically admissible candidate set by the following constraints. They are necessary for matching the specified asymptotic masses in that model; they are not sufficient for agreement of measured plateaus:

1. **Hierarchy constraint** ({prf:ref}`thm-mass-scales`):

$$
m_{\text{friction}} \ll m_{\text{gap}} < m_{\text{MF}} < m_{\text{clone}}.
$$

2. **Dimensionless ratios** ({prf:ref}`thm-dimensionless-ratios`):

$$
\sigma_{\text{sep}} = \frac{\epsilon_c}{\rho}, \quad
\eta_{\text{time}} = \tau\lambda_{\text{gap}}, \quad
\kappa = \frac{1}{\rho \hbar_{\text{eff}} \lambda_{\text{gap}}}.
$$

3. **Gap lower bound (all channels)** ({prf:ref}`thm-qft-channel-gap-bound`):

$$
m_\chi \geq \hbar_{\text{eff}} \lambda_{\text{gap}} \quad \text{for } \chi \in \{\pi,\sigma,\rho,G,N\}.
$$

4. **Ratio-sieve bounds** (from {prf:ref}`cor-qft-ratio-numeric-bounds`):

$$
R_{\rho\pi} = 5.5, \qquad R_{N\pi} = 6.7,
$$

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\kappa \geq \frac{1}{\rho\,m_\pi\,\min(1, R_{\sigma\pi}, R_{G\pi})}.
$$

5. **Coupling inversion manifold** ({prf:ref}`cor-qft-coupling-inversion-manifold`):

$$
\epsilon_c = \sqrt{\frac{2\hbar_{\text{eff}}C_2(2)}{C_2(d)\,g_2^2}}, \quad
\rho = g_2\sqrt{\frac{2\hbar_{\text{eff}}}{m^2}}, \quad
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}.
$$

A candidate failing a required model constraint is excluded from that declared regime. Finite-data estimates require uncertainty margins before they can justify exclusion.

:::

### Pruning procedure (pre-sweep)

Use the checklist above as a deterministic filter before running large parameter sweeps.

1. **Fix absolute time scale**: choose $\Delta t$ (and `record_every`) and hold fixed for all
   runs so ratios are comparable ({prf:ref}`thm-qft-ratio-rescale`).
2. **Invert couplings**: for chosen $(g_1,g_2,g_3)$ and QSD statistics, solve for
   $(\epsilon_d,\epsilon_c,\nu,\epsilon_F,\rho,\tau)$ using
   {prf:ref}`cor-qft-coupling-inversion-manifold`. Discard any candidate that violates the
   hierarchy in {prf:ref}`thm-mass-scales`.
3. **Check dimensionless diagnostics**: compute
   $(\sigma_{\text{sep}}, \eta_{\text{time}}, \kappa)$ from
   {prf:ref}`thm-dimensionless-ratios`. Discard candidates outside the stable regime indicated by
   prior calibrated runs.
4. **Pilot estimate of $m_\pi$**: run a short QSD‑valid trajectory and extract $m_\pi$ from the
   pseudoscalar correlator ({prf:ref}`def-euclidean-correlator-fg`,
   {prf:ref}`def-two-point-connected`, {prf:ref}`def-correlation-length`).
5. **Apply ratio bounds**: enforce {prf:ref}`cor-qft-ratio-numeric-bounds` using the pilot
   estimate of $m_\pi$ (and symbolic $R_{\sigma\pi}, R_{G\pi}$ if still unanchored). Discard
   candidates that violate the inequalities.

:::{div} feynman-prose
Treat a pilot fit as an estimate with uncertainty. Exclusion by an asymptotic spectral constraint is justified only when its hypotheses and the error margin hold for that channel. A missing plateau or an identically zero observable supplies no mass estimate.
:::



:::{prf:corollary} Coupling-Inversion Manifold (Symbolic Constraints)
:label: cor-qft-coupling-inversion-manifold

For fixed positive normalization factors and QSD statistics, the coupling assignments constrain the corresponding ranges and amplitudes. The displayed inversion additionally adopts the indicated relations for $\rho$ and $\tau$; a fitness coupling must be specified to fix $\epsilon_F$. These are algebraic constraints within the chosen model, using:
{prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`,
{prf:ref}`thm-u1-coupling-constant`, and {prf:ref}`thm-effective-planck-constant`. In particular,

$$
\epsilon_c = \sqrt{\frac{2\hbar_{\text{eff}}C_2(2)}{C_2(d)\,g_2^2}},
\qquad
\rho = g_2\sqrt{\frac{2\hbar_{\text{eff}}}{m^2}},
\qquad
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}.
$$

These equations define a restricted candidate set when all their relations are imposed. They do not fix omitted dimensionless parameters or the QSD statistics produced by a new run. Calling that set a manifold additionally requires the usual regularity and rank conditions for its defining equations.
:::

:::{prf:proof}
The first formula follows by solving
$g_2^2=2\hbar_{\mathrm{eff}}C_2(2)/(\epsilon_c^2C_2(d))$ for its positive
range. The displayed $\rho$ relation is equivalent to imposing
$g_2^2=m^2\rho^2/(2\hbar_{\mathrm{eff}})$, and the $\tau$ relation is
equivalent to imposing $\hbar_{\mathrm{eff}}=m\epsilon_c^2/(2\tau)$.
Thus they are compatible algebraic substitutions when these relations are
part of the selected model. The other coupling assignments constrain their
own parameters with their normalization factors held fixed; they supply no
additional equation for a parameter absent from those assignments.
:::


(sec-qft-calibration-code)=
## Theory-to-code map

The QFT modules mirror the notation of Volume 2. The main parameter hooks are:

- Companion selection kernel ({prf:ref}`def-fractal-set-companion-kernel`):
  run parameters are stored in `RunHistory.params` and consumed in the active analysis path by
  `src/fragile/physics/electroweak/electroweak_channels.py`.
  - Distance companion temperature $\epsilon_d$ is resolved from recorded run parameters rather than
    freely retuned inside the active electroweak channel path.
  - Clone companion temperature $\epsilon_c$ is likewise resolved from the recorded run parameters.
  - The active electroweak channel path keeps $\lambda_{\text{alg}} = 0$.
- Two-channel fitness ({prf:ref}`def-fractal-set-two-channel-fitness`):
  `src/fragile/physics/fractal_gas/fitness.py`.
- Cloning score ({prf:ref}`def-fractal-set-cloning-score`): `CloneOperator` parameters
  in `src/fragile/physics/fractal_gas/cloning.py`.
- Viscous force and color coupling ({prf:ref}`def-fractal-set-viscous-force`,
  {prf:ref}`thm-sm-su3-emergence`): `KineticOperator` parameters in
  `src/fragile/physics/fractal_gas/kinetic_operator.py`.
- Anisotropic diffusion ({prf:ref}`def-fractal-set-anisotropic-diffusion`):
  `src/fragile/physics/fractal_gas/kinetic_operator.py`.

The electroweak correlators are computed in:
- `src/fragile/physics/electroweak/chirality.py` (walker-role chirality partition and
  autocorrelation observables).
- `src/fragile/physics/electroweak/electroweak_spinors.py` (Dirac-spinor currents, Yukawa
  bilinears, and parity diagnostics).
- `src/fragile/physics/electroweak/electroweak_channels.py` (legacy proxy channels plus chirality
  channels, merged into the shared correlator pipeline).
- `src/fragile/physics/app/electroweak_correlators.py` (dashboard channel selection and operator
  family wiring).
- `src/fragile/physics/app/electroweak_mass_tab.py` (Bayesian mass extraction for the selected
  electroweak channels).

The broader correlator and mass-extraction machinery lives in:
- `src/fragile/physics/new_channels/correlator_channels.py`
- `src/fragile/physics/mass_extraction/`

Analysis window choices (fit start/stop, plateau detection, covariance model, priors) change
measurement quality, not the underlying physics.

(sec-qft-calibration-workflow)=
## Calibration workflow (parameter tuning loop)

:::{div} feynman-prose
1. Fix the observable, recorded frame convention, masks, and analysis settings.
   Verify its exact algebra, including any zero-channel identity.
2. Choose target ratios and a reference unit, recording which are input anchors.
   Use the report and notebook for the documented comparison protocol.
3. Generate histories with controlled time step, recording stride, and burn-in.
   Use the applicable QSD convergence result and observed stationarity diagnostics
   to assess the measurement window.
4. Measure the nonzero correlators and check fit-window stability, uncertainty,
   competing decay terms, and the effect of connected versus unconnected data.
5. Sweep a generating parameter or an analysis parameter separately, recording
   which changed. Verify the suggested direction from the measured response.
6. Apply scale, ratio, or gap constraints only within the model and uncertainty
   regime that justifies them. Repeat on independent runs before interpreting
   a fitted scale as a reproducible channel feature.
:::
