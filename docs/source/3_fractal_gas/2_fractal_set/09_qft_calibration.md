# QFT Calibration: Channel Knobs and Mass Plateaus

This chapter connects Volume 3 theory to the QFT calibration code. It explains which algorithm
parameters move which channel mass plateaus and why, using only the definitions and theorems
already established in Part II.

**Prerequisites**: {doc}`01_fractal_set`, {doc}`03_lattice_qft`, {doc}`04_standard_model`,
{doc}`05_yang_mills_noether`, {doc}`06_empirical_validation`,
{doc}`07_qft_calibration_report`, {doc}`08_qft_calibration_notebook`.

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

Volume 3 fixes how algorithmic parameters map to gauge couplings. The key identities are:

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

These formulas justify the direct calibration rule: decreasing $\epsilon_d$ or $\epsilon_c$
strengthens the corresponding coupling, while increasing $\nu$ strengthens the color coupling.
Increasing $\epsilon_F$ weakens the U(1) fitness coupling.

The emergent mass scales are fixed by {prf:ref}`thm-mass-scales`:

$$
m_{\text{clone}} = 1/\epsilon_c,\quad
m_{\text{MF}} = 1/\rho,\quad
m_{\text{gap}} = \hbar_{\text{eff}}\lambda_{\text{gap}},\quad
m_{\text{friction}} = \gamma.
$$

Combined with {prf:ref}`def-correlation-length`, these scales determine how quickly correlators
decay. Calibration must preserve the hierarchy in {prf:ref}`thm-mass-scales` and the dimensionless
ratios in {prf:ref}`thm-dimensionless-ratios`.

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

The table below summarizes which knobs primarily move which channel families. The directionality
is an operational expectation derived from the mass-scale relations above; treat it as a guide and
validate by sweep.

| Channel family | Fractal Set ingredient | Primary knobs | Expected qualitative effect (operational) |
| --- | --- | --- | --- |
| Meson / pseudoscalar (color bilinear) | Color state from viscous force ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, $\gamma$, $\beta$, $\Delta t$ | Shorter $\rho$ or larger $\nu$ increases color coupling, typically shortening correlators (heavier masses). |
| Baryon / nucleon (color determinant) | SU(3) invariant of three color vectors ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, neighbor selection | Trilinear color invariants are sensitive to color coherence; adjust $\nu$ and $\rho$ first. |
| Glueball / gauge channel | Gauge field strength and Wilson loops ({prf:ref}`def-fractal-set-viscous-force`, {prf:ref}`def-fractal-set-wilson-loop`) | $\nu$, $\rho$ | Stronger viscous coupling or shorter $\rho$ tends to increase glueball mass scales. |
| Cloning/diversity-dominated channels | Companion kernel + cloning score ({prf:ref}`def-fractal-set-companion-kernel`, {prf:ref}`def-fractal-set-cloning-score`) | $\epsilon_c$, $\epsilon_d$, $\lambda_{\text{alg}}$, $\epsilon_{\text{clone}}$, $p_{\max}$ | Decreasing $\epsilon_c$ or $\epsilon_d$ strengthens the corresponding coupling and can shift correlator decay. |
| Fitness/U(1) phase channels | Phase potential and fitness coupling ({prf:ref}`def-fractal-set-phase-potential`, {prf:ref}`thm-u1-coupling-constant`) | $\epsilon_F$, fitness weights $(\alpha,\beta)$ | Larger $\epsilon_F$ weakens the fitness coupling, softening phase-driven oscillations. |

(sec-qft-calibration-channel-derivations)=
## Channel-by-channel calibration from first principles

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

At large Euclidean time, the correlator decays exponentially, and the channel mass is the inverse
correlation length:

$$
C_\chi(\tau) \sim Z_\chi e^{-m_\chi \tau},
\qquad
m_\chi(\tau) = -\frac{1}{\Delta \tau}\log\frac{C_\chi(\tau+\Delta\tau)}{C_\chi(\tau)},
\qquad
\xi_\chi = \frac{1}{m_\chi}
$$

using the correlation-length definition {prf:ref}`def-correlation-length` and the mass-scale
hierarchy {prf:ref}`thm-mass-scales`. The AIC-weighted plateau in the Channels tab is an
implementation of this $m_\chi(\tau)$ extraction, so all tuning rules below follow directly from
the way $O_\chi$ is constructed from Fractal Set primitives.

Two universal scaling facts apply to **all** channels:

1. **Time-scale normalization**: the extracted masses scale with the Euclidean time unit
   $\Delta\tau = \Delta t \times r_{\text{rec}}$, where $r_{\text{rec}}$ is `record_every`.
   Changing $\Delta t$ rescales all channel masses uniformly, so compare runs at fixed $\Delta t$
   whenever possible.
2. **Coupling hierarchy**: tuning must preserve the hierarchy in {prf:ref}`thm-mass-scales`
   (especially $m_{\text{MF}} = 1/\rho$ and $m_{\text{friction}}=\gamma$), otherwise the plateau
   interpretation breaks down.

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
| $\Delta t$ | `delta_t` | `KineticOperator` | Time step; rescales all masses |
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

### Scalar channel (σ, $0^{++}$)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
- Increase $\nu$ or decrease $\rho$ to strengthen the viscous coupling and shorten the scalar
  correlation length (heavier scalar mass).
- Decrease $\nu$ or increase $\rho$ to soften the coupling and lengthen the plateau (lighter
  scalar mass).
- Increasing $\gamma$ raises $m_{\text{friction}}$ and typically shortens scalar plateaus; keep the
  hierarchy $m_{\text{friction}} \ll m_{\text{gap}}$ intact.

### Pseudoscalar channel (π, $0^{-+}$)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
- Increase $m$ or $\ell_0$, or decrease $\hbar_{\text{eff}}$, to increase phase winding and shorten
  the pseudoscalar correlation length (heavier pseudoscalar).
- Decrease $m$ or $\ell_0$, or increase $\hbar_{\text{eff}}$, to reduce phase dispersion (lighter
  pseudoscalar).
- Use $\nu$ and $\rho$ only to move the **overall** meson mass scale; they shift scalar and
  pseudoscalar together via the same viscous-force coupling.

### Vector channel (ρ, $1^{--}$)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
- Increase $\nu$ or decrease $\rho$ to strengthen alignment and shorten the vector correlator
  (heavier vector mass).
- Increase $\gamma$ to speed velocity relaxation, which typically shortens vector plateaus.
- Keep $\Delta t$ fixed when comparing vector masses across runs (see the time-scale normalization
  rule above).

### Nucleon channel (baryon, color determinant)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
- Increase $\nu$ or decrease $\rho$ to tighten color coherence and increase nucleon masses.
- Adjust $\epsilon_c$, $\epsilon_d$, and $\lambda_{\text{alg}}$ to modify local companion structure
  and triplet availability; this changes baryon plateaus without altering the color definition.
- Implementation constraint: the nucleon channel requires $d=3$ and at least two neighbors; if the
  Channels tab reports `n/a`, verify that the run dimension is three and that neighbor sampling is
  adequate.

### Glueball channel ($0^{++}$, gauge sector)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
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

The current Electroweak dashboard is not a single observable family. It exposes three layers that
share the same correlator and mass-extraction pipeline but encode different pieces of the theory:

1. **Walker-role chirality observables**, built directly from the recorded cloning history.
2. **Dirac-spinor electroweak operators**, built from the Clifford realization, chiral projectors,
   and gauge-dressed bilinears.
3. **Legacy phase/doublet proxy channels**, retained as coherence diagnostics and historical
   comparison tools.

This section now follows that ordering. The first two layers describe the electroweak matter-sector
observables actually used in the current implementation. The legacy proxy layer remains useful, but
it is no longer the primary theoretical description of particle generation in the dashboard.

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
`lr_fraction`, and `lr_coupling_mag`. Their masses are extracted by the same Euclidean correlator
pipeline used elsewhere in this chapter. This is the most direct implementation of the statement
that the weak interaction couples to a left-handed sector created by cloning dynamics.

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

If `cloning_frames_only=True`, these four series are further restricted to the subfamily of frames
with at least one cloning event.

*Proof.*

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

The second electroweak layer uses the Clifford/Dirac structure established in
{prf:ref}`thm-sm-dirac-isomorphism` together with the chiral projectors from
{prf:ref}`def-lqft-chiral-projectors`. After mapping color states to Dirac spinors
$\psi_i \in \mathbb{C}^4$, the implementation constructs

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
O_{LR}=\bar\psi P_L\psi\ \text{on $L\!\to\!R$ pairs}.
$$

Gauge dressing by the U(1) and SU(2) links then gives the implemented operator channels

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

- `j_vector_L_su2`: left current dressed by the SU(2) link, used as the W-like current proxy.
- `j_vector_u1`: vector current dressed by the U(1) link, used as the photon-like proxy.
- `j_vector_L_u1`: left current dressed by the U(1) link, used as the neutral-current Z-like proxy.
- `o_yukawa_LR`: cross-chirality scalar bilinear, used as the Dirac/Yukawa mass proxy.
- `parity_violation_dirac` and `parity_violation_walker`: asymmetry diagnostics comparing left and
  right sectors at the projector and walker-role levels.

This layer is optional in the UI but fully implemented and should be documented as part of the
electroweak theory, not as an afterthought.

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

*Proof.*

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

*Proof.*

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
Only realized correlator channels are fitted. $\square$
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

The coupling estimates displayed for this proxy family follow directly from Volume 3:

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

| Parameter | Symbol | Code parameter | Effect on observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → typically smaller phase winding → often lighter $m_{u1}$. |
| Distance companion selection | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → tighter locality → often heavier $m_{u1}$ (validate by sweep). |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Time step | $\Delta t$ | `KineticOperator.delta_t` | Rescales all electroweak masses uniformly (fix for comparisons). |

**Operator (U(1) phase mean):**

$$
O_{u1}(t) = \left\langle e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}}.
$$

**Tuning rule (first-principles):**
- Increasing $\hbar_{\text{eff}}$ tends to reduce phase dispersion and lengthen the correlator.
- Decreasing $\epsilon_d$ in the generating run typically tightens companion locality and shortens
  the correlator; confirm empirically.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting and can shorten the correlator. The active dashboard route keeps this term off.
- Fitness coupling parameters ($\epsilon_F$, fitness weights) can shift the fitness differences and
  therefore the U(1) phase spread.

### Legacy U(1) dressed channel (`u1_dressed`)

| Parameter | Symbol | Code parameter | Effect on observed mass |
| --- | --- | --- | --- |
| Distance temperature | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → sharper amplitude localization → often heavier $m_{u1,d}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{u1,d}$. |

**Operator (amplitude-weighted U(1) phase):**

$$
O_{u1,d}(t) = \left\langle A_{d,i}\,e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}},
\\qquad A_{d,i}=\sqrt{w_{d,i}}.
$$

**Tuning rule (first-principles):**
- Use $\epsilon_d$ from the recorded run to control the locality of the U(1) amplitude envelope;
  tighter locality often shortens the plateau.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting to the same envelope. The active dashboard route keeps this term fixed at zero.
- Use $\hbar_{\text{eff}}$ to control the overall phase winding without changing locality.

### Legacy SU(2) phase channel (`su2_phase`)

| Parameter | Symbol | Code parameter | Effect on observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{su2}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → smaller score → often lighter $m_{su2}$. |
| Clone companion selection | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → tighter clone locality → often heavier $m_{su2}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier SU(2) proxy. |

**Operator (SU(2) phase mean):**

$$
O_{su2}(t) = \left\langle e^{i\phi_i^{(SU2)}(t)} \right\rangle_{\text{alive}}.
$$

**Tuning rule (first-principles):**
- Decreasing $\epsilon_{\text{clone}}$ tends to increase the phase score magnitude and shorten the
  correlator (confirm by sweep).
- Decreasing $\epsilon_c$ in the generating run typically tightens clone pairing and increases
  $m_{su2}$.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it can
  also raise the proxy mass. The active dashboard route keeps this term off.
- Adjust $\hbar_{\text{eff}}$ to rescale phase winding without changing clone topology.

### Legacy SU(2) doublet channel (`su2_doublet`)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
- Tightening clone locality (smaller $\epsilon_c$) often sharpens the doublet and shortens the
  plateau; verify with sweeps.
- Use $\epsilon_{\text{clone}}$ to regulate phase-score magnitude without changing the pairing
  graph.

### Legacy mixed electroweak channel (`ew_mixed`)

| Parameter | Symbol | Code parameter | Effect on observed mass |
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

**Tuning rule (first-principles):**
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

These ratios are the calibration targets that must be reproduced by a valid parameter set.
The results below turn them into **provable constraints** on parameter space, without introducing
new assumptions beyond Volume 3.

:::{prf:theorem} Ratio Invariance Under Time Rescaling
:label: thm-qft-ratio-rescale

Let $\Delta\tau = \Delta t \cdot r_{\text{rec}}$ be the Euclidean time unit, where $r_{\text{rec}}$
is `record_every`. If
$\Delta\tau \mapsto s\,\Delta\tau$ for any $s>0$, then all extracted channel masses scale as
$m_\chi \mapsto m_\chi/s$, and all channel mass ratios remain unchanged.

*Proof.* The effective mass is

$$
m_\chi(\tau) = -\frac{1}{\Delta\tau}\log\frac{C_\chi(\tau+\Delta\tau)}{C_\chi(\tau)}
$$
so scaling $\Delta\tau$ by $s$ multiplies every $m_\chi$ by $1/s$, while ratios cancel the scale.
$\square$
:::

**Consequence.** Ratio tuning cannot be done with $\Delta t$ (or `record_every`); those knobs only
set the absolute mass scale. Fix them before ratio sweeps.

:::{prf:corollary} Dimensionless Reduction of Ratio Dependence
:label: cor-qft-ratio-dimensionless

Because mass ratios are dimensionless, they can only depend on the **dimensionless combinations**
identified in {prf:ref}`thm-dimensionless-ratios`, together with dimensionless coupling inputs.
Equivalently, the ratio targets constrain only the reduced set

$$
(\sigma_{\text{sep}}, \eta_{\text{time}}, \kappa; \; g_1, g_2, g_3; \; N, d; \; \phi),
$$
where $\phi := m\ell_0/\hbar_{\text{eff}}$ is the phase-winding combination from
{prf:ref}`thm-sm-su3-emergence`.

*Justification.* {prf:ref}`thm-dimensionless-ratios` enumerates the fundamental dimensionless
ratios built from $(m,\tau,\rho,\epsilon_c)$; the gauge couplings are themselves dimensionless
({prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`), and
the only dimensionless combination in the phase factor is $\phi$.
$\square$
:::

:::{prf:theorem} Spectral-Gap Lower Bound for Channel Masses
:label: thm-qft-channel-gap-bound

For any channel operator $O_\chi$ with connected correlator $C_\chi(\tau)$, the exponential
clustering bound in {prf:ref}`thm-os-os3-fg` implies

$$
m_\chi \geq m_{\text{gap}} = \hbar_{\text{eff}} \lambda_{\text{gap}}.
$$

*Proof.* By {prf:ref}`thm-os-os3-fg`, connected correlators satisfy

$$
|C_\chi(\tau)| \leq C\,e^{-\tau/\xi}, \quad \xi = 1/m_{\text{gap}}.
$$
Taking logarithms of the positive-decay regime used for mass extraction yields
$m_\chi(\tau) \geq 1/\xi = m_{\text{gap}}$. $\square$
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

These bounds prune parameter space before any sweeps: any candidate set that violates them cannot
produce the target ratios.
$\square$
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
$\square$
:::
the
:::{prf:corollary} Parameter Sieve Checklist (Admissible Set)
:label: cor-qft-parameter-sieve

A parameter set is admissible for the target ratios if it satisfies all of:

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

Any candidate failing one of these conditions is eliminated before channel sweeps.
$\square$
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

Only survivors of this sieve proceed to full channel sweeps.

:::{prf:corollary} Coupling-Inversion Manifold (Symbolic Constraints)
:label: cor-qft-coupling-inversion-manifold

Fixing $(g_1,g_2,g_3)$ and QSD statistics $(\mathcal{N}_1,\langle K_{\text{visc}}^2\rangle)$
determines $(\epsilon_d,\epsilon_c,\nu,\epsilon_F,\rho,\tau)$ via the Volume 3 coupling identities:
{prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`,
{prf:ref}`thm-u1-coupling-constant`, and {prf:ref}`thm-effective-planck-constant`. In particular,

$$
\epsilon_c = \sqrt{\frac{2\hbar_{\text{eff}}C_2(2)}{C_2(d)\,g_2^2}},
\qquad
\rho = g_2\sqrt{\frac{2\hbar_{\text{eff}}}{m^2}},
\qquad
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}.
$$

Therefore the ratio constraints carve out a **low-dimensional admissible manifold** inside the
full parameter space: only $(\hbar_{\text{eff}}, m, \lambda_{\text{gap}})$ and the QSD statistics
remain free once couplings are fixed.
$\square$
:::

(sec-qft-calibration-code)=
## Theory-to-code map

The QFT modules mirror the notation of Volume 3. The main parameter hooks are:

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

1. **Choose targets**: Select desired mass ratios (for example baryon/meson) and a reference scale.
   Use {doc}`07_qft_calibration_report` and {doc}`08_qft_calibration_notebook` to anchor couplings
   to physical constants if needed.
2. **Run to QSD**: Ensure the hierarchy in {prf:ref}`thm-mass-scales` holds and the system reaches
   QSD before measuring.
3. **Measure channels**: Use connected correlators where appropriate
   ({prf:ref}`def-two-point-connected`) and check plateau stability and $R^2$ in the fit.
4. **Adjust primary knobs**:
   - Color channels (meson/baryon/glueball): tune $\nu$ and $\rho$ first.
   - Cloning/diversity structure: tune $\epsilon_c$, $\epsilon_d$, $\lambda_{\text{alg}}$ while
     holding the hierarchy fixed.
   - Fitness phase: tune $\epsilon_F$ and fitness weights.
5. **Re-check dimensionless ratios**: Keep $\sigma_{\text{sep}}$, $\eta_{\text{time}}$, and
   $\kappa$ in the regime required by {prf:ref}`thm-dimensionless-ratios`.
6. **Iterate with sweeps**: Sweep one primary knob at a time and validate the effect on mass
   plateaus. This avoids conflating coupling changes with measurement artifacts.

This workflow stays within the Volume 3 theoretical constraints and provides a direct path from
theory to implementation.
