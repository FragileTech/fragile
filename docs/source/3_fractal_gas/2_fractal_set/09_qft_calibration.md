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

Implementation note: the time correlators, effective masses, and exponential fits are computed in
`src/fragile/fractalai/qft/particle_observables.py` and
`src/fragile/fractalai/qft/correlator_channels.py`.

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

The Channels tab in `src/fragile/fractalai/qft/dashboard.py` reports **Extracted Masses** computed
from Euclidean two-point correlators in `src/fragile/fractalai/qft/correlator_channels.py`. For any
channel operator $O_\chi$,

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

Implementation note: the Channels tab also exposes analysis knobs (`h_eff`, `mass`, `ell0`,
`knn_k`, `knn_sample`) that belong to the **measurement** map, not the underlying dynamics. The
first three appear directly in the color-state definition ({prf:ref}`thm-sm-su3-emergence`), so set
them consistently with the run; the k-NN parameters approximate the companion locality
({prf:ref}`def-fractal-set-companion-kernel`). Changing these can move extracted masses without
changing the simulation itself.

The table below makes the knob-to-parameter correspondence explicit.

| Knob (symbol) | Algorithm parameter name | Location (code) | Role in calibration |
| --- | --- | --- | --- |
| $\nu$ | `nu` | `KineticOperator` (`src/fragile/fractalai/core/kinetic_operator.py`) | Viscous coupling strength (color/gauge sector) |
| $\rho$ | `viscous_length_scale` | `KineticOperator` | Localization range of viscous kernel |
| $\gamma$ | `gamma` | `KineticOperator` | Friction mass scale ($m_{\text{friction}}$) |
| $\beta$ | `beta` | `KineticOperator` | Inverse temperature (noise scale) |
| $\Delta t$ | `delta_t` | `KineticOperator` | Time step; rescales all masses |
| $\epsilon_F$ | `epsilon_F` | `KineticOperator` | Fitness/U(1) coupling scale |
| $\epsilon_c$ | `companion_epsilon_clone` | `OperatorConfig` (`src/fragile/fractalai/qft/simulation.py`) | Clone companion temperature |
| $\epsilon_d$ | `companion_epsilon` | `OperatorConfig` | Diversity companion temperature |
| $\lambda_{\text{alg}}$ | `lambda_alg` | `OperatorConfig` | Algorithmic distance weight |
| $\epsilon_{\text{clone}}$ | `epsilon_clone` | `CloneOperator` (`src/fragile/fractalai/core/cloning.py`) | Cloning score regularization |
| $p_{\max}$ | `p_max` | `CloneOperator` | Max cloning probability |
| Fitness weights $(\alpha,\beta,\eta,A,\rho)$ | `fitness_alpha`, `fitness_beta`, `fitness_eta`, `fitness_A`, `fitness_rho` | `FitnessOperator` (`src/fragile/fractalai/core/fitness.py`) | Fitness coupling shape |
| $\hbar_{\text{eff}}$ | `h_eff` | `ChannelSettings` (`src/fragile/fractalai/qft/dashboard.py`) | Measurement: phase scale for color state |
| $m$ (phase mass) | `mass` | `ChannelSettings` | Measurement: momentum-phase factor |
| $\ell_0$ | `ell0` | `ChannelSettings` | Measurement: phase length scale |
| k-NN $k$ | `knn_k` | `ChannelSettings` | Measurement: locality proxy for channels |
| k-NN sample | `knn_sample` | `ChannelSettings` | Measurement: sample size per timestep |
| Connected correlator | `use_connected` | `ChannelSettings` | Measurement: connected vs raw $C(t)$ |
| Max lag | `max_lag` | `ChannelSettings` | Measurement: correlator window length |
| Warmup fraction | `warmup_fraction` | `ChannelSettings` | Measurement: drop transient steps |
| Fit window widths | `window_widths_spec` | `ChannelSettings` | Measurement: AIC fit windows |

Below, each channel is tied to its operator, the Fractal Set ingredients that define it, and the
parameters that control its correlator decay.

### Scalar channel (σ, $0^{++}$)

| Parameter | Symbol | Code parameter | Effect on observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger color coupling → shorter correlator → heavier $m_\sigma$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter localization → heavier $m_\sigma$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster velocity relaxation → heavier $m_\sigma$ (keep hierarchy). |
| Phase scale | $\hbar_{\text{eff}}$ | `ChannelSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → weaker phase winding → slightly lighter $m_\sigma$. |
| Phase mass | $m$ | `ChannelSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\sigma$. |
| Phase length | $\ell_0$ | `ChannelSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\sigma$. |

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
| Phase scale | $\hbar_{\text{eff}}$ | `ChannelSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase dispersion → lighter $m_\pi$. |
| Phase mass | $m$ | `ChannelSettings.mass` | Increase $m$ → more phase winding → heavier $m_\pi$. |
| Phase length | $\ell_0$ | `ChannelSettings.ell0` | Increase $\ell_0$ → more phase winding → heavier $m_\pi$. |
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
| Phase scale | $\hbar_{\text{eff}}$ | `ChannelSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase winding → slightly lighter $m_\rho$. |
| Phase mass | $m$ | `ChannelSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\rho$. |
| Phase length | $\ell_0$ | `ChannelSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\rho$. |

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
| Clone temperature | $\epsilon_c$ | `OperatorConfig.companion_epsilon_clone` | Decrease $\epsilon_c$ → stronger clone coupling → heavier $m_N$. |
| Diversity temperature | $\epsilon_d$ | `OperatorConfig.companion_epsilon` | Decrease $\epsilon_d$ → tighter locality → heavier $m_N$. |
| Alg. distance weight | $\lambda_{\text{alg}}$ | `OperatorConfig.lambda_alg` | Increase $\lambda_{\text{alg}}$ → velocity-weighted locality → can raise $m_N$. |
| k-NN neighbors | $k$ | `ChannelSettings.knn_k` | Increase $k$ → broader triplet sampling → typically stabilizes/lower $m_N$. |
| k-NN sample size | — | `ChannelSettings.knn_sample` | Increase sample → less noise in $m_N$; minimal systematic shift. |

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
  `CompanionSelection` in `src/fragile/fractalai/core/companion_selection.py`.
  - Distance companion temperature $\epsilon_d$ -> `OperatorConfig.companion_epsilon` in
    `src/fragile/fractalai/qft/simulation.py`.
  - Clone companion temperature $\epsilon_c$ -> `OperatorConfig.companion_epsilon_clone` in
    `src/fragile/fractalai/qft/simulation.py`.
  - Algorithmic distance weight $\lambda_{\text{alg}}$ -> `OperatorConfig.lambda_alg`.
- Two-channel fitness ({prf:ref}`def-fractal-set-two-channel-fitness`): `FitnessOperator`
  parameters `alpha`, `beta`, `eta`, `A`, and `rho`.
- Cloning score ({prf:ref}`def-fractal-set-cloning-score`): `CloneOperator` parameters
  `epsilon_clone`, `p_max`, `sigma_x`, `alpha_restitution`.
- Viscous force and color coupling ({prf:ref}`def-fractal-set-viscous-force`,
  {prf:ref}`thm-sm-su3-emergence`): `KineticOperator` parameters `nu` and
  `viscous_length_scale` (this is $\rho$).
- Anisotropic diffusion ({prf:ref}`def-fractal-set-anisotropic-diffusion`):
  `KineticOperator.use_anisotropic_diffusion` and `epsilon_Sigma`.

The particle and channel correlators are computed in:
- `src/fragile/fractalai/qft/particle_observables.py` (baryon/meson/glueball operators and fits).
- `src/fragile/fractalai/qft/correlator_channels.py` (scalar, pseudoscalar, vector, nucleon,
  glueball, etc.).

Analysis window choices (fit start/stop, plateau detection) live in
`src/fragile/fractalai/qft/analysis.py` and change measurement quality, not the underlying physics.

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
