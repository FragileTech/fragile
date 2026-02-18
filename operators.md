# Operator Variants: Theoretical Grounding Report

This report catalogs every operator variant implemented in the companion correlator
channels, explains its mathematical definition and physical interpretation, and ranks
each variant by how directly it derives from the Fractal Gas → QFT theory developed
in `docs/source/3_fractal_gas/2_fractal_set/`.

## Theoretical Framework (Summary)

The Fractal Gas theory derives the Standard Model gauge group
$SU(3)_C \times SU(2)_L \times U(1)_Y$ from three independent redundancies in the
optimization algorithm:

| Symmetry | Origin | Physical Force |
|----------|--------|----------------|
| $U(1)_Y$ | Absolute fitness baseline is unphysical | Hypercharge / EM |
| $SU(2)_L$ | Cloning creates a doublet (cloner/target); local convention ambiguity | Weak isospin |
| $SU(3)_C$ | Viscous force has $d=3$ components; local basis rotation redundancy | Strong / color |

**Color states** are the fundamental building blocks for strong-sector operators.
Each walker $i$ at time $t$ carries a 3-component complex color vector:

$$
\tilde{c}_i^{(\alpha)} = F_\alpha^{(\text{visc})}(i)\,
\exp\!\left(i\,p_i^{(\alpha)}\ell_0/\hbar_{\text{eff}}\right),
\quad
c_i^{(\alpha)} = \frac{\tilde{c}_i^{(\alpha)}}{\|\tilde{c}_i\|}
$$

where $F^{(\text{visc})}$ is the viscous force (coupling walkers to neighbors) and
the exponential encodes the momentum phase. This construction follows directly from
the $SU(3)$ emergence theorem.

**Companion pairs** $(i, j)$ and **triplets** $(i, j, k)$ are formed from the
Fractal Set interaction graph (IG): walker $j$ is the nearest companion by distance,
$k$ by clone lineage. These correspond to IG edges — the lattice links on which
gauge fields live.

**The color inner product** $z_{ij} = c_i^\dagger \cdot c_j$ is the fundamental
bilinear from which meson (quark-antiquark) operators are constructed:
- $\text{Re}(z_{ij})$ = scalar channel ($J^{PC} = 0^{++}$, analogue of $\bar\psi\psi$)
- $\text{Im}(z_{ij})$ = pseudoscalar channel ($J^{PC} = 0^{-+}$, analogue of $\bar\psi\gamma_5\psi$)

**The color determinant** $\det(c_i, c_j, c_k)$ is the $SU(3)$-invariant trilinear
that creates a color-singlet from three quarks — the baryon operator.

**The plaquette** $\Pi_i = (c_i^\dagger c_j)(c_j^\dagger c_k)(c_k^\dagger c_i)$ is
the closed color loop (Wilson loop trace), whose real part is the gauge-invariant
glueball operator.

### Ranking Tiers

| Tier | Meaning | Criteria |
|------|---------|----------|
| **Tier 1** | Canonical | Directly derived from theory: color bilinears, determinants, plaquettes, Wilson-loop structure |
| **Tier 2** | Well-motivated | Uses theoretically meaningful quantities (cloning scores as fitness gradients) in a physically motivated way |
| **Tier 3** | Exploratory | Heuristic combinations; useful for signal enhancement but not directly predicted by theory |

---

## 1. Scalar Channel ($\sigma / f_0$, $J^{PC} = 0^{++}$)

The scalar meson operator is $O_\sigma \propto \bar\psi\psi \propto \text{Re}(c_i^\dagger c_j)$.

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 1 | `standard` | $\text{Re}(c_i^\dagger c_j)$ | **Tier 1** | Direct color bilinear; the canonical scalar meson interpolating operator |
| 2 | `score_weighted` | $\text{Re}(z_{ij}) \cdot |\Delta S_{ij}|$ | **Tier 2** | Weights by fitness gradient magnitude; enhances pairs with large field-strength differences |
| 3 | `score_directed` | $\text{Re}(z_{ij})$ or $\text{Re}(z_{ij}^*)$ depending on $\text{sign}(\Delta S)$ | **Tier 2** | Orients phase along score gradient; physically motivated as an oriented transport |
| 4 | `abs2_vacsub` | $|c_i^\dagger c_j|^2$ | **Tier 3** | Squared modulus removes phase information; collapses scalar/pseudoscalar distinction; useful as vacuum-subtraction diagnostic |

### Details

**`standard`** — This is the theoretically predicted operator. The real part of the
color inner product between companion walkers corresponds to the scalar quark bilinear
$\bar\psi\psi$ in lattice QCD. The companion pair plays the role of the lattice
quark-antiquark source/sink. No additional quantities beyond the color states are needed.

**`score_weighted`** — The cloning score difference $\Delta S_{ij} = S_j - S_i$ is the
discrete analogue of the fitness gradient between paired walkers. In the theory, fitness
differences drive the $U(1)$ gauge phase. Weighting by $|\Delta S|$ amplifies pairs where
the fitness landscape has steep gradients — regions of strong gauge-field activity. This
is analogous to measuring meson correlators in the presence of a background chromoelectric
field.

**`score_directed`** — Conjugates the inner product when the score gradient points
"downhill" between the pair, ensuring the phase is always oriented along the fitness
gradient. This breaks the left-right conjugation symmetry and probes oriented
transport along the gauge field. Motivated but not directly predicted.

**`abs2_vacsub`** — Takes $|z_{ij}|^2$, losing all phase information. This is a
positive-definite overlap intensity measure. It cannot distinguish scalar from pseudoscalar
and does not correspond to a well-defined quantum number channel. Useful mainly as a
baseline or vacuum-subtraction diagnostic.

---

## 2. Pseudoscalar Channel ($\pi$, $J^{PC} = 0^{-+}$)

The pseudoscalar meson operator is $O_\pi \propto \bar\psi\gamma_5\psi \propto \text{Im}(c_i^\dagger c_j)$.

### Meson-based modes

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 1 | `standard` | $\text{Im}(c_i^\dagger c_j)$ | **Tier 1** | Direct color bilinear with $\gamma_5$ projection; the canonical pion operator |
| 2 | `score_weighted` | $\text{Im}(z_{ij}) \cdot |\Delta S_{ij}|$ | **Tier 2** | Score-gradient weighting; same motivation as scalar score_weighted |

### Fitness-based modes

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 3 | `fitness_pseudoscalar` (C_PP) | $P(t) = \langle \sigma_i \cdot \delta_i \rangle_{\text{alive}}$ | **Tier 2** | Parity-weighted log-fitness fluctuation; $\sigma = \pm 1$ from score sign gives pseudoscalar (odd-parity) character |
| 4 | `fitness_axial` (C_JP) | $J(t) = P(t+1) - P(t)$; cross-corr $\langle J \cdot P \rangle$ | **Tier 2** | Discrete axial current; enables PCAC mass consistency check — a direct QCD diagnostic |
| 5 | `fitness_scalar_variance` (C_SS) | $\Sigma(t) = \langle \delta_i^2 \rangle_{\text{alive}}$ | **Tier 3** | Scalar variance of log-fitness; probes radial ($\sigma/f_0$) mode in fitness landscape; no parity quantum number |

### Details

**`standard`** — The imaginary part of the color inner product. Because the color state
includes the momentum-phase factor $\exp(i p \ell_0/\hbar_{\text{eff}})$, the imaginary
part is sensitive to phase dispersion — it is the discrete analogue of the $\gamma_5$
insertion that creates pion quantum numbers. This is the most theoretically grounded
pseudoscalar operator.

**`score_weighted`** — Same gradient-enhancement as in the scalar channel. The imaginary
part preserves the pseudoscalar quantum numbers while the $|\Delta S|$ weighting enhances
signal from regions of strong gauge activity.

**`fitness_pseudoscalar` (C_PP)** — An entirely different construction: uses log-fitness
fluctuations rather than color states. The parity label $\sigma_i = +1$ (dominant,
$S_i \le 0$) or $-1$ (subordinate, $S_i > 0$) from the cloning score sign provides
the odd-parity character. The operator $P(t) = \langle \sigma \cdot \delta \rangle$
measures whether dominant walkers have systematically different fitness fluctuations.
Well-motivated because cloning scores are the $U(1)$ gauge field source, but the
construction does not use color states directly.

**`fitness_axial` (C_JP)** — The discrete time derivative of $P(t)$ acts as an axial
current. The cross-correlator $\langle J(t) P(t+\tau) \rangle$ enables computing the
PCAC mass $m_{\text{PCAC}} = [C_{JP}(\tau) - C_{JP}(\tau-1)] / [2 C_{PP}(\tau)]$.
This is a standard QCD consistency check: if the system has a pseudo-Goldstone pion,
$m_{\text{PCAC}}$ should be small. Valuable as a diagnostic even though the operator
construction is non-standard.

**`fitness_scalar_variance` (C_SS)** — Autocorrelation of the per-frame fitness variance.
This is a scalar (even-parity) quantity despite being grouped with pseudoscalar modes.
It probes the radial excitation ($\sigma/f_0$) in the fitness landscape. Exploratory —
it does not arise from the color bilinear construction.

---

## 3. Vector Channel ($\rho$, $J^{PC} = 1^{--}$)

The vector meson operator is $O_\rho^\mu \propto \bar\psi\gamma^\mu\psi \propto \text{Re}(c_i^\dagger c_j) \cdot \Delta x_{ij}^\mu$.

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 1 | `standard` | $\text{Re}(z_{ij}) \cdot \Delta x_{ij}$ | **Tier 1** | Color bilinear weighted by displacement; canonical $J=1$ vector meson interpolator |
| 2 | `score_directed` | Phase-oriented $z_{ij} \cdot \Delta x_{ij}$ | **Tier 2** | Orients color phase along fitness gradient before forming vector operator |
| 3 | `score_gradient` | $\text{Re}(z_{ij}) \cdot \nabla_i S$ | **Tier 3** | Replaces displacement with local score gradient estimate; exploratory |

### Projection modes (apply to all vector operator modes)

| Rank | Projection | Description | Tier |
|------|-----------|-------------|------|
| 1 | `full` | Full 3-vector displacement | **Tier 1** |
| 2 | `longitudinal` | $(\Delta x \cdot \hat{n}) \hat{n}$ along score gradient | **Tier 2** |
| 3 | `transverse` | $\Delta x - (\Delta x \cdot \hat{n}) \hat{n}$ perpendicular to gradient | **Tier 2** |

### Details

**`standard` + `full`** — The canonical vector meson operator. The color inner product
provides the spin-0 color factor, and the spatial displacement $\Delta x_{ij}$ between
companions provides the $J=1$ (vector) character. This corresponds to the lattice QCD
operator $\bar\psi\gamma_\mu\psi$ where the Dirac $\gamma_\mu$ matrix is represented by
the spatial direction of the companion displacement. The `full` projection uses the raw
displacement vector without projecting onto any preferred axis.

**`score_directed`** — Orients the color phase so it always points "uphill" in score
space before forming the vector operator. This is physically motivated as aligning the
gauge transport with the fitness gradient direction.

**`score_gradient`** — Replaces the companion displacement entirely with a local estimate
of the score gradient. This creates an operator aligned with the local fitness landscape
rather than the spatial geometry. Exploratory — the displacement between companions is the
more theoretically grounded spatial structure.

**Longitudinal/transverse projections** — Decompose the vector into components along
and perpendicular to the local score gradient. In QCD, longitudinal and transverse
gluon polarizations have different physics. Here the score gradient serves as the
preferred direction. Well-motivated for studying the polarization structure of the
vector channel.

---

## 4. Axial-Vector Channel ($a_1$, $J^{PC} = 1^{++}$)

The axial-vector channel uses the same operator structure as the vector channel but
with $\text{Im}(z_{ij})$ instead of $\text{Re}(z_{ij})$:

$$O_{a_1}^\mu = \text{Im}(c_i^\dagger c_j) \cdot \Delta x_{ij}^\mu$$

This is automatically produced alongside the vector channel by the same
`compute_vector_meson_correlator_from_color_positions` function. All vector mode
variants and projection modes apply identically, with the same tier rankings.

| Rank | Mode | Formula | Tier |
|------|------|---------|------|
| 1 | `standard` + `full` | $\text{Im}(z_{ij}) \cdot \Delta x_{ij}$ | **Tier 1** |
| 2 | `score_directed` | Phase-oriented $\text{Im}(z) \cdot \Delta x$ | **Tier 2** |
| 3 | `score_gradient` | $\text{Im}(z_{ij}) \cdot \nabla_i S$ | **Tier 3** |

The axial-vector is the $\gamma_5 \gamma_\mu$ channel — the parity partner of the
vector. In QCD, the mass splitting $m_{a_1} - m_\rho$ is a direct measure of chiral
symmetry breaking. The `standard` mode is canonical.

---

## 5. Nucleon / Baryon Channel ($p/n$, $J^P = 1/2^+$)

The baryon operator uses the $SU(3)$-invariant determinant of three color vectors from
a companion triplet $(i, j, k)$:

$$B(t) = \det_3(c_i, c_j, c_k) = \epsilon_{abc}\, c_i^a c_j^b c_k^c$$

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 1 | `det_abs` | $\det_3(c_i, c_j, c_k)$ | **Tier 1** | The canonical color-singlet baryon operator; direct $SU(3)$ invariant |
| 2 | `flux_action` | $|\det_3| \cdot (1 - \cos\arg\Pi)$ | **Tier 1** | Baryon modulated by Wilson plaquette action; probes baryon-flux coupling |
| 3 | `flux_sin2` | $|\det_3| \cdot \sin^2(\arg\Pi)$ | **Tier 2** | Alternative flux weighting; sensitive to different flux angles |
| 4 | `flux_exp` | $|\det_3| \cdot \exp(\alpha(1-\cos\arg\Pi))$ | **Tier 2** | Boltzmann-weighted Wilson action; lattice-QCD-style reweighting |
| 5 | `score_signed` | $\text{Re}(\det_3(c_{i'}, c_{j'}, c_{k'}))$ (score-ordered) | **Tier 2** | Score-ordering removes permutation ambiguity; sign encodes chirality |
| 6 | `score_abs` | $|\text{Re}(\det_3)|$ (score-ordered) | **Tier 3** | Unsigned variant; loses chirality information |

### Details

**`det_abs`** — The 3×3 determinant of the color triplet is the unique $SU(3)$-invariant
antisymmetric trilinear — the epsilon tensor contraction $\epsilon_{abc} c_i^a c_j^b c_k^c$.
This is the lattice QCD baryon interpolating operator. Non-zero determinant means the
triplet spans the full color space (genuine color-singlet baryon state). This is the most
theoretically canonical baryon operator.

**`flux_action`** — Multiplies the baryon operator by the Wilson plaquette action
$(1 - \cos\arg\Pi)$ computed from the same triplet. The plaquette phase
$\arg\Pi = \arg[(c_i^\dagger c_j)(c_j^\dagger c_k)(c_k^\dagger c_i)]$ is the gauge
flux through the triangle. This probes baryons in the presence of non-trivial gauge
flux — diagnostically important for confinement physics. Ranked Tier 1 because both
the determinant and plaquette are first-principles constructions from the theory.

**`flux_sin2`** — Uses $\sin^2(\arg\Pi)$ instead of $(1-\cos\arg\Pi)$. This has
periodicity $\pi$ rather than $2\pi$, giving different sensitivity to the flux angle.
Vanishes at both $0$ and $\pi$ flux, peaking at $\pm\pi/2$. A reasonable variant but
the $(1-\cos)$ form is closer to the standard Wilson action.

**`flux_exp`** — Exponential of the Wilson action: $\exp(\alpha(1-\cos\arg\Pi))$.
This is the Boltzmann weight used in lattice QCD Monte Carlo sampling
($e^{-\beta S_\text{Wilson}}$). The parameter $\alpha$ controls the sensitivity.
Well-motivated from lattice QCD methodology.

**`score_signed`** — Sorts the triplet by cloning score before computing the determinant,
then takes the real part. The score-ordering removes the $S_3$ permutation ambiguity,
making the sign of $\text{Re}(\det)$ physically meaningful (chirality-like). Interesting
but the ordering criterion (cloning score) is not part of the color structure per se.

**`score_abs`** — Absolute value of the score-ordered determinant. Loses the sign
(chirality) information. Mainly useful as a magnitude diagnostic.

---

## 6. Glueball Channel ($0^{++}$ glueball)

The glueball operator is constructed from the color plaquette — the Wilson loop trace
on the companion triplet:

$$\Pi_i = (c_i^\dagger c_j)(c_j^\dagger c_k)(c_k^\dagger c_i)$$

| Rank | Mode | Formula | Tier | Justification |
|------|------|---------|------|---------------|
| 1 | `action_re_plaquette` | $1 - \text{Re}(\Pi_i)$ | **Tier 1** | Standard Wilson plaquette action density; the canonical glueball source |
| 2 | `re_plaquette` | $\text{Re}(\Pi_i)$ | **Tier 1** | Plaquette trace; equivalent to action form up to a constant shift |
| 3 | `phase_action` | $1 - \cos(\arg\Pi_i)$ | **Tier 2** | Phase-only reduction of Wilson action; probes topological flux |
| 4 | `phase_sin2` | $\sin^2(\arg\Pi_i)$ | **Tier 2** | Alternative flux weighting; $\pi$-periodic, sensitive to $\mathbb{Z}_2$ flux |

### Details

**`action_re_plaquette`** — The standard Wilson plaquette action density
$S_\square = \beta(1 - \text{Re}\,\text{Tr}\, U_\square / N)$. In the Fractal Gas theory,
this emerges directly from the path integral over walker trajectories. The plaquette
$\Pi_i$ is the holonomy around the closed loop $i \to j \to k \to i$, and $1 - \text{Re}(\Pi)$
vanishes for trivial gauge configurations and is maximal for strong field configurations.
The correlator of action densities is the standard glueball interpolating operator in
lattice QCD.

**`re_plaquette`** — The raw plaquette trace $\text{Re}(\Pi)$. This differs from
`action_re_plaquette` only by a constant shift ($1 - \text{Re}(\Pi)$ vs $\text{Re}(\Pi)$),
which affects only the disconnected part. For connected correlators, they are equivalent.
Ranked equally at Tier 1.

**`phase_action`** — Extracts only the phase $\theta = \arg(\Pi)$ of the plaquette
and uses $1 - \cos\theta$ as the observable. This is the $U(1)$ reduction of the Wilson
action — it ignores the plaquette magnitude and probes only the gauge flux angle.
Useful for studying topological flux excitations, but the full plaquette (including
magnitude) is the more complete observable.

**`phase_sin2`** — Uses $\sin^2(\arg\Pi)$. This is $\pi$-periodic (vs $2\pi$ for
$1-\cos$), giving a different sensitivity pattern. Vanishes at both trivial ($0$) and
maximally non-trivial ($\pi$) flux. Potentially useful for detecting
$\mathbb{Z}_2$-flux-like excitations (visons), but more exploratory.

### Momentum projection

All glueball modes support optional Fourier momentum projection:

$$O_{\cos}(p, t) = \frac{1}{N} \sum_i O_i(t) \cos(k_p x_i), \quad k_p = \frac{2\pi p}{L}$$

Zero-momentum ($p=0$) isolates the rest-mass glueball; finite-$p$ modes enable
dispersion relation measurements $E(p)^2 = m^2 + p^2$.

---

## 7. Tensor Channel ($a_2$, $J^{PC} = 2^{++}$)

The tensor meson operator uses a spin-2 traceless symmetric tensor constructed from
the companion displacement:

$$O_i^{\alpha\beta}(t) = \text{Re}(c_i^\dagger c_j) \cdot Q^{\alpha\beta}(\Delta x_{ij})$$

where $Q^{\alpha\beta}$ is the traceless symmetric part of $\Delta x^\alpha \Delta x^\beta$
(5 independent components in $d=3$, corresponding to $l=2$ spherical harmonics).

| Feature | Description | Tier |
|---------|-------------|------|
| Operator | $\text{Re}(z_{ij}) \cdot Q^{\alpha\beta}(\Delta x_{ij})$ | **Tier 1** |
| Momentum projection | Fourier modes $k_n = 2\pi n / L$ along chosen axis | **Tier 1** |

**No operator mode variants** — the tensor channel has a single canonical construction.

### Details

The tensor operator is the natural $J=2$ extension of the vector ($J=1$) construction.
The color inner product provides the spin-0 color factor while the traceless symmetric
tensor $Q^{\alpha\beta}$ of the displacement provides the $J=2$ spatial structure.
This corresponds to the lattice QCD operator for tensor mesons ($a_2$, $f_2$) or
graviton-like excitations.

The 5-component traceless basis in Cartesian coordinates:

| Component | Formula |
|-----------|---------|
| $q_{xy}$ | $x \cdot y$ |
| $q_{xz}$ | $x \cdot z$ |
| $q_{yz}$ | $y \cdot z$ |
| $q_{x^2-y^2}$ | $(x^2 - y^2)/\sqrt{2}$ |
| $q_{2z^2-x^2-y^2}$ | $(2z^2 - x^2 - y^2)/\sqrt{6}$ |

The contracted correlator sums over all 5 components:
$C_n(\Delta t) = \sum_\alpha C_{n,\alpha}(\Delta t)$.

Momentum projection follows the same Fourier approach as the glueball channel, with
bootstrap error support.

---

## Summary: Complete Ranking by Channel

### Tier 1 — Canonical (directly from theory)

| Channel | Mode | Operator |
|---------|------|----------|
| Scalar | `standard` | $\text{Re}(c_i^\dagger c_j)$ |
| Pseudoscalar | `standard` | $\text{Im}(c_i^\dagger c_j)$ |
| Vector | `standard` + `full` | $\text{Re}(z_{ij}) \cdot \Delta x$ |
| Axial-vector | `standard` + `full` | $\text{Im}(z_{ij}) \cdot \Delta x$ |
| Nucleon | `det_abs` | $\det_3(c_i, c_j, c_k)$ |
| Nucleon | `flux_action` | $|\det_3| \cdot (1-\cos\arg\Pi)$ |
| Glueball | `action_re_plaquette` | $1 - \text{Re}(\Pi)$ |
| Glueball | `re_plaquette` | $\text{Re}(\Pi)$ |
| Tensor | (single mode) | $\text{Re}(z_{ij}) \cdot Q^{\alpha\beta}(\Delta x)$ |

### Tier 2 — Well-motivated

| Channel | Mode | Rationale |
|---------|------|-----------|
| Scalar | `score_weighted` | Fitness gradient weighting enhances strong-field regions |
| Scalar | `score_directed` | Phase orientation along gauge transport direction |
| Pseudoscalar | `score_weighted` | Same gradient enhancement, preserves quantum numbers |
| Pseudoscalar | `fitness_pseudoscalar` | Parity from score sign; probes $U(1)$ sector |
| Pseudoscalar | `fitness_axial` | PCAC consistency check; standard QCD diagnostic |
| Vector | `score_directed` | Phase alignment along fitness gradient |
| Vector | `longitudinal` / `transverse` | Polarization decomposition along gradient |
| Nucleon | `flux_sin2` | Alternative flux weighting with $\pi$-periodicity |
| Nucleon | `flux_exp` | Boltzmann-weighted Wilson action (lattice QCD style) |
| Nucleon | `score_signed` | Score-ordering gives canonical chirality sign |
| Glueball | `phase_action` | Topological flux from plaquette phase |
| Glueball | `phase_sin2` | $\mathbb{Z}_2$-flux sensitivity |

### Tier 3 — Exploratory

| Channel | Mode | Limitation |
|---------|------|-----------|
| Scalar | `abs2_vacsub` | Loses phase/quantum-number information |
| Pseudoscalar | `fitness_scalar_variance` | Even-parity; not a true pseudoscalar |
| Vector | `score_gradient` | Replaces spatial displacement with score gradient |
| Nucleon | `score_abs` | Loses chirality sign information |

---

## Recommendations

1. **For production mass extraction**, use Tier 1 operators as the primary channels.
   The `standard` modes for scalar, pseudoscalar, vector, and axial-vector, `det_abs`
   for baryon, `action_re_plaquette` for glueball, and the single tensor mode.

2. **For systematic error estimation**, include Tier 2 variants (especially
   `score_weighted`) as alternative operators fitting to the same mass. Consistency
   across variants is a strong validation signal.

3. **For PCAC diagnostics**, use `fitness_pseudoscalar` + `fitness_axial` to compute
   the PCAC mass and verify chiral symmetry properties.

4. **For confinement studies**, use the baryon `flux_action` mode and glueball
   momentum-projected correlators to measure the string tension and glueball
   dispersion relation.
