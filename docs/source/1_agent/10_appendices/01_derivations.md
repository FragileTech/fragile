(sec-appendix-a-full-derivations)=
# {ref}`Appendix A <sec-appendix-a-full-derivations>`: Full Derivations (Capacity-Constrained Curvature Functional)

## TLDR

- This appendix records the **full derivations** behind the main geometric/information-theoretic claims in Volume 1.
- It is reference material: use it when you want the complete steps, not first-pass intuition.
- Most readers can skim the statements and return to details when implementing or auditing proofs.

(sec-appendix-a-capacity-constrained-curvature-functional)=
## A.1 Capacity-Constrained Curvature Functional (Variational Principle)

The capacity diagnostic is the operational postulate
$I_{\text{bulk}}\le C_{\partial}$. A measured violation is rejected by the
Sieve ({ref}`Section 3 <sec-diagnostics-stability-checks>`, Node 13). The
curvature variation below is a separate identity for a curvature--risk
functional. It does not impose the equality $I_{\text{bulk}}=C_{\partial}$;
an active constraint would require an explicit multiplier term and a
$G$-differentiable information model.

:::{prf:definition} A.1.1 (Boundary capacity form)
:label: def-a-boundary-capacity-form

Define the boundary capacity $(n\!-\!1)$-form

For a declared cutoff hypersurface $\partial_{\varepsilon}\mathcal Z$, use

$$
\omega_{\partial} := \frac{1}{\eta_\ell}\, dA_G,

$$
so that $C_{\partial}(\partial_{\varepsilon}\mathcal Z)= \oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial}$ (Definition
{prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`). If no
geometric cutoff is part of the model, replace this area expression by the
interface-channel capacity.

:::
:::{prf:definition} A.1.2 (Boundary-capacity constraint functional)
:label: def-a-boundary-capacity-constraint-functional

Define the diagnostic difference (not a variational constraint in the action)

$$
\mathcal{C}[G,V]
:=
\underbrace{\int_{\mathcal{Z}} \iota_{\mathrm{bulk}}\, d\mu_G}_{I_{\text{bulk}}}
\;-\;
\underbrace{\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial}}_{C_{\partial}},

$$
where $\iota_{\mathrm{bulk}}$ is the relative-information density in
{prf:ref}`def-information-density-and-bulk-information-volume`. The realised
shutter inflow is the separate rate
$\lambda_{\mathrm{in}}=\mathbb{E}[I(X;K)]$ (Definition
{prf:ref}`def-grounding-rate`). The coupling-window definition
{prf:ref}`thm-information-stability-window-operational` supplies its
admissible operating range.

:::
:::{prf:definition} A.1.3 (Risk Lagrangian density)
:label: def-a-risk-lagrangian-density

Fix a smooth potential $V\in C^\infty(\mathcal{Z})$. A canonical risk Lagrangian density is the scalar-field functional

$$
\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V),

$$
where $U:\mathbb{R}\to\mathbb{R}$ is a (possibly learned) on-site potential capturing non-gradient costs. (The sign convention is chosen for a Riemannian metric; see e.g. Lee, *Riemannian Manifolds*, 2018, for the variational identities used below.)

:::
:::{prf:definition} A.1.4 (Curvature--risk functional with a cutoff penalty)
:label: def-a-capacity-constrained-curvature-functional

Let $R(G)$ be the scalar curvature of $G$ and let $\Lambda\in\mathbb{R}$ be a constant. Define the functional

$$
\mathcal{S}[G,V]
:=
\int_{\mathcal{Z}}\left(R(G)-2\Lambda - 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G
\;-\;
2\kappa\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial},

$$
with coupling $\kappa\in\mathbb{R}$. The sign convention makes the
positive Riemannian risk tensor below appear on the right-hand side of the
stationarity equation. The cutoff term has no interior variation under
clamping, and $\Lambda$ remains a free curvature offset rather than a
multiplier determined by $C_{\partial}$.

*Remark (why $\Lambda$ is allowed).* A constant term in the integrand is the simplest coordinate-invariant scalar density and produces a $\Lambda G_{ij}$ term in the metric Euler–Lagrange equation. Here $\Lambda$ plays the role of a baseline curvature / capacity offset.

:::
(sec-appendix-a-first-variation)=
## A.2 First Variation (Expanded Derivation)

We work in the standard calculus of variations on Riemannian manifolds with boundary. Assume:
1) $G$ is $C^2$ and $V$ is $C^2$ (so curvature and gradients are well-defined),
2) variations $\delta G^{ij}$ are smooth, symmetric, and compactly supported in $\mathcal{Z}$ or satisfy Dirichlet boundary conditions $\delta G^{ij}\vert_{\partial\mathcal{Z}}=0$ (the boundary is clamped by the sensorium; cf. Definition {prf:ref}`def-observation-inflow-form` / Theorem {prf:ref}`thm-generalized-conservation-of-belief`).

Under these hypotheses, the first variation of $\mathcal{S}$ is well-defined as a distribution; the standard identities below can be found in standard differential-geometry references (e.g. {cite}`lee2018riemannian`).

(sec-appendix-a-variation-of-the-volume-form)=
### A.2.1 Variation of the volume form

Let $d\mu_G=\sqrt{|G|}\,dz^n$. The determinant identity gives

$$
\delta \sqrt{|G|} = -\frac{1}{2}\sqrt{|G|}\,G_{ij}\,\delta G^{ij},

$$
equivalently $\delta d\mu_G = -\tfrac12\,G_{ij}\,\delta G^{ij}\, d\mu_G$.

(sec-appendix-a-variation-of-the-curvature-term)=
### A.2.2 Variation of the curvature term

Write the curvature functional as $\mathcal{S}_{\text{geo}}[G]:=\int_{\mathcal{Z}}R(G)\,d\mu_G$. The variation splits as

$$
\delta(R\,d\mu_G) = (\delta R)\,d\mu_G + R\,\delta d\mu_G.

$$
For the scalar curvature, use

$$
R = G^{ij}R_{ij},

$$
hence

$$
\delta R = R_{ij}\,\delta G^{ij} + G^{ij}\,\delta R_{ij}.

$$
The Palatini identity gives

$$
\delta R_{ij} = \nabla_k(\delta \Gamma^k_{ij})-\nabla_j(\delta\Gamma^k_{ik}),

$$
and the Christoffel variation is

$$
\delta\Gamma^k_{ij} = \frac12\,G^{k\ell}\left(\nabla_i \delta G_{j\ell}+\nabla_j \delta G_{i\ell}-\nabla_\ell \delta G_{ij}\right),

$$
where $\delta G_{ij} = -G_{ia}G_{jb}\,\delta G^{ab}$.

Substituting and collecting terms yields the standard decomposition

$$
\delta\mathcal{S}_{\text{geo}} = \int_{\mathcal{Z}}\left(R_{ij}-\frac12 R\,G_{ij}\right)\delta G^{ij}\,d\mu_G + \oint_{\partial\mathcal{Z}} \mathcal{B}_{\text{curv}}(\delta G,\nabla\delta G),

$$
where $\mathcal{B}_{\text{curv}}$ is an explicit boundary $(n\!-\!1)$-form built from $\delta\Gamma$ (equivalently from $\delta G$ and its first derivatives). For a well-posed Dirichlet variational problem one can add an appropriate boundary term to cancel $\mathcal{B}_{\text{curv}}$. On the declared cutoff we impose $\delta G\vert_{\partial_{\varepsilon}\mathcal Z}=0$ and the boundary term vanishes.

(sec-appendix-a-variation-of-the-risk-term)=
### A.2.3 Variation of the risk term

Let $\mathcal{S}_{\text{risk}}[G,V] := \int_{\mathcal{Z}}\mathcal{L}_{\text{risk}}(V;G)\,d\mu_G$. Define the (Riemannian-signature) risk tensor by

$$
T_{ij} := \frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\text{risk}})}{\delta G^{ij}}.

$$
Holding $V$ fixed under $\delta G$ and using $\delta d\mu_G = -\tfrac12 G_{ij}\delta G^{ij} d\mu_G$ gives the standard identity

$$
\delta \mathcal{S}_{\text{risk}} = \frac12 \int_{\mathcal{Z}} T_{ij}\,\delta G^{ij}\,d\mu_G.

$$
For the risk Lagrangian

$$
\mathcal{L}_{\text{risk}}=\tfrac12 G^{ab}\nabla_a V\nabla_b V + U(V),

$$
the explicit computation yields

$$
T_{ij} = \nabla_i V\,\nabla_j V - G_{ij}\left(\frac12\,G^{ab}\nabla_a V\nabla_b V + U(V)\right).

$$
(sec-appendix-a-capacity-term-and-the-emergence-of)=
### A.2.4 Capacity (boundary) term and the emergence of $\Lambda$

The explicit boundary penalty $-2\kappa\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial}$ depends only on the induced cutoff metric through $dA_G$. Under the clamped boundary condition $\delta G\vert_{\partial_{\varepsilon}\mathcal Z}=0$, its first variation vanishes.

The remaining constant $\Lambda$ in Definition A.1.4 is a free bulk curvature offset. It is not fixed by the interface capacity unless an additional multiplier equation is supplied. Formally, varying $-2\Lambda\int_{\mathcal{Z}} d\mu_G$ gives

$$
\delta\left(-2\Lambda\int_{\mathcal{Z}} d\mu_G\right) = \int_{\mathcal{Z}} \Lambda G_{ij}\,\delta G^{ij}\,d\mu_G.

$$
(sec-appendix-a-recovery-of-the-metric-stationarity-condition)=
## A.3 Recovery of the Metric Stationarity Condition

:::{prf:lemma} A.3.1 (Divergence-to-boundary conversion)
:label: lem-a-divergence-to-boundary-conversion

For any sufficiently regular information flux field $\mathbf{j}$ on $\mathcal{Z}$,

$$
\int_{\mathcal{Z}} \operatorname{div}_G(\mathbf{j})\, d\mu_G = \oint_{\partial \mathcal{Z}} \langle \mathbf{j}, \mathbf{n}\rangle\, dA_G,

$$
which is the Riemannian divergence theorem underlying the global balance equation in Theorem {prf:ref}`thm-generalized-conservation-of-belief`.

:::
:::{prf:theorem} A.3.2 (Capacity-consistency identity; proof of Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
:label: thm-a-capacity-consistency-identity-proof-of-theorem

Under the hypotheses of Section A.2, stationarity of $\mathcal{S}[G,V]$ with respect to arbitrary variations $\delta G^{ij}$ that vanish on $\partial\mathcal{Z}$ implies the Euler–Lagrange equation

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
with $T_{ij}$ given by Section A.2.3.

*Proof.* Combine Sections A.2.1–A.2.4:

$$
\delta\mathcal{S} = \int_{\mathcal{Z}}\left[\left(R_{ij}-\frac12 R\,G_{ij}\right) + \Lambda G_{ij} - \kappa T_{ij}\right]\delta G^{ij}\,d\mu_G + \text{(boundary terms)}.

$$
Boundary terms vanish under the clamped boundary condition (or after adding an appropriate boundary term). Because $\delta G^{ij}$ is arbitrary in the interior, the fundamental lemma of the calculus of variations implies the bracketed tensor must vanish pointwise almost everywhere, yielding the stated identity (see e.g. Evans, *Partial Differential Equations*, 2010, for the functional-analytic lemma).

*Interpretation.* The Ricci curvature governs local volume growth. This
stationarity identity is a consistency equation for the declared risk model;
the separate capacity postulate tests whether the resulting belief is
grounded at the interface.

*Remark (regularizer).* The squared residual of this identity is the metric-law
loss $\mathcal{L}_{\mathrm{EFE}}$ in {ref}`Appendix F
<sec-appendix-f-loss-terms-reference>`.

:::
(sec-appendix-a-pitchfork-bifurcation-at-the-origin)=
## A.3b Radial Escape and Angular Selection at the Origin (Crossover calculation)

This section records the radial and angular calculation used by Definition {prf:ref}`def-control-field-at-origin`.
It establishes an outward radial drift and a finite-time direction-selection mechanism; it does not establish a
pitchfork bifurcation or a critical temperature.

**Setup.** Consider the Langevin equation on the Poincare disk $\mathbb{D}$ from Definition {prf:ref}`prop-so-d-symmetry-at-origin`:

$$
dz_\tau = -\nabla_G U(z_\tau)\, d\tau + \sqrt{2T_c}\, G^{-1/2}(z_\tau)\, dW_\tau,

$$
with $U(z) = -2\operatorname{artanh}(|z|)$ and initial condition $z(0) = 0$.

**Step 1: Radial expansion near the origin.**

For $r=|z|>0$, expand the radial potential near the origin:

$$
U(r) = -2\operatorname{artanh}(r) = -2r - \frac{2}{3}r^3 + O(r^5).

$$
The Euclidean gradient for $r>0$ is:

$$
\nabla U(z) = -\frac{2z}{r(1-r^2)}.

$$
The Riemannian gradient (with $G^{-1} = \frac{(1-r^2)^2}{4}I$) is:

$$
\nabla_G U(z) = -\frac{1-r^2}{2r}z.

$$
**Step 2: Generator and polar-coordinate caveat.**

Away from $r=0$, the generator of the stated covariant Langevin model is

$$
\mathcal{L}f=-\langle\nabla_G U,\nabla_G f\rangle_G+T_c\Delta_G f.
$$

In polar coordinates the Laplace--Beltrami operator supplies the radial (Bessel-type) correction, and the coordinates
are singular at $r=0$. A flat Cartesian linearization therefore cannot be used to infer a stationary law or a Landau
normal form at the origin.

**Step 3: Formal zero-flux density and its scope.**

If one formally imposes zero flux for the conservative diffusion, the density relative to Riemannian volume would be

$$
p_{\mathrm{formal}}(z)\propto\exp\!\left(-\frac{U(z)}{T_c}\right)
=\exp\!\left(\frac{2\operatorname{artanh}(r)}{T_c}\right).
$$

This is rotationally symmetric, but it is not normalizable on the full Poincare disk because it diverges as $r\uparrow1$.
Consequently the displayed model has no invariant Gibbs probability without an outer boundary, a confining modification,
or another integrability hypothesis.

The detailed-balance calculation is therefore only a formal local statement:

$$
\nabla_G U + T_c\,G^{-1}\,\nabla \log p_{\mathrm{formal}} = 0.

$$
Substituting and solving:

$$
\nabla \log p_{\mathrm{formal}} = -\frac{1}{T_c}\,G\,\nabla_G U = -\frac{1}{T_c}\,\nabla U.

$$
Integrating:

$$
p_{\mathrm{formal}}(z) \propto \exp\left(-\frac{U(z)}{T_c}\right) = \exp\left(\frac{2\operatorname{artanh}(r)}{T_c}\right).

$$
The formal density is **rotationally symmetric** (depends only on $r$), but its non-normalizability limits the claim to
local symmetry of the coefficients.

**Step 4: Radial and angular drift.**

Write $z = re^{i\theta}$ in polar coordinates. The effective potential in the radial direction is:

$$
U_{\text{eff}}(r) = -2\operatorname{artanh}(r).

$$
The radial force is $F_r = -\frac{dU_{\text{eff}}}{dr} = \frac{2}{1-r^2} > 0$ for all $r \in [0, 1)$.

This means:
- The origin $r = 0$ is an **unstable equilibrium** (force points outward).
- There is no stable equilibrium in the interior; the "stable point" is at the boundary $r = 1$.
- The angular direction $\theta$ is **neutral** (no restoring force).

**Step 5: Symmetry breaking mechanism.**

For small $\tau$:
1. The noise term dominates: $z(\tau) \approx \sqrt{2T_c}\int_0^\tau G^{-1/2} dW_\tau$ performs a random walk.
2. This random walk samples directions $\theta$ uniformly from $[0, 2\pi)$.
3. Once $|z|$ exceeds a threshold (order $\sqrt{T_c}$), the deterministic drift $-\nabla_G U$ takes over.
4. The trajectory then flows radially outward along the selected direction $\theta$.

This is a finite-time radial escape with angular selection by noise and, when present, the policy drift. It is not a
supercritical pitchfork: there is no pair of stable interior equilibria or cubic normal form in this SDE.

**Step 6: Exit-time and direction statements.**

For a declared stopping radius $R\in(0,1)$, define

$$
\tau_R:=\inf\{\tau\ge0:r(\tau)\ge R\}.

$$
Its expectation depends on $R$, the metric, the boundary condition, and the full radial drift/noise coefficients. No
Kramers barrier formula follows here because the origin is unstable and $U$ has no interior barrier.

If the policy field is zero and the noise and stopping rule are rotationally invariant, the exit angle satisfies
$\theta^*\sim\mathrm{Uniform}[0,2\pi)$. A nonzero tangential policy or anisotropic boundary changes this law and must be
analyzed from the angular SDE.

This completes the finite-time radial/angular calculation supporting Definition {prf:ref}`def-control-field-at-origin`.

(sec-appendix-a-overdamped-limit-via-singular-perturbation)=
## A.4 Overdamped Limit via Singular Perturbation (Proof of Theorem {prf:ref}`thm-overdamped-limit`)

This section provides the full proof of Theorem {prf:ref}`thm-overdamped-limit` using singular perturbation theory.

**Setup.** Work with the conservative second-order SDE in Theorem
{prf:ref}`thm-overdamped-limit`, with $\beta_{\mathrm{curl}}=0$ and
$u_\pi=0$:

$$
m\,\ddot z^k+\gamma\,\dot z^k+G^{kj}\partial_j\Phi_{
\mathrm{eff}}+\Gamma^k_{ij}\dot z^i\dot z^j
=\sqrt{2\gamma T_c}\,(G^{-1/2})^{kj}\,\xi^j.
$$

The factor $\sqrt{\gamma}$ in the noise is required by the
fluctuation--dissipation relation.  Let $t$ be physical time and put
$s=t/\gamma$ for the computation-time variable used in the theorem.

**Step 1: Fast velocity scale.**

The velocity equation has relaxation time $m/\gamma$.  On every compact
coordinate patch on which $G$, $G^{-1}$, $\Phi_{\mathrm{eff}}$, and the
Christoffel symbols are bounded and locally Lipschitz, the velocity
therefore has a fast Ornstein--Uhlenbeck layer while the position evolves
on the slower time scale.  The Smoluchowski--Kramers reduction gives, in
physical time,

$$
dz^k=-\frac1\gamma G^{k\ell}\partial_\ell\Phi_{\mathrm{eff}}\,dt
-\frac{T_c}{\gamma}G^{ij}\Gamma^k_{ij}\,dt
+\sqrt{\frac{2T_c}{\gamma}}\,(G^{-1/2})^{kj}\,dW_t^j
+o(1).
$$

The $o(1)$ term is in the chosen local topology as $m/\gamma\to0$; a
global estimate additionally requires the confinement and boundary
hypotheses of the theorem.

**Step 2: Computation-time rescaling.**

Since $dt=\gamma\,ds$ and $dW_t=\sqrt{\gamma}\,dW_s$, the preceding
equation becomes

$$
dz^k=\left[-G^{k\ell}\partial_\ell\Phi_{\mathrm{eff}}
-T_cG^{ij}\Gamma^k_{ij}\right]ds
+\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW_s^j.
$$

This is the Ito equation stated in Theorem {prf:ref}`thm-overdamped-limit`.
The contraction $G^{ij}\Gamma^k_{ij}$ is the coordinate correction for the
chosen Riemannian reference measure; it is not the inertial geodesic term.

**Step 3: Inertial and geodesic terms.**

On the fast layer $\dot z$ is $O(\gamma^{-1})$ in physical time for a
bounded force, so $\Gamma(\dot z,\dot z)$ is $O(\gamma^{-2})$ there.  Its
contribution vanishes in the singular limit after the declared rescaling.
This argument is conservative: adding a curl mobility, policy forcing, or
non-reversible jumps changes the reduced generator and is not covered by
this appendix.

**Step 4: Scope of the error statement.**

The fast layer has width $O(m/\gamma)$, but the displayed exponential
bound sometimes quoted for Euclidean constant-coefficient Langevin
systems does not follow from the hypotheses above.  A quantitative bound
requires uniform derivative, confinement, and boundary estimates (and a
choice of norm); none is asserted here.  Under those additional standard
Smoluchowski--Kramers hypotheses, the local reduction can be promoted to a
finite-horizon error estimate.

This completes the singular-perturbation proof sketch of Theorem
{prf:ref}`thm-overdamped-limit`. $\square$

:::{prf:remark} Physical interpretation
:label: rem-physical-interpretation

The overdamped limit corresponds to:
- **Information geometry:** The "friction" $\gamma$ represents the rate of information dissipation (forgetting). High friction means the system equilibrates quickly to the local gradient.
- **Diffusion models:** Standard score-based diffusion models operate entirely in the overdamped regime, with $\gamma \to \infty$ implicitly.
- **Neural network training:** The geodesic term $\Gamma(\dot{z},\dot{z})$ can be interpreted as a "momentum correction" that accounts for the curvature of the loss landscape. In standard gradient descent (overdamped), this term is ignored.

:::
(sec-appendix-a-classification-as-relaxation)=
## A.5 Classification as Relaxation (Conditional derivation)

:::{prf:proposition} Conditional Classification Relaxation
:label: thm-classification-as-relaxation-a

Under the conservative, deterministic overdamped dynamics with the smooth class-conditioned potential $V_y$:

$$
dz = -G^{-1}(z)\nabla V_y(z)\,ds, \qquad T_c=0,

$$
The limiting chart assignment satisfies $K(\lim_{s\to\infty}z(s))\in\mathcal{A}_y$ whenever the trajectory converges to a minimum in $K^{-1}(\mathcal{A}_y)$ and the initial condition lies in its basin.

:::

(proof-thm-classification-as-relaxation-a)=
:::{prf:proof}

**Step 1: Lyapunov function.**

Use the smooth potential itself,

$$
L(z):=V_y(z).
$$

The soft-router definition of $V_y$ is differentiable wherever the router weights are differentiable. No
claim that a class-$y$ chart is a global minimum is made here; that is a training or modeling condition.

**Step 2: Deterministic descent.**

For the conservative overdamped equation,

$$
\frac{dL}{ds}=dV_y(\dot z)
=-\nabla V_y(z)^{\mathsf T}G^{-1}(z)\nabla V_y(z)\le 0.
$$

Thus every trajectory that remains in a compact sublevel set approaches the largest invariant subset of
$\{z:\nabla V_y(z)=0\}$ under the usual smoothness and completeness hypotheses. If the selected minimum is in
$K^{-1}(\mathcal A_y)$, the limiting chart belongs to $\mathcal A_y$.

This is a conditional deterministic statement. At $T_c>0$, noise permits barrier crossing and the basin claim becomes a
finite-time or small-noise estimate; it is not almost-sure convergence to a class at fixed temperature. The same
argument does not cover a hard chart index, nonzero curl, or an unspecified jump process.

This completes the conditional derivation. \(\square\)

:::
:::{prf:remark} Connection to Classification Accuracy
:label: rem-connection-to-classification-accuracy

The theorem provides a geometric interpretation of classification accuracy: a sample $x$ is correctly classified if and only if $\text{Enc}(x) \in \mathcal{B}_{y_{\text{true}}}$. Misclassification occurs when the encoder maps $x$ to the wrong basin—either due to encoder limitations or overlap between class distributions in observation space.

:::



(sec-appendix-a-area-law)=
## A.6 Operational area-law normalization and conditional counting

This section records the normalization used for the Causal Information Bound and
the hypotheses of the associated counting and radial calculations.  It does not
prove a universal bulk-to-boundary area law or a physical identification with
black-hole entropy.  The $1/4$ value in the two-dimensional convention comes
from the normalized Poincaré metric together with an explicit one-nat cell
permit.

**Setup.** Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold. We seek the maximum bulk information $I_{\text{bulk}}$ that can be distinguished by an external observer through the boundary $\partial\mathcal{Z}$.

**Scope.** Sections A.6.0–A.6.0h give local geometric facts and a
conditional independent-cell counting model.  Sections A.6.1–A.6.3 record a
formal spherical ansatz and state the additional bulk-to-boundary permit that a
field-theoretic derivation would require.  Sections A.6.4–A.6.7 assemble the
operational normalization and its dimension-dependent coefficient.  The former
field-theoretic route is retained for auditability, not as an established
derivation.



(sec-appendix-a-foundational-axioms)=
### A.6.0 Foundational Axioms for Microstate Counting

This section states the information-theoretic conventions used by the
conditional counting model.  The black-hole microstate literature is a
structural analogy only; no physical entropy identification is imported
{cite}`strominger1996microscopic`.

:::{prf:axiom} A.6.0a (Operational Distinguishability)
:label: ax-a-operational-distinguishability

Two probability distributions $p, q \in \mathcal{P}(\mathcal{Z})$ are **operationally distinguishable** if and only if:

$$
D_{\text{KL}}(p \| q) \geq 1 \text{ nat}.

$$
*Justification.* This is an **operational definition**, not a derived fact. The choice of 1 nat as the threshold is grounded in:

1. **Asymptotic error exponent.** For $n$ i.i.d. samples, the optimal Type II error probability at fixed Type I error decays as $\exp(-n \cdot D_{\text{KL}})$ (Stein's lemma). Thus $D_{\text{KL}} = 1$ nat corresponds to error decay rate $e^{-n}$.

2. **Information-theoretic meaning.** 1 nat = log(e) ≈ 1.44 bits represents a "natural unit" of information, where the likelihood ratio $p(x)/q(x)$ has expected log-value 1 under $p$.

3. **Dimensional analysis.** The nat is the natural unit when using natural logarithms; choosing 1 nat as the threshold makes the subsequent formulas dimensionally consistent.

*Remark.* Alternative thresholds (e.g., 1 bit = ln 2 nats) would change the numerical coefficient in the Area Law but not its structure.

:::

:::{prf:theorem} A.6.0b (Chentsov's Uniqueness Theorem)
:label: thm-a-chentsov-uniqueness

For a regular finite-dimensional family of strictly positive distributions, the
**Fisher Information Metric** is, up to constant scaling, the unique Riemannian
metric invariant under all Markov morphisms in the statistical category used by
Chentsov's theorem.

**Statement.** Let $\mathcal{M}$ be a statistical manifold parameterized by $\theta \in \Theta$. Any Riemannian metric $g$ on $\mathcal{M}$ satisfying:
1. **Markov invariance:** $g$ is preserved under the specified Markov morphisms (conditional expectations)
2. **Smoothness and regularity:** $g$ varies smoothly with $\theta$ and the model satisfies the regularity assumptions of the theorem

is proportional to the Fisher Information Metric:

$$
g_{ij}(\theta) = c \cdot \mathbb{E}_\theta\left[\frac{\partial \log p(x|\theta)}{\partial \theta^i} \frac{\partial \log p(x|\theta)}{\partial \theta^j}\right]

$$
for some constant $c > 0$.

*Proof.* This is the standard external uniqueness theorem; see Chentsov (1982)
{cite}`chentsov1982statistical` and Campbell (1986)
{cite}`campbell1986extended`.  The present volume invokes it only as a
regularity and normalization reference.  It does not use the theorem to infer
a finite cell-to-nat assignment or an area law. $\square$

*Scope.* Within that statistical category, Chentsov's theorem identifies the
Fisher metric up to scale.  The scale and the operational cell convention used
below remain model permits.

:::

::::{admonition} Physics Isomorphism: Fisher Information Metric
:class: note
:name: pi-fisher-information

**In Physics:** The Fisher Information Metric $\mathcal{F}_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta^i}\frac{\partial \log p}{\partial \theta^j}\right]$ is the unique Riemannian metric on statistical manifolds invariant under sufficient statistics (Chentsov's Theorem) {cite}`chentsov1982statistical,amari1985differential`.

**In Implementation:** The latent metric $G(z)$ combines value curvature with Fisher Information ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`):

$$
G_{ij}(z) = \nabla^2_{ij} V(z) + \lambda\,\mathcal{F}_{ij}(z)

$$
where $\mathcal{F}_{ij} = \mathbb{E}_{a\sim\pi}[\partial_i \log\pi \cdot \partial_j \log\pi]$ is the state-space Fisher component $G_\pi$.

**Correspondence Table:**
| Information Geometry | Agent (Latent Metric) |
|:---------------------|:----------------------|
| Parameter space $\Theta$ | Latent state space $\mathcal{Z}$ |
| Fisher metric $\mathcal{F}_{ij}$ | Base metric contribution |
| Sufficient statistics | Macro-state $K$ (chart index) |
| KL divergence $D_{KL}$ | Squared geodesic distance (locally) |
| Natural gradient | Metric-aware policy gradient |

**Significance:** Chentsov's theorem (Theorem {prf:ref}`thm-a-chentsov-uniqueness`) proves the Fisher metric is not a choice but a necessity—any geometry respecting statistical structure must be proportional to Fisher.
::::

:::{prf:definition} A.6.0c (Computational Microstate)
:label: def-a-computational-microstate

A **computational microstate** at resolution $\ell$ is a complete specification of the agent's internal configuration $\mu = (\rho, K, \theta)$ where:
- $\rho \in \mathcal{P}(\mathcal{Z})$ is the belief distribution over the latent manifold
- $K \in \{1, \ldots, |\mathcal{K}|\}$ is the active chart assignment
- $\theta$ are the model parameters

discretized at the Levin Length scale: positions resolved to precision $\ell_L$, probabilities resolved to precision $e^{-1}$ in KL divergence.

Two microstates $\mu_1, \mu_2$ are **boundary-distinguishable** if an external observer, receiving only boundary observations $\partial\mathcal{Z}$, can distinguish them with probability $> 1 - e^{-1}$.

*Remark (Analogy to Physics).* In black hole thermodynamics, a microstate is a specific quantum configuration of the horizon degrees of freedom. Here, a microstate is a specific configuration of the agent's belief state. The boundary plays the role of the horizon: internal distinctions not visible at the boundary do not count toward the entropy.

:::



(sec-appendix-a-microstate-counting)=
### A.6.0d Conditional microstate counting

We now state the conditional counting model for boundary-distinguishable
microstates.  It uses the explicit channel permit below and does not derive a
capacity law for an arbitrary latent model.

:::{prf:lemma} A.6.0d (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-probability-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with Fisher Information Metric, the geodesic distance from the uniform distribution $(1/2, 1/2)$ to a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\tfrac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
*Proof.* The Fisher metric on $\Delta^1$ is:

$$
ds^2 = \frac{dp^2}{p(1-p)}.

$$
Introduce the angular parameterization $p = \cos^2(\theta/2)$, so that $1-p = \sin^2(\theta/2)$ and:

$$
dp = -\cos(\theta/2)\sin(\theta/2)d\theta = -\frac{1}{2}\sin\theta \, d\theta.

$$
Then:

$$
ds^2 = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\cos^2(\theta/2)\sin^2(\theta/2)} = \frac{\frac{1}{4}\sin^2\theta \, d\theta^2}{\frac{1}{4}\sin^2\theta} = d\theta^2.

$$
The uniform distribution $(1/2, 1/2)$ corresponds to $\theta = \pi/2$. The vertex $(1, 0)$ corresponds to $\theta = 0$. The geodesic distance is:

$$
d = \int_0^{\pi/2} d\theta = \frac{\pi}{2}. \quad \square

$$
*Interpretation.* One bit of information (distinguishing "heads" from "tails") corresponds to geodesic distance $\pi/2$ in Fisher geometry. This is a derived quantity, not an assumption.

:::

:::{prf:lemma} A.6.0e (Curvature Normalization and the Factor of 4)
:label: lem-a-curvature-normalization-factor-4

The Poincare disk model with constant sectional curvature $K = -1$ has metric:

$$
ds^2 = \frac{4(dx^2 + dy^2)}{(1-|z|^2)^2}.

$$
The factor of 4 is uniquely determined by the curvature normalization.

*Proof.* For a 2D Riemannian manifold with conformal metric $ds^2 = \lambda(z)(dx^2 + dy^2)$, the Gaussian curvature is {cite}`docarmo1992riemannian`:

$$
K = -\frac{1}{2\lambda}\Delta(\log \lambda),

$$
where $\Delta = \partial_x^2 + \partial_y^2$ is the flat Laplacian.

For $\lambda = c/(1-r^2)^2$ where $r^2 = x^2 + y^2$ and $c > 0$:

**Step 1:** Compute $\log \lambda = \log c - 2\log(1-r^2)$.

**Step 2:** Compute the Laplacian. Let $f = \log(1-r^2)$. Then:

$$
\partial_x f = \frac{-2x}{1-r^2}.

$$
Applying the quotient rule to $\partial_x f = -2x \cdot (1-r^2)^{-1}$:

$$
\partial_x^2 f = \frac{-2(1-r^2) - (-2x)(-2x)}{(1-r^2)^2} = \frac{-2 + 2r^2 - 4x^2}{(1-r^2)^2}.

$$
Similarly for $y$. Adding:

$$
\Delta f = \frac{(-2 + 2r^2 - 4x^2) + (-2 + 2r^2 - 4y^2)}{(1-r^2)^2} = \frac{-4 + 4r^2 - 4r^2}{(1-r^2)^2} = \frac{-4}{(1-r^2)^2}.

$$
**Step 3:** Therefore $\Delta(\log \lambda) = -2\Delta f = \frac{8}{(1-r^2)^2}$.

**Step 4:** The curvature is:

$$
K = -\frac{1}{2\lambda} \cdot \frac{8}{(1-r^2)^2} = -\frac{(1-r^2)^2}{2c} \cdot \frac{8}{(1-r^2)^2} = -\frac{4}{c}.

$$
**Step 5:** For $K = -1$, we require $c = 4$. $\square$

*Significance.* The choice $K = -1$ is canonical: it sets the "radius of curvature" to unity, making the hyperbolic distance formula $d(0,z) = 2\text{arctanh}|z|$ dimensionless. The factor of 4 in the metric is a *derived consequence* of the curvature normalization, not an assumption.

:::

:::{prf:definition} A.6.0f (Fisher-coordinate cell convention)
:label: prop-a-area-minimal-distinguishable-cell

On a two-dimensional Poincaré chart with curvature normalization $K=-1$,
$G(0)=4I$.  If the declared coordinate resolution is $\ell_L$, a square
coordinate cell of side $\ell_L$ has Riemannian area

$$
A_{\mathrm{cell}}=4\ell_L^2.
$$

Calling this cell one nat is an operational capacity permit.  The Fisher
metric and Chentsov's theorem fix the local metric up to scale; they do not,
by themselves, identify a finite cell with one nat or prove an area law.
This convention is a two-dimensional chart calculation and is not the
$(D-1)$-dimensional boundary normalization used for the general operational
capacity in {ref}`sec-causal-information-bound`.

:::

:::{prf:proposition} A.6.0g (Conditional boundary-channel convention)
:label: thm-a-boundary-channel-capacity

Suppose a **two-dimensional boundary channel** is explicitly tiled by
independent cells of Riemannian area $4\ell_L^2$, and suppose the channel
permit assigns one nat to each such cell.  Then its declared capacity is

$$
C_\partial=\frac{A}{4\ell_L^2}\ \mathrm{nats}.
$$

This is a conditional counting statement for a two-dimensional boundary
(the boundary dimension is not the same as the $D=2$ Poincaré-disk bulk
case).  The general $D$-dimensional operational capacity is defined with
$\nu_D$ and the dimensionally normalized Levin length in
{ref}`sec-causal-information-bound`; no field equation or Fisher theorem
supplies the cell-to-nat permit.

:::

:::{prf:proposition} A.6.0h (Conditional microstate count)
:label: thm-a-microstate-count-area-law

Under the channel and achievability permit of Proposition
{prf:ref}`thm-a-boundary-channel-capacity`, the number of distinguishable
messages is bounded by

$$
\Omega\le e^{C_\partial},\qquad
\log\Omega\le \frac{A}{4\ell_L^2}.
$$

Equality is an additional coding assumption: it requires an achievable
independent-cell code and a specified input distribution.  The data
processing inequality supplies the upper-bound direction, but it does not
prove equality for an arbitrary latent model.

:::

(sec-appendix-a-holographic-reduction)=
### A.6.1 Conditional bulk-to-boundary route

:::{prf:remark} A.6.1 (Conditional bulk-to-boundary permit)
:label: lem-a-bulk-to-boundary-conversion

A relation of the form

$$
I_{\mathrm{bulk}}=\frac1\kappa\oint_{\partial\mathcal Z}
\operatorname{Tr}(K)\,dA_G
$$

may be adopted as an additional bulk-to-boundary permit for a specified
stationary field theory.  It is not a consequence of the contracted Bianchi
identity: that identity gives a covariantly conserved Einstein tensor, while
$\rho_I$ is an independently defined information density.  A derivation of
this permit must specify the action, source coupling, boundary term, and
units.  None is assumed by the Metric Law elsewhere in the volume.

:::

(sec-appendix-a-saturation-geometry)=
### A.6.2 Formal spherical saturation ansatz

For an isotropic manifold with spherical symmetry, we use the ansatz:

$$
ds^2 = A(r) \, dr^2 + r^2 \, d\Omega_{n-1}^2,

$$
where $d\Omega_{n-1}^2$ is the metric on the unit $(n-1)$-sphere.

:::{prf:remark} A.6.2 (Formal spherical saturation ansatz)
:label: prop-a-saturation-metric-solution

For a selected spherical coordinate model one may study

$$
ds^2=A(r)\,dr^2+r^2d\Omega_{n-1}^2,
$$

and introduce a Schwarzschild-style denominator

$$
A(r)^{-1}=1-\frac{2\mu(r)}{(n-2)r^{n-2}}
-\frac{\Lambda_{\mathrm{eff}}r^2}{n(n-1)}.
$$

This is an ansatz, not a solution of the capacity-constrained Metric Law.
Uniform $T_{ij}=\sigma G_{ij}$ instead gives a constant-curvature source
term after the field equation is written out; a mass term and the sign of
$\Lambda_{\mathrm{eff}}$ require a separate boundary-value calculation.
Consequently the functions $\mu$ and $\Lambda_{\mathrm{eff}}$ below are
bookkeeping parameters for the ansatz, not an information mass derived from
$\rho_I$.

:::

(sec-appendix-a-horizon-condition)=
### A.6.3 Horizon condition for the ansatz

:::{prf:definition} A.6.3 (Information Horizon)
:label: def-a-information-horizon

The **information horizon** $r_h$ is the smallest positive root of:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $A(r_h) \to \infty$ and $G^{rr}(r_h) \to 0$.

:::

For $n = 2$ (the Poincare disk case), the formula simplifies. The Poincare metric already encodes the horizon at $|z| = 1$:

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1-|z|^2)^2} \xrightarrow{|z| \to 1} \infty.

$$


(sec-appendix-a-fisher-normalization)=
### A.6.4 Fisher normalization and the local factor 4

The local factor $G(0)=4I$ fixes the coordinate-to-Riemannian-area
conversion on the normalized Poincaré chart.  It does not, by itself, fix a
capacity coefficient or a bulk-to-boundary identity.

:::{prf:remark} A.6.4a (Scope of the Fisher normalization)
:label: rem-a-connection-microstate-counting

The simplex distance and curvature normalization below justify the local
metric convention used in Proposition {prf:ref}`prop-a-area-minimal-cell`.
The statement that a cell carries one nat remains the explicit channel
permit of Proposition {prf:ref}`thm-a-boundary-channel-capacity`; it is not a
consequence of Chentsov's uniqueness theorem.

:::

:::{prf:lemma} A.6.4 (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with the Fisher Information Metric, the geodesic distance between the uniform distribution $(1/2, 1/2)$ and a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\frac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
*Proof.* See Lemma {prf:ref}`lem-a-geodesic-distance-probability-simplex` for the full derivation. $\square$

:::

:::{prf:proposition} A.6.5 (Poincaré chart area conversion)
:label: prop-a-area-minimal-cell

On the normalized two-dimensional Poincaré chart, a coordinate cell of side
$\ell_L$ has Riemannian area $4\ell_L^2$ at the origin.  This is the local
geometric conversion used by the conditional cell-counting convention; it
is not a statement about the entropy of an arbitrary data distribution.

*Proof.* $G(0)=4I$, so $\sqrt{\det G(0)}=4$. $\square$

:::

(sec-appendix-a-assembly)=
### A.6.5 Assembly of the operational normalization

:::{prf:definition} A.6.6 (Operational area-law normalization)
:label: thm-a-complete-derivation-area-law

For a declared latent dimension $D$, boundary measure, coefficient $\nu_D$,
and Levin length $\ell_L$, define the operational capacity

$$
I_{\max}:=\nu_D\,
\frac{\operatorname{Area}(\partial\mathcal Z)}{\ell_L^{D-1}}.
$$

This is the normalization used by the Causal Information Capacity in
{ref}`sec-causal-information-bound`.  A field-theoretic derivation would
need, as separate hypotheses, a valid bulk-to-boundary identity, a solution
of the chosen metric equation, a boundary extrinsic-curvature estimate, and
a dimensionally consistent coupling.  The former A.6.1--A.6.3 argument does
not establish those hypotheses, so this definition carries no claim that
the Metric Law generates the area law.

:::

:::{prf:corollary} A.6.7 (Dimension-Dependent Coefficient)
:label: cor-a-dimension-dependent-coefficient

Under the operational capacity convention, a $D$-dimensional latent manifold
with $(D-1)$-sphere boundary uses the dimension-dependent normalization:

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}},

$$
where the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`) is:

$$
\nu_D = \frac{(D-1)\Omega_{D-1}}{8\pi} = \frac{(D-1)\pi^{(D-2)/2}}{4\,\Gamma(D/2)},

$$
with $\Omega_{D-1} = 2\pi^{D/2}/\Gamma(D/2)$ the surface area of the unit $(D-1)$-sphere.

**Explicit values:**

| $D$ | $\Omega_{D-1}$ | $\nu_D$    | Numerical |
|-----|----------------|------------|-----------|
| 2   | $2\pi$         | $1/4$      | 0.250     |
| 3   | $4\pi$         | $1$        | 1.000     |
| 4   | $2\pi^2$       | $3\pi/4$   | 2.356     |
| 5   | $8\pi^2/3$     | $4\pi/3$   | 4.189     |
| 6   | $\pi^3$        | $5\pi^2/8$ | 6.169     |

*Remark.* The coefficient $\nu_D$ is **not monotonic** in $D$: it increases from $D=2$ to a peak at $D \approx 9$ ($\nu_9 \approx 9.4$), then decreases toward zero. For typical latent dimensions ($3 \le D \le 20$), $\nu_D > \nu_2 = 1/4$, so using the 2D coefficient **underestimates** capacity. For very high dimensions ($D \gtrsim 22$), $\nu_D < 1/4$, so the 2D coefficient **overestimates** capacity—this is the dangerous case (false safety). Implementers should always use the dimension-appropriate coefficient.

:::

:::{warning}
:name: warning-dimension-dependent-node-56

**Implementation Note for Node 56 (CapacityHorizonCheck):**

The saturation ratio $\eta_{\text{Sch}} = I_{\text{bulk}} / I_{\max}$ depends on the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`):

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}.

$$
The default $\nu_2 = 1/4$ assumes a 2-dimensional latent manifold (Poincare disk). For $D$-dimensional latent spaces, use the appropriate $\nu_D$ from Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`.

Using the wrong coefficient leads to:
- **$\nu > \nu_D$ (typical for $D > 21$):** **Dangerous.** False safety—agent enters super-saturated regime undetected.
- **$\nu < \nu_D$ (typical for $3 \le D \le 20$):** Conservative—unnecessary fusion triggered, but safe.

**Implementation Code:**
```python
def holographic_coefficient(D: int) -> float:
    """Compute nu_D = (D-1) * Omega_{D-1} / (8 * pi)"""
    import math
    if D < 2:
        return 0.0
    omega = 2 * (math.pi ** (D / 2)) / math.gamma(D / 2)
    return (D - 1) * omega / (8 * math.pi)
```

:::

:::{prf:remark} A.6.8 (Scope of curvature identities)
:label: rem-a-gauss-bonnet-generalization

The contracted Bianchi identity states $\nabla^iG_{ij}=0$; it is not the
boundary-divergence identity
$\int R\,d\mu_G=2\oint\operatorname{Tr}(K)\,dA_G$.  The latter is not valid
for a general manifold or dimension without additional curvature terms,
boundary terms, and field equations.  Classical Gauss--Bonnet identities
have their own dimension and topology hypotheses.  Therefore no such
identity is used to prove the operational capacity in this volume.

:::

:::{prf:remark} A.6.9 (Status of the area-law arguments)
:label: rem-a-non-circularity

The local Fisher/Poincaré calculation and the independent channel permit are
useful normalization checks. They do not derive a universal area law: the
cell-to-nat assignment is a coding assumption, and the former field-theoretic
route requires additional lemmas that are not established here. The main
text consequently presents
$I_{\max}=\nu_D\operatorname{Area}(\partial\mathcal Z)/\ell_L^{D-1}$ as an
operational capacity convention. Any comparison with the
Bekenstein--Hawking formula is an explicitly labelled mathematical analogy,
not a physical identification.

:::

(sec-appendix-a-remark-bekenstein-hawking)=
### A.6.6 Remark: Connection to Bekenstein-Hawking

The expression $A/(4\ell_P^2)$ is a physical result of gravitational
thermodynamics. The agent quantity
$\nu_D\operatorname{Area}(\partial\mathcal Z)/\ell_L^{D-1}$ is an
operational normalization for a declared interface channel. Their similar
form supports a mathematical analogy only; no field equation, entropy
identification, or physical equivalence follows from the notation.
