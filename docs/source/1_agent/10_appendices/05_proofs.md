(sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws)=
# {ref}`Appendix E <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`: Rigorous Proof Sketches for Ontological and Metabolic Laws

## TLDR

- This appendix contains rigorous proof sketches backing the ontology/metabolism results in the later cognition chapters.
- Read it when you want the mathematical spine behind the narrative statements; otherwise treat it as a reference.

:::{div} feynman-prose
This appendix collects calculations for the cognition and multi-agent chapters. The multi-agent proofs track the operator, measure, and variational functional used in each result, so that changes of representation can be checked directly.
:::

We operate on the latent Riemannian manifold $(\mathcal{Z}, G)$ with belief measures $\rho \in \mathcal{P}(\mathcal{Z})$.



(sec-appendix-e-proof-of-theorem-prf-ref)=
## E.1 Proof of Theorem {prf:ref}`thm-fission-criterion`

**Statement:** The ontology should expand from $N_c$ to $N_c + 1$ charts if and only if $\Xi > \Xi_{\text{crit}}$ and $\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}$.

**Hypothesis:** Let $\mathcal{S}[N_c] = \inf_{\theta} \mathcal{S}_{\text{onto}}(\theta, N_c)$ be the value function of the ontological action for $N_c$ charts.

(proof-thm-the-fission-trigger)=
:::{prf:proof}

Consider the discrete variation $\Delta \mathcal{S} = \mathcal{S}[N_c + 1] - \mathcal{S}[N_c]$. By the definition of the Ontological Action ({ref}`Section 30.3 <sec-the-fission-criterion>`):

$$
\mathcal{S}_{\text{onto}} = -\mathcal{S}_{\text{task}} + \mu_{\text{size}} \cdot N_c,

$$
where $\mathcal{S}_{\text{task}} = \mathbb{E}[\langle V \rangle]$ is the expected task value.

Expanding $\mathcal{S}_{\text{task}}$ via a first-order Taylor approximation in the space of representations:

$$
\mathcal{S}_{\text{task}}[N_c + 1] \approx \mathcal{S}_{\text{task}}[N_c] + \frac{\partial \langle V \rangle}{\partial N_c}.

$$
The marginal utility of a new chart is $\frac{\partial \langle V \rangle}{\partial N_c} = \Delta V_{\text{proj}}$. The complexity cost is $\mu_{\text{size}}$. Therefore:

$$
\Delta \mathcal{S} = -\Delta V_{\text{proj}} + \mu_{\text{size}}.

$$
The transition $N_c \to N_c + 1$ is the global minimizer iff $\Delta \mathcal{S} < 0$, which yields:

$$
\Delta V_{\text{proj}} > \mu_{\text{size}} = \mathcal{C}_{\text{complexity}}.

$$
The condition $\Xi > \Xi_{\text{crit}}$ ensures that the second variation of the texture-entropy functional $\delta^2 H(z_{\text{tex}})$ is negative-definite at the vacuum. This precludes the absorption of the signal into the existing noise floor: if $\Xi \le \Xi_{\text{crit}}$, the texture residual $z_{\text{tex}}$ is truly unpredictable noise, and adding a chart provides no informational benefit. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-a)=
## E.2 Proof of Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`

**Statement:** The emergence of a new chart follows a supercritical pitchfork bifurcation with control parameter $\mu = \Xi - \Xi_{\text{crit}}$.

**Hypothesis:** The potential $\Phi_{\text{onto}}(r)$ is $SO(n)$-invariant near $r=0$, where $r = \|q_* - q_{\text{parent}}\|$ is the radial distance of the new query from the parent.

(proof-thm-supercritical-pitchfork-bifurcation)=
:::{prf:proof}

Let $f(\Xi) = \Xi - \Xi_{\text{crit}}$ be the control parameter. By $SO(n)$ symmetry, the Ontological Action can only depend on even powers of $r$ near the origin. We expand in a power series:

$$
\mathcal{S}(r) = \mathcal{S}_0 - \frac{1}{2}f(\Xi)r^2 + \frac{1}{4}\beta r^4 + O(r^6),

$$
where $\beta > 0$ for stability (the quartic term must be positive for bounded energy).

The stationarity condition $\frac{\partial \mathcal{S}}{\partial r} = 0$ yields:

$$
-f(\Xi)r + \beta r^3 = 0 \implies r(f(\Xi) - \beta r^2) = 0.

$$
This has solutions:
1. $r = 0$ (trivial, no new chart)
2. $r^2 = f(\Xi)/\beta$ (symmetry-broken state)

**Analysis of stability:**
- For $f(\Xi) < 0$ (i.e., $\Xi < \Xi_{\text{crit}}$): The Hessian at $r=0$ is $\frac{\partial^2 \mathcal{S}}{\partial r^2}|_{r=0} = -f(\Xi) > 0$. Thus $r=0$ is a stable minimum.
- For $f(\Xi) > 0$ (i.e., $\Xi > \Xi_{\text{crit}}$): The Hessian at $r=0$ becomes $-f(\Xi) < 0$ (unstable). New minima appear at $r^* = \sqrt{f(\Xi)/\beta}$.

Since $r \ge 0$ is a radial coordinate, this constitutes a **supercritical pitchfork bifurcation** where the symmetry-broken state $r^* > 0$ becomes the unique stable equilibrium for $\Xi > \Xi_{\text{crit}}$.

The bifurcation diagram: for $\Xi < \Xi_{\text{crit}}$, the system has a single stable fixed point at $r=0$; for $\Xi > \Xi_{\text{crit}}$, the origin becomes unstable and two symmetric branches (in the full space, a sphere of radius $r^*$) emerge. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-b)=
## E.3 Proof of Theorem {prf:ref}`thm-generalized-landauer-bound`

**Statement.** Under the compact-domain, positive-density, no-flux, mass-preservation,
and calibration hypotheses in the theorem, the conditional Landauer-form estimate is
$\dot{\mathcal M}(s)\ge T_c\lvert dH(\rho_s)/ds\rvert$.

(proof-thm-generalized-landauer-bound)=
:::{prf:proof}
**Proof.** The WFR equation is
$\partial_s\rho=-\nabla\!\cdot(\rho v)+\rho r$. For
$H(\rho)=-\int\rho\ln\rho\,d\mu_G$, differentiation gives

$$
\frac{dH}{ds}=-\int_{\mathcal Z}(1+\ln\rho)\partial_s\rho\,d\mu_G.
$$

The no-flux condition removes the boundary contribution. Mass preservation removes
the term $\int\rho r\,d\mu_G$, so

$$
\frac{dH}{ds}
=-\int_{\mathcal Z}\rho\langle\nabla\ln\rho,v\rangle_G\,d\mu_G
 -\int_{\mathcal Z}\rho r\ln\rho\,d\mu_G.
$$

Set

$$
E_v=\int\rho\|v\|_G^2d\mu_G,\quad
E_r=\int\rho r^2d\mu_G,\quad
I_\rho=\int\rho\|\nabla\ln\rho\|_G^2d\mu_G,\quad
J_\rho=\int\rho(\ln\rho)^2d\mu_G.
$$

Cauchy--Schwarz in $L^2(\rho d\mu_G)$ gives

$$
\left|\int\rho\langle\nabla\ln\rho,v\rangle_Gd\mu_G\right|
\le\sqrt{I_\rho E_v},\qquad
\left|\int\rho r\ln\rho\,d\mu_G\right|
\le\sqrt{J_\rho E_r}.
$$

Adding the estimates yields
$|dH/ds|\le\sqrt{I_\rho E_v}+\sqrt{J_\rho E_r}$. The theorem's explicit
calibration compares this right-hand side with
$\dot{\mathcal M}=\sigma_{\mathrm{met}}(E_v+\lambda^2E_r)$ and therefore gives
the claimed Landauer-form inequality. No de Bruijn identity or universal physical
identification is required. $\square$

:::



## E.4 Proof of Theorem {prf:ref}`thm-deliberation-optimality-condition`

**Statement:** The optimal computation budget $S^*$ satisfies $\frac{d}{ds} \langle V \rangle_{\rho_s}|_{s=S^*} = \dot{\mathcal{M}}(S^*)$.

**Hypothesis:** $S^*$ is an interior point of $[0, S_{\max}]$.

(proof-thm-deliberation-optimality)=
:::{prf:proof}

Define the deliberation functional:

$$
\mathcal{F}(S) = -\int_{\mathcal{Z}} V(z) \rho(S, z) \, d\mu_G + \int_0^S \dot{\mathcal{M}}(u) \, du.

$$
The necessary condition for an extremum is $\mathcal{F}'(S) = 0$. By the Leibniz integral rule:

$$
\mathcal{F}'(S) = -\int_{\mathcal{Z}} V(z) \partial_s \rho(S, z) \, d\mu_G + \dot{\mathcal{M}}(S).

$$
Using the result that $\partial_s \rho$ is governed by the WFR operator $\mathcal{L}_{\text{WFR}}$:

$$
\mathcal{F}'(S) = -\int_{\mathcal{Z}} V \mathcal{L}_{\text{WFR}}\rho \, d\mu_G + \dot{\mathcal{M}}(S).

$$
By the adjoint property of the WFR operator (the formal $L^2(\rho)$ adjoint):

$$
\int V \mathcal{L}_{\text{WFR}}\rho \, d\mu_G = \int \rho \mathcal{L}_{\text{WFR}}^* V \, d\mu_G,

$$
where $\mathcal{L}_{\text{WFR}}^* V = -\langle \nabla V, v \rangle_G + Vr$ (transport-adjoint plus reaction).

For gradient flows in the covariant case, $v = -G^{-1}\nabla_A V$ with $\nabla_A V := \nabla V - A$:

$$
\mathcal{L}_{\text{WFR}}^* V = G^{-1}(\nabla V, \nabla_A V) + Vr.

$$
Thus:

$$
\mathcal{F}'(S) = -\int \rho \left( G^{-1}(\nabla V, \nabla_A V) + Vr \right) d\mu_G + \dot{\mathcal{M}}(S).

$$
In the conservative case ($A=0$), $G^{-1}(\nabla V, \nabla_A V) = \|\nabla V\|_G^2$, the power dissipated by the value-gradient flow. The stationarity condition $\mathcal{F}'(S^*) = 0$ gives:

$$
\frac{d}{ds} \langle V \rangle_{\rho_s}\bigg|_{s=S^*} = \dot{\mathcal{M}}(S^*).

$$
This states that the optimal stopping time $S^*$ is reached when the power dissipated by the value-gradient flow exactly matches the metabolic cost rate. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-d)=
## E.5 Proof of Theorem {prf:ref}`thm-augmented-drift-law`

**Statement:** $F_{\text{total}} = -G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}}$.

**Hypothesis:** The agent's path minimizes $\mathcal{S} = \int L(z, \dot{z}) \, dt$ with Lagrangian $L = \frac{1}{2}\|\dot{z}\|_G^2 - (V + \beta_{\text{exp}}\Psi_{\text{causal}})$.

(proof-thm-the-augmented-drift-law)=
:::{prf:proof}

The Euler-Lagrange equations for the functional are:

$$
\frac{d}{dt} \frac{\partial L}{\partial \dot{z}^k} - \frac{\partial L}{\partial z^k} = 0.

$$
**Computing the momentum:**

$$
\frac{\partial L}{\partial \dot{z}^k} = \frac{\partial}{\partial \dot{z}^k}\left( \frac{1}{2}G_{ij}(z)\dot{z}^i \dot{z}^j \right) = G_{kj}\dot{z}^j = p_k.

$$
**Time derivative of momentum:**

$$
\frac{d}{dt}(G_{kj}\dot{z}^j) = G_{kj}\ddot{z}^j + \frac{\partial G_{kj}}{\partial z^m}\dot{z}^m \dot{z}^j.

$$
**Potential gradient:**

$$
\frac{\partial L}{\partial z^k} = \frac{1}{2}\frac{\partial G_{ij}}{\partial z^k}\dot{z}^i\dot{z}^j - \partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
**Euler-Lagrange equation:**

$$
G_{kj}\ddot{z}^j + \frac{\partial G_{kj}}{\partial z^m}\dot{z}^m \dot{z}^j - \frac{1}{2}\frac{\partial G_{ij}}{\partial z^k}\dot{z}^i\dot{z}^j = -\partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
Recognizing the Christoffel symbols of the first kind $[ij, k] = \frac{1}{2}(\partial_i G_{jk} + \partial_j G_{ik} - \partial_k G_{ij})$:

$$
G_{kj}\ddot{z}^j + [ij, k]\dot{z}^i\dot{z}^j = -\partial_k V - \beta_{\text{exp}}\partial_k \Psi_{\text{causal}}.

$$
Contracting with $G^{mk}$ and using $\Gamma^m_{ij} = G^{mk}[ij, k]$:

$$
\ddot{z}^m + \Gamma^m_{ij}\dot{z}^i\dot{z}^j = -G^{mk}\partial_k V - \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}}.

$$
This is the geodesic equation with forcing terms. In the **overdamped limit** ({ref}`Section 22.3 <sec-the-unified-effective-potential>`), inertia is negligible and the acceleration term vanishes, leaving:

$$
\dot{z}^m = -G^{mk}\partial_k V + \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}} = F^m_{\text{total}}.

$$
The drift field $F_{\text{total}}$ is the first-order velocity approximation, proving the additive force of curiosity. $\square$

:::



(sec-appendix-e-proof-of-theorem-prf-ref-e)=
## E.6 Proof of Theorem {prf:ref}`thm-interventional-closure`

**Statement:** The macro-ontology $K$ is interventionally closed iff $I(K_{t+1}; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0$.

**Hypothesis:** Let $\mathcal{M}$ be a Markov Blanket for $K$.

(proof-thm-interventional-closure)=
:::{prf:proof}

We compare the mutual information under the observational measure $P$ and the interventional measure $P_{do(K^{\text{act}})}$.

**Observational case:** By the Causal Enclosure condition ({ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`):

$$
I(K_{t+1}; Z_{\text{micro}, t} | K_t, K^{\text{act}}_t) = 0 \quad \text{under } P.

$$
This states that the macro-state $K_{t+1}$ is conditionally independent of the micro-texture $Z_{\text{micro}, t}$ given the current macro-state and action.

**Interventional case:** The $do(K^{\text{act}}_t)$ operator performs a graph surgery that removes all incoming edges to $K^{\text{act}}_t$ while preserving all other mechanisms. By Pearl's Causal Markov Condition {cite}`pearl2009causality`:

$$
P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}) \text{ remains invariant under } do(K^{\text{act}}_t).

$$
This is because the mechanism $P(K_{t+1} | \text{parents}(K_{t+1}))$ is a structural equation that does not depend on how $K^{\text{act}}_t$ was generated.

**Combining the conditions:**
If the observational distribution satisfies $I = 0$, then:

$$
P(K_{t+1} | K_t, K^{\text{act}}_t) = P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}) \quad \forall Z_{\text{micro}, t}.

$$
Since the mechanism is invariant under intervention:

$$
P(K_{t+1} | K_t, do(K^{\text{act}}_t)) = P(K_{t+1} | K_t, K^{\text{act}}_t) = P(K_{t+1} | K_t, K^{\text{act}}_t, Z_{\text{micro}, t}).

$$
Therefore, $I(K_{t+1}; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0$.

**Contrapositive (violation):** If $I > 0$ under $do(K^{\text{act}}_t)$, there exists a back-door path through $Z_{\text{micro}, t}$:

$$
K_t \leftarrow Z_{\text{micro}, t} \to K_{t+1}.

$$
This path was confounded in observational data (the correlation between $Z_{\text{micro}}$ and $K_{t+1}$ was screened by the policy generating $K^{\text{act}}_t$). The intervention breaks this screening, exposing the hidden variable. The remedy is **Ontological Expansion** ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`): promote the relevant component of $Z_{\text{micro}}$ to a new macro-variable in $K$. $\square$

:::



(sec-appendix-e-rigorous-proof-of-multi-agent-strategic-tunneling)=
(proof-thm-e7-ground-state-positivity)=
(proof-thm-e7-agmon-decay-bound)=
(proof-cor-e7-adversarial-suppression)=
(proof-thm-e7-feynman-kac)=
(sec-appendix-e-ground-state-existence)=
(pi-spectral-gap)=
(insight-tunneling-inevitable)=
(pi-agmon-estimates)=
(interpretation-adversarial-barrier)=
(pi-feynman-kac)=
(pi-large-deviation)=
## E.7 Scalar spectral and barrier calculations

:::{prf:definition} The scalar metric realization
:label: def-e7-strategic-metric

Use the compact connected smooth scalar manifold and its positive strategic
metric from the existing Appendix E.7 setup. The same spectral margin is
$\|G^{-1/2}hG^{-1/2}\|_{\mathrm{op}}<1$ for sign-indefinite perturbations;
positive perturbations preserve positivity directly. The joint volume is
$w=\sqrt{\det\widetilde G}$, as in
{prf:ref}`thm-game-augmented-laplacian`. Compactness makes a fixed smooth
positive metric uniformly elliptic. This statement concerns the compact
realization already used here, not a replacement of unbounded confinement
by an artificial compact domain.
:::

:::{prf:definition} Scalar form realization
:label: def-e7-strategic-hamiltonian

For the existing real $C^2$ potential $U$ bounded below, set
$q_\sigma[u]=\tfrac{\sigma^2}{2}\int|\nabla u|^2+\int U|u|^2$.
The form domain is $H^1$ on a closed manifold or for the Neumann realization,
and $H_0^1$ for the Dirichlet realization. Its self-adjoint operator is
$H_\sigma=-\sigma^2\Delta_{\widetilde G}/2+U$ with the selected boundary
condition. The domain is the operator domain associated to this form,
not unrestricted $H^2$ on a domain with boundary. Compact embedding gives
compact resolvent for this fixed model.
:::

:::{prf:definition} Forbidden and allowed regions
:label: def-e7-forbidden-region

For an energy $E$ define $A_E=\{U\le E\}$ and $K_E=\{U>E\}$.
These are sets of a scalar potential. Their relation to payoff basins is
tested separately by the unilateral inequalities of
{prf:ref}`thm-nash-equilibrium-as-geometric-stasis`.
:::

:::{prf:theorem} Fixed scalar ground state
:label: thm-e7-ground-state-positivity

The compact connected nonmagnetic scalar realization has a simple ground
eigenvalue $E_0$ and an eigenfunction positive in the interior.

*Proof.* Compact resolvent and the lower form bound give a minimizer of
the normalized Rayleigh quotient. Replacing it by its modulus does not
increase its gradient energy, so a nonnegative minimizer exists. Interior
elliptic regularity and the strong maximum principle make it strictly
positive there. The scalar heat kernel is positivity improving on a connected
domain with the selected Dirichlet or Neumann realization; a bounded real
potential preserves this through its positive Feynman--Kac weight.
For compact positive time the leading eigenspace of this positivity-improving
self-adjoint semigroup is one-dimensional, hence so is the ground eigenspace.
In the Dirichlet case the eigenfunction vanishes on the boundary.
Every nonempty interior open set contains a relatively compact ball on
which its continuous square has a positive minimum, giving positive mass
in that open set. This proves stationary positivity, not a dynamical
crossing rate. Compact resolvent and simplicity also give $E_1>E_0$ for
this fixed operator. $\square$
:::

:::{prf:definition} Barrier metric at a fixed energy
:label: def-e7-agmon-metric

Define $g_E=2(U-E)_+\widetilde G$ and
$d_E(x,A)=\inf_{\gamma:A\to x}\int\sqrt{2(U-E)_+}\,|\dot\gamma|_{\widetilde G}$.
The factor two matches the kinetic normalization $-\sigma^2\Delta/2$.
Distances may vanish within an allowed connected region.
:::

:::{prf:theorem} Exact weighted eigenfunction identity
:label: thm-e7-agmon-decay-bound

For an eigenfunction $(H_\sigma-E)u=0$ in the scalar realization and a
bounded smooth real weight $f$, put $v=e^{f/\sigma}u$. Then
$$
\frac{\sigma^2}{2}\int|\nabla v|^2
+\int\left(U-E-\frac12|\nabla f|^2\right)|v|^2=0.
$$
All integrals use $d\mu_{\widetilde G}$; the weight is taken constant in
the normal direction for the Neumann case. Compactly supported cutoffs
give the local form, with their explicit derivative terms retained.

*Proof.* Test the weak eigenvalue equation against $e^{2f/\sigma}u$ and
take real parts. The gradient product is
$$
\operatorname{Re}\langle\nabla u,\nabla(e^{2f/\sigma}u)\rangle
=|\nabla(e^{f/\sigma}u)|^2
-\sigma^{-2}|\nabla f|^2e^{2f/\sigma}|u|^2.
$$
Substitution proves the identity. For $0<\epsilon<1$ and a bounded Lipschitz approximation to
$f=(1-\epsilon)d_E(\cdot,A_E)$, its gradient satisfies
$|\nabla f|^2\le2(1-\epsilon)^2(U-E)_+$ almost everywhere.
Weak approximation preserves the resulting energy inequality. On the
forbidden region the remaining potential coefficient is at least
$(2\epsilon-\epsilon^2)(U-E)$; the allowed-region negative term controls
the weighted integral. For a region where $U-E\ge\delta>0$ and $f\ge a$,
$$
\int_{\mathrm{region}}|u|^2
\le\frac{e^{-2a/\sigma}}{(2\epsilon-\epsilon^2)\delta}
\int_{A_E}(E-U)|u|^2.
$$
This is a weighted stationary estimate, including its constants. No
dimension-independent $H^1\to L^\infty$ embedding or uniform pointwise
prefactor is used. $\square$
:::

:::{prf:corollary} Metric comparison at fixed potential and energy
:label: cor-e7-adversarial-suppression

For $g_1\succeq g_0$ and the same $U,E$, every path obeys
$\int\sqrt{2(U-E)_+}|\dot\gamma|_{g_1}
\ge\int\sqrt{2(U-E)_+}|\dot\gamma|_{g_0}$.
Taking infima gives $d_E^{g_1}\ge d_E^{g_0}$. This compares barrier
actions at the same energy. Changing a Hamiltonian generally changes its
ground energy as well, so one cannot substitute two different ground
energies into this fixed-energy inequality or deduce an ordering of
transition probabilities from two upper bounds. $\square$
:::

:::{prf:theorem} Feynman--Kac clock and normalization
:label: thm-e7-feynman-kac

Let $X_s$ have generator $\Delta_{\widetilde G}/2$, killed at a Dirichlet
boundary or reflected for the Neumann realization. Then
$$
(e^{-tH_\sigma/\sigma^2}\phi)(x)
=\mathbb E_x\left[e^{-\sigma^{-2}\int_0^tU(X_s)ds}\phi(X_t)\right],
$$
with the survival indicator in the killed case. To obtain the ground vector,
$$
e^{tE_0/\sigma^2}e^{-tH_\sigma/\sigma^2}\phi
\longrightarrow\langle u_0,\phi\rangle u_0
\quad\text{in }L^2.
$$
*Proof.* The diffusion generator and multiplication weight give
$\partial_tu=(\Delta/2-U/\sigma^2)u=-H_\sigma u/\sigma^2$
with the same initial and boundary data, which is the Feynman--Kac
semigroup. The spectral expansion multiplies each eigencomponent by
$e^{-t(E_n-E_0)/\sigma^2}$; dominated convergence leaves exactly its
ground projection. Divide by the nonzero overlap to recover $u_0$.
The original WFR generator is compared separately; Brownian motion here
is the process associated to this displayed scalar semigroup. $\square$
:::

:::{prf:corollary} Action-length inequality
:label: cor-e7-large-deviations

For any absolutely continuous path in the forbidden region, set
$a=|\dot\gamma|_{\widetilde G}$, $b=\sqrt{2(U-E)}$. Then
$a^2/2+U-E-ab=(a-b)^2/2\ge0$.
Integration gives
$\int(|\dot\gamma|^2/2+U-E)dt
\ge\int\sqrt{2(U-E)}|\dot\gamma|dt$.
Equality is attained on a parametrization with $a=b$ wherever that
parametrization is defined. This relates an action to barrier length.
An asymptotic probability additionally belongs to a specified stochastic
law and event; it is not furnished by this algebraic inequality.
:::



(sec-appendix-e-proof-of-corollary-varentropy-stability)=
## E.8 Proof of Corollary {prf:ref}`cor-varentropy-stability`

**Statement:** $V_H(z) = T_c^2 \frac{\partial H(\pi)}{\partial T_c}$.

**Hypothesis:** Let $\pi(a|z) = \frac{1}{Z} \exp\left(\frac{Q(z,a)}{T_c}\right)$ be the policy, where $Z = \sum_a \exp(Q/T_c)$ is the partition function. Let $\beta_{\text{ent}} = 1/T_c$ be the inverse {prf:ref}`def-cognitive-temperature`.

(proof-cor-varentropy-stability)=
:::{prf:proof}

**Step 1: Express Entropy in terms of $\beta_{\text{ent}}$.**
The entropy of the policy is:

$$
H(\pi) = -\sum_a \pi(a) \ln \pi(a).

$$
Substituting $\ln \pi(a) = \beta_{\text{ent}} Q(a) - \ln Z$:

$$
H(\pi) = -\sum_a \pi(a) [\beta_{\text{ent}} Q(a) - \ln Z] = \ln Z - \beta_{\text{ent}} \mathbb{E}_\pi[Q].

$$
**Step 2: Derivative of Entropy w.r.t. $\beta_{\text{ent}}$.**
Differentiating with respect to $\beta_{\text{ent}}$:

$$
\frac{\partial H}{\partial \beta_{\text{ent}}} = \frac{\partial \ln Z}{\partial \beta_{\text{ent}}} - \mathbb{E}_\pi[Q] - \beta_{\text{ent}} \frac{\partial \mathbb{E}_\pi[Q]}{\partial \beta_{\text{ent}}}.

$$
Using the identity $\frac{\partial \ln Z}{\partial \beta_{\text{ent}}} = \mathbb{E}_\pi[Q]$:

$$
\frac{\partial H}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \frac{\partial \mathbb{E}_\pi[Q]}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \mathrm{Var}_\pi(Q),

$$
where we used $\frac{\partial \mathbb{E}[Q]}{\partial \beta_{\text{ent}}} = \mathrm{Var}(Q)$ (standard fluctuation-response relation).

**Step 3: Relate $\mathrm{Var}(Q)$ to Varentropy.**
Recall $\mathcal{I}(a) = -\ln \pi(a) = -\beta_{\text{ent}} Q(a) + \ln Z$. The variance of the surprisal is:

$$
V_H(\pi) = \mathrm{Var}(\mathcal{I}) = \mathrm{Var}(-\beta_{\text{ent}} Q + \ln Z) = \beta_{\text{ent}}^2 \mathrm{Var}(Q).

$$
**Step 4: Change of variables to $T_c$.**
We have $V_H = \beta_{\text{ent}}^2 \mathrm{Var}(Q)$ and $\frac{\partial H}{\partial \beta_{\text{ent}}} = -\beta_{\text{ent}} \mathrm{Var}(Q)$.
Therefore $V_H = -\beta_{\text{ent}} \frac{\partial H}{\partial \beta_{\text{ent}}}$.

Using the chain rule $\frac{\partial}{\partial T_c} = -\frac{1}{T_c^2} \frac{\partial}{\partial \beta_{\text{ent}}}$:

$$
\frac{\partial H}{\partial T_c} = -\frac{1}{T_c^2} \frac{\partial H}{\partial \beta_{\text{ent}}} = \frac{1}{T_c^2} \cdot \beta_{\text{ent}} \mathrm{Var}(Q) = \frac{V_H}{T_c^2 \cdot \beta_{\text{ent}}} = \frac{V_H}{T_c}.

$$
**Final Result:** Rearranging yields:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c} = \beta_{\text{ent}}^2 \mathrm{Var}(Q) = C_v.

$$
This proves that Varentropy equals the heat capacity and measures the sensitivity of the entropy to temperature fluctuations. $\square$

:::



(sec-appendix-e-proof-of-corollary-bimodal-instability)=
## E.9 Proof of Corollary {prf:ref}`cor-bimodal-instability`

**Statement:** For a bimodal policy on a value ridge, $V_H$ is significant, distinguishing it from uniform noise.

**Hypothesis:** Let $\pi$ be a mixture of two dominant modes with values $Q_1, Q_2$ and a background of $N-2$ negligible modes.

(proof-cor-bimodal-instability)=
:::{prf:proof}

**Step 1: Variance of Surprisal Form.**

$$
V_H = \mathbb{E}[\mathcal{I}^2] - (\mathbb{E}[\mathcal{I}])^2.

$$
Since $\mathcal{I} = -\beta Q + \ln Z$, we have $V_H = \beta^2 \mathrm{Var}(Q)$.

**Step 2: Two-Point Statistics.**
Consider two actions $a_1, a_2$ with probabilities $p, 1-p$. The variance of a Bernoulli variable taking values $Q_1, Q_2$ is:

$$
\mathrm{Var}(Q) = p(1-p)(Q_1 - Q_2)^2.

$$
Thus:

$$
V_H = \beta^2 p(1-p) (\Delta Q)^2 = p(1-p) \left( \frac{\Delta Q}{T_c} \right)^2.

$$
For equally weighted modes ($p = 1/2$), this simplifies to:

$$
V_H = \frac{1}{4} \left( \frac{\Delta Q}{T_c} \right)^2.

$$
**Step 3: Interpretation of $\Delta Q$.**
$\Delta Q$ is the value gap between the two modes.

- **Perfect Symmetry (The Ridge):** If $Q_1 = Q_2$ exactly, then $\Delta Q = 0 \implies V_H = 0$.
- **Structural Instability:** When the agent is *slightly* off-center or when sampling includes the *tails*, the effective $\Delta Q > 0$.

**Step 4: Distinguishing Structure from Noise.**
For a distribution with structure (peaks and valleys), $\mathrm{Var}(Q) > 0$. For a flat distribution (noise), $\mathrm{Var}(Q) = 0$.

Specifically, on a ridge, the agent samples $a_{\text{left}}$ and $a_{\text{right}}$ (high $Q$) but also transitively samples the separating region (lower $Q$) during exploration. The variance of $Q$ along the trajectory corresponds to $V_H$:

$$
V_H \propto (\Delta Q_{\text{peak-valley}})^2.

$$
This proves that $V_H$ detects the topological feature (the valley) that distinguishes a fork from a flat plane. $\square$

:::



(sec-appendix-e-proof-of-corollary-varentropy-brake)=
## E.10 Proof of Corollary {prf:ref}`cor-varentropy-brake`

**Statement:** To maintain stability, the cooling rate must satisfy $|\dot{T}_c| \ll T_c / \sqrt{V_H}$.

**Hypothesis:** We require the probability distribution $\pi_t$ to remain close to the equilibrium Boltzmann distribution $\pi^*_{T_c(t)}$ during annealing. This is the **Adiabatic Condition**.

(proof-cor-varentropy-brake)=
:::{prf:proof}

**Step 1: Thermodynamic Speed.**
The rate of change of the policy distribution with respect to temperature is measured by the Fisher Information metric $g_{TT}$ on the statistical manifold parameterized by $T_c$:

$$
g_{TT} = \mathbb{E}\left[ \left( \frac{\partial \ln \pi}{\partial T_c} \right)^2 \right].

$$
**Step 2: Relate Fisher Metric to Varentropy.**
Recall $\ln \pi = \frac{Q}{T_c} - \ln Z$. Then:

$$
\frac{\partial \ln \pi}{\partial T_c} = -\frac{Q}{T_c^2} + \frac{\mathbb{E}[Q]}{T_c^2} = -\frac{1}{T_c^2}(Q - \mathbb{E}[Q]).

$$
Substituting into the Fisher definition:

$$
g_{TT} = \frac{1}{T_c^4} \mathbb{E}\left[ (Q - \mathbb{E}[Q])^2 \right] = \frac{\mathrm{Var}(Q)}{T_c^4}.

$$
Using $V_H = \frac{\mathrm{Var}(Q)}{T_c^2}$ (from Proof E.8):

$$
g_{TT} = \frac{V_H}{T_c^2}.

$$
**Step 3: Thermodynamic Length.**
The "distance" traversed in probability space for a small temperature change $dT_c$ is $ds^2 = g_{TT} dT_c^2$:

$$
ds = \sqrt{g_{TT}} |dT_c| = \frac{\sqrt{V_H}}{T_c} |dT_c|.

$$
**Step 4: Adiabatic Condition.**
For the system to relax to equilibrium (stay in the basin of attraction), the speed of change in distribution space must be bounded:

$$
\left| \frac{ds}{dt} \right| \leq C \cdot \tau_{\text{relax}}^{-1}.

$$
Substituting $ds/dt$:

$$
\frac{\sqrt{V_H}}{T_c} \left| \frac{dT_c}{dt} \right| \leq C.

$$
Solving for the cooling rate:

$$
\left| \frac{dT_c}{dt} \right| \leq C \frac{T_c}{\sqrt{V_H}}.

$$
**Conclusion:** When Varentropy $V_H$ is large (phase transition/critical point), the permissible cooling rate goes to zero. The Governor must apply the "Varentropy Brake" to prevent quenching the system into a suboptimal metastable state. $\square$

:::



(sec-appendix-e-proof-of-corollary-epistemic-curiosity-filter)=
## E.11 Proof of Corollary {prf:ref}`cor-epistemic-curiosity-filter`

**Statement:** $\nabla \Psi_{\text{causal}} \propto \nabla \mathbb{E}_{z'} [ V_H[P(\theta_W | z, a, z')] ]$.

**Hypothesis:** We define $\Psi_{\text{causal}}$ as the Expected Information Gain (EIG) about model parameters $\theta$ given a transition $(z, a) \to z'$.

(proof-cor-epistemic-curiosity-filter)=
:::{prf:proof}

**Step 1: Definition of EIG.**

$$
\text{EIG}(z, a) = I(\theta; z' | z, a) = H(z' | z, a) - \mathbb{E}_{\theta} [ H(z' | z, a, \theta) ].

$$
This is the **Total Predictive Entropy** minus the **Expected Aleatoric Entropy**.

**Step 2: Decomposition of Uncertainty.**
For the "noisy TV" case (outcomes are stochastic noise independent of $\theta$):

$$
H(z' | z, a, \theta) \approx H(z' | z, a) \implies \text{EIG} \approx 0.

$$
**Step 3: Varentropy as Structure Detector.**
The varentropy $V_H(z' | z, a)$ measures the variance of log-probabilities.

- **Uniform noise:** $V_H^{\text{noise}} \to 0$ (all outcomes equally likely).
- **Structured uncertainty:** $V_H^{\text{structured}} > 0$ (some outcomes much more likely).

**Step 4: Connection to Multimodality.**
If the model is uncertain about structure ($\theta$), the predictive distribution $p(z')$ is a mixture of distinct hypotheses $p(z'|\theta_1), p(z'|\theta_2)$. As established in Proof E.9, a mixture of distinct modes has high Varentropy compared to a broad unimodal distribution (noise).

**Step 5: Operational Equivalence.**
Thus, maximizing EIG is functionally equivalent to maximizing the **Varentropy of the expected outcome**, provided the aleatoric noise floor is constant:

$$
\nabla \Psi_{\text{causal}} \propto \nabla \mathrm{Var}_{z' \sim p(z'|z,a)} [ -\ln p(z'|z,a) ].

$$
**Conclusion:** The agent should seek states where the World Model's prediction has high Varentropy (conflicting hypotheses), as these offer the maximum potential for falsification (reduction of parameter variance). $\square$

:::



(proof-hjb-klein-gordon)=
(proof-madelung-transform)=
(proof-markov-restoration)=
(proof-nash-standing-wave)=
(proof-game-tensor-derivation)=
(proof-bianchi-identity)=
(proof-higgs-mechanism)=
(proof-nash-ground-state)=
(sec-references)=

## E.12 Bellman generator and scalar wave variation

:::{prf:proof}

For the diffusion and discount already used in
{prf:ref}`thm-the-hjb-helmholtz-correspondence`, write
$\mathcal L=b\cdot\nabla+T_c\Delta_G$ and $\gamma_h=e^{-\lambda h}$.
The smooth Bellman equation has continuous-time form
$$
\partial_tV+\mathcal LV-\lambda V+r=0.
$$
In its stationary zero-drift sector,
$(-\Delta_G+\lambda/T_c)V=r/T_c$; denote this screening coefficient by
$\kappa_B^2=\lambda/T_c$. The scalar field action used in this chapter
instead defines the wave operator
$$
\Box_g=-|g|^{-1/2}\partial_\mu(|g|^{1/2}g^{\mu\nu}\partial_\nu),
\qquad(\Box_g+\kappa^2)V=\rho_r.
$$
The stationary operators coincide under the coefficient identification
$\kappa^2=\kappa_B^2$ and the same sources and boundary realization.

*Proof.* Generator consistency gives
$\mathbb E[V(Z_h,t+h)]=V+h(\partial_t+\mathcal L)V+o(h)$.
Insert this and $e^{-\lambda h}=1-\lambda h+o(h)$ into
$V=rh+e^{-\lambda h}\mathbb E[V(Z_h,t+h)]$, cancel $V$, and divide by $h$.
The $\partial_t^2V$ Taylor term has coefficient $h/2$ after division and
vanishes. Finite signal speed does not change that coefficient.
For the wave model vary
$$
S[V]=\int\left[-\tfrac12g^{\mu\nu}\partial_\mu V\partial_\nu V
-\tfrac12\kappa^2V^2+\rho_rV\right]\sqrt{|g|}\,dx.
$$
Integration by parts against a compactly supported variation $\eta$ gives
$\delta S=\int\eta[-\Box_gV-\kappa^2V+\rho_r]\sqrt{|g|}\,dx$.
Stationarity proves the field equation. For a fixed product metric
$g=\operatorname{diag}(-c^2,G)$, $\Box_g=c^{-2}\partial_t^2-\Delta_G$.
These are explicit equations for two defined evolutions; equality of their
stationary operators is the comparison established here. $\square$
:::

## E.13 Polar WFR calculation with exact signs

:::{prf:proof}

On a smooth positive-density chart with the fixed spatial metric of the
WFR equations, put $R=\sqrt\rho$, $p=dV-B$, $v=G^{-1}p$,
$D=\nabla-iB/\sigma$, and $Q_B=-\sigma^2\Delta_GR/(2R)$.
For the equations already stated,
$\partial_s\rho+\operatorname{div}_G(\rho v)=r\rho$ and
$\partial_sV+|p|_G^2/2+\Phi_{\mathrm{eff}}=0$, the exact amplitude equation is

$$
i\sigma\partial_s\psi=
\left[-\tfrac{\sigma^2}{2}\Delta_B+\Phi_{\mathrm{eff}}-Q_B
+\tfrac{i\sigma}{2}r\right]\psi,
\qquad\psi=Re^{iV/\sigma}.
$$

*Proof.* The product rule gives

$$
\frac{\Delta_B\psi}{\psi}=\frac{\Delta_GR}{R}
-\frac{|p|_G^2}{\sigma^2}
+\frac{i}{\sigma}\left(2\frac{\langle dR,p\rangle_G}{R}
+\operatorname{div}_Gv\right).
$$

Thus the kinetic real part is $Q_B+|p|_G^2/2$.
The $-Q_B$ term cancels it to the given classical HJB expression.
The imaginary part is
$-\sigma\operatorname{div}_G(\rho v)/(2\rho)+\sigma r/2$,
equal to $\sigma\partial_s\rho/(2\rho)$ by continuity.
The time derivative is
$i\sigma\partial_s\psi/\psi=i\sigma\partial_s\rho/(2\rho)-\partial_sV$,
so both parts agree. Reading these two parts backwards proves the local
equivalence. At zeros use the density/current equations without division by
$R$; a global phase additionally retains its existing circulation data.

The compensating $Q_B$ depends on $|\psi|$, making this amplitude equation
nonlinear. Omitting the compensation gives the distinct linear Schrödinger
model with a $+Q_B$ term in its Hamilton--Jacobi equation. A curl-modified
mobility must be substituted into its own continuity equation; the above
Laplacian yields precisely the canonical velocity $G^{-1}p$.
For time-dependent volume density $w_s$, conservation reads
$\partial_s(w_s\rho)+\partial_i(w_s\rho v^i)=w_sr\rho$;
the corresponding amplitude equation acquires
$-i\sigma\partial_s\log w_s/2$. $\square$
:::

## E.14 Complete-history conditional kernels

:::{prf:proof}

On the standard measurable path spaces of the specified process, the complete
history state $(t,\mathsf H_t)$ is Markov with its conditional extension kernel.
Neither a positive delay alone nor the reward occupation screen determines
whether a smaller state is Markov.

*Proof.* The sigma-algebra generated by $(t,\mathsf H_t)$ contains the entire
modeled history through $t$. Let $K_{t,u}(h,\cdot)$ be the regular conditional
law of the extended history through $u$ given $\mathsf H_t=h$.
For a bounded history functional $F$,
$$
\mathbb E[F(\mathsf H_u)\mid\sigma(\mathsf H_s:s\le t)]
=\mathbb E[F(\mathsf H_u)\mid\mathsf H_t]
=K_{t,u}F(\mathsf H_t).
$$
The tower property gives $K_{t,u}=K_{t,v}K_{v,u}$ on realized histories.
Including $t$ in the state accounts for time-inhomogeneous coefficients.
This proves the representation without identifying a finite compression.
For the occupation screen take $\alpha=0$: every history has screen zero,
while histories with the same current position can have different
$z_{t-\tau}$ and therefore different delayed drifts. Conversely a delayed
signal with zero coupling leaves a Markov local process Markov. These examples
prove both limitations of the smaller-state claims. $\square$
:::

## E.15 Time averages and unilateral variations

:::{prf:proof}

For a bounded differentiable density trajectory,
$T^{-1}\int_0^T\partial_t\rho\,dt=(\rho(T)-\rho(0))/T\to0$.
This holds for many nonequilibrium trajectories and does not test unilateral
payoff improvements. Moreover $\langle\rho v\rangle$ need not vanish when
$\langle v\rangle=0$: on a periodic clock take $v=\sin t$ and
$\rho=1+\epsilon\sin t$, $0<\epsilon<1$, giving
$\langle\rho v\rangle=\epsilon/2$.
Standing-wave expansions describe solutions of the defined wave operator;
Nash conditions are the payoff inequalities in
{prf:ref}`thm-nash-equilibrium-as-geometric-stasis`. $\square$
:::

## E.16 Strategic Hessian and response composition

:::{prf:proof}

Use the smooth local best-response branch and Strategic Jacobian already
specified in {prf:ref}`def-strategic-jacobian`. With the intrinsic connection
on agent $j$'s manifold define the covariant tensor
$$
H^{(i)}_{jj,mn}=\nabla^{(j)}_m\nabla^{(j)}_nV^{(i)},\qquad
\mathcal G^{(i)}_{ij,ab}=\mathcal J_{ji}^{m}{}_{a}
H^{(i)}_{jj,mn}\mathcal J_{ji}^{n}{}_{b}.
$$
No additional lowering of the Hessian indices is applied. The strategic
metric prescription is $\widetilde G^{(i)}=G^{(i)}+h^{(i)}$ with
$h^{(i)}=\sum_{j\ne i}\beta_{ij}\mathcal G^{(i)}_{ij}$.
Its positive-definite domain is checked using the spectral margin already
specified in {prf:ref}`def-e7-strategic-metric`.
The curvature equation {prf:ref}`thm-capacity-constrained-metric-law` remains
a separate differential identity; this algebraic prescription is not its solution.

For a $C^2$ response $y=b(x)$, direct differentiation gives
$$
\partial_{ab}V(x,b(x))=V_{ab}+V_{am}b^m_b+V_{bm}b^m_a
+V_{mn}b^m_ab^n_b+V_m\partial_{ab}b^m.
$$
Thus the pulled-back $H_{jj}$ is one contribution, not the full Hessian of
the composed value. The final term vanishes at a stationary point in $y$.
For the positive metric $\widetilde G=G+h$, subtraction of the two
metric-compatible torsion-free connections gives the exact identity
$$
\widetilde\Gamma^a_{bc}-\Gamma^a_{bc}
=\tfrac12\widetilde G^{ad}
(\nabla_bh_{dc}+\nabla_ch_{db}-\nabla_dh_{bc}).
$$
Replacing $\widetilde G^{-1}$ by $G^{-1}$ gives its first-order expansion,
with a remainder controlled by the inverse-metric identity
$\widetilde G^{-1}-G^{-1}=-G^{-1}h\widetilde G^{-1}$.
:::

:::{prf:definition} Strategic Jacobian on the established response branch
:label: def-strategic-jacobian

On the smooth, nondegenerate local best-response branch specified in the
original strategic construction, differentiate
$\nabla_jV_j(b_j(z_{-j}),z_{-j})=0$. The chain rule gives
$H_{jj}^{(j)}\mathcal J_{ji}+H_{ji}^{(j)}=0$, hence
$\mathcal J_{ji}=-(H_{jj}^{(j)})^{-1}H_{ji}^{(j)}$.
This identifies the branch derivative on the domain where the declared
inverse exists. It is not a global selection of a multivalued best-response
correspondence. Its pullback action on covariant Hessians is exactly that
used in {prf:ref}`def-the-game-tensor`.
:::

## E.17 Curvature and Bianchi identity

:::{prf:proof}

For the defined connection, the adjoint covariant derivative is
$\mathcal D_\rho F_{\mu\nu}=\partial_\rho F_{\mu\nu}-ig[A_\rho,F_{\mu\nu}]$.
On a test section $u$, expansion gives
$[D_\rho,F_{\mu\nu}]u=(\mathcal D_\rho F_{\mu\nu})u$.
Insert $[D_\mu,D_\nu]=-igF_{\mu\nu}$ into
$[D_\rho,[D_\mu,D_\nu]]+\mathrm{cyclic}=0$. Dividing by $-ig$ gives
$$
\mathcal D_\rho F_{\mu\nu}+\mathcal D_\mu F_{\nu\rho}
+\mathcal D_\nu F_{\rho\mu}=0.
$$
For $g=0$ the same identity is $d(dA)=0$. The geometric Christoffel
terms cancel under cyclic antisymmetrization for the torsion-free connection.
The identity holds in each smooth gauge chart and is preserved under
transition functions by conjugation; nontrivial bundle topology does not
violate it. $\square$
:::

## E.18 Vacuum expansion in the declared representation

:::{prf:proof}

For the stable quartic potential already specified, $\lambda>0$ and
$\mu^2<0$, write $\Phi_0=(v/\sqrt2)n$, $n^\dagger n=1$ and
$v^2=-\mu^2/\lambda$. The radial mass is $m_h^2=2\lambda v^2$.
The gauge quadratic term is $-\tfrac12A_\mu^a(M^2)_{ab}A^{\mu b}$ with
$$
(M^2)_{ab}=g^2\Phi_0^\dagger\{T_a,T_b\}\Phi_0.
$$
*Proof.* Put $q=\Phi^\dagger\Phi$. The minimum solves
$\mu^2+2\lambda q=0$. Substitute $q=(v+h)^2/2$; the coefficient of
$h^2$ in $U$ is $\lambda v^2=m_h^2/2$.
For a constant vacuum, $D_\mu\Phi_0=-igA_\mu^aT_a\Phi_0$;
the symmetric product of the commuting coefficients $A^aA^b$ gives the
displayed anticommutator. For any real $u^a$,
$u^a(M^2)_{ab}u^b=2g^2\|(u^aT_a)\Phi_0\|^2\ge0$.
Its kernel is the stabilizer Lie algebra of the vacuum. For a single
$SU(2)$ doublet with $T_a=\sigma_a/2$, this evaluates to
$(M^2)_{ab}=g^2v^2\delta_{ab}/4$. It is this representation that gives
$m_A=gv/2$. Other declared representations are evaluated by the same
matrix formula. If $\lambda\le0$, the stated stable quartic expansion does
not apply; the potential itself reveals the failure. $\square$
:::

## E.19 Spectral minimization and the Nash comparison

:::{prf:proof}

The Rayleigh quotient of the scalar Hamiltonian minimizes its single joint
energy. The Nash test instead compares each agent's own payoff under a
unilateral change. Their equality must be checked by differentiating the
actual objectives and by evaluating their global inequalities.

*Proof by explicit comparison.* Let
$V_1(x,y)=-(x-y)^2$ and $V_2(x,y)=-(y-1)^2-Kx$ with $K>0$.
Both are strictly concave in their own coordinate; their best responses are
$x=y$ and $y=1$, hence the unique Nash profile is $(1,1)$.
The sum of costs is $U=(x-y)^2+(y-1)^2+Kx$.
Its $x$ derivative at $(1,1)$ is $K$, so Nash is not a stationary point of
this joint energy. This example meets the smooth nondegenerate best-response
structure of the Strategic Jacobian. Thus that machinery cannot justify
the former general identification of Nash with a joint ground state.
The ground-state and variational calculations retain their meaning for
the scalar operator actually defined. $\square$
:::



## References

```{bibliography}
```
