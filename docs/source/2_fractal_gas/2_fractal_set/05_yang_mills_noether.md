# Discrete Yang–Mills Actions, Noether Identities, and Quantum Reconstruction

## TLDR

:::{div} feynman-prose
A link matrix tells you how to transport an internal vector between vertices.
For the reconstructed doublet frames, this transport is a change of coordinates:
each link compares the frames at its two endpoints. Go around a closed loop and
every frame cancels, including the starting frame. The resulting holonomy is
exactly the identity. These are genuine $SU(2)$ matrices, with a precisely
specified flat connection. Independent Wilson links permit nonidentity loop
holonomies and define a different field measure.

Chapter 04 gives an exact way to compare such measures: average the complete
path likelihood over records with the same descriptor. This yields the
descriptor density; its support determines which field integrals weighting can
represent. The same chapter constructs CAR operators and the replica generator.
The full-gradient LSI controls centered empirical fluctuations at the
$\sqrt N$ scale for the observables it covers. We use these established results
below, keeping track of the observable, law, and generator in each calculation.
Reflection positivity then concerns the time-dependent correlations of that
specified law.
:::

(sec-ym-intro)=
## 1. Objects and analytic inputs

Write $d$ for spatial dimension and $D=d+1$ for spacetime dimension. A finite
Fractal Set supplies vertices, recorded attributes, and several kinds of edges.
An oriented two-complex additionally specifies closed face boundaries; a face is
not determined by an unordered collection of nearby vertices. The constructions
below use this incidence data and a declared representation of a compact group.
The recorded algorithm and the auxiliary field model retain their respective
transition laws.

:::{prf:remark} Direct observables and the scope of field actions
:label: rem-ym-direct-observable-route

The primary direct observable formulation is
{ref}`sec-sm-direct-observables`. It represents the recorded color and
companion data through Hermitian contractions, determinant contractions,
and composite triangle observables. The exact orbit and measure
isomorphisms are {prf:ref}`thm-sm-direct-orbit-isomorphism` and
{prf:ref}`thm-sm-direct-measure-isomorphism`; their application of the
Fractal Set reconstruction and LSI is
{prf:ref}`thm-sm-direct-existing-machinery`.

The finite path likelihood in {prf:ref}`thm-action-from-path-integral`
uses the full particle kernel and supplies its recorded path measure.
The Dirac matter action and Wilson link integral below are additional
field constructions. The direct formulation uses the numerical observable
algebra without them. All uses of the transfer and reconstruction results
in this chapter retain the same law and operators as their proofs.
:::

The distinction between a QSD and an invariant probability is fixed throughout.
For a killed semigroup $Q_t$, a QSD $\nu_N$ satisfies
$\nu_NQ_t=e^{-\alpha_Nt}\nu_N$. A conservative semigroup $P_t$ has invariant
law $\pi_N$ when $\pi_NP_t=\pi_N$. The two statements enter different proofs.

The following results provide the analytic inputs used here:

- {prf:ref}`thm-main-convergence` gives QSD convergence under its complete
  killed-kernel hypotheses; {prf:ref}`thm-convergence-conservative-harris`
  treats conservative invariant laws.
- {prf:ref}`thm-chaos-finite-time-consistency` and
  {prf:ref}`thm-thermodynamic-limit` give empirical and stationary limits under
  the respective consistency, tightness, and uniqueness hypotheses.
- {prf:ref}`thm-mixing-variance-corrected` controls bounded empirical observables
  through total relative entropy. {prf:ref}`cor-n-uniform-lsi` proves a
  full-gradient LSI for four specified families of joint laws, including
  contractive additive-noise invariant flows.
- {prf:ref}`thm-langevin-baoab-discretization-error` and
  {prf:ref}`thm-full-system-discretization-error` retain the local consistency
  and stability conditions needed for time discretization.
- {prf:ref}`cor-continuum-consistency-conditional`,
  {prf:ref}`assm-cst-continuum-geometry`, and
  {prf:ref}`assm-cst-episode-sampling` specify geometry and sampling on the
  same reconstructed space. Operator consistency alone supplies neither a
  Lorentzian embedding nor convergence of distances.

For later reference, the full-gradient convention is

$$
\operatorname{Ent}_{\pi_N}(f^2)
 \le 2C_*\int\sum_i(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2)\,d\pi_N.
\tag{YM.1}
$$

The constant is uniform in $N$ when the chosen joint law meets one of the
criteria in {prf:ref}`cor-n-uniform-lsi`. Discrete alive/dead strata require the
additional entropy term in {prf:ref}`prop-kl-status-entropy`. A velocity-only
form cannot replace the right-hand side of (YM.1) for spatial observables.

(sec-ym-symmetry)=
## 2. Internal frames and dressed states

### 2.1. Groups acting on the finite data

:::{prf:definition} Local frame group and relabeling group
:label: def-hybrid-gauge-structure-ym

Let $\mathcal V$ be a finite vertex set and choose
$G=SU(2)\times U(1)$ with a specified unitary representation on each fiber.
The local frame group is $G^{\mathcal V}$. If a group $\Gamma$ permutes the
vertices and transports every edge, face, and attribute with them, it acts on
$G^{\mathcal V}$ by permuting its factors. The combined action is
$G^{\mathcal V}\rtimes\Gamma$.

Walker relabeling by $S_N$ is an action on complete records. It is a symmetry of
a probability law when the complete transition kernel and initial law are
relabeling equivariant, as in {prf:ref}`thm-qsd-exchangeability`. The choice of
internal $G$ is the representation construction of {doc}`04_standard_model`.
:::

:::{div} feynman-prose
Renaming walkers and rotating an internal two-component vector are different
operations. A permutation acts on the records. A frame rotation acts on the
coordinates used to describe a field on those records. Keeping the two actions
separate makes the covariance calculation almost mechanical.
:::

:::{prf:definition} Normalized companion amplitudes and doublets
:label: def-dressed-walker-state-ym

For an allowed companion set $A_i$ and probabilities
$P_i(k)\ge0$, $\sum_{k\in A_i}P_i(k)=1$, choose phases $\theta_{ik}$ and set

$$
 |\psi_i\rangle=\sum_{k\in A_i}\sqrt{P_i(k)}e^{i\theta_{ik}}|k\rangle
 \in\mathbb C^N.
$$

Excluded companions have zero coordinates. For a chosen pair $(i,j)$ and
$p\in[0,1]$, its normalized doublet is

$$
 |\Psi_{ij}\rangle=
 \sqrt p\,|\uparrow\rangle\otimes|\psi_i\rangle+
 \sqrt{1-p}\,|\downarrow\rangle\otimes|\psi_j\rangle
 \in\mathbb C^2\otimes\mathbb C^N.
$$

The squared norm is $p+(1-p)=1$, since the two internal basis vectors are
orthogonal. The symmetric choice is $p=1/2$.
:::

:::{prf:definition} Doublet transformation
:label: def-su2-transformation-ym

For

$$
 U=\begin{pmatrix}a&b\\-\overline b&\overline a\end{pmatrix},
 \qquad |a|^2+|b|^2=1,
$$

write $\Psi=|\uparrow\rangle\otimes u+|\downarrow\rangle\otimes v$.
Then $(U\otimes I)\Psi$ has components
$(au+bv,-\overline b u+\overline a v)$. A unitary transformation can produce
components whose squared norms differ from the original mixing weights.
:::

:::{prf:definition} Fitness operators
:label: def-fitness-operator-ym

For real recorded values $V(i\mid k)$ define

$$
 \widehat V_i=\sum_k V(i\mid k)|k\rangle\langle k|,
 \qquad
 \langle\psi_i,\widehat V_i\psi_i\rangle
 =\sum_kP_i(k)V(i\mid k).
$$

A doublet operator can be
$M=\operatorname{diag}(\widehat V_i,\widehat V_j)$ or a score operator
$\operatorname{diag}(\widehat V_i,-\widehat V_j)$. Its transformation law must
be specified together with the doublet.
:::

:::{prf:proposition} Invariant norms and invariant quadratic observables
:label: prop-su2-invariance-ym

Every doublet norm is $SU(2)$ invariant. A fixed Hermitian operator $A$ on
$\mathbb C^2\otimes K$ has invariant quadratic form under all $U\otimes I$
if and only if $A=I_2\otimes B$ for a Hermitian operator $B$ on $K$.
For any $A$, simultaneous transformation
$\Psi'=U\Psi$, $A'=UAU^\dagger$ preserves its expectation.
:::

:::{prf:proof}
Unitarity proves the norm identity and
$\langle U\Psi,UAU^\dagger U\Psi\rangle=\langle\Psi,A\Psi\rangle$.
For a fixed operator, equality of all quadratic forms implies
$U^\dagger AU=A$ by polarization. Commutation with
$\operatorname{diag}(e^{it},e^{-it})$ eliminates the off-diagonal blocks.
Commutation with $\left(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right)$
then equates the diagonal blocks. Conversely $I_2\otimes B$ commutes with
every $U\otimes I$.

Thus fixed unequal fitness blocks select an internal direction. A sum of
cloning probabilities has this symmetry only if its actual formula has the
same invariance; normalization of a separately defined doublet does not prove
that identity.
:::

### 2.2. Recorded quantities and field representations

The role masks in {prf:ref}`def-sm-walker-role-partition` remain operational
partitions of a frame. Their mean chirality obeys
$\overline\chi=2f_L-|A|/N$, and the same-frame count
$N_{\Delta\to R}$ vanishes by
{prf:ref}`prop-sm-walker-role-partition`. Spinor projectors act on a chosen
Clifford representation. A scalar phase returned by the electroweak analysis
is a $U(1)$-valued observable; an $SU(2)$ comparison matrix additionally
requires a declared lift and the inverse-edge convention below. These
identifications are detailed in {prf:ref}`thm-sm-ew-operator-layers`.

(sec-ym-first-principles)=
## 3. Path likelihoods, field actions, and connections

### 3.1. The action of a specified transition law

:::{prf:theorem} Finite path likelihood and Gaussian quadratic action
:label: thm-action-from-path-integral

Let $P_h$ be the complete one-step Markov kernel, including companion choices,
cloning, status changes, and kinetic updates. Suppose
$P_h(y,dz)=p_h(y,z)R_h(y,dz)$ relative to a reference kernel, and similarly
$d\mu_0=r_0\,d\lambda_0$. Relative to the reference path measure, the
$K$-step path density is $e^{-\mathcal S_h}$ with

$$
 \mathcal S_h(y_0,\ldots,y_K)
 =-\log r_0(y_0)-\sum_{k=0}^{K-1}\log p_h(y_k,y_{k+1}).
\tag{YM.2}
$$

In the particular Euler Gaussian model
$Y_{k+1}=Y_k+h b(Y_k)+\sigma\sqrt h\,\xi_k$ on $\mathbb R^m$,
$\xi_k\sim N(0,I)$, the negative log density relative to Lebesgue increments
is

$$
 \sum_{k=0}^{K-1}
 \left[\frac{|Y_{k+1}-Y_k-hb(Y_k)|^2}{2\sigma^2h}
       +\frac m2\log(2\pi\sigma^2h)\right].
\tag{YM.3}
$$

For a fixed $C^1$ path $y:[0,T]\to\mathbb R^m$, continuous $b$, and
$Kh=T$, the first sum in (YM.3) evaluated at $y(kh)$ converges to
$(2\sigma^2)^{-1}\int_0^T|\dot y-b(y)|^2dt$.
:::

:::{prf:proof}
Conditional multiplication of the one-step Radon–Nikodym derivatives gives
(YM.2). The normal density gives (YM.3), including its normalization. On a
fixed $C^1$ path, the difference quotients converge uniformly to $\dot y$;
continuity of $b$ on the compact image of that path makes the quadratic sum a
Riemann sum.

This last calculation concerns smooth comparison paths. Typical diffusion
paths have nonzero quadratic variation. For example, the expected quadratic
sum in (YM.3) along the Gaussian process is $mK/2$, which diverges as
$h\downarrow0$. The smooth-path formula is therefore not an ordinary density
on differentiable sample paths. Singular deterministic substeps and cloning
atoms must retain their appropriate kernel reference in (YM.2).
:::

:::{prf:lemma} Reversible diffusion and its ground-state transform
:label: lem-ym-ground-state-transform

Let $\pi(dx)=Z^{-1}e^{-U(x)/T}dx$ on $\mathbb R^m$, $T>0$, with $Z<\infty$,
$U\in C^2$, and a conservative reversible realization of
$L=T\Delta-\nabla U\cdot\nabla$. On smooth compactly supported functions the
unitary map $\mathcal U f=\pi^{1/2}f$ gives

$$
 H=-\mathcal U L\mathcal U^{-1}
 =-T\Delta+\frac{|\nabla U|^2}{4T}-\frac{\Delta U}{2}.
\tag{YM.4}
$$

Its quadratic-form realization is nonnegative, and its normalized ground
state is $\pi^{1/2}$.
:::

:::{prf:proof}
Integration by parts gives
$\langle f,-Lf\rangle_\pi=T\int|\nabla f|^2d\pi$.
Substitute $f=e^{U/(2T)}\psi$ into $Lf$ and differentiate twice. The first
order derivatives cancel, leaving (YM.4). The closed nonnegative form
transports unitarily to the form of $H$. Since $L1=0$, its image
$\mathcal U1=\pi^{1/2}$ has energy zero.
:::

A Feynman–Kac weight $e^{-\int V(Y_t)dt}$ defines a killed or tilted process
for a specified potential $V$. Pairwise cloning reproduces that law only when
its transition probabilities establish the required identification. The
kinetic, generally nonreversible Fractal Gas requires its own full kernel in
(YM.2). Formula (YM.4) also shows why a stochastic potential cannot simply be
copied into a Schrödinger action: derivatives of the potential enter the
transformation.

### 3.2. Chosen matter fields

:::{prf:definition} Matter action and mass matrix
:label: def-matter-lagrangian-ym

On a supplied Lorentzian spin geometry, choose a Dirac field with an internal
doublet and write, in units $\hbar=c=1$,

$$
 \mathcal L_m=\overline\Psi(i\gamma^\mu D_\mu-M)\Psi.
\tag{YM.5}
$$

For two assigned real coefficients,
$M=\operatorname{diag}(m_i,m_j)=m_0I+\delta m\,T^3$ with
$m_0=(m_i+m_j)/2$, $\delta m=m_i-m_j$, and $T^a=\sigma^a/2$.
Assigning a physical mass to a fitness value additionally specifies its units
and normalization. Gauge covariance of the mass term requires either a
commuting fixed $M$ or a transforming field or background $M'=\Omega M\Omega^{-1}$.
The Clifford and fermionic Fock constructions are those of
{doc}`03_lattice_qft`; their CAR refer to those constructed operators.
:::

:::{prf:definition} Connection obtained from a declared link lift
:label: def-gauge-field-from-phases

A recorded antisymmetric phase $\theta_{ji}=-\theta_{ij}$ can define
$u_{ij}=e^{i\theta_{ij}}$. To define a doublet comparison choose a Hermitian
traceless generator $n_{ij}^aT^a$ and set

$$
 U_{ij}=\exp(i\theta_{ij}n_{ij}^aT^a),\qquad U_{ji}=U_{ij}^{-1}.
$$

The generator choice is part of the lift. Locally write a smooth
anti-Hermitian connection $\mathcal A=igA$, where $A=A^aT^a$ is Hermitian,
and require $U_{i,i+h\xi}=I+h\mathcal A(\xi)+O(h^2)$.
The comparison $U_{ij}$ transports the coordinates at $j$ into the frame at
$i$. A phase formula symmetric under $i\leftrightarrow j$ must be oriented
before it supplies such comparisons.
:::

:::{prf:proposition} Wilson and matter terms for the reconstructed frame links
:label: prop-ym-recorded-frame-sector

Use the normalized doublets and $B(z)\in SU(2)$ of
{prf:ref}`prop-sm-direct-su2-frames`. On their valid vertex graph set
$B_i=B(z_i)$ and $U_{ij}=B_iB_j^\dagger$. These links obey inverse reversal
and local frame covariance. For every closed path $C$ and every declared face,

$$
U_C=I_2,\qquad W(C)=1,\qquad S_W=0.
$$

For $H_i=r_i z_i$, $r_i\ge0$, their covariant difference satisfies
$\|U_{ij}H_j-H_i\|^2=(r_j-r_i)^2$. These identities concern this specific
frame construction; the separately chosen lift above has its own holonomies.

*Proof.* The upstream completion gives $B_i^\dagger B_i=I$,
$\det B_i=1$, and $B(\Omega_i z_i)=\Omega_i B_i$. Therefore

$$
U_{ji}=U_{ij}^\dagger=U_{ij}^{-1},\qquad
U'_{ij}=\Omega_iU_{ij}\Omega_j^{-1}.
$$

For $i_k=i_0$, associativity and adjacent unitarity give

$$
\prod_{r=0}^{k-1}U_{i_ri_{r+1}}
=B_{i_0}(B_{i_1}^\dagger B_{i_1})\cdots
(B_{i_{k-1}}^\dagger B_{i_{k-1}})B_{i_0}^\dagger=I_2.
$$

Its normalized trace is one, and each Wilson summand is
$\beta_P(1-\tfrac12\operatorname{Re}\operatorname{Tr}I_2)=0$.
Since $B_j e_1=z_j$, we have
$U_{ij}H_j=r_jB_iB_j^\dagger B_je_1=r_jz_i$.
Subtracting $r_iz_i$ and using $\|z_i\|=1$ proves the radial identity,
including the weighted action in {prf:ref}`cor-sm-frame-link-radial-action`.
The constant loop expectation corresponds to zero exponential area-decay
rate. Thus the $0<a<1$ plaquette law in
{prf:ref}`thm-area-law-confinement` describes a different link law.
$\square$
:::

:::{prf:definition} Covariant derivative and curvature
:label: def-covariant-derivative-ym

Set

$$
 D=d+\mathcal A,
 \qquad \mathcal F=d\mathcal A+\mathcal A\wedge\mathcal A,
 \qquad F=\mathcal F/(ig).
\tag{YM.6}
$$

For $\Psi'=\Omega\Psi$ the connection transforms by
$\mathcal A'=\Omega\mathcal A\Omega^{-1}-(d\Omega)\Omega^{-1}$.
Then $D'\Psi'=\Omega D\Psi$ and
$\mathcal F'=\Omega\mathcal F\Omega^{-1}$, by expansion and the product
rule. This fixes the sign convention in all link and curvature formulas.
:::

:::{prf:definition} Discrete derivatives and a consistency criterion
:label: def-discrete-derivatives-ym

Given local reconstructed tangent displacements $\xi_{ij}$, define

$$
 (D_\mu^{(h)}\Psi)_i
 =\sum_j a_{ij}^{\mu}(U_{ij}\Psi_j-\Psi_i).
\tag{YM.7}
$$

Assume $\sum_j a_{ij}^{\mu}\xi_{ij}^{\nu}=\delta_\mu^\nu$ and
$\sum_j|a_{ij}^{\mu}|\,|\xi_{ij}|^2\to0$. If the field and comparison
maps admit a common Taylor bound
$|U_{ij}\Psi_j-\Psi_i-\xi_{ij}^{\nu}D_\nu\Psi_i|
\le C|\xi_{ij}|^2$, then

$$
 |D_\mu^{(h)}\Psi_i-D_\mu\Psi_i|
 \le C\sum_j|a_{ij}^{\mu}|\,|\xi_{ij}|^2\longrightarrow0.
$$

The bound follows by substituting the Taylor expansion into (YM.7).
Geometry, tangent reconstruction, and these moment conditions refer to the
same sample as the derivative. Time derivatives use the actual assigned
proper-time increments; these need not equal a common multiple of the recorded
timestep, by {doc}`02_causal_set_theory`.
:::

(sec-ym-noether)=
## 4. Noether identities and recorded balance laws

:::{div} feynman-prose
A symmetry supplies a conserved current when the action and its equations of
motion possess that symmetry. A recorded quantity also has an exact balance
law: conditional expectation separates its predictable change from a martingale
increment. This tells you which update changed the quantity and by how much.

For a continuous description, use the generator already derived for the
algorithm. State-dependent noise can contribute to its Itô drift; boundary
killing has its own stopped-process or boundary formulation. Tracking those
terms is what makes the balance equation describe the recorded dynamics.
:::

### 4.1. Phase current and fitness balance

:::{prf:theorem} Phase conservation and the exact stochastic balance
:label: thm-u1-noether-current

For (YM.5) with Hermitian $M$, the common phase symmetry has current
$j^\mu=\overline\Psi\gamma^\mu\Psi$. On the field equations,
$\partial_\mu j^\mu=0$ in flat coordinates, or the corresponding covariant
divergence vanishes on a supplied curved geometry.

For a recorded Markov chain $S_k$ with kernel $P_h$, any integrable observable
$Q$ instead has the exact decomposition

$$
 Q(S_n)-Q(S_0)
 =\sum_{k=0}^{n-1}(P_hQ-Q)(S_k)+M_n,
\tag{YM.8}
$$

where $M_n$ is a martingale. In particular, a total fitness
$Q(S)=\sum_i a_i(S)F_i(S)$ is conserved in conditional expectation precisely
when $P_hQ=Q$ on the states considered.
:::

:::{prf:proof}
The Dirac equation and its adjoint give
$\gamma^\mu D_\mu\Psi=-iM\Psi$ and
$(D_\mu\overline\Psi)\gamma^\mu=i\overline\Psi M$.
Taking the divergence cancels the two mass terms and the connection terms.
This proves the phase current identity.

For the chain let
$\Delta M_{k+1}=Q(S_{k+1})-(P_hQ)(S_k)$. Its conditional expectation given
$S_0,\ldots,S_k$ vanishes. Summing
$Q(S_{k+1})-Q(S_k)=\Delta M_{k+1}+(P_hQ-Q)(S_k)$ proves (YM.8).
No correspondence between $Q$ and the phase charge is needed for this balance.
:::

For a continuous kinetic jump model, the analogous generator formula is

$$
 LQ=\sum_i\left[v_i\cdot\nabla_{x_i}Q+b_i\cdot\nabla_{v_i}Q\right]
 +\frac12\operatorname{Tr}(\Sigma\Sigma^{\mathsf T}\nabla_v^2Q)
 +\int[Q(S')-Q(S)]\,r(S,dS').
\tag{YM.9}
$$

All swarm derivatives occur when fitness depends on all walkers. The
martingale form is $Q(S_t)-Q(S_0)-\int_0^tLQ(S_s)ds$. Kinetic displacement
contributes $v_i\cdot\nabla_{x_i}Q$; acceleration contributes through the
velocity derivatives. Cloning contributes the jump term when it is implemented by the stated
continuous-time jump kernel. Boundary killing is treated through the stopped
process and its boundary domain, or an explicitly specified cemetery-state
transition. The discrete algorithm retains the exact kernel balance (YM.8).
For smooth weights along absolutely continuous trajectories,
$\partial_t\sum_iF_i\delta_{x_i}+\nabla\cdot\sum_iF_iv_i\delta_{x_i}
=\sum_i\dot F_i\delta_{x_i}$ between jumps, as follows by testing against a
smooth function and differentiating. Jumps add their signed atomic increments.

:::{prf:proposition} Identification of the drift and the stopped balance
:label: prop-ym-generator-balance

In (YM.9), $b$ is the Itô velocity drift. With the geometric kinetic
coefficients of {prf:ref}`def-gg-generator-decomp`, it is

$$
b_i=-\nabla U_i+F_i-\gamma v_i-\nu(L_XV)_i+b_{\mathrm{geo},i},
\qquad b_{\mathrm{geo}}=\frac12\sum_\ell(DB_\ell)B_\ell.
$$

Here $B_\ell$ are the full phase-space noise columns, and the covariance
in (YM.9) is that of the same columns. The jump kernel contains the complete
selected cloning update. For $Q\in C^2$ with locally integrable jump
increments, the process stopped before explosion, boundary exit, and failure
of the coefficient chart satisfies the localized Dynkin martingale identity.
Removing localization requires integrability of that martingale and drift.

*Proof.* Import {prf:ref}`lem-gg-geometric-drift`. For
$B_\ell=(0,\Sigma_{:\ell})$, its velocity component reads

$$
(b_{\mathrm{geo}})_a
=\frac12\sum_{\ell,b}\Sigma_{b\ell}\partial_{v_b}\Sigma_{a\ell},
$$

where $a,b$ range over all swarm velocity coordinates. When $\Sigma$
depends only on positions every displayed derivative vanishes. Applying
Itô's formula to the continuous motion gives drift
$v\cdot\nabla_xQ+b\cdot\nabla_vQ+
\tfrac12\Sigma\Sigma^{\mathsf T}:\nabla_v^2Q$ and noise
$\nabla_vQ\,\Sigma\,dW$. Compensating the jump increment
$Q(S')-Q(S)$ adds its integral against $r(S,dS')$ to the drift and its
compensated stochastic integral to the martingale. This is (YM.9).

For a localization time $\tau_R$ strictly before boundary exit, integration gives

$$
Q(S_{t\wedge\tau_R})-Q(S_0)
-\int_0^{t\wedge\tau_R}LQ(S_s)\,ds=M_{t\wedge\tau_R}.
$$

On bounded localizations with integrable compensated jumps this is a true
martingale. A boundary exit can instead introduce the jump
$Q(\dagger)-Q(S_{\tau-})$ when the state is sent to $\dagger$; its
compensator must be derived from that killing construction. It is not
automatically a finite-rate $r(S,dS')\,dt$ term. For the recorded chain,
conditional expectation directly gives (YM.8), with no passage from a
fixed per-step cloning probability to a continuous jump rate. $\square$
:::



### 4.2. Internal currents and symmetry breaking

:::{prf:theorem} Internal current identity, including chiral sources
:label: thm-su2-noether-current

For a vectorlike internal representation with Hermitian generator $T$, the
flat-space Dirac equations without a background connection give

$$
 \partial_\mu(\overline\Psi\gamma^\mu T\Psi)
 =i\overline\Psi[M,T]\Psi.
\tag{YM.10}
$$

With a gauge connection, the same identity uses the adjoint covariant
divergence of the current multiplet. For a left current
$j_{L,T}^\mu=\overline\Psi\gamma^\mu T P_L\Psi$, with internal $M,T$
commuting with the spin matrices, the ungauged identity is

$$
 \partial_\mu j_{L,T}^\mu
 =i\overline\Psi(MT P_L-T P_RM)\Psi.
\tag{YM.11}
$$

Thus a nonzero scalar Dirac mass sources the left current even when $[M,T]=0$.
A full gauge-invariant Yukawa model includes the scalar current in its total
Noether identity.
:::

:::{prf:proof}
Differentiate the vector current and substitute the two Dirac equations used
above. The terms are $i\overline\Psi MT\Psi$ and
$-i\overline\Psi TM\Psi$. In the chiral calculation use
$\gamma^\mu P_L=P_R\gamma^\mu$ in the second term, which gives (YM.11).
For $M=mI$, its right-hand side is
$-im\overline\Psi T\gamma^5\Psi$. With a connection, collect its two terms
into the adjoint commutator in the divergence. Finally, varying a complete
invariant matter-plus-scalar action with a local symmetry parameter makes
its coefficient the divergence of the sum of currents, with the field
Euler–Lagrange terms. On those equations the total source cancels.
:::

:::{prf:definition} Recorded flow diagnostics
:label: def-noether-flow-equations

The algorithmic flow diagnostic for an observable $Q$ is its one-step drift
$\mathcal D_hQ=(P_hQ-Q)/h$, or $LQ$ for an identified continuous generator.
A residual measured over $n$ steps is
$Q(S_n)-Q(S_0)-h\sum_{k<n}\mathcal D_hQ(S_k)$. By (YM.8) it is a
martingale residual, whose variance can be estimated from its conditional
increments. A field-current residual uses (YM.10) or (YM.11) with the
specified mass, connection, and scalar sources. These are distinct diagnostics.
:::

:::{prf:definition} Hamiltonian field formulation
:label: def-hamiltonian-formulation-ym

For a chosen Lorentzian field Lagrangian with nonsingular velocity Hessian on
its unconstrained variables, set
$\Pi=\partial\mathcal L/\partial\dot\Phi$ and
$\mathcal H=\Pi\dot\Phi-\mathcal L$. The variational equations become
$\dot\Phi=\delta H/\delta\Pi$ and
$\dot\Pi=-\delta H/\delta\Phi$. Gauge components with no time derivatives
supply constraints, including Gauss' law, rather than invertible Legendre
coordinates. Friction, stochastic forcing, and cloning remain terms of the
recorded Markov dynamics; the Legendre transform does not turn them into
conservative Hamiltonian motion.
:::

(sec-ym-action)=
## 5. Wilson action and discrete field equations

### 5.1. Oriented links and faces

:::{prf:definition} Comparison matrices on an oriented graph
:label: def-link-variable-ym

Choose one orientation of each unoriented edge and assign $U_{ij}\in SU(2)$,
with $U_{ji}=U_{ij}^{-1}$. A local frame change acts by

$$
 \Psi_i\mapsto\Omega_i\Psi_i,
 \qquad U_{ij}\mapsto\Omega_iU_{ij}\Omega_j^{-1}.
\tag{YM.12}
$$

The assignment applies to each edge type used in the field construction.
Setting temporal links to $I$ is a gauge choice on a temporal spanning forest.
Its residual transformations satisfy $\Omega_i=\Omega_j$ along those links.
:::

Indeed, on a tree choose the root frame and recursively set
$\Omega_j=\Omega_iU_{ij}$; the transformed tree links equal $I$. On a cycle,
the product transforms by conjugation, so a nonidentity cycle holonomy
obstructs setting every link to $I$. This also specifies the boundary
restrictions of a temporal gauge.

:::{prf:definition} Face holonomy and local curvature
:label: def-plaquette-field-strength-ym

An oriented face $P$ has a specified cyclic boundary
$(i_0,i_1,\ldots,i_r=i_0)$. Its based holonomy is

$$
 U_P=U_{i_0i_1}U_{i_1i_2}\cdots U_{i_{r-1}i_0}.
$$

Reversing the boundary inverts this matrix. If $U_P$ is sufficiently close to
$I$ to have a traceless anti-Hermitian principal logarithm $X_P=\log U_P$,
and a nonzero oriented area $A_P$ has been assigned on the reconstructed
geometry, define $\mathcal F_P=X_P/A_P$ and $F_P=\mathcal F_P/(ig)$.
The latter is Hermitian with convention (YM.6). A generic finite loop need
not admit this local curvature chart.
:::

To combine adjacent triangles, orient the shared edge oppositely and transport
both holonomies to the same basepoint. The shared comparisons then cancel in
the product. Multiplying traces or ignoring the basepoint transport does not
perform this cancellation. Face areas and orientations enter the continuum
quadrature separately from their graph boundaries.

:::{prf:definition} Finite Wilson action
:label: def-wilson-action-ym

For a finite face set $\mathcal P$ and nonnegative coefficients $\beta_P$,
let

$$
 s_P=1-\frac12\operatorname{Re}\operatorname{Tr}U_P,
 \qquad S_W(U)=\sum_{P\in\mathcal P}\beta_Ps_P.
\tag{YM.13}
$$

This is a dimensionless Euclidean action. Each $s_P\in[0,2]$, hence
$0\le S_W\le2\sum_P\beta_P$. Other representations replace $1/2$ by the
inverse representation dimension and require their own normalization.
:::

:::{prf:theorem} Gauge invariance of Wilson observables and actions
:label: thm-wilson-action-gauge-invariance

The face action (YM.13), and the normalized trace around every closed loop,
are invariant under (YM.12). Gauge invariance also holds for any function of
these traces; in particular $\sum_P(\beta_Ps_P+\lambda_Ps_P^2)$ is invariant.
:::

:::{prf:proof}
Substitute (YM.12) into the ordered product. Every intermediate pair
$\Omega_j^{-1}\Omega_j$ cancels, leaving
$U_P'=\Omega_{i_0}U_P\Omega_{i_0}^{-1}$. Cyclicity of trace proves the
claim. Consequently gauge invariance specifies a class of admissible actions;
it does not uniquely select their linear combination in (YM.13).
:::

### 5.2. Variation and the discrete Ward identity

:::{prf:theorem} Link equations and their continuum variational form
:label: thm-yang-mills-eom

For a differentiable matter action $S_m$, stationary links of $S_W+S_m$ obey

$$
 \frac12\sum_{P\ni e}\beta_P
 \operatorname{Im}\operatorname{Tr}(T^aU_e\Sigma_{P,e})
 =J_e^a,
 \qquad
 J_e^a=-\left.\frac d{dt}S_m(e^{itT^a}U_e)\right|_{t=0}.
\tag{YM.14}
$$

Here each occurrence of $e$ is counted separately. Orient and cyclically
rebase its face so that $\operatorname{Re}\operatorname{Tr}U_P
=\operatorname{Re}\operatorname{Tr}(U_e\Sigma_{P,e})$; inverse occurrences
use the reversed face.

For a smooth continuum Hermitian field with (YM.6), set
$S_{\rm YM}=\frac12\int\operatorname{Tr}(F_{\mu\nu}F^{\mu\nu})$.
If the matter variation is
$\delta S_m=2\int\operatorname{Tr}(J^\nu\delta A_\nu)$, compactly
supported variations give

$$
 D_\mu F^{\mu\nu}=J^\nu.
\tag{YM.15}
$$

A conventional coupling can be included in the definition of $J$.
To obtain (YM.15) as a limit of (YM.14), the discrete first variations and
sources must converge as well as the actions.
:::

:::{prf:proof}
Since $\operatorname{Re}(iz)=-\operatorname{Im}z$,
$d[-\beta_P\operatorname{Re}\operatorname{Tr}(e^{itT^a}U_e\Sigma)/2]/dt$
at zero is the corresponding term in (YM.14). Stationarity and the source
definition give the equation.

In the continuum,
$\delta F_{\mu\nu}=D_\mu\delta A_\nu-D_\nu\delta A_\mu$ with the
adjoint derivative $D_\mu B=\partial_\mu B+ig[A_\mu,B]$.
Antisymmetry and integration by parts give

$$
 \delta S_{\rm YM}
 =2\int\operatorname{Tr}(F^{\mu\nu}D_\mu\delta A_\nu)
 =-2\int\operatorname{Tr}((D_\mu F^{\mu\nu})\delta A_\nu).
$$

Adding $\delta S_m$ proves (YM.15). If discrete critical fields converge,
convergence of these first variations against every compactly supported test
variation makes their zero values pass to the continuum equation.
:::

:::{prf:lemma} Exact graph Ward identity
:label: lem-ym-discrete-ward

Use the invariant inner product $\langle X,Y\rangle=-\operatorname{Re}\operatorname{Tr}(XY)$
on anti-Hermitian matrices. Let the left link force $G_{ij}$ of an invariant
action satisfy
$dS=\sum_{i\to j}\langle G_{ij},\delta U_{ij}U_{ij}^{-1}\rangle$
plus its matter variations. If $R_i$ denotes the coefficient of the matter
variation generated by $X_i\Psi_i$, then

$$
 \sum_{i\to j}G_{ij}
 -\sum_{k\to i}\operatorname{Ad}_{U_{ki}^{-1}}G_{ki}+R_i=0.
\tag{YM.16}
$$

On the matter equations $R_i=0$. This identity holds on the finite graph.
:::

:::{prf:proof}
An infinitesimal frame change gives
$\delta U_{ij}U_{ij}^{-1}=X_i-\operatorname{Ad}_{U_{ij}}X_j$.
Invariance makes its total action variation zero. Invariance of the inner
product moves each adjoint action onto its force. Collecting the coefficient
of each arbitrary $X_i$ yields (YM.16). The matter coefficient vanishes when
its Euler–Lagrange derivatives vanish.
:::

(sec-ym-path-integral)=
## 6. Finite field measures

:::{prf:definition} Finite gauge partition function
:label: def-partition-function-ym

Give each independently oriented link normalized Haar measure $dU_e$. For a
positive matter measure $d\mu_m$ and real total action bounded below with
finite nonzero integral, define

$$
 Z=\int e^{-S_W-S_m}\prod_e dU_e\,d\mu_m,
 \qquad d\mu=Z^{-1}e^{-S_W-S_m}\prod_e dU_e\,d\mu_m.
\tag{YM.17}
$$

For the pure finite gauge model, $e^{-2\sum\beta_P}\le Z\le1$.
A finite Grassmann integral is instead an algebraic integral; after integrating
fermions its determinant is a probability weight only when it is real and
nonnegative and the resulting integral can be normalized.
:::

:::{prf:theorem} Gauge invariance of the finite field integral
:label: thm-path-integral-gauge-invariance

If $S_m$ and its measure transform invariantly, (YM.17) is invariant under all
local frame changes. In a finite vectorlike Grassmann model, the paired
transformation $\Psi\mapsto\Omega\Psi$,
$\overline\Psi\mapsto\overline\Psi\Omega^{-1}$ has Berezin Jacobian one.
:::

:::{prf:proof}
Haar measure is left and right invariant, so
$d(\Omega_iU_{ij}\Omega_j^{-1})=dU_{ij}$. The action is invariant by
{prf:ref}`thm-wilson-action-gauge-invariance` and the matter hypothesis.
Changing variables in the finite integral proves the result. A linear change
of Grassmann coordinates $\Psi'=A\Psi$ has Jacobian $(\det A)^{-1}$;
the barred transformation has the inverse determinant, and their product is
one. For the specified chiral Standard Model representation, the applicable
anomaly coefficients are already evaluated in
{prf:ref}`prop-sm-generation-anomaly-cancellation`, as detailed below.
:::

:::{prf:corollary} Anomaly tests for the established generation
:label: cor-ym-inherited-anomaly-tests

For the left-handed generation in {prf:ref}`thm-sm-so10-isomorphism`,
the perturbative gauge and mixed gauge-gravitational coefficients vanish,
and the ordinary $SU(2)$ doublet parity test is satisfied on spin backgrounds.
These are the tests established in
{prf:ref}`prop-sm-generation-anomaly-cancellation`.

*Proof.* With quadratic fundamental index $1/2$ and cubic color index one,
the nonzero candidate coefficients reduce to

$$
\begin{aligned}
\mathcal A_{333}&=2-1-1=0,\\
\mathcal A_{33Y}&=\tfrac16-\tfrac13+\tfrac16=0,\\
\mathcal A_{22Y}&=\tfrac14-\tfrac14=0,\\
\mathcal A_{\mathrm{grav}\,Y}&=1-2+1-1+1=0,\\
\mathcal A_{YYY}&=\tfrac1{36}-\tfrac89+\tfrac19-\tfrac14+1=0.
\end{aligned}
$$

The multiplicities are the six $Q$, three $u^c$, three $d^c$, two $L$,
one $e^c$, and neutral singlet components. Terms with a single generator
of a simple factor vanish by tracelessness. For the weak cubic trace,
$\{T^b,T^c\}=\delta^{bc}I/2$ gives
$\operatorname{Tr}(T^a\{T^b,T^c\})=0$. The $3+1=4$ weak doublets
give sign $(-1)^{4n_g}=1$ for $n_g$ generations. These calculations import
the anomaly criteria with precisely the representation and background scope
of the cited proposition; regulator construction and other quotient-bundle
global tests retain their respective requirements. $\square$
:::

:::{prf:proposition} Exact recorded density and comparison with the field integral
:label: prop-ym-density-and-support

Let $R$ be the normalized reference path law, $P=\mathcal L R$ the complete
algorithm path law, and $\mathcal L=e^{-S_h}$ its likelihood. Use the same
descriptor $\mathscr D$ as in {prf:ref}`thm-sm-path-descriptor-density` and set
$\lambda=\mathscr D_\#R$, $\nu=\mathscr D_\#P$. Then

$$
\nu=a\lambda,\qquad
a(y)=\mathbb E_R[\mathcal L\mid\mathscr D=y],\qquad \int a\,d\lambda=1.
$$

If the integrated field model has density $b$ against this same $\lambda$,
with $\int|b|d\lambda<\infty$ and $Z_b=\int b\,d\lambda\ne0$, equality
of all bounded observable expectations holds exactly when $b/Z_b=a$ almost
everywhere. An integrable weighting of the direct samples represents that
field functional exactly when $b=0$ almost everywhere on $\{a=0\}$.

*Proof.* For bounded measurable $f$, conditional expectation gives

$$
\begin{aligned}
\int f\,d\nu
&=\int f(\mathscr D(s))\mathcal L(s)\,dR(s)\\
&=\int f(\mathscr D(s))
  \mathbb E_R[\mathcal L\mid\sigma(\mathscr D)](s)\,dR(s)
=\int f(y)a(y)\,d\lambda(y).
\end{aligned}
$$

Taking $f=1$ gives normalization. Equality with $Z_b^{-1}\int fb\,d\lambda$
for indicator functions is equality of finite measures, hence equality of
their densities. On $\{a>0\}$ put $w=b/a$, and set $w=0$ elsewhere.
Then $wa=b$ exactly under the stated support condition, and

$$
\frac{\mathbb E_\nu[wf]}{\mathbb E_\nu[w]}
=\frac{\int fb\,d\lambda}{Z_b},\qquad
\mathbb E_\nu|w|=\int|b|d\lambda,\qquad
\mathbb E_\nu|wf|^2=\int_{a>0}\frac{|bf|^2}{a}\,d\lambda.
$$

The last integral is the corresponding second-moment test; a ratio of finite
sample means has its own sampling error. If the reference measures differ,
their Radon--Nikodym factor must first be included in $b$. Survival
conditioning replaces $\mathcal L$ by
$\mathcal L\mathbf1_E/P(E)$ for $P(E)>0$ before taking its conditional
expectation. In particular the effective descriptor action is $-\log a$;
averaging the path action before exponentiation does not give this density.
$\square$
:::

:::{prf:corollary} Support obstruction for frame-link sampling
:label: cor-ym-flat-support

On a finite graph containing a simple cycle, no integrable weighting of the
frame links in {prf:ref}`prop-ym-recorded-frame-sector` represents a normalized
independent-link field measure absolutely continuous with respect to product
Haar measure. This includes integrable complex densities of nonzero total mass.

*Proof.* Let $\mathcal Z$ be the identity-holonomy set for that cycle. The
frame-link law gives $\nu(\mathcal Z)=1$. Condition product Haar measure on
all edges except one occurring once in the cycle. The remaining holonomy
is $AUB$ or $AU^{-1}B$, hence Haar distributed. Haar measure on $SU(2)$
has no atoms: a positive singleton mass would, by translation, assign
arbitrarily many distinct points arbitrarily large total mass. Thus
$\operatorname{Haar}(\mathcal Z)=0$. Every absolutely continuous field
measure assigns $\mathcal Z$ mass zero, while every weighted frame measure
is supported there. Equality would force its total mass to vanish, contrary
to normalization. This is the support calculation of
{prf:ref}`prop-sm-flat-link-support`. $\square$
:::

Normalized Haar measure already has finite total mass. Gauge fixing is an
optional coordinate reduction of this finite integral. On a spanning tree the
recursive transformation above fixes the tree links, leaving the cycle
holonomies and any residual root frame. A continuum Faddeev–Popov expression
requires a gauge slice, its Jacobian, and treatment of multiple intersections.
It is unnecessary for existence of the finite pure-gauge integral.

The law (YM.17) is a specified field measure. To apply it to recorded links,
one must prove equality with their pushforward law, or bound the discrepancy
for the observables in question. Neither QSD uniqueness nor exchangeability
identifies that density. If killed-kernel eigenobjects are available,
{prf:ref}`prop-kl-doob-transform` provides the exact stationary alternative

$$
 P_t^\eta F=\frac{e^{\alpha_Nt}}{\eta_N}Q_t(\eta_NF),
 \qquad \widehat\pi_N=\eta_N\nu_N,\qquad \nu_N(\eta_N)=1.
\tag{YM.18}
$$

The eigenfunction is scaled by the finite positive integral $\nu_N(\eta_N)$,
as in the cited proposition. Scaling cancels in $P_t^\eta$ and gives
$\widehat\pi_N(1)=1$. Indeed $P_t^\eta1=1$ and
$\widehat\pi_N(P_t^\eta F)=e^{\alpha_Nt}\nu_NQ_t(\eta_NF)
=\nu_N(\eta_NF)$. Reversibility and functional inequalities must then be
checked for this transformed law and generator.

(sec-ym-observables)=
## 7. Loop observables, sampling, and clustering

:::{prf:definition} Wilson loop
:label: def-wilson-loop-observable-ym

For an oriented closed path $C$ define
$W(C)=\frac12\operatorname{Tr}\prod_{e\in C}U_e$, using inverse matrices
when traversing a reversed edge. For $SU(2)$ this is real and $|W(C)|\le1$.
It is gauge invariant by the same cancellation as a face trace.
:::

:::{prf:theorem} An area law under an explicit plaquette law
:label: thm-area-law-confinement

Suppose a loop enclosing $k$ faces has holonomy, in a specified gauge,
$V_1\cdots V_k$, where the $SU(2)$ matrices $V_j$ are independent with a
common conjugation-invariant law satisfying $\mathbb EV_j=aI$, $0<a<1$.
Then

$$
 \mathbb EW(C)=a^k=\exp(-\sigma A(C)),
 \qquad \sigma=-\log a/A_0>0,
\tag{YM.19}
$$

when each face has area $A_0$ and $A(C)=kA_0$.
An exact example is the central one-face law
$Z_\beta^{-1}\exp[(\beta/2)\operatorname{Tr}V]dV$, $\beta>0$, together
with this independent-face construction. Application to the Fractal Set
requires establishing its loop factorization and face law, or a suitable
replacement estimate.
:::

:::{prf:proof}
A conjugation-invariant matrix expectation commutes with every $SU(2)$
matrix and hence is scalar by the block argument of
{prf:ref}`prop-su2-invariance-ym`. Independence, applied to each matrix entry,
gives $\mathbb E(V_1\cdots V_k)=a^kI$ even though the matrices need not
commute. Taking the normalized trace proves (YM.19).

For the example put $X=\operatorname{Tr}V/2\in[-1,1]$. Haar measure is
invariant under $V\mapsto-V$, so its $X$ distribution is symmetric. Thus
$\mathbb E_{\rm Haar}[Xe^{\beta X}]=\mathbb E[X\sinh(\beta X)]>0$.
The tilted expectation of $X$ is strictly less than one because $X<1$
almost surely. Therefore $0<a=\mathbb E_\beta X<1$.
:::

The independent-face example proves a finite area law for a specified model.
Correlated plaquettes sharing links require a separate integration or expansion
argument. Exponential relaxation of the walker process controls time
correlations; it supplies no independent-face factorization.

:::{prf:theorem} Temporal clustering and the geometric condition for spatial decay
:label: thm-cluster-decomposition

Let a stationary Markov semigroup satisfy
$\|P_tf\|_{L^2(\pi)}\le M e^{-\lambda t}\|f\|_{L^2(\pi)}$ for centered
$f$, with $M<\infty$ and $\lambda>0$. For real square-integrable $F,G$,

$$
 |\operatorname{Cov}_\pi(F(S_0),G(S_t))|
 \le M e^{-\lambda t}
 \sqrt{\operatorname{Var}_\pi F\operatorname{Var}_\pi G}.
\tag{YM.20}
$$

If a Euclidean spacetime field law is also invariant under rotations mixing
time and space, and the rotated observables have the same norm bounds,
(YM.20) transfers to separations along any rotated axis. These are additional
symmetries of that field law.
:::

:::{prf:proof}
The covariance is $\langle F-\pi F,P_t(G-\pi G)\rangle_\pi$.
Cauchy–Schwarz and the assumed semigroup estimate prove (YM.20).
For the second assertion, change variables by the stated Euclidean rotation
in the correlation and apply the first assertion to the rotated observables.
:::

For a bounded single-walker observable $|\varphi|\le B$,
{prf:ref}`thm-mixing-variance-corrected` gives the useful same-frame estimate

$$
 \operatorname{Var}_{\pi_N}\!\left(\frac1N\sum_i\varphi(z_i)\right)
 \le\frac{4B^2}{N}\left(H_N+\frac12\log2\right),
 \qquad H_N=D_{\rm KL}(\pi_N\Vert\rho^{\otimes N}).
\tag{YM.21}
$$

For an $L$-Lipschitz single-walker observable, (YM.1) gives instead
$\operatorname{Var}(N^{-1}\sum_i\varphi(z_i))\le C_*L^2/N$.
A loop observable depending on a reconstructed graph is a function of the
whole swarm. The applicable bound is $C_*\int|\nabla W|^2d\pi_N$ after
establishing its regularity on the relevant status stratum. The factor $1/N$
requires the corresponding bound on this gradient; discontinuous changes of
neighbors need their own control.

Temporal averaging also has an elementary error estimate. For stationary
centered $F$ satisfying (YM.20),

$$
 \operatorname{Var}\!\left(\frac1T\int_0^TF(S_t)dt\right)
 \le\frac{2M\operatorname{Var}_\pi F}{\lambda T},
\tag{YM.22}
$$

because the double covariance integral is bounded by
$2M\operatorname{Var}F\int_0^T(T-t)e^{-\lambda t}dt/T^2$.
This provides an observable-level uncertainty estimate without identifying
the time decay rate with a particle mass.

(sec-ym-continuum)=
## 8. Classical continuum consistency and perturbative running

### 8.1. The quadratic small-loop limit

:::{prf:theorem} Wilson action consistency under geometric quadrature
:label: thm-continuum-limit-ym

Let a sequence of finite two-complexes approximate a specified smooth
Euclidean geometry, and let their links approximate a fixed smooth connection.
For each face write

$$
 X_P=\log U_P=Y_P+R_P,
 \qquad Y_P=igA_PF(n_P),
$$

where $n_P$ is its oriented tangent two-plane. Assume

$$
 \sum_P\frac{\beta_P}{4}\|Y_P\|_F^2\longrightarrow S_{\rm YM},
 \qquad
 \sum_P\beta_P\|R_P\|_F(2\|Y_P\|_F+\|R_P\|_F)\longrightarrow0,
\tag{YM.23}
$$

and $\sum_P\beta_P\|X_P\|_{\rm op}^2\|X_P\|_F^2\to0$.
Then $S_W\to S_{\rm YM}$. The first condition is a quadrature assumption
about the actual face areas, orientations, and weights, not a consequence of
vertex propagation of chaos.
:::

:::{prf:proof}
If the eigenvalues of the anti-Hermitian $X$ are $i\theta_j$, then

$$
 \left|1-\frac12\operatorname{Re}\operatorname{Tr}e^X
       -\frac14\|X\|_F^2\right|
 \le\frac1{48}\sum_j\theta_j^4
 \le\frac1{48}\|X\|_{\rm op}^2\|X\|_F^2.
\tag{YM.24}
$$

This is $|1-\cos\theta-\theta^2/2|\le\theta^4/24$ applied to both
eigenvalues. Also
$|\|Y+R\|_F^2-\|Y\|_F^2|\le\|R\|_F(2\|Y\|_F+\|R\|_F)$.
Multiply by $\beta_P$, sum, and use (YM.23). This is the same local
quadratic estimate used in {prf:ref}`lem-lqft-wilson-quadratic`.
:::

For example, on a $D$-dimensional cubic mesh of spacing $a$ in a bounded
region, with one face per $\mu<\nu$ at each site and
$X_P=ig a^2F_{\mu\nu}+O(a^3)$ uniformly, choose

$$
 \beta=\frac{4a^{D-4}}{g^2}.
\tag{YM.25}
$$

Then $\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$ gives the leading sum
$\frac12\sum_{x,\mu<\nu}a^D(F_{\mu\nu}^a)^2$, converging to
$\frac14\int\sum_{\mu,\nu,a}(F_{\mu\nu}^a)^2$.
The accumulated mixed remainder is $O(a)$ and the quartic error is $O(a^4)$.
In $D=4$, (YM.25) is $\beta=4/g^2$. This example verifies the hypotheses
for smooth test connections on that mesh. It does not modify the Fractal Set
reconstruction. Unbounded domains require tail integrability of the action
density in addition to local convergence; confinement bounds are used as
moment estimates, without replacing the domain by a compact space.

A quantum continuum limit concerns the measures and their fluctuating fields,
which need not resemble smooth connections at every scale. Its requirements
appear in {ref}`sec-qft-axioms-verification`.

### 8.2. Renormalized couplings

:::{prf:definition} Beta function of a specified field model
:label: def-beta-function-ym

Given a field theory, a renormalization prescription, and a coupling measured
at momentum scale $\mu$, define $\beta_g(g)=\mu\,dg/d\mu$. A perturbative
asymptotically free model has

$$
 \beta_g(g)=-\frac{b_0}{16\pi^2}g^3+O(g^5),\qquad b_0>0.
\tag{YM.26}
$$

For pure $SU(n)$ Yang–Mills the usual perturbative coefficient is
$b_0=11n/3$; adding the chosen matter representations changes it. The
Standard Model matter coefficients and their representation assumptions are
specified in {doc}`04_standard_model`. These coefficients concern a
renormalized quantum field model, after its connection to the sampled law is
established. They do not follow from the mesh size or from a cloning ratio.
:::

:::{prf:theorem} Running implied by the one-loop equation
:label: thm-asymptotic-freedom

If the one-loop truncation of (YM.26) is the adopted running equation, then

$$
 \frac1{g^2(\mu)}=\frac1{g^2(\mu_0)}
   +\frac{b_0}{8\pi^2}\log\frac\mu{\mu_0},
\tag{YM.27}
$$

on its positive-coupling branch. Thus $g(\mu)\to0$ as $\mu\to\infty$.
For the full equation with a remainder bounded by $C g^5$, sufficiently
small positive initial coupling also decreases to zero toward the ultraviolet.
:::

:::{prf:proof}
Differentiate $g^{-2}$ to obtain
$d(g^{-2})/d\log\mu=b_0/(8\pi^2)$ and integrate. For the full equation,
choose $g_*$ so the remainder has magnitude at most half the leading cubic
term for $0<g\le g_*$. In that interval $dg/d\log\mu$ lies between two
strictly negative constant multiples of $g^3$. Integration of the resulting
bounds on $g^{-2}$ keeps the solution positive, decreasing, and tending to
zero. This proves the implication of the beta function; the loop coefficient
itself is perturbative field-theory input.
:::

(sec-ym-constants)=
## 9. Units and calibration of scales

:::{div} feynman-prose
An inverse time is a relaxation rate. Multiplying it by an action unit produces
an energy. To identify that energy with a particle, we also need the correlation
channel, the time coordinate, and the represented generator. Chapter 04 supplies
a precise generator-comparison criterion for that step.

The same care applies to coupling proxies. Two dimensionless formulas can
measure different combinations of algorithm parameters. Their units agree;
their numerical values agree only after the parameter relation has been checked.
Each proxy below therefore retains its defining formula and calibration.
:::

:::{prf:theorem} Action unit from a Gaussian phase convention
:label: thm-effective-planck-constant

If a Gaussian comparison kernel is written
$\exp[-|\Delta x|^2/(2\epsilon_c^2)]$ and the adopted action convention is
$\exp[-m|\Delta x|^2/(4\hbar_{\rm eff}\tau)]$, equality of the quadratic
coefficients gives

$$
 \hbar_{\rm eff}=\frac{m\epsilon_c^2}{2\tau}.
\tag{YM.28}
$$

It has units of action. Its factor of two belongs to this specified convention.
:::

:::{prf:proof}
Equating the coefficients gives
$1/(2\epsilon_c^2)=m/(4\hbar_{\rm eff}\tau)$, hence (YM.28).
Since $[m\epsilon_c^2/\tau]=ML^2/T$, the units agree. A different quadratic
normalization changes the coefficient; experimental Planck normalization
requires a calibration of that convention and the field observables.
:::

:::{prf:theorem} Dimensionless weak-coupling proxy
:label: thm-su2-coupling-constant

For a reference action unit $\hbar_0$ and speed unit $c$, define the
parameter proxy

$$
 \widehat g_{2,\mathrm{clock}}^{\,2}
 =\frac{mc^2\tau}{\hbar_0}\left(\frac\rho{\epsilon_c}\right)^2.
\tag{YM.29}
$$

It is dimensionless. In units $c=\hbar_0=1$ it has the recorded dictionary
form $m\tau\rho^2/\epsilon_c^2$. It equals a physical gauge coupling when
action or correlation matching establishes that equality in the same
normalization.
:::

:::{prf:proof}
The first factor is energy times time divided by action; the second is a
squared length ratio. Substitution of the unit choices gives the displayed
natural-unit formula. Dimensional consistency establishes the proxy, while
matching is the additional identification asserted in the last sentence.
:::

:::{prf:proposition} Relation between the two weak-coupling proxies
:label: prop-ym-weak-proxy-comparison

Denote the $n=3$ proxy of {prf:ref}`def-sm-coupling-definition` by
$\widehat g_{2,\mathrm{Cas}}^2=9\widehat\hbar/(8\widehat\epsilon_c^2)$.
Use its scales $\ell_0,t_0,m_0$ and
$\hbar_{\mathrm{eff}}=m\epsilon_c^2/(2\tau)$ from (YM.28). Then

$$
\widehat g_{2,\mathrm{Cas}}^2=\frac{9mt_0}{16m_0\tau},\qquad
\frac{\widehat g_{2,\mathrm{clock}}^2}
     {\widehat g_{2,\mathrm{Cas}}^2}
=\frac{16m_0c^2\tau^2\rho^2}{9\hbar_0t_0\epsilon_c^2}.
$$

For positive scales they agree precisely when the displayed ratio is one.

*Proof.* Substitute $\widehat\hbar=\hbar_{\mathrm{eff}}t_0/(m_0\ell_0^2)$
and $\widehat\epsilon_c=\epsilon_c/\ell_0$ to cancel $\ell_0^2$:
$9\widehat\hbar/(8\widehat\epsilon_c^2)
=9\hbar_{\mathrm{eff}}t_0/(8m_0\epsilon_c^2)=9mt_0/(16m_0\tau)$.
Division of (YM.29) by this positive expression gives the ratio.
The two proxy definitions therefore retain distinct names until this
parameter relation is imposed; identification with a canonically normalized
field coupling uses the measure and generator comparisons. $\square$
:::

:::{prf:theorem} Dimensionless fitness-coupling proxy
:label: thm-u1-coupling-constant

If $\epsilon_F$ is assigned energy units, the fitness proxy
$\widehat e^{\,2}=mc^2/\epsilon_F$ is dimensionless and becomes
$m/\epsilon_F$ in units $c=1$. If the recorded fitness scale is dimensionless,
one first specifies an energy conversion for it. A physical Abelian coupling
requires the corresponding field normalization and response measurement.
:::

:::{prf:proof}
Both numerator and denominator are energies. Changing the conversion of a
dimensionless fitness value changes the ratio, so that conversion is part of
the definition and cannot be inferred from the value alone.
:::

:::{prf:theorem} Energy scales supplied by lengths and rates
:label: thm-mass-scales

For positive $\epsilon_c,\rho,\lambda,\gamma$ and a calibrated action unit,

$$
 E_c=\hbar_{\rm eff}c/\epsilon_c,\quad
 E_\rho=\hbar_{\rm eff}c/\rho,\quad
 E_\lambda=\hbar_{\rm eff}\lambda,\quad
 E_\gamma=\hbar_{\rm eff}\gamma
\tag{YM.30}
$$

are energy scales; the associated mass units are $E/c^2$. Their ordering is
exactly the ordering of $c/\epsilon_c,c/\rho,\lambda,\gamma$. A particle
mass identification requires a spectral or correlation measurement in that
channel.
:::

:::{prf:proof}
Each inverse time multiplies an action. Dividing all four quantities by the
same positive action proves the ordering statement. The parameters alone
impose no ordering of those four inverse times.
:::

:::{prf:definition} Correlation time and spatial correlation length
:label: def-correlation-length

An exponential temporal estimate $|C(t)|\le A e^{-\lambda t}$ has decay
time $\lambda^{-1}$. A spatial estimate $|C(r)|\le A'e^{-r/\xi}$ defines a
correlation-length bound $\xi$. Under a Euclidean rotation identification
$r=ct$ for that same correlation, $\xi=c/\lambda$ and the corresponding
energy scale is $\hbar_{\rm eff}c/\xi$. Distinct channels can have distinct
thresholds. The condition $\tau\lambda\ll1$ resolves the relaxation time
with many recorded steps.
:::

:::{prf:definition} Fine-structure convention
:label: def-fine-structure-constant-ym

In rationalized natural units a canonically normalized physical electric
charge has $\alpha=e^2/(4\pi)$. A parameter proxy yields
$\widehat\alpha=\widehat e^{\,2}/(4\pi)$ under the same convention.
Experimental identification of these quantities requires the charge and
field calibration; it is not fixed by the notation $e$.
:::

:::{prf:theorem} Dimensionless ratios in the scale dictionary
:label: thm-dimensionless-ratios

The dictionary obeys

$$
 \frac{E_c}{E_\rho}=\frac\rho{\epsilon_c},\qquad
 \frac{E_\lambda}{E_\gamma}=\frac\lambda\gamma,\qquad
 \frac{E_\lambda}{E_c}=\frac{\lambda\epsilon_c}{c}.
\tag{YM.31}
$$

These ratios remove the common action calibration, while retaining their
channel-identification conditions.
:::

:::{prf:proof}
Cancel the common factors in (YM.30).
:::

:::{prf:theorem} Scaling of the declared constants
:label: thm-rg-flow-constants

At fixed $m,c,\hbar_0$, the rescaling
$\tau\mapsto s\tau$, $\epsilon_c\mapsto\sqrt s\epsilon_c$,
$\rho\mapsto\sqrt s\rho$ preserves (YM.28) and $\rho/\epsilon_c$, and
sends $\widehat g_{2,\mathrm{clock}}^{\,2}\mapsto s\widehat g_{2,\mathrm{clock}}^{\,2}$.
If an ultraviolet momentum is proportional to $1/\rho$, this power law
has a different form from the logarithmic law (YM.27).
:::

:::{prf:proof}
Substitute the three rescalings into (YM.28) and (YM.29). With
$\mu\propto1/\rho$, one has $s\propto\mu^{-2}$ along this family,
whereas (YM.27) gives $g^2\sim1/\log\mu$. Thus this parameter rescaling
is an algebraic calibration family. A renormalization trajectory must instead
keep the chosen physical observables fixed and determine its running from them.
:::

:::{prf:definition} Observable validation targets
:label: def-experimental-signatures

A validation record specifies the link lift and orientations, measured loop
traces, action normalization, time and length conversions, and the correlation
channel. It includes uncertainty from (YM.21) or (YM.22) when their law and
regularity hypotheses hold. Field-current residuals include the sources in
(YM.11); fitness residuals use (YM.8). Reported chirality masks and scalar
phase observables retain the operational meanings of
{prf:ref}`thm-sm-ew-operator-layers`.
:::

(sec-ym-mass-gap)=
## 10. Discretization, coercivity, and survival of a gap

### 10.1. What regularization bounds imply

:::{prf:theorem} Noise floors and their coercivity scope
:label: thm-uv-protection-mechanism

A matrix bound $a_*I\preceq a(S)\preceq a^*I$ gives two-sided ellipticity
on the coordinates on which the noise acts. For kinetic noise acting only on
velocities it bounds the velocity form
$\int\nabla_v f^{\mathsf T}a\nabla_vf\,d\pi$. It supplies no positive
lower bound for the full-gradient form on functions depending only on $x$.
Even with a fixed positive velocity noise and friction, the relaxation rate
can tend to zero as spatial confinement weakens.
:::

:::{prf:proof}
The quadratic-form bounds follow by multiplying the matrix inequalities by
$\nabla_vf$. For $f=f(x)$, that gradient vanishes. For the last assertion
consider

$$
 dx_t=v_tdt,\qquad
 dv_t=-\kappa x_tdt-\gamma v_tdt+\sqrt{2\gamma T}\,dW_t,
 \qquad \kappa,\gamma,T>0.
$$

Its invariant Gaussian has position variance $T/\kappa$ and velocity variance
$T$. The deterministic drift eigenvalues solve
$r^2+\gamma r+\kappa=0$. For $4\kappa<\gamma^2$, the slow linear observable
decays at rate
$(\gamma-\sqrt{\gamma^2-4\kappa})/2\sim\kappa/\gamma$.
Thus this rate tends to zero with $\kappa$, while the velocity noise floor
remains fixed. Also, testing a position Poincaré inequality on $f=x$ requires
its constant to be at least $T/\kappa$.
:::

The full-gradient LSI (YM.1) is available from the actual-law criteria in
{prf:ref}`cor-n-uniform-lsi`; it is not left as an unspecified analytic input.
Its use for a kinetic time generator requires the extra evolution estimate
proved, for example, under the hypotheses of
{prf:ref}`thm-villani-hypocoercivity` or
{prf:ref}`thm-kl-contractive-diffusion-lsi`. A modified-norm hypocoercive
estimate must retain its norm comparison and prefactor. It does not make the
kinetic generator self-adjoint.

:::{prf:theorem} Time consistency for the specified transition family
:label: thm-correct-continuum-limit

Let $T_h=e^{hL}$ be a specified conservative semigroup and $P_h$ the actual
approximating transition family. In a weighted test norm assume

$$
 \|(P_h-T_h)T_s\varphi\|_V\le K_T h^{p+1},\quad
 \sup_{jh\le T}\|P_h^j\|_{V\to V}\le M_T,
 \qquad 0\le s\le T.
$$

Then, for $nh\le T$,

$$
 \|P_h^n\varphi-T_{nh}\varphi\|_V\le M_TK_TT h^p.
\tag{YM.32}
$$

The BAOAB and finite-rate splitting hypotheses in
{prf:ref}`thm-langevin-baoab-discretization-error` and
{prf:ref}`thm-full-system-discretization-error` supply concrete instances.
Fixed per-step cloning probabilities require a separate local consistency
calculation before a finite-rate continuous generator can be used.
:::

:::{prf:proof}
Expand the exact identity
$P_h^n-T_h^n=\sum_{j=0}^{n-1}P_h^j(P_h-T_h)T_h^{n-1-j}$.
Each term has norm at most $M_TK_Th^{p+1}$ and there are at most $T/h$
terms. This is the analytic telescoping proof of the cited discretization
results. Survival-conditioned laws additionally require control of the
normalizing survival probabilities, as in
{prf:ref}`lem-quantitative-qsd-perturbation`.
:::

Shrinking a reconstruction bandwidth at fixed algorithm parameters is the
consistency problem of {doc}`../convergence_program/16_continuum_discharge`.
Changing cloning width, locality radius, or noise strength changes the
transition family in (YM.32). A uniform rate along such a change requires
uniform analytic constants for that family.

### 10.2. A gap that passes to a limit

:::{prf:theorem} Survival of a uniformly controlled self-adjoint gap
:label: thm-mass-gap-rg-fixed-point

Let $H_a\ge0$ be self-adjoint on $\mathcal H_a$, with normalized
$\Omega_a\in\ker H_a$, and suppose

$$
 \|e^{-tH_a}(I-P_{\Omega_a})\|\le e^{-\lambda_*t},
 \qquad \lambda_*>0,
\tag{YM.33}
$$

uniformly in the cutoff and volume parameters under consideration.
Let $J_a:\mathcal H_a\to\mathcal H$ be isometries with
$J_aJ_a^*\to I$ strongly and $J_a\Omega_a\to\Omega$. Suppose, for every
$t>0$,

$$
 J_ae^{-tH_a}J_a^*\longrightarrow e^{-tH}
 \quad\hbox{strongly},
\tag{YM.34}
$$

where $H\ge0$ is self-adjoint. Then $H\Omega=0$ and
$\|e^{-tH}(I-P_\Omega)\|\le e^{-\lambda_*t}$. In particular
$\operatorname{spec}(H)\subset\{0\}\cup[\lambda_*,\infty)$ and
$\ker H=\mathbb C\Omega$. For a fixed calibrated $\hbar_{\rm eff}>0$,
the Hamiltonian $\hbar_{\rm eff}H$ has energy gap at least
$\hbar_{\rm eff}\lambda_*$.
:::

:::{prf:proof}
Since $e^{-tH_a}\Omega_a=\Omega_a$, (YM.34), contraction of each operator,
and $J_a\Omega_a\to\Omega$ imply $e^{-tH}\Omega=\Omega$.
For $f\in\mathcal H$ take $f_a=J_a^*f$. Then $J_af_a\to f$ and
$J_aP_{\Omega_a}f_a\to P_\Omega f$. Applying (YM.33) to $f_a$ and
passing to the limit gives

$$
 \|e^{-tH}(f-P_\Omega f)\|
 \le e^{-\lambda_*t}\|f-P_\Omega f\|.
$$

The spectral theorem excludes spectral support of $H$ in
$(0,\lambda_*)$ and any zero-energy vector orthogonal to $\Omega$.
Multiplication by the fixed action unit gives the energy statement.
:::

This theorem specifies the necessary Hilbert-space identification and
semigroup convergence. Pointwise graph-Laplacian consistency does not provide
(YM.34). Population-uniform estimates do not automatically control an
infinite spatial volume or a vanishing cutoff. Identification of this
Hamiltonian with the gauge-field transfer Hamiltonian is a further condition
on the field measure.

:::{prf:theorem} Three finite-scale estimates
:label: thm-triple-protection

The following bounds can be used together when their stated objects occur in
the transition family:

1. A frozen Ornstein–Uhlenbeck update
   $v'=e^{-\gamma h}v+\sqrt{T(1-e^{-2\gamma h})}\,\xi$ has
   $\mathbb E|v'|^2=e^{-2\gamma h}|v|^2+dT(1-e^{-2\gamma h})$.
2. A positive matrix $A\succeq a_*I$ has
   $\|A^{-1}\|_{\rm op}\le a_*^{-1}$.
3. A local weak defect and stability bound as in (YM.32) produce the displayed
   finite-time global error.

These estimates control a kinetic substep, an inverse matrix, and a specified
discretization error, respectively.
:::

:::{prf:proof}
The Gaussian increment is centered and independent, so its cross term with
$v$ vanishes and its squared norm has mean $d$. Diagonalize $A$ to obtain
the inverse bound. The third assertion is (YM.32). Uniform application along
a parameter family requires uniform $T$, $a_*^{-1}$, $K_T$, and $M_T$.
:::

(sec-ym-summary)=
## 11. Results and conditions for the field interpretation

The finite gauge construction now has normalized states, oriented comparisons,
Wilson invariance, explicit link equations, and exact Ward identities. The
analytic chapters supply specified-law concentration, time approximation,
QSD convergence, and mean-field tools. Applying those results preserves their
joint-law, survival, parameter, and geometry hypotheses.

A continuum quantum Yang–Mills construction additionally requires a nontrivial
limiting field law, its quantum reconstruction properties, and a uniform
physical gap for its transfer Hamiltonians. These are the objects in the
[Yang–Mills existence and mass-gap formulation of Jaffe and Witten](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf).
The following section proves several reconstruction ingredients directly and
states the precise remaining conditions, rather than identifying the
algorithmic relaxation rate with that physical gap.

(sec-qft-axioms-verification)=
## 12. Field axioms: finite proofs and reconstruction conditions

:::{div} feynman-prose
We already have an observable Hilbert space, a CAR representation, and an
explicit replica generator. These constructions identify the operators and the
dynamics represented by the recorded process. Here we examine their spacetime
properties and the conditions of the reconstruction theorems.

The LSI supplies a concrete starting point for empirical fields: after centering
and multiplying by $\sqrt N$, the smooth averages covered by its gradient bound
have moments uniformly bounded in $N$. This prevents a fixed collection of
those fluctuations from escaping to infinity. Identifying the limiting
covariance requires a further calculation with the dynamics.

Reflection positivity uses another feature of the same law: how its past and
future join at a time slice. Its proof below follows that joining explicitly,
so we can see which time evolution supplies the reconstructed energy operator.
:::

### 12.1. Fields, covariance, and the spectrum

:::{prf:definition} Recorded fields and their observable Hilbert space
:label: def-wightman-field-fg

For a slice with $N$ walker records at specified coordinates $X_i$, define
the intensive empirical field

$$
 \Phi_N(f)=\frac1N\sum_i a_i f(X_i),\qquad a_i\in\{0,1\}.
\tag{YM.35}
$$

Let $\pi_N$ be its actual law and let $\mathcal H_N=L^2(\pi_N)$.
Multiplication by $\Phi_N(f)$ defines a bounded operator for bounded $f$,
with norm at most $\|f\|_\infty$. The field observable space is
$\mathcal H_{\rm obs}=L^2(\sigma(\Phi_N(f):f\in\mathcal S),\pi_N)$.
An extensive convention replaces $\Phi_N$ by $N\Phi_N$.

For an episode with $K$ frames on a supplied spacetime chart, use instead
$\Phi_{N,K}(f)=(NK)^{-1}\sum_{k=1}^K\sum_{i=1}^N
a_{ki}f(t_k,x_{ki})$. Its empirical measure again has mass at most one,
so the same bounded-field arguments apply to its actual joint path law.
A one-frame LSI for $\pi_N$ controls episode observables only after an
appropriate path-law inequality or temporal estimate has been established.

A reconstructed quantum field, when available, is an operator-valued
distribution on its own common invariant dense domain. It must be supplied
with its spacetime covariance and locality properties. Multiplication by the
recorded density is the finite classical construction above.
:::

:::{prf:theorem} Tempered finite correlations and controlled limits
:label: thm-wightman-w0-fg

For every fixed $n$, the correlations of (YM.35) satisfy

$$
 |\mathbb E\Phi_N(f_1)\cdots\Phi_N(f_n)|
 \le\prod_{j=1}^n\|f_j\|_\infty.
\tag{YM.36}
$$

More generally their $n$-point distribution has absolute value at most
$\|F\|_\infty$ on $F\in\mathcal S((\mathbb R^D)^n)$. Thus it is tempered,
uniformly in $N$. Any distributional limit obeys the same bound.
If each $\Phi_N(f)$ converges in $L^1$ to a deterministic $\rho(f)$, the
limits of these intensive correlations factorize.
:::

:::{prf:proof}
The empirical measure in (YM.35) is positive with total mass at most one.
Its $n$-fold product therefore also has mass at most one, proving both bounds
by integration. The supremum norm is a Schwartz seminorm, so this proves
temperedness and its preservation in distributional limits.
For products, telescope one factor at a time and bound the other factors by
their supremum norms. Each error tends to zero in $L^1$, leaving
$\prod_j\rho(f_j)$. Consequently nontrivial fluctuation fields, such as
$\sqrt N(\Phi_N-\rho)$, need additional fluctuation-limit estimates; the
intensive law of large numbers alone gives deterministic limits.
:::

:::{prf:lemma} Moment bounds recovered from the full-gradient LSI
:label: lem-ym-lsi-moments

Under (YM.1), let $F$ be a real $L$-Lipschitz observable with finite mean on
the continuous joint state space, with $L>0$. Then

$$
 \log\mathbb E_\pi e^{t(F-\pi F)}\le C_*L^2t^2/2,
 \qquad
 \mathbb P(|F-\pi F|\ge r)\le2e^{-r^2/(2C_*L^2)}.
\tag{YM.37}
$$

In particular, for $k>0$,
$\mathbb E|F-\pi F|^k\le
2(2C_*L^2)^{k/2}\Gamma(1+k/2)$.
:::

:::{prf:proof}
First truncate $F$ smoothly. Apply (YM.1) to $e^{tF/2}$, and put
$\psi(t)=\log\mathbb E e^{tF}$. Dividing by $\mathbb E e^{tF}$ gives
$t\psi'(t)-\psi(t)\le C_*L^2t^2/2$.
Integrate the inequality for $(\psi(t)/t)'$, using
$\lim_{t\to0}\psi(t)/t=\pi F$. Applying the result to both signs of $F$
and optimizing Markov's inequality proves the two tails. Truncation passes
by the corresponding exponential bound and Fatou's lemma. Finally use
$\mathbb E|F-\pi F|^k=k\int_0^\infty r^{k-1}
\mathbb P(|F-\pi F|>r)dr$ and substitute
$u=r^2/(2C_*L^2)$. Status variables require their additional entropy control.
:::

:::{prf:corollary} Uniform moments of centered empirical fluctuations
:label: cor-ym-empirical-fluctuations

Use the actual joint law $\pi_N$ covered by (YM.1), with a common $C_*$.
For a real single-particle function $f$ with $\|\nabla f\|\le L_f$,
define $F_N=N^{-1}\sum_i f(S_i)$ and
$Z_N(f)=\sqrt N(F_N-\pi_NF_N)$. Then

$$
\begin{aligned}
\operatorname{Var}_{\pi_N}F_N&\le C_*L_f^2/N,\\
\log\mathbb E e^{tZ_N(f)}&\le C_*L_f^2t^2/2,\\
\mathbb P(|Z_N(f)|\ge r)&\le2e^{-r^2/(2C_*L_f^2)},\\
\mathbb E|Z_N(f)|^k&\le
2(2C_*L_f^2)^{k/2}\Gamma(1+k/2)\quad(k>0).
\end{aligned}
$$

For each fixed finite collection $f_1,\ldots,f_m$, the vectors
$(Z_N(f_1),\ldots,Z_N(f_m))$ form a tight family in $\mathbb R^m$.
The statements apply on the continuous state space of the cited LSI;
status-dependent or masked observables use their established entropy and
channel estimates in {prf:ref}`prop-sm-channel-estimate-routes`.

*Proof.* In the product gradient geometry, even for correlated $\pi_N$,

$$
\sum_i|\nabla_{S_i}F_N|^2
=N^{-2}\sum_i|\nabla f(S_i)|^2\le L_f^2/N.
$$

Poincaré gives the variance estimate. The Lipschitz constant of
$\sqrt N F_N$ is at most $L_f$, so
{prf:ref}`lem-ym-lsi-moments` gives the remaining scalar estimates, with
centering by the exact finite-$N$ mean. Constant observables have $Z_N=0$.
For a finite collection the union bound gives

$$
\mathbb P\left(\max_{j\le m}|Z_N(f_j)|>R\right)
\le\sum_{j:L_{f_j}>0}2e^{-R^2/(2C_*L_{f_j}^2)}\longrightarrow0
\quad(R\to\infty),
$$

uniformly in $N$. Compact cubes prove tightness. More generally the same
gradient calculation for $\sum_j t_jf_j$ bounds every linear combination
by Lipschitz constant $\sum_j|t_j|L_{f_j}$. Cauchy--Schwarz also gives
$|\operatorname{Cov}(Z_N(f),Z_N(g))|\le C_*L_fL_g$.
Concentration bounds this covariance but does not determine its limit.
Centering instead at a limiting mean $\rho(f)$ introduces the deterministic
term $\sqrt N(\pi_NF_N-\rho(f))$; its convergence is a separate bias estimate.
Spacetime distributional tightness additionally requires control over a
test-function topology, beyond a fixed finite collection. $\square$
:::

:::{prf:proposition} Established fermionic generator and transport to field coordinates
:label: prop-ym-fermionic-generator-transport

Let $P_t$ be the strongly continuous contraction semigroup on the centered
complete-swarm mode space in {prf:ref}`thm-lqft-record-fock-reconstruction`,
with generator $L$. Its established antisymmetric replica lift has

$$
L^{(k)}(f_1\wedge\cdots\wedge f_k)
=\sum_{r=1}^k f_1\wedge\cdots\wedge Lf_r\wedge\cdots\wedge f_k,
\qquad f_r\in\operatorname{Dom}L,\qquad L^{(0)}=0.
$$

Let $K$ denote the represented generator and $\mathcal W$ a specified unitary
onto a field Hilbert space. If $K_{\mathrm{field}}$ and
$\mathcal W K\mathcal W^{-1}$ generate strongly continuous semigroups and
agree on a common core, their semigroups, spectra, and transported bounded
operator correlations agree.

*Proof.* The difference of wedges has the telescoping expression

$$
\bigwedge_{j=1}^kP_tf_j-\bigwedge_{j=1}^kf_j
=\sum_{r=1}^k
P_tf_1\wedge\cdots\wedge P_tf_{r-1}
\wedge(P_tf_r-f_r)\wedge f_{r+1}\wedge\cdots\wedge f_k.
$$

Divide by $t$. Strong continuity and $(P_tf_r-f_r)/t\to Lf_r$, together
with continuity of the multilinear wedge map, give the generator formula.
Each $f_r$ is a function of the complete swarm, so $L$ retains its kinetic,
companion, and cloning interactions. The different factors are independent
whole-swarm replicas before antisymmetrization, as established in
{prf:ref}`prop-sm-replica-generator`.

If $\mathcal C$ is the common core, closedness gives

$$
K_{\mathrm{field}}
=\overline{K_{\mathrm{field}}|_{\mathcal C}}
=\overline{(\mathcal W K\mathcal W^{-1})|_{\mathcal C}}
=\mathcal W K\mathcal W^{-1}.
$$

Uniqueness of the generated semigroup then gives
$T_t^{\mathrm{field}}=\mathcal W T_t^{\mathrm{rec}}\mathcal W^{-1}$.
For every resolvent parameter,
$(z-K_{\mathrm{field}})^{-1}=\mathcal W(z-K)^{-1}\mathcal W^{-1}$,
which proves spectral equality. Transport a state vector and every bounded
observable by $\mathcal W$ in a finite matrix element; consecutive
$\mathcal W^{-1}\mathcal W$ factors cancel, proving correlation equality.
This is {prf:ref}`prop-sm-field-generator-comparison` in the present notation.
It uses the already constructed CAR and replica representation and requires
no choice of Dirac matrices. A self-adjoint transfer Hamiltonian is identified
when this comparison holds with $K_{\mathrm{field}}=-H$ for that Hamiltonian.
$\square$
:::

:::{prf:theorem} Covariance under a symmetry of the actual law
:label: thm-wightman-w1-fg

Suppose a group acts measurably on the configuration space, preserves its
probability law, and intertwines the chosen fields with the corresponding
spacetime action. Then $U_gF=F\circ g^{-1}$ is unitary on $L^2(\pi)$,
fixes $1$, and implements that field covariance. If $U_gF\to F$ in $L^2$
on a dense set as $g\to e$, the representation is strongly continuous.
For Poincaré covariance the group and field law must satisfy these hypotheses
for the full Poincaré group on Minkowski spacetime.
:::

:::{prf:proof}
Changing variables by the measure-preserving action gives
$\|U_gF\|_2=\|F\|_2$ and $U_gU_h=U_{gh}$. Conjugation of a multiplication
operator gives multiplication by its pullback, which is the assumed
transformed field. The constant function is fixed. Approximate an arbitrary
$L^2$ vector by a dense test vector and use $\|U_g\|=1$ to extend strong
continuity. A periodic spatial domain has its own isometry group; periodicity
alone supplies neither Lorentz boosts nor rotations exchanging space and time.
:::

:::{prf:lemma} Causal dependence criterion for recorded interventions
:label: lem-no-signaling-fg

In a finite acyclic update graph, suppose every updated variable is a
measurable function of its parent variables and its assigned exogenous noise.
If an intervention changes none of the variables or noises in the ancestral
set of an observable, its value is unchanged under the common-noise coupling.
Its law is therefore unchanged. Relativistic no-signaling follows from this
criterion when all actual dependencies are contained in the prescribed
Lorentzian causal cones.
:::

:::{prf:proof}
Order the ancestral vertices topologically. Source values and noises agree.
If the parent values of the next vertex agree, the common update function
and noise give the same value there. Induction proves equality at the
observable. All dependencies enter this induction: companion selection,
fitness statistics, cloning, and kinetic interactions as well as genealogy.
A Gaussian companion kernel with full support has dependencies beyond an
arbitrary finite spatial radius, so its support cannot be discarded in
asserting the cone hypothesis.
:::

:::{prf:theorem} Positive energy and the full spectrum condition
:label: thm-wightman-w2-fg

A self-adjoint nonnegative transfer generator $H$ supplies a nonnegative
energy operator. Suppose additionally that there is a strongly continuous
unitary representation of spacetime translations and proper orthochronous
Lorentz transformations, whose strongly commuting translation generators
are $(H,\mathbf P)$ and transform as a Lorentz vector. Then

$$
 \operatorname{spec}(H,\mathbf P)
 \subset\{(p^0,\mathbf p):p^0\ge|\mathbf p|\}
\tag{YM.38}
$$

in units $c=1$. A general nonreversible Markov generator is not identified
with this self-adjoint $-H$ without a separate construction.
:::

:::{prf:proof}
Unitary Lorentz covariance sends the joint spectral measure into its Lorentz
transform, so its support is Lorentz invariant. Positivity of $H$ excludes
negative $p^0$. If a spectral point satisfies $0\le p^0<|\mathbf p|$, a
boost with velocity $v$ along $\mathbf p$, chosen so
$p^0/|\mathbf p|<v<1$, sends its time coordinate to
$(p^0-v|\mathbf p|)/\sqrt{1-v^2}<0$. This contradicts positivity. Hence
(YM.38). Reversibility gives a self-adjoint Markov generator in its invariant
$L^2$ space; positive static entropy bounds alone do not give reversibility.
:::

:::{prf:theorem} Locality of the available operator constructions
:label: thm-wightman-w3-fg

The bounded multiplication fields (YM.35) commute for every pair of tests.
A reconstructed quantum field satisfies microcausality when its operators
commute on a common invariant domain for spacelike separated supports; this
is an additional property of that reconstruction. Equal-time multiplication
commutativity does not identify its Heisenberg time evolution.
:::

:::{prf:proof}
For any $\psi\in L^2(\pi)$,
$\Phi_N(f)\Phi_N(g)\psi=\Phi_N(g)\Phi_N(f)\psi$ because scalar
multiplication commutes. This proves the finite statement for all supports.
For a separately supplied field with the stated spacelike commutator
identity, integrating that identity against spacelike separated test supports
is precisely microcausality on the common domain. Unbounded operators need
this domain control; their formal symbols alone cannot be multiplied freely.
:::

:::{prf:corollary} Local gauge observables and even CAR observables
:label: cor-microcausality-gauge

In a tensor product of edge or vertex Hilbert spaces, bounded observables
supported on disjoint factors commute, including gauge-invariant combinations
with those supports. In the CAR construction of {doc}`03_lattice_qft`, even
observables with disjoint mode supports commute. For continuum spacelike
regions, the relation of such supports to spacelike locality must also hold.
:::

:::{prf:proof}
Operators on disjoint tensor factors can be interchanged. For CAR monomials
of even degrees $p,q$ with disjoint modes, moving every factor past the other
monomial produces $(-1)^{pq}=1$ and no contraction term. Extend by linearity
and norm closure. Gauge invariance restricts the observable set and preserves
these commutators; it does not by itself localize an extended loop.
:::

:::{prf:theorem} Cyclicity on the field observable space
:label: thm-wightman-w4-fg

The vector $\Omega=1$ is cyclic for the bounded algebra generated by the
fields (YM.35) on $\mathcal H_{\rm obs}$. Cyclicity on the full
$L^2(\pi_N)$ requires these fields to generate its full sigma algebra modulo
null sets.
:::

:::{prf:proof}
The bounded field algebra contains constants, is closed under complex
conjugation, and generates the sigma algebra defining $\mathcal H_{\rm obs}$.
For any finite list of these bounded real fields, polynomials are dense in
continuous functions on their compact joint range, and bounded measurable
functions follow by the monotone class argument. Cylinder functions then
span a dense subspace of the generated $L^2$ space. Acting on $1$ produces
exactly these functions. If a nonzero full-state function has conditional
mean zero given all fields, it is orthogonal to that subspace, so cyclicity
fails on the larger space. Position densities can, for example, omit velocity
information. QSD uniqueness is unnecessary for this proof.
:::

:::{prf:theorem} Restriction of a gap to an invariant gauge sector
:label: thm-gauge-sector-mass-gap

Let a self-adjoint $H\ge0$ on $L^2(\pi)$ have ground state $1$ and gap
$\lambda>0$. Suppose the compact gauge group acts unitarily, preserves $\pi$,
and commutes with $e^{-tH}$. Then its invariant subspace is preserved by
$e^{-tH}$, and every centered invariant vector satisfies the same gap bound.
The sector may contain no nonconstant vector; existence of a nontrivial gauge
excitation is a separate condition.
:::

:::{prf:proof}
Average the unitary representation against normalized Haar measure to obtain
the orthogonal projection $P_G$ onto invariant vectors. Commutation with the
semigroup gives $P_Ge^{-tH}=e^{-tH}P_G$. Restrict
$\|e^{-tH}(I-P_1)\|\le e^{-\lambda t}$ to its range. Equivalently, the
Rayleigh infimum over a smaller centered form domain is at least the global
infimum. An actual gauge transfer identification and the uniform convergence
in {prf:ref}`thm-mass-gap-rg-fixed-point` transfer this bound to its limiting
Hamiltonian. A QSD sampling gap without those identifications has the scope
of its own stochastic semigroup.
:::

### 12.2. Euclidean correlations and reflection positivity

:::{prf:definition} Euclidean correlations and the reflection form
:label: def-euclidean-correlator-fg

For a specified law $\mu$ of real Euclidean fields with the required moments,
set $S_n(f_1,\ldots,f_n)=\mathbb E_\mu\prod_j\Phi(f_j)$.
Let $\vartheta(t,x)=(-t,x)$ and let $\Theta F$ be the pullback of a field
functional by this time reflection. On an algebra $\mathcal A_+$ of bounded
functionals supported at nonnegative times, define

$$
 (F,G)_{\rm OS}=\mathbb E_\mu[\overline{\Theta F}\,G].
\tag{YM.39}
$$

Reflection positivity means $(F,F)_{\rm OS}\ge0$ for every
$F\in\mathcal A_+$, including every finite linear combination of such
functionals. One useful class of laws is a two-sided stationary reversible
Markov path law, with $t$ its Euclidean time coordinate.
:::

:::{prf:theorem} Euclidean regularity from actual moment bounds
:label: thm-os-os0-fg

If $S_n$ is continuous on the Schwartz test space with a bound uniform along
a proposed cutoff sequence, its distributional limits are tempered.
For intensive empirical fields (YM.35), (YM.36) supplies such a bound.
For smooth Lipschitz observables of a law satisfying (YM.1), (YM.37) supplies
explicit moments; continuity of their field reconstruction in the stated
Schwartz seminorm is also required.
:::

:::{prf:proof}
A uniform estimate $|S_n(F)|\le C_np_n(F)$ passes to the limit for every
test $F$, giving continuity of the limiting linear functional.
For the empirical construction take $p_n(F)=\|F\|_\infty$ and $C_n=1$.
For other reconstructions, Hölder's inequality combines the individual moment
bounds, once the stated continuity of the reconstruction supplies their test
seminorm dependence. Bounds at each $n$ are distinct from the growth in $n$
required by a chosen quantum reconstruction theorem.
:::

:::{prf:theorem} Euclidean covariance of an invariant field law
:label: thm-os-os1-fg

If the Euclidean field law is invariant under the pullback of a Euclidean
isometry $g$, then all of its correlations transform covariantly under $g$.
Full Euclidean invariance requires this equality for translations and all
orthogonal transformations of the supplied $D$-dimensional Euclidean space,
including rotations that mix time with space.
:::

:::{prf:proof}
Apply the change of variables $\Phi\mapsto g\Phi$ in
$\mathbb E\prod_j\Phi(f_j)$. Invariance replaces the transformed law by
the original law and transforms every test function by the corresponding
pullback. A spatially isotropic update supplies this equality only for the
spatial symmetries of the complete law. A confining potential can break
translations, and a periodic box has its own finite-volume isometries.
:::

:::{prf:lemma} Reflection factorization for a reversible Markov path law
:label: lem-transfer-matrix-fg

Let $(X_t)_{t\in\mathbb R}$ have a stationary reversible Markov law with
invariant probability $\pi$. For bounded future functionals define
$h_F(x)=\mathbb E[F\mid X_0=x]$. Then

$$
 \mathbb E[\overline{\Theta F}\,G]
 =\int\overline{h_F(x)}h_G(x)\,d\pi(x).
\tag{YM.40}
$$

If its strongly continuous Markov semigroup is $P_t$, then $P_t$ is a
self-adjoint contraction and $P_t=e^{-tH}$ for a self-adjoint $H\ge0$.
For a functional whose earliest time is $s\ge0$, write
$h_F=P_s v_F$, where $v_F$ is the conditional future functional translated
back by $s$. In particular
$(F,F)_{\rm OS}=\langle v_F,P_{2s}v_F\rangle_\pi\ge0$.
:::

:::{prf:proof}
Conditional on $X_0$, the past and future of a two-sided Markov process are
independent. Stationary reversibility identifies the conditional law of the
reflected past with that of the future. Their conditional expectations in
(YM.39) are therefore $\overline{h_F}$ and $h_G$, proving (YM.40).
Reversibility gives
$\langle f,P_tg\rangle_\pi=\langle P_tf,g\rangle_\pi$.
Jensen's inequality and invariance give $\|P_tg\|_2\le\|g\|_2$.
Moreover $\langle g,P_tg\rangle=\|P_{t/2}g\|_2^2\ge0$.
The self-adjoint strongly continuous semigroup has the spectral form
$e^{-tH}$ with $H\ge0$. The Markov property gives $h_F=P_sv_F$; applying
self-adjointness and the semigroup law proves the last formula.
:::

This is a direct construction for the same measure as its correlations.
For a reversible killed semigroup with a positive ground state, its Doob
transform supplies one candidate conservative process, but the invariant
measure is the transformed law (YM.18).

There is also a finite reversibility calculation for a frozen companion
selection. If $w_{ij}=w_{ji}\ge0$, $Z_i=\sum_jw_{ij}>0$, and
$P_{ij}=w_{ij}/Z_i$, then $q_i=Z_i/\sum_kZ_k$ satisfies
$q_iP_{ij}=q_jP_{ji}$. The resulting chain on companion indices is reversible.
This proves a property of that frozen chain; reversibility of the complete
swarm kernel requires every kinetic and cloning component as well.

:::{prf:theorem} Reflection positivity under the factorization condition
:label: thm-os-os2-fg

The path law in {prf:ref}`lem-transfer-matrix-fg` is reflection positive on
bounded future functionals. More generally, if the actual Euclidean field law
has a common factorization
$(F,G)_{\rm OS}=\langle h_F,h_G\rangle_{\mathcal K}$ with a linear map
$F\mapsto h_F$ into a Hilbert space, it is reflection positive on that algebra.
The full-gradient LSI of its one-time marginal does not imply this
factorization.
:::

:::{prf:proof}
For $F=\sum_jc_jF_j$, (YM.40) gives

$$
 (F,F)_{\rm OS}
 =\left\|\sum_jc_jh_{F_j}\right\|^2\ge0.
$$

This proves positivity of every finite Gram matrix and the general
factorization statement.

To check the last assertion explicitly, consider the stationary two-dimensional
Ornstein–Uhlenbeck process
$dX_t=(-I+\omega J)X_tdt+\sqrt2\,dW_t$, where
$J=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ and
$\omega>0$. Its invariant law is the standard Gaussian: its covariance
solves $(-I+\omega J)I+I(-I-\omega J)+2I=0$.
This law has full-gradient LSI constant one. Its stationary covariance is

$$
 \mathbb E[X_1(-t)X_1(t)]=e^{-2t}\cos(2\omega t).
\tag{YM.41}
$$

Indeed the deterministic propagator is $e^{-t}e^{\omega tJ}$. At
$t=\pi/(2\omega)$, (YM.41) is negative, so the reflection form for
$F=X_1(t)$ is negative. Bounded truncations converge to $F$ in $L^2$ and
have reflection forms converging to this negative value by Cauchy–Schwarz.
Thus even bounded future functionals violate reflection positivity for this
nonreversible law, despite its Gaussian marginal and static LSI.
:::

:::{div} feynman-prose
The rotating Gaussian gives a useful picture. At every instant its cloud is
perfectly round, with the same concentration bound as an ordinary Gaussian.
But a particle tends to rotate between observations. Reflection sees that
circulation; a one-time snapshot does not. This is why the path-law hypothesis
in the positivity proof matters.
:::

:::{prf:corollary} Positive quotient and its possible dimension
:label: cor-os2-nondegeneracy

Under reflection positivity, quotient $\mathcal A_+$ by its nullspace
$\mathcal N=\{F:(F,F)_{\rm OS}=0\}$ and complete it. The result is a
Hilbert space, isometrically the closure of $\{h_F\}$ when (YM.40) holds.
The class of $1$ has norm one. Infinite dimension requires arbitrarily large
positive-rank Gram matrices; it does not follow from positivity alone.
:::

:::{prf:proof}
Positivity of the form on $F+zG$ for every complex $z$ implies its
Cauchy–Schwarz inequality. Hence every null vector is orthogonal to every
vector, and the form descends to a positive definite inner product on the
quotient. Formula (YM.40) identifies its nullspace with the kernel of
$F\mapsto h_F$, proving the isometry. Normalization gives $(1,1)_{\rm OS}=1$.
The maximal number of linearly independent quotient vectors is exactly the
supremum of the ranks of their Gram matrices.
:::

:::{prf:theorem} LSI, compatible dynamics, and Euclidean clustering
:label: thm-os-os3-fg

Under (YM.1), the full-gradient Poincaré bound is

$$
 \operatorname{Var}_\pi f\le C_*\int|\nabla f|^2d\pi.
\tag{YM.42}
$$

If the actual stationary reversible generator has Dirichlet form
$\mathcal E(f,f)\ge a\int|\nabla f|^2d\pi$ for $a>0$, then its centered
semigroup norm is at most $e^{-at/C_*}$. Its temporal correlations satisfy
(YM.20) with $M=1$ and $\lambda=a/C_*$. Spatial Euclidean clustering follows
when the full spacetime covariance and the rotation of the observables required
in {prf:ref}`thm-cluster-decomposition` also hold.
:::

:::{prf:proof}
For bounded centered $g$, insert $f=1+\varepsilon g$ into (YM.1).
Expansion gives
$\operatorname{Ent}((1+\varepsilon g)^2)
=2\varepsilon^2\operatorname{Var}g+o(\varepsilon^2)$.
Divide by $2\varepsilon^2$ and pass to zero; closure extends the bound to
the form domain. The assumed form comparison then gives
$\operatorname{Var}g\le(C_*/a)\mathcal E(g,g)$.
For centered $g$,
$d\|P_tg\|_2^2/dt=-2\mathcal E(P_tg,P_tg)
\le-2a\|P_tg\|_2^2/C_*$.
Integration proves the semigroup bound. Apply (YM.20) and the stated
rotation argument. A velocity-only kinetic form fails the form comparison
on nonconstant $g(x)$, so this proof does not assign it that rate.
:::

:::{prf:theorem} Symmetry of bosonic correlations
:label: thm-os-os4-fg

Correlations of real commuting random fields are symmetric under simultaneous
permutation of their tests. This property passes to their distributional
limits. For Grassmann or CAR fields the corresponding rule is graded and
uses the chosen fermionic algebra.
:::

:::{prf:proof}
Scalar multiplication gives
$\prod_j\Phi(f_j)=\prod_j\Phi(f_{\sigma(j)})$ for every permutation.
Taking expectation and then a distributional limit preserves the identity.
For disjoint fermionic generators each interchange contributes its
anticommutation sign, as proved in {doc}`03_lattice_qft`. Permuting walker
labels does not introduce that algebraic sign into a classical density product.
:::

### 12.3. Conditions on cutoff and infinite-volume limits

:::{prf:theorem} Passage of correlation identities to a controlled limit
:label: thm-infinite-volume-limit

Let $S_{n,a,L}$ be a common sequence of Euclidean field correlations, with
cutoff $a$ and volume scale $L$. Suppose:

1. For every test $F$, the values $S_{n,a,L}(F)$ are Cauchy along the chosen
   joint limit, with $|S_{n,a,L}(F)|\le C_np_n(F)$ for a common Schwartz
   seminorm.
2. Reflection Gram forms are nonnegative on a fixed common test algebra and
   all entries of those forms are among the converging correlations.
3. Bosonic permutation identities hold, and covariance defects for each fixed
   Euclidean isometry and fixed test tend to zero.
4. For separated test clusters, a clustering estimate holds uniformly in
   $a,L$, with an error tending to zero with their separation.

Then the limiting correlations are tempered, reflection positive, symmetric,
Euclidean covariant, and clustering with that bound. Applying an
Osterwalder–Schrader reconstruction theorem additionally requires its full
regularity and growth conditions for this same family, and a nontrivial
limiting observable sector. A uniform transfer gap passes under the separate
Hilbert-space convergence hypotheses of
{prf:ref}`thm-mass-gap-rg-fixed-point`.
:::

:::{prf:proof}
The Cauchy condition defines a limit for each test, and the common seminorm
bound makes the limit a continuous linear functional. Every reflection
quadratic form is a finite sum of its correlation entries; its limit is
nonnegative. Permutation identities are linear equalities and therefore pass
to the limit. The covariance defect hypothesis gives the transformed test
identity in the limit. For fixed separation the uniform clustering bound also
passes to the limit. Taking separation to infinity then gives clustering.
These arguments prove precisely the listed properties. They do not exchange
limits without a uniform bound or supply the additional reconstruction growth
conditions.
:::

The necessary regularity in a quantum reconstruction is stronger than a list
of fixed-order moment bounds. It must be checked against the chosen version
of the [Osterwalder–Schrader reconstruction theorem](https://link.springer.com/article/10.1007/BF01608978),
which specifies conditions for analytic continuation to relativistic fields.
The direct Markov proof above constructs a positive time-transfer space under
reversibility; full spacetime covariance and field-domain control enter the
relativistic construction separately.

The analytic $N$-uniform estimates are useful inputs to condition 1 and to
observable errors, when the law and reconstruction match. They do not bound
the constants uniformly in an independently increasing volume. For example,
on a circle of length $L$ with uniform probability, the test
$f(x)=\cos(2\pi x/L)$ has
$\operatorname{Var}f=1/2$ and
$\int|f'|^2=(2\pi/L)^2/2$. Its Poincaré constant is therefore at least
$L^2/(4\pi^2)$, despite compactness at each $L$.

### 12.4. Local algebras and their conditional spacetime structure

:::{prf:definition} Local observable algebras
:label: def-local-algebra-fg

For a bounded spacetime region $\mathcal O$, let $\mathcal B(\mathcal O)$
be a specified self-adjoint set of bounded field observables supported there,
and define

$$
 \mathfrak A(\mathcal O)=\mathcal B(\mathcal O)'',\qquad
 \mathfrak A=\overline{\bigcup_{\mathcal O}\mathfrak A(\mathcal O)}^{\|\cdot\|},
\tag{YM.43}
$$

where the regions form a directed family. For self-adjoint unbounded smeared
fields use their bounded spectral functions, or their unitary exponentials
when defined, as generators. An unbounded operator itself is not an element
of a von Neumann algebra of bounded operators.
:::

:::{prf:theorem} Isotony
:label: thm-hk-isotony-fg

If local generator sets are nested under inclusion of regions, then
$\mathcal O_1\subset\mathcal O_2$ implies
$\mathfrak A(\mathcal O_1)\subset\mathfrak A(\mathcal O_2)$.
:::

:::{prf:proof}
Every von Neumann algebra containing $\mathcal B(\mathcal O_2)$ contains
$\mathcal B(\mathcal O_1)$. Intersect all such algebras, or use monotonicity
of the double commutant, to obtain the inclusion. Test functions supported
in the smaller region give the required inclusion of generator sets.
:::

:::{prf:theorem} Locality from commuting bounded generators
:label: thm-hk-locality-fg

If every bounded generator in $\mathcal B(\mathcal O_1)$ commutes with every
bounded generator in $\mathcal B(\mathcal O_2)$ for spacelike separated
regions, then the generated von Neumann algebras commute. For unbounded
fields this requires commuting bounded spectral functions, not only a formal
commutator on a domain.
:::

:::{prf:proof}
The generator hypothesis gives
$\mathcal B(\mathcal O_1)\subset\mathcal B(\mathcal O_2)'$.
The commutant on the right is a von Neumann algebra, so it contains
$\mathcal B(\mathcal O_1)''$. Its elements therefore commute with every
second-region generator. Their commutant is again a von Neumann algebra and
contains $\mathcal B(\mathcal O_2)''$. This proves the conclusion without
assuming joint weak-operator continuity of multiplication, which need not
hold.
:::

:::{prf:theorem} Covariance of the local net
:label: thm-hk-covariance-fg

Suppose a strongly continuous unitary group representation $U_g$ sends the
bounded generators of $\mathcal O$ onto those of $g\mathcal O$. Then
$\alpha_g(A)=U_gAU_g^{-1}$ is an automorphism of the net and

$$
 \alpha_g(\mathfrak A(\mathcal O))=\mathfrak A(g\mathcal O).
$$

For each fixed bounded $A$, $g\mapsto\alpha_g(A)$ is strongly operator
continuous. Full Poincaré covariance uses the representation required in
{prf:ref}`thm-wightman-w1-fg` and {prf:ref}`thm-wightman-w2-fg`.
:::

:::{prf:proof}
Unitary conjugation preserves sums, products, adjoints, inverses, and
commutants; applying it to the double commutant proves the net identity.
For a fixed vector $\psi$, insert and subtract $U_gAU_{g_0}^{-1}\psi$
to bound the continuity error by the strong continuity of $U_g$ and of
$U_g^{-1}$, with the fixed norm $\|A\|$. The group law gives
$\alpha_g\alpha_h=\alpha_{gh}$.
:::

:::{prf:theorem} Spectrum of a covariant local net
:label: thm-hk-spectrum-fg

Under the strongly commuting translation and Lorentz covariance hypotheses of
{prf:ref}`thm-wightman-w2-fg`, the implementing energy-momentum spectrum of
the net lies in the forward cone. If its actual Hamiltonian also satisfies
the gap hypotheses of {prf:ref}`thm-mass-gap-rg-fixed-point`, its vacuum
sector has the corresponding positive energy gap in the limit.
:::

:::{prf:proof}
The net uses the same implementing unitaries, hence the same joint spectral
measure as in (YM.38). The Lorentz-boost argument there excludes every
spectral point outside the forward cone. The self-adjoint semigroup argument
in (YM.33)--(YM.34) gives the stated additional gap. Spatial periodicity or
positivity of one time generator alone supplies only the corresponding
finite-volume symmetry or positive-energy assertion.
:::

:::{prf:theorem} Ground-state uniqueness in the specified representation
:label: thm-hk-vacuum-fg

Let $H\ge0$, $H\Omega=0$, $\|\Omega\|=1$, and suppose
$\|e^{-tH}(I-P_\Omega)\|\le e^{-\lambda t}$ with $\lambda>0$.
Then $\Omega$ is the unique ground-state vector up to a scalar, and
$\omega_0(A)=\langle\Omega,A\Omega\rangle$ is a normalized positive
state. If a symmetry fixes $\Omega$ and implements the net, $\omega_0$ is
invariant under it. Uniqueness of the ground vector does not imply uniqueness
among all symmetry-invariant states.

Separately, if the complete killed kernel is equivariant under a group and
has a unique QSD $\nu_N$, that QSD is group invariant. This is an assertion
about the killed stochastic process, independent of identifying a field vacuum.
:::

:::{prf:proof}
If $H\psi=0$, write $\psi=c\Omega+\psi_\perp$. The gap estimate gives
$\|\psi_\perp\|\le e^{-\lambda t}\|\psi_\perp\|$ for every $t$, hence
$\psi_\perp=0$. Positivity follows from
$\omega_0(A^*A)=\|A\Omega\|^2\ge0$, and normalization and invariance
follow from $\|\Omega\|=1$ and $U_g\Omega=\Omega$.
For example, $H=\operatorname{diag}(0,1)$ has a unique ground vector, but
both energy eigenvector states on $M_2(\mathbb C)$ are invariant under its
time-translation automorphisms. This proves the stated distinction.

For the stochastic assertion, push the QSD identity forward by a symmetry.
Kernel equivariance shows the pushed measure satisfies the same killed
eigenmeasure identity with the same survival factor. It is a QSD, so
uniqueness makes it $\nu_N$. This is the proof of
{prf:ref}`thm-qsd-exchangeability`, applied to the specified group. No Gibbs
density or compactification is used.
:::

The positive state admits its usual cyclic representation: quotient the
observable algebra by vectors of zero $\omega_0(A^*A)$ norm, use
$\langle[A],[B]\rangle=\omega_0(A^*B)$, and complete. The class of $1$ is
cyclic by construction. Ground-state and spacetime properties of this
representation require the respective hypotheses above.

(sec-spectral-gap-variational)=
## 13. Optimizing a specified relaxation gap

:::{div} feynman-prose
You can optimize a sampler by making its slowest mode decay faster. First you
must fix the clock: multiplying every transition rate by ten makes the gap ten
times larger without changing the stationary distribution. Once the allowed
parameters, the gap, and the time unit are fixed, existence of an optimizer is
a theorem under compactness and continuity. Relating that optimizer to nature
is a separate physical conjecture.
:::

:::{prf:definition} Fixed-clock gap-selection model
:label: def-maximal-convergence-selection

Let $K$ be the admissible parameter set. For each $p\in K$, let $L_p$ be the
declared generator in one common time coordinate, and let
$\lambda_{\rm gap}(p)\in[0,\infty)$ be its specified relaxation gap. For a
reversible generator with invariant law $\pi_p$, one possible choice is the
Poincare gap

$$
 \lambda_{\rm gap}(p)=
 \inf_{\substack{f\in\mathsf D(\mathcal E_p)\\
                         \operatorname{Var}_{\pi_p}(f)>0}}
 \frac{\mathcal E_p(f,f)}{\operatorname{Var}_{\pi_p}(f)}.
$$

For a self-adjoint transfer generator $H_p\ge0$ with ground-space projection
$P_{0,p}$, the same notation denotes the lower edge of the spectrum of
$H_p$ restricted to $\operatorname{Ran}(I-P_{0,p})$. The gap notion, generator
family, clock, and parameterization are part of the model. Define the
optimizer set by

$$
 \mathcal P_*:=\operatorname*{argmax}_{p\in K}\lambda_{\rm gap}(p).
 \tag{YM.44}
$$

The selection model chooses a member of $\mathcal P_*$ whenever this set is
nonempty. A parameter-dependent rescaling $L_p\mapsto c(p)L_p$ changes the
objective and is therefore excluded by the fixed-clock declaration.
:::

:::{prf:theorem} Existence and uniqueness of a fixed-clock gap optimizer
:label: thm-maximal-convergence-existence

Assume that $K$ is a nonempty compact metric space, that every $L_p$ in
{prf:ref}`def-maximal-convergence-selection` uses the same time coordinate,
and that $p\mapsto\lambda_{\rm gap}(p)$ is finite and continuous on $K$. Then
$\mathcal P_*$ is nonempty and compact, and every $p_*\in\mathcal P_*$
satisfies

$$
 \lambda_{\rm gap}(p_*)=\max_{p\in K}\lambda_{\rm gap}(p).
$$

If, in addition, $K$ is convex in a normed vector space and
$\lambda_{\rm gap}$ is strictly concave on $K$, then $\mathcal P_*$ is a
singleton. The theorem is a statement about the specified mathematical
family; it does not identify a physical parameter.
:::

:::{prf:proof}
By compactness and continuity, the extreme-value theorem gives a point
$p_*$ at which $\lambda_{\rm gap}$ attains its supremum. Thus
$\mathcal P_*$ is nonempty. It is the inverse image of the closed singleton
$\{\lambda_{\rm gap}(p_*)\}$, so it is closed in the compact set $K$ and is
therefore compact. If two distinct points $p_0,p_1$ were maximizers, strict
concavity would give

$$
 \lambda_{\rm gap}(tp_0+(1-t)p_1)
 >t\lambda_{\rm gap}(p_0)+(1-t)\lambda_{\rm gap}(p_1)
 =\lambda_{\rm gap}(p_0)
$$

for $0<t<1$, contradicting maximality.
:::

:::{prf:theorem} Stability of the gap optimizer under objective errors
:label: thm-maximal-convergence-stability

Let $K$ be compact, let $\lambda_{\rm gap}$ be continuous with unique
maximizer $p_*$, and let $\widetilde\lambda$ be another continuous objective
on the same fixed-clock parameter set. If

$$
 \sup_{p\in K}|\widetilde\lambda(p)-\lambda_{\rm gap}(p)|\le\varepsilon,
$$

then every $\widetilde p\in\operatorname*{argmax}_{p\in K}
\widetilde\lambda(p)$ satisfies

$$
 0\le\lambda_{\rm gap}(p_*)-\lambda_{\rm gap}(\widetilde p)
 \le 2\varepsilon.
 \tag{YM.44a}
$$

If a neighborhood $U$ of $p_*$ has a strict margin

$$
 \lambda_{\rm gap}(p_*)-\sup_{p\in K\setminus U}\lambda_{\rm gap}(p)
 \ge\eta_U>0,
$$

then $\widetilde p\in U$ whenever $2\varepsilon<\eta_U$. More quantitatively,
if for some metric $d$ and $m>0$

$$
 \lambda_{\rm gap}(p_*)-\lambda_{\rm gap}(p)
 \ge\frac m2d(p,p_*)^2\qquad(p\in K),
$$

then $d(\widetilde p,p_*)\le2\sqrt{\varepsilon/m}$.
:::

:::{prf:proof}
Optimality of $\widetilde p$ and the uniform error bound give

$$
 \lambda_{\rm gap}(p_*)-\lambda_{\rm gap}(\widetilde p)
 \le [\lambda_{\rm gap}(p_*)-\widetilde\lambda(p_*)]
   +[\widetilde\lambda(\widetilde p)-\lambda_{\rm gap}(\widetilde p)]
 \le2\varepsilon.
$$

The margin condition rules out $K\setminus U$ when $2\varepsilon<\eta_U$.
Combining the quadratic margin with (YM.44a) gives
$m d(\widetilde p,p_*)^2/2\le2\varepsilon$, which is the final bound.
:::

The continuity hypothesis can be checked in a concrete model by transporting
the specified invariant observables and their law through the orbit and
measure isomorphisms
{prf:ref}`thm-sm-direct-orbit-isomorphism` and
{prf:ref}`thm-sm-direct-measure-isomorphism`, and then applying the existing
gap and convergence estimates in
{prf:ref}`thm-sm-direct-existing-machinery`. This transport preserves the
represented observables and their correlations; it does not supply the
continuity estimate automatically.

:::{prf:conjecture} Physical maximal-convergence identification
:label: conj-physical-maximal-convergence

After the gauge/field observable representation has been specified, the
physical parameter $p_{\rm phys}$ is conjectured to obey

$$
 p_{\rm phys}\in\mathcal P_*.
 \tag{YM.44b}
$$

This is a physical identification, not a consequence of
{prf:ref}`thm-maximal-convergence-existence` or of the Fractal Gas gap
estimates. The orbit and measure isomorphisms
{prf:ref}`thm-sm-direct-orbit-isomorphism` and
{prf:ref}`thm-sm-direct-measure-isomorphism` establish the mathematical
correspondence between the invariant recorded observables and the auxiliary
field representation. They do not prove that nature selects the maximizing
parameter. That final identification remains conjectural and requires a
declared physical parameter map, common time normalization, and empirical or
independent theoretical support.
:::

(sec-strong-cp-solution)=
### 13.1. A conditional theta-selection calculation

:::{prf:theorem} Theta maximization under a controlled spectral expansion
:label: thm-strong-cp-spectral

Suppose a specified gap-selection model has a $2\pi$-periodic parameter
$\theta$ and

$$
 \lambda(\theta)=\lambda_0-\kappa(1-\cos\theta)+R(\theta),
 \quad \kappa>0,
 \quad |R(\theta)-R(0)|\le r_\kappa(1-\cos\theta),
 \quad r_\kappa<\kappa.
\tag{YM.45}
$$

Then its unique maximizing parameter is $\theta=0$ modulo $2\pi$.
One sufficient remainder condition is that $R$ is even, periodic, twice
continuously differentiable, and
$\|R''\|_\infty\le C\kappa^2$ with $C\pi^2\kappa/4<1$.
Identification of this parameter with the physical strong-CP parameter requires
an actual gauge-theory spectral family obeying (YM.45) and the selection model
(YM.44); that identification is the conjecture
{prf:ref}`conj-physical-maximal-convergence`.
:::

:::{prf:proof}
Subtract the value at zero. The remainder bound gives

$$
 \lambda(\theta)-\lambda(0)
 \le-(\kappa-r_\kappa)(1-\cos\theta),
$$

which is strictly negative unless $\theta=0$ modulo $2\pi$.
For the sufficient condition restrict by periodicity to $|\theta|\le\pi$.
Evenness gives $R'(0)=0$, so Taylor's integral remainder gives
$|R(\theta)-R(0)|\le C\kappa^2\theta^2/2$.
Concavity of sine on $[0,\pi/2]$ gives
$\sin(|\theta|/2)\ge|\theta|/\pi$, hence
$1-\cos\theta\ge2\theta^2/\pi^2$.
Take $r_\kappa=C\pi^2\kappa^2/4$ to obtain (YM.45).
:::

:::{prf:lemma} The sign of theta dependence is not fixed by reversibility
:label: lem-ym-theta-gap-examples

Let a two-state continuous-time chain have generator

$$
 L_\theta=\begin{pmatrix}-r(\theta)&r(\theta)\\r(\theta)&-r(\theta)\end{pmatrix},
 \qquad r(\theta)=r_0\pm\kappa\cos\theta,
 \qquad r_0>\kappa>0.
$$

Both choices are reversible with respect to the same uniform probability and
are even in $\theta$. Their gaps are $2r(\theta)$. The plus choice has its
gap maximum at zero; the minus choice has its minimum there.
:::

:::{prf:proof}
The rows sum to zero and the off-diagonal rates are positive. The two basis
vectors $(1,1)$ and $(1,-1)$ have eigenvalues $0$ and $-2r(\theta)$.
Symmetry of the matrix proves reversibility; the cosine proves evenness.
The locations of the maxima and minima follow immediately.
:::

For a differentiable finite-dimensional self-adjoint family
$H_\theta$ with simple normalized eigenvectors $u_j(\theta)$,
differentiating $H_\theta u_j=E_j u_j$ and taking the inner product with
$u_j$ gives $E_j'=\langle u_j,H_\theta'u_j\rangle$.
Thus the derivative of a spectral gap $E_1-E_0$ is the difference of two
matrix elements, with no universal sign. A theta-dependent Euclidean phase
weight also need not be a positive Markov transition law. An instanton
amplitude or a symmetry argument alone therefore cannot substitute for the
spectral estimate (YM.45).

:::{prf:corollary} Scope of the theta-selection model
:label: cor-no-axions

Under (YM.45) and the selection model (YM.44), the specified model selects
$\theta=0$ without adding another variable to that optimization problem.
An inference about the physical strong-CP problem or the presence or absence
of axion fields requires the additional physical identifications stated in
{prf:ref}`thm-strong-cp-spectral`.
:::

:::{prf:proof}
Apply the strict maximization result to the stated one-parameter family.
The result concerns the parameters of that family; it contains no statement
about the spectrum or field content of a different physical theory.
:::

:::{prf:remark} Dependency order of the Yang--Mills comparisons
:label: rem-ym-proof-dependency-order

The Fractal Set codec and record coverage precede the invariant-coordinate
and measure isomorphisms. Frame-link flatness is finite matrix algebra.
The complete-kernel likelihood precedes conditional descriptor integration,
which in turn gives the field-density and support comparisons. The CAR and
replica construction precedes its generator differentiation and the
common-core comparison with any specified field evolution. Uniform
ellipticity, full-gradient LSI, and channel fluctuation estimates enter with
the laws, coefficient conventions, and domains already established upstream.
The Wilson variation, independent-plaquette area law, and smooth-connection
continuum limit apply to their specified link sectors. Reflection positivity
and the limiting transfer gap apply to their stated path laws and operators.
Thus no Wilson law, field Hamiltonian, or continuum conclusion is used to
prove the record representation subsequently compared with it.
:::

## References and further reading

The finite constructions and analytic estimates are developed in
{doc}`01_fractal_set`, {doc}`02_causal_set_theory`, {doc}`03_lattice_qft`,
{doc}`04_standard_model`, and the convergence chapters cited above.
The perturbative group coefficient used in (YM.26) is the specialization of
{prf:ref}`cor-sm-beta-functions` to zero matter content.

The distinction between a classical Yang–Mills action and the quantum
existence-and-gap problem follows the
[Jaffe–Witten formulation](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf).
The analytic continuation requirements are those of
[Osterwalder and Schrader, *Axioms for Euclidean Green's functions II*](https://link.springer.com/article/10.1007/BF01608978).
