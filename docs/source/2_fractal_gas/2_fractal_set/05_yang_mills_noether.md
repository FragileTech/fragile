# Discrete Yang–Mills Actions, Noether Identities, and Quantum Reconstruction

## TLDR

:::{div} feynman-prose
The Fractal Set supplies oriented CST, IG, and IA transports together with
recorded companion doublets. On an interaction triangle, the holonomy measures
the mismatch between IA and IG transport. Its Wilson readout is a squared
matrix difference. The doublet energy retains both amplitude differences
and the angular mismatch after transport.

Their field law comes from the complete recorded history. Average the
algorithm's path likelihood over records with the same descriptor and take
the negative logarithm: this is its effective action. Integrating recorded
loop and doublet readouts against that law reproduces their algorithmic
correlations. Alternating insertions of recorded modes faithfully realize their
exterior algebra. The recorded transition also constructs a unital completely
positive CAR evolution whose two-point functions reproduce the recorded
correlations. Compression to selected channels has exact evolution equations
with memory from the discarded coordinates.

For the stationary joint laws covered by the full-gradient LSI, centered
empirical fluctuations at the $\sqrt N$ scale have a common subsequential
distributional hierarchy. The complete kinetic and cloning update supplies
their drift and covariance equations. The paired momentum bound uses this
same stationary LSI family; the collision calculation explicitly tests why
its law must be identified.

The same LSI energy constructs a self-adjoint equilibrium transfer, reflection
positivity, and a temporal gap. Its equilibrium time has its own generator;
recorded kinetic time retains the algorithm's transition law. For the established
product and uniformly bounded whole-joint tilt families, the equilibrium
fluctuations have a subsequential continuum limit with an infinite-dimensional
OS space and a transfer gap. The product case has an explicitly identified
Gaussian hierarchy. Bounded tilts also identify the limiting empirical law;
their fluctuation laws retain the dependence on their joint dynamics.
Finite mode approximations converge to the equilibrium transfer of the proved
mean-field marginal. A compressed equilibrium channel closes precisely when
its observable space is invariant; hard masks retain bounded correlations
even when they have infinite gradient energy. Relativistic Yang--Mills
identification retains the spacetime and field requirements specified below.
:::

(sec-ym-intro)=
## 1. Objects and analytic inputs

:::{div} feynman-prose
Start with a complete recorded swarm history. Its oriented interaction
transports tell us how to move a doublet between vertices. Products around
interaction loops give holonomies; comparing transported doublets gives
their amplitude and angular energy. Keeping the complete history likelihood
assigns probabilities to these field readouts, including their correlations
across selected updates.

The effective action expresses that same likelihood in descriptor
coordinates. The established gradient energy also constructs an equilibrium
transfer for its specified one-time law. Each calculation below uses the
corresponding fields, law, and time evolution, so its matrix elements have
a definite meaning for the recorded observables.
:::

Write $d$ for spatial dimension and $D=d+1$ for spacetime dimension. A finite
Fractal Set supplies vertices, recorded attributes, and several kinds of edges.
An oriented two-complex additionally specifies closed face boundaries; a face is
not determined by an unordered collection of nearby vertices. The constructions
below use this incidence data and a declared representation of a compact group.
The direct field action and its time correlations are fixed by the complete
algorithm likelihood in {ref}`(YM.N3) <eq-fg-ym-n3>`--{ref}`(YM.N4) <eq-fg-ym-n4>`.

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
The Wilson functional is evaluated under this recorded law. Its normalized
partition and characteristic functional are
{prf:ref}`def-partition-function-ym`. The displayed Dirac expression is a
coordinate action template; the native matter observables and their evolution
are the recorded CAR construction in {doc}`03_lattice_qft`. All uses of the transfer and reconstruction results
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

(eq-fg-ym-1)=
$$
\operatorname{Ent}_{\pi_N}(f^2)
 \le 2C_*\int\sum_i(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2)\,d\pi_N.
\tag{YM.1}
$$

The constant is uniform in $N$ when the chosen joint law meets one of the
criteria in {prf:ref}`cor-n-uniform-lsi`. Discrete alive/dead strata require the
additional entropy term in {prf:ref}`prop-kl-status-entropy`. A velocity-only
form cannot replace the right-hand side of {ref}`(YM.1) <eq-fg-ym-1>` for spatial observables.

(sec-ym-symmetry)=
## 2. Internal frames and dressed states

### 2.1. Groups acting on the finite data

:::{prf:definition} Local frame group and relabeling group
:label: def-hybrid-gauge-structure-ym

Let $\mathcal V$ be the vertex set of the recorded Fractal Set and use
the $G=SU(2)\times U(1)$ doublet and phase representations specified in
{doc}`04_standard_model` on its recorded fibers.
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

Use the implemented allowed companion set $A_i$, its conditional
probabilities $P_i(k)\ge0$ with $\sum_{k\in A_i}P_i(k)=1$, and the
recorded phase potentials $\theta_{ik}$ of
{prf:ref}`def-fractal-set-phase-potential`. Their amplitude encoding is

$$
 |\psi_i\rangle=\sum_{k\in A_i}\sqrt{P_i(k)}e^{i\theta_{ik}}|k\rangle
 \in\mathbb C^N.
$$

Excluded companions have zero coordinates. For a recorded ordered companion
pair $(i,j)$, set $p$ to its implemented conditional clone probability in
{ref}`(YM.Z60) <eq-fg-ym-z60>`. Its normalized role doublet is

$$
 |\Psi_{ij}\rangle=
 \sqrt p\,|\uparrow\rangle\otimes|\psi_i\rangle+
 \sqrt{1-p}\,|\downarrow\rangle\otimes|\psi_j\rangle
 \in\mathbb C^2\otimes\mathbb C^N.
$$

The squared norm is $p+(1-p)=1$, since the two internal basis vectors are
orthogonal. The symmetric case occurs at $p=1/2$.
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

(eq-fg-ym-2)=
$$
 \mathcal S_h(y_0,\ldots,y_K)
 =-\log r_0(y_0)-\sum_{k=0}^{K-1}\log p_h(y_k,y_{k+1}).
\tag{YM.2}
$$

In the particular Euler Gaussian model
$Y_{k+1}=Y_k+h b(Y_k)+\sigma\sqrt h\,\xi_k$ on $\mathbb R^m$,
$\xi_k\sim N(0,I)$, the negative log density relative to Lebesgue increments
is

(eq-fg-ym-3)=
$$
 \sum_{k=0}^{K-1}
 \left[\frac{|Y_{k+1}-Y_k-hb(Y_k)|^2}{2\sigma^2h}
       +\frac m2\log(2\pi\sigma^2h)\right].
\tag{YM.3}
$$

For a fixed $C^1$ path $y:[0,T]\to\mathbb R^m$, continuous $b$, and
$Kh=T$, the first sum in {ref}`(YM.3) <eq-fg-ym-3>` evaluated at $y(kh)$ converges to
$(2\sigma^2)^{-1}\int_0^T|\dot y-b(y)|^2dt$.
:::

:::{prf:proof}
Conditional multiplication of the one-step Radon–Nikodym derivatives gives
{ref}`(YM.2) <eq-fg-ym-2>`. The normal density gives {ref}`(YM.3) <eq-fg-ym-3>`, including its normalization. On a
fixed $C^1$ path, the difference quotients converge uniformly to $\dot y$;
continuity of $b$ on the compact image of that path makes the quadratic sum a
Riemann sum.

This last calculation concerns smooth comparison paths. Typical diffusion
paths have nonzero quadratic variation. For example, the expected quadratic
sum in {ref}`(YM.3) <eq-fg-ym-3>` along the Gaussian process is $mK/2$, which diverges as
$h\downarrow0$. The smooth-path formula is therefore not an ordinary density
on differentiable sample paths. Singular deterministic substeps and cloning
atoms must retain their appropriate kernel reference in {ref}`(YM.2) <eq-fg-ym-2>`.
:::

:::{div} feynman-prose
Build the action where the algorithm generates its probabilities: companion
draws, cloning decisions, position jitter, and kinetic noise. Each draw is
conditioned on the complete preceding record. Multiplying its likelihood
ratio by those of the other stages gives the history likelihood; taking
the negative logarithm gives the additive recorded action.

The Gaussian kinetic term measures the increment using the inverse
covariance. On the identified Hessian branch, this is the fitness-metric
quadratic form. Its determinant term comes from the same Gaussian
normalization and stays in the calculation. To obtain the field action,
integrate the full likelihood over records that give the same geometry and
gauge channels, then take the logarithm. The order of these operations
retains the probability carried by unresolved records.

The following face lemma supplies a different calculation within this same
law: it expands the recorded Wilson observable into a quadratic expression
with an explicit remainder. That expansion estimates the face readout;
the conditional history likelihood determines its probability and the
effective action.
:::

:::{prf:theorem} Action emergence from the recorded stochastic dynamics
:label: thm-ym-recorded-action-emergence

Use the complete staged transition record of
{prf:ref}`thm-sm-instantiated-record-transition`, retaining the intermediate
arrays, companion choices, clone masks, and random inputs used by the
configured update. The action in {prf:ref}`thm-action-from-path-integral`
has the following explicit realization for that algorithm.

At stochastic stage $\ell$, let $w_\ell$ be its sampled output and
$\mathcal H_\ell$ its complete preceding record. Write $p_\ell$ for the
actual conditional density and $q_\ell$ for the reference conditional
density on the same carrier. Deterministic stages retain their actual
conditional Dirac kernel in both laws. A normalized dominating reference
$R$ can be constructed with the actual initial law, positive uniform
probabilities for each finite categorical draw, fair probabilities for
recorded Bernoulli draws, and standard normal densities for nondegenerate
Gaussian increments. All other random stages retain their actual
conditional kernels. Zero-noise stages retain their deterministic kernels.
Then, on the actual support,

(eq-fg-ym-z59)=
$$
\mathcal L=\frac{dP}{dR}
 =\prod_\ell\frac{p_\ell(w_\ell\mid\mathcal H_\ell)}
                         {q_\ell(w_\ell\mid\mathcal H_\ell)},
\qquad
\mathcal S=-\log\mathcal L
 =\sum_\ell\bigl[-\log p_\ell+\log q_\ell\bigr],
\qquad \mathbb E_R e^{-\mathcal S}=1.
\tag{YM.Z59}
$$

In particular, the following are the actual negative log densities
$-\log p_\ell$ before subtracting the reference action:

(eq-fg-ym-z60)=
$$
\begin{aligned}
A_{\mathrm{comp}}&=-\log\kappa_\ell(j\mid\mathcal H_\ell),\\
A_{\mathrm{gate}}&=-\log\left[p_i^{c_i}(1-p_i)^{1-c_i}\right],\\
p_i&=\begin{cases}
\displaystyle\min\!\left(1,\max\!\left(0,
 \frac{V_{j_i}-V_i}{p_{\max}(V_i+\epsilon_{\mathrm{clone}})}\right)\right),
 &i\text{ alive},\\
1,&i\text{ dead},
\end{cases}\\
A_{\mathrm{jitter},i}
 &=\frac{|x_i'-x_{j_i}|^2}{2\sigma_x^2}
          +\frac d2\log(2\pi\sigma_x^2),\qquad c_i=1,\ \sigma_x>0,\\
A_{\mathrm O}
 &=\frac12 n^{\mathsf T}C^{-1}n
          +\frac12\log\det C+\frac m2\log(2\pi),
\qquad n=v^+-c_1v^-,\quad C=\Sigma\Sigma^{\mathsf T}.
\end{aligned}
\tag{YM.Z60}
$$

Here $\kappa_\ell$ is the implemented conditional companion law, $c_i$
is the recorded clone decision, and $x_{j_i}$ is the companion position
before simultaneous replacement. The last line uses the $m$ active
nondegenerate velocity coordinates and the actual O-stage coefficient
$\Sigma$ of {prf:ref}`thm-ym-kinetic-metric-correspondence`.
For a singular Gaussian stage, use its sampled standard Gaussian input
and the deterministic output map as the record; this stage then has its
actual seed law as reference. No inverse of a singular covariance occurs.

For the already recorded geometry and gauge descriptor $Y=(G,D)$, the
joint and conditional gauge actions are consequently

(eq-fg-ym-z61)=
$$
\begin{aligned}
e^{-S^{\mathrm{joint}}(g,d)}
 &=\mathbb E_R\!\left[
       \prod_\ell\frac{p_\ell}{q_\ell}\,\middle|\,G=g,D=d\right],\\
e^{-S^{\mathrm{fiber}}(g,d)}
 &=\frac{\mathbb E_R[\prod_\ell(p_\ell/q_\ell)\mid G=g,D=d]}
         {\mathbb E_R[\prod_\ell(p_\ell/q_\ell)\mid G=g]}.
\end{aligned}
\tag{YM.Z61}
$$

Thus the field action is obtained by integrating the complete algorithmic
likelihood over unresolved records. The reference in
{ref}`(YM.Z59) <eq-fg-ym-z59>` supplies a concrete choice for the
reference of {prf:ref}`thm-sm-path-descriptor-density` and
{prf:ref}`thm-ym-native-geometry-fiber-action`; both subsequent
constructions use this same choice.
:::

:::{prf:proof}
**1. Ordered conditional kernels.** At a finite recorded horizon there
are finitely many draws. Treat each draw as a separate stage, including
conditional draws within a collision group. At each history the stated
reference assigns positive mass to every possible categorical outcome
and positive density to every nondegenerate Gaussian output. The actual
law is therefore absolutely continuous with respect to it. A deterministic
update has identical conditional support in both laws even when its map
is not invertible. Multiplying the conditional densities proves
{ref}`(YM.Z59) <eq-fg-ym-z59>`. Integrating the last stage gives
$\int(p_\ell/q_\ell)q_\ell=1$ at each preceding history. Repeating
backwards gives normalization. This argument retains all adaptive
history dependence; no independence of companion choices is required.
If another initial reference is used, include its actual density $r_0$
as in {ref}`(YM.2) <eq-fg-ym-2>`.

**2. Cloning and persistence.** Conditional on the companion and the
fitness array, the implemented uniform threshold gives
$\mathbb P(c_i=1\mid\mathcal H_\ell)=p_i$. Its two probabilities
are $p_i$ and $1-p_i$, giving the second line of
{ref}`(YM.Z60) <eq-fg-ym-z60>`. Interpret $0^0=1$ here; an impossible
outcome has infinite action and an enforced outcome has zero negative
log probability. For a fair reference the gate contribution to
$\mathcal S$ is $A_{\mathrm{gate}}-\log2$, including enforced gates.
If raw thresholds are retained, factor their joint law by first sampling
the gate and then its conditional threshold: uniform on $[0,p_i)$ for
a cloning outcome and on $[p_i,1]$ for persistence. Retain this latter
conditional kernel in both laws, assigning any probability kernel at
impossible outcomes. Its likelihood ratio is one on the actual support,
so the gate formula also holds for the complete threshold record.
The Gaussian cloning displacement is
$x_i'-x_{j_i}=\sigma_x\zeta_i$, so the normal change of variables gives
$A_{\mathrm{jitter},i}$. A persistent walker has a Dirac update and no
jitter-density term. Retaining collision randomness in both conditional
kernels gives likelihood ratio one at that stage, while its resulting
velocities still enter every subsequent kernel and descriptor. Thus
ratio one does not remove the collision dynamics from the path law.

**3. Kinetic metric and normalization.** The O update is
$v^+=c_1v^-+\Sigma\xi$, with its coefficients fixed by the preceding
record and $\xi\sim N(0,I_m)$. Its Gaussian density follows from
$|\det\Sigma|=\sqrt{\det C}$ and
$|\Sigma^{-1}n|^2=n^{\mathsf T}C^{-1}n$. Against the standard normal
increment reference this gives the explicit relative contribution

(eq-fg-ym-z62)=
$$
A_{\mathrm O}+\log q_{\mathrm O}(n)
 =\frac12n^{\mathsf T}(C^{-1}-I)n+\frac12\log\det C.
\tag{YM.Z62}
$$

For the Hessian diffusion branch already identified in
{prf:ref}`thm-ym-kinetic-metric-correspondence`,
$C=c_2^2G^{-1}$ on the retained coordinates. Substitution yields

(eq-fg-ym-z63)=
$$
A_{\mathrm O}
 =\frac{n^{\mathsf T}Gn}{2c_2^2}
       +m\log c_2-\frac12\log\det G+\frac m2\log(2\pi).
\tag{YM.Z63}
$$

This identifies the fitness metric in the native quadratic action,
including its determinant term. The B, A, and configured deterministic
rotation stages retain their conditional maps. They are not replaced
by an overdamped position transition. The Gaussian-input construction
for singular stages proves the same path identity without a Lebesgue
density on their outputs.

**4. Survival and observable reconstruction.** At a prescribed event
$E$ with $P(E)>0$, replace $\mathcal L$ by
$\mathbf1_E\mathcal L/P(E)$, giving action
$\mathcal S+\log P(E)$ on $E$ and $+\infty$ off $E$.
Complete record encoding preserves these factors by
{prf:ref}`thm-sm-instantiated-record-transition`. For any bounded
measurable $F$, conditioning the likelihood gives

$$
\mathbb E_P F(G,D)
 =\mathbb E_R[F(G,D)e^{-\mathcal S}]
 =\int F(g,d)\mathbb E_R[e^{-\mathcal S}\mid g,d]\,
                  (G,D)_*R(dg,dd).
$$

Conditioning once more on $G$ proves
{ref}`(YM.Z61) <eq-fg-ym-z61>` and its fiber normalization. This is
exactly the action in {ref}`(YM.Z52) <eq-fg-ym-z52>`, now with every
nontrivial likelihood factor specified by the implemented update.
In particular one integrates $e^{-\mathcal S}$ before taking the
logarithm; replacing this operation by an average of $\mathcal S$
would change the field law.
:::

:::{prf:lemma} Quantitative curvature expansion of the recorded face action
:label: lem-ym-recorded-face-action-remainder

For a recorded unitary face holonomy $U_P=e^{iX_P}$ in an $r$-dimensional
representation, choose a Hermitian logarithm $X_P$. With the coefficients
and faces of {prf:ref}`def-wilson-action-ym`,

(eq-fg-ym-z64)=
$$
\left|S_W-\sum_P\frac{\beta_P}{2r}\operatorname{Tr}X_P^2\right|
 \le\sum_P\frac{\beta_P}{24r}\operatorname{Tr}X_P^4
 \le\frac1{24}\sum_P\beta_P\|X_P\|_{\mathrm{op}}^4.
\tag{YM.Z64}
$$

For a specified local curvature approximation $Y_P=gA_PF_P$ with
$\|X_P-Y_P\|_{\mathrm{op}}\le\eta_P$, its quadratic expression obeys

(eq-fg-ym-z65)=
$$
\left|S_W-\sum_P\frac{\beta_Pg^2A_P^2}{2r}
                         \operatorname{Tr}F_P^2\right|
 \le\sum_P\beta_P\left[
 \frac{\|X_P\|_{\mathrm{op}}^4}{24}
 +\frac{\eta_P(\|X_P\|_{\mathrm{op}}+\|Y_P\|_{\mathrm{op}})}2
 \right].
\tag{YM.Z65}
$$

These are pathwise estimates for the recorded face observable under the
same algorithmic law used in {ref}`(YM.Z61) <eq-fg-ym-z61>`.
:::

:::{prf:proof}
Diagonalize $X_P$ with real eigenvalues $x_a$. Taylor's theorem gives
$|1-\cos x-x^2/2|\le |x|^4/24$ for every real $x$.
Sum this inequality over the $r$ eigenvalues, divide by $r$, multiply by
$\beta_P\ge0$, and sum over faces. This proves
{ref}`(YM.Z64) <eq-fg-ym-z64>` without a small-angle assumption.
Cyclicity of trace gives
$\operatorname{Tr}(X^2-Y^2)=\operatorname{Tr}((X-Y)(X+Y))$;
$|\operatorname{Tr}(AB)|\le r\|A\|_{\mathrm{op}}\|B\|_{\mathrm{op}}$
then proves {ref}`(YM.Z65) <eq-fg-ym-z65>`. Thus a continuum use of the
face expansion retains an explicit summed remainder as well as the
geometric quadrature error. The expansion concerns the observable $S_W$;
the logarithm defining the native action remains the conditional
likelihood in {ref}`(YM.Z61) <eq-fg-ym-z61>`.
:::

:::{prf:lemma} Reversible diffusion and its ground-state transform
:label: lem-ym-ground-state-transform

Let $\pi(dx)=Z^{-1}e^{-U(x)/T}dx$ on $\mathbb R^m$, $T>0$, with $Z<\infty$,
$U\in C^2$, and a conservative reversible realization of
$L=T\Delta-\nabla U\cdot\nabla$. On smooth compactly supported functions the
unitary map $\mathcal U f=\pi^{1/2}f$ gives

(eq-fg-ym-4)=
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
order derivatives cancel, leaving {ref}`(YM.4) <eq-fg-ym-4>`. The closed nonnegative form
transports unitarily to the form of $H$. Since $L1=0$, its image
$\mathcal U1=\pi^{1/2}$ has energy zero.
:::

A Feynman–Kac weight $e^{-\int V(Y_t)dt}$ defines a killed or tilted process
for a specified potential $V$. Pairwise cloning reproduces that law only when
its transition probabilities establish the required identification. The
kinetic, generally nonreversible Fractal Gas requires its own full kernel in
{ref}`(YM.2) <eq-fg-ym-2>`. Formula {ref}`(YM.4) <eq-fg-ym-4>` also shows why a stochastic potential cannot simply be
copied into a Schrödinger action: derivatives of the potential enter the
transformation.

### 3.2. Chosen matter fields

:::{prf:definition} Matter action and mass matrix
:label: def-matter-lagrangian-ym

On a supplied Lorentzian spin geometry, choose a Dirac field with an internal
doublet and write, in units $\hbar=c=1$,

(eq-fg-ym-5)=
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

:::{prf:definition} Interaction connection carried by the Fractal Set
:label: def-gauge-field-from-phases

The gauge transport data are the IG, CST, and IA edge variables of
{prf:ref}`def-fractal-set-gauge-connection`, evaluated on the complete
interaction record. In the cloning-doublet sector they are
$U_{\mathrm{IG}}^{(2)},U_{\mathrm{IA}}^{(2)}\in SU(2)$ and
$U_{\mathrm{CST}}^{(2)}=I$ in temporal gauge, with
$U_{-e}^{(2)}=(U_e^{(2)})^{-1}$. The scalar phase sector uses the
corresponding recorded IG, CST, and IA phase variables.

A matrix $U_{ij}$ transports the coordinates at $j$ into the fiber at $i$.
Its products follow the oriented interaction boundary. The complete
algorithm likelihood determines the law of these transport descriptors
through {prf:ref}`thm-sm-effective-recorded-gauge-dynamics`; no separate
link sampling law is introduced here.

For local continuum coordinates, $\mathcal A(\xi)$ denotes the
first-order coefficient in
$U_{i,i+h\xi}=I+h\mathcal A(\xi)+O(h^2)$ when this expansion is supplied
by the reconstruction estimates for the same edge variables. Its unitary,
determinant-one identities make $\mathcal A$ anti-Hermitian and traceless.
The finite formulation uses the recorded edge products directly.
:::



:::{div} feynman-prose
Follow the two interaction transports around a recorded triangle. In temporal
gauge, its loop product compares the IA transport with the IG transport.
The Wilson readout measures their mismatch by a squared matrix norm, so it
can be calculated directly from these edge transports.

Now transport a companion doublet from one vertex to the other. Its mismatch
has two contributions: the amplitudes can differ, and the transported
directions can differ. Equal nonzero amplitudes still leave angular energy
when those directions disagree. The identity below keeps both contributions and
shows how their common gauge transformation preserves the readout.

To compute a correlation, retain the complete algorithm likelihood and
average it over histories giving the same field descriptor. Its negative
logarithm supplies the effective action. This gives the loop and matter
readouts their probabilities and time correlations through the same selected
updates that generated the record.
:::

:::{prf:proposition} Interaction holonomy and angular matter energy of the recorded connection
:label: prop-ym-recorded-interaction-sector

Use the CST, IG, and IA transports of
{prf:ref}`def-fractal-set-gauge-connection`, with inverse reversal and
their recorded edge orientations. In temporal gauge the interaction
triangle has $H_\triangle=U_{\mathrm{IA}}U_{\mathrm{IG}}^\dagger$.
Its curvature readout is the actual transport mismatch

(eq-fg-ym-n1)=
$$
w_\triangle
=1-\tfrac12\operatorname{Re}\operatorname{Tr}H_\triangle
=\tfrac14\|U_{\mathrm{IA}}-U_{\mathrm{IG}}\|_{\mathrm F}^2.
\tag{YM.N1}
$$

Let $d_i=r_i z_i$ be the normalized decomposition of the recorded
companion doublets in {prf:ref}`def-sm-direct-companion-doublet`, and
let $U_{ij}$ transport the doublet at $j$ into the fiber at $i$.
For these actual fields and transports,

(eq-fg-ym-n2)=
$$
\begin{aligned}
\|U_{ij}d_j-d_i\|^2
&=r_i^2+r_j^2-2r_ir_j\operatorname{Re}(z_i^\dagger U_{ij}z_j)\\
&=(r_j-r_i)^2
  +r_ir_j\|U_{ij}z_j-z_i\|^2.
\end{aligned}
\tag{YM.N2}
$$

The second term retains the angular mismatch between the transported
cloning doublets. Both this readout and the triangle readout are
invariant under simultaneous local frame changes of fields and edge
transports. Their complete time correlations are evaluated with the
algorithm-derived descriptor law of
{prf:ref}`thm-sm-effective-recorded-gauge-dynamics`.
:::

:::{prf:proof}
The triangle product and the Frobenius-norm expansion in
{prf:ref}`prop-sm-attribution-holonomy-defect` give
{ref}`(YM.N1) <eq-fg-ym-n1>`. Unitarity and $\|z_i\|=\|z_j\|=1$
give

$$
\begin{aligned}
\|r_jU_{ij}z_j-r_iz_i\|^2
&=r_j^2z_j^\dagger U_{ij}^\dagger U_{ij}z_j
 +r_i^2z_i^\dagger z_i
 -r_ir_j(z_i^\dagger U_{ij}z_j+z_j^\dagger U_{ij}^\dagger z_i),\\
\|U_{ij}z_j-z_i\|^2
&=2-2\operatorname{Re}(z_i^\dagger U_{ij}z_j).
\end{aligned}
$$

Substitution gives both equalities in
{ref}`(YM.N2) <eq-fg-ym-n2>`. If
$z_i'=\Omega_i z_i$ and $U_{ij}'=\Omega_iU_{ij}\Omega_j^{-1}$,
then $U_{ij}'z_j'-z_i'=\Omega_i(U_{ij}z_j-z_i)$, preserving its
norm. A closed transport product transforms by conjugation at its
basepoint, preserving its trace. Thus both readouts descend to their
recorded gauge-invariant descriptors.

Write $\mathscr D$ for the complete descriptor history containing these
readouts, $R$ for the already defined reference history law, and
$\mathcal L$ for the complete algorithm likelihood. Its field measure and
effective action are

(eq-fg-ym-n3)=
$$
a(y)=\mathbb E_R[\mathcal L\mid\mathscr D=y],\qquad
S^{\mathrm{eff}}(y)=-\log a(y),\qquad
\lambda=\mathscr D_*R.
\tag{YM.N3}
$$

For any finite product $F$ of bounded recorded loop and doublet readouts,
conditional expectation gives the exact dynamical correspondence

(eq-fg-ym-n4)=
$$
\mathbb E_P F(\mathscr D)
=\mathbb E_R[\mathcal L F(\mathscr D)]
=\int F(y)e^{-S^{\mathrm{eff}}(y)}\lambda(dy).
\tag{YM.N4}
$$

This uses the full path likelihood, including its initial factor and every
selected update. The same identity holds for any integrable such product.
Successive prefix densities give the actual predictive kernel by
{ref}`(SM.G2) <eq-fg-sm-g2>`, so the loop action and its time correlations
refer to the same algorithmic evolution. The Wilson readout
{ref}`(YM.N1) <eq-fg-ym-n1>` and the matter readout
{ref}`(YM.N2) <eq-fg-ym-n2>` are observables of this effective law;
its action is calculated from {ref}`(YM.N3) <eq-fg-ym-n3>`.
:::

:::{div} feynman-prose
Keep one recorded history fixed and evaluate its fitness Hessian twice.
On the full Hessian branch below, clipping and the positive metric floor
turn that discrepancy into explicit bounds on the metric, noise covariance,
and volume weights. With the same cells and masks retained, these bounds
control how much a geometrically weighted field readout changes.

The error has two contributions: changed site values and changed normalized
weights. Replacing one factor at a time then bounds the error in any finite
recorded correlation. Centered fluctuations retain their chosen normalization,
including the factor $\sqrt N$ when it is used.

Geometry and fields here come from the same history. Conditioning the
complete algorithm likelihood on their joint descriptor preserves that
dependence and recovers the existing field law by marginalization. Thus the
geometric error and the correlation error are measured under one recorded law.
:::

:::{prf:theorem} Same-record metric reconstruction and gauge-field correlations
:label: thm-ym-same-record-metric-field

Consider the full Hessian branch identified in
{prf:ref}`prop-geometry-clipped-metric`, on its successful, unshifted
solver branch, with $c_2>0$. Write $\epsilon=\epsilon_\Sigma>0$. For every recorded
site, its metric and normalized noise covariance are

(eq-fg-ym-z1)=
$$
g_i=\epsilon I+(H_{i,\mathrm{sym}})_+,
\qquad D_i=c_2^{-2}\Sigma_i\Sigma_i^T=g_i^{-1}.
\tag{YM.Z1}
$$

All the following comparisons are on the same realized history. Let
$\widehat H_i$ be a second evaluation of that history's fitness Hessian,
put $\widehat g_i=\epsilon I+(\widehat H_{i,\mathrm{sym}})_+$, and define
$\delta=\max_i\|\widehat H_{i,\mathrm{sym}}-H_{i,\mathrm{sym}}\|_{\mathrm F}$.
Thus $\delta$ is the actual reconstruction discrepancy, not a new
regularity or convergence condition. Then

(eq-fg-ym-z2)=
$$
\begin{aligned}
\|\widehat g_i-g_i\|_{\mathrm F}&\le\delta,&
\|\widehat D_i-D_i\|_{\mathrm F}&\le\epsilon^{-2}\delta,\\
\|\widehat g_i^{-1/2}-g_i^{-1/2}\|_{\mathrm F}
&\le\frac{\delta}{2\epsilon^{3/2}},&
q^{-1}\le\frac{\sqrt{\det\widehat g_i}}{\sqrt{\det g_i}}
&\le q,\qquad q=(1+\delta/\epsilon)^{d/2}.
\end{aligned}
\tag{YM.Z2}
$$

Use the finite, nonnegative Euclidean Voronoi volumes and the retained-site
mask of `compute_riemannian_volume_weights`, writing their product as
$b_i$. On a nonempty retained frame with $\sum_i b_i>0$, define the
normalized geometric readout

(eq-fg-ym-z3)=
$$
v_i=b_i\sqrt{\det g_i},\quad
p_i=\frac{v_i}{\sum_jv_j},\quad
\Phi=\sum_i p_i B_i,\qquad
\widehat\Phi=\sum_i\widehat p_i\widehat B_i.
\tag{YM.Z3}
$$

Here $B_i$ is a bounded gauge-invariant readout at the recorded site,
including its test-function factor. Examples are the interaction-loop
readout in {ref}`(YM.N1) <eq-fg-ym-n1>` and bounded functions of the
angular matter readout in {ref}`(YM.N2) <eq-fg-ym-n2>`. The hatted
weights use the same $b_i$ and $\widehat g_i$. If
$|B_i|,|\widehat B_i|\le M$ and
$\eta=\max_i|\widehat B_i-B_i|$, set
$e=\min\{2M,\eta+M(q^2-1)\}$. Then, pathwise,

(eq-fg-ym-z4)=
$$
\sum_i|\widehat p_i-p_i|\le\min\{2,q^2-1\},
\qquad |\widehat\Phi-\Phi|\le e.
\tag{YM.Z4}
$$

If reconstruction also changes the recorded cell volumes or mask to
$\widehat b_i$, retain the original walker-slot indexing and compute

$$
t=\frac{\sum_i|\widehat b_i-b_i|\sqrt{\det\widehat g_i}}
        {\sum_i b_i\sqrt{\det\widehat g_i}}.
$$

When both normalizations are nonzero, the same conclusions hold with
$e=\min\{2M,\eta+M(q^2-1+2t)\}$. This includes the actual
retessellation discrepancy instead of inferring it from fill distance.
The empty retained frame has readout zero by definition; if only one
of the two normalizations vanishes, use the bound $e=2M$.
For comparisons across the complete history law, let $A$ denote the
recorded event on which the branch calculation above applies and both
cell normalizations are positive and finite. On $A^c$ keep the actual bounded readouts
and set $e=2M$. Thus $|\widehat\Phi-\Phi|\le e$ holds on the
entire history space, and
$\mathbb E_Pe\le\mathbb E_P[\mathbf1_A
\min\{2M,\eta+M(q^2-1+2t)\}]+2M P(A^c)$,
with $t=0$ for a fixed tessellation.
For $n$ such readouts at arbitrary recorded times, evaluated under the
same actual history law $P$ of
{prf:ref}`thm-sm-effective-recorded-gauge-dynamics`,

(eq-fg-ym-z5)=
$$
\left|\mathbb E_P\prod_{k=1}^n\widehat\Phi_k
       -\mathbb E_P\prod_{k=1}^n\Phi_k\right|
\le\sum_{k=1}^n\mathbb E_P e_k\prod_{j\ne k}M_j.
\tag{YM.Z5}
$$

For any deterministic normalization $a_N\ge0$, centering each field at
its own expectation gives the explicit fluctuation estimate, for $p\ge1$,

(eq-fg-ym-z6)=
$$
\left\|a_N\big[(\widehat\Phi-\mathbb E_P\widehat\Phi)
                  -(\Phi-\mathbb E_P\Phi)\big]\right\|_{L^p(P)}
\le2a_N\|e\|_{L^p(P)}.
\tag{YM.Z6}
$$

Finally, let $\mathscr G$ record these metrics, volumes, masks and sites,
and let $\mathscr D$ be the field descriptor history in
{ref}`(YM.N3) <eq-fg-ym-n3>`. Their joint effective law is determined by
that same algorithm likelihood $\mathcal L=dP/dR$:

(eq-fg-ym-z7)=
$$
\begin{gathered}
Y=(\mathscr G,\mathscr D),\qquad \lambda_Y=Y_*R,\qquad
 a_Y(y)=\mathbb E_R[\mathcal L\mid Y=y],\\
\mathbb E_P F(Y)=\int F(y)a_Y(y)\,\lambda_Y(dy),\qquad
\mathbb E_R[a_Y(Y)\mid\mathscr D]
 =\mathbb E_R[\mathcal L\mid\mathscr D]=a(\mathscr D).
\end{gathered}
\tag{YM.Z7}
$$

This holds for every integrable $F(Y)$, including the finite products in
{ref}`(YM.Z5) <eq-fg-ym-z5>`. The joint law therefore recovers the
previous field law by marginalization; it does not multiply independently
sampled geometry and field laws.
:::

:::{prf:proof}
The covariance identity {ref}`(YM.Z1) <eq-fg-ym-z1>` is exactly
{prf:ref}`prop-geometry-clipped-metric`. To obtain its quantitative
stability, equip real symmetric matrices with the Frobenius inner
product. For $A=A_+-A_-$, $A_\pm\succeq0$ and $A_+A_-=0$. For every
$C\succeq0$,

$$
\langle A-A_+,C-A_+\rangle
=-\operatorname{Tr}(A_-C)\le0.
$$

Apply this inequality to $(A,C)=(A,B_+)$ and $(B,A_+)$ and add.
It gives
$\|A_+-B_+\|_{\mathrm F}^2
\le\langle A_+-B_+,A-B\rangle$.
Cauchy--Schwarz proves contraction of the positive-part map, including
at repeated and zero eigenvalues. Since both metrics are bounded below
by $\epsilon I$, the exact identity

$$
\widehat g^{-1}-g^{-1}
=\widehat g^{-1}(g-\widehat g)g^{-1}
$$

proves the inverse bound. The spectral theorem applied to positive
matrices gives

$$
g^{-1/2}=\frac1\pi\int_0^\infty t^{-1/2}(g+tI)^{-1}\,dt.
$$

Subtract the two integrals and use the same inverse identity under the
integral. The resulting bound is

$$
\frac{\delta}{\pi}\int_0^\infty
 \frac{t^{-1/2}}{(\epsilon+t)^2}\,dt
=\frac{\delta}{\pi\epsilon^{3/2}}
 \int_0^\infty\frac{u^{-1/2}}{(1+u)^2}\,du
=\frac{\delta}{2\epsilon^{3/2}}.
$$

For the last integral set $u=\tan^2\theta$: its value is
$2\int_0^{\pi/2}\cos^2\theta\,d\theta=\pi/2$.
Also $\|\widehat g-g\|_{\mathrm{op}}\le\delta$ implies

$$
\widehat g\preceq g+\delta I
 \preceq(1+\delta/\epsilon)g,\qquad
 g\preceq(1+\delta/\epsilon)\widehat g.
$$

Conjugating by $g^{-1/2}$ bounds every generalized eigenvalue between
$(1+\delta/\epsilon)^{-1}$ and $1+\delta/\epsilon$.
Multiplying these $d$ eigenvalues and taking the square root proves
{ref}`(YM.Z2) <eq-fg-ym-z2>`.

For the full positive noise matrix, the implementation's determinant
calculation gives
$c_2^d/\det\Sigma_i=\sqrt{\det g_i}$ in exact arithmetic.
Its positive clamp on the scalar $c_2$ only multiplies all weights on
a frame by the same positive factor, which cancels in $p_i$.
The retained-site mask is already included in $b_i$. Thus
{ref}`(YM.Z3) <eq-fg-ym-z3>` uses the normalized implemented weights
on the finite retained cells. This identifies the recorded quadrature;
it does not replace the cell integral by an equality to its site value.

For $b_i>0$ put $r_i=\widehat v_i/v_i$ and
$\bar r=\sum_i p_i r_i$. The determinant bound gives
$q^{-1}\le r_i,\bar r\le q$, so

$$
\widehat p_i=p_i\frac{r_i}{\bar r},\qquad
\sum_i|\widehat p_i-p_i|
=\sum_i p_i\left|\frac{r_i}{\bar r}-1\right|
\le q^2-1.
$$

Both weight vectors are probabilities, giving also the bound $2$.
Zero-volume cells carry zero weight in both vectors. Now expand

$$
\widehat\Phi-\Phi
=\sum_i\widehat p_i(\widehat B_i-B_i)
 +\sum_i(\widehat p_i-p_i)B_i.
$$

This proves {ref}`(YM.Z4) <eq-fg-ym-z4>`, including its alternative
bound $2M$. For example, if $B_i=O_i f(x_i)$ and
$\widehat B_i=\widehat O_i f(\widehat x_i)$ with $|O_i|\le C$,
then the directly measurable readout error satisfies

$$
\eta\le\|f\|_\infty\max_i|\widehat O_i-O_i|
+C\operatorname{Lip}(f)\max_i|\widehat x_i-x_i|.
$$

To verify the retessellation bound, put
$u_i=b_i\sqrt{\det\widehat g_i}$,
$\widehat u_i=\widehat b_i\sqrt{\det\widehat g_i}$,
$U=\sum_i u_i$, and $\widehat U=\sum_i\widehat u_i$.
For positive normalizations, direct subtraction gives

$$
\begin{aligned}
\sum_i\left|\frac{\widehat u_i}{\widehat U}-\frac{u_i}{U}\right|
&\le\frac{\sum_i|\widehat u_i-u_i|}{U}
  +\widehat U\left|\frac1{\widehat U}-\frac1U\right|\\
&\le\frac{2\sum_i|\widehat u_i-u_i|}{U}=2t.
\end{aligned}
$$

Insert the intermediate weights $u_i/U$ between the original and fully
reconstructed weights. The triangle inequality adds $2t$ to the
metric-only weight bound, proving the claimed replacement for $e$.
If a normalization vanishes, the zero-readout convention and boundedness
give the stated global bound. No cell-shape conclusion is inferred from
coverage alone.

For the correlation bound use the exact product identity

$$
\prod_{k=1}^n\widehat\Phi_k-\prod_{k=1}^n\Phi_k
=\sum_{k=1}^n(\widehat\Phi_k-\Phi_k)
 \prod_{j<k}\widehat\Phi_j\prod_{j>k}\Phi_j.
$$

Take absolute values and expectations. This proves
{ref}`(YM.Z5) <eq-fg-ym-z5>` without independence across sites,
geometry, fields or times. If $\Delta=\widehat\Phi-\Phi$, then
$\|\Delta-\mathbb E_P\Delta\|_p
\le\|\Delta\|_p+|\mathbb E_P\Delta|\le2\|e\|_p$, proving
{ref}`(YM.Z6) <eq-fg-ym-z6>`. In particular the algorithm's
$\sqrt N$ fluctuation normalization retains the factor $\sqrt N$
in the reconstruction error; an intensive-field error estimate cannot
silently remove that factor.

All the metrics, determinants, masks and readouts above are measurable
functions of the recorded history. Conditional expectation with respect
to their joint descriptor therefore gives
$\mathbb E_R[\mathcal L F(Y)]=\mathbb E_R[a_Y(Y)F(Y)]$.
Applying the tower property to $\sigma(\mathscr D)\subseteq\sigma(Y)$
proves both identities in {ref}`(YM.Z7) <eq-fg-ym-z7>`.
Consequently the geometric comparison and every correlation estimate
hold under the very law used for the effective recorded action.
This is the same pathwise treatment of sample-dependent geometry used
in {prf:ref}`lem-scutoid-adaptive-coverage`, now applied to the field
readouts and their complete multi-time correlations.
:::

:::{div} feynman-prose
The stored noise amplitude determines the covariance used by the executed
kinetic step. Decoding it from the Fractal Set recovers that same matrix
exactly. On the specified full Hessian branch, its normalized inverse is
the regularized fitness metric.

Estimating this covariance from the sampled O-stage increments introduces
sampling error. Condition on everything known before the Gaussian draw,
including the current cloning choices, and fix the measurement weights
there. The amplitude is then known, while the new Gaussian noises are
independent across walker slots. Their centered quadratic increments form
the martingale calculated below.

For fixed observation time, the accumulated covariance error has root-mean-square
size proportional to $\sqrt{h/N}$. Multiplication by $\sqrt N$ leaves an
error of order $\sqrt h$. The same estimates control insertion of that
error into field products from the very same history, even when those fields
use the sampled noises. For episodes conditioned on future survival, the
explicit survival likelihood supplies the corresponding error normalization.
:::

:::{prf:theorem} Recorded kinetic covariance identifies the fitness metric at fluctuation scale
:label: thm-ym-kinetic-metric-correspondence

Use the complete O-stage record on the full-step sampling set of
{prf:ref}`def-fractal-set-record-coverage` and the Hessian diffusion
modes of `KineticOperator.apply`, with their
fixed positive regularizer $\epsilon_\Sigma$ and nonzero OU coefficient
$c_h=c_2$ at step size $h$. Let $\mathbb U_{N,h}$ be the law of the
executed algorithm, with its specified initial law and before conditioning
on future survival. After termination, all sums below have zero
contributions; inverse metrics refer to executed stages. At an executed
step $k$, $\mathcal F_k$ contains all information
available before the kinetic Gaussian draw: the preceding record, current
cloning decisions, fitness evaluations, and the first B and A stages.
Write $\Sigma_{ki}$ for the amplitude actually used and $n_{ki}$ for
the stored `noise` at walker slot $i$. The implemented O step gives

(eq-fg-ym-z8)=
$$
n_{ki}=\Sigma_{ki}\xi_{ki},\qquad
D_{ki}=c_h^{-2}\Sigma_{ki}\Sigma_{ki}^{\mathsf T},\qquad
\mathbb E_{\mathbb U}[n_{ki}n_{ki}^{\mathsf T}\mid\mathcal F_k]
 =c_h^2D_{ki},\qquad g_{ki}^{\mathrm{kin}}=D_{ki}^{-1}.
\tag{YM.Z8}
$$

Conditional on $\mathcal F_k$, the $\xi_{ki}$ are independent standard
$d$-dimensional Gaussians. On the successful, unshifted full Hessian
branch, $g_{ki}^{\mathrm{kin}}=\epsilon_\Sigma I+(H_{ki,\mathrm{sym}})_+$.
Here $H_{ki}$ is the supplied fitness Hessian at its recorded evaluation
stage; the O step uses this supplied tensor after the first A stage.
On retry, diagonal, and isotropic fallback branches, $D_{ki}$ is the
covariance of that executed branch. For all these branches,

$$
0\prec D_{ki}\preceq K I,\qquad K=\max\{\epsilon_\Sigma^{-1},1\}.
$$

By {prf:ref}`thm-fractal-set-diffusion` and
{prf:ref}`thm-fractal-set-lossless`, decoding the retained amplitude,
increment, and thermostat convention recovers $D_{ki}$ and
$g_{ki}^{\mathrm{kin}}$ exactly. Thus the encoded and original metric
readouts agree under every history law, at every population size and
every retained step. The corresponding reconstruction error in
{prf:ref}`thm-ym-same-record-metric-field` is zero.

For the statistical covariance readout, take real predictable weights
$w_{ki}$ with $|w_{ki}|\le W$. For example, these may be a bounded
smooth spacetime test at the pre-O position times a pre-O alive mask.
For $m\le\lfloor T/h\rfloor$, define

(eq-fg-ym-z9)=
$$
\begin{aligned}
Q_m&=\frac{h}{Nc_h^2}\sum_{k<m}\sum_{i=1}^N
                  w_{ki}n_{ki}n_{ki}^{\mathsf T},\\
A_m&=\frac hN\sum_{k<m}\sum_{i=1}^N w_{ki}D_{ki},
\qquad R_m=Q_m-A_m.
\end{aligned}
\tag{YM.Z9}
$$

This normalization follows the implemented thermostat:
$c_h^2=\beta_{\mathrm{eff}}^{-1}(1-e^{-2\gamma h})$. For fixed
thermostat parameters, $c_h^2/h\le2\gamma/\beta_{\mathrm{eff}}$ and

$$
0\le\frac{2\gamma}{\beta_{\mathrm{eff}}}-\frac{c_h^2}{h}
\le\frac{2\gamma^2h}{\beta_{\mathrm{eff}}}.
$$

The unnormalized quadratic-increment sum is exactly $(c_h^2/h)Q_m$,
with compensator $(c_h^2/h)A_m$. Its error therefore obeys the same
bounds below, multiplied by $c_h^2/h$.
Here $A_m$ is the weighted accumulated inverse fitness metric on the
actual pre-O sites. Its sampling sites, metric, and weights may depend
on the entire preceding selected swarm evolution. The covariance
measurement error $R_m$ is a matrix martingale and satisfies

(eq-fg-ym-z10)=
$$
\begin{aligned}
\mathbb E_{\mathbb U}\|R_m\|_{\mathrm F}^2
 &=\frac{h^2}{N^2}\sum_{k<m}\sum_i
 \mathbb E_{\mathbb U}\!\left[w_{ki}^2
       \bigl((\operatorname{tr}D_{ki})^2+
                     \operatorname{tr}(D_{ki}^2)\bigr)\right]\\
 &\le d(d+1)W^2K^2\frac{Th}{N},\\
\mathbb E_{\mathbb U}\max_{m\le\lfloor T/h\rfloor}
                      \|\sqrt N R_m\|_{\mathrm F}^2
 &\le4d(d+1)W^2K^2Th.
\end{aligned}
\tag{YM.Z10}
$$

The error vanishes in every finite moment at fluctuation scale, with
an explicit bound. Put $b_d=d(d+1)/2$ and, for $p\ge1$, set

(eq-fg-ym-z11)=
$$
\begin{gathered}
C_{p,d}(N,h,T)=2^{1/p}b_dWK
\left[2\sqrt{Th}\,\Gamma(1+p/2)^{1/p}
       +\frac{2h}{\sqrt N}\,\Gamma(1+p)^{1/p}\right],\\
\|\sqrt N R_m\|_{L^p(\mathbb U;\mathrm F)}
       \le C_{p,d}(N,h,T),\qquad
\|\sqrt N(R_m-\mathbb E_{\mathbb U}R_m)\|_p
       \le C_{p,d}(N,h,T).
\end{gathered}
\tag{YM.Z11}
$$

In particular $C_{p,d}(N,h,T)\to0$ as $h\to0$, uniformly for
$N\ge1$. At fixed $h$ the estimate retains the Gaussian covariance
fluctuation of the recorded algorithm. For any scalar field product
$F$ of the same history with finite second moment and any deterministic
symmetric $B$ with $\|B\|_{\mathrm F}\le1$,

(eq-fg-ym-z12)=
$$
\left|\mathbb E_{\mathbb U}
       [\langle B,\sqrt N(Q_m-A_m)\rangle_{\mathrm F}F]\right|
\le WK\sqrt{d(d+1)Th}\,\|F\|_{L^2(\mathbb U)}.
\tag{YM.Z12}
$$

For any finite recorded descriptor vector $Y$ on this same history,
write $\mathcal L_{\mathbb U}$ for its law together with the indicated
matrix readout. In the product Euclidean metric, the bounded-Lipschitz
distance between the joint centered laws satisfies

(eq-fg-ym-z14)=
$$
\begin{aligned}
d_{\mathrm{BL}}\big(&\mathcal L_{\mathbb U}
  (\sqrt N(Q_m-\mathbb E_{\mathbb U}Q_m),Y),\\
 &\mathcal L_{\mathbb U}
  (\sqrt N(A_m-\mathbb E_{\mathbb U}A_m),Y)\big)
\le C_{1,d}(N,h,T).
\end{aligned}
\tag{YM.Z14}
$$

Thus replacing accumulated metric covariance by its recorded noise
measurement preserves every weak subsequential joint limit as $h\to0$.
No independence of $Y$ and the metric or noise is used.

For a survival-conditioned history law
$P_T=\mathbb U(\,\cdot\mid\tau_\dagger>T)$, write
$s_{N,h,T}=\mathbb U(\tau_\dagger>T)>0$ for its actual survival
probability. The precise transfer to that law is

(eq-fg-ym-z13)=
$$
\begin{aligned}
\|\sqrt N R_m\|_{L^p(P_T;\mathrm F)}
 &\le s_{N,h,T}^{-1/p}C_{p,d}(N,h,T),\\
\|\sqrt N(R_m-\mathbb E_{P_T}R_m)\|_{L^p(P_T;\mathrm F)}
 &\le2s_{N,h,T}^{-1/p}C_{p,d}(N,h,T).
\end{aligned}
\tag{YM.Z13}
$$

Thus encoded metric reconstruction is exact for the selected law as
well, while the statistical covariance estimate retains its already
specified survival normalization. The relevant normalization is supplied
by {prf:ref}`prop-ym-qsd-history-identification`; under a QSD it is
$\alpha_h^m$ at $m$ full steps. The martingale estimate concerns the
stored O-stage increment. The final velocity also includes the remaining
A/B stages and any velocity squashing.
:::

:::{prf:proof}
**The metric and its recorded stage.** In `KineticOperator.apply`, the
amplitude is computed from the pre-O coordinates and the supplied
fitness Hessians. Although the code allocates $\xi$ before computing
this amplitude, the amplitude calculation does not use $\xi$. The
update is $v^+=c_1v^-+\Sigma\xi$, and `info["noise"]` stores
$\Sigma\xi$ before the second A/B stages and velocity squashing.
`RunHistory` retains this increment and the diagonal or full amplitude
in its corresponding optional amplitude field. Consequently all the
conditioning variables in {ref}`(YM.Z8) <eq-fg-ym-z8>` precede the
Gaussian randomization. Its mean is zero and its covariance is the
displayed matrix, for any dependence among the walkers before this draw.

On the full Hessian branch the eigenvalues of $\Sigma/c_h$ are
$[\epsilon_\Sigma+\max(\lambda_j(H_{\mathrm{sym}}),0)]^{-1/2}$,
which proves the metric equality by inversion. A retry uses the
additional solver shift before the same lower clipping;
its covariance eigenvalues are still at most $\epsilon_\Sigma^{-1}$.
The diagonal branch and diagonal fallback obey the same scalar bound.
If the amplitude computation produces a nonfinite tensor, `apply`
uses $\Sigma=c_h I$, so $D=I$. These are exactly the branches in the
stated constant $K$. The supplied-tensor and proxy diffusion modes have
their own amplitude formulas; they are not included in this Hessian-mode
bound.

The Fractal Set diffusion decoder reconstructs each column of $\Sigma$
in its retained output and noise frames. Multiplication, division by
$c_h^2$, and inversion therefore give exactly the original $D$ and
$g^{\mathrm{kin}}$. Isotropic fallback uses the header's thermostat
coefficient and its executed-branch convention. Retained volume weights
and masks are also unchanged by the lossless decoder. Thus the two
versions of every readout in {ref}`(YM.Z3) <eq-fg-ym-z3>` coincide:
$\widehat\Phi=\Phi$. Its centered fluctuation and every correlation
coincide at every $N,h$, so passing through the record isomorphism adds
no error to any of the existing limit constructions.

**Conditional second moments.** Put $y=\Sigma\xi/c_h$. Given
$\mathcal F_k$, $y$ is centered Gaussian with covariance $D$. Its
fourth moment is obtained by differentiating
$\mathbb E e^{t\cdot y}=e^{t^{\mathsf T}Dt/2}$ four times at zero:

$$
\mathbb E[y_a y_b y_c y_d\mid\mathcal F_k]
=D_{ab}D_{cd}+D_{ac}D_{bd}+D_{ad}D_{bc}.
$$

In particular

$$
\mathbb E\|yy^{\mathsf T}-D\|_{\mathrm F}^2
=\sum_{a,b}(D_{aa}D_{bb}+D_{ab}^2)
=(\operatorname{tr}D)^2+\operatorname{tr}(D^2).
$$

The thermostat bounds follow by integrating
$1-e^{-u}\le u$: for $x\ge0$,
$0\le x-(1-e^{-x})=\int_0^x(1-e^{-u})\,du\le x^2/2$.
Set $x=2\gamma h$ and divide by $\beta_{\mathrm{eff}}h$.

At fixed $k$ the centered matrices from different slots are conditionally
independent. Their cross terms vanish after conditioning on
$\mathcal F_k$. At different steps, the earlier centered matrix is
measurable before the later Gaussian draw, so its cross term with the
later martingale difference also has expectation zero. Summing these
identities gives the first line of
{ref}`(YM.Z10) <eq-fg-ym-z10>`. The bounds
$\operatorname{tr}D\le dK$, $\operatorname{tr}(D^2)\le dK^2$,
and $mh\le T$ give its second line.

For completeness, let $X=\max_{j\le m}\|R_j\|_{\mathrm F}$.
Conditional Jensen shows that $\|R_j\|_{\mathrm F}$ is a nonnegative
submartingale. Stopping at its first crossing of $a$ gives
$a\,\mathbb P(X>a)\le\mathbb E[\|R_m\|_{\mathrm F}\mathbf1_{X>a}]$.
Integrating $2a\mathbb P(X>a)$ and using Tonelli yields
$\mathbb E X^2\le2\mathbb E[X\|R_m\|_{\mathrm F}]$.
Cauchy--Schwarz proves $\mathbb E X^2\le4\mathbb E\|R_m\|_{\mathrm F}^2$;
truncation justifies the calculation before finiteness is concluded.
This proves the last line of {ref}`(YM.Z10) <eq-fg-ym-z10>`.

**All finite moments.** Fix symmetric $B$ with Frobenius norm at most
one and write $r_m=\langle B,R_m\rangle_{\mathrm F}$. For a single
slot put $C=wD^{1/2}BD^{1/2}$. Conditional rotational invariance of the
Gaussian permits this positive covariance square root even when the
recorded amplitude is in another noise frame. Its eigenvalues $\lambda_j$
satisfy $\|C\|_{\mathrm{op}}\le WK$ and
$\sum_j\lambda_j^2\le W^2K^2$. For $2|u|WK<1$, direct Gaussian
integration and the power series for $-\log(1-x)$ give

$$
\begin{aligned}
\log\mathbb E e^{u(\xi^{\mathsf T}C\xi-\operatorname{tr}C)}
&=\sum_j\left[-u\lambda_j-\tfrac12\log(1-2u\lambda_j)\right]\\
&=\sum_{r\ge2}\frac{2^{r-1}u^r}{r}\sum_j\lambda_j^r\\
&\le\frac{u^2\sum_j\lambda_j^2}{1-2|u|WK}.
\end{aligned}
$$

The last inequality follows termwise from
$2^{r-1}/r\le2^{r-2}$ and
$|\sum_j\lambda_j^r|\le(WK)^{r-2}\sum_j\lambda_j^2$.
Use $u=\theta h/N$, multiply conditional moment generating functions
across slots, and iterate conditional expectation across the $m$ steps.
With $v=ThW^2K^2/N$ and $b=2hWK/N$, the result is

$$
\log\mathbb E e^{\theta r_m}
 \le\frac{v\theta^2}{1-b|\theta|},\qquad b|\theta|<1.
$$

For $u>0$ choose $\theta=\sqrt u/(\sqrt v+b\sqrt u)$ in the
exponential Markov bound, and apply the same calculation to $-r_m$.
Substitution gives

$$
\mathbb P\{|r_m|\ge2\sqrt{vu}+bu\}\le2e^{-u}.
$$

For $v=0$ the centered readout is zero. Otherwise, integrate this tail
using $f(u)=2\sqrt{vu}+bu$: the identity
$\mathbb E|r_m|^p=\int_0^\infty\mathbb P(|r_m|^p>s)\,ds$
and integration by parts give
$\mathbb E|r_m|^p\le2\int_0^\infty f(u)^p e^{-u}\,du$.
Minkowski's inequality then yields

$$
\|r_m\|_p\le2^{1/p}
 \left[2\sqrt v\,\Gamma(1+p/2)^{1/p}
                  +b\,\Gamma(1+p)^{1/p}\right].
$$

Expand the symmetric matrix $R_m$ in an orthonormal Frobenius basis
of size $b_d$. The inequality
$\|R_m\|_{\mathrm F}\le\sum_{a=1}^{b_d}|\langle B_a,R_m\rangle|$
and Minkowski prove {ref}`(YM.Z11) <eq-fg-ym-z11>` after multiplication
by $\sqrt N$. Its centering identity uses $\mathbb E_{\mathbb U}R_m=0$.

**Insertion into the joint field law.** Cauchy--Schwarz applied directly
to the actual history gives {ref}`(YM.Z12) <eq-fg-ym-z12>`; $F$ may
use these same Gaussian increments. For several error insertions and
field factors, Hölder gives the explicit extension

$$
\left|\mathbb E_{\mathbb U}
   \prod_{a=1}^r\langle B_a,\sqrt N R_{m_a}\rangle
   \prod_{j=1}^s F_j\right|
\le C_{r+s,d}(N,h,T)^r
                       \prod_{j=1}^s\|F_j\|_{r+s},
\qquad r\ge1.
$$

Each error readout may use its own predictable weights bounded by the
same $W$. The field moments in this formula are evaluated under the
same law as the covariance readout. In particular, a bounded recorded
loop or a bounded function of a companion doublet has its direct
supremum bound in every one of these norms.

To prove {ref}`(YM.Z14) <eq-fg-ym-z14>`, take any real function $f$
bounded by one and with Lipschitz constant at most one on the product
space. Couple both joint vectors by using their original common history.
Their $Y$ coordinates cancel, and their centered matrix difference is
$\sqrt N R_m$ since $\mathbb E_{\mathbb U}R_m=0$. Hence

$$
\left|\mathbb E_{\mathbb U}f(\sqrt N(Q_m-\mathbb E_{\mathbb U}Q_m),Y)
-\mathbb E_{\mathbb U}f(\sqrt N(A_m-\mathbb E_{\mathbb U}A_m),Y)\right|
\le\mathbb E_{\mathbb U}\|\sqrt N R_m\|_{\mathrm F}.
$$

Take the supremum over $f$ and use
{ref}`(YM.Z11) <eq-fg-ym-z11>`. For several covariance readouts,
the right-hand side is bounded by the sum of their $L^1$ errors.
The all-moment insertion estimate proves the corresponding replacement
for finite moment hierarchies whenever the other factors have their
established uniform moments under this same law.

In particular, for survival conditioning,
$dP_T/d\mathbb U=\mathbf1_{\{\tau_\dagger>T\}}/s_{N,h,T}$.
Multiplying the nonnegative error moment by this density proves the
first line of {ref}`(YM.Z13) <eq-fg-ym-z13>`. The inequality
$\|X-\mathbb E_{P_T}X\|_p\le2\|X\|_p$ proves the second.
The same Hölder calculation applies under $P_T$ with these transferred
bounds. It keeps $s_{N,h,T}$ explicitly when taking a limit. Gaussian
conditioning was used only before the O draw under $\mathbb U$;
future survival conditioning is handled by its full likelihood.

The argument has therefore identified three quantities on their actual
recorded stages: the fitness metric, its decoded covariance, and the
Gaussian quadratic-covariance readout. The first identification is
pathwise exact. The difference in the last identification is the
explicit martingale in {ref}`(YM.Z9) <eq-fg-ym-z9>`, whose fluctuation
error is controlled by {ref}`(YM.Z10) <eq-fg-ym-z10>` and
{ref}`(YM.Z11) <eq-fg-ym-z11>`.
:::

:::{prf:definition} Covariant derivative and curvature
:label: def-covariant-derivative-ym

Set

(eq-fg-ym-6)=
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

(eq-fg-ym-7)=
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

The bound follows by substituting the Taylor expansion into {ref}`(YM.7) <eq-fg-ym-7>`.
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

For {ref}`(YM.5) <eq-fg-ym-5>` with Hermitian $M$, the common phase symmetry has current
$j^\mu=\overline\Psi\gamma^\mu\Psi$. On the field equations,
$\partial_\mu j^\mu=0$ in flat coordinates, or the corresponding covariant
divergence vanishes on a supplied curved geometry.

For a recorded Markov chain $S_k$ with kernel $P_h$, any integrable observable
$Q$ instead has the exact decomposition

(eq-fg-ym-8)=
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
$Q(S_{k+1})-Q(S_k)=\Delta M_{k+1}+(P_hQ-Q)(S_k)$ proves {ref}`(YM.8) <eq-fg-ym-8>`.
No correspondence between $Q$ and the phase charge is needed for this balance.
:::

For a continuous kinetic jump model, the analogous generator formula is

(eq-fg-ym-9)=
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
transition. The discrete algorithm retains the exact kernel balance {ref}`(YM.8) <eq-fg-ym-8>`.
For smooth weights along absolutely continuous trajectories,
$\partial_t\sum_iF_i\delta_{x_i}+\nabla\cdot\sum_iF_iv_i\delta_{x_i}
=\sum_i\dot F_i\delta_{x_i}$ between jumps, as follows by testing against a
smooth function and differentiating. Jumps add their signed atomic increments.

:::{prf:proposition} Identification of the drift and the stopped balance
:label: prop-ym-generator-balance

In {ref}`(YM.9) <eq-fg-ym-9>`, $b$ is the Itô velocity drift. With the geometric kinetic
coefficients of {prf:ref}`def-gg-generator-decomp`, it is

$$
b_i=-\nabla U_i+F_i-\gamma v_i-\nu(L_XV)_i+b_{\mathrm{geo},i},
\qquad b_{\mathrm{geo}}=\frac12\sum_\ell(DB_\ell)B_\ell.
$$

Here $B_\ell$ are the full phase-space noise columns, and the covariance
in {ref}`(YM.9) <eq-fg-ym-9>` is that of the same columns. The jump kernel contains the complete
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
compensated stochastic integral to the martingale. This is {ref}`(YM.9) <eq-fg-ym-9>`.

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
conditional expectation directly gives {ref}`(YM.8) <eq-fg-ym-8>`, with no passage from a
fixed per-step cloning probability to a continuous jump rate. $\square$
:::



### 4.2. Internal currents and symmetry breaking

:::{prf:theorem} Internal current identity, including chiral sources
:label: thm-su2-noether-current

For a vectorlike internal representation with Hermitian generator $T$, the
flat-space Dirac equations without a background connection give

(eq-fg-ym-10)=
$$
 \partial_\mu(\overline\Psi\gamma^\mu T\Psi)
 =i\overline\Psi[M,T]\Psi.
\tag{YM.10}
$$

With a gauge connection, the same identity uses the adjoint covariant
divergence of the current multiplet. For a left current
$j_{L,T}^\mu=\overline\Psi\gamma^\mu T P_L\Psi$, with internal $M,T$
commuting with the spin matrices, the ungauged identity is

(eq-fg-ym-11)=
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
$\gamma^\mu P_L=P_R\gamma^\mu$ in the second term, which gives {ref}`(YM.11) <eq-fg-ym-11>`.
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
$Q(S_n)-Q(S_0)-h\sum_{k<n}\mathcal D_hQ(S_k)$. By {ref}`(YM.8) <eq-fg-ym-8>` it is a
martingale residual, whose variance can be estimated from its conditional
increments. A field-current residual uses {ref}`(YM.10) <eq-fg-ym-10>` or {ref}`(YM.11) <eq-fg-ym-11>` with the
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

(eq-fg-ym-12)=
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
The latter is Hermitian with convention {ref}`(YM.6) <eq-fg-ym-6>`. A generic finite loop need
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

(eq-fg-ym-13)=
$$
 s_P=1-\frac12\operatorname{Re}\operatorname{Tr}U_P,
 \qquad S_W(U)=\sum_{P\in\mathcal P}\beta_Ps_P.
\tag{YM.13}
$$

On the recorded link descriptor, $S_W$ is a dimensionless Wilson-loop
functional evaluated under the algorithm-derived law
{ref}`(YM.N3) <eq-fg-ym-n3>`--{ref}`(YM.N4) <eq-fg-ym-n4>`.
Its source response is determined by
{prf:ref}`thm-ym-native-source-response` below. Each $s_P\in[0,2]$, hence
$0\le S_W\le2\sum_P\beta_P$. Other representations replace $1/2$ by the
inverse representation dimension and require their own normalization.
:::

:::{prf:theorem} Gauge invariance of Wilson observables and actions
:label: thm-wilson-action-gauge-invariance

The face action {ref}`(YM.13) <eq-fg-ym-13>`, and the normalized trace around every closed loop,
are invariant under {ref}`(YM.12) <eq-fg-ym-12>`. Gauge invariance also holds for any function of
these traces; in particular $\sum_P(\beta_Ps_P+\lambda_Ps_P^2)$ is invariant.
:::

:::{prf:proof}
Substitute {ref}`(YM.12) <eq-fg-ym-12>` into the ordered product. Every intermediate pair
$\Omega_j^{-1}\Omega_j$ cancels, leaving
$U_P'=\Omega_{i_0}U_P\Omega_{i_0}^{-1}$. Cyclicity of trace proves the
claim. Consequently gauge invariance specifies a class of admissible actions;
it does not uniquely select their linear combination in {ref}`(YM.13) <eq-fg-ym-13>`.
:::

### 5.2. Variation and the discrete Ward identity

:::{div} feynman-prose
Generate a complete run from its Gaussian kinetic inputs and the independent
seeds used for the other random choices. A source shifts the mean of those
Gaussian inputs while running the same algorithm. Companion decisions still
use the evolving state, so their outcomes can change when the source changes
the trajectory.

A change of variables puts this perturbation into an explicit Gaussian
likelihood weight. Differentiating that smooth weight computes the response
of recorded observables while the gates and masks remain inside the execution
map. Conditioning the weight on the joint geometry and field descriptor
then gives the change in its existing effective action.

For survival-selected episodes, the source also changes the probability of
survival. Keeping its normalization in the denominator gives the centered
response formulas below, including all higher orders. These are exact
responses of the algorithmic action and its measured field correlations.
:::

:::{prf:theorem} Exact source response of the algorithm-derived field action
:label: thm-ym-native-source-response

Fix a finite execution horizon and the complete record used in
{prf:ref}`thm-ym-kinetic-metric-correspondence`. Collect its reserved
standard Gaussian O-stage innovations into $\Xi\in\mathbb R^q$.
The initial state and independent random seeds driving all other choices
are denoted by $V$. Their state-dependent outcomes remain inside the
execution map. The
algorithm's execution map $\Gamma(\Xi,V)$ includes companion selection,
cloning, all kinetic stages, masks, and termination. Under the native
execution law $\mathbb U$, $\Xi$ is standard Gaussian and independent
of the random inputs in $V$. Reserving unused innovations after
termination does not change the executed record.

Let $\ell=1$ for the execution law, or
$\ell=\mathbf1_{\{\tau_\dagger>T\}}$ for its already specified
survival selection. Set $s=\mathbb E_{\mathbb U}\ell>0$ and
$dP=\ell\,d\mathbb U/s$. Let $Y$ be the joint geometry and gauge-field
descriptor of {ref}`(YM.Z7) <eq-fg-ym-z7>`, evaluated on this record,
and let $F(Y)$ be any bounded measurable function, in particular a
finite product of the recorded bounded gauge channels.

A source $\theta\in\mathbb R^q$ means evaluating this same execution
map on $\Xi+\theta$ and $V$, and applying the same selection rule to
that execution. Its selected expectation is given exactly by

(eq-fg-ym-z15)=
$$
\begin{gathered}
W_\theta(\Xi)=e^{\theta\cdot\Xi-|\theta|^2/2},\qquad
Z_\theta=\mathbb E_P W_\theta>0,\\
M_F(\theta)
=\frac{\mathbb E_{\mathbb U}
 [\ell(\Gamma(\Xi+\theta,V))F(Y(\Gamma(\Xi+\theta,V)))]}
 {\mathbb E_{\mathbb U}\ell(\Gamma(\Xi+\theta,V))}
=\frac{\mathbb E_P[F(Y)W_\theta]}{Z_\theta}.
\end{gathered}
\tag{YM.Z15}
$$

Write $\mu=Y_*P$. The source-dependent descriptor law has the exact
Radon--Nikodym derivative

(eq-fg-ym-z16)=
$$
r_\theta(y)=\frac{\mathbb E_P[W_\theta\mid Y=y]}{Z_\theta},
\qquad d\mu_\theta=r_\theta\,d\mu.
\tag{YM.Z16}
$$

Thus, for the existing effective density
$d\mu=a_0\,d\lambda$ of
{prf:ref}`thm-sm-effective-recorded-gauge-dynamics`,
$a_\theta=a_0r_\theta$ and
$S_\theta^{\mathrm{eff}}=S_0^{\mathrm{eff}}-\log r_\theta$
on the actual descriptor support. This construction uses that same
reference measure $\lambda$ and the same complete update map.
Its first two source variations are

(eq-fg-ym-z17)=
$$
\begin{aligned}
\left.\partial_{\theta_a}S_\theta^{\mathrm{eff}}(y)\right|_0
 &=\mathbb E_P\Xi_a-\mathbb E_P[\Xi_a\mid Y=y],\\
\left.\partial_{\theta_a}\partial_{\theta_b}
                   S_\theta^{\mathrm{eff}}(y)\right|_0
 &=\operatorname{Cov}_P(\Xi_a,\Xi_b)
   -\operatorname{Cov}_P(\Xi_a,\Xi_b\mid Y=y).
\end{aligned}
\tag{YM.Z17}
$$

These identities hold $\mu$-almost everywhere. In the unselected law,
$\mathbb E_{\mathbb U}\Xi_a=0$ and
$\operatorname{Cov}_{\mathbb U}(\Xi_a,\Xi_b)=\delta_{ab}$.
The selected formulas keep the change in these moments.

The complete response hierarchy is explicit at every order. Define
$H_\alpha$ by the finite-dimensional Gaussian generating function

$$
e^{\theta\cdot x-|\theta|^2/2}
=\sum_{\alpha\in\mathbb N^q}\frac{\theta^\alpha}{\alpha!}H_\alpha(x).
$$

For every multi-index $\alpha\ne0$,

(eq-fg-ym-z18)=
$$
\begin{aligned}
M_F(0)&=\mathbb E_P F,\\
\partial^\alpha M_F(0)
&=\mathbb E_P[F H_\alpha(\Xi)]
 -\sum_{0<\beta\le\alpha}\binom\alpha\beta
   \mathbb E_P H_\beta(\Xi)\,
                      \partial^{\alpha-\beta}M_F(0).
\end{aligned}
\tag{YM.Z18}
$$

In particular

(eq-fg-ym-z19)=
$$
\begin{aligned}
\partial_a M_F(0)&=\operatorname{Cov}_P(F,\Xi_a),\\
\partial_a\partial_b M_F(0)
&=\operatorname{Cov}_P(F,\Xi_a\Xi_b)
 -\mathbb E_P\Xi_a\operatorname{Cov}_P(F,\Xi_b)
 -\mathbb E_P\Xi_b\operatorname{Cov}_P(F,\Xi_a).
\end{aligned}
\tag{YM.Z19}
$$

All coefficients are integrable, with the explicit selection bound

(eq-fg-ym-z20)=
$$
\mathbb E_P|H_\alpha(\Xi)|^2\le\frac{\alpha!}{s},\qquad
\left|\mathbb E_P[F H_\alpha(\Xi)]\right|
 \le\|F\|_\infty\sqrt{\frac{\alpha!}{s}}.
\tag{YM.Z20}
$$

The identities apply to the algorithm-derived generating functional
by taking $F(Y)=\exp(\sum_jJ_jO_j(Y))$ for finitely many bounded
recorded channels $O_j$. Differentiating in their sources $J_j$ then
produces every mixed finite field correlation and innovation response.
The Gaussian source changes the input of the executed update; the
formulas do not require differentiability of its cloning gate, masks,
or descriptor map.
:::

:::{prf:proof}
**Change of the native Gaussian input.** Write
$\gamma_q(x)=(2\pi)^{-q/2}e^{-|x|^2/2}$. At fixed $V=v$, make the
change of variable $x=\xi+\theta$. Its density transforms as

$$
\gamma_q(x-\theta)
=\gamma_q(x)\exp(\theta\cdot x-|\theta|^2/2).
$$

Therefore, for every bounded measurable function $B$ of the complete
execution,

$$
\mathbb E_{\mathbb U}B(\Gamma(\Xi+\theta,V))
=\mathbb E_{\mathbb U}[B(\Gamma(\Xi,V))W_\theta(\Xi)].
$$

Use $B=\ell F(Y)$ and $B=\ell$ and divide. The factors $s$ cancel,
giving {ref}`(YM.Z15) <eq-fg-ym-z15>`. Every selection or mask is
inside $B$ during this change of variables. In particular, the survival
event in the numerator and denominator is evaluated on the shifted
execution before changing variables. Positivity of $W_\theta$ gives a
positive denominator since $s>0$. Its expectation is finite because
$0\le\ell\le1$ and Gaussian exponential moments are finite.

Conditional expectation of the right-hand side of
{ref}`(YM.Z15) <eq-fg-ym-z15>` given $Y$ proves
{ref}`(YM.Z16) <eq-fg-ym-z16>`. This is a density relative to the
already constructed descriptor law and so includes any singular support
or algebraic constraints in its image. Multiplication by its existing
density $a_0$ yields the stated effective action.

**Differentiation and the action coefficients.** For $|\theta|\le R$,
every derivative of $W_\theta$ of a fixed finite order is bounded by
$C(1+|\Xi|)^m e^{R|\Xi|+R^2/2}$. To see integrability, use
$R|x|\le |x|^2/4+R^2$ against $\gamma_q(x)$; the remaining
polynomial times $e^{-|x|^2/4}$ is integrable. Division by $s$ gives
the same bound under $P$. This justifies all unconditional derivatives
by dominated convergence. The conditional expectation of each such
envelope is finite almost everywhere. Intersecting the resulting
full-measure sets over integer $R$ and integer derivative order gives
a common set on which conditional derivatives are also justified.

Let $b_\theta(y)=\mathbb E_P[W_\theta\mid Y=y]$. At zero,
$b_0=Z_0=1$, and direct differentiation gives

$$
\begin{aligned}
\partial_ab_0(y)&=\mathbb E_P[\Xi_a\mid Y=y],&
\partial_aZ_0&=\mathbb E_P\Xi_a,\\
\partial_a\partial_bb_0(y)
 &=\mathbb E_P[\Xi_a\Xi_b-\delta_{ab}\mid Y=y],&
\partial_a\partial_bZ_0
 &=\mathbb E_P[\Xi_a\Xi_b-\delta_{ab}].
\end{aligned}
$$

Use $\partial_a\partial_b\log b
=b^{-1}\partial_a\partial_bb-b^{-2}(\partial_ab)(\partial_bb)$
and $S_\theta^{\mathrm{eff}}-S_0^{\mathrm{eff}}
=-\log b_\theta+\log Z_\theta$.
The two $\delta_{ab}$ terms cancel. This gives both lines of
{ref}`(YM.Z17) <eq-fg-ym-z17>`. The conditional score therefore
retains the posterior of the actual innovations given the recorded
geometry and gauge fields.

**All orders and selection normalization.** Multiply
{ref}`(YM.Z15) <eq-fg-ym-z15>` by $Z_\theta$. Differentiation at zero
and the multivariate product rule give

$$
\sum_{\beta\le\alpha}\binom\alpha\beta
 \mathbb E_P H_\beta(\Xi)\,
            \partial^{\alpha-\beta}M_F(0)
=\mathbb E_P[F H_\alpha(\Xi)].
$$

The term $\beta=0$ is $\partial^\alpha M_F(0)$.
Moving the other terms to the right proves
{ref}`(YM.Z18) <eq-fg-ym-z18>`. For order one use $H_{e_a}=\Xi_a$.
For order two use
$\partial_a\partial_bW_0=\Xi_a\Xi_b-\delta_{ab}$.
The corresponding $\delta_{ab}\mathbb E_PF$ contributions cancel,
and substitution of the first-order formula gives
{ref}`(YM.Z19) <eq-fg-ym-z19>`, including the case $a=b$ with its
two equal cross terms.

To compute the Hermite moment, multiply the two Gaussian generating
functions and integrate:

$$
\mathbb E_{\mathbb U}
 [e^{t\cdot\Xi-|t|^2/2}e^{u\cdot\Xi-|u|^2/2}]
=e^{t\cdot u}.
$$

Differentiate in $t$ and $u$ at zero. The coefficient comparison gives
$\mathbb E_{\mathbb U}[H_\alpha H_\beta]
=\mathbf1_{\{\alpha=\beta\}}\alpha!$.
Since $\ell\le1$,
$\mathbb E_P H_\alpha^2
=s^{-1}\mathbb E_{\mathbb U}[\ell H_\alpha^2]\le\alpha!/s$.
Cauchy--Schwarz proves the second bound in
{ref}`(YM.Z20) <eq-fg-ym-z20>`.

**Recorded applicability.** At an executed nondegenerate O stage,
{prf:ref}`thm-ym-kinetic-metric-correspondence` supplies the retained
amplitude and increment, so its innovation is exactly
$\xi_{ki}=\Sigma_{ki}^{-1}n_{ki}$. The reconstruction is therefore
a function of the same recorded data used by the gauge channels.
Unused reserved coordinates are integrated out in the conditional
expectations. All other random inputs remain inside $V$ and the
complete execution map. No derivative has been commuted through the
sampled-companion gate or a hard mask.

For channel sources $J$ in a bounded set, boundedness of the finite
list $O_j$ supplies a constant dominating $e^{\sum_jJ_jO_j}$ and
all of its finite source derivatives. The preceding Gaussian envelopes
therefore justify mixed differentiation in $J$ and $\theta$.
Finally, {prf:ref}`thm-fractal-set-lossless` commutes with evaluation
of every retained factor. The same response identities hold in the
Fractal Set representation and in the effective descriptor integral.
:::

:::{div} feynman-prose
Apply a local vector test through the diffusion frame actually used before
the O-stage draw. This gives a small mean kick whose direction and amplitude
follow the recorded kinetic geometry. The test, mask, and frame are known
before that draw. Later companion and cloning decisions adapt to the
resulting states through their existing rules, and multiplying the
conditional Gaussian ratios gives the complete history likelihood.

The source normalization controls its accumulated energy. Squaring the
$1/\sqrt N$ factor cancels the sum over walkers. The thermostat variance
factor scales with $h$, so the squared standardized kick contributes a factor
proportional to $h$; summing over at most $T/h$ steps leaves a bound
independent of both $N$ and $h$.

Condition this likelihood on the joint recorded descriptor to obtain the
action response. Its second derivative includes the conditional source
energy because that energy depends on the adaptive trajectory. Survival
selection retains its full normalization throughout. These calculations
control the new source using the established record, covariance, and
fluctuation setup, with explicit bounds for its likelihood and responses.
:::

:::{prf:theorem} Fitness-metric force sources and quantitative action response
:label: thm-ym-metric-force-sources

Use the complete staged execution and decoded amplitudes of
{prf:ref}`thm-ym-kinetic-metric-correspondence`. Fix its positive
thermostat parameters $\gamma,\beta_{\mathrm{eff}}$, a finite horizon
$T$, and a maximum step size $h_0>0$. For $0<h\le h_0$, let
$m=\lfloor T/h\rfloor$ and
$c_h^2=\beta_{\mathrm{eff}}^{-1}(1-e^{-2\gamma h})$.
The native execution law, with its given initial law, is $\mathbb U$.

Choose a real vector test
$f\in C_c^\infty(\mathbb R\times\mathbb R^{2d};\mathbb R^d)$ and
write $B=\sup|f|$. At an executed pre-O stage put
$f_{ki}=a_{ki}f(kh,x_{ki}^-,v_{ki}^-)$, where $a_{ki}$ is the
pre-O alive mask. Set the contribution to zero after termination.
The mask, sites, and diffusion amplitude are evaluated on the actual
preceding history, including its companion and cloning decisions.
Construct a source parameter $\lambda\in\mathbb R$ by replacing
only this stage's conditional increment law with the Gaussian law
having the same covariance and the following added mean:

(eq-fg-ym-z22)=
$$
\delta n_{ki}
=\lambda\frac h{\sqrt N}\frac{\Sigma_{ki}}{c_h}f_{ki}
=\lambda\Sigma_{ki}u_{ki},\qquad
u_{ki}:=\frac h{c_h\sqrt N}f_{ki}.
\tag{YM.Z22}
$$

The factor $N^{-1/2}$ uses the fluctuation normalization already fixed
in {prf:ref}`cor-ym-empirical-fluctuations`.
All other conditional update rules and the initial law are kept fixed
as functions of their current state. This defines the sourced execution
law $\mathbb U_\lambda$ by the same chronological composition of
kernels. At $\lambda=0$ it is exactly $\mathbb U$.
On the full Hessian branch the added mean is
$\lambda h\,g_{ki}^{-1/2}f_{ki}/\sqrt N$ by
{ref}`(YM.Z8) <eq-fg-ym-z8>`. On the other Hessian-mode branches,
$\Sigma_{ki}/c_h$ is their actual decoded noise frame. In particular,
$|\delta n_{ki}|\le |\lambda|h\sqrt K B/\sqrt N$, with the same
$K=\max\{\epsilon_\Sigma^{-1},1\}$ as in that theorem.

Define two scalar functions of the complete record,

(eq-fg-ym-z23)=
$$
\begin{gathered}
M_f=\sum_{k<m}\sum_i u_{ki}\cdot\xi_{ki},\qquad
E_f=\sum_{k<m}\sum_i|u_{ki}|^2,\\
0\le E_f\le C_f,
\qquad C_f:=\frac{T\beta_{\mathrm{eff}}e^{2\gamma h_0}}{2\gamma}B^2,
\qquad \mathbb E_{\mathbb U}M_f=0,
\quad \mathbb E_{\mathbb U}M_f^2=\mathbb E_{\mathbb U}E_f\le C_f.
\end{gathered}
\tag{YM.Z23}
$$

At executed stages $\xi_{ki}=\Sigma_{ki}^{-1}n_{ki}$, so these
quantities are recovered from the same Fractal Set record as the fields.
The source energy $E_f$ retains all dependence on the recorded trajectory.
Its bound $C_f$ is independent of $N$ and $h\le h_0$.

The complete history likelihood and its moment estimates are

(eq-fg-ym-z24)=
$$
\begin{gathered}
L_\lambda=\frac{d\mathbb U_\lambda}{d\mathbb U}
 =\exp(\lambda M_f-\lambda^2E_f/2),\qquad
\mathbb E_{\mathbb U}L_\lambda=1,\\
\mathbb E_{\mathbb U}L_\lambda^p
 \le e^{p(p-1)\lambda^2C_f/2}\quad(p\ge1),\qquad
\mathbb E_{\mathbb U}e^{tM_f}\le e^{t^2C_f/2}\quad(t\in\mathbb R).
\end{gathered}
\tag{YM.Z24}
$$

For the joint recorded geometry and field descriptor $Y$ of
{ref}`(YM.Z7) <eq-fg-ym-z7>`, put
$\mu_\lambda=Y_*\mathbb U_\lambda$, $\mu=Y_*\mathbb U$, and
$r_\lambda=\mathbb E_{\mathbb U}[L_\lambda\mid Y]$. Then

(eq-fg-ym-z25)=
$$
\begin{aligned}
D_{\mathrm{KL}}(\mu_\lambda\Vert\mu)
&\le D_{\mathrm{KL}}(\mathbb U_\lambda\Vert\mathbb U)
 =\frac{\lambda^2}{2}\mathbb E_{\mathbb U_\lambda}E_f
 \le\frac{\lambda^2C_f}{2},\\
\mathbb E_\mu|r_\lambda-1|^2
&\le e^{\lambda^2C_f}-1,\\
|\mathbb E_{\mu_\lambda}F-\mathbb E_\mu F|
&\le\|F\|_\infty\sqrt{e^{\lambda^2C_f}-1}.
\end{aligned}
\tag{YM.Z25}
$$

For the native law or its specified survival selection, use
$\ell=1$ or $\ell=\mathbf1_{\{\tau_\dagger>T\}}$ respectively,
$s=\mathbb E_{\mathbb U}\ell>0$, and $P=\ell\mathbb U/s$.
The selected source law has density
$L_\lambda/\mathbb E_P L_\lambda$ relative to $P$. Its existing
joint-descriptor action therefore changes by
$-\log\mathbb E_P[L_\lambda\mid Y]+\log\mathbb E_P L_\lambda$.
Its first two derivatives are the explicit recorded quantities

(eq-fg-ym-z26)=
$$
\begin{aligned}
\partial_\lambda S^{\mathrm{eff}}_\lambda(Y)|_0
 &=\mathbb E_PM_f-\mathbb E_P[M_f\mid Y],\\
\partial_\lambda^2 S^{\mathrm{eff}}_\lambda(Y)|_0
 &=\operatorname{Var}_P M_f-\operatorname{Var}_P(M_f\mid Y)
   +\mathbb E_P[E_f\mid Y]-\mathbb E_PE_f,\\
\mathbb E_P L_\lambda
 &\ge\exp\left(-|\lambda|\sqrt{C_f/s}-\lambda^2C_f/2\right)>0.
\end{aligned}
\tag{YM.Z26}
$$

All response orders admit bounds with the same $C_f$. For $n\ge0$
define the explicitly evaluated likelihood derivative

(eq-fg-ym-z27)=
$$
\begin{gathered}
Q_n(M,E)=n!\sum_{j=0}^{\lfloor n/2\rfloor}
 \frac{(-E/2)^jM^{n-2j}}{j!(n-2j)!},\qquad Q_0=1,\\
A_r=\left[2(2)^{r/2}\Gamma(1+r/2)\right]^{1/r}\quad(r>0),\\
B_{n,p}=n!\sum_{j=0}^{\lfloor n/2\rfloor}
 \frac{A_{p(n-2j)}^{\,n-2j}}{2^j j!(n-2j)!},\qquad A_0^0:=1,\\
a_{n,p}:=s^{-1/p}B_{n,p}C_f^{n/2}\quad(n\ge1,p\ge1),\qquad
\|Q_n(M_f,E_f)\|_{L^p(P)}\le a_{n,p}.
\end{gathered}
\tag{YM.Z27}
$$

For $n\ge1$ let $\Pi_n$ be the finite set of partitions of
$\{1,\ldots,n\}$. Set $b_j(Y)=\mathbb E_P[Q_j(M_f,E_f)\mid Y]$
and $z_j=\mathbb E_P Q_j(M_f,E_f)$. With $k=|\pi|$ for a partition,

(eq-fg-ym-z28)=
$$
\begin{aligned}
\partial_\lambda^n S^{\mathrm{eff}}_\lambda(Y)|_0
 &=\sum_{\pi\in\Pi_n}(-1)^{k-1}(k-1)!
       \left[\prod_{B'\in\pi}z_{|B'|}
                       -\prod_{B'\in\pi}b_{|B'|}(Y)\right],\\
\|\partial_\lambda^n S^{\mathrm{eff}}_\lambda|_0\|_{L^p(P)}
 &\le2\sum_{\pi\in\Pi_n}(k-1)!
                         \prod_{B'\in\pi}a_{|B'|,pk}.
\end{aligned}
\tag{YM.Z28}
$$

For a bounded recorded field functional $F(Y)$, write
$R_F(\lambda)=\mathbb E_P[F L_\lambda]/\mathbb E_P L_\lambda$.
Its derivatives and explicit scalar bounds satisfy

(eq-fg-ym-z29)=
$$
\begin{aligned}
R_F^{(n)}(0)
 &=\mathbb E_P[FQ_n]
       -\sum_{j=1}^n\binom nj z_j R_F^{(n-j)}(0),\\
d_0&=\|F\|_\infty,\qquad
 d_n=\|F\|_\infty a_{n,1}
                  +\sum_{j=1}^n\binom nj a_{j,1}d_{n-j},
 \qquad |R_F^{(n)}(0)|\le d_n.
\end{aligned}
\tag{YM.Z29}
$$

These formulas apply to finite products of the recorded bounded gauge
channels and to their finite-source generating functional. The sources
are local tests in the recorded phase-space coordinates; their subsequent
influence is propagated by the complete algorithm. All unselected bounds
above are uniform in $N$ and $h\le h_0$. Selected bounds keep the
actual survival probability $s$ from
{prf:ref}`prop-ym-qsd-history-identification`.
:::

:::{prf:proof}
**Construction of the physical source and its energy.** Before the O
draw, every $f_{ki}$ and $\Sigma_{ki}$ is measurable in the pre-O
sigma algebra of {prf:ref}`thm-ym-kinetic-metric-correspondence`.
Adding the conditional mean in
{ref}`(YM.Z22) <eq-fg-ym-z22>` therefore specifies a Gaussian kernel
at this very stage. The remainder of the step is executed with the
same A/B rules, squashing, and termination rule. The next companion
probabilities are computed from that resulting state by their original
rule. This chronological prescription defines $\mathbb U_\lambda$
without prescribing a field action.

The amplitude bound $\|\Sigma/c_h\|_{\mathrm{op}}\le\sqrt K$
gives the stated physical-kick bound. For $x\ge0$,
$1-e^{-x}=\int_0^x e^{-u}\,du\ge xe^{-x}$. Consequently

$$
c_h^2\ge\frac{2\gamma h}{\beta_{\mathrm{eff}}}e^{-2\gamma h}
\ge\frac{2\gamma h}{\beta_{\mathrm{eff}}}e^{-2\gamma h_0},
\qquad
E_f\le\frac{mh^2}{c_h^2}B^2
\le\frac{Th}{c_h^2}B^2\le C_f.
$$

Conditional independence of the O-stage noises gives
$\mathbb E[\sum_i u_{ki}\cdot\xi_{ki}\mid\mathcal F_k]=0$
and conditional second moment $\sum_i|u_{ki}|^2$.
Earlier increments are measurable before later draws, so their cross
moments vanish by conditioning. Summing gives the last two identities
in {ref}`(YM.Z23) <eq-fg-ym-z23>`. These calculations hold with the
pre-O alive masks and zero contributions after termination.

**The full likelihood, including adaptive choices.** Conditional on a
fixed preceding history, the ratio of the Gaussian density with mean
$\lambda\Sigma u$ to the density with mean zero, evaluated at
$n=\Sigma\xi$, is

$$
\frac{\exp[-|\xi-\lambda u|^2/2]}{\exp[-|\xi|^2/2]}
=\exp(\lambda u\cdot\xi-\lambda^2|u|^2/2).
$$

The covariance determinant cancels because both kernels use the same
amplitude at this preceding history. Other stochastic substeps have
identical conditional kernels at the same history, so their likelihood
ratios are one. Their outcomes can depend on earlier sourced states;
no independence of those outcomes is asserted. Multiplying these
stage ratios yields $L_\lambda$ in
{ref}`(YM.Z24) <eq-fg-ym-z24>`.

To verify both normalization and the law identity without assuming a
joint transition density, condition successively on each pre-O history.
The Gaussian expectation of its stage factor is exactly one, by
completing the square. Multiplication by each intervening unchanged
kernel preserves the cylinder-integral identity. Induction over the
finite sequence of stages proves
$\mathbb E_{\mathbb U_\lambda}G=\mathbb E_{\mathbb U}[L_\lambda G]$
for every bounded measurable recorded cylinder $G$, and in particular
$\mathbb E_{\mathbb U}L_\lambda=1$. Deterministic clipping or squashing
maps require no density or Jacobian in this kernel argument. Retaining
the Gaussian increment at its own stage makes $M_f,E_f,L_\lambda$
functions of the complete record.

For $p\ge1$ the exact identity

$$
L_\lambda^p
=L_{p\lambda}\exp\left(\frac{p(p-1)\lambda^2}{2}E_f\right)
$$

and $E_f\le C_f$ give the likelihood moment bound. Similarly
$e^{tM_f}=L_t e^{t^2E_f/2}$ proves the sub-Gaussian bound in
{ref}`(YM.Z24) <eq-fg-ym-z24>`.

**Entropy and descriptor response.** Under $\mathbb U_\lambda$, the
conditional standardized increment is $\xi_{ki}=\lambda u_{ki}+
\zeta_{ki}$, with $\zeta_{ki}$ centered standard Gaussian given the
preceding history. Hence
$\mathbb E_{\mathbb U_\lambda}M_f
=\lambda\mathbb E_{\mathbb U_\lambda}E_f$.
Integrating $\log L_\lambda=\lambda M_f-\lambda^2E_f/2$ gives the
path relative entropy in {ref}`(YM.Z25) <eq-fg-ym-z25>`.
All these integrals are finite by the energy and Gaussian moment bounds.

For $r_\lambda=\mathbb E[L_\lambda\mid Y]$, conditional Jensen
for the convex function $x\log x$ gives
$\mathbb E[r_\lambda\log r_\lambda]
\le\mathbb E[L_\lambda\log L_\lambda]$.
Conditional Jensen for the square likewise gives

$$
\mathbb E_\mu(r_\lambda-1)^2
\le\mathbb E_{\mathbb U}(L_\lambda-1)^2
=\mathbb E_{\mathbb U}L_\lambda^2-1
\le e^{\lambda^2C_f}-1.
$$

Integrating $F(r_\lambda-1)$ and applying Cauchy--Schwarz proves
the final line of {ref}`(YM.Z25) <eq-fg-ym-z25>`.
These are bounds for the existing joint descriptor law and its actual
source deformation.

**Selected likelihood and first variations.** The same survival event
is evaluated on each sourced execution. Its density calculation gives

$$
\frac{dP_\lambda}{dP}
=\frac{L_\lambda}{\mathbb E_PL_\lambda},\qquad
\mathbb E_PL_\lambda
=\frac{\mathbb U_\lambda(\ell=1)}{s}
$$

in the survival case, with normalization one when $\ell=1$ identically.
Jensen and {ref}`(YM.Z23) <eq-fg-ym-z23>` give

$$
\log\mathbb E_P L_\lambda
\ge\lambda\mathbb E_PM_f-\lambda^2\mathbb E_PE_f/2
\ge-|\lambda|\sqrt{\mathbb E_PM_f^2}-\lambda^2 C_f/2
\ge-|\lambda|\sqrt{C_f/s}-\lambda^2C_f/2.
$$

Conditional expectation given $Y$ now identifies the selected
source-dependent effective density, exactly as in
{ref}`(YM.Z16) <eq-fg-ym-z16>`. Its first two likelihood derivatives
are $M_f$ and $M_f^2-E_f$. Differentiating the two logarithms in
its action gives {ref}`(YM.Z26) <eq-fg-ym-z26>`, including the
conditional source-energy term. That term is retained because $E_f$
depends on the actual path.

**Quantitative differentiation at every order.** Exponential Markov
applied with $t=r/C_f$ and with $-t$ gives
$\mathbb P_{\mathbb U}(|M_f|>r)\le2e^{-r^2/(2C_f)}$ when
$C_f>0$. Integrating the tail gives, for every $p>0$,

$$
\mathbb E_{\mathbb U}|M_f|^p
\le 2(2C_f)^{p/2}\Gamma(1+p/2),\qquad
\|M_f\|_{L^p(P)}\le s^{-1/p}A_p\sqrt{C_f}.
$$

If $C_f=0$, the source is zero and all positive-order responses vanish.
For $C_f>0$, differentiating the two factors in
$e^{\lambda M_f}e^{-\lambda^2E_f/2}$ gives exactly the polynomial
$Q_n$ in {ref}`(YM.Z27) <eq-fg-ym-z27>`.
For a term with $r=n-2j>0$,

$$
\|E_f^j M_f^r\|_{L^p(P)}
\le C_f^j\|M_f\|_{L^{pr}(P)}^r
\le s^{-1/p}A_{pr}^r C_f^{n/2}.
$$

For $r=0$ the same inequality holds with $A_0^0=1$, since $s\le1$.
The triangle inequality proves the bound by $a_{n,p}$.
On $|\lambda|\le R$, each likelihood derivative is bounded by a
polynomial in $|M_f|$ times $e^{R|M_f|+R^2C_f/2}$ with deterministic
coefficients. Its integrability follows from the proved Gaussian tail:
$Rr\le r^2/(4C_f)+R^2C_f$, leaving an integrable Gaussian tail
after multiplying by any fixed polynomial. These envelopes justify
differentiation under both ordinary and conditional expectations.
Conditional envelopes are finite almost everywhere; countably many
integer $R$ and derivative orders give one common full-measure set.

For clarity, the logarithmic derivative formula in
{ref}`(YM.Z28) <eq-fg-ym-z28>` follows directly from its Taylor
coefficients. If $b(0)=1$, expand

$$
\log b(\lambda)
=\sum_{k=1}^n\frac{(-1)^{k-1}}k(b(\lambda)-1)^k
 +O(\lambda^{n+1}).
$$

The coefficient of $\lambda^n$ in the $k$th summand is a sum over
ordered positive integers $n_1+\cdots+n_k=n$, weighted by
$\prod_j b^{(n_j)}(0)/n_j!$. Multiplication by $n!$ converts the
ordered choices into ordered partitions of $\{1,\ldots,n\}$.
Each unordered partition occurs $k!$ times; division by $k$ leaves
$(k-1)!$. Apply this identity to
$b(\lambda)=\mathbb E_P[L_\lambda\mid Y]$ and
$z(\lambda)=\mathbb E_PL_\lambda$ and subtract their logarithms.
This proves the displayed partition formula.

Conditional Jensen gives $\|b_j\|_q\le\|Q_j\|_q$ and
$|z_j|\le\|Q_j\|_q$ for $q\ge1$. For each partition with $k$
blocks, Hölder with $k$ equal exponents bounds each product in
$L^p(P)$ by $\prod_{B'\in\pi}a_{|B'|,pk}$.
Summing and bounding the difference of the two products proves the
last line of {ref}`(YM.Z28) <eq-fg-ym-z28>`.

Finally, differentiate
$R_F(\lambda)\mathbb E_P L_\lambda
=\mathbb E_P[F L_\lambda]$ $n$ times at zero.
The term with no derivative on the denominator is $R_F^{(n)}(0)$;
moving the other terms gives the first line of
{ref}`(YM.Z29) <eq-fg-ym-z29>`.
The inequalities $|z_j|\le a_{j,1}$ and
$|\mathbb E_P[FQ_n]|\le\|F\|_\infty a_{n,1}$ prove its scalar
recursion by induction. This computes every finite response order
with constants determined by the algorithm's thermostat, the chosen
local test, the horizon, and the retained survival normalization.
:::

:::{div} feynman-prose
Place each recorded innovation at its pre-O time and position, with the
source weight already calculated above. Pairing this vector distribution
with a smooth test recovers the force-response sum. Its coordinates are
recorded time and the $d$ spatial coordinates; its vector components belong
to the recorded noise frame.

The existing Hermite method now organizes these measurements into modes.
The source moment bounds control their weighted sum and give an explicit
error for the omitted modes, uniformly in population and time step. This
provides finite approximations and tightness for the whole current.

Averaging the current conditional on the joint geometry and field descriptor
gives the negative first variation of the recorded action. When taking a
subsequential limit, retain this conditional current as a descriptor
coordinate. It is already determined by the finite record. Its inclusion
lets the defining conditional-average identities pass through the limit,
proving the conditional expectation identity for the retained limiting
descriptor.
:::

:::{prf:theorem} Native response current in recorded spacetime
:label: thm-ym-native-response-current

Retain the execution law, full-step recording, horizon, and thermostat
parameters of {prf:ref}`thm-ym-metric-force-sources`. Put
$D=1+d$, with recorded coordinates $y=(t,x)\in\mathbb R^D$.
For a real vector test $f\in\mathcal S(\mathbb R^D;\mathbb R^d)$,
use the pre-O force test $f_{ki}=a_{ki}f(kh,x_{ki}^-)$.
Its source-energy bound is
$E_f\le K_T\|f\|_\infty^2$, where
$K_T=T\beta_{\mathrm{eff}}e^{2\gamma h_0}/(2\gamma)$.
Define the vector-valued finite atomic distribution

(eq-fg-ym-z30)=
$$
\mathcal I_{N,h}
=\frac h{c_h\sqrt N}\sum_{k<m}\sum_i
          a_{ki}\xi_{ki}\,\delta_{(kh,x_{ki}^-)},\qquad
\mathcal I_{N,h}(f)=M_f.
\tag{YM.Z30}
$$

Every location, mask, and innovation is evaluated at its recorded stage
and decoded by the existing reconstruction. For a spatial algorithm
with $d=3$, this distribution has four test coordinates and three
components in the recorded noise frame.

Use the Hermite-space construction of
{prf:ref}`thm-ym-spacetime-fluctuation-compactness`, now on these
$D$ coordinates: $A=1-\Delta_y+|y|^2$, basis $e_k$, and eigenvalues
$\lambda_k=1+D+2|k|$. For vector distributions set
$\|u\|_{-s}^2=\sum_{a=1}^d\sum_k\lambda_k^{-2s}|u_a(e_k)|^2$.
The following constants make the spatial mode estimates explicit:

(eq-fg-ym-z31)=
$$
\begin{gathered}
b=D+1,\qquad r=D+3,\qquad s_0=r+D,\qquad s_1=s_0+1,\\
B_D=(2\pi)^{-D/2}
 \left[\frac{\pi^{D/2}\Gamma(b-D/2)}{\Gamma(b)}\right]^{1/2}
 [2(b+1)(D+1)]^{b/2},\\
\|e_k\|_\infty\le B_D\lambda_k^r,\qquad
C_D:=\sum_k\lambda_k^{-2D}
 \le\frac{2}{(D-1)!D^{D+1}}.
\end{gathered}
\tag{YM.Z31}
$$

Let $A_p$ be the scalar moment constant in
{ref}`(YM.Z27) <eq-fg-ym-z27>`. The current has the uniform bounds

(eq-fg-ym-z32)=
$$
\begin{aligned}
\mathbb E_{\mathbb U}\|\mathcal I_{N,h}\|_{-s_0}^2
 &\le dK_TB_D^2C_D=:C_I,\\
\|\mathcal I_{N,h}\|_{L^p(\mathbb U;\mathscr H_{-s_0})}
 &\le A_p\sqrt{C_I}\quad(p\ge2),\\
\mathbb E_{\mathbb U}
 \|(I-\Pi_L)\mathcal I_{N,h}\|_{-s_0}^2
 &\le\frac{dK_TB_D^2}{D(D-1)!}(L+D)^{-D}.
\end{aligned}
\tag{YM.Z32}
$$

Here $\Pi_L$ keeps all Hermite modes with $|k|\le L$.
In particular the laws are tight in $\mathscr H_{-s_1}$, uniformly
in $N$ and $0<h\le h_0$.

Let $Y_{N,h}$ be the actual joint geometry and field descriptor. Its
source-response current is the Hilbert-valued conditional expectation

(eq-fg-ym-z33)=
$$
\mathcal J_{N,h}
=\mathbb E_{\mathbb U}[\mathcal I_{N,h}\mid Y_{N,h}],\qquad
\left.\partial_\lambda S_{\lambda,f}^{\mathrm{eff}}(Y_{N,h})
 \right|_0=-\mathcal J_{N,h}(f).
\tag{YM.Z33}
$$

It satisfies all three bounds in
{ref}`(YM.Z32) <eq-fg-ym-z32>`. Thus the first variation of the
algorithm-derived action is a random distribution with an explicit
population-independent mode-tail estimate.

The conditional current has a concrete finite construction. Encode the
finite recorded descriptor by a fixed countable list of bounded real
coordinates, using $(2/\pi)\arctan x$ for real entries and presence
and shape tags for variable-length entries. The tags make this code
injective with measurable inverse on its image. Denote it by
$c(Y)\in[-1,1]^{\mathbb N}$. Let $\mathcal P_j$ partition the first
$j$ coordinates into dyadic intervals of mesh $2^{-j}$. Zero-probability
cells are assigned value zero. Then

(eq-fg-ym-z34)=
$$
\mathcal J^{j,L}_{N,h}
=\sum_{C\in\mathcal P_j:\,\mathbb U(c(Y)\in C)>0}
 \mathbf1_{\{c(Y)\in C\}}
 \frac{\mathbb E_{\mathbb U}
       [\mathbf1_{\{c(Y)\in C\}}\Pi_L\mathcal I_{N,h}]}
      {\mathbb U(c(Y)\in C)}.
\tag{YM.Z34}
$$

For fixed $N,h$, these conditional finite-mode vectors converge to
$\mathcal J_{N,h}$ in $L^2(\mathbb U;\mathscr H_{-s_0})$ as
$j,L\to\infty$. The Hermite truncation error has the bound in
{ref}`(YM.Z32) <eq-fg-ym-z32>`. The remaining partition error is the
exact orthogonal-projection residual; no unproved partition rate is
used.

For a sequence with $h\to0$ and any population sequence, retain the
augmented descriptor
$\widehat Y_{N,h}=(c(Y_{N,h}),\mathcal J_{N,h})$.
The joint laws of
$(\mathcal I_{N,h},\mathcal J_{N,h},c(Y_{N,h}))$
have weakly convergent subsequences in
$\mathscr H_{-s_1}^2\times[-1,1]^{\mathbb N}$.
Every such limit $(\mathcal I,\mathcal J,B)$ obeys

(eq-fg-ym-z35)=
$$
\mathcal J=\mathbb E[\mathcal I\mid\widehat Y],
\qquad \widehat Y=(B,\mathcal J),\qquad
\mathbb E\mathcal I=\mathbb E\mathcal J=0.
\tag{YM.Z35}
$$

Finite polynomial moments of the current pairings, multiplied by any
bounded continuous function of the augmented descriptor, converge
along this same subsequence. The augmentation is a measurable function
of the original descriptor at every finite $N,h$. Its explicit inclusion
retains the response information in the limiting descriptor.
:::

:::{prf:proof}
**Evaluation on recorded spacetime.** The source proof in
{prf:ref}`thm-ym-metric-force-sources` uses predictability and the
supremum bound of its test. A test $f(t,x)$ gives those same bounds
when evaluated at $(kh,x_{ki}^-)$, regardless of the velocities.
Therefore it defines the same physical mean shift
$\lambda h\Sigma_{ki}a_{ki}f(kh,x_{ki}^-)/(c_h\sqrt N)$,
with energy bounded by $K_T\|f\|_\infty^2$.
This establishes its applicability directly, without identifying a
phase-space marginal with a new field law.
The finite sum in {ref}`(YM.Z30) <eq-fg-ym-z30>` then gives $M_f$
by evaluation. Schwartz tests are bounded, so the same argument applies
to them. Finite-mode and compactly supported evaluations agree with
this finite atomic distribution.

**The existing Hermite method with explicit constants.** Use the
unitary Fourier transform on $\mathbb R^D$. Fourier inversion and
Cauchy--Schwarz give

$$
\|u\|_\infty
\le(2\pi)^{-D/2}
 \left[\int_{\mathbb R^D}(1+|\zeta|^2)^{-b}d\zeta\right]^{1/2}
 \left[\int_{\mathbb R^D}(1+|\zeta|^2)^b|\widehat u(\zeta)|^2
                                                    d\zeta\right]^{1/2}.
$$

Polar coordinates followed by $v=|\zeta|^2$ evaluate the first
integral as $\pi^{D/2}\Gamma(b-D/2)/\Gamma(b)$; it is finite
because $b>D/2$. For the second, expand
$(1+\sum_j\zeta_j^2)^b$ by the multinomial theorem and apply
Plancherel. The coefficients are nonnegative and their sum is
$(D+1)^b$.

The Hermite derivative formula is
$\partial_j e_k=(\sqrt{k_j}e_{k-e_j}-\sqrt{k_j+1}e_{k+e_j})/\sqrt2$.
A derivative of order $a\le b$ has at most $2^a$ terms. Each term's
coefficient has absolute value at most
$2^{-a/2}(|k|+b+1)^{a/2}$, and every basis vector has norm one.
The triangle inequality therefore gives

$$
\|\partial^\alpha e_k\|_2^2
\le[2(|k|+b+1)]^{|\alpha|}
\le[2(b+1)]^b\lambda_k^b\qquad(|\alpha|\le b).
$$

Substitution proves
$\|e_k\|_\infty\le B_D\lambda_k^{b/2}\le B_D\lambda_k^r$.
This is the same raising/lowering argument used in
{ref}`(YM.F4) <eq-fg-ym-f4>`, with the supremum seminorm needed
for these force tests.

There are $\binom{n+D-1}{D-1}\le(n+D)^{D-1}/(D-1)!$ indices
with $|k|=n$, and $\lambda_k\ge n+D$. Consequently

$$
\begin{aligned}
C_D&\le\frac1{(D-1)!}\sum_{n\ge0}(n+D)^{-D-1}
 \le\frac{2}{(D-1)!D^{D+1}},\\
\sum_{|k|>L}\lambda_k^{-2D}
 &\le\frac1{(D-1)!}\int_L^\infty(x+D)^{-D-1}dx
 =\frac{(L+D)^{-D}}{D(D-1)!}.
\end{aligned}
$$

This proves all constants in {ref}`(YM.Z31) <eq-fg-ym-z31>`.

**Moments, tails, and compactness.** Testing component $a$ against
$e_k$ is the force test $e_k\mathbf e_a$. Equations
{ref}`(YM.Z23) <eq-fg-ym-z23>` and
{ref}`(YM.Z27) <eq-fg-ym-z27>` give respectively

$$
\mathbb E|\mathcal I_a(e_k)|^2
 \le K_TB_D^2\lambda_k^{2r},\qquad
\|\mathcal I_a(e_k)\|_p
 \le A_p\sqrt{K_T}B_D\lambda_k^r.
$$

Multiply the first inequality by $\lambda_k^{-2s_0}$ and sum over
components and modes. Tonelli's theorem proves the first and third
bounds in {ref}`(YM.Z32) <eq-fg-ym-z32>`, and also shows that
$\mathcal I$ belongs to the indicated Hilbert space almost surely.
For $p\ge2$, Minkowski in $L^{p/2}$ gives

$$
\left\|\sum_{a,k}\lambda_k^{-2s_0}
                  |\mathcal I_a(e_k)|^2\right\|_{p/2}
\le\sum_{a,k}\lambda_k^{-2s_0}\|\mathcal I_a(e_k)\|_p^2
\le A_p^2 C_I.
$$

Taking a square root proves the second bound. The inclusion
$\mathscr H_{-s_0}\hookrightarrow\mathscr H_{-s_1}$ is compact:
on a norm-bounded set the squared tail in the latter norm is at most
$\bigl(\min_{|k|>L}\lambda_k\bigr)^{-2}$ times its former squared norm, and tends
to zero uniformly. Finite-mode balls are finite dimensional and have
compact closure. Markov's inequality with $C_I$ now proves tightness.

**Constructing the conditional current.** For each finite-mode
projection, conditional expectations of its real coefficients define
an ordinary finite-dimensional conditional vector. Conditional Jensen
bounds its squared Hilbert norm by that of $\Pi_L\mathcal I$ and
its mode tail by that of $(I-\Pi_L)\mathcal I$. The quantitative
tail estimate makes these vectors Cauchy in Hilbert-valued $L^2$.
Their limit defines $\mathcal J$ and satisfies
$\mathcal J(f)=\mathbb E[\mathcal I(f)\mid Y]$ for every Schwartz
test, first for finite-mode tests and then by the Hilbert pairing.
Conditional Jensen proves its $L^p$ bounds as well.
Since $\mathbb E M_f=0$, the first line of
{ref}`(YM.Z26) <eq-fg-ym-z26>` identifies its negative with the
first variation of the actual action, proving
{ref}`(YM.Z33) <eq-fg-ym-z33>`.

The coordinate code copies the descriptor's discrete tags and applies
an invertible bounded transform to each real entry; inverse tangent
and the retained tags recover every entry on the image. Dyadic cylinder
partitions therefore generate its sigma algebra. For each finite-mode
coefficient, simple functions of these partitions are dense in
$L^2(\sigma(Y))$. One way to verify density is to let $\mathcal C$
be the sets whose indicators are approximable by cylinder simple
functions. Finite unions, complements, and increasing countable unions
preserve this property by $L^2$ approximation, so $\mathcal C$
contains the generated sigma algebra. Truncation then covers all
square-integrable measurable functions.

The cell average in {ref}`(YM.Z34) <eq-fg-ym-z34>` is precisely
the orthogonal projection onto these simple functions. Density proves
its convergence to the finite-mode conditional vector. For
$\mathcal J^j=\mathbb E[\mathcal I\mid\mathcal P_j(c(Y))]$,
projection orthogonality also gives the exact error identity

$$
\mathbb E\|\mathcal J-\mathcal J^j\|_{-s_0}^2
=\mathbb E\|\mathcal J\|_{-s_0}^2
 -\mathbb E\|\mathcal J^j\|_{-s_0}^2.
$$

Combining finite-mode convergence with the uniform mode-tail bound
proves the stated two-parameter approximation. This implements the
same finite-partition method used in
{prf:ref}`thm-sm-predictive-partition-convergence` for the present
recorded response observable.

**The joint limiting correspondence.** The two Hilbert laws are tight
by the preceding bounds; the countable cube with its product metric
is compact. Their joint laws are therefore tight in the stated
separable complete metric space and have weakly convergent subsequences.
At finite $N,h$, $\mathcal J_{N,h}$ is measurable with respect to
$Y_{N,h}$. For every bounded continuous $G$ on the augmented descriptor
space and every Schwartz vector test $f$,

$$
\mathbb E_{\mathbb U}
 [(\mathcal I_{N,h}(f)-\mathcal J_{N,h}(f))
                 G(c(Y_{N,h}),\mathcal J_{N,h})]=0.
$$

The current pairings are continuous in $\mathscr H_{-s_1}$ because
Schwartz tests belong to its dual positive Hilbert space. The uniform
second moments give uniform integrability of this product. Weak
convergence thus passes this identity to the limit. Bounded continuous
functions determine finite measures on this Polish descriptor space;
a monotone-class extension gives the same identity for all bounded
measurable $G$. Testing a countable Hermite basis identifies the
Hilbert conditional expectation and proves
{ref}`(YM.Z35) <eq-fg-ym-z35>`. In particular the limiting current
is measurable in the retained augmented descriptor by construction.

For a polynomial product of finitely many current pairings, Hölder
and the uniform bounds at an order strictly greater than its degree
give uniform integrability. The same weak-convergence argument passes
all its moments, including bounded continuous descriptor insertions,
along this one subsequence. Conditional expectations were identified
through these identities and the explicit augmentation; no interchange
of conditional projection and weak convergence was assumed.
:::

:::{div} feynman-prose
The force source already gives a score and an energy on each recorded
history. For several tests, collect them into a vector and a matrix; their
exponential gives the complete source likelihood. Its conditional average
given the recorded descriptor is the density that changes the descriptor
law.

Retain that density as a whole function of the source parameters while
taking a common subsequential limit. At every finite size it is determined
by the original descriptor. Keeping it explicitly preserves the conditional
information needed to pass the likelihood identities to the limiting
descriptor.

The established source-energy bounds control moments of every derivative
and of the inverse conditional density, uniformly in population and time
step. Those estimates allow their negative logarithms, the action increments, and
their correlation responses to converge smoothly on bounded source domains.
A countable source dictionary fits on the same subsequence, and the explicit
continuity estimate extends its normalized laws to every Schwartz source.
All these limiting densities come from the original execution likelihood.
:::

:::{prf:theorem} Common continuum limit of the native source likelihood and descriptor action
:label: thm-ym-continuum-source-action

Use the native execution law and current construction of
{prf:ref}`thm-ym-native-response-current`, on the same fixed horizon
and with the same $K_T$. Fix real vector source tests
$f_1,\ldots,f_q\in\mathcal S(\mathbb R^{1+d};\mathbb R^d)$,
and put $C=K_T\sum_{a=1}^q\|f_a\|_\infty^2$.
For $\theta\in\mathbb R^q$, apply the already constructed force
source with test $f_\theta=\sum_a\theta_af_a$. Its recorded score
vector and source-energy matrix are

(eq-fg-ym-z36)=
$$
\begin{gathered}
M_a=\mathcal I_{N,h}(f_a),\qquad
E_{ab}=\frac{h^2}{Nc_h^2}\sum_{k<m}\sum_i
 a_{ki}f_a(kh,x_{ki}^-)\cdot f_b(kh,x_{ki}^-),\\
E\succeq0,\quad \operatorname{tr}E\le C,\qquad
L_{N,h}(\theta)=e^{\theta\cdot M-\theta^{\mathsf T}E\theta/2},\\
r_{N,h}(\theta)=\mathbb E_{\mathbb U_{N,h}}
                  [L_{N,h}(\theta)\mid Y_{N,h}].
\end{gathered}
\tag{YM.Z36}
$$

The source law has density $L_{N,h}(\theta)$ on the complete execution
record, and density $r_{N,h}(\theta)$ on its original joint descriptor.
The latter is a random smooth function of $\theta$, constructed by
conditional expectation of the existing algorithm likelihood.
Let $\|D^n u\|_{R}$ denote the supremum, for $|\theta|\le R$,
of the operator norm of the $n$th derivative, with $D^0u=u$.
For $n\ge0$, $p\ge1$, and $R\ge0$, define

(eq-fg-ym-z37)=
$$
\begin{gathered}
B_{n,R}=e^{3/2}n!(1+C)^n(1+CR)^n,\\
V_{n,p,R}=B_{n,R}\,2^{q/p}
                       e^{p(R+n)^2qC/2},\qquad
W_{p,R}=2^{q/p}e^{CR^2/2+pqCR^2/2},\\
\bigl\|\|D^n L_{N,h}\|_R\bigr\|_{L^p(\mathbb U_{N,h})}
 \le V_{n,p,R},\qquad
\bigl\|\|D^n r_{N,h}\|_R\bigr\|_{L^p(\mathbb U_{N,h})}
 \le V_{n,p,R},\\
\left\|\sup_{|\theta|\le R}r_{N,h}(\theta)^{-1}\right\|_p
 \le W_{p,R}.
\end{gathered}
\tag{YM.Z37}
$$

All constants are independent of population and time step.

For any sequence $N\to\infty$, $h\to0$, append the function
$r_{N,h}\in C^\infty_{\mathrm{loc}}(\mathbb R^q)$ to the augmented
descriptor in {ref}`(YM.Z35) <eq-fg-ym-z35>`. At each finite size it
is still a measurable function of the original $Y_{N,h}$. There is a
common subsequence on which

$$
(\mathcal I_{N,h},\mathcal J_{N,h},E,c(Y_{N,h}),r_{N,h})
\Longrightarrow (\mathcal I,\mathcal J,E_\infty,B,r)
$$

in the two negative Hermite spaces used previously, the finite-dimensional
matrix space, the countable descriptor cube, and
$C^\infty_{\mathrm{loc}}(\mathbb R^q)$ respectively.
Write $\widehat Y=(B,\mathcal J,r)$ and
$M_\infty=(\mathcal I(f_a))_{a=1}^q$. Then, simultaneously for all
real $\theta$,

(eq-fg-ym-z38)=
$$
\begin{gathered}
L_\infty(\theta)
=e^{\theta\cdot M_\infty-\theta^{\mathsf T}E_\infty\theta/2},
\quad \mathbb E L_\infty(\theta)=1,\quad
r(\theta)=\mathbb E[L_\infty(\theta)\mid\widehat Y]>0,\\
\mathcal J=\mathbb E[\mathcal I\mid\widehat Y],\qquad
 d\mu_\theta=r(\theta)\,d\mu,\quad
 \mu=\operatorname{Law}(\widehat Y),\qquad
\Delta S_\infty(\theta)=-\log r(\theta).
\end{gathered}
\tag{YM.Z38}
$$

Thus $\mu_\theta$ is a normalized source-dependent law of the limiting
recorded descriptor. Its action increment is defined relative to $\mu$,
exactly as the finite action increment is defined relative to the actual
finite descriptor law. Its normalization and positivity are derived
from the common limit of the native likelihood.

The finite action increments
$\Delta S_{N,h}=-\log r_{N,h}$ converge jointly in
$C^\infty_{\mathrm{loc}}$ to $\Delta S_\infty$. For $n\ge1$,
let $\Pi_n$ denote the partitions of $\{1,\ldots,n\}$, and write
$k=|\pi|$. Their derivatives have the explicit bound

(eq-fg-ym-z39)=
$$
\bigl\|\|D^n\Delta S_{N,h}\|_R\bigr\|_p
\le\sum_{\pi\in\Pi_n}(k-1)!
 W_{kp(k+1),R}^{\,k}
                \prod_{B'\in\pi}V_{|B'|,p(k+1),R}.
\tag{YM.Z39}
$$

Their zeroth-order norm is bounded by $V_{0,p,R}+W_{p,R}$.
The same bounds hold in the limiting law. Every finite polynomial
moment of these derivatives and current pairings, with bounded
continuous descriptor insertions, converges along this subsequence.
In particular, differentiation has not been interchanged with a
merely pointwise action limit.

For a fixed bounded continuous function $F$ of the augmented descriptor,
including products of its explicitly retained bounded gauge-channel
coordinates, set

(eq-fg-ym-z40)=
$$
G_{N,h}(\theta)=\mathbb E_{\mathbb U_{N,h}}
 [F(c(Y_{N,h}),\mathcal J_{N,h},r_{N,h})L_{N,h}(\theta)].
\tag{YM.Z40}
$$

Then $G_{N,h}\to G$ in $C^\infty_{\mathrm{loc}}$, where
$G(\theta)=\mathbb E[F(\widehat Y)L_\infty(\theta)]
=\int F\,d\mu_\theta$. For every derivative order,
$\sup_{|\theta|\le R}\|D^nG_{N,h}(\theta)\|
\le\|F\|_\infty V_{n,1,R}$.

The construction can retain a fixed countable dense source dictionary
on a single subsequence. For two tests with
$\|f\|_\infty,\|g\|_\infty\le B_0$, its likelihoods satisfy

(eq-fg-ym-z41)=
$$
\|L_f-L_g\|_p,
\ \|\mathbb E[L_f-L_g\mid Y]\|_p
\le2e^{(2p-1)K_TB_0^2/2}
       (A_{2p}\sqrt{K_T}+K_TB_0)\|f-g\|_\infty.
\tag{YM.Z41}
$$

Here $L_f$ denotes the source of strength one with test $f$, and
$A_{2p}$ is the already defined scalar moment constant. This bound
extends the limiting source law from that dictionary to every Schwartz
test by approximation. The native execution law is used throughout;
survival-conditioned source laws retain the normalization in
{ref}`(YM.Z26) <eq-fg-ym-z26>`.
:::

:::{prf:proof}
**The matrix and likelihood come from the executed source.** The energy
of $f_\theta$ is the quadratic form $\theta^{\mathsf T}E\theta$ by
expansion of the actual finite sum in
{ref}`(YM.Z23) <eq-fg-ym-z23>`. The matrix is a sum of positive
Gram matrices. Applying that energy bound to each $f_a$ and summing
gives $\operatorname{tr}E\le C$, hence $E\preceq CI$.
The source score is linear in the test, so it is $\theta\cdot M$.
Thus {ref}`(YM.Z24) <eq-fg-ym-z24>` gives the likelihood in
{ref}`(YM.Z36) <eq-fg-ym-z36>`, its expectation one, and

$$
\mathbb E L(\theta)^p\le e^{p(p-1)C|\theta|^2/2},\qquad
\mathbb E e^{u\cdot M}\le e^{C|u|^2/2}.
$$

These are consequences of the already constructed source kernel;
$E$ remains the random recorded energy matrix.

**Uniform smooth-function estimates.** For $a\ge0$, use
$|M|\le\sum_j|M_j|=\max_{\varepsilon\in\{-1,1\}^q}
\varepsilon\cdot M$ to obtain

$$
\mathbb E e^{a|M|}
\le\sum_{\varepsilon\in\{-1,1\}^q}
                    \mathbb E e^{a\varepsilon\cdot M}
\le2^q e^{a^2qC/2}.
$$

The exponent of $L$ has gradient $M-E\theta$, Hessian $-E$, and
zero higher derivatives. The product rule expresses $D^nL$ as a sum
over partitions into singletons and pairs. Counting $j$ pairs gives
$n!/[2^j j!(n-2j)!]$. For unit input vectors and $|\theta|\le R$,

$$
\|D^nL(\theta)\|
\le e^{R|M|}n!\sum_{j=0}^{\lfloor n/2\rfloor}
 \frac{C^j(|M|+CR)^{n-2j}}{2^j j!(n-2j)!}.
$$

Each power product is at most
$(1+C)^n(1+CR)^n(1+|M|)^n$.
The sum of the remaining coefficients without $n!$ is bounded by
$(\sum_{j\ge0}2^{-j}/j!)(\sum_{a\ge0}1/a!)=e^{3/2}$.
Finally $(1+x)^n\le e^{nx}$ for $x\ge0$.
Taking $L^p$ norms and using the preceding radial exponential estimate
proves $V_{n,p,R}$ in {ref}`(YM.Z37) <eq-fg-ym-z37>`.
The resulting integrable derivative envelopes justify conditional
differentiation, so conditional Jensen gives the same estimate for $r$.
This also constructs a simultaneous smooth version of $r$: apply the
conditional envelopes on countably many integer balls and derivative
orders, as in {prf:ref}`thm-ym-native-source-response`.

For the inverse estimate, $L(\theta)\ge
 e^{-R|M|-CR^2/2}$ on the same ball. Conditional Jensen gives

$$
\inf_{|\theta|\le R}r(\theta)
\ge\exp[-R\mathbb E(|M|\mid Y)-CR^2/2]>0.
$$

Raising its inverse to the power $p$, taking expectation, and using
conditional Jensen again yields

$$
\mathbb E\sup_{|\theta|\le R}r(\theta)^{-p}
\le e^{pCR^2/2}\mathbb E e^{pR|M|}
\le2^q e^{pCR^2/2+p^2R^2qC/2}.
$$

This is the bound $W_{p,R}$.

**Constructing the conditional function and its compactness.** The
finite descriptor partitions in
{ref}`(YM.Z34) <eq-fg-ym-z34>` also construct

$$
r^j(\theta)
=\sum_{C'\in\mathcal P_j:\,\mathbb U(Y\in C')>0}
 \mathbf1_{\{Y\in C'\}}
 \frac{\mathbb E[\mathbf1_{\{Y\in C'\}}L(\theta)]}
      {\mathbb U(Y\in C')}.
$$

Here cell membership means membership of the bounded descriptor code.
For each fixed $\theta$ and derivative, the finite-partition projection
argument of {ref}`(YM.Z34) <eq-fg-ym-z34>` gives $L^2$
convergence to the corresponding conditional expectation. If $d_j$
is the projection error, conditional contraction bounds its $2p$th
moment uniformly. For $p>2$ and $A>0$,

$$
\mathbb E|d_j|^p
\le A^{p-2}\mathbb E|d_j|^2+A^{-p}\mathbb E|d_j|^{2p}.
$$

First let $j\to\infty$ and then $A\to\infty$.
For $1<p\le2$, use $\|d_j\|_p\le\|d_j\|_2$.
This proves convergence in every stated $L^p$ without a new
conditional-limit premise. To verify convergence in the supremum norm on a ball, take a finite
$\delta$-net. The supremum of a derivative difference is bounded by
its maximum on that net plus $\delta$ times the two next-derivative
suprema. The net errors converge in $L^p$, and the latter suprema
have the uniform bound $2V_{n+1,p,R+1}$. First let $j\to\infty$
and then $\delta\to0$. This gives an explicit finite approximation
procedure for the entire conditional smooth function.

The bounds $V_{n+1,1,R+1}$ also prove tightness of $r_{N,h}$ in
$C^\infty_{\mathrm{loc}}$. For any error probability $\epsilon$,
apply Markov with thresholds
$2^{n+R+1}V_{n,1,R+1}/\epsilon$ for integer $n\ge0$, $R\ge1$.
The sum of the exceptional probabilities is at most
$\epsilon\sum_{n\ge0,R\ge1}2^{-n-R-1}=\epsilon$.
On the remaining set all derivatives
are uniformly bounded on each ball. The next derivative makes each
such family equicontinuous. Finite nets and diagonal extraction give
compact closure in $C^\infty_{\mathrm{loc}}$ by the Arzela--Ascoli
argument, simultaneously on all integer balls.

The two current variables are already tight by
{prf:ref}`thm-ym-native-response-current`. The energy matrices belong
to the compact set $\{E\succeq0:\operatorname{tr}E\le C\}$,
and the coded descriptors belong to the compact countable cube.
These estimates prove joint tightness in the stated product space
and supply the common subsequence.

**Identification of the limiting conditional density.** Evaluation of
$\mathcal I(f_a)$ is continuous in the negative Hermite space.
Consequently $L_{N,h}(\theta)$ converges in law jointly with all
retained coordinates to $L_\infty(\theta)$ for every fixed $\theta$.
Its uniform $p$th moment for $p>1$ gives uniform integrability, so
$\mathbb E L_\infty(\theta)=1$.
At finite $N,h$, both $\mathcal J_{N,h}$ and the whole function
$r_{N,h}$ are measurable in the original descriptor. Therefore, for
every bounded continuous $G$ of the augmented descriptor,

$$
\mathbb E[G(c(Y_{N,h}),\mathcal J_{N,h},r_{N,h})r_{N,h}(\theta)]
=\mathbb E[G(c(Y_{N,h}),\mathcal J_{N,h},r_{N,h})L_{N,h}(\theta)].
$$

Both sides pass to the common limit by their uniform second moments.
A monotone-class extension from bounded continuous $G$ gives
$r(\theta)=\mathbb E[L_\infty(\theta)\mid\widehat Y]$.
Use the same argument on
$\mathcal I_{N,h}(f)-\mathcal J_{N,h}(f)$ to obtain the current
identity in {ref}`(YM.Z38) <eq-fg-ym-z38>` for this enlarged
limiting descriptor.

The identities first hold on a common full-measure set for rational
$\theta$. The derivative envelopes pass to the limit by the moment
bounds, so conditional expectations of $L_\infty$ have continuous
versions on each ball. Density of rational vectors extends the
identities to all $\theta$. Moreover

$$
\inf_{|\theta|\le R}r(\theta)
\ge\mathbb E[e^{-R|M_\infty|-CR^2/2}\mid\widehat Y]>0
$$

almost surely. The random variable inside this conditional expectation
is strictly positive; its conditional expectation can vanish only on
a null set. Integer balls give simultaneous strict positivity and a
smooth action increment on all finite source domains.

**Action derivatives and correlation limits.** Taking $-\log$ is
continuous in $C^\infty$ on a compact source ball when the limiting
function has a strictly positive minimum there. The preceding result
therefore identifies the joint limit of the finite action increments.
For $n\ge1$, the derivative of $\log r$ is the partition formula
with terms
$(-1)^{k-1}(k-1)!r^{-k}\prod_{B'\in\pi}D^{|B'|}r$,
evaluated on the corresponding input vectors.
The coefficient calculation is the one proved in
{ref}`(YM.Z28) <eq-fg-ym-z28>`. Apply Hölder with $k+1$ equal
exponents to the inverse factor and the $k$ derivative factors.
Their norms are bounded by $W_{kp(k+1),R}^k$ and
$V_{|B'|,p(k+1),R}$, giving
{ref}`(YM.Z39) <eq-fg-ym-z39>`. For order zero use
$|\log x|\le x+x^{-1}$. The bounds pass to the limit, and bounds
at an order greater than a polynomial's degree give the asserted
uniform integrability of all finite moments.

For {ref}`(YM.Z40) <eq-fg-ym-z40>`, dominated differentiation gives
$D^nG_{N,h}(\theta)=\mathbb E[F D^nL_{N,h}(\theta)]$.
Joint convergence and the derivative moment bounds pass each fixed
$\theta$ to $\mathbb E[F D^nL_\infty(\theta)]$.
The bound at order $n+1$ gives uniform equicontinuity on a compact
source ball. A finite-net argument therefore upgrades this convergence
to uniform convergence there, at every order. This proves the stated
$C^\infty_{\mathrm{loc}}$ convergence of correlations and source
responses, for the same limiting density and descriptor.

**One countable dictionary and extension to arbitrary tests.** Choose
a countable dense dictionary of Schwartz vector tests and retain the
conditional source function for its first $q$ entries for every
$q\ge1$. Each coordinate family is tight by the preceding proof.
For a prescribed error probability, choose compact sets with errors
$\epsilon2^{-q}$ and take their product. This is compact in the
countable product topology and its joint probability is at least
$1-\epsilon$. Thus a single subsequence exists for the entire
dictionary and its energy matrices. The finite identities
$r^{q+1}(\theta,0)=r^q(\theta)$ and the upper-left-block
consistency of the energy matrices pass to the limit by continuity
of restriction, proving consistency. The conditional identification
uses the descriptor retaining all these source functions; finite
cylinder tests followed by a monotone-class extension prove it there.

For two bounded tests $f,g$, the actual finite energy sum gives

$$
|E_f-E_g|\le K_T(\|f\|_\infty+\|g\|_\infty)\|f-g\|_\infty,
\qquad
\|M_f-M_g\|_{2p}\le A_{2p}\sqrt{K_T}\|f-g\|_\infty.
$$

For real $u,v$, the integral of $e^t$ between them gives
$|e^u-e^v|\le|u-v|(e^u+e^v)$.
Apply this with $u=M_f-E_f/2$, $v=M_g-E_g/2$ and use Hölder.
The likelihood moment estimate in
{ref}`(YM.Z24) <eq-fg-ym-z24>` gives
$\|L_f\|_{2p},\|L_g\|_{2p}\le e^{(2p-1)K_TB_0^2/2}$.
This proves {ref}`(YM.Z41) <eq-fg-ym-z41>`; conditional Jensen
proves its second bound.

The same estimates hold for the jointly limiting dictionary
coordinates. The limiting matrix energies form positive bilinear
forms bounded by $K_T\|f\|_\infty\|g\|_\infty$, and therefore
extend uniquely from the dictionary's rational linear span by uniform
approximation. The current pairings extend by their proved $L^p$
continuity, consistently with the distributional pairing in
{prf:ref}`thm-ym-native-response-current`.
Consequently the exponential likelihoods and their conditional
densities extend in $L^p$ by
{ref}`(YM.Z41) <eq-fg-ym-z41>`, independently of the approximating
sequence. Their expectations remain one. The conditional expectation
identity extends by $L^1$ contraction, so every Schwartz source has
its normalized limiting descriptor law. No independent field measure
has entered the construction.
:::

:::{prf:theorem} Recorded occupation, continuum source energy, and descriptor information
:label: thm-ym-occupation-source-identification

Use the native law, pre-O sites and masks, and common source limit of
{prf:ref}`thm-ym-continuum-source-action`. The recorded occupation is

(eq-fg-ym-z42)=
$$
\zeta_{N,h}=\frac hN\sum_{k<m}\sum_i a_{ki}\delta_{(kh,x_{ki}^-)},
\qquad \kappa_h=\frac h{c_h^2},\quad
\kappa=\frac{\beta_{\mathrm{eff}}}{2\gamma},\qquad
\zeta_{N,h}(\mathbb R^{1+d})\le T.
\tag{YM.Z42}
$$

The source-energy matrix already defined in
{ref}`(YM.Z36) <eq-fg-ym-z36>` satisfies

(eq-fg-ym-z43)=
$$
\begin{gathered}
E_{ab}=\kappa_h\zeta_{N,h}(f_a\cdot f_b),\qquad
0\le\kappa_h-\kappa\le\beta_{\mathrm{eff}}h e^{2\gamma h_0},\\
|E_{ab}-\kappa\zeta_{N,h}(f_a\cdot f_b)|
\le\beta_{\mathrm{eff}}Th e^{2\gamma h_0}
                         \|f_a\|_\infty\|f_b\|_\infty.
\end{gathered}
\tag{YM.Z43}
$$

Retain this measure jointly with the raw variables of the preceding
theorem. A further common subsequence has a vague occupation limit
$\zeta$, positive, of mass at most $T$, and supported in
$[0,T]\times\mathbb R^d$. On that same joint limit, for every Schwartz
source pair,

(eq-fg-ym-z44)=
$$
\begin{gathered}
(E_\infty)_{ab}=\kappa\zeta(f_a\cdot f_b),\qquad
L_\infty(\theta)=\exp\left[\mathcal I(f_\theta)
                   -\frac\kappa2\zeta(|f_\theta|^2)\right],\\
\mathbb E\mathcal I(f)=0,\qquad
\mathbb E[\mathcal I(f)\mathcal I(g)]
                  =\kappa\mathbb E\zeta(f\cdot g).
\end{gathered}
\tag{YM.Z44}
$$

Here $\zeta$ belongs to the raw joint law; the descriptor remains
$\widehat Y$ of {prf:ref}`thm-ym-continuum-source-action`, and
$r(\theta)=\mathbb E[L_\infty(\theta)\mid\widehat Y]$.
Occupation insertions against $C_0(\mathbb R^{1+d})$ tests converge
on this same subsequence. The statements concern local occupation
integrals and retain the mass upper bound.

Write $M_a=\mathcal I(f_a)$, $J_a=\mathcal J(f_a)$,
$Q_{ab}=\kappa\zeta(f_a\cdot f_b)$, and
$\mathcal A_{ab}=\partial_a\partial_b\Delta S_\infty(0)$.
Then

(eq-fg-ym-z45)=
$$
\begin{gathered}
\partial_a\Delta S_\infty(0)=-J_a,\qquad
\mathcal A_{ab}=\mathbb E[Q_{ab}\mid\widehat Y]
                     -\operatorname{Cov}(M_a,M_b\mid\widehat Y),\\
\mathbb E\mathcal A=\mathbb E[JJ^{\mathsf T}],\qquad
\mathbb E Q=\mathbb E[JJ^{\mathsf T}]
             +\mathbb E[(M-J)(M-J)^{\mathsf T}],\\
0\preceq\mathbb E\mathcal A\preceq\mathbb E Q.
\end{gathered}
\tag{YM.Z45}
$$

For every real square-integrable descriptor observable $F$, including
bounded recorded gauge observables, the zero-source response obeys

(eq-fg-ym-z46)=
$$
\begin{gathered}
\left.\frac d{d\lambda}\int F\,d\mu_{\lambda f}\right|_0
=\mathbb E[(F-\mathbb EF)\mathcal J(f)],\\
\left|\left.\frac d{d\lambda}\int F\,d\mu_{\lambda f}\right|_0\right|^2
\le\operatorname{Var}(F)\mathbb E\mathcal J(f)^2
\le\operatorname{Var}(F)\kappa\mathbb E\zeta(|f|^2).
\end{gathered}
\tag{YM.Z46}
$$

These identities also hold at finite $N,h$, using the finite descriptor
and $\kappa_h\zeta_{N,h}$. They identify the mean action Hessian with
the descriptor's Fisher information and with the covariance of the
projected recorded innovation current.
:::

:::{prf:proof}
**Recorded thermostat normalization.** Substitution of
{ref}`(YM.Z42) <eq-fg-ym-z42>` into the existing energy sum gives
the first equality of {ref}`(YM.Z43) <eq-fg-ym-z43>`.
For $x=2\gamma h>0$,

$$
1-e^{-x}=\int_0^x e^{-s}ds,\qquad
xe^{-x}\le1-e^{-x}\le x,\qquad e^x-1\le xe^x.
$$

Thus $1\le x/(1-e^{-x})\le e^x$, and

$$
0\le\kappa_h-\kappa
=\kappa\left(\frac{x}{1-e^{-x}}-1\right)
\le\kappa xe^x\le\beta_{\mathrm{eff}}h e^{2\gamma h_0}.
$$

Since $a_{ki}\in\{0,1\}$ and $mh\le T$, the occupation mass is at
most $T$. Multiplication by this mass bound and
$\|f_a\cdot f_b\|_\infty$ proves the stated error.

**Common occupation limit.** Positive measures of mass at most $T$
form a compact metrizable set in the weak-star topology against
$C_0(\mathbb R^{1+d})$. Indeed, choose a countable dense set of tests,
extract a subsequence on which their integrals converge, and extend
the limiting functional using
$|\zeta_{N,h}(\phi)|\le T\|\phi\|_\infty$.
Positivity represents it as a positive measure with mass at most $T$;
the same countable tests metrize this bounded set. This proves
compactness for the occupation coordinate and tightness of its laws.
Combining it with the preceding theorem gives a common joint
subsequence. Tests supported outside the closed time slab have zero
integral at every finite size, and hence in the limit.

Each $f_a\cdot f_b$ is in $C_0$, so its occupation integral is a
continuous coordinate. The deterministic error in
{ref}`(YM.Z43) <eq-fg-ym-z43>` tends to zero and identifies the
limiting energy matrix almost surely. Apply this calculation to the
countable source dictionary already constructed. The bound

$$
|\zeta(f\cdot g)-\zeta(\widetilde f\cdot\widetilde g)|
\le T\bigl(\|f-\widetilde f\|_\infty\|g\|_\infty
 +\|\widetilde f\|_\infty\|g-\widetilde g\|_\infty\bigr)
$$

extends the identity to all Schwartz pairs. Substitution in the
previously constructed likelihood proves its formula in
{ref}`(YM.Z44) <eq-fg-ym-z44>`. Keeping the same descriptor preserves
its proved conditional-density identity. No measurability of the raw
occupation with respect to that smaller descriptor is used.

**Executed innovation covariance.** At each stage the source
coefficients are measurable before its independent standard Gaussian
innovations. Its score increments have conditional mean zero and
conditional cross covariance $\sum_i u_{ki}(f)\cdot u_{ki}(g)$.
Increments at different stages are orthogonal: condition the product
on the history before the later stage, which contains the earlier
increment. Therefore

$$
\mathbb E M_fM_g
=\mathbb E\sum_{k,i}u_{ki}(f)\cdot u_{ki}(g)
=\kappa_h\mathbb E\zeta_{N,h}(f\cdot g).
$$

The uniform fourth score moments make these products uniformly
integrable. Occupation integrals are bounded. Their common joint
convergence passes this identity and the zero first moments to the
limit. The calculation uses the adaptive innovations directly.

**Action Hessian and projection.** Differentiating the actual limiting
likelihood at zero gives

$$
\partial_aL_\infty(0)=M_a,\qquad
\partial_a\partial_bL_\infty(0)=M_aM_b-Q_{ab}.
$$

The derivative envelopes in {ref}`(YM.Z37) <eq-fg-ym-z37>` justify
conditional differentiation. Since $r(0)=1$ and
$\partial_ar(0)=\mathbb E[M_a\mid\widehat Y]=J_a$, differentiation
of $-\log r$ yields

$$
\mathcal A_{ab}
=-\mathbb E[M_aM_b-Q_{ab}\mid\widehat Y]+J_aJ_b.
$$

This gives the first line of {ref}`(YM.Z45) <eq-fg-ym-z45>`.
The preceding covariance calculation gives
$\mathbb E[MM^{\mathsf T}]=\mathbb E Q$.
Also $\mathbb E[M-J\mid\widehat Y]=0$, so both cross terms in
the expansion of $MM^{\mathsf T}$ into $J+(M-J)$ have expectation
zero. This proves the second line. Averaging the Hessian formula
gives $\mathbb E\mathcal A=\mathbb E[JJ^{\mathsf T}]$.
Both this covariance and the residual covariance are positive
semidefinite, proving the inequalities. All terms are integrable
by the source moment bounds.

Finally the derivative envelopes give differentiability of
$r_{\lambda f}$ in $L^2$ at zero, with derivative $\mathcal J(f)$.
Pairing this derivative with $F\in L^2$ proves the response formula;
centering follows from $\mathbb E\mathcal J(f)=0$.
Cauchy--Schwarz and the scalar projection identity prove both
inequalities in {ref}`(YM.Z46) <eq-fg-ym-z46>`.
Each calculation before taking limits also proves the finite native-law
identities with its recorded occupation and descriptor.
:::


:::{prf:theorem} Continuum entropy of the recorded source and its descriptor
:label: thm-ym-source-entropy-identification

Use the joint limit of {prf:ref}`thm-ym-occupation-source-identification`.
Write $\mathbb P$ for that raw limiting law, and define its sourced law
by $d\mathbb P_\theta=L_\infty(\theta)d\mathbb P$. This is the
limiting likelihood of the executed force source. Its descriptor
pushforward is the already constructed $\mu_\theta$.
With $Q_\theta=\kappa\zeta(|f_\theta|^2)$, one has

(eq-fg-ym-z47)=
$$
\begin{gathered}
D_{\mathrm{KL}}(\mathbb P_\theta\Vert\mathbb P)
 =\frac12\mathbb E[L_\infty(\theta)Q_\theta]
 =\frac\kappa2\mathbb E_{\mathbb P_\theta}\zeta(|f_\theta|^2),\\
D_{\mathrm{KL}}(\mu_\theta\Vert\mu)
 =\mathbb E[r(\theta)\log r(\theta)],\\
0\le D_{\mathrm{KL}}(\mu_\theta\Vert\mu)
 \le D_{\mathrm{KL}}(\mathbb P_\theta\Vert\mathbb P)
 \le\frac{\kappa T}{2}\|f_\theta\|_\infty^2,\\
D_{\mathrm{KL}}(\mathbb P_\theta\Vert\mathbb P)
 -D_{\mathrm{KL}}(\mu_\theta\Vert\mu)
 =\mathbb E\left[L_\infty(\theta)
                       \log\frac{L_\infty(\theta)}{r(\theta)}\right].
\end{gathered}
\tag{YM.Z47}
$$

Both finite native source entropies converge to these expressions,
locally uniformly in the source parameter. In particular the finite
descriptor entropy converges for the same descriptor and source law
as its action increment. Their Hessians at zero are

(eq-fg-ym-z48)=
$$
\begin{aligned}
\nabla_\theta^2D_{\mathrm{KL}}(\mathbb P_\theta\Vert\mathbb P)|_0
 &=\mathbb E Q,\\
\nabla_\theta^2D_{\mathrm{KL}}(\mu_\theta\Vert\mu)|_0
 &=\mathbb E[JJ^{\mathsf T}]=\mathbb E\mathcal A,\\
\nabla_\theta^2\bigl(D_{\mathrm{KL}}(\mathbb P_\theta\Vert\mathbb P)
 -D_{\mathrm{KL}}(\mu_\theta\Vert\mu)\bigr)|_0
 &=\mathbb E[(M-J)(M-J)^{\mathsf T}].
\end{aligned}
\tag{YM.Z48}
$$
:::

:::{prf:proof}
The preceding source theorem proves $\mathbb E L_\infty(t\theta)=1$
for all real $t$, with integrable derivative envelopes. Differentiate
at $t=1$ to obtain

$$
\mathbb E\bigl[L_\infty(\theta)
                 (\theta\cdot M-Q_\theta)\bigr]=0.
$$

Since $\log L_\infty(\theta)=\theta\cdot M-Q_\theta/2$, this is
the first entropy identity. Its bound uses
$0\le Q_\theta\le\kappa T\|f_\theta\|_\infty^2$ and
$\mathbb E L_\infty(\theta)=1$.
The descriptor density is $r(\theta)$, so its entropy is
$\mathbb E[r(\theta)\log r(\theta)]$.
Conditional Jensen for the convex function $x\log x$ gives

$$
\mathbb E[L_\infty\log L_\infty\mid\widehat Y]
 \ge r\log r.
$$

Taking expectations proves the entropy inequality. Furthermore,
$\log r$ is descriptor-measurable, and hence
$\mathbb E[L_\infty\log r]=\mathbb E[r\log r]$.
This proves the difference formula. Its terms are integrable:
$L_\infty$ has every positive moment, and
$|\log r|\le r+r^{-1}$ with the moment bounds in
{ref}`(YM.Z37) <eq-fg-ym-z37>`. The same estimates hold before
the limit.

For convergence, the raw entropy equals
$\mathbb E[L_{N,h}(\theta)\theta^{\mathsf T}E_{N,h}\theta]/2$
by the identical differentiation of its normalized likelihood.
The energy matrices are bounded, their joint limit is identified,
and likelihood moments give uniform integrability.
For the descriptor entropy, $r_{N,h}(\theta)$ converges jointly to
$r(\theta)>0$. Since $|x\log x|\le x^2+1$ for $x>0$,
the uniform fourth density moment gives uniform integrability.
Thus both expectations converge at each source parameter.

Here are explicit equicontinuity bounds for this passage. For the
finite source list let $C$ be as in
{prf:ref}`thm-ym-continuum-source-action`. On $|\theta|\le R$,
the gradient norm of the raw entropy is bounded by

$$
\frac{CR^2}{2}V_{1,1,R}+CRV_{0,1,R}.
$$

Indeed, differentiate $L\theta^{\mathsf T}E\theta/2$, use
$\|E\|\le C$, and take expectations of the derivative envelopes.
For the descriptor entropy, differentiation under expectation gives
$\mathbb E[(1+\log r)Dr]$. Its norm is bounded by

$$
V_{1,2,R}\bigl(1+V_{0,2,R}+W_{2,R}\bigr)
$$

by Cauchy--Schwarz and $|\log r|\le r+r^{-1}$.
These constants are independent of $N,h$. Finite nets of each compact
source ball upgrade pointwise convergence to local uniform convergence.

Finally $L(0)=r(0)=1$ and normalization gives
$\mathbb E D^2L(0)=\mathbb E D^2r(0)=0$.
Twice differentiating $\mathbb E[r\log r]$ at zero therefore
leaves $\mathbb E[Dr(0)Dr(0)^{\mathsf T}]=\mathbb E[JJ^{\mathsf T}]$.
The same calculation with $L$ gives $\mathbb E[MM^{\mathsf T}]
=\mathbb E Q$. The derivative and inverse-density envelopes justify
both differentiations. Subtracting and using
{ref}`(YM.Z45) <eq-fg-ym-z45>` proves the residual formula in
{ref}`(YM.Z48) <eq-fg-ym-z48>`.
:::


:::{div} feynman-prose
Fix the recorded geometry and examine the gauge channels on that geometry.
The complete likelihood gives a joint density $b$ and a geometry density
$a$. The conditional gauge density is $b/a$: dividing by the geometry
weight makes its integral on each geometry fiber equal to one. Taking
negative logarithms splits the recorded joint action into these two
contributions.

The actual force source changes both contributions. It weights the geometry
law by $q$ and the conditional gauge law by $r/q$. In the iterated
correlation integral, the two factors of $q$ cancel, leaving the joint
source density $r$ and hence the recorded sourced correlations.

Retaining the geometry source curves together with the joint source curves
preserves this same identity through their common subsequential limit.
The established derivative and inverse-density bounds control the smooth
fiber-action increments. Each retained curve is already a function of the
finite recorded descriptor.
:::

:::{prf:theorem} Exact action on the recorded geometry fibers
:label: thm-ym-native-geometry-fiber-action

Use the joint recorded descriptor $Y=(G,D)$ in
{ref}`(YM.Z7) <eq-fg-ym-z7>`, where $G$ retains the recorded
geometry and $D$ the recorded gauge channels. Use the existing complete
reference path law $R$ and actual law $P=\mathcal L R$ from
{prf:ref}`thm-sm-path-descriptor-density`, with the prescribed survival
normalization already included in $\mathcal L$ when it is used.
Write $\lambda=Y_*R$, $\mu=Y_*P$, and let
$\lambda_G,\mu_G$ be their geometry marginals. Their disintegrations
are denoted by $\lambda^g$ and $\mu^g$. Define

(eq-fg-ym-z52)=
$$
\begin{gathered}
b(g,d)=\mathbb E_R[\mathcal L\mid G=g,D=d],\qquad
a(g)=\mathbb E_R[\mathcal L\mid G=g],\\
d\mu_G(g)=a(g)d\lambda_G(g),\qquad
d\mu^g(d)=\frac{b(g,d)}{a(g)}d\lambda^g(d)
\quad\text{for }\mu_G\text{-almost every }g,\\
S^{\mathrm{joint}}(g,d)=-\log b(g,d),\qquad
S^{\mathrm{geom}}(g)=-\log a(g),\qquad
S^{\mathrm{fiber}}(g,d)=-\log\frac{b(g,d)}{a(g)},\\
S^{\mathrm{joint}}=S^{\mathrm{geom}}+S^{\mathrm{fiber}}.
\end{gathered}
\tag{YM.Z52}
$$

The logarithmic identities hold on the actual support. These reference
fibers are the fibers of the already fixed recorded reference law.
The conditional gauge action and its normalization are therefore
determined by the same complete path likelihood as the joint action.
Every integrable recorded gauge observable obeys

(eq-fg-ym-z53)=
$$
\mathbb E_P F(G,D)
=\int a(g)\lambda_G(dg)
        \int F(g,d)\frac{b(g,d)}{a(g)}\lambda^g(dd).
\tag{YM.Z53}
$$

Next use the actual native force sources of
{prf:ref}`thm-ym-continuum-source-action` at finite $N,h$.
Here the baseline joint law is $\mu=Y_*\mathbb U$,
$r_\theta(Y)=\mathbb E_{\mathbb U}[L_\theta\mid Y]$, and set
$q_\theta(G)=\mathbb E_{\mathbb U}[L_\theta\mid G]$.
The source changes the geometry law and its gauge fiber by

(eq-fg-ym-z54)=
$$
\begin{gathered}
d\mu_{G,\theta}=q_\theta d\mu_G,\qquad
\frac{d\mu_\theta^g}{d\mu^g}(d)
 =\frac{r_\theta(g,d)}{q_\theta(g)},\qquad
\int\frac{r_\theta(g,d)}{q_\theta(g)}\mu^g(dd)=1,\\
\Delta S_\theta^{\mathrm{fiber}}
 =-\log r_\theta+\log q_\theta,\qquad
\Delta S_\theta^{\mathrm{geom}}=-\log q_\theta,\\
\Delta S_\theta^{\mathrm{joint}}
 =\Delta S_\theta^{\mathrm{geom}}+\Delta S_\theta^{\mathrm{fiber}}.
\end{gathered}
\tag{YM.Z54}
$$

Let $M_a=\mathcal I_{N,h}(f_a)$ and $E_{ab}$ be the actual source
score and energy of {ref}`(YM.Z36) <eq-fg-ym-z36>`.
Write $J_a=\mathbb E[M_a\mid Y]$ and
$J_a^G=\mathbb E[M_a\mid G]$. Then the fiber variations at zero are

(eq-fg-ym-z55)=
$$
\begin{aligned}
\partial_a\Delta S^{\mathrm{fiber}}_0
 &=-(J_a-J_a^G),\\
\partial_a\partial_b\Delta S^{\mathrm{fiber}}_0
 &=\mathbb E[E_{ab}\mid Y]-\mathbb E[E_{ab}\mid G]
   -\operatorname{Cov}(M_a,M_b\mid Y)
   +\operatorname{Cov}(M_a,M_b\mid G),\\
\mathbb E[JJ^{\mathsf T}]
 &=\mathbb E[J^G(J^G)^{\mathsf T}]
   +\mathbb E[(J-J^G)(J-J^G)^{\mathsf T}].
\end{aligned}
\tag{YM.Z55}
$$

For a source list and parameter ball as in
{ref}`(YM.Z37) <eq-fg-ym-z37>`, $q$ obeys the same bounds
$V_{n,p,R}$ and $W_{p,R}$ as $r$. Consequently, for $n\ge1$,

(eq-fg-ym-z56)=
$$
\bigl\|\|D^n\Delta S^{\mathrm{fiber}}\|_R\bigr\|_p
\le 2\sum_{\pi\in\Pi_n}(|\pi|-1)!
 W_{|\pi|p(|\pi|+1),R}^{|\pi|}
 \prod_{B\in\pi}V_{|B|,p(|\pi|+1),R}.
\tag{YM.Z56}
$$

At order zero the bound is $2(V_{0,p,R}+W_{p,R})$.
These constants are independent of population and time step in the
native source regime already proved. For the separately specified
survival-selected source law, the same density factorization holds
with its normalized density from {ref}`(YM.Z16) <eq-fg-ym-z16>`;
the selected moment and survival factors remain those proved there.
:::

:::{prf:proof}
**Disintegrate the actual likelihood.** The recorded arrays, finite
graph labels, and countable encodings used in the earlier theorems
are standard Borel. Hence both joint probabilities admit regular
conditional laws over $G$. By the tower property,

$$
\int b(g,d)\lambda^g(dd)=a(g)
\quad\text{for }\lambda_G\text{-almost every }g.
$$

The exceptional set $\{a=0\}$ has $\mu_G$ measure zero because
$\mu_G=a\lambda_G$. On every remaining fiber, $b/a$ is nonnegative
and has integral one. For bounded measurable $u(g)$ and $v(d)$,

$$
\begin{aligned}
\int u(g)\left[\int v(d)\frac{b(g,d)}{a(g)}\lambda^g(dd)\right]
                \mu_G(dg)
&=\int u(g)v(d)b(g,d)\lambda(dg,dd)\\
&=\mathbb E_R[u(G)v(D)\mathcal L].
\end{aligned}
$$

Rectangles generate the descriptor sigma-algebra, so a monotone-class
argument identifies this kernel with $\mu^g$. This proves the density
identities. Their logarithms give the additive action decomposition.
The density $b$ is positive and finite $\mu$-almost everywhere, and
$a$ is positive and finite $\mu_G$-almost everywhere; thus those
logarithms are well defined on the actual support. The same integral
calculation, followed by truncation of positive and negative parts,
proves {ref}`(YM.Z53) <eq-fg-ym-z53>` for every integrable $F$.
In particular it preserves all the existing finite gauge correlations.

**Disintegrate the executed source law.** The force-source theorem
already proves $d\mu_\theta=r_\theta d\mu$ and strict positivity
of $r_\theta$. Its geometry marginal is obtained by the tower property:

$$
\int r_\theta(g,d)\mu^g(dd)
=\mathbb E[r_\theta(Y)\mid G=g]
=\mathbb E[L_\theta\mid G=g]=q_\theta(g).
$$

Repeating the preceding rectangle calculation with baseline $\mu$
and likelihood $r_\theta$ gives the fiber ratio $r_\theta/q_\theta$.
Both factors are positive almost everywhere. Taking logarithms
proves {ref}`(YM.Z54) <eq-fg-ym-z54>`. The geometry factor and
fiber denominator cancel in the joint integral, recovering exactly
$\mathbb E_{\mathbb U}[F(Y)L_\theta]$.

**Calculate the fiber response from recorded scores.** At zero,
$r_0=q_0=1$. The previously proved derivative envelopes justify
conditioning the derivatives of $L_\theta$:

$$
\begin{array}{ll}
\partial_a r_0=\mathbb E[M_a\mid Y],&
\partial_a\partial_b r_0=\mathbb E[M_aM_b-E_{ab}\mid Y],\\
\partial_a q_0=\mathbb E[M_a\mid G],&
\partial_a\partial_b q_0=\mathbb E[M_aM_b-E_{ab}\mid G].
\end{array}
$$

Apply $D^2\log r=r^{-1}D^2r-r^{-2}Dr\otimes Dr$ to the two
logarithms in the fiber action. This gives the first two lines of
{ref}`(YM.Z55) <eq-fg-ym-z55>`, with the adaptive energy terms
retained. Since $G$ is a component of $Y$,
$\mathbb E[J\mid G]=J^G$. Hence
$\mathbb E[(J-J^G)(J^G)^{\mathsf T}]=0$ and its transpose
vanishes. Expanding $J=J^G+(J-J^G)$ proves the information identity.

Finally, the proof of {ref}`(YM.Z37) <eq-fg-ym-z37>` uses only
conditional Jensen and the native likelihood's derivative envelopes.
Conditioning on $G$ gives those same bounds for $q$.
For instance, with the same $C$ and $M$ as there,

$$
\inf_{|\theta|\le R}q_\theta(G)
\ge\exp[-R\mathbb E(|M|\mid G)-CR^2/2],
$$

so the inverse bound is exactly $W_{p,R}$. Apply the partition
formula and Hölder estimate of {ref}`(YM.Z39) <eq-fg-ym-z39>`
to $-\log r$ and $-\log q$, and add their bounds. This proves
{ref}`(YM.Z56) <eq-fg-ym-z56>`. The order-zero estimate follows
from $|\log x|\le x+x^{-1}$. For survival selection the argument
starts with its existing normalized likelihood and preserves that
normalization throughout.
:::


:::{prf:theorem} Geometry-fiber action in the common native continuum limit
:label: thm-ym-native-fiber-continuum

Retain the source curves $q_{N,h}$ of
{prf:ref}`thm-ym-native-geometry-fiber-action` together with the
variables in {prf:ref}`thm-ym-continuum-source-action`. Encode the
recorded geometry by the same countable bounded-coordinate convention
as the joint descriptor, and denote this code by $c_G(G_{N,h})$.
There is a common subsequence with limiting coordinates
$(\mathcal I,\mathcal J,E_\infty,B,r,B_G,q)$.
Define the retained geometry and joint descriptor by

$$
\widehat G=(B_G,q),\qquad
\widehat Y=(B,\mathcal J,r,B_G,q).
$$

At every finite size the added geometry coordinates are measurable
functions of $G_{N,h}$, and hence of the existing joint descriptor.
Let $\mu=\operatorname{Law}(\widehat Y)$ and
$\mu_G=\operatorname{Law}(\widehat G)$, with disintegration $\mu^g$.
Then simultaneously on all bounded source domains,

(eq-fg-ym-z57)=
$$
\begin{gathered}
r_\theta=\mathbb E[L_\infty(\theta)\mid\widehat Y],\qquad
q_\theta=\mathbb E[L_\infty(\theta)\mid\widehat G]
        =\mathbb E[r_\theta\mid\widehat G]>0,\\
d\mu_\theta=r_\theta d\mu,\qquad
d\mu_{G,\theta}=q_\theta d\mu_G,\qquad
\frac{d\mu_\theta^g}{d\mu^g}(y)=\frac{r_\theta(y)}{q_\theta(g)},\\
\Delta S_\infty^{\mathrm{fiber}}(\theta)
=-\log r_\theta+\log q_\theta.
\end{gathered}
\tag{YM.Z57}
$$

The finite fiber-action increments converge jointly to this increment
in $C^\infty_{\mathrm{loc}}$ along that same subsequence. The bounds
in {ref}`(YM.Z56) <eq-fg-ym-z56>` hold in the limit. All finite
polynomial moments of their source derivatives and retained current
pairings converge with bounded continuous descriptor insertions.

For bounded measurable $H$ and $F$, the limiting fiber integral is
exactly the native sourced correlation:

(eq-fg-ym-z58)=
$$
\int H(g)q_\theta(g)\mu_G(dg)
  \int F(y)\frac{r_\theta(y)}{q_\theta(g)}\mu^g(dy)
=\mathbb E[H(\widehat G)F(\widehat Y)L_\infty(\theta)].
\tag{YM.Z58}
$$

For bounded continuous $H,F$, the corresponding finite integrals
converge to this expression, including every source derivative, locally
uniformly in its parameters. The countable source dictionary of
{prf:ref}`thm-ym-continuum-source-action` can be retained on this
same construction.
:::

:::{prf:proof}
The geometry codes lie in the compact countable cube. The curves
$q_{N,h}$ have the same derivative bounds as $r_{N,h}$ by
{prf:ref}`thm-ym-native-geometry-fiber-action`. The finite-net and
Arzela--Ascoli argument already proved for $r_{N,h}$ therefore gives
tightness of $q_{N,h}$ in $C^\infty_{\mathrm{loc}}$ with those same
constants. Combining these bounds with the existing joint tightness
gives the stated common subsequence.

At finite size the augmented joint descriptor is still measurable in
the original $Y_{N,h}$, and the augmented geometry is measurable in
$G_{N,h}$. For bounded continuous $F$ and $H$, respectively,

$$
\begin{aligned}
\mathbb E[F(\widehat Y_{N,h})r_{N,h}(\theta)]
 &=\mathbb E[F(\widehat Y_{N,h})L_{N,h}(\theta)],\\
\mathbb E[H(\widehat G_{N,h})q_{N,h}(\theta)]
 &=\mathbb E[H(\widehat G_{N,h})L_{N,h}(\theta)].
\end{aligned}
$$

Joint convergence and the uniform second moments pass these identities
to the limit. A monotone-class extension identifies the two conditional
expectations in {ref}`(YM.Z57) <eq-fg-ym-z57>`. Since
$\widehat G$ is a component of $\widehat Y$, the tower property
gives $q_\theta=\mathbb E[r_\theta\mid\widehat G]$.
Rational parameter vectors first give common almost-sure versions;
the derivative envelopes extend the identities continuously to all
parameters on each integer ball. Strict positivity follows, as before,
by conditioning the positive lower envelope
$\exp[-R|M_\infty|-CR^2/2]$ of the native likelihood. Thus both
curves have a positive minimum on each compact parameter ball.

The rectangle calculation in
{prf:ref}`thm-ym-native-geometry-fiber-action`, now applied to this
identified limiting joint law, proves the fiber-density ratio and its
normalization. Taking the difference of logarithms is continuous in
$C^\infty$ on each compact parameter ball with positive limiting
minima. Hence the finite fiber-action increments converge jointly in
$C^\infty_{\mathrm{loc}}$. The uniform bounds of
{ref}`(YM.Z56) <eq-fg-ym-z56>` pass by lower semicontinuity, and
their higher-moment versions give uniform integrability for every
stated finite polynomial. This proves the derivative moment limits.

Cancel $q_\theta$ in the iterated integral to obtain
$\mathbb E[H(\widehat G)F(\widehat Y)r_\theta]$.
The first conditional identity in
{ref}`(YM.Z57) <eq-fg-ym-z57>` turns this into
{ref}`(YM.Z58) <eq-fg-ym-z58>`. At finite size the identical
cancellation gives the corresponding complete-record likelihood
integral. For bounded continuous $H,F$, joint convergence and the
likelihood derivative envelopes pass every derivative to the limit.
The next derivative bounds equicontinuity, and finite nets yield
local uniform convergence, exactly as in
{ref}`(YM.Z40) <eq-fg-ym-z40>`.

For a countable source dictionary, retain both $r^{(q)}$ and
$q^{(q)}$ for each initial finite list of tests. Apply the existing
countable-product compactness argument to these paired curves.
Parameter restriction identities hold at finite size and pass to
the limit. The likelihood continuity estimate in
{ref}`(YM.Z41) <eq-fg-ym-z41>` also holds after conditioning on
geometry, by conditional contraction. It therefore extends both
conditional densities to arbitrary Schwartz sources consistently.
All these identities refer to the common native likelihood and its
recorded geometry projection.
:::


:::{prf:theorem} Recorded Wilson response and its link variation
:label: thm-yang-mills-eom

For the recorded Wilson functional of
{prf:ref}`def-wilson-action-ym`, its actual algorithmic source response is

(eq-fg-ym-z21)=
$$
\left.\partial_{\theta_a}\mathbb E_{P_\theta}S_W\right|_0
=\operatorname{Cov}_P(S_W,\Xi_a)
=\sum_{P\in\mathcal P}\beta_P\operatorname{Cov}_P(s_P,\Xi_a).
\tag{YM.Z21}
$$

Here the law and Gaussian sources are exactly those of
{prf:ref}`thm-ym-native-source-response`. The sum uses a fixed finite
list of recorded loop readouts with their specified bounded validity
convention and deterministic coefficients. Every higher response is
obtained by substituting $F=S_W$ into
{ref}`(YM.Z18) <eq-fg-ym-z18>`. Thus this observable's dynamics is
computed with the actual descriptor action, including its selection
normalization. The relation between a source response and a connection
variation is written explicitly in
{prf:ref}`prop-ym-native-connection-variation-identity`, including the
conditional measure term in
{prf:ref}`rem-ym-source-connection-measure-term`.

For the coordinate variation of this same functional, with a
differentiable matter functional $S_m$, stationary links of $S_W+S_m$ obey

(eq-fg-ym-14)=
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

For a smooth continuum Hermitian field with {ref}`(YM.6) <eq-fg-ym-6>`, set
$S_{\rm YM}=\frac12\int\operatorname{Tr}(F_{\mu\nu}F^{\mu\nu})$.
If the matter variation is
$\delta S_m=2\int\operatorname{Tr}(J^\nu\delta A_\nu)$, compactly
supported variations give

(eq-fg-ym-15)=
$$
 D_\mu F^{\mu\nu}=J^\nu.
\tag{YM.15}
$$

A conventional coupling can be included in the definition of $J$.
To obtain {ref}`(YM.15) <eq-fg-ym-15>` as a limit of {ref}`(YM.14) <eq-fg-ym-14>`, the discrete first variations and
sources must converge as well as the actions.
:::

:::{prf:proof}
The Wilson functional is bounded by $2\sum_P\beta_P$. It is therefore
admissible in {prf:ref}`thm-ym-native-source-response`. Substitute it
in {ref}`(YM.Z19) <eq-fg-ym-z19>` and use linearity of covariance to
obtain {ref}`(YM.Z21) <eq-fg-ym-z21>`. The finite-loop masks and
recorded update decisions remain inside each $s_P(Y)$ during this
calculation. This evaluates the algorithmic source derivative even
when the masked readout has no pointwise derivative.

Since $\operatorname{Re}(iz)=-\operatorname{Im}z$,
$d[-\beta_P\operatorname{Re}\operatorname{Tr}(e^{itT^a}U_e\Sigma)/2]/dt$
at zero is the corresponding term in {ref}`(YM.14) <eq-fg-ym-14>`. Stationarity and the source
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

Adding $\delta S_m$ proves {ref}`(YM.15) <eq-fg-ym-15>`. If discrete critical fields converge,
convergence of these first variations against every compactly supported test
variation makes their zero values pass to the continuum equation.
:::

:::{div} feynman-prose
Changing the kinetic source changes how the algorithm samples a recorded
observable. A connection variation instead changes the connection arguments
of that observable, giving its directional derivative $X_fO$. To compare
these operations at fixed recorded geometry, first normalize the conditional
gauge law. This subtracts the geometry score and leaves $s_f=j_f-j_f^G$;
the full source response also retains the change in the geometry law.

The proposed first-order identification must hold after averaging every
test observable against the actual conditional measure, as specified in
{ref}`(YM.Z69) <eq-fg-ym-z69>`. Where a smooth chart description is
available, integration by parts makes the required test explicit: the
source score must equal $X_fS-\operatorname{div}_{\lambda^g}X_f$.
The divergence accounts for the reference volume under the connection
displacement. Thus the source-action derivative $-s_f$ and the coordinate
derivative $X_fS$ enter different places in the calculation. The proposition
computes the source response; the measure identity specifies what must be
verified to represent it by a connection variation.
:::

:::{prf:proposition} Native response and the weak connection-variation identity
:label: prop-ym-native-connection-variation-identity

Use the actual force source $f$ of
{prf:ref}`thm-ym-metric-force-sources`, its complete-record score $M_f$,
and the geometry disintegration $\mu(dg,dd)=\mu_G(dg)\mu^g(dd)$ of
{prf:ref}`thm-ym-native-geometry-fiber-action`. Denote its conditional
scores by

(eq-fg-ym-z66)=
$$
j_f=\mathbb E[M_f\mid G,D],\qquad
j_f^G=\mathbb E[M_f\mid G],\qquad
s_f=j_f-j_f^G,\qquad
\mathbb E[s_f\mid G]=0,\qquad
\|s_f\|_{L^2(\mu)}^2\le\mathbb E M_f^2\le C_f.
\tag{YM.Z66}
$$

The symbols $j_f,j_f^G,s_f$ refer to likelihood scores; the matter
current $J^\nu$ in {ref}`(YM.15) <eq-fg-ym-15>` retains its separate
definition. For every bounded recorded observable $O(g,d)$, the native
source derivative at fixed recorded geometry is

(eq-fg-ym-z67)=
$$
\begin{aligned}
\mathscr R_f^g(O)
&:=\left.\partial_\theta\int O(g,d)\,\mu_\theta^g(dd)\right|_0
 =\int O(g,d)s_f(g,d)\,\mu^g(dd),\\
\left.\partial_\theta\Delta S_\theta^{\mathrm{fiber}}\right|_0
&=-s_f,\qquad
\|\mathscr R_f^G(O)\|_{L^1(\mu_G)}
 \le\sqrt{C_f}\,\|O\|_{L^2(\mu)}.
\end{aligned}
\tag{YM.Z67}
$$

The derivative of the conditional expectation is understood in
$L^1(\mu_G)$; its right-hand side defines its conditional version.
The full response includes the change in the geometry law:

(eq-fg-ym-z68)=
$$
\left.\partial_\theta\mathbb E_{\mu_\theta}O\right|_0
=\mathbb E_\mu[O j_f]
=\mathbb E_{\mu_G}\!\left[
 \mathscr R_f^G(O)+j_f^G\,\mathbb E_\mu[O\mid G]\right].
\tag{YM.Z68}
$$

For a proposed fixed-geometry connection variation $X_f$, acting on
smooth link cylinder observables by directional differentiation, its
identification with this native source response means the following
weak identity on that test class:

(eq-fg-ym-z69)=
$$
\int X_fO\,d\mu^g=\int O s_f\,d\mu^g.
\tag{YM.Z69}
$$

All integrals here use the actual gauge fiber $\mu^g$. Equivalently,
with the distributional convention
$\langle\operatorname{div}(X_f\mu^g),O\rangle
=-\int X_fO\,d\mu^g$, the required identity is
$\operatorname{div}(X_f\mu^g)=-s_f\mu^g$.
It identifies the first-order response on the stated test class;
it does not assert equality of finite-source laws.
:::

:::{prf:proof}
**Conditional likelihood.** The existing likelihood is
$L_\theta=\exp(\theta M_f-\theta^2E_f/2)$. Its derivative at zero is
$M_f$. Set $r_\theta=\mathbb E[L_\theta\mid G,D]$ and
$q_\theta=\mathbb E[L_\theta\mid G]$. Their baseline values are one,
and their first derivatives are $j_f$ and $j_f^G$. The quotient rule
therefore gives

$$
\left.\partial_\theta\frac{r_\theta}{q_\theta}\right|_0
=j_f-j_f^G=s_f,
\qquad
\left.\partial_\theta
 \left(-\log r_\theta+\log q_\theta\right)\right|_0=-s_f.
$$

The derivative and inverse-density bounds in
{ref}`(YM.Z37) <eq-fg-ym-z37>` justify these differentiations in
$L^p$ on bounded source intervals. The tower property gives
$\mathbb E[j_f\mid G]=j_f^G$, and orthogonality gives

$$
\mathbb E s_f^2
=\mathbb E j_f^2-\mathbb E(j_f^G)^2
\le\mathbb E M_f^2\le C_f.
$$

This proves {ref}`(YM.Z66) <eq-fg-ym-z66>`. Differentiating the
conditional density $r_\theta/q_\theta$ under the fiber integral
proves {ref}`(YM.Z67) <eq-fg-ym-z67>`. Its norm estimate follows from
$\mathbb E|\mathbb E[O s_f\mid G]|
\le\mathbb E|O s_f|\le\|O\|_2\|s_f\|_2$.
Differentiating the joint density $r_\theta$ and splitting
$j_f=s_f+j_f^G$ proves {ref}`(YM.Z68) <eq-fg-ym-z68>`.

**Connection differentiation.** At a fixed recorded geometry, a
connection displacement changes a test observable by $X_fO$.
The native source changes its fiber expectation by
$\mathscr R_f^g(O)$. Equality of these two first-order responses for
every test is exactly {ref}`(YM.Z69) <eq-fg-ym-z69>`.
This statement remains meaningful for singular fiber laws: it defines
the derivative distribution by its action on tests. The distributional
divergence formula is the same identity with a minus sign, by its
specified convention. No replacement of the actual fiber by Haar
measure is used.

**Existing continuum passage.** Restrict here to the spacetime source
tests and retained observables of
{prf:ref}`thm-ym-continuum-source-action`. For their source-dependent fiber
likelihoods, {prf:ref}`thm-ym-native-fiber-continuum` gives convergence
of the action increments and their derivatives on the common
subsequence. Consequently the source-score identities
{ref}`(YM.Z66) <eq-fg-ym-z66>`–{ref}`(YM.Z68) <eq-fg-ym-z68>` have
those same limiting versions, with the inherited $C_f$ bound and the
retained limiting observables specified by that theorem. The actual source-induced update and readout derivatives are calculated
in {prf:ref}`lem-ym-native-source-readout-tangent`. They include a
geometry displacement and are defined on the specified differentiable
branches. To use $X_fO$ in {ref}`(YM.Z69) <eq-fg-ym-z69>`, the
fixed-geometry response must retain the contributions of geometry
conditioning and branch boundaries for these same observables.
:::

:::{prf:lemma} Force-source tangent of the executed update and recorded channels
:label: lem-ym-native-source-readout-tangent

Use the force source of {prf:ref}`thm-ym-metric-force-sources` in the
`KineticOperator.apply` execution of
{prf:ref}`thm-ym-kinetic-metric-correspondence`. Fix a realized pre-O
history and differentiate this step with respect to its source parameter
at zero. Write a dot for that derivative and retain all Gaussian draws.
The source contribution at this O stage and the following A stage is

$$
\dot v^{O}_{ki}=\frac{h}{c_h\sqrt N}\Sigma_{ki}f_{ki},
\qquad
\dot x^{\mathrm{end}}_{ki}
=\frac{h^2}{2c_h\sqrt N}\Sigma_{ki}f_{ki}.
$$

The second formula uses the executed Euclidean A update
$x\leftarrow x+(h/2)v$; the final B block and velocity squashing leave
this position unchanged. On the full Hessian branch these quantities are
$h g_{ki}^{-1/2}f_{ki}/\sqrt N$ and
$h^2 g_{ki}^{-1/2}f_{ki}/(2\sqrt N)$, respectively. In particular, for
an alive slot with invertible $\Sigma_{ki}$, this one-stage source keeps
its recorded endpoint position fixed precisely when $f_{ki}=0$.

On a differentiable branch of the remaining executed maps, its endpoint
velocity derivative is

$$
\dot v^{\mathrm{end}}
=D\psi_v\left[
 D_x\mathcal B\,\dot x^{\mathrm{end}}
 +D_v\mathcal B\,\dot v^O\right].
$$

Here $\mathcal B$ denotes the existing final `_apply_boris_kick` call,
including both force half-kicks, its Boris rotation, and the second
viscous-force evaluation. Its other arguments are held at the values
supplied by this call. When velocity squashing is disabled,
$D\psi_v=I$. For a source acting at several stages, apply the same chain
rule to the actual chronological stage maps, adding
$h\Sigma_{ki}f_{ki}/(c_h\sqrt N)$ at each O stage. Earlier changes in
fitness, diffusion, geometry, and copied states enter through those
maps' derivatives. A recorded force evaluated before the current O stage
has zero derivative with respect to that stage's source when its preceding
history is fixed.

For a recorded viscous-force evaluation of
{prf:ref}`thm-sm-su3-emergence`, retain its actual edge weights $K_{ij}$.
On a fixed graph and validity branch its induced derivative is

$$
\dot F_i^{\mathrm{visc}}
=\nu\sum_j K_{ij}(\dot v_j-\dot v_i)
 +\nu\sum_j\dot K_{ij}(v_j-v_i).
$$

For the normalized Gaussian branch of
{prf:ref}`def-latent-fractal-gas-viscous-force`, put
$\ell=\ell_{\mathrm{visc}}$. Its normalization contributes exactly

$$
\begin{aligned}
L_{ij}&=-\frac{(x_i-x_j)\cdot(\dot x_i-\dot x_j)}{\ell^2},\\
\dot K_{ij}&=K_{ij}\left(L_{ij}-\sum_lK_{il}L_{il}\right).
\end{aligned}
$$

Consequently, with $X=\max_j\|\dot x_j\|$,
$V=\max_j\|v_j\|$, and $W=\max_j\|\dot v_j\|$ on its alive set,

$$
\|\dot F_i^{\mathrm{visc}}\|
\le 2\nu W+
 \frac{8\nu VX}{\ell^2}\sum_jK_{ij}\|x_i-x_j\|.
$$

For the other implemented weight branches, $\dot K_{ij}$ differentiates
the supplied weight, its destination-volume factor, normalization, and
cap on the chosen branch. A weight stored before the current source stage
is constant in that stage's calculation. These evaluation conventions
retain the actual force used in the record.

Substitute these $\dot F_i^{\mathrm{visc}}$ and $\dot v_i$ into the
color and contraction derivatives already calculated in
{prf:ref}`thm-sm-direct-existing-machinery`. On valid colors the resulting
quantitative bounds are

$$
\begin{aligned}
\|\dot c_i\|&\le
 \frac{\|\dot F_i^{\mathrm{visc}}\|}{\|F_i^{\mathrm{visc}}\|}
 +|\kappa|\|\dot v_i\|,\\
|\dot\Pi_{ijk}|&\le
 2\bigl(\|\dot c_i\|+\|\dot c_j\|+\|\dot c_k\|\bigr).
\end{aligned}
$$

The denominator is bounded below by the recorded validity threshold
$\delta_c$ on this branch. The direct pair, determinant, and doublet
bounds are those of {prf:ref}`prop-sm-channel-estimate-routes` with these
same induced directions.

For a normalized metric readout
$\Phi=\sum_i p_i B_i f(x_i)$ in
{prf:ref}`thm-ym-same-record-metric-field`, restrict to a differentiable
branch with positive retained cell weights and fixed masks. Set

$$
\ell_i=\frac{\dot b_i}{b_i}
       +\frac12\operatorname{Tr}(g_i^{-1}\dot g_i).
$$

Then the actual readout derivative, including its normalization, is

$$
\dot\Phi
=\sum_i p_i\left[\dot B_i f(x_i)
             +B_i\nabla f(x_i)\cdot\dot x_i\right]
 +\sum_i p_i\ell_i\bigl(B_i f(x_i)-\Phi\bigr).
$$

For $|B_i|\le C$ it obeys

$$
|\dot\Phi|
\le\|f\|_\infty\max_i|\dot B_i|
 +C\|\nabla f\|_\infty X
 +2C\|f\|_\infty\max_i|\ell_i|.
$$

All derivatives in this statement concern the existing executed update
and its readouts. Their image gives the realizable channel directions
on the stated branch. The source also has the displayed geometry
component. Its fixed-geometry weak response remains the conditional
identity {ref}`(YM.Z67) <eq-fg-ym-z67>`, including the geometry score.
:::

:::{prf:proof}
The pre-O state, noise amplitude, and predictable test are fixed in the
one-stage calculation. Differentiating the added Gaussian mean gives
$\dot v^O$; differentiating the following A update gives
$\dot x^{\mathrm{end}}=(h/2)\dot v^O$. The final B and squashing calls
change only velocity. Their chain rule proves the endpoint formula.
The invertibility assertion follows from $h,c_h>0$ and invertibility of
$\Sigma_{ki}$. Iterating this derivative through the chronological maps
gives the multistage formula, at each step keeping the evaluation time
of every recorded coefficient.

Differentiate the recorded viscous sum term by term. For the Gaussian
branch, differentiation of its numerator gives $L_{ij}$ and of its
normalizer gives $\sum_lK_{il}L_{il}$. Since $\sum_jK_{ij}=1$,

$$
\sum_j|\dot K_{ij}|
\le2\sum_jK_{ij}|L_{ij}|
\le\frac{4X}{\ell^2}\sum_jK_{ij}\|x_i-x_j\|.
$$

Use $\|v_j-v_i\|\le2V$ and
$\|\dot v_j-\dot v_i\|\le2W$ to obtain the force bound.
The existing color and contraction estimates then give the asserted
bounds without treating edge variations as independent inputs.

Finally, differentiate
$p_i=b_i\sqrt{\det g_i}/\sum_jb_j\sqrt{\det g_j}$ to obtain
$\dot p_i=p_i(\ell_i-\sum_jp_j\ell_j)$. Substitution in the derivative
of $\Phi$ gives its formula and bound. Zero weights, changes of graph,
validity jumps, and cap boundaries retain their actual branch rules.
For expectations involving those boundaries the complete likelihood
response in {prf:ref}`thm-ym-metric-force-sources` applies; integrating
only the interior derivatives calculated here requires a separate
justification of the boundary contribution. The lemma makes no such
interchange of derivative and expectation.
:::

:::{prf:remark} Coordinate form of the identification and the Yang–Mills force
:label: rem-ym-source-connection-measure-term

The measure term in {ref}`(YM.Z69) <eq-fg-ym-z69>` can be calculated
explicitly wherever a differentiable chart description of the actual
fiber is available. Write that description as
$d\mu^g=e^{-S}d\lambda^g$ and $d\lambda^g=w(z)\,dz$,
with $X=X^i\partial_i$. For compactly supported tests in such a chart,
integration by parts gives

(eq-fg-ym-z70)=
$$
\begin{aligned}
\int XO\,e^{-S}w\,dz
&=-\int O\,\partial_i(X^i e^{-S}w)\,dz\\
&=\int O\left[XS-\frac1w\partial_i(wX^i)\right]d\mu^g
 =\int O[XS-\operatorname{div}_{\lambda^g}X]d\mu^g.
\end{aligned}
\tag{YM.Z70}
$$

Thus, in this chart description, the weak correspondence is
$s_f=X_fS-\operatorname{div}_{\lambda^g}X_f$, rather than an
equality between the source-action derivative $-s_f$ and the coordinate
action derivative $X_fS$. Formula
{ref}`(YM.Z69) <eq-fg-ym-z69>` is the measure-level formulation and
requires no assertion that every native fiber has a smooth chart density.

For the Wilson and matter functionals already differentiated in
{prf:ref}`thm-yang-mills-eom`, a link variation
$U_e\mapsto e^{it\eta_e^aT^a}U_e$ gives

(eq-fg-ym-z71)=
$$
X(S_W+S_m)
=\sum_{e,a}\eta_e^a\left[
 \frac12\sum_{P\ni e}\beta_P
       \operatorname{Im}\operatorname{Tr}(T^aU_e\Sigma_{P,e})
 -J_e^a\right].
\tag{YM.Z71}
$$

Indeed, differentiating each oriented occurrence gives its displayed
trace term; differentiating the matter functional gives $-J_e^a$.
For the smooth continuum functional in that same theorem, the
corresponding calculation is

(eq-fg-ym-z72)=
$$
X(S_{\mathrm{YM}}+S_m)
=2\int\operatorname{Tr}\!\left[
 (J^\nu-D_\mu F^{\mu\nu})\,\delta A_\nu\right].
\tag{YM.Z72}
$$

The source-score calculation proves
{ref}`(YM.Z67) <eq-fg-ym-z67>`, while the link and curvature
calculations prove {ref}`(YM.Z71) <eq-fg-ym-z71>` and
{ref}`(YM.Z72) <eq-fg-ym-z72>`. To use the latter as the force in
the native identity, one must establish
{ref}`(YM.Z69) <eq-fg-ym-z69>` for the induced connection variations
and identify the action and reference-measure contribution in that
identity. None of these substitutions follows from equating the names
of the two derivatives. In a quantum field law the relevant equation
is the identity with test-observable insertions; stationarity of every
sampled connection is not asserted.
:::

:::{prf:lemma} Exact graph Ward identity
:label: lem-ym-discrete-ward

Use the invariant inner product $\langle X,Y\rangle=-\operatorname{Re}\operatorname{Tr}(XY)$
on anti-Hermitian matrices. Let the left link force $G_{ij}$ of an invariant
action satisfy
$dS=\sum_{i\to j}\langle G_{ij},\delta U_{ij}U_{ij}^{-1}\rangle$
plus its matter variations. If $R_i$ denotes the coefficient of the matter
variation generated by $X_i\Psi_i$, then

(eq-fg-ym-16)=
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
of each arbitrary $X_i$ yields {ref}`(YM.16) <eq-fg-ym-16>`. The matter coefficient vanishes when
its Euler–Lagrange derivatives vanish.
:::

(sec-ym-path-integral)=
## 6. Finite field measures

:::{prf:definition} Partition and generating functional of the recorded law
:label: def-partition-function-ym

Use the complete likelihood $\mathcal L=dP/dR$ constructed in
{prf:ref}`thm-ym-recorded-action-emergence` and the actual descriptor
$Y=\mathscr D(\omega)$. Its reference and probability laws are
$\lambda=\mathscr D_*R$ and $\nu=\mathscr D_*P$. Define
$a(y)=\mathbb E_R[\mathcal L\mid Y=y]$ and
$S_{\mathrm{alg}}=-\log a$ on the actual support. The normalized
partition and observable characteristic functional are

(eq-fg-ym-17)=
$$
\begin{gathered}
Z_{\mathrm{alg}}=\int e^{-S_{\mathrm{alg}}}\,d\lambda
=\mathbb E_R\mathcal L=1,\qquad
 d\nu=e^{-S_{\mathrm{alg}}}\,d\lambda,\\
\mathcal Z(J)=\int\exp\!\left(i\sum_{a=1}^mJ_aO_a(y)\right)
                           e^{-S_{\mathrm{alg}}(y)}\lambda(dy)
=\mathbb E_P\exp\!\left(i\sum_{a=1}^mJ_aO_a(Y)\right).
\tag{YM.17}
$$

Here $O_a$ are specified bounded recorded observables and $J_a\in\mathbb R$.
In particular $\mathcal Z(0)=1$, $|\mathcal Z(J)|\le1$, and
$\partial_{J_{a_1}}\cdots\partial_{J_{a_k}}\mathcal Z(0)
=i^k\mathbb E_P\prod_{j=1}^kO_{a_j}(Y)$.
Boundedness justifies each derivative by dominated convergence.
The characteristic functional records the original law; it changes no
transition rule. The source-dependent algorithm laws themselves are those
constructed from the actual updates in
{prf:ref}`thm-ym-metric-force-sources`.
:::

:::{prf:theorem} Frame covariance of the recorded field integral
:label: thm-path-integral-gauge-invariance

Express the same covered record in a different internal frame by its
invertible coordinate map $C$. Push forward both descriptor measures:
$\lambda'=C_*\lambda$ and $\nu'=C_*\nu$. Then
$a'=a\circ C^{-1}$ is the density of $\nu'$ relative to $\lambda'$.
For every recorded observable, expressed in the new frame as
$O'=O\circ C^{-1}$,

$$
\int O'e^{-S'_{\mathrm{alg}}}\,d\lambda'
=\int Oe^{-S_{\mathrm{alg}}}\,d\lambda
=\mathbb E_P O(Y).
$$

The Wilson observables of
{prf:ref}`thm-wilson-action-gauge-invariance` have equal values in these
frames. Their characteristic functional in
{ref}`(YM.17) <eq-fg-ym-17>` is consequently frame-independent.
:::

:::{prf:proof}
For bounded $F$, change variables under pushforward to obtain

$$
\int F(y')\,d\nu'(y')
=\int F(Cy)a(y)\,d\lambda(y)
=\int F(y')a(C^{-1}y')\,d\lambda'(y').
$$

This proves the claimed density, and substitution of $F=O'$ proves the
integral identity. For a loop, adjacent endpoint frame changes cancel
inside the ordered product; the remaining basepoint conjugation leaves
its trace unchanged. Apply this to each insertion in the characteristic
functional. The transformation changes coordinates of the same record
and transports its reference measure. Invariance of the probability
under an active transformation with a fixed reference is a separate law
identity, checked as in
{prf:ref}`rem-ym-physical-symmetry-application`.
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

:::{prf:proposition} Recorded density, support, and descriptor refinement
:label: prop-ym-density-and-support

Use the complete reference $R$, actual law $P=\mathcal L R$, and
recorded descriptor $Y$ of {prf:ref}`def-partition-function-ym`.
Then $\lambda=Y_*R$ and $\nu=Y_*P$ satisfy

$$
\frac{d\nu}{d\lambda}(y)
=a(y)=\mathbb E_R[\mathcal L\mid Y=y],\qquad
\int a\,d\lambda=1,\qquad \nu\{a=0\}=0.
$$

For a richer covered descriptor $Z$ with $Y=p(Z)$ and density
$a_Z=\mathbb E_R[\mathcal L\mid Z]$, the same algorithmic law obeys

$$
a(Y)=\mathbb E_R[a_Z(Z)\mid Y],\qquad
\mathbb E_P F(Y)=\mathbb E_P F(p(Z)).
$$

Consequently adding recorded coordinates and then integrating them out
preserves the original field law and all integrable original observables.
For a specified survival event $E$ with $P(E)>0$, the selected version
uses $\mathcal L_E=\mathcal L\mathbf1_E/P(E)$ in every one of these
formulas, including the partition.

*Proof.* For bounded $F$, conditional expectation gives

$$
\begin{aligned}
\mathbb E_P F(Y)
&=\mathbb E_R[F(Y)\mathcal L]\\
&=\mathbb E_R[F(Y)\mathbb E_R[\mathcal L\mid Y]]
 =\int F(y)a(y)\,d\lambda(y).
\end{aligned}
$$

Setting $F=1$ proves normalization. Setting $F=\mathbf1_{\{a=0\}}$
proves the support statement. Since $\sigma(Y)\subset\sigma(Z)$,
the tower property gives
$\mathbb E_R[\mathbb E_R[\mathcal L\mid Z]\mid Y]
=\mathbb E_R[\mathcal L\mid Y]$. The equality $Y=p(Z)$ gives the
observable identity directly. Truncation extends it to integrable
observables. Finally $\mathbb E_R\mathcal L_E=1$ and
$\mathbb E_R[F(Y)\mathcal L_E]=\mathbb E_P[F(Y)\mid E]$,
so the same proof applies to the prescribed selected law.
$\square$
:::

:::{div} feynman-prose
The partition in {ref}`(YM.17) <eq-fg-ym-17>` is normalized by the
algorithm itself. Its complete likelihood has expectation one under the
recorded reference law. Averaging that likelihood over records with the
same descriptor gives the field density and preserves its normalization.
Differentiating the characteristic functional then recovers the recorded
channel correlations. The edge transports, geometry, and masks keep the
joint distribution generated by the run.

To express those records in another internal frame, transform both the
probability measure and its reference measure, together with the observable.
The density becomes the original density composed with the inverse
coordinate map. Changing variables therefore gives exactly the same
expectation. This is the frame covariance of the recorded field integral,
with its actual joint law carried through the transformation.
:::

If killed-kernel eigenobjects are available,
{prf:ref}`prop-kl-doob-transform` provides the exact stationary alternative

(eq-fg-ym-18)=
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

(eq-fg-ym-19)=
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
commute. Taking the normalized trace proves {ref}`(YM.19) <eq-fg-ym-19>`.

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

(eq-fg-ym-20)=
$$
 |\operatorname{Cov}_\pi(F(S_0),G(S_t))|
 \le M e^{-\lambda t}
 \sqrt{\operatorname{Var}_\pi F\operatorname{Var}_\pi G}.
\tag{YM.20}
$$

If a Euclidean spacetime field law is also invariant under rotations mixing
time and space, and the rotated observables have the same norm bounds,
{ref}`(YM.20) <eq-fg-ym-20>` transfers to separations along any rotated axis. These are additional
symmetries of that field law.
:::

:::{prf:proof}
The covariance is $\langle F-\pi F,P_t(G-\pi G)\rangle_\pi$.
Cauchy–Schwarz and the assumed semigroup estimate prove {ref}`(YM.20) <eq-fg-ym-20>`.
For the second assertion, change variables by the stated Euclidean rotation
in the correlation and apply the first assertion to the rotated observables.
:::

For a bounded single-walker observable $|\varphi|\le B$,
{prf:ref}`thm-mixing-variance-corrected` gives the useful same-frame estimate

(eq-fg-ym-21)=
$$
 \operatorname{Var}_{\pi_N}\!\left(\frac1N\sum_i\varphi(z_i)\right)
 \le\frac{4B^2}{N}\left(H_N+\frac12\log2\right),
 \qquad H_N=D_{\rm KL}(\pi_N\Vert\rho^{\otimes N}).
\tag{YM.21}
$$

For an $L$-Lipschitz single-walker observable, {ref}`(YM.1) <eq-fg-ym-1>` gives instead
$\operatorname{Var}(N^{-1}\sum_i\varphi(z_i))\le C_*L^2/N$.
A loop observable depending on a reconstructed graph is a function of the
whole swarm. The applicable bound is $C_*\int|\nabla W|^2d\pi_N$ after
establishing its regularity on the relevant status stratum. The factor $1/N$
requires the corresponding bound on this gradient; discontinuous changes of
neighbors need their own control.

Temporal averaging also has an elementary error estimate. For stationary
centered $F$ satisfying {ref}`(YM.20) <eq-fg-ym-20>`,

(eq-fg-ym-22)=
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

(eq-fg-ym-23)=
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

(eq-fg-ym-24)=
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
Multiply by $\beta_P$, sum, and use {ref}`(YM.23) <eq-fg-ym-23>`. This is the same local
quadratic estimate used in {prf:ref}`lem-lqft-wilson-quadratic`.
:::

For example, on a $D$-dimensional cubic mesh of spacing $a$ in a bounded
region, with one face per $\mu<\nu$ at each site and
$X_P=ig a^2F_{\mu\nu}+O(a^3)$ uniformly, choose

(eq-fg-ym-25)=
$$
 \beta=\frac{4a^{D-4}}{g^2}.
\tag{YM.25}
$$

Then $\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$ gives the leading sum
$\frac12\sum_{x,\mu<\nu}a^D(F_{\mu\nu}^a)^2$, converging to
$\frac14\int\sum_{\mu,\nu,a}(F_{\mu\nu}^a)^2$.
The accumulated mixed remainder is $O(a)$ and the quartic error is $O(a^4)$.
In $D=4$, {ref}`(YM.25) <eq-fg-ym-25>` is $\beta=4/g^2$. This example verifies the hypotheses
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

(eq-fg-ym-26)=
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

If the one-loop truncation of {ref}`(YM.26) <eq-fg-ym-26>` is the adopted running equation, then

(eq-fg-ym-27)=
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

(eq-fg-ym-28)=
$$
 \hbar_{\rm eff}=\frac{m\epsilon_c^2}{2\tau}.
\tag{YM.28}
$$

It has units of action. Its factor of two belongs to this specified convention.
:::

:::{prf:proof}
Equating the coefficients gives
$1/(2\epsilon_c^2)=m/(4\hbar_{\rm eff}\tau)$, hence {ref}`(YM.28) <eq-fg-ym-28>`.
Since $[m\epsilon_c^2/\tau]=ML^2/T$, the units agree. A different quadratic
normalization changes the coefficient; experimental Planck normalization
requires a calibration of that convention and the field observables.
:::

:::{prf:theorem} Dimensionless weak-coupling proxy
:label: thm-su2-coupling-constant

For a reference action unit $\hbar_0$ and speed unit $c$, define the
parameter proxy

(eq-fg-ym-29)=
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
$\hbar_{\mathrm{eff}}=m\epsilon_c^2/(2\tau)$ from {ref}`(YM.28) <eq-fg-ym-28>`. Then

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
Division of {ref}`(YM.29) <eq-fg-ym-29>` by this positive expression gives the ratio.
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

(eq-fg-ym-30)=
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

(eq-fg-ym-31)=
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
Cancel the common factors in {ref}`(YM.30) <eq-fg-ym-30>`.
:::

:::{prf:theorem} Scaling of the declared constants
:label: thm-rg-flow-constants

At fixed $m,c,\hbar_0$, the rescaling
$\tau\mapsto s\tau$, $\epsilon_c\mapsto\sqrt s\epsilon_c$,
$\rho\mapsto\sqrt s\rho$ preserves {ref}`(YM.28) <eq-fg-ym-28>` and $\rho/\epsilon_c$, and
sends $\widehat g_{2,\mathrm{clock}}^{\,2}\mapsto s\widehat g_{2,\mathrm{clock}}^{\,2}$.
If an ultraviolet momentum is proportional to $1/\rho$, this power law
has a different form from the logarithmic law {ref}`(YM.27) <eq-fg-ym-27>`.
:::

:::{prf:proof}
Substitute the three rescalings into {ref}`(YM.28) <eq-fg-ym-28>` and {ref}`(YM.29) <eq-fg-ym-29>`. With
$\mu\propto1/\rho$, one has $s\propto\mu^{-2}$ along this family,
whereas {ref}`(YM.27) <eq-fg-ym-27>` gives $g^2\sim1/\log\mu$. Thus this parameter rescaling
is an algebraic calibration family. A renormalization trajectory must instead
keep the chosen physical observables fixed and determine its running from them.
:::

:::{prf:definition} Observable validation targets
:label: def-experimental-signatures

A validation record specifies the link lift and orientations, measured loop
traces, action normalization, time and length conversions, and the correlation
channel. It includes uncertainty from {ref}`(YM.21) <eq-fg-ym-21>` or {ref}`(YM.22) <eq-fg-ym-22>` when their law and
regularity hypotheses hold. Field-current residuals include the sources in
{ref}`(YM.11) <eq-fg-ym-11>`; fitness residuals use {ref}`(YM.8) <eq-fg-ym-8>`. Reported chirality masks and scalar
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

The full-gradient LSI {ref}`(YM.1) <eq-fg-ym-1>` is available from the actual-law criteria in
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

(eq-fg-ym-32)=
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
transition family in {ref}`(YM.32) <eq-fg-ym-32>`. A uniform rate along such a change requires
uniform analytic constants for that family.

### 10.2. A gap that passes to a limit

:::{prf:theorem} Survival of a uniformly controlled self-adjoint gap
:label: thm-mass-gap-rg-fixed-point

Let $H_a\ge0$ be self-adjoint on $\mathcal H_a$, with normalized
$\Omega_a\in\ker H_a$, and suppose

(eq-fg-ym-33)=
$$
 \|e^{-tH_a}(I-P_{\Omega_a})\|\le e^{-\lambda_*t},
 \qquad \lambda_*>0,
\tag{YM.33}
$$

uniformly in the cutoff and volume parameters under consideration.
Let $J_a:\mathcal H_a\to\mathcal H$ be isometries with
$J_aJ_a^*\to I$ strongly and $J_a\Omega_a\to\Omega$. Suppose, for every
$t>0$,

(eq-fg-ym-34)=
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
Since $e^{-tH_a}\Omega_a=\Omega_a$, {ref}`(YM.34) <eq-fg-ym-34>`, contraction of each operator,
and $J_a\Omega_a\to\Omega$ imply $e^{-tH}\Omega=\Omega$.
For $f\in\mathcal H$ take $f_a=J_a^*f$. Then $J_af_a\to f$ and
$J_aP_{\Omega_a}f_a\to P_\Omega f$. Applying {ref}`(YM.33) <eq-fg-ym-33>` to $f_a$ and
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
{ref}`(YM.34) <eq-fg-ym-34>`. Population-uniform estimates do not automatically control an
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
3. A local weak defect and stability bound as in {ref}`(YM.32) <eq-fg-ym-32>` produce the displayed
   finite-time global error.

These estimates control a kinetic substep, an inverse matrix, and a specified
discretization error, respectively.
:::

:::{prf:proof}
The Gaussian increment is centered and independent, so its cross term with
$v$ vanishes and its squared norm has mean $d$. Diagonalize $A$ to obtain
the inverse bound. The third assertion is {ref}`(YM.32) <eq-fg-ym-32>`. Uniform application along
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
dynamics represented by the recorded process. Now we can construct a field
limit and ask which time evolution supplies its energy.

Start with a smooth empirical average, subtract its exact finite-population
mean, and multiply by $\sqrt N$. The stationary LSI bounds these fluctuations
after integration against spacetime tests. The bounds control a whole random
distribution and yield one subsequence on which every correlation order
converges. Computing the full update then gives the drift and noise terms
that any identified algorithmic fluctuation evolution must retain.

Next use the gradient energy in that LSI to construct its equilibrium
transfer. Self-adjointness makes the reflected past and future meet in a
squared Hilbert-space norm. This proves reflection positivity and supplies
a temporal gap. For the specified bounded whole-joint tilt family, a lower
bound also keeps a nonzero fluctuation alive in the continuum limit. Product
independence goes further and determines the entire Gaussian hierarchy.

There are two clocks here: the recorded algorithmic clock and the equilibrium
energy clock. Their common one-time law fixes static moments. Their respective
generators fix the time correlations used in each reconstruction.
:::

### 12.1. Fields, covariance, and the spectrum

:::{prf:definition} Recorded fields and their observable Hilbert space
:label: def-wightman-field-fg

For a slice with $N$ walker records at specified coordinates $X_i$, define
the intensive empirical field

(eq-fg-ym-35)=
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

For every fixed $n$, the correlations of {ref}`(YM.35) <eq-fg-ym-35>` satisfy

(eq-fg-ym-36)=
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
The empirical measure in {ref}`(YM.35) <eq-fg-ym-35>` is positive with total mass at most one.
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

Under {ref}`(YM.1) <eq-fg-ym-1>`, let $F$ be a real $L$-Lipschitz observable with finite mean on
the continuous joint state space, with $L>0$. Then

(eq-fg-ym-37)=
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
First truncate $F$ smoothly. Apply {ref}`(YM.1) <eq-fg-ym-1>` to $e^{tF/2}$, and put
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

Use the actual joint law $\pi_N$ covered by {ref}`(YM.1) <eq-fg-ym-1>`, with a common $C_*$.
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
For the stationary family,
{prf:ref}`thm-ym-spacetime-fluctuation-compactness` below supplies the
test-function topology, distributional tightness, and a common
subsequential hierarchy at every order. $\square$
:::

### Spacetime fluctuations of the reconstructed empirical field

:::{div} feynman-prose
A field is measured by averaging it against a smooth test function. We need
control as that test varies, so that the limiting object answers all these
measurements consistently. Stationarity gives the same concentration estimate
at each time. Jensen's inequality combines those estimates under a time
integral, even when observations at different times are strongly dependent.

The Hermite expansion below organizes the resulting measurements into modes.
Their weighted bounds give tightness in one distribution space, and the
higher moments let every correlation order follow the same converging
subsequence. This proves existence of a hierarchy. Its particular covariance
still depends on the chosen dynamics. Here the test coordinates are recorded
time and particle phase space; their use does not assert relativistic
spacetime covariance.
:::

:::{prf:theorem} Distribution-valued compactness from the established stationary LSI
:label: thm-ym-spacetime-fluctuation-compactness

Use the stationary continuous joint-law family and uniform constant $C_*$
already identified in {prf:ref}`cor-ym-empirical-fluctuations`. Write
$z\in\mathbb R^{2d}$ and let $\varphi\in\mathcal S(\mathbb R\times
\mathbb R^{2d};\mathbb R)$. The reconstructed empirical fluctuation is

(eq-fg-ym-f1)=
$$
\mathcal Z_N(\varphi)
=\int_{\mathbb R}\sqrt N\left[
 \frac1N\sum_i\varphi(t,Z_i(t))
 -\pi_N\left(\frac1N\sum_i\varphi(t,Z_i)\right)\right]dt,
\quad
q(\varphi)=\int_{\mathbb R}\|\nabla_z\varphi(t,\cdot)\|_\infty dt.
\tag{YM.F1}
$$

For a stationary discrete algorithm, $Z_i(t)$ denotes its piecewise constant
record at the fixed observation spacing. Then

(eq-fg-ym-f2)=
$$
\begin{aligned}
\mathbb E e^{u\mathcal Z_N(\varphi)}
 &\le \exp\bigl(C_*u^2q(\varphi)^2/2\bigr),\\
\mathbb E|\mathcal Z_N(\varphi)|^p
 &\le 2(2C_*)^{p/2}\Gamma(1+p/2)q(\varphi)^p,
 \qquad p>0.
\end{aligned}
\tag{YM.F2}
$$

Their laws are tight in a fixed separable Hilbert space continuously
embedded in $\mathcal S'(\mathbb R^{1+2d})$. Along any weakly convergent
subsequence, every finite-order joint moment converges along that same
subsequence. The resulting hierarchy consists of tempered distributions
and has the growth bound

(eq-fg-ym-f3)=
$$
|S_n(\varphi_1,\ldots,\varphi_n)|
\le 2(2C_*)^{n/2}\Gamma(1+n/2)
                 \prod_{j=1}^n q(\varphi_j).
\tag{YM.F3}
$$

Encoding the samples in the Fractal Set leaves these random distributions
and their bounds unchanged. This uses stationarity and the one-time LSI,
without a path-space functional inequality.
:::

:::{prf:proof}
**Time integration.** Let $Y_t$ be the centered integrand in {ref}`(YM.F1) <eq-fg-ym-f1>` and
$\ell(t)=\|\nabla_z\varphi(t,\cdot)\|_\infty$. Stationarity and
{prf:ref}`cor-ym-empirical-fluctuations` give
$\mathbb E e^{aY_t}\le e^{C_*a^2\ell(t)^2/2}$.
At $\ell(t)=0$, the phase-space function is constant and $Y_t=0$.
When $Q=\int\ell(t)dt>0$, Jensen's inequality with probability weight
$\ell(t)dt/Q$ gives

$$
\begin{aligned}
\mathbb E\exp\left(u\int Y_tdt\right)
&\le\int\frac{\ell(t)}Q
 \mathbb E\exp\left(\frac{uQ}{\ell(t)}Y_t\right)dt\\
&\le e^{C_*u^2Q^2/2}.
\end{aligned}
$$

Restriction to bounded time intervals and then $L^p$ convergence justify
the integrals; Minkowski bounds their tails by a constant times
$\int_{|t|>T}\ell(t)dt$. The case $Q=0$ gives zero. The same tail
integration used in {prf:ref}`lem-ym-lsi-moments` proves {ref}`(YM.F2) <eq-fg-ym-f2>`.

**One topology for all tests.** Put $m=1+2d$ and
$A=1-\Delta_{t,z}+t^2+|z|^2$. Let $e_k$ be its tensor Hermite basis,
with eigenvalues $\lambda_k=1+m+2|k|$ for $k\in\mathbb N^m$.
For the integer $r=m+3$, there is a dimension-dependent constant $B_m$ with

(eq-fg-ym-f4)=
$$
q(\varphi)\le B_m\|A^r\varphi\|_2.
\tag{YM.F4}
$$

Here is an explicit route to this seminorm estimate. Cauchy--Schwarz in
time, followed by the Sobolev bound in the $2d$ spatial phase variables,
gives, for any integer $k_0>d$,

$$
q(\varphi)^2
\le\pi\int(1+t^2)\|\nabla_z\varphi(t,\cdot)\|_\infty^2dt
\le B\sum_{|\beta|\le k_0+1}
 \bigl(\|\partial_z^\beta\varphi\|_2^2
       +\|t\partial_z^\beta\varphi\|_2^2\bigr).
$$

Choose $k_0=m$. Multiplication by a coordinate and differentiation are
sums of Hermite raising and lowering operators. A product of $j$ such
operators changes a Hermite multi-index by at most $j$ and has coefficients
bounded by a constant times $(1+|k|)^{j/2}$. Summing squares therefore
bounds its $L^2$ norm by a constant times $\|A^r\varphi\|_2$ whenever
$j\le2r$. Every term just displayed has $j\le m+2$, proving {ref}`(YM.F4) <eq-fg-ym-f4>`.

Define $\mathscr H_{-s}$ by
$\|u\|_{-s}^2=\sum_k\lambda_k^{-2s}|u(e_k)|^2$.
From {ref}`(YM.F2) <eq-fg-ym-f2>`, or the variance bound obtained by differentiating its
moment-generating function at zero,

(eq-fg-ym-f5)=
$$
\sup_N\mathbb E\|\mathcal Z_N\|_{-s_0}^2
\le C_*B_m^2\sum_k\lambda_k^{-2(s_0-r)}<\infty
\quad\text{for }s_0>r+m/2.
\tag{YM.F5}
$$

The sum converges because the number of multi-indices with $|k|=j$ is
$\binom{j+m-1}{m-1}$. For $s_1>s_0$, the inclusion
$\mathscr H_{-s_0}\hookrightarrow\mathscr H_{-s_1}$ is compact: its
coordinate singular values are $\lambda_k^{-(s_1-s_0)}\to0$, and finite
Hermite projections uniformly approximate each bounded ball. Markov's
inequality applied to {ref}`(YM.F5) <eq-fg-ym-f5>` proves tightness in $\mathscr H_{-s_1}$.
This Hilbert space is separable and complete and embeds continuously in
$\mathcal S'$, since Schwartz Hermite coefficients decay faster than
every power. For each $N$, the originally defined distribution is the
same coefficient expansion: the pairing is bounded by
$2\sqrt N\int\|\varphi(t,\cdot)\|_\infty dt$, and Hermite expansions
converge in that Schwartz seminorm. The argument uses the whole unbounded
phase space.

**A common hierarchy.** Weak convergence in $\mathscr H_{-s_1}$ implies
joint convergence of any finite collection of Schwartz pairings. For a
product of $n$ pairings, Hölder's inequality and {ref}`(YM.F2) <eq-fg-ym-f2>` at order $2n$
give a uniform second moment of that product. It is uniformly integrable,
so its expectation converges. Hölder at order $n$ gives {ref}`(YM.F3) <eq-fg-ym-f3>`.
The Hermite coefficients of this multilinear functional have polynomial
growth by {ref}`(YM.F4) <eq-fg-ym-f4>`; pairing them with the rapidly decreasing Hermite
coefficients of an arbitrary Schwartz function of $n$ variables defines
its continuous distributional extension. Thus one subsequential law gives
all orders on one test space. Finally, the exact trajectory reconstruction
preserves every summand of {ref}`(YM.F1) <eq-fg-ym-f1>`.
:::

:::{div} feynman-prose
To find the noise in an empirical field, follow one complete update and add
every change it makes. If a cloning event writes several particle records,
those changes belong to the same random event. Multiplying the total
increments therefore produces cross terms between particles. Centering a
jitter distribution removes that jitter's mean; the replacement drift and
the shared-event covariance still have to be calculated.

The following identities do that bookkeeping for both continuous and discrete
time. The physics application selects disjoint mutual pairs; the general
companion interface also permits overlapping groups. Even within a single
pair, the two changes enter the same increment. For total momentum their
cross terms cancel the individual cloning contributions exactly. For paired
Euclidean stationary laws already covered by the established joint LSI, the
corollary below combines that cancellation with the kinetic noise to prove
a positive variance bound uniform in population size. Pairing fixes the
collision identity; the LSI belongs to the stationary law identified in that
corollary.
:::

:::{prf:proposition} Drift and covariance equations for the complete selected update
:label: prop-ym-complete-fluctuation-equations

Write $F_\varphi(s)=N^{-1}\sum_i\varphi(z_i)$ and
$Z_\varphi=\sqrt N(F_\varphi-\pi_NF_\varphi)$ for a real test in the
applicable generator domain. For a full transition event $s\mapsto s'$,
define the total increment

(eq-fg-ym-f6)=
$$
\Delta_\varphi(s,s')=\sum_i[\varphi(z_i')-\varphi(z_i)].
\tag{YM.F6}
$$

All changed companions and offspring enter this sum, with the implemented
collision assignment in {prf:ref}`prop-sm-implemented-collision-increments`.
For the conservative continuous realization in
{prf:ref}`def-kl-full-generator`, Dynkin's martingale for $Z_\varphi$
has predictable cross-variation density

(eq-fg-ym-f7)=
$$
\mathcal B_N(\varphi,\psi)(s)
=\frac2N\sum_{i,j}\nabla\varphi(z_i)^{\mathsf T}
 a_{ij}(s)\nabla\psi(z_j)
 +\frac1N\int\Delta_\varphi\Delta_\psi\,r_N(s,ds').
\tag{YM.F7}
$$

For its stationary law the covariance identities are

(eq-fg-ym-f8)=
$$
\begin{aligned}
0&=\pi_N\bigl(Z_\varphi LZ_\psi+Z_\psi LZ_\varphi
                       +\mathcal B_N(\varphi,\psi)\bigr),\\
\frac d{dt}\mathbb E_\pi[Z_\varphi(S_0)Z_\psi(S_t)]
 &=\mathbb E_\pi[Z_\varphi(S_0)LZ_\psi(S_t)].
\end{aligned}
\tag{YM.F8}
$$

For the actual discrete step, put
$d_\varphi=\sqrt N(P_hF_\varphi-F_\varphi)$ and

(eq-fg-ym-f9)=
$$
\mathcal Q_N(\varphi,\psi)
=\frac1N\left[\mathbb E(\Delta_\varphi\Delta_\psi\mid s)
 -\mathbb E(\Delta_\varphi\mid s)
  \mathbb E(\Delta_\psi\mid s)\right].
\tag{YM.F9}
$$

Its stationary identity, including its finite-step term, is

(eq-fg-ym-f10)=
$$
0=\pi_N\bigl(Z_\varphi d_\psi+Z_\psi d_\varphi
                  +d_\varphi d_\psi+\mathcal Q_N(\varphi,\psi)\bigr).
\tag{YM.F10}
$$

In particular,
$\operatorname{Var}_\pi Z_\varphi\ge\pi_N\mathcal Q_N(\varphi,\varphi)$.
For the continuous model, with $K_N=\pi_N|LZ_\varphi|^2<\infty$,

(eq-fg-ym-f11)=
$$
\operatorname{Var}_\pi Z_\varphi
\ge\frac{[\pi_N\mathcal B_N(\varphi,\varphi)]^2}{4K_N}
\quad(K_N>0).
\tag{YM.F11}
$$
:::

:::{prf:proof}
For the diffusion part the product rule gives
$L(fg)-fLg-gLf=2\nabla f^{\mathsf T}a_N\nabla g$.
For a jump it gives
$\int[f(s')-f(s)][g(s')-g(s)]r_N(s,ds')$.
Since $\nabla_iZ_\varphi=N^{-1/2}\nabla\varphi(z_i)$ and
$Z_\varphi(s')-Z_\varphi(s)=N^{-1/2}\Delta_\varphi$, this proves
{ref}`(YM.F7) <eq-fg-ym-f7>`. Localization in the existing confining envelope, followed by its
integrable moment bounds, justifies Dynkin's formula on the indicated
domain. Stationarity applied to $L(Z_\varphi Z_\psi)$ proves the first
identity in {ref}`(YM.F8) <eq-fg-ym-f8>`; differentiation of $P_tZ_\psi$ proves the second.

For clarity, the drift before centering is the explicit expression

(eq-fg-ym-f12)=
$$
LF_\varphi
=\frac1N\sum_i\left[b_i\cdot\nabla\varphi(z_i)
       +\operatorname{tr}(a_{ii}\nabla^2\varphi(z_i))\right]
 +\frac1N\int\Delta_\varphi(s,s')r_N(s,ds').
\tag{YM.F12}
$$

The jump part of {ref}`(YM.F7) <eq-fg-ym-f7>` contains
$\sum_{i,j}\delta_i\varphi\,\delta_j\psi$, including $i\ne j$.
These cross terms account for common companions and the group writes in
{ref}`(SM.K8) <eq-fg-sm-k8>`. Centered position jitter removes its own first moment in {ref}`(SM.K7) <eq-fg-sm-k7>`;
it removes neither the replacement drift in {ref}`(YM.F12) <eq-fg-ym-f12>` nor these cross terms.

In discrete time decompose

$$
Z_\varphi(S_{k+1})-Z_\varphi(S_k)
=d_\varphi(S_k)+\xi_{k+1}^\varphi,\qquad
\mathbb E[\xi_{k+1}^\varphi\mid S_k]=0.
$$

Expanding the conditional covariance gives {ref}`(YM.F9) <eq-fg-ym-f9>`. Expand the product of
the two new $Z$ values, average conditionally, and then use stationarity to
obtain {ref}`(YM.F10) <eq-fg-ym-f10>`. Conditional variance decomposition gives

$$
\operatorname{Var}_\pi Z_\varphi
=\pi_N\mathcal Q_N(\varphi,\varphi)
 +\operatorname{Var}_\pi(P_hZ_\varphi),
$$

which proves the discrete lower bound. Finally {ref}`(YM.F8) <eq-fg-ym-f8>` with
$\psi=\varphi$ and Cauchy--Schwarz give
$\pi_N\mathcal B_N\le2\sqrt{\operatorname{Var}Z_\varphi}\sqrt{K_N}$,
proving {ref}`(YM.F11) <eq-fg-ym-f11>`. For the Euclidean velocity observable
$\varphi(x,v)=e\cdot v$, $|e|=1$, the diffusion contribution alone is
$2D$ in {ref}`(YM.F7) <eq-fg-ym-f7>`. Thus {ref}`(YM.F11) <eq-fg-ym-f11>` reads $\operatorname{Var}Z_\varphi
\ge D^2/K_N$ where the established moment domain covers this observable.
An $N$-uniform lower bound retains the actual $N$ dependence of $K_N$.
:::

:::{div} feynman-prose
A collision can preserve the sum of two velocities while reducing the sum
of their squared lengths. Think of two particles with opposite velocities:
reducing both speeds preserves their zero total momentum and removes kinetic
energy. The paired collision formula computes that loss for every accepted
pair.

Now test the product reference against the complete generator. Its kinetic
heating, friction, and force terms balance already. Accepted inelastic
collisions with unequal velocities add a strictly negative mean contribution
to the squared-velocity observable. The resulting nonzero residual proves
that this reference is not stationary for that combined process.

The reference still has its proved LSI and its constructed equilibrium
theory. To apply an LSI to the selected stationary process, use a route that
identifies that very joint law. The following remark specifies the existing
density, curvature, and flow criteria; conservation of momentum alone does
not verify them.
:::

:::{prf:proposition} Stationarity residual of the paired cloning law
:label: prop-ym-paired-reference-residual

For the Euclidean conservative continuous generator
{prf:ref}`def-kl-full-generator` with the mutual-pair collisions
of {prf:ref}`cor-sm-physics-paired-cloning`, write
$L=L_{\mathrm{kin}}+J$ and $V_2(s)=\sum_i|v_i|^2$.
Define its nonnegative collision loss rate by

(eq-fg-ym-l1)=
$$
\mathfrak d_N(s)
=\int [V_2(s)-V_2(s')]\,r_N(s,ds')\ge0.
\tag{YM.L1}
$$

The exact generator identity and its stationary consequence are

(eq-fg-ym-l2)=
$$
\begin{aligned}
LV_2
 &=2DdN-2\gamma V_2-2\sum_i v_i\cdot\nabla U(x_i)
   -\mathfrak d_N,\\
2DdN
 &=2\gamma\pi_N V_2
   +2\pi_N\sum_i v_i\cdot\nabla U(x_i)
   +\pi_N\mathfrak d_N .
\end{aligned}
\tag{YM.L2}
$$

The second line uses the actual stationary law and the existing
integrable generator domain. On the upstream product reference
$\rho_N^0=m_U^{\otimes N}$,

(eq-fg-ym-l3)=
$$
\rho_N^0(LV_2)=-\rho_N^0\mathfrak d_N.
\tag{YM.L3}
$$

In particular, whenever the implemented gate accepts an inelastic pair
with distinct velocities on a set of positive reference rate,
$\rho_N^0\mathfrak d_N>0$ and this product reference is not
stationary for the combined generator. The reference LSI continues
to apply to that reference, and the equilibrium construction on it
remains the product theory already proved.
:::

:::{prf:proof}
The pairwise identity {ref}`(SM.K10) <eq-fg-sm-k10>` gives the
loss in each complete jump:

$$
V_2(s)-V_2(s')
=(1-\alpha^2)
 \sum_{\{i,j\}\ {\rm accepted}}\frac{|v_i-v_j|^2}{2}.
$$

Its integral proves nonnegativity of
{ref}`(YM.L1) <eq-fg-ym-l1>` and $JV_2=-\mathfrak d_N$.
Gaussian position jitter contributes zero to this observable.
For the kinetic part, $\nabla_{v_i}|v_i|^2=2v_i$ and
$\Delta_{v_i}|v_i|^2=2d$ give the first line of
{ref}`(YM.L2) <eq-fg-ym-l2>`. Integrating the generator in its
stationary law proves the second line. Localization followed by
the established confining moment bounds justifies the unbounded
observable in its stated generator domain.

Under $m_U^{\otimes N}$ the velocities are independent centered
Gaussians of covariance $\theta I_d$, independent of the positions.
Thus $\rho_N^0 V_2=Nd\theta$,
$\rho_N^0\sum_i v_i\cdot\nabla U(x_i)=0$, and
$D=\gamma\theta$. The three kinetic terms cancel, proving
{ref}`(YM.L3) <eq-fg-ym-l3>`.
For $\alpha<1$, any accepted pair with unequal velocities
contributes strictly positive loss. Integration over a set of
positive reference rate makes the residual strictly negative.
A stationary law would have zero residual on this observable,
which proves the asserted distinction.
:::

:::{prf:remark} Applying the established LSI to the same selected law
:label: rem-ym-same-law-lsi-application

The four routes in {prf:ref}`cor-n-uniform-lsi` identify their
respective product, bounded whole-joint tilt, joint-curvature, and
contractive additive-noise laws. The preceding residual tests
stationarity of the first route against the actual paired jump.
The additive-noise theorem concerns a diffusion flow; its statement
does not contain the selected jump kernel.

For a selected invariant law, a use of the bounded-tilt route must
identify its actual density ratio and the uniform oscillation of
its logarithm. A curvature route uses the Hessian of that law's
negative log density. These are the existing upstream entry
conditions, distinct from bounded derivatives of the algorithmic
fitness and from velocity ellipticity. Exact reconstruction carries
each verified inequality on that same law to the field coordinates.

Accordingly the momentum bound
{ref}`(YM.F13) <eq-fg-ym-f13>` applies to paired Euclidean
stationary laws already covered by the joint LSI. The nontrivial
equilibrium hierarchy
{ref}`(YM.E9) <eq-fg-ym-e9>`--{ref}`(YM.E14) <eq-fg-ym-e14>`
uses its stated product or bounded whole-joint tilt law. The
collision residual supplies an explicit applicability calculation
and prevents replacing either law by the other without identification.
:::


:::{div} feynman-prose
The convergence proof already bounds one complete block of recorded updates
between two multiples of a reference probability. Start that block in the
selected QSD. Its surviving endpoints have the same QSD, with total mass
given by the block survival probability. Dividing the block bounds by that
mass gives density bounds for the selected law itself, including the effects
of every selection and kinetic step in the block.

Those bounds also give a direct entropy calculation. The upper density bound
controls entropy; the lower bound controls gradient energy. Combining them
multiplies the reference LSI constant by the ratio of the block constants.
The common survival factor cancels. This supplies the explicit comparison
for the selected QSD using the convergence theorem's own reference and
constants, with their population dependence retained.
:::

:::{prf:proposition} Joint density of the selected QSD from the proved block kernel
:label: prop-ym-selected-qsd-block-density

Use the complete killed finite-swarm kernel $Q_N$ in
{prf:ref}`def-cemetery-state`, in the block-kernel regime of
{prf:ref}`thm-main-convergence`. Retain exactly that theorem's block length
$m_N$, probability $\eta_N$, constants $c_N,C_N$, selected QSD $\nu_N$,
and survival eigenvalue $\alpha_N$. The QSD has a density $r_N$ with
respect to the block reference, and

(eq-fg-ym-j1)=
$$
\frac{c_N}{\alpha_N^{m_N}}
\le r_N=\frac{d\nu_N}{d\eta_N}
\le\frac{C_N}{\alpha_N^{m_N}},\qquad
\operatorname*{ess\,osc}_{\eta_N}\log r_N
\le\log\frac{C_N}{c_N}.
\tag{YM.J1}
$$

For continuous coordinates, define $\mathsf C(\mu)$ to be the optimal
full-gradient LSI constant of a probability $\mu$, allowing $+\infty$.
The exact same-law comparison is

(eq-fg-ym-j2)=
$$
\mathsf C(\nu_N)\le\frac{C_N}{c_N}\mathsf C(\eta_N).
\tag{YM.J2}
$$

Thus the block reference in the convergence proof supplies the density
comparison for the actual selected QSD. Its reference constant and block
ratio retain their actual $N$ dependence. This statement does not identify
$\eta_N$ with the product kinetic reference. For discrete status variables,
the entropy comparison below remains valid, but the energy must include
the status contribution specified in {prf:ref}`cor-n-uniform-lsi`.
:::

:::{prf:proof}
**1. Integrate the complete kernel.** The QSD identity gives
$\nu_NQ_N^{m_N}=\alpha_N^{m_N}\nu_N$. Integrating both bounds in
{prf:ref}`thm-main-convergence` against $\nu_N$ gives, for every measurable
$A$,

$$
c_N\eta_N(A)\le\alpha_N^{m_N}\nu_N(A)
\le C_N\eta_N(A).
$$

The Radon--Nikodym theorem gives both bounds in
{ref}`(YM.J1) <eq-fg-ym-j1>`. Their ratio is $C_N/c_N$; the survival
factor cancels in the logarithmic oscillation. This integrates the full
selected block, rather than one kinetic or companion-selection substep.

**2. Compare entropy and energy explicitly.** Write
$\ell_N=c_N/\alpha_N^{m_N}$ and $u_N=C_N/\alpha_N^{m_N}$.
For $g\ge0$, the nonnegative integrand
$D_a(g)=g\log(g/a)-g+a$ satisfies

$$
\operatorname{Ent}_{\mu}(g)=\inf_{a>0}\int D_a(g)\,d\mu.
$$

Indeed, differentiating the integral in $a$ gives
$1-\mu(g)/a$, so the minimum is attained at $a=\mu(g)$;
the zero-integral case follows by $a\downarrow0$. Consequently

$$
\operatorname{Ent}_{\nu_N}(f^2)
\le u_N\operatorname{Ent}_{\eta_N}(f^2),\qquad
\int|\nabla f|^2d\eta_N
\le\ell_N^{-1}\int|\nabla f|^2d\nu_N.
$$

Inserting any finite valid reference LSI constant $K$ between these two
inequalities gives

$$
\operatorname{Ent}_{\nu_N}(f^2)
\le 2\frac{u_N}{\ell_N}K\int|\nabla f|^2d\nu_N.
$$

Take the infimum over valid $K$ to obtain
{ref}`(YM.J2) <eq-fg-ym-j2>`. If none is finite, the extended inequality
does not assert a finite constant. The proof applies first to the common
smooth core; truncation and lower semicontinuity give the relaxed energy
used in {prf:ref}`thm-ym-equilibrium-form-construction`.

**3. Identify the population dependence.** The quantity supplied by this
calculation is precisely $(C_N/c_N)\mathsf C(\eta_N)$.
The upstream block theorem proves finite positive $c_N,C_N$ at fixed
$N$ and explicitly permits their dependence on $N$. Neither that statement
nor {ref}`(YM.J1) <eq-fg-ym-j1>` bounds the supremum of this quantity.
Accordingly this calculation transfers every established reference bound
to the selected QSD without substituting a different stationary law or
claiming an unproved population-uniform constant.
:::

:::{div} feynman-prose
Start recorded episodes in the selected QSD and retain precisely those that
survive through step $K$. At an earlier step $k$, a state is weighted by its probability of
surviving the remaining $K-k$ steps. The terminal frame has no remaining
steps to survive, so its law is the QSD. The earlier frames carry the
explicit weight calculated below.

For each transition, multiply the complete recorded kernel by the ratio
of the remaining survival probabilities. Multiplying these transitions
along an episode cancels every intermediate survival factor and recovers
the complete likelihood conditioned on survival. Averaging that likelihood
over histories with the same descriptor then gives the selected descriptor
law. The calculation therefore applies directly to recorded channel
correlations, with their companion choices, gates, jitter, and collisions.
:::

:::{prf:proposition} Exact selected histories under finite-horizon survival
:label: prop-ym-qsd-history-identification

For the same complete kernel and its QSD, put $h_j=Q_N^j1$ and let
$K\ge1$ be an integer survival horizon. For $0\le k\le K$, the actual
interior marginal conditioned on survival through $K$ is

(eq-fg-ym-j3)=
$$
\mathbb P_{\nu_N}(S_k\in ds\mid T_\dagger>K)
=\frac{h_{K-k}(s)}{\alpha_N^{K-k}}\nu_N(ds).
\tag{YM.J3}
$$

For $0\le k<K$, its transition from time $k$ to time $k+1$ is

(eq-fg-ym-j4)=
$$
R_{k,K}(s,ds')
=\frac{Q_N(s,ds')h_{K-k-1}(s')}{h_{K-k}(s)}.
\tag{YM.J4}
$$

These kernels and the initial law
$h_K\nu_N/\alpha_N^K$ reconstruct every bounded finite-history
correlation under that conditioning. In particular the terminal marginal
is $\nu_N$, while earlier marginals carry the displayed survival weight.
The complete QSD history density with respect to any recorded reference
history law is obtained by multiplying its actual likelihood by
$\mathbf1_{\{T_\dagger>K\}}/\alpha_N^K$ before applying the descriptor
conditional expectation in {prf:ref}`prop-ym-density-and-support`.
:::

:::{prf:proof}
**1. Compute the marginal.** The Markov property and the QSD eigenmeasure
give

$$
\begin{aligned}
\mathbb P_{\nu_N}(S_k\in A,T_\dagger>K)
&=\int_A(\nu_NQ_N^k)(ds)\,Q_N^{K-k}1(s)\\
&=\alpha_N^k\int_Ah_{K-k}(s)\nu_N(ds).
\end{aligned}
$$

Divide by $\mathbb P_{\nu_N}(T_\dagger>K)=\alpha_N^K$ to obtain
{ref}`(YM.J3) <eq-fg-ym-j3>`. The denominator in
{ref}`(YM.J4) <eq-fg-ym-j4>` is positive in the upstream block regime.
Conditional probability gives its numerator, and
$Q_Nh_{K-k-1}=h_{K-k}$ proves that the kernel is conservative.

**2. Verify every correlation by cancellation.** For bounded $F$ of the
whole history, integrate $F$ against

$$
\frac{h_K(s_0)}{\alpha_N^K}\nu_N(ds_0)
\prod_{k=0}^{K-1}
\frac{Q_N(s_k,ds_{k+1})h_{K-k-1}(s_{k+1})}
     {h_{K-k}(s_k)}.
$$

Every intermediate $h$ cancels; $h_0=1$ leaves

(eq-fg-ym-j5)=
$$
\mathbb E_{\nu_N}[F\mid T_\dagger>K]
=\alpha_N^{-K}\int F(s_0,\ldots,s_K)
\nu_N(ds_0)\prod_{k=0}^{K-1}Q_N(s_k,ds_{k+1}).
\tag{YM.J5}
$$

This proves the whole-history identity, including all companion choices,
gates, jitter, and collision effects already present in $Q_N$.
If $P=\mathcal L R$ denotes this unconditioned recorded path law,
then its conditioned likelihood is
$\mathcal L\mathbf1_{\{T_\dagger>K\}}/\alpha_N^K$.
For its descriptor $\mathscr D$, conditional expectation under $R$ gives
the exact descriptor density

$$
a_K(y)=\frac1{\alpha_N^K}
\mathbb E_R[\mathcal L\mathbf1_{\{T_\dagger>K\}}
                 \mid\mathscr D=y].
$$

Integrating $f(y)a_K(y)$ against $\mathscr D_\#R$ reproduces
{ref}`(YM.J5) <eq-fg-ym-j5>` for $F=f\circ\mathscr D$.

**3. Distinguish the equilibrium transfer on the same static law.**
Equation {ref}`(YM.J3) <eq-fg-ym-j3>` equals $\nu_N$ at time $k$
exactly when $h_{K-k}=\alpha_N^{K-k}$ $\nu_N$-almost everywhere.
The QSD identity fixes its integral, not this pointwise equality.
Thus the static law used in the gradient-energy construction may be the
selected QSD, while its conservative reversible equilibrium history
and the survival-conditioned algorithm history have the respective
transition formulas already given. Equality of their static endpoint
law does not remove the explicit interior weight in
{ref}`(YM.J3) <eq-fg-ym-j3>`.
:::


:::{prf:theorem} Relaxation to the selected law of a recorded gauge window
:label: thm-ym-qsd-window-relaxation

Use the complete killed kernel $Q_N$, its selected QSD $\nu_N$, and
survival eigenvalue $\alpha_N>0$ from
{prf:ref}`prop-ym-qsd-history-identification`. Fix an integer window
length $w\ge0$. Let $\mathcal K_{N,w}(s,d\omega)$ be the
subprobability kernel of the complete recorded $w$-step window starting
at $s$, including its initial and terminal states, restricted to survival
through the window. Thus $\mathcal K_{N,w}1=Q_N^w1=h_w$.
This kernel is the recorded update already instantiated in
{prf:ref}`thm-sm-instantiated-record-transition`, with each companion
draw, gate, jitter, collision and kinetic stage retained.

Let $D_w(\omega)$ be any fixed measurable gauge-window descriptor
from {prf:ref}`thm-sm-effective-recorded-gauge-dynamics`, with times
labelled relative to the beginning of that window. For an initial law
$\mu$, write $\mu_n=\mu Q_N^n/(\mu Q_N^n1)$ and
$\delta_n=\|\mu_n-\nu_N\|_{\rm TV}$. Whenever the displayed
survival conditioning has positive probability, the descriptor after
$n$ relaxation steps and $w$ recording steps has the exact law

(eq-fg-ym-z49)=
$$
\begin{gathered}
\Lambda_{n,w}^{\mu}
 =(D_w)_*\frac{\mu_n\mathcal K_{N,w}}{\mu_nh_w},\qquad
\Lambda_{w}^{\nu}
 =(D_w)_*\frac{\nu_N\mathcal K_{N,w}}{\alpha_N^w},\\
\|\Lambda_{n,w}^{\mu}-\Lambda_w^{\nu}\|_{\rm TV}
 \le\min\{1,2\alpha_N^{-w}\delta_n\}.
\end{gathered}
\tag{YM.Z49}
$$

The first law is conditioned on survival through $n+w$ in the actual
run. The second is precisely the selected descriptor law in
{ref}`(YM.J5) <eq-fg-ym-j5>`. In particular, for any finite list
of bounded real or complex recorded gauge observables $F_1,\ldots,F_q$,

(eq-fg-ym-z50)=
$$
\left|\mathbb E_{\Lambda_{n,w}^{\mu}}\prod_{j=1}^qF_j
       -\mathbb E_{\Lambda_w^{\nu}}\prod_{j=1}^qF_j\right|
\le 4\alpha_N^{-w}\delta_n\prod_{j=1}^q\|F_j\|_\infty.
\tag{YM.Z50}
$$

In the complete-kernel regime of {prf:ref}`thm-main-convergence`,
reuse its block size $m$, constants $c_N,C_N$, and
$\rho_N=1-c_N/C_N$. Then $\delta_n\le\rho_N^{\lfloor n/m\rfloor}$.
For $0<\rho_N<1$ and $0<\varepsilon<1$, the explicit choice

(eq-fg-ym-z51)=
$$
n\ge m\left\lceil
 \frac{\log(2/\varepsilon)+w\log(1/\alpha_N)}{-\log\rho_N}
 \right\rceil
\quad\Longrightarrow\quad
\|\Lambda_{n,w}^{\mu}-\Lambda_w^{\nu}\|_{\rm TV}\le\varepsilon.
\tag{YM.Z51}
$$

For $\rho_N=0$, $n\ge m$ gives equality of the two laws. The already
proved lower bound $\alpha_N\ge c_N^{1/m}$ can be substituted in
the recording cost. No new mixing or survival estimate is introduced.

Starting in $\nu_N$ makes the window law independent of the number
of preceding relaxation steps, after the stated terminal survival
conditioning. Moreover, deleting the first $k$ recorded transitions
from a selected $w$-step window gives exactly the selected
$(w-k)$-step window law, for $0\le k\le w$.

For an indexed family of these same selected descriptors, choose
$n_j$ so that $2\alpha_j^{-w_j}\delta_{n_j}\le\varepsilon_j\to0$.
Any tightness or weak-limit result for $\Lambda_{w_j}^{\nu_j}$ then
passes to the actual relaxed-run laws $\Lambda_{n_j,w_j}^{\mu_j}$
on the same Polish descriptor space. All bounded gauge correlation limits
pass with the error in {ref}`(YM.Z50) <eq-fg-ym-z50>`.
Here $n$ counts algorithmic relaxation steps; no identification of
that counter with a physical spacetime coordinate enters the result.
:::

:::{prf:proof}
**Keep the recorded kernel and survival normalization together.**
The Markov property at step $n$, before conditioning on the future
window, gives for a descriptor event $A$

$$
\mathbb P_\mu(D_w\in A,T_\dagger>n+w)
=\mu Q_N^n\mathcal K_{N,w}(D_w^{-1}A).
$$

Divide by $\mu Q_N^{n+w}1$, and then cancel $\mu Q_N^n1$.
This proves the first formula in {ref}`(YM.Z49) <eq-fg-ym-z49>`.
The QSD identity gives $\nu_Nh_w=\alpha_N^w$ and proves the
second formula using the same kernel. Intermediate recorded marks
are integrated only after evaluation of $D_w$.

**Normalization error in total variation.** Put
$u=\mu_n\mathcal K_{N,w}$, $v=\nu_N\mathcal K_{N,w}$,
$a=u1>0$, and $b=v1=\alpha_N^w>0$.
For any window event $A$, the function
$s\mapsto\mathcal K_{N,w}(s,A)$ takes values in $[0,1]$.
The layer-cake formula and the definition
$\|\mu_n-\nu_N\|_{\rm TV}=\sup_B|\mu_n(B)-\nu_N(B)|$
give $|u(A)-v(A)|\le\delta_n$ and $|a-b|\le\delta_n$.
Now compute

$$
\frac{u(A)}a-\frac{v(A)}b
=\frac{u(A)-v(A)}b+\frac{u(A)(b-a)}{ab}.
$$

Since $u(A)\le a$, its absolute value is at most $2\delta_n/b$.
Taking the supremum over events and then pushing forward by $D_w$
proves the TV bound. For any bounded complex $G$, integration
against the total variation measure of the signed difference of
two probabilities gives an error at most $2\|G\|_\infty$ times
their TV distance. Substitution of $G=\prod_jF_j$ proves
{ref}`(YM.Z50) <eq-fg-ym-z50>` with its stated factor four.

**Reuse the established relaxation rate.** Insert the existing bound
$\delta_n\le\rho_N^{\lfloor n/m\rfloor}$ and solve
$2\alpha_N^{-w}\rho_N^{\lfloor n/m\rfloor}\le\varepsilon$.
Taking logarithms gives {ref}`(YM.Z51) <eq-fg-ym-z51>`.
When $\rho_N=0$, the upstream bound vanishes after one block.
When the initial law is $\nu_N$, $\mu_n=\nu_N$ for every $n$,
so the window equality holds without a relaxation error.

**Consistency after deleting the beginning of a window.** Integrate
the first $k$ transitions, including their discarded marks, in
$\alpha_N^{-w}\nu_N\mathcal K_{N,w}$. Their terminal measure is
$\nu_NQ_N^k=\alpha_N^k\nu_N$. The remaining window therefore
has law

$$
\alpha_N^{-w}\nu_NQ_N^k\mathcal K_{N,w-k}
=\alpha_N^{-(w-k)}\nu_N\mathcal K_{N,w-k}.
$$

This is the claimed selected shorter window with its relative labels.
The terminal conditioning and the direction of this marginalization
are specified by the calculation; the interior survival weights remain
those of {ref}`(YM.J3) <eq-fg-ym-j3>`.

**Pass the same descriptor limits to relaxed runs.** For a bounded
continuous test $G$, the difference of expectations under the two
indexed laws is at most $2\varepsilon_j\|G\|_\infty$.
Hence convergence of the selected-law expectation implies the same
limit for the relaxed-run expectation. For tightness, a compact set
with selected probability at least $1-\eta$ has relaxed probability
at least $1-\eta-\varepsilon_j$; the finitely many remaining indices
are handled individually. Products of bounded gauge observables use
the already proved explicit bound. Thus the selected field law and
the algorithm's relaxation limit are identified on precisely the
same recorded observables.
:::


:::{div} feynman-prose
For the next calculation, fix a paired Euclidean stationary law already
covered by the established joint LSI family. Choose an observable whose
cloning increment we can compute completely: the sum of velocities in a
fixed direction. Each mutual-pair collision preserves that sum. Its
fluctuations therefore retain the kinetic force and friction drift, and
a strictly positive contribution from velocity diffusion.

The bound on the potential's Hessian controls the gradient of this drift.
Together with the LSI for this same stationary law, it gives a drift estimate
independent of population size. Stationary balance then puts a positive lower
bound on the momentum variance. A short time average retains that variance,
proving a nonzero subsequential fluctuation limit in the model's kinetic time.
The collision rule supplies the exact cancellation; membership in the
stationary joint-LSI family supplies the analytic bound. Gaussian
identification and reflection positivity retain their separate requirements
for this law.
:::

:::{prf:corollary} Nonzero momentum fluctuations in the established Euclidean paired model
:label: cor-ym-paired-momentum-fluctuations

Use a conservative Euclidean paired stationary law already covered by
{prf:ref}`cor-n-uniform-lsi`, with the generator of
{prf:ref}`def-kl-full-generator` and the collision rule of
{prf:ref}`cor-sm-physics-paired-cloning`. The same-law applicability
calculation is {prf:ref}`prop-ym-paired-reference-residual`.
Let $M$ be the global Hessian bound in the selected kinetic regime of
{prf:ref}`lem-kinetic-evolution-bounds`, and let $D>0$ be its velocity
diffusion coefficient. For $|e|=1$ put

$$
Y_N=\frac1{\sqrt N}\sum_i
 [e\cdot v_i-\pi_N(e\cdot v_i)].
$$

Then the kinetic and cloning contributions give the explicit uniform bounds

(eq-fg-ym-f13)=
$$
\begin{aligned}
\mathcal B_N(e\cdot v,e\cdot v)&=2D,\qquad
\pi_N|LY_N|^2\le C_*(M^2+\gamma^2),\\
\frac{D^2}{C_*(M^2+\gamma^2)}
&\le\operatorname{Var}_{\pi_N}Y_N\le C_*.
\end{aligned}
\tag{YM.F13}
$$

The stationary time-distribution $t\mapsto Y_N(S_t)$ consequently has
a nonzero subsequential limit, jointly with the fields in
{prf:ref}`thm-ym-spacetime-fluctuation-compactness`. The constants in
{ref}`(YM.F13) <eq-fg-ym-f13>` are those of the specified Euclidean realization; optional
Boris rotations, unequal directed viscous weights, and fixed-step
integration retain their own complete-update equation {ref}`(YM.F10) <eq-fg-ym-f10>`.
:::

:::{prf:proof}
The total velocity is unchanged by each paired cloning event, so its jump
increment in {ref}`(YM.F6) <eq-fg-ym-f6>` is zero. The drift and bracket therefore reduce to

$$
LY_N=-\frac1{\sqrt N}\sum_i e\cdot\nabla U(x_i)
      -\frac\gamma{\sqrt N}\sum_i e\cdot v_i,
\qquad \mathcal B_N=2D.
$$

The derivative in the $i$th position block of the drift is
$-N^{-1/2}\nabla^2U(x_i)e$, and its velocity derivative is
$-N^{-1/2}\gamma e$. Thus its total squared gradient is at most
$M^2+\gamma^2$. Its stationary mean is zero. The inherited Poincaré
inequality proves the bound on $\pi_N|LY_N|^2$, while the same
inequality applied to $Y_N$ gives its variance upper bound. Formula
{ref}`(YM.F11) <eq-fg-ym-f11>` gives the lower bound. The unbounded linear velocity observable
and its drift have the required finite moments by the existing confining
moment estimates; truncation and their bounded-gradient estimates extend
the LSI calculation to these observables.

Write $c=D^2/[C_*(M^2+\gamma^2)]$ and
$b=C_*\sqrt{M^2+\gamma^2}$. The actual stationary Markov contraction
and Dynkin's formula give

$$
\left|\mathbb E[Y_N(S_0)Y_N(S_t)]-\pi_NY_N^2\right|
\le\int_0^t\|Y_N\|_2\|P_sLY_N\|_2ds\le bt.
$$

Choose a smooth nonnegative time test $a$ of integral one with support
in an interval of length at most $c/(2b)$. Stationarity and the scalar
covariance symmetry give
$\operatorname{Var}[\int a(t)Y_N(S_t)dt]\ge c/2$.
The LSI moment calculation for this time-distribution uses
$q(a)=\int|a(t)|dt$, since the phase-space Lipschitz constant of $Y_N$
is one. The Hermite argument of {ref}`(YM.F4) <eq-fg-ym-f4>`--{ref}`(YM.F5) <eq-fg-ym-f5>`, now in the time
variable, gives tightness in a fixed negative Hilbert norm. Apply joint
tightness with {ref}`(YM.F1) <eq-fg-ym-f1>` and uniform integrability to pass the positive
variance bound to a common subsequential law. This proves nontrivial
algorithmic fluctuations in this established paired Euclidean regime,
without a Gaussian or reflection-positivity assertion for its kinetic law.
:::


:::{prf:remark} Identification of the algorithmic fluctuation limit
:label: rem-ym-fluctuation-identification

The distributional compactness theorem strengthens the fixed-dimensional
statement in {prf:ref}`cor-ym-empirical-fluctuations`: it supplies a common
subsequential law and every correlation order on a specified test topology.
Equations {ref}`(YM.F7) <eq-fg-ym-f7>`--{ref}`(YM.F12) <eq-fg-ym-f12>` specify the drift, noise, and covariance that a
limit of the selected model must retain. They are computed from that model,
including the whole event rather than a donor-only replacement.

To identify a unique fluctuation evolution, the quantities to control are
the limit of $\sqrt N LF_\varphi$ (or $d_\varphi$), its finite-population
remainder, and the bracket {ref}`(YM.F7) <eq-fg-ym-f7>` (or {ref}`(YM.F9) <eq-fg-ym-f9>`). The first-coordinate fitness
derivative bounds already proved in the regularity chapters remain bounds
on those derivatives; particle-index sums, clone-gate boundaries, and
derivatives with respect to a population law are separate expressions in
{ref}`(YM.F12) <eq-fg-ym-f12>`. Their values cannot be supplied by a covariance upper bound.
For a general overlapping-group companion law, {ref}`(SM.K9) <eq-fg-sm-k9>` prevents that
pathwise exchangeability argument. The current physics application uses
the mutual-pair law of {prf:ref}`cor-sm-physics-paired-cloning`, which
excludes those events and proves the cloning factor equivariant.

The established mean-field limit identifies the intensive field:
$N^{-1}\sum_i\varphi(Z_i)\to\mu_*\varphi$. Its centered fluctuation
retains the centering in {ref}`(YM.F1) <eq-fg-ym-f1>`; replacing it by $\mu_*\varphi$ adds
$\sqrt N(\pi_NF_\varphi-\mu_*\varphi)$. Accordingly the proved
conclusion here is the common subsequential hierarchy with the exact
finite-population evolution and covariance equations. In the paired
Euclidean regime, {prf:ref}`cor-ym-paired-momentum-fluctuations` also proves
nontriviality with the explicit uniform constants {ref}`(YM.F13) <eq-fg-ym-f13>`. The LSI
alone does not select a Gaussian law or a unique fluctuation covariance.
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

For the encoded algorithmic field, {ref}`(SM.K1) <eq-fg-sm-k1>` and
{prf:ref}`thm-lqft-instantiated-word-evolution` identify its generator as
$\widehat K=\Gamma_-(U)K\Gamma_-(U)^{-1}$ with the transported full domain.
Its semigroups, spectra, and transported bounded operator correlations
therefore agree exactly with the recorded replica representation.

For comparison with an independently specified field operator, let
$\mathcal W$ be the comparison unitary. Agreement of
$K_{\mathrm{field}}$ and $\mathcal W K\mathcal W^{-1}$ on a common
core of their strongly continuous generators proves the same equality
for that additional operator.

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

For the actual encoded field, the difference quotient in {ref}`(SM.K3) <eq-fg-sm-k3>` and
its exterior version {ref}`(LQ.R3) <eq-fg-lq-r3>` give the full operator equality directly.
Their explicit conditional-expectation kernel retains each selected update.
For the additional comparison, if $\mathcal C$ is the common core,
closedness gives

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

(eq-fg-ym-38)=
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
{ref}`(YM.38) <eq-fg-ym-38>`. Reversibility gives a self-adjoint Markov generator in its invariant
$L^2$ space; positive static entropy bounds alone do not give reversibility.
:::

:::{prf:theorem} Locality of the available operator constructions
:label: thm-wightman-w3-fg

The bounded multiplication fields {ref}`(YM.35) <eq-fg-ym-35>` commute for every pair of tests.
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
fields {ref}`(YM.35) <eq-fg-ym-35>` on $\mathcal H_{\rm obs}$. Cyclicity on the full
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

(eq-fg-ym-39)=
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
For intensive empirical fields {ref}`(YM.35) <eq-fg-ym-35>`, {ref}`(YM.36) <eq-fg-ym-36>` supplies such a bound.
For smooth Lipschitz observables of a law satisfying {ref}`(YM.1) <eq-fg-ym-1>`, {ref}`(YM.37) <eq-fg-ym-37>` supplies
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

### Equilibrium transfer from the established energy

:::{div} feynman-prose
The LSI already compares an observable's fluctuations with its gradient
energy. We can use that very energy to define a time evolution. First close
the energy: admit limits of smooth observables while keeping track of the
least energy needed to approximate them. The resulting closed quadratic
form determines a self-adjoint operator and its equilibrium transfer.

This construction keeps the specified one-time law. The LSI becomes a
spectral bound for the constructed operator, so centered equilibrium
observables decay at a controlled rate. The normalization of the gradient
energy fixes this new clock. An original kinetic trajectory also transports
position through velocity, and its generator must still be used for recorded
time correlations. Even when the chosen law is a QSD, the energy construction
has its own conservative equilibrium evolution.
:::

:::{prf:theorem} Equilibrium transfer operator of the algorithmic LSI law
:label: thm-ym-equilibrium-form-construction

Let $\rho_N$ denote the particular continuous joint law already covered by
{prf:ref}`cor-n-uniform-lsi`, with its proved constant $C_*$. This may be
the identified invariant law, QSD, or reference law in that corollary; its
identity is retained. On real smooth bounded functions with bounded
derivatives, use precisely its full-gradient energy

(eq-fg-ym-e1)=
$$
\mathcal E_N^0(f,f)
=\int\sum_i(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2)d\rho_N.
\tag{YM.E1}
$$

Its lower semicontinuous energy on $L^2(\rho_N)$ is

(eq-fg-ym-e2)=
$$
\mathcal E_N(f,f)
=\inf_{f_j\to f\ {\rm in}\ L^2(\rho_N)}
                 \liminf_j\mathcal E_N^0(f_j,f_j),
\tag{YM.E2}
$$

where the approximants belong to the stated smooth class. This is a closed,
densely defined Markov quadratic form. It agrees with the Sobolev closure
when the original gradient is closable. It satisfies

(eq-fg-ym-e3)=
$$
\operatorname{Ent}_{\rho_N}(f^2)\le2C_*\mathcal E_N(f,f),\qquad
\operatorname{Var}_{\rho_N}f\le C_*\mathcal E_N(f,f).
\tag{YM.E3}
$$

There is a unique nonnegative self-adjoint operator $H_N^{\mathrm{eq}}$
associated with this form. Its semigroup $T_{N,\sigma}^{\mathrm{eq}}$
is conservative, positivity preserving, reversible with respect to
$\rho_N$, and satisfies

(eq-fg-ym-e4)=
$$
T_{N,\sigma}^{\mathrm{eq}}1=1,\qquad
\|T_{N,\sigma}^{\mathrm{eq}}f\|_2
\le e^{-\sigma/C_*}\|f\|_2\quad(\rho_Nf=0).
\tag{YM.E4}
$$

The kernel of $H_N^{\mathrm{eq}}$ consists of constants. The complete
Fractal Set unitary transports this entire construction, including the
energy domain, resolvent, and gap, to the encoded state.
:::

:::{prf:proof}
**Closure without changing the law.** Put $\mathcal H=L^2(\rho_N)$ and
$\mathcal V=L^2(\rho_N;\mathbb R^{2dN})$. Let $G$ be the closure in
$\mathcal H\oplus\mathcal V$ of all pairs $(f,\nabla f)$ in the
smooth class. Smooth bounded functions are dense in $\mathcal H$: first
approximate by continuous functions on a compact set of arbitrarily large
probability, then use smooth uniform approximation there. Let
$V_0=\{v:(0,v)\in G\}$, a closed subspace of $\mathcal V$.
For each $f$ in the projection of $G$ onto $\mathcal H$, the set of
vectors $v$ with $(f,v)\in G$ is one coset of $V_0$. Its unique
minimum-norm representative is $Df=\operatorname{proj}_{V_0^\perp}v$.
The graph of $D$ is
$G\cap(\mathcal H\oplus V_0^\perp)$, hence closed. Thus
$\|Df\|_2^2$ is a closed quadratic form with dense domain.

Choose smooth $f_j$ with $(f_j,\nabla f_j)\to(f,Df)$ to obtain a
recovery sequence with energies tending to $\|Df\|_2^2$. Conversely,
if $f_j\to f$ and the gradient energies have finite liminf, pass to a
subsequence whose gradients converge weakly to $v$. The closed linear
space $G$ is weakly closed, so $(f,v)\in G$ and
$\|Df\|_2^2\le\|v\|_2^2\le\liminf\|\nabla f_j\|_2^2$.
This proves {ref}`(YM.E2) <eq-fg-ym-e2>`. When the gradient is closable, $V_0=\{0\}$ and
this is its usual Sobolev energy.

For a smooth scalar contraction $\eta$ with $|\eta'|\le1$,
$\mathcal E_N^0(\eta\circ f_j)\le\mathcal E_N^0(f_j)$.
Apply a recovery sequence and {ref}`(YM.E2) <eq-fg-ym-e2>`, then approximate the scalar clipping
map, to obtain
$\mathcal E_N((0\vee f)\wedge1)\le\mathcal E_N(f)$.
This is the Markov property. The constant $1$ belongs to the domain and
has zero energy.

**The same LSI.** If $f_j\to f$ in $L^2$, then $f_j^2\to f^2$ in
$L^1$. Entropy is lower semicontinuous under this convergence. Indeed,
for $g\ge0$ its variational expression is

$$
\operatorname{Ent}_\rho(g)
=\sup_{b\ \mathrm{bounded}}
 \left\{\rho(gb)-\rho(g)\log\rho(e^b)\right\},
$$

and every expression in braces is $L^1$ continuous. Apply the upstream
LSI to a recovery sequence from the preceding paragraph to obtain the first
inequality in {ref}`(YM.E3) <eq-fg-ym-e3>`. For bounded centered $u$, its application to
$1+\varepsilon u$ gives

$$
\operatorname{Ent}_\rho((1+\varepsilon u)^2)
=2\varepsilon^2\rho(u^2)+o(\varepsilon^2),\qquad
\mathcal E_N(1+\varepsilon u)=\varepsilon^2\mathcal E_N(u).
$$

Division and truncation give the second inequality on the full domain.

**Operator and Markov semigroup.** On the complete form domain, for every
$\lambda>0$ and $g\in\mathcal H$ the equation

(eq-fg-ym-e5)=
$$
\lambda\langle u,v\rangle+\mathcal E_N(u,v)=\langle g,v\rangle
\quad(v\in\operatorname{Dom}\mathcal E_N)
\tag{YM.E5}
$$

has a unique solution by the Hilbert-space representation theorem.
The resulting resolvents are self-adjoint, satisfy the resolvent identity,
and have norm at most $1/\lambda$. They define the unique nonnegative
self-adjoint $H_N^{\mathrm{eq}}$ with this form. Equivalently,
$(I+\tau H_N^{\mathrm{eq}})^{-1}g$ minimizes
$\|u-g\|_2^2+\tau\mathcal E_N(u)$. For $0\le g\le1$ clipping $u$
to $[0,1]$ decreases both terms. This proves the interval-preserving
property of the resolvent; its value at $1$ is $1$. The strong limit

$$
T_{N,\sigma}^{\mathrm{eq}}
=\lim_{j\to\infty}(I+\sigma H_N^{\mathrm{eq}}/j)^{-j}
$$

inherits positivity, the constant function, and self-adjointness.
Consequently $\rho_N(Tf)=\langle T1,f\rangle=\rho_Nf$.
For centered $f$, {ref}`(YM.E3) <eq-fg-ym-e3>` gives
$\mathcal E_N(f)\ge C_*^{-1}\|f\|_2^2$. Spectral calculus, or
differentiation of $\|T_{N,\sigma}^{\mathrm{eq}}f\|_2^2$, proves
{ref}`(YM.E4) <eq-fg-ym-e4>`. Zero energy therefore implies a constant function.

**Reconstruction.** Transport the form by
$\widehat{\mathcal E}_N(Uf,Ug)=\mathcal E_N(f,g)$ on
$U\operatorname{Dom}\mathcal E_N$. Substituting into {ref}`(YM.E5) <eq-fg-ym-e5>` proves
the transported resolvent identity and hence
$\widehat H_N^{\mathrm{eq}}=UH_N^{\mathrm{eq}}U^{-1}$ with equality
of domains. This is the closed-form completion of
{prf:ref}`prop-fractal-set-analytic-transfer` and the induced-energy
calculation in {prf:ref}`thm-sm-direct-existing-machinery`.
:::

:::{prf:remark} Energy normalization, statuses, and the recorded evolution
:label: rem-ym-equilibrium-form-identity

The energy in {ref}`(YM.E1) <eq-fg-ym-e1>` fixes equilibrium time. Multiplying it by $a>0$
multiplies $H_N^{\mathrm{eq}}$ and its gap by $a$. For a smooth positive
density $\rho_N(ds)=p_N(s)ds$, integration by parts on compactly supported
tests gives

(eq-fg-ym-e6)=
$$
H_N^{\mathrm{eq}}f=-p_N^{-1}\nabla\cdot(p_N\nabla f).
\tag{YM.E6}
$$

The closure, rather than this formal expression, fixes its domain. On the
kinetic product reference, with $\theta=D/\gamma$,

$$
-H_N^{\mathrm{eq}}
=\Delta_x+\Delta_v-\theta^{-1}\nabla U_N\cdot\nabla_x
                         -\theta^{-1}v\cdot\nabla_v.
$$

The actual kinetic generator is the distinct recorded operator containing
Hamiltonian transport. Its conjugation in {ref}`(SM.K3) <eq-fg-sm-k3>` preserves that transport.
Equality with $-H_N^{\mathrm{eq}}$ is therefore an operator identity to
test, rather than a consequence of a shared one-time law. The ground-state
transform in {prf:ref}`lem-ym-ground-state-transform` is recovered when
its overdamped energy and normalization are used.

For a marked law, {prf:ref}`prop-kl-status-entropy` contributes
$\operatorname{Ent}_p(g^2)$ with $g_s=(\rho_s f^2)^{1/2}$.
The continuous energy alone vanishes on functions constant on each status
stratum. Its kernel then contains those functions; a full marked-law gap
uses the already established discrete energy controlling that entropy term.
Likewise a law on recorded random companions or a transition block retains
its conditional entropy. Applying {ref}`(YM.E1) <eq-fg-ym-e1>` to a continuous marginal does
not insert those additional variables into its domain.
:::

:::{div} feynman-prose
With the equilibrium transfer in hand, construct a correlation by alternating
two operations: evolve to the next time, then insert the observable measured
there. To examine reflection, condition on the state at time zero. The whole
future experiment becomes a function of that state. Reversibility gives the
same function for the reflected past, and their pairing is a squared norm.

This works for linear combinations of experiments, which is what reflection
positivity requires. It also keeps the equal-time moments of the specified
law exactly. Measurements spanning several times use the newly constructed
equilibrium transitions, so their values must be read with that clock.
:::

:::{prf:theorem} Complete equilibrium correlation and reflection construction
:label: thm-ym-equilibrium-hierarchy

For $\rho_N$ and $T_{N,\sigma}^{\mathrm{eq}}$ just constructed, the
ordered kernels define a stationary reversible cylinder law. Every bounded
one-time direct observable $A(s)$ has its original $\rho_N$ distribution
under this law. Its finite-time correlations are

(eq-fg-ym-e7)=
$$
S_N(A_1,\sigma_1;\ldots;A_n,\sigma_n)
=\langle1,M_{A_1}T_{N,\sigma_2-\sigma_1}^{\mathrm{eq}}M_{A_2}
 \cdots T_{N,\sigma_n-\sigma_{n-1}}^{\mathrm{eq}}M_{A_n}1\rangle,
\tag{YM.E7}
$$

for ordered times. At equal times this is $\rho_N(A_1\cdots A_n)$.
For a future cylinder $F=\prod_{j=1}^nA_j(X_{\sigma_j})$, define

$$
h_F=T_{N,\sigma_1}^{\mathrm{eq}}M_{A_1}
       T_{N,\sigma_2-\sigma_1}^{\mathrm{eq}}M_{A_2}\cdots M_{A_n}1.
$$

Then

(eq-fg-ym-e8)=
$$
(F,G)_{\mathrm{OS}}=\langle h_F,h_G\rangle_{\rho_N},\qquad
\|h_{\tau_\sigma F}-\rho_Nh_F\|_2
\le e^{-\sigma/C_*}\|h_F-\rho_Nh_F\|_2.
\tag{YM.E8}
$$

This constructs the equilibrium transfer Hilbert space, reflection positivity,
vacuum, and temporal gap for the direct observable algebra. The CAR lift in
{prf:ref}`cor-lqft-equilibrium-hamiltonian-lift` gives its specified
fermionic field evolution.
:::

:::{prf:proof}
**Cylinder law.** To realize the positive conservative operator by a
probability kernel, start with a countable uniformly dense rational
linear subspace of $C_0(\mathbb R^{2dN})$, enlarged to contain compact
cutoffs $0\le\chi_j\uparrow1$. Choose versions of all their images
on one common set of full measure. Linearity, positivity, and
$\|Tf\|_\infty\le\|f\|_\infty$ then hold simultaneously there.
At each such point these images extend to a bounded positive functional
on $C_0$, hence to a finite Radon measure. Since $T\chi_j\uparrow1$
almost surely (monotonicity and $L^2$ convergence give this), that measure
has mass one. Its integrals on the countable test class are measurable;
monotone approximation makes its evaluations on Borel sets measurable.
This gives the required probability kernel. The semigroup identities on
the same separating test class make the ordered cylinder distributions
consistent. Invariance makes them stationary, and self-adjointness reverses
their ordered products. These are the reversible finite-dimensional laws
used in {prf:ref}`lem-transfer-matrix-fg`.

For bounded $A$, stationarity and self-adjointness imply
$\mathbb E|A(X_{\sigma+t})-A(X_\sigma)|^2
=2\langle A,(I-T_{N,t}^{\mathrm{eq}})A\rangle\to0$ for real $A$.
Thus time-smeared observables also have an $L^2$ definition by approximation
with step functions, without requiring a pathwise differentiability claim.

**All correlations and reflection.** Conditional expectation successively
gives {ref}`(YM.E7) <eq-fg-ym-e7>` and the displayed $h_F$. Conditional independence of past
and future at zero and the established reversibility give {ref}`(YM.E8) <eq-fg-ym-e8>`, exactly
as in the transfer lemma. In particular, for any coefficients $c_j$,
$\sum_{i,j}\overline c_i c_j(F_i,F_j)_{\mathrm{OS}}
=\|\sum_jc_jh_{F_j}\|_2^2\ge0$. Translating every time of a future
functional by $\sigma$ gives
$h_{\tau_\sigma F}=T_{N,\sigma}^{\mathrm{eq}}h_F$.
Apply {ref}`(YM.E4) <eq-fg-ym-e4>` to its centered part. At equal times, each zero-time
transition is the identity, so {ref}`(YM.E7) <eq-fg-ym-e7>` equals the actual static moment.
A channel spanning several recorded stages uses all of those times;
replacing its transition law by $T^{\mathrm{eq}}$ changes that block law.

**Growth and fields.** For bounded insertions,
$|S_N|\le\prod_j\|A_j\|_\infty$. For the smooth empirical fields
covered by {ref}`(YM.F1) <eq-fg-ym-f1>`, the newly constructed stationarity and the same LSI
give {ref}`(YM.F2) <eq-fg-ym-f2>`--{ref}`(YM.F3) <eq-fg-ym-f3>` in equilibrium time. Products and reflected products
are integrable by these moment estimates, so truncation extends the
reflection formula to their polynomial algebra.

For the CAR construction set $\mathbb T_\sigma=e^{-\sigma\mathbb H_{\mathrm{eq}}}$.
A word of bounded CAR insertions has boundary vector
$v_F=\mathbb T_{\sigma_1}A_1\mathbb T_{\sigma_2-\sigma_1}
A_2\cdots A_n\Omega$. Its reflected word is the adjoint word, in
reverse order, and its reflected matrix element is $\langle v_F,v_G\rangle$.
For example $(a^\dagger(f)a^\dagger(g))^*=a(g)a(f)$; retaining this
order and then using the CAR gives the determinant in {ref}`(LQ.R4) <eq-fg-lq-r4>`. This fixes
the signs and proves positivity on these represented words. A classical
commuting field product retains its own multiplication algebra.
:::


:::{div} feynman-prose
Suppose you keep only a descriptor of the swarm. A function of that descriptor
may, after evolution, depend on coordinates you omitted. Projecting back to
the retained observables removes that dependence. Over a second time interval,
however, those omitted coordinates can affect a retained observable again.
This is the origin of the exact channel memory.

For the self-adjoint equilibrium transfer, the calculation below measures
the two-step discrepancy by the squared norm of the component removed after
one step. It vanishes at every time precisely when the retained observable
space stays invariant under evolution. In that case the channel has its own
exact semigroup and inherits the gap. Otherwise, the complete correlation
formulas and the memory terms still compute its statistics. Repeating the
compressed one-step operator alone would omit the calculated contribution.
:::

:::{prf:theorem} Equilibrium channel closure and its exact projection defect
:label: thm-ym-equilibrium-channel-closure

Use the established equilibrium law $\rho_N$ and transfer
$T_\sigma=T_{N,\sigma}^{\mathrm{eq}}$. For a recorded state descriptor
$q$ let $\Pi=\mathbb E_{\rho_N}[\cdot\mid\sigma(q)]$,
$R=I-\Pi$, and $\mathcal H_q=\operatorname{Ran}\Pi$. The compressed
transfers $\overline T_\sigma=\Pi T_\sigma|_{\mathcal H_q}$ satisfy

(eq-fg-ym-p1)=
$$
\overline T_{2\sigma}-\overline T_\sigma^2
 =(RT_\sigma|_{\mathcal H_q})^*
   (RT_\sigma|_{\mathcal H_q})\succeq0.
\tag{YM.P1}
$$

For all $\sigma\ge0$, these compressed transfers form a semigroup
precisely when $\mathcal H_q$ is invariant under every $T_\sigma$.
In that case it is reducing, its generator is the restriction of
$H_N^{\mathrm{eq}}$, and it has the same centered gap bound $1/C_*$.
The restricted closed form specifies the energy and its complete
domain on the closed channel theory. On smooth pullbacks its energy
is bounded above by the original gradient energy, with equality
when that gradient is closable as in the established bounded-tilt family.

Without this invariance, the complete equilibrium correlations remain
the boundary-vector construction
{ref}`(YM.E7) <eq-fg-ym-e7>`--{ref}`(YM.E8) <eq-fg-ym-e8>`.
At any chosen observation spacing their exact channel memory is
{ref}`(SM.M2) <eq-fg-sm-m2>`, with this same equilibrium kernel
and this same conditional projection. Formula
{ref}`(YM.P1) <eq-fg-ym-p1>` measures its two-step defect.
:::

:::{prf:proof}
Insert $I=\Pi+R$ between the two factors of $T_{2\sigma}=T_\sigma^2$.
For $f\in\mathcal H_q$ this gives

$$
\Pi T_{2\sigma}f
=\Pi T_\sigma\Pi T_\sigma f+\Pi T_\sigma R T_\sigma f.
$$

Self-adjointness of $T_\sigma$, $\Pi$, and $R$ identifies the second
term with the positive operator in
{ref}`(YM.P1) <eq-fg-ym-p1>`. If the compressed family is a
semigroup, that positive operator is zero; its quadratic form is
$\|RT_\sigma f\|_2^2$, so $T_\sigma f\in\mathcal H_q$.
Conversely invariance removes $R$ from all products and proves
the semigroup law.

For $g\perp\mathcal H_q$ and $f\in\mathcal H_q$,
$\langle T_\sigma g,f\rangle=\langle g,T_\sigma f\rangle=0$.
Thus the orthogonal complement is invariant as well. The projection
commutes with the transfers and hence with their resolvents and spectral
projections. Restricting the spectral measure proves the generator
and domain identification, and gives

$$
\mathcal E_q(f,f)
=\int\lambda\,d\langle f,E_H(\lambda)f\rangle
=\mathcal E_N(f,f),\qquad
f\in\mathcal H_q\cap\operatorname{Dom}\mathcal E_N.
$$

This restriction has dense form domain in $\mathcal H_q$:
the spectral truncations $E_H([0,M])f$ belong to that domain and
converge to every $f\in\mathcal H_q$. Its constant function is $1$.
The full-law Poincaré inequality therefore gives the same centered
gap and its only zero-energy vectors are constants.
On smooth descriptor pullbacks relaxation gives
$\mathcal E_N(f)\le\mathcal E_N^0(f)$. When the original gradient
is closable these energies agree. If those pullbacks also form a
core for the restricted form, closing their induced energy gives
that restriction. In every case the displayed spectral restriction
specifies the complete domain; the bounded-tilt Sobolev realization
is treated explicitly in the next proposition.

Finally choose $P=T_{\sigma_0}$ in
{prf:ref}`thm-sm-direct-channel-memory`.
Self-adjointness makes its blocks satisfy
$\mathsf B=\mathsf C^*$, so its first memory term
$\mathsf B\mathsf C$ is exactly
{ref}`(YM.P1) <eq-fg-ym-p1>` at $\sigma_0$.
Multiplication by a bounded channel commutes with $\Pi$, which
identifies every inserted block product with its original
equilibrium correlation.
:::

:::{div} feynman-prose
A validity mask reports zero on one side of a threshold and one on the
other. Its values are bounded, so inserting it into a correlation is
straightforward. Gradient energy measures something else: how sharply its
value changes across nearby states.

Smooth that jump across a layer of width $\varepsilon$. Its slope is of
order $1/\varepsilon$, so integrating the squared slope across the layer
costs energy of order $1/\varepsilon$. The proof below establishes this
divergence on the regular force chart under the specified law. A hard mask
therefore remains a valid bounded observable while lying outside the
finite-energy domain. A simulator can use its bounded observable matrix;
the following threshold calculation gives a quantitative way to approximate
its bounded correlations by replacing the jump with a ramp.
:::

:::{prf:proposition} Hard color masks and the inherited energy domain
:label: prop-ym-color-mask-domain

In the product or bounded-tilt equilibrium family, consider the valid-color
indicator $m_i=1_{\{\|F_i^{\mathrm{visc}}\|>\delta\}}$ with its positive
implemented threshold $\delta$. On the regular geometric chart used in
{prf:ref}`cor-ym-nonzero-direct-color-sector`, take a vertex with
a nonzero incident force row. Then this indicator has no finite
full-gradient Sobolev energy. Consequently the restriction of that
energy to the binary descriptor $q=m_i$ has only constants in its
form domain and is not densely defined on $L^2(q_*\rho_N)$.

Its bounded recorded and equilibrium correlations are nevertheless
well-defined by {ref}`(SM.K4) <eq-fg-sm-k4>` and
{ref}`(YM.E7) <eq-fg-ym-e7>`. The conditional compression and
memory formulas apply to it. A finite channel simulator must therefore
distinguish a bounded observable matrix from a finite-energy basis mode.
:::

:::{prf:proof}
For fixed positions in that chart,
$F_i^{\mathrm{visc}}=\nu\sum_jw_{ij}(v_j-v_i)$ is a nonzero
linear map onto $\mathbb R^d$ in the velocity variables: a nonzero
scalar coefficient multiplies every component of one velocity.
On the level set $\|F_i^{\mathrm{visc}}\|=\delta>0$, the gradient
of its norm in those variables is nonzero. Smooth local coordinates
therefore give a normal coordinate
$u=\|F_i^{\mathrm{visc}}\|-\delta$ across a patch of that surface.
The product density is smooth and positive. The bounded-tilt
comparison bounds the actual density above and below on a compact
subpatch, and the coordinate Jacobian has the same bounds.

In these coordinates $m_i=1_{\{u>0\}}$.
For a test supported in the patch, integration by parts in $u$
gives its distributional normal derivative as surface measure at
$u=0$. This measure is nonzero and singular with respect to volume;
it cannot be represented by an $L^2$ weak derivative.
The weighted Sobolev domain is locally the ordinary Sobolev domain
because the density is bounded above and below. Thus $m_i$ does
not belong to it.

The energy cost can also be seen directly. A smooth transition
$h_\varepsilon$ from zero at $-\varepsilon$ to one at
$\varepsilon$ obeys

(eq-fg-ym-p2)=
$$
1=\left|\int_{-\varepsilon}^{\varepsilon}
          h_\varepsilon'(u)\,du\right|^2
\le2\varepsilon\int_{-\varepsilon}^{\varepsilon}
                |h_\varepsilon'(u)|^2du.
\tag{YM.P2}
$$

Integrating over a fixed tangential subpatch gives an energy cost
at least a positive constant times $\varepsilon^{-1}$.
Bounded whole-law density ratios preserve closability of the
reference gradient and equivalence of its weighted norms, so the
relaxed form here has this same Sobolev domain.

Both sides of the surface contain open sets of positive probability.
Every function of the binary descriptor is $a+b m_i$.
For $b\ne0$ it has the same nonzero surface derivative, while
$b=0$ gives a constant. The domain on this two-dimensional
$L^2$ space is therefore only its one-dimensional constant subspace.
The established bounded-observable correlation formulas require
no Sobolev derivative of $m_i$, which proves the final assertion.
:::


:::{div} feynman-prose
Fix the recorded positions and evaluate the implemented viscous force at
the same stage as the velocities. Under the product reference, this force
is a linear combination of Gaussian velocities. Its length therefore has
an explicit radial density. Integrating that density across a narrow shell
around the validity threshold measures how often smoothing the mask changes
the color value. Maximizing over the Gaussian scale makes the bound work
for every force row; a zero row never reaches the positive threshold.

The bounded joint tilt transfers this shell bound to its specified
equilibrium law. A ramp agrees with the hard mask outside the shell, so
the shell probability controls the color's squared error. Expanding a pair,
a determinant, or a product of measurements one changed factor at a time
then gives the displayed correlation errors. For a normalized frame average,
the last calculation also retains the recorded weights, denominator, and
empty-average convention.
:::

:::{prf:proposition} Quantitative removal of the color-threshold discontinuity
:label: prop-ym-color-threshold-correlation-limit

Use the product or bounded whole-joint tilt equilibrium family of
{prf:ref}`thm-ym-equilibrium-fluctuation-limit`, with its bound $B_*$,
and the same-stage viscous color map of
{prf:ref}`def-sm-direct-observable-law`. Write $\delta_c>0$ for the
implemented validity threshold. For $0<\varepsilon<\delta_c/2$, put

(eq-fg-ym-v1)=
$$
K_d=\frac{2^{1-d/2}d^{d/2}e^{-d/2}}{\Gamma(d/2)},\qquad
p_\varepsilon=e^{B_*}K_d
 \log\frac{\delta_c+\varepsilon}{\delta_c-\varepsilon}.
\tag{YM.V1}
$$

For every vertex, including a vertex with zero viscous-force row,

(eq-fg-ym-v2)=
$$
\rho_N\{\left|\|F_i^{\mathrm{visc}}\|-\delta_c\right|
                 \le\varepsilon\}
\le p_\varepsilon
\le 4e^{B_*}K_d\varepsilon/\delta_c.
\tag{YM.V2}
$$

Let $\chi_\varepsilon:[0,\infty)\to[0,1]$ be a continuous ramp,
zero below $\delta_c-\varepsilon$ and one above
$\delta_c+\varepsilon$. Replace the hard extension $c_i$ by
$c_i^\varepsilon=\chi_\varepsilon(\|F_i\|)F_i
\exp(i\kappa v_i)/\|F_i\|$, with value zero at $F_i=0$.
For a fixed pair contraction $q_{ij}=c_i^\dagger c_j$ and the
three-color determinant (in dimension $d=3$) $b_{ijk}=\det[c_i,c_j,c_k]$, this gives

(eq-fg-ym-v3)=
$$
\|c_i^\varepsilon-c_i\|_{L^2(\rho_N;\mathbb C^d)}
 \le\sqrt{p_\varepsilon},\qquad
\|q_{ij}^\varepsilon-q_{ij}\|_2\le2\sqrt{p_\varepsilon},\qquad
\|b_{ijk}^\varepsilon-b_{ijk}\|_2\le3\sqrt{p_\varepsilon}.
\tag{YM.V3}
$$

For any finite collection of such contractions at arbitrary equilibrium
times, let $r_a=2$ for a pair and $r_a=3$ for a determinant. Then

(eq-fg-ym-v4)=
$$
\left|\mathbb E\prod_{a=1}^nO_a^\varepsilon(X_{t_a})
       -\mathbb E\prod_{a=1}^nO_a(X_{t_a})\right|
\le\left(\sum_{a=1}^nr_a\right)\sqrt{p_\varepsilon}.
\tag{YM.V4}
$$

The constants are independent of $N$ and of the observation times.
This proves convergence of these actual bounded color insertions under
threshold-ramp removal. It neither assigns finite gradient energy to the
hard mask nor asserts convergence of a population-rescaled fluctuation.
:::

:::{prf:proof}
**1. Integrate the force-radius shell.** Under the product reference,
conditional on positions, the velocities are independent centered
Gaussians with covariance $\theta I_d$. The implemented force is
$F_i=\nu\sum_jw_{ij}(v_j-v_i)$. Its scalar coefficients depend on
the positions, so its conditional covariance is $s_i^2 I_d$, where
$s_i^2=\theta\sum_j a_{ij}^2$ and $a_{ij}$ are its complete velocity
coefficients. When $s_i=0$, its radius is zero and the shell in
{ref}`(YM.V2) <eq-fg-ym-v2>` is empty. When $s_i>0$, its radial density is

$$
f_{s_i}(r)=\frac{2^{1-d/2}}{\Gamma(d/2)}
              r^{d-1}s_i^{-d}e^{-r^2/(2s_i^2)},\qquad r>0.
$$

For fixed $r$, differentiation of
$-d\log s-r^2/(2s^2)$ gives its maximum at $s=r/\sqrt d$.
Thus $f_s(r)\le K_d/r$ for every positive scale, without needing a
uniform lower bound on the incident weights. Integrating this bound
between $\delta_c-\varepsilon$ and $\delta_c+\varepsilon$, then over
positions, proves the reference bound. The already proved density
comparison $d\rho_N/dm_U^{\otimes N}\le e^{B_*}$ proves the first
inequality in {ref}`(YM.V2) <eq-fg-ym-v2>`. Finally

$$
\log\frac{\delta_c+\varepsilon}{\delta_c-\varepsilon}
=\int_{\delta_c-\varepsilon}^{\delta_c+\varepsilon}\frac{dr}{r}
\le\frac{2\varepsilon}{\delta_c-\varepsilon}
\le\frac{4\varepsilon}{\delta_c}
$$

proves the second inequality. The calculation uses the same-stage
position-dependent scalar force weights. It does not replace a different
velocity-dependent force or a lagged mixed-stage record by this map.

**2. Bound the actual insertions.** Both color extensions have norm at
most one. Their difference is supported on the shell, with norm at most
one, proving the first bound in {ref}`(YM.V3) <eq-fg-ym-v3>`.
The exact telescoping identities are

$$
\begin{aligned}
q_{ij}^\varepsilon-q_{ij}
 &=(c_i^\varepsilon-c_i)^\dagger c_j^\varepsilon
                    +c_i^\dagger(c_j^\varepsilon-c_j),\\
b_{ijk}^\varepsilon-b_{ijk}
 &=\det[c_i^\varepsilon-c_i,c_j^\varepsilon,c_k^\varepsilon]
 +\det[c_i,c_j^\varepsilon-c_j,c_k^\varepsilon]
 +\det[c_i,c_j,c_k^\varepsilon-c_k].
\end{aligned}
$$

Cauchy--Schwarz for pairs, Hadamard's inequality for determinants,
and the $L^2$ triangle inequality give the remaining bounds.
All contraction magnitudes are at most one. Telescoping the product
of $n$ insertions and using stationarity bounds each expectation by
the corresponding $L^1$, hence $L^2$, error. This proves
{ref}`(YM.V4) <eq-fg-ym-v4>` for the same joint history law, without
independence between vertices or observation times.

**3. Retain the recorded averaging denominator.** For a weighted recorded
average in {prf:ref}`def-sm-direct-observable-law`, let
$D=\sum_Iw_Im_I$, $D^\varepsilon=\sum_Iw_Im_I^\varepsilon$ and
$\Delta=\sum_Iw_I|m_I-m_I^\varepsilon|$. With unchanged bounded
unmasked values $|O_I|\le B$, the numerator difference is at most
$B\Delta$ and $|D-D^\varepsilon|\le\Delta$. On $D>0$,

$$
|\mathcal A^\varepsilon(O)-\mathcal A(O)|
\le\min\{2B,2B\Delta/D\}.
$$

For $D^\varepsilon>0$ this follows by adding and subtracting its
numerator divided by $D$; for $D^\varepsilon=0$, $\Delta=D$ and the
zero-denominator convention gives the same bound. On $D=0$ the hard
average is zero, while the ramp average is bounded by
$B\mathbf1_{\{D^\varepsilon>0\}}$. These are the exact denominator
terms for the implemented normalization. Consequently the fixed-contraction
bound {ref}`(YM.V4) <eq-fg-ym-v4>` does not silently supply a uniform
bound for a population-dependent ratio with a small random denominator.
:::

:::{div} feynman-prose
Now take the bounded channel value actually computed from each frame,
including its masks and normalization, and sample it along the constructed
equilibrium process. A lagged correlation estimate averages products from
pairs of frames separated by the chosen lag. Nearby products have
overlapping observation windows, so the proof counts their covariance
directly. For separated windows, the equilibrium transfer gap bounds the
remaining covariance by a geometric series.

Adding both contributions gives the explicit mean-square error proportional
to $1/K$. Subtracting an estimated mean introduces the second displayed
error term, which is controlled using the same channel bound. Both
estimates apply to hard masks as recorded, and the mean and lag products
can be estimated from the same run. This gives a sampling bound for the
equilibrium correlations of the actual frame channels.
:::

:::{prf:proposition} Consistent lag correlations of the bounded recorded channels
:label: prop-ym-channel-lag-estimation

Sample the established equilibrium process of
{prf:ref}`thm-ym-equilibrium-hierarchy` at spacing $h>0$, and let $O$
be a bounded actual frame channel from
{prf:ref}`prop-sm-channel-estimate-routes`, including its masks and
zero-denominator convention. Put $|O|\le B$, $\mu=\rho_NO$,
$r=e^{-h/C_*}$ and, for a fixed integer lag $\ell\ge0$,

$$
\widehat R_{K,\ell}=\frac1K\sum_{a=0}^{K-1}
            \overline{O(X_{ah})}O(X_{(a+\ell)h}),\qquad
\Lambda_\ell=1+2\ell+\frac{2r}{1-r}.
$$

For the same equilibrium two-point function
$R_\ell=\mathbb E[\overline{O(X_0)}O(X_{\ell h})]$,

(eq-fg-ym-v5)=
$$
\mathbb E\widehat R_{K,\ell}=R_\ell,\qquad
\mathbb E|\widehat R_{K,\ell}-R_\ell|^2
\le\frac{B^4\Lambda_\ell}{K}.
\tag{YM.V5}
$$

If $\widehat\mu_T=T^{-1}\sum_{a=0}^{T-1}O(X_{ah})$ and
$\widehat C_{K,\ell}=\widehat R_{K,\ell}-|\widehat\mu_T|^2$,
the connected correlation $C_\ell=R_\ell-|\mu|^2$ satisfies

(eq-fg-ym-v6)=
$$
\|\widehat C_{K,\ell}-C_\ell\|_{L^2}
\le B^2\sqrt{\Lambda_\ell/K}
             +2B^2\sqrt{\Lambda_0/T}.
\tag{YM.V6}
$$

Here $B=1$ for the masked pair and determinant channels, $B=2$ for
$1-\operatorname{Re}\Pi$ and the bounded standard doublet sums and
differences. The inequalities hold without smoothness of these channels.
They give the sampling limit of the identified equilibrium hierarchy;
survival-conditioned algorithm records retain their exact history law
in {prf:ref}`prop-ym-qsd-history-identification`.
:::

:::{prf:proof}
Set $Z_a=\overline{O(X_{ah})}O(X_{(a+\ell)h})$.
Stationarity gives $\mathbb EZ_a=R_\ell$ and
$\mathbb E|Z_a-R_\ell|^2\le B^4$. If $1\le n\le\ell$, the
windows may overlap, and Cauchy--Schwarz gives
$|\operatorname{Cov}(Z_0,Z_n)|\le B^4$.
If $n>\ell$, condition the first centered window on its right endpoint
and the second on its left endpoint. Both conditional boundary functions
are centered and have $L^2$ norm at most $B^2$. The Markov property
separates them by $(n-\ell)h$. The proved equilibrium gap therefore gives

$$
|\operatorname{Cov}(Z_0,Z_n)|\le B^4r^{n-\ell}.
$$

This is a conditional-expectation contraction and applies equally to
complex observables. Expanding the squared error of the average gives

$$
\begin{aligned}
\mathbb E|\widehat R_{K,\ell}-R_\ell|^2
&\le\frac{B^4}{K^2}
\left[K+2\sum_{n=1}^{K-1}(K-n)
 \left(\mathbf1_{n\le\ell}+r^{n-\ell}\mathbf1_{n>\ell}\right)\right]\\
&\le\frac{B^4}{K}
       \left(1+2\ell+2\sum_{j\ge1}r^j\right),
\end{aligned}
$$

which proves {ref}`(YM.V5) <eq-fg-ym-v5>`. Applying the same sum to
the single-time observable yields
$\|\widehat\mu_T-\mu\|_2\le B\sqrt{\Lambda_0/T}$.
Since $|\widehat\mu_T|,|\mu|\le B$,

$$
\big||\widehat\mu_T|^2-|\mu|^2\big|
\le2B|\widehat\mu_T-\mu|.
$$

The triangle inequality proves
{ref}`(YM.V6) <eq-fg-ym-v6>`. No independence between the estimated
mean and the estimated lag product is used. The result estimates the
stated correlation; a later logarithmic mass fit retains its denominator
and signal-sign requirements.
:::


:::{prf:corollary} An infinite-dimensional direct color OS space at finite population
:label: cor-ym-nonzero-direct-color-sector

For the active viscous color channel of Chapter 04, its coupling $\nu$
and phase coefficient $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$ are
nonzero. Evaluate force and velocity at the same recorded stage, with the
actual positive graph weights on one of its regular geometric charts.
For the continuous law in the bounded-tilt family of
{prf:ref}`thm-ym-equilibrium-fluctuation-limit`, a contraction
$q_{ij}=c_i^\dagger c_j$ between two vertices with nonzero incident
force rows has a nonconstant real or imaginary component. Its centered
observable $A$ consequently satisfies

$$
\|A\|_{L^2(\rho_N)}>0,\qquad
\langle A,T_{N,2\sigma}^{\mathrm{eq}}A\rangle>0
\quad(\sigma>0).
$$

For every integer $q\ge1$ and fixed $\sigma>0$, the future color
observables $1,A(X_\sigma),\ldots,A(X_\sigma)^q$ have a positive
definite OS Gram matrix. Thus the OS space generated by the actual direct
color algebra is infinite dimensional.
:::

:::{prf:proof}
Fix positions in the regular chart, so its finite weights $w_{k\ell}$
are fixed. Choose velocities $v_k=t_ke_1$. The implemented viscous force
is

$$
F_k^{\mathrm{visc}}=\nu\left[\sum_\ell
       w_{k\ell}(t_\ell-t_k)\right]e_1=:b_k(t)e_1.
$$

For the selected vertices the linear functionals $b_i,b_j$ are nonzero.
Choose $t$ outside their two zero hyperplanes and scale it until both
forces exceed the already specified valid-color threshold. On a small
neighborhood their signs $\epsilon_i,\epsilon_j$ and validity masks
are fixed. The recorded normalization and phase formula then give

$$
c_i=\epsilon_i e^{i\kappa t_i}e_1,\qquad
c_j=\epsilon_j e^{i\kappa t_j}e_1,\qquad
q_{ij}=\epsilon_i\epsilon_j e^{i\kappa(t_j-t_i)}.
$$

Changing $t_j-t_i$ by a sufficiently small nonzero amount preserves the
force signs and changes $q_{ij}$. At least one of its real and imaginary
components therefore has two distinct values. The regular-chart
continuity and the strict validity margins give two full-dimensional
neighborhoods where these component values remain separated. Both have
positive reference probability, since $m_U^{\otimes N}$ has a strictly
positive density. The bounded-tilt comparison preserves their positive
probabilities under $\rho_N$. Hence that component has positive variance,
including the implemented invalid-sample convention outside the chart.

For the centered component $A$, spectral calculus gives
$\langle A,T_{N,2\sigma}^{\mathrm{eq}}A\rangle
=\int e^{-2\sigma\lambda}d\langle A,E_H(\lambda)A\rangle>0$.
The boundary vector of the future observable $A(X_\sigma)$ has exactly
this squared norm by {prf:ref}`thm-ym-equilibrium-hierarchy`.
To compute the higher Gram ranks, restrict the preceding velocity
variation to a sufficiently short interval where the selected sine or
cosine component is strictly monotone. Its values fill an interval $I$.
For every $y\in I$, continuity at a configuration with value $y$ and the
positive density imply that every neighborhood of $y$ has positive
pushforward probability. Thus $I$ lies in the support of the law of $A$,
after its centering translation. A nonzero polynomial $p$ cannot vanish
on $I$; continuity then gives $\rho_N|p(A)|^2>0$.

The boundary vector of $p(A(X_\sigma))$ is
$T_{N,\sigma}^{\mathrm{eq}}p(A)$, since all its factors are measured at
the same time. Consequently, for $p_a(x)=\sum_{j=0}^qa_jx^j$,

$$
\begin{aligned}
\sum_{j,k=0}^q\overline{a_j}a_k
 (A(X_\sigma)^j,A(X_\sigma)^k)_{\mathrm{OS}}
&=\|T_{N,\sigma}^{\mathrm{eq}}p_a(A)\|_2^2\\
&=\int_{[0,\infty)}e^{-2\sigma\lambda}
       d\langle p_a(A),E_H(\lambda)p_a(A)\rangle>0
       \quad(a\ne0).
\end{aligned}
$$

The last inequality follows because the spectral measure has total mass
$\|p_a(A)\|_2^2>0$ and the integrand is strictly positive. These are
bounded color insertions, so no derivative of a validity mask is taken.
Arbitrarily large Gram ranks prove the asserted infinite dimension.
This is a finite-population statement. A population-uniform
lower bound for a specified normalized color-channel sequence retains
its actual channel variance and force-normalization estimates; the
empirical phase-space fluctuation limit does not replace that calculation.
:::


:::{prf:lemma} Reflection factorization for a reversible Markov path law
:label: lem-transfer-matrix-fg

Let $(X_t)_{t\in\mathbb R}$ have a stationary reversible Markov law with
invariant probability $\pi$. For bounded future functionals define
$h_F(x)=\mathbb E[F\mid X_0=x]$. Then

(eq-fg-ym-40)=
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
{ref}`(YM.39) <eq-fg-ym-39>` are therefore $\overline{h_F}$ and $h_G$, proving {ref}`(YM.40) <eq-fg-ym-40>`.
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
measure is the transformed law {ref}`(YM.18) <eq-fg-ym-18>`.

There is also a finite reversibility calculation for a frozen companion
selection. If $w_{ij}=w_{ji}\ge0$, $Z_i=\sum_jw_{ij}>0$, and
$P_{ij}=w_{ij}/Z_i$, then $q_i=Z_i/\sum_kZ_k$ satisfies
$q_iP_{ij}=q_jP_{ji}$. The resulting chain on companion indices is reversible.
This proves a property of that frozen chain; reversibility of the complete
swarm kernel requires every kinetic and cloning component as well.

:::{prf:proposition} Momentum reflection in the existing kinetic reference
:label: prop-ym-kinetic-reflection-form

For a stationary process with a measure-preserving state involution $R$,
suppose its already specified transition obeys $P_t^*=RP_tR$. Define
$\Theta_RF=F((RX_{-t})_{t\ge0})$ on future functionals and
$h_F(x)=\mathbb E[F\mid X_0=x]$. The exact reflected form is

(eq-fg-ym-r1)=
$$
\mathbb E[\overline{\Theta_RF}G]=\langle Rh_F,h_G\rangle_\pi.
\tag{YM.R1}
$$

For the harmonic kinetic reference used in
{prf:ref}`thm-uv-protection-mechanism`, this form takes negative values
even on position observables, which are even under velocity reversal.
:::

:::{prf:proof}
The backward conditional kernel in a stationary process is $P_t^*$.
Conjugating it by $R$ gives $P_t$, so conditional on $X_0=x$ the
reflected past has the forward law started at $Rx$. Its conditional
expectation is $h_F(Rx)$. The past and future are conditionally independent
given $X_0$, proving {ref}`(YM.R1) <eq-fg-ym-r1>`. The involution $R$ is a self-adjoint unitary;
its quadratic form is the difference of squared norms of its even and odd
components, rather than a squared norm in general.

Now use the established kinetic reference in one spatial dimension,

$$
dX_t=V_tdt,\qquad
dV_t=-\kappa X_tdt-\gamma V_tdt+\sqrt{2\gamma\theta}\,dW_t.
$$

Its stationary covariance is $C=\operatorname{diag}(\theta/\kappa,\theta)$.
For $B=\left(\begin{smallmatrix}0&1\\-\kappa&-\gamma\end{smallmatrix}\right)$
and $R=\operatorname{diag}(1,-1)$, direct multiplication gives

$$
BC+CB^{\mathsf T}+2\gamma\theta
 \begin{pmatrix}0&0\\0&1\end{pmatrix}=0,\qquad
CB^{\mathsf T}C^{-1}=RBR.
$$

The stationary Gaussian two-time law therefore satisfies $P_t^*=RP_tR$:
its reversed conditional mean matrix is $Ce^{tB^{\mathsf T}}C^{-1}
=Re^{tB}R$, and the conditional covariance transforms by the same
involution. For $\omega=\sqrt{\kappa-\gamma^2/4}>0$,

$$
\mathbb E[X_t\mid X_0=x,V_0=v]
=e^{-\gamma t/2}
 \left[\left(\cos\omega t+\frac\gamma{2\omega}\sin\omega t\right)x
       +\frac{\sin\omega t}{\omega}v\right].
$$

Let these two coefficients be $a(t)$ and $b(t)$. Since $X_t$ itself is
even under velocity reversal, {ref}`(YM.R1) <eq-fg-ym-r1>` for $F=X_t$ gives

(eq-fg-ym-r2)=
$$
\begin{aligned}
(F,F)_{\Theta_R}
&=\frac\theta\kappa a(t)^2-\theta b(t)^2\\
&=\frac\theta\kappa e^{-\gamma t}
       \left[\cos(2\omega t)+\frac\gamma{2\omega}\sin(2\omega t)\right].
\end{aligned}
\tag{YM.R2}
$$

At $t=\pi/(2\omega)$ this equals
$-(\theta/\kappa)e^{-\gamma\pi/(2\omega)}<0$.
The choice $\kappa=\gamma=\theta=1$ already lies in this reference
family. Its full-gradient Gaussian LSI constant is one. Bounded
truncations of $X_t$ converge in $L^2$; by Cauchy--Schwarz their reflected
forms converge to the same negative value. This proves the failure on
bounded future observables as well. The equilibrium transfer of {ref}`(YM.E1) <eq-fg-ym-e1>`
has the positive form {ref}`(YM.E8) <eq-fg-ym-e8>` because it is constructed from the symmetric
energy, with its own time evolution.
:::


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
For $F=\sum_jc_jF_j$, {ref}`(YM.40) <eq-fg-ym-40>` gives

$$
 (F,F)_{\rm OS}
 =\left\|\sum_jc_jh_{F_j}\right\|^2\ge0.
$$

This proves positivity of every finite Gram matrix and the general
factorization statement.

For the last assertion, {prf:ref}`prop-ym-kinetic-reflection-form`
computes the reflected form in the kinetic reference already used in the
volume. At $\kappa=\gamma=\theta=1$ and $t=\pi/\sqrt3$, it gives

(eq-fg-ym-41)=
$$
 \mathbb E[X(-t)X(t)]=-e^{-\pi/\sqrt3}<0.
\tag{YM.41}
$$

The same law has full-gradient LSI constant one and satisfies the
momentum-reversal transition identity. The bounded-truncation calculation
there establishes the failure on the bounded future algebra. Thus neither
the static LSI nor that reversal identity supplies the positive factorization.
The equilibrium construction {ref}`(YM.E8) <eq-fg-ym-e8>` establishes it for its own transition.

:::

:::{div} feynman-prose
Picture the damped harmonic particle in the calculation above. Its stationary
position and velocity have a Gaussian distribution, and reversing a trajectory
reverses velocity. Nevertheless, knowing its present velocity helps predict a
future position: the conditional expectation of that position contains both
an $x$ term and a $v$ term.

Momentum reflection leaves the first term even and makes the second odd.
Their contributions to the reflected norm have opposite signs. At the
displayed underdamped time the negative contribution wins, even though the
measured position itself is unchanged by velocity reversal. Thus the exact
momentum-reversal identity can coexist with a negative reflection form.
The equilibrium construction proves positivity through its own symmetric
transfer and squared-norm factorization.
:::

:::{prf:corollary} Positive quotient and its possible dimension
:label: cor-os2-nondegeneracy

Under reflection positivity, quotient $\mathcal A_+$ by its nullspace
$\mathcal N=\{F:(F,F)_{\rm OS}=0\}$ and complete it. The result is a
Hilbert space, isometrically the closure of $\{h_F\}$ when {ref}`(YM.40) <eq-fg-ym-40>` holds.
The class of $1$ has norm one. Infinite dimension requires arbitrarily large
positive-rank Gram matrices; it does not follow from positivity alone.
:::

:::{prf:proof}
Positivity of the form on $F+zG$ for every complex $z$ implies its
Cauchy–Schwarz inequality. Hence every null vector is orthogonal to every
vector, and the form descends to a positive definite inner product on the
quotient. Formula {ref}`(YM.40) <eq-fg-ym-40>` identifies its nullspace with the kernel of
$F\mapsto h_F$, proving the isometry. Normalization gives $(1,1)_{\rm OS}=1$.
The maximal number of linearly independent quotient vectors is exactly the
supremum of the ranks of their Gram matrices. For the bounded-tilt
fluctuation hierarchy constructed below, those ranks are explicitly
unbounded by {prf:ref}`thm-ym-fluctuation-os-infinite-dimension`.
:::

:::{div} feynman-prose
The equilibrium gap tells us how a measured average settles down. Two
observations separated by a long equilibrium time have exponentially small
covariance. To compute the variance of a time average, add these covariances
over every pair of observation times. Pairs with separation $u$ contribute
with weight $T-u$, which gives the integral below.

The result bounds the averaging variance by a constant times $1/T$, with
the constant supplied by the established LSI and the observable's variance.
The same temporal decay survives in the limiting OS correlations. Its
translation into a spatial decay statement uses the transformations actually
proved to preserve the field law and transfer.
:::

:::{prf:theorem} Quantitative clustering for the constructed equilibrium transfer
:label: thm-os-os3-fg

Use the same law, relaxed gradient form, and transfer
$T_t^{\mathrm{eq}}=e^{-tH^{\mathrm{eq}}}$ constructed in
{prf:ref}`thm-ym-equilibrium-form-construction`. Their established LSI
constant $C_*$ gives

(eq-fg-ym-42)=
$$
 \operatorname{Var}_\rho f\le C_*\mathcal E(f,f),\qquad
 \|T_t^{\mathrm{eq}}f-\rho f\|_2
 \le e^{-t/C_*}\|f-\rho f\|_2.
\tag{YM.42}
$$

Thus {prf:ref}`thm-cluster-decomposition` applies to this transfer with
$M=1$ and $\lambda=1/C_*$. For its stationary process and real
$A\in L^2(\rho)$, the time average satisfies

(eq-fg-ym-q1)=
$$
\operatorname{Var}\left(\frac1T\int_0^T A(X_t)\,dt\right)
\le\frac{2\operatorname{Var}_\rho A}{T^2}
 \left[C_*T-C_*^2(1-e^{-T/C_*})\right]
\le\frac{2C_*}{T}\operatorname{Var}_\rho A.
\tag{YM.Q1}
$$

The limiting equilibrium hierarchy of
{prf:ref}`thm-ym-equilibrium-fluctuation-limit` has the same exponential
bound on centered OS matrix elements. These conclusions concern equilibrium
time. Spatial clustering follows for the spacetime transformations actually
proved to preserve this law and transfer, as specified in
{prf:ref}`thm-cluster-decomposition`.
:::

:::{prf:proof}
The relaxed-form Poincaré inequality in
{prf:ref}`thm-ym-equilibrium-form-construction` is already proved for this
law. For centered $f$ and $t>0$, spectral calculus gives

$$
\frac{d}{dt}\|T_t^{\mathrm{eq}}f\|_2^2
=-2\mathcal E(T_t^{\mathrm{eq}}f,T_t^{\mathrm{eq}}f)
\le-\frac2{C_*}\|T_t^{\mathrm{eq}}f\|_2^2.
$$

Integration and strong continuity at zero give
{ref}`(YM.42) <eq-fg-ym-42>`. Stationarity and the Markov property give,
with $A_c=A-\rho A$ and $B_c=B-\rho B$,

$$
\left|\operatorname{Cov}(A(X_0),B(X_t))\right|
=|\langle A_c,T_t^{\mathrm{eq}}B_c\rangle|
\le e^{-t/C_*}\sqrt{\operatorname{Var}_\rho A\,
                              \operatorname{Var}_\rho B}.
$$

All integrals below are justified by this $L^2$ bound. In particular,

$$
\begin{aligned}
\operatorname{Var}\left(T^{-1}\int_0^T A(X_t)dt\right)
&=\frac2{T^2}\int_0^T(T-u)
           \langle A_c,T_u^{\mathrm{eq}}A_c\rangle du\\
&\le\frac{2\operatorname{Var}_\rho A}{T^2}
            \int_0^T(T-u)e^{-u/C_*}du,\\
\int_0^T(T-u)e^{-u/C_*}du
&=C_*T-C_*^2(1-e^{-T/C_*}).
\end{aligned}
$$

This proves {ref}`(YM.Q1) <eq-fg-ym-q1>`. For centered future
polynomials, the boundary-vector construction gives

$$
|(F_c,\tau_tG_c)_{\mathrm{OS},N}|
\le e^{-t/C_*}\|F_c\|_{\mathrm{OS},N}\|G_c\|_{\mathrm{OS},N}.
$$

Every entry here belongs to the common converging polynomial hierarchy.
Passing to that limit, and then using density in the OS completion, proves
the asserted limiting bound. This step uses the construction and convergence
in {prf:ref}`thm-ym-equilibrium-fluctuation-limit`; it is not an input to
its proof. The velocity-only kinetic dissipation is a different form and
is governed by its previously proved hypocoercive estimate.
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

### Limits supplied by the equilibrium construction

:::{div} feynman-prose
An upper fluctuation bound prevents a field from escaping to infinity. To
obtain a nontrivial limit we also need some fluctuation to survive. In the
family used here, the density of the entire joint law stays between fixed
multiples of its product reference density. This gives a variance lower
bound independent of population size. The bound concerns the whole joint
tilt, so its constant must remain uniform as particles are added.

The energy estimate then prevents that variance from disappearing immediately
in equilibrium time. A short time average retains a positive reflected norm,
which survives the common subsequential limit. The same correlation limits
construct the limiting transfer and preserve its gap. This proves a
nontrivial field hierarchy for the stated family. Product independence will
identify a particular hierarchy explicitly; general bounded tilts retain
their own subsequential laws.
:::

:::{prf:theorem} Nontrivial equilibrium fluctuation hierarchy for the established bounded-tilt family
:label: thm-ym-equilibrium-fluctuation-limit

Use the product or bounded whole-joint tilt family already proved in
{prf:ref}`cor-n-uniform-lsi`:

(eq-fg-ym-e9)=
$$
\rho_N=Z_N^{-1}e^{-B_N}m_U^{\otimes N},\qquad
\operatorname{osc}B_N\le B_*.
\tag{YM.E9}
$$

Here $B_N=0$ includes the product family. Use its actual $B_N$ and the
equilibrium form {ref}`(YM.E1) <eq-fg-ym-e1>`--{ref}`(YM.E2) <eq-fg-ym-e2>`, with $C_*=e^{B_*}C_0$ as proved
upstream. The stationary equilibrium fluctuation fields {ref}`(YM.F1) <eq-fg-ym-f1>` have a
common subsequential continuum hierarchy that is tempered, symmetric,
invariant under time translation, and reflection positive. Its OS transfer
semigroup has a unique vacuum and gap at least $1/C_*$ on its nonvacuum
observable space. This space is nonzero; {prf:ref}`thm-ym-fluctuation-os-infinite-dimension`
further proves that it is infinite dimensional.

The limit is taken in population size at fixed algorithmic parameters and
on the actual phase-space test coordinates. Spatial covariance is inherited
only for transformations that preserve the same law and form. The theorem
does not identify the equilibrium time with the recorded kinetic time.
:::

:::{prf:proof}
**The upper estimates and a lower estimate from the same law.** The
established density comparison gives

$$
e^{-B_*}\le\frac{d\rho_N}{dm_U^{\otimes N}}\le e^{B_*}.
$$

For a smooth compactly supported nonconstant $\varphi$ of one particle,
put $F_N=N^{-1}\sum_i\varphi(z_i)$ and
$Y_N=\sqrt N(F_N-\rho_NF_N)$. Minimizing over real constants gives

(eq-fg-ym-e10)=
$$
\begin{aligned}
\rho_N(Y_N^2)
&=N\inf_b\rho_N[(F_N-b)^2]\\
&\ge Ne^{-B_*}\inf_bm_U^{\otimes N}[(F_N-b)^2]
=e^{-B_*}\operatorname{Var}_{m_U}\varphi=:c_\varphi>0.
\end{aligned}
\tag{YM.E10}
$$

The product variance in the last line is computed by expanding the sum:
off-diagonal centered terms have expectation zero and the $N$ diagonal
terms each equal $\operatorname{Var}_{m_U}\varphi/N^2$.
The reference density is positive, so this variance is positive for the
chosen test. The original smooth function is an admissible approximation
in {ref}`(YM.E2) <eq-fg-ym-e2>`, and therefore

(eq-fg-ym-e11)=
$$
\mathcal E_N(Y_N,Y_N)
\le\frac1N\sum_i\rho_N|\nabla\varphi(z_i)|^2
\le\|\nabla\varphi\|_\infty^2=:b_\varphi.
\tag{YM.E11}
$$

For $C_N(\sigma)=\langle Y_N,T_{N,\sigma}^{\mathrm{eq}}Y_N\rangle$,
spectral calculus and $1-e^{-\sigma\lambda}\le\sigma\lambda$ give

(eq-fg-ym-e12)=
$$
C_N(\sigma)\ge c_\varphi-\sigma b_\varphi.
\tag{YM.E12}
$$

Choose $\delta=c_\varphi/(2b_\varphi)$ and a smooth nonnegative
time test $a$ supported in $(0,\delta/2)$ with $\int a=1$.
For the future field $F_N=\mathcal Z_N(a\otimes\varphi)$,

(eq-fg-ym-e13)=
$$
(F_N,F_N)_{\mathrm{OS}}
=\iint a(s)a(t)C_N(s+t)dsdt\ge c_\varphi/2.
\tag{YM.E13}
$$

It is centered. This lower bound survives time smearing and reflection;
it is stronger than a nonzero one-time variance alone.

**Common convergence and reflection.** Apply
{prf:ref}`thm-ym-spacetime-fluctuation-compactness` to the stationary
equilibrium processes from {prf:ref}`thm-ym-equilibrium-hierarchy`.
One subsequence gives weak convergence of the random distributions and
convergence of every polynomial correlation. Reflection and time
translation act continuously on the Schwartz test space. The law at each
$N$ has the corresponding identities, and every reflected polynomial
product is among those converging correlations. Thus the limit has all
the stated symmetries and reflection positivity. Bound {ref}`(YM.E13) <eq-fg-ym-e13>` gives a
nonzero centered vector in its OS quotient.

**The limiting transfer and its gap.** For any future polynomial $F$ let
$F_c=F-\mathbb E F$. At finite $N$, {ref}`(YM.E8) <eq-fg-ym-e8>` yields

(eq-fg-ym-e14)=
$$
\begin{aligned}
\|\tau_\sigma F_c\|_{\mathrm{OS},N}^2
&\le e^{-2\sigma/C_*}\|F_c\|_{\mathrm{OS},N}^2,\\
(F,\tau_\sigma G)_{\mathrm{OS},N}
&=(\tau_\sigma F,G)_{\mathrm{OS},N},\qquad
(F,\tau_\sigma F)_{\mathrm{OS},N}\ge0.
\end{aligned}
\tag{YM.E14}
$$

All these are finite combinations of the common hierarchy and pass to the
limit. They make positive time translation a well-defined self-adjoint
positive contraction on the OS quotient, with the semigroup law. To verify
strong continuity, translate each time test in a polynomial. Its difference
from the original polynomial is a finite telescoping sum with one
translated-minus-original factor. Hölder's inequality and {ref}`(YM.F2) <eq-fg-ym-f2>` bound
each term in $L^2$ by a constant times the Schwartz seminorm of that test
difference, which tends to zero. The absolute reflected pairing is bounded
by the corresponding $L^2$ norm squared, using Cauchy--Schwarz and time
reflection invariance. Thus translation is strongly continuous on the
polynomial quotient and, by contraction and density, on its completion.

The resulting self-adjoint generator $H_\infty^{\mathrm{eq}}\ge0$
fixes the normalized vector $[1]$. The first inequality in {ref}`(YM.E14) <eq-fg-ym-e14>`
gives its centered norm bound $e^{-\sigma/C_*}$, and hence its gap.
If a vector is fixed, subtract its vacuum component and let $\sigma$
tend to infinity in that bound; its centered part is zero. This proves
vacuum uniqueness. No independently assumed convergence of embedded
Hamiltonians is used: the converging correlation forms construct the
Hilbert space and its transfer together.
:::

:::{div} feynman-prose
In the product case, each particle contributes an independent copy of an
entire equilibrium trajectory. Apply the central-limit calculation to a
time-smeared measurement of one such trajectory. Its variance is determined
by the one-particle transfer, which gives the covariance below. The resulting
Gaussian law fixes every higher correlation through pairings and makes the
whole population sequence converge.

Independence is doing the identification here. A bounded joint tilt provides
the preceding survival and compactness estimates, but those estimates do
not supply independence of the particle trajectories or force the limiting
fluctuation law to be Gaussian.
:::

:::{prf:corollary} Identified Gaussian hierarchy in the established product case
:label: cor-ym-product-equilibrium-hierarchy

For $B_N=0$ in {ref}`(YM.E9) <eq-fg-ym-e9>`, the whole equilibrium fluctuation sequence
converges. Let $T_\sigma^{(1)}$ be the equilibrium transfer of the already
specified one-particle law $m_U$. Its limiting centered Gaussian field has
covariance

(eq-fg-ym-e15)=
$$
\mathcal C(\varphi,\psi)
=\iint\left\langle\varphi(s,\cdot)-m_U\varphi(s,\cdot),
 T_{|t-s|}^{(1)}[\psi(t,\cdot)-m_U\psi(t,\cdot)]
                 \right\rangle_{m_U}dsdt.
\tag{YM.E15}
$$

Odd moments vanish and each even moment is the sum, over pairings of its
tests, of products of {ref}`(YM.E15) <eq-fg-ym-e15>`. Its nontriviality, reflection positivity,
and uniform transfer gap are those just constructed.
:::

:::{prf:proof}
The product reference has its strictly positive smooth kinetic density.
Its gradient form is closable: if $f_j\to0$ and $\nabla f_j\to v$
in their weighted $L^2$ spaces, integration by parts against compactly
supported smooth vector tests shows $v=0$; the logarithmic density
derivative is bounded on each such support. On finite sums of smooth
tensor products the form {ref}`(YM.E1) <eq-fg-ym-e1>` is the sum of the one-particle forms.
Smooth compactly supported functions and their first derivatives can be
approximated by such sums, and spatial cutoffs exhaust the weighted Sobolev
domain. Thus the closed form and its semigroup are the tensor products
of the one-particle construction. This is the energy realization of the
product structure used in {prf:ref}`thm-tensorization`.

Consequently the equilibrium paths of different particles are independent.
For any finite real linear combination of the tests, let
$Y_i=\int[\varphi(t,X_i(t))-m_U\varphi(t,\cdot)]dt$.
These variables are independent, identically distributed, centered, and
have every moment by {ref}`(YM.F2) <eq-fg-ym-f2>` at $N=1$. Taylor's formula gives

$$
\mathbb E e^{iuY_1/\sqrt N}
=1-\frac{u^2\mathbb E Y_1^2}{2N}+R_N,
\qquad
|R_N|\le\frac{|u|^3\mathbb E|Y_1|^3}{6N^{3/2}}.
$$

Taking its $N$th power gives
$\mathbb E e^{iuN^{-1/2}\sum_iY_i}
\to\exp[-u^2\mathbb E Y_1^2/2]$. Stationarity and the one-particle
Markov property compute $\mathbb E Y_1^2$ and its polarization as
{ref}`(YM.E15) <eq-fg-ym-e15>`. This determines every finite-dimensional distribution. The
distributional tightness already proved makes all subsequential laws equal,
so the whole sequence converges. Differentiating the Gaussian generating
function, justified also by the uniform moment bounds, gives the pairing
formula for its hierarchy.

For a nonzero bounded tilt $B_N$, the same independence step is unavailable.
Its proven conclusion is {prf:ref}`thm-ym-equilibrium-fluctuation-limit`
with the actual tilt retained. The product calculation identifies a
specific family already present in the framework; it does not substitute
a product law for the selected swarm law.
:::

:::{div} feynman-prose
We can identify the bulk empirical law even when the fluctuation law remains
undetermined. Under the product reference, the mean-square error of a bounded
empirical average is proportional to $1/N$. The bounded whole-joint density
ratio can increase that error by only a fixed factor. The error therefore
still vanishes under the tilted law. If that law is exchangeable, sampling
a fixed number of distinct particle labels also gives the product marginal
in the limit.

Now look more closely at the fluctuations. A bias of order $1/\sqrt N$
disappears from an empirical average but survives after multiplication by
$\sqrt N$. The centering calculation below keeps this correction explicitly.
Thus identifying the limiting bulk measure does not replace the finite
joint laws or determine their fluctuation covariances and equilibrium
transitions.
:::

:::{prf:theorem} Empirical and fixed-marginal identification within the bounded-tilt family
:label: thm-ym-bounded-tilt-bulk-identification

For the same family {ref}`(YM.E9) <eq-fg-ym-e9>`, write $m=m_U$ and
$L_N=N^{-1}\sum_i\delta_{z_i}$. Every bounded real test $\varphi$ satisfies

(eq-fg-ym-q2)=
$$
\mathbb E_{\rho_N}|L_N\varphi-m\varphi|^2
\le\frac{e^{B_*}}{N}\operatorname{Var}_m\varphi.
\tag{YM.Q2}
$$

Hence the empirical law tends to $m$ in probability in the weak topology
on the finite-dimensional particle state space. For the exchangeable
members of this family, every fixed marginal tends to $m^{\otimes k}$.
Whenever the upstream stationary-chaos theorem is applied to this same
sequence of laws, its one-particle limit $\mu_*$ is therefore $m$.
:::

:::{prf:proof}
Let $r_N=d\rho_N/dm^{\otimes N}$. The established bound
$e^{-B_*}\le r_N\le e^{B_*}$ gives

$$
\begin{aligned}
\mathbb E_{\rho_N}|L_N\varphi-m\varphi|^2
&\le e^{B_*}\mathbb E_{m^{\otimes N}}
 \left|\frac1N\sum_i(\varphi(z_i)-m\varphi)\right|^2\\
&=\frac{e^{B_*}}{N^2}\sum_i\operatorname{Var}_m\varphi.
\end{aligned}
$$

The mixed terms vanish by product independence. This proves
{ref}`(YM.Q2) <eq-fg-ym-q2>` without requiring independence under $\rho_N$.
For empirical tightness choose a compact $K$ with $m(K^c)$ small. The same
density comparison gives
$\mathbb E_{\rho_N}L_N(K^c)\le e^{B_*}m(K^c)$, and Markov's inequality
makes the mass outside $K$ small in probability, uniformly in $N$.
More explicitly, choose compacts $K_j$ with
$e^{B_*}m(K_j^c)\le\varepsilon 2^{-j}\eta_j$, where $\eta_j\downarrow0$.
Then the probability that some $L_N(K_j^c)>\eta_j$ is at most
$\varepsilon$. The probability measures satisfying all these constraints
form a tight set; its weak closure is compact. Thus the laws of $L_N$ are
tight. A countable convergence-determining family of bounded continuous
tests, together with {ref}`(YM.Q2) <eq-fg-ym-q2>`, identifies every
subsequential limit as the deterministic measure $m$. Convergence in law
to this constant is convergence in probability.

For bounded tests $f_1,\ldots,f_k$, set $K_f=\prod_j\|f_j\|_\infty$.
Sampling $k$ indices independently with replacement gives
$\prod_jL_Nf_j$. Conditional on distinct indices the average is
$D_{N,k}=(N)_k^{-1}\sum_{i_1,\ldots,i_k\text{ distinct}}
\prod_j f_j(z_{i_j})$, where $(N)_k=N(N-1)\cdots(N-k+1)$.
The probability of any collision is at most $k(k-1)/(2N)$, so

(eq-fg-ym-q3)=
$$
\left|\mathbb E_{\rho_N}\prod_jL_Nf_j
       -\mathbb E_{\rho_N}D_{N,k}\right|
\le\frac{k(k-1)}{N}K_f.
\tag{YM.Q3}
$$

Exchangeability identifies the second expectation with
$\rho_N^{(k)}(f_1\otimes\cdots\otimes f_k)$.
A telescoping expansion and {ref}`(YM.Q2) <eq-fg-ym-q2>` give

$$
\mathbb E_{\rho_N}\left|\prod_jL_Nf_j-\prod_jmf_j\right|
\le\sum_{j=1}^k\left(\prod_{\ell\ne j}\|f_\ell\|_\infty\right)
 \sqrt{\frac{e^{B_*}\operatorname{Var}_m f_j}{N}}
\longrightarrow0.
$$

The marginal density comparison also gives
$\rho_N^{(k)}(A)\le e^{B_*}m^{\otimes k}(A)$, hence tightness.
Bounded continuous product tests identify the unique weak limit as
$m^{\otimes k}$. Uniqueness of a weak limit proves the claimed
identification with $\mu_*^{\otimes k}$ when the stationary-chaos result
concerns these same laws.

For later use the reference-centered fluctuation and the actual-centered
one differ by the deterministic quantity

(eq-fg-ym-q4)=
$$
\begin{aligned}
G_N&=\sqrt N(L_N\varphi-m\varphi),\qquad
Y_N=G_N-\beta_N,\\
\beta_N&=\mathbb E_{m^{\otimes N}}[(r_N-1)G_N],\qquad
|\beta_N|\le(e^{B_*}-1)\sqrt{\operatorname{Var}_m\varphi}.
\end{aligned}
\tag{YM.Q4}
$$

The bound follows from Cauchy--Schwarz and
$\mathbb E_{m^{\otimes N}}G_N^2=\operatorname{Var}_m\varphi$.
This calculation identifies the bulk law but retains the possible
order-one fluctuation-centering correction. It neither replaces $\rho_N$
by a product law nor identifies its equilibrium transitions.
:::

:::{div} feynman-prose
A single nonzero fluctuation establishes one direction beyond the vacuum.
To prove infinite dimension, we need arbitrarily many independent directions.
Try the powers of one field measurement, from the constant through degree
$q$. Their Gram matrix tests whether any nonzero polynomial combination
has zero reflected norm.

The product reference supplies a positive polynomial Gram matrix in the
large-population limit: a nonzero polynomial cannot vanish almost everywhere
under a Gaussian with positive variance. The bounded density comparison
keeps a positive lower bound for the tilted laws, including their bounded
centering shifts. A sufficiently short future time average then stays close
enough to the equal-time measurement to preserve that lower bound in the
reflection form.

All entries pass through the already constructed common hierarchy. For
every finite $q$, this produces $q+1$ independent vectors. The time window
may shrink with $q$; a single window need not work for all degrees. The
Gaussian is used to bound the reference Gram matrix, without identifying
the tilted fluctuation law as Gaussian.
:::

:::{prf:theorem} Infinite dimension of the constructed fluctuation OS space
:label: thm-ym-fluctuation-os-infinite-dimension

Every common subsequential hierarchy constructed in
{prf:ref}`thm-ym-equilibrium-fluctuation-limit` has an infinite-dimensional
OS Hilbert space. For each integer $q\ge1$, a single smooth future test
$a_q\otimes\varphi$ supplies $q+1$ linearly independent vectors
$[1],[\mathcal Z(a_q\otimes\varphi)],\ldots,
[\mathcal Z(a_q\otimes\varphi)^q]$.
:::

:::{prf:proof}
Fix the real smooth compactly supported test $\varphi$ used in
{ref}`(YM.E10) <eq-fg-ym-e10>` and put
$v=\operatorname{Var}_m\varphi>0$, $b=\|\nabla\varphi\|_\infty^2>0$.
Use $G_N,Y_N,\beta_N$ from {ref}`(YM.Q4) <eq-fg-ym-q4>` and write
$B_0=(e^{B_*}-1)\sqrt v$.

**A positive polynomial Gram matrix at one time.** Under the product
reference, expansion of each integer moment gives

$$
\mathbb E G_N^r
=N^{-r/2}\sum_{i_1,\ldots,i_r}
 \mathbb E\prod_{j=1}^r\bigl(\varphi(z_{i_j})-m\varphi\bigr).
$$

A term containing an index exactly once vanishes. For $r=2\ell$, the
terms with $\ell$ distinct indices all repeated twice contribute
$(N)_\ell N^{-\ell}(2\ell)!v^\ell/(2^\ell\ell!)$.
Every other nonzero term has at most $\ell-1$ distinct indices and is
$O(N^{-1})$ after normalization; boundedness of $\varphi$ bounds its
coefficient. For $r=2\ell+1$, at most $\ell$ distinct indices occur,
so the entire moment is $O(N^{-1/2})$. Thus all moments tend to those
of a real Gaussian $G$ with variance $v$.

For $a\in\mathbb C^{q+1}$ put $p_a(x)=\sum_{j=0}^qa_jx^j$ and define

(eq-fg-ym-q5)=
$$
\mu_q=\min_{\|a\|_2=1,\,|\beta|\le B_0}
       \mathbb E|p_a(G-\beta)|^2>0,\qquad
\lambda_q=\tfrac12e^{-B_*}\mu_q.
\tag{YM.Q5}
$$

The minimum exists by compactness and continuity. A zero minimum would
make a nonzero polynomial vanish almost everywhere for a Gaussian of
positive variance, which is impossible since its density is strictly
positive and a nonzero polynomial has finitely many real zeros.
Convergence of the finitely many moments up to $2q$ is uniform in
$\|a\|_2=1$ and $|\beta|\le B_0$: expanding the polynomial gives a
finite sum of moment errors with uniformly bounded coefficients.
Consequently, for all sufficiently large $N$,

$$
\mathbb E_{\rho_N}|p_a(Y_N)|^2
\ge e^{-B_*}\mathbb E_{m^{\otimes N}}|p_a(G_N-\beta_N)|^2
\ge\lambda_q\|a\|_2^2.
$$

**Moving the Gram matrix into strictly positive times.** The LSI moment
bound already used in {ref}`(YM.F2) <eq-fg-ym-f2>` gives, for $r>0$,

$$
\|Y_N(t)\|_r\le M_r
:=\left[2(2C_*)^{r/2}\Gamma(1+r/2)\right]^{1/r}\sqrt b.
$$

Stationarity, the spectral bound $1-e^{-t\lambda}\le t\lambda$, and
{ref}`(YM.E11) <eq-fg-ym-e11>` give the increment estimate

(eq-fg-ym-q6)=
$$
\mathbb E|Y_N(t)-Y_N(0)|^2
=2\langle Y_N,(I-T_{N,|t|}^{\mathrm{eq}})Y_N\rangle
\le2b|t|.
\tag{YM.Q6}
$$

Choose $a_\delta\in C_c^\infty(0,\delta)$ nonnegative with integral one
and set $X_N^\pm=\int a_\delta(t)Y_N(\pm t)dt$.
Jensen's inequality yields
$\|X_N^\pm\|_r\le M_r$ for $r\ge1$ and
$\|X_N^\pm-Y_N(0)\|_2\le\sqrt{2b\delta}$.
Interpolation between $L^2$ and $L^8$ gives

$$
\|X_N^\pm-Y_N(0)\|_4
\le(2b\delta)^{1/6}(2M_8)^{2/3}.
$$

Indeed $1/4=(1/3)/2+(2/3)/8$ and the $L^8$ norm of this difference
is at most $2M_8$. For $j\ge1$, factor the difference of powers and
apply Hölder to each term:

$$
\|(X_N^\pm)^j-Y_N(0)^j\|_2
\le j\|X_N^\pm-Y_N(0)\|_4 M_{4(j-1)}^{j-1},
$$

where the last factor is defined to be $1$ for $j=1$.
Set

$$
\begin{aligned}
U_q&=\left(1+\sum_{j=1}^qM_{2j}^{2j}\right)^{1/2},\\
K_q&=(2b)^{1/6}(2M_8)^{2/3}
 \left(\sum_{j=1}^qj^2M_{4(j-1)}^{2(j-1)}\right)^{1/2}.
\end{aligned}
$$

Cauchy--Schwarz in the coefficient index now gives
$\|p_a(X_N^\pm)\|_2,\|p_a(Y_N(0))\|_2\le U_q\|a\|_2$ and
$\|p_a(X_N^\pm)-p_a(Y_N(0))\|_2
\le K_q\delta^{1/6}\|a\|_2$. Expanding the difference of the two
pairings and applying Cauchy--Schwarz once more gives

(eq-fg-ym-q7)=
$$
\left|\mathbb E\overline{p_a(X_N^-)}p_a(X_N^+)
             -\mathbb E|p_a(Y_N(0))|^2\right|
\le2U_qK_q\delta^{1/6}\|a\|_2^2.
\tag{YM.Q7}
$$

Choose $0<\delta\le(\lambda_q/(4U_qK_q))^6$ and take
$a_q=a_\delta$. Reflection positivity makes the first pairing real, and
{ref}`(YM.Q5) <eq-fg-ym-q5>`--{ref}`(YM.Q7) <eq-fg-ym-q7>` imply

(eq-fg-ym-q8)=
$$
\|p_a(\mathcal Z_N(a_q\otimes\varphi))\|_{\mathrm{OS},N}^2
\ge\tfrac12\lambda_q\|a\|_2^2.
\tag{YM.Q8}
$$

**The same Gram matrix in the limit.** Its $(j,k)$ entry is the
correlation of $j$ reflected copies and $k$ future copies of the fixed
Schwartz test, with $j+k\le2q$. All these entries converge along the
common subsequence already constructed. Therefore
{ref}`(YM.Q8) <eq-fg-ym-q8>` passes to its limiting Gram matrix.
Its rank is $q+1$. This works for every $q$ on the same limiting
hierarchy, although the time test may depend on $q$. Hence the OS space
has arbitrarily large finite-dimensional subspaces and is infinite
dimensional. The Gaussian calculation was used only under the product
reference to bound this Gram matrix; it makes no Gaussian identification
of a bounded-tilt fluctuation limit.
:::

:::{div} feynman-prose
There is another useful limit with a different approximation parameter.
Begin with the mean-field marginal already proved upstream and construct
its equilibrium energy. Then represent observables using finitely many
modes. The mass matrix measures their inner products; the stiffness matrix
measures their energy. Increasing the mode count gives a convergent
approximation to this fixed operator and its bounded-observable correlations.

Here the number of retained particles $k$ is fixed and the mode count $M$
grows. The theorem therefore gives a precise route to approximating the
equilibrium theory of the identified limiting law. Identifying a population
fluctuation covariance for interacting swarms remains the separate dynamical
calculation described above.
:::

:::{prf:theorem} Equilibrium theory on the proved mean-field law and convergent mode approximations
:label: thm-ym-limit-law-galerkin-transfer

Take the continuous limiting law $\rho^{(k)}$ already identified by the
stationary-chaos results and covered by
{prf:ref}`cor-kl-lsi-mean-field-limit`, for fixed $k$. Its smooth-test LSI
has the same $C_*$. Applying {ref}`(YM.E1) <eq-fg-ym-e1>`--{ref}`(YM.E5) <eq-fg-ym-e5>` in these $k$ particle
coordinates constructs its equilibrium operator $H_*^{(k)}$ directly.
For the nondegenerate continuous stationary density $\mu_*$,
$\rho^{(k)}=\mu_*^{\otimes k}$ has nonconstant modes.

Let $V_M$ be nested finite-dimensional spaces generated by the constant
and a countable form-dense set of smooth record observables, and let $P_M$
be their $L^2(\rho^{(k)})$ orthogonal projections. The stiffness and mass
matrices are the actual form and moment integrals

(eq-fg-ym-e16)=
$$
K_{ab}=\mathcal E_*(u_a,u_b),\qquad
G_{ab}=\langle u_a,u_b\rangle_{\rho^{(k)}}.
\tag{YM.E16}
$$

After quotienting zero-norm combinations, these matrices define a
self-adjoint $H_M$ on $V_M$, with constant vacuum and gap at least
$1/C_*$. If $J_M:V_M\hookrightarrow L^2(\rho^{(k)})$ is inclusion,
then for every $\sigma\ge0$,

(eq-fg-ym-e17)=
$$
J_Me^{-\sigma H_M}J_M^*\longrightarrow e^{-\sigma H_*^{(k)}}
\quad\text{strongly}.
\tag{YM.E17}
$$

Finite products of these transfers and compressed bounded direct
observables converge to the corresponding equilibrium correlations.
This instantiates the embedded-semigroup limit and uniform gap in
{prf:ref}`thm-mass-gap-rg-fixed-point` for the mode cutoff $M$.
:::

:::{prf:proof}
**The identified limiting measure.** For a smooth bounded function of the
first $k$ particles, every other gradient in the $N$-particle LSI vanishes.
Weak convergence of the fixed marginal passes its entropy and gradient
integrals to the identified law, exactly as proved in
{prf:ref}`cor-kl-lsi-mean-field-limit`. The recovery and entropy argument
of {ref}`(YM.E2) <eq-fg-ym-e2>`--{ref}`(YM.E3) <eq-fg-ym-e3>` extends it to the closed energy even before a separate
closability statement. This constructs $H_*^{(k)}$ from the existing
limit. A continuous probability density is not a point mass: choose two
disjoint compact neighborhoods of positive probability and a smooth bounded
function equal to one on one and zero on the other. Its variance is
positive, providing a nonzero mode. The equilibrium and CAR spectral
formulas therefore yield nonzero correlations.

**Choice of a form core.** The graph used in {ref}`(YM.E2) <eq-fg-ym-e2>` is a subspace of a
separable Hilbert direct sum. Choose a countable dense subset of its
minimum-gradient graph, approximate each pair by the smooth recovery
sequences used there, and enumerate those smooth functions, together with
$1$. Their finite spans are form dense. Each such function is a function
of the already reconstructed particle coordinates, so its values and the
integrals in {ref}`(YM.E16) <eq-fg-ym-e16>` have the exact record representation. The dense core
is retained even when a selected finite set of channel observables spans
only a proper subspace.

**Recovery and lower bound.** Form density gives, for every form-domain
$u$, elements $u_M\in V_M$ with
$\|u_M-u\|_2^2+\mathcal E_*(u_M-u)\to0$. Conversely, a sequence with
bounded $L^2$ norm and bounded form energy is weakly precompact in the
Hilbert form domain. Any weak $L^2$ limit is the same form-domain limit
along a further subsequence, and weak lower semicontinuity gives
$\mathcal E_*(u)\le\liminf_M\mathcal E_*(u_M)$.
These are the recovery and lower-bound calculations for this particular
approximation.

**Resolvent and transfer convergence.** Let
$u=(\lambda+H_*^{(k)})^{-1}f$ and let
$u_M=(\lambda+H_M)^{-1}P_Mf$. Subtract their variational equations
{ref}`(YM.E5) <eq-fg-ym-e5>`. For every $v\in V_M$,
$\lambda\langle u-u_M,v\rangle+\mathcal E_*(u-u_M,v)=0$.
Thus $u_M$ is the orthogonal projection of $u$ in the form inner product
$\lambda\langle\cdot,\cdot\rangle+\mathcal E_*$, and

$$
\lambda\|u-u_M\|_2^2+\mathcal E_*(u-u_M)
\le\inf_{v\in V_M}
 \{\lambda\|u-v\|_2^2+\mathcal E_*(u-v)\}\longrightarrow0.
$$

This proves strong convergence of the embedded resolvents. For fixed
$\sigma>0$, the function
$r\mapsto\exp[-\sigma(r^{-1}-\lambda)]$ on
$0<r\le1/\lambda$, extended by zero at $r=0$, is continuous.
Uniform polynomial approximation with zero constant term and the
resolvent convergence give {ref}`(YM.E17) <eq-fg-ym-e17>`. At $\sigma=0$ the claim is
$P_M\to I$, which follows from core density.

The Poincaré inequality on each $V_M$ gives the same gap, and $1\in V_M$
gives its exact vacuum. A bounded multiplication operator $M_A$ has
$P_MM_AP_M\to M_A$ strongly with norms at most $\|A\|_\infty$.
Telescoping a finite product of uniformly bounded strongly convergent
factors proves convergence of every bounded-insertion correlation. Its
finite-dimensional reflected word uses the adjoint reversed product, so
it also has a nonnegative boundary-vector norm. The compressed operator
need not be a classical Markov kernel; its self-adjoint transfer and
represented observable matrix elements are the finite mode approximation.
:::

:::{div} feynman-prose
The same mass and stiffness matrices now give a finite fermionic simulator.
Remove the constant mode, form antisymmetric combinations of the retained
modes, and add their one-mode energies. The empty sector supplies the vacuum;
every occupied mode costs at least the established gap.

To approximate a chosen correlation, project each insertion mode into the
retained space and use its finite transfer. Mode convergence then passes
through each finite exterior sector and each finite operator word. The
completely positive regression formula expresses its correlations through
these same transfer products, so those calculations converge as well. Thus
increasing the mode cutoff approximates the equilibrium fermionic theory
while preserving a common gap. The finite Hamiltonians use the existing
mass and stiffness integrals; their time parameter remains the equilibrium
energy time of the fixed limiting law.
:::

:::{prf:corollary} Convergent fermionic simulator on the established equilibrium mode spaces
:label: cor-ym-galerkin-car-convergence

Use the fixed limiting law and the actual nested mode spaces $V_M$ of
{prf:ref}`thm-ym-limit-law-galerkin-transfer`. Remove the constant
mode, putting $E_M=V_M\cap1^\perp$ and
$\mathcal H_*=L^2_0(\rho^{(k)})$. The restriction $h_M$ of $H_M$
to $E_M$ gives the finite fermionic Hamiltonian
$\mathbb H_M=d\Gamma_-(h_M)$. The isometric construction in
{prf:ref}`thm-lqft-record-car-channel`, applied to the contractions
$e^{-\sigma h_M}$, gives its completely positive CAR evolution.
These finite theories have their vacuum and gap at least $1/C_*$.

Let $j_M:E_M\hookrightarrow\mathcal H_*$ and
$h_*=H_*^{(k)}|_{\mathcal H_*}$. Then

(eq-fg-ym-p3)=
$$
\Gamma_-(j_M)e^{-\sigma\mathbb H_M}\Gamma_-(j_M)^*
\longrightarrow e^{-\sigma d\Gamma_-(h_*)}
\quad\text{strongly for every }\sigma\ge0.
\tag{YM.P3}
$$

For every finite CAR word, use the projected modes $j_M^*f$ in
its finite theory. All its finite-time vacuum matrix elements and
nested completely positive regression expressions converge to those
of the limiting equilibrium CAR theory. Its matrices are computed
from the already specified mass and stiffness integrals
{ref}`(YM.E16) <eq-fg-ym-e16>`.
:::

:::{prf:proof}
The constant mode belongs to every $V_M$ and is annihilated by $H_M$,
so its orthogonal complement is invariant. Centering
{ref}`(YM.E17) <eq-fg-ym-e17>` gives the strong convergence of
contractions

$$
C_M(\sigma):=j_Me^{-\sigma h_M}j_M^*
\longrightarrow C_\infty(\sigma):=e^{-\sigma h_*}.
$$

On a fixed decomposable $r$-wedge, telescope the difference between
$\bigwedge_{\ell=1}^r C_M(\sigma)f_\ell$ and
$\bigwedge_{\ell=1}^r C_\infty(\sigma)f_\ell$.
Each term tends to zero because one factor converges strongly and
all others have uniformly bounded norms. Density of decomposable
wedges gives strong convergence on each sector. Truncate the
sector sum of an arbitrary Fock vector; the common contraction
bound controls its remaining tail. This proves
{ref}`(YM.P3) <eq-fg-ym-p3>`, since its left side is
$\Gamma_-(C_M(\sigma))$. At zero time the same proof uses
$j_Mj_M^*\to I$.

Projected insertions on the full mode space satisfy
$\|a^\dagger(j_Mj_M^*f)-a^\dagger(f)\|
=\|j_Mj_M^*f-f\|\to0$, and the same holds for annihilation.
Finite products of these insertions and the uniformly bounded
strongly convergent transfers therefore converge on every vector.
They leave the finite-mode Fock space invariant, so their vacuum
matrix elements there are precisely the finite simulator's matrix
elements. Formula {ref}`(LQ.C4) <eq-fg-lq-c4>` identifies its
regression expressions with these same products, proving their
convergence as well.

Finally the Poincaré inequality on $E_M$ gives
$h_M\ge C_*^{-1}I$. The existing sector-sum calculation gives
energy at least $r/C_*$ on each nonvacuum $r$-sector.
This proves the common gap and unique vacuum. The construction
uses the mode cutoff of the proved limiting law. Its conversion
to numerical matrices retains an independent basis, its mass
matrix, and the corresponding generalized eigenvalue problem;
entrywise sampling errors remain those of the actual estimators.
:::


:::{prf:remark} Which limits have been identified
:label: rem-ym-identified-equilibrium-limits

The population limit in {prf:ref}`thm-ym-equilibrium-fluctuation-limit`
uses the actual bounded-tilt laws and constructs a nontrivial subsequential
hierarchy with a uniform transfer gap. The product case in
{prf:ref}`cor-ym-product-equilibrium-hierarchy` identifies its entire
sequence and covariance. The limiting-law construction in
{prf:ref}`thm-ym-limit-law-galerkin-transfer` uses the already identified
mean-field marginal and proves convergence of its energy-mode approximation.
Its index $M$ is a mode cutoff, not the swarm population $N$ or a geometric
mesh size. The matrices {ref}`(YM.E16) <eq-fg-ym-e16>` specify the mathematical inputs of a
transfer simulator; convergence of estimators of their entries uses the
applicable sampling and channel estimates already proved in the volume.

For interacting $\rho_N$, convergence of fixed marginals alone does not
identify the transitions $T_{N,\sigma}^{\mathrm{eq}}$ or the covariance
of its fluctuation field. Nor does a mode approximation of the limiting
law establish that identification. Geometric reconstruction uses the
separate bandwidth and quadrature estimates of
{prf:ref}`lem-lqft-energy-sampling` and the continuum discharge chapter.
Each resulting limit retains its confining envelope and its own scale
dependence. An independent change of volume, potential, or algorithmic
timestep is not part of {ref}`(YM.E17) <eq-fg-ym-e17>`.
:::


### 12.3. Continuum limits established by the framework

:::{div} feynman-prose
We can now collect the limits that have actually been constructed. Increasing
the population in the bounded-tilt family gives a common fluctuation
hierarchy, an infinite-dimensional OS space, and a gapped equilibrium
transfer. Product independence identifies the entire Gaussian sequence.
Increasing a mode cutoff on an already identified limiting law instead
approximates that law's equilibrium operator and its fermionic correlations.

The bulk identification connects these constructions when they concern
the same exchangeable bounded-tilt laws. It fixes the limiting particle
measure, while each fluctuation field retains its joint time evolution.
In each passage, the correlations, positivity, and operator bounds belong
to the same construction. The scope statements following the theorem
specify which geometric scales and spacetime identifications remain
separate from these population and mode limits.
:::

:::{prf:theorem} Continuum identification for the constructed field families
:label: thm-infinite-volume-limit

The preceding constructions supply the following limits without a separate
continuum assumption.

1. For the established bounded whole-joint tilt family
   {ref}`(YM.E9) <eq-fg-ym-e9>`, the population fluctuation fields have
   a common subsequential, all-order tempered correlation hierarchy.
   It is symmetric, stationary in equilibrium time, and reflection
   positive. Its strongly continuous OS transfer has a unique vacuum,
   an infinite-dimensional Hilbert space, and gap at least $1/C_*$.
2. For the product member of that family, the whole sequence converges
   to the Gaussian hierarchy with covariance
   {ref}`(YM.E15) <eq-fg-ym-e15>`.
3. On each already identified fixed-marginal mean-field law, the inherited
   LSI and the form approximations of
   {prf:ref}`thm-ym-limit-law-galerkin-transfer` give strong convergence
   of the embedded equilibrium transfers and their bounded-observable
   correlations. The centered fermionic lift and its CAR correlations
   converge as proved in {prf:ref}`cor-ym-galerkin-car-convergence`.

Within the exchangeable bounded-tilt family, the fixed-marginal law in
item 3 is $m_U^{\otimes k}$ by
{prf:ref}`thm-ym-bounded-tilt-bulk-identification`. This identifies the
bulk law shared by those constructions; each fluctuation transfer still
uses its own equilibrium joint dynamics.
:::

:::{prf:proof}
For item 1, {prf:ref}`thm-ym-spacetime-fluctuation-compactness` supplies
a single distributional subsequence and uniform moments at every order.
For fixed tests $f_1,\ldots,f_n$, Hölder gives

$$
\sup_N\mathbb E\left|\prod_{j=1}^n\mathcal Z_N(f_j)\right|^2
\le\prod_{j=1}^n
 \left(\sup_N\mathbb E|\mathcal Z_N(f_j)|^{2n}\right)^{1/n}<\infty.
$$

The products are therefore uniformly integrable. Their convergence in
law under the continuous test pairings implies convergence of their
expectations, which are precisely $S_n(f_1,\ldots,f_n)$. The common
test-seminorm bound in {ref}`(YM.F3) <eq-fg-ym-f3>` passes to these
limits and gives their tempered extensions. For any fixed future
polynomials $F_1,\ldots,F_r$ and coefficients $c_1,\ldots,c_r$,

$$
\sum_{i,j}\overline{c_i}c_j
 \mathbb E[\overline{F_i(\theta\mathcal Z)}F_j(\mathcal Z)]
=\lim_N\sum_{i,j}\overline{c_i}c_j
 \mathbb E[\overline{F_i(\theta\mathcal Z_N)}F_j(\mathcal Z_N)]
\ge0.
$$

This proves reflection positivity on the same limiting hierarchy.
Permutation and time-translation identities pass through those same
entries. Equation {ref}`(YM.E14) <eq-fg-ym-e14>` constructs its
strongly continuous transfer and proves vacuum uniqueness and the gap;
{ref}`(YM.Q8) <eq-fg-ym-q8>` proves infinite dimension.
The limiting centered matrix elements obey the explicit clustering bound
of {prf:ref}`thm-os-os3-fg`.

For item 2, the characteristic-function calculation in
{prf:ref}`cor-ym-product-equilibrium-hierarchy` identifies every finite
collection of test pairings. Tightness and this unique identification
force full-sequence convergence. The same uniform integrability argument
gives every moment, with the Gaussian pairing formula stated there.

For item 3, the upstream fixed-marginal limit first inherits its LSI;
the form recovery and resolvent calculation then yield
{ref}`(YM.E17) <eq-fg-ym-e17>`. To see explicitly why bounded
insertions also converge, denote the embedded transfers by $S_M(t)$,
with $S_M(t)\to S(t)$ strongly, and let
$B_{j,M}=P_MB_jP_M$ for fixed bounded $B_j$. The projections tend
strongly to the identity, so $B_{j,M}\to B_j$ strongly with
$\|B_{j,M}\|\le\|B_j\|$. For any finite product of these factors,
write $A_{j,M}$ for a transfer or insertion and $A_j$ for its limit.
The exact telescoping identity is

$$
\left(\prod_{j=1}^rA_{j,M}-\prod_{j=1}^rA_j\right)u
=\sum_{j=1}^r\left(\prod_{i<j}A_{i,M}\right)
 (A_{j,M}-A_j)\left(\prod_{i>j}A_i\right)u.
$$

Each difference acts on a fixed vector and tends to zero; the preceding
factors are uniformly bounded. Taking vacuum matrix elements proves
convergence of the finite correlation words. The Fock lift in
{ref}`(YM.P3) <eq-fg-ym-p3>` supplies the same strong convergence and
bounds for CAR words, so this calculation applies there as well.
Finally {ref}`(YM.Q2) <eq-fg-ym-q2>`--{ref}`(YM.Q3) <eq-fg-ym-q3>`
identify the common bulk law when the bounded-tilt and stationary-chaos
constructions concern the same exchangeable sequence. None of these
steps uses its eventual limiting identity to establish an upstream LSI.
:::

:::{prf:remark} Scales retained by the continuum construction
:label: rem-ym-continuum-scale-scope

These are limits in population size or energy-mode cutoff, with the
confining envelope and geometric scales fixed. Geometric reconstruction
uses the observable, bandwidth, and quadrature identifications in
{prf:ref}`lem-lqft-energy-sampling` and the continuum discharge chapter.
Full spacetime covariance and relativistic field-domain control must be
established for those reconstructed observables before applying the
relativistic Osterwalder--Schrader reconstruction. They are distinct from
the equilibrium time-transfer reconstruction proved here.

Population-uniform constants do not automatically control an independently
increasing geometric volume. For example, on a circle of length $L$ with
uniform probability, $f(x)=\cos(2\pi x/L)$ satisfies
$\operatorname{Var}f=1/2$ and
$\int|f'|^2=(2\pi/L)^2/2$. Its Poincaré constant is at least
$L^2/(4\pi^2)$. The limits above retain their actual confinement and
therefore do not make this change of volume implicitly.
:::

(sec-ym-physical-correspondence-checks)=
### 12.3.1. Application to physical spacetime observables

:::{prf:proposition} Recorded spacetime readouts and their induced transformations
:label: prop-ym-recorded-physical-transformations

Use the covered execution records of
{prf:ref}`def-fractal-set-record-coverage` and the exact history law of
{prf:ref}`thm-sm-instantiated-record-transition`, with four position
coordinates. A complete record $\omega$
contains each stage label, position $x_n(\omega)\in\mathbb R^4$, and
recorded time $\tau_n(\omega)$. Its embedding and physical projection are

(eq-fg-ym-z81)=
$$
\Xi_\omega(n)=(\tau_n(\omega),x_n(\omega)),\qquad
p_{\mathrm{phys}}\Xi_\omega(n)=x_n(\omega),\qquad
p_{\mathrm{sample}}\Xi_\omega(n)=\tau_n(\omega).
\tag{YM.Z81}
$$

The positions, stage labels, units, and recorded times are retained in the
complete record and its header. This section uses the four-position-coordinate
convention: $x^0$ and $\tau$ are separate stored coordinates.

For a face $P=(n_0,\ldots,n_{\ell_P-1})$ in the configured recorded
face family, recover $x_P=\ell_P^{-1}\sum_{a=0}^{\ell_P-1}x_{n_a}$
and its ordered holonomy $U_P$ from those same records. The boundary
length is three for an interaction triangle and four for a derived outer
plaquette. The configured family is retained throughout the calculation;
its triangle and outer-plaquette readouts are distinguished explicitly in
{prf:ref}`prop-ym-native-scalar-face-evaluation`. Fix the representation dimension $r$ and the
coefficients of the Wilson readout already specified in
{prf:ref}`def-wilson-action-ym`. Its localized observable has the exact
finite-measure representation

(eq-fg-ym-z82)=
$$
\begin{gathered}
s_P(\omega)=1-r^{-1}\operatorname{Re}\operatorname{Tr}U_P(\omega),\qquad
\Phi_\omega=\sum_{P\in\mathcal P(\omega)}
                       \beta_Ps_P(\omega)\delta_{x_P(\omega)},\\
\Phi_\omega(f)=\sum_P\beta_Ps_P(\omega)f(x_P(\omega)),\qquad
|\Phi_\omega(f)|\le2\|f\|_\infty\sum_P\beta_P.
\end{gathered}
\tag{YM.Z82}
$$

The coefficients and masks are those of the specified recorded readout;
this formula adds no scaling factor. For the localized $SU(3)$ readout with Wilson convention
$\beta_P=2r/g_3^2$, these are $r=3$ and $\beta_P=6/g_3^2$.
Each finite measure is a tempered distribution. Let $\mathbb P_{N,h}$ here denote
the actual complete execution probability law at the specified finite
horizon, including the selected conditioning when used, and define
$\nu_{N,h}=\Phi_*\mathbb P_{N,h}$. For a bounded cylinder functional
$F(\phi)=B(\phi(f_1),\ldots,\phi(f_m))$, the reconstructed observable is

(eq-fg-ym-z83)=
$$
(\mathcal J_{\mathrm{rec}}F)(\omega)=F(\Phi_\omega),\qquad
\mathbb E_{\nu_{N,h}}\prod_jF_j
=\mathbb E_{\mathbb P_{N,h}}\prod_j\mathcal J_{\mathrm{rec}}F_j,
\qquad
\|\mathcal J_{\mathrm{rec}}F\|_{L^2(\mathbb P_{N,h})}
=\|F\|_{L^2(\nu_{N,h})}.
\tag{YM.Z83}
$$

For the retained joint descriptor $Y=(G,D)$, disintegrate its actual law as
$\mu_{N,h}(dg,dd)=\lambda_{N,h}(dg)\mu_{N,h}^g(dd)$.
Here $\lambda_{N,h}$ is the actual geometry marginal. For every integrable
recorded word $O$, including a reflected product, the same-law identity is

$$
\mathbb E_{\mathbb P_{N,h}}O(G,D)
=\int\lambda_{N,h}(dg)\int\mu_{N,h}^g(dd)\,O(g,d).
$$

This is the unsourced disintegration already used in
{prf:ref}`thm-ym-native-geometry-fiber-action`. A regulator-conditional
correlation is integrated against this marginal before it is used as an
unconditional physical correlation. For a QSD-derived surviving window,
$\mathbb P_{N,h}$ includes exactly the interior and endpoint weights of
{prf:ref}`prop-ym-qsd-history-identification`; the stationary Doob law
retains instead the endpoint factor in {ref}`(SM.K5) <eq-fg-sm-k5>`.

Thus the readout map into the algorithmic probability space is constructed,
with its law and every finite bounded correlation fixed by the record.
It is an isometry onto the closed subspace of $\sigma(\Phi)$-measurable
functions. It is not asserted to be onto the full execution space.

For $g x=Qx+b$, define the geometric pushforward
$\mathsf T_g\phi=g_*\phi$ and its action on cylinder observables by
$\alpha_gF=F\circ\mathsf T_{g^{-1}}$. Then

(eq-fg-ym-z84)=
$$
\begin{gathered}
\mathsf T_g\Phi_\omega
 =\sum_P\beta_Ps_P(\omega)\delta_{g x_P(\omega)},\qquad
\mathsf T_g\mathsf T_h=\mathsf T_{gh},\qquad
\alpha_g\alpha_h=\alpha_{gh},\\
(\mathcal J_{\mathrm{rec}}\alpha_gF)(\omega)
=B\!\left(\sum_P\beta_Ps_P f_1(g^{-1}x_P),\ldots,
          \sum_P\beta_Ps_P f_m(g^{-1}x_P)\right).
\end{gathered}
\tag{YM.Z84}
$$

In particular physical translations, rotations, and reflection are
computed by transforming the test coordinates of the same recorded face
readouts. Every $\tau_n$, companion index, clone mask, and likelihood
factor remains inside the same execution expectation.
:::

:::{prf:proof}
**Recovery and localization.** The complete-record decoding of
{prf:ref}`thm-sm-instantiated-record-transition` recovers the positions,
faces, and specified link payloads. Arithmetic averaging recovers $x_P$;
the ordered link product recovers $U_P$. For a unitary matrix in dimension
$r$, $|\operatorname{Tr}U_P|\le r$, so $0\le s_P\le2$ and the bound in
{ref}`(YM.Z82) <eq-fg-ym-z82>` follows. At fixed finite size the face
list and its weights are finite. The resulting finite atomic measure acts
continuously on Schwartz tests by that bound. Every operation is measurable
in the original record.

**Law and isometry.** The definition of pushforward gives, for any
bounded measurable cylinder $F$,
$\int F\,d\nu_{N,h}=\int F(\Phi_\omega)\,d\mathbb P_{N,h}(\omega)$.
Apply this first to a product and then to $|F|^2$ to obtain
{ref}`(YM.Z83) <eq-fg-ym-z83>`. Simple-function approximation extends
the map to $L^2$. A countable determining family of Schwartz tests
encodes these finite measures, so every $\sigma(\Phi)$-measurable
function factors measurably through $\Phi$. This identifies the image.

**Physical transformations.** Pushforward of each atom gives the first
line of {ref}`(YM.Z84) <eq-fg-ym-z84>`. Since
$(\mathsf T_g\phi)(f)=\phi(f\circ g)$, inserting $g^{-1}$ gives its
last line. Composition of the maps on measures proves their group law;
inverse composition in the observable definition proves the stated
observable group law. The barycenter transforms as
$\ell_P^{-1}\sum_a(Qx_{n_a}+b)=Qx_P+b$.
The ordered face factors stay attached to their transported edges, so
the trace weight is unchanged. These transformations are therefore
computed from the existing embedded face observable, without introducing
an independently distributed field or independently sampled links.

**Reflection evaluated under the actual law.** Put
$\vartheta(x^0,\mathbf x)=(-x^0,\mathbf x)$ and
$\Theta F=\overline{F\circ\mathsf T_\vartheta}$. For cylinders $F_i$ of the scalar
readout, their physical reflected matrix is explicitly

(eq-fg-ym-z85)=
$$
Q^{\mathrm{phys}}_{ij}
=\mathbb E_{\mathbb P_{N,h}}\!\left[
 \overline{F_i\!\left(\sum_P\beta_Ps_P\delta_{\vartheta x_P}\right)}
 F_j\!\left(\sum_P\beta_Ps_P\delta_{x_P}\right)\right].
\tag{YM.Z85}
$$

This substitutes the reconstructed readouts into
{ref}`(YM.Z76) <eq-fg-ym-z76>` and fixes its left-hand side. It
requires no reversal of the sampling record. Localizing an entire recorded
face to a physical half-space additionally retains its vertex masks, as
calculated in {prf:ref}`prop-ym-native-physical-reflection-calculation`.
Similarly, all transformed
correlations in the symmetry check are computed by the last line of
{ref}`(YM.Z84) <eq-fg-ym-z84>`.

**Norm and application.** The transformed observable has norm

(eq-fg-ym-z86)=
$$
\|\alpha_gF\|_{L^2(\nu_{N,h})}^2
=\int|F|^2\,d(\mathsf T_{g^{-1}})_*\nu_{N,h}.
\tag{YM.Z86}
$$

This follows by the same pushforward calculation. Thus invariance of the
actual readout law makes the geometric action unitary; the isometry
{ref}`(YM.Z83) <eq-fg-ym-z83>` alone does not supply invariance.
Likewise {ref}`(YM.Z85) <eq-fg-ym-z85>` defines the actual physical
reflection form, while its positivity is a property to check under that
law. The map $\mathcal J_{\mathrm{rec}}$ is an ordinary probability-space
isometry. The OS isometry $V$ in
{ref}`(YM.Z78) <eq-fg-ym-z78>` additionally preserves the reflected
forms. These formulas identify the concrete algorithmic quantities in
both checks without selecting a new hierarchy to satisfy them.
:::

:::{prf:lemma} Bounds for recorded gauge words and localized face sums
:label: lem-ym-bounded-recorded-gauge-words

Use the selected complete-record law in
{prf:ref}`prop-ym-recorded-physical-transformations`. A normalized unitary
loop readout $u_\gamma=r^{-1}\operatorname{Tr}U_\gamma$, set to zero
on its recorded invalid slots, satisfies $|u_\gamma|\le1$. A valid
Wilson defect $s_P=1-r^{-1}\operatorname{Re}\operatorname{Tr}U_P$
satisfies $0\le s_P\le2$. Thus, for fixed coefficients and bounded
tests, a finite word

$$
F=\sum_{a=1}^m c_a\prod_{b=1}^{d_a}
   [m_{ab}\chi_{ab}u_{\gamma_{ab}}f_{ab}(x_{ab})]
$$

obeys $|F|\le M_F:=\sum_a|c_a|\prod_b\|f_{ab}\|_\infty$.
Using a Wilson defect in place of a normalized trace multiplies the
corresponding factor bound by two. Here $m_{ab}$ and $\chi_{ab}$ are
the actual validity and full-support masks. The same bound holds for
the reflected word, without an invariance assumption on the law. Hence

$$
\left|\overline{F_i(\vartheta Y_{N,h})}F_j(Y_{N,h})\right|
\le M_{F_i}M_{F_j}.
$$

For a countable dictionary of these bounded words, retain their values
and their reflected values among the bounded coordinates of
{prf:ref}`thm-ym-native-fiber-continuum`. Their finite products have
convergent expectations on a common further subsequence of that theorem's
subsequence. This assertion concerns the retained word coordinates.
Identification with a continuous physical field readout additionally requires
its geometric continuity estimates.

For the localized Wilson sum in {ref}`(YM.Z82) <eq-fg-ym-z82>`,
write $m_P\in\{0,1\}$ for the validity mask when its realized faces
are indexed by the configured face slots. Thus its weight is precisely
$m_P\beta_Ps_P$; summing only over valid realized faces suppresses
$m_P$. This introduces no extra volume factor. Define

$$
H_{N,h}(K)=\sum_{P:x_P\in K}
 m_P\frac{\beta_P}{2r}\operatorname{Tr}X_P^2,
\qquad U_P=e^{iX_P},\quad X_P=X_P^*.
$$

Then $|\Phi^+(f)|\le\|f\|_\infty H_{N,h}(\operatorname{supp}f)$.
For a finite family of words of degrees at most $d$ in these localized
sums, with support union $K$, an explicit sufficient bound for uniform
integrability of their reflected products is

$$
\sup_{N,h}\mathbb E
 [1+H_{N,h}(K\cup\vartheta K)]^{2d+\epsilon}<\infty
\quad\text{for some }\epsilon>0.
$$

No such moment bound for the summed quantity is asserted by the bounded-loop
estimate above. In the exact curvature coordinates $X_P=g_3A_PF_P$,
$r=3$ and $\beta_P=6/g_3^2$ give the precise contribution
$A_P^2\operatorname{Tr}F_P^2$ to $H_{N,h}$. With approximate curvature
coordinates, the additional quadratic error is bounded by

$$
\sum_{P:x_P\in K}m_P\frac{\beta_P}{2}
 \eta_P(\|X_P\|_{\mathrm{op}}+\|g_3A_PF_P\|_{\mathrm{op}}),
\qquad \|X_P-g_3A_PF_P\|_{\mathrm{op}}\le\eta_P.
$$

These sums retain the original face count, coefficients, and validity
masks. The normalized geometric weights in {ref}`(YM.Z3) <eq-fg-ym-z3>`
enter the separate readout calculated below.
:::

:::{prf:proof}
The normalized trace bound follows by summing the unit-modulus eigenvalues.
Multiplication and the triangle inequality prove the word bounds, including
their reflected versions. Bounded random variables are uniformly integrable
under every selected probability law, independently of its survival
probability. At finite cutoff the word coordinates are functions of the
covered descriptor. If extra recorded coordinates are needed, first use the
refinement in {prf:ref}`prop-ym-density-and-support` and condition the source
likelihood on that refined descriptor; its projection has the original
conditional density by the tower identity. The conditional likelihood bounds
used in {prf:ref}`thm-ym-native-fiber-continuum` apply to the refinement by
conditional contraction. Retain the countable bounded word coordinates in
that theorem's existing compact-coordinate construction. Products are
continuous functions of finitely many such coordinates, so joint weak
convergence passes their expectations. This also retains the reflected
products on the same further subsequence.

For each eigenvalue $x$ of $X_P$, $0\le1-\cos x\le x^2/2$.
Summation proves $m_P\beta_Ps_P\le
m_P\beta_P\operatorname{Tr}X_P^2/(2r)$ and the localized bound.
A product of two words of degrees at most $d$ is bounded by a fixed
constant times $(1+H)^{2d}$. The stated $(2d+\epsilon)$ moment then
bounds its $1+\epsilon/(2d)$ moment when $d>0$; constants need no
extra estimate. Substitute the specified $SU(3)$ coefficient to obtain
the curvature weight. The trace-difference calculation in
{prf:ref}`lem-ym-recorded-face-action-remainder` gives exactly the
additional quadratic error displayed above.
:::

:::{prf:lemma} Full-face localization and faces crossing the physical cut
:label: lem-ym-physical-cut-face-error

Keep the localized coefficients and masks of
{prf:ref}`lem-ym-bounded-recorded-gauge-words`. Let $\Phi^{\mathrm{bar},+}(f)$
use the barycenter condition $x_P^0>0$ and let $\Phi^+(f)$ use the
full-face condition. For $f$ supported in the positive half-space, put
$K=\operatorname{supp}f$ and
$D_P=\max_{v,w\in P}|x_v-x_w|$. Then

$$
|\Phi^{\mathrm{bar},+}(f)-\Phi^+(f)|
\le\|f\|_\infty
\sum_{P:x_P\in K}m_P\frac{\beta_P}{2r}\operatorname{Tr}X_P^2
 \mathbf1_{\{\min_{v\in P}x_v^0\le0<x_P^0\}}.
$$

For every $\rho>0$, the right-hand side is at most

$$
\|f\|_\infty\left[
 H_{N,h}(K\cap\{0<x^0\le\rho\})
 +\sum_{P:x_P\in K}m_P\frac{\beta_P}{2r}\operatorname{Tr}X_P^2
                       \mathbf1_{\{D_P>\rho\}}\right].
$$

If $\operatorname{dist}(K,\{x^0=0\})=\delta>0$, only faces with
$D_P\ge\delta$ contribute to the first displayed difference. Passage
of the difference to zero in $L^p$ therefore requires the corresponding
weighted large-face estimate, or the two estimates in the second display.
:::

:::{prf:proof}
Subtract the two masks in the same finite sum and apply the preceding
nonnegative face bound. On a contributing face, some vertex has
$x_v^0\le0$ and $x_P^0>0$. Since its barycenter is a convex combination
of its vertices, $0<x_P^0\le|x_P-x_v|\le D_P$. Splitting at
$D_P=\rho$ proves the second bound. If $x_P\in K$ lies at distance
at least $\delta$ from the cut, the same inequality gives
$D_P\ge\delta$. Reflection gives the identical negative-side estimate.
:::

:::{prf:theorem} Common native hierarchy of the normalized geometric gauge readouts
:label: thm-ym-native-normalized-gauge-hierarchy

Use the existing normalized geometric weights of
{prf:ref}`thm-ym-same-record-metric-field`, with that theorem's branch
and validity conventions, including the empty-frame convention. For a fixed recorded gauge mark $b_i$ with $|b_i|\le C_b$,
retain the site's complete face data whenever its support is a face, and set

$$
\Psi_{N,h}^\pm(f)
=\sum_i p_i a_i\chi_i^\pm b_i f(x_i),\qquad
p_i=\frac{b_i^{\mathrm{vol}}\sqrt{\det g_i}}
 {\sum_j b_j^{\mathrm{vol}}\sqrt{\det g_j}},
\qquad \sum_i p_i\le1.
$$

Here $b_i^{\mathrm{vol}}$ is the retained finite Voronoi volume times its
existing site mask, $a_i$ is the gauge-readout validity mask, and $b_i$ is
the gauge mark, not a volume. The point $x_i$ is that readout's recorded
localization point. The mask $\chi_i^+$ requires every vertex of its
recorded face to have $x^0>0$, and $\chi_i^-$ uses $x^0<0$.
Normalized loop traces have $C_b=1$ and Wilson defects have $C_b=2$.
Every finite polynomial word $F$ in these readouts has a deterministic
bound $M_F$ independent of $N,h$. In particular,

$$
|\Psi_{N,h}^\pm(f)|\le C_b\|f\|_\infty,\qquad
|\Psi_{N,h}^\pm(f)-\Psi_{N,h}^\pm(g)|
 \le C_b\|f-g\|_\infty,\qquad
|\overline{F_i(\vartheta Y_{N,h})}F_j(Y_{N,h})|
 \le M_{F_i}M_{F_j}.
$$

Retain these normalized readouts for a countable uniformly dense dictionary
of compactly supported continuous tests, together with their reflected
values, in the descriptor of {prf:ref}`thm-ym-native-fiber-continuum`.
On a common further subsequence, all finite polynomial words and reflected
products converge in expectation. Their limits extend uniquely to all
$C_0(\mathbb R^4)$ tests by the displayed continuity bound, and hence to
the stated smooth compactly supported and Schwartz test classes. Every
finite reflected matrix for this normalized class belongs to that same
hierarchy. Positivity of these matrices is a separate property of their
computed limits.

The full-face convention is retained in this construction. Replacing its
mask by a barycenter mask changes a readout by at most

$$
C_b\|f\|_\infty\sum_i p_i a_i
 \mathbf1_{\{x_i\in\operatorname{supp}f,\,
              \min_{v\in P_i}x_v^0\le0<x_i^0\}}.
$$

When $x_i$ is the face barycenter, this is bounded by the normalized
weighted cut-layer and large-face terms from the geometric splitting in
{prf:ref}`lem-ym-physical-cut-face-error`, with
$p_i a_i C_b$ in place of
$m_P\beta_P\operatorname{Tr}X_P^2/(2r)$. No localization convention
is changed in the asserted convergence.
:::

:::{prf:proof}
All normalization factors in this statement are those of
{ref}`(YM.Z3) <eq-fg-ym-z3>`. On a nonempty frame the nonnegative weights
sum to one; an empty frame contributes zero. Each validity or support mask
can only reduce the sum of absolute values. This proves both bounds for
$\Psi^\pm$ directly, without a face-count estimate or an inverse
survival-probability factor. For a monomial, multiply its individual bounds;
for a polynomial, sum these products multiplied by the absolute values of
its coefficients. Reflection exchanges the full-face masks and preserves
the normalized geometric weights under the recorded coordinate transport,
so it has the same bound. Every reflected product is consequently uniformly
integrable under the actual selected probability law.

Apply the retained-coordinate conclusion of
{prf:ref}`lem-ym-bounded-recorded-gauge-words` to these bounded normalized
readouts. This uses the existing native subsequence construction and its
likelihood refinement identity. A polynomial in finitely many retained
coordinates is continuous on their compact range; convergence of its
expectation follows. The same retained list includes both factors of every
reflected product. Source curves, geometry, and the other original retained
coordinates remain on that same subsequence.

For arbitrary tests, approximate each by the countable dictionary in
uniform norm. The second displayed estimate is uniform in the cutoff.
For products use the exact telescoping identity in
{ref}`(YM.Z5) <eq-fg-ym-z5>`; it bounds the expectation error by the sum
of the individual test errors times the bounds for the other factors.
First pass to the subsequence limit for dictionary tests, then let the
test errors tend to zero. This defines the unique limit for every stated
test and for every reflected word, with the same continuity estimate.
For bounded continuous cylinder functions of finitely many pairings, uniform
continuity on the compact range gives the same extension.

Subtracting the two support masks proves the cut-error bound. For a
barycenter at distance at least $\delta$ from the cut, a contributing
face has diameter at least $\delta$, exactly as in
{prf:ref}`lem-ym-physical-cut-face-error`. The estimate keeps the
normalized geometric weights, including their validity masks. The
unnormalized face-action sum in {ref}`(YM.Z87) <eq-fg-ym-z87>` remains
a different recorded observable and retains the weighted moment estimate
in {prf:ref}`lem-ym-bounded-recorded-gauge-words`.
:::

:::{prf:lemma} Weighted reconstruction error for full-face reflected words
:label: lem-ym-physical-reflected-reconstruction-error

Use the actual normalized readouts and selected law of
{prf:ref}`thm-ym-native-normalized-gauge-hierarchy`. Compare a readout
and its reconstruction on the retained face-slot correspondence:

$$
\Psi^+(f)=\sum_i p_i a_i\chi_i^+ b_i f(x_i),\qquad
\widehat\Psi^+(f)=
 \sum_i\widehat p_i\widehat a_i\widehat\chi_i^+
                         \widehat b_i f(\widehat x_i).
$$

Here $b_i,\widehat b_i$ denote the gauge marks, with absolute values
at most $C$; the geometric cell volumes entering $p_i,\widehat p_i$
retain their definition in {ref}`(YM.Z3) <eq-fg-ym-z3>`.
The indicators $a_i,\widehat a_i$ include the corresponding gauge and
face validity. The two support masks test all vertices of their respective
faces. The positions $x_i,\widehat x_i$ are their barycenters.
Let $f\in C_c^\infty(\{x^0>0\})$, $K=\operatorname{supp}f$, and
$\delta=\operatorname{dist}(K,\{x^0=0\})>0$.

On a frame where both normalizations are nonzero, retain the metric and
retessellation bound from {prf:ref}`thm-ym-same-record-metric-field`:

$$
\sum_i|\widehat p_i-p_i|\le q^2-1+2t.
$$

For a corresponding face, let $r_i$ be the largest displacement between
its corresponding vertices and $D_i$ its original diameter. On slots
where only one face is valid, extend the unavailable mark and geometry
by their counterparts; the validity indicators retain this discrepancy.
All terms below are evaluated on this same record. Put

$$
\begin{gathered}
\eta=\sum_i p_i|\widehat b_i-b_i|,\qquad
\rho=\sum_i p_i r_i,\qquad
\zeta=\sum_i p_i|\widehat a_i-a_i|,\\
\Lambda_\delta(K)=
 \sum_i p_i a_i\mathbf1_{\{x_i\in K,\ D_i\ge\delta/2\}}.
\end{gathered}
$$

Then the full-face reconstruction error is bounded by

$$
\begin{aligned}
|\widehat\Psi^+(f)-\Psi^+(f)|\le e_f^+
:=\min\Bigl\{2C\|f\|_\infty,\;&
 \|f\|_\infty\eta
 +C\bigl(\|\nabla f\|_\infty+2\|f\|_\infty/\delta\bigr)\rho\\
&+C\|f\|_\infty
       (q^2-1+2t+\zeta+\Lambda_\delta(K))\Bigr\}.
\end{aligned}
$$

If exactly one normalization is zero, use $e_f^+=2C\|f\|_\infty$;
if both are zero, use $e_f^+=0$. Applying the same calculation to the
reflected records defines $e_f^-$. Both errors are integrated under the
original selected law; reflection invariance is unnecessary.

For any fixed finite polynomial family $F_i$ in these readouts, let $M_i$
bound both $F_i$ and its reconstructed and reflected versions. The product
estimate {ref}`(YM.Z5) <eq-fg-ym-z5>` gives errors $e_i^\pm$ for those
words by summing the individual factor errors times the bounds of the
other factors, and then summing with the absolute polynomial coefficients.
Write $\varepsilon_i^\pm=\mathbb E e_i^\pm$. The actual reflected
matrices satisfy

$$
\begin{aligned}
|\widehat Q_{ij}-Q_{ij}|
 &\le M_j\varepsilon_i^-+M_i\varepsilon_j^+,\\
\|\widehat Q-Q\|_{\mathrm{op}}
 &\le\|M\|_2
       \bigl(\|\varepsilon^-\|_2+\|\varepsilon^+\|_2\bigr).
\end{aligned}
$$

Thus the existing metric, retessellation, mark, vertex, validity, and
weighted large-face estimates imply convergence of these same reflected
matrices whenever their displayed errors tend to zero, with the
probability of a mismatched empty-frame normalization also tending to
zero. The statement applies along the common native subsequence and
retains its selected probability law and full-face support convention.
:::

:::{prf:proof}
Insert the intermediate sum with original weights and reconstructed
integrands. The weight difference contributes at most
$C\|f\|_\infty(q^2-1+2t)$. At a slot, the difference of the mark and
test factors is bounded by

$$
|\widehat b_i f(\widehat x_i)-b_i f(x_i)|
\le\|f\|_\infty|\widehat b_i-b_i|
 +C\|\nabla f\|_\infty r_i,
$$

because the barycenter displacement is at most $r_i$. The difference of
validity and support factors contributes at most

$$
C|f(x_i)|\bigl(
 |\widehat a_i-a_i|+a_i|\widehat\chi_i^+-\chi_i^+|\bigr).
$$

Suppose $x_i\in K$ and $r_i<\delta/2$. If the original face is positive
and the reconstructed face is not, one original vertex has time coordinate
at most $r_i$, whereas $x_i^0\ge\delta$. Hence
$D_i\ge\delta-r_i>\delta/2$. If the original face is not positive and
the reconstructed face is, an original vertex has time coordinate at most
zero, giving $D_i\ge\delta$. Consequently

$$
\mathbf1_{\{x_i\in K\}}
 |\widehat\chi_i^+-\chi_i^+|
\le\mathbf1_{\{x_i\in K,\ D_i\ge\delta/2\}}
   +\mathbf1_{\{r_i\ge\delta/2\}}.
$$

Multiply by $p_i a_i$ and sum. The second sum is at most $2\rho/\delta$.
This proves the stated error; boundedness supplies its cap. Reflection
preserves distances and exchanges the two full-face masks, proving the
negative-side estimate under the same law.

Apply the existing telescoping product estimate to obtain the word errors.
Then insert one intermediate product in each reflected matrix entry:

$$
|\overline{\widehat F_i^-}\widehat F_j^+
       -\overline{F_i^-}F_j^+|
\le M_j e_i^-+M_i e_j^+.
$$

Expectation proves the entry bound. For unit vectors $u,v$, the sum of
these bounds against $|u_i||v_j|$ is at most
$\|\varepsilon^-\|_2\|M\|_2+
\|M\|_2\|\varepsilon^+\|_2$, proving the operator bound.
All primitive errors are bounded independently of the number of faces.
Their convergence in probability to zero therefore gives convergence of
their expectations; the finite polynomial estimates give the same
conclusion for every stated word. No sign of either matrix is assumed
in this reconstruction estimate.
:::

:::{prf:proposition} Physical reflected correlations of the recorded Wilson readouts
:label: prop-ym-native-physical-reflection-calculation

Use precisely the four-coordinate recorded readout and selected execution
law of {prf:ref}`prop-ym-recorded-physical-transformations`. Write
$w_P(\omega)=\beta_Ps_P(\omega)$, so
$\Phi_\omega(f)=\sum_Pw_P(\omega)f(x_P(\omega))$.
Let $f_1,\ldots,f_k$ be smooth compactly supported scalar tests in
$\mathbb R^4_+=\{x:x^0>0\}$. Retain the complete recorded face
vertices in $Y=(G,D)$ and let $\chi_P^+$ indicate that every vertex of
$P$ has $x^0>0$; let $\chi_P^-$ indicate that every vertex has $x^0<0$.
The positive-side observable is
$F_i(Y)=\sum_Pw_P\chi_P^+f_i(x_P)$. These masks ensure the entire face,
not only its barycenter, lies on the indicated side of the physical cut.
The physical reflection is $\vartheta(x^0,\mathbf x)=(-x^0,\mathbf x)$.
Its reflected matrix is the following explicit algorithmic expectation:

(eq-fg-ym-z87)=
$$
\begin{aligned}
Q_{ij}
&=\mathbb E_{\mathbb P_{N,h}}
 \sum_{P\ne Q}w_Pw_Q\chi_P^-\chi_Q^+\,
             \overline{f_i(\vartheta x_P)}f_j(x_Q)\\
&=\mathbb E_R\!\left[
 \mathcal L\sum_{P\ne Q}\beta_P\beta_Qs_Ps_Q\chi_P^-\chi_Q^+\,
             \overline{f_i(\vartheta x_P)}f_j(x_Q)\right].
\end{aligned}
\tag{YM.Z87}
$$

Here $\mathcal L=d\mathbb P_{N,h}/dR$ is the complete recorded
likelihood, including the prescribed selection normalization. The sums
use the actual realized faces and their actual positions and holonomies.
In particular the only contributing pairs have $x_P^0<0<x_Q^0$.
The diagonal $P=Q$ contributes exactly zero.

For the configured finite list of possible face slots, retain its
validity masks and set invalid weights to zero. With the fixed
nonnegative coefficients of {prf:ref}`def-wilson-action-ym`, let
$B_{N,h}$ be the sum of these coefficients over all slots. Then

(eq-fg-ym-z88)=
$$
|Q_{ij}|\le4B_{N,h}^2\|f_i\|_\infty\|f_j\|_\infty.
\tag{YM.Z88}
$$

For an arbitrary bounded future cylinder $F_i$, put
$a_i(\omega)=F_i(Y_\omega)$,
$b_i(\omega)=F_i(\mathsf T_\vartheta Y_\omega)$ and form the
recorded even and odd readouts
$e_i=(a_i+b_i)/2$, $o_i=(a_i-b_i)/2$. Define matrices by their actual
execution expectations:
$A_{ij}=\mathbb E[\overline e_i e_j]$,
$B_{ij}=\mathbb E[\overline o_i o_j]$, and
$C_{ij}=\mathbb E[\overline e_i o_j]$.
Then

(eq-fg-ym-z89)=
$$
\begin{gathered}
Q=A-B+C-C^*,\qquad
\frac{Q+Q^*}{2}=A-B,\qquad
\frac{Q-Q^*}{2}=C-C^*,\\
c^*Qc=\mathbb E|e_c|^2-\mathbb E|o_c|^2
            +2i\operatorname{Im}\mathbb E[\overline e_c o_c],
\quad e_c=\sum_ic_ie_i,\quad o_c=\sum_ic_io_i.
\end{gathered}
\tag{YM.Z89}
$$

Consequently physical reflection positivity on a given future cylinder
algebra is equivalent to $C=C^*$ and $A-B\succeq0$ for every finite
family in that algebra. These are conditions on the recorded
correlations, not on a replacement probability law.
:::

:::{prf:proof}
**The two-face calculation.** Reflecting the recorded vertices exchanges
$\chi_P^+$ and $\chi_P^-$. Insert the localized version of
{ref}`(YM.Z82) <eq-fg-ym-z82>` into the physical reflected form
{ref}`(YM.Z85) <eq-fg-ym-z85>` and multiply the two finite sums.
The retained masks multiply the two factors by $\chi_P^-$ and
$\chi_Q^+$. All $w_P$ are real. Since each $f_i$ is supported strictly in the
positive half-space, the product
$\overline{f_i(\vartheta x_P)}f_j(x_P)$ vanishes pointwise:
its first factor requires $x_P^0<0$, while its second requires
$x_P^0>0$. Removing those zero terms gives the first line of
{ref}`(YM.Z87) <eq-fg-ym-z87>`. Changing from the actual execution law
to its existing likelihood reference gives the second line. In the
survival-selected case the same expression contains
$\mathcal L\mathbf1_E/\mathbb P(E)$, as in
{prf:ref}`prop-ym-density-and-support`. Thus selection remains inside
the cross-plane correlation.

For each run, $\sum_Pw_P\le2B_{N,h}$ by the recorded Wilson bound.
Therefore the absolute value of the full double sum is at most
$4B_{N,h}^2\|f_i\|_\infty\|f_j\|_\infty$, proving
{ref}`(YM.Z88) <eq-fg-ym-z88>`. This finite-size estimate is not a
new population-uniform bound. It retains the configured face count and
normalization. Any continuum use must use the existing uniform estimates
for the particular normalized readouts being passed to the limit.

For positive-time linear combinations $f=\sum_i c_if_i$, the same
formula gives

(eq-fg-ym-z90)=
$$
c^*Qc
=\mathbb E_{\mathbb P_{N,h}}
 \sum_{\substack{P,Q\\x_P^0<0<x_Q^0}}
 w_Pw_Q\chi_P^-\chi_Q^+\,\overline{f(\vartheta x_P)}f(x_Q).
\tag{YM.Z90}
$$

Although $w_Pw_Q\ge0$, the two test-function factors are evaluated at
different recorded points. This expression is not a sum of absolute
squares. For real nonnegative $f$ it is nonnegative term by term; the
OS test class also contains signed and complex linear combinations.
This is why testing only nonnegative localized readouts does not complete
the matrix check.

**Arbitrary recorded cylinders.** Expand directly:

$$
\overline b_i a_j
=(\overline e_i-\overline o_i)(e_j+o_j)
=\overline e_i e_j-\overline o_i o_j
 +\overline e_i o_j-\overline o_i e_j.
$$

Taking the actual expectation identifies the last two matrices as
$C$ and $-C^*$ and proves {ref}`(YM.Z89) <eq-fg-ym-z89>`.
Both $A$ and $B$ are Gram matrices. A complex matrix $Q$ is positive
Hermitian precisely when its skew-Hermitian part vanishes and its
Hermitian part is positive, proving the stated equivalence.

Here $\mathsf T_\vartheta$ acts on the complete recorded geometry
and its attached face variables as constructed in
{prf:ref}`prop-ym-recorded-physical-transformations`.
If the actual readout law is invariant under $\mathsf T_\vartheta$,
then $e_i\circ\mathsf T_\vartheta=e_i$ and
$o_i\circ\mathsf T_\vartheta=-o_i$ show $C=-C$, hence $C=0$.
The positivity test is then the explicit inequality
$\mathbb E|o_c|^2\le\mathbb E|e_c|^2$ for all future words and
coefficients. Reflection invariance supplies the cancellation of $C$;
it does not by itself compare these two variances.

**Relation to the existing bounds.** Where the existing LSI controls
these particular readouts, it bounds the variance of each readout from
above by its own energy. Such upper bounds do not evaluate the difference
$A-B$ in {ref}`(YM.Z89) <eq-fg-ym-z89>`. The equilibrium-transfer
factorization supplies a nonnegative Gram form for its specified
$\sigma$-hierarchy. Formula {ref}`(YM.Z87) <eq-fg-ym-z87>` is the
physical $x^0$-reflection calculation under the original execution law.
The comparison in {ref}`(YM.Z76) <eq-fg-ym-z76>` must evaluate these
same entries. No sampling-time adjoint is used in the present calculation.
:::

:::{prf:proposition} Scalar triangle and outer-plaquette laws in the raw record
:label: prop-ym-native-scalar-face-evaluation

Use the current raw-array record identified in
{prf:ref}`rem-fractal-set-history-codec`, in the real-arithmetic readout
convention, and its scalar connection in
{prf:ref}`def-fractal-set-gauge-connection`. In
`src/fragile/fractalai/core/fractal_set.py`, the CST and IA builders store
$\phi_{\mathrm{CST}}=\phi_{\mathrm{IA}}=0$, while the IG builder stores

$$
\theta_{ij}=-\frac{V_j-V_i}{\hbar_{\mathrm{eff}}}.
$$

Here $V_i,V_j$ are the recorded fitness values and
$\hbar_{\mathrm{eff}}\ne0$ is the configured phase scale. Thus the
recorded scalar transports are exactly

$$
U_{\mathrm{CST}}^{(1)}=U_{\mathrm{IA}}^{(1)}=1,
\qquad U_{\mathrm{IG},ij}^{(1)}=e^{i\theta_{ij}}.
$$

The interaction triangle of
{prf:ref}`def-fractal-set-wilson-loop` consequently has

$$
W_\triangle^{(1)}=e^{-i\theta_{ij}},\qquad
s_\triangle^{(1)}
=1-\cos\left(\frac{V_j-V_i}{\hbar_{\mathrm{eff}}}\right).
$$

The function `_compute_wilson_loops` in
`src/fragile/fractalai/qft/analysis.py` averages these triangle cosine
readouts. Its use of the opposite sign for $\theta_{ij}$ gives the same
cosine because the other two stored phases are zero. For this scalar
triangle channel, $s_\triangle^{(1)}>0$ precisely when
$(V_j-V_i)/\hbar_{\mathrm{eff}}\notin2\pi\mathbb Z$.

In contrast, every valid outer plaquette of
{prf:ref}`def-fractal-set-plaquette` has

$$
W_P^{(1)}=1,\qquad s_P^{(1)}=0.
$$

In particular, for the scalar outer-plaquette specialization of the
physical Wilson-defect readout in
{prf:ref}`prop-ym-recorded-physical-transformations`,

$$
\Phi^{(1)}(f)=\sum_P\beta_Ps_P^{(1)}f(x_P)=0
$$

on every execution. The same is true with any of its full-face support
or validity masks. Its complete finite and limiting gauge hierarchy is
therefore evaluated at the zero field. For any finite family of bounded
future cylinders in this scalar defect field, the actual physical
reflected matrix is explicitly

$$
Q_{ij}=\overline{F_i(0)}F_j(0),\qquad
\sum_{i,j}\overline{c_i}c_jQ_{ij}
=\left|\sum_i c_iF_i(0)\right|^2\ge0.
$$

This is the exact factorization through the existing constant vector
$1\in L^2(P)$: $BF=F(0)1$. It holds for the execution law, each
survival-conditioned QSD window, and the stationary Doob law, with their
original normalizations. Its OS quotient contains only the vacuum class.
The statement evaluates the scalar outer-plaquette defect algebra; it
supplies no identification of the separate SU(2) transport matrices or
SU(3) color contractions with that algebra.
:::

:::{prf:proof}
Exponentiate the stored scalar phases and substitute them into the existing
oriented triangle formula. This proves the triangle identity and its
nonzero criterion. Substitution into the implemented diagnostic gives
$\cos(\theta_{ij})=\cos(-\theta_{ij})$, proving the stated correspondence
for that diagnostic.

The boundary of an outer plaquette consists of two CST and two IA edges.
Each scalar transport is one, including inverse orientations, so their
product is one. Equivalently, the IG factors cancel in the ordered
factorization of {prf:ref}`prop-fractal-set-wilson-factorization`.
This uses the complete scalar holonomies before taking their real parts;
multiplying two triangle cosine diagnostics would give a different value.
It follows pointwise that every outer-plaquette defect vanishes.

Every localized defect field and every reflected version are consequently
zero on the same recorded history. Insert this value into each cylinder
and integrate against the selected probability law to obtain the displayed
matrix. The normalization of that law gives $\|1\|_{L^2(P)}=1$, proving
the factorization. Every centered cylinder is zero, so the OS quotient of
this algebra is exactly its vacuum span. These identities pass to every
subsequence without a moment or boundary estimate.
:::

:::{prf:remark} Cross-plane dependence of companion and fitness normalization
:label: rem-ym-native-cross-plane-kernel-calculation

For the Gaussian companion branch of
{prf:ref}`def-fg-soft-companion-kernel`, fix a preceding record and a
positive-side alive walker $i$. Split the eligible indices by their physical
positions, retaining zero-plane indices separately. Then its actual
normalizer is

$$
Z_i=Z_i^++Z_i^-+Z_i^0,\qquad
Z_i^\pm=\sum_{j\ne i:\,\pm x_j^0>0}
             e^{-d_{\mathrm{alg}}(i,j)^2/(2\epsilon^2)}.
$$

Even when the selected companion $j$ lies on the positive side, the
probability $\kappa_i(j)=w_{ij}/Z_i$ depends on negative-side records.
On a stratum with fixed statuses and masks, a perturbation of only the
negative-side distances, holding $w_{ij}$ fixed, gives

$$
\partial_-\log\kappa_i(j)=-\frac{\partial_-Z_i^-}{Z_i}.
$$

Every eligible finite-distance negative-side walker has strictly positive
weight at fixed $\epsilon>0$. A proposed interface consisting only of faces
meeting the physical plane therefore does not record all arguments of these
conditional probabilities. A complete conditional factorization must retain
these normalizers and the actual crossing choices, together with the input
records on which their conditional laws depend.

The empirical reward standardization adds another crossing dependence. On
a smooth stratum with $n$ alive walkers and
$\sigma^2=n^{-1}\sum_k(r_k-\bar r)^2+\varepsilon_{\mathrm{std}}^2$
above the patch threshold, for $j\ne i$ one has

$$
\frac{\partial}{\partial r_j}\frac{r_i-\bar r}{\sigma}
=-\frac1{n\sigma}
 -\frac{(r_i-\bar r)(r_j-\bar r)}{n\sigma^3}.
$$

On a constant-scale patch the second term is zero and the first remains.
Logistic rescaling and the fitness exponents feed these derivatives into
both numerator and denominator of the clone gate in
{ref}`(YM.Z60) <eq-fg-ym-z60>`. A single-coordinate factor $1/n$ does
not bound the sum of the crossing contributions as $n$ grows.
The subsequent collision groups and kinetic inputs depend on the resulting
clone choices. Survival conditioning further retains
$h_{K-k-1}(s')/h_{K-k}(s)$ from
{prf:ref}`prop-ym-qsd-history-identification`.

There is also no positive reflected kernel supplied by the Gaussian distance
factor alone. In a flat metric, with equal remaining coordinates and
velocities, its value between reflected positive coordinates $s,t>0$ is
$k_\epsilon(s,t)=e^{-(s+t)^2/(2\epsilon^2)}$. For distinct $a,b>0$,

$$
\det\begin{pmatrix}
k_\epsilon(a,a)&k_\epsilon(a,b)\\
k_\epsilon(b,a)&k_\epsilon(b,b)
\end{pmatrix}
=e^{-(a+b)^2/\epsilon^2}
 \left(e^{-(a-b)^2/\epsilon^2}-1\right)<0.
$$

This tests only the indicated companion factor. It neither evaluates the
full physical reflected matrix nor disproves positivity after the complete
recorded likelihood is integrated. It does exclude using that factor alone
as the asserted positive cross-plane kernel. For the complete selected law,
{prf:ref}`prop-ym-translated-qsd-reflection-test` evaluates an actual
future-word form for translated QSD histories and obtains a negative value
under its stated nonzero-readout premise. The bounds in
{prf:ref}`lem-ym-bounded-recorded-gauge-words` justify a passage of
expectations for their specified retained words; they supply no sign for
that difference.
:::

:::{prf:remark} Physical transformations and the selected random-regulator law
:label: rem-ym-physical-symmetry-application

The physical coordinate and the recorded readout action in this check
are constructed in {prf:ref}`prop-ym-recorded-physical-transformations`.
The physical Euclidean coordinate is $x\in\mathbb R^4$.
Write $g x=Qx+b$, $Q\in O(4)$. The sampling parameter of a
Parisi–Wu realization is denoted by $\tau$, and the equilibrium transfer
parameter in {prf:ref}`thm-ym-equilibrium-hierarchy` by $\sigma$.
A translation of $x^0$, a shift of $\tau$, and a shift of $\sigma$
are three separately specified operations.

The geometric action on a realized embedded regulator $\mathsf r$
sends every vertex $x_v$ to $gx_v$, preserves incidence, and transports
each ordered edge to its image. For a recorded internal connection,
set $(gU)_{ge}=U_e$ in the corresponding transported internal frames.
The continuum one-form transformation is

(eq-fg-ym-z73)=
$$
A^g_\mu(x)=Q_{\mu\nu}A_\nu(g^{-1}x),\qquad
F^g_{\mu\nu}(x)=Q_{\mu\alpha}Q_{\nu\beta}
                         F_{\alpha\beta}(g^{-1}x).
\tag{YM.Z73}
$$

For smooth fields, $\partial_\mu(g^{-1}x)^\alpha=Q_{\mu\alpha}$
and bilinearity of the commutator prove the second identity directly
from the first. The finite face product satisfies
$U_{gP}(gU)=U_P(U)$, because its ordered factors and inverse factors
are unchanged. These are transformations of the recorded geometric
and connection data. The causal-set factorization in
{prf:ref}`prop-fractal-set-wilson-factorization` is preserved by this
operation. For distributional fields the same transformations are
defined by duality on their test functions, retaining their tensor indices.

Write the actual joint gauge law, disintegrated over its realized
regulator, as $\mu(d\mathsf r,dU)=\nu(d\mathsf r)\mu^{\mathsf r}(dU)$.
Here $\nu$ is the regulator marginal, not a gauge reference measure.
For every bounded measurable $O$ the exact symmetry check is

(eq-fg-ym-z74)=
$$
\begin{aligned}
(g_*\mu)(O)
 &=\int\nu(d\mathsf r)\int\mu^{\mathsf r}(dU)
                                      O(g\mathsf r,gU)\\
 &=\int(g_*\nu)(d\mathsf r')
        \int(g_*\mu^{g^{-1}\mathsf r'})(dU')O(\mathsf r',U').
\end{aligned}
\tag{YM.Z74}
$$

Consequently $g_*\mu=\mu$ is equivalent to both
$g_*\nu=\nu$ and
$g_*\mu^{\mathsf r}=\mu^{g\mathsf r}$ for $\nu$-almost every
$\mathsf r$. Necessity follows by first testing regulator observables
and then uniqueness of disintegration; sufficiency follows by substitution
in {ref}`(YM.Z74) <eq-fg-ym-z74>`. Allowing the regulator to move is
essential in this check.

The uniqueness argument in {prf:ref}`thm-hk-vacuum-fg` applies after
the transformed law is shown to solve the same selected problem.
For QSDs, the complete-kernel and survival calculation is given in
{prf:ref}`lem-ym-native-qsd-translation`. For example, on a realization
where a physical transformation acts on the sampling state, write
$\alpha_gF=F\circ g^{-1}$ as in
{prf:ref}`prop-ym-recorded-physical-transformations`, let $P_t$ be its sampling semigroup, and put
$\delta_{g,t}=P_t\alpha_g-\alpha_gP_t$. For its invariant law $\pi$,

(eq-fg-ym-z75)=
$$
(g_*\pi)P_tF-(g_*\pi)F=-\pi(\delta_{g^{-1},t}F),\qquad
\left|(g_*\pi)P_tF-(g_*\pi)F\right|
 \le\|\delta_{g^{-1},t}F\|_{L^1(\pi)}.
\tag{YM.Z75}
$$

Indeed $\pi P_t\alpha_{g^{-1}}F=\pi\alpha_{g^{-1}}F$; subtract
$\pi\delta_{g^{-1},t}F$ to obtain the left side. Exact equivariance
sets this defect to zero, and uniqueness then identifies the laws.
A vanishing-defect route must also pass these invariant-law equations
to the selected limit on a determining test class. This calculation
does not identify $P_t$ with a physical time-transfer operator.

**Application status.** The face identity verifies transport of the
recorded observable. The actual physical law is the joint descriptor law
and its geometry disintegration in
{prf:ref}`prop-ym-recorded-physical-transformations`. Translating the
complete spatial data transports its selected QSD and surviving histories by
{prf:ref}`lem-ym-native-qsd-translation`. Invariance with data held fixed
requires the stage defects in
{prf:ref}`rem-ym-native-fixed-data-translation-defect` to vanish, or a
proved vanishing-defect limit on the physical determining observables.
Uniqueness supplies the final same-law implication only after that check.
For the full regulator law retaining an absolute-position anchor,
{prf:ref}`prop-ym-anchored-regulator-translation-test` proves that this
check fails for some fixed translation, including in any limit retaining
that finite coordinate. Symmetry of a projected physical law requires its
own observable-level identity.
:::

:::{prf:lemma} Translation of the native QSD and its selected recorded histories
:label: lem-ym-native-qsd-translation

Use the complete Euclidean Gas update of {prf:ref}`alg-euclidean-gas`
and its recorded kernels in {prf:ref}`thm-ym-recorded-action-emergence`.
Write $\mathfrak b$ for its configured spatial data: reward, force, metric
and noise coefficients, validity domain, and any coordinate-dependent
readout frames. For $a\in\mathbb R^4$, let $t_a$ be the existing
translation of positions in
{prf:ref}`prop-ym-recorded-physical-transformations`; velocities,
recorded times, and walker indices are unchanged. Transport the data by

$$
R_{\mathfrak b_a}(x+a,v)=R_{\mathfrak b}(x,v),\qquad
F_{\mathfrak b_a}(x+a,v)=F_{\mathfrak b}(x,v),\qquad
\mathcal X_{\mathrm{valid},\mathfrak b_a}
 =\mathcal X_{\mathrm{valid},\mathfrak b}+a,
$$

and the same pullback convention for metric, noise coefficients, and
readout frames. Scalar algorithm parameters remain fixed. Let
$Q_N^{\mathfrak b}$ be the killed one-step kernel. Then

$$
Q_N^{\mathfrak b_a}(t_as,t_aA)=Q_N^{\mathfrak b}(s,A).
$$

Consequently, if $\nu_N^{\mathfrak b}Q_N^{\mathfrak b}
=\alpha_N\nu_N^{\mathfrak b}$ is the selected QSD, its translation
$\nu_N^{\mathfrak b_a}=(t_a)_*\nu_N^{\mathfrak b}$ is a QSD for
$Q_N^{\mathfrak b_a}$ with the same $\alpha_N$. Uniqueness identifies
it with that translated model's selected QSD whenever the cited uniqueness
hypotheses hold. The complete QSD histories conditioned on survival through
the same $K$ steps obey the identical pushforward relation.

Let $\mu_{\mathfrak b}$ be the joint regulator/gauge law of one such
selected history, with regulator marginal $\lambda_{\mathfrak b}$ and
conditional laws $\mu_{\mathfrak b}^{\mathsf r}$. The recorded readout
translation, with its frames transported as above, satisfies

$$
\mu_{\mathfrak b_a}=(t_a)_*\mu_{\mathfrak b},\qquad
\lambda_{\mathfrak b_a}=(t_a)_*\lambda_{\mathfrak b},\qquad
\mu_{\mathfrak b_a}^{t_a\mathsf r}
 =(t_a)_*\mu_{\mathfrak b}^{\mathsf r}
\quad\text{for }\lambda_{\mathfrak b}\text{-almost every }\mathsf r.
$$

These are translation covariance identities for the QSD family. If the
translation acts within the specified state space and the complete kernel
with fixed data is equivariant, uniqueness gives same-law invariance by
{prf:ref}`thm-hk-vacuum-fg`. Confinement does not prevent translating a
QSD together with its confining data.
:::

:::{prf:proof}
**Companion and selection stages.** Couple the two updates with identical
random inputs. Under translated metric data every algorithmic pair distance
is unchanged. Hence every companion weight, its sum over eligible indices,
and its normalized probability are unchanged. Raw rewards agree at the
translated positions. Their empirical mean, regularized standard deviation,
patched standardized scores, logistic rescaling, and fitness vector therefore
agree, including all global normalization factors. Alive sets agree because
the validity domain is transported. The same companion inputs select the
same indices and the same thresholds give the same clone masks. This
argument also covers a configured matching sampler whose probabilities
are functions of the same translated data.

**Cloning and kinetic stages.** The cloning position update satisfies
$(x_j+a)+\sigma_x\zeta=(x_j+\sigma_x\zeta)+a$.
The collision groups, velocities, random rotations, and restitution
parameters agree, so their velocity outputs agree. In BAOAB, the B stages
use equal forces at corresponding positions, the A stages add the same
velocity displacement, and the O stage has the same conditional coefficient
$\Sigma$ and Gaussian input. Any configured deterministic stages must use
the transported coordinate data specified in the statement. The final
positions consequently differ by $a$ and the velocity outputs agree.
The final validity tests and the cemetery event agree. This proves the
killed-kernel identity, with every stage and its normalizing factor retained.

**QSD and survival normalization.** Push forward the QSD identity using the
kernel relation. This gives
$\nu_N^{\mathfrak b_a}Q_N^{\mathfrak b_a}
=\alpha_N\nu_N^{\mathfrak b_a}$.
Writing $h_j^{\mathfrak b}=(Q_N^{\mathfrak b})^j1$, induction gives
$h_j^{\mathfrak b_a}(t_as)=h_j^{\mathfrak b}(s)$.
Thus the interior law and selected transition in
{prf:ref}`prop-ym-qsd-history-identification` transform as

$$
\frac{h_{K-k}^{\mathfrak b_a}}{\alpha_N^{K-k}}
 \nu_N^{\mathfrak b_a}
=(t_a)_*\left(
 \frac{h_{K-k}^{\mathfrak b}}{\alpha_N^{K-k}}
 \nu_N^{\mathfrak b}\right),\qquad
R_{k,K}^{\mathfrak b_a}(t_as,t_a ds')=R_{k,K}^{\mathfrak b}(s,ds').
$$

The survival denominator remains $\alpha_N^K$. This proves covariance
of the whole conditioned episode, not only its terminal QSD marginal.
If the chosen law is instead the stationary Doob law, transport its
positive eigenfunction by $\eta_{\mathfrak b_a}(t_as)=\eta_{\mathfrak b}(s)$;
the endpoint factor in {ref}`(SM.K5) <eq-fg-sm-k5>` then agrees as well.

The existing record encoder retains the translated positions and header,
the identical discrete decisions, and the correspondingly transported
readouts. Pushforward gives the joint-law identity; its regulator marginal
and uniqueness of disintegration give the last two identities. Every
unconditional gauge word still integrates against that regulator marginal.
:::

:::{prf:remark} Fixed-data translation defects in the actual stages
:label: rem-ym-native-fixed-data-translation-defect

For an active translation with $\mathfrak b$ held fixed, the preceding
proof identifies the quantities that must be compared. Raw reward and
force defects are
$\Delta_aR_i=R_{\mathfrak b}(x_i+a,v_i)-R_{\mathfrak b}(x_i,v_i)$
and the corresponding $\Delta_aF_i$. A fixed validity domain contributes
$\mathbf1_{\mathcal X_{\mathrm{valid}}}(x_i+a)
-\mathbf1_{\mathcal X_{\mathrm{valid}}}(x_i)$.
For unchanged eligible indices, put
$d_{ij,a}^2=d_{\mathrm{alg},\mathfrak b}(t_as;i,j)^2$ and
$Z_{i,a}=\sum_{j\ne i}\exp[-d_{ij,a}^2/(2\epsilon^2)]$.
The configured Gaussian companion probabilities have exact defect

$$
\kappa_{i,a}(j)-\kappa_i(j)
=\kappa_i(j)\left[
 e^{-(d_{ij,a}^2-d_{ij}^2)/(2\epsilon^2)}\frac{Z_i}{Z_{i,a}}-1
\right].
$$

When the metric is translation invariant this term is zero. On a fixed
alive set, a standardized reward $z_i=(r_i-\bar r)/\sigma$ has defect

$$
z_{i,a}-z_i
=\frac{\Delta_a r_i-\overline{\Delta_a r}}{\sigma_a}
 +(r_i-\bar r)\left(\frac1{\sigma_a}-\frac1\sigma\right),
$$

with the actual patched scales $\sigma_a,\sigma$. The resulting
fitnesses enter the clipped gate probability in
{ref}`(YM.Z60) <eq-fg-ym-z60>`; its normalization remains
$p_{\max}(V_i+\epsilon_{\mathrm{clone}})$.
Cloning jitter, velocity collisions, and the A drift have zero direct
translation defect when their inputs agree. The B and O stages retain the
force and noise-coefficient defects at their actual intermediate positions.
A change of alive set additionally changes every affected companion and
empirical normalization.

Thus translated QSDs are available by
{prf:ref}`lem-ym-native-qsd-translation`. To infer invariance with fixed
data, or restoration in a specified limit, these fixed-data defects must
vanish exactly or in the required determining correlations. The physical
readout algebra and a state space modulo translations must be specified if
such a quotient is used; the complete record in
{prf:ref}`def-fractal-set-record-coverage` retains absolute position anchors.
:::

:::{prf:proposition} Physical reflection test on translated native QSD histories
:label: prop-ym-translated-qsd-reflection-test

Fix a finite recorded cutoff and its selected QSD history law $P$ from
{prf:ref}`prop-ym-qsd-history-identification`. Use the nonnegative Wilson
weights $w_P=m_P\beta_Ps_P$ on the configured face slots of
{prf:ref}`prop-ym-native-physical-reflection-calculation`, with
$m_P$ their recorded validity mask. Suppose one nonnegative compactly supported smooth
test $f$ has

$$
m=\mathbb E_P\sum_Pw_Pf(x_P)>0.
$$

Let $P_a$ be the actual QSD history law obtained by translating the spatial
data and positions by $ae_0$ as in
{prf:ref}`lem-ym-native-qsd-translation`, with the same survival horizon.
Set $f_a(x)=f(x-ae_0)$. For sufficiently large $a$, $f_a$ is supported
strictly in $x^0>0$. Define its full-face future readout
$F_a=\sum_Pw_P\chi_P^+f_a(x_P)$ on the translated record and let
$M=2B_{N,h}\|f\|_\infty$, so $0\le F_a\le M$.
Then, for the two future words $(1,F_a)$, write

$$
u_a=\mathbb E_{P_a}F_a,\qquad
v_a=\mathbb E_{P_a}F_a(\vartheta Y),\qquad
q_a=\mathbb E_{P_a}[F_a(\vartheta Y)F_a(Y)].
$$

Their actual reflected matrix is

$$
Q_a=\begin{pmatrix}1&u_a\\v_a&q_a\end{pmatrix},\qquad
u_a\longrightarrow m,\quad v_a\longrightarrow0,\quad
0\le q_a\le Mv_a\longrightarrow0.
$$

In particular, for all sufficiently large $a$, the bounded future
word $G_a=F_a-m/2$ has strictly negative physical reflected form:

$$
\mathbb E_{P_a}[\overline{G_a(\vartheta Y)}G_a(Y)]
=q_a-\frac m2(u_a+v_a)+\frac{m^2}{4}
\le-\frac{m^2}{16}<0.
$$

Thus translation covariance of native QSDs does not establish physical
reflection positivity for every member of that translated family.
The calculation uses the complete selected law and its actual face words.
It makes no assertion that a finite-cutoff negative value persists in a
continuum limit without uniform versions of its bounds.
:::

:::{prf:proof}
Use the exact QSD history pushforward of
{prf:ref}`lem-ym-native-qsd-translation` to express the three expectations
under $P$. Translation leaves each $w_P$ unchanged and gives

$$
\begin{aligned}
u_a&=\mathbb E_P\sum_Pw_P
 \mathbf1_{\{\min_{v\in P}x_v^0>-a\}}f(x_P),\\
v_a&=\mathbb E_P\sum_Pw_P
 \mathbf1_{\{\max_{v\in P}x_v^0<-a\}}
 f(\vartheta x_P-2ae_0).
\end{aligned}
$$

These are integrations of the full companion, cloning, kinetic, and
survival likelihood. No side independence is used. Every recorded finite
face has finite vertex coordinates. Its first mask tends to one. In the
second line the compactly supported test eventually vanishes on every
fixed record. Both sums are bounded by $M$, so dominated convergence gives
$u_a\to m$ and $v_a\to0$. Pointwise $F_a\le M$ gives
$0\le q_a\le Mv_a$ under the same law. Choose $a$ so large that
$u_a\ge3m/4$ and $Mv_a\le m^2/16$. The displayed bound for $G_a$
follows by expanding its reflected form. The constants and the full-face
masks remain in this calculation throughout.
:::

:::{prf:proposition} Translation invariance and the existing global geometric normalization
:label: prop-ym-normalized-physical-translation-test

Take the globally normalized geometric readout in
{ref}`(YM.Z3) <eq-fg-ym-z3>` with a bounded nonnegative Wilson mark
$0\le s_i\le2$, its existing validity mask, and a physical support test:

$$
\Phi_{N,h}(f)=\sum_i p_i a_i s_i f(x_i),\qquad
p_i\ge0,\quad \sum_i p_i\le1.
$$

Here $a_i\in\{0,1\}$ is the recorded gauge-validity mask; the
empty-frame readout is zero.
Every continuum limit identified by the same normalized test pairings has
$|\Phi(f)|\le2\|f\|_\infty$ and is a nonnegative finite measure of total
mass at most two. If its physical law is invariant under all translations
of $x\in\mathbb R^4$, then $\Phi=0$ almost surely.
Consequently a nonzero physical gauge hierarchy built from these globally
normalized nonnegative marks cannot satisfy both same-law translation
invariance and the requested nonvacuum conclusion. A different scaling,
such as a specified centered fluctuation hierarchy, requires its own
physical reflected-product calculation.
:::

:::{prf:proof}
The finite bound follows from the recorded normalized weights and $s_i\le2$.
Retain the test pairings on a countable uniformly dense subspace of
$C_0(\mathbb R^4)$ in the existing native continuum construction. Their
linearity, positivity, and uniform norm bound pass jointly to the limit and
extend to all of $C_0$ by continuity. The representing nonnegative measure
has total mass at most two. This conclusion also allows mass to escape to
infinity along the sequence.

Let $\overline\Phi=\mathbb E\Phi$, a finite nonnegative measure. Invariance
of the physical law makes $\overline\Phi$ translation invariant. If
$\overline\Phi(K)>0$ for some compact $K$, place arbitrarily many disjoint
translates of $K$ in $\mathbb R^4$. Invariance would give
$\overline\Phi(\mathbb R^4)\ge n\overline\Phi(K)$ for every $n$,
contradicting its bound by two. Thus $\overline\Phi$ vanishes on every
compact set, hence is zero. A countable compact exhaustion and nonnegativity
then give $\Phi=0$ almost surely. All its positive-degree gauge words vanish;
the generated unital algebra supplies only the vacuum class.

This concerns the fixed global normalization in (YM.Z3). For the
unnormalized Wilson sum in {ref}`(YM.Z87) <eq-fg-ym-z87>`, a uniform
bound on the total mass has not been obtained by this argument. For a
$\sqrt N$-scaled centered field, the total-variation bound instead grows
with $N$. Neither construction may be substituted for (YM.Z3) without
tracking that change of observable and its correlations.
:::

:::{prf:proposition} Translation test for the retained absolute-position regulator law
:label: prop-ym-anchored-regulator-translation-test

Let $\lambda$ be a probability law for a recorded regulator that retains an
absolute position $X(\mathsf r)\in\mathbb R^4$ of a specified vertex
label, as in {prf:ref}`def-fractal-set-record-coverage`. Suppose the
recorded translation preserves that label, so
$X(t_a\mathsf r)=X(\mathsf r)+a$.
Then $\lambda$ cannot be invariant under all $a\in\mathbb R^4$.
More quantitatively, there are a fixed translation $a$ and a bounded smooth
regulator observable $O(\mathsf r)=f(X(\mathsf r)-a)$, with
$0\le f\le1$, such that

$$
\int O\,d((t_a)_*\lambda)-\int O\,d\lambda>\frac12.
$$

If a selected regulator sequence retains that position and its position
marginals converge weakly to a probability on $\mathbb R^4$, the same
observable and translation have a nonzero limiting defect. This applies to
a QSD-derived selected history just as to any other probability law.
It is consistent with the translation covariance of the QSD family in
{prf:ref}`lem-ym-native-qsd-translation`.
:::

:::{prf:proof}
Let $\rho=X_*\lambda$. Choose $f\in C_c^\infty(\mathbb R^4)$ with
$0\le f\le1$ and $\rho(f)>3/4$, using a compact set carrying more
than three quarters of this probability. Let $K=\operatorname{supp}f$.
Choose $a$ so large that $K$ and $K+a$ are disjoint. Since
$\rho(K)>3/4$, one has $\rho(K+a)<1/4$. Therefore

$$
\int O\,d((t_a)_*\lambda)=\rho(f)>\frac34,
\qquad
\int O\,d\lambda=\rho(f(\cdot-a))\le\rho(K+a)<\frac14.
$$

This proves the claimed defect on a bounded smooth determining observable.
For weakly convergent position marginals, both integrands $f$ and
$f(\cdot-a)$ are bounded continuous, so the same difference converges to
a value greater than $1/2$. Thus this defect cannot vanish on the limiting
anchored regulator algebra.

In particular, the two-component same-law requirement
$g_*\lambda=\lambda$ and
$g_*\mu^{\mathsf r}=\mu^{g\mathsf r}$ from
{ref}`(YM.Z74) <eq-fg-ym-z74>` cannot hold for all translations on a
probability law retaining this absolute-position coordinate. A covariance
statement transporting the background and its QSD is instead exactly the
identity proved in {prf:ref}`lem-ym-native-qsd-translation`. A proposed
unlabelled or relative-coordinate limiting law must be specified separately,
with its physical observable correspondence, before applying a same-law
translation argument to that different state space.
:::

:::{prf:remark} Common hierarchy and physical reflection check
:label: rem-ym-physical-hierarchy-application

There are two established continuum constructions to compare.
{prf:ref}`thm-ym-native-fiber-continuum` retains the native descriptor,
geometry, source curves, and their conditional likelihoods on a common
subsequence. {prf:ref}`thm-ym-equilibrium-fluctuation-limit` constructs
the nontrivial hierarchy of the specified equilibrium fluctuation fields,
with reflection in $\sigma$. Its lower reflected-norm estimate is
{ref}`(YM.E13) <eq-fg-ym-e13>`. These conclusions, including their
population-independent constants, are already available.

In the comparison below, $S^{\mathrm{phys}}$ denotes moments of the
physical gauge readouts under the actual selected recorded law, or their
identified common limit. Every physical word is evaluated under that law.
For physical scalar observables, set
$\vartheta(x^0,\mathbf x)=(-x^0,\mathbf x)$ and
$\Theta_{\mathrm{phys}}O(f)=O(\overline{f\circ\vartheta})$
for Hermitian $O$. Tensor observables additionally carry the reflection
matrix from {ref}`(YM.Z73) <eq-fg-ym-z73>` on each spacetime index.
On products use the adjoint order and conjugate coefficients; this also
fixes the convention for represented fermionic words. These operations
refer to support in physical spacetime, independently of $\tau$ and
$\sigma$.

Let $\iota$ denote a proposed identification of the physical future
observable words with words of the established equilibrium hierarchy.
It must specify each observable, smearing, normalization, and support.
For any finite list $F_1,\ldots,F_k$, the quantities to compare are

(eq-fg-ym-z76)=
$$
Q^{\mathrm{phys}}_{ij}
 =S^{\mathrm{phys}}(\Theta_{\mathrm{phys}}F_i\,F_j),\qquad
Q^{\mathrm{eq}}_{ij}
 =S^{\mathrm{eq}}(\Theta_\sigma\iota F_i\,\iota F_j),\qquad
\epsilon_{ij}=|Q^{\mathrm{phys}}_{ij}-Q^{\mathrm{eq}}_{ij}|.
\tag{YM.Z76}
$$

The established equilibrium reflection calculation gives
$Q^{\mathrm{eq}}_{ij}=\langle h_{\iota F_i},h_{\iota F_j}\rangle$.
Consequently, whenever both matrices are defined at the comparison
scale, their Hermitian quadratic forms obey

(eq-fg-ym-z77)=
$$
\operatorname{Re}\sum_{i,j}\overline c_i c_jQ^{\mathrm{phys}}_{ij}
\ge-\sum_{i,j}|c_i||c_j|\epsilon_{ij}.
\tag{YM.Z77}
$$

This follows by subtracting the two matrices, bounding each error entry,
and using positivity of the equilibrium Gram matrix. If the limiting
entries agree, the physical matrix equals that Gram matrix and is
positive Hermitian. For a retained witness $F$ with equilibrium
reflected norm at least $c>0$, its physical reflected norm has real
part at least $c-\epsilon_{FF}$. Thus the already proved nontriviality
passes with the same observable once this error tends to zero.
The complete hierarchy comparison uses all finite observable words;
reflected words are the particular products required for this positivity
calculation. A common subsequence is sufficient for these passages.

**Application status.** The native continuum theorem proves convergence
with its specified retained descriptor insertions. It does not identify
those insertions with the physical local gauge products in
{ref}`(YM.Z76) <eq-fg-ym-z76>`. The equilibrium hierarchy theorem
identifies static moments of its one-time direct observables with
$\rho_N$ moments. Its proof explicitly retains a separate law for
blocks spanning recorded stages. Agreement of these static moments
therefore does not evaluate $\epsilon_{ij}$ for physical reflected
words. The application to physical reflection requires that word map
and its correlations, rather than a new proof of equilibrium reflection
positivity or equality of the sampling generator with a physical Hamiltonian.
:::

:::{prf:remark} Physical Hamiltonian, sector, and time normalization
:label: rem-ym-physical-gap-application

The established equilibrium OS transfer has generator $H_{\mathrm{eq}}$
and centered gap $\lambda_*=1/C_*$ in
{prf:ref}`thm-ym-equilibrium-fluctuation-limit`. The application to a
physical hierarchy can be checked directly from the preceding word map.
Equality of its reflected Gram matrices defines an isometry

(eq-fg-ym-z78)=
$$
V[F]_{\mathrm{phys}}=[\iota F]_{\mathrm{eq}}.
\tag{YM.Z78}
$$

To verify this, the norm squared of any finite linear combination on
either side is the same matrix quadratic form in
{ref}`(YM.Z76) <eq-fg-ym-z76>`. Null vectors therefore map to null
vectors; the map extends by completion to an isometry with closed image.
The constant word supplies the same vacuum when $\iota1=1$.

The physical time check is, for a fixed $\kappa>0$,

(eq-fg-ym-z79)=
$$
V T_t^{\mathrm{phys}}=T_{\kappa t}^{\mathrm{eq}}V,
\qquad \sigma=\kappa t.
\tag{YM.Z79}
$$

This is checked on translated future words and then extended by their
density and contraction of the transfers. It makes $\operatorname{Ran}V$
invariant under every $T_{\kappa t}^{\mathrm{eq}}$. Self-adjointness
also makes its orthogonal complement invariant: for $u\perp\operatorname{Ran}V$
and $v\in\operatorname{Ran}V$,
$\langle T^{\mathrm{eq}}u,v\rangle
=\langle u,T^{\mathrm{eq}}v\rangle=0$.
The generator and its domain are consequently

(eq-fg-ym-z80)=
$$
\begin{gathered}
H_{\mathrm{phys}}=\kappa V^*H_{\mathrm{eq}}V,\qquad
\operatorname{Dom}H_{\mathrm{phys}}
 =\{u:Vu\in\operatorname{Dom}H_{\mathrm{eq}}\},\\
\|T_t^{\mathrm{phys}}u\|
\le e^{-\kappa t/C_*}\|u\|\quad(u\perp\Omega_{\mathrm{phys}}),
\qquad
\operatorname{spec}H_{\mathrm{phys}}
 \subset\{0\}\cup[\kappa/C_*,\infty).
\end{gathered}
\tag{YM.Z80}
$$

The domain identity follows either from the strong derivative of the
intertwined semigroups or their spectral resolution on the reducing
image. The norm bound follows by applying the equilibrium bound to
$Vu$, whose orthogonality to the vacuum is preserved. With the calibrated
action unit, the energy gap is at least
$\hbar_{\mathrm{eff}}\kappa/C_*$. A positive reflected witness from
{ref}`(YM.Z77) <eq-fg-ym-z77>` ensures a nonzero physical nonvacuum
space; an isometry of the vacuum alone would not establish this.

**Application status.** These calculations specify the sector, domain,
and time factor needed to apply the existing gap. The chapter already
constructs the equilibrium operator and its limiting gap, so that route
needs no second operator-limit argument. If instead the physical operator
is constructed as a separate cutoff limit, the embeddings and strong
semigroup convergence in {prf:ref}`thm-mass-gap-rg-fixed-point` must
be identified for that cutoff family. The current native source-limit
construction does not supply the word map in
{ref}`(YM.Z78) <eq-fg-ym-z78>` or physical time intertwining in
{ref}`(YM.Z79) <eq-fg-ym-z79>`. Its source derivatives and their
bounds remain valid without making either identification.
:::

### 12.4. Local algebras and their conditional spacetime structure

:::{prf:definition} Local observable algebras
:label: def-local-algebra-fg

For a bounded spacetime region $\mathcal O$, let $\mathcal B(\mathcal O)$
be a specified self-adjoint set of bounded field observables supported there,
and define

(eq-fg-ym-43)=
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
measure as in {ref}`(YM.38) <eq-fg-ym-38>`. The Lorentz-boost argument there excludes every
spectral point outside the forward cone. The self-adjoint semigroup argument
in {ref}`(YM.33) <eq-fg-ym-33>`--{ref}`(YM.34) <eq-fg-ym-34>` gives the stated additional gap. Spatial periodicity or
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

(eq-fg-ym-44)=
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

(eq-fg-ym-44a)=
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
Combining the quadratic margin with {ref}`(YM.44a) <eq-fg-ym-44a>` gives
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

(eq-fg-ym-44b)=
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

(eq-fg-ym-45)=
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
an actual gauge-theory spectral family obeying {ref}`(YM.45) <eq-fg-ym-45>` and the selection model
{ref}`(YM.44) <eq-fg-ym-44>`; that identification is the conjecture
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
Take $r_\kappa=C\pi^2\kappa^2/4$ to obtain {ref}`(YM.45) <eq-fg-ym-45>`.
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
spectral estimate {ref}`(YM.45) <eq-fg-ym-45>`.

:::{prf:corollary} Scope of the theta-selection model
:label: cor-no-axions

Under {ref}`(YM.45) <eq-fg-ym-45>` and the selection model {ref}`(YM.44) <eq-fg-ym-44>`, the specified model selects
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

The record codec precedes the invariant-coordinate maps and the implemented
kernel calculation {ref}`(SM.K1) <eq-fg-sm-k1>`. Its complete history law precedes the exact
fermionic word evolution {ref}`(LQ.R1) <eq-fg-lq-r1>`--{ref}`(LQ.R4) <eq-fg-lq-r4>`. The analytical inputs are the
independently proved law-specific convergence, LSI, regularity, and
ellipticity results. Their use in {ref}`(YM.F1) <eq-fg-ym-f1>`--{ref}`(YM.F12) <eq-fg-ym-f12>` retains the original
state, observation domain, and update convention.

The equilibrium energy closure {ref}`(YM.E2) <eq-fg-ym-e2>` uses that identified LSI law and
constructs its own self-adjoint transfer before applying the reflection
lemma. The bounded-tilt comparison supplies the nontriviality estimate;
the common correlation limit and transfer inequality {ref}`(YM.E14) <eq-fg-ym-e14>` then
construct its continuum OS space and gap. The product limit and the
limiting-law mode approximation have the distinct proofs {ref}`(YM.E15) <eq-fg-ym-e15>` and
{ref}`(YM.E17) <eq-fg-ym-e17>`. The synthesis below records these dependencies explicitly.

The Wilson variation, independent-plaquette area law, and smooth-connection
continuum limit keep their specified link sectors. None of those field
models supplies an input to the record, LSI, or equilibrium construction.

:::

(sec-ym-algorithmic-qft-synthesis)=
## 14. The connected algorithmic and equilibrium field theory

:::{div} feynman-prose
Now follow a recorded interaction through the whole construction. Its IA and
IG transports determine the triangle holonomy, and its transported companion
doublets retain their angular matter energy. Averaging the complete history
likelihood over these descriptors supplies their effective action and exact
time correlations, including every cloning write. The established analytic
estimates
therefore control the same observable after reconstruction. For the
identified stationary joint-LSI family, they give the common subsequential
fluctuation hierarchy and the paired momentum bound.

The same native joint action splits into geometry and normalized conditional
gauge contributions by {prf:ref}`thm-ym-native-geometry-fiber-action`
and {ref}`(YM.Z52) <eq-fg-ym-z52>`.
{prf:ref}`thm-ym-native-fiber-continuum` carries this sourced
split to the common limit, where the geometry weight and fiber denominator
cancel to recover the native sourced correlations in
{ref}`(YM.Z58) <eq-fg-ym-z58>`.

After $n$ relaxation steps, {prf:ref}`thm-ym-qsd-window-relaxation`
compares the actual survival-conditioned $w$-step descriptor with the QSD
window law in {ref}`(YM.J5) <eq-fg-ym-j5>`, with total-variation error
at most $2\alpha_N^{-w}\delta_n$ in {ref}`(YM.Z49) <eq-fg-ym-z49>`.
The rate already proved in {prf:ref}`thm-main-convergence` gives the
explicit burn-in in {ref}`(YM.Z51) <eq-fg-ym-z51>`.
This burn-in counts algorithmic relaxation steps and does not identify
that counter with a physical spacetime coordinate.

The alternating insertion construction preserves the full exterior algebra
of recorded modes: independent exterior combinations remain independent as
operators. The recorded contraction also supplies a unital completely
positive CAR evolution, with its explicit two-point correspondence and
higher-word prescription. If we retain only selected channels, the discarded
coordinates reappear in their exact memory terms. These are computed from
the complete transition, so compressing the record still leaves a precise
account of its time dependence.

The gradient energy of the specified one-time law supplies a second,
equilibrium evolution. We have constructed its self-adjoint transfer,
reflection form, and gap. For the product and uniformly bounded joint-tilt
families, the polynomial Gram bounds preserve arbitrarily many independent
observable vectors through the population limit. In the product case the
Gaussian covariance is explicit. The bounded-tilt comparison identifies
the empirical bulk measure while retaining each fluctuation law's joint
dynamics. Starting instead from the proved mean-field marginal, increasing
a form-dense mode cutoff approximates that law's equilibrium operator.

For equilibrium channels the positive two-step defect tests closure
directly. The mask calculation also identifies which bounded insertions
have no finite gradient energy. These facts determine how channel
observables and energy modes enter a simulation.

The theorem gathers these connections with each law, clock, and limit intact.
These distinctions tell us which correlations a simulation computes and
which further spacetime properties a relativistic reconstruction would need.
:::

:::{prf:theorem} Algorithmic QFT from reconstruction, recorded evolution, and equilibrium energy
:label: thm-ym-algorithmic-qft-synthesis

The existing Volume 2 constructions and the calculations in this chapter
give the following connected theory, in their identified law families.

1. **Recorded dynamics.** The complete Fractal Set encoding represents
   the implemented transition by {ref}`(SM.K1) <eq-fg-sm-k1>`, with its exact domain and adjoint
   in {ref}`(SM.K2) <eq-fg-sm-k2>`--{ref}`(SM.K3) <eq-fg-sm-k3>`. Direct invariant observables have the same finite
   history law and correlations {ref}`(SM.K4) <eq-fg-sm-k4>`, including recorded selection and
   mask data. The CAR construction represents its fermionic words by
   {ref}`(LQ.R1) <eq-fg-lq-r1>`--{ref}`(LQ.R4) <eq-fg-lq-r4>` and transports the actual generator by {ref}`(LQ.R3) <eq-fg-lq-r3>`.
   Its faithful insertion algebra is {prf:ref}`thm-lqft-oriented-word-algebra`,
   and its unital completely positive CAR evolution is
   {prf:ref}`thm-lqft-record-car-channel`. Compressed channel histories
   retain the exact memory in {prf:ref}`thm-sm-direct-channel-memory`.
   Closing the original channel algebra under its actual kernel constructs
   the prediction-complete descriptor of
   {prf:ref}`thm-sm-prediction-complete-descriptors`; its finite transition
   matrices converge by {prf:ref}`thm-sm-predictive-partition-convergence`.
   The prefix likelihood gives the same channel history through its
   effective action, predictive kernel, and generating functional in
   {prf:ref}`thm-sm-effective-recorded-gauge-dynamics` and
   {prf:ref}`cor-sm-recorded-gauge-generating-functional`.
2. **Analytical control.** The reconstructed observables inherit the
   estimates proved for their original law and energy. For the stationary
   LSI family, {ref}`(YM.F1) <eq-fg-ym-f1>`--{ref}`(YM.F5) <eq-fg-ym-f5>` give a distribution-valued fluctuation
   hierarchy along a common subsequence. For the same identified LSI
   family, the paired Euclidean regime has the nonzero momentum bound in
   {prf:ref}`cor-ym-paired-momentum-fluctuations`. The complete update determines
   its drift and covariance equations {ref}`(YM.F6) <eq-fg-ym-f6>`--{ref}`(YM.F12) <eq-fg-ym-f12>`.
   For the selected QSD, the complete block kernel gives its actual density
   comparison in {prf:ref}`prop-ym-selected-qsd-block-density`, and
   {prf:ref}`prop-ym-qsd-history-identification` computes every
   survival-conditioned history weight. The implemented same-stage color
   threshold and bounded channel correlations have the explicit
   approximation and sampling bounds in
   {prf:ref}`prop-ym-color-threshold-correlation-limit` and
   {prf:ref}`prop-ym-channel-lag-estimation`, with the law specified there.
3. **Equilibrium transfer.** The full-gradient LSI law has the closed
   equilibrium energy {ref}`(YM.E2) <eq-fg-ym-e2>`, self-adjoint operator, and gap {ref}`(YM.E4) <eq-fg-ym-e4>`.
   Its direct observable hierarchy is {ref}`(YM.E7) <eq-fg-ym-e7>`--{ref}`(YM.E8) <eq-fg-ym-e8>`, and its
   fermionic Hamiltonian is {ref}`(LQ.R5) <eq-fg-lq-r5>`. All static moments of the same
   one-time observables agree with their identified algorithmic law.
   The nonzero finite-population color sector is
   {prf:ref}`cor-ym-nonzero-direct-color-sector`. The actual color feature
   map has the state-space curvature calculated in
   {prf:ref}`prop-sm-recorded-color-connection-curvature`. The distinct
   Fractal Set attribution connection has its triangle holonomy and
   Wilson defect in {prf:ref}`prop-sm-attribution-holonomy-defect`.
   Channel closure is tested
   by {prf:ref}`thm-ym-equilibrium-channel-closure`; hard mask domains are
   calculated in {prf:ref}`prop-ym-color-mask-domain`.
4. **Constructed limits.** The established product and uniformly bounded
   joint-tilt families give a reflection-positive subsequential equilibrium
   fluctuation hierarchy with a uniform transfer gap and infinite-dimensional
   OS space by {prf:ref}`thm-ym-fluctuation-os-infinite-dimension`. In the
   product case the whole hierarchy converges to the explicitly calculated
   Gaussian law {ref}`(YM.E15) <eq-fg-ym-e15>`. The already proved mean-field marginal has its
   own directly constructed equilibrium transfer, approximated strongly
   by the energy-mode matrices {ref}`(YM.E16) <eq-fg-ym-e16>`--{ref}`(YM.E17) <eq-fg-ym-e17>`.
   The resulting finite fermionic simulators converge by
   {prf:ref}`cor-ym-galerkin-car-convergence`. Within the same exchangeable
   bounded-tilt family, {prf:ref}`thm-ym-bounded-tilt-bulk-identification`
   identifies the fixed-marginal mean-field law explicitly.

All four parts use the original reconstruction maps and the stated laws.
The recorded evolution and the equilibrium energy evolution share their
identified static observables. Their time-dependent correlations are
specified by their respective transition operators.
:::

:::{prf:proof}
The dependency chain starts with record coverage and the lossless decoder,
followed by the direct orbit and measure isomorphisms in Chapter 04. The
substitution in {ref}`(SM.K1) <eq-fg-sm-k1>` then proves the encoded update and its history
measure. The exterior Hilbert construction in Chapter 03 is applied to
this actual contraction; differentiating its wedges and transporting the
derivative gives {ref}`(LQ.R3) <eq-fg-lq-r3>`. The explicit insertion norm proves faithfulness, and the isometric
compression in {prf:ref}`thm-lqft-record-car-channel` proves complete
positivity from the same recorded contraction. Conditional block
elimination gives {ref}`(SM.M2) <eq-fg-sm-m2>` for compressed channels.
The prediction-complete algebra is generated by repeated application of
this same kernel and by the original bounded insertions. Its invariance
is proved by the monotone-class calculation, which makes the channel
intertwining exact without imposing closure on the initial readouts.
The transition entries {ref}`(SM.T3) <eq-fg-sm-t3>` are conditional
averages of the actual stationary two-time law. Strong convergence of the
partition projections and {ref}`(SM.T5) <eq-fg-sm-t5>` then prove their
correlation and CAR approximation. The prefix-density ratio
{ref}`(SM.G2) <eq-fg-sm-g2>` gives the same full-history predictions
before this algebra completion. These steps use the recorded evolution.

Independently, the convergence and regularity chapters establish their
law-specific concentration, ellipticity, and derivative bounds. The
reconstruction equality transports the same tested quantities. The
stationary one-time LSI gives {ref}`(YM.F2) <eq-fg-ym-f2>`; the Hermite calculation supplies
the common distribution topology and all-order moment convergence. The
generator product rule and the conditional variance decomposition give
the selected drift and noise, including the actual overlapping collision
increments. These calculations precede every proposed identification of a
fluctuation law. Integrating the complete killed block against its
selected eigenmeasure gives {ref}`(YM.J1) <eq-fg-ym-j1>`; the explicit
entropy comparison gives the corresponding same-law constant
{ref}`(YM.J2) <eq-fg-ym-j2>`. The cancellation of successive survival
weights proves {ref}`(YM.J5) <eq-fg-ym-j5>` for the full conditioned
history. For the established equilibrium family, the Gaussian force-radius
calculation and bounded-product telescoping give
{ref}`(YM.V2) <eq-fg-ym-v2>`--{ref}`(YM.V4) <eq-fg-ym-v4>` for the
implemented color threshold. The stationary window calculation gives
{ref}`(YM.V5) <eq-fg-ym-v5>`--{ref}`(YM.V6) <eq-fg-ym-v6>` for its
measured correlations, including overlap and empirical centering.

The same established LSI closes the energy through {ref}`(YM.E2) <eq-fg-ym-e2>`, producing its
resolvent, gap, and conservative reversible transfer. The boundary-vector
calculation then establishes reflection positivity for this constructed
law, and the previously defined exterior construction lifts its Hamiltonian.
Neither the kinetic Hamiltonian transport nor a centered cloning noise is
used as a symmetry assertion. Indeed {ref}`(YM.R2) <eq-fg-ym-r2>` computes the obstruction to
the momentum-reflection shortcut in the existing kinetic reference.

Finally, the actual bounded-tilt estimate gives {ref}`(YM.E10) <eq-fg-ym-e10>`, while the
equilibrium energy gives {ref}`(YM.E12) <eq-fg-ym-e12>`. They preserve a positive reflected
norm through the common subsequential limit. Inequality {ref}`(YM.E14) <eq-fg-ym-e14>`
constructs its gapped transfer on the same limiting hierarchy. The
polynomial Gram estimate {ref}`(YM.Q8) <eq-fg-ym-q8>` proves infinite
dimension, and {ref}`(YM.Q1) <eq-fg-ym-q1>` computes its finite-law
time-average control. The product characteristic-function calculation identifies its full-sequence
limit in that family. The separate fixed-marginal limit inherits its LSI
from the upstream chaos theorem; the explicit form recovery and resolvent
calculation then prove {ref}`(YM.E17) <eq-fg-ym-e17>`. Each arrow thus has a calculation for
its own law and limit, with no use of its eventual conclusion upstream.
:::

:::{prf:remark} Consequences for simulation and relativistic reconstruction
:label: rem-ym-synthesis-simulation-scope

The exact simulation observables are the direct recorded channels. Their
algorithmic multi-time statistics use {ref}`(SM.K4) <eq-fg-sm-k4>`; their CAR word statistics
use the replica contractions in {ref}`(LQ.R4) <eq-fg-lq-r4>` or the explicitly
specified CAR regression in {ref}`(LQ.C1) <eq-fg-lq-c1>`--{ref}`(LQ.C2) <eq-fg-lq-c2>`.
The native fermion generator and regional covariance are computed in
{prf:ref}`thm-lqft-native-local-fermion-evolution` and
{prf:ref}`thm-lqft-record-locality-defect`. The finite predictive
transition matrices in {ref}`(SM.T3) <eq-fg-sm-t3>` give a convergent
approximation of these recorded observables and their CAR evolution.
A compressed channel simulator retains {ref}`(SM.M2) <eq-fg-sm-m2>` until
closure is established. A simulator of the equilibrium
transfer uses the mass and stiffness matrices {ref}`(YM.E16) <eq-fg-ym-e16>`, the specified
observable matrices, and the fermionic lift {ref}`(LQ.R5) <eq-fg-lq-r5>`.
The finite-energy core and bounded masked insertions have the distinct
domains proved in {prf:ref}`prop-ym-color-mask-domain`. These definitions
identify what each numerical matrix element estimates.

The hierarchy in {prf:ref}`thm-ym-equilibrium-fluctuation-limit` is a
constructed continuum field theory with positive time reflection and a
uniform transfer gap in the stated family. Full Euclidean spacetime
covariance, the specific interacting selected-law fluctuation limit, and
an independently varying geometric or volume limit retain their separate
identification requirements. The present construction proves neither an
equality of the full recorded generator with $-H_{\mathrm{eq}}$ nor a
four-dimensional relativistic Yang--Mills identification. A use of the
already cited OS reconstruction theorem keeps those exact spacetime and
field-domain requirements for the same hierarchy.

For the scalar phase record implemented by the raw-array codec,
{prf:ref}`prop-ym-native-scalar-face-evaluation` evaluates the outer-plaquette
defect hierarchy exactly: it is zero, with a positive reflected form and
only the vacuum class. Its potentially nonzero triangle cosine diagnostic
is a different observable.

For the existing normalized geometric gauge readouts,
{prf:ref}`thm-ym-native-normalized-gauge-hierarchy` now places all finite
words and reflected products on a common native subsequence.
{prf:ref}`lem-ym-physical-reflected-reconstruction-error` bounds the
change in every finite reflected matrix under the recorded reconstruction,
including full-face support errors. The unnormalized face-action sums
retain the separate weighted-moment estimate
in {prf:ref}`lem-ym-bounded-recorded-gauge-words`.
{prf:ref}`lem-ym-native-qsd-translation` transports the QSD, its survival
weights, and both components of its regulator/gauge disintegration.
{prf:ref}`prop-ym-translated-qsd-reflection-test` gives an actual negative
physical reflected form for the specified translated finite-cutoff QSD
family. Its continuum use requires uniform versions of its bounds.
{prf:ref}`prop-ym-anchored-regulator-translation-test` excludes full
translation invariance on a limiting regulator retaining an absolute anchor.
For the globally normalized nonnegative physical readouts,
{prf:ref}`prop-ym-normalized-physical-translation-test` further proves
that a translation-invariant limiting law is zero. These conclusions fix
the scope of any subsequent relativistic identification without replacing
the selected law, its observable normalization, or its physical coordinates.
:::


## References and further reading

The finite constructions and analytic estimates are developed in
{doc}`01_fractal_set`, {doc}`02_causal_set_theory`, {doc}`03_lattice_qft`,
{doc}`04_standard_model`, and the convergence chapters cited above.
The perturbative group coefficient used in {ref}`(YM.26) <eq-fg-ym-26>` is the specialization of
{prf:ref}`cor-sm-beta-functions` to zero matter content.

The distinction between a classical Yang–Mills action and the quantum
existence-and-gap problem follows the
[Jaffe–Witten formulation](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf).
The analytic continuation requirements are those of
[Osterwalder and Schrader, *Axioms for Euclidean Green's functions II*](https://link.springer.com/article/10.1007/BF01608978).
