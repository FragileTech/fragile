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
coordinates. Its correlations use the complete recorded transition, including
companion selection, cloning, and kinetic transport.
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
{prf:ref}`def-partition-function-ym`. The native matter observables and their evolution
use the recorded CAR construction in {doc}`03_lattice_qft`. All uses of the transfer and reconstruction results
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


### 3.2. Recorded interaction connections


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

### 4.1. Recorded fitness balance

:::{prf:theorem} Exact stochastic balance of recorded observables
:label: thm-u1-noether-current

For the recorded chain with transition $P_h$ and an observable $Q$ integrable at each retained step,

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
For the chain let
$\Delta M_{k+1}=Q(S_{k+1})-(P_hQ)(S_k)$. Its conditional expectation given
$S_0,\ldots,S_k$ vanishes. Summing
$Q(S_{k+1})-Q(S_k)=\Delta M_{k+1}+(P_hQ-Q)(S_k)$ proves {ref}`(YM.8) <eq-fg-ym-8>`.

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


### 4.2. Recorded flow diagnostics


:::{prf:definition} Recorded flow diagnostics
:label: def-noether-flow-equations

The algorithmic flow diagnostic for an observable $Q$ is its one-step drift
$\mathcal D_hQ=(P_hQ-Q)/h$, or $LQ$ for an identified continuous generator.
A residual measured over $n$ steps is
$Q(S_n)-Q(S_0)-h\sum_{k<n}\mathcal D_hQ(S_k)$. By {ref}`(YM.8) <eq-fg-ym-8>` it is a
martingale residual, whose variance can be estimated from its conditional
increments.
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


:::{prf:theorem} Exact cancellation for disjoint native source likelihoods
:label: thm-ym-disjoint-source-cancellation

Use the native unselected execution law and its common limiting law in
{prf:ref}`thm-ym-continuum-source-action`, with the occupation identity of
{prf:ref}`thm-ym-occupation-source-identification`. Let $f,g$ be real
sources with disjoint supports in the recorded source coordinates
$(\tau,x)$. Retain the existing source currents $M_f,M_g$, energies
$E_{ff},E_{gg}$, and likelihoods

$$
L_f(a)=\exp(aM_f-a^2E_{ff}/2),\qquad
L_g(b)=\exp(bM_g-b^2E_{gg}/2).
$$

At finite size and on the same constructed limit, for every real $a,b$,

$$
E_{fg}=0,\qquad L_f(a)L_g(b)=L_{af+bg}(1),\qquad
\mathbb E[(L_f(a)-1)(L_g(b)-1)]=0.
$$

For $p,q\ge1$, their already defined likelihood derivatives
$H_{f,p}=\partial_a^pL_f(0)$ and $H_{g,q}=\partial_b^qL_g(0)$
therefore satisfy

$$
\mathbb EH_{f,p}=\mathbb EH_{g,q}=0,\qquad
\mathbb E[H_{f,p}H_{g,q}]=0.
$$

In particular $H_{f,1}=M_f$, $H_{g,1}=M_g$, and for every cut
$\mathcal F_k$ of the finite execution history both terms in the
total covariance formula vanish:

$$
\operatorname{Cov}\bigl(\mathbb E[M_f\mid\mathcal F_k],
                         \mathbb E[M_g\mid\mathcal F_k]\bigr)
=0,\qquad
\operatorname{Cov}(M_f,M_g\mid\mathcal F_k)=0
\quad\text{almost surely}.
$$

The same identities hold for $H_{f,p},H_{g,q}$. Thus their full
covariance is zero even when a variance-only bound is positive.
This evaluates the signed sum underlying
{prf:ref}`cor-lqft-full-history-locality-bound` for these existing source
variables. Their integrability follows from the likelihood derivative
envelopes, so the square-integrable martingale proof applies directly.

For the original descriptor $Y$ at finite size, or $\widehat Y$ in
the common limit, write $r_f(a)=\mathbb E[L_f(a)\mid Y]$ with the
corresponding descriptor understood. Its exact projected identity is

$$
\mathbb E[(r_f(a)-1)(r_g(b)-1)]
=-\mathbb E[(L_f(a)-r_f(a))(L_g(b)-r_g(b))].
$$

At first derivative order this is the existing information decomposition
specialized to disjoint sources:

$$
\mathbb E[J_fJ_g]
=-\mathbb E[(M_f-J_f)(M_g-J_g)],\qquad
J_f=\mathbb E[M_f\mid Y].
$$

For a complete lossless record retaining the source inputs and occupation,
the likelihoods are measurable in that record and the residuals vanish.
For the prescribed smaller gauge descriptor the displayed residual uses
that descriptor itself.

For the existing survival selection $\ell$, retain its normalization and
write $Z_f(a)=\mathbb E[\ell L_f(a)]/\mathbb E\ell$ and
$Z_{f,g}(a,b)=\mathbb E[\ell L_{af+bg}(1)]/\mathbb E\ell$.
Under $\mathbb P^\ell$, the normalized source densities obey instead
the exact formula

$$
\mathbb E_{\mathbb P^\ell}
 \left[\left(\frac{L_f(a)}{Z_f(a)}-1\right)
       \left(\frac{L_g(b)}{Z_g(b)}-1\right)\right]
=\frac{Z_{f,g}(a,b)}{Z_f(a)Z_g(b)}-1.
$$

*Proof.* Disjoint support gives $f\cdot g=0$ pointwise. The occupation
identity gives $E_{fg}=\kappa_h\zeta_{N,h}(f\cdot g)=0$ at finite size
and $E_{fg}=\kappa\zeta(f\cdot g)=0$ in the limit. Linearity of the
source current and expansion of the source energy now prove the
likelihood product identity pointwise. The already proved normalization
of the likelihood for the combined source gives expectation one for
this product; each separate likelihood also has expectation one.
Expanding their centered product proves its vanishing.

The derivative envelopes (YM.Z37) hold in every finite $L^p$ norm
on bounded parameter sets. Hölder's inequality therefore justifies
differentiating the product expectation to every finite order. Its
mixed derivatives vanish, giving the asserted derivative pairings and
their zero means. The same envelopes and the established joint source
limit justify these statements for the limiting likelihoods.

For the finite-history assertion, let $L_f^{\le k}(a)$ denote the
product of the already executed source likelihood factors through the
cut. Each next factor has conditional expectation one by the Gaussian
exponential identity with its predictable coefficient. Backward conditional
integration of future factors therefore gives

$$
\begin{aligned}
\mathbb E[L_f(a)\mid\mathcal F_k]&=L_f^{\le k}(a),\\
\mathbb E[L_f(a)L_g(b)\mid\mathcal F_k]
 &=L_{af+bg}^{\le k}(1)
  =L_f^{\le k}(a)L_g^{\le k}(b).
\end{aligned}
$$

The second equality uses the zero cross energy at every executed stage,
not independence of the adaptive coefficients. Conditional expectation
is an $L^p$ contraction, so the same derivative envelopes justify
differentiating these identities. Their mixed derivatives state
that the conditional covariance of $H_{f,p},H_{g,q}$ is zero almost
surely. Moreover, the two prefix likelihoods each have mean one and
their product has mean one by the same martingale normalization.
Differentiating this prefix identity proves that their conditional-mean
derivatives have covariance zero. This proves both asserted vanishings
without estimating either term by a positive variance bound.

Conditional-expectation residuals are orthogonal to every square-integrable
function of the descriptor. Decompose each centered likelihood into its
centered projection and residual. The cross terms have expectation zero,
so the zero raw pairing equals the sum of the two displayed pairings.
Differentiation gives the current identity. Lossless reconstruction makes
each likelihood an explicit function of the complete recorded inputs,
which proves its measurability assertion. Finally, retain the same
pointwise likelihood product under selection and divide by the two
normalizations. Each normalized density has selected expectation one;
expanding their product proves the last formula. $\square$
:::

:::{prf:theorem} Projected source covariance from the smaller physical gauge readout
:label: thm-ym-physical-gauge-score-projection

Use the unselected native source law of
{prf:ref}`thm-ym-disjoint-source-cancellation` and the prescribed smaller
physical gauge descriptor $Y_{\mathrm{phys}}$: the retained bounded
masked and normalized readouts in
{prf:ref}`thm-ym-native-physical-gauge-hierarchy`. All expectations below
use their joint law with the source currents, at finite size or on the
same constructed joint limit. Set

$$
J_f^{\mathrm{phys}}=\mathbb E[M_f\mid Y_{\mathrm{phys}}],\qquad
J_g^{\mathrm{phys}}=\mathbb E[M_g\mid Y_{\mathrm{phys}}].
$$

Enumerate the monomials in the retained countable gauge-coordinate
dictionary and its conjugates, and center them under this same law.
Let $p_1,\ldots,p_m$ be the first $m$ centered polynomials. They retain
the original masks, weights, and normalizations inside their coordinates.
Define the native moment matrix and source-response columns

$$
\begin{gathered}
(G_m)_{ij}=\mathbb E[\overline{p_i}p_j],\qquad
(d_{f,m})_i=\mathbb E[\overline{p_i}M_f]
 =\left.\frac{d}{da}\mathbb E[\overline{p_i}L_f(a)]\right|_{a=0},\\
J_{f,m}=\sum_{i=1}^m p_i(G_m^+d_{f,m})_i,\qquad
q_{f,m}=d_{f,m}^*G_m^+d_{f,m},\qquad
z_m=d_{f,m}^*G_m^+d_{g,m}.
\end{gathered}
$$

Here $G_m^+$ is the Moore--Penrose inverse of the finite recorded Gram
matrix, including its null modes. These polynomials are an expansion
of the fixed descriptor projection, not a reassignment of regional modes.
The exact evaluation of its cross coefficient is

$$
\begin{gathered}
J_{f,m}\longrightarrow J_f^{\mathrm{phys}}\quad\text{in }L^2,\qquad
q_{f,m}\uparrow q_f:=\mathbb E|J_f^{\mathrm{phys}}|^2,\\
z_m\longrightarrow z:=\mathbb E[J_f^{\mathrm{phys}}J_g^{\mathrm{phys}}],\qquad
|z-z_m|\le\sqrt{(q_f-q_{f,m})(q_g-q_{g,m})}.
\end{gathered}
$$

The matrices and response columns are respectively the native polynomial
moments and source derivatives already constructed in
{prf:ref}`thm-ym-native-physical-gauge-hierarchy` and
{prf:ref}`thm-ym-continuum-source-action`.

For disjoint source supports put
$a_f=\mathbb EM_f^2=\kappa_h\mathbb E\zeta_{N,h}(|f|^2)$ at finite
size, or $a_f=\kappa\mathbb E\zeta(|f|^2)$ in the limit, and similarly
for $g$. The occupation and information identities give

$$
\begin{gathered}
z=-\mathbb E[(M_f-J_f^{\mathrm{phys}})
             (M_g-J_g^{\mathrm{phys}})],\\
|z|\le\min\left\{\sqrt{q_fq_g},
                 \sqrt{(a_f-q_f)(a_g-q_g)}\right\}
\le\tfrac12\sqrt{a_fa_g},\\
|z-z_m|\le\sqrt{(a_f-q_{f,m})(a_g-q_{g,m})}.
\end{gathered}
$$

Thus the zero-coefficient calculation for this smaller descriptor is
the signed native-moment limit $\lim_m d_{f,m}^*G_m^+d_{g,m}$.
The occupation identity sets the raw cross coefficient to zero; its
projected coefficient is the displayed limit.

*Proof.* The retained bounded coordinate ranges have compact closure.
Cylinder polynomials in coordinates and their conjugates separate points
of that closure and contain constants. The polynomial density established
in {prf:ref}`thm-ym-native-physical-gauge-hierarchy`, followed by density
of continuous functions in $L^2$ of the retained Borel probability law,
shows that the centered polynomial spans are dense in
$L^2_0(\sigma(Y_{\mathrm{phys}}))$.

For a null vector $c$ of $G_m$, the polynomial $\sum_i c_ip_i$ is zero
in $L^2$, so $c^*d_{f,m}=0$. Hence $d_{f,m}$ lies in the range of
$G_m$. The normal equations $G_mc=d_{f,m}$ show that $J_{f,m}$ is
the orthogonal projection of $M_f$ onto this polynomial span; the
pseudoinverse chooses its coefficients without changing the projected
vector. The Gram formula gives its squared norm $q_{f,m}$ and mixed
inner product $z_m$. Increasing dense spans give the stated strong
convergence and monotone squared norms. Orthogonality eliminates the
cross terms between the span and its complement, yielding

$$
z-z_m=\langle J_f^{\mathrm{phys}}-J_{f,m},
                 J_g^{\mathrm{phys}}-J_{g,m}\rangle.
$$

Cauchy--Schwarz proves the error bound. Differentiation of the likelihood
expectation is justified by (YM.Z37) and boundedness of each finite
polynomial on the retained coordinate range, giving the native-response
formula for $d_{f,m}$.

The raw orthogonality theorem and orthogonality of conditional residuals
give the formula for $z$. Cauchy--Schwarz applied once to the projections
and once to the residuals gives the two bounds. For $a_f,a_g>0$, put
$x=q_f/a_f$, $y=q_g/a_g$. Both belong to $[0,1]$. If $x+y\le1$,
$\sqrt{xy}\le1/2$; otherwise
$\sqrt{(1-x)(1-y)}\le1/2$. This proves the factor $1/2$ bound.
If either raw variance is zero, all its projections and mixed coefficients
are zero. Finally $q_f\le a_f$, $q_g\le a_g$ gives the computable
occupation-based error bound. $\square$
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

For the latent kinetic update of
{prf:ref}`def-latent-fractal-gas-kinetic`, the same Gaussian mean-shift
operation has its own exact tangent. At a fixed pre-O history, write
$S=c_2G^{1/2}\Sigma_{\mathrm{reg}}$ for the recorded momentum-noise
amplitude and $u_f$ for the predictable shift of its standardized draw.
The unchanged latent O and following A maps give

$$
\begin{gathered}
p^O(\lambda)=c_1p+S(\xi+\lambda u_f),\qquad
v=G^{-1}p^O(0),\qquad
w=\tfrac h2\psi_v(v),\\
\dot p^O=Su_f,\qquad
\dot z^{\mathrm{end}}
=D\operatorname{Exp}_z(w)
 \left[\tfrac h2D\psi_v(v)G^{-1}Su_f\right].
\end{gathered}
$$

For $c=V_{\mathrm{alg}}$, $r=\|v\|_G>0$, and a tangent variation $a$,
the derivative in this formula is

$$
D\psi_v(v)a
=\frac{c}{c+r}a
 -\frac{c}{(c+r)^2}\frac{\langle v,a\rangle_G}{r}v,
\qquad D\psi_v(0)=I.
$$

Its radial eigenvalue is $c^2/(c+r)^2$ and its transverse eigenvalues
are $c/(c+r)$, so $\|D\psi_v(v)\|_G\le1$. The position endpoint is
still the squashed exponential-map endpoint in
{prf:ref}`cor-ym-squashed-kinetic-support`; the final B stage changes
momentum at fixed position. An earlier source also differentiates the
base point, metric, and coefficients through their actual preceding
updates. Both A stages use this squashed transport map.

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

For the latent branch, differentiate the displayed O map to obtain
$Su_f$. The pre-O base point and metric are fixed for this one-stage
derivative. Applying the chain rule to the following squashed A map
gives its displayed endpoint tangent. For $r>0$,
$Dr(v)a=\langle v,a\rangle_G/r$; differentiating
$\psi_v(v)=cv/(c+r)$ proves the derivative formula. Decomposing $a$
into its radial and transverse parts gives the eigenvalues. At zero,
$\psi_v(a)-a=-\|a\|_Ga/(c+\|a\|_G)=O(\|a\|_G^2)$,
so the derivative is the identity there as well. This calculation retains
the differential of the actual exponential map and introduces no linear
position update in its place.

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
## 8. Classical continuum consistency

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
quadratic trace estimate $|\operatorname{Tr}(AB)|\le r\|A\|_{\mathrm{op}}\|B\|_{\mathrm{op}}$.
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

:::

:::{prf:proof}
Substitute the three rescalings into {ref}`(YM.28) <eq-fg-ym-28>`
and {ref}`(YM.29) <eq-fg-ym-29>`.
:::

:::{prf:definition} Observable validation targets
:label: def-experimental-signatures

A validation record specifies the link lift and orientations, measured loop
traces, action normalization, time and length conversions, and the correlation
channel. It includes uncertainty from {ref}`(YM.21) <eq-fg-ym-21>` or {ref}`(YM.22) <eq-fg-ym-22>` when their law and
regularity hypotheses hold. Fitness residuals use {ref}`(YM.8) <eq-fg-ym-8>`. Reported chirality masks and scalar
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

To apply an LSI to the selected stationary process, identify that very
joint law. The following remark specifies the existing
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
to apply to that reference.
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
stationary laws already covered by the joint LSI.
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
smooth core; truncation extends it wherever both sides remain defined.

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
stationary joint-LSI family supplies the analytic bound.
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

:::{prf:lemma} Translation spectrum of the recorded pullback representation
:label: lem-ym-pullback-translation-spectrum

Apply {prf:ref}`thm-wightman-w1-fg` to a same-law, strongly continuous
spacetime translation action. On its recorded $L^2$ space, or the
translation-invariant observable $L^2$ subspace generated by a sigma
algebra, let $E$ be the joint spectral measure of the pullback translation
unitaries $U(a)$. Then complex conjugation $Jf=\overline f$ satisfies

$$
JU(a)J=U(a),\qquad JE(B)J=E(-B)
$$

for every Borel set $B$ in energy-momentum space. In particular the joint
spectral support is invariant under $p\mapsto-p$. If this same
representation satisfies the forward-cone spectrum condition, its spectral
support is $\{0\}$ and all its translation unitaries are the identity.
If it also has a unique translation-invariant vacuum vector, its Hilbert
space is one-dimensional.

The same conclusion holds for the existing exterior lift of these
unitaries on the even observable vacuum sector in
{prf:ref}`thm-ym-hk-record-instantiation`. Its regional mode spaces are
closed under complex conjugation, so their closed union $\mathcal K$
and its even exterior space inherit the conjugation commuting with the
lifted translations. In particular the full positive-energy HK vacuum
requirements cannot hold nontrivially for these specific lifted pullback
implementers either.

The Fractal Set record isomorphism of
{prf:ref}`prop-fractal-set-analytic-transfer` preserves these conclusions
by unitary conjugation.
:::

:::{prf:proof}
Pullback by a measurable transformation commutes with complex conjugation,
which proves the first identity. Write the joint spectral representation
with a fixed translation pairing as
$U(a)=\int e^{i\langle a,p\rangle}E(dp)$. Antiunitarity gives

$$
JU(a)J=\int e^{-i\langle a,p\rangle}JE(dp)J.
$$

Uniqueness of the joint spectral resolution, compared with $U(a)$, gives
$JE(B)J=E(-B)$. The forward cone intersects its negative only at zero.
Thus spectral support in the forward cone together with this symmetry
forces $E(\{0\})=I$, and the spectral representation gives $U(a)=I$.
Every vector is then translation invariant; uniqueness of the invariant
vacuum forces dimension one.

On an exterior sector define the inherited conjugation on dense wedges by
$J_\wedge(f_1\wedge\cdots\wedge f_k)
=Jf_1\wedge\cdots\wedge Jf_k$, extended antilinearly, and fix the
vacuum. The determinant inner product makes this map antiunitary.
Each regional $L^2_0$ space is closed under $J$, so this conjugation
preserves the even observable vacuum sector. The identity $JU(a)=U(a)J$
on each mode implies
$J_\wedge\Gamma_-(U(a))J_\wedge=\Gamma_-(U(a))$ on wedges,
hence on the completed sector. The same joint spectral calculation
therefore applies there. Unitary transport preserves the joint spectral
projections and the dimension of the invariant subspace.
:::

:::{prf:lemma} Recorded interventions and operational no-signaling
:label: lem-no-signaling-fg

For the latent kinetic step of
{prf:ref}`def-latent-fractal-gas-kinetic`, set $c=V_{\mathrm{alg}}$
and use its spatial metric $G$ and step clock $\tau$. On each A substep,
the position interpolation is timelike for

$$
\mathfrak g=-c^2d\tau^2+G.
$$

Here $G$ is the metric used by that exponential map, held fixed as a
metric field during the substep. Where the metric changes between stages,
the assertion applies to the corresponding piecewise metric. If the
O-stage noise amplitude is invertible and $c_2>0$, the support of the
initial slopes of its following A substep is the closed $G$-ball of radius
$c$. Its interior consists of timelike slopes; its boundary is the null
cone section. The cone and the given spatial metric determine the displayed
Lorentzian quadratic form uniquely. In a $G$-orthonormal frame its matrix
is $\operatorname{diag}(-c^2,1,\ldots,1)$.

A kinetic trajectory starts at the recorded post-cloning position. At a
subsequent cloning event the old trajectory terminates and the new kinetic
trajectory starts at the new post-cloning position. Thus this assertion
applies to the perturbative trajectories, without assigning a transport
segment to a clone replacement. The pre-cloning and post-cloning states
and the clone mask are the existing coordinates of
{prf:ref}`def-fractal-set-record-coverage`; their recovery uses
{prf:ref}`thm-fractal-set-lossless`. The clock in this assertion is the
clock of the specified latent kinetic update. Application to a physical
spacetime embedding uses its stated clock and spatial-metric correspondence.

In a finite acyclic update graph, every updated variable is a measurable
function of its parent variables and its assigned exogenous noise. If an
intervention changes none of the variables or noises in the ancestral set
of an observable, its value is unchanged under the common-noise coupling,
and hence its law is unchanged. This is a sufficient no-signaling test.

For the operational test, use the already defined local kinetic source of
{prf:ref}`thm-ym-metric-force-sources`, with its test $f$ supported in the
sender's specified region. The sender chooses its parameter $\lambda$.
The initial law and all fitness, companion, cloning, collision, and kinetic
rules other than the specified Gaussian source stay fixed as functions of
the current state. In particular this intervention gives the sender no
independent control of the global fitness array or of realized clone choices.
Let $D_B$ be the receiver's existing recorded observable descriptor and let
$\mathbb U$ be the unsourced native law. The existing complete source
likelihood is

$$
L_\lambda=\exp(\lambda M_f-\lambda^2E_f/2),\qquad
\mathbb E_{\mathbb U}L_\lambda=1.
$$

By {prf:ref}`thm-sm-path-descriptor-density`, the receiver marginal is
unchanged for this intervention precisely when

$$
\mathbb E_{\mathbb U}[L_\lambda\mid D_B]=1
\quad\text{almost surely}.
$$

No-signaling for this source family requires this identity for every
admissible message value $\lambda$. It concerns the receiver's marginal
law; equality of individual receiver realizations is sufficient but is
not required. Thus stochastic cross-region dependence alone neither proves
nor disproves operational signaling.

For the selected law prescribed upstream, write
$\ell=1$ or the same fixed survival indicator as in
{prf:ref}`thm-ym-metric-force-sources`, and
$d\mathbb P^\ell=\ell\,d\mathbb U/\mathbb U(\ell)$.
Where the selected sourced law is defined, its receiver density relative
to the unsourced selected receiver law is exactly

$$
\frac{\mathbb E_{\mathbb P^\ell}[L_\lambda\mid D_B]}
     {\mathbb E_{\mathbb P^\ell}L_\lambda}.
$$

Consequently the selected-law test is equality of its numerator and
denominator. This retains the actual survival normalization. For every
bounded receiver observable $O(D_B)$, the existing source derivative gives

$$
\left.\frac d{d\lambda}\right|_{0}
\mathbb E_{\mathbb P^\ell_\lambda}O
=\operatorname{Cov}_{\mathbb P^\ell}(O,M_f).
$$

Vanishing of these first derivatives is necessary for no-signaling;
the full likelihood identity tests finite messages.

There is an exact application to a single sourced O stage. Condition on its
complete pre-O history $\mathcal F_k$. Let $A$ be the predictable set of
slots on which the source is nonzero. A receiver descriptor measurable in
$\mathcal F_k$ and the O-stage Gaussian draws at slots outside $A$ has
unchanged conditional law under this one-stage source. This includes
receiver O-stage readouts before any later update mixes the slots. The
statement holds with the original pre-O companion and cloning records
retained, irrespective of their correlations.
:::

:::{prf:proof}
Write $u=\psi_v(G^{-1}p)$. The A interpolation is
$\gamma(s)=\operatorname{Exp}_z(su)$ for $0\le s\le h/2$.
Geodesic metric compatibility gives
$\|\dot\gamma(s)\|_{G(\gamma(s))}=\|u\|_{G(z)}<c$.
Consequently

$$
\mathfrak g((1,\dot\gamma),(1,\dot\gamma))
=-c^2+\|u\|_G^2<0.
$$

The B and O stages change momentum at fixed position. Both A segments
have the stated bound, including after an arbitrarily large finite
Gaussian momentum draw. Their concatenation is a piecewise timelike
kinetic path. In a common time-independent spatial metric its endpoints
satisfy $d_G(z_{\mathrm{start}},z_{\mathrm{end}})<ch$.

For any $u$ with $r=\|u\|_G<c$, the vector
$w=cu/(c-r)$ satisfies $\psi_v(w)=u$. Hence the radial map in
{prf:ref}`def-latent-velocity-squashing` maps the entire tangent space
onto the open speed ball. Conditional on the pre-O state, invertibility
of $c_2G^{1/2}\Sigma_{\mathrm{reg}}$ gives momentum full support.
Applying $G^{-1}$ and then $\psi_v$ gives precisely the asserted slope
support. This argument uses $G$ from the kinetic specification; it does
not identify it with the inverse adaptive-noise covariance.

To verify uniqueness, write a quadratic form with spatial restriction $G$
as $q(a,u)=\alpha a^2+2a b(u)+G(u,u)$. Its null values at
$(1,u)$ and $(1,-u)$ for every $\|u\|_G=c$ give $b(u)=0$
and $\alpha=-c^2$. The sphere spans the tangent space, so $b=0$.
Positive definiteness of $G$ proves Lorentzian signature. This identifies
the local metric and cone; the argument makes no flatness assertion.
The recorded clone mask specifies the trajectory endpoints described in
the statement. The speed bound and causal character of the kinetic paths
are pathwise, so restricting to a positive-probability survival event
preserves them. The full-support assertion above concerns the native
conditional O kernel before future survival selection.

For the ancestral statement, order the ancestral vertices topologically.
Source values and noises agree. If the parents of the next vertex agree,
the same update function and noise give the same value there. Induction
proves equality at the observable. This uses all actual dependencies,
including companion selection, fitness statistics, cloning, and kinetics.

For a bounded measurable receiver test $b$, the already established
likelihood and descriptor identities give

$$
\mathbb E_{\mathbb U_\lambda}b(D_B)
=\mathbb E_{\mathbb U}[L_\lambda b(D_B)]
=\mathbb E_{\mathbb U}
 [\mathbb E_{\mathbb U}(L_\lambda\mid D_B)b(D_B)].
$$

Equality with the unsourced expectation for all such $b$ is precisely
that the displayed conditional density is one. Equality for every message
value makes the receiver distribution independent of the chosen message.
Conversely, a failure of this density identity distinguishes the two
receiver distributions. This is a distributional test and does not require
the receiver to recover or control individual clone events.

For survival selection, changing the native law and then applying the
same selection gives

$$
\mathbb E_{\mathbb P^\ell_\lambda}b(D_B)
=\frac{\mathbb E_{\mathbb P^\ell}[L_\lambda b(D_B)]}
       {\mathbb E_{\mathbb P^\ell}L_\lambda}.
$$

Conditioning the numerator proves its receiver density. Differentiating
at zero uses $L_0=1$ and $L'_0=M_f$; the source exponential-moment bounds
in {prf:ref}`thm-ym-metric-force-sources` justify the derivative for bounded
$b$ and the fixed positive-probability selection. The quotient derivative
is the stated covariance. Unselected no-signaling and no-signaling after
global survival selection are therefore evaluated with their respective
normalizations.

At the single O stage the source likelihood factor is

$$
L_{k,\lambda}
=\prod_{i\in A}
 \exp\bigl(\lambda u_{ki}\cdot\xi_{ki}
                  -\lambda^2|u_{ki}|^2/2\bigr).
$$

The $u_{ki}$ and $A$ are fixed conditional on $\mathcal F_k$.
Conditional independence of the Gaussian draws and their exponential
normalization give

$$
\mathbb E[L_{k,\lambda}\mid
 \mathcal F_k,(\xi_{kj})_{j\notin A}]=1.
$$

Apply the tower identity to any specified receiver descriptor measurable
in these variables. This proves the claimed exact marginal invariance.
For later receiver readouts the complete likelihood identity already
includes the intervening fitness and cloning updates; their rules remain
fixed, while their state arguments follow the perturbed execution.
:::

:::{prf:corollary} Exact support bound for the squashed kinetic transition
:label: cor-ym-squashed-kinetic-support

Use the A--O--A stages already defined in
{prf:ref}`def-latent-fractal-gas-kinetic`, with $c=V_{\mathrm{alg}}$.
Condition on the complete pre-O record $Y$, whose position is $z$ and
whose momentum is $p$. The next transported position is exactly

$$
\begin{aligned}
p^+&=c_1p+c_2G^{1/2}(z)\Sigma_{\mathrm{reg}}(z,S)\xi,
       \qquad \xi\sim\mathcal N(0,I),\\
u^+&=\psi_v(G^{-1}(z)p^+),\\
z^+&=\operatorname{Exp}_z\!\left(\frac h2u^+\right).
\end{aligned}
$$

In particular its conditional transition probability is the native
Gaussian integral through this exact map:

$$
K_A(Y,D)=\int\mathbf1_D\!\left(
\operatorname{Exp}_z\!\left[\frac h2\psi_v\!\left(
G^{-1}(z)(c_1p+c_2G^{1/2}(z)\Sigma_{\mathrm{reg}}(z,S)\xi)
\right)\right]\right)\,\gamma_d(d\xi).
$$

Here $\gamma_d$ is the Gaussian draw already present in the O stage.
For the metric of this A substep,

$$
d_G(z,z^+)<\frac{ch}{2},\qquad
K_A\bigl(Y,\{w:d_G(z,w)\ge ch/2\}\bigr)=0.
$$

Both A stages obey this bound, including the one immediately after the
Gaussian draw. In a common time-independent $G$, a complete kinetic step
from its recorded post-cloning start satisfies
$d_G(z_{\mathrm{start}},z_{\mathrm{end}})<ch$.
With stage-dependent metrics, the recorded path is timelike on each
stage for the piecewise metric in {prf:ref}`lem-no-signaling-fg`.

*Proof.* For every finite value of the O-stage momentum, put
$r=\|G^{-1}p^+\|_G$. The specified squashing map gives

$$
\|u^+\|_G=\frac{cr}{c+r}<c.
$$

This includes every Gaussian draw except a null set; no bound on the
Gaussian momentum itself is needed. The exponential-map segment
$s\mapsto\operatorname{Exp}_z(su^+)$, $0\le s\le h/2$, has constant
metric speed $\|u^+\|_G$ by geodesic metric compatibility. Its length
is therefore less than $ch/2$, and distance is at most path length.
The integrand of $K_A$ is zero for the stated exterior set, proving the
conditional zero-probability identity. Applying the same calculation to
the first A stage and adding the two lengths proves the complete-step
bound; B and O do not move position. The Lorentzian norm of each tangent
is $-c^2+\|u\|_G^2<0$, proving the stagewise causal assertion.

These are identities and bounds for each recorded kinetic path.
Integration over the preceding native history, or restriction to the
algorithm's surviving paths, preserves them. At a clone replacement,
the trajectory restarts at the recorded post-cloning state as specified
in {prf:ref}`lem-no-signaling-fg`; the bound is applied to that kinetic
segment. $\square$
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

The bounded multiplication fields {ref}`(YM.35) <eq-fg-ym-35>` commute
for every pair of tests. The corresponding local von Neumann algebra
identity is proved for the native readouts in
{prf:ref}`thm-ym-native-multiplication-locality`.
The actual squashed kinetic support is calculated separately in
{prf:ref}`cor-ym-squashed-kinetic-support` from the prescribed update.
:::

:::{prf:proof}
For any $\psi\in L^2(\pi)$,
$\Phi_N(f)\Phi_N(g)\psi=\Phi_N(g)\Phi_N(f)\psi$ because scalar
multiplication commutes. This proves the finite statement for all supports.
The double-commutant proof in
{prf:ref}`thm-ym-native-multiplication-locality` extends this equality
to the native local algebras and their bounded spectral generators.
All operator products in this argument are defined on the whole
observable Hilbert space.
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


### 12.3. Continuum limits established by the framework


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

This evaluates the reflected product under the recorded law and
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
isometry.
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
The direct complex pair, determinant, and triangle marks also have $C_b=1$
by {prf:ref}`lem-ym-physical-gauge-word-uniform-integrability`.
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

:::{prf:lemma} Uniform integrability and reconstruction bounds for the full direct color algebra
:label: lem-ym-physical-gauge-word-uniform-integrability

Use the three-component readout, force/velocity time alignment, and masked
extension of {prf:ref}`def-sm-direct-observable-law`. Retain the full complex
$q_{ij}$, $b_{ijk}$, and $\Pi_{ijk}$ of
{prf:ref}`def-sm-direct-color-contractions`, their conjugates, and their
recorded masks. The selected record law is unchanged. The following bounds
hold for every record:

$$
|q_{ij}|\le m_im_j,\qquad
|b_{ijk}|\le m_im_jm_k,\qquad
|\Pi_{ijk}|\le m_im_jm_k.
$$

Consequently each of these complex marks has bound one; the valid triangle
defect has bound two. Every fixed polynomial word in bounded localized
coordinates and in the already specified normalized averages has a bound
independent of population size and step length. If
$F=\sum_\alpha a_\alpha\prod_{r=1}^{d_\alpha} z_{\alpha r}$,
where $|z_{\alpha r}|\le M_{\alpha r}$, its explicit bound is

$$
M_F=\sum_\alpha|a_\alpha|\prod_{r=1}^{d_\alpha}M_{\alpha r},
\qquad
\mathbb E_{\mathbb P_{N,h}}
 |\overline{F_i(\mathsf T_\vartheta Y)}F_j(Y)|^{1+\epsilon}
\le(M_{F_i}M_{F_j})^{1+\epsilon}
\quad(\epsilon>0).
$$

Here localization uses the existing recorded coordinates and requires all
vertices of the pair or triangle to lie in the specified half-space. For a
geometric average, the weights are exactly those in
{prf:ref}`thm-ym-native-normalized-gauge-hierarchy`. For a configured channel
average, they are exactly those in {prf:ref}`def-sm-direct-observable-law`,
including its own denominator and empty-average indicator. Neither average
replaces the other, and no bound for an unnormalized growing sum is inferred.

There is also an explicit mark estimate for the existing reconstruction.
For two masked color arrays in the same retained component frame, set
$e_i=\|\widehat c_i-c_i\|$. Then

$$
\begin{aligned}
|\widehat q_{ij}-q_{ij}|&\le e_i+e_j,\\
|\widehat b_{ijk}-b_{ijk}|&\le e_i+e_j+e_k,\\
|\widehat\Pi_{ijk}-\Pi_{ijk}|&\le2(e_i+e_j+e_k).
\end{aligned}
$$

These inequalities hold across validity changes, with the masked vectors
set to zero as in the original readout. Thus the mark error $\eta$ in
{prf:ref}`lem-ym-physical-reflected-reconstruction-error` is bounded,
respectively, by the actual weighted sums of these right-hand sides.
They introduce no inverse overlap or inverse anchor determinant.
:::

:::{prf:proof}
The masked colors have norm $m_i\in\{0,1\}$. Cauchy--Schwarz bounds
$q_{ij}$, Hadamard's determinant inequality bounds $b_{ijk}$, and
multiplication bounds $\Pi_{ijk}$. A nonnegative normalized average of
marks of modulus at most one has modulus at most one, including the
specified zero-denominator branch. Multiplying by support and validity
indicators can only reduce this bound. Tests supply their supremum norms.
The same calculation applies after geometric reflection of the recorded
support. Multiplication and the triangle inequality give $M_F$ and its
moment estimate under any of the specified probability laws, without a
survival-probability estimate.

For the reconstruction, expand
$\widehat c_i^\dagger\widehat c_j-c_i^\dagger c_j$
by changing one vector at a time. Each unchanged vector has norm at most
one. Multilinearity gives the corresponding three-term expansion of the
determinant; Hadamard bounds each term by the changed vector's norm.
Telescoping the three overlap factors of $\Pi$ then bounds its error by
$(e_i+e_j)+(e_j+e_k)+(e_k+e_i)$. The zero extensions have the same norm
bound, so the argument also covers a changed color mask. Other face,
selection, and geometry masks remain in the separate mask term of
{prf:ref}`lem-ym-physical-reflected-reconstruction-error`.
:::

:::{prf:theorem} Native physical hierarchy of the full recorded color invariants
:label: thm-ym-native-physical-gauge-hierarchy

Choose the existing direct three-component color branch of
{prf:ref}`def-sm-direct-observable-law` and its complete recorded law.
Retain the full Gram and complex determinant coordinates of
{prf:ref}`thm-sm-direct-orbit-isomorphism`, together with the recorded
positions, frame labels, validity data, and the configured readout weights.
Triangle coordinates are their existing products
$\Pi_{ijk}=q_{ij}q_{jk}q_{ki}$. Frames compared across recorded times use
exactly the common-frame or retained-transport convention of
{prf:ref}`thm-sm-direct-measure-isomorphism`.

Physical localization uses the position coordinates actually supplied with
these records and the pushforward of
{prf:ref}`prop-ym-recorded-physical-transformations`. This application
requires the chosen color readout and physical embedding to be available
on the same record. The algebraic estimates hold in any recorded position
dimension. A four-position-coordinate application additionally requires
its specified three-component color readout, as required in
{prf:ref}`def-sm-direct-observable-law`; the dimension-three color formula
alone supplies no such change of position dimension. Recorded time $\tau$
and the physical coordinate $x^0$ remain distinct.

For these observables the following statements hold.

1. The native law and all invariant history correlations are exactly those
   in {prf:ref}`thm-sm-direct-measure-isomorphism`. The full complex
   determinants are retained, including at rank-deficient configurations.
   On each finite mask stratum, polynomials in the Gram and determinant
   coordinates and their conjugates are uniformly dense in the continuous
   common-$SU(3)$ invariant color observables. The masks select the strata.

2. Include any omitted recorded coordinates by the descriptor refinement
   in {prf:ref}`prop-ym-density-and-support`. The density and predictive
   kernel remain the ones in
   {prf:ref}`thm-sm-path-descriptor-density` and
   {prf:ref}`thm-sm-effective-recorded-gauge-dynamics` for that refined
   descriptor. In particular companion choices, cloning, kinetic stages,
   normalization, and the prescribed survival or Doob weights remain
   inside the same likelihood. Refinement introduces no extra sampling.

3. For a fixed countable collection of bounded direct coordinates and
   configured normalized readouts, retain their values and their physical
   reflected values in {prf:ref}`thm-ym-native-fiber-continuum`. On a common
   further subsequence, every finite polynomial moment and every finite
   reflected product has a limit. This includes the full complex pair,
   determinant, and triangle channels jointly with the originally retained
   source and geometry coordinates.

4. The limit extends to continuous cylinders of these bounded coordinates.
   For localized normalized readouts it extends from a uniformly dense test
   dictionary to $C_0$ tests, hence to compactly supported smooth and
   Schwartz tests. Full-support masks are retained. Whenever a geometric
   reconstruction is used, its reflected-matrix error is the one in
   {prf:ref}`lem-ym-physical-reflected-reconstruction-error`, with the
   explicit color mark errors in
   {prf:ref}`lem-ym-physical-gauge-word-uniform-integrability`.

These statements identify the hierarchy of the specified direct invariant
observables. Its full labeled algebra has the exact negative reflected test
in {prf:ref}`prop-ym-native-labeled-color-reflection-sign` whenever the
specified localized diagonal channel survives. Thus the complete descriptor
is retained to compute the native law; admitting all its labeled
localizations into the physical future algebra is incompatible with that
nonzero channel and reflection positivity.
Full-coordinate orbit separation pertains to a common color
frame; local gauge covariance and a pure Yang--Mills force identity require
their native correspondence calculations. Normalized channel averages
retain their status as further pushforwards of the full coordinates.
:::

:::{prf:proof}
For a fixed mask pattern, remove the zero columns and apply
{prf:ref}`thm-sm-direct-orbit-isomorphism` to the remaining unit columns.
If every column is invalid, the color stratum is a single point. The
invariant image is compact. Coordinate polynomials contain constants,
are closed under conjugation, and separate points of this image; the
complex Stone--Weierstrass theorem therefore gives the stated uniform
approximation. There are finitely many mask patterns at a fixed finite
frame. Multiplication by their recorded indicators combines the
approximations with the maximum of their errors. No anchor inversion is
needed at a change of rank. Apply
{prf:ref}`thm-sm-direct-measure-isomorphism` on these strata to obtain
exact invariant moments under the selected law.

If $\widetilde D$ refines the old descriptor $D$, the already established
conditional density obeys

$$
\mathbb E_R\!\left[
 \mathbb E_R[\mathcal L\mid\widetilde D]\mid D\right]
=\mathbb E_R[\mathcal L\mid D].
$$

Thus its projection has exactly the old native law. The source-likelihood
conditional contraction used in
{prf:ref}`thm-ym-native-fiber-continuum` continues to apply. For a selected
finite window the full window likelihood is projected to its prefixes
before using the predictive-kernel ratio, precisely as in
{prf:ref}`thm-sm-effective-recorded-gauge-dynamics`. This preserves future
survival weights in prediction.

The preceding lemma supplies the compact bounds for the retained direct
coordinates and all specified normalized averages. Apply the existing
compact-coordinate subsequence construction simultaneously to their real
and imaginary parts, their reflected values, and the original coordinates.
Every polynomial in finitely many of them is bounded and continuous on
the compact coordinate range, so weak convergence passes its expectation.
This proves the third assertion without requiring continuity of a mask as
a function of an unretained raw variable.

Uniform polynomial approximation on a finite compact coordinate range
extends the moment functional to continuous cylinders. Approximate tests
in supremum norm and use the normalized-readout continuity estimate and
product telescoping in
{prf:ref}`thm-ym-native-normalized-gauge-hierarchy`. The same estimates
hold on the reflected side. This proves the fourth assertion. The stated
reconstruction estimate accounts for the remaining coordinate, weight,
mask, and cut errors when comparing with a geometric field readout.
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
$A-B$ in {ref}`(YM.Z89) <eq-fg-ym-z89>`.
:::

:::{prf:proposition} Physical reflection on the full native color algebra
:label: prop-ym-native-color-reflected-matrices

Use the hierarchy in {prf:ref}`thm-ym-native-physical-gauge-hierarchy`.
For each configured pair or triangle channel $A$, write $I$ for its
recorded tuple and $z_I^A$ for its complex mark $q$, $b$, or $\Pi$.
Write $\rho_I^A$ for its existing nonnegative normalized weight, with
validity included and $\sum_I\rho_I^A\le1$. Thus this notation can
express either of the existing normalized averages, with its own weights.
For its existing localization point $x_I$, the linear future readout is

$$
\Psi_A^+(f)=\sum_I\rho_I^A\chi_I^+z_I^A f(x_I).
$$

The physical pushforward transports the recorded tuple and its attached
mark, preserving its index order and component-frame labels. It acts on
each complex channel as in
{prf:ref}`prop-ym-recorded-physical-transformations`; OS conjugation is
applied once, to the reflected observable. For two such readouts the
actual reflected entry is therefore

$$
Q_{(A,f),(B,g)}
=\mathbb E_{\mathbb P_{N,h}}
 \sum_{I,J}\rho_I^A\rho_J^B\chi_I^-\chi_J^+
 \overline{z_I^A}z_J^B\,
 \overline{f(\vartheta x_I)}g(x_J).
$$

Tuples with a common recorded vertex contribute zero because their
full-support masks are incompatible. In particular, for determinant
channels the color factor is $\overline{b_I}b_J$, not
$|b_I||b_J|$ or a product of real parts. Mixed entries such as
$\overline{q_I}b_J$ remain in the same matrix. At a common color frame,
$\overline{b_I}b_J=\det(q_{i_a j_b})_{a,b=1}^3$ by
{prf:ref}`thm-sm-direct-orbit-isomorphism`; this identity does not remove
the complex determinants from mixed words or single insertions.

For an arbitrary finite future polynomial family, expand each word using
these same readouts and their conjugates. Every matrix entry is then a
finite linear combination of the joint moments identified in
{prf:ref}`thm-ym-native-physical-gauge-hierarchy`. Its expectation uses the
full selected likelihood and the actual geometry marginal:

$$
Q_{ij}
=\mathbb E_R[\mathcal L\,\overline{F_i(\mathsf T_\vartheta Y)}F_j(Y)]
=\int\lambda_{N,h}(dg)\int\mu_{N,h}^g(dd)\,
       \overline{F_i(\mathsf T_\vartheta(g,d))}F_j(g,d).
$$

Both equalities use the existing native law, including its prescribed
selection normalization. Neither conditional law is replaced by a product
of its physical half-space marginals.

Reflection positivity on continuous future cylinders of this bounded
hierarchy is equivalent to reflection positivity on its polynomial future
words. More quantitatively, if $P_i$ uniformly approximates $F_i$ on the
coordinate range by an error at most $\epsilon_i$, with
$\|F_i\|_\infty\le M_i$, then at every cutoff and at the common limit

$$
|Q(F)_{ij}-Q(P)_{ij}|
\le \epsilon_iM_j+(M_i+\epsilon_i)\epsilon_j.
$$

For any coefficient vector $c$, the quadratic-form error is at most

$$
2\left(\sum_i|c_i|M_i\right)
 \left(\sum_i|c_i|\epsilon_i\right)
+\left(\sum_i|c_i|\epsilon_i\right)^2.
$$

Thus a sign estimate for the native polynomial matrices extends to the
stated continuous observable class without an extra integrability
assumption. For these matrices the exact remaining sign test is
$C=C^*$ and $A-B\succeq0$ with the even and odd words of
{prf:ref}`prop-ym-native-physical-reflection-calculation`. The positive
ordinary Gram matrix of the color columns and the covariance matrix of
native channel increments in
{prf:ref}`thm-sm-effective-recorded-gauge-dynamics` refer to different
pairings; neither supplies that reflected sign estimate.
:::

:::{prf:proof}
Under the specified geometric pushforward, the future full-support mask
becomes $\chi_I^-$ and the test becomes $f(\vartheta x_I)$.
Conjugating the resulting complex readout conjugates both its mark and
its test. Multiplying by the second readout proves the displayed sum.
If a recorded vertex belongs to both tuples, its physical coordinate
would have to be simultaneously positive and negative, so the corresponding
term is zero. The determinant identity follows by expanding
$\det(C_I^\dagger C_J)$ with the same column orders as the recorded
baryons. It concerns the joint color array when that common frame is
specified.

Expansion of finite polynomials introduces only products of the retained
coordinates and their conjugates. Apply the native density identity and
geometry disintegration in
{prf:ref}`prop-ym-recorded-physical-transformations` to each bounded
product. This keeps every companion, cloning, kinetic, and selection
factor in $\mathcal L$ and proves the two same-law expressions. The
common-subsequence passage is the preceding hierarchy theorem.

For uniform approximation, use

$$
\overline{F_i^-}F_j^+-\overline{P_i^-}P_j^+
=\overline{F_i^--P_i^-}F_j^+
 +\overline{P_i^-}(F_j^+-P_j^+),
$$

where the superscripts denote reflected and unreflected evaluations.
The two terms have the claimed bounds since
$\|P_i\|_\infty\le M_i+\epsilon_i$. Integrate under the actual law,
then pass to its identified limit. Summing against
$\overline c_i c_j$ gives the quadratic-form bound. Uniform polynomial
approximation on the compact bounded-coordinate ranges makes this error
arbitrarily small, proving the equivalence of the two positivity tests.
Finally the even/odd identity applies to arbitrary complex cylinders,
including the determinant and mixed channels retained here.
:::

:::{prf:proposition} Exact reflection sign for a localized labeled color coordinate
:label: prop-ym-native-labeled-color-reflection-sign

Use the full labeled descriptor retained in
{prf:ref}`thm-ym-native-physical-gauge-hierarchy`, with its actual selected
record law and its specified geometric reflection. Fix a recorded vertex
label $i$ and a nonnegative $f\in C_c^\infty(\{x^0>0\})$. The diagonal
Gram coordinate is $q_{ii}=\|c_i\|^2=m_i$. Its bounded future readout and
its reflected evaluation are

$$
F_i^+=m_i\mathbf1_{\{x_i^0>0\}}f(x_i),\qquad
F_i^-=m_i\mathbf1_{\{x_i^0<0\}}f(\vartheta x_i).
$$

A missing recorded slot has $m_i=0$. Set
$u=\mathbb E_{\mathbb P_{N,h}}F_i^+$ and
$v=\mathbb E_{\mathbb P_{N,h}}F_i^-$. The physical reflected matrix of
$(1,F_i^+)$ is exactly

$$
Q=\begin{pmatrix}1&u\\v&0\end{pmatrix}.
$$

If $u+v>0$, the future word $G=F_i^+-(u+v)/2$ has strictly negative
physical reflected form:

$$
\mathbb E_{\mathbb P_{N,h}}
 [\overline{G(\mathsf T_\vartheta Y)}G(Y)]
=-\frac{(u+v)^2}{4}<0.
$$

This conclusion applies to the complete execution, survival-conditioned,
QSD-derived, or stationary law used to define $u,v$. It requires no
independence of opposite-side stages and no reflection invariance of the
law. Under reflection invariance $u=v$, and the same value is $-u^2$.
If $\mathbb P_{N,h}(m_i=1,\,x_i^0\ne0)>0$, some such compactly
supported $f$ has $u+v>0$.

On the common subsequence retaining these coordinates, let
$u_{N,h}\to u_*$ and $v_{N,h}\to v_*$. If $u_*+v_*>0$, the fixed
limiting future word $F_i^+-(u_*+v_*)/2$ has reflected form
$-(u_*+v_*)^2/4$. Hence a nonzero retained labeled channel of this kind
also prevents reflection positivity in that limit. In particular the
uniform bounds and convergence of the full descriptor cannot make it a
nontrivial physical OS future algebra containing these insertions.

This test concerns individually labeled localizations in the full
recorded algebra. A summed physical readout has cross-label terms, as
calculated in {prf:ref}`prop-ym-native-color-reflected-matrices`, and
its diagonal reflected entry need not vanish. Such an observable remains
a further pushforward of the full descriptor, in the sense of
{prf:ref}`thm-sm-direct-measure-isomorphism`.
:::

:::{prf:proof}
The same recorded vertex cannot satisfy $x_i^0>0$ and $x_i^0<0$.
Consequently $F_i^-F_i^+=0$ for every record, before any expectation.
The diagonal color identity follows from the original masked unit-vector
normalization. The three remaining matrix entries are the expectation of
one and the two one-point functions, proving the matrix formula.
For any real number $a$, direct expansion gives

$$
\mathbb E[(F_i^--a)(F_i^+-a)]=a^2-a(u+v).
$$

Its minimum occurs at $a=(u+v)/2$ and has the claimed negative value.
All factors are bounded, so the calculation applies under the selected
likelihood, with its companion, cloning, kinetic, and selection factors
unchanged. Conditioning on any retained interface also leaves the
pointwise zero product equal to zero.

Choose nonnegative smooth compact cutoffs on the positive half-space
increasing to one. Their values at $x_i$ and $\vartheta x_i$, with the
respective masks, increase jointly to
$m_i\mathbf1_{\{x_i^0\ne0\}}$. Monotone convergence proves the stated
existence of $f$. Finally the hierarchy theorem passes the bounded
one-point functions and the identically zero product to the same limit.
Expanding with the fixed constant $(u_*+v_*)/2$ proves the limiting
negative form. No population-uniform lower bound on $u+v$ is assumed
from a finite-cutoff nonzero value.
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
record is denoted by $\tau$. A translation of $x^0$ and a shift of
$\tau$ act on their respective recorded coordinates.

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

where the regions form a directed family and the generator sets are nested.
Fix one common Hilbert-space representation and one law for this definition;
the norm closure is then a unital $C^*$-algebra by isotony and directedness.
For self-adjoint unbounded smeared fields use their bounded spectral
functions, or their unitary exponentials when defined, as generators.
An unbounded operator itself is not an element of a von Neumann algebra
of bounded operators.

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

:::{div} feynman-prose
Start with the prescribed regional gauge readout. Decode a complete Fractal
Set, apply its original mask, and evaluate the observable. The result is
exactly the original readout by {prf:ref}`thm-fractal-set-lossless`; a face
retains its complete vertex mask. The unitary in
{prf:ref}`prop-fractal-set-analytic-transfer` carries its regional observable
space to the encoded space. Conjugating an already constructed operator by
this unitary preserves products and commutators. Reconstruction therefore
lets us carry an operator calculation through the record without changing
its regional assignment.

For the multiplication operators below, the calculation is pointwise:
either order multiplies by the same two scalar readouts. No independence
of the readouts is needed.

Now follow the actual kinetic transport. Both A stages use
$\operatorname{Exp}_z((h/2)\psi_v(G^{-1}p))$, including the stage after O.
However large the Gaussian momentum draw, the transported velocity remains
below $V_{\mathrm{alg}}$. This gives the kinetic support in
{prf:ref}`cor-ym-squashed-kinetic-support`, with cloning stages recorded
separately. The trajectory uses sampling time $\tau$; physical regions retain their
declared $x^0$ coordinate. Each transported operator identity retains its
specified representation and clock.
:::

:::{prf:theorem} Exact HK locality of the native multiplication representation
:label: thm-ym-native-multiplication-locality

Use the recorded fields and their multiplication representation from
{prf:ref}`def-wightman-field-fg`. For the physical gauge readout, use
exactly the existing map (YM.Z82)--(YM.Z83) and the bounded cylinders of
{prf:ref}`prop-ym-recorded-physical-transformations`, with their configured
full-face masks. Keep the native law and the observable space of that
readout. For its bounded regional generators, the local algebras in
{prf:ref}`def-local-algebra-fg` satisfy

$$
[\mathfrak A(O_1),\mathfrak A(O_2)]=0
$$

for spacelike separated regions. In this representation the identity
holds for every pair of regions. It holds at each recorded particle count
and horizon and on the native limiting probability law already constructed
in {prf:ref}`thm-ym-native-fiber-continuum`, for its retained bounded
cylinders. No independence, factorization, or new geometric assumption
is required for this commutator identity.

*Proof.* First insert the original Wilson readout itself. With its
configured full-face mask $\chi_P^O$ and a physical test $f$ in $O$, its
value is exactly

$$
\Phi_\omega^O(f)=\sum_{P\in\mathcal P(\omega)}
 \beta_P\chi_P^O(\omega)
 \left(1-r^{-1}\operatorname{Re}\operatorname{Tr}U_P(\omega)\right)
 f(x_P(\omega)).
$$

Here $\Phi_\omega^O$ denotes the existing masked readout, not an additional
field. For the corresponding readout in $V$, expanding the two orders
on each finite record gives

$$
\begin{aligned}
&\Phi_\omega^O(f)\Phi_\omega^V(g)
 -\Phi_\omega^V(g)\Phi_\omega^O(f)\\
&=\sum_{P,Q}\beta_P\beta_Q\chi_P^O\chi_Q^V
 \left(1-r^{-1}\operatorname{Re}\operatorname{Tr}U_P\right)
 \left(1-r^{-1}\operatorname{Re}\operatorname{Tr}U_Q\right)
 \bigl(f(x_P)g(x_Q)-g(x_Q)f(x_P)\bigr)=0.
\end{aligned}
$$

In the second product only the two finite summation indices have been
renamed. The order of links within each $U_P$ remains its original
recorded order. Their Wilson traces and the configured scalar weights
and masks commute as numbers; no commutation of the holonomy matrices
is asserted. All terms vanish individually, including terms whose faces
share vertices. The coordinates here are the physical $x$ of (YM.Z81),
with no substitution of sampling time for $x^0$.

A bounded cylinder of the actual readout is already the scalar
random variable

$$
F(\omega)=B(\Phi_\omega(f_1),\ldots,\Phi_\omega(f_m)).
$$

The multiplication realization is $(M_F\psi)(\omega)=F(\omega)\psi(\omega)$.
For two such actual regional readouts $F,G$ and every vector in the
existing observable space,

$$
([M_F,M_G]\psi)(\omega)
=\bigl(F(\omega)G(\omega)-G(\omega)F(\omega)\bigr)\psi(\omega)=0.
$$

This equality holds separately for every execution. Companion choices,
cloning, Gaussian draws, and the original selection weights can change
$F$, $G$, and their joint distribution, but not this product identity.
All masks remain inside $F,G$; no labeled occupancy is substituted for a
summed field. For complex invariant coordinates, their real and imaginary
parts and conjugates obey the same calculation. Boundedness makes both
products defined on the whole Hilbert space.

The existing observable space is invariant under these multipliers and
their adjoints because their products remain measurable in the same
readout sigma algebra. Its orthogonal complement is invariant as well:
for $u$ in that complement and $v$ in the observable space,
$\langle M_Fu,v\rangle=\langle u,M_{\overline F}v\rangle=0$.
Thus restriction preserves the identity without changing the
representation. For a real recorded field, every bounded Borel spectral
function is multiplication by that same function of its scalar value.
Consequently the spectral generators also commute.

Apply the two commutant inclusions proved in
{prf:ref}`thm-hk-locality-fg` to these now explicitly commuting bounded
generators. This gives the claimed von Neumann algebra identity.
The complete-record unitary of
{prf:ref}`prop-fractal-set-analytic-transfer` carries the identity to
its encoded realization by
$[\mathcal U M_F\mathcal U^{-1},\mathcal U M_G\mathcal U^{-1}]
=\mathcal U[M_F,M_G]\mathcal U^{-1}=0$.

Finally, on the already constructed native limiting law, the retained
bounded cylinder coordinates are again scalar random variables. The
same pointwise calculation proves their multiplication-operator identity
directly. This uses that existing limiting law and its original readout;
it does not infer operator convergence from correlation convergence.
$\square$
:::

:::{prf:remark} Dynamics and support in the recorded locality calculation
:label: rem-ym-hk-microcausality-verdict

The kinetic transition is the squashed update of
{prf:ref}`def-latent-fractal-gas-kinetic`. Its exact conditional support
is computed in {prf:ref}`cor-ym-squashed-kinetic-support` using the
recorded clock $\tau$ and metric $G$. Both A stages apply $\psi_v$
before transporting position. Its position transition is therefore the
pushforward through that squashed exponential map, with zero probability
outside the reachable kinetic ball.

The original recorded readouts have the multiplication-algebra identity
of {prf:ref}`thm-ym-native-multiplication-locality`. Recorded temporal
correlations use the complete native kernel in
{prf:ref}`thm-sm-instantiated-record-transition`; interventions use the
same-kernel source likelihood in {prf:ref}`lem-no-signaling-fg`.
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
continuous. Full Poincaré covariance uses the same physical implementers as the spectrum
condition in {prf:ref}`thm-wightman-w2-fg`. For the recorded pullback
representation of {prf:ref}`thm-wightman-w1-fg`, the mode, region, and
vacuum-sector calculation is {prf:ref}`thm-ym-hk-record-instantiation`,
and its translation spectrum is computed in
{prf:ref}`lem-ym-pullback-translation-spectrum`.
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

:::{prf:proposition} Signed locality coefficient of the original recorded gauge readouts
:label: prop-ym-native-signed-gauge-locality

Use the original bounded normalized gauge channels of
{prf:ref}`thm-ym-native-physical-gauge-hierarchy`, with the tuple marks,
weights, and localization points in
{prf:ref}`prop-ym-native-color-reflected-matrices`. For two disjoint
physical regions $O,V$, write their original full-face masks as
$\chi_I^O,\chi_J^V$. The readouts are

$$
F=\sum_I\rho_I^A\chi_I^O z_I^A f(x_I),\qquad
G=\sum_J\rho_J^B\chi_J^V z_J^B g(x_J).
$$

All expectations below use their same native execution law, including
its prescribed survival normalization. Their signed centered product is
exactly

$$
\begin{aligned}
\mathbb E[\overline{(F-\mathbb EF)}(G-\mathbb EG)]
={}&\mathbb E\sum_{I\cap J=\varnothing}
 \rho_I^A\rho_J^B\chi_I^O\chi_J^V
 \overline{z_I^A}z_J^B\,
 \overline{f(x_I)}g(x_J)
 -\overline{\mathbb EF}\,\mathbb EG.
\end{aligned}
$$

The condition $I\cap J=\varnothing$ concerns recorded vertices, including
their stage labels. Every omitted overlapping-tuple term is zero for
each execution. The retained weights and marks are jointly evaluated in
that execution. For the existing determinant channels,
$\overline{z_I^A}z_J^B=\overline{b_I}b_J
=\det(q_{i_a j_b})_{a,b=1}^3$ in their common color frame.
Thus the determinant identity retains the cross-tuple Gram entries.

The signed history calculation of
{prf:ref}`thm-lqft-inherited-history-covariance` evaluates the left-hand
side without absolute-value estimates:

$$
B_0+\sum_{j=1}^T\mathbb E[\overline{d_j^F}d_j^G]
=\mathbb E[\overline F G]-\overline{\mathbb EF}\,\mathbb EG.
$$

In particular, for its stationary native specialization
$F=A(S_{k+r})$, $G=B(S_{k+s})$, $0\le r\le s$, set
$a=A-\pi A$, $b=B-\pi B$ as in that theorem. The second term in the
conditional covariance formula evaluates to

$$
\begin{aligned}
\mathbb E\operatorname{Cov}(\overline F,G\mid\mathcal F_k)
&=\langle a,C_{s-r}b\rangle_\pi
  -\langle C_ra,C_sb\rangle_\pi,\\
B_k+\mathbb E\operatorname{Cov}(\overline F,G\mid\mathcal F_k)
&=\langle a,C_{s-r}b\rangle_\pi.
\end{aligned}
$$

Here $C_{s-r}$ is the original recorded contraction, with the original
cloning, companion, and kinetic stages. The indices $r,s$ count those
updates; the regions retain their physical recorded coordinates.
For the original modes in the stationary CAR application of
{prf:ref}`thm-ym-hk-record-instantiation`, division by
$\sigma_F\sigma_G$ gives its coefficient $s_{FG}$ when both variances
are positive. At one common recorded state the full-face formula above
therefore evaluates that coefficient directly. Zero-variance readouts
give the zero centered mode.

*Proof.* Expand the two finite recorded sums before taking expectation.
If $I,J$ share a recorded vertex, their two full-face masks require that
same vertex to belong to both $O$ and $V$. Their product is zero because
$O\cap V=\varnothing$. Remove precisely these terms, then subtract the
product of the two native means. The common-frame determinant identity
is the existing Gram identity in
{prf:ref}`thm-sm-direct-orbit-isomorphism`.

For the history formula, martingale increments at distinct update indices
are orthogonal by conditional expectation. Expanding the two terminal
martingale sums leaves their initial product and the displayed same-index
products. This is an equality of signed terms under the original law.

For the stationary specialization, condition first at $k+r$. The native
Markov property gives
$\mathbb E[\overline{a(S_{k+r})}b(S_{k+s})]
=\langle a,C_{s-r}b\rangle_\pi$.
Conditioning the two readouts at $k$ instead gives conditional means
$C_ra(S_k)$ and $C_sb(S_k)$. Subtract their expected product to obtain
the conditional covariance. Adding the already evaluated $B_k$ cancels
exactly that product, proving the second identity. For a selected finite
history, the first two formulas use the survival-weighted kernels and
initial law already given in
{prf:ref}`thm-lqft-inherited-history-covariance`. No stationary formula
is substituted for that selected history. $\square$
:::

:::{prf:theorem} Exact commutation test for the existing quantum regional algebras
:label: thm-ym-quantum-regional-commutation-test

Use exactly the regional mode spaces $\mathcal H_O,\mathcal H_V$, global
mode space $\mathcal K$, and even observable vacuum representation of
{prf:ref}`thm-ym-hk-record-instantiation`, with its same stationary law.
The following exhausts the possibilities for these two regional algebras.

1. If $\dim\mathcal K\le1$, they commute.
2. If $\dim\mathcal K=2$, they commute exactly when one regional mode
   space is zero or both regional mode spaces have dimension at most one.
3. If $\dim\mathcal K\ge3$, they commute exactly when
   $\mathcal H_O\perp\mathcal H_V$ or both regional mode spaces are the
   same one-dimensional subspace.

In the third case, whenever either regional mode space has dimension at
least two, the commutation test is therefore precisely

$$
\Pi_O\Pi_V=0
\quad\Longleftrightarrow\quad
\mathbb E_\pi[G\mid\sigma(q_O)]=\mathbb E_\pi G
\quad\text{for every bounded }\sigma(q_V)\text{-measurable }G.
$$

Equivalently, the two existing descriptor sigma algebras are independent
under this same $\pi$. This equivalence is a test of the prescribed
quantum net, not an independence assumption on the algorithm.

*Proof.* If $\dim\mathcal K\le1$, its even vacuum sector is the vacuum
line, so every restricted regional algebra is scalar. A zero regional
mode space likewise supplies only scalars in any dimension.

Suppose $\dim\mathcal K=2$. For a one-dimensional regional space with
unit mode $f$, its even CAR algebra is generated by $I,n_f$.
On $\Lambda^0\mathcal K\oplus\Lambda^2\mathcal K$, every such $n_f$
is zero on the vacuum and identity on $\Lambda^2\mathcal K$. Hence any
two such regional algebras commute. If one regional space instead equals
$\mathcal K$ and the other contains a unit mode $g$, choose a unit
$h\in\mathcal K$ orthogonal to $g$. The first regional algebra contains
$a^\dagger(g)a^\dagger(h)$ and the second contains $n_g$. The existing
CAR identities give

$$
[n_g,a^\dagger(g)a^\dagger(h)]=a^\dagger(g)a^\dagger(h),
\qquad
\|a^\dagger(g)a^\dagger(h)\Omega\|=1.
$$

Thus they do not commute, proving the second case.

Now let $\dim\mathcal K\ge3$. If the algebras commute, the exact
number-operator norm in {prf:ref}`thm-ym-hk-record-instantiation` implies
that every pair of unit modes $f\in\mathcal H_O$, $g\in\mathcal H_V$
has $|\langle f,g\rangle|\in\{0,1\}$. If all cross inner products
are zero, the spaces are orthogonal. Otherwise equality in
Cauchy--Schwarz gives a common unit mode $u$ after a phase adjustment.
If $\mathcal H_O$ contained a unit $v\perp u$, then
$(u+v)/\sqrt2\in\mathcal H_O$ would have inner product $1/\sqrt2$
with $u\in\mathcal H_V$, a contradiction. The same argument exchanges
the two regions. Consequently both spaces equal the line spanned by $u$.

Conversely orthogonal mode spaces have commuting even words by
{prf:ref}`thm-lqft-record-locality-defect`; the existing commutant
argument extends this to their von Neumann algebras. If both spaces are
the same line, both even algebras are generated by the same projection
$n_u$, and commute. This proves the third case.

Finally, the existing formula for $\Pi_O$ is conditional expectation
on centered modes. Apply it to $G-\pi G$ to obtain the displayed
equivalence. Taking $G$ to be an indicator of an event in $\sigma(q_V)$
and integrating over any event in $\sigma(q_O)$ gives the product rule
for their probabilities. Conversely that product rule first gives
orthogonality for centered indicators, then for simple functions, and
finally for their $L^2$ closures. All conditional expectations, modes,
and CAR operators here belong to the already specified quantum net.
$\square$
:::

:::{prf:theorem} Separate local-net application to the recorded CAR representation
:label: thm-ym-hk-record-instantiation

Use the existing spacetime episode map and comparison geometry of
{prf:ref}`assm-cst-continuum-geometry`, with the geometric conclusions
proved for that map in {prf:ref}`thm-fractal-faithful-embedding` and
{prf:ref}`cor-continuum-consistency-conditional`. A region $O$ supplies
its actual recorded descriptor $q_O$. Use the conservative stationary law
$\pi$, mode spaces $\mathcal H_O$, and CAR representation of
{prf:ref}`def-lqft-record-fock-space` and
{prf:ref}`thm-lqft-record-locality-defect`. A QSD application uses the
specified conservative stationary Doob law when invoking these stationary
results; a finite survival window retains the different law in
{prf:ref}`prop-ym-qsd-history-identification`.

For this application, the generators in
{prf:ref}`def-local-algebra-fg` are all bounded even CAR words with modes
in $\mathcal H_O$. Write $\mathfrak A_{\mathrm{ev}}(O)$ for their
von Neumann algebra in the existing Fock representation. Regions are
directed, and their descriptor sigma algebras are nested. Let

$$
\mathcal K=
\overline{\bigcup_O\mathcal H_O},\qquad
\mathcal H_{\mathrm{vac}}
=\overline{\bigcup_O\mathfrak A_{\mathrm{ev}}(O)\Omega}.
$$

These are subspaces of the already constructed one-mode and Fock spaces.
The observable vacuum representation is the restriction to
$\mathcal H_{\mathrm{vac}}$. Its HK inputs are as follows.

**Isotony and the vacuum sector.**
The local algebras are isotone, and

$$
\mathcal H_{\mathrm{vac}}
=\bigoplus_{k\ge0}\Lambda^{2k}\mathcal K.
$$

This subspace reduces every local observable algebra. The vacuum state is
normalized and positive, and $\Omega$ is cyclic for the global observable
algebra on this subspace. On the full Fock space, even observables cannot
make $\Omega$ cyclic whenever its odd sector is nonzero.

**Locality with the original regional modes.**
For even words $X=x_1\cdots x_p$ in $O$ and $Y=y_1\cdots y_q$ in $V$,
with modes $f_i\in\mathcal H_O$ and $g_j\in\mathcal H_V$, the existing
locality estimate specializes to

$$
\|[X,Y]|_{\mathcal H_{\mathrm{vac}}}\|
\le\sum_{i,j}|\langle f_i,g_j\rangle_\pi|
 \prod_{\ell\ne i}\|f_\ell\|
 \prod_{r\ne j}\|g_r\|.
$$

For unit modes $f\in\mathcal H_O$ and $g\in\mathcal H_V$, let
$n_f=a^\dagger(f)a(f)$, $n_g=a^\dagger(g)a(g)$, and
$s=\langle f,g\rangle_\pi$. These are even local generators and obey

$$
[n_f,n_g]
=s\,a^\dagger(f)a(g)-\overline s\,a^\dagger(g)a(f).
$$

If $\dim\mathcal K\ge3$, their norm in the observable vacuum sector is

$$
\|[n_f,n_g]|_{\mathcal H_{\mathrm{vac}}}\|
=|s|\sqrt{1-|s|^2}.
$$

Hence $0<|s|<1$ is an explicit failure of observable locality if these
two descriptors are assigned spacelike separated regions. This checks
even observables themselves, rather than inferring their noncommutation
from a nonzero odd anticommutator.

The existing fermionic lift in
{prf:ref}`cor-sm-direct-fock-isomorphism` transports these same operators
and coefficients: for its one-mode unitary $V$,

$$
\langle Vf,Vg\rangle=\langle f,g\rangle,\qquad
\Gamma_-(V)[n_f,n_g]\Gamma_-(V)^{-1}=[n_{Vf},n_{Vg}].
$$

Thus encoding the modes preserves the computed commutator, including its
norm. The multiplication identity in
{prf:ref}`thm-ym-native-multiplication-locality` applies to $M_F,M_G$;
it cannot be substituted for this CAR identity. This follows from the
already established product distinction in
{prf:ref}`thm-lqft-product-obstruction`.

The causal episode map specifies which regions are spacelike; the
operator commutators use these covariance coefficients.
For a split at an existing recorded history, the inherited contribution
to each coefficient is evaluated by the backward native-kernel integrals
in {prf:ref}`thm-lqft-inherited-history-covariance`. That theorem gives
its exact update contributions and its bound by the two predictable
variances, retaining the chosen survival law.
The substitution for the full covariance, including its conditional
remainder, and the resulting even-observable norm bound are proved in
{prf:ref}`cor-lqft-full-history-locality-bound`.

**Covariance of this representation.**
For an actual same-law symmetry $g$ satisfying
{prf:ref}`thm-wightman-w1-fg`, suppose its already defined physical action
transports the regional descriptor sigma algebras onto those of $gO$.
Then its one-mode unitary $U_g$ satisfies

$$
U_g\mathcal H_O=\mathcal H_{gO},\qquad
U_g\Pi_OU_g^{-1}=\Pi_{gO}.
$$

Its existing exterior lift $\Gamma_-(U_g)$ preserves
$\mathcal H_{\mathrm{vac}}$, fixes $\Omega$, and implements the covariance
identity in {prf:ref}`thm-hk-covariance-fg` for the restricted local net.
Strong continuity on the one-mode space implies strong continuity here.
The transported-background QSD identity in
{prf:ref}`lem-ym-native-qsd-translation` instead gives covariance between
the corresponding law spaces. It gives a symmetry on one such space
when the transformed data and selected law agree with the original ones.

**Dynamics and spectrum.**
The actual recorded CAR channel remains the map of
{prf:ref}`thm-lqft-record-car-channel`. Its computed multiplicativity
defect is

$$
\mathcal Q_t(a(f)a^\dagger(g))
-\mathcal Q_t(a(f))\mathcal Q_t(a^\dagger(g))
=\bigl(\langle f,g\rangle-\langle C_tf,C_tg\rangle\bigr)I.
$$

Consequently this prescribed map gives CAR automorphisms when $C_t$ is
unitary; the recorded contraction formula alone gives channels. The physical spectrum assertion of {prf:ref}`thm-hk-spectrum-fg` uses the
generators of the same physical translation implementers as the covariance
assertion. The pullback implementers specifically have the spectral
restriction in {prf:ref}`lem-ym-pullback-translation-spectrum`.
:::

:::{prf:proof}
Nested descriptor sigma algebras give nested centered $L^2$ mode spaces.
Their even word sets are nested, so
{prf:ref}`thm-hk-isotony-fg` applies to their double commutants.
Directedness makes $\bigcup_O\mathcal H_O$ a linear subspace and makes
the union of the local algebras an algebra.

Every local even word preserves parity and the Fock subspace over
$\mathcal K$. The projections onto these two subspaces commute with all
local generators, hence with their double commutants. Therefore all local
algebra vectors $A\Omega$ lie in the even Fock subspace over $\mathcal K$.
Conversely an even wedge with modes in $\bigcup_O\mathcal H_O$ is produced
from $\Omega$ by a product of creation operators in one common region,
using directedness. Such wedges span a dense subspace of
$\bigoplus_k\Lambda^{2k}\mathcal K$. This proves the sector identity.
For $A$ in any local algebra and $B\Omega$ in the defining union, choose a
region containing both supports. Then $AB\Omega$ and $A^*B\Omega$ lie
in the same union. Boundedness extends both invariances to its closure,
so the subspace is reducing. Positivity follows from
$\omega(A^*A)=\|A\Omega\|^2$, and cyclicity follows from the definition
and the proved sector identity.

Apply the exact word estimate (LQ.S3) in
{prf:ref}`thm-lqft-record-locality-defect` with even $p,q$, then restrict
to the reducing vacuum sector. Its operator norm cannot increase.
Orthogonal mode spaces therefore give commuting even generators, and
{prf:ref}`thm-hk-locality-fg` passes this to the von Neumann algebras.


For the number operators, apply the CAR once to each product; their
four-operator terms cancel, giving the displayed commutator. On the
one-mode space it is the commutator of the rank-one projections onto
$f$ and $g$. In their two-dimensional span, writing
$g=sf+\sqrt{1-|s|^2}e$ for $0<|s|<1$ gives eigenvalues
$\pm i|s|\sqrt{1-|s|^2}$. On Fock space over this span the commutator
vanishes in degree zero and degree two, so its norm is the same value.
Spectator modes do not change its norm. If $\dim\mathcal K\ge3$,
choose a unit spectator orthogonal to $f,g$ and wedge it with the
one-mode test vectors; this realizes the same norm in degree two of
$\mathcal H_{\mathrm{vac}}$. The cases $s=0$ and $|s|=1$ give zero
directly. This proves the exact even-observable locality test.

The same-law pullback unitary preserves means and maps functions of $q_O$
onto functions of $q_{gO}$. It therefore maps the centered subspaces onto
each other. Unitary conjugation of their orthogonal projections proves the
projection identity. The existing CAR lift sends $a^\dagger(f)$ to
$a^\dagger(U_gf)$ and preserves even word degree. Conjugation thus maps
local double commutants onto their transformed counterparts and fixes the
vacuum. It preserves their cyclic sector by its defining formula. Strong
continuity follows first on each finite wedge, then on finite sector sums,
and then on the Fock space by unitary norm bounds. This verifies the
hypotheses used in {prf:ref}`thm-hk-covariance-fg` for precisely the
same-law symmetries specified in the statement.

The channel defect is the established identity (LQ.C3) applied in this
representation.
:::

:::{prf:remark} Dependency order of the Yang--Mills comparisons
:label: rem-ym-proof-dependency-order

The record codec precedes the invariant-coordinate maps and the implemented
kernel calculation {ref}`(SM.K1) <eq-fg-sm-k1>`. Its complete history law precedes the exact
fermionic word evolution {ref}`(LQ.R1) <eq-fg-lq-r1>`--{ref}`(LQ.R4) <eq-fg-lq-r4>`. The analytical inputs are the
independently proved law-specific convergence, LSI, regularity, and
ellipticity results. Their use in {ref}`(YM.F1) <eq-fg-ym-f1>`--{ref}`(YM.F12) <eq-fg-ym-f12>` retains the original
state, observation domain, and update convention.

:::

(sec-ym-algorithmic-qft-synthesis)=
## 14. The connected algorithmic field theory

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

:::

:::{prf:theorem} Algorithmic QFT from recorded evolution and native bounds
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
   survival-conditioned history weight. The full complex color invariant algebra and its bounded physical
   reflected words belong to one native subsequence by
   {prf:ref}`thm-ym-native-physical-gauge-hierarchy`; their explicit
   uniform-integrability and reconstruction bounds are
   {prf:ref}`lem-ym-physical-gauge-word-uniform-integrability`.
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
history.
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
closure is established. The separate recorded CAR local-net application is
{prf:ref}`thm-ym-hk-record-instantiation`. It establishes isotony,
positivity, and global cyclicity in the even observable vacuum sector,
uses the existing regional covariance coefficients for locality, and
lifts the actual same-law symmetries to that sector. The recorded channel retains the complete native transition.
The native multiplication representation satisfies HK locality by
{prf:ref}`thm-ym-native-multiplication-locality`, directly for the
configured recorded readouts and their law. The prescribed kinetic
transition has the exact squashed support in
{prf:ref}`cor-ym-squashed-kinetic-support`. Its dynamics and clocks
are retained as specified in
{prf:ref}`rem-ym-hk-microcausality-verdict`.
{prf:ref}`lem-ym-pullback-translation-spectrum` checks the physical
spectrum condition specifically for the pullback implementers.

For the full direct color algebra,
{prf:ref}`thm-ym-native-physical-gauge-hierarchy` retains the complex Gram,
determinant, and triangle observables with their masks and native selected
law on one common subsequence.
{prf:ref}`lem-ym-physical-gauge-word-uniform-integrability` supplies their
explicit moment and mark-reconstruction bounds.
{prf:ref}`prop-ym-native-color-reflected-matrices` identifies all mixed
physical reflected entries and extends a polynomial sign estimate to
continuous cylinders. This application uses the existing invariant-coordinate
and native predictive-kernel results directly. The color orbit identification
is the common-frame $SU(3)$ identification of the Standard Model chapter.
The exact test in {prf:ref}`prop-ym-native-labeled-color-reflection-sign`
excludes reflection positivity for the full labeled future algebra when
its specified nonzero localized channel survives. Its role as a complete
native descriptor is distinct from selecting the physical observable
pushforward on which OS reconstruction is sought.

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

The distinction between a classical Yang–Mills action and the quantum
existence-and-gap problem follows the
[Jaffe–Witten formulation](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf).
The analytic continuation requirements are those of
[Osterwalder and Schrader, *Axioms for Euclidean Green's functions II*](https://link.springer.com/article/10.1007/BF01608978).
