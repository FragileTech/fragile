(sec-structural-landscape-convergence)=
# Structural landscape estimates and quantitative convergence

(sec-slc-ledger)=
## 1. The complete update and its structural data

:::{div} feynman-prose
Keep the machine fixed and change the landscape. Which parts of the argument survive, and which constants change? Answering that question requires a ledger: the force bounds available in each region, the opportunities for selection, and the cost of leaving those regions. These are analytical properties to establish from the function and the update rules. A simulation may help inspect them, but does not define them. We begin by specifying exactly what one complete update does so that every later estimate concerns the same gas.
:::

:::{prf:definition} Process, partition and normalization
:label: def-slc-process

Fix the canonical update of {doc}`02_euclidean_gas`, its population size $N$,
timestep $h>0$, and parameters $\theta$. A state $S$ retains position, capped
velocity, alive/dead status and every coordinate used by revival and collisions.
Write $P=P_CP_K$ for the complete conservative kernel. For a killed configuration
write $Q$ for the sub-Markov kernel, with extinction time $\tau_\dagger$; use its
absorbing probability extension when discussing stopping events. None of these
kernels is replaced by a continuous-time surrogate.

Let $F=-\nabla U$ where this gradient is defined; more generally the configured
force must be a specified finite measurable map on every point reached by the
update. Reward $R$, including its transformation from error $U$, is separate data.
Undefined force evaluations are not an infinite convergence constant: they fail
to specify a process.

Declare disjoint Borel position regions $B_1,\ldots,B_m,T,E$ covering the physical
space: basin regions, a transition region, and an exterior region. For a killed
configuration distinguish these regions from the valid domain $D$ and count only
alive walkers when specified. These supplied position regions are not presumed
to be attraction basins of the nonlinear population map.

All rates below are per complete update unless stated otherwise. An $n$-update
bound corresponds to time $nh$ and to the total charged work of those $n$ updates
on $N$ walkers; it is not a population-independent computational cost.
:::

:::{prf:definition} Configured parameters and deterministic analysis choices
:label: def-slc-parameter-register

The canonical parameter record used here is

$$
\theta=(d,N,h,\gamma,b_O,\sigma_x,\sigma_J,V_{\max},\alpha_{\rm col},
 R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg},\epsilon_D,\epsilon_C,
 \delta_D,A_r,A_s,\eta_r,\eta_s,p_r,p_s,\sigma_r,\sigma_s,s_c,\epsilon_c;
 U,R,D).
$$

Here $d,N\ge1$, $h>0$, $\gamma\ge0$, $b_O,\sigma_x,\sigma_J\ge0$ are the
thermostat factor and two other Gaussian amplitudes, $V_{\max}>0$ is the cap,
and $\alpha_{\rm col}$ is the collision multiplier. The feature radii, metric
weight, fitness floors, amplitudes, standardization regularizers
and acceptance scale $s_c$ are positive; companion widths belong to
$(0,\infty]$ with the uniform convention of
{prf:ref}`def-eg-frozen-measurements`; $p_r,p_s\ge0$ and
$\delta_D,\epsilon_c\ge0$.
The stationary-noise and active-diversity results impose their additional strict
positivity explicitly. The reward function $R$ may differ from $-U$. No viscosity,
donor history, population-dependent force or alternative noise normalization is
silently included in this record.

Use the actual standard deviations and collision bound

$$
\begin{aligned}
c&=h/2,& a&=e^{-\gamma h},& B&=c(1+a),& \eta&=c^2(1+a),\\
q^2&=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}
&s^2&=\sigma_x^2h,& V_c&=(1+2|\alpha_{\rm col}|)V_{\max}.
\end{aligned}
$$

The letter $B$ in a kinetic formula denotes this drift coefficient; basin sets
and noise matrices are not that coefficient. The cap is
$C_V(v)=V_{\max}v/(V_{\max}+|v|)$. Its radial/tangential derivative eigenvalues
are $V_{\max}^2/(V_{\max}+|v|)^2$ and $V_{\max}/(V_{\max}+|v|)$, respectively;
continuity at zero gives the 1-Lipschitz bound. The collision rule
$v_i^c=\bar v+\alpha_{\rm col}O(v_i-\bar v)$ has
$|v_i^c|\le V_{\max}+2|\alpha_{\rm col}|V_{\max}=V_c$, since $O$ is orthogonal.

For a radius $r>0$ put

$$
v_d(r)=\frac{\pi^{d/2}r^d}{\Gamma(1+d/2)},\qquad
G_d(t)=\frac1{\Gamma(d/2)}\int_0^{t^2/2}u^{d/2-1}e^{-u}\,du.
$$

Thus $|B(0,r)|=v_d(r)$, $p_J=G_d(J/\sigma_J)$ for $\sigma_J>0$ and $p_J=1$
for $\sigma_J=0$, and $p_O=G_d(G)$ for a standard OU innovation cutoff $G$.
Analysis choices (regions, cutoffs, matrix weights, tolerances and test functions)
are declared arguments of a certificate, not configured changes to the algorithm.
A bound may be independent of some entries of $\theta$; that independence is
part of its conclusion, not an omitted dependency.
:::

:::{prf:definition} Landscape, algorithm and response profiles
:label: def-slc-profiles

For a Borel region $A$, radii $r\ge0$, $L,k\ge0$, define extended-real profiles

$$
\begin{aligned}
M_A&=\sup_{x\in A}|F(x)|,&
\omega_A(r)&=\sup_{x,y\in A,\ |x-y|\le r}|F(x)-F(y)|,\\
D_A(L,r)&=[\omega_A(r)-Lr]_+,&
b_A(k)&=\sup_{x\in A}[x\cdot F(x)+k|x|^2]_+,\\
J_A(k,r)&=\sup_{x,y\in A,\ |x-y|\le r}
[\langle x-y,F(x)-F(y)\rangle+k|x-y|^2]_+.
\end{aligned}
$$

An empty supremum is zero. Reward oscillation, distance to $D^c$, and volumes of
landing subsets are further landscape descriptors. No convexity is built into
these definitions. $b_A$ measures radial confinement; $J_A$ measures pairwise
restoration. They are different properties.

Algorithm data include friction, noise scales, companion kernels, normalization
regularizers, fitness exponents, acceptance, jitter, collisions and cap. Response
data include donor probabilities, error-weighted cloning pressure, conditional
excursion probabilities and the certificates constructed below. A profile is
analytical data even when a numerical run estimates it. A finite sample maximum
is not automatically an upper bound on a global supremum.
:::

:::{prf:remark} Dependency ledger
:label: rem-slc-ledger

| Existing input | Quantitative replacement or retained certificate | Dependencies and use |
|---|---|---|
| Force Lipschitz and boundedness in {prf:ref}`axiom-confining-potential` | $\omega_A,D_A,M_A$ and excursion charges | Region, scale, force, $h$; kinetic comparison |
| Confinement | Full-update inward selection flux, regional adverse transfers and Gaussian tail envelopes; radial $b_A(k)$ is one optional route | Donor coverage, fitness gaps, force growth and explicit failure defects; no convex-at-infinity requirement |
| Pairwise restoring estimates | $J_A(k,r)$, independently of radial coercivity | Local relaxation and coupling |
| Diffusion/friction assumptions in {doc}`05_kinetic_contraction` | Actual Gaussian covariance and friction; positive lower noise only in smoothing/minorization results | Noise degeneracy can set a minorization certificate to zero |
| Earlier Keystone coverage premises | Already discharged constants of {prf:ref}`thm-keystone-discharged-averaged-pressure`; regional versions below | Entering geometry, fitness and companion parameters; includes $N^{-2}$ correction |
| Revival and positive alive mass | Actual eligible-donor mass; {prf:ref}`cor-mean-field-positive-alive-mass` where applicable | Boundary/initialization and parameters; not a substitute for feedback sensitivity |
| Composed drift in {prf:ref}`thm-foster-lyapunov-main` | Comparison matrices plus propagated defect vectors | Full update and intermediate states |
| TV mixing in {prf:ref}`thm-convergence-conservative-harris` | Drift, common mass, explicit rates below | Full finite-$N$ law, generally $N$-dependent |
| Mean-field consistency | Constants of {prf:ref}`thm-chaos-canonical-conditional-variance` and {prf:ref}`thm-chaos-canonical-quantitative-bias` | Their canonical regime, moment and donor hypotheses |
| Long-time population approximation | Phase-specific attraction, residence and transition estimates; optionally a matching two-swarm contraction | Same initial-law trajectory; distinct phases may persist |

This chapter replaces quantitative uses of assumptions, not the basic measurability
and integrability needed for their equations. Extended-real profiles specify when
a certificate gives no finite bound. Known finite-particle estimates remain usable
without claiming that every constant is uniform in $N$.
:::

:::{prf:proposition} Quantitative domain restrictions already implied by the source axioms
:label: prop-slc-source-domain-restrictions

The following consequences concern the original source conditions, without
altering them or the algorithm.

1. Parts 2 and 3 of {prf:ref}`axiom-confining-potential`, with their
   constants $\alpha_U>0$, $R_U<\infty$ and $F_{\max}<\infty$, imply

   $$
   |x|\le R_{\mathrm{axiom}}
   :=\frac{F_{\max}+\sqrt{F_{\max}^2+4\alpha_U[R_U]_+}}
           {2\alpha_U}
   \quad\text{at every point where both bounds apply.}
   $$

   Consequently these two source bounds, with the same finite constants,
   cannot both hold throughout an unbounded domain. Localizing them to
   $A\subset B(0,R_{\mathrm{axiom}})$ retains their literal content;
   it does not establish either bound outside $A$.

2. Let the raw positional reward be continuous, as in the declared
   mean-field kernel. In dimension $d\ge2$, if the domain contains a
   circle of radius $r\ge L_{\mathrm{grad}}/2$, the endpoint-separation
   condition {prf:ref}`axiom-non-deceptive-landscape` cannot hold with
   $\kappa_{\mathrm{raw},r}>0$. In particular that condition cannot
   hold on all of $\mathbb R^d$ with finite $L_{\mathrm{grad}}$ and
   positive $\kappa_{\mathrm{raw},r}$. This is the endpoint-reward
   axiom in Chapter 3, not the distinct segment-integrated gradient
   condition {prf:ref}`axiom-non-deceptive` in Chapter 2.

These are domain restrictions of the cited axioms. They do not assert
failure of the defined unbounded gas. They prevent using those bounded
domain conditions as an unchanged global proof of its confinement or
mean-field attraction. The regional profiles of
{prf:ref}`def-slc-profiles` retain the actual finite, infinite or zero
values on the domain where an estimate is being evaluated.
:::

:::{prf:proof}
For the first assertion, write $r=|x|$. The two source bounds and
Cauchy--Schwarz imply

$$
 \alpha_U r^2-R_U\le x\cdot\nabla U(x)
                  \le r|\nabla U(x)|\le F_{\max}r.
$$

Thus $\alpha_U r^2-F_{\max}r-[R_U]_+\le0$.
The positive root of this polynomial is exactly
$R_{\mathrm{axiom}}$; the polynomial is positive at every larger
$r$, proving the restriction.

For the second assertion, parameterize a circle in the domain by
$x(t)=c+r(e_1\cos t+e_2\sin t)$ for two orthonormal vectors.
The function
$g(t)=R_{\mathrm{pos}}(x(t))-R_{\mathrm{pos}}(x(t+\pi))$
is continuous and $g(t+\pi)=-g(t)$. Either $g(0)=0$, or the
intermediate value theorem applied on $[0,\pi]$ supplies $t_*$ with
$g(t_*)=0$. The corresponding points have equal rewards and distance
$2r\ge L_{\mathrm{grad}}$. The endpoint-separation axiom would require
$0\ge\kappa_{\mathrm{raw},r}>0$, a contradiction. No property of the
segment-integrated squared gradient is used or contradicted.
:::

(sec-slc-kinetics)=
## 2. Regional kinetic estimates and excursions

:::{div} feynman-prose
Inside a region with controlled curvature, the kinetic calculation has useful information to work with. The difficulty is that a Gaussian kick can leave that region in one step. We cannot declare the kick bounded merely because a large excursion is unlikely. Instead, estimate the motion where the regional assumptions apply and charge the remaining contribution explicitly. The size of that charge depends both on the escape probability and on how large the observable can become after escape. This is how a local force estimate can enter an honest global calculation.
:::

:::{prf:lemma} Force comparison with localized defects
:label: lem-slc-force-defect

For finite force values define $d_L(x,y)=[|F(x)-F(y)|-L|x-y|]_+$. Then

$$
|F(x)-F(y)|\le L|x-y|+d_L(x,y),
$$

and $d_L(x,y)\le D_A(L,|x-y|)$ when $x,y\in A$. If random $X,Y$ obey
$\mathbb E(|F(X)|+|F(Y)|)^2\le H^2$ and
$\Pr\{(X,Y)\notin A^2\}\le p$, then

$$
\mathbb E d_L(X,Y)
\le\mathbb E[\mathbf1_{A^2}D_A(L,|X-Y|)]+H\sqrt p.
$$

*Proof.* The first inequality follows from $z\le a+[z-a]_+$ for nonnegative
$z,a$. On $A^2$, use the definition of $\omega_A$. Outside, $d_L\le|F(X)|+|F(Y)|$;
Cauchy–Schwarz proves the expectation bound. No bounded Gaussian support is used.
$\square$
:::

:::{prf:corollary} A force-moment envelope for the excursion charge
:label: cor-slc-force-moment

If $|F(x)|\le a_F+b_F|x|$, $a_F,b_F\ge0$, and
$\mathbb E|X|^2\le m_X$, $\mathbb E|Y|^2\le m_Y$, one may use
$H^2=8a_F^2+4b_F^2(m_X+m_Y)$ in {prf:ref}`lem-slc-force-defect`.
For $Y=m+\sigma Z$ with $|m|\le M$ and an independent standard $d$-Gaussian,
for any $p>0$ one may use

$$
H_p=\max(1,2^{p-1})
\left[M^p+\sigma^p2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)}\right].
$$

*Proof.* Bound the sum of force norms by $2a_F+b_F(|X|+|Y|)$ and square using
$(u+v)^2\le2u^2+2v^2$. For the second claim use
$(u+v)^p\le\max(1,2^{p-1})(u^p+v^p)$ and integrate the radial Gaussian density;
substitution $t=r^2/2$ gives the displayed Gamma ratio. Conditional versions hold
for past-measurable $m$ and an independent next innovation. $\square$
:::

:::{prf:theorem} Exact BAOAB comparison with force defects
:label: thm-slc-baoab-defect

Consider two rows immediately before the first kinetic kick, after copying,
jitter and collisions. In the isotropic canonical BAOAB configuration write
$c=h/2$, $a=e^{-\gamma h}$, OU standard deviation $q$, final position standard
deviation $s$, and a cap $C_V$ that is 1-Lipschitz. The row is

$$
v_1=v+cF(x),\quad x_1=x+cv_1,\quad v_2=av_1+q\xi,
\quad x_2=x_1+cv_2,\quad v_3=v_2+cF(x_2),
\quad x^+=x_2+s\zeta,\quad v^+=C_V(v_3).
$$

Couple the two rows by the same independent standard Gaussian $\xi,\zeta$.
Let $u=|x-y|$, $w=|v-z|$, $d_0=d_{L_0}(x,y)$,
$d_2=d_{L_2}(x_2,y_2)$ and $\eta=c^2(1+a)$. Then pathwise

$$
\begin{aligned}
u^+&:=|x^+-y^+|\le A u+B w+\eta d_0,\\
w^+&:=|v^+-z^+|
\le(acL_0+cL_2A)u+(a+cL_2B)w
 +(ac+cL_2\eta)d_0+cd_2,\\
A&=1+\eta L_0,\qquad B=c(1+a).
\end{aligned}
$$

For a box terminal test, if $s>0$, the probability of differing alive marks,
conditional on $x_2,y_2$, is at most

$$
\min\left\{1,\frac{2\|x_2-y_2\|_1}{s\sqrt{2\pi}}\right\}.
$$

*Proof.* Subtract the first four updates: the shared OU noise cancels and
$x_2-y_2=(x-y)+B(v-z)+\eta(F(x)-F(y))$. Apply
{prf:ref}`lem-slc-force-defect`. Subtract $v_3$, apply the same bound at $x_2,y_2$
and the 1-Lipschitz cap. Final position noise cancels. For one coordinate, the
symmetric difference of translated membership intervals has length at most twice
the translation; its Gaussian density is at most $1/(s\sqrt{2\pi})$. A union bound
over coordinates proves the status estimate. $\square$
:::

:::{prf:lemma} Explicit excursion and moment charges
:label: lem-slc-excursion

For $Y=m+\sigma Z$, $Z\sim N(0,I_d)$ independent of the possibly random
mean $m$, if $|m|\le M<R$ and $\sigma>0$, then

$$
\Pr\{|Y|>R\}\le p_R:=\min\{1,2d\exp[-(R-M)^2/(2d\sigma^2)]\}.
$$

If $\mathbb E|Y|^p\le H_p$ for $p>r>0$, then

$$
\mathbb E[|Y|^r\mathbf1_{\{|Y|>R\}}]
\le H_p^{r/p}\Pr\{|Y|>R\}^{1-r/p}.
$$

*Proof.* An exit requires $|Z|>(R-M)/\sigma$, hence some coordinate exceeds
$(R-M)/(\sigma\sqrt d)$ in absolute value. Apply the scalar Gaussian tail bound
and a union bound. Hölder proves the second formula. For several stages, sum their
exit-probability bounds; use conditional bounds when their means depend on earlier
innovations. $\square$
:::

:::{prf:theorem} Absolute-position drift from radial structure
:label: thm-slc-radial-drift

Use the row update above with $|v|\le V_c$ after collision. Suppose for all entering
$x$ that

$$
x\cdot F(x)\le-k|x|^2+b+e_r(x),\qquad
|F(x)|^2\le G^2|x|^2+g^2+e_f(x),
$$

where all constants and defects are nonnegative. Unless a sharper bound is supplied,
use the explicitly defined defects

$$
e_r(x)=[x\cdot F(x)+k|x|^2-b]_+,\qquad
e_f(x)=[|F(x)|^2-G^2|x|^2-g^2]_+.
$$

They depend on $F,k,b,G,g$ and contain no unknown convergence rate. Let
$A_0=1-2\eta k+\eta^2G^2\ge0$. For any $t>0$, conditional on $x,v$,

$$
\begin{aligned}
\mathbb E|x^+|^2&\le r_K|x|^2+b_K+e_K(x),\\
r_K&=(1+t)A_0,\\
b_K&=(1+t)(2\eta b+\eta^2g^2)
 +(1+t^{-1})B^2V_c^2+d(c^2q^2+s^2),\\
e_K(x)&=(1+t)(2\eta e_r(x)+\eta^2e_f(x)).
\end{aligned}
$$

For $0<A_0<1$, the explicit choice $t=(1-A_0)/(2A_0)$ gives
$r_K=(1+A_0)/2<1$. For $A_0=0$, choose $t=1$ and obtain $r_K=0$.
This statement concerns absolute position, not two-law contraction.

*Proof.* Expand $|x+\eta F(x)|^2$ and substitute the two hypotheses. The
conditional mean of $x^+$ is $x+\eta F(x)+Bv$; Young's inequality bounds its square
by $(1+t)|x+\eta F(x)|^2+(1+t^{-1})B^2V_c^2$. The independent centred noises add
exactly $d(c^2q^2+s^2)$. The second kick and cap do not alter $x^+$. $\square$
:::

:::{prf:corollary} Copying and jitter in the position ledger
:label: cor-slc-copy-moment

Let $W_N=N^{-1}\sum_i|x_i|^2$, and suppose the expected pre-jitter copied second
moment is at most $r_CW_N+b_C+e_C(S)$. With recipient jitter that is conditionally independent and centred given its
pre-jitter source, gate and marks, with conditional covariance at most $\sigma_J^2 I_d$, the complete conservative update satisfies

$$
PW_N\le r_Kr_CW_N+r_K(b_C+d\sigma_J^2)+b_K
+r_Ke_C+P_C\overline e_K,
$$

where $\overline e_K$ averages $e_K$ at the actual jittered entering positions.
Here $P_C$ includes the cloning/jitter/collision preparation, and the same
conditional independence of kinetic noises is retained.

*Proof.* A jitter gate determined before that recipient's independent centred
jitter gives expected squared norm at most copied squared norm plus
$d\sigma_J^2$. Average {prf:ref}`thm-slc-radial-drift` and condition on the full
prepared swarm. $\square$
:::

:::{prf:lemma} A baseline copying moment bound
:label: lem-slc-copy-moment

For the conservative all-alive canonical gas with $N\ge2$, Gaussian cloning
weights in $[\kappa_C,1]$ and gate probabilities at most one,
$\mathbb E[W_N^{\rm copy}\mid S]\le(1+\kappa_C^{-1})W_N$.
Thus the preceding corollary permits
$r_C=1+\kappa_C^{-1}, b_C=e_C=0$; for $N=1$ take $r_C=1$.

*Proof.* Conditional on all fitness marks, the probability $b_{ij}$ of copying
$j\ne i$ is at most $1/[\kappa_C(N-1)]$. The recipient's expected squared
position is at most $|x_i|^2+\sum_{j\ne i}b_{ij}|x_j|^2$, after discarding the
nonpositive removal term. Sum over $i$ and divide by $N$. Each $|x_j|^2$
occurs in $N-1$ donor sums, yielding the bound. Integrating the marks preserves
it. This is a conservative upper estimate; signed donor gains can sharpen it.
For partially alive inputs the denominator is the eligible alive count and this
particular $N$-uniform formula must not be substituted without an alive-mass
bound. $\square$
:::

(sec-slc-composition)=
## 3. Keystone pressure and complete-update composition

:::{div} feynman-prose
The keystone argument asks whether cloning directs enough corrective action towards walkers carrying positional error. The kinetic argument asks what happens to that error during motion. Their estimates must be composed in the actual update order, because the state handed to the second stage has already changed. Regional assumptions introduce additional terms whenever the required configuration is lost. Keeping those terms visible lets us see when corrective selection and kinetic control outweigh the costs, and when the calculation has not yet established contraction.
:::

:::{prf:lemma} Regional use of discharged Keystone pressure
:label: lem-slc-regional-keystone

For two complete entering swarms let $\mathcal A\ge0$ denote the measured
error-weighted cloning activity and $W\ge0$ its positional comparison observable.
Suppose on a measurable entering-state class $\mathcal G$ the constants of
{prf:ref}`thm-keystone-discharged-averaged-pressure` have common bounds
$\chi>0,W_0\ge0,B_*\ge0$. Then, conditional on the entering pair,

$$
\mathbb E\mathcal A\ge
\chi W-\chi W_0-B_*/N^2-\chi W\mathbf1_{\mathcal G^c}.
$$

For disjoint entering classes $\mathcal G_j$, sum the corresponding inequalities
multiplied by $\mathbf1_{\mathcal G_j}$. Velocity, common-alive and unmatched-label
remainders in the cited theorem remain present when converting $W$ to its full
structural observable.

*Proof.* On $\mathcal G$ this is the cited all-population affine bound. On its
complement the right side is nonpositive, so nonnegativity suffices. The classes
are determined before measurement; no conditioning on a favourable measurement
outcome is substituted for its original law. $\square$
:::

:::{prf:definition} Computable regional Keystone constants
:label: def-slc-keystone-constants

Write $R_x=R_x^{\rm feat}$ and $R_v=R_v^{\rm feat}$ in this definition only.
For the bounded logistic pipeline in the parameter register, with $p_s>0$,
alive positions in $B(0,B_x)$ and velocities bounded by $B_v=V_{\max}$, set

$$
\begin{gathered}
D_0=2\sqrt{R_x^2+\lambda_{\rm alg}R_v^2},\quad
B_f=\max(R_x,\sqrt{\lambda_{\rm alg}}R_v),\\
m_x=R_x^2/(R_x+B_x)^2,\quad
m_z=\min\{m_x,\sqrt{\lambda_{\rm alg}}R_v^2/(R_v+B_v)^2\},\\
\kappa_D=e^{-D_0^2/(2\epsilon_D^2)},\quad
\kappa_C=e^{-D_0^2/(2\epsilon_C^2)},\quad
D_m=\sqrt{D_0^2+\delta_D^2}-\delta_D,\\
s_*=\sqrt{D_m^2/4+\sigma_s^2},\quad Z_*=D_m/\sigma_s,\\
A_-=\eta_r^{p_r},\quad f_+=(A_s+\eta_s)^{p_s},\quad
F_{\max}=(A_r+\eta_r)^{p_r}f_+,\\
L_H=\frac{A_rp_r}{4}\max\{\eta_r^{p_r-1},(A_r+\eta_r)^{p_r-1}\},\quad
L_A=L_H L_R/(\sigma_r m_z).
\end{gathered}
$$

Here $L_H=0$ for $p_r=0$, and $L_R$ is a supplied bound on the joint
position/velocity reward increments on this region. It can be obtained from
the supremum of the reward gradient on its convex hull when that gradient exists.
This requirement concerns the reward, separately from the kinetic force.
Set $E_{\max}=16B_x^2$ and choose $0<W_0\le E_{\max}$ with
$m_x^2W_0/4<D_0^2$. Then put

$$
\begin{gathered}
v_0=m_x^2W_0/4,\quad h_f=\sqrt{v_0/2},\quad
\rho_f=(v_0/2)/(D_0^2-v_0/2),\\
t_f=\big[\sqrt{h_f^2+\delta_D^2}-\sqrt{h_f^2/4+\delta_D^2}\big]/s_*,\\
m_f=p_s\min\{\eta_s^{p_s-1},(A_s+\eta_s)^{p_s-1}\}
       \frac{A_se^{-Z_*}}{(1+e^{-Z_*})^2},\quad \omega_f=m_ft_f,\\
r_f=\begin{cases}\min\{h_f/2,A_-\omega_f/(2f_+L_A)\},&L_A>0,\\
h_f/2,&L_A=0,\end{cases}\\
a_0=\min\{1,A_-\omega_f/[2s_c(F_{\max}+\epsilon_c)]\},\quad
C_0=\kappa_C\kappa_D^2\rho_fa_0,\\
M_f=\left\lceil2B_f\sqrt{2d}/r_f\right\rceil^{2d},\quad
\chi=C_0W_0^2/(2E_{\max}^2M_f^2),\quad
B_{\rm key}=C_0E_{\max}^2/W_0.
\end{gathered}
$$

These give $\chi,W_0,B_*=B_{\rm key}$ in
{prf:ref}`lem-slc-regional-keystone`, under exactly the source theorem's
comparison and alive-label conventions. The bandwidths $\epsilon_D,\epsilon_C$
are denoted $\sigma_D,\sigma_C$ in Chapter 3; its $\varepsilon_r,\varepsilon_s$
are $\sigma_r,\sigma_s$ here, and its gate scale $p_{\max}$ is $s_c$ here.

*Verification.* The squashing derivative has smallest eigenvalue
$R^2/(R+B)^2$ on the radius-$B$ ball; integration along segments gives $m_x,m_z$.
Gaussian donor weights lie between $\kappa$ and one. Logistic differentiation
gives the stated upper reward derivative and the lower diversity derivative
$m_f$ on $[-Z_*,Z_*]$. Integrating the latter over a score increment $t_f$
proves the admissible lower bound $\omega_f$ without an unspecified minimum.
The coverage argument in {prf:ref}`thm-keystone-discharged-averaged-pressure`
then gives the displayed $\chi,B_{\rm key}$. All replacements weaken its bounds
in the required direction. If $L_R=\infty$, $p_s=0$, or no positive $W_0$ is
available, this recipe does not provide positive pressure; the regional defect
formulation remains valid with a separately verified certificate. $\square$
:::

:::{prf:lemma} Explicit donor advantage and basin-count balance
:label: lem-slc-count-balance

Freeze a prepared cloning input with marks sufficient to determine its donor/gate
law. With eligible donor set $E_i$, the actual conditional probabilities are

$$
w_{ij}=\frac{\exp(-|z_i-z_j|^2/(2\epsilon_C^2))}
 {\sum_{\ell\in E_i}\exp(-|z_i-z_\ell|^2/(2\epsilon_C^2))},\qquad
b_{ij}=w_{ij}\min\{1,(F_j-F_i)_+/[s_c(F_i+\epsilon_c)]\}
$$

for alive recipients; the mandatory revival gate is one for dead recipients
when $E_i$ is nonempty. For a certified gap $F_j-F_i\ge\Delta>0$,
take $\underline w=\kappa_C$ and
$a_0=\min\{1,\Delta/[s_c(F_{\max}+\epsilon_c)]\}$.
Let $b_{ij}$ be the conditional probability that recipient $i$ copies donor
$j$, with at most one copy per recipient. For the unweighted position count
$Y_A=\sum_i\mathbf1_A(x_i)$, immediately after copying and before jitter,

$$
\mathbb E[\Delta Y_A\mid S]
=\sum_{i\notin A,j\in A}b_{ij}
 -\sum_{i\in A,j\notin A}b_{ij}.
$$

If $q_A$ eligible donors lie in $A$, every relevant donor probability is at least
$\underline w/M$ for $M$ eligible choices, and the actual gate has probability at
least $a_0$ on those donor choices, each outside recipient gains probability at
least $q_A\underline w a_0/M$. A lower bound follows by summing these gains and
subtracting upper bounds on reverse transfers. Integrate over sampled marks if
they were not frozen at the original input.

*Proof.* Copying changes recipient $i$'s indicator by
$\mathbf1_A(x_j)-\mathbf1_A(x_i)$ on its selected donor event. Sum expectations.
The donor lower bound follows by summing $q_A$ disjoint choices. Subsequent jitter,
kinetics and terminal marking contribute their own signed indicator changes; the
identity does not omit them from a claimed full-update drift. $\square$
:::

:::{prf:theorem} Full-step defect composition
:label: thm-slc-defect-composition

Let $\mathbf V$ be a nonnegative observable vector on the input, intermediate and
output spaces of the same conservative kernel. Suppose constant nonnegative
matrices and vectors satisfy

$$
P_C\mathbf V\le A_C\mathbf V+\mathbf b_C+\mathbf e_C,
\qquad P_K\mathbf V\le A_K\mathbf V+\mathbf b_K+\mathbf e_K.
$$

Then

$$
P\mathbf V\le M\mathbf V+\mathbf b+\mathbf e,
\quad M=A_KA_C,\quad
\mathbf b=A_K\mathbf b_C+\mathbf b_K,\quad
\mathbf e=A_K\mathbf e_C+P_C\mathbf e_K.
$$

If $w>0$ and $w^TM\le rw^T$ with $0\le r<1$, put $V=w^T\mathbf V$,
$b=w^T\mathbf b$, $e=w^T\mathbf e$. For every $n$ with finite displayed terms,

$$
\mathbb EV(S_n)\le r^n\mathbb EV(S_0)
+b\frac{1-r^n}{1-r}
+\sum_{j=0}^{n-1}r^{n-1-j}\mathbb Ee(S_j).
$$

If $e(S)\le\delta V(S)+\bar e$ and $r+\delta<1$, replace $r,b$ by
$r+\delta,b+\bar e$ to obtain uniform-time control. For $r=1$ the corresponding
bound is $\mathbb EV(S_0)+nb+\sum_{j<n}\mathbb Ee(S_j)$.

*Proof.* Apply $P_C$ to the kinetic inequality and use linearity, positivity and
$P_C1=1$. Substitute the cloning inequality in each component, then multiply by
$w^T$. Conditional expectation gives the scalar recursion; induction yields its
geometric convolution. Absorb $\delta V+\bar e$ for the last assertion. For
sub-Markov kernels the same nonnegative composition inequality holds with
$P_C1\le1$, but conditioned moments require division by survival probabilities;
the probability-kernel iteration above does not silently condition them.
$\square$
:::

:::{prf:remark} Regional matrices and certified contraction
:label: rem-slc-matrices

A regional matrix may be replaced by a constant envelope on the states under
consideration, with the excess placed in $\mathbf e$. Otherwise retain the exact
expectation of the intermediate-state matrix; multiplying two random matrices
at their initial values is invalid. Positive weights exist for a constant
nonnegative $M$ exactly when $\rho(M)<1$, by
{prf:ref}`thm-synergistic-rate-derivation`. For
$M=\left(\begin{smallmatrix}1-d_x&a_{xv}\\a_{vx}&1-d_v\end{smallmatrix}\right)$
with $0<d_x,d_v\le1$, the explicit condition is $a_{xv}a_{vx}<d_xd_v$.
All these entries may depend on regional profiles and $\theta$.
There is no need to leave the weights implicit: when $\rho(M)<1$, take
$w=(I-M^T)^{-1}\mathbf1$ and $r=1-1/\max_i w_i$.
Indeed the nonnegative Neumann series gives $w\ge\mathbf1$ and
$M^Tw=w-\mathbf1\le rw$, so $0\le r<1$. This computes the scalar factor
from the actual finite matrix entries; it is not a definition by an unknown
optimal dynamical rate.

Apply this theorem to a proved coupled kernel to control two-swarm discrepancies,
or to the one-swarm kernel to control moments. The two uses require their own
observables and estimates. In particular, the positive-entry synchronous bound
{prf:ref}`thm-slc-baoab-defect` is a regularity bound; it is not by itself a
hypocoercive contraction. Use the matching kinetic coupling/dissipation result
for its negative terms, retaining the force defects just calculated.
:::

(sec-slcr-selection-confinement)=
### 3.1. Selection flux, tail confinement and structural regeneration

:::{div} feynman-prose
Track the swarm's mean squared distance from the declared origin through a
complete update. Copying can remove an outer walker and replace it with an
inner donor; jitter and kinetic motion can increase that distance again.
The balance must charge all of these contributions. An inward-pointing force is one possible source of confinement,
but selection can supply another.

The quantity to inspect is the actual expected copying flux, with the random
fitness normalization retained. Available inner donors and a favorable fitness
gap give a negative contribution; copying in the opposite direction gives a
positive contribution. Their difference enters the full-update drift. When a
population lacks the required donors or fitness advantage, the defect records
that failure and accumulates over time. Thus a confinement certificate describes
both what selection achieves and which population configurations its argument
has yet to control.
:::

:::{prf:definition} Exact selection flux and a finite structural envelope
:label: def-slcr-flux

Use the actual conservative, all-alive, current-frame kernel on
$S=((x_i,v_i))_{i=1}^N$, with $|v_i|\le V_{\max}$, $N\ge2$, positive
fitness floors, independent measurement companions, simultaneous copying
from frozen donor positions, recipient Gaussian jitter and the stated
component collisions. There is no viscosity or historical donor rule.
Set

$$
W(S)=\frac1N\sum_i|x_i|^2,\qquad
p_{ij}=\frac{w_C(z_i,z_j)}{\sum_{k\ne i}w_C(z_i,z_k)}\quad(j\ne i),
\qquad \kappa_C\le w_C\le1.
$$

For the random frozen fitness vector $\mathbf F$ generated by the
actual measurement stage, put

$$
a_{ij}(\mathbf F)=\min\left\{1,
 \frac{(F_j-F_i)_+}{s_c(F_i+\epsilon_c)}\right\},\quad
\bar a_{ij}(S)=\mathbb E[a_{ij}(\mathbf F)\mid S],
$$

$$
\Phi(S)=\frac1N\sum_{i\ne j}p_{ij}\bar a_{ij}(S)
 (|x_j|^2-|x_i|^2),\qquad
\Phi_+(S)=\frac1N\sum_{i\ne j}p_{ij}\bar a_{ij}(S)
 (|x_j|^2-|x_i|^2)_+.
$$

These quantities use the full update's actual finite donor and measurement
laws. In particular they retain the shared random normalization of
fitness; averaging fitness before applying the acceptance gate would give
a different quantity.

The following directly computable envelope avoids that averaging. Let
$\underline F_i(S)\le F_i\le\overline F_i(S)$ hold for every measurement
assignment, and set

$$
\overline a_{ij}(S)=\min\left\{1,
 \frac{(\overline F_j(S)-\underline F_i(S))_+}
 {s_c(\underline F_i(S)+\epsilon_c)}\right\},\qquad
\mathcal A(S)=\frac1N\sum_{i\ne j}p_{ij}\overline a_{ij}(S)
 (|x_j|^2-|x_i|^2)_+.
$$

Then $\Phi_+\le\mathcal A$. Explicit admissible fitness bands are as
follows. Write

$$
r_i=\frac{R(z_i)-\overline R_N}
 {\sqrt{N^{-1}\sum_j(R(z_j)-\overline R_N)^2+\sigma_r^2}},
\quad h_b(t)=\left(\eta_b+\frac{A_b}{1+e^{-t}}\right)^{p_b}.
$$

If the raw sampled diversity has range at most $S_*$, its standardized
value lies in $[-Z_s,Z_s]$, $Z_s=S_*/\sigma_s$. Thus one may use

$$
\underline F_i=h_r(r_i)h_s(-Z_s),\qquad
\overline F_i=h_r(r_i)h_s(Z_s).
$$

The raw reward, feature radii, metric weight, diversity floor, bandwidths,
fitness amplitudes, exponents and normalization floors all enter these
expressions through their declared formulas; no unknown convergence rate
is used in defining the flux.
:::

:::{prf:lemma} Exact copying-moment balance
:label: lem-slcr-copy-balance

Let $S^C$ denote the population after copying, jitter and component
collisions, before kinetics. Then

$$
\mathbb E[W(S^C)\mid S]
=W(S)+\Phi(S)+d\sigma_J^2\,\overline p_{\mathrm{clone}}(S),
\qquad
\overline p_{\mathrm{clone}}(S)=\frac1N\sum_{i\ne j}p_{ij}\bar a_{ij}(S)
\le1.
$$
:::

:::{prf:proof}
Conditional on all measurement marks, recipient $i$ chooses donor $j$
with probability $p_{ij}$ and accepts with probability $a_{ij}$.
On acceptance its copied position is the frozen $x_j$; on rejection it
retains $x_i$. Therefore its conditional second-moment increment before
jitter is exactly $\sum_jp_{ij}a_{ij}(|x_j|^2-|x_i|^2)$.
Its centered Gaussian jitter is independent of the graph and adds
$d\sigma_J^2$ only on acceptance. Sum over recipients, divide by $N$,
and average the measurement marks. Collision changes velocities but
not positions, so it does not change this identity. No independence
between acceptance events of different rows is needed.
:::

:::{prf:theorem} A quantitative inward selection estimate on a declared population class
:label: thm-slcr-inward-selection

Declare a population class $\mathcal G$, a core $C$ consisting of a declared
union of basin regions with $C\subset B(0,R_c)$, and numbers
$m\in(0,1]$, $\Delta>0$, $p_g\in(0,1]$, and an adverse-flux envelope
$\delta W+b_{\rm rev}$ with $\delta,b_{\rm rev}\ge0$.
For every $S\in\mathcal G$, require the following explicitly testable
conditions:

1. At least $mN$ frozen donor positions belong to $C$.
2. For every recipient with $|x_i|>R_c$ and every donor in $C$,
   $\Pr(F_j-F_i\ge\Delta\mid S)\ge p_g$.
3. $\mathcal A(S)\le\delta W(S)+b_{\rm rev}$, with $\mathcal A$
   given by the finite sum in {prf:ref}`def-slcr-flux`.

Condition 2 holds with $p_g=1$ whenever
$\min_{j\text{ core}}\underline F_j-
 \max_{i:|x_i|>R_c}\overline F_i\ge\Delta$.
For less conservative bands its probability can be evaluated exactly by
summing the product of the independent measurement-companion
probabilities over their finite joint assignments; the event uses the
actual shared normalizers for that assignment.

Let

$$
F^*=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},\quad
 a_g=\min\{1,\Delta/[s_c(F^*+\epsilon_c)]\},\quad
\chi_0=\kappa_Cmp_ga_g,\quad \chi=\chi_0-\delta.
$$

Then, on $\mathcal G$,

$$
\Phi(S)\le-\chi W(S)+\chi_0R_c^2+b_{\rm rev}.
$$

All three conditions concern the current population, not merely the
location of a minimum of the spatial reward. In particular the theorem
does not infer core donor coverage for an arbitrary initial population.
:::

:::{prf:proof}
Since the donor denominator is at most $N-1$, each core donor has
$p_{ij}\ge\kappa_C/(N-1)$. On the favourable fitness event the
acceptance probability is at least $a_g$. For a recipient with $|x_i|>R_c$,
each core transfer decreases squared radius by at least
$|x_i|^2-R_c^2$. There are at least $mN$ such donors, and none is the
outside recipient. The total negative contribution from these edges is
therefore at least

$$
\frac{\kappa_Cmp_ga_g}{N}
 \sum_{i\text{ outside}}(|x_i|^2-R_c^2),
$$

where “outside” denotes $|x_i|>R_c$ and the factor
$N/(N-1)\ge1$ was discarded. Rows not in this radial tail each have
squared radius at most $R_c^2$, which implies

$$
\frac1N\sum_{i\text{ outside}}(|x_i|^2-R_c^2)
\ge W(S)-R_c^2.
$$

All other negative contributions may be dropped. All positive
contributions together are bounded by $\mathcal A\le\delta W+b_{\rm rev}$.
Combining these two estimates proves the claim.
:::

:::{prf:theorem} Full-update confinement with no radial restoring-force condition
:label: thm-slcr-selection-drift

Assume the preceding selection certificate has $0<\chi\le1$.
Suppose the actual measurable force obeys an explicitly assembled growth
envelope $|F(x)|\le g_0+g_1|x|$, $g_0,g_1\ge0$; no sign condition on
$x\cdot F(x)$ is imposed. Put $A_F=1+\eta g_1$ and require
$r_0=A_F^2(1-\chi)<1$. Keep the actual BAOAB
parameters

$$
c=h/2,\quad a=e^{-\gamma h},\quad
B=c(1+a),\quad\eta=c^2(1+a),\quad
q=b_O\sqrt{(1-e^{-2\gamma h})/(2\gamma)},\quad s=\sigma_x\sqrt h,
$$

with the continuous $\gamma=0$ convention for $q$, and let
$V_c=(1+2|\alpha_{\rm col}|)V_{\max}$.
For arbitrary $t>0$, define

$$
\begin{aligned}
r&=(1+t)r_0,\\
b&=(1+t)A_F^2(\chi_0R_c^2+b_{\rm rev}+d\sigma_J^2)
 +(1+t^{-1})(BV_c+\eta g_0)^2+d(c^2q^2+s^2).
\end{aligned}
$$

For $0<r_0<1$, the explicit choice
$t=(1-r_0)/(2r_0)$ gives $r=(1+r_0)/2<1$; for $r_0=0$ use $t=1$
and $r=0$. The bounded-force case is $g_1=0$. Even outward-growing
forces are admitted when their displayed growth is dominated by selection. On the declared class $\mathcal G$, the actual full kernel
satisfies $PW\le rW+b$.

Globally, define the computable nonnegative selection defect

$$
E_{\mathrm{sel}}(S)=
\left[\Phi(S)+\chi W(S)-\chi_0R_c^2-b_{\rm rev}\right]_+.
$$

Then the unconditional full-update inequality is

$$
PW(S)\le rW(S)+b+(1+t)A_F^2E_{\mathrm{sel}}(S).
$$

Consequently, for every finite horizon with finite expectations,

$$
\mathbb EW(S_n)\le r^n\mathbb EW(S_0)
 +b\frac{1-r^n}{1-r}
 +(1+t)A_F^2\sum_{j=0}^{n-1}r^{n-1-j}\mathbb EE_{\mathrm{sel}}(S_j).
$$

The same proof applies to the rooted population law, replacing normalized
row sums by their actual root expectations. Thus selection can supply
confinement even for $F\equiv0$, provided the declared coverage and
fitness-flux estimates, or their accumulated defects, are controlled.
:::

:::{prf:proof}
Write the post-copy position and collision velocity of a row as $(X,V)$.
The actual final position is
$X^+=X+BV+\eta F(X)+cq\xi+s\zeta$.
The two innovations are independent centered standard Gaussians,
independent of the copying and collision stage. Since $|V|\le V_c$,
Young's inequality gives

$$
\mathbb E[|X^+|^2\mid X,V]
\le(1+t)A_F^2|X|^2+(1+t^{-1})(BV_c+\eta g_0)^2
 +d(c^2q^2+s^2).
$$

Insert the exact copying balance and the definition of $E_{\mathrm{sel}}$,
using $\overline p_{\rm clone}\le1$. This proves the global inequality;
the inward-selection theorem makes the defect zero on $\mathcal G$.
The geometric convolution follows by iterated conditional expectation
and induction. The argument depends only on the law of one output root
and its actual frozen donor, so the same conditional identities hold in
the population map without asserting independence of interacting rows.
:::

:::{prf:corollary} Explicit control of the selection-class failure term
:label: cor-slcr-failure-defect

Write $D_C=2/\kappa_C$. The exact flux always satisfies
$\Phi(S)\le D_CW(S)$, hence

$$
0\le E_{\mathrm{sel}}(S)
\le(D_C+\chi)W(S)\mathbf1_{\{S\notin\mathcal G\}}.
$$

If $p>1$, $\mathbb EW(S_j)^p\le H_j$ and
$\Pr(S_j\notin\mathcal G)\le\varepsilon_j$, the accumulated term in
{prf:ref}`thm-slcr-selection-drift` is at most

$$
(1+t)A_F^2(D_C+\chi)
\sum_{j=0}^{n-1}r^{n-1-j}H_j^{1/p}
 \varepsilon_j^{1-1/p}.
$$

If the class-failure probability is not controlled, this expression
retains that obligation rather than replacing donor coverage by a
spatial confinement assumption. A uniform bound on the displayed
weighted defects gives a uniform moment bound; it alone does not prove
attraction to a stationary nonlinear population law.
:::

:::{prf:proof}
Drop all negative terms in $\Phi$ and use $a_{ij}\le1$ and
$p_{ij}\le1/[\kappa_C(N-1)]$. The sum of positive incoming donor
contributions is bounded by $W/\kappa_C\le D_CW$.
The fixed subtracted offset in $E_{\mathrm{sel}}$ is nonnegative, and the
selection theorem gives zero defect on $\mathcal G$.
Hölder's inequality applied to $W\mathbf1_{\mathcal G^c}$ gives the
stated bound. Substitute it into the proved convolution.
:::

:::{prf:corollary} Initial basin coverage from a uniform population
:label: cor-slcr-initial-coverage

Initialize the positions independently and uniformly in a declared ball
$B(x_0,R_I)$ with $R_I>0$, and let $C$ be the union of favourable core
basins used in {prf:ref}`thm-slcr-inward-selection`. Define

$$
p_C=\frac{|C\cap B(x_0,R_I)|}{v_d(R_I)},\qquad
k=\lceil mN\rceil,\qquad
\varepsilon_{N,m}=\sum_{j=0}^{k-1}{N\choose j}p_C^j(1-p_C)^{N-j},
$$

where $v_d(R_I)=\pi^{d/2}R_I^d/\Gamma(1+d/2)$. This is the exact
probability that the initial donor coverage is below $mN$. If the
fitness-gap and adverse-flux conditions hold on every initial
configuration with at least $k$ core donors, then

$$
\mathbb E E_{\rm sel}(S_0)
\le(2/\kappa_C+\chi)(|x_0|+R_I)^2\varepsilon_{N,m}.
$$

A requested initial coverage failure probability $\varepsilon$ is
certified by any $N$ for which the displayed binomial sum is at most
$\varepsilon$. If $p_C=0$ and $m>0$, no population size supplies this
initial coverage certificate; a discovery or recovery bound must be
used instead. For the initial mean-field law the core fraction is
exactly $p_C$.
:::

:::{prf:proof}
The independent core indicators have common success probability $p_C$,
so their count has the binomial law. Since $W(S_0)\le(|x_0|+R_I)^2$
pointwise, substitute that bound and the exact failure probability into
{prf:ref}`cor-slcr-failure-defect`. The law-level core fraction is the
same geometric probability by definition of the uniform initial law.
No statement of persistent coverage follows from this initialization
calculation; later failures retain their own conditional bounds.
:::

:::{prf:definition} Basin, transition and exterior contributions to selection
:label: def-slcr-pair-decomposition

For the declared measurable spatial partition $(A_a)_a$ into basins,
transition regions and exterior regions, define

$$
\mathcal A_{ab}(S)=\frac1N
\sum_{i:x_i\in A_a}\sum_{j\ne i:x_j\in A_b}
 p_{ij}\overline a_{ij}(|x_j|^2-|x_i|^2)_+.
$$

Then $\mathcal A=\sum_{a,b}\mathcal A_{ab}$ exactly. If explicit
regional envelopes give
$\mathcal A_{ab}(S)\le\delta_{ab}W(S)+b_{ab}$ on the controlled class,
use $\delta=\sum_{a,b}\delta_{ab}$ and
$b_{\rm rev}=\sum_{a,b}b_{ab}$. For a countable exterior decomposition,
these are nonnegative series; divergence makes this particular
certificate uninformative. Thus leakage toward distant basins,
transition regions and exterior shells appears separately in the
selection-confinement constant rather than being hidden in one
regularity assumption.
:::

:::{prf:theorem} Explicit selection confinement for the nonlinear population equation
:label: thm-slcr-population-flux

Let $\mu$ be an all-alive capped population law with finite
$W(\mu)=\int|x|^2\,\mu(dz)$ and finite $\int R(z)^2\,\mu(dz)$
whenever reward normalization is evaluated. Let $\eta_\mu(dt)$ be its actual
measurement-marked law, with type $t=(z,y_D,F_\mu(z,y_D))$.
For two such types $t,u$, let

$$
Z_C(\mu,z)=\int w_C(z,z')\,\mu(dz'),\qquad
\beta_\mu(t,u)=\frac{w_C(z_t,z_u)}{Z_C(\mu,z_t)}
 \min\left\{1,\frac{(F_u-F_t)_+}{s_c(F_t+\epsilon_c)}\right\}.
$$

This is the actual accepted outgoing-edge density, not a continuous-time
rate. Define the finite integral

$$
\Phi(\mu)=\iint\beta_\mu(t,u)
 (|x_u|^2-|x_t|^2)\,\eta_\mu(dt)\eta_\mu(du),\qquad
p_{\rm cl}(\mu)=\iint\beta_\mu(t,u)\,\eta_\mu(dt)\eta_\mu(du)\le1.
$$

The post-copy population law $\mathcal J(\mu)$ satisfies exactly

$$
W(\mathcal J(\mu))=W(\mu)+\Phi(\mu)+d\sigma_J^2p_{\rm cl}(\mu).
$$

Declare a population-law class $\mathfrak G$. Suppose for each
$\mu\in\mathfrak G$:

- The union of declared core basins $C\subset B(0,R_c)$ has
  $\mu(C\times\overline B_{V_{\max}})\ge m$.
- For almost every pair of physical types $(z,z')$ with $|x|>R_c$
  and $x'\in C$, the conditional product of their actual measurement
  laws gives probability at least $p_g$ to $F_{z'}-F_z\ge\Delta$.
- The outward integral $\Phi_+(\mu)$ obeys
  $\Phi_+(\mu)\le\delta W(\mu)+b_{\rm rev}$.
  It may be bounded by the fitness-band integrals corresponding exactly
  to {prf:ref}`def-slcr-pair-decomposition`.

Require $0<\chi\le1$ and $A_F^2(1-\chi)<1$, and select $t$ by
{prf:ref}`thm-slcr-selection-drift`. Then, with the same primitive constants
$\chi_0=\kappa_Cmp_ga_g$, $\chi=\chi_0-\delta$, $A_F,r,b,t$ as above,

$$
\Phi(\mu)\le-\chi W(\mu)+\chi_0R_c^2+b_{\rm rev},\qquad
W(\mathcal F_h\mu)\le rW(\mu)+b.
$$

For arbitrary admissible $\mu$, define

$$
E_{\mathrm{sel}}(\mu)=
[\Phi(\mu)+\chi W(\mu)-\chi_0R_c^2-b_{\rm rev}]_+.
$$

Then $W(\mathcal F_h\mu)\le rW(\mu)+b+
(1+t)A_F^2E_{\mathrm{sel}}(\mu)$, and its iterates obey the same
explicit geometric defect convolution as the particle moment bound.
:::

:::{prf:proof}
Condition on the root's physical and measurement type. It has either
no accepted outgoing edge, in which case it retains its position, or an
accepted donor with density $\beta_\mu(t,u)\eta_\mu(du)$, in which case
it copies that donor's frozen position. Incoming edges change the root's
collision velocity but do not change its frozen position or outgoing
copying rule. Thus integrating these two possibilities gives the exact
moment increment $\Phi(\mu)$. The independent root jitter contributes
$d\sigma_J^2$ times the actual acceptance probability. Since
$\beta_\mu\le1/\kappa_C$, the integral is absolutely finite under the
assumed second moment.

The denominator $Z_C$ is at most one. Hence the donor weight is at
least $\kappa_C$ relative to $\mu$, and the stated conditional
measurement event gives gate at least $a_g$ with probability $p_g$.
The inward part of the integral is consequently bounded below by
$\kappa_Cmp_ga_g\int_{|x|>R_c}(|x|^2-R_c^2)\,\mu(dz)$,
which is at least $\chi_0[W(\mu)-R_c^2]$. Subtract this from the
outward-flux envelope to obtain the first inequality. Apply the actual
kinetic second-moment inequality proved above to the root law
$\mathcal J(\mu)$ to obtain the second. The positive-part defect makes
the same inequality valid outside $\mathfrak G$. Iterating the
deterministic scalar inequality proves its convolution. Neither the
identity nor its estimate requires two different population laws to
approach one another.
:::

:::{prf:corollary} Tail probabilities and finite-horizon containment
:label: cor-slcr-tail-probabilities

Let $M_n$ denote the proved right-hand side of the particle moment
convolution, and let $\overline M_n$ denote its deterministic
population-law counterpart. For $R>0$ and $\varepsilon\in(0,1]$,

$$
\Pr\left\{\frac1N\#\{i:|x_i(n)|>R\}>\varepsilon\right\}
\le\frac{M_n}{\varepsilon R^2},\qquad
\Pr\{\max_i|x_i(n)|>R\}\le\frac{NM_n}{R^2},
$$

$$
\mu_n(|x|>R)\le\frac{\overline M_n}{R^2},\qquad
\Pr\{\exists n\le T:\max_i|x_i(n)|>R\}
\le\min\left\{1,\frac N{R^2}\sum_{n=0}^TM_n\right\}.
$$

These are confinement bounds on an unbounded state space. They do not
replace it with a hard reflecting boundary, and they retain the cost
of controlling all $N$ walkers rather than one typical walker.
:::

:::{prf:proof}
Pointwise, the fraction outside radius $R$ is at most $W/R^2$.
Markov's inequality proves its probability bound. If at least one row
is outside, $W>R^2/N$, which gives the second bound. Apply the same
pointwise integral inequality to $\mu_n$ for the population statement,
and use a union bound over the times for the last assertion. No temporal
independence is used.
:::

:::{prf:remark} Coincident distant populations and recovery estimates
:label: rem-slcr-recovery-obligation

For a population whose positions all equal $x$, the exact copying flux
$\Phi$ is zero, regardless of fitness values, since every radius difference
in its definition is zero. Therefore for $\chi>0$ its displayed selection
defect equals $[\chi|x|^2-\chi_0R_c^2-b_{\rm rev}]_+$ and is unbounded
as $|x|\to\infty$. A globally bounded one-step defect cannot be inferred
from the existence of favourable basins. With also identical velocities
zero and $F=0$, equal measured fitness gives no cloning, and
$PW=|x|^2+d(c^2q^2+s^2)$: a global one-step quadratic drift with a fixed
factor below one is impossible in this configuration.

This is a limitation of that one-step quadratic certificate. It neither
invalidates the mean-field evolution nor implies escape of the actual gas.
A usable confinement argument must control the displayed accumulated
selection defects, prove recovery over several updates, or use a suitable
coercive observable whose proved drift closes. No restoring-force or
convex-at-infinity hypothesis follows from this distinction.
:::

:::{prf:proposition} Multi-update recovery expressed through the actual selection flux
:label: prop-slcr-block-recovery

Under the linear-growth force envelope of
{prf:ref}`thm-slcr-selection-drift`, choose $t>0$ and put

$$
a=(1+t)(1+\eta g_1)^2,\qquad
b_0=a d\sigma_J^2+(1+t^{-1})(BV_c+\eta g_0)^2+d(c^2q^2+s^2).
$$

No inward-selection hypothesis is imposed here. The exact flux gives
$PW\le aW+b_0+a\Phi$ on every finite state. Fix a block length $m\ge1$
and a declared measurable cover $(\mathcal C_i)_i$ of population states,
including configurations without core donors. Supply structural bounds

$$
(P^j\Phi)(S)\le-c_{ij}W(S)+d_{ij},
\quad S\in\mathcal C_i,\quad 0\le j<m,
$$

where $c_{ij}$ may have either sign and $d_{ij}\ge0$. These are bounds
on finite-step flux integrals, defined without a limiting law by

$$
(P^j\Phi)(S)=\int\Phi(S_j)
 P(S,dS_1)P(S_1,dS_2)\cdots P(S_{j-1},dS_j),
$$

with $P^0\Phi=\Phi$. The integrand is the finite companion/fitness sum
in {prf:ref}`def-slcr-flux`, and the update laws are the declared Gaussian,
companion, gate and rotation kernels. Regional discovery and recovery
estimates must establish these bounds; they are not automatically
supplied by the existence of a spatial basin.

Define

$$
r_i=\max\left\{0,a^m-\sum_{j=0}^{m-1}a^{m-j}c_{ij}\right\},\qquad
b_i=b_0\sum_{j=0}^{m-1}a^j+\sum_{j=0}^{m-1}a^{m-j}d_{ij}.
$$

Then $P^mW\le r_iW+b_i$ on $\mathcal C_i$. Consequently, if the
explicit envelopes $r_*=\sup_i r_i<1$ and $b_*=\sup_i b_i<\infty$
are established, the sampled full kernel has global drift

$$
P^m(1+W)\le r_*(1+W)+(1-r_*+b_*).
$$

This permits initial updates with zero inward flux and later recovery.
It is a sufficient structural block certificate, not an assertion that
such uniform constants exist for every landscape or initialization.
:::

:::{prf:proof}
The copying balance and Young inequality, without subtracting any
selection gain, give $PW\le aW+b_0+a\Phi$. Iterating this inequality
by positivity of $P$ gives

$$
P^mW\le a^mW+b_0\sum_{j=0}^{m-1}a^j
                 +\sum_{j=0}^{m-1}a^{m-j}P^j\Phi.
$$

All integrals are finite when the displayed moment/growth bounds apply;
otherwise finiteness must be supplied as an explicit input. Substitute
the regional flux bounds term by term. Replacing a negative coefficient
of the nonnegative $W$ by zero preserves an upper bound. Taking the
regional suprema and adding one proves the claim. These are deterministic
coefficients of verified conditional estimates; no state-dependent
comparison matrices were multiplied as constants.
:::

:::{prf:theorem} Coercive tail envelopes beyond quadratic moments
:label: thm-slcr-coercive-envelope

Keep the conservative all-alive update and a declared basin, transition and
exterior partition. Let $\psi:[0,\infty)\to[0,\infty)$ be a continuous,
nondecreasing function with $\psi(r)\to\infty$ as $r\to\infty$. Define
$W_\psi(S)=N^{-1}\sum_i\psi(|x_i|)$. This is an auxiliary tail observable;
it need not equal the reward or its negative, and no convexity is required.
Suppose $|F(x)|\le g_0+g_1|x|$, with nonnegative constants assembled from
the regional growth profiles. Put

$$
A_x=1+\eta g_1,\quad b_x=BV_c+\eta g_0,\quad
\tau^2=c^2q^2+s^2,\quad
\zeta_d(u)=\frac{2^{1-d/2}}{\Gamma(d/2)}u^{d-1}e^{-u^2/2}\ (u>0),
$$

and define the explicit Gaussian increment envelope

$$
\mathcal B_\psi(r)=\int_0^\infty\!\int_0^\infty
\psi\bigl(A_x(r+\sigma_Ju)+b_x+\tau v\bigr)
\zeta_d(u)\zeta_d(v)\,du\,dv.
$$

Choose numerical $a_\psi\ge0$, $b_\psi\ge0$ and measurable regional
nonnegative defects $e_i(x)$ satisfying

$$
\mathcal B_\psi(|x|)\le a_\psi\psi(|x|)+b_\psi+e_i(x),
\qquad x\in A_i.
$$

The coefficients and defects are bounds on this displayed Gaussian integral,
not on an unknown mixing rate. An explicit default is
$e_i(x)=[\mathcal B_\psi(|x|)-a_\psi\psi(|x|)-b_\psi]_+$ on $A_i$;
a sharper proved regional upper bound may replace it. Define $e(x)=e_i(x)$ on $A_i$, and use
$p_{ij},\bar a_{ij}$ from {prf:ref}`def-slcr-flux` to set

$$
\begin{aligned}
\Phi_\psi(S)&=\frac1N\sum_{i\ne j}p_{ij}\bar a_{ij}
 [\psi(|x_j|)-\psi(|x_i|)],\\
\mathcal E_\psi(S)&=\frac1N\sum_i\left[
 \left(1-\sum_{j\ne i}p_{ij}\bar a_{ij}\right)e(x_i)
 +\sum_{j\ne i}p_{ij}\bar a_{ij}e(x_j)\right].
\end{aligned}
$$

If the structural selection-flux estimate is
$\Phi_\psi\le-\chi W_\psi+b_{\rm sel}+E_{\rm sel,\psi}$, then

$$
PW_\psi\le r_\psi W_\psi+b_\star+
 a_\psi E_{\rm sel,\psi}+\mathcal E_\psi,
\qquad r_\psi=a_\psi(1-\chi),\quad
b_\star=a_\psi b_{\rm sel}+b_\psi.
$$

The inward-selection proof supplies such an estimate by replacing squared
radii with $\psi(|x|)$ in its finite flux sums and adverse-flux bands.
The adverse-flux estimate must be re-established for $W_\psi$; the quadratic
constants do not automatically transfer. With that estimate,
$b_{\rm sel}=\chi_0\psi(R_c)+b_{\rm rev}$ and
$\chi=\chi_0-\delta$. When $0\le r_\psi<1$, iterating gives the explicit
geometric convolution of these displayed defects. In particular, if their
expected sum is at most $D$ at every update, then

$$
\sup_n\mathbb EW_\psi(S_n)
\le\mathbb EW_\psi(S_0)+\frac{b_\star+D}{1-r_\psi},
\qquad
\mathbb E L_N(S_n)(|x|>R)
\le\frac{\mathbb EW_\psi(S_n)}{\psi(R)}
$$

whenever $\psi(R)>0$. Also
$\Pr(\max_i|X_{n,i}|>R)\le N\mathbb EW_\psi(S_n)/\psi(R)$.
Thus unbounded-space confinement can be certified with a general coercive
tail observable; a finite quadratic moment is not built into this theorem.
The stronger reward moments needed by a particular mean-field error bound
remain separate requirements.
:::

:::{prf:proof}
For a frozen copied source $y$, the actual post-jitter position is
$X=y+C\sigma_JZ_J$, $C\in\{0,1\}$. The full position identity and the
velocity cap imply

$$
|X^+|\le A_x(|y|+\sigma_J|Z_J|)+b_x+\tau|Z|.
$$

Here $Z_J,Z$ are independent standard Gaussian vectors after conditioning
on the prepared graph. Monotonicity of $\psi$ and polar integration give
$\mathbb E[\psi(|X^+|)\mid\text{prepared graph}]\le\mathcal B_\psi(|y|)$.
The bound is uniform in the collision velocity, so its dependence on the
rest of the component introduces no omitted term. A latent $Z_J$ can be
sampled also for rows with $C=0$.

Average over the source choices. Their expected average $\psi$ value is
exactly $W_\psi+\Phi_\psi$ by the frozen-source copying identity. Their
expected defect is exactly $\mathcal E_\psi$; all source weights are
nonnegative and each recipient's weights sum to one. The displayed
Gaussian inequality therefore gives
$PW_\psi\le a_\psi(W_\psi+\Phi_\psi)+b_\psi+\mathcal E_\psi$.
Substitution proves the drift statement. For inward selection, every
favourable tail-to-core edge decreases the observable by at least
$\psi(|x_i|)-\psi(R_c)$, and the average positive part of these tail
excesses is at least $W_\psi-\psi(R_c)$. The same finite-sum argument as
in {prf:ref}`thm-slcr-inward-selection` thus proves the asserted constants.
Finally iterate the drift by conditional expectation. On $|x|>R$,
$\psi(|x|)\ge\psi(R)$; integration and a union bound yield the two
explicit tail inequalities. Every assertion presupposes the finiteness
of the displayed Gaussian integral and initial expectation, or is read
as a non-informative extended-real bound.
:::


(sec-slc-reward-geometry)=
### 3.2. Reward geometry as the source of the confinement coefficients

:::{div} feynman-prose
Begin with the reward landscape and the population occupying it. Which regions
contain favorable donors? How strongly does the companion rule connect them to
walkers farther out? Which transfers carry population outward instead? Regional
reward ranges, mass bounds and feature distances answer these questions through
the actual normalized fitness and acceptance rules. Their contributions give a
quantitative inward selection coefficient and an outward leakage coefficient.

The complete update then determines whether that selection balance controls
kinetic motion and noise. No convexity-at-infinity condition enters this
construction. Its coefficients nevertheless depend on the algorithm: a favorable
region cannot influence walkers that lack access to its donors, and changing
fitness exponents or bandwidths changes the accepted transfers. These are
landscape properties expressed through a specified probe and population class.
A positive certificate supplies sufficient control; a weak certificate leaves
room for sharper regional bounds or a different tail observable.
:::

:::{prf:definition} Regional reward and companion geometry
:label: def-slcg-geometry

Refine the declared spatial partition so that a finite union of bounded
regions $\mathcal B$ partitions the closed ball $\overline B(0,R_c)$,
and the remaining regions $\mathcal E$ partition $\{|x|>R_c\}$.
Boundary points belong to the bounded regions, including for atomic laws.
Declare favourable core regions $\mathcal C\subset\mathcal B$.
All statements concern the capped phase-space cylinders over these regions.
Let $m_a^\pm$ bound their population masses. Let $R_a^-,R_a^+$ bound
reward on region $a$; infinite reward endpoints are allowed in the logistic
formulas below by continuity. Supply regional integral bounds

$$
u_a^-\le\int_{A_a}R\,d\mu\le u_a^+,\qquad
v_a^-\le\int_{A_a}R^2\,d\mu\le v_a^+.
$$

Require the sums defining $u^\pm=\sum_a u_a^\pm$ and
$v^\pm=\sum_a v_a^\pm$ to be finite, and set

$$
s_r^- =\sqrt{\sigma_r^2+\max\{0,v^--\max[(u^-)^2,(u^+)^2]\}},\qquad
s_r^+ =\sqrt{\sigma_r^2+\max\{0,v^+-\operatorname{dist}(0,[u^-,u^+])^2\}}.
$$

The actual reward mean belongs to $[u^-,u^+]$ and its regularized
standard deviation belongs to $[s_r^-,s_r^+]$. Consequently define

$$
z_a^- =\min_{s\in\{s_r^-,s_r^+\}}\frac{R_a^- -u^+}{s},\qquad
z_a^+ =\max_{s\in\{s_r^-,s_r^+\}}\frac{R_a^+ -u^-}{s}.
$$

With $h_b(z)=(\eta_b+A_b/(1+e^{-z}))^{p_b}$ and
$Z_s=S_*/\sigma_s$, valid regional fitness bands are

$$
F_a^-=h_r(z_a^-)h_s(-Z_s),\qquad
F_a^+=h_r(z_a^+)h_s(Z_s).
$$

Here $S_*$ is the explicit bounded diversity range from the parameter
register. These worst-case bands are uniform over all measurement companions;
refined probabilities may retain their actual bandwidth-dependent law.
Define gate bounds

$$
g_{ab}^- =\min\left\{1,\frac{(F_b^- -F_a^+)_+}{s_c(F_a^++\epsilon_c)}\right\},\qquad
g_{ab}^+ =\min\left\{1,\frac{(F_b^+ -F_a^-)_+}{s_c(F_a^-+\epsilon_c)}\right\}.
$$

Let $D_{ab}^-,D_{ab}^+$ bound the actual squashed feature distance
between the two capped regions. Put

$$
w_{ab}^- =e^{-(D_{ab}^+)^2/(2\epsilon_C^2)},\qquad
w_{ab}^+ =e^{-(D_{ab}^-)^2/(2\epsilon_C^2)},
$$

$$
Z_a^- =\max\{\kappa_C,\sum_b m_b^- w_{ab}^-\},\qquad
Z_a^+ =\min\{1,\sum_b m_b^+ w_{ab}^+\}.
$$

The supplied intervals must be consistent with the input class. They then
bound the actual companion denominator on region $a$.
These are reward, regional geometry and population-allocation data. No
restoring-force coefficient enters their definition.
:::

:::{prf:proof}
Regional integrals add to the reward mean and second moment. Subtracting
its squared mean gives the stated variance bounds; adding $\sigma_r^2$
ensures strictly positive denominators. For fixed positive denominator,
standardization decreases with the mean and increases with the reward.
At either resulting numerator its extrema over a positive interval of
denominators occur at an endpoint. Monotonicity of $h_r,h_s$ gives the
fitness bands. The acceptance gate increases with donor fitness and decreases
with recipient fitness, giving $g_{ab}^\pm$. The Gaussian companion weight
decreases with feature distance. Summing its regional bounds against the
mass intervals proves $Z_a^\pm$, including the global bounds
$\kappa_C\le Z_C\le1$.
:::

:::{prf:theorem} Confinement coefficients derived from reward accumulation and leakage
:label: thm-slcg-flux-coefficients

For an admissible population law in {prf:ref}`def-slcg-geometry`, write
$W_b=\int_{A_b}|x|^2\,d\mu$ and $W=\sum_b W_b<\infty$. Define

$$
\chi_{\rm in}=\inf_{a\in\mathcal E}
 \sum_{b\in\mathcal C}\frac{m_b^-w_{ab}^-g_{ab}^-}{Z_a^+},\qquad
D_b=\sum_a\frac{m_a^+w_{ab}^+g_{ab}^+}{Z_a^-},
$$

$$
\delta_{\rm out}=\sup_{b\in\mathcal E}D_b,\qquad
b_{\rm mix}=R_c^2\sum_{b\in\mathcal B}D_bm_b^+,\qquad
\chi_{\rm geom}=\chi_{\rm in}-\delta_{\rm out}.
$$

Only source regions that occur in the declared class need enter the
infimum; its value is defined as zero when there are no such exterior
source regions. A zero inward bound is allowed. Require the nonnegative sums
used in the conclusion to be finite. For the exact population selection
flux of {prf:ref}`thm-slcr-population-flux`,

$$
\boxed{\quad
\Phi(\mu)\le-\chi_{\rm geom}W
                  +\chi_{\rm in}R_c^2+b_{\rm mix}.
\quad}
$$

Thus the confinement input to the full-update theorem is determined by
reward gaps, within-basin redistribution, mass in favourable regions,
companion distances and outward reward-supported transfers. Geometry that
gives $\chi_{\rm geom}>0$ supplies inward selection without any confining
force assumption. Whether this exceeds kinetic growth and noise is decided
by the complete-update inequalities, not by reward integrability alone.

For a finite all-alive population, use its actual regional counts $N_a$
and empirical reward integrals. Replace the companion denominators by

$$
Z_{a,N}^\pm=\sum_b (N_b-\mathbf1_{b=a})w_{ab}^\pm,
$$

for occupied source regions $N_a>0$, and use

$$
\chi_{{\rm in},N}=\min_{a\in\mathcal E:N_a>0}
 \sum_{b\in\mathcal C}\frac{N_bw_{ab}^-g_{ab}^-}{Z_{a,N}^+},\qquad
D_{b,N}=\sum_{a:N_a>0}\frac{N_aw_{ab}^+g_{ab}^+}{Z_{a,N}^-}.
$$

For $N\ge2$ these denominators are positive. Take the empty tail minimum
to be zero and define $\delta_{{\rm out},N},b_{{\rm mix},N}$ by the
same formulas with $m_b^+=N_b/N$. The identical flux bound holds. For
$N=1$ there is no copying and its flux is zero. These finite-population
coefficients depend on the entering configuration. A deterministic rate
requires certified envelopes on the declared population class, or the
explicit accumulated defects already proved in this chapter.
:::

:::{prf:proof}
For a recipient in exterior region $a$, the accepted probability of choosing
a donor in core region $b$ is at least
$m_b^-w_{ab}^-g_{ab}^-/Z_a^+$. Each such transfer decreases squared
radius by at least $|x|^2-R_c^2$. Summing and integrating proves a negative
flux of magnitude at least
$\chi_{\rm in}\int_{|x|>R_c}(|x|^2-R_c^2)\,d\mu
\ge\chi_{\rm in}(W-R_c^2)$.

For every region pair the positive part of the squared-radius difference
is at most the donor squared radius. The actual accepted-edge density is
at most $w_{ab}^+g_{ab}^+/Z_a^-$. Its positive flux is therefore at most
$m_a^+w_{ab}^+g_{ab}^+W_b/Z_a^-$. Summing gives
$\Phi_+\le\sum_bD_bW_b\le\delta_{\rm out}W+b_{\rm mix}$,
because $W_b\le R_c^2m_b^+$ for bounded target regions. Combine the
two bounds, discarding other negative contributions.

For finite populations freeze the actual measurements. The recipient's
companion denominator excludes exactly itself, giving $Z_{a,N}^\pm$.
The same pointwise fitness bands hold for every shared normalization
assignment. Sum donor probabilities over the $N_b$ core rows. For positive
flux, summing its uniform per-pair bound over $N_a$ recipients gives
$D_{b,N}W_b$; counting a diagonal term when $a=b$ only enlarges this
upper bound. Averaging the measurement marks preserves all inequalities.
No independent-walker assumption was made.
:::

:::{prf:remark} Tail information and the dependence on the algorithm
:label: rem-slcg-tail-data

The reward integrals and core masses in this construction must be bounded
on the asserted population class. For instance, if
$|R(x,v)|\le K_0+K_2|x|^2$ and
$\int_{|x|>R_0}|x|^8d\mu\le H_E$ with $R_0>0$, then the exterior contributions obey

$$
\int_{|x|>R_0}|R|d\mu\le K_0H_E/R_0^8+K_2H_E/R_0^6,\qquad
\int_{|x|>R_0}R^2d\mu\le2K_0^2H_E/R_0^8+2K_2^2H_E/R_0^4.
$$

A reward that favours distant regions can reduce inward gates and increase
outward leakage; a reward with favourable bounded regions can produce the
opposite inequalities. The resulting coefficients also depend on whether
the configured selection rule uses that reward and on current donor coverage.
They are joint landscape–algorithm descriptors. An uninformative coefficient
is not replaced by a force requirement: the optional-force theorem tests a
separate parameter and distinguishes a sufficient repair from proved necessity.
:::


(sec-slco-optional-trap)=
### 3.3. An optional trapping coefficient determined by the selection certificate

:::{div} feynman-prose
Keep the original force and expose an auxiliary trapping strength as an explicit
parameter. The value zero means the original algorithm. We can first test
whether the reward-driven selection estimate already confines it, then calculate
which positive strengths would close the same full-update drift bound if needed.
Each chosen strength defines its own update law; adding a trap changes the
kinetics even when the reward and fitness formula remain fixed.

The admissible strengths form a sufficient interval, with an upper limit because
a strong force can overshoot at a fixed timestep. Regional radial information
can sharpen that interval without requiring an inward force everywhere. Failure
at zero does not prove that a trap is necessary: the estimate may lose useful
selection or force cancellations. Such a necessity claim needs a separate escape
obstruction. When comparing strengths, donor coverage, reachable populations and
local regularity must also be checked for the corresponding kernels.
:::

:::{prf:theorem} Quantitative admissible interval for an auxiliary linear trap
:label: thm-slco-trap-interval

Keep the actual full update and the declared controlled population class.
Let its proved geometry and cloning estimates give the exact copying
bound, before recipient jitter,

$$
\mathbb E[W(S^{\rm copy})\mid S]
\le(1-\chi)W(S)+b_{\rm sel}+E_{\rm sel}(S),
\qquad W(S)=N^{-1}\sum_i|x_i|^2,
$$

where $\chi\le1$, $b_{\rm sel}\ge0$, and $E_{\rm sel}\ge0$ are the
explicit quantities supplied by the region-pair selection flux. For
example {prf:ref}`thm-slcg-flux-coefficients` supplies
$\chi=\chi_{\rm geom}$ and
$b_{\rm sel}=\chi_{\rm in}R_c^2+b_{\rm mix}$. The earlier core estimate gives
$\chi=\chi_0-\delta$ and
$b_{\rm sel}=\chi_0R_c^2+b_{\rm rev}$. The defect vanishes wherever
the declared coverage and flux certificate holds. Negative $\chi$ is
permitted and records a certified expansion bound for copying.

Declare an optional auxiliary force coefficient $\lambda\ge0$ and
use the force

$$
F_\lambda(x)=F_{\rm geom}(x)-\lambda x,
\qquad |F_{\rm geom}(x)|\le g_0+g_1|x|,
\quad g_0,g_1\ge0.
$$

Here $F_{\rm geom}$ denotes the specified original force; the reward
and fitness law remain the ones declared in the algorithm. The symbol
$\lambda$ in this theorem is the auxiliary force coefficient, distinct
from any comparison-feature metric weight. For each selected value it
specifies one fixed full-update kernel $P_\lambda$.
Put

$$
\begin{aligned}
c&=h/2,\quad a=e^{-\gamma h},\quad
B=c(1+a),\quad\eta=c^2(1+a)>0,\\
q^2&=b_O^2(1-e^{-2\gamma h})/(2\gamma),\quad
s^2=\sigma_x^2h,\quad V_c=(1+2|\alpha_{\rm col}|)V_{\max},\\
A_\lambda&=|1-\eta\lambda|+\eta g_1,\qquad
r_{0,\lambda}=(1-\chi)A_\lambda^2,
\end{aligned}
$$

using $q^2=b_O^2h$ at $\gamma=0$. For any $t>0$ define

$$
\begin{aligned}
r_\lambda&=(1+t)r_{0,\lambda},\\
b_\lambda&=(1+t)A_\lambda^2(b_{\rm sel}+d\sigma_J^2)
 +(1+t^{-1})(BV_c+\eta g_0)^2+d(c^2q^2+s^2),\\
e_\lambda(S)&=(1+t)A_\lambda^2E_{\rm sel}(S).
\end{aligned}
$$

Then the actual full update satisfies

$$
\boxed{\quad
P_\lambda W(S)\le r_\lambda W(S)+b_\lambda+e_\lambda(S).
\quad}
$$

There exists $t>0$ with $r_\lambda<1$ precisely when the displayed
certificate has $r_{0,\lambda}<1$. In that case use

$$
t=\frac{1-r_{0,\lambda}}{2r_{0,\lambda}},\qquad
r_\lambda=\frac{1+r_{0,\lambda}}2
\quad\text{if }0<r_{0,\lambda}<1;
$$

use $t=1$, $r_\lambda=0$ if $r_{0,\lambda}=0$.
For $\chi<1$, set

$$
H=(1-\chi)^{-1/2}-\eta g_1.
$$

The complete sufficient interval is empty if $H\le0$. If $H>0$, it is

$$
\boxed{\quad
\lambda\in[0,\infty)\cap
 \left(\frac{1-H}{\eta},\frac{1+H}{\eta}\right).
\quad}
$$

For $\chi=1$, $r_{0,\lambda}=0$ for every finite $\lambda\ge0$,
provided the same copying certificate holds. In particular the unmodified
force $\lambda=0$ is certified exactly when

$$
(1-\chi)(1+\eta g_1)^2<1.
$$

Thus the reward-driven selection certificate can make an auxiliary trap
unnecessary. When this inequality fails but the sufficient interval is
nonempty, the displayed positive coefficients supply a quantitative
way to close this drift estimate. The upper endpoint records possible
discrete-step overshoot; increasing trap strength without bound is not
justified by this estimate.
:::

:::{prf:proof}
Write $(X,V)$ for a row's post-copy, post-jitter position and collision
velocity. The exact final position is

$$
X^+=X+BV+\eta F_\lambda(X)+cq\xi+s\zeta,
$$

where $\xi,\zeta$ are independent standard Gaussians, independent of
preparation. The specified force envelope gives

$$
|X+\eta F_\lambda(X)|
\le |1-\eta\lambda||X|+\eta|F_{\rm geom}(X)|
\le A_\lambda|X|+\eta g_0.
$$

Since $|V|\le V_c$, the deterministic center has norm at most
$A_\lambda|X|+BV_c+\eta g_0$. Young's inequality therefore gives

$$
\mathbb E[|X^+|^2\mid X,V]
\le(1+t)A_\lambda^2|X|^2
 +(1+t^{-1})(BV_c+\eta g_0)^2+d(c^2q^2+s^2).
$$

Centered recipient jitter adds at most $d\sigma_J^2$ to the averaged
copying moment. Insert the assumed, already quantified copying bound
into this conditional inequality to obtain exactly $r_\lambda$,
$b_\lambda$ and $e_\lambda$. There is no substitution of a continuous
Langevin generator for the complete discrete update.

Because $r_{0,\lambda}\ge0$, a positive $t$ with
$(1+t)r_{0,\lambda}<1$ exists if and only if $r_{0,\lambda}<1$.
The stated choice verifies this algebraically. For $\chi<1$, taking
nonnegative square roots transforms $r_{0,\lambda}<1$ into
$|1-\eta\lambda|<H$. This has no solution if $H\le0$ and gives
exactly the open interval displayed if $H>0$. If $\chi=1$, the zero
prefactor gives the separate conclusion. Substitution of $\lambda=0$
gives the unmodified-force criterion.
:::

:::{prf:corollary} Parameter-uniform profiles, accumulated defects and tail bounds
:label: cor-slco-uniform-profiles

Suppose trap coefficients are being compared on a declared range
$\Lambda\subset[0,\infty)$. The same interval calculation can be used
uniformly only when its selection bounds hold uniformly over the
associated controlled population classes. More explicitly, prove common
constants $\chi_*\le1$, $b_*<\infty$ such that, for each candidate
$\lambda\in\Lambda$, the actual copying law satisfies

$$
\mathbb E[W(S^{\rm copy})\mid S]
\le(1-\chi_*)W(S)+b_*+E_\lambda(S).
$$

Then substitute $\chi_*,b_*$ into
{prf:ref}`thm-slco-trap-interval` and intersect its interval with
$\Lambda$. Alternatively, compute each candidate's own
$\chi(\lambda)$ and $b_{\rm sel}(\lambda)$ and test
$(1-\chi(\lambda))A_\lambda^2<1$ separately. The change of force alters
reachable populations, fitness normalizers and donor coverage; their
certificates cannot be silently held fixed when that uniform statement
has not been established.

For a fixed certified coefficient and its fixed $t$, define

$$
M_n=r_\lambda^n\mathbb EW(S_0)
 +b_\lambda\frac{1-r_\lambda^n}{1-r_\lambda}
 +(1+t)A_\lambda^2\sum_{j=0}^{n-1}
 r_\lambda^{n-1-j}\mathbb EE_\lambda(S_j).
$$

Then $\mathbb EW(S_n)\le M_n$,
$\Pr(\max_i|x_i(n)|>R)\le NM_n/R^2$, and

$$
\Pr\{\exists n\le T:\max_i|x_i(n)|>R\}
\le\min\left\{1,\frac N{R^2}\sum_{n=0}^TM_n\right\}.
$$

The same copying-flux integral and conditional kinetic calculation apply
to the exact rooted population map $\mathcal F_{h,\lambda}$, giving the
identical deterministic moment convolution and
$\mu_n(|x|>R)\le M_n/R^2$ with population moments in place of empirical
expectations.
:::

:::{prf:proof}
A common lower bound $\chi_*$ on selection contraction and common upper
bound $b_*$ preserve the copying inequality because $W\ge0$. Thus the
preceding algebra applies separately to every candidate kernel with
these same numerical bounds. If common bounds have not been proved,
only the candidate-specific inequality is available. Iterate the full
conditional drift by induction to obtain the displayed convolution.
A row outside radius $R$ implies $W>R^2/N$; Markov's inequality and a
union bound over times prove the tail statements. The root moment uses
the same actual copying donor and independent kinetic innovations, so
integrating its proved population-flux identity gives the corresponding
population statements.
:::

:::{prf:lemma} Effect of the auxiliary trap on local regularity certificates
:label: lem-slco-local-regularity

For any declared region $A$ and displacement $r\ge0$, suppose the
original force has increment profile

$$
\omega_A^{\rm geom}(r)=
\sup_{x,y\in A,\ |x-y|\le r}|F_{\rm geom}(x)-F_{\rm geom}(y)|.
$$

Then

$$
\omega_A^\lambda(r)\le\omega_A^{\rm geom}(r)+\lambda r.
$$

In particular a local force Lipschitz bound $L_A^{\rm geom}$ becomes
$L_A^\lambda=L_A^{\rm geom}+\lambda$. If $F_{\rm geom}=-\nabla U$,
the added potential is $\lambda|x|^2/2$ and its Hessian contribution is
$\lambda I$. Every kinetic density or local-minorization proof using
$c^2L_A<1$ must consequently check
$c^2(L_A^{\rm geom}+\lambda)<1$ as well as the selection-drift
interval. Their intersection is the certified parameter set for the
combined conclusion.
:::

:::{prf:proof}
Subtract the two force values and use
$|\lambda(x-y)|\le\lambda r$. The Lipschitz statement follows by
dividing by $|x-y|$. Differentiate the added quadratic potential for the
Hessian statement. The final restriction is substitution into the
stated kinetic hypothesis; a moment-drift certificate alone does not
supply that separate density estimate.
:::

:::{prf:remark} Sufficient assistance is distinct from a necessary trap
:label: rem-slco-not-necessity

Failure of the unmodified-force drift inequality means that this
particular structural upper bound does not certify confinement.
It does not prove that $\lambda>0$ is necessary: finer selection fluxes,
regional force cancellation, or a different Lyapunov function may still
prove unmodified confinement. Likewise an empty displayed interval means
that this combined certificate is inconclusive, not that every force
choice fails. A statement that a trap is genuinely required needs a
separate lower escape or non-tightness obstruction for the unmodified
kernel. The available outward-tail and transition lower bounds can be
used for that purpose only with their full stated hypotheses.
:::

:::{prf:theorem} Sharper trap intervals from two-sided regional radial profiles
:label: thm-slco-radial-profile-trap

Let $(A_i)_i$ partition the positions reached after copying and jitter.
Suppose the following explicitly established profiles hold on each region:

$$
\ell_i|x|^2-b_i^-\le x\cdot F_{\rm geom}(x)
 \le u_i|x|^2+b_i^+,\qquad
|F_{\rm geom}(x)|^2\le G_i^2|x|^2+g_i^2,
$$

where $b_i^\pm,g_i^2\ge0$. Neither $u_i$ nor $\ell_i$ is required
to be negative. Write

$$
\beta=1-\eta\lambda,\quad\beta_+=\max\{\beta,0\},\quad
\beta_-=\min\{\beta,0\},
$$

where $\beta_-$ is the signed negative part. Define

$$
\begin{aligned}
a_i(\lambda)&=\beta^2+
 2\eta(\beta_+u_i+\beta_-\ell_i)+\eta^2G_i^2,\\
d_i(\lambda)&=2\eta(\beta_+b_i^+-\beta_-b_i^-)+\eta^2g_i^2,\\
A_{r}(\lambda)&=\max\{0,\sup_i a_i(\lambda)\},\qquad
D_{r}(\lambda)=\sup_i d_i(\lambda).
\end{aligned}
$$

If these envelopes are finite, the same actual copying certificate gives

$$
\begin{aligned}
P_\lambda W\le{}&
 (1+t)A_{r}(\lambda)(1-\chi)W
 +(1+t)\{A_{r}(\lambda)(b_{\rm sel}+d\sigma_J^2)
                         +D_{r}(\lambda)\}\\
&+(1+t^{-1})B^2V_c^2+d(c^2q^2+s^2)
 +(1+t)A_{r}(\lambda)E_{\rm sel}.
\end{aligned}
$$

Thus $A_{r}(\lambda)(1-\chi)<1$ is a sharper sufficient test, with
$t=(1-r_0)/(2r_0)$ when $r_0=A_{r}(\lambda)(1-\chi)\in(0,1)$ and
$t=1$ when $r_0=0$.

For a finite regional partition and $\chi<1$, its admissible coefficients
are obtained by an explicit intersection of quadratic intervals.
Set $T=(1-\chi)^{-1}$. On the branch $\beta\ge0$, define

$$
\Delta_i^+=T+\eta^2(u_i^2-G_i^2),\qquad
I_i^+=(-\eta u_i-\sqrt{\Delta_i^+},
        -\eta u_i+\sqrt{\Delta_i^+})
$$

if $\Delta_i^+>0$, and $I_i^+=\varnothing$ otherwise.
On the branch $\beta\le0$, replace $u_i$ by $\ell_i$ to obtain
$\Delta_i^-$ and $I_i^-$. Then the permitted values of $\beta$ are

$$
\left([0,1]\cap\bigcap_i I_i^+\right)
\;\cup\;
\left(( -\infty,0]\cap\bigcap_i I_i^-\right),
$$

and $\lambda=(1-\beta)/\eta$. The restriction $\beta\le1$ is exactly
$\lambda\ge0$. For a countable partition, also verify the strict
uniform margin $\sup_i a_i(\lambda)<T$; individual strict inequalities
alone may approach equality in the tail. Finiteness of
$D_{r}(\lambda)$ is a separate required check.
:::

:::{prf:proof}
Expand the exact deterministic position contribution:

$$
|x+\eta F_\lambda(x)|^2
=|\beta x+\eta F_{\rm geom}(x)|^2
=\beta^2|x|^2+2\eta\beta x\cdot F_{\rm geom}(x)
 +\eta^2|F_{\rm geom}(x)|^2.
$$

If $\beta\ge0$, use the upper radial bound; if $\beta\le0$, multiplying
the lower radial bound by $\beta$ reverses its inequality. This gives
exactly $a_i|x|^2+d_i$. Replacing $a_i$ by the common nonnegative
$A_{r}$ and $d_i$ by $D_{r}$ preserves the upper bound. Apply
Young's inequality to the vector sum
$[x+\eta F_\lambda(x)]+BV$, then add the centered Gaussian variance.
The resulting moment term has coefficient $(1+t)A_{r}$; inserting
the actual copying estimate and jitter variance proves the full drift.

For $\chi<1$, the condition $A_{r}(1-\chi)<1$ is equivalent to
$\sup_i a_i<T$, since $T>0$. On the positive branch,
$a_i<T$ is equivalent to
$(\beta+\eta u_i)^2<\Delta_i^+$, giving precisely $I_i^+$.
The negative branch has the identical calculation with $\ell_i$.
Intersect the inequalities with their branch and the condition
$\beta\le1$. For finitely many regions, their strict inequalities give
a strict maximum below $T$. For countably many regions this implication
requires the separately stated uniform margin. The choice of $t$ is the
same verified algebra as in {prf:ref}`thm-slco-trap-interval`.
:::

(sec-slc-basins)=
## 4. Basin discovery, establishment, residence and crossings

:::{div} feynman-prose
A walker reaching a new valley is the beginning of a transition, not the whole transition. It must remain useful long enough to contribute descendants, and those descendants must survive their own motion. Meanwhile, the old valley can continue sending walkers back. Separating discovery, establishment, residence and crossing gives each mechanism a quantity we can bound. It also explains why one successful scout can matter greatly without making every discovery a takeover. The reward and diversity rules decide the balance through the actual donor probabilities.
:::

:::{prf:definition} Stopping events on the full swarm
:label: def-slc-stopping

For $B\subset D$ in the killed case, let $Y_B(S)$ count alive walkers in $B$;
for a conservative configuration count all walkers. Fix $K\in\{1,\ldots,N\}$ and
$\ell\ge1$. Discovery is $\tau_{B,1}=\inf\{n:Y_B(S_n)\ge1\}$, establishment is
$\tau_{B,K}=\inf\{n:Y_B(S_n)\ge K\}$, and sustained establishment is the first
$n$ for which $Y_B(S_{n+j})\ge K$ for $0\le j<\ell$. Its completion time is
$n+\ell-1$; the starting index is generally not a stopping time when $\ell>1$; for
$\ell=1$ it is the establishment stopping time.

Population transfer from $A$ to $B$ means $\tau_{B,K}$ from an initial law supported
on a declared $A$-population class. Residence is exit from a declared population
set $\mathcal G$. Track $\tau_\dagger$ and exit from any localization class
separately. All conditional probabilities below use the full history, not an
assumed Markov law of basin labels.
:::

:::{prf:theorem} Actual-noise landing and population amplification of discovery
:label: thm-slc-landing

At the final position-noise stage, suppose with conditional probability at least
$g$, at least $M$ rows have centres $m_i$ satisfying $|m_i-y|\le D_0$. Let
$B(y,r)\subset B$, and in the killed case also $B(y,r)\subset D$.
For independent final noises of standard deviation $s>0$, put

$$
p=|B(0,r)|(2\pi s^2)^{-d/2}
 \exp[-(D_0+r)^2/(2s^2)].
$$

Then, for $1\le K\le M$,

$$
\Pr\{Y_B(S_{n+1})\ge K\mid\mathcal H_n\}
\ge g\sum_{j=K}^M{M\choose j}p^j(1-p)^{M-j}.
$$

In particular discovery has probability at least $g[1-(1-p)^M]$.
The good-row event may depend on all preceding marks, collisions and noises,
but not on these final position innovations.

*Proof.* On the good event, condition on all pre-final data and choose a measurable
set of $M$ good rows. The Gaussian density throughout the ball is at least the
displayed infimum. Their terminal memberships are conditionally independent
Bernoulli variables with success probabilities at least $p$, and hence dominate
a binomial by coupling with independent uniforms. Integrate the conditional
bound. $\square$
:::

:::{prf:corollary} Explicit good-row constants in a localized swarm
:label: cor-slc-good-rows

If every copied source position has norm at most $R_0$ and every post-collision
velocity at most $V_c$, restrict independent latent jitters and OU Gaussians to
$|\sigma_J Z_i^J|\le J$ and $|Z_i^O|\le G$. Set

$$
R_x=R_0+J,\qquad
L=R_x+B V_c+\eta M_{B(0,R_x)}+cqG.
$$

All final centres then have norm at most $L$. If
$p_J=\Pr(|\sigma_J Z|\le J)$, $p_O=\Pr(|Z|\le G)$, one admissible choice in
{prf:ref}`thm-slc-landing` is $M=N$, $g=(p_Jp_O)^N$, $D_0=L+|y|$.
An unused latent jitter can be assigned to rows that do not clone.

*Proof.* Substitute the bounds in $x_2=x+Bv+\eta F(x)+cqZ^O$. The latent row
innovations are independent of the preceding graph; the stated good event has
the claimed probability. Conditional shared collision dependence does not change
the deterministic bound $V_c$. $\square$
:::

:::{prf:theorem} Conditional hazards, finite-budget crossings and residence
:label: thm-slc-hazard

Let a target population set be tested every $b$ updates. If, on each unsuccessful
history, its probability of being reached during the next block is at least
$p>0$, then

$$
\Pr\{\tau>kb\}\le(1-p)^k,\qquad \mathbb E\tau\le b/p.
$$

For $0<p<1$, success probability at least $1-\delta$ follows after
$b\lceil\log(\delta)/\log(1-p)\rceil$ updates. For $p=1$, one block suffices.
If the bound holds only before localization exit $\sigma$, then

$$
\Pr\{\tau>kb\}\le(1-p)^k+\Pr\{\sigma\le kb\}.
$$

If each step from a residence set has exit probability at most $u\in[0,1]$, then
$\Pr\{\sigma>n\}\ge(1-u)^n$ from that set. If discovery succeeds with probability
$p_d$ in a block and, conditional on its discovery history, establishment succeeds
within a further $b_e$ steps with probability $p_e$, the combined success
probability is at least $p_dp_e$. A subsequent $\ell$-step residence guarantee
multiplies this lower bound by $(1-u)^\ell$.

*Proof.* Iterated conditional expectation bounds successive failure probabilities
by $1-p$. Sum survival probabilities in blocks to bound $\mathbb E\tau$.
Before $\sigma$, apply the same argument to the event that neither success nor
localization exit occurred, then add $\Pr(\sigma\le kb)$. For a killed configuration, include extinction in $\sigma$ unless the hypothesis
really holds on every unsuccessful history; an absorbing dead swarm has no positive
discovery hazard. The residence argument
uses conditional survival probabilities at least $1-u$. For composed events,
condition at the discovery and establishment stopping times. No independence
between these events is assumed. $\square$
:::

:::{prf:lemma} Residence from a stopped moment estimate
:label: lem-slc-residence

Let $\mathcal G$ be a population set, $V\ge0$ on the absorbing extension, and
$V\ge R>0$ on every state reached when leaving $\mathcal G$, including extinction
if counted as exit. Suppose $PV\le V+b$ on $\mathcal G$. For
$\sigma=\inf\{n:S_n\notin\mathcal G\}$ and $S_0\in\mathcal G$,

$$
\Pr\{\sigma\le n\}\le\min\{1,(\mathbb EV(S_0)+nb)/R\}.
$$

*Proof.* Conditional increments of $V(S_{n\wedge\sigma})$ have expectation at
most $b\mathbf1_{\{\sigma>n\}}$. Induction gives expectation at most
$\mathbb EV(S_0)+nb$. On exit by $n$, the stopped observable is at least $R$;
Markov's inequality proves the result. Assume $V$ is finite on reachable states and $\mathbb EV(S_0)<\infty$.
The stopped inequality itself inductively proves integrability: its next
expectation is at most the current one plus $b$. Thus no optional-stopping
limit or unstated integrability argument is needed. An arbitrary basin need not have such an exit observable. $\square$
:::

:::{prf:proposition} Exact full-state committor and coarse-graining error
:label: prop-slc-committor

Let $\mathcal A,\mathcal B$ and the declared failure set be pairwise disjoint
population sets, and kill on their union. On the remaining states let $K$ be the restricted
kernel and $r(S)=P(S,\mathcal B)$. The probability of hitting $\mathcal B$ before
$\mathcal A$ or failure is the minimal nonnegative solution

$$
h=r+Kh=\sum_{j\ge0}K^jr.
$$

The truncation after $J$ terms has error at most $K^J1$. If $K^b1\le1-p$, that
error is at most $(1-p)^{\lfloor J/b\rfloor}$.

For a finite partition $\Gamma$ of full states, suppose a stochastic matrix $T$
satisfies $\|P(S,\Gamma\in\cdot)-T(i,\cdot)\|_{\rm TV}\le\varepsilon$ for every
$S$ in cell $i$. Then at $n$ steps the actual label marginal and the chain with
matrix $T$ and the same initial labels differ in TV by at most $n\varepsilon$.

*Proof.* Decompose first hitting according to its step and apply monotone
convergence. Iterating any nonnegative solution bounds it below by each partial
sum. The remaining event requires survival for $J$ steps. The block estimate
follows by iteration. For the final assertion, conditional on any label history,
the next actual label law is a mixture of row laws each within $\varepsilon$ of
$T(i,\cdot)$. Sequential maximal couplings fail with probability at most
$n\varepsilon$. Thus a basin table requires a quantified closure bound unless
$\varepsilon=0$. $\square$
:::

(sec-slcpn-structural-passages)=
### 4.1. Basin, passage and mixture-region communication from structural data

:::{div} feynman-prose
Draw a route through the landscape using measurable waypoint regions. Some
waypoints lie inside a basin, some occupy a narrow passage, and a mixture
waypoint specifies populations distributed among several pieces. To cross an
edge of this route, the actual update must place the required number of walkers
in its target pieces. Gaussian landing bounds make the dependence on distance,
volume, noise and population size explicit.

Each edge estimate is conditional on every possible successful history, so the
bounds can be composed despite shared ancestors and selection. Discovery,
partial establishment and complete transfer have different probabilities and
different prerequisites for the next edge. Repeated attempts also spend the
tail budget: more time permits more attempts and more opportunities to leave
the controlled region. A narrow waypoint weakens this particular route's bound;
it does not establish that every possible trajectory must pass through it.
:::

:::{prf:definition} Declared regional communication data
:label: def-slcpn-data

Use the actual complete canonical kernel $P_N$ and parameter register
{prf:ref}`def-slc-parameter-register`, with $\tau^2=c^2q^2+s^2>0$.
Let $C\subset\mathbb R^d$ be a nonempty bounded measurable source region,
$J\ge0$ a declared jitter cutoff, and
$C^{[J]}=\{x:\operatorname{dist}(x,C)\le J\}$. Require a specified measurable,
finite-valued force on all reachable states. Define the regional force profile

$$
M_{C,J}=\sup_{x\in C^{[J]}}|F(x)|\in[0,\infty].
$$

For a target piece $A\subset B(z_A,r_A)$ of finite positive Lebesgue volume,
define the sharper structural displacement profile and a computable envelope

$$
D(C,A,J)=\sup_{x\in C^{[J]}}|x+\eta F(x)-z_A|+BV_c,
$$

$$
\overline D(C,A,J)=R_C+J+|z_C-z_A|+\eta M_{C,J}+BV_c
\quad\text{when }C\subset B(z_C,R_C).
$$

Then $D\le\overline D$. Either may be used below; write $D_A$ for the chosen
upper bound. Put

$$
p_J=\begin{cases}
\displaystyle\frac1{\Gamma(d/2)}\int_0^{J^2/(2\sigma_J^2)}
 t^{d/2-1}e^{-t}\,dt,&\sigma_J>0,\\
1,&\sigma_J=0,
\end{cases}
\qquad
p(C,A;J)=p_J|A|(2\pi\tau^2)^{-d/2}
 e^{-(D_A+r_A)^2/(2\tau^2)}.
$$

If $D_A=\infty$, define $p(C,A;J)=0$; if $\sigma_J>0$ and $J=0$, it is also
zero. These values mean the chosen regional certificate supplies no positive
rate, not that the dynamics or transition are undefined. In a killed model all
target pieces used as successful waypoints must lie inside the valid domain.
The full population source class $\mathcal G_C$ consists of states with all
input positions in $C$ and the configured input velocity cap. Thus every copied
source lies in $C$, independently of the accepted graph. In the killed setting
these source classes are all alive; extinction is an uncontrolled exit.
:::

:::{prf:lemma} A structural landing bound valid for every accepted cloning graph
:label: lem-slcpn-landing

Under {prf:ref}`def-slcpn-data`, every $S\in\mathcal G_C$ satisfies

$$
P_N(S,Y_A\ge K)\ge
\sum_{l=K}^N{N\choose l}p(C,A;J)^l[1-p(C,A;J)]^{N-l},
\qquad 1\le K\le N.
$$

In particular the lower bounds for discovery and complete transfer are
$1-(1-p)^N$ and $p^N$, respectively. For complete transfer the successful
output belongs to $\mathcal G_A$, so the estimate can be composed with the
next regional estimate without imposing any law on the basin labels.
:::

:::{prf:proof}
Condition on all companion, fitness, gate and collision variables. The exact
position formula in {prf:ref}`lem-slc-position-gaussian` is
$X_i'=x_i+\eta F(x_i)+Bv_i^c+\tau Z_i$, with
$x_i=y_i+C_i\sigma_J Z_i^J$, $y_i\in C$ and $|v_i^c|\le V_c$.
On $|\sigma_J Z_i^J|\le J$ the Gaussian mean is within $D_A$ of $z_A$.
Throughout $A$ its density is therefore at least
$(2\pi\tau^2)^{-d/2}e^{-(D_A+r_A)^2/(2\tau^2)}$.
Integrating over $A$ and the independent latent jitter event gives probability
at least $p$. For a row that does not copy, its latent jitter may still be
sampled and restricted without changing the output law. Conditional row
innovations are independent, so their indicators dominate independent
Bernoulli$(p)$ variables. Integrate over the shared preparation variables.
The cap supplies the next input velocity bound, and complete transfer places
all next copied sources in $A$. No derivative bound on $F$ was used.
:::

:::{prf:theorem} Communication through declared passage and mixture regions
:label: thm-slcpn-network

Choose a finite sequence of bounded measurable waypoint regions
$C_0,C_1,\ldots,C_L$, each with specified cutoff and force profile as above,
and integers $m_1,\ldots,m_L\ge0$. Set

$$
p_l=p(C_{l-1},C_l;J_l),\quad
r_l=p(C_l,C_l;\widehat J_l),\quad
b=L+\sum_{l=1}^L m_l,
\qquad
P_{\rm route}=\prod_{l=1}^L p_l^N r_l^{Nm_l}.
$$

Starting from $\mathcal G_{C_0}$, the probability of transferring completely
into each successive $C_l$ and remaining there for the next $m_l$ updates
before continuing is at least $P_{\rm route}$. The final completion time is
$b$ updates, physical time $bh$, and work $Nb$ walker updates.
A stronger proved retention lower bound $s_l$ can replace $r_l$ wherever
available, in which case replace $r_l^{Nm_l}$ by $s_l^{Nm_l}$.

For mixture waypoints, let $A_{l,1},\ldots,A_{l,k_l}$ be pairwise disjoint
bounded pieces and choose counts $n_{l,j}\ge0$ summing to $N$. Set
$C_l=\bigcup_j A_{l,j}$. The edge probability may instead be taken as

$$
q_l=\frac{N!}{\prod_j n_{l,j}!}
       \prod_{j:n_{l,j}>0}p(C_{l-1},A_{l,j};J_l)^{n_{l,j}}.
$$

Then $\prod_l q_l$ bounds below the probability of the prescribed sequence of
mixture counts at successive updates. Identical mixture edges can be repeated
to prescribe dwell times. A terminal discovery or establishment event may replace
the final complete-transfer factor by its binomial tail, but a subsequent edge
requiring all copied sources in the target cannot follow a count threshold
$K<N$ without an additional source-availability estimate.
:::

:::{prf:proof}
On each successful prefix the next input belongs to its declared source class.
The preceding lemma gives conditional probability at least $p_l^N$ for complete
transfer and $r_l^N$ for every required extra residence step. Repeated use of
the tower property multiplies these deterministic lower bounds. The lower
bound is uniform over every successful-prefix history, regardless of shared
ancestor information, fitness statistics or collision components.

For a mixture edge, condition on the full prepared graph. Each output row has
probability at least $p(C_{l-1},A_{l,j};J_l)$ of entering piece $j$. Conditional
row independence implies that a particular assignment of rows to pieces with
counts $n_{l,j}$ has probability at least the displayed product. There are
$N!/\prod_j n_{l,j}!$ such assignments, and their events are disjoint because
the pieces are disjoint. Sum over assignments, then average over the graph.
Every successful assignment has all positions in $C_l$, which verifies the
next edge's source hypothesis. The terminal count assertion is exactly the
binomial conclusion of the preceding lemma. This calculation is a full-state
conditional probability argument, not a Markov-chain approximation on labels.
:::

:::{prf:corollary} Quantitative passage width and distance dependence
:label: cor-slcpn-width

Suppose a target passage piece contains a Euclidean cylinder
$[0,\ell]\times B_{d-1}(0,w)$ after a rigid motion and is contained in its
declared target ball. For $d\ge2$, its landing factor can use the explicit
volume

$$
|A|\ge\ell\frac{\pi^{(d-1)/2}}{\Gamma(1+(d-1)/2)}w^{d-1}.
$$

Hence its complete-transfer edge factor is bounded below by

$$
\left[
 p_J\ell\frac{\pi^{(d-1)/2}w^{d-1}}{\Gamma(1+(d-1)/2)}
 (2\pi\tau^2)^{-d/2}
 e^{-(D_A+r_A)^2/(2\tau^2)}
\right]^N.
$$

For discovery the corresponding factor is $1-(1-p)^N$, and for population
fraction $a\in(0,1]$ it is the binomial tail with $K=\lceil aN\rceil$.
The dependence on width is therefore explicit, as is the Gaussian penalty for
the displacement between successive pieces. For a route of $L$ such pieces,
the logarithm of the complete-transfer lower bound is the explicit sum

$$
\log P_{\rm route}=N\sum_{l=1}^L
 \left[\log p_{J_l}+\log|C_l|-\frac d2\log(2\pi\tau^2)
       -\frac{(D_l+r_l^{\rm geom})^2}{2\tau^2}\right]
 +N\sum_{l=1}^L m_l\log s_l,
$$

when all factors are positive and $s_l$ denotes the chosen row retention
factor. Here $r_l^{\rm geom}$ is the geometric enclosing radius, distinct from
any probability. An empty or zero-volume passage makes this particular landing
certificate zero. Since the configured dynamics can jump between disjoint
regions, this lower bound is not an upper bound on all possible communication
paths; mandatory passage would require a separate geometric statement about
the actual kernel.
:::

:::{prf:proof}
The cylinder volume is the product of its length and cross-sectional ball
volume. Insert that smaller volume into the preceding lemma and theorem;
all other factors are nonnegative. Taking logarithms of their finite product
gives the sum. The formulas distinguish one successful row from $N$ successful
rows, and do not discard that population-size dependence.
:::

:::{prf:corollary} Repeated route attempts with an explicit tail budget
:label: cor-slcpn-repeated-route

Let $C$ be a bounded controlled envelope containing every waypoint, and replace
the first edge source by $C$, so that the same route success bound
$P_{\rm route}=p>0$ holds from every controlled input. Define
$\sigma=\inf\{n:S_n\notin\mathcal G_C\}$, and let $\tau$ be first arrival
in the terminal target population set. For $k$ route blocks,

$$
\Pr(\tau>kb)\le(1-p)^k+\Pr(\sigma\le kb).
$$

If a nonnegative stopped observable satisfies the explicit structural tail
certificate $P_NV\le V+b_V$, $b_V\ge0$, on $\mathcal G_C$, $V\ge R_V>0$ on its exit
states, and $v_0=\mathbb EV(S_0)<\infty$, then

$$
\Pr(\tau\le kb)\ge
\left[1-(1-p)^k-\min\{1,(v_0+kb\,b_V)/R_V\}\right]_+.
$$

Thus a requested failure budget $\delta\in(0,1)$ is certified by any integer
$k$ satisfying the two explicit inequalities

$$
(1-p)^k\le\delta/2,\qquad
v_0+kb\,b_V\le R_V\delta/2.
$$

For $0<p<1$, the first requires
$k\ge\lceil\log(\delta/2)/\log(1-p)\rceil$; for $p=1$, $k=1$ suffices.
If $b_V>0$, the second permits only
$k\le\lfloor(R_V\delta/2-v_0)/(b\,b_V)\rfloor$.
An empty interval means these supplied regional and tail certificates do not
certify the requested global success probability. Physical time and work are
$kbh$ and $Nkb$ respectively.
:::

:::{prf:proof}
On every unsuccessful block history that remains controlled, following the
specified route in the next block has conditional probability at least $p$.
All of that successful route remains in the envelope. Consequently the
probability of neither success nor envelope exit through $k$ blocks is at most
$(1-p)^k$ by induction. Add the envelope-exit probability. The stopped drift
argument of {prf:ref}`lem-slc-residence` bounds that probability by
$(v_0+kb\,b_V)/R_V$. The two displayed budget inequalities allocate at most
$\delta/2$ to each contribution. Solving them uses only logarithms and integer
rounding; no unknown optimal convergence rate enters the result.
:::

(sec-slc-tv)=
## 5. Explicit certificates for total-variation convergence

:::{prf:remark} Keystone route and the finite-swarm alternative
:label: rem-slc-tv-route-separation

The population-size-uniform *one-step route* in this chapter is
{prf:ref}`thm-slc-keystone-tagged-tv`: use the signed Keystone pressure
inside the full update, then the exact positional Gaussian law to
convert the normalized paired error into TV for fixed sampled marked
positions. The all-row minorization and Harris theorems below are
separate finite-swarm results. Their product minorization constant is
not a coefficient in the Keystone argument and is not used in
(SCK.TV1)--(SCK.TV3). A full-swarm TV claim and a sampled-row TV
claim have different targets and must retain their respective rates.
:::

:::{div} feynman-prose
To compare two evolving laws, we need more than a picture of walkers gathering near a minimum. A drift estimate controls excursions; a minorization estimate supplies a common probabilistic component where different starting states can lose their distinction. An entropy estimate can provide another route when its reference law and dissipation are identified. This section explains how certified versions of these ingredients imply total-variation bounds. The resulting conclusion belongs to the specified conservative or conditioned process, with the hypotheses and constants carried along.
:::

:::{prf:theorem} Gaussian common mass for the complete conservative step
:label: thm-slc-minorization

Consider the conservative canonical gas with isotropic $q,s>0$, independent
row OU/final innovations, $q,s,V_c$ from {prf:ref}`def-slc-parameter-register`, the radial injective 1-Lipschitz cap, and globally
$L_F$-Lipschitz force satisfying $\lambda=1-c^2L_F>0$. Suppose on a declared
input set $\mathcal C$ all copied source positions have norm at most $R_0$ and
all post-collision velocities at most $V_c$, irrespective of the sampled graph.
Choose jitter radius $J$ with per-row latent probability
$p_J=\Pr(|\sigma_JZ|\le J)>0$, independent across rows and of the preceding
graph; unused jitters are assigned latently. Choose a final position ball $B(0,r)$
and a pre-cap velocity ball $B(0,u)$. Define

$$
\begin{aligned}
R_x&=R_0+J,& F_x&=M_{B(0,R_x)},\\
R_1&=R_x+c(V_c+cF_x),& F_1&=M_{B(0,R_1)},\\
Q_u&=(u+cF_1)/\lambda,& m_v&=a(V_c+cF_x),\\
k_v&=(2\pi q^2)^{-d/2}
 e^{-(Q_u+m_v)^2/(2q^2)}(1+c^2L_F)^{-d},\\
k_x&=(2\pi s^2)^{-d/2}
 e^{-(r+R_1+cQ_u)^2/(2s^2)},\\
\epsilon&=\left[p_J |B(0,u)|\,|B(0,r)|\,k_vk_x\right]^N.
\end{aligned}
$$

Let $\eta$ be the product over rows of independent uniform positions in $B(0,r)$
and capped uniform velocities from $B(0,u)$. Then
$P(S,\cdot)\ge\epsilon\eta(\cdot)$ for every $S\in\mathcal C$.
For a killed terminal-domain kernel the same surviving subkernel lower bound holds
when the target position ball lies inside $D$; it is not a conservative invariant-law
conclusion for that kernel.

*Proof.* Condition on the graph and bounded-jitter event, of probability $p_J^N$.
For each row $|x_1|\le R_1$. Condition further on all prepared rows (including their jitter and collision outputs), so that the kinetic innovations remain independent. For the fixed $x_1$, the map
$T(v)=v+cF(x_1+cv)$ satisfies
$|T(v)-T(w)|\ge\lambda|v-w|$. It is onto: solving
$v=z-cF(x_1+cv)$ is a contraction with coefficient $c^2L_F<1$.
Indeed $|T(0)|=c|F(x_1)|\le cF_1$, so
$\lambda|v|\le|T(v)-T(0)|\le u+cF_1$ when $T(v)=z$, proving $|v|\le Q_u$.
The forward map has $\operatorname{Lip}(T)\le1+c^2L_F$; its inverse has
$\operatorname{Lip}(T^{-1})\le\lambda^{-1}$. The change-of-variables formula for this bi-Lipschitz map bounds the
$v_3$ density below by $k_v$ on $B(0,u)$: the Gaussian $v_2$ mean has norm at most
$m_v$, and the Jacobian determinant is at most $(1+c^2L_F)^d$ almost everywhere.
Conditional on this $v_3$, the pre-final position is $x_1+cv_2$, of norm at most
$R_1+cQ_u$, so its final Gaussian density is at least $k_x$ on $B(0,r)$.
Push forward the velocity ball by the cap. Conditional independence of the row
noises gives the product lower bound, uniform over the preceding graph and jitter
values. Integrate them out. Marks on the target are prescribed by position.
$\square$
:::

:::{prf:theorem} Explicit Harris rate and time to TV accuracy
:label: thm-slc-tv-rate

Suppose a conservative complete kernel on the complete standard Borel swarm
space has a finite measurable $V\ge0$ and certified $PV\le rV+b$,
$0\le r<1$, $b\ge0$, and
$P(S,\cdot)\ge\epsilon\eta$ on $\{V\le R\}$, with $R>2b/(1-r)$ and
$0<\epsilon\le1$. Choose $\beta>0$ with $\beta(rR+2b)\le\epsilon$ and put

$$
\rho=\max\left\{1-\epsilon/2,
\frac{2+\beta(rR+2b)}{2+\beta R}\right\}<1.
$$

The weighted norm in this statement is
$\|\mu-\nu\|_\beta=\int(1+\beta V)\,d|\mu-\nu|$.
A completely specified choice is
$R=4b/(1-r)+R_{\rm ref}$ for any declared $R_{\rm ref}>0$ in the units of $V$,
and $\beta=\epsilon/[R(1+r)+2b]$. These choices satisfy the required inequalities.

There is a unique invariant probability $\pi$ within the finite-$V$-moment
class, and for every initial probability $\mu$ with $\mu V<\infty$,

$$
\|\mu P^n-\pi\|_{\rm TV}\le A_\mu\rho^n,
\quad A_\mu=\tfrac12\|\mu-\pi\|_\beta
\le\overline A_\mu:=1+\tfrac\beta2[\mu V+b/(1-r)].
$$

Here $1/2\le\rho<1$. For $0<\delta<1$,
$n\ge\lceil\log(\overline A_\mu/\delta)/(-\log\rho)\rceil$ suffices for
accuracy $\delta$ without knowing $\pi$. Uniformity in $N$
requires uniformity of every displayed ingredient.

*Proof.* For distinct $S,T$ use the pair cost $2+\beta[V(S)+V(T)]$, zero on the
diagonal. If $s_0=V(S)+V(T)>R$, any coupling gives expected cost at most
$2+\beta(rs_0+2b)$, whose ratio to $2+\beta s_0$ is bounded by the displayed
fraction (it decreases in $s_0$). If $s_0\le R$, couple the common minorization
part identically. The expectation is at most
$2(1-\epsilon)+\beta(rs_0+2b)\le2-\epsilon$, giving ratio at most
$1-\epsilon/2$. For any two probability measures, couple their common part diagonally and
the mutually singular remainders arbitrarily. Its expected pair cost is exactly
$\|\mu-\nu\|_\beta$: both remainders are charged $1+\beta V$ once. Conversely any
coupling has at least this cost, by testing bounded functions with absolute value
at most $1+\beta V$. Use the common-mass coupling above on the small set and a
product coupling outside; these are measurable kernels. Integration therefore
gives $\|\mu P-\nu P\|_\beta\le\rho\|\mu-\nu\|_\beta$.

Finite-$V$-moment probabilities form a closed subset of the Banach space of
signed measures with norm $\int(1+\beta V)d|\mu|$: positivity and total mass pass
to a norm limit. The drift maps this subset into itself. Starting from any
point mass with finite $V$, consecutive iterates have geometrically summable
norm differences and hence converge in this complete space. Contraction makes
the limit invariant and unique in the subset. Integrating drift under this
finite-moment invariant law gives $\pi V\le r\pi V+b$, hence the claimed bound.
Finally $\|\mu-\pi\|_\beta\le2+\beta(\mu V+\pi V)$ and unweighted TV is at most
half this weighted norm. Taking logarithms proves the explicit time bound. $\square$
:::

:::{prf:proposition} Entropy floors and conditioned convergence retain their type
:label: prop-slc-entropy-tv

For a specified invariant law $\pi$, suppose the complete evolution has a proved
entropy estimate $H_{n+1}\le q_HH_n+e_n$, $0\le q_H<1$, with
$H_n=H(\mu_n\mid\pi)$. Then

$$
\|\mu_n-\pi\|_{\rm TV}
\le\left[\tfrac12\left(q_H^nH_0+
 \sum_{j<n}q_H^{n-1-j}e_j\right)\right]^{1/2}.
$$

*Proof.* Iterate the scalar inequality and apply Pinsker. A nonzero uniform defect
only proves the corresponding entropy floor. $\square$

For the killed gas, use the surviving-block criterion of
{prf:ref}`thm-main-convergence` when its hypotheses hold, or the proved
canonical QSD theorem {prf:ref}`thm-chaos-canonical-finite-n-qsd` in its stated
quadratic configuration. A lower surviving mass alone does not supply its
necessary upper comparison or conditioned contraction. Static LSI, hypocoercive
entropy decay, conservative TV and QSD convergence retain their separate laws
and normalizations.
:::
:::{prf:lemma} A structural communication edge for the actual full kernel
:label: lem-slcr-gaussian-edge

Let $C_i$ be a declared population region whose entering positions satisfy
$\max_k|x_k|\le R_i$. Cloning and collisions may be fully active.
Choose a jitter cutoff $J>0$ and set $p_J=\Pr(|\sigma_JZ|\le J)$,
with $p_J=1$ if $\sigma_J=0$. This Gaussian-ball probability is
$\Gamma(d/2)^{-1}\int_0^{J^2/(2\sigma_J^2)}t^{d/2-1}e^{-t}\,dt$
when $\sigma_J>0$.
Let

$$
R_0=R_i+J,\quad M_0=\sup_{|x|\le R_0}|F(x)|,\quad
R_1=R_0+c(V_c+cM_0),\quad M_1=\sup_{|x|\le R_1}|F(x)|,
\quad m_v=a(V_c+cM_0).
$$

For a target region $C_j$, prescribe position balls
$B(b_{j,k},r_{j,k})$, $k=1,\ldots,N$, a pre-cap velocity radius $u_j>0$,
and a number $Q_{ij}>0$. Let $L_{ij}$ be a proved force Lipschitz bound
on $B(0,R_1+cQ_{ij})$. Require the numerical inequalities

$$
\lambda_{ij}=1-c^2L_{ij}>0,\qquad
u_j+cM_1\le\lambda_{ij}Q_{ij},\qquad q,s>0.
$$

In the second inequality $u_j$ is the declared velocity radius.
Let $\nu_j$ be the product of the uniform target-position distributions
and the cap-pushforwards of independent uniforms on $B(0,u_j)$.
Assume this explicitly described measure is supported in $C_j$.
Define

$$
\begin{aligned}
k_{v,ij}&=(2\pi q^2)^{-d/2}
 \exp[-(Q_{ij}+m_v)^2/(2q^2)](1+c^2L_{ij})^{-d},\\
k_{x,ij,k}&=(2\pi s^2)^{-d/2}
 \exp[-(|b_{j,k}|+r_{j,k}+R_1+cQ_{ij})^2/(2s^2)],\\
\epsilon_{ij}&=p_J^N
 \prod_{k=1}^N[v_d(u_j)v_d(r_{j,k})k_{v,ij}k_{x,ij,k}].
\end{aligned}
$$

Then $P(S,\cdot)\ge\epsilon_{ij}\nu_j$ for every $S\in C_i$.
The source and target regions may represent mixed-basin populations;
they need not place every row in the same spatial basin.
:::

:::{prf:proof}
Expose the complete copying graph, collision rotations, frozen donor
choices and all latent jitter innovations. The event that the latter
all have norm at most $J$ has probability $p_J^N$, independently of
the graph; unused jitters can be sampled as latent marks. On this event
post-copy positions have norm at most $R_0$, irrespective of which
edges were accepted, and collision velocities have norm at most $V_c$.
Thus the first drift position satisfies $|x_1|\le R_1$ and the mean
of the row's OU velocity $v_2$ has norm at most $m_v$.

For each $|w|\le u_j$, the map $v\mapsto w-cF(x_1+cv)$ sends
$\overline B_{Q_{ij}}$ to itself: its norm is at most
$u_j+cM_1+c^2L_{ij}Q_{ij}\le Q_{ij}$. Its Lipschitz constant is
$c^2L_{ij}<1$, so successive iteration is Cauchy and gives a fixed
point in that ball. Equivalently $T(v)=v+cF(x_1+cv)$ reaches every
such $w$. On the ball, $T$ has Lipschitz constant at most
$1+c^2L_{ij}$ and inverse Lipschitz constant at most
$1/\lambda_{ij}$. The change-of-variables inequality consequently
bounds the pre-cap output velocity density below by $k_{v,ij}$.
Every preimage used has $|v_2|\le Q_{ij}$, so the conditional final
position density on $B(b_{j,k},r_{j,k})$ is at least $k_{x,ij,k}$.
The OU and final position innovations are independent across rows
conditional on the exposed preparation. Multiplying their lower joint
densities and applying the cap proves domination by the displayed
product probability, uniformly over that preparation. Integration over
the good jitter event proves the claim. The argument uses no
independence between cloning events.
:::

:::{prf:theorem} Basin, transition-region and tail certificates assembled into a TV rate
:label: thm-slcr-structural-path-rate

For the actual finite-$N$ conservative kernel $P$, choose a declared
block length $m_0\ge1$ and set $\mathscr P=P^{m_0}$. Let
$\mathcal V=1+W$ and suppose proved structural estimates give

$$
\mathscr P\mathcal V\le r\mathcal V+b_V,
\qquad 0\le r<1,\quad b_V<\infty.
$$

For example, {prf:ref}`prop-slcr-block-recovery` supplies these constants
when its regional finite-step flux envelopes close. The controlled-class
selection bound alone does not supply this global premise. When its recovery
defects remain uncontrolled, retain the accumulated-defect conclusion instead.

Choose $R>2b_V/(1-r)$ and a finite measurable cover
$\{\mathcal V\le R\}\subset\bigcup_{i=1}^M C_i$.
For each $i$, supply a path of exactly $L\ge1$ certified edges

$$
i=i_0\longrightarrow i_1\longrightarrow\cdots
 \longrightarrow i_L=\star,
$$

where every edge is a bound for $\mathscr P$. For $m_0=1$ use
{prf:ref}`lem-slcr-gaussian-edge`; for larger $m_0$, concatenate $m_0$
verified complete-update edges. One may also use another
already proved complete-kernel domination with a strictly positive edge
coefficient and the same target probability $\nu_j$ supported in $C_j$. Shorter paths can be padded
using a certified self-loop at $\star$. Put

$$
\epsilon_*=\min_i\prod_{k=0}^{L-1}\epsilon_{i_ki_{k+1}},\quad
r_L=r^L,\quad b_L=b_V\frac{1-r^L}{1-r},\quad
\beta=\frac{\epsilon_*}{R(1+r_L)+2b_L},
$$

$$
\rho=\max\left\{1-\frac{\epsilon_*}2,
 \frac{2+\beta(r_LR+2b_L)}{2+\beta R}\right\}<1.
$$

Then the full kernel has a unique invariant law $\pi$, and

$$
\|\mu P^n-\pi\|_{\rm TV}
\le\left[1+\frac\beta2\left(\mu\mathcal V+
 \frac{b_V}{1-r}\right)\right]\rho^{\lfloor n/(m_0L)\rfloor}
$$

for every $\mu\mathcal V<\infty$. A sufficient iteration count for
TV error $\delta\in(0,1)$ is

$$
n=m_0L\left\lceil\frac{\log(A_\mu/\delta)}{-\log\rho}\right\rceil,
\qquad t=nh,
\qquad A_\mu=1+\frac\beta2
 \left(\mu\mathcal V+\frac{b_V}{1-r}\right).
$$

Every graph edge is a verified full-kernel measure bound. The region
labels are not assumed to form a Markov chain, and the products are
products of deterministic certified edge bounds, not products of
unconditioned state-dependent comparison matrices.
:::

:::{prf:proof}
Successive conditional integration along a supplied path gives
$\mathscr P^L(S,\cdot)\ge(\prod_k\epsilon_{i_ki_{k+1}})\nu_\star$
for $S\in C_i$: after each dominated step its reference probability
is supported in the next region. Taking the minimum proves the common
minorization on $\{\mathcal V\le R\}$. Iterating the drift gives
$\mathscr P^L\mathcal V\le r_L\mathcal V+b_L$. The chosen $R$ obeys
$R>2b_L/(1-r_L)$, and the chosen $\beta$ obeys
$\beta(r_LR+2b_L)\le\epsilon_*$.

The weighted-variation argument in this chapter therefore contracts the
sampled kernel $\mathscr P^L$ with precisely the displayed $\rho$. It gives a
unique invariant probability with moment at most $b_V/(1-r)$.
For each deterministic finite state the estimate gives convergence in TV.
Integrating that convergence against any initial probability uses domination
by one, so an invariant probability for $\mathscr P^L$ cannot differ from $\pi$
even without a moment assumption. Since $\pi P$ is also invariant for
$\mathscr P^L$, uniqueness makes $\pi$ invariant for $P$. The remaining fewer than
$m_0L$ Markov updates do not increase TV. Substituting the moment bound
into the weighted initial distance gives $A_\mu$ and the stated rate.
The parameters $N$, dimension, Gaussian noise, friction, time step,
cap, collision strength and jitter occur in the edge bounds; selection,
fitness and landscape profiles additionally occur in the drift and
its explicitly retained defects.
:::

(sec-slc-meanfield)=
## 6. Quantitative mean-field transfer and stationary phases

:::{div} feynman-prose
Start with a chosen population and follow its limiting evolution. A long-time mean-field theorem can describe that trajectory and the phase it approaches without requiring populations started elsewhere to approach the same phase. The task is to control approximation errors along the relevant evolution, establish arrival and residence where claimed, and specify the order of the population and time limits. Global contraction is one sufficient route when there is a single attractor. Several attracting phases call for a phase-dependent description. A spatial well is a region of positions; a population phase is a law of the whole population evolution, and the two need not correspond one for one. Likewise, long residence in different wells at finite population size does not by itself imply several exact finite-particle invariant laws.
:::

:::{prf:definition} A weak empirical metric and one-step constants
:label: def-slc-empirical-metric

Choose a countable convergence-determining family $(\varphi_j)$ of continuous
functions on the complete marked row space, $\|\varphi_j\|_\infty\le1$, and set

$$
d(\mu,\nu)=\frac12\sum_{j\ge1}2^{-j}|\mu\varphi_j-\nu\varphi_j|\le1.
$$

Take the family to separate probability laws; this defines a bounded metric for
weak convergence. Under the canonical input hypotheses of Chapter 9, put
$A=2[A_D+10M_2(C)+1]$, with $C,A_D,M_2$ as defined there, and let $B_*$ be the
constructive constant of {prf:ref}`thm-chaos-canonical-quantitative-bias`.
The following explicit dependency chain evaluates them. Use $D_0,\kappa_D,
\kappa_C,F_{\max}$ above, $F_{\min}=\eta_r^{p_r}\eta_s^{p_s}$, and a declared
alive-fraction floor $m_*>0$. Set

$$
\begin{gathered}
S_*=\sqrt{D_0^2+\delta_D^2},\quad C=2/(\kappa_Cm_*),\quad
M_1(t)=e^{2t},\quad M_2(t)=(1+2t)e^{4t},\quad
M_3(t)=(1+6t+3t^2)e^{8t},\\
L_q=S_*/(m_*\sigma_s)+3S_*^3/(2m_*\sigma_s^3),\\
H_s=(A_r+\eta_r)^{p_r}\frac{A_sp_s}{4}
 \max\{\eta_s^{p_s-1},(A_s+\eta_s)^{p_s-1}\},\\
L_a=\max\{[s_c(F_{\min}+\epsilon_c)]^{-1},
 (F_{\max}+\epsilon_c)/[s_c(F_{\min}+\epsilon_c)^2]\},\\
B_{\rm inf}=C+2L_aH_sL_q,\quad
A_D=9M_2(2C)[(1+B_{\rm inf})^2+B_{\rm inf}]+\max\{1,4B_{\rm inf}^2\},\\
D_D=2/(\kappa_Dm_*),\quad
A_T=(m_*^{-1}+D_D^2)(2S_*^2/\sigma_s^2+5S_*^6/\sigma_s^6),\\
L_T=2L_aH_s,\quad A_{\rm exp}=1+C+D_D,\quad
N_0=\lceil(8A_{\rm exp})^{6/5}\rceil,\\
B_*=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
       +64A_{\rm exp}^2+16M_3(C)+\sqrt{N_0}.
\end{gathered}
$$

Take $H_s=0$ when $p_s=0$. These are the constructive constants of the cited
Chapter 9 proof, with names changed to avoid confusing exploration with the
variance constant $A$. The component moments control the ordered collision
forest, and the final kinetic kernel is a common Markov postprocessing of
matched prepared states; it cannot increase this bounded-test coupling error.
Consequently kinetic noise, friction and force do not occur in this particular
one-step error constant. They still determine the population map, its stability
and its moment class. No claim of independence of their long-time effect follows.

The imported theorem retains its canonical hypotheses: current-state Gaussian
companions, positive regularizers, the specified ordered collision rule, no
history or viscosity force, and the source's regularity and moment assumptions
for its population map. Its unbounded stationary result uses its stated quadratic
configuration. A regional measurable force alone does not discharge those
hypotheses. These are update constants, not unknown optimal convergence rates.
:::

:::{prf:proposition} Explicit bounded-domain alive floor
:label: prop-slc-alive-floor

For the canonical terminal-killing gas assume $D\subset B(0,R_D)$ contains
$B(0,r_0)$, the entering state is nonextinct, every dead row is revived from an
alive donor, and prepared velocities obey $V_c$. Choose $J,G>0$, $q,s>0$ and
finite $M_{B(0,R_D+J)}$. Put

$$
\begin{gathered}
p=p_JG_d(G),\quad
L=R_D+J+B V_c+\eta M_{B(0,R_D+J)}+cqG,\\
p_0=v_d(r_0)(2\pi s^2)^{-d/2}
             e^{-(L+r_0)^2/(2s^2)},\quad m_*=p_0p/4,\\
\delta_N=e^{-pN/8}+e^{-p_0pN/16}.
\end{gathered}
$$

Then $\Pr\{N_{\rm alive}'/N<m_*\mid S\}\le\min(1,\delta_N)$.
In particular this bounds one-step extinction and supplies a next-step
input floor for the preceding constants. Over $n$ updates starting nonextinct,
the probability of a floor failure among the outputs $S_1,\ldots,S_n$ is at most $\min(1,n\delta_N)$. The initial nonextinct state need not itself satisfy the floor.
For the conservative gas use $m_*=1$ and zero extinction error.

*Proof.* Condition on the graph and collision preparation before the independent
latent jitter, OU and final-position innovations. The independent events
$|\sigma_JZ_i|\le J$, $|\xi_i|\le G$ have joint probability $p$ per row.
Each such row has final-position centre bounded by $L$, regardless of its donor.
The conditional final-position landing probability in $B(0,r_0)$ is at least
$p_0$, by the Gaussian density lower bound. Independence of final innovations
therefore gives binomial domination conditional on all earlier innovations.
For a binomial variable with mean $m$, exponential Markov at $t=\log2$ gives
$\Pr(X\le m/2)\le\exp[-(1-\log2)m/2]\le e^{-m/8}$;
the same bound holds for independent Bernoulli variables whose means have the
stated lower bound, by their product moment-generating functions.
The good-row count is at least $pN/2$ except with probability $e^{-pN/8}$.
Given at least that many good rows, fewer than $p_0pN/4$ landings have probability
at most $e^{-p_0pN/16}$. A union bound proves the first assertion, without
assuming independent walker trajectories. Iterate conditionally until the first
failure and take another union bound for the horizon assertion. $\square$
:::

:::{prf:lemma} Quantitative empirical one-step consistency
:label: lem-slc-empirical-error

Uniformly on the input class just specified, for the retained physical output,

$$
\mathbb E[d(L_N',\mathcal F_h(L_N))\mid S]
\le\varepsilon_N:=\frac{\sqrt{A+4B_*^2}}{2\sqrt N}.
$$

For independent initial rows of law $\mu_0$, $e_0\le1/(2\sqrt N)$:
each bounded test has sample-mean variance at most $1/N$, and the same weighted
Cauchy–Schwarz argument applies. Other initializations retain their actual $e_0$.

Changing the empirical law to a fixed probability on extinction adds at most its
conditional probability $\delta_N$. For input localization failure of probability
$p_n$, add at most $p_n$ to the unconditional estimate.

*Proof.* The mean-square bound in the cited bias theorem, followed by
Cauchy–Schwarz, bounds each test error by $\sqrt{A+4B_*^2}/\sqrt N$. Multiply by
$2^{-j}/2$ and sum by Tonelli. The metric is at most one, which bounds both changes
on exceptional events. This does not assert a dimension-free Wasserstein rate
or TV convergence of atomic empirical measures. $\square$
:::

:::{prf:remark} Mean-field consistency is relative to the initial law
:label: rem-slc-initial-law

The governing population law is already explicitly derived in
{prf:ref}`thm-mean-field-equation` and {prf:ref}`proof-mean-field-equation`:
the marked fitness law, accepted collision graph and root readout are composed
with the exact stages of {prf:ref}`def-baoab-update-rule`. Its exact weak balances
are {prf:ref}`thm-mass-conservation`. This section uses that established law to
study phase-dependent long-time behavior; it does not reopen its derivation.
The mean-field assertion compares $L_N(S_n)$ with
$\mu_n=\mathcal F_h^n(\mu_0)$ for its own initial population law $\mu_0$.
It does not require $\mathcal F_h^n(\mu_0)$ and
$\mathcal F_h^n(\nu_0)$ to approach one another for different initial laws.
Uniqueness of evolution means that the same specified initial law has one
solution under the same configured law. It is distinct from uniqueness of a
stationary law and from global attraction. For the fixed-step construction,
$\mu_{n+1}=\mathcal F_h(\mu_n)$ has a unique sequence whenever $\mathcal F_h$
is a single-valued map on its invariant domain: induction determines every
successive term from $\mu_0$. Different initial laws can give different sequences
and different stationary limits without any ambiguity in this evolution rule.

The continuous-time solution-map notation for a well-posed
McKean–Vlasov–Fokker–Planck evolution expresses the same distinction:
$\mu_t=\mathcal S_t\mu_0$, with $\mathcal S_{t+s}=\mathcal S_t\mathcal S_s$,
while $\mathcal S_t\mu_0$ need not approach $\mathcal S_t\nu_0$.
Multiple stationary solutions are compatible with uniqueness for each initial
condition. The volume gives the kinetic Fokker–Planck equation in
{prf:ref}`prop-fokker-planck-kinetic`, the complete fixed-step population equation
in {prf:ref}`thm-mean-field-equation`, and the exact spatial field balances in
{prf:ref}`thm-algorithmic-spatial-field-equations`. The present estimates retain
those specified evolution laws and their time conventions. Multiple phases
invalidate none of them. For random initial laws, evolve each realization
by the same map. Its distribution is pushed forward by that map; averaging the
realizations is generally not a solution with the averaged initial law because
the evolution is nonlinear.
If $\mu_0\in\mathcal A_i$ implies $\mathcal F_h^n(\mu_0)\to\pi_i$, distinct
fixed points $\pi_i$ are fully compatible with a single well-defined nonlinear
population map. The sets $\mathcal A_i$ are attraction basins in population-law
space, not the spatial basin regions $B_i$.

Indeed if two fixed points satisfy $d(\pi_i,\pi_j)>0$, a global estimate
$d(\mathcal F_h^r\mu,\mathcal F_h^r\nu)\le qd(\mu,\nu)$ with $q<1$
is impossible: substitute the two fixed points to obtain $1\le q$.
Thus such contraction is an optional single-attractor specialization, not a
requirement on the structural programme. Phase arrival, residence and transition
estimates provide a different route, proved next.
:::

:::{prf:theorem} Phase-resolved long-time approximation from arrival and residence
:label: thm-slc-phase-residence-mf

Use the complete fixed-step kernel $P_N$ on the marked swarm space and the bounded
metric $d\le1$ of {prf:ref}`def-slc-empirical-metric`. If needed, extend the killed
chain with its absorbing cemetery state and assign that state a fixed empirical
law; count killing as failure below. Let $\pi_i$ be a fixed point of
$\mathcal F_h$, and let the specified initial population satisfy

$$
d(\mu_n,\pi_i)\le a_i(n),\qquad \mu_n=\mathcal F_h^n\mu_0.
$$

Here $a_i(n)$ is a proved phase-attraction bound, not an assertion about other
initial phases. Choose integer $T\ge0$ and $0<r\le R\le1$ such that
$a_i(T)\le r/2$. Suppose finite-horizon consistency supplies
$\mathbb E d(L_N(S_T),\mu_T)\le e_{N,T}$.
For all subsequent times $T+j$, assume the actual full kernel has a certified
conditional escape bound $u_{N,i,j}\in[0,1]$ from

$$
\mathcal G_{N,i}(R)=\{S:S\text{ is nonextinct},\ d(L_N(S),\pi_i)\le R\}.
$$

Precisely, require $P_N(S,\mathcal G_{N,i}(R)^c)\le u_{N,i,j}$
on every reachable state in $\mathcal G_{N,i}(R)$ at that time. Always include the probability of extinction by $T$ in $p_{N,T}$, even with
a cemetery empirical-law convention. Also include any initial localization
failure not already charged in $e_{N,T}$. For $k\ge0$ put

$$
B_{N,i}(k)=\min\left\{1,\ p_{N,T}+2e_{N,T}/r+
                           \sum_{j=0}^{k-1}u_{N,i,j}\right\}.
$$

Then, without contraction between any two populations,

$$
\begin{aligned}
\Pr\{S_{T+j}\notin\mathcal G_{N,i}(R)\text{ for some }0\le j\le k\}
 &\le B_{N,i}(k),\\
\mathbb E d(L_N(S_{T+k}),\pi_i)&\le R+B_{N,i}(k),\\
\mathbb E d(L_N(S_{T+k}),\mu_{T+k})
 &\le R+a_i(T+k)+B_{N,i}(k).
\end{aligned}
$$

All bounds may be truncated at one. A constant escape bound $u_{N,i}$ yields
the explicit residence horizon $ku_{N,i}\le\delta$; its physical duration is
$kh$ and its walker-update work is $kN$. If $u_{N,i}=0$, this restriction is absent.
More generally summable conditional bounds give an infinite-horizon certificate.
In a family of certificates with $R_N\to0$, $r_N\le R_N$,
$p_{N,T_N}+e_{N,T_N}/r_N+\sum_{j\ge0}u_{N,i,j}\to0$ and
$\sup_{n\ge T_N}a_i(n)\to0$, the last bound is uniform for $n\ge T_N$.
If also $\sup_{n\le T_N}\mathbb E d(L_N(S_n),\mu_n)\to0$, approximation is
uniform for all times. A finite-horizon theorem at fixed $T$ alone does not
establish that last bound at a growing $T_N$.

*Proof.* If $d(L_N(S_T),\mu_T)\le r/2$, the triangle inequality and
$a_i(T)\le r/2$ put the empirical law within $r$ of $\pi_i$. Markov's inequality
bounds failure by $2e_{N,T}/r$; add $p_{N,T}$ for the declared excluded states.
On survival in $\mathcal G_{N,i}(R)$ up to a given time, the conditional
probability of the first exit at the next step is at most $u_{N,i,j}$.
Sum these disjoint first-exit probabilities and the entry failure. No
independence between updates or walkers is used. On the resulting good event
the distance to $\pi_i$ is at most $R$, and on its complement it is at most one.
This gives the second inequality; another triangle inequality gives the third.
Taking the displayed suprema proves the uniform-time assertions. $\square$

The inputs $a_i,e_{N,T},u_{N,i,j}$ must be supplied by phase-attraction,
finite-horizon consistency and full-kernel residence estimates, respectively.
The stopping/drift bounds in Section 4 can supply residence when applied to this
population set with an appropriate exit observable. Spatial landing probabilities
alone do not identify a population phase or prove this attraction bound.
:::

:::{prf:corollary} Random phase selection without synchronization
:label: cor-slc-phase-selection

For the same full kernel and metric, let $E_1,\ldots,E_m$ be disjoint events
measurable at a deterministic selection time $T$, with $w_i=\Pr(E_i)$ and
$p=1-\sum_iw_i$. Suppose $E_i$ guarantees a nonextinct empirical law within
$R_i$ of a fixed population law $\pi_i$. Let $u_{i,j}$ bound the conditional
next-step exit probability from that phase neighborhood, as in
{prf:ref}`thm-slc-phase-residence-mf`. Choose any reference phase $\pi_0$ for
the unclassified mass. With $\mathcal W_d$ denoting the 1-Wasserstein distance
between distributions of population laws, with bounded cost $d$, one has

$$
\mathcal W_d\left(\operatorname{Law}(L_N(S_{T+k})),
 p\delta_{\pi_0}+\sum_{i=1}^m w_i\delta_{\pi_i}\right)
\le p+\sum_{i=1}^m w_i
 \left[R_i+\min\left\{1,\sum_{j<k}u_{i,j}\right\}\right].
$$

*Proof.* On $E_i$, couple the empirical law to $\pi_i$; on the unclassified
event couple it to $\pi_0$. This gives exactly the displayed target marginal.
Conditional on $E_i$, the full-state Markov property after $T$ preserves the
stated uniform escape bounds. The first-exit argument gives expected cost at
most $R_i+\min(1,\sum_{j<k}u_{i,j})$. The unclassified event has cost at most
one. Average these costs; their infimum over couplings is no larger. $\square$

The weights are actual arrival/establishment probabilities for $P_N^T$ and
can be bounded using Section 4. They may depend on initialization, $N$, landscape
and algorithm parameters. This statement concerns distributions of empirical
laws and allows random draws to select a phase. It neither identifies every
spatial well with a phase nor asserts that the displayed mixture is an exact
finite-$N$ invariant law. After significant phase exits, use the full-state
transition/committor estimates instead of retaining these initial weights.
:::

:::{prf:theorem} Optional Lipschitz error propagation and single-attractor specialization
:label: thm-slc-mf-recursion

On an invariant population class suppose a proved bound gives
$d(\mathcal F_h\mu,\mathcal F_h\nu)\le Ld(\mu,\nu)$, with finite specified $L$.
Let $e_n=\mathbb E d(L_N(S_n),\mu_n)$ and $\mu_{n+1}=\mathcal F_h(\mu_n)$.
Require both $L_N(S_n)$ and $\mu_n$ to belong to the stability class on the
good event; deterministic invariance alone does not guarantee empirical membership.
If one-step consistency, extinction and all such membership/localization failures
sum to $a_n$, then

$$
e_n\le L^ne_0+\sum_{j<n}L^{n-1-j}a_j.
$$

A block version with certified factor $q<1$ and block error at most $a_N$ gives
$e_{kr}\le q^ke_0+a_N/(1-q)$; intermediate-step bounds follow from their finite
stability factors: if each substep has factor $L$ and error at most $a$,
then at $0\le\ell<r$,
$e_{kr+\ell}\le L^\ell(q^ke_0+a_N/(1-q))+
 a\sum_{j=0}^{\ell-1}L^j$.
The actual block error can be bounded by $a_N=a\sum_{j=0}^{r-1}L^j$.
Empty sums are zero, and for $L=1$ these sums equal their number of terms.
A nonvanishing structural block defect $b_{\rm def}$ gives
floor $(a_N+b_{\rm def})/(1-q)$ instead.

*Proof.* Insert $\mathcal F_h(L_N(S_n))$ between the two output measures and apply
the triangle inequality, conditional consistency and the stability bound.
Iterate. For a block use its actual $r$-step consistency, obtained by the same
finite-horizon recursion, rather than reusing a one-step error unchanged.
$\square$
:::

:::{prf:theorem} Optional transfer of an existing two-swarm contraction
:label: thm-slc-contraction-transfer

Suppose deterministic empirical approximations to every $\mu,\nu$ in the declared
class satisfy finite-horizon consistency, and a coupling of their actual $r$-step
swarm updates obeys

$$
\mathbb E d(L_N(S_r),L_N(T_r))
\le q\,d(L_N(S_0),L_N(T_0))+b_N,\qquad b_N\to0,
$$

with $q$ independent of $N$. Then
$d(\mathcal F_h^r\mu,\mathcal F_h^r\nu)\le qd(\mu,\nu)$.
Thus a matching proved keystone–kinetic two-swarm estimate supplies the population
bound without a new dynamical assumption.

*Proof.* By consistency both empirical outputs converge in probability to their
deterministic population outputs. Their joint coupling does not affect this fact.
The bounded metric and its triangle inequality give convergence of the expected
distance. Pass to the limit in the displayed inequality. If the estimate is in
another metric, first prove its required comparison and convergence; a
single-swarm moment inequality cannot replace this coupling premise. $\square$
:::

:::{prf:proposition} Finite-horizon continuity and phase-resolved stationary limits
:label: prop-slc-phases

Assume $\mathcal F_h$ is continuous on a compact invariant population class $\mathcal K$. Define
$\omega(u)=\sup\{d(\mathcal F_h\mu,\mathcal F_h\nu):\mu,\nu\in\mathcal K,
d(\mu,\nu)\le u\}$. Continuity gives $\omega(u)\to0$. This modulus is a qualitative
continuity descriptor, not a numerically discharged Lipschitz or mixing constant.
Require empirical and deterministic inputs to belong to $\mathcal K$ on the good
event, and include all membership failures in $p_n$. For any $a>0$ and random input discrepancy with expectation
$e_n$, boundedness of $d$ gives

$$
e_{n+1}\le\varepsilon_N+\delta_N+p_n+\omega(a)+e_n/a.
$$

Consequently vanishing one-step errors and $e_0\to0$ suffice at each fixed horizon,
without contraction.

Suppose the underlying stationary or quasi-stationary particle laws are
exchangeable, their empirical laws $\Lambda_N$ are tight, and their limits satisfy
$(\mathcal F_h)_\#\Lambda=\Lambda$, as proved for the applicable canonical QSD
class in Chapter 9. If a bounded continuous functional $\mathcal V$ satisfies
$\mathcal V(\mu)-\mathcal V(\mathcal F_h\mu)=\mathcal D(\mu)\ge0$, with zero set
exactly the fixed points, then every such $\Lambda$ is supported on fixed points.
If all admissible laws instead converge to one fixed point $\mu_*$, then
$\Lambda=\delta_{\mu_*}$. Along the same subsequence $\Lambda_N\Rightarrow\Lambda$, a phase-supported
limit gives labelled marginal limit $\int\mu^{\otimes\ell}\Lambda(d\mu)$.
A unique whole-sequence phase mixture requires uniqueness of that limiting $\Lambda$.

*Proof.* Split on input distance at most $a$ and apply Markov's inequality on its
complement. Choose $a$ small and then $N$ large to obtain the finite-horizon
assertion. Invariance implies $\int\mathcal D\,d\Lambda=0$, hence support on its
zero set. For global attraction, invariance gives
$\int H(\mathcal F_h^n\mu)d\Lambda=\int H(\mu)d\Lambda$; bounded convergence for
bounded continuous $H$ identifies the point mass. Finally, exchangeability and
sampling labels with versus without replacement differ only if a repeated
label is drawn. A union bound over the $\ell(\ell-1)/2$ pairs bounds this
probability by $\ell(\ell-1)/(2N)$. Thus the labelled $\ell$-marginal differs
in TV from $\int\mu^{\otimes\ell}\Lambda_N(d\mu)$ by at most this amount.
Weak convergence of $\Lambda_N$ and continuity of $\mu\mapsto\mu^{\otimes\ell}$
give the asserted weak marginal limit. No TV convergence to a diffuse limiting
product is inferred from weak empirical convergence. $\square$
:::

:::{prf:remark} Remaining estimates are explicit, not new universal assumptions
:label: rem-slc-remaining

Phase-resolved approximation in {prf:ref}`thm-slc-phase-residence-mf` requires no
contraction between different initial populations. Its obligations are quantitative
attraction to the selected phase, entry accuracy and residence or transition
control. The optional contraction results specify another sufficient route. They do not certify
$q<1$ for every parameter choice or construct a strict population Lyapunov
functional for every multimodal landscape. Chapter 9's full-variation stability
constant controls a different input metric and can exceed one; it cannot be
inserted into an empirical weak-metric recursion without a comparison proof.
Gaussian product minorization above usually deteriorates exponentially with $N$.
It proves finite-$N$ mixing in a certified drift regime, not population-uniform
mixing by itself. Finite-horizon consistency does not interchange stationary and
population limits; use the stated tightness, concentration or attraction route.
All quantities are at fixed $h$; no $h\downarrow0$ limit is required here.
:::

(sec-slc-examples)=
## 7. Analytical landscapes and certificate boundaries

:::{div} feynman-prose
Examples tell us which parts of the theorem are doing work. A quadratic bowl tests the contraction calculation; several wells test communication; weak confinement tests the tail estimates. For each example, the useful output is a list of verified hypotheses and their constants, followed by the conclusion they actually support. This is an early form of analytical landscape tomography: the fixed gas supplies a language for describing the function's structure. A missing estimate remains a specific mathematical task rather than an inference drawn from a reassuring trajectory.
:::

:::{prf:example} Quadratic bowl and bounded oscillatory wells
:label: ex-slc-bowl-rastrigin

For $U=\kappa|x|^2/2$, $F=-\kappa x$, take $k=G=\kappa$, $b=g=e_r=e_f=0$.
Then $A_0=(1-\eta\kappa)^2$ in {prf:ref}`thm-slc-radial-drift`, below one when
$0<\eta\kappa<2$. The force defects vanish for $L\ge\kappa$.
The final full-step moment criterion is $r_Kr_C<1$ with the stated copying
certificate; the positional calculation does not assert that copying preserves
second moments. With no copying and no jitter, $r_C=1,b_C=e_C=0$ recovers the
kinetic drift. Whenever the actual cloning certificate and full observable matrix
close, {prf:ref}`thm-slc-defect-composition` and {prf:ref}`thm-slc-tv-rate` apply.

For standard Rastrigin,

$$
U(x)=|x|^2+10\sum_{j=1}^d(1-\cos(2\pi x_j)),
$$

$L_F\le2+40\pi^2$, $M_{B(0,R)}\le2R+20\pi\sqrt d$, and

$$
x\cdot F(x)\le-|x|^2+100\pi^2d,
\qquad |F(x)|^2\le8|x|^2+800\pi^2d.
$$

*Verification.* Use $20\pi|x_j|\le x_j^2+100\pi^2$ for radial drift and
$|u+v|^2\le2|u|^2+2|v|^2$ for force growth. Differentiating the force gives the
Lipschitz bound. Thus $k=1,b=100\pi^2d,G^2=8,g^2=800\pi^2d$ are explicit inputs,
and $A_0=1-2\eta+8\eta^2<1$ when $0<\eta<1/4$. Choose
$0<t<A_0^{-1}-1$ when $A_0>0$. Pairwise convexity is not invoked. For any declared
basin ball and localized source radius the landing constants are also explicit.
Local confinement and inter-well arrival have different constants; no equal
basin weights or rapid global equilibration is inferred.
:::

:::{prf:corollary} A complete finite-particle application with active cloning
:label: cor-slc-quadratic-full

Consider the conservative canonical gas, including its actual cloning, jitter and
collisions, with capped entering velocities and nonzero $q,s$. Let
$F(x)=-\kappa x+f(x)$ with $\kappa>0$, $|f(x)|\le H$, and globally Lipschitz
$F$ with constant $L_F$. Choose configured parameters satisfying

$$
\eta\kappa=1,\qquad c^2L_F<1.
$$

For $V_N=N^{-1}\sum_i|x_i|^2$, the actual full update satisfies

$$
PV_N\le b_0:=(B V_c+\eta H)^2+d(c^2q^2+s^2),
$$

independently of $N$ and of the copied/jittered positions. For each fixed $N$,
choose $R>2b_0$, let $R_0=\sqrt{NR}$, and use
{prf:ref}`thm-slc-minorization` on $\{V_N\le R\}$ with any finite $J,r,u>0$.
It gives a strictly positive $\epsilon_N$ and hence
{prf:ref}`thm-slc-tv-rate` gives a unique invariant law within the finite-moment
class and a computable geometric TV rate from every finite-moment input.

*Proof.* At this timestep the deterministic position centre after the OU/drift
stages is
$x+\eta F(x)+Bv=\eta f(x)+Bv$, of norm at most $\eta H+B V_c$ regardless of
copying or jitter. Independent kinetic noises give the claimed second moment.
On $V_N\le R$, every entering position has norm at most $\sqrt{NR}$, so every
copied source does too. Component collisions retain the uniform velocity bound.
All minorization hypotheses now hold, and the Harris theorem applies with
$r=0,b=b_0$. The $N$-independent moment bound and the $N$-dependent mixing rate
are distinct conclusions. $\square$

For the pure bowl, $H=0,L_F=\kappa$ and
$c^2\kappa=1/(1+a)<1$, so the parameter conditions are compatible. This is a
sufficient explicit family, not a restriction imposed on all later applications.
For Rastrigin, $\kappa=2$, $H=20\pi\sqrt d$ makes the moment calculation valid
at $\eta=1/2$. The original injective-map minorization does not close there;
{prf:ref}`thm-slc-separable-minorization` supplies the required noninjective
smoothing estimate, and {prf:ref}`cor-slc-rastrigin-tv` completes that application.
:::

:::{prf:example} Narrow passages and nonlocal crossings
:label: ex-slc-passages

Suppose a retained Gaussian stage has standard deviation $s$ and a mandatory
landing passage $C$ of finite volume. Its landing probability is at most
$|C|(2\pi s^2)^{-d/2}$, by the Gaussian density supremum, and a ball inside $C$
gives the lower bound in {prf:ref}`thm-slc-landing` when the centres are controlled.
If every successful crossing by update $n$ requires such a landing, a union bound
gives probability at most $nN|C|(2\pi s^2)^{-d/2}$.

The mandatory-landing premise must follow from the actual transition rule. The
canonical gas can jump over a geometric corridor. If crossings can bypass $C$,
add their probability to the upper bound; do not assign an artificial continuous
path constraint. Both passage volume and bypass probability belong in the
structural description. Capacities for local diffusions cannot replace these
full-update events without a comparison theorem.
:::

:::{prf:example} Irregular, weakly confined and outward forces
:label: ex-slc-irregular

If $F$ is Hölder on $A$ with $|F(x)-F(y)|\le H|x-y|^\alpha$, $0<\alpha<1$, then
$D_A(L,r)\le[Hr^\alpha-Lr]_+$. The BAOAB defect theorem remains valid with this
nonlinear error profile; a linear contraction rate is not implied. Outside $A$
use {prf:ref}`lem-slc-force-defect` with a force moment and an excursion bound.
A jump discontinuity can give a nonvanishing small-scale modulus. Dynamics with
finite measurable force still have the displayed update, but continuity needed
for a mean-field theorem requires a separate argument, for example negligible
mass on the discontinuity set with adequate integrability.

For $U(x)=(1+|x|^2)^{p/2}$ with $0<p<2$, $x\cdot\nabla U$ grows subquadratically,
so $b_{\mathbb R^d}(k)=\infty$ for every $k>0$. This invalidates the quadratic
radial certificate, not every possible slower or nonquadratic confinement result.
For a smooth periodic potential the gradient is bounded and the same quadratic
certificate fails. For $U=-|x|^2$, the force points outward and its radial deficit
again diverges. The failure is an explicit profile value, not a claim of a
normalizable equilibrium or a convergence theorem for these examples.
:::

:::{prf:example} Coincident and near-boundary initialization
:label: ex-slc-initialization

A coincident population has zero positional spread. Its instantaneous positional
Keystone pressure can vanish without contradicting an affine threshold estimate.
Positive normalization regularizers keep the canonical fitness defined, and
positive independent position noise produces a diffuse next-position law. This
does not by itself prove a particular establishment time.

With at least one eligible donor in a bounded valid box, the explicit alive-mass
and extinction bounds of {prf:ref}`cor-mean-field-positive-alive-mass` remain
available even near the boundary. The landing and residence results quantify
stronger claims only for their specified regions. An all-dead initial state is
absorbing in the canonical killed model; increasing $N$ alone does not revive it.
The revived source positions, actual status convention, and initial probability
of reaching a controlled class must be included in any convergence-time bound.
:::

(sec-slc-evaluated)=
## 8. Evaluated basin and stationary-limit certificates

:::{div} feynman-prose
Now put numbers into the certificates. At one noise scale and timestep, a population can remain in a Rastrigin well for a very long time. At another parameter choice, the complete finite-particle kernel admits an explicit global mixing bound. These conclusions describe different configured experiments on the same function. Each configuration has one consistent population evolution rule, whose trajectory depends on its initial law; this does not require every initial law to approach the same phase. The calculations below distinguish spatial residence, finite-particle equilibration and stationarity of the limiting population dynamics.
:::

:::{prf:lemma} The exact position Gaussian after a complete update
:label: lem-slc-position-gaussian

Use the conservative canonical kernel $P_N$, with no population force, and
condition on the input, all fitness/companion/gate variables and all component
rotations. These determine the copied source $y_i$, the copy indicator $C_i$,
and the post-collision velocity $v_i^c$, with $|v_i^c|\le V_c$. Jitters and kinetic
innovations have not yet been sampled. Define

$$
g(x)=x+\eta F(x),\qquad \tau^2=c^2q^2+s^2.
$$

The final positions have the exact representation

$$
X_i'=g(y_i+C_i\sigma_J Z_i^J)+Bv_i^c+\tau Z_i,
$$

where the pairs $(Z_i^J,Z_i)$ are independent standard Gaussian vectors across
rows and within each pair. In particular, conditional on the prepared sources,
copy indicators and velocities, the final positions are independent. The final
velocity cap does not alter this position formula. For a nonextinct input in a
terminal-killed domain, the same formula holds before classifying these positions.

*Proof.* The BAOAB identities give $x_2=x+\eta F(x)+Bv+cq\xi^O$ and
$x'=x_2+s\xi^x$. The independent Gaussian sum $cq\xi^O+s\xi^x$ has covariance
$(c^2q^2+s^2)I_d$, independently of the jitter. The last force kick and cap
change velocity only. Conditioning on all shared component variables leaves
exactly the independent row innovations stated above. $\square$
:::

:::{prf:theorem} Explicit local retention, establishment and inter-basin landing
:label: thm-slc-evaluated-basins

Let $Q_i=z_i+[-R_i,R_i]^d$ be a declared spatial core and suppose every copied
source lies in $Q_i$. Thus the hypothesis holds whenever every input position is
in $Q_i$ in the conservative gas, regardless of the cloning graph. Suppose
$F(z_i)=0$ and $F_\ell(x)=F_\ell(x_\ell)$ is separable, with

$$
-M_i\le F_\ell'(u)\le-m_i<0
\quad\text{for }|u-z_{i,\ell}|\le R_i+J_i,\qquad \ell=1,\ldots,d.
$$

All constants below depend on the declared core and cutoff, the landscape
through $m_i,M_i$, and the algorithm through $\eta,B,V_c,\sigma_J,\tau$.
Set

$$
\rho_i=\max\{|1-\eta m_i|,|1-\eta M_i|\},\quad
H_i=\rho_i(R_i+J_i)+BV_c,
$$

and suppose $\tau>0$, $J_i>0$ and $H_i<R_i$. Write $\Phi$ for the standard
one-dimensional Gaussian CDF and
$p_{J,i}=2\Phi(J_i/\sigma_J)-1$ when $\sigma_J>0$, or $p_{J,i}=1$ otherwise.
Then each output row remains in $Q_i$ with probability at least

$$
s_i=p_{J,i}^{\,d}
 \left[\Phi((R_i-H_i)/\tau)-\Phi((-R_i-H_i)/\tau)\right]^d.
$$

For a target cube $Q_j=z_j+[-R_j,R_j]^d$, possibly a different basin, put

$$
p_{ij}=p_{J,i}^{\,d}(2R_j)^d(2\pi\tau^2)^{-d/2}
 \exp\left[-\frac{\sum_{\ell=1}^d
 (|z_{j,\ell}-z_{i,\ell}|+R_j+H_i)^2}{2\tau^2}\right].
$$

Conditional on the entering swarm, its next count in $Q_i$ dominates
$\operatorname{Bin}(N,s_i)$ and its next count in $Q_j$ dominates
$\operatorname{Bin}(N,p_{ij})$. Thus, for $1\le K\le N$,

$$
\Pr\{Y_{Q_j}'\ge K\mid S\}\ge
\sum_{l=K}^N{N\choose l}p_{ij}^{l}(1-p_{ij})^{N-l},
\qquad
\Pr\{Y_{Q_j}'\ge1\mid S\}\ge1-(1-p_{ij})^N.
$$

These establishment probabilities use the actual update and include kinetic
survival in the target. From the all-in-$Q_i$ population set, the one-step exit
probability is at most $u_i=1-s_i^N$. For its exit time $\sigma_i$,

$$
\Pr(\sigma_i>n)\ge s_i^{Nn},\qquad
\Pr(\sigma_i\le n)\le\min\{1,nN(1-s_i)\}.
$$

The duration is $nh$ and the work is $nN$ walker updates. A simpler fully
explicit upper bound is

$$
1-s_i\le
2d\exp[-J_i^2/(2\sigma_J^2)]
+2d\exp[-(R_i-H_i)^2/(2\tau^2)],
$$

with the first term zero for $\sigma_J=0$. The same probabilities apply to an
absorbing configuration when both cores lie inside its valid domain and sources
satisfy the stated hypothesis.

*Proof.* The derivative of $g_\ell$ lies between $1-\eta M_i$ and
$1-\eta m_i$. Integrating from the stationary center gives
$|g_\ell(x)-z_{i,\ell}|\le\rho_i|x-z_{i,\ell}|$ within the enlarged core.
On the latent jitter cutoff in every coordinate, the pre-Gaussian mean is at
coordinate distance at most $H_i$ from $z_i$. A centered interval's Gaussian
probability decreases with the absolute value of its mean: differentiation
of $\Phi((R-m)/\tau)-\Phi((-R-m)/\tau)$ proves this for $m\ge0$.
This gives $s_i$ after multiplying coordinate probabilities and the jitter
cutoff probability. For $Q_j$, the Gaussian density throughout its cube is at
least the displayed infimum; integrate its volume and the jitter event.

Conditional on the graph, rotations and input, independent row jitters and
kinetic innovations give independent output memberships with these same lower
bounds. Independent uniforms couple them above the stated binomials. Averaging
over the shared variables preserves each domination. In particular all $N$
rows remain in the core with probability at least $s_i^N$. Iteration up to
first exit, without assuming independence between updates, proves residence.
A coordinate union bound on the jitter and kinetic tails gives the last formula.
$\square$
:::

:::{prf:corollary} Exact quadratic improvement without a jitter cutoff
:label: cor-slc-quadratic-retention

For $F(x)=-\kappa(x-z)$, write $b=1-\eta\kappa$,
$H=|b|R+BV_c$ and $\tau_*^2=\tau^2+b^2\sigma_J^2$. If $H<R$, the retention
constant for $z+[-R,R]^d$ can be replaced by

$$
s_{\rm quad}=\left[
\Phi((R-H)/\tau_*)-\Phi((-R-H)/\tau_*)\right]^d.
$$

*Proof.* For a fixed copy indicator, the jitter and position innovations combine
into variance $\tau^2+C_i b^2\sigma_J^2$ per coordinate. The mean is bounded
by $H$. When the mean lies inside the target interval, its probability decreases
with increasing variance: write it as
$\Phi((R-m)/\sigma)+\Phi((R+m)/\sigma)-1$, with both numerators positive,
and differentiate. The largest variance is $\tau_*^2$; the preceding theorem's
conditional independence and residence proof apply unchanged. $\square$
:::

:::{prf:example} A certified Rastrigin residence calculation in several wells
:label: ex-slc-rastrigin-residence

For the standard Rastrigin potential, $F_\ell(x)=-2x_\ell-20\pi\sin(2\pi x_\ell)$.
There are stationary points $z_0=0$ and $z_1\in(99/100,1)$, and by symmetry
$z_{-1}=-z_1$. Each is a strict local minimum. Choose any center
$z\in\{z_{-1},z_0,z_1\}^d$ and

$$
\begin{gathered}
R=1/16,\quad J=1/64,\quad
m=2+20\sqrt2\pi^2,\quad M=2+40\pi^2,\\
\eta=2/(m+M),\quad a=1/2,\quad
h=2\sqrt{\eta/(1+a)},\quad \gamma=(\log2)/h,\\
V_{\max}=10^{-3},\quad \alpha_{\rm col}=1/2,\quad
\sigma_J=q=s=10^{-3}.
\end{gathered}
$$

The configured thermostat and position amplitudes are, explicitly,
$b_O=q\sqrt{2\gamma/(1-a^2)}$ and $\sigma_x=s/\sqrt h$.
All companion, fitness and acceptance parameters may take any admissible values;
the bound holds for every resulting cloning graph. Here $V_c=2\cdot10^{-3}$.
For any swarm initially entirely in this core,

$$
\boxed{\Pr(\sigma\le n)\le\min\{1,4nNd\,e^{-122}\}.}
$$

For $d=1$, $N=128$, and $n=10^6$, the failure bound is strictly below
$6\cdot10^{-45}$. These are $10^6h$ physical time units and $128\cdot10^6$
walker updates, not free exploration. The same certificate applies separately
to the distinct cores; it does not require swarms initialized in them to
approach one another over this horizon.

*Proof.* On $|x-k|\le1/8$, $U''(x)=2+40\pi^2\cos(2\pi x)$ lies in $[m,M]$.
At $x=99/100$,
$U'(x)=99/50-20\pi\sin(\pi/50)<0$, since
$\sin t\ge2t/\pi$ for $0\le t\le\pi/2$ and $\pi>3$.
At $x=1$ the derivative is $2>0$; its strict increase on $[7/8,9/8]$
gives a unique root in the stated interval. The enlarged core around that root
lies in this interval because $1/100+R+J<1/8$. The zero and negative cores
satisfy the same estimates.

The chosen $\eta$ gives
$\rho=(M-m)/(M+m)<3-2\sqrt2<1/5$.
Using $\sqrt2>7/5$ and $\pi>3$ gives $m+M>616$, hence $h<1/10$.
Therefore $B=3h/4<3/40$, and

$$
R-H>1/16-(1/5)(5/64)-(3/40)(2/1000)>9/200.
$$

Also $\tau^2=(1+c^2)10^{-6}<401/(400\cdot10^6)$, so
$(R-H)^2/(2\tau^2)>1000$. The jitter exponent is
$J^2/(2\sigma_J^2)=10^6/8192>122$. Substitute in the coordinate-tail bound
and use $e^{-1000}<e^{-122}$. The numerical bound is certified without
floating-point tail evaluation by the rational inequality
$\sum_{k=0}^{200}122^k/k!>4\cdot10^6\cdot128/(6\cdot10^{-45})$;
this Taylor sum is a strict lower bound on $e^{122}$.
$\square$

Here $h\approx0.0886960$ and $\rho\approx0.170561$ are descriptive rounded
values; the proof uses the exact expressions. Inter-core discovery and establishment
are explicitly given by $p_{ij}$ and its binomial tail above, using
$|z_1|<1$ for lower bounds and $|z_1|>99/100$ for separation bounds.
The tiny residence failure probability entails correspondingly slow certified
inter-core communication at this noise scale. Residence is not a proof that a
core is absorbing or that its finite-particle conditional law is stationary.
:::

:::{prf:theorem} Evaluated attraction inside a basin to a positional noise floor
:label: thm-slc-local-attraction

Use the Rastrigin force and a core $Q_i=z_i+[-R_i,R_i]^d$ from
{prf:ref}`thm-slc-evaluated-basins`. Assume $S_0$ lies in the all-in-$Q_i$
population set almost surely. Let $\sigma_i$ be the first completed-state
exit of the swarm from the all-in-$Q_i$ population set, and
$V_i(S)=N^{-1}\sum_j|x_j-z_i|^2$. For the squashed feature metric put

$$
\kappa_i=\exp[-(4dR_i^2+4\lambda_{\rm alg}V_{\max}^2)/(2\epsilon_C^2)],
\qquad r_{C,i}=1+\kappa_i^{-1}.
$$

For $N=1$ use $r_{C,i}=1$. Define $l=|1-2\eta|$, $H=20\pi\sqrt d$ and

$$
\begin{gathered}
A_i=l\sqrt d R_i+2\eta H+BV_c,\quad D_i=l\sigma_J,\quad
p_i^J=\min\{1,2d e^{-J_i^2/(2\sigma_J^2)}\},\\
E_i=\sqrt{8[A_i^4+D_i^4d(d+2)]}\sqrt{p_i^J},\\
r_i=(1+t)\rho_i^2 r_{C,i},\\
b_i=(1+t)\rho_i^2d\sigma_J^2+(1+1/t)B^2V_c^2+d\tau^2+E_i,
\qquad t>0.
\end{gathered}
$$

When $\sigma_J=0$, set $p_i^J=E_i=0$. On this population set the actual complete
kernel satisfies $P_NV_i\le r_iV_i+b_i$. Consequently, when $r_i<1$,

$$
\mathbb E[V_i(S_n)\mathbf1_{\{\sigma_i>n\}}]
\le r_i^n\mathbb EV_i(S_0)+b_i\frac{1-r_i^n}{1-r_i}.
$$

If $nu_i<1$, division by $1-nu_i$ gives an upper bound on the conditional mean
given $\sigma_i>n$, with $u_i$ from the residence theorem. For a target mean-square
radius $v_*>0$, Markov's inequality bounds
$\Pr\{V_i(S_n)>v_*,\sigma_i>n\}$ by the displayed right side divided by $v_*$.
Thus this is a quantitative basin-attraction estimate with an explicit noise
floor and an explicit exit charge. It is a one-swarm estimate, not contraction
between populations in different wells.

*Proof.* The squash maps are 1-Lipschitz. Inside the core, positional distances
are at most $2\sqrt dR_i$ and velocity distances at most $2V_{\max}$.
Thus the actual cloning weights are at least $\kappa_i$. Applying the copying
moment proof about center $z_i$ gives factor $r_{C,i}$.
On the coordinatewise jitter cutoff,
$|g(x)-z_i|\le\rho_i|x-z_i|$. Young's inequality bounds the squared deterministic
position mean by
$(1+t)\rho_i^2|x-z_i|^2+(1+1/t)B^2V_c^2$.
The unrestricted expectation of $|x-z_i|^2$ is the copied-source moment plus
at most $d\sigma_J^2$, since gates precede centered independent jitter.

For a failed jitter cutoff, $F(z_i)=0$ and the global sine bound give
$|g(x)-z_i|\le l|x-z_i|+2\eta H$.
Thus the deterministic mean's norm is at most $A_i+D_i|Z^J|$.
Its fourth moment is at most $8[A_i^4+D_i^4d(d+2)]$; Cauchy–Schwarz charges
at most $E_i$ on the bad event. Adding this bad-event contribution to the
unrestricted upper bound for the good event only increases the estimate.
Finally the independent centered position Gaussian adds exactly $d\tau^2$.
This proves the full-kernel drift before stopping. For the stopped expectation,
condition on $\{\sigma_i>n\}$ and discard the nonnegative contributions from
exits at the next step, obtaining $A_{n+1}\le r_iA_n+b_i$.
Iteration, the residence lower bound and Markov's inequality prove the claims.
$\square$
:::

:::{prf:example} An explicit within-well relaxation rate
:label: ex-slc-rastrigin-attraction

Use the one-dimensional low-noise Rastrigin parameters above, set
$\lambda_{\rm alg}=1$, $\epsilon_C=2$ and $t=1$. The other fitness and measurement
parameters are arbitrary admissible values. Then

$$
\boxed{r_i<6/25,\qquad b_i<113/10^8.}
$$

For $N=128$, any all-in-core initial population has
$V_i(S_0)\le1/256$. After eight updates,

$$
\mathbb E[V_i(S_8)\mid\sigma_i>8]<1.6\cdot10^{-6},
\qquad \Pr(\sigma_i\le8)<10^{-40}.
$$

These estimates hold for each of the three cores independently. They measure
relaxation toward a small positional neighborhood with nonzero thermal width;
they do not identify the full stationary velocity/position law with a point mass.

*Verification.* The exponent defining $\kappa_i$ is less than $1/2$, so
$\kappa_i\ge1-x>1/2$ by $e^{-x}\ge1-x$; hence $r_{C,i}<3$.
Together with $\rho_i<1/5$, this gives $r_i<6/25$.
The non-excursion terms in $b_i$ are at most $451/(400\cdot10^6)$.
Using $\eta<1/308$, $\pi<22/7$, $l<1$ gives $A_i<1/2$ and $D_i<10^{-3}$,
so the square-root prefactor of $E_i$ is less than one.
Thus $E_i<2e^{-61}<1/(400\cdot10^6)$, giving $b_i<113/10^8$.
The last exponential comparison follows already from the eighth positive Taylor
term $e^{61}>61^8/8!>8\cdot10^8$.
Substitution gives
$(6/25)^8/256+(113/10^8)/(1-6/25)<1.54\cdot10^{-6}$.
The residence bound has failure below $10^{-40}$; division by its survival
probability leaves the asserted $1.6\cdot10^{-6}$ bound. All numerical
inequalities here are rational or follow from the specified Taylor bounds.
$\square$
:::

:::{prf:proposition} Evaluated selection gain for a newly discovered site
:label: prop-slc-single-discoverer

Consider a conservative entering swarm with $N\ge3$, $N-1$ identical rows at
$(x_A,0)$ and one at $(x_B,0)$, $x_A\ne x_B$. Let $D$ be their configured feature
distance and $w_D=e^{-D^2/(2\epsilon_D^2)}$,
$w_C=e^{-D^2/(2\epsilon_C^2)}$. Take the admissible diversity-only fitness
configuration $p_r=0$, $p_s>0$. Put

$$
\begin{gathered}
\Delta=\sqrt{D^2+\delta_D^2}-\delta_D,\quad
S=\sqrt{\Delta^2/4+\sigma_s^2},\quad Z=\Delta/\sigma_s,\\
m_f=p_s\min\{\eta_s^{p_s-1},(A_s+\eta_s)^{p_s-1}\}
          A_se^{-Z}/(1+e^{-Z})^2,\\
a_*=\min\{1,m_f\Delta/[S s_c((A_s+\eta_s)^{p_s}+\epsilon_c)]\}.
\end{gathered}
$$

Let $K_B^c$ count rows at $x_B$ immediately after copying, before jitter. Then

$$
\mathbb E[K_B^c\mid S]\ge
1+(N-1)\frac{w_C}{N-2+w_C}\frac{N-2}{N-2+w_D}a_*.
$$

There is also an exact finite sum, avoiding the conservative derivative bound.
Put $p_D=w_D/(N-2+w_D)$ and $p_C=w_C/(N-2+w_C)$. For $0\le k\le N-1$ define

$$
\begin{gathered}
t_k=(k+1)/N,\quad s_k=\sqrt{t_k(1-t_k)\Delta^2+\sigma_s^2},\\
f_k^-=\left[\frac{A_s}{1+e^{t_k\Delta/s_k}}+\eta_s\right]^{p_s},\quad
f_k^+=\left[\frac{A_s}{1+e^{-(1-t_k)\Delta/s_k}}+\eta_s\right]^{p_s},\\
g_k=\min\{1,(f_k^+-f_k^-)/[s_c(f_k^-+\epsilon_c)]\}.
\end{gathered}
$$

Then

$$
\mathbb E K_B^c=1+p_C\sum_{k=0}^{N-1}{N-1\choose k}
 p_D^k(1-p_D)^{N-1-k}(N-1-k)g_k.
$$

There is no reverse-copy loss from the discovered row in this configuration.
If $x_B$ belongs to a core with a uniform copied-source kinetic survival bound
$s_B$ from {prf:ref}`thm-slc-evaluated-basins`, its complete-update count satisfies
$\mathbb E Y_{Q_B}'\ge s_B\mathbb E K_B^c$.

*Proof.* The discovered row must measure a resident, so it always has the far
separation. A resident measures another resident with probability
$(N-2)/(N-2+w_D)$. On that event, the standardized diversity gap is at least
$\Delta/S$, since the common empirical variance is at most $\Delta^2/4$.
Both standardized values lie in $[-Z,Z]$, where the derivative of the diversity
factor is at least $m_f$. The resident independently chooses the discovered donor
with probability $w_C/(N-2+w_C)$, and the gate is at least $a_*$.
Sum over residents. For the exact sum, the number $k$ of resident far
measurements is binomial with parameters $N-1,p_D$; together with the discoverer
it gives exactly $k+1$ far measurements and the displayed standardization.
Each of the remaining $N-1-k$ residents copies it with probability $p_Cg_k$.
No resident's diversity exceeds the discovered row's, so
its own acceptance of a resident donor is zero. Collision velocities remain zero
because all entering velocities are zero. Finally condition on each copied source
and apply its jitter/kinetic survival bound; linearity needs no independence
between recipients. $\square$

This is an evaluated expected establishment gain for the actual sampled-fitness
rule, not a high-probability establishment theorem. With nonzero reward exponent,
the four near/far mark combinations must retain the reward factors and reverse
acceptances; a reward disadvantage cannot be discarded. The displayed lower bound on the additional count approaches
$w_Ca_*$ walkers per update as $N\to\infty$, so population size does
not make the first discoverer's amplification arbitrarily fast for free.
:::

:::{prf:proposition} Reward-aware establishment gain, including reverse transfer
:label: prop-slc-reward-aware-discovery

Retain the two-site entering configuration and the $p_D,p_C,t_k,s_k,f_k^\pm$
of {prf:ref}`prop-slc-single-discoverer`, now allowing $p_r\ge0$.
For the actual reward values $r_A,r_B$ put

$$
\bar r=((N-1)r_A+r_B)/N,\quad
S_r=\sqrt{(N-1)(r_B-r_A)^2/N^2+\sigma_r^2},\quad
H_j=\left[\frac{A_r}{1+e^{-(r_j-\bar r)/S_r}}+\eta_r\right]^{p_r}.
$$

Write $a(f,g)=\min\{1,(g-f)_+/[s_c(f+\epsilon_c)]\}$ and $F_{B,k}=H_Bf_k^+$.
Define the completely explicit signed gain

$$
\begin{aligned}
G_k={}&(N-1-k)\left[p_Ca(H_Af_k^-,F_{B,k})
             -\frac{a(F_{B,k},H_Af_k^-)}{N-1}\right]\\
&+k\left[p_Ca(H_Af_k^+,F_{B,k})
             -\frac{a(F_{B,k},H_Af_k^+)}{N-1}\right].
\end{aligned}
$$

Then the actual expected pre-jitter count is exactly

$$
\mathbb E K_B^c=1+\sum_{k=0}^{N-1}{N-1\choose k}
 p_D^k(1-p_D)^{N-1-k}G_k.
$$

*Proof.* Conditional on the $k$ resident far measurements, the resident fitnesses
have the two displayed values and the discoverer has $F_{B,k}$. Each resident
selects the discoverer with probability $p_C$. The discoverer selects uniformly
among the $N-1$ residents, because their positions and velocities coincide and
all its donor weights are equal. Therefore the two negative terms are exactly
its probabilities of reverse transfer to each measured resident group. Sum the
changes of the position indicators and average the binomial measurement count.
No replacement of sampled fitness by averaged fitness occurs. $\square$

For a fully specified evaluation take $N=128$, $d=1$, $x_A,x_B\in\{0,1\}$,
common velocity zero, positional feature radius $2$, $\epsilon_D=\epsilon_C=2$,
$\delta_D=10^{-3}$, $\sigma_r=\sigma_s=1/10$, $A_r=A_s=2$,
$\eta_r=\eta_s=1/10$, $p_s=1$, $s_c=1$, $\epsilon_c=10^{-6}$.
Use reward $r(x)=-U_{\rm Rastrigin}(x)$, so $r(0)=0$ and $r(1)=-1$.
The following are rigorous enclosures of the finite sum:

| Resident site $x_A$ | Discovered site $x_B$ | Reward exponent $p_r$ | Expected count $\mathbb E K_B^c$ |
|---|---|---|---|
| $0$ | $1$ | $0$ | $1.9060<\mathbb E K_B^c<1.9061$ |
| $1$ | $0$ | $1$ | $1.9460<\mathbb E K_B^c<1.9462$ |
| $0$ | $1$ | $1$ | $0\le\mathbb E K_B^c<10^{-4}$ |

Thus discovery does not imply amplification irrespective of reward. The actual
sampled-fitness rule amplifies the better discovered site in this evaluation,
while the reverse direction is strongly suppressed. Under the low-noise kinetic
parameters above, copied sources at either site have core survival at least
$1-4e^{-122}$, giving the corresponding rigorous lower complete-update count
bounds by multiplication. This does not turn an expected count into a success
probability or iterate a two-atom closure after kinetics has spread the law.

The enclosures are checked by the executable rational-interval certificate
specified below: endpoints are rounded outward to dyadic rationals, square roots
are bracketed by integer square roots, and exponentials by 100 Taylor terms
plus a geometric upper bound on the positive remainder. Only rational/integer
inequalities are used to validate the displayed decimal bounds.
:::

:::{prf:theorem} Separable nonconvex smoothing without global force contraction
:label: thm-slc-separable-minorization

Consider the conservative full kernel with $q,s>0$, the actual cap, and
$F_\ell(x_\ell)=-\kappa x_\ell+f_\ell(x_\ell)$, with $\kappa>0$,
$H_\ell,L_\ell\ge0$, where $f_\ell$ is $C^1$,
$|f_\ell|\le H_\ell$ and $|F_\ell'|\le L_\ell$ on $\mathbb R$.
Suppose $\lambda_0=1-c^2\kappa>0$; no condition $c^2L_\ell<1$ is required.
On the input set require every copied source coordinate to satisfy
$|y_{i,\ell}|\le R_{0,\ell}$. For $J,r,u>0$ define

$$
\begin{aligned}
R_{x,\ell}&=R_{0,\ell}+J,& F_{x,\ell}&=\kappa R_{x,\ell}+H_\ell,\\
R_{1,\ell}&=R_{x,\ell}+c(V_c+cF_{x,\ell}),&
m_\ell&=a(V_c+cF_{x,\ell}),\\
Q_\ell&=(u+c\kappa R_{1,\ell}+cH_\ell)/\lambda_0,\\
k_{v,\ell}&=(2\pi q^2)^{-1/2}
 e^{-(Q_\ell+m_\ell)^2/(2q^2)}/(1+c^2L_\ell),\\
k_{x,\ell}&=(2\pi s^2)^{-1/2}
 e^{-(r+R_{1,\ell}+cQ_\ell)^2/(2s^2)},\\
\epsilon_N&=\left[p_J^d(2u)^d(2r)^d
                    \prod_{\ell=1}^d k_{v,\ell}k_{x,\ell}\right]^N,
\quad p_J=2\Phi(J/\sigma_J)-1.
\end{aligned}
$$

Set $p_J=1$ for zero jitter. Let $\nu$ be the product over rows of uniform
positions in $[-r,r]^d$ and independently capped uniform velocities from
$[-u,u]^d$. Then $P_N(S,\cdot)\ge\epsilon_N\nu$ on this input set.

*Proof.* Condition on the graph, rotations and the coordinatewise jitter cutoffs,
whose joint probability is $p_J^{dN}$. All source and collision bounds are now
deterministic. In coordinate $\ell$ the final pre-cap velocity is

$$
T_\ell(v)=v+cF_\ell(x_{1,\ell}+cv)
=\lambda_0v-c\kappa x_{1,\ell}+cf_\ell(x_{1,\ell}+cv).
$$

Its bounded perturbation of $\lambda_0v$ tends to opposite infinities at the two
ends of the line. The intermediate value theorem makes it onto. Every preimage
of $|z|\le u$ has $|v|\le Q_\ell$, and
$|T_\ell'(v)|\le1+c^2L_\ell$. For almost every such $z$, the one-dimensional
change-of-variables formula sums the Gaussian density over its inverse branches;
at least one branch contributes, and each regular contribution is bounded below
by $k_{v,\ell}$.

For completeness, critical values form a null set here. On a compact interval,
the critical set $\{T_\ell'=0\}$ can be enclosed in an open set where
$|T_\ell'|<\varepsilon$; its component intervals have bounded total length, and
the sum of their image lengths is at most that length times $\varepsilon$.
Let $\varepsilon\downarrow0$ and exhaust the line by compact intervals.
For a noncritical target, all its preimages in the compact bound are isolated
and hence finite; inverse-function neighborhoods and the ordinary substitution
formula give the asserted sum. Any singular output mass only improves the
lower bound. Taking coordinate products yields the velocity-cube density bound.

For every possible inverse branch, the pre-final position coordinate is bounded
by $R_{1,\ell}+cQ_\ell$, so independent final position noise has density at least
$k_{x,\ell}$ on $[-r,r]$. This bound holds branch by branch; injectivity is not
being assumed. Push forward the pre-cap velocity cube through the cap, multiply
row bounds conditional on the prepared state, and integrate over that state.
$\square$
:::

:::{prf:corollary} Fully evaluated finite-particle mixing for Rastrigin
:label: cor-slc-rastrigin-tv

For standard Rastrigin take $\kappa=2$, $H_\ell=20\pi$ and
$L_\ell=2+40\pi^2$ in the preceding theorem. Choose any finite $\gamma\ge0$
and $h>0$ satisfying $\eta\kappa=1$, with $q,s>0$; equivalently one can choose
$a\in(0,1]$, set $h=2/\sqrt{\kappa(1+a)}$ and
$\gamma=-\log(a)/h$. Other admissible algorithm parameters are arbitrary.
With $H=(\sum_\ell H_\ell^2)^{1/2}$ put

$$
b_0=(BV_c+\eta H)^2+d\tau^2,\quad R=2b_0,\quad
R_{0,\ell}=\sqrt{NR}.
$$

Choose and declare $J,r,u>0$, compute $\epsilon_N$ above, and set
$\delta_N=\epsilon_N/2$. Then the complete kernel has a unique invariant
probability $\pi_N$, and for every initial probability $\zeta$ on capped states,

$$
\boxed{\|\zeta P_N^n-\pi_N\|_{\rm TV}
\le(1-\delta_N)^{\lfloor n/2\rfloor}.}
$$

For $0<\varepsilon<1$, it suffices to take
$2\lceil\log(1/\varepsilon)/[-\log(1-\delta_N)]\rceil$ iterations.
All constants depend explicitly on $d,N,h,\gamma,b_O,\sigma_x,\sigma_J,
V_{\max},\alpha_{\rm col},J,r,u$ and the displayed force bounds. Independence
from the fitness and companion parameters follows because every possible
cloning graph obeys the same bounds. This is a nonconvex finite-particle
certificate, not a population-uniform synchronization theorem.

*Proof.* The resonance identity gives
$X_i'=\eta f(y_i+C_i\sigma_JZ_i^J)+Bv_i^c+\tau Z_i$.
Thus $P_NV_N\le b_0$ for $V_N=N^{-1}\sum_i|x_i|^2$, at every input.
Markov's inequality gives $P_N(S,\{V_N\le2b_0\})\ge1/2$.
On that set every source coordinate is at most $\sqrt{2Nb_0}$, while
$\lambda_0=1-c^2\kappa=a/(1+a)>0$. Hence the preceding theorem applies and
$P_N^2(S,\cdot)\ge\delta_N\nu$ for every $S$.
Write this kernel as $\delta_N\nu+(1-\delta_N)R_N(S,\cdot)$.
Markov kernels contract TV, so its contraction factor is $1-\delta_N$ on
probabilities. Iterates are Cauchy in the complete TV space by summing their
geometric successive differences. The limit is the unique invariant law of
$P_N^2$. Applying $P_N$ to it gives another $P_N^2$-invariant law and hence the
same law. TV contraction of the remaining single step gives the displayed bound.
The logarithmic iteration count follows directly. $\square$

This parameter regime differs from the low-noise, small-step residence example.
The two calculations must not be combined as though they concerned one parameter
choice. Both keep the original algorithm; neither imposes a convex landscape.
:::

:::{prf:theorem} Stationary population limits of the evaluated multimodal gas
:label: thm-slc-evaluated-stationary-mf

In the preceding resonance regime, let the reward be continuous of at most
quadratic growth (in particular, minus the Rastrigin potential). Define

$$
L_0=BV_c+\eta H,\qquad
M_6=32\left[L_0^6+\tau^6d(d+2)(d+4)\right].
$$

The actual fixed-step population map has at least one stationary law on the
all-alive capped state space, with sixth position moment at most $M_6$.
For every bounded test $\Psi$ on population laws with
$|\Psi(\mu)-\Psi(\nu)|\le d(\mu,\nu)$, the finite-particle stationary law obeys

$$
\left|\int[\Psi(\mathcal F_h\mu)-\Psi(\mu)]\,\Lambda_N(d\mu)\right|
\le\varepsilon_N=\frac{\sqrt{A+4B_*^2}}{2\sqrt N},
\qquad \Lambda_N=(L_N)_\#\pi_N,
$$

with the fully specified constants above and $m_*=1$. More generally the
one-step distributional consistency bound
$\mathcal W_d(\operatorname{Law}(L_N(S_{n+1})),
(\mathcal F_h)_\#\operatorname{Law}(L_N(S_n)))\le\varepsilon_N$
holds at every $n$; it compares the same evolution rule at the current input,
not its indefinitely iterated deterministic trajectory from a fixed initial law.
The stationary empirical-law distributions $\Lambda_N$ are tight in the topology
of $W_4$ on population laws. Every subsequential limit obeys

$$
(\mathcal F_h)_\#\Lambda=\Lambda.
$$

This is stationarity under the already derived nonlinear mean-field law.
It neither assumes a unique attracting phase nor identifies an invariant
population distribution with a mixture of fixed points without the additional
phase-identification argument of {prf:ref}`prop-slc-phases`.

*Proof.* The exact Gaussian position formula and
$(u+v)^6\le32(u^6+v^6)$ give the output moment bound, since
$\mathbb E|Z|^6=d(d+2)(d+4)$. This is uniform over inputs and $N$ and thus
also holds under $\pi_N$ by integration, without requiring its moment in advance.
The set $\mathcal K=\{\mu:\int|x|^6d\mu\le M_6,\ |v|\le V_{\max},\ a=1\}$
is nonempty, convex and weakly compact; sixth moments give tightness and are
lower semicontinuous. They also give uniform fourth-moment tails,
$\int_{|x|>R}|x|^4d\mu\le M_6/R^2$, so weak convergence on $\mathcal K$
implies $W_4$ convergence. The moment bound gives
$\mathcal F_h(\mathcal K)\subseteq\mathcal K$.

The map is continuous on this class: reward first and second moments converge
by quadratic growth and the displayed fourth-moment uniform integrability;
companion denominators are bounded below by $\kappa_D,\kappa_C$; the finite
rooted-component approximation is continuous, with a uniform component-size
tail; and the Lipschitz, linear-growth kinetic stages preserve the convergence.
These are precisely the steps of {prf:ref}`lem-mean-field-map-continuity`.
Its weak continuity on the compact convex set gives a fixed point by the
compact-convex fixed-point theorem.

For any input distribution, couple $L_N'$ with $\mathcal F_h(L_N)$ using
the same entering swarm. The one-step empirical estimate bounds this expected
cost by $\varepsilon_N$, which proves the distributional consistency bound.
Under stationarity, $L_N'$ and $L_N$ have the same law, so integrating the
Lipschitz test difference proves the stated stationary residual estimate.

For the stationary particle laws,
$\mathbb E_{\Lambda_N}\int|x|^6d\mu\le M_6$.
Markov's inequality places probability at least $1-M_6/A$ in the $W_4$-compact
set of laws with sixth moment at most $A$. Hence $\Lambda_N$ is tight there.
For a stationary one-step pair $(L_N,L_N')$, its joint laws are tight by the
same marginal bound. The all-alive one-step estimate
{prf:ref}`lem-slc-empirical-error` gives
$d(L_N',\mathcal F_h(L_N))\to0$ in probability, with its explicit
$\sqrt{A+4B_*^2}/(2\sqrt N)$ bound and $m_*=1$.
Continuity of $\mathcal F_h$ under $W_4$ follows by the preceding truncation
argument. Every joint limit $(X,Y)$ therefore has $Y=\mathcal F_h(X)$ almost
surely; the bounded weak metric $d$ separates laws. Stationarity gives identical
marginal distributions $\Lambda$ for $X,Y$, proving invariance. $\square$
:::

(slct-quantitative-trajectories)=
## 9. A constructive weak-metric trajectory estimate

:::{div} feynman-prose
Imagine preparing a large swarm with a specified initial distribution. The
mean-field equation predicts how that distribution evolves. Our task is to
bound the discrepancy between this prediction and the empirical distribution
of the finite swarm, update by update. Preparing another swarm in another well
changes the initial distribution and therefore changes the predicted trajectory.
Both trajectories obey the same evolution law.

Two errors enter the calculation. A finite population introduces sampling error
at each update, and the population map can amplify an error already present.
The estimates below calculate both effects from the force, reward,
regularization and interaction parameters. The comparison uses a transport
metric that permits discrete empirical distributions to approach continuous
laws. Moment estimates pay for excursions into the unbounded part of space.
Even when the resulting error grows with the observation horizon, it can still
vanish as the population grows for every fixed horizon. That is a quantitative
mean-field limit. Attraction to a stationary law is an additional question.
:::

:::{prf:definition} Metric and explicit landscape regime
:label: def-slct-regime

In this section $D=\mathbb R^d$, every row is alive, all stored velocities
satisfy $|v|\le V_{\max}$, and the canonical current-step algorithm has no
viscosity or history term. The force obeys
$|F(x)-F(y)|\le L_F|x-y|$. The reward obeys
$|R(z)|\le R_b$ and $|R(z)-R(z')|\le L_R|z-z'|$.
These conditions permit multiple wells and impose no convexity. Let
$z=(x,v)$ and
\[
 c_0(z,z')=\min\{1,|z-z'|\},\qquad
 \mathsf d(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\int c_0\,d\pi.
\]
All coordinates and the cutoff 1 refer to the declared units of the state
space. Unlike total variation, this metric allows atomic empirical laws to
approach continuous laws. The parameters $c,a,B,\eta,q,s,V_c$ have the
values in the parameter register. Put
\[
 \ell_f=\max\{1,\sqrt\lambda\},\quad
 D_*=2\sqrt{R_x^2+\lambda R_v^2},\quad
 S_*=\sqrt{D_*^2+\delta_D^2},\quad
 \kappa_b=e^{-D_*^2/(2\epsilon_b^2)},\quad
 w_b'=D_*\ell_f/\epsilon_b^2\quad(b=D,C).
\]
Here $R_x=R_x^{\rm feat}$, $R_v=R_v^{\rm feat}$ and
$\lambda=\lambda_{\rm alg}$ are the configured feature parameters,
not spatial cutoffs.
The squashing maps are 1-Lipschitz, so $w_b'$ bounds the derivative of the
weight in either physical argument and $\ell_f$ bounds the separation
increment in either argument.
:::

:::{prf:theorem} Explicit continuity of the full population map
:label: thm-slct-modulus

In {prf:ref}`def-slct-regime`, define
\[
 b_0=1+(3+4w_D')/\kappa_D,\qquad
 m_r=L_R+2R_b,\qquad m_s=2\ell_f+S_*b_0,
\]
\[
 Q_r={L_R+m_r\over\sigma_r}
       +{4R_b^2m_r\over\sigma_r^3},\qquad
 Q_s={2\ell_f+m_s\over\sigma_s}
       +{2S_*^2m_s\over\sigma_s^3}.
\]
For $b=r,s$, let
\[
 H_b={A_b\over4}p_b
 \max\{\eta_b^{p_b-1},(A_b+\eta_b)^{p_b-1}\}
 (A_{b'}+\eta_{b'})^{p_{b'}},\quad b'\ne b,
\]
with $H_b=0$ when $p_b=0$. Set
\[
 E_F=H_rQ_r+H_sQ_s,\quad
 F_*=\eta_r^{p_r}\eta_s^{p_s},\quad
 F^*=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},
\]
\[
 L_a=\max\left\{{1\over s_c(F_*+\epsilon_c)},
 {F^*+\epsilon_c\over s_c(F_*+\epsilon_c)^2}\right\},
\]
\[
 D_\beta={2w_C'\over\kappa_C}
 +{2w_C'+1\over\kappa_C^2}
 +{2L_aE_F\over\kappa_C},\qquad C=2/\kappa_C,
\]
\[
 k_v=1+2|\alpha_{\rm col}|,\quad
 U=1+\eta L_F+Bk_v,\quad
 V=acL_F+ak_v+cL_FU,
\]
\[
 C_{\rm mod}=b_0+8(D_\beta+2b_0/\kappa_C)
                   +2e^{2C}+U+V.
\]
Then, for the actual rooted collision map followed by the actual kinetic
update,
\[
 \boxed{\mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
 \le\min\{1,C_{\rm mod}\mathsf d(\mu,\nu)^{1/4}\}.}
\]
Noise amplitudes and clone jitter do not enter this continuity constant:
the proof uses identical additive noises. Their dependence enters the
moment and sampling constants below. No stationary law or attraction
assumption is used.
:::

:::{prf:proof}
Write $\delta=\mathsf d(\mu,\nu)$. The case $\delta=0$ follows by identical
inputs. For $0<\delta\le1$ choose an optimal coupling $\pi$ and put
$r=\sqrt\delta$. The event $|z-z'|>r$ has probability at most
$\delta/r=\sqrt\delta$.

**Measurement coupling.** Conditional on a good recipient pair, lift both
weighted companion laws to the same base coupling $\pi(dy,dy')$. On good
donor pairs their weight difference is at most $2w_D'r$; on bad pairs it
is at most one. Normalizing masses bounded below by $\kappa_D$ gives a
coupling failure probability at most
$2(2w_D'r+\delta/r)/\kappa_D$. The successfully matched companion can be
bad with probability at most $\delta/(\kappa_Dr)$, because its matched
subdensity is at most $1/\kappa_D$ relative to $\pi$. Together with the
recipient failure this constructs a coupling $\Lambda$ of the two complete
measurement laws whose bad mass is at most $b_0\sqrt\delta$.
On its complement both physical inputs and measurement companions are
within $r$.

**Normalization.** Under $\pi$, the expected reward difference is at most
$m_r\sqrt\delta$. For a variable bounded in absolute value by $R_b$, the
variance difference is at most $4R_b$ times its expected coupled absolute
difference: apply $|u^2-v^2|\le2R_b|u-v|$ to the second moment and the same
bound to the squared means. Since
$|(t+\sigma^2)^{-1/2}-(u+\sigma^2)^{-1/2}|
\le |t-u|/(2\sigma^3)$, the standardized reward differs on a good pair
by at most $Q_r\sqrt\delta$. The separation is in $[0,S_*]$, its good-pair
difference is at most $2\ell_fr$, and its expected difference is at most
$m_s\sqrt\delta$. The identical variance calculation, now with numerator
bounded by $S_*$, gives $Q_s\sqrt\delta$. The logistic derivative and the
power derivative give the displayed $H_r,H_s$, hence fitness differences
on good types are at most $E_F\sqrt\delta$.

**Collision exploration.** The two cloning denominators at a good source
pair differ by at most $(2w_C'+1)\sqrt\delta$, by integration against
$\pi$. The accepted-edge density $\beta$ of Chapter 8 therefore differs
at good source/target type pairs by at most $D_\beta\sqrt\delta$; this is
obtained by subtracting the weight, reciprocal denominator, and acceptance
factor in turn. The acceptance factor has sum-norm Lipschitz constant
$L_a$. Each density is bounded by $1/\kappa_C$.

Lift the two outgoing subprobabilities and incoming Poisson intensities
to $\Lambda$. Discarding bad paired types and comparing good intensities
gives total absolute intensity discrepancy at most
\[
 t_\delta=(D_\beta+2b_0/\kappa_C)\sqrt\delta.
\]
For outgoing subprobabilities, include a cemetery point for no edge; the
full absolute discrepancy is at most $2t_\delta$. For incoming Poisson
processes use their common minimum intensity and independent residual
processes. The chance that a residual point exists is at most the total
residual intensity, hence at most $t_\delta$. Thus $4t_\delta$ is a
conservative failure bound per explored vertex for the combined outgoing
and incoming constructions. This coupling respects the rule that an
incoming child has already used its outgoing edge.

Stop both explorations at $K$ vertices. Before the first mismatch every
paired vertex is good, so the union bound is at most $4Kt_\delta$.
The root-type mismatch costs at most $b_0\sqrt\delta$. The two component
size tails cost at most $2e^{2C}/K$ by {prf:ref}`lem-chaos-component-truncation`. That bound also holds for these limiting rooted laws by its
finite-exploration limit argument.

**Readout.** On matching finite components use one common rotation and
common row jitter. The component means differ by at most $r$, and each
collision velocity differs by at most $(1+2|\alpha_{\rm col}|)r$.
The copied positions differ by at most $r$. Synchronous kinetic noises,
the force Lipschitz bound, and the 1-Lipschitz cap give final position and
velocity differences at most $Ur$ and $Vr$. Consequently
\[
 \mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
 \le b_0\sqrt\delta+4Kt_\delta+2e^{2C}/K+(U+V)\sqrt\delta.
\]
Take $K=\lceil\delta^{-1/4}\rceil$. Then
$K\le2\delta^{-1/4}$ and $K^{-1}\le\delta^{1/4}$, proving the claim.
:::

:::{prf:theorem} Explicit finite-horizon empirical trajectory bound
:label: thm-slct-trajectory

Use the same regime and the explicit Chapter 9 constant
$G=A+4B_*^2$ recorded in this chapter: for every
measurable $|\varphi|\le1$,
\[
 \mathbb E[|(L_N(S_{n+1})-\mathcal F_hL_N(S_n))\varphi|^2\mid S_n]
 \le G/N.
\]
A fully explicit admissible moment budget is obtained as follows. Let
$M_0\ge\mathbb E L_N(S_0)|x|^2$, $f_0=|F(0)|$, and put
\[
 r_C=1+2/\kappa_C,\quad A_K=3(1+\eta L_F)^2,\quad
 b_K=3B^2V_c^2+3\eta^2f_0^2+d(c^2q^2+s^2),
\]
\[
 M_{n+1}=A_Kr_CM_n+A_Kd\sigma_J^2+b_K,\qquad
 \mathcal M_{n+1}=M_{n+1}.
\]
This budget bounds both
$\mathbb E L_N(S_{n+1})|x|^2$ and
$\mathbb E(\mathcal F_hL_N(S_n))|x|^2$.
For arbitrary numerical choices $R>0$ and $0<\ell\le1$, put
\[
 J(R,\ell)=
 \left(1+\left\lceil{2R\sqrt{2d}\over\ell}\right\rceil\right)^d
 \left(1+\left\lceil{2V_{\max}\sqrt{2d}\over\ell}\right\rceil\right)^d,
\]
\[
 a_{N,n}(R,\ell)=2\ell+\frac12J(R,\ell)\sqrt{G/N}
                      +2\mathcal M_{n+1}/R^2.
\]
Define $u_0\ge\mathbb E\mathsf d(L_N(S_0),\mu_0)$ and recursively
\[
 u_{n+1}=\min\{1,a_{N,n}(R_n,\ell_n)+C_{\rm mod}u_n^{1/4}\}.
\]
For $\mu_{n+1}=\mathcal F_h\mu_n$,
\[
 \mathbb E\mathsf d(L_N(S_n),\mu_n)\le u_n,\qquad
 \Pr\{\max_{0\le n\le T}\mathsf d(L_N(S_n),\mu_n)>\varepsilon\}
 \le\min\{1,\varepsilon^{-1}\sum_{n=0}^T u_n\}.
\]
In particular, for each fixed $T$, moment budgets bounded independently
of $N$ and $u_0\to0$ imply convergence in probability of the full
trajectory. One explicit choice is
$R_N=N^{1/(16d)}$, $\ell_N=N^{-1/(16d)}$; then
$J(R_N,\ell_N)=O(N^{3/16})$, with the full coefficient supplied by the
displayed ceiling formula, and $a_{N,n}\to0$. Physical horizon is $hT$;
there are $N(T+1)$ recorded particle states, and each update retains its
actual computational cost.
:::

:::{prf:proof}
For the moment budget, each donor receives expected copy multiplicity at
most $N/[\kappa_C(N-1)]\le2/\kappa_C$ for $N\ge2$, conditionally
on all measurement marks; acceptance only reduces this bound. Retaining
the original row costs one further copy. A singleton retains itself.
Independent centered jitter adds at most $d\sigma_J^2$. In the limiting
rooted law the donor multiplicity bound is $1/\kappa_C$, also bounded by
$2/\kappa_C$. After copying, $|F(x)|\le f_0+L_F|x|$ and
$|v|\le V_c$ give
$|x+Bv+\eta F(x)|^2\le A_K|x|^2+3B^2V_c^2+3\eta^2f_0^2$.
The two independent centered position noises add $d(c^2q^2+s^2)$.
Taking expectations proves the displayed moment recursion in both cases.

Partition the rectangular set
$[-R,R]^d\times[-V_{\max},V_{\max}]^d$ into cells of side at most
$\ell/\sqrt{2d}$, and choose one representative per cell. Its number of
cells is at most $J(R,\ell)$. Match mass $\min\{\rho(C_j),\nu(C_j)\}$ within each cell, at
cost at most $\ell$ per unit mass. The remaining total mass is
$[\sum_j|\rho(C_j)-\nu(C_j)|+\rho(E)+\nu(E)]/2$, where $E$ is the
exterior of the rectangle, and costs at most one per unit. Since
$E\subset\{|x|>R\}$ on the capped state space, this proves, with room
to spare,
\[
 \mathsf d(\rho,\nu)\le2\ell+\frac12\sum_{j=1}^{J}
 |\rho(C_j)-\nu(C_j)|+\rho(|x|>R)+\nu(|x|>R).
\]
Apply the conditional mean-square estimate to every indicator $1_{C_j}$,
then Cauchy--Schwarz, expectation, and Markov's moment bound to obtain
$\mathbb E\mathsf d(L_N(S_{n+1}),\mathcal F_hL_N(S_n))\le a_{N,n}$.
The triangle inequality and the preceding modulus give
$e_{n+1}\le a_{N,n}+C_{\rm mod}\mathbb E[d_n^{1/4}]
\le a_{N,n}+C_{\rm mod}e_n^{1/4}$ by concavity. Induction proves the
recursion; Markov's inequality and a union bound prove the trajectory
probability. Fixed-horizon convergence follows by finite induction, even
though this modulus does not establish any long-time contraction.
:::

:::{prf:corollary} Independent initialization and a displayed rate
:label: cor-slct-iid

For independent initial rows of law $\mu_0$ with
$\mu_0|x|^2\le M_0$, an admissible initialization bound is
\[
 u_0=\min\{1,2\ell+J(R,\ell)/(2\sqrt N)+2M_0/R^2\}.
\]
With the choices $R_N=N^{1/(16d)}$, $\ell_N=N^{-1/(16d)}$, all finite
horizon error bounds above are numerical formulas in the displayed
parameters, $M_0$, $N$, $T$, and $\varepsilon$. They converge to zero for
each fixed $T$. No uniformity as $T\to\infty$ is claimed by this
particular, deliberately conservative, fourth-root recursion.
:::

:::{prf:proof}
For each initial cell indicator the empirical variance is at most $1/N$;
apply exactly the matching argument in the preceding proof, with $G=1$
and target $\mu_0$. The second moment bound supplies both exterior terms.
The remaining claims follow by substitution and finite induction.
:::

:::{prf:remark} Scope of the imported consistency estimate
:label: rem-slct-consistency-scope

The all-alive bounded-reward regime used here extends the list of examples
in the canonical-regime definition. Its use of the explicit scalar
one-step estimate does not require a new unproved concentration result.
In its proof, reward statistics are frozen functions of the deterministic
input array; the sampled normalization perturbation concerns only bounded
squashed diversity. The bounds on accepted edges, replacement influences,
and component moments use the bounded logistic fitness factors, positive
regularizers, and the same companion lower bounds. They remain unchanged
for bounded Lipschitz reward. The final kinetic step is a Markov kernel,
so for a bounded measurable output test its conditional kinetic average
is still bounded by the same norm; all those scalar bounds pass through
unchanged. The present explicit moment calculation and continuity proof
supply the remaining existence and finite-horizon integrability arguments.

An unbounded reward, absorbing boundary, or merely regional force
regularity requires adding its normalization, status-change, and excursion
terms to this proof. The following two theorems supply the unbounded
quadratic-growth reward extension explicitly. One must not substitute its parameters into
$C_{\rm mod}$ while dropping those terms. The theorem establishes a full
quantitative mean-field limit in the stated nonconvex regime; it does not
identify the bounded-reward condition with strong convexity or with a
single attracting phase.
:::

:::{prf:theorem} Active quadratic-growth reward and an explicit moment-class modulus
:label: thm-slct-quadratic-reward

Keep the all-alive algorithm and force assumptions of
{prf:ref}`def-slct-regime`, but replace bounded reward by
\[
 |R(z)|\le K_0+K_2|x|^2,\qquad
 |R(z)-R(z')|\le(L_0+L_1R)|z-z'|
 \quad(|x|,|x'|\le R,\ |v|,|v'|\le V_{\max}).
\]
The constants $K_0,K_2,L_0,L_1$ are nonnegative. For laws with
$\mu|x|^8,\nu|x|^8\le H$, define
\[
 B_I=2H+1,\quad R_0=K_0+K_2,\quad L_*=L_0+L_1,\quad
 \overline R=K_0+K_2H^{1/4},
\]
\[
 C_1=L_*+2K_0B_I+2K_2H^{1/4}B_I^{3/4},
\]
\[
 C_2=2R_0L_*+4K_0^2B_I+4K_2^2H^{1/2}B_I^{1/2},\qquad
 C_V=C_2+2\overline R C_1,
\]
\[
 Q_{r,8}={L_*+C_1\over\sigma_r}
          +{(R_0+\overline R)C_V\over2\sigma_r^3},
\]
\[
 B_M=(1+3/\kappa_D)B_I+4w_D'/\kappa_D,\quad
 M_S=2\ell_f+S_*B_M,\quad
 Q_{s,8}={2\ell_f+M_S\over\sigma_s}
               +{2S_*^2M_S\over\sigma_s^3},
\]
\[
 E_8=H_rQ_{r,8}+H_sQ_{s,8},\quad
 D_8={2w_C'\over\kappa_C}
       +{2w_C'+B_I\over\kappa_C^2}
       +{2L_aE_8\over\kappa_C},
\]
\[
 C_8(H)=B_M+8(D_8+2B_M/\kappa_C)+2e^{2C}+U+V.
\]
All symbols on the right were given primitive formulas in
{prf:ref}`thm-slct-modulus`. Then
\[
 \boxed{\mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
       \le\min\{1,C_8(H)\mathsf d(\mu,\nu)^{1/32}\}.}
\]
The exponent $p_r$ may be strictly positive. The scalar consistency constant $G=A+4B_*^2$
also applies in this regime: for each deterministic finite input every
reward and reward normalizer is finite, and frozen during the innovation-
replacement proof. Its estimates depend on the bounded logistic reward
factor rather than on an upper bound for the raw reward. The eighth
moment bound makes the population reward second moment finite. Thus no
new unproved concentration estimate is being substituted.
For the ordinary Rastrigin
reward
\[
 R(x)=-\sum_{j=1}^d[x_j^2+A(1-\cos(2\pi x_j))],\qquad A\ge0,
\]
one may take
\[
 K_0=2Ad,\quad K_2=1,\quad L_0=2\pi A\sqrt d,\quad L_1=2.
\]
For force $F=-\nabla(-R)$, take $L_F=2+4\pi^2A$ and $f_0=|F(0)|=0$.
:::

:::{prf:proof}
For $0<\delta=\mathsf d(\mu,\nu)\le1$ couple inputs optimally and choose
$r=\delta^{1/2}$ and spatial cutoff $R=\delta^{-1/32}$. Declare a pair
bad if its distance exceeds $r$ or either position exceeds $R$. Its mass
$b$ obeys
\[
 b\le\delta/r+2H/R^8\le B_I\delta^{1/4}.
\]
Under this coupling, the reward difference on good pairs is at most
$(L_0+L_1R)r$. Hölder's inequality gives
$\mathbb E[|x|^2 1_{\rm bad}]\le H^{1/4}b^{3/4}$ and
$\mathbb E[|x|^4 1_{\rm bad}]\le H^{1/2}b^{1/2}$.
Thus, with $q_1=\mathbb E|R(z)-R(z')|$ and
$q_2=\mathbb E|R(z)^2-R(z')^2|$,
\[
 q_1\le(L_0+L_1R)r+2K_0b+2K_2H^{1/4}b^{3/4}
       \le C_1\delta^{3/16},
\]
\[
 q_2\le2(K_0+K_2R^2)(L_0+L_1R)r
          +4K_0^2b+4K_2^2H^{1/2}b^{1/2}
       \le C_2\delta^{1/8}.
\]
Both absolute reward means are at most $\overline R$. The variance
 difference is consequently at most $q_2+2\overline Rq_1
\le C_V\delta^{1/8}$. On good pairs the standardized reward difference
is at most
\[
 { (L_0+L_1R)r+q_1\over\sigma_r}
 +{(K_0+K_2R^2+\overline R)(q_2+2\overline Rq_1)
       \over2\sigma_r^3}
 \le Q_{r,8}\delta^{1/16}.
\]
The exponents follow directly from $R=\delta^{-1/32}$:
$R^2\delta^{1/8}=\delta^{1/16}$, and all other displayed terms have
at least this power.

Apply the measurement coupling from the bounded-reward proof, now
counting the larger input bad set. Its bad marked mass is at most
$(1+3/\kappa_D)b+4w_D'r/\kappa_D\le B_M\delta^{1/4}$.
The coupled mean separation difference is at most
$2\ell_fr+S_*B_M\delta^{1/4}\le M_S\delta^{1/4}$.
The same variance and inverse-scale calculation gives standardized
separation difference at most $Q_{s,8}\delta^{1/4}$, hence at most
$Q_{s,8}\delta^{1/16}$. The fitness difference is therefore at most
$E_8\delta^{1/16}$. Subtracting weights, denominators and gates as before
bounds good-pair edge-density differences by $D_8\delta^{1/16}$.

Use the same outgoing/Poisson coupling, with
$K=\lceil\delta^{-1/32}\rceil$. Its total error is at most
\[
 B_M\delta^{1/4}
 +4K(D_8\delta^{1/16}+2B_M\delta^{1/4}/\kappa_C)
 +2e^{2C}/K+(U+V)\delta^{1/2}
 \le C_8(H)\delta^{1/32}.
\]
The case $\delta=0$ follows by identical laws. The Rastrigin constants
follow from $0\le1-\cos t\le2$, the bound on the sine vector by
$\sqrt d$, and its Hessian diagonal entries
$2+4\pi^2A\cos(2\pi x_j)$.
:::

:::{prf:theorem} Explicit trajectory probability with unbounded reward
:label: thm-slct-unbounded-trajectory

Use {prf:ref}`thm-slct-quadratic-reward` and assume
$\mathbb E L_N(S_0)|x|^8\le M_{8,0}$ and
$\mu_0|x|^8\le M_{8,0}$. Let
\[
 g_{8,d}=d(d+2)(d+4)(d+6),\quad
 \sigma_{\rm pos}^2=c^2q^2+s^2,\quad r_C=1+2/\kappa_C,
\]
\[
 M_{8,n+1}=3^7\left[
 (1+\eta L_F)^8 2^7(r_CM_{8,n}+\sigma_J^8g_{8,d})
 +(BV_c+\eta f_0)^8+\sigma_{\rm pos}^8g_{8,d}\right].
\]
For arbitrary thresholds $H_n\ge M_{8,n}$, tolerances $t_n>0$, and
cutoffs $R_n,\ell_n>0$ with $\ell_n\le1$, define
\[
 a_{N,n}=2\ell_n+\tfrac12J(R_n,\ell_n)\sqrt{G/N}
                     +2M_{8,n+1}^{1/4}/R_n^2,
\]
where $G=A+4B_*^2$ is the explicit scalar consistency constant. Let
$v_0>0$ and
\[
 v_{n+1}=\min\{1,t_n+C_8(H_n)v_n^{1/32}\}.
\]
Then
\[
 \Pr\{\exists n\le T:\mathsf d(L_N(S_n),\mu_n)>v_n\}
 \le \Pr\{\mathsf d(L_N(S_0),\mu_0)>v_0\}
       +\sum_{n=0}^{T-1}\left({M_{8,n}\over H_n}
                                      +{a_{N,n}\over t_n}\right).
\]
For independent initialization its first term is at most
$[2\ell+J(R,\ell)/(2\sqrt N)+2M_{8,0}^{1/4}/R^2]/v_0$.
All constants, horizon dependence, and truncation probabilities are
explicit. This proves a valid quantitative mean-field limit for active
raw Rastrigin reward, without attraction or synchronization assumptions.
:::

:::{prf:proof}
The copy-multiplicity argument used for the second moment applies to
$|x|^8$. For $u,w\in\mathbb R^d$,
$|u+w|^8\le2^7(|u|^8+|w|^8)$, so post-copy jitter gives the bound
$2^7(r_CM_{8,n}+\sigma_J^8g_{8,d})$. Combine the deterministic force and
position term, the bounded velocity and force-at-origin term, and the
centered Gaussian position increment. The inequality
$|u+w+y|^8\le3^7(|u|^8+|w|^8+|y|^8)$ gives exactly the stated recursion.
The Gaussian identity
$\mathbb E|Z|^8=d(d+2)(d+4)(d+6)$ follows by expanding the fourth moment
of a chi-square variable, or by four integrations of its gamma density.
The same calculation applies to the limiting rooted update. Induction
therefore bounds both empirical expected eighth moments and deterministic
population eighth moments by $M_{8,n}$.

By Markov's inequality the event
$L_N(S_n)|x|^8>H_n$ has probability at most $M_{8,n}/H_n$; the population
law always belongs to the same moment class since $H_n\ge M_{8,n}$.
The finite-cell sampling proof gives expected one-step metric defect at
most $a_{N,n}$, because second moments are bounded by the fourth root of
eighth moments. Its probability of exceeding $t_n$ is at most
$a_{N,n}/t_n$. On the intersection of all these good events, the moment-
class modulus and induction give $d_n\le v_n$. Taking the union bound
proves the displayed probability, preserving all temporal and particle
dependence.

For completeness, choose $R_n=N^{1/(16d)}$, $\ell_n=N^{-1/(16d)}$,
$H_n=\max\{1,M_{8,n}\}\log(N+e)$ and $t_n=\sqrt{a_{N,n}}$.
Under independent initialization choose $v_0$ to be the square root of
its displayed expected-distance bound. For each fixed $T$ the failure
probability tends to zero. The sampling quantities decay as positive
powers of $N$. For $M=\max\{1,H\}$ its explicit coefficients obey
$B_I(H)\le B_I(1)M$, $C_1(H)\le C_1(1)M$,
$C_2(H)\le C_2(1)M$, and $B_M(H)\le B_M(1)M$.
Also $\overline R(H)\le\overline R(1)M^{1/4}$, so
$C_V(H)\le C_V(1)M^{5/4}$ and
$Q_{r,8}(H)\le Q_{r,8}(1)M^{3/2}$.
The remaining separation and edge coefficients are sums of nonnegative
terms growing no faster than this power. Consequently
\[
 C_8(H)\le C_8(1)\max\{1,H\}^{3/2}.
\]
Thus the constants used here grow at most as
$C_8(1)[\max\{1,M_{8,n}\}\log(N+e)]^{3/2}$. Finite induction in
$v_{n+1}\le t_n+C_8(H_n)v_n^{1/32}$ therefore gives $v_n\to0$ at every
fixed $n$. This establishes the limit without substituting a qualitative
continuity constant at any stage.
:::

:::{prf:corollary} Uniform-ball and coincident initialization
:label: cor-slct-initialization-moments

For independent positions uniform in $x_0+B_{R_{\rm init}}$ and arbitrary
independent identically distributed capped velocities, the eighth-moment
trajectory theorem admits

$$
M_{8,0}=2^7\left(|x_0|^8+\frac{d}{d+8}R_{\rm init}^8\right).
$$
When $x_0=0$, the sharper exact value is
$M_{8,0}=dR_{\rm init}^8/(d+8)$. Use the independent-initialization
bound in that theorem. If every initial row is exactly $(x_0,v_0)$,
use $M_{8,0}=|x_0|^8$ and initial distance zero. No separation between
initial rows is required. These conclusions concern the stated all-alive
configuration; an absorbing boundary adds its own survival requirement.
:::

:::{prf:proof}
For a uniform ball, polar integration gives
$\mathbb E|X-x_0|^8=dR_{\rm init}^{-d}\int_0^{R_{\rm init}}r^{d+7}dr
=dR_{\rm init}^8/(d+8)$; use continuity for zero radius.
The inequality $|x_0+y|^8\le2^7(|x_0|^8+|y|^8)$ gives the translated
bound. Coincident deterministic rows have empirical law exactly equal
to their Dirac initial population law. Measurement and acceptance
normalizers remain defined by the positive floors and standardization
regularizers in the parameter register. The previous proofs allow
atoms, fitness ties and zero accepted edges; no distinct-position
hypothesis was used.
:::

:::{prf:remark} Zero reward exponent
:label: rem-slct-zero-reward

If $p_r=0$, the reward rescaling factor is identically one. One can set
$H_r=Q_r=m_r=0$ in the bounded-reward modulus and remove its reward
boundedness and Lipschitz hypotheses entirely. Only the quantities
actually evaluated by the chosen algorithm need exist; an implementation
that still evaluates unused reward normalizers must define them or skip
them. The active-reward theorem above instead allows $p_r>0$ and keeps
all reward moment and normalization terms.
:::

:::{prf:corollary} Quantitative propagation of chaos for finitely many rows
:label: cor-slct-finite-marginals

Assume the initialization is exchangeable and the canonical update uses its
permutation-equivariant randomization. Use either {prf:ref}`thm-slct-trajectory`, or
{prf:ref}`thm-slct-unbounded-trajectory` with $u_n=\min(1,v_n+P_n)$,
where $P_n$ is its displayed failure bound through time $n$. Let $\mathcal L_{N,n}^{(k)}$ denote the law
of the first $k\le N$ rows after $n$ complete updates. Give the $k$-row
space the bounded metric

$$
c_k((z_i),(z_i'))=\frac1k\sum_{i=1}^k\min\{1,|z_i-z_i'|\}.
$$

For its induced Wasserstein distance $\mathsf d_k$,

$$
\mathsf d_k(\mathcal L_{N,n}^{(k)},\mu_n^{\otimes k})
\le \min\left\{1,u_n+1-\frac{N(N-1)\cdots(N-k+1)}{N^k}\right\}
\le\min\left\{1,u_n+\frac{k(k-1)}{2N}\right\}.
$$

The same conclusion holds without exchangeable initialization for $k$
ordered distinct labels sampled uniformly independently of the swarm at
observation time. This is a statement about rows at one common time;
no assertion of independent time histories follows from it.
:::

:::{prf:proof}
In the unbounded-reward case, $\mathsf d\le1$ gives
$\mathbb E\mathsf d(L_N,\mu_n)\le v_n+P_n$. Condition on the full swarm. Sampling $k$ labels independently with replacement
gives conditional row law $L_N^{\otimes k}$. Sampling ordered distinct labels
gives the conditional without-replacement law. Couple the two label samples
identically whenever the first sample has no repeated label, and on the
remaining event draw the distinct sample from its conditional complement
coupling. Equivalently use rejection sampling: keep the first sample if it
is distinct, and otherwise draw a fresh uniform distinct tuple. This has
exactly the uniform distinct marginal. The probability of failure is
$1-(N)_k/N^k\le\binom{k}{2}/N$, and $c_k\le1$ bounds its cost.

For each empirical law choose an optimal coupling with $\mu_n$ for $c_0$;
measurable such choices exist for this continuous bounded cost on a Polish
space (or use measurable arbitrarily close approximations). Its $k$-fold
product has average cost $\mathsf d(L_N,\mu_n)$ and second marginal
$\mu_n^{\otimes k}$. Averaging over the swarm bounds the distance of the
with-replacement mixture to $\mu_n^{\otimes k}$ by $u_n$.
The triangle inequality proves the result for uniformly sampled distinct
labels. Exchangeability identifies that law with the first $k$ rows.
Permutation equivariance preserves exchangeability at every update.
:::

(slcs-structural-assembly)=
### 9.1. Structural assembly and regional force defects

:::{div} feynman-prose
Start with the declared regions: basins, transition regions and the exterior.
For each region, record how large the force can be and how much it changes over
a specified distance. At an interface, record the change between the two sides.
A sharp interface must appear somewhere in the estimate; calling both interiors
regular does not make the interface disappear.

These profiles enter the actual kinetic update at its two force evaluations.
Gaussian motion can carry a walker outside the region where a bound was
measured, so the calculation also charges that excursion probability and the
corresponding tail budget. Finite profiles give numerical error bounds. A
nonvanishing defect or an infinite envelope identifies precisely why this
certificate fails to establish the requested accuracy. The underlying
measurable dynamics may still be defined. Named landscapes serve as checks of
these regional formulas; the decomposition supplies the argument.
:::

:::{prf:definition} Pairwise structural data
:label: def-slcs-data

Let $(A_i)_{i\in I}$ be a finite or countable Borel partition of physical
space into declared basin, transition, and exterior regions. Regions are
mathematical input; no identification with attraction basins of the
population map is made. For a spatial cutoff $R$, use the closed ball
$B_R$ and let
\[
 d_{ij}(R)=\inf\{|x-y|:x\in A_i\cap B_R,
                               y\in A_j\cap B_R\},
\]
with infimum $+\infty$ for an empty pair. Supply upper bounds
\[
 |F(x)|\le g_{0,i}+g_{1,i}|x|\quad(x\in A_i),
\]
\[
 |F(x)-F(y)|\le L^F_{ij}(R)|x-y|+J^F_{ij}(R)
 \quad(x\in A_i\cap B_R, y\in A_j\cap B_R),
\]
\[
 |R(z)|\le K_{0,i}+K_{2,i}|x|^2\quad(x\in A_i),
\]
\[
 |R(z)-R(z')|\le[L^R_{0,ij}+L^R_{1,ij}R]|z-z'|+J^R_{ij}(R)
 \quad(x\in A_i\cap B_R, x'\in A_j\cap B_R),
\]
for stored velocities in the cap. All bounds concern the actual specified
measurable force and reward. Constants may be $+\infty$. The $J$ profiles
record unresolved inter-region jumps; they must not be dropped when
assembling regional regularity bounds. A valid alternative is to supply
any nondecreasing pair modulus $\omega^F_{ij}(R,r)$ bounding the force
increment whenever additionally $|x-y|\le r$.

The partition may be chosen to isolate narrow communication regions, but
these regularity profiles alone do not assert any lower bound on crossing
probabilities. Those are separate full-update structural quantities.
:::

:::{prf:proposition} Assembly with explicit interface costs
:label: prop-slcs-assembly

Use $0/0=0$ when the numerator is zero, and $a/0=+\infty$ for $a>0$.
Pairs with empty intersections are excluded. Set
\[
 g_0=\sup_i g_{0,i},\quad g_1=\sup_i g_{1,i},\quad
 K_0=\sup_i K_{0,i},\quad K_2=\sup_i K_{2,i}.
\]
Define
\[
 L_F=\sup_{R>0}\sup_{i,j}
       \left[L^F_{ij}(R)+{J^F_{ij}(R)\over d_{ij}(R)}\right].
\]
If supplied numbers $L_0,L_1\ge0$ satisfy, for every $R>0$ and nonempty
pair,
\[
 L^R_{0,ij}+L^R_{1,ij}R+J^R_{ij}(R)/d_{ij}(R)
                                  \le L_0+L_1R,
\]
then the force and reward hypotheses in the quantitative trajectory
theorems hold with exactly these assembled constants, whenever they are
finite. A simpler sufficient choice is
\[
 L_0=\sup_{i,j}L^R_{0,ij}
       +\sup_{R>0,i,j}J^R_{ij}(R)/d_{ij}(R),\qquad
 L_1=\sup_{i,j}L^R_{1,ij}.
\]
No basin-count factor is required in these suprema.

For regional eighth-moment budgets
$\int_{x\in A_i}|x|^8\,\mu(dz)\le m_{8,i}$, an admissible population
budget is $H=\sum_i m_{8,i}$. More precisely, for each $R>0$,
\[
 \mu(|x|>R)\le\sum_i t_i(R),\quad
 \int_{|x|>R}|x|^p\,d\mu\le\sum_i q_{p,i}(R)
\]
whenever $t_i(R)$ and $q_{p,i}(R)$ bound the corresponding integrals on
$A_i\cap B_R^c$. In the absence of sharper regional tail formulas,
$t_i(R)=m_{8,i}/R^8$ and
$q_{p,i}(R)=m_{8,i}/R^{8-p}$ are admissible for $0\le p\le8$.
:::

:::{prf:proof}
For $x\in A_i$, $y\in A_j$ inside $B_R$, their distance is at least
$d_{ij}(R)$. Thus $J^F_{ij}(R)\le[J^F_{ij}(R)/d_{ij}(R)]|x-y|$
whenever the coefficient is finite; the zero-jump case holds also when
the separation infimum is zero. Apply the declared pair inequality and
take suprema. The same argument applies to rewards because
$|z-z'|\ge|x-y|$. Pointwise growth bounds pass to the suprema directly.
Partition integrals add by countable additivity, since the integrands are
nonnegative. Finally, on $|x|>R$, $1\le|x|^8/R^8$ and
$|x|^p\le|x|^8/R^{8-p}$. These observations prove every claim.
:::

:::{prf:lemma} A justified zero-jump interface certificate
:label: lem-slcs-gluing

Suppose every line segment has a finite subdivision into subsegments
whose interiors lie in single regions, the force has continuous matching
traces at the subdivision endpoints, and on each subsegment the force
is $L_i$-Lipschitz with $L_i\le L$. Then the force is globally
$L$-Lipschitz. The identical conclusion applies to reward on any declared
bounded spatial ball and capped velocity domain for which these segment
conditions hold.
:::

:::{prf:proof}
For endpoints $x,y$, let $x=x_0,\ldots,x_k=y$ be the subdivision in
segment order. Continuity of the traces extends the regional increment
bound to both endpoints of each subsegment. The triangle inequality gives
$|F(x)-F(y)|\le\sum_{j=1}^kL|x_j-x_{j-1}|=L|x-y|$.
This argument uses the specified finite subdivision; separate Lipschitz
bounds on arbitrary measurable regions alone do not imply the claim.
:::

:::{prf:definition} A finite-radius force modulus assembled from regions
:label: def-slcs-local-modulus

Let $M_R=\sup_i(g_{0,i}+g_{1,i}R)$. Define the nondecreasing bound
\[
 \Omega_R(r)=\min\left\{2M_R,
       \sup_{i,j:\ d_{ij}(R)\le r}
                      [L^F_{ij}(R)r+J^F_{ij}(R)]\right\}.
\]
Empty pairs are excluded. If explicit pair moduli are supplied instead,
replace the affine expression by $\omega^F_{ij}(R,r)$ and take its
nondecreasing envelope in $r$ if necessary. This bounds
$|F(x)-F(y)|$ for all $x,y\in B_R$ with $|x-y|\le r$.
It is a force-increment bound, not an unknown convergence constant.
:::

:::{prf:theorem} Regional force regularity with charged Gaussian excursions
:label: thm-slcs-regional-modulus

Keep the all-alive canonical algorithm and the active quadratic-growth
reward assumptions from {prf:ref}`thm-slct-quadratic-reward`, assembled
by {prf:ref}`prop-slcs-assembly`. Replace global force Lipschitzness by
finite linear-growth constants $g_0,g_1$ and the declared moduli
$\Omega_R$. For $\mu|x|^8,\nu|x|^8\le H$ and
$0<\delta=\mathsf d(\mu,\nu)\le1$, retain the explicit constants
$B_M,D_8,C$ from that theorem. Choose $G>0$ and $J>0$ when $\sigma_J>0$ (allow $J=0$ otherwise), and set
\[
 r=\delta^{1/2},\quad R_0=\delta^{-1/32},\quad
 R_p=R_0+J,
\]
\[
 R_m=(1+\eta g_1)R_p+BV_c+\eta g_0+cqG,
\]
\[
 u=(1+Bk_v)r+\eta\Omega_{R_p}(r),\qquad
 w=ak_vr+ac\Omega_{R_p}(r)+c\Omega_{R_m}(u).
\]
With $\sigma_J=0$, define $p_J=0$; otherwise let
\[
 p_J=\min\{1,2d e^{-J^2/(2d\sigma_J^2)}\},\qquad
 p_O=\min\{1,2d e^{-G^2/(2d)}\}.
\]
Then the following full-update continuity bound holds without any global
force-Lipschitz constant:
\[
 \boxed{\mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
 \le\min\{1,C_{\rm graph}(H)\delta^{1/32}+u+w+p_J+p_O\},}
\]
where the completely explicit graph constant is
\[
 C_{\rm graph}(H)=B_M+8(D_8+2B_M/\kappa_C)+2e^{2C}.
\]
For example, the choices
\[
 J=\sigma_J\sqrt{2d\log(2d/\delta^{1/32})},\qquad
 G=\sqrt{2d\log(2d/\delta^{1/32})}
\]
give $p_J+p_O\le2\delta^{1/32}$, with $J=0$ and $p_J=0$ when
$\sigma_J=0$. The certificate tends to zero whenever the two displayed
regional increment contributions $\Omega_{R_p}(r)$ and
$\Omega_{R_m}(u)$ tend to zero. If they do not, this particular structural
bound records that failure; it does not declare the measurable dynamics
undefined.
:::

:::{prf:proof}
The measurement, reward-normalization and component-exploration parts of
{prf:ref}`thm-slct-quadratic-reward` do not use force regularity. They give
a coupling with graph failure probability at most
$C_{\rm graph}(H)\delta^{1/32}$ and, on graph success, paired source
positions within $r$, both source positions inside $B_{R_0}$, and paired
collision velocities differing by at most $k_vr$. The source bound also
covers a copied donor because all explored types are declared good on
this event.

Use common root jitter and common O-noise. The Gaussian coordinate union
bound gives the stated $p_J,p_O$. On the complementary good-noise event,
both post-jitter positions belong to $B_{R_p}$, so their first force
increment is bounded by $\Omega_{R_p}(r)$. The explicit BAOAB position
formula is $x_2=x+Bv+\eta F(x)+cq\xi$. Its norm is at most $R_m$ and
its paired difference is at most $u$. The final pre-cap velocity is
$a(v+cF(x))+q\xi+cF(x_2)$, whose paired difference is at most $w$.
The cap is 1-Lipschitz and the final additive position noise is common.
Therefore the final state cost is at most $u+w$ on the good event and at
most one otherwise. Sum the probabilities and the good-event cost.
The displayed choices of $J,G$ follow by direct substitution into the
Gaussian bounds. No independent-walker assumption is used.
:::

:::{prf:corollary} Structural moment recursion and trajectory certificate
:label: cor-slcs-trajectory

In {prf:ref}`thm-slcs-regional-modulus`, replace the eighth-moment recursion
by the following linear-growth version:
\[
 M_{8,n+1}=3^7\left[
 (1+\eta g_1)^8 2^7(r_CM_{8,n}+\sigma_J^8g_{8,d})
 +(BV_c+\eta g_0)^8+\sigma_{\rm pos}^8g_{8,d}\right].
\]
Let $\Psi_H(\delta)$ denote the explicit bound in that theorem, with
chosen Gaussian cutoffs, and set $\Psi_H(0)=0$.
For a usable deterministic error level $v>0$, use the nondecreasing
bound
\[
 \widehat\Psi_H(v)=\min\{1,\sup_{0<\delta\le\min\{v,1\}}
            [C_{\rm graph}(H)\delta^{1/32}+u(\delta)+w(\delta)
                                         +p_J(\delta)+p_O(\delta)]\}.
\]
This supremum is over displayed structural profiles and scalar numerical
cutoffs, not over the unknown evolution or its optimal convergence rate.
Then the complete high-probability trajectory theorem remains valid with
\[
 v_{n+1}=\min\{1,t_n+\widehat\Psi_{H_n}(v_n)\}
\]
in place of its power recursion and with the same explicit failure bound
\[
 \Pr\{d_0>v_0\}+\sum_{n<T}
              [M_{8,n}/H_n+a_{N,n}/t_n].
\]
Thus the regional profiles, their interface defects, noise excursions and
moment-tail budgets explicitly determine whether this trajectory
certificate closes at the requested horizon and tolerance.
:::

:::{prf:proof}
This is an unbounded-space moment and tail argument; the partition and
Gaussian cutoffs do not replace the dynamics by compact support.
The growth constants $g_0,g_1$ are distinct from any force-increment
constant: moments can remain finite even when the continuity certificate
is uninformative. The moment calculation uses only
$|F(x)|\le g_0+g_1|x|$; replacing its
previous bound gives the displayed recursion term by term. On each
localized event where both input laws have eighth moment at most $H_n$
and their metric distance is at most $v_n$, the regional theorem and its
nondecreasing envelope bound the population-map discrepancy by
$\widehat\Psi_{H_n}(v_n)$. The one-step sampling, Markov, and union-bound
arguments are unchanged. This proves the claim without assuming a
positive rate when the structural modulus does not vanish.
:::

:::{prf:corollary} A closed regional Hölder certificate
:label: cor-slch-holder-closure

In {prf:ref}`thm-slcs-regional-modulus`, suppose the assembled regional
force profiles satisfy the explicit bound
\[
 \Omega_R(s)\le K(1+R)^p s^\alpha\quad(R,s>0),
 \qquad K\ge0,\quad0<\alpha\le1,\quad p\ge0.
\]
This is a bound on all relevant region pairs, including interfaces; it
does not follow from within-region estimates alone. Assume the separate
linear-growth envelope $|F(x)|\le g_0+g_1|x|$, and require
\[
 p<\frac{16\alpha^2}{1+\alpha}.
\]
Define entirely from the declared landscape and algorithm parameters
\[
 C_G=\sqrt{2d}\,[\sqrt{\log(2d)}+1],\qquad
 K_p=1+\sigma_JC_G,
\]
\[
 K_m=(1+\eta g_1)K_p+BV_c+\eta g_0+cqC_G,
\]
\[
 K_{F,1}=K(1+K_p)^p,\quad e_1=\alpha/2-p/32,\quad
 C_u=1+Bk_v+\eta K_{F,1},
\]
\[
 K_{F,2}=K(1+K_m)^pC_u^\alpha,\quad
 e_2=\alpha e_1-p/32,\quad
 C_w=ak_v+acK_{F,1}+cK_{F,2},
\]
\[
 \beta_H=\min\{1/32,e_2\}>0,\qquad
 C_H(H)=C_{\rm graph}(H)+2+C_u+C_w.
\]
For the actual full population map and input eighth moments at most $H$,
\[
 \boxed{\mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
       \le\min\{1,C_H(H)\mathsf d(\mu,\nu)^{\beta_H}\}.}
\]
Moreover
\[
 C_H(H)\le[C_{\rm graph}(1)+2+C_u+C_w]
                               \max\{1,H\}^{3/2}.
\]
Consequently the structural high-probability trajectory recursion closes
at every fixed horizon with these fully explicit constants. In particular,
it allows $\alpha<1$ and polynomial growth of local regularity constants;
no global force-Lipschitz certificate is needed.
:::

:::{prf:proof}
Let $0<\delta\le1$, write $u_0=\log(1/\delta)$, and use the Gaussian
cutoffs in {prf:ref}`thm-slcs-regional-modulus`. Since
$\sqrt{u_0/32}\le e^{u_0/32}$ and $e^{u_0/32}\ge1$,
\[
 G=\sqrt{2d}\sqrt{\log(2d)+u_0/32}
       \le C_G\delta^{-1/32},\qquad
 J=\sigma_JG\le\sigma_JC_G\delta^{-1/32}.
\]
For example, the elementary inequality used here follows from
$\sqrt v e^{-v}\le(2e)^{-1/2}<1$ for $v\ge0$.
Thus $R_p\le K_p\delta^{-1/32}$ and
$R_m\le K_m\delta^{-1/32}$. With $r=\delta^{1/2}$,
\[
 \Omega_{R_p}(r)\le
 K(1+K_p\delta^{-1/32})^p\delta^{\alpha/2}
 \le K_{F,1}\delta^{e_1}.
\]
The assumed inequality on $p$ is precisely
$e_2=\alpha^2/2-p(1+\alpha)/32>0$.
It also implies $0<e_2\le e_1\le1/2$ because $0<\alpha\le1$ and
$p\ge0$. Therefore the paired intermediate position bound is
\[
 u=(1+Bk_v)\delta^{1/2}+\eta\Omega_{R_p}(r)
       \le C_u\delta^{e_1},
\]
and the second force evaluation satisfies
\[
 \Omega_{R_m}(u)\le
 K(1+K_m\delta^{-1/32})^p C_u^\alpha\delta^{\alpha e_1}
       \le K_{F,2}\delta^{e_2}.
\]
The paired final velocity bound is consequently
$w\le C_w\delta^{e_2}$. The graph error and the Gaussian exceptional
probabilities together are at most
$[C_{\rm graph}(H)+2]\delta^{1/32}$.
Adding $u+w$ gives the stated power modulus. The case $\delta=0$
follows by identical input laws.

For the growth in $H$, the same coefficient comparison used for $C_8$
in {prf:ref}`thm-slct-unbounded-trajectory` applies after omitting its
kinetic summand $U+V$: each graph coefficient is a sum of nonnegative
terms of growth at most $\max\{1,H\}^{3/2}$. Hence
$C_{\rm graph}(H)\le C_{\rm graph}(1)\max\{1,H\}^{3/2}$.
The additional constants $2+C_u+C_w$ are independent of $H$, proving
the displayed bound.

To make fixed-horizon closure explicit, use the linear-growth moment
recursion in {prf:ref}`cor-slcs-trajectory` and set
\[
 R_n=N^{1/(16d)},\quad\ell_n=N^{-1/(16d)},\quad
 H_n=\max\{1,M_{8,n}\}\log(N+e),\quad
 t_n=\sqrt{a_{N,n}}.
\]
The probability bound is exactly
\[
 \Pr\{d_0>v_0\}+\sum_{n<T}
       \left[\frac{M_{8,n}}{\max\{1,M_{8,n}\}\log(N+e)}
                                      +\sqrt{a_{N,n}}\right],
\]
with the fully numerical recursion
\[
 v_{n+1}=\min\{1,t_n+C_H(H_n)v_n^{\beta_H}\}.
\]
For independent initialization, take $v_0$ to be the square root of
its explicit expected-distance bound from
{prf:ref}`thm-slct-unbounded-trajectory`. To display the decay constants themselves, write
$G_{\rm cons}=A+4B_*^2$ for the scalar one-step consistency constant
(distinct from the Gaussian cutoff $G$), and put
\[
 J_0=(2+2\sqrt{2d})^d(2+2V_{\max}\sqrt{2d})^d,\quad
 A_n=2+\tfrac12J_0\sqrt{G_{\rm cons}}+2M_{8,n+1}^{1/4},
\]
\[
 A_{\rm init}=2+\tfrac12J_0+2M_{8,0}^{1/4},\quad
 C_{*,n}=[C_{\rm graph}(1)+2+C_u+C_w]
                       \max\{1,M_{8,n}\}^{3/2}.
\]
The elementary bound $1+\lceil y\rceil\le2+y$ gives
$J(R_n,\ell_n)\le J_0N^{3/16}$ and
$a_{N,n}\le A_nN^{-1/(16d)}$. The initial expected-distance bound is
at most $A_{\rm init}N^{-1/(16d)}$.
Write $\beta=\beta_H$, $D_0=\sqrt{A_{\rm init}}$, and
\[
 D_{n+1}=\sqrt{A_n}+C_{*,n}D_n^\beta,\qquad
 b_n=\frac32\frac{1-\beta^n}{1-\beta}.
\]
Induction in the displayed numerical recursion now gives
\[
 v_n\le D_n[\log(N+e)]^{b_n}
                         N^{-\beta^n/(32d)}.
\]
Indeed $b_{n+1}=3/2+\beta b_n$ and
$N^{-1/(32d)}\le N^{-\beta^{n+1}/(32d)}$; all logarithms here are
at least one. The failure probability is consequently bounded by
\[
 \left[\sqrt{A_{\rm init}}+\sum_{n<T}\sqrt{A_n}\right]
           N^{-1/(32d)}+\frac{T}{\log(N+e)}.
\]
Both this probability and every fixed-time error threshold tend to zero.
These formulas retain the complete fixed-horizon constants, rather than
only asserting the existence of a continuity modulus.
:::

(sec-slc-explicit-full-law-base)=
## 10. Full-law attraction for a nonconvex canonical parameter regime

:::{div} feynman-prose
A cloud remaining near a well tells us where its mass is concentrated. Full-law
relaxation asks a stronger question: how quickly does its entire position and
velocity distribution approach a stationary distribution? Here we can complete
that calculation for a specific canonical parameter regime, including forces
with several local wells.

Setting both fitness exponents to zero makes every fitness equal, so accepted
cloning edges disappear. The actual algorithm then evolves each walker through
its kinetic update independently. A specified relation between timestep and
friction cancels the linear position dependence in one part of that update.
Confinement and a Gaussian lower bound then give an explicit probability of
regeneration, from which the relaxation time follows. Every parameter choice
needed for this argument appears in the statements below.

This calculation supplies a fully evaluated case and a useful point of
comparison. With active selection, the population changes its own transition
mechanism; extending the same long-time conclusion requires controlling that
feedback. The preceding finite-horizon mean-field result already includes active
selection in its stated regimes.
:::

:::{prf:lemma} Kinetic minorization without global invertibility
:label: lem-slcp-surjective-minorization

Use the exact BAOAB, final position noise and radial cap of
{prf:ref}`def-baoab-update-rule`, with $c=h/2$, $a=e^{-\gamma h}$,
$q>0$, $s=\sigma_x\sqrt h>0$. Suppose

$$
F(x)=-\kappa x+f(x),\quad |f(x)|\le H,\quad
\operatorname{Lip}(F)\le L_F,\quad
\alpha_0=1-c^2\kappa>0.
$$

The input has $|x|\le R_0$ and $|v|\le V_c$. For arbitrary declared
$r,u>0$, define

$$
\begin{aligned}
R_1&=\alpha_0R_0+cV_c+c^2H,\\
m_v&=a(V_c+c\kappa R_0+cH),\\
Q&=(u+c\kappa R_1+cH)/\alpha_0,\\
k_v&=(2\pi q^2)^{-d/2}
 \exp[-(Q+m_v)^2/(2q^2)](1+c^2L_F)^{-d},\\
k_x&=(2\pi s^2)^{-d/2}
 \exp[-(r+R_1+cQ)^2/(2s^2)],\\
\epsilon&=v_d(u)v_d(r)k_vk_x.
\end{aligned}
$$

Here $v_d(t)=\pi^{d/2}t^d/\Gamma(1+d/2)$. Let $\nu$ be the
product of uniform position on $B_r$ and the cap-pushforward of uniform
pre-cap velocity on $B_u$. For the all-alive unbounded configuration,
the full kinetic kernel satisfies $K((x,v),\cdot)\ge\epsilon\nu$.
There is no requirement $c^2L_F<1$.
:::

:::{prf:proof}
Write $x_1=x+c(v+cF(x))$ and $v_2=a(v+cF(x))+q\xi$.
Then $|x_1|\le R_1$, and the Gaussian mean of $v_2$ has norm at most
$m_v$. The pre-cap velocity is

$$
T(v_2)=v_2+cF(x_1+cv_2)
 =\alpha_0v_2-c\kappa x_1+cf(x_1+cv_2).
$$

For every $w\in B_u$, the continuous map

$$
v\longmapsto [w+c\kappa x_1-cf(x_1+cv)]/\alpha_0
$$

maps the closed ball $\overline B_Q$ into itself. Brouwer's fixed-point
theorem supplies a preimage $T(v)=w$ with $|v|\le Q$. In fact every
preimage of $w\in B_u$ obeys this bound by the displayed equation.
The map $T$ is Lipschitz with constant at most $1+c^2L_F$, so its
absolute Jacobian is at most $(1+c^2L_F)^d$ almost everywhere.
The area formula for Lipschitz maps applies here in equal domain and
target dimension; see [Lang, Theorem 8.3](https://people.math.ethz.ch/~lang/rect_notes.pdf).
It implies that the image of the set where this Jacobian
vanishes has Lebesgue measure zero: its multiplicity integral is zero.
The same is true for the null set where a Lipschitz map is not differentiable.
Consequently almost every $w\in B_u$ has a regular preimage in
$\overline B_Q$. Apply its weighted form on the regular set with weight equal to the
Gaussian density divided by $|\det DT|$, first truncated and then by
monotone convergence. The resulting sum over these preimages bounds
the density of $T(v_2)$ below by $k_v$ on $B_u$; extra preimages and any
singular part only increase the measure.

Conditionally on any such preimage $v_2$, final position has Gaussian
mean $x_1+cv_2$ and covariance $s^2I$. Its density on $B_r$ is at least
$k_x$. Applying the same area formula with this nonnegative conditional
density as a weight gives the joint lower density $k_vk_x$ on
$B_r\times B_u$. Applying the deterministic radial cap to velocity
preserves measure domination and proves the claim. In particular
$0<\epsilon\le1$, since both sides are probability measures.
:::

:::{prf:theorem} Explicit full-law relaxation with an arbitrary bounded nonconvex force perturbation
:label: thm-slcp-full-law-nonconvex-base

Consider the canonical all-alive unbounded algorithm with no viscosity,
no history and fitness exponents $p_r=p_s=0$. Thus every live fitness
is exactly one, every live acceptance probability is zero, all collision
components are isolated, and recipient jitter is never activated. The
population map is exactly $\mathcal F_h(\mu)=\mu K$, where $K$ is the
actual one-row kinetic kernel, including both force kicks, both position
drifts, OU noise, final position noise and the radial velocity cap.

Suppose $F=-\kappa x+f$ with $\kappa>0$, $|f|\le H$ and
$\operatorname{Lip}(F)\le L_F<\infty$. Select any $a\in(0,1)$ and set

$$
h=\frac{2}{\sqrt{\kappa(1+a)}},\qquad
\gamma=-\frac{\log a}{h},\qquad
c=h/2,\quad B=c(1+a),\quad \eta=c^2(1+a)=\kappa^{-1}.
$$

Choose $b_O>0$, $\sigma_x>0$, $V_{\max}>0$, and put

$$
q=b_O\sqrt{\frac{1-a^2}{2\gamma}},\qquad
s=\sigma_x\sqrt h,\qquad
b=1+(BV_{\max}+\eta H)^2+d(c^2q^2+s^2).
$$

Let $\mathcal V(x,v)=1+|x|^2$, $R=4b$, and compute $\epsilon$ by
{prf:ref}`lem-slcp-surjective-minorization` with
$R_0=\sqrt{R-1}$, $V_c=V_{\max}$ and declared $r,u>0$. Define

$$
\beta=\frac{\epsilon}{R+2b},\qquad
\rho=\max\left\{1-\frac\epsilon2,
 \frac{2+2\beta b}{2+\beta R}\right\}<1,\qquad
A_\mu=1+\frac\beta2[\mu\mathcal V+b].
$$

There is exactly one invariant probability $\pi$, it satisfies
$\pi\mathcal V\le b$, and every initial law with finite $\mu\mathcal V$
obeys

$$
\boxed{\quad
\|\mathcal F_h^n(\mu)-\pi\|_{\rm TV}\le A_\mu\rho^n.
\quad}
$$

Therefore error at most $\delta\in(0,1)$ is guaranteed after

$$
n\ge\left\lceil
\frac{\log(A_\mu/\delta)}{-\log\rho}\right\rceil,
\qquad t=nh.
$$

All constants are independent of $N$. Noise, friction, time step, cap,
dimension and force parameters occur above. Companion bandwidths,
feature regularizers, fitness regularizers, cloning saturation,
collision strength and jitter amplitude do not affect this parameter
regime because its accepted-edge graph is empty.
:::

:::{prf:proof}
The zero exponents make the two positive fitness factors equal to one,
so the positive fitness difference in every acceptance gate is zero.
The rooted population construction therefore contains only its root;
no nonlinear approximation is made in reducing its update to $K$.
The actual resonance identity gives

$$
x^+=Bv+\eta f(x)+cq\xi+s\zeta.
$$

Independence and centering of the two Gaussian innovations imply

$$
K\mathcal V(x,v)
=1+|Bv+\eta f(x)|^2+d(c^2q^2+s^2)\le b.
$$

Also $\alpha_0=1-c^2\kappa=a/(1+a)>0$, so the preceding minorization
lemma applies without imposing convexity or smallness of $H$ or $L_F$.
On $\{\mathcal V\le R\}$ it gives the common probability $\nu$ and
coefficient $\epsilon$. These are the drift and minorization inputs of
the weighted-variation theorem in this chapter, with drift coefficient
zero. Its inequalities hold because $R=4b>2b$ and
$2\beta b\le\epsilon$. In detail, for two distinct states set
$d_\beta(z,z')=2+\beta[\mathcal V(z)+\mathcal V(z')]$.
If their total $\mathcal V$ exceeds $R$, independent outputs have
expected cost at most $2+2\beta b$, giving the second displayed
coefficient. If the total is at most $R$, both inputs belong to the
minorized set. Couple the common $\epsilon\nu$ parts identically;
the remaining expected cost is at most $2(1-\epsilon)+2\beta b$
which is at most $2-\epsilon$ and hence at most
$(1-\epsilon/2)d_\beta(z,z')$. Identical inputs use identical outputs.

Integrating these measurable couplings contracts the weighted total
variation norm with factor $\rho$. Completeness of probability laws with
finite $\mathcal V$ moment in this norm gives a unique fixed point.
Its moment is at most $b$ by invariance and the uniform moment bound.
The weighted initial difference is at most
$2+\beta[\mu\mathcal V+\pi\mathcal V]$, and unweighted TV is at most
half that norm. This proves the rate. Any invariant probability,
including one not initially assumed to have a finite second moment,
has moment at most $b$ by invariance and monotone convergence, so
uniqueness holds among all invariant probabilities.
:::

:::{prf:example} A genuinely multimodal force covered by full-law attraction
:label: ex-slcp-full-law-rastrigin

For

$$
U(x)=\frac\kappa2|x|^2+
 A\sum_{j=1}^d(1-\cos(2\pi x_j)),\qquad A\ge0,
$$

use $H=2\pi A\sqrt d$ and $L_F=\kappa+4\pi^2A$ in
{prf:ref}`thm-slcp-full-law-nonconvex-base`. These follow directly from
$F_j=-\kappa x_j-2\pi A\sin(2\pi x_j)$ and its diagonal Jacobian.
For $\kappa=A=1$ the one-dimensional derivative of $U$ is positive at
$x=1/2$, negative at $x=3/4$, and positive at $x=1$; hence there is
an additional local minimum in $(3/4,1)$ and its reflection, beside
the strict minimum at zero. The theorem therefore covers a landscape
with multiple local wells. Its stationary law is a full distribution
on the entire space; the theorem does not claim separate exact invariant
laws supported inside these spatial wells.
:::

:::{prf:remark} Scope of this evaluated full-law regime
:label: rem-slcp-full-law-scope

The preceding theorem proves complete-law attraction for the exact
population equation in a nonconvex, unbounded parameter regime. It
neither proves nor requires that active cloning has the same stationary
law. Turning on either fitness exponent changes the rooted collision
map. The signed perturbation bound of {prf:ref}`lem-slcc-selection-perturbation`
and the two-update theorem {prf:ref}`thm-slcc-active-contraction` now
supply a separate closed active-cloning regime, retaining all collision
components. Its bounded-reward and discrete-center hypotheses are essential.
The explicit finite-horizon population law remains valid whether or not
such a long-time estimate closes. A universal assertion that every
nonlinear stationary phase attracts all laws in an unspecified region
would require an independently defined population attraction region and
a proved dissipation or regeneration estimate there; a spatial basin
and positional second-moment decay do not imply that assertion.
:::

:::{prf:corollary} Global two-update regeneration and arbitrary-initial-law relaxation
:label: cor-slcp-global-two-step

Under {prf:ref}`thm-slcp-full-law-nonconvex-base`, use its explicit
$b,R=4b,\epsilon,\nu$ and put $\varepsilon_2=3\epsilon/4$. Then

$$
K^2(z,\cdot)\ge\varepsilon_2\nu(\cdot)
\quad\hbox{for every physical input }z,
$$

and, for every probability $\mu$ without a moment assumption,

$$
\|\mu K^n-\pi\|_{\rm TV}
\le(1-\varepsilon_2)^{\lfloor n/2\rfloor}.
$$

A sufficient iteration count for error $\delta\in(0,1)$ is

$$
n=2\left\lceil\frac{\log(1/\delta)}
 {-\log(1-\varepsilon_2)}\right\rceil,
\qquad t=nh.
$$
:::

:::{prf:proof}
The uniform moment bound gives
$K(z,\{\mathcal V>R\})\le b/R=1/4$ by Markov's inequality.
Integrating the small-set minorization over the first transition yields
$K^2(z,A)\ge(3/4)\epsilon\nu(A)$. Write
$K^2=\varepsilon_2\nu+(1-\varepsilon_2)\widetilde K$;
the residual is a Markov kernel. Integration against a signed measure
of total mass zero and the total-variation contraction of a Markov
kernel give
$\|\mu K^2-\eta K^2\|_{\rm TV}
\le(1-\varepsilon_2)\|\mu-\eta\|_{\rm TV}$.
Since the TV distance between probabilities is at most one, iteration
with $\eta=\pi$, followed by the optional last Markov transition,
proves the formula. Equivalently, completeness in TV first gives a
unique invariant law for $K^2$; its image under $K$ is also invariant
for $K^2$, so uniqueness makes it invariant for $K$.
:::

:::{prf:corollary} Uniform-time empirical mean-field approximation in the evaluated regime
:label: cor-slcp-uniform-iid-mean-field

Keep the same exact constant-fitness canonical parameter regime, and
initialize the $N$ rows independently with common law $\mu_0$. Use the
metric

$$
\mathsf d(\mu,\eta)=\inf_{\lambda\in\Pi(\mu,\eta)}
 \int\min\{1,|z-z'|\}\,\lambda(dz,dz').
$$

At every time $n$, the rows are independent with common law
$\mu_n=\mu_0K^n$. For $R_x>0$, $0<\ell\le1$, define

$$
\begin{aligned}
J(R_x,\ell)&=
\left(1+\left\lceil\frac{2R_x\sqrt{2d}}\ell\right\rceil\right)^d
\left(1+\left\lceil\frac{2V_{\max}\sqrt{2d}}\ell\right\rceil\right)^d,\\
a_N(R_x,\ell)&=
\min\left\{1,2\ell+\frac{J(R_x,\ell)}{2\sqrt N}
 +\frac{2b}{R_x^2}\right\}.
\end{aligned}
$$

Then, without an initial moment assumption,

$$
\boxed{\quad
\sup_{n\ge1}\mathbb E\mathsf d(L_N(S_n),\mu_n)
\le a_N(R_x,\ell).
\quad}
$$

For every $\tau>0$ and integer $T\ge1$,

$$
\begin{aligned}
\sup_{n\ge1}\Pr\{\mathsf d(L_N(S_n),\mu_n)>a_N+\tau\}
 &\le e^{-2N\tau^2},\\
\Pr\{\max_{1\le n\le T}\mathsf d(L_N(S_n),\mu_n)>a_N+\tau\}
 &\le\min\{1,T e^{-2N\tau^2}\}.
\end{aligned}
$$

In particular choose $R_x=N^{1/(16d)}$, $\ell=N^{-1/(16d)}$.
The displayed finite formula then tends to zero with $N$, uniformly
in the deterministic observation time $n\ge1$. Moreover

$$
\mathbb E\mathsf d(L_N(S_n),\pi)
\le a_N+(1-\varepsilon_2)^{\lfloor n/2\rfloor}.
$$

The finite-$N$ chain has unique stationary law $\pi^{\otimes N}$.
Its empirical stationary laws obey the same $a_N$ bound; its $k$-row
stationary marginal equals $\pi^{\otimes k}$ exactly for every $N\ge k$.
Thus both stationary and long-time mean-field conclusions hold in this
regime, with explicit constants independent of $N$.
:::

:::{prf:proof}
An empty accepted-edge graph creates no sharing of component rotations,
no copying and no activated clone jitter. Each row therefore uses only
its own state and independent kinetic innovations. Product-law evolution
and exact independence follow by induction, including when the initial
law lacks moments. For every $n\ge1$, the uniform one-step moment bound
implies $\mu_n|x|^2\le b$.

Partition the box
$[-R_x,R_x]^d\times[-V_{\max},V_{\max}]^d$ into at most
$J(R_x,\ell)$ measurable cells of diameter at most $\ell$.
Match the common empirical and target mass within each cell; unmatched
mass costs at most one. The expected absolute discrepancy of a cell
mass is at most $1/\sqrt N$, since its empirical variance is at most
$1/N$. Exterior empirical and target masses each have expectation at
most $b/R_x^2$. Summing the cell discrepancies with the factor $1/2$
from total variation, allowing $2\ell$ for the within-cell displacement,
gives the claimed $a_N$ uniformly over $n\ge1$.

Changing one row changes $\mathsf d(L_N(S_n),\mu_n)$ by at most $1/N$,
by the triangle inequality and coupling all unchanged atoms identically.
Reveal the independent rows successively. The Doob martingale differences
of this functional have conditional ranges of length at most $1/N$.
For a centered random variable in an interval of length $c$, convexity
of the exponential (or its two-point extremal bound) gives
$\mathbb E e^{tY}\le e^{t^2c^2/8}$.
Iterated conditioning therefore bounds the centered functional's
moment-generating function by $e^{t^2/(8N)}$.
Exponential Markov inequality and the choice $t=4N\tau$ give
$e^{-2N\tau^2}$. The bound using $a_N$ follows from the preceding
expectation estimate. A union bound proves the finite-horizon statement
without asserting independence between times.

The triangle inequality and $\mathsf d\le\|\cdot\|_{\rm TV}$ give
the bound relative to $\pi$. The product kernel has invariant law
$\pi^{\otimes N}$. Its two-update minorization is
$(K^2)^{\otimes N}\ge\varepsilon_2^N\nu^{\otimes N}$,
which proves uniqueness by the same elementary contraction argument.
Applying the empirical bound directly to independent samples from
$\pi$, whose moment is at most $b$, proves the stationary assertion.
Taking $n\to\infty$ and $N\to\infty$ in either order in the displayed
expectation bound gives zero. This exchange concerns the empirical
weak metric and fixed-size marginals, not TV distance between an atomic
empirical measure and a continuous law.
:::

(sec-slch-stopped-phase-transitions)=
## 11. Explicit competing transitions and validity horizons

:::{div} feynman-prose
Suppose the swarm occupies one well and we want it to establish itself in
another. There are competing outcomes: reaching the target, remaining in the
controlled region, and leaving that region somewhere else. A lower bound on
target discovery becomes useful only when we also account for those competing
exits. The formulas below do this by conditioning on the complete swarm history.
They preserve the dependence created by cloning.

Once establishment occurs, a separate retention estimate bounds residence for
a declared duration. This makes the exploration budget explicit: how many
updates, how much physical time, and how many walker updates support the
claimed probability? The Rastrigin example gives a very small certified
probability for direct collective transfer. Its size exposes the conservatism
of that event; it does not calculate every possible route through intermediate
populations. Finally, a positive escape hazard can coexist with extremely long
residence. Eventual departure along a sample path and accuracy of the evolving
mean-field distribution are distinct statements, and each needs its own bound.
:::

:::{prf:theorem} Target establishment before uncontrolled exit
:label: thm-slch-competing-events

Use the full canonical conservative kernel $P_N$ of
{prf:ref}`def-slc-parameter-register`, or its absorbing extension with extinction
included in the failure set. Let $\mathcal G,\mathcal B$ be disjoint measurable
population sets and let

$$
\tau=\inf\{n\ge1:S_n\in\mathcal B\},\qquad
\sigma=\inf\{n\ge1:S_n\notin\mathcal G\cup\mathcal B\}.
$$

Assume $S_0\in\mathcal G$. For every $S\in\mathcal G$ suppose the *actual
complete-update probabilities* satisfy

$$
P_N(S,\mathcal B)\ge p>0,\qquad
P_N(S,\mathcal G^c)\le u\le1.
$$

Necessarily $p\le u$. Define
$H_n(u)=\sum_{j=0}^{n-1}(1-u)^j$, with $H_0(u)=0$. Then

$$
\Pr(\tau\le n,\tau<\sigma)\ge pH_n(u)
=\frac p u[1-(1-u)^n],
\qquad
\Pr(\tau\wedge\sigma>n)\le(1-p)^n.
$$

If the separate failure bound $P_N(S,(\mathcal G\cup\mathcal B)^c)\le f$
holds, then also

$$
\Pr(\sigma\le n,\sigma<\tau)\le\min\{1,fH_n(p)\},
$$

and consequently

$$
\Pr(\tau\le n,\tau<\sigma)
\ge \max\{pH_n(u),\ 1-(1-p)^n-fH_n(p)\}.
$$

All displayed bounds retain dependence on shared cloning variables and on the
entire preceding history. They require no Markov property of basin labels.

:::

:::{prf:proof}
Put $A_j=\{\tau\wedge\sigma>j\}$. Since the three output sets are
pairwise disjoint and exhaustive, on $A_j$ the next-step probability of $A_{j+1}$
is between $1-u$ and $1-p$. Induction gives
$(1-u)^j\le\Pr(A_j)\le(1-p)^j$. The mutually disjoint target-first events at
steps $j+1$ have probabilities
$\mathbb E[\mathbf1_{A_j}P_N(S_j,\mathcal B)]\ge p(1-u)^j$.
Summation proves the first bound. Replacing target by failure bounds each
failure-first probability by $f(1-p)^j$. Finally target first, failure first,
and $A_n$ partition the sample space, which gives the alternative lower bound.

:::

:::{prf:corollary} Complete transfer and residence at a random establishment time
:label: cor-slch-stopping-composition

In {prf:ref}`thm-slch-competing-events`, suppose every state in $\mathcal B$
has next-step probability at least $1-v$ of remaining in $\mathcal B$.
For integers $n,\ell\ge0$,

$$
\Pr\{\tau\le n,\tau<\sigma,\ S_{\tau+j}\in\mathcal B
       \text{ for }0\le j\le\ell\}
\ge pH_n(u)(1-v)^\ell.
$$

The completed residence event is observable by update $n+\ell$, whose physical
time is $(n+\ell)h$ and walker-update count is $N(n+\ell)$. Any computational
cost per walker update must additionally charge companion selection and the
actual component construction; this count is not a claim of linear runtime.

:::

:::{prf:proof}
On each event $\{\tau=k<\sigma\}$, which is measurable at the stopping
time $\tau$, iterated conditional expectation gives conditional residence
probability at least $(1-v)^\ell$. Sum over $1\le k\le n$ and use the preceding
theorem. No event concerning future residence was used to select the stopping
time.
:::

:::{prf:corollary} Evaluated Rastrigin transfer and residence budgets
:label: cor-slch-rastrigin-budget

Use exactly the parameters in {prf:ref}`ex-slc-rastrigin-residence`. Let $Q_i,Q_j$
be distinct cores whose centers satisfy
$|z_{j,l}-z_{i,l}|<1$ in each coordinate that changes, and are equal otherwise.
This includes transfer from the origin core to a neighboring minimum core.
Set

$$
\bar u=\min\{1,4Nd e^{-122}\},\qquad
p_*=(1-2e^{-122})^d(1/8)^d(2\pi\tau^2)^{-d/2}
     \exp\left[-\frac{d(27/25)^2}{2\tau^2}\right],
\qquad p=p_*^N,
$$

where $\tau^2=(1+h^2/4)10^{-6}$ and $h$ has the exact expression in that
example. Let $\mathcal G=Q_i^N\times\{|v_l|\le V_{\max}\}_{l=1}^N$,
$\mathcal B=Q_j^N\times\{|v_l|\le V_{\max}\}_{l=1}^N$, with the usual
full-state marks included. Then

$$
\Pr\{\text{complete transfer to }Q_j\text{ before another exit, by }n,
       \text{ followed by }\ell\text{ updates entirely in }Q_j\}
\ge p_*^N H_n(\bar u)(1-\bar u)^\ell.
$$

For $d=1,N=128,n=\ell=10^6$, a simpler, very conservative expression is

$$
\Pr\{\text{the displayed event}\}
>10^6 e^{-74649600}(1-12\cdot10^{-45}).
$$

For discovery only, take $\mathcal B=\{Y_{Q_j}\ge1\}$ instead. Its target
hazard is at least $1-(1-p_*)^N$, so the target-before-other-exit lower bound is

$$
[1-(1-p_*)^N]H_n(\bar u).
$$

More generally $Y_{Q_j}\ge K$ uses the explicit binomial tail
$\sum_{l=K}^N{N\choose l}p_*^l(1-p_*)^{N-l}$. Residence in $Q_j^N$ may be
appended directly only when $K=N$; a count threshold $K<N$ needs a residence
estimate for that different population set.

:::

:::{prf:proof}
The existing example proves $H_i<7/400$, $R=1/16$, and gives the
per-coordinate jitter cutoff probability at least $1-2e^{-122}$. Thus
$|z_{j,l}-z_{i,l}|+R+H_i<1+1/16+7/400=27/25$.
Substitute into {prf:ref}`thm-slc-evaluated-basins` to obtain $p_*$. The same
theorem gives complete transfer probability at least $p_*^N$, arbitrary count
thresholds by binomial domination, and total exit probability at most
$\bar u$ from either core. Apply the two preceding results.

For the simplified one-dimensional bound, $10^{-6}\le\tau^2<401/(400\cdot
10^6)$. The prefactor
$(1-2e^{-122})(1/8)(2\pi\tau^2)^{-1/2}$ exceeds one: $\pi<22/7$ makes the
Gaussian denominator less than $1/300$, while $1-2e^{-122}>1/2$.
Hence $p_*>e^{-583200}$, because $(27/25)^2/(2\cdot10^{-6})=583200$.
The verified residence estimate gives $10^6\bar u<6\cdot10^{-45}$.
Bernoulli's inequality yields
$H_n(\bar u)\ge n(1-(n-1)\bar u)$ and
$(1-\bar u)^\ell\ge1-\ell\bar u$; multiply, drop their nonnegative product
term, and use $583200\cdot128=74649600$. The very small lower bound is the
value certified by these parameters and this direct all-row transition event;
it does not estimate the faster path that repeated selection might provide.

:::

:::{prf:proposition} What a positive transition hazard excludes
:label: prop-slch-pathwise-horizon

For the conservative kernel in the preceding corollary, starting in
$\mathcal G=Q_i^N\times\{|v_l|\le V_{\max}\}_{l=1}^N$, let
$\sigma_i=\inf\{n\ge1:S_n\notin\mathcal G\}$. Then

$$
(1-\bar u)^n\le\Pr(\sigma_i>n)\le(1-p_*^N)^n,
\qquad \Pr(\sigma_i<\infty)=1,
\qquad \mathbb E\sigma_i\le p_*^{-N}.
$$

These estimates concern pathwise residence. They do not imply failure of the
finite-horizon mean-field limit, failure of uniform marginal approximation,
or nonuniqueness of the finite-particle invariant law.

More generally use the bounded-Lipschitz metric on phase space with Euclidean
product distance, normalized as
$d_{\rm BL}(\nu,\mu)=\sup\{|\nu f-\mu f|:\|f\|_\infty\le1,
\operatorname{Lip}(f)\le1\}$. Let

$$
\Delta=\min\{1,\operatorname{dist}(Q_i,Q_j)\}>0,\qquad
\phi(x,v)=\min\{\Delta,\operatorname{dist}(x,Q_i)\}.
$$

For $\mu_n=\mathcal F_h^n(\mu_0)$ suppose the *evolving* population laws obey
$\mu_n\phi\le b\Delta$ over the horizon. If the target stopping event gives
$Y_{Q_j}(S_\tau)/N\ge a>b$, then

$$
\max_{k\le n}d_{\rm BL}(L_N(S_k),\mu_k)\ge\Delta(a-b)
\quad\text{on }\{\tau\le n\}.
$$

A target-before-failure probability from the preceding theorem is consequently
an explicit lower bound on the probability of exceeding that pathwise
threshold. The additional condition on $\mu_n\phi$ must itself be verified;
finite-particle residence does not establish it. For the neighboring Rastrigin
cores in dimension one, $\Delta>99/100-1/8=173/200$.

:::

:::{prf:proof}
The retention lower bound was proved above. From every state of
$\mathcal G$, the complete transfer event exits $\mathcal G$ with probability
at least $p_*^N$. Conditional iteration therefore bounds the survival tail above
by $(1-p_*^N)^n$. Taking $n\to\infty$ proves almost-sure exit and summing the
tail proves the expectation bound. The function $\phi$ is bounded by one, is 1-Lipschitz, is nonnegative,
and equals $\Delta$ throughout $Q_j$. At the target time its empirical integral
is at least $a\Delta$, whereas the deterministic law gives at most $b\Delta$.
It is an admissible test function in $d_{\rm BL}$, proving the assertion.
Neither argument identifies the law after departure or compares it with the
simultaneously evolving mean-field trajectory.
:::

:::{prf:proposition} Quantified phase memory from interval transition data
:label: prop-slch-phase-memory

Let $\Gamma$ be a finite measurable partition of full population states into
$m$ labels, including a declared uncontrolled label. Suppose computed bounds
$\ell_{ij}\le P_N(S,\Gamma=j)\le u_{ij}$ hold for every state in label $i$.
Choose any explicit stochastic row $T_i$ satisfying
$\ell_{ij}\le T_{ij}\le u_{ij}$; if no such row exists the supplied bounds are
inconsistent. Such a row can be computed by starting at $\ell_i$ and assigning
remaining mass to coordinates in their fixed order, up to capacities
$u_{ij}-\ell_{ij}$. Define

$$
\varepsilon_i=\min\left\{1,\frac12\sum_{j=1}^m
\max(T_{ij}-\ell_{ij},u_{ij}-T_{ij})\right\},\qquad
\varepsilon=\max_i\varepsilon_i.
$$

For every conditional history ending in label $i$, the next label distribution
is within $\varepsilon_i$ in TV of $T_i$. The law of the complete label path
through update $n$ differs from the corresponding Markov-chain path law by at
most $\min(1,n\varepsilon)$. Hence every path-dependent arrival, residence or
transfer event has probability error bounded by this same number. If the bounds
are certified only until exit from a controlled class, add the proved probability
of that exit by update $n$.

:::

:::{prf:proof}
Conditional on a label history, the current full-state law is a mixture
of states in that label. Averaging preserves each coordinate interval. For any
row $p$ in these intervals,
$\frac12\sum_j|p_j-T_{ij}|\le\varepsilon_i$ by the stated formula (and TV is
always at most one). Sequential maximal coupling of the actual conditional
label laws with the chosen Markov rows fails in at most $n$ trials, each with
conditional failure probability at most $\varepsilon$. The union bound controls
the path-law TV and therefore every measurable path event. Stop the construction
at localization exit to obtain the last assertion. This is a bound on memory
error, not an assertion of exact lumpability.
:::

(slcd-decision-bounds)=
## 12. Quantitative decisions at a declared resolution and confidence

:::{div} feynman-prose
Specify the question before computing a convergence bound. We may want to know
whether today's swarm approximates today's mean-field law, whether that law is
within a chosen tolerance of a candidate phase, or whether the entire sequence
of laws settles as time tends to infinity. These questions have different error
budgets. A swarm can accurately follow an evolving law that has not settled.

The estimates below turn regional masses, geometric separation and tail bounds
into upper and lower limits on the relevant distance. An upper limit below the
requested tolerance certifies accuracy; a lower limit above it certifies failure
at the specified time. If the interval straddles the tolerance, the calculation
is inconclusive at that resolution. Total variation also needs control within
regions: matching their masses alone can conceal different distributions inside
them. Every conclusion therefore retains its metric, horizon and confidence.
:::

:::{prf:definition} Quantities being compared
:label: def-slcd-comparison

Use the complete capped single-row state space
$\mathbb R^d\times\overline B_{V_{\max}}$, the cost
$c_0(z,z')=\min\{1,|z-z'|\}$, and its transport metric $\mathsf d$.
Let $\mu_n=\mathcal F_h^n\mu_0$ and $L_{N,n}$ denote respectively the
population evolution and the actual empirical swarm law. Fix a specified
comparison probability $\pi$; calling it a stationary phase additionally
requires the proved identity $\mathcal F_h\pi=\pi$.
The trajectory theorems supply explicitly computed numbers $v_n$ and
$\alpha_T$ with
$$
 \Pr\{\mathsf d(L_{N,n},\mu_n)\le v_n\text{ for every }n\le T\}
       \ge1-\alpha_T.
$$
These numbers contain the landscape profiles, algorithm parameters,
population size, initial moments and horizon; no independence of the
empirical rows is required below.
:::

:::{prf:theorem} Finite-resolution upper bounds with charged tails
:label: thm-slcd-upper

Partition a declared bounded state region into measurable cells
$C_1,\ldots,C_J$ of Euclidean diameter at most $\ell\le1$ and let
$E$ be its complement. Write $m_j=\rho(C_j)$ and $p_j=\pi(C_j)$.
Then
$$
 \mathsf d(\rho,\pi)\le
 \min\{1,\ell+\tfrac12\sum_{j=1}^J|m_j-p_j|
                     +\tfrac12[\rho(E)+\pi(E)]\}.
$$
Suppose certified intervals contain these masses:
$m_j\in[m_j^-,m_j^+]$, $p_j\in[p_j^-,p_j^+]$, and
$\rho(E)\le t_\rho$, $\pi(E)\le t_\pi$. Set
$$
 U(\rho,\pi)=\min\left\{1,\ell+
 \tfrac12\sum_{j=1}^J
 \max\{|m_j^--p_j^+|,|m_j^+-p_j^-|\}
                         +\tfrac12(t_\rho+t_\pi)\right\}.
$$
This is an explicit upper bound on $\mathsf d(\rho,\pi)$.
If the union of the interior cells contains
$\overline B_R\times\overline B_{V_{\max}}$, so its exterior
is contained in $\{|x|>R\}$, admissible tail budgets include $t_\rho=M_{8,\rho}/R^8$, or
$t_\rho=M_{\psi,\rho}/\psi(R)$ under the proved coercive-tail envelope,
and the corresponding formulas for $\pi$, capped by one. Sharper sums of
regional tail budgets may be substituted.

Apply this construction to $\rho=L_{N,n}$: empirical cell masses and
empirical tails are exact counts divided by $N$. If the target mass
intervals hold simultaneously with failure probability at most
$\alpha_\pi$, then
$$
 \Pr\{\mathsf d(\mu_n,\pi)\le
           \min\{1,U(L_{N,n},\pi)+v_n\}\ \forall n\le T\}
                \ge1-\alpha_T-\alpha_\pi.
$$
No sampling interpretation is imposed on deterministic interval-integration
bounds, for which $\alpha_\pi=0$.
:::

:::{prf:proof}
Match mass $\min(m_j,p_j)$ inside each cell, with cost at most $\ell$
per unit mass. The unmatched total mass is
$$
 1-\sum_j\min(m_j,p_j)
       =\tfrac12\sum_j|m_j-p_j|+\tfrac12[\rho(E)+\pi(E)].
$$
It can be coupled arbitrarily at cost at most one. This proves the first
bound, after enlarging the matched-mass cost to $\ell$.
The largest absolute difference between numbers in two closed intervals
occurs at an endpoint pair, proving the interval formula. The tail
budgets follow from their stated moment inequalities. Finally apply the
triangle inequality on the simultaneous trajectory event and intersect
with the event of valid target intervals. The union bound needs no
independence between these events.
:::

:::{prf:theorem} Lower bounds from separated regions and witness intervals
:label: thm-slcd-lower

Let $f$ be a measurable function with
$|f(z)-f(z')|\le c_0(z,z')$. For any two probabilities,
$$
 \mathsf d(\rho,\pi)\ge|\rho f-\pi f|.
$$
If certified intervals give $\rho f\in[a,b]$ and $\pi f\in[c,d]$, the
computable lower bound is
$$
 L_f(\rho,\pi)=\max\{0,a-d,c-b\}.
$$
A geometrically explicit family of witnesses is
$$
 f_{A,s}(z)=\min\{s,\operatorname{dist}(z,A)\},\qquad 0<s\le1,
$$
for a nonempty measurable set $A$, where distance means distance to its
closure. If $\rho(\operatorname{dist}(z,A)\ge s)\ge q$ and
$\pi(A)\ge1-b_A$, then
$$
 \mathsf d(\rho,\pi)\ge s(q-b_A)_+.
$$
The constants $s$ and $b_A$ expose geometric separation and target leakage
rather than identifying different cell labels with a positive distance.

On the trajectory event, any such empirical lower bound yields
$$
 \mathsf d(\mu_n,\pi)\ge
                  [L_f(L_{N,n},\pi)-v_n]_+.
$$
Consequently, a strictly positive lower endpoint above the requested
accuracy certifies failure to reach that accuracy at the specified time,
with the same simultaneous confidence as the upper bounds.
:::

:::{prf:proof}
For every coupling $\Gamma$,
$|\rho f-\pi f|\le\int|f(z)-f(z')|\,d\Gamma
\le\int c_0\,d\Gamma$. Taking the infimum proves the first inequality.
The minimum absolute separation of two intervals is the displayed
positive-part formula. Distance to a set is 1-Lipschitz; truncation at
$s\le1$ also bounds the oscillation by one, so $f_{A,s}$ is admissible.
It has $\rho f_{A,s}\ge sq$ and $\pi f_{A,s}\le s\pi(A^c)\le sb_A$.
The final claim follows from the triangle inequality, or from the same
witness inequality for $L_{N,n}$ and $\mu_n$.
:::

:::{prf:proposition} Boundary-layer uncertainty in region probabilities
:label: prop-slcd-boundary

A weak-metric trajectory bound is not an indicator bound. Explicitly,
if $\mathsf d(\rho,\nu)\le v$, $0<s\le1$, and
$A^s=\{z:\operatorname{dist}(z,A)<s\}$, then
$$
 \rho(A)\le\nu(A^s)+v/s,\qquad
 \nu(A)\le\rho(A^s)+v/s.
$$
Define $A_{-s}=\{z:\operatorname{dist}(z,A^c)\ge s\}$. Then
$$
 \nu(A_{-s})-v/s\le\rho(A)\le\nu(A^s)+v/s.
$$
Thus replacing an empirical region mass by a population region mass costs
both $v/s$ and the explicitly bounded boundary-layer mass. For cells in
a partition, the same inequalities apply cell by cell with their actual
geometry. Merely dividing by $\sqrt N$ does not remove these terms.
:::

:::{prf:proof}
For a coupling of cost at most $v+\varepsilon$, the probability of
$|z-z'|\ge s$ is at most $(v+\varepsilon)/s$. Outside that event,
$z\in A$ implies $z'\in A^s$. This proves the first inequality after
$\varepsilon\downarrow0$; interchange the measures for the second.
Also $z'\in A_{-s}$ and $|z-z'|<s$ imply $z\in A$, proving the lower
bound. No regularity of the cell boundary is assumed; it appears in the
size of the layer rather than being silently discarded.
:::

:::{prf:proposition} What a partition does and does not prove in TV
:label: prop-slcd-tv

Use $\|\rho-\pi\|_{\rm TV}=\sup_A|\rho(A)-\pi(A)|$.
For a finite measurable partition including its exterior cell,
$$
 \|\rho-\pi\|_{\rm TV}\ge\tfrac12\sum_j
                                  |\rho(C_j)-\pi(C_j)|.
$$
Mass intervals give the rigorous lower bound
$$
 \tfrac12\sum_j\max\{0,m_j^--p_j^+,p_j^--m_j^+\}.
$$
An upper bound requires additional information within cells. Specifically,
if for every cell with both masses positive the normalized restrictions
satisfy the separately proved estimate
$\|\rho(\cdot\mid C_j)-\pi(\cdot\mid C_j)\|_{\rm TV}\le\epsilon_j$,
then
$$
 \|\rho-\pi\|_{\rm TV}\le
 \tfrac12\sum_j|m_j-p_j|+\sum_j\min(m_j,p_j)\epsilon_j.
$$
A cell with zero mass in either measure contributes zero to the second
sum. In the absence of a proved conditional estimate, $\epsilon_j=1$
is valid and can make the upper bound uninformative. A spatial mesh alone
does not establish TV convergence of an atomic empirical measure to a
continuous population law.
:::

:::{prf:proof}
Choose the union of cells where $m_j\ge p_j$; its signed mass difference
is half the sum of all absolute differences, proving the lower bound.
The interval bound follows term by term. For the upper bound, match the
common cell mass $\min(m_j,p_j)$ using a maximal coupling of the two
conditional probabilities, which fails with probability at most
$\epsilon_j$. Couple remaining cell mass arbitrarily. Its total is
$\tfrac12\sum_j|m_j-p_j|$. The resulting mismatch probability bounds
TV by the displayed expression.
:::

:::{prf:theorem} Finite windows, asymptotic convergence and explicit obstructions
:label: thm-slcd-decisions

The following conclusions distinguish finite-time accuracy from an
asymptotic claim.

1. If the computed upper bound $U(L_{N,n},\pi)+v_n$ is at most a
   declared tolerance $\varepsilon$, then the population law at iteration
   $n$ is within $\varepsilon$ of $\pi$ on the certified event. Its
   physical time is $nh$; evaluating $T+1$ complete empirical populations
   uses $N(T+1)$ recorded row states, with the actual algorithmic update
   cost charged separately.
2. For $m,n\le T$, any empirical witness lower bound for
   $\mathsf d(L_{N,m},L_{N,n})$, reduced by $v_m+v_n$, bounds
   $\mathsf d(\mu_m,\mu_n)$ below. Their computable distance upper bound,
   enlarged by $v_m+v_n$, bounds it above. Alternatively, comparison to
   a common phase gives the upper bound
   $U(L_{N,m},\pi)+U(L_{N,n},\pi)+v_m+v_n$.
3. Suppose structural estimates prove, for deterministic $T_k\uparrow
   \infty$ and numerical $\varepsilon_k\downarrow0$,
   $\mathsf d(\mu_m,\mu_n)\le\varepsilon_k$ for every $m,n\ge T_k$.
   Then $\mu_n$ converges to a probability $\mu_\infty$. If the proved
   population-map continuity estimate applies along this sequence and
   at its limit, then $\mathcal F_h\mu_\infty=\mu_\infty$.
4. Conversely, if one fixed admissible witness $f$ and deterministic
   subsequences $n_k,m_k\to\infty$ have certified integral intervals
   separated by at least $\varepsilon_*>0$, then $\mu_n$ has no weak
   limit. A finite list of separated observations proves only the
   corresponding finite-window failure.
5. If deterministic $n_k\to\infty$, $R_k\to\infty$ and
   $\varepsilon_*>0$ satisfy
   $\mu_{n_k}(|x|>R_k)\ge\varepsilon_*$ for every $k$, the family of
   laws is not tight and $\mu_n$ cannot converge weakly to a probability.
   The same obstruction for the annealed particle laws follows if
   $$
   \Pr\{L_{N,n_k}(|x|>R_k)\ge\varepsilon_*\}\ge p_*>0
   \quad\text{for every }k.
   $$
   These inequalities concern deterministic times. Almost-sure eventual
   visits beyond every radius at random hitting times do not imply this
   non-tightness condition.

An infinite sequence of statistical certificates can support these
conclusions on a common event when its failure probabilities are
summable: the common-event probability is at least one minus their sum.
This does not convert finitely many observed windows into an unproved
infinite-horizon certificate.
:::

:::{prf:proof}
Items 1 and 2 follow from the proved bounds and the triangle inequality.
For item 3, the displayed estimate is the Cauchy criterion in the complete
transport space over the complete bounded metric $c_0$. Here is a direct
justification of the needed completeness. From a Cauchy sequence choose
indices $j_k$ so successive transport distances are at most $2^{-2k}$.
Glue couplings of these successive laws, each of expected cost at most
$2^{-2k}+2^{-3k}$. Markov's inequality makes the probabilities of a
successive distance exceeding $2^{-k}$ summable. On the resulting
probability space, the sampled states are almost surely Cauchy in $c_0$
and hence converge to a state in the capped complete state space.
Bounded convergence gives transport convergence of this subsequence to
the law of its limit; the original Cauchy property gives convergence of
the full sequence. Apply the proved continuity to
$\mu_{n+1}=\mathcal F_h\mu_n$ to obtain stationarity.

For item 4, the witness inequality bounds the distance between the two
subsequences below by $\varepsilon_*$. Equivalently, the bounded
continuous witness integrals cannot converge. For item 5, if the family
were tight, a compact set would have complement mass less than
$\varepsilon_*/2$ uniformly in $n$. Its spatial projection is bounded,
so for sufficiently large $k$ its complement contains $\{|x|>R_k\}$,
a contradiction. To rule out weak convergence directly, fix $R$ and use the bounded
continuous witness $f_R(z)=\min\{1,(|x|-R)_+\}$. For sufficiently large
$k$, $R_k>R+1$, so $\mu_{n_k}f_R\ge\varepsilon_*$. A putative weak
limit $\mu_\infty$ would satisfy $\mu_\infty f_R\ge\varepsilon_*$
for every $R$, contradicting bounded convergence as $R\to\infty$. For the particle version, expectation
of the stated event gives
$\mathbb E L_{N,n_k}(|x|>R_k)\ge p_*\varepsilon_*$, and the same
argument applies to the annealed laws. Finally a countable union bound
proves the common-event confidence assertion.
:::

(sec-slcpd-phase-decision)=
### 12.1. Full-law phase convergence certified along one nonlinear trajectory

:::{div} feynman-prose
Follow one initial population law and compare each update with the preceding
one. The accumulated distance of these successive laws bounds how far the
trajectory can still travel. If we prove a summable bound on all future
increments, its remaining sum gives an explicit distance to a limiting law.
Continuity of the actual population map then identifies that limit as stationary.
This argument permits another initial population to approach a different phase.

The kinetic density and finite rooted integrals below make individual increment
bounds accessible, with truncation and integration errors charged explicitly.
The essential long-time step is to control their whole tail. A finite sequence
of small increments does not establish that control. Conversely, a positive
lower bound recurring at infinitely many updates rules out settling in total
variation. Bounds that merely approach zero, without a summable tail or another
Cauchy estimate, leave the asymptotic question open.
:::

:::{prf:definition} Full-law increment and declared phase class
:label: def-slcpd-increment

Let $\mu_{n+1}=\mathcal F_h(\mu_n)$ be the actual fixed-step all-alive
population equation, with its active fitness, copying, rooted collision
components, jitter, BAOAB, final position noise and cap. Define

$$
r_n=\|\mu_{n+1}-\mu_n\|_{\rm TV}.
$$

A declared population phase class $\mathfrak P$ is a TV-closed subset
of probability laws. Examples include simultaneous constraints
$\mu(B_i)\ge m_i$, $\mu(T_j)\le t_j$ and $\mu(E)\le e$ on the declared
basin, transition and exterior sets. These are conditions on population
laws; a spatial basin alone is not a population phase.

The sufficient criterion below estimates successive laws along this one
trajectory. It does not require contraction between two differently
initialized populations.
:::

:::{prf:lemma} Exact kinetic density in coordinates before the final cap
:label: lem-slcpd-density

Assume $q>0$, $s>0$, $F\in C^1(\mathbb R^d;\mathbb R^d)$ and
$\sup_x\|DF(x)\|\le L_F$ with $\ell_T=c^2L_F<1$.
The potential need not be convex. For the prepared root position and
collision velocity $(x,v)$, set

$$
x_1=x+c[v+cF(x)],\quad m=a[v+cF(x)],\quad
T_{x_1}(z)=z+cF(x_1+cz),\quad z_w=T_{x_1}^{-1}(w).
$$

In coordinates $(y,w)$ consisting of final position and pre-cap velocity,
the actual conditional kinetic density is

$$
k_\theta(y,w\mid x,v)=
\frac{\exp[-|z_w-m|^2/(2q^2)]}{(2\pi q^2)^{d/2}}
\frac{\exp[-|y-x_1-cz_w|^2/(2s^2)]}{(2\pi s^2)^{d/2}}
\frac1{|\det[I+c^2DF(x_1+cz_w)]|}.
$$

Its parameters are exactly $c=h/2$, $a=e^{-\gamma h}$,
$q^2=b_O^2(1-e^{-2\gamma h})/(2\gamma)$ (with its $\gamma=0$
limit), and $s^2=\sigma_x^2h$. The inverse $z_w$ can be evaluated by
$z^{(j+1)}=w-cF(x_1+cz^{(j)})$, with rigorous error

$$
|z^{(j)}-z_w|\le
\frac{\ell_T^j}{1-\ell_T}|z^{(1)}-z^{(0)}|.
$$

If $\mathcal R_\mu$ is the actual prepared-root law after copying,
jitter and collision, the density of $\mathcal F_h(\mu)$ in these
coordinates is $f_\mu(y,w)=\int k_\theta(y,w\mid x,v)\,
\mathcal R_\mu(dx,dv)$.
:::

:::{prf:proof}
For fixed $w$, the displayed inverse iteration has Lipschitz constant
$\ell_T<1$ on the complete space $\mathbb R^d$. Its successive
increments are bounded by a geometric series, proving existence,
uniqueness and the displayed error. Also
$|T(z)-T(z')|\ge(1-\ell_T)|z-z'|$; its derivative is everywhere
invertible. The inverse-function theorem and the global bijection make
$T$ a $C^1$ diffeomorphism. Conditional on preparation, $v_2$ has
Gaussian density with mean $m$, covariance $q^2I$, and pre-cap velocity
is exactly $T(v_2)$. Conditional on $v_2=z_w$, final position has the
second displayed Gaussian density, independently of the OU innovation.
Change variables from $v_2$ to $w$ to obtain $k_\theta$.

The cap is the bijection $C_V(w)=Vw/(V+|w|)$ from $\mathbb R^d$ to
the open ball $B_V$, with inverse $Vv/(V-|v|)$. Applying this common
bijection to both measures preserves TV. Thus TV of two full physical
output laws equals half the $L^1(dy\,dw)$ distance of their displayed
pre-cap-coordinate densities; no cap Jacobian is omitted from a
physical-coordinate calculation. Integrating over the exact prepared
root law proves the final identity.
:::

:::{prf:lemma} Finite rooted integrals with a rigorous residual error bar
:label: lem-slcpd-rooted-residual

Keep the preceding regime, including fully active cloning. The Gaussian
companion lower bound is
$\kappa_C=\exp[-D_*^2/(2\epsilon_C^2)]$, where
$D_*=2\sqrt{R_x^2+\lambda R_v^2}$. Put

$$
C=2/\kappa_C,\qquad M_1=e^{2C}.
$$

Let $\mathcal R_\mu^{[K]}$ be the subprobability prepared-root law
restricted to actual collision components of at most $K\ge1$ vertices,
and set

$$
f_\mu^{[K]}(y,w)=
\int k_\theta(y,w\mid x,v)\,\mathcal R_\mu^{[K]}(dx,dv),\qquad
I_n^{[K]}=\frac12\int
|f_{\mu_n}^{[K]}-f_{\mu_{n-1}}^{[K]}|\,dy\,dw,
\quad n\ge1.
$$

The rooted component moment bound gives

$$
\boxed{\quad
\max\{0,I_n^{[K]}-M_1/K\}
\le r_n\le\min\{1,I_n^{[K]}+M_1/K\}.
\quad}
$$

For each fixed $K$, the two subprobability laws are specified by a finite
sum over rooted directed-tree shapes with at most $K$ vertices, finite-
dimensional integrals of their input and measurement types, the actual
edge weights and acceptance gates, incoming Poisson probabilities, shared
component Haar mark, and Gaussian jitter. Every density and normalizer
is the one in the defined population map. This is a finite-dimensional
analytic integration problem; it is not an independence approximation.

If rigorous integration supplies
$|\widehat I_n^{[K]}-I_n^{[K]}|\le e_n^{[K]}$, then the fully numerical
upper bound is

$$
\overline r_n=
\min\{1,\widehat I_n^{[K]}+e_n^{[K]}+M_1/K\},
$$

and the lower bound is the positive part of
$\widehat I_n^{[K]}-e_n^{[K]}-M_1/K$.
The integration tolerance is a separately supplied, verified error
bound; an unvalidated numerical estimate cannot substitute for it.
:::

:::{prf:proof}
The complete component is finite almost surely and has expected size
at most $M_1$ by the actual ordered-forest estimate. Therefore its
omitted probability is at most $M_1/K$ by Markov's inequality.
The nonnegative kinetic kernel integrates to one, so
$\|f_\mu-f_\mu^{[K]}\|_{L^1}$ equals that omitted mass.
The elementary inequality
$|\|u\|_1-\|v\|_1|\le\|u-v\|_1$, applied to the difference of
the two densities, gives an error at most half the sum of the omitted
masses, hence at most $M_1/K$. Since
$\mu_{n+1}=\mathcal F_h\mu_n$ and
$\mu_n=\mathcal F_h\mu_{n-1}$, the exact half-$L^1$ difference is
$r_n$. This proves both bounds. The component restriction is decided
by exploring at most $K$ vertices and checking for termination. Incoming
no-further-child probabilities are part of that exploration; dropping
them would not define the same subprobability law. The finitely many
possible tree shapes and their finite mark integrals give the stated
representation. The last inequalities follow by the triangle inequality
for the certified integration error.
:::

:::{prf:theorem} A summable-increment certificate for full-law attraction to a population phase
:label: thm-slcpd-phase-attraction

Let the actual population trajectory lie in one of the quantitative
continuity regimes established in this chapter: either bounded Lipschitz
reward, or the stated quadratic-growth reward with a uniform eighth-
moment bound $\sup_n\mu_n|x|^8\le H<\infty$. The full map then has
its already proved explicit weak-metric continuity modulus on this
class. Suppose $n_0\ge1$ and a declared numerical sequence
$(R_n)_{n\ge n_0}$ satisfies

$$
r_n\le R_n\quad(n\ge n_0),\qquad
\sum_{n=n_0}^\infty R_n<\infty.
$$

One directly checkable sufficient inequality is
$I_n^{[K_n]}+M_1/K_n\le R_n$ from the preceding lemma, with its
certified integration error included when needed. These inequalities
must hold for the whole asserted tail; a finite measured prefix does
not by itself establish them.

Then there exists a full-state fixed law $\pi_{\mu_0}$ such that

$$
\mathcal F_h(\pi_{\mu_0})=\pi_{\mu_0},\qquad
\boxed{\quad
\|\mu_n-\pi_{\mu_0}\|_{\rm TV}
\le\sum_{j=n}^\infty R_j\quad(n\ge n_0).
\quad}
$$

If $\mu_n\in\mathfrak P$ for all $n\ge n_0$, the limit belongs to
that declared phase class. Different initial laws may have different
limits. No assertion of contraction between them is required.

In particular:

- If $R_n=Dq^{n-n_0}$ with specified $0<D<\infty$, $0<q<1$, the error
  is at most $Dq^{n-n_0}/(1-q)$. A sufficient index for error $\varepsilon$
  is
  $n=n_0+\max\{0,\lceil\log[D/(\varepsilon(1-q))]/(-\log q)\rceil\}$.
- If $R_n=D(n+1)^{-1-\alpha}$ with specified $0<D<\infty$, $\alpha>0$,
  the error is at most $Dn^{-\alpha}/\alpha$ for $n\ge\max\{1,n_0\}$.
  Thus $n\ge\max\{n_0,1,\lceil[D/(\alpha\varepsilon)]^{1/\alpha}\rceil\}$
  suffices.

Physical time is $nh$. The numbers $D,q,\alpha$ in these alternatives
are proposed scalar envelopes checked against the displayed operator
integrals; they are not defined as unknown optimal convergence constants.
:::

:::{prf:proof}
For $m>n\ge n_0$, the triangle inequality gives
$\|\mu_m-\mu_n\|_{\rm TV}\le\sum_{j=n}^{m-1}R_j$.
Hence $(\mu_n)$ is Cauchy in TV. Completeness of finite signed measures
in total variation and preservation of nonnegativity and total mass
under TV limits give a probability limit $\pi_{\mu_0}$.
Letting $m\to\infty$ gives the claimed tail bound.

TV convergence implies convergence in the bounded Wasserstein metric
used in the chapter. In the unbounded-reward case, lower
semicontinuity gives $\pi_{\mu_0}|x|^8\le H$, so the proved common
moment-class continuity modulus applies to the pair
$(\mu_n,\pi_{\mu_0})$. In the bounded-reward case its global modulus
applies directly. Thus
$\mathcal F_h\mu_n\to\mathcal F_h\pi_{\mu_0}$ in that metric.
But $\mathcal F_h\mu_n=\mu_{n+1}\to\pi_{\mu_0}$; separation of
probability laws by the metric proves the fixed-point identity.
TV-closedness of $\mathfrak P$ proves phase membership. Sum the
geometric series for the first rate. For the second, comparison with
$\int_n^\infty x^{-1-\alpha}\,dx=n^{-\alpha}/\alpha$ gives the
bound and its inversion.
:::

:::{prf:corollary} A quantitative obstruction to full-law settling
:label: cor-slcpd-nonsettling

The certified lower bounds of {prf:ref}`lem-slcpd-rooted-residual` also
provide negative conclusions. If they exceed a fixed $a>0$ at
infinitely many indices, the trajectory cannot converge in TV to any
law. At each such index, for every candidate law $\pi$,

$$
\max\{\|\mu_n-\pi\|_{\rm TV},
        \|\mu_{n+1}-\pi\|_{\rm TV}\}\ge a/2.
$$

If the upper bounds tend to zero but their sum has not been controlled,
these certificates alone decide neither convergence nor nonconvergence.
This case is distinguished from a proved positive lower obstruction.
:::

:::{prf:proof}
The triangle inequality gives
$r_n\le\|\mu_n-\pi\|_{\rm TV}+\|\mu_{n+1}-\pi\|_{\rm TV}$,
which proves the stated lower bound. Every TV-convergent sequence has
$r_n\to0$ by the same inequality, contradicting the infinitely many
certified lower bounds. Summability was used to obtain the Cauchy
property in the preceding theorem; merely tending to zero does not
supply that step.
:::

:::{prf:corollary} Explicit truncation and integration budget for a proposed convergence rate
:label: cor-slcpd-error-allocation

For any proposed positive residual envelope $R_n$, select

$$
K_n=\max\{1,\lceil3M_1/R_n\rceil\}.
$$

A verified integration error $e_n^{[K_n]}\le R_n/3$ and a computed
integral estimate $\widehat I_n^{[K_n]}\le R_n/3$ then imply $r_n\le R_n$.
For the geometric and polynomial envelopes in
{prf:ref}`thm-slcpd-phase-attraction`, the respective explicit cutoffs are

$$
K_n=\max\left\{1,\left\lceil
 \frac{3e^{2C}}Dq^{-(n-n_0)}\right\rceil\right\},\qquad
K_n=\max\left\{1,\left\lceil
 \frac{3e^{2C}}D(n+1)^{1+\alpha}\right\rceil\right\}.
$$

These formulas expose the cost of using the conservative component-size
tail. For $T$ consecutive geometric certificates, the sum of component
cutoffs is at most

$$
T+\frac{3e^{2C}}D\frac{q^{-T}-1}{q^{-1}-1}.
$$

A cutoff counts permitted component vertices, not total arithmetic work:
the number of tree shapes and the cost of validated integration can grow
much faster. No computational efficiency is inferred from this bound.
:::

:::{prf:proof}
The cutoff choice gives $M_1/K_n\le R_n/3$. Adding truncation,
integration and computed-integral bounds gives $r_n\le R_n$ by
{prf:ref}`lem-slcpd-rooted-residual`. The two formulas follow by
substitution. For a positive number $x$, $\max\{1,\lceil x\rceil\}
\le1+x$. Sum this inequality over the geometric cutoffs and sum the
finite geometric series to obtain the displayed total.
:::

:::{prf:theorem} Phase-local dissipation of the actual nonlinear increment
:label: thm-slcpd-local-dissipation

Let $\mathfrak G$ be a declared class of population laws lying in one
of the proved continuity regimes of
{prf:ref}`thm-slcpd-phase-attraction`. In the unbounded-reward case,
assume the class has a uniform eighth-moment bound. Suppose its
invariance $\mathcal F_h(\mathfrak G)\subset\mathfrak G$ has been
verified from the actual update and the structural estimates.
Define the actual one-trajectory residual

$$
\mathscr R(\nu)=
\|\mathcal F_h^2(\nu)-\mathcal F_h(\nu)\|_{\rm TV}.
$$

Under the density conditions of {prf:ref}`lem-slcpd-density`, compute

$$
\mathscr I^{[K]}(\nu)=\frac12\int
|f_{\mathcal F_h(\nu)}^{[K]}-f_\nu^{[K]}|\,dy\,dw.
$$

With verified integration error $e^{[K]}(\nu)$, put

$$
\begin{aligned}
\underline{\mathscr R}(\nu)&=
[\widehat{\mathscr I}^{[K]}(\nu)-e^{[K]}(\nu)-M_1/K]_+,\\
\overline{\mathscr R}(\nu)&=
\min\{1,\widehat{\mathscr I}^{[K]}(\nu)+e^{[K]}(\nu)+M_1/K\}.
\end{aligned}
$$

Different cutoffs and integration tolerances may be used at different
laws; the resulting bounds must be valid there. Choose a declared
number $0<q_*<1$. The following structural integral inequality is a
sufficient dissipation certificate:

$$
\boxed{\qquad
\overline{\mathscr R}(\mathcal F_h\nu)
 \le q_*\underline{\mathscr R}(\nu)
\quad\text{for every }\nu\in\mathfrak G
\text{ with }\mathscr R(\nu)>0.
\qquad}
$$

At laws with zero residual, the exact identity
$\mathcal F_h^2\nu=\mathcal F_h\nu$ supplies the zero-residual branch;
there is no requirement that finite truncation alone recognize an exact
zero. Then every initial law $\mu_0\in\mathfrak G$ has a full-state
stationary limit $\pi_{\mu_0}$ and

$$
\|\mathcal F_h^n\mu_0-\pi_{\mu_0}\|_{\rm TV}
\le\frac{q_*^{n-1}}{1-q_*}\mathscr R(\mu_0)
\le\frac{q_*^{n-1}}{1-q_*}\overline{\mathscr R}(\mu_0),
\qquad n\ge1.
$$

For a TV-closed phase class containing the trajectory, the limit lies in
that class. The condition does not compare two separate initial laws
and allows different limits in different phase classes. The number
$q_*$ is a trial scalar verified against the displayed actual-operator
integrals, not an unknown optimal contraction coefficient. A finite set
of test laws does not prove the condition on $\mathfrak G$; a uniform
analytic inequality or validated enclosure over that class is required.
:::

:::{prf:proof}
The residual interval follows from exactly the rooted truncation and
integration-error argument already proved. Thus the assumed integral
inequality gives

$$
\mathscr R(\mathcal F_h\nu)
\le\overline{\mathscr R}(\mathcal F_h\nu)
\le q_*\underline{\mathscr R}(\nu)
\le q_*\mathscr R(\nu).
$$

If $\mathscr R(\nu)=0$, then $\mathcal F_h\nu$ is already fixed,
so its next residual is zero and the same inequality holds. Invariance
of $\mathfrak G$ permits iteration. For $\mu_n=\mathcal F_h^n\mu_0$
we have $\|\mu_{n+1}-\mu_n\|_{\rm TV}=\mathscr R(\mu_{n-1})$
for $n\ge1$, hence this increment is at most
$q_*^{n-1}\mathscr R(\mu_0)$. Sum this geometric envelope and apply
{prf:ref}`thm-slcpd-phase-attraction` to obtain the fixed point and rate.
Its proof also gives membership in any TV-closed phase class containing
the trajectory. No assertion concerning
$\|\mathcal F_h\mu-\mathcal F_h\nu\|$ for distinct initial laws was
used.
:::

:::{prf:proposition} Quantified residual dissipation defects and finite-window motion
:label: prop-slcpd-dissipation-defect

Along one actual population trajectory in a continuity regime of
{prf:ref}`thm-slcpd-phase-attraction` (with its uniform moment bound
when required), suppose verified bounds give

$$
r_{n+1}\le q_*r_n+e_n,\qquad n\ge n_0,\qquad
0<q_*<1,\quad e_n\ge0.
$$

Then

$$
r_n\le q_*^{n-n_0}r_{n_0}
 +\sum_{j=n_0}^{n-1}q_*^{n-1-j}e_j.
$$

For every integer window length $T\ge1$,

$$
\|\mu_{n+T}-\mu_n\|_{\rm TV}
\le\frac{1-q_*^T}{1-q_*}r_n
 +\sum_{j=n}^{n+T-2}
 \frac{1-q_*^{n+T-1-j}}{1-q_*}e_j,
$$

where the sum is empty for $T=1$. In particular, $e_j\le e_*$ yields

$$
\|\mu_{n+T}-\mu_n\|_{\rm TV}
\le\frac{1-q_*^T}{1-q_*}r_n+
\frac{e_*}{1-q_*}
\left[T-\frac{1-q_*^T}{1-q_*}\right].
$$

If $\sum_{j=n_0}^\infty e_j<\infty$, the increments are summable
and the same full-law convergence theorem applies, with

$$
\|\mu_n-\pi_{\mu_0}\|_{\rm TV}
\le\frac{r_n+\sum_{j=n}^\infty e_j}{1-q_*}.
$$

A persistent positive defect bound gives only the finite-window motion
estimate and $\limsup_n r_n\le e_* /(1-q_*)$. It does not by itself
prove a stationary law or a stationary-error floor.
:::

:::{prf:proof}
Iteration proves the first geometric convolution by induction. Sum the
corresponding bound for $r_{n+k}$ over $k=0,\ldots,T-1$ and use
$\|\mu_{n+T}-\mu_n\|\le\sum_{k=0}^{T-1}r_{n+k}$.
Reversing the two finite sums gives the displayed window weights.
Their sum for constant $e_*$ is
$[T-(1-q_*^T)/(1-q_*)]/(1-q_*)$, proving that formula.
When the defects are summable, sum the same nonnegative convolution
over all future indices; each defect contributes at most
$e_j/(1-q_*)$, and the initial increment contributes $r_n/(1-q_*)$.
This proves summability and the limit error. The constant-defect
limsup follows from the first convolution. It bounds increments only;
without summability it supplies no Cauchy conclusion, which proves the
stated limitation.
:::

(sec-slcd-escape)=
### 12.2. Escape and the information needed for a global decision

:::{div} feynman-prose
Tail information is part of a global conclusion. A cloud can appear settled
inside the region we inspect while probability moves elsewhere. To certify
escape, we need a structural estimate that persists along the actual update,
including the donor transfers that selection permits. The directional estimate
below compares outward motion with the opposing contributions of bounded
velocities, donor displacement and noise.

Its conclusion concerns mass beyond growing thresholds at specified times.
That is strong enough to rule out convergence to a probability law on the
original space. A rare excursion at some random hitting time does not establish
the same conclusion. If the structural inequalities hold only while the swarm
remains in a controlled class, the probability of leaving that class must enter
the bound. Finite observations can support finite-horizon decisions; an
infinite-horizon verdict additionally needs proved control of the unobserved
tail in space and time.
:::

:::{prf:theorem} A structural certificate of escape and failure of tightness
:label: thm-slcd-directional-escape

Use the actual conservative all-alive kernel, with finite positions, capped
velocities, and moments sufficient to define its population map. Fix a unit
vector $e$. Suppose the declared regional force and accepted-transfer profiles
supply, on all reachable states,

$$
e\cdot F(x)\ge f_e,\qquad
e\cdot(y-x)\ge-d_e
$$

for every possible accepted frozen donor transfer from $x$ to $y$, with
$d_e\ge0$. A rejected transfer has displacement zero. The second bound is
on the actual accepted-edge support, including all measurement marks; it is
not a statement about a donor drawn from an independently averaged fitness.
For example its explicit profile is the supremum of
$[e\cdot(x-y)]_+$ over the region pairs and measurement assignments where
the actual gate is positive. A pair is excluded when its upper donor fitness
band is no larger than its lower recipient fitness band. Infinite $d_e$
makes the following certificate unavailable.

Let

$$
a_e=\eta f_e-BV_c-d_e,\qquad
s_e^2=\sigma_J^2+c^2q^2+s^2.
$$

If $a_e>0$, every tagged particle of the finite swarm, and the root trajectory
with successive nonlinear transition kernels prescribed by
$\mu_{n+1}=\mathcal F_h\mu_n$, satisfy for $n\ge1,t>0$

$$
\Pr\{e\cdot X_n<L+na_e-t\}
\le\Pr\{e\cdot X_0<L\}
       +\exp[-t^2/(2ns_e^2)]
$$

when $s_e>0$. When $s_e=0$, the exponential term is replaced by zero.
In particular, with $t=na_e/2$, the moving threshold is
$L+na_e/2$ and the exponential is $\exp[-na_e^2/(8s_e^2)]$.
For finite $N$, writing
$p_{0,N}=\mathbb E L_N(S_0)\{e\cdot x<L\}$, one obtains

$$
\mathbb E L_N(S_n)\{e\cdot x<L+na_e/2\}
\le p_{0,N}+e^{-na_e^2/(8s_e^2)},
$$

and the probability that this fraction exceeds $\varepsilon>0$ is at most
$[p_{0,N}+e^{-na_e^2/(8s_e^2)}]/\varepsilon$, truncated at one.
The same mass inequality holds for $\mu_n$, with $p_{0,N}$ replaced by
$\mu_0\{e\cdot x<L\}$. Thus the laws escape every fixed compact set and
are not tight; they cannot converge weakly or in TV to a probability law
on the original phase space. The bound has physical time $nh$ and charges
$nN$ finite-particle updates.

If the two profile inequalities are proved only before a stopping time
$\sigma$, the same finite-horizon bounds acquire the additional term
$\Pr(\sigma\le n)$. An asymptotic escape claim then requires that failure
term to be controlled; a mere increasing moment is not substituted for it.
:::

:::{prf:proof}
Condition on the complete graph, collisions and acceptance marks before the
independent jitter and kinetic innovations. For a fixed row the exact position
identity and the assumed directional bounds give

$$
e\cdot X_{j+1}\ge e\cdot X_j+a_e+\xi_j,
\qquad
\xi_j=C_j\sigma_J e\cdot Z_j^J+cq\,e\cdot Z_j^O+s\,e\cdot Z_j^x.
$$

Here $C_j\in\{0,1\}$ is its frozen acceptance indicator. The force can
depend on its jittered position; its lower bound is pointwise, so this dependence
has already been accounted for in the inequality. Conditional on the preparation,
$\xi_j$ is centered Gaussian of variance
$C_j\sigma_J^2+c^2q^2+s^2\le s_e^2$. Consequently, conditional on the
preceding history,
$\mathbb E[e^{-\lambda\xi_j}\mid\mathcal H_j]
\le e^{\lambda^2s_e^2/2}$ for every $\lambda\ge0$.
Iterated conditioning gives the bound
$\mathbb E e^{-\lambda\sum_{j<n}\xi_j}
\le e^{n\lambda^2s_e^2/2}$, also conditionally on the initial row.
Exponential Markov inequality optimized at $\lambda=t/(ns_e^2)$ proves
the claimed lower-tail bound on the event $e\cdot X_0\ge L$. The noiseless
case follows directly from the pathwise inequality.

At the population level the root construction, conditional on its entering
type, defines a probability transition kernel depending on the deterministic
$\mu_j$. Iterating these kernels constructs a tagged process with marginal
$\mu_j$ at time $j$. The same conditional calculation applies; it requires
no independence of interacting finite-particle rows. Averaging the tagged-row
bounds proves the empirical mass estimate, and Markov's inequality proves its
probability form.

For any fixed compact set, its $e$ coordinate is bounded above. Fix $L$ first,
then let $n$ tend to infinity in the moving-threshold estimate. Its mass has
limsup at most the initial mass below $L$. Letting $L\to-\infty$ sends this
mass to zero for any probability initial law with finite coordinates. This
proves escape and non-tightness. For localization, retain the actual Gaussian innovation sum through all
$n$ updates; its exponential bound holds whether or not the profile conditions
fail. On $\{\sigma>n\}$ the pathwise increment inequalities still telescope.
The adverse displacement event there is contained in the corresponding Gaussian
sum event. Add $\Pr(\sigma\le n)$ for its complement.
:::

:::{prf:theorem} Finite trajectory information does not determine unobserved-tail behavior
:label: thm-slcfi-finite-information

Fix dimension $d\ge1$, particle count $N\ge1$, a finite observation
horizon $T\ge1$, and $0<\delta<1$. Use the actual conservative canonical
algorithm with fitness exponents $p_r=p_s=0$, no viscosity and no history,
and initial positions and velocities equal to zero. All other algorithm
parameters are held identical between the two landscapes below.
Choose $\kappa>0$, $a\in(0,1)$, and the actual resonant parameters
$$
 h=\frac{2}{\sqrt{\kappa(1+a)}},\quad
 \gamma=-\frac{\log a}{h},\quad c=h/2,\quad
 B=c(1+a),\quad\eta=c^2(1+a)=\kappa^{-1}.
$$
Take $b_O,\sigma_x,V_{\max}>0$, so
$$
 q=b_O\sqrt{\frac{1-a^2}{2\gamma}}>0,\qquad s=\sigma_x\sqrt h>0.
$$
Define the explicit observation-radius quantities
$$
 G=\sqrt{2d\log(4dNT/\delta)},\qquad
 D=BV_{\max}+(cq+s)G,\qquad R=TD+1.
$$
Let $\chi:[0,\infty)\to[0,1]$ be
$$
 \chi(t)=
 \begin{cases}
 0,&t\le1,\\
 3(t-1)^2-2(t-1)^3,&1<t<2,\\
 1,&t\ge2.
 \end{cases}
$$
Consider the two force fields
$$
 F_0(x)=0,\qquad F_1(x)=-\kappa\chi(|x|/R)x.
$$
Both are gradients of smooth enough radial reward landscapes, agree on
$B_R$, and obey explicit regional structural profiles. Write
$\mathbb P_i^{[0,T]}$ for the law of the complete swarm state trajectory
through iteration $T$ under $F_i$. Then
$$
 \boxed{\|\mathbb P_0^{[0,T]}-\mathbb P_1^{[0,T]}\|_{\rm TV}
                                  \le\delta.}
$$
Nevertheless their asymptotic laws differ categorically:

- Under $F_0$, for every measurable spatial set $A$ of finite volume,
  each particle and the population law satisfy
  $$
  \Pr\{X_n\in A\}\le |A|(2\pi ns^2)^{-d/2},\qquad n\ge1.
  $$
  In particular every compact set loses all mass and no invariant
  probability or weak probability limit exists.
- Under $F_1$, the full-law relaxation theorem
  {prf:ref}`thm-slcp-full-law-nonconvex-base` applies with
  $$
  H=2\kappa R,\qquad L_F=4\kappa,
  $$
  and gives its displayed explicit TV convergence rate to a unique
  invariant probability. Its constant
  $$
  b=1+(BV_{\max}+2R)^2+d(c^2q^2+s^2)
  $$
  and the theorem's minorization and rate formulas are fully specified
  by these parameters. The finite swarm consists of independent copies
  in this regime, so it also converges to the product invariant law;
  its TV error is at most $N$ times the one-row bound, capped by one.

Consequently any rule based only on this finite trajectory that must
answer either “converges to an invariant probability” or “escapes” has
maximum error over these two explicitly constructed landscapes at least
$(1-\delta)/2$. The two landscapes may depend on the finite observation
budget. This is an obstruction to a uniformly reliable binary decision
from finite trajectory information, not a claim that one fixed pair
remains indistinguishable for every horizon.
:::

:::{prf:proof}
**The two structural profiles.** The cubic satisfies
$0\le\chi\le1$ and $0\le\chi'\le3/2$, with derivative zero at its
endpoints. Thus $F_1$ is continuously differentiable, including at the
origin where it vanishes on a neighborhood. In its transition annulus,
$$
 DF_1(x)=-\kappa\left[\chi(|x|/R)I+
       \chi'(|x|/R)\frac{xx^\top}{R|x|}\right],
$$
whose norm is at most $\kappa(1+2\cdot3/2)=4\kappa$.
Outside that annulus the same bound holds directly. Integrating along
segments proves the global Lipschitz bound. Also
$$
 F_1(x)=-\kappa x+f(x),\qquad
 f(x)=\kappa[1-\chi(|x|/R)]x,\qquad |f(x)|\le2\kappa R.
$$
A potential is $U_1(x)=\kappa\int_0^{|x|}\chi(t/R)t\,dt$, so
$F_1=-\nabla U_1$; $U_0=0$ gives $F_0$. These potentials and forces
agree inside $B_R$. One may use reward $-U_i$; their fitness factors
are still exactly one because both exponents are zero. Raw reward
normalizers, if evaluated, are finite on every finite state. The
population laws used here have the requisite finite reward moments.

**Finite-prefix coupling.** Every acceptance probability is zero.
Therefore copying, component collisions and recipient jitter are inactive,
and every velocity entering an update obeys the cap. Couple the two
systems using exactly the same O-noise and final-position-noise vectors
for each row and update. For a standard Gaussian vector $Z$,
$$
 \Pr(|Z|>G)\le2d\exp[-G^2/(2d)],
$$
by the coordinate union bound. There are $2NT$ relevant innovation
vectors, so the probability that any exceeds $G$ is at most
$4dNT\exp[-G^2/(2d)]=\delta$.

On their common good event, prove equality of the two paths inductively.
If both forces have remained zero through update $j$, then
$$
 x_{j+1}=x_j+Bv_j+cq\xi_j+s\zeta_j,
 \qquad |x_{j+1}|\le |x_j|+D,
$$
so $|x_j|\le jD$. The first drift position is
$x_1=x_j+cv_j$ and the position at the second force evaluation is
$x_2=x_j+Bv_j+cq\xi_j$. Both have norm at most $(j+1)D$, because
$c\le B$ and all omitted terms in $D$ are nonnegative. For $j<T$,
these radii are at most $TD<R$, so both force evaluations really are
zero. The final cap and all states thus agree under the common noises.
This closes the induction and shows path equality with probability at
least $1-\delta$. For any event in trajectory space its two indicator
values differ only when the coupled trajectories differ, proving the
TV bound. The same coupling can include the unused measurement and donor
marks if they are recorded: on path equality their input states, rewards,
companion laws and random addresses can all be matched.

**Escape under the zero force.** Its exact position identity is
$$
 X_n=B\sum_{j=0}^{n-1}V_j+cq\sum_{j=0}^{n-1}\xi_j
                              +s\sum_{j=0}^{n-1}\zeta_j.
$$
The velocity recursion is $V_{j+1}=\operatorname{cap}(aV_j+q\xi_j)$,
so all velocities are independent of all final-position noises
$\zeta_j$. Conditional on the O-noises, the displayed position is
Gaussian with covariance $ns^2I$ and some mean. Its density is everywhere
at most $(2\pi ns^2)^{-d/2}$. Integration over $A$ and then over the
conditioning proves the claimed bound. The same argument holds for an
arbitrary initial law independent of fresh innovations, by also
conditioning on that initial state. Hence an invariant probability would
assign zero mass to every bounded spatial set, which is impossible.
The bounded continuous spatial witnesses supported on increasing balls
also rule out a weak probability limit. Since the kinetic rows are
independent and identical, the population map is this exact one-row
Markov evolution.

**Convergence under the modified tail.** The proved decomposition
$F_1=-\kappa x+f$, the bound $|f|\le H$, finite $L_F$, positive $q,s$,
and $\eta\kappa=1$ verify every hypothesis of
{prf:ref}`thm-slcp-full-law-nonconvex-base`. Its surjective-minorization
argument requires no smallness of $c^2L_F$; imposing the older
$1-c^2L_F>0$ condition here would be incorrect. Substituting $H=2\kappa R$
gives the stated $b$ and the fully numerical rate from that theorem.
For independent rows, couple each one-row law to its invariant law with
its TV mismatch probability; a union bound over the $N$ coordinates
gives the product-law bound.

**Decision error.** Let $A$ be the event that a proposed binary rule
answers “converges.” Its two error probabilities are
$\mathbb P_0^{[0,T]}(A)$ and $\mathbb P_1^{[0,T]}(A^c)$.
Their sum is
$$
 1+\mathbb P_0^{[0,T]}(A)-\mathbb P_1^{[0,T]}(A)
 \ge1-\delta.
$$
At least one is at least $(1-\delta)/2$. Randomized rules satisfy the
same argument after adjoining their independent randomization to the
observation. Thus a rigorous decision procedure needs a certified tail
profile or an explicit “not determined at this resolution and horizon”
result; finite trajectory agreement cannot supply the missing tail data.
:::

(sec-slcj-full-long-time)=
## 13. Full long-time and stationary mean-field limits

:::{div} feynman-prose
At long times, specify what should converge. One population trajectory may
approach a particular stationary phase. A collection of trajectories may approach
a set of phases. The stationary finite-swarm laws may instead approach a mixture
of population laws. Each is a meaningful mean-field conclusion, and each asks
for a different estimate.

A finite noisy swarm can change phase after a long residence. Taking infinite
population first can preserve a phase that those finite-population transitions
eventually leave. The two orders of limits can therefore disagree even though
the same mean-field law correctly describes every fixed time interval. The
results below address the long-time question directly: they identify conditions
for uniform-time control and stationary limits, and show when distinct phases
prevent a common deterministic limit. The spatial presence of several wells
alone does not prove that several stationary population phases exist.
:::

:::{prf:theorem} Geometry-derived moment budgets uniform in time and population size
:label: thm-slcj-uniform-moments

Use the full conservative canonical update and a real exponent $p\ge2$.
Write $W_p(S)=N^{-1}\sum_i|x_i|^p$ and
$W_p(\mu)=\int|x|^p\,d\mu$. The region-pair geometry of
{prf:ref}`thm-slcg-flux-coefficients` gives, for this observable,

$$
\mathbb E[W_p(S^{\rm copy})\mid S]
 \le(1-\chi)W_p(S)+b_{p,\rm sel}+E_p(S),
$$

where $\chi=\chi_{\rm in}-\delta_{\rm out}\le1$ and

$$
b_{p,\rm sel}=R_c^p\left(\chi_{\rm in}
                +\sum_{b\in\mathcal B}D_bm_b^+\right).
$$

Use the finite-$N$ count formulas for particle states and the integral
formulas for population laws. Deterministic constants in this theorem
are uniform envelopes over the asserted classes and all $N$ being
compared. Outside a class, include the actual nonnegative copying excess
in $E_p$; no empirical membership is inferred from invariance of a
population class.

Let $F_\lambda=F_{\rm geom}-\lambda x$ and
$|F_{\rm geom}(x)|\le g_0+g_1|x|$, with the optional coefficient
$\lambda\ge0$. Set

$$
\begin{aligned}
A_\lambda&=|1-\eta\lambda|+\eta g_1,\qquad
b_0=BV_c+\eta g_0,\qquad \tau^2=c^2q^2+s^2,\\
m_{d,p}&=2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)},\\
b_p^{\rm noise}&=b_0+(A_\lambda\sigma_J+\tau)m_{d,p}^{1/p}.
\end{aligned}
$$

For every $u>0$ define

$$
a_p=(1+u)^{p-1}A_\lambda^p,\qquad
r_p=a_p(1-\chi),\qquad
B_p=a_pb_{p,\rm sel}+(1+u^{-1})^{p-1}(b_p^{\rm noise})^p.
$$

Then $P_NW_p\le r_pW_p+B_p+a_pE_p$, and the identical inequality
holds for $W_p(\mathcal F_h\mu)$. If

$$
r_{p,0}=A_\lambda^p(1-\chi)<1,
$$

choose $u=[(1+r_{p,0})/(2r_{p,0})]^{1/(p-1)}-1$ when
$r_{p,0}>0$, giving $r_p=(1+r_{p,0})/2<1$; use $u=1$ when
$r_{p,0}=0$. In particular if, uniformly in $N,n$,

$$
\mathbb E W_p(S_0)\le M_{p,0},\qquad
\mathbb E E_p(S_n)\le\bar e_p,
$$

then

$$
\sup_{N,n}\mathbb EW_p(S_n)
\le M_p:=\max\left\{M_{p,0},\frac{B_p+a_p\bar e_p}{1-r_p}\right\}.
$$

The analogous deterministic defect bound gives the same population moment
budget. Every finite-$W_p$ stationary swarm law under which
$\mathbb E E_p\le\bar e_p$ satisfies the stationary version $(B_p+a_p\bar e_p)/(1-r_p)$. Existence of such a stationary law
is supplied separately by the full-kernel recurrence theorem; it is not
inferred merely by naming a stationary expectation.

For $\chi<1$ the sufficient trap interval for this $p$-moment is

$$
H_p=(1-\chi)^{-1/p}-\eta g_1>0,\qquad
\lambda\in[0,\infty)\cap
 \left((1-H_p)/\eta,(1+H_p)/\eta\right).
$$

For $p=8$, $m_{d,8}=d(d+2)(d+4)(d+6)$. Thus the higher moments used
by the unbounded-reward mean-field theorems have an explicit
landscape–algorithm condition of their own. Uniform envelopes across
candidate $\lambda$ values must be justified as in
{prf:ref}`cor-slco-uniform-profiles`.
:::

:::{prf:proof}
For an exterior recipient and a core donor, the decrement in $|x|^p$
is at least $|x|^p-R_c^p$. For every accepted pair its positive increment
is at most the donor's $p$th power. Repeating the negative and positive
flux sums with these two pointwise inequalities gives exactly the stated
copying coefficient and $b_{p,\rm sel}$. The proof uses only
monotonicity of $r\mapsto r^p$, not an identity special to squares.

Sample a uniform row in the conditional full update. Let $Y$ be its
frozen selected position, $J\in\{0,1\}$ its cloning indicator, and
$X=Y+J\sigma_JZ_J$ its prepared position. Its final position obeys

$$
|X^+|\le A_\lambda|X|+b_0+|cqZ_O+sZ_x|.
$$

All norms below are $L^p$ norms on this single joint probability space,
including the uniform row; no independence between different rows is
needed. The Gaussian radial integral in polar coordinates is
$\mathbb E|Z|^p=m_{d,p}$ by the substitution $t=r^2/2$ in the
Gamma integral. Minkowski's inequality, $J\le1$, and the Gaussian
law of $cqZ_O+sZ_x$ give

$$
\|X^+\|_p
 \le A_\lambda\|Y\|_p+b_p^{\rm noise}.
$$

For nonnegative $a,b$ and $u>0$, convexity of $x^p$, with weights
$1/(1+u)$ and $u/(1+u)$, gives
$(a+b)^p\le(1+u)^{p-1}a^p+(1+u^{-1})^{p-1}b^p$.
Apply this inequality and insert the copying moment bound. This proves
the full-update drift, including its exact defect multiplier. Conditional
integration gives the root-population version.

Iterating expectations gives
$r_p^nM_{p,0}+(B_p+a_p\bar e_p)(1-r_p^n)/(1-r_p)$,
which is at most $M_p$. The same computation at a finite stationary
moment proves the stationary bound. Direct substitution verifies the
choice of $u$. Taking a nonnegative $p$th root of
$A_\lambda^p(1-\chi)<1$ gives $|1-\eta\lambda|<H_p$,
which is precisely the displayed interval. The Gamma recurrence gives
the stated eighth moment.
:::

(sec-slcm-joint-population-limits)=
### 13.1. Joint population and long-time limits without synchronizing phases

:::{div} feynman-prose
Sample an entire swarm from its stationary law, then look at its empirical
population. This gives a probability distribution on population laws. Moment
confinement prevents that distribution from losing mass at infinity. The
one-step mean-field error then identifies any suitable limit as invariant under
the deterministic population map.

Invariance still permits motion among population laws. To conclude that the
limit is a mixture of stationary phases, we need an additional dissipation
estimate whose vanishing identifies fixed points. When that estimate is proved,
the mixture has a direct interpretation: first choose a phase according to its
weight, then sample walkers from that phase. Walkers share the chosen phase,
which can leave dependence after averaging over it. The phase weights reflect
the stationary finite-population dynamics; they need not retain the phase
selected by an earlier initialization. The formal estimates below keep this
extra fixed-point-support step separate from stationary invariance.
:::

:::{prf:definition} Population occupation laws and quantitative consistency input
:label: def-slcm-occupation

Let $L_n^N=L_N(S_n)$ be the empirical law of the actual finite-$N$
algorithm and let $F=\mathcal F_h$ be its actual fixed-step population
map. Use the bounded metric
$\mathsf d(\mu,\nu)=\inf_{\pi}\int\min\{1,|z-z'|\}\,d\pi$ on population
laws. Define their time-averaged distributions

$$
Q_{N,T}=\frac1T\sum_{n=0}^{T-1}\operatorname{Law}(L_n^N),
\qquad T\ge1.
$$

These are probability laws on the space of population laws, not averages
of walker positions. Suppose the proved one-step consistency and moment
bounds supply

$$
\sup_n\mathbb E \mathsf d(L_{n+1}^N,F(L_n^N))\le a_N.
$$

For the canonical scalar consistency constant $G=A+4B_*^2$
and a uniform bound $M$ on both relevant expected second moments, an
explicit choice, for any $R>0$, $0<\ell\le1$, is

$$
\begin{aligned}
J(R,\ell)&=
\left(1+\left\lceil\frac{2R\sqrt{2d}}\ell\right\rceil\right)^d
\left(1+\left\lceil\frac{2V_{\max}\sqrt{2d}}\ell\right\rceil\right)^d,\\
a_N&=\min\left\{1,2\ell+
 \frac{J(R,\ell)}2\sqrt{G/N}+\frac{2M}{R^2}\right\}.
\end{aligned}
$$

All constants in $G$ have the parameter formulas already given in this
chapter. The existing copying and kinetic drift supplies $M$ when its
geometric inequality and weighted defects close. For example,
$\mathbb EW_{n+1}\le r\mathbb EW_n+b+e_*$, $r<1$, gives
$M\le\max\{\mathbb EW_0,(b+e_*)/(1-r)\}$ when the same bound applies
to the conditional population update. Taking
$R=N^{1/(16d)}$, $\ell=N^{-1/(16d)}$ gives $a_N\to0$ with the
explicit ceiling formula. The uniform moment premise must be proved;
a finite-horizon moment bound growing with $n$ does not supply it.
{prf:ref}`thm-slcj-uniform-moments` supplies the required explicit
second- or eighth-moment budget when its geometry-driven copying
coefficient, optional-force condition and defect envelope close.
:::

:::{prf:theorem} Joint Cesaro limit and invariant population dynamics
:label: thm-slcm-joint-invariant

Under {prf:ref}`def-slcm-occupation`, let $\mathcal W_{\mathsf d}$ denote the
Wasserstein metric on probability laws of populations with cost $\mathsf d$.
Then, for every $N,T$,

$$
\boxed{\qquad
\mathcal W_{\mathsf d}(Q_{N,T},F_\#Q_{N,T})\le a_N+1/T.
\qquad}
$$

For a finite-$N$ stationary law with empirical pushforward $Q_N$, the
same argument gives $\mathcal W_{\mathsf d}(Q_N,F_\#Q_N)\le a_N$.
Equivalently, every real function $H$ on population laws with
$\operatorname{Lip}_{\mathsf d}(H)\le1$ satisfies

$$
|Q_{N,T}(H\circ F-H)|\le a_N+1/T,
\qquad |Q_N(H\circ F-H)|\le a_N.
$$

Suppose a nonnegative coercive lower-semicontinuous state function $\psi$ supplies
$\sup_{N,n}\mathbb E L_n^N\psi\le M_\psi$, and the proved population
map is weakly continuous on each compact sublevel
$\mathcal K_H=\{\nu:\nu\psi\le H\}$. Then the population laws are
tight in the weak topology. Every subsequential limit $Q$ along
$N_j,T_j\to\infty$ satisfies $F_\#Q=Q$. The same statement holds for
stationary $Q_{N_j}$ as $N_j\to\infty$.

For the bounded-reward force-Lipschitz regime, take $\psi=1+|x|^2$.
For quadratic-growth reward with the proved regional continuity modulus,
take $\psi=1+|x|^8$. The moment sublevels are compact in the weak
topology, and the chapter's explicit common moment-class modulus supplies
the required restricted continuity. Global weak continuity outside all
controlled moment sublevels is not needed. The stronger moment condition
in the unbounded-reward case retains the reward-normalization integrals.

An invariant law $Q$ of population dynamics need not yet be supported
on fixed population laws; periodic or other invariant population
behavior is not excluded by this theorem.
:::

:::{prf:proof}
Choose an index uniformly from $\{0,\ldots,T-1\}$ independently of the
process. Pair $F(L_n^N)$ with $L_{n+1}^N$ at that index. Their expected
cost is at most $a_N$, and their marginal laws are $F_\#Q_{N,T}$ and
$Q'_{N,T}=T^{-1}\sum_{n=1}^T\operatorname{Law}(L_n^N)$.
Couple the $T-1$ identical mixture terms of $Q'_{N,T}$ and $Q_{N,T}$
identically; the two endpoint terms have total weight $1/T$ and cost
at most one. Thus $\mathcal W_{\mathsf d}(Q'_{N,T},Q_{N,T})\le1/T$.
The triangle inequality proves the first assertion. At stationarity
$Q'_{N,T}=Q_N$, removing the endpoint error. A Lipschitz test changes
by at most the coupling cost, proving the test inequalities.

For tightness, Markov's inequality gives
$Q_{N,T}(\mathcal K_H^c)\le M_\psi/H$. Coercivity supplies uniformly
small spatial tails for laws in $\mathcal K_H$; capped velocities and
lower semicontinuity make that sublevel compact in the weak topology.
The displayed estimate therefore proves tightness on population space.
Any limit $Q$ satisfies $Q(\mathcal K_H)\ge1-M_\psi/H$ by the
closed-set inequality for weak convergence.

It remains to justify passing the nonlinear map through the limit despite
only restricted continuity. Shift a bounded $1$-Lipschitz test $H_0$ for $\mathsf d$ so
its range is in $[0,1]$, and put $g=H_0\circ F$ on $\mathcal K_H$.
This function is uniformly continuous there. For $m>0$, define on all
population laws

$$
g_m(\nu)=\min\{1,\max\{0,\inf_{\xi\in\mathcal K_H}
                   [g(\xi)+m \mathsf d(\nu,\xi)]\}\}.
$$

It is bounded and $m$-Lipschitz. Given $\varepsilon>0$, choose
$\delta>0$ so the restricted modulus is at most $\varepsilon$ at
$\delta$, and take $m\delta\ge1$. At $\nu\in\mathcal K_H$, the
choice $\xi=\nu$ shows $g_m\le g$; points with $\mathsf d(\nu,\xi)<\delta$
give a lower bound $g(\nu)-\varepsilon$, and the other points give
at least one. Thus $|g_m-g|\le\varepsilon$ on $\mathcal K_H$.
Integrals against $Q_{N,T}$ and $Q$ differ from those of $g_m$ by at
most $\varepsilon+M_\psi/H$ each. Pass to the limit for the bounded
continuous $g_m$, then let $\varepsilon\downarrow0$ and $H\to\infty$.
This proves convergence of the integrals of $H_0\circ F$.
The integrals of $H_0$ converge directly. The invariance residual tends
to zero, and bounded Lipschitz tests separate laws, giving $F_\#Q=Q$.
The stationary proof is identical. This also proves the same passage
principle for any bounded function continuous on each $\mathcal K_H$.

:::

:::{prf:lemma} A bounded path-length functional for a phase-resolved evolution
:label: lem-slcm-path-functional

Let $\mathfrak G$ be a declared measurable forward-invariant class on which the
actual map has the explicit continuity modulus

$$
\mathsf d(F\mu,F\nu)\le\Psi(\mathsf d(\mu,\nu)),\qquad
\Psi(0)=0,
$$

with $\Psi$ nondecreasing and continuous at zero. Suppose the proved
phase-local residual estimate gives, uniformly for $\nu\in\mathfrak G$,

$$
\|F^{n+1}\nu-F^n\nu\|_{\rm TV}\le Dq^{n-1},
\qquad n\ge1,\quad D<\infty,\quad0<q<1.
$$

Different $\nu$ may converge to different fixed laws. Define

$$
V(\nu)=\sum_{k=0}^\infty \mathsf d(F^{k+1}\nu,F^k\nu),\qquad
V_* =1+\frac D{1-q}.
$$

Then $0\le V\le V_*$ and

$$
V(\nu)-V(F\nu)=\mathsf d(\nu,F\nu).
$$

Write $\Psi_0(\delta)=\delta$ and
$\Psi_{k+1}(\delta)=\Psi(\Psi_k(\delta))$.
For every integer $K\ge1$ an explicit common continuity bound is

$$
\begin{aligned}
B_K(\delta)&=\delta+2\sum_{k=1}^{K-1}\Psi_k(\delta)+\Psi_K(\delta),\\
\Omega_K(\delta)&=\min\left\{V_*,B_K(\delta)+
 \frac{2Dq^{K-1}}{1-q}\right\},\\
|V(\mu)-V(\nu)|&\le\Omega_K(\mathsf d(\mu,\nu)).
\end{aligned}
$$

For $\Psi(\delta)=\min\{1,C\delta^\alpha\}$, $C\ge1$,
$0<\alpha<1$, one can bound its iterates explicitly by

$$
\Psi_k(\delta)\le
\min\left\{1,C^{(1-\alpha^k)/(1-\alpha)}\delta^{\alpha^k}\right\}.
$$

For $\alpha=1$ replace this expression by
$\min\{1,C^k\delta\}$. These formulas contain no comparison of the
limiting phases with one another.
:::

:::{prf:proof}
The first increment is at most one, and all subsequent increments are
bounded by the stated TV estimate because $\mathsf d\le\|\cdot\|_{\rm TV}$.
Sum the geometric series for $V_*$. The series after index $K-1$ has
tail at most $Dq^{K-1}/(1-q)$ for $K\ge1$.
Shifting an absolutely convergent series gives the exact telescoping
identity. For the first $K$ terms, the metric triangle inequality gives

$$
|\mathsf d(F^{k+1}\mu,F^k\mu)-\mathsf d(F^{k+1}\nu,F^k\nu)|
\le \mathsf d(F^{k+1}\mu,F^{k+1}\nu)+\mathsf d(F^k\mu,F^k\nu).
$$

Iterate $\Psi$ and sum these inequalities to obtain $B_K$; bound the
two remaining tails by twice their common bound. Since both values of
$V$ lie in $[0,V_*]$, truncating at $V_*$ preserves the upper bound.
First let $K$ grow and then $\delta$ decrease to prove continuity.
The power-modulus formula follows by induction from the recurrence
$C_{k+1}=CC_k^\alpha$, $C_0=1$.
:::

:::{prf:theorem} Quantitative fixed-phase support for stationary and joint occupation limits
:label: thm-slcm-fixed-support

Use {prf:ref}`def-slcm-occupation` and the forward-invariant class and
constants of {prf:ref}`lem-slcm-path-functional`. For the limiting conclusions,
retain the uniform nonnegative coercive-moment bound and continuity on its
compact sublevels required by {prf:ref}`thm-slcm-joint-invariant`. Let

$$
p_{N,T}=\frac1T\sum_{n=0}^{T-1}
 \Pr\{L_n^N\notin\mathfrak G\}.
$$

For $0<\tau\le1$ and $K\ge1$, the following bound holds:

$$
\boxed{\quad
\int \mathsf d(\nu,F\nu)\,Q_{N,T}(d\nu)
\le\frac{V_*}{T}+\Omega_K(\tau)
 +\frac{V_*a_N}{\tau}+(V_*+1)p_{N,T}.
\quad}
$$

At stationarity replace $Q_{N,T}$ by $Q_N$, omit $V_*/T$, and use
$p_N=Q_N(\mathfrak G^c)$. Thus, if the relevant population laws are
tight as above and their right-hand sides tend to zero, every limit
$Q$ is supported on the full-state fixed-point set
$\{\nu:F\nu=\nu\}$. The statement allows an arbitrary mixture of
stationary phases; it does not require that set to be a singleton.

For common $D,q,\Psi$ and $p_{N,T}\to0$, one may first choose $K$
large, then $\tau$ small, and finally let $N,T\to\infty$.
This proves the conclusion for every joint sequence for which
$a_N\to0$, $T\to\infty$ and $p_{N,T}\to0$.
A fully numerical vanishing choice when
$\Psi(\delta)=\min\{1,C\delta^\alpha\}$, $0<\alpha<1$, is

$$
\tau_N=\sqrt{a_N},\qquad
K_N=\max\left\{1,
\left\lfloor\frac{\log(1+\log(1/a_N))}{2|\log\alpha|}\right\rfloor
\right\}
$$

for $0<a_N<e^{-4}$. For $\alpha=1$ take
$K_N=\max\{1,\lfloor\sqrt{\log(1/a_N)}\rfloor\}$.
If $a_N=0$, use any $K\to\infty$, $\tau\to0$ instead.
:::

:::{prf:proof}
Extend $V$ by zero outside $\mathfrak G$, calling this bounded
measurable function $\overline V$. Put $\nu=L_n^N$,
$\eta=L_{n+1}^N$. On $\nu\in\mathfrak G$, forward invariance and the
exact telescoping identity give
$\mathsf d(\nu,F\nu)=\overline V(\nu)-V(F\nu)$.
Add and subtract $\overline V(\eta)$ and take expectations.
If $\nu\in\mathfrak G$ but $\eta\notin\mathfrak G$, the remainder
$\overline V(\eta)-V(F\nu)=-V(F\nu)$ is nonpositive.
If both are in the class and $\mathsf d(\eta,F\nu)\le\tau$, it is at most
$\Omega_K(\tau)$. The complementary-distance event costs at most
$V_*\Pr(\mathsf d(\eta,F\nu)>\tau)\le V_*a_N/\tau$.
The term with $\nu\notin\mathfrak G$ contributes at most
$V_*\Pr(\nu\notin\mathfrak G)$; the residual itself there is at most
one, costing one additional such probability. Therefore

$$
\mathbb E \mathsf d(\nu,F\nu)
\le\mathbb E[\overline V(\nu)-\overline V(\eta)]
 +\Omega_K(\tau)+V_*a_N/\tau
 +(V_*+1)\Pr(\nu\notin\mathfrak G).
$$

Sum over $n$ and divide by $T$. The bounded endpoint difference is at
most $V_*/T$; at stationarity it is zero. This proves the quantitative
inequality without requiring the next empirical population to remain in
the class almost surely.

The residual $\mathsf d(\nu,F\nu)$ is bounded and continuous on every
controlled moment sublevel. The compact-sublevel passage argument in
{prf:ref}`thm-slcm-joint-invariant` therefore applies to it. Its limiting
integral is zero whenever
the displayed upper bound vanishes. Since the residual is nonnegative
and vanishes exactly at fixed points, $Q$ is supported on that set.
For the explicit choice with $\alpha<1$, write $L=\log(1/a_N)$.
Then $K_N=O(\log L)$, $q^{K_N}\to0$, and
$\alpha^{K_N}\ge(1+L)^{-1/2}$ for large $N$.
Hence every summand in $B_{K_N}(\sqrt{a_N})$ is at most
$C^{1/(1-\alpha)}\exp[-L/(2\sqrt{1+L})]$; their number grows only
as $O(\log L)$. The sum tends to zero, as does
$a_N/\tau_N=\sqrt{a_N}$. If $\alpha=1$, the corresponding sum is
bounded by a constant times $K_NC^{K_N}e^{-L/2}$, which also tends to
zero, and again $q^{K_N}\to0$. This proves the explicit choices.
:::

:::{prf:corollary} Stationary and time-averaged finite-row mixtures
:label: cor-slcm-row-mixture

Assume permutation-equivariance of the actual kernel and exchangeability
of the initial or stationary finite-$N$ laws. At a uniformly selected
index from $\{0,\ldots,T-1\}$, let $\Gamma_{N,T}^{(k)}$ be the law of
$k$ distinct tagged rows, $N\ge k$. Then

$$
\left\|\Gamma_{N,T}^{(k)}-
 \int\nu^{\otimes k}Q_{N,T}(d\nu)\right\|_{\rm TV}
\le\frac{k(k-1)}{2N}.
$$

The identical bound holds for stationary $Q_N$ and its stationary
$k$-row law. Along any population-law limit $Q$, the tagged-row laws
therefore converge weakly to $\int\nu^{\otimes k}Q(d\nu)$.
Under {prf:ref}`thm-slcm-fixed-support`, this is a mixture of products
of nonlinear stationary laws. Phase weights are determined by the
limiting $Q$; neither uniqueness of those weights nor convergence of
the whole sequence follows merely from tightness.
:::

:::{prf:proof}
Conditional on the empirical law, exchangeability makes distinct tagged
rows an ordered sample without replacement from the $N$ row labels.
The product empirical law is ordered sampling with replacement. Couple
the index samples until the first repeated label; by a union bound its
probability is at most $\sum_{j=0}^{k-1}j/N=k(k-1)/(2N)$.
This proves the TV bound after averaging the empirical law and time.
For bounded continuous tests of $k$ states, the map
$\nu\mapsto\nu^{\otimes k}$ is weakly continuous, so passage to the
population-law limit identifies the mixture. The fixed-support theorem
identifies its components when its additional hypotheses hold.
:::

:::{prf:remark} What the joint theorem establishes and what remains to be checked
:label: rem-slcm-joint-scope

Uniform moment and one-step consistency bounds alone give invariant
population dynamics in joint $N,T\to\infty$ occupation limits and in
stationary finite-particle subsequential limits. Fixed-phase support
additionally uses a proved uniform phase-local residual tail and vanishing
population-class failure probability. These hypotheses can be checked
with the structural flux, containment and actual-operator residual
certificates; they are not consequences of the existence of a spatial
partition alone. A uniform expectation bound at one moment order gives
$\Pr(L_N\psi>H)\le M_\psi/H$, not a probability tending to zero at a
fixed $H$. If phase constants are used on growing moment classes, their
$D,q,\Psi$ and localization dependence must be inserted in the displayed
bound before taking limits. Finally, an occupation-law conclusion does
not by itself replace a theorem about a prescribed instantaneous
observation time $t_N\to\infty$.
:::

(slclt-longtime-blocks)=
### 13.2. Uniform long-time approximation from phase-wise block estimates

:::{div} feynman-prose
Repeatedly extending one short-time error bound can produce an error that grows
without limit. A block argument starts the comparison again at the current
empirical population. The finite-horizon theorem controls the fresh error over
each block; a separate attraction estimate places the resulting population law
near the declared stationary phase or phase set. Adding those errors gives a
bound that does not accumulate every error since initialization.

Confinement keeps the needed moments under control, while explicit defects pay
for departure from a region or population class where the estimates apply. The
attraction step must hold uniformly over the declared class of possible restart
laws. It does not require every initial population to choose the same
phase. The resulting estimate extends beyond a fixed observation horizon when
its constants and defects are uniform; this uniformity is a mathematical
condition to verify, not a consequence of simply restarting the comparison.
:::

:::{prf:definition} Explicit finite-window error used at each restart
:label: def-slclt-window

Use the actual conservative all-alive $N$-row kernel $P_N$, its empirical
law $L_N$, and the exact population map $\mathcal F_h$. All constants
below refer to one fixed choice of algorithm and structural profiles.
Assume one of the proved power-modulus regimes in this chapter, written
$$
 \mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
 \le \min\{1,C(H)\mathsf d(\mu,\nu)^\beta\},\quad0<\beta\le1,
 \qquad C(H)\le C_*\max\{1,H\}^{3/2}.
$$
Here $\beta,C_*$ are the explicit constants of that theorem, including
its interface and excursion profiles; they are not unknown contraction
coefficients. The same construction works with its displayed regional
modulus whenever the resulting finite-window errors tend to zero.

In this restart formula, $g_0,g_1$ bound the **actual configured force**
as $|F(x)|\le g_0+g_1|x|$. If the original force envelope is for
$F_{\rm geom}$ and an auxiliary trap is present, replace its growth
coefficient by $g_1+\lambda$ here; the modulus constants likewise use
the configured force. For input eighth moment at most $H_0$, compute $M_{8,0}=H_0$ and
$$
 M_{8,j+1}=3^7\left[
 (1+\eta g_1)^8 2^7(r_CM_{8,j}+\sigma_J^8g_{8,d})
 +(BV_c+\eta g_0)^8+\sigma_{\rm pos}^8g_{8,d}\right],
$$
where $r_C=1+2/\kappa_C$, $g_{8,d}=d(d+2)(d+4)(d+6)$ and
$\sigma_{\rm pos}^2=c^2q^2+s^2$. Set
$$
 R_N=N^{1/(16d)},\quad\ell_N=N^{-1/(16d)},\quad
 H_{N,j}=\max\{1,M_{8,j}\}\log(N+e),
$$
$$
 a_{N,j}=2\ell_N+\tfrac12J(R_N,\ell_N)
                 \sqrt{(A+4B_*^2)/N}+2M_{8,j+1}^{1/4}/R_N^2,
 \qquad t_{N,j}=\sqrt{a_{N,j}},
$$
with the previously displayed grid-count formula $J$ and primitive scalar
consistency constants $A,B_*$. Define
$$
 v_{N,0}=0,\qquad
 v_{N,j+1}=\min\{1,t_{N,j}+C(H_{N,j})v_{N,j}^{\beta}\},
$$
$$
 \alpha_{N,b}=\sum_{j=0}^{b-1}
          [M_{8,j}/H_{N,j}+\sqrt{a_{N,j}}],\qquad
 e_N(b;H_0)=\min\{1,v_{N,b}+\alpha_{N,b}\}.
$$
For every deterministic entering configuration $S$ with
$L_N(S)|x|^8\le H_0$, the proved trajectory theorem gives
$$
 \mathbb E_S\mathsf d\big(L_N(S_b),\mathcal F_h^bL_N(S)\big)
                  \le e_N(b;H_0).
$$
The initialization error here is exactly zero: the population comparison
starts from the entering empirical law itself. For every fixed $b$,
$e_N(b;H_0)\to0$ as $N\to\infty$, uniformly over these entering
configurations. Its finite-$N$ expression retains every algorithm,
landscape, moment and block-length parameter.
:::

:::{prf:proof}
Condition on the entering configuration. Apply the high-probability
trajectory theorem with initial law $L_N(S)$, hence initial metric error
zero, and with the displayed moment budgets, thresholds and integration
mesh. On its good event the final error is at most $v_{N,b}$; outside
it the bounded metric is at most one. This proves the expectation bound.
The explicit power-modulus closure already proved in this chapter, or
finite induction in the displayed recursion, gives convergence for every
fixed block length. No deterministic population attraction is used in
this finite-window step.
:::

:::{prf:theorem} Uniform-time block reset to a stationary phase or phase set
:label: thm-slclt-reset

Let $\mathfrak G_1,\ldots,\mathfrak G_m$ be declared Borel classes of population
laws, all with eighth moment at most a specified $H_0<\infty$. For each
class, let $\mathcal A_i$ be a nonempty closed set of stationary
population laws, and suppose an explicit uniform attraction estimate
has been proved:
$$
 \sup_{\nu\in\mathfrak G_i}
 \operatorname{dist}_{\mathsf d}(\mathcal F_h^b\nu,\mathcal A_i)
                    \le a_i(b),\qquad a_i(b)\longrightarrow0.
$$
Its constants must be supplied by an actual dissipation or relaxation
certificate, as detailed below. Define
$\mathfrak G=\bigcup_i\mathfrak G_i$,
$\mathcal A=\bigcup_i\mathcal A_i$, and $a(b)=\max_i a_i(b)$.
For the actual swarm law from its specified initialization, let
$$
 \Delta_N(t)=\Pr\{L_N(S_t)\notin\mathfrak G\}.
$$
For every $n\ge b\ge1$,
$$
 \boxed{\quad
 \mathbb E\operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le\min\{1,e_N(b;H_0)+a(b)+\Delta_N(n-b)\}.
 \quad}
$$
Therefore, if $\overline\Delta_N=\sup_{t\ge0}\Delta_N(t)$ is bounded
by an explicit proved expression, then
$$
 \sup_{n\ge b}\mathbb E\operatorname{dist}_{\mathsf d}
       (L_N(S_n),\mathcal A)
       \le e_N(b;H_0)+a(b)+\overline\Delta_N.
$$
Markov's inequality divides each right-hand side by a requested error
tolerance to give its probability bound, capped by one. At any specified
$(N,n)$ one may minimize the displayed numerical bound over
$1\le b\le n$.

For a single class $\mathfrak G_i$ and a proved common stationary limit
$\pi_i$, take $\mathcal A_i=\{\pi_i\}$ and
$\Delta_{N,i}(t)=\Pr\{L_N(S_t)\notin\mathfrak G_i\}$. This gives the
same uniform-time estimate for distance to that particular phase.
It never compares populations in different phase classes.
:::

:::{prf:proof}
Condition on the complete swarm at time $n-b$. If its empirical law is
in class $i$, apply the conditional finite-window bound from
{prf:ref}`def-slclt-window`. The deterministic population evolution from
that exact empirical law is within $a_i(b)$ of $\mathcal A_i$.
Distance to a set is 1-Lipschitz, so their sum bounds the conditional
expected distance of the actual final empirical law to $\mathcal A$.
If the empirical law belongs to several classes, choose its least index;
this is measurable for measurable declared classes. Off $\mathfrak G$
use the bound one. Average and enlarge the resulting bound to the
stated sum. The supremum, probability, minimization and single-phase
conclusions follow directly. Intermediate empirical states need not
remain in the class for this finite-window argument; their excursions
are already charged in $e_N(b;H_0)$. The class probability is needed at
the restart time, not silently assumed at every future update.
:::

:::{prf:corollary} Arbitrary simultaneous large-population and long-time limits
:label: cor-slclt-diagonal

In {prf:ref}`thm-slclt-reset`, suppose the proved uniform retention bound
satisfies $\overline\Delta_N\to0$. For every deterministic sequence
$n_N\to\infty$,
$$
 \operatorname{dist}_{\mathsf d}(L_N(S_{n_N}),\mathcal A)
                    \longrightarrow0
 \quad\text{in mean and in probability}.
$$
In the single-phase version this is convergence to $\pi_i$ for every
such diagonal. If the deterministic population trajectory from $\mu_0$
also converges to that phase with a proved bound
$\mathsf d(\mathcal F_h^n\mu_0,\pi_i)\le a_0(n)$, then
$$
 \mathbb E\mathsf d(L_N(S_n),\mathcal F_h^n\mu_0)
 \le e_N(b;H_0)+a_i(b)+\overline\Delta_{N,i}+a_0(n),
 \qquad n\ge b.
$$
Together with the existing finite-horizon estimate for $0\le n<b$,
this supplies a rigorous uniform-in-time mean-field approximation when
these same-phase hypotheses are verified. For a union of phases, only
distance to their stationary set is asserted; phase weights need not
match the deterministic initialization.
:::

:::{prf:proof}
Fix $b$. For all sufficiently large $N$, $n_N\ge b$. The reset theorem
and $e_N(b;H_0)\to0$ give limsup of the expected distance at most $a(b)$.
Let $b\to\infty$. Markov's inequality gives convergence in probability.
In the single-phase case the target set is a singleton. Apply the triangle
inequality with $\pi_i$ to obtain the last bound. For uniform approximation,
first choose $b$ large enough to control both phase-attraction terms for
all $n\ge b$, then take $N$ large enough for the reset and retention
errors and the separate finite initial window. This uses no cross-phase
contraction or unjustified exchange of limits.
:::

:::{prf:corollary} Stationary finite populations and stationary population mixtures
:label: cor-slclt-stationary

Suppose $\Lambda_N$ is the distribution of the empirical law under an
actual invariant finite-population law, and
$\Lambda_N(\mathfrak G^c)\le\delta_N^{\rm stat}$ is a proved estimate.
Then for every $b\ge1$,
$$
 \int\operatorname{dist}_{\mathsf d}(\nu,\mathcal A)\,\Lambda_N(d\nu)
 \le e_N(b;H_0)+a(b)+\delta_N^{\rm stat}.
$$
If $\delta_N^{\rm stat}\to0$, every weak subsequential limit of
$\Lambda_N$ is supported on $\mathcal A$. Such subsequences exist:
the uniform moment cap on $\mathfrak G$ and its probability tending to
one give tightness of these laws on population space.
When $\mathcal A=\{\pi_1,\ldots,\pi_m\}$ is finite, every limiting
law is $\sum_i\theta_i\delta_{\pi_i}$ for some nonnegative weights
summing to one. In the single-phase case $\Lambda_N\Rightarrow
\delta_{\pi_i}$. No uniqueness of mixture weights is inferred without
an additional phase-selection estimate.
:::

:::{prf:proof}
Start the swarm in its invariant law and apply the reset bound; its
empirical distribution at both endpoints is $\Lambda_N$.
For fixed $b$, pass to the limit in any convergent subsequence using
bounded continuity of distance to the closed set $\mathcal A$.
Then let $b\to\infty$ to obtain zero expected distance, hence support
on $\mathcal A$. The set of single-row laws with eighth moment at most
$H_0$ is tight (by its explicit moment tails) and closed by lower
semicontinuity, hence compact in the weak topology. Since $\Lambda_N$
places probability tending to one in that compact set, the sequence of
population-law distributions is tight; finitely many initial indices
can be covered by enlarging compact sets. The mixture and singleton
statements follow from the support conclusion.
:::

:::{prf:proposition} Which existing phase estimate supplies the attraction term
:label: prop-slclt-attraction-input

Suppose the verified class-wide residual-dissipation inequality
of {prf:ref}`thm-slcpd-local-dissipation` holds on invariant
$\mathfrak G_i$ with its numerical $q_i\in(0,1)$ and uniform eighth
moment bound. For each $\nu\in\mathfrak G_i$, let $\pi_\nu$ be the
stationary limit supplied by that theorem. Since every TV residual is
at most one,
$$
 \|\mathcal F_h^b\nu-\pi_\nu\|_{\rm TV}
               \le\min\{1,q_i^{b-1}/(1-q_i)\},\qquad b\ge1.
$$
Thus one may take
$a_i(b)=\min\{1,q_i^{b-1}/(1-q_i)\}$ and choose $\mathcal A_i$
as the closure of these stationary limits. They remain stationary:
the uniform moment bound and the proved continuity modulus pass the
fixed-point identity to that closure. This supplies a phase-set
attraction bound even when $\pi_\nu$ depends on $\nu$ within the class.

A singleton target additionally requires proof that all these limits
coincide, for example from an existing matching full-law relaxation
estimate. The per-trajectory summable-increment theorem by itself gives
neither a uniform tail rate over a class nor a common phase limit.
The finite-horizon consistency theorem supplies $e_N$ but does not
supply $a_i$ or uniform retention. A finite-horizon residence bound
$\Pr(\text{exit by }t)\le tu_N$ alone does not give
$\sup_{t\ge0}\Delta_N(t)\to0$ when $u_N>0$.
:::

:::{prf:proof}
Insert $\mathscr R(\nu)\le1$ into the proved residual-dissipation
rate. TV bounds the bounded transport metric, so it supplies the stated
attraction term. Limits have eighth moment at most the same bound by
lower semicontinuity. For a sequence of these fixed laws converging
weakly, the explicit common moment-class modulus carries
$\mathcal F_h\pi_k=\pi_k$ to its limit; the closed moment bound is
retained. The final distinctions follow from the respective quantifiers:
pointwise convergence does not give a supremum over entering laws,
and a bound growing with $t$ cannot control a supremum over all times.
:::


:::{prf:corollary} Uniform-time approximation localized by a proved moment tail
:label: cor-slclt-moment-localization

Suppose the actual particle dynamics have the explicit uniform budget
$$
 \sup_{N\ge1}\sup_{t\ge0}\mathbb E W_8(S_t)\le M_8<\infty,
$$
for example the numerical budget in
{prf:ref}`thm-slcj-uniform-moments` when all its defect bounds hold.
There is no assumption that empirical eighth moments are almost surely
bounded by a deterministic constant.

For every cutoff $H>0$ under consideration, let $\mathfrak G_H$ be a
Borel class contained in $\{\nu:\nu|x|^8\le H\}$. Let $\mathcal A$
be one common nonempty closed set of stationary population laws. Suppose
an actual quantitative phase estimate proves
$$
 \sup_{\nu\in\mathfrak G_H}
 \operatorname{dist}_{\mathsf d}(\mathcal F_h^b\nu,\mathcal A)
                \le a_H(b),\qquad \lim_{b\to\infty}a_H(b)=0
 \quad\text{for each fixed }H.
$$
Also supply the quantitative coverage-failure bound
$$
 \sup_{t\ge0}\Pr\{W_8(S_t)\le H,
                       L_N(S_t)\notin\mathfrak G_H\}
                       \le\delta_N(H).
$$
If $\mathfrak G_H$ is the whole moment class, this term is zero.
Otherwise basin coverage, normalization or phase restrictions in its
definition must be accounted for by this actual probability; the moment
bound alone does not imply them.

Then, for every $n\ge b\ge1$,
$$
 \boxed{\quad
 \mathbb E\operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le\min\{1,e_N(b;H)+a_H(b)+M_8/H+\delta_N(H)\}.
 \quad}
$$
The same bound holds for the supremum over all $n\ge b$. In particular,
at a specified $(N,n)$ the right-hand side can be replaced by
$$
 \min\left\{1,\inf_{\substack{H>0,\ 1\le b\le n\\
                         \text{certified }H,b}}
 [e_N(b;H)+a_H(b)+M_8/H+\delta_N(H)]\right\}.
$$
The infimum is over the displayed numerical certificate parameters,
not an unknown optimal convergence constant.

If these certificates are available for arbitrarily large $H$ and
$\delta_N(H)\to0$ for each such fixed $H$, then for every deterministic
$n_N\to\infty$,
$$
 \operatorname{dist}_{\mathsf d}(L_N(S_{n_N}),\mathcal A)
                 \longrightarrow0
 \quad\text{in expectation and probability}.
$$
A singleton target $\mathcal A=\{\pi_i\}$ gives the corresponding
joint limit to that phase. A union of stationary phases gives the
phase-set conclusion without fixing its weights.
:::

:::{prf:proof}
Markov's inequality gives
$\Pr\{W_8(S_t)>H\}\le M_8/H$ uniformly in $N,t$.
Splitting the complement of $\mathfrak G_H$ at this moment event yields
$$
 \sup_t\Pr\{L_N(S_t)\notin\mathfrak G_H\}
                         \le M_8/H+\delta_N(H).
$$
Apply the block-reset proof with moment threshold $H$ and the common
stationary target set; it does not require bounded empirical moments
outside the good restart event. This proves the finite bound, its
supremum and its infimum over certified choices.

To prove the unrestricted diagonal statement, let $\varepsilon>0$.
First choose a certified $H$ so $M_8/H<\varepsilon/3$.
Then choose a fixed $b$ with $a_H(b)<\varepsilon/3$.
For all sufficiently large $N$, $n_N\ge b$ and
$e_N(b;H)+\delta_N(H)<\varepsilon/3$, because this is a fixed finite
window and a fixed cutoff. The expected distance is then at most
$\varepsilon$. Markov's inequality gives convergence in probability.
This order of choices explicitly justifies the joint limit without
requiring a pathwise population-moment cap or exchanging two uncontrolled
limits.
:::

:::{prf:corollary} Stationary limits under moment-localized coverage
:label: cor-slclt-stationary-localization

Let $\Lambda_N$ be empirical-law distributions under invariant swarm
laws with the proved bound
$\int \nu|x|^8\,\Lambda_N(d\nu)\le M_8^{\rm stat}$.
Use the same $\mathfrak G_H,\mathcal A,a_H$ as above and suppose
$$
 \Lambda_N\{\nu:\nu|x|^8\le H,\ \nu\notin\mathfrak G_H\}
                        \le\delta_N^{\rm stat}(H).
$$
Then
$$
 \int\operatorname{dist}_{\mathsf d}(\nu,\mathcal A)\,\Lambda_N(d\nu)
 \le e_N(b;H)+a_H(b)+M_8^{\rm stat}/H+\delta_N^{\rm stat}(H).
$$
If $\delta_N^{\rm stat}(H)\to0$ for every certified fixed cutoff and
arbitrarily large cutoffs are available, every subsequential limit of
$\Lambda_N$ is supported on $\mathcal A$, and such subsequences exist.
For finitely many stationary phases this is a stationary-mixture
statement, with phase weights requiring their own transfer analysis.
:::

:::{prf:proof}
Apply the reset inequality to a stationary entering swarm and split its
class-failure probability at moment cutoff $H$, exactly as above.
The family $\Lambda_N$ is tight: for any $K>0$, its mass outside the
compact set of population laws with eighth moment at most $K$ is at most
$M_8^{\rm stat}/K$. Compactness follows from the explicit spatial moment
tails, the compact velocity cap, and lower semicontinuity of the moment.
Pass to a weakly convergent subsequence, keep $H,b$ fixed first, and use
bounded continuity of distance to the closed set $\mathcal A$.
Then choose $H$ large and $b$ large at that fixed cutoff, as in the
preceding proof. The limiting expected distance is zero, proving support
on $\mathcal A$. The finite-phase interpretation follows directly.
:::

(sec-slca-active-stationary)=
### 13.3. A fully evaluated stationary mean-field regime with active cloning

:::{prf:theorem} Uniform moments and finite-population stationarity with unrestricted fitness exponents
:label: thm-slca-active-stationary

Use the actual conservative all-alive canonical algorithm on
$(\mathbb R^d\times\overline B_{V_{\max}})^N$, $N\ge2$, with the
bounded comparison features, positive regularization floors, independent
current-step measurement and cloning companions, simultaneous frozen-
source copying, Gaussian recipient jitter, component collisions, BAOAB,
independent final position noise and the stated radial cap. There is no
viscosity or historical donor mechanism. Reward and diversity fitness
exponents may be any finite nonnegative values, including strictly
positive values; copying and collision are not disabled.

Choose $h>0$, finite $\gamma\ge0$, $b_O>0$, $\sigma_x>0$, and define

$$
\begin{aligned}
c&=h/2,\quad a=e^{-\gamma h}>0,\quad
B=c(1+a),\quad\eta=c^2(1+a),\quad\lambda=\eta^{-1},\\
q^2&=b_O^2(1-e^{-2\gamma h})/(2\gamma),\quad
s^2=\sigma_x^2h,\quad \tau^2=c^2q^2+s^2,\\
V_c&=(1+2|\alpha_{\rm col}|)V_{\max},
\end{aligned}
$$

with $q^2=b_O^2h$ at $\gamma=0$. Let the actual force be

$$
F(x)=f(x)-\lambda x,\qquad
\sup_x|f(x)|\le g_0<\infty,\qquad
\operatorname{Lip}(F)\le L_F<\infty.
$$

No convexity, smallness of $L_F$, or inequality $c^2L_F<1$ is assumed.
The raw reward must be a finite Borel function on the state space, so
the regularized formulas define a measurable transition kernel. Put

$$
b_0=BV_c+\eta g_0,\quad
M_2=b_0^2+d\tau^2,\quad b=1+M_2,
$$

and, for every $p\ge1$, put

$$
m_{d,p}=2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)},\qquad
M_p^{\rm all}=(b_0+\tau m_{d,p}^{1/p})^p.
$$

The exact full update satisfies, for every entering population $S$,

$$
\mathbb E[L_N(S_1)|x|^2\mid S]\le M_2,
\qquad
\mathbb E[L_N(S_1)|x|^p\mid S]\le M_p^{\rm all}.
$$

These bounds are uniform in $N$ and require no moment bound on the
entering population distribution. For every admissible population law,
the actual nonlinear map obeys the same bounds:
$\mathcal F_h(\mu)|x|^2\le M_2$ and
$\mathcal F_h(\mu)|x|^p\le M_p^{\rm all}$.

The following formulas give explicit finite-$N$ regeneration constants.
Choose proof parameters $J,r,u>0$, set $R=4b$, and define

$$
\begin{aligned}
R_N&=\sqrt{N(R-1)},\qquad R_{\mathrm{prep}}=R_N+J,\qquad
\alpha_0=1-c^2\lambda=\frac a{1+a}>0,\\
R_1&=\alpha_0R_{\mathrm{prep}}+cV_c+c^2g_0,\qquad
m_v=a(V_c+c\lambda R_{\mathrm{prep}}+cg_0),\\
Q&=(u+c\lambda R_1+cg_0)/\alpha_0,\\
k_v&=(2\pi q^2)^{-d/2}
 \exp[-(Q+m_v)^2/(2q^2)](1+c^2L_F)^{-d},\\
k_x&=(2\pi s^2)^{-d/2}
 \exp[-(r+R_1+cQ)^2/(2s^2)].
\end{aligned}
$$

The preparation radius is distinct from the comparison-feature radii. Let $v_d(t)=\pi^{d/2}t^d/\Gamma(1+d/2)$ and

$$
p_J=\begin{cases}
\displaystyle\frac1{\Gamma(d/2)}
 \int_0^{J^2/(2\sigma_J^2)}t^{d/2-1}e^{-t}\,dt,&\sigma_J>0,\\
1,&\sigma_J=0,
\end{cases}
\quad
\epsilon_N=[p_Jv_d(u)v_d(r)k_vk_x]^N,
\quad\varepsilon_N^{\rm reg}=3\epsilon_N/4.
$$

There is a unique invariant full-swarm probability $\Pi_N$, and for
any initial probability $\Lambda_N$ on the stated state space,

$$
\boxed{\quad
\|\Lambda_NP_N^n-\Pi_N\|_{\rm TV}
\le(1-\varepsilon_N^{\rm reg})^{\lfloor n/2\rfloor}.
\quad}
$$

Thus $n=2\lceil\log(1/\delta)/[-\log(1-\varepsilon_N^{\rm reg})]\rceil$
updates suffice for error $\delta\in(0,1)$, with physical time $nh$.
The moment bounds are uniform in $N$; this whole-swarm TV regeneration
constant is explicitly $N$-dependent and is not asserted to be uniform.
:::

:::{prf:proof}
Write $(X,V)$ for a row's prepared position and collision velocity after
its actual graph has been generated, copying has occurred, and jitter
has been added. Regardless of graph size or selection strength,
$|V|\le V_c$. The exact completed position is

$$
X^+=X+BV+\eta F(X)+cq\xi+s\zeta
=BV+\eta f(X)+cq\xi+s\zeta,
$$

because $\eta\lambda=1$. The deterministic center is bounded by $b_0$;
it still depends on the actual nonlinear copying and collision law.
The two noises are independent centered Gaussians, independent of
preparation, with combined covariance $\tau^2I$. Conditional Gaussian
second moments give $M_2$. Minkowski's inequality and the Gaussian
radial integral give $M_p^{\rm all}$. Average over rows and all
preparation randomness. The same conditional calculation for the rooted
population output proves its bounds without suppressing active cloning.

Let $\mathcal V(S)=1+N^{-1}\sum_i|x_i|^2$. Then $P_N\mathcal V\le b$.
On the set $\mathcal V\le R$, every entering position has norm at most
$R_N$. Expose all copying and collision marks and sample the latent
jitter for every row, including unused jitter. With probability $p_J^N$
all jitter norms are at most $J$; this event is independent of the
pre-jitter graph. On it all prepared positions have norm at most $R_{\mathrm{prep}}$.
The actual first drift position is
$x_1=\alpha_0X+cV+c^2f(X)$, hence $|x_1|\le R_1$; the OU velocity
mean has norm at most $m_v$.

For pre-cap target velocity $|w|\le u$, consider

$$
T(z)=z+cF(x_1+cz)
=\alpha_0z-c\lambda x_1+cf(x_1+cz).
$$

The continuous map
$z\mapsto[w+c\lambda x_1-cf(x_1+cz)]/\alpha_0$
sends $\overline B_Q$ into itself. Brouwer's theorem gives a preimage
$T(z)=w$ there. Every such preimage has norm at most $Q$ by the same
equation. The Lipschitz constant of $T$ is at most $1+c^2L_F$.
The area formula therefore bounds the density of the absolutely continuous
part of the Gaussian pushforward on $B_u$ below by $k_v$: the absolute Jacobian is at most
$(1+c^2L_F)^d$, and almost every target has a regular preimage; the
critical and nondifferentiability sets have null images under the
Lipschitz map. A possible additional singular part only increases the
resulting measure domination. This is the noninjective minorization argument already
proved in {prf:ref}`lem-slcp-surjective-minorization`.
Conditional on a preimage, final position has Gaussian mean
$x_1+cz$ of norm at most $R_1+cQ$. Its density on $B_r$ is at least
$k_x$. Applying the area formula with this conditional density as weight
proves the joint lower bound $k_vk_x$ on $B_r\times B_u$.

The rowwise kinetic innovations are independent after the complete
preparation is exposed, even though collision rotations and cloning
are shared. Multiplying these lower densities across rows and then
integrating the good jitter event gives
$P_N(S,\cdot)\ge\epsilon_N\nu_N$ on $\{\mathcal V\le R\}$,
where $\nu_N$ is the product of uniform positions on $B_r$ and cap-
pushforwards of uniform pre-cap velocities on $B_u$.

For every entering $S$, Markov's inequality gives
$P_N(S,\{\mathcal V\le R\})\ge1-b/R=3/4$.
Consequently $P_N^2(S,\cdot)\ge\varepsilon_N^{\rm reg}\nu_N$
globally. Splitting this common probability shows contraction of
$P_N^2$ on probability measures in TV by
$1-\varepsilon_N^{\rm reg}$. Completeness gives its unique invariant
probability; its image under $P_N$ is also $P_N^2$-invariant, so it is
$P_N$-invariant. Iteration and the final optional Markov step prove
the TV rate. This argument establishes existence directly and does not
leave a Feller or compactness hypothesis unchecked. Invariance and the
uniform conditional moment bounds imply all stated stationary moments
by truncation and monotone convergence.
:::

:::{prf:corollary} Stationary mean-field limits for the active regime
:label: cor-slca-active-stationary-mf

In addition to the preceding assumptions, let the raw landscape reward
be continuous and satisfy either the bounded-Lipschitz reward conditions
already proved in this chapter, or

$$
|R(x)|\le K_0+K_2|x|^2,\qquad
\operatorname{Lip}(R|_{B(0,L)})\le L_0+L_1L
\quad(L>0)
$$

with declared finite nonnegative constants. These hypotheses and the
force regularity put the actual population map in the established
mean-field continuity and scalar-consistency regime. Let
$Q_N=(L_N)_\#\Pi_N$, and use the explicit canonical scalar constant
$G=A+4B_*^2$. For any $R'>0$, $0<\ell\le1$, define

$$
\begin{aligned}
J(R',\ell)&=
\left(1+\left\lceil\frac{2R'\sqrt{2d}}\ell\right\rceil\right)^d
\left(1+\left\lceil\frac{2V_{\max}\sqrt{2d}}\ell\right\rceil\right)^d,\\
a_N&=\min\left\{1,2\ell+\frac{J(R',\ell)}2\sqrt{G/N}
                         +\frac{2M_2}{(R')^2}\right\}.
\end{aligned}
$$

Then, for the bounded population metric $\mathsf d$ of this chapter,

$$
\mathcal W_{\mathsf d}(Q_N,\mathcal F_{h\#}Q_N)\le a_N.
$$

With $R'=N^{1/(16d)}$, $\ell=N^{-1/(16d)}$, this tends to zero.
Moreover $Q_N$ is tight in the fourth-moment topology on population
laws, because

$$
\int\mu|x|^8\,Q_N(d\mu)\le
M_8^{\rm all}=
\left[b_0+\tau\{d(d+2)(d+4)(d+6)\}^{1/8}\right]^8
$$

uniformly in $N$. Every subsequential limit $Q$ consequently satisfies
$\mathcal F_{h\#}Q=Q$.

The invariant swarm law is exchangeable by uniqueness and permutation-
equivariance. For every fixed $k$, along the same subsequence its
$k$-row marginal converges weakly to
$\int\mu^{\otimes k}Q(d\mu)$, with finite-$N$ empirical-product
comparison error at most $k(k-1)/(2N)$ in TV.
This is a stationary mean-field theorem for a parameter regime admitting
active cloning. No phase-attraction assumption is needed for population-
dynamics invariance. Support on fixed population phases additionally
requires the residual-dissipation criterion of
{prf:ref}`thm-slcm-fixed-support`; it is not inferred from finite-$N$
uniqueness.
:::

:::{prf:proof}
The uniform root moment estimate and stationary empirical second moment
are both bounded by $M_2$. Apply the proved scalar-consistency and finite-
cell matching estimate to obtain the displayed $a_N$. Pair the
stationary next empirical law with $\mathcal F_h$ of the current one;
the expected bounded-metric discrepancy is at most $a_N$, giving the
population-law invariance bound.

For tightness, Markov's inequality gives
$Q_N\{\mu:\mu|x|^8>L\}\le M_8^{\rm all}/L$.
Sublevels of the eighth moment are compact in the fourth-moment topology
on capped states, since their fourth-moment tails are at most $L/r^4$
beyond radius $r$. The same sublevels are weakly compact. The explicit moment-class population
modulus proves restricted weak continuity there. The compact-sublevel
passage argument of {prf:ref}`thm-slcm-joint-invariant`, with the uniform
eighth-moment budget just proved, therefore gives
$\mathcal F_{h\#}Q=Q$ for every convergent subsequence.


Permuting row labels preserves the full kernel, hence sends $\Pi_N$ to
another invariant probability. Uniqueness makes it the same law.
The without-replacement versus with-replacement sampling argument of
{prf:ref}`cor-slcm-row-mixture` gives the explicit $k(k-1)/(2N)$ bound
and identifies the weak marginal limit. The conclusion concerns an
invariant distribution of population laws; the additional statement
about fixed phases uses exactly the separate hypotheses specified above.
:::

:::{prf:corollary} An explicit simultaneous population-size and observation-time limit
:label: cor-slca-joint-stationary-time

Under the two preceding results, for each $N\ge2$ choose

$$
n_N=2\left\lceil
 \frac{\log N}{-\log(1-\varepsilon_N^{\rm reg})}
\right\rceil,\qquad t_N=hn_N.
$$

For any initial swarm law $\Lambda_N$, let
$\widehat Q_N=\operatorname{Law}_{\Lambda_N}(L_N(S_{n_N}))$.
Then

$$
\|\widehat Q_N-Q_N\|_{\rm TV}\le N^{-1},\qquad
\mathcal W_{\mathsf d}(\widehat Q_N,\mathcal F_{h\#}\widehat Q_N)
\le a_N+2/N.
$$

Consequently the observed population laws are tight and every
subsequential limit along this explicit simultaneous limit
$N\to\infty$, $t_N\to\infty$ is invariant under the actual nonlinear
population map. At any larger observation index the same conclusions
hold with the same upper bounds. For $k$ distinct tagged rows, along the
same subsequence their law converges to
$\int\mu^{\otimes k}Q(d\mu)$; its finite-$N$ comparison with
$\int\mu^{\otimes k}Q_N(d\mu)$ has TV error at most
$N^{-1}+k(k-1)/(2N)$.

This theorem has a completely specified population-size/time relation;
it does not exchange two unspecified limits. Its elapsed algorithmic
time is $hn_N$, and it advances $Nn_N$ walker slots, before accounting
for the actual additional companion and collision work per update.
The guaranteed times can be extremely large because the proved
whole-population minorization is conservative and explicitly depends on
$N$.
:::

:::{prf:proof}
Substitution of $n_N$ into the full-swarm TV estimate gives
$\|\Lambda_NP_N^{n_N}-\Pi_N\|_{\rm TV}\le1/N$.
A measurable pushforward cannot increase TV, proving the first bound.
The triangle inequality for $\mathcal W_{\mathsf d}$, with $\mathsf d\le1$, gives

$$
\begin{aligned}
\mathcal W_{\mathsf d}(\widehat Q_N,\mathcal F_{h\#}\widehat Q_N)
\le{}&\mathcal W_{\mathsf d}(\widehat Q_N,Q_N)
 +\mathcal W_{\mathsf d}(Q_N,\mathcal F_{h\#}Q_N)\\
&+\mathcal W_{\mathsf d}(\mathcal F_{h\#}Q_N,
                  \mathcal F_{h\#}\widehat Q_N)
\le{}&N^{-1}+a_N+N^{-1}.
\end{aligned}
$$

The last term is bounded by TV contraction under the deterministic
pushforward, not by an assumed Lipschitz or contraction property of
$\mathcal F_h$. Vanishing TV distance transfers tightness and limiting
laws from $Q_N$ to $\widehat Q_N$. The tagged-row marginal is likewise
within $1/N$ of its stationary counterpart, so the stationary empirical-
product comparison gives the stated error without requiring the initial
law to be exchangeable. Finally $\varepsilon_N^{\rm reg}\le3/4$,
so $n_N\ge2\log N/\log4$ and $t_N\to\infty$ for fixed $h>0$.
:::

(sec-slcj-stationary-weights)=
### 13.4. Identifying the stationary phase weights

:::{prf:theorem} Full stationary mean-field limit from quantified phase transfers
:label: thm-slcj-phase-weights

Let $\Pi_N$ be exchangeable invariant laws of the actual conservative
swarm kernel. Partition swarm space into measurable phase classes
$G_{1,N},\ldots,G_{m,N}$ and an exterior class $G_{0,N}$.
Let $w_{i,N}=\Pi_N(G_{i,N})$ and suppose $w_{0,N}\le\delta_N$.
For declared distinct stationary population laws $\pi_i$, let

$$
\zeta_N=\sum_{i=1}^m\int_{G_{i,N}}
       \mathsf d(L_N(S),\pi_i)\,\Pi_N(dS).
$$

This quantity must be bounded using the proved phase concentration
estimates or the definition of the phase neighborhoods. Choose a
positive scale $c_N$ and nonnegative numbers $a_{ij}$, $i\ne j$,
such that the directed graph of positive $a_{ij}$ is strongly connected.
Suppose the actual full kernel satisfies, for all $S\in G_{i,N}$,

$$
\left|\frac{P_N(S,G_{j,N})}{c_N}-a_{ij}\right|\le\epsilon_N
\quad(j\ne i,\ j\ge1),\qquad
P_N(S,G_{0,N})\le c_N\eta_N.
$$

These are uniform conditional transition bounds, not a lumpability
assumption. They can equally apply to a declared sampled kernel $P_N^b$;
stationarity is unchanged, and physical transition time is then $bh$.
Define the numerical generator $A$ by the displayed off-diagonal entries
and $a_{ii}=-\sum_{j\ne i}a_{ij}$. Form the $m\times m$ matrix $B$
whose first $m-1$ columns are those of $A$ and whose last column is all
ones. Put

$$
\theta=e_m^\top B^{-1},\qquad
K_A=\frac{\max_i\sum_j|\operatorname{adj}(B)_{ij}|}{|\det B|}.
$$

Then $B$ is invertible, $\theta$ is a probability row vector, and
$\theta A=0$. For $w_N=(w_{1,N},\ldots,w_{m,N})$,

$$
\boxed{\quad
\|w_N-\theta\|_1\le K_A
 \left[2(m-1)\epsilon_N+\eta_N+
             \frac{\delta_N}{c_N}+\delta_N\right].
\quad}
$$

Writing $Q_N=(L_N)_\#\Pi_N$ and $Q_* =\sum_i\theta_i\delta_{\pi_i}$,
the bounded population transport metric satisfies

$$
\mathcal W(Q_N,Q_*)\le
\zeta_N+\frac32\delta_N+
\frac{K_A}{2}\left[2(m-1)\epsilon_N+\eta_N+
                 \frac{\delta_N}{c_N}+\delta_N\right].
$$

In particular if $\zeta_N,\epsilon_N,\eta_N,\delta_N\to0$ and
$\delta_N/c_N\to0$, the entire stationary sequence converges to $Q_*$,
not merely its subsequences. For every fixed $k$, the stationary $k$-row
law converges weakly to $\sum_i\theta_i\pi_i^{\otimes k}$. With the
average coordinate cost $k^{-1}\sum_{j=1}^k\min\{1,|z_j-z'_j|\}$,
its transport error is at most the preceding population error plus
$k(k-1)/(2N)$.

All entries of $A$ must be established from the full-kernel transition
bounds. For very rare crossings, absolute transition errors tending to
zero do not suffice: the errors must be controlled relative to $c_N$.
If no common limiting generator or phase weights are certified, retain
the earlier set-valued or subsequential conclusion.
:::

:::{prf:proof}
For $m=1$, use $B=(1)$ and $\theta=(1)$; the same estimates hold with
empty inter-phase sums. For $m\ge2$, let $M\ge\max_i\sum_{j\ne i}a_{ij}$,
$M>0$, and $T=I+A/M$. This is an irreducible stochastic matrix. A limit
point of the finite-dimensional Cesaro averages of any row probability
under $T$ is a row probability $\theta$ with $\theta T=\theta$:
the difference between an average multiplied by $T$ and itself is its
two endpoints divided by the number of terms. If $\theta_j=0$, stationarity
forces every positive predecessor of $j$ to have zero weight; strong
connectivity would then force every weight to vanish. Hence $\theta>0$.

If $Av=0$ for a real column vector, take an index at which $v$ is maximal.
The equation $\sum_{j\ne i}a_{ij}(v_j-v_i)=0$ forces equality at every
outgoing positive edge, then everywhere by connectivity. Thus the right
nullspace consists of constants, the rank is $m-1$, and the left nullspace
is one-dimensional. If a row $x$ satisfies $xB=0$, its sum is zero and
the first $m-1$ coordinates of $xA$ are zero. Since $A\mathbf1=0$, the
last is also zero. Therefore $x$ is a multiple of $\theta$ with sum zero,
hence $x=0$. This proves invertibility and the stated formula for $\theta$.

Let $f_{ij}=\int_{G_{i,N}}P_N(S,G_{j,N})\,\Pi_N(dS)$.
Stationarity gives $\sum_{j\ne i}f_{ji}=\sum_{j\ne i}f_{ij}$,
including the exterior index. For $i,j\ge1$, $i\ne j$,
$|f_{ij}/c_N-w_{i,N}a_{ij}|\le w_{i,N}\epsilon_N$.
Each inter-phase flow error enters the vector balance twice with opposite
signs, so their total absolute contribution is at most
$2(m-1)\epsilon_N\sum_iw_{i,N}\le2(m-1)\epsilon_N$.
The exterior contributions sum to at most
$\sum_i(f_{0i}+f_{i0})/c_N\le\delta_N/c_N+\eta_N$.
Thus $\|w_NA\|_1$ is bounded by the first three terms in brackets.
The row $(w_N-\theta)B$ has its first $m-1$ coordinates equal to
those of $w_NA$ and its last equal to $-w_{0,N}$. Multiplication by
$B^{-1}$ and the row-vector $\ell^1$ matrix bound
$\|xB^{-1}\|_1\le K_A\|x\|_1$ proves the weight estimate.

Map the empirical law to $\pi_i$ on phase $i$ and to $\pi_1$ on the
exterior. This coupling costs at most $\zeta_N+\delta_N$ and gives
a discrete target with weights $w_N+ w_{0,N}e_1^\top$.
Its TV distance from $\theta$ is at most
$(\|w_N-\theta\|_1+\delta_N)/2$, which bounds transport because the
cost is at most one. This proves the claimed population error.
A coupling of two single-row laws of expected cost $\mathsf d$ can be
sampled independently $k$ times; the average coordinate cost remains
$\mathsf d$. Mix such couplings over population laws, then use the
without-replacement estimate of {prf:ref}`cor-slcm-row-mixture`.
This proves the finite-row error and hence the full-sequence limits.
:::

:::{prf:corollary} A complete simultaneous limit to an identified stationary mixture
:label: cor-slcj-full-mixture-limit

Combine the evaluated active-cloning regime of
{prf:ref}`thm-slca-active-stationary` with the verified phase-transfer
and concentration bounds of {prf:ref}`thm-slcj-phase-weights`. Denote the
explicit right-hand side of its population bound by $E_N^{\rm phase}$.
For every initial swarm law $\Lambda_N$ and every $n\ge0$,

$$
\boxed{\quad
\mathcal W\!\left(\operatorname{Law}_{\Lambda_N}(L_N(S_n)),
                  \sum_i\theta_i\delta_{\pi_i}\right)
\le(1-\varepsilon_N^{\rm reg})^{\lfloor n/2\rfloor}
      +E_N^{\rm phase}.
\quad}
$$

For $k$ tagged rows, the corresponding average-coordinate transport
bound to $\sum_i\theta_i\pi_i^{\otimes k}$ is the same right-hand side
plus $k(k-1)/(2N)$. This holds without exchangeability of the initial law,
because its full swarm law first approaches the exchangeable stationary law.

If $E_N^{\rm phase}\to0$, then every schedule $n_N\to\infty$ with
$\varepsilon_N^{\rm reg}\lfloor n_N/2\rfloor\to\infty$ gives the
full-sequence joint limit to the specified mixture, together with all fixed
finite-row marginal limits. The explicit schedule in
{prf:ref}`cor-slca-joint-stationary-time` instead gives the finite error
$N^{-1}+E_N^{\rm phase}$ directly. To reach population error at most
$\epsilon$, it suffices to choose $N$ with $E_N^{\rm phase}\le\epsilon/2$
and

$$
n\ge2\left\lceil
\frac{\log(2/\epsilon)}{-\log(1-\varepsilon_N^{\rm reg})}
\right\rceil.
$$

Every term is a displayed function of the swarm parameters, regional
moment bounds, phase concentration, scaled full-kernel transition intervals,
and finite matrix coefficients. The requirement that these inequalities
close is retained; in particular fixed-phase concentration is not implied
by finite-$N$ stationarity alone.
:::

:::{prf:proof}
Insert $Q_N=(L_N)_\#\Pi_N$ between the two population laws. The first
distance is at most the full-swarm TV mixing bound, since measurable
pushforward contracts TV and the population cost is bounded by one.
The second is bounded by {prf:ref}`thm-slcj-phase-weights`.
For tagged rows, first use full-swarm TV contraction under the coordinate
projection, then the stationary finite-row bound from the same theorem.
The inequality $(1-x)^m\le e^{-xm}$ proves convergence under the first
schedule. The specified $n_N$ and the accuracy inversion follow by taking
logarithms of the exact geometric mixing bound. These steps prove a joint
limit with a quantitative relation between the two parameters, rather than
exchanging limits without controlling that relation.
:::

(sec-slcj-limit-order)=
### 13.5. The order of limits and distinct phases

:::{prf:theorem} Distinct nonlinear phases obstruct a common exchange of limits
:label: thm-slcj-order-obstruction

Use the conservative canonical kernels $P_N$, the actual population map
$\mathcal F_h$, and the bounded transport metric $\mathsf d$ on capped
single-row laws. Let $\mathcal W$ be transport with cost $\mathsf d$ on
probabilities on those laws. Suppose the finite-$N$ Harris theorem applies
for each $N$, giving a unique invariant swarm law $\Pi_N$ and convergence
to it from the two initializations considered below. Write
$Q_N=(L_N)_\#\Pi_N$.

Suppose $\pi_1\ne\pi_2$ are two fixed population laws, with
$D=\mathsf d(\pi_1,\pi_2)>0$, and their independent-row initializations
satisfy the finite-horizon mean-field theorem. Let
$Q_{N,n}^{(i)}=\operatorname{Law}_{\pi_i^{\otimes N}}(L_N(S_n))$.
Then for every fixed $n$,

$$
\lim_{N\to\infty}\mathcal W(Q_{N,n}^{(i)},\delta_{\pi_i})=0,
\qquad
\lim_{n\to\infty}Q_{N,n}^{(i)}=Q_N\quad(N\text{ fixed}).
$$

Consequently the limits in the two orders cannot agree for both initial
phases. Quantitatively, for every $N$,

$$
\max_{i\in\{1,2\}}\mathcal W(Q_N,\delta_{\pi_i})\ge D/2.
$$

If $(Q_N)$ has a limit $Q$, the time-first limit is $Q$ for both
initializations, whereas the population-first limits are
$\delta_{\pi_1}$ and $\delta_{\pi_2}$. If $(Q_N)$ has no limit, its
convergent subsequences satisfy the same incompatibility.
:::

:::{prf:proof}
The population trajectory started at $\pi_i$ is constant. Finite-horizon
convergence in probability in the bounded metric implies convergence of
its expected distance to zero: for any $\varepsilon>0$, that expectation
is at most $\varepsilon+\Pr(\mathsf d>\varepsilon)$. Transport to a
point mass equals this expectation, proving the first limit. Total-variation
convergence of the swarm law implies convergence of its pushforward by
$L_N$, since inverse images preserve measurable events. This proves the
second limit and its independence of initialization.

For any probability $Q_N$ on population laws, the triangle inequality
holds pointwise as
$D\le\mathsf d(\pi_1,\nu)+\mathsf d(\nu,\pi_2)$.
Integrate against $Q_N$; each integral is transport to the corresponding
point mass. At least one is at least $D/2$. Passing to a convergent
subsequence preserves these inequalities, since the distances are bounded
continuous. Thus a common stationary limit cannot equal both point masses.
This is a conditional implication about actual distinct fixed phases and
certified finite-$N$ ergodicity, not an assertion that every multimodal
spatial landscape has multiple nonlinear fixed phases.
:::

:::{prf:remark} Which full long-time statement is being certified
:label: rem-slcj-target

A full long-time mean-field result must declare its target. Uniform
approximation to one deterministic phase, convergence to the set of
stationary phases, and convergence of stationary population-law mixtures
are distinct conclusions. A stationary mixture
$Q=\int\delta_\pi\,Q(d\pi)$ supported on
$\{\pi:\mathcal F_h\pi=\pi\}$ is consistent with a unique nonlinear
evolution from each initial law. Its $k$-row limit is
$\int\pi^{\otimes k}Q(d\pi)$, not generally the $k$-fold product of
$\int\pi Q(d\pi)$. Its weights need not preserve the initial phase
selection after finite-population transitions have had unbounded time
to occur. The preceding obstruction precludes replacing these targets
by one unqualified, order-independent deterministic limit.
:::

(sec-slcn-N-independent)=
## 14. Population-independent mechanisms and vanishing particle errors

(sec-slcn-keystone-population)=
### 14.1. The Keystone mechanism survives the population limit

:::{div} feynman-prose
The Keystone mechanism counts favorable measurement and donor events over a
complete geometric covering. Choosing the analysis scale from the current
centered error gives pressure of the form $k_{\rm key}W^p-E_{\max}/N^2$.
There is no fixed threshold below which the argument discards the error. The
population proof repeats the same coverage calculation with probability masses,
so its restoring coefficient survives without a particle-count penalty.

Pressure measures how strongly accepted updates act on the error; a signed
drift estimate must still determine their net effect after copying, collisions
and kinetics. That distinction also separates two kinds of nonzero remainder.
Configured physical noise can maintain a stationary cloud of positive width
even at infinite population. Finite-population sampling and self-exclusion errors
measure the discrepancy from that population evolution and should vanish with
$N$. A stationary cloud's width is not itself a failure of mean-field convergence.
:::

:::{prf:theorem} Zero-threshold Keystone pressure with an explicit population-size correction
:label: thm-slcn-keystone-power

Use the conservative all-alive canonical cloning kernel and the regional
parameter conventions of {prf:ref}`def-slc-keystone-constants`. In particular
$B_x>0$, $p_s>0$, $m_z>0$ and $L_R<\infty$. Retain all the displayed
primitive constants there, including $E_{\max}=16B_x^2$, and define

$$
\begin{aligned}
h_{\max}&=m_x\sqrt{E_{\max}/8},\\
k_t&=\frac{3m_x^2}{64s_*\sqrt{h_{\max}^2+\delta_D^2}},
\qquad k_\omega=m_fk_t,\\
k_a&=\min\left\{E_{\max}^{-1},
 \frac{A_-k_\omega}{2s_c(F_{\max}+\epsilon_c)}\right\},\\
k_r&=\begin{cases}
\min\{m_x/(4\sqrt{2E_{\max}}),A_-k_\omega/(2f_+L_A)\},&L_A>0,\\
m_x/(4\sqrt{2E_{\max}}),&L_A=0,
\end{cases}\\
c_0&=\kappa_C\kappa_D^2\frac{m_x^2}{8D_0^2}k_a,
\qquad K_M=E_{\max}+\frac{2B_f\sqrt{2d}}{k_r},\\
p&=5+4d,\qquad
k_{\rm key}=\frac{c_0}{2^{3+4d}E_{\max}^2K_M^{4d}}>0.
\end{aligned}
$$

For the actual entering comparison labels and centered errors in
{prf:ref}`thm-keystone-discharged-averaged-pressure`, write

$$
W_N=\frac1N\sum_i|\Delta\delta_{x,i}|^2,\qquad
\mathcal A_N=\frac1N\sum_i(p_{1,i}+p_{2,i})|\Delta\delta_{x,i}|^2.
$$

Then for every $N\ge1$ and every such pair of entering swarms,

$$
\boxed{\quad
\mathbb E[\mathcal A_N\mid S_1,S_2]
\ge k_{\rm key}W_N^{p}-\frac{E_{\max}}{N^2}.
\quad}
$$

For $N=1$ the stipulated no-distinct-live-donor convention makes
$\mathcal A_1=W_1=0$; the displayed inequality is interpreted through
that convention and does not invoke the $1/(N-1)$ companion formula.
All nontrivial companion and cluster calculations in its proof use
$N\ge2$.

All constants in the restoring term are independent of $N$. There is
no fixed analysis-threshold offset $W_0$ in this result.

There is also a direct population statement. Let $\mu_1,\mu_2$ be
single-row probabilities with positions in the same declared bounded
region, and let $\Gamma$ be any coupling of them. Set

$$
e(z_1,z_2)=|(x_1-\mu_1x)-(x_2-\mu_2x)|^2,\qquad
W=\int e\,d\Gamma.
$$

Let $\overline p_{\mu_s}(z_s)$ be the acceptance probability under the
actual marked population cloning law, averaged over its measurement and
donor marks, conditional on the recipient state. Then

$$
\boxed{\quad
\int[\overline p_{\mu_1}(z_1)+\overline p_{\mu_2}(z_2)]
                 e(z_1,z_2)\,d\Gamma
\ge 2k_{\rm key}W^p.
\quad}
$$

This is a population operator inequality obtained from the same favorable
measurement events and complete geometric coverage as the finite-particle
Keystone proof. It is not inferred from dividing a particle sum by $N$.
The exponent is a conservative explicit bound, not an optimal rate claim.
:::

:::{prf:proof}
For any auxiliary threshold $0<w\le E_{\max}$, the constants in the
regional recipe obey

$$
\rho_f\ge\frac{m_x^2w}{8D_0^2},\qquad
\omega_f\ge k_\omega w,\qquad a_0\ge k_aw,\qquad r_f\ge k_rw.
$$

For the second inequality, $h_f^2=m_x^2w/8$ and rationalization give

$$
\frac{\sqrt{h_f^2+\delta_D^2}-\sqrt{h_f^2/4+\delta_D^2}}{s_*}
=\frac{3h_f^2}{4s_*
 [\sqrt{h_f^2+\delta_D^2}+\sqrt{h_f^2/4+\delta_D^2}]}
\ge k_tw.
$$

The acceptance lower bound follows from $w\le E_{\max}$ and the
clipped-linear formula. Also $h_f/2=m_x\sqrt w/(4\sqrt2)
\ge m_xw/(4\sqrt{2E_{\max}})$; combine this with the reward-increment
branch of $r_f$ to obtain its bound. The lower bound on $\rho_f$ follows
by enlarging its denominator to $D_0^2$.
Thus

$$
C_0\ge c_0w^2,\qquad
M_f^2\le\left(1+\frac{2B_f\sqrt{2d}}{k_rw}\right)^{4d}
          \le(K_M/w)^{4d}.
$$

The admissibility condition $m_x^2w/4<D_0^2$ holds over this range:
$m_xB_x\le R_x/4$ follows from $t/(1+t)^2\le1/4$, whence
$m_x^2E_{\max}/4\le R_x^2/4<D_0^2$.

If $W_N=0$, the claim follows from nonnegativity. Otherwise set the
analysis threshold to $w=W_N/2$ after conditioning on the entering
states; it is not chosen using random measurement outcomes. The source
Keystone proof selects a spread swarm and, in the all-alive case,
its complete-coverage inequality implies

$$
\mathbb E\mathcal A_N
\ge\frac{C_0W_N^3}{2E_{\max}^2M_f^2}
          -\frac{C_0W_N}{N^2}.
$$

Indeed replace $N-1$ by $N$ in its favorable direction and use
$(a-b)_+^2\ge a^2/2-b^2$. Since $C_0\le1$ and $W_N\le E_{\max}$,
substitution of the bounds above with $w=W_N/2$ proves exactly the
stated $k_{\rm key}$ and finite-population correction. For $N=1$ both
centered configurations have zero error.

For the population proof, $W=0$ follows from nonnegativity. If $W>0$,
$W\le2(\operatorname{Var}_{\mu_1}x+
\operatorname{Var}_{\mu_2}x)$, so one marginal has positional variance
at least $W/4$. Squashing gives its feature variance at least $m_x^2W/4$.
Use the same threshold $w=W/2$ and the same finite feature partition.
The near-measurement/far-measurement event argument of
{prf:ref}`lem-keystone-near-neighbor-pressure` now has no self-exclusion:
for a recipient in cell $c$, its averaged acceptance is at least
$C_0\mu_s(c)^2$. This follows by integrating the independent recipient
and donor measurement kernels: the two near choices each have probability
at least $\kappa_D\mu_s(c)$ and $\kappa_C\mu_s(c)$, while the donor's
far measurement has probability at least $\kappa_D\rho_f$; shared
standardization retains the same fitness gap and acceptance lower bound.

Let $E_c=\int_{z_s\in c}e\,d\Gamma$. Then
$\sum_cE_c=W$ and $E_c\le E_{\max}\mu_s(c)$. Hence the pressure of
this marginal alone is at least

$$
C_0\sum_cE_c\mu_s(c)^2
\ge\frac{C_0}{E_{\max}^2}\sum_cE_c^3
\ge\frac{C_0W^3}{E_{\max}^2M_f^2}.
$$

The last inequality is the finite-sum convexity inequality for the cube.
Inserting the same $w$-dependent bounds gives $2k_{\rm key}W^p$.
The other marginal's pressure is nonnegative. This proves the limiting
operator inequality directly and retains arbitrary comparison couplings.
:::

(sec-slkd-signed-keystone)=
### 14.2. The signed donor balance that connects Keystone pressure to actual dispersion

:::{prf:theorem} Exact finite-particle positional-variance balance
:label: thm-slkd-finite-balance

For an all-alive entering swarm $S$, let
$\bar x=N^{-1}\sum_i x_i$ and
$W_N=N^{-1}\sum_i|x_i-\bar x|^2$. Condition on the actual sampled
fitness vector $\mathbf F$. Write

$$
b_{ij}=P_C(j\mid i,S)
 \min\left\{1,\frac{(F_j-F_i)_+}{s_c(F_i+\epsilon_c)}\right\},
\quad b_{ii}=0,\quad p_i=\sum_jb_{ij},
$$

For a singleton the canonical persistence rule sets $b_{11}=p_1=0$.
Define

$$
\begin{aligned}
t_i&=\sum_jb_{ij}(x_j-x_i),\quad
\bar t=N^{-1}\sum_i t_i,\quad\bar p=N^{-1}\sum_i p_i,\\
\sigma_i^2&=\sum_jb_{ij}|x_j-x_i|^2-|t_i|^2
                         +d\sigma_J^2p_i\ge0,\\
A_{\rm rec}&=N^{-1}\sum_i p_i|x_i-\bar x|^2,\quad
D_{\rm donor}=N^{-1}\sum_{i,j}b_{ij}|x_j-\bar x|^2.
\end{aligned}
$$

Then, for the actual copying, jitter and collision proposal,

$$
\boxed{\quad
\mathbb E[W_N(S^C)-W_N(S)\mid S,\mathbf F]
=-A_{\rm rec}+D_{\rm donor}+d\sigma_J^2\bar p
 -|\bar t|^2-\frac1{N^2}\sum_i\sigma_i^2.
\quad}
$$

Averaging this identity over the actual measurement vector gives the
unconditional cloning drift $H_x(S)$ of Chapter 3. All terms retain
their signs and the same frozen fitness and companion normalizers.
:::

:::{prf:proof}
Conditional on $S,\mathbf F$, copied positions of distinct rows use
independent cloning companions, acceptance coins and jitters. The mean
of row $i$ is $x_i+t_i$, and its variance trace is exactly $\sigma_i^2$.
Thus the output empirical barycenter has mean $\bar x+\bar t$ and
variance trace $N^{-2}\sum_i\sigma_i^2$. The empirical variance equals
$N^{-1}\sum_i|X_i^C-\bar x|^2-|\bar X^C-\bar x|^2$ identically.
A frozen copy from $i$ to $j$ changes the first term by
$N^{-1}(|x_j-\bar x|^2-|x_i-\bar x|^2)$, and accepted-row jitter
adds $N^{-1}d\sigma_J^2$. Summing and subtracting the barycenter's
second moment gives the displayed formula. Collision changes velocities
only. Finally average the conditional formula, including
$\mathbb E|\bar t(\mathbf F)|^2$; replacing this by
$|\mathbb E\bar t(\mathbf F)|^2$ would discard measurement-induced
center fluctuations and would not be the same identity.
:::

:::{prf:theorem} Exact population positional-variance balance
:label: thm-slkd-population-balance

Let $\mu$ be an admissible all-alive capped population law with finite
second positional moment, $m=\mu x$, $W(\mu)=\mu|x-m|^2$.
Let $\eta_\mu(dt)$ be its actual measurement-marked type law and let
$\beta_\mu(t,u)$ be its actual accepted outgoing-edge density,

$$
\beta_\mu(t,u)=
\frac{w_C(z_t,z_u)}{\int w_C(z_t,z')\,\mu(dz')}
\min\left\{1,\frac{(F_u-F_t)_+}{s_c(F_t+\epsilon_c)}\right\}.
$$

Define explicitly

$$
\begin{aligned}
A_{\rm rec}(\mu)&=\iint\beta_\mu(t,u)|x_t-m|^2
                       \,\eta_\mu(dt)\eta_\mu(du),\\
D_{\rm donor}(\mu)&=\iint\beta_\mu(t,u)|x_u-m|^2
                       \,\eta_\mu(dt)\eta_\mu(du),\\
\bar t(\mu)&=\iint\beta_\mu(t,u)(x_u-x_t)
                       \,\eta_\mu(dt)\eta_\mu(du),\\
\bar p(\mu)&=\iint\beta_\mu(t,u)\,\eta_\mu(dt)\eta_\mu(du).
\end{aligned}
$$

The actual rooted copying and collision law $\mathcal J(\mu)$ satisfies

$$
\boxed{\quad
W(\mathcal J(\mu))-W(\mu)
=-A_{\rm rec}(\mu)+D_{\rm donor}(\mu)
 +d\sigma_J^2\bar p(\mu)-|\bar t(\mu)|^2.
\quad}
$$

There is no $N^{-1}$ or $N^{-2}$ factor in the recipient, donor or
center-shift terms. The population law has a deterministic barycenter;
it therefore has no empirical barycenter sampling-variance correction.
:::

:::{prf:proof}
The actual root either retains $x_t$ or copies frozen donor $x_u$ with
subprobability density $\beta_\mu(t,u)\eta_\mu(du)$. Thus its mean
moves from $m$ to $m+\bar t$, while its second moment about $m$ changes
by $-A_{\rm rec}+D_{\rm donor}+d\sigma_J^2\bar p$. Subtract the
squared displacement $|\bar t|^2$ of its new mean. Incoming component
edges and the component rotation do not change this root position rule.
All integrals are absolutely finite since $\beta_\mu\le1/\kappa_C$
and $\mu$ has a finite second moment. This derives the population
identity directly from the rooted operator, rather than by dropping
finite-$N$ factors in a particle formula.
:::

:::{prf:corollary} The Keystone recipient pressure is a single-population quantity
:label: cor-slkd-one-population-pressure

Use the bounded-region constants $k_{\rm key}>0$, $E_{\max}$ and
$p=5+4d$ in {prf:ref}`thm-slcn-keystone-power`, with active diversity
exponent $p_s>0$. Choose the containing region to be a convex ball,
so it also contains the entering positional mean. Then

$$
\mathbb E_{\mathbf F}A_{\rm rec}(S)
\ge k_{\rm key}W_N(S)^p-\frac{E_{\max}}{N^2},\qquad
A_{\rm rec}(\mu)\ge2k_{\rm key}W(\mu)^p.
$$

Every constant in the positive pressure term is the already explicit
landscape and algorithm constant of that theorem and is independent of
$N$. The finite-particle correction vanishes as $N\to\infty$.
:::

:::{prf:proof}
For the comparison population, put every walker at the original
population's positional mean and at the same capped velocity, for
example zero. Its rewards are identical, every measured comparison
separation equals the configured distance floor, and every standardized
reward and diversity is zero. Therefore every frozen fitness is
identical and every accepted live-cloning probability is zero.
The centered positional discrepancy from this collapsed population is
exactly $x_i-\bar x$; its average squared norm is $W_N$. Apply
{prf:ref}`thm-slcn-keystone-power` to this pair to obtain the first
bound. For a population law compare $\mu$ with the point mass at
$(\mu x,0)$ and use the direct population part of that theorem. The
comparison point lies in the declared containing ball, so the same
regional reward and feature constants apply. The comparison is a device
for evaluating one operator inequality; it does not assert that this
collapsed law is invariant or that the two trajectories synchronize.
:::

:::{prf:definition} Signed donor excess and structural pair certificates
:label: def-slkd-donor-excess

Fix a declared retained-pressure fraction $0<\theta\le1$. For either
particle or population quantities above, set

$$
\Gamma_\theta=D_{\rm donor}-(1-\theta)A_{\rm rec}.
$$

This is an explicitly specified signed donor integral, not a convergence
constant. The exact copying drift is the sum of
$-\theta A_{\rm rec}$, $\Gamma_\theta$, jitter, and the negative
center terms already displayed.

For a declared partition into basin, transition and exterior regions
$(A_a)_a$, and the entering mean $m$, supply radial-square intervals

$$
e_a^-\le |x-m|^2\le e_a^+\quad(x\in A_a),\qquad
0\le e_a^-<\infty,\qquad e_a^-\le e_a^+\le\infty.
$$

Let $B_{ab}$ be the actual accepted-edge mass from region $a$ to region
$b$, normalized by $1/N$ for particles and by the marked root law for
the population operator. Put

$$
g_{ab}(\theta)=e_b^+-(1-\theta)e_a^-.
$$

Whenever the displayed signed sum is absolutely convergent, the exact signed
donor excess obeys

$$
\Gamma_\theta\le\sum_{a,b}B_{ab}g_{ab}(\theta).
$$

For fully explicit upper and lower bounds on $B_{ab}$, let the actual
fitness on each region satisfy $F_a^-\le F\le F_a^+$ for every
measurement assignment (or almost surely under the population marked
law). Define

$$
\alpha_{ab}^+=\min\left\{1,
 \frac{(F_b^+-F_a^-)_+}{s_c(F_a^-+\epsilon_c)}\right\},\qquad
\alpha_{ab}^-=\min\left\{1,
 \frac{(F_b^--F_a^+)_+}{s_c(F_a^++\epsilon_c)}\right\}.
$$

For a finite swarm with $N\ge2$ and region counts $n_a$, set

$$
\begin{aligned}
L_{ab}^{N}&=\frac{\kappa_C\alpha_{ab}^-
 n_a(n_b-\mathbf1_{a=b})}{N(N-1)},\\
U_{ab}^{N}&=\min\left\{\frac{n_a}N,
 \frac{\alpha_{ab}^+n_a(n_b-\mathbf1_{a=b})}
 {N\kappa_C(N-1)}\right\}.
\end{aligned}
$$

For population region masses $m_a=\mu(A_a)$, set

$$
L_{ab}^{\infty}=\kappa_C\alpha_{ab}^-m_am_b,
\qquad
U_{ab}^{\infty}=\min\{m_a,\alpha_{ab}^+m_am_b/\kappa_C\}.
$$

A singleton has no accepted live edge, so its block masses and donor
excess are zero without using a denominator $N-1$.
Require the actual positive fitness bands to satisfy
$F_a^-+\epsilon_c>0$. Blocks with zero accepted-edge upper bound contribute
zero, including when their geometric upper bound is infinite; other infinite
positive contributions give an uninformative infinite bound.
Then $L_{ab}\le B_{ab}\le U_{ab}$ and the signed numerical upper
certificate is

$$
\boxed{\quad
\Gamma_\theta\le\mathcal G_\theta:=
\sum_{g_{ab}\ge0}U_{ab}g_{ab}
 +\sum_{g_{ab}<0}L_{ab}g_{ab}.
\quad}
$$

The fitness bands are obtained from the actual reward normalization and
sampled-diversity bounds as in {prf:ref}`def-slcr-flux`; thus the formula
retains companion bandwidths, feature radii, floors, amplitudes,
exponents, acceptance saturation and the declared landscape profiles.
Refining the spatial partition can improve this certificate without
changing the algorithm. Infinite profiles report an uninformative
bound and can instead be localized with the established excursion terms.
:::

:::{prf:proof}
On an edge from $a$ to $b$, its contribution to $\Gamma_\theta$ is
$|x_j-m|^2-(1-\theta)|x_i-m|^2\le g_{ab}$.
Summing the actual nonnegative edge masses proves the first inequality.
The acceptance gate is bounded between $\alpha_{ab}^-$ and
$\alpha_{ab}^+$. Finite-particle companion probabilities lie between
$\kappa_C/(N-1)$ and $1/[\kappa_C(N-1)]$; there are
$n_a(n_b-\mathbf1_{a=b})$ eligible ordered pairs in the block.
Every recipient has total acceptance mass at most one, giving the
additional upper bound $n_a/N$. In the population law the normalized
donor density lies between $\kappa_C$ and $1/\kappa_C$, giving the
corresponding integral bounds. For nonnegative $g_{ab}$ use the upper
edge-mass bound; for negative $g_{ab}$ use the lower bound, reversing
that scalar inequality. This proves $\mathcal G_\theta$ with all signs
retained. The bounds remain valid after averaging shared measurement
normalizers because the fitness bands were imposed on the actual
measurement assignments.
:::

:::{prf:theorem} Signed Keystone drift for the actual cloning stage
:label: thm-slkd-signed-cloning

Under the preceding hypotheses, the actual finite-particle cloning drift
satisfies

$$
\begin{aligned}
H_x(S)\le{}&-\theta k_{\rm key}W_N^p
 +\frac{\theta E_{\max}}{N^2}
 +\mathbb E_{\mathbf F}\Gamma_\theta
 +d\sigma_J^2\mathbb E_{\mathbf F}\bar p\\
&-\mathbb E_{\mathbf F}|\bar t|^2
 -\frac1{N^2}\mathbb E_{\mathbf F}\sum_i\sigma_i^2.
\end{aligned}
$$

For the actual population cloning law,

$$
W(\mathcal J\mu)-W(\mu)
\le-2\theta k_{\rm key}W(\mu)^p
 +\Gamma_\theta(\mu)+d\sigma_J^2\bar p(\mu)-|\bar t(\mu)|^2.
$$

One may replace $\Gamma_\theta$ by its proved structural upper bound
$\mathcal G_\theta$, or retain the exact signed integral. Thus the
Keystone restoring term survives in the actual population operator.
Its net effect is decided by computed donor destinations and noise,
not by its recipient pressure alone.
:::

:::{prf:proof}
Use the exact algebra
$-A_{\rm rec}+D_{\rm donor}=-\theta A_{\rm rec}+\Gamma_\theta$
in the two proved variance identities, then apply the corresponding
Keystone lower bound to $A_{\rm rec}$. Since $\theta>0$, the signs
are as displayed. No estimate on the positive donor term is suppressed;
its optional replacement follows from the preceding structural bound.
:::

:::{div} feynman-prose
Compare two populations using the same source label whenever their actual
copying probabilities permit it. When both copies accept that source, give
them the same jitter. The jitter then disappears from their difference;
its cost remains only where the acceptance decisions disagree. The formulas
below measure that disagreement using the retained fitnesses, companion
weights and regional reward oscillations.

Now subtract the moving population centers. This removes the complete mean
square of their displacement, including its random part. Keeping this
negative term preserves a contribution that an uncentered estimate would
lose. Finally compare the kinetic update with the same update at the actual
stationary phase. Their common Gaussian variance cancels. What remains is
an explicit combination of donor transfers, mismatches and force increments,
with every contribution tied to the corresponding population difference.
:::

:::{prf:proposition} Common-source donor flux with diagonal-vanishing mismatch cost
:label: prop-cloning-common-source-signed-estimate

Consider two all-alive canonical input swarms with $N\ge2$, paired
labels, and their complete frozen fitness vectors $F_i,\widetilde F_i$.
All expectations below condition on these data. Set
$b_{ij}=P_C(j\mid i)a(F_i,F_j)$ and
$\widetilde b_{ij}=\widetilde P_C(j\mid i)a(\widetilde F_i,\widetilde F_j)$
for $j\ne i$, and $b_{ii}=\widetilde b_{ii}=0$. Define
$$
p_i=\sum_jb_{ij},\quad\widetilde p_i=\sum_j\widetilde b_{ij},\quad
q_{ij}=(1-p_i)\mathbf1_{j=i}+b_{ij},\quad
\widetilde q_{ij}=(1-\widetilde p_i)\mathbf1_{j=i}+\widetilde b_{ij},
$$
$$
c_{ij}=\min(b_{ij},\widetilde b_{ij}),\quad c_i=\sum_jc_{ij},\quad
\ell_i=\sum_j|b_{ij}-\widetilde b_{ij}|,\quad
\lambda_{ij}=\min(q_{ij},\widetilde q_{ij}),\quad
 t_i=1-\sum_j\lambda_{ij}.
$$
Then
$$
c_i=\frac{p_i+\widetilde p_i-\ell_i}{2},\qquad
t_i=\frac{|p_i-\widetilde p_i|+\ell_i}{2}\le\min(1,\ell_i).
$$
Couple the two source labels in row $i$ by mass $\lambda_{ij}$ on
$(j,j)$ and, when $t_i>0$, residual mass
$$
\gamma_{i,jk}=
\frac{(q_{ij}-\lambda_{ij})(\widetilde q_{ik}-\lambda_{ik})}{t_i}
$$
on $(j,k)$. Set all residual masses zero if $t_i=0$. Use independent
copies of this coupling across rows and the same row Gaussian jitter in
both swarms. This preserves the actual independent row donor/gate laws
conditional on the frozen fitnesses.

Put $d_i=x_i-y_i$, $\bar d=N^{-1}\sum_i d_i$,
$e_i=|d_i-\bar d|^2$, $D=N^{-1}\sum_i e_i$, and
$$
 h_i=\sum_jq_{ij}x_j-\sum_k\widetilde q_{ik}y_k-d_i,
 \qquad \bar h=N^{-1}\sum_i h_i.
$$
For the paired centered discrepancy $D'$ after cloning positions,
let $V_i$ be the conditional variance trace of
$x_{J_i}-y_{K_i}+\sigma_J(\mathbf1_{J_i\ne i}-\mathbf1_{K_i\ne i})\xi_i$.
Explicitly, with $\Pi_i(j,k)=\lambda_{ij}\mathbf1_{j=k}+\gamma_{i,jk}$,
$$
V_i=\sum_{j,k}\Pi_i(j,k)|x_j-y_k|^2
 +d\sigma_J^2\sum_{j,k}\gamma_{i,jk}
 (\mathbf1_{j\ne i}-\mathbf1_{k\ne i})^2-|d_i+h_i|^2.
$$

Then the exact identity is
$$
\begin{aligned}
\mathbb E D'-D={}&\frac1N\sum_{i,j}c_{ij}(e_j-e_i)\\
&+\frac1N\sum_{i,j,k}\gamma_{i,jk}
  \bigl(|x_j-y_k-\bar d|^2-e_i\bigr)\\
&+\frac{d\sigma_J^2}{N}\sum_{i,j,k}\gamma_{i,jk}
  (\mathbf1_{j\ne i}-\mathbf1_{k\ne i})^2
 -|\bar h|^2-\frac1{N^2}\sum_i V_i .
\end{aligned}                                                   \tag{C.S1}
$$
In particular, for any explicit bounds
$R_i^2\ge\max_{j,k:\gamma_{i,jk}>0}|x_j-y_k-\bar d|^2$,
$$
\begin{aligned}
\mathbb E D'-D\le{}&
 -\frac1{2N}\sum_i(p_i+\widetilde p_i)e_i
 +\frac1N\sum_{i,j}c_{ij}e_j\\
&+\frac1{2N}\sum_i\ell_i e_i
 +\frac1N\sum_it_i(R_i^2+d\sigma_J^2)
 -|\bar h|^2-\frac1{N^2}\sum_iV_i .
\end{aligned}                                                   \tag{C.S2}
$$
A finite empirical maximum gives an exact admissible $R_i$; declared
basin/transition/tail diameter envelopes can replace it. Thus mismatch
costs vanish when the paired configurations and retained fitnesses agree;
they are not replaced by a nonzero physical noise floor. Neither formula
claims that the common donor insertion is absent.

For an explicit structural estimate of that insertion and its signed
recipient counterpart, partition the paired labels into clusters $H$,
choose numbers $e_H$ and $\rho_H\ge0$ with
$|e_i-e_H|\le\rho_H$ for $i\in H$, and define
$$
C_{HL}=\frac1N\sum_{i\in H,j\in L}c_{ij},\qquad
E_{HL}=\frac1N\sum_{i\in H,j\in L}|b_{ij}-\widetilde b_{ij}|.
$$
For these *label* clusters, the upper edge bound can be computed without
identifying them with the spatial regions of
{prf:ref}`def-slkd-donor-excess`. In swarm $a=1,2$ choose actual retained
fitness bands $F_{a,H}^-\le F_{a,i}\le F_{a,H}^+$ on each $H$ and set

$$
\alpha_{a,HL}^+=\min\left\{1,
 \frac{(F_{a,L}^+-F_{a,H}^-)_+}
 {s_c(F_{a,H}^-+\epsilon_c)}\right\},\qquad
U_{a,HL}=\min\left\{\frac{|H|}{N},
 \frac{\alpha_{a,HL}^+|H|(|L|-\mathbf1_{H=L})}
      {N\kappa_C(N-1)}\right\}.
$$

Use $U_{HL}=\min(U_{1,HL},U_{2,HL})$ in the bound below.
These formulas are the same accepted-edge calculation as
{prf:ref}`def-slkd-donor-excess`, now applied to the actual paired label
clusters; they also cover within-cluster edges through the self-exclusion
factor. Let $\mathcal L^1_{HL}$ and
$\mathcal L^2_{HL}$ be the signed retained-fitness lower bounds
(3.S1) or (3.S2), evaluated separately in the two actual swarms on
these same label clusters. Then, with each unordered pair oriented
by $e_H\ge e_L$,
$$
C_{HL}-C_{LH}\ge
\frac{\mathcal L^1_{HL}+\mathcal L^2_{HL}}2-\frac{E_{HL}}2,
$$
$$
\frac1N\sum_{i,j}c_{ij}(e_j-e_i)
\le-\sum_{\{H,L\}}(e_H-e_L)
 \left[\frac{\mathcal L^1_{HL}+\mathcal L^2_{HL}-E_{HL}}2\right]
 +\sum_{H,L}U_{HL}(\rho_H+\rho_L).                 \tag{C.S3}
$$
This evaluates the common donor term using retained fitness gaps,
within-cluster fitness variance, companion weights and their actual
normalizers. The finite sum on the right, together with the mismatch
and negative barycenter terms in (C.S1), is a numerical signed bound;
a positive total decrement is obtained precisely when that evaluated
upper bound is negative. There is no unknown optimal convergence
constant in this test.

The mismatch itself has an explicit primitive-parameter estimate. Put
$$
F_* =\eta_r^{p_r}\eta_s^{p_s},\quad
F^*=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},\quad
L_{\rm rec}=\frac{F^*+\epsilon_c}{s_c(F_*+\epsilon_c)^2},\quad
L_{\rm don}=\frac1{s_c(F_*+\epsilon_c)}.
$$
With $\kappa_C=\exp[-D_*^2/(2\epsilon_C^2)]$ and actual weights
$w_{ij}=\exp[-D(z_i,z_j)^2/(2\epsilon_C^2)]$,
$$
\ell_i\le\min\left\{2,
 \frac{2}{\kappa_C(N-1)}\sum_{j\ne i}|w_{ij}-\widetilde w_{ij}|
 +L_{\rm rec}|F_i-\widetilde F_i|
 +L_{\rm don}\sum_{j\ne i}\widetilde P_C(j\mid i)
                          |F_j-\widetilde F_j|\right\}.           \tag{C.S4}
$$
All fitness differences here are the actual logistic-power formulas,
including their respective retained reward/diversity means and variance
floors. If desired the already derived fitness-normalization estimates
can be substituted; averaging fitness before applying the gate is not
permitted. The constants in (C.S4) do not grow with $N$.
:::

:::{prf:proof}
The identity $\min(a,b)=(a+b-|a-b|)/2$ gives the formulas for $c_i$
and $t_i$. The residual marginals each have mass $t_i$, so their product
divided by $t_i$ gives a coupling. Their supports are disjoint on the
diagonal. Hence common-label choices use the same acceptance indicator,
and only residual pairs can produce unmatched jitter.

Compute the post-update average squared discrepancy about the *entering*
center difference $\bar d$. A common persisting label contributes $e_i$;
a common accepted label $j$ contributes $e_j$. Residual pairs contribute
the displayed cross-source square. Centered independent Gaussian jitter
adds exactly its displayed mismatch second moment. The expected new
center difference minus $\bar d$ is $\bar h$. Independence across
conditional rows makes the variance trace of that mean
$N^{-2}\sum_iV_i$. Subtracting its complete second moment gives (C.S1).
This subtraction includes donor randomness and jitter; it is not merely
a deterministic barycenter correction. Drop only the nonpositive
residual term $-N^{-1}\sum_it_ie_i$, bound the residual cross squares
and mismatch probability by $t_iR_i^2$ and $t_i$, and substitute the
formula for $c_i$ to get (C.S2).

On a cluster block, the common flux differs from
$C_{HL}(e_L-e_H)$ by at most $C_{HL}(\rho_H+\rho_L)$.
Also
$$
C_{HL}-C_{LH}
=\tfrac12[(B^1_{HL}-B^1_{LH})+(B^2_{HL}-B^2_{LH})
                 -E_{HL}+E_{LH}].
$$
Discard $E_{LH}\ge0$ and apply the two actual signed-fitness bounds.
Each live recipient has accepted mass at most one, while each eligible
donor has probability at most $1/[\kappa_C(N-1)]$ and acceptance at most
$\alpha_{a,HL}^+$; hence $C_{HL}\le U_{a,HL}$ in both swarms.
Pair the two directed blocks and use
$e_H-e_L\ge0$ to obtain (C.S3).

For (C.S4), normalized positive weights satisfy
$\sum_j|P_j-\widetilde P_j|\le2\sum_j|w_j-\widetilde w_j|/Z$
with $Z\ge\kappa_C(N-1)$. The clipped positive-part gate is globally
Lipschitz on $[F_*,F^*]^2$ with recipient and donor constants
$L_{\rm rec},L_{\rm don}$: on its unsaturated positive branch these
bound the two partial derivatives, and clipping preserves the bound
across both junctions. Split
$P_ja_j-\widetilde P_j\widetilde a_j$
as $(P_j-\widetilde P_j)a_j+\widetilde P_j(a_j-\widetilde a_j)$,
sum absolute values, and use $0\le a_j\le1$.
Finally each accepted row measure has mass at most one, so $\ell_i\le2$.
:::

:::{prf:corollary} Regional evaluation of the barycenter subtraction
:label: cor-slc-common-source-barycenter

For a completely regional bound on the negative barycenter term in
(C.S1), let $\Pi_i(j,k)=\lambda_{ij}\mathbf1_{j=k}+\gamma_{i,jk}$.
For each coordinate $r=1,\ldots,d$, use declared regional coordinate
intervals
$L_{i,jk,r}\le(x_j-y_k-\bar d)_r\le U_{i,jk,r}$ and set
$$
 l_r=\frac1N\sum_{i,j,k}\Pi_i(j,k)L_{i,jk,r},\qquad
 u_r=\frac1N\sum_{i,j,k}\Pi_i(j,k)U_{i,jk,r}.
$$
The zero-mean jitter does not alter this mean. Consequently
$$
 -|\bar h|^2\le
 -\sum_{r=1}^d\operatorname{dist}(0,[l_r,u_r])^2,
\qquad
\operatorname{dist}(0,[l,u])=\max\{l,-u,0\}.
$$
Indeed $\bar h$ is the average expected centered output difference,
so its $r$th coordinate lies in $[l_r,u_r]$. Squaring the coordinate
lower bounds and summing proves the claim. For singleton cells the
coordinate bounds are equalities and recover the exact subtraction.

:::

:::{prf:lemma} Explicit normalization and weight terms in the common-source bound
:label: lem-slc-common-source-normalizers

Use the two all-alive retained arrays in
{prf:ref}`prop-cloning-common-source-signed-estimate`. Write their raw
reward and sampled-diversity entries as $r_i,s_i$ and
$\widetilde r_i,\widetilde s_i$. Suppose the union of the two reward
ranges has length at most $R_r$, and the union of the two diversity
ranges has length at most $R_s$. These are declared regional oscillation
bounds or the actual retained ranges. Put
$$
\delta_{b,i}=|b_i-\widetilde b_i|,\qquad
\overline\delta_b=N^{-1}\sum_i\delta_{b,i},\qquad b\in\{r,s\},
$$
$$
H_b=\frac{A_bp_b}{4}
 \max\{\eta_b^{p_b-1},(A_b+\eta_b)^{p_b-1}\}
 (A_{b'}+\eta_{b'})^{p_{b'}},\qquad b'\ne b,
$$
with $H_b=0$ if $p_b=0$. The actual fitness difference satisfies
$$
|F_i-\widetilde F_i|
\le\sum_{b\in\{r,s\}}H_b\left[
 \frac{\delta_{b,i}}{\sigma_b}
 +\left(\frac1{\sigma_b}+\frac{R_b^2}{\sigma_b^3}\right)
                         \overline\delta_b\right].
$$
The companion weights in (C.S4) obey
$$
|w_{ij}-\widetilde w_{ij}|
\le\frac{|D(z_i,z_j)-D(\widetilde z_i,\widetilde z_j)|}
             {\epsilon_C\sqrt e}.
$$
Thus (C.S4) has a bound expressed entirely through the actual retained
raw arrays, regional oscillations, metric increments and primitive
algorithm parameters. No expectation is taken before a gate is evaluated.

*Proof.* For a raw array $b$, its empirical variance is
$(2N^2)^{-1}\sum_{i,j}(b_i-b_j)^2$. Since both pair differences have
absolute value at most $R_b$, subtracting these expressions gives
$|\operatorname{Var}_N(b)-\operatorname{Var}_N(\widetilde b)|
\le2R_b\overline\delta_b$. The derivative of
$(u+\sigma_b^2)^{-1/2}$ on $u\ge0$ has absolute value at most
$1/(2\sigma_b^3)$. Subtract the two standardized entries, using
$|\widetilde b_i-\overline{\widetilde b}|\le R_b$ and
$|\overline b-\overline{\widetilde b}|\le\overline\delta_b$.
This proves the bracketed estimate. The logistic derivative is at most
$A_b/4$; differentiating its positive power and the product gives $H_b$.
The two-coordinate mean-value bound proves the fitness estimate.
Finally the maximum of $u\exp[-u^2/(2\epsilon_C^2)]/\epsilon_C^2$
for $u\ge0$ is $1/(\epsilon_C\sqrt e)$. Apply the scalar mean-value
inequality to the two actual algorithmic distances. $\square$
:::

:::{prf:lemma} Common-source mismatch on unbounded position domains
:label: lem-slc-common-source-tails

In {prf:ref}`prop-cloning-common-source-signed-estimate`, let
$\overline t=N^{-1}\sum_i t_i$, and for any $p>2$ set
$$
M_{x,p}=N^{-1}\sum_i|x_i-\overline x|^p,\qquad
M_{y,p}=N^{-1}\sum_i|y_i-\overline y|^p,
$$
$$
a_* =\min\{1,(F^*-F_*)/[s_c(F_*+\epsilon_c)]\},\qquad
\mathcal M_p=2^{p-1}(1+a_*/\kappa_C)(M_{x,p}+M_{y,p}).
$$
Then the entire residual cross-source contribution is bounded by
$$
\frac1N\sum_{i,j,k}\gamma_{i,jk}|x_j-y_k-\bar d|^2
\le \mathcal M_p^{2/p}\overline t^{1-2/p}.
$$
Consequently the term $N^{-1}\sum_i t_i(R_i^2+d\sigma_J^2)$ in
(C.S2) may be replaced by
$\mathcal M_p^{2/p}\overline t^{1-2/p}+d\sigma_J^2\overline t$.
This bound uses moments, not a bound on the most distant walker. Its
coefficient is uniform in population size whenever the stated centered
moment budgets are uniform. It vanishes at zero mismatch; when
$\overline t=0$ the integral is zero.

*Proof.* The measure assigning mass $\gamma_{i,jk}/N$ to each residual
triple has total mass $\overline t$. Each of its source marginals is
bounded by the corresponding complete row-source marginal. For the first
swarm, the normalized incoming accepted mass at any donor index $j$ is
at most $a_*/(N\kappa_C)$, because
$b_{ij}\le a_*/[\kappa_C(N-1)]$ and there are $N-1$ possible recipients.
The persisting contribution is at most $1/N$. Therefore its averaged
source $p$th moment about $\bar x$ is at most
$(1+a_*/\kappa_C)M_{x,p}$. The second swarm has the analogous bound.
Use $|u-v|^p\le2^{p-1}(|u|^p+|v|^p)$ to bound the residual $p$th
moment by $\mathcal M_p$, then apply Hölder on this subprobability
measure. No independence among source labels in different swarms is
needed. $\square$
:::

:::{prf:theorem} Exact full-update positional bridge through the kinetic stage
:label: thm-slkd-full-position

Require finite prepared second moments of position, velocity and force,
under the conditional preparation law for each entering swarm and under
$\mathcal J(\mu)$ for the population identity. The actual linear force-growth
profile and a finite prepared second moment suffice. These conditions make
every covariance and square term below absolutely integrable.
For a prepared finite swarm $S^C$, let $\operatorname{Var}_N$ and
$\operatorname{Cov}_N(U,V)=N^{-1}\sum_i(U_i-\bar U)\cdot(V_i-\bar V)$
use its empirical centering. Define

$$
\begin{aligned}
\mathcal K_N(S^C)={}&2B\operatorname{Cov}_N(X,V)
 +2\eta\operatorname{Cov}_N(X,F(X))
 +B^2\operatorname{Var}_N(V)\\
&+2B\eta\operatorname{Cov}_N(V,F(X))
 +\eta^2\operatorname{Var}_N(F(X)),
\end{aligned}
$$

with $B=c(1+a)$, $\eta=c^2(1+a)$ and
$\tau^2=c^2q^2+s^2$ for the actual BAOAB and final position noise.
Then the complete finite update has the exact identity

$$
\mathbb E[W_N(S_1)-W_N(S)\mid S]
=H_x(S)+\mathbb E[\mathcal K_N(S^C)\mid S]
 +(1-N^{-1})d\tau^2.
$$

For $\rho=\mathcal J(\mu)$, define $\mathcal K(\rho)$ by the same
formula with population variances and covariances. Then

$$
W(\mathcal F_h\mu)-W(\mu)
=W(\mathcal J\mu)-W(\mu)+\mathcal K(\mathcal J\mu)+d\tau^2.
$$

Consequently substitution of {prf:ref}`thm-slkd-signed-cloning` gives
the full-update negative term $-\theta k_{\rm key}W_N^p$ or
$-2\theta k_{\rm key}W(\mu)^p$, with the explicitly displayed signed
donor, center, collision-preparation, force and noise terms retained.
Every force covariance can also be evaluated from pairwise regional
increments through the exact identity

$$
\operatorname{Cov}_\rho(X,F(X))=
\frac12\iint(x-y)\cdot(F(x)-F(y))\,\rho(dxdv)\rho(dydw),
$$

and the analogous empirical and force-variance identities. The established
regional increment and excursion estimates apply to these actual
intermediate positions.
:::

:::{prf:proof}
Conditional on preparation, the exact completed position of row $i$ is
$X_i+BV_i+\eta F(X_i)+cq\xi_i+s\zeta_i$.
Expand the centered empirical square of its deterministic part. Subtract
$\operatorname{Var}_N(X)$; the five cross and square terms left are
exactly $\mathcal K_N$. Independent centered row noises add total
variance $d\tau^2$ to the average square and $d\tau^2/N$ to the
barycenter square. Their difference is the displayed factor
$(1-N^{-1})d\tau^2$. Average preparation and use the cloning identity.
For a population law, the root's centered Gaussian noise has deterministic
zero mean and variance $d\tau^2$, with no empirical center correction;
the same expansion gives its formula. For the pair identity, expand
$(x-y)\cdot(F(x)-F(y))$ under two independent copies of $\rho$ and
collect the two identical centered covariance terms. This use of a
product measure is an identity for a fixed population law, not an
assumption that finite interacting walkers are independent.
:::

:::{prf:theorem} Structural full-update variance drift with an evaluated threshold
:label: thm-slkd-structural-variance-threshold

Use the unchanged all-alive cloning, collision and BAOAB kernel of
{prf:ref}`thm-slkd-full-position`. This statement applies to either an
entering finite swarm with $N\ge2$ or its actual population operator.
Let $W$ be the entering centered positional variance and $W_C$ the
variance after copying, jitter and component collision. Suppose a
basin, passage or exterior phase class supplies the following pair
profiles for every prepared positional law reached from that class:

$$
(x-y)\cdot(F(x)-F(y))\le-m|x-y|^2+b,\qquad
|F(x)-F(y)|\le L|x-y|+J,                              \tag{SLKD.T1}
$$

where $m\in\mathbb R$ and $b,L,J\ge0$ are declared structural numbers.
The inequalities need only hold almost everywhere under the actual
prepared pair law. Regional exceptions may be included by integrating
positive excesses: for every such prepared law $\rho$, it is enough
to check the two explicit pair integrals

$$
\iint[(x-y)\cdot(F(x)-F(y))+m|x-y|^2]_+\,d\rho^{\otimes2}\le b,
\qquad
\left\|(|F(x)-F(y)|-L|x-y|)_+\right\|_{L^2(\rho^{\otimes2})}\le J.
$$

These integrals include prepared jitter excursions under their actual
law. The OU Gaussian enters the completed position through the exact
$d(c^2q^2+s^2)$ term; its second-force excursion affects velocity and
the next step, not the current completed position. Infinite values make this certificate
uninformative. For a finite region table with pointwise pair constants
$(m_{ab},b_{ab},L_{ab},J_{ab})$, the fully explicit coarse substitution is
$m=\min_{ab}m_{ab}$, $b=\max_{ab}b_{ab}$,
$L=\max_{ab}L_{ab}$ and $J=\max_{ab}J_{ab}$ over prepared pair labels.
The displayed pair integrals instead retain the actual regional masses
and are preferable when an exterior pointwise maximum is infinite.
Retain the bounded-region Keystone premises of
{prf:ref}`thm-slcn-keystone-power` for the entering phase. Define,
with the actual algorithm parameters,

$$
\begin{gathered}
V_c=(1+2|\alpha_{\rm col}|)V_{\max},\quad
A_0=-2\eta m+\eta^2L^2,\quad
C_1=2BV_c+2B\eta V_cL+\sqrt2\eta^2LJ,\\
C_0=\eta b+B^2V_c^2+\sqrt2 B\eta V_cJ+\tfrac12\eta^2J^2,\\
a_t=[1+A_0+t]_+,\qquad
d_t=C_0+C_1^2/(4t)+d(c^2q^2+s^2),\qquad t>0.
\end{gathered}                                                     \tag{SLKD.T2}
$$

These constants have no population-size factor. The complete update
satisfies

$$
\boxed{\begin{aligned}
\mathbb E[W_N(S^+)|S]
&\le a_t\left[W_N(S)-\theta k_{\rm key}W_N(S)^p
 +\mathbb E_{\mathbf F}\Gamma_\theta
 +d\sigma_J^2\mathbb E_{\mathbf F}\bar p
 +\frac{\theta E_{\max}}{N^2}\right]+d_t,\\
W(\mathcal F_h\mu)
&\le a_t\left[W(\mu)-2\theta k_{\rm key}W(\mu)^p
 +\Gamma_\theta(\mu)+d\sigma_J^2\bar p(\mu)\right]+d_t.
\end{aligned}}                                                     \tag{SLKD.T3}
$$

Here $p=5+4d$ and $k_{\rm key},E_{\max}$ are the primitive-parameter
constants of {prf:ref}`thm-slcn-keystone-power`. The signed donor
excess $\Gamma_\theta$ is the actual integral of
{prf:ref}`def-slkd-donor-excess`; the negative center terms have only
been discarded after their signs were established. The particle
Keystone self-exclusion correction is exactly $O(N^{-2})$; the finite
donor-mass correction is tracked separately below.

For a phase class with the radial-square and fitness bands of
{prf:ref}`def-slkd-donor-excess`, and mass intervals
$l_a\le n_a/N\le u_a$ in the particle mode and
$l_a\le\mu(A_a)\le u_a$ in the population mode, define a *fixed phase-wide* donor
bound as follows. Use that definition's $g_{ab},\alpha_{ab}^\pm$ and
companion floor $\kappa_C$, and put $\delta_{ab}=\mathbf1_{a=b}$:

$$
\begin{array}{ll}
\overline U_{ab}^N=\min\{u_a,
 \alpha_{ab}^+u_aNu_b/[\kappa_C(N-1)]\},&
\underline L_{ab}^N=\kappa_C\alpha_{ab}^-l_a
 (Nl_b-\delta_{ab})_+/(N-1),\\
\overline U_{ab}^\infty=\min\{u_a,
 \alpha_{ab}^+u_au_b/\kappa_C\},&
\underline L_{ab}^\infty=\kappa_C\alpha_{ab}^-l_al_b,\\
G_j=\displaystyle\sum_{g_{ab}\ge0}\overline U_{ab}^j g_{ab}
 +\displaystyle\sum_{g_{ab}<0}\underline L_{ab}^j g_{ab},&
j\in\{N,\infty\}.
\end{array}                                                       \tag{SLKD.T2a}
$$

When the displayed products are finite, these are finite sums
determined by the declared landscape and phase bands, including
negative donor contributions. An infinite product gives an
uninformative certificate under the conventions of the donor theorem.
Set
$K_N=\theta k_{\rm key}$,
$K_\infty=2\theta k_{\rm key}$, $e_N=\theta E_{\max}/N^2$ and
$e_\infty=0$. Because $\bar p\le1$, both modes obey

$$
\boxed{\quad W^+\le a_t[W-K_jW^p+G_j+d\sigma_J^2+e_j]+d_t,
\qquad j\in\{N,\infty\}.\quad}                               \tag{SLKD.T4}
$$

The phase-wide donor certificates have the explicit comparison

$$
G_N\le G_\infty+\frac{C_G}{N-1},\qquad
C_G=\sum_{g_{ab}\ge0}\frac{\alpha_{ab}^+u_au_b}{\kappa_C}g_{ab}
 +\sum_{g_{ab}<0}\kappa_C\alpha_{ab}^-l_a\delta_{ab}|g_{ab}|.
                                                               \tag{SLKD.T4a}
$$

Thus one may use the population block sum and the displayed vanishing
finite-donor correction in (SLKD.T4); it is separate from the
$N^{-2}$ Keystone correction. The physical jitter and kinetic noise
terms remain at infinite population and are not particle errors.

Here $W^+$ denotes the conditional expected finite variance or the
population output variance. Put
$H_j=[a_t(G_j+d\sigma_J^2+e_j)+d_t]_+$. For $a_t>0$ define

$$
R_j=\max\left\{
 \left[\frac{4(a_t-1)_+}{a_tK_j}\right]^{1/(p-1)},
 \left[\frac{4H_j}{a_tK_j}\right]^{1/p}\right\}.                                                   \tag{SLKD.T5}
$$

Then every entering state of the declared phase with $W\ge R_j$ and
$W>0$
satisfies the **strict signed full-update estimate**

$$
\boxed{\quad W^+-W\le-\frac{a_tK_j}{2}W^p.\quad}             \tag{SLKD.T6}
$$

If $a_t=0$, then $W^+\le d_t$ and $W^+-W\le-W/2$ for $W\ge2d_t$.
The estimate iterates up to phase exit; global iteration additionally
requires that the actual kernel keeps the trajectory in the declared
phase. This is a variance drift and noise-floor certificate, not yet
full-law TV attraction.
If the phase has an upper attainable variance $W_{\max}<R_j$, the
negative-drift region of this certificate is empty; that inequality
does not assert that the dynamics diverge.

For any $r\ge R_N$ with $r>0$, stop the finite chain at the first
completed state with $W_N<r$ or outside the declared phase, and call
that index $\tau_r$. The same conditional drift gives the explicit
arrival-or-exit bound

$$
\mathbb E(\tau_r\wedge n)\le
 \frac{W_N(S_0)}{(a_tK_N/2)r^p},\qquad
\Pr(\tau_r>n)\le
 \min\left\{1,\frac{W_N(S_0)}{n(a_tK_N/2)r^p}\right\}
\quad(n\ge1),                                                 \tag{SLKD.T7}
$$

when $a_t>0$ and the initial state lies in the phase. In the
population equation, the deterministic trajectory reaches $W<r$
or exits the phase within
$\lceil W(\mu_0)/[(a_tK_\infty/2)r^p]\rceil$ updates.
Multiply these iteration counts by $h$ for physical time; finite
swarm work is $N$ times the iteration count. The event in (SLKD.T7)
explicitly includes phase exit, so it is not a basin-residence claim.

*Proof.* For the empirical prepared law or the rooted population law,
independent draws $X,X'$ yield
$\operatorname{Cov}(X,F(X))=\tfrac12\mathbb E[(X-X')\cdot
(F(X)-F(X'))]$ and
$\operatorname{Var}(F(X))=\tfrac12\mathbb E|F(X)-F(X')|^2$.
Since $\mathbb E|X-X'|^2=2W_C$, (SLKD.T1) and Minkowski give

$$
\operatorname{Cov}(X,F(X))\le-mW_C+b/2,\qquad
\sqrt{\operatorname{Var}(F(X))}
 \le L\sqrt{W_C}+J/\sqrt2.
$$

Every prepared collision velocity has norm at most $V_c$; hence its
variance is at most $V_c^2$. Cauchy--Schwarz in the exact five-term
kinetic polynomial of {prf:ref}`thm-slkd-full-position` gives
$\mathcal K\le A_0W_C+C_1\sqrt{W_C}+C_0$. In particular the
force-velocity term contributes at most
$2B\eta V_c(L\sqrt{W_C}+J/\sqrt2)$, and expanding the squared
force-variance bound gives the $\sqrt2\eta^2LJ$ and
$\eta^2J^2/2$ contributions. Young's inequality gives
$C_1\sqrt{W_C}\le tW_C+C_1^2/(4t)$. Include the exact independent
position-noise variance, bounded above by $d(c^2q^2+s^2)$ in the
finite case, to obtain $W^+\le(1+A_0+t)W_C+d_t\le a_tW_C+d_t$.
Insert {prf:ref}`thm-slkd-signed-cloning`; multiplication by
$a_t\ge0$ preserves its direction. For (SLKD.T2a), substitute
$n_a/N\le u_a$ and $n_b/N\le u_b$ into the upper edge-mass bound of
{prf:ref}`def-slkd-donor-excess`; substitute $n_a/N\ge l_a$ and
$n_b/N\ge l_b$ into its lower bound, preserving self-exclusion
$\delta_{ab}$. The population bounds follow from the same definition.
Use the upper mass for a positive $g_{ab}$ and the lower mass for a
negative one. Bound $\bar p$ by one, proving
(SLKD.T3)--(SLKD.T4). The map $x\mapsto\min(u_a,x)$ is
1-Lipschitz, so the finite positive-edge upper mass exceeds its
population counterpart by at most
$\alpha_{ab}^+u_au_b/[\kappa_C(N-1)]$. For a negative edge,
$(Nl_b-\delta_{ab})_+/(N-1)\ge
l_b-\delta_{ab}/(N-1)$; multiplication by its negative $g_{ab}$
gives the second term of (SLKD.T4a). For $W\ge R_j$, each of
$(a_t-1)_+W$ and $H_j$ is at most $a_tK_jW^p/4$.
Subtract $W$ in (SLKD.T4) to get (SLKD.T6).
Before $\tau_r$, the conditional decrease in (SLKD.T6) is at least
$(a_tK_N/2)r^p$. Apply it to the nonnegative stopped variance and
sum from step zero to $n-1$; the nonnegative value at stopping may be
discarded. This proves the expectation inequality in (SLKD.T7).
Since $n\mathbf1_{\{\tau_r>n\}}\le\tau_r\wedge n$, Markov's bound
follows. The deterministic population count follows by summing the
same decrease until its threshold or exit. $\square$
:::

:::{div} feynman-prose
Follow a pair of walkers through one complete update. Cloning first decides which positions and velocities the kinetic step will receive. Its contribution already contains a negative term proportional to the actual centered positional error. The discharged Keystone estimate bounds that error-weighted cloning activity; in the common-source calculation, one half of this pressure appears with its negative sign intact. We should carry it into the kinetic calculation before estimating the remaining terms.

The velocities entering BAOAB are the velocities produced by copying and collisions. Their position–velocity cross terms tell us whether this prepared motion reduces or increases the positional discrepancy. The force terms then act on those same prepared states. Keeping the signs lets the calculation combine corrective selection with corrective motion before taking upper bounds.

Using the same Gaussian draws in the paired kinetic steps cancels their direct additive contribution to the difference. The draws still change the positions at which the second force is evaluated, and the velocities passed to the cap. They therefore remain inside those evaluations and their expectations. The formula below follows this complete sequence. Its purpose is to put the Keystone pressure, donor transfer, collisions, force terms, and cap into one signed estimate whose net change can be evaluated.
:::

:::{prf:theorem} Signed Keystone–collision–kinetic balance for the complete physical update
:label: thm-slc-signed-complete-update

Use the two actual canonical all-alive swarms, frozen-fitness coupling,
source-label coupling and row jitters of
{prf:ref}`prop-cloning-common-source-signed-estimate`. The complete update
here includes copying, the full accepted-component collision, both force
kicks, OU noise, positional diffusion and the configured radial velocity
cap. It refers to physical coordinates before attaching the terminal
alive/dead mark. It does not change the boundary classification or its
law. All expectations exist when the displayed squared quantities are
integrable; otherwise nonnegative upper bounds are interpreted in the
extended sense.

Choose fixed metric coefficients $\alpha>0$, $\gamma_P>0$ and
$\alpha\gamma_P>\beta^2$, and put
$$
\mathscr Q(S,\widetilde S)=\frac1N\sum_i
 [\alpha|d_i|^2+2\beta d_i\cdot u_i+\gamma_P|u_i|^2],
\quad d_i=x_i-y_i,\quad u_i=v_i-\widetilde v_i.
$$
Its coercivity constants, independent of $N$, are
$$
\lambda_\pm=\frac{\alpha+\gamma_P\pm
 \sqrt{(\alpha-\gamma_P)^2+4\beta^2}}2.
$$
Thus $\lambda_-N^{-1}\sum_i(|d_i|^2+|u_i|^2)\le\mathscr Q$
and the reverse upper bound uses $\lambda_+$.

Retain $D,e_i,h_i,V_i,c_{ij},\gamma_{i,jk},t_i$ from (C.S1).
Write the actual prepared paired differences after copying and full
collisions as
$$
r_i=X_i-Y_i,\qquad z_i=V_i^C-\widetilde V_i^C.
$$
Here $V_i^C$ denotes a collision velocity and is distinct from the
scalar conditional variance $V_i$ of (C.S1). Define
$$
\begin{aligned}
\mathscr R_C={}&\frac1N\sum_{ij}c_{ij}e_j
 +\frac1{2N}\sum_i\ell_i e_i
 +\frac1N\sum_{ijk}\gamma_{i,jk}
       (|x_j-y_k-\bar d|^2-e_i)\\
&+\frac{d\sigma_J^2}{N}\sum_{ijk}\gamma_{i,jk}
       (\mathbf1_{j\ne i}-\mathbf1_{k\ne i})^2
 -|\bar h|^2-\frac1{N^2}\sum_iV_i,\\
\mathscr B_C={}&2\bar d\cdot\bar h+|\bar h|^2
                         +\frac1{N^2}\sum_iV_i,\\
\mathscr T_C={}&\frac1N\sum_i
 \{2\beta[\mathbb E(r_i\cdot z_i)-d_i\cdot u_i]
       +\gamma_P[\mathbb E|z_i|^2-|u_i|^2]\}.
\end{aligned}
$$
All these quantities condition on the entering swarms and retained
fitnesses. The exact preparation identity is
$$
\mathbb E\mathscr Q(S^C,\widetilde S^C)-\mathscr Q(S,\widetilde S)
=-\frac\alpha{2N}\sum_i(p_i+\widetilde p_i)e_i
 +\alpha(\mathscr R_C+\mathscr B_C)+\mathscr T_C.       \tag{SCK.1}
$$
In particular the center contribution $\mathscr B_C$ is retained.
The negative barycenter corrections in the centered positional metric
cancel the corresponding terms in $\mathscr B_C$ when one uses the
uncentered metric $\mathscr Q$; they cannot be counted twice.

For kinetics put $c=h/2$, $a=e^{-\gamma h}$,
$B=c(1+a)$ and $\eta=c^2(1+a)$. The OU standard deviation is
$q=[b_O^2(1-e^{-2\gamma h})/(2\gamma)]^{1/2}$, with its continuous
$\gamma=0$ convention, and $s=\sigma_x\sqrt h$.
Share the OU Gaussian $\xi_i$ and final position Gaussian $\zeta_i$
between the paired rows, independently across rows and of preparation.
Set
$$
f_i=F(X_i)-F(Y_i),\quad
L_i=X_i+BV_i^C+\eta F(X_i)+cq\xi_i,
\quad \widetilde L_i=Y_i+B\widetilde V_i^C+\eta F(Y_i)+cq\xi_i,
$$
$$
g_i=F(L_i)-F(\widetilde L_i),\quad
R_i=r_i+Bz_i+\eta f_i,\quad Z_i=az_i+ac f_i+cg_i.
$$
Thus $R_i$ is the final position difference, including the cancellation
of the common final positional diffusion; $Z_i$ is the pre-cap velocity
difference. The second force is evaluated before the final positional
diffusion, exactly as in the algorithm.

Define the following fully expanded signed polynomial:
$$
\begin{aligned}
\mathscr K(r,z,f,g)={}&
 2[\alpha B+\beta(a-1)]r\cdot z
 +[\alpha B^2+2\beta aB+\gamma_P(a^2-1)]|z|^2\\
&+2(\alpha\eta+\beta ac)r\cdot f+2\beta c\,r\cdot g\\
&+2[\alpha B\eta+\beta acB+\beta a\eta+\gamma_Pa^2c]z\cdot f\\
&+2[\beta Bc+\gamma_Pac]z\cdot g\\
&+[\alpha\eta^2+2\beta ac\eta+\gamma_Pa^2c^2]|f|^2\\
&+2[\beta c\eta+\gamma_Pac^2]f\cdot g+\gamma_Pc^2|g|^2.
\end{aligned}                                                     \tag{SCK.2}
$$
Let $w_i,\widetilde w_i$ be the two actual pre-cap velocities. For
$C_V(w)=Vw/(V+|w|)$ with configured $V=V_{\max}>0$, put
$$
k_i=C_V(w_i)-C_V(\widetilde w_i)-Z_i,\qquad
\mathscr C_i=2\beta R_i\cdot k_i
 +\gamma_P(2Z_i\cdot k_i+|k_i|^2).
$$
Then the exact complete-update identity is
$$
\boxed{
\begin{aligned}
\mathbb E[\mathscr Q(S^+,\widetilde S^+)-\mathscr Q(S,\widetilde S)]
={}&-\frac\alpha{2N}\sum_i(p_i+\widetilde p_i)e_i\\
&+\alpha(\mathscr R_C+\mathscr B_C)+\mathscr T_C
 +\frac1N\sum_i\mathbb E[\mathscr K(r_i,z_i,f_i,g_i)+\mathscr C_i].
\end{aligned}}                                                     \tag{SCK.3}
$$
The expectations on the last line include the intermediate OU excursion.
There is no additive Gaussian forcing term in this paired discrepancy.
Noise still affects the second force and the cap through their actual
random evaluation points.

The collision quantities in $\mathscr T_C$ also have an explicit
finite-sum evaluation. Conditional on the two accepted graphs, couple
Haar rotations for identical components and use independent rotations
for all other components. For row $i$, let $C_i,\widetilde C_i$ be its
components, and put
$$
m_i=|C_i|^{-1}\sum_{j\in C_i}v_j,\quad
\widetilde m_i=|\widetilde C_i|^{-1}\sum_{j\in\widetilde C_i}\widetilde v_j,
\quad b_i=v_i-m_i,\quad\widetilde b_i=\widetilde v_i-\widetilde m_i.
$$
These are the **frozen slot velocities**, before the positional copy;
the algorithm does not copy donor velocities. Singletons have
$b_i=\widetilde b_i=0$ in their respective marginal.
With the configured restitution $\alpha_{\rm col}$,
$$
\begin{aligned}
\mathbb E_R z_i&=m_i-\widetilde m_i,\\
\mathbb E_R|z_i|^2&=|m_i-\widetilde m_i|^2
 +\alpha_{\rm col}^2
  [|b_i|^2+|\widetilde b_i|^2
       -2\mathbf1_{C_i=\widetilde C_i}b_i\cdot\widetilde b_i],\\
\mathbb E_R[r_i\cdot z_i]&=r_i\cdot(m_i-\widetilde m_i).
\end{aligned}                                                     \tag{SCK.4}
$$
Here $\mathbb E_R$ averages only rotations. Averaging these expressions
over the explicitly specified independent source-label rows and Gaussian
jitters gives exactly the two preparation expectations in $\mathscr T_C$.
In particular the full connected component, including noncopying donors,
is used; no pair-collision approximation has entered.

Finally average (SCK.3) over the actual measurement marks. Define
$\mathscr D_N$ to be the conditional expectation of its entire second
line, including the kinetic and cap terms. Substitution of
{prf:ref}`lem-quantitative-keystone` yields, on its stated family,
$$
\mathbb E\Delta\mathscr Q
\le-\frac\alpha2\chi(\epsilon)V_{\rm struct}
       +\frac\alpha2g_{\max}(\epsilon)+\mathscr D_N.       \tag{SCK.5}
$$
On the regional parameter family of {prf:ref}`thm-slcn-keystone-power`, that theorem gives
$$
\mathbb E\Delta\mathscr Q
\le-\frac\alpha2 k_{\rm key}D^p
       +\frac{\alpha E_{\max}}{2N^2}+\mathscr D_N.         \tag{SCK.6}
$$
The constants $p,k_{\rm key},E_{\max}$ are the explicit primitive-parameter
expressions in that theorem, and have no population dependence. These
are complete-update bounds: the Keystone term has not been replaced by
an absolute-value bound on fitness feedback. The displayed
$\alpha E_{\max}/(2N^2)$ is the self-exclusion error, whereas
$\alpha g_{\max}/2$ is the chosen threshold version's structural offset;
the two are not interchangeable.
$\mathscr D_N$ is the exact residual of this one-step identity, evaluated
as the finite plan and Gaussian integrals in
{prf:ref}`prop-slc-reward-full-plan`; it is not a proved uniform
contraction coefficient. Any claimed numerical rate must upper-bound
that same residual on its stated input class.
:::

:::{prf:proof}
Condition first on the input and actual sampled fitnesses. The row-source
coupling has the correct donor and rejection probabilities by construction.
Add and subtract $(p_i+\widetilde p_i)e_i/2$ in (C.S1), and use
$c_i=(p_i+\widetilde p_i-\ell_i)/2$. This gives the centered positional
increment $-(2N)^{-1}\sum_i(p_i+\widetilde p_i)e_i+\mathscr R_C$.
The row differences after cloning have means $d_i+h_i$ and are
independent conditionally on these data. Their empirical mean therefore
has squared expectation $|\bar d+\bar h|^2+N^{-2}\sum_iV_i$.
Subtracting $|\bar d|^2$ gives precisely $\mathscr B_C$.
The identity between centered and uncentered squares, followed by
expansion of the two velocity-containing terms of $\mathscr Q$, proves
(SCK.1).

For (SCK.4), the actual component rule is
$V_i^C=m_i+\alpha_{\rm col}R_{C_i}b_i$. Haar invariance gives
$\mathbb ER_C=0$ and $R_C^TR_C=I$. Matched components share a single
rotation, so their cross product averages to $b_i\cdot\widetilde b_i$;
unmatched components have independent zero-mean rotations and zero
cross product. Positions and jitters are independent of these rotations.
Expansion proves every formula in (SCK.4), also for singletons since
their centered component velocities vanish.

The exact two-force BAOAB algebra gives final positional difference
$r+Bz+\eta f$ and pre-cap velocity difference $az+acf+cg$.
Expand $\alpha|R|^2+2\beta R\cdot Z+\gamma_P|Z|^2$
and subtract $\alpha|r|^2+2\beta r\cdot z+\gamma_P|z|^2$.
Collecting the nine displayed monomial types gives exactly (SCK.2).
The final cap changes $Z$ into $Z+k$; its increment is precisely
$\mathscr C_i$. Conditional expectation, then (SCK.1), proves (SCK.3).
No deterministic substitution for the random intermediate force is used.
The law of total expectation permits the final average over measurement
marks; the pressure estimates apply before discarding any term.
Multiplication by $-\alpha/2$ reverses their lower inequalities and gives
(SCK.5)–(SCK.6).
:::

:::{prf:theorem} Keystone-first TV bound for sampled marked positions
:label: thm-slc-keystone-tagged-tv

Use the actual paired complete update and the signed quadratic metric
$\mathscr Q$ of {prf:ref}`thm-slc-signed-complete-update`. Both entering
swarms must be nonextinct, so mandatory revival makes every prepared
row active. Let $\tau^2=c^2q^2+s^2>0$ with exactly the configured
BAOAB and final position noises. Independently of the update, select
an ordered $k$-tuple $I=(I_1,\ldots,I_k)$ of distinct labels uniformly
from $\{1,\ldots,N\}$, where $1\le k\le N$; use the same tuple in
the two swarms. Let $\mathsf T_k(S^+)$ denote the law of the resulting
positions and their terminal marks. Then
$$
\begin{aligned}
\bigl\|\mathsf T_k(S^+)-\mathsf T_k(\widetilde S^+)\bigr\|_{\rm TV}
&\le \min\left\{1,
 \frac{\sqrt{k}}{\tau\sqrt{2\pi}}
 \left[\mathbb E\frac1N\sum_i
  |\mu_i-\widetilde\mu_i|^2\right]^{1/2}\right\}\\
&\le \min\left\{1,
 \sqrt{\frac{k}{2\pi\tau^2\lambda_-}}
 \bigl[\mathbb E\mathscr Q(S^+,\widetilde S^+)\bigr]^{1/2}
 \right\},
\end{aligned}                                                     \tag{SCK.TV1}
$$
where $\mu_i=X_i+B V_i^C+\eta F(X_i)$ and $\lambda_->0$ is the
explicit metric eigenvalue in the signed-update theorem. The
expectation is over the *same* measurement, accepted-plan, jitter,
component-Haar and kinetic coupling used in (SCK.3). The marked
position includes the actual terminal classification; there is no
conditioning on survival.

On the all-alive family of {prf:ref}`thm-slcn-keystone-power`, insert
the already proved complete-update bound (SCK.6) to obtain the
fully parameterized one-step consequence
$$
\boxed{\quad
\bigl\|\mathsf T_k(S^+)-\mathsf T_k(\widetilde S^+)\bigr\|_{\rm TV}
\le\min\left\{1,
\sqrt{\frac{k}{2\pi\tau^2\lambda_-}}
\left[\mathscr Q(S,\widetilde S)
-\frac\alpha2 k_{\rm key}W_N^p
+\frac{\alpha E_{\max}}{2N^2}
+\mathscr D_N(S,\widetilde S)\right]_+^{1/2}\right\}.
\quad}                                                          \tag{SCK.TV2}
$$
Every term on the right is the existing signed Keystone--kinetic
accounting, with its reward and regional substitutions in
{prf:ref}`thm-slc-regional-parameter-one-step`. In the uniform
companion specialization $\kappa_D=\kappa_C=1$ in the explicit
$k_{\rm key}$ and donor constants; no product-minorization coefficient
appears. Equation (SCK.TV2) is a TV bound for a fixed number of
randomly sampled marked *positions*. It does not identify its
right-hand side as a decaying function of time without a proved
signed estimate on the displayed residual.

For a conservative all-alive trajectory coupled at every step by the
same prescribed kernel, write
$e_n=\mathbb E\mathscr Q(S_n,\widetilde S_n)$ and
$w_n=\mathbb E W_N(S_n,\widetilde S_n)^p$. Whenever the Keystone
family applies at each entering state, conditional expectation of
(SCK.6) gives the exact proof-chain recurrence
$$
e_{n+1}\le e_n-\frac\alpha2k_{\rm key}w_n
 +\frac{\alpha E_{\max}}{2N^2}
 +\mathbb E\mathscr D_N(S_n,\widetilde S_n),
\qquad
\|\mathsf T_{k,n}-\widetilde{\mathsf T}_{k,n}\|_{\rm TV}
\le\sqrt{\frac{k e_n}{2\pi\tau^2\lambda_-}}\quad(n\ge1).
                                                               \tag{SCK.TV3}
$$
The first inequality keeps the actual signed donor, barycenter,
collision, force and cap contributions. Terminal marking is retained
by the position-to-mark map in the TV step. The coefficient of
the Keystone power and the TV conversion constant have no population
factor for fixed $k$; the sole displayed finite-population pressure
error is $O(N^{-2})$.
:::

:::{prf:proof}
Condition on the complete prepared populations, including the accepted
graphs, jitter and Haar rotations. The BAOAB positional calculation
in (SCK.G9) gives, independently across rows in either marginal,
$x_i^+=\mu_i+cq\xi_i+s\chi_i$. Hence the selected $k$ positions
have Gaussian laws with common covariance $\tau^2I_{kd}$ and means
$(\mu_{I_j})_{j=1}^k$ and
$(\widetilde\mu_{I_j})_{j=1}^k$. Their TV distance is
$2\Phi(\Delta_I/(2\tau))-1$, where
$\Delta_I^2=\sum_{j=1}^k|\mu_{I_j}-\widetilde\mu_{I_j}|^2$;
the one-dimensional Gaussian half-space calculation proves this
identity. Since $\Phi'\le1/\sqrt{2\pi}$, it is at most
$\Delta_I/(\tau\sqrt{2\pi})$. Terminal marks are deterministic
functions of these positions, so adjoining them cannot increase TV.
Convexity of TV under the common coupling of preparations and label
tuples, followed by Jensen, gives the first inequality in (SCK.TV1):
the average of $\Delta_I^2$ over tuples is exactly
$kN^{-1}\sum_i|\mu_i-\widetilde\mu_i|^2$.

Under the synchronous kinetic innovations of (SCK.3), the two final
positions differ by $\mu_i-\widetilde\mu_i$; final velocity capping
does not alter positions. Coercivity of $\mathscr Q$ therefore gives
$N^{-1}\sum_i|\mu_i-\widetilde\mu_i|^2
\le\mathscr Q(S^+,\widetilde S^+)/\lambda_-$ pathwise.
This proves the second inequality. Substitute (SCK.6), whose
$\mathscr D_N$ has exactly the same conditional coupling, to prove
(SCK.TV2). Iterating conditional expectation of that inequality in
the conservative all-alive case proves the first part of (SCK.TV3);
applying (SCK.TV1) to the coupled input laws at time $n-1$ proves
its second part. No independence of walkers before preparation or
all-row minorization is used.
:::

:::{prf:theorem} Full-state TV smoothing for the actual Keystone kinetic update
:label: thm-slc-keystone-full-state-tv

Retain the actual all-alive prepared-source coupling and complete BAOAB
update of {prf:ref}`thm-slc-signed-complete-update`. Suppose $q,s>0$,
$F\in C^1(\mathbb R^d)$, $\|DF\|\le L<\infty$, and
$\ell=c^2L<1$. These are structural force-profile values; the
algorithm is unchanged. Let
$\omega_D(r)=\sup_{|u-v|\le r}\|DF(u)-DF(v)\|$; an infinite or
nonvanishing modulus leaves the corresponding small-distance
certificate uninformative. For two prepared row states
$\theta=(x,v)$ and $\theta'=(x',v')$, define

$$
\begin{gathered}
x_1=x+c(v+cF(x)),\quad x_1'=x'+c(v'+cF(x')),\\
m=a(v+cF(x)),\quad m'=a(v'+cF(x')),\quad
D_1=|x_1-x_1'|,\quad M=|m-m'|,\\
A=\frac{cLD_1}{1-\ell},\quad D=M+A,\quad
\varepsilon_D=\frac{c^2}{1-\ell}
 \omega_D\!\left(\frac{D_1}{1-\ell}\right),\\
J_D=d\min\left\{
 \begin{cases}-\log(1-\varepsilon_D),&\varepsilon_D<1,\\
 +\infty,&\varepsilon_D\ge1,
 \end{cases}
 \log\frac{1+\ell}{1-\ell}\right\},\\
\mathcal U(\theta,\theta')=
\min\left\{1,
 \sqrt{\frac12\left(\frac{\sqrt d D}{q}
       +\frac{D^2}{2q^2}+J_D\right)}
 +\frac{D_1}{s\sqrt{2\pi}(1-\ell)}\right\}.
\end{gathered}                                                     \tag{SCK.FTV1}
$$

Let $K_\theta$ be the conditional law of the **full** completed row
$(x^+,v^+,\text{terminal mark})$, including the configured radial cap
and boundary classification. Then

$$
\boxed{\quad\|K_\theta-K_{\theta'}\|_{\rm TV}
 \le\mathcal U(\theta,\theta').\quad}                       \tag{SCK.FTV2}
$$

For two actual prepared $N$-row swarms coupled by the Keystone source
plans, independent kinetic innovations across rows, and a uniformly
sampled ordered $k$-tuple of distinct labels, $1\le k\le N$, their
full-state marked $k$-row output laws satisfy

$$
\boxed{\quad
\|\mathsf T_k^{\rm full}(S^+)-
       \mathsf T_k^{\rm full}(\widetilde S^+)\|_{\rm TV}
\le\min\left\{1,
 k\,\mathbb E\frac1N\sum_{i=1}^N
      \mathcal U((X_i,V_i^C),(Y_i,\widetilde V_i^C))\right\}.
\quad}                                                          \tag{SCK.FTV3}
$$

The expectation uses exactly the same finite source plans, shared
jitters and component-Haar coupling as (SCK.3). No independence of
walkers before preparation is assumed. For fixed $k$, all numerical
coefficients in (SCK.FTV1)--(SCK.FTV3) are independent of $N$.
This is TV of the complete $(x,v,a)$ law of each sampled row. Taking
$k=N$ gives a valid full-configuration bound with its displayed
factor $N$, not an $N$-uniform full-configuration prefactor.
The right side tends to zero whenever the prepared paired differences
tend to zero in probability and
$\omega_D(r)\to0$ as $r\downarrow0$. Thus the TV conversion covers
velocity and terminal status as well as position, while keeping the
force-regularity profile explicit.

*Proof.* For fixed $x_1$ define
$T_{x_1}(z)=z+cF(x_1+cz)$. Its Lipschitz perturbation of the
identity has constant $\ell<1$, so it is a $C^1$ diffeomorphism by
the inverse argument of {prf:ref}`lem-slcpd-density`. The actual
pre-cap velocity is $w=T_{x_1}(z)$ with
$z\sim N(m,q^2I)$, and conditional on $z$ the completed position is
$N(x_1+cz,s^2I)$. Compare it with the second prepared state by
$H=T_{x_1'}^{-1}\circ T_{x_1}$. The inverse Lipschitz bound gives
$|H(z)-z|\le A$ and
$|x_1+cz-x_1'-cH(z)|\le D_1/(1-\ell)$.

The same pre-cap velocity is now represented by $T_{x_1'}(H(z))$
in the first law and $T_{x_1'}(z')$ in the second, where
$z'\sim N(m',q^2I)$. Applying the common bijection
$T_{x_1'}^{-1}$ preserves TV. Splitting the two joint laws first by
their transformed-$z$ marginals and then by conditional position
Gaussians bounds their TV by

$$
\|H_\#N(m,q^2I)-N(m',q^2I)\|_{\rm TV}
 +\frac{D_1}{s\sqrt{2\pi}(1-\ell)}.
$$

The Gaussian term is the exact common-covariance TV bound used in
(SCK.TV1). The change-of-variables formula gives

$$
D_{\rm KL}(H_\#N(m,q^2I)\|N(m',q^2I))
=\mathbb E\left[
 \frac{|H(z)-m'|^2-|z-m|^2}{2q^2}
 -\log\det DH(z)\right].
$$

Since $|H(z)-z|\le A$ and $\mathbb E|z-m|\le q\sqrt d$, the
quadratic expectation is at most $\sqrt d D/q+D^2/(2q^2)$.
Differentiating $T_{x_1'}(H(z))=T_{x_1}(z)$ gives

$$
DH(z)=[I+c^2DF(x_1'+cH(z))]^{-1}
        [I+c^2DF(x_1+cz)].
$$

The two force arguments differ by at most $D_1/(1-\ell)$, hence
$\|DH-I\|\le\varepsilon_D$. If $\varepsilon_D<1$, the smallest
singular value is at least $1-\varepsilon_D$ and
$-\log\det DH\le-d\log(1-\varepsilon_D)$. Without this local
estimate, the two factors of $DH$ have singular values between
$1-\ell$ and $1+\ell$, giving
$-\log\det DH\le d\log[(1+\ell)/(1-\ell)]$.
Their minimum is $J_D$. Pinsker's inequality proves the first term
of (SCK.FTV1). The radial cap is a common bijection from pre-cap
velocity space to its open velocity ball, and adjoining the terminal
mark is a common deterministic map, so neither increases TV. This
proves (SCK.FTV2).

Conditional on the complete prepared swarms, row kinetic innovations
are independent. TV of two $k$-fold product laws is at most the sum of
their row TVs, by telescoping the product measures. Average the same
ordered label tuple and the coupled preparation laws. Every row appears
with probability $k/N$, proving (SCK.FTV3) without an $N$-dependent
coefficient. $\square$
:::

:::{prf:corollary} Population-uniform full-state TV transfer from prepared error
:label: cor-slc-keystone-full-state-tv-transfer

Under {prf:ref}`thm-slc-keystone-full-state-tv`, set

$$
\begin{gathered}
C_x=\sqrt{(1+c^2L)^2+c^2},\quad
C_m=a\sqrt{(cL)^2+1},\quad
C_D=C_m+\frac{cLC_x}{1-\ell},\\
\epsilon(r)=\frac{c^2}{1-\ell}
 \omega_D\!\left(\frac{C_xr}{1-\ell}\right),\\
J(r)=d\min\left\{
 \begin{cases}-\log(1-\epsilon(r)),&\epsilon(r)<1,\\
 +\infty,&\epsilon(r)\ge1,
 \end{cases}
 \log\frac{1+\ell}{1-\ell}\right\},\\
\Psi(r)=\min\left\{1,
\sqrt{\frac12\left(\frac{\sqrt d C_Dr}{q}
 +\frac{C_D^2r^2}{2q^2}+J(r)\right)}
 +\frac{C_xr}{s\sqrt{2\pi}(1-\ell)}\right\}.
\end{gathered}                                                     \tag{SCK.FTV4}
$$

For the actual paired preparation at step $n$, put
$P_n=\mathbb E N^{-1}\sum_i(|r_{n,i}|^2+|z_{n,i}|^2)$.
In the bounded all-alive class of
{prf:ref}`lem-slc-uniform-collision-budget`, its already proved
full-preparation estimate is

$$
P_n\le\mathbb E[M_{r,n}+A_cU_n+4V_c^2\Theta_n],
\qquad \Theta_n=\min\{1,6e^{4/\kappa_C}\bar\ell_{N,n}\}.
                                                               \tag{SCK.FTV4a}
$$

The fitness and reward profiles enter $\bar\ell_{N,n}$ through
(C.S4) and (SCK.G3)--(SCK.G7), and every coefficient in this
particular one-step bound is independent of $N$. More explicitly,
write $e_n=\mathbb E\mathscr Q(S_n,\widetilde S_n)$ and
$\overline t_n=\mathbb E\bar t_{N,n}$. The common/residual split in
(SCK.UM5) and the first inequality of (SCK.UM7), which bounds
component sensitivity directly by $\bar t_N$, give

$$
\boxed{\quad P_n\le C_Qe_n+C_t\overline t_n,\qquad
C_Q=\frac{3+1/\kappa_C+A_c}{\lambda_-},
\quad C_t=8D_x^2+d\sigma_J^2+24V_c^2e^{4/\kappa_C}.\quad} \tag{SCK.FTV4b}
$$

Indeed $(|\bar d|+2D_x)^2\le2P+8D_x^2$, and the prepared
position, frozen-velocity collision and mismatched-component terms contribute
respectively $(3+1/\kappa_C)P$, $A_cU$ and the
displayed multiple of $\bar t_N$. This bound vanishes on the
diagonal and has no population-size multiplier.

In the explicit uniform-measurement and uniform-clone-companion
specialization $\epsilon_D=\epsilon_C=\infty$, use the *same*
uniform distinct measurement label in the paired swarms. Suppose
the raw reward has structural Lipschitz bound
$|r(z)-r(z')|\le L_R|z-z'|$ on the entering phase, with the
physical state norm $|z-z'|^2=|x-x'|^2+|v-v'|^2$.
Let $K_b=H_b(2/\sigma_b+R_b^2/\sigma_b^3)$ for
$b\in\{r,s\}$, using the explicit $H_b,R_b,\sigma_b$ of
{prf:ref}`lem-slc-common-source-normalizers`, and put
$L_z=\max\{1,\sqrt{\lambda_{\rm alg}}\}$. Then the same
fitness and gate calculation gives

$$
\boxed{\quad
\overline t_n\le L_T\sqrt{e_n},\qquad
L_T=\frac{(L_{\rm rec}+L_{\rm don})
       (K_rL_R+2K_sL_z)}{\sqrt{\lambda_-}}.
\quad}                                                          \tag{SCK.FTV4c}
$$

This keeps both active fitness channels; infinite companion width
only makes their label choices uniform. To verify the factor two,
the floored separation is 1-Lipschitz in its pair-feature distance,
and the squashing maps are 1-Lipschitz. Its paired raw difference is
at most $L_z(t_i+t_{m_i})$, with
$t_i=(|d_i|^2+|u_i|^2)^{1/2}$.
The common uniform distinct label has expected average
$N^{-1}\sum_i t_{m_i}=N^{-1}\sum_i t_i$.
Average the normalizer lemma, apply (SCK.UK2), then
Cauchy--Schwarz and $\mathscr Q\ge\lambda_-N^{-1}\sum_i t_i^2$.
Finally average over the coupled trajectories and use Jensen to
obtain (SCK.FTV4c).
For every radius $r_0>0$ and every fixed $k\le N$,

$$
\boxed{\quad
\|\mathsf T_{k,n+1}^{\rm full}-
  \widetilde{\mathsf T}_{k,n+1}^{\rm full}\|_{\rm TV}
\le\min\{1,k[\Psi(r_0)+P_n/r_0^2]\}.
\quad}                                                          \tag{SCK.FTV5}
$$

The same bound with $k=1$ holds for the actual population map, with
$P_n$ the second moment of any coupling of its two prepared-root
laws. If a phase-local signed Keystone calculation proves an
$N$-independent estimate $P_n\le C\rho^n$ with displayed
$0<C<\infty$, $0<\rho<1$, then choosing
$r_0=(C\rho^n)^{1/4}$ gives the **full-state TV** rate

$$
\|\mathsf T_{k,n+1}^{\rm full}-
  \widetilde{\mathsf T}_{k,n+1}^{\rm full}\|_{\rm TV}
\le\min\{1,k[\Psi((C\rho^n)^{1/4})+
                 (C\rho^n)^{1/2}]\}.                         \tag{SCK.FTV6}
$$

This consequence compares trajectories in the same certified phase;
it makes no comparison between distinct stationary phases. Its
premise is a decaying *prepared discrepancy*, not the one-swarm
variance drift of {prf:ref}`thm-slkd-structural-variance-threshold`.
The bound therefore identifies precisely what the signed full-update
calculation must establish to obtain a long-time TV theorem.
In the uniform-companion specialization (SCK.FTV4c), a signed
quadratic estimate $e_n\le C\rho^n$ already implies the required
prepared bound, with the completely displayed value
$P_n\le C_QC\rho^n+C_tL_T\sqrt C\rho^{n/2}$.

*Proof.* For $t=(|x-x'|^2+|v-v'|^2)^{1/2}$, the force Lipschitz
bound in the definitions of $x_1,m$ gives $D_1\le C_xt$ and
$M\le C_mt$. Hence $D\le C_Dt$ and
$\varepsilon_D\le\epsilon(t)$, proving
$\mathcal U(\theta,\theta')\le\Psi(t)$ by the monotonicity of every
term in (SCK.FTV1). Split the average in (SCK.FTV3) into
$t_i\le r_0$ and $t_i>r_0$. On the first part use $\Psi(r_0)$;
on the second use $\mathcal U\le1$ and
$\mathbf1_{\{t_i>r_0\}}\le t_i^2/r_0^2$.
This proves (SCK.FTV5). The population proof is the identical
one-root calculation. Substitute the stated prepared-error estimate
and the displayed choice of $r_0$ for (SCK.FTV6). $\square$
:::

:::{prf:proposition} Exact finite-plan prepared discrepancy and full-state TV accounting
:label: prop-slc-exact-prepared-tv

Use the actual all-alive update and the paired measurement and source
coupling of (SCK.F1)--(SCK.F5). For each pair of complete retained
measurement vectors $(m,\widetilde m)$, let
$Q_{m,\widetilde m}(\mathbf p)$ be (SCK.P1). For a paired plan
$\mathbf p=((j_i,k_i))_i$, put
$A_i=\mathbf1_{j_i\ne i}$, $\widetilde A_i=\mathbf1_{k_i\ne i}$,
and compute $H_i(\mathbf p)$ from the two **frozen-slot-velocity**
component partitions in (SCK.P2). Then the following conditional
preparation cost is a finite, nonnegative expression:

$$
\mathcal P_N(S,\widetilde S;m,\widetilde m)
=\sum_{\mathbf p}Q_{m,\widetilde m}(\mathbf p)\frac1N\sum_i
\left[|x_{j_i}-y_{k_i}|^2
 +d\sigma_J^2(A_i-\widetilde A_i)^2+H_i(\mathbf p)\right].
                                                               \tag{SCK.EP1}
$$

For any specified coupling $\Lambda$ of the complete measurement
vectors, including their within-swarm shared fitness normalizers,
define $\overline{\mathcal P}_N=\sum_{m,\widetilde m}
\Lambda(m,\widetilde m)\mathcal P_N(S,\widetilde S;m,\widetilde m)$.
For a contraction comparison choose the explicit diagonal-preserving
$\Lambda$ of (SCK.F4a); then $\overline{\mathcal P}_N(S,S)=0$.
This is **exactly**
$\mathbb E N^{-1}\sum_i(|r_i|^2+|z_i|^2)$ under the same paired
preparation used in (SCK.3), with no mismatch or component-size
envelope. At time $n$, average it over the actual coupled entering
states and denote the result by $P_n^{\rm exact}$.

There is also a direct finite-plan TV certificate. Let
$\mathcal H_{\mathbf p}$ couple the actual component rotations as in
(SCK.P4), and let $G^J$ be the shared row jitters. Construct the
prepared rows $\theta_i=(X_i,V_i^C)$ and
$\widetilde\theta_i=(Y_i,\widetilde V_i^C)$ from the plan. Put

$$
\begin{gathered}
z_\theta(w)=T_{x_1}^{-1}(w),\qquad
p_\theta(y,w)=
\frac{\varphi_{q,d}(z_\theta(w)-m)
      \varphi_{s,d}(y-x_1-cz_\theta(w))}
     {|\det[I+c^2DF(x_1+cz_\theta(w))]|},\\
\mathfrak t(\theta,\theta')=\frac12\int_{\mathbb R^{2d}}
 |p_\theta(y,w)-p_{\theta'}(y,w)|\,dy\,dw.
\end{gathered}                                                     \tag{SCK.EP2a}
$$

Here $x_1,m,T_{x_1}$ are those in (SCK.FTV1) and its proof, and
$\varphi_{\sigma,d}$ is the centered $d$-Gaussian density with
standard deviation $\sigma$. The radial cap is bijective onto its
image, and the terminal mark is a deterministic function of $y$;
therefore $\mathfrak t$ is the **exact** full marked one-row TV
distance conditional on the prepared states, with
$\mathfrak t(\theta,\theta')\le\mathcal U(\theta,\theta')$.
Define $\mathcal V_N^{\rm exact}$ by the following formula with
$\mathfrak t$ in place of $\mathcal U$:

$$
\mathcal V_N(S,\widetilde S)=
\sum_{m,\widetilde m}\Lambda(m,\widetilde m)
\sum_{\mathbf p}Q_{m,\widetilde m}(\mathbf p)
\int\frac1N\sum_i\mathcal U(\theta_i,\widetilde\theta_i)
\,d\gamma_{Nd}(G^J)\,d\mathcal H_{\mathbf p}.
                                                               \tag{SCK.EP2}
$$

Under the force and noise conditions of
{prf:ref}`thm-slc-keystone-full-state-tv`, every fixed $k\le N$
therefore has the two explicit full-state marked bounds

$$
\boxed{\quad
\|\mathsf T_{k,n+1}^{\rm full}-
\widetilde{\mathsf T}_{k,n+1}^{\rm full}\|_{\rm TV}
\le\min\{1,k\mathbb E\mathcal V_N^{\rm exact}(S_n,\widetilde S_n),
 k\mathbb E\mathcal V_N(S_n,\widetilde S_n),
 k[\Psi(r)+P_n^{\rm exact}/r^2]\},\qquad r>0.
\quad}                                                          \tag{SCK.EP3}
$$

The numerical multipliers are independent of $N$ for fixed $k$;
the finite plan and the actual state law retain their $N$ dependence.
The complete-step signed physical drift of this **same** coupling is
(SCK.F5) with $\eta_A=0$, or (SCK.3) on an all-alive segment.
Consequently its exact physical
quadratic expectation satisfies
$e_n=e_0+\sum_{j<n}\mathbb E\Delta\mathscr Q_j$ whenever these
signed terms are integrable. Equations (SCK.EP1)--(SCK.EP3) do not
replace that signed drift by a positive mismatch allowance. A
long-time rate follows from evaluating its sign and the resulting
prepared cost; the finite-plan identities alone do not assert that
either decreases.

*Proof.* Conditional on $m,\widetilde m,\mathbf p$, the prepared
position difference in row $i$ is
$x_{j_i}-y_{k_i}+\sigma_J(A_i-\widetilde A_i)G_i^J$.
Its Gaussian squared mean is the first two terms of (SCK.EP1).
The rotations are independent of the jitter, and (SCK.4) makes their
squared velocity-difference mean exactly $H_i(\mathbf p)$, using the
frozen velocity of slot $i$. Sum rows, plans and measurement pairs to
obtain the claimed equality. For fixed preparation, the change of
variables $w=T_{x_1}(z)$ gives (SCK.EP2a) as the joint density of
final position and pre-cap velocity. The cap and mark map is injective
on this pair, so the marked row TV is exactly $\mathfrak t$.
Conditional kinetic independence and product telescoping bound the
$k$-row TV by the sum of these exact row TVs. Integrating the same
preparation proves the first branch of (SCK.EP3); using
$\mathfrak t\le\mathcal U$ proves the second. Applying (SCK.FTV5)
with the exact prepared second moment proves its third branch.
Finally (SCK.F5)
and the tower property give the signed telescoping equality. $\square$
:::

:::{prf:proposition} The exact sign that the Keystone pressure does not determine
:label: prop-slc-keystone-net-flux-gap

In the all-alive conditional source coupling of (C.S1), put
$P_x=N^{-1}\sum_i|d_i|^2$ and let $P_x^C$ be the corresponding
post-cloning positional discrepancy. Directly from the **same**
finite plan, without a regional envelope,

$$
\boxed{\quad
\mathbb E(P_x^C-P_x)=\frac1N\sum_{i,j}c_{ij}
 (|d_j|^2-|d_i|^2)
 +\frac1N\sum_{i,j,k}\gamma_{i,jk}
 (|x_j-y_k|^2-|d_i|^2)
 +\frac{d\sigma_J^2}{N}\sum_{i,j,k}\gamma_{i,jk}
 (\mathbf1_{j\ne i}-\mathbf1_{k\ne i})^2.
\quad}                                                         \tag{SCK.NF1}
$$

The Keystone lower bound controls an outgoing portion of this
identity, $N^{-1}\sum_i(p_i+\widetilde p_i)e_i$. It does **not**
bound the signed common-edge flux in (SCK.NF1) from above: its
corresponding incoming donor term is
$N^{-1}\sum_{i,j}c_{ij}e_j$. Moreover the centered-to-uncentered
conversion in (SCK.1) cancels its negative barycenter corrections.
Thus using the Keystone pressure alone as a negative coefficient
of the complete paired metric is the first invalid closure step.
The finite-plan formula (SCK.EP1) repairs the accounting but does
not by itself establish a negative sign or a decaying TV bound.

For clarity about the scale lost by the scalar envelope, the residual
source mass in row $i$ is exactly $t_i$ and its contribution is
$\sum_{j,k}\gamma_{i,jk}|x_j-y_k|^2$. If the row-law difference is
of order $\varepsilon$ while cross-source squared separations remain
bounded away from zero, this contribution can be of order
$\varepsilon$, although the entering paired squared error is of
order $\varepsilon^2$. This is a statement about the specified
common-label coupling and its quadratic metric; it is not a
nonconvergence conclusion for the Markov law. The full update's
signed kinetic, cap, and terminal terms must be evaluated with this
net cloning flux before any contraction claim.

*Proof.* On a common source label $j$, the two post-copy positions
differ by $d_j$ and their shared jitter cancels. On a residual pair
$(j,k)$, their difference is
$x_j-y_k+\sigma_J(\mathbf1_{j\ne i}-\mathbf1_{k\ne i})G_i^J$.
Average its square, subtract $|d_i|^2$, and use
$\Pi_i(j,k)=\lambda_{ij}\mathbf1_{j=k}+\gamma_{i,jk}$.
The common persistence term $j=i$ has zero increment, while a common
accepted label has mass $c_{ij}$, proving (SCK.NF1). The Keystone
comparison and barycenter cancellation follow by expanding (C.S1)
and (SCK.1). The final scale statement follows from the displayed
residual mass and cost, without an independence or sign assumption.
$\square$
:::

:::{prf:theorem} Unsimplified signed ledger for one canonical complete update
:label: thm-slc-unsimplified-complete-ledger

Condition on two all-alive frozen inputs and their complete retained
measurement vectors $(m,\widetilde m)$. Keep the exact row coupling
$\Pi_i$, complete-plan weights $Q(\mathbf p)$, accepted components,
shared jitters, matched-component Haar coupling, both BAOAB force
evaluations, and radial cap of (SCK.P1)--(SCK.P4). Put
$P_x=N^{-1}\sum_i|d_i|^2$ and define $\mathcal A_x$ to be the
entire right-hand side of (SCK.NF1). For each plan use
$H_i(\mathbf p)$ from (SCK.P2), and let

$$
\begin{aligned}
\mathcal A_{xv}={}&\sum_{\mathbf p}Q(\mathbf p)\frac1N\sum_i
 2\beta\big[(x_{j_i}-y_{k_i})\cdot
       (m_i-\widetilde m_i)-d_i\cdot u_i\big],\\
\mathcal A_v={}&\sum_{\mathbf p}Q(\mathbf p)\frac1N\sum_i
 \gamma_P\big[H_i(\mathbf p)-|u_i|^2\big],\\
\mathcal A_K={}&\sum_{\mathbf p}Q(\mathbf p)
 \int\frac1N\sum_i
  [\mathscr K(r_i,z_i,f_i,g_i)+\mathscr C_i]
  \,d\gamma_{2Nd}\,d\mathcal H_{\mathbf p}.
\end{aligned}                                                     \tag{SCK.NET1}
$$

The symbols $m_i,\widetilde m_i$ in the first line are collision
component means of **frozen slot velocities**, not measurement
marks. The latter appear only in the plan weights. The force and cap
integrand is the fully expanded (SCK.2)--(SCK.3); it is evaluated on
the actual jittered positions and rotated velocities, with the same
OU Gaussian in paired rows. If its terms are absolutely integrable,
then the exact conditional complete-step identity is

$$
\boxed{\quad
\mathbb E[\mathscr Q(S^+,\widetilde S^+)
          -\mathscr Q(S,\widetilde S)\mid S,\widetilde S,m,\widetilde m]
=\alpha\mathcal A_x+\mathcal A_{xv}+\mathcal A_v+\mathcal A_K.
\quad}                                                         \tag{SCK.NET2}
$$

There is no unspecified remainder in (SCK.NET2). The reward and
diversity channels, companion temperature, acceptance cap and
regularization enter its $Q(\mathbf p)$ through the actual measured
fitness and (SCK.F1)--(SCK.F3). Clone jitter enters $\mathcal A_x$
and the kinetic integral; restitution enters $H_i$ and the same
integral; $h,\gamma,b_O,\sigma_x,F,V_{\max}$ enter the kinetic
integral through (SCK.2)--(SCK.3). The initial phase-space metric
coefficients are $\alpha,\beta,\gamma_P$. Average (SCK.NET2) over
the diagonal-preserving joint measurement law (SCK.F4a) to obtain the
unconditional physical drift. For nonextinct marked inputs the
identical statement uses the marked source rows (SCK.F2), the
full-slot preparation (SCK.7), and adds **exactly** the terminal
status increment (SCK.8), as already expanded in (SCK.F5).

Equation (SCK.NET2) is the signed quantity whose negativity must be
established in a claimed phase. In particular no negative multiple
of the Keystone pressure may be removed from $\mathcal A_x$ without
retaining its donor insertion and residual-source costs.

*Proof.* Equation (SCK.NF1) is the exact conditional positional
increment after source selection and jitter. Conditional on the
same plan, jitter has zero cross mean with the collision velocity,
while (SCK.4) gives the first and second Haar moments; these are
exactly $\mathcal A_{xv}$ and $\mathcal A_v$. For each realized
prepared state, expanding the two BAOAB difference equations gives
(SCK.2), and adding the cap difference gives $\mathscr C_i$ with
no estimate. Average those terms over the plan's **joint** jitter,
rotation and OU law to obtain $\mathcal A_K$. Summing the four
increments proves (SCK.NET2). The marked statement follows by the
same conditioning and the Gaussian terminal-indicator identity
(SCK.8). $\square$
:::

:::{prf:corollary} Full-state TV estimate driven by the signed Keystone ledger
:label: cor-slc-keystone-signed-full-tv-ledger

Retain the hypotheses of (SCK.KM4) at every entering step of two
coupled all-alive trajectories in one declared structural phase, and
the regularity and positive-noise conditions of
{prf:ref}`thm-slc-keystone-full-state-tv`.
Use the diagonal-preserving measurement coupling (SCK.F4a) at each
step, followed by the same common-source, Haar and Gaussian coupling.

The primary quantitative certificate is (SCK.EP3): it evaluates the
actual finite-plan prepared cost and, in its first branch, the exact
conditional one-row TV integral. The following closed scalar bound
is a further envelope for situations where only the aggregated
Keystone and regional data are retained; it loses the signs and
plan correlations eliminated by (SCK.UM1) and (SCK.UM7).
Use the exact conditional quantities of (SCK.KM4) and put

$$
\begin{aligned}
B_j={}&-\mathcal G_{N,j}+\alpha H_{N,j}+C_{{\rm col},N,j}
 -(\delta_{*,j}-\varepsilon)M_{C,N,j}+C_{*,j}\mathcal E_{F,N,j},\\
E_n={}&\left[e_0+\sum_{j=0}^{n-1}\mathbb E B_j\right]_+,
\qquad e_0=\mathbb E\mathscr Q(S_0,\widetilde S_0),\\
T_n={}&\mathbb E\bar t_{N,n},\qquad
Z_n=C_QE_n+C_tT_n,
\end{aligned}                                                     \tag{SCK.FTV7}
$$

where $C_Q,C_t$ are the displayed primitive-parameter coefficients
of (SCK.FTV4b). The region labels in $\delta_{*,j},C_{*,j}$ are
those actually reached, including second-force Gaussian excursions.
For every $n\ge0$, $r_0>0$ and fixed $k\le N$,

$$
\boxed{\quad
\|\mathsf T_{k,n+1}^{\rm full}-
 \widetilde{\mathsf T}_{k,n+1}^{\rm full}\|_{\rm TV}
\le\min\{1,k[\Psi(r_0)+Z_n/r_0^2]\}.
\quad}                                                          \tag{SCK.FTV8}
$$

Every term in $B_j,T_n$ is a finite-plan and Gaussian expectation of
the *actual* reward, fitness, cloning, collision, force, cap and
measurement update as specified in (SCK.P1)--(SCK.P4), (SCK.R1)--
(SCK.R3) and (SCK.G3)--(SCK.G8). Formula (SCK.FTV8) is an
$N$-uniform-coefficient TV bound for the full marked state of each
fixed sampled row. If the evaluated signed ledger and mismatch terms
satisfy $Z_n\to0$ uniformly in $N$ within the asserted phase and
$\omega_D(r)\to0$ as $r\downarrow0$, choosing
$r_0=Z_n^{1/4}$ proves full-state TV convergence there at the
explicit bound $k[\Psi(Z_n^{1/4})+Z_n^{1/2}]$.
If the second trajectory is initialized from a separately proved
stationary law of the same phase, this is convergence to that law's
full-state $k$-row marginal. Otherwise it is convergence of the two
compared marginal trajectories, without an assertion that either has
a stationary limit.
The condition $Z_n\to0$ is a mathematical conclusion to be checked
from those signed terms; (SCK.FTV8) does not infer it from the
one-swarm variance drift.
Because the configured Gaussian position noise has unbounded support,
a bounded positional input box is not automatically invariant for
all time. Applying (SCK.FTV8) indefinitely therefore requires the
chapter's actual tail/excursion estimates to validate its
entering-class premises at every step, and the marked balance (SCK.F5)
when deaths or revivals occur. The present theorem applies along
all-alive segments; a compact-support assertion cannot replace those
calculations.

The scale of the *coarse* mismatch envelope is explicit: in the
uniform-companion class (SCK.FTV4c) gives
$\bar t_N\le L_T\sqrt{\mathscr Q}$ after the stated averages.
Replacing the exact residual plan in (SCK.R3) by (SCK.UM1), and
the exact component response by (SCK.UM7), therefore contributes
terms of order $\sqrt{\mathscr Q}$ to the quadratic upper bound.
The kinetic quadratic and the Keystone pressure have orders
$\mathscr Q$ and $\mathscr Q^p$ respectively. Hence those *coarse
upper bounds alone* cannot certify a negative sign at every
arbitrarily small discrepancy. The signed plan and Gaussian
expectations retained in $B_j$ are the quantities to evaluate at
that scale; this observation is not a claim that the actual dynamics
fail to converge.

*Proof.* Conditional expectation of (SCK.KM4) and telescoping give
$e_n\le E_n$. The frozen-velocity and component calculation
(SCK.FTV4b) gives $P_n\le C_Qe_n+C_tT_n\le Z_n$.
Insert this in (SCK.FTV5). The limiting assertion follows by its
stated choice of $r_0$ and continuity of $\Psi$ at zero when
$\omega_D(r)\to0$. Every coefficient in this chain is independent
of population size for fixed $k$; all remaining $N$ dependence is
inside the actual empirical signed terms and self-exclusion factors.
$\square$
:::

:::{prf:theorem} Marked Keystone pressure ported into the actual complete update
:label: thm-slc-marked-keystone-port

Use two nonextinct marked inputs in the bounded entering region and
the *same* comparison-label and source hypotheses as
{prf:ref}`thm-keystone-discharged-averaged-pressure`. In particular,
let $I_{11}$ be their common-alive labels, and let
$e_i=|\Delta\delta_{x,i}|^2$ and
$W=N^{-1}\sum_{i\in I_{11}}e_i$ use that theorem's alive-centered
coordinates. Do not replace these by the full-slot discrepancy in
$\mathscr Q_{\rm marked}$. For each input measurement realization $m$,
let $p_i^m=\sum_{j\ne i}q_i^m(j)$ on an alive row; put
$\bar p_i=\sum_mW_S(m)p_i^m$ and similarly for the second swarm.
The actual conditional activity is
$$
\mathcal A_{11}(S,\widetilde S)=\frac1N\sum_{i\in I_{11}}
      (\bar p_i+\overline{\widetilde p}_i)e_i.
                                                               \tag{SCK.M1}
$$
It is evaluated by (SCK.F1)--(SCK.F2), including the reward and
diversity normalizers and, when selected, their uniform-companion
specialization.

Let $\mathscr E_N^{\rm full}(S,\widetilde S)$ be the *explicit*
right-hand side of (SCK.F5), with $\mathscr P_C$ from (SCK.7),
$\mathscr K$ from (SCK.2), the actual cap term and terminal indicators.
Set the signed remainder by the finite expression
$$
\mathscr R_N^{\rm mark}:=
\mathscr E_N^{\rm full}+\frac\alpha2\mathcal A_{11}.
                                                               \tag{SCK.M2}
$$
This addition and subtraction keeps every accepted donor, revival,
component collision, jitter, two-force kinetic and terminal-status
term in the same expectation. The source Keystone theorem's
all-population estimate (3.CC11a) gives the complete-update inequality
$$
\boxed{\quad
\mathbb E\mathscr Q_{\rm marked}(S^+,\widetilde S^+)
\le\mathscr Q_{\rm marked}(S,\widetilde S)
-\frac\alpha2\chi_*(W-W_0)
+\frac{\alpha B_*}{2N^2}
+\mathscr R_N^{\rm mark}.
\quad}                                                          \tag{SCK.M3}
$$
Here $\chi_*,B_*,W_0$ are *exactly* the source theorem's explicit
primitive-parameter constants, not new fitting or mixing constants.
For uniform measurement and clone companions put
$\kappa_D=\kappa_C=1$ in those constants and use (SCK.UK1) for the
accepted-source array. The alive-mass, velocity and unmatched-label
terms of (3.CC13) remain visible when its structural-error comparison
is used; they are not absorbed into $\chi_*$.

With the noise and sampled-row definitions of
{prf:ref}`thm-slc-keystone-tagged-tv`, the same marked update obeys
$$
\bigl\|\mathsf T_k(S^+)-\mathsf T_k(\widetilde S^+)\bigr\|_{\rm TV}
\le\min\left\{1,
\sqrt{\frac{k}{2\pi\tau^2\lambda_-}}
\left[\mathscr Q_{\rm marked}(S,\widetilde S)
-\frac\alpha2\chi_*(W-W_0)
+\frac{\alpha B_*}{2N^2}
+\mathscr R_N^{\rm mark}\right]_+^{1/2}\right\}.
                                                               \tag{SCK.M4}
$$
The metric and its status coefficient are those in (SCK.7)--(SCK.8).
The position-to-terminal-mark map costs no extra factor in (SCK.M4).
This is the direct marked analogue of (SCK.TV2); no all-row
minorization or additional algorithmic kernel is used.
:::

:::{prf:proof}
Conditional on the complete entering marked states, each $p_i^m$ is
the actual alive recipient's donor-averaged gate probability by
(SCK.F2). Averaging its measurement law gives (SCK.M1), with exactly
the source Keystone theorem's activity normalization. Equation
(SCK.F5) is the exact marked full-update difference, so adding and
subtracting $\alpha\mathcal A_{11}/2$ gives the identity
$\mathbb E\Delta\mathscr Q_{\rm marked}
=-\alpha\mathcal A_{11}/2+\mathscr R_N^{\rm mark}$.
Substitute (3.CC11a), whose hypotheses and constants were retained,
to prove (SCK.M3). The metric $\mathscr Q_{\rm marked}$ dominates
its physical part, and that part dominates
$\lambda_-N^{-1}\sum_i|x_i^+-\widetilde x_i^+|^2$.
The Gaussian and terminal-mark argument in the proof of (SCK.TV1)
therefore applies unchanged and yields (SCK.M4). The specializations
of the companion probabilities follow from (SCK.U4), while the
Keystone source constant depends on their widths only through the
displayed $\kappa_D,\kappa_C$ factors. No status mismatch or revived
row was removed from (SCK.F5).
:::

:::{prf:corollary} The uniform-donor Keystone ledger with no companion mismatch
:label: cor-slc-uniform-keystone-ledger

For two all-alive canonical input swarms with $N\ge2$, specialize only
the clone companion width to $\epsilon_C=\infty$ and keep their
*realized* measurement-dependent fitness arrays $F,\widetilde F$.
In the common-source coupling of
{prf:ref}`prop-cloning-common-source-signed-estimate`, put
$a_{ij}=a(F_i,F_j)$ and
$\widetilde a_{ij}=a(\widetilde F_i,\widetilde F_j)$.
Then every row probability, mismatch and common directed flux is
the following finite array expression:
$$
\begin{gathered}
b_{ij}=\frac{a_{ij}}{N-1},\quad
\widetilde b_{ij}=\frac{\widetilde a_{ij}}{N-1}
 \quad(i\ne j),\qquad
p_i=\frac1{N-1}\sum_{j\ne i}a_{ij},\\
\ell_i=\frac1{N-1}\sum_{j\ne i}|a_{ij}-\widetilde a_{ij}|,
\qquad c_{ij}=\frac{\min(a_{ij},\widetilde a_{ij})}{N-1},\\
C_{HL}=\frac1{N(N-1)}
\sum_{\substack{i\in H,\ j\in L\\j\ne i}}
          \min(a_{ij},\widetilde a_{ij}),\qquad
E_{HL}=\frac1{N(N-1)}
\sum_{\substack{i\in H,\ j\in L\\j\ne i}}
          |a_{ij}-\widetilde a_{ij}|.
\end{gathered}                                                     \tag{SCK.UK1}
$$
The exact $C_{HL}-C_{LH}$ substitutes directly into (C.S3), or its
reward-band lower bound uses (SCK.R1)--(SCK.R2) with
$\kappa_C=1$. The acceptance-gate derivatives from (C.S4) yield
$$
\ell_i\le\min\left\{2,
L_{\rm rec}|F_i-\widetilde F_i|
+\frac{L_{\rm don}}{N-1}
\sum_{j\ne i}|F_j-\widetilde F_j|\right\}.
                                                               \tag{SCK.UK2}
$$
The fitness differences in (SCK.UK2) have the explicit reward and
sampled-diversity normalization bound of
{prf:ref}`lem-slc-common-source-normalizers`; their regional endpoints
are (SCK.G3)--(SCK.G7). Thus the complete cloning preparation
(SCK.1), kinetic polynomial (SCK.2), cap term (SCK.3), Keystone
pressure (SCK.6), and sampled-row TV conversion (SCK.TV2) use
the same arrays without a companion-distance mismatch coefficient.
If measurement width is also infinite, its draw probabilities in
(SCK.F1) are exactly $1/(N-1)$, while the measured separation values
and the fitness arrays remain in these equations.
:::

:::{prf:proof}
Uniform selection assigns probability $1/(N-1)$ to every distinct
alive donor. Multiplication by the unchanged gate and the definitions
of $p_i,\ell_i,c_{ij},C_{HL},E_{HL}$ in (C.S1)--(C.S3) give
(SCK.UK1) term by term. Since the two companion probabilities are
identical, the weight-difference term in (C.S4) is exactly zero.
Apply its already proved recipient and donor gate Lipschitz constants
to each summand and average over $j$ to get (SCK.UK2). All subsequent
statements are direct substitutions into the cited exact identities;
the kinetic, collision and terminal rules have not changed.
:::

:::{prf:lemma} Population-uniform unequal-donor budget in the Keystone balance
:label: lem-slc-uniform-mismatch-budget

Retain the all-alive paired update and row-source coupling of (C.S1).
Suppose the eligible input positions of **each** swarm have diameter
at most $D_x$. No common center or common basin is required. Put
$\bar\ell_N=N^{-1}\sum_i\ell_i$ and
$\bar t_N=N^{-1}\sum_i t_i$. Then the exact unequal-source term in
(SCK.R3) obeys
$$
\boxed{\quad
\mathscr J_C
\le (4D_x^2+d\sigma_J^2)\bar t_N
\le (4D_x^2+d\sigma_J^2)\bar\ell_N.
\quad}                                                        \tag{SCK.UM1}
$$
The multiplier has no $N$ dependence and the bound vanishes when the
accepted row laws match. For uniform clone companions, use the actual
fitnesses after measurement to define
$\overline{\Delta F}_N=N^{-1}\sum_i|F_i-\widetilde F_i|$.
The row calculation (SCK.UK2) gives the further bound
$$
\boxed{\quad
\mathscr J_C\le(4D_x^2+d\sigma_J^2)
\min\{1,(L_{\rm rec}+L_{\rm don})\overline{\Delta F}_N\}.
\quad}                                                        \tag{SCK.UM2}
$$
The same coupling also bounds the center correction in (SCK.R3):
$$
\boxed{\quad
|\bar h|\le\left(1+\sqrt{2/\kappa_C}\right)\sqrt D
              +2D_x\bar t_N,
\qquad
2\bar d\cdot\bar h\le
2|\bar d|\left[
\left(1+\sqrt{2/\kappa_C}\right)\sqrt D
              +2D_x\bar\ell_N\right].
\quad}                                                        \tag{SCK.UM3}
$$
Here $D=N^{-1}\sum_i|d_i-\bar d|^2$ and
$\kappa_C=e^{-D_*^2/(2\epsilon_C^2)}$ is the declared
companion-weight floor. For uniform companions $\kappa_C=1$.
The coefficient is independent of $N\ge2$; the cross term has not
been assigned a favourable sign.

Here $L_{\rm rec},L_{\rm don}$ are their explicit acceptance-gate
constants in (C.S4), and fitness differences receive the reward,
diversity, and normalization estimates of
{prf:ref}`lem-slc-common-source-normalizers`. These are conditional
one-step bounds; averaging over measurement marks uses their actual
law. In the nonuniform companion mode, (C.S4) supplies the additional
explicit weight-difference term inside $\bar\ell_N$.
:::

:::{prf:proof}
Write $\bar x=N^{-1}\sum_i x_i$ and
$\bar y=N^{-1}\sum_i y_i$. For every eligible $j,k$,
$$
x_j-y_k-\bar d=(x_j-\bar x)-(y_k-\bar y),
\qquad |x_j-y_k-\bar d|\le2D_x.
$$
The last inequality follows because every point lies within its
population's diameter of its barycenter. In the definition of
$\mathscr J_C$, discard the nonpositive $-e_i$ and note that the
squared difference of the two copying indicators is at most one.
The residual source-plan mass is $\sum_{j,k}\gamma_{i,jk}=t_i$.
Summing gives the first inequality of (SCK.UM1); $t_i\le\ell_i$
from (C.S1) gives the second.

Under uniform companions, (SCK.UK2) implies
$\ell_i\le L_{\rm rec}|\Delta F_i|
+L_{\rm don}(N-1)^{-1}\sum_{j\ne i}|\Delta F_j|$.
Average over $i$. Each $|\Delta F_j|$ occurs exactly $N-1$
times in the double sum, so
$\bar\ell_N\le(L_{\rm rec}+L_{\rm don})
\overline{\Delta F}_N$ with no population factor.
Also $\bar t_N\le1$. Substitute into (SCK.UM1) to prove
(SCK.UM2).

For the center calculation, split the row expectation defining $h_i$
by common source mass and residual paired mass:
$$
h_i=\sum_{j\ne i}c_{ij}(d_j-d_i)
 +\sum_{j,k}\gamma_{i,jk}(x_j-y_k-d_i).
$$
The residual displacement is bounded by $2D_x$, and its total
average mass is $\bar t_N$. For the common-source part,
Cauchy--Schwarz and $\sum_jc_{ij}\le1$ give
$$
\frac1N\sum_{i,j}c_{ij}|d_i-\bar d|\le\sqrt D.
$$
Each normalized clone weight is at most
$1/[\kappa_C(N-1)]$, so
$\sum_i c_{ij}\le N/[\kappa_C(N-1)]\le2/\kappa_C$.
Another Cauchy--Schwarz bound gives
$$
\frac1N\sum_{i,j}c_{ij}|d_j-\bar d|
\le\sqrt{(2/\kappa_C)D}.
$$
Add these estimates and use $\bar t_N\le\bar\ell_N$.
Cauchy--Schwarz on $2\bar d\cdot\bar h$ proves (SCK.UM3).
$\square$
:::

:::{prf:corollary} Explicit population-uniform positional remainder in the signed update
:label: cor-slc-uniform-positional-remainder

Under the hypotheses of (SCK.R3) and
{prf:ref}`lem-slc-uniform-mismatch-budget`, the same complete update
satisfies
$$
\begin{aligned}
\mathbb E\Delta\mathscr Q\le{}&
-\alpha\sum_{\{H,L\}}(e_H-e_L)
 \frac{\mathcal L^{R,s,1}_{HL}+\mathcal L^{R,s,2}_{HL}-E_{HL}}2
 +\alpha\sum_{H,L}U_{HL}(\rho_H+\rho_L)\\
&+\alpha(4D_x^2+d\sigma_J^2)\bar\ell_N
 +2\alpha|\bar d|
 \left[\left(1+\sqrt{2/\kappa_C}\right)\sqrt D
             +2D_x\bar\ell_N\right]
 +\mathscr T_C+\mathscr K_C .
\end{aligned}                                                     \tag{SCK.UM4}
$$
For uniform companions substitute
$\kappa_C=1$ and
$\bar\ell_N\le\min\{2,
(L_{\rm rec}+L_{\rm don})\overline{\Delta F}_N\}$.
Every multiplier in the second line is independent of population
size. The exact collision and kinetic contributions remain
$\mathscr T_C$ and $\mathscr K_C$ from (SCK.3)--(SCK.4), with
their finite-plan and regional-force evaluation in (SCK.P1)--(SCK.P4)
and (SCK.G8). This inequality is a one-step estimate; its right-hand
side has not been shown negative on every region.
:::

:::{prf:proof}
Substitute (SCK.UM1) and (SCK.UM3) into the exact reward-flux
bound (SCK.R3). Then apply (SCK.UK2) for uniform companions.
No term is dropped from the collision or kinetic part. $\square$
:::

:::{prf:lemma} Collision contribution by matched components and exact graph sensitivity
:label: lem-slc-uniform-collision-budget

Use the same paired source plans and component-Haar coupling as
(SCK.4), with input velocities bounded by $V_{\max}$. Define
$U=N^{-1}\sum_i|u_i|^2$,
$P=N^{-1}\sum_i|d_i|^2$,
$V_c=(1+2|\alpha_{\rm col}|)V_{\max}$ and
$A_c=\max\{1,\alpha_{\rm col}^2\}$.
The accepted rows copy frozen donor **positions**, whereas each
collision component acts on its members' **own frozen velocities**.
Consequently the entering velocity discrepancy is exactly $U$ even
when the two source plans disagree. The source plan affects the
velocity calculation through the two component partitions.
For a paired accepted plan $\mathbf p$, let
$C_i(\mathbf p),\widetilde C_i(\mathbf p)$ be the **actual** two
connected components and set
$$
\theta_N=\sum_{\mathbf p}Q(\mathbf p)
 \frac1N\sum_i\mathbf1_{C_i(\mathbf p)\ne
                         \widetilde C_i(\mathbf p)}.
$$
This is a finite plan sum, not an independent-edge approximation.
With
$$
M_r=\min\left\{(|\bar d|+2D_x)^2+d\sigma_J^2\bar t_N,
\left(1+\frac1{\kappa_C}\right)P+
\big[(|\bar d|+2D_x)^2+d\sigma_J^2\big]\bar t_N\right\},
$$
the collision term in
(SCK.3) satisfies
$$
\boxed{\begin{aligned}
\frac1N\sum_i\mathbb E|z_i|^2
 &\le A_cU+4V_c^2\theta_N,\\
\mathscr T_C
 &\le2|\beta|\left[
   \sqrt{M_r(A_cU+4V_c^2\theta_N)}+\sqrt{PU}\right]
   +\gamma_P\left[(A_c-1)U+4V_c^2\theta_N\right].
\end{aligned}}                                                   \tag{SCK.UM5}
$$
The multipliers are independent of $N$. The exact graph-sensitivity
term $\theta_N$ vanishes if the accepted plans and hence their
components agree. Let $B(\mathbf p)=\{i:J_i\ne\widetilde J_i\}$,
and let $M(\mathbf p)$ be the largest component size in either graph.
The same finite plans give the fully explicit comparison
$$
\theta_N\le
 \sum_{\mathbf p}Q(\mathbf p)
 \min\left\{1,\frac{6M(\mathbf p)|B(\mathbf p)|}{N}\right\},
\qquad \mathbb E|B|=N\bar t_N.                         \tag{SCK.UM6}
$$
Equation (SCK.UM6) retains the correlation between a mismatched edge
and the size of the components it changes. Replacing its expectation
by a product of expectations is unjustified.

The fitness-ordered path count of
{prf:ref}`lem-mean-field-component-bound` gives a population-uniform
coefficient **without** a weak-selection restriction. In the all-alive
class every accepted edge points to strictly higher frozen fitness,
and its probability is at most
$1/[\kappa_C(N-1)]\le (2/\kappa_C)/N$.
The actual independent accepted rows therefore satisfy
$$
\boxed{\quad
\theta_N\le\min\{1,6e^{4/\kappa_C}\bar t_N\}
\le\min\{1,6e^{4/\kappa_C}\bar\ell_N\}.
\quad}                                                        \tag{SCK.UM7}
$$
The coefficient may be loose when $\kappa_C$ is small, but it has no
$N$ dependence. The exact plan quantity (SCK.UM6) remains available
when it is sharper.
:::

:::{prf:proof}
On a matched component $C=\widetilde C$, the shared Haar rotation
and (SCK.4) give
$\mathbb E_R|z_i|^2=|\bar u_C|^2+
\alpha_{\rm col}^2|u_i-\bar u_C|^2$,
where $u_i=v_i-\widetilde v_i$ is the frozen discrepancy of slot $i$.
Summing over every row in that component and using the orthogonal
mean/deviation decomposition bounds its contribution by
$A_c\sum_{i\in C}|u_i|^2$.
For an unmatched component, both collision velocities have norm at
most $V_c$, so $|z_i|^2\le4V_c^2$. Average the plan to obtain the
first line of (SCK.UM5).

For the positional common-source bound, every input label receives
persistence mass at most one and accepted incoming mass at most
$1/\kappa_C$, because $b_{ij}\le1/[\kappa_C(N-1)]$ for the
$N-1$ other recipients. Thus $\sum_i\lambda_{ij}\le
1+1/\kappa_C$, and common-label positional cost is at most
$(1+1/\kappa_C)P$. This column calculation concerns positions only.

Before collision, $r_i=x_{J_i}-y_{\widetilde J_i}$ plus the shared
jitter multiplied by the difference of copying indicators. The two
input diameters give
$|x_{J_i}-y_{\widetilde J_i}|\le|\bar d|+2D_x$.
The centered jitter contributes at most
$d\sigma_J^2\bar t_N$ to the averaged second moment, so
$N^{-1}\sum_i\mathbb E|r_i|^2\le M_r$: the two entries in its
minimum are the coarse diameter bound and the common/residual
source split just proved.
Cauchy--Schwarz yields
$|N^{-1}\sum_i\mathbb E(r_i\cdot z_i)|
\le\sqrt{M_r(A_cU+4V_c^2\theta_N)}$ and
$|N^{-1}\sum_i d_i\cdot u_i|\le\sqrt{PU}$.
Insert these into the definition of $\mathscr T_C$ in (SCK.1).

If the two accepted graphs differ only at rows in $B$, a vertex with
different components must lie in a component, in at least one graph,
containing an endpoint of a changed edge. Each changed row contributes
at most three endpoints ($i,J_i,\widetilde J_i$); taking their
components in both graphs covers at most $6M|B|$ vertices. Divide
by $N$, clip at one, and average over the exact plan probabilities
$Q(\mathbf p)$. Finally independent source rows conditional on
fitness have mismatch probabilities $t_i$, giving
$\mathbb E|B|=\sum_i t_i=N\bar t_N$.

For (SCK.UM7), condition on the two source choices at one mismatched
row $i$. All other rows remain independent, and every remaining
accepted edge has probability at most $C/N$, with $C=2/\kappa_C$.
Seed each graph with the at most three endpoints
$\{i,J_i,\widetilde J_i\}$. Since both ends of the one fixed edge
are seeds, every vertex connected to a seed has a simple path from
some seed avoiding that fixed edge. Accepted edges strictly increase
frozen fitness, so the path has one increasing and one decreasing
leg, exactly as in {prf:ref}`lem-mean-field-component-bound`.
For a path of length $l$ with increasing-leg length $a$, its
conditional expected count from one seed is at most
$C^l/[a!(l-a)!]$. Sum over $a$ to get $(2C)^l/l!$, and then over
$l\ge0$ to get $e^{2C}=e^{4/\kappa_C}$ vertices per seed.
Across three seeds and two graphs, one changed row therefore
affects at most $6e^{4/\kappa_C}$ labels in conditional expectation.
Sum over the row mismatch indicators without separating those
indicators from component size, divide by $N$, and use
$\sum_i\mathbb P(B_i)=N\bar t_N$. Clip at one and apply
$\bar t_N\le\bar\ell_N$. This proves (SCK.UM7). $\square$
:::

:::{prf:corollary} Keystone one-step inequality with population-uniform cloning and collision coefficients
:label: cor-slc-uniform-keystone-prekinetic

Under the preceding regional and bounded-entering-position premises,
let $\Theta_N=\min\{1,6e^{4/\kappa_C}\bar\ell_N\}$ from
(SCK.UM7). The exact finite-plan $\theta_N$ may replace this upper
bound whenever it is smaller.
Put
$$
\begin{aligned}
H_N={}&(4D_x^2+d\sigma_J^2)\bar\ell_N
+2|\bar d|\left[(1+\sqrt{2/\kappa_C})\sqrt D
                    +2D_x\bar\ell_N\right],\\
C_{\rm col,N}={}&2|\beta|
 \left[\sqrt{M_r(A_cU+4V_c^2\Theta_N)}+\sqrt{PU}\right]
 +\gamma_P[(A_c-1)U+4V_c^2\Theta_N].
\end{aligned}
$$
Then the **same signed Keystone calculation** gives
$$
\boxed{\begin{aligned}
\mathbb E\Delta\mathscr Q\le{}&
-\alpha\sum_{\{H,L\}}(e_H-e_L)
 \frac{\mathcal L^{R,s,1}_{HL}+\mathcal L^{R,s,2}_{HL}-E_{HL}}2
 +\alpha\sum_{H,L}U_{HL}(\rho_H+\rho_L)\\
&+\alpha H_N+C_{\rm col,N}+\mathscr K_C.
\end{aligned}}                                                   \tag{SCK.UM8}
$$
The mismatch and collision multipliers in $H_N,C_{\rm col,N}$ are
independent of $N$ for every $\kappa_C>0$; the empirical discrepancies
and regional fluxes retain their actual values. The kinetic term is the signed
two-force and cap expectation (SCK.2)--(SCK.3), evaluated with the
regional and Gaussian-excursion bounds (SCK.G8)--(SCK.G9). No
favourable kinetic sign is inferred from the cloning estimate.
:::

:::{prf:proof}
Start from (SCK.UM4). Replace its $\mathscr T_C$ by (SCK.UM5),
then use (SCK.UM7) or the exact finite-plan $\theta_N$ when sharper.
The square-root and linear expressions are
nondecreasing in that substitution because their coefficients are
nonnegative. This yields (SCK.UM8). $\square$
:::

:::{prf:theorem} Explicit two-force kinetic dissipation test in the Keystone balance
:label: thm-slc-keystone-kinetic-matrix

Use exactly the BAOAB and radial cap of (SCK.2)--(SCK.3). Write
$G=\begin{psmallmatrix}\alpha&\beta\\\beta&\gamma_P\end{psmallmatrix}$,
$\widehat G=G+|\beta|I_2$, and keep their already computed
eigenvalues $\lambda_\pm$. For any real reference curvature $k$,
define the **explicit** two-by-two matrix
$$
T_k=\begin{pmatrix}
1-\eta k&B\\
-ck(a+1-\eta k)&a-ckB
\end{pmatrix},
\qquad
D_k=G-T_k^\top\widehat G T_k,
$$
$$
\delta_k=\frac{D_{k,11}+D_{k,22}
 -\sqrt{(D_{k,11}-D_{k,22})^2+4D_{k,12}^2}}2,
\qquad
t_k^2=\frac{\operatorname{tr}(T_k^\top T_k)
 +\sqrt{(\operatorname{tr}T_k^\top T_k)^2
        -4\det(T_k)^2}}2.                              \tag{SCK.KM1}
$$
Writing $t_{ab}$ for the entries of $T_k$ and
$A=\alpha+|\beta|$, $G_v=\gamma_P+|\beta|$, the entries are
$$
\begin{aligned}
D_{11}&=\alpha-A t_{11}^2-2\beta t_{11}t_{21}-G_vt_{21}^2,\\
D_{22}&=\gamma_P-A t_{12}^2-2\beta t_{12}t_{22}-G_vt_{22}^2,\\
D_{12}&=\beta-A t_{11}t_{12}
 -\beta(t_{11}t_{22}+t_{21}t_{12})-G_vt_{21}t_{22}.
\end{aligned}
$$
Thus $\delta_k>\varepsilon$ holds **exactly when**
$D_{11}>\varepsilon$ and
$(D_{11}-\varepsilon)(D_{22}-\varepsilon)>D_{12}^2$.
Direct multiplication also gives
$\det T_k=a$ and $\operatorname{tr}T_k=1+a-2\eta k$.
When $\gamma>0$ (hence $0<a<1$), a necessary condition for
$\delta_k>0$ is
$$
0<\eta k<1+a,
\qquad\text{equivalently}\qquad 0<k<4/h^2.
                                                               \tag{SCK.KM1a}
$$
Inside this BAOAB stability interval, the preceding two-by-two
determinant test, including the cap charge $|\beta|I_2$, decides
whether the chosen metric certifies kinetic dissipation. Outside it,
the reward flux may still contribute to the complete balance.
These depend only on $h,\gamma,\alpha,\beta,\gamma_P$ and $k$;
$B,\eta,a,c$ retain their configured definitions in (SCK.2).
For a prepared paired row set
$$
e_{0,i}=f_i+k r_i,
\quad e_{1,i}=g_i+k R_i,
\quad E_{k,i}=
\begin{pmatrix}\eta e_{0,i}\\
c(a-k\eta)e_{0,i}+c e_{1,i}\end{pmatrix}.
$$
For every $\varepsilon>0$, with
$\Lambda=\lambda_++|\beta|$ and
$C_{k,\varepsilon}=\Lambda+
\Lambda^2t_k^2/\varepsilon$, the **full capped kinetic step** obeys
$$
\boxed{\quad
\mathscr K_C\le
-(\delta_k-\varepsilon)
 \frac1N\sum_i\mathbb E(|r_i|^2+|z_i|^2)
 +C_{k,\varepsilon}
 \frac1N\sum_i\mathbb E|E_{k,i}|^2.
\quad}                                                        \tag{SCK.KM2}
$$
All coefficients are independent of $N$. The second force defect
$e_{1,i}$ is evaluated at the *random intermediate positions* and
therefore includes the actual OU Gaussian excursions. No force
linearization is assumed: $k$ is a freely chosen reference number,
and $e_0,e_1$ are exact differences.

For the declared basin, passage and exterior partition, choose one
reference $k_{ab,ce}$ for each pair of first-force and second-force
region-pair labels. Apply the inequality pointwise with the chosen
label, and define $\delta_* =\min_{ab,ce}\delta_{k_{ab,ce}}$ and
$C_* =\max_{ab,ce}C_{k_{ab,ce},\varepsilon}$ over the declared
finite table. Then (SCK.KM2) holds with $\delta_*,C_*$ and the
corresponding $E_{k_{ab,ce},i}$. If a profile is unbounded, the
corresponding defect integral may be infinite, identifying an
uninformative certificate rather than undefined dynamics. The
existing (SCK.G8) bounds give, on a labelled excursion,
$$
|e_{0,i}|\le(L^F_{ab}(R_0)+|k|)|r_i|+J^F_{ab}(R_0),
\quad
|e_{1,i}|\le(L^F_{ce}(R_1)+|k|)|R_i|+J^F_{ce}(R_1).
                                                               \tag{SCK.KM3}
$$
For a completely expanded regional excess bound, set
$L_0=L^F_{ab}(R_0)$, $L_1=L^F_{ce}(R_1)$,
$J_0=J^F_{ab}(R_0)$, $J_1=J^F_{ce}(R_1)$,
$A_0=L_0+|k|$, $A_1=L_1+|k|$, and
$C_0=\eta^2+2c^2(a-k\eta)^2$. On this labelled excursion,
$$
\begin{aligned}
|E_{k,i}|^2\le{}&
 [2C_0A_0^2+12c^2A_1^2(1+\eta L_0)^2]|r_i|^2
 +12c^2A_1^2B^2|z_i|^2\\
&+[2C_0+12c^2A_1^2\eta^2]J_0^2+4c^2J_1^2.
\end{aligned}                                                    \tag{SCK.KM3a}
$$
Its expectation uses the actual region labels and OU Gaussian law;
the coefficients contain no $N$. The exact $|E_{k,i}|^2$ should be
used when this triangle bound discards useful force cancellation.
Sharper direct force-increment profiles may replace these triangle
bounds without changing the proof.

The configured cap scale can be retained more sharply than the
universal $|\beta|I_2$ charge. For a pre-cap velocity threshold
$R_v>0$ set
$$
\chi_V(R_v)=1-\left(\frac{V_{\max}}{V_{\max}+R_v}\right)^2,
\quad \widehat G_{R_v}=G+|\beta|\chi_V(R_v)I_2,
\quad D_{k,R_v}=G-T_k^\top\widehat G_{R_v}T_k,
$$
and let $\delta_{k,R_v}$ be its explicit smaller eigenvalue by
(SCK.KM1), replacing $D_k$ with $D_{k,R_v}$.
Put $\Lambda_{R_v}=\lambda_++|\beta|\chi_V(R_v)$ and
$C_{k,\varepsilon,R_v}=\Lambda_{R_v}+
\Lambda_{R_v}^2t_k^2/\varepsilon$. For the actual paired pre-cap
velocities $w_i,\widetilde w_i$, define
$$
\mathcal T_{R_v,N}=\frac1N\sum_i\mathbb E\left[
 \mathbf1_{\{\max(|w_i|,|\widetilde w_i|)>R_v\}}
 (|R_i|^2+|Z_i|^2)\right].
$$
Then the full capped step obeys
$$
\boxed{\begin{aligned}
\mathscr K_C\le{}&-(\delta_{k,R_v}-\varepsilon)
 \frac1N\sum_i\mathbb E(|r_i|^2+|z_i|^2)
 +C_{k,\varepsilon,R_v}
 \frac1N\sum_i\mathbb E|E_{k,i}|^2\\
&+|\beta|[1-\chi_V(R_v)]\mathcal T_{R_v,N}.
\end{aligned}}                                                   \tag{SCK.KM3b}
$$
This displays $V_{\max}$ explicitly. The tail is evaluated with
the same finite-plan Gaussian law as (SCK.P4), including both force
evaluations; the declared Gaussian or moment envelopes may bound it.
Taking $R_v\to\infty$ recovers (SCK.KM2).
:::

:::{prf:proof}
The exact difference map before the final cap is
$(R_i,Z_i)=T_k(r_i,z_i)+E_{k,i}$: substitute
$f_i=-kr_i+e_{0,i}$ and $g_i=-kR_i+e_{1,i}$ into (SCK.2).
The radial cap $C_V(w)=Vw/(V+|w|)$ has symmetric Jacobian with radial
eigenvalue $V^2/(V+|w|)^2$ and tangential eigenvalue
$V/(V+|w|)$, both in $[0,1]$. Integrating that Jacobian along the
segment between paired pre-cap velocities shows
$C_V(w_i)-C_V(\widetilde w_i)=A_iZ_i$ for a symmetric
$0\preceq A_i\preceq I$. Hence the cap correction satisfies
$$
\mathscr C_i
=2\beta R_i\cdot(A_i-I)Z_i
 +\gamma_P(|A_iZ_i|^2-|Z_i|^2)
\le2|\beta||R_i||Z_i|
\le|\beta|(|R_i|^2+|Z_i|^2).
$$
Consequently the completed row quadratic is at most
$(T_kw_i+E_{k,i})^\top\widehat G(T_kw_i+E_{k,i})$,
where $w_i=(r_i,z_i)$ and the matrix acts identically on every
spatial coordinate. Subtract $w_i^\top G w_i$. The leading term is
$-w_i^\top D_kw_i\le-\delta_k|w_i|^2$.
The cross term is bounded by
$2\Lambda t_k|w_i||E_{k,i}|
\le\varepsilon|w_i|^2+
\Lambda^2t_k^2|E_{k,i}|^2/\varepsilon$;
the last quadratic is at most $\Lambda|E_{k,i}|^2$.
For (SCK.KM1a), if $D_k\succ0$, then
$T_k^\top G T_k\prec G$ and the eigenvalues of $T_k$ lie strictly
inside the unit disk. Its characteristic polynomial is
$\lambda^2-(\operatorname{tr}T_k)\lambda+a$.
For $0<a<1$, its roots lie inside that disk exactly when its values
at $+1$ and $-1$ are positive:
$2\eta k>0$ and $2(1+a-\eta k)>0$.
Since $\eta=c^2(1+a)$ and $c=h/2$, these give (SCK.KM1a).
Average over the same paired plans, jitters, Haar rotations and
Gaussian draws as (SCK.3) to obtain (SCK.KM2).
The labelled form follows by a pointwise choice from the finite
table before expectation. Equation (SCK.KM3) is the triangle
inequality applied to (SCK.G8). For (SCK.KM3a), use
$|E|^2\le C_0|e_0|^2+2c^2|e_1|^2$ and
$|e_j|^2\le2A_j^2|\text{input difference}|^2+2J_j^2$.
Also (SCK.G8) gives
$|R|^2\le3[(1+\eta L_0)^2|r|^2+B^2|z|^2+\eta^2J_0^2]$.
Substitution yields the four coefficients in (SCK.KM3a).
For (SCK.KM3b), every Jacobian along the segment between the paired
pre-cap velocities has smallest eigenvalue at least
$[V_{\max}/(V_{\max}+W_i)]^2$, where
$W_i=\max(|w_i|,|\widetilde w_i|)$.
Thus the cap inequality improves to
$\mathscr C_i\le|\beta|\chi_V(W_i)(|R_i|^2+|Z_i|^2)$.
On $W_i\le R_v$ use $\chi_V(W_i)\le\chi_V(R_v)$; on its complement
use $\chi_V(W_i)\le1$. Repeat the matrix expansion with
$\widehat G_{R_v}$ and retain the excess on the complement.
The Gaussian law in (SCK.P4) is the actual law of this event, so no
independence is inserted. Under the stated integrability,
$R_v\to\infty$ gives the coarse bound by dominated convergence.
$\square$
:::

:::{prf:corollary} Obstruction to the coarse cap-charged kinetic certificate
:label: cor-slc-coarse-cap-kinetic-obstruction

In {prf:ref}`thm-slc-keystone-kinetic-matrix`, let $k>0$ and put
$x=hk/2$. Strict positivity of the coarse matrix
$D_k=G-T_k^{\mathsf T}(G+|\beta|I_2)T_k$ requires

$$
\beta>0,\qquad 2-\sqrt3<x<2+\sqrt3.
$$

For the reference curvature $k=1$ and step $h=0.04$, this matrix is
never positive definite, for any positive-definite choice of $G$ and any
friction $\gamma$. This conclusion concerns the coarse cap charge used in
(SCK.KM2). The cap-sensitive estimate (SCK.KM3b) retains its explicit
Gaussian tail, and the finite-population conditioned convergence theorem
{prf:ref}`thm-w2-finite-n-conditioned-convergence` uses a separate smoothing
and survival argument.
:::

:::{prf:proof}
With $c=h/2$, $B=c(1+a)$, and $\eta=cB$, direct substitution gives
$T_k(1,ck)^{\mathsf T}=(1,-ck)^{\mathsf T}$. Therefore, for
$w=(1,x)^{\mathsf T}$,

$$
w^{\mathsf T}D_kw=4\beta x-|\beta|(1+x^2).
$$

For $\beta\le0$ this is nonpositive. For $\beta>0$, its strict
positivity is equivalent to $x^2-4x+1<0$, giving the stated interval.
At $h=1/25$, $k=1$, one has $x=1/50$ and the displayed quadratic form equals
$-2301\beta/2500$ when $\beta>0$; it is nonpositive for the other
signs of $\beta$. Positive definiteness is consequently impossible.
$\square$
:::

:::{prf:corollary} Complete Keystone drift certificate and its parameter regimes
:label: cor-slc-keystone-complete-drift-test

Retain the hypotheses and notation of (SCK.UM8) and choose the
regional reference-curvature table of
{prf:ref}`thm-slc-keystone-kinetic-matrix`. Set
$$
\mathcal E_{F,N}=\frac1N\sum_i
\mathbb E|E_{k_i,i}|^2,
\quad
M_{C,N}=\frac1N\sum_i\mathbb E(|r_i|^2+|z_i|^2),
$$
and define the signed reward-flux amount, including its within-cluster
cost, by
$$
\mathcal G_N=\alpha\sum_{\{H,L\}}(e_H-e_L)
 \frac{\mathcal L^{R,s,1}_{HL}+\mathcal L^{R,s,2}_{HL}-E_{HL}}2
 -\alpha\sum_{H,L}U_{HL}(\rho_H+\rho_L).
$$
Then the
canonical complete update satisfies the explicit inequality
$$
\boxed{\quad
\mathbb E\Delta\mathscr Q\le
-\mathcal G_N+\alpha H_N+C_{\rm col,N}
 -(\delta_*-\varepsilon)M_{C,N}
 +C_*\mathcal E_{F,N}.
\quad}                                                        \tag{SCK.KM4}
$$
When $\tau^2=c^2q^2+s^2>0$, the Gaussian comparison of
{prf:ref}`thm-slc-keystone-tagged-tv` gives the corresponding
one-step TV estimate for $k$ uniformly sampled marked positions:
$$
\left\|\mathsf T_k(S^+)-\mathsf T_k(\widetilde S^+)\right\|_{\rm TV}
\le\min\left\{1,
\sqrt{\frac{k}{2\pi\tau^2\lambda_-}}
\left[\mathscr Q(S,\widetilde S)-\mathcal G_N+
\alpha H_N+C_{\rm col,N}-(\delta_*-\varepsilon)M_{C,N}
+C_*\mathcal E_{F,N}\right]_+^{1/2}\right\}.       \tag{SCK.KM5}
$$
Its coefficient is independent of $N$ for fixed $k$.
All five terms are computed from (SCK.R1)--(SCK.R3),
(SCK.UM1)--(SCK.UM7), (SCK.KM1)--(SCK.KM3), and the finite-plan
Gaussian integral (SCK.P1)--(SCK.P4). Their scalar multipliers do
not depend on $N$ for every $\kappa_C>0$. In particular, the following tests
separate the full parameter regime without changing the algorithm:

1. The fitness-ordered component bound (SCK.UM7) gives the
   population-uniform sensitivity coefficient
   $6e^{4/\kappa_C}$. Narrow companion weights make it large;
   uniform companions set $\kappa_C=1$. The exact finite-plan
   $\theta_N$ may give a smaller value at a particular state.
2. Each oriented regional edge contributes favourably when its
   explicitly evaluated
   $\mathcal L^{R,s,1}_{HL}+\mathcal L^{R,s,2}_{HL}-E_{HL}$ is
   positive. Its contribution is weighted by the actual error gap
   $e_H-e_L$; the within-cluster price is the displayed
   $\sum U_{HL}(\rho_H+\rho_L)$. No favourable sign is assigned to
   an edge whose evaluated bracket is nonpositive.
3. $\delta_*>\varepsilon$ gives a strictly negative kinetic
   quadratic contribution. If $\delta_*\le\varepsilon$, the kinetic
   matrix certificate contributes no negative quadratic term, and
   the evaluated reward flux must cover its signed remainder.
   For $\gamma>0$, a necessary kinetic-only range for each reference
   $k$ is $0<k<4/h^2$; the exact determinant inequalities after
   (SCK.KM1) decide the chosen metric within that range.
4. The force-excursion cost is the explicit $C_*\mathcal E_{F,N}$.
   It is zero for an exactly linear force with the selected $k_i$
   on both evaluations, and otherwise is evaluated using the actual
   labelled excursions. The OU scale $q$ enters these intermediate
   excursion probabilities even though shared additive Gaussian
   noise cancels from the paired difference.
5. The complete update has certified negative drift at the entering
   paired state whenever the *calculated* inequality
   $$
   \mathcal G_N+(\delta_*-\varepsilon)M_{C,N}
   >\alpha H_N+C_{\rm col,N}+C_*\mathcal E_{F,N}
   $$
   holds. This is a sufficient numerical test using the declared
   landscape profiles and configured parameters. A failed test is
   inconclusive; no sign is assigned by definition.

For a cap-sensitive test, replace $\delta_*,C_*$ by the minimum and
maximum of $\delta_{k,R_v},C_{k,\varepsilon,R_v}$ over the same
regional table and add
$|\beta|[1-\chi_V(R_v)]\mathcal T_{R_v,N}$ to the right-hand side
of (SCK.KM4). This is exactly (SCK.KM3b), not an altered update.

The result is conditional on the stated all-alive bounded entering
class. For killed or revived marked states, apply the separate exact
status term of (SCK.F5); (SCK.KM4) alone does not erase it.
:::

:::{prf:proof}
Substitute (SCK.KM2), with the pointwise regional table, for
$\mathscr K_C$ in (SCK.UM8). The remaining terms are unchanged.
Apply (SCK.TV1) to that same coupled output to get (SCK.KM5).
The five tests are direct comparisons of explicit terms in the
resulting inequality, with no replacement of the signed reward flux
by an absolute-value error. $\square$
:::

:::{prf:remark} Global quadratic closure and the terminal and revival obstructions
:label: rem-slc-global-quadratic-closure-obstructions

The all-alive preterminal calculation (SCK.KM4) retains its stated
conditional hypotheses. A global terminal marked estimate with a
population-independent contraction factor and a residual tending to
zero is false in the normalized quadratic used here. For every
positive status coefficient, even separately survival-conditioned
consensus output laws have a first-order status discrepancy against
a second-order entering physical discrepancy, as proved in
{prf:ref}`prop-ku-terminal-mark-quadratic-obstruction`.
Setting the status coefficient to zero leaves the global nonextinct
revival obstruction of
{prf:ref}`prop-ku-singleton-revival-uniform-obstruction`.
On all-alive physical inputs, the prescribed common-source and shared-noise
coupling has the separate near-tie obstruction of
{prf:ref}`thm-ku-allalive-source-coupling-obstruction`.
This is a statement about a fixed labeled coupling, not a lower bound
on optimal alive empirical transport or on alive-sampled marginal
transport. The relevant infimum and alive-normalization comparisons
are stated in {prf:ref}`prop-w2-prescribed-coupling-scope`.

Consequently the statewise strict-drift test in (SCK.KM4) is not an
unverified global quadratic contraction hypothesis that can be imposed
on every reachable state. An iterated phase-specific estimate must
prove its phase and excursion conditions. A uniform alive-transport
proof may use another coupling in the same Wasserstein metric; a
survivor-block TV route must verify its comparison directly. A shared
phase label alone does not establish contraction, as
{prf:ref}`rem-w2-phase-versus-law-convergence` explains.
The exact signed identities and pressure coefficients
remain available for those calculations.
:::

:::{prf:remark} Kernel to which the one-step identities apply
:label: rem-slc-one-step-kernel-scope

The collision specialization (SCK.4) and (SCK.7), and the uses of those
formulas in (SCK.1)--(SCK.9), are for
{prf:ref}`def-inelastic-collision-update`: one simultaneous update of
each connected component using one Haar orthogonal rotation. The
original one-step mean-field bounds in Chapters 8--9 use this same declared
Volume 2 kernel. The current Python `clone_walkers` implementation in
`src/fragile/fractalai/core/cloning.py` instead visits donor-centered
groups sequentially, uses unrotated restitution, and can write a shared
vertex more than once. Its collision transition is therefore different;
the following theorem supplies its distinct collision terms while retaining
the common signed cloning and kinetic identities for the declared
canonical update. This collision substitution is not a theorem for the
entire default `EuclideanGas.step` routine: that routine's kinetic noise,
velocity cap, boundary timing and optional substeps must be matched
separately before a full executable-kernel claim is made.
:::

:::{prf:theorem} Ordered donor-star collision and its complete one-step balance
:label: thm-slc-ordered-collision-balance

Fix a complete accepted plan $P=((A_i,J_i))_{i=1}^N$ and frozen incoming
velocities $v_i$. This theorem treats the ordered donor-star collision of
`inelastic_collision_velocity` in `src/fragile/fractalai/core/cloning.py`;
all other stages and their parameters are those in
{prf:ref}`thm-slc-signed-complete-update`. Assume
$0\le\alpha_{\rm col}\le1$ and the integrability required there. Set
$$
D(P)=\{J_i:A_i=1\},\qquad
G_c(P)=\{c\}\cup\{i\ne c:A_i=1,\ J_i=c\},
\qquad
L_i(P)=\max\{c\in D(P):i\in G_c(P)\},
$$
where $L_i=\bot$ if the last set is empty. Define the row-stochastic matrix
$$
T_{ij}(P)=
\begin{cases}
\mathbf1_{j=i},&L_i=\bot,\\
\alpha_{\rm col}\mathbf1_{j=i}
 +(1-\alpha_{\rm col})|G_{L_i}|^{-1}\mathbf1_{j\in G_{L_i}},
 &L_i\ne\bot.
\end{cases}                                                   \tag{SCK.O1}
$$
The exact collision output is $V_i^C=\sum_jT_{ij}(P)v_j$.
For paired accepted plans $P,\widetilde P$, put
$z_i^{\rm ord}=\sum_jT_{ij}(P)v_j-
\sum_jT_{ij}(\widetilde P)\widetilde v_j$.
Conditional on both plans, the collision has no further randomness:
$$
\mathbb E_Cz_i=z_i^{\rm ord},\quad
\mathbb E_C|z_i|^2=|z_i^{\rm ord}|^2,\quad
\mathbb E_C(r_i\cdot z_i)=r_i\cdot z_i^{\rm ord}.             \tag{SCK.O2}
$$
The last expectation fixes the cloning jitter; its centered part vanishes
when that jitter is subsequently averaged.

In the all-alive source-label coupling of (C.S1), let
$Q(\mathbf p)=\prod_i\Pi_i(j_i,k_i)$ be the finite plan law of
{prf:ref}`prop-slc-reward-full-plan`. The accepted plan $P_j$ has
$A_i=\mathbf1_{j_i\ne i}$ and $J_i=j_i$ when $A_i=1$, and similarly for
$\widetilde P_k$. Then the exact collision part of (SCK.1) is
$$
\begin{aligned}
\mathscr T_C^{\rm ord}
=\sum_{\mathbf p}Q(\mathbf p)\frac1N\sum_i\bigl\{&
2\beta[(x_{j_i}-y_{k_i})\cdot z_i^{\rm ord}(\mathbf p)-d_i\cdot u_i]\\
&+\gamma_P[|z_i^{\rm ord}(\mathbf p)|^2-|u_i|^2]\bigr\}.
\end{aligned}                                                   \tag{SCK.O3}
$$
Equations (SCK.1)--(SCK.3), (SCK.5)--(SCK.6), the regional kinetic
formula and the signed reward substitutions hold for this collision mode
with $z=z^{\rm ord}$ and $\mathscr T_C=\mathscr T_C^{\rm ord}$.
In their finite-plan Gaussian integrals, replace the component-Haar law
by the point mass at $T(P_j),T(\widetilde P_k)$; retain the same jitter,
OU and final-position Gaussian laws. In particular, the exact kinetic
residual is
$$
\mathscr K_C^{\rm ord}=\sum_{\mathbf p}Q(\mathbf p)
\int\frac1N\sum_i[\mathscr K(r_i,z_i,f_i,g_i)+\mathscr C_i]
\,\nu^{\rm ord}_{\mathbf p}(d\omega),                       \tag{SCK.O4}
$$
where $\nu^{\rm ord}_{\mathbf p}$ is that specified joint Gaussian law.
Thus both modes retain the identical explicit Keystone contribution,
while their collision residuals are evaluated by (SCK.4) and (SCK.O3),
respectively.

For any two nonextinct marked inputs, use their actual complete accepted
plan laws, retaining the acceptance bit even if a possible self-clone
has $J_i=i$. Define $\delta_i,j_i$ as in (SCK.7). The ordered replacement
of its physical preparation term is exactly
$$
\begin{aligned}
\mathscr P_C^{\rm ord}(P,\widetilde P)=\frac1N\sum_i\bigl\{&
\alpha[2d_i\cdot\delta_i+|\delta_i|^2+d\sigma_J^2j_i^2]\\
&+2\beta[(d_i+\delta_i)\cdot z_i^{\rm ord}-d_i\cdot u_i]
+\gamma_P[|z_i^{\rm ord}|^2-|u_i|^2]\bigr\}.             \tag{SCK.O5}
\end{aligned}
$$
If $\widehat Q(P,\widetilde P)$ is the actual paired complete-plan
probability and $\nu^{\rm ord}_{P,\widetilde P}$ is the joint law of the
shared cloning jitter, OU and final-position Gaussian draws, the exact
marked complete-update identity is
$$
\begin{aligned}
\mathbb E\Delta\mathscr Q_{\rm marked}^{\rm ord}
=\sum_{P,\widetilde P}\widehat Q(P,\widetilde P)
\int\bigg[&\mathscr P_C^{\rm ord}(P,\widetilde P)
+\frac1N\sum_i(\mathscr K_i+\mathscr C_i)\\
&+\frac{\eta_A}{N}\sum_i
 (\mathbf1_{a_i^+\ne\widetilde a_i^+}
  -\mathbf1_{a_i\ne\widetilde a_i})\bigg]
\,\nu^{\rm ord}_{P,\widetilde P}(d\omega).              \tag{SCK.O6}
\end{aligned}
$$
This includes mandatory revival, cap and terminal classification.
The accepted-plan probabilities must be those of the kernel being evaluated;
for a kernel allowing accepted self-clones, a source-label-only plan is
insufficient because acceptance causes positional jitter.

The matrix bounds, independent of $N$ and of the accepted plan, are
$$
\|T(P)\|_{\infty\to\infty}=1,\qquad
\|T(P)\|_{1\to1}\le2-\alpha_{\rm col},\qquad
\|T(P)\|_{2\to2}^2\le2-\alpha_{\rm col}.                    \tag{SCK.O7}
$$
Consequently $\max_i|V_i^C|\le\max_j|v_j|$ and
$N^{-1}\sum_i|V_i^C|^2\le(2-\alpha_{\rm col})N^{-1}\sum_i|v_i|^2$.
If $\max_j|\widetilde v_j|\le V_*$ and
$\theta_i=\frac12\sum_j|T_{ij}(P)-T_{ij}(\widetilde P)|$, then
$$
|z_i^{\rm ord}|\le\sum_jT_{ij}(P)|u_j|+2V_*\theta_i.        \tag{SCK.O8}
$$
These estimates retain any mismatch caused by overlapping donor stars;
they do not assert global energy dissipation of their final composition.
:::

:::{prf:proof}
For each donor $c$, the routine computes the mean over $G_c$ from the
incoming velocity array, then writes
$\alpha_{\rm col}v_i+(1-\alpha_{\rm col})|G_c|^{-1}
\sum_{j\in G_c}v_j$ to all rows in that star. It visits distinct donor
indices in increasing order. The last star containing $i$ therefore
determines its final output, proving (SCK.O1)--(SCK.O2), including star
overlaps and singleton accepted self-clones.

The position-source and jitter laws are unchanged by the collision rule.
Hence (C.S1), its barycenter correction and the Keystone bound remain
exactly the same. Expanding $2\beta r_i\cdot z_i+
\gamma_P|z_i|^2$ conditional on the plans, then averaging centered
jitter, gives (SCK.O3). For any prepared $r,z$, BAOAB gives the same
differences $R=r+Bz+\eta f$ and $Z=az+acf+cg$; expansion gives (SCK.2)
and the cap term without a collision assumption. Integrating those actual
quantities over the same plans and Gaussian innovations proves (SCK.O4)
and the asserted full-update identities. The marked cloned positional
difference is $d_i+\delta_i+\sigma_Jj_i\xi_i$. Its conditional squared
expectation is $|d_i+\delta_i|^2+d\sigma_J^2j_i^2$, while its cross
expectation with the deterministic $z_i^{\rm ord}$ is
$(d_i+\delta_i)\cdot z_i^{\rm ord}$. This proves (SCK.O5).
The later kinetic, cap and terminal calculations depend on the prepared
coordinates, so their existing algebra applies to those exact inputs.
Conditional on the complete paired plan, all remaining innovations have
the specified law $\nu^{\rm ord}_{P,\widetilde P}$. Averaging the
preparation, kinetic polynomial and terminal indicator first under this
law and then under the finite plan law proves (SCK.O6), without
factorizing dependent collision and kinetic outcomes.

Every row of $T$ has nonnegative entries summing to one. A fixed label $j$
belongs to at most two stars: its own donor star and the star of its one
accepted outgoing edge. Its diagonal $\alpha_{\rm col}$ appears in at most
one final row, and its mean contribution in any star is at most
$(1-\alpha_{\rm col})|G_c|/|G_c|$. If no star contains $j$, its column
sum is one. Thus every column sum is at most $2-\alpha_{\rm col}$.
The induced $\ell^2$ bound follows from
$\|T\|_2^2\le\|T\|_1\|T\|_\infty$; Jensen's inequality gives the
empirical squared-velocity bound. Finally
$z^{\rm ord}=T(P)u+[T(P)-T(\widetilde P)]\widetilde v$,
and its rowwise triangle inequality is (SCK.O8).
:::

:::{prf:lemma} Regional evaluation of the signed kinetic and cap terms
:label: lem-slc-signed-kinetic-regions

In (SCK.2) retain each force dot product with its displayed sign. For
the declared regions $A_a$, its first-force increments obey
$|f_i|\le\omega_{ab}(|r_i|)$ on $X_i\in A_a,Y_i\in A_b$;
its second-force increments obey
$|g_i|\le\omega_{ab}(|R_i|)$ on
$L_i\in A_a,\widetilde L_i\in A_b$. These bounds follow directly from
the already declared force-increment profiles. In particular, conditioning
on preparation,
$$
\mathbb E_\xi|g_i|^2\le
\sum_{a,b}\int_{\mathbb R^d}
\mathbf1_{\{L_i^0+cq\xi\in A_a,\ \widetilde L_i^0+cq\xi\in A_b\}}
\omega_{ab}(|R_i|)^2\varphi_d(\xi)\,d\xi,
$$
where $L_i^0=X_i+BV_i^C+\eta F(X_i)$ and its tilded counterpart
is defined analogously. The exterior region is included in the sum.
Here $R_i=L_i^0-\widetilde L_i^0$ does not depend on $\xi$.
The same Gaussian integral with the actual dot product, instead of its
modulus bound, evaluates every second-force signed term in (SCK.2).

For a fully explicit cap evaluation, define
$$
M_i=\int_0^1DC_V(\widetilde w_i+tZ_i)\,dt,
\quad
DC_V(w)=\frac{V}{V+|w|}I-\frac{V}{(V+|w|)^2}
                      \frac{ww^T}{|w|},
$$
with $DC_V(0)=I$. Then $k_i=(M_i-I)Z_i$, $0\preceq M_i\preceq I$,
and
$$
\mathscr C_i
=2\beta R_i\cdot(M_i-I)Z_i
 -\gamma_P Z_i^T(I-M_i^2)Z_i.
$$
Thus the dissipative velocity-cap contribution remains signed; the
position–velocity cap contribution remains in the same metric. These
formulas evaluate the actual complete-update balance for the declared
basin, transition and exterior profiles. A strict rate follows only
from an upper estimate of their *combined* right-hand side; no
independent sign is required for each kinetic cross term.
:::

:::{prf:proof}
The increment-profile definition applies pointwise at both actual pairs
of force-evaluation locations. The second pair has the common translated
Gaussian law displayed above. Partition that Gaussian integral by the
regions and apply Tonelli to its nonnegative square bound. Signed
integrals are justified by the integrability in the preceding theorem.
The cap derivative has tangential eigenvalue $V/(V+|w|)$ and radial
eigenvalue $V^2/(V+|w|)^2$, with continuous value one at zero.
The fundamental theorem of calculus along the segment gives $M_i$,
which is symmetric with spectrum in $[0,1]$. Substituting
$k_i=(M_i-I)Z_i$ into its exact quadratic increment proves the formula.
:::

:::{prf:corollary} Revival and terminal classification in the same signed update
:label: cor-slc-signed-marked-update

For nonextinct marked inputs retain every physical slot, including dead
positions and frozen dead velocities, and use their actual mandatory
revival plans. Condition on the actual accepted plans of both swarms.
Write $A_i,\widetilde A_i$ for acceptance indicators and
$$
\delta_i=A_i(x_{J_i}-x_i)
 -\widetilde A_i(y_{\widetilde J_i}-y_i),\qquad
j_i=A_i-\widetilde A_i.
$$
Products with zero acceptance are zero. With the configured positional
jitter $\sigma_J$, the exact preparation increment of the same physical
metric is
$$
\begin{aligned}
\mathscr P_C={1\over N}\sum_i\{&
\alpha[2d_i\cdot\delta_i+|\delta_i|^2+d\sigma_J^2j_i^2]\\
&+2\beta[(d_i+\delta_i)\cdot(m_i-\widetilde m_i)-d_i\cdot u_i]\\
&+\gamma_P[|m_i-\widetilde m_i|^2+
 \alpha_{\rm col}^2(|b_i|^2+|\widetilde b_i|^2
 -2\mathbf1_{C_i=\widetilde C_i}b_i\cdot\widetilde b_i)-|u_i|^2]\}.
\end{aligned}                                                     \tag{SCK.7}
$$
All components include the actual revived leaves and retained dead
velocities. Averaging (SCK.7) over the actual accepted plans, and adding
$N^{-1}\sum_i\mathbb E[\mathscr K_i+\mathscr C_i]$, gives the exact
physical complete-update increment for these marked inputs. Thus the
kinetic polynomial and its signed cross terms require no all-alive
simplification. The all-alive pressure specialization (SCK.5) or
(SCK.6) is used only on the respective family covered by its cited
Keystone theorem; a marked pressure bound acts on its stated common-alive
index set, with the other terms of (SCK.7) retained.

Let $a_i^+,\widetilde a_i^+$ be the actual terminal classifications, and
add a status cost $\eta_A N^{-1}\sum_i\mathbf1_{a_i\ne\widetilde a_i}$
with a declared $\eta_A\ge0$. Its additional exact increment is
$$
{\eta_A\over N}\sum_i
 [\mathbb P(a_i^+\ne\widetilde a_i^+)-\mathbf1_{a_i\ne\widetilde a_i}].
                                                               \tag{SCK.8}
$$
For the canonical positional validity region $\mathcal X_{\rm valid}$,
conditional on preparation and OU innovations this mismatch probability
is the explicit Gaussian integral
$$
\int\left|
\mathbf1_{\mathcal X_{\rm valid}}(L_i+s\zeta)
-\mathbf1_{\mathcal X_{\rm valid}}(\widetilde L_i+s\zeta)
\right|\varphi_d(\zeta)\,d\zeta.
$$
For the canonical box and $s>0$, this integral is at most
$$
\min\left\{1,\frac{2\|R_i\|_1}{s\sqrt{2\pi}}\right\}.
$$
This coefficient uses the configured $s=\sigma_x\sqrt h$ and has no
population-size factor. At $s=0$ retain the displayed exact indicator
integral. This retains the true terminal marking, including possible full death.
It is a one-step identity for nonextinct inputs; conditioning successive
updates on survival uses the previously established survival filter,
not a replacement transition kernel.

The marked one-step expectation also has an explicit accepted-plan
expansion. Conditional on both retained fitness vectors, a live row
with an eligible distinct donor has source distribution
$$
b_{ij}=P_C(j\mid i)a(F_i,F_j)\quad(j\ne i),\qquad
p_i^{\rm tot}=\sum_{j\ne i}b_{ij},\qquad
q_{ij}=(1-p_i^{\rm tot})\mathbf1_{j=i}+b_{ij},
$$
with $P_C$ normalized on the *current* alive donors other than $i$.
Here $p_i^{\rm tot}$ is the donor-averaged acceptance probability;
it is distinct from the donor-conditional gate $p_i$ in
{prf:ref}`def-eg-component-collision`.
A live row that is the sole alive walker persists, so $q_{ii}=1$.
A dead row has $q_{ij}=P_C(j\mid i)$ on the current nonempty alive
donor set because revival accepts with probability one. The tilded
swarm follows the same rules, with its own current alive set. Couple
these two categorical rows by their common mass and residual product
as in (C.S1), writing the resulting marked coupling as $\Pi_i$;
the all-alive companion lower bounds of (C.S4) are not assigned to
this general marked coupling.
The complete paired plan $\mathbf p=((j_i,k_i))_i$ has probability
$Q(\mathbf p)=\prod_i\Pi_i(j_i,k_i)$. Let $\nu_{\mathbf p}$ denote the
specified joint law of its component Haar rotations, shared row jitters,
shared OU Gaussians and shared final position Gaussians. With
$\mathscr P_C(\mathbf p)$ from (SCK.7), the same complete update obeys

$$
\begin{aligned}
 \mathbb E\Delta\mathscr Q_{\rm marked}
 =\sum_{\mathbf p}Q(\mathbf p)\int\bigg[&
 \mathscr P_C(\mathbf p)
 +\frac1N\sum_i(\mathscr K_i+\mathscr C_i)\\
 &+\frac{\eta_A}{N}\sum_i
   (\mathbf1_{a_i^+\ne\widetilde a_i^+}
    -\mathbf1_{a_i\ne\widetilde a_i})\bigg]
 \,\nu_{\mathbf p}(d\omega).
 \tag{SCK.9}
\end{aligned}
$$

The kinetic terms are constructed from the actual jittered and collided
states of the plan, and the terminal indicators use those same kinetic
innovations. Thus (SCK.9) evaluates the one-step reward dependence of
mandatory revival, collisions, kinetics and terminal classification
without treating their outcomes as independent. For all-alive inputs
its plan law reduces to (SCK.P1).
:::

:::{prf:proof}
Conditional on accepted plans, the positional discrepancy after jitter
is $d_i+\delta_i+\sigma_Jj_i\xi_i$. Its squared expectation gives the
first line of (SCK.7). Jitter is centered and independent of the component
Haar matrices, whose conditional first and second moments are (SCK.4).
These yield the other two lines, equivalently the full-slot balance of
{prf:ref}`thm-cloning-incremental-cluster-balance`. The subsequent physical
BAOAB and cap algebra is identical for revived slots. Terminal marks
are deterministic functions of the completed positions; expectation of
their indicator mismatch gives (SCK.8) and the displayed Gaussian formula. For the box bound, the symmetric difference of two coordinate intervals translated by $R_{i,r}$ has length at most $2|R_{i,r}|$. Bound the one-dimensional Gaussian density by $1/(s\sqrt{2\pi})$, sum over coordinates, and truncate at one.
Finally the row donor, gate and mandatory-revival draws are independent
conditional on the frozen fitness vectors, giving the product plan law.
Conditional on a plan, all remaining innovations have the joint law
$\nu_{\mathbf p}$. Applying the law of total expectation to (SCK.7),
the complete kinetic polynomial and the terminal indicator proves
(SCK.9); the shared Gaussian draws are retained inside its integral.
In that integral $\mathscr P_C(\mathbf p)$ is already the conditional
expectation over jitter and Haar rotations, and is therefore constant
with respect to those two integrations. The kinetic and terminal terms
use the realized innovations.
:::

:::{prf:theorem} Fully parameterized one-step chain for the canonical marked gas
:label: thm-slc-full-parameter-one-step-chain

Let $S=((x_i,v_i,a_i))_{i=1}^N$ have finite coordinates. Use exactly
the measurement, fitness, component collision and kinetic rules of
{prf:ref}`alg-euclidean-gas`. The input parameters of this statement are
the population size $N$, dimension $d$, feature radii $R_x,V_{\rm alg}$,
feature velocity weight $\lambda_{\rm alg}$, companion widths
$\epsilon_D,\epsilon_C$, separation floor $\delta_D$, raw reward $R$,
fitness variance floors $\sigma_{\min,r},\sigma_{\min,d}$, logistic
amplitudes $A_r,A_d$, logistic floors $\eta_r,\eta_d$, fitness powers
$\alpha_r,\alpha_d$, gate parameters $p_{\max},\varepsilon_{\rm clone}$,
clone jitter $\sigma_{\rm clone}$, restitution $\alpha_{\rm restitution}$,
step size $h$, friction $\gamma$, thermostat scale $\sigma_v$, final
position scale $\sigma_x$, acceleration $F$, and terminal region $D$.
Here $\alpha_r,\alpha_d$ are the exponents denoted $\alpha,\beta$ in
{prf:ref}`def-eg-frozen-measurements`; they are distinct from the
comparison-metric coefficients $\alpha,\beta,\gamma_P$ of (SCK.7).
These are algorithm inputs, with the positivity and range conventions
of their cited definitions, rather than added landscape hypotheses.

If $M=\sum_i a_i=0$, the full kernel is the point mass at $S$. For
$M>0$, let $\mathcal M(S)$ be the finite set of possible measurement
label vectors $m=(m_i)_{i\in\mathcal A}$. Its exact mass is
$$
 W_S(m)=\prod_{i\in\mathcal A}P_D^N(i,m_i),                    \tag{SCK.F1}
$$
where the alive singleton has one deterministic zero-distance
measurement. For each $m$, evaluate the raw reward and separation
arrays, their *common* alive means and regularized variances, and the
frozen fitness vector $F^m$ by
{prf:ref}`def-eg-frozen-measurements`. For every row define the
accepted-source distribution
$$
q_i^m(j)=
\begin{cases}
P_C^N(i,j)\,a(F_i^m,F_j^m),&a_i=1,\ j\ne i,\ M\ge2,\\
1-\sum_{k\ne i}P_C^N(i,k)a(F_i^m,F_k^m),&a_i=1,\ j=i,\ M\ge2,\\
1,&a_i=1,\ M=1,\ j=i,\\
P_C^N(i,j),&a_i=0,\ j\in\mathcal A,\\
0,&\text{otherwise},
\end{cases}                                                       \tag{SCK.F2}
$$
where
$a(u,w)=\min\{1,[w-u]_+/[p_{\max}(u+\varepsilon_{\rm clone})]\}$.
In the dead-row line $P_C^N$ is normalized over all currently alive
donors, including the sole survivor. Hence each $q_i^m$ sums to one.
The source label $j=i$ means no accepted edge for an alive row;
every dead row has an accepted edge. Conditional on $m$, the rows are
independent, so a complete accepted plan $p=(j_i)_{i=1}^N$ has mass
$$Q_S^m(p)=\prod_{i=1}^N q_i^m(j_i).                         \tag{SCK.F3}$$

Let $H_p$ be the product of normalized Haar laws for the nontrivial
connected components of the accepted-edge graph. Let $G_p$ be the
product of independent standard $d$-Gaussian laws for accepted-row
jitters, every row's OU innovation and every row's final position
innovation. Write $T_{S,p}(R_C,\zeta,\xi,\chi)$ for the deterministic
map that copies frozen donor positions with jitter, transforms *all*
velocities in each component with its one shared rotation, revives
every slot, applies B1--A1--O--A2--B2, adds final position noise,
caps velocities and finally applies $\mathbf1_D$ to the positions.
The complete marked transition has the explicit finite-sum integral
$$
 \boxed{\quad
 \Psi_N\Phi(S)=\sum_{m\in\mathcal M(S)}W_S(m)
 \sum_p Q_S^m(p)
 \int\Phi\bigl(T_{S,p}(R_C,\zeta,\xi,\chi)\bigr)
                         \,H_p(dR_C)G_p(d\zeta,d\xi,d\chi).
 \quad}                                                          \tag{SCK.F4}
$$
This holds for every bounded measurable $\Phi$ without a force
Lipschitz, confinement, convexity or reward-gap condition: it is an
identity for the specified transition whenever its point evaluations
are defined.

For two nonextinct inputs $S,\widetilde S$, use a **diagonal-preserving**
measurement coupling. Extend each row's measurement-label space by a
sentinel $\bot$: a dead row has the point mass at $\bot$, an alive
singleton has its deterministic zero-distance label, and any other
alive row has the actual categorical distribution $r_i^S(j)=P_D^N(i,j)$.
Write $\widetilde r_i$ for the second swarm and set
$$
\begin{gathered}
\lambda_i^D(j)=\min\{r_i^S(j),\widetilde r_i(j)\},\qquad
t_i^D=1-\sum_j\lambda_i^D(j)
       =\frac12\sum_j|r_i^S(j)-\widetilde r_i(j)|,\\
\Pi_i^D(j,k)=\lambda_i^D(j)\mathbf1_{j=k}
 +\mathbf1_{t_i^D>0}
 \frac{[r_i^S(j)-\lambda_i^D(j)]
       [\widetilde r_i(k)-\lambda_i^D(k)]}{t_i^D},\\
\Lambda(m,\widetilde m)=\prod_{i=1}^N\Pi_i^D(m_i,\widetilde m_i).
\end{gathered}                                                     \tag{SCK.F4a}
$$
Each row of $\Pi_i^D$ has the required two categorical marginals;
the product preserves independence of measurement labels **within**
each swarm. At $S=\widetilde S$ it is supported on $m=\widetilde m$,
so the two complete fitness vectors, including their shared
normalizers, agree. The independent measurement coupling is also
valid for the identity below, but it is unsuitable for a
diagonal-vanishing contraction comparison. Given the two measurement
vectors, couple the
source rows using their common categorical mass and residual product
as in (C.S1), including the marked distributions (SCK.F2). Use
independent paired rows, matched-component Haar rotations, shared
row jitter and shared kinetic Gaussians exactly as in (SCK.9). If the
displayed quadratic terms are absolutely integrable, the *unconditional*
complete-update comparison is
$$
\boxed{\begin{aligned}
\mathbb E\Delta\mathscr Q_{\rm marked}
=\sum_{m,\widetilde m}\Lambda(m,\widetilde m)
 \sum_{\mathbf p}Q_{m,\widetilde m}(\mathbf p)
 \bigg\{\mathscr P_C(\mathbf p)
 +\int\bigg[\frac1N\sum_i(\mathscr K_i+\mathscr C_i)
 +\frac{\eta_A}{N}\sum_i
 (\mathbf1_{a_i^+\ne\widetilde a_i^+}
  -\mathbf1_{a_i\ne\widetilde a_i})\bigg]\nu_{\mathbf p}(d\omega)\bigg\}.
\end{aligned}}                                                     \tag{SCK.F5}
$$
Here $\mathscr P_C$ is (SCK.7), $\mathscr K_i$ is the fully expanded
polynomial (SCK.2), $\mathscr C_i$ is the radial-cap increment in
(SCK.3), and the terminal indicators are evaluated *after* final
position diffusion. Every weight in (SCK.F5) is determined by
(SCK.F1)--(SCK.F4a), the stated categorical coupling, and the primitive
parameters above. The terms retain their signs; (SCK.F5) does not
replace the keystone contribution by a generic absolute-value error.

There is also an exact one-step population-survival accounting. After
the component and OU innovations, but before the independent final
position Gaussians, let $x_{2,i}$ be the actual A2 position and put
$s=\sigma_x\sqrt h$. Define
$$
\theta_i=\begin{cases}
\displaystyle\int_D(2\pi s^2)^{-d/2}
        e^{-|y-x_{2,i}|^2/(2s^2)}\,dy,&s>0,\\
\mathbf1_D(x_{2,i}),&s=0.
\end{cases}                                                        \tag{SCK.F6}
$$
Conditionally on these pre-final-noise data, the terminal marks are
independent Bernoulli variables with respective parameters $\theta_i$.
Consequently
$$
 \mathbb P(M^+=0\mid\text{pre-final-noise data})
       =\prod_{i=1}^N(1-\theta_i),\qquad
 \mathbb E(M^+\mid\text{pre-final-noise data})=\sum_i\theta_i.
                                                                    \tag{SCK.F7}
$$
The unconditional probabilities integrate (SCK.F7) with the same
measurement, plan, Haar, jitter and OU weights as (SCK.F4).
:::

:::{prf:proof}
For fixed input, the measurement draws are independent across alive
rows, giving (SCK.F1). Each live donor proposal has probability
$P_C^N(i,j)$ and its conditional gate succeeds with probability
$a(F_i^m,F_j^m)$. Summing the rejected donor events gives the
diagonal mass in (SCK.F2). The singleton and dead-row branches are
deterministic after their eligible donor draw. This proves row
normalization and (SCK.F3), including mandatory revival. Conditional
on a plan, the accepted undirected graph is deterministic. Its
nontrivial components receive independent Haar rotations, while all
other innovations have exactly the product Gaussian law specified in
the algorithm. The tower property yields (SCK.F4).

For the paired statement, the common-mass/residual formula (SCK.F4a)
has row marginals $r_i^S,\widetilde r_i$; its product therefore
preserves each swarm's entire measurement-vector law, including the
common normalizers computed **after** the draws within that swarm.
When the input states coincide, $t_i^D=0$ in every row and the
retained fitness vectors coincide. The subsequent common-source,
component-Haar, jitter and kinetic coupling is then also diagonal.
The source-row coupling has precisely the categorical marginals
(SCK.F2), and its independent product preserves the conditional law
of complete plans. Conditional on a paired plan, (SCK.7) is the exact
preparation difference after averaging jitter and Haar rotations.
The BAOAB difference is (SCK.2), the cap correction is (SCK.3), and
terminal marking contributes (SCK.8). Integrate those conditional
identities first over the innovations, then the paired plans and
measurement vectors. Absolute integrability permits the signed tower
calculation and proves (SCK.F5). Finally, before final position
diffusion, each position is $x_{2,i}+s\chi_i$ with independent
$\chi_i\sim\mathcal N(0,I_d)$. The terminal events are then
conditionally independent with probabilities (SCK.F6); multiplying
failure probabilities and summing success probabilities proves
(SCK.F7). No independence of rows is asserted after the shared
component rotations are integrated out.
:::

:::{prf:theorem} Regional parameter substitution in the complete marked update
:label: thm-slc-regional-parameter-one-step

Fix a nonextinct input and the declared basin, passage and exterior
partition $(A_a)$. The following are descriptors of the *given*
landscape, not extra restrictions on the update. Bound its raw reward
on each relevant region and capped velocity cylinder by
$r_a^-\le R(x,v)\le r_a^+$, allowing infinite endpoints. For every
eligible ordered pair of input rows choose
$0\le\ell_{ij}\le d_{\rm alg}(i,j)\le u_{ij}\le\infty$; using the
actual distance makes these inequalities equalities. For companion
role $b\in\{D,C\}$ and its actual eligible alive set $H_{b,i}$ put
$$
w_{b,ij}^-=e^{-u_{ij}^2/(2\epsilon_b^2)},\quad
w_{b,ij}^+=e^{-\ell_{ij}^2/(2\epsilon_b^2)},\quad
Z_{b,i}^\pm=\sum_{k\in H_{b,i}}w_{b,ik}^\pm.
                                                               \tag{SCK.G1}
$$
For $Z_{b,i}^->0$ the actual companion probability satisfies
$$
\frac{w_{b,ij}^-}{Z_{b,i}^+}\le P_b^N(i,j)\le
\min\{1,w_{b,ij}^+/Z_{b,i}^-\}.                         \tag{SCK.G2}
$$
If $Z_{b,i}^-=0$, use the exact positive denominator for the actual
finite input, with upper bound one. The alive singleton measurement
has probability one at raw distance zero.

For a fixed measurement vector $m$ define, for each alive row,
$$
y_{r,i}^\pm=r_{a_i}^\pm,\qquad
y_{s,i}^-=\sqrt{\ell_{i,m_i}^2+\delta_D^2},\quad
y_{s,i}^+=\sqrt{u_{i,m_i}^2+\delta_D^2};                 \tag{SCK.G3}
$$
the singleton has $y_{s,i}^-=y_{s,i}^+=\delta_D$.
For $b\in\{r,s\}$ and finite endpoints set
$$
\mu_b^\pm=M^{-1}\sum_{i\in\mathcal A}y_{b,i}^\pm,\quad
y_{b,*}=\min_i y_{b,i}^-,\quad y_b^*=\max_i y_{b,i}^+,\quad
S_b^-=\sigma_b,\quad
S_b^+=\sqrt{\sigma_b^2+(y_b^*-y_{b,*})^2/4},            \tag{SCK.G4}
$$
$$
z_{b,i}^-=\min_{\substack{\mu\in\{\mu_b^-,\mu_b^+\}\\
 S\in\{S_b^-,S_b^+\}}}\frac{y_{b,i}^- -\mu}{S},\qquad
z_{b,i}^+=\max_{\substack{\mu\in\{\mu_b^-,\mu_b^+\}\\
 S\in\{S_b^-,S_b^+\}}}\frac{y_{b,i}^+ -\mu}{S}.       \tag{SCK.G5}
$$
With $h_b(z)=(\eta_b+A_b/(1+e^{-z}))^{p_b}$, the *same empirical
normalizers used by the algorithm* give
$$
f_i^-:=h_r(z_{r,i}^-)h_s(z_{s,i}^-)
 \le F_i^m\le
f_i^+:=h_r(z_{r,i}^+)h_s(z_{s,i}^+).                    \tag{SCK.G6}
$$
Writing $s_c=p_{\max}$ and $\epsilon_c=\varepsilon_{\rm clone}$,
the live-row gate lies in $[g_{ij}^-,g_{ij}^+]$, where
$$
g_{ij}^-=\min\{1,[f_j^- -f_i^+]_+/[s_c(f_i^++\epsilon_c)]\},\quad
g_{ij}^+=\min\{1,[f_j^+ -f_i^-]_+/[s_c(f_i^-+\epsilon_c)]\}.
                                                               \tag{SCK.G7}
$$
For $j\ne i$ multiply the appropriate bounds (SCK.G2),(SCK.G7)
to bound the accepted source mass $q_i^m(j)$ of (SCK.F2). Its
rejection mass is one minus the sum of the *actual* accepted masses,
and is bounded by one minus the respective upper and lower sums,
clipped to $[0,1]$. A dead row instead uses (SCK.G2) with gate one.
Products of row bounds bound each accepted-plan mass (SCK.F3), and
products of measurement bounds bound (SCK.F1). Thus reward bands,
basin or passage geometry, feature radii and metric weight, both
companion widths, the diversity floor, both variance floors, logistic
parameters, acceptance parameters, alive counts and revival all enter
the *same* finite-plan expectation (SCK.F5). If a band is infinite,
the exact finite-input weights in (SCK.F1)--(SCK.F3) remain the
one-step characterization.

Condition on a paired accepted plan, jitter and component rotations.
The prepared force inputs $X_i,Y_i$ and random second-force inputs
$L_i,\widetilde L_i$ receive their actual region labels. If these
pairs lie respectively in $A_a\times A_b$ and $A_c\times A_e$ and
inside declared balls of radii $R_0,R_1$, then the profiles of
{prf:ref}`def-slcs-data` give
$$
|f_i|\le L^F_{ab}(R_0)|r_i|+J^F_{ab}(R_0),\qquad
|g_i|\le L^F_{ce}(R_1)|R_i|+J^F_{ce}(R_1).
                                                               \tag{SCK.G8}
$$
For a same-region pair, the first right-hand side may instead be
$\omega_{A_a}(|r_i|)\le L|r_i|+D_{A_a}(L,|r_i|)$ with its actual
scale-dependent excess; the identical substitution applies to the
second pair in $A_c$. The reward oscillation on $A_a$ is
$r_a^+-r_a^-$ when these endpoints are finite. Thus the regularity
and reward-oscillation profiles in {prf:ref}`def-slc-profiles`
appear explicitly in (SCK.G6)--(SCK.G8).
The signed terms of (SCK.2) retain their actual values. When useful,
$r_i\cdot f_i\le J_{A_a}(k,|r_i|)-k|r_i|^2$ for $a=b$;
the analogous bound for $R_i\cdot g_i$ applies when $c=e$.
The absolute-position ledger can use
$x\cdot F(x)\le b_{A_a}(k)-k|x|^2$. All exterior and excursion
labels remain in the OU Gaussian integral, as in
{prf:ref}`lem-slc-signed-kinetic-regions`. These formulas place
force moduli, interface jumps, restoring or confinement deficits,
noise, friction, timestep, jitter, collision restitution and cap in
the exact signed one-step balance without declaring any of their
defects zero.

Finally, conditional on the *prepared* population, set
$$
\mu_i=X_i+B V_i^C+\eta F(X_i),\qquad
\tau^2=c^2q^2+s^2,
                                                               \tag{SCK.G9}
$$
with $c,B,\eta,q,s$ in {prf:ref}`def-slc-parameter-register`.
For any Borel basin, passage, tail or death region $H$ put
$$
\pi_i(H)=\begin{cases}
\displaystyle\int_H(2\pi\tau^2)^{-d/2}
 e^{-|y-\mu_i|^2/(2\tau^2)}\,dy,&\tau>0,\\
\mathbf1_H(\mu_i),&\tau=0.
\end{cases}                                                       \tag{SCK.G10}
$$
For a finite Borel partition $(H_a)$ the conditional joint regional
count generating function is
$$
\mathbb E\left[\prod_a t_a^{Y_{H_a}^+}\mid S^C\right]
=\prod_{i=1}^N\left(\sum_a\pi_i(H_a)t_a\right).
                                                               \tag{SCK.G11}
$$
In a killed gas, use $H_a\cap D$ for alive categories and include
$D^c$ as a death category. In particular, first arrival in $H$ has
probability $1-\prod_i(1-\pi_i(H))$ and full death has probability
$\prod_i(1-\pi_i(D))$, both conditional on preparation. If
$H\subset B(z_H,r_H)$ has finite positive volume and
$|\mu_i-z_H|\le d_{iH}$, then
$$
\pi_i(H)\ge |H|(2\pi\tau^2)^{-d/2}
 e^{-(d_{iH}+r_H)^2/(2\tau^2)}\quad(\tau>0).           \tag{SCK.G12}
$$
For a prepared row with $X_i\in A_a$, an explicit admissible choice is
$$
d_{iH}=|X_i-z_H|+B|V_i^C|+\eta|F(X_i)|
\le \sup_{x\in A_a}|x-z_H|+BV_c+\eta M_{A_a}.         \tag{SCK.G13}
$$
The second bound uses capped input velocities and the collision
bound $V_c$ in {prf:ref}`def-slc-parameter-register`. If that
regional supremum is infinite, the exact first expression remains
finite for every prepared row with finite force. On a clone-jitter
cutoff $|\sigma_J\zeta_i|\le J$ and donor region $C$, one may use
$C^{[J]}$ and the displacement profile of
{prf:ref}`def-slcpn-data`; the complementary jitter event retains
its actual Gaussian probability.
These exact probabilities and the displayed lower bound insert
basin volume, bottleneck width, tail location, boundary geometry and
all noise scales into (SCK.F4); averaging over preparation uses its
*same* measurement, plan, jitter and Haar weights.
:::

:::{prf:proof}
The Gaussian companion weight decreases with distance. Bound each
weight by (SCK.G1), sum over the actual eligible labels and divide
to get (SCK.G2). The measured separation is increasing in distance,
giving (SCK.G3). The mean of values in their row intervals lies in
$[\mu_b^-,\mu_b^+]$. Popoviciu's inequality gives variance at most
$(y_b^*-y_{b,*})^2/4$, proving (SCK.G4). For a fixed raw endpoint,
$(y-\mu)/S$ is monotone in $\mu$ and, at fixed $\mu$, monotone in
$S$ with direction determined by the numerator's sign. Its extrema
over the containing rectangle are at the four corners, giving
(SCK.G5). The logistic powers are nondecreasing; the gate is
nondecreasing in donor fitness and nonincreasing in recipient fitness.
This proves (SCK.G6)--(SCK.G7) and the asserted row and plan bounds.

The inequalities (SCK.G8) are precisely the declared pair profiles
at the two actual force-evaluation pairs. Partitioning the Gaussian
integral by their regions includes every excursion; definitions of
$J_A$ and $b_A$ give the two signed optional estimates. BAOAB gives
$x_i^+=\mu_i+cq\xi_i+s\chi_i$. Conditional on preparation, these
Gaussian pairs are independent across rows, despite the component
dependence already fixed by conditioning. Their covariance is
$\tau^2I_d$, proving (SCK.G10) and the product (SCK.G11).
For $y\in H$, $|y-\mu_i|\le d_{iH}+r_H$; integrate the resulting
Gaussian-density lower bound to prove (SCK.G12). The triangle
inequality applied to (SCK.G9) gives the first bound in (SCK.G13);
the regional force supremum and component collision velocity bound
give the second. The law of total
expectation over the unchanged complete-plan distribution gives the
unconditional statement.
:::

:::{prf:corollary} Uniform-companion temperature limit of the one-step bounds
:label: cor-slc-uniform-companion-step

For $b\in\{D,C\}$ define the width-$\epsilon_b=\infty$ companion
law to be uniform on the *same eligible alive labels* as the canonical
finite-width law. This is its parameter limit, with every other stage
and parameter of {prf:ref}`alg-euclidean-gas` unchanged. Each width
may be sent to infinity separately. For $M\ge2$, an alive row's
eligible-set size is $M-1$; a dead clone recipient's is $M$.
The alive singleton retains its measurement and no-clone exceptions.

Let
$$
D_0^2=4\bigl[(R_x^{\rm feat})^2+
                \lambda_{\rm alg}(R_v^{\rm feat})^2\bigr],\qquad
\kappa_b(\epsilon_b)=
e^{-D_0^2/(2\epsilon_b^2)},\quad\kappa_b(\infty)=1.
                                                               \tag{SCK.U1}
$$
For every finite input and every eligible set of size $n$,
$$
\left\|P_b^{N,\epsilon_b}(i,\cdot)-
          {\rm Unif}(H_{b,i})\right\|_{\rm TV}
\le1-\kappa_b(\epsilon_b).                              \tag{SCK.U2}
$$
Consequently, for fixed $N$ the *complete marked one-step kernel*
obeys
$$
\left\|\Psi_N^{\epsilon_D,\epsilon_C}(S,\cdot)-
          \Psi_N^{\infty,\infty}(S,\cdot)\right\|_{\rm TV}
\le\min\{1,M[1-\kappa_D(\epsilon_D)]
            +N[1-\kappa_C(\epsilon_C)]\}.               \tag{SCK.U3}
$$
If one width is unchanged in both kernels, omit its term. These
limits hold for the complete boundary-marked update, not merely for
the companion draws.

At infinite clone width, conditional on retained fitness,
$$
q_i(j)=\frac{a(F_i,F_j)}{M-1}\quad
 (a_i=1,\ j\in\mathcal A\setminus\{i\}),\qquad
q_i(j)=\frac1M\quad(a_i=0,\ j\in\mathcal A).           \tag{SCK.U4}
$$
The live rejection probability is
$1-(M-1)^{-1}\sum_{j\in\mathcal A\setminus\{i\}}a(F_i,F_j)$.
In (SCK.G1)--(SCK.G2), $w_{C,ij}^-=w_{C,ij}^+=1$ and
$Z_{C,i}^-=Z_{C,i}^+=|H_{C,i}|$, so the regional companion
bounds become equalities. Every previously derived bound whose
bandwidth enters only through $\kappa_C$ specializes by setting
$\kappa_C=1$; geometric reward, force, collision, jitter, kinetic
and boundary terms retain their displayed values. At infinite
measurement width the analogous equalities hold for $D$, while
the sampled diversity distances and their fitness normalizers remain
random under uniform measurement labels. Thus (SCK.F4)--(SCK.F5)
and (SCK.G3)--(SCK.G13) specialize without deleting any fitness or
post-clone term.
:::

:::{prf:proof}
Squashed positions and velocities lie in balls of radii
$R_x^{\rm feat}$ and $R_v^{\rm feat}$. Their weighted squared
distance is at most $D_0^2$, so every Gaussian weight on an eligible
set lies in $[\kappa_b,1]$. Its normalized probability at each
eligible label is at least $\kappa_b/n$. The overlap with the
uniform distribution is therefore at least $\kappa_b$, proving
(SCK.U2). Couple each of the $M$ measurement draws to its uniform
limit by maximal coupling. If all agree, their raw arrays, common
normalizers and fitness vectors agree. Then couple the $N$ clone
donor draws likewise and reuse the gate uniforms, component Haar
rotations, jitters and kinetic Gaussians. Whenever all companion
labels match, the *entire* marked outputs match, including revival
and terminal status. The union bound and coupling inequality give
(SCK.U3). Letting the widths tend to infinity proves convergence
for fixed $N$. Uniform eligible donors and the unchanged acceptance
gate give (SCK.U4); substituting unit weights in (SCK.G1) gives the
other asserted specializations.
:::

:::{prf:proposition} Reward bands in the signed complete-update cloning term
:label: prop-slc-reward-to-signed-update

Work in the all-alive setting of {prf:ref}`thm-slc-signed-complete-update`.
Condition on the actual measurement marks, so that the reward and sampled
diversity arrays and their shared normalizers are frozen. For every declared
label cluster $H$, supply raw bands
$r_H^-\le r_i\le r_H^+$ and $s_H^-\le s_i\le s_H^+$ for $i\in H$.
These may be the attained extrema of the arrays; regional landscape and
measurement profiles may give wider bands. Let

$$
 m_b=N^{-1}\sum_i b_i,\qquad
 S_b=\left[N^{-1}\sum_i(b_i-m_b)^2+\sigma_b^2\right]^{1/2},
 \qquad b\in\{r,s\},
$$

and use the *same* $m_b,S_b$ for all clusters in this realization. Set

$$
 f_H^\pm=
 \prod_{b\in\{r,s\}}
 \left[\frac{A_b}{1+\exp(-(b_H^\pm-m_b)/S_b)}+\eta_b\right]^{p_b},
 \qquad \Omega_H=f_H^+-f_H^-.
 \tag{SCK.R1}
$$

An inactive exponent contributes a factor one. Thus every actual retained
fitness in $H$ lies in $[f_H^-,f_H^+]$. For two nonempty clusters $H,L$ put

$$
 \Delta_{HL}^-=f_L^--f_H^+,\qquad
 T_{HL}^2=(\Omega_H^2+\Omega_L^2)/4,
$$

and, using the primitive fitness range $F_*,F^*$ and companion lower
weight $\kappa_C$ of {prf:ref}`thm-cloning-signed-cluster-fitness-flux`, put

$$
 A_*=\max\{F^*-F_*,s_c(F^*+\epsilon_c)\},\quad
 c_+=\kappa_C/A_*,\quad
 c_-=[\kappa_Cs_c(F_*+\epsilon_c)]^{-1}.
$$

When $A_*>0$ and $N\ge2$, the accepted edge masses of this actual
cloning step satisfy the explicit reward-parameterized inequality

$$
\boxed{\quad
 B_{HL}-B_{LH}\ge \mathcal L^{R,s}_{HL}:=
 \frac{|H||L|}{N(N-1)}
 \left[\frac{c_++c_-}{2}\Delta_{HL}^-
 -\frac{c_--c_+}{2}
 \sqrt{(\Delta_{HL}^-)^2+T_{HL}^2}\right].\quad}
 \tag{SCK.R2}
$$

Use (SCK.R1)--(SCK.R2) separately in the two swarms of (C.S3), with
their respective measured normalizers. In that inequality one may set
$\mathcal L^1_{HL}=\mathcal L^{R,s,1}_{HL}$ and
$\mathcal L^2_{HL}=\mathcal L^{R,s,2}_{HL}$. Its $E_{HL}$ is controlled
by the already proved (C.S4) and
{prf:ref}`lem-slc-common-source-normalizers`, so this substitution carries
raw reward differences through fitness normalization, the acceptance gate,
the common donor flux and the cloning term of (SCK.3). More explicitly,
write

$$
 \mathscr J_C=\frac1N\sum_{i,j,k}\gamma_{i,jk}
       (|x_j-y_k-\bar d|^2-e_i)
 +\frac{d\sigma_J^2}{N}\sum_{i,j,k}\gamma_{i,jk}
       (\mathbf1_{j\ne i}-\mathbf1_{k\ne i})^2,
 \qquad
 \mathscr K_C=\frac1N\sum_i
       \mathbb E[\mathscr K(r_i,z_i,f_i,g_i)+\mathscr C_i].
$$

Cancellation of the centered barycenter terms against
$\mathscr B_C$ in (SCK.3) gives the exact alternative form

$$
 \mathbb E\Delta\mathscr Q
 =\alpha\left[
   \frac1N\sum_{i,j}c_{ij}(e_j-e_i)
   +\mathscr J_C+2\bar d\cdot\bar h\right]
   +\mathscr T_C+\mathscr K_C.
$$

For clusters oriented by $e_H\ge e_L$, the reward input (SCK.R2)
therefore gives the full-update bound

$$
\boxed{\begin{aligned}
 \mathbb E\Delta\mathscr Q\le{}&
 -\alpha\sum_{\{H,L\}}(e_H-e_L)
 \frac{\mathcal L^{R,s,1}_{HL}+\mathcal L^{R,s,2}_{HL}-E_{HL}}2
 +\alpha\sum_{H,L}U_{HL}(\rho_H+\rho_L)\\
 &+\alpha(\mathscr J_C+2\bar d\cdot\bar h)
 +\mathscr T_C+\mathscr K_C.
\end{aligned}}\tag{SCK.R3}
$$

This evaluates the reward contribution within the existing signed
cloning calculation. It is an alternative expansion of its Keystone
pressure and donor transfer, so that pressure is counted once. The within-cluster
error bands $\rho_H$ and the actual barycenter and collision terms retain
their existing values; no reward factor is attached to the independent
kinetic force $F=-\nabla U$ unless the configured reward is itself tied to
$U$.

If fixed, realization-independent bands are needed, let $b_*,b^*$ bound
the complete raw array for $b=r,s$ and put
$S_b^*=\sqrt{(b^*-b_*)^2/4+\sigma_b^2}$. In (SCK.R1), replace each
standardized lower endpoint by the minimum of
$(b_H^--m)/S$ over the four corners
$(m,S)\in\{b_*,b^*\}\times\{\sigma_b,S_b^*\}$, and each upper endpoint
by the corresponding maximum using $b_H^+$. Monotonicity of the logistic
map then gives deterministic $f_H^\pm$ with the same (SCK.R2). Infinite
bands simply leave that regional certificate uninformative; the
realization-conditional formula remains defined whenever the raw arrays
and their regularized normalizers are defined. Average the resulting
conditional bound over measurement marks only after evaluating its gate.
:::

:::{prf:proof}
The configured logistic-power map is increasing in each standardized
channel because $A_b>0$ and $p_b\ge0$. Its shared normalizers therefore
give (SCK.R1) simultaneously for every row in a cluster. Write the
source theorem's mean fitness gap as
$\Delta=\bar F_L-\bar F_H$; then
$\Delta\ge\Delta_{HL}^-$. Popoviciu's elementary variance inequality,
obtained by averaging $(F-f_H^-)(f_H^+-F)\ge0$, gives each cluster
variance at most $\Omega_H^2/4$ or $\Omega_L^2/4$. Hence the sum
$s^2$ in (3.S1) is at most $T_{HL}^2$.

The bracket in (3.S1) can be written as

$$
 G(\Delta,t)=\frac{c_++c_-}{2}\Delta
             -\frac{c_--c_+}{2}\sqrt{\Delta^2+t}.
$$

For $t\ge0$, its derivative in $\Delta$ is at least $c_+>0$,
while it decreases in $t$. Substituting the lower gap and upper variance
into (3.S1), with $k=N$, proves (SCK.R2), including negative reward
advantages. Formula (C.S3) is valid for any lower bound on each actual
signed edge flux. To obtain (SCK.R3), expand $\mathscr R_C$ in (SCK.3),
use $c_i=(p_i+\widetilde p_i-\ell_i)/2$, and cancel
$-|\bar h|^2-N^{-2}\sum_iV_i$ against the corresponding positive
terms in $\mathscr B_C$. What remains is the exact signed common flux,
$\mathscr J_C$, and $2\bar d\cdot\bar h$. Apply (C.S3) and then the two
copies of (SCK.R2). No reverse edge or actual acceptance probability
has been removed, and all measurement marks remain conditioned upon.

For the deterministic option, the array mean lies in $[b_*,b^*]$.
The same variance inequality gives
$\sigma_b\le S_b\le S_b^*$. For a fixed raw endpoint, the ratio
$(b_H^\pm-m)/S$ has its extrema over this rectangle at corners:
it is monotone in $m$, and at fixed $m$ it is monotone in $S$ with a
direction determined by the numerator's sign. Taking those four
corners therefore preserves the lower and upper standardized bounds.
:::

:::{prf:proposition} Complete finite-plan evaluation of the one-step reward dependence
:label: prop-slc-reward-full-plan

Under the preceding all-alive hypotheses, the reward dependence of
$\mathscr J_C$, $\bar h$, $\mathscr T_C$, and $\mathscr K_C$ in
(SCK.R3) can be evaluated from the same accepted row masses as its signed
flux. Condition on the complete two-swarm retained fitness vectors.
Let $\mathcal P$ consist of all paired source plans
$\mathbf p=((j_i,k_i))_{i=1}^N$ with $j_i,k_i\in\{1,\ldots,N\}$, and
give such a plan its actual coupling probability

$$
 Q(\mathbf p)=\prod_{i=1}^N\Pi_i(j_i,k_i),\qquad
 \Pi_i(j,k)=\lambda_{ij}\mathbf1_{j=k}+\gamma_{i,jk}.
 \tag{SCK.P1}
$$

For each $\mathbf p$, draw the accepted graph in swarm one from edges
$i\to j_i$ with $j_i\ne i$, and in swarm two from $i\to k_i$ with
$k_i\ne i$. Let $m_i,\widetilde m_i,b_i,\widetilde b_i$ be their
frozen-slot-velocity component quantities in (SCK.4), and define

$$
 H_i(\mathbf p)=|m_i-\widetilde m_i|^2+
 \alpha_{\rm col}^2\bigl[|b_i|^2+|\widetilde b_i|^2
 -2\mathbf1_{C_i=\widetilde C_i}b_i\cdot\widetilde b_i\bigr].
$$

Then the collision contribution remaining in (SCK.R3) is the finite sum

$$
\boxed{\quad
\mathscr T_C=
 \sum_{\mathbf p\in\mathcal P}Q(\mathbf p)\frac1N\sum_i
 \left\{2\beta\bigl[(x_{j_i}-y_{k_i})\cdot
       (m_i-\widetilde m_i)-d_i\cdot u_i\bigr]
 +\gamma_P[H_i(\mathbf p)-|u_i|^2]\right\}.\quad}
 \tag{SCK.P2}
$$

The other cloning quantities are already finite sums of the same row
masses: $\mathscr J_C$ is displayed in (SCK.R3), and

$$
\bar h=\frac1N\sum_{i,j,k}\Pi_i(j,k)
                    (x_j-y_k-d_i).
 \tag{SCK.P3}
$$

For completeness, $\mathscr K_C$ also has a specified finite-sum
Gaussian integral. For each $\mathbf p$, let $\mathcal H_{\mathbf p}$
be the product Haar law of its component rotations, sharing one draw
precisely for components with identical vertex sets in the two swarms.
Let $G_i^J,\xi_i$ be independent standard $d$-dimensional Gaussians,
shared between the paired rows for cloning jitter and OU noise,
respectively. In the integrand construct the prepared states using the
sources $j_i,k_i$, their acceptance indicators, their jitters and the
actual collision components; then construct $r_i,z_i,f_i,g_i$ and
$\mathscr C_i$ exactly as in (SCK.2)--(SCK.4). With
$d\gamma_{2Nd}$ the joint standard Gaussian law of all $G_i^J,\xi_i$,

$$
\boxed{\quad
\mathscr K_C=
 \sum_{\mathbf p\in\mathcal P}Q(\mathbf p)
 \int\frac1N\sum_i
       [\mathscr K(r_i,z_i,f_i,g_i)+\mathscr C_i]
       \,d\gamma_{2Nd}\,d\mathcal H_{\mathbf p}.\quad}
 \tag{SCK.P4}
$$

Equations (SCK.P1)--(SCK.P4), (C.S4), and (SCK.R1) specify every
reward-dependent factor of the all-alive one-step expression. For fixed
independent kinetic potential $U$, the raw reward changes the plan
weights $Q(\mathbf p)$; the force integrand is then evaluated on the
prepared states selected by that plan. These are finite sums and
specified probability integrals, not an additional mixing coefficient.
Their direct evaluation may cost exponentially many plans; the regional
bounds (SCK.R2)--(SCK.R3) aggregate those same terms when a structural
certificate is desired. All expectations require the integrability
already stated in {prf:ref}`thm-slc-signed-complete-update`.
:::

:::{prf:proof}
Conditional on retained fitness, each row's accepted source pair has
law $\Pi_i$, and the row draws are independent. Thus the joint plan has
probability (SCK.P1). Conditional on the plan, the copied positions
before jitter are $x_{j_i},y_{k_i}$. The shared jitter has zero mean and
is independent of the collision rotations. The conditional Haar first
and second moments are precisely (SCK.4), giving (SCK.P2). Averaging
the row source differences gives (SCK.P3). Conditional on the same plan,
the remaining independent innovations have exactly the Gaussian and
component-Haar laws stated in (SCK.P4). Substitution into the signed
kinetic polynomial and cap term, then the law of total expectation,
proves that formula. The final position Gaussian cancels from paired
physical differences and hence needs no integration in (SCK.P4);
its terminal classification is separately retained in (SCK.8).
:::

:::{prf:corollary} Regional Keystone balance converted to tagged-position TV
:label: cor-slc-reward-keystone-tv

In the all-alive setting of (SCK.R3), let $G_{HL}(S,\widetilde S;m)$
denote **the entire right-hand side** of (SCK.R3), conditional on the
two measured fitness vectors $m$. Evaluate its signed reward flux with
(SCK.R1)--(SCK.R2), its mismatch flux with (C.S4), and its remaining
terms by the finite plans (SCK.P1)--(SCK.P4). Thus $G_{HL}$ is a finite
expression in the entering positions and velocities, regional reward
bands, the configured cloning parameters, collision law, force, cap,
and noise. Let $\overline G_{HL}$ be its expectation over the actual
measurement marks. If the positional Gaussian variance
$\tau^2=c^2q^2+s^2$ is positive, the same complete update satisfies
$$
\left\|\mathsf T_k(S^+)-\mathsf T_k(\widetilde S^+)\right\|_{\rm TV}
\le \min\left\{1,
\sqrt{\frac{k}{2\pi\tau^2\lambda_-}}
\left[\mathscr Q(S,\widetilde S)+\overline G_{HL}(S,\widetilde S)
\right]_+^{1/2}\right\}.                         \tag{SCK.RTV}
$$
For fixed $k$, the conversion coefficient is independent of $N$.
The flux and finite-plan terms retain their actual $N$ dependence;
this statement does not turn an unevaluated or positive remainder
into a time-decaying rate.
:::

:::{prf:proof}
Condition on the measurement marks. Equation (SCK.R3) bounds the
conditional expectation of the full signed quadratic increment by
$G_{HL}$. Average over those marks, so
$\mathbb E\mathscr Q(S^+,\widetilde S^+)
\le\mathscr Q(S,\widetilde S)+\overline G_{HL}$.
The latter right-hand side is nonnegative whenever the hypotheses
hold, since it bounds a nonnegative expectation. Apply (SCK.TV1) of
{prf:ref}`thm-slc-keystone-tagged-tv` to this same coupled complete
update. Its Jensen step gives (SCK.RTV). The finite-plan formulas
use the actual accepted graph and conditional Gaussian kinetic law,
so no independent-walker or product-minorization step enters.
:::

:::{prf:definition} Phase-centered coupling errors for the actual preparation and copying laws
:label: def-slpke-errors

Use the conservative all-alive canonical kernel. Let $\pi$ be a proved fixed law of the actual population map
$\mathcal F_h$, and let $\mu$ be an admissible input. Keep all configured
cloning, collision and kinetic parameters. Require finite fourth positional
moments for both input laws and finite prepared force second moments;
the stated linear force-growth bound suffices for the latter. Higher
moments are required only when explicitly used below. Put
$\rho=\mathcal J\mu$, $\rho_*=\mathcal J\pi$, and choose any specified
coupling $((X,V),(Y,U))$ of these *actual preparation laws*. Write
$$
 \widetilde X=X-\mathbb EX,\quad\widetilde Y=Y-\mathbb EY,
 \quad\widetilde V=V-\mathbb EV,\quad\widetilde U=U-\mathbb EU,
$$
$$
 e_x=\|\widetilde X-\widetilde Y\|_2,\quad
 e_v=\|\widetilde V-\widetilde U\|_2,\quad
 e_f=\|F(X)-\rho F-[F(Y)-\rho_*F]\|_2,
$$
$$
 s_x=\|\widetilde Y\|_2,\quad
 s_v=\|\widetilde U\|_2,\quad
 s_f=\|F(Y)-\rho_*F\|_2.
$$
The $L^2$ norms refer to this single coupling probability space. All
centerings belong to their own marginal laws; neither common mean nor
independent rows are assumed.

Separately couple the actual single-root frozen copying experiments at
$\mu$ and $\pi$, including their sampled measurement fitness, cloning
companion, and acceptance coin. Denote their recipient positions by
$x,y$, proposed donor positions by $z,w$, and acceptance indicators by
$A,A_*\in\{0,1\}$. Put $m=\mu x$, $m_*=\pi x$ and
$$
 R=x-m,\quad R_*=y-m_*,\quad D=z-m,\quad D_*=w-m_*,
$$
$$
 e_R=\|R-R_*\|_2,\quad e_D=\|D-D_*\|_2,\quad
 a_\Delta=\|A-A_*\|_2=\Pr(A\ne A_*)^{1/2}.
$$
Let $s_R=\|R_*\|_2$, $s_D=\|D_*\|_2$,
$t_R=\|R_*\|_4$, $t_D=\|D_*\|_4$.
The copying and preparation couplings may be chosen separately because
they bound separate scalar differences. At $\mu=\pi$ identical
experiments are permissible and all their error quantities vanish.
:::

:::{prf:lemma} Centered kinetic covariance difference with no constant forcing term
:label: lem-slpke-kinetic-difference

With the preceding notation, define
$$
\begin{aligned}
 Q_K={}&2B(s_ve_x+s_xe_v+e_xe_v)
       +2\eta(s_fe_x+s_xe_f+e_xe_f)\\
 &+B^2(2s_ve_v+e_v^2)
       +2B\eta(s_fe_v+s_ve_f+e_ve_f)
       +\eta^2(2s_fe_f+e_f^2).
\end{aligned}
$$
Then the exact kinetic covariance polynomial of
{prf:ref}`thm-slkd-full-position` satisfies
$$
 |\mathcal K(\rho)-\mathcal K(\rho_*)|\le Q_K.
$$
Every term on the right vanishes when the coupled centered errors
vanish. In particular there is no fresh Gaussian noise offset.
:::

:::{prf:proof}
For any coupled centered pairs $(A,B),(A_*,B_*)$, put
$\Delta A=A-A_*$, $\Delta B=B-B_*$. The exact algebra
$$
 \mathbb E(A\cdot B-A_*\cdot B_*)
 =\mathbb E[\Delta A\cdot B_*+A_*\cdot\Delta B
                                  +\Delta A\cdot\Delta B]
$$
and Cauchy–Schwarz give
$|\Delta\operatorname{Cov}(A,B)|\le
\|B_*\|_2\|\Delta A\|_2+\|A_*\|_2\|\Delta B\|_2
+\|\Delta A\|_2\|\Delta B\|_2$.
Taking $A=B$ gives
$|\Delta\operatorname{Var}(A)|\le2\|A_*\|_2\|\Delta A\|_2
+\|\Delta A\|_2^2$.
Apply these two inequalities term by term to the five terms of
$\mathcal K$. Both full kinetic outputs have the same added variance
$d(c^2q^2+s^2)$, which cancels exactly on subtraction.
:::

:::{prf:lemma} Regional force increments and Gaussian-excursion moment budgets
:label: lem-slpke-force-profiles

For the declared spatial partition $(A_a)_a$, let
$$
 \omega_{ab}(r)=\sup\{|F(x)-F(y)|:x\in A_a,y\in A_b,|x-y|\le r\},
$$
with an empty supremum zero and infinite values allowed. These profiles
include cross-region jumps. Then
$$
 e_f^2\le\mathbb E|F(X)-F(Y)|^2
 \le\sum_{a,b}\mathbb E[
 \mathbf1_{\{X\in A_a,Y\in A_b\}}\omega_{ab}(|X-Y|)^2].
$$
For an explicit truncated version suppose $|F(x)|\le G_0+G_1|x|$
and choose $R>0$, $r>0$, $q>2$. Let
$M_X\ge\mathbb E|X|^q$, $M_Y\ge\mathbb E|Y|^q$,
$P_R=\min\{1,(M_X+M_Y)/R^q\}$ and
$D_x=\mathbb E|X-Y|^2$. Define
$$
 \omega_R(r)=\sup_{|x|,|y|\le R,\ |x-y|\le r}|F(x)-F(y)|.
$$
Then
$$
\begin{aligned}
 e_f^2\le{}&\omega_R(r)^2
       +4(G_0+G_1R)^2\min\{1,D_x/r^2\}\\
 &+12G_0^2P_R
       +3G_1^2(M_X^{2/q}+M_Y^{2/q})P_R^{1-2/q}.
\end{aligned}
$$
A prepared $q$th-moment budget, retaining Gaussian jitter excursions,
is explicitly
$$
 M_X=2^{q-1}\left[(1+2/\kappa_C)\mu|x|^q
                                    +\sigma_J^qm_{d,q}\right],
 \quad m_{d,q}=2^{q/2}\Gamma((d+q)/2)/\Gamma(d/2),
$$
and analogously for $M_Y$ with $\pi$. Also $e_x\le D_x^{1/2}$ and
$e_v\le(\mathbb E|V-U|^2)^{1/2}$. Explicit reference coefficients are
$$
 s_x\le(\rho_*|x|^2)^{1/2},\quad s_v\le V_c,\quad
 s_f\le[2G_0^2+2G_1^2\rho_*|x|^2]^{1/2}.
$$
Under a global force-Lipschitz bound, the sharper direct estimate is
$e_f\le L_FD_x^{1/2}$.
:::

:::{prf:proof}
Centering a square-integrable random variable only decreases its
$L^2$ norm, which gives the first inequalities and the analogous
bounds for $e_x,e_v$. Apply the pairwise profiles pointwise.
For the truncated estimate split according to: both positions are in
$B_R$ and within distance $r$; both are in $B_R$ but farther apart;
at least one is outside. The first contribution is bounded by
$\omega_R(r)^2$. On the second event the force difference is at most
$2(G_0+G_1R)$ and its probability at most $\min(1,D_x/r^2)$.
On the exterior event its square is at most
$12G_0^2+3G_1^2(|X|^2+|Y|^2)$. Markov bounds the event probability
by $P_R$ and Hölder bounds each weighted second moment on it by
$M^{2/q}P_R^{1-2/q}$. The preparation moment formula follows from
the actual copy-multiplicity bound and
$|y+J\sigma_JZ|^q\le2^{q-1}(|y|^q+\sigma_J^q|Z|^q)$.
Collision changes velocities, not the copied positions. The remaining
reference bounds follow from variance being bounded by second moment,
the collision cap, and the stated force growth.
The exact profile integral is zero for identical coupled positions;
a fixed finite truncation may retain a conservative excursion error,
which is explicitly shown rather than treated as phase dissipation.
:::

:::{prf:theorem} Phase-subtracted full-update scalar variance estimate
:label: thm-slpke-full-variance

Keep the actual recipient activity $A_{\rm rec}$, signed donor excess
$\Gamma_\theta=D_{\rm donor}-(1-\theta)A_{\rm rec}$,
acceptance fraction $p_{\rm cl}$ and mean displacement $t$ of Section14,
with fixed $0<\theta\le1$. Define
$$
 e_W(\mu)=W(\mu)-W(\pi),\quad
 \Delta A=A_{\rm rec}(\mu)-A_{\rm rec}(\pi),
$$
$$
 Q_\Gamma=2s_De_D+e_D^2
       +(1-\theta)(2s_Re_R+e_R^2)
       +a_\Delta[t_D^2+(1-\theta)t_R^2],
$$
$$
 T_\Delta=e_D+e_R+a_\Delta(s_D+s_R),\quad
 Q_C=Q_\Gamma+d\sigma_J^2a_\Delta^2
                              +2|t(\pi)|T_\Delta+T_\Delta^2.
$$
Then the unchanged full update satisfies
$$
 \boxed{\quad
 \left|e_W(\mathcal F_h\mu)-e_W(\mu)+\theta\Delta A\right|
 \le Q_C+Q_K.\quad}
$$
All residuals vanish at the phase under identical copying/preparation
couplings; the noise terms have been subtracted, not discarded.
For the copying reference moments, primitive bounds are
$$
 s_R\le(\pi|x|^2)^{1/2},\quad
 s_D\le(1+\kappa_C^{-1/2})(\pi|x|^2)^{1/2},
$$
$$
 t_R\le2(\pi|x|^4)^{1/4},\quad
 t_D\le(1+\kappa_C^{-1/4})(\pi|x|^4)^{1/4},\quad
 |t(\pi)|\le s_D+s_R.
$$
No contraction of the full population law is asserted by this scalar
observable estimate.
:::

:::{prf:proof}
For coupled copying experiments,
$\Gamma_\theta(\mu)=\mathbb E A[|D|^2-(1-\theta)|R|^2]$.
For the donor term subtract and add $A|D_*|^2$:
$$
 |\mathbb E A|D|^2-\mathbb E A_*|D_*|^2|
 \le2s_De_D+e_D^2+a_\Delta t_D^2.
$$
The first two terms use $A\le1$ and Cauchy–Schwarz; the last uses
Cauchy–Schwarz on $|A-A_*||D_*|^2$. The recipient term gives the
same formula with $R$. This proves $|\Delta\Gamma_\theta|\le Q_\Gamma$.
Also $|\Delta p_{\rm cl}|\le\Pr(A\ne A_*)=a_\Delta^2$.
Since $z-x=D-R$, subtracting the two mean displacements gives
$$
 |t(\mu)-t(\pi)|
 \le e_D+e_R+a_\Delta\|D_*-R_*\|_2\le T_\Delta.
$$
Thus $||t(\mu)|^2-|t(\pi)|^2|\le2|t(\pi)|T_\Delta+T_\Delta^2$.
The proposed donor's marginal density relative to the population is
bounded by $1/\kappa_C$, proving the stated reference donor moments
by Minkowski; the centered recipient bounds are immediate.

Subtract the exact full-step variance balance at $\pi$ from the one
at $\mu$. Since $\mathcal F_h\pi=\pi$, the phase balance is zero.
Use the exact identity $-A_{\rm rec}+D_{\rm donor}
=-\theta A_{\rm rec}+\Gamma_\theta$. The remaining difference is
$$
 \Delta\Gamma_\theta+d\sigma_J^2\Delta p_{\rm cl}
       -( |t(\mu)|^2-|t(\pi)|^2 )
       +\mathcal K(\rho)-\mathcal K(\rho_*).
$$
The additive $d\tau^2$ cancels exactly. Apply the proved bounds to
obtain the boxed inequality.
:::

:::{prf:remark} The precise use of the regional Keystone lower bound
:label: rem-slpke-keystone-subtraction

If both inputs lie in a common certified region with
$A_{\rm rec}(\nu)\ge2k_{\rm key}W(\nu)^p$, define the explicit
nonnegative slack
$R_A(\nu)=A_{\rm rec}(\nu)-2k_{\rm key}W(\nu)^p$.
Then the activity difference in the preceding theorem is exactly
$$
 \Delta A=2k_{\rm key}[W(\mu)^p-W(\pi)^p]
                              +R_A(\mu)-R_A(\pi).
$$
Keeping this signed slack yields a phase-centered power term and a
phase-centered residual, both zero at equality. Subtracting two lower
bounds does not justify omitting $R_A(\mu)-R_A(\pi)$.
The present result quantitatively evaluates the quadratic observable;
a separate sign estimate connecting its retained residuals to a full-law
phase distance is not supplied by a variance identity alone.
:::

:::{prf:remark} Transfer into the existing hypocoercive proof
:label: rem-slkd-hypocoercive-transfer

The new bound supplies a signed, quantitatively evaluated candidate for
$H_x$ in {prf:ref}`thm-complete-variance-drift` and
{prf:ref}`thm-complete-cloning-drift`. Those identities additionally
retain the actual collision-energy dissipation, revival terms where
applicable, location error and the declared inter-swarm observable.
Their subsequent kinetic comparison must use the same full kernel,
metric and normalization. In the present all-alive setting there is
no revival term; the genuine collision and kinetic terms above remain.
The complete comparison estimate is established only where these
computed coefficients and defects satisfy its numerical dominance
conditions. No missing sign is repaired by increasing $N$: the donor
flux, population center shift and kinetic noise all remain at order one.

In particular, a bound on one population's positional variance does not
identify its complete stationary law. Distinct probability laws can
have the same position mean and variance. The phase-local full-law
residual certificate and kinetic smoothing results distinguish those
laws when their hypotheses close. This does not invalidate the
population Keystone inequality or impose synchronization of distinct
population phases; it states exactly which observable that inequality
controls.
:::

:::{prf:lemma} Population-independent polynomial relaxation with every forcing term retained
:label: lem-slcn-power-recursion

Let $0<V_{\max}^{\rm err}<\infty$ and let
$0\le V_N\le V_{\max}^{\rm err}$ be a declared full-update
error observable, and suppose its actual signed drift has been bounded by

$$
P_NV_N-V_N\le-\sigma k_{\rm key}V_N^p
                  +\frac{\sigma E_{\max}}{N^2}+D_N,
\qquad \sigma>0.
$$

The pressure coefficient and power are those explicitly computed in
{prf:ref}`thm-slcn-keystone-power`. The multiplier $\sigma$ and the
signed-to-positive defect $D_N\ge0$ must come from the actual observable's
cloning and kinetic balance; pressure alone does not assert this drift.
Suppose $\sup_n\mathbb E D_N(S_n)\le d_N$. Set

$$
a=\min\left\{\sigma k_{\rm key},
          \frac1{p(V_{\max}^{\rm err})^{p-1}}\right\},\qquad
b_N=\frac{\sigma E_{\max}}{N^2}+d_N,\qquad
R_N=(b_N/a)^{1/p}.
$$

With $m_0=\mathbb E V_N(S_0)$, for every $n\ge1$,

$$
\mathbb E V_N(S_n)\le R_N+
\left([(m_0-R_N)_+]^{-(p-1)}+(p-1)an\right)^{-1/(p-1)},
$$

where the last term is zero if $m_0\le R_N$. In particular it is at
most $R_N+[(p-1)an]^{-1/(p-1)}$. If $R_N>V_{\max}^{\rm err}$,
the trivial bound $V_{\max}^{\rm err}$ applies instead. The polynomial
time rate and its coefficient are independent of $N$.

If an actual profile calculation gives
$d_N\le d_\infty+\sum_{j=1}^J C_jN^{-\zeta_j}$,
$C_j\ge0$, $\zeta_j>0$, then the complete floor is bounded by

$$
R_N\le(d_\infty/a)^{1/p}
 + (\sigma E_{\max}/a)^{1/p}N^{-2/p}
 +\sum_j(C_j/a)^{1/p}N^{-\zeta_j/p}.
$$

Thus every finite-particle contribution vanishes, while a nonzero
population defect is retained. In a zero-population-defect regime
$d_\infty=0$, every joint sequence $N,n\to\infty$ makes this error
vanish. Physical time is $nh$. An expectation bound $\alpha\epsilon$
gives error at most $\epsilon$ with probability at least $1-\alpha$
by Markov's inequality.
:::

:::{prf:proof}
Take expectations and use Jensen's inequality for $x^p$ to obtain
$m_{n+1}\le m_n-a m_n^p+b_N$. On
$[0,V_{\max}^{\rm err}]$, the map $g(x)=x-a x^p+b_N$ is
nondecreasing by the chosen upper bound on $a$. If
$R_N\le V_{\max}^{\rm err}$, it satisfies $g(R_N)=R_N$.
Writing $y_n=(m_n-R_N)_+$, monotonicity below $R_N$ and
$(R_N+y)^p\ge R_N^p+y^p$ above it give
$y_{n+1}\le y_n-a y_n^p$. This last right-hand side is nonnegative
on the stated range. Whenever both terms are positive,

$$
y_{n+1}^{-(p-1)}
\ge y_n^{-(p-1)}(1-a y_n^{p-1})^{-(p-1)}
\ge y_n^{-(p-1)}+(p-1)a.
$$

The final inequality follows from convexity of $(1-x)^{-(p-1)}$ on
$[0,1)$. If a term becomes zero, all later excesses remain zero.
Iteration proves the claimed rate in every case. The floor expansion
uses $(\sum_jx_j)^{1/p}\le\sum_jx_j^{1/p}$ for nonnegative terms.
The limiting and probability conclusions follow from the displayed
inequalities, with no exchange of an uncontrolled limit.
:::

(sec-slqc-quadratic-consistency)=
### 14.3. Complete-update consistency for signed quadratic observables

:::{prf:theorem} Explicit quadratic consistency from the actual rooted-forest bound
:label: thm-slqc-quadratic-consistency

Fix an admissible all-alive entering configuration $S$, its empirical
law $\mu=L_N(S)$, the actual full-update empirical output $\widehat\nu$
and the canonical population output $\nu=\mathcal F_h\mu$.
Let $Y$ be either position alone ($D=d$), or the position–velocity
vector ($D=2d$), and apply the following notation to the corresponding
pushforward laws. All expectations in this theorem are conditional on
$S$. Assume the proved actual one-step bounded-test estimate
$$
 \mathbb E|\widehat\nu\phi-\nu\phi|^2\le G/N,
 \qquad |\phi|\le1,
$$
with the explicit rooted-forest constant $G=A+4B_*^2$ already recorded
in this chapter, and the explicit conditional fourth-moment bounds
$$
 \mathbb E\widehat\nu|Y|^4\le M_4,
 \qquad \nu|Y|^4\le M_4<\infty.
$$
Define $\operatorname{Var}(\nu)=\nu|Y-\nu Y|^2$. Then
$$
 \mathbb E|\operatorname{Var}(\widehat\nu)
                 -\operatorname{Var}(\nu)|
 \le C_4(D)M_4^{1/2}(G/N)^{1/4},
 \qquad C_4(D)=2\sqrt2(1+2D^{1/4}).
$$
In particular the absolute bias satisfies the same estimate. No
independence between output rows, and no global attraction assumption,
is used. The two complete-update variance drifts, with their common
entering variance subtracted, differ by at most the same bound.
:::

:::{prf:proof}
Write $\widehat b=\widehat\nu Y$, $b=\nu Y$ and
$T_R(y)=y\min\{1,R/|y|\}$, with $T_R(0)=0$. Every component of
$T_R/R$ is bounded by one, so summing the scalar second-moment bounds
and taking square roots gives
$$
 \|\widehat\nu T_R-\nu T_R\|_{L^2}
 \le R\sqrt{DG/N}.
$$
Jensen's inequality for the probability measure $\widehat\nu$, then
its fourth-moment bound, gives
$$
 \mathbb E|\widehat\nu(Y-T_R)|^2
 \le\mathbb E\widehat\nu[|Y|^2\mathbf1_{|Y|>R}]
 \le M_4/R^2.
$$
The deterministic population tail has the same bound. The triangle
inequality in $L^2$ therefore yields
$$
 \|\widehat b-b\|_{L^2}\le R\sqrt{DG/N}+2\sqrt{M_4}/R.
$$
For $G,M_4>0$ choose
$R^2=2\sqrt{M_4}/\sqrt{DG/N}$, obtaining
$$
 u:=\|\widehat b-b\|_{L^2}
 \le2\sqrt2 D^{1/4}(M_4G/N)^{1/4}.
$$
Furthermore $\|\widehat b\|_{L^2}\le M_4^{1/4}$ and
$|b|\le M_4^{1/4}$ by two applications of Jensen. Consequently
$$
 \mathbb E\big||\widehat b|^2-|b|^2\big|
 \le\|\widehat b-b\|_{L^2}
       (\|\widehat b\|_{L^2}+|b|)
 \le2M_4^{1/4}u.
$$
For the second moment use $\psi_R(y)=\min\{|y|^2,R^2\}$.
The bounded-test estimate and Cauchy–Schwarz imply
$$
 \mathbb E|\widehat\nu\psi_R-\nu\psi_R|
 \le R^2\sqrt{G/N}.
$$
Each discarded second-moment tail is at most $M_4/R^2$ in expectation,
so
$$
 \mathbb E|\widehat\nu|Y|^2-\nu|Y|^2|
 \le R^2\sqrt{G/N}+2M_4/R^2.
$$
Choosing $R^4=2M_4/\sqrt{G/N}$ gives the bound
$2\sqrt2 M_4^{1/2}(G/N)^{1/4}$. These two truncation radii need
not coincide: they bound separate summands. Subtract the barycenter
squares, add the two estimates and use
$\operatorname{Var}(\nu)=\nu|Y|^2-|\nu Y|^2$. This gives precisely
$C_4(D)$. If $M_4=0$, both laws concentrate at zero almost surely.
If $G=0$, the same pre-optimization bounds tend to zero as $R\to\infty$.
Thus the assertion also covers both degenerate cases. Subtracting the
common entering variance changes neither difference nor its bound.
:::

:::{prf:corollary} Primitive-parameter moment substitution and averaged errors
:label: cor-slqc-moment-substitution

For position observables use the proved $p=4$ copying and kinetic
moment profile of {prf:ref}`thm-slcj-uniform-moments`. For this fixed
entering law, let its geometric copying profiles, including the actual
nonnegative excess, give the deterministic common upper envelope
$$
 T_4(S)=\max\left\{0,
 r_4W_4(S)+B_4+a_4\max\{E_4^N(S),E_4^\infty(L_N(S))\}\right\}.
$$
Here the displayed coefficients must be common envelopes valid for
both the finite count and population integral formulas; if different
coefficients are used, take the maximum of the two complete bounds
instead. Explicitly, with the notation of that theorem,
$$
 A_\lambda=|1-\eta\lambda|+\eta g_1,\quad
 a_4=(1+u)^3A_\lambda^4,\quad r_4=a_4(1-\chi),
$$
$$
 B_4=a_4R_c^4\left(\chi_{\rm in}
                  +\sum_{b\in\mathcal B}D_bm_b^+\right)
 +(1+u^{-1})^3
 \left[BV_c+\eta g_0+
       (A_\lambda\sigma_J+\sqrt{c^2q^2+s^2})
                         \{d(d+2)\}^{1/4}\right]^4.
$$
Then $M_4=T_4(S)$ is admissible in the theorem. For joint capped
position–velocity observables one can use
$M_4=2(T_4(S)+V_c^4)$, since
$(|x|^2+|v|^2)^2\le2|x|^4+2|v|^4$.
Thus the vanishing consistency error explicitly retains force growth,
friction, timestep, both kinetic noises, jitter, velocity cap,
companion and reward profiles, selection excess and entering moment.

If the entering swarm is random and these conditional bounds satisfy
$\sup_{N,n}\mathbb E T_4(S_n)\le\overline T_4$, then the averaged
positional error is uniformly bounded by
$$
 C_4(d)\overline T_4^{1/2}(G/N)^{1/4}.
$$
A uniform actual fourth-moment budget alone does not imply a uniform
conditional bound for every entering state: the preceding conditional
profile or its averaged counterpart must cover both $P_N$ and
$\mathcal F_h L_N$. The analogous joint bound uses
$2(\overline T_4+V_c^4)$.
:::

:::{prf:proof}
The $p=4$ full-update moment theorem supplies the two output moment
bounds separately, with $m_{d,4}=d(d+2)$. Taking their common upper
envelope gives exactly $T_4(S)$; the cap gives the stated joint
fourth moment. Condition on the entering state, apply
{prf:ref}`thm-slqc-quadratic-consistency`, and average. Concavity of
$s\mapsto\sqrt s$ gives
$\mathbb E\sqrt{T_4(S_n)}\le\sqrt{\overline T_4}$, proving the
uniform averaged assertion. This controls the complete signed quadratic
drift, including its kinetic contribution, rather than only a cloning
recipient-pressure term. It does not identify different full laws
having the same quadratic observables.
:::

(sec-slcn-rate-budget)=
### 14.4. A complete particle-error budget

:::{div} feynman-prose
Once the signed estimates prove relaxation within a declared phase class, we
can separate elapsed time from population size. The bound below has a time term
whose coefficient and decay rate are independent of $N$, plus an explicit
particle error tending to zero. Moment confinement, uniform phase attraction
and donor-coverage probabilities supply the conditions for that separation;
the Keystone pressure estimate alone does not discharge them.

The target matters. Approaching a set of stationary phases can be fast even
when moving between its phases is slow. A finite swarm initialized in one phase
cannot resemble a stationary law with substantial mass elsewhere until it has
a sufficient chance to leave. The later lower bound makes this obstruction
quantitative through the actual exit probability. Thus a population-independent
relaxation rate inside phases can coexist with a whole-swarm mixing time that
grows with population size. No common phase selection is imposed by the
mean-field law.
:::

:::{prf:theorem} An explicit vanishing particle error and a population-independent time rate
:label: thm-slcn-uniform-rate

Keep the actual all-alive algorithm, the primitive one-step consistency
constant $G_{\rm cons}=A+4B_*^2$, and a proved power-modulus regime
$$
 \mathsf d(\mathcal F_h\mu,\mathcal F_h\nu)
 \le\min\{1,C_*\max(1,H)^{3/2}\mathsf d(\mu,\nu)^\beta\},
 \qquad 0<\beta\le1,
$$
when both eighth moments are at most $H$. The numerical $C_*,\beta$
are the actual structural constants already derived, not optimal
stability constants. Assume the proved uniform particle moment bound
$\sup_{N,n}\mathbb E W_8(S_n)\le M_8$.

Let $\mathcal A$ be a common nonempty closed set of stationary population
laws. For every $H\ge1$, suppose declared Borel classes
$\mathfrak G_H\subset\{\nu:\nu|x|^8\le H\}$ have the verified bounds
$$
 \sup_{\nu\in\mathfrak G_H}
 \operatorname{dist}_{\mathsf d}(\mathcal F_h^b\nu,\mathcal A)
 \le D(1+H)^{s_{\rm ph}}q_{\rm ph}^{b-1},\quad b\ge1,
$$
$$
 \sup_{n\ge0}\Pr\{W_8(S_n)\le H,
                         L_N(S_n)\notin\mathfrak G_H\}
 \le C_{\rm cov}(1+H)^{r_{\rm cov}}N^{-\zeta}.
$$
Here $D>0$, $s_{\rm ph},r_{\rm cov}\ge0$, $0<q_{\rm ph}<1$,
$C_{\rm cov}\ge0$, and $\zeta>0$ are supplied numerical certificates,
independent of $N,H,n$. In particular, the class-wide phase estimate
must be proved; the theorem does not infer it for every landscape.

For the actual configured force use $|F(x)|\le G_0+G_1|x|$.
If an auxiliary trap is present, these are bounds for the complete force,
not just its unmodified part. Define
$$
 r_C=1+2/\kappa_C,\quad g_{8,d}=d(d+2)(d+4)(d+6),\quad
 \tau^2=c^2q^2+s^2,
$$
$$
 A_8=3^7 2^7(1+\eta G_1)^8r_C,
$$
$$
 B_8=3^7\left[2^7(1+\eta G_1)^8\sigma_J^8g_{8,d}
                 +(BV_c+\eta G_0)^8+\tau^8g_{8,d}\right],
 \qquad K=1+A_8+B_8.
$$
Thus the established finite-window eighth-moment recursion is
$M_{j+1}\le A_8M_j+B_8$. Fix any numerical $u>0$ and
$$
 c_{\rm blk}>\frac{u s_{\rm ph}}{|\log q_{\rm ph}|}.
$$
For $N\ge1$ put
$$
 L_N=\log(N+e),\quad \ell_N=1+\log L_N,\quad
 H_N=\ell_N^u,\quad b_N=1+\lceil c_{\rm blk}\log\ell_N\rceil,
 \quad Z_N=K^{b_N}(H_N+1).
$$
The logarithmic symbol $\ell_N$ here is not the finite-cell mesh.
The mesh used in the scalar consistency estimate is $N^{-1/(16d)}$.
Set
$$
 J_0=(2+2\sqrt{2d})^d(2+2V_{\max}\sqrt{2d})^d,
$$
$$
 A_N^{\rm samp}=2+\tfrac12J_0\sqrt{G_{\rm cons}}+2Z_N^{1/4},
 \quad t_N=\sqrt{A_N^{\rm samp}}N^{-1/(32d)},
 \quad C_N=\max\{1,C_*(Z_NL_N)^{3/2}\}.
$$
For an integer $b\ge1$ define
$$
 S_\beta(b)=\begin{cases}
 (1-\beta^{b-1})/(1-\beta),&0<\beta<1,\\
 b-1,&\beta=1.
 \end{cases}
$$
Let
$$
 V_N=\begin{cases}
 \min\{1,(1+C_N)^{S_\beta(b_N)}t_N^{\beta^{b_N-1}}\},&t_N\le1,\\
 1,&t_N>1,
 \end{cases}
$$
$$
 E_N=\min\{1,V_N+b_N/L_N+b_Nt_N\},\qquad
 K_{\rm ph}=2^{s_{\rm ph}}D+M_8,\quad
 \rho=q_{\rm ph}^{1/(s_{\rm ph}+1)}\in(0,1),
$$
$$
 \varepsilon_N=E_N+\frac{M_8}{H_N}
     +C_{\rm cov}(1+H_N)^{r_{\rm cov}}N^{-\zeta}
     +K_{\rm ph}\rho^{b_N-1}.
$$
Then $\varepsilon_N\to0$ and, for every $N\ge1$, $n\ge1$,
$$
 \boxed{\quad
 \mathbb E\operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le\min\{1,K_{\rm ph}\rho^{n-1}+\varepsilon_N\}.
 \quad}
$$
Both the time-decay coefficient $K_{\rm ph}$ and its rate $\rho$ are
independent of population size. Every remaining particle error is the
explicit vanishing function above. Physical time is $nh$; in particular
$\rho^{n-1}=\exp[-(n-1)|\log\rho|]$.
For a singleton target the bound is convergence to that fixed phase;
for multiple stationary phases it is distance to their set. Markov's
inequality gives the probability bound by division by a requested
positive tolerance, capped by one.
:::

:::{prf:proof}
**A common finite-window envelope.** Since
$M_{j+1}+1\le K(M_j+1)$, every window starting with moment at most
$H\le H_N$ has $M_j\le K^j(H_N+1)\le Z_N$ for $j\le b_N$.
The finite-cell ceiling estimate gives
$J(N^{1/(16d)},N^{-1/(16d)})\le J_0N^{3/16}$.
Consequently its one-step expected sampling error satisfies, for all
$j<b_N$,
$$
 a_{N,j}\le
 2N^{-1/(16d)}+\tfrac12J_0\sqrt{G_{\rm cons}}N^{-5/16}
                        +2Z_N^{1/4}N^{-1/(8d)}
 \le A_N^{\rm samp}N^{-1/(16d)}.
$$
Use the deterministic tolerance $t_N$ at every step. Under the law
of the process restarted at the specified initial configuration, Markov's
inequality bounds the step-$j$ failure probability by
$a_{N,j}/t_N\le t_N$. This is an unconditional bound within that
restart law; no bound conditional on each intermediate history is used.
The moment-localization threshold
$\max(1,M_j)L_N$ gives failure at most $1/L_N$ at that step and a
modulus coefficient at most $C_N$. The error recursion is therefore
bounded by
$$
 v_0=0,\qquad v_{j+1}\le\min\{1,t_N+C_Nv_j^\beta\}.
$$
If $t_N\le1$, induction yields
$$
 v_j\le(1+C_N)^{S_\beta(j)}t_N^{\beta^{j-1}},\qquad j\ge1.
$$
Indeed $v_1\le t_N$; if the preceding inequality holds, then
$t_N\le t_N^{\beta^j}$ and
$1+C_N(1+C_N)^{\beta S_\beta(j)}
\le(1+C_N)^{1+\beta S_\beta(j)}$.
The latter exponent is $S_\beta(j+1)$. Clipping at one preserves
the bound. Its right-hand side is nondecreasing in $j$ when
$t_N\le1$, so every window length $b\le b_N$ is bounded by $V_N$.
If $t_N>1$, the metric bound one suffices.
Adding the localization and sampling failures gives the common expected
finite-window error $E_N$, uniformly for $H\le H_N$, $b\le b_N$.

**An independent-of-$N$ time term.** For observation index $n\ge1$,
choose the actual restart length and moment cutoff
$$
 m=\min\{n,b_N\},\qquad
 H_{N,n}=\min\{H_N,q_{\rm ph}^{-(m-1)/(s_{\rm ph}+1)}\}.
$$
This cutoff belongs to $[1,H_N]$. The proved moment-localized reset
inequality gives
$$
 \mathbb E\operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le E_N+D(1+H_{N,n})^{s_{\rm ph}}q_{\rm ph}^{m-1}
       +M_8/H_{N,n}
       +C_{\rm cov}(1+H_N)^{r_{\rm cov}}N^{-\zeta}.
$$
Because $1+H\le2H$ for $H\ge1$,
$$
 D(1+H_{N,n})^{s_{\rm ph}}q_{\rm ph}^{m-1}
            \le2^{s_{\rm ph}}D\rho^{m-1},
$$
$$
 M_8/H_{N,n}
 =M_8\max\{H_N^{-1},\rho^{m-1}\}
 \le M_8/H_N+M_8\rho^{m-1}.
$$
Finally $\rho^{\min(n,b_N)-1}\le\rho^{n-1}+\rho^{b_N-1}$.
Substitution proves the displayed full-time bound.

**Every particle term vanishes.** For explicit coefficient bounds set
$$
 a_*=u+c_{\rm blk}\log K,\quad
 \overline A=2+\tfrac12J_0\sqrt{G_{\rm cons}}+2(2K^2)^{1/4},\quad
 \overline C=\max\{1,C_*(2K^2)^{3/2}\}.
$$
Since $b_N\le c_{\rm blk}\log\ell_N+2$ and $H_N\ge1$,
$$
 Z_N\le2K^2\ell_N^{a_*},\quad
 t_N\le\sqrt{\overline A}\ell_N^{a_*/8}N^{-1/(32d)},\quad
 C_N\le\overline C L_N^{3/2}\ell_N^{3a_*/2}.
$$
In particular $t_N\to0$, $b_Nt_N\to0$ and $b_N/L_N\to0$.
For $0<\beta<1$ let $z_*=c_{\rm blk}|\log\beta|$.
The cutoff formula implies
$\beta^{b_N-1}\ge\beta\ell_N^{-z_*}$.
For all $N$ where the following logarithmic upper bound is nonpositive,
$$
 \log t_N\le\tfrac12\log\overline A+
       \tfrac{a_*}{8}\log\ell_N-\frac{\log N}{32d}\le0,
$$
the untruncated upper bound defining $V_N$ has logarithm at most
$$
 \frac{\log(1+\overline C)+\frac32\log L_N+
                    \frac{3a_*}{2}\log\ell_N}{1-\beta}
 +\beta\ell_N^{-z_*}
       \left[\tfrac12\log\overline A+
         \tfrac{a_*}{8}\log\ell_N-\frac{\log N}{32d}\right].
$$
This expression tends to $-\infty$: its negative term is of order
$\log N/\ell_N^{z_*}$, which dominates both $\log L_N$ and
$\log\ell_N$, since $\ell_N=1+\log L_N$.
For $\beta=1$ the corresponding logarithmic upper bound is
$$
 (c_{\rm blk}\log\ell_N+1)
 [\log(1+\overline C)+\tfrac32\log L_N+
                        \tfrac{3a_*}{2}\log\ell_N]
 +\tfrac12\log\overline A+\tfrac{a_*}{8}\log\ell_N
                                      -\frac{\log N}{32d},
$$
which also tends to $-\infty$. Thus $V_N\to0$ and $E_N\to0$.
The remaining terms have the explicit bounds
$$
 M_8/H_N=M_8\ell_N^{-u},\qquad
 C_{\rm cov}(1+H_N)^{r_{\rm cov}}N^{-\zeta}
       \le2^{r_{\rm cov}}C_{\rm cov}\ell_N^{u r_{\rm cov}}N^{-\zeta},
$$
$$
 K_{\rm ph}\rho^{b_N-1}
 \le K_{\rm ph}\ell_N^{-c_{\rm blk}|\log q_{\rm ph}|/(s_{\rm ph}+1)}.
$$
Each tends to zero. The chosen stronger condition on $c_{\rm blk}$
also gives, for the simpler fixed cutoff $H_N$,
$$
 D(1+H_N)^{s_{\rm ph}}q_{\rm ph}^{b_N-1}
 \le 2^{s_{\rm ph}}D\ell_N^{u s_{\rm ph}
                                      -c_{\rm blk}|\log q_{\rm ph}|}
 \longrightarrow0.
$$
Thus both cutoff constructions close, while the time-dependent cutoff
above additionally removes $N$ from the time-decay coefficient.
:::

:::{prf:corollary} Prescribed accuracy, observation time and joint limits
:label: cor-slcn-accuracy

Under {prf:ref}`thm-slcn-uniform-rate`, for a requested expectation
error $\varepsilon>0$, choose any integer $N$ satisfying the explicitly
evaluable condition $\varepsilon_N\le\varepsilon/2$. It suffices to take
$$
 n\ge1+\max\left\{0,
 \left\lceil\frac{\log(2K_{\rm ph}/\varepsilon)}{-\log\rho}\right\rceil
 \right\}.
$$
For error tolerance $\varepsilon$ with failure probability at most
$\alpha\in(0,1)$, replace $\varepsilon$ by $\alpha\varepsilon$ in
these two numerical inequalities. The resulting update count has no
hidden population-dependent mixing constant. For every deterministic
$n_N\to\infty$, the actual empirical population approaches
$\mathcal A$ in expectation and probability. In the singleton version
it approaches the specified stationary phase; other phase weights
require the separate certified transition analysis.

If a numerical coverage bound $\delta_N(H)$ is available instead of
the displayed power envelope, replace the coverage contribution to
$\varepsilon_N$ by
$\sup_{1\le H\le H_N}\delta_N(H)$. The same conclusion holds whenever
that explicitly bounded term tends to zero. Pointwise convergence at
fixed $H$ alone is not substituted for this growing-cutoff requirement.
:::

:::{prf:proof}
The first two choices bound the particle error and the remaining time
term by $\varepsilon/2$ each. Markov's inequality proves the probability
version. The joint limit follows from the same displayed bound because
$\rho^{n_N-1}\to0$ and $\varepsilon_N\to0$. In the alternative
coverage version, the selected $H_{N,n}$ always lies in $[1,H_N]$,
so its actual failure term is bounded by the stated supremum; the rest
of the proof is unchanged.
:::

:::{prf:corollary} Polynomial and general verified phase-attraction profiles
:label: cor-slcn-general-profile

Retain all moment, finite-window modulus and coverage hypotheses of
{prf:ref}`thm-slcn-uniform-rate`. Replace its geometric phase-attraction
hypothesis by the following explicitly verified estimate, with the same
classes $\mathfrak G_H$, a declared nonincreasing function
$f:\{1,2,\ldots\}\to[0,1]$ satisfying $f(b)\to0$, and constants
$D>0$, $s\ge0$ independent of $N,H,b$:
$$
 \sup_{\nu\in\mathfrak G_H}
 \operatorname{dist}_{\mathsf d}(\mathcal F_h^b\nu,\mathcal A)\le D(1+H)^s f(b),\qquad H\ge1,\ b\ge1.
$$
Choose arbitrary $u>0$ and $c_{\rm blk}>0$. Keep $L_N,\ell_N,H_N,b_N$,
$Z_N,t_N,C_N,V_N,E_N$ exactly as in that theorem, with these choices;
no condition involving a geometric decay rate is imposed. Put
$$
 \theta=\frac1{s+1},\qquad K_f=2^sD+M_8,
$$
$$
 \varepsilon_N^{(f)}=E_N+\frac{M_8}{H_N}
 +C_{\rm cov}(1+H_N)^{r_{\rm cov}}N^{-\zeta}
 +K_f f(b_N)^\theta.
$$
Then, for every $N$ and $n\ge1$,
$$
 \mathbb E \operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le\min\{1,K_f f(n)^\theta+\varepsilon_N^{(f)}\},
 \qquad \varepsilon_N^{(f)}\longrightarrow0.
$$
In particular, a verified polynomial estimate
$f(b)=(1+b)^{-r}$, $r>0$, gives
$$
 \mathbb E \operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le\min\{1,K_f(1+n)^{-r/(s+1)}+\varepsilon_N^{(f)}\},
$$
with the explicit additional finite-population floor
$$
 K_f f(b_N)^\theta
 \le K_f(2+c_{\rm blk}\log\ell_N)^{-r/(s+1)}.
$$
Both the time exponent and its coefficient are independent of population
size. An expectation tolerance $\varepsilon$ follows from
$\varepsilon_N^{(f)}\le\varepsilon/2$ and
$$
 n\ge \max\left\{1,
 \left\lceil(2K_f/\varepsilon)^{(s+1)/r}-1\right\rceil\right\}.
$$
For failure probability $\alpha$, apply these inequalities with
$\varepsilon$ replaced by $\alpha\varepsilon$. Any diagonal
$n_N\to\infty$ converges to $\mathcal A$ in expectation and probability.
The estimate concerns the specified stationary set or specified phase;
it does not select weights among different phases.
:::

:::{prf:proof}
Let $m=\min(n,b_N)$ and restart at time $n-m$. Choose
$$
 H_{N,n}=\min\{H_N,f(m)^{-\theta}\},
$$
where $0^{-\theta}=+\infty$. Since $0\le f\le1$, this cutoff lies
in $[1,H_N]$. The moment and coverage bounds at the restart, followed
by the finite-window estimate uniformly over initial laws in
$\mathfrak G_{H_{N,n}}$, give
$$
 \mathbb E \operatorname{dist}_{\mathsf d}(L_N(S_n),\mathcal A)
 \le E_N+\frac{M_8}{H_{N,n}}
 +C_{\rm cov}(1+H_N)^{r_{\rm cov}}N^{-\zeta}
 +D(1+H_{N,n})^s f(m).
$$
For $f(m)>0$, the last term is at most
$2^sD f(m)^{1-s\theta}=2^sD f(m)^\theta$.
If $f(m)=0$, it vanishes and the same inequality holds. Moreover,
$$
 \frac1{H_{N,n}}=\max\{H_N^{-1},f(m)^\theta\}
 \le H_N^{-1}+f(m)^\theta.
$$
Thus the two cutoff-dependent terms are bounded by
$M_8/H_N+K_f f(m)^\theta$. Since $m$ is either $n$ or $b_N$,
$f(m)^\theta\le f(n)^\theta+f(b_N)^\theta$, proving the stated
bound. The sampling and localization proof of
{prf:ref}`thm-slcn-uniform-rate` uses only $u,c_{\rm blk}>0$;
its stronger lower bound on $c_{\rm blk}$ was used solely for the
optional fixed-cutoff geometric-attraction estimate. Hence that proof
still gives $E_N\to0$. The moment and coverage terms vanish by their
same explicit bounds, while $b_N\to\infty$ and $f(b_N)\to0$.
For polynomial $f$, $1+b_N\ge2+c_{\rm blk}\log\ell_N$ gives the
floor displayed above, and solving $K_f(1+n)^{-r/(s+1)}\le
\varepsilon/2$ gives the observation time. Markov's inequality proves
the probability statement. The joint limit follows directly from the
vanishing time term and floor. In all cases the attraction profile is
a proved input; activity-pressure estimates must first be transferred
through the actual complete-update dynamics before supplying $f$.
:::

(sec-slcn-crossing-obstruction)=
### 14.5. The population-size dependence of inter-phase crossings

:::{prf:theorem} A quantified obstruction to population-size-independent inter-phase mixing
:label: thm-slcn-slow-crossing

Let $P_N$ be the actual conservative swarm kernel and $\Pi_N$ an invariant
law. Let $G_N$ be a declared nonempty population phase region, and suppose
its full-update conditional exit bound and stationary mass satisfy

$$
\sup_{S\in G_N}P_N(S,G_N^c)\le u_N<1,
\qquad \Pi_N(G_N^c)\ge p_*>0.
$$

For any initial law $\Lambda_N$ supported on $G_N$ and every integer $n\ge0$,

$$
\|\Lambda_NP_N^n-\Pi_N\|_{\rm TV}
\ge[(1-u_N)^n-1+p_*]_+
\ge[p_*-nu_N]_+.
$$

Hence for $0<\epsilon<p_*$ and $u_N>0$, convergence within $\epsilon$
requires

$$
n\ge\frac{\log(1-p_*+\epsilon)}{\log(1-u_N)},
\qquad n\ge\frac{p_*-\epsilon}{u_N}.
$$

When $u_N=0$, the TV distance remains at least $p_*$. In particular, an
explicit crossing upper bound $u_N\le C_{\rm exit}e^{-NI_{\rm exit}}$
with $C_{\rm exit},I_{\rm exit}>0$ implies the mixing-time lower bound
$(p_*-\epsilon)e^{NI_{\rm exit}}/C_{\rm exit}$.
The same statements hold for a sampled kernel $P_N^b$, with physical
elapsed time $nbh$.

Thus $u_N\to0$ precludes a bound $C e^{-\kappa n}$ for this whole-swarm
stationary TV distance with both $C<\infty$ and $\kappa>0$ independent
of $N$. It does not preclude an $N$-independent population relaxation
rate within each phase or convergence to a union of phases.
:::

:::{prf:proof}
Conditional on all previous updates having stayed in $G_N$, the next
update remains there with probability at least $1-u_N$. Iterated
conditioning gives probability at least $(1-u_N)^n$ of staying throughout;
therefore the final mass of $G_N$ is at least that quantity, even when
returns after an exit are possible. Test the two laws on $G_N$ and use
$\Pi_N(G_N)\le1-p_*$. Bernoulli's inequality gives the linear lower
bound. Solving both inequalities for $n$ yields the necessary times.
Substitute the stated exponential upper bound on $u_N$ for the last
formula. No independent phase-label process or exact lumping has been
assumed.

For the final assertion, fix $n$ so large that $Ce^{-\kappa n}<p_*/2$.
For all sufficiently large $N$, $nu_N<p_*/2$, contradicting the proposed
upper bound and the proved lower bound. All premises concern the actual
full kernel and its stationary mass. A spatial multimodal landscape alone
does not establish these premises; its certified exit estimates determine
whether this obstruction applies.
:::

(sec-slcc-closed-active)=
## 15. A closed full-law and mean-field limit with active cloning

:::{div} feynman-prose
The kinetic update mixes a distribution, while population-dependent selection
changes the distribution entering that update. To prove relaxation with active
cloning, we must quantify both effects. Here the kinetic kernel supplies a
population-size-independent mixing estimate, and a coupling of the actual marked
root graph bounds the sensitivity of the selection correction. Copying and
component collisions remain in that calculation.

The two-update test is
$q_2=1-\varepsilon_2+2L_R+L_R^2<1$:
the kinetic mixing gain must exceed the explicitly bounded feedback terms.
When verified, this gives a sufficient regime with active selection and a force
that may include a bounded nonconvex perturbation. The calculation determines
how weak the feedback must be; it does not assume that every landscape or every
parameter choice synchronizes all phases.

There are two distances to keep separate. Relaxation of the population law is
measured in total variation. Approximation by a finite empirical swarm uses the
bounded transport metric, which allows an atomic empirical law to approach a
continuous distribution. The rooted population law and the finite-particle
consistency estimates each have their own proof; the completed argument joins
them through the displayed stability and error bounds.
:::


:::{prf:remark} Scope of this sufficient regime
:label: rem-slcc-not-general-kernel

These theorems restrict the declared update to the
conservative all-alive case and impose the displayed bounded-reward and
force-center conditions. They are not a long-time theorem for the
canonical marked update with terminal death, mandatory revival and an
arbitrary configured force. The latter update remains the one in
{prf:ref}`alg-euclidean-gas`; its signed full-update, survival and
phase-local estimates must be closed on that kernel without replacing
its force or suppressing its boundary transitions.
:::

(sec-slcc-population-stability)=
### 15.1. Deriving population stability from the actual kernel

:::{prf:definition} Primitive parameter regime and base kinetic kernel
:label: def-slcc-regime

Work with the conservative all-alive current-frame canonical population
map $\mathcal F_h$ on $E=\mathbb R^d\times\overline B_{V_{\max}}$.
Retain its actual measurement companions, normalized Gaussian cloning
companions, sampled fitness, accepted-edge components, shared component
rotations, collision coefficient $\alpha_{\rm col}$, recipient Gaussian
jitter and full kinetic update. There is no viscosity or historical donor
term. The raw reward is bounded with oscillation at most $R_{\rm osc}<\infty$.
For the weak-metric mean-field consequences, additionally use the stated
Lipschitz reward hypothesis. The symbol $R_{\rm osc}$ denotes oscillation,
not the absolute reward bound used in the weak-modulus formulas. Because
regularized standardization is unchanged by a constant reward shift, one
may center the bounded reward range and use $|R|\le R_{\rm osc}/2$ in
those formulas without changing the actual transition kernel. Let the positive reward/diversity floors,
amplitudes and standardization regularizers be
$(\eta_b,A_b,\sigma_b)$, $b=r,s$, with exponents $p_b\ge0$.

Put $c=h/2$, $a=e^{-\gamma h}>0$, $B=c(1+a)$,
$\eta=c^2(1+a)$, $q^2=b_O^2(1-e^{-2\gamma h})/(2\gamma)>0$,
$s^2=\sigma_x^2h>0$, and $\tau^2=c^2q^2+s^2$, using the continuous
$\gamma=0$ convention. Require the following explicitly evaluated
landscape profiles:

$$
H_c:=\sup_x|x+\eta F(x)|<\infty,
\qquad \operatorname{Lip}(F)\le L_F<\infty.
$$

This is a strong but concrete sufficient kinetic regime, stated in terms
of the actual discrete position-center profile. It imposes no convexity
and no restriction $c^2L_F<1$. It does not cover all confined landscapes.
The bounded reward channel and the actual kinetic force are separately
declared inputs of this certificate. It does not silently clip an
unbounded reward or add an auxiliary force. They can coexist when those
channels have already been configured separately, including a configured
auxiliary trap. If the same potential supplies $R=-U$ and
$F=-\nabla U$ with unbounded raw reward, the present bounded-reward
certificate does not apply; its normalization sensitivity needs the
separate moment-dependent analysis. All configured parameters, profiles
and exponents here are fixed independently of $N$.
Define $K$ first as the actual one-row kinetic Markov kernel from any
finite phase-space input $(x,v)\in\mathbb R^{2d}$ into $E$, with these
same parameters and without a preceding cloning event. Its restriction
$E\to E$ is the reference Markov kernel in the mixing proof. Its natural
extension is used when a collision-prepared velocity has norm up to
$V_c=(1+2|\alpha_{\rm col}|)V_{\max}$. The full population map retains
active cloning; this reference kernel is only part of the proof.
:::

:::{prf:lemma} Explicit two-update mixing of the reference kinetic kernel
:label: lem-slcc-base-mixing

Under {prf:ref}`def-slcc-regime`, define

$$
\begin{aligned}
\lambda_0&=1/\eta,\quad g_0=H_c/\eta,\quad
b_K=1+(BV_{\max}+H_c)^2+d\tau^2,\\
R_0&=\sqrt{4b_K-1},\quad
\alpha_0=1-c^2\lambda_0=a/(1+a),\\
R_1&=\alpha_0R_0+cV_{\max}+c^2g_0,\quad
m_v=a(V_{\max}+c\lambda_0R_0+cg_0).
\end{aligned}
$$

Choose any declared $r,u>0$, and put

$$
\begin{aligned}
Q&=(u+c\lambda_0R_1+cg_0)/\alpha_0,\\
k_v&=(2\pi q^2)^{-d/2}
 e^{-(Q+m_v)^2/(2q^2)}(1+c^2L_F)^{-d},\\
k_x&=(2\pi s^2)^{-d/2}
 e^{-(r+R_1+cQ)^2/(2s^2)},\\
\epsilon_K&=v_d(u)v_d(r)k_vk_x,\qquad
\epsilon_2=3\epsilon_K/4>0,
\qquad v_d(t)=\pi^{d/2}t^d/\Gamma(1+d/2).
\end{aligned}
$$

Then there is the explicit probability $\nu$ consisting of independent
uniform position on $B_r$ and cap-pushed uniform pre-cap velocity on
$B_u$ such that $K^2(z,\cdot)\ge\epsilon_2\nu$ for every $z\in E$.
Thus, for every signed measure $\xi$ of total mass zero,

$$
\|\xi K^2\|_{\rm TV}\le(1-\epsilon_2)\|\xi\|_{\rm TV}.
$$

Here the TV norm for a zero-mass signed measure is half its full
variation norm.
:::

:::{prf:proof}
Set $f(x)=F(x)+\lambda_0x$; the profile gives $|f|\le g_0$.
The exact position update is $x^+=Bv+\eta f(x)+cq\xi+s\zeta$,
so $K(1+|x|^2)\le b_K$. On $1+|x|^2\le4b_K$ the constants
$R_0,R_1,m_v,Q,k_v,k_x$ are precisely those in the noninjective
kinetic minorization {prf:ref}`lem-slcp-surjective-minorization`,
with $V_c$ replaced by the input bound $V_{\max}$. Its hypotheses
hold because $\alpha_0>0$, $f$ is bounded and $F$ is Lipschitz.
Thus $K(z,\cdot)\ge\epsilon_K\nu$ on that set.
Markov's inequality gives probability at least $3/4$ of entering it in
one update from any state. Integrating the minorization gives the global
$K^2$ bound. Subtracting its common component leaves a Markov kernel
with coefficient $1-\epsilon_2$; its action on a zero-mass measure
contracts TV. All constants are independent of population size.
:::

:::{prf:lemma} A small, explicit Lipschitz bound for the actual selection perturbation
:label: lem-slcc-selection-perturbation

Let $D_*$ be the configured comparison-feature diameter and let
$\kappa_D=e^{-D_*^2/(2\epsilon_D^2)}$,
$\kappa_C=e^{-D_*^2/(2\epsilon_C^2)}$.
The range of sampled raw diversity is bounded by
$S_b=\sqrt{D_*^2+\delta_D^2}-\delta_D$.
Define

$$
\begin{aligned}
K_D&=1+2/\kappa_D,\\
T_r&=R_{\rm osc}/\sigma_r+R_{\rm osc}^3/(2\sigma_r^3),\\
T_s&=K_D[S_b/\sigma_s+S_b^3/(2\sigma_s^3)],\\
H_b&=\frac{A_bp_b}{4}
 \max\{\eta_b^{p_b-1},(\eta_b+A_b)^{p_b-1}\}
 (\eta_{b'}+A_{b'})^{p_{b'}},\quad b'\ne b,\\
C_F&=H_rT_r+H_sT_s,\qquad
F_* =\eta_r^{p_r}\eta_s^{p_s},\quad
F^*=(\eta_r+A_r)^{p_r}(\eta_s+A_s)^{p_s},\\
L_g&=\frac1{s_c(F_*+\epsilon_c)}
       +\frac{F^*-F_*}{s_c(F_*+\epsilon_c)^2},\\
a_*&=\min\{1,(F^*-F_*)/[s_c(F_*+\epsilon_c)]\},\quad
c_*=a_*/\kappa_C,\quad r_*=a_*+c_*,\\
L&=2c_*K_D+a_*/\kappa_C^2+2L_gC_F/\kappa_C.
\end{aligned}
$$

Set $H_b=0$ when $p_b=0$. If $2c_*<1$, define

$$
L_R=2r_*K_D+\frac{2L}{1-2c_*}.
$$

For the actual nonlinear signed perturbation
$\mathcal R(\mu)=\mathcal F_h(\mu)-\mu K$, one has the global bound

$$
\boxed{\quad
\|\mathcal R(\mu)-\mathcal R(\nu)\|_{\rm TV}
\le L_R\|\mu-\nu\|_{\rm TV}.
\quad}
$$

This estimate includes the complete accepted component and its actual
collision transformation. It does not set the collision coefficient to
zero, replace the component by a donor pair, or couple different
computed fitness values as if their marked laws coincided.
:::

:::{prf:proof}
Put $\delta=\|\mu-\nu\|_{\rm TV}$. First maximally couple their
physical root states. At equal physical roots, the two normalized
measurement-companion laws can be coupled with failure probability at
most $2\delta/\kappa_D$: subtract their weighted numerators and
normalizers, whose denominators are at least $\kappa_D$.
Hence the laws of underlying physical/measurement pairs can be coupled
with bad mass at most $K_D\delta$. The fitness value is not part of
this equality event.

For a variable with range length $R$, the mean difference of two
probabilities at TV distance $\delta$ is at most $R\delta$.
Writing variance as half the mean squared difference of two independent
copies shows its difference is at most $R^2\delta$: the product-law
TV distance is at most $2\delta$, and the squared difference has range
in $[0,R^2]$. On a matched underlying sample, the numerator of its
standardized value has magnitude at most $R$. The derivative of
$(v+\sigma^2)^{-1/2}$ is bounded by $1/(2\sigma^3)$ for $v\ge0$.
These facts give the standardized-reward bound $T_r\delta$.
Apply the same argument to the coupled measurement-pair laws, at TV
distance at most $K_D\delta$, to obtain $T_s\delta$.
The bounded logistic power derivatives are $H_b$, so matched underlying
types have fitness differences at most $C_F\delta$.

The actual clipped gate is bounded by $a_*$. On its positive branch,
the donor derivative is $1/[s_c(F_i+\epsilon_c)]$, and the absolute
recipient derivative is
$1/[s_c(F_i+\epsilon_c)]+(F_j-F_i)/[s_c(F_i+\epsilon_c)^2]$.
Both are bounded by $L_g$; clipping preserves this sum-norm Lipschitz
bound. Thus gates at two good recipient/donor type pairs differ by at
most $2L_gC_F\delta$. At matched physical recipients the donor
normalizers differ by at most $\delta$ and are at least $\kappa_C$.
Consequently the accepted-edge densities differ on good paired types
by at most

$$
[a_*/\kappa_C^2+2L_gC_F/\kappa_C]\delta,
$$

and each density is bounded by $c_*$.
Lift outgoing accepted measures and incoming Poisson intensities to the
coupling of underlying types. Bad paired types cost at most
$2c_*K_D\delta$ in total intensity, so their full accepted intensity
discrepancy is at most $L\delta$. Complete an outgoing accepted
subprobability by its no-edge outcome; a coupling then fails with
probability at most $L\delta$. Couple incoming Poisson processes by
the common intensity and independent residual intensities. The chance
of any residual point is at most their total intensity, again at most
$L\delta$. The two directions therefore cost at most $2L\delta$
per exposed matched vertex.

Here every fixed marked vertex has outgoing acceptance probability at
most $a_*$, and total incoming Poisson mean at most $c_*$. Exploring
its rooted component exposes an expected number of new vertices at
most $a_*+c_*\le2c_*$ per visited vertex. An incoming child has already
used its outgoing edge, reducing this bound. Conditional on all exposed
types, any remaining incoming points retain their Poisson intensity;
conditioning an outgoing target changes its type distribution but not
the uniform bounds just stated. By induction, the expected generation
sizes are at most $(2c_*)^j$, so the expected number of vertices is at
most $(1-2c_*)^{-1}$. To bound a coupled exploration stopped at its first mismatch, extend
its first marginal to its ordinary complete rooted tree, leaving
unexposed future randomness unconditioned. Pathwise, the number of
queried matched vertices is at most the size of this complete first
tree. Its conditional expected size is bounded by $(1-2c_*)^{-1}$
uniformly over the given good root type. At every queried matched
vertex, the fresh coupling-failure probability, conditional on the
revealed history and that vertex's types, is at most $2L\delta$ by the
uniform intensity estimates. Summing these hazards and using the
pathwise vertex-count domination gives total mismatch probability at
most $2L\delta/(1-2c_*)$. This argument does not assert that offspring
laws are unchanged after conditioning on previous matching events.

For a test $0\le\varphi\le1$, let $g(z)=K\varphi(z)$, also in
$[0,1]$, defined on every possible prepared state. If $Z$ is the root
physical state and $Z^C$ its full prepared state, then

$$
\mathcal R(\mu)\varphi
=\mathbb E_\mu[g(Z^C)-g(Z)].
$$

An isolated root has $Z^C=Z$: it neither copies nor jitters, and its
singleton collision leaves its velocity unchanged. For any fixed
underlying marked root, the probability of a nontrivial component is
at most $r_*=a_*+c_*$. On bad root pairs, each of the two signed
increments therefore has conditional expected absolute value at most
$r_*$; their total contribution is at most $2r_*K_D\delta$.
On good roots the baseline $g(Z)$ agrees. If the component explorations
match, couple their Haar rotation and jitters identically; their full
prepared states agree. If they fail, the two output values of $g$
differ by at most one. Their contribution is therefore at most the
proved mismatch probability. Adding these bounds and taking the
supremum over $0\le\varphi\le1$ proves the claimed TV estimate for
the zero-mass signed difference. Every term vanishes with the gate
bound and fitness-normalizer sensitivity.
:::

:::{prf:theorem} Closed active-cloning full-law contraction
:label: thm-slcc-active-contraction

Use {prf:ref}`def-slcc-regime` and the explicitly evaluated constants
$\epsilon_2,L_R$ above. If

$$
2c_*<1,\qquad
q_2:=1-\epsilon_2+2L_R+L_R^2<1,
$$

then the actual nonlinear map has exactly one stationary probability
$\pi$, and, for every entering population probability $\mu$,

$$
\boxed{\quad
\|\mathcal F_h^n(\mu)-\pi\|_{\rm TV}
\le q_2^{\lfloor n/2\rfloor}.
\quad}
$$

Thus $n=2\lceil\log(1/\varepsilon)/[-\log q_2]\rceil$ updates,
or physical time $nh$, guarantee TV error at most $\varepsilon\in(0,1)$.
All constants are independent of particle number. The full map's uniform
position moment is

$$
M_2=(BV_c+H_c)^2+d\tau^2,
\qquad V_c=(1+2|\alpha_{\rm col}|)V_{\max},
$$

and its $p$th moment, $p\ge1$, is at most
$[BV_c+H_c+\tau m_{d,p}^{1/p}]^p$ with
$m_{d,p}=2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2)$.
These bounds hold after every full update, including active jitter.
:::

:::{prf:proof}
The decomposition $\mathcal F_h\mu=\mu K+\mathcal R(\mu)$ gives
$\|\mathcal F_h\mu-\mathcal F_h\nu\|_{\rm TV}
\le(1+L_R)\|\mu-\nu\|_{\rm TV}$.
Expanding the second iterate gives exactly

$$
\mathcal F_h^2\mu-\mathcal F_h^2\nu
=(\mu-\nu)K^2+
[\mathcal R(\mu)-\mathcal R(\nu)]K+
[\mathcal R(\mathcal F_h\mu)-\mathcal R(\mathcal F_h\nu)].
$$

Apply reference-kernel two-update contraction, one-update Markov
nonexpansion and the proved perturbation bound to get factor
$(1-\epsilon_2)+L_R+L_R(1+L_R)=q_2$.
The space of probability measures is complete in TV, and the map is
defined on all of it under the bounded-reward hypotheses. The contraction
principle gives a unique fixed probability for $\mathcal F_h^2$.
Its image under $\mathcal F_h$ is another such fixed probability, so
it is fixed under $\mathcal F_h$. Conversely every fixed law of
$\mathcal F_h$ is fixed under its second iterate, proving uniqueness.
Iterate the contraction and use TV distance at most one between
probabilities. For odd $n$, start that same even-iterate argument from
$\mathcal F_h\mu$; no extra factor is necessary.

For moments, after the actual cloning and collision preparation the
center of final position is $X+\eta F(X)+BV$, of norm at most
$H_c+BV_c$, irrespective of copying or jitter. Conditional Gaussian
second moments and Minkowski's inequality give the stated bounds and
the corresponding stationary bounds by invariance.
:::

:::{prf:corollary} An explicitly nonempty interval of positive selection exponents
:label: cor-slcc-positive-exponents

Fix reference exponents $\bar p_r\ge0$, $\bar p_s>0$, and put
$p_b=\theta\bar p_b$. Keep all other actual algorithm and landscape
parameters fixed. Define the primitive numbers

$$
\begin{aligned}
M&=\sum_{b=r,s}\bar p_b
 \max\{|\log\eta_b|,|\log(\eta_b+A_b)|\},\\
\Delta&=\sum_{b=r,s}\bar p_b
 \log[(\eta_b+A_b)/\eta_b]>0,\qquad D_0=e^{-M}+\epsilon_c,\\
a_0&=\frac{e^M\Delta}{s_cD_0},\qquad
C_0=e^M\sum_{b=r,s}\frac{\bar p_b A_b}{4\eta_b}T_b,\\
G_0&=\frac1{s_cD_0}+\frac{e^M\Delta}{s_cD_0^2},\\
L_0&=\frac{2a_0K_D}{\kappa_C}+\frac{a_0}{\kappa_C^2}
                              +\frac{2G_0C_0}{\kappa_C},\\
H_0&=2a_0(1+\kappa_C^{-1})K_D+4L_0,\\
\theta_{\max}&=\min\{1,\kappa_C/(4a_0),\epsilon_2/(4H_0)\}>0.
\end{aligned}
$$

For every $0<\theta\le\theta_{\max}$, the complete actual dynamics
satisfies {prf:ref}`thm-slcc-active-contraction`, and

$$
q_2\le1-7\epsilon_2/16<1.
$$

In particular choosing $\theta=\theta_{\max}/2$ supplies one fully
specified positive diversity exponent; when $\bar p_r>0$, the reward
exponent is positive as well. Positive exponents permit actual accepted
edges whenever the realized fitnesses differ. The proof does not assert
that an input with identical fitness must clone.
:::

:::{prf:proof}
For $0<\theta\le1$, every powered fitness factor gives
$e^{-M}\le F_*\le F^*\le e^M$ and
$F^*-F_*\le\theta e^M\Delta$, by integrating its exponential
variation between the lower and upper base values. Hence
$a_*\le\theta a_0$.
For a logistic factor $u\in[\eta_b,\eta_b+A_b]$, its logarithmic
powered derivative is at most
$\theta\bar p_b A_b/(4\eta_b)$. Multiplying by the full fitness
bound gives $C_F\le\theta C_0$. The gate derivative bound satisfies
$L_g\le G_0$. Therefore $L\le\theta L_0$ and
$r_*\le\theta a_0(1+\kappa_C^{-1})$.
If $\theta\le\kappa_C/(4a_0)$, then $2c_*\le1/2$, so

$$
L_R\le2\theta a_0(1+\kappa_C^{-1})K_D+4\theta L_0
=\theta H_0\le\epsilon_2/4.
$$

Since $0<\epsilon_2\le1$,
$2L_R+L_R^2\le\epsilon_2/2+\epsilon_2^2/16
\le9\epsilon_2/16$. Substitution gives the displayed strict
contraction bound. Every factor in $\theta_{\max}$ is finite and
strictly positive under the stated primitive hypotheses, proving
nonemptiness without an unspecified small-selection limit.
:::

(sec-slcf-nonlinear-restart)=
### 15.2. Uniform-time particle approximation and both limit orders

:::{prf:theorem} Explicit global restart closure from actual population TV contraction
:label: thm-slcf-nonlinear-restart

Throughout this theorem and its corollaries, let $N\ge2$.
Use the actual active-cloning two-step TV-contraction regime established
in {prf:ref}`thm-slcc-active-contraction`, with its fully derived constants
$$
 q_2=1-\epsilon_2+2L_R+L_R^2\in(0,1).
$$
Here $L_R$ denotes that theorem's signed cloning-remainder Lipschitz
constant, not the reward Lipschitz constant. Retain the bounded-reward
regularity hypotheses and the explicit global weak-modulus constant
$C_{\rm mod}$ of {prf:ref}`thm-slct-modulus`; in particular the actual
force has a finite stated global Lipschitz constant. No separate phase
attraction, coverage, moment-confinement or weak-to-TV smoothing
hypothesis is required.

The configured resonant force is
$F(x)=-\kappa x+f(x)$, $|f(x)|\le H$, $\eta\kappa=1$.
Set
$$
 b_0=BV_c+\eta H,\qquad \tau^2=c^2q^2+s^2,
 \qquad M_2=b_0^2+d\tau^2.
$$
This bounds the expected second positional moment after a complete
update for every entering configuration and every population input.
No initial positional moment bound is imposed. Define
$$
 A_0=2+\tfrac12(2+2\sqrt{2d})^d
                  (2+2V_{\max}\sqrt{2d})^d\sqrt{A+4B_*^2}+2M_2,
 \qquad a_N=\min\{1,A_0N^{-1/(16d)}\},
$$
$$
 b_N=1+\left\lfloor\frac{\log\log(N+e)}{2\log4}\right\rfloor,
 \quad V_N=\min\{1,(1+C_{\rm mod})^{4/3}
                              a_N^{4^{-(b_N-1)}}\},
 \quad \epsilon_N=V_N+q_2^{\lfloor b_N/2\rfloor}.
$$
The population map has a unique stationary law $\pi$, and
$$
 \|\mathcal F_h^n\mu-\pi\|_{\rm TV}\le q_2^{\lfloor n/2\rfloor}.
$$
For every particle initialization,
$$
 \boxed{\quad
 \mathbb E\mathsf d(L_N(S_n),\pi)
 \le\min\{1,q_2^{\lfloor n/2\rfloor}+\epsilon_N\},\qquad n\ge1,
 \quad \epsilon_N\longrightarrow0.\quad}
$$
For every finite-row assertion take $1\le k\le N$. The bounded product
transport metric is induced by
$$
 c_k(z,z')=\min\left\{1,\sum_{i=1}^k\min(1,|z_i-z_i'|)\right\};
$$
thus its diameter is one, including on the sampling-collision event.

The time rate and its coefficient are independent of $N$. In particular
all deterministic joint limits $N\to\infty$, $n_N\to\infty$ converge
to $\pi$ in expectation and probability. Every finite-particle stationary law
satisfies expected empirical distance at most
$\epsilon_N$. Under exchangeability, their $k$-particle bounded-product-
metric chaos error is at most $k\epsilon_N+k(k-1)/(2N)$.

For a deterministic population trajectory $\mu_n=\mathcal F_h^n\mu_0$,
put $d_{N,0}=\mathbb E\mathsf d(L_N(S_0),\mu_0)$.
For every fixed $T$, the recursion
$$
 u_0=d_{N,0},\qquad u_{j+1}=\min\{1,a_N+C_{\rm mod}u_j^{1/4}\}
$$
bounds the trajectory error through time $T$. Together with the boxed
bound this proves uniform-time trajectory convergence when $d_{N,0}\to0$:
$$
 \sup_{n\ge0}\mathbb E\mathsf d(L_N(S_n),\mu_n)
 \le\max\left\{\max_{0\le j<T}u_j,
                     2q_2^{\lfloor T/2\rfloor}+\epsilon_N\right\}.
$$
One may choose $T$ for the desired time tolerance first and then evaluate
the displayed finite recursion and $\epsilon_N$ to select population size.
:::

:::{prf:proof}
The admissible capped-law space is complete in TV. In this bounded-reward
regime the marked population operator is defined for every such law:
companion weights have positive global floors, fitness normalization is
regularized, and the proved rooted component construction is almost surely
finite. Hence the contraction mapping argument applies to $\mathcal F_h^2$
without imposing a TV-closed moment constraint. For completeness, the
successive even iterates have TV differences at most $q_2^j$;
the geometric sum makes them Cauchy. Completeness gives a limit, and
contraction continuity shows that it is fixed by $\mathcal F_h^2$.
Contracting two such fixed points proves uniqueness. Since
$\mathcal F_h\pi$ is also fixed by $\mathcal F_h^2$, uniqueness gives
$\mathcal F_h\pi=\pi$. For $n=2j+r$, $r\in\{0,1\}$, apply the
$j$ contractions to $\mathcal F_h^r\mu$ and $\pi$. The TV diameter
one gives $q_2^j$ with no odd-step prefactor.

Resonance gives the exact final position
$X^+=BV^C+\eta f(X^C)+cq\xi+s\zeta$. Conditional on preparation,
its deterministic mean has norm at most $b_0$, and its centered Gaussian
noise has covariance $\tau^2I_d$. Thus its second moment is at most
$M_2$, regardless of the prepared position, the jitter, or the realized
cloning and collision graph. The same calculation applies to the rooted
population law. In particular invariant particle laws automatically
have this moment bound, by stationarity and truncation.

The scalar consistency and cell discretization calculation give uniform
expected one-step weak error at most $a_N$: with spatial radius
$N^{1/(16d)}$ and mesh $N^{-1/(16d)}$, the bound is
$$
 2N^{-1/(16d)}+\tfrac12J_0\sqrt{A+4B_*^2}N^{-5/16}
                         +2M_2N^{-1/(8d)}\le A_0N^{-1/(16d)},
$$
where $J_0=(2+2\sqrt{2d})^d(2+2V_{\max}\sqrt{2d})^d$.
The metric bound one permits clipping at $a_N$.
Fix an observation index $n$, let $m=\min(n,b_N)$, and start a random
population trajectory at the actual empirical law at time $n-m$.
Its initial comparison error is zero. At every later step insert the
population forecast of the actual empirical law. The triangle inequality,
the global weak modulus, and Jensen's inequality give the expected-error
recursion $v_0=0$, $v_{j+1}\le\min(1,a_N+C_{\rm mod}v_j^{1/4})$.
Here the sampling errors are averaged under the original particle law;
no uniform conditional moment bound on the restart configuration is needed.
Induction gives, for $j\ge1$,
$$
 v_j\le(1+C_{\rm mod})^{\sum_{k=0}^{j-2}4^{-k}}
                         a_N^{4^{-(j-1)}}
 \le(1+C_{\rm mod})^{4/3}a_N^{4^{-(j-1)}}.
$$
This upper bound is nondecreasing in $j$ because $a_N\le1$, so
$v_m\le V_N$. For each realized restart law, TV attraction bounds its
population iterate's distance to $\pi$ by $q_2^{\lfloor m/2\rfloor}$.
Since $m$ is either $n$ or $b_N$,
$q_2^{\lfloor m/2\rfloor}\le q_2^{\lfloor n/2\rfloor}+
q_2^{\lfloor b_N/2\rfloor}$, this proves the boxed claim.

Set $L=\log(N+e)$. The block definition gives
$4^{-(b_N-1)}\ge L^{-1/2}$. For all sufficiently large $N$,
$\log A_0-\log N/(16d)\le0$, and consequently
$$
 \log\big[(1+C_{\rm mod})^{4/3}
                    a_N^{4^{-(b_N-1)}}\big]
 \le\frac43\log(1+C_{\rm mod})+
           L^{-1/2}\left(\log A_0-\frac{\log N}{16d}\right)
 \longrightarrow-\infty.
$$
Also $b_N\to\infty$, so $q_2^{\lfloor b_N/2\rfloor}\to0$. Thus every particle
term vanishes explicitly. Stationarity allows $n\to\infty$ in the
boxed estimate; exchangeability and the sampling-without-replacement
argument give the marginal bound. The finite-horizon recursion follows
by the same insertion and Jensen argument starting from $d_{N,0}$.
For $n\ge T$, insert $\pi$ between the particle and deterministic
population laws; the respective errors are at most
$q_2^{\lfloor n/2\rfloor}+\epsilon_N$ and $q_2^{\lfloor n/2\rfloor}$. Taking the two time ranges proves
the last display. First choose $T$ large and then $N$ large to obtain
uniform-time trajectory convergence. Markov's inequality proves all
probability claims.
:::

:::{prf:remark} Metrics in the closed limit
:label: rem-slcf-metrics

This closure uses the actual population TV contraction only for population
laws. It never compares an atomic empirical law to a diffuse law in TV.
The contraction coefficient is the explicit $q_2$ of {prf:ref}`thm-slcc-active-contraction`, not an unknown phase-attraction constant.
:::


:::{prf:corollary} Explicit accuracy and probability budgets
:label: cor-slcf-accuracy

For an expected distance tolerance $\varepsilon\in(0,1)$, choose $N$
by the explicitly evaluable condition $\epsilon_N\le\varepsilon/2$
and take
$$
 n\ge 2\left\lceil\frac{\log(2/\varepsilon)}{|\log q_2|}\right\rceil.
$$
Then $\mathbb E\mathsf d(L_N(S_n),\pi)\le\varepsilon$.
For distance tolerance $\varepsilon$ with failure probability at most
$\alpha\in(0,1)$, replace $\varepsilon$ by $\alpha\varepsilon$ in
both displayed choices. The physical observation time is $nh$, and
population work scales with the actual $N$ updates per iteration; this
bound does not treat increased population as free exploration.
:::

:::{prf:proof}
The update count makes $q_2^{\lfloor n/2\rfloor}\le\varepsilon/2$.
Add the particle floor and apply Markov's inequality for the probability
version. Every constant is the displayed structural or algorithmic
quantity in the two preceding theorems.
:::

:::{prf:corollary} Closed-form sufficient population and observation sizes
:label: cor-slcf-closed-sizes

Let $C=(1+C_{\rm mod})^{4/3}$ and
$r=-\log q_2/(4\log4)>0$. For a particle-error budget
$0<\zeta<1$, define
$$
 L_*(\zeta)=\max\left\{
 32d\log A_0+2,
 [32d\log(2C/\zeta)]^2,
 [2/(q_2\zeta)]^{1/r}
 \right\}.
$$
Every integer $N\ge\lceil\exp L_*(\zeta)\rceil$ satisfies
$\epsilon_N\le\zeta$. Consequently, for distance tolerance
$0<\delta<1$ and failure probability $0<\alpha<1$, the fully explicit
choices
$$
 N\ge\left\lceil\exp L_*(\alpha\delta/2)\right\rceil,
 \qquad
 n\ge2\left\lceil
       \frac{\log(2/(\alpha\delta))}{-\log q_2}
       \right\rceil
$$
ensure
$\Pr\{\mathsf d(L_N(S_n),\pi)>\delta\}\le\alpha$.
Their physical observation time is $nh$ and their number of row updates
is $Nn$. These are sufficient analytical sizes, not optimized estimates
of computational cost.
:::

:::{prf:proof}
Write $L=\log(N+e)$. For $N\ge2$,
$L-\log N=\log(1+e/N)<1$, so if $L\ge32d\log A_0+2$, then
$$
 \log A_0-\frac{\log N}{16d}
 \le\log A_0-\frac{L-1}{16d}
 \le-\frac{L}{32d}.
$$
In particular $a_N=A_0N^{-1/(16d)}<1$. Since
$4^{-(b_N-1)}\ge L^{-1/2}$, multiplying the negative logarithmic
bound by that exponent gives
$$
 V_N\le C\exp[-\sqrt L/(32d)].
$$
Also $b_N=1+\lfloor\log L/(2\log4)\rfloor$ implies
$$
 \lfloor b_N/2\rfloor\ge\frac{\log L}{4\log4}-1,
 \qquad
 q_2^{\lfloor b_N/2\rfloor}\le q_2^{-1}L^{-r}.
$$
The second and third terms defining $L_*(\zeta)$ make these two error
contributions at most $\zeta/2$ each. The specified $N$ ensures
$L\ge L_*(\zeta)$ and $N\ge2$, proving $\epsilon_N\le\zeta$.
Finally choose $\zeta=\alpha\delta/2$; the displayed update count
makes the remaining time term at most $\alpha\delta/2$. Thus the
expected metric error is at most $\alpha\delta$, and Markov's inequality
proves the confidence statement.
:::

:::{prf:corollary} Stationary existence, full-sequence chaos and commuting limits
:label: cor-slcf-stationary-limits

In the same active-cloning regime, apply
{prf:ref}`thm-slca-active-stationary` with $\lambda=\kappa$ and residual
force bound $H$. For each $N$ the actual complete swarm kernel has its
unique invariant law $\Pi_N$ and converges to it in whole-swarm TV from
any initial law. Then
$$
 \int\mathsf d(L_N(S),\pi)\,\Pi_N(dS)\le\epsilon_N,
 \qquad
 \Pi_N\circ L_N^{-1}\Longrightarrow\delta_\pi
$$
along the full population-size sequence. Moreover $\Pi_N$ is
exchangeable and, for every fixed $k$, its $k$-row marginal converges
to $\pi^{\otimes k}$ with bounded-product-metric error at most
$k\epsilon_N+k(k-1)/(2N)$.

For initialized empirical laws satisfying
$\mathbb E\mathsf d(L_N(S_0),\mu_0)\to0$, both iterated limits
$N\to\infty$ then $n\to\infty$, and $n\to\infty$ then
$N\to\infty$, equal $\delta_\pi$ for empirical-law distributions in
the stated weak metric. Every diagonal $N\to\infty$, $n_N\to\infty$
has the same limit. If the initialized swarm laws are exchangeable,
the corresponding fixed-$k$ row laws in all these limits converge to
$\pi^{\otimes k}$. None of these assertions requires a
population-independent whole-swarm TV mixing rate.
:::

:::{prf:proof}
The force, cap, noises, cloning and resonance are the same as in
{prf:ref}`thm-slca-active-stationary`; its finite-particle recurrence and
full-kernel minorization therefore supply existence, uniqueness and
whole-swarm TV convergence. Resonance gives the uniform output moment
bound already proved, so every $\Pi_N$ has second moment at most
$M_2$ without an additional assumption. Start the chain at $\Pi_N$ and
let $n\to\infty$ in the uniform expected-distance estimate to obtain
$\epsilon_N$. Markov's inequality then proves convergence in probability
of $L_N$ to $\pi$ under $\Pi_N$, hence the displayed empirical-law
limit along the full sequence.

Permutation equivariance of the full kernel makes every permutation
pushforward of $\Pi_N$ invariant. Uniqueness implies that these
pushforwards equal $\Pi_N$, proving exchangeability; the already proved
sampling estimate gives its marginal rate. At fixed $N$, whole-swarm TV
convergence implies convergence of empirical-law distributions and row
marginals, by pushforward. Sending $N\to\infty$ now gives the stationary
limits just proved. In the opposite order, finite-horizon consistency
and consistent initialization give the empirical-law limit
$\delta_{\mathcal F_h^n\mu_0}$; population TV contraction then sends
this law to $\delta_\pi$. Exchangeability converts each fixed-time
empirical approximation into the asserted row-marginal limit by the
same sampling argument. The uniform bound directly proves the diagonal
statement. The finite-$N$ TV mixing constants supplied by the recurrence
theorem may depend on $N$ throughout this argument.
:::

(sec-slcw-unbounded-active)=
## 16. Closed mean-field convergence with the unbounded raw reward

:::{div} feynman-prose
A small amount of probability far away can change the mean and variance of an
unbounded reward, and therefore change every walker's normalized fitness.
Ordinary total variation records how much probability moved but not how far
out its reward contribution lies. The weight $1+\beta_w|x|^4$ makes that
contribution part of the distance being controlled. This permits the original
quadratic-growth reward to remain in the algorithm without clipping it.

The force condition bounds the completed deterministic position contribution
$|x+\eta F(x)|$. Combined with the velocity cap and Gaussian innovations, it
gives explicit output moments. Those moments control the reward-normalization
feedback and the finite-particle errors. Positive reward and diversity selection,
copying and component collisions remain present. In this regime the reward and
force may come from the same potential, $R=-U$ and $F=-\nabla U$, under the
stated growth and regularity bounds.

The explicit test $q_w<1$ closes the mixing and feedback estimates. Once it
holds, the population law relaxes in weighted total variation, while empirical
swarms approximate it in the bounded transport metric, uniformly in time under
the stated initialization conditions. The stationary and joint limits then
follow from the proved bounds. This supplies a computable sufficient regime;
it does not assert that arbitrary landscapes or all their possible phases must
share one limiting law.
:::


(sec-slcw-population-stability)=
### 16.1. Weighted normalization and full-law stability

:::{prf:definition} Weighted state distance and structural regime
:label: def-slcw-regime

Retain the actual all-alive canonical population map, including its
measurement companions, sampled fitness, simultaneous copying, recipient
jitter and complete collision components. Use the same kinetic parameters
and finite structural profiles

$$
H_c\ge\sup_x|x+\eta F(x)|<\infty,\qquad
\operatorname{Lip}(F)\le L_F<\infty,
$$

with $q,s>0$, as in {prf:ref}`lem-slcc-base-mixing`.
Its explicit reference-kernel coefficient $\epsilon_2>0$ depends only
on these kinetic and force parameters; that calculation does not use a
bounded reward. Replace the bounded-reward assumption by

$$
|R(z)|\le C_r(1+|x|^2),\qquad C_r<\infty.
$$

For the subsequent weak-metric mean-field approximation, also impose
the already stated local Lipschitz reward profile. The weighted full-law
argument below only needs the displayed growth bound and measurability.
All coefficients are fixed independently of $N$.

Let $V_c=(1+2|\alpha_{\rm col}|)V_{\max}$, $\tau^2=c^2q^2+s^2$,
and set

$$
M_4=[BV_c+H_c+\tau\{d(d+2)\}^{1/4}]^4>0,
\quad\beta=\frac{\epsilon_2}{2M_4},\quad
w(z)=1+\beta|x|^4,\quad B_w=1+\beta M_4=1+\epsilon_2/2.
$$

For finite-fourth-moment laws define

$$
\delta_w(\mu,\nu)=\frac12\int w(z)|\mu-\nu|(dz),
\qquad
\mathfrak C=\{\mu:\mu|x|^4\le M_4\}.
$$

The fourth-moment bound is supplied by the exact kinetic position-center
identity, including active copying, rather than assumed as a population
stability hypothesis.
:::

:::{prf:lemma} Complete invariant moment class and weighted kinetic contraction
:label: lem-slcw-weighted-kernel

Under {prf:ref}`def-slcw-regime`, the actual full map sends every
finite-fourth-moment law into $\mathfrak C$. This class is complete for
$\delta_w$. The reference kinetic kernel, restricted to capped inputs,
satisfies

$$
\|\xi K\|_w\le B_w\|\xi\|_w,\qquad
\|\xi K^2\|_w\le q_0\|\xi\|_w,
\qquad q_0=1-\epsilon_2/2<1,
$$

for zero-mass signed measures, with
$\|\xi\|_w=\frac12\int w|\xi|$. The natural extension of $K$ to
collision-prepared velocities also satisfies $Kw\le B_w$ when
$|v|\le V_c$.
:::

:::{prf:proof}
For every actual prepared root $(X,V)$, $|V|\le V_c$ and the completed
position is $X+\eta F(X)+BV+\tau Z$. Its deterministic center has
norm at most $H_c+BV_c$. Minkowski's inequality and
$\mathbb E|Z|^4=d(d+2)$ give the uniform bound $M_4$, independently
of the copied position, component, or activated jitter. The same bound
applies to a reference kinetic step from any capped or collision-
prepared input. Thus $Kw\le B_w$ on these inputs, and $K^2w\le B_w$
on capped inputs.

The space of signed measures with finite $w$-variation is Banach.
Probabilities and the fourth-moment sublevel are closed in that norm:
$1$ and $|x|^4$ are bounded multiples of $w$. Thus $\mathfrak C$ is
complete and is invariant under the full map.
For any signed measure, $|\xi K|\le|\xi|K$ gives
$\|\xi K\|_w\le B_w\|\xi\|_{\rm TV}\le B_w\|\xi\|_w$.
For the second iterate, subtract the common probability in
$K^2\ge\epsilon_2\nu$ from the kinetic minorization. Its integral
against a zero-mass measure vanishes. The remaining nonnegative kernel
has weighted row mass
$K^2w-\epsilon_2\nu w\le B_w-\epsilon_2=q_0$ because $\nu w\ge1$.
Hence $\|\xi K^2\|_w\le q_0\|\xi\|_{\rm TV}\le q_0\|\xi\|_w$.
:::

:::{prf:lemma} Quantitative normalization sensitivity for the unbounded reward
:label: lem-slcw-normalization

For two laws in $\mathfrak C$, put $\delta=\delta_w(\mu,\nu)$ and
use the explicit constants

$$
\begin{aligned}
B_r&=C_r[1+(2\sqrt\beta)^{-1}],\quad
B_{r^2}=2C_r^2\max\{1,\beta^{-1}\},\\
M_r&=C_r(1+\sqrt{M_4}),\qquad
C_{\rm var}=2B_{r^2}+4M_rB_r.
\end{aligned}
$$

Then the reward mean difference is at most $2B_r\delta$ and its
variance difference is at most $C_{\rm var}\delta$.
Use $K_D=1+2/\kappa_D$, the diversity range
$S_b=\sqrt{D_*^2+\delta_D^2}-\delta_D$, and

$$
T_s=K_D[S_b/\sigma_s+S_b^3/(2\sigma_s^3)].
$$

Let $H_r,H_s$ be the explicit bounded logistic-power derivative constants
of {prf:ref}`lem-slcc-selection-perturbation`, with the same configured
fitness exponents. For matched physical/measurement types the two actual
fitness values differ by at most

$$
[h_0+h_1|R(z)|]\delta,
$$

where

$$
h_0=H_r[2B_r/\sigma_r+M_rC_{\rm var}/(2\sigma_r^3)]
       +H_sT_s,\qquad
h_1=H_rC_{\rm var}/(2\sigma_r^3),\quad
\overline C_F=h_0+h_1M_r.
$$
:::

:::{prf:proof}
Writing $y=|x|^2$ gives
$(1+y)/(1+\beta y^2)\le1+(2\sqrt\beta)^{-1}$ and
$(1+y)^2/(1+\beta y^2)\le2\max\{1,\beta^{-1}\}$.
Thus $|R|\le B_rw$ and $R^2\le B_{r^2}w$.
Integration against $|\mu-\nu|$ gives the factor-two mean and raw
second-moment bounds. Each absolute reward mean is at most $M_r$ by
Cauchy--Schwarz. The difference of their squares is at most
$2M_r\cdot2B_r\delta$, giving $C_{\rm var}$.

At a matched physical type the raw reward is identical. Subtract the
two standardized rewards and bound the derivative of
$(v+\sigma_r^2)^{-1/2}$ by $1/(2\sigma_r^3)$.
The numerator has magnitude at most $|R(z)|+M_r$, yielding
$[2B_r/\sigma_r+(|R(z)|+M_r)C_{\rm var}/(2\sigma_r^3)]\delta$.
Since ordinary TV is at most $\delta_w$, the bounded diversity
normalization estimate remains $T_s\delta$. Multiplying by the
actual logistic-power derivative bounds gives $h_0,h_1$.
The fitness values are compared numerically on matched underlying
physical/measurement types; their deterministic fitness marks are not
assumed equal.
:::

:::{prf:lemma} Weighted Lipschitz bound for the complete active selection perturbation
:label: lem-slcw-selection-perturbation

Let $F_*,F^*,a_*,c_*,r_*,L_g$ have their explicit fitness/gate formulas
in {prf:ref}`lem-slcc-selection-perturbation`. In particular
$a_*\le1$, $c_*=a_*/\kappa_C$, $r_*=a_*+c_*$.
Assume $2c_*<1$ and set

$$
L=2c_*K_D+a_*/\kappa_C^2+2L_g\overline C_F/\kappa_C,
\qquad
L_{\rm rem}=B_w\left[2r_*K_D+\frac{2L}{1-2c_*}\right].
$$

For $\mathcal R(\mu)=\mathcal F_h(\mu)-\mu K$ and all
$\mu,\nu\in\mathfrak C$,

$$
\boxed{\qquad
\|\mathcal R(\mu)-\mathcal R(\nu)\|_w
\le L_{\rm rem}\delta_w(\mu,\nu).
\qquad}
$$

All actual component collisions and recipient jitters remain present.
:::

:::{prf:proof}
Couple the underlying physical/measurement root pairs as in the bounded-
reward proof. Their bad mass is at most $K_D\delta$, since ordinary
TV is at most $\delta$. The normalization-sensitivity lemma gives
$h(z)\delta$ on good pairs, where $h(z)=h_0+h_1|R(z)|$.
For a queried good vertex with physical type $z$, subtract donor
normalizers and gates and couple the outgoing accepted measure and
incoming Poisson intensity. In either direction the intensity discrepancy
is at most

$$
L(z)\delta,\qquad
L(z)=2c_*K_D+a_*/\kappa_C^2+
       (L_g/\kappa_C)[h(z)+\overline C_F].
$$

Indeed the bad underlying-type mass costs $2c_*K_D\delta$, the
reciprocal donor normalization costs $a_*\delta/\kappa_C^2$, and
on good paired types the gate difference is at most
$L_g[h(z)+h(z')]\delta$. Integrating the latter over the common
underlying-type coupling costs at most the displayed expression,
because its first marginal is dominated by the first population's
marked law and that law's mean of $h$ is at most $\overline C_F$.
Outgoing completion by no edge and incoming common-Poisson coupling
therefore give total per-query failure hazard at most $2L(z)\delta$.

The first marginal's ordinary complete rooted tree controls the stopped
coupled exploration pathwise. Its expected size is at most
$(1-2c_*)^{-1}$. Moreover, averaged over its root law, its total reward
cost satisfies

$$
\mathbb E\sum_{v\in\mathcal T_\mu}|R(z_v)|
\le\frac{M_r}{1-2c_*}.
$$

To verify this, every newly exposed incoming or outgoing vertex has
expected type measure bounded by $c_*$ times the marked base law.
Thus the expected reward mass in generation $j\ge1$ is at most
$2c_*M_r$ times the expected number of generation-$j-1$ vertices,
which is at most $M_r(2c_*)^j$. The root contributes at most $M_r$.
Sum the series. This is a root-averaged bound; a fixed root with a large
reward contributes that large reward and has no such uniform conditional
bound. Restricting to matched roots or stopping the queried vertices
can only decrease this nonnegative pathwise sum.

Consequently, summing the per-query hazards against the complete-tree
count and reward cost gives mismatch probability at most
$2L\delta/(1-2c_*)$. This calculation does not condition the full tree
on future matching and does not assert that its offspring distribution
is unchanged by previous matching events.

For the weighted variation norm use a dual test $|\varphi|\le w/2$.
The natural kinetic extension obeys
$|K\varphi(z)|\le B_w/2$ at every possible prepared or entering state.
Writing $g=K\varphi$, the signed perturbation equals
$\mathbb E[g(Z^C)-g(Z)]$.
An isolated root contributes zero. On a bad root pair, each increment
has expected absolute value at most $B_wr_*$, giving total bound
$2B_wr_*K_D\delta$. At a good root the baseline $g(Z)$ cancels;
matched full components give identical prepared states, while a mismatch
changes $g$ by at most $B_w$. Multiply the proved mismatch probability
by $B_w$ and add the bad-root contribution. Taking the supremum over
the dual tests proves the theorem.
:::

:::{prf:theorem} Closed full-law convergence with unbounded raw reward
:label: thm-slcw-active-contraction

Use the preceding primitive constants, require $2c_*<1$, and impose the explicit inequality

$$
q_w:=1-\epsilon_2/2+2B_wL_{\rm rem}+L_{\rm rem}^2<1.
$$

Then the actual nonlinear population map has a unique stationary
probability $\pi$ among all its finite-fourth-moment admissible laws.
For every $\mu\in\mathfrak C$,

$$
\delta_w(\mathcal F_h^n\mu,\pi)
\le B_wq_w^{\lfloor n/2\rfloor}.
$$

For any entering law with a finite fourth positional moment,

$$
\boxed{\quad
\|\mathcal F_h^n\mu-\pi\|_{\rm TV}
\le\delta_w(\mathcal F_h^n\mu,\pi)
\le B_wq_w^{\lfloor(n-1)/2\rfloor},\qquad n\ge1.
\quad}
$$

Thus $n=1+2\max\{0,\lceil\log(B_w/\varepsilon)/[-\log q_w]\rceil\}$
updates guarantee error at most $\varepsilon>0$, and physical time is
$nh$. All constants are independent of $N$. The stationary law has the
uniform output moments

$$
\pi|x|^p\le
[BV_c+H_c+\tau m_{d,p}^{1/p}]^p,
\qquad m_{d,p}=2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2),\quad p\ge1.
$$
:::

:::{prf:proof}
The decomposition $\mathcal F_h\mu=\mu K+\mathcal R(\mu)$ and the
proved one-update weighted bounds give Lipschitz coefficient
$B_w+L_{\rm rem}$ on $\mathfrak C$. Expand the second iterate as
in the bounded-reward theorem. Weighted contraction of $K^2$, weighted
boundedness of $K$, and the residual Lipschitz inequality give

$$
\delta_w(\mathcal F_h^2\mu,\mathcal F_h^2\nu)
\le[1-\epsilon_2/2+B_wL_{\rm rem}
                 +L_{\rm rem}(B_w+L_{\rm rem})]\delta_w(\mu,\nu).
$$

This is $q_w$. Completeness and invariance of $\mathfrak C$ give a
unique fixed law for $\mathcal F_h^2$; its image under $\mathcal F_h$
is another such law, hence it is fixed under the original map.
The weighted distance between two laws in $\mathfrak C$ is at most
$B_w$. Iterate the contraction, starting from $\mu$ for even times
and from $\mathcal F_h\mu$ for odd times. Every finite-fourth-moment
law enters $\mathfrak C$ after one update, proving the second bound
and uniqueness on that domain. Uniform root moment bounds follow from
the same bounded deterministic position center and Gaussian radial
moments, regardless of the entering law; invariance transfers them to
$\pi$. No empirical measure is compared with $\pi$ in TV here.
:::

:::{prf:corollary} An explicit positive-selection interval for the unbounded-reward theorem
:label: cor-slcw-positive-exponents

Fix $\bar p_r,\bar p_s\ge0$ with at least one strictly positive, and
set $p_b=\theta\bar p_b$. Let $M,\Delta,D_0,a_0,G_0$ have the
primitive formulas in {prf:ref}`cor-slcc-positive-exponents`, which use
only the fitness bases and floors, not reward boundedness. Define

$$
\begin{aligned}
J_b&=e^M\bar p_bA_b/(4\eta_b),\\
C_{F,0}&=J_r[2B_r/\sigma_r+M_rC_{\rm var}/\sigma_r^3]+J_sT_s,\\
L_0&=2a_0K_D/\kappa_C+a_0/\kappa_C^2
                         +2G_0C_{F,0}/\kappa_C,\\
H_0&=2a_0(1+\kappa_C^{-1})K_D+4L_0,\\
\theta_{\max}^{w}&=
\min\{1,\kappa_C/(4a_0),\epsilon_2/(8B_w^2H_0)\}>0.
\end{aligned}
$$

For every $0<\theta\le\theta_{\max}^{w}$, all the preceding
conditions hold and

$$
q_w\le1-15\epsilon_2/64<1.
$$

Taking both reference exponents positive gives strictly positive reward
and diversity selection. No exponent tends to zero with particle number.
:::

:::{prf:proof}
The same positive-base estimates give $a_*\le\theta a_0$,
$L_g\le G_0$, and $H_b\le\theta J_b$ for $\theta\le1$.
Hence $\overline C_F\le\theta C_{F,0}$ and $L\le\theta L_0$.
The bound $\theta\le\kappa_C/(4a_0)$ gives
$(1-2c_*)^{-1}\le2$. Therefore

$$
L_{\rm rem}\le B_w\theta H_0\le\epsilon_2/(8B_w).
$$

Substitution gives
$2B_wL_{\rm rem}+L_{\rm rem}^2
\le\epsilon_2/4+\epsilon_2^2/(64B_w^2)
\le17\epsilon_2/64$, since $\epsilon_2\le1$ and $B_w\ge1$.
Subtracting this from the gap $\epsilon_2/2$ proves the stated bound.
Every denominator is strictly positive and all constants are finite,
so the interval is explicitly nonempty.
:::

:::{prf:corollary} Compatibility with one potential supplying both force and raw reward
:label: cor-slcw-same-potential

Suppose $U\in C^1$, the actual force is $F=-\nabla U$, the raw reward
is $R=-U$, and the same finite profiles $H_c,L_F$ hold. Put
$g_0=H_c/\eta$, $\lambda_0=1/\eta$ and choose

$$
C_r=|U(0)|+g_0+\lambda_0/2.
$$

Then $|R(x)|\le C_r(1+|x|^2)$, and its reward increment bound on a
radius-$L$ ball is $g_0+\lambda_0L$. Thus the preceding theorem and
its positive-exponent interval apply to this unbounded raw reward
without clipping it, separating it from the potential, or changing the
actual algorithm. The force-center profile remains a substantive
structural restriction; no claim is made that every landscape satisfies
it.
:::

:::{prf:proof}
The profile gives $F(x)=-\lambda_0x+f(x)$ with $|f(x)|\le g_0$.
Integrate $\nabla U=-F$ along the segment from zero to $x$:

$$
U(x)=U(0)+\lambda_0|x|^2/2
       -\int_0^1 f(tx)\cdot x\,dt.
$$

Hence $|U(x)|\le|U(0)|+\lambda_0|x|^2/2+g_0|x|$.
Using $|x|\le(1+|x|^2)/2$ proves the displayed conservative $C_r$.
The gradient norm is at most $g_0+\lambda_0|x|$; integration along
segments in the convex ball gives its local Lipschitz reward bound.
All required quantities are therefore supplied by the same potential.
:::

(sec-slcw-transfer)=
### 16.2. Vanishing particle errors, stationary laws and both limit orders

:::{prf:theorem} Uniform-time transfer from weighted contraction and explicit higher moments
:label: thm-slcw-transfer

Use {prf:ref}`thm-slcw-active-contraction`, retaining its actual raw
quadratic-growth reward and active cloning. In particular
$$
 |x+\eta F(x)|\le H_c,\quad
 w(x,v)=1+\beta_w|x|^4,\quad
 \beta_w=\epsilon_2/(2M_4),\quad B_w=1+\beta_wM_4,
$$
$$
 q_w=1-\epsilon_2/2+2B_wL_{\rm rem}+L_{\rm rem}^2\in(0,1).
$$
The constants $\epsilon_2,L_{\rm rem}$ are the derived quantities of
that theorem. Retain the force and raw-reward regularity hypotheses of
{prf:ref}`thm-slct-quadratic-reward` and its explicit coefficient
$C_*=C_8(1)$; thus its weak modulus on eighth-moment classes is
$C_*\max(1,H)^{3/2}\delta^{1/32}$. There is no bounded-reward replacement.
Let $\pi$ be the unique stationary population law proved by the weighted
contraction theorem. Set
$$
 b_0=BV_c+H_c,\quad\tau^2=c^2q^2+s^2,\quad
 m_{d,p}=2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)},
$$
$$
 M_p=(b_0+\tau m_{d,p}^{1/p})^p\quad(p=4,8,24),
 \qquad M_2=b_0^2+d\tau^2,
$$
$$
 C_{\rm eff}=C_*\sqrt{1+M_{24}+M_8^3}.
$$
For even $p$, $m_{d,p}=\prod_{j=0}^{p/2-1}(d+2j)$, so in particular
all these moment constants are finite explicit polynomials and powers
of the displayed algorithm and force-center parameters.

For $N\ge2$, define
$$
 J_0=(2+2\sqrt{2d})^d(2+2V_{\max}\sqrt{2d})^d,\quad
 A_0=2+\tfrac12J_0\sqrt{A+4B_*^2}+2M_2,
$$
$$
 a_N=\min\{1,A_0N^{-1/(16d)}\},\quad
 b_N=1+\left\lfloor\frac{\log\log(N+e)}{2\log32}\right\rfloor,
$$
$$
 V_N=\min\{1,(1+C_{\rm eff})^{32/31}
                      a_N^{32^{-(b_N-1)}}\},\quad
 \varepsilon_N=V_N+B_wq_w^{\lfloor(b_N-1)/2\rfloor}.
$$
Then $\varepsilon_N\to0$, and every finite-particle initialization obeys
$$
 \boxed{\quad
 \mathbb E\mathsf d(L_N(S_n),\pi)
 \le\min\{1,B_wq_w^{\lfloor(n-1)/2\rfloor}+\varepsilon_N\},
 \qquad n\ge1.\quad}
$$
No expected initial moment is imposed in this statement. The actual
population law, for every admissible finite-fourth-moment input, obeys
$$
 \|\mathcal F_h^n\mu-\pi\|_{\rm TV}
 \le B_wq_w^{\lfloor(n-1)/2\rfloor},\qquad n\ge1.
$$
The time rate and coefficient are independent of $N$. Every stationary
finite-particle law has expected empirical distance to $\pi$ at most
$\varepsilon_N$, and every joint diagonal $n_N\to\infty$ converges
to $\pi$ in expectation and probability.
:::

:::{prf:proof}
The completed position is
$X^+=(X^C+\eta F(X^C))+BV^C+cq\xi+s\zeta$. Conditional on
preparation, its mean has norm at most $b_0$ and its noise is centered
Gaussian with covariance $\tau^2I_d$. The exact second-moment formula
gives $M_2$, while Minkowski gives $\mathbb E|X^+|^p\le M_p$.
These estimates hold uniformly over every entering configuration,
cloning graph, collision and jitter; the same argument applies to the
population root. Therefore every actual output empirical law has
expected $p$th positional moment at most $M_p$, and every population
output law has $p$th moment at most $M_p$, even conditional on its input.

Suppose $\widehat\mu$ is such an empirical output and $\nu$ is a
possibly random population output, with any dependence between them.
Write $\Delta=\mathsf d(\widehat\mu,\nu)$ and
$H=\max\{1,\widehat\mu|x|^8,M_8\}$. Pointwise the structural modulus
applies with this value of $H$. Jensen within the empirical measure gives
$(\widehat\mu|x|^8)^3\le\widehat\mu|x|^{24}$, hence
$$
 \mathbb EH^3\le1+M_{24}+M_8^3.
$$
Cauchy–Schwarz, followed by concavity of $t^{1/16}$, proves
$$
 \mathbb E\mathsf d(\mathcal F_h\widehat\mu,\mathcal F_h\nu)
 \le C_*\sqrt{\mathbb EH^3}\sqrt{\mathbb E\Delta^{1/16}}
 \le C_{\rm eff}(\mathbb E\Delta)^{1/32}.
$$
No independence between $H$ and $\Delta$ is used.

The actual scalar one-step consistency constant is $G=A+4B_*^2$.
Use the existing cell estimate with radius $N^{1/(16d)}$ and mesh
$N^{-1/(16d)}$. The uniform output moment bound gives
$$
 \mathbb E\mathsf d(L_N(S_{j+1}),\mathcal F_hL_N(S_j))
 \le\min\{1,2N^{-1/(16d)}+\tfrac12J_0\sqrt G N^{-5/16}
                              +2M_2N^{-1/(8d)}\}\le a_N,
$$
where clipping follows from the metric diameter one.

For an observation index $n$, restart a population trajectory at the
actual empirical law at time $n-m$, $m=\min(n,b_N)$. Its initial
comparison error is zero, so the first step has error at most $a_N$
without requiring an initial moment. At all subsequent steps both inputs
are actual outputs or population outputs; the averaged modulus just
proved applies. Consequently expected comparison errors satisfy
$$
 v_1\le a_N,\qquad
 v_{j+1}\le\min\{1,a_N+C_{\rm eff}v_j^{1/32}\}.
$$
Induction gives
$$
 v_j\le(1+C_{\rm eff})^{\sum_{i=0}^{j-2}32^{-i}}
                         a_N^{32^{-(j-1)}}
 \le(1+C_{\rm eff})^{32/31}a_N^{32^{-(j-1)}},
$$
so $v_m\le V_N$. The weighted contraction theorem applies to every
realized empirical restart input, which has finite fourth moment, and
gives its population distance to $\pi$ at most
$B_wq_w^{\lfloor(m-1)/2\rfloor}$. This term is bounded by the sum
of its values at $n$ and at $b_N$, proving the boxed estimate.

For $L=\log(N+e)$, the block definition gives
$32^{-(b_N-1)}\ge L^{-1/2}$. Since
$\log a_N=\log A_0-\log N/(16d)<0$ for sufficiently large $N$,
$$
 \log[(1+C_{\rm eff})^{32/31}a_N^{32^{-(b_N-1)}}]
 \le\frac{32}{31}\log(1+C_{\rm eff})+
 L^{-1/2}\left(\log A_0-\frac{\log N}{16d}\right)
 \longrightarrow-\infty.
$$
Also $b_N\to\infty$, so $\varepsilon_N\to0$. Under an invariant
particle law the empirical expected distance is independent of time;
let $n\to\infty$ to obtain the stationary bound. The diagonal result
and probability version follow directly and by Markov's inequality.
:::

:::{prf:corollary} Fully explicit confidence, population and time choices
:label: cor-slcw-sizes

Set $C=(1+C_{\rm eff})^{32/31}$ and
$r=-\log q_w/(4\log32)>0$. For $0<\zeta<1$, let
$$
 L_*(\zeta)=\max\left\{32d\log A_0+2,
 [32d\log(2C/\zeta)]^2,
 [2B_w/(q_w\zeta)]^{1/r}\right\}.
$$
Then $N\ge\lceil\exp L_*(\zeta)\rceil$ implies
$\varepsilon_N\le\zeta$. For distance tolerance $\delta\in(0,1)$
and failure probability $\alpha\in(0,1)$, sufficient choices are
$$
 N\ge\left\lceil\exp L_*(\alpha\delta/2)\right\rceil,
 \qquad
 n\ge1+2\left\lceil
       \frac{\log(2B_w/(\alpha\delta))}{-\log q_w}
       \right\rceil.
$$
They ensure $\Pr\{\mathsf d(L_N(S_n),\pi)>\delta\}\le\alpha$.
Physical time is $nh$ and the number of row updates is $Nn$.
:::

:::{prf:proof}
For $N\ge2$, $\log(N+e)-\log N<1$. If $L\ge32d\log A_0+2$,
then $\log a_N\le-L/(32d)$ and
$V_N\le C\exp[-\sqrt L/(32d)]$. Moreover
$$
 \lfloor(b_N-1)/2\rfloor\ge\frac{\log L}{4\log32}-1,
 \quad B_wq_w^{\lfloor(b_N-1)/2\rfloor}
                  \le B_wq_w^{-1}L^{-r}.
$$
The other two terms in $L_*$ bound these contributions by $\zeta/2$
each. The observation count bounds the time contribution by
$\alpha\delta/2$; the selected particle floor is at most the same
quantity. Markov's inequality proves the result.
:::

:::{prf:corollary} Uniform-time trajectories, stationary chaos and commuting limits
:label: cor-slcw-trajectories

For initialized trajectory comparison assume the explicit moment budget
$$
 \sup_N\mathbb E L_N(S_0)|x|^8\le H_8,\qquad
 \mu_0|x|^8\le H_8,
 \quad d_{N,0}=\mathbb E\mathsf d(L_N(S_0),\mu_0)\to0.
$$
Set $\overline M=\max\{1,H_8,M_8\}$ and
$C_{\rm init}=C_*\overline M^{3/2}+1$. Define the computable recursion
$$
 u_0=d_{N,0},\qquad
 u_{j+1}=\min\{1,a_N+C_{\rm init}u_j^{1/80}\}.
$$
For any integer $T\ge1$,
$$
 \sup_{n\ge0}\mathbb E\mathsf d(L_N(S_n),\mathcal F_h^n\mu_0)
 \le\max\left\{\max_{0\le j<T}u_j,
        2B_wq_w^{\lfloor(T-1)/2\rfloor}+\varepsilon_N\right\}.
$$
This proves uniform-time initialized mean-field convergence by choosing
$T$ and then $N$ large. Mere weak consistency of unbounded-reward input
laws is not substituted for the displayed moment control.

The finite-particle theorem {prf:ref}`thm-slca-active-stationary` applies
with $\lambda=1/\eta$, $g_0=H_c/\eta$, since
$F(x)=-x/\eta+(x+\eta F(x))/\eta$ and the residual is bounded.
Thus the actual invariant law $\Pi_N$ is unique and exchangeable, and
$$
 \Pi_N\circ L_N^{-1}\Longrightarrow\delta_\pi
$$
along the full sequence. For $1\le k\le N$, its $k$-row stationary
marginal has error at most $k\varepsilon_N+k(k-1)/(2N)$ in the
transport metric with cost
$\min\{1,\sum_i\min(1,|z_i-z_i'|)\}$.
Both iterated long-time/population-size limits and every joint diagonal
agree in these weak empirical-law metrics. With exchangeable
initializations the same holds for fixed-row marginals. Whole-swarm
TV convergence at fixed $N$ may retain an $N$-dependent rate.
:::

:::{prf:proof}
Both trajectories have expected or deterministic eighth moment at most
$\overline M$, including at initialization. If their current expected
weak distance is $u\in(0,1]$, choose
$H=\overline M u^{-1/80}$. The deterministic population eighth moment
is below $H$, and the empirical threshold fails with probability at most
$\overline M/H=u^{1/80}$. On the good event, the modulus and Jensen give
$$
 \mathbb E[\mathbf1_{\{W_8\le H\}}
       \mathsf d(\mathcal F_h\widehat\mu,\mathcal F_h\nu)]
 \le C_*H^{3/2}u^{1/32}
 =C_*\overline M^{3/2}u^{1/80}.
$$
Adding the bad-event probability and the output sampling defect $a_N$
proves the displayed recursion. If $u=0$, the inputs coincide almost
surely, so the propagated error is zero directly.
Every fixed recursion value tends to zero. For $n\ge T$,
insert $\pi$ and add the particle and population attraction bounds,
which proves the uniform-time inequality.

The finite-particle stationary theorem gives existence, uniqueness and
fixed-$N$ whole-swarm TV convergence. Every invariant law inherits all
output moment bounds. The theorem's stationary estimate and Markov give
convergence of empirical laws to $\delta_\pi$. Permutation equivariance
and uniqueness imply exchangeability. Sampling $k$ labels without
replacement differs from sampling with replacement with probability at
most $k(k-1)/(2N)$; conditionally the latter has law
$L_N(S)^{\otimes k}$. Coordinatewise coupling costs at most
$k\mathsf d(L_N(S),\pi)$, giving the stated marginal error.
At fixed time, the initialized mean-field limit is
$\mathcal F_h^n\mu_0$, whose limit is $\pi$; in the reverse order the
finite-$N$ stationary laws have the full-sequence limit just proved.
The uniform bounds give every diagonal. No population-independent
whole-swarm TV mixing rate is asserted.
:::

(sec-slcw-alive-wasserstein)=
### 16.3. Population-uniform convergence of the actual alive law in Wasserstein distance

:::{div} feynman-prose
There are two sources of error when we compare a finite swarm with the
stationary population law. The entering law is forgotten at a rate independent
of $N$, while a finite sample still fluctuates around the population law. The
theorem keeps both terms: the time term decays, and the sampling floor vanishes
as $N$ grows. It controls the random empirical measure as well as the law of
a uniformly sampled alive walker, with the unbounded raw reward retained.

This conclusion uses the proved global force-center bound and the explicit
weak, positive selection tests. Measurements use the current frame, kinetics
is nonviscous, and deaths are disabled. Equal-fitness draws remain in the
calculation; no positive fitness-variance floor is required. These hypotheses
certify relaxation in law, rather than monotone motion of every pair of
sampled swarms. They do not certify the original dense-viscous reference or
its survival-conditioned QSD.
:::

:::{prf:lemma} Fourth-moment upgrade of bounded transport
:label: lem-slcw-physical-transport

Let $E\subset\mathbb R^{2d}$, $c_0(z,z')=\min\{1,|z-z'|\}$,
and let $\mathsf d$ be its transport metric. Let $G$ be a fixed positive
definite matrix and let $W_{2,G}$ have squared cost
$(z-z')^TG(z-z')$. For random probability measures $\widehat\mu,\widehat\nu$
with
$$
 \mathbb E\widehat\mu|z|^4\le K_4,\qquad
 \mathbb E\widehat\nu|z|^4\le K_4,
 \qquad e=\mathbb E\mathsf d(\widehat\mu,\widehat\nu),
$$
one has
$$
 \mathbb EW_{2,G}^2(\widehat\mu,\widehat\nu)
 \le\lambda_{\max}(G)[e+4\sqrt{K_4}\sqrt e]
 \le C_G\sqrt e,
 \qquad C_G=\lambda_{\max}(G)(1+4\sqrt{K_4}).
$$
The measures may depend on one another; no independence is required.
:::

:::{prf:proof}
Condition on the two measures and choose a transport plan for $c_0$
with cost within an arbitrary $\epsilon>0$ of the infimum. Such plans
can be chosen measurably, or first chosen for simple random measures
and then approximated. Write $r=|z-z'|$ under this plan. On $r\le1$,
$r^2\le r=c_0(z,z')$. On $r>1$, Cauchy--Schwarz gives
$$
 \int r^2\mathbf1_{\{r>1\}}\,d\Gamma
 \le\left(\int r^4\,d\Gamma\right)^{1/2}
       \Gamma\{r>1\}^{1/2}
 \le[8(\widehat\mu|z|^4+\widehat\nu|z|^4)]^{1/2}
       (\mathsf d(\widehat\mu,\widehat\nu)+\epsilon)^{1/2}.
$$
Here $|z-z'|^4\le8(|z|^4+|z'|^4)$ and the mass of $r>1$
is at most the $c_0$ cost. The same plan is an admissible competitor
for $W_{2,G}$, whose cost is at most $\lambda_{\max}(G)\int r^2d\Gamma$.
Average and apply Cauchy--Schwarz once more to the two random factors.
The averaged fourth-moment sum is at most $2K_4$. Let $\epsilon\downarrow0$.
This proves the first inequality. Since $0\le e\le1$, $e\le\sqrt e$,
which proves the second. The argument only uses marginal moment bounds.
:::

:::{prf:theorem} Completed population-uniform alive-law estimate
:label: thm-slcw-alive-uniform-law

Use the conservative, death-disabled canonical gas and the primitive
regime of {prf:ref}`thm-slcw-transfer`: the actual current-frame
measurement and sampled global standardization, Gaussian donor law,
simultaneous copying, recipient jitter, shared Haar rotation of every
accepted component, full nonviscous BAOAB update and final velocity cap.
The raw reward may be unbounded and supplied by the same potential as
the force, as in {prf:ref}`cor-slcw-same-potential`. Require the verified
force-center profile and the explicit selection inequalities
$$
 H_c=\sup_x|x+\eta F(x)|<\infty,\qquad L_F<\infty,
 \qquad 2c_*<1,\qquad
 q_w=1-\epsilon_2/2+2B_wL_{\rm rem}+L_{\rm rem}^2<1.
$$
The constants are the displayed primitive formulas of Section 16.1;
{prf:ref}`cor-slcw-positive-exponents` supplies an explicit nonempty
interval of fixed, positive selection exponents satisfying them.
There is no lower bound on realized fitness variance or on accepted
cloning pressure. All parameters are independent of $N$.

Let $\pi$ be the proved stationary population probability, let $S_n^N$
be the actual finite swarm, and define the alive empirical measure and
its uniformly alive-sampled law by
$$
 \mu_{N,n}^a=\frac1N\sum_{i=1}^N\delta_{(X_{i,n},V_{i,n})},
 \qquad \lambda_{N,n}^a=\mathbb E\mu_{N,n}^a.
$$
Every row is alive in this declared regime. For $n\ge1$, put
$$
 r_{N,n}=\min\{1,B_wq_w^{\lfloor(n-1)/2\rfloor}+\varepsilon_N\},
 \qquad
 K_4=(\sqrt{M_4}+V_{\max}^2)^2,
 \qquad C_G=\lambda_{\max}(G)(1+4\sqrt{K_4}),
$$
where $M_4$ and the explicit vanishing $\varepsilon_N$ are exactly
those of {prf:ref}`thm-slcw-transfer`. Then, for every $N\ge2$ and
every finite entering configuration or distribution of such configurations,
$$
 \boxed{\quad
 \mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi)
 \le C_G\sqrt{r_{N,n}},\qquad
 W_{2,G}^2(\lambda_{N,n}^a,\pi)
 \le C_G\sqrt{r_{N,n}}.\quad}
$$
In particular a separated time-rate and population-error bound is
$$
 \mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi)
 \le C_G\sqrt{B_w}\,
            q_w^{\lfloor(n-1)/2\rfloor/2}
       +C_G\sqrt{\varepsilon_N},
 \qquad \varepsilon_N\longrightarrow0.
$$
The time coefficient and rate are independent of population size.
This also bounds the squared Wasserstein distance between the law
of the random alive empirical measure and the point mass $\delta_\pi$,
when the ground metric on empirical measures is $W_{2,G}$.

For any $\delta>0$ and $\alpha\in(0,1)$, set
$$
 e_* =\min\{1,(\alpha\delta^2/C_G)^2\}.
$$
The explicit population choice of {prf:ref}`cor-slcw-sizes` with
$\zeta=e_*/2$, and the observation count
$$
 n\ge1+2\left\lceil
       \frac{\log(2B_w/e_*)}{-\log q_w}\right\rceil,
$$
give
$$
 \Pr\{W_{2,G}(\mu_{N,n}^a,\pi)>\delta\}\le\alpha.
$$
The corresponding physical observation time is $nh$.
:::

:::{prf:proof}
The complete-update fourth positional moment is at most $M_4$ in
expectation for the actual empirical law, and deterministically for $\pi$,
by {prf:ref}`thm-slcw-transfer` and stationarity. The final velocity cap
gives $|v|\le V_{\max}$. Expanding $|z|^4=(|x|^2+|v|^2)^2$ and
using Cauchy--Schwarz gives
$$
 \mathbb E\mu_{N,n}^a|z|^4
 \le M_4+2V_{\max}^2\sqrt{M_4}+V_{\max}^4=K_4,
 \qquad \pi|z|^4\le K_4.
$$
No initial moment bound is needed: the resonant force-center estimate
bounds every complete output regardless of the copied and jittered
entering positions. The proved finite-particle transfer gives
$\mathbb E\mathsf d(\mu_{N,n}^a,\pi)\le r_{N,n}$.
Apply {prf:ref}`lem-slcw-physical-transport` to obtain the first
boxed estimate. To obtain the second, average transport plans between
each realized empirical measure and $\pi$. The averaged plan has
marginals $\lambda_{N,n}^a$ and $\pi$, and its expected cost is the
first bound. Approximation by plans within $\epsilon$ of their optimum
and then $\epsilon\downarrow0$ removes any attainment issue.

A transport plan from the random-measure law to $\delta_\pi$ has only
one possible second coordinate. Its squared cost is therefore exactly
$\mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi)$. This proves the assertion for
the law of the empirical measure, rather than only its average.
The inequality $\sqrt{u+v}\le\sqrt u+\sqrt v$ gives the separated
time term and particle floor. The chosen population and time make
$r_{N,n}\le e_*$. Hence the expected squared cost is at most
$C_G\sqrt{e_*}\le\alpha\delta^2$, and Markov's inequality proves
the confidence estimate. All moment, consistency, modulus and mixing
constants were derived from this same complete transition; none depends
on $N$. The finite-particle floor accounts for sampling fluctuations
and is not discarded.
:::

:::{prf:corollary} Stationary alive laws and comparison of differently initialized swarms
:label: cor-slcw-alive-stationary-comparison

In the preceding regime let $\Pi_N$ be the actual finite-swarm invariant
law and $\lambda_N^{a,*}=\int\mu_S^a\,\Pi_N(dS)$. Then
$$
 \int W_{2,G}^2(\mu_S^a,\pi)\,\Pi_N(dS)
 \le C_G\sqrt{\varepsilon_N},\qquad
 W_{2,G}^2(\lambda_N^{a,*},\pi)\le C_G\sqrt{\varepsilon_N}.
$$
For any two initializations at the same $N$, with possibly different
and even independently run swarms $S_n^N,T_n^N$,
$$
 \mathbb EW_{2,G}^2(\mu_{S_n^N}^a,\mu_{T_n^N}^a)
 \le4C_G\sqrt{r_{N,n}}.
$$
The time-then-population limit and every joint diagonal with
$N,n\to\infty$ yield $\delta_\pi$ for the empirical-measure laws
in this Wasserstein metric. For arbitrary initializations the bound also gives
$$
 \lim_{n\to\infty}\limsup_{N\to\infty}
 \mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi)=0.
$$
Under the consistent initialization and moment hypotheses of
{prf:ref}`cor-slcw-trajectories`, the population-then-time iterated
limit exists and has the same value. This eventual comparison does not
assert that discrepancies decrease at every step.
:::

:::{prf:proof}
Every finite-swarm invariant law inherits the same fourth moment bound
and has expected bounded-transport error at most $\varepsilon_N$ by
{prf:ref}`cor-slcw-trajectories`. Apply the moment-upgrade lemma and
average optimal plans to obtain the stationary assertions. For the
two swarms, the metric triangle inequality and $(a+b)^2\le2a^2+2b^2$
give the displayed comparison by inserting $\pi$ and applying the two
individual bounds. This does not impose any coupling on their actual
random updates. The time-then-population limit follows from finite-$N$
convergence to $\Pi_N$ and the stationary bound. Under the stated
initialized trajectory hypotheses, the finite-horizon mean-field limits in
{prf:ref}`cor-slcw-trajectories` give the opposite order. Uniform fourth
moments upgrade both weak limits to the asserted Wasserstein limits.
The separated estimate gives the unconditional double-limsup statement
and every joint diagonal directly.
:::

(sec-slcw-finite-uniform)=
### 16.4. Exact finite-swarm relaxation with a population-independent rate

:::{div} feynman-prose
Now compare each finite swarm with its own stationary law. The bounded-reward
regime gives an exact rate independent of $N$, with no sampling floor: the
alive-sampled law converges in total variation, and the random empirical
measure law converges in Wasserstein distance to its finite-swarm stationary
counterpart.

The proof couples the actual accepted graph components, bounds how far a
preparation mismatch spreads, and uses two-update Gaussian endpoint bridges
to erase a definite fraction of mismatches. The explicit weak-selection test
makes that smoothing dominate the component propagation. The bridges retain
each swarm's prescribed Gaussian law, and cloning remains active when fitness
differences permit it. Complete ties need no positive variance bound.

The bounded force-center, current-frame, nonviscous, death-disabled assumptions
remain essential to this result. It proves relaxation in law; it does not
assert that every observed swarm pair moves closer at each update, or certify
the dense-viscous reference and its conditioned QSD.
:::

:::{prf:lemma} Finite-swarm preparation coupling through the actual global normalizers
:label: lem-slcw-finite-preparation

Use the conservative current-frame nonviscous canonical gas of
{prf:ref}`def-slcc-regime`, with bounded raw reward of oscillation
$R_{\rm osc}$. Retain the finite sampled measurement array and its
actual empirical means and regularized variances. Use $a_*,c_*,L_g,H_b$
from {prf:ref}`lem-slcc-selection-perturbation`, and require $a_*\le1/4$.
Set
$$
 \begin{aligned}
 K_D^f&=1+4/\kappa_D,\qquad
 T_s^f=S_b/\sigma_s+S_b^3/(2\sigma_s^3),\\
 C_r^f&=H_rT_r,\qquad C_s^f=H_sT_s^f,\\
 L_f&=4c_*+(4c_*+2a_*)K_D^f
             +2L_g(C_r^f+K_D^f C_s^f),\\
 M_f&=e^{8c_*},\qquad
 e_1=M_f-1+3M_fL_f,\qquad e_2=e_1+6c_*.
 \end{aligned}
$$
For two entering arrays let $A$ be the labels whose physical states differ
and $r=|A|/N$. The actual complete preparations can be coupled so that
their output mismatch fraction has expectation at most $(1+e_1)r$.
If synchronous reference trajectories agree outside a fixed set $A$,
the second preparations can be coupled so that the expected fraction
of labels where those preparations can destroy an endpoint agreement
is at most $e_2|A|/N$. In the latter statement include every label of
$A$ incident to an accepted edge in either swarm.
All constants apply to every $N\ge2$.
:::

:::{prf:proof}
At a matched physical recipient, at most $|A|$ companion weights can
change. Each actual measurement denominator is at least
$(N-1)\kappa_D$. Maximally couple the companion draws and count a
matched index in $A$ as a bad physical companion. Subtracting weighted
numerators and their normalizers bounds the probability of a bad
measurement at a matched recipient by $4r/\kappa_D$, since
$N/(N-1)\le2$. Couple these draws independently across recipients.
If $m$ is the resulting fraction of bad physical/measurement types,
then $\mathbb Em\le K_D^fr$.

For two scalar arrays in a range of length $R$ differing at a fraction
$u$ of indices, their empirical means differ by at most $Ru$.
Their variances differ by at most $R^2u$: write variance as half the
average squared difference over two independently selected indices;
at most $2u$ of the ordered pairs can change. The derivative of
$(v+\sigma^2)^{-1/2}$ is at most $1/(2\sigma^3)$.
At a good physical/measurement type these facts, followed by the
actual logistic-power derivatives, give
$$
 |F_i(S)-F_i(T)|\le C_r^fr+C_s^fm.
$$
This is a comparison of the two computed fitness numbers, including
their different global random normalizers.

Freeze both complete measurement arrays. An outgoing token records
either its accepted donor label or no edge. Different rows draw their
tokens independently. Couple each token maximally, independently
across rows, and let $H$ be the set of unequal tokens. Every row's
failure probability $q_i$ is at most $2a_*$, because a failure requires
an accepted edge in at least one swarm. For a good recipient,
subtract its two normalized donor laws and then its gates. Changed
physical donor weights cost at most $4c_*r$; bad donor types cost at
most $4c_*m$, using their per-label accepted probability
$c_* /(N-1)$. On the remaining donor types the gate difference is
at most $2L_g(C_r^fr+C_s^fm)$, and their normalized donor masses sum
to at most one. Bad recipient types cost at most $2a_*$.
Consequently
$$
 \mathbb E\frac{|H|}{N}\le
 [4c_*+(4c_*+2a_*)K_D^f
                 +2L_g(C_r^f+K_D^fC_s^f)]r=L_fr.
$$

Condition on the measurements, $H$, and both token outcomes of its rows.
Remove the outgoing edges of rows in $H$. Every remaining row has one
common token; the remaining rows are independent under this conditioning.
An accepted common edge has conditional probability at most
$$
 \frac{c_*}{(N-1)(1-q_i)}\le\frac{2c_*}{N-1}\le\frac{4c_*}{N}.
$$
Every common edge strictly increases the first swarm's frozen fitness,
so this common graph is an ordered forest. The increasing-path and
decreasing-path enumeration in {prf:ref}`lem-chaos-component-truncation`,
with $C=4c_*$, bounds the expected size of the common component
of any specified label by $M_f=e^{8c_*}$. This estimate also applies
with the outgoing rows $H$ removed and to endpoint labels fixed by the
conditioning. It does not multiply probabilities of two correlated
edges from a coupled exceptional row.

Restore the exceptional edges. Every affected common component touches
either an entering physical mismatch or one of at most three endpoint
seeds per exceptional row: the recipient and its two possible donors.
Components meeting none of these seeds coincide, have identical physical
inputs, and may use the same independent Haar rotation and recipient
jitters. Their prepared states agree exactly. A union bound over common
components therefore gives mismatch fraction at most
$M_f(r+3\mathbb E|H|/N)\le(1+e_1)r$.

For the second claim, outside $A$ the entire reference paths and innovations
agree. Common components reaching outside $A$ from it cost at most
$(M_f-1)|A|/N$. Exceptional endpoint components cost at most
$3M_fL_f|A|/N$, since the intermediate mismatch set is a subset of $A$.
Finally, a given label of $A$ has an outgoing accepted edge with
probability at most $a_*$ in each swarm. Its incoming mean in each
swarm is at most $Nc_* /(N-1)\le2c_*$. The probability that it
is incident to any accepted edge in either swarm is therefore at most
$2a_*+4c_*\le6c_*$. These events are charged even if a reference
endpoint reset would have made that label agree. Adding the three
charges proves $e_2|A|/N$. All expectations use the actual finite
forest, rather than a replacement Poisson graph.
:::

:::{prf:theorem} Exact population-uniform convergence to the finite-swarm invariant law
:label: thm-slcw-finite-uniform-law

Use the primitive bounded-reward regime of
{prf:ref}`lem-slcw-finite-preparation`, continuous reward, $q,s>0$,
and the bounded force-center profile $H_c$ with globally Lipschitz force.
Let $K$ be the reference kinetic kernel. Recompute the constants of
{prf:ref}`lem-slcc-base-mixing` using
$$
 b_K'=1+(BV_c+H_c)^2+d\tau^2,\qquad
 R_0'=\sqrt{4b_K'-1}
$$
for its first update, and retain $V_{\max}$ in the local second-update
constants $R_1,m_v,Q,k_v,k_x$. Denote the resulting strictly positive
two-update minorization coefficient by $\epsilon_f$.
It applies to every possible collision-prepared input with $|v|\le V_c$.
Require the explicit strict inequality
$$
 \boxed{\quad q_f=(1-\epsilon_f+e_2)(1+e_1)<1.\quad}
$$
For labeled arrays define the normalized Hamming metric
$$
 \rho_N(S,T)=\frac1N\sum_{i=1}^N\mathbf1_{\{z_i\ne z_i'\}},
$$
and let $\mathcal W_{\rho_N}$ be its transport metric on array laws.
The actual conservative swarm kernel $P_N$ has its unique invariant law
$\Pi_N$, and for every entering law $\Lambda_N$,
$$
 \mathcal W_{\rho_N}(\Lambda_NP_N^n,\Pi_N)
 \le q_f^{\lfloor n/2\rfloor},\qquad N\ge2,\ n\ge0.
$$
Let $\Xi_{N,n}$ and $\Xi_N^*$ be the laws of the alive empirical
measure under $\Lambda_NP_N^n$ and $\Pi_N$, and let
$\lambda_{N,n}^a,\lambda_N^{a,*}$ be their mean alive measures.
Sampling a uniform alive label gives the exact population-uniform law bound
$$
 \boxed{\quad
 \|\lambda_{N,n}^a-\lambda_N^{a,*}\|_{\rm TV}
 \le q_f^{\lfloor n/2\rfloor},\qquad N\ge2,\ n\ge0.\quad}
$$
For a fixed positive definite $G$, put
$$
 K_4=(\sqrt{M_4}+V_{\max}^2)^2,\qquad
 C_f=4\lambda_{\max}(G)\sqrt{K_4},
$$
where $M_4=[BV_c+H_c+\tau\{d(d+2)\}^{1/4}]^4$.
Then for $n\ge1$,
$$
 \boxed{\quad
 \mathcal W_{W_{2,G}}^2(\Xi_{N,n},\Xi_N^*)
 \le C_f q_f^{\lfloor n/2\rfloor/2},\qquad
 W_{2,G}^2(\lambda_{N,n}^a,\lambda_N^{a,*})
 \le C_f q_f^{\lfloor n/2\rfloor/2}.\quad}
$$
Here $\mathcal W_{W_{2,G}}$ is Wasserstein distance on laws of
empirical measures, with ground metric $W_{2,G}$. The coefficient and
rate are independent of $N$. These are exact relaxation bounds to the
finite-swarm invariant law; they have no finite-particle error floor.
:::

:::{prf:proof}
**Two-update coupling with exact individual marginals.** Couple the first
actual preparations by the preceding lemma. Let $A_1$ be their physical
mismatch labels; $\mathbb E|A_1|/N\le(1+e_1)\rho_N(S,T)$.
From these prepared states, couple the reference two-update kinetic
endpoints independently across labels using the common minorization
$K^2(z,\cdot)\ge\epsilon_f\nu$. On equal prepared states use identical
entire reference trajectories. On unequal ones, couple endpoints with
disagreement probability at most $1-\epsilon_f$, and sample the two
innovation paths conditionally on their own starting states and endpoints.
Regular conditional distributions exist for these finite-dimensional
Borel Gaussian innovation spaces. Each swarm's full reference innovation
path has exactly its original product-Gaussian marginal.

The first reference update is the actual first kinetic update. Now,
conditional on both entire paired reference paths, couple the actual
second preparations with each own marginal exactly its prescribed
preparation kernel evaluated at its own intermediate array. Couple
component rotations and jitters with their prescribed own distributions.
This pointwise marginal identity ensures that each own second preparation,
given its own intermediate state and earlier history, is independent of
its own future Gaussian innovations. Apply those future innovations to
the actual second prepared state. This realizes the actual second kinetic
update in each swarm, even though the joint coupling used the reference
endpoints in advance. In particular no second preparation is conditioned
on a favorable future noise event in its individual marginal.

Outside $A_1$ the entire reference paths agree. A second component with
equal inputs and no exceptional edges can use identical transformations.
If it reaches $A_1$, or touches an exceptional edge, charge it by the
second conclusion of the preparation lemma. For a label of $A_1$ with
no incident second-step accepted edge in either swarm, the second
preparation is the identity, so its actual endpoint is its reference
endpoint. Keep all labels of $A_1$ in this accounting even when their
intermediate states happen to agree: their endpoint bridges may still
use different future innovations. The preparation charge is at most
$e_2|A_1|/N$; the reference endpoints disagree on an expected fraction
at most $(1-\epsilon_f)|A_1|/N$. Thus
$$
 \mathbb E\rho_N(S_2,T_2)
 \le(1-\epsilon_f+e_2)\mathbb E|A_1|/N
 \le q_f\rho_N(S,T).
$$
The common noise and graph dependence has been retained throughout.

**Iteration and invariance.** The finite-swarm existence and uniqueness
theorem {prf:ref}`thm-slca-active-stationary` applies with
$\lambda=1/\eta$ and bounded residual $H_c/\eta$; its noninjective
minorization permits the stated globally Lipschitz force. It supplies
$\Pi_N$ for this same complete kernel. Integrate the constructed
two-update coupling over an entering coupling and iterate to contract
$\mathcal W_{\rho_N}$ by $q_f$ every two updates. Its diameter is
one. Starting one marginal at $\Pi_N$ gives the displayed bound at
even times; starting the first marginal at $\Lambda_NP_N$ gives it
at odd times, with no additional prefactor.
Sampling one uniform label in this coupling has physical-state mismatch
probability at most the normalized Hamming cost. Its marginals are the two
mean alive laws, so the coupling inequality for total variation proves
the exact alive-sampled law estimate.

**Physical and alive-law costs.** Every complete output, including
$\Pi_N$ by invariance, has averaged fourth phase moment at most $K_4$.
In the coupling just constructed,
$$
 \mathbb E\frac1N\sum_i|z_i-z_i'|^2
 \le\left(\mathbb E\frac1N\sum_i|z_i-z_i'|^4\right)^{1/2}
       (\mathbb E\rho_N(S,T))^{1/2}
 \le4\sqrt{K_4}\sqrt{\mathbb E\rho_N(S,T)}.
$$
The first inequality is Cauchy--Schwarz on the product of the coupling
probability space and the uniform label measure, because equal rows
contribute zero. The second uses
$|z_i-z_i'|^4\le8(|z_i|^4+|z_i'|^4)$.
Multiply by $\lambda_{\max}(G)$. Matching the same labels is an
admissible empirical transport plan, so it bounds the Wasserstein
cost between the empirical measures in this coupling. Push the coupling
forward to their laws to obtain the first physical bound. Sampling
one uniform label in this same coupling gives the two mean alive
measures as marginals and proves the second. All rows are alive;
there is no survival normalizer or dead-slot penalty in these costs.
:::

:::{prf:corollary} Explicit nonempty interval for exact population-uniform mixing
:label: cor-slcw-finite-positive-exponents

Fix positive reference selection exponents $\bar p_r,\bar p_s$ and
set $p_b=\theta\bar p_b$. Use the primitive $a_0,G_0,M$ of
{prf:ref}`cor-slcc-positive-exponents` and put
$$
 \begin{aligned}
 J_b&=e^M\bar p_bA_b/(4\eta_b),\\
 L_{f,0}&=4a_0/\kappa_C+(4a_0/\kappa_C+2a_0)K_D^f
                 +2G_0(J_rT_r+K_D^fJ_sT_s^f),\\
 E_0&=24a_0/\kappa_C+9L_{f,0},\qquad
 E_2=E_0+6a_0/\kappa_C,\\
 \theta_f&=\min\{1,\kappa_C/(8a_0),\epsilon_f/(8E_2)\}>0.
 \end{aligned}
$$
Every fixed $0<\theta\le\theta_f$ satisfies the theorem and
$q_f\le1-\epsilon_f/2<1$. The accepted-edge and fitness mechanisms
remain active whenever realized fitness differences permit acceptance.
Complete fitness ties are included in the proof.
:::

:::{prf:proof}
The positive-base estimates already proved give
$a_*\le\theta a_0$, $c_*\le\theta a_0/\kappa_C$,
$L_g\le G_0$, and $H_b\le\theta J_b$.
Thus $L_f\le\theta L_{f,0}$. The stated bound on $\theta$ gives
$a_*\le1/8$ and $8c_*\le1$, so $M_f\le e<3$ and
$M_f-1\le8c_*M_f\le24\theta a_0/\kappa_C$.
It follows that $e_1\le\theta E_0$ and $e_2\le\theta E_2$.
Both are at most $\epsilon_f/8$. Therefore
$$
 q_f\le(1-7\epsilon_f/8)(1+\epsilon_f/8)
      =1-3\epsilon_f/4-7\epsilon_f^2/64
      \le1-\epsilon_f/2<1.
$$
Every primitive denominator is positive and finite. This proves an
explicit nonempty parameter interval, rather than assuming a uniform
mixing margin.
:::

:::{prf:corollary} A concrete positive-selection profile with a uniform alive-law rate
:label: cor-slcw-concrete-alive-profile

The conservative all-alive one-dimensional gas has the following
population-size-independent primitive profile satisfying
{prf:ref}`thm-slcw-alive-uniform-law`. Use the current-frame canonical
measurement, donor, copying and accepted-component collision rules,
with no viscosity or history term, and set
$$
\begin{gathered}
h=2,\quad\gamma=0,\quad b_O=\sigma_x=1/\sqrt2,\quad
V_{\max}=10^{-3},\quad\alpha_{\rm col}=1/2,\quad\sigma_J=1/10,\\
U(x)=x^2/4,\quad F(x)=-x/2,\quad R(x,v)=-U(x),\\
R_x=R_v=1,\quad\lambda_{\rm alg}=1,\quad
\epsilon_D=\epsilon_C=4,\quad\delta_D=10^{-3},\\
A_r=A_s=\eta_r=\eta_s=1,\quad
\sigma_r=10^6,\quad\sigma_s=1,\quad
s_c=\epsilon_c=1,\quad p_r=p_s=10^{-15}.
\end{gathered}
$$
The final cap is the actual smooth map
$C_V(v)=V_{\max}v/(V_{\max}+|v|)$. The recipient, OU and final-position
Gaussians retain their unbounded laws, and every accepted connected
component uses its shared Haar rotation, including Haar $O(1)$ here.
The reward is the unbounded raw reward of the same potential supplying
the force. All rows remain alive because terminal death is disabled.

For the kinetic minorization choose $r=1/10$ and $u=1/20$.
The explicit constants of Sections 15.1 and 16.1 satisfy
$$
\epsilon_2>5\cdot10^{-10},\quad
12<M_4<16,\quad B_w\le3/2,\quad
H_0<5000,\quad L_{\rm rem}<7.5\cdot10^{-12},\quad
0<q_w<1-2\cdot10^{-10}.
$$
Consequently the proved vanishing particle floor $\varepsilon_N$ of
{prf:ref}`thm-slcw-transfer`, evaluated on this profile, and every fixed
positive definite phase-space matrix $G$ give
$$
\mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi),\quad
W_{2,G}^2(\lambda_{N,n}^a,\pi)
\le18\lambda_{\max}(G)
\sqrt{\min\{1,\tfrac32(1-2\cdot10^{-10})^{\lfloor(n-1)/2\rfloor}
                         +\varepsilon_N\}},\qquad n\ge1, N\ge2.
$$
Here $\varepsilon_N\to0$. This profile permits positive cloning; it
imposes no positive lower gap on realized fitness and includes exact
ties. The time constants are conservative analytic bounds for the
declared real-coordinate kernel.

With the same kinetic and fitness parameters, alternatively configure
the continuous bounded raw reward $R_b(x,v)=-\tanh(x^2/4)$ while
retaining the declared force $F(x)=-x/2$. This declares two separate
channels, rather than clipping the quadratic reward inside a proof.
For this alternative reward, {prf:ref}`thm-slcw-finite-uniform-law`
applies with
$$
\epsilon_f>5\cdot10^{-10},\qquad E_2<20000,\qquad
0<q_f<1-2.5\cdot10^{-10}.
$$
Its exact finite-swarm stationary empirical-measure law
$\Xi_N^{b,*}$ and mean alive law $\lambda_N^{b,*}$ obey the
following estimates with no particle error floor:
$$
\mathcal W_{W_{2,G}}^2(\Xi_{N,n}^{b},\Xi_N^{b,*}),\quad
W_{2,G}^2(\lambda_{N,n}^{b},\lambda_N^{b,*})
\le17\lambda_{\max}(G)
 (1-2.5\cdot10^{-10})^{\lfloor n/2\rfloor/2},
\qquad n\ge1,\ N\ge2.
$$
Every constant in this exact relaxation estimate is independent of $N$.
The mean alive phase-space law also has the exact total-variation bound
$$
\|\lambda_{N,n}^{b}-\lambda_N^{b,*}\|_{\rm TV}
\le(1-2.5\cdot10^{-10})^{\lfloor n/2\rfloor},
\qquad n\ge0,\ N\ge2.
$$
:::

:::{prf:proof}
The exact kinetic register is
$c=a=q^2=s^2=1$, $B=\eta=2$, $\tau^2=2$,
$V_c=2\cdot10^{-3}$, $H_c=0$, $L_F=1/2$, and $C_r=1/4$.
Thus $x+\eta F(x)=0$ for every prepared position, including all
recipient-jitter realizations. The raw reward obeys the stated quadratic
growth and local Lipschitz conditions with the same potential.
For the reference minorization,
$$
b_K=3+4\cdot10^{-6},\quad
R_0^2=11+16\cdot10^{-6}<(10/3)^2,\quad \alpha_0=1/2.
$$
Hence
$$
R_1=m_v=R_0/2+10^{-3}<1.668,\qquad
Q=2u+R_1<1.768.
$$
The two Gaussian exponents have sum at most
$$
\frac{(1.768+1.668)^2}{2}
+\frac{(0.1+1.668+1.768)^2}{2}<13.
$$
Their determinant factor is $(1+c^2L_F)^{-1}=2/3$;
their Gaussian normalization product is $1/(2\pi)$;
and their ball-volume product is $4ur=1/50$.
Therefore
$$
\epsilon_2>\frac{e^{-13}}{200\pi}
>\frac1{800\cdot3^{13}}>5\cdot10^{-10},
$$
using $e<3$ and $\pi<4$. These inequalities establish a strictly
positive kinetic margin without using floating-point evaluations.

The exact uniform fourth-moment envelope is
$$
M_4=[0.004+\sqrt2\,3^{1/4}]^4.
$$
The Gaussian term alone has fourth power $12$, so $M_4>12$.
Since $12<(1.99)^4$ and $0.004+1.99<2$, $M_4<16$.
With $\beta=\epsilon_2/(2M_4)$ this gives
$\beta^{-1}<6.4\cdot10^{10}$ and $\beta<1$.
The elementary bounds of {prf:ref}`lem-slcw-normalization` consequently
give
$$
B_r<40000,\quad B_{r^2}<8\cdot10^9,\quad
M_r<2,\quad C_{\rm var}<2\cdot10^{10}.
$$
For example $\sqrt{6.4\cdot10^{10}}<3\cdot10^5$ yields
$B_r<(1+150000)/4<40000$, and
$C_{\rm var}<2(8\cdot10^9)+4(2)(40000)<2\cdot10^{10}$.

The feature diameter is $D_*=2\sqrt2$ and
$\kappa_D=\kappa_C=e^{-1/4}>3/4$. Thus
$\kappa_C^{-1}<4/3$, $K_D<4$, and $S_b<3$.
The configured diversity regularizer gives $T_s<4(3+27/2)=66$.
Use reference exponents $\bar p_r=\bar p_s=1$ in
{prf:ref}`cor-slcw-positive-exponents`. Then
$$
e^M=4,\quad\Delta=\log4<2,\quad D_0=5/4,\quad
a_0<6.4,\quad G_0<6,\quad J_r=J_s=1.
$$
The actual reward regularizer $\sigma_r=10^6$ therefore gives
$$
C_{F,0}
<\frac{2(40000)}{10^6}
 +\frac{2(2\cdot10^{10})}{10^{18}}+66<67.
$$
Inserting these bounds into the positive-exponent register gives
$$
L_0<2(6.4)(4)(4/3)+(6.4)(16/9)+2(6)(67)(4/3)<1200,
$$
$$
H_0<2(6.4)(1+4/3)(4)+4(1200)<5000.
$$
Since $B_w\le3/2$, the third endpoint of that register satisfies
$$
\frac{\epsilon_2}{8B_w^2H_0}
>\frac{5\cdot10^{-10}}{18\cdot5000}
>5\cdot10^{-15}>\theta=10^{-15}.
$$
The other two endpoints are at least $1$ and
$\kappa_C/(4a_0)> (3/4)/(4\cdot6.4)>\theta$ respectively.
Thus $\theta$ belongs to the proved positive-selection interval,
including $2c_*<1$, and
$$
L_{\rm rem}\le B_w\theta H_0<7.5\cdot10^{-12}.
$$
Direct substitution into the complete two-update coefficient yields
$$
q_w<1-2.5\cdot10^{-10}
       +3(7.5\cdot10^{-12})+(7.5\cdot10^{-12})^2
<1-2\cdot10^{-10}.
$$
It is positive since $1-\epsilon_2/2\ge1/2$.
In this all-alive regime two rows at unequal positions have equal raw
diversities under their sole distinct companion, while their raw rewards
can differ. The increasing reward logistic and the positive reward
exponent then give a strictly positive actual gate in the improving
direction. This supplies a cloning witness while keeping all sampled
normalizers and component rules unchanged. Exact ties continue to have
their actual zero gate.

Finally $\sqrt{M_4}+V_{\max}^2<4+10^{-6}$, so the preceding
Wasserstein upgrade has $C_G<18\lambda_{\max}(G)$.
Apply {prf:ref}`thm-slcw-alive-uniform-law` and its vanishing particle
floor to obtain the first displayed result.

For the separately configured bounded reward $R_b$, its oscillation is
$1$ and it is continuous. The first kinetic step from any collision-
prepared input uses
$b_K'=3+16\cdot10^{-6}$ and
$(R_0')^2=11+64\cdot10^{-6}<(10/3)^2$.
The same local bounds $R_1,m_v<1.668$, $Q<1.768$ and the same
Gaussian exponents therefore prove $\epsilon_f>5\cdot10^{-10}$.
The finite preparation register gives
$$
K_D^f<7,\qquad T_s^f<18,\qquad
T_r=10^{-6}+\tfrac12\,10^{-18}<10^{-5}.
$$
Since $J_r=J_s=1$, $a_0<6.4$, $G_0<6$ and
$\kappa_C^{-1}<4/3$, its explicit coefficient satisfies
$$
L_{f,0}
<4(6.4)(4/3)
 +[4(6.4)(4/3)+2(6.4)]7
 +2(6)(10^{-5}+7\cdot18)<2000.
$$
Consequently
$$
E_2=30a_0/\kappa_C+9L_{f,0}
<30(6.4)(4/3)+9(2000)<20000.
$$
The finite positive-exponent interval contains the declared
$\theta=10^{-15}$, because
$$
\frac{\epsilon_f}{8E_2}
>\frac{5\cdot10^{-10}}{8(20000)}>10^{-15},\qquad
\frac{\kappa_C}{8a_0}>(3/4)/(8\cdot6.4)>10^{-15}.
$$
Thus {prf:ref}`cor-slcw-finite-positive-exponents` proves
$q_f\le1-\epsilon_f/2<1-2.5\cdot10^{-10}$.
The same fourth moment gives
$C_f=4\lambda_{\max}(G)(\sqrt{M_4}+V_{\max}^2)
<17\lambda_{\max}(G)$.
The exact finite-swarm theorem now proves both no-floor bounds.
To prove the alive-law TV assertion, sample the same uniform label in
the theorem's array coupling. Its two marginals are the mean alive
phase-space laws, and the probability of unequal sampled states is
exactly the expected normalized Hamming cost. The coupling inequality
and the theorem's Hamming estimate give the displayed TV bound.
The actual bounded reward and the unbounded quadratic reward are
distinct configured choices with the same kinetic profile; each estimate
uses its own stationary law. $\square$
:::

:::{prf:corollary} A population-uniform alive-law regime for the standard Rastrigin landscape
:label: cor-slcw-rastrigin-uniform-law

Fix any dimension $d\ge1$ and the standard Rastrigin potential
$$
U(x)=|x|^2+10\sum_{j=1}^d[1-\cos(2\pi x_j)],\qquad
F(x)=-\nabla U(x).
$$
Use the conservative, death-disabled, current-frame nonviscous canonical
gas, retaining sampled companions, global empirical normalizers,
simultaneous copying, component-Haar collisions and all Gaussian
innovations. Fix the following primitive parameters independently of $N$:
$$
\begin{gathered}
h=1,\quad\gamma=0,\quad b_O=\sigma_x=1,\quad
V_{\max}=1,\quad\alpha_{\rm col}=1/2,\quad\sigma_J=1/10,\\
R_x=R_v=1,\quad\lambda_{\rm alg}=1,\quad
\epsilon_D=\epsilon_C=4,\quad\delta_D=10^{-3},\\
A_r=A_s=\eta_r=\eta_s=\sigma_r=\sigma_s=s_c=\epsilon_c=1.
\end{gathered}
$$
Use the actual smooth cap $C_V(v)=v/(1+|v|)$ and choose
minorization radii $r=u=1$. The force and kinetic profiles are
$$
c=1/2,\quad a=B=1,\quad\eta=1/2,\quad
q^2=s^2=1,\quad\tau^2=5/4,\quad V_c=2,
$$
$$
H_c=10\pi\sqrt d,\qquad L_F=2+40\pi^2,\qquad
C_r=1+20\pi\sqrt d.
$$
Compute $\epsilon_2>0$ from {prf:ref}`lem-slcc-base-mixing` using
$$
b_K=1+(1+H_c)^2+5d/4.
$$
Compute $M_4,\beta,B_w,B_r,B_{r^2},M_r,C_{\rm var}$ from the
explicit weighted register of Section 16.1 with these profiles. In this
register the positive-exponent constants reduce to
$$
\begin{gathered}
\kappa_D=\kappa_C=\kappa=e^{-1/4},\quad
K_D=1+2/\kappa,\quad
S_b=\sqrt{8+10^{-6}}-10^{-3},\quad
T_s=K_D(S_b+S_b^3/2),\\
a_0=16\log4/5,\quad G_0=4/5+64\log4/25,\quad
J_r=J_s=1,\\
C_{F,0}^{w}=2B_r+M_rC_{\rm var}+T_s,\\
L_0^{w}=2a_0K_D/\kappa+a_0/\kappa^2
                        +2G_0C_{F,0}^{w}/\kappa,\quad
H_0^{w}=2a_0(1+\kappa^{-1})K_D+4L_0^{w}.
\end{gathered}
$$
For the actual unbounded raw reward $R=-U$, set both selection
exponents equal to the following fixed, explicitly positive number:
$$
p_r=p_s=\theta_w(d)
=\frac12\min\left\{1,\frac{\kappa}{4a_0},
                         \frac{\epsilon_2}{8B_w^2H_0^{w}}\right\}>0.
$$
Then the unique stationary population law $\pi$ and the actual finite
alive empirical and sampled laws obey
$$
\mathbb EW_{2,G}^2(\mu_{N,n}^a,\pi),\quad
W_{2,G}^2(\lambda_{N,n}^a,\pi)
\le C_G\sqrt{\min\{1,
 B_w(1-15\epsilon_2/64)^{\lfloor(n-1)/2\rfloor}
 +\varepsilon_N\}},\qquad n\ge1,\ N\ge2,
$$
where $C_G$ and the explicit vanishing $\varepsilon_N$ are those of
{prf:ref}`thm-slcw-alive-uniform-law`. The time coefficient and rate
are independent of $N$.

Alternatively configure the bounded raw reward $R_b=-\tanh U$ with
the same declared Rastrigin force. For its exact finite-swarm theorem,
compute $\epsilon_f>0$ from the same minorization formulas, replacing
the first-step moment by
$b_K'=1+(2+H_c)^2+5d/4$. Set
$$
\begin{gathered}
K_D^f=1+4/\kappa,\qquad T_s^f=S_b+S_b^3/2,\\
L_{f,0}=4a_0/\kappa+(4a_0/\kappa+2a_0)K_D^f
                           +2G_0(3/2+K_D^fT_s^f),\\
E_2=30a_0/\kappa+9L_{f,0},\qquad
p_r=p_s=\theta_f(d)
=\frac12\min\left\{1,\frac{\kappa}{8a_0},
                         \frac{\epsilon_f}{8E_2}\right\}>0.
\end{gathered}
$$
Let $\Xi_N^{b,*}$ and $\lambda_N^{b,*}$ denote its stationary
empirical-measure law and mean alive law. Then
$$
\|\lambda_{N,n}^{b}-\lambda_N^{b,*}\|_{\rm TV}
\le(1-\epsilon_f/2)^{\lfloor n/2\rfloor},\qquad n\ge0,
$$
$$
\mathcal W_{W_{2,G}}^2(\Xi_{N,n}^{b},\Xi_N^{b,*}),\quad
W_{2,G}^2(\lambda_{N,n}^{b},\lambda_N^{b,*})
\le C_f(1-\epsilon_f/2)^{\lfloor n/2\rfloor/2},\qquad n\ge1,
$$
for every $N\ge2$, with the $N$-independent constant $C_f$ from
{prf:ref}`thm-slcw-finite-uniform-law`. These bounded-reward estimates
have no particle error floor. The two configured reward choices have
their respective stationary laws and explicitly stated exponents.
:::

:::{prf:proof}
The actual force components are
$F_j(x)=-2x_j-20\pi\sin(2\pi x_j)$. Therefore
$$
x+\eta F(x)=-10\pi(\sin(2\pi x_j))_{j=1}^d,
$$
whose norm has supremum exactly $10\pi\sqrt d$.
Its diagonal Jacobian has entries
$-2-40\pi^2\cos(2\pi x_j)$, giving the stated global force
Lipschitz constant. The same potential has $U(0)=0$,
$\lambda_0=1/\eta=2$ and $g_0=H_c/\eta=20\pi\sqrt d$.
{prf:ref}`cor-slcw-same-potential` consequently gives the displayed
quadratic raw-reward growth coefficient $C_r$. On a radius-$L$ ball,
$|\nabla U|\le2L+20\pi\sqrt d$, supplying the required local
reward Lipschitz profile. The force-center identity holds at every
prepared position, including the complete unbounded jitter law.

Both noises are positive and the cap is the stated 1-Lipschitz smooth
map into the unit velocity ball. The kinetic minorization therefore
has finite constants and strictly positive coefficients
$\epsilon_2,\epsilon_f$. The weighted moments are finite and
positive, so $\beta>0$ and every displayed weighted normalization
constant is finite. For the fitness bases in this corollary,
$M=\Delta=\log4$, $D_0=5/4$, and $e^M=4$.
Substitution into {prf:ref}`cor-slcw-positive-exponents` gives exactly
the displayed $a_0,G_0,J_b,C_{F,0}^{w},L_0^{w},H_0^{w}$.
The three endpoints defining $\theta_w$ are strictly positive.
Its half-minimum lies in that corollary's interval, hence
$2c_*<1$ and $q_w\le1-15\epsilon_2/64<1$.
Apply the weighted contraction, finite-particle transfer and
{prf:ref}`thm-slcw-alive-uniform-law` to obtain the raw-reward
Wasserstein conclusions with the proved vanishing particle floor.

The configured alternative $R_b=-\tanh U$ is continuous and has
oscillation $1$, since $U\ge0$, $U(0)=0$ and $U\to\infty$.
Its bounded-reward normalization register has
$T_r=1+1/2=3/2$. The finite register of
{prf:ref}`cor-slcw-finite-positive-exponents` therefore reduces to
the displayed $L_{f,0}$ and $E_2$. Every factor is finite and
strictly positive; the half-minimum defining $\theta_f$ lies in
that proved positive-exponent interval. It follows that
$a_*\le1/4$ and $q_f\le1-\epsilon_f/2<1$.
The exact finite-swarm theorem gives both physical Wasserstein bounds.
Sampling the same uniform label in its Hamming coupling gives the two
mean alive laws with mismatch probability at most
$q_f^{\lfloor n/2\rfloor}$; the coupling inequality proves the TV
bound. All parameters and coefficients are fixed independently of $N$.
Neither argument requires a strictly positive realized fitness gap or
monotone decrease of a discrepancy at every update. $\square$
:::

(sec-slce-entropy-program)=
## 17. Reward-driven entropy confinement and full-law dissipation

:::{div} feynman-prose
Reward-driven copying can confine the gas by moving population toward favorable
regions faster than jitter and kinetic motion move it outward. The regional
selection flux makes that competition quantitative, including donor coverage,
fitness advantages and reverse transfers. It can close with zero auxiliary trap
and without an inward restoring force.

Choose an entropy reference from declared basin costs and tail profiles. Region
volumes enter its normalization: a large exterior region can carry appreciable
reference mass even when its cost is unfavorable. The proved cost drift and
final Gaussian smoothing then give a recursion for the actual joint positional
entropy per walker, with a contraction coefficient, a finite floor and retained
defects, all independent of population size. Conditioning exposes independent
final noise increments; averaging restores the full dependence created by
cloning and collisions. No independent-walker model replaces the swarm.

This entropy floor bounds concentration and tails relative to the chosen
reference. Full-state relaxation toward a stationary phase asks how entropy
relative to that phase dissipates to zero. The next balance isolates the terms
needed for that additional conclusion.
:::

(sec-slceg-profiles)=
### 17.1. Regional selection coefficients and their population defects

:::{prf:definition} Regional selection profiles for entropy and moment control
:label: def-slceg-regional-profiles

Use the conservative all-alive canonical full update with $N\ge2$ and
its declared basin, transition and exterior partition $(A_a)_a$.
The bounded target regions $\mathcal B$ partition $\overline B(0,R_c)$,
including its boundary. The favourable core regions satisfy
$\mathcal C\subset\mathcal B$, and exterior regions $\mathcal E$
partition its complement. Let $w_{ab}^-\le w_C(z,z')\le w_{ab}^+$ and
$g_{ab}^-\le a(F_z,F_{z'})\le g_{ab}^+$ be the actual companion-weight
and acceptance-gate bands for $z\in A_a,z'\in A_b$, uniformly over
all allowed measurement marks and their shared normalization. The bands
are the explicit reward, diversity, regularization and feature formulas
of {prf:ref}`def-slcg-geometry`, including acceptance saturation.

For population laws in a declared class with regional mass bounds
$m_a^-\le\mu(A_a)\le m_a^+$, set
$$
 Z_a^-=\sum_bm_b^-w_{ab}^-,\quad
 Z_a^+=\sum_bm_b^+w_{ab}^+,
$$
$$
 \chi_{\rm in}^{\infty}
 =\inf_{a\in\mathcal E}\sum_{b\in\mathcal C}
             \frac{m_b^-w_{ab}^-g_{ab}^-}{Z_a^+},\quad
 D_b^{\infty}=\sum_a\frac{m_a^+w_{ab}^+g_{ab}^+}{Z_a^-},\quad
 \delta_{\rm out}^{\infty}=\sup_{b\in\mathcal E}D_b^{\infty}.
$$
Only possible occupied source regions enter the infimum and sums, and
all denominators there must be positive. The empty exterior infimum is
set to zero. For an actual finite entering configuration with region
counts $N_a$, use the excluded-self denominators and coefficients
$$
 Z_{a,N}^\pm=\sum_b(N_b-\mathbf1_{b=a})w_{ab}^\pm,
$$
$$
 \chi_{{\rm in},N}
 =\min_{a\in\mathcal E:N_a>0}\sum_{b\in\mathcal C}
          \frac{N_bw_{ab}^-g_{ab}^-}{Z_{a,N}^+},\quad
 D_{b,N}=\sum_{a:N_a>0}\frac{N_aw_{ab}^+g_{ab}^+}{Z_{a,N}^-},\quad
 \delta_{{\rm out},N}=\sup_{b\in\mathcal E}D_{b,N}.
$$
For each representation $\diamond\in\{N,\infty\}$, put
$$
 \chi^\diamond=\chi_{\rm in}^\diamond-\delta_{\rm out}^\diamond,
 \qquad b_{p,\rm sel}^\diamond
 =R_c^p\left(\chi_{\rm in}^\diamond+
                   \sum_{b\in\mathcal B}D_b^\diamond m_b^{+,\diamond}\right),
 \qquad p\ge2,
$$
where $m_b^{+,N}=N_b/N$. Require all displayed upper sums finite.

For classes $\mathfrak G_N$ and $\mathfrak G_\infty$, one sufficient choice is numerical
common envelopes satisfying
$$
 0<\chi\le1,\qquad
 \chi\le\inf_{N,S\in\mathfrak G_N}\chi^N(S),\qquad
 \chi\le\inf_{\mu\in\mathfrak G_\infty}\chi^\infty(\mu),
$$
$$
 b_{p,\rm sel}\ge
 \sup\{b_{p,\rm sel}^N(S),b_{p,\rm sel}^{\infty}(\mu):
                     S\in\mathfrak G_N,\ \mu\in\mathfrak G_\infty\}.
$$
Alternatively, the direct numerical choices of
{prf:ref}`lem-slceg-uniform-counts` are admissible because they prove the
copying inequality itself; the preceding envelope comparisons are then
unnecessary.

These are inequalities on displayed structural profiles, not unknown
optimal convergence constants; an infinite supremum means this choice
has not supplied a finite certificate. Define $W_p=N^{-1}\sum_i|x_i|^p$
for particles and $W_p(\mu)=\mu|x|^p$ for population laws. The actual
selection excess is
$$
 E_p=[W_p^{\rm copy}-(1-\chi)W_p-b_{p,\rm sel}]_+,
$$
where $W_p^{\rm copy}$ denotes the conditional expected frozen-copy
moment for particles and the exact frozen-copy moment for the population
root; Gaussian jitter is not included in this copying observable.
:::

:::{prf:lemma} Population-independent selection coefficients directly from mass bands
:label: lem-slceg-uniform-counts

Use the partition and actual weight/gate bands of
{prf:ref}`def-slceg-regional-profiles`. Supply one set of numerical mass
bands $0\le m_a^-\le m_a^+\le1$ for the population class and all
finite-particle classes, independent of $N\ge2$:
$$
 m_a^-\le\mu(A_a)\le m_a^+,\qquad
 m_a^-\le N_a/N\le m_a^+.
$$
Assume the bands describe a nonempty probability class and take
$\kappa_C\le w_{ab}^-\le w_{ab}^+\le1$,
$0\le g_{ab}^-\le g_{ab}^+\le1$. Define, for possible occupied sources,
$$
 \overline Z_a=\min\left\{1,\sum_bm_b^+w_{ab}^+\right\},\qquad
 \underline Z_a=\max\left\{\kappa_C/2,
            \sum_bm_b^-w_{ab}^- -w_{aa}^-/2\right\}.
$$
These denominators are strictly positive. With
$\mathcal E_+=\{a\in\mathcal E:m_a^+>0\}$, set
$$
 \widehat\chi_{\rm in}
 =\min\left\{1,\inf_{a\in\mathcal E_+}
       \sum_{b\in\mathcal C}
       \frac{m_b^-w_{ab}^-g_{ab}^-}{\overline Z_a}\right\},
 \qquad
 \widehat D_b=\sum_{a:m_a^+>0}
       \frac{m_a^+w_{ab}^+g_{ab}^+}{\underline Z_a}.
$$
The infimum over an empty $\mathcal E_+$ is defined as one before
clipping; its inward-flux requirement is then vacuous. Let
$$
 \widehat\delta_{\rm out}
 =\sup_{b\in\mathcal E:m_b^+>0}\widehat D_b,\quad
 \widehat\chi=\widehat\chi_{\rm in}-\widehat\delta_{\rm out},
$$
with an empty supremum defined as zero, and
$$
 \widehat b_{p,\rm sel}
 =R_c^p\left(\widehat\chi_{\rm in}
       +\sum_{b\in\mathcal B}\widehat D_bm_b^+\right),\qquad p\ge2.
$$
Require the displayed upper sums finite. Then, for every law or finite
configuration satisfying the mass and fitness bands, the actual frozen
copying moment satisfies
$$
 W_p^{\rm copy}\le(1-\widehat\chi)W_p+
                                  \widehat b_{p,\rm sel}.
$$
All constants are computed directly from the declared regional profiles,
with no infimum over particle number or configurations. If
$\widehat\chi>0$, the choices
$\chi=\widehat\chi\le1$ and
$b_{p,\rm sel}=\widehat b_{p,\rm sel}$ can be used in
{prf:ref}`thm-slceg-full-moment` and its coverage-defect theorem.
This is a direct alternative to the sufficient envelope comparisons in
{prf:ref}`def-slceg-regional-profiles`; it does not require
$\chi\le\chi^N(S)$ for an exterior-free configuration whose earlier
empty-minimum convention sets $\chi^N(S)$ to zero.
:::

:::{prf:proof}
For an actual finite root $i$ in region $a$, write its normalized
excluded-self denominator as
$$
 Z_{i,N}=\frac1N\sum_{j\ne i}w_C(z_i,z_j).
$$
The uniform weight floor gives
$Z_{i,N}\ge\kappa_C(N-1)/N\ge\kappa_C/2$.
The regional lower bands separately give
$$
 Z_{i,N}\ge\sum_b(N_b/N)w_{ab}^- -w_{aa}^-/N
 \ge\sum_bm_b^-w_{ab}^- -w_{aa}^-/2.
$$
The two inequalities imply $Z_{i,N}\ge\underline Z_a$.
The upper bounds $Z_{i,N}\le(N-1)/N\le1$ and
$Z_{i,N}\le\sum_b(N_b/N)w_{ab}^+\le\sum_bm_b^+w_{ab}^+$ imply
$Z_{i,N}\le\overline Z_a$. For population roots,
$Z_C(\mu,z)\ge\kappa_C$ and
$Z_C(\mu,z)\ge\sum_bm_b^-w_{ab}^-$, which give the same lower
bound; the same upper proof applies. Positivity of $\overline Z_a$
also follows from $\sum_bm_b^+\ge1$ and the weight floor.

For an exterior recipient in $A_a$, every core donor is a different
finite label. Its total accepted probability of choosing a core donor
is bounded below by
$$
 \sum_{b\in\mathcal C}
       \frac{(N_b/N)w_{ab}^-g_{ab}^-}{\overline Z_a}
 \ge\widehat\chi_{\rm in}.
$$
The population calculation replaces $N_b/N$ by $\mu(A_b)$.
Consequently the magnitude of the selected inward negative flux is at
least
$$
 \widehat\chi_{\rm in}
 \int_{|x|>R_c}(|x|^p-R_c^p)\,d\mu
 \ge\widehat\chi_{\rm in}(W_p-R_c^p),
$$
with the identical empirical formula. This remains true if there are
no exterior recipients: its left side is zero and $W_p\le R_c^p$.
Thus no state-dependent empty-minimum coefficient is needed.

For every accepted edge its positive moment increment is at most the
donor's $p$th power. Its accepted probability is bounded above by
$w_{ab}^+g_{ab}^+/(N\underline Z_a)$ for particles and its accepted
population density by $w_{ab}^+g_{ab}^+/\underline Z_a$. Summing over
recipients gives positive flux at most
$\sum_b\widehat D_bW_{p,b}$. Finite diagonal terms may be included in
this upper sum because they are nonnegative. Exterior terms are at most
$\widehat\delta_{\rm out}W_p$, while bounded target terms are at most
$R_c^p\sum_{b\in\mathcal B}\widehat D_bm_b^+$.
Combine this with the selected negative flux to obtain the displayed
copying inequality. All gate bands were imposed on the actual shared
measurement normalization, so averaging those marks preserves the
bounds. Finally clipping the lower inward coefficient at one preserves
a valid lower bound and ensures $\widehat\chi\le1$.
:::

:::{prf:theorem} Fully evaluated selection-driven moment drift without a prescribed restoring trap
:label: thm-slceg-full-moment

Use {prf:ref}`def-slceg-regional-profiles`, the actual companion floor
$\kappa_C>0$, cap bound $V_c=(1+2|\alpha_{\rm col}|)V_{\max}$,
and a configured force
$F(x)=F_{\rm geom}(x)-\lambda x$ with
$|F_{\rm geom}(x)|\le g_0+g_1|x|$, $\lambda\ge0$.
Set
$$
 A_\lambda=|1-\eta\lambda|+\eta g_1,\quad
 b_0=BV_c+\eta g_0,\quad\tau^2=c^2q^2+s^2,\quad
 m_{d,p}=2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)}.
$$
For any $u>0$ define
$$
 a_p=(1+u)^{p-1}A_\lambda^p,\quad
 r_p=a_p(1-\chi),
$$
$$
 B_p=a_pb_{p,\rm sel}+(1+u^{-1})^{p-1}
 \left[b_0+(A_\lambda\sigma_J+\tau)m_{d,p}^{1/p}\right]^p.
$$
For the actual complete finite kernel and the actual nonlinear population
map, respectively,
$$
 P_NW_p(S)\le r_pW_p(S)+B_p+a_pE_p(S),\qquad
 W_p(\mathcal F_h\mu)\le r_pW_p(\mu)+B_p+a_pE_p(\mu).
$$
If $r_{p,0}=A_\lambda^p(1-\chi)<1$, choose
$$
 u=\left(\frac{1+r_{p,0}}{2r_{p,0}}\right)^{1/(p-1)}-1
 \quad\hbox{when }r_{p,0}>0,
$$
and $u=1$ when $r_{p,0}=0$. This gives $r_p<1$, equal to
$(1+r_{p,0})/2$ in the positive case. In particular the unchanged
algorithm with $\lambda=0$ has a strictly contracting moment estimate
whenever
$$
 \chi>1-(1+\eta g_1)^{-p}.
$$
For $g_1=0$ this requires only a strictly positive verified $\chi$.
No finite global bound on $|x+\eta F(x)|$ is used.

If $\mathbb EW_p(S_0)\le M_{p,0}$ and
$\sup_{N,n}\mathbb EE_p(S_n)\le\bar e_p$, then
$$
 \mathbb EW_p(S_n)
 \le r_p^nM_{p,0}+\frac{B_p+a_p\bar e_p}{1-r_p}(1-r_p^n),
 \quad
 \sup_{N,n}\mathbb EW_p(S_n)
 \le\max\left\{M_{p,0},\frac{B_p+a_p\bar e_p}{1-r_p}\right\}.
$$
The deterministic population version follows with the same defect bound.
Every invariant particle law with finite $p$th moment and expected
selection excess at most $\bar e_p$ obeys the stationary moment bound.
Existence of such invariant laws is a separate recurrence conclusion.
:::

:::{prf:proof}
In the actual signed copying integral, an exterior recipient copied from
a core donor decreases $|x|^p$ by at least $|x|^p-R_c^p$.
Its accepted donor probability is bounded below by the displayed
$w^-g^-/Z^+$ sums. Every positive copying increment is at most the
donor's $p$th power. Summing the upper $w^+g^+/Z^-$ bounds for each
target region therefore bounds positive flux by
$\delta_{\rm out}W_p+R_c^p\sum_{b\in\mathcal B}D_bm_b^+$.
Combining the inward and outward bounds gives
$W_p^{\rm copy}\le(1-\chi^\diamond)W_p+b_{p,\rm sel}^\diamond$
on each certified class. The finite formula excludes the source label
exactly in its denominators. In the positive-flux upper bound counting
an additional same-region diagonal term only enlarges that bound.
The common envelopes and the definition of $E_p$ give the global
copying inequality with its explicit defect, also outside the classes.

Choose a uniform row and let $Y$ be its frozen source, $J\in\{0,1\}$
its cloning indicator, and $X^C=Y+J\sigma_JZ_J$ its prepared position.
The exact completed position satisfies
$$
 |X^+|\le A_\lambda|Y|+A_\lambda\sigma_J|Z_J|
                   +b_0+|cq\xi+s\zeta|.
$$
The jitter bound follows from $J\le1$; no independence of different
rows is required. Minkowski on this one joint probability space gives
$$
 \|X^+\|_{L^p}
 \le A_\lambda(W_p^{\rm copy})^{1/p}
       +b_0+(A_\lambda\sigma_J+\tau)m_{d,p}^{1/p}.
$$
For nonnegative $v,w$, the convexity inequality
$(v+w)^p\le(1+u)^{p-1}v^p+(1+u^{-1})^{p-1}w^p$
now yields precisely the stated full-update coefficients after inserting
the copying bound. The same argument conditions on the actual population
root and its complete collision component, so its cap and moment
bounds agree. Solving $a_p(1-\chi)<1$ gives the displayed choices and
the $\lambda=0$ criterion. Iterating the scalar recurrence gives the
moment bounds; stationarity gives the final assertion when its indicated
integrals are finite.
:::

:::{prf:theorem} Explicit contribution of population-coverage failures
:label: thm-slceg-coverage-defects

For the excess in {prf:ref}`def-slceg-regional-profiles`, put
$C_E=2/\kappa_C+\chi$. On any finite configuration,
$$
 E_p(S)\le C_EW_p(S)\mathbf1_{S\notin\mathfrak G_N}.
$$
Consequently a verified weighted failure budget
$\sup_n\mathbb E[W_p(S_n)\mathbf1_{S_n\notin\mathfrak G_N}]
\le T_{p,N}$ gives $\sup_n\mathbb EE_p(S_n)\le C_ET_{p,N}$.
Alternatively, if
$$
 \sup_n\Pr(S_n\notin\mathfrak G_N)\le\delta_N,\qquad
 \sup_n\mathbb EW_{2p}(S_n)\le M_{2p},
$$
then
$$
 \sup_n\mathbb EE_p(S_n)\le C_E\sqrt{M_{2p}\delta_N}.
$$
For $\delta_N\le C_{\rm cov}N^{-\zeta}$ the additional uniform-time
$p$th-moment floor is at most
$$
 \frac{a_pC_E\sqrt{M_{2p}C_{\rm cov}}}{1-r_p}N^{-\zeta/2}.
$$
The same reasoning applies to a random population-law input when its
copying and failure budgets satisfy the corresponding bounds.
:::

:::{prf:proof}
The actual canonical copy-multiplicity estimate gives
$W_p^{\rm copy}\le(1+2/\kappa_C)W_p$ globally. The excess vanishes
on $\mathfrak G_N$ by the proved flux inequality, and outside it
$$
 E_p\le[(2/\kappa_C+\chi)W_p-b_{p,\rm sel}]_+
                    \le C_EW_p.
$$
Taking expectations proves the weighted-budget assertion. Jensen gives
$W_p^2\le W_{2p}$, and Cauchy–Schwarz yields the probability-budget
bound. Insert it into the preceding full-update recurrence and sum the
geometric series. The bound on $M_{2p}$ is a required proved input;
a small unweighted failure probability alone cannot control distant
failed populations.
:::

:::{prf:corollary} Raw-reward domains and the moments used by mean-field estimates
:label: cor-slceg-meanfield-domains

If the configured raw reward obeys $|R(x,v)|\le K_0+K_2|x|^2$, then
$$
 \mu R^2\le2K_0^2+2K_2^2\mu|x|^4.
$$
Thus the $p=4$ certificate defines the reward variance needed by the
actual mean-field update. The $p=8$ certificate supplies the moment
class in {prf:ref}`thm-slct-quadratic-reward` and its quantitative
finite-horizon trajectories. The $p=24$ certificate supplies the
averaged modulus bound
$$
 \mathbb E\mathsf d(\mathcal F_h\widehat\mu,\mathcal F_h\nu)
 \le C_8(1)\sqrt{1+M_{24}+M_8^3}
                         (\mathbb E\mathsf d(\widehat\mu,\nu))^{1/32}
$$
whenever $\mathbb E\widehat\mu|x|^{24}\le M_{24}$ and
$\nu|x|^8\le M_8$ almost surely. Both moment bounds must cover the
actual particle and compared population evolutions. They can be
supplied by selection drift with $\lambda=0$ when the preceding
certificates close. Higher moments required to turn coverage probabilities
into weighted defects must likewise be proved explicitly. On the
strict structural class, {prf:ref}`thm-slcex-selection-tail-closure`
does so at exponent $2p$, and at $24$ when its displayed test holds,
directly from the same signed selection calculation; it carries their
tails through a growing finite-particle residence window.
The direct mean-field use of that closure, including independent
initialization and its failure probability, is
{prf:ref}`thm-slcex-tail-meanfield-transfer` and
{prf:ref}`cor-slcex-tail-iid-meanfield`.

These are domain, tail and consistency statements. They do not identify
a Gibbs stationary law or turn an entropy bound with a persistent source
into zero-source full-law attraction.
:::

:::{prf:proof}
The raw-reward bound follows from $(a+b)^2\le2a^2+2b^2$.
For the averaged modulus take
$H=\max\{1,\widehat\mu|x|^8,M_8\}$. Jensen within $\widehat\mu$
gives $(\widehat\mu|x|^8)^3\le\widehat\mu|x|^{24}$, hence
$\mathbb EH^3\le1+M_{24}+M_8^3$. Apply the actual pointwise modulus
$C_8(1)H^{3/2}\mathsf d(\widehat\mu,\nu)^{1/32}$, then
Cauchy–Schwarz and concavity of $t^{1/16}$. This proves the displayed
bound without independence between the moment and distance. The
remaining domain statements follow from the cited mean-field theorems.
:::

(sec-slce-selection-entropy)=
### 17.2. Joint entropy confinement and geometric tails

:::{prf:definition} Geometric reference and actual positional entropy
:label: def-slce-reference

Keep the actual conservative all-alive kernel, including its independent final position noises of variance $s^2I_d$, $s>0$. Let $\Psi:\mathbb R^d\to[0,\infty)$ be a declared measurable landscape cost, finite almost everywhere, and define

$$
Z_t=\int_{\mathbb R^d}e^{-t\Psi(x)}\,dx,
\qquad \nu(dx)=Z_1^{-1}e^{-\Psi(x)}\,dx.
$$

Require $0<Z_1<\infty$. This is a reference probability, not an asserted invariant law or Gibbs identification for the nonlinear gas. For the actual joint position law $P_n^x$ of the $N$-walker swarm, put

$$
H_n^{(N)}=\frac1N\operatorname{Ent}(P_n^x\mid\nu^{\otimes N}),
\qquad M_n^{(N)}=\mathbb E\frac1N\sum_{i=1}^N\Psi(X_i^n).
$$

For the nonlinear population law, put $H_n=\operatorname{Ent}(\mu_n^x\mid\nu)$ and $M_n=\mu_n^x\Psi$. No entropy of an atomic empirical measure relative to $\nu$ is used.
:::

:::{prf:lemma} Actual selection flux for a declared landscape cost
:label: lem-slce-potential-flux

For the actual frozen measurement-marked law $\eta_\mu$ and accepted edge density $\beta_\mu$ of {prf:ref}`thm-slcr-population-flux`, set

$$
\Phi_\Psi(\mu)=\iint\beta_\mu(t,u)
 [\Psi(x_u)-\Psi(x_t)]\,\eta_\mu(dt)\eta_\mu(du).
$$

Suppose $\mu\Psi<\infty$. A declared donor core $C$ satisfies $\Psi\le\Psi_c$ and $\mu(C)\ge m$. For almost every individual physical recipient/donor pair with $\Psi(x)>\Psi_c$ and donor in $C$, suppose the actual population product measurement laws, using their declared shared deterministic population normalizers, give probability at least $p_g$ of a fitness advantage at least $\Delta>0$. This is a pairwise lower bound, not an average over exterior pairs. Put

$$
a_g=\min\{1,\Delta/[s_c(F^*+\epsilon_c)]\},
\quad \chi_0=\kappa_Cmp_ga_g.
$$

If the actual outward cost flux is bounded by
$\Phi_{\Psi,+}\le\delta\mu\Psi+b_{\rm rev}$, define $\chi=\chi_0-\delta$ and $B_{\rm sel}=\chi_0\Psi_c+b_{\rm rev}$. Then

$$
\Phi_\Psi(\mu)\le-\chi\mu\Psi+B_{\rm sel}.
$$

The same formula holds for finite particles, replacing integrals by the actual normalized recipient/donor sums, whenever their core count is at least $mN$ and the stated fitness-gap probability lower bound holds for every eligible physical recipient/donor pair conditional on the entering swarm. In that bound, integrate the complete actual random measurement array and its shared random normalizers; do not replace the finite fitness marks by independent marks. Retain the same actual outward-flux bound. These constants are independent of $N$. The nonnegative defect

$$
E_{\rm sel}^{\Psi}=[\Phi_\Psi+\chi M-B_{\rm sel}]_+
$$

makes the inequality valid outside the declared coverage/fitness/flux class. Here $M$ is the entering mean cost, either empirical or population.

To include jitter and kinetics, define the actual Gaussian upper response

$$
Q_0(x)=\sup_{|v|\le V_c}\int\Psi(x+Bv+\eta F(x)+\tau z)\varphi_d(z)\,dz,
\qquad Q_J(x)=\int Q_0(x+\sigma_J z)\varphi_d(z)\,dz,
\quad \tau^2=c^2q^2+s^2.
$$

Assume these envelopes are measurable and have proved bounds
$Q_0(x),Q_J(x)\le A_\Psi\Psi(x)+b_\Psi$, with $A_\Psi,b_\Psi\ge0$. These are Gaussian-response bounds for the specified cost and force, not unspecified convergence rates. Then the actual complete update satisfies

$$
M_{n+1}\le rM_n+B_{\Psi}+D_n,
\quad r=A_\Psi(1-\chi),\quad
B_{\Psi}=A_\Psi B_{\rm sel}+b_\Psi,\quad
D_n=A_\Psi\mathbb E E_{\rm sel}^{\Psi}(S_n)
$$

for particles, and the same formula with $D_n=A_\Psi E_{\rm sel}^{\Psi}(\mu_n)$ for the population map. Require $0\le\chi\le1$ and $r<1$ when using this as a confinement estimate. Regional violations of a Gaussian-response envelope must be retained as their expected positive-part contribution to $D_n$.
:::

:::{prf:proof}
The accepted density is at least $\kappa_Ca_g$ on the declared fitness event, because the donor normalizer is at most one. The inward cost removed is therefore at least
$\chi_0\int(\Psi-\Psi_c)_+\,d\mu\ge\chi_0(\mu\Psi-\Psi_c)$. Subtract this from the outward envelope. For finite particles, every exterior recipient has at least $mN$ eligible core donors, and the per-donor probability is at least $\kappa_C/(N-1)$. Dropping $N/(N-1)\ge1$ gives the identical constant. No independence of walkers is used.

Conditional on the complete cloning preparation, the final position is the prepared position plus $BV+\eta F$ and centered Gaussian noise of covariance $\tau^2I_d$. Its collision velocity satisfies $|V|\le V_c$, even though it depends on the entire component. A nonrecipient therefore contributes at most $Q_0(x)$; an accepted recipient copying frozen donor $x'$ contributes at most $Q_J(x')$. Conditional expectation and the two envelope inequalities give
$M_{n+1}\le A_\Psi(M_n+\Phi_\Psi)+b_\Psi$. Jitter independence is used only after taking the uniform velocity envelope. The stated flux bound or its explicit positive-part defect finishes the proof. The finite sum and rooted population integral are the same conditional calculation.
:::

:::{prf:theorem} Entropy contraction to an explicit confinement floor
:label: thm-slce-entropy-floor

Suppose an actual complete-update landscape-cost estimate has been proved,

$$
M_{n+1}\le rM_n+B_{\Psi}+D_n,
\qquad 0\le r<1,\quad 0\le B_{\Psi}<\infty,\quad 0\le D_n<\infty,
$$

with constants and defects from the preceding flux calculation or {prf:ref}`thm-slcr-selection-drift`. Choose a declared number $a\in(r,1)$ such that $Z_{1-a}<\infty$. Put

$$
M_s=(2\pi s^2)^{-d/2},\qquad \rho=r/a<1,
$$

$$
C_H=\left[B_{\Psi}+\log M_s+\log Z_1
                  +\rho\log(Z_{1-a}/Z_1)\right]_+.
$$

For either the actual normalized joint positional entropy $H_n^{(N)}$ or the nonlinear positional entropy $H_n$, whenever the entering entropy is finite,

$$
\boxed{\quad H_{n+1}\le\rho H_n+C_H+D_n.\quad}
$$

Every displayed coefficient is independent of population size. For $n\ge1$ and finite entering cost $M_0$, the first update supplies the finite bootstrap

$$
H_1\le H_1^+:=[rM_0+B_{\Psi}+D_0+\log M_s+\log Z_1]_+.
$$

Consequently

$$
H_n\le\rho^{n-1}H_1^+
 +C_H\frac{1-\rho^{n-1}}{1-\rho}
 +\sum_{j=1}^{n-1}\rho^{n-1-j}D_j.
$$

If $\sup_jD_j\le\overline D$, the entropy floor is at most
$(C_H+\overline D)/(1-\rho)$. This is entropy confinement to a reference-dependent floor; it does not assert convergence to $\nu$, entropy dissipation to zero, stationary-law uniqueness, or chaos.
:::

:::{prf:proof}
Condition on all input states and all innovations except the final independent position noises. The conditional joint position density is a product of $N$ Gaussian densities of covariance $s^2I_d$, irrespective of the cloning and collision dependence in their centers. It is bounded by $M_s^N$. Mixing over the conditioning preserves this bound. The population root has density bounded by $M_s$ by the same argument. Thus, writing $f$ for the joint density,

$$
\frac1N\operatorname{Ent}(f\mid\nu^{\otimes N})
=\frac1N\int f\log f+\frac1N\int f\sum_i\Psi(x_i)+\log Z_1
\le\log M_s+M+\log Z_1.
$$

This remains valid when initially justified by truncation: the positive part of the log density ratio is bounded by the integrable cost plus a constant, and its negative part integrates to at most $1/e$ against the reference probability. Hence finite cost and the density cap give finite relative entropy. The one-root proof is the $N=1$ calculation.

For completeness, the variational entropy inequality follows by tilting the reference: for any function $G$ with $\nu e^G<\infty$, nonnegativity of entropy relative to $e^G\nu/(\nu e^G)$ gives $\mu G\le\operatorname{Ent}(\mu\mid\nu)+\log\nu e^G$. Bounded truncations and monotone convergence justify nonnegative unbounded $G$.
Apply this with $G=a\sum_i\Psi(x_i)$ and reference $\nu^{\otimes N}$. Its exponential integral is $(Z_{1-a}/Z_1)^N$, so division by $aN$ gives

$$
M_n\le\frac{H_n+\log(Z_{1-a}/Z_1)}a.
$$

Insert this bound into the proved cost drift and then the output density-cap estimate. Enlarging the constant to its positive part preserves the inequality. This proves the displayed entropy recursion without assuming independent particles. Applying only the density-cap estimate after the first update gives the bootstrap for atomic or otherwise infinite-entropy initializations with finite cost. Induction proves the geometric defect convolution and its uniform floor.
:::

:::{prf:corollary} Explicit entropy-to-tail and population-fraction bounds
:label: cor-slce-entropy-tails

For a measurable spatial set $A$ with $p=\nu(A)\in(0,1)$, let $h$ bound the applicable positional entropy. For a nonlinear population, let $t=\mu^x(A)$. For a particle swarm, let $t=\mathbb E L_N^x(A)$. In either case,

$$
 t\le\min\left\{1,\frac{h+\log2}{\log(1/p)},
 \inf_{\lambda>0}\frac{h+\log[1+p(e^\lambda-1)]}{\lambda}\right\}.
$$

Equivalently one may invert the sharper binary inequality
$\operatorname{kl}(t\Vert p)\le h$ on its increasing branch $t\ge p$.
For every $\delta>0$, the actual particle swarm also obeys

$$
\Pr\{L_N^x(A)>\delta\}\le\min\{1,t_{\rm bound}/\delta\},
\qquad
\Pr\{\exists i:X_i\in A\}\le\min\{1,Nt_{\rm bound}\}.
$$

A uniform finite entropy bound therefore gives uniform tightness of the expected empirical laws and nonlinear population laws, because every probability $\nu$ is tight. The maximum-walker escape bound retains its explicit $N$ factor.
:::

:::{prf:proof}
For joint positions, relative entropy to a product reference is at least the sum of marginal relative entropies. This follows by the entropy chain rule and convexity of conditional relative entropy; it also follows by applying the variational formula to sums of bounded one-coordinate tests and taking separate suprema. Convexity then bounds the entropy of the uniformly selected row law $\bar\mu=N^{-1}\sum_i\operatorname{Law}(X_i)$ by $H_n^{(N)}$. No exchangeability is needed.

Push the resulting one-row law and $\nu$ through $x\mapsto\mathbf1_A(x)$. The variational entropy formula restricted to functions of this indicator gives the binary entropy lower bound
$h\ge t\log(t/p)+(1-t)\log((1-t)/(1-p))$. Binary Shannon entropy is at most $\log2$, and $-(1-t)\log(1-p)\ge0$, giving the logarithmic bound. The same variational inequality with $G=\lambda\mathbf1_A$ gives the infimum bound. Markov's inequality and the union bound give the two population statements. For exterior balls, $\nu(A)\to0$, so the logarithmic bound tends to zero for fixed $h$.
:::

:::{prf:corollary} Partition formulas and an entirely evaluated selection certificate
:label: cor-slce-partition-certificates

For the declared basin, transition and exterior-shell partition $(A_j)$, suppose $0<v_j=|A_j|<\infty$ and
$\psi_j^-\le\Psi(x)\le\psi_j^+$ on $A_j$. Then

$$
\sum_jv_je^{-t\psi_j^+}\le Z_t\le\sum_jv_je^{-t\psi_j^-}.
$$

For an exterior union $A=\bigcup_{j\in J_E}A_j$,

$$
\nu(A)\le
\frac{\sum_{j\in J_E}v_je^{-\psi_j^-}}
     {\sum_jv_je^{-\psi_j^+}}.
$$

All nonnegative series are literal series. A finite upper series at $t=1-a$ and a positive lower series at $t=1$ certify the entropy theorem. To obtain an upper entropy-floor bound using only upper partition sums, rewrite its unclipped constant as

$$
B_{\Psi}+\log M_s+(1-\rho)\log Z_1+\rho\log Z_{1-a}.
$$

Both logarithm coefficients are nonnegative, so substitute the respective upper sums. A lower bound for $Z_1$ is needed separately for the displayed tail ratio. The displayed formulas expose narrow-region volumes, basin cost levels and exterior volume growth separately. They require no restoring-force sign or bounded $H_c$.


For any proved $p$th cost drift, an explicit radial reference profile is
$\Psi(x)=(|x|/\ell)^p$, with $p,\ell>0$. Its partition functions are

$$
Z_t=\ell^d\frac{2\pi^{d/2}\Gamma(d/p)}{p\Gamma(d/2)}t^{-d/p},
\qquad \nu\{|x|>R\}=\frac{\Gamma(d/p,(R/\ell)^p)}{\Gamma(d/p)}.
$$

Here $\Gamma(k,z)=\int_z^\infty u^{k-1}e^{-u}du$, so the tail is an explicit one-dimensional integral.

Thus one may always choose

$$
a=(1+r)/2,\qquad \rho=2r/(1+r),\qquad
C_H=\left[B_{\Psi}+\log M_s+\log Z_1+
                \rho\frac d p\log\frac{2}{1-r}\right]_+.
$$

This is a family of reference profiles, not a restriction to a particular reward function. The same choice of $a$ is available for a general geometric cost whenever its displayed $Z_{(1-r)/2}$ is finite; otherwise choose another certified $a\in(r,1)$ or retain failure of this entropy certificate.

A fully evaluated instance uses $\Psi(x)=\lambda|x|^2$, $\lambda>0$, with the actual reward-selection second-moment estimate of {prf:ref}`thm-slcr-selection-drift`. Its $r,b$ and selection-defect coefficient $d_{\rm sel}=(1+t)A_F^2$ are already explicit in reward gaps, donor coverage, regional adverse flux, $g_0,g_1$, and every kinetic/cloning parameter. Then

$$
B_{\Psi}=\lambda b,\quad D_n=\lambda d_{\rm sel}\mathbb E E_{\rm sel}(S_n),
\quad Z_t=(\pi/(t\lambda))^{d/2}.
$$

For the population version remove the expectation on the deterministic defect. Any $a\in(r,1)$ is allowed, and

$$
C_H=\left[\lambda b-\frac d2\log(2\lambda s^2)
                   +\frac{rd}{2a}\log\frac1{1-a}\right]_+,
\qquad \rho=r/a.
$$

Here the reward geometry enters through the proved selection drift, while the Gaussian reference is merely a convenient entropy gauge. A reward-derived $\Psi$ instead uses its own partition sums and Gaussian-response envelopes from the preceding lemma. In particular $F\equiv0$ is allowed: the existing selection estimate has $A_F=1$ and $r<1$ whenever its positive inward selection margin and defect control hold. No confining force is inserted.
:::

:::{prf:proof}
Integrate the pointwise exponential bounds over each partition member and sum by monotone convergence. The tail ratio follows by using an upper bound for its numerator and a lower bound for the normalizing denominator. For the radial profile use polar coordinates and the substitution $u=t(r/\ell)^p$, giving the displayed Gamma integral. In the quadratic instance multiply the actual second-moment drift by $\lambda$ and evaluate the elementary Gaussian integral. Substitution into the entropy-floor formula gives the displayed coefficient. Its use at $F=0$ follows directly from the selection-drift parameters; donor coverage failures remain in their actual expected defect, rather than being erased by the reference choice.
:::

(sec-slce-moving-reference)=
### 17.3. Exact full-law entropy production

:::{div} feynman-prose
Freeze the population environment for one update and transport both the current
law and the reference through that same kernel. The shared update loses
information, giving the two nonnegative dissipation terms below. But the
transported reference generally differs from the reference we started with.
Comparing back to the original reference produces an additional, exactly
specified term. Even a stationary phase is stationary under its own environment;
that identity does not freeze it under every other population's environment.

The full entropy balance therefore keeps both the information losses and this
reference-production term. A quantitative bound showing that dissipation exceeds
production gives contraction toward the phase. The earlier confinement estimate
remains useful independently: it controls tails without declaring the geometric
reference to be the gas's invariant law.
:::

:::{prf:definition} Frozen environment and transported reference
:label: def-slce-frozen-reference

Work with the all-alive canonical map of Chapter 8 on the capped phase
space $E$. For an entering population law $\mu$, let $C_\mu(z,dc)$ be
its actual rooted preparation kernel, including measurement companions,
accepted copying, recipient jitter and complete component collisions.
Assume the environment has the raw-reward moments required to define
its regularized normalizations. Define this kernel by the actual rooted
formula at every finite admissible root (or at least $(\mu+\pi)$-almost
every root), rather than choosing an arbitrary $\mu$-almost-everywhere
version of a conditional law. Its environment, including all reward
normalizations, is frozen at $\mu$ even when a different root distribution
is supplied. Let $K$ be
the actual kinetic/noise/cap kernel on prepared inputs, and write

$$
P_\mu=C_\mu K,\qquad \mathcal F_h\mu=\mu P_\mu.
$$

This kernel uses the usual conditional rooted construction; it is not
obtained by recomputing the environment at each individual root.
For a declared reference probability $\pi$, put

$$
\alpha=\mu C_\mu,\quad \beta=\pi C_\mu,\quad
q=\alpha K,\quad \sigma=\beta K.
$$

Assume $H(\mu\mid\pi)<\infty$, $\sigma\ll\pi$, and that the logarithm
$\log(d\sigma/d\pi)$ is integrable under $q$. The conclusions below
apply to a full stationary population reference if one has actually
proved $\mathcal F_h\pi=\pi$; a reward-weighted reference need not be
stationary.
:::

:::{prf:proposition} Exact preparation, kinetic and target-mismatch balance
:label: prop-slce-exact-balance

Let $A_\mu(dc,dz)$ and $A_\pi(dc,dz)$ denote the two reverse conditional
laws of the entering root given its prepared state, under
$\mu(dz)C_\mu(z,dc)$ and $\pi(dz)C_\mu(z,dc)$. Likewise let
$B_\alpha(dy,dc)$ and $B_\beta(dy,dc)$ be the reverse conditional laws
of the prepared state given the final output, under $\alpha(dc)K(c,dy)$
and $\beta(dc)K(c,dy)$. Then

$$
\boxed{\quad
H(\mathcal F_h\mu\mid\pi)-H(\mu\mid\pi)
=-\mathcal I_C(\mu,\pi)-\mathcal I_K(\mu,\pi)
+\mathcal D_\pi(\mu),\quad}
$$

where the two information losses are nonnegative and explicitly equal to

$$
\mathcal I_C=\int\alpha(dc)H(A_\mu(c,\cdot)\mid A_\pi(c,\cdot)),
\qquad
\mathcal I_K=\int q(dy)H(B_\alpha(y,\cdot)\mid B_\beta(y,\cdot)),
$$

and the actual target-production term is

$$
\mathcal D_\pi(\mu)=
\int (\mathcal F_h\mu)(dy)
\log\frac{d(\pi C_\mu K)}{d\pi}(y).
$$

The same identity for cloning alone replaces $P_\mu$ by $C_\mu$,
omits $\mathcal I_K$, and uses the preparation-space reference when
its absolute continuity hypotheses hold.
:::

:::{prf:proof}
The joint input/preparation laws have relative entropy $H(\mu\mid\pi)$:
the two conditional kernels are identical. The entropy chain rule,
disintegrating by the preparation coordinate, gives
$H(\mu\mid\pi)=H(\alpha\mid\beta)+\mathcal I_C$.
Applying precisely the same calculation to the kinetic joint laws gives
$H(\alpha\mid\beta)=H(q\mid\sigma)+\mathcal I_K$.
All these entropies and information losses are finite because the initial
entropy is finite. Finally the logarithmic Radon--Nikodym factorization
$dq/d\pi=(dq/d\sigma)(d\sigma/d\pi)$, integrated against $q$, gives
$H(q\mid\pi)=H(q\mid\sigma)+\mathcal D_\pi(\mu)$.
Combining the three equalities proves the assertion. These arguments
are valid on standard Borel spaces using conditional distributions;
no Lebesgue density for the intermediate copied population is required.
:::

:::{prf:remark} What the existing cloning entropy theorem supplies
:label: rem-slce-source-scope

The proved result {prf:ref}`thm-cloning-entropy-contraction` in Chapter 15
is data processing for a *single* Markov kernel $P$ having $\pi P=\pi$.
The displayed production term is then zero. For the nonlinear population
map, stationarity $\pi P_\pi=\pi$ only cancels this term at $\mu=\pi$;
it does not imply $\pi P_\mu=\pi$ at another input law. The normalized
multiplication equation in {prf:ref}`prop-hypocoercive-selection-derivative`
is a different specified model and has the explicitly signed covariance
$\omega\operatorname{Cov}_f(V,\log(f/\pi))/\bar V_f$.
Neither result proves that the canonical cloning update separately
contracts entropy toward a reward Gibbs law. The complete update may
still have entropy dissipation through a balance of the displayed terms.
:::

:::{prf:proposition} A canonical cloning obstruction to a fixed reward target
:label: prop-slce-cloning-obstruction

Take the spatial marginal of the canonical population preparation map,
with zero positional jitter, two distinct positions $x_0,x_1$, reward
$R(x_1)>R(x_0)$, a positive reward exponent and diversity exponent zero.
Take all velocities zero. Let
$\pi=(1-p)\delta_{x_0}+p\delta_{x_1}$, $0<p<1$, including the particular
choice of $p$ prescribed by any finite-temperature two-point Gibbs
weight for those rewards. Regularized reward normalization preserves the
strict reward order; the sigmoid and positive exponent therefore give
fitness $f_1>f_0>0$. With the environment $\pi$, let

$$
A=\min\{1,(f_1-f_0)/(s_c(f_0+\epsilon_c))\}>0,
\qquad
k_{01}=\exp[-d_C(x_0,x_1)^2/(2\epsilon_C^2)]>0.
$$

The mass copied from the low-reward location to the high-reward location is

$$
b=(1-p)\frac{p k_{01}}{(1-p)+p k_{01}}A>0.
$$

There is no accepted reverse transfer. Consequently the cloned spatial
marginal is $(1-p-b)\delta_{x_0}+(p+b)\delta_{x_1}$, and

$$
H(\pi C_\pi\mid\pi)
=(p+b)\log\frac{p+b}{p}
 +(1-p-b)\log\frac{1-p-b}{1-p}>0,
$$

whereas $H(\pi\mid\pi)=0$. The formula concerns the spatial marginal;
collision processing leaves these positions unchanged.
:::

:::{prf:proof}
The actual companion kernel assigns a low-state root a high-state donor
with probability $p k_{01}/[(1-p)+p k_{01}]$. Its gate is $A$.
High-state recipients have zero gate toward a lower or equal fitness,
and equal-state copies do not change position. Thus the stated mass
transfer is exactly the root marginal of the canonical construction;
other component members do not alter the root's copied position.
Since $0<b<1-p$, the two probability vectors differ and strict positivity
of relative entropy gives the final inequality. This is a statement about
cloning alone and does not discard kinetic mixing from the full dynamics.
:::

:::{prf:lemma} Quantitative entropy closure with the feedback defect retained
:label: lem-slce-feedback-closure

Suppose a declared class $\mathfrak C$ is invariant under $\mathcal F_h$
and contains an actual fixed point $\pi$. For every $\mu\in\mathfrak C$ with $H(\mu\mid\pi)<\infty$,
suppose its frozen full kernel has a certified common component

$$
P_\mu(z,\cdot)=\varepsilon\nu_\mu(\cdot)
 +(1-\varepsilon)Q_\mu(z,\cdot),\qquad 0<\varepsilon\le1,
$$

with the component independent of $z$. Write
$r_\mu=d(\pi P_\mu)/d\pi-1$. Suppose the explicitly certified
relative-density envelopes on this class give

$$
|r_\mu|\le\tfrac12,\qquad
\int r_\mu^2\,d\pi\le L^2 H(\mu\mid\pi).
$$

Let $t=\lceil2/\varepsilon\rceil$ and
$C_t=(t+1)(3/2)^{t-1}/2$. Then

$$
H(\mathcal F_h\mu\mid\pi)
\le[(1+t^{-1})(1-\varepsilon)+C_tL^2]H(\mu\mid\pi).
$$

In particular $C_tL^2\le\varepsilon/4$ gives contraction with rate
$1-\varepsilon/4$. The common-component coefficient and the two density
envelopes are sufficient estimates to prove, not consequences of
reward normalization or of stationary target identification. A finite
set of evaluated laws does not prove these class-uniform envelopes.
:::

:::{prf:proof}
Joint convexity and data processing yield
$H(\mu P_\mu\mid\pi P_\mu)\le(1-\varepsilon)H(\mu\mid\pi)$.
Set $q=\mu P_\mu$, $\sigma=\pi P_\mu$. The entropy variational inequality
applied under $\sigma$ to $t\log(d\sigma/d\pi)$ gives

$$
\int q\log\frac{d\sigma}{d\pi}
\le t^{-1}H(q\mid\sigma)
 +t^{-1}\log\int(1+r_\mu)^{t+1}\,d\pi.
$$

Taylor's theorem on $[1/2,3/2]$, together with $\pi r_\mu=0$, bounds
the last logarithm by
$t(t+1)(3/2)^{t-1}\pi r_\mu^2/2$; here $\log(1+u)\le u$.
Add $H(q\mid\sigma)$ and use the two assumptions.
Finally $t^{-1}\le\varepsilon/2$ implies
$(1+t^{-1})(1-\varepsilon)\le1-\varepsilon/2$.
The prescribed defect bound leaves contraction $1-\varepsilon/4$.
:::

:::{prf:remark} Relation between confinement and full-law entropy relaxation
:label: rem-slce-confinement-attraction

The full-law identity identifies the missing term without assuming its
sign. Selection-derived Lyapunov drift and the separate spatial
entropy-to-floor estimate can prove reward-geometry self-confinement
without a restoring kinetic force. They do not set
$\mathcal D_\pi(\mu)$ to zero and do not identify a fixed Gibbs target.
A contraction-to-zero proof must bound this actual feedback production,
or use a separately proved full-law contraction mechanism.
:::

(sec-slcs-signed-entropy)=
### 17.4. Signed nonlinear feedback around an actual stationary phase

:::{div} feynman-prose
Imagine starting near an actual stationary population. We can evolve the
current walkers with the stationary population supplying their environment,
or let the current population supply that environment itself. The difference
between these two experiments is the feedback we must calculate. Neither
experiment changes the algorithm used to define the actual evolution.

An absolute-value estimate charges every feedback contribution as a loss.
The entropy identity below keeps its sign. It separates the decrease under
the frozen environment, a signed cross term, and a nonnegative remainder.
A negative cross term can therefore help convergence instead of disappearing
inside an error bound. The regional decomposition keeps this information
when mass moves between basins, passages and tails.

Fitness has a similar bookkeeping rule. Copying gains frozen fitness, but
motion and fresh measurements change what is being measured. At stationarity
that refresh exactly cancels the gain. These identities expose the quantities
whose combined sign decides decay; they do not yet establish attraction
throughout a phase basin.
:::

:::{prf:definition} Three laws in the same frozen-phase comparison
:label: def-slcs-three-laws

Use the actual all-alive kernels $C_\mu,K,P_\mu=C_\mu K$ of
{prf:ref}`def-slce-frozen-reference`. Let $\pi$ be an actual fixed phase,
$\pi P_\pi=\pi$, and take $\mu\ll\pi$. Define

$$
G=\mu P_\pi,\qquad Q=\mu P_\mu=\mathcal F_h\mu,
\qquad p=\frac{d\mu}{d\pi},\quad g=\frac{dG}{d\pi},
\quad b=\frac{d(Q-G)}{d\pi}.
$$

The comparison $G$ transports the *current* root law through the stationary
phase's environment. It is different from $\pi P_\mu$ used in the
transported-reference identity. Assume the displayed Radon--Nikodym
ratios exist, $g>0$ where $Q$ has mass, and the entropy and cross
integrals below are finite. Alternatively the identity can be used first
on finite-integral truncations and then passed to a limit with separately
proved integrability bounds. It is not an $\infty-\infty$ identity.
:::

:::{prf:proposition} Exact signed nonlinear entropy increment
:label: prop-slcs-signed-bregman

With $J(u)=(1+u)\log(1+u)-u$, $u\ge-1$, define

$$
\mathcal I_\pi(\mu)=H(\mu\mid\pi)-H(G\mid\pi)\ge0,
\qquad
\mathcal C_\pi(\mu)=\int b\log g\,d\pi.
$$

Then the actual one-step entropy increment is

$$
\boxed{\quad
H(Q\mid\pi)-H(\mu\mid\pi)
=-\mathcal I_\pi(\mu)+\mathcal C_\pi(\mu)+H(Q\mid G).
\quad}
$$

In particular the last term is nonnegative, whereas the cross term has
no prescribed sign. Its exact Bregman and upper-bound forms are

$$
H(Q\mid G)=\int gJ(b/g)\,d\pi
=\int b^2\int_0^1\frac{1-t}{g+tb}\,dt\,d\pi
\le\int\frac{b^2}{g}\,d\pi.
$$

The inner integral is understood by its limit if $g+b=0$ and $g>0$.
On $\{g=0\}$, absolute continuity $Q\ll G$ gives $b=0$ almost
everywhere; define both complete integrands $gJ(b/g)$ and $b^2/g$
as zero there, as well as the corresponding Taylor remainder.
:::

:::{prf:proof}
Since $\pi P_\pi=\pi$, data processing gives
$H(G\mid\pi)\le H(\mu\mid\pi)$. Also $\int b\,d\pi=0$.
The scalar Taylor identity for $j(v)=v\log v$ is

$$
j(g+b)-j(g)=b(1+\log g)
 +b^2\int_0^1\frac{1-t}{g+tb}\,dt.
$$

Integrate, cancel $\int b$, and identify the remainder as
$\int[(g+b)\log((g+b)/g)-b]d\pi=H(Q\mid G)$.
Finally $\log(1+u)\le u$ and $1+u\ge0$ give
$J(u)\le u^2$ for every $u\ge-1$. This proves the upper bound
without requiring $|b|<g$ or excluding zeros of the output density.
:::

:::{prf:lemma} Actual rooted-kernel formula and kinetic remainder
:label: lem-slcs-rooted-remainder

In any common output coordinates having the actual conditional density
$k(y\mid c)$, the signed density used above is exactly

$$
b(y)=\frac{1}{\pi(y)}\int\mu(dz)
 \left[\int k(y\mid c)C_\mu(z,dc)
             -\int k(y\mid c)C_\pi(z,dc)\right].
$$

The expression is taken only where $\pi(y)>0$. In the regime of
{prf:ref}`lem-slcpd-density`, its displayed $k_\theta$ is this exact
kernel in pre-cap coordinates. The common measurable bijective cap preserves
relative entropy as well as TV; hence no cap Jacobian is dropped in this
coordinate choice. Outside that density lemma's hypotheses retain the
actual kernel and its Radon--Nikodym measures, rather than using the
inverse formula without its assumptions.

Let $\Gamma$ be any coupling of the two actual prepared-root laws
$\mu C_\mu$ and $\mu C_\pi$. Then

$$
H(Q\mid G)\le
\int H(K(c,\cdot)\mid K(c',\cdot))\,\Gamma(dc,dc').
$$

For a known input law $\mu$, both preparation laws are the exact rooted
component constructions, with all copying, jitter and collisions retained.
Their difference cannot be replaced by an independent-donor update.
:::

:::{prf:proof}
Integrate the actual conditional output density against each prepared law
and subtract. The two resulting output densities are $Q$ and $G$ by
iterated conditional expectation. This proves the first identity.
For the second, form the two joint laws
$\Gamma(dc,dc')K(c,dy)$ and $\Gamma(dc,dc')K(c',dy)$.
Their relative entropy is the displayed integrated kinetic relative
entropy. Marginalizing to $y$ yields $Q,G$, so data processing proves
the bound. Infinite right-hand side is permitted and is uninformative.
:::

:::{prf:proposition} Basin, transition and tail decomposition with signed bounds
:label: prop-slcs-regional-test

Lift the declared spatial partition to output phase space, optionally
refining its velocity and spatial cells, to obtain disjoint measurable
sets $(A_j)$. Set

$$
\pi_j=\pi(A_j),\quad G_j=G(A_j),\quad
m_j^\pm=\int_{A_j}(\pm b)_+\,d\pi,
\quad d_j=m_j^+-m_j^-=Q(A_j)-G(A_j),
\quad E_j=\int_{A_j}b^2\,d\pi.
$$

Suppose the actual density envelopes give
$0<g_j^-\le g\le g_j^+<\infty$ on each certified cell. Write
$\ell_j=\log g_j^-$, $u_j=\log g_j^+$,
$z_j=(\ell_j+u_j)/2$, $o_j=(u_j-\ell_j)/2$.
Then

$$
\int_{A_j}b\log g\,d\pi
\le u_jm_j^+-\ell_jm_j^-
=z_jd_j+o_j(m_j^++m_j^-),
\qquad
\int_{A_j}gJ(b/g)\,d\pi\le E_j/g_j^-.
$$

These bounds retain cancellation in the signed regional transfer $d_j$.
They do not replace it by its absolute value. For a tail union $T$ not
having a positive global density-ratio lower bound, retain instead

$$
C_T=\int_T b\log g\,d\pi,
\qquad R_T=\int_T gJ(b/g)\,d\pi,
$$

or proved upper bounds for these actual integrals. A mere tail probability
bound does not bound the reciprocal-density integral $\int_Tb^2/g$.
For any finite family of certified cells with complement $T$, the
one-step signed estimate is therefore

$$
H(Q\mid\pi)-H(\mu\mid\pi)
\le-\mathcal I_\pi(\mu)
 +\sum_j\left[z_jd_j+o_j(m_j^++m_j^-)+E_j/g_j^-\right]
 +C_T+R_T.
$$

The sum uses that finite family only. Countable versions require convergence
of the asserted upper-bound series and integrability of the signed cross
term. The kinetic-coupling bound of {prf:ref}`lem-slcs-rooted-remainder`
can replace the complete sum of the nonnegative remainder bounds.
:::

:::{prf:proof}
On $A_j$, the positive part of $b$ is multiplied by at most $u_j$
and its negative part by at least $\ell_j$. This proves the first
inequality and its algebraic rearrangement. The second follows pointwise
from $gJ(b/g)\le b^2/g\le b^2/g_j^-$. Integrate and sum the exact
identity of {prf:ref}`prop-slcs-signed-bregman`, leaving its tail terms
unaltered. No transition-chain or independent-walker approximation is used.
:::

:::{prf:remark} Evaluating the frozen dissipation without an unknown rate
:label: rem-slcs-evaluated-loss

The nonnegative $\mathcal I_\pi(\mu)$ is an actual entropy difference,
not an assumed optimal convergence constant. It can be evaluated from
$\mu$, $\pi$, and the rooted output $G$. For example, on a full finite
partition of the entering space, the log-sum inequality gives

$$
H(\mu\mid\pi)\ge H_{\rm in}^-:=
\sum_j\mu(A_j)\log[\mu(A_j)/\pi(A_j)].
$$

Zero-mass terms use the usual entropy conventions. For output cells with
$g_j^-<g_j^+$, convexity of $J$ gives the explicit chord bound

$$
\int_{A_j}J(g-1)d\pi\le
\frac{\pi_jg_j^+-G_j}{g_j^+-g_j^-}J(g_j^--1)
+\frac{G_j-\pi_jg_j^-}{g_j^+-g_j^-}J(g_j^+-1).
$$

When the endpoints agree use $\pi_jJ(g_j^--1)$. Add a proved output-tail
entropy bound if using a finite family. The resulting $H_G^+$ gives
$\mathcal I_\pi(\mu)\ge\max(0,H_{\rm in}^--H_G^+)$.
Thus every component of the regional sign test is a specified integral
or density envelope of the actual map. The test can fail to certify a
sign when these bounds are too wide. Finite rooted truncation with a TV
remainder alone does not supply the weighted entropy-tail bounds required
here. Neither this calculation nor the test constructs the fixed phase
$\pi$; its stationary identity must already be established.
:::

:::{prf:proposition} Exact finite secants preserve the feedback cross term
:label: prop-slcs-secant

Within the absolute-continuity domain of {prf:ref}`def-slcs-three-laws`,
let $\varphi$ be a bounded mean-zero function under $\pi$, take $\epsilon\ne0$, and set
$\mu_\epsilon=(1+\epsilon \varphi)\pi$. Define the actual finite secants

$$
A\varphi=\frac{d[(\varphi\pi)P_\pi]}{d\pi},\qquad
B_\epsilon \varphi=
\frac{d[\mu_\epsilon(P_{\mu_\epsilon}-P_\pi)]}
     {\epsilon\,d\pi},\qquad
k_\epsilon=A\varphi+B_\epsilon \varphi.
$$

No derivative of a positive-part acceptance gate at a tie is asserted.
For any declared $0<r<1$ such that
$|\epsilon \varphi|\le r$ and $|\epsilon k_\epsilon|\le r$ almost everywhere,

$$
\frac{H(\mathcal F_h\mu_\epsilon\mid\pi)}
     {H(\mu_\epsilon\mid\pi)}
\le\frac{1+r}{1-r}
\frac{\|A\varphi\|_2^2+2\langle A\varphi,B_\epsilon \varphi\rangle
                       +\|B_\epsilon \varphi\|_2^2}{\|\varphi\|_2^2}
$$

for $\varphi\ne0$. All norms and pairings are in $L^2(\pi)$, and all
three terms on the numerator are the displayed actual kernel integrals.
In particular $\|A\varphi\|_2\le\|\varphi\|_2$, and the explicitly evaluated
signed loss

$$
S_\epsilon(\varphi)=\|\varphi\|_2^2-\|A\varphi\|_2^2
 -2\langle A\varphi,B_\epsilon \varphi\rangle-\|B_\epsilon \varphi\|_2^2
$$

certifies a strict entropy decrease whenever
$S_\epsilon(\varphi)>2r\|\varphi\|_2^2/(1+r)$.
:::

:::{prf:proof}
Stationarity gives
$d(\mathcal F_h\mu_\epsilon)/d\pi=1+\epsilon k_\epsilon$ exactly.
For $|u|\le r$, Taylor's integral formula gives
$u^2/[2(1+r)]\le J(u)\le u^2/[2(1-r)]$.
Apply the lower bound to input entropy and upper bound to output entropy;
expand $\|A\varphi+B_\epsilon \varphi\|_2^2$. To prove the frozen norm bound,
under the stationary joint law $\pi(dz)P_\pi(z,dy)$ the function
$A\varphi(y)$ is $\mathbb E[\varphi(Z)\mid Y=y]$. Conditional Jensen proves
$\pi(A\varphi)^2\le\pi \varphi^2$. Rearranging the entropy ratio gives the
stated strict-decrease criterion.
:::

:::{prf:remark} What these signed calculations establish
:label: rem-slcs-scope

The existing keystone estimate {prf:ref}`thm-slcn-keystone-power`
bounds recipient-weighted positional pressure. Its observable is not
$b\log g$ or the secant cross product above. The kinetic modified-entropy
proof {prf:ref}`thm-explicit-kinetic-decay` concerns its specified
continuous kinetic reference. The common-target result
{prf:ref}`thm-cloning-entropy-contraction` requires invariance under its
single kernel. The full-generator theorem
{prf:ref}`thm-kl-convergence-euclidean` states its signed derivative
inequality as a hypothesis. None of these displayed conclusions assigns
a sign to $\langle A\varphi,B_\epsilon\varphi\rangle$ for the canonical nonlinear
map. The energy calculation {prf:ref}`lem-meanfield-cloning-dissipation-hybrid`
also uses a separately specified rate and observable.

The new identities retain the precise cancellation an unsigned feedback
norm discards. They do not yet prove that it has the required sign
uniformly through any declared phase basin. A checked finite list of
secants is not a proof for every perturbation in an infinite-dimensional
phase class. Even positive loss in every separately checked direction
need not give a uniform positive margin: such losses can tend to zero
along a sequence of directions. To infer a quantitative local phase
attraction rate, the actual regional bounds must control all perturbations
in a neighborhood with an invariant or explicitly controlled trajectory
class. The calculations above have not derived that uniform signed estimate
from the source axioms of confinement, force regularity, nondegenerate
noise and positive keystone pressure. Consequently this insertion provides an exact signed
calculation and evaluable tests, not an asserted new general phase
attraction theorem.
:::

:::{prf:proposition} Signed frozen-fitness gain and exact population refresh balance
:label: prop-slce-signed-fitness-refresh

Use the conservative all-alive canonical population map
$\mathcal F_h$ of {prf:ref}`def-slce-frozen-reference`, with the
parameter record {prf:ref}`def-slc-parameter-register`. Thus every root
is alive, the physical state space is
$E=\mathbb R^d\times\overline B(0,V_{\max})$, and no killing or
conditional normalization is included in this statement. Keep the
actual sampled measurement marks, component collision, recipient
jitter, BAOAB update and final cap. Let $\mu$ and
$\nu=\mathcal F_h\mu$ be laws for which the reward moments defining
the actual fitness normalizations are finite. All expressions below
are bounded once those normalizations are defined.

For this statement use the common type space
$T=E\times E$, writing $t=(z_t,y_t)$ for a root and its measurement
companion. Its law and frozen fitness are

$$
\widehat\eta_\mu(dt)=\mu(dz_t)P_D(\mu;z_t,dy_t),\qquad
f_\mu(t)=F_\mu(z_t,y_t),
$$

where $P_D$ and $F_\mu$ are exactly
{prf:ref}`def-mean-field-moments` and
{prf:ref}`def-mean-field-fitness-potential`. The fitness coordinate of
$\eta_\mu$ is a deterministic function of this type, so suppressing
it changes no sampling law. Put

$$
F_* =\eta_r^{p_r}\eta_s^{p_s},\qquad
F^*=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},\qquad
\Delta_F=F^*-F_*.
$$

Then $0<F_*\le f_\mu\le F^*$ for every admissible environment.
Define the actual donor density and acceptance by

$$
k_\mu(t,u)=\frac{w_C(z_t,z_u)}{Z_C(\mu;z_t)},\qquad
 a_\mu(t,u)=\min\!\left\{1,
 \frac{(f_\mu(u)-f_\mu(t))_+}
 {s_c(f_\mu(t)+\epsilon_c)}\right\},\qquad
 \beta_\mu=k_\mu a_\mu,
$$

with $w_C,Z_C$ from the canonical companion kernel. In particular,
$\int k_\mu(t,u)\widehat\eta_\mu(du)=1$ for each $t$.
Let $t_*$ be the selected source type: it equals the entering root
$t$ if its proposal is rejected, and its donor type $u$ if the
proposal is accepted. This records an existing choice inside the
actual update; it does not copy the donor's velocity or change the
component collision rule. Let $\lambda_\mu$ be the law of $t_*$ and
set

$$
\begin{aligned}
 \overline p_\mu&=\iint\beta_\mu(t,u)
       \widehat\eta_\mu(dt)\widehat\eta_\mu(du),\\
 J(\mu)&=\int f_\mu(t)\widehat\eta_\mu(dt),\\
 G_\mu&=\iint\beta_\mu(t,u)
       [f_\mu(u)-f_\mu(t)]
       \widehat\eta_\mu(dt)\widehat\eta_\mu(du).
\end{aligned}
$$

The signed frozen-fitness gain satisfies the explicit bounds

$$
\boxed{\quad
 \lambda_\mu f_\mu-J(\mu)=G_\mu,\qquad
 s_c(F_*+\epsilon_c)\overline p_\mu^{\,2}
 \le G_\mu\le\Delta_F\overline p_\mu.
 \quad}
$$

To evaluate the refresh term, keep the joint law of $t_*$ and the
actual final physical root $z'$ generated by the complete rooted
update. Conditional on $z'$, draw a fresh measurement companion
$y'\sim P_D(\nu;z',\cdot)$ and put $t'=(z',y')$. Denote this
explicit joint construction by $\mathbb T_\mu$. It retains all
selection/component dependence; the fresh mark has exactly the next
population's measurement law. Define

$$
\begin{aligned}
 R^{\rm move}_\mu
   &=\mathbb E_{\mathbb T_\mu}
          [f_\mu(t')-f_\mu(t_*)],\\
 R^{\rm env}_\mu
   &=\mathbb E_{\mathbb T_\mu}
          [f_\nu(t')-f_\mu(t')],\qquad
 R_\mu=R^{\rm move}_\mu+R^{\rm env}_\mu.
\end{aligned}
$$

Here $R^{\rm move}_\mu$ includes source-to-output physical change and
fresh companion sampling, and $R^{\rm env}_\mu$ is the change of the
normalization environment at the same physical/measurement pair.
These are finite integrals of the specified kernel, rather than
constants defined by an unknown convergence rate. They satisfy

$$
 |R^{\rm move}_\mu|\le\Delta_F,\qquad
 |R^{\rm env}_\mu|\le\Delta_F,\qquad
 |R_\mu|\le\Delta_F,
$$

and the exact complete-update identity is

$$
\boxed{\qquad
 J(\mathcal F_h\mu)-J(\mu)=G_\mu+R_\mu.
 \qquad}
$$

Every quantity is independent of particle number. The gain uses the
actual reward/diversity normalizations
$(\sigma_r,\sigma_s,A_r,A_s,\eta_r,\eta_s,p_r,p_s)$, comparison
features $(R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg})$,
companion widths $(\epsilon_D,\epsilon_C)$, separation floor
$\delta_D$, reward $R$, and gate parameters $(s_c,\epsilon_c)$
through the displayed marked kernel. The refresh construction retains
also the dimension $d$, step $h$, friction $\gamma$, thermostat
$b_O$, final position noise $\sigma_x$, jitter $\sigma_J$, cap
$V_{\max}$, collision multiplier $\alpha_{\rm col}$ and force
$-\nabla U$ through the actual full update. In particular none of
these effects is hidden in a purported vanishing finite-population
error.

For any actual stationary phase $\pi$ with admissible normalizations,
$\mathcal F_h\pi=\pi$ implies

$$
 R^{\rm env}_\pi=0,\qquad R_\pi=-G_\pi,
$$

and hence gives the phase-centered identity

$$
\boxed{\qquad
 J(\mathcal F_h\mu)-J(\mu)
   =(G_\mu-G_\pi)+(R_\mu-R_\pi).
 \qquad}
$$

If $\overline p_\pi>0$, its refresh contribution is strictly negative:
$R_\pi\le-s_c(F_*+\epsilon_c)\overline p_\pi^2<0$.
Thus positive cloning gain alone is not a monotonicity statement for
$J$ along the nonlinear evolution. The centered identity retains the
cancellation which a phase-attraction proof would have to estimate;
it does not assert such attraction.
:::

:::{prf:proof}
The physical marginal of $\widehat\eta_\mu$ is $\mu$. Therefore
integrating $w_C(z_t,z_u)$ over $\widehat\eta_\mu(du)$ gives
$Z_C(\mu;z_t)$, proving the donor-density normalization. In the actual
rooted construction the root either retains its source type or accepts
the sampled donor. Consequently, for every bounded measurable
$\varphi:T\to\mathbb R$,

$$
 \lambda_\mu\varphi-\widehat\eta_\mu\varphi
 =\iint\beta_\mu(t,u)[\varphi(u)-\varphi(t)]
               \widehat\eta_\mu(dt)\widehat\eta_\mu(du).
$$

Taking $\varphi=f_\mu$ proves the formula for $G_\mu$.
Write $\Delta=f_\mu(u)-f_\mu(t)$ and $a=a_\mu(t,u)$.
If $a=0$, then $a\Delta=0$. If $0<a<1$, the gate formula gives
$\Delta=s_c(f_\mu(t)+\epsilon_c)a$; if $a=1$, it gives
$\Delta\ge s_c(f_\mu(t)+\epsilon_c)$. Thus in every case

$$
 a\Delta\ge s_c(F_*+\epsilon_c)a^2.
$$

Integrate against the probability measure
$\widehat\eta_\mu(dt)k_\mu(t,u)\widehat\eta_\mu(du)$.
Jensen's inequality gives the lower bound by
$s_c(F_*+\epsilon_c)\overline p_\mu^2$. On accepted pairs,
$0<\Delta\le\Delta_F$, giving the upper bound. The fitness range
follows immediately from
$\eta_b\le A_b/(1+e^{-q})+\eta_b\le A_b+\eta_b$ and
$p_b\ge0$, including the convention $a^0=1$ for $a>0$.

The full rooted update has final marginal $z'\sim\nu$ by definition.
The fresh conditional draw therefore makes $t'$ have marginal
$\widehat\eta_\nu$. Its other retained variable has marginal
$t_*\sim\lambda_\mu$, without asserting independence between them.
It follows that

$$
 R_\mu
 =\mathbb E_{\mathbb T_\mu}[f_\nu(t')-f_\mu(t_*)]
 =J(\nu)-\lambda_\mu f_\mu.
$$

Combining this with the gain formula proves the complete-update
identity. Each of the three displayed refresh integrands is a
difference of two values in $[F_*,F^*]$, which proves the bounds; in
particular the bound on their sum uses the telescoped integrand, not
the triangle inequality on its two parts. The construction integrates
the actual full component law and all its configured random variables,
so no independence approximation or particle-count limit is used.

Finally, at $\nu=\mu=\pi$, the environment fitness functions agree
pointwise, making $R^{\rm env}_\pi=0$. The complete-update identity
has zero left-hand side and yields $R_\pi=-G_\pi$. Subtract this zero
identity from the one at $\mu$, and use the proved positive gain lower
bound when $\overline p_\pi>0$.
:::


:::{prf:definition} A bounded entropy observable for empirical populations
:label: def-slcec-coarse-entropy

Fix a measurable partition $(A_i)_{i=1}^M$ of the full capped phase
space, $2\le M<\infty$, and a declared reference probability $\pi$
with $\pi_i=\pi(A_i)>0$. The reference is a stationary phase only
when its actual fixed-point equation has been proved. Put
$\pi_{\min}=\min_i\pi_i$, $\pi_{\max}=\max_i\pi_i$.
For any population law $\mu$, write $p_i=\mu(A_i)$ and, for
$\varepsilon>0$, set
$$
 p_i^\varepsilon=\frac{p_i+\varepsilon\pi_i}{1+\varepsilon},\qquad
 \mathcal H_\varepsilon(\mu)
 =\sum_{i=1}^M p_i^\varepsilon\log\frac{p_i^\varepsilon}{\pi_i}.
$$
This is finite for atomic empirical laws and obeys
$$
 0\le\mathcal H_\varepsilon(\mu)\le
 B_\varepsilon:=\frac{\log(1/\pi_{\min})}{1+\varepsilon}.
$$
Define the actual signed population increment
$$
 \mathcal D_\varepsilon(\mu)
 =\mathcal H_\varepsilon(\mu)
                 -\mathcal H_\varepsilon(\mathcal F_h\mu).
$$
The vector of cell masses is not presumed Markovian: the second term
uses the actual full population map at the entire entering law.
:::

:::{prf:theorem} Quantitative transfer of the actual signed coarse-entropy increment
:label: thm-slcec-drift-transfer

Use the actual conservative all-alive canonical full-step kernel, $N\ge2$,
and the constants $A,B_*$ evaluated in {prf:ref}`def-slc-empirical-metric`
with $m_*=1$. Thus the bounded measurable-test estimate of
{prf:ref}`thm-chaos-canonical-quantitative-bias` gives
$G=A+4B_*^2$, independent of $N$. Set
$$
 L_\varepsilon=\log\left(1+\frac1{\varepsilon\pi_{\min}}\right),
$$
$$
 e_{N,\varepsilon}
 =\frac{M L_\varepsilon}{2(1+\varepsilon)}\sqrt{G/N}
       +\frac{MG}{2N\varepsilon(1+\varepsilon)\pi_{\min}}.
$$
For every admissible entering configuration $S$,
$$
 \left|\mathbb E[\mathcal H_\varepsilon(L_N(S'))\mid S]
       -\mathcal H_\varepsilon(\mathcal F_hL_N(S))\right|
 \le\min\{B_\varepsilon,e_{N,\varepsilon}\}.
$$
Consequently, for any initialization and integer $T\ge1$,
$$
 \left|\frac1T\sum_{n=0}^{T-1}
          \mathbb E\mathcal D_\varepsilon(L_N(S_n))\right|
 \le\frac{B_\varepsilon}{T}+e_{N,\varepsilon}.
$$
At finite-particle stationarity the endpoint term vanishes and
$|\mathbb E\mathcal D_\varepsilon(L_N(S))|\le e_{N,\varepsilon}$.
For a fixed partition and reference, the explicit choice
$\varepsilon_N=N^{-1/2}$ gives
$e_{N,\varepsilon_N}=O(N^{-1/2}\log N)$ with the displayed constants.
These estimates transfer the actual signed increment; they do not
assume or assert that $\mathcal D_\varepsilon$ is nonnegative.
:::

:::{prf:proof}
Condition on $S$ and put
$q_i=(\mathcal F_hL_N(S))(A_i)$,
$\widehat q_i=L_N(S')(A_i)$, $\Delta_i=\widehat q_i-q_i$.
The bounded measurable indicator is permitted in the source theorem,
so $\mathbb E\Delta_i^2\le G/N$, and $\sum_i\Delta_i=0$.
For the smooth function of the mass vector defining
$\mathcal H_\varepsilon$, the gradient and Hessian are
$$
 \partial_i\mathcal H_\varepsilon(p)
 =\frac{1+\log[(p_i+\varepsilon\pi_i)/((1+\varepsilon)\pi_i)]}
                 {1+\varepsilon},\qquad
 \partial_{ij}\mathcal H_\varepsilon(p)
 =\frac{\mathbf1_{i=j}}{(1+\varepsilon)(p_i+\varepsilon\pi_i)}.
$$
The gradient's coordinate oscillation is at most
$L_\varepsilon/(1+\varepsilon)$, and the Hessian is bounded above by
$[\varepsilon(1+\varepsilon)\pi_{\min}]^{-1}I$ throughout the simplex.
Taylor's integral remainder therefore lies between zero and
$\sum_i\Delta_i^2/[2\varepsilon(1+\varepsilon)\pi_{\min}]$.
Subtracting the midpoint of the gradient's coordinate range, which
costs nothing because $\sum_i\Delta_i=0$, bounds the expected linear
term by
$$
 \frac{L_\varepsilon}{2(1+\varepsilon)}
                    \sum_i|\mathbb E\Delta_i|
 \le\frac{M L_\varepsilon}{2(1+\varepsilon)}\sqrt{G/N}.
$$
Add the expected remainder to obtain the stated error. Convexity of
relative entropy gives
$\mathcal H_\varepsilon(p)\le H(p\mid\pi)/(1+\varepsilon)
\le B_\varepsilon$, proving the clipping.

Write the conditional expectation error as $r_n$, with
$|r_n|\le e_{N,\varepsilon}$. Then
$$
 \mathbb E\mathcal D_\varepsilon(L_N(S_n))
 =\mathbb E[\mathcal H_\varepsilon(L_N(S_n))
              -\mathcal H_\varepsilon(L_N(S_{n+1}))]+\mathbb Er_n.
$$
Sum and telescope. The endpoint difference is at most $B_\varepsilon$
in absolute value and vanishes at stationarity. Substitution of
$\varepsilon_N=N^{-1/2}$ proves the displayed asymptotic error rate.
No independence among walkers, successive updates or cell counts was
used.
:::

:::{prf:proposition} Exact full-entropy residual and the smoothing error
:label: prop-slcec-full-residual

For population laws $\mu\ll\pi$ with finite $H(\mu\mid\pi)$, set
$$
 \mathcal R_{\mathcal P}(\mu)
 =\sum_{i:p_i>0}p_iH(\mu(\cdot\mid A_i)\mid\pi(\cdot\mid A_i)),
$$
$$
 \Delta_\varepsilon(\mu)
 =\sum_i p_i\log(p_i/\pi_i)-\mathcal H_\varepsilon(\mu)\ge0.
$$
Then
$$
 H(\mu\mid\pi)=\mathcal H_\varepsilon(\mu)
                   +\mathcal R_{\mathcal P}(\mu)+\Delta_\varepsilon(\mu).
$$
Define $h_2(t)=-t\log t-(1-t)\log(1-t)$ and
$$
 \omega_M(t)=\begin{cases}
 h_2(t)+t\log(M-1),&0\le t\le1-1/M,\\
 \log M,&1-1/M<t\le1.
 \end{cases}
$$
With $t_\varepsilon=\varepsilon/(1+\varepsilon)$,
$$
 0\le\Delta_\varepsilon(\mu)
 \le S_\varepsilon:=\omega_M(t_\varepsilon)
                  +t_\varepsilon\log(\pi_{\max}/\pi_{\min}).
$$
In particular $S_{N^{-1/2}}=O(N^{-1/2}\log N)$ for a fixed partition.
If also the actual full output has finite entropy, its exact signed
full dissipation is
$$
 H(\mu\mid\pi)-H(\mathcal F_h\mu\mid\pi)
 =\mathcal D_\varepsilon(\mu)
   +\mathcal R_{\mathcal P}(\mu)-\mathcal R_{\mathcal P}(\mathcal F_h\mu)
   +\Delta_\varepsilon(\mu)-\Delta_\varepsilon(\mathcal F_h\mu).
$$
The absolute smoothing contribution is at most $S_\varepsilon$.
The within-cell residual is not set to zero or bounded for empirical
atomic laws.

For an actual density ratio $r=d\mu/d\pi$, certified bounds
$0<l_i\le r\le u_i<\infty$ on selected cells give
$$
 p_iH(\mu(\cdot\mid A_i)\mid\pi(\cdot\mid A_i))
 \le p_i\log(u_i/l_i).
$$
On a remaining tail cell its exact conditional-entropy integral must be
retained or separately bounded. These quantities give a concrete
coarse-to-full error budget when the density estimates are available.
:::

:::{prf:proof}
Disintegrate the density ratio on each cell:
$r=(p_i/\pi_i)\,d\mu(\cdot\mid A_i)/d\pi(\cdot\mid A_i)$.
Integrating its logarithm proves the entropy chain rule. Convexity gives
$\mathcal H_\varepsilon(\mu)\le H(p\mid\pi)/(1+\varepsilon)$,
hence the nonnegative smoothing defect. The total variation distance
between $p$ and $p^\varepsilon$ is at most $t_\varepsilon$.
The finite-alphabet entropy continuity bound is $\omega_M$; it follows
by maximal coupling, whose mismatch indicator costs $h_2(t)$ and whose
conditional alternative has at most $M-1$ choices, then by the entropy
upper bound $\log M$. The cross-entropy term against $\pi$ changes by
at most $t_\varepsilon$ times the oscillation of $\log\pi_i$.
These prove $S_\varepsilon$. Subtract the two exact chain rules to
obtain the signed full-dissipation identity. Both smoothing defects lie
in $[0,S_\varepsilon]$, so their difference has absolute value at most
$S_\varepsilon$. Finally the conditional density ratio on a cell is
$r/(p_i/\pi_i)$, with $p_i/\pi_i\in[l_i,u_i]$; its logarithm is at
most $\log(u_i/l_i)$. Integrate under the conditional law.
:::

:::{prf:corollary} Transfer under actual survival conditioning
:label: cor-slcec-conditioning

For the canonical terminal-box kernel use the actual conditioned laws
$\eta_n$ and the good-input set $G_N$ of
{prf:ref}`def-chaos-survival-filter`. Evaluate $A,B_*$, and therefore
$G$ and $e_{N,\varepsilon}$, at $m_*=a_0/4$ instead of $1$.
The partition and reference are on the complete marked state space.
For $n\ge1$ and $N\ge N_{\rm surv}$, the signed drift obeys

$$
\left|\mathbb E_{\eta_n}\mathcal D_\varepsilon(L_N)
 -\left(\mathbb E_{\eta_n}\mathcal H_\varepsilon(L_N)
        -\mathbb E_{\eta_{n+1}}\mathcal H_\varepsilon(L_N)\right)\right|
 \le e_{N,\varepsilon}+2B_\varepsilon\delta_N.
$$

Consequently, for $T\ge1$,

$$
\left|\frac1T\sum_{n=1}^{T}
       \mathbb E_{\eta_n}\mathcal D_\varepsilon(L_N)\right|
\le \frac{B_\varepsilon}{T}
      +e_{N,\varepsilon}+2B_\varepsilon\delta_N.
$$

For a QSD the endpoint term is zero. The target in
$\mathcal D_\varepsilon$ remains the actual marked population map
$\mathcal F_h$, evaluated before conditioning a finite swarm. No
rowwise renormalized transition is iterated in this assertion.
:::

:::{prf:proof}
On $G_N$ the preceding Taylor proof applies with the stated alive floor.
On its complement both observable values lie in $[0,B_\varepsilon]$,
so their difference is at most $B_\varepsilon$. The proved uniform bound
$\eta_n(G_N^c)\le\delta_N$ therefore gives error at most
$e_{N,\varepsilon}+B_\varepsilon\delta_N$ between the actual
unconditioned next-output expectation from input law $\eta_n$ and the
population-map observable. The actual normalized survival law
$\eta_{n+1}$ differs from that entire output law by at most
$\delta_N$ in TV, by {prf:ref}`thm-chaos-survival-uniform-floor`.
Since the observable has oscillation at most $B_\varepsilon$, this
adds at most $B_\varepsilon\delta_N$. Summing telescopes with an
endpoint bounded by $B_\varepsilon$. For a QSD the two endpoints
coincide. Neither step divides by the survival probability of the
complete history.
:::

(sec-slcfi-explicit-centered-fisher)=
### 17.5. Explicit full-kernel centered Fisher estimates

:::{div} feynman-prose
The score is the spatial and velocity gradient of a log density. Here we
calculate it from the actual kinetic density already established in the
book, retaining that density formula's conditions. The stationary output
score is the conditional average of these kinetic scores. Subtracting this
average gives an exact variance subtraction, so stationary fluctuations do
not become a constant error charged against convergence.

There is one coordinate detail to keep straight. Removing the velocity cap
simplifies the density, but also changes how a physical gradient is measured.
The inverse cap Jacobian therefore stays inside the Fisher matrix. With that
matrix retained, the resulting Gaussian integrals quantify the output's
relative Fisher information through discrepancies between actual prepared
laws, including their cloning and collision dependence.
:::

:::{prf:definition} Actual preparation and kinetic score regime
:label: def-slcfi-score-regime

These estimates concern the actual canonical all-alive preparation and kinetic kernels. Fix an actual stationary phase $\pi$, and put $\rho=\pi C_\pi$, $\lambda=\mu C_\mu$. Thus $\rho K=\pi$ and $Q=\lambda K=\mathcal F_h\mu$. The constants below use the existing density regime of {prf:ref}`lem-slcpd-density`: $q,s>0$, $F\in C^2$, $L_1=\sup\|DF\|<\infty$, and $\ell=c^2L_1<1$. They do not presume that either preparation or kinetics separately preserves $\pi$.


The remaining symbols use {prf:ref}`def-slc-parameter-register`.
In density and score formulas, $\pi(y)$ denotes the stationary physical
law pulled back through the final cap to pre-cap coordinates.
All derivatives below are output-coordinate derivatives. The stationary
kernel-mixture formulas are used on their differentiation-under-the-integral
domain. The explicit finite score bound below supplies the required weak
first-derivative integrability when its prepared fourth-moment bound is finite.
For classical derivatives, retain the corresponding local dominated-derivative
envelopes. No derivative of a discrete gate at a tie is asserted.
:::

:::{prf:lemma} Actual kinetic score with the physical cap metric
:label: lem-slcfi-kinetic-score

Use the parameter register
$$
c=h/2,\quad a=e^{-\gamma h},\quad
q^2=b_O^2(1-e^{-2\gamma h})/(2\gamma),\quad s^2=\sigma_x^2h,
$$
with the stated continuous $\gamma=0$ convention. Write the prepared state as $z=(x,v)$, $|v|\le V_c=(1+2|\alpha_{\rm col}|)V_{\max}$, and use pre-cap output coordinates $y=(X,w)$. The actual kinetic variables are
$$
x_1=x+c[v+cF(x)],\quad m=a[v+cF(x)],\quad
Z=m+q\xi,\quad X=x_1+cZ+s\zeta,\quad
w=T(Z)=Z+cF(x_1+cZ),
$$
where $\xi,\zeta$ are independent standard $d$-dimensional Gaussians. Set $D=(1-\ell)^{-1}$ and $H_2(u)=\|D^2F(u)\|$. The exact density from {prf:ref}`lem-slcpd-density` has scores
$$
\nabla_X\log k=-\zeta/s,
$$
$$
\nabla_w\log k
=(DT(Z))^{-T}\left[-\xi/q+c\zeta/s
                         -\nabla_Z\log\det DT(Z)\right].
$$
The determinant is positive because $\|DT-I\|\le\ell<1$.
For every direction $e$,
$$
|D_e\log\det DT(Z)|
=|\operatorname{tr}[(DT)^{-1}c^3D^2F(x_1+cZ)[e]]|
\le d c^3D H_2(x_1+cZ)|e|.
$$
Consequently
$$
|\nabla_y\log k|^2
\le A_\xi|\xi|^2+A_\zeta|\zeta|^2+A_2H_2(x_1+cZ)^2,
$$
with fully evaluated coefficients
$$
A_\xi=3D^2/q^2,\qquad
A_\zeta=(1+3D^2c^2)/s^2,\qquad A_2=3d^2c^6D^4.
$$

Let $G_{\rm phys}$ be the constant physical-coordinate matrix in the modified entropy, with largest eigenvalue $g_+$. For the actual cap $C_V(w)=Vw/(V+|w|)$, $V=V_{\max}>0$, set
$$
\mathcal A(y)=\operatorname{diag}(I,DC_V(w))^{-1}
 G_{\rm phys}\operatorname{diag}(I,DC_V(w))^{-T}.
$$
Its norm is at most $g_+(1+|w|/V)^4$: the tangential cap singular value is $(1+|w|/V)^{-1}$ and the radial one is $(1+|w|/V)^{-2}$. Thus the actual physical Fisher information is calculated using $\mathcal A(y)$ in pre-cap coordinates. In particular it is **not** legitimate to keep a constant physical matrix unchanged after removing the cap.

Define the explicitly evaluated Gaussian integrand
$$
\Theta(z,\xi,\zeta)=g_+(1+|T(m+q\xi)|/V)^4
 \left[A_\xi|\xi|^2+A_\zeta|\zeta|^2
                      +A_2H_2(x_1+c(m+q\xi))^2\right].
$$
Then
$$
S(z):=\int k(y\mid z)|\nabla_y\log k(y\mid z)|_{\mathcal A(y)}^2dy
\le\mathbb E\Theta(z,\xi,\zeta).
$$
All right-hand quantities involve the actual force and Gaussian innovations. A regional Hessian profile $H_2\le H_j$ on regions $D_j$ gives the explicit bound
$$
\mathbb E\left[(1+|w|/V)^4H_2(x_1+cZ)^2\right]
\le\sum_jH_j^2\mathbb E\left[(1+|w|/V)^4
                           \mathbf1_{\{x_1+cZ\in D_j\}}\right].
$$
The nonnegative series includes the exterior regions and every Gaussian excursion; it is not truncated without a remainder. Infinite series make the bound uninformative, not invalid.
:::

:::{prf:proof}
For the pre-cap density in {prf:ref}`lem-slcpd-density`, differentiate the two Gaussian log densities and the log Jacobian. The inverse-function derivative is $(DT)^{-1}$; $\|DT-I\|\le\ell$ gives its norm bound $D$. The derivative of the log determinant is the displayed trace. The multilinear operator norm for $D^2F$ bounds that trace in every unit direction by $dc^3D H_2$, and multiplying by the inverse derivative introduces the second factor $D$. The inequality $|u_1+u_2+u_3|^2\le3\sum_j|u_j|^2$ gives precisely $A_\xi,A_\zeta,A_2$, including the additional positional score $|\zeta|^2/s^2$.

The cap's radial and tangential derivatives give the stated inverse-Jacobian matrix norm. The Jacobian terms in the two transformed log densities cancel in their difference, so this is exactly the physical relative Fisher metric. Integrating the resulting nonnegative score bound against the actual independent Gaussian innovations proves the $S$ estimate. The regional Hessian bound follows pointwise and then by Tonelli; the Gaussian excursion outside any bounded region remains in the stated sum.
:::

:::{prf:lemma} Evaluated Gaussian score moments
:label: lem-slcfi-gaussian-moments

Suppose the declared force envelopes give $|F(x)|\le g_0+g_1|x|$ and $H_2\le L_2<\infty$. For a prepared state $z=(x,v)$ define
$$
R_1(z)=(1+c^2g_1)|x|+c|v|+c^2g_0,
\quad M(z)=a(|v|+cg_0+cg_1|x|),
$$
$$
W_0(z)=(1+c^2g_1)M(z)+cg_0+cg_1R_1(z),
\quad W_1=(1+c^2g_1)q.
$$
The actual pre-cap velocity satisfies $|w|\le W_0(z)+W_1|\xi|$.
Put $u(z)=1+W_0(z)/V$, $v_1=W_1/V$ and
$$
m_{d,j}=2^{j/2}\Gamma((d+j)/2)/\Gamma(d/2),\qquad m_{d,0}=1,
$$
$$
\mathsf M_k(u,v_1)=\sum_{j=0}^{4}\binom4j u^{4-j}v_1^j m_{d,j+k}
\quad(k=0,2).
$$
Independence of $\zeta$ from $\xi$ and $\mathbb E|\zeta|^2=d$ give
$$
\boxed{\quad S(z)\le s_G(z):=
 g_+\left[A_\xi\mathsf M_2(u(z),v_1)
              +(dA_\zeta+A_2L_2^2)\mathsf M_0(u(z),v_1)\right].\quad}
$$
This coefficient retains the step, friction, OU noise, final position noise, cap, collision velocity bound, force growth and Hessian profile. Preparation jitter and selection enter through its actual input law $\rho$ or $\lambda$.

The stationary preparation average is also explicit from a fourth-moment budget. Suppose $\rho|x|^4\le M_4^C$, and set $R_0=1$, $R_j=(M_4^C)^{j/4}$ for $1\le j\le4$. Using $|v|\le V_c$, put
$$
 w_{00}=(1+c^2g_1)a(V_c+cg_0)+cg_0+c^2g_1V_c+c^3g_1g_0,
\quad w_{01}=cg_1(1+c^2g_1)(1+a),
$$
$$
 u_0=1+w_{00}/V,\quad u_1=w_{01}/V,
\quad U_j=\sum_{i=0}^j\binom ji u_0^{j-i}u_1^iR_i.
$$
Then replace $u^{4-j}$ by $U_{4-j}$ in the two sums defining $\mathsf M_k$. The resulting numerical expression, denoted $\overline S_G$, satisfies
$$
\int S(z)\rho(dz)\le\int s_G(z)\rho(dz)\le\overline S_G.
$$
No bounded force-center profile is required. The budget $M_4^C$ is the actual post-preparation moment, including jitter; it must be taken from the established selection/moment estimates, not silently replaced by the completed-step moment.
:::

:::{prf:proof}
The force-growth inequality gives $|x_1|\le R_1$, $|m|\le M$, and $|T(m+q\xi)|\le W_0+W_1|\xi|$. Raise $u+v_1|\xi|$ to the fourth power by the binomial theorem. Integration against $|\xi|^k$ gives $\mathsf M_k$ because the stated Gamma expression is the radial standard-Gaussian moment. The $\zeta$ term contributes $dA_\zeta\mathsf M_0$, using independence. The Hessian term contributes $A_2L_2^2\mathsf M_0$. This proves the pointwise coefficient.

Insert $|v|\le V_c$ into $W_0$ and collect its constant and $|x|$ coefficients to obtain $w_{00},w_{01}$. For $0\le j\le4$, Holder's inequality gives $\rho|x|^j\le R_j$. Expanding $(u_0+u_1|x|)^j$ proves the bound $\rho u^j\le U_j$. All polynomial coefficients are nonnegative, so substituting these bounds into the two sums proves $\overline S_G$.
:::

:::{prf:theorem} Stationary cancellation and full centered Fisher production
:label: thm-slcfi-centered-production

Write $s_z(y)=\nabla_y\log k(y\mid z)$ and
$\bar s(y)=\nabla_y\log\pi(y)$. Under the stationary joint law $\rho(dz)k(y\mid z)dy$, differentiation gives
$$
\bar s(Y)=\mathbb E[s_Z(Y)\mid Y].
$$
This conditional identity remains valid with the output-dependent positive matrix $\mathcal A(Y)$. Therefore the centered profile
$$
J_G(z)=\int k(y\mid z)|s_z(y)-\bar s(y)|_{\mathcal A(y)}^2dy
$$
has the sharper averaged estimate
$$
\boxed{\quad
\int J_G(z)\rho(dz)
=\int S(z)\rho(dz)-\int\pi(y)|\bar s(y)|_{\mathcal A(y)}^2dy
\le\overline S_G.\quad}
$$
This is a genuine subtraction of the stationary score, rather than the two-term triangle inequality. It uses only invariance of the complete map through $\rho K=\pi$.

If the actual prepared law satisfies $\lambda=f\rho$ with $f>0$, define
$W_f(z)=(f(z)-1)^2/f(z)$. The centered Fisher calculation and weighted Cauchy–Schwarz give
$$
I_{G_{\rm phys}}(Q\mid\pi)\le\int W_f(z)J_G(z)\rho(dz).
$$
In particular an evaluated upper envelope $W_f\le w_*$ yields the fully specified bound
$$
\boxed{I_{G_{\rm phys}}(\mathcal F_h\mu\mid\pi)
                 \le w_*\overline S_G.}
$$
This is homogeneous in the preparation discrepancy: $w_*=0$ at the actual phase. A direct bound $|f-1|\le e<1$ gives $w_*\le e^2/(1-e)$. No comparison of this discrepancy to input Fisher information is presumed.
:::

:::{prf:proof}
Differentiate $\pi(y)=\int k(y\mid z)\rho(dz)$ to obtain $\bar s(y)=\mathbb E[s_Z(Y)\mid Y=y]$. Conditional variance, with the deterministic matrix $\mathcal A(Y)$ inside that conditioning, gives the displayed exact subtraction. Jensen and the preceding integrable score bound justify the averaged quantities and imply $\int J_Gd\rho\le\overline S_G$.

For $u=Q/\pi$, stationarity gives
$$
\nabla u(y)=\pi(y)^{-1}\int k(y\mid z)[s_z(y)-\bar s(y)](f(z)-1)\rho(dz).
$$
Apply weighted Cauchy--Schwarz with weights $k(y\mid z)f(z)$ to this centered numerator. The first factor is $Q(y)$; division by $Q(y)$ and integration over $y$ gives $I_{G_{\rm phys}}(Q\mid\pi)\le\int W_fJ_Gd\rho$. Tonelli permits an infinite right-hand side. A uniform $W_f$ envelope and the covariance bound give the stated coefficient. The bound $e^2/(1-e)$ follows directly from $|f-1|\le e<1$.
:::

:::{prf:lemma} Regional prepared-law and output-tail bounds
:label: lem-slcfi-regional-production

Partition preparation space into $C_j$ and output space into $O_i$. Put
$$
p_j(y)=\frac{\int_{C_j}k(y\mid z)\rho(dz)}{\pi(y)},
\quad \Gamma_i=\int\rho(dz)\int_{O_i}k(y\mid z)
                      |s_z(y)|_{\mathcal A(y)}^2dy,
\quad S_j=\int_{C_j}S(z)\rho(dz).
$$
If $W_f\le w_j$ on $C_j$ and $p_j\le r_{ji}$ on $O_i$, then
$$
\boxed{\quad
I_{G_{\rm phys}}(Q\mid\pi)
\le2\sum_jw_j S_j+2\sum_{j,i}w_jr_{ji}\Gamma_i.\quad}
$$
Indeed $|s_z-\bar s|_{\mathcal A}^2\le2|s_z|_{\mathcal A}^2+2|\bar s|_{\mathcal A}^2$, while conditional Jensen gives $\pi|\bar s|_{\mathcal A}^2\le\int\rho(dz)k|s_z|_{\mathcal A}^2$. Multiply by $p_j$, integrate and apply its regional upper bounds. The nonnegative sums allow infinite profiles without an invalid truncation. Use the preceding global covariance bound when it is sharper.

Every term admits a structural bound:
$$
S_j\le\int_{C_j}s_G(z)\rho(dz),\qquad
\Gamma_i\le\int\rho(dz)\mathbb E[
\mathbf1_{\{Y(z,\xi,\zeta)\in O_i\}}\Theta(z,\xi,\zeta)].
$$
These are actual Gaussian-response integrals, including the force and cap. If only global bounds are desired use $\Gamma_i\le\overline S_G$; when summing all output cells use the stronger $\sum_i\Gamma_i\le\overline S_G$.

For completeness the posterior envelopes can be evaluated directly. On an input cell $|x|\le R_j$, $|v|\le V_c$ and output cell $|X|\le R_i'$, $|w|\le W_i'$, set
$$
R_{1j}=(1+c^2g_1)R_j+cV_c+c^2g_0,
\quad M_j=a(V_c+cg_0+cg_1R_j),
\quad Z_{ij}=D[W_i'+c(g_0+g_1R_{1j})].
$$
The inverse kinetic map obeys $|Z|\le Z_{ij}$, by its co-Lipschitz bound and $T(0)=cF(x_1)$. Thus
$$
k_{ij}^-=(2\pi q^2)^{-d/2}(2\pi s^2)^{-d/2}(1+\ell)^{-d}
 \exp\left[-\frac{(Z_{ij}+M_j)^2}{2q^2}
           -\frac{(R_i'+R_{1j}+cZ_{ij})^2}{2s^2}\right],
$$
$$
k^+=(2\pi q^2)^{-d/2}(2\pi s^2)^{-d/2}(1-\ell)^{-d}
$$
are valid pointwise lower and upper density bounds. If $m_j^-$ and $m_j^+$ bound $\rho(C_j)$, any positive denominator gives
$$
r_{ji}=\min\left\{1,\frac{m_j^+k^+}{\sum_lm_l^-k_{il}^-}\right\}.
$$
The denominator sum may use any finite collection of bounded input cells; omitted cells have nonnegative contribution. Unbounded output tails can use $r_{ji}=1$ and retain their Gaussian-response integrals rather than asserting a nonexistent uniform positive density lower bound.
:::

:::{prf:proof}
The pointwise square bound for $s_z-\bar s$, followed by conditional Jensen for $\bar s$, yields
$$
\int_{C_j}J_Gd\rho\le2S_j+2\sum_i r_{ji}\Gamma_i.
$$
Multiply by the nonnegative envelope $w_j$ and sum. The $S_j$ and $\Gamma_i$ bounds follow by integrating the explicit Gaussian score majorant, with the output indicator retained before integration.

For the posterior estimate, $|T(Z)-T(0)|\ge(1-\ell)|Z|$ and $|T(0)|\le c(g_0+g_1R_{1j})$ prove $|Z|\le Z_{ij}$. Insert this and the bounds on $x_1,m$ into the exact Gaussian density, and use $(1-\ell)^d\le\det DT\le(1+\ell)^d$. This gives $k_{ij}^-$ and $k^+$. Integrate the numerator of $p_j$ using its upper bound and its denominator using the sum of lower bounds. Dropping unlisted positive contributions in that denominator is valid. These are precisely the displayed $r_{ji}$.
:::

:::{prf:theorem} Coupling bound without prepared-law absolute continuity
:label: thm-slcfi-coupling-production

Keep the score regime of {prf:ref}`def-slcfi-score-regime`. Let $\Gamma$
be any coupling of the actual prepared laws $\lambda=\mu C_\mu$ and
$\rho=\pi C_\pi$. The following bound does not require $\lambda\ll\rho$.
Require the differentiation-under-the-integral domain for both
$Q=\lambda K$ and $\pi=\rho K$. A sufficient quantitative condition
is $\int s_G\,d\lambda+\int s_G\,d\rho<\infty$, with
$G_{\rm phys}\succ0$ and $s_G$ from
{prf:ref}`lem-slcfi-gaussian-moments`; the displayed growth regime reduces
this to finite prepared fourth moments. Alternatively retain proved
local weak-derivative envelopes for both mixtures. A stationary moment
bound alone is not being applied to the evolving preparation law.
For prepared states $z,z'$ define the actual centered kernel derivative

$$
R_z(y)=\nabla_y k(y\mid z)-k(y\mid z)\bar s(y),
\qquad \bar s(y)=\nabla_y\log\pi(y),
$$

and its explicitly specified pair profile

$$
\Xi_G(z,z')=
\int\frac{|R_z(y)-R_{z'}(y)|_{\mathcal A(y)}^2}{k(y\mid z)}\,dy.
$$

The Gaussian density in the present regime is strictly positive, so
this denominator is positive. Then

$$
\boxed{\quad
I_{G_{\rm phys}}(\mathcal F_h\mu\mid\pi)
\le\int\Xi_G(z,z')\,\Gamma(dz,dz'),\qquad
\Xi_G(z,z)=0.
\quad}
$$

This is an actual kinetic Gaussian integral: with $Y_z(\xi,\zeta)$
the pre-cap output in {prf:ref}`lem-slcfi-kinetic-score`, it equals

$$
\Xi_G(z,z')=\mathbb E_{\xi,\zeta}
\left|
 s_z(Y_z)-\bar s(Y_z)
 -\frac{k(Y_z\mid z')}{k(Y_z\mid z)}
       [s_{z'}(Y_z)-\bar s(Y_z)]
\right|_{\mathcal A(Y_z)}^2.
$$

Both densities and kinetic scores are given by the preceding formulas;
the reference score is the stationary kernel-mixture conditional mean
in {prf:ref}`thm-slcfi-centered-production`. No unknown mixing coefficient
is present. For declared prepared pair regions $C_j\times C_l$, any
proved Gaussian-integral envelopes $\Xi_G\le X_{jl}$ give

$$
I_{G_{\rm phys}}(\mathcal F_h\mu\mid\pi)
\le\sum_{j,l}X_{jl}\Gamma(C_j\times C_l).
$$

The pair bound is allowed to be infinite. Finite single-kernel score
moments alone do not assert finiteness of the density-ratio integral
in $\Xi_G$. Tail regions and their integral contributions must be retained.
:::

:::{prf:proof}
The actual output density satisfies $Q(y)=\int k(y\mid z)\Gamma(dz,dz')$.
Stationarity gives $\int R_{z'}(y)\rho(dz')=0$, so

$$
\nabla Q(y)-Q(y)\bar s(y)
=\int[R_z(y)-R_{z'}(y)]\Gamma(dz,dz').
$$

For each $y$, weighted Cauchy--Schwarz in the Hilbert norm induced by
$\mathcal A(y)$ yields

$$
\frac{|\nabla Q-Q\bar s|_{\mathcal A}^2}{Q}
\le\int\frac{|R_z-R_{z'}|_{\mathcal A}^2}{k(y\mid z)}
                                           \Gamma(dz,dz').
$$

Integrate $y$ and apply Tonelli. The left side integrates to the actual
physical relative Fisher information, with the inverse-cap metric already
included. On the diagonal the integrand is identically zero. Finally
factor $k(y\mid z)$ out of the numerator and integrate under its actual
Gaussian generation law to obtain the displayed expectation. Regional
upper bounds then sum by nonnegativity, with no independence assertion
about the coupling coordinates.
:::

:::{prf:corollary} Transfer using a proved actual-law LSI
:label: cor-slcfi-lsi-transfer

The bounds above explicitly evaluate the stationary-centered kinetic score profile and the full Fisher production in terms of actual preparation discrepancy and structural envelopes. They remove the stationary constant source and preserve the averaged covariance subtraction. They do not identify $f$ with a fitness function: it is the density ratio of two complete prepared laws, including law-dependent cloning and collisions. The input-coordinate fitness derivative bounds in Chapters 14a/14b are not automatically bounds on this density ratio.

Where the actual full stationary law has a proved physical full-gradient LSI constant $C_\pi$, the preceding estimates can also control output entropy: if $G_{\rm phys}\succeq g_-I$, then
$$
H(Q\mid\pi)\le\frac{C_\pi}{2g_-}I_{G_{\rm phys}}(Q\mid\pi),
\qquad
\Phi_{G_{\rm phys}}(Q/\pi)
\le\left(1+\frac{C_\pi}{2g_-}\right)I_{G_{\rm phys}}(Q\mid\pi).
$$
The LSI must be one of the actual-law criteria of {prf:ref}`cor-n-uniform-lsi` or {prf:ref}`thm-lsi-companion-dependent-full`; the kinetic Gibbs LSI cannot be substituted without a proved law comparison. Consequently the displayed explicit production bound gives a quantitative contraction test once its evaluated preparation discrepancy is small relative to the entering modified entropy. No unevaluated optimal mixing rate has been introduced, and a strict rate is not asserted when that comparison has not been proved.
:::

:::{prf:proof}
Apply the actual-law LSI to $\sqrt{Q/\pi}$ to obtain $H(Q\mid\pi)\le(C_\pi/2)I(Q\mid\pi)$. Since $G_{\rm phys}\succeq g_-I$, the ordinary Fisher information is at most $g_-^{-1}I_{G_{\rm phys}}$. Add $I_{G_{\rm phys}}$ to obtain the modified-entropy bound. This uses the specified full physical gradient; it does not use a velocity-only LSI or an LSI for another reference.
:::


(sec-slcfu-audit)=
## 18. Full-kernel feedback and the long-time error budget

:::{div} feynman-prose
Freezing the population environment separates two questions: how the resulting
kernel mixes, and how much that kernel changes when the population changes.
The rootwise selection drift and local Gaussian estimate below answer the first
through weighted Harris bounds. A separate marked-component calculation bounds
environment sensitivity while retaining copying, collisions and unbounded
reward normalization.

But adding these bounds loses too much information. For this selection-drift
construction, the resulting unsigned coefficient satisfies
$q_{\mathrm{pop}}>1+3a_*$, so its proposed contraction test has no admissible
regime. This is a failure of that estimate, not a proof that the population
dynamics diverge. Closing the argument requires a signed nonlinear dissipation
estimate that preserves the helpful part of selection instead of charging all
feedback as an adverse perturbation.

The other estimates remain useful. Strict margins in the declared population
class give an explicit one-step exit bound of order $N^{-1/3}$; the proved
moment budgets give a residence horizon proportional to $\log N$. These quantify
how long the certified class can be used. They do not give indefinite retention
of each initial phase. When distinct nonlinear phases coexist with one
finite-population invariant law, rare transitions prevent uniform-time tracking
of every initial phase simultaneously. The section retains this distinction
while identifying exactly where the full-law proof still needs a sign.

There is also a positive trajectory result without either attraction or a
bounded force-center profile. Under its stated regularity and initial moment
conditions, Section 18.5 bounds the actual mean-field error simultaneously up
to an explicit horizon that grows with population size. It also proves
convergence of the entire population-measure path in a product metric whose
weights decrease for later times. This captures the evolution law beyond any
fixed horizon, while keeping separate the stronger questions of an error bound
uniform over all times and convergence to stationary phases.

:::

(sec-slcfz-rootwise)=
### 18.1. Rootwise selection confinement and frozen-environment mixing

:::{prf:definition} Actual frozen root kernel and numerical population class
:label: def-slcfz-root

Keep the actual all-alive canonical rooted map, with bounded comparison weights $\kappa_C\le w_C\le1$, capped entering velocities $V_{\max}$, independent measurement/cloning companions, frozen-source copying, recipient Gaussian jitter, complete component collisions, and the exact kinetic/noise/cap update. For a population environment $\mu$, let $P_\mu=C_\mu K$ be its actual root kernel, with the environment and its normalization statistics frozen at $\mu$ even when the root state $z$ differs from a typical $\mu$ sample. The rooted construction defines this kernel for every finite root state. There is no viscosity or history term.

Choose $p\ge4$, a declared bounded donor core $C\subset B(0,R_c)$ with Lebesgue volume $v_C>0$, and numerical $H<\infty$, $m\in(0,1]$. Put

$$
V(z)=|x|^p,\qquad
\mathfrak C_{H,m}=\{\mu:\mu V\le H,\ \mu(C\times\overline B_{V_{\max}})\ge m\}.
$$

Require this class to be nonempty. A sufficient directly checkable condition is that it contains a declared point mass supported in $C$ with $|x|^p\le H$. Raw reward moments must be defined throughout the class; a sufficient bound is $|R(z)|\le C_r(1+|x|^2)$.

Assume actual force profiles

$$
|F(x)|\le g_0+g_1|x|,\qquad \operatorname{Lip}(F)\le L_F,
\qquad c^2L_F<1,
$$

with $g_0,g_1\ge0$, $q,s>0$. The last inequality is a kinetic invertibility condition, not convexity or restoration. No bounded profile $\sup|x+\eta F(x)|$ is assumed. Use the exact parameters $a,c,B,\eta,q,s$, and put

$$
V_c=(1+2|\alpha_{\rm col}|)V_{\max},\quad
\tau^2=c^2q^2+s^2,\quad A_F=1+\eta g_1,\quad
m_{d,p}=2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2).
$$
:::

:::{prf:theorem} Rootwise drift from actual incoming-core and outgoing-tail profiles
:label: thm-slcfz-root-drift

Choose $R_t\ge R_c$. For every $\mu\in\mathfrak C_{H,m}$, every physical root $z$ with $|x|>R_t$, and almost every donor $z'\in C$, require a probability at least $p_g$ of $F_{z'}-F_z\ge\Delta>0$ under their actual measurement laws in environment $\mu$. This is a bound for each recipient/donor pair, not an average over the root law. Let

$$
a_g=\min\{1,\Delta/[s_c(F^*+\epsilon_c)]\},\quad
\chi_0=\kappa_Cmp_ga_g.
$$

For every $\mu\in\mathfrak C_{H,m}$ and every root $z$, require the actual outward accepted root flux to obey

$$
\Phi_{\mu,+}^{\rm root}(z)
=\mathbb E_{Y_D\mid z}\int\beta_\mu((z,Y_D),u)
 (|x_u|^p-|x|^p)_+\,\eta_\mu(du)
\le\delta |x|^p+b_{\rm rev}.
$$

Here $\delta,b_{\rm rev}\ge0$ are numerical bounds derived from declared regional fitness, companion and donor-moment envelopes. They must be uniform on the displayed class and over all root states, including roots outside the environment's support. Define

$$
\chi=\chi_0-\delta,\quad B_{\rm sel}=\chi_0R_t^p+b_{\rm rev},
\quad r_0=A_F^p(1-\chi).
$$

Require $0<\chi\le1$ and $r_0<1$. If $r_0>0$, choose

$$
u=\left(\frac{1+r_0}{2r_0}\right)^{1/(p-1)}-1>0;
$$

if $r_0=0$, take $u=1$. With $u$ denoting this proof parameter, set

$$
r=(1+u)^{p-1}r_0<1,\qquad
A=(1+u)^{p-1}A_F^p,
$$

$$
b=A B_{\rm sel}+(1+u^{-1})^{p-1}
 \left[BV_c+\eta g_0+(A_F\sigma_J+\tau)m_{d,p}^{1/p}\right]^p.
$$

Then the actual frozen full kernel satisfies the rootwise inequality

$$
\boxed{\quad P_\mu V(z)\le rV(z)+b\quad}
$$

for every $z$ and every $\mu\in\mathfrak C_{H,m}$. In particular, this is stronger than an integrated inequality $\mu P_\mu V\le r\mu V+b$; the latter is not used as a substitute for the former.
:::

:::{prf:proof}
At a root outside $B(0,R_t)$, accepted copying to the core has probability at least $\chi_0$. Every such copy decreases the pre-jitter $p$th cost by at least $|x|^p-R_c^p\ge|x|^p-R_t^p$. Drop all other negative transfers and use the positive-flux envelope. For a root inside $B(0,R_t)$, discard all negative transfers; the quantity $-\chi_0|x|^p+\chi_0R_t^p$ is nonnegative. Thus in both cases the actual selected frozen source $Y$ satisfies

$$
\mathbb E[|Y|^p\mid z]\le(1-\chi)|x|^p+B_{\rm sel}.
$$

Conditional on the selection and collision graph, the prepared position is $Y+I\sigma_J Z_J$, where $I$ is the root's acceptance indicator. Its collision velocity has norm at most $V_c$. The final position is

$$
Y+I\sigma_JZ_J+BV^C+\eta F(Y+I\sigma_JZ_J)+cqZ_O+sZ_x.
$$

The Gaussian variables here are independent of selection and the graph. The force growth bound, followed by Minkowski in these Gaussian variables, gives conditional $L^p$ norm at most

$$
A_F|Y|+BV_c+\eta g_0+(A_F\sigma_J+\tau)m_{d,p}^{1/p}.
$$

Use $(a+b)^p\le(1+u)^{p-1}a^p+(1+u^{-1})^{p-1}b^p$ and integrate the proved selected-source bound. This gives the displayed $r,b$ exactly. Incoming cloners can change $V^C$ but not the frozen position source $Y$, and the uniform velocity bound covers their complete component. No independence between component members is assumed.
:::

:::{prf:remark} How to certify the rootwise profiles from regional data
:label: rem-slcfz-regional-inputs

Refine the declared spatial partition at $R_c,R_t$. Bound the actual fitness on region $A_j$ by $F_j^-\le F\le F_j^+$ uniformly over $\mathfrak C_{H,m}$ and the sampled diversity mark. For quadratic-growth reward one explicit normalization budget is

$$
|\mu R|\le C_r(1+H^{2/p}),\qquad
\mu R^2\le2C_r^2(1+H^{4/p}),
$$

so the reward denominator lies between $\sigma_r$ and
$[2C_r^2(1+H^{4/p})+\sigma_r^2]^{1/2}$. Combine this interval with the regional raw-reward intervals and the bounded diversity interval to obtain the chapter's explicit logistic fitness bands. A uniform positive band gap from every exterior source to the declared core gives the preceding gap bound with $p_g=1$; finer conditional measurement estimates can supply $p_g<1$.

For a source in $A_a$ and donor in $A_b$, let

$$
k_{ab}^+=w_{ab}^+g_{ab}^+/Z_a^-,\quad
Z_a^-\ge\kappa_C,\quad
\mathcal T_b(t)\ge
\sup_{\mu\in\mathfrak C_{H,m}}\int_{A_b}(|y|^p-t)_+\,\mu(dy).
$$

Then the explicit nonnegative series

$$
\Phi_{\mu,+}^{\rm root}(z)
\le\sum_b k_{ab}^+\mathcal T_b(|x|^p)
$$

certifies the required rootwise envelope whenever it is bounded by $\delta|x|^p+b_{\rm rev}$ on every source region. A direct elementary bound is
$\mathcal T_b(t)\le\min\{H,m_b^+(R_b^p-t)_+\}$ for a bounded donor region $A_b\subset B(0,R_b)$ with a proved mass upper bound $m_b^+$; unbounded regions retain their explicitly proved tail-excess profiles. A divergent series or absent rootwise envelope does not become a Harris hypothesis merely because the integrated selection flux was bounded. Regional budgets used here must be consequences of the displayed moment/core class or separately proved preserved conditions.
:::

:::{prf:lemma} Complete root-kernel common mass independent of population size
:label: lem-slcfz-root-minorization

Choose a small-set level $R>0$, a latent root-jitter radius $J\ge0$ with
$p_J=\Pr(|\sigma_J Z|\le J)>0$, and target radii $r_x,u_v>0$. Put

$$
R_s=\max\{R^{1/p},R_c\},\quad R_x=R_s+J,\quad
F_x=g_0+g_1R_x,\quad R_1=R_x+c(V_c+cF_x),\quad F_1=g_0+g_1R_1,
$$

$$
\lambda_T=1-c^2L_F>0,\quad Q=(u_v+cF_1)/\lambda_T,\quad m_v=a(V_c+cF_x),
$$

$$
k_v=(2\pi q^2)^{-d/2}e^{-(Q+m_v)^2/(2q^2)}(1+c^2L_F)^{-d},\quad
k_x=(2\pi s^2)^{-d/2}e^{-(r_x+R_1+cQ)^2/(2s^2)},
$$

$$
\epsilon=\kappa_Cmp_J v_d(u_v)v_d(r_x)k_vk_x>0,
\qquad v_d(r)=\pi^{d/2}r^d/\Gamma(1+d/2).
$$

Let $\vartheta$ be uniform position on $B(0,r_x)$ times the radial-cap pushforward of uniform pre-cap velocity on $B(0,u_v)$. Then, uniformly for every $\mu\in\mathfrak C_{H,m}$ and every root with $V(z)\le R$,

$$
P_\mu(z,\cdot)\ge\epsilon\vartheta.
$$

There is no $N$th power in this root-kernel bound.
:::

:::{prf:proof}
Generate the root's actual proposed cloning donor, including a proposal that may subsequently be rejected. It belongs to the core with probability at least $\kappa_Cm$. On that event, acceptance chooses a core position and rejection retains the root position; either way the selected source has norm at most $R_s$. This argument does not demand a positive acceptance probability on the small set. Condition also on a latent root jitter of norm at most $J$, with probability $p_J$ independently of proposal and graph. Whether the jitter is used or not, prepared position has norm at most $R_x$ and complete collision velocity at most $V_c$.

For every such preparation the one-row Gaussian minorization calculation in {prf:ref}`thm-slc-minorization` applies with exactly $R_x,F_x,R_1,F_1,Q,m_v,k_v,k_x$ above. In particular $v\mapsto v+cF(x_1+cv)$ is globally invertible because $c^2L_F<1$, its preimages of $B(0,u_v)$ lie in $B(0,Q)$, and its Jacobian upper bound is $(1+c^2L_F)^d$. The independent final position noise supplies $k_x$; the cap preserves measure domination. Integrating over every component graph, donor proposal, measurement and bounded jitter gives the root minorization. The component can contain arbitrarily many vertices; only its already proved velocity bound is used.
:::

:::{prf:theorem} A verified invariant environment class and uniform frozen Harris rate
:label: thm-slcfz-harris

Retain all preceding rootwise profile bounds. Choose a number $L>H^{1/p}$ and suppose the global acceptance upper bound

$$
a_* =\min\{1,(F^*-F_*)/[s_c(F_*+\epsilon_c)]\}<1
$$

and the following finite parameter inequalities hold:

$$
rH+b\le H,\qquad
m\le(1-a_*)\left(1-\frac{H}{L^p}\right)v_C(2\pi\tau^2)^{-d/2}
 \exp\left[-\frac{(R_c+A_FL+BV_c+\eta g_0)^2}{2\tau^2}\right].
$$

Then the actual nonlinear map preserves $\mathfrak C_{H,m}$. Thus the environment hypotheses propagate for every initialized population in that class; no unproved class invariance is assumed.

For a declared $R_{\rm ref}>0$, choose

$$
R=4b/(1-r)+R_{\rm ref},\qquad
\beta_H=\frac{\epsilon}{R(1+r)+2b},
$$

where $\epsilon$ is the preceding explicitly evaluated root minorization at this value of $R$. Set

$$
\rho_H=\max\left\{1-\epsilon/2,
\frac{2+\beta_H(rR+2b)}{2+\beta_H R}\right\}<1.
$$

For every fixed environment $\mu\in\mathfrak C_{H,m}$, its actual complete root kernel $P_\mu$ contracts the weighted norm

$$
\|\xi\|_{\beta_H}=\int(1+\beta_H|x|^p)\,d|\xi|
$$

by factor $\rho_H$ on zero-mass signed measures. It has a unique invariant root law $\pi_\mu$ with finite $p$th moment, satisfying $\pi_\mu V\le b/(1-r)$. The same contraction applies to two initial laws propagated through any common sequence of frozen kernels in this class. Every displayed drift, minorization and contraction constant is independent of particle number.
:::

:::{prf:proof}
Integrate the actual rootwise drift to get $(\mathcal F_h\mu)V\le rH+b\le H$. To control core mass, Markov gives $\mu\{|x|\le L\}\ge1-H/L^p$. At each such root the probability of no accepted outgoing edge is at least $1-a_*$. Its position is unchanged before kinetics and receives no jitter. Incoming collisions may change velocity, but it remains bounded by $V_c$. The final position conditional on this preparation is Gaussian of covariance $\tau^2I_d$ and center of norm at most $A_FL+BV_c+\eta g_0$. Its density everywhere in $C\subset B(0,R_c)$ is bounded below by the displayed Gaussian value. Integrating the conditional bound and then the root distribution proves the required output core mass. This proves both defining inequalities for class invariance.

The rootwise drift and root minorization now match every hypothesis of the explicitly proved Harris theorem {prf:ref}`thm-slc-tv-rate`, with state space $E$ and $V=|x|^p$. Its proof is a common-mass coupling of each pair on $V(z)+V(z')\le R$, and the drift estimate outside that set. It gives exactly the displayed $\rho_H$, weighted contraction, fixed point and invariant moment bound. The proof is uniform in the environment because both the minorization probability and its reference $\vartheta$ are uniform. Iterating the same one-step contraction through a common environment sequence proves the last assertion.
:::


(sec-slcef-feedback)=
### 18.2. Uniform reward sensitivity and the complete feedback bound

:::{prf:lemma} Uniform fitness sensitivity for an unbounded raw reward
:label: lem-slcef-logistic-normalization

Use the actual regularized logistic-power fitness with raw reward
$|R(z)|\le C_r(1+|x|^2)$. Fix $p\ge4$ and $\beta>0$,
$w(z)=1+\beta|x|^p$, $\delta_w=\tfrac12\int w|\mu-\nu|$, and a
class $\mathfrak C$ with $\mu|x|^p\le M_p$. Set
$$
 B_w=1+\beta M_p,\quad
 B_r=C_r(1+\beta^{-2/p}),\quad
 B_{r^2}=2C_r^2(1+\beta^{-4/p}),
$$
$$
 M_r=C_r(1+M_p^{2/p}),\quad
 C_{\rm var}=2B_{r^2}+4M_rB_r,\quad K_D=1+2/\kappa_D,
$$
$$
 S_b=\sqrt{D_*^2+\delta_D^2}-\delta_D,\quad
 T_s=K_D[S_b/\sigma_s+S_b^3/(2\sigma_s^3)].
$$
Let $H_r,H_s$ be the primitive logistic-power derivative constants of
{prf:ref}`lem-slcc-selection-perturbation`, with $H_b=0$ for an
inactive exponent. Define
$$
 C_F=\frac{2H_rB_r}{\sigma_r}
       +\frac{2H_rC_{\rm var}}{e\sigma_r^2}+H_sT_s.
$$
For two identical physical/measurement types evaluated in environments
$\mu,\nu\in\mathfrak C$, their actual fitness values differ by at most
$C_F\delta_w(\mu,\nu)$, uniformly in the physical root position.
:::

:::{prf:proof}
For $0\le a\le p$, $|x|^a/w(x)\le\beta^{-a/p}$, since $t^{a/p}/(1+t)\le1$. Thus $|R|\le B_rw$ and $R^2\le B_{r^2}w$; Hölder bounds each reward mean by $M_r$. The weighted reward bounds give mean difference at most $2B_r\delta_w$
and variance difference at most $C_{\rm var}\delta_w$, as in
{prf:ref}`lem-slcw-normalization`. Interpolate their means and variances
linearly. The interpolated variance is nonnegative. Write
$t=(R-m)/\sqrt{v+\sigma_r^2}$ and $\ell(t)=(1+e^{-t})^{-1}$.
Then $\ell'(t)\le1/4$ and
$|t|\ell'(t)\le |t|e^{-|t|}\le1/e$.
The power/floor factors multiplying the logistic derivative are bounded
by $4H_r$. Consequently the full fitness derivatives satisfy
$$
 |\partial_m F|\le H_r/\sigma_r,\qquad
 |\partial_v F|\le2H_r/(e\sigma_r^2).
$$
Integrate these uniform bounds along the interpolation. Interpolate the
diversity moments separately; their already proved bounded-range
estimate gives $H_sT_s\delta_w$. Adding the three changes proves the
claim without a coefficient growing with $|R(z)|$.
:::

:::{prf:lemma} Ordered collision components without a weak-selection premise
:label: lem-slcef-ordered-component

For the actual all-alive frozen marked law $\eta_\mu$, let
$0\le\beta_\mu(t,u)\le c_*=a_*/\kappa_C$. For every specified root
type $t$, its complete population collision component satisfies

$$
\mathbb E_t|\mathcal C|\le M_{\rm graph}:=e^{2c_*}<\infty.
$$

If one accepted edge and both endpoint types have already been exposed,
the remaining exploration, including the endpoints, has expected size at
most $2M_{\rm graph}$. These are consequences of the actual strict
fitness ordering and single-outgoing-edge rule; they require no inequality
$2c_*<1$.
:::

:::{prf:proof}
A live accepted edge strictly increases its frozen fitness. The outgoing
degree is at most one, so a simple undirected path of length $n$ has
$k$ increasing edges followed by $n-k$ decreasing edges for some
$0\le k\le n$. Each outgoing transition is the actual subprobability
$\beta_\mu(t,u)\eta_\mu(du)$. Repeated integration of the incoming
Poisson intensity gives the same factor for every incoming edge.
Bound each density by $c_*$, retaining the strict order restrictions.
The $n$ newly integrated types then have product base law. Discarding
root and cross-block comparisons leaves a strictly increasing block of
$k$ fitness marks and a strictly decreasing block of $n-k$ marks.
Permutation symmetry bounds their probabilities by $1/k!$ and
$1/(n-k)!$. Atomic mark laws cause ties, which only reduce these
strict-order probabilities. Consequently

$$
\mathbb E_t\#\{u:\operatorname{dist}(t,u)=n\}
\le c_*^n\sum_{k=0}^n\frac1{k!(n-k)!}
=\frac{(2c_*)^n}{n!}.
$$

Sum over $n\ge0$. The finite expected count also proves finiteness of
the component almost surely. A vertex reached through its outgoing edge
has that choice consumed, which removes possible paths and preserves the
upper bound.

After exposing a single accepted edge and its endpoint types, delete the
edge. Each endpoint has an exploration bounded by the same free-root
calculation. A used outgoing edge is suppressed; a known incoming child
is excluded by vertex identity, while the additional incoming process is
the original Poisson process. The latter statement is exactly the Palm
construction in the specified rooted kernel. Summing the two bounds gives
$2M_{\rm graph}$.

When this estimate is used in a coupling, complete the first-marginal
component with its original remaining randomness and stop the coupled
queries at their first mismatch. The query count is dominated pathwise
by that completed component. Summing conditional mismatch hazards before
averaging therefore uses this expected count without conditioning all
future offspring on earlier matching successes.
:::

:::{prf:theorem} Full frozen-environment feedback under linear force growth
:label: thm-slcef-environment

Use the class and constants of {prf:ref}`lem-slcef-logistic-normalization`.
The actual force need only satisfy $|F(x)|\le G_0+G_1|x|$, with
$G_0,G_1\ge0$. No finite bound on $|x+\eta F(x)|$ is assumed.
Let $P_\mu(z,\cdot)$ be the actual conditional root-output kernel with
all environment statistics frozen at $\mu$, integrating all measurement
marks, copying, jitter, complete component collisions and kinetics.
It is defined by the rootwise construction for every admissible physical
root $z$, not merely as an unspecified $\mu$-almost-everywhere version.

Keep the explicit fitness bounds $F_*,F^*$ and gate constants
$$
 a_* =\min\{1,(F^*-F_*)/[s_c(F_*+\epsilon_c)]\},\quad
 c_*=a_*/\kappa_C,\qquad M_{\rm graph}=e^{2c_*},
$$
$$
 L_g=\frac1{s_c(F_*+\epsilon_c)}
       +\frac{F^*-F_*}{s_c(F_*+\epsilon_c)^2}.
$$
Put
$$
 K_{D,w}=1+2B_w/\kappa_D,\quad
 D_\beta=a_*/\kappa_C^2+2L_gC_F/\kappa_C,
 \quad L=2c_*K_D+D_\beta.
$$
For the kinetic readout let
$$
 A_x=1+\eta G_1,\quad b_x=BV_c+\eta G_0,\quad
 \tau^2=c^2q^2+s^2,\quad m_{d,p}=2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2),
$$
$$
 A_K=3^{p-1}2^{p-1} A_x^p,\quad
 B_K=3^{p-1}[2^{p-1}A_x^p\sigma_J^pm_{d,p}+b_x^p+\tau^pm_{d,p}],
 \quad D_K=\max\{A_K,1+\beta B_K\}.
$$
Define the finite, deliberately conservative coefficient
$$
 L_{\rm env}=D_K\left[
 \frac{4(a_*+2c_*+c_*B_w)}{\kappa_D}
 +4(c_*K_{D,w}+D_\beta B_w+L)
 +8L(1+c_*B_w)M_{\rm graph}\right].
$$
Then the full conditional kernels satisfy
$$
 \boxed{\quad
 \sup_z\frac{\delta_w(P_\mu(z,\cdot),P_\nu(z,\cdot))}{w(z)}
 \le L_{\rm env}\delta_w(\mu,\nu).
 \quad}
$$
In particular,
$$
 \delta_w(\nu P_\mu,\nu P_\nu)
 \le B_wL_{\rm env}\delta_w(\mu,\nu).
$$
The constants depend only on the displayed moment-class, reward,
regularization, feature, cloning and kinetic parameters. All collision
correlations are retained. The bound vanishes when the acceptance
mechanism is identically zero and its environment derivative vanishes.
:::

:::{prf:proof}
Write $\delta=\delta_w(\mu,\nu)$. Start with the same physical root
$z$ in both environments. Their conditional measurement companion laws
can be coupled with failure probability at most $2\delta/\kappa_D$.
For ordinary marked population types, couple physical states maximally,
then couple measurement companions conditionally at matched states.
This gives bad-type probability at most $K_D\delta$, and weighted bad
mass at most
$$
 \int(w(z_1)+w(z_2))\mathbf1_{\rm bad}\,d\Lambda
 \le2\delta+4B_w\delta/\kappa_D=2K_{D,w}\delta.
$$
Indeed the physical unmatched weighted mass is exactly $2\delta$,
and the conditional measurement failure is bounded uniformly, while
each marginal's mean weight is at most $B_w$.

For matching physical/measurement types, the preceding lemma bounds
fitness changes uniformly by $C_F\delta$. Subtracting the cloning
normalizers and gates gives accepted-edge density difference at most
$D_\beta\delta$. Each edge density is at most $c_*$. Thus the
outgoing subprobability and incoming Poisson intensities have discrepancy
at most $L\delta$; completing the outgoing law by a cemetery point and
coupling common Poisson intensities gives failure probability at most
$2L\delta$ per queried vertex. This is uniform in the queried root
position. An incoming child has already used its outgoing edge, which
can only reduce the number of subsequent queries.

Before exploring any other edge, expose the root's outgoing choice.
This identifies its frozen source position: the root if it does not
copy, or its donor if it copies. Incoming edges and subsequent component
exploration never change this source position. If the root types match,
the weighted cost of a failed coupling of this outgoing choice is at
most
$$
 4(c_*K_{D,w}+D_\beta B_w+Lw(z))\delta.
$$
To see the three terms, bad donor types cost at most $c_*$ times their
weighted bad mass; differences on good donor types cost at most
$D_\beta\delta$ times the sum of marginal donor weights, at most
$2B_w$; the cemetery completion costs at most $2L\delta$ times the
root weight. Doubling these bounds gives the displayed conservative
constant. No unweighted coupling is substituted for a weighted donor
error.

If that outgoing choice is matched, condition on its exposed result and
root/donor types, then sample all remaining graph primitives freshly.
Extend the first marginal to its ordinary complete rooted component,
without conditioning unexposed randomness on future matching. It starts
with at most two vertices. The actual accepted edges strictly increase frozen fitness. The ordered-path
calculation in {prf:ref}`lem-slcef-ordered-component` therefore bounds
this component without any small-selection condition. The known root at its donor is excluded
by vertex identity, not by type, so atoms of the marked law remain
admissible. The outgoing-used rule can only remove a query. The
conditional expected complete-tree size is therefore at most
$2M_{\rm graph}$. Up to the first mismatch, the queried vertices are a
subset of this first-marginal component, pathwise. At each query the
fresh conditional mismatch hazard is at most $2L\delta$, uniformly
in the exposed types. Summing these conditional hazards gives remaining
mismatch probability at most $4L M_{\rm graph}\delta$. This does not
assert that offspring laws remain unchanged after previous coupling
successes. The initial outgoing matching uses only that outgoing
randomization and its types, leaving all other primitives fresh.
The expected weight of the root's common frozen source is at most
$w(z)+c_*B_w$. The conditional probability bound is uniform over the
exposed source type, so their product is a legitimate upper bound on
the weighted mismatch cost. In particular no independence between a
source weight and its collision component is required.

For a fixed frozen source $y$, the actual final position has the bound
$$
 |X^+|\le A_x(|y|+\sigma_J|Z_J|)+b_x+|cq\xi+s\zeta|.
$$
Using $(u+v+w)^p\le3^{p-1}(u^p+v^p+w^p)$ and
$(u+v)^p\le2^{p-1}(u^p+v^p)$ gives conditional $p$th moment at most
$A_K|y|^p+B_K$, uniformly over all component velocities. Thus its
expected output weight is at most $D_Kw(y)$. This bound also covers
an unperturbed root with no copying or jitter. For a test
$|\varphi|\le w/2$, all kinetic/jitter conditional integrals have
absolute value at most $D_Kw(y)/2$. On a matched complete component,
common rotation and noises give identical readouts. For unmatched
components with a matched source, the weighted discrepancy is therefore
at most $D_Kw(y)$. Combining with the previous conditional exploration
bound costs at most
$4D_KL(1+c_*B_w)M_{\rm graph}w(z)\delta$.

It remains to handle mismatched root measurement marks. Conditional on
the paired marks, sample each marginal component using fresh primitives
conditional on its own mark; the mark-coupling coin does not further
condition that component. Subtract the
same kinetic baseline $K\varphi(z)$ in both environments. An isolated
root has zero preparation increment. In one environment the probability
of a root outgoing edge is at most $a_*$, and the probability of an
incoming edge at most $c_*$. The expected source weight on outgoing
edges is at most $c_*B_w$. Therefore the expected absolute increment,
conditional on any root mark, is at most
$$
 \frac{D_K}{2}[(a_*+2c_*)w(z)+c_*B_w].
$$
The terms count the baseline weight on every non-isolated event and
the unchanged source on incoming-only events. Sum both environments,
then multiply by the root-mark mismatch probability at most
$2\delta/\kappa_D$. This contributes at most
$2D_K(a_*+2c_*+c_*B_w)w(z)\delta/\kappa_D$.

The failed-outgoing, matched-source component, and failed-root-mark
contributions just proved are bounded by the three terms in
$L_{\rm env}w(z)\delta$, with spare factors of two where displayed.
Taking the supremum over $|\varphi|\le w/2$ proves the weighted
variation assertion. Integrating over $\nu(dz)$ gives the final bound
because $\nu w\le B_w$. Every bound is on the full conditional root
kernel, so the argument applies to its actual law-dependent preparation.
:::

:::{prf:corollary} Explicit feedback coefficient available to a selection-based Harris proof
:label: cor-slcef-harris-assembly

Suppose the same class is invariant and a separately proved frozen-root
weighted contraction satisfies
$\delta_w(\alpha P_\mu,\gamma P_\mu)\le q_H\delta_w(\alpha,\gamma)$
for every frozen environment in the class, with numerical $q_H<1$.
Then the actual population map satisfies
$$
 \delta_w(\mathcal F_h\mu,\mathcal F_h\nu)
 \le(q_H+B_wL_{\rm env})\delta_w(\mu,\nu).
$$
Strict contraction follows if the displayed computed coefficient is less
than one. This statement removes the bounded-force-center assumption
from the feedback estimate; it does not assert that integrated selection
moment drift alone supplies the required frozen-root contraction.
:::

:::{prf:proof}
Insert $\nu P_\mu$ between $\mu P_\mu$ and $\nu P_\nu$, apply the
assumed frozen-root estimate to the first difference and the proved
feedback estimate to the second, and add. The condition is numerical;
no unknown optimal convergence constant is defined by the conclusion.
:::

:::{prf:remark} What these parameter tests do and do not close
:label: rem-slcfz-feedback-gap

The simultaneous moment/core inequalities are computable sufficient tests; the proof does not assert that a feasible $H,m,L$ exists for every landscape or every parameter choice. The Gaussian worst-center recovery bound can be very conservative. No finite-particle empirical core fraction is asserted to be pathwise invariant: its actual fluctuations require the separate conditional coverage or defect estimates.

The theorem proves mixing when both compared root laws see the same environment. For two nonlinear trajectories the environments differ, so their additional term is the actual signed difference $\nu(P_\mu-P_\nu)$. Frozen Harris contraction alone does not bound it or force the invariant root laws $\pi_\mu$ to agree.

There is a concrete obstruction to closing this step with a crude unsigned feedback estimate. The outside-set Harris fraction is strictly greater than $r$, because its numerator minus $r$ times its denominator is $2(1-r)+2\beta_Hb>0$. Consequently $\rho_H>r$. Here $A_F\ge1$ and $(1+u)^{p-1}\ge1$, so

$$
1-\rho_H<1-r\le\chi\le\chi_0\le a_*.
$$

Therefore a proposed nonlinear bound of the form $q_{\rm pop}=\rho_H+M_wL_{\rm env}$ cannot yield $q_{\rm pop}<1$ whenever its evaluated feedback coefficient satisfies $M_wL_{\rm env}\ge a_*$. This certificate is algebraically unavailable with the displayed selection-driven coefficients. One can instead establish a kinetic drift gap that remains positive as selection weakens: {prf:ref}`thm-kuhw-active-exact-uniform-law` proves the finite-swarm harmonic result this way, and {prf:ref}`thm-pvb-active-population-convergence` proves the conservative population result at weak positive count viscosity. Their explicit feedback budgets close without an assumed restoring sign. A signed selection estimate remains a possible improvement of the present coefficients. The fixed-step mean-field equation and frozen-environment theorem retain their stated hypotheses.
:::

:::{prf:proposition} The computed unsigned feedback test is empty for this selection-drift route
:label: prop-slcfz-unsigned-empty

Use exactly the drift and Harris coefficients above and exactly $L_{\rm env}$ of {prf:ref}`thm-slcef-environment`, evaluated with $\beta=\beta_H$ and $M_p=H$. Put $B_w=1+\beta_HH$. If selection is active, $a_*>0$, then

$$
\boxed{\quad \rho_H+B_wL_{\rm env}>1+3a_*>1.\quad}
$$

Thus the sufficient condition $\rho_H+B_wL_{\rm env}<1$ has no solution using these particular coefficients. This is a failure of this unsigned perturbation proof, not a theorem that the gas fails to converge.
:::

:::{prf:proof}
The explicit feedback formula contains the nonnegative term
$4D_K(a_*+2c_*+c_*B_w)/\kappa_D$. Since $D_K\ge1$, $\kappa_D\le1$ and $B_w\ge1$, it gives $B_wL_{\rm env}\ge4a_*$. The preceding drift calculation gives $\rho_H>r\ge1-\chi\ge1-a_*$. The inequality $\chi\le a_*$ uses the actual feasible fitness-gap bound $a_g\le a_*$ and $\kappa_Cmp_g\le1$. Add the two inequalities. If $a_*=0$, the strictly positive inward-selection hypothesis itself is unavailable. Therefore neither case supplies a nonempty contraction regime from this assembly. This calculation concerns its selection-derived gap; a separately proved force-driven kinetic gap, as in {prf:ref}`thm-pvb-active-population-convergence`, can support a nonempty absolute-feedback test.
:::

(sec-slcex-exit)=
### 18.3. Derived finite-population exit and residence estimates

:::{prf:theorem} Explicit one-step exit probability from a strict population class
:label: thm-slcex-one-step

Use the actual all-alive canonical full update and its bounded-test
consistency constant $G=A+4B_*^2$, in the convention
$$
 \mathbb E\left[|(L_N(S')-\mathcal F_hL_N(S))\varphi|^2\mid S\right]
 \le G/N,\qquad |\varphi|\le1.
$$
Fix $p\ge2$, $H>0$, $m\in[0,1]$, and $R_c>0$. Let
$$
 \mathfrak C=\{\mu:\mu|x|^p\le H,\quad
                         \mu(\overline B(0,R_c)\times\overline B_{V_{\max}})\ge m\}.
$$
Suppose the actual nonlinear full map satisfies the analytically certified
strict self-map margins, for every admissible $\mu\in\mathfrak C$,
$$
 (\mathcal F_h\mu)|x|^p\le H-\Delta_H,\qquad
 (\mathcal F_h\mu)(\overline B(0,R_c)\times\overline B_{V_{\max}})
                                     \ge m+\Delta_m,
$$
where $\Delta_H,\Delta_m>0$. For a proved full-map moment bound
$W_p(\mathcal F_h\mu)\le rH+b$ and proved landing mass at least
$m_{\rm land}$ on this class, the explicit choices are
$\Delta_H=H-(rH+b)$ and $\Delta_m=m_{\rm land}-m$, whenever positive.
These are margins for the complete population update, not for cloning alone.

For a deterministic entering configuration $S$ satisfying
$L_N(S)\in\mathfrak C$ and $W_{2p}(S)\le H_{2p}$, assume the actual
force satisfies $|F(x)|\le G_0+G_1|x|$. Set
$$
 A_x=1+\eta G_1,\quad b_x=BV_c+\eta G_0,\quad
 \tau^2=c^2q^2+s^2,\quad r_C=1+2/\kappa_C,
$$
$$
 m_{d,2p}=2^p\Gamma((d+2p)/2)/\Gamma(d/2),
$$
$$
 Q=3^{2p-1}\left[
 2^{2p-1}A_x^{2p}(r_CH_{2p}+\sigma_J^{2p}m_{d,2p})
                         +b_x^{2p}+\tau^{2p}m_{d,2p}\right].
$$
Then
$$
 \Pr\{L_N(S')\notin\mathfrak C\mid S\}\le u_N,
$$
where the fully explicit bound is
$$
 u_N=\min\left\{1,\frac{G}{N\Delta_m^2}
 +\min\left[
 \frac{2\sqrt2\,Q^{1/2}}{\Delta_H}(G/N)^{1/4},
 3\left(\frac{16GQ^2}{N\Delta_H^4}\right)^{1/3}
 \right]\right\}.
$$
The moment contribution is interpreted as zero if $GQ=0$.
In particular, fixed primitive parameters, strict margins and $H_{2p}$
give $u_N=O(N^{-1/3})$. No Gaussian boundary-shell estimate is needed:
the source one-step theorem applies to the bounded measurable core
indicator itself.
:::

:::{prf:proof}
The copying multiplicity bound and the actual Gaussian moment give
post-preparation $2p$th positional moment at most
$2^{2p-1}(r_CH_{2p}+\sigma_J^{2p}m_{d,2p})$.
The complete position satisfies
$|X^+|\le A_x|X^C|+b_x+|cq\xi+s\zeta|$.
The inequality $(a+b+c)^{2p}\le3^{2p-1}(a^{2p}+b^{2p}+c^{2p})$
therefore proves the displayed common bound $Q$ for both the conditional
expected empirical output $2p$th moment and the population output
$2p$th moment. It uses the actual full kernel and requires no independent
output rows.

Write $\widehat\nu=L_N(S')$, $\nu=\mathcal F_hL_N(S)$ and
$Y(x)=|x|^p$. For $T>0$, use the bounded function
$\varphi_T=\min(Y,T)/T$. The scalar consistency theorem gives
$$
 \mathbb E|\widehat\nu\min(Y,T)-\nu\min(Y,T)|^2\le T^2G/N.
$$
The two discarded tails have total expected mass at most $2Q/T$,
since $Y\mathbf1_{Y>T}\le Y^2/T$. Consequently
$$
 \mathbb E|\widehat\nu Y-\nu Y|\le T\sqrt{G/N}+2Q/T.
$$
Choosing $T^2=2Q/\sqrt{G/N}$ and applying Markov at threshold
$\Delta_H$ proves the first moment-error bound. Alternatively, if the
full moment discrepancy exceeds $\Delta_H$, either the truncated
absolute discrepancy exceeds $\Delta_H/2$, or the sum of its tails
exceeds $\Delta_H/2$. Chebyshev and Markov respectively give
$$
 \Pr\{|\widehat\nu Y-\nu Y|>\Delta_H\}
 \le\frac{4T^2G}{N\Delta_H^2}+\frac{4Q}{T\Delta_H}.
$$
Choosing $T^3=QN\Delta_H/(2G)$ yields
$3(16GQ^2/(N\Delta_H^4))^{1/3}$. The zero cases follow by taking
limits in the unoptimized estimates. The strict population moment
margin makes an empirical violation $\widehat\nu Y>H$ imply such a
discrepancy. For the core indicator the strict mass margin and the
bounded-test second moment give violation probability at most
$G/(N\Delta_m^2)$ directly. The union bound and probability clipping
complete the proof.
:::

:::{prf:corollary} Growing residence windows with an averaged higher-moment budget
:label: cor-slcex-residence

Keep the class and strict margins above. Suppose $L_N(S_0)\in\mathfrak C$
almost surely and define
$\tau_{\mathfrak C}=\inf\{n\ge0:L_N(S_n)\notin\mathfrak C\}$.
Assume the proved stopped higher-moment bound
$$
 \sup_{N,j}\mathbb E[\mathbf1_{\{\tau_{\mathfrak C}>j\}}W_{2p}(S_j)]
 \le\overline H_{2p}<\infty.
$$
Let $\overline Q$ be the displayed formula for $Q$ with
$H_{2p}=\overline H_{2p}$ and let $\overline u_N$ be the same formula
for $u_N$ with $Q=\overline Q$. Then, for every integer $T\ge1$,
$$
 \Pr\{\tau_{\mathfrak C}\le T\}\le\min\{1,T\overline u_N\}.
$$
In particular $T_N=o(N^{1/3})$ gives vanishing exit probability when
all the constants and strict margins are independent of $N$.
More generally the exact sufficient condition is
$T_N\overline u_N\to0$. Physical time is $hT_N$ and work is
$NT_N$ row updates. This is a growing-horizon residence estimate,
not an all-time coverage assertion.

If the stronger pathwise bound $W_{2p}(S_j)\le H_{2p}$ holds whenever
$L_N(S_j)\in\mathfrak C$, the conditional bound $u_N$ of the theorem
holds throughout the class and improves the residence estimate to
$1-(1-u_N)^T$. Without that stronger hypothesis, the averaged result
above does not claim this product bound.
:::

:::{prf:proof}
Before exit the population self-map margins apply. Keep the truncation
level $T_*$ deterministic, chosen using $\overline Q$ rather than the
realized entering moment. The proof's conditional truncated variance
bound is at most $T_*^2G/N$; its conditional tail bound is affine in
$W_{2p}(S_j)$. Multiply these inequalities by
$\mathbf1_{\{\tau_{\mathfrak C}>j\}}$ and average. The assumed stopped
moment budget bounds the resulting tail expression by $2\overline Q/T_*$;
all constant terms are multiplied by a probability at most one.
Thus the probability of a first exit on step $j+1$ is at most
$\overline u_N$. Sum over $0\le j<T$. If a pathwise entering bound
is available, the conditional survival probability at each preceding
step is at least $1-u_N$, whose product proves the stronger assertion.
No independence between successive updates is needed.
:::

:::{prf:theorem} Selection-closed structural tails and polynomial residence windows
:label: thm-slcex-selection-tail-closure

Use the unchanged all-alive complete kernel, the regional bands of
{prf:ref}`lem-slceg-uniform-counts`, and the force and noise parameters of
{prf:ref}`thm-slceg-full-moment`. Fix $p\ge2$, $H>0$, $m>0$ and the
class $\mathfrak C$ of {prf:ref}`thm-slcex-one-step`. The declared
regional weight, gate and mass bands must hold for every population law
and every empirical measure in $\mathfrak C$. Compute $\chi$ and
$b_{q,\mathrm{sel}}$ by {prf:ref}`lem-slceg-uniform-counts` for each
exponent $q$ in a finite set $\mathcal Q$ containing $p$ and $2p$.
Require the resulting evaluated coefficient $0<\chi\le1$.
For each $q\in\mathcal Q$ require the evaluated inequality
$$
 t_q=A_\lambda^q(1-\chi)<1,
 \qquad
 u_q=\begin{cases}
 [(1+t_q)/(2t_q)]^{1/(q-1)}-1,&t_q>0,\\
 1,&t_q=0,
 \end{cases}
$$
and set, with the Gaussian moment $m_{d,q}$ from the full-moment theorem,
$$
 a_q=(1+u_q)^{q-1}A_\lambda^q,
 \qquad r_q=a_q(1-\chi)<1,
$$
$$
 B_q=a_qb_{q,\mathrm{sel}}+(1+u_q^{-1})^{q-1}
  [b_0+(A_\lambda\sigma_J+\tau)m_{d,q}^{1/q}]^q.
$$
Here $A_\lambda=|1-\eta\lambda|+\eta g_1$,
$b_0=BV_c+\eta g_0$, and $\tau^2=c^2q_{\rm OU}^2+s^2$;
$q_{\rm OU}$ denotes the OU noise parameter, distinct from the moment
exponent. Suppose $\tau>0$ and the actual acceptance probability has
the uniform upper bound
$$
 a_* =\min\{1,(F^*-F_*)/[s_c(F_*+\epsilon_c)]\}<1.
$$
Choose $L>H^{1/p}$ and compute, with $v_d(R_c)$ the volume of the
positional core ball,
$$
 m_{\rm land}=(1-a_*)\left(1-\frac{H}{L^p}\right)
 \frac{v_d(R_c)}{(2\pi\tau^2)^{d/2}}
 \exp\!\left[-\frac{(R_c+A_\lambda L+b_0)^2}{2\tau^2}\right].
$$
Require the two **numerical**, $N$-independent margins
$$
 \Delta_H=H-r_pH-B_p>0,
 \qquad \Delta_m=m_{\rm land}-m>0.                 \tag{18.3a}
$$
Finally let $M_{q,0}=\sup_{N\ge2}\mathbb EW_q(S_0)<\infty$ for
$q\in\mathcal Q$, and suppose $L_N(S_0)\in\mathfrak C$ almost surely.
Define
$$
 M_q^{\rm stop}=\max\{M_{q,0},B_q/(1-r_q)\},
 \qquad
 M_q^{\rm pop}=\max\{W_q(\mu_0),B_q/(1-r_q)\}.
$$
Then the nonlinear population trajectory remains in $\mathfrak C$
for every time and satisfies $W_q(\mu_n)\le M_q^{\rm pop}$ for every
$q\in\mathcal Q$. For the finite swarm, with
$\tau_{\mathfrak C}=\inf\{n:L_N(S_n)\notin\mathfrak C\}$,
$$
 \sup_{N,n}\mathbb E[\mathbf1_{\{\tau_{\mathfrak C}>n\}}W_q(S_n)]
 \le M_q^{\rm stop}.                              \tag{18.3b}
$$
For $0\le s<q$, the corresponding moment-weighted exterior tails obey
$$
 \int_{|x|>R}|x|^s\,d\mu_n
 \le M_q^{\rm pop}R^{-(q-s)},\qquad
 \mathbb E\left[\mathbf1_{\{\tau_{\mathfrak C}>n\}}
   L_N(S_n)(|x|^s\mathbf1_{\{|x|>R\}})\right]
 \le M_q^{\rm stop}R^{-(q-s)}.                    \tag{18.3b'}
$$
Thus the stopped higher-moment premise of
{prf:ref}`cor-slcex-residence` is *derived* from the same signed
selection estimate, rather than supplied as an independent tail axiom.

For a completely evaluated finite-$N$ exit bound, set
$$
 A_x=1+\eta(g_1+\lambda),\quad b_x=b_0,\quad
 r_C=1+2/\kappa_C,\quad
 m_{d,2p}=2^p\Gamma((d+2p)/2)/\Gamma(d/2),
$$
$$
 Q_*=3^{2p-1}\left[
 2^{2p-1}A_x^{2p}
  (r_CM_{2p}^{\rm stop}+\sigma_J^{2p}m_{d,2p})
 +b_x^{2p}+\tau^{2p}m_{d,2p}\right],
$$
and let $G$ be the explicit bounded-test full-update consistency
constant in {prf:ref}`thm-slcex-one-step`. Put
$$
 U_N=\min\left\{1,\frac{G}{N\Delta_m^2}+
 \min\left[
 \frac{2\sqrt2\,Q_*^{1/2}}{\Delta_H}(G/N)^{1/4},
 3\left(\frac{16GQ_*^2}{N\Delta_H^4}\right)^{1/3}
 \right]\right\}.
$$
For every integer $T\ge1$,
$$
 \Pr\{\tau_{\mathfrak C}\le T\}\le\min\{1,TU_N\}. \tag{18.3c}
$$
Consequently, for $R>0$ and $0<\varepsilon\le1$, if
$H/R^p\le\varepsilon$, then
$$
 \Pr\left\{\max_{0\le j\le T}
 L_N(S_j)\{|x|>R\}>\varepsilon\right\}
 \le\min\{1,TU_N\},                              \tag{18.3d}
$$
while at each $j\le T$,
$$
 \mathbb E L_N(S_j)\{|x|>R\}
 \le\min\{1,H/R^p+\min(1,TU_N)\}.                \tag{18.3e}
$$
If $TU_N<1$, conditioning on residence through $T$ gives, for every
$n\le T$, $q\in\mathcal Q$ and $0\le s<q$,
$$
 \mathbb E[W_q(S_n)\mid\tau_{\mathfrak C}>T]
 \le\frac{M_q^{\rm stop}}{1-TU_N},\qquad
 \mathbb E[L_N(S_n)(|x|^s\mathbf1_{\{|x|>R\}})
       \mid\tau_{\mathfrak C}>T]
 \le\frac{M_q^{\rm stop}}{(1-TU_N)R^{q-s}}.       \tag{18.3g}
$$
The total-variation distance between the unconditioned law of any
path observable through $T$ and its law conditioned on this residence
event is at most $\min\{1,TU_N\}$. This conditioning calculation
does not assert attraction of different nonlinear population phases.
For the population, $\mu_j\{|x|>R\}\le H/R^p$ at *every* time.
In particular, $T_N=o(N^{1/3})$ makes the exit and finite-swarm tail
error vanish with constants independent of $N$; the physical horizon
is $hT_N$ and its work is $NT_N$ row updates. Setting $\lambda=0$
requires no externally imposed linear restoring force: the finite tests
are then $\chi>1-(1+\eta g_1)^{-q}$ for all $q\in\mathcal Q$ and
(18.3a). An exponent $24$ may be included in $\mathcal Q$ to supply
the moment used by the existing unbounded-reward mean-field modulus.
:::

:::{prf:proof}
The direct regional-count lemma bounds the **actual signed** copying
flux on $\mathfrak C$, simultaneously for every listed exponent:
$W_q^{\rm copy}\le(1-\chi)W_q+b_{q,\mathrm{sel}}$.
Consequently its excess $E_q$ is zero on this class. The complete
kinetic calculation of {prf:ref}`thm-slceg-full-moment`, including
clone-position jitter, capped collision velocity and the combined OU
and final position noise, gives on the same class
$$
 P_NW_q(S)\le r_qW_q(S)+B_q,
 \qquad W_q(\mathcal F_h\mu)\le r_qW_q(\mu)+B_q. \tag{18.3f}
$$
The displayed choice of $u_q$ solves $r_q=(1+t_q)/2<1$ if
$t_q>0$; if $t_q=0$ it gives $r_q=0$. No $N$ occurs in these
coefficients. If $\lambda=0$, $A_\lambda=1+\eta g_1$, yielding the
stated landscape/algorithm test directly.

For any $\mu\in\mathfrak C$, Markov's inequality gives positional
mass at least $1-H/L^p$ inside $B(0,L)$. At such a root, the event of
no accepted outgoing clone has probability at least $1-a_*$. On that
event there is no cloning jitter; incoming collisions can change its
velocity but the cap bounds it by $V_c$. The final position is a
$d$-dimensional Gaussian with variance $\tau^2I_d$ and center of norm
at most $A_\lambda L+b_0$. Its density at each point of
$B(0,R_c)$ is at least the Gaussian factor in $m_{\rm land}$.
Integration over the ball, root set, and no-outgoing event proves the
lower output core mass $m_{\rm land}$. The velocity remains in the
declared capped space. Therefore (18.3a) gives both strict full-map
self-map margins, so $\mu_n\in\mathfrak C$ by induction. Applying
(18.3f) along this trajectory and summing its geometric recursion
proves $W_q(\mu_n)\le M_q^{\rm pop}$.

For the finite swarm let
$D_{q,n}=\mathbb E[\mathbf1_{\{\tau_{\mathfrak C}>n\}}W_q(S_n)]$.
Because the output moment is nonnegative, and because (18.3f) applies
at every pre-exit configuration,
$$
 \begin{aligned}
 D_{q,n+1}
 &\le\mathbb E[\mathbf1_{\{\tau_{\mathfrak C}>n\}}
                         P_NW_q(S_n)]\\
 &\le r_qD_{q,n}+B_q\Pr\{\tau_{\mathfrak C}>n\}
 \le r_qD_{q,n}+B_q.
 \end{aligned}
$$
Induction gives
$D_{q,n}\le r_q^nM_{q,0}+B_q(1-r_q^n)/(1-r_q)
\le M_q^{\rm stop}$, proving (18.3b). This argument bounds the
weighted mass of *all* pre-exit configurations; it does not treat
successive particles or updates as independent.
On $|x|>R$, $|x|^s\le R^{-(q-s)}|x|^q$.
Integrating this pointwise inequality against $\mu_n$ and the stopped
empirical measure proves (18.3b'). In particular, the $q=24$ test
gives explicit $R^{-22}$ exterior control of quadratic rewards and
$R^{-16}$ control of eighth-order positional terms, with coefficients
$M_{24}^{\rm pop}$ and $M_{24}^{\rm stop}$ respectively.

The force-growth bound needed in {prf:ref}`thm-slcex-one-step` is
$|F(x)|\le g_0+(g_1+\lambda)|x|$, which gives exactly the displayed
$A_x,b_x$. Insert (18.3b) at exponent $2p$ into the averaged
truncation and bounded-test calculation of
{prf:ref}`cor-slcex-residence`. Its output $2p$-moment budget is
$Q_*$, and that corollary's optimized one-step first-exit probability
is $U_N$. Summing the first-exit events proves (18.3c).
On $\{\tau_{\mathfrak C}>T\}$, every empirical measure through $T$
has $p$th moment at most $H$, so Markov gives a *pathwise* tail
fraction at most $H/R^p$. This proves (18.3d). Splitting the
expectation at time $j$ according to survival through $j$, using the
trivial tail-fraction bound one on the complement, and then (18.3c)
proves (18.3e). The population tail statement follows from its
all-time $p$th-moment bound. Finally $U_N=O(N^{-1/3})$ for fixed
displayed constants, giving the stated time and work scales.
For (18.3g), the denominator is at least $1-TU_N$ by (18.3c),
while $\mathbf1_{\{\tau_{\mathfrak C}>T\}}\le
\mathbf1_{\{\tau_{\mathfrak C}>n\}}$ makes each numerator no larger
than (18.3b) or (18.3b'). For any event $A$ with positive probability,
the law $P(\cdot\mid A)$ has total-variation distance $P(A^c)$
from $P$; applying this identity on the full path and then projecting
to an observable proves the last conditioning bound.
:::

:::{prf:theorem} Mean-field trajectory estimate from structural tail closure
:label: thm-slcex-tail-meanfield-transfer

Specialize {prf:ref}`thm-slcex-selection-tail-closure` to $p=8$ and
$\mathcal Q\supset\{8,16\}$. Retain the actual all-alive kernel,
quadratic-growth raw reward and evaluated continuity constant $C_8(H)$
of {prf:ref}`thm-slct-quadratic-reward`. Let $\mu_0$ and every initial
empirical measure belong to $\mathfrak C$, and set
$\mu_n=\mathcal F_h^n\mu_0$. Write $G=A+4B_*^2$ for the actual
bounded-test consistency constant. With $J(R,\ell)$ from
{prf:ref}`thm-slct-trajectory`, define for $R>0$, $0<\ell\le1$,
$$
 a_N=\min\{1,2\ell+\tfrac12J(R,\ell)\sqrt{G/N}+2H/R^8\},
 \qquad C_H=C_8(H).
$$
Put $e_0=\mathbb E\mathsf d(L_N(S_0),\mu_0)$ and define the numerical
recursion $v_0=e_0$,
$v_{n+1}=\min\{1,a_N+C_Hv_n^{1/32}\}$.
For every $n\ge0$, the unchanged algorithm obeys
$$
 \mathbb E\mathsf d(L_N(S_n),\mu_n)
 \le\min\{1,v_n+\min(1,nU_N)\}.                    \tag{18.3h}
$$
If $TU_N<1$, its law conditioned on residence through $T$ satisfies,
for $n\le T$,
$$
 \mathbb E[\mathsf d(L_N(S_n),\mu_n)\mid\tau_{\mathfrak C}>T]
 \le\min\{1,v_n/(1-TU_N)\},                       \tag{18.3i}
$$
and the conditioned and unconditioned empirical-law marginals differ
in total variation by at most $\min\{1,TU_N\}$.

For explicit choices take $R_N=N^{1/(16d)}$,
$\ell_N=N^{-1/(16d)}$, $b_N=\max\{e_0,a_N\}$ and
$K_H=(1+C_H)^{32/31}$. Then
$$
 v_n\le\min\{1,K_Hb_N^{32^{-n}}\}.               \tag{18.3j}
$$
If $e_0\to0$, (18.3h) proves the quantitative fixed-horizon mean-field
limit with no separate tail-budget premise. It also proves convergence
uniformly over any growing integer horizon $T_N$ for which
$$
 T_NU_N\to0,\qquad
 32^{-T_N}\log(1/b_N)\to\infty.                  \tag{18.3k}
$$
For example, if $e_0=O(N^{-\gamma})$ for $\gamma>0$, every
$T_N\le(1-\varepsilon)\log_{32}\log N$, with fixed
$\varepsilon\in(0,1)$, meets these tests for large $N$.
The comparison is with the population solution from the same initial
law; no attraction between different stationary phases is used.
:::

:::{prf:proof}
On $\{\tau_{\mathfrak C}>n\}$, both $L_N(S_n)$ and $\mu_n$ have
eighth moment at most $H$. The proved full-map modulus thus gives
$\mathsf d(\mathcal F_hL_N(S_n),\mathcal F_h\mu_n)
\le C_H\mathsf d(L_N(S_n),\mu_n)^{1/32}$.
The conditional bounded-test consistency bound applies to every
entering configuration. Repeat the finite-cell proof of
{prf:ref}`thm-slct-trajectory` at fixed $R,\ell$. On a pre-exit
input, (18.3f) at exponent eight bounds both the conditional expected
empirical output and the population output moment by
$r_8H+B_8=H-\Delta_H$. Each expected exterior mass is therefore at
most $H/R^8$, and the cell sampling defect is at most $a_N$.
This calculation uses $G$ for all cell indicators and does not require
independent output rows.

Let $D_n=\mathbb E[\mathbf1_{\{\tau_{\mathfrak C}>n\}}
\mathsf d(L_N(S_n),\mu_n)]$. Since the next residence event is
contained in the current one and the metric is nonnegative,
$$
 D_{n+1}\le a_N+
 C_H\mathbb E[(\mathbf1_{\{\tau_{\mathfrak C}>n\}}
 \mathsf d(L_N(S_n),\mu_n))^{1/32}]
 \le a_N+C_HD_n^{1/32},
$$
where the last step is Jensen. Induction gives $D_n\le v_n$.
The complementary event costs at most one and has probability at most
$\min(1,nU_N)$ by (18.3c), proving (18.3h). For (18.3i), the
conditional numerator is at most $D_n$ and its denominator is at
least $1-TU_N$. The exact path-conditioning identity used in
(18.3g) gives the total-variation assertion.

For $0<b_N\le1$, induction in the scalar recursion gives (18.3j):
$1+C_HK_H^{1/32}\le K_H$ and
$b_N\le b_N^{32^{-(n+1)}}$. The zero case follows directly.
The displayed cell count has $J(R_N,\ell_N)=O(N^{3/16})$;
the other three terms in $a_N$ are of orders
$N^{-1/(16d)}$, $N^{-5/16}$ and $N^{-1/(2d)}$,
respectively. Thus $a_N\to0$ with all coefficients in its exact
definition. Conditions (18.3k) make (18.3j) and the exit cost vanish
uniformly over $n\le T_N$. Under the example initialization rate,
$b_N=O(N^{-\gamma'})$ for some $\gamma'>0$, so
$32^{-T_N}\log(1/b_N)\ge c(\log N)^\varepsilon\to\infty$;
also $T_NU_N\to0$ because $U_N=O(N^{-1/3})$.
:::

:::{prf:corollary} Explicit independent initialization for the structural mean-field estimate
:label: cor-slcex-tail-iid-meanfield

Keep the kernel, parameter tests and constants of
{prf:ref}`thm-slcex-tail-meanfield-transfer`, replacing its
almost-sure empirical initialization condition as follows. Suppose
the initial rows are independent with common law $\mu_0$ and the numerical
initialization margins
$$
 W_8(\mu_0)\le H-\Delta_{H,0},\qquad
 \mu_0(\overline B(0,R_c)\times\overline B_{V_{\max}})
 \ge m+\Delta_{m,0},\qquad
 W_{16}(\mu_0)\le M_{16,0},
$$
where $\Delta_{H,0},\Delta_{m,0}>0$. Set
$$
 \delta_{N,0}=\min\left\{1,
 \frac{M_{16,0}}{N\Delta_{H,0}^2}
 +\frac{1}{4N\Delta_{m,0}^2}\right\}.
$$
With the same $R,\ell,J$ as above, choose
$$
 e_{N,0}=\min\{1,2\ell+J(R,\ell)/(2\sqrt N)+2H/R^8\}
$$
and define $v_n$ by the preceding scalar recursion. Then
$$
 \mathbb E\mathsf d(L_N(S_n),\mu_n)
 \le\min\{1,v_n+\delta_{N,0}+\min(1,nU_N)\}.     \tag{18.3l}
$$
This is a full quantitative mean-field approximation for independent
initialization, with the tail and initial-class failure costs both
displayed. Under the preceding $R_N,\ell_N$ choices, its fixed-horizon
error vanishes. Its growing-horizon conclusion holds whenever
(18.3k) holds with this $e_{N,0}$.
:::

:::{prf:proof}
For $Y_i=|X_i|^8$, independence gives
$\operatorname{Var}(N^{-1}\sum_iY_i)
\le M_{16,0}/N$. The core indicators are independent Bernoulli
variables with empirical-mean variance at most $1/(4N)$.
Chebyshev and the two strict initialization margins therefore give
$\Pr\{L_N(S_0)\notin\mathfrak C\}\le\delta_{N,0}$.
For the initial empirical metric, apply the same finite-cell matching
inequality as {prf:ref}`thm-slct-trajectory`. Each cell indicator
has empirical variance at most $1/(4N)$, and both expected exterior
masses are at most $H/R^8$, proving the displayed $e_{N,0}$.
Define the first exit as zero on the initial-class failure event.
The stopped recursion in the theorem then starts with
$D_0\le e_{N,0}$ and is unchanged on the pre-exit event; the
probability of its complement at time $n$ is at most
$\delta_{N,0}+\min(1,nU_N)$. Splitting the bounded metric across
these events proves (18.3l). The stated limits follow from the
explicit decay of $e_{N,0},\delta_{N,0},U_N$ and (18.3j).
:::

:::{prf:remark} The higher-moment condition cannot be dropped silently
:label: rem-slcex-moment-scope

A class defined only by $W_p\le H$ and a core mass lower bound need
not bound $W_{2p}$. One empirical atom can have $|x|^p$ of order $N$
while the empirical $p$th moment stays bounded; its copying count can
then change that observable at order one. The extra entering or stopped
higher-moment budget above is therefore a substantive tail requirement.
It is supplied without a separate stopped-moment premise by
{prf:ref}`thm-slcex-selection-tail-closure` when the signed regional
selection coefficients contract at exponent $2p$. The resulting
residence estimate does not prove invariance of the finite-particle
class for all time.
:::

:::{prf:corollary} Unconditional higher-moment control and an explicit logarithmic residence window
:label: cor-slcex-global-growth-window

Keep the actual force-growth constants and analytically certified
self-map margins of {prf:ref}`thm-slcex-one-step`. Assume only
$L_N(S_0)\in\mathfrak C$ almost surely and
$\sup_N\mathbb EW_{2p}(S_0)\le M_0<\infty$.
Define the global, unrestricted full-update moment coefficients
$$
 A_g=3^{2p-1}2^{2p-1}A_x^{2p}r_C>1,
$$
$$
 b_g=3^{2p-1}
 [2^{2p-1}A_x^{2p}\sigma_J^{2p}m_{d,2p}
                   +b_x^{2p}+\tau^{2p}m_{d,2p}],
 \qquad \overline M=M_0+\frac{b_g}{A_g-1}.
$$
Let $a_g=A_g^{2/3}>1$ and
$C_g=3(16G\overline M^2/\Delta_H^4)^{1/3}$. For every integer
$T\ge1$, the actual first-exit time satisfies
$$
 \boxed{\quad
 \Pr\{\tau_{\mathfrak C}\le T\}
 \le\min\left\{1,
       \frac{TG}{N\Delta_m^2}
       +C_gN^{-1/3}\frac{a_g(a_g^T-1)}{a_g-1}\right\}.
 \quad}
$$
In particular, fix any $0<\theta<1$. For every integer
$1\le T_N\le\theta\log N/(2\log A_g)$,
$$
 \Pr\{\tau_{\mathfrak C}\le T_N\}
 \le\frac{\theta G\log N}{2\Delta_m^2\log A_g}\,N^{-1}
       +\frac{C_ga_g}{a_g-1}N^{-(1-\theta)/3}
 \longrightarrow0.
$$
This proves a growing residence window using only an initial moment
budget and the global finite-window moment calculation. No stationary
higher-moment bound or unproved uniform coverage estimate is assumed.
Physical time is $hT_N$ and the work is $NT_N$ row updates.
:::

:::{prf:proof}
The global copying multiplicity and Gaussian calculation in
{prf:ref}`thm-slcex-one-step` did not use membership in $\mathfrak C$.
They therefore prove
$$
 P_NW_{2p}(S)\le A_gW_{2p}(S)+b_g
$$
for every configuration, and the same bound for the population output
at its empirical input. Iteration gives
$$
 \mathbb EW_{2p}(S_j)
 \le A_g^jM_0+b_g\frac{A_g^j-1}{A_g-1}
 \le\overline M A_g^j.
$$
The common expected output moment used for step $j+1$ is consequently
at most $Q_j=\overline M A_g^{j+1}$. Multiplying the truncated-test
and tail estimates by the pre-exit indicator only decreases the
nonnegative moment contributions; on that event the population margins
hold. Choose the deterministic truncation separately for each step,
using this $Q_j$. The optimized one-step argument bounds the probability
of a first exit on step $j+1$ by
$$
 \frac{G}{N\Delta_m^2}
 +3\left(\frac{16GQ_j^2}{N\Delta_H^4}\right)^{1/3}
 =\frac{G}{N\Delta_m^2}+C_gN^{-1/3}A_g^{2(j+1)/3}.
$$
Sum for $0\le j<T$. The geometric sum is exactly
$\sum_{j=1}^T a_g^j=a_g(a_g^T-1)/(a_g-1)$, proving the first bound.
Under the logarithmic schedule,
$a_g^{T_N}=\exp[(2/3)T_N\log A_g]\le N^{\theta/3}$.
Substitute this upper bound and the specified bound on $T_N$ to obtain
the explicit decaying expression. The cases $G=0$ or $\overline M=0$
follow from the unoptimized zero-error limits already proved.
:::

(sec-slcfu-phase-compatibility)=
### 18.4. Compatibility of stationary phases with the order of limits

:::{prf:remark} The unconditional terminal-box limit has an explicit obstruction
:label: rem-slcfu-unconditioned-box

For the unchanged terminal-box kernel, the actual final Gaussian noise
already decides the unconditional uniform-time question.
{prf:ref}`thm-chaos-unconditioned-extinction-obstruction` derives
$q_D=1-\prod_k[2\Phi((u_k-\ell_k)/(2\sigma_x\sqrt h))-1]>0$ and the
population-independent positive landing bound $a_0$, with all parameter
dependencies displayed. For every finite $N$,

$$
 \Pr(\tau_N>n)\le(1-q_D^N)^n,\qquad
 \mathbb E|N^{-1}M_n-m(\mathcal F_h^n\mu_0)|
       \ge a_0-(1-q_D^N)^n\quad(n\ge1).
$$

The error is therefore at least $a_0/2$ by the explicit horizon
$\lceil q_D^{-N}\log(2/a_0)\rceil$, and its supremum over all time
cannot vanish as $N\to\infty$. This follows from the declared algorithm
for every landscape covered by its finite-horizon theorem; it introduces
no attraction or instability hypothesis. The conclusion concerns the
unconditioned absorbing law. The bound $q_D^N$ is a state-uniform
*lower* extinction hazard, not a typical one-step failure estimate.
{prf:ref}`cor-chaos-exact-hazard-recovery-window` gives the exact
preparation-averaged hazard and the complementary exponential lower
survival window. {prf:ref}`prop-chaos-safe-center-noise` gives the
population-fraction and path-TV bounds from safe kinetic centers,
including the zero-displacement-noise endpoint while retaining clone
jitter. {prf:ref}`cor-chaos-noise-conditioned-law` transfers those
noise-dependent bounds to the actual survivor-conditioned law and its
one-step mean-field estimate. {prf:ref}`thm-chaos-conditioned-propagation`
then transfers the existing finite-horizon chaos theorem to the
survivor-conditioned full path, with the exact extinction hazard as
its TV cost and an explicit finite-row sampling term. This is not an
obstruction to the
survival-conditioned problem: {prf:ref}`thm-chaos-survival-uniform-floor`
proves the existing alive-fraction failure bound uniformly at every
conditioned observation time, and
{prf:ref}`thm-chaos-conditioned-quantitative-map` gives the corresponding
one-step mean-field error $\varepsilon_N+2\delta_N$, with no cumulative
survival denominator. The normalized alive law has the same result with
its explicit $a_0^{-1}$ normalization factor. QSD existence for the
general canonical box force and quantitative stationary mean-field
identification are proved in
{prf:ref}`thm-chaos-general-box-qsd-existence` and
{prf:ref}`cor-chaos-conditioned-stationary-defect`. The conservative
unbounded kernel has $q_D=0$ and a separate confinement problem.
:::

:::{prf:proposition} An explicit obstruction to simultaneous phasewise uniform-time chaos
:label: prop-slcfu-phase-obstruction

Let $\mathsf d$ be the chapter's diameter-one bounded transport metric on
population laws, and suppose the same nonlinear map $\mathcal F_h$ has two
stationary laws $\pi_1,\pi_2$ with
$s_{12}=\mathsf d(\pi_1,\pi_2)>0$. For each $N$, consider the actual
finite-particle kernel $P_N$. Suppose its laws from two specified
initializations $\Lambda_{N,1},\Lambda_{N,2}$ both converge to the same
invariant probability $\Pi_N$, at least for the bounded observables
$S\mapsto\mathsf d(L_N(S),\pi_i)$. Define the actual errors

$$
 e_{N,i}=\sup_{n\ge0}\mathbb E_{\Lambda_{N,i}}
                   \mathsf d(L_N(S_n),\pi_i),\qquad i=1,2.
$$

Then, for every such $N$,

$$
 \boxed{e_{N,1}+e_{N,2}\ge s_{12},\qquad
        \max_i e_{N,i}\ge s_{12}/2.}
$$

Consequently finite-horizon mean-field convergence from initial empirical
laws approaching the respective $\pi_i$ is compatible with both phases,
but it cannot be upgraded to uniform-time convergence to both initial
phases when the stated finite-particle ergodicity holds. Confinement,
including an $N$-independent entropy bound, does not remove this obstruction.
:::

:::{prf:proof}
For every empirical law $\nu$, the triangle inequality gives
$s_{12}\le\mathsf d(\nu,\pi_1)+\mathsf d(\nu,\pi_2)$.
Integrate against $\Pi_N\circ L_N^{-1}$. The convergence of the two
bounded observables identifies each integral with its corresponding
long-time expectation, which is at most $e_{N,i}$. Adding yields the
first inequality and then the second. No assumption that the empirical
law under $\Pi_N$ is deterministic is used. In particular the argument
also covers an invariant mixture of phases.
:::

:::{prf:lemma} Quantitative phase-set coverage and finite residence horizons
:label: lem-slcfu-coverage-budget

Let $\mathfrak G$ be a declared Borel set of population laws. For the
actual full finite-particle kernel, let numbers $u_N,r_N\in[0,1]$ satisfy

$$
 \Pr\{L_N(S_{j+1})\notin\mathfrak G\mid\mathscr F_j\}\le u_N
       \quad\text{on }\{L_N(S_j)\in\mathfrak G\},
$$
$$
 \Pr\{L_N(S_{j+1})\in\mathfrak G\mid\mathscr F_j\}\ge r_N
       \quad\text{on }\{L_N(S_j)\notin\mathfrak G\}.
$$

These are bounds for the complete conditional transition probabilities,
including all intervening cloning, collisions, jitter and kinetics. They
may be obtained from the regional landing and count estimates; a failed
lower bound is recorded as $r_N=0$. Put
$\Delta_N(j)=\Pr\{L_N(S_j)\notin\mathfrak G\}$.
For $r_N>0$,

$$
 \Delta_N(j)\le\min\left\{1,(1-r_N)^j\Delta_N(0)
             +\frac{u_N}{r_N}[1-(1-r_N)^j]\right\}.
$$

Under any invariant particle law, $\Delta_N\le\min\{1,u_N/r_N\}$.
If only the exit bound is available and the process starts in
$\mathfrak G$, its first exit time $\tau_{\mathfrak G}$ satisfies

$$
 \Pr\{\tau_{\mathfrak G}\le T\}
       \le1-(1-u_N)^T\le Tu_N.
$$

Thus a within-phase comparison with good-event error $b_N(T)$ has
unconditional error at most $b_N(T)+1-(1-u_N)^T$ in a diameter-one
metric. A sufficient joint-limit schedule is $b_N(T_N)\to0$ and
$T_Nu_N\to0$; physical time is $hT_N$, with $NT_N$ row updates.
No positive fixed-$N$ exit bound is discarded by sending $T$ to infinity.
:::

:::{prf:proof}
Conditioning on the current membership gives
$\Delta_N(j+1)\le u_N(1-\Delta_N(j))+(1-r_N)\Delta_N(j)
\le u_N+(1-r_N)\Delta_N(j)$.
Iteration proves the geometric sum. Stationarity in the same inequality
gives $r_N\Delta_N\le u_N$. For residence, condition on survival
through step $j$: the next survival probability is at least $1-u_N$.
Induction gives $\Pr\{\tau_{\mathfrak G}>T\}\ge(1-u_N)^T$,
and the union bound gives its linear relaxation. On the exit event
charge the metric diameter one; on its complement use the stated
comparison bound. This proves the last assertions while retaining all
inter-walker and temporal dependence.
:::

(sec-slcgt-growing-trajectory)=
### 18.5. Mean-field convergence on growing intervals and the infinite population path

:::{prf:theorem} Uniform trajectory approximation through a diverging horizon
:label: thm-slcgt-growing-trajectory

Use the actual conservative all-alive canonical update, with its current-step measurement and cloning companions, sampled fitness, complete collision components, Gaussian jitter, BAOAB and cap. Retain all canonical feature, regularization and rooted-component hypotheses
of {prf:ref}`thm-slct-quadratic-reward` and its referenced bounded-test
consistency theorem, with finite constants $A,B_*$. In particular the
actual force and raw-reward profiles obey:

$$
|F(x)|\le G_0+G_1|x|,\quad \operatorname{Lip}(F)\le L_F,
$$

$$
|R(z)|\le K_0+K_2|x|^2,\qquad
|R(z)-R(z')|\le(L_0+L_1R_0)|z-z'|
\quad (|x|,|x'|\le R_0).
$$

All constants, noise scales, feature parameters, fitness exponents and regularizers are fixed independently of $N$. There is no assumed attraction, stationary law, bounded force-center profile, or uniform-in-time moment bound.

Let $\mu_n=\mathcal F_h^n\mu_0$. Assume the explicit initial budget

$$
\mu_0|x|^{24}\le M_0,\qquad
\sup_N\mathbb EW_{24}(S_0)\le M_0<\infty,
\qquad
\mathbb E\mathsf d(L_N(S_0),\mu_0)\le A_{\rm init}N^{-\alpha},
\quad \alpha=1/(16d).
$$

The input velocities are capped. Define the following primitive moment and consistency constants:

$$
A_x=1+\eta G_1,\quad b_x=BV_c+\eta G_0,\quad
\tau^2=c^2q^2+s^2,\quad r_C=1+2/\kappa_C,
$$

$$
m_{d,24}=\prod_{j=0}^{11}(d+2j),\qquad
A_{24}=3^{23}2^{23}A_x^{24}r_C>1,
$$

$$
b_{24}=3^{23}\left[2^{23}A_x^{24}\sigma_J^{24}m_{d,24}
                       +b_x^{24}+\tau^{24}m_{d,24}\right],
\qquad \overline M=M_0+\frac{b_{24}}{A_{24}-1},
$$

$$
G=A+4B_*^2,\quad C_*=C_8(1),\quad
\overline C=C_*\sqrt{1+2\overline M},
$$

$$
J_0=(2+2\sqrt{2d})^d(2+2V_{\max}\sqrt{2d})^d,
\qquad A_{\rm samp}=2+\tfrac12J_0\sqrt G+2\overline M^{1/12}.
$$

Here $A,B_*$ are exactly the proved canonical bounded-test consistency constants, and $C_8(1)$ is the displayed structural modulus constant for the actual unbounded raw reward.
For every integer $T\ge0$, put

$$
C_T=\overline C A_{24}^{T/2},\qquad
a_{N,T}=\max\{A_{\rm init},A_{\rm samp}A_{24}^{T/12}\}N^{-\alpha},
$$

$$
V_{N,T}=\begin{cases}
\min\{1,(1+C_T)^{32/31}a_{N,T}^{32^{-T}}\},&a_{N,T}\le1,\\
1,&a_{N,T}>1.
\end{cases}
$$

Then the actual swarm and deterministic population trajectory obey

$$
\boxed{\quad
\max_{0\le n\le T}\mathbb E\mathsf d(L_N(S_n),\mu_n)
\le V_{N,T}.\quad}
$$

In particular, let $L_N=\log(N+e)$ and choose

$$
T_N=\max\left\{0,\left\lfloor\frac{\log L_N}{4\log32}\right\rfloor\right\}.
$$

Then $T_N\to\infty$ and both

$$
\max_{0\le n\le T_N}\mathbb E\mathsf d(L_N(S_n),\mu_n)\to0,
\qquad
\mathbb E\max_{0\le n\le T_N}\mathsf d(L_N(S_n),\mu_n)\to0.
$$

For every tolerance $\varepsilon>0$, the explicit simultaneous-trajectory confidence bound is

$$
\Pr\left\{\max_{0\le n\le T_N}\mathsf d(L_N(S_n),\mu_n)>\varepsilon\right\}
\le\min\{1,(T_N+1)V_{N,T_N}/\varepsilon\}.
$$

The physical horizon is $hT_N$ and the number of row updates is $NT_N$. This is a proved growing-time mean-field limit. It is not a stationary limit or a claim about unrestricted diagonals $n_N\to\infty$.
:::

:::{prf:proof}
**Unrestricted moment propagation.** The actual copying multiplicity, Gaussian jitter and kinetic calculation gives, for every input configuration,

$$
P_NW_{24}(S)\le A_{24}W_{24}(S)+b_{24}.
$$

The same inequality holds for the rooted population law. This calculation uses $|V^C|\le V_c$ on every collision component and the actual frozen source; it does not replace component dependence by independent walkers. Iteration yields

$$
\mathbb EW_{24}(S_n),\ \mu_n|x|^{24}
\le A_{24}^nM_0+b_{24}\frac{A_{24}^n-1}{A_{24}-1}
\le\overline M A_{24}^n=:M_n.
$$

The expected $24$th moment of $\mathcal F_hL_N(S_n)$ also obeys the step-$n+1$ bound, by applying the same affine moment inequality and retaining the exact geometric expression before the last inequality.

**Averaged modulus with dependent error and moments.** Write $\widehat\mu_n=L_N(S_n)$, $\Delta_n=\mathsf d(\widehat\mu_n,\mu_n)$ and
$H_n=\max\{1,\widehat\mu_n|x|^8,\mu_n|x|^8\}$. Jensen within either probability measure gives its cubed eighth moment at most its $24$th moment. Therefore

$$
\mathbb EH_n^3\le1+2M_n.
$$

The proved moment-class modulus, Cauchy--Schwarz, and concavity of $t^{1/16}$ give

$$
\mathbb E\mathsf d(\mathcal F_h\widehat\mu_n,\mathcal F_h\mu_n)
\le C_*\sqrt{\mathbb EH_n^3}\sqrt{\mathbb E\Delta_n^{1/16}}
\le\overline C A_{24}^{n/2}(\mathbb E\Delta_n)^{1/32}.
$$

No independence between the empirical moment and its approximation error is used.

**Sampling error of the actual full update.** Conditional bounded-test consistency supplies $G/N$. Use spatial cutoff $N^{1/(16d)}$ and cell diameter $N^{-1/(16d)}$. Their number is at most $J_0N^{3/16}$. Both output positional second moments are bounded in expectation by $M_{n+1}^{1/12}$, because $\int|x|^2\le(\int|x|^{24})^{1/12}$ and the outer expectation uses concavity. The established finite-cell bound consequently gives

$$
\begin{aligned}
&\mathbb E\mathsf d(L_N(S_{n+1}),\mathcal F_hL_N(S_n))\\
&\quad\le\min\left\{1,\,
2N^{-1/(16d)}+\tfrac12J_0\sqrt G\,N^{-5/16}
                +2M_{n+1}^{1/12}N^{-1/(8d)}\right\}\\
&\quad\le\min\{1,A_{\rm samp}A_{24}^{(n+1)/12}N^{-\alpha}\}.
\end{aligned}
$$

Thus for $n<T$ the triangle inequality yields
$e_{n+1}\le\min\{1,a_{N,T}+C_Te_n^{1/32}\}$, where $e_n=\mathbb E\Delta_n$ and $e_0\le a_{N,T}$. If $a_{N,T}\le1$, induction gives

$$
e_n\le(1+C_T)^{\sum_{j=0}^{n-1}32^{-j}}a_{N,T}^{32^{-n}}
\le(1+C_T)^{32/31}a_{N,T}^{32^{-n}}.
$$

For the induction, use $a_{N,T}\le a_{N,T}^{32^{-(n+1)}}$ and
$1+C_T(1+C_T)^{S/32}\le(1+C_T)^{1+S/32}$ for $S\ge0$. The bound is nondecreasing in $n$ when $a_{N,T}\le1$. Clipping at one proves the displayed $V_{N,T}$ bound. If $a_{N,T}>1$, the metric diameter proves it directly.

**The growing horizon still makes every error vanish.** Set

$$
A_0=\max\{A_{\rm init},A_{\rm samp}\},\quad
k_s=\frac{\log A_{24}}{48\log32},\quad
k_C=\frac{\log A_{24}}{8\log32}.
$$

For $L=L_N$ the horizon formula implies

$$
a_{N,T_N}\le A_0L^{k_s}N^{-\alpha},\qquad
C_{T_N}\le\overline C L^{k_C},\qquad
32^{-T_N}\ge L^{-1/4}.
$$

The first bound tends to zero. Once its logarithm is nonpositive, the logarithm of the unclipped bound for $V_{N,T_N}$ is at most

$$
\frac{32}{31}[\log(1+\overline C)+k_C\log L]
 +L^{-1/4}[\log A_0+k_s\log L-\alpha\log N].
$$

This equals $-\alpha(\log N)^{3/4}+O(\log\log N)$ with coefficients fixed by the displayed primitive formulas, and tends to $-\infty$. For a direct numerical threshold, if

$$
L\ge\max\{4,\,4\log A_0/\alpha,\,(4k_s/\alpha)^2\},\qquad N\ge2,
$$

then $\log a_{N,T_N}\le-\alpha L/4$; use $\log L\le\sqrt L$ and $\log N\ge L-1$. Hence the explicit stronger upper bound is

$$
V_{N,T_N}\le\min\left\{1,
\exp\left(\frac{32}{31}[\log(1+\overline C)+k_C\log L]
                      -\frac\alpha4 L^{3/4}\right)\right\}.
$$

Multiplying this by $T_N+1=O(\log L)$ still gives zero in the limit. Since the maximum of nonnegative errors is at most their sum,
$\mathbb E\max_{n\le T_N}\Delta_n\le\min\{1,(T_N+1)V_{N,T_N}\}$. Markov's inequality proves the simultaneous confidence bound and completes the argument.
:::

:::{prf:corollary} Explicit initialization from independent rows
:label: cor-slcgt-iid-initialization

If the initial rows are sampled independently from the capped law $\mu_0$ with $\mu_0|x|^{24}\le M_0$, the theorem's initial approximation hypothesis is supplied by

$$
A_{\rm init}=2+\tfrac12J_0+2M_0^{1/12}.
$$

No other independence assumption on the evolving swarm is introduced.
:::

:::{prf:proof}
For every bounded measurable test with $|\varphi|\le1$, the initial empirical average is unbiased and has variance at most $1/N$. Apply the same spatial cutoff and finite-cell calculation with $G=1$ and second moment at most $M_0^{1/12}$. Its three terms are bounded by $A_{\rm init}N^{-\alpha}$. The initial expected $24$th moment equals that of $\mu_0$. These are precisely the two required input bounds.
:::

:::{prf:corollary} Uniform-ball initialization and quantitative fixed-row chaos
:label: cor-slcgt-ball-and-rows

For independent initial rows whose positions are uniform on $B(0,R_I)$, $R_I>0$, and whose velocities obey the configured cap, take

$$
M_0=\frac{d}{d+24}R_I^{24}.
$$

For positions uniform on $B(x_c,R_I)$ one may instead use
$M_0=(|x_c|+R_I)^{24}$. The preceding corollary then supplies the complete numerical $A_{\rm init}$.

More generally, whenever the initialized swarm law is exchangeable, so are its actual evolved laws. For $1\le k\le N$ define the product transport cost

$$
c_k((z_i),(z_i'))=\min\{1,\sum_{i=1}^k\min(1,|z_i-z_i'|)\}.
$$

For every $0\le n\le T_N$, the transport distance between the actual $k$-row marginal and $\mu_n^{\otimes k}$ is at most

$$
\min\{1,kV_{N,T_N}+k(k-1)/(2N)\}.
$$

Thus fixed-$k$ chaos is uniform on the displayed diverging horizon. In particular, any $n_N\le T_N$ with $n_N\to\infty$ gives a joint growing-time/population-size approximation to the moving target $\mathcal F_h^{n_N}\mu_0$. It does not require that this target have a stationary limit.
:::

:::{prf:proof}
Radial integration of the uniform-ball density gives
$\mathbb E|X|^{24}=dR_I^{-d}\int_0^{R_I}r^{d+23}dr=dR_I^{24}/(d+24)$. For a shifted ball use $|X|\le|x_c|+R_I$. The velocity coordinates do not enter this positional moment budget.

Permutation equivariance of the actual kernel preserves exchangeability. Sampling $k$ distinct uniform labels then has the actual $k$-row marginal law. Couple these labels to $k$ labels sampled with replacement; their disagreement probability is at most $\sum_{j=0}^{k-1}j/N=k(k-1)/(2N)$. Conditional on the swarm, the latter samples have law $L_N(S_n)^{\otimes k}$. A coordinatewise transport coupling costs at most $k\mathsf d(L_N(S_n),\mu_n)$. The product cost is bounded by one, so the label disagreement contributes at most its probability. Average and apply the theorem's uniform error bound.
:::

:::{prf:corollary} The law of the entire population trajectory
:label: cor-slcgt-infinite-population-path

Let $\mathbf X^N=(L_N(S_n))_{n\ge0}$ be the random path of empirical population laws and $\boldsymbol\mu=(\mathcal F_h^n\mu_0)_{n\ge0}$ its deterministic population path. On this infinite product of population spaces use

$$
D_{\rm path}(\boldsymbol\nu,\boldsymbol\lambda)
=\sum_{n=0}^{\infty}2^{-(n+1)}\mathsf d(\nu_n,\lambda_n).
$$

Then the complete infinite-trajectory law has the explicit bound

$$
\mathbb E D_{\rm path}(\mathbf X^N,\boldsymbol\mu)
\le V_{N,T_N}+2^{-(T_N+1)}\longrightarrow0.
$$

Consequently $\mathbf X^N$ converges in probability to $\boldsymbol\mu$ in this product-topology metric and its law converges weakly to $\delta_{\boldsymbol\mu}$. The probability of path distance exceeding $\varepsilon>0$ is at most the displayed bound divided by $\varepsilon$, capped at one.

This is the law of the path of population measures. It is not a tagged walker's path law, a bound on $\sup_{n\ge0}\mathsf d(L_N(S_n),\mu_n)$, or an exchange of stationary limits.
:::

:::{prf:proof}
Split the nonnegative series after index $T_N$. The expected finite prefix is at most $V_{N,T_N}$ times a sum of weights at most one. Since $\mathsf d\le1$, the remaining deterministic tail is at most
$\sum_{n=T_N+1}^\infty2^{-(n+1)}=2^{-(T_N+1)}$. Both terms tend to zero. Markov's inequality proves convergence in probability and the explicit confidence bound; convergence in probability to a deterministic point in this metric implies weak convergence of the path laws to its point mass.
:::

(sec-slck-unchanged-accounting)=
### 18.6. Keystone accounting for the unchanged cloning kernel

:::{prf:proposition} Exact source law with all configured cloning parameters retained
:label: prop-slck-symmetric-source

Consider an admissible entering all-alive population
$\mu_r=\tfrac12\delta_{(re,0,1)}+\tfrac12\delta_{(-re,0,1)}$,
where $|e|=1$, $r>0$, both positions are valid and their raw rewards
are equal. This specifies an entering configuration, not an objective
function or a change in the transition. If the endpoint-separation
axiom {prf:ref}`axiom-non-deceptive-landscape` is being used, take
$2r<L_{\rm grad}$; no claim of verifying other global landscape
conditions follows merely from this pair. Keep every configured
comparison, rescaling, acceptance, jitter and collision parameter.
Put

$$
D_r=\frac{2R_xr}{R_x+r},\qquad
w_b=e^{-D_r^2/(2\epsilon_b^2)},\qquad
q_b=\frac{w_b}{1+w_b},\quad b\in\{D,C\},
$$
$$
s_0=\delta_D,\quad s_1=\sqrt{D_r^2+\delta_D^2},\quad
\Delta s=s_1-s_0,\quad
\widehat s=\sqrt{q_D(1-q_D)(\Delta s)^2+\sigma_s^2}.
$$

For the actual logistic maps $g_b(t)=\eta_b+A_b/(1+e^{-t})$, set

$$
F_L=g_r(0)^{p_r}g_s(-q_D\Delta s/\widehat s)^{p_s},\qquad
F_H=g_r(0)^{p_r}g_s((1-q_D)\Delta s/\widehat s)^{p_s},
$$
$$
a_r=\min\{1,(F_H-F_L)/(s_c(F_L+\epsilon_c))\},\qquad
P_r=q_D(1-q_D)a_r.
$$

The active-diversity axiom $p_s>0$ gives $P_r>0$. The selected frozen
source, before the configured jitter, has positional law exactly
$\mu_r^x$. With the actual jitter amplitude $\sigma_J$, the complete
cloning preparation law is

$$
\mathcal C(\mu_r)=
\left[(1-P_r)\mu_r^x+
 P_r(\mu_r^x*\mathcal N(0,\sigma_J^2I_d))\right]
 \otimes\delta_0\otimes\delta_1,
$$

and its positional variance is $r^2+d\sigma_J^2P_r$.
Every component collision is included. No kinetic parameter is changed;
this identity describes the actual intermediate law to which kinetics
is subsequently applied.
:::

:::{prf:proof}
The comparison feature difference between the two positions is $D_r$.
At either physical sign, the measurement is across the two sites with
probability $q_D$. Thus the measured separation takes values $s_0,s_1$
with probabilities $1-q_D,q_D$, independently of the recipient's
physical sign. Its mean and variance give precisely the displayed
standardized values. The raw reward standardized value is zero because
the endpoint rewards agree, with the positive variance floor retained.
This proves $F_L,F_H$ and their strict order for $p_s>0$.

Only a low-mark recipient and high-mark donor can create an accepted
edge; that pair has probability $(1-q_D)q_D$, and its gate equals $a_r$.
The actual cloning proposal is across the two sites with probability
$q_C$, independently of the high/low measurement labels conditional
on the physical types. Consequently the probability of switching the
root's physical sign is $P_rq_C$ in either direction. Starting from
equal masses, the selected source therefore has equal masses as well.
Acceptance has probability $P_r$ independently of the root sign, and
conditional on acceptance the selected donor sign is also symmetric.
All velocities are initially zero, so both the component average and
relative velocities in the exact collision formula are zero for every
accepted graph and every rotation. Accepted roots receive their actual
independent Gaussian jitter; unaccepted roots retain their position.
This gives the displayed mixture law and its variance. The preparation
retains the alive mark until the prescribed terminal classification.
:::

:::{prf:remark} The exact implication supplied by Keystone
:label: rem-slck-pressure-increment

The source identity is compatible with positive Keystone activity. For
two such entering radii $r\ne s$, the centered squared transport cost
between their selected source laws is still $(r-s)^2$, while their
error-weighted acceptance is $(P_r+P_s)(r-s)^2>0$. Equal-sign pairing
attains the cost; for any pairing, Cauchy--Schwarz bounds the cross
moment by $rs$, giving the reverse inequality. This calculation does
not identify the full kinetic output law or assert failure of full-update
convergence. It verifies that a negative source-error increment cannot
be substituted for positive acceptance activity.

The already proved {prf:ref}`thm-slcn-keystone-power` supplies a
zero-threshold activity bound $k_{\rm key}W_N^{5+4d}-E_{\max}/N^2$,
with its displayed primitive constants. That estimate is retained.
For any actual coupled accepted plans, write $d_i=x_i-y_i$ and
$r_i=A_i(x_{J_i}-x_i)-\widetilde A_i(y_{\widetilde J_i}-y_i)$.
The exact positional increment under shared recipient Gaussian jitter is

$$
\frac1N\mathbb E_{\xi}\sum_i|x_i^c-y_i^c|^2
-\frac1N\sum_i|d_i|^2
=\frac1N\sum_i\left[2d_i\cdot r_i+|r_i|^2
       +d\sigma_J^2(A_i-\widetilde A_i)^2\right].
$$

This is {prf:ref}`thm-cloning-incremental-cluster-balance` before its
geometric decomposition. Centering additionally subtracts the increment
of the squared barycenter difference; component collisions and the kinetic
stages have their own exact increments. A full convergence proof must
obtain its negative term from this unchanged-kernel expression and its
kinetic composition. The activity bound does not supply a sign for the
remaining donor, barycenter or cross-component terms by itself. No new
convergence axiom is introduced by retaining those algebraic terms.
:::

:::{prf:remark} Reproducible algebraic and numerical certificates
:label: rem-slc-computational-validation

The {download}`validation program <../../../validate_structural_landscape.py>`
{download}`transition checks <../../../validate_structural_transitions.py>`,
{download}`decision checks <../../../validate_structural_decisions.py>`,
and {download}`long-time checks <../../../validate_structural_long_time.py>`
check the BAOAB position identity, resonance cancellation, Gaussian sixth moment,
local-step formula, asymptotic cloning lower bound, Gaussian eighth moment,
truncation exponents, exact copying-flux balance, selection-versus-growth
coefficients and regeneration algebra with symbolic or exact arithmetic.
Decision checks additionally verify trap and radial-interval algebra, residual
convolutions, variance bands, finite-population flux and the escape exponent. Long-time
checks verify moment coefficients, occupation-law telescoping, finite-row
sampling, path-length identities, the phase-weight matrix calculation, exact
Keystone power constants, nonlinear error-floor recursions and the
active-cloning feedback threshold algebra, weighted reward bounds and
the eighth-moment localization exponent. The ordered-forest estimate additionally removes the weak-selection
condition from the full feedback bound: strict fitness increase supplies
the factorial path bound directly. The final checks verify the saturated
normalization derivatives, the frozen-Harris margin identity, optimized moment
truncation, Gaussian twenty-fourth moments, growing-window exponents and the
infinite-population-path geometric tail.
Finite-grid checks of competing hazards and transition intervals supplement
the analytic proofs; they do not replace them.
It proves the stated residence and attraction decimal bounds with exact rational
arithmetic. The three discovery-count enclosures use outward-rounded rational
interval arithmetic: dyadic endpoints of precision $2^{-128}$, integer-square-root
brackets, and positive exponential Taylor sums with a geometric remainder bound.
For $0\le x\le8$, after terms $0,\ldots,100$, that remainder is bounded by

$$
\sum_{k\ge101}\frac{x^k}{k!}
\le\frac{x^{101}}{101!}\frac1{1-x/102}.
$$

Negative arguments use reciprocals and monotonicity. The binomial weights are
propagated by their exact adjacent-weight recurrence, with outward rounding.
Every branch of the clipped acceptance map uses its monotonic endpoint bounds.
Thus the numerical certificates are finite rational inequalities following
explicit analytic enclosures; high-precision decimal illustrations in the output
are separately identified and are not proof inputs.

The evaluated spatial residence and positional-moment estimates do not silently
identify a specific full phase-space stationary law. The stationary population
result instead establishes existence, a quantitative invariance residual and
invariant subsequential mean-field limits. Convergence to an individual nonlinear
fixed law, where claimed, still needs a phase-specific attraction argument;
no contraction between different initial phases is required. Section 9 supplies
a quantitative active-cloning mean-field trajectory proof even for raw Rastrigin
reward. Section 10 separately proves full-law relaxation and uniform-time
mean-field approximation in its constant-fitness regime. Section 12 supplies
phase-local full-law convergence criteria with explicit integration budgets,
finite-resolution upper and lower decisions, and structural escape obstructions.
Section 13 proves quantitative stationary and joint occupation limits,
uniform-time block-reset implications, an evaluated active-cloning stationary
regime with an explicit simultaneous observation schedule, and full-sequence
phase-weight identification under the stated transition estimates. Section 14
derives population-independent Keystone pressure, exact signed donor and kinetic
balances, and complete-update quadratic consistency with error proportional to
$N^{-1/4}$. Its uniform-time conversion theorem retains every particle error
and requires the stated phase-attraction and coverage estimates. Section 15
derives the full-law attraction constant directly in a bounded-reward, finite
discrete-center regime with strictly positive selection parameters. It then
proves uniform-time trajectory approximation, stationary chaos, explicit
confidence budgets and agreement of both limit orders in that regime.
Section 16
removes the bounded-reward restriction using weighted total variation and
explicit output moments. The same unbounded potential may supply both raw
reward and force. Its computed feedback inequality supplies full-law relaxation,
uniform-time initialized mean-field approximation and stationary chaos. Section 17
derives selection-driven joint positional entropy confinement without the
finite-center condition, computes the zero-trap parameter criterion and tail
probabilities, and retains the exact full-law transported-reference production.
:::

:::{div} feynman-prose
The harmonic law estimates now have a completed nonresonant route at the
original step size. Active cloning with zero viscosity mixes to its exact
finite-swarm invariant law. With small positive count viscosity, the actual
population law mixes and the particle comparison has a vanishing error
floor; the raw quadratic reward and harmonic force may come from the same
potential. A sufficiently large fixed box also admits a complete marked
population proof with revival and a uniform-time particle comparison under
actual survival conditioning, including the same-potential raw reward.
Its current-alive observations have a particle
floor that vanishes with $N$. These results use their displayed positive
parameter intervals. The default viscosity $0.3$, box radius $2$, and
row-normalized extension remain separate proof obligations.
:::

:::{prf:remark} Results and remaining proof obligations
:label: rem-slc-completion

The infinite population-trajectory limit and the explicit growing-horizon
mean-field estimate are proved in {prf:ref}`thm-slcgt-growing-trajectory`
and {prf:ref}`cor-slcgt-infinite-population-path`, without a restoring-force
or stationary-attraction premise. Their product topology and finite growing
horizon must not be replaced by a uniform metric over all times.

The full long-time mean-field programme for reward-driven confinement without
a restoring force is not completed by the entropy floor. Section 18 proves
rootwise frozen-kernel mixing, uniform unbounded-reward fitness sensitivity,
and actual finite-population exit bounds. It also proves that combining its
specific unsigned feedback coefficient with its selection drift cannot close:
{prf:ref}`prop-slcfz-unsigned-empty` gives a coefficient strictly greater than
one. This failed sufficient estimate is not counted as a nonlinear attraction
result. The distinct-phase obstruction {prf:ref}`prop-slcfu-phase-obstruction`
also prevents asserting uniform-time approximation to every initial phase
when the finite-particle process mixes to one invariant law.

The signed calculation is now explicit in
{prf:ref}`prop-slcs-signed-bregman`: frozen-phase dissipation, the signed
feedback cross term and its nonnegative relative-entropy remainder are
retained separately. {prf:ref}`prop-slce-signed-fitness-refresh` derives
the positive frozen-fitness gain and the exact compensating refresh at a
stationary phase. {prf:ref}`thm-slcec-drift-transfer` and
{prf:ref}`cor-slcec-conditioning` transfer the signed finite-partition
entropy drift with an explicit vanishing particle error. The outstanding
step is a uniform sign and coercivity estimate for the population terms,
including the within-cell residual of {prf:ref}`prop-slcec-full-residual`;
these terms are not finite-population errors.

The completion criterion is a parameterized implication: declared basin,
transition-zone and tail profiles, together with the algorithm parameters and
initial-law data, determine explicit inequalities; when those inequalities
hold, the theorem supplies a rate, error floor, time horizon and probability.
The admissible parameter region is part of the result. Attraction for every
landscape and every parameter choice is not a target of this programme.

For the terminal-box problem, survival conditioning is treated directly in
{prf:ref}`thm-chaos-conditioned-quantitative-map`: the alive-fraction bound
is uniform over conditioned observation times, and the actual one-step
population-map defect is $\varepsilon_N+2\delta_N$. General canonical
box QSDs exist by {prf:ref}`thm-chaos-general-box-qsd-existence`, and
{prf:ref}`cor-chaos-conditioned-stationary-defect` identifies their
subsequential population limits as invariant laws of the mean-field map.
These results remove survival and moment-control obligations from that
stationary identification. They do not supply the separate nonlinear
dissipation needed to identify fixed-point phase support or attraction.

The distinction from full-law TV attraction is necessary even after
conditioning. {prf:ref}`prop-chaos-conditioned-kinetic-resonance` evaluates
the actual curvature--timestep resonance $h^2\kappa/4=1$, with active
cloning and all configured noises retained. Every QSD then has zero
velocities, but monokinetic nonzero initial data remain at TV distance one
from every QSD at every finite time. Their velocities nevertheless decay
at the explicit algebraic rate and obey the same finite- and
infinite-population update. This is a failed TV-attraction parameter
regime, not an obstruction to weak mean-field consistency. Strict
nonresonance conditions in the earlier smoothing theorems exclude it.

The axiom constants are quantitative structural descriptors, not binary labels
attached to a function. Their size and scale dependence determine how the
estimates deteriorate. A failed sufficient inequality leaves that conclusion
uncertified; a theorem proving escape, non-tightness or another obstruction
establishes a genuine failure. These two outcomes must be reported separately.

The chapter proves force-defect propagation, full-update selection confinement
with general coercive tail envelopes, regional passage and mixture communication,
radial kinetic drift as an optional specialization, and preservation of
the discharged Keystone estimate under regional localization, full-step defect
composition, landing/count/hitting/residence bounds, a complete Gaussian
minorization certificate, explicit Harris and entropy consequences, quantitative
empirical consistency, and contraction/phase transfer. Section 8 evaluates
basin residence, local positional attraction, signed discovery amplification,
nonconvex finite-particle TV mixing and stationary mean-field invariance. These are analytical
estimates for declared landscapes and the actual update. Sections 9–11 add
explicit weak-metric population continuity, active quadratic-growth reward
trajectory probabilities, finite-row chaos, nonconvex constant-fitness full-law
relaxation with uniform-time empirical approximation, and competing transitions.

The constants have three different statuses, which must be preserved in applications:

| Quantity | How its value is obtained | Remaining requirement |
|---|---|---|
| $q,s,V_c,p_J,p_O$ and ball volumes | Parameter register and Gaussian integrals | Declared amplitudes, radii and dimension |
| $M_A,\omega_A,b_A,J_A,L_R$ and tail moments | Landscape suprema or proved analytic upper bounds | Evaluate on the chosen regions; infinity is permitted |
| $\chi,B_{\rm key}$ | Explicit regional Keystone recipe | Its bounded entering geometry and reward increment bound |
| $\chi_0,\delta_{ab},E_{\rm sel},\mathcal B_\psi$ | Exact selected-source flux, regional adverse transfers and displayed Gaussian integrals | Donor coverage, fitness gaps and weighted failure defects; no force restoration required |
| $\Omega_R,\widehat\Psi_H$ | Regional pair increments, interface costs and charged Gaussian excursions | Their displayed modulus must close at the desired tolerance |
| $r_C,r_K,b_K$ | Copying lemma and radial drift formulas | $r_Cr_K<1$ or a sharper composed drift, with defects controlled |
| $\epsilon,\beta,\rho,\overline A_\mu$ | Full-kernel minorization and Harris formulas | Their force, drift, initial-moment and noise hypotheses |
| $A,B_*,\varepsilon_N$ | Expanded Chapter 9 dependency chain | Its input class, alive floor and moment hypotheses |
| $a_i,e_{N,T},u_{N,i,j}$ in phase approximation | Phase attraction, finite-horizon consistency and full-kernel residence | No contraction between different phases is required |
| $L,q,b_N$ in the optional contraction route | A proved matching coupled estimate | A sufficient specialization, not a necessary mean-field hypothesis |
| $\omega$ in the compact phase proposition | Section 9 supplies $C_{\rm mod}\delta^{1/4}$ or $C_8(H)\delta^{1/32}$ | All-alive, force-Lipschitz and stated reward/moment regime |
| $M_{8,n},v_n,P_n$ | Explicit copying, Gaussian moment, modulus and finite-cell formulas | Finite horizon; no attraction premise |
| $\varepsilon_2,a_N$ | Section 10 Gaussian minorization and iid finite-cell bound | Resonant bounded force perturbation, positive noises, zero fitness exponents |
| $\chi_{\rm geom},b_{\rm mix},A_\lambda,r_\lambda$ | Regional reward integrals, fitness bands, companion distances and explicit trap interval | Test $\lambda=0$ first; control coverage defects and parameter-dependent classes |
| $I_n^{[K]},M_1/K,R_n,q_*$ | Actual rooted density integrals and certified integration errors | A summable all-time envelope or uniform phase-local dissipation inequality |
| $L_f,U,v_n,\alpha_T$ | Regional masses, geometry, tail bounds and trajectory estimates | Accuracy, metric, horizon and confidence must be declared; overlapping intervals are inconclusive |
| $r_p,B_p,M_p$ | Regional copying flux and Gaussian $p$th moments | Uniform expected defects, or the evaluated resonant regime |
| $a_N+1/T,\Omega_K,V_*$ | One-step consistency and the bounded nonlinear path-length functional | Moment tightness for invariance; uniform phase dissipation for fixed-point support |
| $e_N(b;H)+a_H(b)+M_8/H+\delta_N(H)$ | Restart consistency, attraction and moment-localized coverage | Verified class-wide attraction and coverage for instantaneous joint limits |
| $\varepsilon_N^{\rm reg},n_N$ | Evaluated active-cloning Gaussian minorization | Resonant bounded force perturbation and positive kinetic noises |
| $K_A,\theta$ | Adjugate and determinant of the finite phase-flow matrix | Full-kernel relative transition bounds and vanishing phase concentration error |
| $k_{\rm key},p$ | Complete-cell Keystone coverage, with $p=5+4d$ and the displayed primitive-parameter formula | Bounded entering geometry, active diversity, positive regularization and regional reward increment bound |
| $\Gamma_\theta,\mathcal K_N$ | Exact donor destinations, barycenter shifts and kinetic covariance balance | Signed structural bounds must close for the observable being controlled |
| $a_t,d_t,G_j,R_j,C_G$ | Prepared force-pair profiles, actual collision speed cap, signed donor block masses and the complete-update variance polynomial in {prf:ref}`thm-slkd-structural-variance-threshold` | $R_j$ must lie within the phase's attainable variance range; the explicit $N^{-1}$ donor and $N^{-2}$ Keystone corrections vanish separately |
| $\mathcal U,\Psi,P_n$ | Exact two-Gaussian BAOAB density, force derivative modulus and the same Keystone prepared-source plan in {prf:ref}`thm-slc-keystone-full-state-tv` | Full-state TV decay follows at an $N$-independent rate only when the signed complete-update balance proves a decaying prepared discrepancy in the asserted phase |
| $U,C_Q,C_t,L_T$ | Frozen-velocity component bound, ordered-component sensitivity, and uniform-companion reward-normalizer calculation | These coefficients are $N$-independent; the signed residual still determines whether their prepared error decays |
| $C_4(D)M_4^{1/2}(G/N)^{1/4}$ | Rooted-forest bounded-test estimate and optimized fourth-moment truncation | A common finite/population output moment envelope |
| $\varepsilon_N$ in Section 14 | Explicit block length, localization scale, finite-cell errors and coverage budget | Verified population phase-attraction profile; all displayed particle terms then vanish |
| $\epsilon_2,L_R,q_2$ in Section 15 | Actual kinetic minorization and the complete rooted-component signed perturbation | Finite discrete-center profile, bounded reward, $2c_*<1$ and $2L_R+L_R^2<\epsilon_2$ |
| $\theta_{\max}$ | Explicit primitive-parameter selection interval | Positive floors and finite profiles; no unknown attraction constant |
| $a_N,V_N,\epsilon_N$ in Section 15 | Rooted consistency, uniform output moments and the proved $q_2$ | Every displayed particle contribution vanishes; all time-rate constants are independent of $N$ |
| $\beta_w,B_w,L_{\rm rem},q_w$ | Fourth-moment weighted normalization, root-averaged component rewards and kinetic mixing | Finite actual center profile, quadratic raw-reward growth and $2c_*<1$, $q_w<1$ |
| $C_{\rm eff},M_{24},\varepsilon_N$ in Section 16 | Actual Gaussian output moments, correlated-error Cauchy–Schwarz bound and explicit restart length | Particle error vanishes without an initial expected moment; initialized trajectory comparison uses its stated eighth-moment budget |
| $\chi,r_p,B_p,a_pE_p$ in Section 17 | Regional accepted-source flux, Gaussian moments and explicit Young coefficient | Common finite-count/population envelopes and controlled coverage defects; $\lambda=0$ is allowed |
| $\rho,C_H,D_n$ | Joint Gaussian density bound and product-reference entropy variational formula | A proved selection cost drift and finite geometric partition sums; no walker-independence or bounded force-center assumption |
| $\mathcal I_C,\mathcal I_K,\mathcal D_\pi$ | Exact reverse-channel entropy losses and transported-reference production | Full-law entropy decay requires dissipation to dominate the displayed production; stationary self-consistency alone does not erase it |
| $r,b,\epsilon,\rho_H$ in Section 18 | Rootwise accepted-source flux, one-row Gaussian minorization and proved Harris formula | Moment/core class tests and actual rootwise regional bounds; this concerns one frozen environment |
| $C_F,L_{\rm env}$ | Saturated logistic derivatives and the complete weighted rooted-component coupling | The computed unsigned assembly is proved insufficient for the selection-only drift; its failed test is not a convergence theorem |
| $u_N,A_g,b_g,T_N$ | Strict full-map class margins, bounded-test consistency, moment truncation and unrestricted moment growth | Explicit one-step and growing-window exit bounds; they do not erase infinite-time phase changes |
| $A_{24},b_{24},V_{N,T},T_N$ | Unrestricted complete-update moments, correlated-error modulus, finite cells and explicit recursion | Vanishing full trajectory error on the stated growing horizon; infinite-path convergence uses the declared product topology |
| $q_H,e_n$ in the entropy consequence | Complete-kernel entropy dissipation | Not inferred from a static LSI or from the Harris rate |

The phase theorem allows different initial populations to converge to different
fixed laws of the same population map. Global population contraction is neither
required nor compatible with distinct fixed phases. The optional contraction
theorems delimit a sufficient parameter region through their explicit premises.
The computed active-cloning finite-particle minorization generally deteriorates
with $N$; it cannot alone supply uniform-time approximation by taking
$N\to\infty$. The constant-fitness specialization has an independent
uniform-time proof. Section 12 proves a full-law active-cloning phase criterion through summable
successive-law increments and phase-local residual dissipation, with explicit
rooted-integral truncation and integration error budgets. Applying that criterion
requires proving its integral inequality uniformly on the asserted invariant
population class; finite tests alone do not discharge this requirement.
It supplies a rate in every region where that inequality closes, without
identifying different initial phases. The finite-horizon modulus and
positional residence estimates remain separate inputs to that calculation.

The closed active-cloning theorem in Section 15 discharges the attraction
and coverage inputs for its explicitly stated global parameter regime. It
retains copying, component collisions and nonconvex force perturbations, and
proves both uniform-time initialized trajectory approximation and commuting
stationary limits. Its bounded-reward condition concerns the configured raw
reward; it is not a theorem for an unbounded raw reward from the same
confining potential. Section 16 supplies that extension for quadratic-growth raw
reward through weighted normalization estimates; {prf:ref}`cor-slcw-same-potential`
verifies compatibility with one potential defining both channels. Beyond these
computed sufficient inequalities, the regional
phase-local criteria above remain conditional and distinct phases remain
permitted.

Section 17 proves a separate reward-driven entropy confinement route. Its
normalized joint positional entropy coefficient is independent of $N$, and
its regional selection inequality can hold with zero auxiliary trap. The
reference is constructed from landscape costs and regional volumes; it need
not be an invariant law. The finite entropy floor proves tightness and
quantified population-fraction tails. Full-law entropy convergence to a
stationary phase requires controlling the exact reference-production term
in {prf:ref}`prop-slce-exact-balance`; the confinement proof does not
silently set that term to zero.

The organizing data are the declared basins, transition/mixture zones, exterior
shells and their analytic profiles. Named landscapes above are substitution checks,
not hypotheses of the structural theorems. A new application must evaluate these
regional profiles, control their
excursions and copying terms, and verify whichever composition, minorization or
phase-attraction/residence or optional population-coupling criterion it uses. Infinite profiles and zero communication
bounds remain recorded outcomes. Strong convexity is one convenient special
case; general global convergence is not asserted when the structural inequalities
do not close. The estimates supply a resolution-dependent structural description,
not a claim that finitely many constants uniquely determine the reward function.
:::


(sec-slc-regional-error-rates)=
## 19. Regional rates with explicit dimension and tail budgets

:::{prf:definition} Regional discrepancy certificate
:label: def-slc-regional-error-certificate

Fix an actual population-normalized coupling or law discrepancy. For a declared
partition into well cores, transition regions, slow-mixing regions and exterior
shells, let $e_i(n)\ge0$ be its regional contributions. The partition is an
analysis parameter and does not modify the kernel. A regional certificate is a
nonnegative matrix $A_d$, a nonnegative defect vector $b_d$, positive weights
$w_i$, and proved inequalities for the complete actual update over $m$ steps:

$$
e_j(n+1)\le\sum_i(A_d)_{ji}e_i(n)+(b_d)_j.
$$

The inequalities include conditioning, interfaces, accepted copying, complete
collision components and Gaussian excursions whenever these occur in the
chosen kernel. An empirical region-transition matrix is not such a certificate:
it measures mass transfer, which need not bound discrepancy transfer. Define

$$
\rho_d=\max_i\frac{\sum_jw_j(A_d)_{ji}}{w_i},\qquad
B_d=\sum_jw_j(b_d)_j,\qquad E_n=\sum_iw_ie_i(n).
$$

Every regional cost uses a probability coupling integral or a per-walker
average. Thus no factor $N$ is introduced by these definitions.
:::

:::{prf:theorem} Regional rate and full tail remainder
:label: thm-slc-regional-error-rate

Under the preceding complete regional inequalities, if $\rho_d<1$, then

$$
E_n\le\rho_d^nE_0+B_d\frac{1-\rho_d^n}{1-\rho_d},\qquad
\lambda_d=-\frac{\log\rho_d}{mh},\qquad
E_\infty^{\rm bound}=\frac{B_d}{1-\rho_d}.
$$

For $m_w=\min_jw_j>0$, $M_w=\max_jw_j$ and
$\overline E_n=\sum_j E_{j,n}$, the same certificate gives

$$
\overline E_n\le\frac{M_w}{m_w}\rho_d^n\overline E_0+
\frac{B_d}{m_w}\frac{1-\rho_d^n}{1-\rho_d}.
$$

Positive weights may be chosen by a numerical search and then checked by the
displayed column inequalities. Such a search neither proves the regional
inputs nor removes this metric-conversion factor. Uniformity in $N$ includes
uniform control of the chosen weights and their ratio.

If a proved uniform moment is $\mu|x|^p\le M_{p,d}$ and $0\le r<p$, an
exterior cutoff $R>0$ admits the full unbounded-tail bounds

$$
\mu\{|x|>R\}\le\min\{1,M_{p,d}/R^p\},\qquad
\int_{|x|>R}|x|^r\,d\mu\le M_{p,d}/R^{p-r}.
$$

These remainders may enter $b_d$ through the proved regional inequalities.
They do not justify inserting sampled moments in place of $M_{p,d}$.
If all certificate inputs are independent of $N$, the rate and remainder
are independent of $N$. A nonzero $B_d$ proves decay to a remainder; it
alone does not prove convergence to a stationary law.
:::

:::{prf:proof}
Multiply each regional inequality by $w_j$, sum, and interchange the finite
nonnegative sums. The definition of $\rho_d$ gives $E_{n+1}\le\rho_dE_n+B_d$.
Induction sums the finite geometric series. The inequalities
$m_w\overline E_n\le E_n\le M_w\overline E_n$ give the unweighted bound.
For the first tail bound use
$\mathbf1_{|x|>R}\le |x|^p/R^p$. For the second use
$|x|^r\mathbf1_{|x|>R}\le |x|^p/R^{p-r}$. Integrating proves both bounds on
the original unbounded state space. No independence between walkers is used.
:::

:::{prf:lemma} Bounded nonquadratic force perturbation of the native sector rate
:label: lem-slc-native-nonquadratic-sector

Consider only the isotropic native BAOAB kinetic update with fixed radial cap,
common Gaussian innovations, $c=h/2$, $a=e^{-\gamma h}$, and force
$F(x)=-\omega x+e(x)$, $\omega>0$. Suppose the harmonic complete-update
sector inequality is certified for

$$
G_\beta(x,v)=\omega|x|^2+2\beta\sqrt\omega\langle x,v\rangle+|v|^2,
\qquad |\beta|<1,
$$

with squared contraction coefficient $q_H=1-\delta_H<1$, uniformly for every
symmetric radial-cap secant $0\preceq D\preceq I$. Suppose the residual force
differences at the two actual kick-query pairs obey $|\Delta e_1|\le D_1$,
$|\Delta e_2|\le D_2$. Define

$$
B_e^2=(1+|\beta|)\left[\omega c^4(1+a)^2D_1^2+
\{c|a-\omega c^2(1+a)|D_1+cD_2\}^2\right].
$$

For every $\theta>0$,

$$
G_\beta(\Delta z^+)\le (1+\theta)q_HG_\beta(\Delta z)
 +(1+\theta^{-1})B_e^2.
$$

For $B_e>0$, choosing $\theta=\delta_H/[2(1-\delta_H)]$ gives
$\rho=1-\delta_H/2<1$ and an explicit additive defect. If $B_e=0$, retain
the harmonic coefficient $q_H$ directly. Integrating a representative coupling
and minimizing the output transport cost preserves the upper bound. Iteration
along this common-noise coupling gives a population-normalized error envelope
from its initial coupling cost. Cloning, viscosity, killing and environment
feedback require their separate complete-update regional estimates.
:::

:::{prf:proof}
Expand the two kicks and drifts before the terminal cap. Relative to the
harmonic difference map, their residual contributions are exactly

$$
r_x=c^2(1+a)\Delta e_1,\qquad
r_v=c\{a-\omega c^2(1+a)\}\Delta e_1+c\Delta e_2.
$$

The OU noise cancels in the shared-noise difference. The final position noise
also cancels. The terminal radial cap has a symmetric secant $D$ with
$0\preceq D\preceq I$, by {prf:ref}`lem-rcap-sector`, so the complete
difference is the harmonic sector map plus $(r_x,Dr_v)$. In the coordinates
$(\sqrt\omega x,v)$, the metric matrix has maximal eigenvalue $1+|\beta|$.
Thus the norm of this residual is at most $B_e$. The harmonic sector
certificate bounds the other summand by $\sqrt{q_H}\|\Delta z\|_{G_\beta}$.
The squared triangle inequality followed by
$(u+v)^2\le(1+\theta)u^2+(1+\theta^{-1})v^2$ proves the assertion.
The cap secant may depend on the actual force and noise; the certificate is
uniform over every admissible $D$, so no independence is assumed. Integrate
against the chosen coupling and use that the optimal output cost is no larger.
:::

:::{prf:corollary} Rastrigin profiles and a conditional within-well rate
:label: cor-slc-rastrigin-regional-sector

For standard $d$-dimensional Rastrigin,

$$
F(x)=-2x-20\pi(\sin(2\pi x_j))_{j=1}^d,
\qquad M_d=20\pi\sqrt d,\qquad L_e=40\pi^2.
$$

Hence the preceding perturbation lemma with $\omega=2$ admits
$D_1=D_2=2M_d$ globally, or $D_i\le\min\{2M_d,L_er_i\}$ when the actual
kick-query separation is bounded by $r_i$. The global defect is proportional
to $d$ and independent of $N$. The raw reward satisfies
$0\le U(x)\le|x|^2+20d$, and for $0<k<2$ the radial profile satisfies

$$
b_{\mathbb R^d}(k)\le\frac{(20\pi)^2d}{4(2-k)}.
$$

For a declared box $|x_j-k_j|\le r\le1/2$ around an integer well, use the analytic curvature interval
$[m_r,M_r]=[2+40\pi^2\cos(2\pi r),2+40\pi^2]$.
The harmonic curvature $\omega_w=(m_r+M_r)/2>0$ minimizes the
regional residual Lipschitz bound, giving
$\ell_r=(M_r-m_r)/2=20\pi^2[1-\cos(2\pi r)]$. Suppose the harmonic sector certificate
holds at $\omega_w$ for the declared metric
$G_{\alpha,\beta}(x,v)=\alpha\omega_w|x|^2+2\beta\sqrt{\omega_w}\langle x,v\rangle+|v|^2$,
with $\alpha>\beta^2$, and both actual kick-query pairs are in this box. Put

$$
t_x=[\omega_w(\alpha-\beta^2)]^{-1/2},\quad
 t_v=(1-\beta^2/\alpha)^{-1/2},\quad d_1=\ell_rt_x,
$$
$$
t_2=|1-\omega_wc^2(1+a)|t_x+c(1+a)t_v+c^2(1+a)d_1,
\qquad d_2=\ell_rt_2,
$$
$$
K_r^2=\lambda_+\left[\omega_wc^4(1+a)^2d_1^2+
\{c|a-\omega_wc^2(1+a)|d_1+cd_2\}^2\right].
$$

Here $\lambda_+=[\alpha+1+\sqrt{(\alpha-1)^2+4\beta^2}]/2$ is the
maximum eigenvalue in scaled coordinates. Then the conditional within-well squared contraction coefficient is
$\rho_r=(\sqrt{q_H}+K_r)^2$. It supplies a positive within-well rate only
when $\rho_r<1$. Slow-zone and escape contributions must enter a complete
regional certificate before this becomes a global statement.
:::

:::{prf:proof}
The sine difference is bounded both by $2\sqrt d$ and by $2\pi|x-y|$;
multiplying by $20\pi$ proves the global residual bounds. Completing the
square in $-(2-k)|x|^2+20\pi\sqrt d|x|$ proves the radial profile.
The potential bound follows from $0\le1-\cos t\le2$ coordinatewise.
Within the declared box the force Jacobian is minus the potential Hessian,
whose eigenvalues belong to $[m_r,M_r]$. The residual Jacobian relative to
$\omega_w$ therefore has operator norm at most
$\max\{|\omega_w-m_r|,|\omega_w-M_r|\}=\ell_r$.
For every alternative scalar harmonic curvature this maximum is at least
$(M_r-m_r)/2$, proving the midpoint minimization. The box is convex, so its force differences obey that
Lipschitz bound. Completing the square in each variable of the positive metric yields
$|\Delta x|\le t_x\|\Delta z\|_G$ and
$|\Delta v|\le t_v\|\Delta z\|_G$. The exact pre-second-kick position
difference formula yields $|\Delta x_2|\le t_2\|\Delta z\|_G$.
Consequently both residual differences are bounded by $d_i\|\Delta z\|_G$.
The triangle estimate in the preceding proof, before Young's inequality,
gives $(\sqrt{q_H}+K_r)\|\Delta z\|_G$ and proves the result.
:::

:::{prf:corollary} Exact accepted-jitter variance and refined root-core geometry
:label: cor-slc-exact-jitter-regional-refinement

Retain every source, acceptance, collision, viscosity and noise hypothesis of
{prf:ref}`thm-kur-evaluated-regional-bound`. In its notation let
$a=e^{-\omega^2\sigma_J^2/2}$, $b_J=a^4$, $c_*=\cos(\omega r_*)$, and define

$$
\Psi(z)=\ell^2\sigma_J^2-2\ell A\omega\sigma_J^2az
+A^2\left[\frac{1+b_J}{2}-a^2+(a^2-b_J)z^2\right],\qquad
K_J^{\rm exact}=\max\{\Psi(c_*),\Psi(1)\}.
$$

Then $0\le K_J^{\rm exact}\le K_J$, and the same theorem holds with
$K_J^{\rm exact}$ replacing $K_J$ throughout its accepted-jitter terms.
In particular, with its original $\rho_J,\lambda,\varepsilon,D_J$,

$$
C_{\rm reg}^{\rm exact}
=C_{\rm reg}-(1+\varepsilon)d(K_J-K_J^{\rm exact}).
$$

The full Gaussian jitter is retained, and $K_J^{\rm exact}=0$ when
$\sigma_J=0$. This refinement is independent of the population size.

For integer root centers $|k|\le K$, set
$m=2+20\sqrt2\pi^2$ and $\delta_0=2K/m<1/8$. Every finite sequence

$$
\delta_{j+1}=\min\left\{\delta_j,
\frac{2K}{2+40\pi^2\cos(2\pi\delta_j)}\right\}
$$

is a valid nonincreasing upper bound on the stable-root displacement
$|z_k-k|$. Thus the declared source core of radius $R$ about $z_k$ lies
within $R+\delta_j$ of $k$. Whenever $R+\delta_j<1/8$, all regional constants
may be recomputed at this smaller enlarged radius. Its derivative supremum
$\rho_J$ and multiplier $\lambda=(1+\rho_J^2)/2$ cannot increase; its noise
floor is recomputed with the corresponding Young parameter. These remain
one-step source-variance bounds with the theorem's full phase accounting.

The same argument permits general isotropic native parameters
$h>0$, $\gamma,\sigma_J,q,s,\nu\ge0$, $V_{\max}>0$ and
$\alpha_{\rm col}\in[0,1]$ with either first viscous normalization,
provided its actual first-kick matrix is stochastic, $0\le(h/2)\nu\le1$,
$\ell\ge0$, $r_*<1/8$, and

$$
0<\rho_J=\max\{|\ell-A\omega a|,
|\ell-A\omega a\cos(\omega r_*)|\}<1.
$$

Use $b=(h/2)(1+e^{-\gamma h})$, $\eta=(h/2)b$,
$\ell=1-2\eta$, $A=20\pi\eta$, $\omega=2\pi$,
$V_c=(1+2|\alpha_{\rm col}|)V_{\max}$ and
$\tau^2=(h/2)^2q^2+s^2$ in the same formulas. Here $q,s$ are the actual
OU and positional standard deviations. The complete source law and
pre-jitter acceptance conditioning remain required.
If every entering frozen velocity, including each revived recipient's
frozen collision input, has a proved bound
$V_*\le V_{\max}$, the same floor permits
$V_c=(1+2|\alpha_{\rm col}|)V_*$ while the algorithm retains its configured
cap $V_{\max}$. For zero entering velocities this removes the velocity
remainder exactly; all three Gaussian noises retain their original laws.
:::

:::{prf:proof}
Condition on the actual frozen source, acceptance pattern and component
rotations as in the cited theorem. For one accepted coordinate put
$g(y)=\ell y-A\sin(\omega y)$ and $Y=u+\sigma_J Z$, $Z\sim N(0,1)$.
The Gaussian characteristic function gives
$\mathbb E\sin(\omega Y)=a\sin(\omega u)$ and
$\mathbb E\sin^2(\omega Y)=[1-b_J\cos(2\omega u)]/2$.
Integration by parts against the Gaussian density gives
$\operatorname{Cov}(Y,\sin(\omega Y))=\omega\sigma_J^2a\cos(\omega u)$;
the boundary terms vanish by Gaussian decay. Expanding the variance and
using $\cos(2\theta)=2\cos^2\theta-1$ yields
$\operatorname{Var}(g(Y))=\Psi(\cos(\omega u))$.
Since $a^2-b_J=a^2(1-a^2)\ge0$, $\Psi$ is convex on $[c_*,1]$ and its
maximum occurs at an endpoint. This proves the uniform conditional
accepted-variance bound. The original $K_J$ bounds the same expression by
discarding its nonpositive squared-mean term and using the original cosine
bounds, so $K_J^{\rm exact}\le K_J$. Substitution in (KUL.9), (KUR.5) and
(KUR.6) proves the stated floor with every original correlation preserved.
In particular the jitter-dependent viscous velocity is still bounded
pathwise before Young's inequality; no graph-jitter independence is used.

For root geometry, the original curvature bound gives
$|z_k-k|\le2|k|/m\le\delta_0$. If $|z_k-k|\le\delta_j<1/8$, the segment
between the root and its integer center has curvature at least
$2+40\pi^2\cos(2\pi\delta_j)$. Since $U'(k)=2k$ and $U'(z_k)=0$, the
mean-value theorem gives the second bound in the displayed minimum.
Induction proves every enclosure. The mean-map derivative is evaluated
over a nested interval, so its absolute supremum cannot increase. All
remaining estimates of the original theorem apply at the recomputed
radius, proving the claim without restricting any output noise.
For the parameterized extension, the displayed $\rho_J$ is the exact
absolute derivative supremum of the mean map on the declared interval.
The pairwise variance identity gives its squared variance multiplier
whether or not that derivative is positive. The actual stochastic matrix
still gives $W_N(W_XV^C)\le V_c^2$ pathwise, and the fresh positional
Gaussian has variance $\tau^2$ per coordinate. The two Young inequalities
of (KUR.5)--(KUR.6) therefore prove the same bound with the displayed
general parameters. This verifies each step of the extension.
For the state-specific velocity bound, the actual collision uses the
frozen entering velocities of all component members, including revived
recipients. Their component mean has norm at most $V_*$, and each
deviation has norm at most $2V_*$. A standalone copied velocity also
retains this bound. Orthogonality of the actual Haar rotation yields the stated
collision bound. The subsequent stochastic velocity average obeys the same
pathwise bound. Substitution in the existing Young estimate proves the
refinement, including $V_*=0$.
:::


(sec-slc-global-selected-moments)=
## 20. Global normalized moments with positive selection

:::{prf:theorem} Dimension-dependent unbounded moments for native weak selection
:label: thm-slc-global-selected-moments

Consider the conservative native Euclidean Gas with death disabled, no elite
or external source injection, and independent one-donor Gaussian proposals
with current self-exclusion. All current and retained historical source frames
are alive. Their physical velocities satisfy the declared cap $V$; the
squashed phase-space radii, logistic fitness parameters, gate parameters, and
force envelope below are the actual configured values. The first viscous
matrix uses a count or row normalization and satisfies $0\le h\nu/2\le1$.
Use the standard affine BAOAB first force kick, with no Boris or additional
force-update program. The OU and final-position innovations are independent
isotropic Gaussians of constant amplitudes $q,s$, and the independent
isotropic recipient jitter has amplitude at most $\sigma_J$.
For a nonempty historical donor window, native component collision must be
disabled by setting restitution to `None`: accepted historical velocities
are copied literally, and the following budget uses effective $\alpha=0$.
Setting restitution to `Some(0)` does not enable historical donors. For the
current-frame collision branch, restitution lies in $[0,1]$.
Physical positions and every noise law retain their full unbounded support.
Then (SLM.1)--(SLM.6) below hold. When the stated multiplier is below one,
they give a rate and floor for normalized moments and tails independent of
$N$. Finite historical windows use the stated block rate.

*Proof and explicit constants.*

Let $N\ge2$. Write $M_p(S)=N^{-1}\sum_i|x_i|^p$, $p\ge1$, and let
$\mu_i$ be the actual zero-jitter source positions in the frozen literal-copy
plan. The independent one-donor companion kernel uses the actual squashed
phase-space features. With feature radii $R_x,R_v$, velocity weight
$\lambda_{\rm alg}\ge0$, and bandwidth $\epsilon_C>0$, its squared diameter
is bounded by

$$
D^2=4(R_x^2+\lambda_{\rm alg}R_v^2),\qquad
\kappa=\exp[-D^2/(2\epsilon_C^2)]>0.
$$

This is a bound on algorithmic features; physical positions remain unbounded.
Actual Gaussian weights lie in $[\kappa,1]$. For current-frame companions,
$p_{ij}\le[(N-1)\kappa]^{-1}$ for every $i\ne j$.

For each actual positive logistic channel with amplitude $A_c\ge0$, floor
$\eta_c>0$ and exponent $e_c\ge0$, define

$$
F_-=\prod_c\eta_c^{e_c},\qquad
F_+=\prod_c(A_c+\eta_c)^{e_c},\qquad
a_* =\min\left\{1,{F_+-F_-\over s_c(F_-+\epsilon_c)}\right\}.
$$

Here $s_c>0$ and $\epsilon_c\ge0$ are the actual competitive-gate parameters.
The bound holds for every complete sampled measurement and nonlinear global
normalization; no fitness of an averaged measurement is substituted. Condition on the actual sampled measurement first; the cloning donor draw
is independent of that measurement given the frozen state. Let $a_{ij}$
be its resulting acceptance probability, or its measurement expectation
after averaging. Simultaneous
frozen-source copying gives the exact conditional identity

$$
\mathbb E[M_p(\mu)\mid S]
=M_p(S)+{1\over N}\sum_{i\ne j}p_{ij}a_{ij}
(|x_j|^p-|x_i|^p).
$$

Discard only the negative source-loss terms. Every incoming accepted-source
load obeys
$\sum_{i\ne j}p_{ij}a_{ij}\le a_*/\kappa$. Therefore

$$
\mathbb E[M_p(\mu)\mid S]\le(1+a_*/\kappa)M_p(S).
\tag{SLM.1}
$$

This uses no labeling in its observable or bound; permutations preserve all
finite sums. It also uses no physical support bound and introduces no factor
growing with $N$. Revived dead recipients are excluded from this conservative
hypothesis. Their source gain cannot silently be controlled by the same
current alive $1/N$ moment when the alive fraction can vanish.

**Native kinetic step and unbounded force.**

Suppose the actual force is $F(x)=-\omega x+r(x)$, with
$|r(x)|\le B\sqrt d$ globally. The harmonic benchmark has $(\omega,B)=(1,0)$;
the actual Rastrigin benchmark has $(\omega,B)=(2,20\pi)$. No sampled force
maximum is used. For BAOAB define

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
\eta=tb,\quad a=|1-\eta\omega|,\quad
\tau=\sqrt{t^2q^2+s^2}.
$$

Here $q$ is the actual OU standard deviation and $s$ the actual final position
standard deviation. Every frozen entering collision input, including a revived
slot when used in a different killed application, and every copied historical
velocity must obey its stated bound $V$. Native restitution obeys
$\alpha\in[0,1]$; the component mean and relative-velocity triangle bound give
$|v_i^C|\le(1+2\alpha)V$. The actual first viscous matrix is stochastic when
$0\le t\nu\le1$, in both declared count and row normalizations. It may depend
on the full recipient jitter. Its pathwise velocity norm bound remains valid
without assuming that graph is independent of the jitter.

Let $g_{d,p}$ be a certified upper bound on $\|Z\|_{L^p}$ for a standard
$d$-dimensional Gaussian. The implementation uses the next even moment:
$k=\lceil p/2\rceil$,
$g_{d,p}=[\prod_{j=0}^{k-1}(d+2j)]^{1/(2k)}$, which is exact at even $p$ and
valid otherwise by monotonicity of probability-normalized $L^p$ norms.
Condition on the complete frozen source/copy plan and entering inputs BEFORE
recipient jitter and its dependent viscous graph. Component rotations may be
held fixed because their independent draws depend only on the frozen copy
components; alternatively average them jointly. Minkowski over the uniform row
law and the original full-support jitter, Haar, OU and position laws gives

$$
\left(\mathbb E[M_p(X^+)\mid\mu,\text{pre-jitter frozen inputs}]\right)^{1/p}
\le aM_p(\mu)^{1/p}+C_p,
\qquad
C_p=a\sigma_Jg_{d,p}+b(1+2\alpha)V+\eta B\sqrt d+\tau g_{d,p}.
\tag{SLM.2}
$$

Accepted rows use jitter $\sigma_J$ and unaccepted rows use zero; replacing
their amplitudes by the common upper amplitude proves the inequality. The
unrestricted OU and final-position noises combine into the stated isotropic
Gaussian. B2 and the final velocity cap do not alter the completed position.

For $p>1$ and $0<a<1$, let
$\lambda_p=(1+a^p)/2$,
$\varepsilon=(\lambda_p/a^p)^{1/(p-1)}-1>0$, and
$B_p=(1+\varepsilon^{-1})^{p-1}C_p^p$.
Young's inequality proves

$$
\mathbb E M_p(X^+)\le\lambda_p\mathbb E M_p(\mu)+B_p.
\tag{SLM.3}
$$

For $p=1$ use $(\lambda_1,B_1)=(a,C_1)$; for $a=0$ the source term vanishes
exactly. Combining (SLM.1) and (SLM.3),

$$
P M_p\le r_pM_p+B_p,\qquad
r_p=\lambda_p(1+a_*/\kappa).
\tag{SLM.4}
$$

When $r_p<1$, iteration yields
$\mathbb E M_p(S_n)\le r_p^n\mathbb E M_p(S_0)+
B_p(1-r_p^n)/(1-r_p)$.
This is a global unbounded-space moment estimate with an $N$-independent rate
and floor, even when the landscape has slow zones. It does not imply a pure
global contraction of distinct phase laws. Any conservative invariant law with
finite moment inherits the floor; this drift alone does not prove its existence
or unique attraction.

**Finite historical windows.**

Use the native literal historical velocity-copy branch with component
collision disabled, as required in the statement; its velocity budget is
$V$ and its effective $\alpha$ is zero.
With $j\ge1$ past all-alive frames, the pool has $N(j+1)$ rows and at most one
excluded current self. Its normalization gives

$$
\mathbb E M_p(\mu)\le M_p(S_n)
 +{a_*\over\kappa}\,{N(j+1)\over N(j+1)-1}
 {1\over j+1}\sum_{l=0}^j M_p(S_{n-l}).
\tag{SLM.5}
$$

The ratio is at most $4/3$ for all $N\ge2,j\ge1$. Current-only warmup has
the sharper coefficient one. After expectation, (SLM.3) gives
$m_{n+1}\le\lambda_pm_n+(4\lambda_pa_*/(3\kappa))
\max_{0\le l\le H}m_{n-l}+B_p$, where $m_n=\mathbb E M_p(S_n)$.
The maximum is over expected moments, not a sampled maximum inside an
expectation. If
$R_p=\lambda_p(1+4a_*/(3\kappa))<1$ and $D_p=B_p/(1-R_p)$,
the positive excesses obey
$[m_{n+1}-D_p]_+\le R_p\max_{0\le l\le H}[m_{n-l}-D_p]_+$.
After each $H+1$ updates every old excess has left the memory window; induction
gives the block factor $R_p^{\lfloor n/(H+1)\rfloor}$ against the initial
window's maximum excess. This proves finite-history normalized moment and tail
control. It requires all source frames alive and every used historical velocity
bounded; it is not applied to a disappearing survivor pool.

**Tail and escape bounds.**

For $0\le r<p$ and a proved moment bound $M$ for the precise law of interest,

$$
\mathbb E{1\over N}\#\{i:|x_i|>R\}\le M/R^p,\qquad
\mathbb E{1\over N}\sum_i|x_i|^r1_{|x_i|>R}\le M/R^{p-r}.
\tag{SLM.6}
$$

Use the time-dependent bound from (SLM.4) or its historical analogue; no compact
support replaces Gaussian tails. The Chapter 6 coupled-tail lemma then charges
cross-coupled costs using both laws' own moment bounds. A regional discrepancy
operator still needs proved transfer coefficients and the full slow-zone flux;
moment control does not manufacture those coefficients.

For the explicit primitive probe $R_x=R_v=2$, $\lambda_{\rm alg}=1$,
$\epsilon_C=3$, $A_r=A_s=2$, $\eta_r=\eta_s=0.1$,
$e_r=e_s=10^{-5}$, $s_c=1$, $\epsilon_c=10^{-6}$, one has
$\kappa=e^{-32/18}\simeq0.1690133$ and
$a_*\le\exp(2\cdot10^{-5}\log21)-1\simeq6.089\cdot10^{-5}$.
At $h=.04,\gamma=1$ both harmonic and Rastrigin p4/p8 interfaces absorb the
current and finite-history source gains. Strong default exponents generally
fail this sufficient inequality and are not certified by these formulas.

:::


:::{prf:corollary} Accepted-jitter and collision-energy tail budgets
:label: cor-slc-sharp-selected-tail-budget

Under the hypotheses of {prf:ref}`thm-slc-global-selected-moments`, suppose
also that the actual first viscous matrix is doubly stochastic, as it is
with no viscosity or with the count normalization and symmetric weights.
The conservative current-frame component collision, or the native literal
historical-copy branch with restitution `None`, then has the refined
budgets and root-moment closure below. General row normalization requires
its original velocity budget unless column control is separately proved.

*Proof and explicit constants.*

For current-frame component collision, every frozen entering velocity has norm
at most $V$. Component momentum is preserved and relative energy is multiplied
by $\alpha^2\le1$, so the full-slot probability-normalized $L^2$ norm is at most
$V$. The pathwise row bound is $(1+2\alpha)V$. Consequently, probability-space
monotonicity for $1\le p\le2$, and interpolation for $p>2$, give

$$
\|V_{\rm collision}\|_{L^p({\rm row})}
\le V(1+2\alpha)^{(1-2/p)_+}=:V_p.
\tag{SLM.8}
$$

The same bound survives the first doubly stochastic matrix by Jensen and its
column sums. For the native historical `restitution=None` branch the literal
copied source velocities are individually capped by hypothesis, so $V_p=V$
with effective $\alpha=0$. No component operation is invented for that branch.

In the all-alive conservative kernel, only accepted recipients receive fresh
position jitter. If $I_i$ is its accepted indicator and $Z_i$ is its independent
standard Gaussian, condition on frozen measurement and donor candidates before the acceptance
uniform is drawn. For that conditional law,
$\mathbb E[\|I_i\sigma_J Z_i\|^p]\le a_*\sigma_J^p g_{d,p}^p$.
Averaging over the actual full nonlinear measurement leaves the same inequality.
After probability normalization over rows, the jitter contribution to the
Minkowski budget is therefore $\sigma_J g_{d,p}a_*^{1/p}$, independent of $N$.
Mandatory revival and elite injection are excluded; their always-jittered or
external recipients do not satisfy this accepted-gate argument.

Thus replace only the additive budget in (SLM.2) by

$$
\widetilde C_p
 =a\sigma_Jg_{d,p}a_*^{1/p}
  +bV(1+2\alpha)^{(1-2/p)_+}
  +\eta B\sqrt d+\tau g_{d,p}.
\tag{SLM.9}
$$

All native parameters retain their original values. The same Young
$\lambda_p,\varepsilon_p$ and source coefficient give
$\widetilde B_p=(1+\varepsilon_p^{-1})^{p-1}\widetilde C_p^p$,
$\widetilde R_p=R_p$, and floor
$\widetilde B_p/(1-R_p)\le B_p/(1-R_p)$ when $R_p<1$.
The moment rate is unchanged, while the floor is reduced. This is a complete
conservative p-moment estimate, not a population-law contraction.

**Root-moment closure and parameter interval.**

Apply Minkowski on the joint space of the complete native randomness and a
uniform row before taking any nonlinear power of expectation. With
$S=1+a_*/\kappa$ for current sources, (SLM.1) and (SLM.9) imply

$$
u_n=(\mathbb E M_p(S_n))^{1/p},\qquad
u_{n+1}\le r_p^{\rm root}u_n+\widetilde C_p,\qquad
r_p^{\rm root}=aS^{1/p}.
\tag{SLM.10}
$$

Here the symbol is $u_n$ (root moment), not the unrooted moment $m_n$.
Whenever $a^pS<1$, the explicit bounds are
$u_n\le (r_p^{\rm root})^nu_0+\widetilde C_p(1-(r_p^{\rm root})^n)/(1-r_p^{\rm root})$ and
$\limsup\mathbb E M_p\le[\widetilde C_p/(1-r_p^{\rm root})]^p$.
The closure criterion is weaker than $\lambda_pS<1$ and the floor is no larger
than the Young floor: the fixed point of the root recurrence is a subsolution
of the Young moment recurrence. One must not replace this by an unproved
linear unrooted-moment recurrence or claim rate $(r_p^{\rm root})^p$ for its excess above
a positive floor.

For finite all-alive historical windows with native collision disabled as
stated above, use $S=1+4a_*/(3\kappa)$ and the maximum of the recent expected
root moments. Subtracting $\widetilde C_p/(1-r_p^{\rm root})$ from this maximum yields a
positive excess contracting by $r_p^{\rm root}$ after each $H+1$ steps. The initial
window maximum remains in the explicit bound. All full-support Gaussian
noise, native gate probabilities, true force centers, and probability
normalization are retained.

For $0<a<1$, an explicit sufficient parameter interval follows without a
numerical optimization. Define
$\eta_F=\sum_c e_c\log[(A_c+\eta_c)/\eta_c]$.
The native nonnegative gate regularizer gives $a_*\le(e^{\eta_F}-1)/s_c$.
Consequently current-frame root closure follows from

$$
\eta_F<\log\{1+s_c\kappa(a^{-p}-1)\}.
\tag{SLM.11}
$$

For the supported finite-history `None` branch replace $s_c\kappa$ in the right
side by $3s_c\kappa/4$. With the actual native equal channel exponents $\zeta$,
amplitude two, floor $0.1$ and the declared saturation, divide the right side by $2\log21$ to get an
explicit sufficient upper bound on $\zeta$. This inequality keeps the
nonnegative native epsilon and gate clipping; dropping their improvements
only makes the bound conservative. It is a global analytic interval, not an
empirical estimate of a worst-case fitness configuration.


The sufficient interval follows by substituting the gate envelope into
$a^pS<1$. It preserves the actual positive gate saturation. The rate is
for the stated expected root moment; a population-law mixing rate retains
its separate transfer hypotheses. $\square$
:::
