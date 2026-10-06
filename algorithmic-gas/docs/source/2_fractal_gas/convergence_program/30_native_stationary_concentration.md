# Population-uniform marked stationary variance in the native contraction regime

(sec-native-concentration-register)=
## 1. Complete record and the additional derived viscosity interval

:::{prf:definition} Finite-population concentration register
:label: def-native-concentration-register

Retain every field of {prf:ref}`def-native-phase-register`, the complete
canonical real-coordinate count kernel, its pre-jitter source/gate/Haar
record, and its actual continuous innovations. Use all coefficients of
{prf:ref}`def-native-phase-matrix` and an actual weight $\omega$ satisfying
(PC.24), with proved population contraction coefficient $r<1$.
Retain $0<V\le\min\{V_0,1\}$, the domain of the proved cap budget
(PC.20). The cap $V$, landscape, boundary, noises and other consumed parameters
are fixed independently of $N$. The complete raw output retains every
coordinate and terminal mark, including an all-dead output; denote its
kernel by $P_N^{\rm raw}$. Its surviving restriction and QSD are $Q_N$
and $\nu_N$, with $\nu_NQ_N=\alpha_N\nu_N$. This raw kernel is distinct
from the stationary Doob kernel.

Choose a proof deviation radius $r_g>0$ for fresh Gaussian averages.
It does not truncate any innovation. Let $M_e$ be (PC.12) and put

$$
\begin{gathered}
M_e^{g}=M_e+[t\lambda|c+a_x|\sigma_J+\alpha q]r_g,\\
E_g=2+t\lambda\rho e^{-1/2}+M_e^{g},\qquad
C_g=1+2t^2\lambda/e+t\ell_\rho(M_e^{g}+E_g),\\
\nu_g=\min\{1/(2t),\alpha/(2tC_g)\}>0.
\end{gathered}
\tag{NC.1}
$$

Require the additional primitive condition $0\le\nu\le\nu_g$.
It implies the previous interval (PC.12) and includes positive viscosity.
No condition is placed on an unknown QSD moment, curvature, spectral gap
or functional inequality. The actual quadratic floor $a_0>0$,
$m_0=a_0/2$ and $C=2/(\kappa_Cm_0)$ retain their uniform cap-$V_0$
values from (PC.2).

On the complete finite marked state set define

$$
\overline d_\omega(S,S')=\frac1N\sum_{i=1}^N d_\omega(z_i,z_i'),
\qquad D_\omega=2+2\omega V,
\qquad H_N=\{S:M(S)\ge m_0N\}.
\tag{NC.2}
$$

This is a true bounded metric with labelled physical coordinates and
discrete statuses. It is not a distance obtained by erasing dead rows.
The condition $H_N$ will be used inside an estimate and its actual QSD
exception will be retained. No deterministic alive floor is imposed on
the executed finite swarm.
:::

(sec-native-concentration-cap)=
## 2. The finite-population cap estimate with its Gaussian tails

:::{prf:lemma} Fresh Gaussian averages outside one labelled row
:label: lem-native-concentration-gaussian-average

Let $\zeta_1,\ldots,\zeta_N$ be independent standard $d$-Gaussians,
$g_1=\mathbb E|\zeta_1|$, and fix a row $i$. Then

$$
\Pr\left\{\frac1N\sum_{j\ne i}|\zeta_j|>g_1+r_g\right\}
 \le e^{-Nr_g^2/2}.
\tag{NC.3}
$$

The same bound holds for the original OU Gaussians. Both statements
are uniform conditional on the complete compact pre-jitter record.
They are not asserted conditional on arbitrary realized post-jitter arrays.
:::

:::{prf:proof}
The displayed Gaussian norm average is $N^{-1/2}$-Lipschitz in the
product Euclidean Gaussian coordinates and has mean at most $g_1$.
For completeness Gaussian log-Sobolev, proved by the OU interpolation
in {prf:ref}`thm-native-stationary-closure-spatial-poincare`, gives the
needed exponential bound directly. For an $L$-Lipschitz smooth bounded
function $F$, apply that inequality to $e^{uF/2}$. If
$\psi(u)=\log\mathbb E e^{uF}$, it yields
$u\psi'(u)-\psi(u)\le u^2L^2/2$.
Integrate the derivative of $\psi(u)/u$ from zero to obtain
$\log\mathbb E e^{u(F-\mathbb EF)}\le u^2L^2/2$.
Smooth approximation and truncation give it for the norm average.
Exponential Markov with $u=r_g/L^2$ proves (NC.3).
The original jitters and OU Gaussians are independent of the pre-jitter
record, so the same proof works under that conditioning.
:::

:::{prf:definition} Fourth-moment error budget
:label: def-native-concentration-cap-budget

Let

$$
\begin{gathered}
G_4=[d(d+2)]^{1/4},\quad R_4=R_D+\sigma_JG_4,\quad
T_4=ck_cV_0+ct\lambda R_4+qG_4,\quad K_P=1+G_4/g_1,\\
A_U^4=|D_c|+2t\nu c+4t\nu\ell_\rho bT_4,\qquad
A_X^4=t\lambda|c+a_x|+2t\nu ct\lambda+4t\nu\ell_\rho|a_x|T_4,\\
\vartheta=4t\nu k_cV\ell_\rho,\qquad
K_V=(1+t\nu)A_U^4,\qquad
K_X=K_P(A_U^4\vartheta+A_X^4),\\
\varepsilon_N=\sqrt2 e^{-Nr_g^2/4},\qquad
L_{\rm cap}=\max\{\omega(K_VL_{Vx}+K_XL_{Xx}),
                           K_VL_{Vv}+K_XL_{Xv}\}.
\end{gathered}
\tag{NC.4}
$$

The superscript $4$ denotes a fourth-Gaussian-moment budget, not a fourth
power of $A_U$ or $A_X$. All preparation coefficients are the already
derived (PC.8), at the actual fixed cap.
:::

:::{prf:theorem} High-alive finite-population contraction of the full raw marked kernel
:label: thm-native-concentration-finite-contraction

For $m_0N\ge2$ and $S,S'\in H_N$, there is a coupling of the complete
actual raw outputs such that

$$
\mathbb E\overline d_\omega(S^+,S^{\prime+})
 \le q_N\overline d_\omega(S,S'),\qquad
q_N=r+\varepsilon_NL_{\rm cap}.
\tag{NC.5}
$$

In particular $q_N<1$ for all sufficiently large $N$, with a primitive
population threshold given below. Unbounded realized jitters and the
OU correlations in B2 are included in the exponentially small error.
:::

:::{prf:proof}
The proof of {prf:ref}`lem-native-phase-preparation-coupling` was first
established for paired deterministic arrays, so apply that finite-array
coupling to $S,S'$. Freeze its entire compact paired pre-jitter record
$(Y_i,I_i,V_i^c;Y_i',I_i',V_i^{c\prime})$.
Interpolate these variables exactly as in the proof of
{prf:ref}`thm-native-phase-contraction`, retaining common original
independent jitter and OU arrays. The count first kick remains a convex
average and bounds each $U_{\theta,i}$ by $k_cV$.

For the conditional one-row B2 map, fix all jitters and all OU rows
other than $i$. Every $y_{\theta,j},z_{\theta,j}$ with $j\ne i$ is now
independent of the still fresh $\xi_i$. The actual second force for row
$i$ is (PC.16), with the fixed subprobability
$N^{-1}\sum_{j\ne i}\delta_{(y_{\theta,j},z_{\theta,j})}$.
For each other row its exact identity is

$$
e_{\theta,j}=z_{\theta,j}-t\lambda y_{\theta,j}
 =D_cU_{\theta,j}-t\lambda(c+a_x)X_{\theta,j}+\alpha q\xi_j.
\tag{NC.6}
$$

On the event that the two excluded Gaussian averages in (NC.3) are
at most $g_1+r_g$, their $N^{-1}$ average of $|e_{\theta,j}|$ is at
most $M_e^g$, for every interpolation value. Let $B_i$ be the complement
of this event. Conditional on the compact pre-jitter record,
$\Pr(B_i)\le2e^{-Nr_g^2/2}$. It does not involve either own Gaussian
$\zeta_i$ or $\xi_i$. The dependence of $U_{\theta,j}$ on all realized
jitters does not change (NC.6)'s deterministic bound $|U_{\theta,j}|\le k_cV$.
This is why the estimate is uniform before jitter, rather than uniform
over arbitrary post-jitter position arrays.

On $B_i^c$, the target-local derivative, properness and degree proof
of {prf:ref}`lem-native-phase-b2-local-density` applies to that
subprobability: its mass is at most one and its $e$-moment is bounded
by $M_e^g$. The interval (NC.1) keeps every relevant singular value
at least $\alpha/2$. Thus the same $H_v$ bounds its conditional
density on $B(0,1)$ and the same $\chi(V)$ in (PC.20) bounds the
conditional squared cap derivative. On $B_i$ retain the actual bound
$\|DC_V\|\le1$.

We give the explicit correlated cost of that exceptional event.
With the compact paired record fixed, put

$$
A_i=|Y_i'-Y_i|,\quad B_i'=\sigma_J|I_i'-I_i|,\quad
C_i=|V_i^{c\prime}-V_i^c|,\quad
P_i=A_i+B_i'|\zeta_i|.
$$

Dots along the comparison path satisfy

$$
\begin{gathered}
|\dot X_i|\le P_i,\quad
|\dot U_i|\le C_i+t\nu\overline C+
                      2t\nu V_c\ell_\rho(P_i+\overline P)=:Q_i,\\
|\dot y_i|\le |a_x|P_i+bQ_i=:Y_i^d,\quad
|\dot z_i|\le ct\lambda P_i+cQ_i=:Z_i^d,\quad
|z_i|\le cV_c+ct\lambda(R_D+\sigma_J|\zeta_i|)+q|\xi_i|=:T_i.
\end{gathered}
\tag{NC.7}
$$

Bars are actual normalized finite sums. Differentiating both the B2
velocity differences and their Gaussian weights bounds $|\dot u_i|$ by

$$
|D_c|Q_i+t\lambda|c+a_x|P_i+t\nu(Z_i^d+\overline Z^d)
 +t\nu\ell_\rho\{Y_i^d(T_i+\overline T)
                   +\overline{Y^dT}+\overline Y^dT_i\}.
\tag{NC.8}
$$

Self exclusion only decreases this bound. Minkowski and Hölder give
$\|T_i\|_4,\|\overline T\|_4\le T_4$ and
$\|P_i\|_4\le A_i+G_4B_i'$. The average of the $L^2$ norms of (NC.8)
is consequently at most

$$
A_U^4[(1+t\nu)\overline C+
                          \vartheta\overline{(A+G_4B')}]
 +A_X^4\overline{(A+G_4B')}.
$$

Conditional norm convexity, as in (PC.26), gives
$\mathbb E|X_i-X_i'|\ge A_i$ and
$\mathbb E|X_i-X_i'|\ge g_1B_i'$.
Writing $D_X=N^{-1}\sum_i\mathbb E|X_i-X_i'|$ and
$D_V=\overline C$, this norm budget is at most $K_VD_V+K_XD_X$.
Cauchy--Schwarz on each $B_i$ therefore bounds the average exceptional
derivative cost by
$\varepsilon_N(K_VD_V+K_XD_X)$.
This step retains the tails of the products of changed-outcome jitter
and the actual second force; it does not assume their independence.

On the complementary events, the same conditional Cauchy--Schwarz
with the own $\xi_i$ and the same weighted jitter estimates as
(PC.26)--(PC.30) give the original population budget
$\chi(V)[A_U(D_V+\vartheta D_X)+A_XD_X]$.
The finite sums use the same symmetric count cancellation; their
self-excluded donor OU variable is independent of the own one.
The weighted jitter bounds also include the random environment averages:
conditional on the compact paired record, for every $i,j$,
$\mathbb E[|X_{\theta,j}||\dot X_i|]\le R_J\mathbb E|\dot X_i|$.
For $j=i$ this is (PC.26); for $j\ne i$ the two recipient jitters
are independent under that conditioning. Thus, besides (PC.27),
$\mathbb E[\overline{|X_\theta|}\,\overline{|\dot U_\theta|}]
\le2R_JT_U$: its direct velocity terms cost at most
$(1+t\nu)(R_D+\sigma_Jg_1)D_V$, and its differentiated kernel terms
cost at most $\vartheta R_JD_X$. These inequalities control all four
root/donor products involving a finite random position average. The
original OU coordinates are independent of every such preparation
derivative; their cap-weighted own term uses $\sqrt d$ as in (PC.29).
Adding the exceptional budget and integrating the path gives the
bottom row of (PC.23) plus

$$
\varepsilon_N\bigl[(K_VL_{Vx}+K_XL_{Xx})\delta_x
                    +(K_VL_{Vv}+K_XL_{Xv})\delta_v\bigr].
$$

The first matrix row is unchanged: independent conditional maximal
couplings of the actual last position Gaussians control all terminal
marks exactly as in (PC.23). Multiplication by $1,\omega$, and (NC.4)
now give (NC.5). These are couplings of the original marginal kernels;
no noise or transition was added.
:::

(sec-native-concentration-one-step)=
## 3. A full marked one-step variance estimate with dense kinetics

:::{prf:lemma} Chronological innovation variance by coupling unrevealed blocks
:label: lem-native-concentration-chronological-variance

Let a full update from fixed input $S$ be a measurable function $F$ of
independent innovation blocks $\xi_1,\ldots,\xi_M$ in chronological
stage order. For block $j$, fix the revealed prefix and replace $\xi_j$
by an independent copy. Couple the two unrevealed suffixes in any manner
that preserves each of their original conditional product laws. Denote
the resulting two output values by $F_j,F_j'$. Then

$$
\operatorname{Var}(F\mid S)
 \le\frac12\sum_{j=1}^M\mathbb E[(F_j-F_j')^2\mid S].
\tag{NC.9}
$$

The suffix couplings may depend on the paired prefixes. In particular
the last position Gaussians need not be identically coupled when an
earlier stage block is varied.
:::

:::{prf:proof}
Let $M_j=\mathbb E[F\mid S,\xi_1,\ldots,\xi_j]$.
Orthogonality of the Doob martingale increments gives
$\operatorname{Var}(F\mid S)=\sum_j\mathbb E[(M_j-M_{j-1})^2\mid S]$.
For the paired copies of $\xi_j$ with their common prefix, the conditional
variance identity gives each summand as one half the expected square
of the two conditional suffix expectations' difference. Every permitted
coupling has exactly those two expectations as its marginal means.
Conditional Jensen bounds their squared difference by the coupled
suffix output difference squared. Summation proves (NC.9).
:::

:::{prf:definition} Explicit complete-update influence constants
:label: def-native-concentration-influence-budget

Use the actual stored cap $V$ and define the inflated Gaussian envelopes

$$
\begin{gathered}
R_I=R_D+2\sigma_JG_4,\qquad
T_I=cV_c+ct\lambda R_I+2qG_4,\\
A_U^I=|D_c|+2t\nu c+4t\nu\ell_\rho bT_I,\qquad
A_X^I=t\lambda|c+a_x|+2t\nu ct\lambda+4t\nu\ell_\rho|a_x|T_I,\\
P_I=2R_D+2\sigma_JG_4,\qquad
Q_I=2(1+t\nu)V_c+\vartheta P_I,\quad
Y_I=|a_x|P_I+bQ_I,\quad U_I=A_U^IQ_I+A_X^IP_I,\\
Y_J=(|a_x|+b\vartheta)2\sigma_JG_4,\qquad
U_J=(A_U^I\vartheta+A_X^I)2\sigma_JG_4,\\
Y_O=2tqG_4,\qquad
U_O=2qG_4[\alpha+2t\nu+4t^2\nu\ell_\rho T_I],
\qquad\zeta_0=1/(\sqrt{2\pi}s).
\end{gathered}
\tag{NC.10}
$$

For a position and velocity influence pair put

$$
\mathcal A(Y,U)=8\zeta_0^2Y^2+8\zeta_0Y+2\omega^2U^2,
\quad A_I=\mathcal A(Y_I,U_I),\quad
A_J=\mathcal A(Y_J,U_J),\quad A_O=\mathcal A(Y_O,U_O).
\tag{NC.11}
$$

Finally, using the actual shifted diversity range $S_b$ in (PC.4), put

$$
\begin{gathered}
L_q=\frac{S_b}{m_0\sigma_s}+\frac{3S_b^3}{2m_0\sigma_s^3},\quad
L_0=H_sL_q,\quad B=C+2L_gL_0,\quad M_2(C)=(1+2C)e^{4C},\\
A_D=9M_2(2C)[(1+B)^2+B]+\max\{1,4B^2\},\\
A_{\rm step}=\tfrac12\{A_I[A_D+10M_2(C)]+A_J+A_O+4\}.
\end{gathered}
\tag{NC.12}
$$

These constants are finite and population-independent. Their Gaussian
envelopes explicitly include both copies of a replaced jitter or OU
block. They do not bound those original noises pointwise.
:::

:::{prf:theorem} Uniform conditional variance for the complete marked dense count update
:label: thm-native-concentration-one-step-variance

For $m_0N\ge2$, $S\in H_N$ and every real-valued function $f$ with
$\operatorname{Lip}_{\overline d_\omega}(f)\le1$ on the complete
marked output space,

$$
\operatorname{Var}(f(S^+)\mid S)\le A_{\rm step}/N.
\tag{NC.13}
$$

The variance is for the actual complete raw output, including terminal
alive/dead statuses. Its dense B2 interaction is not treated as row local.
:::

:::{prf:proof}
Realize the existing staged transition using $N$ measurement blocks,
$N$ clone-donor/gate blocks, $N$ Haar blocks indexed by possible component
representatives, $N$ original jitter blocks, $N$ OU blocks, and $N$ final
position-noise blocks. Assigning an unused independent Haar variable to
a nonrepresentative is a latent realization of the same one-Haar-per-
component law, not an additional executed collision.
Use chronological variance (NC.9). Suffix measurements and primitive
clone uniforms are identically coupled when their actual input arguments
agree; common Haar rotations are used on unchanged components.

First vary one measurement block. Its changed sampled distance changes
the alive mean by at most $S_b/(m_0N)$ and the variance by at most
$3S_b^2/(m_0N)$. Hence every other row's standardized score changes by
at most $L_q/N$, and its fitness by at most $L_0/N$.
Freeze the two full measured arrays. For each other recipient the chance
its accepted outcome differs is at most $B/N$: the $C/N$ term pays for
choosing the exceptional donor, and the other term pays for the two
gate fitness arguments. This includes mandatory revival, whose gate is
one. These exceptional-row events are independent under this conditioning.
Their count $Q$, including the changed measurement row, has
$\mathbb E Q^2\le(1+B)^2+B$.

For $N\ge2B$, expose the exceptional outcomes and delete their outgoing
edges. The remaining common graph retains independent row choices with
specified-edge probabilities at most $2C/N$; conditioning a row on its
unchanged outcome divides its bound by at most $1-B/N\ge1/2$.
Its edges respect both strict fitness orders, so it remains an ordered
forest. At most $3Q$ fixed seeds are incident to exceptional outcomes.
The proved component second moment then bounds the squared number
$D$ of affected rows by $9Q^2M_2(2C)$. For $N<2B$ simply use
$D\le N<2B$. These statements give $\mathbb E D^2\le A_D$.
They are the pre-kinetic component argument of
{prf:ref}`lem-chaos-canonical-innovation-replacement`; its row-local
kinetic conclusion is not used for dense viscosity.

Varying one clone-donor/gate block instead has
$\mathbb ED^2\le9M_2(C)$: delete that one outgoing edge, expose its
two outcomes, and apply the second-moment bound to the at most three
fixed seeds in the unchanged forest. Varying one addressed Haar block
has $\mathbb ED^2\le M_2(C)$. In each of these cases the affected set
and all paired source/gate/Haar variables are independent of every future
jitter and OU innovation. Outside the set the complete preparations agree.

Conditional on such a set of $D$ rows, source differences are at most
$2R_D$, jitter indicators differ by at most one, and collision velocity
differences are at most $2V_c$. Apply the differentiated dense kinetic
formula (NC.7)--(NC.8), without the cap contraction and with its actual
global bound $\|DC_V\|\le1$. Minkowski and Hölder give

$$
\left\|\sum_i|y_i-y_i'|\right\|_2\le Y_ID,\qquad
\left\|\sum_i|v_i^+-v_i^{\prime+}|\right\|_2\le U_ID.
\tag{NC.14}
$$

For the second inequality use the $L^4$ envelopes in (NC.10) for every
kernel derivative times actual velocity. The summed first-kick velocity
terms are bounded by $2(1+t\nu)V_cD$, and its kernel terms by
$\vartheta P_ID$; this gives $Q_I$. These sums propagate through
both kicks via $A_U^I,A_X^I$. The doubled Gaussian envelopes are
larger than required for these three stage types but remain valid.

Couple the unrevealed last position Gaussians independently by common
minimum densities, conditional on the paired earlier arrays. If $K$ is
the number of unequal position outcomes, it is a sum of independent
Bernoulli variables under that conditioning and has mean at most
$\zeta_0\sum_i|y_i-y_i'|$. Therefore

$$
\mathbb E K^2\le
 \zeta_0^2\mathbb E\left(\sum_i|y_i-y_i'|\right)^2
 +\zeta_0\mathbb E\sum_i|y_i-y_i'|.
$$

Each such outcome has marked position cost at most two.
Use $(u+v)^2\le2u^2+2v^2$, (NC.14), and $D\le D^2$ whenever
$D\ge1$, to obtain expected squared normalized output distance at most
$A_I\mathbb ED^2/N^2$. Every terminal status is included in this
bound. Dense interactions may change all velocities; their complete
summed influence is the second term of the distance.

For one jitter block, interpolate its two original Gaussian copies.
Only one $X_i$ changes directly, with $L^4$ discrepancy at most
$2\sigma_JG_4$. The same dense kinetic budget gives $Y_J,U_J$.
For one OU block the summed drift-position derivative has norm budget
$Y_O$, and the pre-cap B2 budget is $U_O$: its linear part is
$\alpha q\Delta\xi$, its viscous velocity terms cost at most
$2t\nu q|\Delta\xi|$, and its four kernel products cost at most
$4t^2\nu\ell_\rho T_Iq\|\Delta\xi\|_4$.
Thus their squared normalized output costs are bounded by
$A_J/N^2$ and $A_O/N^2$. The envelopes $R_I,T_I$ keep every original
and replaced Gaussian tail, including correlations with the force.
Finally replacing one last position block changes only one final marked
row, by cost at most two, and has squared distance at most $4/N^2$.

There are $N$ blocks of each of the six types. Substitute the respective
cost bounds into (NC.9) to obtain (NC.12)--(NC.13).
All prefix averaging in this martingale calculation is over its actual
original law. Component bounds are used after freezing measurements and
deleting exposed exceptional edges; they are not asserted after fixing
an arbitrarily influential revealed component.
:::

(sec-native-concentration-stationary)=
## 4. A population-uniform full marked QSD variance inequality

:::{prf:definition} Explicit population threshold and stationary constant
:label: def-native-concentration-stationary-constant

Let $\bar r=(1+r)/2<1$ and define

$$
\begin{gathered}
N_{\rm cap}=\left\lceil\frac4{r_g^2}
       \max\left\{0,\log\frac{2\sqrt2 L_{\rm cap}}{1-r}\right\}\right\rceil,
\qquad
N_{\rm surv}=\left\lceil\frac1{a_0}\log\frac4{1-\bar r^2}\right\rceil,\\
N_0=\max\{1,\lceil2/m_0\rceil,N_{\rm cap},N_{\rm surv}\},\qquad
\eta=\frac{1-\bar r^2}{2\bar r^2},\qquad
c_*=(1-\log2)/2,\\
C_{\rm stat}=\max\left\{
 \frac{N_0D_\omega^2}4,
 \frac4{1-\bar r^2}\left[
 A_{\rm step}+\frac{D_\omega^2(5/4+1/\eta)}{e c_*a_0}
                         \right]\right\}.
\end{gathered}
\tag{NC.15}
$$

If $L_{\rm cap}=0$, use $N_{\rm cap}=0$.
Every quantity is a finite explicit function of the complete consumed
configuration and proof choices; in particular $C_{\rm stat}$ is
independent of $N$.
:::

:::{prf:theorem} Uniform stationary variance for native marked Lipschitz observables
:label: thm-native-concentration-stationary-variance

Under {prf:ref}`def-native-concentration-register`, for every permitted
population and every real-valued full marked swarm function $f$ with evaluated
Lipschitz constant $L_f$ in the true metric $\overline d_\omega$,

$$
\operatorname{Var}_{\nu_N}f\le\frac{C_{\rm stat}L_f^2}{N},\qquad
\nu_N\{|f-\nu_Nf|\ge u\}
            \le\frac{C_{\rm stat}L_f^2}{Nu^2}\quad(u>0).
\tag{NC.16}
$$

This applies to any averaged bounded row observable that is Lipschitz
in $d_\omega$, including the actual empirical alive fraction. The QSD
is the complete killed-chain QSD, rather than a conservative surrogate
or the Doob invariant law. This is a stationary Lipschitz variance
inequality; it is not an unconditional stationary gradient LSI.
:::

:::{prf:proof}
Scale to $L_f=1$ and subtract a constant so $0\le f\le D_\omega$.
Let
$V_N=\sup_{\operatorname{Lip}_{\overline d_\omega}f\le1}
\operatorname{Var}_{\nu_N}f\le D_\omega^2/4$.
The full raw QSD identity is

$$
\nu_NP_N^{\rm raw}=\alpha_N\nu_N+(1-\alpha_N)\zeta_N^{\dagger},
\qquad\alpha_N\ge1-(1-a_0)^N,
\tag{NC.17}
$$

where $\zeta_N^{\dagger}$ is its actual all-dead output law.
Mixture variance gives
$\alpha_N\operatorname{Var}_{\nu_N}f
\le\operatorname{Var}_{\nu_NP_N^{\rm raw}}f$.
The QSD alive-floor exception proved by the binomial subtraction is

$$
\nu_N(H_N^c)\le\delta_N:=e^{-c_*a_0N}.
\tag{NC.18}
$$

For $N\ge N_0$, (NC.5) gives $q_N\le\bar r$ on $H_N$ and
$\alpha_N\ge1-(1-\bar r^2)/4$.
Set $g=P_N^{\rm raw}f$. It is $q_N$-Lipschitz on $H_N$ by
(NC.5). Extend it to the entire marked state set by
$\widetilde g(s)=\inf_{t\in H_N}[g(t)+q_N\overline d_\omega(s,t)]$,
then clip to $[0,D_\omega]$. The triangle inequality proves its
$q_N$-Lipschitz property, and the same Lipschitz property on $H_N$
proves equality with $g$ there. Thus
$\mathbb E_{\nu_N}|g-\widetilde g|^2\le D_\omega^2\delta_N$ and
$\operatorname{Var}_{\nu_N}\widetilde g\le\bar r^2V_N$.
No law-specific functional inequality was used in this extension.

The variance triangle inequality followed by
$2ab\le\eta a^2+b^2/\eta$ yields

$$
\operatorname{Var}_{\nu_N}g
\le(1+\eta)\bar r^2V_N+(1+1/\eta)D_\omega^2\delta_N.
$$

On $H_N$ the conditional variance is at most $A_{\rm step}/N$ by
(NC.13); on its complement it is at most $D_\omega^2/4$.
Total variance in (NC.17), followed by the supremum over $f$, gives

$$
\bigl[\alpha_N-(1+\eta)\bar r^2\bigr]V_N
 \le A_{\rm step}/N+D_\omega^2(5/4+1/\eta)\delta_N.
$$

The bracket is at least $(1-\bar r^2)/4$ by (NC.15).
Also $N\delta_N\le1/(e c_*a_0)$, since the maximum of
$x e^{-c_*a_0x}$ is that value. This proves (NC.16)'s variance bound
for $N\ge N_0$. For $N<N_0$ the trivial bound
$V_N\le D_\omega^2/4\le N_0D_\omega^2/(4N)$ gives the same conclusion.
Chebyshev proves the stated probability estimate.

An averaged $d_\omega$-Lipschitz row function has the same Lipschitz
constant in $\overline d_\omega$ by summation. The alive indicator has
constant one because $|a-a'|\le d_\omega(z,z')$.
This proves the advertised actual status-sensitive concentration.
:::

(sec-native-concentration-entropy-transport)=
## 5. Unconditional marked transport and an entropy profile

:::{prf:theorem} Native QSD transport from its derived stationary variance
:label: thm-native-concentration-entropy-transport

Under the same complete parameter regime, let $W_N$ be the transport
distance for $\overline d_\omega$ on the ambient capped marked swarm
space, and let $\eta$ be any probability on that space. Put
$v_N=C_{\rm stat}/N$ and $D=D_\omega$.
With infinite divergences interpreted as giving a vacuous upper bound,

$$
W_N(\eta,\nu_N)^2\le v_N\chi^2(\eta\mid\nu_N),\qquad
\chi^2(\eta\mid\nu_N)=\nu_N\left|\frac{d\eta}{d\nu_N}-1\right|^2.
\tag{NC.20}
$$

There is also the unconditional entropy-transport inequality

$$
\operatorname{KL}(\eta\mid\nu_N)
 \ge\frac{v_N}{D^2}
 h\left(\frac{D W_N(\eta,\nu_N)}{v_N}\right),\qquad
h(u)=(1+u)\log(1+u)-u.
\tag{NC.21}
$$

In particular its explicit Bernstein consequence is

$$
W_N(\eta,\nu_N)
 \le\sqrt{2v_N\operatorname{KL}(\eta\mid\nu_N)}
            +\frac D3\operatorname{KL}(\eta\mid\nu_N).
\tag{NC.22}
$$

For each normalized marked Lipschitz function $f$ the corresponding
unconditional stationary exponential-moment bound is

$$
\log\nu_N e^{u(f-\nu_Nf)}
 \le\frac{v_N}{D^2}(e^{uD}-uD-1),\qquad u\ge0.
\tag{NC.23}
$$

These are law-level estimates for the actual full marked QSD. No
transport inequality or stationary entropy estimate is assumed. The
profile (NC.21) is locally quadratic with coefficient $1/(2v_N)$.
Its large-discrepancy profile is weaker, and is not asserted to be an
$N$-uniform gradient LSI or a global quadratic $T_1$ inequality.
:::

:::{prf:proof}
The bounded ambient metric space is Polish: its coordinates are
$\mathbb R^{dN}$ with bounded Euclidean metrics, closed capped velocity
balls and finite discrete status sets. Actual consistency of statuses
restricts the kernel's support but need not alter that ambient space.
For completeness its transport dual is

$$
W_N(\eta,\nu_N)=
 \sup_{\operatorname{Lip}_{\overline d_\omega} f\le1}
                         (\eta f-\nu_Nf).
\tag{NC.24}
$$

To verify this dual without a compact-support replacement of the actual
law, first quantize both probabilities using a finite small net on a
compact set of mass at least $1-\varepsilon$, assigning its complement
to one additional point. Quantization cost is at most
$\varepsilon+D\varepsilon$. On the finite common support, transportation
linear-program duality applies: the primal is feasible via the product
coupling and finite bounded cost. Its dual constraints are
$u_i+v_j\le d_{ij}$. Replace $u$ by
$f_i=\min_j(d_{ij}-v_j)$; the triangle inequality makes $f$ 1-Lipschitz,
$u_i\le f_i$ and $v_i\le-f_i$. Thus its dual reduces exactly to the
displayed Lipschitz difference. Conversely $(f,-f)$ is feasible for
every 1-Lipschitz $f$. Extend finite $f$ by the same infimum formula
as in the preceding proof. Quantization errors in both transport and
the Lipschitz differences are at most twice the displayed cost;
couplings lift by the conditional laws in each quantization cell.
Letting $\varepsilon\downarrow0$ gives (NC.24). The compact sets are
used only to prove duality; no physical coordinate or tail is removed.

If the chi-square divergence is finite, write
$R=d\eta/d\nu_N$. Centered covariance and (NC.16) give for each
1-Lipschitz $f$

$$
|\eta f-\nu_Nf|
=|\nu_N[(R-1)(f-\nu_Nf)]|
 \le\sqrt{v_N\chi^2(\eta\mid\nu_N)}.
$$

Take the supremum in (NC.24) to prove (NC.20).

For the entropy statement, a centered 1-Lipschitz $f$ has
$|f-\nu_Nf|\le D$ and variance at most $v_N$.
Termwise expansion of the exponential, using
$|x|^k\le x^2D^{k-2}$, proves for $|x|\le D$ and $u\ge0$

$$
e^{ux}-1-ux\le\frac{x^2}{D^2}(e^{uD}-1-uD).
$$

Integrate it and use $\log(1+t)\le t$ to obtain (NC.23).
For any bounded $g$, nonnegativity of relative entropy against
$e^g\nu_N/(\nu_Ne^g)$ gives

$$
\operatorname{KL}(\eta\mid\nu_N)
                  \ge\eta g-\log\nu_Ne^g.
$$

All terms are defined when the entropy is finite, and otherwise the
desired assertion is immediate. Set $g=u(f-\nu_Nf)$, take the supremum
in (NC.24), and then optimize $u\ge0$. The optimum satisfies
$uD=\log(1+D W_N/v_N)$, yielding precisely (NC.21).
Since $h(t)=t^2/2+o(t^2)$, this also gives the stated local coefficient.

Finally $k!\ge2\,3^{k-2}$ for $k\ge2$ bounds the same exponential series
by $v_Nu^2/[2(1-uD/3)]$ for $0<u<3/D$.
Put $H=\operatorname{KL}(\eta\mid\nu_N)$ in the entropy variational
bound and choose
$u=\sqrt{2H/v_N}/[1+(D/3)\sqrt{2H/v_N}]$.
For $H=0$ take its limiting value. The resulting bound is (NC.22).
Every moment estimate used here is the newly proved QSD variance,
not a premise about that QSD's curvature or entropy production.
:::

(sec-native-concentration-evaluation)=
## 6. Evaluated regime and exact reach

:::{prf:corollary} Evaluated positive-parameter stationary concentration
:label: cor-native-concentration-positive-witness

Use exactly the complete parameter witness (PC.37), its configured
cap $V=V_{\rm crit}/2$, and its weight
$\omega=\sqrt{m_{12}/m_{21}}$. Retain its certified contraction upper
bound $r=1/2$ and choose proof radius $r_g=1$. This instance satisfies
the additional viscosity interval (NC.1), so every assertion of
(NC.16) applies. Its exact formulas give the diagnostic evaluations

$$
\begin{gathered}
M_e^g\simeq2.588347982,\quad\nu_g\simeq0.06751779354>0.01,
\quad L_{\rm cap}\simeq350978.3494,\\
N_0=59\text{ diagnostically},\quad
A_D\simeq3.675098301\,10^{21},\quad
A_{\rm step}\simeq3.308674903\,10^{29},\\
C_{\rm stat}\simeq3.025074197\,10^{30},\qquad
\log C_{\rm stat}\simeq70.18448841.
\end{gathered}
\tag{NC.19}
$$

The displayed size is a conservative finite coefficient. The exact
expressions (NC.1)--(NC.15), rather than these decimal evaluations,
define the inequality for every population. A fixed finite execution
ceiling retains its original scope; the estimates do not remove it.
:::

:::{prf:proof}
We verify the additional positive regime analytically. In this witness
$t=1/2$, $1/3<c<2/5$, $\lambda<3$, $1/4<\alpha<1/3$,
$q<1$, $R_D<3.5$, $g_1\le\sqrt3<2$, and $|D_c|<1$.
Consequently $M_e<3.1$, $M_e^g<3.5$, $E_g<7$ and $C_g<8$.
Thus $\nu_g>1/32>0.01$, proving the required condition without
depending on the diagnostic decimal. The exact cap and weight have
already proved $r\le1/2$ in (PC.33). All constants in (NC.15) are
finite, so the stationary inequality follows from the preceding theorem.
Substitution in those expressions gives the reported diagnostic scales.
:::

:::{prf:remark} Parameter regimes and the remaining transfer
:label: rem-native-concentration-scope

The positive witness of {prf:ref}`cor-native-phase-positive-witness`
also satisfies (NC.1) at $r_g=1$, so (NC.16) gives an actual nonempty
positive-viscosity, active-fitness parameter regime. Its exact finite
coefficient may be large; the theorem concerns population uniformity,
not a practical finite-reference mixing estimate. The unchanged reference
still fails the contraction certificate as proved in (PC.40). This is
not a counterexample to the reference's stationary chaos.

The inequality includes full labelled swarm observables and actual dead
marks, but its Lipschitz constant is in the bounded marked transport
metric. Transfer to a specified physical gradient, a full stationary
gradient entropy-production inequality, or the Doob invariant law requires the corresponding
additional proved comparison. Neither an $N$-uniform eigenfunction ratio
nor any of those transfers is presumed here. Row normalization, other
landscapes and variant branches retain their separate parameter tests.
:::
