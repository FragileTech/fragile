# Stationary control of the existing dense viscous update

(sec-native-stationary-register)=
## 1. Complete data and the scope of each result

:::{prf:definition} Stationary-control execution register
:label: def-native-stationary-register

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`. The results below use its
restriction to the real-coordinate canonical dense Viscous Euclidean Gas in
{prf:ref}`def-cgd-parameter-register`, with the current-frame independent
count-one companion samplers, the canonical globally regularized sampled
fitness and clipped gate, simultaneous immutable-source copying, recipient
Gaussian jitter, one shared Haar rotation per accepted connected component,
both BAOAB force evaluations, final radial cap and terminal classification.
The two normalization tags remain distinct.

The numerical and landscape block is the complete tuple

$$
\begin{split}
\Theta={}&(d,N,h,\gamma,b_O,\sigma_x,\sigma_J,V,
 \alpha_{\rm col},R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg},
 \epsilon_D,\epsilon_C,\delta_D,A_r,A_s,\eta_r,\eta_s,
 p_r,p_s,\sigma_r,\sigma_s,s_c,\epsilon_c;\ U,R,D;
 \nu,\rho,\mathfrak n,\Gamma,\vartheta_\Gamma).
\end{split}
$$

Every positive floor, width and regularizer keeps its canonical restriction;
the uniform-companion convention is included when configured. The signs and
units are those in Chapters 18 and 19. Write

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
\eta=t^2(1+c),\quad
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}\quad s^2=\sigma_x^2h,
$$

$$
\tau^2=t^2q^2+s^2,\qquad
V_c=(1+2|\alpha_{\rm col}|)V,\qquad \ell_K=e^{-1/2}/\rho.
$$

The kernel also retains the following restrictions of the complete record:
no historical donor pool, elite retention, graph/metric feedback, fitness
gradient force, Boris/curl rotation, adaptive diffusion, innovation shift,
periodic wrapping, intervening boundary test or extra kinetic substep. These
are restrictions to an existing instance, rather than replacements for those
other algorithms. In particular the graph-based Python viscosity in
`KineticOperator._compute_viscous_force` is not identified with this dense
Rust kernel. The innovation convention is independent full Gaussian draws in
real arithmetic; a finite-precision seeded execution is a different record.
Allocation and error policies must allow the stated updates to exist.

The initial law includes every coordinate consumed by this restricted
transition. All passive geometry/recording parameters, stage tags, masks,
tie conventions and calibrations in $\mathfrak P$ are retained; they do not
change this physical state kernel. A stationary law below is a law of these
kernel-consuming physical coordinates. A monotonically increasing history
clock or genealogy label appended solely for recording is not asserted to
have an invariant probability.

Additional choices in a proof, such as a quadratic comparison matrix or
Gaussian-event radius, are named arguments of its certificate. The quadratic
coefficient, bandwidth, viscosity and timestep in every formula are the
already configured values. The reference calculations use $\lambda=1$,
$h=0.04$, $\nu=0.3$ and $\rho=1$; no proof replaces its force.
:::

(sec-native-stationary-signed)=
## 2. Signed viscosity in the complete coupled update

:::{prf:proposition} Signed count and row feedback balances
:label: prop-native-stationary-signed-viscosity

For two prepared arrays $(X,V)$ and $(\widetilde X,\widetilde V)$ put
$\delta X=X-\widetilde X$, $\delta V=V-\widetilde V$. Write
$G(X,V)$ for the actually configured viscous force. For count normalization,
with $L_X$ as in {prf:ref}`lem-cgd-count-kick`,

$$
\delta G=-\nu L_X\delta V-\nu(L_X-L_{\widetilde X})\widetilde V,
\tag{NS.1}
$$

and its exact work is

$$
\begin{split}
\langle\delta V,\delta G\rangle_N
={}&-\frac{\nu}{2N^2}\sum_{i,j}K_{ij}(X)
          |\delta V_i-\delta V_j|^2\\
&-\frac{\nu}{2N^2}\sum_{i,j}[K_{ij}(X)-K_{ij}(\widetilde X)]
 (\delta V_i-\delta V_j)\cdot(\widetilde V_i-\widetilde V_j).
\end{split}
\tag{NS.2}
$$

For row normalization, set $\omega_{ij}(X)=K_{ij}(X)/d_i(X)$ for
$i\ne j$ and zero on the diagonal, and
$a_j(X)=\sum_i\omega_{ij}(X)$. For $N\ge2$ the exact unweighted work is

$$
\begin{split}
\langle\delta V,\delta G\rangle_N
={}&-\frac{\nu}{2N}\sum_{i,j}\omega_{ij}(X)
                    |\delta V_i-\delta V_j|^2
 +\frac{\nu}{2N}\sum_j[a_j(X)-1]|\delta V_j|^2\\
&+\frac{\nu}{N}\sum_{i,j}
 [\omega_{ij}(X)-\omega_{ij}(\widetilde X)]
              \delta V_i\cdot\widetilde V_j.
\end{split}
\tag{NS.3}
$$

The column imbalance and changed-position feedback have no prescribed sign.
The fixed-position degree-weighted dissipation of Chapter 18 does not remove
either term when the two swarms have different positions. For $N=1$ all
viscous terms are zero.

For count normalization there is also the population-independent exact-input
budget

$$
\begin{split}
\mathbb E\|\delta G\|_{2,N}^2
\le{}&2\nu^2\mathbb E\|\delta V\|_{2,N}^2\\
&+\frac{4\nu^2\ell_K^2}{N^2}\sum_{i,j}
 \mathbb E[(|\delta X_i|^2+|\delta X_j|^2)
                       |\widetilde V_i-\widetilde V_j|^2].
\end{split}
\tag{NS.4}
$$

If $M_4=\sup_i\mathbb E|\widetilde V_i|^4<\infty$ and
$D_4=N^{-1}\sum_i\mathbb E|\delta X_i|^4<\infty$, its last term is at most
$32\nu^2\ell_K^2\sqrt{M_4D_4}$. The uncapped OU velocities can be inserted
in these formulas without changing their distributions. A fourth-moment
bound makes this error finite; it does not make it proportional to the
quadratic discrepancy.
:::

:::{prf:proof}
Subtract $G(X,V)=-\nu L_XV$ at the two arrays to obtain (NS.1).
The matrices are symmetric, so pairing the summands $(i,j)$ and $(j,i)$
gives both terms of (NS.2). In the row case subtract
$G_i=-\nu V_i+\nu\sum_j\omega_{ij}V_j$. Expand
$-\frac12\sum_{ij}\omega_{ij}|\delta V_i-\delta V_j|^2$,
using row sums one but retaining the column sums. Adding the changed-row
term gives (NS.3).

For (NS.4), $\|L_X\delta V\|_{2,N}\le\|\delta V\|_{2,N}$.
Jensen on each row and the Gaussian kernel derivative bound give

$$
\|(L_X-L_{\widetilde X})\widetilde V\|_{2,N}^2
\le\frac1{N^2}\sum_{i,j}|K_{ij}(X)-K_{ij}(\widetilde X)|^2
                         |\widetilde V_i-\widetilde V_j|^2
\le\frac{2\ell_K^2}{N^2}\sum_{i,j}
 (|\delta X_i|^2+|\delta X_j|^2)|\widetilde V_i-\widetilde V_j|^2.
$$

Apply $|a+b|^2\le2|a|^2+2|b|^2$ in (NS.1). For the last claim,
$\mathbb E|\widetilde V_i-\widetilde V_j|^4\le16M_4$.
Cauchy--Schwarz bounds each mixed expectation by
$4\sqrt{M_4\mathbb E|\delta X_i|^4}$ or its $j$ version.
The averaged sum is at most $8\sqrt{M_4D_4}$, giving the stated constant.
:::

:::{prf:proposition} Both viscous kicks in the actual signed quadratic ledger
:label: prop-native-stationary-two-kick-ledger

Retain any coupling of the actual cloning/collision preparations with correct
marginal laws, and use equal fresh Gaussian innovations in the two kinetic
updates. For a configured quadratic force $F(x)=-\lambda x$, define

$$
\begin{gathered}
H=\delta V-t\lambda\delta X,\qquad E_0=t\delta G_0,\qquad
Y_0=\delta X+bH,\qquad U_0=cH-t\lambda Y_0,\\
E_1=(c-t\lambda b)E_0+t\delta G_2.
\end{gathered}
\tag{NS.5}
$$

The second force difference $\delta G_2$ is evaluated at the actual arrays

$$
Z=c(V+tF(X)+tG(X,V))+q\xi,\qquad
Y=X+t(V+tF(X)+tG(X,V))+tZ,
\tag{NS.6}
$$

and their counterparts; it retains the entire uncapped $\xi$.
For $A>0$ and $A>B^2$ put
$\mathscr Q(Y,U)=N^{-1}\sum_i(A|Y_i|^2+2B Y_i\cdot U_i+|U_i|^2)$.
Before the actual cap, the exact difference energy is

$$
\begin{split}
\mathscr Q(\delta Y,\delta U)
={}&\mathscr Q(Y_0,U_0)
 +2b\langle E_0,AY_0+BU_0\rangle_N
 +2\langle E_1,BY_0+U_0\rangle_N\\
&+Ab^2\|E_0\|_{2,N}^2
 +2Bb\langle E_0,E_1\rangle_N+\|E_1\|_{2,N}^2.
\end{split}
\tag{NS.7}
$$

The completed energy adds the actual signed cap correction
$\mathscr Q(\delta Y,C_V(U)-C_V(\widetilde U))-
\mathscr Q(\delta Y,\delta U)$, followed, when a marked-state comparison is
used, by the declared terminal-status discrepancy. Add the exact signed
cloning/collision change of Chapter 06a before taking expectation. No positive
selection term, second-kick error, cap cross term or status transition vanishes
merely from the fixed-position alignment sign.
:::

:::{prf:proof}
The difference of the first kicks is $H+E_0$. Equal innovations give
$\delta Z=c(H+E_0)$ and
$\delta Y=\delta X+b(H+E_0)=Y_0+bE_0$. The second quadratic force is
$-\lambda\delta Y$, so
$\delta U=c(H+E_0)-t\lambda(Y_0+bE_0)+t\delta G_2=U_0+E_1$.
Expand the quadratic form at this pair to obtain (NS.7). Final position
innovations cancel in the difference but still enter the terminal labels.
Applying the actual cap and classification gives their displayed corrections.
The preparation coupling was arbitrary subject to the real marginal laws, so
its preceding signed cloning ledger can be added by conditional expectation.
:::

(sec-native-stationary-reference-kinetic)=
## 3. Population-independent kinetic budgets at the unchanged reference force

:::{prf:lemma} Gaussian contraction through the actual radial cap
:label: lem-native-stationary-gaussian-cap

For $C_V(u)=Vu/(V+|u|)$, a Gaussian $Z\sim N(m,\sigma^2I_d)$,
$\sigma\ge\sigma_0>0$, a deterministic vector $a$ and a proof radius
$r_g>0$, write $G_d(u)=P(|Z_d|\le u)$ for $Z_d\sim N(0,I_d)$ and define

$$
\kappa_V=G_d(r_g/\sigma_0)
 +[1-G_d(r_g/\sigma_0)]\left(\frac{V}{V+r_g}\right)^2<1.
$$

Then, uniformly over $m$,
$\mathbb E|C_V(Z+a)-C_V(Z)|^2\le\kappa_V|a|^2$.
:::

:::{prf:proof}
The cap derivative norm is at most $V/(V+|u|)$. A ball's Gaussian
probability is maximized at zero mean: rotate the mean onto the first
axis, condition on the remaining centered coordinates, and differentiate
the probability that a one-dimensional Gaussian of shifted mean lies in
the resulting symmetric interval. That derivative is nonpositive for a
nonnegative shift. Increasing variance decreases the centered ball
probability. Thus every shifted $N(m,\sigma^2I_d)$ puts probability at
most $G_d(r_g/\sigma_0)$ inside $B_{r_g}$.
Integrate the cap derivative along the segment from $Z$ to $Z+a$ and
apply Cauchy--Schwarz in the segment parameter. At every segment point,
the squared derivative norm has expectation at most $\kappa_V$, proving
the inequality. Strict positivity of the Gaussian exterior probability
and $V/(V+r_g)<1$ give $\kappa_V<1$.
:::

:::{prf:theorem} Uniform actual count-normalized B2 force and derivative budgets
:label: thm-native-stationary-reference-b2-budget

Use the complete register with count normalization, configured force
$F(x)=-\lambda x$, $0\le t\nu\le1$, and

$$
A=1-\eta\lambda>0,\qquad \alpha=1-t^2\lambda>0.
\tag{NS.8}
$$

Fix the complete actual post-collision input $(X,V^c)$, with
$|V_i^c|\le V_c$, retaining arbitrary positions and all selection marks.
Set $W_X=I-t\nu L_X$, $B=W_XV^c$, and use the actual stage arrays

$$
Y=AX+bB+tq\xi,\qquad Z=cB-ct\lambda X+q\xi.
\tag{NS.9}
$$

Their pair differences satisfy the exact identity

$$
Z_i-Z_j=\frac cA(B_i-B_j)
       -\frac{ct\lambda}{A}(Y_i-Y_j)
       +\frac{q\alpha}{A}(\xi_i-\xi_j).
\tag{NS.10}
$$

Define the primitive-parameter constants

$$
\begin{split}
S&=2\sqrt2 V_c\ell_K,\\
M_G&=\frac{2cV_c+ct\lambda\rho e^{-1/2}
                       +q\alpha\sqrt{2d}}{A},\\
H_G&=\frac{2cV_c\ell_K+(2/e)ct\lambda
                       +q\alpha\ell_K\sqrt{2d}}{A}.
\end{split}
\tag{NS.11}
$$

Conditional on the actual preparation, for every row and every $N$,

$$
\bigl(\mathbb E|G_i(Y,Z)|^2\bigr)^{1/2}\le\nu M_G.
\tag{NS.12}
$$

For a deterministic tangent $(\dot X,\dot V)$ to that input, let
$\dot B,\dot Y,\dot Z$ be the derivatives of (NS.9), using the same
uncapped innovation. Then

$$
\|\dot B\|_{2,N}\le\|\dot V\|_{2,N}+t\nu S\|\dot X\|_{2,N},
\qquad
\|\dot B-\dot V\|_{2,N}
\le t\nu[\|\dot V\|_{2,N}+S\|\dot X\|_{2,N}],
\tag{NS.13}
$$

and the *actual second-force derivative* satisfies

$$
\bigl(\mathbb E\|DG(Y,Z)[\dot Y,\dot Z]\|_{2,N}^2\bigr)^{1/2}
\le\nu[\|\dot Z\|_{2,N}+\sqrt2 H_G\|\dot Y\|_{2,N}].
\tag{NS.14}
$$

These constants are independent of population size and of prepared-position
maxima. The preparation can be correlated. They condition on its full actual
array, and the uncapped OU innovations remain in the stated expectations.
The final independent position innovation does not enter B2 and is retained
in the completed output.
:::

:::{prf:proof}
**1. Retain both force stages.** The actual first kick is
$V_1=W_XV^c-t\lambda X$. Its two position drifts and OU step give
(NS.9). Solving its position equation for $X$ and substituting in the
velocity equation yields
$Z=(c/A)B-(ct\lambda/A)Y+(q\alpha/A)\xi$, since
$A+ct^2\lambda=\alpha$. This proves (NS.10). The convex-kick bound
in Chapter 18 gives $|B_i|\le V_c$.

**2. Bound the force before taking an expectation.** The elementary
Gaussian-kernel maxima are

$$
\sup_u |u|K_\rho(u)=\rho e^{-1/2},\qquad
\sup_u|\nabla K_\rho(u)|=\ell_K,\qquad
\sup_u|u|\,|\nabla K_\rho(u)|=2/e.
$$

Multiply (NS.10) by $K_\rho(Y_i-Y_j)$. Its right side has norm at most

$$
\frac{2cV_c+ct\lambda\rho e^{-1/2}
                    +q\alpha|\xi_i-\xi_j|}{A}.
$$

The actual force is $\nu N^{-1}\sum_jK_{ij}(Z_j-Z_i)$.
Minkowski and $\mathbb E|\xi_i-\xi_j|^2\le2d$ give (NS.12),
uniformly in every input. This uses Gaussian decay of the actual weight,
rather than a maximum over the uncapped velocities.

**3. Bound the derivatives with the same innovations.** Differentiating
the first kick gives
$\dot B=W_X\dot V-t\nu DL_X[\dot X]V^c$.
For this count Laplacian,

$$
\|DL_X[\dot X]V^c\|_{2,N}^2
\le\frac{4V_c^2\ell_K^2}{N^2}
                 \sum_{i,j}|\dot X_i-\dot X_j|^2
\le8V_c^2\ell_K^2\|\dot X\|_{2,N}^2.
$$

Use $\|W_X\|\le1$, $\|L_X\|\le1$ and subtraction of $\dot V$ to
obtain (NS.13). For fixed input and tangent, $\dot B,\dot Y,\dot Z$
are deterministic: the added innovations in (NS.9) have zero derivative.
At B2 the derivative is

$$
DG=-\nu L_Y\dot Z-\nu DL_Y[\dot Y]Z.
$$

Multiplying (NS.10) by $|\nabla K_\rho(Y_i-Y_j)|$ gives a random
coefficient with $L^2$ norm at most $H_G$, uniformly in the pair and input.
Jensen on the rows then gives

$$
\mathbb E\|DL_Y[\dot Y]Z\|_{2,N}^2
\le\frac{H_G^2}{N^2}\sum_{i,j}|\dot Y_i-\dot Y_j|^2
\le2H_G^2\|\dot Y\|_{2,N}^2.
$$

The remaining Laplacian has norm at most one. Minkowski proves (NS.14).
No dependence between B2 positions, velocities and weights was discarded.
:::

:::{prf:theorem} Reference count kinetics: an explicit complete stability budget
:label: thm-native-stationary-reference-count-stability

Keep the preceding theorem's scope, $\lambda>0$, $q>0$, and choose
$r_g>0$. Let $\kappa_V$ be the actual-cap lemma's constant with
$\sigma_0=q\alpha$. Set

$$
C=t\lambda(c+A),\qquad D=c-\eta\lambda,\qquad
r^2=\sqrt{\kappa_V}C/b,
$$

$$
T=A^2+\kappa_VD^2+2\sqrt{\kappa_V}bC,\qquad
q_K=\left[\frac{T+\sqrt{T^2-4\kappa_Vc^2}}2\right]^{1/2}<1.
\tag{NS.15}
$$

With $S,M_G,H_G$ from (NS.11), define

$$
\begin{split}
B_U&=2|D|V_c+M_G,\\
E_x&=t\nu[|D|S+ct\lambda+ct\nu S
                   +\sqrt2 H_G(A+bt\nu S)],\\
E_v&=t\nu[|D|+c+\sqrt2 H_Gb],\\
e_x&=E_x+(4/V)t\nu B_U C,\qquad
e_v=E_v+(4/V)t\nu B_U|D|,\\
\mathsf E&=\begin{pmatrix}
bt\nu S&rbt\nu\\e_x/r&e_v
\end{pmatrix},\qquad q_\nu=q_K+\|\mathsf E\|_{2\to2}.
\end{split}
\tag{NS.16}
$$

For any two actual prepared arrays with row velocities bounded by $V_c$,
the count-normalized kinetic updates with equal full Gaussian innovations
obey

$$
\mathbb E\left[r^2\|X^+-\widetilde X^+\|_{2,N}^2
                   +\|V^+-\widetilde V^+\|_{2,N}^2\right]
\le q_\nu^2\left[r^2\|X-\widetilde X\|_{2,N}^2
                   +\|V^c-\widetilde V^c\|_{2,N}^2\right].
\tag{NS.17}
$$

The two-by-two norm is explicit: for
$T_E=\sum_{a,b}\mathsf E_{ab}^2$, $D_E=(\det\mathsf E)^2$,
$\|\mathsf E\|^2=(T_E+\sqrt{T_E^2-4D_E})/2$.
If the actual configuration satisfies $q_\nu<1$, it has a contracting
kinetic stage in this metric. This is a derived kinetic test, rather than a
stationary or active-cloning contraction premise.

For the unchanged count reference, with proof radius $r_g=0.2$ in its
declared units,

$$
\begin{gathered}
A\simeq0.9992156842,\quad \alpha=0.9996,
\quad q\simeq0.1960658736,\quad \kappa_V\simeq0.8626766707,\\
S\simeq6.862111080,\quad M_G\simeq8.184458920,
\quad H_G\simeq4.971199911,\\
q_K\simeq0.9999704926,\quad
\|\mathsf E\|\simeq0.2168682962,
\quad q_\nu\simeq1.216838789>1.
\end{gathered}
\tag{NS.18}
$$

Thus this unsigned stability bound is finite and population-independent,
but its contraction test does not certify the configured $\nu=0.3$.
The exact signed cancellation in (NS.2) and (NS.7) has not been included
in this upper norm estimate. A failed upper-bound test is not a proof that
the actual kinetics expands.
:::

:::{prf:proof}
**1. Bound the reference part within the actual-kernel comparison.** Write
the random count kinetic output, using precisely the same input and
innovations, as its zero-viscosity affine-force part plus its actual
viscous increment. This is a proof decomposition; the output retains
$\nu$. The former pre-cap derivatives are
$\dot Y_0=A\dot X+b\dot V$,
$\dot U_0=-C\dot X+D\dot V$. Its pre-cap row noise is
$q\alpha\xi_i$, so the cap lemma gives the mean-square matrix

$$
\begin{pmatrix}A&rb\\-\sqrt{\kappa_V}C/r&\sqrt{\kappa_V}D\end{pmatrix}.
$$

The chosen $r$ balances its off-diagonal magnitudes. Its squared singular
values have sum $T$ and product $\kappa_Vc^2$, proving (NS.15).
Writing $l=\eta\lambda\in(0,1)$, the contraction test follows from

$$
1+\kappa_Vc^2-T
=l(1-\sqrt{\kappa_V})
   [2(1-c\sqrt{\kappa_V})-l(1-\sqrt{\kappa_V})]>0
$$

and $\kappa_Vc^2<1$. The cap lemma can be applied along the segment
between inputs, since its affine pre-cap difference is deterministic.

**2. Differentiate the retained viscous increment.** At a fixed input,
$Y_\nu-Y_0=b(B-V)$ and
$U_\nu-U_0=D(B-V)+tG(Y,Z)$.
The first-kick difference has row norm at most $2t\nu V_c$.
Bound (NS.12) therefore gives a uniform row $L^2$ bound
$\|U_{\nu,i}-U_{0,i}\|_{L^2}\le t\nu B_U$.
Differentiating and applying (NS.13)--(NS.14) gives

$$
\|D(Y_\nu-Y_0)\|_{2,N}
\le bt\nu[S\|\dot X\|_{2,N}+\|\dot V\|_{2,N}],
$$

$$
\|D(U_\nu-U_0)\|_{L^2(\ell^2_N)}
\le E_x\|\dot X\|_{2,N}+E_v\|\dot V\|_{2,N}.
$$

For the actual cap, $\|DC_V(u)\|\le1$ and
$\|DC_V(u)-DC_V(v)\|\le4|u-v|/V$. To verify the latter, write
$DC_V=aI+a'|u|nn^*$, $a=V/(V+|u|)$, $n=u/|u|$.
The derivative terms have norm bounds $1/V$, $1/V$ and $2/V$,
respectively; continuity handles $u=0$. Integrate along a segment.
Thus

$$
D[C_V(U_\nu)-C_V(U_0)]
=DC_V(U_\nu)D(U_\nu-U_0)
 +[DC_V(U_\nu)-DC_V(U_0)]DU_0.
$$

Here $DU_0$ is deterministic at the fixed input. The just proved
uniform row moment bound makes the second term's $L^2$ norm at most
$(4/V)t\nu B_U[C\|\dot X\|_{2,N}+|D|\|\dot V\|_{2,N}]$.
After weighting the position and input coordinates, the resulting
increment derivative is bounded by the matrix $\mathsf E$.

**3. Integrate without changing the configured update.** Interpolate the
two prepared inputs along their straight segment. The velocity ball is
convex, so every intermediate preparation obeys the same $V_c$ bound.
The reference part's global mean-square bound is $q_K$, and the viscous
increment's derivative bound is $\|\mathsf E\|$. Minkowski and integration
along the segment give (NS.17). The final position innovations agree and
cancel in the discrepancy. The arithmetic evaluation in (NS.18) inserts
the unchanged values $h=.04$, $\gamma=1$, $\lambda=1$, $b_O=1$,
$V=2$, $\alpha_{\rm col}=.5$, $d=3$, $\nu=.3$, $\rho=1$ in the
displayed formulas. No configured term or noise distribution was changed.
:::

:::{prf:proposition} Exact row-normalized B2 score term
:label: prop-native-stationary-row-score-budget

For the actual row-normalized B2 arrays $(Y,Z)$ and their derivatives,
let $\bar Z_i=\sum_j\omega_{ij}(Y)Z_j$. In real arithmetic every
off-diagonal Gaussian row mass is positive for $N\ge2$. The exact derivative
is

$$
DG_i=\nu\left[\sum_j\omega_{ij}\dot Z_j-\dot Z_i\right]
-\frac{\nu}{\rho^2}\sum_j\omega_{ij}(Z_j-\bar Z_i)
         (Y_i-Y_j)\cdot(\dot Y_i-\dot Y_j).
\tag{NS.19}
$$

Consequently

$$
|DG_i|\le\nu\left|\sum_j\omega_{ij}\dot Z_j-\dot Z_i\right|
+\frac\nu{\rho^2}
 \left[\sum_j\omega_{ij}|Z_j-\bar Z_i|^2\right]^{1/2}
 \left[\sum_j\omega_{ij}|Y_i-Y_j|^2
                          |\dot Y_i-\dot Y_j|^2\right]^{1/2}.
\tag{NS.20}
$$

The force, derivative and degree terms retain their actual correlations.
The count-normalized constants (NS.11) do not transfer: division by the
Gaussian row mass removes the count proof's bounded-kernel factor. Chapter
19's proved localization/degree comparison can be used at its actual local
mass and moment values; it does not provide a global bounded row degree
after an uncapped Gaussian drift.
:::

:::{prf:proof}
Differentiate the normalized weights. Their derivative is
$\dot\omega_{ij}=\omega_{ij}(s_{ij}-\sum_k\omega_{ik}s_{ik})$, where
$s_{ij}=-(Y_i-Y_j)\cdot(\dot Y_i-\dot Y_j)/\rho^2$.
Substitute in the actual force
$G_i=\nu(\sum_j\omega_{ij}Z_j-Z_i)$; centering by $\bar Z_i$ gives
(NS.19). Weighted Cauchy--Schwarz gives (NS.20).
:::

(sec-native-stationary-discrete-entropy)=
## 4. An actual complete-step full-law entropy inequality

:::{prf:theorem} Complete-step modified logarithmic Sobolev inequality
:label: thm-native-stationary-discrete-mlsi

Let $K$ be an actual Markov kernel with an invariant law $\pi$ and a proved
minorization $K(z,\cdot)\ge e\theta(\cdot)$, $0<e<1$.
For every nonnegative $f$ with $0<\pi f<\infty$,

$$
\operatorname{Ent}_\pi(f)
\le\frac1e\int\pi(dz)K(z,dz')
 \left[f(z)\log\frac{f(z)}{f(z')}-f(z)+f(z')\right].
\tag{NS.21}
$$

The integrand uses the lower semicontinuous convention at zeros. This is
the actual complete-step logarithmic jump form, including every continuous
coordinate and discrete status transition used by $K$. It imposes no
unproved LSI or curvature premise.

For the actual terminal-box gas of Chapter 18 take its Doob transform
$\widehat P_N$, the invariant law
$\widehat\Pi_N=e_N^{\rm eig}\nu_N/\nu_N(e_N^{\rm eig})$, and its proved
minorization $e=\delta_N=\epsilon_N\theta_0(e_N^{\rm eig})/\alpha_N$.
The primitive eigenfunction theorem supplies its explicit lower bound in
its stated regime, including the unchanged count and row reference records.
This is a Doob-law inequality; the killed-chain QSD is not invariant under
its killed kernel. The coefficient is not proved uniform in population size.
:::

:::{prf:proof}
First take $f$ bounded above and below by positive constants and normalized
by $\pi f=1$. The probability $\mu=f\pi$ has relative entropy
$D(\mu\Vert\pi)=\operatorname{Ent}_\pi(f)$.
Write $K=e\theta+(1-e)R$, with $R$ Markov. Joint convexity of relative
entropy and Markov data processing give

$$
D(\mu K\Vert\pi K)
\le(1-e)D(\mu R\Vert\pi R)
\le(1-e)D(\mu\Vert\pi).
$$

Since $\pi K=\pi$, the density of $\mu K$ is $K^*f$.
Convexity of $a\log a$, with tangent at $f(z)$, gives

$$
e\operatorname{Ent}_\pi(f)
\le\operatorname{Ent}_\pi(f)-\operatorname{Ent}_\pi(K^*f)
\le\int(f-K^*f)\log f\,d\pi
=\int\pi(dz)K(z,dz')f(z)\log\frac{f(z)}{f(z')}.
$$

Stationarity makes the integrated $-f(z)+f(z')$ zero, proving (NS.21).
To extend the inequality, apply it to
$f_M=\min\{M,\max\{1/M,f\}\}$. For the scalar divergence
$D_{\log}(a,b)=a\log(a/b)-a+b$, monotone clipping to a common interval
decreases $D_{\log}$: for $a\ge b$ the representation
$D_{\log}(a,b)=\int_b^a(a-u)du/u$ decreases when either endpoint is
moved toward the other, and the case $a\le b$ follows from
$D_{\log}(a,b)=\int_a^b(u-a)du/u$.
Consequently the right side for $f_M$ is bounded by that for $f$.
The entropy of $f_M$ tends to that of $f$; the negative part of
$a\log a$ is bounded and the positive part converges by truncation.
Pass to the limit. Homogeneity removes the normalization.
The application uses the actual eigenfunction identity and Doob minorization
of Chapter 18; the stated law is invariant by direct integration against $\nu_NQ_N
=\alpha_N\nu_N$.
:::

:::{prf:proposition} The reference marked law actually requires a discrete status form
:label: prop-native-stationary-actual-status-obstruction

For $N\ge2$, the actual terminal-box gas in Chapter 18 with $s>0$ has a
QSD giving positive probability to every nonempty alive/dead mask. Its Doob
invariant law has the same mask support. Thus a continuous-gradient-only
LSI on the disjoint marked strata fails for these full marked laws: a
nonconstant mask function has zero continuous gradient and positive entropy.
The jump form in (NS.21) reads the real status transitions.
:::

:::{prf:proof}
Condition on the entire update before the independent final position noises.
Each row position is a finite center plus $s\zeta_i$. The terminal box and
its complement both have positive Gaussian probability, and the row noises
are independent. Hence each specified nonempty mask has strictly positive
conditional output probability from every nonextinct input. Integrating
against the QSD and using its eigenidentity gives positive QSD probability
of that mask. The eigenfunction is positive, so its reweighting has the same
support. A function depending only on the mask is constant on every
continuous stratum. The entropy decomposition
{prf:ref}`prop-kl-status-entropy` gives its positive discrete entropy while
its continuous gradient vanishes. This concerns the actual canonical marked
state space, not an added physical assignment of coordinates.
:::

(sec-native-stationary-residual)=
## 5. Exact residual obligations

:::{prf:remark} Discharge and remaining estimate register
:label: rem-native-stationary-residual

The actual count-normalized second-stage force and its derivatives admit
population-independent bounds from the configured quadratic force, Gaussian
bandwidth, OU amplitude, collision velocity cap and timestep. This improves
the kinetic error budget without replacing the reference force or imposing
joint-law assumptions. Its signed selection, terminal-status and cap
contributions still have to be joined to an adequate negative full-update
balance. The complete-step modified LSI is proved for the already constructed
Doob law, including its real discrete status transitions.

{prf:ref}`thm-ku-complete-signed-update` now supplies the complete signed
Keystone ledger from the same configured stages, including sampled fitness,
donor/revival flux, shared Haar collisions and both viscous kicks.
{prf:ref}`lem-ku-polynomial-row-defect` and
{prf:ref}`thm-ku-cubic-uniform-moments` additionally control row defects
and the configured cubic-force stage moments. The remaining issue is a
negative population-independent coefficient for the required whole-state
comparison, rather than an omitted finite-stage algebraic identity.

For the unchanged active-cloning terminal-box reference, the following
estimates are still not discharged:

1. A population-independent **negative complete signed feedback balance**
   joining the actual selection/preparation terms with (NS.7), the cap and
   terminal labels. Gaussian marginal tails do not control the mixed term
   in (NS.4) by the quadratic discrepancy; row normalization additionally
   retains the column-imbalance and changed-degree terms in (NS.3).
2. Concentration of the stationary empirical law at a single stationary
   population phase, with compatible survival selection. Finite-$N$ QSD
   mixing does not give this.
3. A population-independent functional inequality adequate for the intended
   continuous fluctuation tests **and** their actual discrete status
   coordinates. The derived modified LSI coefficient is $N$-dependent.
   Replacing it by a continuous-gradient-only form is false on the actual
   multiple-mask support.

The kinetic identities are valid for their full declared parameter scope.
The explicit count stability test above fails at the unchanged reference
value $\nu=0.3$; this is a failure of that unsigned bound, not a proof of
noncontraction. No critical-force or zero-selection surrogate is used to
claim completion of this active marked reference task.
:::

:::{prf:remark} Law-level stationary closure beyond the quadratic comparison
:label: rem-native-stationary-law-level-advance

{prf:ref}`thm-native-stationary-closure-spatial-poincare` now proves a
population-independent count-normalized spatial Poincare inequality by the
actual Gaussian stage map and a common-mass mixture estimate.
{prf:ref}`thm-native-stationary-closure-qsd-poincare` transfers it to the
QSD with an explicit exponentially vanishing defect and proves an exact
alive-conditional inequality. The full conditional spatial entropy form is
proved in {prf:ref}`thm-native-stationary-closure-marked-entropy`.
These constants do not supply the full joint marked inequality.

{prf:ref}`thm-native-stationary-closure-full-law-defect` gives the
complete QSD one-step and finite-path TV/KL defects, including terminal
marks. {prf:ref}`thm-native-stationary-closure-population-invariance`
constructs stationary nonlinear population laws and identifies every
stationary empirical subsequential limit as an invariant distribution
under the actual population map. Its fixed-row limits are the corresponding
mixtures of product laws. The remaining deterministic stationary-chaos
step is concentration of that invariant distribution at the intended phase;
invariance alone permits cycles. These are new law-level conclusions for
the actual count and row reference, without assuming the false global
fixed-slot quadratic contraction.
:::

:::{prf:remark} A derived active count regime now closes stationary phase concentration
:label: rem-native-stationary-phase-advance

{prf:ref}`thm-native-phase-contraction` derives a two-channel
transport contraction for the complete marked count population map.
Its coefficients retain sampled global fitness, mandatory revival,
every component Haar rotation, both viscous kicks, cap/OU correlations
and terminal marks. A target-local B2 density bound supplies the
small expected derivative of the actual smooth cap; no inverse
derivative or stationary-law property is assumed.

{prf:ref}`thm-native-phase-reset-regime` gives an explicit positive
interval of configured caps, and
{prf:ref}`cor-native-phase-positive-witness` evaluates a nonempty
active-cloning, positive-viscosity example.
{prf:ref}`thm-native-phase-stationary-chaos` proves its unique
population stationary phase and full marked QSD chaos.
The unchanged reference does not satisfy this sufficient certificate;
its separate phase and unconditional joint marked inequality remain
the reference obligations.
:::

(sec-native-stationary-uniform-doob-transfer)=
## 6. Primitive uniform Doob transfer in the proved active count phase

:::{prf:remark} Uniform eigenfunction transfer and the retained entropy scope
:label: rem-nsc-uniform-doob-transfer

For the actual positive active count regime of
{prf:ref}`def-nue-register`,
{prf:ref}`thm-nue-uniform-eigenfunction` now proves the right
eigenfunction ratio uniformly in population size from the original
kernel and its primitive parameters.
{prf:ref}`thm-nue-doob-comparison` compares its stationary Doob and
QSD laws and its complete attached survivor histories.
In particular {prf:ref}`cor-nue-doob-concentration-mixing` transfers
the full marked stationary variance and quantitative chaos estimates.
These results discharge this transfer in that proved phase.
The complete-step jump-form entropy inequality remains valid with
its displayed coefficient; a uniform per-update gradient
entropy-production estimate for other phases is not inferred from
the eigenfunction comparison.
:::
