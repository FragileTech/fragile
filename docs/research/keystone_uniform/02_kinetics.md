# Population-uniform kinetics in the signed Keystone balance

(sec-kuk-exact-map)=
## 1. The executed kinetic map and its signed paired increment

The inputs in this note are the actual copied, jittered and component-collided
arrays. Their law is the same finite-plan law used in (SCK.F4)--(SCK.F5) of
Chapter 06a. In particular, no independent-row assumption is made about these
arrays. The Rust implementation is
`algorithmic-gas/crates/algorithmic-gas/src/kinetic.rs`, `recorded_force` and the
`KineticKind::Baoab` branch. This note covers the declared terminal-boundary,
no-curl configuration; a substep boundary or graph Boris configuration has a
different map.

**Standing step-size condition.** Every estimate below that uses a convex
viscous kick, an $H_p$ moment factor, or a kinetic contraction assumes
$0\le t\nu\le1$. The exact algebraic identities (KUK.1)--(KUK.6)
hold without this condition. The reference instance satisfies it with
$t\nu=0.006$.

:::{prf:definition} Complete viscous kinetic stages
:label: def-kuk-stages

Put

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad\eta=bt,
\quad q^2=b_O^2\frac{1-e^{-2\gamma h}}{2\gamma},
\quad s^2=\sigma_x^2h,\quad\tau^2=t^2q^2+s^2,
$$

with the continuous value $q^2=b_O^2h$ at $\gamma=0$. Let
$K_{ij}(x)=\exp[-|x_i-x_j|^2/(2\rho^2)]$ for $i\ne j$,
and let $P_x$ have off-diagonal entries $K_{ij}/\sum_{k\ne i}K_{ik}$ and
zero diagonal. For $N=1$ all viscous forces are zero. For $N\ge2$, define

$$
A_x^{\rm count}=I-t\nu L_x,\quad
(L_xv)_i=N^{-1}\sum_{j\ne i}K_{ij}(v_i-v_j),\qquad
A_x^{\rm row}=(1-t\nu)I+t\nu P_x.
$$

The **actual** stages, including both fresh force evaluations, are

$$
\begin{aligned}
u&=A_xv+tF(x),&x_1&=x+tu,\\
w&=cu+q\xi,&y&=x_1+tw=x+bu+tq\xi,\\
z&=A_yw+tF(y),&x^+&=y+s\chi,\\
v_i^+&=C_V(z_i),&a_i^+&=\mathbf1_D(x_i^+),\qquad
C_V(z)=\frac{Vz}{V+|z|}.
\end{aligned}                                                    \tag{KUK.1}
$$

The arrays $\xi,\chi$ consist of independent standard $d$-Gaussians within
each swarm and are independent of preparation. Every row is eligible at
both kicks because mandatory revival occurs before kinetics. In exact real
arithmetic the row degree is positive for every finite $x$ when $N\ge2$;
the finite-precision zero-normalizer branch is not a statement about this
real-coordinate kernel.
:::

:::{prf:lemma} Exact signed paired two-kick identity
:label: lem-kuk-signed-increment

Couple the preparations with the Keystone source, jitter and component-Haar
coupling. Conditional on a realized paired preparation $(x,v),(\widetilde
x,\widetilde v)$, put $r=x-\widetilde x$, $\zeta=v-\widetilde v$,
and define the two **complete** acceleration discrepancies

$$
f=F(x)-F(\widetilde x)-\nu L_xv+\nu L_{\widetilde x}\widetilde v,
\quad
g=F(y)-F(\widetilde y)-\nu L_yw+\nu L_{\widetilde y}\widetilde w,
                                                               \tag{KUK.2}
$$

where in row mode $L_x=I-P_x$. For shared corresponding OU and final
position innovations, the actual uncapped differences are

$$
R=r+b\zeta+\eta f,\qquad
Z=c\zeta+ctf+tg.                                                \tag{KUK.3}
$$

Let $Q(r,\zeta)=\alpha|r|^2+2\beta r\cdot\zeta+
\gamma_P|\zeta|^2$, with $\alpha\gamma_P>\beta^2$, $\alpha>0$.
Its pre-cap increment in row $i$ is exactly

$$
\begin{aligned}
Q(R,Z)-Q(r,\zeta)={}&
2[\alpha b+\beta(c-1)]r\cdot\zeta
+[\alpha b^2+2\beta bc+\gamma_P(c^2-1)]|\zeta|^2\\
&+2[\alpha\eta+\beta ct]r\cdot f+2\beta t r\cdot g\\
&+2[\alpha b\eta+\beta(bct+c\eta)+\gamma_Pc^2t]\zeta\cdot f\\
&+2[\beta bt+\gamma_Pct]\zeta\cdot g\\
&+[\alpha\eta^2+2\beta\eta ct+\gamma_Pc^2t^2]|f|^2\\
&+2[\beta\eta t+\gamma_Pct^2]f\cdot g+
\gamma_Pt^2|g|^2.                                              \tag{KUK.4}
\end{aligned}
$$

The cap adds the exact signed term

$$
2\beta R\cdot(\Delta C-Z)+
\gamma_P(|\Delta C|^2-|Z|^2),\qquad
\Delta C=C_V(z)-C_V(\widetilde z).                              \tag{KUK.5}
$$

The complete averaged kinetic increment is $N^{-1}\sum_i$ of
(KUK.4)--(KUK.5), integrated against the **same** preparation and kinetic
law as the signed cloning ledger. For a general OU coupling put
$e=q(\xi-\widetilde\xi)$ and $j=s(\chi-\widetilde\chi)$; then

$$
R=r+b(\zeta+tf)+te+j,\qquad
Z=c(\zeta+tf)+e+tg.                                            \tag{KUK.6}
$$

This formula also covers independent OU draws. The force $g$ depends on
the realized OU draws, so their cross terms with $g$ remain inside the
Gaussian integral. Independent final position draws alone add exactly
$2\alpha ds^2$ to the expected physical quadratic; shared final draws add
zero. Terminal status uses the same completed positions.

*Proof.* Subtract each line of (KUK.1), retaining the actual first and
second viscous matrices. The first velocity difference is $\zeta+tf$;
the shared OU difference is $c(\zeta+tf)$; the two drifts give (KUK.3).
Expand its quadratic to obtain each displayed coefficient of (KUK.4).
Replacing $Z$ by the actual cap difference gives (KUK.5). The general
coupling subtraction gives (KUK.6). Final position noise is independent
of every quantity entering the velocity and cap, so its centered
cross terms vanish and its independent difference has second moment
$2ds^2$. No OU cross term with $g$ has been removed. $\square$
:::

(sec-kuk-gaussian-column)=
## 2. A Gaussian row-degree estimate without a degree floor

:::{prf:lemma} Dimension-only column mass of the self-excluded Gaussian kernel
:label: lem-kuk-gaussian-column

For every finite configuration in $\mathbb R^d$, every $N\ge2$, every
$\rho>0$, and every label $j$, the actual row probabilities satisfy

$$
\sum_{i\ne j}(P_x)_{ij}\le C_d,
$$

where the following constants are independent of $N$, the configuration,
and the bandwidth:

$$
\begin{aligned}
C_d&=2e^2+9^d\left[1+
\sum_{m=0}^{\infty}\left(
2e^{-(35/128)(5/4)^{2m}}+e^{-(3/8)(5/4)^m}\right)\right],\\
C_d&\le\overline C_d:=2e^2+9^d\left[1+
2e^{-35/128}\left(1+\frac{1}{(35/64)\log(5/4)}\right)
+e^{-3/8}\left(1+\frac{1}{(3/8)\log(5/4)}\right)\right].
                                                               \tag{KUK.7}
\end{aligned}
$$

Consequently, when $0\le t\nu\le1$, for every $p\ge1$,

$$
\|A_x^{\rm row}v\|_{p,N}\le
H_p\|v\|_{p,N},\qquad
H_p=[1+t\nu(\overline C_d-1)]^{1/p},
\quad \|A_x^{\rm row}v\|_{\infty,N}\le\|v\|_{\infty,N}.
                                                               \tag{KUK.8}
$$

For count normalization the corresponding $H_p$ equals one. In both
normalizations the first kick applied to bounded collision velocities
therefore preserves their rowwise bound. No uniform positive degree is
assumed, and positions produced by the OU draws remain unbounded.

*Proof.* Translate $x_j$ to zero and rescale by $\rho$. At most
$9^d$ balls of radius $1/4$ cover the unit sphere: a maximal
$1/4$-separated set has disjoint radius-$1/8$ balls contained in the
ball of radius $9/8$, and comparing volumes bounds its size by $9^d$.
Assign directions deterministically to these covering balls. Two
directions assigned to one ball differ by at most $1/2$, hence their
scalar product is at least $7/8$.

First collect the $n$ nonself points in the unit ball. If $n\ge2$,
each such row degree is at least $(n-1)e^{-2}$, so its total
contribution to column $j$ is at most $ne^2/(n-1)\le2e^2$.
For $n=1$ the contribution is at most one; for $n=0$ it is zero.

In each directional cell, partition radii into
$[(5/4)^m,(5/4)^{m+1})$, $m\ge0$. If a shell-cell contains
$n\ge2$ points, any two of its points have squared separation at
most
$[(1/4)^2+2(1-7/8)(5/4)^2](5/4)^{2m}
=(29/64)(5/4)^{2m}$.
Its target kernel is at most $e^{-(5/4)^{2m}/2}$ and its nonself
degree is at least $(n-1)e^{-(29/128)(5/4)^{2m}}$.
The column contribution of the whole cell is therefore at most
$2e^{-(35/128)(5/4)^{2m}}$.

A singleton in the first occupied shell-cell contributes at most
one. Every singleton in a later occupied shell-cell has a preceding
point in the same directional cell, with radius $r_k\ge1$ and
$r_k\le r_i$. Since their directional scalar product is at least
$7/8$,
$|x_i-x_k|^2-r_i^2\le r_k^2-(7/4)r_i r_k
\le-(3/4)r_i r_k$.
The kernel to that preceding point alone bounds the row denominator,
so $(P_x)_{ij}\le e^{-3r_i r_k/8}
\le e^{-(3/8)(5/4)^m}$.
Sum the dense and singleton bounds over all shell-cells and then
over directional cells. This proves the first line of (KUK.7).
For a decreasing function, its nonnegative integer sum is at most
its first value plus its integral. The substitution $u=a(5/4)^{kx}$
and $\int_a^\infty e^{-u}du/u\le e^{-a}/a$ give its second line.

The row sums of $A_x^{\rm row}$ equal one and its column sums are
at most $1+t\nu(\overline C_d-1)$. Jensen's inequality in each row,
followed by summation over columns, proves (KUK.8). The infinity
bound uses row sums only. Count normalization is doubly stochastic
and gives $H_p=1$. $\square$
:::

(sec-kuk-uncapped-moments)=
## 3. All uncapped stages and Gaussian excursions

:::{prf:theorem} Primitive population-uniform moment budget
:label: thm-kuk-moments

Suppose living input positions are in a declared region $D$ with
$R_D=\sup_{x\in D}|x|<\infty$, every retained input velocity has norm
at most $V$, and the configured force satisfies
$|F(x)|\le B_F+L_F|x|$. All constants refer to this actual force.
After mandatory revival, source copying, jitter and collision, set
$V_c=(1+2|\alpha_{\rm col}|)V$ and

$$
g_{d,p}=\mathbb E|\xi|^p=
2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)},\qquad
X_p=R_D+\sigma_Jg_{d,p}^{1/p}.
$$

For even $p=2m$ one may use the sharper, completely evaluated
$X_{2m}^{2m}=\sum_{k=0}^m\binom mk R_D^{2k}
\sigma_J^{2(m-k)}2^{m-k}\Gamma(d/2+m)/\Gamma(d/2+k)$.
This is the noncentral Gaussian moment at donor radius $R_D$; rows
without accepted jitter only reduce it. Define

$$
U_p=V_c+t(B_F+L_FX_p),\quad
W_p=cU_p+qg_{d,p}^{1/p},\quad
Y_p=X_p+bU_p+tqg_{d,p}^{1/p},\quad
Z_p=H_pW_p+t(B_F+L_FY_p).                                     \tag{KUK.9}
$$

Then, for either actual normalization,

$$
\begin{gathered}
(\mathbb E\|x\|_{p,N}^p)^{1/p}\le X_p,\quad
(\mathbb E\|u\|_{p,N}^p)^{1/p}\le U_p,\quad
(\mathbb E\|w\|_{p,N}^p)^{1/p}\le W_p,\\
(\mathbb E\|y\|_{p,N}^p)^{1/p}\le Y_p,\quad
(\mathbb E\|z\|_{p,N}^p)^{1/p}\le Z_p,\quad
(\mathbb E\|x^+\|_{p,N}^p)^{1/p}\le Y_p+sg_{d,p}^{1/p},\quad
\|v^+\|_{\infty,N}\le V.
                                                               \tag{KUK.10}
\end{gathered}
$$

For $p=2$, Gaussian centering improves the $w,y$ bounds to
$W_2^2=c^2U_2^2+dq^2$ and
$Y_2^2=(X_2+bU_2)^2+dt^2q^2$; final position noise adds
$ds^2$ exactly. These bounds apply to the **uncapped** second
kick, with its actual innovation-dependent matrix, and all constants
are independent of $N$.

For the quadratic force $F(x)=-kx$, write $a_k=1-\eta k$.
For every analysis threshold $R>0$,

$$
\frac1N\sum_i\Pr(|y_i|>R)\le\min\{1,Y_p^p/R^p\},\quad
\frac1N\sum_i\Pr(|z_i|>R)\le\min\{1,Z_p^p/R^p\}.
                                                               \tag{KUK.11}
$$

Weighted cap or regional-force tails are also explicitly controlled.
For any paired actual preparations and any $R>0$,

$$
\begin{aligned}
\frac1N\sum_i\mathbb E[
\mathbf1_{\{|z_i|\vee|\widetilde z_i|>R\}}
(|y_i-\widetilde y_i|^2+|z_i-\widetilde z_i|^2)]
\le \frac{2}{R^2}
(16Y_4^4+16Z_4^4)^{1/2}Z_4^2 .                              \tag{KUK.12}
\end{aligned}
$$

Both marginals use the same primitive moment bounds. Analysis thresholds
in (KUK.11)--(KUK.12) classify an excursion; they do not clip any noise
or force. The full Gaussian law appears in (KUK.2)--(KUK.6).

*Proof.* Conditional on the source plan, every position is a living
source coordinate plus its declared centered Gaussian jitter (or no
jitter), while component collisions give the pointwise bound $V_c$.
Gaussian polar integration gives $g_{d,p}$; differentiating the
noncentral Gaussian moment-generating function gives the displayed
finite even-moment polynomial. Minkowski's inequality on the product
of the normalized counting measure and the full innovation law gives
the $x,u,w,y$ estimates in sequence. The second-kick matrix depends
on $y$, but (KUK.8) holds **pathwise**, so the same Minkowski argument
gives $Z_p$. For count normalization use its contraction instead.
The independent centered OU and final position Gaussians give the
sharper second-moment identities. The original cap gives the last
bound. Averaged Markov's inequality proves (KUK.11).

For (KUK.12), under the normalized row-law let $T$ be its event and
$B=|y-\widetilde y|^2+|z-\widetilde z|^2$. Then
$\Pr(T)\le2Z_4^4/R^4$. The inequalities
$(a+b)^2\le2a^2+2b^2$ and
$|u-\widetilde u|^4\le8(|u|^4+|\widetilde u|^4)$ give
$\mathbb EB^2\le32Y_4^4+32Z_4^4$.
Cauchy--Schwarz therefore gives the explicit
bound $8Z_4^2\sqrt{Y_4^4+Z_4^4}/R^2$; this equals the right side
of (KUK.12). $\square$
:::

:::{prf:corollary} Population-uniform exponential position moments
:label: cor-kuk-exponential-moments

For the same quadratic force, define
$\sigma_*^2=a_k^2\sigma_J^2+t^2q^2$. For any $\lambda>0$,

$$
\frac1N\sum_i\mathbb E e^{\lambda|x_i|^2}
\le(1-2\lambda\sigma_J^2)^{-d/2}
\exp\!\left[\frac{\lambda R_D^2}{1-2\lambda\sigma_J^2}\right]
                                                               \tag{KUK.13}
$$

when $2\lambda\sigma_J^2<1$. For any $\epsilon>0$ with
$2\lambda(1+\epsilon)\sigma_*^2<1$,

$$
\frac1N\sum_i\mathbb E e^{\lambda|y_i|^2}
\le e^{\lambda(1+1/\epsilon)b^2V_c^2}
[1-2\lambda(1+\epsilon)\sigma_*^2]^{-d/2}
\exp\!\left[
\frac{\lambda(1+\epsilon)a_k^2R_D^2}
{1-2\lambda(1+\epsilon)\sigma_*^2}\right].                    \tag{KUK.14}
$$

These estimates are valid for both normalizations even though $A_xv$
depends on the jitter. For every configuration, Jensen's inequality gives
the actual nonself degree lower bound

$$
\frac{d_i(x)}{N-1}\ge
\exp\![-(|x_i|^2+2\|x\|_{2,N}^2)/\rho^2].                     \tag{KUK.15}
$$

It is a row-dependent quantitative estimate, rather than a positive
constant imposed on all rows. In particular

$$
\frac1N\sum_i e^{4(|x_i|^2+2\|x\|_{2,N}^2)/\rho^2}
\le\frac1N\sum_i e^{12|x_i|^2/\rho^2},                         \tag{KUK.16}
$$

and the analogous inequality holds at $y$. Thus the averaged inverse
degree quantities needed for regional row estimates have explicit
$N$-independent moments when (KUK.13)--(KUK.14) pass at the declared
exponent. No maximum Gaussian over all $N$ rows occurs.

*Proof.* Condition on each actual source coordinate and its jitter
indicator. The Gaussian exponential integral is (KUK.13). The exact
quadratic position is
$y_i=a_k x_{{\rm source},i}+a_k\sigma_J I_i\zeta_i+tq\xi_i
+b(A_xv)_i$.
The first three terms have a Gaussian law with mean bounded by
$|a_k|R_D$ and variance at most $\sigma_*^2$, and the last term has
norm at most $bV_c$. Apply
$|u+w|^2\le(1+\epsilon)|u|^2+(1+1/\epsilon)|w|^2$
pointwise, then integrate the Gaussian. This proves (KUK.14) without
assuming its bounded offset is independent of jitter.
For (KUK.15), Jensen gives
$d_i/(N-1)\ge\exp[-(N-1)^{-1}\sum_{j\ne i}|x_i-x_j|^2/(2\rho^2)]$.
Use $|x_i-x_j|^2\le2|x_i|^2+2|x_j|^2$ and
$N/(N-1)\le2$. Finally Hölder and Jensen imply
$e^{8\|x\|_{2,N}^2/\rho^2}\le
(N^{-1}\sum_i e^{12|x_i|^2/\rho^2})^{2/3}$ and
$N^{-1}\sum_i e^{4|x_i|^2/\rho^2}\le
(N^{-1}\sum_i e^{12|x_i|^2/\rho^2})^{1/3}$;
multiplying proves (KUK.16). $\square$
:::

(sec-kuk-signed-variance)=
## 4. A substantive signed Keystone variance estimate for both normalizations

:::{prf:lemma} The second count-normalized kick uses mixed fourth moments
:label: lem-kuk-paired-count-force

Assume the actual configured force is globally $L_F$-Lipschitz and the
two entering collision arrays have row speeds at most $V_c$. Put
$\ell_\rho=e^{-1/2}/\rho$, $L_0=L_F+4\nu V_c\ell_\rho$ and
$r=x-\widetilde x$, $\zeta=v-\widetilde v$. In count mode the actual
first-kick and shared-OU positional differences satisfy, for $p=2,4$,

$$
\|\Delta u\|_{p,N}\le\|\zeta\|_{p,N}+tL_0\|r\|_{p,N},\qquad
\|R\|_{p,N}\le(1+\eta L_0)\|r\|_{p,N}+b\|\zeta\|_{p,N}.
                                                               \tag{KUK.22}
$$

Conditional on the actual paired preparations, $R$ and $\Delta u$
are deterministic even though both intermediate positions are
unbounded Gaussian arrays. Define the **computed conditional** fourth
moment of the second swarm's actual OU velocity by

$$
\mathcal W_4^4=\frac1N\sum_i\left[
c^4|\widetilde u_i|^4+2(d+2)c^2q^2|\widetilde u_i|^2
+d(d+2)q^4\right].                                            \tag{KUK.23}
$$

Then its complete second-kick and original cap differences obey

$$
\left(\mathbb E[\|Z\|_{2,N}^2\mid\text{preparation}]\right)^{1/2}
\le c\|\Delta u\|_{2,N}+tL_F\|R\|_{2,N}
+4t\nu\ell_\rho\|R\|_{4,N}\mathcal W_4,\qquad
\|\Delta C\|_{2,N}\le\|Z\|_{2,N}.                             \tag{KUK.24}
$$

For random preparations the final mixed term is bounded, again with
explicit $N$-independent coefficients, by

$$
\left(\mathbb E[\|R\|_{4,N}^2\mathcal W_4^2]\right)^{1/2}
\le(\mathbb E\|R\|_{4,N}^4)^{1/4}
(\mathbb E\mathcal W_4^4)^{1/4}.
                                                               \tag{KUK.25}
$$

The first factor is computed from the signed preparation law and
(KUK.22); the second from (KUK.23), bounded by the primitive $W_4$
of (KUK.9). No maximum over Gaussian velocities is taken.

*Proof.* Split
$\Delta u=A_x(v-\widetilde v)+t[F(x)-F(\widetilde x)]
-t\nu(L_x-L_{\widetilde x})\widetilde v$.
The count matrix contracts every normalized $p$ norm. The Gaussian
gradient bound is
$|\Delta K_{ij}|\le\ell_\rho(|r_i|+|r_j|)$; with bounded
$\widetilde v$, the last term has $p$ norm at most
$4t\nu V_c\ell_\rho\|r\|_{p,N}$. This proves (KUK.22).
At the second kick,
$Z=A_y(c\Delta u)+t[F(y)-F(\widetilde y)]
-t\nu(L_y-L_{\widetilde y})\widetilde w$.
For arbitrary $w$ and displacement $R$, direct row summation gives

$$
\|(L_y-L_{\widetilde y})w\|_{2,N}
\le\ell_\rho\left[
\|Rw\|_{2,N}+\|R\|_{2,N}\|w\|_{1,N}
+\|R\|_{1,N}\|w\|_{2,N}+\|Rw\|_{1,N}\right]
\le4\ell_\rho\|R\|_{4,N}\|w\|_{4,N}.
$$

The conditional OU fourth moment is exactly (KUK.23), and Jensen
gives $\mathbb E\|\widetilde w\|_{4,N}^2\le\mathcal W_4^2$.
Conditional Minkowski proves (KUK.24). The cap Jacobian has radial
eigenvalue $V^2/(V+|z|)^2$ and tangential eigenvalue $V/(V+|z|)$,
both at most one, which gives its nonexpansiveness. Cauchy--Schwarz
over preparation gives (KUK.25). $\square$
:::

:::{prf:lemma} Explicit averaged row-normalization force defect
:label: lem-kuk-paired-row-force

For a configuration $x$, define its actual inverse-degree envelope
$E_i(x)=e^{(|x_i|^2+2\|x\|_{2,N}^2)/\rho^2}$ and
$\mathcal E_4(x)=\|E(x)\|_{4,N}$.
For bounded first-kick velocities $|\widetilde v_i|\le V_c$, the
actual row-normalized first discrepancy has the global estimate

$$
\|\Delta u\|_{2,N}\le
H_2\|\zeta\|_{2,N}+tL_F\|r\|_{2,N}
+6t\nu V_c\ell_\rho\mathcal E_4(x)\|r\|_{4,N}.              \tag{KUK.26}
$$

At B2, for arbitrary actual OU velocities, the normalized weight
defect satisfies the deterministic, averaged estimate

$$
\|(P_y-P_{\widetilde y})\widetilde w\|_{2,N}
\le2\ell_\rho(3+2\overline C_d^{1/8})
\mathcal E_4(y)\|R\|_{8,N}\|\widetilde w\|_{8,N}.             \tag{KUK.27}
$$

Consequently, conditional on paired preparation and shared OU,

$$
\begin{aligned}
\bigl(\mathbb E\|Z\|_{2,N}^2\bigr)^{1/2}
\le{}&cH_2\|\Delta u\|_{2,N}+tL_F\|R\|_{2,N}\\
&+2t\nu\ell_\rho(3+2\overline C_d^{1/8})\|R\|_{8,N}
\left(\frac1N\sum_i\mathbb E e^{12|y_i|^2/\rho^2}\right)^{1/4}
\left(\frac1N\sum_i\mathbb E|\widetilde w_i|^8\right)^{1/8}.
                                                               \tag{KUK.28}
\end{aligned}
$$

All conditional expectations in this statement use the actual Gaussian
law, so the exponential factor can be integrated exactly conditionally
on preparation, then averaged by (KUK.14). Its primitive integrability
test was evaluated in the reference example. The estimate explicitly
shows the weighted moment budget used by this row-normalization estimate;
it does not pretend that the row force is globally Lipschitz. This
inverse-degree estimate is an alternative certificate: the polynomial
normalized derivative calculation in
{prf:ref}`lem-ku-polynomial-row-defect` removes the exponential weight
and uses the actual sixth moments instead.

*Proof.* For a raw row kernel $k,\widetilde k$ with sums
$d,\widetilde d$, the exact normalization subtraction is
$p_j-\widetilde p_j=(k_j-\widetilde k_j)/d
-\widetilde p_j(d-\widetilde d)/d$.
Use (KUK.15) and $N/(N-1)\le2$. Bounded velocities give
$|(P_x-P_{\widetilde x})\widetilde v|_i
\le2V_c\ell_\rho E_i(x)(|r_i|+2\|r\|_{1,N})$.
Hölder in the row measure gives the last term of (KUK.26), while
the first matrix has norm at most $H_2$.

For unbounded velocities the same exact subtraction gives, writing
$M_w=\|\widetilde w\|_{1,N}$ and $M_R=\|R\|_{1,N}$,

$$
|[(P_y-P_{\widetilde y})\widetilde w]_i|
\le2\ell_\rho E_i(y)\left[
(|R_i|+M_R)(M_w+(P_{\widetilde y}|\widetilde w|)_i)
+\|R\widetilde w\|_{1,N}\right].
$$

Hölder uses exponents $4,8,8$, and (KUK.7) bounds
$\|P_{\widetilde y}|\widetilde w|\|_{8,N}
\le\overline C_d^{1/8}\|\widetilde w\|_{8,N}$.
This yields (KUK.27). In conditional expectation,
$\mathbb E\mathcal E_4(y)^4$ is bounded by the exponential average
in (KUK.16); Cauchy--Schwarz and Jensen give
$\mathbb E\|\widetilde w\|_{8,N}^4
\le(N^{-1}\sum_i\mathbb E|\widetilde w_i|^8)^{1/2}$.
Since $R$ is deterministic conditional on preparation, conditional
Minkowski proves (KUK.28). $\square$
:::

:::{prf:theorem} The existing signed positional drift survives both viscous kicks
:label: thm-kuk-signed-keystone-variance

Write $W_C=\operatorname{Var}_N(x)$ for the actual prepared positional
variance and $\widehat v=A_xv$. The first viscous matrix is
row-stochastic under $t\nu\le1$, hence $|\widehat v_i|\le V_c$.
Suppose the **actual prepared pair law** obeys

$$
(x-y)\cdot(F(x)-F(y))\le-m|x-y|^2+b_F,\qquad
|F(x)-F(y)|\le L|x-y|+J.
$$

Pointwise exceptions may instead be retained in the corresponding
positive-excess pair integrals, with their actual preparation law and
(KUK.11)--(KUK.14) excursion bounds. Then the constants

$$
\begin{aligned}
A_0&=-2\eta m+\eta^2L^2,\\
C_1&=2bV_c+2b\eta V_cL+\sqrt2\eta^2LJ,\\
C_0&=\eta b_F+b^2V_c^2+\sqrt2b\eta V_cJ+\eta^2J^2/2,\\
a_\delta&=[1+A_0+\delta]_+,\qquad
d_\delta=C_0+C_1^2/(4\delta)+d\tau^2
\end{aligned}                                                    \tag{KUK.17}
$$

give $\mathbb EW^+\le a_\delta\mathbb EW_C+d_\delta$ for
each $\delta>0$. Substituting the **signed** Keystone cloning balance
therefore gives, with its unchanged primitive $k_{\rm key},p,E_{\max}$,

$$
\mathbb EW^+\le a_\delta\left[
W-\theta k_{\rm key}W^p+
\mathbb E\Gamma_\theta+d\sigma_J^2\mathbb E\bar p+
\frac{\theta E_{\max}}{N^2}\right]+d_\delta.                  \tag{KUK.18}
$$

The donor flux, barycenter subtraction, jitter, and finite-population
correction retain their Chapter 06a definitions and signs. Both B kicks
are the actual map (KUK.1). The second kick and cap leave positions
unchanged; their moments and their effect on the next step are covered
by (KUK.9)--(KUK.16). No $N$ factor is hidden in (KUK.17).

For the configured quadratic force $F(x)=-kx$, there is the sharper
**exact** finite-array identity

$$
\mathbb E[W^+\mid x,v]-W_C
=(a_k^2-1)W_C+b^2\operatorname{Var}_N(\widehat v)
+2a_kb\operatorname{Cov}_N(x,\widehat v)
+(1-N^{-1})d\tau^2.                                           \tag{KUK.19}
$$

For $0<a_k<1$, the phase test
$\operatorname{Cov}_N(x,\widehat v)\le0$ is an explicitly computable
signed finite-array condition. If its actual velocity variance is at
most $J_v$, then

$$
\mathbb E[W^+\mid x,v]-W_C\le
-(1-a_k^2)W_C+b^2J_v+d\tau^2.                                 \tag{KUK.20}
$$

This is strictly negative by at least $(1-a_k^2)W_C/2$ when
$W_C\ge2(b^2J_v+d\tau^2)/(1-a_k^2)$ and $W_C>0$.
For count normalization, component restitution $|\alpha_{\rm col}|\le1$
and input cap $V$ give $J_v=V^2$: collisions contract the total
velocity square, and the first viscous matrix preserves the velocity
barycenter and contracts its centered square. For row normalization,
the universally valid choice is $J_v=V_c^2$; a smaller actual
finite-array variance in (KUK.19) may be used directly.

Without a signed covariance restriction, let
$A=1-a_k^2>0$, $C=2a_kb\sqrt{J_v}$,
$B=b^2J_v+d\tau^2$. Then the completely primitive bound is

$$
\mathbb E[W^+\mid x,v]-W_C\le-AW_C+C\sqrt{W_C}+B.
                                                               \tag{KUK.20a}
$$

Its optimal fixed Young floor is obtained explicitly:

$$
y_*=\frac{C+\sqrt{C^2+4AB}}{2A},\quad
W_*=y_*^2,\quad\delta_*=\frac{C}{2y_*},\quad
\mathbb EW^+\le(1-A+\delta_*)W_C+B+\frac{C^2}{4\delta_*}.
                                                               \tag{KUK.20b}
$$

The displayed affine recursion has fixed point $W_*$ and
$0<\delta_*<A$ when $C,B>0$. This is the minimum fixed point
among all Young parameters $0<\delta<A$. When $C=0$ its floor
is $B/A$. These formulas apply to count and row normalization
with their actual $J_v$.

*Proof.* From (KUK.1),
$x^+=x+b\widehat v+\eta F(x)+tq\xi+s\chi$.
Expand its empirical centered square; the independent Gaussian terms
add exactly $(1-N^{-1})d\tau^2$. Prepared pair integration gives
$\operatorname{Cov}_N(x,F(x))\le-mW_C+b_F/2$ and
$\sqrt{\operatorname{Var}_N(F(x))}\le L\sqrt{W_C}+J/\sqrt2$.
Apply Cauchy--Schwarz to the remaining velocity cross terms using
$\operatorname{Var}_N(\widehat v)\le V_c^2$. Their expansion is
exactly $A_0W_C+C_1\sqrt{W_C}+C_0$.
Young's inequality gives
$C_1\sqrt{W_C}\le\delta W_C+C_1^2/(4\delta)$.
This proves (KUK.17); substitution of the existing signed cloning
inequality proves (KUK.18). For a quadratic force the deterministic
position is exactly $a_kx+b\widehat v$, yielding (KUK.19) by
direct expansion. Its signed covariance premise and stated velocity
budget give (KUK.20). The threshold follows by comparing the positive
noise term to half the negative term. Count-normalized velocity
contraction and component orthogonality give its stated $J_v$.
$C\sqrt{W_C}\le\delta W_C+C^2/(4\delta)$ proves (KUK.20b).
The fixed point is $(B+C^2/(4\delta))/(A-\delta)$; differentiating
it shows that its minimum obeys
$Ay_*^2=Cy_*+B$ and $\delta_*=C/(2y_*)$, giving the stated root.
$\square$
:::

:::{prf:lemma} Terminal status cost from the actual final noise
:label: lem-kuk-terminal-marks

For a box $D=\prod_{r=1}^d[\ell_r,u_r]$ and $s>0$, conditional
on the actual A2 position $y_i$, its survival probability is exactly

$$
\theta_i=\prod_{r=1}^d\left[
\Phi((u_r-y_{i,r})/s)-\Phi((\ell_r-y_{i,r})/s)\right].
$$

For shared final position noise and paired A2 positions with difference
$R_i$, the mean terminal mismatch satisfies

$$
\frac1N\sum_i\Pr(a_i^+\ne\widetilde a_i^+)
\le\frac1N\sum_i\mathbb E\min\left\{1,
\frac{\sqrt{2/\pi}}s\sum_{r=1}^d|R_{i,r}|\right\}.
                                                               \tag{KUK.21}
$$

For independent final draws its conditional value is instead
$\theta_i+\widetilde\theta_i-2\theta_i\widetilde\theta_i$.
Conditioning on all pre-final-noise marks, complete extinction has
probability $\prod_i(1-\theta_i)$. This exact product is a terminal
accounting identity; it is not used as a population-uniform
minorization probability.

*Proof.* The final position coordinates have the declared independent
Gaussian law. Integrate its density over each interval to obtain
$\theta_i$. The symmetric difference of two intervals translated by
$R_{i,r}$ has length at most $2|R_{i,r}|$, and the Gaussian density
is at most $1/(s\sqrt{2\pi})$. Sum coordinate mismatches and truncate
at one to obtain (KUK.21). Conditional independence of separate final
innovations gives the remaining two identities. $\square$
:::

:::{prf:lemma} Averaged single-row landing probability with unbounded preparation
:label: lem-kuk-landing

Let $B_r(z_H)\subset D$ be a fixed physical target ball, $\tau>0$,
and let the actual conditional completed-position mean be
$M_i=x_i+bu_i$. For any $L>0$, the true transition obeys

$$
\frac1N\sum_i\Pr(x_i^+\in B_r(z_H))\ge
\left[1-\frac{(X_2+bU_2)^2}{L^2}\right]_+
\frac{\pi^{d/2}r^d}{\Gamma(d/2+1)(2\pi\tau^2)^{d/2}}
\exp\!\left[-\frac{(L+|z_H|+r)^2}{2\tau^2}\right].          \tag{KUK.29}
$$

Every quantity on the right is a primitive parameter or the stated target
ball. This is a sampled-row landing estimate, retaining its normalized
probability, and does not impose simultaneous landing of all walkers.

*Proof.* Given preparation, $x_i^+=M_i+tq\xi_i+s\chi_i$ is
a Gaussian with covariance $\tau^2I_d$; B2 and the cap do not alter
this position. On $|M_i|\le L$, every point of the target ball is
within $L+|z_H|+r$ of $M_i$, so its Gaussian density is at least
the stated infimum. Integrate over the exact ball volume. Averaged
Markov's inequality and
$\mathbb E\|M\|_{2,N}^2\le(X_2+bU_2)^2$ give the leading
mass factor after integration over the actual preparation law.
$\square$
:::

(sec-kuk-reference)=
## 5. Original reference parameters, nonempty regime, and limits

:::{prf:example} Evaluated unchanged viscous reference
:label: ex-kuk-reference

For $d=3$, $h=0.04$, $\gamma=b_O=\rho=1$, $\nu=0.3$,
$\sigma_J=\sigma_x=0.1$, $V=2$, $\alpha_{\rm col}=0.5$,
$D=[-2,2]^3$ and $F(x)=-x$, the computed values are

$$
\begin{gathered}
t=0.02,\quad t\nu=0.006,\quad c=0.9607894391523232,\\
b=0.03921578878304646,\quad\eta=0.0007843157756609293,\\
q^2=0.03844182680668212,\quad s^2=0.0004,\quad
\tau^2=0.00041537673072267287,\\
R_D=2\sqrt3,\quad V_c=4,\quad
\overline C_3<17431,\quad H_p\le105.58^{1/p}.
\end{gathered}
$$

Using the simpler Minkowski input $X_p$ in (KUK.9), valid conservative
moment bounds (rounded upward) are

| $p$ | $X_p$ | $U_p$ | $W_p$ | $Y_p$ | $Z_p$, count | $Z_p$, row |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.638 | 4.073 | 4.253 | 3.804 | 4.330 | 43.775 |
| 4 | 3.661 | 4.074 | 4.300 | 3.829 | 4.377 | 13.859 |
| 8 | 3.700 | 4.074 | 4.376 | 3.869 | 4.454 | 7.924 |
| 16 | 3.760 | 4.076 | 4.496 | 3.932 | 4.575 | 6.096 |

The sharper centered second-moment formulas may replace the deliberately
simple $p=2$ entries. Here $a_1=1-\eta$, and
$\sigma_*^2=0.009999696566721812$.
At the explicit inverse-degree exponent $\lambda=12/\rho^2=12$,
both (KUK.13) and (KUK.14) with $\epsilon=1$ pass their Gaussian
integrability tests:
$2\lambda\sigma_J^2=0.24<1$ and
$4\lambda\sigma_*^2<0.48<1$.
Thus the row-normalization estimates include the full unbounded first
jitter and OU excursion. The terminal multiplier is
$\sqrt{2/\pi}/s=39.89422804014327$, independent of $N$.

For the general pair-profile substitution (KUK.17), $m=L=1$,
$b_F=J=0$ give
$A_0=-0.001568016400085908$, $C_1=0.3139723707587519$ and
$C_0=0.024606049438024205$.
Choosing $\delta=-A_0/2$ yields
$a_\delta=0.9992159917999571$ and
$d_\delta=31.460041769251276$. This coarse Young bound is finite
and population-uniform but too pessimistic to prove drift throughout
the bounded reference domain. It must not be presented as a passing
global contraction certificate.

The exact signed test (KUK.19) supplies a nonempty quantitative regime.
For count normalization, $J_v=V^2=4$ and
$\operatorname{Cov}_N(x,A_xv)\le0$ give

$$
1-a_1^2=0.001568016400085908,\qquad
b^2J_v+d\tau^2<0.007398,
$$

so strict kinetic variance reduction by at least
$0.0007840082\,W_C$ occurs whenever $W_C\ge9.436$.
Such variance is attainable inside the stated box. For row normalization
the same numerical regime follows whenever the actual array has
$\operatorname{Var}_N(A_xv)\le4$ and the same signed covariance.
One primitive sufficient row bound is $\max_i|v_i|\le2$ **after
collision**. This phase is nonempty: for zero frozen input velocities,
all component collisions and the first viscous kick preserve zero,
and $J_v=0$ under either normalization. In that case the threshold
improves to $2d\tau^2/(1-a_1^2)<1.590$ while the thermostat and
second kick remain unchanged and unbounded. These are conditional
one-step phase statements; Gaussian innovations can leave that phase.
The signed cloning flux in (KUK.18) is retained in addition to this
kinetic calculation.

Without using the sign of the actual covariance, (KUK.20b) gives
the optimal primitive count floor $W_*<10001.589$, with
$\delta_*=0.0007836383766710447$ and affine coefficient
$0.9992156219765851$. The universal row floor is
$W_*<40001.589$, with $\delta_*=0.0007836850606326733$ and
affine coefficient $0.9992156686605467$.
Both floors exceed the box's maximum living positional variance $12$;
this particular unconditional bound is therefore vacuous in that domain.
The exact covariance and variance in (KUK.19), together with the signed
Keystone flux, supply the useful improvement. They are actual computed
arrays, rather than assumed independent or discarded velocity data.

For the reference count B2 paired estimate,
$\ell_\rho=e^{-1/2}$ and $L_0=1+4(0.3)(4)e^{-1/2}$.
Thus (KUK.22)--(KUK.24) use the unchanged thermostat's noncentral
Gaussian fourth moment with coefficient $4t\nu\ell_\rho$
$=0.014556735833103202$.
The row estimate uses the displayed eighth moment and inverse-degree
exponential instead of an invalid velocity maximum.
Their computed reference values are $L_0=3.9113471666206405$,
$\eta L_0=0.003067731286867246$, and the logarithms of the
primitive exponential bounds at $\lambda=12$ are less than
$189.886$ for the prepared positions and $554.534$ at A2.
These are finite but pessimistic weighted constants.

For the target ball $B_1(0)\subset D$, choose
$L=\sqrt2(X_2+bU_2)=5.369800925509173$ in (KUK.29), using the
same conservative Minkowski $X_2$. The logarithm of its averaged
landing lower bound is greater than $-48830.778$. This strictly
positive $N$-independent bound illustrates its scope and its severe
numerical pessimism; it is not a useful practical landing rate.
:::

:::{prf:remark} Scope of the population-uniform kinetic calculation
:label: rem-kuk-scope

The constants above are genuine $N$-independent one-step coefficients.
They do not establish population-uniform global mixing, a QSD
eigenfunction-ratio bound, stationary chaos, or an LSI. Those require
the entire signed preparation balance and control of actual phase exits
and survival conditioning. In particular, a desired stationary law's
LSI or a desired contraction rate has not been inserted as a premise.

Row normalization has no global physical-coordinate Lipschitz constant:
with a recipient at the origin and two candidate neighbors near
$\pm Re_1$, the normalized weights have derivative of order
$R/\rho^2$. This is why (KUK.15)--(KUK.16) retain actual inverse-degree
weights and Gaussian integrability, and why a bounded feature distance
cannot replace a physical-coordinate estimate. The dimension-only
column bound handles absolute uncapped moments; it does not by itself
prove contraction of paired row forces. The Gaussian integrability
conditions belong to the inverse-degree route (KUK.26)--(KUK.28),
are explicit, and hold for the original reference parameters. Larger
jitter, larger $h$, or a smaller bandwidth can fail those exponential
conditions; this does not invalidate the polynomial row estimate in
{prf:ref}`lem-ku-polynomial-row-defect`, whose sixth moments are
computed from the actual Gaussian innovations. The actual configured
Styblinski--Tang cubic force and both viscous normalizations have the
finite polynomial budgets of
{prf:ref}`thm-ku-cubic-uniform-moments` and
{prf:ref}`cor-ku-cubic-outer-budget`.

For a different landscape, its actual regional force profiles and
positive-excess integrals must be computed. Unbounded superlinear
forces are not silently replaced by Lipschitz forces. No claim is made
for history donors, geometry feedback, graph curl rotation, or substep
terminal checks without a fresh executed-map calculation. Equal or
near-equal fitness does not invalidate any estimate in this note;
fitness affects the explicit upstream preparation and signed Keystone
flux, which remain in the complete ledger.
:::
