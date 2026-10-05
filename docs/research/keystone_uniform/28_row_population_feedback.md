# Row-normalized population feedback with mandatory revival

(sec-rpf-retained)=
## 1. Retained population and endpoint

:::{prf:definition} Fixed row-normalized marked record
:label: def-rpf-record

Retain the canonical marked rooted-component population preparation of
{prf:ref}`def-rkpf-record`, including its raw reward $R(x)=-|x|^2/2$,
force $F(x)=-x$, actual alive-only normalizers, original frozen-slot
velocities, component Haar collisions and recipient jitter
$\sigma=\sigma_J>0$. Replace only the configured viscosity normalization
by its original row mode
$$
K(x,X)=e^{-|x-X|^2/(2\rho^2)},\quad
a_\lambda(x)=\lambda K(x,X),\quad
b_\lambda(x)=\frac{\lambda[K(x,X)v]}{a_\lambda(x)},\quad
C_0(x,v)=b_\lambda(x)-v.
\tag{RPF.1}
$$
No denominator floor is inserted. The second row mean uses the actual
correlated noisy law $\Lambda_2$:
$$
b_2(y)=\frac{\Lambda_2[K(y,Y)W]}{\Lambda_2K(y,Y)},\qquad
z=(1-t\nu)w-ty+t\nu b_2(y).
$$
Keep $t,c,b,m,a_x,q,s,V,V_c,r_8,B_8,H_8$ of
{prf:ref}`def-pvb-record`. In particular
$m=1-t^2>0$, $a_x=1-t^2(1+c)\in(0,1)$,
$b=t(1+c)$ and $V_c=(1+2|\alpha_{\rm col}|)V$.
The first stages are
$$
U=(1-t\nu)v+t\nu b_\lambda(X),\quad
x_1=mX+tU,\quad w=cU-ctX+q\xi,\quad y=x_1+tw.
\tag{RPF.2}
$$
The stored output is $(y+s\zeta,C_V(z),\mathbf1_{D_L}(y+s\zeta))$.
Every own Gaussian innovation has its original independent marginal.
For $0\le t\nu\le1$, $|U|\le V_c$. This uses bounded prepared
velocities, not a bound on the uncapped $w$.

The entering class is the actual fresh final-Gaussian marked class
$\mathfrak G_{8,s,L}$ of {prf:ref}`def-kpf-marked-register`.
The endpoint proved below is a positive-parameter contraction of this
marked population map and of its current-alive restriction. A uniform
finite-particle transfer needs the separate row consistency interface
specified in the final section.

The source revision for this record is repository HEAD
`6107b67b9e85259581c1932565c1871e8a7e253a`, together with the current
research records 19, 22, 23 and 26. Their exact imported interfaces are
the harmonic moment drift, raw marked preparation feedback, branchwise
spatial BV, Gaussian boundary trace and elementary marked Harris proof.
Their count-provider feedback is replaced here in its entirety.
:::

:::{prf:lemma} Mandatory revival supplies a nonzero Gaussian component
:label: lem-rpf-prepared-mixture

Fix a finite $L>0$, put $R_D=\sqrt dL$, and set
$$
p_0=\sqrt{2/\pi}\,e^{-(L/s+1)^2/2}>0.
\tag{RPF.3}
$$
For every entering $\mu\in\mathfrak G_{8,s,L}$ its actual prepared
physical law admits the decomposition
$$
\lambda=\lambda_c+
 \int N(u,\sigma^2I_d)(dX)\,\kappa(du,dv),\qquad
\operatorname{supp}\lambda_c\subset D_L\times\overline B_{V_c},
$$
where $\kappa$ is a subprobability supported on
$D_L\times\overline B_{V_c}$, and $\kappa1\ge p_0$.
Conditional on $(u,v)$ in this copied branch, the root's own recipient
jitter is independent of $v$. The same class of decompositions is
preserved by convex interpolation between two prepared providers.
More generally this decomposition holds for every cross law
$\zeta J_\mu$ with $\zeta,\mu\in\mathfrak G_{8,s,L}$: root types
come from $\zeta$, while the actual normalized environment, descendants
and donor types come from $\mu$.
:::

:::{prf:proof}
For any fixed real center $a$, the probability
$\mathbb P(|a+sZ|\le L)$ is maximized at $a=0$. Indeed its derivative
for $a>0$ is
$s^{-1}[\varphi((L+a)/s)-\varphi((L-a)/s)]\le0$, and the function
is even. Hence the first coordinate of the independent final Gaussian
gives
$$
e_D\ge2\int_{L/s}^{\infty}\varphi(r)\,dr
\ge2\int_{L/s}^{L/s+1}\varphi(r)\,dr\ge p_0.
$$
This is a lower bound on dead mass in the fixed population class; it
does not condition a finite swarm on an extinction event.

A persistent root is alive and keeps its position in $D_L$. Every copied
or revived root uses a frozen alive donor position $u\in D_L$, then adds
its own independent recipient Gaussian. For fixed preparation pattern,
collision depends on the original frozen velocities, graph and Haar
marks, and not on that jitter. Thus the copied branch has exactly the
displayed Gaussian mixture, even when its centers and velocities are
dependent. Every dead root revives, so its total mass is at least $e_D$.
Mixing two such decompositions preserves their supports and lower masses.
For a cross law the same argument uses the dead mass of $\zeta$, its
compact alive root positions and the compact alive donor positions of
$\mu$; no identity between those two laws is required.
:::

(sec-rpf-first)=
## 2. First-provider posterior and global inverse

:::{prf:lemma} Global first row-mean derivatives
:label: lem-rpf-first-posterior

For the mixture class in {prf:ref}`lem-rpf-prepared-mixture`, define
$$
v_0=\rho^2+\sigma^2,\quad k_0=(\rho^2/v_0)^{d/2},\quad
\eta=\sigma^2/v_0,\quad \varsigma^2=\rho^2\sigma^2/v_0,
$$
$$
c_0=\frac{\sigma^2}{2\rho^2v_0},\quad
B_0=R_D(\rho^{-2}+v_0^{-1}),\quad
A_0=(p_0k_0)^{-1}e^{B_0^2/(2c_0)},
$$
$$
H_2=R_D^2+\frac{2A_0}{c_0e},\quad
S_2=d\varsigma^2+3R_D^2+2\eta^2H_2,
$$
$$
L_0=2dV_c\sqrt{S_2}/\rho^2,\qquad
L_2=2d^2V_cS_2/\rho^4,
\quad
Q_0(r)=(p_0k_0)^{-1}e^{(r+R_D)^2/(2v_0)}.
\tag{RPF.4}
$$
Then $a_\lambda(x)\ge Q_0(|x|)^{-1}$,
$\|Db_\lambda\|\le L_0$ and the coordinate Hessian sums of
$b_\lambda$ are bounded by $L_2$ globally.
Moreover
$$
|D\log a_\lambda(x)|\le
 [(1+\eta)|x|+\sqrt{S_2}]/\rho^2.
\tag{RPF.5}
$$
If $D=\|\lambda-\widetilde\lambda\|_{W_4}$, then
$$
|\Delta b_\lambda(x)|\le2V_cD Q_0(|x|),
$$
$$
\|D\Delta b_\lambda(x)\|\le D Q_0(|x|)
\left[2V_c\ell_\rho+L_0+
 \frac{2V_c}{\rho^2}((1+\eta)|x|+\sqrt{S_2})\right],
\tag{RPF.6}
$$
where $\ell_\rho=(\rho\sqrt e)^{-1}$.
For $t^2\nu L_0\le m/2$, the first position map at every fixed
prepared velocity is globally invertible, with inverse derivative norm
at most $L_I=2/m$.
:::

:::{prf:proof}
Convolving the copied Gaussian with $K$ gives
$k_0e^{-|x-u|^2/(2v_0)}$. Its mass and center bound prove the
denominator lower bound. Under this kernel posterior a copied point,
conditional on its center, has mean
$\eta x+(1-\eta)u$ and covariance $\varsigma^2I_d$.
Let $q_c(x)$ be the posterior compact-branch probability. For
$r=|x|\ge R_D$,
$$
q_c(x)\le(p_0k_0)^{-1}
 e^{-(r-R_D)^2/(2\rho^2)+(r+R_D)^2/(2v_0)}
\le A_0e^{-c_0r^2/2}.
$$
Here $-c_0r^2+B_0r\le-c_0r^2/2+B_0^2/(2c_0)$; the remaining
constant is nonpositive. On $r<R_D$ use $q_c\le1$.
Thus $r^2q_c(x)\le H_2$, since
$\sup_{r\ge0}r^2e^{-c_0r^2/2}=2/(c_0e)$.
The posterior second moment about $\eta x$ is at most
$$
d\varsigma^2+R_D^2+
q_c(x)(R_D+\eta r)^2\le S_2.
$$
In particular its covariance is bounded by this same second moment.

Differentiation of the Gaussian likelihood gives the exact identities
$$
\partial_a b_\lambda=\rho^{-2}
 \operatorname{Cov}_{\pi_x}(v,X_a),\qquad
\partial_{ab}b_\lambda=\rho^{-4}
 \mathbb E_{\pi_x}[(v-b_\lambda)
 (X_a-\mathbb E_{\pi_x}X_a)(X_b-\mathbb E_{\pi_x}X_b)].
$$
The speed bound and Cauchy--Schwarz give the stated, conservatively
dimension-enlarged derivative constants. These identities remain valid
for singular compact subprobabilities: all likelihood derivatives are
integrable, and their denominator is strictly positive. The posterior
mean of $X$ differs from $\eta x$ by at most $\sqrt{S_2}$, proving
(RPF.5).

Subtract normalized numerators to write
$\Delta b=(\Delta m-\widetilde b\Delta a)/a$.
The numerator is bounded by $2V_cD$. Its derivative is at most
$(2V_c\ell_\rho+L_0)D$, because
$\|DK\|_\infty\le\ell_\rho$. Differentiating $1/a$ and using
(RPF.5) gives (RPF.6).

Finally $x_1=mX+t(1-t\nu)v+t^2\nu b_\lambda(X)$.
For a target $x_1$ its inverse equation is a contraction in $X$ with
constant $t^2\nu L_0/m\le1/2$. Existence and uniqueness follow on
all of $\mathbb R^d$, and its inverse derivative bound is $2/m$.
This also proves properness, rather than assuming it from a local
Jacobian bound.
:::

:::{prf:remark} The copied-mass premise cannot be silently removed
:label: rem-rpf-crossover

In one dimension the larger mixture class
$$
\lambda_p=(1-p)\delta_{(0,V_c)}+
 p[N(0,\sigma^2)\otimes\delta_{-V_c}]
$$
has compact centers and Gaussian tails but no fixed copied-mass floor.
At
$$
x_p^2=\frac{2\rho^2v_0}{\sigma^2}
 \log\frac{1-p}{pk_0}
$$
its two posterior branch masses are equal and
$b'_\lambda(x_p)=-V_c\sigma^2x_p/(2\rho^2v_0)$.
For every fixed $\nu>0$, $m+t^2\nu b'_\lambda(x_p)<0$ for small
enough $p$. This proves failure of a uniform inverse theorem for that
larger mixture class. It is not asserted to realize every canonical
conservative preparation. The present marked theorem instead uses its
proved $p\ge p_0(L)$, and does not export this premise to death-disabled
or conservative laws.
:::

(sec-rpf-second)=
## 3. The uncapped second provider

:::{prf:lemma} Broadened second denominator and uncapped posterior moments
:label: lem-rpf-second-posterior

Put
$$
\Sigma=a_x^2\sigma^2+(tq)^2,\quad
\Sigma_w=c^2t^2\sigma^2+q^2,\quad
B_y=a_xR_D+bV_c,\quad B_w=ctR_D+cV_c,
$$
$$
v_q=\rho^2+(tq)^2,\quad
\varepsilon=\min\{1,\rho^2/[2(tq)^2]\},\quad
v_2=a_x^2\sigma^2+v_q/(1+\varepsilon)>\Sigma,
$$
$$
k_2=p_0\left[\frac{\rho^2}{(1+\varepsilon)v_2}\right]^{d/2}
 \exp\left[-\frac{(1+\varepsilon^{-1})b^2V_c^2}{2v_q}\right],
\quad Q_2(r)=k_2^{-1}e^{(r+a_xR_D)^2/(2v_2)}.
\tag{RPF.7}
$$
Define $c_w=(8\Sigma_w)^{-1}$, $c_y=(8\Sigma)^{-1}$ and
$$
M_w=2^{d/2}e^{B_w^2/(4\Sigma_w)},\quad
M_y=2^{d/2}e^{B_y^2/(4\Sigma)},
$$
$$
S_w(r)=c_w^{-1}[\log M_w+\log k_2^{-1}
 +(r+a_xR_D)^2/(2v_2)],
$$
$$
S_y(r)=c_y^{-1}[\log M_y+\log k_2^{-1}
 +(r+a_xR_D)^2/(2v_2)],
$$
$$
B_{20}=\sqrt{c_w^{-1}[\log M_w+\log k_2^{-1}
                         +(a_xR_D)^2/v_2]},\quad
B_{21}=(c_wv_2)^{-1/2},\quad
L_b(r)=2\rho^{-2}\sqrt{S_w(r)S_y(r)}.
\tag{RPF.8}
$$
Every actual joint stage provider and every convex interpolation of two
such providers satisfies
$$
a_2(y)\ge Q_2(|y|)^{-1},\qquad
|b_2(y)|\le\sqrt{S_w(|y|)}\le B_{20}+B_{21}|y|,
\qquad \|Db_2(y)\|\le L_b(|y|).
$$
For two providers, in the full variation convention
$D_2=\|\Lambda_2-\widetilde\Lambda_2\|_{1+|Y|^4+|W|^4}$,
$$
|\Delta b_2(y)|\le
 D_2Q_2(|y|)[1+\sqrt{S_w(|y|)}].
\tag{RPF.9}
$$
No uncapped velocity bound is used.
:::

:::{prf:proof}
Integrate the OU Gaussian first. Conditional on $X,v$, its likelihood
is $(\rho^2/v_q)^{d/2}
e^{-|y-a_xX-bU|^2/(2v_q)}$.
The elementary inequality
$|A-B|^2\le(1+\varepsilon)|A|^2+
(1+\varepsilon^{-1})|B|^2$ and $|U|\le V_c$ give a lower bound
independent of the possible dependence of $U$ on $X$.
On the copied mass, integrate its independent $N(u,\sigma^2I_d)$
coordinate. Completing the square gives exactly the coefficient $k_2$
and exponent $(r+a_xR_D)^2/(2v_2)$ in (RPF.7).
The chosen $\varepsilon$ has
$v_q/(1+\varepsilon)>(tq)^2$, so $v_2>\Sigma$ strictly.

In the own joint stage,
$$
Y=a_xu+bU+a_xI\sigma Z+tq\xi,\qquad
W=-ctu+cU-ctI\sigma Z+q\xi.
$$
The bounded terms have norms at most $B_y,B_w$. The displayed Gaussian
sums have variances at most $\Sigma,\Sigma_w$, conditional on the
frozen source and accepted indicator. They may be dependent on $U$;
the pointwise inequality $|B+G|^2\le2|B|^2+2|G|^2$ suffices. It gives
$\mathbb E e^{c_w|W|^2}\le M_w$ and
$\mathbb E e^{c_y|Y|^2}\le M_y$ by the exact centered Gaussian MGF.

For the kernel posterior $\pi_y$, $K\le1$ and the denominator bound
give
$\mathbb E_{\pi_y}e^{c_w|W|^2}\le M_wQ_2(|y|)$.
Jensen gives $\mathbb E_{\pi_y}|W|^2\le S_w(|y|)$.
The identical argument gives the $Y$ bound. The row-mean derivative is
the covariance $\rho^{-2}\operatorname{Cov}_{\pi_y}(W,Y)$,
bounded by the stated Cauchy--Schwarz envelope. The linear mean bound
uses $(r+a_xR_D)^2\le2r^2+2(a_xR_D)^2$.
All arguments apply to convex mixtures with the same constants.
Finally subtract normalized numerators. The difference of their raw
velocity numerator is bounded by $D_2$, since
$|W|\le1+|W|^4$; the denominator difference is also bounded by $D_2$.
The mean bound of the other provider proves (RPF.9).
:::

(sec-rpf-feedback)=
## 4. Complete weighted comparison of both row providers

:::{prf:definition} Explicit Gaussian score envelopes
:label: def-rpf-score-envelopes

Use the branchwise compact-source spatial variation envelope
$$
B_c=d[L_{G,1}+C_r^Q(1+H_8^{1/8})]+\mathcal T_1(L)
$$
from the proof of {prf:ref}`lem-rkpf-raw-interfaces`, evaluated at its
primitive $\theta_0,\epsilon_0$ envelopes. Thus
$\sum_a|D_{X_a}\lambda_c|1\le B_c$ before prepared-velocity mixing.
For the copied Gaussian branch the corresponding derivative is its
own independent jitter score. No cancellation across velocity fibres
is used.
This envelope is uniform for the cross laws $\zeta J_\mu$ as well.
The compact persistent source uses the fresh Gaussian score, moment
budget and boundary trace of $\zeta$; its pattern derivative uses the
fixed environment $\mu$ and its raw gradient at $H_Q$. The measurement
row, outgoing gate and incoming alive/dead derivative bounds depend
only on these already declared budgets. The copied branch still uses
its own Gaussian and has total mass at most one. Consequently the
same $B_c$ and radial score bounds hold when a common root law is frozen
while its environment varies.

Let $G\sim N(0,I_{2d})$, $r=|G|$, and define
$$
R_X(r)=R_D+\sigma r,\quad
R_Y(r)=B_y+\sqrt\Sigma r,\quad
R_W(r)=B_w+\sqrt{\Sigma_w}r,\quad
P(r)=1+R_Y(r)^4+R_W(r)^4,
$$
$$
I_0=\mathbb E\left[P(r)Q_0(R_X(r))(1+R_X(r)+r)\right],
$$
$$
I_2=\mathbb E\left[(1+R_Y(r)^4)Q_2(R_Y(r))
                 (1+\sqrt{S_w(R_Y(r))})(1+r)\right].
\tag{RPF.10}
$$
These are explicit one-dimensional Gaussian radial integrals with
density $2^{1-d}\Gamma(d)^{-1}r^{2d-1}e^{-r^2/2}$ on $r\ge0$.
Their quadratic exponential coefficients are respectively
$\sigma^2/(2v_0)<1/2$ and $\Sigma/(2v_2)<1/2$; hence they are finite.
They may be evaluated as Gaussian moment integrals, rather than as
unknown regularity constants.

At a viscosity satisfying $t\nu\le1/2$ and
$t^2\nu L_0\le m/2$, put $L_I=2/m$, $D_1=t+t\nu L_0$,
$$
A_x=dL_I(B_c+d/\sigma)+d^2L_I^2t^2\nu L_2
                             +d^2cD_1L_I/q,
\quad A_w=d/q,
$$
$$
J_0=2V_c\ell_\rho+L_0+
          2V_c[(1+\eta)+\sqrt{S_2}]/\rho^2,
$$
$$
C_T^{\rm row}=2V_c(t^2A_x+ctA_w)I_0+
                            dt^2L_IJ_0I_0,
$$
$$
C_{B2}^{\rm row}=\frac{tC_{\rm out}}{1-t\nu}
                                  (A_w+tA_x)I_2,
\quad
C_{\rm fb}^{\rm row}=C_{\rm out}C_T^{\rm row}+
 C_{B2}^{\rm row}(C_{\rm stage}+\nu C_T^{\rm row}).
\tag{RPF.11}
$$
Here $C_{\rm stage},C_{\rm out}$ are the explicit polynomial moment
postprocessing bounds in (PVB.6), valid because $|U|\le V_c$ in row
mode as well. All expressions are increasing viscosity envelopes.
:::

:::{prf:theorem} Absolute feedback for the actual row providers
:label: thm-rpf-two-provider-feedback

For two actual prepared laws in this fixed-box class, with the branchwise
BV envelope above, let their actual correlated stage laws be
$\Lambda_2,\widetilde\Lambda_2$. If
$D=\|\lambda-\widetilde\lambda\|_{W_4}$, then
$$
\|\Lambda_2-\widetilde\Lambda_2\|_{1+|Y|^4+|W|^4}
\le(C_{\rm stage}+\nu C_T^{\rm row})D,
$$
$$
\|\widetilde\lambda K[\lambda,\Lambda_2]
 -\widetilde\lambda K[\widetilde\lambda,\widetilde\Lambda_2]\|_{W_4}
\le\nu C_{\rm fb}^{\rm row}D.
\tag{RPF.12}
$$
Both viscous kicks and the actual cap are retained. The same proof for
marked output weight $w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}$
uses the additional factor $1+\beta+\omega/L^4$.
:::

:::{prf:proof}
Freeze the entering prepared law $\widetilde\lambda$, interpolate the
first providers, and disintegrate by its original prepared velocity.
The global inverse in {prf:ref}`lem-rpf-first-posterior` is uniform
on this interpolation. In coordinates $(x_1,w)$ the Eulerian
displacements are
$A=t^2\nu\Delta b(X)$ and $B=ct\nu\Delta b(X)$.
Their density derivative is
$-\operatorname{div}_{x_1}(Af)-\operatorname{div}_w(Bf)$, with zero
$w$ divergence on each velocity fibre. The first divergence is bounded
by $dt^2\nu L_I DQ_0(|X|)J_0(1+|X|)$ using (RPF.6).

The source derivative contributes at most $dL_I$ times the branchwise
spatial variation. The copied source score is at most $d|Z|/\sigma$;
the persistent compact source has total derivative mass at most $B_c$.
The logarithmic determinant derivative contributes at most
$d^2L_I^2t^2\nu L_2$. The Gaussian mean derivative contributes at
most $d^2cD_1L_I|\xi|/q$, and the $w$ score at most $d|\xi|/q$.
These are the density change-of-variables computations of
{prf:ref}`lem-pvb-joint-scores`, with absolute derivatives taken before
velocity mixing.

Use the pointwise bounds $|X|\le R_X(|G|)$,
$|Y|\le R_Y(|G|)$ and $|W|\le R_W(|G|)$ on the copied branch,
with $G=(Z,\xi)$. On the compact branch use $X\in D_L$ and the
same larger radial bounds for its single OU Gaussian. The bounds also
hold under its compact derivative measure, since $|U|\le V_c$.
Consequently the weighted scores with the extra $Q_0(|X|)$ factor
are at most $A_xI_0,A_wI_0$. This proves the first-field change
$\nu C_T^{\rm row}D$. At a fixed provider the stage Markov map has
weighted norm at most $C_{\rm stage}$, yielding the first assertion.

For the second change hold the actual correlated density
$\rho(y,w)=f(y-tw,w)$ fixed. In row mode the velocity coefficient
$l=1-t\nu$ is constant. Its pullback displacement is
$t\nu\Delta b_2(y)/l$, and its $w$ divergence is exactly zero.
Thus its weighted variation derivative is bounded by
$$
\frac{t\nu D_2}{1-t\nu}
 (1+|y|^4)Q_2(|y|)(1+\sqrt{S_w(|y|)})
 |\nabla_w\rho(y,w)|_1.
$$
The shear identity
$\nabla_w\rho=\nabla_wf-t\nabla_{x_1}f$ and the same source
calculation give the integral bound $(A_w+tA_x)I_2$. Finiteness of
$I_2$ follows from kernel broadening $v_2>\Sigma$, and charges the
uncapped noisy velocity through its actual posterior moments.
Final cap is a common pushforward; final Gaussian has the $C_{\rm out}$
position-weight bound. This gives the coefficient $C_{B2}^{\rm row}$.
Change the first provider while the second is fixed, then the second
while the own first-stage law is fixed, and substitute the first stage
estimate to obtain (RPF.12).

Source mollification, Gaussian truncation and weighted lower
semicontinuity justify the derivative formulas for BV sources. The
explicit $I_0,I_2$ dominate all these approximations. At no step is
$\rho$ replaced by a product of its marginals. The marked weight
factor is {prf:ref}`lem-kpf-marked-output-weight`.
:::

(sec-rpf-population)=
## 5. A noncircular positive marked population interval

:::{prf:definition} Row population parameter endpoints
:label: def-rpf-positive-endpoints

First compute $\epsilon_0,m_0,\theta_0,\bar a,\bar c,H_8$ and the
raw $C_A^Q,C_D,B_A,C_J$ of (RKPF.1)--(RKPF.3), independently of
$L$. Use $r_4,B_4,B_g,u_0,r_H,C_B$ of
{prf:ref}`lem-kpf-frozen-marked-harris`; their position drift uses
only row-convex $U$ and is unchanged. Set
$$
R_H=2+4C_B/(1-r_H),\quad R_x=(R_H-1)^{1/4},\quad
R_1=mR_x+tV_c,\quad M_v=cV_c+ctR_x,
$$
$$
Q^\circ=2(2+tR_1)/m,\qquad L_z^\circ=m+2,
$$
$$
\epsilon_H^\circ=(1-\bar a)e^{-\bar c-\epsilon_0/(\kappa_Cm_0)}
 v_d(1)^2(2\pi q^2)^{-d/2}(2\pi s^2)^{-d/2}
 (L_z^\circ)^{-d}
 e^{-(Q^\circ+M_v)^2/(2q^2)
    -(1+R_1+tQ^\circ)^2/(2s^2)},
$$
$$
\beta=\frac{\epsilon_H^\circ}{r_HR_H+2C_B},\quad
q_H=\max\left\{\frac{2+\beta(r_HR_H+2C_B)}{2+\beta R_H},
                                      1-\epsilon_H^\circ/2\right\},
\quad g_H=1-q_H>0,
$$
$$
C_K=1+\beta(1+u_0)(1+B_4),\quad
\omega=\max\{1,\beta R_H+1,\beta B_A,8C_KC_D/g_H\},
$$
$$
\theta_*^{\rm row}=\min\{\theta_0/2,g_H\beta m_0/(16C_KC_A^Q)\},
\quad
\epsilon_*^{\rm row}=\min\{\epsilon_0/2,g_H\beta m_0/(16C_KC_A^Q)\}.
\tag{RPF.13}
$$
Choose one fixed finite $L$ satisfying
$$
L>\max\left\{1,R_x,(H_8/\epsilon_*^{\rm row})^{1/8},
 (\omega/(\beta u_0))^{1/4},
 \frac{bV_c+\sqrt{a_x^2\sigma^2+t^2q^2+s^2}
       \sqrt{2\log(2d/\epsilon_*^{\rm row})}}{1-a_x}\right\}.
\tag{RPF.14}
$$
Only after this box choice compute (RPF.3)--(RPF.10), and choose
$$
\bar\nu_L=\min\left\{1,\frac m{4t},\frac m{2t^2L_0},
 \frac1{t(1+B_{20}+B_{21}R_1)},
 \frac m{4t^2(1+B_{21})},
 \frac1{t^2[1+L_b(R_1+tQ^\circ)]}\right\}>0.
$$
Evaluate $\overline C_{\rm fb}^{\rm row}$ in (RPF.11) at
$\bar\nu_L$, put
$C_{\rm fb}^{w,\rm row}=[1+\beta(1+u_0)]
\overline C_{\rm fb}^{\rm row}$, and set
$$
\nu_*^{\rm row}=\min\left\{\bar\nu_L,
 \frac{g_H}{8C_{\rm fb}^{w,\rm row}
 [C_J/\beta+2C_A^Q(\theta_0+\epsilon_0)/(\beta m_0)+C_D]}\right\},
\qquad r_*^{\rm row}=(1+q_H)/2<1.
\tag{RPF.15}
$$
Every endpoint is positive. The preliminary minorization constants in
(RPF.13) are independent of $L$; all posterior and row feedback
constants are computed after its fixed choice and paid through
$\bar\nu_L,\nu_*^{\rm row}$. Thus no small box or prescribed large
viscosity is certified by this order of choices.
:::

:::{prf:theorem} Complete positive-viscosity marked row population law
:label: thm-rpf-marked-population

If $0<\theta\le\theta_*^{\rm row}$ and
$0<\nu\le\nu_*^{\rm row}$, the actual row-normalized marked
population map satisfies
$$
\|\mathcal F_L^{\rm row}\mu-\mathcal F_L^{\rm row}\mu'\|_w
\le r_*^{\rm row}\|\mu-\mu'\|_w
\quad(\mu,\mu'\in\mathfrak G_{8,s,L}),
\quad w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}.
\tag{RPF.16}
$$
It has a unique fixed point $\pi_L^{\rm row}$ in this class, with
positive alive mass. Every consistent entering population with nonzero
alive mass enters the class after a uniform finite burn-in depending
only on the fixed primitives and $L$. After that burn-in the above
geometric rate holds. Its current-alive law is
$\pi_L^{\rm row,A}=\pi_L^{\rm row}\restriction\{a=1\}/m_A(\pi_L^{\rm row})$.
For iterates already in the class,
$$
\|\alpha_{\mu_n}-\pi_L^{\rm row,A}\|_{\rm TV}
\le \frac{(r_*^{\rm row})^n}{m_0}\|\mu_0-\pi_L^{\rm row}\|_w,
$$
$$
W_{2,G}(\alpha_{\mu_n},\pi_L^{\rm row,A})^2
\le4\lambda_{\max}(G)(dL^2+V^2)
 \frac{(r_*^{\rm row})^n}{m_0}\|\mu_0-\pi_L^{\rm row}\|_w.
\tag{RPF.17}
$$
Here TV is half the full variation. The all-dead absorbing population
and finite-particle QSDs are separate objects.
:::

:::{prf:proof}
Rowwise convexity of the first kick gives the identical moment and
large-box safe-return estimates as
{prf:ref}`lem-kpf-marked-moments` and
{prf:ref}`lem-kpf-large-box-survival`. They retain both viscous kicks,
since the second kick does not alter $y+s\zeta$. Thus the Gaussian
moment class is invariant, $e_D\le\epsilon_*^{\rm row}$, and
the copied-mass floor (RPF.3) applies to every provider in that class.

We prove the preliminary minorization used in (RPF.13) with a floor
independent of $L$. The small-$T$ set of the marked Harris proof has
only alive roots because $\omega/\beta>R_H$. Its persistent isolated
event has probability at least
$(1-\bar a)e^{-\bar c-\epsilon_0/(\kappa_Cm_0)}$.
On this event $|x_1|\le R_1$, and the OU velocity has Gaussian mean
of norm at most $M_v$. At fixed $x_1$, its actual second row map is
$$
Z(w)=(m-t\nu)w-tx_1+t\nu b_2(x_1+tw).
$$
The linear posterior bound and the restrictions on $\bar\nu_L$ give
$$
|Z(w)|\ge(m-t\nu-t^2\nu B_{21})|w|
                 -tR_1-t\nu(B_{20}+B_{21}R_1)
\ge(m/2)|w|-tR_1-1.
$$
Its homotopy replacing $\nu$ by $a\nu$, $0\le a\le1$, obeys the
same bound. Therefore it is proper of degree one, and every target in
$B_1$ has a preimage in $B_{Q^\circ}$. On that ball its derivative
norm is at most
$m+t\nu+t^2\nu L_b(R_1+tQ^\circ)\le L_z^\circ$.
The second posterior fields are smooth, because their strictly positive
Gaussian convolutions have all local derivatives. Sard's theorem and
the area formula yield the joint Gaussian density floor in (RPF.13),
exactly as in {prf:ref}`lem-pvb-frozen-minorization`. Any singular part
is nonnegative. Final position ball $B_1$ is alive because $L>1$;
capping is a common pushforward. This proves the claimed floor without
a bound on uncapped velocities or a global second inverse.

The marked drift proof now gives the elementary frozen-kernel Harris
rate $q_H$ of (RPF.13). It only uses $|U|\le V_c$ and the just-proved
floor. For nonlinear feedback, split an input difference into the
frozen Markov part, the preparation-provider change, and the two
viscous-provider changes. The preparation estimate is the unchanged
raw marked source-first bound (RKPF.3). Its actual velocity readout and
alive-only statistics are independent of viscosity normalization.
The latter changes are exactly (RPF.12), with the terminal-mark weight
bound, not the count-provider theorem.
The alive normalization and marked weight estimates of (KPF.8), the
fixed kinetic weight factor $C_K$ and prepared moment factor $C_J$
therefore give the three charged coefficients
$$
\frac{2C_KC_A^Q(\theta+\epsilon_*^{\rm row})}{\beta m_0},
\quad \frac{C_KC_D}{\omega},
\quad \nu C_{\rm fb}^{w,\rm row}
 [C_J/\beta+2C_A^Q(\theta_0+\epsilon_0)/(\beta m_0)+C_D].
$$
Their sum is at most $g_H/2$ by (RPF.13)--(RPF.15).
Adding the frozen $q_H$ gives (RPF.16).

The same weighted-Banach and Gaussian latent tightness proof as
{prf:ref}`thm-kpf-large-box-population-convergence` proves completeness
of the invariant class and its unique fixed point. For an arbitrary
nonempty consistent input, one update has box-source moment bounded
by $M_{8,\rm box}$; row convexity gives the same bound. The safe-return
estimate gives dead mass at most $\epsilon_*^{\rm row}$ after every
such update, independently of the old retained dead positions.
With $\theta\le\theta_0/2$ and the halved dead-mass envelope, its
eighth-moment recurrence has coefficient at most $(1+3r_8)/4<1$
and limiting sublevel at most $2H_8/3$. A uniform finite burn-in
therefore reaches $H_8$. This is the exact moment argument of that
theorem, with an unchanged row-convex bound.

Finally normalized restriction to the alive set has full variation
at most twice the full input variation divided by $m_0$; in half-TV
this gives the first bound (RPF.17). The physical alive phase space
has squared $G$ diameter at most
$4\lambda_{\max}(G)(dL^2+V^2)$. Matching the common probability
identically and coupling the residual gives the second bound. This
identifies a current-alive population law only.
:::

(sec-rpf-finite-interface)=
## 6. Finite-array interfaces retained separately

:::{prf:lemma} Actual row denominators on a localized finite array
:label: lem-rpf-finite-local-denominator

For any $N\ge2$, any finite array $z_j=(x_j,v_j)$ and any query
$|x|\le R$, if at least $\delta N$ nonself candidates have
$|x_j|\le R_0$, then its exact self-excluded Gaussian denominator
normalized by $N-1$ is at least
$$
\frac{\delta N}{N-1}
 e^{-(R+R_0)^2/(2\rho^2)}.
\tag{RPF.18}
$$
If $N^{-1}\sum_j|x_j|^p\le H$, at most $NH/R_0^p$ labels lie
outside that ball, so there are at least
$N(1-H/R_0^p)-1$ nonself candidates inside. This is a bound on the
original denominator, not a substituted degree floor.

For any two normalized Gaussian-provider rows with their own
denominators $a,\widetilde a\ge a_*>0$, signed numerator and denominator
differences $\Delta n,\Delta a$ satisfy
$$
\left|\frac n a-\frac{\widetilde n}{\widetilde a}\right|
\le\frac{|\Delta n|+|\widetilde n/\widetilde a|\,|\Delta a|}{a_*}.
\tag{RPF.19}
$$
This applies to the uncapped second numerator without bounding every
velocity: use its actual local weighted numerator or moment budget.
:::

:::{prf:proof}
Each stated candidate has kernel weight at least the displayed
exponential. Sum these exact weights and divide by $N-1$.
Markov's deterministic moment inequality supplies the candidate count;
self exclusion discards at most one label. Subtract the fractions as
$(\Delta n-(\widetilde n/\widetilde a)\Delta a)/a$ for the last bound.
:::

:::{prf:lemma} Local row transport with uncapped moments and exact self exclusion
:label: lem-rpf-local-row-transport

Let $\lambda,\widetilde\lambda$ be joint position/velocity providers,
with finite second velocity moments at most $M_2$, and let a coupling
have squared phase cost
$e^2=\mathbb E[|X-\widetilde X|^2+|v-\widetilde v|^2]$.
At queries $x,\widetilde x$ suppose both original denominators are at
least $a_*>0$. Then their row velocity means satisfy
$$
|b_\lambda(x)-b_{\widetilde\lambda}(\widetilde x)|
\le C(a_*,M_2)(|x-\widetilde x|+e),
\quad
C(a_*,M_2)=\frac{1+\ell_\rho\sqrt{M_2}
                          +\ell_\rho\sqrt{M_2}/a_*}{a_*}.
\tag{RPF.20}
$$
This holds for the uncapped joint stage provider, with its actual
second velocity moment; no velocity maximum is substituted.

For an empirical provider $\lambda_N=N^{-1}\sum_j\delta_{(x_j,v_j)}$
at its own query $x_i$, let $b_N$ include all labels and
$b_{N,-i}$ be the exact self-excluded mean. If its self-excluded
denominator normalized by $N-1$ is at least $a_*$, then
$$
b_{N,-i}(x_i)-b_N(x_i)
=\frac{b_N(x_i)-v_i}{Na_{\lambda_N}(x_i)-1},\qquad
|b_{N,-i}-b_N|\le\frac{|b_N|+|v_i|}{(N-1)a_*}.
\tag{RPF.21}
$$
Thus whenever all queried denominators in a localization event have
that bound and $N^{-1}\sum_i|v_i|^2\le M_2$,
$$
\frac1N\sum_i|b_{N,-i}(x_i)-b_N(x_i)|^2
\le\frac{2M_2(1+a_*^{-1})}{(N-1)^2a_*^2}.
\tag{RPF.22}
$$
The same formulas apply to the second empirical row with velocities
$w_i$ and its normalized second moment. The event need not be independent
of those velocities or their OU draws.
:::

:::{prf:proof}
The kernel is globally $\ell_\rho$-Lipschitz in each position argument.
Subtract denominators under the coupling to obtain
$|\Delta a|\le\ell_\rho(|x-\widetilde x|+e)$.
Subtract numerators as
$K(x,X)(v-\widetilde v)+
[K(x,X)-K(\widetilde x,\widetilde X)]\widetilde v$.
Cauchy--Schwarz gives
$|\Delta n|\le e+\ell_\rho\sqrt{M_2}(|x-\widetilde x|+e)$.
Also $|b_{\widetilde\lambda}(\widetilde x)|\le\sqrt{M_2}/a_*$.
Insert these three bounds into (RPF.19) and enlarge the coefficient
of the first $e$ to the common coefficient in (RPF.20).

At an own query the self kernel is exactly one. The full denominator
is $a=N^{-1}\sum_jK(x_i,x_j)$ and its numerator is $n$; hence
$b_{N,-i}=(Nn-v_i)/(Na-1)$, which proves (RPF.21).
The lower bound on the nonself denominator implies $a\ge a_*$,
since $a=[1+(N-1)a_{-i}]/N$ and $a_*\le1$.
Kernel-weighted Jensen gives
$|b_N(x_i)|^2\le M_2/a_*$: its numerator second moment is
at most the full empirical second moment $M_2$.
Squaring (RPF.21), using $(A+B)^2\le2A^2+2B^2$, then averaging
the actual $|v_i|^2$ proves (RPF.22). These are deterministic
array inequalities, so no noise independence is needed.
:::

:::{prf:remark} Completed population result and remaining finite transfer
:label: rem-rpf-finite-transfer

(RPF.16)--(RPF.17) complete a positive row-normalized marked population
and alive-law regime. The exact finite-array source, landing moment,
safe-return and recent-survival-change-of-measure proofs in research
record 24 remain valid in row mode because the first row average is
convex and the second kick does not move the landing position. Its
finite consistency and weak modulus constants in (SPT.5)--(SPT.8)
were proved for count provider feedback and cannot be relabeled as row
constants without an additional proof.

The additional quantitative finite interface is the conditional weak
comparison of the actual self-excluded first and noisy second empirical
row providers against their population fields, uniform over the
localized current-alive input class. (RPF.18)--(RPF.19) supply its
exact local denominator and numerator transport step; (RPF.20)--(RPF.22)
also retain uncapped moments and the precise self-exclusion error.
That comparison
must retain the random prepared component law, the same-row OU query,
the correlated uncapped velocity numerator and the exceptional local
degree event. A suitable proof must produce explicit functions
$A_N(H)\to0$ and a continuity modulus $\omega_H(r)\to0$ at fixed
$H,L$, and then an explicit restart sequence for which the accumulated
consistency, modulus, moment and recent-survival-tilt errors vanish.
The population Gaussian mass floor $p_0(L)$ is not a pathwise finite
sample mass or an empirical denominator floor. This population argument
alone does not assert the finite-particle rate; its separate completed
proof is identified below. Finite-particle QSD identification and the
default $\nu=.3,L=2$ remain outside the proved endpoint.
:::

:::{prf:remark} Subsequent completion of the finite row transfer
:label: rem-rpf-finite-completion

The additional interface has now been completed without an empirical
copied-mass floor. {prf:ref}`thm-rft-conditional-consistency` gives
the explicit actual full marked one-step function $\mathcal A_N\to0$,
retaining both nonself row fields and the correlated uncapped OU stage.
{prf:ref}`thm-rwm-population-modulus` gives a positive power modulus on
the entire alive-floor class, including atomic empirical inputs and
arbitrary retained dead coordinates. Their exact own recent-survival
restart is {prf:ref}`thm-rft-uniform-surviving-law`, and
{prf:ref}`cor-rft-alive-wasserstein` and
{prf:ref}`cor-rft-all-slot-alive-sample` give the alive physical empirical,
random empirical-law and both sampling-order estimates.

Thus the small-positive-viscosity, sufficiently large fixed-box row
population regime of this record now has an $N$-uniform alive-law
relaxation rate with an explicit error tending to zero with $N$.
The target remains $\pi_L^{\rm row,A}$, the current-alive stationary
population law. The unproved default $\nu=.3,L=2$ regime, finite-array
exact invariant-law identification, finite-particle QSD and additional
future-horizon survival conditioning are distinct obligations.
:::
