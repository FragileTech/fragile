# Discrete native-cap certificates at the reference step

(sec-rcap-retained)=
## 1. Retained kernel and scope

:::{prf:definition} The reference discrete residual
:label: def-rcap-retained

Retain research record 27's finite harmonic kinetic kernel and its
actual marked population providers, at
$d=3$, $h=1/25$, $t=1/50$, $\nu=3/10$, $\gamma=b_O=1$,
$V=2$, $\rho=1$, $\sigma_x=\sigma_J=1/10$ and $L=2$.
Thus
$$
c=e^{-1/25},\quad b=t(1+c),\quad m=1-t^2,\quad a_x=1-tb,
\quad q^2=(1-c^2)/2,\quad s^2=1/2500,\quad \ell=e^{-1/2}.
$$
The native cap remains $C_V(z)=Vz/(V+|z|)$. All innovations are
uncut Gaussians. The actual preparation uses original frozen slot
velocities for its component collision, with prepared bound $V_c=4$.
Alive roots retain or copy an alive source; dead roots copy an alive
source. Sources lie in $D=(-2,2)^3$, while retained dead positions
and Gaussian stored output positions remain unbounded.
The current-frame fitness statistics and their conditional-alive
normalizers remain the actual ones.

The carrier of the first theorem is the complete nonviscous finite
kinetic kernel, including its native cap. The carrier of the second
is an actual root OU/second-kick/cap map with deterministic population
providers; those providers may be different between the compared laws.
Their joint landing-position/OU-velocity laws are not factorized.
The final block identity is pathwise for the actual finite count
kernel under shared innovations. A frozen population field is never
substituted for a noisy finite empirical provider.

The accepted prefix comprises research record 27's correlated
second-graph bound, source-box provider moment, and root-kernel
minorization. The target is a cap-compatible discrete own-provider
law block at the reference viscosity. The previously failed
standalone cross-metric cap inference is replaced below by a complete
update certificate and a quantified actual second-kick sensitivity.
These replacements do not themselves close the own-provider block.
:::

(sec-rcap-complete-harmonic)=
## 2. Complete harmonic update with every native-cap secant

:::{prf:lemma} Native-cap increment as a symmetric sector
:label: lem-rcap-sector

For every $z,\widetilde z\in\mathbb R^d$ there is a symmetric
matrix $D=D(z,\widetilde z)$ with $0\le D\le I$ such that
$$
C_V(z)-C_V(\widetilde z)=D(z-\widetilde z).
\tag{RCAP.1}
$$
Writing $Z=z-\widetilde z$, $C=C_V(z)-C_V(\widetilde z)$
and $E=Z-C$, one also has
$$
\langle E,C\rangle\ge0,\qquad
|Z|^2-|C|^2\ge|E|^2.
\tag{RCAP.2}
$$
:::

:::{prf:proof}
The cap is continuously differentiable, including at zero.
Its radial and tangential derivative eigenvalues at radius $r$ are
$V^2/(V+r)^2$ and $V/(V+r)$, respectively.
Integrate its symmetric positive semidefinite Jacobian along the
segment between the two inputs:
$D=\int_0^1 DC_V(\widetilde z+\lambda Z)\,d\lambda$.
The interval $0\le D\le I$ is convex, giving (RCAP.1).
Then $\langle E,C\rangle=Z^T(D-D^2)Z\ge0$ and
$|Z|^2-|C|^2=|E|^2+2\langle E,C\rangle$, proving (RCAP.2).
:::

:::{prf:theorem} A complete discrete contraction despite the native cap
:label: thm-rcap-harmonic-whole-update

Disable count viscosity, cloning and death only for this theorem.
Retain the reference force, OU noise, final position noise and native
cap, with positions unbounded. Define
$$
Q_\beta(r,\zeta)=|r|^2+2\beta r\cdot\zeta+|\zeta|^2,
\qquad \beta=\frac1{25}.
$$
For every pair of entering phase arrays and every shared realization
of the two full Gaussian innovation arrays,
$$
\frac1N\sum_iQ_\beta(x_i^+-\widetilde x_i^+,
                         v_i^+-\widetilde v_i^+)
\le\left(1-\frac1{1040}\right)
      \frac1N\sum_iQ_\beta(x_i-\widetilde x_i,v_i-\widetilde v_i).
\tag{RCAP.3}
$$
The statement holds for every $N$, without a velocity or position
bound on the entering arrays. It bounds optimal physical transport
for this exact kinetic kernel by the exhibited valid coupling.
:::

:::{prf:proof}
For an entering row difference $(r,\zeta)$ the actual uncapped
differences are
$$
\binom{R}{Z}
=H\binom r\zeta,\qquad
H=\begin{pmatrix}a_x&b\\-t(c+a_x)&c-tb\end{pmatrix}.
$$
Both innovation differences cancel before the cap; the shared final
position innovation also cancels. By (RCAP.1), the output difference
is $(R,DZ)$ for a symmetric $0\le D\le I$, possibly depending on
the actual noises. Put
$G=\left(\begin{smallmatrix}1&\beta\\\beta&1\end{smallmatrix}\right)$.
For a scalar $\delta\in[0,1]$ let
$H_\delta=\operatorname{diag}(1,\delta)H$.
For a fixed vector the quadratic
$H_\delta^TG H_\delta$ is convex in $\delta$: its second derivative
as a quadratic form is twice the square of the second row of $H$.
Consequently
$$
H_\delta^TG H_\delta
\le(1-\delta)H_0^TG H_0+\delta H_1^TG H_1.
$$
Both endpoint deficits satisfy
$$
G-H_0^TG H_0\ge\frac1{1000}I,\qquad
G-H_1^TG H_1\ge\frac1{1000}I.
\tag{RCAP.4}
$$
Here is a rational certificate for these inequalities.
The elementary exponential bounds give
$0.96<c<0.9608$, hence
$0.0392<b<0.039216$ and
$0.99921568<a_x<0.999216$.
Write $r_H=t(c+a_x)>0$ and $v_H=c-tb>0$.
The same bounds give
$0.0391843136<r_H<0.03920032$ and
$0.95921568<v_H<0.960016$.

For the first endpoint, the two diagonal entries of the deficit
minus $I/1000$ are strictly greater than $0.00056$ and $0.997$,
respectively, and its off-diagonal absolute value is less than
$0.00084$. These follow by substituting the preceding rational
intervals into
$1-a_x^2-0.001$, $1-b^2-0.001$ and $\beta-a_xb$.
Its determinant therefore exceeds
$0.00056(0.997)-0.00084^2>0$.
For the second endpoint the corresponding expressions are
$$
1-a_x^2+2\beta a_xr_H-r_H^2-0.001,\quad
1-b^2-2\beta bv_H-v_H^2-0.001,
$$
$$
\beta-a_xb-\beta a_xv_H+\beta b r_H+r_Hv_H.
$$
The same intervals bound the two diagonal expressions below by
$0.0021$ and $0.0727$, and the off-diagonal absolute value above
by $0.00025$. Its determinant exceeds
$0.0021(0.0727)-0.00025^2>0$. Positive diagonal entries and
positive determinant prove both matrix inequalities in (RCAP.4).
All numbers used here are terminating rationals.

Orthogonally diagonalize $D$. Because the four blocks of $H$ and
$G\otimes I_d$ are scalar multiples of $I_d$, that change of basis
reduces the full row to the scalar cases just certified. Thus
$$
Q_\beta(R,DZ)\le Q_\beta(r,\zeta)
                       -\frac1{1000}(|r|^2+|\zeta|^2).
$$
Since $Q_\beta\le(1+\beta)(|r|^2+|\zeta|^2)$,
$(1/1000)/(1+\beta)=1/1040$. Summing proves (RCAP.3).
The argument uses the complete harmonic map before applying the
cap sector; it never asserts that the standalone cap contracts
$Q_\beta$.
:::

:::{prf:corollary} Exact nonviscous finite invariant and population-uniform law rate
:label: cor-rcap-nonviscous-invariant

On $E_N=(\mathbb R^d\times\overline B_V)^N$, let $K_N^0$ be
exactly the cloning-disabled, death-disabled kernel of the preceding
theorem, with both full independent Gaussian innovation arrays.
Let $\mathcal W_{2,N}$ use the normalized ground cost
$N^{-1}\sum_iQ_\beta$, and let $W_{2,G}$ denote its one-row
counterpart. Put
$$
q_Q=\sqrt{1-1/1040}<1,\quad
A_{\rm anc}=\sqrt{(1+\beta)[d(t^2q^2+s^2)+V^2]},\qquad
M_\infty=A_{\rm anc}/(1-q_Q).
$$
For every $N$ there is a unique invariant probability
$\Pi_N^0\in\mathcal P_2(E_N)$ and
$$
\mathcal W_{2,N}(\lambda(K_N^0)^n,\Pi_N^0)
\le q_Q^n\mathcal W_{2,N}(\lambda,\Pi_N^0)
\le q_Q^n\left[
\left(\int\frac1N\sum_iQ_\beta(x_i,v_i)\,d\lambda\right)^{1/2}
 +M_\infty\right].
\tag{RCAP.3a}
$$
The invariant is exchangeable, with each slot's $Q_\beta$ second
moment at most $M_\infty^2$. The same right-hand side bounds both
the alive empirical-measure-law Wasserstein distance, with ground
metric $W_{2,G}$, and the Wasserstein distance of uniformly sampled
alive-row laws to their respective invariant targets.
No population-size floor occurs in this exact nonviscous assertion.
:::

:::{prf:proof}
The innovation coupling in (RCAP.3) draws independent standard OU
and final-position arrays in each own kernel; sharing the
corresponding entries between kernels preserves both exact marginals.
Integrating it against an optimal entering coupling gives a
$q_Q$ contraction on $\mathcal P_2(E_N)$.
The closed finite-dimensional phase space is complete under the
positive definite quadratic ground metric, so its Wasserstein
space is complete. The kernel maps this space into itself by its
affine positional identity and bounded output velocity.

At the zero array the positional second moment is
$d(t^2q^2+s^2)$ and the velocity squared norm is at most $V^2$.
The upper bound $Q_\beta\le(1+\beta)(|x|^2+|v|^2)$ proves
the anchor bound $A_{\rm anc}$. Contraction iteration is Cauchy
and gives its unique invariant. Comparing that invariant with the
zero anchor and using the triangle inequality gives its normalized
second-moment square root at most
$A_{\rm anc}+q_QM_\infty=M_\infty$.
This proves (RCAP.3a).

Kernel equivariance and invariant uniqueness imply exchangeability.
Its normalized moment therefore equals every individual slot
moment. For any two arrays, matching corresponding slots bounds the
Wasserstein distance of their empirical measures by the array
ground distance. Choosing a common independent uniform slot gives
the same bound for the sampled-row laws. Push the array coupling
through these maps to obtain the two claimed law estimates.
:::

:::{prf:corollary} A larger complete positive-count interval from the repaired cap baseline
:label: cor-rcap-positive-count-interval

Disable cloning and death, retain both actual dense count kicks, and
fix an entering velocity envelope $V_*\ge V$. Let $C_\nu>0$ be
the explicit primitive perturbation coefficient in
{prf:ref}`thm-ku-count-small-viscosity-contraction`, evaluated at
the unchanged reference kinetic parameters and that envelope.
Set
$$
\eta_Q=\sqrt{\frac{1+\beta}{m(1-\beta)}},\qquad
\nu_Q=\min\left\{1,\frac1t,
                    \frac{1-q_Q}{2\eta_QC_\nu}\right\}>0.
\tag{RCAP.3b}
$$
For every $0\le\nu\le\nu_Q$, the actual shared-innovation kinetic
updates satisfy
$$
\left(\mathbb E\frac1N\sum_iQ_\beta(\Delta x_i^+,\Delta v_i^+)
\right)^{1/2}
\le(q_Q+\nu\eta_QC_\nu)
\left(\frac1N\sum_iQ_\beta(\Delta x_i,\Delta v_i)\right)^{1/2},
\tag{RCAP.3c}
$$
where $q_Q+\nu\eta_QC_\nu\le(1+q_Q)/2<1$.
Consequently the exact finite invariant and all three law estimates
of {prf:ref}`cor-rcap-nonviscous-invariant` hold for this
positive-viscosity kernel with that contraction factor and the
correspondingly changed moment prefactor.
Using $V_*=4$, diagnostic evaluation gives
$\nu_Q\simeq0.000260436$. This exact positive interval remains
strictly distinct from the reference $\nu=.3$ residual.
:::

:::{prf:proof}
The perturbation derivation in research record 15 proves, before
imposing its old small-viscosity endpoint, that for
$0\le\nu\le1$ and $t\nu\le1$ the RMS difference between the
paired actual output and its paired nonviscous reference output is
at most $\nu C_\nu\sqrt{\mathcal Q_m}$, where
$\mathcal Q_m=N^{-1}\sum_i[m|r_i|^2+|\zeta_i|^2]$.
Its actual noisy second-graph estimate and cap-Jacobian perturbation
use only these two viscosity restrictions, the entering $V_*$
bound and the unchanged positive primitives. The derivation therefore
applies on the enlarged interval (RCAP.3b); the old $q_0$ margin
is not imported.

The norm comparisons are
$Q_\beta\le(1+\beta)Q_m/m$ and
$Q_m\le Q_\beta/(1-\beta)$. They bound that perturbation in the
$Q_\beta$ norm by $\nu\eta_QC_\nu\sqrt{Q_\beta}$.
Minkowski and (RCAP.3) give (RCAP.3c), and the new primitive
endpoint makes its factor strictly smaller than one.
At the zero-array anchor, count viscosity does not change the
position identity and the output velocity is capped by $V$.
Thus the same $A_{\rm anc}$ bound applies. The complete
Wasserstein contraction/invariant argument and the two observation
pushforwards from the preceding corollary apply unchanged, with
$M_\infty=A_{\rm anc}/[1-q_Q-\nu\eta_QC_\nu]$.
The decimal endpoint is diagnostic; positivity and every claimed
rate follow from the exact formulas.
:::

(sec-rcap-joint-b2)=
## 3. The actual second kick followed by the cap

:::{prf:definition} Actual deterministic second-provider fields
:label: def-rcap-b2-fields

Let $\Lambda$ be a probability law on the actual joint $(y,w)$
stage, with $\Lambda|w|\le M_w$. Define its count fields
$$
a(y)=\int K_\rho(y-y')\,d\Lambda(y',w'),\qquad
M(y)=\int K_\rho(y-y')w'\,d\Lambda(y',w').
$$
Then $0\le a\le1$, $|M|\le M_w$,
$|Da|\le\ell_\rho$ and $\|DM\|\le\ell_\rho M_w$.
The root's actual second kick at fixed first-drift position $x_1$ is
$$
Z_\Lambda(x_1,w)
=[m-t\nu a(x_1+tw)]w-tx_1+t\nu M(x_1+tw),\qquad
T_\Lambda=C_V\circ Z_\Lambda.
\tag{RCAP.5}
$$
This representation retains $y=x_1+tw$, including its correlation
with the root OU velocity. Put
$$
A=m-t\nu>0,\quad C=tR_1+t\nu M_w,\quad
\alpha=t^2\nu\ell_\rho,\quad r_0=C/A,
$$
$$
K_0=m+\alpha(M_w+r_0),\qquad
K_X=\max\{t+t\nu\ell_\rho(M_w+r_0),\,Vt\nu\ell_\rho/A\}.
$$
Assume $|x_1|\le R_1$ and $K_0>V\alpha/A$.
Define the nonincreasing radial envelope
$$
\mathcal K(r)=
\begin{cases}
K_0,&0\le r\le r_0,\\[1mm]
\displaystyle\frac{V[m+\alpha(M_w+r)]}{V+Ar-C},&r>r_0.
\end{cases}
\tag{RCAP.6}
$$
:::

:::{prf:lemma} The cap controls the noisy-graph derivative at every OU velocity
:label: lem-rcap-global-b2-jacobian

For the actual joint provider fields above, at every $w$ and
$|x_1|\le R_1$,
$$
\|D_wT_\Lambda(x_1,w)\|\le\mathcal K(|w|),\qquad
\|D_{x_1}T_\Lambda(x_1,w)\|\le K_X.
\tag{RCAP.7}
$$
In particular the growing $|w|$ factor in the second graph derivative
does not cause an unbounded capped derivative.
:::

:::{prf:proof}
The actual pre-cap map satisfies
$|Z_\Lambda|\ge A|w|-C$.
Also its derivatives have bounds
$$
\|D_wZ_\Lambda\|
\le m+\alpha(M_w+|w|),\qquad
\|D_{x_1}Z_\Lambda\|
\le t+t\nu\ell_\rho(M_w+|w|).
$$
These follow by differentiating (RCAP.5); the linear coefficient
$m-t\nu a$ is nonnegative and at most $m$.
The cap derivative bound is
$\|DC_V(Z)\|\le V/(V+|Z|)$.
It gives the two bounds as ratios with denominator
$V+(A|w|-C)_+$.
The first ratio increases up to $r_0$, where its value is $K_0$.
Above $r_0$ its derivative has the sign of
$V\alpha/A-K_0$, which is negative. This proves its bound by
(RCAP.6). The second ratio also increases up to $r_0$ and
thereafter is monotone towards $Vt\nu\ell_\rho/A$; its supremum
is the stated maximum $K_X$. All bounds are pointwise, before
averaging the OU noise.
:::

:::{prf:lemma} Gaussian averaging with the actual joint second graph
:label: lem-rcap-averaged-b2-gap

Let $\xi$ be a standard $d$-Gaussian and set
$$
K_G^2=\mathbb E\mathcal K(q|\xi|)^2.
\tag{RCAP.8}
$$
For every fixed $x_1$ in the stated ball and every pair of OU means
$b_0,\widetilde b_0$,
$$
\mathbb E|T_\Lambda(x_1,b_0+q\xi)
       -T_\Lambda(x_1,\widetilde b_0+q\xi)|^2
\le K_G^2|b_0-\widetilde b_0|^2.
\tag{RCAP.9}
$$
One has $K_G<K_0$; if $K_0<1$, this is a strict contraction of
velocity-mean sensitivity for this complete OU/second-kick/cap map.
:::

:::{prf:proof}
For a ball centered at a shifted Gaussian mean, its Gaussian mass is
maximized at mean zero. To verify this directly, rotate the mean
onto the first coordinate and condition on the other $d-1$
coordinates. Each remaining section is a symmetric interval.
Its translated one-dimensional Gaussian mass is maximal at zero:
differentiation of the interval mass shows that it decreases with
the absolute translation. Integrate over the other coordinates.
By layer-cake integration, the same comparison holds for every
nonnegative nonincreasing radial function. Apply it to
$\mathcal K(|w|)^2$ to obtain
$$
\mathbb E\mathcal K(|b_0+q\xi|)^2
\le\mathbb E\mathcal K(q|\xi|)^2.
$$
Integrate the actual derivative $D_wT_\Lambda$ along the segment
between the two shifted OU means and apply Jensen before Gaussian
expectation. The displayed radial comparison holds at every segment
mean and (RCAP.7) then gives (RCAP.9).
The envelope is strictly decreasing beyond $r_0$, a region of
strictly positive Gaussian probability. Thus $K_G<K_0$.
No position/velocity factorization or field/noise independence is
used: the field is evaluated throughout at $x_1+tw$.
:::

:::{prf:lemma} Radial cancellation in the root's own velocity coefficient
:label: lem-rcap-radial-self-coefficient

For the actual pre-cap map in (RCAP.5), put
$$
S_w=\frac{\min\{\max\{V,C\},\,V/4+C\}}A.
$$
Then
$$
\|DC_V(Z_\Lambda(x_1,w))w\|\le S_w
\tag{RCAP.9a}
$$
for every $w$ and $|x_1|\le R_1$. In particular the derivative
of the incoming scalar coefficient $a$ acts on the root velocity
through this bounded radial quantity, rather than through $|w|$.
:::

:::{prf:proof}
The generic derivative estimate and coercivity give
$\|DC_V(Z)w\|\le V|w|/[V+(A|w|-C)_+]$, whose supremum is
$\max\{V,C\}/A$.
For the sharper alternative write
$Z=l(y)w+B(y)$, where $l=m-t\nu a(y)\ge A$ and
$B=-tx_1+t\nu M(y)$ has norm at most $C$.
The cap's radial derivative satisfies
$$
|DC_V(Z)Z|=\frac{V^2|Z|}{(V+|Z|)^2}\le\frac V4;
$$
the last inequality is $(V+|Z|)^2\ge4V|Z|$.
Therefore
$$
\|DC_V(Z)w\|
\le l^{-1}\big[|DC_V(Z)Z|+\|DC_V(Z)B\|\big]
\le(V/4+C)/A.
$$
Taking the smaller bound proves (RCAP.9a).
:::

:::{prf:lemma} Capped sensitivity to two distinct actual providers
:label: lem-rcap-own-provider-b2

Let $\Lambda,\widetilde\Lambda$ be two actual joint population
stage laws with absolute velocity moments at most $M_w$, and use
their own fields $(a,M)$ and $(\widetilde a,\widetilde M)$.
For $|x_1|\le R_1$,
$$
|T_\Lambda(x_1,w)-T_{\widetilde\Lambda}(x_1,w)|
\le t\nu\left[
 \|M-\widetilde M\|_\infty+
 S_w\|a-\widetilde a\|_\infty\right].
\tag{RCAP.10}
$$
Consequently, for two compared root first-stage outputs with
$|x_1|,|\widetilde x_1|\le R_1$ and shared OU innovation,
$$
\begin{aligned}
\big(\mathbb E|\Delta v^+|^2\big)^{1/2}
\le{}&K_G|\Delta\overline w|+K_X|\Delta x_1|\\
&+t\nu\left[
 \|\Delta M\|_\infty+S_w\|\Delta a\|_\infty
\right],
\end{aligned}
\tag{RCAP.11}
$$
where $\overline w$ denotes the root OU mean and both kernels retain
their own actual second providers.
:::

:::{prf:proof}
Interpolate the two joint provider laws by their convex mixtures.
Every interpolation still has absolute velocity moment at most $M_w$
and the same coercivity $|Z|\ge A|w|-C$. Its pre-cap variation at
fixed $(x_1,w)$ is $t\nu[\Delta M(x_1+tw)-w\Delta a(x_1+tw)]$.
After the cap, the multiplier of $|\Delta M|$ is at most one.
The product of the derivative with $w\Delta a$ is bounded by
$S_w|\Delta a|$ by (RCAP.9a), uniformly along the actual
provider interpolation.
Integrating along the provider interpolation proves (RCAP.10).
Now change the OU mean at fixed first-drift position using
(RCAP.9), change first-drift positions along their convex ball
using (RCAP.7), and change the second provider using (RCAP.10).
Minkowski gives (RCAP.11). The interpolated provider is used only
to differentiate the comparison; neither own kernel is replaced.
:::

(sec-rcap-default-envelope)=
## 4. A concrete reference envelope through all preparation sources

:::{prf:corollary} Strict reference cap sensitivity for alive and revived roots
:label: cor-rcap-default-numeric

Use the actual default source box and $\sigma_J=1/10$.
On the event that the root recipient jitter has norm at most one,
regardless of whether the root persists or revives, put
$$
R_X=2\sqrt3+1,\qquad R_1=mR_X+tV_c,\qquad
M_w=cV_c+ct(2\sqrt3+\sigma_JG_1)+qG_1,\quad V_c=4.
$$
The full actual provider satisfies this $M_w$ bound without any
jitter truncation or entering dead-position moment hypothesis.
The constants (RCAP.6)--(RCAP.11) satisfy the rigorous bounds
$$
K_0<0.99992,\qquad K_G<0.998,\qquad K_X<0.036,\qquad
S_w<0.622.
\tag{RCAP.12}
$$
The conditional root OU means and first-drift positions remain their
actual values, and the fitness/conditional-alive normalizers need not
be equal between the laws.
:::

:::{prf:proof}
Every selected source lies in $D$. Hence its unconditional prepared
position moment obeys
$\mathbb E|X|\le2\sqrt3+\sigma_JG_1$, including mandatory
revival of roots with arbitrarily distant retained dead positions.
The first count velocity is a convex average of collision velocities
and is bounded by $V_c$. Therefore the actual joint provider obeys
$\Lambda_2|w|\le M_w$.
On the root good-jitter event $|X|\le R_X$, so $|x_1|\le R_1$.

The rational bounds $c<0.9608$, $q<0.198$, $\ell<0.607$
and $\sqrt3<1.733$, together with $G_1\le\sqrt3$, yield
$M_w<4.26$ and $R_1<4.545$.
Thus $A=0.9936$, $C<0.117$, $r_0<0.118$ and
$\alpha<0.00007284$.
They imply
$$
K_0<0.9996+0.00007284(4.378)<0.99992,
$$
$$
K_X<\max\{0.02+0.003642(4.378),\
                       2(0.003642)/0.9936\}<0.036,
$$
and $C<V$, $S_w\le(0.5+0.117)/0.9936<0.622$.
Also $K_0>V\alpha/A$, as required.

For a fully rational Gaussian-gap upper bound, $c<0.9608$ gives
$q>0.196>r_0$. At radius $q$, (RCAP.6) gives
$$
\mathcal K(q)
<
\frac{2[0.9996+0.00007284(4.26+0.198)]}
     {2+0.9936(0.196)-0.117}
<0.963.
$$
The standard Gaussian event $|\xi_1|\ge1$ has probability at least
$1/16$: integrating its density over the two intervals $[1,2]$
and $[-2,-1]$ gives the lower bound
$2e^{-2}/\sqrt{2\pi}>1/16$.
Since the radial envelope is nonincreasing,
$$
K_G^2\le K_0^2-\frac1{16}[K_0^2-\mathcal K(q)^2]
<0.99992^2-\frac1{16}(0.99992^2-0.963^2)<0.998^2.
$$
This proves every bound in (RCAP.12). Direct evaluation of the
exact radial integral in (RCAP.8) gives approximately $0.9152$
for $K_G$ at the exact default envelopes; this diagnostic value is
not substituted into any proof.
:::

:::{prf:corollary} Complete localized root sensitivity retaining both own kicks
:label: cor-rcap-own-kinetic-local

Couple two actual prepared roots $(X,v)$ and
$(\widetilde X,\widetilde v)$, with their own first-provider fields
$C_0,\widetilde C_0$ and their own joint second providers.
On the paired good-jitter event, both have the preceding bounds.
Put
$$
U=v+t\nu C_0(X,v),\quad
\widetilde U=\widetilde v+t\nu\widetilde C_0(\widetilde X,\widetilde v).
$$
Under shared root OU and final Gaussian innovations,
$$
\Delta x^+=a_x\Delta X+b\Delta U,\quad
\Delta x_1=m\Delta X+t\Delta U,\quad
\Delta\overline w=c(\Delta U-t\Delta X).
\tag{RCAP.13}
$$
Conditionally on these prepared roots, their actual capped velocity
differences satisfy (RCAP.11), with the constants (RCAP.12).
This statement uses both own providers rather than a common frozen
provider, but leaves their field differences explicitly on the right.
:::

:::{prf:proof}
The identities are the actual first kick, harmonic drift and OU
mean identities, with corresponding innovations shared. Substitute
them into (RCAP.11). The first provider differences have not been
removed; they are exactly contained in $\Delta U$. The second
provider differences remain the two supremum norms in that lemma.
Recipient jitters and the component collision precede the independent
OU innovations, so conditioning on the actual prepared pair does not
condition its OU array.
:::

(sec-rcap-block-identity)=
## 5. An exact cap-aware two-update balance and the surviving cases

:::{prf:lemma} Cap-residual balance across two actual discrete updates
:label: lem-rcap-two-update-balance

Use any valid shared-innovation coupling of two actual finite count
kinetic updates. Write the entering differences at update $j$ as
$(r_j,\zeta_j)$, the actual uncapped output differences as
$(R_j,Z_j)$, and the cap residual differences as
$E_j=Z_j-\zeta_{j+1}$. Then $r_{j+1}=R_j$.
For any $a>0$, $\beta^2<a$, let
$Q_{a,\beta}(r,\zeta)=a\|r\|_N^2+
2\beta\langle r,\zeta\rangle_N+\|\zeta\|_N^2$.
Pointwise on every Gaussian outcome,
$$
\begin{aligned}
Q_{a,\beta}(r_2,\zeta_2)-Q_{a,\beta}(r_0,\zeta_0)
\le\sum_{j=0}^1\big\{&
 Q_{a,\beta}(R_j,Z_j)-Q_{a,\beta}(r_j,\zeta_j)\\
&+\beta^2\|R_j\|_N^2-\|E_j+\beta R_j\|_N^2\big\}.
\end{aligned}
\tag{RCAP.14}
$$
Both noisy second graphs in this identity are their actual own graphs.
If an actual preparation is inserted before either kinetic update,
the same statement holds after adding its exact change in
$Q_{a,\beta}$ to the corresponding summand.
:::

:::{prf:proof}
Apply (RCAP.2) to every row and sum:
$\|Z_j\|_N^2-\|\zeta_{j+1}\|_N^2\ge\|E_j\|_N^2$.
Therefore
$$
\begin{aligned}
Q_{a,\beta}(R_j,\zeta_{j+1})
-Q_{a,\beta}(R_j,Z_j)
&\le-\|E_j\|_N^2-2\beta\langle R_j,E_j\rangle_N\\
&=\beta^2\|R_j\|_N^2-\|E_j+\beta R_j\|_N^2.
\end{aligned}
$$
Add the two updates and telescope their stored phase costs.
With an intervening preparation, telescope through its entering
prepared phase as well; this introduces exactly its own cost change.
No sign or smallness of that preparation cost is assumed.
:::

:::{prf:remark} All-outcome interface and first remaining estimate
:label: rem-rcap-own-block-residual

The whole-update certificate (RCAP.3) removes the previous
standalone-cap inference from the nonviscous baseline. The reference
viscous B2 estimates (RCAP.7)--(RCAP.13) supply a strict actual
Gaussian-averaged velocity sensitivity and bounded own-provider
sensitivity. The two-update identity (RCAP.14) retains the cap's
nonnegative residual rather than discarding its cross contribution.
These are complete discrete intermediate results.

The two preparation outcomes remain explicit. On the paired
good-jitter event the proved constants apply to every persistent,
cloned and revived source, including equal-fitness regions.
Outside it, the full actual Gaussians still run. In any coupling,
the event has probability at most
$p_{\rm bad}\le2\Pr\{|\sigma_JZ|>1\}$ and capped velocity
differences obey the unconditional RMS bound
$2V\sqrt{p_{\rm bad}}$ on that event. Position differences have
RMS bound there at most
$$
[2a_x\sqrt dL+2bV_c]\sqrt{p_{\rm bad}}
 +2a_x\sigma_JG_4^{1/4}p_{\rm bad}^{1/4}.
$$
Indeed the two source positions lie in $D$, each first count
velocity is bounded by $V_c$, corresponding OU/final noises cancel,
and Cauchy--Schwarz bounds each jitter's second moment restricted to
the bad event by its fourth moment times $p_{\rm bad}^{1/2}$.
This finite tail bound is not proportional to the entering
difference, and therefore does not close an exact contraction.
Its positive value cannot be deleted on the ground that it is small.

The precise remaining nonlinear estimate is a signed combination of
the two first-stage differences and two actual joint provider
differences, plus a proportional bad-jitter/preparation account, that
makes the right side of (RCAP.14) strictly negative in an admissible
law cost at $\nu=.3$. The current absolute triangle bound from
(RCAP.11) loses the restoring-force sign; a strict bound on its
velocity coefficient alone does not establish a phase-law block gap.
The research 27 signed continuous-field inequalities are constraints
on that calculation, rather than a discretization theorem.

For the marked default-box law there is a further own-provider
account. A proved positive alive-mass floor does not prove small
dead mass. Mandatory revival leaves and the conditional-alive
normalization can therefore contribute feedback proportional to
the dead mass, independently of small positive fitness powers.
The large-box theorem absorbs that term only in its explicitly
proved small-dead invariant class; it does not certify the default
$L=2$ class. Default marked closure still needs a proved tail/burn-in
bound strong enough to absorb that feedback, or a stronger delayed
block estimate that includes it. The good-jitter cap inequalities
above do not discharge this account.

Terminal marking introduces another real outcome: a small physical
position change can change $1_D(x)$. The actual final Gaussian can
control that boundary probability, but the Euclidean cost alone does
not. Current-survival normalization also changes the distribution of
past innovations. Its law comparison requires the actual
recent-window tilt estimate; a common unconditioned innovation
coupling is not automatically a coupling of two current-survivor
laws. None of these normalizers is replaced here by a constant or a
different algorithm.

Result kind: productive discrete reduction with an explicit
all-outcome residual, not full own-provider reference convergence and
not a negative theorem about alive-law transport. Evidence strength:
the displayed lemmas are proved in the supplied arguments; the
numerical radial-integral value is diagnostic only.
:::
