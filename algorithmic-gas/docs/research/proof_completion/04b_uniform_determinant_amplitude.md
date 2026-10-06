# Population-uniform native B2 determinant amplitude

(sec-uda-record)=
## 1. Complete record and derived parameter tests

:::{prf:definition} Native amplitude execution and parameter ledger
:label: def-uda-complete-parameters

Retain the full execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record` and the entire parameter
ledger of {prf:ref}`def-ngf-parameter-ledger`. The present family changes
only $N\ge4$ and the existing count/row bit. Every other dynamical parameter
and every consumed readout/calibration parameter is fixed. It is the unchanged
reference family of {prf:ref}`def-cgd-existing-reference`, including

$$
\begin{gathered}
h=.04,\quad\gamma=b_O=1,\quad \sigma_J=\sigma_x=.1,\quad
V=2,\quad\alpha_{\rm col}=.5,\quad \nu=.3,\quad\rho=1,\\
U(x)=\lambda|x|^2/2,\quad\lambda=1,\quad R=-U,\quad D=[-2,2]^3.
\end{gathered}
$$

All donor, fitness, acceptance, feature, reducer and boundary parameters in
that reference remain exactly their configured values. No donor history,
curl, geometry feedback or intermediate boundary classification is enabled.
The collision is the original simultaneous frozen-source copying and
one-Haar-matrix connected-component collision, including mandatory revival
of every dead row. Terminal marking retains all dead coordinates, and the
configured velocity cap is applied after the actual B2 kick. Geometry and
color recording remain passive. Finite-precision/fixed-stream execution is
retained as a different execution convention and receives no stochastic
variance conclusion from these real-coordinate independent-Gaussian proofs.

For each $N$, $\nu_N$ and $\alpha_N\in(0,1)$ are the actual full killed-kernel
QSD and survival eigenvalue proved in
{prf:ref}`thm-cgd-finite-n-qsd`. Its count/row coercivity tests are
$1-t^2\lambda-t\nu>0$ and $1-t^2\lambda-2t\nu>0$, with $t=h/2$.
The reference gives $.9936$ and $.9876$; its other canonical hypotheses
are verified in {prf:ref}`cor-cgd-reference-qsd`. These tests concern the
configured maps rather than an assumed QSD. All initial-law dependence is
retained when forming the QSD; the statement here concerns this identified
incoming QSD and its one-step recorded selected law
$\mathbb P_N^{\rm sel}=\alpha_N^{-1}\nu_N Q_N$.

Set

$$
\begin{gathered}
c=e^{-\gamma h},\quad
q=b_O\sqrt{(1-e^{-2\gamma h})/(2\gamma)},\quad
s=\sigma_x\sqrt h,\quad R_c=(1+2|\alpha_{\rm col}|)V,\\
B_D=\sup_{x\in D}|x|=2\sqrt3,\quad
R_v=R_c(1+2t\nu),\quad \kappa=m\ell_0/\hbar_{\rm eff}.
\end{gathered}
$$

At $\gamma=0$ the existing convention is $q=b_O\sqrt h$.
The color uses the actual B2 force-input velocity, original Gaussian
normalization bit and configured threshold $\delta_c$. It is the existing
same-stage B2 instrument of {prf:ref}`def-variant-recorded-color-geometry`,
with colors and positions belonging to the post-revival B2 population.
Define

$$
b_N=\det[C_1^{\rm B2},C_2^{\rm B2},C_3^{\rm B2}],\qquad
b_N^{\rm alive}=b_N\prod_{i=1}^3\mathbf1_D(Y_i),
$$

where $Y$ is the actual terminal position record and each color has its
original force-threshold zero extension. These are full complex determinant
coordinates already retained in the native hierarchy. An additional
pre-clone-frame mask deleting rows that accepted clones in this step defines
a different observable and is not covered: the construction below deliberately
uses mandatory revival. B1/default batch readers, capped-phase PF/RO
alignments, and extra localization or normalized averages retain their actual
maps and do not inherit this amplitude bound merely by sharing a channel name.
:::

:::{prf:definition} Explicit analysis radii and finite profiles
:label: def-uda-analysis-profiles

The following radii specify events in the original unbounded Gaussian laws;
they are not truncations or changes of the algorithm:

$$
a=.01,\quad M=80,\quad r=.1,\quad J_0=8,\quad G_0=12,
\quad T_0=5\cdot10^{-10},\quad a_\zeta=.1.
$$

For a three-dimensional standard normal $H$, set

$$
\begin{aligned}
p_3(u)&=\mathbb P(|H|>u)
=\operatorname{erfc}(u/\sqrt2)+\sqrt{2/\pi}\,u e^{-u^2/2},\\
m_3(u)&=\mathbb E[|H|\mathbf1_{|H|>u}]
=\sqrt{2/\pi}(u^2+2)e^{-u^2/2},\qquad
\bar m_3=2\sqrt{2/\pi}.
\end{aligned}
$$

Compute from the actual primitive parameters

$$
\begin{gathered}
M_0=c(R_v+t\lambda B_D),\quad \sigma_m=ct\lambda\sigma_J,\quad
B=M_0+\sigma_m J_0+qG_0,\\
X_b=|1-t^2\lambda|(B_D+\sigma_JJ_0)+tR_v,\quad Y_b=X_b+tB,\\
X_T=|1-t^2\lambda|a+tR_v,\quad
Y_T=X_T+t(M+r),\quad
k_*=e^{-(Y_b+Y_T)^2/(2\rho^2)},\\
E_{\rm bad}=(M_0+q\bar m_3)p_3(J_0)+\sigma_m m_3(J_0)
 +(M_0+\sigma_m\bar m_3)p_3(G_0)+q m_3(G_0),\\
p_G=1-2[p_3(J_0)+p_3(G_0)]-E_{\rm bad}/T_0,\\
N_0=\max\{6,\lceil80(M+r)/k_*\rceil\},\quad
\varepsilon_L=\frac{2}{M}[B+r+4T_0/k_*+.1],\quad
b_L=1-3\varepsilon_L,\\
f_L=\frac{\nu k_*}{4}[M-B-r-4T_0/k_*-.1].
\end{gathered}
$$

The large-population positive regime is the derived test
$p_G>0$, $b_L>0$, $f_L>\delta_c$, and $Y_T+a_\zeta<2$.
For the finite-population route define additionally

$$
\begin{gathered}
k_b=e^{-(2X_T+tM)^2/(2\rho^2)},\quad
b_0=[k_b/(6\sqrt3)]^3,\quad f_0=\nu M k_b/4,\\
H_0=M+1,\quad k_R=e^{-(2X_T+2tH_0)^2/(2\rho^2)},\quad
L_0=\nu[2+8t(e^{-1/2}/\rho)H_0/k_R],\quad Q_0=2L_0/f_0+|\kappa|,\\
r_0=\min\{r,(f_0-\delta_c)/(2L_0),b_0/(6Q_0)\}.
\end{gathered}
$$

Its positive regime is $f_0>\delta_c$, giving $r_0>0$, together with
the same terminal-position test. These formulas retain every consumed
parameter. In other configurations a failed test gives no certificate from
this route; it is not a license to change its noise, force, cap or threshold.
:::

:::{prf:lemma} Verification of the unchanged reference margins
:label: lem-uda-reference-margins

Both existing count and row choices at the unchanged reference, with the
configured threshold $\delta_c=10^{-12}$, satisfy all preceding tests.
The following conservative enclosures suffice:

$$
\begin{gathered}
.19606<q<.19608,\quad B<6.325,\quad X_T<.091,
\quad Y_b<4.471,\quad Y_T<1.694,\\
k_*>5.5\cdot10^{-9},\quad E_{\rm bad}<3.6\cdot10^{-13},
\quad p_G>.9992,\quad \varepsilon_L<.173,\\
b_L>.48,\quad f_L>3\cdot10^{-8},\quad k_b>.20,
\quad f_0>1.2,\quad b_0>7\cdot10^{-6}.
\end{gathered}
$$

The threshold $N_0$ is finite; numerical evaluation is approximately
$1.13234\cdot10^{12}$. It is an analytic certificate threshold for the
already defined real-coordinate kernels, beyond the practical walker and
resource limits of present execution configurations. The reference $N=200$
uses the finite-population certificate, retaining its actual threshold.
:::

:::{prf:proof}
For $u>0$, integration by parts bounds the Gaussian scalar tail by
$\operatorname{erfc}(u/\sqrt2)\le\sqrt{2/\pi}\,e^{-u^2/2}/u$.
Thus $p_3(u)\le\sqrt{2/\pi}(u+1/u)e^{-u^2/2}$.
Integration of the radial density
$\sqrt{2/\pi}\,r^2e^{-r^2/2}$ gives the displayed exact $m_3$ formula.
Use $\sqrt{2/\pi}<.8$,
$e^{-32}<1.267\cdot10^{-14}$ and $e^{-72}<5.381\cdot10^{-32}$.
Then $p_3(8)<8.236\cdot10^{-14}$,
$m_3(8)<6.690\cdot10^{-13}$,
$p_3(12)<5.202\cdot10^{-31}$ and
$m_3(12)<6.286\cdot10^{-30}$.
Substitution of $c=e^{-.04}$, $t=.02$, $R_c=4$, $\lambda=1$,
$B_D=2\sqrt3$ and $\sigma_J=.1$ in the exact profiles gives the
stated bounds on $B,X_T,Y_b,Y_T,E_{\rm bad}$ and $k_*$.
The bounds on $p_G,\varepsilon_L,b_L,f_L$ then follow directly; for
example $4T_0/k_*<.364$ and
$2(6.325+.1+.364+.1)/80<.173$.

All decimal exponential bounds in this calculation can be obtained using
the positive series for $e^u$: its partial sum is a lower bound, and after
an index $m+1>u$ its remaining terms are bounded by the first omitted
term times $1/[1-u/(m+2)]$. Taking reciprocals provides both-sided bounds
for $e^{-u}$; $m=256$ suffices for the displayed arguments $u\le72$.
The simple bounds $3.46410<2\sqrt3<3.46411$ follow by squaring.
In particular $2X_T+tM<1.782$ gives $k_b>.20$;
$\nu M k_b/4>1.2$ and
$[.20/(6\sqrt3)]^3>7\cdot10^{-6}$.
Finally $Y_T+a_\zeta<1.794<2$, and both force margins are strictly
above the actual $10^{-12}$ threshold. The normalization bit does not
enter these conservative pointwise margins.
:::

(sec-uda-qsd-prefix)=
## 2. Population-independent tagged revival under the actual QSD

:::{prf:lemma} Uniform preparation moments and QSD status-cylinder support
:label: lem-uda-qsd-tagged-status

For either existing normalization, every one-step preparation from a
nonextinct capped input obeys the population-independent bound

$$
\mathbb E|x_{2,i}|^2\le C_X^2,\qquad
C_X=|1-tb\lambda|(B_D+\sigma_J\sqrt3)+bR_v+tq\sqrt3,
\quad b=t(1+c).
\tag{UDA.1}
$$

For $B_p>0$ define

$$
d_0(B_p)=\frac{4\pi a^3/3}{(2\pi s^2)^{3/2}}
 e^{-(B_p+a)^2/(2s^2)},\qquad
d_D(B_p)=\frac{4\pi a^3/3}{(2\pi s^2)^{3/2}}
 e^{-(B_p+3+a)^2/(2s^2)}.
$$

The actual QSD has the uniform lower bound

$$
\nu_N\{1,2,3\text{ dead};\ x_4\in B(0,a),\ 4\text{ alive}\}
\ge p_A:=\tfrac12d_D(\sqrt8 C_X)^3d_0(\sqrt8 C_X)>0.
\tag{UDA.2}
$$

For every fixed $N\ge4$ the stronger cylinder in which **every** row except
4 is dead and $x_4\in B(0,a)$ has lower bound

$$
p_{A,N}=\tfrac12 d_D(\sqrt{2N}C_X)^{N-1}
 d_0(\sqrt{2N}C_X)>0.
\tag{UDA.3}
$$

These are actual status strata; no continuous-gradient inequality or
population mixing estimate is used.
:::

:::{prf:proof}
Revival and copying take all pre-jitter positions from actual alive donors
in $D$. An unchanged alive row also lies in $D$. With independent preassigned
standard Gaussian jitters $J_i$, the resulting position satisfies
$|x_i^c|\le B_D+\sigma_J|J_i|$ whether or not that row uses its jitter.
Unused preassigned random variables may be integrated out; their addition
to this probability representation does not change the executed kernel.
The original component collision gives $|v_i^c|\le R_c$.
Each complete Gaussian viscosity, at either bit, is bounded by
$2\nu R_c$ at B1 because it is a nonnegative weighted mean of velocity
differences with total coefficient at most $\nu$.
Writing $r_i^c=v_i^c+tF_i^{\rm visc,B1}$ gives $|r_i^c|\le R_v$,
and the actual first kick/drift, O and A2 satisfy

$$
v_{1,i}=r_i^c-t\lambda x_i^c,\quad
x_{1,i}=(1-t^2\lambda)x_i^c+tr_i^c,\quad
x_{2,i}=(1-tb\lambda)x_i^c+br_i^c+tq\xi_i.
$$

Minkowski's $L^2$ inequality and
$\mathbb E|J_i|^2=\mathbb E|\xi_i|^2=3$ prove (UDA.1). This argument
retains the uncapped OU innovation; the terminal cap occurs later.

By a union bound and (UDA.1), the four preparation means
$x_{2,1},\ldots,x_{2,4}$ are all bounded by $\sqrt8 C_X$ with probability
at least $1/2$. Conditional on the complete preparation, the actual terminal
position Gaussians are independent. The probabilities of placing rows
1--3 in $B(3e_1,a)$ and row 4 in $B(0,a)$ are at least the corresponding
$d_D$ and $d_0$ density-times-volume bounds. The first three balls lie
strictly outside $D$, and the fourth lies strictly inside it. This event
therefore has the exact indicated marks and ensures nonextinction,
independently of all other output rows. Its probability is at least $p_A$
from every nonextinct capped input.

The QSD eigenidentity $\nu_N=\alpha_N^{-1}\nu_N Q_N$, with
$\alpha_N\le1$, transfers this raw probability lower bound directly to
the QSD. Its input cap support follows from the same eigenidentity and the
original cap applied by the kernel. Replacing four rows by all $N$ rows
and using $\sqrt{2N}C_X$ gives the preparation event probability $1/2$.
Its terminal Gaussian product event puts every row except 4 outside and
row 4 inside, proving (UDA.3). No independence of preparations or of
the QSD coordinates has been invoked.
:::

:::{prf:lemma} Tagged mandatory-revival event with untouched bulk Gaussian laws
:label: lem-uda-tagged-revival

On the actual incoming event in (UDA.2), the original mandatory revival
copies alive donors into rows 1--3 and applies their existing independent
$\sigma_J>0$ jitters. Conditional on every preceding donor/gate choice,
the probability of

$$
|x_1^c|,|x_2^c|,|x_3^c|<a
$$

is at least $p_J^3$, where

$$
p_J=\frac{4\pi a^3/3}{(2\pi\sigma_J^2)^{3/2}}
e^{-(B_D+a)^2/(2\sigma_J^2)}>0.
\tag{UDA.4}
$$

Conditioning on these three jitter events leaves every non-tagged jitter
at its original independent Gaussian law. Component collision randomness
and its actual dependency are retained; the deterministic bounds
$|v_i^c|\le R_c$ and $|r_i^c|\le R_v$ still hold for every realization.
For the cylinder in (UDA.3), the probability that every post-clone position
lies in $B(0,a)$ is at least $p_J^{N-1}$.
:::

:::{prf:proof}
Every selected donor is alive and has position in $D$. Conditional on all
donor/gate choices, each required recipient position is that fixed donor
position plus its own independent $\sigma_J$ Gaussian. The minimum Gaussian
density on $B(0,a)$ is at least the density envelope in (UDA.4).
Multiplication proves the three-row event. Those events use only the three
recipient jitters. Other jitters retain their laws; no conditioning on the
later tagged OU velocities has occurred.

In the all-but-4-dead cylinder, every dead row copies the sole alive row 4.
Its position lies in $B(0,a)$. The singleton companion convention and equal
recipient/donor fitness make its own cloning gate probability zero, so the
position of row 4 remains there. The independent recipient jitter argument
on the remaining $N-1$ rows gives the final claim. The original connected
component collision affects velocities and retains its uniform cap-derived
bound; it does not change these post-clone positions.
:::

(sec-uda-large-population)=
## 3. Tagged Gaussian events with a bulk tail budget

:::{prf:lemma} Bulk tail event under the actual coupled first kick
:label: lem-uda-bulk-tail-event

After conditioning on the incoming tagged status event and the three
post-clone position balls in the preceding lemma, retain all original
component collisions and B1/A1 outputs. For non-tagged rows
$\mathcal B=\{4,\ldots,N\}$ set

$$
\mathrm{bad}_j=\{|J_j|>J_0\}\cup\{|\xi_j|>G_0\},\qquad
T_{\rm bad}=N^{-1}\sum_{j\in\mathcal B}|z_j|\mathbf1_{\mathrm{bad}_j}.
$$

Before drawing the tagged OU innovations, the event

$$
\mathcal G=\left\{\sum_{j\in\mathcal B}\mathbf1_{\mathrm{bad}_j}
 \le(N-3)/2,\quad T_{\rm bad}\le T_0\right\}
$$

has conditional probability at least $p_G$. Every good bulk row satisfies
$|z_j|\le B$ and $|x_{2,j}|\le Y_b$. Every tagged row satisfies
$|x_{1,i}|\le X_T$, $|cv_{1,i}|\le M_T=c(R_v+t\lambda a)$.
All statements hold for both normalizations, without independent
post-B2 rows or an all-population bounded-innovation event.
:::

:::{prf:proof}
The preparation bounds give, pointwise for each bulk row,

$$
|z_j|\le M_0+\sigma_m|J_j|+q|\xi_j|,
\qquad
|x_{1,j}|\le |1-t^2\lambda|(B_D+\sigma_J|J_j|)+tR_v.
$$

Each bound is valid even though the component collision and B1 force depend
on the entire jitter array. The remaining $J_j$ are independent standard
Gaussians. Their subsequent O innovations $\xi_j$ are fresh independent
standard Gaussians conditional on the entire preparation. Integrating the
pointwise envelope therefore gives
$\mathbb P(\mathrm{bad}_j)\le p_3(J_0)+p_3(G_0)$ and
$\mathbb E[|z_j|\mathbf1_{\mathrm{bad}_j}]\le E_{\rm bad}$.
For the latter, split the union into its two events and integrate the
four tail/mean terms from the envelope; these are exactly the terms in
the definition of $E_{\rm bad}$.
No output independence is needed to sum their expectations.

Markov's inequality bounds the bad-count failure by
$2[p_3(J_0)+p_3(G_0)]$ and the bad-velocity-mass failure by
$E_{\rm bad}/T_0$. Subtraction gives $p_G$.
The good-row bounds and the tagged preparation bounds follow by direct
substitution. The event $\mathcal G$ is evaluated using non-tagged OU
innovations only. Its probability estimate thus precedes any tagged OU
conditioning, which would otherwise reweight the bulk preparation through
the tagged means.
:::

:::{prf:theorem} Uniform large-population determinant amplitude event
:label: thm-uda-large-n-amplitude

For $N\ge N_0$, retain the original three tagged OU events
$|z_i-Me_i|<r$ and the three terminal Gaussian events
$|s\zeta_i|<a_\zeta$. Define

$$
p_O(u)=\frac{4\pi u^3/3}{(2\pi q^2)^{3/2}}
e^{-(M+M_T+u)^2/(2q^2)},\qquad
p_\zeta=\mathbb P(|H|<a_\zeta/s)>0.
$$

Under the derived reference tests there is an actual joint event of
incoming QSD states and one complete original update on which all three
tags are terminal alive, every tagged B2 force exceeds $\delta_c$, and

$$
|b_N^{\rm alive}|\ge b_L>.48.
$$

Its raw joint probability is at least

$$
p_L=p_A p_J^3 p_G p_O(r)^3 p_\zeta^3>0,
\tag{UDA.5}
$$

independently of $N\ge N_0$. Consequently
$\mathbb E_{\mathbb P_N^{\rm sel}}|b_N^{\rm alive}|^2\ge p_Lb_L^2$.
:::

:::{prf:proof}
First realize the incoming QSD tagged cylinder, its three recipient jitter
balls, and the bulk event $\mathcal G$ in this order. Their probabilities
are bounded below by $p_A,p_J^3,p_G$.
Conditional on the resulting entire preparation and bulk O innovations,
the three tagged O innovations are independent original Gaussians with
means bounded by $M_T$. The density envelope gives the probability
$p_O(r)^3$ of their three target balls. This ordering preserves the bulk
tail budget despite the coupled tagged means.

On these events at least $(N-3)/2\ge N/4$ bulk rows are good.
For a tagged row write its actual count-denominator-normalized degree and
velocity numerator as

$$
a_i=N^{-1}\sum_{j\ne i}K_{ij},\qquad
m_i=N^{-1}\sum_{j\ne i}K_{ij}z_j.
$$

The distances to good bulk rows are at most $Y_b+Y_T$, so
$a_i\ge k_*/4$. Their numerator contribution has norm at most $Ba_i$.
Bad bulk rows contribute at most $T_0$ because $K\le1$.
The other two tags contribute at most $2(M+r)/N$.
Therefore

$$
|m_i/a_i|\le B+4T_0/k_*+8(M+r)/(Nk_*)
\le B+4T_0/k_*+.1.
\tag{UDA.6}
$$

The two actual force formulas are respectively
$F_i^{\rm count}=\nu a_i(m_i/a_i-z_i)$ and
$F_i^{\rm row}=\nu(m_i/a_i-z_i)$; their directions agree.
Their distance before normalization from $-Me_i$ is at most
$B+r+4T_0/k_*+.1$.
The elementary direction estimate gives
$|F_i/|F_i|+e_i|\le\varepsilon_L$.
The count force magnitude is at least $f_L$; the row magnitude is at
least $\nu[M-B-r-4T_0/k_*-.1]\ge f_L$, since $k_*/4\le1$.
They strictly exceed the configured threshold.

After componentwise phases, each unit color is within $\varepsilon_L$
of $-e_i e^{i\kappa z_i^i}$. These three comparison columns are orthonormal
regardless of $\kappa$. Multilinearity and the unit-column bound give
$|b_N|\ge1-3\varepsilon_L=b_L$; no replacement of the complex determinant
by a real Gram determinant has occurred.

The actual terminal positions of the three tags have norm at most
$Y_T+a_\zeta<2$ on their independent final-noise balls. Thus all are
alive, the zero-extended determinant equals $b_N$, and nonextinction holds.
Their conditional probability is $p_\zeta^3$, independent of the
already retained events. Multiplication proves (UDA.5).
The selected law divides the joint raw probability on nonextinction by
$\alpha_N\le1$ and thus only increases this lower bound.
:::

(sec-uda-finite-population)=
## 4. All remaining finite populations with the configured threshold

:::{prf:theorem} Finite-population complex determinant certificate
:label: thm-uda-finite-n-amplitude

For every $N\ge4$ there is an actual QSD/update event with all tags terminal
alive and $|b_N^{\rm alive}|\ge b_0/2$. Its probability is at least

$$
p_{S,N}=p_{A,N}p_J^{N-1}p_O(r_0)^3p_{O,0}(r_0)^{N-3}p_\zeta^3>0,
\tag{UDA.7}
$$

where

$$
p_{O,0}(u)=\frac{4\pi u^3/3}{(2\pi q^2)^{3/2}}
e^{-(M_T+u)^2/(2q^2)}.
$$

In particular the unchanged $N=200$ reference, under either existing
normalization and the actual $\delta_c=10^{-12}$, satisfies
$\mathbb E_{\mathbb P_{200}^{\rm sel}}|b_{200}^{\rm alive}|^2
\ge p_{S,200}b_0^2/4>0$.
The finite Gaussian product event is used only to bound the finitely many
$4\le N<N_0$ in the uniform theorem below.
:::

:::{prf:proof}
Choose the actual all-but-4-dead QSD cylinder from (UDA.3) and the all-revived
near-origin position event of probability $p_J^{N-1}$. Retain its original
component collision. Now every actual A1 position has norm at most $X_T$,
and every O mean has norm at most $M_T$.

Consider first the B2 velocity center $z_i^*=Me_i$ for the three tags and
$z_j^*=0$ for every other row. Every tag-bulk kernel is at least $k_b$.
Let $A$ be the three-by-three principal matrix of the full B2 Laplacian:
$A_{ii}=\sum_{j\ne i}K_{ij}$ and $A_{ij}=-K_{ij}$ for distinct tags.
The tag-bulk degree $\beta_i=\sum_{j\ge4}K_{ij}$ is at least
$(N-3)k_b$. The matrix of unnormalized phased force rows is a positive
row scaling of the complex matrix $H$ with

$$
H_{ii}=A_{ii}e^{i\kappa M},\qquad H_{ij}=A_{ij}\quad(i\ne j).
$$

Its diagonal modulus exceeds its off-diagonal row sum by $\beta_i$.
For a vector $u$ take an index maximizing $|u_i|$. Then
$|Hu|_2\ge|Hu|_\infty\ge\min_i\beta_i\,|u|_\infty
\ge(N-3)k_b|u|_2/\sqrt3$.
Hence its minimum singular value is at least $(N-3)k_b/\sqrt3$.
Each unphased force row has norm at most
$A_{ii}+\sum_{j\ne i,\,j\le3}|A_{ij}|\le2(N-1)$.
The normalized color matrix, in either count or row normalization, is $H$
divided by precisely these row norms; its minimum singular value is at
least

$$
\frac{(N-3)k_b}{2\sqrt3(N-1)}\ge\frac{k_b}{6\sqrt3}.
$$

Taking the product of its three singular values gives the complex
determinant lower bound $b_0$. This argument is valid for every real
$\kappa$, including phases that would invalidate a real determinant
comparison. The center's count force own-coordinate magnitude is
$\nu M A_{ii}/N\ge\nu M k_b/4=f_0$.
For row normalization that own-coordinate magnitude is exactly $\nu M$,
so the same conservative $f_0$ applies.

Perturb every velocity in the product ball
$\max_j|z_j-z_j^*|<r_0\le1$. All velocities have norm at most $H_0$,
all A1 positions have norm at most $X_T$, and every B2 kernel is at least
$k_R$. A kernel changes by at most
$2t(e^{-1/2}/\rho)r_0$ from the center.
For row normalization the total variation of one normalized kernel row
is at most $4t(e^{-1/2}/\rho)r_0/k_R$; for count normalization no
denominator correction occurs. Splitting velocity and kernel changes gives
$|F_i(z)-F_i(z^*)|\le L_0r_0$ for both bits.
The force norms remain greater than $\delta_c$, and each color changes
by at most $Q_0r_0$. The determinant changes by at most
$3Q_0r_0\le b_0/2$, proving $|b_N|\ge b_0/2$ throughout the product ball.

Conditional on the entire preparation, the actual independent O Gaussian
probabilities of the three tag balls and remaining zero-centered balls
are bounded below by $p_O(r_0)^3p_{O,0}(r_0)^{N-3}$.
Tagged terminal positions and final-noise balls have the same $Y_T+a_\zeta<2$
bound as before; their probability is $p_\zeta^3$. This gives the actual
joint event (UDA.7). It implies survival, so division by $\alpha_N\le1$
proves the selected expectation assertion.
:::

(sec-uda-uniform-result)=
## 5. Uniform selected amplitude and native scaled covariance

:::{prf:theorem} Population-uniform reference determinant amplitude
:label: thm-uda-uniform-selected-amplitude

Keep every parameter of the unchanged reference fixed, including its
actual finite $\kappa$, either count/row bit, and $\delta_c=10^{-12}$.
For the already defined real-coordinate full killed kernels at every
$N\ge4$ for which the complete execution record defines the family, put
$B_{\max}=\sqrt{2N_0}C_X$ and

$$
\begin{aligned}
C_S={}&\tfrac12d_D(B_{\max})^{N_0}d_0(B_{\max})
 p_J^{N_0}p_O(r_0)^3p_{O,0}(r_0)^{N_0}p_\zeta^3\,b_0^2/4,\\
C_L={}&p_Ap_J^3p_Gp_O(r)^3p_\zeta^3\,b_L^2,\qquad
C_{\rm amp}=\min\{C_S,C_L\}>0.
\end{aligned}
\tag{UDA.8}
$$

Then the actual one-step survival-selected native determinant with
terminal-alive mask satisfies

$$
\inf_{N\ge4}\mathbb E_{\mathbb P_N^{\rm sel}}
 |b_N^{\rm alive}|^2\ge C_{\rm amp}>0.
\tag{UDA.9}
$$

The raw B2 amplitude under incoming QSD, without terminal-alive masking,
also obeys

$$
\inf_{N\ge4}\mathbb E_{\nu_N\text{ and raw update}}|b_N|^2
\ge C_{\rm amp}>0.
\tag{UDA.10}
$$

The certificate is uniform and explicitly parameterized but astronomically
small. It concerns the existing analytic kernels, and supplies no claim
that the large-$N_0$ executions fit actual memory/resource limits or that
finite arithmetic reproduces this positive probability.
:::

:::{prf:proof}
For $N\ge N_0$, {prf:ref}`thm-uda-large-n-amplitude` proves the lower
bound $C_L$ for both the raw joint surviving event and selected law.
For $4\le N<N_0$, {prf:ref}`thm-uda-finite-n-amplitude` gives its explicit
finite product lower bound. Its preparation radius is at most $B_{\max}$,
and $d_D,d_0$ decrease with that radius. Every density-times-volume lower
bound used here is a probability lower bound in $(0,1]$. Thus
$d_D(B_{\max})^{N-1}\ge d_D(B_{\max})^{N_0}$,
$p_J^{N-1}\ge p_J^{N_0}$ and
$p_{O,0}(r_0)^{N-3}\ge p_{O,0}(r_0)^{N_0}$.
Substitute these inequalities into (UDA.7) to obtain $C_S$ uniformly over
the entire finite range. Both constants are strictly positive real numbers
because their radii and Gaussian amplitudes are positive and every tested
force margin holds. Taking their minimum proves (UDA.9).
Both constructed events are raw one-step events and already enforce
survival. Dropping the terminal-alive indicators only retains their original
determinant, so the same event bounds prove (UDA.10) before any division
by $\alpha_N$. No unproved uniform eigenvalue lower bound enters either
statement.
:::

:::{prf:corollary} Nonzero population-uniform native determinant fluctuation
:label: cor-uda-uniform-scaled-determinant

For the same actual reference law and existing same-stage B2 readout, let
$\chi=s^2/[s^2+t^2q^2]>0$ and $\kappa\ne0$. The exact terminal-geometry
conditioning of {prf:ref}`thm-ngf-terminal-geometry-centroid` and (UDA.9)
give

$$
\operatorname{Var}_{\mathbb C,\mathbb P_N^{\rm sel}}
(\sqrt N\,b_N^{\rm alive})
\ge N(1-e^{-3\kappa^2q^2\chi/N})C_{\rm amp}
\ge3\kappa^2q^2\chi e^{-3\kappa^2q^2\chi/4}C_{\rm amp}>0
\tag{UDA.11}
$$

for every $N\ge4$. For raw B2 under the incoming QSD the same conclusion
holds with $\chi=1$, by (UDA.10) and
{prf:ref}`cor-ngf-scaled-centroid-innovation`.
The amplitude estimate previously missing in (NGF.11) is therefore
discharged for these actual reference readouts and laws, including their
fixed positive numerical threshold and both normalizations.

This proves a native $SU(3)$-invariant determinant fluctuation component.
The full scaled fixed-tag determinant is proved non-tight below; it is
different from this conditionally centered centroid innovation. Triangle or
curvature covariance, a complete gauge drift/bracket limit, physical local
connection correspondence, or a Yang--Mills gap. Additional clone-identity,
physical-localization and growing-graph averages need their own amplitude
estimates. A positive finite covariance lower bound alone cannot pass to a
weak limit without the required moment/tightness bounds.
:::

:::{prf:proof}
Condition on the complete preparation, all actual terminal positions and
the relative O array. The terminal-alive indicator is then fixed, and
$b_N^{\rm alive}$ retains exactly the common determinant phase. Its
relative coefficient has modulus equal to $|b_N^{\rm alive}|$.
The native conditional covariance formula therefore has expectation at
least $(1-e^{-3\kappa^2q^2\chi/N})C_{\rm amp}$ under the actual selected
law. Total variance and multiplication by $N$ give the first inequality.
For $u\ge0$, $1-e^{-u}\ge ue^{-u}$, and $N\ge4$ gives the second.
The raw assertion uses the already proved raw coefficient and amplitude.
All nonzero constants come from the original independent Gaussian stages
and explicitly verified reference margins. Their centroid contribution
cancels for Gram/triangle coordinates, as proved in (NGF.1), so no variance
claim for those different coordinates follows from this calculation.
:::

(sec-uda-actual-fixed-tag-scaling)=
## 6. The full fixed-tag law and its actual scaling

:::{prf:proposition} Exact determinant centering and failure of the full fixed-tag square-root scaling
:label: prop-uda-fixed-tag-centering-nontightness

Use precisely the reference analytic kernels and same-stage B2 readouts
defined above, with no additional slot-dependent weights, clone-identity
exclusion, ordered-star priority, elites or changed phase alignment.
Their complete incoming QSD and one-step survival-selected record laws are
exchangeable in the walker labels. Consequently both full complex variables
$b_N$ and $b_N^{\rm alive}$ have centrally symmetric laws:

$$
b_N\overset{\rm law}=-b_N,\qquad
b_N^{\rm alive}\overset{\rm law}=-b_N^{\rm alive},\qquad
\mathbb E b_N=\mathbb E_{\mathbb P_N^{\rm sel}}b_N^{\rm alive}=0.
\tag{UDA.12}
$$

Thus their fully centered $\sqrt N$ variances satisfy the stronger bounds

$$
\operatorname{Var}_{\mathbb C,\mathbb P_N^{\rm sel}}
 (\sqrt N\,b_N^{\rm alive})
=N\mathbb E_{\mathbb P_N^{\rm sel}}|b_N^{\rm alive}|^2
\ge NC_{\rm amp},
\tag{UDA.13}
$$

and the corresponding raw-QSD bound also holds. Along every unbounded
sequence of these already defined real-coordinate kernels, the full
variables $\sqrt N\,b_N^{\rm alive}$ and $\sqrt N\,b_N$ are **not tight**
as complex random variables. In particular, for every $R>0$ and every
$N\ge N_0$ with $N>(R/b_L)^2$,

$$
\mathbb P_N^{\rm sel}(|\sqrt N\,b_N^{\rm alive}|>R)\ge p_L,
\qquad
\mathbb P_{\nu_N,\rm raw}(|\sqrt N\,b_N|>R)\ge p_L.
\tag{UDA.14}
$$

For the selected record let $\mathcal H_N$ retain the complete post-collision
preparation, actual terminal positions $Y$, and relative O array $r$, and put

$$
\overline b_N=\mathbb E[b_N^{\rm alive}\mid\mathcal H_N],\qquad
I_N=\sqrt N(b_N^{\rm alive}-\overline b_N),\qquad
A_\chi=3\kappa^2q^2\chi.
$$

The identified conditional centroid innovation remains bounded in second
moment and hence tight:

$$
\mathbb E_{\mathbb P_N^{\rm sel}}|I_N|^2
=N(1-e^{-A_\chi/N})\mathbb E_{\mathbb P_N^{\rm sel}}
 |b_N^{\rm alive}|^2\le A_\chi.
\tag{UDA.15}
$$

The scaled conditional mean $\sqrt N\,\overline b_N$ is itself non-tight.
For $\kappa\ne0$ set $K=(2A_\chi/p_L)^{1/2}$; then for every $R>0$ and
$N\ge N_0$ with $\sqrt N b_L>R+K$,

$$
\mathbb P_N^{\rm sel}(|\sqrt N\,\overline b_N|>R)\ge p_L/2.
\tag{UDA.16}
$$

At $\kappa=0$ the centroid innovation is zero and
$\overline b_N=b_N^{\rm alive}$, so non-tightness follows directly from
(UDA.14). These characterize an actual failing scaling of the existing
fixed tagged determinant. They do not imply failure of the unscaled
bounded determinant law or of an existing empirical/growing-graph channel
average, which is a different observable with its own weighting and
normalization. For a bounded set of executable population sizes no
$N\to\infty$ assertion is being made.
:::

:::{prf:proof}
Let $\pi$ be a permutation of the walker labels, acting on every retained
row coordinate, alive mark, donor index and transition record. The unchanged
reference has $n_{\rm elite}=0$ and no ordered priority/history state.
Its common reward and potential maps are rowwise, and global normalization
uses symmetric sums. Its two donor laws depend on transported pair distances,
exclude only the transported self index, and use independent categorical
draws with transported probabilities. The clone gates depend only on
transported fitness/recipient/donor values; complete fitness ties therefore
retain their same gate probabilities. Simultaneous copying and independent
recipient jitters transport under $\pi$.

Each accepted connected component transports to the corresponding component;
its barycenter and single independent Haar matrix law are unchanged by
relabeling. Thus the canonical cloning equivariance of
{prf:ref}`lem-chaos-canonical-equivariance` holds, including the retained
dead coordinates and mandatory revival. Both viscous normalizations also
transport: their Gaussian pair kernel and complete eligible-count or row
sums are unchanged except for their row labels. Applying this identity at
both actual B1 and B2, and transporting the independent O/final Gaussian
rows, proves equivariance of the complete recorded kinetic kernel. The
original common radial cap and terminal box mark are rowwise. Nonextinction
is permutation invariant, so the killed kernel intertwines with $\pi$.
These are distributional assertions under the original independent analytic
innovations, not fixed-seed bit-level equivariance claims.

The unique full QSD of {prf:ref}`thm-cgd-finite-n-qsd` is consequently
exchangeable by the same eigenmeasure argument as
{prf:ref}`lem-exchangeability`: $\pi_\#\nu_N$ is a QSD of the same kernel,
so uniqueness gives $\pi_\#\nu_N=\nu_N$. Combining this input symmetry
with the complete recorded equivariance gives exchangeability of its raw
one-step record. Restricting to nonextinction and dividing by the scalar
$\alpha_N$ preserves the symmetry, proving the selected assertion.

In particular swap labels 1 and 2. The same-stage color map transports
the color columns and their common threshold masks. Its determinant changes
sign, while the product of the three terminal-alive masks is unchanged.
Thus each of the two asserted variables has the law of its negative.
Its bound $|b|\le1$ makes its complex expectation finite, giving (UDA.12).
Substituting this exact zero mean and the already proved amplitude bounds
gives (UDA.13).

Variance divergence alone would not prove non-tightness. Instead use the
actual event of {prf:ref}`thm-uda-large-n-amplitude`, whose raw and selected
probability is at least $p_L$ and whose determinant magnitude is at least
$b_L$. If $\sqrt N b_L>R$, this event is a subset of the corresponding
tail event. This proves (UDA.14). Every compact subset of $\mathbb C$
is contained in a disk of some finite radius. Taking any tightness error
less than $p_L$ contradicts these uniform tail events along an unbounded
population sequence.

Conditional on $\mathcal H_N$, the terminal-geometry phase calculation
of {prf:ref}`thm-ngf-terminal-geometry-centroid` applies with the exact
terminal-alive mask held fixed. Integrating its conditional complex variance
proves the equality in (UDA.15); $|b_N^{\rm alive}|\le1$ and
$1-e^{-u}\le u$ give its bound. Markov's inequality supplies tightness
of $I_N$ without any output independence or stationary-chaos claim.
For $\kappa\ne0$, the same bound gives
$\mathbb P(|I_N|>K)\le A_\chi/K^2=p_L/2$.
On the determinant event with $|\sqrt N b_N^{\rm alive}|>R+K$,
excluding this innovation tail leaves probability at least $p_L/2$.
The triangle inequality then forces
$|\sqrt N\,\overline b_N|>R$, proving (UDA.16).
At zero phase coefficient, the conditional variance is zero and its
conditional mean equals the original determinant almost surely.
All claims therefore refer to the full actual tagged readout and its exact
conditional decomposition, without assigning a fluctuation scaling to an
unexamined empirical or geometric average.
:::

:::{prf:theorem} Nontrivial subsequential law of the actual centered centroid innovation
:label: thm-uda-native-centroid-innovation-limit

Keep the original fixed reference parameters and $\kappa\ne0$. Under the
actual one-step survival-selected law let

$$
\begin{gathered}
\mathcal H_N=(\text{full post-collision preparation},Y,r),\quad
g_Y=N^{-1}\sum_i\left[cv_{1,i}+
 \frac{tq^2}{t^2q^2+s^2}(Y_i-x_{1,i}-tcv_{1,i})\right],\\
A_N=B_{123}(r)\prod_{i=1}^3\mathbf1_D(Y_i)
 e^{i\kappa\mathbf1\cdot g_Y},\quad
\sigma_*=\sqrt3\,\kappa q\sqrt\chi,\quad
I_N=\sqrt N\{b_N^{\rm alive}-\mathbb E[b_N^{\rm alive}\mid\mathcal H_N]\}.
\end{gathered}
$$

Every quantity here is an actual retained record or the exact conditional
centering already computed from it. There is an exact representation of its
law

$$
I_N=A_N\sqrt N\{e^{i\sigma_* Z/\sqrt N}
                     -e^{-\sigma_*^2/(2N)}\},
\quad Z\sim N(0,1),\quad Z\text{ independent of }\mathcal H_N,
\tag{UDA.17}
$$

with $|A_N|\le1$ and
$\mathbb E_{\mathbb P_N^{\rm sel}}|A_N|^2\ge C_{\rm amp}$.
From every unbounded population sequence one can choose a subsequence along
which $A_N\Rightarrow A$, with $|A|\le1$ and $\mathbb E|A|^2\ge C_{\rm amp}$.
On that same subsequence the joint law satisfies

$$
(A_N,I_N)\ \Rightarrow\ (A,i\sigma_* A Z),
\qquad Z\sim N(0,1)\text{ independent of }A.
\tag{UDA.18}
$$

All fixed mixed complex moments of these variables converge. In particular

$$
\mathbb E[i\sigma_* A Z]=0,\quad
\mathbb E|i\sigma_* A Z|^2
=3\kappa^2q^2\chi\,\mathbb E|A|^2
\ge3\kappa^2q^2\chi C_{\rm amp}>0.
\tag{UDA.19}
$$

For each fixed $1\le p<\infty$, the error in replacing $I_N$ by its linear
phase innovation $i\sigma_* A_N Z$ has the explicit bound

$$
\|I_N-i\sigma_* A_N Z\|_p
\le\frac{\sigma_*^2}{2\sqrt N}
 \{(\mathbb E|Z|^{2p})^{1/p}+1\}.
\tag{UDA.20}
$$

The same theorem holds for the raw B2 determinant under incoming QSD,
with $\chi=1$, $\mathcal H_N=(\text{preparation},r)$ and
$A_N=B_{123}(r)e^{i\kappa\mathbf1\cdot c\overline v_1}$.
At zero phase coefficient the defined centered centroid innovation is
identically zero. The positive conclusion concerns the original configured
$q>0$, $s>0$, $\kappa\ne0$ and the derived amplitude regime verified above.
The limiting law is a Gaussian mixture for this exact conditional innovation;
the full fixed-tag $\sqrt N$ variable remains non-tight by
{prf:ref}`prop-uda-fixed-tag-centering-nontightness`. No claim of a Gaussian
full gauge field, a physical-time evolution, or a complete limiting
drift/bracket equation is part of this statement.
:::

:::{prf:proof}
At fixed complete preparation and terminal position array, the exact Gaussian
calculation of {prf:ref}`thm-ngf-terminal-geometry-centroid` gives
$G=g_Y+q\sqrt{\chi/N}\,Z_3$ with $Z_3\sim N(0,I_3)$ independent of the
relative residual array. Given the preparation and $Y$, that relative
residual array and $r$ determine each other by a fixed affine map.
Thus the conditional law of the standardized centroid given $\mathcal H_N$
is always the same standard Gaussian. Conditional probability factorization
then proves its independence of $\mathcal H_N$ under the entire selected
law: one-step survival is a function of $Y$, already retained in that
conditioning. This is exact independence after the declared conditioning,
and does not assert independent completed walkers.

Set $Z=\mathbf1\cdot Z_3/\sqrt3$. The actual B2 determinant and alive
mask then give
$b_N^{\rm alive}=A_Ne^{i\sigma_*Z/\sqrt N}$, and its exact conditional
mean is $A_Ne^{-\sigma_*^2/(2N)}$. This proves (UDA.17).
Its coefficient modulus is precisely $|b_N^{\rm alive}|$, so the disk
bound and the uniform amplitude theorem give its stated second moment.

Probability laws on the closed unit disk have a weakly convergent subsequence
because the disk is a compact metric space. Equivalently the compact-coordinate
argument of {prf:ref}`thm-ym-native-physical-gauge-hierarchy` applies to
the two bounded real/imaginary coordinates here. The limit remains in this
disk. Its squared modulus is a bounded continuous function, so its moment
passes to the limit and is at least $C_{\rm amp}$. Since $Z$ is independent
of the coefficient at every $N$, their joint laws are products of the
coefficient law and this one unchanged Gaussian law. Testing continuous
functions first on compact Gaussian intervals and then bounding the omitted
Gaussian tails proves $(A_N,Z)\Rightarrow(A,Z)$ with the asserted independent
product law.

The elementary Taylor bounds give, pointwise for $a=\sigma_* Z$,

$$
\left|\sqrt N(e^{ia/\sqrt N}-e^{-\sigma_*^2/(2N)})-ia\right|
\le\frac{a^2+\sigma_*^2}{2\sqrt N}.
$$

Multiply by $|A_N|\le1$ and take $L^p$ norms to obtain (UDA.20).
Every Gaussian moment is finite, so this error tends to zero in every
fixed $L^p$. The continuous product map applied to $(A_N,Z)$ gives the
limit of $i\sigma_*A_NZ$; the error bound gives the same limit for $I_N$,
jointly with $A_N$.

To pass all mixed complex moments, note the uniform pointwise bound

$$
|I_N|\le |\sigma_*||Z|+\sigma_*^2/(2\sqrt N)
\le |\sigma_*||Z|+\sigma_*^2/4\qquad(N\ge4).
$$

It gives uniform moments of every order, jointly with the bounded
$A_N$. For any fixed polynomial, telescoping its products and applying
(UDA.20) with a sufficiently high $p$ makes replacement by the linear
innovation converge in expectation. Independence then factors its mixed
moments into bounded coefficient moments, which converge by weak disk
convergence, and unchanged finite Gaussian moments. This proves the full
moment assertion without an assumed fluctuation law. Formula (UDA.19)
follows from independence, $\mathbb EZ=0$, $\mathbb EZ^2=1$, and the
positive amplitude bound.

For the raw record the preceding raw centroid factorization has $\chi=1$
and retains exactly the incoming QSD/preparation law. The raw amplitude
bound (UDA.10) supplies the same disk moment lower bound. The rest of the
argument is identical. At $\sigma_*=0$, the exponential in (UDA.17) and
its conditional mean are both one, giving exactly zero innovation.
:::
