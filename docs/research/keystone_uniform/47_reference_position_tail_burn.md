# Harmonic source, signed position energy and exact terminal tails

(sec-rpt-retained)=
## 1. Actual reference carrier

:::{prf:definition} Position-tail comparison at the default box
:label: def-rpt-retained

Retain the actual harmonic count record of research 34, 37 and 38:
$F=-x$, $R=-|x|^2/2$, $d=3$, $L=2$, $h=.04$,
$t=.02$, $\nu=.3$, $a=t\nu=.006$, $V=2$, $V_c=4$,
$\sigma_J=\sigma_x=.1$, and component restitution
$\alpha_{\rm col}=.5$. Set
$$
c=e^{-.04},\quad b=t(1+c),\quad a_x=1-tb,\quad m=1-t^2,
\quad q^2=(1-c^2)/2,\quad s^2=.0004,\quad
\tau^2=t^2q^2+s^2,\quad \beta=.04.
$$
The actual source/component plan, sampled current-frame fitness and
alive-only normalizers, mandatory revival, raw reward, component
Haar marks on original velocities and all Gaussian innovations are
retained. Every positional source $S_i$ lies in $D=[-2,2]^3$.
The actual stages are
$$
X=S+IJ,\quad P=v^{\rm p},\quad p_1=(I-aL_X)P,\quad
w=c(p_1-tX)+q\xi,
$$
$$
y=a_xX+bp_1+tq\xi,\quad z=(I-aL_y)w-ty,\quad
x^+=y+s\chi,\quad v^+=C_V(z).
\tag{RPT.1}
$$
All finite inner products below are normalized row averages.
Population inner products use the actual lifted root law, with
the same almost-sure finite-component premise as research 37.
The full actual noisy second graph is used in every $L_y$.

The zero comparison in Section 2 is a deterministic Lyapunov
calculation with realized graphs held fixed. It is not a coupling
to a zero-source stochastic gas. In particular no graph/noise or
cap/noise cross term is declared to have zero expectation.
Section 4 conditions only before the fresh independent positional
OU/final innovations, and Section 5 proves an explicit source-tail
criterion. The source-boundary burn needed for an unconditional
small-dead theorem is recorded separately in Section 6.
:::

(sec-rpt-signed-noise)=
## 2. State-to-zero balance with all noisy alignment terms retained

:::{prf:lemma} Signed source-phase Lyapunov account
:label: lem-rpt-signed-source-energy

Let $\kappa_x=.0015$, $\kappa_v=.073$ and
$Q_\beta(X,P)=\|X\|^2+2\beta\langle X,P\rangle+\|P\|^2$.
For the actual proposed update define
$$
D_1=\langle P,L_XP\rangle,\quad E_1=\langle X,L_XP\rangle,
\quad D_2=\langle w,L_yw\rangle,\quad E_2=\langle y,L_yw\rangle,
$$
$$
C_G=dq^2\{t^2+[1+t(\beta-t)]^2\}+ds^2.
$$
Put $E_{\rm cap}=z-C_V(z)$ and retain the actual signed cap loss
$$
\mathcal L_{\rm cap}
=\|E_{\rm cap}+\beta y\|^2
 +2\langle E_{\rm cap},C_V(z)\rangle\ge0.
$$
Then
$$
\begin{split}
\mathbb E Q_\beta(x^+,v^+)\le{}&
\mathbb E Q_\beta(X,P)
-\kappa_x\mathbb E\|X\|^2-\kappa_v\mathbb E\|p_1\|^2\\
&-2a\mathbb E D_1+a^2\mathbb E\|L_XP\|^2
                 -2\beta a\mathbb E E_1\\
&-2a\mathbb E D_2+a^2\mathbb E\|L_yw\|^2
                 -2a(\beta-t)\mathbb E E_2+C_G
                 -\mathbb E\mathcal L_{\rm cap}.
\end{split}
\tag{RPT.2}
$$
Here $D_2,E_2,\mathcal L_{\rm cap}$ are the actual correlated noisy
quantities. In particular the product between $y$ and the nonlinear
cap defect remains inside $\mathbb E\mathcal L_{\rm cap}$.
An unconditional, less sharp consequence is
$$
\mathbb E Q_\beta(x^+,v^+)
\le\mathbb E Q_\beta(X,P)
-.00149\,\mathbb E\|X\|^2
-.0721\,\mathbb E\|P\|^2+C_G^+,
\tag{RPT.3}
$$
where
$$
C_G^+=C_G+dq^2t^2\frac{a}{2-a}(\beta-t)^2<.119.
$$
These are raw proposal estimates. Under the actual next-survival
law their nonnegative output may be divided by its own probability,
bounded below by $1-e_N$; the innovations on that restricted law
are not asserted to be Gaussian.
:::

:::{prf:proof}
The exact native-cap identity is
$$
Q_\beta(y,C_V(z))
=\|y\|^2+\|z+\beta y\|^2-\mathcal L_{\rm cap}.
$$
It follows by expanding $z=C_V(z)+E_{\rm cap}$.
Radial positivity gives
$\langle E_{\rm cap},C_V(z)\rangle\ge0$, hence
$\mathcal L_{\rm cap}\ge0$. The firm-cap majorant follows by
dropping this loss, but (RPT.2) retains it with its actual noise
and state correlation.
Substitute the actual $z=w-ty-aL_yw$ before any averaging.
The majorant expands exactly as
$$
\|y\|^2+\|w+(\beta-t)y\|^2
-2aD_2+a^2\|L_yw\|^2-2a(\beta-t)E_2.
$$
All terms containing the noisy second graph remain in this expression,
as does the subtracted $\mathcal L_{\rm cap}$.

Put $R_0=a_xX+bp_1$ and $W_0=c(p_1-tX)$.
Before the second graph is applied,
$y=R_0+tq\xi$ and $w=W_0+q\xi$.
The prepared $X,P,L_X$ are independent of the fresh OU array.
Consequently the expectation of this affine, pre-second-graph
majorant is exactly
$$
\mathbb E[\|R_0\|^2+\|W_0+(\beta-t)R_0\|^2]
+dq^2\{t^2+[1+t(\beta-t)]^2\}.
$$
This is the only OU centering used. In particular it does not
center $L_y\xi$ or the nonlinear cap.
The anisotropic harmonic majorant (RFK.2) bounds its first term
by $Q_\beta(X,p_1)-\kappa_x\|X\|^2-\kappa_v\|p_1\|^2$.
Expand the first count kick:
$$
Q_\beta(X,p_1)-Q_\beta(X,P)
=-2aD_1+a^2\|L_XP\|^2-2\beta aE_1.
$$
The final $\chi$ is fresh and independent of $y,z,C_V(z)$.
It adds exactly $ds^2$ to expected $Q_\beta$.
This proves (RPT.2).

For (RPT.3), drop only the nonnegative cap loss and apply spectral
calculus to $p_2=aL_yw$:
$\langle p_2,w\rangle\ge\|p_2\|^2/a$.
With $\delta=a/(2-a)$ the second-graph expression is bounded
pointwise by
$\delta(\beta-t)^2\|y\|^2$.
Its expectation is at most
$$
\delta(\beta-t)^2
[\mathbb E(\|X\|^2+\|p_1\|^2)+dt^2q^2],
$$
using $a_x^2+b^2<1$.
The first-graph expression is at most
$\delta\beta^2\|X\|^2$.
Also $\|p_1\|\ge(1-a)\|P\|$.
The same exact rational losses as (RFK.3) are greater than
$.00149$ and $.0721$.
Finally $q^2<.0392$ gives
$$
C_G^+<
3(.0392)\{.0004+1.0004^2
 +.0004\,\delta(.02)^2\}+.0012
<.118941139<.119.
$$
The population proof uses the actual self-adjoint count operator on
its joint stage law. The finite proof is pointwise in each realized
operator before expectation. Neither proof substitutes an
independent second graph.
:::

(sec-rpt-preparation)=
## 3. Exact preparation and position moments

:::{prf:lemma} The source/component cross term cannot be discarded
:label: lem-rpt-preparation-cross

Freeze the actual measured source/acceptance forest before its Haar
marks and recipient jitters. For each component $C$ write
$\bar v_C=|C|^{-1}\sum_{i\in C}v_i$ and let $S_i$ be its actual
frozen positional source. The exact conditional preparation identities
are
$$
\mathbb E_{J,O}\|X\|^2
=\frac1N\sum_i|S_i|^2+d\sigma_J^2\frac1N\sum_i I_i,
$$
$$
\mathbb E_{J,O}\langle X,P\rangle
=\frac1N\sum_C\bar v_C\cdot\sum_{i\in C}S_i,
\tag{RPT.4}
$$
$$
\|P\|^2
=\frac1N\sum_C\left[
|C||\bar v_C|^2+\alpha_{\rm col}^2
                       \sum_{i\in C}|v_i-\bar v_C|^2\right]
\le\|v\|^2.
$$
For the actual proposed position, without an independence assumption
between $X$ and $p_1$,
$$
\mathbb E\|x^+\|^2
=a_x^2\mathbb E\|X\|^2
 +2a_xb\mathbb E\langle X,p_1\rangle
 +b^2\mathbb E\|p_1\|^2+d\tau^2.
\tag{RPT.5}
$$
In particular its correlation-sensitive entries are
$$
\mathbb E\langle X,p_1\rangle
=\mathbb E\langle X,P\rangle-a\mathbb E E_1,
$$
$$
\mathbb E\|p_1\|^2
=\mathbb E\|P\|^2-2a\mathbb E D_1+a^2\mathbb E\|L_XP\|^2.
\tag{RPT.6}
$$
The population formulas are the corresponding actual rooted
expectations of these component/source quantities.
:::

:::{prf:proof}
Recipient jitter has zero conditional mean and is independent of
the preceding source/component/Haar plan. This gives the first
identity and removes its cross term with $P$.
For a Haar matrix in the actual dimension three,
$\mathbb E O_C=0$. Thus
$\mathbb E P_i=\bar v_C$ on $C$, giving (RPT.4).
For every actual Haar realization, orthogonality and the zero
sum of the deviations prove the displayed velocity identity.
The input velocities there are the original slots.

In (RPT.1) the fresh Gaussian $tq\xi+s\chi$ has covariance
$\tau^2I_d$ and is independent of the complete prepared $X,p_1$.
It contributes $d\tau^2$ to the affine position square.
The second kick and cap change only velocity. This proves (RPT.5).
Expanding $p_1=P-aL_XP$ proves (RPT.6).
For the population the same rooted identities hold; its component
mean formulas are the uniform-root limits of the finite ones in
the declared finite-component regime.
:::

A bound such as
$\mathbb E\|x^+\|^2\le(a_x\sqrt{12.03}+br)^2+d\tau^2$
follows when the entering full-slot velocity RMS is at most $r$.
Research 34 supplies $r=.55$ after six population updates and
$r=.56$ after six finite current-survivor updates under its
$e_N\le.01$ premise. The finite next output must again be divided
by its own next-survival probability if conditioned.
This bound is finite and uniform, but its size is too large to
prove a small dead fraction by quadratic Markov alone.
The signed quantities in (RPT.2), (RPT.4)--(RPT.6) retain the
information lost by that scalar upper bound.

:::{prf:remark} Revival is not Euclidean positional-energy contraction
:label: rem-rpt-revival-energy

For an actual two-slot input take alive position $(2,2,2)$,
dead position $(2.1,0,0)$ and both original velocities zero.
The alive slot persists under the singleton convention and the dead
slot must copy the only alive source. Before its Gaussian jitter,
source position energy is $12$, while entering position energy is
$(12+4.41)/2=8.205$.
This example only rules out the inference
$\|S\|^2\le\|x\|^2$ for mandatory revival.
It does not refute population attraction, survivor convergence or
any of the proved kinetic balances.
The actual source/component cross term and incoming-source
weights are therefore required when comparing prepared
$Q_\beta$ with entering $Q_\beta$.
:::

(sec-rpt-exact-tail)=
## 4. Exact terminal alive probabilities before survival restriction

:::{prf:theorem} Full-Gaussian terminal tail interface
:label: thm-rpt-exact-terminal-tail

Freeze the complete actual prepared array $(X,P)$ and its first
count output $p_1$. Put $M_i=a_xX_i+bp_{1,i}$ and define
$$
\Pi_L(M)=\prod_{j=1}^d
\left[
\Phi\!\left(\frac{L-M_j}{\tau}\right)
-\Phi\!\left(\frac{-L-M_j}{\tau}\right)\right].
$$
Then the actual terminal alive indicators are conditionally
independent Bernoulli variables with probabilities $\Pi_L(M_i)$.
In particular
$$
\mathbb E\!\left[\frac{A_N^+}{N}\mid X,P\right]
=\frac1N\sum_i\Pi_L(M_i),
\quad
\Pr(A_N^+=0\mid X,P)=\prod_i[1-\Pi_L(M_i)].
\tag{RPT.7}
$$
For every $u>0$, their actual conditional empirical concentration is
$$
\Pr\left(
\left|\frac{A_N^+}{N}-\frac1N\sum_i\Pi_L(M_i)\right|\ge u
\ \middle|\ X,P\right)\le2e^{-2Nu^2}.
\tag{RPT.8}
$$
For the population, the exact raw output dead mass is
$$
\mu^+(a=0)=\mathbb E_{\rm prep}
[1-\Pi_L(a_xX+bp_1)].
\tag{RPT.9}
$$
For finite current-survivor input, raw averages of these conditional
formulas are valid. A nonnegative output dead fraction restricted
to own next survival obeys
$$
\mathbb E\!\left[1-\frac{A_N^+}{N}\mid\tau_N>n+1\right]
\le\frac{\mathbb E_{\eta_n}[1-N^{-1}\sum_i\Pi_L(M_i)]}{1-e_N}.
\tag{RPT.10}
$$
Conditional independence in (RPT.7) is not asserted after this
survival restriction or after conditioning on a moment event.
:::

:::{prf:proof}
The prepared $X,p_1$ are fixed before the fresh rowwise
independent $\xi_i,\chi_i$. Equation (RPT.1) gives
$x_i^+=M_i+tq\xi_i+s\chi_i$ exactly.
The second noisy count kick and cap do not change positions.
The sum is a centered Gaussian with covariance $\tau^2I_d$.
Its coordinates and its rows are independent at this precise
conditioning stage. Coordinate integration proves (RPT.7).
For a Bernoulli variable $B$ of any success probability, put
$h(\lambda)=\log\mathbb E e^{\lambda(B-\mathbb EB)}$.
Then $h(0)=h'(0)=0$ and $h''(\lambda)$ is the variance of a
Bernoulli under exponential tilting, hence at most $1/4$.
Integrating twice gives $h(\lambda)\le\lambda^2/8$.
Conditional independence multiplies these moment generating
functions. Exponential Markov at $\lambda=4u$ bounds the upper
deviation of their average by $e^{-2Nu^2}$; the negative tilt
gives the same lower bound. Their union proves (RPT.8), including
different individual means. The population root calculation gives
(RPT.9) under its actual joint first provider.
For (RPT.10), restrict the nonnegative actual dead fraction to
own next survival and divide by that event's probability, which
is at least $1-e_N$ by research 29. Its unconditioned mean is
the average in (RPT.7). This retains the reweighting of the
prepared array and both innovations.
:::

(sec-rpt-source-criterion)=
## 5. Explicit source-interior criterion for a small next dead mass

:::{prf:theorem} Source-boundary charge with all jitter outcomes
:label: thm-rpt-source-interior-criterion

For a consistent entering population let $m>0$ be its alive mass
and put
$$
b_\delta=\mu\{a=1,\ |x|_\infty>L-\delta\},
\qquad 0<\delta<L.
$$
For a finite array use the corresponding fractions
$m=M/N$ and $b_\delta=N^{-1}\sum_i
\mathbf1_{\{a_i=1,\ |x_i|_\infty>L-\delta\}}$.
Let $a_*$ be the actual uniform alive acceptance ceiling and
$c_*=a_*/\kappa_C$, with its actual positive donor floor
$\kappa_C$. Define
$$
B_\delta=\min\left\{1,
\left[1+c_*+\frac{1-m}{\kappa_Cm}\right]b_\delta\right\}.
$$
For $0<u\le V_c$, put
$$
r_{\delta,u}=(1-a_x)L+a_x\delta-bu.
$$
When $r_{\delta,u}>0$, a valid raw output dead-mass bound is
$$
\mu^+(a=0)\ \text{or}\
\mathbb E[1-A_N^+/N\mid S]
\le\min\{1,B_\delta+V_u+T_0+
(ma_*+1-m)(T_1-T_0)\},
\tag{RPT.11}
$$
where
$$
\sigma_0=\tau,\quad \sigma_1^2=\tau^2+a_x^2\sigma_J^2,\quad
T_j=1-[1-2\Phi(-r_{\delta,u}/\sigma_j)]^d,
$$
and one may use $V_u=\min\{1,r_{\rm vel}^2/u^2\}$ whenever the
actual entering full-slot velocity RMS is at most $r_{\rm vel}$.
At $u=V_c$, the sharper value $V_u=0$ is always valid.
For finite random entering laws average the state-dependent source
term and use their actual velocity energy; under next survival
divide the resulting raw nonnegative bound by $1-e_N$.

At the unchanged default parameters, $\delta=.5$ and $u=V_c=4$
give the particularly simple bound
$$
\mu^+(a=0)\ \text{or}\
\mathbb E[1-A_N^+/N\mid S]<B_{.5}+.003.
\tag{RPT.12}
$$
If the original configured donor role is uniform
($\kappa_C=1$), $c_*\le1/8$, and the conditional alive
boundary fraction is at most $.001$, this implies the raw
next dead mass is less than $.004125$.
This is a sufficient input-class criterion, not a proved burn
into that class.
:::

:::{prf:proof}
For each alive source label, its expected accepted-alive incoming
column is at most $c_*$. An alive row's persistent contribution
is at most its own boundary indicator. Every dead row selects an
alive donor with probability at most $1/(\kappa_CM)$.
Summing boundary-source contributions therefore gives $B_\delta$.
If $M=1$, the singleton has no accepted-alive edge; the same
upper bound remains valid. For the population the source-weighted
conditional alive donor formula gives exactly the corresponding
bound. The possible dead column is not replaced by a small one.

The actual position identity is
$$
x^+=a_xS+bp_1+G,\qquad G=a_xIJ+tq\xi+s\chi.
$$
If the source is inside the smaller box, $|p_1|\le u$, and
$|G|_\infty\le r_{\delta,u}$, then $|x^+|_\infty\le L$
by a pointwise triangle inequality.
The first count average contracts full-slot velocity energy;
Markov bounds its exceptional $|p_1|>u$ mass by $V_u$.
Its pathwise bound $|p_1|\le V_c$ makes this exception empty
when $u=V_c$.

Conditional on the source/indicator/component plan before
recipient jitter, $G$ is centered Gaussian of variance
$\sigma_I^2I_d$. This centering does not condition on $p_1$,
which can depend on every jitter. The preceding implication is
pointwise, so a union bound needs no independence of $G$ and
$p_1$. The exact Gaussian mixture tail is
$(1-\mathbb EI)T_0+\mathbb EI\,T_1$.
Every dead row copies and each alive acceptance is at most $a_*$,
so $\mathbb EI\le ma_*+1-m$.
Since $\sigma_1>\sigma_0$, $T_1\ge T_0$, proving (RPT.11).

For (RPT.12), the rational bounds of research 38 give
$$
r_{.5,4}=.5-3.97b>.34431248,
$$
$$
\sigma_1^2<.00041568+.999216^2/100
=.01040000614656.
$$
Their exact rational comparison yields
$r_{.5,4}/\sigma_1>10/3$.
Integration by parts, or
$\int_z^\infty e^{-u^2/2}du
\le z^{-1}\int_z^\infty u e^{-u^2/2}du$,
gives the Gaussian Mills bound. A union over $d=3$ coordinates
therefore yields
$$
T_1\le6\Phi(-r_{.5,4}/\sigma_1)
<\frac{18}{25}e^{-50/9}<\frac{18}{6250}<.003.
$$
Here $\sqrt{2\pi}>5/2$, and the entirely rational Taylor
certificate $\sum_{k=0}^{10}(50/9)^k/k!>250$ proves the
last exponential inequality. Thus every jitter and kinetic
Gaussian tail is charged rather than discarded.

For uniform donors write $b_\delta=m\beta_\delta$.
Then $B_\delta\le[m(1+c_*)+(1-m)]\beta_\delta
\le(1+c_*)\beta_\delta$.
Insert $\beta_{.5}\le.001$ and $c_*\le1/8$ to get
$.001125+.003=.004125$.
:::

(sec-rpt-last-inference)=
## 6. Last justified interface and missing boundary burn

The actual default kernel now has the complete signed source-phase
balance (RPT.2), its finite-noise majorant (RPT.3), the exact
original-slot Haar/source cross term (RPT.4), the correlation-sensitive
position identities (RPT.5)--(RPT.6), and the exact Gaussian terminal
tail and concentration formulas (RPT.7)--(RPT.10).
The source-interior criterion (RPT.11)--(RPT.12) is an explicit
nonempty class on which the next raw dead mass is small at $L=2$.
Neither a positional cutoff nor a modified reward, cap, source
center or reset was introduced.

These facts do not show that an arbitrary current survivor enters
the source-interior class after a uniform finite burn. Velocity
RMS alone does not control the local source-boundary fraction or
the signed component/source correlation in (RPT.4). In particular,
the preparation energy comparison required to iterate (RPT.3)
cannot be replaced by an assumed Euclidean contraction of mandatory
revival. The positive alive-count floor supplies a denominator,
but no upper bound tending to zero on actual dead mass.

A completed small-dead burn needs a signed estimate coupling
boundary-source mass, original/component velocities and the exact
Gaussian inward/outward flux in (RPT.9), or another proved
invariant tail class for the unchanged own-provider recursion.
This is the first remaining inference, rather than an impossibility
claim. All default full-law and marked-feedback conclusions still
require their separate completed proof. The force arm here is
harmonic; no Rastrigin position-tail burn is implied.
