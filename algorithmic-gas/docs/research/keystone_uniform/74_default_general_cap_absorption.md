# Full-Gaussian conditional velocity deficit of the actual native cap

(sec-gca74-register)=
## 1. Actual noisy provider and the pre-OU information

:::{prf:definition} Conditional default cap register
:label: def-gca74-register

Retain the harmonic count update in research67--69, with
$$
d=3,\quad t=.02,\quad a=.006,\quad c=e^{-.04},\quad
b=t(1+c),\quad m=1-t^2,\quad a_x=1-tb,
$$
$$
q^2=(1-c^2)/2,\quad V=2,\quad V_c=4,\quad
z_v=c-tb,\quad r_H=mb=t(c+a_x).
\tag{GCA74.1}
$$
For an actual prepared root and its own first count output use
$$
U=(I-aL_X)P,\quad x_1=mX+tU,\quad
w=c(U-tX)+q\xi,\quad y=x_1+tw,
$$
$$
z=w-ty-aL_yw,\qquad C_V(z)=\frac{Vz}{V+|z|}.
\tag{GCA74.2}
$$
The actual second count graph uses the same $y,w$.
Every Gaussian row in $\xi$ is retained.

For a population law assume $|P|\le4$ and that its actual joint
OU-stage provider has first velocity moment at most $.70$:
$$
\int |w'|\,d\Lambda(y',w')\le .70.
\tag{GCA74.3}
$$
This provider is deterministic after fixing the prepared law.
Condition on the prepared root, its physical differential, and that
prepared provider, before the new original OU innovation.

For a finite array condition on the complete prepared array and its
complete physical differential before the original independent OU array.
For $N\ge2$, assume the actual prepared array satisfies
$$
|P_i|\le4,\qquad
\frac1N\sum_i|P_i|^2\le .56^2,\qquad
\frac1N\sum_i|X_i|^2\le12.25.
\tag{GCA74.4}
$$
There is no restriction on an individual prepared position.
The conclusion for $N=1$ below needs only its zero count field.

Write $\mathscr P$ for the appropriate pre-OU information and
$D_i=DC_V(z_i)$. A test vector $T_i$ is fixed by $\mathscr P$.
It can be correlated with all prepared positions, velocities, source
choices, component Haar outcomes and already sampled recipient jitters.
It is not permitted to depend on the new OU array.
:::

:::{prf:remark} Reach of the conditional hypotheses
:label: rem-gca74-hypotheses

The population post-burn budget $\|P\|_2\le.55$ together with the
actual source-box second moment $\mathbb E|X|^2\le12.03$ supplies
(GCA74.3). Indeed first count contraction and the original fresh
OU marginal give
$$
\mathbb E|w|^2
\le c^2(.55+.02\sqrt{12.03})^2+3q^2<.70^2.
$$
Cauchy--Schwarz then gives the required first moment.

The finite hypotheses concern the entire preparation before OU noise.
They can be used on the actual prepared budget event; conditioning on
a future empirical-OU event or on future survival does not supply the
fresh Gaussian array in this definition. The probability estimates of
research41 do not replace a mixed exceptional displacement moment by a
product. Interpolated prepared endpoints satisfying (GCA74.4) retain
its two norm bounds by convexity, without any pointwise velocity band.
:::

(sec-gca74-small-ball)=
## 2. Uniform small-ball bound through the actual second graph

:::{prf:lemma} Bare Gaussian reduction with unbounded prepared position
:label: lem-gca74-small-ball-reduction

At a root set
$$
\mu=z_vU-r_HX,\qquad \sigma=mq,\qquad z_0=\mu+\sigma\xi.
$$
For a population provider satisfying (GCA74.3), every $r>0$ obeys
$$
\{|z|\le r\}
\subseteq\{|z_0|\le A_r+B|\mu|\},
$$
$$
A_r=\frac{r+\delta cV_c+a(.70)}{\alpha},
\qquad B=\frac{\delta}{\alpha},\qquad
\alpha=1-a/m,\qquad \delta=\frac{at}{r_H}.
\tag{GCA74.5}
$$
For a finite array the same inclusion holds on the actual event
$\langle |w|\rangle_N\le K$, with $.70$ replaced by $K$.
No independence of the root, its graph or its empirical provider is
asserted in this inclusion.
:::

:::{prf:proof}

The actual count degree and velocity numerator satisfy
$$
0\le d_y\le1,\qquad
|M_y|\le \int |w'|\,d\Lambda(y',w'),
\qquad z=(m-ad_y)w-tx_1+aM_y.
$$
For finite arrays the integrals are exactly their normalized sums.
The included self numerator and denominator cancel in $L_yw$.
Since $z_0=mw-tx_1$, the event $|z|\le r$ implies
$$
|z_0|\le r+a|w|+aK.
$$
Also $w=(z_0+tx_1)/m$ and the exact affine identity is
$$
x_1=\frac{cU-\mu}{b}.
\tag{GCA74.6}
$$
Indeed $cb^{-1}-t=z_vb^{-1}$ and $r_H=mb$.
Consequently
$$
\alpha|z_0|
\le r+\frac{at}{mb}(c|U|+|\mu|)+aK.
$$
Every own first count output is a convex combination of prepared
velocities, so $|U|\le4$. Substitution gives the asserted inclusion.
It has not bounded $X$ or removed any cancellation event in its
Gaussian marginal.
:::

:::{prf:lemma} Full Gaussian charge of the finite provider event
:label: lem-gca74-finite-provider-tail

For a fixed prepared array satisfying (GCA74.4),
$$
\mathbb P_\xi\{\langle|w|\rangle_N>1.5\}
\le2^{-5N}.
\tag{GCA74.7}
$$
This is a bound under the complete original independent OU array.
The event is not conditioned on when using the row estimates below.
:::

:::{prf:proof}

First count contraction gives
$\langle|U|^2\rangle_N^{1/2}\le.56$.
For $\bar w_i=c(U_i-tX_i)$, the $L^2$ triangle and
$c<.9608$ give
$$
\langle|\bar w|\rangle_N
\le \langle|\bar w|^2\rangle_N^{1/2}
\le c(.56+.02\sqrt{12.25})<.606.
$$
Because $q^2<.0392<.198^2$,
$$
\langle|w|\rangle_N
\le .606+.198\sqrt{\langle|\xi|^2\rangle_N}.
$$
Thus the event in (GCA74.7) implies
$$
\sum_i|\xi_i|^2>20N,
$$
since $[(1.5-.606)/.198]^2=(149/33)^2>20$.
The exact Gaussian integral
$\mathbb E e^{|\xi_i|^2/4}=2^{3/2}$ and Markov's
inequality give
$$
\mathbb P\{\textstyle\sum_i|\xi_i|^2>20N\}
\le e^{-5N}2^{3N/2}\le2^{-5N}.
$$
For the last inequality use $\log2<.7$, so
$(13/2)\log2<5$. This charges the entire empirical-provider
tail under its actual correlated use in every row's graph.
:::

:::{prf:lemma} An exact rational threshold register
:label: lem-gca74-thresholds

Put $r_j=j/20$ for $1\le j\le12$ and define
$$
R_j=\frac{r_j+.025}{(.993)(.195)},\qquad
G_j=\frac45\sum_{k=0}^{30}
 \frac{(-1)^kR_j^{2k+3}}{2^k k!(2k+3)},
$$
$$
T_j=\frac{.0392}{[1-(r_j+.025)/.993]^2}.
\tag{GCA74.8}
$$
For both the population case (GCA74.3) and every finite row
in (GCA74.4), including $N=1$ with its zero count field,
$$
\mathbb P_\xi\{|z_i|\le r_j\mid\mathscr P\}\le u_j,
\tag{GCA74.9}
$$
where the following terminating decimals are exact rationals.

| $j$ | $r_j$ | $u_j$ |
|---:|---:|---:|
| 1 | .05 | .0469 |
| 2 | .10 | .0644 |
| 3 | .15 | .1560 |
| 4 | .20 | .2845 |
| 5 | .25 | .4333 |
| 6 | .30 | .5819 |
| 7 | .35 | .7132 |
| 8 | .40 | .8175 |
| 9 | .45 | .8926 |
| 10 | .50 | .9420 |
| 11 | .55 | .9718 |
| 12 | .60 | .9883 |

:::

:::{prf:proof}

The retained exponential interval $.96<c<.9608$ gives
$$
.195<\sigma,\qquad \sigma^2<.0392,\qquad
\alpha>.993,\qquad \delta<.003063,\qquad
\delta cV_c<.012,\qquad B<.0031.
$$
In the population version of (GCA74.5),
$A_r+B<(r+.020)/.993$.
On the finite event $\langle|w|\rangle_N\le1.5$ it gives
the slightly larger common bound
$$
A_r+B<\frac{r+.025}{.993}.
\tag{GCA74.10}
$$
These are exact rational comparisons. For example
$r_H>(.9996)(.0392)=.03918432$ proves the bound for
$\delta$, and $.003063/.993<.0031$ proves the bound for
$B$. The numerator $.021+.0031(.993)<.025$ proves
(GCA74.10).

For $|\mu|\le1$, (GCA74.5) therefore implies that the
shifted isotropic Gaussian $z_0$ belongs to a ball of radius
$(r+.025)/.993$. The mass of such a ball is maximal when
its mean is zero. To verify this directly, rotate the mean
onto one coordinate and slice the ball over the remaining
coordinates. Each resulting symmetric interval has translated
one-dimensional Gaussian mass maximal at zero, because the
derivative of its mass for a positive shift is the smaller
far-endpoint density minus the larger near-endpoint density.
Integrating slices proves the assertion.

The centered three-dimensional Gaussian radial formula and
$\sqrt{2/\pi}<4/5$ now give
$$
\mathbb P\{|z_0|\le(r+.025)/.993\}
\le\frac45\int_0^{R_j}u^2e^{-u^2/2}\,du\le G_j.
$$
Taylor's formula with its signed remainder gives
$e^{-v}\le\sum_{k=0}^{30}(-v)^k/k!$ for every $v\ge0$.
Integration proves the final rational polynomial bound.

For $|\mu|>1$, project $z_0$ on the unit vector $\mu/|\mu|$.
The small-ball event in (GCA74.5) implies
$$
\xi\cdot\frac{\mu}{|\mu|}
\le-\frac{(1-B)|\mu|-A_{r_j}}{\sigma}
\le-\frac{1-(r_j+.025)/.993}{\sigma}.
$$
The displayed numerator is positive for every retained $r_j$.
Markov's inequality for the squared one-dimensional standard
Gaussian gives probability at most $T_j$.

It follows that the population small-ball probability is at most
$\max\{G_j,T_j\}$. For $N\ge2$, the exact finite graph
has this bound on the finite provider event, and the complete
event's complement has probability at most $2^{-5N}\le1/1024$.
Adding that charge gives
$\max\{G_j,T_j\}+1/1024$. No Gaussian marginal on that
event has been declared fresh; only its unconditional probability
is used.

For $N=1$, $L_yw=0$ identically and $z=z_0$.
The shifted-ball argument gives the smaller bound with radius
$r_j/\sigma$, which is at most $G_j$.

Finally, exact rational evaluation of (GCA74.8) gives, for all
twelve rows,
$$
\max\{G_j,T_j\}+\frac1{1024}\le u_j.
\tag{GCA74.11}
$$
Section 6 supplies the exact arithmetic certificate. This proves
(GCA74.9) uniformly in the unbounded root mean and population size.
:::

(sec-gca74-deficit)=
## 3. A conditional velocity-square deficit for all noisy outcomes

:::{prf:theorem} Uniform actual cap defect on pre-OU test vectors
:label: thm-gca74-cap-deficit

Under the appropriate hypotheses of
{prf:ref}`def-gca74-register`,
$$
\mathbb E_\xi[1-\|D_i\|_{\mathrm{op}}^2\mid\mathscr P]
\ge\frac{41}{200}.
\tag{GCA74.12}
$$
Consequently every $\mathscr P$-fixed vector $T_i$ satisfies
$$
\mathbb E_\xi[|D_iT_i|^2\mid\mathscr P]
\le\frac{159}{200}|T_i|^2,
$$
$$
\mathbb E_\xi[
 \langle T_i,(I-D_i^2)T_i\rangle\mid\mathscr P]
\ge\frac{41}{200}|T_i|^2.
\tag{GCA74.13}
$$
The assertions are population and exact finite-array. They retain
all OU outcomes, both actual count providers, unbounded prepared
positions, and the correlated own second graph. No pointwise
velocity band or fresh-noise law after survival is assumed.
:::

:::{prf:proof}

The native radial derivative is self-adjoint and has tangential
and radial eigenvalues
$$
d_t=\frac2{2+|z|},\qquad d_r=d_t^2.
$$
Thus its operator norm is $d_t$ and, putting
$H(r)=1-[2/(2+r)]^2$, the actual defect is $H(|z|)$.
The function $H$ is increasing, $H(0)=0$, and for every outcome
$$
H(|z|)
\ge\sum_{j=1}^{12}[H(r_j)-H(r_{j-1})]
                  \mathbf1_{\{|z|>r_j\}},\qquad r_0=0.
$$
This finite lower step function does not remove the remaining
large-velocity tail; it assigns that tail its proved nonnegative
contribution. Conditional expectation and (GCA74.9) yield
$$
\mathbb E_\xi H(|z|)\ge
\sum_{j=1}^{12}[H(r_j)-H(r_{j-1})](1-u_j)
>\frac{41}{200}.
\tag{GCA74.14}
$$
The exact rational sum is between $.2057$ and $.2058$;
Section 6 verifies the strict scalar comparison without floating
point. This proves (GCA74.12). Pointwise,
$|D_iT_i|^2\le\|D_i\|_{\rm op}^2|T_i|^2$.
Since $T_i$ is fixed before OU, multiplying the scalar conditional
bound by its actual squared norm proves (GCA74.13), including
zero vectors with non-strict comparisons. All correlations inside
the preparation remain intact.
:::

(sec-gca74-signed-square)=
## 4. The force-centered square with its full correlated consumer

:::{prf:corollary} Conditional bare-velocity deficit in the exact cap account
:label: cor-gca74-signed-cap-account

At a complete prepared comparison point use the register of research68.
Write $r=\dot X$, $E=\dot U$ and
$$
R=a_xr+bE,\qquad W=c(E-tr),\qquad
Z_b=z_vE-r_Hr,\qquad F_2=B_2-L_2W.
$$
Then $R,W,Z_b$ are fixed by $\mathscr P$, and the actual complete
uncapped differential is $Z=Z_b+aF_2$.
The exact cap loss of research69 satisfies
$$
\begin{split}
\mathbb E_\xi[\mathcal L_C\mid\mathscr P]\ge{}&
\frac{41}{200}|Z_b|^2
+2\beta\,\mathbb E_\xi
 \langle R,(I-D)Z_b\rangle+\beta^2|R|^2\\
&+2a\,\mathbb E_\xi
 \langle (I-D^2)Z_b+\beta(I-D)R,F_2\rangle\\
&+a^2\,\mathbb E_\xi
 \langle F_2,(I-D^2)F_2\rangle.
\end{split}
\tag{GCA74.15}
$$
All expectations on the right are conditional on $\mathscr P$.
In particular $F_2,D$ remain under their actual joint Gaussian
and own-provider law.

For $A=\mathbb E_\xi[D\mid\mathscr P]$ and
$B=\mathbb E_\xi[D^2\mid\mathscr P]$, the actual bare capped
phase quadratic also has the exact conditional representation
$$
\mathbb E_\xi Q_\beta(R,DZ_b)
=|R|^2+\langle Z_b,BZ_b\rangle+2\beta\langle R,AZ_b\rangle,
\quad 0\le B\le\frac{159}{200}I,\quad A^2\le B.
\tag{GCA74.16}
$$
No replacement of $A$ by a scalar or of $DF_2$ by $AF_2$ is made.
:::

:::{prf:proof}

The first count and harmonic/OU stages give
$R=a_xr+bE$, $W=c(E-tr)$ and
$W-tR=z_vE-r_Hr=Z_b$. Differentiating the own second
count field gives $Z=Z_b+aF_2$. Both the complete first
spatial force in $E$ and the actual second spatial force in
$F_2$ are retained.

Expand the exact loss before any expectation:
$$
\mathcal L_C=
\langle Z,(I-D^2)Z\rangle
+2\beta\langle R,(I-D)Z\rangle+\beta^2|R|^2.
$$
Substitution of $Z_b+aF_2$ gives the right side of
(GCA74.15), with its first term replaced by
$\mathbb E_\xi\langle Z_b,(I-D^2)Z_b\rangle$.
The conditional fixed-vector estimate (GCA74.13) bounds only
that first term from below. Every mixed force term is unchanged.
This proves (GCA74.15).

Expanding $Q_\beta(R,DZ_b)$ proves (GCA74.16).
The matrix estimate for $B$ is (GCA74.13). For every fixed
vector $v$, conditional variance gives
$|Av|^2\le\mathbb E_\xi|Dv|^2=\langle v,Bv\rangle$,
hence $A^2\le B$. Symmetry of $D$ ensures symmetry of $A$.
The matrices retain root-dependent orientation and the actual
provider law; neither is presumed independent of an OU-dependent
force.
:::

(sec-gca74-survival)=
## 5. Each own normalization retains the weighted removed deficit

:::{prf:proposition} Exact deficit transfer under an own restriction
:label: prop-gca74-own-restriction

Let a raw actual proposal start from its own already current-survivor
finite law, or from an actual population input. Let $\mathcal G$ be the
pre-OU prepared budget event on which (GCA74.13) applies. In the
population case with (GCA74.3), $\mathcal G$ is the entire preparation.
For a $\mathscr P$-fixed random square-integrable test vector define
$$
\mathcal D(T)=|T|^2-|DT|^2\ge0.
$$
Let $\mathcal S$ be any own output restriction with actual probability
$p_{\mathcal S}>0$, such as the finite swarm's next nonextinction
event or its own population root-alive event. Then
$$
\begin{split}
\mathbb E[\mathcal D(T)\mid\mathcal S]\ge{}&
\frac{41}{200}\mathbb E[|T|^2\mid\mathcal S]\\
&-\frac{\frac{159}{200}\mathbb E[
             |T|^2\mathbf1_{\mathcal S^c}]
        +\frac{41}{200}\mathbb E[
             |T|^2\mathbf1_{\mathcal G^c}]}
       {p_{\mathcal S}}.
\end{split}
\tag{GCA74.17}
$$
For a finite array apply this statement to normalized row averages.

Each use divides by that proposal's own $p_{\mathcal S}$.
A different swarm requires its own numerator and denominator.
The theorem does not assert that its OU noises are independent
Gaussians after either restriction.
:::

:::{prf:proof}

Conditional on each eligible preparation, (GCA74.13) gives
$\mathbb E\mathcal D(T)\ge(41/200)
 \mathbb E[|T|^2\mathbf1_{\mathcal G}]$ after outer
integration. On the complement $\mathcal D(T)$ remains
nonnegative, which is why no negative unproved estimate is
needed there. Also $0\le\mathcal D(T)\le|T|^2$ pointwise.
Consequently
$$
p_{\mathcal S}\mathbb E[\mathcal D(T)\mid\mathcal S]
\ge\frac{41}{200}\mathbb E[
             |T|^2\mathbf1_{\mathcal G}]
    -\mathbb E[|T|^2\mathbf1_{\mathcal S^c}].
$$
Write the first expectation as $\mathbb E|T|^2-
\mathbb E[|T|^2\mathbf1_{\mathcal G^c}]$ and split
$\mathbb E|T|^2$ over $\mathcal S,\mathcal S^c$.
Division proves (GCA74.17). The argument uses only the raw
proposal law and its exact own restriction; it never relabels
a conditioned Gaussian as a new innovation.
:::

:::{prf:remark} Why the survival and empirical charges remain weighted
:label: rem-gca74-weighted-survival

Prepared recipient jitters are unbounded. The global extinction
estimate $e_N$ from the complete source-plan law is not asserted
conditional on every prepared array. Therefore it cannot replace
$\mathbb E[|T|^2\mathbf1_{\mathcal S^c}]$ by
$e_N\mathbb E|T|^2$ without a separate conditional or mixed
estimate. The same applies to a probability floor for
$\mathcal G^c$. Formula (GCA74.17) retains both actual lost
moments before each own survival or alive division.

This is a deficit of the physical pre-OU test-vector derivative.
It does not by itself couple two separately surviving output laws,
compare their accepted preparation/component laws, or identify an
exact finite-swarm QSD. Those are the actual readout and signed-law
interfaces already stated in research55,64 and68.
:::

(sec-gca74-certificate)=
## 6. Exact arithmetic certificate and remaining absorption

The following certificate checks every threshold upper bound and the
strict scalar deficit using rational arithmetic.

```python
from fractions import Fraction as F
from math import factorial

u = [
    F(".0469"), F(".0644"), F(".1560"), F(".2845"),
    F(".4333"), F(".5819"), F(".7132"), F(".8175"),
    F(".8926"), F(".9420"), F(".9718"), F(".9883"),
]
previous = F(0)
deficit = F(0)
for j, ceiling in enumerate(u, start=1):
    radius = F(j, 20)
    normalized = (radius + F(".025")) / (F(".993") * F(".195"))
    radial_upper = F(4, 5) * sum(
        F((-1) ** k) * normalized ** (2 * k + 3)
        / (2**k * factorial(k) * (2 * k + 3))
        for k in range(31)
    )
    gap = F(1) - (radius + F(".025")) / F(".993")
    tail_upper = F(".0392") / gap**2
    assert max(radial_upper, tail_upper) + F(1, 1024) <= ceiling
    value = F(1) - (F(2) / (F(2) + radius)) ** 2
    deficit += (value - previous) * (F(1) - ceiling)
    previous = value

assert F(".2057") < deficit < F(".2058")
assert deficit > F(41, 200)
print("All conditional cap-deficit rational comparisons pass.")
```

:::{prf:remark} Completed conditional coefficient and unclosed signed consumer
:label: rem-gca74-scope

The completed new endpoint is a uniform conditional cap-square deficit
$41/200$ for pre-OU-fixed vectors under the actual own second count
graph, with every Gaussian tail retained. It covers all bounded prepared
velocities satisfying the declared provider or array moments, including
genuinely noisy velocity laws. It requires neither a narrow velocity band
nor a lower bound on the physical precap speed. Gaussian cancellations
are included in the exact threshold register.

The general physical signed account remains
$-\mathfrak D_H+\mathfrak J_1+\mathfrak J_2-\mathfrak C$.
Equation (GCA74.15) now gives a uniform negative quadratic on its fixed
bare velocity differential, while retaining the exact mixed force term
$$
2a\,\mathbb E\langle
 (I-D^2)Z_b+\beta(I-D)R,F_2\rangle.
$$
That term, the oriented first-provider response, and actual
preparation/Haar/revival changes have not been absorbed in a passing
general margin here. The exact matrices in (GCA74.16) also retain the
signed bare cross term, rather than replacing it by an absolute scalar
product.

The finite exception and own restriction charges in (GCA74.17) are
additional weighted terms. A population or empirical velocity moment
alone does not discharge them. No global default nonlinear law gap,
invariant source-boundary class, finite current-survivor convergence
rate or $N$-uniform exact QSD rate follows from this conditional result.
It supplies a stronger full-Gaussian cap consumer for those remaining
signed interfaces, while preserving their unfinished scope.
:::

