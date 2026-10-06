# Independent audit of the exact reference alive-readout regularities

(sec-tqpa-retained)=
## 1. Exact source and inherited hypotheses

:::{prf:definition} Audited source and readout conventions
:label: def-tqpa-source

This independent review concerns
`56_reference_alive_readout_regularities.md` at SHA-256
`8789f44b06915bfa33b5791d149ed9e00664c4b5572abb2fd6dfd4912d1e913b`.
Its subsequent syntax-only revision has SHA-256
`6fffe66b95b42cbd84c93e9c4de4161be0bc2f1021736eee8684c22aeae919f2`:
one space was inserted between the fraction and indicator braces in
(TQP.1) to prevent MyST substitution parsing. The mathematical
expression and every audited conclusion are unchanged. Both
revisions pass the present review.
Its inherited register is {prf:ref}`def-tat-register`:
the actual harmonic force and raw same-potential reward,
$h=.04$, $\nu=.3$, $d=3$, $L=2$, native cap $V=2$,
both count and row normalization, current-frame measured fitness
and each own nonextinction event. The other explicitly disabled
feedback branches stay disabled. Positive configured fitness
exponents need not be weakened: collapsed inputs have exact ties.

The input $S(m)$ has every position $m\in[-.5,.5]^3$,
every original velocity zero and every mark alive.
All accepted-copy gates vanish. The first viscous force vanishes,
and the actual proposed positions are independent
$N(am,\tau^2I_3)$, where $a=a_x$ and
$\tau^2=t^2q^2+s^2>0$. The full second field and cap act on
velocities and do not alter these landing positions.
The terminal alive test is the actual positional box test.

The sampled positional laws and the law of the random normalized
alive positional empirical probability are different targets.
Every outer empirical-law metric in this audit uses inner physical
Euclidean positional $W_2$. The inherited phase comparison uses
Euclidean phase $W_2$, as declared in the source.
:::

(sec-tqpa-sampled)=
## 2. Sampled-law identification and quantile contraction

:::{prf:remark} Exact own-survival sampled laws
:label: rem-tqpa-sampled

The source's identification
$\lambda_N^x(m)=\widetilde\lambda_N^x(m)=g_m$ is valid for
every $N\ge1$. The positional independence is an actual property
of the landing coordinates; it does not require independent
second-stage velocity fields. Conditional on any nonempty alive
index set, the alive positions are independent with common
truncated Gaussian law $g_m$. Sampling an alive slot uniformly
therefore has that law before or after mixing over index sets.

For the initially uniform original-slot convention, conditioning
that slot to be alive gives the same truncated law. That event
already implies this swarm's own nonextinction. This proves the
second convention without changing its normalizer or conditioning
a coupling on joint survival. No Gaussian event is discarded.
The argument also permits different population sizes on the two
sides, since the sampled positional laws have no remaining size
dependence.
:::

:::{prf:remark} Quantile differentiation and the physical transport direction
:label: rem-tqpa-quantile

For $0<u<1$, the conditional Gaussian quantile equation in the
source has strictly positive implicit derivative denominator.
Differentiating it gives exactly
$$
\partial_\mu x_\mu(u)
=1-\frac{(1-u)\phi(A)+u\phi(B)}{\phi(z)}.
$$
The Gaussian profile $H(v)=\phi(\Phi^{-1}(v))$ obeys
$H'(v)=-\Phi^{-1}(v)$ and $H''(v)=-1/H(v)<0$.
The quantile equation expresses $\Phi(z)$ as the convex
combination of $\Phi(A),\Phi(B)$, so concavity gives a ratio
between zero and one. Thus $0\le\partial_\mu x_\mu(u)\le1$,
including its sign. Integrating the derivative between any two
finite means is justified because the quantile remains smooth.

The box-truncated isotropic Gaussian factors into its three
coordinate-truncated laws. A shared uniform quantile in each
coordinate is an admissible coupling with squared cost at most
$|am-a\widetilde m|^2$. Taking an infimum over output couplings
has the correct upper-bound direction. Finally
$a=1-.0004(1+c)<.999216$ because $c>.96$.
This proves the source's exact sampled-position contraction with
no minimum separation, finite-particle floor or hidden tail error.
It does not assert full phase sampled-law contraction.
:::

(sec-tqpa-strata)=
## 3. Exact cardinality strata and unavoidable crossing mass

:::{prf:remark} Mixture weights under each actual nonextinction event
:label: rem-tqpa-mixture

At $N=2$ the alive count is binomial with parameter $p(m)$.
Its own nonextinction probability is $p(2-p)$, so the two exact
conditional mixture weights are
$$
w_1(p)=\frac{2(1-p)}{2-p},\qquad
w_2(p)=\frac{p}{2-p}.
$$
The singleton and two-point laws are disjoint almost surely:
the latter has two distinct positions because $g_m$ has a
continuous density. The set of singleton probabilities is Borel
in $\mathcal P_2(D)$, since it is the compact image of $D$ under
$x\mapsto\delta_x$. The mixture's cardinality strata can
therefore be used under every outer-law coupling.

For $m=(r,0,0)$ with $r>0$, the first-coordinate box probability
has derivative
$$
P'(r)=\frac a\tau
\left[\phi((2+ar)/\tau)-\phi((2-ar)/\tau)\right]<0.
$$
The other coordinate factors are fixed and strictly positive.
The derivative $w_1'(p)=-2/(2-p)^2$ is negative, so the
singleton mixture mass strictly increases along this center
segment. The positivity is exact despite its small numerical size.

For a coupling $\Gamma$ and the singleton set $A$, the identity
$$
\Gamma(A^c\times A)-\Gamma(A\times A^c)
=w_1(p(m_\delta))-w_1(p(m_0))=s_\delta
$$
forces $\Gamma(A^c\times A)\ge s_\delta$.
This is a universal coupling lower bound, not a cost of one
chosen source or innovation coupling. It uses the two separate
survivor-normalized marginals exactly as stated.
:::

(sec-tqpa-variance)=
## 4. Physical variance cost, explicit constants and phase projection

:::{prf:remark} The universal positional lower bound
:label: rem-tqpa-variance

For any input two-point measure and output singleton, the only
transport sends both input atoms to that output point. Completing
its square gives
$$
W_2^2\left(\frac{\delta_x+\delta_y}{2},\delta_z\right)
=\left|z-\frac{x+y}{2}\right|^2+\frac{|x-y|^2}{4}.
$$
Thus the required crossing mass pays a positional variance cost
even though no discrete status cost occurs in this metric.

The input $g_{m_0}$ has finite positive maximum density. For
independent input points, the ordinary three-dimensional ball
volume gives $\Pr\{|X-Y|\le r\}\le Cr^3$ with
$C=(4\pi/3)\|g_{m_0}\|_\infty$. Multiplying by the input
two-point mixture weight only decreases this bound. The close-pair
event is invariant under exchanging the two atoms, so its
submass is a well-defined event of the two-point probability.
Its use does not require a chosen labeling in an arbitrary outer
coupling.

Taking $r_\delta=(s_\delta/(2C))^{1/3}$ leaves at least
$s_\delta/2$ crossing mass with physical squared cost at least
$r_\delta^2/4$. The resulting coefficient is exactly
$$
\frac{s_\delta^{5/3}}{8(2C)^{2/3}}.
$$
All exponents and factors in the source are correct.

For the first audited source's explicit positive constants put
$$
P_* = \Phi(2/\tau)-\Phi(-2/\tau),\qquad
p_0=P_*^2\left[\Phi((2-a/4)/\tau)-\Phi((-2-a/4)/\tau)\right].
$$
The source's $c_0=f'(1/4)/2$ can be written as
$$
c_0=\frac{aP_*^2}{\tau(2-p_0)^2}
\left[\phi((2-a/4)/\tau)-\phi((2+a/4)/\tau)\right]>0.
$$
The Gaussian mean $am_0$ lies inside $D$, so
$$
C=\frac{4\pi}{3(2\pi\tau^2)^{3/2}p_0}>0,
\qquad C_0=\frac{c_0^{5/3}}{8(2C)^{2/3}}>0.
$$
A precise realization of its positive neighborhood is
$\delta_0=\ell_*/2$, where
$$
\ell_* = \sup\left\{\ell\in(0,1/4]:
f'(r)\ge c_0\ \text{for every }r\in[1/4,1/4+\ell]\right\}>0.
$$
Continuity and $f'(1/4)=2c_0$ ensure positivity. The admissible
length set is downward closed, so its half-supremum is admissible.
Integration gives $s_\delta\ge c_0\delta$ on the stated
range. This verifies the source's explicit constant definitions.

The deterministic input empirical laws have outer distance
$\delta$. The squared output lower bound $C_0\delta^{5/3}$
therefore gives ratio at least $\sqrt{C_0}\delta^{-1/6}$,
which diverges. Positional projection is $1$-Lipschitz in the
declared Euclidean phase metric. Every phase outer coupling
projects to an admissible positional outer coupling, hence the
phase cost has the same lower bound. The proof does not assert
this coefficient for a different phase quadratic without the
corresponding projection constant.
:::

(sec-tqpa-endpoint)=
## 5. Audit endpoint and limits of the two results

:::{prf:remark} Independent audit conclusion
:label: rem-tqpa-endpoint

Both {prf:ref}`thm-tqp-sampled-position` and
{prf:ref}`thm-tqp-empirical-local-regularity` pass this independent
review. The sampled-position theorem is a positive exact one-step
statement for every population size and every center separation.
The empirical-law theorem is a universal local one-step regularity
lower bound for its different, random-cardinality readout at $N=2$.
Their targets, infimum directions and own-survival normalizers are
consistent. Every Gaussian tail remains in the probabilities and
derivatives; both count and row second fields remain in the full
phase algorithm.

Neither result asserts invariance of the collapsed input class,
nor estimates repeated active preparation, delayed mixing or a
quasi-stationary rate. In particular the local lower bound does
not refute the requested delayed size-independent alive-law
mixing. No manuscript edits are required by this audit.
:::

(sec-tqpa-explicit-update)=
## 6. Strengthened explicit neighborhood

:::{prf:remark} Final constant update and preserved source history
:label: rem-tqpa-explicit-update

The subsequently strengthened source has SHA-256
`e31af59f3dd06f30618d152f56c3bb39646b68c3088c30c6fe104ab76793064e`.
It replaces the first source's continuity neighborhood by the explicit
$\delta_0=1/4$ and uses $c_0=-p'(1/4)/2$. All other targets
and the cardinality-crossing proof retain their meaning.

Indeed, on $r\in[1/4,1/2]$, both $2-ar$ and $2+ar$ are
positive, and differentiating the exact one-dimensional probability
gives
$$
-P''(r)=\frac{a^2}{\tau^3}
\left[(2-ar)\phi((2-ar)/\tau)
                         +(2+ar)\phi((2+ar)/\tau)\right]>0.
$$
Consequently $-p'(r)=P_*^2[-P'(r)]$ increases throughout that
interval. Since $0<p(r)<1$, the exact mixture derivative obeys
$$
f'(r)=\frac{-2p'(r)}{(2-p(r))^2}
\ge\frac{-p'(r)}2\ge\frac{-p'(1/4)}2=c_0>0.
$$
Thus $s_\delta\ge c_0\delta$ for every $0<\delta\le1/4$.
The final explicit constant is
$$
c_0=\frac{aP_*^2}{2\tau}
\left[\phi((2-a/4)/\tau)-\phi((2+a/4)/\tau)\right]>0.
$$
The source's maximum-density expression
$C=4\pi/[3p_0(2\pi\tau^2)^{3/2}]$ is exact because its
Gaussian mean $am_0$ lies inside the box. Therefore the declared
$C_0=c_0^{5/3}/[8(2C)^{2/3}]$ and $\delta_0=1/4$ prove
the same universal outer-law lower bound on this explicit interval.
This strengthening passes the independent review. It changes no
projection constant, Gaussian tail or delayed-mixing scope.
:::
