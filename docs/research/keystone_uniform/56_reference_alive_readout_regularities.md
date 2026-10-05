# Exact sampled-position transport and local empirical-law regularity

(sec-tqp-register)=
## 1. The unchanged tied-input transition

:::{prf:definition} Positional alive readouts of the default tied class
:label: def-tqp-register

Use every algorithmic restriction and constant in
{prf:ref}`def-tat-register`. In particular, the harmonic force and raw
same-potential reward, both count and row normalization, each own
nonextinction event, and every Gaussian tail are retained. Let
$D=[-2,2]^3$, $\tau^2=t^2q^2+s^2>0$ and $a=a_x$.
For the deterministic all-alive collapsed input $S(m)$ with zero
velocities, the actual proposed terminal positions are independent
$N(am,\tau^2I_3)$ variables. This follows from
{prf:ref}`lem-tat-own-array-translation`; the actual second kick and
native cap change velocities only.

Write
$$
p(m)=\int_D\phi_{\tau,3}(x-am)\,dx,\qquad
g_m(dx)=\frac{ {\bf1}_D(x)\phi_{\tau,3}(x-am)}{p(m)}\,dx.
\tag{TQP.1}
$$
Let $\lambda^x_N(m)$ be the swarm-first uniformly alive-sampled
positional law, after conditioning on its own nonextinction.
Let $\widetilde\lambda^x_N(m)$ instead choose an original slot
uniformly and condition that slot to be alive; adding the same
swarm's nonextinction conditioning gives the same law.
Let $\mathcal E_N^x(m)$ be the law of the actual random alive
positional empirical probability, under its own nonextinction.
The outer metric $\mathbb W_{2,x}$ uses the inner physical
positional $W_2$ as ground metric. These are different readouts.
:::

(sec-tqp-sampled)=
## 2. A positive sampled-position estimate without separation

:::{prf:theorem} Exact sampled-position contraction for every population size
:label: thm-tqp-sampled-position

For every $N\ge1$ and $m,\widetilde m\in[-.5,.5]^3$,
$$
\lambda_N^x(m)=\widetilde\lambda_N^x(m)=g_m,\qquad
W_2(g_m,g_{\widetilde m})
\le a|m-\widetilde m|
<.999216|m-\widetilde m|
\quad(m\ne\widetilde m).
\tag{TQP.2}
$$
For identical centers the distance is zero. The estimate has no
minimum center separation, no particle floor, and no dependence
on $N$. It also compares the sampled positional laws for different
population sizes. It is a one-update theorem on collapsed tied
inputs, not an iteration theorem for the noisy output class.
:::

:::{prf:proof}
The alive indicators are independent Bernoulli variables with
parameter $p(m)$. Conditional on any nonempty alive index set,
the positions at its alive slots are independent with common law
$g_m$. Uniform sampling from that set therefore has law $g_m$.
Mixing over nonempty sets under the swarm's own survival
normalization proves the first equality for $\lambda_N^x$.
Conditioning an initially uniform slot to be alive also gives
$g_m$, since each slot has the same Gaussian law. This event already
implies that swarm's nonextinction, proving the other equality.

It remains to construct a coupling of the truncated Gaussians.
In one dimension put $L=2$ and let $x_\mu(u)$ be the quantile of
$N(\mu,\tau^2)$ conditioned on $[-L,L]$, for $0<u<1$.
With
$$
A=\frac{-L-\mu}{\tau},\qquad B=\frac{L-\mu}{\tau},\qquad
z=\frac{x_\mu(u)-\mu}{\tau},
$$
the quantile equation is
$\Phi(z)=(1-u)\Phi(A)+u\Phi(B)$.
The implicit-function theorem applies since $\phi(z)>0$.
Differentiating in $\mu$ gives
$$
\partial_\mu x_\mu(u)
=1-\frac{(1-u)\phi(A)+u\phi(B)}{\phi(z)}.
\tag{TQP.3}
$$
The function $H(v)=\phi(\Phi^{-1}(v))$, $0<v<1$, satisfies
$H'(v)=-\Phi^{-1}(v)$ and $H''(v)=-1/H(v)<0$.
Its concavity gives
$\phi(z)\ge(1-u)\phi(A)+u\phi(B)$.
The numerator is positive, so
$0\le\partial_\mu x_\mu(u)\le1$.
Consequently
$|x_\mu(u)-x_{\widetilde\mu}(u)|\le|\mu-\widetilde\mu|$
for every $u$.

The box-truncated three-dimensional Gaussian in (TQP.1) is the
product of these three one-dimensional laws. Use a common
independent uniform quantile in each coordinate. Its squared
transport cost is at most
$|am-a\widetilde m|^2$. This is an admissible physical coupling,
so taking the optimal cost proves (TQP.2).
The strict numerical coefficient is the exact certificate
$a<.999216$ in {prf:ref}`lem-tat-own-array-translation`.
The proof uses the complete truncated density and each exact
normalizer $p(m)$, without deleting a boundary or Gaussian event.
:::

(sec-tqp-empirical)=
## 3. A narrow obstruction to one-update empirical-law Lipschitz continuity

:::{prf:theorem} Actual alive empirical-law map need not be locally Lipschitz
:label: thm-tqp-empirical-local-regularity

Keep the same unchanged default tied-input transition and take $N=2$.
Set $m_0=(1/4,0,0)$ and $m_\delta=m_0+\delta e_1$.
There are explicitly defined $c_0,C_0>0$ and $\delta_0=1/4$ such that,
for $0<\delta\le\delta_0$,
$$
\mathbb W_{2,x}\bigl(\mathcal E_2^x(m_0),
                         \mathcal E_2^x(m_\delta)\bigr)^2
\ge C_0\delta^{5/3}.
\tag{TQP.4}
$$
Thus its ratio to the squared input empirical distance $\delta^2$
is unbounded as $\delta\downarrow0$.
The same failure of local Lipschitz continuity holds for the
phase empirical-law metric of {prf:ref}`def-tat-register`.
This statement concerns this one-update map. It does not preclude
delayed law mixing or any already proved long-time estimate.
:::

:::{prf:proof}
Let $p_0=p(m_0)$. For $N=2$, each own exact survivor-conditioned
positional empirical law is the mixture
$$
\mathcal E_2^x(m)
=w_1(p(m))\,\operatorname{Law}(\delta_X)
+w_2(p(m))\,\operatorname{Law}
       \left(\frac{\delta_X+\delta_Y}{2}\right),
\tag{TQP.5}
$$
where $X,Y$ are independent with law $g_m$ and
$$
w_1(p)=\frac{2(1-p)}{2-p},\qquad
w_2(p)=\frac{p}{2-p}.
\tag{TQP.6}
$$
Indeed the raw singleton and double-alive probabilities are
$2p(1-p)$ and $p^2$, and the swarm's own survival probability is
$1-(1-p)^2=p(2-p)$. This proves (TQP.5), including its own
survival division.

For a positive first coordinate $r$, its one-dimensional alive
probability is
$$
P(r)=\Phi((2-ar)/\tau)-\Phi((-2-ar)/\tau),
$$
and
$$
P'(r)=\frac a\tau
 \left[\phi((2+ar)/\tau)-\phi((2-ar)/\tau)\right]<0
\quad(0<r\le1/2).
\tag{TQP.7}
$$
The other coordinate probabilities are constant on the chosen
center segment. Also $w_1'(p)=-2/(2-p)^2<0$.
Define $f(r)=w_1(p((r,0,0)))$ and
$$
c_0=-\frac12
 \left.\frac{d}{dr}p((r,0,0))\right|_{r=1/4}>0.
$$
The function $-P'(r)$ increases on $[1/4,1/2]$: differentiating
its positive Gaussian density difference gives a positive sum of
the two terms $(2-ar)\phi((2-ar)/\tau)$ and
$(2+ar)\phi((2+ar)/\tau)$, multiplied by $a^2/\tau^3$.
The other two coordinate probabilities are fixed. Since
$(2-p)^2\le4$, throughout that interval
$f'(r)=-2p'(r)/(2-p(r))^2\ge-p'(r)/2\ge c_0$.
Hence the additional output singleton mass is
$$
s_\delta=f(1/4+\delta)-f(1/4)\ge c_0\delta>0.
\tag{TQP.8}
$$
All these probabilities include the Gaussian tails. Their
derivatives can be very small; their positivity is what is used.

Let $\Gamma$ be any coupling of the two actual positional empirical
laws. The set of singleton probabilities
$\{\delta_x:x\in D\}$ is compact, hence Borel, in $\mathcal P_2(D)$.
Since $g_m$ has a density, the two-point component has two distinct
points almost surely. Because the output singleton mass exceeds
the input singleton mass by $s_\delta$, at least $s_\delta$ of
$\Gamma$ must send an input two-point probability to an output
singleton probability. This mass statement holds for every
coupling, not just the common-innovation pairing.

For every $x,y,z\in D$,
$$
W_2^2\left(\frac{\delta_x+\delta_y}{2},\delta_z\right)
=\left|z-\frac{x+y}{2}\right|^2+\frac{|x-y|^2}{4}
\ge\frac{|x-y|^2}{4}.
\tag{TQP.9}
$$
This is a physical positional variance cost, with no status
penalty. Put
$$
C=\frac{4\pi}{3}\|g_{m_0}\|_\infty
 =\frac{4\pi}{3p_0(2\pi\tau^2)^{3/2}}>0.
$$
For independent $X,Y\sim g_{m_0}$,
$$
\Pr\{|X-Y|\le r\}
=\int g_{m_0}(dx)\int_{D\cap B(x,r)}g_{m_0}(dy)
\le Cr^3.
\tag{TQP.10}
$$
The input two-point mixture weight is at most one, so the same
bound applies to its unconditional submass. Choose
$r_\delta=(s_\delta/(2C))^{1/3}$.
Of the required crossing mass, at most $s_\delta/2$ has
$|X-Y|\le r_\delta$. At least $s_\delta/2$ therefore has
physical cost at least $r_\delta^2/4$, by (TQP.9). Consequently
every $\Gamma$ satisfies
$$
\int W_2^2(\alpha,\widetilde\alpha)\,d\Gamma
\ge \frac{s_\delta^{5/3}}{8(2C)^{2/3}}
\ge\frac{c_0^{5/3}}{8(2C)^{2/3}}\delta^{5/3}.
\tag{TQP.11}
$$
Set $C_0=c_0^{5/3}/[8(2C)^{2/3}]$ and
$\delta_0=1/4$. Taking the infimum over all couplings proves
(TQP.4). The input empirical laws are point masses at $\delta_{m_0}$
and $\delta_{m_\delta}$, with outer distance exactly $\delta$.
Their output/input distance ratio is at least
$\sqrt{C_0}\delta^{-1/6}$, which diverges.

Finally positional projection is $1$-Lipschitz from Euclidean
phase space, hence from its inner measure metric and then its
outer law metric. Every phase empirical-law coupling projects to
a positional empirical-law coupling. Its cost is no smaller,
so the same lower bound holds for the actual phase target.
All second providers, stored velocities and cap outcomes remain
in those phase marginals; projection is used only for a universal
lower bound. It makes no change to the algorithm.
:::

:::{prf:remark} Scope for uniform law estimates
:label: rem-tqp-law-scope

The lower bound uses actual probability laws of alive empirical
measures under each own nonextinction normalization, and a cost
of physical locations. It uses neither retained dead coordinates
nor a prescribed source coupling. It shows why a uniform local
one-update Lipschitz argument for that particular readout cannot
remove the separated-input hypothesis in
{prf:ref}`thm-tat-finite-alive-transport`.
It does not invalidate that theorem or refute the user's delayed
$N$-uniform alive-law mixing target. An additive Gaussian-tail
budget and a minimum separation gave a valid different estimate
there.

The exact sampled-position contraction (TQP.2) has no such
separation condition: averaging the alive empirical probability
removes its random-cardinality strata. That theorem is positional,
not a full phase sampled-law estimate, since conditioned OU
velocities and the actual second provider still enter the latter.
Neither theorem asserts that the collapsed zero-velocity class
is preserved after one update.
:::
