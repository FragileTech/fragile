# Polynomial native joint smoothing and full-law block entropy

(sec-nje-register)=
## 1. Complete native parameters and the sharper comparison

:::{prf:definition} Native joint entropy register
:label: def-nje-register

Retain every field and every primitive test of
{prf:ref}`def-nue-register`, including $\sigma_J>0$,
$\kappa_A>0$, the actual positive-viscosity quadratic count phase,
sampled fitness/normalization, donor conventions, mandatory revival,
simultaneous frozen copy, full component Haar collision, unbounded
original jitters and OU/position Gaussians, configured cap, terminal
statuses, arithmetic, all passive records and native calibration.
All consumed parameters remain fixed independently of $N$.
This chapter sharpens the analysis of the same transition.

Use $P_{x,N},P_{v,N},J_A,K_m,M_*,D,\bar q$ from
(NUE.2), (NUE.6) and (NUE.12), and put

$$
\begin{gathered}
\widehat B_N
=g_1/s+Ng_1/\sigma_J+2P_{x,N}+4/(s\sqrt{2\pi}),\\
\widehat K_{v,N}
=2P_{v,N}+k_cNd[t\widehat B_N/\kappa_A+J_A]
                         +cg_1k_cN/(\kappa_Aq),\\
\widehat L_N=N[K_mM_*+\widehat K_{v,N}\bar q/\omega].
\end{gathered}
\tag{NJE.1}
$$

These constants contain no number of discrete patterns. They keep
the actual donor/component dependence through the proved original
probability and collision budgets.
:::

(sec-nje-aggregate)=
## 2. Joint source variation summed before estimates

:::{prf:lemma} Complete aggregated pattern variation
:label: lem-nje-aggregated-source

In the Gaussian-refreshed input instrument
{prf:ref}`lem-nue-refreshed-instrument`, let
$\mu_v^\pi$ be the unnormalized prepared-position density of a
nonextinct incoming mask/donor/gate pattern, at its fixed normalized
component Haar variables. Sum over every original pattern and integrate
its original normalized Haar law. Then

$$
\begin{gathered}
\sum_\pi\int\mu_v^\pi\,dX\le1,\qquad
\sum_\pi\sum_{i,a}|D_{X_{ia}}\mu_v^\pi|(\mathbb R^{Nd})
                                  \le Nd\widehat B_N,\\
\sum_\pi\|\dot\mu_v^\pi\|_{\rm var}
\le2P_{v,N}\sum_i|\dot v_i|.
\end{gathered}
\tag{NJE.2}
$$

The incoming all-dead branch has zero mass in this submeasure.
No pattern is removed and no pattern probability is divided out.
The spatial variation includes the incoming classification faces.
:::

:::{prf:proof}
Write $p_\pi(x,v)$ for the actual unnormalized probability of the
finite donor/gate part, on each nonempty incoming mask cell.
Its original normalized pattern sum is one on that cell.
The original common-mass companion/gate coupling used in (NUE.6)
gives, in every fixed input coordinate,

$$
\sum_\pi|\partial_{x_{ia}}p_\pi|\le2P_{x,N},\qquad
\sum_\pi|\dot p_\pi|
\le2P_{v,N}\sum_i|\dot v_i|.
\tag{NJE.3}
$$

This is the full variation derivative, twice the probability-TV
derivative. It is proved before any pattern-dependent continuous
change of variables. The sole-alive conventions have zero donor/gate
derivatives, as checked in the native joint bridge proof.

For each prepared coordinate repeat its exact integration by parts.
If it is copied, its own jitter score costs at most $g_1/\sigma_J$.
The pattern is drawn before jitter, so its probability is independent
of that original Gaussian. Summing the copied contributions over
patterns first costs this score times the total copied probability,
which is at most one.

If it is persistent, translate its incoming source and compensate
all copied-recipient jitters having this source. The incoming Gaussian
score costs $g_1/s$ times the total persistent probability, at most
one. A persistent coordinate has at most $N-1$ copied recipients
with that source, since it is not itself copied.
Its recipient scores are independent of the pre-jitter pattern, so
the sum of copied own-score and persistent compensation-score costs
is at most
$[\Pr(\text{copied})+(N-1)\Pr(\text{persistent})]g_1/\sigma_J
\le Ng_1/\sigma_J$.
The original pattern derivatives sum to $2P_{x,N}$ by (NJE.3).
Pushforward may decrease each signed derivative's variation; it
cannot increase these integrated score bounds. Different patterns
need not have the same prepared-coordinate map.

For each incoming face, add the absolute traces of all pattern weights
before bounding. On either side their total is at most one. A source
coordinate has two box faces, and its incoming Gaussian density there
is at most $1/(s\sqrt{2\pi})$. Thus the full two-sided trace budget
is at most $4/(s\sqrt{2\pi})$ in that coordinate.
This safe bound also covers the face incident to the omitted
all-dead cell. There is no differentiation of a fixed terminal
status using a continuous gradient alone.

These four terms are exactly $\widehat B_N$.
There are $Nd$ prepared scalar coordinates, which proves the
spatial derivative bound. Holding the source/jitter map fixed,
the parameter derivative changes only the original pattern weight;
(NJE.3) and signed pushforward give the final bound in (NJE.2).
Finally the actual patterns and normalized Haar laws are a
subprobability partition of the original incoming integral,
proving the mass bound.
:::

:::{prf:theorem} Polynomial full joint two-update TV bridge
:label: thm-nje-polynomial-joint-bridge

The same complete Gaussian-refreshed input instrument satisfies

$$
\|\mathcal R_N(m,v)-\mathcal R_N(m',v')\|_{\rm TV}
\le K_m\sum_i|\Delta m_i|
                         +\widehat K_{v,N}\sum_i|\Delta v_i|.
\tag{NJE.4}
$$

Consequently, for $S,S'\in H_N$ with $m_0N\ge2$ and
$q_N\le\bar q$,

$$
\|Q_N^2(S,\cdot)-Q_N^2(S',\cdot)\|_{\rm TV}
\le\widehat L_N\,\overline d_\omega(S,S').
\tag{NJE.5}
$$

Every inequality also bounds the difference of expectations of an
arbitrary measurable test in $[0,1]$. This is joint full-state smoothing.
:::

:::{prf:proof}
Use the actual first-drift inverse and its velocity transport field
$H^\pi$ from (NUE.10)--(NUE.11), separately for each fixed pattern
and normalized Haar realization. Although these fields can differ,
their maximum-row, sum-row and divergence bounds are uniform:

$$
\begin{gathered}
\|H^\pi\|_\infty
\le tk_c\kappa_A^{-1}\|\dot v\|_\infty,\\
|\operatorname{div}H^\pi|
\le Ndk_cJ_A\|\dot v\|_\infty,\qquad
\sum_i|H_i^\pi|
\le tk_cN\kappa_A^{-1}\sum_i|\dot v_i|.
\end{gathered}
\tag{NJE.6}
$$

The source parameter derivatives now sum to $2P_{v,N}$ by (NJE.2).
The source transport derivatives sum to at most
$tk_c\kappa_A^{-1}\|\dot v\|_\infty$
times the complete summed spatial variation.
Their divergence terms sum with the complete subprobability mass,
which is at most one. The OU mean scores likewise integrate against
these same subprobabilities: at fixed $x_1$ their mean derivative is
$-cH^\pi/t$. Conditional Gaussian score expectation gives
$cg_1k_cN/(\kappa_Aq)\sum_i|\dot v_i|$ in total.
Together these are exactly $\widehat K_{v,N}$.

Every B2, final-noise, cap and classification operation is the same
common measurable instrument as in the original bridge proof.
No B2 inverse or independence between different prepared rows has
been assumed. The input-center bound $K_m$ is unchanged.
Integrate the velocity interpolation to prove (NJE.4).
Applying its actual first-update preparation/OU coupling, exactly
as in (NUE.12)--(NUE.13), proves (NJE.5).
:::

(sec-nje-uniform)=
## 3. Linear horizon and exponentially small native Doob tilt

:::{prf:definition} Polynomial population threshold
:label: def-nje-threshold

Set

$$
\begin{gathered}
\widehat B_1=g_1/s+g_1/\sigma_J+2P_{x,1}+4/(s\sqrt{2\pi}),\\
\widehat K_1=2P_{v,1}
 +k_cd[t\widehat B_1/\kappa_A+J_A]
 +cg_1k_c/(\kappa_Aq),\\
\widehat C_L=K_mM_*+\widehat K_1\bar q/\omega,\qquad
\widehat C_0=\log_+(D\widehat C_L),\\
u=a_0/8,\quad\ell=-\log\bar q,\qquad
\widehat C_k=1+(\widehat C_0+6+u)/\ell,\\
\widehat C_T=\widehat C_k+3,\qquad
\widehat C_\theta=2\widehat C_k+3 .
\end{gathered}
\tag{NJE.7}
$$

Let $\widehat N_*$ be the ceiling of the maximum of $2$, $4/a_0$,

$$
\begin{gathered}
\frac4{r_g^2}\log_+\frac{2\sqrt2L_{\rm cap}}{1-r},
\qquad \frac{\log2}{a_0},\\
\frac2{a_0}\log_+
                   \frac{4\widehat C_T}{ea_0\log2},
\qquad
\frac2u\log_+\frac{32\widehat C_\theta}{eu}.
\end{gathered}
\tag{NJE.8}
$$

For $N\ge\widehat N_*$ define the actual native-update integers and
explicit probability bounds

$$
\begin{gathered}
\widehat k_N=\left\lceil
 \frac{\widehat C_0+6\log N+uN}{\ell}\right\rceil,\qquad
\widehat T_N=\widehat k_N+3,\\
\widehat b_N=2\widehat C_\theta Ne^{-uN}\le1/8,\qquad
\widehat m_N=1-\widehat b_N,\qquad
\Delta_N=\frac{2\widehat b_N}{1-\widehat b_N}\le2/7 .
\end{gathered}
\tag{NJE.9}
$$

These horizons leave every update and recording/calibration rule
unchanged.
:::

:::{prf:theorem} Improved eigenfunction and full Doob block comparison
:label: thm-nje-improved-eigenfunction

For every permitted $N\ge\widehat N_*$,

$$
1-\widehat b_N\le e_N\le1,\qquad
\frac{\max e_N}{\min e_N}\le(1-\widehat b_N)^{-1},\qquad
\sup_{S,S'}\|(P_N^e)^{\widehat T_N}(S,\cdot)
                   -(P_N^e)^{\widehat T_N}(S',\cdot)\|_{\rm TV}
\le\Delta_N .
\tag{NJE.10}
$$

Thus $\widehat T_N=O(N)$ and
$\operatorname{osc}e_N=O(Ne^{-a_0N/8})$ with primitive constants.
All stationary and complete-history comparisons of
{prf:ref}`thm-nue-doob-comparison` hold with
$b_N$ replaced by $\widehat b_N$.
:::

:::{prf:proof}
The expressions $P_{x,N},P_{v,N}$ have nonnegative polynomial
coefficients and degree at most four. Hence
$\widehat B_N\le N^4\widehat B_1$,
$\widehat K_{v,N}\le N^5\widehat K_1$ and
$\widehat L_N\le N^6\widehat C_L$.
Apply the original stopped high-alive coupling (NUE.19)--(NUE.20),
now with the proved joint bridge (NJE.5). For every $0\le f\le1$,

$$
\operatorname{osc}Q_N^{\widehat T_N}f
\le(2\widehat k_N+3)e^{-uN}.
\tag{NJE.11}
$$

The bad-event cost is not multiplied by the bridge constant.
Since $\log N\le N$,
$\widehat k_N\le\widehat C_kN$ and
$\widehat T_N\le\widehat C_TN$.
For $v>0$, $Ne^{-vN}\le2(ev)^{-1}e^{-vN/2}$.
The thresholds (NJE.8) therefore imply
$(1-\varepsilon_N)^{-\widehat T_N}\le2$
and $\widehat b_N\le1/8$.
Use $Q_N^{\widehat T_N}e_N=\alpha_N^{\widehat T_N}e_N$,
$\alpha_N\ge1-\varepsilon_N$ and $\max e_N=1$ to obtain
the improved eigenfunction bound.

Also $\alpha_N^{\widehat T_N}\ge1/2$, so (NJE.11) is at most
$\alpha_N^{\widehat T_N}\widehat b_N$.
The exact Doob formula and its numerator/denominator subtraction
give the event-TV bound $2\widehat b_N/(1-\widehat b_N)$,
as in (NUE.27).
The stationary/history comparisons only use the same eigenfunction
range and exact terminal telescope, so their improvement follows.
:::

(sec-nje-entropy)=
## 4. Exact full-law entropy contraction for a native block

:::{prf:lemma} Dobrushin contraction of every hockey-stick divergence
:label: lem-nje-hockey-stick

Let $K$ be any Markov kernel on the actual measurable state space,
with probability-TV Dobrushin coefficient at most $\delta$.
For probabilities $P,Q$ and $\gamma\ge1$, define

$$
E_\gamma(P\Vert Q)=(P-\gamma Q)^+(\mathsf E).
$$

Then

$$
E_\gamma(PK\Vert QK)
\le\delta E_\gamma(P\Vert Q),\qquad
D(PK\Vert QK)\le\delta D(P\Vert Q).
\tag{NJE.12}
$$

The entropy inequality holds for extended values and includes
discrete status differences and every coordinate.
:::

:::{prf:proof}
Take the Jordan decomposition
$P-\gamma Q=aU-bV$, with $U,V$ probabilities,
$a=E_\gamma(P\Vert Q)$ and $b=a+\gamma-1$.
If $a=0$ positivity preservation gives zero output divergence.
Otherwise

$$
PK-\gamma QK=a(UK-VK)-(\gamma-1)VK.
$$

The positive mass of this signed measure is at most
$a\|UK-VK\|_{\rm TV}\le a\delta$.
The bound for two arbitrary input probabilities follows from the
row Dobrushin coefficient by integrating their row common-mass
couplings, or by taking the event supremum. This proves the
hockey-stick inequality without densities or reversibility.

Choose any common dominating measure with densities $p,q$.
Tonelli and the elementary integrals at $r=p/q$ give

$$
D(P\Vert Q)=\int_1^\infty
\left[\frac{E_\gamma(P\Vert Q)}{\gamma}
             +\frac{E_\gamma(Q\Vert P)}{\gamma^2}\right]\,d\gamma.
\tag{NJE.13}
$$

Indeed the first term contributes
$q(r\log r-r+1)$ at $r>1$, and the second contributes the
same nonnegative expression at $r<1$.
At $p>0,q=0$ both sides are infinite; at $p=0,q>0$ the
second integral is $q$. Integrating cancels the terms $q-p$
because $P,Q$ have equal total mass. Monotone integration
also proves the identity for infinite entropy.
Apply the first inequality to both orientations inside (NJE.13)
to prove the second inequality of (NJE.12).
:::

:::{prf:lemma} Entropy comparison under an actual bounded tilt
:label: lem-nje-entropy-tilt

For $m\le w\le1$ and $T_wP=wP/P(w)$,

$$
mD(T_wP\Vert T_wQ)\le D(P\Vert Q)
\le m^{-1}D(T_wP\Vert T_wQ).
\tag{NJE.14}
$$

These are full probability-law inequalities, with extended-value
interpretation.
:::

:::{prf:proof}
Attach a Bernoulli mark of conditional success probability $w(x)$
to both input laws. Its conditional kernel is identical, so the
relative entropy of the resulting joint laws is exactly $D(P\Vert Q)$.
Decompose it by that two-valued mark. The success term is
$P(w)D(T_wP\Vert T_wQ)$; every other conditional entropy and
the entropy of the mark itself is nonnegative.
Thus the first inequality follows from $P(w)\ge m$.
The common reverse success weight is $m/w\in[m,1]$;
tilting $T_wP,T_wQ$ by it gives $P,Q$.
Apply the first inequality again to obtain the second one.
The entropy chain identity follows directly by substituting joint
densities, with truncation or monotone integration for infinite values.
:::

:::{prf:theorem} Native full-law block entropy and survivor entropy
:label: thm-nje-native-entropy

For $N\ge\widehat N_*$, any two probability laws $\eta,\xi$ on
the actual nonextinct marked physical $(x,v,a)$ state space and any
native update $n$,

$$
D(\eta(P_N^e)^n\Vert\xi(P_N^e)^n)
\le\Delta_N^{\lfloor n/\widehat T_N\rfloor}D(\eta\Vert\xi).
\tag{NJE.15}
$$

In particular, with its actual invariant law $\pi_N$,

$$
D(\eta(P_N^e)^n\Vert\pi_N)
\le\Delta_N^{\lfloor n/\widehat T_N\rfloor}
                                      D(\eta\Vert\pi_N).
\tag{NJE.16}
$$

For the actual native survivor law
$\eta_n=\eta Q_N^n/(\eta Q_N^n1)$,

$$
D(\eta_n\Vert\nu_N)
\le\widehat m_N^{-2}
 \Delta_N^{\lfloor n/\widehat T_N\rfloor}D(\eta\Vert\nu_N)
\le\frac{64}{49}
 \Delta_N^{\lfloor n/\widehat T_N\rfloor}D(\eta\Vert\nu_N).
\tag{NJE.17}
$$

For infinite initial entropy these inequalities retain their extended
meaning; for finite initial entropy they are quantitative decay
estimates. No continuous-gradient inequality is used to erase statuses.
:::

:::{prf:proof}
Apply (NJE.12) to the actual complete block
$(P_N^e)^{\widehat T_N}$ using (NJE.10).
Iterate over its complete blocks and apply ordinary entropy data
processing, the case $\delta=1$, to the remaining native updates.
Its invariance yields (NJE.16).

The original exact Doob conjugacy gives

$$
(T_{e_N}\eta)(P_N^e)^n=T_{e_N}\eta_n,
\qquad T_{e_N}\nu_N=\pi_N.
\tag{NJE.18}
$$

For example the unnormalized first measure is
$e_N(T)\eta Q_N^n(dT)/(\alpha_N^n\eta(e_N))$,
and its normalization is exactly the displayed terminal tilt.
Apply (NJE.14) first to the initial tilt, then to the terminal
inverse tilt, with $w=e_N$ and $m=\widehat m_N$.
Together with (NJE.16) these give (NJE.17).
The uniform numerical factor follows from $\widehat m_N\ge7/8$.
:::

:::{prf:corollary} Direct entropy comparison to the native QSD reference
:label: cor-nje-qsd-entropy-reference

Put $\lambda_N=-\log(1-\widehat b_N)$.
For every law of finite entropy relative to either $\pi_N$ or $\nu_N$,

$$
|D(\eta\Vert\pi_N)-D(\eta\Vert\nu_N)|\le\lambda_N.
\tag{NJE.19}
$$

In particular

$$
D(\eta(P_N^e)^n\Vert\nu_N)
\le\Delta_N^{\lfloor n/\widehat T_N\rfloor}
                         [D(\eta\Vert\nu_N)+\lambda_N]+\lambda_N .
\tag{NJE.20}
$$

:::

:::{prf:proof}
The actual density $d\pi_N/d\nu_N=e_N/\nu_N(e_N)$ lies between
$1-\widehat b_N$ and $(1-\widehat b_N)^{-1}$.
Subtract the two entropy expressions to get the integral of its
logarithm against $\eta$, bounded in absolute value by $\lambda_N$.
Both reference laws have identical null sets, and their bounded
log density ratio makes finiteness equivalent.
Combine (NJE.19) with (NJE.16) to prove (NJE.20).
:::

(sec-nje-rate)=
## 5. Asymptotic entropy rate and the native bounded-observable spectrum

:::{prf:theorem} Population-uniform asymptotic rate in the native phase
:label: thm-nje-asymptotic-rate

Define the explicit per-update block rate

$$
\kappa_N^{\rm ent}=-\frac{\log\Delta_N}{\widehat T_N}>0.
\tag{NJE.21}
$$

Then $\kappa_N^{\rm ent}\to\ell=-\log\bar q>0$ as $N\to\infty$.
The asymptotic entropy-root decay in (NJE.16)--(NJE.17)
is at most $e^{-\kappa_N^{\rm ent}}$.
On the Banach space of bounded measurable functions modulo constants,
with quotient norm $\operatorname{osc}f=\sup_{x,y}|f(x)-f(y)|$,
the actual native Doob operator satisfies

$$
r_{\rm spec}(P_N^e\ {\rm modulo\ constants})
\le\Delta_N^{1/\widehat T_N}
                       =e^{-\kappa_N^{\rm ent}} .
\tag{NJE.22}
$$

These assertions also apply to its bounded $\pi_N$-centered
real or complex observables in the equivalent oscillation norm.

An explicit uniform positive asymptotic-rate regime is
$N\ge N_{\rm rate}$, where

$$
\begin{gathered}
C_{\rm rate}=2\log_+(8\widehat C_\theta)
                     +\widehat C_0+4\ell,\\
N_{\rm rate}=\left\lceil
\max\{\widehat N_*,(16/u)^2,2C_{\rm rate}/u\}\right\rceil .
\end{gathered}
\tag{NJE.23}
$$

For it,
$\kappa_N^{\rm ent}\ge\ell/2$ and
$r_{\rm spec}\le\sqrt{\bar q}<1$.
In the existing physical-time calibration its asymptotic entropy
rate is at least $\ell/(2t_*h)$.
:::

:::{prf:proof}
From the exact formulas,
$\log\Delta_N=-uN+O(\log N)$ and
$\widehat T_N=uN/\ell+O(\log N)$.
Thus (NJE.21) converges to $\ell$.
Taking $n$th roots in the complete-block entropy estimates gives
the claimed asymptotic bounds whenever the initial entropy is finite.
If the entropy becomes zero it remains zero, which also satisfies
the same assertion.

The row common-mass coupling bounds
$\operatorname{osc}Kf\le\delta\operatorname{osc}f$ for every
real or complex bounded test when the row Dobrushin coefficient
is at most $\delta$. Therefore the actual block has quotient
operator norm at most $\Delta_N$.
Its powers have norms at most $\Delta_N^j$;
the spectral-radius formula, or the convergent geometric-resolvent
series for every radius above $\Delta_N^{1/\widehat T_N}$,
proves (NJE.22). This is the spectrum of the stated native bounded
observable operator, not an assigned Hamiltonian.

For the explicit rate note
$\Delta_N\le8\widehat C_\theta Ne^{-uN}$ and
$\widehat T_N\le(uN+6\log N+\widehat C_0)/\ell+4$.
The condition

$$
uN\ge8\log N+2\log_+(8\widehat C_\theta)
                            +\widehat C_0+4\ell
$$

makes $-\log\Delta_N/\widehat T_N\ge\ell/2$.
The elementary $\log N\le\sqrt N$ for $N\ge1$ shows
that (NJE.23) implies that condition:
each half of $uN$ pays, respectively, $8\sqrt N$ and
$C_{\rm rate}$. Dividing the native rate by its actual step
$t_*h$ gives the calibrated statement.
:::

:::{prf:corollary} Primitive asymptotic bound for the entire permitted population family
:label: cor-nje-all-population-rate

For the finitely many permitted $N<N_{\rm rate}$ use
$a_{F,N}=\min\{1/2,\underline\delta_F(N)\}>0$, with the
primitive Doob common-part lower bound (CGD.15), and set

$$
\kappa_{\rm all}=
\min\left\{\ell/2,\,
\min_{\substack{\text{permitted }N<N_{\rm rate}}}
                      [-\log(1-a_{F,N})]\right\}>0.
\tag{NJE.24}
$$

An empty finite minimum is omitted.
The complete native bounded-observable quotient spectral radii
are at most $e^{-\kappa_{\rm all}}<1$, and the full-law
Doob/survivor entropy asymptotic rates are at least
$\kappa_{\rm all}$ per original update throughout that family.
This is a finite minimum of primitive formulas; no spectral
quantity of an unknown law is its input.
:::

:::{prf:proof}
The finite-$N$ theorem retained in
{prf:ref}`cor-nue-all-populations` proves a Doob common part
of mass at least $\underline\delta_F(N)$ for every indicated
population. Therefore its one-update Dobrushin coefficient is
at most $1-a_{F,N}$.
Apply (NJE.12), the actual tilt comparison, and the bounded-observable
operator argument to those finitely many populations.
For every other admitted population use (NJE.23).
The finite minimum combines exactly these same native kernels.
:::

:::{prf:remark} Exact scope of the entropy and spectral bounds
:label: rem-nje-scope

The linear horizon and entropy comparisons use all original
marked-state laws, rather than only fixed-label marginals.
Here the mixing state is the actual physical $(x,v,a)$ array.
An original subsequent finite-window record instrument inherits
entropy data processing when attached by its same conditional law
to both compared current states. The growing archive and absolute
recording clock retain their original record roles; they are not
stationary entropy-mixing coordinates.
They apply to the positive active witness of the phase chapter and
the primitive open parameter regimes passing the stated tests.
All parameters, masks, arithmetic branches and calibration values
remain those of the complete register.
For the published positive witness (PC.37), with its exact cap/weight
and certified upper coefficient $r=1/2$, the primitive formulas give
the orientation diagnostics

$$
\begin{gathered}
\widehat K_1\simeq510.4079159,\qquad
\widehat C_L\simeq0.1613921745,\qquad
\widehat N_*=176,\qquad \widehat T_{176}=172,\\
\widehat b_{176}\simeq5.04197\cdot10^{-4},\qquad
\Delta_{176}\simeq0.001008903,\qquad
\kappa_{176}^{\rm ent}\simeq0.040109835,\\
N_{\rm rate}=26451,\qquad
\lim_N\kappa_N^{\rm ent}=-\log(3/4)\simeq0.2876820725.
\end{gathered}
\tag{NJE.25}
$$

The exact formulas, not the rounded diagnostics, define the bounds.

Writing the block bound as a per-update exponential gives the factor
$\Delta_N^{-1}$:
$\Delta_N^{\lfloor n/\widehat T_N\rfloor}
\le\Delta_N^{-1}e^{-\kappa_N^{\rm ent}n}$.
That factor can grow exponentially with $N$.
The theorem does not assert a population-uniform initial prefactor,
a one-update modified log-Sobolev or Dirichlet coercivity inequality,
or an $L^2$ self-adjoint physical Hamiltonian gap.
The bounded-observable quotient spectrum and the calibrated native
entropy rate have the precisely stated meanings.
Reflection positivity, local non-Abelian sector identification and
the continuum Yang--Mills physical gap require their own actual
correspondence proofs.
:::
