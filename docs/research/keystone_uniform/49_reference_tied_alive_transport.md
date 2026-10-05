# Exact finite alive-law transport from tied-fitness harmonic swarms

(sec-tat-register)=
## 1. The actual finite one-update comparison

:::{prf:definition} Collapsed alive reference inputs
:label: def-tat-register

Use the unchanged harmonic restriction $F=-x$, $R=-|x|^2/2$,
$d=3$, $h=.04$, $\gamma=b_O=\rho=1$, $\nu=.3$, $V=2$,
$L=2$ and $\sigma_x=\sigma_J=.1$. Both dense count and dense row
normalization are allowed with their original singleton convention.
Every fitness floor, standardizer floor, donor law, active gate,
component collision, recipient jitter and terminal mark is retained.
No force-center reset or bounded reward substitution is made.
Use the current-frame restriction with historical, curl, elite and
geometry-feedback branches disabled, as in
{prf:ref}`def-dmc-register`. The positive fitness exponents themselves
need not satisfy that definition's weak-exponent test: this theorem's
exact ties make acceptance zero directly.

Fix $N\ge1$. For $m\in[-.5,.5]^3$, start from the actual deterministic
array $S(m)$ with $x_i=m$, $v_i=0$ and $a_i=1$ for every slot.
Write $S^+(m)$ for one complete proposed update, $E_m$ for its own
nonextinction event, and
$$
\widehat\alpha^N(m)
=\frac1{M(m)}\sum_{i:a_i^+=1}\delta_{(x_i^+,v_i^+)},
\qquad M(m)=\sum_i a_i^+,
$$
on $E_m$. Let
$\mathcal A_N(m)=\operatorname{Law}(\widehat\alpha^N(m)\mid E_m)$.
The outer Wasserstein distance $\mathbb W_2$ uses the inner physical
Euclidean phase $W_2$ as its ground metric.
Its sampled-law counterpart draws the swarm under its own $E_m$,
then draws uniformly from its own actual alive slots.

Put $t=.02$, $c=e^{-.04}$, $b=t(1+c)$,
$a_x=1-tb$, $r_H=t(c+a_x)$,
$q^2=(1-c^2)/2$ and $s^2=.0004$.
This is a one-update law comparison on a stated input class.
That class is not asserted invariant under the update.
:::

(sec-tat-marginals)=
## 2. Every actual provider and Gaussian outcome

:::{prf:lemma} Tied fitness and translation of the actual full array
:label: lem-tat-own-array-translation

Each input $S(m)$ has exactly tied measured fitness and zero active
acceptance. Its actual preparation therefore has $X_i=m$, $P_i=0$.
For two centers $m,\widetilde m$, share the original independent OU
and final Gaussians, separately preserving each transition marginal.
With $\delta=\widetilde m-m$, the actual arrays satisfy
$$
\Delta y_i=\Delta x_i^+=a_x\delta,\qquad
\Delta w_i=-ct\delta,\qquad
\Delta z_i=-r_H\delta,
$$
$$
|\Delta x_i^+|^2+|\Delta v_i^+|^2
\le(a_x^2+r_H^2)|\delta|^2
\le\left(1-\frac1{40000}\right)|\delta|^2
\quad\text{for every row and every Gaussian outcome}.
\tag{TAT.1}
$$
The actual second-stage providers are translated joint array laws;
they are not frozen common providers.
:::

:::{prf:proof}
All alive rewards equal $-|m|^2/2$ and all measured diversity values
are identical. Their own measured normalizers therefore give identical
fitness within the array. The original clipped difference gate is
zero, including exact ties. No dead vertex requires revival.
All original velocities vanish, so every executed component readout
is zero. No row is copied, and the configured recipient jitter is
therefore inactive as prescribed by the algorithm.

The first dense viscous difference is zero, so $U_i=0$.
The original kinetic stages are
$w_i=-ctm+q\xi_i$, $y_i=a_xm+tq\xi_i$.
In the paired arrays the differences between second-stage positions
are identical: $\widetilde y_i-\widetilde y_j=y_i-y_j$.
The original count weights and each original self-excluded row degree
are consequently identical. The velocity differences between rows
are also identical, so each second viscous contribution is unchanged.
This proves the three translations above before the cap.
The native cap is nonexpansive, proving the first inequality in
(TAT.1). Its position Gaussian cancels under the shared coupling.

For an exact rational certificate, $0.96<c<0.9608$ gives
$a_x<.999216$ and $r_H<.03920032$.
Direct squaring verifies
$.999216^2+.03920032^2<1-1/40000$.
No exceptional jitter, noise, component or provider outcome has
been removed.
:::

:::{prf:lemma} Exact survival and an explicit uncut tail budget
:label: lem-tat-survival-budget

For every center in the stated box, the terminal positions are
independent Gaussians with mean $a_xm$ and covariance
$(t^2q^2+s^2)I_3$. For $\epsilon=2^{-100}$,
$$
\Pr\{a_i^+=0\}\le\epsilon,\qquad
\Pr(E_m^c)\le\epsilon^N,\qquad
\mathbb E[(N-M(m))/N]\le\epsilon.
\tag{TAT.2}
$$
:::

:::{prf:proof}
The second kick and cap change velocity only, so terminal positions
are exactly $a_xm+tq\xi_i+s\chi_i$, with the actual independent
original innovations. Their variance is less than $.0005$.
Every coordinate mean lies in $[-.5,.5]$, leaving distance at least
$1.5$ to either box face. The Gaussian exponential tail bound and
a union over six faces give
$$
\Pr\{a_i^+=0\}\le6e^{-1.5^2/(2(.0005))}
=6e^{-2250}<2^{-100}.
$$
The last inequality follows already from $e>2$.
This is a bound on the complete Gaussian law; the tails still run.
Independence of terminal positions gives the extinction inequality.
Linearity gives the expected dead fraction.
:::

(sec-tat-alive-law)=
## 3. Optimal alive transport with each own survival normalizer

:::{prf:theorem} Exact finite tied-fitness alive-law contraction, uniformly in population
:label: thm-tat-finite-alive-transport

Let $m,\widetilde m\in[-.5,.5]^3$ with
$|\widetilde m-m|\ge1/4$. Then for every $N\ge1$,
$$
\mathbb W_2(\mathcal A_N(m),\mathcal A_N(\widetilde m))
\le\left(1-\frac1{160000}\right)|\widetilde m-m|.
\tag{TAT.3}
$$
The same estimate holds for the two swarm-first uniformly
alive-sampled probability laws, each conditioned on its own
nonextinction. It has no particle floor and its coefficient is
independent of $N$. At input the physical empirical-measure distance
is exactly $|\widetilde m-m|$.
This is one complete actual alive-law update, not a mixing theorem
for subsequent unrestricted swarms.
:::

:::{prf:proof}
On extinction define an auxiliary alive readout to be the point mass
at $(0,0)$; this only defines a bounded comparison variable.
The algorithm is still killed and receives no restart.
All alive positions lie in $[-2,2]^3$ and all stored speeds are
at most two. Their Euclidean phase squared diameter is at most
$48+16=64<67=:D^2$.

Couple the two raw auxiliary readouts by the exact common-innovation
array coupling above. Let $A,\widetilde A$ be their alive index sets.
On $M,\widetilde M\ge N/2$, pair each common-alive index with weight
$1/\max(M,\widetilde M)$ and complete the residual marginals
arbitrarily. Its common-alive cost is at most
$k^2|\delta|^2$, where $k^2=1-1/40000$.
Its residual mass is
$$
1-\frac{|A\cap\widetilde A|}{\max(M,\widetilde M)}
\le\frac{|A\mathbin\triangle\widetilde A|}
          {\max(M,\widetilde M)}
\le\frac{2|A\mathbin\triangle\widetilde A|}{N}.
$$
Thus the residual cost is at most
$2D^2|A\mathbin\triangle\widetilde A|/N$.
By (TAT.2), the expected symmetric-difference fraction is at most
$2\epsilon$. Markov's inequality gives
$\Pr\{M<N/2\}\le2\epsilon$, and likewise for the other swarm.
On that exceptional event use the squared diameter $D^2$.
The raw outer law distance is therefore at most
$$
\mathbb W_2(\mathcal B_N(m),\mathcal B_N(\widetilde m))
\le\sqrt{k^2|\delta|^2+8D^2\epsilon},
\tag{TAT.4}
$$
where $\mathcal B_N$ denotes the raw auxiliary law.

Each raw marginal is the mixture of its own survivor law with
weight $\Pr(E_m)$ and its own extinct auxiliary law. Maximal
coupling of this mixture and the survivor law has mismatch probability
at most $\Pr(E_m^c)\le\epsilon^N$. Since the squared diameter is $D^2$,
$$
\mathbb W_2(\mathcal B_N(m),\mathcal A_N(m))
\le D\sqrt{\epsilon^N}\le D\sqrt\epsilon.
$$
The metric triangle and (TAT.4) now give
$$
\mathbb W_2(\mathcal A_N(m),\mathcal A_N(\widetilde m))
\le k|\delta|+(\sqrt{8D^2}+2D)\sqrt\epsilon
< k|\delta|+40\,2^{-50}.
$$
This performs each own actual survival normalization; their events
are not identified. Since $\sqrt{1-u}\le1-u/2$,
$k\le1-1/80000$.
The exact integer inequality $40\cdot640000<2^{50}$ and
$|\delta|\ge1/4$ give
$40\,2^{-50}<|\delta|/160000$.
This proves (TAT.3).

For the sampled-law assertion, draw a coupled pair of surviving
empirical measures and then an optimal inner phase transport pair.
Averaging these plans preserves each own uniformly alive-sampled
law, and its expected squared cost is the outer cost.
Taking the infimum proves that the sampled-law distance is no larger.
This is optimal transport between actual alive laws, with no full-array
status cost or claim about a prescribed source pairing being optimal.
:::

:::{prf:remark} Exact scope of the tied-fitness result
:label: rem-tat-scope

The configured fitness exponents remain positive, with no variance
lower bound. Actual within-swarm ties make the original active gate
zero in this stated input class. Both noisy second providers are
actual and keep their joint position/velocity correlations. Death
is enabled and each output uses its own nonextinction event and own
alive normalization. All Gaussian outcomes are retained.

The theorem includes different initial centers without an equilibrium
orbit assumption. It certifies a single physical alive-law update on
the explicit separated collapsed-input class, at the unchanged
harmonic viscosity and box. The class is not invariant: its next
state is noisy and generally has nonconstant velocities. Therefore
the theorem cannot be iterated as a default convergence rate.
It proves neither exact finite-swarm QSD mixing nor Rastrigin
contraction. Their missing law estimates remain separate.
:::
