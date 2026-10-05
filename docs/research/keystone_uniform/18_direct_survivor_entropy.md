# Complete survivor entropy transfer with no eigenfunction comparison

:::{prf:definition} The physical survivor and rejected-input laws
:label: def-kue-survivor-inputs

Let $Q_N$ be the unchanged full killed kernel, $\nu_NQ_N=\alpha_N\nu_N$,
$k_N=Q_N1$ and $\kappa_N=1-k_N$. This $k_N$ is the one-update survival
probability; it is not the positive right eigenfunction.
For $P\ll\nu_N$ with $H=D(P\Vert\nu_N)<\infty$, put
$a=P(k_N)$ and define
$$
P^{\rm s}=k_NP/a,\qquad \nu_N^{\rm s}=k_N\nu_N/\alpha_N,\qquad
P^+=PQ_N/a.
$$
If $a<1$, let $P^{\rm d}=\kappa_NP/(1-a)$.
If $\alpha_N<1$, let
$\nu_N^{\rm d}=\kappa_N\nu_N/(1-\alpha_N)$.
An entropy term multiplied by a zero rejected mass is defined to be zero.
The rejected input is an observation used in the proof, not a restart or
new transition of the gas. The backward conditional information
$\mathcal B_N(P,\nu_N)$ is exactly that in
{prf:ref}`thm-kl-full-step-survivor-chain-rule`.
:::

:::{prf:theorem} Exact rejection balance and population-uniform physical entropy cost
:label: thm-kue-direct-entropy-transfer

For the actual complete update,
$$
\boxed{\quad
H=D_{\rm Ber}(a\Vert\alpha_N)
+a\left[D(P^+\Vert\nu_N)+\mathcal B_N(P,\nu_N)\right]
+(1-a)D(P^{\rm d}\Vert\nu_N^{\rm d}).
\quad}                                                    \tag{KUE.1}
$$
The identity is an equality of nonnegative extended integrals before
any subtraction. The finite $H$ hypothesis makes the terms on its
right finite when their weights are positive.

Suppose the proved actual one-update extinction envelope is
$\sup_S\kappa_N(S)\le\delta_N<1$. Then
$$
\boxed{\quad
D(P^+\Vert\nu_N)
\le\frac{H}{1-\delta_N}-\mathcal B_N(P,\nu_N).
\quad}                                                    \tag{KUE.2}
$$
In particular, either the original independent lower-trial certificate
or the preparation-exact energy certificate supplies
$\delta_N\le e^{-Na_*}$, for its explicitly computed $a_*>0$.
For every $N\ge N_0\ge1$ the physical survival cost is therefore
$$
C_{{\rm s},N}=\frac1{1-e^{-Na_*}}
\le C_{{\rm s},*}=\frac1{1-e^{-N_0a_*}}<\infty.             \tag{KUE.3}
$$
This is an $N$-uniform full marked-law entropy transfer.
It uses the actual QSD eigenmeasure and the original execution
law, including its shared Haar dependence, not the Doob invariant law.
For the certified Rastrigin coefficients (KURC.16), the stronger
actual envelope $\delta_N\le(1-a_*)^N$ gives respectively
$C_{{\rm s},N}\le1/a_*=500000$ and $10^{11}/7$ for all $N\ge1$.
These constants bound the survival transfer, not an entropy decay rate.

For $T$ updates, set $a_T=PQ_N^T1$ and replace $Q_N$ by $Q_N^T$
in (KUE.1), with the actual $T$-step backward conditional information.
Since $a_T\ge(1-\delta_N)^T$, the corresponding cost is
$$
C_{{\rm s},N,T}\le(1-e^{-Na_*})^{-T},\qquad
\log C_{{\rm s},N,T}
\le\frac{T e^{-Na_*}}{1-e^{-Na_*}}.                         \tag{KUE.4}
$$
For $T_N\le e^{cN}$ with $0<c<a_*$ this cost converges to one,
and
$$
\log C_{{\rm s},N,T_N}
\le C_{{\rm s},*}e^{-(a_*-c)N}.                            \tag{KUE.5}
$$
No right-eigenfunction oscillation or whole-array mixing coefficient
appears in these transfers.
:::

:::{prf:proof}
Attach to the input $S$ the Bernoulli indicator of actual survival,
whose parameter is $k_N(S)$ under both $P$ and $\nu_N$.
The two joint input-and-indicator laws have likelihood ratio
$dP/d\nu_N$; consequently their relative entropy is exactly $H$.
Disintegrating first by the indicator gives
$$
H=D_{\rm Ber}(a\Vert\alpha_N)
+aD(P^{\rm s}\Vert\nu_N^{\rm s})
+(1-a)D(P^{\rm d}\Vert\nu_N^{\rm d}).
$$
The accepted forward kernel is $Q_N(S,\cdot)/k_N(S)$.
Apply the already proved full-step backward chain rule to the
survivor branch. Its reference output is
$\nu_NQ_N/\alpha_N=\nu_N$, so
$$
D(P^{\rm s}\Vert\nu_N^{\rm s})
=D(P^+\Vert\nu_N)+\mathcal B_N(P,\nu_N).
$$
This proves (KUE.1), with every random input in the actual kernel
integrated in its original order. Drop only the two nonnegative
Bernoulli and rejected-input terms and use $a\ge1-\delta_N$ to
obtain (KUE.2). The actual lower-trial or energy proof supplies
the displayed exponential envelope, giving (KUE.3).
The $T$-step eigenmeasure identity is
$\nu_NQ_N^T=\alpha_N^T\nu_N$. Repeating the same chain rule
and multiplying the conditional survival lower bounds proves
(KUE.4). Finally $-\log(1-x)\le x/(1-x)$ and
$T_Ne^{-Na_*}\le e^{-(a_*-c)N}$ give (KUE.5).
All statements preserve the complete marked state; no independence
of prepared or completed swarm rows is postulated. $\square$
:::

:::{prf:proposition} Exact phase entropy and the retained full-kernel flow
:label: prop-kue-phase-entropy-flow

Let $(G_j)$ be a finite or countable measurable partition of nonextinct full states,
including every spatial exterior or other phase needed for coverage.
Discard only zero $\nu_N$-mass cells; $P\ll\nu_N$ gives them zero
$P$ mass. Write $p_j=P(G_j)$, $w_j=\nu_N(G_j)$.
The exact input entropy is
$$
D(P\Vert\nu_N)
=D(p\Vert w)+\sum_jp_jD(P(\cdot\mid G_j)
                                  \Vert\nu_N(\cdot\mid G_j)).   \tag{KUE.6}
$$
For the surviving input/output joint laws define the actual flux tables
$$
F^P_{ij}=\frac1a\int_{G_i}P(dS)Q_N(S,G_j),\qquad
F^\nu_{ij}=\frac1{\alpha_N}
                       \int_{G_i}\nu_N(dS)Q_N(S,G_j).           \tag{KUE.7}
$$
Their output marginals are $p_j^+=P^+(G_j)$ and $w_j$.
Their input marginals are the actual survivor-tilted phase weights,
rather than $p_i$ and $w_i$ when survival varies by phase.

Let $\mathsf J^P_{ij}$ and $\mathsf J^\nu_{ij}$ be the conditional
joint input/output laws in the corresponding positive flow cells.
Then
$$
D(P^{\rm s}\Vert\nu_N^{\rm s})
=D(F^P\Vert F^\nu)+\sum_{ij}F^P_{ij}
                         D(\mathsf J^P_{ij}\Vert\mathsf J^\nu_{ij}), \tag{KUE.8}
$$
and exactly
$$
D(P^+\Vert\nu_N)
=D(p^+\Vert w)+\sum_jp_j^+
             D(P^+(\cdot\mid G_j)\Vert\nu_N(\cdot\mid G_j)).      \tag{KUE.9}
$$
Equations (KUE.1), (KUE.6)--(KUE.9) retain macro phase weights,
within-phase law errors, and all transition-cell information.
They do not set inter-phase entropy or orbit motion to zero.
The tables use the original sampled measurements, gates, donor
choices, jitter, component rotations, both viscous kicks, noises,
cap and terminal marking, exactly as in (KLQ.4).
No lumpability or closure of the phase labels is assumed.
:::

:::{prf:proof}
Disintegrate the input measures by the phase label and apply the
relative-entropy chain rule to obtain (KUE.6).
The two survivor joint measures have input entropy
$D(P^{\rm s}\Vert\nu_N^{\rm s})$ because their accepted forward
conditional kernel is the same. Their phase-pair pushforwards
are precisely (KUE.7). Disintegrating these full joint measures
by their paired labels proves (KUE.8). The QSD equation gives
the stated reference output marginal. Disintegrating only
the output law proves (KUE.9). The disintegrations observe the
original transition and leave all its correlations intact. $\square$
:::

:::{prf:remark} Exact dependency of the convergence programme
:label: rem-kue-programme-dependency

The right-eigenfunction ratio
$R_N=\sup h_N/\inf h_N$ is required by the bounded Doob
reweighting estimate (15.40)--(15.41) if that estimate is to retain
its population-uniform prefactor. It is not a premise of the physical
finite-horizon mean-field theorem
{prf:ref}`cor-cg-mf-full-update`,
the direct survivor propagation
{prf:ref}`thm-chaos-conditioned-propagation`, or the actual
stationary population-distribution invariance
{prf:ref}`thm-native-stationary-closure-population-invariance`.
Their force and moment hypotheses must still be respected.

(KUE.1)--(KUE.5) close the missing uniform survival change-of-measure
cost for the direct physical entropy balance. They do not assert a
positive entropy decay rate merely from rare extinction.
That rate requires the same computed signed backward information
and phase-resolved production as the existing direct proof.
For example, a proved bound
$\mathcal B_N(P,\nu_N)\ge b_*D(P\Vert\nu_N)$ on the actual reached
phase laws gives the explicit factor
$C_{{\rm s},N}-b_*$ in (KUE.2). This example is an implication,
not a derivation of its $b_*$ hypothesis from the Keystone positional
coordinate alone. Likewise stationary population-map invariance
does not identify the invariant distribution with fixed points;
the existing programme retains cycles, phase weights and orbits.
:::
