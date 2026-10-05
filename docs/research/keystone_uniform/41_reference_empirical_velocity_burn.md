# Actual default empirical-provider velocity burn

(sec-dev-register)=
## 1. The finite current-survivor carrier

:::{prf:definition} Reference empirical moment interface
:label: def-dev-register

Retain the unchanged harmonic count record of
{prf:ref}`def-dmc-register`, including its nonempty positive exponent
interval $0<\theta\le\theta_f$. In particular $h=.04$, $\nu=.3$,
$L=2$, $V=2$, $d=3$, $\sigma_J=\sigma_x=.1$ and the reward remains
$R=-|x|^2/2$. Start the actual killed chain from any consistent,
nonempty-alive capped law. Let
$\eta_n=\operatorname{Law}(S_n\mid\tau_N>n)$ and put
$$
H_n=\frac1N\sum_i|v_{n,i}|^2,\qquad
P_N^v(n)=\Pr_{\eta_n}\{H_n>.56^2\}.
$$
For a proposed update from $S_n$, let $(X_i,P_i)$ be its actual
prepared array, $U=(I-t\nu L_X)P$, and
$$
w_i=c(U_i-tX_i)+q\xi_i,\qquad
W_N=\frac1N\sum_i|w_i|^2.
\tag{DEV.1}
$$
$W_N$ is the velocity moment of the actual noisy empirical joint
second provider. It is not a population replacement. Use
$A_N,\alpha,D,e_N,r_N,c_{\rm ret}$ of
{prf:ref}`cor-dmc-finite-horizon`, and define
$$
V_{N,6}=D^{1/(1-\alpha)}A_N^{\alpha^5},\qquad
T_{N,6}=(1-e_N)^{-6},\qquad
g_v=.56^2-.55^2=.0111,
$$
$$
B_N^v=\min\left\{1,\frac4{g_v}T_{N,6}
               [V_{N,6}+(5+c_{\rm ret})r_N]\right\}.
\tag{DEV.2}
$$
All constants are independent of $N$ and elapsed time. Arbitrarily
large retained dead coordinates are allowed. These are full-slot
empirical moments under current survival, rather than moments divided
by the actual number of alive slots.
:::

(sec-dev-concentration)=
## 2. Uniform-time empirical velocity burn

:::{prf:lemma} Stored energy as a bounded marked transport test
:label: lem-dev-energy-test

On the consistent capped input class,
$f(x,v,a)=|v|^2$ is $4$-Lipschitz for the original marked ground cost.
Hence
$$
|\mu f-\mu'f|\le4\mathsf d_{\rm m}(\mu,\mu').
\tag{DEV.3}
$$
:::

:::{prf:proof}
If the cost is below one, the marks agree and
$\big||v|^2-|v'|^2\big|\le(|v|+|v'|)|v-v'|\le4|z-z'|$.
If the cost is one, $f\in[0,4]$ bounds its difference by four.
Integrate a transport plan and take the infimum. This test needs
no retained-position moment.
:::

:::{prf:theorem} Actual empirical velocity burn with a uniform-time vanishing floor
:label: thm-dev-uniform-empirical-burn

For every $n\ge7$,
$$
\Pr\{H_n>.56^2\mid\tau_N>n\}\le B_N^v,\qquad B_N^v\longrightarrow0.
\tag{DEV.4}
$$
No population contraction, invariant phase or QSD limit is assumed.
:::

:::{prf:proof}
At $n-6\ge1$, the actual current-survivor entering law has alive
fraction at least $m_f$ except with probability
$c_{\rm ret}r_N$, by
{prf:ref}`cor-dsa-current-conditional-control`. Apply the recent
six-update comparison of {prf:ref}`cor-dmc-finite-horizon`.
On its good event the population starts at the actual entering
empirical law. This random starting law retains its future-survival
reweighting. Its comparison error satisfies
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S_n),\mu_6)\mid\tau_N>n]
\le\min\{1,T_{N,6}[V_{N,6}+(5+c_{\rm ret})r_N]\}.
\tag{DEV.5}
$$
On the initial bad event choose any consistent capped positive-alive
population law as comparator and charge its bounded distance by one.
That is only a proof comparator; the actual chain receives no restart.

For every such population input,
{prf:ref}`thm-rvb-population-burn` gives $\mu_6f\le.55^2$.
Its actual rooted preparation and finite-component population premise
is discharged by the weak positive interval in
{prf:ref}`def-dmc-register`. Thus $H_n>.56^2$ forces the marked
distance in (DEV.5) above $g_v/4$, by (DEV.3).
Markov's inequality proves (DEV.4). The denominator is the recent
six-update factor $T_{N,6}$ with the starting tilt retained, rather
than a probability charged over the entire elapsed history.
For fixed six, $V_{N,6},r_N\to0$ and $T_{N,6}\to1$.
:::

(sec-dev-proposed-stage)=
## 3. The actual unbounded noisy provider

:::{prf:lemma} Full Gaussian concentration of the actual proposed joint moment
:label: lem-dev-actual-ou-concentration

From any actual consistent input array with $H\le.56^2$,
$$
\Pr\{W_N>.70^2\mid S\}\le\min\{1,2690/N\}.
\tag{DEV.6}
$$
No Gaussian innovation or high-degree component is discarded.
:::

:::{prf:proof}
Freeze the actual discrete source/component/Haar plan.
The exact original-slot collision identity in
{prf:ref}`lem-rvb-preparation-energy` gives
$N^{-1}\sum_i|P_i|^2\le H\le.56^2$ for every plan.
Its prepared sources $S_i$ lie in $D$. Its independent fresh jitters
give $X_i=S_i+I_iJ_i$, and
$$
\mathbb E_J|X_i|^2=|S_i|^2+I_i d\sigma_J^2,\qquad
\operatorname{Var}_J(|X_i|^2)
=4I_i\sigma_J^2|S_i|^2+2I_i d\sigma_J^4.
$$
For $X_N=N^{-1}\sum_i|X_i|^2$, therefore
$\mathbb E_JX_N\le12.03$ and
$\operatorname{Var}_J(X_N)\le.4806/N$. Chebyshev gives
$$
\Pr\{X_N>12.25\mid\mathrm{plan}\}
\le\frac{.4806}{N(.22)^2}<\frac{10}N.
\tag{DEV.7}
$$
Now freeze the actual prepared array with $X_N\le12.25$.
Its first count operator contracts normalized array energy, so
$\|U\|_N\le\|P\|_N\le.56$. Thus
$$
\bar w=c(U-tX),\qquad
\|\bar w\|_N^2
\le c^2(.56+.02\sqrt{12.25})^2
<.9608^2(.63)^2<.367.
$$
Conditional on this complete array the original OU noises are
independent, with $q^2<.0392$. Their exact Gaussian moment formulas give
$$
\mathbb E_\xi W_N=\|\bar w\|_N^2+dq^2<.485,
$$
$$
\operatorname{Var}_\xi(W_N)
=\frac{4q^2\|\bar w\|_N^2+2dq^4}{N}<\frac{.067}N.
\tag{DEV.8}
$$
The actual conditional stage failure probability is consequently
at most $.067/[N(.005)^2]=2680/N$. Add (DEV.7) and average the
complete plan to prove (DEV.6). The first graph may depend on all
jitters; its contraction was used pathwise. No independent second
graph or bounded uncapped velocity was asserted.
:::

:::{prf:theorem} Actual empirical providers under their own current survival
:label: thm-dev-current-provider-budget

For every $n\ge7$, the proposed update from $\eta_n$ satisfies
$$
\Pr\{W_N>.70^2\mid\tau_N>n\}
\le\min\{1,B_N^v+2690/N\}.
\tag{DEV.9}
$$
Under that same update's own next current-survival law,
$$
\Pr\{H_n>.56^2\ \text{or}\ W_N>.70^2\mid\tau_N>n+1\}
\le B_N^{\rm prov}:=
\min\left\{1,\frac{B_N^v+2690/N}{1-e_N}\right\}\longrightarrow0.
\tag{DEV.10}
$$
On the complementary event its actual empirical first and second
count providers satisfy for every spatial query $x$,
$$
0\le a_{0,N}(x),a_{2,N}(x)\le1,\qquad
|M_{0,N}(x)|\le.56,\qquad |M_{2,N}(x)|\le.70.
\tag{DEV.11}
$$
The count normalization and zero self contribution remain unchanged.
:::

:::{prf:proof}
Split the current proposal according to $H_n\le.56^2$.
Theorem {prf:ref}`thm-dev-uniform-empirical-burn` bounds the bad
entering event by $B_N^v$ and (DEV.6) bounds stage failure on its
complement. This proves (DEV.9) and the raw union bound in (DEV.10).
Actual next survival has probability at least $1-e_N$ from every
nonextinct input. Restrict the nonnegative union indicator to that
survival and divide by its own probability. This retains the
reweighting of both the entering array and the previous innovations.

For each realized count provider, $K\le1$ and Cauchy--Schwarz give
$|N^{-1}\sum_i K(x-X_i)P_i|\le\|P\|_N$ and the same bound for
$(y_i,w_i)$. The collision identity gives (DEV.11).
At own empirical queries the self term cancels exactly as
$K(0)(v_i-v_i)=0$; this is the executed count operator.
All bounds tend to their stated limits uniformly in $n\ge7$.
:::

:::{prf:remark} A moment event does not replace its conditional Gaussian law
:label: rem-dev-provider-scope

This closes an actual finite empirical-provider interface.
Its unbounded OU velocities remain unbounded, with an explicit
vanishing exceptional probability for their empirical second moment.

The event in (DEV.10) depends on the actual Gaussian draws. It does
not make conditional root jitters or OU noises fresh Gaussians.
A consumer must use (DEV.11) pointwise on the event and charge its
exceptional outcomes with (DEV.10) and the required moment inequality.
It cannot factor an averaged Jacobian from correlated displacements.
Every Gaussian tail remains in the actual transition marginal.

These tests concern all slots. Alive normalization, the complete signed
preparation/B2/cap feedback, and full law convergence are separate.
No default population attraction or finite-swarm QSD mixing is inferred.
The harmonic force and declared weak positive exponent interval are
essential hypotheses.
:::
