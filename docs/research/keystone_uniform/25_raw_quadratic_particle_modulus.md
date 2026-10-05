# Uniform-time particle transfer with the raw quadratic reward

(sec-rqpt-register)=
## 1. Raw reward, actual finite swarm and population target

:::{prf:definition} Raw harmonic particle-transfer record
:label: def-rqpt-record

Use the complete raw-reward conservative population regime of
{prf:ref}`cor-rqf-active-population`, with its explicit endpoints
$0<\theta\le\theta_*^Q$, $0<\nu\le\nu_*^Q$. Both actual channels are
$$
F(x)=-x,\qquad R(x)=-|x|^2/2.
$$
Death and history terms are disabled; all slots are alive. The actual
finite swarm $S_n^N$, $N\ge2$, retains sampled global normalizers,
simultaneous copied positions, original frozen-slot component Haar
velocities, recipient jitters, both count-viscous kicks and their actual
joint noisy second provider, final position noise and the native smooth cap.
Let $\widehat\mu_n^N=L_N(S_n^N)$ and write $\mathcal F^Q$ for the
actual raw-reward population map. Its stationary population law is $\pi_Q$.

Use $\mathsf d$, the bounded transport with ground cost
$\min\{1,|z-z'|\}$, from {prf:ref}`def-vupt-register`. Retain the
kinetic and noise constants of that register, the raw-reward population
values $\beta,r_*,H_8$ and
$$
\lambda_8=(1+3r_8)/4<1,\qquad
M_8^+\le\lambda_8M_8+B_8,\qquad
B_8/(1-\lambda_8)=2H_8/3.
$$
The conditional averaged moment inequality holds for the actual finite
array and for the actual population. Set
$$
a=1/(128d),\qquad \alpha_Q=1/64.
\tag{RQPT.1}
$$
For any moment level $H\ge0$, define
$$
\begin{gathered}
A(H)=H^{1/4}/2,\quad
S(H)=(H^{1/2}/4+\sigma_r^2)^{1/2},\quad
k(H)=[2S(H)]^{-1},\quad B(H)=A(H)/\sigma_r,\\
E_j(H)=\left[\frac{j}{2e k(H)}\right]^{j/2}\quad(j>0),\\
Q_x(H)=4J_re^{B(H)}E_1(H)/\sigma_r,\qquad
Q_m(H)=4J_re^{B(H)}/\sigma_r,\\
Q_v(H)=2J_re^{B(H)}[A(H)+E_2(H)/2]/\sigma_r^3,\\
T_m(H)=H^{1/8}+H^{1/4},\qquad
T_v(H)=2H^{3/8}+\tfrac32H^{1/2}.
\end{gathered}
\tag{RQPT.2}
$$
Here $J_r,J_s$ are the positive-power linear derivative envelopes used
in {prf:ref}`def-rqf-record`. In particular $Q_x,Q_m,Q_v$ bound the
corresponding actual reward-fitness derivatives after multiplication by
$\theta$.
:::

(sec-rqpt-normalization)=
## 2. Raw empirical normalizers under weak transport

:::{prf:lemma} Weak-transport changes of the quadratic reward moments
:label: lem-rqpt-normalization

For capped laws $\mu,\mu'$ with positional eighth moments at most $H$,
write $\delta=\mathsf d(\mu,\mu')$. Then
$$
\begin{aligned}
|\mu R-\mu'R|&\le T_m(H)\delta^{1/4},\\
|\operatorname{Var}_\mu R-\operatorname{Var}_{\mu'}R|
                         &\le T_v(H)\delta^{1/4}.
\end{aligned}
\tag{RQPT.3}
$$
These estimates apply to atomic empirical laws as well as continuous laws.
:::

:::{prf:proof}
For $0<\delta\le1$, choose a bounded-transport optimal coupling and put
$r=\sqrt\delta$. Its bad event $E=\{|z-z'|>r\}$ has probability at
most $\sqrt\delta$. On its complement,
$$
|R(x)-R(x')|\le\tfrac12r(|x|+|x'|).
$$
Its expected good contribution is at most $H^{1/8}\sqrt\delta$.
On $E$, bound the difference by $(|x|^2+|x'|^2)/2$. Cauchy--Schwarz
and each fourth positional moment at most $H^{1/2}$ bound the bad
contribution by $H^{1/4}\delta^{1/4}$. Since
$\sqrt\delta\le\delta^{1/4}$, this proves the mean estimate.

For the second reward moment, use
$$
|R(x)^2-R(x')^2|
\le\tfrac12|x-x'|(|x|^3+|x'|^3).
$$
Indeed $(u+u')(u^2+u'^2)\le2(u^3+u'^3)$ for $u,u'\ge0$.
The good contribution is at most $H^{3/8}\sqrt\delta$.
The bad contribution is at most
$\tfrac14\mathbb E[(|x|^4+|x'|^4)1_E]
\le\tfrac12H^{1/2}\delta^{1/4}$.
Each reward mean has absolute value at most $A(H)$, so the difference
of their squares is at most
$2A(H)T_m(H)\delta^{1/4}$. Combining these three terms gives precisely
$T_v(H)$ in (RQPT.2). The case $\delta=0$ follows by identical laws.
:::

(sec-rqpt-modulus)=
## 3. Complete raw-reward population modulus

:::{prf:lemma} Explicit raw-reward moment-local modulus
:label: lem-rqpt-population-modulus

Retain $\ell_f,D_*,S_*,\kappa_b,w_b',b_0,m_s,Q_s,k_v$ from
{prf:ref}`lem-vupt-population-modulus`; their formulas depend on
comparison features and diversity, without using bounded reward.
Use the actual clipped-gate constant $L_a$ and define
$$
\begin{aligned}
E_F^Q(H)&=\theta[Q_x(H)+Q_m(H)T_m(H)+Q_v(H)T_v(H)+J_sQ_s],\\
D_\beta^Q(H)&=2w_C'/\kappa_C+(2w_C'+1)/\kappa_C^2
                                      +2L_aE_F^Q(H)/\kappa_C,\\
C_{J,Q}(H)&=b_0+8[D_\beta^Q(H)+2b_0/\kappa_C]
                                 +2e^{2C}+1+k_v,\qquad C=2/\kappa_C,\\
\mathcal M_Q(H)&=C_{K,{\rm p}}(1+16P_0^4)^{1/4}
                     (1+H)^{1/4}C_{J,Q}(H)^{1/8}.
\end{aligned}
\tag{RQPT.4}
$$
Here $P_0,C_{K,{\rm p}}$ are the source moment and dense kinetic
stability constants of {prf:ref}`thm-vupt-conditional-consistency`,
evaluated at the actual raw-reward parameters. For capped inputs with
positional eighth moments at most $H$,
$$
\boxed{\quad
\mathsf d(\mathcal F^Q\mu,\mathcal F^Q\mu')
 \le\min\{1,\mathcal M_Q(H)\mathsf d(\mu,\mu')^{1/64}\}.
\quad}
\tag{RQPT.5}
$$
Every coefficient in (RQPT.4) is an explicit primitive function of $H$.
As $H\to\infty$ it satisfies
$$
\log[1+\mathcal M_Q(H)]
                   =O(H^{1/4}+\log(1+H)).
\tag{RQPT.6}
$$
:::

:::{prf:proof}
Let $\delta=\mathsf d(\mu,\mu')>0$ and $r=\sqrt\delta$. The
measurement-law coupling of {prf:ref}`thm-slct-modulus` uses bounded
comparison features only. It gives bad marked-type mass at most
$b_0\sqrt\delta$, with both own and companion states within $r$ on
its complement. Apply {prf:ref}`lem-rqpt-normalization` to the exact
reward moments. Interpolated means and variances remain in the intervals
of {prf:ref}`lem-rqf-logistic-tail`, with $H_8$ replaced by $H$.
Their full fitness contribution is at most
$\theta[Q_m(H)T_m(H)+Q_v(H)T_v(H)]\delta^{1/4}$.
The good roots themselves can differ physically. The same tail lemma
bounds the reward contribution to the fitness gradient globally by
$\theta Q_x(H)$, including along the segment joining those roots;
its physical change is at most $\theta Q_x(H)r$.
The bounded-diversity normalization calculation gives at most
$\theta J_sQ_s\sqrt\delta$, including the physical and measurement
changes. Combining them and using $\sqrt\delta\le\delta^{1/4}$
gives good-type fitness error $E_F^Q(H)\delta^{1/4}$.
This step retains both actual raw reward normalizers and does not use a
bounded-reward Lipschitz constant.

Subtracting the cloning weight, reciprocal normalizer and gate gives
accepted-edge density difference at most
$D_\beta^Q(H)\delta^{1/4}$ on good paired types. The bad-type
contribution is at most $2b_0\sqrt\delta/\kappa_C$, hence at most
$2b_0\delta^{1/4}/\kappa_C$. The outgoing subprobability and
incoming marked Poisson process can therefore be coupled with failure
hazard at most
$4[D_\beta^Q(H)+2b_0/\kappa_C]\delta^{1/4}$ per queried vertex.
Use the actual rooted forest with the outgoing edge already consumed
at each incoming child. Stop at $K=\lceil\delta^{-1/8}\rceil$.
Since $K\le2\delta^{-1/8}$, the cumulative failure bound is at most
$8[D_\beta^Q(H)+2b_0/\kappa_C]\delta^{1/8}$.
The two complete component tails are at most
$2e^{2C}/K\le2e^{2C}\delta^{1/8}$.
These are complete first-marginal component tails and conditional
query hazards; independence of matched future components is not assumed.

On matching components, give both preparations their common Haar
matrix and their actual common recipient jitters. Source positions
differ by at most $r$, and original frozen-slot collision velocities
by at most $k_vr$. Their bounded output cost is at most $(1+k_v)r$.
Together with the root-mark failure this proves
$$
\mathsf d(J(\mu),J(\mu'))\le C_{J,Q}(H)\delta^{1/8}.
$$
Both prepared phase eighth moments are at most $P_0^8(1+H)$; this
uses the incoming donor column and Gaussian jitter, not bounded raw
reward. The deterministic version of {prf:ref}`lem-vupt-cell` gives
$$
W_4(J(\mu),J(\mu'))
\le(1+16P_0^4)^{1/4}(1+H)^{1/8}
                             C_{J,Q}(H)^{1/8}\delta^{1/64}.
$$
The actual two-kick kinetic stability in {prf:ref}`lem-vupt-kinetics`
has coefficient at most $C_{K,{\rm p}}(1+H)^{1/8}$ on these laws.
It compares the actual correlated second provider and is independent
of the reward channel. Since $\mathsf d\le W_2$, (RQPT.5) follows.
The case $\delta=0$ uses identical laws.

For (RQPT.6), $S(H)\le(\sigma_r+1/2)(1+H)^{1/4}$, so
$E_1(H)=O((1+H)^{1/8})$ and $E_2(H)=O((1+H)^{1/4})$.
Equations (RQPT.2)--(RQPT.4) give
$C_{J,Q}(H)=O(e^{B(H)}(1+H)^{3/4})$.
Consequently
$\mathcal M_Q(H)=O(e^{B(H)/8}(1+H)^{11/32})$.
The constants in these asymptotic bounds depend only on the fixed
primitives; the executable bounds remain the exact formulas (RQPT.4).
This proves the asserted logarithmic growth.
:::

(sec-rqpt-consistency)=
## 4. Conditional consistency at the actual empirical input

:::{prf:lemma} The raw quadratic complete finite update has the same consistency rate
:label: lem-rqpt-conditional-consistency

For every actual fixed all-alive input array $S$ with
$M_8(L_N(S))\le H$, the raw-reward finite update satisfies
$$
\mathbb E[\mathsf d(L_N(S^+),\mathcal F^QL_N(S))\mid S]
\le\min\{1,C_{\rm cons}(1+H)^{9/32}N^{-1/(128d)}\}.
\tag{RQPT.7}
$$
The explicit $C_{\rm cons}$ is the same primitive formula in
{prf:ref}`thm-vupt-conditional-consistency`, using actual raw-reward
positive-base derivative bounds. No iid hypothesis on $S$ is imposed.
:::

:::{prf:proof}
For preparation, repeat the fixed-input extraction of
{prf:ref}`lem-vupt-preparation`. Resampling one measurement changes
only sampled diversity statistics. Every raw reward value and its
actual empirical mean and variance are unchanged. The diversity
fitness derivative is bounded by $H_s$ because the logistic reward
base always lies between its positive floor and its finite maximum,
independently of how large the raw quadratic rewards are.
The measurement, donor/gate, component-Haar and jitter influence
bounds and their constants $A_D,A_{\rm p}$ are therefore unchanged.

In the bias integration device of
{prf:ref}`thm-chaos-canonical-quantitative-bias`, only the sampled
diversity moments are replaced by their population values at the
same atomic input $L_N(S)$. Reward moments of that population input
are exactly the actual input array's raw empirical moments. Thus
the diversity normalization comparison, finite-label exploration,
self-exclusion, shared failed-proposal marks, Poisson comparison
and component tails have exactly the same $B_{\rm p}$ as before.
Their prepared scalar mean-square bound is $G_{\rm p}/N$.
This is a separate proof at fixed input; bounded raw reward is not
an assumption needed by these graph and scalar estimates.

The preparation eighth-moment estimate and cell upgrade use only
the accepted donor column, recipient Gaussian and frozen-slot speed
bound. They too are unchanged. Conditional on the entire prepared
array, the first OU stage has independent own innovations and the
actual count field at that array's empirical law. The following
second kick uses that actual joint noisy empirical provider.
The conditional cell estimates and $W_4$-to-$W_2$ force stability
in {prf:ref}`lem-vupt-kinetics` contain no reward assumption.
The independent final Gaussian estimate likewise uses its realized
conditional moment and then averages by Jensen. Therefore the same
transport triangle and $C_{\rm cons}$ proof give (RQPT.7), retaining
all measured fitness statistics and both dense kicks.
:::

(sec-rqpt-uniform)=
## 5. Uniform-time raw-reward particle floor

:::{prf:theorem} Raw-reward alive particle laws approach the population stationary law
:label: thm-rqpt-uniform-time

In {prf:ref}`def-rqpt-record`, assume only
$\sup_{N\ge2}\mathbb EM_8(\widehat\mu_0^N)\le M_{8,0}<\infty$.
Define
$$
\begin{gathered}
q'=\max\{r_*,\sqrt{\lambda_8}\}<1,\qquad
D_\beta^{\rm cl}=2+2\beta(1+\sqrt{H_8}),\\
C_{\rm mix}=q'^{-2}\max\{1,D_\beta^{\rm cl}\}\max\{1,3/H_8\},\\
\overline H=\max\{M_{8,0},2H_8/3\},\qquad
C_*=C_{\rm mix}(1+\overline H),\qquad L_N=\log(N+e),\\
H_N=1+H_8+\log L_N,\qquad
b_N=1+\left\lfloor\frac{\log(1+\log L_N)}{2\log64}\right\rfloor,\\
A_N=\min\{1,C_{\rm cons}(1+H_N)^{9/32}N^{-a}\},\qquad
D_N=1+\mathcal M_Q(H_N),\\
V_N=D_N^{1/(1-\alpha_Q)}A_N^{\alpha_Q^{b_N-1}},\\
\varepsilon_N^Q=V_N+\frac{(b_N+1)\overline H}{H_N}+C_*q'^{b_N}.
\end{gathered}
\tag{RQPT.8}
$$
Then $\varepsilon_N^Q\to0$ and, for every $N\ge2$ and every $n\ge0$,
$$
\boxed{\quad
\mathbb E\mathsf d(\widehat\mu_n^N,\pi_Q)
\le\min\{1,C_*q'^n+\varepsilon_N^Q\}.
\quad}
\tag{RQPT.9}
$$
All constants and both terms are explicit functions of the primitive
parameters, dimension and initial averaged moment budget. The target
is the raw-reward conservative population stationary law, rather than
a finite-array invariant law or a killed-chain quasi-stationary law.
:::

:::{prf:proof}
The actual finite conditional moment drift gives
$\sup_{n,N}\mathbb EM_8(\widehat\mu_n^N)\le\overline H$.
The population attraction proof of {prf:ref}`lem-vupt-global-population`
uses only that drift, fresh final Gaussian regularity and the complete
population weighted contraction. All three hold for the actual raw map
by {prf:ref}`cor-rqf-active-population`. It therefore gives, for every
capped law of finite eighth moment $M$ including atomic laws,
$$
\mathsf d((\mathcal F^Q)^j\mu,\pi_Q)
       \le\min\{1,C_{\rm mix}(1+M)q'^j\}.
$$

Fix $n,N$, put $m=\min\{n,b_N\}$, $k=n-m$, and start the actual
population forecast at $\mu_0=\widehat\mu_k^N$:
$\mu_j=(\mathcal F^Q)^j\widehat\mu_k^N$.
Let $G$ be the single whole-window event that each actual empirical
eighth moment at times $k,\ldots,k+m$ is at most $H_N$.
Its failure probability is at most $(m+1)\overline H/H_N$.
Since $H_N\ge H_8$ and $\lambda_8H_N+B_8\le H_N$, all forecast
moments are also at most $H_N$ on $G$.

With $e_j=\mathbb E[1_G\mathsf d(\widehat\mu_{k+j}^N,\mu_j)]$,
the transport triangle, (RQPT.5) and (RQPT.7) give
$$
e_{j+1}\le A_N+(D_N-1)e_j^{\alpha_Q},\qquad e_0=0.
$$
For the local consistency term, enlarge $1_G$ to the past-measurable
event $\{M_8(\widehat\mu_{k+j}^N)\le H_N\}$ before invoking its
conditional estimate. Thus the own kernel is not conditioned on
future good outcomes. Jensen on the subprobability $1_Gd\Pr$
justifies the modulus term. The induction of
{prf:ref}`thm-vupt-uniform-time` now yields
$e_m\le V_N$ because $A_N\le1$ and $m\le b_N$.
Charge the one outside-window failure only after this induction.
The forecast attraction, averaged over its actual initial law,
is at most $C_*q'^m\le C_*q'^n+C_*q'^{b_N}$.
These facts prove (RQPT.9).

It remains to check the raw-reward envelope does not destroy the
vanishing floor. Here $H_N=O(\log L_N)$ and
$b_N=O(\log(1+\log L_N))$, so the outside-window term tends to zero.
Also $b_N\to\infty$, hence $q'^{b_N}\to0$.
The exact cutoff gives
$\alpha_Q^{b_N-1}\ge(1+\log L_N)^{-1/2}$.
By (RQPT.6),
$$
\log D_N=O((\log L_N)^{1/4}+\log(1+\log L_N)),\qquad
\log A_N\le-a\log N+O(\log(1+\log L_N)).
$$
For sufficiently large $N$, the latter bound is negative, and therefore
$$
\log V_N\le
O((\log L_N)^{1/4}+\log(1+\log L_N))
-\frac{a\log N-O(\log(1+\log L_N))}
       {\sqrt{1+\log L_N}}\longrightarrow-\infty.
$$
This verifies $V_N\to0$, including the actual exponential logistic
tail constant $e^{B(H_N)}$. It has not been replaced by a uniform
bounded-reward constant. Every term of (RQPT.8) consequently vanishes.
:::

:::{prf:corollary} Optimal alive physical transport for the raw reward
:label: cor-rqpt-alive-w2

Put $u_{N,n}^Q=\min\{1,C_*q'^n+\varepsilon_N^Q\}$ and
$$
K_4=2[\max\{\sqrt{\overline H},\sqrt{H_8}\}+V^4],\qquad
C_G=\lambda_{\max}(G_{\rm ph})(1+4\sqrt{K_4})
$$
for any positive physical matrix $G_{\rm ph}$. Then
$$
\mathbb E W_{2,G_{\rm ph}}(\widehat\mu_n^N,\pi_Q)^2
\le C_G\sqrt{u_{N,n}^Q}.
\tag{RQPT.10}
$$
The identical bound holds for the squared Wasserstein distance from
the random empirical-law distribution to $\delta_{\pi_Q}$, using
ground distance $W_{2,G_{\rm ph}}$, and for the squared physical
Wasserstein distance from the uniformly sampled alive slot law
$\mathbb E\widehat\mu_n^N$ to $\pi_Q$.
:::

:::{prf:proof}
The uniform eighth-moment estimate and cap give each averaged phase
fourth moment at most $K_4$. Split the optimal bounded-transport coupling
into physical distance at most one and greater than one. This yields
$\mathbb E W_2^2\le(1+4\sqrt{K_4})\sqrt{\mathbb E\mathsf d}$
by the exact Cauchy--Schwarz argument in
{prf:ref}`cor-vupt-alive-w2`. Multiplication by
$\lambda_{\max}(G_{\rm ph})$ and (RQPT.9) prove (RQPT.10).
The random-law target is Dirac, so its squared ground-Wasserstein cost
equals the expected squared empirical cost. Integrating measurable
almost-optimal empirical couplings gives the uniformly sampled law
bound. No exchangeability or empirical total-variation comparison
is required.
:::
