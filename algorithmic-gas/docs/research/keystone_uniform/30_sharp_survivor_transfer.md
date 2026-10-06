# Source-box sharpening of the surviving particle transfer

The source-box identity removes the input-moment localization in the
completed count-viscous survivor transfer. This note uses the actual
configured alive box only for donor sources; stored dead positions and all
Gaussian innovations remain unbounded. The population attraction input is
the independently proved large-box theorem, not an assumed default-preset
gap.

The retained source revision is `6107b67b9e85259581c1932565c1871e8a7e253a`.
The accepted research24 SHA-256 is
`d6d2e00f52df2e244318693ee68b6e7773d27eac4d9cfeef6e99669441ba678c`;
the accepted research26 SHA-256 is
`4abb3dff733374668e0dc86bf691543295ec6868f1f824eeb0d5d0f24a32b494`.
Their complete proofs are in Chapter 18a Sections 6.8–6.9.

(sec-sbst-register)=
## 1. The actual source and comparison constants

:::{prf:definition} Source-box survivor register
:label: def-sbst-register

Use either the bounded-reward register of
{prf:ref}`def-spt-register` or the raw same-potential register of
{prf:ref}`cor-rqk-surviving-alive-law`. In the latter case every
population and survival constant below has its raw value, and the
finite reward comparison uses $R_b=dL^2/2$, $L_R=\sqrt dL$ only after
the fixed box has been selected. Write $\mathcal F_L$, $\pi_L$,
$r_*$, $C_{\rm pop}$, $m_f$, $e_N$, $r_N$ and $c_s$ for these
proved population, alive-count and survival constants. Retain
$a=1/(128d)$, $\alpha=1/32$, $\kappa=1/(16d)$, $g_8$ and $J_d$
from {prf:ref}`def-vupt-register` and {prf:ref}`lem-vupt-cell`.

Compute $G_{\rm p}$ and $C_{\rm p}^{\rm m}$ by the full alive-floor
substitutions in (SPT.7)–(SPT.8). These preparation quantities use
bounded tests and bounded comparison features, not a moment of stored
dead positions. Define
$$
X_L=\sqrt dL+\sigma_Jg_8,\qquad Z_L=X_L+V_c,
$$
$$
A_{{\rm p},L}=2+\tfrac12J_d\sqrt{G_{\rm p}}+2Z_L^2,
\qquad
U_{{\rm p},L}=(1+16Z_L^4)^{1/4}A_{{\rm p},L}^{1/8}.
\tag{SBST.1}
$$
Use $C_K(M)$, $B_2$, $U_{\rm o}$ and $A_{\rm f}^{\rm m}$ from
{prf:ref}`lem-vupt-kinetics` and
{prf:ref}`lem-spt-marked-consistency`. Put
$T_s=\max\{1,(s\sqrt{2\pi})^{-1}\}$ and
$$
\begin{aligned}
\mathcal C_{\rm cons}
&=[T_sB_2U_{\rm o}+A_{\rm f}^{\rm m}]
              (1+X_L^8)^{9/32}
                  +T_sC_K(X_L^8)U_{{\rm p},L},\\
\mathcal C_{\rm mod}
&=T_sC_K(X_L^8)(1+16Z_L^4)^{1/4}
                             (C_{\rm p}^{\rm m})^{1/8}.
\end{aligned}
\tag{SBST.2}
$$
All displayed constants are finite functions of the already fixed
parameters and box, independent of the particle number, observation
time and entering dead-coordinate distribution.
:::

(sec-sbst-source)=
## 2. Universal prepared moments with unbounded stored dead positions

:::{prf:lemma} Prepared source moments independent of the entering dead tail
:label: lem-sbst-source-moments

For any consistent capped surviving array or population law, every
prepared source lies in $D_L$ before recipient jitter. For the actual
prepared empirical phase law $\eta_N$ and the population prepared phase
law $\eta=J(L_N(S))$ at the same fixed empirical input,
$$
\mathbb E[M_8(\eta_N)\mid S]\vee M_8(\eta)\le X_L^8,
$$
$$
\mathbb E[M_{8,z}(\eta_N)\mid S]\vee M_{8,z}(\eta)\le Z_L^8.
\tag{SBST.3}
$$
The deterministic population bounds also hold for any consistent
entering marked law with positive alive mass, even when its retained
dead positions have infinite eighth moment.
:::

:::{prf:proof}
A persistent row is alive and retains a position in $D_L$. An accepted
alive copy takes an alive donor position in $D_L$. Every dead row revives
from such a donor. These three outcomes cover every prepared row, so
$|x_{\rm src}|\le\sqrt dL$ pointwise. There is no fourth outcome retaining
a dead position as a prepared source.

Conditional on all sources and component primitives, the actual recipient
position is $X=x_{\rm src}+I\sigma_J Z$, where $I\in\{0,1\}$ is its
copy/revival indicator and $Z$ is its own independent standard Gaussian.
Minkowski gives $(\mathbb E|X|^8)^{1/8}\le X_L$. Average this
conditional inequality over rows and sources. It applies equally to the
actual empirical preparation and to the exact rooted population
preparation. The collision readout uses original frozen slot velocities
and has the pointwise bound $|v^{\rm p}|\le V_c$. Since a phase norm is
at most $|X|+V_c$, a second Minkowski inequality gives the phase bound
$Z_L$. These computations use the full Gaussian moments and make no
truncation of jitter or stored output positions.
:::

(sec-sbst-comparison)=
## 3. Uniform local comparisons without an input moment condition

:::{prf:lemma} Source-box conditional consistency and population modulus
:label: lem-sbst-local-comparison

For every consistent capped input array $S$ with alive fraction at least
$m_f$, without a condition on its positional eighth moment,
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S^+),\mathcal F_LL_N(S))\mid S]
\le\min\{1,\mathcal C_{\rm cons}N^{-a}\}.
\tag{SBST.4}
$$
For any two consistent capped marked input laws with alive masses at
least $m_f$, without a condition on their dead-position moments,
$$
\mathsf d_{\rm m}(\mathcal F_L\mu,\mathcal F_L\mu')
\le\min\{1,\mathcal C_{\rm mod}
                              \mathsf d_{\rm m}(\mu,\mu')^\alpha\}.
\tag{SBST.5}
$$
:::

:::{prf:proof}
First freeze the input array. The preparation-only conditional scalar
mean-square estimate $G_{\rm p}/N$ in
{prf:ref}`lem-spt-marked-consistency` needs only its alive floor and
bounded test. Alive reward statistics are fixed at this empirical input;
only the sampled alive diversity normalizers fluctuate. Accepted alive
edges and mandatory dead edges retain their actual probabilities and
component readouts. The complete conditional influence and finite-label
bias arguments use the constants displayed in that lemma, with no stored
positional moment. For raw reward the alive values are bounded on $D_L$,
and dead rewards enter neither normalizers nor a mandatory gate.

Apply {prf:ref}`lem-vupt-cell` to the unmarked prepared phase empirical
law and its actual population preparation. Equation (SBST.3) supplies
the phase moment bound with $L=Z_L$. Thus
$$
\mathbb E[\mathsf d(\eta_N,\eta)\mid S]
 \le A_{{\rm p},L}N^{-\kappa},\qquad
\mathbb E[W_4(\eta_N,\eta)\mid S]
 \le U_{{\rm p},L}N^{-a}.
$$
This comparison does not require independent prepared output rows.

Conditional on the complete prepared array, the actual first count
field is exactly the field at its empirical law. Its independent OU
draw gives the conditional joint-cell estimate, and its second kick
uses that actual joint noisy empirical provider. The proof of
{prf:ref}`lem-vupt-kinetics`, followed by the final-Gaussian marked
coupling in {prf:ref}`lem-spt-marked-consistency`, gives conditional
marked kinetic error at most
$$
[T_sB_2U_{\rm o}+A_{\rm f}^{\rm m}]
              (1+M_8(\eta_N))^{9/32}N^{-a}.
$$
Average over preparation; (SBST.3) and concavity replace its moment
factor by $(1+X_L^8)^{9/32}$. For the remaining comparison between
the population kinetic images of $\eta_N$ and $\eta$, the deterministic
target has eighth moment at most $X_L^8$. The proved kinetic stability
coefficient is therefore $C_K(X_L^8)$, followed by the same factor $T_s$
for terminal marking. Combine it with the preparation $W_4$ bound.
The triangle inequality proves (SBST.4) with (SBST.2).

For the modulus let $\delta=\mathsf d_{\rm m}(\mu,\mu')$. The
preparation prefix of {prf:ref}`lem-spt-marked-consistency` gives
$$
\mathsf d(J(\mu),J(\mu'))\le C_{\rm p}^{\rm m}\delta^{1/4}.
$$
That prefix compares alive-only normalizers, good marked-type pairs
and the complete actual ordered components. Their bounded output cost,
component tails and query hazards need no moment of a retained dead
position. In particular a good dead root always copies an alive source.
Use (SBST.3) for the two prepared phase laws in the deterministic
weak-to-$W_4$ upgrade. Their transport is at most
$$
(1+16Z_L^4)^{1/4}(C_{\rm p}^{\rm m})^{1/8}\delta^{1/32}.
$$
The kinetic target moment is again at most $X_L^8$. Apply its actual
joint-provider kinetic stability and final-Gaussian maximal coupling.
This gives (SBST.5). The latter coupling charges mark disagreement
through Gaussian mismatch, including at the actual box faces; it never
treats the terminal indicator as a Lipschitz function.
:::

(sec-sbst-uniform)=
## 4. A sharper uniform-time surviving particle floor

:::{prf:theorem} Survivor transfer without a dead-moment localization term
:label: thm-sbst-sharp-survivor-transfer

In {prf:ref}`def-sbst-register`, set
$$
L_N=\log(N+e),\qquad
b_N=1+\left\lfloor\frac{\log L_N}{2\log32}\right\rfloor,
\qquad D=1+\mathcal C_{\rm mod},
$$
$$
A_N=\min\{1,\mathcal C_{\rm cons}N^{-a}\},\qquad
V_N=D^{1/(1-\alpha)}A_N^{\alpha^{b_N-1}},\qquad
T_N=(1-e_N)^{-b_N},
$$
$$
\widetilde\varepsilon_N
=T_N[V_N+(b_N+1)c_sr_N]+C_{\rm pop}r_*^{b_N}.
\tag{SBST.6}
$$
Then $\widetilde\varepsilon_N\to0$ and for every $N\ge2$, $n\ge1$,
$$
\boxed{\quad
\mathbb E[\mathsf d_{\rm m}(\widehat\mu_n^N,\pi_L)
                                     \mid\tau_N>n]
\le\min\{1,C_{\rm pop}r_*^{n-1}
                                 +\widetilde\varepsilon_N\}.
\quad}
\tag{SBST.7}
$$
The actual alive empirical-law and sampled-law squared Wasserstein
bounds in {prf:ref}`cor-spt-alive-w2` or
{prf:ref}`cor-rqk-surviving-alive-law` hold with this smaller-form
particle estimate in place of (SPT.10) or (RKPF.12). One may also
take the minimum of the two independently proved particle floors.
:::

:::{prf:proof}
Fix $n,N$, put $m=\min\{n-1,b_N\}$ and $k=n-m\ge1$. Start the
ordinary stopped continuation from the actual law at $k$ conditioned
on $\tau_N>k$. Its population forecast starts at the actual marked
empirical law there. Let $G$ be the single event that the alive fraction
is at least $m_f$ at all window times $k,\ldots,k+m$. Extinction is a
failure of this event. The conditional-past bound at the initial time
and the uniform one-update lower-tail estimate give
$$
\mathbb P_k(G^c)\le(m+1)c_sr_N.
$$
No moment event is included in $G$. Its forecast begins with a valid
alive floor; each forecast output has alive mass at least
$p>m_f$ by the proved population safe-return bound. Every local
comparison therefore satisfies (SBST.4)–(SBST.5) on $G$.

Write $e_j=\mathbb E_k[1_G\mathsf d_{\rm m}
(\widehat\mu_{k+j}^N,\mathcal F_L^j\widehat\mu_k^N)]$.
Enlarge $1_G$ to the past-measurable alive-floor event before invoking
the conditional estimate. The triangle inequality and concavity on
the subprobability $1_Gd\mathbb P_k$ give
$$
e_{j+1}\le A_N+(D-1)e_j^\alpha,\qquad e_0=0.
$$
The induction in {prf:ref}`thm-spt-uniform-surviving-law` gives
$e_m\le V_N$ (and zero when $m=0$). Add the single outside-window
error $(m+1)c_sr_N$ after this recurrence. The exact recent-window
survival change of measure, including its starting-state tilt, costs
at most $T_N$. Thus the conditional endpoint forecast error is at
most $T_N[V_N+(b_N+1)c_sr_N]$.

The proved global population estimate is uniform over every
positive-alive forecast input, including arbitrary retained dead
coordinates. Its forecast-stationarity error is at most
$C_{\rm pop}r_*^m$, which is bounded by
$C_{\rm pop}r_*^{n-1}+C_{\rm pop}r_*^{b_N}$. This proves (SBST.7).
For the raw channel this input is supplied by the separate raw
population theorem; no endpoints are recomputed from a box-dependent
bounded reward extension.

Finally $\alpha^{b_N-1}\ge L_N^{-1/2}$ and $D$ is a fixed constant.
For all sufficiently large $N$,
$$
\log V_N\le\frac{\log D}{1-\alpha}
              -\frac{a\log N-\log\mathcal C_{\rm cons}}{\sqrt{L_N}}
\longrightarrow-\infty.
$$
The alive lower tail $r_N$ and extinction bound $e_N$ decay
exponentially in $N$, hence $(b_N+1)r_N\to0$ and $T_N\to1$.
Also $r_*^{b_N}\to0$. All terms in (SBST.6) therefore vanish.
The alive restriction and optimal transport argument is the same
proved normalized coupling in {prf:ref}`cor-spt-alive-w2`; it only
uses this conditional marked distance and its own alive-count bound.
Substitute (SBST.7) to obtain all asserted physical observation bounds.
:::

:::{prf:remark} Completed sharpening and the remaining default gap
:label: rem-sbst-default-gap

This theorem sharpens the completed large-box, small-positive-count-
viscosity regime. It does not prove the population attraction estimate
at $\nu=0.3$, $L=2$. The source-moment and local-comparison lemmas do
not need that attraction assumption and can be reused at a fixed box
with a proved alive floor. An extension of the final theorem additionally
needs the actual population estimate and its alive-mass class.
The removed moment-localization error is not a survival denominator
or a truncation of Gaussian tails.
:::
