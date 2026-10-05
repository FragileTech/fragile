# Uniform-time particle transfer for the active count-viscous gas

This note transfers the conservative population attraction proved in
`19_population_viscous_block.md` to the actual finite swarm. Its comparison
metric is weak transport. An atomic empirical measure is never
compared to a continuous population law in total variation. The finite swarm
retains the sampled global fitness normalizers, the full component Haar
collision, recipient jitters, both count-normalized viscous kicks, their
correlated second-stage provider, and the native velocity cap.

The resulting estimate has a population contraction term and an explicit
floor tending to zero with the particle number, uniformly over all times.
It does not assert exact mixing of the full finite-array law.

(sec-vupt-register)=
## 1. The conservative register and comparison metric

:::{prf:definition} Actual particle and population register
:label: def-vupt-register

Use the entire primitive regime and the positive endpoints of
{prf:ref}`thm-pvb-active-population-convergence`, with $F(x)=-x$,
$D=\mathbb R^d$, current-frame fitness, death disabled, and $N\ge2$.
Write $\mathcal F=\mathcal F_{\nu,\theta}$ for its actual population map,
$S_n^N$ for the actual finite-array chain, and $\widehat\mu_n^N=L_N(S_n^N)$.
The stored velocities obey $|v|\le V$ and prepared velocities obey
$|v^{\rm p}|\le V_c=(1+2|\alpha_{\rm col}|)V$.
Positions are copied; the component collision uses the original frozen slot
velocities. Let $J(\mu)=\mu J_\mu$ denote preparation alone, and $K$ the
actual population count kinetic map, so $\mathcal F=KJ$.

Retain the constants $t,c,b,q,s,\rho,\sigma_J,a_*,c_*,r_8,B_8,H_8,\beta,r_*$
of that theorem. In particular
$$
\lambda_8=\lambda_{\rm burn}=(1+3r_8)/4<1,\qquad
M_8^+\le\lambda_8M_8+B_8,\qquad
B_8/(1-\lambda_8)=2H_8/3.
\tag{VUPT.1}
$$
The last moment recursion holds both for a population law and for the
expected averaged positional eighth moment of the actual finite array,
conditionally on its entering array.

For phase points $z=(x,v)$ put
$$
c_0(z,z')=\min\{1,|z-z'|\},\qquad
\mathsf d(\mu,\mu')=\inf_{\pi\in\Pi(\mu,\mu')}\int c_0\,d\pi.
$$
$M_8(\mu)=\int|x|^8\,d\mu$ is positional; $M_{8,z}(\mu)=\int|z|^8\,d\mu$
is a phase moment. Let
$$
\ell_\rho=e^{-1/2}/\rho,\quad
g_8=[d(d+2)(d+4)(d+6)]^{1/8},\quad
\kappa=\frac1{16d},\quad a=\frac{\kappa}{8}=\frac1{128d},\quad
\alpha=\frac1{32}.
$$
Write $\log_+(u)=\log\max\{1,u\}$ for $u\ge0$.
All numerical cutoffs below are used only in estimates and do not change
the algorithm.
:::

The strict moment gap in (VUPT.1) matters for finite entrance into the
population contraction class. The weaker endpoint with stationary drift
level exactly $H_8$ would only prove asymptotic approach to that sublevel.

(sec-vupt-cell)=
## 2. Weak sampling and the moment upgrade

:::{prf:lemma} A conditional cell estimate without independent output rows
:label: lem-vupt-cell

Let $\eta_N$ be a random phase probability measure and $\eta$ a deterministic
phase probability measure. Suppose, for every measurable $|\varphi|\le1$,
$$
\mathbb E|(\eta_N-\eta)\varphi|^2\le G/N,
\qquad
\mathbb E M_{8,z}(\eta_N)\vee M_{8,z}(\eta)\le L^8 .
$$
Define $J_d=(2+2\sqrt{2d})^{2d}$. Then
$$
\mathbb E\mathsf d(\eta_N,\eta)
\le [2+\tfrac12J_d\sqrt G+2L^2]N^{-\kappa}.
\tag{VUPT.2}
$$
Moreover, for any random or deterministic pair of probability measures with
$\mathbb E M_{8,z}(\eta_N)\vee M_{8,z}(\eta)\le L^8$,
$$
\mathbb E W_4(\eta_N,\eta)
\le (1+16L^4)^{1/4}
      [\mathbb E\mathsf d(\eta_N,\eta)]^{1/8}.
\tag{VUPT.3}
$$
The same assertions hold conditionally on any fixed earlier state.
:::

:::{prf:proof}
Partition the phase box $[-R,R]^{2d}$ into cells of diameter at most
$\ell$, using coordinate intervals of length at most $\ell/\sqrt{2d}$.
There are at most
$(1+\lceil2R\sqrt{2d}/\ell\rceil)^{2d}$ cells. Match common mass in each
cell, and match the remaining mass arbitrarily. Its bounded cost is at most
$$
2\ell+\tfrac12\sum_C|\eta_N(C)-\eta(C)|
                  +\eta_N(|z|>R)+\eta(|z|>R).
$$
The cell second-moment hypothesis, Cauchy--Schwarz, and Markov's inequality
bound its expectation by
$2\ell+\tfrac12J(R,\ell)\sqrt{G/N}+2L^2/R^2$.
Take $R=N^\kappa$, $\ell=N^{-\kappa}$. Since $N\ge1$,
$J(R,\ell)\le J_dN^{4d\kappa}=J_dN^{1/4}$.
Both $N^{-1/4}$ and $N^{-2\kappa}$ are at most $N^{-\kappa}$, which proves
(VUPT.2). No independence among the rows of $\eta_N$ was used.

Choose a coupling minimizing bounded transport and write $u=|z-z'|$.
On $u\le1$, $u^4\le u$; on $u>1$,
$u^4\le8(|z|^4+|z'|^4)$. Also
$(|z|^4+|z'|^4)^2\le2(|z|^8+|z'|^8)$ and
$\Pr(u>1)\le\mathsf d(\eta_N,\eta)$. Thus, pointwise in the random laws,
$$
W_4^4\le \mathsf d
 +8\sqrt{2[M_{8,z}(\eta_N)+M_{8,z}(\eta)]}\sqrt{\mathsf d}.
$$
Take expectations and use Cauchy--Schwarz. Since $\mathsf d\le1$,
$\mathbb E W_4^4\le(1+16L^4)\sqrt{\mathbb E\mathsf d}$.
Jensen proves (VUPT.3). Optimal bounded-cost couplings exist on phase space;
measurable almost-optimal choices give the same assertion for random laws.
:::

(sec-vupt-preparation)=
## 3. The actual conditional preparation estimate

:::{prf:lemma} Preparation consistency at the actual atomic empirical input
:label: lem-vupt-preparation

Freeze an actual all-alive input array $S$ and put $\mu_N=L_N(S)$.
Let $\eta_N$ be the empirical post-preparation phase law, and let
$\eta=J(\mu_N)$ be the actual population rooted preparation at that same
atomic input. For every measurable $|\varphi|\le1$,
$$
\mathbb E[|(\eta_N-\eta)\varphi|^2\mid S]\le G_{\rm p}/N,
\tag{VUPT.4}
$$
where the following constants are finite functions of the primitive
parameters alone:
$$
\begin{gathered}
C=2/\kappa_C,\quad D_D=2/\kappa_D,\quad
L_q=S_*/\sigma_s+3S_*^3/(2\sigma_s^3),\\
L_0=H_sL_q,\quad B=C+2L_aL_0,\\
A_D=9M_2(2C)[(1+B)^2+B]+\max\{1,4B^2\},\\
A_{\rm p}=2[A_D+10M_2(C)+1],\\
A_T=(1+D_D^2)(2S_*^2/\sigma_s^2+5S_*^6/\sigma_s^6),\quad
L_T=2L_aH_s,\\
A_{\rm e}=1+C+D_D,\quad N_0=\lceil(8A_{\rm e})^{6/5}\rceil,\\
B_{\rm p}=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
              +64A_{\rm e}^2+16M_3(C)+\sqrt{N_0},\\
G_{\rm p}=A_{\rm p}+4B_{\rm p}^2 .
\end{gathered}
$$
Here $S_*,H_s,L_a$ are the explicit separation, fitness derivative, and
acceptance constants in {prf:ref}`lem-chaos-canonical-innovation-replacement`,
evaluated with alive floor $m_*=1$. The component moments are
$$
M_p(C)=e^{(2^p-1)C}
 \sum_{k\ge0}[(k+1)^p-k^p]\frac{C^k}{k!}.
$$
Only preparation is asserted in (VUPT.4); no finite-array kinetic locality
is assumed.

If $M_8(\mu_N)\le H$, then, with
$p_0=(1+c_*)^{1/8}+\sigma_Jg_8$ and $P_0=p_0+V_c$,
$$
\mathbb E M_{8,z}(\eta_N)\vee M_{8,z}(\eta)
                 \le P_0^8(1+H).
\tag{VUPT.5}
$$
Consequently set
$$
A_{\rm p,w}=2+\tfrac12J_d\sqrt{G_{\rm p}}+2P_0^2,\qquad
U_{\rm p}=(1+16P_0^4)^{1/4}A_{\rm p,w}^{1/8}.
$$
Then
$$
\mathbb E[W_4(\eta_N,J(\mu_N))\mid S]
\le U_{\rm p}(1+H)^{5/32}N^{-a}.
\tag{VUPT.6}
$$
:::

:::{prf:proof}
Stop the innovation-replacement argument of
{prf:ref}`lem-chaos-canonical-innovation-replacement` immediately after
preparation. The reward statistics depend on the fixed input array and
do not change when a measurement companion is resampled. The diversity
statistics change by the exact amounts used there, giving $L_q,L_0$ above.
The clipped gate is sum-norm Lipschitz with constant $L_a$. Expose changed
outgoing rows and their endpoints before exposing the common forest. Its
independent remaining edges have density at most $2C/N$, so the proof gives
the displayed $A_D$. Donor/gate replacement costs $9M_2(C)$, an addressed
Haar replacement costs $M_2(C)$, and a local jitter replacement affects one
row. The conditional innovation variance argument gives $A_{\rm p}/N$.

Likewise stop the proof of
{prf:ref}`thm-chaos-canonical-quantitative-bias` at the prepared root.
Integrating the empirical diversity standardizer error costs at most
$$
2\|\varphi\|_\infty
[3M_1(2C)L_T\sqrt{A_T}/\sqrt N+4L_T^2A_T/N].
$$
The finite marked exploration retains the auxiliary targets and marks of
failed proposals. For $K=\lfloor N^{1/6}\rfloor$ its finite-label,
self-exclusion, and Poisson comparison costs at most
$64A_{\rm e}^2K^3/N$; the two component tails cost
$2M_3(C)/K^3$. The common component uses the same actual Haar matrix and
the same recipient jitter. Combining those terms, and using the elementary
coupling bound for $N<N_0$, gives bias at most
$2B_{\rm p}\|\varphi\|_\infty/\sqrt N$.
Variance plus squared bias proves (VUPT.4).

These are preparation arguments, so the no-viscosity condition in the
earlier full-update statements is not needed here. Nor is a quadratic
reward needed: their only reward requirement in these arguments is the
bounded logistic reward factor, which holds for the present bounded reward.
All sampled normalizers, graph sharing, and component rotations are
retained. The later dense kicks are dealt with separately below.

The accepted incoming donor column is at most $c_*$; hence the average
source eighth moment is at most $(1+c_*)H$. Minkowski for the Gaussian
recipient jitter gives positional $L^8$ norm at most
$[(1+c_*)H]^{1/8}+\sigma_Jg_8\le p_0(1+H)^{1/8}$.
The frozen original slot velocities and the native component formula give
the speed bound $V_c$. Since $|z|\le|x|+|v|$, this proves (VUPT.5)
for the finite and rooted laws. Apply (VUPT.2) with
$L=P_0(1+H)^{1/8}$ and then (VUPT.3).
The coefficient contributes powers $1/8$ and $1/32$ of $1+H$, which gives
(VUPT.6).
:::

(sec-vupt-kinetics)=
## 4. Conditional dense kinetics with the joint second provider

:::{prf:lemma} Population kinetic stability and conditional particle consistency
:label: lem-vupt-kinetics

For prepared phase laws of speed at most $V_c$, let $M$ bound the target
positional eighth moment. Define
$$
\begin{aligned}
A_1&=1+t+t\nu(2+4\ell_\rho V_c),\\
C_y&=1+bA_1,\qquad C_w=cA_1,\\
u_0&=(1+2t\nu)V_c+t,\qquad
w_0=cu_0+qg_8,\qquad y_0=1+bu_0+tqg_8,\\
k_0&=y_0+w_0,\qquad o_0=y_0+sg_8+V,\\
A_{\rm o}&=2+\tfrac12J_d+2k_0^2,\qquad
U_{\rm o}=(1+16k_0^4)^{1/4}A_{\rm o}^{1/8},\\
A_{\rm f}&=2+\tfrac12J_d+2o_0^2,\\
B_2&=2+t+t\nu(2+4\ell_\rho w_0),\\
C_{\rm kin}&=B_2U_{\rm o}+A_{\rm f}.
\end{aligned}
$$
For an arbitrary fixed prepared array with positional eighth moment $M$,
let $\zeta_N$ be the empirical law after its actual count kinetic update.
Then
$$
\mathbb E\mathsf d(\zeta_N,K\eta_N)
\le C_{\rm kin}(1+M)^{9/32}N^{-a},
\tag{VUPT.7}
$$
where here $\eta_N$ is the fixed entering empirical prepared law.

For two prepared laws $\eta,\eta'$ with $M_8(\eta')\le M$,
$$
W_2(K\eta,K\eta')
\le C_K(M)W_4(\eta,\eta'),
\tag{VUPT.8}
$$
with
$$
\begin{aligned}
w(M)&=c[(1+2t\nu)V_c+t M^{1/8}]+qg_8,\\
C_K(M)&=C_y+C_w+tC_y+
             t\nu(2+4\ell_\rho w(M))(C_y+C_w).
\end{aligned}
$$
No independence of the second provider from its own OU innovations is
needed in either estimate.
:::

:::{prf:proof}
For any coupling of phase laws, the Gaussian count kernel satisfies the
two stability estimates of {prf:ref}`lem-cg-viscous-force-stability`:
$$
\|\nu C_\eta-\nu C_{\eta'}\|_{L^4}
\le\nu(2+4\ell_\rho V_c)W_4(\eta,\eta')
$$
when both entering speeds are bounded by $V_c$, and
$$
\|\nu C_\Lambda-\nu C_{\Lambda'}\|_{L^2}
\le\nu(2+4\ell_\rho\|w'\|_{L^4})W_4(\Lambda,\Lambda')
\tag{VUPT.9}
$$
for the joint noisy phase laws. These follow by subtracting the kernel
and velocity factors under a product of the same coupling; the latter
bound uses Hölder and the target's fourth velocity moment.
They hold for atomic laws as well.

Couple entering laws by $W_4$ and use the same own OU and final Gaussian
innovations. The first kick has $L^4$ difference at most $A_1W_4$.
The joint landing position and OU velocity have respective differences
at most $C_yW_4,C_wW_4$, since the first landing position is
$x+bv_1+tq\xi$. Its target OU velocity fourth norm is at most $w(M)$.
Apply (VUPT.9) to the joint law, then the second kick $w-ty+t\nu C$.
The smooth native cap is 1-Lipschitz. Position contributes $C_yW_4$,
and velocity contributes
$[C_w+tC_y+t\nu(2+4\ell_\rho w(M))(C_y+C_w)]W_4$.
Their sum bounds the phase $L^2$ difference and proves (VUPT.8).

For (VUPT.7), first freeze the entire prepared array. Its first count
kick is exactly the population first kick at its empirical law, including
the zero self-contribution $K_\rho(x_i,x_i)(v_i-v_i)/N$.
After the OU draw, the empirical joint law $\Lambda_N$ of $(y_i,w_i)$
is an average of independent, possibly nonidentical random variables.
Its conditional mean is the joint population law $\Lambda$ obtained from
that same entering empirical law. Every bounded test has conditional
variance at most $1/N$.

The first count force has averaged $L^8$ norm at most $2\nu V_c$.
The kinetic moment recursion gives
$$
\|v_1\|_8\le u_0(1+M)^{1/8},\quad
\|w\|_8\le w_0(1+M)^{1/8},\quad
\|y\|_8\le y_0(1+M)^{1/8}.
$$
These inequalities hold for the joint population law and for the expected
empirical eighth moments, without assuming equal row distributions.
Apply (VUPT.2) and (VUPT.3) with phase moment bound
$k_0^8(1+M)$ to obtain
$$
\mathbb E W_4(\Lambda_N,\Lambda)
\le U_{\rm o}(1+M)^{5/32}N^{-a}.
$$
Now apply the actual second-kick/cap/final-Gaussian population map to
$\Lambda_N$ and $\Lambda$. For a $W_4$ coupling the difference of its phase
outputs is at most
$[2+t+t\nu(2+4\ell_\rho w_0(1+M)^{1/8})]W_4$ in $L^2$.
This is bounded by $B_2(1+M)^{1/8}W_4$.
In the map at $\Lambda_N$, its count field is exactly the field of the
actual noisy array. This comparison keeps every correlation of the
second kick with the OU innovations. It costs at most
$B_2U_{\rm o}(1+M)^{9/32}N^{-a}$ in expected bounded transport.

Finally freeze that entire noisy array, including the second kick and the
stored capped velocities. The remaining position Gaussians are independent
across slots. Their empirical law has conditional mean equal to the last
Gaussian pushforward of the array's joint landing/capped-velocity law, and
conditional bounded-test variance at most $1/N$.
The expected averaged output phase eighth moment is at most $o_0^8(1+M)$:
the second kick changes velocity, which is capped by $V$, and does not
change landing position. If $L_{\rm f}^8$ is that moment conditional on the
realized OU array, its value is random; only
$\mathbb E L_{\rm f}^8\le o_0^8(1+M)$ has been proved.
Apply the conditional cell estimate with this realized $L_{\rm f}$, then
use $\mathbb E L_{\rm f}^2\le(\mathbb E L_{\rm f}^8)^{1/4}$.
After averaging this costs at most
$A_{\rm f}(1+M)^{1/4}N^{-\kappa}$.
This is at most $A_{\rm f}(1+M)^{9/32}N^{-a}$.
The transport triangle inequality proves (VUPT.7).
:::

:::{prf:theorem} Quantitative conditional consistency of the full dense update
:label: thm-vupt-conditional-consistency

Freeze an actual input array $S$ with $M_8(L_N(S))\le H$.
Define $p_0,P_0,U_{\rm p}$ as in
{prf:ref}`lem-vupt-preparation`, and put
$$
w_{\rm p}=c[(1+2t\nu)V_c+tp_0]+qg_8,
$$
$$
C_{K,{\rm p}}=C_y+C_w+tC_y+
                 t\nu(2+4\ell_\rho w_{\rm p})(C_y+C_w),
$$
$$
C_{\rm cons}=C_{\rm kin}(1+p_0^8)^{9/32}
                              +C_{K,{\rm p}}U_{\rm p}.
$$
Then the actual complete finite update satisfies
$$
\boxed{\quad
\mathbb E[\mathsf d(L_N(S^+),\mathcal F L_N(S))\mid S]
\le\min\{1,C_{\rm cons}(1+H)^{9/32}N^{-a}\}.
\quad}
\tag{VUPT.10}
$$
This estimate is conditional at the actual empirical input, for every
$N\ge2$, and contains no iid assumption on the entering swarm.
:::

:::{prf:proof}
Couple the actual preparation to its empirical law $\eta_N$ and use
$\eta=J(L_N(S))$. Conditional on preparation, (VUPT.7) applies with
$M=M_8(\eta_N)$. The positional part of (VUPT.5) gives
$\mathbb E M\le p_0^8(1+H)$. Since $9/32<1$, Jensen bounds the first kinetic
comparison by
$C_{\rm kin}(1+p_0^8)^{9/32}(1+H)^{9/32}N^{-a}$.
For the comparison $K\eta_N$ to $K\eta$, its target moment is deterministic
conditional on $S$ and at most $p_0^8(1+H)$.
(VUPT.8) therefore has coefficient at most
$C_{K,{\rm p}}(1+H)^{1/8}$. Multiply this by (VUPT.6).
The transport triangle inequality gives (VUPT.10), and $\mathsf d\le1$
gives its minimum. The second provider has been compared as a joint law;
no part of this proof replaces the dense array kernel by a product kernel.
:::

(sec-vupt-modulus)=
## 5. A weak modulus for the complete viscous population map

:::{prf:lemma} Explicit moment-local Hölder modulus
:label: lem-vupt-population-modulus

For the feature and fitness constants of {prf:ref}`def-slct-regime` and
{prf:ref}`thm-slct-modulus`, evaluated with the present bounded reward,
put
$$
\begin{gathered}
\ell_f=\max\{1,\sqrt{\lambda_{\rm alg}}\},\quad
D_*=2\sqrt{(R_x^{\rm feat})^2+
                    \lambda_{\rm alg}(R_v^{\rm feat})^2},\\
S_*=\sqrt{D_*^2+\delta_D^2},\quad
\kappa_b=e^{-D_*^2/(2\epsilon_b^2)},\quad
w_b'=D_*\ell_f/\epsilon_b^2\quad(b=D,C),\\
b_0=1+(3+4w_D')/\kappa_D,\quad
m_r=L_R+2R_b,\quad m_s=2\ell_f+S_*b_0,\\
Q_r=(L_R+m_r)/\sigma_r+4R_b^2m_r/\sigma_r^3,\quad
Q_s=(2\ell_f+m_s)/\sigma_s+2S_*^2m_s/\sigma_s^3,\\
E_F=H_rQ_r+H_sQ_s,\quad
D_\beta=2w_C'/\kappa_C+(2w_C'+1)/\kappa_C^2
                              +2L_aE_F/\kappa_C,\\
C_J^{\rm weak}=b_0+8(D_\beta+2b_0/\kappa_C)+2e^{2C}+1+k_v,\quad
k_v=1+2|\alpha_{\rm col}|,\\
C_{\rm mod}=C_{K,{\rm p}}(1+16P_0^4)^{1/4}
                                       (C_J^{\rm weak})^{1/8}.
\end{gathered}
$$
Here $|R|\le R_b$ and $|\nabla R|\le L_R$; $H_r,H_s,L_a$ have the actual
logistic/power/gate formulas in the preceding proved references.
Every constant above is independent of $H$ and $N$.
For capped input laws with $M_8(\mu)\vee M_8(\mu')\le H$,
$$
\boxed{\quad
\mathsf d(\mathcal F\mu,\mathcal F\mu')
\le\min\{1,C_{\rm mod}(1+H)^{1/4}
                         \mathsf d(\mu,\mu')^{1/32}\}.
\quad}
\tag{VUPT.11}
$$
The feature radii in these formulas bound the configured squashed feature
maps; they do not impose bounded physical positions.
:::

:::{prf:proof}
The preparation prefix of the proof of {prf:ref}`thm-slct-modulus` is
independent of the later kinetic operator. For completeness, write
$\delta=\mathsf d(\mu,\mu')$, $r=\sqrt\delta$ and couple input types
optimally in bounded transport. Their bad mass, with physical distance
greater than $r$, is at most $\sqrt\delta$.
The actual weighted measurement laws can be coupled with bad marked-type
mass at most $b_0\sqrt\delta$. On its complement both own and companion
physical inputs are within $r$.
Their reward/diversity mean and regularized variance differences give
the standardized differences $Q_r\sqrt\delta,Q_s\sqrt\delta$.
The exact logistic/power derivative gives fitness difference at most
$E_F\sqrt\delta$ there.

Subtracting the cloning weight, its normalization, and its clipped gate
gives accepted-edge density difference at most $D_\beta\sqrt\delta$.
Lift both rooted explorations to that marked-type coupling. The discrepancy
of outgoing subprobabilities and incoming intensities is at most
$(D_\beta+2b_0/\kappa_C)\sqrt\delta$.
Common-intensity Poisson coupling, retaining the outgoing choice already
used by an incoming child, costs at most four times this quantity per
explored vertex. Stopping at
$K=\lceil\delta^{-1/4}\rceil$ costs at most
$8(D_\beta+2b_0/\kappa_C)\delta^{1/4}$.
The two complete component tails cost at most
$2e^{2C}/K\le2e^{2C}\delta^{1/4}$.
On matching components use the actual common Haar rotation and recipient
jitter. Copied positions differ by at most $r$; original frozen velocity
means and their collision readouts differ by at most $k_vr$.
Consequently
$$
\mathsf d(J(\mu),J(\mu'))\le
                  C_J^{\rm weak}\delta^{1/4}.
$$
This prefix retains the sampled mark information and the full rooted
component; the later dense kicks have not been used.

The two prepared phase eighth moments are at most $P_0^8(1+H)$.
The deterministic version of (VUPT.3) gives
$$
W_4(J(\mu),J(\mu'))
\le (1+16P_0^4)^{1/4}(1+H)^{1/8}
                    (C_J^{\rm weak})^{1/8}\delta^{1/32}.
$$
Apply (VUPT.8) with the target prepared moment bound
$p_0^8(1+H)$; its coefficient is at most
$C_{K,{\rm p}}(1+H)^{1/8}$. Since $\mathsf d\le W_2$, this proves
(VUPT.11). The case $\delta=0$ follows by identical input laws.
:::

(sec-vupt-global-population)=
## 6. Global attraction from atomic finite-moment inputs

:::{prf:lemma} Global weak population attraction with a moment prefactor
:label: lem-vupt-global-population

Let $\pi=\pi_{\nu,\theta}$ be the population stationary law in
{prf:ref}`thm-pvb-active-population-convergence`. Define
$$
q'=\max\{r_*,\sqrt{\lambda_8}\}<1,\quad
D_\beta^{\rm cl}=2+2\beta(1+\sqrt{H_8}),\quad
$$
$$
C_{\rm mix}=q'^{-2}\max\{1,D_\beta^{\rm cl}\}
                                     \max\{1,3/H_8\}.
$$
For every capped phase law with finite positional eighth moment $M$,
including every deterministic atomic empirical law, and every $n\ge0$,
$$
\mathsf d(\mathcal F^n\mu,\pi)
\le\min\{1,C_{\rm mix}(1+M)q'^n\}.
\tag{VUPT.12}
$$
:::

:::{prf:proof}
Choose
$$
\tau=\max\{1,\lceil\log_+(3M/H_8)/\log(1/\lambda_8)\rceil\}.
$$
The drift (VUPT.1) gives $M_8(\mathcal F^\tau\mu)\le H_8$.
After the first update the actual fresh final position Gaussian is
independent of the stored capped velocity. Thus
$\mathcal F^\tau\mu$ belongs to the complete Gaussian-regularized class
$\mathfrak G_{8,s}$ of the population contraction theorem.
The full weighted variation diameter of that class is at most
$D_\beta^{\rm cl}$. Since bounded transport is at most full weighted
variation, for $n\ge\tau$,
$$
\mathsf d(\mathcal F^n\mu,\pi)
\le D_\beta^{\rm cl}r_*^{n-\tau}
\le D_\beta^{\rm cl}q'^nq'^{-\tau}.
$$
The choice of $\tau$ implies
$$
q'^{-\tau}\le q'^{-2}
   \max\{1,3M/H_8\}^{\,\log(1/q')/\log(1/\lambda_8)}.
$$
The exponent is at most $1/2$, because $q'\ge\sqrt{\lambda_8}$.
In particular the last expression is at most
$q'^{-2}\max\{1,3/H_8\}(1+M)$.
This proves (VUPT.12) for $n\ge\tau$.
For $n<\tau$, the same bound on $q'^{-\tau}$ implies
$C_{\rm mix}(1+M)q'^n\ge1$; use $\mathsf d\le1$.
No Gaussian representation of the entering atomic measure has been
asserted: the representation used above is created by its actual first
population update.
:::

(sec-vupt-uniform-time)=
## 7. Uniform-time particle transfer with an explicit vanishing floor

:::{prf:theorem} Actual alive finite-swarm attraction to the population law
:label: thm-vupt-uniform-time

Assume the full register of {prf:ref}`def-vupt-register` and initial
averaged moment bound
$$
\sup_{N\ge2}\mathbb E M_8(\widehat\mu_0^N)\le M_{8,0}<\infty.
$$
No exchangeability or iid condition is required. Put
$$
\overline H=\max\{M_{8,0},2H_8/3\},\qquad
C_*=C_{\rm mix}(1+\overline H),\qquad
L_N=\log(N+e),
$$
$$
H_N=1+H_8+\sqrt{L_N},\qquad
b_N=1+\left\lfloor\frac{\log L_N}{2\log32}\right\rfloor,
$$
$$
A_N=\min\{1,C_{\rm cons}(1+H_N)^{9/32}N^{-a}\},\qquad
D_N=1+C_{\rm mod}(1+H_N)^{1/4},
$$
$$
V_N=D_N^{1/(1-\alpha)}A_N^{\alpha^{b_N-1}},\qquad
\varepsilon_N=
 V_N+\frac{(b_N+1)\overline H}{H_N}+C_*q'^{b_N}.
\tag{VUPT.13}
$$
These formulas are explicit functions of the stated primitive parameters,
dimension, initial moment budget, and particle number. They satisfy
$\varepsilon_N\to0$. For every $N\ge2$ and every $n\ge0$,
$$
\boxed{\quad
\mathbb E\mathsf d(\widehat\mu_n^N,\pi_{\nu,\theta})
\le\min\{1,C_*q'^n+\varepsilon_N\}.
\quad}
\tag{VUPT.14}
$$
Thus the particle error floor vanishes uniformly over all observation
times. The stationary target is the conservative population law.
No claim of exact finite-array convergence to a finite-array invariant
law is needed for this conclusion.
:::

:::{prf:proof}
The actual conditional moment recursion (VUPT.1) gives
$$
\sup_{n,N}\mathbb E M_8(\widehat\mu_n^N)\le\overline H.
\tag{VUPT.15}
$$
Fix $n,N$ and let $m=\min\{n,b_N\}$, $k=n-m$.
Start a population forecast at the actual empirical law at time $k$:
$\mu_j=\mathcal F^j\widehat\mu_k^N$ for $0\le j\le m$.
This is a deterministic population evolution conditional on that one
initial empirical law; the finite chain continues with its actual
innovations.
Let $G$ be the single event
$$
G=\{M_8(\widehat\mu_{k+j}^N)\le H_N\text{ for }0\le j\le m\}.
$$
By (VUPT.15) and the union bound,
$$
\Pr(G^c)\le(m+1)\overline H/H_N.
\tag{VUPT.16}
$$
On $G$, the forecast begins with eighth moment at most $H_N$.
Since $H_N\ge H_8$ and
$\lambda_8H_N+B_8\le H_N$, all its moments remain at most $H_N$.

Define
$e_j=\mathbb E[1_G\mathsf d(\widehat\mu_{k+j}^N,\mu_j)]$.
The transport triangle inequality gives
$$
\begin{aligned}
\mathsf d(\widehat\mu_{k+j+1}^N,\mu_{j+1})
&\le\mathsf d(\widehat\mu_{k+j+1}^N,
                                      \mathcal F\widehat\mu_{k+j}^N)\\
&\quad+\mathsf d(\mathcal F\widehat\mu_{k+j}^N,\mathcal F\mu_j).
\end{aligned}
$$
On $G$, the population modulus (VUPT.11) applies to the second term.
For the first term, $G$ is contained in the past-measurable event
$\{M_8(\widehat\mu_{k+j}^N)\le H_N\}$.
Drop $1_G$ in favor of that event before applying the actual conditional
estimate (VUPT.10). This bounds its expectation by $A_N$ without
conditioning the kernel on a future event.
Concavity on the subprobability $1_G\,d\mathbb P$ gives
$$
e_{j+1}\le A_N+(D_N-1)e_j^\alpha,\qquad e_0=0.
\tag{VUPT.17}
$$
If $m\ge1$, elementary induction with
$R_N=D_N^{1/(1-\alpha)}$ gives
$$
e_j\le R_N A_N^{\alpha^{j-1}}\quad(1\le j\le m).
$$
Indeed $A_N\le1$,
$A_N\le A_N^{\alpha^j}$, and
$1+(D_N-1)R_N^\alpha\le R_N$.
As $m\le b_N$, $e_m\le V_N$. For $m=0$ use $e_0=0$.
Add the one outside-window error (VUPT.16) only after this iteration:
$$
\mathbb E\mathsf d(\widehat\mu_n^N,\mu_m)
\le V_N+(b_N+1)\overline H/H_N.
\tag{VUPT.18}
$$
In particular a bad-moment probability is not inserted into every Hölder
recurrence and then repeatedly raised to $\alpha$.

Apply the global population estimate (VUPT.12) to the forecast's atomic
initial law and average over it. Its initial expected eighth moment is
at most $\overline H$, so
$\mathbb E\mathsf d(\mu_m,\pi)\le C_*q'^m$.
When $n<b_N$, this is $C_*q'^n$; when $n\ge b_N$, this is $C_*q'^{b_N}$.
Consequently it is always at most $C_*q'^n+C_*q'^{b_N}$.
The triangle inequality and $\mathsf d\le1$ prove (VUPT.14).

To verify that the stated floor vanishes, $b_N=O(\log L_N)$ while
$H_N$ grows as $\sqrt{L_N}$, so the second term of (VUPT.13) tends to zero.
Also $b_N\to\infty$ and $q'<1$, so its third term tends to zero.
For the first term,
$$
\alpha^{b_N-1}\ge L_N^{-1/2},\quad
\log D_N=O(\log L_N),\quad
\log A_N\le -a\log N+O(\log L_N).
$$
For all sufficiently large $N$ the last upper bound is negative.
It follows that
$$
\log V_N\le O(\log L_N)
        -\frac{a\log N-O(\log L_N)}{\sqrt{L_N}}\longrightarrow-\infty.
$$
Thus $V_N\to0$. This proof uses a logarithmic-logarithmic restart length
because the verified full-map modulus is Hölder. An exponential-in-horizon
Lipschitz propagation estimate has not been assumed.
:::

(sec-vupt-physical)=
## 8. Optimal physical transport of the alive observations

:::{prf:corollary} Alive empirical-law and alive-sampled Wasserstein estimates
:label: cor-vupt-alive-w2

Under {prf:ref}`thm-vupt-uniform-time`, let $G_{\rm ph}$ be any positive
physical phase-space matrix and define
$$
K_4=2\bigl[\max\{\sqrt{\overline H},\sqrt{H_8}\}+V^4\bigr],\qquad
C_G=\lambda_{\max}(G_{\rm ph})(1+4\sqrt{K_4}).
$$
Set $u_{N,n}=\min\{1,C_*q'^n+\varepsilon_N\}$. Then
$$
\mathbb E W_{2,G_{\rm ph}}(\widehat\mu_n^N,\pi_{\nu,\theta})^2
\le C_G\sqrt{u_{N,n}}.
\tag{VUPT.19}
$$
Let $\mathfrak L_{N,n}$ be the law of the random empirical probability
measure $\widehat\mu_n^N$. On the space of physical probability measures,
equipped with $W_{2,G_{\rm ph}}$ as its distance,
$$
\mathbb W_{2,G_{\rm ph}}
 (\mathfrak L_{N,n},\delta_{\pi_{\nu,\theta}})^2
\le C_G\sqrt{u_{N,n}}.
\tag{VUPT.20}
$$
If $I$ is a uniform slot independent of the swarm, its alive sampled law
$\rho_{N,n}=\operatorname{Law}(x_I^n,v_I^n)
=\mathbb E\widehat\mu_n^N$ obeys
$$
W_{2,G_{\rm ph}}(\rho_{N,n},\pi_{\nu,\theta})^2
\le C_G\sqrt{u_{N,n}}.
\tag{VUPT.21}
$$
Thus all three optimal-transport observations have a population-independent
exponential relaxation term and a vanishing particle floor.
The distance rate furnished by these estimates in physical time is
$-\log(q')/(4h)$.
They concern alive laws with death disabled, not conditioning a killed
chain on survival.
:::

:::{prf:proof}
By (VUPT.15), Cauchy--Schwarz, and the native speed cap,
$$
\mathbb E M_{4,z}(\widehat\mu_n^N)
\le2(\sqrt{\overline H}+V^4)\le K_4,\qquad
M_{4,z}(\pi)\le2(\sqrt{H_8}+V^4)\le K_4.
$$
For a bounded-transport optimal coupling, on $u=|z-z'|\le1$ use
$u^2\le u$. On $u>1$ use $u^2\le2(|z|^2+|z'|^2)$ and
Cauchy--Schwarz, as in (VUPT.3). After averaging over the random laws this
gives
$$
\mathbb E W_2(\widehat\mu_n^N,\pi)^2
\le \mathbb E\mathsf d+
    2\sqrt{2[\mathbb E M_{4,z}(\widehat\mu_n^N)+M_{4,z}(\pi)]}
                                      \sqrt{\mathbb E\mathsf d}
\le(1+4\sqrt{K_4})\sqrt{\mathbb E\mathsf d}.
$$
The physical quadratic cost is at most $\lambda_{\max}(G_{\rm ph})$
times Euclidean squared cost. Substituting (VUPT.14) proves (VUPT.19).
The only coupling to a Dirac empirical-law target pairs each random
empirical measure with $\pi$, so its squared transport cost is exactly the
left side of (VUPT.19). This proves (VUPT.20).
Finally integrate measurable almost-optimal couplings of each empirical
measure to $\pi$. Their first marginal is $\mathbb E\widehat\mu_n^N$,
their second marginal remains $\pi$, and their mean cost is at most the
right side of (VUPT.19). Let the optimization slack tend to zero.
This proves (VUPT.21) without an exchangeability assumption.
:::

(sec-vupt-scope)=
## 9. Scope of the completed transfer

The theorem supplies an actual finite-particle uniform-time estimate for the
positive-viscosity, positive-fitness-power interval already made primitive
in (PVB.14). It allows the original harmonic step size and every finite
dimension. Its moment hypothesis is the explicit initial averaged eighth
moment budget. Gaussian tails and confinement remain in the estimates;
physical positions have not been truncated.

The bound is conservative: the proved conditional one-step exponent is
$1/(128d)$ and the verified weak modulus exponent is $1/32$. The resulting
particle floor decreases slowly. A sharper modulus could improve that
floor, but is unnecessary for uniform-time convergence to the population
law as $N\to\infty$.

The population invariant law $\pi_{\nu,\theta}$ is distinct from a full
finite-array invariant measure. The theorem bounds the actual swarm's
observable laws relative to the former. It neither proves uniqueness or
mixing of the full dense finite-array invariant law nor yields a killed-chain
quasi-stationary estimate. Those assertions require separate theorems.
