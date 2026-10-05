# Population-uniform native QSD eigenfunction and Doob comparison

(sec-nue-register)=
## 1. The unchanged active count phase

:::{prf:definition} Native uniform eigenfunction register
:label: def-nue-register

Retain every algorithm, landscape, arithmetic, boundary, observation and
calibration field of {prf:ref}`def-native-concentration-register`. Thus this
is the quadratic real-coordinate dense count gas of Chapters 28 and 30,
with sampled reward/diversity normalization, both current-companion laws,
their singleton conventions, mandatory revival, original gates, frozen
simultaneous sources, full connected-component Haar collisions, recipient
jitter, both viscous kicks, original independent OU and position Gaussians,
the configured smooth radial cap, and every terminal alive/dead mark.
The result requires the consumed jitter amplitude $\sigma_J>0$ and

$$
\kappa_A=1-t^2\lambda-4t^2\nu V_c\ell_\rho>0,
\qquad V_c=k_cV,\qquad \ell_\rho=e^{-1/2}/\rho .
\tag{NUE.1}
$$

This is a primitive additional first-drift test, not regularity of an
unknown QSD. All the concentration-register tests, including
$0<V\le\min(V_0,1)$, $r<1$, $\nu\le\nu_g$, and the inherited finite-QSD
smoothing tests, remain in force. In particular $0\le t\nu\le1/2$.
All consumed numerical parameters are fixed as $N$ varies. Other
normalizations, force providers, history/feedback/curl/elite branches,
fixed-seed finite arithmetic and unavailable population sizes retain
their own kernels and are not conclusions of this theorem.

Use the original coefficients $t,c,b,a_x,q,s,\tau,\lambda,\rho$ and the
actual floor $a_0\in(0,1)$ of the phase register. Write

$$
D=2+2\omega V,\qquad
\varepsilon_N=(1-a_0)^N,\qquad
\zeta_N=e^{-a_0N/8},\qquad
\bar q=(1+r)/2<1.
\tag{NUE.2}
$$

$Q_N$ is the actual surviving restriction of the raw physical kernel,
$\nu_NQ_N=\alpha_N\nu_N$, and $e_N$ is its positive right eigenfunction
normalized by $\max e_N=1$. The maximum and positive minimum exist on the
effective input compactification already constructed in the finite-QSD
theorem. Dead coordinates are retained in the physical state; that
compactification is not a replacement of its transition law.

The calibration remains $a_{\rm phys}=t_*h$, with the configured
$t_*,x_*,\hbar_{\rm eff}$ and native recording gaps. Passive source,
color, graph and action records are attached by their original
transition instrument. They do not feed back into this physical kernel.
:::

(sec-nue-killing)=
## 2. Exponentially small sensitivity of the actual extinction event

:::{prf:lemma} Gaussian-score survival sensitivity
:label: lem-nue-killing-score

Choose a finite proof radius $R_J>0$. Set

$$
\begin{gathered}
A=|a_x|,\quad
M_J=A(L_D+\sigma_JR_J)+bV_c,\qquad
a_\dagger=G_d(R_J)[\ell_{L_D,\tau}(M_J)]^d>0,\\
G_4=[d(d+2)]^{1/4},\quad
C_m=A+4bt\nu V_c\ell_\rho,\\
B_x=C_m(L_{Xx}+\sigma_JG_4R_x)
                 +b(1+t\nu)L_{Vx},\\
B_v=C_m(L_{Xv}+\sigma_JG_4R_v)
                 +b(1+t\nu)L_{Vv},\\
\mathcal L_{{\rm kill},N}
=\frac{NG_4}{\tau}(1-a_\dagger)^{N/2}
                             \max\{B_x,B_v/\omega\}.
\end{gathered}
\tag{NUE.3}
$$

Here $\ell_{L,\tau}(M)$ is the one-coordinate Gaussian interval
probability at mean $M$, as in the original survival certificate.
The radius is integrated in the estimate and does not truncate any draw.
For $S,S'\in H_N=\{M(S)\ge a_0N/2\}$ with $a_0N/2\ge2$,

$$
|Q_N1(S)-Q_N1(S')|
\le\mathcal L_{{\rm kill},N}\,
                         \overline d_\omega(S,S').
\tag{NUE.4}
$$

Every source, gate and component outcome is included.
:::

:::{prf:proof}
Freeze a compact preparation $(Y,I,v^c)$ and a second preparation.
Interpolate $Y$, $v^c$ and $I$ linearly. The interpolation has
$Y_\theta\in\overline D$, $|v_\theta^c|\le V_c$ and
$0\le I_\theta\le1$; it is an integration path, not a new algorithm.
Use the same original jitter and write
$X_\theta=Y_\theta+\sigma_JI_\theta Z^J$. The first count velocity
$U_\theta=(I-t\nu L_{X_\theta})v_\theta^c$ is a convex average, so
$|U_{\theta,i}|\le V_c$. The exact terminal position is

$$
Y_i^+=a_xX_{\theta,i}+bU_{\theta,i}+\tau Z_i,
\tag{NUE.5}
$$

where $Z_i$ are independent standard Gaussians independent of all
jitters. This combines the original OU and final-position draws only
for the position event; it does not declare the velocity independent.
On the own-jitter event $|Z_i^J|\le R_J$, every coordinate of the
conditional mean is bounded by $M_J$. Therefore the conditional
row landing probability is at least
$\mathbf1_{\{|Z_i^J|\le R_J\}}
[\ell_{L_D,\tau}(M_J)]^d$. As in the original binomial proof,
independent landing uniforms and independent own jitters give
independent lower success indicators of mean $a_\dagger$.
Consequently the probability $p_\theta$ that every row is dead is at
most $(1-a_\dagger)^N$, uniformly along the entire interpolation.

Differentiate the last Gaussian density. For its all-dead event the
derivative is the expectation of its indicator times
$\tau^{-1}\sum_iZ_i\cdot\dot m_i$, with
$m_i=a_xX_i+bU_i$. This identity follows first with truncated Gaussian
integrals and then by the integrable Gaussian score bound below.
Hölder with exponents $2,4,4$ gives

$$
|\dot p_\theta|
\le\frac{G_4}{\tau}(1-a_\dagger)^{N/2}
                              \sum_i\|\dot m_i\|_4 .
$$

The first count matrix has nonnegative entries. Bounding its diagonal
by one and its off-diagonal entries by $t\nu/N$, and using
$|\nabla K_\rho|\le\ell_\rho$, gives

$$
\sum_i\|\dot U_i\|_4
\le(1+t\nu)\sum_i|\Delta v_i^c|
 +4t\nu V_c\ell_\rho
      \sum_i(|\Delta Y_i|+\sigma_JG_4|\Delta I_i|).
$$

This keeps every unbounded jitter moment and every dependence of
$U_i$ on the other jitters. Hence the bracket in (NUE.3) bounds
the integrated derivative for each paired compact preparation.
Use the actual preparation coupling of
{prf:ref}`lem-native-phase-preparation-coupling`:
its expected average source difference is bounded by (PC.8),
its indicator mismatch by $R_x\delta_x+R_v\delta_v$,
and its collision-velocity difference by (PC.8).
Average over that complete coupling. The relation
$\delta_x+\omega\delta_v=\overline d_\omega$ proves (NUE.4).
:::

(sec-nue-joint-bridge)=
## 3. A full joint two-update smoothing bridge

:::{prf:definition} Primitive joint bridge coefficients
:label: def-nue-joint-budget

Evaluate the algebraic coefficients (PC.4)--(PC.7) with
$m_0=1/N$, keeping all original feature, donor, fitness and landscape
parameters fixed. Put a superscript $[N]$ on those evaluated values and
define

$$
\begin{gathered}
P_{x,N}=2B_{Dx}^{[N]}+2B_{Cx}^{[N]}+R_x^{[N]},\qquad
P_{v,N}=2B_{Dv}^{[N]}+2B_{Cv}^{[N]}+R_v^{[N]},\\
B_N^{\rm src}
=g_1/s+Ng_1/\sigma_J+P_{x,N}+2/(s\sqrt{2\pi}),\\
J_A=\kappa_A^{-1}
 [4t^2\nu\ell_\rho+
                   16t^3\nu V_c/(\kappa_A\rho^2)],\\
K_{v,N}^{\rm pat}
=P_{v,N}+k_cNd[tB_N^{\rm src}/\kappa_A+J_A]
                         +\frac{cg_1k_cN}{\kappa_Aq},\\
A_N^{\rm pat}=(4N^2)^N,\qquad
K_{v,N}=A_N^{\rm pat}K_{v,N}^{\rm pat},\qquad K_m=g_1/s.
\end{gathered}
\tag{NUE.6}
$$

The $m_0=1/N$ values in this definition are derivative overestimates,
not a deterministic alive-floor hypothesis on the transition. Every
quantity is finite. None is a conditional density divided by a pattern
probability.
:::

:::{prf:lemma} Gaussian-refreshed complete input instrument
:label: lem-nue-refreshed-instrument

Let $v=(v_i)$ be any deterministic stored velocity array with
$|v_i|\le V$, and $m=(m_i)$ any finite position-center array. Draw the
original independent positions $x_i=m_i+sZ_i$, classify their actual
incoming marks, and, on nonextinction, perform one complete original
update. Let $\mathcal R_N(m,v)$ be its surviving output submeasure,
with zero contribution when the incoming array is all dead. Then

$$
\|\mathcal R_N(m,v)-\mathcal R_N(m',v')\|_{\rm TV}
\le K_m\sum_i|m_i-m_i'|+K_{v,N}\sum_i|v_i-v_i'|.
\tag{NUE.7}
$$

This is full labelled $(x,v,a)$ total variation, including every
dead row. It is not a one-row or marginal smoothing statement.
:::

:::{prf:proof}
Changing $m$ alone changes the product incoming Gaussian law.
Integrating its directional Gaussian scores bounds its TV by
$K_m\sum_i|\Delta m_i|$. All subsequent classifications, gates and
updates are the same measurable instrument, so data processing gives
this part of (NUE.7), including its zero all-dead contribution.

For the velocity comparison, partition the incoming position integral
into its actual masks, measurement donors, clone donors and gates.
There are at most $A_N^{\rm pat}$ such finite patterns. Integrate
unconsumed/rejected donor choices if desired; retaining them only
increases this bound. For each pattern integrate its normalized original
component Haar variables. At a fixed Haar realization the simultaneous
copied/collided velocity array $w(v)$ obeys

$$
|w_i|\le V_c,\qquad
\|\dot w\|_\infty\le k_c\|\dot v\|_\infty,\qquad
\sum_i|\dot w_i|\le k_cN\sum_i|\dot v_i|.
\tag{NUE.8}
$$

The last bound pays for every possible donor collision. No typical
component-size statement is used here.

On an open cell with a fixed nonempty incoming mask, the joint discrete
pattern probabilities have coordinate derivative bounds $P_{x,N}$ and
$P_{v,N}$. To verify them, couple the two original companion roles by
their common masses, then the actual gates by common uniforms.
The proof of (PC.10)--(PC.11), with at least one alive source, gives
respectively the terms $2B_D$, $2B_C$ and $R$ in (NUE.6).
For at least two alive rows the actual self-excluded denominator is
at least $\kappa_b(M-1)\ge\kappa_bM/2$; its reciprocal is bounded
by the displayed $m_0=1/N$ algebra. With exactly one alive row both
donor roles use their existing unique-source/self convention,
the sole measured standardized scores are constant, the alive gate
is zero and every dead gate is one. Their derivatives vanish.
Clipping and the positive standardizer/power maps are Lipschitz, so
the bounds hold almost everywhere, including all fitness ties.
For each individual pattern its absolute probability derivative
is no larger than this complete pattern-TV derivative.

Let $\mu_v^\pi$ be the subprobability density of its jittered
prepared position array $X$, for this pattern and Haar realization.
It exists by the triangular Gaussian-source argument of
{prf:ref}`lem-cgd-two-update-density`: persistent rows retain
distinct incoming position labels with variance $s^2$, and every
copied/revived row has its own variance $\sigma_J^2$.
For each prepared coordinate its distributional spatial derivative
has variation at most $B_N^{\rm src}$. Here is the complete
integration-by-parts check. A copied coordinate is differentiated
through its own jitter, costing at most $g_1/\sigma_J$.
A persistent coordinate $X_i=x_i$ is differentiated through $x_i$;
every copied recipient with donor $i$ has its jitter translated
in compensation, so its prepared coordinate stays fixed.
The incoming Gaussian score costs $g_1/s$, the at most $N$
compensation scores cost $Ng_1/\sigma_J$, and the original pattern
weight derivative costs $P_{x,N}$. The two faces of the incoming
box classification in that coordinate cost at most
$2/(s\sqrt{2\pi})$, since the incoming one-coordinate Gaussian
density has that uniform face bound. All other coordinates and
all unused incoming positions are integrated with their normalized
original densities. This includes mask-boundary derivative measures,
not just interior derivatives. It yields

$$
\sum_{i,a}|D_{X_{ia}}\mu_v^\pi|(\mathbb R^{Nd})
\le NdB_N^{\rm src},\qquad
\|\dot\mu_v^\pi\|_{\rm var}
\le P_{v,N}\sum_i|\dot v_i|.
\tag{NUE.9}
$$

The second derivative changes the original pattern probability;
its sources and jitters are held fixed. No division by its mass occurs.

Use now the actual first drift at that fixed collided array:

$$
A_w(X)=X+tw+t^2[-\lambda X+\nu C(X,w)],
\qquad
C_i(X,w)=N^{-1}\sum_jK_\rho(X_i,X_j)(w_j-w_i).
\tag{NUE.10}
$$

Its derivative differs from the identity in maximum-row and
sum-row operator norm by at most
$1-\kappa_A=t^2\lambda+4t^2\nu V_c\ell_\rho<1$.
The contraction inverse argument therefore makes $A_w$ a global
$C^2$ diffeomorphism, with inverse derivative norm at most
$\kappa_A^{-1}$ in both norms. This bound is uniform over all
prepared positions, including unbounded jitters.
At a fixed $x_1=A_w(X)$, vary $v$ and put
$H=\dot X=-DA_w^{-1}D_wA_w\dot w$. The count matrix
$D_wA_w=t(I-t\nu L_X)$ is a nonnegative doubly stochastic matrix
times $t$, so

$$
\|H\|_\infty\le(t/\kappa_A)\|\dot w\|_\infty,\quad
\sum_i|H_i|\le(t/\kappa_A)\sum_i|\dot w_i|,\quad
\|D_XH\|_\infty\le J_A\|\dot w\|_\infty.
\tag{NUE.11}
$$

For the last inequality,
$\|D_XD_wA_w[\dot w]\|_\infty
\le4t^2\nu\ell_\rho\|\dot w\|_\infty$,
and $|\operatorname{Hess}K_\rho|\le2/\rho^2$ gives
$\|D_X^2A_w[H]\|_\infty
\le16t^2\nu V_c\|H\|_\infty/\rho^2$.
Differentiate the inverse equation to obtain exactly (NUE.11).
In particular $|\operatorname{div}H|
\le NdJ_A\|\dot w\|_\infty$.

The original OU variable in these coordinates is
$z=cv_1+q\xi$, where $v_1=(x_1-X)/t$.
At fixed $(x_1,z)$ its Gaussian mean derivative is $-cH/t$.
The variation of the joint $(x_1,z)$ density is consequently
bounded by the sum of (i) the parameter derivative of $\mu_v^\pi$,
(ii) its transport derivative $\operatorname{div}(H\mu_v^\pi)$,
and (iii) the OU score. Equations (NUE.8)--(NUE.11) bound these
three terms by, respectively,

$$
P_{v,N}\sum_i|\dot v_i|,\quad
k_cNd[tB_N^{\rm src}/\kappa_A+J_A]\sum_i|\dot v_i|,
\quad
\frac{cg_1k_cN}{\kappa_Aq}\sum_i|\dot v_i|.
$$

The BV product rule, or smooth approximation followed by variation
lower semicontinuity, justifies the same computation for the
classification boundary measures in (NUE.9).

Finally $(x_1,z)\mapsto(x_2=x_1+tz,z)$ and the actual B2 map
$(x_2,z)\mapsto(x_2,(I-t\nu L_{x_2})z-t\lambda x_2)$
are common measurable maps independent of $v$. Apply the original
final-position Gaussian, cap and terminal mark as a common instrument.
None increases variation; no one-step B2 inverse has been assumed.
Sum the unnormalized pattern bounds, integrate their normalized Haar
variables and integrate the velocity interpolation. This proves
the second term of (NUE.7).
:::

:::{prf:theorem} Quantitative full-state two-update TV smoothing
:label: thm-nue-two-update-tv

For $S,S'\in H_N$ with $a_0N/2\ge2$, put

$$
\begin{gathered}
M_x=(|a_x|+4bt\nu V_c\ell_\rho)L_{Xx}+bL_{Vx},\\
M_v=(|a_x|+4bt\nu V_c\ell_\rho)L_{Xv}+bL_{Vv},\qquad
M_*=\max(M_x,M_v/\omega).
\end{gathered}
\tag{NUE.12}
$$

If $q_N\le\bar q$, then

$$
\|Q_N^2(S,\cdot)-Q_N^2(S',\cdot)\|_{\rm TV}
\le L_N\overline d_\omega(S,S'),\qquad
L_N=N[K_mM_*+K_{v,N}\bar q/\omega].
\tag{NUE.13}
$$

In particular $L_N$ is an explicit joint constant; it is not claimed
uniform or polynomial in $N$.
:::

:::{prf:proof}
Use the complete first-update coupling of (NC.5) up to, but not
including, its final position Gaussians. Its conditional final position
centers are $m_i=a_xX_i+bU_i+tq\xi_i$, with the same OU array on
both sides. The first count derivative estimate gives
$\mathbb E N^{-1}\sum_i|\Delta m_i|
\le M_*\overline d_\omega(S,S')$.
For its velocity term, the actual symmetric count matrix
$I-t\nu L_X$ is nonnegative and doubly stochastic, hence contracts
the sum-row norm. At fixed $X$ its first velocity difference costs
$D_V$, rather than the more general population overestimate
$(1+t\nu)D_V$; its position derivative costs
$4t\nu V_c\ell_\rho D_X$. This proves exactly the coefficients
$bL_{Vx},bL_{Vv}$ in (NUE.12).
The same coupling's capped velocity part satisfies
$\mathbb E N^{-1}\sum_i|\Delta v_i^+|
\le(q_N/\omega)\overline d_\omega(S,S')$; the final-position
maximal coupling in (NC.5) changes none of these velocities.
Conditional on these centers and capped velocities, the remaining
first position Gaussian and the entire second killed update are
precisely $\mathcal R_N$. In particular its zero incoming all-dead
branch implements the first killing event. Apply (NUE.7) and
average this actual preceding coupling to prove (NUE.13).
:::

(sec-nue-eigenfunction)=
## 4. Stopped high-alive coupling closes the eigenfunction ratio

:::{prf:definition} Explicit large-population constants and threshold
:label: def-nue-population-threshold

Let $P_{x,1},P_{v,1}$ mean the algebraic expressions (NUE.6)
evaluated at $N=1$, and put

$$
\begin{gathered}
B_1^{\rm src}=g_1/s+g_1/\sigma_J+P_{x,1}+2/(s\sqrt{2\pi}),\\
K_1=P_{v,1}+k_cd[tB_1^{\rm src}/\kappa_A+J_A]
                         +cg_1k_c/(\kappa_Aq),\\
C_L=K_mM_*+K_1\bar q/\omega,\quad
u_0=a_0/8,\quad \ell_q=-\log\bar q,\quad
C_0=\log_+(DC_L),\\
C_k=\frac1{\log2}
 +\frac{C_0/\log2+2+(\log4+u_0)/\log2+6}{\ell_q},\\
C_T=C_k+3/\log2,\qquad C_\theta=2C_k+3/\log2 .
\end{gathered}
\tag{NUE.14}
$$

All are independent of $N$. Define $N_*$ as the ceiling of the maximum
of $2$, $4/a_0$, and the following explicit nonnegative numbers:

$$
\begin{gathered}
\frac4{r_g^2}\log_+\frac{2\sqrt2L_{\rm cap}}{1-r},
\qquad \frac{\log2}{a_0},\\
\frac2{a_0}\log_+\frac{32C_T}{e^2a_0^2\log2},
\qquad
\frac2{u_0}\log_+\frac{256C_\theta}{e^2u_0^2}.
\end{gathered}
\tag{NUE.15}
$$

For $N\ge N_*$ use the actual integer native-update horizon

$$
\begin{gathered}
k_N=\left\lceil
\frac{C_0+N\log(4N^2)+6\log N+u_0N}{\ell_q}
\right\rceil,\qquad T_N=k_N+3,\\
b_N=2C_\theta N\log(N+1)e^{-u_0N}\le1/8 .
\end{gathered}
\tag{NUE.16}
$$

The horizon is an analysis horizon for the existing transition, not
a change to its recording stride or physical clock.
:::

:::{prf:theorem} Population-uniform QSD eigenfunction and asymptotically trivial tilt
:label: thm-nue-uniform-eigenfunction

Under {prf:ref}`def-nue-register`, for every permitted $N\ge N_*$,

$$
1-b_N\le e_N(S)\le1,\qquad
\frac{\max e_N}{\min e_N}\le(1-b_N)^{-1}\le8/7
\tag{NUE.17}
$$

on every actual nonextinct state and its effective compactification.
In particular its oscillation tends to zero at the explicit exponential
rate $O(N\log(N+1)e^{-a_0N/8})$. The conclusion does not assume that
$e_N$ is Lipschitz or that the joint transition mixes uniformly per update.
:::

:::{prf:proof}
The unconditional actual binomial row floor gives, from every
nonextinct input,

$$
Q_N1\ge1-\varepsilon_N,\qquad
P_N^{\rm raw}(H_N^c)\le\zeta_N .
\tag{NUE.18}
$$

The second bound is the elementary binomial lower-tail Chernoff
bound at half its mean, with exponent $a_0N/8$.
Integrating the first bound against the QSD gives
$\alpha_N\ge1-\varepsilon_N$.

Construct a coupling of two actual killed paths. Their cemetery
notation records the original killing event and introduces no restart.
As long as both entering states lie in $H_N$, use (NC.5); after
the first exit use any coupling of their actual remaining marginals.
Starting in $H_N$, the chance of an exit in $k$ updates is at most
$2k\zeta_N$. On the event of no previous exit the expected distance
after $k$ updates is at most $q_N^k$ times its initial distance:
the one-step contraction is conditional on the entering pair, and
discarding an exiting event can only decrease its nonnegative cost.
Apply (NUE.13) to the final two updates. For every measurable
$0\le f\le1$ the result is

$$
|Q_N^{k+2}f(S)-Q_N^{k+2}f(S')|
\le L_Nq_N^k\overline d_\omega(S,S')+2k\zeta_N .
\tag{NUE.19}
$$

The exit term has not been multiplied by $L_N$.
For arbitrary nonextinct starting states use one first actual update.
Both outputs enter $H_N$ except with probability $2\zeta_N$;
their distance there is at most $D$. Thus

$$
\operatorname{osc}(Q_N^{k+3}f)
\le DL_N\bar q^k+2(k+1)\zeta_N .
\tag{NUE.20}
$$

For clarity all growth constants in this smoothing/mixing composition
can be checked explicitly. Each $P_{x,N},P_{v,N}$ is a polynomial
in $N$ with nonnegative coefficients and degree at most four,
by (PC.4)--(PC.7), without the unused component factor $G$.
Hence $B_N^{\rm src}\le N^4B_1^{\rm src}$,
$K_{v,N}^{\rm pat}\le N^5K_1$, and
$L_N\le A_N^{\rm pat}N^6C_L$.
The first threshold in (NUE.15) gives $q_N\le\bar q$.
The choice $k=k_N$ therefore makes the first term of (NUE.20)
at most $\zeta_N$, so its right side is at most
$(2k_N+3)\zeta_N$.

Elementary $\log N\le\log(N+1)$ and
$N\log(N+1)\ge\log2$ give
$k_N\le C_kN\log(N+1)$ and $T_N\le C_TN\log(N+1)$.
For any $u>0$,

$$
N\log(N+1)e^{-uN}\le N^2e^{-uN}
\le\frac{16}{e^2u^2}e^{-uN/2}.
$$

The remaining thresholds in (NUE.15) ensure
$\varepsilon_N\le1/2$,
$2T_N\varepsilon_N\le\log2$, and $b_N\le1/8$.
Consequently $(1-\varepsilon_N)^{-T_N}\le2$.
Apply (NUE.20) to the actual bounded eigenfunction, using
$Q_N^{T_N}e_N=\alpha_N^{T_N}e_N$. It gives
$\operatorname{osc}e_N\le b_N$.
Since its maximum is exactly one, (NUE.17) follows.
All inequalities are uniform at the compactification boundary by the
already proved actual kernel TV continuity.
:::

:::{prf:corollary} A primitive bound covering every admitted finite population
:label: cor-nue-all-populations

For the permitted populations $N<N_*$ use the primitive fixed-$N$
minimum $\underline m_F(N)>0$ of
{prf:ref}`thm-cgd-primitive-eigenfunction` with its prescribed radii.
Then

$$
\sup_{\text{permitted }N}\frac{\max e_N}{\min e_N}
\le
\max\left\{\frac87,\,
        \max_{\substack{\text{permitted }N<N_*}}
                         \underline m_F(N)^{-1}\right\}<\infty .
\tag{NUE.21}
$$

An empty finite maximum contributes one. This is a finite maximum of
primitive formulas; no uncomputed spectral datum occurs.
:::

:::{prf:proof}
The count first-drift margin (NUE.1) is independent of the jitter radii
used in the fixed-$N$ density proof, and the retained QSD tests supply
its other hypotheses. Thus that theorem supplies each indicated
positive primitive minimum. There are finitely many remaining permitted
populations; combine their formulas with (NUE.17).
:::

(sec-nue-doob)=
## 5. The actual stationary and history Doob comparisons

:::{prf:theorem} Uniform stationary, one-update and complete-history comparison
:label: thm-nue-doob-comparison

For $N\ge N_*$ let

$$
\pi_N(dS)=\frac{e_N(S)\nu_N(dS)}{\nu_N(e_N)},\qquad
P_N^e(S,dT)=\frac{Q_N(S,dT)e_N(T)}{\alpha_Ne_N(S)},\qquad
d_N=\frac{b_N}{1-b_N}.
\tag{NUE.22}
$$

These are the existing Doob law and kernel, with no freely chosen
Hamiltonian or new physical projection. Then

$$
\|\pi_N-\nu_N\|_{\rm TV}\le d_N,\qquad
\sup_S\left\|P_N^e(S,\cdot)
-\frac{Q_N(S,\cdot)}{Q_N1(S)}\right\|_{\rm TV}\le d_N .
\tag{NUE.23}
$$

The one-update comparison to the complete raw output adds only
$\varepsilon_N$.
For any native horizon $T\ge0$, attach every actual source/innovation/
color/geometry/action record, and compare the stationary Doob path
started from $\pi_N$ to the original killed path started from $\nu_N$
and conditioned once on survival through $T$. Their total variation
is at most $d_N$, independently of $T$. Relative to the unconditioned
original killed path the bound is

$$
d_N+1-\alpha_N^T\le d_N+T\varepsilon_N .
\tag{NUE.24}
$$

Every original measurable readout and configured calibration inherits
these bounds by data processing.
:::

:::{prf:proof}
If a probability is tilted by a positive function in $[1-b_N,1]$,
its normalized density differs from one by at most $d_N$.
Apply this first to $\nu_N$, and then to each actual one-update
survivor law. The raw output is its mixture with an all-dead
submeasure of mass at most $\varepsilon_N$, proving (NUE.23).

For the complete history let $\mathbb S_{N,T}$ be the original
$\nu_N$-started killed path conditioned on surviving $T$ updates,
with every actual transition record retained. The terminal physical
state has law $\nu_N$ by the QSD equation. Telescoping the actual
Doob factors, including the initial density
$e_N(S_0)/\nu_N(e_N)$, gives the exact identity

$$
\frac{d\mathbb P_{\pi_N}^{e,[0,T]}}{d\mathbb S_{N,T}}
=\frac{e_N(S_T)}{\nu_N(e_N)}.
\tag{NUE.25}
$$

The terminal tilt bound is $d_N$. The original unconditioned
killed path has survival probability $\alpha_N^T$, and conditioning
changes it by TV exactly $1-\alpha_N^T$.
The original attached record kernel is tilted through its actual
terminal state; (NUE.25) does not resample or discard those marks.
The triangle inequality proves (NUE.24).
:::

:::{prf:corollary} Doob stationary concentration and block mixing
:label: cor-nue-doob-concentration-mixing

Any variance bound $\operatorname{Var}_{\nu_N}F\le C/N$ proved
in Chapter 30 transfers to

$$
\operatorname{Var}_{\pi_N}F\le\frac{C}{(1-b_N)N}.
\tag{NUE.26}
$$

The marked chaos and quantitative empirical bounds of Chapter 34,
including {prf:ref}`thm-nqc-stationary-rate`,
transfer with an additional bounded-metric error $Dd_N$.
Furthermore the actual Doob $T_N$-step kernel satisfies

$$
\sup_{S,S'}\|(P_N^e)^{T_N}(S,\cdot)
                  -(P_N^e)^{T_N}(S',\cdot)\|_{\rm TV}
\le2d_N\le2/7 .
\tag{NUE.27}
$$

It therefore contracts signed probability differences by $2d_N$
per $T_N$ native updates. This is a proved population-dependent block
horizon $T_N=O(N\log(N+1))$, not a claimed uniform positive physical
Hamiltonian gap per unit time.
:::

:::{prf:proof}
Use
$\operatorname{Var}_{\pi_N}F
\le\pi_N[(F-\nu_NF)^2]$
and the density upper bound $(1-b_N)^{-1}$.
Transport costs of diameter $D$ change by at most $D$ times TV,
which gives the marked concentration/chaos transfer.
For the last bound the proof of (NUE.20) yields
$\operatorname{osc}(Q_N^{T_N}f)
\le\alpha_N^{T_N}b_N$ for all $0\le f\le1$.
Indeed its raw bound is $(2k_N+3)\zeta_N$,
which is at most $C_\theta N\log(N+1)e^{-u_0N}$;
$\alpha_N^{T_N}\ge1/2$ and (NUE.16) make this no larger than
$\alpha_N^{T_N}b_N$.
For an event $A$, subtract
$(P_N^e)^{T_N}1_A=Q_N^{T_N}(e_N1_A)/
(\alpha_N^{T_N}e_N)$ at two inputs.
The numerator difference costs $b_N/(1-b_N)$ and the
denominator difference costs the same amount. Taking the event
supremum proves (NUE.27). The usual common-mass coupling of
probability rows, integrated against two initial laws, proves its
signed-probability contraction statement.
:::

(sec-nue-scope)=
## 6. Parameter regimes and physical scope

:::{prf:remark} Tested primitive regimes and the remaining physical identification
:label: rem-nue-scope

The positive active witness (PC.37)--(PC.39) has $\sigma_J=0.1>0$,
$t=1/2$, $\lambda=4/(1+e^{-1})$, $\nu=0.01$, $\rho=1$,
$k_c=2$ and its exact configured $V=V_{\rm crit}/2\le V_0=0.1$.
Therefore

$$
\kappa_A=\frac{e^{-1}}{1+e^{-1}}
                  -0.02Ve^{-1/2}
>\frac14-0.002>0,
\tag{NUE.28}
$$

so it passes the new joint bridge test with every original active
fitness power, normalizer, uniform donor tag and component collision
left at its configured value. The bridge constant and population
threshold can be very large because the all-pattern joint estimate is
conservative.
Their displayed formulas still prove a uniform family and an
asymptotically vanishing Doob tilt. A numerical mixing-time claim
for a practical population is not inferred from this certificate.
For orientation, evaluating the primitive formulas with the published
witness diagnostics gives

$$
\begin{gathered}
K_1\simeq356.4944622,\qquad C_L\simeq0.1130003191,\qquad
N_*=255,\\
T_{255}=11259,\qquad b_{255}\simeq2.79912\cdot10^{-6}.
\end{gathered}
\tag{NUE.29}
$$

The exact formulas (NUE.6), (NUE.12), and (NUE.14)--(NUE.16),
rather than the rounded diagnostics, define all inequalities.

The unchanged larger-cap viscous reference remains outside the
proved population-contraction regime, as recorded in (PC.40).
Its finite QSD/eigenfunction formulas remain valid; this theorem
does not silently substitute the small-cap witness for that run.
The present statements compare the actual native physical-state,
record and history laws. They establish neither reflection positivity
of the full color history nor the local Yang--Mills sector
identification or its uniform continuum physical mass gap.
:::
