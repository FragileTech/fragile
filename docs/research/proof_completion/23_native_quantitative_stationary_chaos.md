# Quantitative stationary chaos and commuting time limits for the native count gas

(sec-nqc-register)=
## 1. The complete contraction regime and compact preparation

:::{prf:definition} Quantitative native count register
:label: def-nqc-register

Retain every execution, landscape, donor, measurement, fitness, cloning,
component-Haar, jitter, kinetic, boundary, arithmetic and passive recording
parameter of {prf:ref}`def-native-concentration-register`. Thus the
transition is the existing quadratic count gas, with both actual viscous
kicks, sampled global normalization, mandatory current-donor revival,
simultaneous copying, a shared Haar rotation on each complete accepted
component, unbounded original Gaussian innovations, its configured cap and
terminal alive/dead marks. No history, feedback, elite or curl channel is
inserted. The positive parameter tests of that register, including
$0<V\le\min\{V_0,1\}$, $\nu\le\nu_g$, and the proved population
contraction coefficient $r<1$, remain in force. The actual row metric is
$d_\omega$, of diameter $D=2+2\omega V$.

Write $\Pi(\mu)$ for the complete population preparation stopped just
before its original recipient jitter. Its physical variables are
$W=(Y,I,v^c)$, with

$$
Y\in\overline D,\qquad I\in\{0,1\},\qquad |v^c|\le V_c=k_cV.
$$

Here $Y$ is the actual frozen own/donor position and $I$ the indicator
that the original jitter is applied. The actual finite preparation is
$\Theta_N=N^{-1}\sum_i\delta_{W_i}$. Its shared component rotations
and correlations are retained. On this compact preparation space put

$$
d_J(W,W')=|Y-Y'|+|v^c-v^{c\prime}|+|I-I'|,
\qquad D_J=2R_D+2V_c+1.
\tag{NQC.1}
$$

The map $\mathcal K_J$ first applies $X=Y+\sigma_J I Z^J$ and then
both population count kicks and all remaining original stages. Thus
$\mathcal F=\mathcal K_J\circ\Pi$. Every comparison below concerns
this complete map at the actual empirical input, rather than a prescribed
independent replacement input.
:::

:::{prf:definition} Preparation bias and variance constants
:label: def-nqc-preparation-budget

Use $m_0,C,S_b,H_s,L_g,A_D,M_2$ of (PC.2), (PC.4), (PC.6),
(PC.7) and (NC.12). Define

$$
\begin{gathered}
D_D=2/(\kappa_Dm_0),\qquad
A_T=(m_0^{-1}+D_D^2)
       [2S_b^2/\sigma_s^2+5S_b^6/\sigma_s^6],\qquad
L_T=2L_gH_s,\\
A=1+C+D_D,\qquad N_b=\lceil(8A)^{6/5}\rceil,\\
M_1(u)=e^{2u},\qquad
M_3(u)=(1+6u+3u^2)e^{8u},\\
B_J=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
       +64A^2+16M_3(C)+\sqrt{N_b},\\
A_{\rm prep}=2[A_D+10M_2(C)].
\end{gathered}
\tag{NQC.2}
$$

These are the original component and normalization constants, evaluated
at the same uniform positive alive floor. Shifting the diversity by
$\delta_D$ gives its exact range $[0,S_b]$ without changing its
standardized value. Consequently the displayed shifted constants are
valid for the actual unshifted implementation. Every constant is finite
and independent of $N$, although it may be large.
:::

(sec-nqc-preparation)=
## 2. Quantitative transport of the correlated preparation

:::{prf:lemma} Actual compact preparation bias and concentration
:label: lem-nqc-preparation

For a deterministic entering swarm $S\in H_N$ with $m_0N\ge2$,
write $\mu_N=L_N(S)$. For every bounded measurable preparation test
$f$, the actual preparation satisfies

$$
\left|\mathbb E\Theta_N f-\Pi(\mu_N)f\right|
 \le 2\|f\|_\infty B_J/\sqrt N,
\qquad
\operatorname{Var}(\Theta_N f\mid S)
 \le A_{\rm prep}\|f\|_\infty^2/N.
\tag{NQC.3}
$$

Fix a proof moment order $p\ge2$ and set $a_p=(2+2d+d/p)^{-1}$. Put

$$
J_J=2[(1+4\sqrt d R_D)(1+4\sqrt d V_c)]^d,
\quad
C_J=2+\frac{D_J\sqrt{J_JA_{\rm prep}}}2+D_JB_J.
\tag{NQC.4}
$$

Then, without independent component outputs,

$$
\mathbb E W_{d_J}(\Theta_N,\Pi(\mu_N))\le C_JN^{-a_p}.
\tag{NQC.5}
$$
:::

:::{prf:proof}
The normalization and marked-exploration proof of
{prf:ref}`thm-chaos-canonical-quantitative-bias` applies when stopped
before jitter. Its proof retains every rejected proposal's auxiliary
target and measurement mark, incoming children with their already fixed
outgoing choice, self exclusion and the entire component Haar variable.
Replacing only the empirical diversity normalizers by their population
values is an integration coupling, with $\mathbb ET^2\le A_T/N$.
The affected fraction is at most
$3M_1(2C)L_T\mathbb ET+4L_T^2A_T/N$.
The remaining finite exploration, stopped at
$K=\lfloor N^{1/6}\rfloor$, has mismatch at most
$64A^2K^3/N+2M_3(C)/K^3$ when $N\ge N_b$; the elementary
bound one covers smaller $N$. These are precisely (NQC.2)'s bias terms.
On matching explorations the stopped variables $(Y,I,v^c)$ agree
exactly. Hence their mean probabilities differ in TV by at most
$B_J/\sqrt N$, proving the first inequality.

For the variance, expose the $N$ original measurement blocks, $N$
donor/gate blocks and $N$ addressed component-Haar blocks. The
pre-kinetic squared-influence proof in (NC.12) gives respectively
$A_D$, $9M_2(C)$ and $M_2(C)$ as expected squared affected-row counts.
The stopped test changes by at most $2\|f\|_\infty D_i/N$ under
such a replacement. The original independent-block resampling inequality
therefore gives the second inequality in (NQC.3). No dense kinetic
stage is included in this stopped influence argument.

More precisely, for any finite measurable partition the empirical cell
mass vector changes by $\ell^1$ norm at most $2D_i/N$ under that same
replacement. Its squared $\ell^2$ change is therefore at most
$4D_i^2/N^2$. Apply resampling to every cell and sum before bounding.
The sum of all cell-mass variances is at most $A_{\rm prep}/N$,
independently of the number of cells.

For $0<\epsilon\le1$, partition the two bounded Euclidean coordinates
into boxes of diameters at most $\epsilon/2$, retaining $I$ exactly.
Choose one representative in each nonempty box. This gives a quantizer
of $d_J$ error at most $\epsilon$, with at most

$$
J(\epsilon)\le
2(1+4\sqrt d R_D/\epsilon)^d
 (1+4\sqrt d V_c/\epsilon)^d
\le J_J\epsilon^{-2d}
$$

cells. Cauchy--Schwarz and the summed variance estimate give total
expected absolute fluctuation at most
$\sqrt{J(\epsilon)A_{\rm prep}/N}$.
The mean law differs from $\Pi(\mu_N)$ in TV by at most
$B_J/\sqrt N$. Transport on this finite space costs at most its
diameter times TV. The two quantization couplings consequently give

$$
\mathbb E W_{d_J}(\Theta_N,\Pi(\mu_N))
\le2\epsilon+\frac{D_J}2\sqrt{J(\epsilon)A_{\rm prep}/N}
                 +D_JB_J/\sqrt N.
$$

Choose $\epsilon=N^{-a_p}$. Since $a_p\le(2d+2)^{-1}$,
$da_p-1/2\le-a_p$ and $a_p\le1/2$ prove (NQC.5). This finite
quantization uses bounded measurable tests, so no regularity across
component, gate or status changes is presumed.
:::

(sec-nqc-kinetic)=
## 3. The complete dense kinetic comparison

:::{prf:definition} Primitive kinetic transport coefficients
:label: def-nqc-kinetic-budget

Let $g_1=\mathbb E|Z_d|$, $\zeta_s=2/(\sqrt{2\pi}s)$ and
$\ell_\rho=e^{-1/2}/\rho$. Let
$g_{d,p}=(\mathbb E|Z_d|^p)^{1/p}$ for the stated proof order. Put

$$
\begin{gathered}
R_J=R_D+\sigma_J(g_1+d/g_1),\qquad
Z_J=cV_c+ct\lambda R_J+qg_1,\qquad
\theta=4t\nu V_c\ell_\rho,\\
A_U=|D_c|+2t\nu c+4t\nu\ell_\rho bZ_J,\\
A_X=t\lambda|c+a_x|+2t\nu ct\lambda
                         +4t\nu\ell_\rho|a_x|Z_J,\\
K_x=\zeta_s(|a_x|+b\theta)+\omega(A_U\theta+A_X),\qquad
K_v=(1+t\nu)(\zeta_sb+\omega A_U),\\
L_J=\max\{K_x\max(1,\sigma_Jg_1),K_v\}.
\end{gathered}
\tag{NQC.6}
$$

For the finite/reference kinetic comparison define

$$
\begin{gathered}
Z_2=cV_c+ct\lambda(R_D+\sigma_J\sqrt d)+q\sqrt d,\\
A_{U,2}=|D_c|+2t\nu c+4t\nu\ell_\rho bZ_2,\qquad
E_1=2t\nu V_c,\\
C_{\rm kin}=\zeta_sbE_1
             +\omega[A_{U,2}E_1+2\sqrt2t\nu Z_2],\\
H_p=[|a_x|(R_D+\sigma_Jg_{d,p})+bV_c+\tau g_{d,p}]^p,\\
J_O=2[(1+4\sqrt d)(1+4\sqrt d\omega V)]^d,\\
C_O=2+2DH_p+(D/2)\sqrt{J_O+1},\qquad
C_{\rm upd}=L_JC_J+C_{\rm kin}+C_O.
\end{gathered}
\tag{NQC.7}
$$

All coefficients use the actual fixed cap, both force evaluations and
original Gaussian amplitudes. These estimates do not require any
curvature of an unknown stationary law.
:::

:::{prf:lemma} Population kinetic continuity on the compact source record
:label: lem-nqc-kinetic-lipschitz

For any two probabilities $\Theta,\Theta'$ on the preparation space,

$$
W_\omega(\mathcal K_J\Theta,\mathcal K_J\Theta')
 \le L_J W_{d_J}(\Theta,\Theta').
\tag{NQC.8}
$$
:::

:::{prf:proof}
Couple their compact source variables and use the same original recipient
jitter. Denote
$D_X=\mathbb E|Y-Y'|+\sigma_Jg_1\Pr(I\ne I')$ and
$D_V=\mathbb E|v^c-v^{c\prime}|$. Interpolate the paired source
variables linearly; their positions and velocities remain bounded and
their jitter coefficients remain in $[0,1]$. Use the same OU innovations.
Writing $U=v^c+t\nu C_{\lambda_J}(X,v^c)$, the first count
kick is a convex average because $t\nu\le1/2$. Thus $|U|\le V_c$.
Differentiating its own and source variables, with
$|\nabla K_\rho|\le\ell_\rho$, gives

$$
\mathbb E|\dot U|\le B=(1+t\nu)D_V+\theta D_X.
$$

The original jitter is independent of the compact source pair. In its
own-row products, $\mathbb E|Z^J|^2=d$ bounds
$\mathbb E|X||\dot X|\le R_JD_X$. The same derivative formula
then gives $\mathbb E|X||\dot U|\le R_JB$.
These bounds also control the products with an independent source row.
Consequently, for the actual A2/OU coordinates
$y=a_xX+bU+tq\xi$ and $z=cU-ct\lambda X+q\xi$,

$$
\mathbb E|z||\dot X|\le Z_JD_X,
\qquad\mathbb E|z||\dot U|\le Z_JB.
$$

Differentiate the complete second count kick
$u=z-t\lambda y+t\nu C_{\lambda_2}(y,z)$, including the
source-law derivative at this actual stage. Its velocity-difference
terms contribute $2t\nu(cB+ct\lambda D_X)$; its kernel
derivatives contribute
$4t\nu\ell_\rho Z_J(bB+|a_x|D_X)$.
The linear term contributes $|D_c|B+t\lambda|c+a_x|D_X$.
Thus $\mathbb E|\dot u|\le A_UB+A_XD_X$.
The configured cap is 1-Lipschitz. Couple the original final-position
Gaussians by their common density conditional on the paired preceding
coordinates. Their full marked position cost is at most
$\zeta_s\mathbb E|\dot y|\le\zeta_s(|a_x|D_X+bB)$.
Integration along the interpolation proves
$W_\omega\le K_xD_X+K_vD_V$, and (NQC.6) converts this to
(NQC.8). The bound was obtained before averaging correlated jitter
products; no conditional bound over arbitrary post-jitter arrays is used.
:::

:::{prf:lemma} Conditional finite count kinetics with independent reference rows
:label: lem-nqc-kinetic-consistency

Freeze any realized complete compact preparation $(W_i)_{i=1}^N$.
For its actual interacting raw output $L_N^+$,

$$
\mathbb E[W_\omega(L_N^+,\mathcal K_J\Theta_N)
                  \mid(W_i)_i]
\le C_{\rm kin}/\sqrt N+C_ON^{-a_p}.
\tag{NQC.9}
$$

The actual complete output rows need not be independent.
:::

:::{prf:proof}
Draw every original jitter and OU row independently under this
conditioning. Let $\lambda_J$ be the deterministic mixture obtained
by jittering $\Theta_N$. The reference first-kick row uses
$U_i^0=v_i^c+t\nu C_{\lambda_J}(X_i,v_i^c)$, whereas the
actual row uses $U_i=v_i^c+t\nu C_{L_N^J}(X_i,v_i^c)$.
Conditioning also on row $i$'s jitter, the remaining terms are independent.
Each kernel-weighted velocity difference is bounded by $2V_c$.
The own-type contribution to the mixture is zero, since that type's
velocity is exactly $v_i^c$. Hence

$$
\frac1N\sum_i\mathbb E|U_i-U_i^0|^2\le E_1^2/N.
\tag{NQC.10}
$$

Both $U_i$ and $U_i^0$ are bounded by $V_c$. The reference A2/OU
rows $(y_i^0,z_i^0)$ are independent under the frozen source record,
with mixture law exactly $\lambda_2$ of $\mathcal K_J\Theta_N$.
Each has velocity second moment at most $Z_2^2$.
First compare the actual second kick with the finite second kick on
this reference array. Interpolate $U_i$ to $U_i^0$, keeping all
original $X_i,\xi_i$ fixed. Then $\dot y_i=b\dot U_i$ and
$\dot z_i=c\dot U_i$. The summed kernel derivative is bounded
by $4t\nu\ell_\rho b$ times the product of the RMS change
and the RMS intermediate velocity. Cauchy--Schwarz and (NQC.10)
therefore give an average expected uncapped velocity change at most
$A_{U,2}E_1/\sqrt N$. This retains the correlation of $U_i-U_i^0$
with all original jitters.

Next compare the finite reference B2 force with
$\nu C_{\lambda_2}(y_i^0,z_i^0)$. Conditional on its own reference
row, the other rows are independent but may have different source types.
Their centered squared error, averaged over the root, is at most
$4\nu^2Z_2^2/N$. The missing own-type mixture contributes squared
bias at most $4\nu^2Z_2^2/N^2$. The cross term vanishes under that
conditioning. The average $L^2$ error is thus at most
$2\sqrt2\nu Z_2/\sqrt N$. Multiplication by $t$ gives the final
term of $C_{\rm kin}$.

Use the same cap and couple each original final-position Gaussian by
common densities, conditional on these paired preceding arrays. The
reference terminal rows have their independently specified original
Gaussian laws; the coupling gives the actual terminal marginals without
altering them. Their average marked-position discrepancy is at most
$\zeta_sbE_1/\sqrt N$. Matching labelled rows gives a transport
coupling of the two empirical laws of total expected cost at most
$C_{\rm kin}/\sqrt N$.

It remains to compare the independent, nonidentically distributed
reference rows with their mixture $\mathcal K_J\Theta_N$.
For $R\ge1$ and $0<\epsilon\le1$, quantize $|x|\le R$ and
$|v|\le V$ with $d_\omega$ error at most $\epsilon$, retaining
the terminal mark exactly, and collapse its spatial tail to one atom.
There are at most

$$
J_O(R,\epsilon)=
2(1+4\sqrt d R/\epsilon)^d
 (1+4\sqrt d\omega V/\epsilon)^d
$$

bounded cells. The exact position formula and Minkowski give the
conditional $p$th moment bound $H_p$ for every reference row.
Both quantization errors are therefore at most
$\epsilon+DH_p/R^p$ in expectation.
Independent row cell counts have variance at most their mean mass
divided by $N$. Cauchy--Schwarz over the $J_O(R,\epsilon)+1$ cells
bounds their expected absolute mass fluctuation by
$\sqrt{(J_O(R,\epsilon)+1)/N}$. Consequently the additional
transport cost is at most

$$
2\epsilon+2DH_p/R^p
       +(D/2)\sqrt{(J_O(R,\epsilon)+1)/N}.
$$

Choose $\epsilon=N^{-a_p}$ and $R=N^{a_p/p}$.
Then $J_O(R,\epsilon)\le J_ON^{da_p(2+1/p)}$ and
$-1/2+(d/2)a_p(2+1/p)=-a_p$ by its explicit definition. This proves
(NQC.9). Independence has been used only for the original fresh rows
after freezing preparation and for these constructed reference rows;
the actual dense-kick output remains interacting.
:::

:::{prf:theorem} Explicit complete one-step empirical consistency
:label: thm-nqc-one-step

For $S\in H_N$ and $m_0N\ge2$, the actual raw complete output satisfies

$$
\mathbb E_S W_\omega(L_N^+,\mathcal F(L_N(S)))
 \le C_{\rm upd}N^{-a_p}.
\tag{NQC.11}
$$
:::

:::{prf:proof}
Condition on the entire actual compact preparation and apply (NQC.9).
Then compare its kinetic mixture with
$\mathcal K_J\Pi(L_N(S))$ using (NQC.8) and (NQC.5).
The triangle inequality and $N^{-1/2}\le N^{-a_p}$ give (NQC.11).
All three terms in $C_{\rm upd}$ have been derived from the original
normalization, component, both-kick and Gaussian records.
:::

(sec-nqc-stationary)=
## 4. Quantitative full marked stationary chaos

:::{prf:theorem} Primitive stationary empirical rate
:label: thm-nqc-stationary-rate

Let $\mu_*$ be the uniquely proved population fixed point in this
regime. Write $c_*=(1-\log2)/2$,
$\delta_N=e^{-c_*a_0N}$ and $e_N=(1-a_0)^N$. For the actual full
marked QSD and $m_0N\ge2$,

$$
\mathbb E_{\nu_N}W_\omega(L_N,\mu_*)
\le R_N:=\frac{C_{\rm upd}N^{-a_p}
                         +D(\delta_N+e_N)}{1-r}.
\tag{NQC.12}
$$

For smaller permitted populations the diameter bound $D$ applies.
Every constant in the rate is an explicit function of the complete
algorithm/landscape register. The same original positive witness
of (PC.37)--(PC.39), with its additional (NC.1) check, is included.
No deterministic alive floor is imposed on the finite gas.
:::

:::{prf:proof}
Let $X_N=\mathbb E_{\nu_N}W_\omega(L_N,\mu_*)$.
On the actual event $H_N$, one-step consistency and the proved
population contraction give

$$
\mathbb E_S W_\omega(L_N^+,\mu_*)
\le C_{\rm upd}N^{-a_p}+rW_\omega(L_N(S),\mu_*).
$$

On $H_N^c$ use the metric diameter $D$. The previously proved QSD
binomial estimate gives $\nu_N(H_N^c)\le\delta_N$.
Its exact raw selected-law identity is
$\nu_NP_N^{\rm raw}=\alpha_N\nu_N+(1-\alpha_N)\zeta_N^{\dagger}$,
with $1-\alpha_N\le e_N$. Thus the raw output's expected bounded
distance differs from $X_N$ by at most $De_N$. Integrating the last
display gives

$$
X_N\le C_{\rm upd}N^{-a_p}+rX_N+D\delta_N+De_N.
$$

Division by $1-r$ proves (NQC.12). This uses the raw QSD identity,
not a Doob invariant law or a presumed stationary entropy inequality.
:::

:::{prf:corollary} Quantitative fixed-label marked chaos
:label: cor-nqc-fixed-label-chaos

For fixed $k\le N$, use the mean product metric
$d_{\omega,k}=k^{-1}\sum_{j=1}^kd_\omega$ on $k$ actual marked
rows. Then

$$
W_{d_{\omega,k}}(\nu_N^{(k)},\mu_*^{\otimes k})
 \le R_N+Dk(k-1)/(2N).
\tag{NQC.13}
$$

This compares the original labelled coordinates, including dead
coordinates and statuses. It is a transport estimate; no TV bound to
the continuous product law is inferred from it.
:::

:::{prf:proof}
The actual complete kernel is permutation equivariant, and uniqueness
of its QSD gives exchangeability. Conditional on its empirical multiset,
the $k$ labels are sampled without replacement. Sampling with replacement
has collision probability at most $k(k-1)/(2N)$, so the two conditional
laws have TV distance at most that number. Their metric diameter is $D$.
For the with-replacement law, use an optimal one-row coupling between
$L_N$ and $\mu_*$, independently for the $k$ samples. Its mean product
cost is $W_\omega(L_N,\mu_*)$. Integrate (NQC.12) and apply the
triangle inequality to prove (NQC.13).
:::

:::{prf:corollary} Quantitative full marked quadratic transport
:label: cor-nqc-quadratic-transport

For $p>2$ use the actual full marked squared row metric

$$
d_2(z,z')^2=|x-x'|^2+|v-v'|^2+|a-a'|^2,
\qquad
\mathsf W_2(\mu,\zeta)^2=\inf_{\pi\in\Gamma(\mu,\zeta)}
                                      \int d_2^2\,d\pi .
$$

Put $K_p=[2^{p-1}H_p(1+a_0^{-1})]^{2/p}$, with the original
Gaussian $H_p$ in (NQC.7). Then the actual full marked stationary
empirical law satisfies

$$
\mathbb E_{\nu_N}\mathsf W_2(L_N,\mu_*)^2
\le (2+2V/\omega)R_N+K_pR_N^{1-2/p}.
\tag{NQC.19}
$$

In particular its expected $\mathsf W_2$ is bounded by the square root
of this explicit expression. Its algebraic rate is
$N^{-a_p(p-2)/(2p)}$; for $p=4,d=3$ it is $N^{-1/35}$,
with the same retained survival and high-alive defects. Positions,
velocities and statuses all occur in this quadratic transport; the
large spatial tail has not been clipped or omitted.
:::

:::{prf:proof}
Let $\mu$ be a realized empirical law, and choose a coupling to
$\mu_*$ whose $d_\omega$ cost is arbitrarily close to
$\Delta=W_\omega(\mu,\mu_*)$. The position contribution where
$|x-x'|\le1$ is at most that bounded position cost. The status
contribution is at most its actual mark cost. Together they are at
most $2\Delta$ in the limit of optimal couplings.
Since both velocities obey the configured cap,

$$
\int |v-v'|^2\,d\pi\le2V\int|v-v'|\,d\pi
                                      \le(2V/\omega)\Delta.
$$

The remaining spatial event has probability at most $\Delta$.
Hölder on that same coupling gives

$$
\int |x-x'|^2\mathbf1_{\{|x-x'|>1\}}\,d\pi
\le [2^{p-1}(\mu|x|^p+\mu_*|x|^p)]^{2/p}
                                           \Delta^{1-2/p}.
$$

Every original raw output has position $p$-moment at most $H_p$:
its source is in the box, $|U|\le V_c$, and its two original
position Gaussians combine with amplitude $\tau$. The raw QSD
identity therefore gives
$\mathbb E_{\nu_N}L_N|x|^p\le H_p/\alpha_N\le H_p/a_0$.
The actual population fixed point is an output of the same map and
satisfies $\mu_*|x|^p\le H_p$.
Apply Hölder again, now over the actual QSD, with conjugate exponents
$p/2$ and $p/(p-2)$. The averaged spatial tail term is at most

$$
[2^{p-1}H_p(1+a_0^{-1})]^{2/p}
       (\mathbb E_{\nu_N}\Delta)^{1-2/p}.
$$

The displayed candidate coupling bounds the infimum defining
$\mathsf W_2^2$. Insert (NQC.12) and let the coupling approximation
error vanish to prove (NQC.19). Jensen bounds the expected
$\mathsf W_2$ by its RMS. The stated exponent follows by taking its
square root; its defects remain the full explicit $R_N$ expression.
:::

(sec-nqc-time)=
## 5. Survivor evolution, finite-population contraction and commuting time limits

:::{prf:theorem} Uniform quantitative survivor mean-field evolution
:label: thm-nqc-survivor-evolution

Start the existing killed gas from any actual capped, terminally
consistent nonextinct law $\eta_{N,0}$, and let
$\eta_{N,n}=\eta_{N,0}Q_N^n/(\eta_{N,0}Q_N^n1)$.
For a deterministic admitted population trajectory
$\mu_{n+1}=\mathcal F(\mu_n)$, write
$X_n=\mathbb E_{\eta_{N,n}}W_\omega(L_N,\mu_n)$.
For $m_0N\ge2$ and every $n\ge1$,

$$
X_{n+1}\le rX_n+B_N,\qquad
B_N=C_{\rm upd}N^{-a_p}+D(\delta_N+e_N),
$$
$$
X_n\le r^{n-1}X_1+\frac{1-r^{n-1}}{1-r}B_N.
\tag{NQC.14}
$$

Every $\mu_n$ has alive mass at least $a_0$ for $n\ge1$.
If the actual initial law is supported on $H_N$, the same recurrence
also applies at $n=0$, with its $D\delta_N$ term omitted there.
In particular taking the already proved fixed point trajectory gives
a time-uniform empirical attraction estimate to $\mu_*$.
:::

:::{prf:proof}
The original binomial landing certificate gives
$\eta_{N,n}(H_N^c)\le\delta_N$ for every $n\ge1$.
This is the current-time survivor estimate, with no inverse probability
of surviving from time zero. The same certificate bounds the one-step
extinction probability $e_n=\eta_{N,n}P_N^{\rm raw}(M^+=0)$ by
$e_N$. Its exact identity is

$$
\eta_{N,n}P_N^{\rm raw}
 =(1-e_n)\eta_{N,n+1}+e_n\zeta_{N,n}^{\dagger}.
\tag{NQC.15}
$$

Consequently replacing the raw output by the actual next survivor law
changes a bounded distance expectation by at most $De_N$.
On $H_N$, (NQC.11) and population contraction give
$\mathbb E_S W_\omega(L_N^+,\mu_{n+1})
\le C_{\rm upd}N^{-a_p}+rW_\omega(L_N(S),\mu_n)$.
On its complement the distance is at most $D$.
Integration and (NQC.15) give the recurrence. Induction sums its
geometric series. Population positivity and preservation of the actual
terminal output class were already proved before its contraction, so
all $\mu_n$ used in this recurrence are admitted.
:::

:::{prf:theorem} Actual high-alive finite-population relaxation and joint time/population limits
:label: thm-nqc-finite-relaxation

Let $q_N=r+\sqrt2e^{-Nr_g^2/4}L_{\rm cap}$ be the derived complete
raw-kernel coefficient in (NC.5), and retain $\overline d_\omega$
on complete labelled swarms. For $m_0N\ge2$ and $q_N<1$,

$$
W_{\overline d_\omega}(\eta_{N,n},\nu_N)
\le Dq_N^{n-1}
       +\frac{2D(\delta_N+e_N)}{1-q_N},\qquad n\ge1.
\tag{NQC.16}
$$

It follows for the same actual survivor laws that

$$
\mathbb E_{\eta_{N,n}}W_\omega(L_N,\mu_*)
\le R_N+Dq_N^{n-1}
             +\frac{2D(\delta_N+e_N)}{1-q_N}.
\tag{NQC.17}
$$

For $N$ above the explicit (NC.18) cap threshold and $2/m_0$,
$q_N\le(1+r)/2<1$. Thus every joint sequence
$N\to\infty$, $n_N\to\infty$ has
$\mathbb E_{\eta_{N,n_N}}W_\omega(L_N,\mu_*)\to0$,
and the two iterated time/population limits agree with this same
stationary population law. The result concerns fixed configured native
timestep and the actual calibration $a_{\rm phys}=t_*h$; it does
not exchange this limit with a vanishing-timestep or different physical
reconstruction limit.
:::

:::{prf:proof}
Couple $\eta_{N,n}$ and $\nu_N$ with entering mean row cost arbitrarily
close to their transport distance. For a pair in $H_N\times H_N$,
apply the exact full marked finite-population coupling (NC.5).
For any other pair use the diameter $D$. Both actual entering laws
have $H_N^c$ probability at most $\delta_N$ for $n\ge1$.
The resulting raw-output transport cost is consequently at most

$$
q_NW_{\overline d_\omega}(\eta_{N,n},\nu_N)+2D\delta_N.
$$

Their respective survivor identities (NQC.15) and (NC.17) each cost
at most $De_N$ to replace the raw laws by $\eta_{N,n+1}$ and
$\nu_N$. Hence their distance satisfies the affine recurrence with
coefficient $q_N$ and defect $2D(\delta_N+e_N)$.
The first-time distance is at most $D$; geometric summation proves
(NQC.16).

Matching labelled rows gives
$W_\omega(L_N(S),L_N(S'))\le\overline d_\omega(S,S')$.
Therefore $S\mapsto W_\omega(L_N(S),\mu_*)$ is 1-Lipschitz,
and (NQC.12), (NQC.16) prove (NQC.17).
The explicit cap threshold makes $q_N\le(1+r)/2$; its exponential
defects and $R_N$ vanish independently of time. This proves the joint
limit and both orders using their displayed limsup bounds. At each
fixed $N$, the inherited finite-population QSD convergence also
identifies the exact time limit with $\nu_N$. No whole-horizon
conditioning error proportional to $n$ is needed.
:::

(sec-nqc-scope)=
## 6. Evaluated positive regime and residual scopes

:::{prf:remark} Rate and parameter scope
:label: rem-nqc-rate-scope

The strictly positive original witness (PC.37) satisfies the additional
(NC.1) condition with $r_g=1$, as explicitly checked in Chapter NC.
It therefore realizes every theorem above with $d=3$, $p=4$ and empirical
rate $N^{-4/35}$. Its cap is the exact positive $V_{\rm crit}/2$;
the cap is fixed at that value for every population. Its donor tag is
the existing uniform kernel, and its actual positive viscosity,
sampled exponents, noises, collision coefficient and quadratic provider
retain their complete original registers. The preparation constants
$M_3(C)$ and $A_D$ are large but finite. The rate is an analytic
upper bound rather than a practical population-size prediction.

For the same exact witness, using $p=4$ and the actual weight of
(PC.39), the coefficient formulas have the diagnostics

$$
\begin{gathered}
B_J\simeq7.933849338\,10^{20},\qquad
A_{\rm prep}\simeq7.350196602\,10^{21},\\
C_J\simeq6.290117055\,10^{21},\qquad
L_J\simeq1586.431072,\qquad Z_2\simeq3.095302660,\\
C_{\rm kin}\simeq104.420001,\qquad
H_4\simeq18.41772200,\qquad C_O\simeq107.2569091,\\
C_{\rm upd}\simeq9.978837144\,10^{24},\qquad
\log C_{\rm upd}\simeq57.56250880.
\end{gathered}
\tag{NQC.18}
$$

The certified $r\le1/2$ consequently bounds the stationary particle
coefficient by $2C_{\rm upd}$, with logarithm about $58.25565598$.
The exact positive witness and formulas define these constants;
the rounded values serve only to report their scale. The complete
alive and extinction defects remain the explicit (NQC.12) terms.

The unchanged large-cap, nonreset reference fails the sufficient
population-contraction test in (PC.40). This chapter adds no phase
selection conclusion for that reference. A failed sufficient test is
not an actual failure of chaos. Other landscape and algorithm tags,
row normalization, history or geometry feedback, and alternate
arithmetic retain their own separately proved laws. The new rates do
not establish a central limit theorem, gradient entropy production,
a full stationary Doob inequality, or a local continuum gauge-field
Hamiltonian. They do close quantitative stationary marked chaos and
commuting native time/population limits in the stated existing positive
count regime.
:::
