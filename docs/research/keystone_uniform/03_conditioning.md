# Quantitative survivor conditioning for the unchanged coupled gas

(sec-ku-conditioning-record)=
## 1. Complete law and the position information actually used

:::{prf:definition} Kernel, preparation and coefficients
:label: def-ku-conditioning-record

Use the complete record $\Theta$ in
{prf:ref}`def-cgd-parameter-register`. All companion conventions, sampled
measurement, reward and diversity standardization, clone acceptance,
mandatory revival, connected-component Haar collision, clone jitter,
both viscous B stages, OU noise, final position noise, cap and terminal
marking remain those of that record. Historical donors and geometry
feedback are absent as specified there; a passive readout is retained
as a measurable output.

Let $P_N(S,\cdot)$ retain the complete physical output including an
all-dead output, $E_N=\{\sum_i a_i>0\}$, and
$Q_N(S,A)=P_N(S,A\cap E_N)$. Write

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
\eta=bt,\quad \sigma_h^2=t^2q^2+s^2>0,
\quad V_c=(1+2|\alpha_{\rm col}|)V,\quad
\kappa_\nu=\max\{1,2t\nu-1\}.
$$

Here $q,s$ are the actual Gaussian amplitudes in
{prf:ref}`def-cgd-parameter-register`. Assume
$D=[-L_D,L_D]^d$, $L_D>0$, and set $R_D=\sqrt d L_D$.
Freeze the entire measurement/companion/gate pattern and its actual
component rotations before drawing the independent clone jitters.
For each row the frozen source satisfies

$$
Y_i\in\overline D,\qquad I_i\in\{0,1\},\qquad
X_i=Y_i+I_i\sigma_J Z_i^J,\qquad |v_i^{\rm col}|\le V_c.
\tag{KU.S1}
$$

The source and gate arrays need not have independent rows. Conditional
on this frozen information, $(Z_i^J)_i$ are independent standard
Gaussians. Unused latent $Z_i^J$ may be added when $I_i=0$ without
changing any output. Conditional on the subsequent complete preparation
$(X,v^{\rm col})$, the actual final position law is

$$
x_i^+=M_i+\sigma_h Z_i,\qquad
M_i=X_i+\eta F(X_i)+bU_i,\qquad
U=W_Xv^{\rm col},\qquad |U_i|\le\kappa_\nu V_c,
\tag{KU.S2}
$$

where $(Z_i)_i$ are independent standard Gaussians. For count
normalization $W_X=I-t\nu L_X$; for row normalization
$W_X=(1-t\nu)I+t\nu\omega_X$ when $N>1$. The singleton has $W_X=I$.
:::

:::{prf:proof}
Mandatory revival places every dead row at an eligible living source;
an accepted live row also copies a living source, and an unaccepted
live row already lies in $D$. This proves the source assertion even
when incoming dead positions are unbounded. The collision formula
$v_i^{\rm col}=\bar v+\alpha_{\rm col}O(v_i-\bar v)$ and
$|v_j|\le V$ give its stated norm bound for every component and rotation.
Their shared randomness is frozen rather than replaced by independent
rows.

For count normalization, writing
$a_i=t\nu N^{-1}\sum_{j\ne i}K_{ij}\in[0,t\nu]$, the absolute
row sum of $W_X$ is $|1-a_i|+a_i\le\kappa_\nu$.
For row normalization it is $|1-t\nu|+t\nu=\kappa_\nu$.
This proves the rowwise bound, without a degree lower bound or bounded
jitter assumption. When $t\nu\le1$, both matrices are convex
averaging matrices and $\kappa_\nu=1$.

The two A drifts and the O innovation give exactly
$x_2=X+b(W_Xv^{\rm col}+tF(X))+tq\xi^O$.
B2 changes velocity only. Final position diffusion adds $s\xi^x$,
so the independent row sums $tq\xi_i^O+s\xi_i^x$ have covariance
$\sigma_h^2I_d$. Neither the cap nor terminal classification changes
these positions. This proves (KU.S2), including both actual alignment
normalizations. No independence of final velocities is claimed.
$\square$
:::

(sec-ku-conditioning-landing)=
## 2. Independent row trials after retaining all preparation dependence

:::{prf:lemma} A primitive landing floor and binomial domination
:label: lem-ku-coupled-binomial-survival

For $C\ge0$ and $\sigma>0$ put

$$
\ell_{L,\sigma}(C)=
\Phi((L-C)/\sigma)-\Phi((-L-C)/\sigma)>0.
$$

Fix $J>0$ and compute a finite bound on the actual force,
$\mathcal F_J\ge\sup_{|x|\le R_D+J}|F(x)|$.
It may be obtained by evaluating the configured formula and its
primitive coefficients on this specified ball. Define

$$
p_J=\begin{cases}G_d(J/\sigma_J),&\sigma_J>0,\\1,&\sigma_J=0,\end{cases}
\quad C_J=L_D+J+\eta\mathcal F_J+b\kappa_\nu V_c,
\quad a_J=p_J[\ell_{L_D,\sigma_h}(C_J)]^d>0.
\tag{KU.S3}
$$

For every nonextinct entering state, every $N\ge1$, and
$1\le k\le N$,

$$
\begin{aligned}
P_N(S,M^+<k)&\le B_{N,k}(a_J)
=\sum_{j=0}^{k-1}{N\choose j}a_J^j(1-a_J)^{N-j},\\
\mathfrak h_N(S):=P_N(S,M^+=0)&\le (1-a_J)^N,\\
Q_N1(S)&\ge1-(1-a_J)^N,\qquad
\mathbb E_S(M^+/N)\ge a_J.
\end{aligned}
\tag{KU.S4}
$$

No simultaneous $N$-row bounded-noise event is imposed. One may take
the maximum of any explicitly evaluated collection of valid $a_J$.
If $|F(x)|\le B_F+L_F|x|$, the completely specified replacement
$\mathcal F_J=B_F+L_F(R_D+J)$ is valid. More generally a finite local
force profile suffices for (KU.S3); a global Lipschitz hypothesis is
not needed for this survival estimate.
:::

:::{prf:proof}
Condition on the frozen pre-jitter information of (KU.S1), and then
on every jitter. On the event $|\sigma_J Z_i^J|\le J$, row $i$ has
$|M_{i,k}|\le C_J$ in every coordinate, regardless of all other
jitters, rotations or gates. The interval Gaussian probability is
even in its mean and decreasing in the absolute value of that mean:
differentiate its integral to obtain this assertion. Consequently
the conditional landing probability $p_i=P(x_i^+\in D\mid X,v^{\rm col})$
is at least $[\ell_{L_D,\sigma_h}(C_J)]^d$ on that row's tagged event.

Conditional on all jitters, the landing indicators are independent
Bernoulli variables with parameters $p_i$. They can therefore be
represented, without changing their joint distribution, by
$\mathbf1_{\{T_i\le p_i\}}$ for fresh independent uniform $T_i$.
Each dominates

$$
\mathbf1_{\{|\sigma_JZ_i^J|\le J\}}
\mathbf1_{\{T_i\le[\ell_{L_D,\sigma_h}(C_J)]^d\}}.
$$

These lower indicators are independent across rows conditional on
the frozen information, with common success probability $a_J$.
This remains true for $I_i=0$ because its unused latent jitter is
independent and the required bound also holds without its jitter.
Thus their sum is $\operatorname{Bin}(N,a_J)$ and is bounded above
by $M^+$ in the constructed coupling. Mixing over the entire
pattern and rotation law preserves the binomial lower-tail and
expectation bounds. Taking $k=1$ proves the extinction and survival
bounds. This argument explicitly retains the dependence of $W_X$
on every realized jitter. $\square$
:::

:::{prf:lemma} Exact preparation-averaged hazard
:label: lem-ku-coupled-exact-hazard

For either normalization the actual extinction hazard is

$$
\mathfrak h_N(S)=\mathbb E_S^{\rm prep}
\prod_{i=1}^N[1-P_{D,\sigma_h}(M_i)],\qquad
P_{D,\sigma}(m)=\prod_{k=1}^d\ell_{L_D,\sigma}(|m_k|).
\tag{KU.S5}
$$

The expectation uses the actual measured features, two companion
roles, acceptance/revival pattern, component rotations and clone
jitter. In particular the $M_i$ need not be independent. Put
$q_D=1-[2\Phi(L_D/\sigma_h)-1]^d>0$. Then

$$
q_D^N\le\mathfrak h_N(S)\le(1-a_J)^N.
\tag{KU.S6}
$$
:::

:::{prf:proof}
Condition on preparation in (KU.S2). The final positions are
independent, giving the product in (KU.S5). Integrating the specified
preparation law proves the identity. A translated isotropic Gaussian
lands in the symmetric box with probability at most
$[2\Phi(L_D/\sigma_h)-1]^d$, because each interval probability is
maximized at its midpoint. Each exit factor is consequently at least
$q_D$. The upper bound is (KU.S4). $\square$
:::

(sec-ku-conditioning-quadratic)=
## 3. Sharper explicit floors for the unchanged quadratic reference

:::{prf:lemma} Gaussian shift lower bound
:label: lem-ku-gaussian-shift-landing

For a Gaussian measure $\mu=N(m,\sigma^2I_d)$, a Borel set $A$ with
$\mu(A)=p\in(0,1)$, and $|u|\le R$,

$$
N(m+u,\sigma^2I_d)(A)
\ge\Phi(\Phi^{-1}(p)-R/\sigma).
\tag{KU.S7}
$$
:::

:::{prf:proof}
At $u=0$ the assertion is equality with $R=0$. For $u\ne0$, the
likelihood ratio relative to $\mu$ is
$\exp(u\cdot(z-m)/\sigma^2-|u|^2/(2\sigma^2))$.
Among sets of fixed $\mu$-mass $p$, its integral is minimized by
the sublevel halfspace
$u\cdot(z-m)/(|u|\sigma)\le\Phi^{-1}(p)$.
Indeed subtract the two indicators, multiply by the likelihood
ratio minus its value on the bounding hyperplane, and integrate;
the integrand is nonnegative and the constant term integrates to
zero. On that halfspace the shifted one-dimensional Gaussian has
mean $|u|/\sigma$, so the integral is
$\Phi(\Phi^{-1}(p)-|u|/\sigma)$.
Monotonicity in $|u|$ proves (KU.S7). $\square$
:::

:::{prf:theorem} Quadratic no-copy and copy floors without fitness separation
:label: thm-ku-quadratic-binomial-survival

Use the actual force $F(x)=-\lambda x$, $\lambda\ge0$, and put
$A=|1-\eta\lambda|$, $R=b\kappa_\nu V_c$,
$\tau_1^2=A^2\sigma_J^2+\sigma_h^2$. Define

$$
P_{\rm base}=[\ell_{L_D,\sigma_h}(A L_D)]^d,\qquad
a_{0}=\Phi\bigl(\Phi^{-1}(P_{\rm base})-R/\sigma_h\bigr).
\tag{KU.S8}
$$

If $R<L_D$ and $\sigma_J>0$, define

$$
a_{1}=[\ell_{L_D-R,\tau_1}(A L_D)]^d,\qquad
a_*=\min\{a_{0},a_{1}\}>0.
\tag{KU.S9}
$$

If $\sigma_J=0$, use $a_*=a_{0}$ irrespective of acceptance.
If $R\ge L_D$ and $\sigma_J>0$, use the positive $a_J$ from
(KU.S3) instead. All of (KU.S4)--(KU.S6) hold with this $a_*$.
Taking the maximum with other evaluated valid floors is allowed.
There is no minimum acceptance probability or fitness gap.
:::

:::{prf:proof}
Write $a=1-\eta\lambda$, so that the conditional center is
$aX_i+bU_i$. For an unaccepted row, $X_i=Y_i\in\overline D$.
The unshifted Gaussian landing probability at $aY_i$ is at least
$P_{\rm base}$. The shift $bU_i$ has norm at most $R$, including
its dependence on all other jitters. Apply (KU.S7), and use its
monotonicity in the unshifted probability, to obtain the conditional
row survival bound $a_{0}$.

For an accepted row define the eroded box
$D_R=[-L_D+R,L_D-R]^d$ and the own-jitter function
$g_i(Z_i^J)=P_{D_R,\sigma_h}(a(Y_i+\sigma_J Z_i^J))$.
For every shift $bU_i$ of norm at most $R$, membership of the
unshifted noisy position in $D_R$ implies membership of its shifted
position in $D$. Hence its conditional survival probability is at
least $g_i(Z_i^J)$ pointwise. Integrating its own Gaussian jitter
gives exactly

$$
\mathbb E g_i(Z_i^J)=P_{D_R,\tau_1}(aY_i)
\ge [\ell_{L_D-R,\tau_1}(A L_D)]^d=a_{1}.
$$

Conditional on all jitters, use independent uniform landing
variables as in the proof of (KU.S4). Lower success indicators are
$\mathbf1_{\{T_i\le a_{0}\}}$ for $I_i=0$ and
$\mathbf1_{\{T_i\le g_i(Z_i^J)\}}$ for $I_i=1$.
Conditional on the frozen pre-jitter pattern they are independent
and have success probabilities at least $a_*$. Independent
thinning, or their uniform quantile coupling, makes them dominate
independent Bernoulli variables with parameter $a_*$. The resulting
binomial bound is valid after mixing over all original patterns.
When $\sigma_J=0$, accepted and unaccepted positions have the same
source bound and the first argument applies to both. $\square$
:::

:::{prf:corollary} Evaluated unchanged viscous reference
:label: cor-ku-reference-survival-coefficients

For {prf:ref}`def-cgd-existing-reference`, and its specified row
normalization, $t\nu=0.006$, $\kappa_\nu=1$ and

$$
\begin{gathered}
b=0.02(1+e^{-0.04}),\quad A=1-0.0004(1+e^{-0.04}),\quad R=4b<2,\\
\sigma_h^2=0.0002(1-e^{-0.08})+0.0004,
\qquad\tau_1^2=0.01A^2+\sigma_h^2.
\end{gathered}
\tag{KU.S10}
$$

The defining formulas (KU.S8)--(KU.S10) give, diagnostically,

$$
\begin{aligned}
P_{\rm base}&\simeq0.14944635329282643,\\
a_{0}&\simeq1.2136800208248258\,10^{-18},\\
a_{1}&\simeq2.609549193618306\,10^{-4},\\
a_*&=a_{0},\qquad\log a_*\simeq-41.252874590276704.
\end{aligned}
\tag{KU.S11}
$$

The formulas, rather than these floating-point approximations,
define the analytic certificate. Its worst-state bound is
conservative: at $N=200$ it only guarantees survival probability
$1-(1-a_*)^{200}\simeq2.42736\,10^{-16}$. This is a positive
population-independent row floor and an exponentially improving
large-$N$ swarm estimate, not a practical reference mixing time.
:::

:::{prf:proof}
Substitute the unchanged coefficients of the reference into
(KU.S8)--(KU.S9). The proof uses the actual radial cap only to
bound incoming stored velocities and the exact Haar collision
formula to bound its output; unbounded OU and clone draws are
integrated. Both normalizations share the first-kick row bound,
so these certificate constants coincide although their laws differ.
$\square$
:::

(sec-ku-conditioning-transfer)=
## 4. Exact conditional and stationary transfers with uniform coefficients

:::{prf:theorem} Alive fraction, inverse alive mass and current-time survival
:label: thm-ku-uniform-alive-inverse-moments

Let $a\in(0,1)$ be any valid floor above and put

$$
e_N=(1-a)^N,\quad\ell_N=1-e_N\ge a,
\quad m_*=a/2,\quad c_*=(1-\log2)/2,
\quad\delta_N=e^{-c_*aN}.
$$

For any initial probability on $E_N$, let
$\eta_n=\eta_0Q_N^n/(\eta_0Q_N^n1)$.
At every $n\ge1$, and under every QSD of this same killed kernel,

$$
\begin{aligned}
\Pr(M/N<m_*)&\le\delta_N,\\
\mathbb E (N/M)^r
&\le (2/a)^r+\left(\frac{r}{e c_*a}\right)^r=:C_{{\rm inv},r}
\qquad(r>0),\\
\alpha_N&\ge\ell_N\quad\text{for a QSD}.
\end{aligned}
\tag{KU.S12}
$$

The following exact finite-$N$ alternative is also valid for the
inverse moment:

$$
\mathbb E (N/M)^r
\le1+N^r\sum_{k=1}^{N-1}
[k^{-r}-(k+1)^{-r}]\Pr(\operatorname{Bin}(N,a)\le k).
\tag{KU.S13}
$$

For any $T\ge1$,

$$
\Pr(\tau_\dagger>T)\ge\ell_N^T,
\qquad\|\mathsf P_T-\mathsf P_T(\cdot\mid\tau_\dagger>T)\|_{\rm TV}
\le1-\ell_N^T\le Te_N.
\tag{KU.S14}
$$

Here $\mathsf P_T$ is the original absorbing full-path law. The
inverse moment and alive-floor bounds have no factor $T$ and no
cumulative survival denominator.
:::

:::{prf:proof}
For $B\sim\operatorname{Bin}(N,a)$, exponential Markov inequality
with $u=\log2$ gives

$$
\Pr(B<aN/2)
\le e^{uaN/2}(1-a+ae^{-u})^N
\le\exp[-(1-\log2)aN/2]=\delta_N.
$$

The second inequality uses $\log(1-a/2)\le-a/2$.
For any entering law $\zeta$ on $E_N$, put
$H=\zeta P_N(E_N^c)$. The event $\{M<k\}$ includes extinction,
so its next surviving law satisfies exactly

$$
\frac{\zeta P_N(M<k)-H}{1-H}
\le\frac{B_{N,k}(a)-H}{1-H}\le B_{N,k}(a).
\tag{KU.S15}
$$

The final inequality is equivalent to
$H[1-B_{N,k}(a)]\ge0$. It is this subtraction that removes an
unnecessary inverse survival factor. It applies to every entering
law, hence to every current-time survivor law and every QSD.
Taking the binomial half-mean threshold proves the alive floor.

On $M/N\ge a/2$, $(N/M)^r\le(2/a)^r$.
On its complement $(N/M)^r\le N^r$, since surviving outputs have
$M\ge1$. Thus its expectation is at most
$(2/a)^r+N^re^{-c_*aN}$. The maximum of
$x^re^{-c_*ax}$ over $x>0$ is $(r/(e c_*a))^r$,
giving the uniform bound. For (KU.S13), write the decreasing
function $k\mapsto(N/k)^r$ as its value at $N$ plus its
telescoping increments and use (KU.S15) at each $k+1$.

Integrating $Q_N1\ge\ell_N$ against a QSD gives
$\alpha_N\ge\ell_N$. Before absorption the conditional next
survival probability is at least $\ell_N$ for every actual input;
iteration proves the first bound in (KU.S14). For any probability
$P$ and event $A$ of positive probability,
$\|P-P(\cdot\mid A)\|_{\rm TV}=P(A^c)$: split $P$ over $A,A^c$
and evaluate the distance on $A^c$. Apply it to the full survival
event and use $1-(1-e_N)^T\le Te_N$. $\square$
:::

:::{prf:corollary} Reference inverse alive-mass coefficients
:label: cor-ku-reference-inverse-alive-mass

With the reference floor $a=a_*$ in (KU.S11),

$$
C_{{\rm inv},1}\simeq3.6234906930170614\,10^{18},\qquad
C_{{\rm inv},2}\simeq1.8327648966287514\,10^{37}.
$$

These are evaluations of the exact expression in (KU.S12), not
measured stationary constants. Its large values come from demanding
coverage uniformly over all admissible entering geometries,
including boundary sources and outward collision velocities.
:::

:::{prf:proof}
Substitute (KU.S11) into (KU.S12). $\square$
:::

:::{prf:theorem} Primitive moments and exact killed drift
:label: thm-ku-killed-lyapunov-transfer

For an actual force with $|F(x)|\le B_F+L_F|x|$, define

$$
g_{d,p}=\left[2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)}\right]^{1/p},
\quad
K_p=\left[(1+\eta L_F)(R_D+\sigma_Jg_{d,p})
+\eta B_F+b\kappa_\nu V_c+\sigma_hg_{d,p}\right]^p
\quad(p\ge1).
\tag{KU.S16}
$$

For each row and every nonextinct input,
$P_N|x_i^+|^p\le K_p$ and $|v_i^+|^p\le V^p$.
The current-time survivor laws and every QSD obey

$$
\eta_n|x_i|^p\le K_p/\ell_N\le K_p/a,\qquad
\nu_N|x_i|^p\le K_p/\alpha_N\le K_p/\ell_N.
\tag{KU.S17}
$$

For any nonnegative complete-state statistic $G$, the exact identity
is

$$
Q_NG=P_NG-P_N(G;E_N^c).
\tag{KU.S18}
$$

If a complete-update calculation establishes
$P_NG\le rG+B$, with computed $0\le r<1$ and $B\ge0$, then

$$
\eta_{n+1}G\le\frac{r\eta_nG+B}{\ell_N},\qquad
\nu_NG\le\frac{B}{\alpha_N-r}
\le\frac{B}{1-e_N-r}\quad\text{when }e_N<1-r.
\tag{KU.S19}
$$

The stationary assertion uses $\nu_NG<\infty$; for normalized
position moments this is already proved by (KU.S16)--(KU.S17).
For another statistic it must follow from its computed output
envelope before solving the stationary inequality.
For any chosen $\epsilon\in(0,1-r)$, the explicit threshold
$N\ge\lceil\log\epsilon/\log(1-a)\rceil$ gives a contraction
coefficient $r/(1-\epsilon)<1$ and stationary bound
$B/(1-r-\epsilon)$ independent of $N$.
:::

:::{prf:proof}
Apply the triangle inequality in $L^p$ to (KU.S2) and use
$\|X_i\|_{L^p}\le R_D+\sigma_Jg_{d,p}$,
$|U_i|\le\kappa_\nu V_c$ and the specified force-growth bound.
The remaining Gaussian has $L^p$ norm $\sigma_hg_{d,p}$.
This proves (KU.S16), without a maximum over Gaussian rows.
Removing extinct outputs decreases each nonnegative unnormalized
moment. The next-step normalization is at least $\ell_N$;
applying this fact to every entering law proves (KU.S17).
The QSD eigenmeasure identity gives its stronger denominator
$\alpha_N$. The actual cap proves the velocity estimate.

The partition of output events $E_N,E_N^c$ proves (KU.S18).
Discard its nonnegative extinction contribution, integrate the
computed drift, and divide by the current next survival probability
to prove the first inequality in (KU.S19). At a QSD,
$\alpha_N\nu_NG=\nu_NQ_NG\le r\nu_NG+B$;
solving this inequality when $\alpha_N>r$ gives the stationary
claim. Use $\alpha_N\ge1-e_N$ and solve
$(1-a)^N\le\epsilon$ for the displayed threshold.
Every signed cloning, donor, barycenter and kinetic term must
therefore be inside the proved raw drift before this transfer is
used. The transfer itself assumes no QSD eigenfunction bound.
$\square$
:::

:::{prf:proposition} Exact conditioning correction and bounded-test transfer
:label: prop-ku-exact-conditioning-correction

For any entering law $\zeta$, put $H=\zeta\mathfrak h_N\le e_N$.
The actual survivor law $\zeta^+=\zeta Q_N/(1-H)$ satisfies

$$
\zeta^+g-\zeta P_Ng
=\frac{H\zeta P_Ng-\zeta P_N(g;E_N^c)}{1-H},
\qquad\|\zeta^+-\zeta P_N\|_{\rm TV}=H\le e_N.
\tag{KU.S20}
$$

For an output statistic with range of diameter $D_g$, its expectation
changes by at most $D_g e_N$. A diameter-one population observation
or its law consequently incurs at most $e_N$ in bounded-test
transport when conditioning the complete output once. In particular,
an already proved raw consistency bound $\varepsilon_N$ transfers
to $\varepsilon_N+e_N$ wherever its input hypotheses hold.
For an unbounded $g$ with explicit raw bounds
$\zeta P_N|g|\le K_1$ and $\zeta P_Ng^2\le K_2$,

$$
|\zeta^+g-\zeta P_Ng|
\le\frac{e_NK_1+\sqrt{e_NK_2}}{1-e_N}.
\tag{KU.S21}
$$

The $K_1,K_2$ can be computed from (KU.S16) for normalized row
position moments by Jensen's inequality, retaining their actual
fourth or higher moment budget. No joint concentration estimate
is inferred from a marginal moment estimate.
:::

:::{prf:proof}
Use (KU.S18) and subtract $\zeta P_Ng$ to get the identity.
The raw law is the mixture of its survivor and extinct conditional
laws with weights $1-H,H$ on disjoint marked events; its TV
distance to the first conditional law is exactly $H$.
The range-diameter bound follows by this mixture, and any
diameter-one observation contracts that TV bound.
Cauchy--Schwarz gives
$|\zeta P_N(g;E_N^c)|\le\sqrt{HK_2}$.
Insert it into the identity, use $H\le e_N$ and
$1-H\ge1-e_N$, and obtain (KU.S21). $\square$
:::

(sec-ku-conditioning-eigenfunction)=
## 5. The full-state estimate required for a uniform eigenfunction ratio

:::{prf:theorem} Eigenfunction comparison from an actual survivor block
:label: thm-ku-block-eigenfunction-comparison

Assume the actual fixed-$N$ kernel has a QSD with eigenvalue
$\alpha_N$ and a positive bounded eigenfunction for that same
eigenvalue, $Q_Nh_N=\alpha_Nh_N$, normalized by
$\sup h_N=1$. For an integer $m\ge1$ define the actual whole-block
survivor kernel

$$
H_N^{[m]}(S,\cdot)=Q_N^m(S,\cdot)/Q_N^m1(S),
\quad
\beta_N(m)=\sup_{S,T\in E_N}
\|H_N^{[m]}(S,\cdot)-H_N^{[m]}(T,\cdot)\|_{\rm TV},
$$

and $u_N(m)=1-(1-e_N)^m$. If
$2u_N(m)+\beta_N(m)<1$, then

$$
\frac{\sup h_N}{\inf h_N}
\le\frac{1-u_N(m)-\beta_N(m)}{1-2u_N(m)-\beta_N(m)}.
\tag{KU.S22}
$$

This statement uses full marked arrays and all admissible inputs,
not normalized empirical observations, one tagged row, or a
phase projection. It conditions the original kernel once through
the entire block rather than rejecting fatal steps individually.
:::

:::{prf:proof}
The uniform one-step survival bound gives
$1-u_N(m)\le Q_N^m1(S)\le1$ for every input and
$\alpha_N^m\ge1-u_N(m)$.
Write $d=\sup h_N-\inf h_N$. For any inputs $S,T$,
the eigen-equation and the decomposition
$Q_N^mh_N=q_mH_N^{[m]}h_N$ give

$$
\alpha_N^m|h_N(S)-h_N(T)|
\le\beta_N(m)d+u_N(m),
$$

since $0\le H_N^{[m]}h_N\le1$ and the two $q_m$ values
differ by at most $u_N(m)$. Take the supremum over the two inputs.
Then $(1-u_N(m)-\beta_N(m))d\le u_N(m)$.
Consequently

$$
\inf h_N\ge1-\frac{u_N(m)}{1-u_N(m)-\beta_N(m)}
=\frac{1-2u_N(m)-\beta_N(m)}{1-u_N(m)-\beta_N(m)}>0,
$$

which proves (KU.S22). $\square$
:::

:::{prf:corollary} Exact sufficient route from a globally decaying Keystone coupling
:label: cor-ku-uniform-ratio-comparison-route

If a complete global coupling calculation actually proves
$\beta_N(m)\le C\sqrt N\,r^m$ with computed $C>0$,
$0<r<1$ independent of $N$, take

$$
m_N=\max\left\{1,
\left\lceil\frac{\log(4C\sqrt N)}{-\log r}\right\rceil\right\}.
$$

Then $\beta_N(m_N)\le1/4$ and $u_N(m_N)\le m_N(1-a)^N$.
An entirely explicit sufficient threshold is as follows. Put

$$
A_0=1+\frac{\max\{0,\log(4C)\}}{-\log r},\qquad
B_0=\frac1{2(-\log r)},
$$

$$
N_0=\left\lceil\frac1a
\max\left\{1,\log(16A_0),
2\log(16B_0/\sqrt a)\right\}\right\rceil.
\tag{KU.S23}
$$

For $N\ge N_0$, (KU.S22) gives
$\sup h_N/\inf h_N\le5/4$, and the ratio tends to one as
$N\to\infty$. If primitive finite-$N$ certificates are available
for $N<N_0$, their finite maximum and $5/4$ form an
$N$-independent bound. This corollary does not establish its
global coupling premise.
:::

:::{prf:proof}
The block choice proves $C\sqrt N r^{m_N}\le1/4$.
Also $m_N\le A_0+B_0\log N\le A_0+B_0\sqrt N$.
Set $x=aN$ and use $(1-a)^N\le e^{-aN}$ and
$\sqrt x\le e^{x/2}$ to obtain

$$
m_N(1-a)^N\le A_0e^{-x}+(B_0/\sqrt a)e^{-x/2}.
$$

Each term is at most $1/16$ when (KU.S23) holds, so
$u_N(m_N)\le1/8$. The ratio in (KU.S22) is increasing in
both $u$ and $\beta$ on its positive-denominator range.
Substitute $u\le1/8$, $\beta\le1/4$ to get
$(3/4-1/8)/(3/4-2/8)=5/4$.
The exponentially vanishing $m_N(1-a)^N$ proves the limiting
claim; the finite maximum handles the explicitly enumerated
small populations only when each such bound is proved.
$\square$
:::

:::{prf:proposition} Existing reference certificates do not close the block premise
:label: prop-ku-reference-block-certificate-audit

The finite-population certificate in
{prf:ref}`thm-cgd-primitive-eigenfunction` proves bounds
$m_N\ge\underline m_N>0$ and
$\delta_N\ge\underline\delta_N>0$. Its conditioned estimate
alone gives the sufficient full-state block estimate

$$
\beta_N(m)\le
\min\{1,8\underline m_N^{-2}(1-\underline\delta_N)^m\}.
\tag{KU.S24}
$$

At the unchanged count-normalized reference with $N=200$, the
existing logarithmic evaluator diagnostically returns

$$
\log\underline\delta_{200}\simeq-318127493.1640316,\qquad
\log\underline m_{200}\simeq-306309593.74042994.
$$

Using $1-\delta\le e^{-\delta}$, the conservative sufficient
block length for (KU.S24) to be at most $1/4$ is

$$
m\ge\frac{\log32-2\log\underline m_{200}}
{\underline\delta_{200}},\qquad
\log m\gtrsim318127513.3972857.
\tag{KU.S25}
$$

The hazard certificate at this population has
$e_{200}=(1-a_*)^{200}>1/2$, so already
$u_{200}(m)\ge e_{200}>1/2$ for every $m\ge1$.
Thus the strict comparison
$2u_{200}(m)+\beta_{200}(m)<1$ is not verified by these supplied
certificate bounds even before the mixing estimate is used.
At the sufficient length in (KU.S25), its $u_{200}(m)$ is
arbitrarily close to one. This is a failure of this combination of
bounds. It does not prove that the actual eigenfunction ratio
is nonuniform, that the actual process mixes this slowly, or
that the Keystone route cannot supply a stronger block bound.
:::

:::{prf:proof}
For two point-mass input laws, apply (CGD.6) and the triangle
inequality through the actual QSD to obtain the factor
$8\underline m_N^{-2}$. The exponential bound on the last
factor gives (KU.S25). For a fully explicit coarse check,
$e^{-0.04}\ge1-0.04$ gives $R\ge0.1568$, and
$1-e^{-0.08}\le0.08$ gives $\sigma_h<0.0204$.
Thus $R/\sigma_h>7$. Also
$(2-2A)/\sigma_h\le0.0016/0.02=0.08$, so
$P_{\rm base}\le(1/2+0.08/\sqrt{2\pi})^3<1/2$.
Hence $a_*\le a_0<\Phi(-7)$.
Integrating $z/7\ge1$ against the Gaussian density for
$z\ge7$ proves the Mills bound
$\Phi(-7)\le e^{-49/2}/(7\sqrt{2\pi})<1/400$;
the last inequality follows already from
$e^{49/2}>1+49/2+(49/2)^2/2$ and $\sqrt{2\pi}>2$.
Bernoulli's inequality now yields
$e_{200}\ge1-200a_*>1/2$, proving the failure at every block.
With any block satisfying (KU.S25),
$1-e_{200}=1-(1-a_*)^{200}$ is the certificate
value approximately $2.42736\,10^{-16}$;
$u_{200}(m)=1-(1-e_{200})^m$ computed from this lower
survival certificate is therefore extremely close to one.
The strict sufficient inequality in (KU.S22) fails for these
chosen upper bounds. All numbers quoted here are diagnostic
evaluations of explicit analytic formulas. $\square$
:::

:::{prf:remark} Scope of the completed uniform transfers
:label: rem-ku-conditioning-obligations

The completed results are primitive population-independent row
survival, a binomial alive-count bound, exact preparation-averaged
extinction, inverse alive-fraction moments, current-time survivor
moments, and transfers of an actually proved complete signed drift.
They require no global eigenfunction ratio, joint LSI, stationary
chaos, or global attraction assumption. Both reference force
normalizations are covered without an unbounded-noise cutoff.

A normalized averaged discrepancy that decays under the complete
Keystone balance can be transferred by (KU.S18)--(KU.S21).
It becomes a full-state eigenfunction certificate through
(KU.S22) only after it also supplies a uniform full-array
survivor block comparison, including donor, collision, both
viscous kicks, intermediate uncapped noise and terminal marks.
Positive cloning pressure by itself does not establish that
premise. Phase-local contraction does not give the global
supremum over $E_N$ required by (KU.S22).

The exact hazard (KU.S5) includes every recorded algorithmic
source of randomness. The simpler floor deliberately bounds
some of those quantities uniformly: all feature, diversity,
reward, standardization, donor and gate parameters enter that
preparation integral, while their detailed values drop out of
the floor after the source-domain and collision bounds have
been proved. The cap controls stored input velocity; it does
not truncate Gaussian noise or uncapped second-kick velocity.
The B2 map affects phase-space comparison and velocity laws
although it cannot affect the already determined landing
position. A passive geometry readout is included by measurable
pushforward and supplies no additional contraction.

Equal or near-equal fitness is not excluded in any statement
above: $I_i=0$, complete absence of optional cloning, arbitrary
accepted graphs and mandatory revival are all included. These
survival and moment theorems do not decide whether the
remaining complete signed balance decays in that regime.
:::
