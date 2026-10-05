# Survivor-law limits and uniform spatial inequalities for the native viscous gas

(sec-native-stationary-closure-register)=
## 1. Complete parameters and the unchanged transition

:::{prf:definition} Stationary closure register
:label: def-native-stationary-closure-register

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, restricted to the canonical
real-coordinate dense gas of {prf:ref}`def-cgd-parameter-register`. Every
measurement, global standardizer, positive fitness map, gate, donor width,
self-exclusion rule, mandatory revival, recipient jitter, frozen-source copy,
connected-component Haar collision, velocity cap and terminal classification
has its configured value. In particular weighted revival uses its configured
current-companion law; it is not replaced by independent uniform revival.
Both B stages use the selected normalization
$\mathfrak n\in\{\mathrm{count},\mathrm{row}\}$.

The positive regime in this chapter has a configured quadratic force
$F(x)=-\lambda x$, $\lambda\ge0$, a box $D=[-L_D,L_D]^d$ with $L_D>0$,
$h>0$, $q>0$, $s>0$, a finite cap $V>0$, and all canonical positive donor,
fitness and standardization parameters. For finite-population QSD conclusions
also retain the already evaluated phase-smoothing margin $\kappa>0$ of
{prf:ref}`thm-cgd-finite-n-qsd`. This includes the existing count reference
and its stated row-normalization configuration. No sign condition on a
complete two-swarm quadratic discrepancy is imposed.

For the population-map continuity conclusions, evaluate the actual configured
reward profile
$\omega_R(u)=\sup\{|R(x,v)-R(y,w)|:x,y\in\overline D,
|v|,|w|\le V,\ |x-y|+|v-w|\le u\}$ and require
$\omega_R(u)\to0$ as $u\downarrow0$. This is a directly tested restriction
on the provider, rather than regularity of an unknown law. The actual
reference reward $R=-\lambda|x|^2/2$ has
$\omega_R(u)\le\lambda R_Du$. The raw spatial and QSD functional
inequalities do not consume reward continuity.

Write

$$
\begin{gathered}
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
\eta=bt,\quad a_x=1-\eta\lambda,\quad R_D=\sqrt dL_D,\\
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}\qquad s^2=\sigma_x^2h,\qquad
\tau^2=t^2q^2+s^2,\\
V_c=(1+2|\alpha_{\rm col}|)V,\quad
\kappa_\nu=\max\{1,2t\nu-1\},\quad
\ell_\rho=e^{-1/2}/\rho.
\end{gathered}
\tag{SC.1}
$$

Choose and record proof radii $J,r>0$. They do not truncate any innovation.
The notation $G_d$ and $v_d$ is that of
{prf:ref}`def-slc-parameter-register`. Put

$$
\begin{gathered}
p_J=\begin{cases}G_d(J/\sigma_J),&\sigma_J>0,\\1,&\sigma_J=0,\end{cases}
\quad M_J=|a_x|(R_D+J)+b\kappa_\nu V_c,\quad
H_\tau=(2\pi\tau^2)^{-d/2},\\
g_J(z)=p_JH_\tau\exp[-(|z|+M_J)^2/(2\tau^2)],\qquad
\varepsilon_{J,r}=p_Jv_d(r)H_\tau
                       \exp[-(r+M_J)^2/(2\tau^2)]>0.
\end{gathered}
\tag{SC.2}
$$

The complete physical kernel is denoted $P_N$, including an all-dead
output. Its surviving restriction is $Q_N$, its QSD is $\nu_N$, and
$\nu_NQ_N=\alpha_N\nu_N$. Passive history, geometry, color and calibration
parameters remain in $\mathfrak P$ and do not enter this state kernel. A
monotonically increasing recording clock is not asserted to have an invariant
probability. Python graph viscosity, ordered donor-star collisions,
historical donors, geometry feedback and changed noise branches retain their
separate kernels and are not identified with this dense Haar-component
instance. Fixed-seed finite arithmetic is not a continuous Gaussian law.

The population limit below is a family of these existing real-coordinate
kernels at increasing $N$, with all consumed numerical and landscape data
fixed. Every complete execution record must permit its specified population
and update. A fixed finite memory or walker ceiling supplies no infinite
sequence of executable populations; the analytic limit is not a claim that
such a ceiling has been removed from a numerical execution.
:::

:::{prf:lemma} The actual terminal spatial kernel
:label: lem-native-stationary-closure-spatial-kernel

Freeze the entire pre-jitter source, companion, fitness, gate and component
Haar record. Its source positions $Y_i$ lie in $\overline D$, its collision
velocities satisfy $|v_i^c|\le V_c$, and its jitter indicators are
$I_i\in\{0,1\}$. The independent original clone draws give
$X_i=Y_i+I_i\sigma_JZ_i^J$. Define

$$
U=W_Xv^c,\qquad M_i=a_xX_i+bU_i,
\tag{SC.3}
$$

where $W_X=I-t\nu L_X$ for count normalization, and
$W_X=(1-t\nu)I+t\nu\omega_X$ for row normalization at $N\ge2$.
The singleton has $W_X=I$. The complete terminal positions have the exact law

$$
x_i^+=M_i+\tau Z_i,
\tag{SC.4}
$$

conditionally on the jitter array, with independent standard Gaussian $Z_i$.
This is the actual sum $tq\xi_i^O+s\xi_i^x$; the noises used in velocity
and in the second force are retained. Formula (SC.4) is a spatial marginal
identity and does not assert independence of terminal velocities.

For every entering law, every row and both normalizations, its raw terminal
position density satisfies

$$
g_J(z)\le p_{N,i}^{\rm raw}(z)\le H_\tau.
\tag{SC.5}
$$

The same bounds hold after conditioning on any frozen pre-jitter record.
:::

:::{prf:proof}
The first kick is $v_1=W_Xv^c-t\lambda X$. The two A drifts and the OU
step give $x_2=X+bv_1+tq\xi^O=a_xX+bU+tq\xi^O$. B2 and the velocity
cap leave this position unchanged, and the configured final diffusion adds
$s\xi^x$. Combining these independent spatial innovations proves (SC.4).
The absolute row sum of $W_X$ is at most $\kappa_\nu$, so
$|U_i|\le\kappa_\nu V_c$ for both normalizations. On the tagged event
$|\sigma_JZ_i^J|\le J$, which has probability $p_J$ independently of the
frozen record, $|M_i|\le M_J$ regardless of all other jitters. The conditional
Gaussian density is therefore at least the expression defining $g_J$ on
that event, and is always at most $H_\tau$. Integrating all jitters and then
the unchanged frozen-record law proves (SC.5). Unused jitter can be retained
as an independent latent variable when $I_i=0$. No simultaneous bounded
event over all $N$ rows is required.
:::

(sec-native-stationary-closure-inequalities)=
## 2. Uniform inequalities from the existing Gaussian channel

:::{prf:lemma} Poincare inequality for a family sharing actual mass
:label: lem-native-stationary-closure-mixture-poincare

Let $(p_\omega)$ be probability laws on $\mathbb R^d$. Suppose each obeys

$$
\operatorname{Var}_{p_\omega}(f)
\le C_0\int|\nabla f|^2\,dp_\omega,
$$

and $p_\omega\ge\varepsilon\vartheta$ for the same probability law
$\vartheta$ and $\varepsilon>0$. Then every mixture
$p=\int p_\omega\,\zeta(d\omega)$ obeys

$$
\operatorname{Var}_p(f)
\le C_0(1+\varepsilon^{-1})\int|\nabla f|^2\,dp.
\tag{SC.6}
$$

No concentration, entropy inequality or regularity of the mixing law
$\zeta$ is required.
:::

:::{prf:proof}
Put $m_\omega=p_\omega f$ and $m_\vartheta=\vartheta f$.
The actual common-mass inequality gives

$$
\operatorname{Var}_{p_\omega}(f)
\ge\varepsilon\int(f-m_\omega)^2\,d\vartheta
\ge\varepsilon(m_\vartheta-m_\omega)^2.
$$

Total variance and the fact that variance minimizes squared error about a
constant yield

$$
\operatorname{Var}_p(f)
=\int\operatorname{Var}_{p_\omega}(f)d\zeta
 +\operatorname{Var}_\zeta(m_\omega)
\le(1+\varepsilon^{-1})
                       \int\operatorname{Var}_{p_\omega}(f)d\zeta.
$$

Insert the given component inequality and use Tonelli. Truncation extends
the argument from bounded smooth functions to the closure of their energy
domain.
:::

:::{prf:theorem} Uniform raw spatial Poincare and conditional full spatial entropy
:label: thm-native-stationary-closure-spatial-poincare

Use count normalization, with every other parameter unchanged. Set

$$
B_\nu=2bt\nu V_c\ell_\rho,\qquad
C_0=\sigma_J^2[(|a_x|+B_\nu)^2+B_\nu^2]+\tau^2,
\qquad C_{\rm sp}=C_0(1+\varepsilon_{J,r}^{-1}).
\tag{SC.7}
$$

For every $N$, every nonextinct entering law and every terminal row, its
raw position marginal satisfies

$$
\operatorname{Var}(f(x_i^+))
\le C_{\rm sp}\,\mathbb E|\nabla f(x_i^+)|^2.
\tag{SC.8}
$$

The constant is independent of $N$, the incoming law, fitness activity,
the alive count and the actual component sizes. All parameter dependence
is shown in (SC.1)--(SC.2) and (SC.7).

There is also a population-uniform inequality for the complete spatial
array conditional on the frozen pre-jitter record. With

$$
C_{\rm cond}=\sigma_J^2(|a_x|+2B_\nu)^2+\tau^2,
$$

that conditional law $p_\omega^N$ obeys

$$
\operatorname{Var}_{p_\omega^N}(f)
\le C_{\rm cond}\int|\nabla f|^2dp_\omega^N,\qquad
\operatorname{Ent}_{p_\omega^N}(f^2)
\le2C_{\rm cond}\int|\nabla f|^2dp_\omega^N.
\tag{SC.9}
$$

The entropy statement is conditional on the actual source/gate/Haar record.
It is not a joint stationary LSI, and its gradient acts on physical
positions. A test of the discontinuous terminal mark alone is not an
admissible smooth spatial test in (SC.9).
:::

:::{prf:proof}
Freeze the record used in (SC.3). In count normalization,

$$
U_i=v_i^c+\frac{t\nu}{N}\sum_{j\ne i}
 K_\rho(X_i,X_j)(v_j^c-v_i^c).
$$

Its velocity array is fixed here. Since each spatial kernel derivative
has norm at most $\ell_\rho$ and $|v_j^c-v_i^c|\le2V_c$,

$$
\|\partial_{X_i}M_i\|\le |a_x|+B_\nu,\qquad
\|\partial_{X_j}M_i\|\le B_\nu/N\quad(j\ne i).
$$

The squared operator norm of this block row is at most the sum of squared
block norms, and hence is at most $(|a_x|+B_\nu)^2+B_\nu^2$.
The row position is a function of the independent standard Gaussian
array $(Z^J,Z_i)$ with squared Lipschitz constant at most $C_0$.
Gaussian Poincare, followed by the chain rule, proves the component
Poincare inequality with constant $C_0$. Its hypotheses hold globally;
the Gaussian kernel is smooth and has the displayed bounded derivative.

For completeness, the standard Gaussian inequality follows by the
Ornstein--Uhlenbeck interpolation: for its semigroup $T_u$,
$\operatorname{Var}_\gamma(g)=2\int_0^\infty
\gamma|\nabla T_ug|^2du$, and
$|\nabla T_ug|^2\le e^{-2u}T_u|\nabla g|^2$. Integration gives
$\operatorname{Var}_\gamma(g)\le\gamma|\nabla g|^2$.
The entropy calculation is the same interpolation for positive $g$,
$\operatorname{Ent}_\gamma(g)=\int_0^\infty
\gamma(|\nabla T_ug|^2/T_ug)du$; Cauchy--Schwarz inside $T_u$
and the gradient commutation give
$\operatorname{Ent}_\gamma(f^2)\le2\gamma|\nabla f|^2$.
These calculations first apply to bounded positive smooth functions;
positive regularization and truncation give the stated domains.

By (SC.5), each conditional row law dominates
$\varepsilon_{J,r}$ times uniform measure on $B(0,r)$.
The preceding mixture lemma therefore proves (SC.8) after mixing the
entire original frozen-record law and the entering state law.

For the full array, the kernel derivative bound also gives

$$
|U_i(X)-U_i(\widetilde X)|
\le\frac{2t\nu V_c\ell_\rho}{N}
 \sum_j(|X_i-\widetilde X_i|+|X_j-\widetilde X_j|).
$$

Jensen and Minkowski in normalized row $L^2$ give
$\|U(X)-U(\widetilde X)\|_{2,N}
\le4t\nu V_c\ell_\rho\|X-\widetilde X\|_{2,N}$.
Consequently $M$ is $(|a_x|+2B_\nu)$-Lipschitz for the ordinary
Euclidean array norm as well. The joint Gaussian map (SC.4) has squared
Lipschitz constant $C_{\rm cond}$. Apply both Gaussian inequalities to
this map to obtain (SC.9). Conditioning has retained every shared Haar
rotation, donor and gate; none has been replaced by independent rows.
:::

:::{prf:theorem} Conditional complete marked entropy from the existing innovations
:label: thm-native-stationary-closure-marked-entropy

For either normalization, freeze the actual pre-jitter record $\omega$.
Let $\Phi_\omega$ be the unchanged full remaining update, including jitter,
both forces, drifts, thermostat, final diffusion, cap and terminal marks.
Its input consists of the original independent Gaussian blocks
$\mathcal I=\{(J,i),(O,i),(x,i):1\le i\le N\}$. An unused latent
jitter block when $I_i=0$ has no effect. Write
$S^+=\Phi_\omega(Z)$ and
$S^{+,\ell}=\Phi_\omega(Z^{\ell})$, where $Z^{\ell}$ replaces just
block $\ell$ by an independent draw from that block's unchanged law.

For every bounded positive measurable complete marked-state test $f$,

$$
\operatorname{Ent}(f(S^+)\mid\omega)
\le\sum_{\ell\in\mathcal I}\mathbb E\left[
f(S^+)\log\frac{f(S^+)}{f(S^{+,\ell})}
-f(S^+)+f(S^{+,\ell})\ \middle|\ \omega\right].
\tag{SC.23}
$$

The coefficient is one independently of $N$. The form sums the actual
three innovation stages over recipients; it is not divided by population
size. It sees alive/dead changes and the coupled B2 output, rather than
claiming to measure those changes by continuous gradients alone. This is
a conditional full-state inequality. It retains the mixing law of
pre-jitter records outside its conditioning and makes no claim that that
mixing law satisfies an LSI.
:::

:::{prf:proof}
For a positive function $g$ on a product of the original independent
innovation laws, entropy tensorization gives
$\operatorname{Ent}(g)\le\sum_\ell
\mathbb E\operatorname{Ent}_{Z_\ell}(g\mid Z_{-\ell})$.
It follows inductively from the two-block entropy chain rule and the
convexity of entropy for the conditional average in the other block.
For a single block, with $g'$ an independent same-law copy conditional
on the other blocks, Jensen for the logarithm gives

$$
\operatorname{Ent}(g)
=\mathbb E(g\log g)-(\mathbb Eg)\log\mathbb Eg
\le\mathbb E\left[g\log(g/g')-g+g'\right].
$$

Use $g=f\circ\Phi_\omega$ in these inequalities. All changes to a
second force, cap or terminal status caused by the resampled block are
part of the same map $\Phi_\omega$, so the complete actual marked
output appears on both sides of every comparison. Resampling here is
integration against the existing innovation law in a Dirichlet form;
it does not add a stage or alter the transition. Positive regularization
and monotone limits extend the formula whenever its entropy and displayed
form are finite.
:::

:::{prf:theorem} Actual QSD spatial inequalities and the negligible selection defect
:label: thm-native-stationary-closure-qsd-poincare

Let $a_*>0$ be any explicitly evaluated valid row landing floor of
{prf:ref}`lem-ku-coupled-binomial-survival` or
{prf:ref}`thm-ku-quadratic-binomial-survival`, and put

$$
e_N=(1-a_*)^N,\qquad \ell_N=1-e_N.
$$

Then the existing finite-$N$ QSD satisfies

$$
\alpha_N\ge\ell_N\longrightarrow1.
\tag{SC.10}
$$

For count normalization its row-position law $\nu_{N,i}^x$ obeys,
for every bounded $C^1$ test with bounded gradient,

$$
\operatorname{Var}_{\nu_{N,i}^x}(f)
\le C_{\rm sp}\nu_{N,i}^x|\nabla f|^2
 +C_{\rm sp}\frac{e_N}{1-e_N}\|\nabla f\|_\infty^2.
\tag{SC.11}
$$

If $\nabla f=0$ outside $D$, the second term is exactly zero for every
$N$. Every weak limit of these row-position QSD marginals obeys the
exact Poincare inequality with constant $C_{\rm sp}$.

For both normalizations the law of a specified row's position conditional
on that row being alive has the exact uniform inequality

$$
\operatorname{Var}_{\nu_{N,i}^x(\cdot\mid D)}(f)
\le C_D\int_D|\nabla f|^2d\nu_{N,i}^x(\cdot\mid D),\qquad
C_D=\frac{4L_D^2}{\pi^2p_J}
                   e^{(R_D+M_J)^2/(2\tau^2)}.
\tag{SC.12}
$$

This last statement is a single labelled row conditioned alive, rather
than a uniformly selected alive slot or the full interacting law.
:::

:::{prf:proof}
The binomial landing theorem gives $Q_N1\ge\ell_N$ at every entering
state, and integrating against $\nu_N$ proves (SC.10). Its raw output law
$\rho_N=\nu_NP_N$ decomposes exactly as

$$
\rho_N=\alpha_N\nu_N+(1-\alpha_N)\rho_N^\dagger,
\tag{SC.13}
$$

where $\rho_N^\dagger$ is its actual all-dead conditional output law.
For any center $m$, the QSD marginal squared error is at most the raw
error divided by $\alpha_N$. Taking the raw optimal center, then applying
(SC.8) to the incoming law $\nu_N$, gives

$$
\operatorname{Var}_{\nu_{N,i}^x}(f)
\le C_{\rm sp}\nu_{N,i}^x|\nabla f|^2
 +C_{\rm sp}\frac{1-\alpha_N}{\alpha_N}
                    (\rho_N^\dagger)_i|\nabla f|^2.
$$

Use (SC.10) and the bounded gradient to obtain (SC.11). On extinction
every position is outside $D$, so a gradient zero there removes the
last term exactly. For $f\in C_c^\infty$, weak convergence passes both
its variance and its bounded continuous gradient energy to the limit;
$e_N/(1-e_N)\to0$. Closure gives the limiting Poincare domain.

Inside $D$, survival of the complete swarm is automatic as soon as the
specified row is alive. Thus (SC.13) identifies the conditional alive
row density as the raw row density restricted to $D$ and normalized.
The bounds (SC.5) give density ratio at most
$H_\tau/[p_JH_\tau e^{-(R_D+M_J)^2/(2\tau^2)}]$ on $D$.
Uniform measure on this box has Poincare constant $4L_D^2/\pi^2$:
the one-dimensional cosine expansion gives this constant on
$[-L_D,L_D]$, and successive conditional variances tensorize it over
the coordinates. Comparing the maximal and minimal normalized
densities transfers this box inequality and proves (SC.12).
:::

:::{prf:lemma} A moment alternative for the same QSD eigenvalue
:label: lem-native-stationary-closure-moment-hazard

Fix an interior ball $B(0,r_D)\subset D$ and $p\ge1$. The actual
preterminal means in (SC.3) obey, uniformly over incoming states,

$$
\mathbb E\frac1N\sum_i|M_i|^p\le H_p,\qquad
H_p=\left[|a_x|(R_D+\sigma_Jg_{d,p})+b\kappa_\nu V_c\right]^p,
\quad g_{d,p}=\left(2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)}\right)^{1/p}.
$$

For every proof radius $M>0$, define

$$
p_D(M)=v_d(r_D)H_\tau e^{-(M+r_D)^2/(2\tau^2)}.
$$

The same kernel and its QSD satisfy

$$
1-\alpha_N\le\sup_S P_N(S,E_N^c)
\le\min\left\{e_N,\frac{2H_p}{M^p}+e^{-Np_D(M)/2}\right\}.
\tag{SC.14}
$$

In particular $M_N=(\log N)^{1/4}$, at $N\ge2$, makes the second
certificate tend to zero, with the actual fixed variance $\tau^2$.
:::

:::{prf:proof}
Minkowski and the original jitter distribution give the displayed mean
moment. By Markov, the event that their average $p$th moment exceeds
$M^p/2$ has probability at most $2H_p/M^p$. On its complement at
least $N/2$ means have norm at most $M$. Conditional on the complete
preparation their remaining spatial innovations are independent, and
each such mean lands in the interior ball with probability at least
$p_D(M)$. Their joint failure probability is at most
$(1-p_D(M))^{N/2}\le e^{-Np_D(M)/2}$. Add the exceptional moment
event and integrate against the QSD. The binomial certificate supplies
the first term in the minimum. With the stated $M_N$,
$\log[Np_D(M_N)]=\log N-O(\sqrt{\log N})\to\infty$, whereas
$H_p/M_N^p\to0$.
:::

(sec-native-stationary-closure-survivor-law)=
## 3. Full-law discrepancy with the actual terminal event

:::{prf:theorem} Exact QSD entropy defect of the raw complete law
:label: thm-native-stationary-closure-full-law-defect

For either normalization and the actual complete QSD output,

$$
\|\nu_NP_N-\nu_N\|_{\rm TV}=1-\alpha_N\le e_N,\qquad
D(\nu_N\Vert\nu_NP_N)=-\log\alpha_N\le-\log(1-e_N).
\tag{SC.15}
$$

Start the original absorbing path from $\nu_N$, and condition once on
survival through $T$ actual updates. Denote its path law by
$\mathsf R_{N,T}$ and the original absorbing path law by
$\mathsf P_{N,T}$. Then

$$
\mathsf P_{N,T}(\tau_\dagger>T)=\alpha_N^T,\qquad
D(\mathsf R_{N,T}\Vert\mathsf P_{N,T})=-T\log\alpha_N,
\qquad
\|\mathsf R_{N,T}-\mathsf P_{N,T}\|_{\rm TV}
=1-\alpha_N^T\le Te_N.
\tag{SC.16}
$$

For $0\le j\le T$, its $j$th state marginal differs from $\nu_N$ in
TV by at most $1-\alpha_N^{T-j}\le(T-j)e_N$. The state and all
consumed stages, marks and donor correlations are retained in these
comparisons. They are valid on growing horizons satisfying $T_Ne_N\to0$.
:::

:::{prf:proof}
In (SC.13), the survivor and extinct laws have disjoint marked support.
The TV equality follows immediately, and the Radon--Nikodym derivative
of $\nu_N$ relative to its raw output is $1/\alpha_N$ on survivor
support. This proves the entropy equality and (SC.15).
Iteration of the QSD equation gives survival probability $\alpha_N^T$.
Conditioning any probability on an event of mass $p$ has relative
entropy $-\log p$ and TV $1-p$, by its density $\mathbf1_A/p$.
This proves (SC.16); $1-\alpha_N^T\le T(1-\alpha_N)$ gives the
last bound. Conditional on survival through $j$, the $j$th state law
is exactly $\nu_N$. Conditioning this already selected path further
on survival through $T$ changes it by TV $1-\alpha_N^{T-j}$.
Projection to the $j$th state cannot increase TV. No independent
restart or rowwise normalization is used.
:::

(sec-native-stationary-closure-population-law)=
## 4. Stationary population limits for both native normalizations

:::{prf:theorem} Existence and actual QSD population-law invariance
:label: thm-native-stationary-closure-population-invariance

Let $\mathcal F_h^{\nu,\mathfrak n}=\mathcal K_h^{\nu,\mathfrak n}
\circ\mathcal J$ be the complete rooted-component population map in
{prf:ref}`def-cg-mf-kinetic-map`, using the unchanged sampled fitness,
weighted current revival and shared Haar component rule. For the parameter
regime in {prf:ref}`def-native-stationary-closure-register`, and separately
for each of its count and row normalization tags:

1. This map has at least one stationary population probability $\mu_*$.
   It has capped velocities, positive alive mass at least $a_*$, and
   terminally consistent marks. Its spatial density is smooth and satisfies
   $g_J\le\rho_*\le H_\tau$.
2. Write $\Lambda_N=(L_N)_\#\nu_N$ for the empirical-law distribution
   of the actual full marked QSD. These laws are tight in $W_4$ on
   population probabilities. Every subsequential limit satisfies

   $$
   (\mathcal F_h^{\nu,\mathfrak n})_\#\Lambda=\Lambda.
   \tag{SC.17}
   $$

   Almost surely under $\Lambda$, the law has terminally consistent marks,
   velocity support in $\overline B(0,V)$, alive mass at least $a_*$,
   and a smooth spatial density between $g_J$ and $H_\tau$.
3. Along that same subsequence, the QSD law of every fixed $k$ labelled
   rows converges weakly to

   $$
   \int\mu^{\otimes k}\,\Lambda(d\mu).
   \tag{SC.18}
   $$

   Distinct uniformly selected alive rows instead converge to
   $\int\mathcal R(\mu)^{\otimes k}\Lambda(d\mu)$, with
   $\mathcal R(\mu)=\mu(a\,\cdot)/\mu(a)$.

This conclusion identifies a stationary distribution of population laws.
It does not identify it with a mixture of fixed points, a unique phase,
or a deterministic stationary-chaos limit.
:::

:::{prf:proof}
**Uniform output class.** For $p>4$ put

$$
K_p=\left[|a_x|(R_D+\sigma_Jg_{d,p})
                    +b\kappa_\nu V_c+\tau g_{d,p}\right]^p.
\tag{SC.19}
$$

The spatial identity (SC.4) gives this output moment bound, uniformly
over input states and $N$, regardless of the donor, gate and component
law. The full kinetic map has the same bound by its population version.
Its terminal spatial density is an actual $\tau$-Gaussian mixture over
the preterminal center law; (SC.5) therefore holds for it. Its alive
mass is at least $a_*$: apply the existing binomial floor to deterministic
empirical approximants of any admitted population law and use the
proved one-step population limit. Boundedness of the alive indicator
in the marked row space passes its expectation to the limit. Equivalently,
the lower-trial argument integrates directly over the rooted preparation.

Consider the set of population laws with $p$th position moment at most
$K_p$, capped velocities, alive mass at least $a_*$, position density
bounded above by $H_\tau$, and terminally consistent marks. It is nonempty
because it contains every output of an admitted input law. It is convex
and weakly compact: the moment bound gives tightness and is lower
semicontinuous, density domination is closed when tested against
nonnegative continuous compactly supported functions, and this domination
excludes mass on $\partial D$. The last fact preserves terminal mark
consistency under weak limits. The map preserves this set.

**Continuity without a contraction hypothesis.** Positive alive mass
uniformly bounds the actual eligible donor denominators. Bounded squashed
features, positive regularizers and bounded eligible rewards give continuity
of the measured type law and its global statistics. This verifies the
selection hypotheses of {prf:ref}`thm-mean-field-one-step-consistency`.
At every finite exploration cutoff the unchanged rooted-component integrals
and readouts are continuous. The uniform component truncation bound of
{prf:ref}`lem-chaos-component-truncation` removes that cutoff; it applies
because alive accepted edges strictly increase the actual sampled fitness,
dead vertices have one edge to an eligible live donor, and the positive
Gaussian donor weights have their configured lower bound. Thus the shared
Haar component law is retained in this continuity argument.

To verify its hypotheses quantitatively, the configured squashed feature
distance satisfies
$D_{\rm feat}\le D_0=2\sqrt{(R_x^{\rm feat})^2+
\lambda_{\rm alg}(R_v^{\rm feat})^2}$.
Consequently each donor role has the primitive lower bound
$\kappa_b=\exp[-D_0^2/(2\epsilon_b^2)]>0$, with $\kappa_b=1$
at its configured uniform-width convention. On a class with alive mass
$m\ge m_*>0$, take $C=2/(\kappa_Cm_*)$. For empirical inputs with
$M\ge m_*N$ and $m_*N\ge2$, self exclusion leaves each live recipient
at least $M-1\ge m_*N/2$ eligible candidates, and each dead recipient
has $M$ candidates. Every specified accepted edge therefore has conditional
probability at most $C/N$. Recipient donor draws and gate uniforms are
independent across rows after freezing the entire measured fitness array.
A positive live gate implies the strict inequality $F_j>F_i$, including
all actual sampled ties. Dead vertices cannot be targets and have one
revival edge to a live vertex. Thus outdegree is at most one and the
strict fitness order excludes directed cycles; an undirected cycle in
an outdegree-one graph would be a directed cycle, so the actual graph is
a forest. These are every hypothesis of the cited truncation lemma, giving
$\mathbb E|\mathcal C_N(i)|\le e^{2C}$ and
$\Pr(|\mathcal C_N(i)|>K)\le e^{2C}/K$.
For the fixed-point set use $m_*=a_*$; for stationary compact localization
use $m_*=a_*/2$. All constants are fixed over the respective class and
independent of $N$. This statement removes the cutoff only in the proof;
it does not impose a component cutoff on the algorithm.

The bounded-source preparation has velocities bounded by $V_c$ and
all position moments from its configured Gaussian jitter. Its weak
continuity therefore upgrades to $W_4$ by the $p>4$ moment bound.
The kinetic continuity proof is the deterministic-law version of
{prf:ref}`thm-cg-mf-kinetic-limit` for count, or
{prf:ref}`thm-cg-mf-row-kinetic-limit` for row. In the row case its
local-normalization proof keeps $a_{L_N}-1/N$ and uses actual Gaussian
moments; it does not presume a global positive degree. Final position
noise excludes boundary mass, giving marked continuity. The compact-convex
fixed-point argument of {prf:ref}`thm-mean-field-stationary-existence`
now applies to this continuous self-map and proves the first assertion.
Every fixed point is an output law, so it has the displayed density bounds.
Gaussian convolution gives smoothness: its derivatives are convolutions
with the bounded derivatives of the $\tau$-Gaussian density.

**QSD compactness.** Apply the QSD equation to the nonnegative position
moment and the cap, using (SC.10) and (SC.19), to obtain

$$
\mathbb E_{\Lambda_N}\mu|x|^p\le K_p/\ell_N\le K_p/a_*.
$$

Markov places arbitrarily high probability in a set with bounded $p$th
moment. Such sets are compact in $W_4$: weak compactness follows from
the moment bound, and the fourth-moment tails are at most $K/R^{p-4}$.
This proves tightness. The current-time/QSD binomial bound gives
$\nu_N\{L_Na<a_*/2\}\le e^{-c_*a_*N}$ with the explicit
$c_*=(1-\log2)/2$. Hence every subsequential limit has positive alive
mass at least $a_*/2$ almost surely. The QSD position density bound is
$H_\tau/\alpha_N$, by (SC.13). Testing boundary neighborhoods therefore
shows that every limit has zero mass on $\partial D$ almost surely;
terminal mark consistency passes to the limit.

**Uniform consistency on compact sets.** If deterministic empirical
inputs converge in $W_4$ within a class with uniformly bounded $p$th
moment and positive alive fraction, the actual selection theorem and
bounded-source moment bound yield post-collision convergence in $W_4$.
Apply the appropriate count or row kinetic theorem to get complete
one-step empirical convergence to $\mathcal F_h^{\nu,\mathfrak n}$.
This is uniform on each compact subset of that input class: otherwise
choose inputs with a discrepancy bounded away from zero, extract a
convergent subsequence by compactness, and apply the just established
sequential statement to obtain a contradiction. A bounded metric $d_*$
for weak marked-law convergence thus satisfies

$$
\mathbb E_{\nu_N}d_*\!\left(L_N(S^+),
                    \mathcal F_h^{\nu,\mathfrak n}(L_N(S))\right)
\longrightarrow0,
\tag{SC.20}
$$

with any fixed cemetery observation on extinction. Compact localization,
the QSD moment bound and the exponentially small alive-floor exception
justify integration over the actual random QSD input.

By (SC.15), the empirical law of the raw output differs in TV from
$\Lambda_N$ by at most $e_N$. Apply a bounded Lipschitz population test
to (SC.20), pass to any convergent subsequence, and use the established
continuity at its admitted limits. This proves (SC.17). Since every
population output has alive mass at least $a_*$ and density between
$g_J$ and $H_\tau$, invariance upgrades the almost-sure support to those
properties. The class of Gaussian mixtures is retained: any weak limit
of their center laws is a probability after the uniform center moment
bound, so their position marginals are again $\tau$-Gaussian mixtures.

**Labelled mixtures.** The actual count and row kernels commute with
slot permutations: both nonself Gaussian sums, simultaneous sources,
component rotations, noises, cap and terminal test commute after
relabeling their random inputs. Uniqueness of the QSD therefore makes
$\nu_N$ exchangeable. Conditional on its empirical multiset, fixed
labels are sampled without replacement. Their TV distance from independent
sampling from that multiset is at most $k(k-1)/(2N)$, by the repeated-label
union bound. Passing $\Lambda_N\Rightarrow\Lambda$ gives (SC.18).
For alive sampling, on $M/N\ge a_*/2$ that error is at most
$k(k-1)/(a_*N)$. Its complement has the above exponentially small
probability. The map $\mathcal R$ is continuous for marked weak
convergence with positive alive mass, so the same passage gives the
stated alive mixture. No independence of finite component outputs is
assumed in this argument.
:::

:::{prf:corollary} The stationary empirical trajectory of the existing QSD
:label: cor-native-stationary-closure-trajectory

Along the subsequence of the preceding theorem, start the actual gas from
$\nu_N$ and condition once on survival through a fixed $T$. Then the
joint empirical trajectory converges in distribution to

$$
(\mu_0,\mu_1,\ldots,\mu_T),\qquad
\mu_0\sim\Lambda,\quad
\mu_{j+1}=\mathcal F_h^{\nu,\mathfrak n}(\mu_j).
\tag{SC.21}
$$

This limiting population process is stationary under integer shifts.
For count normalization, the spatial marginal of its one-row stationary
mixture obeys the exact uniform Poincare inequality of (SC.8).
:::

:::{prf:proof}
Apply compact-localized one-step consistency successively at the fixed
number of times, using the moment and alive-floor bounds at every
current-time survivor law. The raw absorbing-path limit is the stated
ordered deterministic map from its random initial population law.
Equation (SC.16) changes that complete path by at most $Te_N$, so
whole-horizon conditioning has the same limit. Invariance (SC.17)
then gives stationarity of every shifted finite population trajectory.
The one-row spatial law is the weak limit of the QSD row marginals;
the exact limiting Poincare statement follows from (SC.11).
:::

(sec-native-stationary-closure-scope)=
## 5. Derived regimes and the remaining central estimate

:::{prf:remark} Positive certificates, actual failures and the remaining phase task
:label: rem-native-stationary-closure-scope

The stationary population-law identification, raw full-law entropy defect,
alive conditional spatial Poincare and density bounds hold for both existing
dense normalizations in the regime of the register. The unconditional raw
spatial Poincare, its QSD defect tending to zero and the conditional full
spatial LSI have the stated count-normalization scope. Their constants are
finite for every $J,r>0$, $\tau>0$ in that regime. At the unchanged
reference, $t\nu=0.006$, $V_c=4$, $\tau^2\simeq0.00041537673$,
$a_x=1-0.0004(1+e^{-0.04})$, and
$B_\nu=2(0.02)(0.02)(1+e^{-0.04})(0.3)(4)e^{-1/2}$.
The small-noise common-mass constants may be extremely large; finiteness
and independence of population size do not make them reference mixing rates.

For the explicit proof choices $J=r=0.1$ at that same reference,
the defining expressions give

$$
\begin{gathered}
p_J\simeq0.1987480430987992,\quad
M_J\simeq3.7181693891471293,\quad
B_\nu\simeq0.0011417077556031582,\\
C_0\simeq0.010422538882577783,\quad
C_{\rm cond}\simeq0.010445381198433752,\\
\log\varepsilon_{J,r}\simeq-17546.59660632163,\quad
\log C_{\rm sp}\simeq17542.03282170405,\quad
\log C_D\simeq62096.35324671150.
\end{gathered}
\tag{SC.22}
$$

These floating evaluations display the scale; formulas (SC.1)--(SC.2),
(SC.7) and (SC.12) define the constants. In particular each required
common-mass and noise parameter is strictly positive at the actual
reference, without relying on a rounded exponential that underflows.
Its donor lower bounds are $\kappa_D=\kappa_C=e^{-4}>0$.

When $\tau=0$, the smoothing and common-density arguments do not apply.
The unchanged zero-noise consensus configurations with zero stored velocities
can have an atomic spatial output; the displayed absolutely continuous
conclusions fail there. Changing the collision/kinetic/provider/feedback tag
requires its own derivative and consistency proof. A failed sufficient
phase-smoothing margin does not by itself prove failure of QSD existence.
A configured reward failing the continuity profile retains the spatial
inequalities when its finite sampled update is defined, but has no
population-map continuity certificate from this argument.

The new law-level argument handles terminal marks and singleton revival
through the actual complete survivor kernel and proved output-law bounds.
It does not repair the false global fixed-slot quadratic margin disproved
in Chapter 18a. The exact remaining stationary-chaos implication is to
identify the support of the invariant population distribution in (SC.17),
prove attraction of a specified native population phase, or prove its
concentration directly. Invariance alone allows cycles and nontrivial
invariant population distributions. The conditional spatial entropy
inequality does not control the mixing distribution of source/gate/Haar
records, nor a pure alive/dead test by continuous gradients. Thus the
full joint marked stationary functional inequality and phase-specific
deterministic chaos remain separate targets after these completed steps.
:::

:::{prf:remark} A derived active count regime now closes stationary phase concentration
:label: rem-native-stationary-closure-phase-advance

{prf:ref}`thm-native-phase-contraction` derives a two-channel
transport contraction for the complete marked count population map.
Its coefficients retain sampled global fitness, mandatory revival,
every component Haar rotation, both viscous kicks, cap/OU correlations
and terminal marks. A target-local B2 density bound supplies the
small expected derivative of the actual smooth cap; no inverse
derivative or stationary-law property is assumed.

{prf:ref}`thm-native-phase-reset-regime` gives an explicit positive
interval of configured caps, and
{prf:ref}`cor-native-phase-positive-witness` evaluates a nonempty
active-cloning, positive-viscosity example.
{prf:ref}`thm-native-phase-stationary-chaos` proves its unique
population stationary phase and full marked QSD chaos.
The unchanged reference does not satisfy this sufficient certificate;
its separate phase and unconditional joint marked inequality remain
the reference obligations.
:::
